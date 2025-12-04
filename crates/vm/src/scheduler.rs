//! Scheduler for lightweight processes.
//!
//! The scheduler manages:
//! - Process registry (Pid -> Process mapping)
//! - Run queue (ready-to-run processes)
//! - Message routing between processes
//! - Link/monitor notification on exit
//!
//! JIT compatibility: The scheduler works with any execution engine.
//! It just manages which Process runs, not how it runs.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::gc::{GcConfig, GcNativeFn, GcValue, Heap};
use crate::process::{ExitReason, Process, ProcessState};
use crate::value::{FunctionValue, Pid, RefId, RuntimeError, TypeValue, Value};

/// The process scheduler.
///
/// Manages all processes and coordinates their execution.
/// JIT-compatible: scheduler doesn't care if process runs
/// interpreted or JIT-compiled code.
pub struct Scheduler {
    /// All processes by Pid.
    processes: HashMap<Pid, Process>,

    /// Run queue (Pids of ready processes).
    run_queue: VecDeque<Pid>,

    /// Waiting processes (in receive, not in run queue).
    waiting: Vec<Pid>,

    /// Currently running process Pid.
    current: Option<Pid>,

    /// Next Pid to allocate (internal counter).
    next_pid: u64,

    /// Next RefId for monitors (internal counter).
    next_ref: u64,

    /// Global functions (shared across processes).
    pub functions: HashMap<String, Rc<FunctionValue>>,

    /// Native functions (shared across processes).
    pub natives: HashMap<String, Rc<GcNativeFn>>,

    /// Type definitions (shared).
    pub types: HashMap<String, Rc<TypeValue>>,

    /// Global variables (shared, read-only after init).
    pub globals: HashMap<String, Value>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            run_queue: VecDeque::new(),
            waiting: Vec::new(),
            current: None,
            next_pid: 1, // Pid 0 is reserved for "init" process
            next_ref: 1,
            functions: HashMap::new(),
            natives: HashMap::new(),
            types: HashMap::new(),
            globals: HashMap::new(),
        }
    }

    /// Spawn a new process.
    /// Returns the new process's Pid.
    pub fn spawn(&mut self) -> Pid {
        self.spawn_with_config(GcConfig::default())
    }

    /// Spawn a new process with custom GC config.
    pub fn spawn_with_config(&mut self, gc_config: GcConfig) -> Pid {
        let pid = Pid(self.next_pid);
        self.next_pid += 1;

        let process = Process::with_gc_config(pid, gc_config);
        self.processes.insert(pid, process);
        self.run_queue.push_back(pid);

        pid
    }

    /// Spawn a linked process.
    /// If either process dies with an error, the other is killed too.
    pub fn spawn_link(&mut self, parent_pid: Pid) -> Pid {
        let child_pid = self.spawn();

        // Create bidirectional link
        if let Some(parent) = self.processes.get_mut(&parent_pid) {
            parent.link(child_pid);
        }
        if let Some(child) = self.processes.get_mut(&child_pid) {
            child.link(parent_pid);
        }

        child_pid
    }

    /// Spawn a monitored process.
    /// Returns (child_pid, monitor_ref).
    pub fn spawn_monitor(&mut self, parent_pid: Pid) -> (Pid, RefId) {
        let child_pid = self.spawn();
        let ref_id = RefId(self.next_ref);
        self.next_ref += 1;

        // Set up monitor (parent watches child)
        if let Some(parent) = self.processes.get_mut(&parent_pid) {
            parent.add_monitor(ref_id, child_pid);
        }
        if let Some(child) = self.processes.get_mut(&child_pid) {
            child.add_monitored_by(ref_id, parent_pid);
        }

        (child_pid, ref_id)
    }

    /// Allocate a new unique reference.
    pub fn make_ref(&mut self) -> RefId {
        let r = RefId(self.next_ref);
        self.next_ref += 1;
        r
    }

    /// Get a process by Pid (immutable).
    pub fn get_process(&self, pid: Pid) -> Option<&Process> {
        self.processes.get(&pid)
    }

    /// Get a process by Pid (mutable).
    pub fn get_process_mut(&mut self, pid: Pid) -> Option<&mut Process> {
        self.processes.get_mut(&pid)
    }

    /// Get the currently running process.
    pub fn current_process(&self) -> Option<&Process> {
        self.current.and_then(|pid| self.processes.get(&pid))
    }

    /// Get the currently running process (mutable).
    pub fn current_process_mut(&mut self) -> Option<&mut Process> {
        self.current.and_then(|pid| self.processes.get_mut(&pid))
    }

    /// Get current process Pid.
    pub fn current_pid(&self) -> Option<Pid> {
        self.current
    }

    /// Send a message from one process to another.
    /// The message is deep-copied to the target's heap.
    pub fn send(&mut self, from_pid: Pid, to_pid: Pid, message: GcValue) -> Result<(), RuntimeError> {
        // Get source heap for deep copy
        let source_heap = if let Some(from) = self.processes.get(&from_pid) {
            // Clone heap reference for borrow checker
            // In practice, we need to be careful here
            &from.heap as *const Heap
        } else {
            return Err(RuntimeError::Panic(format!("Sender process {:?} not found", from_pid)));
        };

        // Deliver to target
        if let Some(to) = self.processes.get_mut(&to_pid) {
            // Safety: source_heap is valid, we're not modifying it
            unsafe {
                to.deliver_message(message, &*source_heap);
            }

            // If target was waiting, add back to run queue
            if to.state == ProcessState::Running && !self.run_queue.contains(&to_pid) {
                // Was waiting, now has message - add to run queue
                self.waiting.retain(|&p| p != to_pid);
                self.run_queue.push_back(to_pid);
            }

            Ok(())
        } else {
            // Target doesn't exist - message is silently dropped (Erlang behavior)
            Ok(())
        }
    }

    /// Pick next process to run (round-robin).
    pub fn schedule_next(&mut self) -> Option<Pid> {
        // Put current back in queue if still runnable
        if let Some(current_pid) = self.current {
            if let Some(proc) = self.processes.get(&current_pid) {
                match proc.state {
                    ProcessState::Running | ProcessState::Suspended => {
                        self.run_queue.push_back(current_pid);
                    }
                    ProcessState::Waiting => {
                        if !self.waiting.contains(&current_pid) {
                            self.waiting.push(current_pid);
                        }
                    }
                    ProcessState::Exited(_) => {
                        // Don't re-queue, will be cleaned up
                    }
                }
            }
        }

        // Get next from run queue
        self.current = self.run_queue.pop_front();

        // Reset reductions for new process
        if let Some(pid) = self.current {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.reset_reductions();
                proc.state = ProcessState::Running;
            }
        }

        self.current
    }

    /// Mark current process as yielded and schedule next.
    pub fn yield_current(&mut self) -> Option<Pid> {
        if let Some(pid) = self.current {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.suspend();
            }
        }
        self.schedule_next()
    }

    /// Process exited - handle links and monitors.
    pub fn process_exit(&mut self, pid: Pid, reason: ExitReason, value: Option<GcValue>) {
        let (links, monitored_by) = if let Some(proc) = self.processes.get_mut(&pid) {
            proc.exit(reason.clone(), value.clone());
            (proc.links.clone(), proc.monitored_by.clone())
        } else {
            return;
        };

        // Propagate to linked processes
        let is_error = !matches!(reason, ExitReason::Normal);
        if is_error {
            for linked_pid in links {
                if let Some(linked) = self.processes.get_mut(&linked_pid) {
                    linked.unlink(pid);
                    if linked.state != ProcessState::Exited(ExitReason::Normal) {
                        // Kill linked process
                        linked.exit(
                            ExitReason::LinkedExit(pid, format!("{:?}", reason)),
                            None,
                        );
                    }
                }
            }
        }

        // Send DOWN messages to monitors
        for (ref_id, watcher_pid) in monitored_by {
            if let Some(watcher) = self.processes.get_mut(&watcher_pid) {
                // Create DOWN message: {:DOWN, ref, :process, pid, reason}
                // For now, just send a simple tuple
                let down_msg = GcValue::Tuple(watcher.heap.alloc_tuple(vec![
                    GcValue::Int(ref_id.0 as i64), // ref
                    GcValue::Pid(pid.0),           // pid of exited process
                ]));
                watcher.mailbox.push_back(down_msg);

                // Wake if waiting
                if watcher.state == ProcessState::Waiting {
                    watcher.state = ProcessState::Running;
                    if !self.run_queue.contains(&watcher_pid) {
                        self.waiting.retain(|&p| p != watcher_pid);
                        self.run_queue.push_back(watcher_pid);
                    }
                }

                // Remove monitor
                watcher.monitors.remove(&ref_id);
            }
        }

        // Remove from queues
        self.run_queue.retain(|&p| p != pid);
        self.waiting.retain(|&p| p != pid);
        if self.current == Some(pid) {
            self.current = None;
        }
    }

    /// Check if any processes are still alive.
    pub fn has_processes(&self) -> bool {
        self.processes.values().any(|p| !p.is_exited())
    }

    /// Get count of active (non-exited) processes.
    pub fn process_count(&self) -> usize {
        self.processes.values().filter(|p| !p.is_exited()).count()
    }

    /// Get count of runnable processes.
    pub fn runnable_count(&self) -> usize {
        self.run_queue.len() + if self.current.is_some() { 1 } else { 0 }
    }

    /// Clean up exited processes.
    pub fn cleanup_exited(&mut self) {
        self.processes.retain(|_, p| !p.is_exited());
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn() {
        let mut sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        assert_ne!(pid1, pid2);
        assert_eq!(sched.process_count(), 2);
        assert_eq!(sched.runnable_count(), 2);
    }

    #[test]
    fn test_scheduling() {
        let mut sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        // First schedule should get pid1
        let next = sched.schedule_next();
        assert_eq!(next, Some(pid1));
        assert_eq!(sched.current_pid(), Some(pid1));

        // Next schedule should get pid2
        let next = sched.schedule_next();
        assert_eq!(next, Some(pid2));

        // Then back to pid1 (round-robin)
        let next = sched.schedule_next();
        assert_eq!(next, Some(pid1));
    }

    #[test]
    fn test_message_passing() {
        let mut sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        // Send message from pid1 to pid2
        let msg = GcValue::Int(42);
        sched.send(pid1, pid2, msg).unwrap();

        // Check pid2 received it
        let proc2 = sched.get_process_mut(pid2).unwrap();
        assert!(proc2.has_messages());
        let received = proc2.try_receive().unwrap();
        assert_eq!(received, GcValue::Int(42));
    }

    #[test]
    fn test_spawn_link() {
        let mut sched = Scheduler::new();

        let parent = sched.spawn();
        let child = sched.spawn_link(parent);

        // Both should be linked
        let parent_proc = sched.get_process(parent).unwrap();
        let child_proc = sched.get_process(child).unwrap();

        assert!(parent_proc.links.contains(&child));
        assert!(child_proc.links.contains(&parent));
    }

    #[test]
    fn test_process_exit_kills_linked() {
        let mut sched = Scheduler::new();

        let parent = sched.spawn();
        let child = sched.spawn_link(parent);

        // Kill child with error
        sched.process_exit(child, ExitReason::Error("crash".to_string()), None);

        // Parent should also be dead
        let parent_proc = sched.get_process(parent).unwrap();
        assert!(parent_proc.is_exited());
    }

    #[test]
    fn test_spawn_monitor() {
        let mut sched = Scheduler::new();

        let watcher = sched.spawn();
        let (child, ref_id) = sched.spawn_monitor(watcher);

        // Kill child
        sched.process_exit(child, ExitReason::Normal, None);

        // Watcher should have DOWN message
        let watcher_proc = sched.get_process(watcher).unwrap();
        assert!(watcher_proc.has_messages());
    }

    #[test]
    fn test_waiting_wakeup() {
        let mut sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        // Put pid2 in waiting state
        sched.get_process_mut(pid2).unwrap().wait_for_message();
        assert_eq!(sched.get_process(pid2).unwrap().state, ProcessState::Waiting);

        // Send message should wake it up
        sched.send(pid1, pid2, GcValue::Int(1)).unwrap();
        assert_eq!(sched.get_process(pid2).unwrap().state, ProcessState::Running);
    }
}
