//! Process abstraction for Erlang-style concurrency.
//!
//! Each process has:
//! - Its own heap (GC isolation)
//! - Its own call stack (independent execution)
//! - A mailbox (message queue)
//! - Reduction counter (for preemptive scheduling)
//!
//! This design is JIT-compatible: JIT-compiled code operates on
//! the same Process struct, accessing heap/registers directly.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::gc::{GcConfig, GcValue, Heap};
use crate::value::{FunctionValue, Pid, RefId, Reg};

/// A call frame on the stack.
#[derive(Clone)]
pub struct CallFrame {
    /// Function being executed
    pub function: Rc<FunctionValue>,
    /// Instruction pointer
    pub ip: usize,
    /// Register file for this frame (GC-managed values)
    pub registers: Vec<GcValue>,
    /// Captured variables (for closures, GC-managed)
    pub captures: Vec<GcValue>,
    /// Return register in caller's frame
    pub return_reg: Option<Reg>,
}

/// Exception handler info.
#[derive(Clone)]
pub struct ExceptionHandler {
    pub frame_index: usize,
    pub catch_ip: usize,
}

/// Default reductions per time slice.
pub const REDUCTIONS_PER_SLICE: usize = 2000;

/// Process execution state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    /// Ready to run or currently running.
    Running,
    /// Waiting for a message in receive.
    Waiting,
    /// Yielded, ready to be scheduled.
    Suspended,
    /// Process has exited with a value.
    Exited(ExitReason),
}

/// Reason for process exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// Normal exit with return value.
    Normal,
    /// Exited due to error.
    Error(String),
    /// Killed by another process.
    Killed,
    /// Linked process died.
    LinkedExit(Pid, String),
    /// Clean shutdown (e.g., by supervisor).
    Shutdown,
}

/// A lightweight process (like Erlang processes).
///
/// Each process is isolated with its own heap. Communication
/// happens only through message passing (deep copy).
pub struct Process {
    /// Unique process identifier.
    pub pid: Pid,

    /// Process-local garbage-collected heap.
    /// JIT code accesses this directly via Process pointer.
    pub heap: Heap,

    /// Call stack frames.
    /// JIT code pushes/pops frames here.
    pub frames: Vec<CallFrame>,

    /// Pool of reusable register vectors (avoids allocation on function calls).
    pub register_pool: Vec<Vec<GcValue>>,

    /// Message queue (inbox).
    /// Messages are deep-copied into process's heap.
    pub mailbox: VecDeque<GcValue>,

    /// Current state.
    pub state: ProcessState,

    /// Reductions remaining in current slice.
    /// JIT code decrements this at safepoints.
    pub reductions: usize,

    /// Linked processes (bidirectional failure propagation).
    pub links: Vec<Pid>,

    /// Monitors (unidirectional failure notification).
    pub monitors: HashMap<RefId, Pid>,

    /// Processes monitoring this one.
    pub monitored_by: HashMap<RefId, Pid>,

    /// Exception handlers stack.
    pub handlers: Vec<ExceptionHandler>,

    /// Current exception (if any).
    pub current_exception: Option<GcValue>,

    /// Exit value (when state is Exited).
    pub exit_value: Option<GcValue>,

    /// Output buffer (for testing/REPL).
    pub output: Vec<String>,
}

impl Process {
    /// Create a new process with default configuration.
    pub fn new(pid: Pid) -> Self {
        Self::with_gc_config(pid, GcConfig::default())
    }

    /// Create a new process with custom GC configuration.
    pub fn with_gc_config(pid: Pid, gc_config: GcConfig) -> Self {
        Self {
            pid,
            heap: Heap::with_config(gc_config),
            frames: Vec::new(),
            register_pool: Vec::new(),
            mailbox: VecDeque::new(),
            state: ProcessState::Running,
            reductions: REDUCTIONS_PER_SLICE,
            links: Vec::new(),
            monitors: HashMap::new(),
            monitored_by: HashMap::new(),
            handlers: Vec::new(),
            current_exception: None,
            exit_value: None,
            output: Vec::new(),
        }
    }

    /// Get a registers vector from the pool, or allocate a new one.
    /// The vector will be cleared and resized to the requested capacity.
    #[inline]
    pub fn alloc_registers(&mut self, size: usize) -> Vec<GcValue> {
        if let Some(mut regs) = self.register_pool.pop() {
            regs.clear();
            regs.resize(size, GcValue::Unit);
            regs
        } else {
            vec![GcValue::Unit; size]
        }
    }

    /// Return a registers vector to the pool for reuse.
    #[inline]
    pub fn free_registers(&mut self, mut regs: Vec<GcValue>) {
        // Only keep vectors up to a reasonable size in the pool
        if regs.capacity() <= 64 && self.register_pool.len() < 16 {
            regs.clear();
            self.register_pool.push(regs);
        }
        // Otherwise just drop it
    }

    /// Reset reduction counter for a new time slice.
    pub fn reset_reductions(&mut self) {
        self.reductions = REDUCTIONS_PER_SLICE;
    }

    /// Consume reductions. Returns true if should yield.
    /// Called at safepoints (function calls, backward jumps).
    /// JIT code calls this same method.
    #[inline]
    pub fn consume_reductions(&mut self, count: usize) -> bool {
        if self.reductions <= count {
            self.reductions = 0;
            true // Should yield
        } else {
            self.reductions -= count;
            false
        }
    }

    /// Check if process should yield (out of reductions).
    #[inline]
    pub fn should_yield(&self) -> bool {
        self.reductions == 0
    }

    /// Deliver a message to this process's mailbox.
    /// The message is deep-copied from the sender's heap.
    pub fn deliver_message(&mut self, message: GcValue, source_heap: &Heap) {
        let copied = self.heap.deep_copy(&message, source_heap);
        self.mailbox.push_back(copied);

        // Wake up if waiting for messages
        if self.state == ProcessState::Waiting {
            self.state = ProcessState::Running;
        }
    }

    /// Try to receive a message (simple FIFO for now).
    /// Returns None if mailbox is empty.
    pub fn try_receive(&mut self) -> Option<GcValue> {
        self.mailbox.pop_front()
    }

    /// Check if mailbox has messages.
    pub fn has_messages(&self) -> bool {
        !self.mailbox.is_empty()
    }

    /// Set process to waiting state (blocked in receive).
    pub fn wait_for_message(&mut self) {
        if self.mailbox.is_empty() {
            self.state = ProcessState::Waiting;
        }
    }

    /// Suspend process (yielded, ready to run again).
    pub fn suspend(&mut self) {
        self.state = ProcessState::Suspended;
    }

    /// Exit the process.
    pub fn exit(&mut self, reason: ExitReason, value: Option<GcValue>) {
        self.state = ProcessState::Exited(reason);
        self.exit_value = value;
    }

    /// Check if process has finished.
    pub fn is_exited(&self) -> bool {
        matches!(self.state, ProcessState::Exited(_))
    }

    /// Check if process is runnable.
    pub fn is_runnable(&self) -> bool {
        matches!(self.state, ProcessState::Running | ProcessState::Suspended)
    }

    /// Add a link to another process.
    pub fn link(&mut self, other: Pid) {
        if !self.links.contains(&other) {
            self.links.push(other);
        }
    }

    /// Remove a link.
    pub fn unlink(&mut self, other: Pid) {
        self.links.retain(|&p| p != other);
    }

    /// Add a monitor for another process.
    pub fn add_monitor(&mut self, ref_id: RefId, target: Pid) {
        self.monitors.insert(ref_id, target);
    }

    /// Record that another process is monitoring us.
    pub fn add_monitored_by(&mut self, ref_id: RefId, watcher: Pid) {
        self.monitored_by.insert(ref_id, watcher);
    }
}

impl std::fmt::Debug for Process {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Process")
            .field("pid", &self.pid)
            .field("state", &self.state)
            .field("reductions", &self.reductions)
            .field("mailbox_len", &self.mailbox.len())
            .field("frames", &self.frames.len())
            .field("links", &self.links)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_creation() {
        let proc = Process::new(Pid(1));
        assert_eq!(proc.pid, Pid(1));
        assert_eq!(proc.state, ProcessState::Running);
        assert_eq!(proc.reductions, REDUCTIONS_PER_SLICE);
        assert!(proc.mailbox.is_empty());
    }

    #[test]
    fn test_reduction_counting() {
        let mut proc = Process::new(Pid(1));

        // Should not yield with plenty of reductions
        assert!(!proc.consume_reductions(100));
        assert_eq!(proc.reductions, REDUCTIONS_PER_SLICE - 100);

        // Consume most reductions
        proc.reductions = 10;
        assert!(!proc.consume_reductions(5));
        assert_eq!(proc.reductions, 5);

        // Should yield when out of reductions
        assert!(proc.consume_reductions(10));
        assert_eq!(proc.reductions, 0);
        assert!(proc.should_yield());
    }

    #[test]
    fn test_message_delivery() {
        let sender = Process::new(Pid(1));
        let mut receiver = Process::new(Pid(2));

        // Allocate a message on sender's heap
        let msg = GcValue::Int(42);

        // Deliver to receiver (deep copy)
        receiver.deliver_message(msg, &sender.heap);

        assert!(receiver.has_messages());
        let received = receiver.try_receive().unwrap();
        assert_eq!(received, GcValue::Int(42));
        assert!(!receiver.has_messages());
    }

    #[test]
    fn test_waiting_state() {
        let mut proc = Process::new(Pid(1));

        // Empty mailbox -> waiting
        proc.wait_for_message();
        assert_eq!(proc.state, ProcessState::Waiting);

        // Message delivery wakes up
        proc.deliver_message(GcValue::Int(1), &Heap::new());
        assert_eq!(proc.state, ProcessState::Running);
    }

    #[test]
    fn test_links() {
        let mut proc = Process::new(Pid(1));

        proc.link(Pid(2));
        proc.link(Pid(3));
        proc.link(Pid(2)); // Duplicate, should not add

        assert_eq!(proc.links, vec![Pid(2), Pid(3)]);

        proc.unlink(Pid(2));
        assert_eq!(proc.links, vec![Pid(3)]);
    }
}
