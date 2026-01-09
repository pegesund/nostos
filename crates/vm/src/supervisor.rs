//! Erlang-style process supervision.
//!
//! Supervisors manage child processes and restart them according to
//! a configurable strategy when they fail.
//!
//! ## Restart Strategies
//!
//! - `OneForOne`: If a child terminates, only that child is restarted
//! - `OneForAll`: If a child terminates, all children are terminated and restarted
//! - `RestForOne`: If a child terminates, it and all children started after it are restarted
//!
//! ## Child Restart Types
//!
//! - `Permanent`: Always restart (default for long-running services)
//! - `Transient`: Restart only if terminated abnormally
//! - `Temporary`: Never restart (for one-off tasks)

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::gc::GcValue;
use crate::process::ExitReason;
use crate::scheduler::Scheduler;
use crate::value::{FunctionValue, Pid, RefId};

/// Strategy for restarting children when one fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Restart only the failed child
    OneForOne,
    /// Restart all children if one fails
    OneForAll,
    /// Restart the failed child and all children started after it
    RestForOne,
}

impl Default for RestartStrategy {
    fn default() -> Self {
        RestartStrategy::OneForOne
    }
}

/// When to restart a child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartType {
    /// Always restart, regardless of exit reason
    Permanent,
    /// Restart only if exit was abnormal (not Normal)
    Transient,
    /// Never restart
    Temporary,
}

impl Default for RestartType {
    fn default() -> Self {
        RestartType::Permanent
    }
}

/// Specification for a supervised child process.
#[derive(Clone)]
pub struct ChildSpec {
    /// Unique identifier for this child
    pub id: String,
    /// Function to start the child
    pub start_func: Arc<FunctionValue>,
    /// Arguments to pass to start function
    pub start_args: Vec<GcValue>,
    /// When to restart
    pub restart: RestartType,
    /// Maximum shutdown time in milliseconds (0 = immediate kill)
    pub shutdown_ms: u64,
}

impl ChildSpec {
    /// Create a new child specification.
    pub fn new(id: impl Into<String>, start_func: Arc<FunctionValue>) -> Self {
        Self {
            id: id.into(),
            start_func,
            start_args: Vec::new(),
            restart: RestartType::Permanent,
            shutdown_ms: 5000,
        }
    }

    /// Set arguments for start function.
    pub fn with_args(mut self, args: Vec<GcValue>) -> Self {
        self.start_args = args;
        self
    }

    /// Set restart type.
    pub fn with_restart(mut self, restart: RestartType) -> Self {
        self.restart = restart;
        self
    }

    /// Set shutdown timeout.
    pub fn with_shutdown(mut self, ms: u64) -> Self {
        self.shutdown_ms = ms;
        self
    }
}

/// Configuration for a supervisor.
#[derive(Clone)]
pub struct SupervisorConfig {
    /// Restart strategy
    pub strategy: RestartStrategy,
    /// Maximum restarts allowed in the time period
    pub max_restarts: u32,
    /// Time period for restart counting (in seconds)
    pub max_seconds: u32,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            strategy: RestartStrategy::OneForOne,
            max_restarts: 3,
            max_seconds: 5,
        }
    }
}

impl SupervisorConfig {
    /// Create with one_for_one strategy.
    pub fn one_for_one() -> Self {
        Self {
            strategy: RestartStrategy::OneForOne,
            ..Default::default()
        }
    }

    /// Create with one_for_all strategy.
    pub fn one_for_all() -> Self {
        Self {
            strategy: RestartStrategy::OneForAll,
            ..Default::default()
        }
    }

    /// Create with rest_for_one strategy.
    pub fn rest_for_one() -> Self {
        Self {
            strategy: RestartStrategy::RestForOne,
            ..Default::default()
        }
    }

    /// Set restart intensity.
    pub fn with_intensity(mut self, max_restarts: u32, max_seconds: u32) -> Self {
        self.max_restarts = max_restarts;
        self.max_seconds = max_seconds;
        self
    }
}

/// Information about a running child.
#[derive(Clone)]
struct ChildInfo {
    spec: ChildSpec,
    pid: Pid,
    monitor_ref: RefId,
}

/// Record of a restart event.
#[derive(Clone)]
struct RestartRecord {
    timestamp: Instant,
}

/// A supervisor that manages child processes.
pub struct Supervisor {
    /// Supervisor configuration
    config: SupervisorConfig,
    /// Child specifications (in start order)
    specs: Vec<ChildSpec>,
    /// Running children
    children: Vec<ChildInfo>,
    /// Restart history for intensity tracking
    restart_history: VecDeque<RestartRecord>,
    /// The scheduler
    scheduler: Arc<Scheduler>,
    /// Supervisor's own pid (if running as a process)
    self_pid: Option<Pid>,
}

impl Supervisor {
    /// Create a new supervisor.
    pub fn new(scheduler: Arc<Scheduler>, config: SupervisorConfig) -> Self {
        Self {
            config,
            specs: Vec::new(),
            children: Vec::new(),
            restart_history: VecDeque::new(),
            scheduler,
            self_pid: None,
        }
    }

    /// Add a child specification.
    pub fn add_child(&mut self, spec: ChildSpec) {
        self.specs.push(spec);
    }

    /// Start all children.
    pub fn start_children(&mut self) -> Result<(), SupervisorError> {
        for spec in self.specs.clone() {
            self.start_child_internal(&spec)?;
        }
        Ok(())
    }

    /// Start a single child from its spec.
    fn start_child_internal(&mut self, spec: &ChildSpec) -> Result<Pid, SupervisorError> {
        // Spawn the child process with monitoring
        let parent_pid = self.self_pid.unwrap_or(Pid(0));
        let (child_pid, monitor_ref) = self.scheduler.spawn_monitor(parent_pid);

        // Set up the child's initial frame
        self.scheduler.with_process_mut(child_pid, |proc| {
            let mut registers = vec![GcValue::Unit; 256];
            for (i, arg) in spec.start_args.iter().enumerate() {
                if i < 256 {
                    registers[i] = arg.clone();
                }
            }

            let frame = crate::process::CallFrame {
                function: spec.start_func.clone(),
                ip: 0,
                registers,
                captures: Vec::new(),
                return_reg: None,
            };
            proc.frames.push(frame);
        });

        let info = ChildInfo {
            spec: spec.clone(),
            pid: child_pid,
            monitor_ref,
        };
        self.children.push(info);

        Ok(child_pid)
    }

    /// Handle a child exit.
    /// Returns Ok(true) if restart succeeded, Ok(false) if no restart needed,
    /// Err if supervisor should terminate.
    pub fn handle_child_exit(
        &mut self,
        pid: Pid,
        reason: &ExitReason,
    ) -> Result<bool, SupervisorError> {
        // Find the child
        let child_idx = self
            .children
            .iter()
            .position(|c| c.pid == pid)
            .ok_or(SupervisorError::UnknownChild(pid))?;

        let child = &self.children[child_idx];
        let spec = child.spec.clone();

        // Check if we should restart
        let should_restart = match spec.restart {
            RestartType::Permanent => true,
            RestartType::Transient => !matches!(reason, ExitReason::Normal),
            RestartType::Temporary => false,
        };

        if !should_restart {
            // Remove from children list
            self.children.remove(child_idx);
            return Ok(false);
        }

        // Check restart intensity
        self.check_restart_intensity()?;

        // Record this restart
        self.restart_history.push_back(RestartRecord {
            timestamp: Instant::now(),
        });

        // Apply restart strategy
        match self.config.strategy {
            RestartStrategy::OneForOne => {
                // Just restart the failed child
                self.children.remove(child_idx);
                self.start_child_internal(&spec)?;
            }
            RestartStrategy::OneForAll => {
                // Stop all children, then restart all
                self.stop_all_children();
                self.children.clear();
                for spec in self.specs.clone() {
                    self.start_child_internal(&spec)?;
                }
            }
            RestartStrategy::RestForOne => {
                // Stop children started after this one, then restart them
                let specs_to_restart: Vec<ChildSpec> = self
                    .children
                    .drain(child_idx..)
                    .map(|c| c.spec)
                    .collect();

                for spec in specs_to_restart {
                    self.start_child_internal(&spec)?;
                }
            }
        }

        Ok(true)
    }

    /// Check if we've exceeded restart intensity.
    fn check_restart_intensity(&mut self) -> Result<(), SupervisorError> {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(self.config.max_seconds as u64);

        // Remove old restart records
        while let Some(front) = self.restart_history.front() {
            if front.timestamp < cutoff {
                self.restart_history.pop_front();
            } else {
                break;
            }
        }

        // Check if we've exceeded the limit
        if self.restart_history.len() >= self.config.max_restarts as usize {
            return Err(SupervisorError::MaxRestartsExceeded {
                restarts: self.restart_history.len() as u32,
                seconds: self.config.max_seconds,
            });
        }

        Ok(())
    }

    /// Stop all children.
    fn stop_all_children(&mut self) {
        for child in &self.children {
            self.scheduler
                .process_exit(child.pid, ExitReason::Shutdown, None);
        }
    }

    /// Get list of child pids.
    pub fn child_pids(&self) -> Vec<Pid> {
        self.children.iter().map(|c| c.pid).collect()
    }

    /// Get number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Check if a pid is a supervised child.
    pub fn is_child(&self, pid: Pid) -> bool {
        self.children.iter().any(|c| c.pid == pid)
    }

    /// Get child spec by id.
    pub fn get_child_spec(&self, id: &str) -> Option<&ChildSpec> {
        self.specs.iter().find(|s| s.id == id)
    }

    /// Set the supervisor's own pid.
    pub fn set_self_pid(&mut self, pid: Pid) {
        self.self_pid = Some(pid);
    }
}

/// Errors that can occur during supervision.
#[derive(Debug, Clone)]
pub enum SupervisorError {
    /// Maximum restart intensity exceeded
    MaxRestartsExceeded { restarts: u32, seconds: u32 },
    /// Unknown child pid
    UnknownChild(Pid),
    /// Failed to start child
    StartFailed(String),
}

impl std::fmt::Display for SupervisorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SupervisorError::MaxRestartsExceeded { restarts, seconds } => {
                write!(
                    f,
                    "Maximum restart intensity exceeded: {} restarts in {} seconds",
                    restarts, seconds
                )
            }
            SupervisorError::UnknownChild(pid) => {
                write!(f, "Unknown child process: {:?}", pid)
            }
            SupervisorError::StartFailed(msg) => {
                write!(f, "Failed to start child: {}", msg)
            }
        }
    }
}

impl std::error::Error for SupervisorError {}

/// Thread-safe supervisor handle for use across workers.
pub struct SupervisorHandle {
    inner: Arc<Mutex<Supervisor>>,
}

impl SupervisorHandle {
    /// Create a new supervisor handle.
    pub fn new(scheduler: Arc<Scheduler>, config: SupervisorConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Supervisor::new(scheduler, config))),
        }
    }

    /// Add a child specification.
    pub fn add_child(&self, spec: ChildSpec) {
        self.inner.lock().add_child(spec);
    }

    /// Start all children.
    pub fn start_children(&self) -> Result<(), SupervisorError> {
        self.inner.lock().start_children()
    }

    /// Handle a child exit.
    pub fn handle_child_exit(
        &self,
        pid: Pid,
        reason: &ExitReason,
    ) -> Result<bool, SupervisorError> {
        self.inner.lock().handle_child_exit(pid, reason)
    }

    /// Get list of child pids.
    pub fn child_pids(&self) -> Vec<Pid> {
        self.inner.lock().child_pids()
    }

    /// Get number of children.
    pub fn child_count(&self) -> usize {
        self.inner.lock().child_count()
    }

    /// Check if a pid is a supervised child.
    pub fn is_child(&self, pid: Pid) -> bool {
        self.inner.lock().is_child(pid)
    }
}

impl Clone for SupervisorHandle {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Chunk, Instruction, Value};
    use std::sync::atomic::AtomicU32;

    fn make_test_function(name: &str) -> Arc<FunctionValue> {
        Arc::new(FunctionValue {
            name: name.to_string(),
            arity: 0,
            param_names: Vec::new(),
            code: Arc::new(Chunk {
                code: vec![
                    Instruction::LoadConst(0, 0),
                    Instruction::Return(0),
                ],
                constants: vec![Value::Int64(42)],
                lines: Vec::new(),
                locals: Vec::new(),
                register_count: 256,
            }),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols: vec![],
        })
    }

    #[test]
    fn test_supervisor_config_default() {
        let config = SupervisorConfig::default();
        assert_eq!(config.strategy, RestartStrategy::OneForOne);
        assert_eq!(config.max_restarts, 3);
        assert_eq!(config.max_seconds, 5);
    }

    #[test]
    fn test_supervisor_config_builders() {
        let config = SupervisorConfig::one_for_all().with_intensity(10, 60);
        assert_eq!(config.strategy, RestartStrategy::OneForAll);
        assert_eq!(config.max_restarts, 10);
        assert_eq!(config.max_seconds, 60);
    }

    #[test]
    fn test_child_spec_builder() {
        let func = make_test_function("test");
        let spec = ChildSpec::new("worker1", func)
            .with_restart(RestartType::Transient)
            .with_shutdown(10000)
            .with_args(vec![GcValue::Int64(1), GcValue::Int64(2)]);

        assert_eq!(spec.id, "worker1");
        assert_eq!(spec.restart, RestartType::Transient);
        assert_eq!(spec.shutdown_ms, 10000);
        assert_eq!(spec.start_args.len(), 2);
    }

    #[test]
    fn test_supervisor_add_and_start_children() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(Arc::clone(&scheduler), SupervisorConfig::default());

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func.clone()));
        supervisor.add_child(ChildSpec::new("worker2", func.clone()));

        supervisor.start_children().unwrap();

        assert_eq!(supervisor.child_count(), 2);
        assert_eq!(supervisor.child_pids().len(), 2);
    }

    #[test]
    fn test_supervisor_one_for_one_restart() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(
            Arc::clone(&scheduler),
            SupervisorConfig::one_for_one().with_intensity(5, 10),
        );

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func.clone()));
        supervisor.add_child(ChildSpec::new("worker2", func.clone()));

        supervisor.start_children().unwrap();

        let original_pids = supervisor.child_pids();
        let pid_to_kill = original_pids[0];

        // Simulate child exit
        let result = supervisor
            .handle_child_exit(pid_to_kill, &ExitReason::Error("crash".to_string()))
            .unwrap();

        assert!(result); // Should have restarted

        // Should still have 2 children
        assert_eq!(supervisor.child_count(), 2);

        // The second child should be unchanged
        let new_pids = supervisor.child_pids();
        assert_eq!(new_pids[0], original_pids[1]); // worker2 unchanged
        assert_ne!(new_pids[1], pid_to_kill); // worker1 got new pid
    }

    #[test]
    fn test_supervisor_one_for_all_restart() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(
            Arc::clone(&scheduler),
            SupervisorConfig::one_for_all().with_intensity(5, 10),
        );

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func.clone()));
        supervisor.add_child(ChildSpec::new("worker2", func.clone()));

        supervisor.start_children().unwrap();

        let original_pids = supervisor.child_pids();

        // Simulate child exit
        supervisor
            .handle_child_exit(original_pids[0], &ExitReason::Error("crash".to_string()))
            .unwrap();

        // Should still have 2 children, but all with new pids
        assert_eq!(supervisor.child_count(), 2);

        let new_pids = supervisor.child_pids();
        assert_ne!(new_pids[0], original_pids[0]);
        assert_ne!(new_pids[1], original_pids[1]);
    }

    #[test]
    fn test_supervisor_rest_for_one_restart() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(
            Arc::clone(&scheduler),
            SupervisorConfig::rest_for_one().with_intensity(5, 10),
        );

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func.clone()));
        supervisor.add_child(ChildSpec::new("worker2", func.clone()));
        supervisor.add_child(ChildSpec::new("worker3", func.clone()));

        supervisor.start_children().unwrap();

        let original_pids = supervisor.child_pids();

        // Kill the second child
        supervisor
            .handle_child_exit(original_pids[1], &ExitReason::Error("crash".to_string()))
            .unwrap();

        // Should still have 3 children
        assert_eq!(supervisor.child_count(), 3);

        let new_pids = supervisor.child_pids();
        // First child unchanged
        assert_eq!(new_pids[0], original_pids[0]);
        // Second and third children restarted
        assert_ne!(new_pids[1], original_pids[1]);
        assert_ne!(new_pids[2], original_pids[2]);
    }

    #[test]
    fn test_supervisor_transient_no_restart_on_normal() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(Arc::clone(&scheduler), SupervisorConfig::default());

        let func = make_test_function("worker");
        supervisor.add_child(
            ChildSpec::new("worker1", func).with_restart(RestartType::Transient),
        );

        supervisor.start_children().unwrap();

        let pid = supervisor.child_pids()[0];

        // Normal exit should NOT restart
        let result = supervisor
            .handle_child_exit(pid, &ExitReason::Normal)
            .unwrap();

        assert!(!result); // No restart
        assert_eq!(supervisor.child_count(), 0);
    }

    #[test]
    fn test_supervisor_temporary_never_restarts() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(Arc::clone(&scheduler), SupervisorConfig::default());

        let func = make_test_function("worker");
        supervisor.add_child(
            ChildSpec::new("worker1", func).with_restart(RestartType::Temporary),
        );

        supervisor.start_children().unwrap();

        let pid = supervisor.child_pids()[0];

        // Even error exit should NOT restart
        let result = supervisor
            .handle_child_exit(pid, &ExitReason::Error("crash".to_string()))
            .unwrap();

        assert!(!result); // No restart
        assert_eq!(supervisor.child_count(), 0);
    }

    #[test]
    fn test_supervisor_max_restarts_exceeded() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(
            Arc::clone(&scheduler),
            SupervisorConfig::one_for_one().with_intensity(2, 60), // Max 2 restarts
        );

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func));

        supervisor.start_children().unwrap();

        // First restart - OK
        let pid1 = supervisor.child_pids()[0];
        supervisor
            .handle_child_exit(pid1, &ExitReason::Error("crash".to_string()))
            .unwrap();

        // Second restart - OK
        let pid2 = supervisor.child_pids()[0];
        supervisor
            .handle_child_exit(pid2, &ExitReason::Error("crash".to_string()))
            .unwrap();

        // Third restart - should fail
        let pid3 = supervisor.child_pids()[0];
        let result = supervisor.handle_child_exit(pid3, &ExitReason::Error("crash".to_string()));

        assert!(matches!(
            result,
            Err(SupervisorError::MaxRestartsExceeded { .. })
        ));
    }

    #[test]
    fn test_supervisor_handle_thread_safe() {
        let scheduler = Arc::new(Scheduler::new());
        let handle = SupervisorHandle::new(Arc::clone(&scheduler), SupervisorConfig::default());

        let func = make_test_function("worker");
        handle.add_child(ChildSpec::new("worker1", func));

        handle.start_children().unwrap();

        assert_eq!(handle.child_count(), 1);

        // Clone and use from another "thread"
        let handle2 = handle.clone();
        assert_eq!(handle2.child_count(), 1);
    }

    #[test]
    fn test_supervisor_is_child() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(Arc::clone(&scheduler), SupervisorConfig::default());

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func));

        supervisor.start_children().unwrap();

        let child_pid = supervisor.child_pids()[0];
        assert!(supervisor.is_child(child_pid));
        assert!(!supervisor.is_child(Pid(999)));
    }

    #[test]
    fn test_supervisor_get_child_spec() {
        let scheduler = Arc::new(Scheduler::new());
        let mut supervisor = Supervisor::new(Arc::clone(&scheduler), SupervisorConfig::default());

        let func = make_test_function("worker");
        supervisor.add_child(ChildSpec::new("worker1", func));

        assert!(supervisor.get_child_spec("worker1").is_some());
        assert!(supervisor.get_child_spec("nonexistent").is_none());
    }
}
