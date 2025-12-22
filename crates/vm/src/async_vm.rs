//! Async VM implementation using Tokio.
//!
//! Each process runs as a tokio task, enabling:
//! - Natural async/await for blocking operations (mvar locks, receive, I/O)
//! - Work-stealing across CPU cores via tokio's multi-threaded runtime
//! - Proper yielding without manual try_lock + retry loops
//!
//! JIT Compatibility:
//! - JIT-compiled code runs synchronously within a task
//! - For blocking ops, JIT returns control to interpreter which does .await
//! - yield_now() is called every N instructions for fairness

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use imbl::{HashMap as ImblHashMap, HashSet as ImblHashSet};

use tokio::sync::{mpsc, RwLock as TokioRwLock, OwnedRwLockReadGuard, OwnedRwLockWriteGuard};
// LocalSet removed - now using multi-threaded runtime with tokio::spawn

use std::pin::Pin;
use std::task::{Context, Poll};
use std::future::Future;

/// Wrapper to assert that a future is Send.
///
/// SAFETY: This is safe because all types used in AsyncProcess are Send:
/// - GcValue, GcPtr, Heap, etc. have explicit `unsafe impl Send`
/// - All collections use Send-safe types
/// - Cross-process communication uses ThreadSafeValue
///
/// The compiler can't prove this automatically because async fn generates
/// opaque future types that don't propagate unsafe impl Send bounds.
pub struct AssertSend<F>(pub F);

unsafe impl<F: Future> Send for AssertSend<F> {}

impl<F: Future> Future for AssertSend<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We're just forwarding the poll to the inner future.
        // The Pin projection is safe because we never move the inner future.
        unsafe { Pin::new_unchecked(&mut self.get_unchecked_mut().0) }.poll(cx)
    }
}

use crate::gc::{GcConfig, GcList, GcMapKey, GcValue, Heap, GcNativeFn};

/// Held mvar lock guard (owned so it can be stored).
pub enum HeldMvarLock {
    Read(OwnedRwLockReadGuard<ThreadSafeValue>),
    Write(OwnedRwLockWriteGuard<ThreadSafeValue>),
}

// HeldMvarLock is Send because:
// - OwnedRwLockReadGuard<T> is Send if T: Send + Sync
// - OwnedRwLockWriteGuard<T> is Send if T: Send
// - ThreadSafeValue is designed to be Send + Sync (only contains primitives, String, Arc)
unsafe impl Send for HeldMvarLock {}
use crate::process::{CallFrame, ExceptionHandler, ProcessState, ThreadSafeValue, ProfileData};
use crate::value::{FunctionValue, Pid, TypeValue, RefId, RuntimeError, Value};
use crate::shared_types::SendableValue;
use crate::io_runtime::{IoRequest, IoRuntime};
use crate::process::IoResponseValue;

/// Reductions per yield (how often we call yield_now for fairness).
const REDUCTIONS_PER_YIELD: usize = 100; // Reduced for better fairness under lock contention

/// Type for process mailbox sender.
pub type MailboxSender = mpsc::UnboundedSender<ThreadSafeValue>;
/// Type for process mailbox receiver.
pub type MailboxReceiver = mpsc::UnboundedReceiver<ThreadSafeValue>;

/// Handle for a threaded evaluation, allowing independent cancellation.
pub struct ThreadedEvalHandle {
    /// Channel receiver for the result.
    pub result_rx: std::sync::mpsc::Receiver<Result<SendableValue, String>>,
    /// Interrupt flag for this specific evaluation.
    pub interrupt: Arc<AtomicBool>,
}

impl ThreadedEvalHandle {
    /// Interrupt this evaluation.
    pub fn cancel(&self) {
        self.interrupt.store(true, Ordering::SeqCst);
    }

    /// Check if this evaluation was interrupted.
    pub fn is_cancelled(&self) -> bool {
        self.interrupt.load(Ordering::SeqCst)
    }

    /// Poll for result (non-blocking).
    pub fn try_recv(&self) -> Result<Result<SendableValue, String>, std::sync::mpsc::TryRecvError> {
        self.result_rx.try_recv()
    }
}

/// Shared state across all async processes.
pub struct AsyncSharedState {
    /// Global functions - uses RwLock for concurrent eval support.
    pub functions: RwLock<HashMap<String, Arc<FunctionValue>>>,
    /// Function list for indexed calls - uses RwLock for concurrent eval support.
    pub function_list: RwLock<Vec<Arc<FunctionValue>>>,
    /// Native functions.
    pub natives: HashMap<String, Arc<GcNativeFn>>,
    /// Type definitions - uses RwLock for concurrent eval support.
    pub types: RwLock<HashMap<String, Arc<TypeValue>>>,

    /// JIT-compiled functions (by arity) - uses RwLock for concurrent eval support.
    pub jit_int_functions: RwLock<HashMap<u16, crate::shared_types::JitIntFn>>,
    pub jit_int_functions_0: RwLock<HashMap<u16, crate::shared_types::JitIntFn0>>,
    pub jit_int_functions_2: RwLock<HashMap<u16, crate::shared_types::JitIntFn2>>,
    pub jit_int_functions_3: RwLock<HashMap<u16, crate::shared_types::JitIntFn3>>,
    pub jit_int_functions_4: RwLock<HashMap<u16, crate::shared_types::JitIntFn4>>,
    pub jit_loop_array_functions: RwLock<HashMap<u16, crate::shared_types::JitLoopArrayFn>>,
    /// JIT recursive array fill functions (arity 2: arr, idx)
    pub jit_array_fill_functions: RwLock<HashMap<u16, crate::shared_types::JitArrayFillFn>>,
    /// JIT recursive array sum functions (arity 3: arr, idx, acc)
    pub jit_array_sum_functions: RwLock<HashMap<u16, crate::shared_types::JitArraySumFn>>,

    /// Shutdown signal (permanent).
    pub shutdown: AtomicBool,

    /// Interrupt signal (temporary, for Ctrl+C).
    /// Stops current execution but allows VM to be reused.
    pub interrupt: AtomicBool,

    /// Process registry: Pid -> mailbox sender.
    /// Protected by tokio RwLock for async access.
    pub process_registry: TokioRwLock<HashMap<Pid, MailboxSender>>,

    /// Module-level mutable variables (mvars).
    /// Key is "module_name.var_name", value is protected by tokio RwLock.
    pub mvars: HashMap<String, Arc<TokioRwLock<ThreadSafeValue>>>,

    /// Dynamic mvars from eval() - uses std RwLock for compatibility with ParallelVM API.
    pub dynamic_mvars: Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>>,

    /// Dynamic functions from eval() - uses std RwLock for compatibility.
    pub dynamic_functions: Arc<RwLock<HashMap<String, Arc<FunctionValue>>>>,

    /// Dynamic types from eval() - uses std RwLock for compatibility.
    pub dynamic_types: Arc<RwLock<HashMap<String, Arc<TypeValue>>>>,

    /// Stdlib functions - uses std RwLock for compatibility.
    pub stdlib_functions: Arc<RwLock<HashMap<String, Arc<FunctionValue>>>>,

    /// Stdlib types - uses std RwLock for compatibility.
    pub stdlib_types: Arc<RwLock<HashMap<String, Arc<TypeValue>>>>,

    /// Prelude imports - uses std RwLock for compatibility.
    pub prelude_imports: Arc<RwLock<HashMap<String, String>>>,

    /// Stdlib function list (ordered names) - preserves indices for CallDirect.
    pub stdlib_function_list: Arc<RwLock<Vec<String>>>,

    /// Eval callback - uses std RwLock for compatibility.
    pub eval_callback: Arc<RwLock<Option<Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>>>>,

    /// IO request sender.
    pub io_sender: Option<mpsc::UnboundedSender<crate::io_runtime::IoRequest>>,

    /// Inspect sender (for sending values to TUI inspector).
    pub inspect_sender: Option<crate::shared_types::InspectSender>,

    /// Output sender (for println from any process to TUI console).
    pub output_sender: Option<crate::shared_types::OutputSender>,

    /// Panel command sender (for Panel.* calls from Nostos code).
    pub panel_command_sender: Option<crate::shared_types::PanelCommandSender>,

    /// PID counter for generating unique PIDs.
    pub next_pid: AtomicU64,

    /// Whether profiling is enabled.
    pub profiling_enabled: bool,
}

impl AsyncSharedState {
    /// Allocate a new unique PID.
    pub fn alloc_pid(&self) -> Pid {
        Pid(self.next_pid.fetch_add(1, Ordering::SeqCst))
    }

    /// Register a process in the registry.
    pub async fn register_process(&self, pid: Pid, sender: MailboxSender) {
        self.process_registry.write().await.insert(pid, sender);
    }

    /// Unregister a process from the registry.
    pub async fn unregister_process(&self, pid: Pid) {
        self.process_registry.write().await.remove(&pid);
    }

    /// Send a message to a process by PID.
    pub async fn send_message(&self, target_pid: Pid, message: ThreadSafeValue) -> bool {
        if let Some(sender) = self.process_registry.read().await.get(&target_pid) {
            sender.send(message).is_ok()
        } else {
            false // Process not found or dead
        }
    }
}

/// An async process (runs as a tokio task).
pub struct AsyncProcess {
    /// Unique process identifier.
    pub pid: Pid,

    /// Process-local garbage-collected heap.
    pub heap: Heap,

    /// Call stack frames.
    pub frames: Vec<CallFrame>,

    /// Pool of reusable register vectors.
    pub register_pool: Vec<Vec<GcValue>>,

    /// Mailbox receiver for incoming messages.
    pub mailbox: MailboxReceiver,

    /// Mailbox sender (for cloning to other processes).
    pub mailbox_sender: MailboxSender,

    /// Current state.
    pub state: ProcessState,

    /// Instruction count since last yield.
    pub instructions_since_yield: usize,

    /// Linked processes.
    pub links: Vec<Pid>,

    /// Monitors.
    pub monitors: HashMap<RefId, Pid>,

    /// Processes monitoring this one.
    pub monitored_by: HashMap<RefId, Pid>,

    /// Exception handlers stack.
    pub handlers: Vec<ExceptionHandler>,

    /// Current exception.
    pub current_exception: Option<GcValue>,

    /// Exit value.
    pub exit_value: Option<GcValue>,

    /// Output buffer.
    pub output: Vec<String>,

    /// When this process was created.
    pub started_at: Instant,

    /// Shared state reference.
    pub shared: Arc<AsyncSharedState>,

    /// Held mvar locks (name -> guard).
    pub held_mvar_locks: HashMap<String, HeldMvarLock>,

    /// Reentrant lock depth counters (name -> depth).
    pub mvar_lock_depths: HashMap<String, usize>,

    /// Profiling data (only populated when profiling is enabled).
    pub profile: Option<ProfileData>,

    /// Local interrupt flag for this process (used for independent eval cancellation).
    /// If Some, this takes precedence over the shared interrupt flag.
    pub local_interrupt: Option<Arc<AtomicBool>>,

    // === Debugger state ===

    /// Active breakpoints
    pub breakpoints: std::collections::HashSet<crate::shared_types::Breakpoint>,

    /// Current step mode
    pub step_mode: crate::shared_types::StepMode,

    /// Last source line (for step-line detection)
    pub debug_last_line: usize,

    /// Frame depth when step-over/step-out started
    pub debug_step_frame_depth: usize,

    /// Debug command receiver (from debugger)
    pub debug_command_receiver: Option<crate::shared_types::DebugCommandReceiver>,

    /// Debug event sender (to debugger)
    pub debug_event_sender: Option<crate::shared_types::DebugEventSender>,

    /// Skip the next breakpoint check (used after Continue to avoid re-hitting same breakpoint)
    pub skip_next_breakpoint_check: bool,
}

/// Helper trait to convert register/constant indices (u8, u16, etc.) to usize.
/// Works with both owned values and references (needed because matching on &Instruction
/// gives references to fields).
trait AsIdx {
    fn as_idx(&self) -> usize;
}

impl AsIdx for u8 {
    #[inline(always)]
    fn as_idx(&self) -> usize { *self as usize }
}

impl AsIdx for u16 {
    #[inline(always)]
    fn as_idx(&self) -> usize { *self as usize }
}

impl AsIdx for i16 {
    #[inline(always)]
    fn as_idx(&self) -> usize { *self as usize }
}

// AsyncProcess is safe to Send between threads:
// - Each process has its own Heap and state
// - Processes don't share mutable data with other processes
// - Cross-process communication uses ThreadSafeValue
unsafe impl Send for AsyncProcess {}

impl AsyncProcess {
    /// Create a new async process.
    pub fn new(pid: Pid, shared: Arc<AsyncSharedState>) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let profiling_enabled = shared.profiling_enabled;
        Self {
            pid,
            heap: Heap::with_config(GcConfig::default()),
            frames: Vec::new(),
            register_pool: Vec::new(),
            mailbox: receiver,
            mailbox_sender: sender,
            state: ProcessState::Running,
            instructions_since_yield: 0,
            links: Vec::new(),
            monitors: HashMap::new(),
            monitored_by: HashMap::new(),
            handlers: Vec::new(),
            current_exception: None,
            exit_value: None,
            output: Vec::new(),
            started_at: Instant::now(),
            shared,
            held_mvar_locks: HashMap::new(),
            mvar_lock_depths: HashMap::new(),
            profile: if profiling_enabled { Some(ProfileData::new()) } else { None },
            local_interrupt: None,
            // Debugger state
            breakpoints: std::collections::HashSet::new(),
            step_mode: crate::shared_types::StepMode::Run,
            debug_last_line: 0,
            debug_step_frame_depth: 0,
            debug_command_receiver: None,
            debug_event_sender: None,
            skip_next_breakpoint_check: false,
        }
    }

    /// Create a new async process with a pre-created mailbox.
    /// Used when the mailbox must be registered BEFORE the process starts (to avoid race conditions).
    pub fn new_with_mailbox(pid: Pid, shared: Arc<AsyncSharedState>, sender: MailboxSender, receiver: MailboxReceiver) -> Self {
        let profiling_enabled = shared.profiling_enabled;
        Self {
            pid,
            heap: Heap::with_config(GcConfig::default()),
            frames: Vec::new(),
            register_pool: Vec::new(),
            mailbox: receiver,
            mailbox_sender: sender,
            state: ProcessState::Running,
            instructions_since_yield: 0,
            links: Vec::new(),
            monitors: HashMap::new(),
            monitored_by: HashMap::new(),
            handlers: Vec::new(),
            current_exception: None,
            exit_value: None,
            output: Vec::new(),
            started_at: Instant::now(),
            shared,
            held_mvar_locks: HashMap::new(),
            mvar_lock_depths: HashMap::new(),
            profile: if profiling_enabled { Some(ProfileData::new()) } else { None },
            local_interrupt: None,
            // Debugger state
            breakpoints: std::collections::HashSet::new(),
            step_mode: crate::shared_types::StepMode::Run,
            debug_last_line: 0,
            debug_step_frame_depth: 0,
            debug_command_receiver: None,
            debug_event_sender: None,
            skip_next_breakpoint_check: false,
        }
    }

    /// Maybe yield to other tasks for fairness.
    /// Called periodically during instruction execution.
    #[inline]
    pub async fn maybe_yield(&mut self) {
        self.instructions_since_yield += 1;
        if self.instructions_since_yield >= REDUCTIONS_PER_YIELD {
            self.instructions_since_yield = 0;
            tokio::task::yield_now().await;
        }
    }

    /// Record function entry for profiling.
    #[inline]
    pub fn profile_enter(&mut self, func_name: &str) {
        if let Some(ref mut profile) = self.profile {
            profile.enter_function(func_name.to_string());
        }
    }

    /// Record function exit for profiling.
    #[inline]
    pub fn profile_exit(&mut self) {
        if let Some(ref mut profile) = self.profile {
            profile.exit_function();
        }
    }

    /// Record a JIT function call for profiling.
    /// This is called at the JIT dispatch boundary, not inside the JIT code.
    #[inline]
    pub fn profile_jit_call(&mut self, func_name: &str, duration: std::time::Duration) {
        if let Some(ref mut profile) = self.profile {
            let jit_name = format!("[JIT] {}", func_name);
            profile.stats
                .entry(jit_name)
                .or_insert_with(crate::process::FunctionStats::new)
                .record_call(duration.as_nanos() as u64);
        }
    }

    /// Check if profiling is enabled.
    #[inline]
    pub fn is_profiling(&self) -> bool {
        self.profile.is_some()
    }

    // === Debugger Methods ===

    /// Get current line number from instruction pointer.
    fn debug_current_line(&self) -> Option<usize> {
        if let Some(frame) = self.frames.last() {
            let ip = if frame.ip > 0 { frame.ip - 1 } else { 0 };
            frame.function.code.lines.get(ip).copied()
        } else {
            None
        }
    }

    /// Get current function name.
    fn debug_current_function(&self) -> String {
        self.frames.last()
            .map(|f| f.function.name.clone())
            .unwrap_or_else(|| "<unknown>".to_string())
    }

    /// Get current source file.
    fn debug_current_file(&self) -> Option<String> {
        self.frames.last().and_then(|f| f.function.source_file.clone())
    }

    /// Get the first line number of the current function.
    fn debug_function_first_line(&self) -> usize {
        self.frames.last()
            .and_then(|f| f.function.code.lines.first().copied())
            .unwrap_or(0)
    }

    /// Check if we should pause execution (breakpoint or step mode).
    fn debug_should_pause(&mut self) -> bool {
        use crate::shared_types::StepMode;

        // Skip if not debugging
        if self.debug_event_sender.is_none() {
            return false;
        }

        // Skip breakpoint check once after Continue (to avoid re-hitting same breakpoint)
        if self.skip_next_breakpoint_check {
            self.skip_next_breakpoint_check = false;
            return false;
        }

        let current_line = match self.debug_current_line() {
            Some(line) => line,
            None => return false,
        };

        match self.step_mode {
            StepMode::Run => {
                // Check function breakpoints
                let fn_name = self.debug_current_function();
                if fn_name != "<unknown>" {
                    // Check for exact match
                    let fn_bp = crate::shared_types::Breakpoint::Function(fn_name.clone());
                    if self.breakpoints.contains(&fn_bp) {
                        // Only break on function entry (first line of function)
                        if current_line == self.debug_function_first_line() {
                            self.step_mode = StepMode::Paused;
                            return true;
                        }
                    }
                    // Also check without arity suffix (e.g., "foo" matches "foo/2")
                    if let Some(base_name) = fn_name.split('/').next() {
                        let base_bp = crate::shared_types::Breakpoint::Function(base_name.to_string());
                        if self.breakpoints.contains(&base_bp) {
                            if current_line == self.debug_function_first_line() {
                                self.step_mode = StepMode::Paused;
                                return true;
                            }
                        }
                    }
                }

                // Check line breakpoints
                let file = self.debug_current_file();
                let bp = crate::shared_types::Breakpoint::Line { file, line: current_line };
                if self.breakpoints.contains(&bp) {
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                // Also check file-agnostic breakpoint
                let bp_any = crate::shared_types::Breakpoint::Line { file: None, line: current_line };
                if self.breakpoints.contains(&bp_any) {
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                false
            }
            StepMode::Paused => {
                // Already paused, stay paused
                true
            }
            StepMode::StepInstruction => {
                // Pause on every instruction
                self.step_mode = StepMode::Paused;
                true
            }
            StepMode::StepLine => {
                // Pause when line changes
                if current_line != self.debug_last_line {
                    self.debug_last_line = current_line;
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                false
            }
            StepMode::StepOver => {
                // Pause when line changes at same or lower call depth
                if self.frames.len() <= self.debug_step_frame_depth && current_line != self.debug_last_line {
                    self.debug_last_line = current_line;
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                false
            }
            StepMode::StepOut => {
                // Pause when we return to lower call depth
                if self.frames.len() < self.debug_step_frame_depth {
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                false
            }
        }
    }

    /// Send a debug event.
    fn debug_send_event(&self, event: crate::shared_types::DebugEvent) {
        if let Some(ref sender) = self.debug_event_sender {
            let _ = sender.send(event);
        }
    }

    /// Handle debug commands while paused.
    fn debug_handle_commands(&mut self) {
        use crate::shared_types::{DebugCommand, DebugEvent, StepMode, StackFrame};

        let receiver = match &self.debug_command_receiver {
            Some(r) => r.clone(),
            None => return,
        };

        // Send paused event with current location
        let line = self.debug_current_line().unwrap_or(0);
        let function = self.debug_current_function();
        let file = self.debug_current_file();
        self.debug_send_event(DebugEvent::Paused { pid: self.pid.0, file: file.clone(), line, function: function.clone() });

        // Process commands until we get a continue/step command
        while self.step_mode == StepMode::Paused {
            match receiver.recv() {
                Ok(cmd) => match cmd {
                    DebugCommand::Continue => {
                        self.step_mode = StepMode::Run;
                        // Skip next breakpoint check to avoid re-hitting the same breakpoint
                        self.skip_next_breakpoint_check = true;
                    }
                    DebugCommand::StepInstruction => {
                        self.step_mode = StepMode::StepInstruction;
                    }
                    DebugCommand::StepLine => {
                        self.debug_last_line = line;
                        self.step_mode = StepMode::StepLine;
                    }
                    DebugCommand::StepOver => {
                        self.debug_last_line = line;
                        self.debug_step_frame_depth = self.frames.len();
                        self.step_mode = StepMode::StepOver;
                    }
                    DebugCommand::StepOut => {
                        self.debug_step_frame_depth = self.frames.len();
                        self.step_mode = StepMode::StepOut;
                    }
                    DebugCommand::AddBreakpoint(bp) => {
                        self.breakpoints.insert(bp);
                    }
                    DebugCommand::RemoveBreakpoint(bp) => {
                        self.breakpoints.remove(&bp);
                    }
                    DebugCommand::ListBreakpoints => {
                        self.debug_send_event(DebugEvent::Breakpoints {
                            breakpoints: self.breakpoints.iter().cloned().collect(),
                        });
                    }
                    DebugCommand::PrintVariable(name) => {
                        // Find variable in current frame's debug symbols
                        if let Some(frame) = self.frames.last() {
                            let mut found = false;
                            for sym in &frame.function.debug_symbols {
                                if sym.name == name {
                                    let value = frame.registers.get(sym.register as usize)
                                        .map(|v| format!("{:?}", v))
                                        .unwrap_or_else(|| "<invalid register>".to_string());
                                    self.debug_send_event(DebugEvent::Variable {
                                        name: name.clone(),
                                        value,
                                        type_name: "unknown".to_string(),
                                    });
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                self.debug_send_event(DebugEvent::Error {
                                    message: format!("Variable '{}' not found", name),
                                });
                            }
                        }
                    }
                    DebugCommand::PrintLocals => {
                        if let Some(frame) = self.frames.last() {
                            let variables: Vec<(String, String, String)> = frame.function.debug_symbols.iter()
                                .filter_map(|sym| {
                                    frame.registers.get(sym.register as usize).map(|v| {
                                        (sym.name.clone(), format!("{:?}", v), "unknown".to_string())
                                    })
                                })
                                .collect();
                            self.debug_send_event(DebugEvent::Locals { variables });
                        }
                    }
                    DebugCommand::PrintStack => {
                        let frames: Vec<StackFrame> = self.frames.iter().rev().map(|f| {
                            let line = f.function.code.lines.get(f.ip.saturating_sub(1)).copied().unwrap_or(0);
                            StackFrame {
                                function: f.function.name.clone(),
                                file: f.function.source_file.clone(),
                                line,
                                locals: f.function.debug_symbols.iter().map(|s| s.name.clone()).collect(),
                            }
                        }).collect();
                        self.debug_send_event(DebugEvent::Stack { frames });
                    }
                },
                Err(_) => {
                    // Channel closed, resume execution
                    self.step_mode = StepMode::Run;
                    break;
                }
            }
        }
    }

    /// Get a register vec from the pool, or allocate a new one.
    #[inline]
    fn alloc_registers(&mut self, size: usize) -> Vec<GcValue> {
        // Pop from pool if available (LIFO for cache locality)
        // We don't search for exact size match - any capacity works
        if let Some(mut regs) = self.register_pool.pop() {
            // Truncate to exact size (drops extra elements for GC correctness)
            regs.truncate(size);
            // Extend if needed
            if regs.len() < size {
                regs.resize(size, GcValue::Unit);
            }
            regs
        } else {
            vec![GcValue::Unit; size]
        }
    }

    /// Return a register vec to the pool for reuse.
    #[inline]
    fn free_registers(&mut self, mut regs: Vec<GcValue>) {
        regs.clear();
        if regs.capacity() <= 64 && self.register_pool.len() < 32 {
            self.register_pool.push(regs);
        }
    }

    /// Convert ThreadSafeValue to GcValue for mvar read.
    /// Maps are returned as SharedMap (O(1) Arc clone) instead of deep copy.
    fn mvar_value_to_gc(&mut self, value: &ThreadSafeValue) -> GcValue {
        match value {
            // For maps: return SharedMap directly (O(1) sharing for mvars)
            ThreadSafeValue::Map(shared_map) => GcValue::SharedMap(shared_map.clone()),
            // For everything else: use standard conversion
            _ => value.to_gc_value(&mut self.heap),
        }
    }

    /// Read from an mvar (async - yields if locked).
    pub async fn mvar_read(&mut self, name: &str) -> Result<GcValue, RuntimeError> {
        // Check if we already hold a lock on this mvar
        if let Some(guard) = self.held_mvar_locks.get(name) {
            // Read from held guard
            let value: ThreadSafeValue = match guard {
                HeldMvarLock::Read(g) => (**g).clone(),
                HeldMvarLock::Write(g) => (**g).clone(),
            };
            let gc_value = self.mvar_value_to_gc(&value);
            return Ok(gc_value);
        }

        let var = self.shared.mvars.get(name)
            .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?
            .clone();

        // Async read - will yield if write-locked
        let guard = var.read().await;
        let value = (*guard).clone();
        drop(guard);  // Release lock before converting
        let gc_value = self.mvar_value_to_gc(&value);
        Ok(gc_value)
    }

    /// Write to an mvar (async - yields if locked).
    pub async fn mvar_write(&mut self, name: &str, value: GcValue) -> Result<(), RuntimeError> {
        let safe_value = ThreadSafeValue::from_gc_value(&value, &self.heap)
            .ok_or_else(|| RuntimeError::Panic(format!("Cannot convert value for mvar: {}", name)))?;

        // Check if we already hold a WRITE lock on this mvar
        if let Some(guard) = self.held_mvar_locks.get_mut(name) {
            match guard {
                HeldMvarLock::Write(g) => {
                    **g = safe_value;
                    return Ok(());
                }
                HeldMvarLock::Read(_) => {
                    // Have read lock but need write - fall through to acquire
                }
            }
        }

        let var = self.shared.mvars.get(name)
            .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?;

        // Async write - will yield if any lock held
        let mut guard = var.write().await;
        *guard = safe_value;
        Ok(())
    }

    /// Receive a message (async - yields until message arrives).
    pub async fn receive(&mut self) -> Option<GcValue> {
        match self.mailbox.recv().await {
            Some(msg) => Some(msg.to_gc_value(&mut self.heap)),
            None => None, // Channel closed
        }
    }

    /// Receive a message with timeout.
    pub async fn receive_timeout(&mut self, timeout: Duration) -> Option<GcValue> {
        match tokio::time::timeout(timeout, self.mailbox.recv()).await {
            Ok(Some(msg)) => Some(msg.to_gc_value(&mut self.heap)),
            Ok(None) => None, // Channel closed
            Err(_) => None, // Timeout
        }
    }

    /// Try to receive without blocking (for non-blocking checks).
    pub fn try_receive(&mut self) -> Option<GcValue> {
        match self.mailbox.try_recv() {
            Ok(msg) => Some(msg.to_gc_value(&mut self.heap)),
            Err(_) => None,
        }
    }

    /// Send a message to another process.
    pub async fn send(&self, target_pid: Pid, value: GcValue) -> bool {
        let safe_value = match ThreadSafeValue::from_gc_value(&value, &self.heap) {
            Some(v) => v,
            None => return false,
        };
        self.shared.send_message(target_pid, safe_value).await
    }

    /// Spawn a new process running the given function.
    /// Returns the new process's PID.
    ///
    /// Note: This must be called from within a LocalSet context because
    /// GcValue contains raw pointers that aren't Send.
    pub async fn spawn_process(
        shared: Arc<AsyncSharedState>,
        function: Arc<FunctionValue>,
        args: Vec<GcValue>,
        parent_heap: &Heap,
    ) -> Pid {
        let pid = shared.alloc_pid();
        let shared_clone = shared.clone();

        // Convert args to thread-safe values (deep copy)
        let safe_args: Vec<ThreadSafeValue> = args.iter()
            .filter_map(|v| ThreadSafeValue::from_gc_value(v, parent_heap))
            .collect();

        // Create process and register it
        let mut process = AsyncProcess::new(pid, shared_clone.clone());
        let sender = process.mailbox_sender.clone();
        shared.register_process(pid, sender).await;

        // Spawn as tokio task (can run on any thread)
        // Wrap with AssertSend because all underlying types are Send but the compiler
        // can't prove it through async fn's opaque future types.
        tokio::spawn(AssertSend(async move {
            // Set up initial call frame
            let args_gc: Vec<GcValue> = safe_args.iter()
                .map(|v| v.to_gc_value(&mut process.heap))
                .collect();

            let mut registers = vec![GcValue::Unit; function.code.register_count as usize];
            for (i, arg) in args_gc.into_iter().enumerate() {
                if i < registers.len() {
                    registers[i] = arg;
                }
            }

            process.frames.push(CallFrame {
                function,
                ip: 0,
                registers,
                captures: Vec::new(),
                return_reg: None,
            });

            // Run the process
            let result = process.run().await;

            // Unregister on exit
            shared_clone.unregister_process(pid).await;

            result
        }));

        pid
    }

    /// Main execution loop for this process.
    pub async fn run(&mut self) -> Result<GcValue, RuntimeError> {
        loop {
            // Only check shutdown periodically to avoid atomic load overhead
            self.instructions_since_yield += 1;
            if self.instructions_since_yield >= REDUCTIONS_PER_YIELD {
                self.instructions_since_yield = 0;

                // Check shutdown
                if self.shared.shutdown.load(Ordering::SeqCst) {
                    return Ok(GcValue::Unit);
                }

                // Check interrupt (Ctrl+C) - local flag takes precedence
                let interrupted = self.local_interrupt
                    .as_ref()
                    .map(|i| i.load(Ordering::SeqCst))
                    .unwrap_or_else(|| self.shared.interrupt.load(Ordering::SeqCst));
                if interrupted {
                    return Err(RuntimeError::Interrupted);
                }

                // Yield for fairness
                tokio::task::yield_now().await;
            }

            // Check if we have frames to execute
            if self.frames.is_empty() {
                return Ok(self.exit_value.take().unwrap_or(GcValue::Unit));
            }

            // Execute one instruction
            match self.step().await {
                Ok(StepResult::Continue) => {
                    // Continue to next instruction (yield check is at top of loop)
                }
                Ok(StepResult::Finished(value)) => {
                    return Ok(value);
                }
                Err(e) => {
                    // Try to handle exception
                    if !self.handle_exception(&e) {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Execute instructions in a tight loop until we need to yield or change frames.
    async fn step(&mut self) -> Result<StepResult, RuntimeError> {
        use crate::value::Instruction::{self, *};

        // Execute multiple instructions without returning to reduce async overhead
        'instruction_loop: loop {

        // Periodically check for interrupts and yield for fairness
        self.instructions_since_yield += 1;
        if self.instructions_since_yield >= REDUCTIONS_PER_YIELD {
            self.instructions_since_yield = 0;

            // Check interrupt (Ctrl+C) - local flag takes precedence
            let interrupted = self.local_interrupt
                .as_ref()
                .map(|i| i.load(Ordering::SeqCst))
                .unwrap_or_else(|| self.shared.interrupt.load(Ordering::SeqCst));
            if interrupted {
                return Err(RuntimeError::Interrupted);
            }

            // Yield for fairness (allow other tasks to run)
            tokio::task::yield_now().await;
        }

        // Check for debugger breakpoints/stepping
        if self.debug_should_pause() {
            self.debug_handle_commands();
        }

        // Cache frame index once - avoid repeated len() calls
        let cur_frame = match self.frames.len().checked_sub(1) {
            Some(idx) => idx,
            None => return Ok(StepResult::Finished(GcValue::Unit)),
        };

        // Get frame info and instruction pointer in one access
        // SAFETY: cur_frame is valid (computed from len() - 1)
        // SAFETY: ip is always valid - compiler ensures code ends with Return
        let instruction_ptr: *const Instruction = unsafe {
            let frame = self.frames.get_unchecked(cur_frame);
            frame.function.code.code.get_unchecked(frame.ip) as *const _
        };

        // Increment IP - SAFETY: cur_frame is valid (computed from len() - 1)
        unsafe { self.frames.get_unchecked_mut(cur_frame).ip += 1; }

        // SAFETY: The instruction pointer is valid because:
        // 1. frame.function is Arc<FunctionValue>, so it won't be deallocated
        // 2. We don't modify the code array during execution
        // 3. This dereference happens immediately, within the same step() call
        let instruction = unsafe { &*instruction_ptr };

        // Helper macros for register access - use cached frame index
        // SAFETY: cur_frame is computed from frames.len() - 1, so it's always valid.
        // Using get_unchecked avoids bounds checking overhead.
        // Note: $r and $idx may be references (when matching on &Instruction),
        // so we use AsIdx trait which works with both owned and referenced values
        macro_rules! reg {
            ($r:expr) => {{
                unsafe { self.frames.get_unchecked(cur_frame).registers.get_unchecked($r.as_idx()).clone() }
            }};
        }
        // Reference version - avoids clone when we only need to read the value
        macro_rules! reg_ref {
            ($r:expr) => {{
                unsafe { self.frames.get_unchecked(cur_frame).registers.get_unchecked($r.as_idx()) }
            }};
        }
        macro_rules! set_reg {
            ($r:expr, $v:expr) => {{
                unsafe { *self.frames.get_unchecked_mut(cur_frame).registers.get_unchecked_mut($r.as_idx()) = $v; }
            }};
        }
        macro_rules! get_const {
            ($idx:expr) => {{
                unsafe { self.frames.get_unchecked(cur_frame).function.code.constants.get_unchecked($idx.as_idx()).clone() }
            }};
        }

        match instruction {
            // === Constants and moves ===
            LoadConst(dst, idx) => {
                let value = get_const!(idx);
                let gc_value = self.heap.value_to_gc(&value);
                set_reg!(dst, gc_value);
            }
            LoadUnit(dst) => set_reg!(dst, GcValue::Unit),
            LoadTrue(dst) => set_reg!(dst, GcValue::Bool(true)),
            LoadFalse(dst) => set_reg!(dst, GcValue::Bool(false)),
            Move(dst, src) => {
                let v = reg!(src);
                set_reg!(dst, v);
            }

            // === Integer arithmetic ===
            AddInt(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Int8(x), GcValue::Int8(y)) => GcValue::Int8(x.wrapping_add(*y)),
                    (GcValue::Int16(x), GcValue::Int16(y)) => GcValue::Int16(x.wrapping_add(*y)),
                    (GcValue::Int32(x), GcValue::Int32(y)) => GcValue::Int32(x.wrapping_add(*y)),
                    (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x.wrapping_add(*y)),
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => GcValue::UInt8(x.wrapping_add(*y)),
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => GcValue::UInt16(x.wrapping_add(*y)),
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => GcValue::UInt32(x.wrapping_add(*y)),
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => GcValue::UInt64(x.wrapping_add(*y)),
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x + y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x + y),
                    (GcValue::BigInt(px), GcValue::BigInt(py)) => {
                        let bx = self.heap.get_bigint(*px).unwrap().value.clone();
                        let by = self.heap.get_bigint(*py).unwrap().value.clone();
                        let result_ptr = self.heap.alloc_bigint(&bx + &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x + *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(dst, result);
            }
            SubInt(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Int8(x), GcValue::Int8(y)) => GcValue::Int8(x.wrapping_sub(*y)),
                    (GcValue::Int16(x), GcValue::Int16(y)) => GcValue::Int16(x.wrapping_sub(*y)),
                    (GcValue::Int32(x), GcValue::Int32(y)) => GcValue::Int32(x.wrapping_sub(*y)),
                    (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x.wrapping_sub(*y)),
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => GcValue::UInt8(x.wrapping_sub(*y)),
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => GcValue::UInt16(x.wrapping_sub(*y)),
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => GcValue::UInt32(x.wrapping_sub(*y)),
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => GcValue::UInt64(x.wrapping_sub(*y)),
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x - y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x - y),
                    (GcValue::BigInt(px), GcValue::BigInt(py)) => {
                        let bx = self.heap.get_bigint(*px).unwrap().value.clone();
                        let by = self.heap.get_bigint(*py).unwrap().value.clone();
                        let result_ptr = self.heap.alloc_bigint(&bx - &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x - *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(dst, result);
            }
            MulInt(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Int8(x), GcValue::Int8(y)) => GcValue::Int8(x.wrapping_mul(*y)),
                    (GcValue::Int16(x), GcValue::Int16(y)) => GcValue::Int16(x.wrapping_mul(*y)),
                    (GcValue::Int32(x), GcValue::Int32(y)) => GcValue::Int32(x.wrapping_mul(*y)),
                    (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x.wrapping_mul(*y)),
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => GcValue::UInt8(x.wrapping_mul(*y)),
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => GcValue::UInt16(x.wrapping_mul(*y)),
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => GcValue::UInt32(x.wrapping_mul(*y)),
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => GcValue::UInt64(x.wrapping_mul(*y)),
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x * y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x * y),
                    (GcValue::BigInt(px), GcValue::BigInt(py)) => {
                        let bx = self.heap.get_bigint(*px).unwrap().value.clone();
                        let by = self.heap.get_bigint(*py).unwrap().value.clone();
                        let result_ptr = self.heap.alloc_bigint(&bx * &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x * *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(dst, result);
            }
            DivInt(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Int8(x), GcValue::Int8(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int8(x.wrapping_div(*y))
                    }
                    (GcValue::Int16(x), GcValue::Int16(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int16(x.wrapping_div(*y))
                    }
                    (GcValue::Int32(x), GcValue::Int32(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int32(x.wrapping_div(*y))
                    }
                    (GcValue::Int64(x), GcValue::Int64(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int64(x.wrapping_div(*y))
                    }
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt8(x.wrapping_div(*y))
                    }
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt16(x.wrapping_div(*y))
                    }
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt32(x.wrapping_div(*y))
                    }
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt64(x.wrapping_div(*y))
                    }
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x / y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x / y),
                    (GcValue::BigInt(px), GcValue::BigInt(py)) => {
                        let bx = self.heap.get_bigint(*px).unwrap().value.clone();
                        let by = self.heap.get_bigint(*py).unwrap().value.clone();
                        if by == num_bigint::BigInt::from(0) { return Err(RuntimeError::Panic("Division by zero".into())); }
                        let result_ptr = self.heap.alloc_bigint(&bx / &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => {
                        if *y == rust_decimal::Decimal::ZERO { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Decimal(*x / *y)
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(dst, result);
            }
            ModInt(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Int8(x), GcValue::Int8(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int8(x.wrapping_rem(*y))
                    }
                    (GcValue::Int16(x), GcValue::Int16(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int16(x.wrapping_rem(*y))
                    }
                    (GcValue::Int32(x), GcValue::Int32(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int32(x.wrapping_rem(*y))
                    }
                    (GcValue::Int64(x), GcValue::Int64(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::Int64(x.wrapping_rem(*y))
                    }
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt8(x.wrapping_rem(*y))
                    }
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt16(x.wrapping_rem(*y))
                    }
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt32(x.wrapping_rem(*y))
                    }
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => {
                        if *y == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                        GcValue::UInt64(x.wrapping_rem(*y))
                    }
                    (GcValue::BigInt(px), GcValue::BigInt(py)) => {
                        let bx = self.heap.get_bigint(*px).unwrap().value.clone();
                        let by = self.heap.get_bigint(*py).unwrap().value.clone();
                        if by == num_bigint::BigInt::from(0) { return Err(RuntimeError::Panic("Division by zero".into())); }
                        let result_ptr = self.heap.alloc_bigint(&bx % &by);
                        GcValue::BigInt(result_ptr)
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(dst, result);
            }
            NegInt(dst, src) => {
                let v = reg_ref!(src);
                let result = match v {
                    GcValue::Int8(n) => GcValue::Int8(-n),
                    GcValue::Int16(n) => GcValue::Int16(-n),
                    GcValue::Int32(n) => GcValue::Int32(-n),
                    GcValue::Int64(n) => GcValue::Int64(-n),
                    GcValue::Float32(n) => GcValue::Float32(-n),
                    GcValue::Float64(n) => GcValue::Float64(-n),
                    GcValue::BigInt(ptr) => {
                        let bi = self.heap.get_bigint(*ptr).unwrap().value.clone();
                        GcValue::BigInt(self.heap.alloc_bigint(-bi))
                    }
                    GcValue::Decimal(d) => GcValue::Decimal(-*d),
                    _ => return Err(RuntimeError::Panic("NegInt: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            AbsInt(dst, src) => {
                let v = reg_ref!(src);
                let result = match v {
                    GcValue::Int8(n) => GcValue::Int8(n.abs()),
                    GcValue::Int16(n) => GcValue::Int16(n.abs()),
                    GcValue::Int32(n) => GcValue::Int32(n.abs()),
                    GcValue::Int64(n) => GcValue::Int64(n.abs()),
                    GcValue::Float32(n) => GcValue::Float32(n.abs()),
                    GcValue::Float64(n) => GcValue::Float64(n.abs()),
                    GcValue::BigInt(ptr) => {
                        let bi = self.heap.get_bigint(*ptr).unwrap().value.clone();
                        let abs_bi = if bi < num_bigint::BigInt::from(0) { -bi } else { bi };
                        GcValue::BigInt(self.heap.alloc_bigint(abs_bi))
                    }
                    GcValue::Decimal(d) => GcValue::Decimal(d.abs()),
                    _ => return Err(RuntimeError::Panic("AbsInt: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            MinInt(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("MinInt: expected Int64".into())) };
                let vb = match reg_ref!(b) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("MinInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.min(vb)));
            }
            MaxInt(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("MaxInt: expected Int64".into())) };
                let vb = match reg_ref!(b) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("MaxInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.max(vb)));
            }

            // === Float arithmetic ===
            AddFloat(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x + y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x + y),
                    _ => return Err(RuntimeError::Panic("AddFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            SubFloat(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x - y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x - y),
                    _ => return Err(RuntimeError::Panic("SubFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            MulFloat(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x * y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x * y),
                    _ => return Err(RuntimeError::Panic("MulFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            DivFloat(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x / y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x / y),
                    _ => return Err(RuntimeError::Panic("DivFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            NegFloat(dst, src) => {
                let v = reg_ref!(src);
                let result = match v {
                    GcValue::Float64(n) => GcValue::Float64(-n),
                    GcValue::Float32(n) => GcValue::Float32(-n),
                    _ => return Err(RuntimeError::Panic("NegFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            AbsFloat(dst, src) => {
                let v = reg_ref!(src);
                let result = match v {
                    GcValue::Float64(n) => GcValue::Float64(n.abs()),
                    GcValue::Float32(n) => GcValue::Float32(n.abs()),
                    _ => return Err(RuntimeError::Panic("AbsFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            SqrtFloat(dst, src) => {
                let v = reg_ref!(src);
                let result = match v {
                    GcValue::Float64(n) => GcValue::Float64(n.sqrt()),
                    GcValue::Float32(n) => GcValue::Float32(n.sqrt()),
                    _ => return Err(RuntimeError::Panic("SqrtFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }
            PowFloat(dst, a, b) => {
                let va = reg_ref!(a);
                let vb = reg_ref!(b);
                let result = match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x.powf(*y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x.powf(*y)),
                    _ => return Err(RuntimeError::Panic("PowFloat: expected Float".into())),
                };
                set_reg!(dst, result);
            }

            // === Float comparisons ===
            EqFloat(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Float64(n) => *n, _ => return Err(RuntimeError::Panic("EqFloat: expected Float64".into())) };
                let vb = match reg_ref!(b) { GcValue::Float64(n) => *n, _ => return Err(RuntimeError::Panic("EqFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Bool(va == vb));
            }
            LtFloat(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Float64(n) => *n, _ => return Err(RuntimeError::Panic("LtFloat: expected Float64".into())) };
                let vb = match reg_ref!(b) { GcValue::Float64(n) => *n, _ => return Err(RuntimeError::Panic("LtFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Bool(va < vb));
            }
            LeFloat(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Float64(n) => *n, _ => return Err(RuntimeError::Panic("LeFloat: expected Float64".into())) };
                let vb = match reg_ref!(b) { GcValue::Float64(n) => *n, _ => return Err(RuntimeError::Panic("LeFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Bool(va <= vb));
            }

            // === Type conversions ===
            IntToFloat(dst, src) => {
                let result = match reg!(src) {
                    GcValue::Int64(v) => GcValue::Float64(v as f64),
                    GcValue::Int32(v) => GcValue::Float64(v as f64),
                    GcValue::Float64(v) => GcValue::Float64(v),
                    GcValue::Float32(v) => GcValue::Float64(v as f64),
                    _ => return Err(RuntimeError::Panic("IntToFloat: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            FloatToInt(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::Int64(*v as i64),
                    GcValue::Int16(v) => GcValue::Int64(*v as i64),
                    GcValue::Int32(v) => GcValue::Int64(*v as i64),
                    GcValue::Int64(v) => GcValue::Int64(*v),
                    GcValue::UInt8(v) => GcValue::Int64(*v as i64),
                    GcValue::UInt16(v) => GcValue::Int64(*v as i64),
                    GcValue::UInt32(v) => GcValue::Int64(*v as i64),
                    GcValue::UInt64(v) => GcValue::Int64(*v as i64),
                    GcValue::Float32(v) => GcValue::Int64(*v as i64),
                    GcValue::Float64(v) => GcValue::Int64(*v as i64),
                    GcValue::BigInt(ptr) => {
                        let bi = self.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        if let Some(i) = bi.value.to_i64() {
                            GcValue::Int64(i)
                        } else {
                            return Err(RuntimeError::Panic("BigInt too large for Int64".into()));
                        }
                    }
                    GcValue::Decimal(d) => GcValue::Int64(d.to_string().parse::<f64>().unwrap_or(0.0) as i64),
                    _ => return Err(RuntimeError::Panic("FloatToInt: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToBigInt(dst, src) => {
                let result = match reg!(src) {
                    GcValue::Int64(v) => {
                        let bi = num_bigint::BigInt::from(v);
                        GcValue::BigInt(self.heap.alloc_bigint(bi))
                    }
                    GcValue::Int32(v) => {
                        let bi = num_bigint::BigInt::from(v);
                        GcValue::BigInt(self.heap.alloc_bigint(bi))
                    }
                    GcValue::BigInt(_) => reg!(src), // Already a BigInt
                    _ => return Err(RuntimeError::Panic("ToBigInt: expected integer".into())),
                };
                set_reg!(dst, result);
            }

            // === Comparisons ===
            EqInt(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("EqInt: expected Int64".into())) };
                let vb = match reg_ref!(b) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("EqInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va == vb));
            }
            LtInt(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("LtInt: expected Int64".into())) };
                let vb = match reg_ref!(b) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("LtInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va < vb));
            }
            LeInt(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("LeInt: expected Int64".into())) };
                let vb = match reg_ref!(b) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("LeInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va <= vb));
            }
            GtInt(dst, a, b) => {
                let va = match reg_ref!(a) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("GtInt: expected Int64".into())) };
                let vb = match reg_ref!(b) { GcValue::Int64(n) => *n, _ => return Err(RuntimeError::Panic("GtInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va > vb));
            }

            // === Control flow ===
            // Note: Jump offsets are RELATIVE - formula is: original_ip + 1 + offset
            // Since we already incremented ip before this match, the current ip is (original_ip + 1)
            // So we need: current_ip + offset = (original_ip + 1) + offset
            Jump(offset) => {
                // SAFETY: cur_frame is valid
                let frame = unsafe { self.frames.get_unchecked_mut(cur_frame) };
                // frame.ip was already incremented, so current ip = original_ip + 1
                // target = original_ip + 1 + offset = current_ip + offset
                frame.ip = (frame.ip as isize + *offset as isize) as usize;
            }
            JumpIfTrue(cond, offset) => {
                if matches!(reg_ref!(cond), GcValue::Bool(true)) {
                    // SAFETY: cur_frame is valid
                    let frame = unsafe { self.frames.get_unchecked_mut(cur_frame) };
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                }
            }
            JumpIfFalse(cond, offset) => {
                if matches!(reg_ref!(cond), GcValue::Bool(false)) {
                    // SAFETY: cur_frame is valid
                    let frame = unsafe { self.frames.get_unchecked_mut(cur_frame) };
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                }
            }

            // === Return ===
            Return(src) => {
                // Record function exit for profiling
                self.profile_exit();
                // SAFETY: cur_frame is valid
                let cur = unsafe { self.frames.get_unchecked_mut(cur_frame) };
                // Get return_reg BEFORE taking value (avoid aliasing)
                let return_reg = cur.return_reg;
                // Take ownership of return value (avoid clone) - frame is about to be destroyed
                let value = std::mem::take(unsafe { cur.registers.get_unchecked_mut(src.as_idx()) });
                // Pop frame and recycle its registers
                if let Some(frame) = self.frames.pop() {
                    self.free_registers(frame.registers);
                }
                if self.frames.is_empty() {
                    return Ok(StepResult::Finished(value));
                } else if let Some(ret_reg) = return_reg {
                    // Store return value in caller's return register
                    // SAFETY: After pop, if frames is not empty, last frame exists
                    let frame = self.frames.last_mut().unwrap();
                    frame.registers[ret_reg as usize] = value;
                }
            }

            // === Function calls ===
            CallDirect(dst, func_idx, ref args) => {
                // Check for JIT-compiled version first based on arity
                let func_idx_u16 = *func_idx;
                let profiling = self.is_profiling();
                match args.len() {
                    0 => {
                        let jit_fn_opt = self.shared.jit_int_functions_0.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            let (result, duration) = if profiling {
                                let start = Instant::now();
                                let r = jit_fn();
                                (r, Some(start.elapsed()))
                            } else {
                                (jit_fn(), None)
                            };
                            set_reg!(dst, GcValue::Int64(result));
                            if let Some(d) = duration {
                                let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                self.profile_jit_call(&name, d);
                            }
                            return Ok(StepResult::Continue);
                        }
                    }
                    1 => {
                        // Pure numeric JIT
                        let jit_fn_opt = self.shared.jit_int_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let GcValue::Int64(n) = reg!(args[0]) {
                                let (result, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(n);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(n), None)
                                };
                                set_reg!(dst, GcValue::Int64(result));
                                if let Some(d) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, d);
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Loop array JIT
                        let jit_fn_opt = self.shared.jit_loop_array_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let GcValue::Int64Array(arr_ptr) = reg!(args[0]) {
                                if let Some(arr) = self.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let (result, duration) = if profiling {
                                        let start = Instant::now();
                                        let r = jit_fn(ptr as *const i64, len);
                                        (r, Some(start.elapsed()))
                                    } else {
                                        (jit_fn(ptr as *const i64, len), None)
                                    };
                                    set_reg!(dst, GcValue::Int64(result));
                                    if let Some(d) = duration {
                                        let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                        self.profile_jit_call(&name, d);
                                    }
                                    return Ok(StepResult::Continue);
                                }
                            }
                        }
                    }
                    2 => {
                        // Try numeric JIT first
                        let jit_fn_opt = self.shared.jit_int_functions_2.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(a), GcValue::Int64(b)) = (reg!(args[0]), reg!(args[1])) {
                                let (result, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(a, b);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(a, b), None)
                                };
                                set_reg!(dst, GcValue::Int64(result));
                                if let Some(d) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, d);
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Recursive array fill JIT: (arr, idx)
                        let jit_fn_opt = self.shared.jit_array_fill_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let GcValue::Int64(idx) = reg!(args[1]) {
                                if let GcValue::Int64Array(arr_ptr) = reg!(args[0]) {
                                    if let Some(arr) = self.heap.get_int64_array_mut(arr_ptr) {
                                        let ptr = arr.items.as_mut_ptr();
                                        let len = arr.items.len() as i64;
                                        let (result, duration) = if profiling {
                                            let start = Instant::now();
                                            let r = jit_fn(ptr as *const i64, len, idx);
                                            (r, Some(start.elapsed()))
                                        } else {
                                            (jit_fn(ptr as *const i64, len, idx), None)
                                        };
                                        // Returns unit, but function may modify array in place
                                        let _ = result;
                                        set_reg!(dst, GcValue::Unit);
                                        if let Some(d) = duration {
                                            let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                            self.profile_jit_call(&name, d);
                                        }
                                        return Ok(StepResult::Continue);
                                    }
                                }
                            }
                        }
                    }
                    3 => {
                        // Try numeric JIT first
                        let jit_fn_opt = self.shared.jit_int_functions_3.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c)) =
                                (reg!(args[0]), reg!(args[1]), reg!(args[2])) {
                                let (result, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(a, b, c);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(a, b, c), None)
                                };
                                set_reg!(dst, GcValue::Int64(result));
                                if let Some(d) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, d);
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Recursive array sum JIT: (arr, idx, acc)
                        let jit_fn_opt = self.shared.jit_array_sum_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(idx), GcValue::Int64(acc)) = (reg!(args[1]), reg!(args[2])) {
                                if let GcValue::Int64Array(arr_ptr) = reg!(args[0]) {
                                    if let Some(arr) = self.heap.get_int64_array_mut(arr_ptr) {
                                        let ptr = arr.items.as_mut_ptr();
                                        let len = arr.items.len() as i64;
                                        let (result, duration) = if profiling {
                                            let start = Instant::now();
                                            let r = jit_fn(ptr as *const i64, len, idx, acc);
                                            (r, Some(start.elapsed()))
                                        } else {
                                            (jit_fn(ptr as *const i64, len, idx, acc), None)
                                        };
                                        set_reg!(dst, GcValue::Int64(result));
                                        if let Some(d) = duration {
                                            let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                            self.profile_jit_call(&name, d);
                                        }
                                        return Ok(StepResult::Continue);
                                    }
                                }
                            }
                        }
                    }
                    4 => {
                        let jit_fn_opt = self.shared.jit_int_functions_4.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c), GcValue::Int64(d)) =
                                (reg!(args[0]), reg!(args[1]), reg!(args[2]), reg!(args[3])) {
                                let (result, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(a, b, c, d);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(a, b, c, d), None)
                                };
                                set_reg!(dst, GcValue::Int64(result));
                                if let Some(dur) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, dur);
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    _ => {}
                }

                // Fall back to interpreter
                let function = {
                    self.shared.function_list.read().unwrap().get(*func_idx as usize)
                        .ok_or_else(|| RuntimeError::Panic(format!("Unknown function index: {}", func_idx)))?
                        .clone()
                };

                // Get registers from pool or allocate new
                let mut registers = self.alloc_registers(function.code.register_count as usize);
                for (i, r) in args.iter().enumerate() {
                    if i < registers.len() {
                        registers[i] = reg!(*r);
                    }
                }

                // Record function entry for profiling
                self.profile_enter(&function.name);

                self.frames.push(CallFrame {
                    function,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(*dst),
                });
            }

            // Call function/closure stored in a register
            Call(dst, func_reg, ref args) => {
                let callee = reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();

                // Fast path for binary operations - check InlineOp first
                if arg_values.len() == 2 {
                    use crate::gc::InlineOp;
                    let inline_op = match &callee {
                        GcValue::Closure(_, op) => *op,
                        GcValue::Function(func) => InlineOp::from_function(func),
                        _ => InlineOp::None,
                    };
                    if inline_op != InlineOp::None {
                        if let (GcValue::Int64(x), GcValue::Int64(y)) = (&arg_values[0], &arg_values[1]) {
                            let result = match inline_op {
                                InlineOp::AddInt => x + y,
                                InlineOp::SubInt => x - y,
                                InlineOp::MulInt => x * y,
                                InlineOp::None => unreachable!(),
                            };
                            set_reg!(dst, GcValue::Int64(result));
                            return Ok(StepResult::Continue);
                        }
                    }
                }

                match callee {
                    GcValue::Function(func) => {
                        // Regular function call
                        let mut registers = self.alloc_registers(func.code.register_count as usize);
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            if i < registers.len() {
                                registers[i] = arg;
                            }
                        }
                        // Record function entry for profiling
                        self.profile_enter(&func.name);
                        self.frames.push(CallFrame {
                            function: func,
                            ip: 0,
                            registers,
                            captures: Vec::new(),
                            return_reg: Some(*dst),
                        });
                    }
                    GcValue::Closure(ptr, _inline_op) => {
                        // Regular closure call (fast path already checked above)
                        let closure = self.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid closure reference".into()))?;
                        let func = closure.function.clone();
                        let captures = closure.captures.clone();

                        let mut registers = self.alloc_registers(func.code.register_count as usize);
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            if i < registers.len() {
                                registers[i] = arg;
                            }
                        }

                        // Record closure entry for profiling
                        self.profile_enter(&func.name);
                        self.frames.push(CallFrame {
                            function: func,
                            ip: 0,
                            registers,
                            captures,
                            return_reg: Some(*dst),
                        });
                    }
                    _ => return Err(RuntimeError::Panic(format!("Call: expected function or closure, got {:?}", callee))),
                }
            }

            // === Process operations (async!) ===
            SelfPid(dst) => {
                set_reg!(dst, GcValue::Pid(self.pid.0));
            }

            ProcessAll(dst) => {
                // Get all registered process PIDs
                let registry = self.shared.process_registry.read().await;
                let pids: Vec<GcValue> = registry.keys()
                    .map(|pid| GcValue::Pid(pid.0))
                    .collect();
                drop(registry);
                let list = GcValue::List(crate::gc::GcList::from_vec(pids));
                set_reg!(dst, list);
            }

            ProcessTime(dst, pid_reg) => {
                // Return uptime in milliseconds for the process
                // For async VM, we can only check if process is registered
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessTime: expected Pid".into())),
                };

                let uptime_ms = if target_pid == self.pid {
                    // Own process - use started_at
                    self.started_at.elapsed().as_millis() as i64
                } else {
                    // Other process - we don't have direct access to their start time
                    // Return -1 for processes we can't query (matches parallel VM behavior)
                    -1
                };
                set_reg!(dst, GcValue::Int64(uptime_ms));
            }

            ProcessAlive(dst, pid_reg) => {
                // Check if a process is registered (alive)
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessAlive: expected Pid".into())),
                };

                // Check if process is registered in the registry
                let registry = self.shared.process_registry.read().await;
                let alive = registry.contains_key(&target_pid);
                drop(registry);
                set_reg!(dst, GcValue::Bool(alive));
            }

            ProcessInfo(dst, pid_reg) => {
                // Get process info - only works for own process in async VM
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessInfo: expected Pid".into())),
                };

                let result = if target_pid == self.pid {
                    // Own process - return full info
                    let status = match self.state {
                        ProcessState::Running => "running",
                        ProcessState::Waiting => "waiting",
                        ProcessState::WaitingTimeout => "waiting",
                        ProcessState::WaitingIO => "io",
                        ProcessState::WaitingForMvar(_, _) => "waiting",
                        ProcessState::Sleeping => "sleeping",
                        ProcessState::Suspended => "suspended",
                        ProcessState::Exited(_) => "exited",
                    };
                    let mailbox_len = 0i64; // Can't easily query own mailbox length
                    let uptime_ms = self.started_at.elapsed().as_millis() as i64;

                    let status_str = self.heap.alloc_string(status.to_string());
                    let record = self.heap.alloc_record(
                        "ProcessInfo".to_string(),
                        vec!["status".to_string(), "mailbox".to_string(), "uptime".to_string()],
                        vec![GcValue::String(status_str), GcValue::Int64(mailbox_len), GcValue::Int64(uptime_ms)],
                        vec![false, false, false],
                    );
                    GcValue::Record(record)
                } else {
                    // Other process - can't access directly
                    GcValue::Unit
                };
                set_reg!(dst, result);
            }

            ProcessKill(dst, pid_reg) => {
                // Kill a process by unregistering it (its messages will fail)
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessKill: expected Pid".into())),
                };

                // Can't kill self
                let killed = if target_pid == self.pid {
                    false
                } else {
                    // Unregister the process - this effectively kills it
                    // (its mailbox will be dropped, messages will fail)
                    let mut registry = self.shared.process_registry.write().await;
                    registry.remove(&target_pid).is_some()
                };
                set_reg!(dst, GcValue::Bool(killed));
            }

            // === MVar operations (async!) ===
            MvarRead(dst, name_idx) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarRead: expected string constant".into())),
                };
                let gc_value = self.mvar_read(&name).await?;
                set_reg!(dst, gc_value);
                // Yield after mvar access for fairness
                tokio::task::yield_now().await;
            }

            MvarWrite(name_idx, src) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarWrite: expected string constant".into())),
                };
                let value = reg!(src);
                self.mvar_write(&name, value).await?;
                // Yield after mvar access for fairness
                tokio::task::yield_now().await;
            }

            MvarLock(name_idx, is_write) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarLock: expected string constant".into())),
                };

                // Check if we already hold this lock (reentrant locking)
                if self.held_mvar_locks.contains_key(&name) {
                    // Already holding the lock - just increment depth
                    let depth = self.mvar_lock_depths.entry(name.clone()).or_insert(1);
                    *depth += 1;
                } else {
                    // Get the mvar
                    let var = self.shared.mvars.get(&name)
                        .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?
                        .clone();

                    // Acquire lock (async - will yield if contended)
                    let guard = if *is_write {
                        HeldMvarLock::Write(var.write_owned().await)
                    } else {
                        HeldMvarLock::Read(var.read_owned().await)
                    };

                    // Store the guard and set initial depth
                    self.held_mvar_locks.insert(name.clone(), guard);
                    self.mvar_lock_depths.insert(name, 1);
                    // Yield after acquiring lock to let other tasks notice
                    tokio::task::yield_now().await;
                }
            }

            MvarUnlock(name_idx, _is_write) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarUnlock: expected string constant".into())),
                };

                // Decrement depth and release lock if depth reaches 0
                let depth = self.mvar_lock_depths.get_mut(&name)
                    .ok_or_else(|| RuntimeError::Panic(format!("MvarUnlock: no lock held on mvar: {}", name)))?;

                *depth -= 1;
                if *depth == 0 {
                    // Release the actual lock
                    self.mvar_lock_depths.remove(&name);
                    self.held_mvar_locks.remove(&name);
                    // Force multiple yields to give waiting tasks time to wake up and acquire
                    // This helps prevent starvation under high contention (tokio RwLock is not fair)
                    self.instructions_since_yield = REDUCTIONS_PER_YIELD;
                    tokio::task::yield_now().await;
                    tokio::task::yield_now().await;
                    tokio::task::yield_now().await;
                }
            }

            // === Print ===
            Print(_dst, src) => {
                let value = reg!(src);
                let s = self.heap.display_value(&value);
                self.output.push(s.clone());
                // Send to output channel if available (for REPL/TUI)
                if let Some(ref sender) = self.shared.output_sender {
                    let _ = sender.send(s.clone());
                } else {
                    println!("{}", s);
                }
            }

            Println(src) => {
                let value = reg!(src);
                let s = self.heap.display_value(&value);
                self.output.push(s.clone());
                // Send to output channel if available (for REPL/TUI)
                if let Some(ref sender) = self.shared.output_sender {
                    let _ = sender.send(s.clone());
                } else {
                    println!("{}", s);
                }
            }

            // === Tail call (replaces current frame) ===
            TailCallDirect(func_idx, ref args) => {
                // Check for JIT-compiled version first
                let func_idx_u16 = *func_idx;
                let profiling = self.is_profiling();
                match args.len() {
                    0 => {
                        let jit_fn_opt = self.shared.jit_int_functions_0.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            let (res, duration) = if profiling {
                                let start = Instant::now();
                                let r = jit_fn();
                                (r, Some(start.elapsed()))
                            } else {
                                (jit_fn(), None)
                            };
                            if let Some(d) = duration {
                                let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                self.profile_jit_call(&name, d);
                            }
                            let result = GcValue::Int64(res);
                            // Pop frame and return result
                            let return_reg = self.frames.last().unwrap().return_reg;
                            self.frames.pop();
                            if self.frames.is_empty() {
                                return Ok(StepResult::Finished(result));
                            } else if let Some(ret_reg) = return_reg {
                                let frame = self.frames.last_mut().unwrap();
                                frame.registers[ret_reg as usize] = result;
                            }
                            return Ok(StepResult::Continue);
                        }
                    }
                    1 => {
                        // Pure numeric JIT
                        let jit_fn_opt = self.shared.jit_int_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let GcValue::Int64(n) = reg!(args[0]) {
                                let (res, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(n);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(n), None)
                                };
                                if let Some(d) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, d);
                                }
                                let result = GcValue::Int64(res);
                                let return_reg = self.frames.last().unwrap().return_reg;
                                self.frames.pop();
                                if self.frames.is_empty() {
                                    return Ok(StepResult::Finished(result));
                                } else if let Some(ret_reg) = return_reg {
                                    let frame = self.frames.last_mut().unwrap();
                                    frame.registers[ret_reg as usize] = result;
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Loop array JIT
                        let jit_fn_opt = self.shared.jit_loop_array_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let GcValue::Int64Array(arr_ptr) = reg!(args[0]) {
                                if let Some(arr) = self.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let (res, duration) = if profiling {
                                        let start = Instant::now();
                                        let r = jit_fn(ptr as *const i64, len);
                                        (r, Some(start.elapsed()))
                                    } else {
                                        (jit_fn(ptr as *const i64, len), None)
                                    };
                                    if let Some(d) = duration {
                                        let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                        self.profile_jit_call(&name, d);
                                    }
                                    let result = GcValue::Int64(res);
                                    let return_reg = self.frames.last().unwrap().return_reg;
                                    self.frames.pop();
                                    if self.frames.is_empty() {
                                        return Ok(StepResult::Finished(result));
                                    } else if let Some(ret_reg) = return_reg {
                                        let frame = self.frames.last_mut().unwrap();
                                        frame.registers[ret_reg as usize] = result;
                                    }
                                    return Ok(StepResult::Continue);
                                }
                            }
                        }
                    }
                    2 => {
                        // Try numeric JIT first
                        let jit_fn_opt = self.shared.jit_int_functions_2.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(a), GcValue::Int64(b)) = (reg!(args[0]), reg!(args[1])) {
                                let (res, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(a, b);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(a, b), None)
                                };
                                if let Some(d) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, d);
                                }
                                let result = GcValue::Int64(res);
                                let return_reg = self.frames.last().unwrap().return_reg;
                                self.frames.pop();
                                if self.frames.is_empty() {
                                    return Ok(StepResult::Finished(result));
                                } else if let Some(ret_reg) = return_reg {
                                    let frame = self.frames.last_mut().unwrap();
                                    frame.registers[ret_reg as usize] = result;
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Recursive array fill JIT: (arr, idx)
                        let jit_fn_opt = self.shared.jit_array_fill_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let GcValue::Int64(idx) = reg!(args[1]) {
                                if let GcValue::Int64Array(arr_ptr) = reg!(args[0]) {
                                    if let Some(arr) = self.heap.get_int64_array_mut(arr_ptr) {
                                        let ptr = arr.items.as_mut_ptr();
                                        let len = arr.items.len() as i64;
                                        let (res, duration) = if profiling {
                                            let start = Instant::now();
                                            let r = jit_fn(ptr as *const i64, len, idx);
                                            (r, Some(start.elapsed()))
                                        } else {
                                            (jit_fn(ptr as *const i64, len, idx), None)
                                        };
                                        let _ = res;
                                        if let Some(d) = duration {
                                            let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                            self.profile_jit_call(&name, d);
                                        }
                                        let result = GcValue::Unit;
                                        let return_reg = self.frames.last().unwrap().return_reg;
                                        self.frames.pop();
                                        if self.frames.is_empty() {
                                            return Ok(StepResult::Finished(result));
                                        } else if let Some(ret_reg) = return_reg {
                                            let frame = self.frames.last_mut().unwrap();
                                            frame.registers[ret_reg as usize] = result;
                                        }
                                        return Ok(StepResult::Continue);
                                    }
                                }
                            }
                        }
                    }
                    3 => {
                        // Try numeric JIT first
                        let jit_fn_opt = self.shared.jit_int_functions_3.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c)) =
                                (reg!(args[0]), reg!(args[1]), reg!(args[2])) {
                                let (res, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = jit_fn(a, b, c);
                                    (r, Some(start.elapsed()))
                                } else {
                                    (jit_fn(a, b, c), None)
                                };
                                if let Some(d) = duration {
                                    let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                    self.profile_jit_call(&name, d);
                                }
                                let result = GcValue::Int64(res);
                                let return_reg = self.frames.last().unwrap().return_reg;
                                self.frames.pop();
                                if self.frames.is_empty() {
                                    return Ok(StepResult::Finished(result));
                                } else if let Some(ret_reg) = return_reg {
                                    let frame = self.frames.last_mut().unwrap();
                                    frame.registers[ret_reg as usize] = result;
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Recursive array sum JIT: (arr, idx, acc)
                        let jit_fn_opt = self.shared.jit_array_sum_functions.read().unwrap().get(&func_idx_u16).copied();
                        if let Some(jit_fn) = jit_fn_opt {
                            if let (GcValue::Int64(idx), GcValue::Int64(acc)) = (reg!(args[1]), reg!(args[2])) {
                                if let GcValue::Int64Array(arr_ptr) = reg!(args[0]) {
                                    if let Some(arr) = self.heap.get_int64_array_mut(arr_ptr) {
                                        let ptr = arr.items.as_mut_ptr();
                                        let len = arr.items.len() as i64;
                                        let (res, duration) = if profiling {
                                            let start = Instant::now();
                                            let r = jit_fn(ptr as *const i64, len, idx, acc);
                                            (r, Some(start.elapsed()))
                                        } else {
                                            (jit_fn(ptr as *const i64, len, idx, acc), None)
                                        };
                                        if let Some(d) = duration {
                                            let name = self.shared.function_list.read().unwrap().get(*func_idx as usize).map(|f| f.name.clone()).unwrap_or_else(|| "unknown".to_string());
                                            self.profile_jit_call(&name, d);
                                        }
                                        let result = GcValue::Int64(res);
                                        let return_reg = self.frames.last().unwrap().return_reg;
                                        self.frames.pop();
                                        if self.frames.is_empty() {
                                            return Ok(StepResult::Finished(result));
                                        } else if let Some(ret_reg) = return_reg {
                                            let frame = self.frames.last_mut().unwrap();
                                            frame.registers[ret_reg as usize] = result;
                                        }
                                        return Ok(StepResult::Continue);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }

                // Fall back to interpreter
                let function = {
                    self.shared.function_list.read().unwrap().get(*func_idx as usize)
                        .ok_or_else(|| RuntimeError::Panic(format!("Unknown function index: {}", func_idx)))?
                        .clone()
                };

                // OPTIMIZATION: If calling same function, reuse registers (no heap allocation!)
                // This is critical for recursive functions like fold
                let current_func = &self.frames.last().unwrap().function;
                if std::sync::Arc::ptr_eq(&function, current_func) && args.len() <= 8 {
                    // Same function - reuse registers, no allocation!
                    let mut saved_args: [std::mem::MaybeUninit<GcValue>; 8] =
                        unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                    // Save args to stack (take ownership, leave Unit behind)
                    for (i, &r) in args.iter().enumerate() {
                        saved_args[i] = std::mem::MaybeUninit::new(
                            std::mem::take(&mut self.frames.last_mut().unwrap().registers[r as usize])
                        );
                    }

                    let frame = self.frames.last_mut().unwrap();

                    // Clear all registers to Unit (in-place, no allocation)
                    for reg in frame.registers.iter_mut() {
                        *reg = GcValue::Unit;
                    }

                    // Write back saved args to parameter positions
                    for (i, _) in args.iter().enumerate() {
                        frame.registers[i] = unsafe { saved_args[i].assume_init_read() };
                    }

                    frame.ip = 0;
                    frame.captures.clear();
                } else {
                    // Different function or >8 args - need to set up new frame
                    let function = function.clone();
                    let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                    let mut registers = vec![GcValue::Unit; function.code.register_count as usize];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < registers.len() {
                            registers[i] = arg;
                        }
                    }

                    // Tail call: replace current frame instead of pushing new one
                    let frame = self.frames.last_mut().unwrap();
                    frame.function = function;
                    frame.ip = 0;
                    frame.registers = registers;
                    frame.captures.clear();
                }
                // Note: return_reg stays the same since we're replacing this call
            }

            GeInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("GeInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("GeInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va >= vb));
            }

            // === Boolean operations ===
            And(dst, a, b) => {
                let va = match reg!(a) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("And: expected Bool".into())) };
                let vb = match reg!(b) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("And: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(va && vb));
            }
            Or(dst, a, b) => {
                let va = match reg!(a) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("Or: expected Bool".into())) };
                let vb = match reg!(b) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("Or: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(va || vb));
            }
            Not(dst, src) => {
                let v = match reg!(src) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("Not: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(!v));
            }

            // === Generic equality ===
            Eq(dst, a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                let equal = self.heap.gc_values_equal(&va, &vb);
                set_reg!(dst, GcValue::Bool(equal));
            }

            // === Pattern matching ===
            TestConst(dst, value_reg, const_idx) => {
                let constant = get_const!(const_idx);
                let value = reg!(value_reg);
                let gc_const = self.heap.value_to_gc(&constant);
                let result = self.heap.gc_values_equal(&value, &gc_const);
                set_reg!(dst, GcValue::Bool(result));
            }

            TestNil(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            TestUnit(dst, value_reg) => {
                let val = reg!(value_reg);
                let result = matches!(val, GcValue::Unit);
                set_reg!(dst, GcValue::Bool(result));
            }

            TestTag(dst, value_reg, ctor_idx) => {
                let expected_ctor = match get_const!(ctor_idx) {
                    Value::String(s) => s.clone(),
                    _ => return Err(RuntimeError::Panic("TestTag: expected string constant".into())),
                };
                let val = reg!(value_reg);
                let result = match val {
                    GcValue::Variant(ptr) => {
                        if let Some(var) = self.heap.get_variant(ptr) {
                            var.constructor.as_str() == expected_ctor.as_str()
                        } else {
                            false
                        }
                    }
                    GcValue::Record(ptr) => {
                        // Records: compare type_name directly
                        if let Some(rec) = self.heap.get_record(ptr) {
                            rec.type_name.as_str() == expected_ctor.as_str()
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            // === Self-recursive calls ===
            CallSelf(dst, ref args) => {
                // Call the current function recursively
                // SAFETY: cur_frame is valid
                let func = unsafe { self.frames.get_unchecked(cur_frame).function.clone() };
                // Get registers from pool
                let mut registers = self.alloc_registers(func.code.register_count as usize);
                for (i, r) in args.iter().enumerate() {
                    if i < registers.len() {
                        registers[i] = reg!(*r);
                    }
                }
                // Record function entry for profiling
                self.profile_enter(&func.name);
                self.frames.push(CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(*dst),
                });
            }
            TailCallSelf(ref args) => {
                // OPTIMIZATION: Reuse registers instead of allocating new Vec
                // Use stack array to save args before overwriting (avoids heap allocation)
                if args.len() <= 8 {
                    let mut saved_args: [std::mem::MaybeUninit<GcValue>; 8] =
                        unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                    // Save args to stack (take ownership, leave Unit behind)
                    for (i, &r) in args.iter().enumerate() {
                        saved_args[i] = std::mem::MaybeUninit::new(
                            std::mem::take(&mut self.frames.last_mut().unwrap().registers[r as usize])
                        );
                    }

                    let frame = self.frames.last_mut().unwrap();

                    // Clear all registers to Unit (in-place, no allocation)
                    for reg in frame.registers.iter_mut() {
                        *reg = GcValue::Unit;
                    }

                    // Write back saved args to parameter positions
                    for (i, _) in args.iter().enumerate() {
                        frame.registers[i] = unsafe { saved_args[i].assume_init_read() };
                    }

                    frame.ip = 0;
                    // NOTE: Do NOT clear captures - closures calling themselves recursively
                    // still need their captured variables!
                } else {
                    // Fallback for >8 args (rare)
                    let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                    let frame = self.frames.last_mut().unwrap();
                    for reg in frame.registers.iter_mut() {
                        *reg = GcValue::Unit;
                    }
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < frame.registers.len() {
                            frame.registers[i] = arg;
                        }
                    }
                    frame.ip = 0;
                    // NOTE: Do NOT clear captures - closures need them!
                }
            }

            // === Assertions ===
            Assert(src) => {
                let val = reg!(src);
                match val {
                    GcValue::Bool(true) => {}
                    GcValue::Bool(false) => {
                        return Err(RuntimeError::Panic("Assertion failed".into()));
                    }
                    _ => {
                        return Err(RuntimeError::Panic(format!("Assert: expected Bool, got {:?}", val)));
                    }
                }
            }

            AssertEq(a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                if !self.heap.gc_values_equal(&va, &vb) {
                    let sa = self.heap.display_value(&va);
                    let sb = self.heap.display_value(&vb);
                    return Err(RuntimeError::Panic(format!("Assertion failed: {} != {}", sa, sb)));
                }
            }

            // === Native function calls ===
            CallNative(dst, name_idx, ref args) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("CallNative: expected string constant".into())),
                };
                let native = self.shared.natives.get(&name)
                    .ok_or_else(|| RuntimeError::Panic(format!("Unknown native function: {}", name)))?
                    .clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                let result = (native.func)(&arg_values, &mut self.heap)?;
                set_reg!(dst, result);
            }

            // === Closures ===
            MakeClosure(dst, func_idx, ref captures) => {
                let function = match get_const!(func_idx) {
                    Value::Function(f) => f,
                    _ => return Err(RuntimeError::Panic(format!("MakeClosure: expected function at constant index {}", func_idx))),
                };
                let capture_values: Vec<GcValue> = captures.iter().map(|r| reg!(*r)).collect();
                let capture_names = function.param_names.clone();
                // Compute InlineOp for fast binary closure calls
                let inline_op = crate::gc::InlineOp::from_function(&function);
                let closure = self.heap.alloc_closure(function, capture_values, capture_names);
                set_reg!(dst, GcValue::Closure(closure, inline_op));
            }

            GetCapture(dst, idx) => {
                let frame = self.frames.last().unwrap();
                if (*idx as usize) < frame.captures.len() {
                    let value = frame.captures[*idx as usize].clone();
                    set_reg!(dst, value);
                } else {
                    return Err(RuntimeError::Panic(format!("Capture index {} out of bounds (frame has {} captures)",
                                                           idx, frame.captures.len())));
                }
            }

            // === Data structures ===
            MakeTuple(dst, ref elems) => {
                let values: Vec<GcValue> = elems.iter().map(|r| reg!(*r)).collect();
                let tuple = self.heap.alloc_tuple(values);
                set_reg!(dst, GcValue::Tuple(tuple));
            }

            GetTupleField(dst, tuple_reg, idx) => {
                let tuple = reg!(tuple_reg);
                if let GcValue::Tuple(ptr) = tuple {
                    if let Some(t) = self.heap.get_tuple(ptr) {
                        if (*idx as usize) < t.items.len() {
                            let value = t.items[*idx as usize].clone();
                            set_reg!(dst, value);
                        } else {
                            return Err(RuntimeError::Panic("Tuple index out of bounds".into()));
                        }
                    } else {
                        return Err(RuntimeError::Panic("Invalid tuple pointer".into()));
                    }
                } else {
                    return Err(RuntimeError::Panic("Expected tuple".into()));
                }
            }

            GetField(dst, record, field_idx) => {
                let field_name = match get_const!(field_idx) {
                    Value::String(s) => (*s).clone(),
                    _ => return Err(RuntimeError::Panic("GetField: field name must be string".into())),
                };
                let rec_val = reg!(record);
                match rec_val {
                    GcValue::Record(ptr) => {
                        let rec = self.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".into()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        let value = rec.fields[idx].clone();
                        set_reg!(dst, value);
                    }
                    GcValue::Variant(ptr) => {
                        let var = self.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".into()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid variant field index: {}", field_name)))?;
                        let value = var.fields.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of range", idx)))?
                            .clone();
                        set_reg!(dst, value);
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid tuple index: {}", field_name)))?;
                        let value = tuple.items.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Tuple index {} out of bounds", idx)))?
                            .clone();
                        set_reg!(dst, value);
                    }
                    _ => return Err(RuntimeError::Panic("GetField expects record, variant, or tuple".into())),
                }
            }

            MakeList(dst, ref elems) => {
                let values: Vec<GcValue> = elems.iter().map(|r| reg!(*r)).collect();
                let list = self.heap.make_list(values);
                set_reg!(dst, GcValue::List(list));
            }

            Cons(dst, head, tail) => {
                let head_val = reg!(head);
                let tail_val = reg!(tail);
                if let GcValue::List(tail_list) = tail_val {
                    // O(log n) cons using persistent data structure
                    let new_list = tail_list.cons(head_val);
                    set_reg!(dst, GcValue::List(new_list));
                } else {
                    return Err(RuntimeError::Panic("Cons: tail must be a list".into()));
                }
            }

            Decons(head_dst, tail_dst, list_reg) => {
                let list_val = reg!(list_reg);
                if let GcValue::List(list) = list_val {
                    if !list.is_empty() {
                        let head = list.head().cloned().unwrap_or(GcValue::Unit);
                        let tail = list.tail();
                        set_reg!(head_dst, head);
                        set_reg!(tail_dst, GcValue::List(tail));
                    } else {
                        return Err(RuntimeError::Panic("Decons: empty list".into()));
                    }
                } else {
                    return Err(RuntimeError::Panic("Decons: expected list".into()));
                }
            }

            // Native range list creation - [1..n]
            RangeList(dst, n_reg) => {
                let n = match reg!(n_reg) {
                    GcValue::Int64(n) => n,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "non-integer".to_string(),
                    }),
                };
                let items: Vec<GcValue> = (1..=n).map(|i| GcValue::Int64(i)).collect();
                let list = self.heap.make_list(items);
                set_reg!(dst, GcValue::List(list));
            }

            ListHead(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => {
                        if let Some(head) = list.items().first() {
                            head.clone()
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: format!("{:?}", list_val),
                    }),
                };
                set_reg!(dst, result);
            }

            ListTail(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => {
                        if !list.is_empty() {
                            GcValue::List(list.tail())
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: format!("{:?}", list_val),
                    }),
                };
                set_reg!(dst, result);
            }

            Length(dst, src) => {
                let val = reg!(src);
                let len = match val {
                    GcValue::List(list) => list.len() as i64,
                    GcValue::String(s) => {
                        if let Some(str_val) = self.heap.get_string(s) {
                            str_val.data.chars().count() as i64
                        } else {
                            0
                        }
                    }
                    GcValue::Tuple(ptr) => {
                        if let Some(t) = self.heap.get_tuple(ptr) {
                            t.items.len() as i64
                        } else {
                            0
                        }
                    }
                    GcValue::Int64Array(ptr) => {
                        self.heap.get_int64_array(ptr).map(|a| a.items.len() as i64).unwrap_or(0)
                    }
                    GcValue::Float64Array(ptr) => {
                        self.heap.get_float64_array(ptr).map(|a| a.items.len() as i64).unwrap_or(0)
                    }
                    _ => return Err(RuntimeError::Panic("Length: unsupported type".into())),
                };
                set_reg!(dst, GcValue::Int64(len));
            }

            // === Typed arrays ===
            MakeInt64Array(dst, size_reg) => {
                let size = match reg!(size_reg) {
                    GcValue::Int64(n) => n as usize,
                    _ => return Err(RuntimeError::Panic("MakeInt64Array: size must be Int64".into())),
                };
                let ptr = self.heap.alloc_int64_array(vec![0i64; size]);
                set_reg!(dst, GcValue::Int64Array(ptr));
            }

            MakeFloat64Array(dst, size_reg) => {
                let size = match reg!(size_reg) {
                    GcValue::Int64(n) => n as usize,
                    _ => return Err(RuntimeError::Panic("MakeFloat64Array: size must be Int64".into())),
                };
                let ptr = self.heap.alloc_float64_array(vec![0.0f64; size]);
                set_reg!(dst, GcValue::Float64Array(ptr));
            }

            Index(dst, coll, idx) => {
                let idx_val = match reg!(idx) {
                    GcValue::Int64(i) => i as usize,
                    _ => return Err(RuntimeError::Panic("Index: index must be Int64".into())),
                };
                let coll_val = reg!(coll);
                let value = match coll_val {
                    GcValue::List(list) => {
                        list.items().get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        tuple.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Array(ptr) => {
                        let array = self.heap.get_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".into()))?;
                        array.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Int64Array(ptr) => {
                        let array = self.heap.get_int64_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".into()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Int64(val)
                    }
                    GcValue::Float64Array(ptr) => {
                        let array = self.heap.get_float64_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".into()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Float64(val)
                    }
                    _ => return Err(RuntimeError::Panic("Index expects list, tuple, or array".into())),
                };
                set_reg!(dst, value);
            }

            IndexSet(coll, idx, val) => {
                let idx_val = match reg!(idx) {
                    GcValue::Int64(i) => i as usize,
                    _ => return Err(RuntimeError::Panic("IndexSet: index must be Int64".into())),
                };
                let coll_val = reg!(coll);
                match coll_val {
                    GcValue::Array(ptr) => {
                        let new_value = reg!(val);
                        let array = self.heap.get_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".into()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Int64Array(ptr) => {
                        let new_value = match reg!(val) {
                            GcValue::Int64(v) => v,
                            _ => return Err(RuntimeError::Panic("Int64Array expects Int64 value".into())),
                        };
                        let array = self.heap.get_int64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".into()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Float64Array(ptr) => {
                        let new_value = match reg!(val) {
                            GcValue::Float64(v) => v,
                            _ => return Err(RuntimeError::Panic("Float64Array expects Float64 value".into())),
                        };
                        let array = self.heap.get_float64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".into()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    _ => return Err(RuntimeError::Panic("IndexSet expects array".into())),
                }
            }

            ListIsEmpty(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let is_empty = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(is_empty));
            }

            // === String operations ===
            Concat(dst, a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                match (&va, &vb) {
                    (GcValue::String(sa), GcValue::String(sb)) => {
                        let str_a = self.heap.get_string(*sa).map(|x| x.data.clone()).unwrap_or_default();
                        let str_b = self.heap.get_string(*sb).map(|x| x.data.clone()).unwrap_or_default();
                        let result = format!("{}{}", str_a, str_b);
                        set_reg!(dst, GcValue::String(self.heap.alloc_string(result)));
                    }
                    (GcValue::List(la), GcValue::List(lb)) => {
                        let mut items: Vec<GcValue> = la.items().to_vec();
                        items.extend(lb.items().iter().cloned());
                        let new_list = self.heap.make_list(items);
                        set_reg!(dst, GcValue::List(new_list));
                    }
                    _ => return Err(RuntimeError::Panic("Concat: expected string or list".into())),
                }
            }

            EqStr(dst, a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                let equal = match (&va, &vb) {
                    (GcValue::String(sa), GcValue::String(sb)) => {
                        let str_a = self.heap.get_string(*sa).map(|x| &x.data);
                        let str_b = self.heap.get_string(*sb).map(|x| &x.data);
                        str_a == str_b
                    }
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(equal));
            }

            EqBool(dst, a, b) => {
                let va = match reg!(a) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("EqBool: expected Bool".into())) };
                let vb = match reg!(b) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("EqBool: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(va == vb));
            }

            // === Base64 Encoding ===
            Base64Encode(dst, src) => {
                let input = match reg!(src) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::Panic("Base64Encode: expected String".into())),
                };
                use base64::{Engine as _, engine::general_purpose};
                let encoded = general_purpose::STANDARD.encode(input.as_bytes());
                let ptr = self.heap.alloc_string(encoded);
                set_reg!(dst, GcValue::String(ptr));
            }

            Base64Decode(dst, src) => {
                let input = match reg!(src) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::Panic("Base64Decode: expected String".into())),
                };
                use base64::{Engine as _, engine::general_purpose};
                match general_purpose::STANDARD.decode(input.as_bytes()) {
                    Ok(bytes) => {
                        let decoded = String::from_utf8_lossy(&bytes).to_string();
                        let status_ptr = self.heap.alloc_string("ok".to_string());
                        let decoded_ptr = self.heap.alloc_string(decoded);
                        let tuple_ptr = self.heap.alloc_tuple(vec![
                            GcValue::String(status_ptr),
                            GcValue::String(decoded_ptr),
                        ]);
                        set_reg!(dst, GcValue::Tuple(tuple_ptr));
                    }
                    Err(e) => {
                        let status_ptr = self.heap.alloc_string("error".to_string());
                        let error_ptr = self.heap.alloc_string(format!("invalid base64: {}", e));
                        let tuple_ptr = self.heap.alloc_tuple(vec![
                            GcValue::String(status_ptr),
                            GcValue::String(error_ptr),
                        ]);
                        set_reg!(dst, GcValue::Tuple(tuple_ptr));
                    }
                }
            }

            // === URL Encoding ===
            UrlEncode(dst, src) => {
                let input = match reg!(src) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::Panic("UrlEncode: expected String".into())),
                };
                use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
                let encoded = utf8_percent_encode(&input, NON_ALPHANUMERIC).to_string();
                let ptr = self.heap.alloc_string(encoded);
                set_reg!(dst, GcValue::String(ptr));
            }

            UrlDecode(dst, src) => {
                let input = match reg!(src) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::Panic("UrlDecode: expected String".into())),
                };
                use percent_encoding::percent_decode_str;
                match percent_decode_str(&input).decode_utf8() {
                    Ok(decoded) => {
                        let status_ptr = self.heap.alloc_string("ok".to_string());
                        let decoded_ptr = self.heap.alloc_string(decoded.to_string());
                        let tuple_ptr = self.heap.alloc_tuple(vec![
                            GcValue::String(status_ptr),
                            GcValue::String(decoded_ptr),
                        ]);
                        set_reg!(dst, GcValue::Tuple(tuple_ptr));
                    }
                    Err(e) => {
                        let status_ptr = self.heap.alloc_string("error".to_string());
                        let error_ptr = self.heap.alloc_string(format!("invalid URL encoding: {}", e));
                        let tuple_ptr = self.heap.alloc_tuple(vec![
                            GcValue::String(status_ptr),
                            GcValue::String(error_ptr),
                        ]);
                        set_reg!(dst, GcValue::Tuple(tuple_ptr));
                    }
                }
            }

            // === UTF-8 Encoding ===
            Utf8Encode(dst, src) => {
                // Convert string to list of bytes
                let input = match reg!(src) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::Panic("Utf8Encode: expected String".into())),
                };
                let bytes: Vec<GcValue> = input.as_bytes().iter().map(|&b| GcValue::Int64(b as i64)).collect();
                let list = self.heap.make_list(bytes);
                set_reg!(dst, GcValue::List(list));
            }

            Utf8Decode(dst, src) => {
                // Convert list of bytes to string
                let bytes = match reg!(src) {
                    GcValue::List(list) => {
                        list.items().iter().filter_map(|v| {
                            if let GcValue::Int64(n) = v {
                                Some(*n as u8)
                            } else {
                                None
                            }
                        }).collect::<Vec<u8>>()
                    }
                    _ => return Err(RuntimeError::Panic("Utf8Decode: expected List of Int64".into())),
                };
                match String::from_utf8(bytes) {
                    Ok(decoded) => {
                        let status_ptr = self.heap.alloc_string("ok".to_string());
                        let decoded_ptr = self.heap.alloc_string(decoded);
                        let tuple_ptr = self.heap.alloc_tuple(vec![
                            GcValue::String(status_ptr),
                            GcValue::String(decoded_ptr),
                        ]);
                        set_reg!(dst, GcValue::Tuple(tuple_ptr));
                    }
                    Err(e) => {
                        let status_ptr = self.heap.alloc_string("error".to_string());
                        let error_ptr = self.heap.alloc_string(format!("invalid UTF-8: {}", e));
                        let tuple_ptr = self.heap.alloc_tuple(vec![
                            GcValue::String(status_ptr),
                            GcValue::String(error_ptr),
                        ]);
                        set_reg!(dst, GcValue::Tuple(tuple_ptr));
                    }
                }
            }

            // === Concurrency: Spawn ===
            Spawn(dst, func_reg, ref args) => {
                let func_val = reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();

                let (func, captures) = match func_val {
                    GcValue::Function(f) => (f, vec![]),
                    GcValue::Closure(ptr, _) => {
                        let closure = self.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid closure pointer".into()))?;
                        (closure.function.clone(), closure.captures.clone())
                    }
                    _ => return Err(RuntimeError::Panic("Spawn: expected function or closure".into())),
                };

                // Convert args and captures to thread-safe values (deep copy)
                let safe_args: Vec<ThreadSafeValue> = arg_values.iter()
                    .filter_map(|v| ThreadSafeValue::from_gc_value(v, &self.heap))
                    .collect();
                let safe_captures: Vec<ThreadSafeValue> = captures.iter()
                    .filter_map(|v| ThreadSafeValue::from_gc_value(v, &self.heap))
                    .collect();

                // Allocate new PID
                let child_pid = self.shared.alloc_pid();
                let shared_clone = self.shared.clone();

                // Create mailbox channel BEFORE spawning to avoid race condition
                // The child process will receive messages through this channel
                let (mailbox_sender, mailbox_receiver) = tokio::sync::mpsc::unbounded_channel();

                // Register the process BEFORE spawning - this ensures messages can be
                // delivered immediately after spawn returns
                self.shared.register_process(child_pid, mailbox_sender.clone()).await;

                // Spawn as tokio task (can run on any thread)
                // Wrap with AssertSend because all underlying types are Send but the compiler
                // can't prove it through async fn's opaque future types.
                let func_name = func.name.clone();
                tokio::spawn(AssertSend(async move {
                    if std::env::var("ASYNC_VM_DEBUG").is_ok() {
                        eprintln!("[AsyncVM] Spawned process {} running function: {}", child_pid.0, func_name);
                    }
                    // Create new process with pre-created mailbox
                    let mut process = AsyncProcess::new_with_mailbox(child_pid, shared_clone.clone(), mailbox_sender, mailbox_receiver);

                    // Convert thread-safe values back to GcValues in new heap
                    let gc_args: Vec<GcValue> = safe_args.iter()
                        .map(|v| v.to_gc_value(&mut process.heap))
                        .collect();
                    let gc_captures: Vec<GcValue> = safe_captures.iter()
                        .map(|v| v.to_gc_value(&mut process.heap))
                        .collect();

                    // Set up initial call frame
                    let reg_count = func.code.register_count as usize;
                    let mut registers = vec![GcValue::Unit; reg_count];
                    for (i, arg) in gc_args.into_iter().enumerate() {
                        if i < reg_count {
                            registers[i] = arg;
                        }
                    }

                    process.frames.push(CallFrame {
                        function: func.clone(),
                        ip: 0,
                        registers,
                        captures: gc_captures,
                        return_reg: None,
                    });

                    // Run the process
                    let _result = process.run().await;

                    // Unregister on exit
                    shared_clone.unregister_process(child_pid).await;
                }));

                // Yield to allow the spawned task to start running
                // This is critical for recursive spawns where parent immediately waits
                tokio::task::yield_now().await;

                set_reg!(dst, GcValue::Pid(child_pid.0));
            }

            // === Concurrency: Send ===
            Send(target_reg, msg_reg) => {
                let target_val = reg!(target_reg);
                let message = reg!(msg_reg);

                let target_pid = match target_val {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("Send: expected Pid".into())),
                };

                // Convert message to thread-safe value
                let safe_msg = ThreadSafeValue::from_gc_value(&message, &self.heap)
                    .ok_or_else(|| RuntimeError::Panic("Send: cannot convert message".into()))?;

                // Send via async channel (non-blocking)
                self.shared.send_message(target_pid, safe_msg).await;
            }

            // === Concurrency: Receive (async!) ===
            Receive(dst) => {
                // This is where async shines - we await the message!
                // Unlike the parallel VM which polls, we yield to tokio scheduler
                match self.mailbox.recv().await {
                    Some(msg) => {
                        let gc_msg = msg.to_gc_value(&mut self.heap);
                        set_reg!(dst, gc_msg);
                    }
                    None => {
                        // Channel closed - process is dying
                        return Ok(StepResult::Finished(GcValue::Unit));
                    }
                }
            }

            // === Concurrency: Receive with Timeout ===
            ReceiveTimeout(dst, timeout_reg) => {
                let timeout_ms = match reg!(timeout_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::Panic("ReceiveTimeout: expected Int64 for timeout".into())),
                };

                let timeout = Duration::from_millis(timeout_ms);

                // Use tokio timeout - properly yields to scheduler
                match tokio::time::timeout(timeout, self.mailbox.recv()).await {
                    Ok(Some(msg)) => {
                        // Message received before timeout
                        let gc_msg = msg.to_gc_value(&mut self.heap);
                        set_reg!(dst, gc_msg);
                    }
                    Ok(None) => {
                        // Channel closed
                        set_reg!(dst, GcValue::Unit);
                    }
                    Err(_) => {
                        // Timeout expired
                        set_reg!(dst, GcValue::Unit);
                    }
                }
            }

            // === Sleep ===
            Sleep(ms_reg) => {
                let ms = match reg!(ms_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::Panic("Sleep: expected Int64 for milliseconds".into())),
                };
                tokio::time::sleep(Duration::from_millis(ms)).await;
            }

            // === Type constructors ===
            MakeVariant(dst, type_idx, ctor_idx, ref field_regs) => {
                let type_name = match get_const!(type_idx) {
                    Value::String(s) => Arc::clone(&s),
                    _ => return Err(RuntimeError::Panic("Variant type must be string".to_string())),
                };
                let constructor = match get_const!(ctor_idx) {
                    Value::String(s) => Arc::clone(&s),
                    _ => return Err(RuntimeError::Panic("Variant constructor must be string".to_string())),
                };
                let fields: Vec<GcValue> = field_regs.iter().map(|&r| reg!(r)).collect();
                let ptr = self.heap.alloc_variant(type_name, constructor, fields);
                set_reg!(dst, GcValue::Variant(ptr));
            }

            MakeRecord(dst, type_idx, ref field_regs) => {
                let type_name = match get_const!(type_idx) {
                    Value::String(s) => (*s).clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let fields: Vec<GcValue> = field_regs.iter().map(|&r| reg!(r)).collect();
                // Look up type in static types first, then dynamic_types (eval-defined)
                let type_info = self.shared.types.read().unwrap().get(&type_name).cloned()
                    .or_else(|| self.shared.dynamic_types.read().unwrap().get(&type_name).cloned());
                let field_names: Vec<String> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.name.clone()).collect())
                    .unwrap_or_else(|| (0..fields.len()).map(|i| format!("_{}", i)).collect());
                let mutable_fields: Vec<bool> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.mutable).collect())
                    .unwrap_or_else(|| vec![false; fields.len()]);
                let ptr = self.heap.alloc_record(type_name, field_names, fields, mutable_fields);
                set_reg!(dst, GcValue::Record(ptr));
            }

            GetVariantField(dst, src, idx) => {
                let src_val = reg!(src);
                let value = match src_val {
                    GcValue::Variant(ptr) => {
                        let variant = self.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        variant.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of bounds", idx)))?
                    }
                    GcValue::Record(ptr) => {
                        let record = self.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        record.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Record field {} out of bounds", idx)))?
                    }
                    _ => return Err(RuntimeError::Panic("GetVariantField expects variant or record".to_string())),
                };
                set_reg!(dst, value);
            }

            GetVariantFieldByName(dst, src, name_idx) => {
                let field_name = match get_const!(name_idx) {
                    Value::String(s) => s.clone(),
                    _ => return Err(RuntimeError::Panic("GetVariantFieldByName: expected string constant".into())),
                };
                let src_val = reg!(src);
                let value = match src_val {
                    GcValue::Record(ptr) => {
                        let record = self.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        // Find field index by name
                        // Records store fields in order matching field_names
                        if let Some(idx) = record.field_names.iter().position(|n| n == field_name.as_str()) {
                            record.fields.get(idx).cloned()
                                .ok_or_else(|| RuntimeError::Panic(format!("Record field {} out of bounds", field_name)))?
                        } else {
                            return Err(RuntimeError::Panic(format!("Field '{}' not found in record '{}'", field_name, record.type_name)));
                        }
                    }
                    _ => return Err(RuntimeError::Panic("GetVariantFieldByName expects record".to_string())),
                };
                set_reg!(dst, value);
            }

            MakeSet(dst, ref elements) => {
                let mut items = ImblHashSet::new();
                for &r in elements.iter() {
                    let val = reg!(r);
                    if let Some(key) = val.to_gc_map_key(&self.heap) {
                        items.insert(key);
                    } else {
                        return Err(RuntimeError::TypeError {
                            expected: "hashable type".to_string(),
                            found: format!("{:?}", val),
                        });
                    }
                }
                let ptr = self.heap.alloc_set(items);
                set_reg!(dst, GcValue::Set(ptr));
            }

            MakeMap(dst, ref entries) => {
                let mut map = ImblHashMap::new();
                for (key_reg, val_reg) in entries.iter() {
                    let key_val = reg!(*key_reg);
                    let val = reg!(*val_reg);
                    if let Some(key) = key_val.to_gc_map_key(&self.heap) {
                        map.insert(key, val);
                    } else {
                        return Err(RuntimeError::TypeError {
                            expected: "hashable type".to_string(),
                            found: format!("{:?}", key_val),
                        });
                    }
                }
                let ptr = self.heap.alloc_map(map);
                set_reg!(dst, GcValue::Map(ptr));
            }

            // === Exception handling ===
            PushHandler(offset) => {
                let frame_index = self.frames.len() - 1;
                let catch_ip = (self.frames[frame_index].ip as isize + *offset as isize) as usize;
                self.handlers.push(ExceptionHandler {
                    frame_index,
                    catch_ip,
                });
            }

            PopHandler => {
                self.handlers.pop();
            }

            GetException(dst) => {
                let exception = self.current_exception.clone().unwrap_or(GcValue::Unit);
                set_reg!(dst, exception);
            }

            Throw(src) => {
                let exception = reg!(src);
                self.current_exception = Some(exception);

                // Find the most recent handler
                if let Some(handler) = self.handlers.pop() {
                    // Unwind stack to handler's frame
                    while self.frames.len() > handler.frame_index + 1 {
                        self.frames.pop();
                    }
                    // Jump to catch block
                    self.frames[handler.frame_index].ip = handler.catch_ip;
                } else {
                    // No handler - propagate as runtime error
                    let msg = self.heap.display_value(self.current_exception.as_ref().unwrap());
                    return Err(RuntimeError::Panic(format!("Uncaught exception: {}", msg)));
                }
            }

            // === Type checking instructions ===
            IsMap(dst, src) => {
                let val = reg!(src);
                let is_map = matches!(val, GcValue::Map(_) | GcValue::SharedMap(_));
                set_reg!(dst, GcValue::Bool(is_map));
            }

            IsSet(dst, src) => {
                let val = reg!(src);
                let is_set = matches!(val, GcValue::Set(_));
                set_reg!(dst, GcValue::Bool(is_set));
            }

            // === TailCall (preserve return_reg from current frame) ===
            TailCall(func_reg, ref args) => {
                let func_val = reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|&r| reg!(r)).collect();
                // Preserve return_reg from current frame
                let return_reg = self.frames.last().unwrap().return_reg;

                match func_val {
                    GcValue::Function(func) => {
                        // Pop current frame and push new one (tail call optimization)
                        self.frames.pop();
                        let reg_count = func.code.register_count as usize;
                        let mut registers = vec![GcValue::Unit; reg_count.max(arg_values.len())];
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            registers[i] = arg;
                        }
                        self.frames.push(CallFrame {
                            function: func.clone(),
                            ip: 0,
                            registers,
                            captures: Vec::new(),
                            return_reg,
                        });
                    }
                    GcValue::Closure(ptr, _) => {
                        let closure = self.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid closure".into()))?;
                        let func = closure.function.clone();
                        let captures = closure.captures.clone();
                        self.frames.pop();
                        let reg_count = func.code.register_count as usize;
                        let mut registers = vec![GcValue::Unit; reg_count.max(arg_values.len() + captures.len())];
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            registers[i] = arg;
                        }
                        self.frames.push(CallFrame {
                            function: func,
                            ip: 0,
                            registers,
                            captures,
                            return_reg,
                        });
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function or Closure".to_string(),
                        found: format!("{:?}", func_val),
                    }),
                }
            }

            // === Type conversions ===
            ToInt32(dst, src) => {
                let val = reg!(src);
                let result = match val {
                    GcValue::Int64(v) => GcValue::Int32(v as i32),
                    GcValue::Int32(v) => GcValue::Int32(v),
                    GcValue::Float64(v) => GcValue::Int32(v as i32),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: format!("{:?}", val),
                    }),
                };
                set_reg!(dst, result);
            }

            // === Map operations ===
            MapContainsKey(dst, map_reg, key_reg) => {
                let map_val = reg!(map_reg);
                let key_val = reg!(key_reg);
                let result = match &map_val {
                    GcValue::Map(ptr) => {
                        if let Some(map) = self.heap.get_map(*ptr) {
                            if let Some(key) = key_val.to_gc_map_key(&self.heap) {
                                map.entries.contains_key(&key)
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    GcValue::SharedMap(shared_map) => {
                        if let Some(gc_key) = key_val.to_gc_map_key(&self.heap) {
                            let shared_key = gc_key.to_shared_key();
                            shared_map.contains_key(&shared_key)
                        } else {
                            false
                        }
                    }
                    _ => false
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            MapGet(dst, map_reg, key_reg) => {
                let map_val = reg!(map_reg);
                let key_val = reg!(key_reg);
                let result = match &map_val {
                    GcValue::Map(ptr) => {
                        if let Some(map) = self.heap.get_map(*ptr) {
                            if let Some(key) = key_val.to_gc_map_key(&self.heap) {
                                map.entries.get(&key).cloned()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    GcValue::SharedMap(shared_map) => {
                        if let Some(gc_key) = key_val.to_gc_map_key(&self.heap) {
                            let shared_key = gc_key.to_shared_key();
                            shared_map.get(&shared_key).map(|v| self.heap.shared_to_gc_value(v))
                        } else {
                            None
                        }
                    }
                    _ => None
                };
                match result {
                    Some(val) => set_reg!(dst, val),
                    None => return Err(RuntimeError::Panic("Key not found in map".to_string())),
                }
            }

            SetContains(dst, set_reg, val_reg) => {
                let set_val = reg!(set_reg);
                let elem_val = reg!(val_reg);
                let result = if let GcValue::Set(ptr) = &set_val {
                    if let Some(set) = self.heap.get_set(*ptr) {
                        if let Some(key) = elem_val.to_gc_map_key(&self.heap) {
                            set.items.contains(&key)
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            // === String operations ===
            StringDecons(head_dst, tail_dst, str_reg) => {
                let str_val = reg!(str_reg);
                match str_val {
                    GcValue::String(str_ptr) => {
                        let s = self.heap.get_string(str_ptr).map(|h| h.data.clone()).unwrap_or_default();
                        if s.is_empty() {
                            return Err(RuntimeError::Panic("Cannot decons empty string".to_string()));
                        }
                        let mut chars = s.chars();
                        let head_char = chars.next().unwrap();
                        let tail_str = chars.as_str();
                        let head_ptr = self.heap.alloc_string(head_char.to_string());
                        let tail_ptr = self.heap.alloc_string(tail_str.to_string());
                        set_reg!(head_dst, GcValue::String(head_ptr));
                        set_reg!(tail_dst, GcValue::String(tail_ptr));
                    }
                    _ => return Err(RuntimeError::Panic("StringDecons expects string".to_string())),
                }
            }

            // === Panic ===
            Panic(msg_reg) => {
                let msg = reg!(msg_reg);
                let msg_str = self.heap.display_value(&msg);
                return Err(RuntimeError::Panic(msg_str));
            }

            Nop => {}

            // === External process execution ===
            ExecRun(dst, cmd_reg, args_reg) => {
                // Get command string
                let cmd = match reg!(cmd_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid command string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                // Get args list
                let args = match reg!(args_reg) {
                    GcValue::List(list) => {
                        let mut result = Vec::new();
                        for item in list.items() {
                            if let GcValue::String(s_ptr) = item {
                                if let Some(s) = self.heap.get_string(s_ptr) {
                                    result.push(s.data.clone());
                                }
                            }
                        }
                        result
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecRun {
                        command: cmd,
                        args,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    // Await the result (async!)
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ExecSpawn(dst, cmd_reg, args_reg) => {
                let cmd = match reg!(cmd_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid command string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let args = match reg!(args_reg) {
                    GcValue::List(list) => {
                        let mut result = Vec::new();
                        for item in list.items() {
                            if let GcValue::String(s_ptr) = item {
                                if let Some(s) = self.heap.get_string(s_ptr) {
                                    result.push(s.data.clone());
                                }
                            }
                        }
                        result
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecSpawn {
                        command: cmd,
                        args,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ExecReadLine(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecReadLine {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ExecReadStderr(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecReadStderr {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ExecWrite(dst, handle_reg, data_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let data = match reg!(data_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.as_bytes().to_vec())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecWrite {
                        handle,
                        data,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ExecWait(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecWait {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ExecKill(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ExecKill {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === File IO Operations ===
            FileReadAll(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileReadToString {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileWriteAll(dst, path_reg, data_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let data = match reg!(data_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.as_bytes().to_vec())
                            .ok_or_else(|| RuntimeError::IOError("Invalid data string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileWriteAll {
                        path: std::path::PathBuf::from(path_str),
                        data,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileAppend(dst, path_reg, data_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let data = match reg!(data_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.as_bytes().to_vec())
                            .ok_or_else(|| RuntimeError::IOError("Invalid data string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileAppend {
                        path: std::path::PathBuf::from(path_str),
                        data,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileOpen(dst, path_reg, mode_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let mode_str = match reg!(mode_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid mode string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let mode = match mode_str.as_str() {
                    "r" | "read" => crate::io_runtime::FileMode::Read,
                    "w" | "write" => crate::io_runtime::FileMode::Write,
                    "a" | "append" => crate::io_runtime::FileMode::Append,
                    "rw" | "readwrite" => crate::io_runtime::FileMode::ReadWrite,
                    _ => return Err(RuntimeError::IOError(format!("Invalid file mode: {}", mode_str))),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileOpen {
                        path: std::path::PathBuf::from(path_str),
                        mode,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileWrite(dst, handle_reg, data_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let data = match reg!(data_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.as_bytes().to_vec())
                            .ok_or_else(|| RuntimeError::IOError("Invalid data string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileWrite {
                        handle,
                        data,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileRead(dst, handle_reg, size_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let size = match reg!(size_reg) {
                    GcValue::Int64(s) => s as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileRead {
                        handle,
                        size,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileReadLine(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileReadLine {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileFlush(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileFlush {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileClose(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileClose {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileSeek(dst, handle_reg, offset_reg, whence_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(h) => h as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let offset = match reg!(offset_reg) {
                    GcValue::Int64(o) => o,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let whence_str = match reg!(whence_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid whence string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let whence = match whence_str.as_str() {
                    "start" => crate::io_runtime::SeekWhence::Start,
                    "current" => crate::io_runtime::SeekWhence::Current,
                    "end" => crate::io_runtime::SeekWhence::End,
                    _ => return Err(RuntimeError::IOError(format!("Invalid seek whence: {}", whence_str))),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileSeek {
                        handle,
                        offset,
                        whence,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === Directory Operations ===
            DirCreate(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::DirCreate {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirCreateAll(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::DirCreateAll {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirList(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::DirList {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirRemove(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::DirRemove {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirRemoveAll(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::DirRemoveAll {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === File Utility Operations ===
            FileExists(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileExists {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirExists(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::DirExists {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileRemove(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileRemove {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileRename(dst, old_path_reg, new_path_reg) => {
                let old_path_str = match reg!(old_path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let new_path_str = match reg!(new_path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileRename {
                        old_path: std::path::PathBuf::from(old_path_str),
                        new_path: std::path::PathBuf::from(new_path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileCopy(dst, src_path_reg, dest_path_reg) => {
                let src_path_str = match reg!(src_path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let dest_path_str = match reg!(dest_path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileCopy {
                        src_path: std::path::PathBuf::from(src_path_str),
                        dest_path: std::path::PathBuf::from(dest_path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileSize(dst, path_reg) => {
                let path_str = match reg!(path_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid path string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::FileSize {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === HTTP Client Operations ===
            HttpGet(dst, url_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpGet { url, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpPost(dst, url_reg, body_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let body = match reg!(body_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.as_bytes().to_vec()),
                    GcValue::Unit => None,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String or ()".to_string(),
                        found: "other".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Post,
                            url,
                            headers: vec![],
                            body,
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpPut(dst, url_reg, body_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let body = match reg!(body_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.as_bytes().to_vec()),
                    GcValue::Unit => None,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String or ()".to_string(),
                        found: "other".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Put,
                            url,
                            headers: vec![],
                            body,
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpDelete(dst, url_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Delete,
                            url,
                            headers: vec![],
                            body: None,
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpPatch(dst, url_reg, body_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let body = match reg!(body_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.as_bytes().to_vec()),
                    GcValue::Unit => None,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String or ()".to_string(),
                        found: "other".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Patch,
                            url,
                            headers: vec![],
                            body,
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpHead(dst, url_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Head,
                            url,
                            headers: vec![],
                            body: None,
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpRequest(dst, method_reg, url_reg, headers_reg, body_reg) => {
                let method_str = match reg!(method_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let method = match method_str.to_uppercase().as_str() {
                    "GET" => crate::io_runtime::HttpMethod::Get,
                    "POST" => crate::io_runtime::HttpMethod::Post,
                    "PUT" => crate::io_runtime::HttpMethod::Put,
                    "DELETE" => crate::io_runtime::HttpMethod::Delete,
                    "PATCH" => crate::io_runtime::HttpMethod::Patch,
                    "HEAD" => crate::io_runtime::HttpMethod::Head,
                    _ => return Err(RuntimeError::Panic(format!("Unknown HTTP method: {}", method_str))),
                };
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let headers = match reg!(headers_reg) {
                    GcValue::List(list) => {
                        let mut result = Vec::new();
                        for item in list.iter() {
                            if let GcValue::Tuple(ptr) = item {
                                if let Some(tuple) = self.heap.get_tuple(*ptr) {
                                    if tuple.items.len() >= 2 {
                                        let name = match &tuple.items[0] {
                                            GcValue::String(ptr) => self.heap.get_string(*ptr)
                                                .map(|s| s.data.clone())
                                                .unwrap_or_default(),
                                            _ => String::new(),
                                        };
                                        let value = match &tuple.items[1] {
                                            GcValue::String(ptr) => self.heap.get_string(*ptr)
                                                .map(|s| s.data.clone())
                                                .unwrap_or_default(),
                                            _ => String::new(),
                                        };
                                        result.push((name, value));
                                    }
                                }
                            }
                        }
                        result
                    }
                    GcValue::Unit => vec![],
                    _ => vec![],
                };
                let body = match reg!(body_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.as_bytes().to_vec()),
                    GcValue::Unit => None,
                    _ => None,
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method,
                            url,
                            headers,
                            body,
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === HTTP Server Operations ===
            ServerBind(dst, port_reg) => {
                let port = match reg!(port_reg) {
                    GcValue::Int64(n) => n as u16,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ServerBind { port, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ServerAccept(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ServerAccept { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ServerRespond(dst, request_id_reg, status_reg, headers_reg, body_reg) => {
                let request_id = match reg!(request_id_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let status = match reg!(status_reg) {
                    GcValue::Int64(n) => n as u16,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let headers = match reg!(headers_reg) {
                    GcValue::List(list) => {
                        let mut result = Vec::new();
                        for item in list.iter() {
                            if let GcValue::Tuple(ptr) = item {
                                if let Some(tuple) = self.heap.get_tuple(*ptr) {
                                    if tuple.items.len() >= 2 {
                                        let name = match &tuple.items[0] {
                                            GcValue::String(ptr) => self.heap.get_string(*ptr)
                                                .map(|s| s.data.clone())
                                                .unwrap_or_default(),
                                            _ => String::new(),
                                        };
                                        let value = match &tuple.items[1] {
                                            GcValue::String(ptr) => self.heap.get_string(*ptr)
                                                .map(|s| s.data.clone())
                                                .unwrap_or_default(),
                                            _ => String::new(),
                                        };
                                        result.push((name, value));
                                    }
                                }
                            }
                        }
                        result
                    }
                    GcValue::Unit => vec![],
                    _ => vec![],
                };
                let body = match reg!(body_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr)
                        .map(|s| s.data.as_bytes().to_vec())
                        .unwrap_or_default(),
                    GcValue::Unit => Vec::new(),
                    _ => Vec::new(),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ServerRespond {
                        request_id,
                        status,
                        headers,
                        body,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ServerClose(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::ServerClose { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // PostgreSQL operations - return values directly, panic on error
            PgConnect(dst, conn_str_reg) => {
                let conn_string = match reg!(conn_str_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgConnect { connection_string: conn_string, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            // Throw catchable exception
                            self.throw_exception("connection_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgQuery(dst, handle_reg, query_reg, params_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let query = match reg!(query_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid query string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let params = match reg!(params_reg) {
                    GcValue::List(list) => {
                        let mut pg_params = Vec::new();
                        for item in list.iter() {
                            let param = self.gc_value_to_pg_param(&item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgQuery { handle, query, params, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            // Throw catchable exception
                            self.throw_exception("query_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgExecute(dst, handle_reg, query_reg, params_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let query = match reg!(query_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid query string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let params = match reg!(params_reg) {
                    GcValue::List(list) => {
                        let mut pg_params = Vec::new();
                        for item in list.iter() {
                            let param = self.gc_value_to_pg_param(&item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgExecute { handle, query, params, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            // Throw catchable exception
                            self.throw_exception("execute_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgClose(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgClose { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => return Err(RuntimeError::Panic(format!("Pg.close: {}", e))),
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgBegin(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgBegin { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("transaction_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgCommit(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgCommit { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("transaction_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgRollback(dst, handle_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgRollback { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("transaction_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgPrepare(dst, handle_reg, name_reg, query_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let name = match reg!(name_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid name string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let query = match reg!(query_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid query string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgPrepare { handle, name, query, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("prepare_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgQueryPrepared(dst, handle_reg, name_reg, params_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let name = match reg!(name_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid name string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let params = match reg!(params_reg) {
                    GcValue::List(list) => {
                        let mut pg_params = Vec::new();
                        for item in list.iter() {
                            let param = self.gc_value_to_pg_param(&item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgQueryPrepared { handle, name, params, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("query_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgExecutePrepared(dst, handle_reg, name_reg, params_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let name = match reg!(name_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid name string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let params = match reg!(params_reg) {
                    GcValue::List(list) => {
                        let mut pg_params = Vec::new();
                        for item in list.iter() {
                            let param = self.gc_value_to_pg_param(&item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgExecutePrepared { handle, name, params, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("execute_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgDeallocate(dst, handle_reg, name_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let name = match reg!(name_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid name string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgDeallocate { handle, name, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(resp) => {
                            let gc_value = self.io_response_to_gc_value(resp);
                            set_reg!(dst, gc_value);
                        }
                        Err(e) => {
                            self.throw_exception("deallocate_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === Time builtins ===
            TimeNow(dst) => {
                use chrono::Utc;
                let millis = Utc::now().timestamp_millis();
                set_reg!(dst, GcValue::Int64(millis));
            }

            TimeFromDate(dst, year_reg, month_reg, day_reg) => {
                use chrono::NaiveDate;
                let year = match reg!(year_reg) {
                    GcValue::Int64(v) => v as i32,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let month = match reg!(month_reg) {
                    GcValue::Int64(v) => v as u32,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let day = match reg!(day_reg) {
                    GcValue::Int64(v) => v as u32,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let date = NaiveDate::from_ymd_opt(year, month, day)
                    .ok_or_else(|| RuntimeError::Panic(format!("Invalid date: {}-{}-{}", year, month, day)))?;
                let dt = date.and_hms_opt(0, 0, 0).unwrap();
                let millis = dt.and_utc().timestamp_millis();
                set_reg!(dst, GcValue::Int64(millis));
            }

            TimeFromTime(dst, hour_reg, min_reg, sec_reg) => {
                let hour = match reg!(hour_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let min = match reg!(min_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let sec = match reg!(sec_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let millis = hour * 3600_000 + min * 60_000 + sec * 1000;
                set_reg!(dst, GcValue::Int64(millis));
            }

            TimeFromDateTime(dst, year_reg, month_reg, day_reg, hour_reg, min_reg, sec_reg) => {
                use chrono::NaiveDate;
                let year = match reg!(year_reg) { GcValue::Int64(v) => v as i32, _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }) };
                let month = match reg!(month_reg) { GcValue::Int64(v) => v as u32, _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }) };
                let day = match reg!(day_reg) { GcValue::Int64(v) => v as u32, _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }) };
                let hour = match reg!(hour_reg) { GcValue::Int64(v) => v as u32, _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }) };
                let min = match reg!(min_reg) { GcValue::Int64(v) => v as u32, _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }) };
                let sec = match reg!(sec_reg) { GcValue::Int64(v) => v as u32, _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }) };
                let date = NaiveDate::from_ymd_opt(year, month, day)
                    .ok_or_else(|| RuntimeError::Panic(format!("Invalid date: {}-{}-{}", year, month, day)))?;
                let dt = date.and_hms_opt(hour, min, sec)
                    .ok_or_else(|| RuntimeError::Panic(format!("Invalid time: {}:{}:{}", hour, min, sec)))?;
                let millis = dt.and_utc().timestamp_millis();
                set_reg!(dst, GcValue::Int64(millis));
            }

            TimeYear(dst, ts_reg) => {
                use chrono::{DateTime, Datelike};
                let millis = match reg!(ts_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let dt = DateTime::from_timestamp_millis(millis).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                set_reg!(dst, GcValue::Int64(dt.year() as i64));
            }

            TimeMonth(dst, ts_reg) => {
                use chrono::{DateTime, Datelike};
                let millis = match reg!(ts_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let dt = DateTime::from_timestamp_millis(millis).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                set_reg!(dst, GcValue::Int64(dt.month() as i64));
            }

            TimeDay(dst, ts_reg) => {
                use chrono::{DateTime, Datelike};
                let millis = match reg!(ts_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let dt = DateTime::from_timestamp_millis(millis).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                set_reg!(dst, GcValue::Int64(dt.day() as i64));
            }

            TimeHour(dst, ts_reg) => {
                use chrono::{DateTime, Timelike};
                let millis = match reg!(ts_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let dt = DateTime::from_timestamp_millis(millis).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                set_reg!(dst, GcValue::Int64(dt.hour() as i64));
            }

            TimeMinute(dst, ts_reg) => {
                use chrono::{DateTime, Timelike};
                let millis = match reg!(ts_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let dt = DateTime::from_timestamp_millis(millis).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                set_reg!(dst, GcValue::Int64(dt.minute() as i64));
            }

            TimeSecond(dst, ts_reg) => {
                use chrono::{DateTime, Timelike};
                let millis = match reg!(ts_reg) {
                    GcValue::Int64(v) => v,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "non-int".to_string() }),
                };
                let dt = DateTime::from_timestamp_millis(millis).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                set_reg!(dst, GcValue::Int64(dt.second() as i64));
            }

            // === Type introspection and reflection ===
            TypeOf(dst, val_reg) => {
                let type_name = match reg!(val_reg) {
                    GcValue::Unit => "Unit",
                    GcValue::Bool(_) => "Bool",
                    GcValue::Int8(_) => "Int8",
                    GcValue::Int16(_) => "Int16",
                    GcValue::Int32(_) => "Int32",
                    GcValue::Int64(_) => "Int",
                    GcValue::UInt8(_) => "UInt8",
                    GcValue::UInt16(_) => "UInt16",
                    GcValue::UInt32(_) => "UInt32",
                    GcValue::UInt64(_) => "UInt64",
                    GcValue::Float32(_) => "Float32",
                    GcValue::Float64(_) => "Float",
                    GcValue::Decimal(_) => "Decimal",
                    GcValue::Char(_) => "Char",
                    GcValue::String(_) => "String",
                    GcValue::BigInt(_) => "BigInt",
                    GcValue::List(_) => "List",
                    GcValue::Array(_) => "Array",
                    GcValue::Int64Array(_) => "Int64Array",
                    GcValue::Float64Array(_) => "Float64Array",
                    GcValue::Tuple(_) => "Tuple",
                    GcValue::Record(_) => "Record",
                    GcValue::Variant(_) => "Variant",
                    GcValue::Map(_) => "Map",
                    GcValue::SharedMap(_) => "Map",
                    GcValue::Set(_) => "Set",
                    GcValue::Closure(_, _) => "Closure",
                    GcValue::Pid(_) => "Pid",
                    GcValue::Ref(_) => "Ref",
                    GcValue::Function(_) => "Function",
                    GcValue::NativeFunction(_) => "NativeFunction",
                    GcValue::Type(_) => "Type",
                    GcValue::Pointer(_) => "Pointer",
                }.to_string();
                let str_ptr = self.heap.alloc_string(type_name);
                set_reg!(dst, GcValue::String(str_ptr));
            }

            TagOf(dst, val_reg) => {
                let tag = match reg!(val_reg) {
                    GcValue::Variant(ptr) => {
                        self.heap.get_variant(ptr.clone())
                            .map(|v| v.constructor.as_ref().clone())
                            .unwrap_or_default()
                    }
                    _ => String::new(),
                };
                let str_ptr = self.heap.alloc_string(tag);
                set_reg!(dst, GcValue::String(str_ptr));
            }

            Reflect(dst, val_reg) => {
                let val = reg!(val_reg).clone();
                let json_val = self.value_to_json(val)?;
                set_reg!(dst, json_val);
            }

            TypeInfo(dst, name_reg) => {
                let type_name = match reg!(name_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr.clone())
                            .map(|s| s.data.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "non-string".to_string() }),
                };
                let map_val = self.type_info_to_map(&type_name)?;
                set_reg!(dst, map_val);
            }

            Construct(dst, type_reg, json_reg) => {
                let type_name = match reg!(type_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr.clone())
                            .map(|s| s.data.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "non-string".to_string() }),
                };
                let json_val = reg!(json_reg).clone();
                match self.construct_from_json(&type_name, json_val) {
                    Ok(result) => {
                        set_reg!(dst, result);
                    }
                    Err(e) => {
                        // Convert error to catchable exception
                        let error_msg = match e {
                            RuntimeError::Panic(msg) => msg,
                            other => format!("{:?}", other),
                        };
                        let str_ptr = self.heap.alloc_string(error_msg.clone());
                        self.current_exception = Some(GcValue::String(str_ptr));

                        // Find handler (same as Throw instruction)
                        if let Some(handler) = self.handlers.pop() {
                            while self.frames.len() > handler.frame_index + 1 {
                                self.frames.pop();
                            }
                            self.frames[handler.frame_index].ip = handler.catch_ip;
                        } else {
                            return Err(RuntimeError::Panic(format!("Uncaught exception: {}", error_msg)));
                        }
                    }
                }
            }

            MakeRecordDyn(dst, type_reg, fields_reg) => {
                let type_name = match reg!(type_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr.clone())
                            .map(|s| s.data.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "non-string".to_string() }),
                };
                let fields_map = reg!(fields_reg).clone();
                match self.make_record_from_map(&type_name, fields_map) {
                    Ok(result) => {
                        set_reg!(dst, result);
                    }
                    Err(e) => {
                        let error_msg = match e {
                            RuntimeError::Panic(msg) => msg,
                            other => format!("{:?}", other),
                        };
                        let str_ptr = self.heap.alloc_string(error_msg.clone());
                        self.current_exception = Some(GcValue::String(str_ptr));

                        if let Some(handler) = self.handlers.pop() {
                            while self.frames.len() > handler.frame_index + 1 {
                                self.frames.pop();
                            }
                            self.frames[handler.frame_index].ip = handler.catch_ip;
                        } else {
                            return Err(RuntimeError::Panic(format!("Uncaught exception: {}", error_msg)));
                        }
                    }
                }
            }

            MakeVariantDyn(dst, type_reg, ctor_reg, fields_reg) => {
                let type_name = match reg!(type_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr.clone())
                            .map(|s| s.data.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "non-string".to_string() }),
                };
                let ctor_name = match reg!(ctor_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr.clone())
                            .map(|s| s.data.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "non-string".to_string() }),
                };
                let fields_map = reg!(fields_reg).clone();
                match self.make_variant_from_map(&type_name, &ctor_name, fields_map) {
                    Ok(result) => {
                        set_reg!(dst, result);
                    }
                    Err(e) => {
                        let error_msg = match e {
                            RuntimeError::Panic(msg) => msg,
                            other => format!("{:?}", other),
                        };
                        let str_ptr = self.heap.alloc_string(error_msg.clone());
                        self.current_exception = Some(GcValue::String(str_ptr));

                        if let Some(handler) = self.handlers.pop() {
                            while self.frames.len() > handler.frame_index + 1 {
                                self.frames.pop();
                            }
                            self.frames[handler.frame_index].ip = handler.catch_ip;
                        } else {
                            return Err(RuntimeError::Panic(format!("Uncaught exception: {}", error_msg)));
                        }
                    }
                }
            }

            // Unimplemented instructions - add as needed
            _ => {
                eprintln!("[AsyncVM] Unimplemented instruction: {:?}", instruction);
                return Err(RuntimeError::Panic(format!(
                    "Unimplemented instruction in async VM: {:?}",
                    instruction
                )));
            }
        }

        // Continue executing more instructions in the tight loop
        // (Frame-changing instructions like Call/Return already returned above)
        continue 'instruction_loop;

        } // end of 'instruction_loop
    }

    /// Handle an IO result - on success returns the value, on error throws an exception.
    /// Returns Ok(Some(value)) on success, Ok(None) if exception was caught (IP jumped),
    /// or Err if exception was uncaught.
    fn handle_io_result(&mut self, result: Result<IoResponseValue, crate::io_runtime::IoError>, error_kind: &str) -> Result<Option<GcValue>, RuntimeError> {
        match result {
            Ok(response) => {
                let value = self.io_response_to_gc_value(response);
                Ok(Some(value))
            }
            Err(e) => {
                // Throw catchable exception - if caught, handler will jump and we return None
                self.throw_exception(error_kind, e.to_string())?;
                Ok(None)
            }
        }
    }

    /// Convert an IO response value to a GcValue.
    fn io_response_to_gc_value(&mut self, response: IoResponseValue) -> GcValue {
        match response {
            IoResponseValue::Unit => GcValue::Unit,
            IoResponseValue::Bytes(bytes) => {
                match std::string::String::from_utf8(bytes.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let values: Vec<GcValue> = bytes.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(values))
                    }
                }
            }
            IoResponseValue::String(s) => GcValue::String(self.heap.alloc_string(s)),
            IoResponseValue::FileHandle(handle_id) => GcValue::Int64(handle_id as i64),
            IoResponseValue::Int(n) => GcValue::Int64(n),
            IoResponseValue::Bool(b) => GcValue::Bool(b),
            IoResponseValue::StringList(strings) => {
                let values: Vec<GcValue> = strings
                    .into_iter()
                    .map(|s| GcValue::String(self.heap.alloc_string(s)))
                    .collect();
                GcValue::List(GcList::from_vec(values))
            }
            IoResponseValue::HttpResponse { status, headers, body } => {
                let header_tuples: Vec<GcValue> = headers
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(self.heap.alloc_string(k));
                        let val = GcValue::String(self.heap.alloc_string(v));
                        GcValue::Tuple(self.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let headers_list = GcValue::List(GcList::from_vec(header_tuples));

                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
                    }
                };

                GcValue::Record(self.heap.alloc_record(
                    "HttpResponse".to_string(),
                    vec!["status".to_string(), "headers".to_string(), "body".to_string()],
                    vec![GcValue::Int64(status as i64), headers_list, body_value],
                    vec![false, false, false],
                ))
            }
            IoResponseValue::OptionString(opt) => {
                match opt {
                    Some(s) => GcValue::String(self.heap.alloc_string(s)),
                    None => GcValue::String(self.heap.alloc_string("eof".to_string())),
                }
            }
            IoResponseValue::ServerHandle(handle_id) => GcValue::Int64(handle_id as i64),
            IoResponseValue::ServerRequest { request_id, method, path, headers, body } => {
                let header_tuples: Vec<GcValue> = headers
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(self.heap.alloc_string(k));
                        let val = GcValue::String(self.heap.alloc_string(v));
                        GcValue::Tuple(self.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let headers_list = GcValue::List(GcList::from_vec(header_tuples));

                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
                    }
                };

                let method_str = GcValue::String(self.heap.alloc_string(method));
                let path_str = GcValue::String(self.heap.alloc_string(path));

                GcValue::Record(self.heap.alloc_record(
                    "HttpRequest".to_string(),
                    vec!["id".to_string(), "method".to_string(), "path".to_string(), "headers".to_string(), "body".to_string()],
                    vec![GcValue::Int64(request_id as i64), method_str, path_str, headers_list, body_value],
                    vec![false, false, false, false, false],
                ))
            }
            IoResponseValue::ExecResult { exit_code, stdout, stderr } => {
                let stdout_value = match std::string::String::from_utf8(stdout.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = stdout.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
                    }
                };

                let stderr_value = match std::string::String::from_utf8(stderr.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = stderr.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
                    }
                };

                GcValue::Record(self.heap.alloc_record(
                    "ExecResult".to_string(),
                    vec!["exitCode".to_string(), "stdout".to_string(), "stderr".to_string()],
                    vec![GcValue::Int64(exit_code as i64), stdout_value, stderr_value],
                    vec![false, false, false],
                ))
            }
            IoResponseValue::ExecHandle(handle_id) => GcValue::Int64(handle_id as i64),
            IoResponseValue::ExitCode(code) => GcValue::Int64(code as i64),
            IoResponseValue::PgHandle(handle_id) => GcValue::Int64(handle_id as i64),
            IoResponseValue::PgRows(rows) => {
                let row_values: Vec<GcValue> = rows
                    .into_iter()
                    .map(|row| {
                        let col_values: Vec<GcValue> = row
                            .into_iter()
                            .map(|col| self.pg_value_to_gc_value(col))
                            .collect();
                        GcValue::List(GcList::from_vec(col_values))
                    })
                    .collect();
                GcValue::List(GcList::from_vec(row_values))
            }
            IoResponseValue::PgAffected(count) => GcValue::Int64(count as i64),
        }
    }

    /// Convert a PgValue to GcValue
    fn pg_value_to_gc_value(&mut self, pv: crate::process::PgValue) -> GcValue {
        use crate::process::PgValue;
        match pv {
            PgValue::Null => GcValue::Unit,
            PgValue::Bool(b) => GcValue::Bool(b),
            PgValue::Int(i) => GcValue::Int64(i),
            PgValue::Float(f) => GcValue::Float64(f),
            PgValue::String(s) => GcValue::String(self.heap.alloc_string(s)),
            PgValue::Bytes(bytes) => {
                let values: Vec<GcValue> = bytes.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                GcValue::List(GcList::from_vec(values))
            }
            PgValue::Timestamp(millis) => GcValue::Int64(millis),
        }
    }

    /// Convert a GcValue to a PgParam for query parameters
    fn gc_value_to_pg_param(&self, value: &GcValue) -> Result<crate::io_runtime::PgParam, RuntimeError> {
        use crate::io_runtime::PgParam;
        match value {
            GcValue::Unit => Ok(PgParam::Null),
            GcValue::Bool(b) => Ok(PgParam::Bool(*b)),
            GcValue::Int64(i) => Ok(PgParam::Int(*i)),
            GcValue::Int32(i) => Ok(PgParam::Int(*i as i64)),
            GcValue::Int16(i) => Ok(PgParam::Int(*i as i64)),
            GcValue::Int8(i) => Ok(PgParam::Int(*i as i64)),
            GcValue::Float64(f) => Ok(PgParam::Float(*f)),
            GcValue::Float32(f) => Ok(PgParam::Float(*f as f64)),
            GcValue::String(ptr) => {
                if let Some(s) = self.heap.get_string(*ptr) {
                    Ok(PgParam::String(s.data.clone()))
                } else {
                    Err(RuntimeError::IOError("Invalid string pointer".to_string()))
                }
            }
            _ => Err(RuntimeError::TypeError {
                expected: "Int, Float, String, Bool, or Unit".to_string(),
                found: "unsupported type for Pg param".to_string(),
            }),
        }
    }

    /// Try to handle an exception with registered handlers.
    fn handle_exception(&mut self, _error: &RuntimeError) -> bool {
        // TODO: Implement exception handling
        false
    }

    /// Throw a catchable exception as a (kind, message) tuple.
    /// Returns Ok(true) if caught, Err if uncaught.
    fn throw_exception(&mut self, kind: &str, message: String) -> Result<bool, RuntimeError> {
        // Create a tuple (kind, message) for pattern matching
        let kind_val = GcValue::String(self.heap.alloc_string(kind.to_string()));
        let msg_val = GcValue::String(self.heap.alloc_string(message.clone()));
        let tuple_ptr = self.heap.alloc_tuple(vec![kind_val, msg_val]);
        let exception = GcValue::Tuple(tuple_ptr);
        self.current_exception = Some(exception);

        // Find the most recent handler
        if let Some(handler) = self.handlers.pop() {
            // Unwind stack to handler's frame
            while self.frames.len() > handler.frame_index + 1 {
                self.frames.pop();
            }
            // Jump to catch block
            self.frames[handler.frame_index].ip = handler.catch_ip;
            Ok(true)
        } else {
            // No handler - propagate as runtime error
            Err(RuntimeError::Panic(format!("Uncaught exception: ({}, {})", kind, message)))
        }
    }

    /// Convert a value to Json variant (for reflect builtin)
    fn value_to_json(&mut self, val: GcValue) -> Result<GcValue, RuntimeError> {
        let json_type = Arc::new("stdlib.json.Json".to_string());

        match val {
            GcValue::Unit => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Null".to_string()), vec![]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Bool(b) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Bool".to_string()), vec![GcValue::Bool(b)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Int64(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Int8(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Int16(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Int32(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::UInt8(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::UInt16(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::UInt32(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::UInt64(n) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Float64(f) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(f)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Float32(f) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(f as f64)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Decimal(d) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(d.to_string().parse::<f64>().unwrap_or(0.0))]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Char(c) => {
                let str_ptr = self.heap.alloc_string(c.to_string());
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::String(str_ptr) => {
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::BigInt(bi_ptr) => {
                let n = self.heap.get_bigint(bi_ptr)
                    .map(|bi| bi.value.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Number".to_string()), vec![GcValue::Float64(n)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::List(list) => {
                let items: Vec<GcValue> = list.data.iter().cloned().collect();
                let mut json_items = Vec::new();
                for item in items {
                    let json_item = self.value_to_json(item)?;
                    json_items.push(json_item);
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Array(arr_ptr) => {
                let items = self.heap.get_array(arr_ptr)
                    .map(|a| a.items.clone())
                    .unwrap_or_default();
                let mut json_items = Vec::new();
                for item in items {
                    let json_item = self.value_to_json(item)?;
                    json_items.push(json_item);
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Tuple(tup_ptr) => {
                let items = self.heap.get_tuple(tup_ptr)
                    .map(|t| t.items.clone())
                    .unwrap_or_default();
                let mut json_items = Vec::new();
                for item in items {
                    let json_item = self.value_to_json(item)?;
                    json_items.push(json_item);
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Record(rec_ptr) => {
                let (field_names, fields) = self.heap.get_record(rec_ptr)
                    .map(|r| (r.field_names.clone(), r.fields.clone()))
                    .unwrap_or_default();
                let mut pairs = Vec::new();
                for (name, value) in field_names.into_iter().zip(fields) {
                    let json_value = self.value_to_json(value)?;
                    let name_ptr = self.heap.alloc_string(name);
                    let tuple_ptr = self.heap.alloc_tuple(vec![GcValue::String(name_ptr), json_value]);
                    pairs.push(GcValue::Tuple(tuple_ptr));
                }
                let list = GcList::from_vec(pairs);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Object".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Variant(var_ptr) => {
                let (constructor, fields) = self.heap.get_variant(var_ptr)
                    .map(|v| {
                        (v.constructor.as_ref().clone(), v.fields.clone())
                    })
                    .unwrap_or_else(|| ("Unknown".to_string(), vec![]));

                // Determine the inner value based on field count
                let inner_value = if fields.is_empty() {
                    // Unit variant: use null
                    self.heap.alloc_variant(json_type.clone(), Arc::new("Null".to_string()), vec![])
                } else if fields.len() == 1 {
                    // Single-field variant: use value directly (no _0 wrapper)
                    let json_value = self.value_to_json(fields[0].clone())?;
                    match json_value {
                        GcValue::Variant(ptr) => ptr,
                        other => {
                            // Wrap primitive in appropriate Json variant
                            return Err(RuntimeError::Panic(format!("Unexpected json value: {:?}", other)));
                        }
                    }
                } else {
                    // Multi-field variant: use array [val0, val1, ...]
                    let mut array_items = Vec::new();
                    for value in fields.into_iter() {
                        let json_value = self.value_to_json(value)?;
                        array_items.push(json_value);
                    }
                    let array_list = GcList::from_vec(array_items);
                    self.heap.alloc_variant(json_type.clone(), Arc::new("Array".to_string()), vec![GcValue::List(array_list)])
                };

                // Create outer object with constructor as key
                let constructor_ptr = self.heap.alloc_string(constructor);
                let outer_tuple = self.heap.alloc_tuple(vec![GcValue::String(constructor_ptr), GcValue::Variant(inner_value)]);
                let outer_list = GcList::from_vec(vec![GcValue::Tuple(outer_tuple)]);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Object".to_string()), vec![GcValue::List(outer_list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Map(map_ptr) => {
                let entries: Vec<(GcMapKey, GcValue)> = self.heap.get_map(map_ptr)
                    .map(|m| m.entries.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let mut pairs = Vec::new();
                for (key, value) in entries {
                    let key_str = match &key {
                        GcMapKey::String(s) => s.clone(),
                        GcMapKey::Unit => "()".to_string(),
                        GcMapKey::Bool(b) => b.to_string(),
                        GcMapKey::Char(c) => format!("'{}'", c),
                        GcMapKey::Int8(n) => n.to_string(),
                        GcMapKey::Int16(n) => n.to_string(),
                        GcMapKey::Int32(n) => n.to_string(),
                        GcMapKey::Int64(n) => n.to_string(),
                        GcMapKey::UInt8(n) => n.to_string(),
                        GcMapKey::UInt16(n) => n.to_string(),
                        GcMapKey::UInt32(n) => n.to_string(),
                        GcMapKey::UInt64(n) => n.to_string(),
                        GcMapKey::Record { type_name, .. } => format!("<{}>", type_name),
                        GcMapKey::Variant { type_name, constructor, .. } => format!("{}.{}", type_name, constructor),
                    };
                    let json_value = self.value_to_json(value)?;
                    let key_ptr = self.heap.alloc_string(key_str);
                    let tuple_ptr = self.heap.alloc_tuple(vec![GcValue::String(key_ptr), json_value]);
                    pairs.push(GcValue::Tuple(tuple_ptr));
                }
                let list = GcList::from_vec(pairs);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Object".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::SharedMap(shared_map) => {
                let mut pairs = Vec::new();
                for (key, value) in shared_map.iter() {
                    let key_str = format!("{:?}", key);
                    let gc_value = self.heap.shared_to_gc_value(value);
                    let json_value = self.value_to_json(gc_value)?;
                    let key_ptr = self.heap.alloc_string(key_str);
                    let tuple_ptr = self.heap.alloc_tuple(vec![GcValue::String(key_ptr), json_value]);
                    pairs.push(GcValue::Tuple(tuple_ptr));
                }
                let list = GcList::from_vec(pairs);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Object".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Set(set_ptr) => {
                let items: Vec<GcMapKey> = self.heap.get_set(set_ptr)
                    .map(|s| s.items.iter().cloned().collect())
                    .unwrap_or_default();
                let mut json_items = Vec::new();
                for key in items {
                    let gc_val = key.to_gc_value(&mut self.heap);
                    let json_item = self.value_to_json(gc_val)?;
                    json_items.push(json_item);
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Int64Array(arr_ptr) => {
                let items = self.heap.get_int64_array(arr_ptr)
                    .map(|a| a.items.clone())
                    .unwrap_or_default();
                let mut json_items = Vec::new();
                for n in items {
                    let ptr = self.heap.alloc_variant(json_type.clone(), Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                    json_items.push(GcValue::Variant(ptr));
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Float64Array(arr_ptr) => {
                let items = self.heap.get_float64_array(arr_ptr)
                    .map(|a| a.items.clone())
                    .unwrap_or_default();
                let mut json_items = Vec::new();
                for f in items {
                    let ptr = self.heap.alloc_variant(json_type.clone(), Arc::new("Number".to_string()), vec![GcValue::Float64(f)]);
                    json_items.push(GcValue::Variant(ptr));
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Pid(pid) => {
                let str_ptr = self.heap.alloc_string(format!("<pid:{}>", pid));
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Ref(r) => {
                let str_ptr = self.heap.alloc_string(format!("<ref:{}>", r));
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Closure(_, _) | GcValue::Function(_) | GcValue::NativeFunction(_) => {
                let str_ptr = self.heap.alloc_string("<function>".to_string());
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Type(t) => {
                let str_ptr = self.heap.alloc_string(format!("<type:{}>", t.name));
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Pointer(p) => {
                let str_ptr = self.heap.alloc_string(format!("<ptr:0x{:x}>", p));
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
        }
    }

    /// Get type info as a native Map (for typeInfo builtin)
    fn type_info_to_map(&mut self, type_name: &str) -> Result<GcValue, RuntimeError> {
        // Look up type in static types, then dynamic types
        let type_info = self.shared.types.read().unwrap().get(type_name).cloned()
            .or_else(|| self.shared.dynamic_types.read().unwrap().get(type_name).cloned())
            .or_else(|| self.shared.stdlib_types.read().unwrap().get(type_name).cloned());

        let type_val = match type_info {
            Some(t) => t,
            None => {
                // Return empty map for unknown types
                let empty_map: ImblHashMap<GcMapKey, GcValue> = ImblHashMap::new();
                let ptr = self.heap.alloc_map(empty_map);
                return Ok(GcValue::Map(ptr));
            }
        };

        // Build Map structure: %{ "name": ..., "kind": ..., "fields": [...], "constructors": [...] }
        let mut entries: ImblHashMap<GcMapKey, GcValue> = ImblHashMap::new();

        // name
        let name_str = self.heap.alloc_string(type_val.name.clone());
        entries.insert(GcMapKey::String("name".to_string()), GcValue::String(name_str));

        // kind
        let kind_str = match &type_val.kind {
            crate::value::TypeKind::Primitive => "primitive",
            crate::value::TypeKind::Record { .. } => "record",
            crate::value::TypeKind::Variant => "variant",
            crate::value::TypeKind::Alias { .. } => "alias",
        };
        let kind_val = self.heap.alloc_string(kind_str.to_string());
        entries.insert(GcMapKey::String("kind".to_string()), GcValue::String(kind_val));

        // fields (for records) - List of Maps
        let mut field_list = Vec::new();
        for field in &type_val.fields {
            let mut field_map: ImblHashMap<GcMapKey, GcValue> = ImblHashMap::new();
            let fname = self.heap.alloc_string(field.name.clone());
            field_map.insert(GcMapKey::String("name".to_string()), GcValue::String(fname));
            let ftype = self.heap.alloc_string(field.type_name.clone());
            field_map.insert(GcMapKey::String("type".to_string()), GcValue::String(ftype));
            let field_map_ptr = self.heap.alloc_map(field_map);
            field_list.push(GcValue::Map(field_map_ptr));
        }
        entries.insert(GcMapKey::String("fields".to_string()), GcValue::List(GcList::from_vec(field_list)));

        // constructors (for variants) - List of Maps
        let mut ctor_list = Vec::new();
        for ctor in &type_val.constructors {
            let mut ctor_map: ImblHashMap<GcMapKey, GcValue> = ImblHashMap::new();
            let cname = self.heap.alloc_string(ctor.name.clone());
            ctor_map.insert(GcMapKey::String("name".to_string()), GcValue::String(cname));

            // Constructor fields
            let mut cfield_list = Vec::new();
            for cfield in &ctor.fields {
                let mut cfield_map: ImblHashMap<GcMapKey, GcValue> = ImblHashMap::new();
                let cfname = self.heap.alloc_string(cfield.name.clone());
                cfield_map.insert(GcMapKey::String("name".to_string()), GcValue::String(cfname));
                let cftype = self.heap.alloc_string(cfield.type_name.clone());
                cfield_map.insert(GcMapKey::String("type".to_string()), GcValue::String(cftype));
                let cfield_map_ptr = self.heap.alloc_map(cfield_map);
                cfield_list.push(GcValue::Map(cfield_map_ptr));
            }
            ctor_map.insert(GcMapKey::String("fields".to_string()), GcValue::List(GcList::from_vec(cfield_list)));

            let ctor_map_ptr = self.heap.alloc_map(ctor_map);
            ctor_list.push(GcValue::Map(ctor_map_ptr));
        }
        entries.insert(GcMapKey::String("constructors".to_string()), GcValue::List(GcList::from_vec(ctor_list)));

        // type_params (for parameterized types like Option[T], Result[T, E])
        if !type_val.type_params.is_empty() {
            let param_list: Vec<GcValue> = type_val.type_params.iter()
                .map(|p| {
                    let s = self.heap.alloc_string(p.clone());
                    GcValue::String(s)
                })
                .collect();
            entries.insert(GcMapKey::String("type_params".to_string()), GcValue::List(GcList::from_vec(param_list)));
        }

        let ptr = self.heap.alloc_map(entries);
        Ok(GcValue::Map(ptr))
    }

    /// Construct a typed value from Json (for construct builtin)
    fn construct_from_json(&mut self, type_name: &str, json: GcValue) -> Result<GcValue, RuntimeError> {
        // Look up type info
        let type_info = self.shared.types.read().unwrap().get(type_name).cloned()
            .or_else(|| self.shared.dynamic_types.read().unwrap().get(type_name).cloned())
            .or_else(|| self.shared.stdlib_types.read().unwrap().get(type_name).cloned());

        let type_val = match type_info {
            Some(t) => t,
            None => return Err(RuntimeError::Panic(format!("Unknown type: {}", type_name))),
        };

        // Extract Json object pairs from the json value
        let pairs = self.extract_json_object_pairs(&json)?;

        if type_val.constructors.is_empty() {
            // Record type - fields are directly in the json object
            let mut field_values = Vec::new();
            let mut is_float = Vec::new();

            for field in &type_val.fields {
                let field_json = pairs.iter()
                    .find(|(k, _)| k == &field.name)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| RuntimeError::Panic(format!("Missing field: {}", field.name)))?;
                let field_value = self.json_to_primitive(&field_json, &field.type_name)?;
                is_float.push(field.type_name == "Float" || field.type_name == "Float64" || field.type_name == "Float32");
                field_values.push(field_value);
            }

            let rec_ptr = self.heap.alloc_record(
                type_name.to_string(),
                type_val.fields.iter().map(|f| f.name.clone()).collect(),
                field_values,
                is_float,
            );
            Ok(GcValue::Record(rec_ptr))
        } else {
            // Variant type - json object has a single key (constructor name) with fields object
            if pairs.len() != 1 {
                return Err(RuntimeError::Panic(format!(
                    "Variant json must have exactly one key (constructor), found {}",
                    pairs.len()
                )));
            }

            let (ctor_name, fields_json) = &pairs[0];

            // Find the constructor
            let ctor = type_val.constructors.iter()
                .find(|c| &c.name == ctor_name)
                .ok_or_else(|| RuntimeError::Panic(format!("Unknown constructor: {}", ctor_name)))?;

            if ctor.fields.is_empty() {
                // Unit variant
                let var_ptr = self.heap.alloc_variant(
                    Arc::new(type_name.to_string()),
                    Arc::new(ctor_name.clone()),
                    vec![],
                );
                Ok(GcValue::Variant(var_ptr))
            } else if ctor.fields.len() == 1 {
                // Single-field variant: value is directly the field value (no _0 wrapper)
                let field = &ctor.fields[0];
                let field_value = self.json_to_primitive(fields_json, &field.type_name)?;
                let var_ptr = self.heap.alloc_variant(
                    Arc::new(type_name.to_string()),
                    Arc::new(ctor_name.clone()),
                    vec![field_value],
                );
                Ok(GcValue::Variant(var_ptr))
            } else {
                // Multi-field variant: uses array [val0, val1, ...]
                let array_items = self.extract_json_array(fields_json)?;

                if array_items.len() != ctor.fields.len() {
                    return Err(RuntimeError::Panic(format!(
                        "Expected {} fields for {}, got {}",
                        ctor.fields.len(), ctor_name, array_items.len()
                    )));
                }

                let mut field_values = Vec::new();
                for (i, field) in ctor.fields.iter().enumerate() {
                    let field_value = self.json_to_primitive(&array_items[i], &field.type_name)?;
                    field_values.push(field_value);
                }

                let var_ptr = self.heap.alloc_variant(
                    Arc::new(type_name.to_string()),
                    Arc::new(ctor_name.clone()),
                    field_values,
                );
                Ok(GcValue::Variant(var_ptr))
            }
        }
    }

    /// Extract key-value pairs from a Json Object variant
    fn extract_json_object_pairs(&self, json: &GcValue) -> Result<Vec<(String, GcValue)>, RuntimeError> {
        match json {
            GcValue::Variant(var_ptr) => {
                let variant = self.heap.get_variant(var_ptr.clone())
                    .ok_or_else(|| RuntimeError::Panic("Invalid variant".to_string()))?;

                if variant.constructor.as_ref() != "Object" {
                    return Err(RuntimeError::Panic(format!(
                        "Expected Json Object, found {}",
                        variant.constructor
                    )));
                }

                if variant.fields.is_empty() {
                    return Ok(vec![]);
                }

                match &variant.fields[0] {
                    GcValue::List(list) => {
                        let mut pairs = Vec::new();
                        for item in list.data.iter() {
                            match item {
                                GcValue::Tuple(tup_ptr) => {
                                    let tup = self.heap.get_tuple(tup_ptr.clone())
                                        .ok_or_else(|| RuntimeError::Panic("Invalid tuple".to_string()))?;
                                    if tup.items.len() >= 2 {
                                        let key = match &tup.items[0] {
                                            GcValue::String(s) => self.heap.get_string(s.clone())
                                                .map(|st| st.data.clone())
                                                .unwrap_or_default(),
                                            _ => return Err(RuntimeError::Panic("Object key must be string".to_string())),
                                        };
                                        pairs.push((key, tup.items[1].clone()));
                                    }
                                }
                                _ => return Err(RuntimeError::Panic("Object entry must be tuple".to_string())),
                            }
                        }
                        Ok(pairs)
                    }
                    _ => Err(RuntimeError::Panic("Object fields must be list".to_string())),
                }
            }
            _ => Err(RuntimeError::Panic("Expected Json variant".to_string())),
        }
    }

    /// Extract items from a Json Array variant
    fn extract_json_array(&self, json: &GcValue) -> Result<Vec<GcValue>, RuntimeError> {
        match json {
            GcValue::Variant(var_ptr) => {
                let variant = self.heap.get_variant(var_ptr.clone())
                    .ok_or_else(|| RuntimeError::Panic("Invalid variant".to_string()))?;

                if variant.constructor.as_ref() != "Array" {
                    return Err(RuntimeError::Panic(format!(
                        "Expected Json Array, found {}",
                        variant.constructor
                    )));
                }

                if variant.fields.is_empty() {
                    return Ok(vec![]);
                }

                match &variant.fields[0] {
                    GcValue::List(list) => Ok(list.items().to_vec()),
                    _ => Err(RuntimeError::Panic("Array field must be list".to_string())),
                }
            }
            _ => Err(RuntimeError::Panic("Expected Json variant".to_string())),
        }
    }

    /// Convert a Json value to a primitive GcValue based on expected type
    fn json_to_primitive(&mut self, json: &GcValue, expected_type: &str) -> Result<GcValue, RuntimeError> {
        match json {
            GcValue::Variant(var_ptr) => {
                let variant = self.heap.get_variant(var_ptr.clone())
                    .ok_or_else(|| RuntimeError::Panic("Invalid variant".to_string()))?;

                let ctor = variant.constructor.as_str();
                if ctor == "Null" {
                    Ok(GcValue::Unit)
                } else if ctor == "Bool" {
                    if variant.fields.is_empty() {
                        Ok(GcValue::Bool(false))
                    } else {
                        match &variant.fields[0] {
                            GcValue::Bool(b) => Ok(GcValue::Bool(*b)),
                            _ => Ok(GcValue::Bool(false)),
                        }
                    }
                } else if ctor == "Number" {
                    if variant.fields.is_empty() {
                        Ok(GcValue::Float64(0.0))
                    } else {
                        match &variant.fields[0] {
                            GcValue::Float64(f) => {
                                match expected_type {
                                    "Int" | "Int64" => Ok(GcValue::Int64(*f as i64)),
                                    "Int32" => Ok(GcValue::Int32(*f as i32)),
                                    "Int16" => Ok(GcValue::Int16(*f as i16)),
                                    "Int8" => Ok(GcValue::Int8(*f as i8)),
                                    "UInt64" => Ok(GcValue::UInt64(*f as u64)),
                                    "UInt32" => Ok(GcValue::UInt32(*f as u32)),
                                    "UInt16" => Ok(GcValue::UInt16(*f as u16)),
                                    "UInt8" => Ok(GcValue::UInt8(*f as u8)),
                                    "Float32" => Ok(GcValue::Float32(*f as f32)),
                                    _ => Ok(GcValue::Float64(*f)),
                                }
                            }
                            GcValue::Int64(n) => Ok(GcValue::Int64(*n)),
                            _ => Ok(GcValue::Float64(0.0)),
                        }
                    }
                } else if ctor == "String" {
                    if variant.fields.is_empty() {
                        Ok(GcValue::String(self.heap.alloc_string(String::new())))
                    } else {
                        match &variant.fields[0] {
                            GcValue::String(s) => Ok(GcValue::String(s.clone())),
                            _ => Ok(GcValue::String(self.heap.alloc_string(String::new()))),
                        }
                    }
                } else if ctor == "Array" {
                    if variant.fields.is_empty() {
                        // Check if expected type is a tuple
                        if expected_type.starts_with('(') && expected_type.ends_with(')') {
                            let tup_ptr = self.heap.alloc_tuple(vec![]);
                            Ok(GcValue::Tuple(tup_ptr))
                        } else {
                            Ok(GcValue::List(GcList::from_vec(vec![])))
                        }
                    } else {
                        match &variant.fields[0] {
                            GcValue::List(list) => {
                                // Check if expected type is a tuple like (Int, String, Bool)
                                if expected_type.starts_with('(') && expected_type.ends_with(')') {
                                    // Parse tuple element types
                                    let inner = &expected_type[1..expected_type.len()-1];
                                    let element_types: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                                    let list_items: Vec<GcValue> = list.data.iter().cloned().collect();

                                    let mut items = Vec::new();
                                    for (i, item) in list_items.iter().enumerate() {
                                        let elem_type = element_types.get(i).copied().unwrap_or("a");
                                        items.push(self.json_to_primitive(item, elem_type)?);
                                    }
                                    let tup_ptr = self.heap.alloc_tuple(items);
                                    Ok(GcValue::Tuple(tup_ptr))
                                } else {
                                    // Recursively convert each item as list - clone data to avoid borrow conflict
                                    let inner_type = if expected_type.starts_with("List[") {
                                        expected_type[5..expected_type.len()-1].to_string()
                                    } else {
                                        "a".to_string()
                                    };
                                    let list_items: Vec<GcValue> = list.data.iter().cloned().collect();
                                    let mut items = Vec::new();
                                    for item in list_items {
                                        items.push(self.json_to_primitive(&item, &inner_type)?);
                                    }
                                    Ok(GcValue::List(GcList::from_vec(items)))
                                }
                            }
                            _ => Ok(GcValue::List(GcList::from_vec(vec![]))),
                        }
                    }
                } else if ctor == "Object" {
                    // Check if this is a nested record or variant
                    let _pairs = self.extract_json_object_pairs(json)?;

                    // Try to look up as a type
                    let type_info = self.shared.types.read().unwrap().get(expected_type).cloned()
                        .or_else(|| self.shared.dynamic_types.read().unwrap().get(expected_type).cloned())
                        .or_else(|| self.shared.stdlib_types.read().unwrap().get(expected_type).cloned());

                    if type_info.is_some() {
                        // Recursively construct the nested type
                        return self.construct_from_json(expected_type, json.clone());
                    }

                    // Unknown object type - return as-is
                    Ok(json.clone())
                } else {
                    Ok(json.clone())
                }
            }
            _ => Ok(json.clone()),
        }
    }

    /// Construct a record from a Map[String, Json] (for makeRecord builtin)
    fn make_record_from_map(&mut self, type_name: &str, fields_map: GcValue) -> Result<GcValue, RuntimeError> {
        use crate::gc::GcMapKey;

        // Look up type info
        let type_info = self.shared.types.read().unwrap().get(type_name).cloned()
            .or_else(|| self.shared.dynamic_types.read().unwrap().get(type_name).cloned())
            .or_else(|| self.shared.stdlib_types.read().unwrap().get(type_name).cloned());

        let type_val = match type_info {
            Some(t) => t,
            None => return Err(RuntimeError::Panic(format!("Unknown type: {}", type_name))),
        };

        // Ensure it's a record type (no constructors)
        if !type_val.constructors.is_empty() {
            return Err(RuntimeError::Panic(format!(
                "makeRecord cannot be used for variant type '{}', use makeVariant instead",
                type_name
            )));
        }

        // Extract entries from the Map
        let map_entries = match &fields_map {
            GcValue::Map(ptr) => {
                let map = self.heap.get_map(ptr.clone())
                    .ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                map.entries.clone()
            }
            _ => return Err(RuntimeError::TypeError {
                expected: "Map".to_string(),
                found: "non-map".to_string(),
            }),
        };

        // Build field values in the correct order
        let mut field_values = Vec::new();
        let mut is_float = Vec::new();

        for field in &type_val.fields {
            let key = GcMapKey::String(field.name.clone());
            let field_json = map_entries.get(&key)
                .ok_or_else(|| RuntimeError::Panic(format!("Missing field: {}", field.name)))?
                .clone();
            let field_value = self.json_to_primitive(&field_json, &field.type_name)?;
            is_float.push(field.type_name == "Float" || field.type_name == "Float64" || field.type_name == "Float32");
            field_values.push(field_value);
        }

        let rec_ptr = self.heap.alloc_record(
            type_name.to_string(),
            type_val.fields.iter().map(|f| f.name.clone()).collect(),
            field_values,
            is_float,
        );
        Ok(GcValue::Record(rec_ptr))
    }

    /// Construct a variant from a Map[String, Json] (for makeVariant builtin)
    fn make_variant_from_map(&mut self, type_name: &str, ctor_name: &str, fields_map: GcValue) -> Result<GcValue, RuntimeError> {
        use crate::gc::GcMapKey;

        // Look up type info
        let type_info = self.shared.types.read().unwrap().get(type_name).cloned()
            .or_else(|| self.shared.dynamic_types.read().unwrap().get(type_name).cloned())
            .or_else(|| self.shared.stdlib_types.read().unwrap().get(type_name).cloned());

        let type_val = match type_info {
            Some(t) => t,
            None => return Err(RuntimeError::Panic(format!("Unknown type: {}", type_name))),
        };

        // Find the constructor
        let ctor = type_val.constructors.iter()
            .find(|c| c.name == ctor_name)
            .ok_or_else(|| RuntimeError::Panic(format!("Unknown constructor '{}' for type '{}'", ctor_name, type_name)))?
            .clone();

        if ctor.fields.is_empty() {
            // Unit variant - no fields needed
            let var_ptr = self.heap.alloc_variant(
                Arc::new(type_name.to_string()),
                Arc::new(ctor_name.to_string()),
                vec![],
            );
            return Ok(GcValue::Variant(var_ptr));
        }

        // Extract entries from the Map
        let map_entries = match &fields_map {
            GcValue::Map(ptr) => {
                let map = self.heap.get_map(ptr.clone())
                    .ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                map.entries.clone()
            }
            _ => return Err(RuntimeError::TypeError {
                expected: "Map".to_string(),
                found: "non-map".to_string(),
            }),
        };

        // Build field values in the correct order
        let mut field_values = Vec::new();

        for field in &ctor.fields {
            let key = GcMapKey::String(field.name.clone());
            let field_json = map_entries.get(&key)
                .ok_or_else(|| RuntimeError::Panic(format!("Missing field '{}' for constructor '{}'", field.name, ctor_name)))?
                .clone();
            let field_value = self.json_to_primitive(&field_json, &field.type_name)?;
            field_values.push(field_value);
        }

        let var_ptr = self.heap.alloc_variant(
            Arc::new(type_name.to_string()),
            Arc::new(ctor_name.to_string()),
            field_values,
        );
        Ok(GcValue::Variant(var_ptr))
    }
}

/// Result of executing one instruction.
pub enum StepResult {
    /// Continue execution.
    Continue,
    /// Process finished with a value.
    Finished(GcValue),
}

/// Configuration for the async VM.
#[derive(Clone, Debug)]
pub struct AsyncConfig {
    /// Number of threads for work distribution.
    pub num_threads: usize,
    /// Reductions per yield for fairness.
    pub reductions_per_yield: usize,
    /// Enable function call profiling.
    pub profiling_enabled: bool,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            reductions_per_yield: REDUCTIONS_PER_YIELD,
            profiling_enabled: false,
        }
    }
}

/// The async VM entry point.
pub struct AsyncVM {
    /// Shared state.
    shared: Arc<AsyncSharedState>,
    /// IO runtime.
    #[allow(dead_code)]
    io_runtime: Option<IoRuntime>,
    /// Config.
    #[allow(dead_code)]
    config: AsyncConfig,
}

impl AsyncVM {
    /// Create a new async VM with the given configuration.
    pub fn new(config: AsyncConfig) -> Self {
        // Create IO runtime for async operations
        let io_runtime = IoRuntime::new();
        let io_sender = io_runtime.request_sender();

        let shared = Arc::new(AsyncSharedState {
            functions: RwLock::new(HashMap::new()),
            function_list: RwLock::new(Vec::new()),
            natives: HashMap::new(),
            types: RwLock::new(HashMap::new()),
            jit_int_functions: RwLock::new(HashMap::new()),
            jit_int_functions_0: RwLock::new(HashMap::new()),
            jit_int_functions_2: RwLock::new(HashMap::new()),
            jit_int_functions_3: RwLock::new(HashMap::new()),
            jit_int_functions_4: RwLock::new(HashMap::new()),
            jit_loop_array_functions: RwLock::new(HashMap::new()),
            jit_array_fill_functions: RwLock::new(HashMap::new()),
            jit_array_sum_functions: RwLock::new(HashMap::new()),
            shutdown: AtomicBool::new(false),
            interrupt: AtomicBool::new(false),
            process_registry: TokioRwLock::new(HashMap::new()),
            mvars: HashMap::new(),
            dynamic_mvars: Arc::new(RwLock::new(HashMap::new())),
            dynamic_functions: Arc::new(RwLock::new(HashMap::new())),
            dynamic_types: Arc::new(RwLock::new(HashMap::new())),
            stdlib_functions: Arc::new(RwLock::new(HashMap::new())),
            stdlib_types: Arc::new(RwLock::new(HashMap::new())),
            prelude_imports: Arc::new(RwLock::new(HashMap::new())),
            stdlib_function_list: Arc::new(RwLock::new(Vec::new())),
            eval_callback: Arc::new(RwLock::new(None)),
            io_sender: Some(io_sender),
            inspect_sender: None,
            output_sender: None,
            panel_command_sender: None,
            next_pid: AtomicU64::new(1),
            profiling_enabled: config.profiling_enabled,
        });

        Self {
            shared,
            io_runtime: Some(io_runtime),
            config,
        }
    }

    /// Register a function - safe to call during concurrent evals.
    pub fn register_function(&mut self, name: &str, func: Arc<FunctionValue>) {
        self.shared.functions.write().unwrap()
            .insert(name.to_string(), func);
    }

    /// Set the function list for indexed calls - safe during concurrent evals.
    pub fn set_function_list(&mut self, functions: Vec<Arc<FunctionValue>>) {
        *self.shared.function_list.write().unwrap() = functions;
    }

    /// Register a native function.
    pub fn register_native(&mut self, name: &str, native: Arc<GcNativeFn>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .natives
            .insert(name.to_string(), native);
    }

    /// Register a type - safe during concurrent evals.
    pub fn register_type(&mut self, name: &str, type_val: Arc<TypeValue>) {
        self.shared.types.write().unwrap()
            .insert(name.to_string(), type_val);
    }

    /// Register an mvar with initial value.
    pub fn register_mvar(&mut self, name: &str, initial_value: ThreadSafeValue) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .mvars
            .insert(name.to_string(), Arc::new(TokioRwLock::new(initial_value)));
    }

    /// Register a JIT-compiled function (arity 0) - safe during concurrent evals.
    pub fn register_jit_int_function_0(&mut self, func_index: u16, jit_fn: crate::shared_types::JitIntFn0) {
        self.shared.jit_int_functions_0.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 1) - safe during concurrent evals.
    pub fn register_jit_int_function(&mut self, func_index: u16, jit_fn: crate::shared_types::JitIntFn) {
        self.shared.jit_int_functions.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 2) - safe during concurrent evals.
    pub fn register_jit_int_function_2(&mut self, func_index: u16, jit_fn: crate::shared_types::JitIntFn2) {
        self.shared.jit_int_functions_2.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 3) - safe during concurrent evals.
    pub fn register_jit_int_function_3(&mut self, func_index: u16, jit_fn: crate::shared_types::JitIntFn3) {
        self.shared.jit_int_functions_3.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 4) - safe during concurrent evals.
    pub fn register_jit_int_function_4(&mut self, func_index: u16, jit_fn: crate::shared_types::JitIntFn4) {
        self.shared.jit_int_functions_4.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT loop array function - safe during concurrent evals.
    pub fn register_jit_loop_array_function(&mut self, func_index: u16, jit_fn: crate::shared_types::JitLoopArrayFn) {
        self.shared.jit_loop_array_functions.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT recursive array fill function (arity 2) - safe during concurrent evals.
    pub fn register_jit_array_fill_function(&mut self, func_index: u16, jit_fn: crate::shared_types::JitArrayFillFn) {
        self.shared.jit_array_fill_functions.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT recursive array sum function (arity 3) - safe during concurrent evals.
    pub fn register_jit_array_sum_function(&mut self, func_index: u16, jit_fn: crate::shared_types::JitArraySumFn) {
        self.shared.jit_array_sum_functions.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Get dynamic functions (for REPL).
    pub fn get_dynamic_functions(&self) -> Arc<RwLock<HashMap<String, Arc<FunctionValue>>>> {
        self.shared.dynamic_functions.clone()
    }

    /// Get stdlib functions.
    pub fn get_stdlib_functions(&self) -> Arc<RwLock<HashMap<String, Arc<FunctionValue>>>> {
        self.shared.stdlib_functions.clone()
    }

    /// Set stdlib functions.
    pub fn set_stdlib_functions(&self, functions: HashMap<String, Arc<FunctionValue>>) {
        *self.shared.stdlib_functions.write().unwrap() = functions;
    }

    /// Get dynamic types.
    pub fn get_dynamic_types(&self) -> Arc<RwLock<HashMap<String, Arc<TypeValue>>>> {
        self.shared.dynamic_types.clone()
    }

    /// Get dynamic mvars.
    pub fn get_dynamic_mvars(&self) -> Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>> {
        self.shared.dynamic_mvars.clone()
    }

    /// Set dynamic mvars.
    pub fn set_dynamic_mvars(&mut self, mvars: Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot set dynamic mvars after execution started")
            .dynamic_mvars = mvars;
    }

    /// Get stdlib types.
    pub fn get_stdlib_types(&self) -> Arc<RwLock<HashMap<String, Arc<TypeValue>>>> {
        self.shared.stdlib_types.clone()
    }

    /// Set stdlib types.
    pub fn set_stdlib_types(&self, types: HashMap<String, Arc<TypeValue>>) {
        *self.shared.stdlib_types.write().unwrap() = types;
    }

    /// Get prelude imports.
    pub fn get_prelude_imports(&self) -> Arc<RwLock<HashMap<String, String>>> {
        self.shared.prelude_imports.clone()
    }

    /// Set prelude imports.
    pub fn set_prelude_imports(&self, imports: HashMap<String, String>) {
        *self.shared.prelude_imports.write().unwrap() = imports;
    }

    /// Get stdlib function list (ordered names for CallDirect).
    pub fn get_stdlib_function_list(&self) -> Arc<RwLock<Vec<String>>> {
        self.shared.stdlib_function_list.clone()
    }

    /// Set stdlib function list.
    pub fn set_stdlib_function_list(&self, names: Vec<String>) {
        *self.shared.stdlib_function_list.write().unwrap() = names;
    }

    /// Set the interrupt flag to stop execution (for Ctrl+C handling).
    /// This will cause the VM to return Interrupted error within ~100 instructions.
    pub fn interrupt(&self) {
        self.shared.interrupt.store(true, Ordering::SeqCst);
    }

    /// Clear the interrupt flag (call before starting new execution).
    pub fn clear_interrupt(&self) {
        self.shared.interrupt.store(false, Ordering::SeqCst);
    }

    /// Check if the interrupt flag is set.
    pub fn is_interrupted(&self) -> bool {
        self.shared.interrupt.load(Ordering::SeqCst)
    }

    /// Get a clone of the shared state Arc for external interrupt handling.
    /// This allows setting the interrupt flag from another thread (e.g., Ctrl+C handler).
    pub fn get_interrupt_handle(&self) -> Arc<AsyncSharedState> {
        Arc::clone(&self.shared)
    }

    /// Setup inspect channel and register inspect native function.
    /// Returns a receiver that will receive InspectEntry messages.
    pub fn setup_inspect(&mut self) -> crate::shared_types::InspectReceiver {
        let (sender, receiver) = crossbeam::channel::unbounded();

        Arc::get_mut(&mut self.shared)
            .expect("Cannot setup inspect after execution started")
            .inspect_sender = Some(sender.clone());

        // Register the inspect native function
        self.register_native("inspect", Arc::new(GcNativeFn {
            name: "inspect".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let name = match &args[1] {
                    GcValue::String(ptr) => {
                        heap.get_string(*ptr).map(|s| s.data.clone()).unwrap_or_else(|| String::new())
                    }
                    _ => return Err(RuntimeError::Panic("inspect: second argument must be a string".to_string())),
                };
                let value = crate::process::ThreadSafeValue::from_gc_value(&args[0], heap)
                    .unwrap_or(crate::process::ThreadSafeValue::Unit);
                let entry = crate::shared_types::InspectEntry { name, value };
                let _ = sender.send(entry);
                Ok(GcValue::Unit)
            }),
        }));

        receiver
    }

    /// Setup output channel for println.
    /// Returns a receiver that will receive output strings.
    pub fn setup_output(&mut self) -> crate::shared_types::OutputReceiver {
        let (sender, receiver) = crossbeam::channel::unbounded();

        Arc::get_mut(&mut self.shared)
            .expect("Cannot setup output after execution started")
            .output_sender = Some(sender);

        receiver
    }

    /// Setup panel channel and register Panel.* native functions.
    /// Returns a receiver that will receive PanelCommand messages.
    pub fn setup_panel(&mut self) -> crate::shared_types::PanelCommandReceiver {
        use std::sync::atomic::AtomicU64;
        let (sender, receiver) = crossbeam::channel::unbounded();

        Arc::get_mut(&mut self.shared)
            .expect("Cannot setup panel after execution started")
            .panel_command_sender = Some(sender.clone());

        let next_panel_id = Arc::new(AtomicU64::new(1));

        // Helper to extract string from GcValue
        fn get_string(val: &GcValue, heap: &Heap, name: &str) -> Result<String, RuntimeError> {
            match val {
                GcValue::String(ptr) => {
                    heap.get_string(*ptr)
                        .map(|s| s.data.clone())
                        .ok_or_else(|| RuntimeError::Panic(format!("{}: invalid string pointer", name)))
                }
                _ => Err(RuntimeError::Panic(format!("{}: expected string", name))),
            }
        }

        // Panel.create(title: String) -> Int (panel ID)
        let sender_create = sender.clone();
        let next_id = next_panel_id.clone();
        self.register_native("Panel.create", Arc::new(GcNativeFn {
            name: "Panel.create".to_string(),
            arity: 1,
            func: Box::new(move |args, heap| {
                let title = get_string(&args[0], heap, "Panel.create")?;
                let id = next_id.fetch_add(1, Ordering::SeqCst);
                let _ = sender_create.send(crate::shared_types::PanelCommand::Create { id, title });
                Ok(GcValue::Int64(id as i64))
            }),
        }));

        // Panel.setContent(id: Int, content: String) -> ()
        let sender_content = sender.clone();
        self.register_native("Panel.setContent", Arc::new(GcNativeFn {
            name: "Panel.setContent".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let id = match &args[0] {
                    GcValue::Int64(n) => *n as u64,
                    _ => return Err(RuntimeError::Panic("Panel.setContent: expected int".to_string())),
                };
                let content = get_string(&args[1], heap, "Panel.setContent")?;
                let _ = sender_content.send(crate::shared_types::PanelCommand::SetContent { id, content });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.show(id: Int) -> ()
        let sender_show = sender.clone();
        self.register_native("Panel.show", Arc::new(GcNativeFn {
            name: "Panel.show".to_string(),
            arity: 1,
            func: Box::new(move |args, _heap| {
                let id = match &args[0] {
                    GcValue::Int64(n) => *n as u64,
                    _ => return Err(RuntimeError::Panic("Panel.show: expected int".to_string())),
                };
                let _ = sender_show.send(crate::shared_types::PanelCommand::Show { id });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.hide(id: Int) -> ()
        let sender_hide = sender.clone();
        self.register_native("Panel.hide", Arc::new(GcNativeFn {
            name: "Panel.hide".to_string(),
            arity: 1,
            func: Box::new(move |args, _heap| {
                let id = match &args[0] {
                    GcValue::Int64(n) => *n as u64,
                    _ => return Err(RuntimeError::Panic("Panel.hide: expected int".to_string())),
                };
                let _ = sender_hide.send(crate::shared_types::PanelCommand::Hide { id });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.onKey(id: Int, handler: String) -> ()
        let sender_onkey = sender.clone();
        self.register_native("Panel.onKey", Arc::new(GcNativeFn {
            name: "Panel.onKey".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let id = match &args[0] {
                    GcValue::Int64(n) => *n as u64,
                    _ => return Err(RuntimeError::Panic("Panel.onKey: expected int".to_string())),
                };
                let handler_fn = get_string(&args[1], heap, "Panel.onKey")?;
                let _ = sender_onkey.send(crate::shared_types::PanelCommand::OnKey { id, handler_fn });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.registerHotkey(key: String, callback: String) -> ()
        let sender_hotkey = sender.clone();
        self.register_native("Panel.registerHotkey", Arc::new(GcNativeFn {
            name: "Panel.registerHotkey".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let key = get_string(&args[0], heap, "Panel.registerHotkey")?;
                let callback_fn = get_string(&args[1], heap, "Panel.registerHotkey")?;
                let _ = sender_hotkey.send(crate::shared_types::PanelCommand::RegisterHotkey { key, callback_fn });
                Ok(GcValue::Unit)
            }),
        }));

        receiver
    }

    /// Get a function by name.
    pub fn get_function(&self, name: &str) -> Option<Arc<FunctionValue>> {
        self.shared.functions.read().unwrap().get(name).cloned()
    }

    /// Setup eval callback for REPL.
    pub fn setup_eval(&mut self) {
        // Empty for now - async VM doesn't need special setup
    }

    /// Set eval callback.
    pub fn set_eval_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        *self.shared.eval_callback.write().unwrap() = Some(Arc::new(callback));
    }

    /// Register default native functions.
    pub fn register_default_natives(&mut self) {
        // show - convert value to string
        self.register_native("show", Arc::new(GcNativeFn {
            name: "show".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                Ok(GcValue::String(heap.alloc_string(s)))
            }),
        }));

        // copy - deep copy a value
        self.register_native("copy", Arc::new(GcNativeFn {
            name: "copy".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                Ok(heap.clone_value(&args[0]))
            }),
        }));

        // print - print without newline
        self.register_native("print", Arc::new(GcNativeFn {
            name: "print".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                print!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // println - print with newline
        self.register_native("println", Arc::new(GcNativeFn {
            name: "println".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                println!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // String.length
        self.register_native("String.length", Arc::new(GcNativeFn {
            name: "String.length".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::Int64(str_val.data.chars().count() as i64))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // hash - compute hash of any value
        self.register_native("hash", Arc::new(GcNativeFn {
            name: "hash".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                fn hash_value(val: &GcValue, heap: &Heap) -> Result<u64, RuntimeError> {
                    const FNV_OFFSET: u64 = 14695981039346656037;
                    const FNV_PRIME: u64 = 1099511628211;

                    fn fnv1a_hash(bytes: &[u8]) -> u64 {
                        let mut hash: u64 = FNV_OFFSET;
                        for byte in bytes {
                            hash ^= *byte as u64;
                            hash = hash.wrapping_mul(FNV_PRIME);
                        }
                        hash
                    }

                    fn combine_hash(h1: u64, h2: u64) -> u64 {
                        h1.wrapping_mul(FNV_PRIME) ^ h2
                    }

                    fn hash_gc_map_key(key: &crate::gc::GcMapKey) -> u64 {
                        use crate::gc::GcMapKey;
                        match key {
                            GcMapKey::Unit => 0,
                            GcMapKey::Bool(b) => if *b { 1 } else { 0 },
                            GcMapKey::Char(c) => fnv1a_hash(&(*c as u32).to_le_bytes()),
                            GcMapKey::Int8(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::Int16(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::Int32(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::Int64(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::UInt8(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::UInt16(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::UInt32(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::UInt64(n) => fnv1a_hash(&n.to_le_bytes()),
                            GcMapKey::String(s) => fnv1a_hash(s.as_bytes()),
                            GcMapKey::Record { type_name, field_names, fields } => {
                                let mut h = fnv1a_hash(type_name.as_bytes());
                                for name in field_names {
                                    h = combine_hash(h, fnv1a_hash(name.as_bytes()));
                                }
                                for field in fields {
                                    h = combine_hash(h, hash_gc_map_key(field));
                                }
                                h
                            }
                            GcMapKey::Variant { type_name, constructor, fields } => {
                                let mut h = fnv1a_hash(type_name.as_bytes());
                                h = combine_hash(h, fnv1a_hash(constructor.as_bytes()));
                                for field in fields {
                                    h = combine_hash(h, hash_gc_map_key(field));
                                }
                                h
                            }
                        }
                    }

                    match val {
                        GcValue::Unit => Ok(0),
                        GcValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
                        GcValue::Char(c) => Ok(fnv1a_hash(&(*c as u32).to_le_bytes())),
                        GcValue::Int8(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::Int16(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::Int32(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::Int64(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::UInt8(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::UInt16(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::UInt32(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::UInt64(n) => Ok(fnv1a_hash(&n.to_le_bytes())),
                        GcValue::Float32(f) => Ok(fnv1a_hash(&f.to_le_bytes())),
                        GcValue::Float64(f) => Ok(fnv1a_hash(&f.to_le_bytes())),
                        GcValue::String(s) => {
                            if let Some(str_val) = heap.get_string(*s) {
                                Ok(fnv1a_hash(str_val.data.as_bytes()))
                            } else {
                                Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                            }
                        }
                        GcValue::List(list) => {
                            let mut h = fnv1a_hash(b"list");
                            for item in list.items() {
                                h = combine_hash(h, hash_value(&item, heap)?);
                            }
                            Ok(h)
                        }
                        GcValue::Tuple(ptr) => {
                            if let Some(tuple) = heap.get_tuple(*ptr) {
                                let mut h = fnv1a_hash(b"tuple");
                                for item in &tuple.items {
                                    h = combine_hash(h, hash_value(&item, heap)?);
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid tuple pointer".to_string()))
                            }
                        }
                        GcValue::Record(ptr) => {
                            if let Some(rec) = heap.get_record(*ptr) {
                                let mut h = fnv1a_hash(rec.type_name.as_bytes());
                                for field in &rec.fields {
                                    h = combine_hash(h, hash_value(field, heap)?);
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid record pointer".to_string()))
                            }
                        }
                        GcValue::Variant(ptr) => {
                            if let Some(var) = heap.get_variant(*ptr) {
                                let mut h = fnv1a_hash(var.type_name.as_bytes());
                                h = combine_hash(h, fnv1a_hash(var.constructor.as_bytes()));
                                for field in &var.fields {
                                    h = combine_hash(h, hash_value(field, heap)?);
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid variant pointer".to_string()))
                            }
                        }
                        GcValue::Map(ptr) => {
                            if let Some(map) = heap.get_map(*ptr) {
                                let mut h: u64 = fnv1a_hash(b"map");
                                for (k, v) in map.entries.iter() {
                                    let kh = hash_gc_map_key(k);
                                    let vh = hash_value(v, heap)?;
                                    h ^= combine_hash(kh, vh);
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid map pointer".to_string()))
                            }
                        }
                        GcValue::Set(ptr) => {
                            if let Some(set) = heap.get_set(*ptr) {
                                let mut h: u64 = fnv1a_hash(b"set");
                                for k in set.items.iter() {
                                    h ^= hash_gc_map_key(k);
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid set pointer".to_string()))
                            }
                        }
                        GcValue::BigInt(ptr) => {
                            if let Some(bi) = heap.get_bigint(*ptr) {
                                Ok(fnv1a_hash(&bi.value.to_signed_bytes_le()))
                            } else {
                                Err(RuntimeError::Panic("Invalid bigint pointer".to_string()))
                            }
                        }
                        GcValue::Decimal(dec) => {
                            Ok(fnv1a_hash(&dec.serialize()))
                        }
                        GcValue::Closure(ptr, _) => Ok(fnv1a_hash(&ptr.as_raw().to_le_bytes())),
                        GcValue::Function(f) => Ok(fnv1a_hash(f.name.as_bytes())),
                        GcValue::NativeFunction(f) => Ok(fnv1a_hash(f.name.as_bytes())),
                        GcValue::Pid(p) => Ok(fnv1a_hash(&p.to_le_bytes())),
                        GcValue::Ref(r) => Ok(fnv1a_hash(&r.to_le_bytes())),
                        GcValue::Int64Array(ptr) => {
                            if let Some(arr) = heap.get_int64_array(*ptr) {
                                let mut h = fnv1a_hash(b"int64array");
                                for n in &arr.items {
                                    h = combine_hash(h, fnv1a_hash(&n.to_le_bytes()));
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid array pointer".to_string()))
                            }
                        }
                        GcValue::Float64Array(ptr) => {
                            if let Some(arr) = heap.get_float64_array(*ptr) {
                                let mut h = fnv1a_hash(b"float64array");
                                for f in &arr.items {
                                    h = combine_hash(h, fnv1a_hash(&f.to_le_bytes()));
                                }
                                Ok(h)
                            } else {
                                Err(RuntimeError::Panic("Invalid array pointer".to_string()))
                            }
                        }
                        _ => Ok(fnv1a_hash(b"unknown")),
                    }
                }

                let hash_val = hash_value(&args[0], heap)?;
                Ok(GcValue::Int64(hash_val as i64))
            }),
        }));

        // === Additional native functions (copied from ParallelVM) ===
        self.register_native("String.chars", Arc::new(GcNativeFn {
            name: "String.chars".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                             let chars: Vec<GcValue> = str_val.data.chars().map(GcValue::Char).collect();
                             Ok(GcValue::List(heap.make_list(chars)))
                        } else {
                             Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.from_chars
        self.register_native("String.from_chars", Arc::new(GcNativeFn {
            name: "String.from_chars".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::List(list) => {
                        let mut s = String::new();
                        for item in list.items() {
                            match item {
                                GcValue::Char(c) => s.push(c),
                                _ => return Err(RuntimeError::TypeError { expected: "Char".to_string(), found: "other".to_string() })
                            }
                        }
                        Ok(GcValue::String(heap.alloc_string(s)))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "List".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.to_int
        self.register_native("String.to_int", Arc::new(GcNativeFn {
            name: "String.to_int".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                             match str_val.data.parse::<i64>() {
                                 Ok(n) => Ok(GcValue::Int64(n)),
                                 Err(_) => Ok(GcValue::Unit)
                             }
                        } else {
                             Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // unwrapOr - unwrap Option or return default
        // Works with stdlib.list.Option specifically
        self.register_native("unwrapOr", Arc::new(GcNativeFn {
            name: "unwrapOr".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Variant(ptr) => {
                        if let Some(var) = heap.get_variant(*ptr) {
                            // Check if it's Some or None from stdlib.list.Option
                            if var.type_name.as_str() == "stdlib.list.Option" {
                                if var.constructor.as_str() == "Some" && !var.fields.is_empty() {
                                    Ok(var.fields[0].clone())
                                } else {
                                    // It's None, return the default value
                                    Ok(args[1].clone())
                                }
                            } else {
                                // Not stdlib.list.Option, just return as-is or default
                                // This handles user-defined Option types by just checking constructor name
                                if var.constructor.as_str() == "Some" && !var.fields.is_empty() {
                                    Ok(var.fields[0].clone())
                                } else {
                                    Ok(args[1].clone())
                                }
                            }
                        } else {
                            Err(RuntimeError::Panic("Invalid variant pointer".to_string()))
                        }
                    }
                    _ => {
                        // Not a variant, return the default
                        Ok(args[1].clone())
                    }
                }
            }),
        }));

        self.register_native("String.toInt", Arc::new(GcNativeFn {
            name: "String.toInt".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let option_type = Arc::new("stdlib.list.Option".to_string());
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            match str_val.data.trim().parse::<i64>() {
                                Ok(n) => {
                                    // Return Some(n)
                                    let ptr = heap.alloc_variant(option_type, Arc::new("Some".to_string()), vec![GcValue::Int64(n)]);
                                    Ok(GcValue::Variant(ptr))
                                }
                                Err(_) => {
                                    // Return None
                                    let ptr = heap.alloc_variant(option_type, Arc::new("None".to_string()), vec![]);
                                    Ok(GcValue::Variant(ptr))
                                }
                            }
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.toFloat
        self.register_native("String.toFloat", Arc::new(GcNativeFn {
            name: "String.toFloat".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let option_type = Arc::new("stdlib.list.Option".to_string());
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            match str_val.data.trim().parse::<f64>() {
                                Ok(n) => {
                                    // Return Some(n)
                                    let ptr = heap.alloc_variant(option_type, Arc::new("Some".to_string()), vec![GcValue::Float64(n)]);
                                    Ok(GcValue::Variant(ptr))
                                }
                                Err(_) => {
                                    // Return None
                                    let ptr = heap.alloc_variant(option_type, Arc::new("None".to_string()), vec![]);
                                    Ok(GcValue::Variant(ptr))
                                }
                            }
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.trim
        self.register_native("String.trim", Arc::new(GcNativeFn {
            name: "String.trim".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::String(heap.alloc_string(str_val.data.trim().to_string())))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.trimStart
        self.register_native("String.trimStart", Arc::new(GcNativeFn {
            name: "String.trimStart".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::String(heap.alloc_string(str_val.data.trim_start().to_string())))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.trimEnd
        self.register_native("String.trimEnd", Arc::new(GcNativeFn {
            name: "String.trimEnd".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::String(heap.alloc_string(str_val.data.trim_end().to_string())))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.toUpper
        self.register_native("String.toUpper", Arc::new(GcNativeFn {
            name: "String.toUpper".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::String(heap.alloc_string(str_val.data.to_uppercase())))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.toLower
        self.register_native("String.toLower", Arc::new(GcNativeFn {
            name: "String.toLower".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::String(heap.alloc_string(str_val.data.to_lowercase())))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.contains
        self.register_native("String.contains", Arc::new(GcNativeFn {
            name: "String.contains".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let sub = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, sub) {
                    (Some(s), Some(sub)) => Ok(GcValue::Bool(s.contains(&sub))),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.startsWith
        self.register_native("String.startsWith", Arc::new(GcNativeFn {
            name: "String.startsWith".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let prefix = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, prefix) {
                    (Some(s), Some(p)) => Ok(GcValue::Bool(s.starts_with(&p))),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.endsWith
        self.register_native("String.endsWith", Arc::new(GcNativeFn {
            name: "String.endsWith".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let suffix = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, suffix) {
                    (Some(s), Some(suf)) => Ok(GcValue::Bool(s.ends_with(&suf))),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.replace
        self.register_native("String.replace", Arc::new(GcNativeFn {
            name: "String.replace".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let from = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let to = match &args[2] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, from, to) {
                    (Some(s), Some(from), Some(to)) => Ok(GcValue::String(heap.alloc_string(s.replacen(&from, &to, 1)))),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.replaceAll
        self.register_native("String.replaceAll", Arc::new(GcNativeFn {
            name: "String.replaceAll".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let from = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let to = match &args[2] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, from, to) {
                    (Some(s), Some(from), Some(to)) => Ok(GcValue::String(heap.alloc_string(s.replace(&from, &to)))),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.indexOf
        self.register_native("String.indexOf", Arc::new(GcNativeFn {
            name: "String.indexOf".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let sub = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, sub) {
                    (Some(s), Some(sub)) => {
                        let idx = s.find(&sub).map(|i| i as i64).unwrap_or(-1);
                        Ok(GcValue::Int64(idx))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.lastIndexOf
        self.register_native("String.lastIndexOf", Arc::new(GcNativeFn {
            name: "String.lastIndexOf".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let sub = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, sub) {
                    (Some(s), Some(sub)) => {
                        let idx = s.rfind(&sub).map(|i| i as i64).unwrap_or(-1);
                        Ok(GcValue::Int64(idx))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.substring
        self.register_native("String.substring", Arc::new(GcNativeFn {
            name: "String.substring".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let start = match &args[1] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let end = match &args[2] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                match s {
                    Some(s) => {
                        let chars: Vec<char> = s.chars().collect();
                        let start = start.min(chars.len());
                        let end = end.min(chars.len());
                        let result: String = chars[start..end].iter().collect();
                        Ok(GcValue::String(heap.alloc_string(result)))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.repeat
        self.register_native("String.repeat", Arc::new(GcNativeFn {
            name: "String.repeat".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let n = match &args[1] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                match s {
                    Some(s) => Ok(GcValue::String(heap.alloc_string(s.repeat(n)))),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.padStart
        self.register_native("String.padStart", Arc::new(GcNativeFn {
            name: "String.padStart".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let len = match &args[1] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let pad = match &args[2] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pad) {
                    (Some(s), Some(pad)) => {
                        if s.chars().count() >= len {
                            Ok(GcValue::String(heap.alloc_string(s)))
                        } else {
                            let pad_len = len - s.chars().count();
                            let pad_char = pad.chars().next().unwrap_or(' ');
                            let padding: String = std::iter::repeat(pad_char).take(pad_len).collect();
                            Ok(GcValue::String(heap.alloc_string(format!("{}{}", padding, s))))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.padEnd
        self.register_native("String.padEnd", Arc::new(GcNativeFn {
            name: "String.padEnd".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let len = match &args[1] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let pad = match &args[2] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pad) {
                    (Some(s), Some(pad)) => {
                        if s.chars().count() >= len {
                            Ok(GcValue::String(heap.alloc_string(s)))
                        } else {
                            let pad_len = len - s.chars().count();
                            let pad_char = pad.chars().next().unwrap_or(' ');
                            let padding: String = std::iter::repeat(pad_char).take(pad_len).collect();
                            Ok(GcValue::String(heap.alloc_string(format!("{}{}", s, padding))))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // String.reverse
        self.register_native("String.reverse", Arc::new(GcNativeFn {
            name: "String.reverse".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::String(heap.alloc_string(str_val.data.chars().rev().collect())))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.lines
        self.register_native("String.lines", Arc::new(GcNativeFn {
            name: "String.lines".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            // Clone data first to avoid borrow issues
                            let data = str_val.data.clone();
                            let line_strs: Vec<String> = data.lines().map(|l| l.to_string()).collect();
                            let lines: Vec<GcValue> = line_strs.into_iter()
                                .map(|l| GcValue::String(heap.alloc_string(l)))
                                .collect();
                            Ok(GcValue::List(heap.make_list(lines)))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.words
        self.register_native("String.words", Arc::new(GcNativeFn {
            name: "String.words".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            // Clone data first to avoid borrow issues
                            let data = str_val.data.clone();
                            let word_strs: Vec<String> = data.split_whitespace().map(|w| w.to_string()).collect();
                            let words: Vec<GcValue> = word_strs.into_iter()
                                .map(|w| GcValue::String(heap.alloc_string(w)))
                                .collect();
                            Ok(GcValue::List(heap.make_list(words)))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // String.isEmpty
        self.register_native("String.isEmpty", Arc::new(GcNativeFn {
            name: "String.isEmpty".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::Bool(str_val.data.is_empty()))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // === Time functions ===

        // Time.now - returns milliseconds since Unix epoch
        self.register_native("Time.now", Arc::new(GcNativeFn {
            name: "Time.now".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use chrono::Utc;
                Ok(GcValue::Int64(Utc::now().timestamp_millis()))
            }),
        }));

        // Time.nowSecs - returns seconds since Unix epoch
        self.register_native("Time.nowSecs", Arc::new(GcNativeFn {
            name: "Time.nowSecs".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use chrono::Utc;
                Ok(GcValue::Int64(Utc::now().timestamp()))
            }),
        }));

        // Time.format - format timestamp with format string
        self.register_native("Time.format", Arc::new(GcNativeFn {
            name: "Time.format".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use chrono::{Local, TimeZone};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let fmt = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match fmt {
                    Some(fmt) => {
                        let dt = Local.timestamp_millis_opt(ts).single()
                            .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                        Ok(GcValue::String(heap.alloc_string(dt.format(&fmt).to_string())))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // Time.formatUtc - format timestamp as UTC
        self.register_native("Time.formatUtc", Arc::new(GcNativeFn {
            name: "Time.formatUtc".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use chrono::{Utc, TimeZone};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let fmt = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match fmt {
                    Some(fmt) => {
                        let dt = Utc.timestamp_millis_opt(ts).single()
                            .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                        Ok(GcValue::String(heap.alloc_string(dt.format(&fmt).to_string())))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // Time.parse - parse time string
        self.register_native("Time.parse", Arc::new(GcNativeFn {
            name: "Time.parse".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use chrono::NaiveDateTime;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let fmt = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, fmt) {
                    (Some(s), Some(fmt)) => {
                        match NaiveDateTime::parse_from_str(&s, &fmt) {
                            Ok(dt) => Ok(GcValue::Int64(dt.and_utc().timestamp_millis())),
                            Err(_) => Ok(GcValue::Unit) // Return None/Unit on parse failure
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // Time component extractors
        self.register_native("Time.year", Arc::new(GcNativeFn {
            name: "Time.year".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Datelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                Ok(GcValue::Int64(dt.year() as i64))
            }),
        }));

        self.register_native("Time.month", Arc::new(GcNativeFn {
            name: "Time.month".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Datelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                Ok(GcValue::Int64(dt.month() as i64))
            }),
        }));

        self.register_native("Time.day", Arc::new(GcNativeFn {
            name: "Time.day".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Datelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                Ok(GcValue::Int64(dt.day() as i64))
            }),
        }));

        self.register_native("Time.hour", Arc::new(GcNativeFn {
            name: "Time.hour".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Timelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                Ok(GcValue::Int64(dt.hour() as i64))
            }),
        }));

        self.register_native("Time.minute", Arc::new(GcNativeFn {
            name: "Time.minute".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Timelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                Ok(GcValue::Int64(dt.minute() as i64))
            }),
        }));

        self.register_native("Time.second", Arc::new(GcNativeFn {
            name: "Time.second".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Timelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                Ok(GcValue::Int64(dt.second() as i64))
            }),
        }));

        self.register_native("Time.weekday", Arc::new(GcNativeFn {
            name: "Time.weekday".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use chrono::{Local, TimeZone, Datelike};
                let ts = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let dt = Local.timestamp_millis_opt(ts).single()
                    .ok_or_else(|| RuntimeError::Panic("Invalid timestamp".to_string()))?;
                // Sunday = 0, Saturday = 6
                Ok(GcValue::Int64(dt.weekday().num_days_from_sunday() as i64))
            }),
        }));

        self.register_native("Time.timezone", Arc::new(GcNativeFn {
            name: "Time.timezone".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                use chrono::Local;
                let now = Local::now();
                let tz_name = now.format("%Z").to_string();
                Ok(GcValue::String(heap.alloc_string(tz_name)))
            }),
        }));

        self.register_native("Time.timezoneOffset", Arc::new(GcNativeFn {
            name: "Time.timezoneOffset".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use chrono::Local;
                let now = Local::now();
                let offset_secs = now.offset().local_minus_utc();
                Ok(GcValue::Int64((offset_secs / 60) as i64)) // Return minutes
            }),
        }));

        self.register_native("Time.toUtc", Arc::new(GcNativeFn {
            name: "Time.toUtc".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                // Timestamps are already in UTC internally, this is a no-op
                match &args[0] {
                    GcValue::Int64(n) => Ok(GcValue::Int64(*n)),
                    _ => Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                }
            }),
        }));

        self.register_native("Time.fromUtc", Arc::new(GcNativeFn {
            name: "Time.fromUtc".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                // Timestamps are already in UTC internally, this is a no-op
                match &args[0] {
                    GcValue::Int64(n) => Ok(GcValue::Int64(*n)),
                    _ => Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // === Random functions ===

        self.register_native("Random.int", Arc::new(GcNativeFn {
            name: "Random.int".to_string(),
            arity: 2,
            func: Box::new(|args, _heap| {
                use rand::Rng;
                let min = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let max = match &args[1] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let mut rng = rand::thread_rng();
                Ok(GcValue::Int64(rng.gen_range(min..=max)))
            }),
        }));

        self.register_native("Random.float", Arc::new(GcNativeFn {
            name: "Random.float".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                Ok(GcValue::Float64(rng.gen::<f64>()))
            }),
        }));

        self.register_native("Random.bool", Arc::new(GcNativeFn {
            name: "Random.bool".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                Ok(GcValue::Bool(rng.gen::<bool>()))
            }),
        }));

        self.register_native("Random.choice", Arc::new(GcNativeFn {
            name: "Random.choice".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                use rand::Rng;
                match &args[0] {
                    GcValue::List(list) => {
                        let items = list.items();
                        if items.is_empty() {
                            return Err(RuntimeError::Panic("Cannot choose from empty list".to_string()));
                        }
                        let mut rng = rand::thread_rng();
                        let idx = rng.gen_range(0..items.len());
                        Ok(items[idx].clone())
                    }
                    _ => Err(RuntimeError::TypeError { expected: "List".to_string(), found: "other".to_string() })
                }
            }),
        }));

        self.register_native("Random.shuffle", Arc::new(GcNativeFn {
            name: "Random.shuffle".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use rand::seq::SliceRandom;
                match &args[0] {
                    GcValue::List(list) => {
                        let mut items: Vec<GcValue> = list.items().to_vec();
                        let mut rng = rand::thread_rng();
                        items.shuffle(&mut rng);
                        Ok(GcValue::List(heap.make_list(items)))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "List".to_string(), found: "other".to_string() })
                }
            }),
        }));

        self.register_native("Random.bytes", Arc::new(GcNativeFn {
            name: "Random.bytes".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use rand::Rng;
                let n = match &args[0] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let mut rng = rand::thread_rng();
                let bytes: Vec<GcValue> = (0..n).map(|_| GcValue::Int64(rng.gen_range(0..=255))).collect();
                Ok(GcValue::List(heap.make_list(bytes)))
            }),
        }));

        // === List utility functions ===

        self.register_native("range", Arc::new(GcNativeFn {
            name: "range".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let start = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let end = match &args[1] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let items: Vec<GcValue> = (start..end).map(GcValue::Int64).collect();
                Ok(GcValue::List(heap.make_list(items)))
            }),
        }));

        self.register_native("product", Arc::new(GcNativeFn {
            name: "product".to_string(),
            arity: 1,
            func: Box::new(|args, _heap| {
                match &args[0] {
                    GcValue::List(list) => {
                        let items = list.items();
                        let mut result: i64 = 1;
                        for item in &items {
                            match item {
                                GcValue::Int64(n) => result *= n,
                                GcValue::Float64(_f) => return Ok(GcValue::Float64(items.iter().fold(1.0, |acc, v| {
                                    match v {
                                        GcValue::Float64(x) => acc * x,
                                        GcValue::Int64(x) => acc * (*x as f64),
                                        _ => acc
                                    }
                                }))),
                                _ => return Err(RuntimeError::TypeError { expected: "Num".to_string(), found: "other".to_string() })
                            }
                        }
                        Ok(GcValue::Int64(result))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "List".to_string(), found: "other".to_string() })
                }
            }),
        }));

        // === Environment functions ===

        self.register_native("Env.get", Arc::new(GcNativeFn {
            name: "Env.get".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let key = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match key {
                    Some(key) => {
                        match std::env::var(&key) {
                            Ok(val) => Ok(GcValue::String(heap.alloc_string(val))),
                            Err(_) => Ok(GcValue::Unit) // Return None/Unit if not found
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Env.set", Arc::new(GcNativeFn {
            name: "Env.set".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let key = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let val = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (key, val) {
                    (Some(key), Some(val)) => {
                        std::env::set_var(&key, &val);
                        Ok(GcValue::Unit)
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Env.remove", Arc::new(GcNativeFn {
            name: "Env.remove".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let key = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match key {
                    Some(key) => {
                        std::env::remove_var(&key);
                        Ok(GcValue::Unit)
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Env.all", Arc::new(GcNativeFn {
            name: "Env.all".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                let vars: Vec<GcValue> = std::env::vars()
                    .map(|(k, v)| {
                        let k_ptr = heap.alloc_string(k);
                        let v_ptr = heap.alloc_string(v);
                        let tuple_ptr = heap.alloc_tuple(vec![GcValue::String(k_ptr), GcValue::String(v_ptr)]);
                        GcValue::Tuple(tuple_ptr)
                    })
                    .collect();
                Ok(GcValue::List(heap.make_list(vars)))
            }),
        }));

        self.register_native("Env.cwd", Arc::new(GcNativeFn {
            name: "Env.cwd".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                match std::env::current_dir() {
                    Ok(path) => Ok(GcValue::String(heap.alloc_string(path.to_string_lossy().to_string()))),
                    Err(e) => Err(RuntimeError::Panic(format!("Failed to get cwd: {}", e)))
                }
            }),
        }));

        self.register_native("Env.setCwd", Arc::new(GcNativeFn {
            name: "Env.setCwd".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let path = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match path {
                    Some(path) => {
                        match std::env::set_current_dir(&path) {
                            Ok(_) => Ok(GcValue::Unit),
                            Err(e) => Err(RuntimeError::Panic(format!("Failed to set cwd: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Env.home", Arc::new(GcNativeFn {
            name: "Env.home".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                match std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
                    Ok(home) => Ok(GcValue::String(heap.alloc_string(home))),
                    Err(_) => Ok(GcValue::Unit) // Return None if not found
                }
            }),
        }));

        self.register_native("Env.args", Arc::new(GcNativeFn {
            name: "Env.args".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                let args: Vec<GcValue> = std::env::args()
                    .map(|a| GcValue::String(heap.alloc_string(a)))
                    .collect();
                Ok(GcValue::List(heap.make_list(args)))
            }),
        }));

        self.register_native("Env.platform", Arc::new(GcNativeFn {
            name: "Env.platform".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                let platform = if cfg!(target_os = "linux") { "linux" }
                    else if cfg!(target_os = "macos") { "macos" }
                    else if cfg!(target_os = "windows") { "windows" }
                    else { "unknown" };
                Ok(GcValue::String(heap.alloc_string(platform.to_string())))
            }),
        }));

        // === Path functions ===

        self.register_native("Path.join", Arc::new(GcNativeFn {
            name: "Path.join".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use std::path::PathBuf;
                let p1 = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let p2 = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (p1, p2) {
                    (Some(p1), Some(p2)) => {
                        let path: PathBuf = [&p1, &p2].iter().collect();
                        Ok(GcValue::String(heap.alloc_string(path.to_string_lossy().to_string())))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.dirname", Arc::new(GcNativeFn {
            name: "Path.dirname".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::Path;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => {
                        let path = Path::new(&p);
                        let parent = path.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
                        Ok(GcValue::String(heap.alloc_string(parent)))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.basename", Arc::new(GcNativeFn {
            name: "Path.basename".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::Path;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => {
                        let path = Path::new(&p);
                        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                        Ok(GcValue::String(heap.alloc_string(name)))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.extension", Arc::new(GcNativeFn {
            name: "Path.extension".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::Path;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => {
                        let path = Path::new(&p);
                        let ext = path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
                        Ok(GcValue::String(heap.alloc_string(ext)))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.withExtension", Arc::new(GcNativeFn {
            name: "Path.withExtension".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use std::path::PathBuf;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let ext = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (p, ext) {
                    (Some(p), Some(ext)) => {
                        let mut path = PathBuf::from(&p);
                        path.set_extension(&ext);
                        Ok(GcValue::String(heap.alloc_string(path.to_string_lossy().to_string())))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.normalize", Arc::new(GcNativeFn {
            name: "Path.normalize".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::PathBuf;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => {
                        let path = PathBuf::from(&p);
                        // Use components to normalize
                        let normalized: PathBuf = path.components().collect();
                        Ok(GcValue::String(heap.alloc_string(normalized.to_string_lossy().to_string())))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.isAbsolute", Arc::new(GcNativeFn {
            name: "Path.isAbsolute".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::Path;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => Ok(GcValue::Bool(Path::new(&p).is_absolute())),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.isRelative", Arc::new(GcNativeFn {
            name: "Path.isRelative".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::Path;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => Ok(GcValue::Bool(Path::new(&p).is_relative())),
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Path.split", Arc::new(GcNativeFn {
            name: "Path.split".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::path::Path;
                let p = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match p {
                    Some(p) => {
                        let path = Path::new(&p);
                        let parts: Vec<GcValue> = path.components()
                            .map(|c| GcValue::String(heap.alloc_string(c.as_os_str().to_string_lossy().to_string())))
                            .collect();
                        Ok(GcValue::List(heap.make_list(parts)))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // === Regex functions ===

        self.register_native("Regex.matches", Arc::new(GcNativeFn {
            name: "Regex.matches".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern) {
                    (Some(s), Some(pattern)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => Ok(GcValue::Bool(re.is_match(&s))),
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Regex.find", Arc::new(GcNativeFn {
            name: "Regex.find".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern) {
                    (Some(s), Some(pattern)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => {
                                match re.find(&s) {
                                    Some(m) => Ok(GcValue::String(heap.alloc_string(m.as_str().to_string()))),
                                    None => Ok(GcValue::Unit)
                                }
                            },
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Regex.findAll", Arc::new(GcNativeFn {
            name: "Regex.findAll".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern) {
                    (Some(s), Some(pattern)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => {
                                let matches: Vec<GcValue> = re.find_iter(&s)
                                    .map(|m| GcValue::String(heap.alloc_string(m.as_str().to_string())))
                                    .collect();
                                Ok(GcValue::List(heap.make_list(matches)))
                            },
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Regex.replace", Arc::new(GcNativeFn {
            name: "Regex.replace".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let replacement = match &args[2] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern, replacement) {
                    (Some(s), Some(pattern), Some(replacement)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => {
                                let result = re.replace(&s, replacement.as_str()).to_string();
                                Ok(GcValue::String(heap.alloc_string(result)))
                            },
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Regex.replaceAll", Arc::new(GcNativeFn {
            name: "Regex.replaceAll".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let replacement = match &args[2] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern, replacement) {
                    (Some(s), Some(pattern), Some(replacement)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => {
                                let result = re.replace_all(&s, replacement.as_str()).to_string();
                                Ok(GcValue::String(heap.alloc_string(result)))
                            },
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Regex.split", Arc::new(GcNativeFn {
            name: "Regex.split".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern) {
                    (Some(s), Some(pattern)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => {
                                let parts: Vec<GcValue> = re.split(&s)
                                    .map(|p| GcValue::String(heap.alloc_string(p.to_string())))
                                    .collect();
                                Ok(GcValue::List(heap.make_list(parts)))
                            },
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Regex.captures", Arc::new(GcNativeFn {
            name: "Regex.captures".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                use regex::Regex;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (s, pattern) {
                    (Some(s), Some(pattern)) => {
                        match Regex::new(&pattern) {
                            Ok(re) => {
                                match re.captures(&s) {
                                    Some(caps) => {
                                        let groups: Vec<GcValue> = caps.iter()
                                            .map(|m| match m {
                                                Some(m) => GcValue::String(heap.alloc_string(m.as_str().to_string())),
                                                None => GcValue::Unit
                                            })
                                            .collect();
                                        Ok(GcValue::List(heap.make_list(groups)))
                                    },
                                    None => Ok(GcValue::Unit)
                                }
                            },
                            Err(e) => Err(RuntimeError::Panic(format!("Invalid regex: {}", e)))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // === UUID Functions ===

        self.register_native("Uuid.v4", Arc::new(GcNativeFn {
            name: "Uuid.v4".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                use uuid::Uuid;
                let id = Uuid::new_v4().to_string();
                Ok(GcValue::String(heap.alloc_string(id)))
            }),
        }));

        self.register_native("Uuid.isValid", Arc::new(GcNativeFn {
            name: "Uuid.isValid".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use uuid::Uuid;
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match s {
                    Some(s) => Ok(GcValue::Bool(Uuid::parse_str(&s).is_ok())),
                    None => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        // === Crypto Functions ===

        self.register_native("Crypto.sha256", Arc::new(GcNativeFn {
            name: "Crypto.sha256".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use sha2::{Sha256, Digest};
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match s {
                    Some(s) => {
                        let mut hasher = Sha256::new();
                        hasher.update(s.as_bytes());
                        let result = hasher.finalize();
                        let hex = result.iter().map(|b| format!("{:02x}", b)).collect::<String>();
                        Ok(GcValue::String(heap.alloc_string(hex)))
                    },
                    None => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Crypto.sha512", Arc::new(GcNativeFn {
            name: "Crypto.sha512".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use sha2::{Sha512, Digest};
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match s {
                    Some(s) => {
                        let mut hasher = Sha512::new();
                        hasher.update(s.as_bytes());
                        let result = hasher.finalize();
                        let hex = result.iter().map(|b| format!("{:02x}", b)).collect::<String>();
                        Ok(GcValue::String(heap.alloc_string(hex)))
                    },
                    None => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Crypto.md5", Arc::new(GcNativeFn {
            name: "Crypto.md5".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use md5::{Md5, Digest};
                let s = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match s {
                    Some(s) => {
                        let mut hasher = Md5::new();
                        hasher.update(s.as_bytes());
                        let result = hasher.finalize();
                        let hex = result.iter().map(|b| format!("{:02x}", b)).collect::<String>();
                        Ok(GcValue::String(heap.alloc_string(hex)))
                    },
                    None => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Crypto.bcryptHash", Arc::new(GcNativeFn {
            name: "Crypto.bcryptHash".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let password = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let cost = match &args[1] {
                    GcValue::Int64(n) => *n as u32,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                match password {
                    Some(password) => {
                        match bcrypt::hash(password, cost) {
                            Ok(hash) => Ok(GcValue::String(heap.alloc_string(hash))),
                            Err(e) => Err(RuntimeError::Panic(format!("bcrypt error: {}", e)))
                        }
                    },
                    None => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Crypto.bcryptVerify", Arc::new(GcNativeFn {
            name: "Crypto.bcryptVerify".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let password = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let hash = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                match (password, hash) {
                    (Some(password), Some(hash)) => {
                        match bcrypt::verify(password, &hash) {
                            Ok(valid) => Ok(GcValue::Bool(valid)),
                            Err(_) => Ok(GcValue::Bool(false))
                        }
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
            }),
        }));

        self.register_native("Crypto.randomBytes", Arc::new(GcNativeFn {
            name: "Crypto.randomBytes".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use rand::RngCore;
                let n = match &args[0] {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let mut bytes = vec![0u8; n];
                rand::thread_rng().fill_bytes(&mut bytes);
                let hex = bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();
                Ok(GcValue::String(heap.alloc_string(hex)))
            }),
        }));

        // === Map Functions ===

        // Map.insert(map, key, value) -> new map with key-value inserted
        self.register_native("Map.insert", Arc::new(GcNativeFn {
            name: "Map.insert".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let key = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;
                let value = args[2].clone();

                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let new_entries = map.entries.update(key, value);
                        let new_ptr = heap.alloc_map(new_entries);
                        Ok(GcValue::Map(new_ptr))
                    }
                    GcValue::SharedMap(shared_map) => {
                        let shared_key = key.to_shared_key();
                        let shared_value = heap.gc_value_to_shared(&value).ok_or_else(||
                            RuntimeError::Panic("Cannot convert value to shared type".to_string()))?;
                        let new_map = std::sync::Arc::new((**shared_map).clone().update(shared_key, shared_value));
                        Ok(GcValue::SharedMap(new_map))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.remove(map, key) -> new map with key removed
        self.register_native("Map.remove", Arc::new(GcNativeFn {
            name: "Map.remove".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let key = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;

                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let new_entries = map.entries.without(&key);
                        let new_ptr = heap.alloc_map(new_entries);
                        Ok(GcValue::Map(new_ptr))
                    }
                    GcValue::SharedMap(shared_map) => {
                        let shared_key = key.to_shared_key();
                        let new_map = std::sync::Arc::new((**shared_map).clone().without(&shared_key));
                        Ok(GcValue::SharedMap(new_map))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.get(map, key) -> Option value
        self.register_native("Map.get", Arc::new(GcNativeFn {
            name: "Map.get".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let key = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;

                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        match map.entries.get(&key) {
                            Some(value) => Ok(value.clone()),
                            None => Ok(GcValue::Unit)
                        }
                    }
                    GcValue::SharedMap(shared_map) => {
                        let shared_key = key.to_shared_key();
                        match shared_map.get(&shared_key) {
                            Some(value) => Ok(heap.shared_to_gc_value(value)),
                            None => Ok(GcValue::Unit)
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.contains(map, key) -> Bool
        self.register_native("Map.contains", Arc::new(GcNativeFn {
            name: "Map.contains".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let key = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;

                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        Ok(GcValue::Bool(map.entries.contains_key(&key)))
                    }
                    GcValue::SharedMap(shared_map) => {
                        let shared_key = key.to_shared_key();
                        Ok(GcValue::Bool(shared_map.contains_key(&shared_key)))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.keys(map) -> [keys]
        self.register_native("Map.keys", Arc::new(GcNativeFn {
            name: "Map.keys".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let keys_cloned: Vec<_> = map.entries.keys().cloned().collect();
                        let keys: Vec<GcValue> = keys_cloned.into_iter().map(|k| k.to_gc_value(heap)).collect();
                        Ok(GcValue::List(heap.make_list(keys)))
                    }
                    GcValue::SharedMap(shared_map) => {
                        let keys: Vec<GcValue> = shared_map.keys()
                            .map(|k| crate::gc::GcMapKey::from_shared_key(k).to_gc_value(heap))
                            .collect();
                        Ok(GcValue::List(heap.make_list(keys)))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.values(map) -> [values]
        self.register_native("Map.values", Arc::new(GcNativeFn {
            name: "Map.values".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let values: Vec<GcValue> = map.entries.values().cloned().collect();
                        Ok(GcValue::List(heap.make_list(values)))
                    }
                    GcValue::SharedMap(shared_map) => {
                        let values: Vec<GcValue> = shared_map.values()
                            .map(|v| heap.shared_to_gc_value(v))
                            .collect();
                        Ok(GcValue::List(heap.make_list(values)))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.size(map) -> Int
        self.register_native("Map.size", Arc::new(GcNativeFn {
            name: "Map.size".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        Ok(GcValue::Int64(map.entries.len() as i64))
                    }
                    GcValue::SharedMap(shared_map) => {
                        Ok(GcValue::Int64(shared_map.len() as i64))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.isEmpty(map) -> Bool
        self.register_native("Map.isEmpty", Arc::new(GcNativeFn {
            name: "Map.isEmpty".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        Ok(GcValue::Bool(map.entries.is_empty()))
                    }
                    GcValue::SharedMap(shared_map) => {
                        Ok(GcValue::Bool(shared_map.is_empty()))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // === Set Functions ===

        // Set.insert(set, elem) -> new set with element inserted
        self.register_native("Set.insert", Arc::new(GcNativeFn {
            name: "Set.insert".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let set_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };
                let elem = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;

                let set = heap.get_set(set_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let new_items = set.items.update(elem);
                let new_ptr = heap.alloc_set(new_items);
                Ok(GcValue::Set(new_ptr))
            }),
        }));

        // Set.remove(set, elem) -> new set with element removed
        self.register_native("Set.remove", Arc::new(GcNativeFn {
            name: "Set.remove".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let set_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };
                let elem = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;

                let set = heap.get_set(set_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let new_items = set.items.without(&elem);
                let new_ptr = heap.alloc_set(new_items);
                Ok(GcValue::Set(new_ptr))
            }),
        }));

        // Set.contains(set, elem) -> Bool
        self.register_native("Set.contains", Arc::new(GcNativeFn {
            name: "Set.contains".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let set_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };
                let elem = args[1].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                    expected: "hashable".to_string(),
                    found: args[1].type_name(heap).to_string()
                })?;

                let set = heap.get_set(set_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                Ok(GcValue::Bool(set.items.contains(&elem)))
            }),
        }));

        // Set.size(set) -> Int
        self.register_native("Set.size", Arc::new(GcNativeFn {
            name: "Set.size".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let set_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };

                let set = heap.get_set(set_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                Ok(GcValue::Int64(set.items.len() as i64))
            }),
        }));

        // Set.isEmpty(set) -> Bool
        self.register_native("Set.isEmpty", Arc::new(GcNativeFn {
            name: "Set.isEmpty".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let set_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };

                let set = heap.get_set(set_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                Ok(GcValue::Bool(set.items.is_empty()))
            }),
        }));

        // Set.toList(set) -> [elements]
        self.register_native("Set.toList", Arc::new(GcNativeFn {
            name: "Set.toList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let set_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };

                // Clone items first to release borrow on heap
                let set = heap.get_set(set_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let items_cloned: Vec<_> = set.items.iter().cloned().collect();
                let _ = set;
                // Now convert to GcValues
                let elements: Vec<GcValue> = items_cloned.into_iter().map(|k| k.to_gc_value(heap)).collect();
                Ok(GcValue::List(heap.make_list(elements)))
            }),
        }));

        // Set.union(set1, set2) -> new set with all elements from both
        self.register_native("Set.union", Arc::new(GcNativeFn {
            name: "Set.union".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let set1_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };
                let set2_ptr = match &args[1] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[1].type_name(heap).to_string() })
                };

                let set1 = heap.get_set(set1_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let set2 = heap.get_set(set2_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let new_items = set1.items.clone().union(set2.items.clone());
                let new_ptr = heap.alloc_set(new_items);
                Ok(GcValue::Set(new_ptr))
            }),
        }));

        // Set.intersection(set1, set2) -> new set with elements in both
        self.register_native("Set.intersection", Arc::new(GcNativeFn {
            name: "Set.intersection".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let set1_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };
                let set2_ptr = match &args[1] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[1].type_name(heap).to_string() })
                };

                let set1 = heap.get_set(set1_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let set2 = heap.get_set(set2_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let new_items = set1.items.clone().intersection(set2.items.clone());
                let new_ptr = heap.alloc_set(new_items);
                Ok(GcValue::Set(new_ptr))
            }),
        }));

        // Set.difference(set1, set2) -> new set with elements in set1 but not set2
        self.register_native("Set.difference", Arc::new(GcNativeFn {
            name: "Set.difference".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let set1_ptr = match &args[0] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[0].type_name(heap).to_string() })
                };
                let set2_ptr = match &args[1] {
                    GcValue::Set(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError { expected: "Set".to_string(), found: args[1].type_name(heap).to_string() })
                };

                let set1 = heap.get_set(set1_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let set2 = heap.get_set(set2_ptr).ok_or_else(|| RuntimeError::Panic("Invalid set pointer".to_string()))?;
                let new_items = set1.items.clone().relative_complement(set2.items.clone());
                let new_ptr = heap.alloc_set(new_items);
                Ok(GcValue::Set(new_ptr))
            }),
        }));

        // Default inspect - just prints the value with its name (used outside TUI mode)
        self.register_native("inspect", Arc::new(GcNativeFn {
            name: "inspect".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                // Get the name (second argument must be a string)
                let name = match &args[1] {
                    GcValue::String(ptr) => {
                        if let Some(s) = heap.get_string(*ptr) {
                            s.data.clone()
                        } else {
                            "unknown".to_string()
                        }
                    }
                    _ => "unknown".to_string(),
                };

                // Display the value
                let value_str = heap.display_value(&args[0]);
                println!("[inspect] {}: {}", name, value_str);

                Ok(GcValue::Unit)
            }),
        }));
    }
    /// Run the main function and return the result.
    pub fn run(&mut self, main_fn_name: &str) -> Result<SendableValue, String> {
        let (result, _profile) = self.run_with_profile(main_fn_name)?;
        Ok(result)
    }

    /// Run the main function and return both the result and profile summary (if profiling enabled).
    pub fn run_with_profile(&mut self, main_fn_name: &str) -> Result<(SendableValue, Option<String>), String> {
        // Create multi-threaded tokio runtime for parallel execution
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        // Run with work-stealing across multiple threads
        rt.block_on(self.run_async_with_profile(main_fn_name))
    }

    /// Run with multi-threaded work distribution (same as run, kept for API compat).
    pub fn run_parallel(&mut self, main_fn_name: &str, _num_threads: usize) -> Result<SendableValue, String> {
        self.run(main_fn_name)
    }

    /// Run in a background thread, returning a handle for the result and cancellation.
    /// This allows the caller to poll for completion while remaining responsive.
    /// Each call gets its own interrupt flag, enabling independent cancellation.
    pub fn run_threaded(&self, main_fn_name: &str) -> ThreadedEvalHandle {
        let shared = Arc::clone(&self.shared);
        let fn_name = main_fn_name.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        let interrupt = Arc::new(AtomicBool::new(false));
        let interrupt_clone = Arc::clone(&interrupt);

        std::thread::spawn(move || {
            let result = Self::run_in_thread(shared, &fn_name, interrupt_clone);
            let _ = tx.send(result);
        });

        ThreadedEvalHandle {
            result_rx: rx,
            interrupt,
        }
    }

    /// Internal: run execution in the current thread with given shared state and interrupt.
    fn run_in_thread(shared: Arc<AsyncSharedState>, main_fn_name: &str, interrupt: Arc<AtomicBool>) -> Result<SendableValue, String> {
        // Each eval thread gets its own runtime - this is safe because TokioRwLock
        // is runtime-agnostic and works across different runtime instances
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
        rt.block_on(Self::run_async_inner(shared, main_fn_name, interrupt))
    }

    /// Internal: async execution with shared state and local interrupt.
    async fn run_async_inner(shared: Arc<AsyncSharedState>, main_fn_name: &str, interrupt: Arc<AtomicBool>) -> Result<SendableValue, String> {
        // Find main function
        let main_fn = shared.functions.read().unwrap().get(main_fn_name)
            .ok_or_else(|| format!("Main function '{}' not found", main_fn_name))?
            .clone();

        // Create main process with local interrupt
        let pid = shared.alloc_pid();
        let mut process = AsyncProcess::new(pid, Arc::clone(&shared));
        process.local_interrupt = Some(interrupt);
        let sender = process.mailbox_sender.clone();
        shared.register_process(pid, sender).await;

        // Set up initial call frame
        let registers = vec![GcValue::Unit; main_fn.code.register_count as usize];
        process.frames.push(CallFrame {
            function: main_fn,
            ip: 0,
            registers,
            captures: Vec::new(),
            return_reg: None,
        });

        // Run main process (blocks until complete)
        let result = process.run().await;

        match result {
            Ok(value) => {
                let sendable = SendableValue::from_gc_value(&value, &process.heap);
                Ok(sendable)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Async entry point for running main with profile data.
    async fn run_async_with_profile(&self, main_fn_name: &str) -> Result<(SendableValue, Option<String>), String> {
        // Find main function
        let main_fn = self.shared.functions.read().unwrap().get(main_fn_name)
            .ok_or_else(|| format!("Main function '{}' not found", main_fn_name))?
            .clone();

        // Create main process
        let pid = self.shared.alloc_pid();
        let mut process = AsyncProcess::new(pid, self.shared.clone());
        let sender = process.mailbox_sender.clone();
        self.shared.register_process(pid, sender).await;

        // Record initial function entry for profiling
        process.profile_enter(&main_fn.name);

        // Set up initial call frame
        let registers = vec![GcValue::Unit; main_fn.code.register_count as usize];
        process.frames.push(CallFrame {
            function: main_fn,
            ip: 0,
            registers,
            captures: Vec::new(),
            return_reg: None,
        });

        // Run main process (blocks until complete)
        let result = process.run().await;

        // Get profiling summary if enabled
        let profile_summary = process.profile.as_ref().map(|p| p.format_summary());

        match result {
            Ok(value) => {
                let sendable = SendableValue::from_gc_value(&value, &process.heap);
                Ok((sendable, profile_summary))
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Run the main function with debug support.
    /// Returns a DebugSession that can be used to control execution.
    pub fn run_debug(&self, main_fn_name: &str) -> Result<DebugSession, String> {
        use crate::shared_types::{DebugCommand, DebugEvent, StepMode};

        // Create debug channels
        let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded::<DebugCommand>();
        let (event_tx, event_rx) = crossbeam::channel::unbounded::<DebugEvent>();

        let shared = Arc::clone(&self.shared);
        let fn_name = main_fn_name.to_string();
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        // Spawn execution thread
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");

            let result = rt.block_on(async {
                // Find main function
                let main_fn = shared.functions.read().unwrap().get(&fn_name)
                    .ok_or_else(|| format!("Main function '{}' not found", fn_name))?
                    .clone();

                // Create main process with debug channels
                let pid = shared.alloc_pid();
                let mut process = AsyncProcess::new(pid, Arc::clone(&shared));
                process.debug_command_receiver = Some(cmd_rx);
                process.debug_event_sender = Some(event_tx.clone());
                // Start paused so debugger can set breakpoints
                process.step_mode = StepMode::Paused;

                let sender = process.mailbox_sender.clone();
                shared.register_process(pid, sender).await;

                // Set up initial call frame
                let registers = vec![GcValue::Unit; main_fn.code.register_count as usize];
                process.frames.push(CallFrame {
                    function: main_fn,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: None,
                });

                // Run main process
                let result = process.run().await;

                // Send exit event
                match &result {
                    Ok(value) => {
                        let value_str = process.heap.display_value(value);
                        let _ = event_tx.send(DebugEvent::Exited { pid: pid.0, value: Some(value_str) });
                    }
                    Err(_) => {
                        let _ = event_tx.send(DebugEvent::Exited { pid: pid.0, value: None });
                    }
                }

                match result {
                    Ok(value) => {
                        let sendable = SendableValue::from_gc_value(&value, &process.heap);
                        Ok(sendable)
                    }
                    Err(e) => Err(e.to_string()),
                }
            });

            let _ = result_tx.send(result);
        });

        Ok(DebugSession {
            command_sender: cmd_tx,
            event_receiver: event_rx,
            result_receiver: result_rx,
            _thread_handle: handle,
        })
    }
}

/// A debug session for controlling program execution.
pub struct DebugSession {
    /// Send commands to the running process.
    pub command_sender: crate::shared_types::DebugCommandSender,
    /// Receive events from the running process.
    pub event_receiver: crate::shared_types::DebugEventReceiver,
    /// Receive the final result when execution completes.
    result_receiver: std::sync::mpsc::Receiver<Result<SendableValue, String>>,
    /// Thread handle (kept alive to prevent premature join).
    _thread_handle: std::thread::JoinHandle<()>,
}

impl DebugSession {
    /// Send a debug command.
    pub fn send(&self, cmd: crate::shared_types::DebugCommand) -> Result<(), String> {
        self.command_sender.send(cmd).map_err(|e| e.to_string())
    }

    /// Receive the next debug event (blocking).
    pub fn recv_event(&self) -> Option<crate::shared_types::DebugEvent> {
        self.event_receiver.recv().ok()
    }

    /// Try to receive the next debug event (non-blocking).
    pub fn try_recv_event(&self) -> Option<crate::shared_types::DebugEvent> {
        self.event_receiver.try_recv().ok()
    }

    /// Wait for execution to complete and return the result.
    pub fn wait(self) -> Result<SendableValue, String> {
        self.result_receiver.recv().map_err(|e| e.to_string())?
    }

    /// Check if execution has completed (non-blocking).
    pub fn try_result(&self) -> Option<Result<SendableValue, String>> {
        self.result_receiver.try_recv().ok()
    }

    /// Set a breakpoint at a line.
    pub fn set_breakpoint(&self, line: usize) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::AddBreakpoint(
            crate::shared_types::Breakpoint::Line { file: None, line }
        ))
    }

    /// Set a breakpoint at a file:line.
    pub fn set_breakpoint_file(&self, file: &str, line: usize) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::AddBreakpoint(
            crate::shared_types::Breakpoint::Line { file: Some(file.to_string()), line }
        ))
    }

    /// Set a breakpoint on a function (break when function is entered).
    pub fn set_breakpoint_function(&self, function_name: &str) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::AddBreakpoint(
            crate::shared_types::Breakpoint::Function(function_name.to_string())
        ))
    }

    /// Continue execution.
    pub fn continue_exec(&self) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::Continue)
    }

    /// Step to next line.
    pub fn step_line(&self) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::StepLine)
    }

    /// Step over function calls.
    pub fn step_over(&self) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::StepOver)
    }

    /// Step out of current function.
    pub fn step_out(&self) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::StepOut)
    }

    /// Print a variable's value.
    pub fn print_var(&self, name: &str) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::PrintVariable(name.to_string()))
    }

    /// Print all local variables.
    pub fn print_locals(&self) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::PrintLocals)
    }

    /// Print the call stack.
    pub fn print_stack(&self) -> Result<(), String> {
        self.send(crate::shared_types::DebugCommand::PrintStack)
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn test_debug_types_exist() {
        use crate::shared_types::{DebugCommand, DebugEvent, Breakpoint, StepMode, StackFrame};

        // Test that debug types can be constructed
        let bp = Breakpoint::Line { file: Some("test.nos".to_string()), line: 10 };
        assert!(matches!(bp, Breakpoint::Line { line: 10, .. }));

        let fn_bp = Breakpoint::Function("main".to_string());
        assert!(matches!(fn_bp, Breakpoint::Function(_)));

        let mode = StepMode::Paused;
        assert_eq!(mode, StepMode::Paused);

        let cmd = DebugCommand::Continue;
        assert!(matches!(cmd, DebugCommand::Continue));

        let frame = StackFrame {
            function: "main".to_string(),
            file: Some("test.nos".to_string()),
            line: 1,
            locals: vec!["x".to_string()],
        };
        assert_eq!(frame.function, "main");
    }
}
