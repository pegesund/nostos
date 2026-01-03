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

use crate::extensions::{ext_value_to_vm, vm_value_to_ext, ExtensionManager};

/// Log debug message to file when ASYNC_VM_DEBUG is set.
/// Logs to /tmp/nostos_debug.log
#[allow(unused)]
fn debug_log_to_file(msg: &str) {
    if std::env::var("ASYNC_VM_DEBUG").is_ok() {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/nostos_debug.log")
        {
            use std::io::Write;
            let _ = writeln!(f, "{}", msg);
        }
    }
}

/// Reactive render context for tracking component dependencies during RHtml rendering.
/// A node in the component tree
#[derive(Debug, Clone)]
pub struct ComponentTreeNode {
    pub name: String,
    pub children: Vec<ComponentTreeNode>,
}

/// This enables automatic re-rendering when reactive records change.
#[derive(Debug, Clone, Default)]
pub struct ReactiveRenderContext {
    /// Stack of component IDs currently being rendered.
    /// When non-empty, we're inside an RHtml render and should track dependencies.
    pub render_stack: Vec<String>,

    /// Dependencies: reactive record pointer -> list of component IDs that read from it.
    /// Key is the Arc pointer address as u64 for fast lookup.
    pub dependencies: HashMap<u64, Vec<String>>,

    /// Components queued for re-render due to reactive record changes.
    pub pending_rerenders: Vec<String>,

    /// Record IDs that were modified (for Reactive.getChangedRecordIds).
    pub changed_record_ids: Vec<u64>,

    /// Stack for building component tree during render.
    /// Each entry is (component_name, children_so_far).
    pub component_tree_stack: Vec<(String, Vec<ComponentTreeNode>)>,

    /// Root-level components (when stack is empty, new components go here).
    pub root_components: Vec<ComponentTreeNode>,

    /// Renderer functions: component name -> render function.
    /// Used to re-render individual components.
    pub renderers: HashMap<String, GcValue>,

    /// Nesting depth for RHtml calls. Only clear context on depth=0.
    pub nesting_depth: usize,
}
use crate::gc::{GcConfig, GcInt64List, GcList, GcMapKey, GcValue, Heap, GcNativeFn};

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
use crate::value::{FunctionValue, Pid, TypeValue, RefId, RuntimeError, Value, ReactiveRecordValue};
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
    /// Native functions (string-based lookup).
    pub natives: HashMap<String, Arc<GcNativeFn>>,
    /// Native functions indexed by u16 for fast CallNativeIdx lookup.
    pub natives_vec: Vec<Arc<GcNativeFn>>,
    /// Maps native function names to their index in natives_vec.
    pub native_name_to_idx: HashMap<String, u16>,
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
    /// JIT list sum functions: (data_ptr, len) -> sum
    pub jit_list_sum_functions: RwLock<HashMap<u16, crate::shared_types::JitListSumFn>>,
    /// JIT tail-recursive list sum functions: (data_ptr, len, acc) -> sum
    pub jit_list_sum_tr_functions: RwLock<HashMap<u16, crate::shared_types::JitListSumTrFn>>,

    /// Shutdown signal (permanent).
    pub shutdown: AtomicBool,

    /// Interrupt signal (temporary, for Ctrl+C).
    /// Stops current execution but allows VM to be reused.
    pub interrupt: AtomicBool,

    /// Interactive mode flag (true in REPL/TUI, false in script mode).
    pub interactive_mode: AtomicBool,

    /// Runtime handle for spawning long-lived processes.
    /// In interactive mode, spawned processes use this handle instead of the
    /// current runtime so they outlive individual eval calls.
    pub spawn_runtime_handle: Option<tokio::runtime::Handle>,

    /// Process registry: Pid -> mailbox sender.
    /// Protected by tokio RwLock for async access.
    pub process_registry: TokioRwLock<HashMap<Pid, MailboxSender>>,

    /// Process abort handles: Pid -> tokio AbortHandle.
    /// Used to actually stop tasks when Process.kill is called.
    pub process_abort_handles: TokioRwLock<HashMap<Pid, tokio::task::AbortHandle>>,

    /// Process server handles: Pid -> Vec<server_handle>.
    /// Used to close servers when a process is killed.
    pub process_servers: TokioRwLock<HashMap<Pid, Vec<u64>>>,

    /// Debug counters for process lifecycle tracking
    pub spawned_count: AtomicU64,
    pub exited_count: AtomicU64,

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

    /// Extension manager for native library functions.
    pub extensions: RwLock<Option<Arc<ExtensionManager>>>,
}

impl AsyncSharedState {
    /// Allocate a new unique PID.
    pub fn alloc_pid(&self) -> Pid {
        Pid(self.next_pid.fetch_add(1, Ordering::SeqCst))
    }

    /// Register a process in the registry.
    pub async fn register_process(&self, pid: Pid, sender: MailboxSender) {
        self.spawned_count.fetch_add(1, Ordering::Relaxed);
        self.process_registry.write().await.insert(pid, sender);
    }

    /// Unregister a process from the registry.
    pub async fn unregister_process(&self, pid: Pid) {
        self.exited_count.fetch_add(1, Ordering::Relaxed);
        self.process_registry.write().await.remove(&pid);
    }

    /// Get process statistics for debugging.
    /// Returns (spawned, exited, currently_active)
    pub async fn process_stats(&self) -> (u64, u64, usize) {
        let spawned = self.spawned_count.load(Ordering::Relaxed);
        let exited = self.exited_count.load(Ordering::Relaxed);
        let active = self.process_registry.read().await.len();
        (spawned, exited, active)
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

    /// Reactive render context for RHtml dependency tracking.
    /// Tracks which components depend on which reactive records.
    pub reactive_context: ReactiveRenderContext,
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
            reactive_context: ReactiveRenderContext::default(),
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
            reactive_context: ReactiveRenderContext::default(),
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

    /// Get the first non-zero line number of the current function.
    /// This is used when the current instruction has no line info (line 0).
    fn debug_function_first_nonzero_line(&self) -> usize {
        self.frames.last()
            .and_then(|f| f.function.code.lines.iter().find(|&&line| line > 0).copied())
            .unwrap_or(1)
    }

    /// Get the source code of the current function (if available).
    fn debug_current_source(&self) -> Option<String> {
        self.frames.last()
            .and_then(|f| f.function.source_code.as_ref())
            .map(|s| s.to_string())
    }

    /// Get the starting line number of the current function (first non-zero line in the function).
    fn debug_source_start_line(&self) -> usize {
        self.frames.last()
            .and_then(|f| f.function.code.lines.iter().find(|&&line| line > 0).copied())
            .unwrap_or(1)
    }

    /// Check if the current instruction is a Return instruction.
    fn debug_is_return_instruction(&self) -> bool {
        if let Some(frame) = self.frames.last() {
            if let Some(instr) = frame.function.code.code.get(frame.ip) {
                matches!(instr, crate::value::Instruction::Return(_))
            } else {
                false
            }
        } else {
            false
        }
    }

    #[allow(unused)]
    fn debug_log(msg: &str) {
        // Debug logging enabled for troubleshooting
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/nostos_vm_debug.log")
        {
            use std::io::Write;
            let _ = writeln!(f, "{}", msg);
        }
    }

    /// Check if a function name matches a breakpoint pattern.
    /// Handles: exact match, without arity suffix, normalized separators (: vs .)
    fn function_name_matches(fn_name: &str, bp_pattern: &str) -> bool {
        // Normalize separators: convert : to . for comparison
        let normalized_bp = bp_pattern.replace(':', ".");
        let normalized_fn = fn_name.replace(':', ".");

        // Also strip arity suffix from function name (e.g., "foo/2" -> "foo")
        let fn_base = normalized_fn.split('/').next().unwrap_or(&normalized_fn);

        // Check various match patterns:
        // 1. Exact match (normalized)
        if normalized_fn == normalized_bp {
            return true;
        }
        // 2. Function base matches pattern (for "foo/2" matching "foo")
        if fn_base == normalized_bp {
            return true;
        }
        // 3. Pattern is just the function name without module (e.g., "simple" matches "module.simple")
        if let Some(simple_name) = fn_base.rsplit('.').next() {
            if simple_name == normalized_bp {
                return true;
            }
        }
        // 4. Pattern ends with the function name (for partial module matching)
        if fn_base.ends_with(&format!(".{}", normalized_bp)) {
            return true;
        }

        false
    }

    /// Check if we should pause execution (breakpoint or step mode).
    fn debug_should_pause(&mut self) -> bool {
        use crate::shared_types::StepMode;

        // Skip if not debugging
        if self.debug_event_sender.is_none() {
            return false;
        }

        let fn_name = self.debug_current_function();
        Self::debug_log(&format!("debug_should_pause: fn={}, step_mode={:?}, breakpoints={:?}",
            fn_name, self.step_mode, self.breakpoints));

        // Check for Paused mode FIRST - this is the initial state and should always pause
        // regardless of line info availability
        if self.step_mode == StepMode::Paused {
            Self::debug_log("  -> Paused mode, returning true");
            return true;
        }

        // Skip breakpoint check once after Continue (to avoid re-hitting same breakpoint)
        if self.skip_next_breakpoint_check {
            self.skip_next_breakpoint_check = false;
            Self::debug_log("  -> skipping breakpoint check");
            return false;
        }

        let current_line = match self.debug_current_line() {
            Some(line) => line,
            None => {
                Self::debug_log("  -> no current line, returning false");
                return false;
            }
        };
        Self::debug_log(&format!("  current_line={}, first_line={}", current_line, self.debug_function_first_line()));

        match self.step_mode {
            StepMode::Run => {
                // Check function breakpoints
                let fn_name = self.debug_current_function();
                Self::debug_log(&format!("  Run mode: checking breakpoints for fn={}", fn_name));
                if fn_name != "<unknown>" {
                    // Check if any breakpoint matches this function
                    // Handles: exact match, without arity suffix, with normalized separators (: vs .)
                    let matches_breakpoint = self.breakpoints.iter().any(|bp| {
                        if let crate::shared_types::Breakpoint::Function(bp_name) = bp {
                            Self::function_name_matches(&fn_name, bp_name)
                        } else {
                            false
                        }
                    });
                    Self::debug_log(&format!("  breakpoint_match={}", matches_breakpoint));
                    if matches_breakpoint {
                        // Only break on function entry (IP=0, first instruction)
                        let ip = self.frames.last().map(|f| f.ip).unwrap_or(0);
                        Self::debug_log(&format!("  checking function entry: ip={}", ip));
                        if ip == 0 {
                            Self::debug_log("  -> BREAKPOINT HIT");
                            self.step_mode = StepMode::Paused;
                            return true;
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
                // Pause when line changes (skip line 0 which means "no line info")
                // Also pause before Return to show return value
                Self::debug_log(&format!("  StepLine: current_line={}, last_line={}", current_line, self.debug_last_line));
                if self.debug_is_return_instruction() {
                    Self::debug_log("  -> Return instruction, pausing");
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                if current_line > 0 && current_line != self.debug_last_line {
                    Self::debug_log(&format!("  -> line changed, pausing"));
                    self.debug_last_line = current_line;
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                false
            }
            StepMode::StepOver => {
                // Pause when line changes at same or lower call depth (skip line 0)
                // Also pause before Return to show return value
                Self::debug_log(&format!("  StepOver: current_line={}, last_line={}, frames={}, step_depth={}",
                    current_line, self.debug_last_line, self.frames.len(), self.debug_step_frame_depth));
                if self.debug_is_return_instruction() && self.frames.len() <= self.debug_step_frame_depth {
                    Self::debug_log("  -> Return instruction, pausing");
                    self.step_mode = StepMode::Paused;
                    return true;
                }
                if current_line == 0 {
                    Self::debug_log("  -> skip: line=0");
                } else if self.frames.len() > self.debug_step_frame_depth {
                    Self::debug_log(&format!("  -> skip: frame depth {} > {}", self.frames.len(), self.debug_step_frame_depth));
                } else if current_line == self.debug_last_line {
                    Self::debug_log(&format!("  -> skip: same line {}", current_line));
                } else {
                    Self::debug_log("  -> line changed, pausing");
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
        // If current line is 0 (no line info), use the first non-zero line of the function
        let line = match self.debug_current_line() {
            Some(0) | None => self.debug_function_first_nonzero_line(),
            Some(line) => line,
        };
        let function = self.debug_current_function();
        let file = self.debug_current_file();
        let source = self.debug_current_source();
        let source_start_line = self.debug_source_start_line();

        // Dump bytecode line info for debugging
        if let Some(frame) = self.frames.last() {
            let lines: Vec<usize> = frame.function.code.lines.clone();
            Self::debug_log(&format!("[Paused] fn={}, ip={}, bytecode_lines={:?}", function, frame.ip, lines));
        }

        self.debug_send_event(DebugEvent::Paused { pid: self.pid.0, file: file.clone(), line, function: function.clone(), source, source_start_line });

        // Process commands until we get a continue/step command
        while self.step_mode == StepMode::Paused {
            match receiver.recv() {
                Ok(cmd) => match cmd {
                    DebugCommand::Continue => {
                        Self::debug_log(&format!("[Continue cmd] breakpoints={:?}", self.breakpoints));
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
                        Self::debug_log(&format!("[StepOver cmd] setting debug_last_line={}, step_depth={}", line, self.frames.len()));
                        self.debug_last_line = line;
                        self.debug_step_frame_depth = self.frames.len();
                        self.step_mode = StepMode::StepOver;
                    }
                    DebugCommand::StepOut => {
                        self.debug_step_frame_depth = self.frames.len();
                        self.step_mode = StepMode::StepOut;
                    }
                    DebugCommand::AddBreakpoint(bp) => {
                        Self::debug_log(&format!("[AddBreakpoint cmd] bp={:?}", bp));
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
                    DebugCommand::PrintLocalsForFrame(frame_index) => {
                        // frames are stored bottom-to-top, but UI shows top-to-bottom
                        // frame_index 0 = current (last), 1 = caller, etc.
                        let frame_count = self.frames.len();
                        if frame_index < frame_count {
                            let actual_index = frame_count - 1 - frame_index;
                            let frame = &self.frames[actual_index];
                            let mut variables: Vec<(String, String, String)> = frame.function.debug_symbols.iter()
                                .filter_map(|sym| {
                                    frame.registers.get(sym.register as usize).map(|v| {
                                        (sym.name.clone(), format!("{:?}", v), "local".to_string())
                                    })
                                })
                                .collect();

                            // If at a Return instruction, show the return value
                            if frame_index == 0 {
                                if let Some(instr) = frame.function.code.code.get(frame.ip) {
                                    if let crate::value::Instruction::Return(src) = instr {
                                        if let Some(val) = frame.registers.get(*src as usize) {
                                            variables.push((
                                                "(return value)".to_string(),
                                                format!("{:?}", val),
                                                "return".to_string()
                                            ));
                                        }
                                    }
                                }
                            }

                            // Also include module-level mvars (only for frame 0 to avoid clutter)
                            if frame_index == 0 {
                                for (name, mvar) in self.shared.mvars.iter() {
                                    if let Ok(guard) = mvar.try_read() {
                                        variables.push((
                                            format!("mvar {}", name),
                                            format!("{:?}", *guard),
                                            "mvar".to_string()
                                        ));
                                    }
                                }
                                // Also include dynamic mvars from REPL
                                if let Ok(dynamic) = self.shared.dynamic_mvars.read() {
                                    for (name, mvar) in dynamic.iter() {
                                        if let Ok(guard) = mvar.read() {
                                            variables.push((
                                                format!("mvar {}", name),
                                                format!("{:?}", *guard),
                                                "mvar".to_string()
                                            ));
                                        }
                                    }
                                }
                            }

                            self.debug_send_event(DebugEvent::LocalsForFrame { frame_index, variables });
                        } else {
                            self.debug_send_event(DebugEvent::Error {
                                message: format!("Frame index {} out of range (stack has {} frames)", frame_index, frame_count),
                            });
                        }
                    }
                    DebugCommand::PrintStack => {
                        let frames: Vec<StackFrame> = self.frames.iter().rev().map(|f| {
                            let line = f.function.code.lines.get(f.ip.saturating_sub(1)).copied().unwrap_or(0);
                            let source_start_line = f.function.code.lines.iter().find(|&&l| l > 0).copied().unwrap_or(1);
                            StackFrame {
                                function: f.function.name.clone(),
                                file: f.function.source_file.clone(),
                                line,
                                locals: f.function.debug_symbols.iter().map(|s| s.name.clone()).collect(),
                                source: f.function.source_code.as_ref().map(|s| s.to_string()),
                                source_start_line,
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

                // Run garbage collection if threshold exceeded
                self.maybe_gc();

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
        if self.debug_event_sender.is_some() {
            Self::debug_log(&format!("[step] checking debug, step_mode={:?}", self.step_mode));
        }
        if self.debug_should_pause() {
            self.debug_handle_commands();
            Self::debug_log(&format!("[step] after debug_handle_commands, step_mode={:?}", self.step_mode));
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
            ToInt8(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::Int8(*v),
                    GcValue::Int16(v) => GcValue::Int8(*v as i8),
                    GcValue::Int32(v) => GcValue::Int8(*v as i8),
                    GcValue::Int64(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt8(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt16(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt32(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt64(v) => GcValue::Int8(*v as i8),
                    GcValue::Float32(v) => GcValue::Int8(*v as i8),
                    GcValue::Float64(v) => GcValue::Int8(*v as i8),
                    _ => return Err(RuntimeError::Panic("asInt8: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToInt16(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::Int16(*v as i16),
                    GcValue::Int16(v) => GcValue::Int16(*v),
                    GcValue::Int32(v) => GcValue::Int16(*v as i16),
                    GcValue::Int64(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt8(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt16(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt32(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt64(v) => GcValue::Int16(*v as i16),
                    GcValue::Float32(v) => GcValue::Int16(*v as i16),
                    GcValue::Float64(v) => GcValue::Int16(*v as i16),
                    _ => return Err(RuntimeError::Panic("asInt16: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToInt32(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::Int32(*v as i32),
                    GcValue::Int16(v) => GcValue::Int32(*v as i32),
                    GcValue::Int32(v) => GcValue::Int32(*v),
                    GcValue::Int64(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt8(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt16(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt32(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt64(v) => GcValue::Int32(*v as i32),
                    GcValue::Float32(v) => GcValue::Int32(*v as i32),
                    GcValue::Float64(v) => GcValue::Int32(*v as i32),
                    _ => return Err(RuntimeError::Panic("asInt32: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToUInt8(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::UInt8(*v as u8),
                    GcValue::Int16(v) => GcValue::UInt8(*v as u8),
                    GcValue::Int32(v) => GcValue::UInt8(*v as u8),
                    GcValue::Int64(v) => GcValue::UInt8(*v as u8),
                    GcValue::UInt8(v) => GcValue::UInt8(*v),
                    GcValue::UInt16(v) => GcValue::UInt8(*v as u8),
                    GcValue::UInt32(v) => GcValue::UInt8(*v as u8),
                    GcValue::UInt64(v) => GcValue::UInt8(*v as u8),
                    GcValue::Float32(v) => GcValue::UInt8(*v as u8),
                    GcValue::Float64(v) => GcValue::UInt8(*v as u8),
                    _ => return Err(RuntimeError::Panic("asUInt8: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToUInt16(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::UInt16(*v as u16),
                    GcValue::Int16(v) => GcValue::UInt16(*v as u16),
                    GcValue::Int32(v) => GcValue::UInt16(*v as u16),
                    GcValue::Int64(v) => GcValue::UInt16(*v as u16),
                    GcValue::UInt8(v) => GcValue::UInt16(*v as u16),
                    GcValue::UInt16(v) => GcValue::UInt16(*v),
                    GcValue::UInt32(v) => GcValue::UInt16(*v as u16),
                    GcValue::UInt64(v) => GcValue::UInt16(*v as u16),
                    GcValue::Float32(v) => GcValue::UInt16(*v as u16),
                    GcValue::Float64(v) => GcValue::UInt16(*v as u16),
                    _ => return Err(RuntimeError::Panic("asUInt16: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToUInt32(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::UInt32(*v as u32),
                    GcValue::Int16(v) => GcValue::UInt32(*v as u32),
                    GcValue::Int32(v) => GcValue::UInt32(*v as u32),
                    GcValue::Int64(v) => GcValue::UInt32(*v as u32),
                    GcValue::UInt8(v) => GcValue::UInt32(*v as u32),
                    GcValue::UInt16(v) => GcValue::UInt32(*v as u32),
                    GcValue::UInt32(v) => GcValue::UInt32(*v),
                    GcValue::UInt64(v) => GcValue::UInt32(*v as u32),
                    GcValue::Float32(v) => GcValue::UInt32(*v as u32),
                    GcValue::Float64(v) => GcValue::UInt32(*v as u32),
                    _ => return Err(RuntimeError::Panic("asUInt32: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToUInt64(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::UInt64(*v as u64),
                    GcValue::Int16(v) => GcValue::UInt64(*v as u64),
                    GcValue::Int32(v) => GcValue::UInt64(*v as u64),
                    GcValue::Int64(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt8(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt16(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt32(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt64(v) => GcValue::UInt64(*v),
                    GcValue::Float32(v) => GcValue::UInt64(*v as u64),
                    GcValue::Float64(v) => GcValue::UInt64(*v as u64),
                    _ => return Err(RuntimeError::Panic("asUInt64: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToFloat32(dst, src) => {
                let val = reg!(src);
                let result = match &val {
                    GcValue::Int8(v) => GcValue::Float32(*v as f32),
                    GcValue::Int16(v) => GcValue::Float32(*v as f32),
                    GcValue::Int32(v) => GcValue::Float32(*v as f32),
                    GcValue::Int64(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt8(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt16(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt32(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt64(v) => GcValue::Float32(*v as f32),
                    GcValue::Float32(v) => GcValue::Float32(*v),
                    GcValue::Float64(v) => GcValue::Float32(*v as f32),
                    _ => return Err(RuntimeError::Panic("asFloat32: expected numeric".into())),
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
                } else {
                    // Store return value in caller's return register
                    if let Some(ret_reg) = return_reg {
                        // SAFETY: After pop, if frames is not empty, last frame exists
                        let frame = self.frames.last_mut().unwrap();
                        frame.registers[ret_reg as usize] = value;
                    }
                    // Must return to recompute cur_frame (frame was popped)
                    return Ok(StepResult::Continue);
                }
            }

            // === Function calls ===
            CallDirect(dst, func_idx, ref args) => {
                // Skip JIT when debugging - must use interpreted execution for breakpoints
                let use_jit = self.debug_event_sender.is_none();

                // Check for JIT-compiled version first based on arity
                let func_idx_u16 = *func_idx;
                let profiling = self.is_profiling();
                if use_jit { match args.len() {
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
                        // List sum optimization - use native sum() instead of interpreter
                        // This avoids copying and is O(n) directly on the imbl::Vector
                        if self.shared.jit_list_sum_functions.read().unwrap().contains_key(&func_idx_u16) {
                            if let GcValue::Int64List(ref list) = reg_ref!(args[0]) {
                                // Direct sum on imbl::Vector - no copy needed!
                                let (result, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = list.sum();
                                    (r, Some(start.elapsed()))
                                } else {
                                    (list.sum(), None)
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
                        // Tail-recursive list sum JIT: sumTR(list, acc) -> sum
                        if self.shared.jit_list_sum_tr_functions.read().unwrap().contains_key(&func_idx_u16) {
                            if let (GcValue::Int64List(ref list), GcValue::Int64(acc)) = (reg_ref!(args[0]), reg!(args[1])) {
                                // Direct sum on imbl::Vector + initial accumulator
                                let (result, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = list.sum() + acc;
                                    (r, Some(start.elapsed()))
                                } else {
                                    (list.sum() + acc, None)
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
                } } // close if use_jit and match

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
                // Must return to recompute cur_frame (frame was pushed)
                return Ok(StepResult::Continue);
            }

            // Call function/closure stored in a register
            Call(dst, func_reg, ref args) => {
                let callee = reg!(func_reg);
                let mut arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();

                // Auto-untupling: if calling a function with arity N but passing 1 arg that's
                // a tuple of N elements, auto-destructure the tuple. This enables:
                //   [(1,2),(3,4)].map((a,b) => a + b)
                // where the lambda has 2 params but map passes 1 tuple argument.
                let func_arity = match &callee {
                    GcValue::Function(f) => Some(f.arity),
                    GcValue::Closure(ptr, _) => {
                        self.heap.get_closure(*ptr).map(|c| c.function.arity)
                    }
                    _ => None,
                };
                if let Some(arity) = func_arity {
                    if arg_values.len() == 1 && arity > 1 {
                        if let GcValue::Tuple(ptr) = &arg_values[0] {
                            if let Some(tuple) = self.heap.get_tuple(*ptr) {
                                if tuple.items.len() == arity {
                                    // Destructure the tuple into separate arguments
                                    arg_values = tuple.items.clone();
                                }
                            }
                        }
                    }
                }

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
                        // Must return to recompute cur_frame (frame was pushed)
                        return Ok(StepResult::Continue);
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
                        // Must return to recompute cur_frame (frame was pushed)
                        return Ok(StepResult::Continue);
                    }
                    _ => return Err(RuntimeError::Panic(format!("Call: expected function or closure, got {:?}", callee))),
                }
            }

            // === Process operations (async!) ===
            SelfPid(dst) => {
                set_reg!(dst, GcValue::Pid(self.pid.0));
            }

            // === GC operations ===
            GcCollect(dst) => {
                // Gather roots from current execution state (same as maybe_gc)
                let mut roots = Vec::new();
                for frame in &self.frames {
                    for val in &frame.registers {
                        roots.extend(val.gc_pointers());
                    }
                    for val in &frame.captures {
                        roots.extend(val.gc_pointers());
                    }
                }
                if let Some(ref exc) = self.current_exception {
                    roots.extend(exc.gc_pointers());
                }
                if let Some(ref val) = self.exit_value {
                    roots.extend(val.gc_pointers());
                }
                self.heap.set_roots(roots);
                let live_before = self.heap.live_objects();
                self.heap.collect();
                let live_after = self.heap.live_objects();
                self.heap.clear_roots();
                let collected = (live_before - live_after) as i64;

                // Return record with collection results
                let record = self.heap.alloc_record(
                    "GcResult".to_string(),
                    vec!["collected".to_string(), "live".to_string()],
                    vec![GcValue::Int64(collected), GcValue::Int64(live_after as i64)],
                    vec![false, false],
                );
                set_reg!(dst, GcValue::Record(record));
            }

            GcStats(dst) => {
                let stats = self.heap.stats();
                let live = self.heap.live_objects() as i64;

                // Return record with GC statistics
                let record = self.heap.alloc_record(
                    "GcStats".to_string(),
                    vec![
                        "live".to_string(),
                        "totalAllocated".to_string(),
                        "totalFreed".to_string(),
                        "collections".to_string(),
                    ],
                    vec![
                        GcValue::Int64(live),
                        GcValue::Int64(stats.total_allocated as i64),
                        GcValue::Int64(stats.total_freed as i64),
                        GcValue::Int64(stats.collections as i64),
                    ],
                    vec![false, false, false, false],
                );
                set_reg!(dst, GcValue::Record(record));
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
                // Kill a process by unregistering it AND aborting its task
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessKill: expected Pid".into())),
                };

                // Can't kill self
                let killed = if target_pid == self.pid {
                    false
                } else {
                    // First close any servers owned by this process to release ports
                    let servers = self.shared.process_servers.write().await.remove(&target_pid);
                    if let Some(server_handles) = servers {
                        if let Some(sender) = &self.shared.io_sender {
                            for handle in server_handles {
                                let (tx, _rx) = tokio::sync::oneshot::channel();
                                let request = crate::io_runtime::IoRequest::ServerClose { handle, response: tx };
                                let _ = sender.send(request);
                            }
                        }
                    }

                    // Then abort the task so it actually stops running
                    let abort_handle = self.shared.process_abort_handles.write().await.remove(&target_pid);
                    if let Some(handle) = abort_handle {
                        handle.abort();
                    }

                    // Then unregister the process (its mailbox will be dropped, messages will fail)
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
                    // Lock stdout for synchronized, atomic line output
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut handle = stdout.lock();
                    let _ = writeln!(handle, "{}", s);
                }
            }

            Println(src) => {
                let value = reg!(src);
                let s = self.heap.display_value(&value);
                self.output.push(s.clone());
                // Send to output channel if available (for REPL/TUI)
                if let Some(ref sender) = self.shared.output_sender {
                    let sender_ptr = sender as *const _ as usize;
                    Self::debug_log(&format!("[Println] sender_ptr={:#x}, sending: {}", sender_ptr, s));
                    match sender.send(s.clone()) {
                        Ok(()) => Self::debug_log(&format!("[Println] send OK")),
                        Err(e) => Self::debug_log(&format!("[Println] send FAILED: {:?}", e)),
                    }
                } else {
                    Self::debug_log(&format!("[Println] no output_sender, using stdout: {}", s));
                    // Lock stdout for synchronized, atomic line output
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut handle = stdout.lock();
                    let _ = writeln!(handle, "{}", s);
                }
            }

            // === Tail call (replaces current frame) ===
            TailCallDirect(func_idx, ref args) => {
                // Skip JIT when debugging - must use interpreted execution for breakpoints
                let use_jit = self.debug_event_sender.is_none();

                // Check for JIT-compiled version first
                let func_idx_u16 = *func_idx;
                let profiling = self.is_profiling();
                if use_jit { match args.len() {
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
                        // List sum optimization - use native sum() directly
                        if self.shared.jit_list_sum_functions.read().unwrap().contains_key(&func_idx_u16) {
                            if let GcValue::Int64List(ref list) = reg_ref!(args[0]) {
                                // Direct sum on imbl::Vector - no copy!
                                let (res, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = list.sum();
                                    (r, Some(start.elapsed()))
                                } else {
                                    (list.sum(), None)
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
                        // Tail-recursive list sum JIT: sumTR(list, acc) -> sum
                        if self.shared.jit_list_sum_tr_functions.read().unwrap().contains_key(&func_idx_u16) {
                            if let (GcValue::Int64List(ref list), GcValue::Int64(acc)) = (reg_ref!(args[0]), reg!(args[1])) {
                                // Direct sum on imbl::Vector + initial accumulator
                                let (res, duration) = if profiling {
                                    let start = Instant::now();
                                    let r = list.sum() + acc;
                                    (r, Some(start.elapsed()))
                                } else {
                                    (list.sum() + acc, None)
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
                } } // close if use_jit and match

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
                // Use reg_ref to avoid cloning the list
                let list_val = reg_ref!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    GcValue::Int64List(list) => list.is_empty(),
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
                // Must return to recompute cur_frame (frame was pushed)
                return Ok(StepResult::Continue);
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
                    let num_args = args.len();

                    // Only clear registers beyond the argument positions
                    // Registers 0..num_args will be overwritten with saved args
                    // This avoids dropping values that might trigger atomic ops
                    for reg in frame.registers.iter_mut().skip(num_args) {
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
                    let num_args = arg_values.len();
                    // Only clear registers beyond arguments
                    for reg in frame.registers.iter_mut().skip(num_args) {
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

            // Fast path for native function calls - uses index instead of string lookup
            CallNativeIdx(dst, native_idx, ref args) => {
                let native = self.shared.natives_vec.get(*native_idx as usize)
                    .ok_or_else(|| RuntimeError::Panic(format!("Invalid native index: {}", native_idx)))?
                    .clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                let result = (native.func)(&arg_values, &mut self.heap)?;
                set_reg!(dst, result);
            }

            // === Extension function calls ===
            CallExtension(dst, name_idx, ref args) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("CallExtension: expected string constant".into())),
                };

                // Convert GcValues to extension Values
                let arg_values: Vec<nostos_extension::Value> = args.iter().map(|r| {
                    let gc_val = reg!(*r);
                    let vm_val = self.heap.gc_to_value(&gc_val);
                    vm_value_to_ext(&vm_val)
                }).collect();

                // Get extension manager and call function
                let extensions_guard = self.shared.extensions.read().unwrap();
                let result = if let Some(ref ext_mgr) = *extensions_guard {
                    ext_mgr.call(&name, &arg_values, nostos_extension::Pid(self.pid.0))
                } else {
                    Err(format!("No extension manager configured"))
                };
                drop(extensions_guard);

                match result {
                    Ok(ext_val) => {
                        // Convert extension Value back to VM Value
                        let vm_val = ext_value_to_vm(&ext_val);
                        // Allocate on process heap
                        let gc_val = self.heap.value_to_gc(&vm_val);
                        set_reg!(dst, gc_val);
                    }
                    Err(e) => {
                        return Err(RuntimeError::Panic(format!("Extension call failed: {}", e)));
                    }
                }
            }

            // Fast path for extension function calls - uses index instead of string lookup
            CallExtensionIdx(dst, ext_idx, ref args) => {
                // Convert GcValues to extension Values
                let arg_values: Vec<nostos_extension::Value> = args.iter().map(|r| {
                    let gc_val = reg!(*r);
                    let vm_val = self.heap.gc_to_value(&gc_val);
                    vm_value_to_ext(&vm_val)
                }).collect();

                // Get extension manager and call function by index
                let extensions_guard = self.shared.extensions.read().unwrap();
                let result = if let Some(ref ext_mgr) = *extensions_guard {
                    ext_mgr.call_by_index(*ext_idx, &arg_values, nostos_extension::Pid(self.pid.0))
                } else {
                    Err(format!("No extension manager configured"))
                };
                drop(extensions_guard);

                match result {
                    Ok(ext_val) => {
                        // Convert extension Value back to VM Value
                        let vm_val = ext_value_to_vm(&ext_val);
                        // Allocate on process heap
                        let gc_val = self.heap.value_to_gc(&vm_val);
                        set_reg!(dst, gc_val);
                    }
                    Err(e) => {
                        return Err(RuntimeError::Panic(format!("Extension call failed: {}", e)));
                    }
                }
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
                    GcValue::ReactiveRecord(rec) => {
                        // Handle special introspection fields
                        if field_name == "parents" {
                            // Return List[(ReactiveRecord, String)] of parent references
                            let parents = rec.get_parents();
                            let mut items = Vec::new();
                            for (parent_arc, field_name) in parents {
                                let parent_gc = GcValue::ReactiveRecord(parent_arc);
                                let field_name_gc = GcValue::String(self.heap.alloc_string(field_name));
                                let tuple_ptr = self.heap.alloc_tuple(vec![parent_gc, field_name_gc]);
                                items.push(GcValue::Tuple(tuple_ptr));
                            }
                            let list = self.heap.make_list(items);
                            set_reg!(dst, GcValue::List(list));
                        } else if field_name == "children" {
                            // Return List[ReactiveRecord] of child reactive records
                            let children = rec.get_children();
                            let items: Vec<GcValue> = children.into_iter()
                                .map(|child| GcValue::ReactiveRecord(child))
                                .collect();
                            let list = self.heap.make_list(items);
                            set_reg!(dst, GcValue::List(list));
                        } else {
                            let idx = rec.field_names.iter().position(|n| n == &field_name)
                                .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                            let value = rec.get_field(idx)
                                .ok_or_else(|| RuntimeError::Panic("Failed to read reactive record field".into()))?;
                            // Convert Value to GcValue
                            let gc_value = self.heap.value_to_gc(&value);

                            // Track dependency if we're inside an RHtml render context
                            if let Some(current_component) = self.reactive_context.render_stack.last() {
                                let record_id = rec.id;
                                let deps = self.reactive_context.dependencies.entry(record_id).or_insert_with(Vec::new);
                                if !deps.contains(current_component) {
                                    deps.push(current_component.clone());
                                }
                            }

                            // Invoke onRead callbacks synchronously
                            let read_callbacks = rec.get_read_callbacks();
                            if !read_callbacks.is_empty() {
                                let field_name_gc = self.heap.value_to_gc(&Value::String(field_name.into()));
                                let value_gc = gc_value.clone();

                                // Push callbacks in reverse order so they execute in forward order
                                for callback in read_callbacks.into_iter().rev() {
                                    let (func, captures) = match callback {
                                        Value::Closure(c) => {
                                            let gc_captures: Vec<GcValue> = c.captures.iter()
                                                .map(|v| self.heap.value_to_gc(v))
                                                .collect();
                                            (c.function.clone(), gc_captures)
                                        }
                                        Value::Function(f) => (f, vec![]),
                                        _ => continue,
                                    };

                                    // Set up registers with arguments: (fieldName, value)
                                    let reg_count = func.code.register_count as usize;
                                    let mut registers = vec![GcValue::Unit; reg_count];
                                    if reg_count > 0 { registers[0] = field_name_gc.clone(); }
                                    if reg_count > 1 { registers[1] = value_gc.clone(); }

                                    // Push frame with return_reg = None (callback, no return value needed)
                                    self.frames.push(CallFrame {
                                        function: func,
                                        ip: 0,
                                        registers,
                                        captures,
                                        return_reg: None,
                                    });
                                    // TODO: This callback pattern has a potential cur_frame staleness issue
                                    // but the set_reg! below needs to run first, so we don't return here.
                                    // The callback will run on next step() call after this instruction completes.
                                }
                            }

                            set_reg!(dst, gc_value);
                        }
                    }
                    _ => return Err(RuntimeError::Panic("GetField expects record, variant, tuple, or reactive record".into())),
                }
            }

            SetField(record_reg, field_idx, value_reg) => {
                let field_name = match get_const!(field_idx) {
                    Value::String(s) => (*s).clone(),
                    _ => return Err(RuntimeError::Panic("SetField: field name must be string".into())),
                };
                let value = reg!(value_reg);
                let rec_val = reg!(record_reg);
                match rec_val {
                    GcValue::Record(ptr) => {
                        let rec = self.heap.get_record_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".into()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        if !rec.mutable_fields[idx] {
                            return Err(RuntimeError::Panic(format!("Field '{}' is not mutable", field_name)));
                        }
                        rec.fields[idx] = value;
                    }
                    GcValue::ReactiveRecord(rec) => {
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;

                        // Convert GcValue to Value for storage
                        let value_as_value = self.heap.gc_to_value(&value);

                        // Get old value for callbacks
                        let old_value = rec.get_field(idx);

                        // Check if old value was a reactive record (need to remove parent ref)
                        if let Some(ref ov) = old_value {
                            if let Value::ReactiveRecord(old_child) = ov {
                                old_child.remove_parent(Arc::as_ptr(&rec));
                            }
                        }

                        // Set the new value
                        rec.set_field(idx, value_as_value.clone());

                        // If new value is a reactive record, add parent reference
                        if let GcValue::ReactiveRecord(new_child) = &value {
                            new_child.add_parent(Arc::downgrade(&rec), idx as u16);
                        }

                        // Invoke callbacks synchronously by pushing call frames
                        let callbacks = rec.get_callbacks();
                        if !callbacks.is_empty() {
                            let field_name_gc = self.heap.value_to_gc(&Value::String(field_name.into()));
                            let old_val_gc = self.heap.value_to_gc(&old_value.unwrap_or(Value::Unit));
                            let new_val_gc = self.heap.value_to_gc(&value_as_value);

                            // Push callbacks in reverse order so they execute in forward order
                            for callback in callbacks.into_iter().rev() {
                                let (func, captures) = match callback {
                                    Value::Closure(c) => {
                                        let gc_captures: Vec<GcValue> = c.captures.iter()
                                            .map(|v| self.heap.value_to_gc(v))
                                            .collect();
                                        (c.function.clone(), gc_captures)
                                    }
                                    Value::Function(f) => (f, vec![]),
                                    _ => continue,
                                };

                                // Set up registers with arguments
                                let reg_count = func.code.register_count as usize;
                                let mut registers = vec![GcValue::Unit; reg_count];
                                if reg_count > 0 { registers[0] = field_name_gc.clone(); }
                                if reg_count > 1 { registers[1] = old_val_gc.clone(); }
                                if reg_count > 2 { registers[2] = new_val_gc.clone(); }

                                // Push frame with return_reg = None (callback, no return value needed)
                                self.frames.push(CallFrame {
                                    function: func,
                                    ip: 0,
                                    registers,
                                    captures,
                                    return_reg: None,
                                });
                                // TODO: Same callback cur_frame staleness issue as in GetField
                            }
                        }

                        // Queue dependent components for re-render
                        let record_id = rec.id;

                        // Track this record as changed
                        if !self.reactive_context.changed_record_ids.contains(&record_id) {
                            self.reactive_context.changed_record_ids.push(record_id);
                        }

                        if let Some(deps) = self.reactive_context.dependencies.get(&record_id) {
                            for comp_id in deps {
                                if !self.reactive_context.pending_rerenders.contains(comp_id) {
                                    self.reactive_context.pending_rerenders.push(comp_id.clone());
                                }
                            }
                        }
                    }
                    _ => return Err(RuntimeError::Panic("SetField expects record or reactive record".into())),
                }
            }

            ReactiveAddCallback(record_reg, callback_reg) => {
                let callback = reg!(callback_reg);
                let rec_val = reg!(record_reg);
                match rec_val {
                    GcValue::ReactiveRecord(rec) => {
                        // Convert GcValue callback to Value and store
                        let callback_value = self.heap.gc_to_value(&callback);
                        rec.add_callback(callback_value);
                    }
                    _ => return Err(RuntimeError::Panic("ReactiveAddCallback expects reactive record".into())),
                }
            }

            ReactiveAddReadCallback(record_reg, callback_reg) => {
                let callback = reg!(callback_reg);
                let rec_val = reg!(record_reg);
                match rec_val {
                    GcValue::ReactiveRecord(rec) => {
                        // Convert GcValue callback to Value and store
                        let callback_value = self.heap.gc_to_value(&callback);
                        rec.add_read_callback(callback_value);
                    }
                    _ => return Err(RuntimeError::Panic("ReactiveAddReadCallback expects reactive record".into())),
                }
            }

            MakeList(dst, ref elems) => {
                let values: Vec<GcValue> = elems.iter().map(|r| reg!(*r)).collect();
                let list = self.heap.make_list(values);
                set_reg!(dst, GcValue::List(list));
            }

            MakeInt64List(dst, ref elems) => {
                let mut values = Vec::with_capacity(elems.len());
                for r in elems.iter() {
                    match reg!(*r) {
                        GcValue::Int64(n) => values.push(n),
                        other => return Err(RuntimeError::Panic(
                            format!("MakeInt64List: expected Int64, got {:?}", other.type_name(&self.heap))
                        )),
                    }
                }
                set_reg!(dst, GcValue::Int64List(GcInt64List::from_vec(values)));
            }

            Cons(dst, head, tail) => {
                let head_val = reg!(head);
                // Use reg_ref to avoid cloning the tail list
                let tail_val = reg_ref!(tail);
                match tail_val {
                    GcValue::List(tail_list) => {
                        // Auto-specialize: if head is Int64 and tail is empty,
                        // create an Int64List for better performance on subsequent ops
                        if tail_list.is_empty() {
                            if let GcValue::Int64(n) = head_val {
                                set_reg!(dst, GcValue::Int64List(GcInt64List::from_vec(vec![n])));
                            } else {
                                let new_list = tail_list.cons(head_val);
                                set_reg!(dst, GcValue::List(new_list));
                            }
                        } else {
                            // O(log n) cons using persistent data structure
                            let new_list = tail_list.cons(head_val);
                            set_reg!(dst, GcValue::List(new_list));
                        }
                    }
                    GcValue::Int64List(tail_list) => {
                        // Specialized path: if head is Int64, keep it as Int64List
                        // Now O(log n) using imbl::Vector
                        if let GcValue::Int64(n) = head_val {
                            let new_list = tail_list.cons(n);
                            set_reg!(dst, GcValue::Int64List(new_list));
                        } else {
                            // Head is not Int64, convert to regular list
                            let items: Vec<GcValue> = std::iter::once(head_val)
                                .chain(tail_list.iter().map(GcValue::Int64))
                                .collect();
                            let list = self.heap.make_list(items);
                            set_reg!(dst, GcValue::List(list));
                        }
                    }
                    _ => {
                        return Err(RuntimeError::Panic("Cons: tail must be a list".into()));
                    }
                }
            }

            Decons(head_dst, tail_dst, list_reg) => {
                // Use reg_ref to avoid cloning the entire list
                let list_val = reg_ref!(list_reg);
                match list_val {
                    GcValue::List(list) => {
                        if !list.is_empty() {
                            // Use unchecked versions - we just verified non-empty
                            let head = list.head_unchecked().clone();
                            let tail = list.tail_unchecked();
                            set_reg!(head_dst, head);
                            set_reg!(tail_dst, GcValue::List(tail));
                        } else {
                            return Err(RuntimeError::Panic("Decons: empty list".into()));
                        }
                    }
                    GcValue::Int64List(list) => {
                        if !list.is_empty() {
                            // Use unchecked versions - we just verified non-empty
                            let head = list.head_unchecked();
                            let tail = list.tail_unchecked();
                            set_reg!(head_dst, GcValue::Int64(head));
                            set_reg!(tail_dst, GcValue::Int64List(tail));
                        } else {
                            return Err(RuntimeError::Panic("Decons: empty Int64List".into()));
                        }
                    }
                    _ => {
                        return Err(RuntimeError::Panic("Decons: expected list".into()));
                    }
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

            // === Specialized Int64List operations ===
            TestNilInt64(dst, list_reg) => {
                let result = match reg_ref!(list_reg) {
                    GcValue::Int64List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            DeconsInt64(head_dst, tail_dst, list_reg) => {
                let list_val = reg_ref!(list_reg);
                if let GcValue::Int64List(list) = list_val {
                    if !list.is_empty() {
                        // Use unchecked versions - we just verified non-empty
                        let head = list.head_unchecked();
                        let tail = list.tail_unchecked();
                        set_reg!(head_dst, GcValue::Int64(head));
                        set_reg!(tail_dst, GcValue::Int64List(tail));
                    } else {
                        return Err(RuntimeError::Panic("DeconsInt64: empty list".into()));
                    }
                } else {
                    return Err(RuntimeError::Panic("DeconsInt64: expected Int64List".into()));
                }
            }

            ConsInt64(dst, head_reg, tail_reg) => {
                let head = match reg!(head_reg) {
                    GcValue::Int64(n) => n,
                    _ => return Err(RuntimeError::Panic("ConsInt64: head must be Int64".into())),
                };
                let tail = reg_ref!(tail_reg);
                match tail {
                    GcValue::Int64List(list) => {
                        let new_list = list.cons(head);
                        set_reg!(dst, GcValue::Int64List(new_list));
                    }
                    GcValue::List(list) if list.is_empty() => {
                        // Empty regular list - create new Int64List
                        set_reg!(dst, GcValue::Int64List(GcInt64List::from_vec(vec![head])));
                    }
                    _ => {
                        return Err(RuntimeError::Panic("ConsInt64: tail must be Int64List".into()));
                    }
                }
            }

            RangeInt64List(dst, n_reg) => {
                let n = match reg!(n_reg) {
                    GcValue::Int64(n) => n,
                    _ => return Err(RuntimeError::Panic("RangeInt64List: expected Int64".into())),
                };
                let items: Vec<i64> = (1..=n).rev().collect();
                set_reg!(dst, GcValue::Int64List(GcInt64List::from_vec(items)));
            }

            ToInt64List(dst, src_reg) => {
                let src = reg_ref!(src_reg);
                match src {
                    GcValue::List(list) => {
                        let mut items = Vec::with_capacity(list.len());
                        for item in list.iter() {
                            match item {
                                GcValue::Int64(n) => items.push(*n),
                                _ => return Err(RuntimeError::Panic("ToInt64List: all elements must be Int64".into())),
                            }
                        }
                        set_reg!(dst, GcValue::Int64List(GcInt64List::from_vec(items)));
                    }
                    GcValue::Int64List(list) => {
                        set_reg!(dst, GcValue::Int64List(list.clone()));
                    }
                    _ => return Err(RuntimeError::Panic("ToInt64List: expected List".into())),
                }
            }

            SumInt64List(dst, src_reg) => {
                let src = reg_ref!(src_reg);
                if let GcValue::Int64List(list) = src {
                    set_reg!(dst, GcValue::Int64(list.sum()));
                } else {
                    return Err(RuntimeError::Panic("SumInt64List: expected Int64List".into()));
                }
            }

            ListHead(dst, list_reg) => {
                let list_val = reg_ref!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => {
                        // Use O(log n) head() instead of O(n) items().first()
                        if let Some(head) = list.head() {
                            head.clone()
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                    }
                    GcValue::Int64List(list) => {
                        if let Some(head) = list.head() {
                            GcValue::Int64(head)
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
                    GcValue::Int64List(list) => {
                        if !list.is_empty() {
                            GcValue::Int64List(list.tail())
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
                    GcValue::Int64List(list) => list.len() as i64,
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

            // === List operations ===
            ListSum(dst, src) => {
                let val = reg!(src);
                match val {
                    GcValue::List(list) => {
                        let mut sum: i64 = 0;
                        for item in list.iter() {
                            if let GcValue::Int64(n) = item {
                                sum += n;
                            } else {
                                return Err(RuntimeError::Panic(
                                    format!("listSum: expected Int64, got {:?}", item.type_name(&self.heap))
                                ));
                            }
                        }
                        set_reg!(dst, GcValue::Int64(sum));
                    }
                    GcValue::Int64List(list) => {
                        // Fast path for specialized Int64List
                        set_reg!(dst, GcValue::Int64(list.sum()));
                    }
                    _ => {
                        return Err(RuntimeError::Panic(
                            format!("listSum: expected List, got {:?}", val.type_name(&self.heap))
                        ));
                    }
                }
            }

            ListProduct(dst, src) => {
                let val = reg!(src);
                match val {
                    GcValue::List(list) => {
                        let mut product: i64 = 1;
                        for item in list.iter() {
                            if let GcValue::Int64(n) = item {
                                product *= n;
                            } else {
                                return Err(RuntimeError::Panic(
                                    format!("listProduct: expected Int64, got {:?}", item.type_name(&self.heap))
                                ));
                            }
                        }
                        set_reg!(dst, GcValue::Int64(product));
                    }
                    GcValue::Int64List(list) => {
                        // Fast path for specialized Int64List
                        set_reg!(dst, GcValue::Int64(list.product()));
                    }
                    _ => {
                        return Err(RuntimeError::Panic(
                            format!("listProduct: expected List, got {:?}", val.type_name(&self.heap))
                        ));
                    }
                }
            }

            ListMax(dst, src) => {
                let val = reg!(src);
                match val {
                    GcValue::List(list) => {
                        let mut max_val: Option<i64> = None;
                        for item in list.iter() {
                            if let GcValue::Int64(n) = item {
                                max_val = Some(max_val.map_or(*n, |m| m.max(*n)));
                            } else {
                                return Err(RuntimeError::Panic(
                                    format!("listMax: expected Int64, got {:?}", item.type_name(&self.heap))
                                ));
                            }
                        }
                        if let Some(m) = max_val {
                            set_reg!(dst, GcValue::Int64(m));
                        } else {
                            return Err(RuntimeError::Panic("listMax: empty list".into()));
                        }
                    }
                    GcValue::Int64List(list) => {
                        if list.is_empty() {
                            return Err(RuntimeError::Panic("listMax: empty list".into()));
                        }
                        let max_val = list.iter().max().unwrap();
                        set_reg!(dst, GcValue::Int64(max_val));
                    }
                    _ => {
                        return Err(RuntimeError::Panic(
                            format!("listMax: expected List, got {:?}", val.type_name(&self.heap))
                        ));
                    }
                }
            }

            ListMin(dst, src) => {
                let val = reg!(src);
                match val {
                    GcValue::List(list) => {
                        let mut min_val: Option<i64> = None;
                        for item in list.iter() {
                            if let GcValue::Int64(n) = item {
                                min_val = Some(min_val.map_or(*n, |m| m.min(*n)));
                            } else {
                                return Err(RuntimeError::Panic(
                                    format!("listMin: expected Int64, got {:?}", item.type_name(&self.heap))
                                ));
                            }
                        }
                        if let Some(m) = min_val {
                            set_reg!(dst, GcValue::Int64(m));
                        } else {
                            return Err(RuntimeError::Panic("listMin: empty list".into()));
                        }
                    }
                    GcValue::Int64List(list) => {
                        if list.is_empty() {
                            return Err(RuntimeError::Panic("listMin: empty list".into()));
                        }
                        let min_val = list.iter().min().unwrap();
                        set_reg!(dst, GcValue::Int64(min_val));
                    }
                    _ => {
                        return Err(RuntimeError::Panic(
                            format!("listMin: expected List, got {:?}", val.type_name(&self.heap))
                        ));
                    }
                }
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
                let coll_val = reg_ref!(coll);
                let value = match coll_val {
                    GcValue::List(list) => {
                        // Use O(log n) get() instead of O(n) items().get()
                        list.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Int64List(list) => {
                        // Use O(log n) indexing via imbl::Vector
                        let val = list.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Int64(val)
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = self.heap.get_tuple(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        tuple.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Array(ptr) => {
                        let array = self.heap.get_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".into()))?;
                        array.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Int64Array(ptr) => {
                        let array = self.heap.get_int64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".into()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Int64(val)
                    }
                    GcValue::Float64Array(ptr) => {
                        let array = self.heap.get_float64_array(*ptr)
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
                    GcValue::Int64List(list) => list.is_empty(),
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
                    // Int64List concat - keep as Int64List
                    (GcValue::Int64List(la), GcValue::Int64List(lb)) => {
                        let items: Vec<i64> = la.iter().chain(lb.iter()).collect();
                        set_reg!(dst, GcValue::Int64List(GcInt64List::from_vec(items)));
                    }
                    // Mixed: Int64List + List or List + Int64List -> regular List
                    (GcValue::Int64List(la), GcValue::List(lb)) => {
                        let mut items: Vec<GcValue> = la.iter().map(GcValue::Int64).collect();
                        items.extend(lb.items().iter().cloned());
                        let new_list = self.heap.make_list(items);
                        set_reg!(dst, GcValue::List(new_list));
                    }
                    (GcValue::List(la), GcValue::Int64List(lb)) => {
                        let mut items: Vec<GcValue> = la.items().to_vec();
                        items.extend(lb.iter().map(GcValue::Int64));
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
                let shared_for_cleanup = self.shared.clone();
                let spawn_task = AssertSend(async move {
                    debug_log_to_file(&format!("[AsyncVM] Spawned process {} running function: {}", child_pid.0, func_name));
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
                    let result = process.run().await;

                    // Log errors from spawned processes (helps debug silent failures)
                    if let Err(ref e) = result {
                        debug_log_to_file(&format!("[AsyncVM] Spawned process {} error: {:?}", child_pid.0, e));
                    }
                    debug_log_to_file(&format!("[AsyncVM] Spawned process {} finished", child_pid.0));

                    // Unregister on exit (cleanup abort handle too)
                    shared_clone.unregister_process(child_pid).await;
                    shared_clone.process_abort_handles.write().await.remove(&child_pid);
                });

                // In interactive mode, use the long-lived IO runtime for spawned processes
                // so they survive past individual eval calls. Otherwise use the current runtime.
                let is_interactive = self.shared.interactive_mode.load(Ordering::SeqCst);
                let has_spawn_handle = self.shared.spawn_runtime_handle.is_some();
                debug_log_to_file(&format!("[AsyncVM] Spawn: interactive={}, has_handle={}", is_interactive, has_spawn_handle));
                let join_handle = if let Some(ref handle) = self.shared.spawn_runtime_handle {
                    if is_interactive {
                        handle.spawn(spawn_task)
                    } else {
                        tokio::spawn(spawn_task)
                    }
                } else {
                    tokio::spawn(spawn_task)
                };

                // Store the abort handle so Process.kill can actually stop the task
                shared_for_cleanup.process_abort_handles.write().await.insert(child_pid, join_handle.abort_handle());

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

            // === VM Stats ===
            VmStats(dst) => {
                let (spawned, exited, active) = self.shared.process_stats().await;
                let values = vec![
                    GcValue::Int64(spawned as i64),
                    GcValue::Int64(exited as i64),
                    GcValue::Int64(active as i64),
                ];
                let tuple = self.heap.alloc_tuple(values);
                set_reg!(dst, GcValue::Tuple(tuple));
            }

            // === Runtime info ===
            RuntimeIsInteractive(dst) => {
                let is_interactive = self.shared.interactive_mode.load(Ordering::SeqCst);
                set_reg!(dst, GcValue::Bool(is_interactive));
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

            MakeReactiveRecord(dst, type_idx, ref field_regs) => {
                let type_name = match get_const!(type_idx) {
                    Value::String(s) => (*s).clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let gc_fields: Vec<GcValue> = field_regs.iter().map(|&r| reg!(r)).collect();
                // Look up type in static types first, then dynamic_types (eval-defined)
                let type_info = self.shared.types.read().unwrap().get(&type_name).cloned()
                    .or_else(|| self.shared.dynamic_types.read().unwrap().get(&type_name).cloned());
                let field_names: Vec<String> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.name.clone()).collect())
                    .unwrap_or_else(|| (0..gc_fields.len()).map(|i| format!("_{}", i)).collect());

                // Convert GcValue fields to Value fields and compute reactive field mask
                let mut reactive_mask: u64 = 0;
                let fields: Vec<Value> = gc_fields.iter().enumerate().map(|(i, gv)| {
                    // Check if this field is a reactive record
                    if matches!(gv, GcValue::ReactiveRecord(_)) {
                        reactive_mask |= 1 << i;
                    }
                    self.heap.gc_to_value(gv)
                }).collect();

                let reactive_record = Arc::new(ReactiveRecordValue::new(
                    type_name,
                    field_names,
                    fields,
                    reactive_mask,
                ));

                // Register this record as parent of any child reactive records
                for (i, gv) in gc_fields.iter().enumerate() {
                    if let GcValue::ReactiveRecord(child) = gv {
                        child.add_parent(Arc::downgrade(&reactive_record), i as u16);
                    }
                }

                set_reg!(dst, GcValue::ReactiveRecord(reactive_record));
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
                let mut arg_values: Vec<GcValue> = args.iter().map(|&r| reg!(r)).collect();
                // Preserve return_reg from current frame
                let return_reg = self.frames.last().unwrap().return_reg;

                // Auto-untupling for tail calls (same logic as regular Call)
                let func_arity = match &func_val {
                    GcValue::Function(f) => Some(f.arity),
                    GcValue::Closure(ptr, _) => {
                        self.heap.get_closure(*ptr).map(|c| c.function.arity)
                    }
                    _ => None,
                };
                if let Some(arity) = func_arity {
                    if arg_values.len() == 1 && arity > 1 {
                        if let GcValue::Tuple(ptr) = &arg_values[0] {
                            if let Some(tuple) = self.heap.get_tuple(*ptr) {
                                if tuple.items.len() == arity {
                                    arg_values = tuple.items.clone();
                                }
                            }
                        }
                    }
                }

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
                        // Must return to recompute cur_frame (frame was replaced)
                        return Ok(StepResult::Continue);
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
                        // Must return to recompute cur_frame (frame was replaced)
                        return Ok(StepResult::Continue);
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function or Closure".to_string(),
                        found: format!("{:?}", func_val),
                    }),
                }
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
                    debug_log_to_file(&format!("[VM] ServerBind: sending request for port={} from pid={}", port, self.pid.0));
                    let request = IoRequest::ServerBind { port, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    debug_log_to_file("[VM] ServerBind: waiting for response");
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    debug_log_to_file("[VM] ServerBind: got response");
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        // Track this server handle as owned by this process
                        if let GcValue::Int64(handle) = gc_value {
                            let mut servers = self.shared.process_servers.write().await;
                            servers.entry(self.pid).or_insert_with(Vec::new).push(handle as u64);
                        }
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
                    debug_log_to_file(&format!("[VM] ServerAccept: sending request from pid={}", self.pid.0));
                    let request = IoRequest::ServerAccept { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    debug_log_to_file("[VM] ServerAccept: waiting for request from browser");
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    debug_log_to_file("[VM] ServerAccept: got request");
                    if let Some(gc_value) = self.handle_io_result(result, "io_error")? {
                        set_reg!(dst, gc_value);
                    }
                    // Trigger GC after accepting request (allocates ServerRequest with lists)
                    self.maybe_gc();
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

            // WebSocket operations
            WebSocketAccept(dst, request_id_reg) => {
                let request_id = match reg!(request_id_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    debug_log_to_file(&format!("[VM] WebSocketAccept: sending request for request_id={} from pid={}", request_id, self.pid.0));
                    let request = IoRequest::WebSocketAccept { request_id, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    debug_log_to_file("[VM] WebSocketAccept: waiting for connection");
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    debug_log_to_file("[VM] WebSocketAccept: got response");
                    match result {
                        Ok(IoResponseValue::Int(ws_handle)) => {
                            set_reg!(dst, GcValue::Int64(ws_handle));
                        }
                        Ok(_) => {
                            return Err(RuntimeError::IOError("Unexpected response type".to_string()));
                        }
                        Err(e) => {
                            self.throw_exception("websocket_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            WebSocketSend(dst, request_id_reg, message_reg) => {
                let request_id = match reg!(request_id_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let message = match reg!(message_reg) {
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
                    let request = IoRequest::WebSocketSend { request_id, message, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(_) => {
                            set_reg!(dst, GcValue::Unit);
                        }
                        Err(e) => {
                            self.throw_exception("websocket_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            WebSocketReceive(dst, request_id_reg) => {
                let request_id = match reg!(request_id_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::WebSocketReceive { request_id, response: tx };
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
                            self.throw_exception("websocket_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            WebSocketClose(dst, request_id_reg) => {
                let request_id = match reg!(request_id_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::WebSocketClose { request_id, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    match result {
                        Ok(_) => {
                            set_reg!(dst, GcValue::Unit);
                        }
                        Err(e) => {
                            self.throw_exception("websocket_error", format!("{}", e))?;
                        }
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
                    GcValue::Int64List(list) => {
                        // Handle specialized Int64List
                        use crate::io_runtime::PgParam;
                        list.iter().map(|i| PgParam::Int(i)).collect()
                    }
                    GcValue::Tuple(ptr) => {
                        // Handle tuples for heterogeneous parameter types
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        let mut pg_params = Vec::new();
                        for item in &tuple.items {
                            let param = self.gc_value_to_pg_param(item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List or Tuple".to_string(),
                        found: "non-list/tuple".to_string(),
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
                    GcValue::Int64List(list) => {
                        // Handle specialized Int64List
                        use crate::io_runtime::PgParam;
                        list.iter().map(|i| PgParam::Int(i)).collect()
                    }
                    GcValue::Tuple(ptr) => {
                        // Handle tuples for heterogeneous parameter types
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        let mut pg_params = Vec::new();
                        for item in &tuple.items {
                            let param = self.gc_value_to_pg_param(item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List or Tuple".to_string(),
                        found: "non-list/tuple".to_string(),
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
                    GcValue::Int64List(list) => {
                        // Handle specialized Int64List
                        use crate::io_runtime::PgParam;
                        list.iter().map(|i| PgParam::Int(i)).collect()
                    }
                    GcValue::Tuple(ptr) => {
                        // Handle tuples for heterogeneous parameter types
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        let mut pg_params = Vec::new();
                        for item in &tuple.items {
                            let param = self.gc_value_to_pg_param(item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List or Tuple".to_string(),
                        found: "non-list/tuple".to_string(),
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
                    GcValue::Int64List(list) => {
                        // Handle specialized Int64List
                        use crate::io_runtime::PgParam;
                        list.iter().map(|i| PgParam::Int(i)).collect()
                    }
                    GcValue::Tuple(ptr) => {
                        // Handle tuples for heterogeneous parameter types
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        let mut pg_params = Vec::new();
                        for item in &tuple.items {
                            let param = self.gc_value_to_pg_param(item)?;
                            pg_params.push(param);
                        }
                        pg_params
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List or Tuple".to_string(),
                        found: "non-list/tuple".to_string(),
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

            // LISTEN/NOTIFY builtins
            PgListenConnect(dst, conn_str_reg) => {
                let conn_str = match reg!(conn_str_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid connection string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgListenConnect { connection_string: conn_str, response: tx };
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
                            self.throw_exception("connect_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgListen(dst, handle_reg, channel_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (listener handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let channel = match reg!(channel_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid channel string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgListen { handle, channel, response: tx };
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
                            self.throw_exception("listen_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgUnlisten(dst, handle_reg, channel_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (listener handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let channel = match reg!(channel_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid channel string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgUnlisten { handle, channel, response: tx };
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
                            self.throw_exception("unlisten_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgNotify(dst, handle_reg, channel_reg, payload_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (pg handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let channel = match reg!(channel_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid channel string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let payload = match reg!(payload_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr).map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid payload string".to_string()))?
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgNotify { handle, channel, payload, response: tx };
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
                            self.throw_exception("notify_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            PgAwaitNotification(dst, handle_reg, timeout_reg) => {
                let handle = match reg!(handle_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (listener handle)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let timeout_ms = match reg!(timeout_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int (timeout ms)".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::PgAwaitNotification { handle, timeout_ms, response: tx };
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
                            self.throw_exception("await_notification_error", format!("{}", e))?;
                        }
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === Selenium WebDriver Operations ===
            SeleniumConnect(dst, url_reg) => {
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumConnect { webdriver_url: url, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumGoto(dst, driver_reg, url_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let url = match reg!(url_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumGoto { driver_handle, url, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumClick(dst, driver_reg, selector_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let selector = match reg!(selector_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumClick { driver_handle, selector, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumText(dst, driver_reg, selector_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let selector = match reg!(selector_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumText { driver_handle, selector, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumSendKeys(dst, driver_reg, selector_reg, text_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let selector = match reg!(selector_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let text = match reg!(text_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumSendKeys { driver_handle, selector, text, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumExecuteJs(dst, driver_reg, script_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let script = match reg!(script_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumExecuteJs { driver_handle, script, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumExecuteJsWithArgs(dst, driver_reg, script_reg, args_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let script = match reg!(script_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let args = match reg!(args_reg) {
                    GcValue::List(list) => {
                        list.iter().map(|v| {
                            match v {
                                GcValue::String(ptr) => self.heap.get_string(*ptr).map(|s| s.data.clone()).unwrap_or_default(),
                                GcValue::Int64(n) => n.to_string(),
                                GcValue::Float64(n) => n.to_string(),
                                GcValue::Bool(b) => b.to_string(),
                                _ => String::new(),
                            }
                        }).collect()
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: "non-list".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumExecuteJsWithArgs { driver_handle, script, args, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumWaitFor(dst, driver_reg, selector_reg, timeout_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let selector = match reg!(selector_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let timeout_ms = match reg!(timeout_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumWaitFor { driver_handle, selector, timeout_ms, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumGetAttribute(dst, driver_reg, selector_reg, attr_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let selector = match reg!(selector_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let attribute = match reg!(attr_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumGetAttribute { driver_handle, selector, attribute, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumExists(dst, driver_reg, selector_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let selector = match reg!(selector_reg) {
                    GcValue::String(ptr) => self.heap.get_string(ptr).map(|s| s.data.clone()).unwrap_or_default(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumExists { driver_handle, selector, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
                    }
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            SeleniumClose(dst, driver_reg) => {
                let driver_handle = match reg!(driver_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "non-int".to_string(),
                    }),
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = IoRequest::SeleniumClose { driver_handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let result = rx.await.map_err(|_| RuntimeError::IOError("IO response channel closed".to_string()))?;
                    if let Some(gc_value) = self.handle_io_result(result, "selenium_error")? {
                        set_reg!(dst, gc_value);
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
                    GcValue::Int64List(_) => "Int64List",
                    GcValue::Float32Array(_) => "Float32Array",
                    GcValue::Buffer(_) => "Buffer",
                    GcValue::NativeHandle(_) => "NativeHandle",
                    GcValue::ReactiveRecord(_) => "ReactiveRecord",
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

            ReactiveId(dst, val_reg) => {
                let id = match reg!(val_reg) {
                    GcValue::ReactiveRecord(rec) => rec.id,
                    _ => return Err(RuntimeError::TypeError { expected: "ReactiveRecord".to_string(), found: "non-reactive".to_string() }),
                };
                set_reg!(dst, GcValue::Int64(id as i64));
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

            RequestToType(dst, request_reg, type_name_reg) => {
                let type_name = match reg!(type_name_reg) {
                    GcValue::String(ptr) => {
                        self.heap.get_string(ptr.clone())
                            .map(|s| s.data.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "non-string".to_string() }),
                };
                let request = reg!(request_reg).clone();
                let result = self.request_to_type(&type_name, request);
                set_reg!(dst, result);
            }

            // === String Buffer (for efficient HTML rendering) ===
            BufferNew(dst) => {
                let buf_ptr = self.heap.alloc_buffer();
                set_reg!(dst, GcValue::Buffer(buf_ptr));
            }

            BufferAppend(buf_reg, str_reg) => {
                let buf = reg!(buf_reg);
                let str_val = reg!(str_reg);
                match &buf {
                    GcValue::Buffer(buf_ptr) => {
                        let s = self.heap.display_value(&str_val);
                        if let Some(gc_buf) = self.heap.get_buffer(buf_ptr.clone()) {
                            gc_buf.append(&s);
                        }
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Buffer".to_string(),
                            found: buf.type_name(&self.heap).to_string(),
                        });
                    }
                }
            }

            BufferToString(dst, buf_reg) => {
                let buf = reg!(buf_reg);
                match &buf {
                    GcValue::Buffer(buf_ptr) => {
                        if let Some(gc_buf) = self.heap.get_buffer(buf_ptr.clone()) {
                            let s = gc_buf.to_string();
                            let str_ptr = self.heap.alloc_string(s);
                            set_reg!(dst, GcValue::String(str_ptr));
                        } else {
                            return Err(RuntimeError::TypeError {
                                expected: "Buffer".to_string(),
                                found: "invalid buffer".to_string(),
                            });
                        }
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Buffer".to_string(),
                            found: buf.type_name(&self.heap).to_string(),
                        });
                    }
                }
            }

            // === Reactive Render Context (for RHtml) ===

            RenderStackPush(id_reg) => {
                let id = reg!(id_reg);
                match &id {
                    GcValue::String(s) => {
                        if let Some(str_val) = self.heap.get_string(*s) {
                            let name = str_val.data.clone();
                            self.reactive_context.render_stack.push(name.clone());
                            // Also push to component tree stack
                            self.reactive_context.component_tree_stack.push((name, Vec::new()));
                        } else {
                            return Err(RuntimeError::Panic("Invalid string for render stack push".into()));
                        }
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: id.type_name(&self.heap).to_string(),
                        });
                    }
                }
            }

            RenderStackPop => {
                self.reactive_context.render_stack.pop();
                // Pop from component tree stack and add to parent's children
                if let Some((name, children)) = self.reactive_context.component_tree_stack.pop() {
                    let node = ComponentTreeNode { name, children };
                    // Add to parent's children, or to root if no parent
                    if let Some((_, parent_children)) = self.reactive_context.component_tree_stack.last_mut() {
                        parent_children.push(node);
                    } else {
                        self.reactive_context.root_components.push(node);
                    }
                }
            }

            RenderStackCurrent(dst) => {
                let current = self.reactive_context.render_stack.last()
                    .map(|s| s.clone())
                    .unwrap_or_default();
                let str_ptr = self.heap.alloc_string(current);
                set_reg!(dst, GcValue::String(str_ptr));
            }

            FlushPendingRerenders(dst) => {
                // Move pending rerenders out and clear the list
                let pending = std::mem::take(&mut self.reactive_context.pending_rerenders);
                // Convert to GcValue list of strings
                let items: Vec<GcValue> = pending.into_iter()
                    .map(|s| GcValue::String(self.heap.alloc_string(s)))
                    .collect();
                let list = self.heap.make_list(items);
                set_reg!(dst, GcValue::List(list));
            }

            GetChangedRecordIds(dst) => {
                // Move changed record IDs out and clear the list
                // Essential for action handler flow - mutations outside render context
                let changed = std::mem::take(&mut self.reactive_context.changed_record_ids);
                // Convert to GcValue list of ints
                let items: Vec<GcValue> = changed.into_iter()
                    .map(|id| GcValue::Int64(id as i64))
                    .collect();
                let list = self.heap.make_list(items);
                set_reg!(dst, GcValue::List(list));
            }

            ClearReactiveDependencies => {
                self.reactive_context.dependencies.clear();
                self.reactive_context.pending_rerenders.clear();
                self.reactive_context.changed_record_ids.clear();
            }

            ClearComponentDeps(id_reg) => {
                let id = reg!(id_reg);
                match &id {
                    GcValue::String(s) => {
                        if let Some(str_val) = self.heap.get_string(*s) {
                            let comp_id = &str_val.data;
                            // Remove this component from all dependency lists
                            for deps in self.reactive_context.dependencies.values_mut() {
                                deps.retain(|d| d != comp_id);
                            }
                            // Also remove from pending rerenders
                            self.reactive_context.pending_rerenders.retain(|d| d != comp_id);
                        }
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: id.type_name(&self.heap).to_string(),
                        });
                    }
                }
            }

            ReactiveGetId(dst, record_reg) => {
                let rec_val = reg!(record_reg);
                match &rec_val {
                    GcValue::ReactiveRecord(rec) => {
                        set_reg!(dst, GcValue::Int64(rec.id as i64));
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "ReactiveRecord".to_string(),
                            found: rec_val.type_name(&self.heap).to_string(),
                        });
                    }
                }
            }

            ReactiveSetDeps(deps_reg) => {
                let deps_val = reg!(deps_reg);
                // deps_val should be a Map[Int, List[String]]
                match &deps_val {
                    GcValue::Map(map_ptr) => {
                        if let Some(map) = self.heap.get_map(*map_ptr) {
                            self.reactive_context.dependencies.clear();
                            for (key, value) in map.entries.iter() {
                                if let GcMapKey::Int64(record_id) = key {
                                    if let GcValue::List(list) = value {
                                        let mut comp_ids = Vec::new();
                                        for item in list.iter() {
                                            if let GcValue::String(s) = item {
                                                if let Some(str_val) = self.heap.get_string(*s) {
                                                    comp_ids.push(str_val.data.clone());
                                                }
                                            }
                                        }
                                        self.reactive_context.dependencies.insert(*record_id as u64, comp_ids);
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Map[Int, List[String]]".to_string(),
                            found: deps_val.type_name(&self.heap).to_string(),
                        });
                    }
                }
            }

            ReactiveGetDeps(dst) => {
                // Convert dependencies HashMap to a GcValue Map
                let mut map_entries = ImblHashMap::new();
                for (record_id, comp_ids) in &self.reactive_context.dependencies {
                    let key = GcMapKey::Int64(*record_id as i64);
                    let items: Vec<GcValue> = comp_ids.iter()
                        .map(|s| GcValue::String(self.heap.alloc_string(s.clone())))
                        .collect();
                    let list = self.heap.make_list(items);
                    map_entries.insert(key, GcValue::List(list));
                }
                let map = self.heap.alloc_map(map_entries);
                set_reg!(dst, GcValue::Map(map));
            }

            GetComponentTree(dst) => {
                // Convert component tree to CNode variants
                fn convert_tree(heap: &mut Heap, nodes: &[ComponentTreeNode]) -> GcValue {
                    let items: Vec<GcValue> = nodes.iter().map(|node| {
                        let name = GcValue::String(heap.alloc_string(node.name.clone()));
                        let children = convert_tree(heap, &node.children);
                        // Create CNode(name, children) variant
                        heap.make_variant("stdlib.rhtml.ComponentTree", "CNode", vec![name, children])
                    }).collect();
                    GcValue::List(heap.make_list(items))
                }
                let tree = convert_tree(&mut self.heap, &self.reactive_context.root_components);
                set_reg!(dst, tree);
            }

            RenderContextStart => {
                // Only clear context on outermost RHtml call
                if self.reactive_context.nesting_depth == 0 {
                    self.reactive_context.dependencies.clear();
                    self.reactive_context.pending_rerenders.clear();
                    // Don't clear render_stack - it may have been set by RenderStack.push
                    // for re-rendering components with proper dep tracking
                    self.reactive_context.component_tree_stack.clear();
                    self.reactive_context.root_components.clear();
                    self.reactive_context.renderers.clear();
                }
                self.reactive_context.nesting_depth += 1;
            }

            RenderContextFinish(dst) => {
                // Get deps map
                let mut map_entries = ImblHashMap::new();
                for (record_id, comp_ids) in &self.reactive_context.dependencies {
                    let key = GcMapKey::Int64(*record_id as i64);
                    let items: Vec<GcValue> = comp_ids.iter()
                        .map(|s| GcValue::String(self.heap.alloc_string(s.clone())))
                        .collect();
                    let list = self.heap.make_list(items);
                    map_entries.insert(key, GcValue::List(list));
                }
                let deps_map = self.heap.alloc_map(map_entries);

                // Get component tree
                fn convert_tree(heap: &mut Heap, nodes: &[ComponentTreeNode]) -> GcValue {
                    let items: Vec<GcValue> = nodes.iter().map(|node| {
                        let name = GcValue::String(heap.alloc_string(node.name.clone()));
                        let children = convert_tree(heap, &node.children);
                        heap.make_variant("stdlib.rhtml.ComponentTree", "CNode", vec![name, children])
                    }).collect();
                    GcValue::List(heap.make_list(items))
                }
                let components = convert_tree(&mut self.heap, &self.reactive_context.root_components);

                // Get renderers map
                let mut renderers_entries = ImblHashMap::new();
                for (name, func) in &self.reactive_context.renderers {
                    let key = GcMapKey::String(name.clone());
                    renderers_entries.insert(key, func.clone());
                }
                let renderers_map = self.heap.alloc_map(renderers_entries);

                // Get changed record IDs and clear the list
                let changed = std::mem::take(&mut self.reactive_context.changed_record_ids);
                let changed_items: Vec<GcValue> = changed.into_iter()
                    .map(|id| GcValue::Int64(id as i64))
                    .collect();
                let changed_list = self.heap.make_list(changed_items);

                // Decrement nesting depth
                self.reactive_context.nesting_depth = self.reactive_context.nesting_depth.saturating_sub(1);

                // Return tuple (deps, components, renderers, changedIds)
                let tuple_ptr = self.heap.alloc_tuple(vec![
                    GcValue::Map(deps_map),
                    components,
                    GcValue::Map(renderers_map),
                    GcValue::List(changed_list),
                ]);
                set_reg!(dst, GcValue::Tuple(tuple_ptr));
            }

            RegisterRenderer(name_reg, func_reg) => {
                // Get component name
                let name = match reg!(name_reg) {
                    GcValue::String(s) => self.heap.get_string(s).map(|s| s.data.clone()).unwrap_or_else(|| String::new()),
                    _ => return Err(RuntimeError::Panic("RegisterRenderer: name must be string".into())),
                };
                // Store function in renderers map
                let func = reg!(func_reg).clone();
                self.reactive_context.renderers.insert(name, func);
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
                        GcValue::List(self.heap.make_list(values))
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
                GcValue::List(self.heap.make_list(values))
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
                let headers_list = GcValue::List(self.heap.make_list(header_tuples));

                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(self.heap.make_list(bytes))
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
            IoResponseValue::ServerRequest { request_id, method, path, headers, body, query_params, cookies, form_params, is_websocket } => {
                let header_tuples: Vec<GcValue> = headers
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(self.heap.alloc_string(k));
                        let val = GcValue::String(self.heap.alloc_string(v));
                        GcValue::Tuple(self.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let headers_list = GcValue::List(self.heap.make_list(header_tuples));

                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(self.heap.make_list(bytes))
                    }
                };

                // Convert query params to list of tuples
                let query_param_tuples: Vec<GcValue> = query_params
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(self.heap.alloc_string(k));
                        let val = GcValue::String(self.heap.alloc_string(v));
                        GcValue::Tuple(self.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let query_params_list = GcValue::List(self.heap.make_list(query_param_tuples));

                // Convert cookies to list of tuples
                let cookie_tuples: Vec<GcValue> = cookies
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(self.heap.alloc_string(k));
                        let val = GcValue::String(self.heap.alloc_string(v));
                        GcValue::Tuple(self.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let cookies_list = GcValue::List(self.heap.make_list(cookie_tuples));

                // Convert form params to list of tuples
                let form_param_tuples: Vec<GcValue> = form_params
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(self.heap.alloc_string(k));
                        let val = GcValue::String(self.heap.alloc_string(v));
                        GcValue::Tuple(self.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let form_params_list = GcValue::List(self.heap.make_list(form_param_tuples));

                let method_str = GcValue::String(self.heap.alloc_string(method));
                let path_str = GcValue::String(self.heap.alloc_string(path));

                GcValue::Record(self.heap.alloc_record(
                    "HttpRequest".to_string(),
                    vec!["id".to_string(), "method".to_string(), "path".to_string(), "headers".to_string(), "body".to_string(), "queryParams".to_string(), "cookies".to_string(), "formParams".to_string(), "isWebSocket".to_string()],
                    vec![GcValue::Int64(request_id as i64), method_str, path_str, headers_list, body_value, query_params_list, cookies_list, form_params_list, GcValue::Bool(is_websocket)],
                    vec![false, false, false, false, false, false, false, false, false],
                ))
            }
            IoResponseValue::ExecResult { exit_code, stdout, stderr } => {
                let stdout_value = match std::string::String::from_utf8(stdout.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = stdout.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(self.heap.make_list(bytes))
                    }
                };

                let stderr_value = match std::string::String::from_utf8(stderr.clone()) {
                    Ok(s) => GcValue::String(self.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = stderr.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(self.heap.make_list(bytes))
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
                        // Return each row as a tuple for heterogeneous column types
                        GcValue::Tuple(self.heap.alloc_tuple(col_values))
                    })
                    .collect();
                GcValue::List(self.heap.make_list(row_values))
            }
            IoResponseValue::PgAffected(count) => GcValue::Int64(count as i64),
            IoResponseValue::PgNotification { channel, payload } => {
                // Return as tuple (channel, payload)
                let channel_val = GcValue::String(self.heap.alloc_string(channel));
                let payload_val = GcValue::String(self.heap.alloc_string(payload));
                GcValue::Tuple(self.heap.alloc_tuple(vec![channel_val, payload_val]))
            }
            IoResponseValue::PgNotificationOption(opt) => {
                // Return as Option: Some((channel, payload)) or None
                match opt {
                    Some((channel, payload)) => {
                        let channel_val = GcValue::String(self.heap.alloc_string(channel));
                        let payload_val = GcValue::String(self.heap.alloc_string(payload));
                        let tuple = GcValue::Tuple(self.heap.alloc_tuple(vec![channel_val, payload_val]));
                        // Create Some variant
                        GcValue::Variant(self.heap.alloc_variant(
                            Arc::new("Option".to_string()),
                            Arc::new("Some".to_string()),
                            vec![tuple],
                        ))
                    }
                    None => {
                        // Create None variant
                        GcValue::Variant(self.heap.alloc_variant(
                            Arc::new("Option".to_string()),
                            Arc::new("None".to_string()),
                            vec![],
                        ))
                    }
                }
            }
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
            PgValue::Json(s) => {
                // Return JSON as a string - user can parse with jsonParse if needed
                GcValue::String(self.heap.alloc_string(s))
            }
            PgValue::Vector(floats) => {
                // Return vector as Float32Array (native type for pgvector)
                let ptr = self.heap.alloc_float32_array(floats);
                GcValue::Float32Array(ptr)
            }
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
            GcValue::Float64Array(ptr) => {
                // Convert Float64Array to vector (for pgvector)
                if let Some(arr) = self.heap.get_float64_array(*ptr) {
                    let floats: Vec<f32> = arr.items.iter().map(|f| *f as f32).collect();
                    Ok(PgParam::Vector(floats))
                } else {
                    Err(RuntimeError::IOError("Invalid float64 array pointer".to_string()))
                }
            }
            GcValue::Float32Array(ptr) => {
                // Float32Array is the native type for pgvector
                if let Some(arr) = self.heap.get_float32_array(*ptr) {
                    Ok(PgParam::Vector(arr.items.clone()))
                } else {
                    Err(RuntimeError::IOError("Invalid float32 array pointer".to_string()))
                }
            }
            GcValue::Variant(ptr) => {
                // Check if this is a Json variant (handle both "Json" and "stdlib.json.Json")
                if let Some(var) = self.heap.get_variant(*ptr) {
                    let type_name = var.type_name.as_ref();
                    if type_name == "Json" || type_name.ends_with(".Json") {
                        // Serialize Json variant to JSON string
                        let json_str = self.json_variant_to_string(value)?;
                        Ok(PgParam::Json(json_str))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "Json variant".to_string(),
                            found: format!("{} variant", var.type_name),
                        })
                    }
                } else {
                    Err(RuntimeError::IOError("Invalid variant pointer".to_string()))
                }
            }
            GcValue::List(list) => {
                // Check if this is a list of floats (for pgvector)
                let items = list.items();
                if items.is_empty() {
                    Ok(PgParam::Vector(vec![]))
                } else if matches!(items.first(), Some(GcValue::Float64(_)) | Some(GcValue::Float32(_))) {
                    let floats: Vec<f32> = items.iter().filter_map(|v| match v {
                        GcValue::Float64(f) => Some(*f as f32),
                        GcValue::Float32(f) => Some(*f),
                        _ => None,
                    }).collect();
                    Ok(PgParam::Vector(floats))
                } else {
                    Err(RuntimeError::TypeError {
                        expected: "list of floats for vector".to_string(),
                        found: "list of other types".to_string(),
                    })
                }
            }
            _ => Err(RuntimeError::TypeError {
                expected: "Int, Float, String, Bool, Unit, Float32Array, Float64Array, List[Float], or Json".to_string(),
                found: "unsupported type for Pg param".to_string(),
            }),
        }
    }

    /// Convert a Json variant to a JSON string
    fn json_variant_to_string(&self, value: &GcValue) -> Result<String, RuntimeError> {
        match value {
            GcValue::Variant(ptr) => {
                let var = self.heap.get_variant(*ptr)
                    .ok_or_else(|| RuntimeError::IOError("Invalid variant pointer".to_string()))?;

                match var.constructor.as_str() {
                    "Null" => Ok("null".to_string()),
                    "Bool" => {
                        if let Some(GcValue::Bool(b)) = var.fields.first() {
                            Ok(if *b { "true" } else { "false" }.to_string())
                        } else {
                            Err(RuntimeError::IOError("Invalid Bool variant".to_string()))
                        }
                    }
                    "Number" => {
                        if let Some(GcValue::Float64(f)) = var.fields.first() {
                            Ok(f.to_string())
                        } else {
                            Err(RuntimeError::IOError("Invalid Number variant".to_string()))
                        }
                    }
                    "String" => {
                        if let Some(GcValue::String(ptr)) = var.fields.first() {
                            if let Some(s) = self.heap.get_string(*ptr) {
                                // Escape JSON string
                                let escaped = s.data
                                    .replace('\\', "\\\\")
                                    .replace('"', "\\\"")
                                    .replace('\n', "\\n")
                                    .replace('\r', "\\r")
                                    .replace('\t', "\\t");
                                Ok(format!("\"{}\"", escaped))
                            } else {
                                Err(RuntimeError::IOError("Invalid string pointer".to_string()))
                            }
                        } else {
                            Err(RuntimeError::IOError("Invalid String variant".to_string()))
                        }
                    }
                    "Array" => {
                        if let Some(GcValue::List(list)) = var.fields.first() {
                            let items: Result<Vec<String>, _> = list.items()
                                .iter()
                                .map(|item| self.json_variant_to_string(item))
                                .collect();
                            Ok(format!("[{}]", items?.join(",")))
                        } else {
                            Err(RuntimeError::IOError("Invalid Array variant".to_string()))
                        }
                    }
                    "Object" => {
                        if let Some(GcValue::List(list)) = var.fields.first() {
                            let entries: Result<Vec<String>, _> = list.items()
                                .iter()
                                .map(|item| {
                                    // Each item is a tuple (String, Json)
                                    if let GcValue::Tuple(ptr) = item {
                                        if let Some(tuple) = self.heap.get_tuple(*ptr) {
                                            if tuple.items.len() == 2 {
                                                let key = if let GcValue::String(sptr) = &tuple.items[0] {
                                                    if let Some(s) = self.heap.get_string(*sptr) {
                                                        s.data.clone()
                                                    } else {
                                                        return Err(RuntimeError::IOError("Invalid key".to_string()));
                                                    }
                                                } else {
                                                    return Err(RuntimeError::IOError("Key must be string".to_string()));
                                                };
                                                let val = self.json_variant_to_string(&tuple.items[1])?;
                                                let escaped_key = key
                                                    .replace('\\', "\\\\")
                                                    .replace('"', "\\\"");
                                                return Ok(format!("\"{}\":{}", escaped_key, val));
                                            }
                                        }
                                    }
                                    Err(RuntimeError::IOError("Invalid Object entry".to_string()))
                                })
                                .collect();
                            Ok(format!("{{{}}}", entries?.join(",")))
                        } else {
                            Err(RuntimeError::IOError("Invalid Object variant".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::IOError(format!("Unknown Json constructor: {}", var.constructor))),
                }
            }
            _ => Err(RuntimeError::TypeError {
                expected: "Json variant".to_string(),
                found: "other type".to_string(),
            }),
        }
    }

    /// Run garbage collection if threshold exceeded.
    /// Collects roots from all stack frames and runs mark-sweep.
    fn maybe_gc(&mut self) {
        if !self.heap.should_collect() {
            return;
        }

        // Collect all GC pointers from registers in all frames as roots
        let mut roots = Vec::new();
        for frame in &self.frames {
            for val in &frame.registers {
                roots.extend(val.gc_pointers());
            }
            for val in &frame.captures {
                roots.extend(val.gc_pointers());
            }
        }

        // Also include current exception if any
        if let Some(ref exc) = self.current_exception {
            roots.extend(exc.gc_pointers());
        }

        // Also include exit value if any
        if let Some(ref val) = self.exit_value {
            roots.extend(val.gc_pointers());
        }

        self.heap.set_roots(roots);
        self.heap.collect();
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
                let items: Vec<GcValue> = list.iter().cloned().collect();
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
            GcValue::Float32Array(arr_ptr) => {
                let items = self.heap.get_float32_array(arr_ptr)
                    .map(|a| a.items.clone())
                    .unwrap_or_default();
                let mut json_items = Vec::new();
                for f in items {
                    let ptr = self.heap.alloc_variant(json_type.clone(), Arc::new("Number".to_string()), vec![GcValue::Float64(f as f64)]);
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
            GcValue::Int64List(list) => {
                // Convert to JSON array of numbers
                let mut json_items = Vec::new();
                for n in list.iter() {
                    let ptr = self.heap.alloc_variant(json_type.clone(), Arc::new("Number".to_string()), vec![GcValue::Float64(n as f64)]);
                    json_items.push(GcValue::Variant(ptr));
                }
                let list = GcList::from_vec(json_items);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Array".to_string()), vec![GcValue::List(list)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::Buffer(buf_ptr) => {
                // Convert buffer contents to JSON string
                let s = self.heap.get_buffer(buf_ptr).map(|b| b.to_string()).unwrap_or_default();
                let str_ptr = self.heap.alloc_string(s);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::NativeHandle(handle) => {
                // Native handles don't have a meaningful JSON representation
                let s = format!("<native:type={}>", handle.type_id);
                let str_ptr = self.heap.alloc_string(s);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("String".to_string()), vec![GcValue::String(str_ptr)]);
                Ok(GcValue::Variant(ptr))
            }
            GcValue::ReactiveRecord(rec) => {
                // Convert reactive record to JSON object
                let fields = rec.fields.read().map_err(|_| RuntimeError::Panic("Failed to read reactive record".to_string()))?;
                let mut pairs = Vec::new();
                for (name, value) in rec.field_names.iter().zip(fields.iter()) {
                    let gc_value = self.heap.value_to_gc(value);
                    let json_value = self.value_to_json(gc_value)?;
                    let name_ptr = self.heap.alloc_string(name.clone());
                    let tuple_ptr = self.heap.alloc_tuple(vec![GcValue::String(name_ptr), json_value]);
                    pairs.push(GcValue::Tuple(tuple_ptr));
                }
                let list = GcList::from_vec(pairs);
                let ptr = self.heap.alloc_variant(json_type, Arc::new("Object".to_string()), vec![GcValue::List(list)]);
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
            crate::value::TypeKind::Reactive => "reactive",
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
                        for item in list.iter() {
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
                                    let list_items: Vec<GcValue> = list.iter().cloned().collect();

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
                                    let list_items: Vec<GcValue> = list.iter().cloned().collect();
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

    /// Parse HTTP request params to typed record
    /// Returns Result[T, String] variant - Ok(record) or Err(error_message)
    fn request_to_type(&mut self, type_name: &str, request: GcValue) -> GcValue {
        // Extract params from the HttpRequest record
        let params = match self.extract_request_params(&request) {
            Ok(p) => p,
            Err(e) => return self.make_err_variant(&e),
        };

        // Look up type info
        let type_info = self.shared.types.read().unwrap().get(type_name).cloned()
            .or_else(|| self.shared.dynamic_types.read().unwrap().get(type_name).cloned())
            .or_else(|| self.shared.stdlib_types.read().unwrap().get(type_name).cloned());

        let type_val = match type_info {
            Some(t) => t,
            None => return self.make_err_variant(&format!("Unknown type: {}", type_name)),
        };

        // Must be a record type
        if !type_val.constructors.is_empty() {
            return self.make_err_variant(&format!("requestToType only works with record types, '{}' is a variant", type_name));
        }

        // Parse each field from params
        let mut field_values = Vec::new();
        let mut is_float = Vec::new();
        let mut errors = Vec::new();

        for field in &type_val.fields {
            let field_name = &field.name;
            let field_type = &field.type_name;

            match params.get(field_name) {
                Some(value_str) => {
                    match self.parse_string_to_type(value_str, field_type) {
                        Ok(val) => {
                            is_float.push(field_type == "Float" || field_type == "Float64" || field_type == "Float32");
                            field_values.push(val);
                        }
                        Err(e) => {
                            errors.push(format!("field '{}': {}", field_name, e));
                        }
                    }
                }
                None => {
                    // Check if field type is Option - if so, use None as default
                    if field_type.starts_with("Option") {
                        is_float.push(false);
                        // Create None variant for Option type
                        let var_ptr = self.heap.alloc_variant(
                            Arc::new("Option".to_string()),
                            Arc::new("None".to_string()),
                            vec![],
                        );
                        field_values.push(GcValue::Variant(var_ptr));
                    } else {
                        errors.push(format!("missing required field '{}'", field_name));
                    }
                }
            }
        }

        if !errors.is_empty() {
            return self.make_err_variant(&errors.join(", "));
        }

        // Construct the record
        let rec_ptr = self.heap.alloc_record(
            type_name.to_string(),
            type_val.fields.iter().map(|f| f.name.clone()).collect(),
            field_values,
            is_float,
        );

        // Return Ok(record)
        self.make_ok_variant(GcValue::Record(rec_ptr))
    }

    /// Extract params from HttpRequest record as HashMap
    fn extract_request_params(&self, request: &GcValue) -> Result<std::collections::HashMap<String, String>, String> {
        use std::collections::HashMap;

        let record = match request {
            GcValue::Record(ptr) => {
                self.heap.get_record(*ptr)
                    .ok_or_else(|| "Invalid request record".to_string())?
            }
            _ => return Err("Expected HttpRequest record".to_string()),
        };

        let mut params = HashMap::new();

        // Find queryParams field (index 5 in HttpRequest)
        // HttpRequest fields: id, method, path, headers, body, queryParams, cookies, formParams, isWebSocket
        let query_params_idx = record.field_names.iter().position(|n| n == "queryParams");
        let form_params_idx = record.field_names.iter().position(|n| n == "formParams");

        // Extract query params
        if let Some(idx) = query_params_idx {
            if let Some(GcValue::List(list)) = record.fields.get(idx) {
                for item in list.iter() {
                    if let GcValue::Tuple(tuple_ptr) = item {
                        if let Some(tuple) = self.heap.get_tuple(*tuple_ptr) {
                            if tuple.items.len() >= 2 {
                                let key = match &tuple.items[0] {
                                    GcValue::String(ptr) => self.heap.get_string(*ptr)
                                        .map(|s| s.data.clone())
                                        .unwrap_or_default(),
                                    _ => continue,
                                };
                                let value = match &tuple.items[1] {
                                    GcValue::String(ptr) => self.heap.get_string(*ptr)
                                        .map(|s| s.data.clone())
                                        .unwrap_or_default(),
                                    _ => continue,
                                };
                                params.insert(key, value);
                            }
                        }
                    }
                }
            }
        }

        // Extract form params (override query params if same key)
        if let Some(idx) = form_params_idx {
            if let Some(GcValue::List(list)) = record.fields.get(idx) {
                for item in list.iter() {
                    if let GcValue::Tuple(tuple_ptr) = item {
                        if let Some(tuple) = self.heap.get_tuple(*tuple_ptr) {
                            if tuple.items.len() >= 2 {
                                let key = match &tuple.items[0] {
                                    GcValue::String(ptr) => self.heap.get_string(*ptr)
                                        .map(|s| s.data.clone())
                                        .unwrap_or_default(),
                                    _ => continue,
                                };
                                let value = match &tuple.items[1] {
                                    GcValue::String(ptr) => self.heap.get_string(*ptr)
                                        .map(|s| s.data.clone())
                                        .unwrap_or_default(),
                                    _ => continue,
                                };
                                params.insert(key, value);
                            }
                        }
                    }
                }
            }
        }

        Ok(params)
    }

    /// Parse a string value to the expected type
    fn parse_string_to_type(&mut self, value: &str, type_name: &str) -> Result<GcValue, String> {
        match type_name {
            "String" => {
                Ok(GcValue::String(self.heap.alloc_string(value.to_string())))
            }
            "Int" | "Int64" => {
                value.parse::<i64>()
                    .map(GcValue::Int64)
                    .map_err(|_| format!("expected Int, got '{}'", value))
            }
            "Int32" => {
                value.parse::<i32>()
                    .map(|n| GcValue::Int64(n as i64))
                    .map_err(|_| format!("expected Int32, got '{}'", value))
            }
            "Int16" => {
                value.parse::<i16>()
                    .map(|n| GcValue::Int64(n as i64))
                    .map_err(|_| format!("expected Int16, got '{}'", value))
            }
            "Int8" => {
                value.parse::<i8>()
                    .map(|n| GcValue::Int64(n as i64))
                    .map_err(|_| format!("expected Int8, got '{}'", value))
            }
            "Float" | "Float64" => {
                value.parse::<f64>()
                    .map(GcValue::Float64)
                    .map_err(|_| format!("expected Float, got '{}'", value))
            }
            "Float32" => {
                value.parse::<f32>()
                    .map(|n| GcValue::Float64(n as f64))
                    .map_err(|_| format!("expected Float32, got '{}'", value))
            }
            "Bool" => {
                match value.to_lowercase().as_str() {
                    "true" | "1" | "yes" => Ok(GcValue::Bool(true)),
                    "false" | "0" | "no" => Ok(GcValue::Bool(false)),
                    _ => Err(format!("expected Bool, got '{}'", value)),
                }
            }
            _ if type_name.starts_with("Option[") => {
                // Option[T] - parse inner type and wrap in Some
                let inner_type = &type_name[7..type_name.len()-1]; // Extract T from Option[T]
                match self.parse_string_to_type(value, inner_type) {
                    Ok(inner_val) => {
                        let var_ptr = self.heap.alloc_variant(
                            Arc::new("Option".to_string()),
                            Arc::new("Some".to_string()),
                            vec![inner_val],
                        );
                        Ok(GcValue::Variant(var_ptr))
                    }
                    Err(e) => Err(e),
                }
            }
            _ => {
                Err(format!("unsupported type '{}' for request parsing", type_name))
            }
        }
    }

    /// Create Ok(value) variant for Result type
    fn make_ok_variant(&mut self, value: GcValue) -> GcValue {
        let var_ptr = self.heap.alloc_variant(
            Arc::new("Result".to_string()),
            Arc::new("Ok".to_string()),
            vec![value],
        );
        GcValue::Variant(var_ptr)
    }

    /// Create Err(message) variant for Result type
    fn make_err_variant(&mut self, message: &str) -> GcValue {
        let msg_ptr = self.heap.alloc_string(message.to_string());
        let var_ptr = self.heap.alloc_variant(
            Arc::new("Result".to_string()),
            Arc::new("Err".to_string()),
            vec![GcValue::String(msg_ptr)],
        );
        GcValue::Variant(var_ptr)
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
        let spawn_runtime_handle = io_runtime.runtime_handle();

        let shared = Arc::new(AsyncSharedState {
            functions: RwLock::new(HashMap::new()),
            function_list: RwLock::new(Vec::new()),
            natives: HashMap::new(),
            natives_vec: Vec::new(),
            native_name_to_idx: HashMap::new(),
            types: RwLock::new(HashMap::new()),
            jit_int_functions: RwLock::new(HashMap::new()),
            jit_int_functions_0: RwLock::new(HashMap::new()),
            jit_int_functions_2: RwLock::new(HashMap::new()),
            jit_int_functions_3: RwLock::new(HashMap::new()),
            jit_int_functions_4: RwLock::new(HashMap::new()),
            jit_loop_array_functions: RwLock::new(HashMap::new()),
            jit_array_fill_functions: RwLock::new(HashMap::new()),
            jit_array_sum_functions: RwLock::new(HashMap::new()),
            jit_list_sum_functions: RwLock::new(HashMap::new()),
            jit_list_sum_tr_functions: RwLock::new(HashMap::new()),
            shutdown: AtomicBool::new(false),
            interrupt: AtomicBool::new(false),
            interactive_mode: AtomicBool::new(false),
            spawn_runtime_handle: Some(spawn_runtime_handle),
            process_registry: TokioRwLock::new(HashMap::new()),
            process_abort_handles: TokioRwLock::new(HashMap::new()),
            process_servers: TokioRwLock::new(HashMap::new()),
            spawned_count: AtomicU64::new(0),
            exited_count: AtomicU64::new(0),
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
            extensions: RwLock::new(None),
        });

        Self {
            shared,
            io_runtime: Some(io_runtime),
            config,
        }
    }

    /// Set interactive mode (true in REPL/TUI, false in script mode).
    pub fn set_interactive_mode(&self, interactive: bool) {
        self.shared.interactive_mode.store(interactive, Ordering::SeqCst);
    }

    /// Check if running in interactive mode.
    pub fn is_interactive(&self) -> bool {
        self.shared.interactive_mode.load(Ordering::SeqCst)
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
    /// Also adds to indexed structures for CallNativeIdx optimization.
    pub fn register_native(&mut self, name: &str, native: Arc<GcNativeFn>) {
        let shared = Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started");
        let idx = shared.natives_vec.len() as u16;
        shared.natives_vec.push(native.clone());
        shared.native_name_to_idx.insert(name.to_string(), idx);
        shared.natives.insert(name.to_string(), native);
    }

    /// Get all native function indices for the compiler.
    pub fn get_native_indices(&self) -> HashMap<String, u16> {
        self.shared.native_name_to_idx.clone()
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

    /// Register a JIT list sum function - safe during concurrent evals.
    pub fn register_jit_list_sum_function(&mut self, func_index: u16, jit_fn: crate::shared_types::JitListSumFn) {
        self.shared.jit_list_sum_functions.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Register a JIT tail-recursive list sum function - safe during concurrent evals.
    pub fn register_jit_list_sum_tr_function(&mut self, func_index: u16, jit_fn: crate::shared_types::JitListSumTrFn) {
        self.shared.jit_list_sum_tr_functions.write().unwrap()
            .insert(func_index, jit_fn);
    }

    /// Set the extension manager for native library functions.
    pub fn set_extension_manager(&self, manager: Arc<ExtensionManager>) {
        *self.shared.extensions.write().unwrap() = Some(manager);
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

        let shared = Arc::get_mut(&mut self.shared)
            .expect("Cannot setup output after execution started");
        shared.output_sender = Some(sender);

        // Log AFTER the sender is stored
        let shared_ptr = &*shared as *const _ as usize;
        if let Some(ref s) = shared.output_sender {
            let sender_ptr = s as *const _ as usize;
            AsyncProcess::debug_log(&format!("[setup_output] shared_ptr={:#x}, sender_ptr={:#x}", shared_ptr, sender_ptr));
        }

        receiver
    }

    /// Test the output channel by sending a message through the sender.
    pub fn test_output_channel(&self, msg: &str) {
        if let Some(ref sender) = self.shared.output_sender {
            let sender_ptr = sender as *const _ as usize;
            AsyncProcess::debug_log(&format!("[test_output_channel] sender_ptr={:#x}, sending: {}", sender_ptr, msg));
            let _ = sender.send(msg.to_string());
        } else {
            AsyncProcess::debug_log("[test_output_channel] NO sender!");
        }
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
                use std::io::Write;
                let s = heap.display_value(&args[0]);
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                let _ = write!(handle, "{}", s);
                let _ = handle.flush();
                Ok(GcValue::Unit)
            }),
        }));

        // println - print with newline
        self.register_native("println", Arc::new(GcNativeFn {
            name: "println".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::io::Write;
                let s = heap.display_value(&args[0]);
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                let _ = writeln!(handle, "{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // eprint - print to stderr without newline
        self.register_native("eprint", Arc::new(GcNativeFn {
            name: "eprint".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::io::Write;
                let s = heap.display_value(&args[0]);
                eprint!("{}", s);
                let _ = std::io::stderr().flush();
                Ok(GcValue::Unit)
            }),
        }));

        // eprintln - print to stderr with newline
        self.register_native("eprintln", Arc::new(GcNativeFn {
            name: "eprintln".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                use std::io::Write;
                let s = heap.display_value(&args[0]);
                eprintln!("{}", s);
                let _ = std::io::stderr().flush();
                Ok(GcValue::Unit)
            }),
        }));

        // flushStdout - flush stdout buffer
        self.register_native("flushStdout", Arc::new(GcNativeFn {
            name: "flushStdout".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use std::io::Write;
                let _ = std::io::stdout().flush();
                Ok(GcValue::Unit)
            }),
        }));

        // flushStderr - flush stderr buffer
        self.register_native("flushStderr", Arc::new(GcNativeFn {
            name: "flushStderr".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                use std::io::Write;
                let _ = std::io::stderr().flush();
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
            func: Box::new(|args, _heap| {
                let start = match &args[0] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                let end = match &args[1] {
                    GcValue::Int64(n) => *n,
                    _ => return Err(RuntimeError::TypeError { expected: "Int".to_string(), found: "other".to_string() })
                };
                // Use specialized Int64List for better performance
                let items: Vec<i64> = (start..end).collect();
                Ok(GcValue::Int64List(GcInt64List::from_vec(items)))
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

        self.register_native("Env.isInteractive", Arc::new(GcNativeFn {
            name: "Env.isInteractive".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                // Check if NOSTOS_INTERACTIVE env var is set (set by TUI on startup)
                let is_interactive = std::env::var("NOSTOS_INTERACTIVE").is_ok();
                Ok(GcValue::Bool(is_interactive))
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

        // Map.union(map1, map2) -> new map with all entries from both (map2 wins on conflict)
        self.register_native("Map.union", Arc::new(GcNativeFn {
            name: "Map.union".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                match (&args[0], &args[1]) {
                    (GcValue::Map(ptr1), GcValue::Map(ptr2)) => {
                        let map1 = heap.get_map(*ptr1).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let map2 = heap.get_map(*ptr2).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let new_entries = map1.entries.clone().union(map2.entries.clone());
                        let new_ptr = heap.alloc_map(new_entries);
                        Ok(GcValue::Map(new_ptr))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.intersection(map1, map2) -> new map with only keys in both (values from map1)
        self.register_native("Map.intersection", Arc::new(GcNativeFn {
            name: "Map.intersection".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                match (&args[0], &args[1]) {
                    (GcValue::Map(ptr1), GcValue::Map(ptr2)) => {
                        let map1 = heap.get_map(*ptr1).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let map2 = heap.get_map(*ptr2).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let new_entries = map1.entries.clone().intersection(map2.entries.clone());
                        let new_ptr = heap.alloc_map(new_entries);
                        Ok(GcValue::Map(new_ptr))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.difference(map1, map2) -> new map with keys in map1 but not in map2
        self.register_native("Map.difference", Arc::new(GcNativeFn {
            name: "Map.difference".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                match (&args[0], &args[1]) {
                    (GcValue::Map(ptr1), GcValue::Map(ptr2)) => {
                        let map1 = heap.get_map(*ptr1).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let map2 = heap.get_map(*ptr2).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let new_entries = map1.entries.clone().relative_complement(map2.entries.clone());
                        let new_ptr = heap.alloc_map(new_entries);
                        Ok(GcValue::Map(new_ptr))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.toList(map) -> [(key, value), ...]
        self.register_native("Map.toList", Arc::new(GcNativeFn {
            name: "Map.toList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Map(ptr) => {
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let entries_cloned: Vec<_> = map.entries.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        let pairs: Vec<GcValue> = entries_cloned.into_iter().map(|(k, v)| {
                            let key = k.to_gc_value(heap);
                            let tuple_ptr = heap.alloc_tuple(vec![key, v]);
                            GcValue::Tuple(tuple_ptr)
                        }).collect();
                        Ok(GcValue::List(heap.make_list(pairs)))
                    }
                    GcValue::SharedMap(shared_map) => {
                        let entries_cloned: Vec<_> = shared_map.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        let pairs: Vec<GcValue> = entries_cloned.into_iter().map(|(k, v)| {
                            let key = crate::gc::GcMapKey::from_shared_key(&k).to_gc_value(heap);
                            let value = heap.shared_to_gc_value(&v);
                            let tuple_ptr = heap.alloc_tuple(vec![key, value]);
                            GcValue::Tuple(tuple_ptr)
                        }).collect();
                        Ok(GcValue::List(heap.make_list(pairs)))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "Map".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // Map.fromList([(key, value), ...]) -> Map
        self.register_native("Map.fromList", Arc::new(GcNativeFn {
            name: "Map.fromList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::List(list) => {
                        let mut entries = imbl::HashMap::new();
                        for item in list.items().iter() {
                            match item {
                                GcValue::Tuple(tuple_ptr) => {
                                    let tuple = heap.get_tuple(*tuple_ptr).ok_or_else(|| RuntimeError::Panic("Invalid tuple pointer".to_string()))?;
                                    if tuple.items.len() != 2 {
                                        return Err(RuntimeError::TypeError {
                                            expected: "(key, value) tuple".to_string(),
                                            found: format!("tuple of {} elements", tuple.items.len())
                                        });
                                    }
                                    let key = tuple.items[0].to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                                        expected: "hashable".to_string(),
                                        found: tuple.items[0].type_name(heap).to_string()
                                    })?;
                                    entries.insert(key, tuple.items[1].clone());
                                }
                                _ => return Err(RuntimeError::TypeError {
                                    expected: "(key, value) tuple".to_string(),
                                    found: item.type_name(heap).to_string()
                                })
                            }
                        }
                        let new_ptr = heap.alloc_map(entries);
                        Ok(GcValue::Map(new_ptr))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "List".to_string(), found: args[0].type_name(heap).to_string() })
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

        // Set.symmetricDifference(set1, set2) -> elements in either but not both
        self.register_native("Set.symmetricDifference", Arc::new(GcNativeFn {
            name: "Set.symmetricDifference".to_string(),
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
                let new_items = set1.items.clone().symmetric_difference(set2.items.clone());
                let new_ptr = heap.alloc_set(new_items);
                Ok(GcValue::Set(new_ptr))
            }),
        }));

        // Set.isSubset(set1, set2) -> true if set1 is a subset of set2
        self.register_native("Set.isSubset", Arc::new(GcNativeFn {
            name: "Set.isSubset".to_string(),
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
                Ok(GcValue::Bool(set1.items.is_subset(&set2.items)))
            }),
        }));

        // Set.isProperSubset(set1, set2) -> true if set1 is a proper subset of set2
        self.register_native("Set.isProperSubset", Arc::new(GcNativeFn {
            name: "Set.isProperSubset".to_string(),
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
                Ok(GcValue::Bool(set1.items.is_proper_subset(&set2.items)))
            }),
        }));

        // Set.fromList(list) -> Set
        self.register_native("Set.fromList", Arc::new(GcNativeFn {
            name: "Set.fromList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::List(list) => {
                        let mut items = imbl::HashSet::new();
                        for elem in list.items().iter() {
                            let key = elem.to_gc_map_key(heap).ok_or_else(|| RuntimeError::TypeError {
                                expected: "hashable".to_string(),
                                found: elem.type_name(heap).to_string()
                            })?;
                            items.insert(key);
                        }
                        let new_ptr = heap.alloc_set(items);
                        Ok(GcValue::Set(new_ptr))
                    }
                    _ => Err(RuntimeError::TypeError { expected: "List".to_string(), found: args[0].type_name(heap).to_string() })
                }
            }),
        }));

        // === Float64Array Functions ===

        // Float64Array.fromList(list) -> Float64Array
        self.register_native("Float64Array.fromList", Arc::new(GcNativeFn {
            name: "Float64Array.fromList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let floats: Vec<f64> = match &args[0] {
                    GcValue::List(list) => {
                        list.items().iter().map(|v| match v {
                            GcValue::Float64(f) => Ok(*f),
                            GcValue::Float32(f) => Ok(*f as f64),
                            GcValue::Int64(i) => Ok(*i as f64),
                            GcValue::Int32(i) => Ok(*i as f64),
                            _ => Err(RuntimeError::TypeError {
                                expected: "Float or Int".to_string(),
                                found: v.type_name(heap).to_string(),
                            }),
                        }).collect::<Result<Vec<_>, _>>()?
                    }
                    GcValue::Int64List(int_list) => {
                        int_list.iter().map(|i| i as f64).collect()
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let ptr = heap.alloc_float64_array(floats);
                Ok(GcValue::Float64Array(ptr))
            }),
        }));

        // Float64Array.length(arr) -> Int
        self.register_native("Float64Array.length", Arc::new(GcNativeFn {
            name: "Float64Array.length".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Float64Array(ptr) => {
                        let arr = heap.get_float64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid Float64Array pointer".to_string()))?;
                        Ok(GcValue::Int64(arr.items.len() as i64))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Float64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // Float64Array.get(arr, index) -> Float
        self.register_native("Float64Array.get", Arc::new(GcNativeFn {
            name: "Float64Array.get".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let ptr = match &args[0] {
                    GcValue::Float64Array(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let index = match &args[1] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let arr = heap.get_float64_array(ptr)
                    .ok_or_else(|| RuntimeError::Panic("Invalid Float64Array pointer".to_string()))?;
                if index >= arr.items.len() {
                    return Err(RuntimeError::Panic(format!("Index {} out of bounds for Float64Array of length {}", index, arr.items.len())));
                }
                Ok(GcValue::Float64(arr.items[index]))
            }),
        }));

        // Float64Array.set(arr, index, value) -> Float64Array (new array with value at index)
        self.register_native("Float64Array.set", Arc::new(GcNativeFn {
            name: "Float64Array.set".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let ptr = match &args[0] {
                    GcValue::Float64Array(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let index = match &args[1] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let value = match &args[2] {
                    GcValue::Float64(f) => *f,
                    GcValue::Float32(f) => *f as f64,
                    GcValue::Int64(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float or Int".to_string(),
                        found: args[2].type_name(heap).to_string(),
                    }),
                };
                let arr = heap.get_float64_array(ptr)
                    .ok_or_else(|| RuntimeError::Panic("Invalid Float64Array pointer".to_string()))?;
                if index >= arr.items.len() {
                    return Err(RuntimeError::Panic(format!("Index {} out of bounds for Float64Array of length {}", index, arr.items.len())));
                }
                let mut new_items = arr.items.clone();
                new_items[index] = value;
                let new_ptr = heap.alloc_float64_array(new_items);
                Ok(GcValue::Float64Array(new_ptr))
            }),
        }));

        // Float64Array.toList(arr) -> List[Float]
        self.register_native("Float64Array.toList", Arc::new(GcNativeFn {
            name: "Float64Array.toList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Float64Array(ptr) => {
                        let arr = heap.get_float64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid Float64Array pointer".to_string()))?;
                        let values: Vec<GcValue> = arr.items.iter().map(|f| GcValue::Float64(*f)).collect();
                        Ok(GcValue::List(GcList::from_vec(values)))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Float64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // Float64Array.make(size, initial_value) -> Float64Array
        self.register_native("Float64Array.make", Arc::new(GcNativeFn {
            name: "Float64Array.make".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let size = match &args[0] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let value = match &args[1] {
                    GcValue::Float64(f) => *f,
                    GcValue::Float32(f) => *f as f64,
                    GcValue::Int64(i) => *i as f64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float or Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let items = vec![value; size];
                let ptr = heap.alloc_float64_array(items);
                Ok(GcValue::Float64Array(ptr))
            }),
        }));

        // === Int64Array Functions ===

        // Int64Array.fromList(list) -> Int64Array
        self.register_native("Int64Array.fromList", Arc::new(GcNativeFn {
            name: "Int64Array.fromList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let ints: Vec<i64> = match &args[0] {
                    GcValue::List(list) => {
                        list.items().iter().map(|v| match v {
                            GcValue::Int64(i) => Ok(*i),
                            GcValue::Int32(i) => Ok(*i as i64),
                            GcValue::Float64(f) => Ok(*f as i64),
                            _ => Err(RuntimeError::TypeError {
                                expected: "Int or Float".to_string(),
                                found: v.type_name(heap).to_string(),
                            }),
                        }).collect::<Result<Vec<_>, _>>()?
                    }
                    GcValue::Float64Array(ptr) => {
                        let arr = heap.get_float64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid Float64Array pointer".to_string()))?;
                        arr.items.iter().map(|f| *f as i64).collect()
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                Ok(GcValue::Int64List(GcInt64List::from_vec(ints)))
            }),
        }));

        // Int64Array.length(arr) -> Int
        self.register_native("Int64Array.length", Arc::new(GcNativeFn {
            name: "Int64Array.length".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Int64List(list) => {
                        Ok(GcValue::Int64(list.len() as i64))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Int64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // sumInt64Array(arr) -> Int - Fast native SIMD-optimized sum
        self.register_native("sumInt64Array", Arc::new(GcNativeFn {
            name: "sumInt64Array".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Int64Array(ptr) => {
                        let arr = heap.get_int64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid Int64Array pointer".to_string()))?;
                        // Native sum - LLVM will auto-vectorize this
                        let sum: i64 = arr.items.iter().sum();
                        Ok(GcValue::Int64(sum))
                    }
                    GcValue::Int64List(list) => {
                        // Also support Int64List
                        let sum: i64 = list.sum();
                        Ok(GcValue::Int64(sum))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Int64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // Int64Array.get(arr, index) -> Int
        self.register_native("Int64Array.get", Arc::new(GcNativeFn {
            name: "Int64Array.get".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let list = match &args[0] {
                    GcValue::Int64List(list) => list.clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let index = match &args[1] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                if index >= list.len() {
                    return Err(RuntimeError::Panic(format!("Index {} out of bounds for Int64Array of length {}", index, list.len())));
                }
                // Collect all values, then index
                let items: Vec<i64> = list.iter().collect();
                Ok(GcValue::Int64(items[index]))
            }),
        }));

        // Int64Array.set(arr, index, value) -> Int64Array (new array with value at index)
        self.register_native("Int64Array.set", Arc::new(GcNativeFn {
            name: "Int64Array.set".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let list = match &args[0] {
                    GcValue::Int64List(list) => list.clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let index = match &args[1] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let value = match &args[2] {
                    GcValue::Int64(i) => *i,
                    GcValue::Int32(i) => *i as i64,
                    GcValue::Float64(f) => *f as i64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[2].type_name(heap).to_string(),
                    }),
                };
                if index >= list.len() {
                    return Err(RuntimeError::Panic(format!("Index {} out of bounds for Int64Array of length {}", index, list.len())));
                }
                // Collect to vec, modify, then create new list
                let mut new_items: Vec<i64> = list.iter().collect();
                new_items[index] = value;
                Ok(GcValue::Int64List(GcInt64List::from_vec(new_items)))
            }),
        }));

        // Int64Array.toList(arr) -> List[Int]
        self.register_native("Int64Array.toList", Arc::new(GcNativeFn {
            name: "Int64Array.toList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Int64List(list) => {
                        let values: Vec<GcValue> = list.iter().map(|i| GcValue::Int64(i)).collect();
                        Ok(GcValue::List(GcList::from_vec(values)))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Int64Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // Int64Array.make(size, initial_value) -> Int64Array
        self.register_native("Int64Array.make", Arc::new(GcNativeFn {
            name: "Int64Array.make".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let size = match &args[0] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let value = match &args[1] {
                    GcValue::Int64(i) => *i,
                    GcValue::Int32(i) => *i as i64,
                    GcValue::Float64(f) => *f as i64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let items = vec![value; size];
                Ok(GcValue::Int64List(GcInt64List::from_vec(items)))
            }),
        }));

        // === Float32Array Functions (for vectors/pgvector) ===

        // newFloat32Array(size) -> Float32Array
        self.register_native("newFloat32Array", Arc::new(GcNativeFn {
            name: "newFloat32Array".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let size = match &args[0] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let items = vec![0.0f32; size];
                let ptr = heap.alloc_float32_array(items);
                Ok(GcValue::Float32Array(ptr))
            }),
        }));

        // Float32Array.fromList(list) -> Float32Array
        self.register_native("Float32Array.fromList", Arc::new(GcNativeFn {
            name: "Float32Array.fromList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let floats: Vec<f32> = match &args[0] {
                    GcValue::List(list) => {
                        list.items().iter().map(|v| match v {
                            GcValue::Float32(f) => Ok(*f),
                            GcValue::Float64(f) => Ok(*f as f32),
                            GcValue::Int64(i) => Ok(*i as f32),
                            GcValue::Int32(i) => Ok(*i as f32),
                            _ => Err(RuntimeError::TypeError {
                                expected: "Float or Int".to_string(),
                                found: v.type_name(heap).to_string(),
                            }),
                        }).collect::<Result<Vec<_>, _>>()?
                    }
                    GcValue::Int64List(int_list) => {
                        int_list.iter().map(|i| i as f32).collect()
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let ptr = heap.alloc_float32_array(floats);
                Ok(GcValue::Float32Array(ptr))
            }),
        }));

        // Float32Array.length(arr) -> Int
        self.register_native("Float32Array.length", Arc::new(GcNativeFn {
            name: "Float32Array.length".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Float32Array(ptr) => {
                        let arr = heap.get_float32_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid Float32Array pointer".to_string()))?;
                        Ok(GcValue::Int64(arr.items.len() as i64))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Float32Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // Float32Array.get(arr, index) -> Float
        self.register_native("Float32Array.get", Arc::new(GcNativeFn {
            name: "Float32Array.get".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let ptr = match &args[0] {
                    GcValue::Float32Array(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float32Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let index = match &args[1] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let arr = heap.get_float32_array(ptr)
                    .ok_or_else(|| RuntimeError::Panic("Invalid Float32Array pointer".to_string()))?;
                if index >= arr.items.len() {
                    return Err(RuntimeError::Panic(format!("Index {} out of bounds for Float32Array of length {}", index, arr.items.len())));
                }
                Ok(GcValue::Float32(arr.items[index]))
            }),
        }));

        // Float32Array.set(arr, index, value) -> Float32Array (new array with value at index)
        self.register_native("Float32Array.set", Arc::new(GcNativeFn {
            name: "Float32Array.set".to_string(),
            arity: 3,
            func: Box::new(|args, heap| {
                let ptr = match &args[0] {
                    GcValue::Float32Array(ptr) => *ptr,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float32Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let index = match &args[1] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let value = match &args[2] {
                    GcValue::Float32(f) => *f,
                    GcValue::Float64(f) => *f as f32,
                    GcValue::Int64(i) => *i as f32,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float or Int".to_string(),
                        found: args[2].type_name(heap).to_string(),
                    }),
                };
                let arr = heap.get_float32_array(ptr)
                    .ok_or_else(|| RuntimeError::Panic("Invalid Float32Array pointer".to_string()))?;
                if index >= arr.items.len() {
                    return Err(RuntimeError::Panic(format!("Index {} out of bounds for Float32Array of length {}", index, arr.items.len())));
                }
                let mut new_items = arr.items.clone();
                new_items[index] = value;
                let new_ptr = heap.alloc_float32_array(new_items);
                Ok(GcValue::Float32Array(new_ptr))
            }),
        }));

        // Float32Array.toList(arr) -> List[Float]
        self.register_native("Float32Array.toList", Arc::new(GcNativeFn {
            name: "Float32Array.toList".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::Float32Array(ptr) => {
                        let arr = heap.get_float32_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid Float32Array pointer".to_string()))?;
                        let values: Vec<GcValue> = arr.items.iter().map(|f| GcValue::Float32(*f)).collect();
                        Ok(GcValue::List(GcList::from_vec(values)))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Float32Array".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                }
            }),
        }));

        // Float32Array.make(size, initial_value) -> Float32Array
        self.register_native("Float32Array.make", Arc::new(GcNativeFn {
            name: "Float32Array.make".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let size = match &args[0] {
                    GcValue::Int64(i) => *i as usize,
                    GcValue::Int32(i) => *i as usize,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: args[0].type_name(heap).to_string(),
                    }),
                };
                let value = match &args[1] {
                    GcValue::Float32(f) => *f,
                    GcValue::Float64(f) => *f as f32,
                    GcValue::Int64(i) => *i as f32,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float or Int".to_string(),
                        found: args[1].type_name(heap).to_string(),
                    }),
                };
                let items = vec![value; size];
                let ptr = heap.alloc_float32_array(items);
                Ok(GcValue::Float32Array(ptr))
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

        // Runtime.threadCount() -> Int - number of available CPU threads
        self.register_native("Runtime.threadCount", Arc::new(GcNativeFn {
            name: "Runtime.threadCount".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                let count = std::thread::available_parallelism()
                    .map(|n| n.get() as i64)
                    .unwrap_or(1);
                Ok(GcValue::Int64(count))
            }),
        }));

        // Runtime.uptimeMs() -> Int - milliseconds since program start
        use std::sync::OnceLock;
        static START_TIME: OnceLock<std::time::Instant> = OnceLock::new();
        START_TIME.get_or_init(std::time::Instant::now);

        self.register_native("Runtime.uptimeMs", Arc::new(GcNativeFn {
            name: "Runtime.uptimeMs".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                let uptime = START_TIME.get()
                    .map(|start| start.elapsed().as_millis() as i64)
                    .unwrap_or(0);
                Ok(GcValue::Int64(uptime))
            }),
        }));

        // Runtime.memoryKb() -> Int - current process memory usage in KB (Linux only, returns 0 on other platforms)
        self.register_native("Runtime.memoryKb", Arc::new(GcNativeFn {
            name: "Runtime.memoryKb".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                #[cfg(target_os = "linux")]
                {
                    // Read from /proc/self/statm - second field is RSS in pages
                    if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
                        let parts: Vec<&str> = content.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(pages) = parts[1].parse::<i64>() {
                                // Convert pages to KB (assuming 4KB pages)
                                return Ok(GcValue::Int64(pages * 4));
                            }
                        }
                    }
                    Ok(GcValue::Int64(0))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Ok(GcValue::Int64(0))
                }
            }),
        }));

        // Runtime.pid() -> Int - current process ID
        self.register_native("Runtime.pid", Arc::new(GcNativeFn {
            name: "Runtime.pid".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                Ok(GcValue::Int64(std::process::id() as i64))
            }),
        }));

        // Runtime.loadAvg() -> (Float, Float, Float) - 1, 5, 15 minute load averages (Linux only)
        self.register_native("Runtime.loadAvg", Arc::new(GcNativeFn {
            name: "Runtime.loadAvg".to_string(),
            arity: 0,
            func: Box::new(|_args, heap| {
                #[cfg(target_os = "linux")]
                {
                    if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
                        let parts: Vec<&str> = content.split_whitespace().collect();
                        if parts.len() >= 3 {
                            let load1 = parts[0].parse::<f64>().unwrap_or(0.0);
                            let load5 = parts[1].parse::<f64>().unwrap_or(0.0);
                            let load15 = parts[2].parse::<f64>().unwrap_or(0.0);
                            let tuple = heap.alloc_tuple(vec![
                                GcValue::Float64(load1),
                                GcValue::Float64(load5),
                                GcValue::Float64(load15),
                            ]);
                            return Ok(GcValue::Tuple(tuple));
                        }
                    }
                    let tuple = heap.alloc_tuple(vec![
                        GcValue::Float64(0.0),
                        GcValue::Float64(0.0),
                        GcValue::Float64(0.0),
                    ]);
                    Ok(GcValue::Tuple(tuple))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let tuple = heap.alloc_tuple(vec![
                        GcValue::Float64(0.0),
                        GcValue::Float64(0.0),
                        GcValue::Float64(0.0),
                    ]);
                    Ok(GcValue::Tuple(tuple))
                }
            }),
        }));

        // Runtime.numThreads() -> Int - number of OS threads in process (Linux only)
        self.register_native("Runtime.numThreads", Arc::new(GcNativeFn {
            name: "Runtime.numThreads".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                #[cfg(target_os = "linux")]
                {
                    // Read from /proc/self/status - Threads: N
                    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
                        for line in content.lines() {
                            if line.starts_with("Threads:") {
                                if let Some(count_str) = line.split_whitespace().nth(1) {
                                    if let Ok(count) = count_str.parse::<i64>() {
                                        return Ok(GcValue::Int64(count));
                                    }
                                }
                            }
                        }
                    }
                    Ok(GcValue::Int64(0))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Ok(GcValue::Int64(0))
                }
            }),
        }));

        // Runtime.tokioWorkers() -> Int - number of tokio worker threads (Linux only)
        self.register_native("Runtime.tokioWorkers", Arc::new(GcNativeFn {
            name: "Runtime.tokioWorkers".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                #[cfg(target_os = "linux")]
                {
                    let mut count = 0i64;
                    // Read thread names from /proc/self/task/*/comm
                    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
                        for entry in entries.flatten() {
                            let comm_path = entry.path().join("comm");
                            if let Ok(name) = std::fs::read_to_string(&comm_path) {
                                let name = name.trim();
                                if name.starts_with("tokio-runtime-w") {
                                    count += 1;
                                }
                            }
                        }
                    }
                    Ok(GcValue::Int64(count))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Ok(GcValue::Int64(0))
                }
            }),
        }));

        // Runtime.blockingThreads() -> Int - number of tokio blocking threads (Linux only)
        self.register_native("Runtime.blockingThreads", Arc::new(GcNativeFn {
            name: "Runtime.blockingThreads".to_string(),
            arity: 0,
            func: Box::new(|_args, _heap| {
                #[cfg(target_os = "linux")]
                {
                    let mut count = 0i64;
                    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
                        for entry in entries.flatten() {
                            let comm_path = entry.path().join("comm");
                            if let Ok(name) = std::fs::read_to_string(&comm_path) {
                                let name = name.trim();
                                if name.starts_with("tokio-runtime-b") {
                                    count += 1;
                                }
                            }
                        }
                    }
                    Ok(GcValue::Int64(count))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Ok(GcValue::Int64(0))
                }
            }),
        }));

        // Server.matchPath(path, pattern) -> [(String, String)]
        // Matches a path against a pattern with :params, returns extracted params or empty list if no match
        // e.g., matchPath("/users/123/posts/456", "/users/:id/posts/:postId") -> [("id", "123"), ("postId", "456")]
        self.register_native("Server.matchPath", Arc::new(GcNativeFn {
            name: "Server.matchPath".to_string(),
            arity: 2,
            func: Box::new(|args, heap| {
                let path = match &args[0] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };
                let pattern = match &args[1] {
                    GcValue::String(ptr) => heap.get_string(*ptr).map(|s| s.data.clone()),
                    _ => return Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                };

                match (path, pattern) {
                    (Some(path), Some(pattern)) => {
                        // Split both path and pattern into segments
                        let path_segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
                        let pattern_segs: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();

                        // Must have same number of segments
                        if path_segs.len() != pattern_segs.len() {
                            return Ok(GcValue::List(GcList::from_vec(vec![])));
                        }

                        // Match and extract params
                        let mut params: Vec<GcValue> = vec![];
                        for (path_seg, pattern_seg) in path_segs.iter().zip(pattern_segs.iter()) {
                            if pattern_seg.starts_with(':') {
                                // This is a param - extract the name (without the colon)
                                let param_name = &pattern_seg[1..];
                                // Create tuple (param_name, path_seg)
                                let key = GcValue::String(heap.alloc_string(param_name.to_string()));
                                let val = GcValue::String(heap.alloc_string(path_seg.to_string()));
                                params.push(GcValue::Tuple(heap.alloc_tuple(vec![key, val])));
                            } else if pattern_seg != path_seg {
                                // Literal segment doesn't match
                                return Ok(GcValue::List(GcList::from_vec(vec![])));
                            }
                            // Literal matches - continue
                        }

                        Ok(GcValue::List(GcList::from_vec(params)))
                    },
                    _ => Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                }
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

        // Spawn extension message handler task (if extension manager is configured)
        let ext_handler = {
            let extensions_guard = shared.extensions.read().unwrap();
            if let Some(ref ext_mgr) = *extensions_guard {
                if let Some(mut rx) = ext_mgr.take_message_receiver() {
                    let shared_for_ext = Arc::clone(&shared);
                    Some(tokio::spawn(async move {
                        while let Some(msg) = rx.recv().await {
                            // Convert extension Value to ThreadSafeValue
                            if let Some(ts_value) = crate::extensions::ext_value_to_thread_safe(&msg.value) {
                                let target_pid = crate::value::Pid(msg.target.0);
                                shared_for_ext.send_message(target_pid, ts_value).await;
                            }
                        }
                    }))
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Run main process (blocks until complete)
        let result = process.run().await;

        // Abort extension handler task when main completes
        if let Some(handle) = ext_handler {
            handle.abort();
        }

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

        // Spawn extension message handler task (if extension manager is configured)
        let ext_handler = {
            let extensions_guard = self.shared.extensions.read().unwrap();
            if let Some(ref ext_mgr) = *extensions_guard {
                if let Some(mut rx) = ext_mgr.take_message_receiver() {
                    let shared_for_ext = Arc::clone(&self.shared);
                    Some(tokio::spawn(async move {
                        while let Some(msg) = rx.recv().await {
                            // Convert extension Value to ThreadSafeValue
                            if let Some(ts_value) = crate::extensions::ext_value_to_thread_safe(&msg.value) {
                                let target_pid = crate::value::Pid(msg.target.0);
                                shared_for_ext.send_message(target_pid, ts_value).await;
                            }
                        }
                    }))
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Run main process (blocks until complete)
        let result = process.run().await;

        // Abort extension handler task when main completes
        if let Some(handle) = ext_handler {
            handle.abort();
        }

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

        // Log shared state info
        let shared_ptr = Arc::as_ptr(&self.shared) as usize;
        let has_output = self.shared.output_sender.is_some();
        AsyncProcess::debug_log(&format!("[run_debug] shared_ptr={:#x}, has_output_sender={}", shared_ptr, has_output));
        if let Some(ref sender) = self.shared.output_sender {
            let sender_ptr = sender as *const _ as usize;
            AsyncProcess::debug_log(&format!("[run_debug] output_sender_ptr={:#x}", sender_ptr));
        }

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

                // Log output_sender status after process creation
                if process.shared.output_sender.is_some() {
                    let sender_ptr = process.shared.output_sender.as_ref().unwrap() as *const _ as usize;
                    AsyncProcess::debug_log(&format!("[run_debug spawned] output_sender_ptr={:#x}", sender_ptr));
                } else {
                    AsyncProcess::debug_log("[run_debug spawned] NO output_sender!");
                }

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
            source: None,
            source_start_line: 0,
        };
        assert_eq!(frame.function, "main");
    }
}
