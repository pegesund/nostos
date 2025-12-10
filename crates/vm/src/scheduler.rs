//! Thread-safe scheduler for lightweight processes.
//!
//! The scheduler manages:
//! - Process registry (Pid -> Process mapping)
//! - Run queue (ready-to-run processes)
//! - Message routing between processes
//! - Link/monitor notification on exit
//!
//! Thread-safety: Uses parking_lot for fast locking and atomic
//! counters for Pid/RefId allocation. Ready for multi-CPU execution.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

use crate::gc::{GcConfig, GcNativeFn, GcValue};
use crate::process::{ExitReason, Process, ProcessState, ThreadSafeValue};
use crate::parallel::{JitIntFn, JitLoopArrayFn};
use crate::value::{FunctionValue, Pid, RefId, RuntimeError, TypeValue, Value};

/// Configuration for JIT compilation.
#[derive(Clone, Debug)]
pub struct JitConfig {
    /// Call count threshold before JIT compilation
    pub hot_threshold: usize,
    /// Maximum functions to queue for JIT at once
    pub max_queue_size: usize,
    /// Enable JIT compilation
    pub enabled: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 10, // Lowered for aggressive JIT compilation in benchmarks
            max_queue_size: 64,
            enabled: true,
        }
    }
}

/// JIT-compiled function signature.
///
/// Takes a pointer to the register file and returns an i64 status code.
/// Status codes:
/// - 0: Success, result is in r0
/// - Non-zero: Need to fall back to interpreter
///
/// This signature is designed for:
/// - Minimal overhead (single pointer arg)
/// - Easy Cranelift code generation
/// - Simple fallback to interpreter
pub type JitFn = unsafe extern "C" fn(*mut Value) -> i64;

/// A compiled function with metadata.
#[derive(Clone)]
pub struct CompiledFunction {
    /// The compiled function pointer
    pub code: JitFn,
    /// Size of the compiled code in bytes (for memory tracking)
    pub code_size: usize,
}

/// Trait for JIT compilation backends.
///
/// This trait allows different compilation backends to be plugged in.
/// The primary implementation will be Cranelift, but this abstraction
/// allows for testing with stubs.
pub trait JitBackend: Send + Sync {
    /// Compile a function to native code.
    /// Returns None if compilation fails or is not possible.
    fn compile(&self, func: &FunctionValue) -> Option<CompiledFunction>;

    /// Get the name of this backend (for logging/debugging).
    fn name(&self) -> &str;
}

/// A no-op JIT backend for when JIT is disabled.
pub struct NoJitBackend;

impl JitBackend for NoJitBackend {
    fn compile(&self, _func: &FunctionValue) -> Option<CompiledFunction> {
        None
    }

    fn name(&self) -> &str {
        "none"
    }
}

/// Tracks hot functions for JIT compilation.
///
/// Thread-safe: uses atomic counters and locks for queuing.
pub struct JitTracker {
    /// Call counts per function name
    call_counts: RwLock<HashMap<String, AtomicUsize>>,
    /// Functions queued for JIT compilation
    jit_queue: Mutex<VecDeque<String>>,
    /// Functions already JIT compiled (to avoid re-queuing)
    compiled: RwLock<HashSet<String>>,
    /// Compiled function code (fast lookup for dispatch)
    compiled_code: RwLock<HashMap<String, CompiledFunction>>,
    /// Total compiled code size in bytes
    total_code_size: AtomicUsize,
    /// Configuration
    config: JitConfig,
}

impl JitTracker {
    /// Create a new JIT tracker.
    pub fn new(config: JitConfig) -> Self {
        Self {
            call_counts: RwLock::new(HashMap::new()),
            jit_queue: Mutex::new(VecDeque::new()),
            compiled: RwLock::new(HashSet::new()),
            compiled_code: RwLock::new(HashMap::new()),
            total_code_size: AtomicUsize::new(0),
            config,
        }
    }

    /// Record a function call and check if it should be JIT compiled.
    /// Returns true if function just became hot (should be queued for JIT).
    pub fn record_call(&self, func_name: &str) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Check if already compiled
        if self.compiled.read().contains(func_name) {
            return false;
        }

        // Increment call count
        let counts = self.call_counts.read();
        if let Some(counter) = counts.get(func_name) {
            let prev = counter.fetch_add(1, Ordering::Relaxed);
            if prev + 1 == self.config.hot_threshold {
                // Just reached threshold - queue for JIT
                drop(counts);
                self.queue_for_jit(func_name);
                return true;
            }
            return false;
        }
        drop(counts);

        // First call - add to tracking
        let mut counts = self.call_counts.write();
        counts.entry(func_name.to_string())
            .or_insert_with(|| AtomicUsize::new(1));
        false
    }

    /// Queue a function for JIT compilation.
    fn queue_for_jit(&self, func_name: &str) {
        let mut queue = self.jit_queue.lock();
        if queue.len() < self.config.max_queue_size {
            queue.push_back(func_name.to_string());
        }
    }

    /// Get next function to JIT compile.
    pub fn pop_jit_queue(&self) -> Option<String> {
        self.jit_queue.lock().pop_front()
    }

    /// Mark a function as JIT compiled.
    pub fn mark_compiled(&self, func_name: &str) {
        self.compiled.write().insert(func_name.to_string());
    }

    /// Check if a function is hot (above threshold).
    pub fn is_hot(&self, func_name: &str) -> bool {
        if let Some(counter) = self.call_counts.read().get(func_name) {
            counter.load(Ordering::Relaxed) >= self.config.hot_threshold
        } else {
            false
        }
    }

    /// Get call count for a function.
    pub fn get_call_count(&self, func_name: &str) -> usize {
        self.call_counts.read()
            .get(func_name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Check if JIT is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the hot threshold.
    pub fn hot_threshold(&self) -> usize {
        self.config.hot_threshold
    }

    /// Register a compiled function.
    /// This stores the compiled code and marks the function as compiled.
    pub fn register_compiled(&self, func_name: &str, compiled: CompiledFunction) {
        let code_size = compiled.code_size;
        self.compiled_code.write().insert(func_name.to_string(), compiled);
        self.compiled.write().insert(func_name.to_string());
        self.total_code_size.fetch_add(code_size, Ordering::Relaxed);
    }

    /// Get compiled function if available.
    /// Returns the JIT function pointer for fast dispatch.
    #[inline]
    pub fn get_compiled(&self, func_name: &str) -> Option<JitFn> {
        self.compiled_code.read().get(func_name).map(|f| f.code)
    }

    /// Fast check if function has JIT code.
    /// This is optimized for the hot path - uses read lock.
    #[inline]
    pub fn is_compiled(&self, func_name: &str) -> bool {
        self.compiled.read().contains(func_name)
    }

    /// Get total compiled code size in bytes.
    pub fn total_code_size(&self) -> usize {
        self.total_code_size.load(Ordering::Relaxed)
    }

    /// Get number of compiled functions.
    pub fn compiled_count(&self) -> usize {
        self.compiled.read().len()
    }

    /// Process the JIT queue with a backend.
    /// Compiles queued functions and registers them.
    /// Returns number of functions successfully compiled.
    pub fn process_queue<B: JitBackend>(&self, backend: &B, functions: &HashMap<String, Arc<FunctionValue>>) -> usize {
        let mut compiled = 0;

        while let Some(func_name) = self.pop_jit_queue() {
            // Skip if already compiled (might have been compiled by another thread)
            if self.is_compiled(&func_name) {
                continue;
            }

            // Get the function definition
            if let Some(func) = functions.get(&func_name) {
                // Try to compile
                if let Some(compiled_fn) = backend.compile(func) {
                    self.register_compiled(&func_name, compiled_fn);
                    compiled += 1;
                }
            }
        }

        compiled
    }
}

/// A thread-safe handle to a process.
///
/// Processes are wrapped in Arc<Mutex<>> to allow safe access from
/// multiple worker threads. The Mutex is from parking_lot for speed.
pub type ProcessHandle = Arc<Mutex<Process>>;

/// The thread-safe process scheduler.
///
/// Manages all processes and coordinates their execution.
/// All fields are thread-safe, ready for multi-CPU scheduling.
pub struct Scheduler {
    /// All processes by Pid (thread-safe).
    processes: RwLock<HashMap<Pid, ProcessHandle>>,

    /// Run queue (Pids of ready processes).
    run_queue: Mutex<VecDeque<Pid>>,

    /// Waiting processes (in receive, not in run queue).
    waiting: Mutex<Vec<Pid>>,

    /// Currently running process Pid (for single-threaded mode).
    /// In multi-threaded mode, each worker tracks its own current.
    current: Mutex<Option<Pid>>,

    /// Next Pid to allocate (atomic counter).
    next_pid: AtomicU64,

    /// Next RefId for monitors (atomic counter).
    next_ref: AtomicU64,

    /// Global functions (shared across processes).
    /// RwLock since these are typically only written at startup.
    pub functions: RwLock<HashMap<String, Arc<FunctionValue>>>,

    /// Functions by index (for CallDirect/TailCallDirect).
    /// Parallel to functions map, indexed by func_idx.
    pub function_list: RwLock<Vec<Arc<FunctionValue>>>,

    /// Native functions (shared across processes).
    pub natives: RwLock<HashMap<String, Arc<GcNativeFn>>>,

    /// Type definitions (shared).
    pub types: RwLock<HashMap<String, Arc<TypeValue>>>,

    /// Global variables (shared, read-only after init).
    pub globals: RwLock<HashMap<String, Value>>,

    /// JIT compilation tracker (hot function detection).
    pub jit_tracker: JitTracker,

    /// JIT-compiled integer functions (func_index -> JIT fn).
    pub jit_int_functions: RwLock<HashMap<u16, JitIntFn>>,

    /// JIT-compiled loop array functions (func_index -> JIT fn).
    pub jit_loop_array_functions: RwLock<HashMap<u16, JitLoopArrayFn>>,

    /// Active (non-exited) process count - atomic for fast access.
    /// Avoids expensive iteration in process_count().
    active_process_count: AtomicUsize,

    /// Timer heap for Sleep/ReceiveTimeout - stores (wake_time, Pid) pairs.
    /// Uses Reverse for min-heap behavior (earliest wake time first).
    pub timer_heap: Mutex<BinaryHeap<Reverse<(Instant, Pid)>>>,

    /// Claimed processes (being executed by a worker).
    /// Prevents race conditions where a process is added to a queue while already executing.
    claimed: Mutex<HashSet<u64>>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self::with_jit_config(JitConfig::default())
    }

    /// Create a scheduler with custom JIT configuration.
    pub fn with_jit_config(jit_config: JitConfig) -> Self {
        Self {
            processes: RwLock::new(HashMap::new()),
            run_queue: Mutex::new(VecDeque::new()),
            waiting: Mutex::new(Vec::new()),
            current: Mutex::new(None),
            next_pid: AtomicU64::new(1), // Pid 0 is reserved
            next_ref: AtomicU64::new(1),
            functions: RwLock::new(HashMap::new()),
            function_list: RwLock::new(Vec::new()),
            natives: RwLock::new(HashMap::new()),
            types: RwLock::new(HashMap::new()),
            globals: RwLock::new(HashMap::new()),
            jit_tracker: JitTracker::new(jit_config),
            jit_int_functions: RwLock::new(HashMap::new()),
            jit_loop_array_functions: RwLock::new(HashMap::new()),
            active_process_count: AtomicUsize::new(0),
            timer_heap: Mutex::new(BinaryHeap::new()),
            claimed: Mutex::new(HashSet::new()),
        }
    }

    /// Get the active process count (fast, no locking).
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_process_count.load(Ordering::Relaxed)
    }

    /// Register the default native functions (show, copy, print, println).
    pub fn register_default_natives(&self) {
        use crate::gc::GcNativeFn;

        // Show - convert value to string
        self.natives.write().insert("show".to_string(), Arc::new(GcNativeFn {
            name: "show".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                Ok(GcValue::String(heap.alloc_string(s)))
            }),
        }));

        // Copy - deep copy a value
        self.natives.write().insert("copy".to_string(), Arc::new(GcNativeFn {
            name: "copy".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                Ok(heap.clone_value(&args[0]))
            }),
        }));

        // Print - print without newline
        self.natives.write().insert("print".to_string(), Arc::new(GcNativeFn {
            name: "print".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                print!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // Println - print with newline
        self.natives.write().insert("println".to_string(), Arc::new(GcNativeFn {
            name: "println".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                println!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // String.length
        self.natives.write().insert("String.length".to_string(), Arc::new(GcNativeFn {
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

        // String.chars
        self.natives.write().insert("String.chars".to_string(), Arc::new(GcNativeFn {
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
        self.natives.write().insert("String.from_chars".to_string(), Arc::new(GcNativeFn {
            name: "String.from_chars".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::List(list) => {
                        let mut s = String::new();
                        for item in list.items() {
                            match item {
                                GcValue::Char(c) => s.push(*c),
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
        self.natives.write().insert("String.to_int".to_string(), Arc::new(GcNativeFn {
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

    }

    /// Spawn a new process.
    /// Returns the new process's Pid.
    /// Note: Adds process to internal run queue. For WorkerPool use, consider spawn_unqueued().
    pub fn spawn(&self) -> Pid {
        self.spawn_with_config(GcConfig::default())
    }

    /// Spawn a child process with lightweight heap (for mass spawning).
    /// Returns the new process's Pid.
    pub fn spawn_child(&self) -> Pid {
        self.spawn_with_config(GcConfig::lightweight())
    }

    /// Spawn a new process with custom GC config.
    /// Note: Adds process to internal run queue. For WorkerPool use, consider spawn_unqueued().
    pub fn spawn_with_config(&self, gc_config: GcConfig) -> Pid {
        let pid = Pid(self.next_pid.fetch_add(1, Ordering::SeqCst));

        let process = Process::with_gc_config(pid, gc_config);
        let handle = Arc::new(Mutex::new(process));

        self.processes.write().insert(pid, handle);
        self.run_queue.lock().push_back(pid);
        self.active_process_count.fetch_add(1, Ordering::Relaxed);

        pid
    }

    /// Spawn a new process without adding to the run queue.
    /// Used by WorkerPool which manages its own work queues.
    pub fn spawn_unqueued(&self) -> Pid {
        self.spawn_unqueued_with_config(GcConfig::default())
    }

    /// Spawn a lightweight process without adding to the run queue.
    /// Uses minimal heap pre-allocation for memory-efficient mass spawning.
    pub fn spawn_lightweight(&self) -> Pid {
        self.spawn_unqueued_with_config(GcConfig::lightweight())
    }

    /// Spawn with config without adding to run queue.
    pub fn spawn_unqueued_with_config(&self, gc_config: GcConfig) -> Pid {
        let pid = Pid(self.next_pid.fetch_add(1, Ordering::SeqCst));

        let process = Process::with_gc_config(pid, gc_config);
        let handle = Arc::new(Mutex::new(process));

        self.processes.write().insert(pid, handle);
        self.active_process_count.fetch_add(1, Ordering::Relaxed);
        // Don't add to run_queue - caller will manage scheduling

        pid
    }

    /// Spawn a linked process.
    /// If either process dies with an error, the other is killed too.
    pub fn spawn_link(&self, parent_pid: Pid) -> Pid {
        let child_pid = self.spawn();

        // Create bidirectional link
        // Lock ordering: always lock lower Pid first to prevent deadlock
        let (first_pid, second_pid) = if parent_pid.0 < child_pid.0 {
            (parent_pid, child_pid)
        } else {
            (child_pid, parent_pid)
        };

        let processes = self.processes.read();
        if let (Some(first), Some(second)) = (processes.get(&first_pid), processes.get(&second_pid)) {
            let mut first_lock = first.lock();
            let mut second_lock = second.lock();

            if first_pid == parent_pid {
                first_lock.link(child_pid);
                second_lock.link(parent_pid);
            } else {
                first_lock.link(parent_pid);
                second_lock.link(child_pid);
            }
        }

        child_pid
    }

    /// Spawn a monitored process.
    /// Returns (child_pid, monitor_ref).
    pub fn spawn_monitor(&self, parent_pid: Pid) -> (Pid, RefId) {
        let child_pid = self.spawn();
        let ref_id = RefId(self.next_ref.fetch_add(1, Ordering::SeqCst));

        // Lock ordering: always lock lower Pid first
        let (first_pid, second_pid) = if parent_pid.0 < child_pid.0 {
            (parent_pid, child_pid)
        } else {
            (child_pid, parent_pid)
        };

        let processes = self.processes.read();
        if let (Some(first), Some(second)) = (processes.get(&first_pid), processes.get(&second_pid)) {
            let mut first_lock = first.lock();
            let mut second_lock = second.lock();

            if first_pid == parent_pid {
                first_lock.add_monitor(ref_id, child_pid);
                second_lock.add_monitored_by(ref_id, parent_pid);
            } else {
                second_lock.add_monitor(ref_id, child_pid);
                first_lock.add_monitored_by(ref_id, parent_pid);
            }
        }

        (child_pid, ref_id)
    }

    /// Allocate a new unique reference.
    pub fn make_ref(&self) -> RefId {
        RefId(self.next_ref.fetch_add(1, Ordering::SeqCst))
    }

    /// Get a process handle by Pid.
    /// Returns a cloned Arc - safe to hold across await points.
    pub fn get_process_handle(&self, pid: Pid) -> Option<ProcessHandle> {
        self.processes.read().get(&pid).cloned()
    }

    /// Execute a function with a locked process.
    /// This is the preferred way to access process data.
    pub fn with_process<F, R>(&self, pid: Pid, f: F) -> Option<R>
    where
        F: FnOnce(&Process) -> R,
    {
        let processes = self.processes.read();
        processes.get(&pid).map(|handle| f(&handle.lock()))
    }

    /// Execute a function with a mutable locked process.
    pub fn with_process_mut<F, R>(&self, pid: Pid, f: F) -> Option<R>
    where
        F: FnOnce(&mut Process) -> R,
    {
        let processes = self.processes.read();
        processes.get(&pid).map(|handle| f(&mut handle.lock()))
    }

    /// Try to get exclusive mutable access to a process without blocking.
    /// Returns None if the process doesn't exist or if another thread holds the lock.
    /// This is used by the parallel VM to claim processes for execution.
    pub fn try_with_process_mut<F, R>(&self, pid: Pid, f: F) -> Option<R>
    where
        F: FnOnce(&mut Process) -> R,
    {
        let processes = self.processes.read();
        processes.get(&pid).and_then(|handle| {
            handle.try_lock().map(|mut guard| f(&mut guard))
        })
    }

    /// Get current process Pid.
    pub fn current_pid(&self) -> Option<Pid> {
        *self.current.lock()
    }

    /// Send a message from one process to another.
    /// The message is deep-copied to the target's heap.
    pub fn send(&self, from_pid: Pid, to_pid: Pid, message: GcValue) -> Result<(), RuntimeError> {
        let processes = self.processes.read();

        // 1. Convert to ThreadSafeValue using source heap
        let safe_msg = if let Some(handle) = processes.get(&from_pid) {
            let source = handle.lock();
            ThreadSafeValue::from_gc_value(&message, &source.heap)
        } else {
            return Err(RuntimeError::Panic(format!("Sender process {:?} not found", from_pid)));
        };

        // 2. Send to target
        if let Some(msg) = safe_msg {
            if let Some(handle) = processes.get(&to_pid) {
                let mut target = handle.lock();
                let _ = target.sender.send(msg);

                // Wake up the target if it was waiting for a message
                if target.state == ProcessState::Waiting {
                    target.state = ProcessState::Running;
                } else if target.state == ProcessState::WaitingTimeout {
                    target.state = ProcessState::Running;
                }
                drop(target); // Unlock before waking

                // Add to run queue if was waiting
                self.wake_process(to_pid);
            }
        }

        Ok(())
    }

    /// Wake a waiting process (add to run queue if not there).
    fn wake_process(&self, pid: Pid) {
        let mut run_queue = self.run_queue.lock();
        let mut waiting = self.waiting.lock();

        waiting.retain(|&p| p != pid);
        if !run_queue.contains(&pid) {
            run_queue.push_back(pid);
        }
    }

    /// Drain all pids from the run queue.
    /// Used by WorkerPool to pick up newly spawned/woken processes.
    pub fn drain_run_queue(&self) -> Vec<Pid> {
        self.run_queue.lock().drain(..).collect()
    }

    /// Pop a single pid from the run queue.
    /// Returns None if queue is empty.
    pub fn pop_run_queue(&self) -> Option<Pid> {
        self.run_queue.lock().pop_front()
    }

    /// Pick next process to run (round-robin).
    pub fn schedule_next(&self) -> Option<Pid> {
        let mut current_guard = self.current.lock();

        // Put current back in queue if still runnable
        if let Some(current_pid) = *current_guard {
            if let Some(handle) = self.processes.read().get(&current_pid) {
                let proc = handle.lock();
                match proc.state {
                    ProcessState::Running | ProcessState::Suspended => {
                        drop(proc);
                        self.run_queue.lock().push_back(current_pid);
                    }
                    ProcessState::Waiting | ProcessState::WaitingTimeout
                    | ProcessState::Sleeping | ProcessState::WaitingIO => {
                        drop(proc);
                        let mut waiting = self.waiting.lock();
                        if !waiting.contains(&current_pid) {
                            waiting.push(current_pid);
                        }
                    }
                    ProcessState::Exited(_) => {
                        // Don't re-queue, will be cleaned up
                    }
                }
            }
        }

        // Get next from run queue
        let next = self.run_queue.lock().pop_front();
        *current_guard = next;

        // Reset reductions for new process
        if let Some(pid) = next {
            if let Some(handle) = self.processes.read().get(&pid) {
                let mut proc = handle.lock();
                proc.reset_reductions();
                proc.state = ProcessState::Running;
            }
        }

        next
    }

    /// Mark current process as yielded and schedule next.
    pub fn yield_current(&self) -> Option<Pid> {
        if let Some(pid) = self.current_pid() {
            self.with_process_mut(pid, |proc| proc.suspend());
        }
        self.schedule_next()
    }

    /// Process exited - handle links and monitors.
    pub fn process_exit(&self, pid: Pid, reason: ExitReason, value: Option<GcValue>) {
        let (links, monitored_by) = {
            let processes = self.processes.read();
            if let Some(handle) = processes.get(&pid) {
                let mut proc = handle.lock();
                proc.exit(reason.clone(), value.clone());
                (proc.links.clone(), proc.monitored_by.clone())
            } else {
                return;
            }
        };

        // Propagate to linked processes
        let is_error = !matches!(reason, ExitReason::Normal);
        if is_error {
            for linked_pid in links {
                self.with_process_mut(linked_pid, |linked| {
                    linked.unlink(pid);
                    if !matches!(linked.state, ProcessState::Exited(_)) {
                        linked.exit(
                            ExitReason::LinkedExit(pid, format!("{:?}", reason)),
                            None,
                        );
                    }
                });
            }
        }

        // Send DOWN messages to monitors
        for (ref_id, watcher_pid) in monitored_by {
            self.with_process_mut(watcher_pid, |watcher| {
                // Create DOWN message: (ref, pid)
                let down_msg = ThreadSafeValue::Tuple(vec![
                    ThreadSafeValue::Int64(ref_id.0 as i64),
                    ThreadSafeValue::Pid(pid.0),
                ]);
                let _ = watcher.sender.send(down_msg);

                // Wake if waiting
                if watcher.state == ProcessState::Waiting {
                    watcher.state = ProcessState::Running;
                }

                watcher.monitors.remove(&ref_id);
            });

            self.wake_process(watcher_pid);
        }

        // Remove from queues
        self.run_queue.lock().retain(|&p| p != pid);
        self.waiting.lock().retain(|&p| p != pid);
        self.active_process_count.fetch_sub(1, Ordering::Relaxed);

        let mut current_guard = self.current.lock();
        if *current_guard == Some(pid) {
            *current_guard = None;
        }
    }

    /// Check if any processes are still alive.
    pub fn has_processes(&self) -> bool {
        self.processes.read().values().any(|handle| {
            !handle.lock().is_exited()
        })
    }

    /// Get count of active (non-exited) processes.
    pub fn process_count(&self) -> usize {
        self.processes.read().values()
            .filter(|handle| !handle.lock().is_exited())
            .count()
    }

    /// Get count of runnable processes.
    pub fn runnable_count(&self) -> usize {
        self.run_queue.lock().len() + if self.current_pid().is_some() { 1 } else { 0 }
    }

    /// Clean up exited processes.
    pub fn cleanup_exited(&self) {
        self.processes.write().retain(|_, handle| {
            !handle.lock().is_exited()
        });
    }

    /// Add a timer for a process to wake at a specific time.
    pub fn add_timer(&self, wake_time: Instant, pid: Pid) {
        self.timer_heap.lock().push(Reverse((wake_time, pid)));
    }

    /// Check for expired timers and return (wake_time, pid) pairs.
    /// The wake_time is included so callers can verify it matches the process's current wake_time
    /// (to detect stale timer entries from processes that were woken by messages).
    pub fn check_timers(&self) -> Vec<(Instant, Pid)> {
        let now = Instant::now();
        let mut heap = self.timer_heap.lock();
        let mut woken = Vec::new();

        while let Some(&Reverse((wake_time, pid))) = heap.peek() {
            if wake_time <= now {
                heap.pop();
                woken.push((wake_time, pid));
            } else {
                break;
            }
        }

        woken
    }

    /// Get the next timer deadline (if any) for efficient sleeping.
    pub fn next_timer_deadline(&self) -> Option<Instant> {
        self.timer_heap.lock().peek().map(|Reverse((t, _))| *t)
    }

    /// Try to claim a process for execution.
    /// Returns true if successfully claimed, false if already claimed by another worker.
    pub fn claim_process(&self, pid: Pid) -> bool {
        let mut claimed = self.claimed.lock();
        if claimed.contains(&pid.0) {
            false
        } else {
            claimed.insert(pid.0);
            true
        }
    }

    /// Unclaim a process after execution.
    pub fn unclaim_process(&self, pid: Pid) {
        self.claimed.lock().remove(&pid.0);
    }

    /// Check if a process is claimed.
    pub fn is_claimed(&self, pid: Pid) -> bool {
        self.claimed.lock().contains(&pid.0)
    }
}

// SAFETY: Scheduler uses only thread-safe primitives internally
unsafe impl Send for Scheduler {}
unsafe impl Sync for Scheduler {}

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
        let sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        assert_ne!(pid1, pid2);
        assert_eq!(sched.process_count(), 2);
        assert_eq!(sched.runnable_count(), 2);
    }

    #[test]
    fn test_scheduling() {
        let sched = Scheduler::new();

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
        let sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        // Send message from pid1 to pid2
        let msg = GcValue::Int64(42);
        sched.send(pid1, pid2, msg).unwrap();

        // Check pid2 received it
        sched.with_process_mut(pid2, |proc| {
            assert!(proc.has_messages());
            let received = proc.try_receive().unwrap();
            assert_eq!(received, GcValue::Int64(42));
            assert!(!proc.has_messages()); // Mailbox should now be empty
        });
    }

    #[test]
    fn test_spawn_link() {
        let sched = Scheduler::new();

        let parent = sched.spawn();
        let child = sched.spawn_link(parent);

        // Both should be linked
        sched.with_process(parent, |proc| {
            assert!(proc.links.contains(&child));
        });
        sched.with_process(child, |proc| {
            assert!(proc.links.contains(&parent));
        });
    }

    #[test]
    fn test_process_exit_kills_linked() {
        let sched = Scheduler::new();

        let parent = sched.spawn();
        let child = sched.spawn_link(parent);

        // Kill child with error
        sched.process_exit(child, ExitReason::Error("crash".to_string()), None);

        // Parent should also be dead
        sched.with_process(parent, |proc| {
            assert!(proc.is_exited());
        });
    }

    #[test]
    fn test_spawn_monitor() {
        let sched = Scheduler::new();

        let watcher = sched.spawn();
        let (child, _ref_id) = sched.spawn_monitor(watcher);

        // Kill child
        sched.process_exit(child, ExitReason::Normal, None);

        // Watcher should have DOWN message
        sched.with_process(watcher, |proc| {
            assert!(proc.has_messages());
        });
    }

    #[test]
    fn test_waiting_wakeup() {
        let sched = Scheduler::new();

        let pid1 = sched.spawn();
        let pid2 = sched.spawn();

        // Put pid2 in waiting state
        sched.with_process_mut(pid2, |proc| {
            proc.wait_for_message();
        });
        sched.with_process(pid2, |proc| {
            assert_eq!(proc.state, ProcessState::Waiting);
        });

        // Send message should wake it up
        sched.send(pid1, pid2, GcValue::Int64(1)).unwrap();
        sched.with_process(pid2, |proc| {
            assert_eq!(proc.state, ProcessState::Running);
        });
    }

    #[test]
    fn test_concurrent_spawn() {
        use std::thread;

        let sched = Arc::new(Scheduler::new());
        let mut handles = vec![];

        // Spawn from multiple threads
        for _ in 0..4 {
            let sched_clone = Arc::clone(&sched);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    sched_clone.spawn();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 400 processes
        assert_eq!(sched.process_count(), 400);
    }

    #[test]
    fn test_concurrent_message_passing() {
        use std::thread;

        let sched = Arc::new(Scheduler::new());
        let receiver = sched.spawn();

        let mut handles = vec![];

        // Send messages from multiple threads
        for i in 0..4 {
            let sched_clone = Arc::clone(&sched);
            let sender = sched.spawn();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let msg = GcValue::Int64((i * 100 + j) as i64);
                    sched_clone.send(sender, receiver, msg).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Receiver should have 400 messages
        sched.with_process_mut(receiver, |proc| {
            // Drain all messages from the receiver's channel
            let mut count = 0;
            while proc.try_receive().is_some() {
                count += 1;
            }
            assert_eq!(count, 400);
        });
    }

    #[test]
    fn test_jit_tracker_call_counting() {
        let tracker = JitTracker::new(JitConfig {
            hot_threshold: 10,
            max_queue_size: 64,
            enabled: true,
        });

        // First 9 calls shouldn't trigger
        for _ in 0..9 {
            assert!(!tracker.record_call("test_func"));
        }

        assert_eq!(tracker.get_call_count("test_func"), 9);
        assert!(!tracker.is_hot("test_func"));

        // 10th call should trigger (reach threshold)
        assert!(tracker.record_call("test_func"));
        assert!(tracker.is_hot("test_func"));

        // Should be queued for JIT
        assert_eq!(tracker.pop_jit_queue(), Some("test_func".to_string()));
        assert_eq!(tracker.pop_jit_queue(), None);

        // More calls shouldn't trigger again
        for _ in 0..10 {
            assert!(!tracker.record_call("test_func"));
        }
    }

    #[test]
    fn test_jit_tracker_compiled_not_queued() {
        let tracker = JitTracker::new(JitConfig {
            hot_threshold: 5,
            max_queue_size: 64,
            enabled: true,
        });

        // Reach threshold
        for _ in 0..4 {
            tracker.record_call("func1");
        }
        assert!(tracker.record_call("func1")); // 5th call triggers

        // Mark as compiled
        tracker.mark_compiled("func1");

        // More calls shouldn't re-queue
        for _ in 0..10 {
            assert!(!tracker.record_call("func1"));
        }

        // Queue should only have func1 once (from before mark_compiled)
        assert_eq!(tracker.pop_jit_queue(), Some("func1".to_string()));
        assert_eq!(tracker.pop_jit_queue(), None);
    }

    #[test]
    fn test_jit_tracker_disabled() {
        let tracker = JitTracker::new(JitConfig {
            hot_threshold: 5,
            max_queue_size: 64,
            enabled: false, // Disabled
        });

        // Calls should not be tracked
        for _ in 0..100 {
            assert!(!tracker.record_call("func"));
        }

        // Should not be hot (not tracked)
        assert!(!tracker.is_hot("func"));
        assert_eq!(tracker.pop_jit_queue(), None);
    }

    #[test]
    fn test_jit_tracker_multiple_functions() {
        let tracker = JitTracker::new(JitConfig {
            hot_threshold: 3,
            max_queue_size: 64,
            enabled: true,
        });

        // Call multiple functions
        for _ in 0..2 {
            tracker.record_call("slow");
        }
        for _ in 0..3 {
            tracker.record_call("fast");
        }

        // fast should be hot, slow should not
        assert!(!tracker.is_hot("slow"));
        assert!(tracker.is_hot("fast"));

        // fast should be in queue
        assert_eq!(tracker.pop_jit_queue(), Some("fast".to_string()));
    }

    #[test]
    fn test_jit_register_and_get_compiled() {
        let tracker = JitTracker::new(JitConfig::default());

        // Create a stub JIT function for testing
        unsafe extern "C" fn stub_jit_fn(_regs: *mut Value) -> i64 {
            0 // Success
        }

        // Register compiled function
        let compiled = CompiledFunction {
            code: stub_jit_fn,
            code_size: 128,
        };
        tracker.register_compiled("test_func", compiled);

        // Should be marked as compiled
        assert!(tracker.is_compiled("test_func"));

        // Should be retrievable
        let retrieved = tracker.get_compiled("test_func");
        assert!(retrieved.is_some());

        // Code size should be tracked
        assert_eq!(tracker.total_code_size(), 128);
        assert_eq!(tracker.compiled_count(), 1);
    }

    #[test]
    fn test_jit_multiple_compiled_functions() {
        let tracker = JitTracker::new(JitConfig::default());

        unsafe extern "C" fn stub_fn(_regs: *mut Value) -> i64 { 0 }

        // Register multiple functions
        for i in 0..5 {
            let compiled = CompiledFunction {
                code: stub_fn,
                code_size: 100 + i * 10,
            };
            tracker.register_compiled(&format!("func_{}", i), compiled);
        }

        // All should be compiled
        assert_eq!(tracker.compiled_count(), 5);
        assert_eq!(tracker.total_code_size(), 100 + 110 + 120 + 130 + 140);

        // All should be retrievable
        for i in 0..5 {
            assert!(tracker.is_compiled(&format!("func_{}", i)));
            assert!(tracker.get_compiled(&format!("func_{}", i)).is_some());
        }

        // Non-existent function should return None
        assert!(!tracker.is_compiled("nonexistent"));
        assert!(tracker.get_compiled("nonexistent").is_none());
    }

    #[test]
    fn test_jit_process_queue_with_mock_backend() {
        use crate::value::Chunk;

        struct MockBackend {
            should_compile: bool,
        }

        impl JitBackend for MockBackend {
            fn compile(&self, _func: &FunctionValue) -> Option<CompiledFunction> {
                if self.should_compile {
                    unsafe extern "C" fn mock_fn(_regs: *mut Value) -> i64 { 0 }
                    Some(CompiledFunction {
                        code: mock_fn,
                        code_size: 64,
                    })
                } else {
                    None
                }
            }

            fn name(&self) -> &str {
                "mock"
            }
        }

        let tracker = JitTracker::new(JitConfig {
            hot_threshold: 2,
            max_queue_size: 64,
            enabled: true,
        });

        // Create function definitions
        let mut functions: HashMap<String, Arc<FunctionValue>> = HashMap::new();
        functions.insert("hot_func".to_string(), Arc::new(FunctionValue {
            name: "hot_func".to_string(),
            arity: 0,
            param_names: vec![],
            code: Arc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: std::sync::atomic::AtomicU32::new(0),
            debug_symbols: vec![],
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        }));

        // Make function hot
        tracker.record_call("hot_func");
        tracker.record_call("hot_func"); // Triggers queueing

        assert!(!tracker.jit_queue.lock().is_empty());

        // Process with mock backend that succeeds
        let backend = MockBackend { should_compile: true };
        let compiled = tracker.process_queue(&backend, &functions);

        assert_eq!(compiled, 1);
        assert!(tracker.is_compiled("hot_func"));
        assert_eq!(tracker.total_code_size(), 64);
    }

    #[test]
    fn test_jit_no_jit_backend() {
        use crate::value::Chunk;

        let backend = NoJitBackend;

        let func = FunctionValue {
            name: "test".to_string(),
            arity: 0,
            param_names: vec![],
            code: Arc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: std::sync::atomic::AtomicU32::new(0),
            debug_symbols: vec![],
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        };

        // NoJitBackend should never compile
        assert!(backend.compile(&func).is_none());
        assert_eq!(backend.name(), "none");
    }
}
