//! Parallel VM with CPU affinity for Erlang-style concurrency.
//!
//! Design principles:
//! - Each thread owns its processes (no locking for local operations)
//! - Processes stay on the thread that spawned them (affinity)
//! - Cross-thread messages via lock-free channels
//! - Pid encodes thread ID for instant routing
//!
//! This avoids contention in the hot path:
//! - Local scheduling: no locks
//! - Local process access: no locks
//! - Cross-thread messaging: lock-free channels

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam::channel::{self, Sender, Receiver, TryRecvError};

use crate::gc::{GcConfig, GcNativeFn, GcValue, Heap};
use crate::process::{Process, ProcessState};
use crate::runtime::{JitIntFn, JitLoopArrayFn};
use crate::value::{FunctionValue, Instruction, Pid, RuntimeError, TypeValue, Value};

/// Number of bits reserved for thread ID in Pid.
/// With 16 bits, we support up to 65536 threads.
const THREAD_ID_BITS: u32 = 16;
const THREAD_ID_SHIFT: u32 = 64 - THREAD_ID_BITS;
const LOCAL_ID_MASK: u64 = (1u64 << THREAD_ID_SHIFT) - 1;

/// Encode a Pid from thread ID and local sequence.
#[inline]
fn encode_pid(thread_id: u16, local_id: u64) -> Pid {
    Pid(((thread_id as u64) << THREAD_ID_SHIFT) | (local_id & LOCAL_ID_MASK))
}

/// Extract thread ID from a Pid.
#[inline]
fn pid_thread_id(pid: Pid) -> u16 {
    (pid.0 >> THREAD_ID_SHIFT) as u16
}

/// Extract local ID from a Pid.
#[inline]
fn pid_local_id(pid: Pid) -> u64 {
    pid.0 & LOCAL_ID_MASK
}

/// Thread-safe message value for cross-thread communication.
/// This is a subset of values that can be serialized and sent between threads.
#[derive(Debug, Clone)]
pub enum ThreadSafeValue {
    Unit,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Pid(u64),
    String(String),
    Char(char),
    List(Vec<ThreadSafeValue>),
    Tuple(Vec<ThreadSafeValue>),
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<ThreadSafeValue>,
        mutable_fields: Vec<bool>,
    },
}

impl ThreadSafeValue {
    /// Convert a GcValue to a thread-safe value (deep copy).
    fn from_gc_value(value: &GcValue, heap: &Heap) -> Option<Self> {
        Some(match value {
            GcValue::Unit => ThreadSafeValue::Unit,
            GcValue::Bool(b) => ThreadSafeValue::Bool(*b),
            GcValue::Int64(i) => ThreadSafeValue::Int64(*i),
            GcValue::Float64(f) => ThreadSafeValue::Float64(*f),
            GcValue::Pid(p) => ThreadSafeValue::Pid(*p),
            GcValue::Char(c) => ThreadSafeValue::Char(*c),
            GcValue::String(ptr) => {
                let s = heap.get_string(*ptr)?;
                ThreadSafeValue::String(s.data.clone())
            }
            GcValue::List(ptr) => {
                let list = heap.get_list(*ptr)?;
                let items: Option<Vec<_>> = list.items.iter()
                    .map(|v| ThreadSafeValue::from_gc_value(v, heap))
                    .collect();
                ThreadSafeValue::List(items?)
            }
            GcValue::Tuple(ptr) => {
                let tuple = heap.get_tuple(*ptr)?;
                let items: Option<Vec<_>> = tuple.items.iter()
                    .map(|v| ThreadSafeValue::from_gc_value(v, heap))
                    .collect();
                ThreadSafeValue::Tuple(items?)
            }
            GcValue::Record(ptr) => {
                let rec = heap.get_record(*ptr)?;
                let fields: Option<Vec<_>> = rec.fields.iter()
                    .map(|v| ThreadSafeValue::from_gc_value(v, heap))
                    .collect();
                ThreadSafeValue::Record {
                    type_name: rec.type_name.clone(),
                    field_names: rec.field_names.clone(),
                    fields: fields?,
                    mutable_fields: rec.mutable_fields.clone(),
                }
            }
            // Functions and closures cannot be sent between threads
            _ => return None,
        })
    }

    /// Convert back to GcValue, allocating on the given heap.
    fn to_gc_value(&self, heap: &mut Heap) -> GcValue {
        match self {
            ThreadSafeValue::Unit => GcValue::Unit,
            ThreadSafeValue::Bool(b) => GcValue::Bool(*b),
            ThreadSafeValue::Int64(i) => GcValue::Int64(*i),
            ThreadSafeValue::Float64(f) => GcValue::Float64(*f),
            ThreadSafeValue::Pid(p) => GcValue::Pid(*p),
            ThreadSafeValue::Char(c) => GcValue::Char(*c),
            ThreadSafeValue::String(s) => {
                let ptr = heap.alloc_string(s.clone());
                GcValue::String(ptr)
            }
            ThreadSafeValue::List(items) => {
                let gc_items: Vec<GcValue> = items.iter()
                    .map(|v| v.to_gc_value(heap))
                    .collect();
                let ptr = heap.alloc_list(gc_items);
                GcValue::List(ptr)
            }
            ThreadSafeValue::Tuple(items) => {
                let gc_items: Vec<GcValue> = items.iter()
                    .map(|v| v.to_gc_value(heap))
                    .collect();
                let ptr = heap.alloc_tuple(gc_items);
                GcValue::Tuple(ptr)
            }
            ThreadSafeValue::Record { type_name, field_names, fields, mutable_fields } => {
                let gc_fields: Vec<GcValue> = fields.iter()
                    .map(|v| v.to_gc_value(heap))
                    .collect();
                let ptr = heap.alloc_record(
                    type_name.clone(),
                    field_names.clone(),
                    gc_fields,
                    mutable_fields.clone(),
                );
                GcValue::Record(ptr)
            }
        }
    }
}

/// Message sent between threads.
#[derive(Debug)]
struct CrossThreadMessage {
    /// Target process
    target_pid: Pid,
    /// Message payload (thread-safe copy)
    payload: ThreadSafeValue,
}

/// Shared state across all threads (read-only after init).
pub struct SharedState {
    /// Global functions (read-only after startup)
    pub functions: HashMap<String, Arc<FunctionValue>>,
    /// Function list for indexed calls
    pub function_list: Vec<Arc<FunctionValue>>,
    /// Native functions (read-only after startup)
    pub natives: HashMap<String, Arc<GcNativeFn>>,
    /// Type definitions (read-only after startup)
    pub types: HashMap<String, Arc<TypeValue>>,
    /// JIT-compiled integer functions (func_index → native fn)
    pub jit_int_functions: HashMap<u16, JitIntFn>,
    /// JIT-compiled loop array functions (func_index → native fn)
    pub jit_loop_array_functions: HashMap<u16, JitLoopArrayFn>,
    /// Shutdown signal
    pub shutdown: AtomicBool,
}

/// Configuration for the parallel VM.
#[derive(Clone)]
pub struct ParallelConfig {
    /// Number of worker threads (0 = auto-detect CPU count)
    pub num_threads: usize,
    /// Reductions per time slice before yielding
    pub reductions_per_slice: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // auto-detect
            reductions_per_slice: 2000,
        }
    }
}

/// The parallel VM entry point.
pub struct ParallelVM {
    /// Shared state (Arc for thread sharing)
    shared: Arc<SharedState>,
    /// Channel senders to each thread (for cross-thread messages)
    thread_senders: Vec<Sender<CrossThreadMessage>>,
    /// Thread handles
    threads: Vec<JoinHandle<ThreadResult>>,
    /// Number of threads
    num_threads: usize,
    /// Next thread for round-robin initial spawn
    next_thread: AtomicU64,
    /// Configuration
    config: ParallelConfig,
}

/// Simple value that can be sent between threads.
#[derive(Debug, Clone)]
pub enum SendableValue {
    Unit,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Pid(u64),
    String(String),
    Error(String),
}

impl SendableValue {
    fn from_gc_value(value: &GcValue, heap: &Heap) -> Self {
        match value {
            GcValue::Unit => SendableValue::Unit,
            GcValue::Bool(b) => SendableValue::Bool(*b),
            GcValue::Int64(i) => SendableValue::Int64(*i),
            GcValue::Float64(f) => SendableValue::Float64(*f),
            GcValue::Pid(p) => SendableValue::Pid(*p),
            GcValue::String(ptr) => {
                if let Some(s) = heap.get_string(*ptr) {
                    SendableValue::String(s.data.clone())
                } else {
                    SendableValue::String("<string>".to_string())
                }
            }
            // For other values, use their display representation
            _ => SendableValue::String(heap.display_value(value)),
        }
    }

    /// Convert to GcValue for output (lossy for strings).
    pub fn to_gc_value(&self) -> GcValue {
        match self {
            SendableValue::Unit => GcValue::Unit,
            SendableValue::Bool(b) => GcValue::Bool(*b),
            SendableValue::Int64(i) => GcValue::Int64(*i),
            SendableValue::Float64(f) => GcValue::Float64(*f),
            SendableValue::Pid(p) => GcValue::Pid(*p),
            // Use a placeholder - the CLI should use display() instead
            SendableValue::String(s) => {
                // Create a simple heap for the string
                let mut heap = Heap::new();
                let ptr = heap.alloc_string(s.clone());
                GcValue::String(ptr)
            }
            SendableValue::Error(_) => GcValue::Unit,
        }
    }

    /// Display the value as a string.
    pub fn display(&self) -> String {
        match self {
            SendableValue::Unit => "()".to_string(),
            SendableValue::Bool(b) => b.to_string(),
            SendableValue::Int64(i) => i.to_string(),
            SendableValue::Float64(f) => f.to_string(),
            SendableValue::Pid(p) => format!("<pid {}>", p),
            SendableValue::String(s) => s.clone(),
            SendableValue::Error(e) => format!("Error: {}", e),
        }
    }

    /// Check if this is Unit.
    pub fn is_unit(&self) -> bool {
        matches!(self, SendableValue::Unit)
    }
}

/// Result from a thread when it finishes.
#[derive(Debug)]
struct ThreadResult {
    thread_id: u16,
    main_result: Option<Result<SendableValue, String>>,
}

/// Per-thread worker state.
struct ThreadWorker {
    /// This thread's ID
    thread_id: u16,
    /// Local processes owned by this thread
    processes: HashMap<u64, Process>, // local_id -> Process
    /// Local run queue (ready processes)
    run_queue: VecDeque<u64>, // local_ids
    /// Next local ID for this thread
    next_local_id: u64,
    /// Inbox for messages from other threads
    inbox: Receiver<CrossThreadMessage>,
    /// Senders to other threads (for cross-thread sends)
    thread_senders: Vec<Sender<CrossThreadMessage>>,
    /// Shared state reference
    shared: Arc<SharedState>,
    /// Configuration
    config: ParallelConfig,
    /// Main process Pid (if on this thread)
    main_pid: Option<Pid>,
    /// Main process result (sendable between threads)
    main_result: Option<Result<SendableValue, String>>,
}

impl ParallelVM {
    /// Create a new parallel VM with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        let num_threads = if config.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        } else {
            config.num_threads
        };

        let shared = Arc::new(SharedState {
            functions: HashMap::new(),
            function_list: Vec::new(),
            natives: HashMap::new(),
            types: HashMap::new(),
            jit_int_functions: HashMap::new(),
            jit_loop_array_functions: HashMap::new(),
            shutdown: AtomicBool::new(false),
        });

        Self {
            shared,
            thread_senders: Vec::new(),
            threads: Vec::new(),
            num_threads,
            next_thread: AtomicU64::new(0),
            config,
        }
    }

    /// Register a function (must be called before run).
    pub fn register_function(&mut self, name: &str, func: Arc<FunctionValue>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .functions
            .insert(name.to_string(), func);
    }

    /// Set the function list for indexed calls.
    pub fn set_function_list(&mut self, functions: Vec<Arc<FunctionValue>>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot set after threads started")
            .function_list = functions;
    }

    /// Register a native function.
    pub fn register_native(&mut self, name: &str, native: Arc<GcNativeFn>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .natives
            .insert(name.to_string(), native);
    }

    /// Register a JIT-compiled integer function.
    pub fn register_jit_int_function(&mut self, func_index: u16, jit_fn: JitIntFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_int_functions
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled loop array function.
    pub fn register_jit_loop_array_function(&mut self, func_index: u16, jit_fn: JitLoopArrayFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_loop_array_functions
            .insert(func_index, jit_fn);
    }

    /// Register the default native functions (show, copy, print, println, etc.)
    pub fn register_default_natives(&mut self) {
        // Show - convert value to string
        self.register_native("show", Arc::new(GcNativeFn {
            name: "show".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                Ok(GcValue::String(heap.alloc_string(s)))
            }),
        }));

        // Copy - deep copy a value
        self.register_native("copy", Arc::new(GcNativeFn {
            name: "copy".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                Ok(heap.clone_value(&args[0]))
            }),
        }));

        // Print - print without newline
        self.register_native("print", Arc::new(GcNativeFn {
            name: "print".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                print!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // Println - print with newline
        self.register_native("println", Arc::new(GcNativeFn {
            name: "println".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                println!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));
    }

    /// Register a type.
    pub fn register_type(&mut self, name: &str, type_val: Arc<TypeValue>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .types
            .insert(name.to_string(), type_val);
    }

    /// Run the VM with the given main function.
    /// Returns the result of the main process.
    pub fn run(&mut self, main_func: Arc<FunctionValue>) -> Result<Option<SendableValue>, RuntimeError> {
        // Create channels for each thread
        let mut receivers = Vec::with_capacity(self.num_threads);
        for _ in 0..self.num_threads {
            let (tx, rx) = channel::unbounded();
            self.thread_senders.push(tx);
            receivers.push(rx);
        }

        // Spawn worker threads
        for thread_id in 0..self.num_threads {
            let thread_id = thread_id as u16;
            let inbox = receivers.remove(0);
            let thread_senders = self.thread_senders.clone();
            let shared = Arc::clone(&self.shared);
            let config = self.config.clone();

            // Thread 0 gets the main function
            let main_func_for_thread = if thread_id == 0 {
                Some(main_func.clone())
            } else {
                None
            };

            let handle = thread::spawn(move || {
                let mut worker = ThreadWorker::new(
                    thread_id,
                    inbox,
                    thread_senders,
                    shared,
                    config,
                );

                // Thread 0 spawns the main process
                if let Some(func) = main_func_for_thread {
                    let pid = worker.spawn_process(func, vec![], vec![]);
                    worker.main_pid = Some(pid);
                }

                worker.run()
            });

            self.threads.push(handle);
        }

        // Wait for all threads to finish
        let mut main_result = None;
        for handle in self.threads.drain(..) {
            match handle.join() {
                Ok(result) => {
                    if result.main_result.is_some() {
                        main_result = result.main_result;
                    }
                }
                Err(e) => {
                    return Err(RuntimeError::Panic(format!("Thread panicked: {:?}", e)));
                }
            }
        }

        match main_result {
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(e)) => Err(RuntimeError::Panic(e)),
            None => Ok(None),
        }
    }

    /// Signal all threads to shut down.
    pub fn shutdown(&self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
    }
}

impl ThreadWorker {
    fn new(
        thread_id: u16,
        inbox: Receiver<CrossThreadMessage>,
        thread_senders: Vec<Sender<CrossThreadMessage>>,
        shared: Arc<SharedState>,
        config: ParallelConfig,
    ) -> Self {
        Self {
            thread_id,
            processes: HashMap::new(),
            run_queue: VecDeque::new(),
            next_local_id: 1,
            inbox,
            thread_senders,
            shared,
            config,
            main_pid: None,
            main_result: None,
        }
    }

    /// Spawn a new process on this thread.
    fn spawn_process(
        &mut self,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
    ) -> Pid {
        let local_id = self.next_local_id;
        self.next_local_id += 1;
        let pid = encode_pid(self.thread_id, local_id);

        // Create process with lightweight heap
        let mut process = Process::with_gc_config(pid, GcConfig::lightweight());

        // Set up initial call frame
        let reg_count = func.code.register_count;
        let mut registers = vec![GcValue::Unit; reg_count];

        // Copy arguments to registers
        for (i, arg) in args.into_iter().enumerate() {
            if i < reg_count {
                registers[i] = arg;
            }
        }

        let frame = crate::process::CallFrame {
            function: func,
            ip: 0,
            registers,
            captures,
            return_reg: None,
        };
        process.frames.push(frame);

        self.processes.insert(local_id, process);
        self.run_queue.push_back(local_id);

        pid
    }

    /// Main execution loop for this thread.
    fn run(mut self) -> ThreadResult {
        loop {
            // Check for shutdown
            if self.shared.shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Drain inbox (deliver cross-thread messages)
            self.drain_inbox();

            // Get next process to run
            let local_id = match self.run_queue.pop_front() {
                Some(id) => id,
                None => {
                    // No work - check if we should exit
                    if self.processes.is_empty() {
                        break;
                    }
                    // All processes waiting - brief sleep then retry
                    std::thread::sleep(std::time::Duration::from_micros(100));
                    continue;
                }
            };

            // Execute process for a time slice
            match self.execute_slice(local_id) {
                Ok(SliceResult::Continue) => {
                    // Process yielded, re-queue
                    self.run_queue.push_back(local_id);
                }
                Ok(SliceResult::Waiting) => {
                    // Process waiting for message, don't re-queue
                    // It will be re-queued when a message arrives
                }
                Ok(SliceResult::Finished(value)) => {
                    // Process finished
                    let pid = encode_pid(self.thread_id, local_id);
                    if Some(pid) == self.main_pid {
                        // Convert GcValue to SendableValue for thread-safe result
                        let proc = self.processes.get(&local_id).unwrap();
                        let sendable = SendableValue::from_gc_value(&value, &proc.heap);
                        self.main_result = Some(Ok(sendable));
                        // Signal shutdown
                        self.shared.shutdown.store(true, Ordering::SeqCst);
                    }
                    self.processes.remove(&local_id);
                }
                Err(e) => {
                    // Process error
                    let pid = encode_pid(self.thread_id, local_id);
                    if Some(pid) == self.main_pid {
                        self.main_result = Some(Err(e.to_string()));
                        self.shared.shutdown.store(true, Ordering::SeqCst);
                    }
                    self.processes.remove(&local_id);
                }
            }
        }

        ThreadResult {
            thread_id: self.thread_id,
            main_result: self.main_result,
        }
    }

    /// Drain the inbox and deliver messages to local processes.
    fn drain_inbox(&mut self) {
        loop {
            match self.inbox.try_recv() {
                Ok(msg) => {
                    let local_id = pid_local_id(msg.target_pid);
                    if let Some(process) = self.processes.get_mut(&local_id) {
                        // Convert thread-safe value to GcValue on this process's heap
                        let gc_value = msg.payload.to_gc_value(&mut process.heap);

                        // Deliver to mailbox
                        process.mailbox.push_back(gc_value);

                        // If process was waiting, re-queue it
                        if process.state == ProcessState::Waiting {
                            process.state = ProcessState::Running;
                            self.run_queue.push_back(local_id);
                        }
                    }
                    // If process doesn't exist, message is dropped (process died)
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    /// Send a message to a process (local or remote).
    fn send_message(&mut self, target: Pid, message: GcValue, sender_local_id: u64) {
        let target_thread = pid_thread_id(target);

        if target_thread == self.thread_id {
            // Local send - need to deep copy from sender's heap to target's heap
            let target_local_id = pid_local_id(target);

            // Convert to thread-safe value first (to avoid borrow conflicts)
            let sender_process = self.processes.get(&sender_local_id).unwrap();
            let safe_value = ThreadSafeValue::from_gc_value(&message, &sender_process.heap);

            // Then convert back to target's heap
            if let (Some(safe), Some(target_process)) = (safe_value, self.processes.get_mut(&target_local_id)) {
                let copied_message = safe.to_gc_value(&mut target_process.heap);
                target_process.mailbox.push_back(copied_message);

                if target_process.state == ProcessState::Waiting {
                    target_process.state = ProcessState::Running;
                    self.run_queue.push_back(target_local_id);
                }
            }
        } else {
            // Cross-thread send
            // Convert to thread-safe value
            let sender_process = self.processes.get(&sender_local_id).unwrap();
            if let Some(safe_value) = ThreadSafeValue::from_gc_value(&message, &sender_process.heap) {
                if let Some(sender) = self.thread_senders.get(target_thread as usize) {
                    let _ = sender.send(CrossThreadMessage {
                        target_pid: target,
                        payload: safe_value,
                    });
                }
            }
            // If value can't be converted (e.g., function), message is dropped
        }
    }

    /// Execute a process for one time slice.
    fn execute_slice(&mut self, local_id: u64) -> Result<SliceResult, RuntimeError> {
        let reductions = self.config.reductions_per_slice;

        // FAST PATH: Execute as many instructions as possible with ONE HashMap lookup
        // Only break out for slow-path instructions or when done
        loop {
            match self.execute_fast_loop(local_id, reductions)? {
                FastLoopResult::Continue => {
                    // Did all reductions - time to yield
                    return Ok(SliceResult::Continue);
                }
                FastLoopResult::Finished(v) => return Ok(SliceResult::Finished(v)),
                FastLoopResult::NeedSlowPath(instr) => {
                    // Fall back to slow path for complex instructions
                    let constants = {
                        let proc = self.processes.get(&local_id).unwrap();
                        proc.frames.last().unwrap().function.code.constants.clone()
                    };
                    match self.execute_instruction(local_id, &instr, &constants)? {
                        StepResult::Continue => continue, // Back to fast loop
                        StepResult::Yield => return Ok(SliceResult::Continue),
                        StepResult::Waiting => return Ok(SliceResult::Waiting),
                        StepResult::Finished(v) => return Ok(SliceResult::Finished(v)),
                    }
                }
            }
        }
    }

    /// Execute multiple instructions in a tight loop without HashMap lookups.
    /// Returns when: we've done `max_iters` instructions, hit a slow-path instruction,
    /// or the process finishes.
    fn execute_fast_loop(&mut self, local_id: u64, max_iters: usize) -> Result<FastLoopResult, RuntimeError> {
        use Instruction::*;

        let proc = self.processes.get_mut(&local_id).unwrap();

        for _ in 0..max_iters {
            let frame_len = proc.frames.len();
            if frame_len == 0 {
                return Ok(FastLoopResult::Finished(GcValue::Unit));
            }
            let frame_idx = frame_len - 1;

            // Get frame data using raw pointers like runtime.rs - avoids cloning!
            // SAFETY: frame_idx is valid (frame_len > 0 checked above)
            let (ip, code_ptr, code_len) = unsafe {
                let frame = proc.frames.get_unchecked(frame_idx);
                (frame.ip, frame.function.code.code.as_ptr(), frame.function.code.code.len())
            };

            if ip >= code_len {
                return Ok(FastLoopResult::Finished(GcValue::Unit));
            }

            // SAFETY: ip < code_len checked above
            let instr = unsafe { &*code_ptr.add(ip) };

            match instr {
                AddInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(a.wrapping_add(b));
                }
                SubInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(a.wrapping_sub(b));
                }
                MulInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(a.wrapping_mul(b));
                }
                LtInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(a < b);
                }
                LeInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(a <= b);
                }
                GtInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(a > b);
                }
                GeInt(dst, l, r) => {
                    proc.frames[frame_idx].ip += 1;
                    let a = match &proc.frames[frame_idx].registers[*l as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let b = match &proc.frames[frame_idx].registers[*r as usize] {
                        GcValue::Int64(v) => *v,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(a >= b);
                }
                Move(dst, src) => {
                    proc.frames[frame_idx].ip += 1;
                    let val = proc.frames[frame_idx].registers[*src as usize].clone();
                    proc.frames[frame_idx].registers[*dst as usize] = val;
                }
                Jump(offset) => {
                    proc.frames[frame_idx].ip = (ip as isize + 1 + *offset as isize) as usize;
                }
                JumpIfFalse(cond, offset) => {
                    if let GcValue::Bool(false) = &proc.frames[frame_idx].registers[*cond as usize] {
                        proc.frames[frame_idx].ip = (ip as isize + 1 + *offset as isize) as usize;
                    } else {
                        proc.frames[frame_idx].ip += 1;
                    }
                }
                JumpIfTrue(cond, offset) => {
                    if let GcValue::Bool(true) = &proc.frames[frame_idx].registers[*cond as usize] {
                        proc.frames[frame_idx].ip = (ip as isize + 1 + *offset as isize) as usize;
                    } else {
                        proc.frames[frame_idx].ip += 1;
                    }
                }
                Index(dst, coll, idx) => {
                    proc.frames[frame_idx].ip += 1;
                    let idx_val = match &proc.frames[frame_idx].registers[*idx as usize] {
                        GcValue::Int64(i) => *i as usize,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let value = match &proc.frames[frame_idx].registers[*coll as usize] {
                        GcValue::Int64Array(ptr) => {
                            let array = proc.heap.get_int64_array(*ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid int64 array".to_string()))?;
                            GcValue::Int64(*array.items.get(idx_val)
                                .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?)
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = value;
                }
                IndexSet(coll, idx, val) => {
                    proc.frames[frame_idx].ip += 1;
                    let idx_val = match &proc.frames[frame_idx].registers[*idx as usize] {
                        GcValue::Int64(i) => *i as usize,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let (ptr, new_value) = match &proc.frames[frame_idx].registers[*coll as usize] {
                        GcValue::Int64Array(ptr) => {
                            let new_value = match &proc.frames[frame_idx].registers[*val as usize] {
                                GcValue::Int64(v) => *v,
                                _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                            };
                            (*ptr, new_value)
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let array = proc.heap.get_int64_array_mut(ptr)
                        .ok_or_else(|| RuntimeError::Panic("Invalid int64 array".to_string()))?;
                    if idx_val >= array.items.len() {
                        return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                    }
                    array.items[idx_val] = new_value;
                }
                TailCallSelf(args) => {
                    let arg_values: Vec<GcValue> = args.iter()
                        .map(|r| proc.frames[frame_idx].registers[*r as usize].clone())
                        .collect();
                    let reg_count = proc.frames[frame_idx].function.code.register_count;
                    proc.frames[frame_idx].ip = 0;
                    proc.frames[frame_idx].registers.clear();
                    proc.frames[frame_idx].registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            proc.frames[frame_idx].registers[i] = arg;
                        }
                    }
                }
                Return(src) => {
                    let ret_val = proc.frames[frame_idx].registers[*src as usize].clone();
                    let return_reg = proc.frames[frame_idx].return_reg;
                    proc.frames.pop();

                    if proc.frames.is_empty() {
                        return Ok(FastLoopResult::Finished(ret_val));
                    }

                    if let Some(dst) = return_reg {
                        let parent_frame = proc.frames.last_mut().unwrap();
                        parent_frame.registers[dst as usize] = ret_val;
                    }
                }
                LoadUnit(dst) => {
                    proc.frames[frame_idx].ip += 1;
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Unit;
                }
                LoadTrue(dst) => {
                    proc.frames[frame_idx].ip += 1;
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(true);
                }
                LoadFalse(dst) => {
                    proc.frames[frame_idx].ip += 1;
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(false);
                }
                Length(dst, src) => {
                    proc.frames[frame_idx].ip += 1;
                    let len = match &proc.frames[frame_idx].registers[*src as usize] {
                        GcValue::Int64Array(ptr) => {
                            proc.heap.get_int64_array(*ptr)
                                .map(|a| a.items.len() as i64)
                                .unwrap_or(0)
                        }
                        GcValue::List(ptr) => {
                            proc.heap.get_list(*ptr)
                                .map(|l| l.items.len() as i64)
                                .unwrap_or(0)
                        }
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr)
                                .map(|s| s.data.chars().count() as i64)
                                .unwrap_or(0)
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(len);
                }
                _ => {
                    // Slow path instruction - need to release proc and call method on self
                    proc.frames[frame_idx].ip += 1;
                    return Ok(FastLoopResult::NeedSlowPath(instr.clone()));
                }
            }
        }

        Ok(FastLoopResult::Continue)
    }

    /// Execute one instruction.
    fn execute_one(&mut self, local_id: u64) -> Result<StepResult, RuntimeError> {
        use Instruction::*;

        // Single HashMap lookup - get instruction and increment IP together
        let proc = self.processes.get_mut(&local_id).unwrap();

        if proc.frames.is_empty() {
            return Ok(StepResult::Finished(GcValue::Unit));
        }

        let frame_idx = proc.frames.len() - 1;
        {
            let frame = &proc.frames[frame_idx];
            if frame.ip >= frame.function.code.code.len() {
                return Ok(StepResult::Finished(GcValue::Unit));
            }
        }

        let instr = proc.frames[frame_idx].function.code.code[proc.frames[frame_idx].ip].clone();
        proc.frames[frame_idx].ip += 1;

        // ULTRA-FAST PATH: Handle the most critical instructions inline
        // These don't need constants and avoid ALL extra lookups
        macro_rules! fast_reg {
            ($r:expr) => {
                &proc.frames[frame_idx].registers[$r as usize]
            };
        }

        macro_rules! fast_set {
            ($r:expr, $v:expr) => {
                proc.frames[frame_idx].registers[$r as usize] = $v
            };
        }

        match &instr {
            AddInt(dst, l, r) => {
                let result = match (fast_reg!(*l), fast_reg!(*r)) {
                    (GcValue::Int64(a), GcValue::Int64(b)) => GcValue::Int64(a.wrapping_add(*b)),
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                fast_set!(*dst, result);
                return Ok(StepResult::Continue);
            }
            SubInt(dst, l, r) => {
                let result = match (fast_reg!(*l), fast_reg!(*r)) {
                    (GcValue::Int64(a), GcValue::Int64(b)) => GcValue::Int64(a.wrapping_sub(*b)),
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                fast_set!(*dst, result);
                return Ok(StepResult::Continue);
            }
            MulInt(dst, l, r) => {
                let result = match (fast_reg!(*l), fast_reg!(*r)) {
                    (GcValue::Int64(a), GcValue::Int64(b)) => GcValue::Int64(a.wrapping_mul(*b)),
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                fast_set!(*dst, result);
                return Ok(StepResult::Continue);
            }
            LtInt(dst, l, r) => {
                let result = match (fast_reg!(*l), fast_reg!(*r)) {
                    (GcValue::Int64(a), GcValue::Int64(b)) => a < b,
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                fast_set!(*dst, GcValue::Bool(result));
                return Ok(StepResult::Continue);
            }
            GeInt(dst, l, r) => {
                let result = match (fast_reg!(*l), fast_reg!(*r)) {
                    (GcValue::Int64(a), GcValue::Int64(b)) => a >= b,
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                fast_set!(*dst, GcValue::Bool(result));
                return Ok(StepResult::Continue);
            }
            Move(dst, src) => {
                let val = fast_reg!(*src).clone();
                fast_set!(*dst, val);
                return Ok(StepResult::Continue);
            }
            Jump(offset) => {
                proc.frames[frame_idx].ip = (proc.frames[frame_idx].ip as isize + *offset as isize) as usize;
                return Ok(StepResult::Continue);
            }
            JumpIfFalse(cond, offset) => {
                if let GcValue::Bool(false) = fast_reg!(*cond) {
                    proc.frames[frame_idx].ip = (proc.frames[frame_idx].ip as isize + *offset as isize) as usize;
                }
                return Ok(StepResult::Continue);
            }
            JumpIfTrue(cond, offset) => {
                if let GcValue::Bool(true) = fast_reg!(*cond) {
                    proc.frames[frame_idx].ip = (proc.frames[frame_idx].ip as isize + *offset as isize) as usize;
                }
                return Ok(StepResult::Continue);
            }
            Index(dst, coll, idx) => {
                let idx_val = match fast_reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                let value = match fast_reg!(*coll) {
                    GcValue::Int64Array(ptr) => {
                        let array = proc.heap.get_int64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                        GcValue::Int64(*array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?)
                    }
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                fast_set!(*dst, value);
                return Ok(StepResult::Continue);
            }
            IndexSet(coll, idx, val) => {
                let idx_val = match fast_reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                };
                match fast_reg!(*coll) {
                    GcValue::Int64Array(ptr) => {
                        let new_value = match fast_reg!(*val) {
                            GcValue::Int64(v) => *v,
                            _ => {
                                let constants = proc.frames[frame_idx].function.code.constants.clone();
                                drop(proc);
                                return self.execute_instruction(local_id, &instr, &constants);
                            }
                        };
                        let ptr = *ptr;
                        let array = proc.heap.get_int64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    _ => {
                        let constants = proc.frames[frame_idx].function.code.constants.clone();
                        drop(proc);
                        return self.execute_instruction(local_id, &instr, &constants);
                    }
                }
                return Ok(StepResult::Continue);
            }
            TailCallSelf(args) => {
                let arg_values: Vec<GcValue> = args.iter().map(|r| proc.frames[frame_idx].registers[*r as usize].clone()).collect();
                let frame = &mut proc.frames[frame_idx];
                let reg_count = frame.function.code.register_count;
                frame.ip = 0;
                frame.registers.clear();
                frame.registers.resize(reg_count, GcValue::Unit);
                for (i, arg) in arg_values.into_iter().enumerate() {
                    if i < reg_count {
                        frame.registers[i] = arg;
                    }
                }
                return Ok(StepResult::Continue);
            }
            Return(src) => {
                let ret_val = fast_reg!(*src).clone();
                let return_reg = proc.frames[frame_idx].return_reg;
                proc.frames.pop();

                if proc.frames.is_empty() {
                    return Ok(StepResult::Finished(ret_val));
                }

                if let Some(dst) = return_reg {
                    let parent_frame = proc.frames.last_mut().unwrap();
                    parent_frame.registers[dst as usize] = ret_val;
                }
                return Ok(StepResult::Continue);
            }
            _ => {}
        }

        // Slow path - clone constants
        let constants = proc.frames[frame_idx].function.code.constants.clone();
        drop(proc);

        // Execute instruction
        self.execute_instruction(local_id, &instr, &constants)
    }

    /// Execute a single instruction.
    fn execute_instruction(
        &mut self,
        local_id: u64,
        instr: &Instruction,
        constants: &[Value],
    ) -> Result<StepResult, RuntimeError> {
        use Instruction::*;

        // FAST PATH: Handle hot-path instructions with minimal overhead
        // These instructions only need register access, no &mut self methods
        {
            let proc = self.processes.get_mut(&local_id).unwrap();
            let frame_idx = proc.frames.len() - 1;

            macro_rules! fast_reg {
                ($r:expr) => {
                    &proc.frames[frame_idx].registers[$r as usize]
                };
            }

            macro_rules! fast_set {
                ($r:expr, $v:expr) => {
                    proc.frames[frame_idx].registers[$r as usize] = $v
                };
            }

            // Handle arithmetic and comparison instructions inline
            match instr {
                AddInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => GcValue::Int64(a.wrapping_add(*b)),
                        (GcValue::Int32(a), GcValue::Int32(b)) => GcValue::Int32(a.wrapping_add(*b)),
                        (GcValue::Int16(a), GcValue::Int16(b)) => GcValue::Int16(a.wrapping_add(*b)),
                        (GcValue::Int8(a), GcValue::Int8(b)) => GcValue::Int8(a.wrapping_add(*b)),
                        (GcValue::UInt64(a), GcValue::UInt64(b)) => GcValue::UInt64(a.wrapping_add(*b)),
                        (GcValue::UInt32(a), GcValue::UInt32(b)) => GcValue::UInt32(a.wrapping_add(*b)),
                        (GcValue::UInt16(a), GcValue::UInt16(b)) => GcValue::UInt16(a.wrapping_add(*b)),
                        (GcValue::UInt8(a), GcValue::UInt8(b)) => GcValue::UInt8(a.wrapping_add(*b)),
                        (GcValue::BigInt(a), GcValue::BigInt(b)) => {
                            let a_val = proc.heap.get_bigint(*a).unwrap();
                            let b_val = proc.heap.get_bigint(*b).unwrap();
                            let result = &a_val.value + &b_val.value;
                            GcValue::BigInt(proc.heap.alloc_bigint(result))
                        }
                        (GcValue::Decimal(a), GcValue::Decimal(b)) => GcValue::Decimal(*a + *b),
                        _ => return Err(RuntimeError::TypeError { expected: "numeric".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                SubInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => GcValue::Int64(a.wrapping_sub(*b)),
                        (GcValue::Int32(a), GcValue::Int32(b)) => GcValue::Int32(a.wrapping_sub(*b)),
                        (GcValue::Int16(a), GcValue::Int16(b)) => GcValue::Int16(a.wrapping_sub(*b)),
                        (GcValue::Int8(a), GcValue::Int8(b)) => GcValue::Int8(a.wrapping_sub(*b)),
                        (GcValue::UInt64(a), GcValue::UInt64(b)) => GcValue::UInt64(a.wrapping_sub(*b)),
                        (GcValue::UInt32(a), GcValue::UInt32(b)) => GcValue::UInt32(a.wrapping_sub(*b)),
                        (GcValue::UInt16(a), GcValue::UInt16(b)) => GcValue::UInt16(a.wrapping_sub(*b)),
                        (GcValue::UInt8(a), GcValue::UInt8(b)) => GcValue::UInt8(a.wrapping_sub(*b)),
                        (GcValue::BigInt(a), GcValue::BigInt(b)) => {
                            let a_val = proc.heap.get_bigint(*a).unwrap();
                            let b_val = proc.heap.get_bigint(*b).unwrap();
                            let result = &a_val.value - &b_val.value;
                            GcValue::BigInt(proc.heap.alloc_bigint(result))
                        }
                        (GcValue::Decimal(a), GcValue::Decimal(b)) => GcValue::Decimal(*a - *b),
                        _ => return Err(RuntimeError::TypeError { expected: "numeric".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                MulInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => GcValue::Int64(a.wrapping_mul(*b)),
                        (GcValue::Int32(a), GcValue::Int32(b)) => GcValue::Int32(a.wrapping_mul(*b)),
                        (GcValue::Int16(a), GcValue::Int16(b)) => GcValue::Int16(a.wrapping_mul(*b)),
                        (GcValue::Int8(a), GcValue::Int8(b)) => GcValue::Int8(a.wrapping_mul(*b)),
                        (GcValue::UInt64(a), GcValue::UInt64(b)) => GcValue::UInt64(a.wrapping_mul(*b)),
                        (GcValue::UInt32(a), GcValue::UInt32(b)) => GcValue::UInt32(a.wrapping_mul(*b)),
                        (GcValue::UInt16(a), GcValue::UInt16(b)) => GcValue::UInt16(a.wrapping_mul(*b)),
                        (GcValue::UInt8(a), GcValue::UInt8(b)) => GcValue::UInt8(a.wrapping_mul(*b)),
                        (GcValue::BigInt(a), GcValue::BigInt(b)) => {
                            let a_val = proc.heap.get_bigint(*a).unwrap();
                            let b_val = proc.heap.get_bigint(*b).unwrap();
                            let result = &a_val.value * &b_val.value;
                            GcValue::BigInt(proc.heap.alloc_bigint(result))
                        }
                        (GcValue::Decimal(a), GcValue::Decimal(b)) => GcValue::Decimal(*a * *b),
                        _ => return Err(RuntimeError::TypeError { expected: "numeric".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                AddFloat(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a + b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a + b),
                        _ => return Err(RuntimeError::TypeError { expected: "float".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                SubFloat(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a - b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a - b),
                        _ => return Err(RuntimeError::TypeError { expected: "float".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                MulFloat(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a * b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a * b),
                        _ => return Err(RuntimeError::TypeError { expected: "float".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                DivFloat(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a / b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a / b),
                        _ => return Err(RuntimeError::TypeError { expected: "float".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                LtInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a < b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a < b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                LeInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a <= b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a <= b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                GtInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a > b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a > b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                GeInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a >= b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a >= b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                Move(dst, src) => {
                    let val = fast_reg!(*src).clone();
                    fast_set!(*dst, val);
                    return Ok(StepResult::Continue);
                }
                LoadUnit(dst) => {
                    fast_set!(*dst, GcValue::Unit);
                    return Ok(StepResult::Continue);
                }
                LoadTrue(dst) => {
                    fast_set!(*dst, GcValue::Bool(true));
                    return Ok(StepResult::Continue);
                }
                LoadFalse(dst) => {
                    fast_set!(*dst, GcValue::Bool(false));
                    return Ok(StepResult::Continue);
                }
                Index(dst, coll, idx) => {
                    let idx_val = match fast_reg!(*idx) {
                        GcValue::Int64(i) => *i as usize,
                        _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                    };
                    let value = match fast_reg!(*coll) {
                        GcValue::List(ptr) => {
                            let list = proc.heap.get_list(*ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid list reference".to_string()))?;
                            list.items.get(idx_val).cloned()
                                .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                        }
                        GcValue::Tuple(ptr) => {
                            let tuple = proc.heap.get_tuple(*ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".to_string()))?;
                            tuple.items.get(idx_val).cloned()
                                .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                        }
                        GcValue::Int64Array(ptr) => {
                            let array = proc.heap.get_int64_array(*ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                            let val = *array.items.get(idx_val)
                                .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                            GcValue::Int64(val)
                        }
                        GcValue::Float64Array(ptr) => {
                            let array = proc.heap.get_float64_array(*ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".to_string()))?;
                            let val = *array.items.get(idx_val)
                                .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                            GcValue::Float64(val)
                        }
                        _ => return Err(RuntimeError::Panic("Index expects list, tuple, or array".to_string())),
                    };
                    fast_set!(*dst, value);
                    return Ok(StepResult::Continue);
                }
                IndexSet(coll, idx, val) => {
                    let idx_val = match fast_reg!(*idx) {
                        GcValue::Int64(i) => *i as usize,
                        _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                    };
                    let coll_val = fast_reg!(*coll).clone();
                    match coll_val {
                        GcValue::Int64Array(ptr) => {
                            let new_value = match fast_reg!(*val) {
                                GcValue::Int64(v) => *v,
                                _ => return Err(RuntimeError::Panic("Int64Array expects Int64 value".to_string())),
                            };
                            let array = proc.heap.get_int64_array_mut(ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                            if idx_val >= array.items.len() {
                                return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                            }
                            array.items[idx_val] = new_value;
                        }
                        GcValue::Float64Array(ptr) => {
                            let new_value = match fast_reg!(*val) {
                                GcValue::Float64(v) => *v,
                                _ => return Err(RuntimeError::Panic("Float64Array expects Float64 value".to_string())),
                            };
                            let array = proc.heap.get_float64_array_mut(ptr)
                                .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".to_string()))?;
                            if idx_val >= array.items.len() {
                                return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                            }
                            array.items[idx_val] = new_value;
                        }
                        _ => return Err(RuntimeError::Panic("IndexSet expects array".to_string())),
                    }
                    return Ok(StepResult::Continue);
                }
                TailCallSelf(args) => {
                    // Self tail-recursion: reuse current frame
                    let arg_values: Vec<GcValue> = args.iter().map(|r| proc.frames[frame_idx].registers[*r as usize].clone()).collect();
                    let frame = &mut proc.frames[frame_idx];
                    let reg_count = frame.function.code.register_count;
                    frame.ip = 0;
                    frame.registers.clear();
                    frame.registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            frame.registers[i] = arg;
                        }
                    }
                    return Ok(StepResult::Continue);
                }
                Jump(offset) => {
                    proc.frames[frame_idx].ip = (proc.frames[frame_idx].ip as isize + *offset as isize) as usize;
                    return Ok(StepResult::Continue);
                }
                JumpIfFalse(cond, offset) => {
                    if let GcValue::Bool(false) = fast_reg!(*cond) {
                        proc.frames[frame_idx].ip = (proc.frames[frame_idx].ip as isize + *offset as isize) as usize;
                    }
                    return Ok(StepResult::Continue);
                }
                JumpIfTrue(cond, offset) => {
                    if let GcValue::Bool(true) = fast_reg!(*cond) {
                        proc.frames[frame_idx].ip = (proc.frames[frame_idx].ip as isize + *offset as isize) as usize;
                    }
                    return Ok(StepResult::Continue);
                }
                Return(src) => {
                    let ret_val = fast_reg!(*src).clone();
                    let return_reg = proc.frames[frame_idx].return_reg;
                    proc.frames.pop();

                    if proc.frames.is_empty() {
                        // Main process finished
                        return Ok(StepResult::Finished(ret_val));
                    }

                    // Set return value in parent frame
                    if let Some(dst) = return_reg {
                        let parent_frame = proc.frames.last_mut().unwrap();
                        parent_frame.registers[dst as usize] = ret_val;
                    }
                    return Ok(StepResult::Continue);
                }
                _ => {} // Fall through to slow path
            }
        }

        // SLOW PATH: Instructions that need &mut self or call methods
        // Re-acquire process reference with macros that do HashMap lookup
        macro_rules! reg {
            ($r:expr) => {{
                let proc = self.processes.get(&local_id).unwrap();
                let frame = proc.frames.last().unwrap();
                &frame.registers[$r as usize]
            }};
        }

        macro_rules! set_reg {
            ($r:expr, $v:expr) => {{
                let proc = self.processes.get_mut(&local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                frame.registers[$r as usize] = $v;
            }};
        }

        match instr {
            // === Constants ===
            LoadConst(dst, idx) => {
                let value = self.load_constant(&constants[*idx as usize], local_id);
                set_reg!(*dst, value);
            }

            LoadUnit(dst) => {
                set_reg!(*dst, GcValue::Unit);
            }

            LoadTrue(dst) => {
                set_reg!(*dst, GcValue::Bool(true));
            }

            LoadFalse(dst) => {
                set_reg!(*dst, GcValue::Bool(false));
            }

            // === Register Operations ===
            Move(dst, src) => {
                let val = reg!(*src).clone();
                set_reg!(*dst, val);
            }

            // === Arithmetic (polymorphic for all numeric types) ===
            AddInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
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
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let bx = proc.heap.get_bigint(*px).unwrap().value.clone();
                        let by = proc.heap.get_bigint(*py).unwrap().value.clone();
                        let result_ptr = proc.heap.alloc_bigint(&bx + &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x + *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(*dst, result);
            }

            SubInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
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
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let bx = proc.heap.get_bigint(*px).unwrap().value.clone();
                        let by = proc.heap.get_bigint(*py).unwrap().value.clone();
                        let result_ptr = proc.heap.alloc_bigint(&bx - &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x - *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(*dst, result);
            }

            MulInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
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
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let bx = proc.heap.get_bigint(*px).unwrap().value.clone();
                        let by = proc.heap.get_bigint(*py).unwrap().value.clone();
                        let result_ptr = proc.heap.alloc_bigint(&bx * &by);
                        GcValue::BigInt(result_ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x * *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(*dst, result);
            }

            MulFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => {
                        set_reg!(*dst, GcValue::Float64(x * y));
                    }
                    (GcValue::Float32(x), GcValue::Float32(y)) => {
                        set_reg!(*dst, GcValue::Float32(x * y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            AddFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => {
                        set_reg!(*dst, GcValue::Float64(x + y));
                    }
                    (GcValue::Float32(x), GcValue::Float32(y)) => {
                        set_reg!(*dst, GcValue::Float32(x + y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            SubFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => {
                        set_reg!(*dst, GcValue::Float64(x - y));
                    }
                    (GcValue::Float32(x), GcValue::Float32(y)) => {
                        set_reg!(*dst, GcValue::Float32(x - y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            DivFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => {
                        if *y == 0.0 { return Err(RuntimeError::DivisionByZero); }
                        set_reg!(*dst, GcValue::Float64(x / y));
                    }
                    (GcValue::Float32(x), GcValue::Float32(y)) => {
                        if *y == 0.0 { return Err(RuntimeError::DivisionByZero); }
                        set_reg!(*dst, GcValue::Float32(x / y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            DivInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => {
                        if *y == 0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Int64(x / y)
                    }
                    (GcValue::Float64(x), GcValue::Float64(y)) => {
                        if *y == 0.0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Float64(x / y)
                    }
                    (GcValue::Float32(x), GcValue::Float32(y)) => {
                        if *y == 0.0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Float32(x / y)
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: format!("{:?}", va),
                    }),
                };
                set_reg!(*dst, result);
            }

            // === Comparisons ===
            Eq(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let proc = self.processes.get(&local_id).unwrap();
                let result = proc.heap.gc_values_equal(&va, &vb);
                set_reg!(*dst, GcValue::Bool(result));
            }

            EqInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => x == y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            LtInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => x < y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            LeInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => x <= y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            GtInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => x > y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            GeInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => x >= y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            LtFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => x < y,
                    (GcValue::Float32(x), GcValue::Float32(y)) => x < y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            LeFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => x <= y,
                    (GcValue::Float32(x), GcValue::Float32(y)) => x <= y,
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            // === Builtin math ===
            AbsInt(dst, src) => {
                let val = match reg!(*src) {
                    GcValue::Int64(i) => *i,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, GcValue::Int64(val.abs()));
            }

            AbsFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.abs()),
                    GcValue::Float32(f) => GcValue::Float32(f.abs()),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            SqrtFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.sqrt()),
                    GcValue::Float32(f) => GcValue::Float32(f.sqrt()),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            PowFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x.powf(*y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x.powf(*y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            // === Type Conversions ===
            ToInt32(dst, src) => {
                let val = reg!(*src).clone();
                let result = match val {
                    GcValue::Int8(v) => GcValue::Int32(v as i32),
                    GcValue::Int16(v) => GcValue::Int32(v as i32),
                    GcValue::Int32(v) => GcValue::Int32(v),
                    GcValue::Int64(v) => GcValue::Int32(v as i32),
                    GcValue::UInt8(v) => GcValue::Int32(v as i32),
                    GcValue::UInt16(v) => GcValue::Int32(v as i32),
                    GcValue::UInt32(v) => GcValue::Int32(v as i32),
                    GcValue::UInt64(v) => GcValue::Int32(v as i32),
                    GcValue::Float32(v) => GcValue::Int32(v as i32),
                    GcValue::Float64(v) => GcValue::Int32(v as i32),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            FloatToInt(dst, src) => {
                let val = reg!(*src).clone();
                let result = match val {
                    GcValue::Int8(v) => GcValue::Int64(v as i64),
                    GcValue::Int16(v) => GcValue::Int64(v as i64),
                    GcValue::Int32(v) => GcValue::Int64(v as i64),
                    GcValue::Int64(v) => GcValue::Int64(v),
                    GcValue::UInt8(v) => GcValue::Int64(v as i64),
                    GcValue::UInt16(v) => GcValue::Int64(v as i64),
                    GcValue::UInt32(v) => GcValue::Int64(v as i64),
                    GcValue::UInt64(v) => GcValue::Int64(v as i64),
                    GcValue::Float32(v) => GcValue::Int64(v as i64),
                    GcValue::Float64(v) => GcValue::Int64(v as i64),
                    GcValue::BigInt(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let bi = proc.heap.get_bigint(ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Int64(bi.value.to_i64().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            // === Control Flow ===
            Jump(offset) => {
                let proc = self.processes.get_mut(&local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                frame.ip = (frame.ip as isize + *offset as isize) as usize;
            }

            JumpIfFalse(cond, offset) => {
                let val = reg!(*cond).clone();
                if let GcValue::Bool(false) = val {
                    let proc = self.processes.get_mut(&local_id).unwrap();
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                }
            }

            JumpIfTrue(cond, offset) => {
                let val = reg!(*cond).clone();
                if let GcValue::Bool(true) = val {
                    let proc = self.processes.get_mut(&local_id).unwrap();
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                }
            }

            Return(src) => {
                let ret_val = reg!(*src).clone();
                let proc = self.processes.get_mut(&local_id).unwrap();

                // Get return_reg from current frame BEFORE popping (for tail call support)
                let return_reg = proc.frames.last().unwrap().return_reg;
                proc.frames.pop();

                if proc.frames.is_empty() {
                    return Ok(StepResult::Finished(ret_val));
                } else if let Some(ret_reg) = return_reg {
                    // Store return value in caller's return register
                    let frame = proc.frames.last_mut().unwrap();
                    frame.registers[ret_reg as usize] = ret_val;
                }
            }

            // === Function Calls ===
            Call(dst, func_reg, args) => {
                let func_val = reg!(*func_reg).clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();

                match func_val {
                    GcValue::Function(func) => {
                        self.call_function(local_id, func, arg_values, Some(*dst))?;
                    }
                    GcValue::Closure(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let closure = proc.heap.get_closure(ptr).unwrap();
                        let func = closure.function.clone();
                        let captures = closure.captures.clone();
                        drop(proc);
                        self.call_closure(local_id, func, arg_values, captures, Some(*dst))?;
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: format!("{:?}", func_val),
                    }),
                }
            }

            CallDirect(dst, func_idx, args) => {
                // Check if we have a JIT-compiled version (arity=1, int argument)
                if args.len() == 1 {
                    // First check for pure numeric JIT
                    if let Some(jit_fn) = self.shared.jit_int_functions.get(func_idx) {
                        let arg = reg!(args[0]).clone();
                        if let GcValue::Int64(n) = arg {
                            // Call JIT function directly!
                            let result = jit_fn(n);
                            set_reg!(*dst, GcValue::Int64(result));
                            return Ok(StepResult::Continue);
                        }
                    }
                    // Check for loop array JIT
                    if let Some(jit_fn) = self.shared.jit_loop_array_functions.get(func_idx) {
                        let arg = reg!(args[0]).clone();
                        if let GcValue::Int64Array(arr_ptr) = arg {
                            // Get array data from heap (mutable for IndexSet support)
                            let proc = self.processes.get_mut(&local_id).unwrap();
                            if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                let ptr = arr.items.as_mut_ptr();
                                let len = arr.items.len() as i64;
                                // Call JIT function with raw ptr and len
                                let result = jit_fn(ptr as *const i64, len);
                                set_reg!(*dst, GcValue::Int64(result));
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                }

                // Fall back to interpreted execution
                let func = self.shared.function_list.get(*func_idx as usize)
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("index {}", func_idx)))?
                    .clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();
                self.call_function(local_id, func, arg_values, Some(*dst))?;
            }

            CallSelf(dst, args) => {
                // Self-recursion: call the same function
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();
                let proc = self.processes.get(&local_id).unwrap();
                let func = proc.frames.last().unwrap().function.clone();
                self.call_function(local_id, func, arg_values, Some(*dst))?;
            }

            TailCallDirect(func_idx, args) => {
                // Check for JIT-compiled version (tail call with 1 arg)
                if args.len() == 1 {
                    // Check for pure numeric JIT
                    if let Some(jit_fn) = self.shared.jit_int_functions.get(func_idx) {
                        let arg = reg!(args[0]).clone();
                        if let GcValue::Int64(n) = arg {
                            // Call JIT function directly!
                            let result = jit_fn(n);
                            // For tail call: pop current frame, set result in parent
                            let proc = self.processes.get_mut(&local_id).unwrap();
                            let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                            proc.frames.pop();
                            if let Some(dst) = return_reg {
                                if let Some(parent) = proc.frames.last_mut() {
                                    parent.registers[dst as usize] = GcValue::Int64(result);
                                }
                            }
                            return Ok(StepResult::Continue);
                        }
                    }
                    // Check for loop array JIT
                    if let Some(jit_fn) = self.shared.jit_loop_array_functions.get(func_idx) {
                        let arg = reg!(args[0]).clone();
                        if let GcValue::Int64Array(arr_ptr) = arg {
                            let proc = self.processes.get_mut(&local_id).unwrap();
                            if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                let ptr = arr.items.as_mut_ptr();
                                let len = arr.items.len() as i64;
                                let result = jit_fn(ptr as *const i64, len);
                                // For tail call: pop current frame, set result in parent
                                let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                                proc.frames.pop();
                                if let Some(dst) = return_reg {
                                    if let Some(parent) = proc.frames.last_mut() {
                                        parent.registers[dst as usize] = GcValue::Int64(result);
                                    }
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                }

                // Fall back to interpreted tail call
                let func = self.shared.function_list.get(*func_idx as usize)
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("index {}", func_idx)))?
                    .clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();
                // Tail call: replace current frame
                self.tail_call_function(local_id, func, arg_values)?;
            }

            TailCallSelf(args) => {
                // Self tail-recursion: reuse current frame
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();
                let proc = self.processes.get_mut(&local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                let reg_count = frame.function.code.register_count;
                frame.ip = 0;
                frame.registers.clear();
                frame.registers.resize(reg_count, GcValue::Unit);
                for (i, arg) in arg_values.into_iter().enumerate() {
                    if i < reg_count {
                        frame.registers[i] = arg;
                    }
                }
            }

            TailCall(func_reg, args) => {
                let func_val = reg!(*func_reg).clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();
                match func_val {
                    GcValue::Function(func) => {
                        self.tail_call_function(local_id, func, arg_values)?;
                    }
                    GcValue::Closure(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let closure = proc.heap.get_closure(ptr).unwrap();
                        let func = closure.function.clone();
                        let captures = closure.captures.clone();
                        self.tail_call_closure(local_id, func, arg_values, captures)?;
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function or Closure".to_string(),
                        found: format!("{:?}", func_val),
                    }),
                }
            }

            CallNative(dst, name_idx, args) => {
                let name = match &constants[*name_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();

                // Check for trait overrides for "show" and "copy"
                let trait_method = if !arg_values.is_empty() && (name == "show" || name == "copy") {
                    let trait_name = if name == "show" { "Show" } else { "Copy" };
                    let proc = self.processes.get(&local_id).unwrap();
                    let type_name = arg_values[0].type_name(&proc.heap).to_string();
                    let qualified_name = format!("{}.{}.{}", type_name, trait_name, name);
                    self.shared.functions.get(&qualified_name).cloned()
                } else {
                    None
                };

                if let Some(func) = trait_method {
                    // Call the trait method instead of native
                    self.call_function(local_id, func, arg_values, Some(*dst))?;
                } else {
                    let result = self.call_native(local_id, &name, arg_values)?;
                    set_reg!(*dst, result);
                }
            }

            // === Closures ===
            MakeClosure(dst, func_idx, capture_regs) => {
                let func_val = match &constants[*func_idx as usize] {
                    Value::Function(f) => f.clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: "non-function".to_string(),
                    }),
                };
                let captures: Vec<GcValue> = capture_regs.iter().map(|r| reg!(*r).clone()).collect();
                let capture_names = func_val.param_names.clone();

                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_closure(func_val, captures, capture_names);
                set_reg!(*dst, GcValue::Closure(ptr));
            }

            GetCapture(dst, idx) => {
                let proc = self.processes.get(&local_id).unwrap();
                let frame = proc.frames.last().unwrap();
                let val = frame.captures.get(*idx as usize)
                    .cloned()
                    .unwrap_or(GcValue::Unit);
                set_reg!(*dst, val);
            }

            // === Process Operations ===
            SelfPid(dst) => {
                let pid = encode_pid(self.thread_id, local_id);
                set_reg!(*dst, GcValue::Pid(pid.0));
            }

            Spawn(dst, func_reg, args) => {
                let func_val = reg!(*func_reg).clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();

                let (func, captures) = match func_val {
                    GcValue::Function(f) => (f, vec![]),
                    GcValue::Closure(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let closure = proc.heap.get_closure(ptr).unwrap();
                        (closure.function.clone(), closure.captures.clone())
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: format!("{:?}", func_val),
                    }),
                };

                // Spawn on this thread (affinity)
                let child_pid = self.spawn_process(func, arg_values, captures);
                set_reg!(*dst, GcValue::Pid(child_pid.0));
            }

            Send(target_reg, msg_reg) => {
                let target_val = reg!(*target_reg).clone();
                let message = reg!(*msg_reg).clone();

                let target_pid = match target_val {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Pid".to_string(),
                        found: format!("{:?}", target_val),
                    }),
                };

                self.send_message(target_pid, message, local_id);
            }

            Receive => {
                let proc = self.processes.get_mut(&local_id).unwrap();
                if let Some(msg) = proc.mailbox.pop_front() {
                    // Result goes in register 0
                    let frame = proc.frames.last_mut().unwrap();
                    frame.registers[0] = msg;
                } else {
                    // No message - block
                    proc.state = ProcessState::Waiting;
                    // Decrement IP so we retry receive next time
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip -= 1;
                    return Ok(StepResult::Waiting);
                }
            }

            // === I/O ===
            Println(src) => {
                let proc = self.processes.get(&local_id).unwrap();
                let s = proc.heap.display_value(reg!(*src));
                println!("{}", s);
            }

            // === Assertions ===
            Assert(src) => {
                let val = reg!(*src).clone();
                match val {
                    GcValue::Bool(true) => {}
                    GcValue::Bool(false) => {
                        return Err(RuntimeError::AssertionFailed("Assertion failed".to_string()));
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Bool".to_string(),
                            found: format!("{:?}", val),
                        });
                    }
                }
            }

            AssertEq(a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let proc = self.processes.get(&local_id).unwrap();
                if !proc.heap.gc_values_equal(&va, &vb) {
                    let sa = proc.heap.display_value(&va);
                    let sb = proc.heap.display_value(&vb);
                    return Err(RuntimeError::Panic(format!("Assertion failed: {} != {}", sa, sb)));
                }
            }

            Nop => {}

            // === Pattern Matching ===
            TestConst(dst, value_reg, const_idx) => {
                let value = reg!(*value_reg).clone();
                let constant = &constants[*const_idx as usize];
                let proc = self.processes.get_mut(&local_id).unwrap();
                let gc_const = proc.heap.value_to_gc(constant);
                let result = proc.heap.gc_values_equal(&value, &gc_const);
                proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Bool(result);
            }

            TestNil(dst, list) => {
                let list_val = reg!(*list).clone();
                let result = match list_val {
                    GcValue::List(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        proc.heap.get_list(ptr).map(|l| l.items.is_empty()).unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            TestTag(dst, value, tag_idx) => {
                let tag = match &constants[*tag_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let value_clone = reg!(*value).clone();
                let result = match &value_clone {
                    GcValue::Variant(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        proc.heap.get_variant(*ptr).map(|v| v.constructor == tag).unwrap_or(false)
                    }
                    GcValue::Record(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        proc.heap.get_record(*ptr).map(|r| r.type_name == tag).unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            MakeVariant(dst, type_idx, ctor_idx, ref field_regs) => {
                let type_name = match &constants[*type_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::Panic("Variant type must be string".to_string())),
                };
                let constructor = match &constants[*ctor_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::Panic("Variant constructor must be string".to_string())),
                };
                let fields: Vec<GcValue> = field_regs.iter().map(|&r| reg!(r).clone()).collect();
                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_variant(type_name, constructor, fields);
                set_reg!(*dst, GcValue::Variant(ptr));
            }

            GetField(dst, record, field_idx) => {
                let field_name = match &constants[*field_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::Panic("Field name must be string".to_string())),
                };
                let rec_val = reg!(*record).clone();
                match rec_val {
                    GcValue::Record(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let rec = proc.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        let value = rec.fields[idx].clone();
                        set_reg!(*dst, value);
                    }
                    GcValue::Variant(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let var = proc.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid variant field index: {}", field_name)))?;
                        let value = var.fields.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of range", idx)))?
                            .clone();
                        set_reg!(*dst, value);
                    }
                    _ => return Err(RuntimeError::Panic("GetField expects record or variant".to_string())),
                }
            }

            GetVariantField(dst, src, idx) => {
                let src_val = reg!(*src).clone();
                let value = match src_val {
                    GcValue::Variant(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let variant = proc.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        variant.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of bounds", idx)))?
                    }
                    GcValue::Record(ptr) => {
                        let proc = self.processes.get(&local_id).unwrap();
                        let record = proc.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        record.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Record field {} out of bounds", idx)))?
                    }
                    _ => return Err(RuntimeError::Panic("GetVariantField expects variant or record".to_string())),
                };
                set_reg!(*dst, value);
            }

            SetField(record, field_idx, value) => {
                let field_name = match &constants[*field_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::Panic("Field name must be string".to_string())),
                };
                let new_value = reg!(*value).clone();
                let rec_val = reg!(*record).clone();
                match rec_val {
                    GcValue::Record(ptr) => {
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let rec = proc.heap.get_record_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        if !rec.mutable_fields[idx] {
                            return Err(RuntimeError::Panic(format!("Field {} is not mutable", field_name)));
                        }
                        rec.fields[idx] = new_value;
                    }
                    _ => return Err(RuntimeError::Panic("SetField expects record".to_string())),
                }
            }

            Length(dst, src) => {
                let val = reg!(*src).clone();
                let proc = self.processes.get(&local_id).unwrap();
                let len = match val {
                    GcValue::List(ptr) => proc.heap.get_list(ptr).map(|l| l.items.len()).unwrap_or(0),
                    GcValue::Tuple(ptr) => proc.heap.get_tuple(ptr).map(|t| t.items.len()).unwrap_or(0),
                    GcValue::Array(ptr) => proc.heap.get_array(ptr).map(|a| a.items.len()).unwrap_or(0),
                    GcValue::Int64Array(ptr) => proc.heap.get_int64_array(ptr).map(|a| a.items.len()).unwrap_or(0),
                    GcValue::Float64Array(ptr) => proc.heap.get_float64_array(ptr).map(|a| a.items.len()).unwrap_or(0),
                    GcValue::String(ptr) => proc.heap.get_string(ptr).map(|s| s.data.len()).unwrap_or(0),
                    _ => return Err(RuntimeError::Panic("Length expects collection or string".to_string())),
                };
                set_reg!(*dst, GcValue::Int64(len as i64));
            }

            // === Collections ===
            MakeList(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| reg!(r).clone()).collect();
                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_list(items);
                set_reg!(*dst, GcValue::List(ptr));
            }

            MakeTuple(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| reg!(r).clone()).collect();
                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_tuple(items);
                set_reg!(*dst, GcValue::Tuple(ptr));
            }

            GetTupleField(dst, tuple_reg, idx) => {
                let ptr = match reg!(*tuple_reg) {
                    GcValue::Tuple(ptr) => *ptr,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Tuple".to_string(),
                        found: format!("{:?}", other),
                    }),
                };
                let proc = self.processes.get(&local_id).unwrap();
                let item = proc.heap.get_tuple(ptr)
                    .and_then(|t| t.items.get(*idx as usize).cloned());
                match item {
                    Some(val) => set_reg!(*dst, val),
                    None => return Err(RuntimeError::IndexOutOfBounds {
                        index: *idx as i64,
                        length: 0,
                    }),
                }
            }

            MakeRecord(dst, type_idx, ref field_regs) => {
                let type_name = match &constants[*type_idx as usize] {
                    Value::String(s) => (**s).clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let fields: Vec<GcValue> = field_regs.iter().map(|&r| reg!(r).clone()).collect();
                let type_info = self.shared.types.get(&type_name).cloned();
                let field_names: Vec<String> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.name.clone()).collect())
                    .unwrap_or_else(|| (0..fields.len()).map(|i| format!("_{}", i)).collect());
                let mutable_fields: Vec<bool> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.mutable).collect())
                    .unwrap_or_else(|| vec![false; fields.len()]);
                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_record(type_name, field_names, fields, mutable_fields);
                set_reg!(*dst, GcValue::Record(ptr));
            }

            Cons(dst, head, tail) => {
                let head_val = reg!(*head).clone();
                let tail_val = reg!(*tail).clone();
                let proc = self.processes.get_mut(&local_id).unwrap();
                match tail_val {
                    GcValue::List(ptr) => {
                        let tail_list = proc.heap.get_list(ptr).unwrap();
                        let mut items = vec![head_val];
                        items.extend(tail_list.items.iter().cloned());
                        let new_ptr = proc.heap.alloc_list(items);
                        set_reg!(*dst, GcValue::List(new_ptr));
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: format!("{:?}", tail_val),
                        });
                    }
                }
            }

            ListIsEmpty(dst, list_reg) => {
                let list_val = reg!(*list_reg).clone();
                let proc = self.processes.get(&local_id).unwrap();
                let is_empty = match list_val {
                    GcValue::List(ptr) => {
                        let list = proc.heap.get_list(ptr).unwrap();
                        list.items.is_empty()
                    }
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(is_empty));
            }

            ListHead(dst, list_reg) => {
                let list_val = reg!(*list_reg).clone();
                let result = {
                    let proc = self.processes.get(&local_id).unwrap();
                    match &list_val {
                        GcValue::List(ptr) => {
                            let list = proc.heap.get_list(*ptr).unwrap();
                            if let Some(head) = list.items.first() {
                                Ok(head.clone())
                            } else {
                                Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 })
                            }
                        }
                        _ => Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: format!("{:?}", list_val),
                        }),
                    }
                }?;
                set_reg!(*dst, result);
            }

            ListTail(dst, list_reg) => {
                let list_val = reg!(*list_reg).clone();
                let proc = self.processes.get_mut(&local_id).unwrap();
                match list_val {
                    GcValue::List(ptr) => {
                        let list = proc.heap.get_list(ptr).unwrap();
                        if list.items.is_empty() {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                        let tail_items: Vec<GcValue> = list.items.iter().skip(1).cloned().collect();
                        let new_ptr = proc.heap.alloc_list(tail_items);
                        set_reg!(*dst, GcValue::List(new_ptr));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: format!("{:?}", list_val),
                    }),
                }
            }

            // === Typed Arrays ===
            MakeInt64Array(dst, size_reg) => {
                let size = match reg!(*size_reg) {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::Panic("Array size must be Int64".to_string())),
                };
                let items = vec![0i64; size];
                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_int64_array(items);
                set_reg!(*dst, GcValue::Int64Array(ptr));
            }

            MakeFloat64Array(dst, size_reg) => {
                let size = match reg!(*size_reg) {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::Panic("Array size must be Int64".to_string())),
                };
                let items = vec![0.0f64; size];
                let proc = self.processes.get_mut(&local_id).unwrap();
                let ptr = proc.heap.alloc_float64_array(items);
                set_reg!(*dst, GcValue::Float64Array(ptr));
            }

            Index(dst, coll, idx) => {
                let idx_val = match reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                };
                let coll_val = reg!(*coll).clone();
                let proc = self.processes.get(&local_id).unwrap();
                let value = match coll_val {
                    GcValue::List(ptr) => {
                        let list = proc.heap.get_list(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid list reference".to_string()))?;
                        list.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = proc.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".to_string()))?;
                        tuple.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Array(ptr) => {
                        let array = proc.heap.get_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".to_string()))?;
                        array.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Int64Array(ptr) => {
                        let array = proc.heap.get_int64_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Int64(val)
                    }
                    GcValue::Float64Array(ptr) => {
                        let array = proc.heap.get_float64_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".to_string()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Float64(val)
                    }
                    _ => return Err(RuntimeError::Panic("Index expects list, tuple, or array".to_string())),
                };
                set_reg!(*dst, value);
            }

            IndexSet(coll, idx, val) => {
                let idx_val = match reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                };
                let coll_val = reg!(*coll).clone();
                match coll_val {
                    GcValue::Array(ptr) => {
                        let new_value = reg!(*val).clone();
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let array = proc.heap.get_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Int64Array(ptr) => {
                        let new_value = match reg!(*val) {
                            GcValue::Int64(v) => *v,
                            _ => return Err(RuntimeError::Panic("Int64Array expects Int64 value".to_string())),
                        };
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let array = proc.heap.get_int64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Float64Array(ptr) => {
                        let new_value = match reg!(*val) {
                            GcValue::Float64(v) => *v,
                            _ => return Err(RuntimeError::Panic("Float64Array expects Float64 value".to_string())),
                        };
                        let proc = self.processes.get_mut(&local_id).unwrap();
                        let array = proc.heap.get_float64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    _ => return Err(RuntimeError::Panic("IndexSet expects array".to_string())),
                }
            }

            Decons(head_dst, tail_dst, list) => {
                let list_val = reg!(*list).clone();
                let list_ptr = match list_val {
                    GcValue::List(ptr) => ptr,
                    _ => return Err(RuntimeError::Panic("Decons expects list".to_string())),
                };
                let proc = self.processes.get_mut(&local_id).unwrap();
                let items = proc.heap.get_list(list_ptr)
                    .map(|l| l.items.clone())
                    .unwrap_or_default();
                if items.is_empty() {
                    return Err(RuntimeError::Panic("Cannot decons empty list".to_string()));
                }
                let head = items[0].clone();
                let tail = items[1..].to_vec();
                let tail_ptr = proc.heap.alloc_list(tail);
                set_reg!(*head_dst, head);
                set_reg!(*tail_dst, GcValue::List(tail_ptr));
            }

            Concat(dst, a, b) => {
                let a_val = reg!(*a).clone();
                let b_val = reg!(*b).clone();
                let (a_ptr, b_ptr) = match (a_val, b_val) {
                    (GcValue::String(a), GcValue::String(b)) => (a, b),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let proc = self.processes.get_mut(&local_id).unwrap();
                let a_str = proc.heap.get_string(a_ptr).map(|s| s.data.as_str()).unwrap_or("");
                let b_str = proc.heap.get_string(b_ptr).map(|s| s.data.as_str()).unwrap_or("");
                let result = format!("{}{}", a_str, b_str);
                let result_ptr = proc.heap.alloc_string(result);
                set_reg!(*dst, GcValue::String(result_ptr));
            }

            Throw(src) => {
                // Simple throw - just return the exception as an error
                let exception = reg!(*src).clone();
                let proc = self.processes.get(&local_id).unwrap();
                let msg = proc.heap.display_value(&exception);
                return Err(RuntimeError::Panic(format!("Exception: {}", msg)));
            }

            // === I/O ===
            Print(dst, src) => {
                let val = reg!(*src).clone();
                let proc = self.processes.get_mut(&local_id).unwrap();
                let s = proc.heap.display_value(&val);
                println!("{}", s);
                let str_ptr = proc.heap.alloc_string(s);
                set_reg!(*dst, GcValue::String(str_ptr));
            }

            // === Negation ===
            Not(dst, src) => {
                let val = reg!(*src).clone();
                match val {
                    GcValue::Bool(b) => set_reg!(*dst, GcValue::Bool(!b)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: format!("{:?}", val),
                    }),
                }
            }

            NegInt(dst, src) => {
                let val = reg!(*src).clone();
                match val {
                    GcValue::Int64(i) => set_reg!(*dst, GcValue::Int64(-i)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: format!("{:?}", val),
                    }),
                }
            }

            NegFloat(dst, src) => {
                let val = reg!(*src).clone();
                match val {
                    GcValue::Float64(f) => set_reg!(*dst, GcValue::Float64(-f)),
                    GcValue::Float32(f) => set_reg!(*dst, GcValue::Float32(-f)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: format!("{:?}", val),
                    }),
                }
            }

            // === Modulo ===
            ModInt(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => {
                        if *y == 0 {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        set_reg!(*dst, GcValue::Int64(x % y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            // Unimplemented
            other => {
                return Err(RuntimeError::Panic(format!(
                    "Instruction {:?} not yet implemented in ParallelVM",
                    other
                )));
            }
        }

        Ok(StepResult::Continue)
    }

    /// Load a constant value into the process heap.
    fn load_constant(&mut self, value: &Value, local_id: u64) -> GcValue {
        let proc = self.processes.get_mut(&local_id).unwrap();
        proc.heap.value_to_gc(value)
    }

    /// Call a function.
    fn call_function(
        &mut self,
        local_id: u64,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
        return_reg: Option<u8>,
    ) -> Result<(), RuntimeError> {
        let reg_count = func.code.register_count;
        let mut registers = vec![GcValue::Unit; reg_count];

        for (i, arg) in args.into_iter().enumerate() {
            if i < reg_count {
                registers[i] = arg;
            }
        }

        let frame = crate::process::CallFrame {
            function: func,
            ip: 0,
            registers,
            captures: vec![],
            return_reg,
        };

        let proc = self.processes.get_mut(&local_id).unwrap();
        proc.frames.push(frame);
        Ok(())
    }

    /// Call a closure.
    fn call_closure(
        &mut self,
        local_id: u64,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
        return_reg: Option<u8>,
    ) -> Result<(), RuntimeError> {
        let reg_count = func.code.register_count;
        let mut registers = vec![GcValue::Unit; reg_count];

        for (i, arg) in args.into_iter().enumerate() {
            if i < reg_count {
                registers[i] = arg;
            }
        }

        let frame = crate::process::CallFrame {
            function: func,
            ip: 0,
            registers,
            captures,
            return_reg,
        };

        let proc = self.processes.get_mut(&local_id).unwrap();
        proc.frames.push(frame);
        Ok(())
    }

    /// Tail call a function (replaces current frame).
    fn tail_call_function(
        &mut self,
        local_id: u64,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
    ) -> Result<(), RuntimeError> {
        let reg_count = func.code.register_count;
        let mut registers = vec![GcValue::Unit; reg_count];

        for (i, arg) in args.into_iter().enumerate() {
            if i < reg_count {
                registers[i] = arg;
            }
        }

        let proc = self.processes.get_mut(&local_id).unwrap();

        // Get return_reg from current frame before popping
        let return_reg = proc.frames.last().and_then(|f| f.return_reg);

        // Pop current frame
        proc.frames.pop();

        // Push new frame with same return_reg
        let frame = crate::process::CallFrame {
            function: func,
            ip: 0,
            registers,
            captures: vec![],
            return_reg,
        };
        proc.frames.push(frame);
        Ok(())
    }

    /// Tail call a closure (function with captures).
    fn tail_call_closure(
        &mut self,
        local_id: u64,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
    ) -> Result<(), RuntimeError> {
        let reg_count = func.code.register_count;
        let mut registers = vec![GcValue::Unit; reg_count];

        for (i, arg) in args.into_iter().enumerate() {
            if i < reg_count {
                registers[i] = arg;
            }
        }

        let proc = self.processes.get_mut(&local_id).unwrap();

        // Get return_reg from current frame before popping
        let return_reg = proc.frames.last().and_then(|f| f.return_reg);

        // Pop current frame
        proc.frames.pop();

        // Push new frame with same return_reg
        let frame = crate::process::CallFrame {
            function: func,
            ip: 0,
            registers,
            captures,
            return_reg,
        };
        proc.frames.push(frame);
        Ok(())
    }

    /// Call a native function.
    fn call_native(
        &mut self,
        local_id: u64,
        name: &str,
        args: Vec<GcValue>,
    ) -> Result<GcValue, RuntimeError> {
        // Note: Trait overrides for "show" and "copy" are handled in CallNative instruction
        let native = self.shared.natives.get(name)
            .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?
            .clone();

        let proc = self.processes.get_mut(&local_id).unwrap();
        (native.func)(&args, &mut proc.heap)
    }
}

/// Result of executing one instruction.
enum StepResult {
    Continue,
    Yield,
    Waiting,
    Finished(GcValue),
}

/// Result of executing a time slice.
enum SliceResult {
    Continue,
    Waiting,
    Finished(GcValue),
}

/// Result from fast loop execution.
enum FastLoopResult {
    Continue,
    Finished(GcValue),
    NeedSlowPath(Instruction),
}
