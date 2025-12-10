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

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Sender, Receiver, TryRecvError};

use tokio::sync::mpsc as tokio_mpsc;

use crate::gc::{GcConfig, GcList, GcMapKey, GcNativeFn, GcValue, Heap, InlineOp};
use crate::io_runtime::{IoRequest, IoRuntime};
use crate::process::{CallFrame, IoResponseValue, Process, ProcessState, ThreadSafeValue};
use crate::value::{FunctionValue, Instruction, Pid, RuntimeError, TypeValue, Value};

// JIT function pointer types (moved from runtime.rs)
pub type JitIntFn = fn(i64) -> i64;
pub type JitIntFn0 = fn() -> i64;
pub type JitIntFn2 = fn(i64, i64) -> i64;
pub type JitIntFn3 = fn(i64, i64, i64) -> i64;
pub type JitIntFn4 = fn(i64, i64, i64, i64) -> i64;
pub type JitLoopArrayFn = fn(*const i64, i64) -> i64;

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





/// Message sent between threads.
#[derive(Debug)]
enum CrossThreadMessage {
    /// Send message to existing process
    SendMessage {
        target_pid: Pid,
        payload: ThreadSafeValue,
    },
    /// Spawn a new process on this thread
    SpawnProcess {
        func: Arc<FunctionValue>,
        args: Vec<ThreadSafeValue>,
        captures: Vec<ThreadSafeValue>,
        /// Pre-allocated local_id for the new process
        pre_allocated_local_id: u64,
    },
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
    /// JIT-compiled integer functions (func_index → native fn) - arity 1
    pub jit_int_functions: HashMap<u16, JitIntFn>,
    /// JIT-compiled integer functions with arity 0
    pub jit_int_functions_0: HashMap<u16, JitIntFn0>,
    /// JIT-compiled integer functions with arity 2
    pub jit_int_functions_2: HashMap<u16, JitIntFn2>,
    /// JIT-compiled integer functions with arity 3
    pub jit_int_functions_3: HashMap<u16, JitIntFn3>,
    /// JIT-compiled integer functions with arity 4
    pub jit_int_functions_4: HashMap<u16, JitIntFn4>,
    /// JIT-compiled loop array functions (func_index → native fn)
    pub jit_loop_array_functions: HashMap<u16, JitLoopArrayFn>,
    /// Shutdown signal
    pub shutdown: AtomicBool,
    /// Spawn counter for round-robin distribution across threads
    pub spawn_counter: AtomicU64,
    /// Per-thread local_id counters for pre-allocating PIDs in cross-thread spawns
    pub thread_local_ids: Vec<AtomicU64>,
    /// Number of threads (for round-robin modulo)
    pub num_threads: usize,
    /// IO request sender (for async file/HTTP operations)
    pub io_sender: Option<tokio_mpsc::UnboundedSender<IoRequest>>,
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
    /// IO runtime for async file/HTTP operations
    io_runtime: Option<IoRuntime>,
}

/// Thread-safe sendable record value.
#[derive(Clone, Debug)]
pub struct SendableRecord {
    pub type_name: String,
    pub field_names: Vec<String>,
    pub fields: Vec<SendableValue>,
}

/// Thread-safe sendable map key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SendableMapKey {
    Unit,
    Bool(bool),
    Char(char),
    Int64(i64),
    String(String),
}

/// Simple value that can be sent between threads.
#[derive(Clone, Debug)]
pub enum SendableValue {
    Unit,
    Bool(bool),
    Char(char),
    // Signed integers
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    // Unsigned integers
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    // Floating point
    Float32(f32),
    Float64(f64),
    // Arbitrary precision
    BigInt(num_bigint::BigInt),
    Decimal(rust_decimal::Decimal),
    // Other
    Pid(u64),
    String(String),
    List(Vec<SendableValue>),
    Tuple(Vec<SendableValue>),
    Record(SendableRecord),
    Map(std::collections::HashMap<SendableMapKey, SendableValue>),
    Set(std::collections::HashSet<SendableMapKey>),
    Error(String),
}

impl SendableValue {
    fn from_gc_value(value: &GcValue, heap: &Heap) -> Self {
        match value {
            GcValue::Unit => SendableValue::Unit,
            GcValue::Bool(b) => SendableValue::Bool(*b),
            GcValue::Char(c) => SendableValue::Char(*c),
            // Signed integers
            GcValue::Int8(i) => SendableValue::Int8(*i),
            GcValue::Int16(i) => SendableValue::Int16(*i),
            GcValue::Int32(i) => SendableValue::Int32(*i),
            GcValue::Int64(i) => SendableValue::Int64(*i),
            // Unsigned integers
            GcValue::UInt8(i) => SendableValue::UInt8(*i),
            GcValue::UInt16(i) => SendableValue::UInt16(*i),
            GcValue::UInt32(i) => SendableValue::UInt32(*i),
            GcValue::UInt64(i) => SendableValue::UInt64(*i),
            // Floating point
            GcValue::Float32(f) => SendableValue::Float32(*f),
            GcValue::Float64(f) => SendableValue::Float64(*f),
            // BigInt
            GcValue::BigInt(ptr) => {
                if let Some(bi) = heap.get_bigint(*ptr) {
                    SendableValue::BigInt(bi.value.clone())
                } else {
                    SendableValue::String("<bigint>".to_string())
                }
            }
            // Decimal
            GcValue::Decimal(d) => SendableValue::Decimal(*d),
            // Process ID
            GcValue::Pid(p) => SendableValue::Pid(*p),
            GcValue::String(ptr) => {
                if let Some(s) = heap.get_string(*ptr) {
                    SendableValue::String(s.data.clone())
                } else {
                    SendableValue::String("<string>".to_string())
                }
            }
            GcValue::List(list) => {
                let items: Vec<SendableValue> = list.items()
                    .iter()
                    .map(|v| SendableValue::from_gc_value(v, heap))
                    .collect();
                SendableValue::List(items)
            }
            GcValue::Tuple(ptr) => {
                if let Some(tuple) = heap.get_tuple(*ptr) {
                    let items: Vec<SendableValue> = tuple.items
                        .iter()
                        .map(|v| SendableValue::from_gc_value(v, heap))
                        .collect();
                    SendableValue::Tuple(items)
                } else {
                    SendableValue::String("<tuple>".to_string())
                }
            }
            GcValue::Record(ptr) => {
                if let Some(record) = heap.get_record(*ptr) {
                    let fields: Vec<SendableValue> = record.fields
                        .iter()
                        .map(|v| SendableValue::from_gc_value(v, heap))
                        .collect();
                    SendableValue::Record(SendableRecord {
                        type_name: record.type_name.clone(),
                        field_names: record.field_names.clone(),
                        fields,
                    })
                } else {
                    SendableValue::String("<record>".to_string())
                }
            }
            GcValue::Set(ptr) => {
                if let Some(set) = heap.get_set(*ptr) {
                    let items: std::collections::HashSet<SendableMapKey> = set.items
                        .iter()
                        .filter_map(|k| Self::gc_map_key_to_sendable(k, heap))
                        .collect();
                    SendableValue::Set(items)
                } else {
                    SendableValue::String("<set>".to_string())
                }
            }
            GcValue::Map(ptr) => {
                if let Some(map) = heap.get_map(*ptr) {
                    let entries: std::collections::HashMap<SendableMapKey, SendableValue> = map.entries
                        .iter()
                        .filter_map(|(k, v)| {
                            Self::gc_map_key_to_sendable(k, heap)
                                .map(|sk| (sk, SendableValue::from_gc_value(v, heap)))
                        })
                        .collect();
                    SendableValue::Map(entries)
                } else {
                    SendableValue::String("<map>".to_string())
                }
            }
            // For other values, use their display representation
            _ => SendableValue::String(heap.display_value(value)),
        }
    }

    fn gc_map_key_to_sendable(key: &crate::gc::GcMapKey, _heap: &Heap) -> Option<SendableMapKey> {
        use crate::gc::GcMapKey;
        match key {
            GcMapKey::Unit => Some(SendableMapKey::Unit),
            GcMapKey::Bool(b) => Some(SendableMapKey::Bool(*b)),
            GcMapKey::Char(c) => Some(SendableMapKey::Char(*c)),
            GcMapKey::Int64(i) => Some(SendableMapKey::Int64(*i)),
            GcMapKey::Int8(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::Int16(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::Int32(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::UInt8(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::UInt16(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::UInt32(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::UInt64(i) => Some(SendableMapKey::Int64(*i as i64)),
            GcMapKey::String(s) => Some(SendableMapKey::String(s.clone())),
        }
    }

    /// Convert to GcValue for output (lossy for strings).
    pub fn to_gc_value(&self) -> GcValue {
        match self {
            SendableValue::Unit => GcValue::Unit,
            SendableValue::Bool(b) => GcValue::Bool(*b),
            SendableValue::Char(c) => GcValue::Char(*c),
            SendableValue::Int8(i) => GcValue::Int8(*i),
            SendableValue::Int16(i) => GcValue::Int16(*i),
            SendableValue::Int32(i) => GcValue::Int32(*i),
            SendableValue::Int64(i) => GcValue::Int64(*i),
            SendableValue::UInt8(i) => GcValue::UInt8(*i),
            SendableValue::UInt16(i) => GcValue::UInt16(*i),
            SendableValue::UInt32(i) => GcValue::UInt32(*i),
            SendableValue::UInt64(i) => GcValue::UInt64(*i),
            SendableValue::Float32(f) => GcValue::Float32(*f),
            SendableValue::Float64(f) => GcValue::Float64(*f),
            SendableValue::Decimal(d) => GcValue::Decimal(*d),
            SendableValue::Pid(p) => GcValue::Pid(*p),
            SendableValue::String(s) => {
                let mut heap = Heap::new();
                let ptr = heap.alloc_string(s.clone());
                GcValue::String(ptr)
            }
            _ => GcValue::Unit, // Complex types don't convert back cleanly
        }
    }

    /// Display the value as a string.
    pub fn display(&self) -> String {
        match self {
            SendableValue::Unit => "()".to_string(),
            SendableValue::Bool(b) => b.to_string(),
            SendableValue::Char(c) => format!("'{}'", c),
            SendableValue::Int8(i) => format!("{}i8", i),
            SendableValue::Int16(i) => format!("{}i16", i),
            SendableValue::Int32(i) => format!("{}i32", i),
            SendableValue::Int64(i) => i.to_string(),
            SendableValue::UInt8(i) => format!("{}u8", i),
            SendableValue::UInt16(i) => format!("{}u16", i),
            SendableValue::UInt32(i) => format!("{}u32", i),
            SendableValue::UInt64(i) => format!("{}u64", i),
            SendableValue::Float32(f) => format!("{}f32", f),
            SendableValue::Float64(f) => f.to_string(),
            SendableValue::BigInt(bi) => format!("{}n", bi),
            SendableValue::Decimal(d) => format!("{}d", d),
            SendableValue::Pid(p) => format!("<pid {}>", p),
            SendableValue::String(s) => s.clone(),
            SendableValue::List(items) => {
                let items_str: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", items_str.join(", "))
            }
            SendableValue::Tuple(items) => {
                let items_str: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("({})", items_str.join(", "))
            }
            SendableValue::Record(r) => {
                let fields_str: Vec<String> = r.field_names.iter()
                    .zip(r.fields.iter())
                    .map(|(n, v)| format!("{}: {}", n, v.display()))
                    .collect();
                format!("{}{{{}}}", r.type_name, fields_str.join(", "))
            }
            SendableValue::Set(items) => {
                format!("#{{...{} items}}", items.len())
            }
            SendableValue::Map(entries) => {
                format!("%{{...{} entries}}", entries.len())
            }
            SendableValue::Error(e) => format!("Error: {}", e),
        }
    }

    /// Check if this is Unit.
    pub fn is_unit(&self) -> bool {
        matches!(self, SendableValue::Unit)
    }

    /// Convert to Value (for tests and compatibility).
    pub fn to_value(&self) -> Value {
        match self {
            SendableValue::Unit => Value::Unit,
            SendableValue::Bool(b) => Value::Bool(*b),
            SendableValue::Char(c) => Value::Char(*c),
            SendableValue::Int8(i) => Value::Int8(*i),
            SendableValue::Int16(i) => Value::Int16(*i),
            SendableValue::Int32(i) => Value::Int32(*i),
            SendableValue::Int64(i) => Value::Int64(*i),
            SendableValue::UInt8(i) => Value::UInt8(*i),
            SendableValue::UInt16(i) => Value::UInt16(*i),
            SendableValue::UInt32(i) => Value::UInt32(*i),
            SendableValue::UInt64(i) => Value::UInt64(*i),
            SendableValue::Float32(f) => Value::Float32(*f),
            SendableValue::Float64(f) => Value::Float64(*f),
            SendableValue::BigInt(bi) => Value::BigInt(Arc::new(bi.clone())),
            SendableValue::Decimal(d) => Value::Decimal(*d),
            SendableValue::Pid(_) => Value::Unit,
            SendableValue::String(s) => Value::String(Arc::new(s.clone())),
            SendableValue::List(items) => {
                let values: Vec<Value> = items.iter().map(|v| v.to_value()).collect();
                Value::List(Arc::new(values))
            }
            SendableValue::Tuple(items) => {
                let values: Vec<Value> = items.iter().map(|v| v.to_value()).collect();
                Value::Tuple(Arc::new(values))
            }
            SendableValue::Record(r) => {
                let fields: Vec<Value> = r.fields.iter().map(|v| v.to_value()).collect();
                Value::Record(Arc::new(crate::value::RecordValue {
                    type_name: r.type_name.clone(),
                    field_names: r.field_names.clone(),
                    fields,
                    mutable_fields: vec![false; r.fields.len()],
                }))
            }
            SendableValue::Set(items) => {
                let set: std::collections::HashSet<crate::value::MapKey> = items.iter()
                    .map(|k| Self::sendable_key_to_map_key(k))
                    .collect();
                Value::Set(Arc::new(set))
            }
            SendableValue::Map(entries) => {
                let map: std::collections::HashMap<crate::value::MapKey, Value> = entries.iter()
                    .map(|(k, v)| (Self::sendable_key_to_map_key(k), v.to_value()))
                    .collect();
                Value::Map(Arc::new(map))
            }
            SendableValue::Error(_) => Value::Unit,
        }
    }

    fn sendable_key_to_map_key(key: &SendableMapKey) -> crate::value::MapKey {
        use crate::value::MapKey;
        match key {
            SendableMapKey::Unit => MapKey::Unit,
            SendableMapKey::Bool(b) => MapKey::Bool(*b),
            SendableMapKey::Char(c) => MapKey::Char(*c),
            SendableMapKey::Int64(i) => MapKey::Int64(*i),
            SendableMapKey::String(s) => MapKey::String(Arc::new(s.clone())),
        }
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
    /// Local processes owned by this thread - Vec indexed by local_id for O(1) access
    processes: Vec<Option<Process>>,
    /// Local run queue (ready processes)
    run_queue: VecDeque<u64>, // local_ids
    /// Timer heap for sleeping/timed-out processes: min-heap of (wake_time, local_id)
    timer_heap: BinaryHeap<Reverse<(Instant, u64)>>,
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
    /// List of local_ids waiting for async IO (for efficient polling)
    io_waiting: Vec<u64>,
    /// Idle backoff counter (doubles sleep time up to max when idle)
    idle_backoff: u32,
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

        // Create per-thread local_id counters (starting at 1, 0 is initial process)
        let thread_local_ids: Vec<AtomicU64> = (0..num_threads)
            .map(|_| AtomicU64::new(1))
            .collect();

        // Create IO runtime for async operations
        let io_runtime = IoRuntime::new();
        let io_sender = io_runtime.request_sender();

        let shared = Arc::new(SharedState {
            functions: HashMap::new(),
            function_list: Vec::new(),
            natives: HashMap::new(),
            types: HashMap::new(),
            jit_int_functions: HashMap::new(),
            jit_int_functions_0: HashMap::new(),
            jit_int_functions_2: HashMap::new(),
            jit_int_functions_3: HashMap::new(),
            jit_int_functions_4: HashMap::new(),
            jit_loop_array_functions: HashMap::new(),
            shutdown: AtomicBool::new(false),
            spawn_counter: AtomicU64::new(0),
            thread_local_ids,
            num_threads,
            io_sender: Some(io_sender),
        });

        Self {
            shared,
            thread_senders: Vec::new(),
            threads: Vec::new(),
            num_threads,
            next_thread: AtomicU64::new(0),
            config,
            io_runtime: Some(io_runtime),
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

    /// Register a JIT-compiled integer function (arity 1).
    pub fn register_jit_int_function(&mut self, func_index: u16, jit_fn: JitIntFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_int_functions
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 0).
    pub fn register_jit_int_function_0(&mut self, func_index: u16, jit_fn: JitIntFn0) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_int_functions_0
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 2).
    pub fn register_jit_int_function_2(&mut self, func_index: u16, jit_fn: JitIntFn2) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_int_functions_2
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 3).
    pub fn register_jit_int_function_3(&mut self, func_index: u16, jit_fn: JitIntFn3) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_int_functions_3
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 4).
    pub fn register_jit_int_function_4(&mut self, func_index: u16, jit_fn: JitIntFn4) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_int_functions_4
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

        // String.chars
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

        // Hash - compute hash value for a value (FNV-1a algorithm)
        self.register_native("hash", Arc::new(GcNativeFn {
            name: "hash".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                fn fnv1a_hash(bytes: &[u8]) -> i64 {
                    let mut hash: u64 = 14695981039346656037; // FNV offset basis
                    for byte in bytes {
                        hash ^= *byte as u64;
                        hash = hash.wrapping_mul(1099511628211); // FNV prime
                    }
                    hash as i64
                }

                let hash_val = match &args[0] {
                    GcValue::Unit => 0i64,
                    GcValue::Bool(b) => if *b { 1 } else { 0 },
                    GcValue::Char(c) => fnv1a_hash(&(*c as u32).to_le_bytes()),
                    GcValue::Int8(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::Int16(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::Int32(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::Int64(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::UInt8(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::UInt16(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::UInt32(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::UInt64(n) => fnv1a_hash(&n.to_le_bytes()),
                    GcValue::Float32(f) => fnv1a_hash(&f.to_le_bytes()),
                    GcValue::Float64(f) => fnv1a_hash(&f.to_le_bytes()),
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            fnv1a_hash(str_val.data.as_bytes())
                        } else {
                            return Err(RuntimeError::Panic("Invalid string pointer".to_string()));
                        }
                    }
                    // For complex types, return an error - user should implement Hash trait
                    _ => return Err(RuntimeError::TypeError {
                        expected: "hashable type (primitive or String)".to_string(),
                        found: "complex type - implement Hash trait".to_string()
                    }),
                };
                Ok(GcValue::Int64(hash_val))
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
                    let pid = worker.spawn_main_process(func);
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
            processes: vec![None], // Slot 0 unused, local_id starts at 1
            run_queue: VecDeque::new(),
            timer_heap: BinaryHeap::new(),
            next_local_id: 1,
            inbox,
            thread_senders,
            shared,
            config,
            main_pid: None,
            main_result: None,
            io_waiting: Vec::new(),
            idle_backoff: 0,
        }
    }

    /// Insert a process at the given local_id, growing the Vec if needed.
    #[inline]
    fn insert_process(&mut self, local_id: u64, process: Process) {
        let idx = local_id as usize;
        if idx >= self.processes.len() {
            self.processes.resize_with(idx + 1, || None);
        }
        self.processes[idx] = Some(process);
    }

    /// Get an immutable reference to a process by local_id.
    #[inline]
    fn get_process(&self, local_id: u64) -> Option<&Process> {
        self.processes.get(local_id as usize).and_then(|opt| opt.as_ref())
    }

    /// Get a mutable reference to a process by local_id.
    #[inline]
    fn get_process_mut(&mut self, local_id: u64) -> Option<&mut Process> {
        self.processes.get_mut(local_id as usize).and_then(|opt| opt.as_mut())
    }

    /// Remove a process by local_id.
    #[inline]
    fn remove_process(&mut self, local_id: u64) {
        if let Some(slot) = self.processes.get_mut(local_id as usize) {
            *slot = None;
        }
    }

    /// Spawn the main process on this thread (no arguments).
    fn spawn_main_process(&mut self, func: Arc<FunctionValue>) -> Pid {
        let local_id = self.next_local_id;
        self.next_local_id += 1;
        let pid = encode_pid(self.thread_id, local_id);

        // Create process with lightweight heap
        let mut process = Process::with_gc_config(pid, GcConfig::lightweight());

        // Set up initial call frame (no arguments for main)
        let reg_count = func.code.register_count;
        let registers = vec![GcValue::Unit; reg_count];

        let frame = crate::process::CallFrame {
            function: func,
            ip: 0,
            registers,
            captures: vec![],
            return_reg: None,
        };
        process.frames.push(frame);

        self.insert_process(local_id, process);
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

            // Wake up any processes whose timers have expired
            self.check_timers();

            // Check for completed async IO operations
            self.check_io();

            // Get next process to run
            let local_id = match self.run_queue.pop_front() {
                Some(id) => {
                    // Found work - reset backoff
                    self.idle_backoff = 0;
                    id
                }
                None => {
                    // No runnable work - use exponential backoff
                    // Start at 50µs, double each iteration, cap at 10ms
                    let base_us = 50u64;
                    let max_us = 10_000u64; // 10ms max
                    let sleep_us = (base_us << self.idle_backoff.min(8)).min(max_us);

                    // Increment backoff (will reset when we find work)
                    if self.idle_backoff < 8 {
                        self.idle_backoff += 1;
                    }

                    std::thread::sleep(std::time::Duration::from_micros(sleep_us));
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
                        let proc = self.get_process(local_id).unwrap();
                        let sendable = SendableValue::from_gc_value(&value, &proc.heap);
                        self.main_result = Some(Ok(sendable));
                        // Signal shutdown
                        self.shared.shutdown.store(true, Ordering::SeqCst);
                    }
                    self.remove_process(local_id);
                }
                Err(e) => {
                    // Process error
                    let pid = encode_pid(self.thread_id, local_id);
                    if Some(pid) == self.main_pid {
                        self.main_result = Some(Err(e.to_string()));
                        self.shared.shutdown.store(true, Ordering::SeqCst);
                    }
                    self.remove_process(local_id);
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
                    match msg {
                        CrossThreadMessage::SendMessage { target_pid, payload } => {
                            let local_id = pid_local_id(target_pid);
                            if let Some(process) = self.get_process_mut(local_id) {
                                // Send to process channel (thread-safe)
                                let _ = process.sender.send(payload);

                                // Wake process if waiting for message (with or without timeout)
                                if matches!(process.state, ProcessState::Waiting | ProcessState::WaitingTimeout) {
                                    process.state = ProcessState::Running;
                                    process.wake_time = None; // Cancel any pending timeout
                                    process.timeout_dst = None; // Clear timeout destination
                                    self.run_queue.push_back(local_id);
                                }
                            }
                            // If process doesn't exist, message is dropped (process died)
                        }
                        CrossThreadMessage::SpawnProcess { func, args, captures, pre_allocated_local_id } => {
                            // Spawn the process on this thread using the pre-allocated local_id
                            let local_id = pre_allocated_local_id;
                            // Ensure next_local_id is at least pre_allocated_local_id + 1
                            // to avoid conflicts with future local spawns
                            if self.next_local_id <= local_id {
                                self.next_local_id = local_id + 1;
                            }
                            let pid = encode_pid(self.thread_id, local_id);

                            let mut process = Process::new(pid);

                            // Convert args from ThreadSafeValue to GcValue
                            let gc_args: Vec<GcValue> = args.iter()
                                .map(|v| v.to_gc_value(&mut process.heap))
                                .collect();
                            let gc_captures: Vec<GcValue> = captures.iter()
                                .map(|v| v.to_gc_value(&mut process.heap))
                                .collect();

                            // Set up initial frame
                            let reg_count = func.code.register_count;
                            let mut registers = vec![GcValue::Unit; reg_count];
                            for (i, arg) in gc_args.into_iter().enumerate() {
                                if i < reg_count {
                                    registers[i] = arg;
                                }
                            }

                            process.frames.push(CallFrame {
                                function: func,
                                ip: 0,
                                registers,
                                captures: gc_captures,
                                return_reg: None,
                            });

                            self.insert_process(local_id, process);
                            self.run_queue.push_back(local_id);
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    /// Check timer heap and wake up any processes whose timers have expired.
    #[inline]
    fn check_timers(&mut self) {
        // Fast path: no timers, no work
        if self.timer_heap.is_empty() {
            return;
        }
        let now = Instant::now();
        while let Some(&Reverse((wake_time, local_id))) = self.timer_heap.peek() {
            if wake_time <= now {
                self.timer_heap.pop();
                if let Some(proc) = self.get_process_mut(local_id) {
                    match proc.state {
                        ProcessState::Sleeping => {
                            // Sleep completed - wake the process
                            proc.state = ProcessState::Running;
                            proc.wake_time = None;
                            self.run_queue.push_back(local_id);
                        }
                        ProcessState::WaitingTimeout => {
                            // Receive timeout expired - wake with timeout indicator
                            proc.state = ProcessState::Running;
                            proc.wake_time = None;
                            // Set destination register to Unit to indicate timeout
                            if let Some(frame) = proc.frames.last_mut() {
                                let dst = proc.timeout_dst.unwrap_or(0) as usize;
                                frame.registers[dst] = GcValue::Unit; // Unit indicates timeout
                                // Increment IP to skip re-executing ReceiveTimeout
                                // (IP was decremented when entering wait state)
                                frame.ip += 1;
                            }
                            proc.timeout_dst = None; // Clear it after use
                            self.run_queue.push_back(local_id);
                        }
                        _ => {
                            // Process state changed (e.g., got message), ignore timer
                        }
                    }
                }
            } else {
                break; // No more ready timers
            }
        }
    }

    /// Check for completed async IO operations and wake up waiting processes.
    #[inline]
    fn check_io(&mut self) {
        // Fast path: no IO-waiting processes
        if self.io_waiting.is_empty() {
            return;
        }

        // Collect indices to check (to avoid borrow issues)
        let to_check: Vec<u64> = self.io_waiting.clone();
        let mut completed = Vec::new();
        let mut to_requeue = Vec::new();

        for local_id in to_check {
            if let Some(proc) = self.get_process_mut(local_id) {
                if proc.state == ProcessState::WaitingIO {
                    if let Some((result, result_reg)) = proc.poll_io() {
                        // IO completed - convert result to GcValue and store in register
                        let gc_value = Self::io_result_to_gc_value(result, proc);

                        // Store result in destination register
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.registers[result_reg as usize] = gc_value;
                        }

                        completed.push(local_id);
                        to_requeue.push(local_id);
                    }
                }
            } else {
                // Process was removed while waiting for IO
                completed.push(local_id);
            }
        }

        // Re-queue completed processes
        for local_id in to_requeue {
            self.run_queue.push_back(local_id);
        }

        // Remove completed entries from io_waiting
        if !completed.is_empty() {
            self.io_waiting.retain(|id| !completed.contains(id));
        }
    }

    /// Convert an IO result to a GcValue. This is a static method to avoid borrow issues.
    /// All results are wrapped in ("ok", value) or ("error", message) tuples for consistency.
    fn io_result_to_gc_value(
        result: Result<IoResponseValue, crate::io_runtime::IoError>,
        proc: &mut Process,
    ) -> GcValue {
        match result {
            Ok(response) => {
                let value = Self::io_response_to_gc_value_static(response, proc);
                // Wrap in ("ok", value) tuple for consistency
                let ok_tag = GcValue::String(proc.heap.alloc_string("ok".to_string()));
                GcValue::Tuple(proc.heap.alloc_tuple(vec![ok_tag, value]))
            }
            Err(io_err) => {
                // Create error tuple: ("error", message)
                let err_tag = GcValue::String(proc.heap.alloc_string("error".to_string()));
                let msg = GcValue::String(proc.heap.alloc_string(io_err.to_string()));
                GcValue::Tuple(proc.heap.alloc_tuple(vec![err_tag, msg]))
            }
        }
    }

    /// Convert an IO response value to a GcValue. Static method to avoid borrow issues.
    fn io_response_to_gc_value_static(response: IoResponseValue, proc: &mut Process) -> GcValue {
        use crate::process::IoResponseValue::*;
        match response {
            Unit => GcValue::Unit,
            Bytes(bytes) => {
                // Try to convert bytes to string (UTF-8), fallback to list of integers
                match std::string::String::from_utf8(bytes.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let values: Vec<GcValue> = bytes.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList { data: Arc::new(values), start: 0 })
                    }
                }
            }
            String(s) => GcValue::String(proc.heap.alloc_string(s)),
            FileHandle(handle_id) => {
                // Return as int (file handle id) - wrapping in ("ok", ...) is done by caller
                GcValue::Int64(handle_id as i64)
            }
            Int(n) => GcValue::Int64(n),
            HttpResponse { status, headers, body } => {
                // Return as HttpResponse{status, headers, body} - wrapping in ("ok", ...) is done by caller
                // Build headers list of {key, value} tuples
                let header_tuples: Vec<GcValue> = headers
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(proc.heap.alloc_string(k));
                        let val = GcValue::String(proc.heap.alloc_string(v));
                        GcValue::Tuple(proc.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let headers_list = GcValue::List(GcList { data: Arc::new(header_tuples), start: 0 });

                // Build body as string (try UTF-8, fallback to bytes list)
                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList { data: Arc::new(bytes), start: 0 })
                    }
                };

                // Build response as a record with named fields
                GcValue::Record(proc.heap.alloc_record(
                    "HttpResponse".to_string(),
                    vec!["status".to_string(), "headers".to_string(), "body".to_string()],
                    vec![GcValue::Int64(status as i64), headers_list, body_value],
                    vec![false, false, false], // all immutable
                ))
            }
            OptionString(opt) => {
                // Return value directly - wrapping in ("ok", ...) is done by caller
                match opt {
                    Some(s) => GcValue::String(proc.heap.alloc_string(s)),
                    None => {
                        // EOF - return special eof indicator that caller will wrap
                        GcValue::String(proc.heap.alloc_string("eof".to_string()))
                    }
                }
            }
            Bool(b) => {
                GcValue::Bool(b)
            }
            StringList(strings) => {
                let values: Vec<GcValue> = strings
                    .into_iter()
                    .map(|s| GcValue::String(proc.heap.alloc_string(s)))
                    .collect();
                GcValue::List(GcList { data: Arc::new(values), start: 0 })
            }
            ServerHandle(handle_id) => {
                // Return server handle as an integer
                GcValue::Int64(handle_id as i64)
            }
            ServerRequest { request_id, method, path, headers, body } => {
                // Build headers list of {key, value} tuples
                let header_tuples: Vec<GcValue> = headers
                    .into_iter()
                    .map(|(k, v)| {
                        let key = GcValue::String(proc.heap.alloc_string(k));
                        let val = GcValue::String(proc.heap.alloc_string(v));
                        GcValue::Tuple(proc.heap.alloc_tuple(vec![key, val]))
                    })
                    .collect();
                let headers_list = GcValue::List(GcList { data: Arc::new(header_tuples), start: 0 });

                // Build body as string (try UTF-8, fallback to bytes list)
                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList { data: Arc::new(bytes), start: 0 })
                    }
                };

                // Pre-allocate strings to avoid borrow issues
                let method_str = GcValue::String(proc.heap.alloc_string(method));
                let path_str = GcValue::String(proc.heap.alloc_string(path));

                // Build request as a record with named fields
                GcValue::Record(proc.heap.alloc_record(
                    "HttpRequest".to_string(),
                    vec!["id".to_string(), "method".to_string(), "path".to_string(), "headers".to_string(), "body".to_string()],
                    vec![
                        GcValue::Int64(request_id as i64),
                        method_str,
                        path_str,
                        headers_list,
                        body_value
                    ],
                    vec![false, false, false, false, false], // all immutable
                ))
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
            let sender_process = self.get_process(sender_local_id).unwrap();
            let safe_value = ThreadSafeValue::from_gc_value(&message, &sender_process.heap);

            if let (Some(safe), Some(target_process)) = (safe_value, self.get_process_mut(target_local_id)) {
                // Send to target channel
                let _ = target_process.sender.send(safe);

                // Wake process if waiting for message (with or without timeout)
                if matches!(target_process.state, ProcessState::Waiting | ProcessState::WaitingTimeout) {
                    target_process.state = ProcessState::Running;
                    target_process.wake_time = None; // Cancel any pending timeout
                    target_process.timeout_dst = None; // Clear timeout destination
                    self.run_queue.push_back(target_local_id);
                }
            }
        } else {
            // Cross-thread send
            // Convert to thread-safe value
            let sender_process = self.get_process(sender_local_id).unwrap();
            if let Some(safe_value) = ThreadSafeValue::from_gc_value(&message, &sender_process.heap) {
                if let Some(sender) = self.thread_senders.get(target_thread as usize) {
                    let _ = sender.send(CrossThreadMessage::SendMessage {
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
        let mut remaining = self.config.reductions_per_slice;

        // FAST PATH: Execute as many instructions as possible with ONE HashMap lookup
        // Only break out for slow-path instructions or when done
        loop {
            match self.execute_fast_loop(local_id, remaining) {
                Err(e) => {
                    // Wrap error with stack trace
                    let stack_trace = if let Some(proc) = self.get_process(local_id) {
                        crate::process::format_stack_trace(&proc.frames)
                    } else {
                        String::from("  <no stack frames>")
                    };
                    return Err(e.with_stack_trace(stack_trace));
                }
                Ok(fast_result) => match fast_result {
                FastLoopResult::Continue => {
                    // Did all reductions - time to yield
                    // GC safepoint: collect garbage if heap threshold exceeded
                    if let Some(proc) = self.get_process_mut(local_id) {
                        proc.maybe_gc();
                    }
                    return Ok(SliceResult::Continue);
                }
                FastLoopResult::Finished(v) => return Ok(SliceResult::Finished(v)),
                FastLoopResult::NeedSlowPath(instr) => {
                    // Fall back to slow path for complex instructions
                    let constants = {
                        let proc = self.get_process(local_id).unwrap();
                        proc.frames.last().unwrap().function.code.constants.clone()
                    };
                    let step_result = match self.execute_instruction(local_id, &instr, &constants) {
                        Ok(result) => result,
                        Err(e) => {
                            // Wrap slow-path error with stack trace
                            let stack_trace = if let Some(proc) = self.get_process(local_id) {
                                crate::process::format_stack_trace(&proc.frames)
                            } else {
                                String::from("  <no stack frames>")
                            };
                            return Err(e.with_stack_trace(stack_trace));
                        }
                    };
                    match step_result {
                        StepResult::Continue => {
                            // GC safepoint after slow-path instruction (where allocations happen)
                            if let Some(proc) = self.get_process_mut(local_id) {
                                proc.maybe_gc();
                            }
                            remaining = remaining.saturating_sub(1);
                            if remaining == 0 {
                                return Ok(SliceResult::Continue);
                            }
                            continue; // Back to fast loop
                        }
                        StepResult::Yield => {
                            // GC safepoint at slow-path yield
                            if let Some(proc) = self.get_process_mut(local_id) {
                                proc.maybe_gc();
                            }
                            return Ok(SliceResult::Continue);
                        }
                        StepResult::Waiting => return Ok(SliceResult::Waiting),
                        StepResult::Finished(v) => return Ok(SliceResult::Finished(v)),
                    }
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
        use crate::process::CallFrame;

        // Clone Arc to access shared state without conflicting with processes borrow
        let shared = Arc::clone(&self.shared);
        let proc = self.get_process_mut(local_id).unwrap();

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
                    // OPTIMIZATION: Use stack array instead of heap allocation
                    // This is critical for recursive functions like fold
                    if args.len() <= 8 {
                        let mut saved_args: [std::mem::MaybeUninit<GcValue>; 8] =
                            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                        // Save args to stack (take ownership)
                        for (i, &r) in args.iter().enumerate() {
                            saved_args[i] = std::mem::MaybeUninit::new(
                                std::mem::take(&mut proc.frames[frame_idx].registers[r as usize])
                            );
                        }

                        // Note: We don't clear remaining registers - they're either
                        // overwritten or unused. GC handles cleanup.

                        // Write args to positions 0, 1, 2, ...
                        let arg_count = args.len();
                        for i in 0..arg_count {
                            proc.frames[frame_idx].registers[i] =
                                unsafe { saved_args[i].assume_init_read() };
                        }

                        proc.frames[frame_idx].ip = 0;
                    } else {
                        // Fall back to heap allocation for many args
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
                        GcValue::List(list) => {
                            list.len() as i64
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
                MakeInt64Array(dst, size_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let size = match &proc.frames[frame_idx].registers[*size_reg as usize] {
                        GcValue::Int64(n) => *n as usize,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    let ptr = proc.heap.alloc_int64_array(vec![0i64; size]);
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64Array(ptr);
                }
                CallDirect(dst, func_idx, args) => {
                    proc.frames[frame_idx].ip += 1;

                    // Check for JIT-compiled version first based on arity
                    match args.len() {
                        0 => {
                            // Arity 0: no arguments
                            if let Some(jit_fn) = shared.jit_int_functions_0.get(func_idx) {
                                let result = jit_fn();
                                proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                continue;
                            }
                        }
                        1 => {
                            // Arity 1: single argument (most common case)
                            // Pure numeric JIT
                            if let Some(jit_fn) = shared.jit_int_functions.get(func_idx) {
                                if let GcValue::Int64(n) = &proc.frames[frame_idx].registers[args[0] as usize] {
                                    let result = jit_fn(*n);
                                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                    continue;
                                }
                            }
                            // Loop array JIT (only check if we have any registered)
                            if !shared.jit_loop_array_functions.is_empty() {
                                if let Some(jit_fn) = shared.jit_loop_array_functions.get(func_idx) {
                                    if let GcValue::Int64Array(arr_ptr) = &proc.frames[frame_idx].registers[args[0] as usize] {
                                        if let Some(arr) = proc.heap.get_int64_array_mut(*arr_ptr) {
                                            let ptr = arr.items.as_mut_ptr();
                                            let len = arr.items.len() as i64;
                                            let result = jit_fn(ptr as *const i64, len);
                                            proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                        2 => {
                            // Arity 2: two arguments
                            if let Some(jit_fn) = shared.jit_int_functions_2.get(func_idx) {
                                if let (GcValue::Int64(a), GcValue::Int64(b)) = (
                                    &proc.frames[frame_idx].registers[args[0] as usize],
                                    &proc.frames[frame_idx].registers[args[1] as usize],
                                ) {
                                    let result = jit_fn(*a, *b);
                                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                    continue;
                                }
                            }
                        }
                        3 => {
                            // Arity 3: three arguments
                            if let Some(jit_fn) = shared.jit_int_functions_3.get(func_idx) {
                                if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c)) = (
                                    &proc.frames[frame_idx].registers[args[0] as usize],
                                    &proc.frames[frame_idx].registers[args[1] as usize],
                                    &proc.frames[frame_idx].registers[args[2] as usize],
                                ) {
                                    let result = jit_fn(*a, *b, *c);
                                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                    continue;
                                }
                            }
                        }
                        4 => {
                            // Arity 4: four arguments
                            if let Some(jit_fn) = shared.jit_int_functions_4.get(func_idx) {
                                if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c), GcValue::Int64(d)) = (
                                    &proc.frames[frame_idx].registers[args[0] as usize],
                                    &proc.frames[frame_idx].registers[args[1] as usize],
                                    &proc.frames[frame_idx].registers[args[2] as usize],
                                    &proc.frames[frame_idx].registers[args[3] as usize],
                                ) {
                                    let result = jit_fn(*a, *b, *c, *d);
                                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                    continue;
                                }
                            }
                        }
                        _ => {
                            // More than 4 arguments: no JIT support
                        }
                    }

                    // Fall back to interpreted call
                    let func = match shared.function_list.get(*func_idx as usize) {
                        Some(f) => Arc::clone(f),
                        None => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };

                    // Collect arguments
                    let arg_values: Vec<GcValue> = args.iter()
                        .map(|&r| proc.frames[frame_idx].registers[r as usize].clone())
                        .collect();

                    // Push new frame
                    let reg_count = func.code.register_count;
                    let mut registers = vec![GcValue::Unit; reg_count];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            registers[i] = arg;
                        }
                    }

                    proc.frames.push(CallFrame {
                        function: func,
                        ip: 0,
                        registers,
                        captures: Vec::new(),
                        return_reg: Some(*dst),
                    });
                }
                LoadConst(dst, const_idx) => {
                    proc.frames[frame_idx].ip += 1;
                    // SAFETY: const_idx was validated at compile time
                    let constant = unsafe {
                        let frame = proc.frames.get_unchecked(frame_idx);
                        frame.function.code.constants.get_unchecked(*const_idx as usize).clone()
                    };
                    // Convert Value to GcValue (allocate on heap if needed)
                    let gc_val = proc.heap.value_to_gc(&constant);
                    proc.frames[frame_idx].registers[*dst as usize] = gc_val;
                }
                // Fast path for list operations
                TestNil(dst, list_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let is_nil = match &proc.frames[frame_idx].registers[*list_reg as usize] {
                        GcValue::List(list) => list.is_empty(),
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(is_nil);
                }
                ListIsEmpty(dst, list_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let is_empty = match &proc.frames[frame_idx].registers[*list_reg as usize] {
                        GcValue::List(list) => list.is_empty(),
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(is_empty);
                }
                ListSum(dst, list_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let sum = match &proc.frames[frame_idx].registers[*list_reg as usize] {
                        GcValue::List(list) => {
                            let mut total: i64 = 0;
                            for item in list.items() {
                                match item {
                                    GcValue::Int64(n) => total += n,
                                    _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                                }
                            }
                            total
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(sum);
                }
                // Native range list creation - equivalent to [1..n]
                RangeList(dst, n_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let n = match &proc.frames[frame_idx].registers[*n_reg as usize] {
                        GcValue::Int64(n) => *n,
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    // Create list [1, 2, ..., n] efficiently
                    let items: Vec<GcValue> = (1..=n).map(|i| GcValue::Int64(i)).collect();
                    let list = proc.heap.make_list(items);
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::List(list);
                }
                ListHead(dst, list_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let head = match &proc.frames[frame_idx].registers[*list_reg as usize] {
                        GcValue::List(list) => {
                            if !list.is_empty() {
                                list.items()[0].clone()
                            } else {
                                return Ok(FastLoopResult::NeedSlowPath(instr.clone()));
                            }
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = head;
                }
                ListTail(dst, list_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    match &proc.frames[frame_idx].registers[*list_reg as usize] {
                        GcValue::List(list) => {
                            // Lists are inline - no heap lookup needed!
                            if !list.is_empty() {
                                // O(1) tail with no allocation
                                let tail_list = list.tail();
                                proc.frames[frame_idx].registers[*dst as usize] = GcValue::List(tail_list);
                            } else {
                                return Ok(FastLoopResult::NeedSlowPath(instr.clone()));
                            }
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    }
                }
                Decons(head_dst, tail_dst, list_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    match &proc.frames[frame_idx].registers[*list_reg as usize] {
                        GcValue::List(list) => {
                            if !list.is_empty() {
                                let head = list.items()[0].clone();
                                let tail_list = list.tail();
                                proc.frames[frame_idx].registers[*head_dst as usize] = head;
                                proc.frames[frame_idx].registers[*tail_dst as usize] = GcValue::List(tail_list);
                            } else {
                                return Ok(FastLoopResult::NeedSlowPath(instr.clone()));
                            }
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    }
                }
                // Fast path for Call with simple binary function inlining
                // Uses cached InlineOp to avoid pattern matching on every call
                Call(dst, func_reg, args) => {
                    proc.frames[frame_idx].ip += 1;
                    // Only try fast path for 2-arg calls with cached InlineOp
                    if args.len() == 2 {
                        let func_val = &proc.frames[frame_idx].registers[*func_reg as usize];
                        // Check cached InlineOp - no heap lookup or pattern matching needed!
                        let inline_op = match func_val {
                            GcValue::Closure(_, op) => *op,
                            GcValue::Function(func) => InlineOp::from_function(func),
                            _ => InlineOp::None,
                        };
                        if inline_op != InlineOp::None {
                            let arg0 = &proc.frames[frame_idx].registers[args[0] as usize];
                            let arg1 = &proc.frames[frame_idx].registers[args[1] as usize];
                            if let (GcValue::Int64(x), GcValue::Int64(y)) = (arg0, arg1) {
                                let result = match inline_op {
                                    InlineOp::AddInt => x + y,
                                    InlineOp::SubInt => x - y,
                                    InlineOp::MulInt => x * y,
                                    InlineOp::None => unreachable!(),
                                };
                                proc.frames[frame_idx].registers[*dst as usize] = GcValue::Int64(result);
                                continue; // Stay in fast loop!
                            }
                        }
                    }
                    // Fall back to slow path for non-inlinable calls
                    return Ok(FastLoopResult::NeedSlowPath(instr.clone()));
                }
                // Fast path for tail call to named function
                TailCallDirect(func_idx, args) => {
                    // Get target function from shared state
                    let func = match shared.function_list.get(*func_idx as usize) {
                        Some(f) => f,
                        None => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };

                    // OPTIMIZATION: If calling same function, reuse registers (no heap allocation!)
                    // This is critical for recursive functions like fold
                    let current_func = &proc.frames[frame_idx].function;
                    if Arc::ptr_eq(func, current_func) && args.len() <= 8 {
                        // Same function - reuse registers, no allocation!
                        // Use stack array to avoid heap allocation
                        let mut saved_args: [std::mem::MaybeUninit<GcValue>; 8] =
                            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                        // Save args to stack (take ownership, leave Unit behind)
                        for (i, &r) in args.iter().enumerate() {
                            saved_args[i] = std::mem::MaybeUninit::new(
                                std::mem::take(&mut proc.frames[frame_idx].registers[r as usize])
                            );
                        }

                        // Note: We don't clear remaining registers - they're either
                        // overwritten or unused. GC handles cleanup.

                        // Write args to positions 0, 1, 2, ...
                        let arg_count = args.len();
                        for i in 0..arg_count {
                            proc.frames[frame_idx].registers[i] =
                                unsafe { saved_args[i].assume_init_read() };
                        }

                        // Reset IP to start - continue in same frame
                        proc.frames[frame_idx].ip = 0;
                        // Continue in fast loop
                    } else {
                        // Different function - need new frame
                        let func = Arc::clone(func);

                        // Collect arguments from current frame
                        let arg_values: Vec<GcValue> = args.iter()
                            .map(|&r| proc.frames[frame_idx].registers[r as usize].clone())
                            .collect();

                        // Tail call: replace current frame with new frame
                        let reg_count = func.code.register_count;
                        let mut registers = vec![GcValue::Unit; reg_count];
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            if i < reg_count {
                                registers[i] = arg;
                            }
                        }

                        // Keep the same return_reg (we're replacing, not pushing)
                        let return_reg = proc.frames[frame_idx].return_reg;
                        proc.frames[frame_idx] = CallFrame {
                            function: func,
                            ip: 0,
                            registers,
                            captures: Vec::new(),
                            return_reg,
                        };
                    }
                    // Continue in fast loop with new frame
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
    #[allow(unused, dropping_references)]
    fn execute_one(&mut self, local_id: u64) -> Result<StepResult, RuntimeError> {
        use Instruction::*;

        // Single HashMap lookup - get instruction and increment IP together
        let proc = self.get_process_mut(local_id).unwrap();

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
            let proc = self.get_process_mut(local_id).unwrap();
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
                        // Handle floats (type may not be known at compile time for pattern bindings)
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a + b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a + b),
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
                        // Handle floats (type may not be known at compile time for pattern bindings)
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a - b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a - b),
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
                        // Handle floats (type may not be known at compile time for pattern bindings)
                        (GcValue::Float64(a), GcValue::Float64(b)) => GcValue::Float64(a * b),
                        (GcValue::Float32(a), GcValue::Float32(b)) => GcValue::Float32(a * b),
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
                        (GcValue::Char(a), GcValue::Char(b)) => a < b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                LeInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a <= b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a <= b,
                        (GcValue::Char(a), GcValue::Char(b)) => a <= b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                GtInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a > b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a > b,
                        (GcValue::Char(a), GcValue::Char(b)) => a > b,
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, GcValue::Bool(result));
                    return Ok(StepResult::Continue);
                }
                GeInt(dst, l, r) => {
                    let result = match (fast_reg!(*l), fast_reg!(*r)) {
                        (GcValue::Int64(a), GcValue::Int64(b)) => a >= b,
                        (GcValue::Int32(a), GcValue::Int32(b)) => a >= b,
                        (GcValue::Char(a), GcValue::Char(b)) => a >= b,
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
                        GcValue::List(list) => {
                            list.items().get(idx_val).cloned()
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
                let proc = self.get_process(local_id).unwrap();
                let frame = proc.frames.last().unwrap();
                &frame.registers[$r as usize]
            }};
        }

        macro_rules! set_reg {
            ($r:expr, $v:expr) => {{
                let proc = self.get_process_mut(local_id).unwrap();
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                let proc = self.get_process(local_id).unwrap();
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

            EqFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => x == y,
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
                        let proc = self.get_process(local_id).unwrap();
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
                let proc = self.get_process_mut(local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                frame.ip = (frame.ip as isize + *offset as isize) as usize;
            }

            JumpIfFalse(cond, offset) => {
                let val = reg!(*cond).clone();
                if let GcValue::Bool(false) = val {
                    let proc = self.get_process_mut(local_id).unwrap();
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                }
            }

            JumpIfTrue(cond, offset) => {
                let val = reg!(*cond).clone();
                if let GcValue::Bool(true) = val {
                    let proc = self.get_process_mut(local_id).unwrap();
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                }
            }

            Return(src) => {
                let ret_val = reg!(*src).clone();
                let proc = self.get_process_mut(local_id).unwrap();

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
            Call(dst, callee_reg, args) => {
                let callee = reg!(*callee_reg).clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r).clone()).collect();
                match callee {

                    GcValue::Function(func) => {
                        // Fast path: inline simple binary functions like (a, b) => a + b
                        if arg_values.len() == 2 {
                            let instrs = &func.code.code;
                            // Check for pattern: BinaryOp(dst, 0, 1); Return(dst)
                            if instrs.len() == 2 {
                                if let Instruction::Return(ret_reg) = &instrs[1] {
                                    let result = match &instrs[0] {
                                        Instruction::AddInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) => Some(GcValue::Int64(x + y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::SubInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) => Some(GcValue::Int64(x - y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::MulInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) => Some(GcValue::Int64(x * y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::DivInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) if *y != 0 => Some(GcValue::Int64(x / y)),
                                                _ => None,
                                            }
                                        }
                                        _ => None,
                                    };
                                    if let Some(r) = result {
                                        set_reg!(*dst, r);
                                        return Ok(StepResult::Continue);
                                    }
                                }
                            }
                        }
                        // Normal function call
                        self.call_function(local_id, func, arg_values, Some(*dst))?;
                    }
                    GcValue::Closure(ptr, inline_op) => {
                        let proc = self.get_process(local_id).unwrap();
                        let closure = proc.heap.get_closure(ptr).unwrap();
                        let func = closure.function.clone();
                        let captures = closure.captures.clone();
                        drop(proc);

                        // Fast path: use cached InlineOp for simple binary closures
                        if arg_values.len() == 2 && captures.is_empty() && inline_op != InlineOp::None {
                            if let (GcValue::Int64(x), GcValue::Int64(y)) = (&arg_values[0], &arg_values[1]) {
                                let result = match inline_op {
                                    InlineOp::AddInt => x + y,
                                    InlineOp::SubInt => x - y,
                                    InlineOp::MulInt => x * y,
                                    InlineOp::None => unreachable!(),
                                };
                                let proc = self.get_process_mut(local_id).unwrap();
                                proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Int64(result);
                                return Ok(StepResult::Continue);
                            }
                        }
                        // Fall back to slow path for other closures
                        {
                            let instrs = &func.code.code;
                            // Check for pattern: BinaryOp(dst, 0, 1); Return(dst)
                            if arg_values.len() == 2 && instrs.len() == 2 {
                                if let Instruction::Return(ret_reg) = &instrs[1] {
                                    let result = match &instrs[0] {
                                        Instruction::AddInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) => Some(GcValue::Int64(x + y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::AddFloat(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Float64(x), GcValue::Float64(y)) => Some(GcValue::Float64(x + y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::SubInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) => Some(GcValue::Int64(x - y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::MulInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) => Some(GcValue::Int64(x * y)),
                                                _ => None,
                                            }
                                        }
                                        Instruction::DivInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                                            match (&arg_values[0], &arg_values[1]) {
                                                (GcValue::Int64(x), GcValue::Int64(y)) if *y != 0 => Some(GcValue::Int64(x / y)),
                                                _ => None,
                                            }
                                        }
                                        _ => None,
                                    };
                                    if let Some(r) = result {
                                        set_reg!(*dst, r);
                                        return Ok(StepResult::Continue);
                                    }
                                }
                            }
                        }

                        // Normal closure call
                        self.call_closure(local_id, func, arg_values, captures, Some(*dst))?;
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: format!("{:?}", callee),
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
                    if let Some(&jit_fn) = self.shared.jit_loop_array_functions.get(func_idx) {
                        let arg = reg!(args[0]).clone();
                        if let GcValue::Int64Array(arr_ptr) = arg {
                            // Copy jit_fn to avoid borrow conflict, then get process
                            let proc = self.get_process_mut(local_id).unwrap();
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
                let proc = self.get_process(local_id).unwrap();
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
                            let result_val = GcValue::Int64(result);
                            // For tail call: pop current frame, set result in parent
                            let proc = self.get_process_mut(local_id).unwrap();
                            let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                            proc.frames.pop();
                            // If no more frames, this is the final result
                            if proc.frames.is_empty() {
                                return Ok(StepResult::Finished(result_val));
                            }
                            if let Some(dst) = return_reg {
                                if let Some(parent) = proc.frames.last_mut() {
                                    parent.registers[dst as usize] = result_val;
                                }
                            }
                            return Ok(StepResult::Continue);
                        }
                    }
                    // Check for loop array JIT (only check if we have any registered)
                    if !self.shared.jit_loop_array_functions.is_empty() {
                        if let Some(&jit_fn) = self.shared.jit_loop_array_functions.get(func_idx) {
                            let arg = reg!(args[0]).clone();
                            if let GcValue::Int64Array(arr_ptr) = arg {
                                let proc = self.get_process_mut(local_id).unwrap();
                                if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let result = jit_fn(ptr as *const i64, len);
                                    let result_val = GcValue::Int64(result);
                                    // For tail call: pop current frame, set result in parent
                                    let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                                    proc.frames.pop();
                                    // If no more frames, this is the final result
                                    if proc.frames.is_empty() {
                                        return Ok(StepResult::Finished(result_val));
                                    }
                                    if let Some(dst) = return_reg {
                                        if let Some(parent) = proc.frames.last_mut() {
                                            parent.registers[dst as usize] = result_val;
                                        }
                                    }
                                    return Ok(StepResult::Continue);
                                }
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
                let proc = self.get_process_mut(local_id).unwrap();
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
                    GcValue::Closure(ptr, _) => {
                        let proc = self.get_process(local_id).unwrap();
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

                // Check for trait overrides for "show", "copy", and "hash"
                let trait_method = if !arg_values.is_empty() && (name == "show" || name == "copy" || name == "hash") {
                    let trait_name = match name.as_str() {
                        "show" => "Show",
                        "copy" => "Copy",
                        "hash" => "Hash",
                        _ => unreachable!(),
                    };
                    let proc = self.get_process(local_id).unwrap();
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
                let inline_op = InlineOp::from_function(&func_val);

                let proc = self.get_process_mut(local_id).unwrap();
                let ptr = proc.heap.alloc_closure(func_val, captures, capture_names);
                set_reg!(*dst, GcValue::Closure(ptr, inline_op));
            }

            GetCapture(dst, idx) => {
                let proc = self.get_process(local_id).unwrap();
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
                    GcValue::Closure(ptr, _) => {
                        let proc = self.get_process(local_id).unwrap();
                        let closure = proc.heap.get_closure(ptr).unwrap();
                        (closure.function.clone(), closure.captures.clone())
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: format!("{:?}", func_val),
                    }),
                };

                // Convert args and captures to thread-safe values (deep copy)
                // This ensures heap values are properly copied to the new process
                let proc = self.get_process(local_id).unwrap();
                let safe_args: Vec<ThreadSafeValue> = arg_values.iter()
                    .filter_map(|v| ThreadSafeValue::from_gc_value(v, &proc.heap))
                    .collect();
                let safe_captures: Vec<ThreadSafeValue> = captures.iter()
                    .filter_map(|v| ThreadSafeValue::from_gc_value(v, &proc.heap))
                    .collect();

                // Round-robin distribution across threads
                let num_threads = self.shared.num_threads;
                let spawn_idx = self.shared.spawn_counter.fetch_add(1, Ordering::Relaxed);
                let target_thread = (spawn_idx as usize % num_threads) as u16;

                let child_pid = if target_thread == self.thread_id {
                    // Spawn on this thread - allocate a new local_id
                    let child_local_id = self.next_local_id;
                    self.next_local_id += 1;
                    let child_pid = encode_pid(self.thread_id, child_local_id);

                    // Create process with lightweight heap
                    let mut process = Process::with_gc_config(child_pid, GcConfig::lightweight());

                    // Convert thread-safe values back to GcValues in new heap
                    let gc_args: Vec<GcValue> = safe_args.iter()
                        .map(|v| v.to_gc_value(&mut process.heap))
                        .collect();
                    let gc_captures: Vec<GcValue> = safe_captures.iter()
                        .map(|v| v.to_gc_value(&mut process.heap))
                        .collect();

                    // Set up initial call frame
                    let reg_count = func.code.register_count;
                    let mut registers = vec![GcValue::Unit; reg_count];
                    for (i, arg) in gc_args.into_iter().enumerate() {
                        if i < reg_count {
                            registers[i] = arg;
                        }
                    }

                    let frame = crate::process::CallFrame {
                        function: func,
                        ip: 0,
                        registers,
                        captures: gc_captures,
                        return_reg: None,
                    };
                    process.frames.push(frame);

                    self.insert_process(child_local_id, process);
                    self.run_queue.push_back(child_local_id);

                    child_pid
                } else {
                    // Spawn on another thread
                    // Pre-allocate local_id for the target thread
                    let pre_allocated_local_id = self.shared.thread_local_ids[target_thread as usize]
                        .fetch_add(1, Ordering::Relaxed);
                    let child_pid = encode_pid(target_thread, pre_allocated_local_id);

                    // Send spawn request to target thread
                    if let Some(sender) = self.thread_senders.get(target_thread as usize) {
                        let _ = sender.send(CrossThreadMessage::SpawnProcess {
                            func: func.clone(),
                            args: safe_args,
                            captures: safe_captures,
                            pre_allocated_local_id,
                        });
                    }

                    // Return the pre-computed PID
                    child_pid
                };
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

            Receive(dst) => {
                let proc = self.get_process_mut(local_id).unwrap();
                if let Some(msg) = proc.try_receive() {
                    // Result goes in destination register
                    let frame = proc.frames.last_mut().unwrap();
                    frame.registers[*dst as usize] = msg;
                } else {
                    // No message - block
                    proc.state = ProcessState::Waiting;
                    // Decrement IP so we retry receive next time
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip -= 1;
                    return Ok(StepResult::Waiting);
                }
            }

            ReceiveTimeout(dst, timeout_reg) => {
                // First check if there's a message
                let has_msg = {
                    let proc = self.get_process(local_id).unwrap();
                    proc.has_messages()
                };

                if has_msg {
                    let proc = self.get_process_mut(local_id).unwrap();
                    let msg = proc.try_receive().unwrap();
                    // Message available - put in destination register
                    let frame = proc.frames.last_mut().unwrap();
                    frame.registers[*dst as usize] = msg;
                    // Clear timeout state since message arrived before timeout
                    proc.wake_time = None;
                    proc.timeout_dst = None;
                } else {
                    // No message - check if this is a timeout wake-up or first entry
                    // First get timeout value and state
                    let (timeout_ms, is_first_entry) = {
                        let proc = self.get_process(local_id).unwrap();
                        let frame = proc.frames.last().unwrap();
                        let timeout_ms = match frame.registers[*timeout_reg as usize] {
                            GcValue::Int64(n) => n as u64,
                            _ => return Err(RuntimeError::TypeError {
                                expected: "Int64".to_string(),
                                found: "non-integer".to_string(),
                            }),
                        };
                        let is_first = proc.state == ProcessState::Running && proc.wake_time.is_none();
                        (timeout_ms, is_first)
                    };

                    if is_first_entry {
                        // First entry - set up timeout and wait
                        let wake_time = Instant::now() + Duration::from_millis(timeout_ms);
                        // Push to timer_heap before borrowing proc
                        self.timer_heap.push(Reverse((wake_time, local_id)));
                        // Now update proc state
                        let proc = self.get_process_mut(local_id).unwrap();
                        proc.wake_time = Some(wake_time);
                        proc.timeout_dst = Some(*dst); // Store destination register for check_timers
                        proc.state = ProcessState::WaitingTimeout;
                        // Decrement IP so we retry when woken
                        let frame = proc.frames.last_mut().unwrap();
                        frame.ip -= 1;
                        return Ok(StepResult::Waiting);
                    } else {
                        // Woken by timeout (not by message) - dst already set to Unit by check_timers
                        // Clear timeout_dst since we're done
                        let proc = self.get_process_mut(local_id).unwrap();
                        proc.timeout_dst = None;
                    }
                }
            }

            Sleep(duration_reg) => {
                let proc = self.get_process_mut(local_id).unwrap();
                let frame = proc.frames.last().unwrap();
                let duration_ms = match frame.registers[*duration_reg as usize] {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "non-integer".to_string(),
                    }),
                };

                let wake_time = Instant::now() + Duration::from_millis(duration_ms);
                proc.wake_time = Some(wake_time);
                proc.state = ProcessState::Sleeping;
                self.timer_heap.push(Reverse((wake_time, local_id)));
                return Ok(StepResult::Waiting);
            }

            // === Async I/O ===
            FileReadAll(dst, path_reg) => {
                // Get path from register
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => {
                            if let Some(s) = proc.heap.get_string(*ptr) {
                                s.data.clone()
                            } else {
                                return Err(RuntimeError::IOError("Invalid string pointer".to_string()));
                            }
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };

                // Create oneshot channel for response
                let (tx, rx) = tokio::sync::oneshot::channel();

                // Send request to IO runtime
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileReadToString {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }

                    // Set process to wait for IO
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileWriteAll(dst, path_reg, content_reg) => {
                // Get path and content from registers
                let (path_str, content) = {
                    let proc = self.get_process(local_id).unwrap();
                    let path = match reg!(*path_reg) {
                        GcValue::String(ptr) => {
                            if let Some(s) = proc.heap.get_string(*ptr) {
                                s.data.clone()
                            } else {
                                return Err(RuntimeError::IOError("Invalid string pointer".to_string()));
                            }
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let data = match reg!(*content_reg) {
                        GcValue::String(ptr) => {
                            if let Some(s) = proc.heap.get_string(*ptr) {
                                s.data.as_bytes().to_vec()
                            } else {
                                return Err(RuntimeError::IOError("Invalid string pointer".to_string()));
                            }
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (path, data)
                };

                // Create oneshot channel for response
                let (tx, rx) = tokio::sync::oneshot::channel();

                // Send request to IO runtime
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileWriteAll {
                        path: std::path::PathBuf::from(path_str),
                        data: content,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }

                    // Set process to wait for IO
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpGet(dst, url_reg) => {
                // Get URL from register
                let url = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            if let Some(s) = proc.heap.get_string(*ptr) {
                                s.data.clone()
                            } else {
                                return Err(RuntimeError::IOError("Invalid string pointer".to_string()));
                            }
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };

                // Create oneshot channel for response
                let (tx, rx) = tokio::sync::oneshot::channel();

                // Send request to IO runtime
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpGet {
                        url,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }

                    // Set process to wait for IO
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpPost(dst, url_reg, body_reg) => {
                let (url, body) = {
                    let proc = self.get_process(local_id).unwrap();
                    let url = match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let body = match reg!(*body_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.as_bytes().to_vec())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (url, body)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Post,
                            url,
                            headers: vec![],
                            body: Some(body),
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpPut(dst, url_reg, body_reg) => {
                let (url, body) = {
                    let proc = self.get_process(local_id).unwrap();
                    let url = match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let body = match reg!(*body_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.as_bytes().to_vec())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (url, body)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Put,
                            url,
                            headers: vec![],
                            body: Some(body),
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpDelete(dst, url_reg) => {
                let url = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpRequest {
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
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpPatch(dst, url_reg, body_reg) => {
                let (url, body) = {
                    let proc = self.get_process(local_id).unwrap();
                    let url = match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let body = match reg!(*body_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.as_bytes().to_vec())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (url, body)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpRequest {
                        request: crate::io_runtime::HttpRequest {
                            method: crate::io_runtime::HttpMethod::Patch,
                            url,
                            headers: vec![],
                            body: Some(body),
                            timeout_ms: None,
                        },
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpHead(dst, url_reg) => {
                let url = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpRequest {
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
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            HttpRequest(dst, method_reg, url_reg, headers_reg, body_reg) => {
                let (method, url, headers, body) = {
                    let proc = self.get_process(local_id).unwrap();

                    // Get method string
                    let method_str = match reg!(*method_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
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
                        _ => return Err(RuntimeError::IOError(format!("Unknown HTTP method: {}", method_str))),
                    };

                    // Get URL
                    let url = match reg!(*url_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };

                    // Get headers (list of tuples)
                    let headers = match reg!(*headers_reg) {
                        GcValue::List(list) => {
                            let mut hdrs = vec![];
                            for item in list.items() {
                                if let GcValue::Tuple(ptr) = item {
                                    if let Some(tuple) = proc.heap.get_tuple(*ptr) {
                                        if tuple.items.len() == 2 {
                                            let key = match &tuple.items[0] {
                                                GcValue::String(ptr) => proc.heap.get_string(*ptr).map(|s| s.data.clone()).unwrap_or_default(),
                                                _ => continue,
                                            };
                                            let value = match &tuple.items[1] {
                                                GcValue::String(ptr) => proc.heap.get_string(*ptr).map(|s| s.data.clone()).unwrap_or_default(),
                                                _ => continue,
                                            };
                                            hdrs.push((key, value));
                                        }
                                    }
                                }
                            }
                            hdrs
                        }
                        _ => vec![],
                    };

                    // Get body (optional string)
                    let body = match reg!(*body_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.as_bytes().to_vec())
                        }
                        GcValue::Unit => None,
                        _ => None,
                    };

                    (method, url, headers, body)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::HttpRequest {
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
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === HTTP Server Operations ===
            ServerBind(dst, port_reg) => {
                let port = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*port_reg) {
                        GcValue::Int64(p) => *p as u16,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ServerBind {
                        port,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ServerAccept(dst, handle_reg) => {
                let handle = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (server handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ServerAccept {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ServerRespond(dst, req_id_reg, status_reg, headers_reg, body_reg) => {
                let (request_id, status, headers, body) = {
                    let proc = self.get_process(local_id).unwrap();

                    let request_id = match reg!(*req_id_reg) {
                        GcValue::Int64(id) => *id as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (request id)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };

                    let status = match reg!(*status_reg) {
                        GcValue::Int64(s) => *s as u16,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (status code)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };

                    // Extract headers from list of tuples
                    let headers = match reg!(*headers_reg) {
                        GcValue::List(list) => {
                            let mut hdrs = Vec::new();
                            for item in list.items() {
                                if let GcValue::Tuple(tuple_ptr) = item {
                                    if let Some(tuple) = proc.heap.get_tuple(*tuple_ptr) {
                                        if tuple.items.len() == 2 {
                                            if let (GcValue::String(k_ptr), GcValue::String(v_ptr)) = (&tuple.items[0], &tuple.items[1]) {
                                                let key = proc.heap.get_string(*k_ptr).map(|s| s.data.clone()).unwrap_or_default();
                                                let val = proc.heap.get_string(*v_ptr).map(|s| s.data.clone()).unwrap_or_default();
                                                hdrs.push((key, val));
                                            }
                                        }
                                    }
                                }
                            }
                            hdrs
                        }
                        _ => Vec::new(),
                    };

                    // Extract body as string -> bytes
                    let body = match reg!(*body_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.as_bytes().to_vec()).unwrap_or_default()
                        }
                        _ => Vec::new(),
                    };

                    (request_id, status, headers, body)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ServerRespond {
                        request_id,
                        status,
                        headers,
                        body,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            ServerClose(dst, handle_reg) => {
                let handle = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (server handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ServerClose {
                        handle,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === String Encoding ===
            Base64Encode(dst, str_reg) => {
                use base64::{Engine as _, engine::general_purpose};
                // Extract input string with immutable borrow
                let input = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*str_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let encoded = general_purpose::STANDARD.encode(input.as_bytes());
                let proc = self.get_process_mut(local_id).unwrap();
                let str_ptr = proc.heap.alloc_string(encoded);
                set_reg!(*dst, GcValue::String(str_ptr));
            }

            Base64Decode(dst, str_reg) => {
                use base64::{Engine as _, engine::general_purpose};
                // Extract input string with immutable borrow
                let input = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*str_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let proc = self.get_process_mut(local_id).unwrap();
                match general_purpose::STANDARD.decode(&input) {
                    Ok(bytes) => {
                        match String::from_utf8(bytes) {
                            Ok(decoded) => {
                                let str_ptr = proc.heap.alloc_string(decoded);
                                let ok_str = proc.heap.alloc_string("ok".to_string());
                                let tuple = proc.heap.alloc_tuple(vec![GcValue::String(ok_str), GcValue::String(str_ptr)]);
                                set_reg!(*dst, GcValue::Tuple(tuple));
                            }
                            Err(e) => {
                                let err_str = proc.heap.alloc_string("error".to_string());
                                let msg_str = proc.heap.alloc_string(format!("Invalid UTF-8: {}", e));
                                let tuple = proc.heap.alloc_tuple(vec![GcValue::String(err_str), GcValue::String(msg_str)]);
                                set_reg!(*dst, GcValue::Tuple(tuple));
                            }
                        }
                    }
                    Err(e) => {
                        let err_str = proc.heap.alloc_string("error".to_string());
                        let msg_str = proc.heap.alloc_string(format!("Base64 decode error: {}", e));
                        let tuple = proc.heap.alloc_tuple(vec![GcValue::String(err_str), GcValue::String(msg_str)]);
                        set_reg!(*dst, GcValue::Tuple(tuple));
                    }
                }
            }

            UrlEncode(dst, str_reg) => {
                use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
                // Extract input string with immutable borrow
                let input = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*str_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let encoded = utf8_percent_encode(&input, NON_ALPHANUMERIC).to_string();
                let proc = self.get_process_mut(local_id).unwrap();
                let str_ptr = proc.heap.alloc_string(encoded);
                set_reg!(*dst, GcValue::String(str_ptr));
            }

            UrlDecode(dst, str_reg) => {
                use percent_encoding::percent_decode_str;
                // Extract input string with immutable borrow
                let input = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*str_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let proc = self.get_process_mut(local_id).unwrap();
                match percent_decode_str(&input).decode_utf8() {
                    Ok(decoded) => {
                        let str_ptr = proc.heap.alloc_string(decoded.to_string());
                        let ok_str = proc.heap.alloc_string("ok".to_string());
                        let tuple = proc.heap.alloc_tuple(vec![GcValue::String(ok_str), GcValue::String(str_ptr)]);
                        set_reg!(*dst, GcValue::Tuple(tuple));
                    }
                    Err(e) => {
                        let err_str = proc.heap.alloc_string("error".to_string());
                        let msg_str = proc.heap.alloc_string(format!("URL decode error: {}", e));
                        let tuple = proc.heap.alloc_tuple(vec![GcValue::String(err_str), GcValue::String(msg_str)]);
                        set_reg!(*dst, GcValue::Tuple(tuple));
                    }
                }
            }

            Utf8Encode(dst, str_reg) => {
                // Convert string to list of bytes (Int64)
                // Extract input string with immutable borrow
                let input = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*str_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let bytes: Vec<GcValue> = input.as_bytes().iter().map(|b| GcValue::Int64(*b as i64)).collect();
                set_reg!(*dst, GcValue::List(GcList { data: Arc::new(bytes), start: 0 }));
            }

            Utf8Decode(dst, bytes_reg) => {
                // Convert list of bytes to string
                // Extract bytes list with immutable borrow
                let bytes = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*bytes_reg) {
                        GcValue::List(list) => {
                            let mut bytes = Vec::new();
                            for item in list.items() {
                                match item {
                                    GcValue::Int64(n) => bytes.push(*n as u8),
                                    _ => return Err(RuntimeError::TypeError {
                                        expected: "Int".to_string(),
                                        found: "non-int in list".to_string(),
                                    }),
                                }
                            }
                            bytes
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "non-list".to_string(),
                        }),
                    }
                };
                let proc = self.get_process_mut(local_id).unwrap();
                match String::from_utf8(bytes) {
                    Ok(s) => {
                        let str_ptr = proc.heap.alloc_string(s);
                        let ok_str = proc.heap.alloc_string("ok".to_string());
                        let tuple = proc.heap.alloc_tuple(vec![GcValue::String(ok_str), GcValue::String(str_ptr)]);
                        set_reg!(*dst, GcValue::Tuple(tuple));
                    }
                    Err(e) => {
                        let err_str = proc.heap.alloc_string("error".to_string());
                        let msg_str = proc.heap.alloc_string(format!("Invalid UTF-8: {}", e));
                        let tuple = proc.heap.alloc_tuple(vec![GcValue::String(err_str), GcValue::String(msg_str)]);
                        set_reg!(*dst, GcValue::Tuple(tuple));
                    }
                }
            }

            // === File Handle Operations ===
            FileOpen(dst, path_reg, mode_reg) => {
                let (path_str, mode) = {
                    let proc = self.get_process(local_id).unwrap();
                    let path = match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let mode_str = match reg!(*mode_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let mode = match mode_str.as_str() {
                        "r" => crate::io_runtime::FileMode::Read,
                        "w" => crate::io_runtime::FileMode::Write,
                        "a" => crate::io_runtime::FileMode::Append,
                        "rw" => crate::io_runtime::FileMode::ReadWrite,
                        _ => return Err(RuntimeError::IOError(format!("Invalid file mode: {}", mode_str))),
                    };
                    (path, mode)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileOpen {
                        path: std::path::PathBuf::from(path_str),
                        mode,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileWrite(dst, handle_reg, data_reg) => {
                let (handle, data) = {
                    let proc = self.get_process(local_id).unwrap();
                    let handle = match reg!(*handle_reg) {
                        GcValue::Int64(n) => *n as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (file handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    let data = match reg!(*data_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.as_bytes().to_vec())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (handle, data)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileWrite { handle, data, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileRead(dst, handle_reg, size_reg) => {
                let (handle, size) = {
                    let proc = self.get_process(local_id).unwrap();
                    let handle = match reg!(*handle_reg) {
                        GcValue::Int64(n) => *n as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (file handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    let size = match reg!(*size_reg) {
                        GcValue::Int64(n) => *n as usize,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    (handle, size)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileRead { handle, size, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileReadLine(dst, handle_reg) => {
                let handle = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(n) => *n as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (file handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileReadLine { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileFlush(dst, handle_reg) => {
                let handle = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(n) => *n as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (file handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileFlush { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileClose(dst, handle_reg) => {
                let handle = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(n) => *n as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (file handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileClose { handle, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileSeek(dst, handle_reg, offset_reg, whence_reg) => {
                let (handle, offset, whence) = {
                    let proc = self.get_process(local_id).unwrap();
                    let handle = match reg!(*handle_reg) {
                        GcValue::Int64(n) => *n as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (file handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    let offset = match reg!(*offset_reg) {
                        GcValue::Int64(n) => *n,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    let whence_str = match reg!(*whence_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
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
                    (handle, offset, whence)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileSeek { handle, offset, whence, response: tx };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileAppend(dst, path_reg, content_reg) => {
                let (path_str, content) = {
                    let proc = self.get_process(local_id).unwrap();
                    let path = match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let data = match reg!(*content_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.as_bytes().to_vec())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (path, data)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileAppend {
                        path: std::path::PathBuf::from(path_str),
                        data: content,
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === Directory Operations ===
            DirCreate(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::DirCreate {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirCreateAll(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::DirCreateAll {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirList(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::DirList {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirRemove(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::DirRemove {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirRemoveAll(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::DirRemoveAll {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === File Utilities ===
            FileExists(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileExists {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            DirExists(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::DirExists {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileRemove(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileRemove {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileRename(dst, old_path_reg, new_path_reg) => {
                let (old_path, new_path) = {
                    let proc = self.get_process(local_id).unwrap();
                    let old = match reg!(*old_path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let new = match reg!(*new_path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (old, new)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileRename {
                        old_path: std::path::PathBuf::from(old_path),
                        new_path: std::path::PathBuf::from(new_path),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileCopy(dst, src_path_reg, dest_path_reg) => {
                let (src_path, dest_path) = {
                    let proc = self.get_process(local_id).unwrap();
                    let src = match reg!(*src_path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let dest = match reg!(*dest_path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    (src, dest)
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileCopy {
                        src_path: std::path::PathBuf::from(src_path),
                        dest_path: std::path::PathBuf::from(dest_path),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            FileSize(dst, path_reg) => {
                let path_str = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*path_reg) {
                        GcValue::String(ptr) => proc.heap.get_string(*ptr)
                            .map(|s| s.data.clone())
                            .ok_or_else(|| RuntimeError::IOError("Invalid string pointer".to_string()))?,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };
                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::FileSize {
                        path: std::path::PathBuf::from(path_str),
                        response: tx,
                    };
                    if sender.send(request).is_err() {
                        return Err(RuntimeError::IOError("IO runtime shutdown".to_string()));
                    }
                    let proc = self.get_process_mut(local_id).unwrap();
                    proc.start_io_wait(rx, *dst);
                    self.io_waiting.push(local_id);
                    return Ok(StepResult::Waiting);
                } else {
                    return Err(RuntimeError::IOError("IO runtime not available".to_string()));
                }
            }

            // === I/O ===
            Println(src) => {
                let proc = self.get_process(local_id).unwrap();
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
                let proc = self.get_process(local_id).unwrap();
                if !proc.heap.gc_values_equal(&va, &vb) {
                    let sa = proc.heap.display_value(&va);
                    let sb = proc.heap.display_value(&vb);
                    return Err(RuntimeError::Panic(format!("Assertion failed: {} != {}", sa, sb)));
                }
            }

            Panic(msg_reg) => {
                let msg = reg!(*msg_reg).clone();
                let proc = self.get_process(local_id).unwrap();
                let msg_str = proc.heap.display_value(&msg);
                return Err(RuntimeError::Panic(msg_str));
            }

            Nop => {}

            // === Pattern Matching ===
            TestConst(dst, value_reg, const_idx) => {
                let value = reg!(*value_reg).clone();
                let constant = &constants[*const_idx as usize];
                let proc = self.get_process_mut(local_id).unwrap();
                let gc_const = proc.heap.value_to_gc(constant);
                let result = proc.heap.gc_values_equal(&value, &gc_const);
                proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Bool(result);
            }

            TestNil(dst, list) => {
                let list_val = reg!(*list).clone();
                let result = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            TestUnit(dst, value) => {
                let val = reg!(*value).clone();
                let result = matches!(val, GcValue::Unit);
                set_reg!(*dst, GcValue::Bool(result));
            }

            TestTag(dst, value, ctor_idx) => {
                // Compare constructor name exactly (stored in constants)
                let expected_ctor = match &constants[*ctor_idx as usize] {
                    Value::String(s) => s.as_str(),
                    _ => return Err(RuntimeError::Panic("TestTag: expected string constant".to_string())),
                };
                let value_clone = reg!(*value).clone();
                let result = match &value_clone {
                    GcValue::Variant(ptr) => {
                        let proc = self.get_process(local_id).unwrap();
                        proc.heap.get_variant(*ptr).map(|v| v.constructor.as_str() == expected_ctor).unwrap_or(false)
                    }
                    GcValue::Record(ptr) => {
                        // Records: compare type_name directly
                        let proc = self.get_process(local_id).unwrap();
                        proc.heap.get_record(*ptr).map(|r| r.type_name.as_str() == expected_ctor).unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            MakeVariant(dst, type_idx, ctor_idx, ref field_regs) => {
                let type_name = match &constants[*type_idx as usize] {
                    Value::String(s) => Arc::clone(s),
                    _ => return Err(RuntimeError::Panic("Variant type must be string".to_string())),
                };
                let constructor = match &constants[*ctor_idx as usize] {
                    Value::String(s) => Arc::clone(s),
                    _ => return Err(RuntimeError::Panic("Variant constructor must be string".to_string())),
                };
                let fields: Vec<GcValue> = field_regs.iter().map(|&r| reg!(r).clone()).collect();
                let proc = self.get_process_mut(local_id).unwrap();
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
                        let proc = self.get_process(local_id).unwrap();
                        let rec = proc.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        let value = rec.fields[idx].clone();
                        set_reg!(*dst, value);
                    }
                    GcValue::Variant(ptr) => {
                        let proc = self.get_process(local_id).unwrap();
                        let var = proc.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid variant field index: {}", field_name)))?;
                        let value = var.fields.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of range", idx)))?
                            .clone();
                        set_reg!(*dst, value);
                    }
                    GcValue::Tuple(ptr) => {
                        // Support tuple field access with numeric indices (t.0, t.1, etc.)
                        let proc = self.get_process(local_id).unwrap();
                        let tuple = proc.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".to_string()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid tuple index: {}", field_name)))?;
                        let value = tuple.items.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Tuple index {} out of bounds", idx)))?
                            .clone();
                        set_reg!(*dst, value);
                    }
                    _ => return Err(RuntimeError::Panic("GetField expects record, variant, or tuple".to_string())),
                }
            }

            GetVariantField(dst, src, idx) => {
                let src_val = reg!(*src).clone();
                let value = match src_val {
                    GcValue::Variant(ptr) => {
                        let proc = self.get_process(local_id).unwrap();
                        let variant = proc.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        variant.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of bounds", idx)))?
                    }
                    GcValue::Record(ptr) => {
                        let proc = self.get_process(local_id).unwrap();
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                let proc = self.get_process(local_id).unwrap();
                let len = match val {
                    GcValue::List(list) => list.len(),
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
                let proc = self.get_process_mut(local_id).unwrap();
                let list = proc.heap.make_list(items);
                set_reg!(*dst, GcValue::List(list));
            }

            MakeTuple(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| reg!(r).clone()).collect();
                let proc = self.get_process_mut(local_id).unwrap();
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
                let proc = self.get_process(local_id).unwrap();
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
                let proc = self.get_process_mut(local_id).unwrap();
                let ptr = proc.heap.alloc_record(type_name, field_names, fields, mutable_fields);
                set_reg!(*dst, GcValue::Record(ptr));
            }

            Cons(dst, head, tail) => {
                let head_val = reg!(*head).clone();
                let tail_val = reg!(*tail).clone();
                let proc = self.get_process_mut(local_id).unwrap();
                match tail_val {
                    GcValue::List(tail_list) => {
                        let mut items = vec![head_val];
                        items.extend(tail_list.items().iter().cloned());
                        let new_list = proc.heap.make_list(items);
                        set_reg!(*dst, GcValue::List(new_list));
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
                let is_empty = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(is_empty));
            }

            ListSum(dst, list_reg) => {
                let list_val = reg!(*list_reg).clone();
                let proc = self.get_process(local_id).unwrap();
                let sum = match &list_val {
                    GcValue::List(list) => {
                        let mut total: i64 = 0;
                        for item in list.items() {
                            match item {
                                GcValue::Int64(n) => total += n,
                                _ => return Err(RuntimeError::TypeError {
                                    expected: "Int64".to_string(),
                                    found: item.type_name(&proc.heap).to_string(),
                                }),
                            }
                        }
                        total
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: list_val.type_name(&proc.heap).to_string(),
                    }),
                };
                set_reg!(*dst, GcValue::Int64(sum));
            }

            ListHead(dst, list_reg) => {
                let list_val = reg!(*list_reg).clone();
                let result = match &list_val {
                    GcValue::List(list) => {
                        if let Some(head) = list.items().first() {
                            Ok(head.clone())
                        } else {
                            Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 })
                        }
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: format!("{:?}", list_val),
                    }),
                }?;
                set_reg!(*dst, result);
            }

            ListTail(dst, list_reg) => {
                let list_val = reg!(*list_reg).clone();
                match list_val {
                    GcValue::List(list) => {
                        if list.is_empty() {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                        let tail_list = list.tail();
                        set_reg!(*dst, GcValue::List(tail_list));
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
                let proc = self.get_process_mut(local_id).unwrap();
                let ptr = proc.heap.alloc_int64_array(items);
                set_reg!(*dst, GcValue::Int64Array(ptr));
            }

            MakeFloat64Array(dst, size_reg) => {
                let size = match reg!(*size_reg) {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::Panic("Array size must be Int64".to_string())),
                };
                let items = vec![0.0f64; size];
                let proc = self.get_process_mut(local_id).unwrap();
                let ptr = proc.heap.alloc_float64_array(items);
                set_reg!(*dst, GcValue::Float64Array(ptr));
            }

            Index(dst, coll, idx) => {
                let idx_val = match reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                };
                let coll_val = reg!(*coll).clone();
                let proc = self.get_process(local_id).unwrap();
                let value = match coll_val {
                    GcValue::List(list) => {
                        list.items().get(idx_val).cloned()
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                        let proc = self.get_process_mut(local_id).unwrap();
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
                let list_data = match list_val {
                    GcValue::List(list) => list,
                    _ => return Err(RuntimeError::Panic("Decons expects list".to_string())),
                };
                let items = list_data.items();
                if items.is_empty() {
                    return Err(RuntimeError::Panic("Cannot decons empty list".to_string()));
                }
                let head = items[0].clone();
                let tail_list = list_data.tail();
                set_reg!(*head_dst, head);
                set_reg!(*tail_dst, GcValue::List(tail_list));
            }

            Concat(dst, a, b) => {
                let a_val = reg!(*a).clone();
                let b_val = reg!(*b).clone();
                match (a_val, b_val) {
                    (GcValue::String(a_ptr), GcValue::String(b_ptr)) => {
                        let proc = self.get_process_mut(local_id).unwrap();
                        let a_str = proc.heap.get_string(a_ptr).map(|s| s.data.as_str()).unwrap_or("");
                        let b_str = proc.heap.get_string(b_ptr).map(|s| s.data.as_str()).unwrap_or("");
                        let result = format!("{}{}", a_str, b_str);
                        let result_ptr = proc.heap.alloc_string(result);
                        set_reg!(*dst, GcValue::String(result_ptr));
                    }
                    (GcValue::List(a_list), GcValue::List(b_list)) => {
                        let mut new_items = a_list.items().to_vec();
                        new_items.extend(b_list.items().iter().cloned());
                        let proc = self.get_process_mut(local_id).unwrap();
                        let result = proc.heap.make_list(new_items);
                        set_reg!(*dst, GcValue::List(result));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String or List".to_string(),
                        found: "other".to_string(),
                    }),
                };
            }

            Throw(src) => {
                let exception = reg!(*src).clone();
                let proc = self.get_process_mut(local_id).unwrap();
                proc.current_exception = Some(exception);

                // Find the most recent handler
                if let Some(handler) = proc.handlers.pop() {
                    // Unwind stack to handler's frame
                    while proc.frames.len() > handler.frame_index + 1 {
                        proc.frames.pop();
                    }
                    // Jump to catch block
                    proc.frames[handler.frame_index].ip = handler.catch_ip;
                } else {
                    // No handler - propagate as runtime error
                    let proc = self.get_process(local_id).unwrap();
                    let msg = proc.heap.display_value(proc.current_exception.as_ref().unwrap());
                    return Err(RuntimeError::Panic(format!("Uncaught exception: {}", msg)));
                }
            }

            // === I/O ===
            Print(dst, src) => {
                let val = reg!(*src).clone();
                let proc = self.get_process_mut(local_id).unwrap();
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

            And(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Bool(x), GcValue::Bool(y)) => {
                        set_reg!(*dst, GcValue::Bool(*x && *y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            Or(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                match (&va, &vb) {
                    (GcValue::Bool(x), GcValue::Bool(y)) => {
                        set_reg!(*dst, GcValue::Bool(*x || *y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: format!("{:?}", va),
                    }),
                }
            }

            // === Exception handling ===
            PushHandler(offset) => {
                let proc = self.get_process_mut(local_id).unwrap();
                let frame_index = proc.frames.len() - 1;
                let catch_ip = (proc.frames[frame_index].ip as isize + *offset as isize) as usize;
                proc.handlers.push(crate::process::ExceptionHandler {
                    frame_index,
                    catch_ip,
                });
            }

            PopHandler => {
                let proc = self.get_process_mut(local_id).unwrap();
                proc.handlers.pop();
            }

            GetException(dst) => {
                let proc = self.get_process(local_id).unwrap();
                let exception = proc.current_exception.clone().unwrap_or(GcValue::Unit);
                set_reg!(*dst, exception);
            }

                        // === Collection literals ===

                        MakeSet(dst, ref elements) => {

                            let mut items = std::collections::HashSet::new();

                            for &r in elements.iter() {

                                let val = reg!(r).clone();

                                let proc = self.get_process(local_id).unwrap();

                                if let Some(key) = val.to_gc_map_key(&proc.heap) {

                                    items.insert(key);

                                } else {

                                    return Err(RuntimeError::TypeError {

                                        expected: "hashable type".to_string(),

                                        found: format!("{:?}", val),

                                    });

                                }

                            }

                            let proc = self.get_process_mut(local_id).unwrap();

                            let ptr = proc.heap.alloc_set(items);

                            set_reg!(*dst, GcValue::Set(ptr));

                        }

            

                        MakeMap(dst, ref entries) => {

                            let mut map = std::collections::HashMap::new();

                            for (key_reg, val_reg) in entries.iter() {

                                let key_val = reg!(*key_reg).clone();

                                let val = reg!(*val_reg).clone();

                                let proc = self.get_process(local_id).unwrap();

                                if let Some(key) = key_val.to_gc_map_key(&proc.heap) {

                                    map.insert(key, val);

                                } else {

                                    return Err(RuntimeError::TypeError {

                                        expected: "hashable type".to_string(),

                                        found: format!("{:?}", key_val),

                                    });

                                }

                            }

                            let proc = self.get_process_mut(local_id).unwrap();

                            let ptr = proc.heap.alloc_map(map);

                            set_reg!(*dst, GcValue::Map(ptr));

                        }

            

                        Decons(head, tail, list) => {

                            let list_val = reg!(*list).clone();

                            if let GcValue::List(l) = list_val {

                                if let Some(h) = l.head() {

                                    let t = l.tail();

                                    // set_reg! macro might not handle multiple sets?

                                    // set_reg! uses self.get_process_mut(local_id).unwrap()

                                    // calling it twice is fine (scopes are separate statements)

                                    set_reg!(*head, h.clone());

                                    set_reg!(*tail, GcValue::List(t));

                                } else {

                                    return Err(RuntimeError::Panic("Cannot decons empty list".to_string()));

                                }

                            } else {

                                let proc = self.get_process(local_id).unwrap();

                                return Err(RuntimeError::TypeError {

                                    expected: "List".to_string(),

                                    found: list_val.type_name(&proc.heap).to_string(),

                                });

                            }

                        }

            

                        IsMap(dst, src) => {

                            let val = reg!(*src).clone();

                            let is_map = matches!(val, GcValue::Map(_));

                            set_reg!(*dst, GcValue::Bool(is_map));

                        }

            

                        IsSet(dst, src) => {

                            let val = reg!(*src).clone();

                            let is_set = matches!(val, GcValue::Set(_));

                            set_reg!(*dst, GcValue::Bool(is_set));

                        }

            

                        MapContainsKey(dst, map_reg, key_reg) => {

                            let map_val = reg!(*map_reg).clone();

                            let key_val = reg!(*key_reg).clone();

                            

                            let result = {

                                let proc = self.get_process(local_id).unwrap();

                                if let GcValue::Map(ptr) = &map_val {

                                    if let Some(map) = proc.heap.get_map(*ptr) {

                                        if let Some(key) = key_val.to_gc_map_key(&proc.heap) {

                                            map.entries.contains_key(&key)

                                        } else {

                                            false

                                        }

                                    } else {

                                        false

                                    }

                                }

                                else {

                                    false

                                }

                            };

                            

                            if let GcValue::Map(_) = map_val {

                                set_reg!(*dst, GcValue::Bool(result));

                            } else {

                                let proc = self.get_process(local_id).unwrap();

                                return Err(RuntimeError::TypeError {

                                    expected: "Map".to_string(),

                                    found: map_val.type_name(&proc.heap).to_string(),

                                });

                            }

                        }

            

                        MapGet(dst, map_reg, key_reg) => {

                            let map_val = reg!(*map_reg).clone();

                            let key_val = reg!(*key_reg).clone();

                            

                            let result = {

                                let proc = self.get_process(local_id).unwrap();

                                if let GcValue::Map(ptr) = &map_val {

                                    if let Some(map) = proc.heap.get_map(*ptr) {

                                        if let Some(key) = key_val.to_gc_map_key(&proc.heap) {

                                            map.entries.get(&key).cloned()

                                        } else {

                                            None

                                        }

                                    }

                                    else {

                                        None

                                    }

                                }

                                else {

                                    None

                                }

                            };

                            

                            if let GcValue::Map(_) = map_val {

                                match result {

                                    Some(val) => set_reg!(*dst, val),

                                    None => return Err(RuntimeError::Panic("Key not found in map".to_string())),

                                }

                            } else {

                                let proc = self.get_process(local_id).unwrap();

                                return Err(RuntimeError::TypeError {

                                    expected: "Map".to_string(),

                                    found: map_val.type_name(&proc.heap).to_string(),

                                });

                            }

                        }

            

                        SetContains(dst, set_reg, val_reg) => {

                            let set_val = reg!(*set_reg).clone();

                            let elem_val = reg!(*val_reg).clone();

                            

                            let result = {

                                let proc = self.get_process(local_id).unwrap();

                                if let GcValue::Set(ptr) = &set_val {

                                    if let Some(set) = proc.heap.get_set(*ptr) {

                                        if let Some(key) = elem_val.to_gc_map_key(&proc.heap) {

                                            set.items.contains(&key)

                                        }

                                        else {

                                            false

                                        }

                                    }

                                    else {

                                        false

                                    }

                                }

                                else {

                                    false

                                }

                            };

                            

                            if let GcValue::Set(_) = set_val {

                                set_reg!(*dst, GcValue::Bool(result));

                            } else {

                                let proc = self.get_process(local_id).unwrap();

                                return Err(RuntimeError::TypeError {

                                    expected: "Set".to_string(),

                                    found: set_val.type_name(&proc.heap).to_string(),

                                });

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
        let proc = self.get_process_mut(local_id).unwrap();
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

        let proc = self.get_process_mut(local_id).unwrap();
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

        let proc = self.get_process_mut(local_id).unwrap();
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

        let proc = self.get_process_mut(local_id).unwrap();

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

        let proc = self.get_process_mut(local_id).unwrap();

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

        let proc = self.get_process_mut(local_id).unwrap();
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
    /// Completed all iterations
    Continue,
    /// Process finished with a value
    Finished(GcValue),
    /// Need slow path for this instruction
    NeedSlowPath(Instruction),
}
