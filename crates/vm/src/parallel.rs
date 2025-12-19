//! Parallel VM with CPU affinity for Erlang-style concurrency.
//!
//! **DEPRECATED**: This is the old, non-maintained VM implementation.
//! Use `AsyncVM` from `async_vm.rs` instead, which is the current default.
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

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Sender, Receiver, TryRecvError};
use imbl::HashMap as ImblHashMap;
use imbl::HashSet as ImblHashSet;
use parking_lot::RwLock;
use smallvec::smallvec;

// Thread-local storage for tracking which mvar locks are held by the current thread.
// This allows MvarRead/MvarWrite to skip locking if MvarLock already holds the lock.
// Key: mvar name, Value: true if write lock, false if read lock
thread_local! {
    static HELD_MVAR_LOCKS: RefCell<HashMap<String, bool>> = RefCell::new(HashMap::new());
}

use tokio::sync::mpsc as tokio_mpsc;

use crate::gc::{GcConfig, GcList, GcMapKey, GcNativeFn, GcValue, Heap, InlineOp};
use crate::io_runtime::{IoRequest, IoRuntime};
use crate::process::{CallFrame, ExitReason, IoResponseValue, Process, ProcessState, ThreadSafeMapKey, ThreadSafeValue};
use crate::shared_types::{SharedMap, SharedMapKey, SharedMapValue};
use crate::value::{FunctionValue, Instruction, Pid, RuntimeError, TypeValue, Value};

/// An entry for the inspect queue - a named value to display in the inspector
#[derive(Debug, Clone)]
pub struct InspectEntry {
    /// Tab name for the inspector
    pub name: String,
    /// The value (deep-copied for thread safety)
    pub value: ThreadSafeValue,
}

/// Type alias for the inspect sender channel
pub type InspectSender = crossbeam::channel::Sender<InspectEntry>;
/// Type alias for the inspect receiver channel
pub type InspectReceiver = crossbeam::channel::Receiver<InspectEntry>;

/// Type alias for the output sender channel (for println from any process)
pub type OutputSender = crossbeam::channel::Sender<String>;
/// Type alias for the output receiver channel
pub type OutputReceiver = crossbeam::channel::Receiver<String>;

/// Panel command - sent from Panel.* native functions to the TUI
#[derive(Clone, Debug)]
pub enum PanelCommand {
    /// Create a new panel with given ID and title
    Create { id: u64, title: String },
    /// Set the content of a panel
    SetContent { id: u64, content: String },
    /// Show a panel
    Show { id: u64 },
    /// Hide a panel
    Hide { id: u64 },
    /// Register a key handler for a panel
    OnKey { id: u64, handler_fn: String },
    /// Register a global hotkey that triggers a callback
    RegisterHotkey { key: String, callback_fn: String },
}

/// Type alias for the panel command sender channel
pub type PanelCommandSender = crossbeam::channel::Sender<PanelCommand>;
/// Type alias for the panel command receiver channel
pub type PanelCommandReceiver = crossbeam::channel::Receiver<PanelCommand>;

/// Command for evaluating code from Nostos via the main thread (which has the compiler).
/// Uses a request-response pattern with a reply channel.
pub struct EvalCommand {
    /// The code to evaluate
    pub code: String,
    /// Channel to send the result back to the caller
    pub reply: crossbeam::channel::Sender<EvalResult>,
}

/// Result of an eval command
pub enum EvalResult {
    /// Successful evaluation with string representation of the result
    Ok(String),
    /// Compilation or runtime error message
    Err(String),
}

/// Type alias for the eval command sender channel
pub type EvalCommandSender = crossbeam::channel::Sender<EvalCommand>;
/// Type alias for the eval command receiver channel
pub type EvalCommandReceiver = crossbeam::channel::Receiver<EvalCommand>;

// JIT function pointer types (moved from runtime.rs)
pub type JitIntFn = fn(i64) -> i64;
pub type JitIntFn0 = fn() -> i64;
pub type JitIntFn2 = fn(i64, i64) -> i64;
pub type JitIntFn3 = fn(i64, i64, i64) -> i64;
pub type JitIntFn4 = fn(i64, i64, i64, i64) -> i64;
pub type JitLoopArrayFn = fn(*const i64, i64) -> i64;
/// Recursive array fill function: (arr_ptr, len, idx) -> i64 (unit)
pub type JitArrayFillFn = fn(*const i64, i64, i64) -> i64;
/// Recursive array sum function: (arr_ptr, len, idx, acc) -> i64
pub type JitArraySumFn = fn(*const i64, i64, i64, i64) -> i64;

/// Result from running a function, including captured output.
#[derive(Debug)]
pub struct RunResult {
    /// The return value from the function
    pub value: Option<SendableValue>,
    /// Captured output (from println, print, etc.)
    pub output: Vec<String>,
}

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
    /// JIT-compiled recursive array fill functions (func_index → native fn)
    pub jit_array_fill_functions: HashMap<u16, JitArrayFillFn>,
    /// JIT-compiled recursive array sum functions (func_index → native fn)
    pub jit_array_sum_functions: HashMap<u16, JitArraySumFn>,
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
    /// Inspect sender (for sending values to TUI inspector)
    pub inspect_sender: Option<InspectSender>,
    /// Output sender (for println from any process to TUI console)
    pub output_sender: Option<OutputSender>,
    /// Panel command sender (for Panel.* calls from Nostos code)
    pub panel_command_sender: Option<PanelCommandSender>,
    /// Eval callback for synchronous code evaluation (set by engine)
    /// Wrapped in RwLock so the native function can read it at call time
    pub eval_callback: Arc<RwLock<Option<Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>>>>,
    /// Dynamically added functions (from eval) - can be modified at runtime
    pub dynamic_functions: Arc<RwLock<HashMap<String, Arc<FunctionValue>>>>,
    /// Dynamically added types (from eval) - can be modified at runtime
    pub dynamic_types: Arc<RwLock<HashMap<String, Arc<TypeValue>>>>,
    /// Stdlib functions (from main compiler) - available to eval
    pub stdlib_functions: Arc<RwLock<HashMap<String, Arc<FunctionValue>>>>,
    /// Stdlib types (from main compiler) - available to eval
    pub stdlib_types: Arc<RwLock<HashMap<String, Arc<TypeValue>>>>,
    /// Stdlib function list (ordered names) - preserves indices for CallDirect
    pub stdlib_function_list: Arc<RwLock<Vec<String>>>,
    /// Prelude imports (local name -> qualified name) - for eval to resolve unqualified function names
    pub prelude_imports: Arc<RwLock<HashMap<String, String>>>,
    /// Module-level mutable variables (mvars) - shared across threads with RwLock.
    /// Key is "module_name.var_name", value is ThreadSafeValue protected by RwLock.
    pub mvars: HashMap<String, Arc<RwLock<ThreadSafeValue>>>,
    /// Dynamic mvars from eval() - can be added at runtime
    pub dynamic_mvars: Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>>,
}

/// Configuration for the parallel VM (DEPRECATED - use AsyncVM instead).
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

/// The parallel VM entry point (DEPRECATED - use AsyncVM instead).
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
    #[allow(dead_code)]
    next_thread: AtomicU64,
    /// Configuration
    config: ParallelConfig,
    /// IO runtime for async file/HTTP operations
    #[allow(dead_code)]
    io_runtime: Option<IoRuntime>,
}

/// Thread-safe sendable record value.
#[derive(Clone, Debug)]
pub struct SendableRecord {
    pub type_name: String,
    pub field_names: Vec<String>,
    pub fields: Vec<SendableValue>,
}

/// Thread-safe variant value.
#[derive(Clone, Debug)]
pub struct SendableVariant {
    pub type_name: String,
    pub constructor: String,
    pub fields: Vec<SendableValue>,
}

/// Thread-safe sendable map key.
#[derive(Clone, Debug)]
pub enum SendableMapKey {
    Unit,
    Bool(bool),
    Char(char),
    Int64(i64),
    String(String),
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<SendableMapKey>,
    },
    Variant {
        type_name: String,
        constructor: String,
        fields: Vec<SendableMapKey>,
    },
}

impl PartialEq for SendableMapKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SendableMapKey::Unit, SendableMapKey::Unit) => true,
            (SendableMapKey::Bool(a), SendableMapKey::Bool(b)) => a == b,
            (SendableMapKey::Char(a), SendableMapKey::Char(b)) => a == b,
            (SendableMapKey::Int64(a), SendableMapKey::Int64(b)) => a == b,
            (SendableMapKey::String(a), SendableMapKey::String(b)) => a == b,
            (
                SendableMapKey::Record { type_name: tn1, field_names: fn1, fields: f1 },
                SendableMapKey::Record { type_name: tn2, field_names: fn2, fields: f2 },
            ) => tn1 == tn2 && fn1 == fn2 && f1 == f2,
            (
                SendableMapKey::Variant { type_name: tn1, constructor: c1, fields: f1 },
                SendableMapKey::Variant { type_name: tn2, constructor: c2, fields: f2 },
            ) => tn1 == tn2 && c1 == c2 && f1 == f2,
            _ => false,
        }
    }
}

impl Eq for SendableMapKey {}

impl std::hash::Hash for SendableMapKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            SendableMapKey::Unit => {}
            SendableMapKey::Bool(b) => b.hash(state),
            SendableMapKey::Char(c) => c.hash(state),
            SendableMapKey::Int64(n) => n.hash(state),
            SendableMapKey::String(s) => s.hash(state),
            SendableMapKey::Record { type_name, field_names, fields } => {
                type_name.hash(state);
                for name in field_names {
                    name.hash(state);
                }
                for field in fields {
                    field.hash(state);
                }
            }
            SendableMapKey::Variant { type_name, constructor, fields } => {
                type_name.hash(state);
                constructor.hash(state);
                for field in fields {
                    field.hash(state);
                }
            }
        }
    }
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
    Variant(SendableVariant),
    Map(std::collections::HashMap<SendableMapKey, SendableValue>),
    Set(std::collections::HashSet<SendableMapKey>),
    Error(String),
}

impl SendableValue {
    pub fn from_gc_value(value: &GcValue, heap: &Heap) -> Self {
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
            GcValue::Variant(ptr) => {
                if let Some(variant) = heap.get_variant(*ptr) {
                    let fields: Vec<SendableValue> = variant.fields
                        .iter()
                        .map(|v| SendableValue::from_gc_value(v, heap))
                        .collect();
                    SendableValue::Variant(SendableVariant {
                        type_name: variant.type_name.to_string(),
                        constructor: variant.constructor.to_string(),
                        fields,
                    })
                } else {
                    SendableValue::String("<variant>".to_string())
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
            GcValue::SharedMap(shared_map) => {
                // Convert SharedMap entries to SendableValue::Map
                let entries: std::collections::HashMap<SendableMapKey, SendableValue> = shared_map
                    .iter()
                    .filter_map(|(k, v)| {
                        Self::shared_key_to_sendable_key(k)
                            .map(|sk| (sk, Self::shared_value_to_sendable(v, heap)))
                    })
                    .collect();
                SendableValue::Map(entries)
            }
            // For other values, use their display representation
            _ => SendableValue::String(heap.display_value(value)),
        }
    }

    fn gc_map_key_to_sendable(key: &crate::gc::GcMapKey, heap: &Heap) -> Option<SendableMapKey> {
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
            GcMapKey::Record { type_name, field_names, fields } => {
                let sendable_fields: Option<Vec<_>> = fields.iter()
                    .map(|f| Self::gc_map_key_to_sendable(f, heap))
                    .collect();
                sendable_fields.map(|f| SendableMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: f,
                })
            }
            GcMapKey::Variant { type_name, constructor, fields } => {
                let sendable_fields: Option<Vec<_>> = fields.iter()
                    .map(|f| Self::gc_map_key_to_sendable(f, heap))
                    .collect();
                sendable_fields.map(|f| SendableMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: f,
                })
            }
        }
    }

    fn shared_key_to_sendable_key(key: &SharedMapKey) -> Option<SendableMapKey> {
        match key {
            SharedMapKey::Unit => Some(SendableMapKey::Unit),
            SharedMapKey::Bool(b) => Some(SendableMapKey::Bool(*b)),
            SharedMapKey::Char(c) => Some(SendableMapKey::Char(*c)),
            SharedMapKey::Int64(i) => Some(SendableMapKey::Int64(*i)),
            SharedMapKey::Int8(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::Int16(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::Int32(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::UInt8(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::UInt16(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::UInt32(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::UInt64(i) => Some(SendableMapKey::Int64(*i as i64)),
            SharedMapKey::String(s) => Some(SendableMapKey::String(s.clone())),
            SharedMapKey::Record { type_name, field_names, fields } => {
                let sendable_fields: Option<Vec<_>> = fields.iter()
                    .map(Self::shared_key_to_sendable_key)
                    .collect();
                sendable_fields.map(|f| SendableMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: f,
                })
            }
            SharedMapKey::Variant { type_name, constructor, fields } => {
                let sendable_fields: Option<Vec<_>> = fields.iter()
                    .map(Self::shared_key_to_sendable_key)
                    .collect();
                sendable_fields.map(|f| SendableMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: f,
                })
            }
        }
    }

    fn shared_value_to_sendable(value: &SharedMapValue, heap: &Heap) -> SendableValue {
        match value {
            SharedMapValue::Unit => SendableValue::Unit,
            SharedMapValue::Bool(b) => SendableValue::Bool(*b),
            SharedMapValue::Int64(i) => SendableValue::Int64(*i),
            SharedMapValue::Float64(f) => SendableValue::Float64(*f),
            SharedMapValue::Pid(p) => SendableValue::Pid(*p),
            SharedMapValue::String(s) => SendableValue::String(s.clone()),
            SharedMapValue::Char(c) => SendableValue::Char(*c),
            SharedMapValue::List(items) => {
                SendableValue::List(items.iter().map(|v| Self::shared_value_to_sendable(v, heap)).collect())
            }
            SharedMapValue::Tuple(items) => {
                SendableValue::Tuple(items.iter().map(|v| Self::shared_value_to_sendable(v, heap)).collect())
            }
            SharedMapValue::Record { type_name, field_names, fields } => {
                SendableValue::Record(SendableRecord {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|v| Self::shared_value_to_sendable(v, heap)).collect(),
                })
            }
            SharedMapValue::Variant { type_name, constructor, fields } => {
                SendableValue::Variant(SendableVariant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|v| Self::shared_value_to_sendable(v, heap)).collect(),
                })
            }
            SharedMapValue::Map(map) => {
                let entries: std::collections::HashMap<SendableMapKey, SendableValue> = map
                    .iter()
                    .filter_map(|(k, v)| {
                        Self::shared_key_to_sendable_key(k)
                            .map(|sk| (sk, Self::shared_value_to_sendable(v, heap)))
                    })
                    .collect();
                SendableValue::Map(entries)
            }
            SharedMapValue::Set(items) => {
                SendableValue::Set(items.iter().filter_map(Self::shared_key_to_sendable_key).collect())
            }
            // Convert typed arrays to lists of their element types
            SharedMapValue::Int64Array(items) => {
                SendableValue::List(items.iter().map(|i| SendableValue::Int64(*i)).collect())
            }
            SharedMapValue::Float64Array(items) => {
                SendableValue::List(items.iter().map(|f| SendableValue::Float64(*f)).collect())
            }
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
            SendableValue::Variant(v) => {
                if v.fields.is_empty() {
                    v.constructor.clone()
                } else {
                    let fields_str: Vec<String> = v.fields.iter().map(|f| f.display()).collect();
                    format!("{}({})", v.constructor, fields_str.join(", "))
                }
            }
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
            SendableValue::Variant(v) => {
                let fields: Vec<Value> = v.fields.iter().map(|f| f.to_value()).collect();
                Value::Variant(Arc::new(crate::value::VariantValue {
                    type_name: Arc::new(v.type_name.clone()),
                    constructor: Arc::new(v.constructor.clone()),
                    fields,
                    named_fields: None,
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
            SendableMapKey::Record { type_name, field_names, fields } => {
                MapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(Self::sendable_key_to_map_key).collect(),
                }
            }
            SendableMapKey::Variant { type_name, constructor, fields } => {
                MapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(Self::sendable_key_to_map_key).collect(),
                }
            }
        }
    }

    /// Convert to ThreadSafeValue for storage in mvars.
    pub fn to_thread_safe(&self) -> ThreadSafeValue {
        match self {
            SendableValue::Unit => ThreadSafeValue::Unit,
            SendableValue::Bool(b) => ThreadSafeValue::Bool(*b),
            SendableValue::Char(c) => ThreadSafeValue::Char(*c),
            SendableValue::Int8(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::Int16(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::Int32(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::Int64(i) => ThreadSafeValue::Int64(*i),
            SendableValue::UInt8(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::UInt16(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::UInt32(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::UInt64(i) => ThreadSafeValue::Int64(*i as i64),
            SendableValue::Float32(f) => ThreadSafeValue::Float64(*f as f64),
            SendableValue::Float64(f) => ThreadSafeValue::Float64(*f),
            SendableValue::BigInt(_) => ThreadSafeValue::String(self.display()),
            SendableValue::Decimal(_) => ThreadSafeValue::String(self.display()),
            SendableValue::Pid(p) => ThreadSafeValue::Pid(*p),
            SendableValue::String(s) => ThreadSafeValue::String(s.clone()),
            SendableValue::List(items) => {
                ThreadSafeValue::List(items.iter().map(|v| v.to_thread_safe()).collect())
            }
            SendableValue::Tuple(items) => {
                ThreadSafeValue::Tuple(items.iter().map(|v| v.to_thread_safe()).collect())
            }
            SendableValue::Record(r) => {
                ThreadSafeValue::Record {
                    type_name: r.type_name.clone(),
                    field_names: r.field_names.clone(),
                    fields: r.fields.iter().map(|v| v.to_thread_safe()).collect(),
                    mutable_fields: vec![false; r.fields.len()],
                }
            }
            SendableValue::Variant(v) => {
                ThreadSafeValue::Variant {
                    type_name: Arc::new(v.type_name.clone()),
                    constructor: Arc::new(v.constructor.clone()),
                    fields: v.fields.iter().map(|f| f.to_thread_safe()).collect(),
                }
            }
            SendableValue::Set(items) => {
                ThreadSafeValue::Set(items.iter().map(|k| Self::sendable_key_to_thread_safe_key(k)).collect())
            }
            SendableValue::Map(entries) => {
                let shared_entries: ImblHashMap<SharedMapKey, SharedMapValue> = entries.iter()
                    .map(|(k, v)| (Self::sendable_key_to_shared_key(k), Self::sendable_to_shared(v)))
                    .collect();
                ThreadSafeValue::Map(Arc::new(shared_entries))
            }
            SendableValue::Error(e) => ThreadSafeValue::String(format!("Error: {}", e)),
        }
    }

    fn sendable_key_to_thread_safe_key(key: &SendableMapKey) -> ThreadSafeMapKey {
        match key {
            SendableMapKey::Unit => ThreadSafeMapKey::Unit,
            SendableMapKey::Bool(b) => ThreadSafeMapKey::Bool(*b),
            SendableMapKey::Char(c) => ThreadSafeMapKey::Char(*c),
            SendableMapKey::Int64(i) => ThreadSafeMapKey::Int64(*i),
            SendableMapKey::String(s) => ThreadSafeMapKey::String(s.clone()),
            SendableMapKey::Record { type_name, field_names, fields } => {
                ThreadSafeMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(Self::sendable_key_to_thread_safe_key).collect(),
                }
            }
            SendableMapKey::Variant { type_name, constructor, fields } => {
                ThreadSafeMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(Self::sendable_key_to_thread_safe_key).collect(),
                }
            }
        }
    }

    fn sendable_key_to_shared_key(key: &SendableMapKey) -> SharedMapKey {
        match key {
            SendableMapKey::Unit => SharedMapKey::Unit,
            SendableMapKey::Bool(b) => SharedMapKey::Bool(*b),
            SendableMapKey::Char(c) => SharedMapKey::Char(*c),
            SendableMapKey::Int64(i) => SharedMapKey::Int64(*i),
            SendableMapKey::String(s) => SharedMapKey::String(s.clone()),
            SendableMapKey::Record { type_name, field_names, fields } => {
                SharedMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(Self::sendable_key_to_shared_key).collect(),
                }
            }
            SendableMapKey::Variant { type_name, constructor, fields } => {
                SharedMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(Self::sendable_key_to_shared_key).collect(),
                }
            }
        }
    }

    fn sendable_to_shared(&self) -> SharedMapValue {
        match self {
            SendableValue::Unit => SharedMapValue::Unit,
            SendableValue::Bool(b) => SharedMapValue::Bool(*b),
            SendableValue::Char(c) => SharedMapValue::Char(*c),
            SendableValue::Int8(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::Int16(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::Int32(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::Int64(i) => SharedMapValue::Int64(*i),
            SendableValue::UInt8(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::UInt16(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::UInt32(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::UInt64(i) => SharedMapValue::Int64(*i as i64),
            SendableValue::Float32(f) => SharedMapValue::Float64(*f as f64),
            SendableValue::Float64(f) => SharedMapValue::Float64(*f),
            SendableValue::BigInt(_) => SharedMapValue::String(self.display()),
            SendableValue::Decimal(_) => SharedMapValue::String(self.display()),
            SendableValue::Pid(p) => SharedMapValue::Pid(*p),
            SendableValue::String(s) => SharedMapValue::String(s.clone()),
            SendableValue::List(items) => {
                SharedMapValue::List(items.iter().map(|v| v.sendable_to_shared()).collect())
            }
            SendableValue::Tuple(items) => {
                SharedMapValue::Tuple(items.iter().map(|v| v.sendable_to_shared()).collect())
            }
            SendableValue::Record(r) => {
                SharedMapValue::Record {
                    type_name: r.type_name.clone(),
                    field_names: r.field_names.clone(),
                    fields: r.fields.iter().map(|v| v.sendable_to_shared()).collect(),
                }
            }
            SendableValue::Variant(v) => {
                SharedMapValue::Variant {
                    type_name: v.type_name.clone(),
                    constructor: v.constructor.clone(),
                    fields: v.fields.iter().map(|f| f.sendable_to_shared()).collect(),
                }
            }
            SendableValue::Set(items) => {
                SharedMapValue::Set(items.iter().map(Self::sendable_key_to_shared_key).collect())
            }
            SendableValue::Map(entries) => {
                let shared_entries: ImblHashMap<SharedMapKey, SharedMapValue> = entries.iter()
                    .map(|(k, v)| (Self::sendable_key_to_shared_key(k), v.sendable_to_shared()))
                    .collect();
                SharedMapValue::Map(Arc::new(shared_entries))
            }
            SendableValue::Error(e) => SharedMapValue::String(format!("Error: {}", e)),
        }
    }
}

/// Result from a thread when it finishes.
#[derive(Debug)]
struct ThreadResult {
    #[allow(dead_code)]
    thread_id: u16,
    main_result: Option<Result<SendableValue, String>>,
    /// Captured output from the main process (println, print, etc.)
    main_output: Vec<String>,
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
    /// Main process output (captured from println, print, etc.)
    main_output: Vec<String>,
    /// List of local_ids waiting for async IO (for efficient polling)
    io_waiting: Vec<u64>,
    /// List of local_ids waiting for mvar locks (for efficient retry)
    mvar_waiting: Vec<u64>,
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
            jit_array_fill_functions: HashMap::new(),
            jit_array_sum_functions: HashMap::new(),
            shutdown: AtomicBool::new(false),
            spawn_counter: AtomicU64::new(0),
            thread_local_ids,
            num_threads,
            io_sender: Some(io_sender),
            inspect_sender: None,
            output_sender: None,
            panel_command_sender: None,
            eval_callback: Arc::new(RwLock::new(None)),
            dynamic_functions: Arc::new(RwLock::new(HashMap::new())),
            dynamic_types: Arc::new(RwLock::new(HashMap::new())),
            stdlib_functions: Arc::new(RwLock::new(HashMap::new())),
            stdlib_types: Arc::new(RwLock::new(HashMap::new())),
            stdlib_function_list: Arc::new(RwLock::new(Vec::new())),
            prelude_imports: Arc::new(RwLock::new(HashMap::new())),
            mvars: HashMap::new(),
            dynamic_mvars: Arc::new(RwLock::new(HashMap::new())),
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

    /// Register a JIT-compiled recursive array fill function.
    pub fn register_jit_array_fill_function(&mut self, func_index: u16, jit_fn: JitArrayFillFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_array_fill_functions
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled recursive array sum function.
    pub fn register_jit_array_sum_function(&mut self, func_index: u16, jit_fn: JitArraySumFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .jit_array_sum_functions
            .insert(func_index, jit_fn);
    }

    /// Register a module-level mutable variable with its initial value.
    /// Called during compilation to set up mvars before threads start.
    ///
    /// # Arguments
    /// * `name` - Fully qualified name (e.g., "MyModule.counter")
    /// * `initial_value` - Initial value as ThreadSafeValue
    pub fn register_mvar(&mut self, name: &str, initial_value: ThreadSafeValue) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register mvars after threads started")
            .mvars
            .insert(name.to_string(), Arc::new(RwLock::new(initial_value)));
    }

    /// Get a clone of the dynamic_functions Arc for runtime function registration.
    /// This is used by eval() to add functions that persist during the VM run.
    pub fn get_dynamic_functions(&self) -> Arc<RwLock<HashMap<String, Arc<FunctionValue>>>> {
        self.shared.dynamic_functions.clone()
    }

    /// Get a clone of the stdlib_functions Arc.
    /// This allows eval() to access functions from the main compiler (stdlib).
    pub fn get_stdlib_functions(&self) -> Arc<RwLock<HashMap<String, Arc<FunctionValue>>>> {
        self.shared.stdlib_functions.clone()
    }

    /// Set the stdlib functions from the main compiler.
    /// This should be called after the main compiler has loaded stdlib.
    pub fn set_stdlib_functions(&self, functions: HashMap<String, Arc<FunctionValue>>) {
        let mut stdlib = self.shared.stdlib_functions.write();
        *stdlib = functions;
    }

    /// Get a clone of the dynamic_types Arc for runtime type registration.
    /// This is used by eval() to add types that persist during the VM run.
    pub fn get_dynamic_types(&self) -> Arc<RwLock<HashMap<String, Arc<TypeValue>>>> {
        self.shared.dynamic_types.clone()
    }

    /// Get a clone of the dynamic_mvars Arc for runtime mvar registration.
    /// This is used by eval() to add mvars that persist during the VM run.
    pub fn get_dynamic_mvars(&self) -> Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>> {
        self.shared.dynamic_mvars.clone()
    }

    /// Set the dynamic_mvars Arc to share with another VM.
    /// This allows eval VMs to access mvars defined in previous evals.
    pub fn set_dynamic_mvars(&mut self, mvars: Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot set dynamic_mvars after threads started")
            .dynamic_mvars = mvars;
    }

    /// Get a clone of the stdlib_types Arc.
    /// This allows eval() to access types from the main compiler (stdlib).
    pub fn get_stdlib_types(&self) -> Arc<RwLock<HashMap<String, Arc<TypeValue>>>> {
        self.shared.stdlib_types.clone()
    }

    /// Set the stdlib types from the main compiler.
    /// This should be called after the main compiler has loaded stdlib.
    pub fn set_stdlib_types(&self, types: HashMap<String, Arc<TypeValue>>) {
        let mut stdlib = self.shared.stdlib_types.write();
        *stdlib = types;
    }

    /// Get a clone of the stdlib_function_list Arc.
    pub fn get_stdlib_function_list(&self) -> Arc<RwLock<Vec<String>>> {
        self.shared.stdlib_function_list.clone()
    }

    /// Set the stdlib function list from the main compiler.
    /// This preserves function indices for CallDirect instructions.
    pub fn set_stdlib_function_list(&self, list: Vec<String>) {
        let mut stdlib_list = self.shared.stdlib_function_list.write();
        *stdlib_list = list;
    }

    /// Get a clone of the prelude_imports Arc.
    pub fn get_prelude_imports(&self) -> Arc<RwLock<HashMap<String, String>>> {
        self.shared.prelude_imports.clone()
    }

    /// Set the prelude imports from the main compiler.
    /// This maps local names (e.g., "map") to qualified names (e.g., "stdlib.list.map").
    pub fn set_prelude_imports(&self, imports: HashMap<String, String>) {
        let mut prelude = self.shared.prelude_imports.write();
        *prelude = imports;
    }

    /// Read a module-level mutable variable (mvar).
    /// Acquires a read lock, converts ThreadSafeValue to GcValue for the given heap.
    ///
    /// # Arguments
    /// * `name` - Fully qualified variable name
    /// * `heap` - The process's heap to allocate the value on
    pub fn read_mvar(&self, name: &str, heap: &mut Heap) -> Option<GcValue> {
        let var = self.shared.mvars.get(name)?;
        let guard = var.read();
        Some(guard.to_gc_value(heap))
    }

    /// Write to a module-level mutable variable (mvar).
    /// Converts GcValue to ThreadSafeValue, acquires write lock, stores the value.
    ///
    /// # Arguments
    /// * `name` - Fully qualified variable name
    /// * `value` - The new value (will be deep-copied)
    /// * `heap` - The process's heap (for reading complex values)
    pub fn write_mvar(&self, name: &str, value: &GcValue, heap: &Heap) -> Result<(), RuntimeError> {
        let var = self.shared.mvars.get(name)
            .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?;

        let safe_value = ThreadSafeValue::from_gc_value(value, heap)
            .ok_or_else(|| RuntimeError::Panic(format!("Cannot convert value for mvar: {}", name)))?;

        let mut guard = var.write();
        *guard = safe_value;
        Ok(())
    }

    /// Check if an mvar exists.
    pub fn has_mvar(&self, name: &str) -> bool {
        self.shared.mvars.contains_key(name)
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

        // Hash - compute hash value for any value (FNV-1a algorithm)
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
                        // Functions/closures hash by identity (pointer)
                        GcValue::Closure(ptr, _) => Ok(fnv1a_hash(&ptr.as_raw().to_le_bytes())),
                        GcValue::Function(f) => Ok(fnv1a_hash(f.name.as_bytes())),
                        GcValue::NativeFunction(f) => Ok(fnv1a_hash(f.name.as_bytes())),
                        // Other types
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

        // === Additional String functions ===

        // String.toInt (alias for String.to_int)
        self.register_native("String.toInt", Arc::new(GcNativeFn {
            name: "String.toInt".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            match str_val.data.trim().parse::<i64>() {
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

        // String.toFloat
        self.register_native("String.toFloat", Arc::new(GcNativeFn {
            name: "String.toFloat".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            match str_val.data.trim().parse::<f64>() {
                                Ok(n) => Ok(GcValue::Float64(n)),
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
                        // Convert key and value to shared types, update the shared map
                        let shared_key = key.to_shared_key();
                        let shared_value = heap.gc_value_to_shared(&value).ok_or_else(||
                            RuntimeError::Panic("Cannot convert value to shared type".to_string()))?;
                        let new_map = Arc::new((**shared_map).clone().update(shared_key, shared_value));
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
                        let new_map = Arc::new((**shared_map).clone().without(&shared_key));
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
                        // Clone keys first to release borrow on heap
                        let map = heap.get_map(*ptr).ok_or_else(|| RuntimeError::Panic("Invalid map pointer".to_string()))?;
                        let keys_cloned: Vec<_> = map.entries.keys().cloned().collect();
                        let _ = map;
                        // Now convert to GcValues
                        let keys: Vec<GcValue> = keys_cloned.into_iter().map(|k| k.to_gc_value(heap)).collect();
                        Ok(GcValue::List(heap.make_list(keys)))
                    }
                    GcValue::SharedMap(shared_map) => {
                        // Convert SharedMapKeys to GcValues
                        let keys: Vec<GcValue> = shared_map.keys()
                            .map(|k| GcMapKey::from_shared_key(k).to_gc_value(heap))
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
                        // Convert SharedMapValues to GcValues
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
        // This will be overwritten by setup_inspect() when TUI is active
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

    /// Set up the inspect channel and register the inspect native function.
    /// Returns a receiver that will receive InspectEntry messages when inspect() is called.
    pub fn setup_inspect(&mut self) -> InspectReceiver {
        let (sender, receiver) = channel::unbounded();

        // Store sender in shared state
        Arc::get_mut(&mut self.shared)
            .expect("Cannot setup inspect after threads started")
            .inspect_sender = Some(sender.clone());

        // Register the inspect native function
        // inspect(value, name: String) -> ()
        self.register_native("inspect", Arc::new(GcNativeFn {
            name: "inspect".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                // Get the name (second argument must be a string)
                let name = match &args[1] {
                    GcValue::String(ptr) => {
                        if let Some(s) = heap.get_string(*ptr) {
                            s.data.clone()
                        } else {
                            return Err(RuntimeError::Panic("Invalid string pointer".to_string()));
                        }
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "other".to_string(),
                    }),
                };

                // Deep copy the value to ThreadSafeValue
                if let Some(safe_value) = ThreadSafeValue::from_gc_value(&args[0], heap) {
                    // Send to the inspector (ignore send errors - TUI might not be listening)
                    let _ = sender.send(InspectEntry {
                        name,
                        value: safe_value,
                    });
                }
                // If value can't be converted, silently ignore (like spawn does)

                Ok(GcValue::Unit)
            }),
        }));

        receiver
    }

    /// Set up the output channel for capturing println from all processes.
    /// Returns a receiver that will receive output strings from any process.
    pub fn setup_output(&mut self) -> OutputReceiver {
        let (sender, receiver) = channel::unbounded();

        // Store sender in shared state
        Arc::get_mut(&mut self.shared)
            .expect("Cannot setup output after threads started")
            .output_sender = Some(sender);

        receiver
    }

    /// Set up the panel channel and register all Panel.* native functions.
    /// Returns a receiver that will receive PanelCommand messages.
    pub fn setup_panel(&mut self) -> PanelCommandReceiver {
        let (sender, receiver) = channel::unbounded();

        // Store sender in shared state
        Arc::get_mut(&mut self.shared)
            .expect("Cannot setup panel after threads started")
            .panel_command_sender = Some(sender.clone());

        // Create a separate atomic counter for panel IDs
        // This must be Arc so it can be shared with the closure without borrowing self.shared
        let next_panel_id = Arc::new(AtomicU64::new(1));

        // Helper to extract string from GcValue
        fn get_string(val: &GcValue, heap: &Heap, name: &str) -> Result<String, RuntimeError> {
            match val {
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        Ok(s.data.clone())
                    } else {
                        Err(RuntimeError::Panic(format!("Invalid string pointer for {}", name)))
                    }
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: "other".to_string(),
                }),
            }
        }

        // Helper to extract Int64 from GcValue
        fn get_int(val: &GcValue, _name: &str) -> Result<u64, RuntimeError> {
            match val {
                GcValue::Int64(n) => Ok(*n as u64),
                _ => Err(RuntimeError::TypeError {
                    expected: "Int".to_string(),
                    found: "other".to_string(),
                }),
            }
        }

        // Panel.create(title: String) -> Int
        // Allocates a panel ID and sends Create command
        let next_id_for_create = Arc::clone(&next_panel_id);
        let sender_create = sender.clone();
        self.register_native("Panel.create", Arc::new(GcNativeFn {
            name: "Panel.create".to_string(),
            arity: 1,
            func: Box::new(move |args, heap| {
                let title = get_string(&args[0], heap, "title")?;
                let id = next_id_for_create.fetch_add(1, Ordering::SeqCst);
                let _ = sender_create.send(PanelCommand::Create { id, title });
                Ok(GcValue::Int64(id as i64))
            }),
        }));

        // Panel.setContent(id: Int, content: String) -> ()
        let sender_content = sender.clone();
        self.register_native("Panel.setContent", Arc::new(GcNativeFn {
            name: "Panel.setContent".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let id = get_int(&args[0], "id")?;
                let content = get_string(&args[1], heap, "content")?;
                let _ = sender_content.send(PanelCommand::SetContent { id, content });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.show(id: Int) -> ()
        let sender_show = sender.clone();
        self.register_native("Panel.show", Arc::new(GcNativeFn {
            name: "Panel.show".to_string(),
            arity: 1,
            func: Box::new(move |args, _heap| {
                let id = get_int(&args[0], "id")?;
                let _ = sender_show.send(PanelCommand::Show { id });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.hide(id: Int) -> ()
        let sender_hide = sender.clone();
        self.register_native("Panel.hide", Arc::new(GcNativeFn {
            name: "Panel.hide".to_string(),
            arity: 1,
            func: Box::new(move |args, _heap| {
                let id = get_int(&args[0], "id")?;
                let _ = sender_hide.send(PanelCommand::Hide { id });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.onKey(id: Int, handlerFn: String) -> ()
        let sender_onkey = sender.clone();
        self.register_native("Panel.onKey", Arc::new(GcNativeFn {
            name: "Panel.onKey".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let id = get_int(&args[0], "id")?;
                let handler_fn = get_string(&args[1], heap, "handlerFn")?;
                let _ = sender_onkey.send(PanelCommand::OnKey { id, handler_fn });
                Ok(GcValue::Unit)
            }),
        }));

        // Panel.registerHotkey(key: String, callbackFn: String) -> ()
        let sender_hotkey = sender.clone();
        self.register_native("Panel.registerHotkey", Arc::new(GcNativeFn {
            name: "Panel.registerHotkey".to_string(),
            arity: 2,
            func: Box::new(move |args, heap| {
                let key = get_string(&args[0], heap, "key")?;
                let callback_fn = get_string(&args[1], heap, "callbackFn")?;
                let _ = sender_hotkey.send(PanelCommand::RegisterHotkey { key, callback_fn });
                Ok(GcValue::Unit)
            }),
        }));

        receiver
    }

    /// Setup the eval native function.
    /// Call set_eval_callback() to provide the evaluation implementation.
    pub fn setup_eval(&mut self) {
        // Get a clone of the eval_callback Arc to capture in the closure
        let eval_callback = self.shared.eval_callback.clone();

        // Helper to extract string from GcValue
        fn get_string(val: &GcValue, heap: &Heap, name: &str) -> Result<String, RuntimeError> {
            match val {
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        Ok(s.data.clone())
                    } else {
                        Err(RuntimeError::Panic(format!("Invalid string pointer for {}", name)))
                    }
                }
                _ => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: "other".to_string(),
                }),
            }
        }

        // eval(code: String) -> String
        // Synchronously evaluates code using the registered callback
        self.register_native("eval", Arc::new(GcNativeFn {
            name: "eval".to_string(),
            arity: 1,
            func: Box::new(move |args, heap| {
                let code = get_string(&args[0], heap, "code")?;

                // Get the callback from shared state
                let callback_guard = eval_callback.read();
                let callback = callback_guard.as_ref().ok_or_else(|| {
                    RuntimeError::Panic("eval() not available - no eval callback registered".to_string())
                })?;

                // Call the callback synchronously
                match callback(&code) {
                    Ok(result) => Ok(GcValue::String(heap.alloc_string(result))),
                    Err(err) => Ok(GcValue::String(heap.alloc_string(format!("Error: {}", err)))),
                }
            }),
        }));
    }

    /// Set the eval callback function.
    /// This should be called by the engine to provide the evaluation implementation.
    pub fn set_eval_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        *self.shared.eval_callback.write() = Some(Arc::new(callback));
    }

    /// Register a type.
    pub fn register_type(&mut self, name: &str, type_val: Arc<TypeValue>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after threads started")
            .types
            .insert(name.to_string(), type_val);
    }

    /// Run the VM with the given main function.
    /// Returns the result of the main process, including captured output.
    pub fn run(&mut self, main_func: Arc<FunctionValue>) -> Result<RunResult, RuntimeError> {
        // Clear previous run state
        self.thread_senders.clear();
        self.shared.shutdown.store(false, Ordering::SeqCst);

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
        let mut main_output = Vec::new();
        for handle in self.threads.drain(..) {
            match handle.join() {
                Ok(result) => {
                    if result.main_result.is_some() {
                        main_result = result.main_result;
                        main_output = result.main_output;
                    }
                }
                Err(e) => {
                    return Err(RuntimeError::Panic(format!("Thread panicked: {:?}", e)));
                }
            }
        }

        match main_result {
            Some(Ok(value)) => Ok(RunResult { value: Some(value), output: main_output }),
            Some(Err(e)) => Err(RuntimeError::Panic(e)),
            None => Ok(RunResult { value: None, output: main_output }),
        }
    }

    /// Run a function with a single string argument.
    /// This is optimized for calling handlers (like key handlers) without parsing/compiling.
    pub fn run_with_string_arg(&mut self, func: Arc<FunctionValue>, arg: String) -> Result<RunResult, RuntimeError> {
        // Clear previous run state
        self.thread_senders.clear();
        self.shared.shutdown.store(false, Ordering::SeqCst);

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

            // Thread 0 gets the main function and argument
            let main_func_for_thread = if thread_id == 0 {
                Some((func.clone(), arg.clone()))
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

                // Thread 0 spawns the main process with argument
                if let Some((func, arg)) = main_func_for_thread {
                    let pid = worker.spawn_main_process_with_string_arg(func, arg);
                    worker.main_pid = Some(pid);
                }

                worker.run()
            });

            self.threads.push(handle);
        }

        // Wait for all threads to finish
        let mut main_result = None;
        let mut main_output = Vec::new();
        for handle in self.threads.drain(..) {
            match handle.join() {
                Ok(result) => {
                    if result.main_result.is_some() {
                        main_result = result.main_result;
                        main_output = result.main_output;
                    }
                }
                Err(e) => {
                    return Err(RuntimeError::Panic(format!("Thread panicked: {:?}", e)));
                }
            }
        }

        match main_result {
            Some(Ok(value)) => Ok(RunResult { value: Some(value), output: main_output }),
            Some(Err(e)) => Err(RuntimeError::Panic(e)),
            None => Ok(RunResult { value: None, output: main_output }),
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
            main_output: Vec::new(),
            io_waiting: Vec::new(),
            mvar_waiting: Vec::new(),
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

    /// Spawn the main process on this thread with a single string argument.
    fn spawn_main_process_with_string_arg(&mut self, func: Arc<FunctionValue>, arg: String) -> Pid {
        let local_id = self.next_local_id;
        self.next_local_id += 1;
        let pid = encode_pid(self.thread_id, local_id);

        // Create process with lightweight heap
        let mut process = Process::with_gc_config(pid, GcConfig::lightweight());

        // Allocate the string argument on the process heap
        let str_ptr = process.heap.alloc_string(arg);

        // Set up initial call frame with argument in register 0
        let reg_count = func.code.register_count;
        let mut registers = vec![GcValue::Unit; reg_count];
        if reg_count > 0 {
            registers[0] = GcValue::String(str_ptr);
        }

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
            // Drain inbox FIRST - this delivers cross-thread spawns before we check shutdown
            self.drain_inbox();

            // Check for shutdown
            if self.shared.shutdown.load(Ordering::Relaxed) {
                // Run any processes in run_queue one more time before exiting
                // This gives newly spawned processes a chance to execute
                while let Some(local_id) = self.run_queue.pop_front() {
                    if let Ok(SliceResult::Continue) = self.execute_slice(local_id) {
                        // If process needs more time, just drop it (we're shutting down)
                    }
                }
                break;
            }

            // Wake up any processes whose timers have expired
            self.check_timers();

            // Check for completed async IO operations
            self.check_io();

            // Retry processes waiting for mvar locks
            self.check_mvar_waiters();

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
                        // Capture output before removing process
                        self.main_output = proc.output.clone();
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
            main_output: self.main_output,
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

    /// Check processes waiting for mvar locks and retry acquiring.
    fn check_mvar_waiters(&mut self) {
        // Fast path: no mvar-waiting processes
        if self.mvar_waiting.is_empty() {
            return;
        }

        // First pass: collect info about what each process needs (to avoid borrow issues)
        let to_check: Vec<u64> = self.mvar_waiting.clone();
        let mut wait_info: Vec<(u64, String, bool)> = Vec::new(); // (local_id, mvar_name, is_write)
        let mut completed = Vec::new();

        for local_id in &to_check {
            if let Some(proc) = self.get_process_mut(*local_id) {
                if let ProcessState::WaitingForMvar(ref mvar_name, is_write) = proc.state {
                    wait_info.push((*local_id, mvar_name.clone(), is_write));
                }
            } else {
                // Process was removed while waiting
                completed.push(*local_id);
            }
        }

        // Second pass: try to acquire locks (no process borrow needed)
        let mut acquired_locks: Vec<(u64, String, bool)> = Vec::new(); // (local_id, mvar_name, is_write)
        for (local_id, mvar_name, is_write) in wait_info {
            if let Some(var) = self.shared.mvars.get(&mvar_name) {
                use parking_lot::lock_api::RawRwLock as _;
                let acquired = unsafe {
                    if is_write {
                        var.raw().try_lock_exclusive()
                    } else {
                        var.raw().try_lock_shared()
                    }
                };
                if acquired {
                    acquired_locks.push((local_id, mvar_name, is_write));
                }
            } else {
                // Mvar no longer exists - shouldn't happen, but handle gracefully
                completed.push(local_id);
            }
        }

        // Third pass: update processes that acquired locks
        for (local_id, mvar_name, is_write) in &acquired_locks {
            if let Some(proc) = self.get_process_mut(*local_id) {
                proc.held_mvar_locks.insert(mvar_name.clone(), (*is_write, 1));
                proc.state = ProcessState::Running;
                // Increment IP to move past the MvarLock instruction
                if let Some(frame) = proc.frames.last_mut() {
                    frame.ip += 1;
                }
                // Track in thread-local for compatibility
                HELD_MVAR_LOCKS.with(|locks| {
                    locks.borrow_mut().insert(mvar_name.clone(), *is_write);
                });
                completed.push(*local_id);
                self.run_queue.push_back(*local_id);
            }
        }

        // Remove completed/acquired entries from mvar_waiting
        if !completed.is_empty() {
            self.mvar_waiting.retain(|id| !completed.contains(id));
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
                        GcValue::List(GcList::from_vec(values))
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
                let headers_list = GcValue::List(GcList::from_vec(header_tuples));

                // Build body as string (try UTF-8, fallback to bytes list)
                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
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
                GcValue::List(GcList::from_vec(values))
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
                let headers_list = GcValue::List(GcList::from_vec(header_tuples));

                // Build body as string (try UTF-8, fallback to bytes list)
                let body_value = match std::string::String::from_utf8(body.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = body.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
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
            ExecResult { exit_code, stdout, stderr } => {
                // Build stdout as string (try UTF-8, fallback to bytes list)
                let stdout_value = match std::string::String::from_utf8(stdout.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = stdout.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
                    }
                };

                // Build stderr as string (try UTF-8, fallback to bytes list)
                let stderr_value = match std::string::String::from_utf8(stderr.clone()) {
                    Ok(s) => GcValue::String(proc.heap.alloc_string(s)),
                    Err(_) => {
                        let bytes: Vec<GcValue> = stderr.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                        GcValue::List(GcList::from_vec(bytes))
                    }
                };

                // Build result as a record: { exitCode: Int, stdout: String, stderr: String }
                GcValue::Record(proc.heap.alloc_record(
                    "ExecResult".to_string(),
                    vec!["exitCode".to_string(), "stdout".to_string(), "stderr".to_string()],
                    vec![GcValue::Int64(exit_code as i64), stdout_value, stderr_value],
                    vec![false, false, false], // all immutable
                ))
            }
            ExecHandle(handle_id) => {
                // Return process handle as an integer
                GcValue::Int64(handle_id as i64)
            }
            ExitCode(code) => {
                GcValue::Int64(code as i64)
            }
            PgHandle(handle_id) => {
                // Return Postgres connection handle as an integer
                GcValue::Int64(handle_id as i64)
            }
            PgRows(rows) => {
                // Convert rows to list of lists
                let row_values: Vec<GcValue> = rows
                    .into_iter()
                    .map(|row| {
                        let col_values: Vec<GcValue> = row
                            .into_iter()
                            .map(|col| Self::pg_value_to_gc_value_static(col, proc))
                            .collect();
                        GcValue::List(GcList::from_vec(col_values))
                    })
                    .collect();
                GcValue::List(GcList::from_vec(row_values))
            }
            PgAffected(count) => {
                GcValue::Int64(count as i64)
            }
        }
    }

    /// Convert a PgValue to GcValue
    fn pg_value_to_gc_value_static(pv: crate::process::PgValue, proc: &mut Process) -> GcValue {
        use crate::process::PgValue;
        match pv {
            PgValue::Null => GcValue::Unit,
            PgValue::Bool(b) => GcValue::Bool(b),
            PgValue::Int(i) => GcValue::Int64(i),
            PgValue::Float(f) => GcValue::Float64(f),
            PgValue::String(s) => GcValue::String(proc.heap.alloc_string(s)),
            PgValue::Bytes(bytes) => {
                let values: Vec<GcValue> = bytes.into_iter().map(|b| GcValue::Int64(b as i64)).collect();
                GcValue::List(GcList::from_vec(values))
            }
        }
    }

    /// Convert a GcValue to a PgParam for query parameters
    fn gc_value_to_pg_param(value: &GcValue, heap: &Heap) -> Result<crate::io_runtime::PgParam, RuntimeError> {
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
                if let Some(s) = heap.get_string(*ptr) {
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
                StringDecons(head_dst, tail_dst, str_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    match &proc.frames[frame_idx].registers[*str_reg as usize] {
                        GcValue::String(str_ptr) => {
                            // Clone the string data to avoid borrow conflicts
                            let s = proc.heap.get_string(*str_ptr).map(|h| h.data.clone()).unwrap_or_default();
                            if !s.is_empty() {
                                let mut chars = s.chars();
                                let head_char = chars.next().unwrap();
                                let tail_str = chars.as_str();
                                let head_ptr = proc.heap.alloc_string(head_char.to_string());
                                let tail_ptr = proc.heap.alloc_string(tail_str.to_string());
                                proc.frames[frame_idx].registers[*head_dst as usize] = GcValue::String(head_ptr);
                                proc.frames[frame_idx].registers[*tail_dst as usize] = GcValue::String(tail_ptr);
                            } else {
                                return Ok(FastLoopResult::NeedSlowPath(instr.clone()));
                            }
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    }
                }
                TestEmptyString(dst, str_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let is_empty = match &proc.frames[frame_idx].registers[*str_reg as usize] {
                        GcValue::String(str_ptr) => {
                            proc.heap.get_string(*str_ptr).map(|h| h.data.is_empty()).unwrap_or(true)
                        }
                        _ => return Ok(FastLoopResult::NeedSlowPath(instr.clone())),
                    };
                    proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(is_empty);
                }
                EqStr(dst, left_reg, right_reg) => {
                    proc.frames[frame_idx].ip += 1;
                    let left = &proc.frames[frame_idx].registers[*left_reg as usize];
                    let right = &proc.frames[frame_idx].registers[*right_reg as usize];
                    match (left, right) {
                        (GcValue::String(left_ptr), GcValue::String(right_ptr)) => {
                            // Clone strings to avoid borrow conflicts
                            let left_str = proc.heap.get_string(*left_ptr).map(|h| h.data.clone());
                            let right_str = proc.heap.get_string(*right_ptr).map(|h| h.data.clone());
                            let is_equal = match (left_str, right_str) {
                                (Some(l), Some(r)) => l == r,
                                _ => false,
                            };
                            proc.frames[frame_idx].registers[*dst as usize] = GcValue::Bool(is_equal);
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
                IntToFloat(dst, src) => {
                    let result = match fast_reg!(*src) {
                        GcValue::Int64(v) => GcValue::Float64(*v as f64),
                        GcValue::Int32(v) => GcValue::Float64(*v as f64),
                        GcValue::Float64(v) => GcValue::Float64(*v),
                        GcValue::Float32(v) => GcValue::Float64(*v as f64),
                        _ => return Err(RuntimeError::TypeError { expected: "numeric".into(), found: "other".into() }),
                    };
                    fast_set!(*dst, result);
                    return Ok(StepResult::Continue);
                }
                ToBigInt(dst, src) => {
                    let result = match fast_reg!(*src) {
                        GcValue::Int64(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::Int32(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::Int16(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::Int8(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::UInt64(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::UInt32(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::UInt16(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::UInt8(v) => {
                            let bi = num_bigint::BigInt::from(*v);
                            GcValue::BigInt(proc.heap.alloc_bigint(bi))
                        }
                        GcValue::BigInt(v) => GcValue::BigInt(*v), // Already BigInt
                        _ => return Err(RuntimeError::TypeError { expected: "integer".into(), found: "other".into() }),
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
                    (GcValue::Int32(x), GcValue::Int32(y)) => {
                        if *y == 0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Int32(x / y)
                    }
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let proc = self.get_process(local_id).unwrap();
                        let x_val = proc.heap.get_bigint(*x).unwrap();
                        let y_val = proc.heap.get_bigint(*y).unwrap();
                        if y_val.value == num_bigint::BigInt::from(0) {
                            return Err(RuntimeError::DivisionByZero);
                        }
                        let result = &x_val.value / &y_val.value;
                        let proc_mut = self.get_process_mut(local_id).unwrap();
                        GcValue::BigInt(proc_mut.heap.alloc_bigint(result))
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

            MinInt(dst, a, b) => {
                let va = match reg!(*a) {
                    GcValue::Int64(i) => *i,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "other".to_string(),
                    }),
                };
                let vb = match reg!(*b) {
                    GcValue::Int64(i) => *i,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, GcValue::Int64(va.min(vb)));
            }

            MaxInt(dst, a, b) => {
                let va = match reg!(*a) {
                    GcValue::Int64(i) => *i,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "other".to_string(),
                    }),
                };
                let vb = match reg!(*b) {
                    GcValue::Int64(i) => *i,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, GcValue::Int64(va.max(vb)));
            }

            MinFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x.min(*y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x.min(*y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            MaxFloat(dst, a, b) => {
                let va = reg!(*a).clone();
                let vb = reg!(*b).clone();
                let result = match (&va, &vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x.max(*y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x.max(*y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            SinFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.sin()),
                    GcValue::Float32(f) => GcValue::Float32(f.sin()),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            CosFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.cos()),
                    GcValue::Float32(f) => GcValue::Float32(f.cos()),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            TanFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.tan()),
                    GcValue::Float32(f) => GcValue::Float32(f.tan()),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            FloorFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Int64(f.floor() as i64),
                    GcValue::Float32(f) => GcValue::Int64(f.floor() as i64),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            CeilFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Int64(f.ceil() as i64),
                    GcValue::Float32(f) => GcValue::Int64(f.ceil() as i64),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            RoundFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Int64(f.round() as i64),
                    GcValue::Float32(f) => GcValue::Int64(f.round() as i64),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            LogFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.ln()),
                    GcValue::Float32(f) => GcValue::Float32(f.ln()),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(*dst, result);
            }

            Log10Float(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(f) => GcValue::Float64(f.log10()),
                    GcValue::Float32(f) => GcValue::Float32(f.log10()),
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
                        let _ = proc;

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
                            let proc = self.get_process_mut(local_id).unwrap();
                            if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                let ptr = arr.items.as_mut_ptr();
                                let len = arr.items.len() as i64;
                                let result = jit_fn(ptr as *const i64, len);
                                set_reg!(*dst, GcValue::Int64(result));
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                }

                // Check for recursive array fill JIT (arity 2: arr, idx)
                if args.len() == 2 {
                    if let Some(&jit_fn) = self.shared.jit_array_fill_functions.get(func_idx) {
                        if let GcValue::Int64(idx) = reg!(args[1]).clone() {
                            if let GcValue::Int64Array(arr_ptr) = reg!(args[0]).clone() {
                                let proc = self.get_process_mut(local_id).unwrap();
                                if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let _ = jit_fn(ptr as *const i64, len, idx);
                                    set_reg!(*dst, GcValue::Unit);
                                    return Ok(StepResult::Continue);
                                }
                            }
                        }
                    }
                }

                // Check for recursive array sum JIT (arity 3: arr, idx, acc)
                if args.len() == 3 {
                    if let Some(&jit_fn) = self.shared.jit_array_sum_functions.get(func_idx) {
                        if let (GcValue::Int64(idx), GcValue::Int64(acc)) = (reg!(args[1]).clone(), reg!(args[2]).clone()) {
                            if let GcValue::Int64Array(arr_ptr) = reg!(args[0]).clone() {
                                let proc = self.get_process_mut(local_id).unwrap();
                                if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let result = jit_fn(ptr as *const i64, len, idx, acc);
                                    set_reg!(*dst, GcValue::Int64(result));
                                    return Ok(StepResult::Continue);
                                }
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

                // Check for recursive array fill JIT (arity 2: arr, idx)
                if args.len() == 2 {
                    if let Some(&jit_fn) = self.shared.jit_array_fill_functions.get(func_idx) {
                        if let GcValue::Int64(idx) = reg!(args[1]).clone() {
                            if let GcValue::Int64Array(arr_ptr) = reg!(args[0]).clone() {
                                let proc = self.get_process_mut(local_id).unwrap();
                                if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let _ = jit_fn(ptr as *const i64, len, idx);
                                    let result_val = GcValue::Unit;
                                    // For tail call: pop current frame, set result in parent
                                    let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                                    proc.frames.pop();
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

                // Check for recursive array sum JIT (arity 3: arr, idx, acc)
                if args.len() == 3 {
                    if let Some(&jit_fn) = self.shared.jit_array_sum_functions.get(func_idx) {
                        if let (GcValue::Int64(idx), GcValue::Int64(acc)) = (reg!(args[1]).clone(), reg!(args[2]).clone()) {
                            if let GcValue::Int64Array(arr_ptr) = reg!(args[0]).clone() {
                                let proc = self.get_process_mut(local_id).unwrap();
                                if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                    let ptr = arr.items.as_mut_ptr();
                                    let len = arr.items.len() as i64;
                                    let result = jit_fn(ptr as *const i64, len, idx, acc);
                                    let result_val = GcValue::Int64(result);
                                    // For tail call: pop current frame, set result in parent
                                    let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                                    proc.frames.pop();
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

            // === Process Introspection ===
            ProcessAll(dst) => {
                // Collect all PIDs from this thread's processes
                let pids: Vec<GcValue> = self.processes.iter()
                    .enumerate()
                    .filter_map(|(idx, opt)| {
                        opt.as_ref().map(|_| {
                            let pid = encode_pid(self.thread_id, idx as u64);
                            GcValue::Pid(pid.0)
                        })
                    })
                    .collect();
                let proc = self.get_process_mut(local_id).unwrap();
                let list = GcValue::List(GcList::from_vec(pids));
                let frame = proc.frames.last_mut().unwrap();
                frame.registers[*dst as usize] = list;
            }

            ProcessTime(dst, pid_reg) => {
                let target_pid = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*pid_reg) {
                        GcValue::Pid(p) => Pid(*p),
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: "non-pid".to_string(),
                        }),
                    }
                };

                let target_thread = pid_thread_id(target_pid);
                let uptime_ms = if target_thread == self.thread_id {
                    let target_local_id = pid_local_id(target_pid);
                    if let Some(target_proc) = self.get_process(target_local_id) {
                        target_proc.started_at.elapsed().as_millis() as i64
                    } else {
                        -1 // Process not found
                    }
                } else {
                    -1 // Remote process - not accessible directly
                };

                let proc = self.get_process_mut(local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                frame.registers[*dst as usize] = GcValue::Int64(uptime_ms);
            }

            ProcessAlive(dst, pid_reg) => {
                let target_pid = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*pid_reg) {
                        GcValue::Pid(p) => Pid(*p),
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: "non-pid".to_string(),
                        }),
                    }
                };

                let target_thread = pid_thread_id(target_pid);
                let alive = if target_thread == self.thread_id {
                    let target_local_id = pid_local_id(target_pid);
                    if let Some(target_proc) = self.get_process(target_local_id) {
                        !matches!(target_proc.state, ProcessState::Exited(_))
                    } else {
                        false
                    }
                } else {
                    // Remote process - assume alive if we can't check
                    // (could be extended with cross-thread query)
                    true
                };

                let proc = self.get_process_mut(local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                frame.registers[*dst as usize] = GcValue::Bool(alive);
            }

            ProcessInfo(dst, pid_reg) => {
                let target_pid = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*pid_reg) {
                        GcValue::Pid(p) => Pid(*p),
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: "non-pid".to_string(),
                        }),
                    }
                };

                let target_thread = pid_thread_id(target_pid);
                let info = if target_thread == self.thread_id {
                    let target_local_id = pid_local_id(target_pid);
                    if let Some(target_proc) = self.get_process(target_local_id) {
                        let status = match target_proc.state {
                            ProcessState::Running => "running",
                            ProcessState::Waiting => "waiting",
                            ProcessState::WaitingTimeout => "waiting",
                            ProcessState::WaitingIO => "io",
                            ProcessState::WaitingForMvar(_, _) => "waiting",
                            ProcessState::Sleeping => "sleeping",
                            ProcessState::Suspended => "suspended",
                            ProcessState::Exited(_) => "exited",
                        };
                        let mailbox_len = target_proc.receiver.len() as i64;
                        let uptime_ms = target_proc.started_at.elapsed().as_millis() as i64;
                        Some((status, mailbox_len, uptime_ms))
                    } else {
                        None
                    }
                } else {
                    None // Remote process
                };

                let proc = self.get_process_mut(local_id).unwrap();
                let result = if let Some((status, mailbox_len, uptime_ms)) = info {
                    // Build a record: { status: String, mailbox: Int, uptime: Int }
                    let status_str = proc.heap.alloc_string(status.to_string());
                    let record = proc.heap.alloc_record(
                        "ProcessInfo".to_string(),
                        vec!["status".to_string(), "mailbox".to_string(), "uptime".to_string()],
                        vec![GcValue::String(status_str), GcValue::Int64(mailbox_len), GcValue::Int64(uptime_ms)],
                        vec![false, false, false],
                    );
                    GcValue::Record(record)
                } else {
                    GcValue::Unit // Process not found or remote
                };

                let frame = proc.frames.last_mut().unwrap();
                frame.registers[*dst as usize] = result;
            }

            ProcessKill(dst, pid_reg) => {
                let target_pid = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*pid_reg) {
                        GcValue::Pid(p) => Pid(*p),
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: "non-pid".to_string(),
                        }),
                    }
                };

                let target_thread = pid_thread_id(target_pid);
                let killed = if target_thread == self.thread_id {
                    let target_local_id = pid_local_id(target_pid);
                    if target_local_id != local_id { // Can't kill self
                        if let Some(target_proc) = self.get_process_mut(target_local_id) {
                            target_proc.state = ProcessState::Exited(ExitReason::Killed);
                            true
                        } else {
                            false
                        }
                    } else {
                        false // Can't kill self
                    }
                } else {
                    // Remote process - would need cross-thread message
                    // For now, return false
                    false
                };

                let proc = self.get_process_mut(local_id).unwrap();
                let frame = proc.frames.last_mut().unwrap();
                frame.registers[*dst as usize] = GcValue::Bool(killed);
            }

            // === External process execution ===
            ExecRun(dst, cmd_reg, args_reg) => {
                // Get command string
                let cmd = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*cmd_reg) {
                        GcValue::String(ptr) => {
                            if let Some(s) = proc.heap.get_string(*ptr) {
                                s.data.clone()
                            } else {
                                return Err(RuntimeError::IOError("Invalid command string".to_string()));
                            }
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };

                // Get args list
                let args = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*args_reg) {
                        GcValue::List(list) => {
                            let mut result = Vec::new();
                            for item in list.items() {
                                if let GcValue::String(s_ptr) = item {
                                    if let Some(s) = proc.heap.get_string(s_ptr) {
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
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecRun {
                        command: cmd,
                        args,
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

            ExecSpawn(dst, cmd_reg, args_reg) => {
                let cmd = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*cmd_reg) {
                        GcValue::String(ptr) => {
                            if let Some(s) = proc.heap.get_string(*ptr) {
                                s.data.clone()
                            } else {
                                return Err(RuntimeError::IOError("Invalid command string".to_string()));
                            }
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    }
                };

                let args = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*args_reg) {
                        GcValue::List(list) => {
                            let mut result = Vec::new();
                            for item in list.items() {
                                if let GcValue::String(s_ptr) = item {
                                    if let Some(s) = proc.heap.get_string(s_ptr) {
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
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecSpawn {
                        command: cmd,
                        args,
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

            ExecReadLine(dst, handle_reg) => {
                let handle = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecReadLine {
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

            ExecReadStderr(dst, handle_reg) => {
                let handle = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecReadStderr {
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

            ExecWrite(dst, handle_reg, data_reg) => {
                let handle = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let data = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*data_reg) {
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
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecWrite {
                        handle,
                        data,
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

            ExecWait(dst, handle_reg) => {
                let handle = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecWait {
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

            ExecKill(dst, handle_reg) => {
                let handle = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::ExecKill {
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
                                    if let Some(tuple) = proc.heap.get_tuple(ptr) {
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
                    let _proc = self.get_process(local_id).unwrap();
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
                    let _proc = self.get_process(local_id).unwrap();
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
                                    if let Some(tuple) = proc.heap.get_tuple(tuple_ptr) {
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
                    let _proc = self.get_process(local_id).unwrap();
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

            // === PostgreSQL Operations ===
            PgConnect(dst, conn_str_reg) => {
                let conn_string = {
                    let proc = self.get_process(local_id).unwrap();
                    match reg!(*conn_str_reg) {
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
                    let request = crate::io_runtime::IoRequest::PgConnect {
                        connection_string: conn_string,
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

            PgQuery(dst, handle_reg, query_reg, params_reg) => {
                use crate::io_runtime::PgParam;
                let (handle, query, params) = {
                    let proc = self.get_process(local_id).unwrap();
                    let h = match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (pg handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    let q = match reg!(*query_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid query string".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let p = match reg!(*params_reg) {
                        GcValue::List(list) => {
                            let mut pg_params = Vec::new();
                            for item in list.iter() {
                                let param = Self::gc_value_to_pg_param(&item, &proc.heap)?;
                                pg_params.push(param);
                            }
                            pg_params
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "non-list".to_string(),
                        }),
                    };
                    (h, q, p)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::PgQuery {
                        handle,
                        query,
                        params,
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

            PgExecute(dst, handle_reg, query_reg, params_reg) => {
                use crate::io_runtime::PgParam;
                let (handle, query, params) = {
                    let proc = self.get_process(local_id).unwrap();
                    let h = match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (pg handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    };
                    let q = match reg!(*query_reg) {
                        GcValue::String(ptr) => {
                            proc.heap.get_string(*ptr).map(|s| s.data.clone())
                                .ok_or_else(|| RuntimeError::IOError("Invalid query string".to_string()))?
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "non-string".to_string(),
                        }),
                    };
                    let p = match reg!(*params_reg) {
                        GcValue::List(list) => {
                            let mut pg_params = Vec::new();
                            for item in list.iter() {
                                let param = Self::gc_value_to_pg_param(&item, &proc.heap)?;
                                pg_params.push(param);
                            }
                            pg_params
                        }
                        _ => return Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "non-list".to_string(),
                        }),
                    };
                    (h, q, p)
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::PgExecute {
                        handle,
                        query,
                        params,
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

            PgClose(dst, handle_reg) => {
                let handle = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*handle_reg) {
                        GcValue::Int64(h) => *h as u64,
                        _ => return Err(RuntimeError::TypeError {
                            expected: "Int (pg handle)".to_string(),
                            found: "non-int".to_string(),
                        }),
                    }
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                if let Some(sender) = &self.shared.io_sender {
                    let request = crate::io_runtime::IoRequest::PgClose {
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
                set_reg!(*dst, GcValue::List(GcList::from_vec(bytes)));
            }

            Utf8Decode(dst, bytes_reg) => {
                // Convert list of bytes to string
                // Extract bytes list with immutable borrow
                let bytes = {
                    let _proc = self.get_process(local_id).unwrap();
                    match reg!(*bytes_reg) {
                        GcValue::List(list) => {
                            let mut bytes = Vec::new();
                            for item in list.items() {
                                match item {
                                    GcValue::Int64(n) => bytes.push(n as u8),
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
                    let _proc = self.get_process(local_id).unwrap();
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
                    let _proc = self.get_process(local_id).unwrap();
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
                    let _proc = self.get_process(local_id).unwrap();
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
                    let _proc = self.get_process(local_id).unwrap();
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
                // Get the value and display it first (immutable borrow)
                let s = {
                    let proc = self.get_process(local_id).unwrap();
                    let frame = proc.frames.last().unwrap();
                    proc.heap.display_value(&frame.registers[*src as usize])
                };
                // Send to output channel if available (for TUI), otherwise print to stdout
                if let Some(ref sender) = self.shared.output_sender {
                    let _ = sender.send(s.clone());
                } else {
                    println!("{}", s);
                }
                // Still push to per-process output for main process result
                let proc = self.get_process_mut(local_id).unwrap();
                proc.output.push(s);
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

            // === Module-level mutable variables (mvars) ===
            MvarLock(name_idx, is_write) => {
                let name = match &constants[*name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarLock: name must be a string constant".to_string())),
                };

                // Check mvar exists
                if !self.shared.mvars.contains_key(&name) {
                    return Err(RuntimeError::Panic(format!("Unknown mvar: {}", name)));
                }

                // First check if we already hold this lock (borrow proc mutably)
                let proc = self.get_process_mut(local_id).unwrap();
                if let Some((held_write, depth)) = proc.held_mvar_locks.get_mut(&name) {
                    // Already held - check if we need to upgrade from read to write
                    if *is_write && !*held_write {
                        // Need to upgrade: release read lock, acquire write lock
                        // This is safe because MVar values are copied anyway
                        let current_depth = *depth;
                        proc.held_mvar_locks.remove(&name);
                        let _ = proc;

                        // Release the read lock
                        let var = self.shared.mvars.get(&name).unwrap();
                        use parking_lot::lock_api::RawRwLock as _;
                        unsafe { var.raw().unlock_shared(); }

                        // Try to acquire write lock
                        let acquired = unsafe { var.raw().try_lock_exclusive() };

                        let proc = self.get_process_mut(local_id).unwrap();
                        if acquired {
                            // Got write lock - record with preserved depth
                            proc.held_mvar_locks.insert(name.clone(), (true, current_depth + 1));
                            HELD_MVAR_LOCKS.with(|locks| {
                                locks.borrow_mut().insert(name, true);
                            });
                        } else {
                            // Couldn't acquire write lock - yield and retry
                            // Note: we've already released the read lock, so we need to re-acquire everything
                            proc.state = ProcessState::WaitingForMvar(name, true);
                            proc.frames.last_mut().unwrap().ip -= 1;
                            self.mvar_waiting.push(local_id);
                            return Ok(StepResult::Waiting);
                        }
                    } else {
                        // Same or lower privilege - just increment depth
                        *depth += 1;
                    }
                } else {
                    // Not held - try to acquire (need to access self.shared.mvars)
                    // Drop proc borrow first
                    let _ = proc;

                    let var = self.shared.mvars.get(&name).unwrap();
                    use parking_lot::lock_api::RawRwLock as _;
                    let acquired = unsafe {
                        if *is_write {
                            var.raw().try_lock_exclusive()
                        } else {
                            var.raw().try_lock_shared()
                        }
                    };

                    // Now get proc again to update state
                    let proc = self.get_process_mut(local_id).unwrap();
                    if acquired {
                        // Got the lock - record in process
                        proc.held_mvar_locks.insert(name.clone(), (*is_write, 1));
                        // Also track in thread-local for MvarRead/MvarWrite compatibility
                        HELD_MVAR_LOCKS.with(|locks| {
                            locks.borrow_mut().insert(name, *is_write);
                        });
                    } else {
                        // Couldn't acquire - yield and retry
                        proc.state = ProcessState::WaitingForMvar(name, *is_write);
                        // Decrement IP so we retry this instruction
                        proc.frames.last_mut().unwrap().ip -= 1;
                        // Add to mvar_waiting list for efficient retry
                        self.mvar_waiting.push(local_id);
                        return Ok(StepResult::Waiting);
                    }
                }
            }

            MvarUnlock(name_idx, _is_write) => {
                let name = match &constants[*name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarUnlock: name must be a string constant".to_string())),
                };

                // Check mvar exists
                if !self.shared.mvars.contains_key(&name) {
                    return Err(RuntimeError::Panic(format!("Unknown mvar: {}", name)));
                }

                // Check held locks and decrement depth
                let proc = self.get_process_mut(local_id).unwrap();
                let needs_unlock = if let Some((held_write, depth)) = proc.held_mvar_locks.get_mut(&name) {
                    *depth -= 1;
                    if *depth == 0 {
                        // Last reference - mark for unlock
                        let was_write = *held_write;
                        proc.held_mvar_locks.remove(&name);
                        Some(was_write)
                    } else {
                        None
                    }
                } else {
                    // Unlocking a lock we don't hold - this shouldn't happen
                    return Err(RuntimeError::Panic(format!(
                        "MvarUnlock: process doesn't hold lock on mvar: {}", name
                    )));
                };

                // Actually release the lock if needed (after dropping proc borrow)
                if let Some(was_write) = needs_unlock {
                    let var = self.shared.mvars.get(&name).unwrap();
                    use parking_lot::lock_api::RawRwLock as _;
                    unsafe {
                        if was_write {
                            var.raw().unlock_exclusive();
                        } else {
                            var.raw().unlock_shared();
                        }
                    }
                    // Remove from thread-local tracking
                    HELD_MVAR_LOCKS.with(|locks| {
                        locks.borrow_mut().remove(&name);
                    });
                }
            }

            MvarRead(dst, name_idx) => {
                let name = match &constants[*name_idx as usize] {
                    Value::String(s) => s.as_str(),
                    _ => return Err(RuntimeError::Panic("MvarRead: name must be a string constant".to_string())),
                };
                // Check static mvars first, then dynamic_mvars from eval
                let var = if let Some(v) = self.shared.mvars.get(name) {
                    v.clone()
                } else if let Some(v) = self.shared.dynamic_mvars.read().get(name) {
                    v.clone()
                } else {
                    return Err(RuntimeError::Panic(format!("Unknown mvar: {}", name)));
                };
                // Check if this process already holds a lock on this mvar (from MvarLock)
                let proc = self.get_process(local_id).unwrap();
                let already_locked = proc.held_mvar_locks.contains_key(name);
                let value = if already_locked {
                    // Lock already held by MvarLock - read directly via raw pointer
                    unsafe {
                        let ptr = var.data_ptr();
                        (*ptr).clone()
                    }
                } else {
                    // Fine-grained locking: try to acquire read lock, yield if busy
                    match var.try_read() {
                        Some(guard) => guard.clone(),
                        None => {
                            // Couldn't acquire lock - yield and retry
                            let proc = self.get_process_mut(local_id).unwrap();
                            proc.state = ProcessState::WaitingForMvar(name.to_string(), false);
                            proc.frames.last_mut().unwrap().ip -= 1;
                            self.mvar_waiting.push(local_id);
                            return Ok(StepResult::Waiting);
                        }
                    }
                };
                let proc = self.get_process_mut(local_id).unwrap();
                let gc_value = value.to_gc_value(&mut proc.heap);
                proc.frames.last_mut().unwrap().registers[*dst as usize] = gc_value;
            }

            MvarWrite(name_idx, src) => {
                let name = match &constants[*name_idx as usize] {
                    Value::String(s) => s.as_str(),
                    _ => return Err(RuntimeError::Panic("MvarWrite: name must be a string constant".to_string())),
                };
                let value = reg!(*src).clone();
                // Check static mvars first, then dynamic_mvars from eval
                let var = if let Some(v) = self.shared.mvars.get(name) {
                    v.clone()
                } else if let Some(v) = self.shared.dynamic_mvars.read().get(name) {
                    v.clone()
                } else {
                    return Err(RuntimeError::Panic(format!("Unknown mvar: {}", name)));
                };
                let proc = self.get_process(local_id).unwrap();
                let safe_value = ThreadSafeValue::from_gc_value(&value, &proc.heap)
                    .ok_or_else(|| RuntimeError::Panic(format!("Cannot convert value for mvar: {}", name)))?;
                // Check if this process already holds a WRITE lock on this mvar (from MvarLock)
                let already_locked = proc.held_mvar_locks.get(name)
                    .map(|(is_write, _)| *is_write)
                    .unwrap_or(false);
                if already_locked {
                    // Write lock already held by MvarLock - write directly via raw pointer
                    unsafe {
                        let ptr = var.data_ptr();
                        *ptr = safe_value;
                    }
                } else {
                    // Fine-grained locking: try to acquire write lock, yield if busy
                    match var.try_write() {
                        Some(mut guard) => *guard = safe_value,
                        None => {
                            // Couldn't acquire lock - yield and retry
                            let proc = self.get_process_mut(local_id).unwrap();
                            proc.state = ProcessState::WaitingForMvar(name.to_string(), true);
                            proc.frames.last_mut().unwrap().ip -= 1;
                            self.mvar_waiting.push(local_id);
                            return Ok(StepResult::Waiting);
                        }
                    }
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
                // Look up type in static types first, then dynamic_types (eval-defined)
                let type_info = self.shared.types.get(&type_name).cloned()
                    .or_else(|| self.shared.dynamic_types.read().get(&type_name).cloned());
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
                match tail_val {
                    GcValue::List(tail_list) => {
                        // O(log n) cons using persistent data structure
                        let new_list = tail_list.cons(head_val);
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

            StringDecons(head_dst, tail_dst, str_reg) => {
                let str_val = reg!(*str_reg).clone();
                match str_val {
                    GcValue::String(str_ptr) => {
                        let proc = self.get_process_mut(local_id).unwrap();
                        let s = proc.heap.get_string(str_ptr).map(|h| h.data.clone()).unwrap_or_default();
                        if s.is_empty() {
                            return Err(RuntimeError::Panic("Cannot decons empty string".to_string()));
                        }
                        let mut chars = s.chars();
                        let head_char = chars.next().unwrap();
                        let tail_str = chars.as_str();
                        let head_ptr = proc.heap.alloc_string(head_char.to_string());
                        let tail_ptr = proc.heap.alloc_string(tail_str.to_string());
                        set_reg!(*head_dst, GcValue::String(head_ptr));
                        set_reg!(*tail_dst, GcValue::String(tail_ptr));
                    }
                    _ => return Err(RuntimeError::Panic("StringDecons expects string".to_string())),
                }
            }

            TestEmptyString(dst, str_reg) => {
                let str_val = reg!(*str_reg).clone();
                let is_empty = match str_val {
                    GcValue::String(str_ptr) => {
                        let proc = self.get_process_mut(local_id).unwrap();
                        proc.heap.get_string(str_ptr).map(|h| h.data.is_empty()).unwrap_or(true)
                    }
                    _ => return Err(RuntimeError::Panic("TestEmptyString expects string".to_string())),
                };
                set_reg!(*dst, GcValue::Bool(is_empty));
            }

            EqStr(dst, left_reg, right_reg) => {
                let left_val = reg!(*left_reg).clone();
                let right_val = reg!(*right_reg).clone();
                match (left_val, right_val) {
                    (GcValue::String(left_ptr), GcValue::String(right_ptr)) => {
                        let proc = self.get_process_mut(local_id).unwrap();
                        let left_str = proc.heap.get_string(left_ptr).map(|h| h.data.clone());
                        let right_str = proc.heap.get_string(right_ptr).map(|h| h.data.clone());
                        let is_equal = match (left_str, right_str) {
                            (Some(l), Some(r)) => l == r,
                            _ => false,
                        };
                        set_reg!(*dst, GcValue::Bool(is_equal));
                    }
                    _ => return Err(RuntimeError::Panic("EqStr expects two strings".to_string())),
                }
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
                proc.output.push(s.clone());
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
                    GcValue::Int32(i) => set_reg!(*dst, GcValue::Int32(-i)),
                    GcValue::Int16(i) => set_reg!(*dst, GcValue::Int16(-i)),
                    GcValue::Int8(i) => set_reg!(*dst, GcValue::Int8(-i)),
                    GcValue::BigInt(bi) => {
                        let proc = self.get_process(local_id).unwrap();
                        let bi_val = proc.heap.get_bigint(bi).unwrap();
                        let result = -&bi_val.value;
                        let proc_mut = self.get_process_mut(local_id).unwrap();
                        let result_ptr = proc_mut.heap.alloc_bigint(result);
                        set_reg!(*dst, GcValue::BigInt(result_ptr));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "integer".to_string(),
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

                            let mut items = ImblHashSet::new();

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

                            let mut map = ImblHashMap::new();

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
                                match &map_val {
                                    GcValue::Map(ptr) => {
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
                                    GcValue::SharedMap(shared_map) => {
                                        if let Some(gc_key) = key_val.to_gc_map_key(&proc.heap) {
                                            let shared_key = gc_key.to_shared_key();
                                            shared_map.contains_key(&shared_key)
                                        } else {
                                            false
                                        }
                                    }
                                    _ => {
                                        return Err(RuntimeError::TypeError {
                                            expected: "Map".to_string(),
                                            found: map_val.type_name(&proc.heap).to_string(),
                                        });
                                    }
                                }
                            };
                            set_reg!(*dst, GcValue::Bool(result));
                        }

            

                        MapGet(dst, map_reg, key_reg) => {
                            let map_val = reg!(*map_reg).clone();
                            let key_val = reg!(*key_reg).clone();

                            // For SharedMap, we need to get the value first, then convert outside borrow
                            let shared_value_opt: Option<SharedMapValue>;
                            let gc_result: Option<GcValue>;

                            {
                                let proc = self.get_process(local_id).unwrap();
                                match &map_val {
                                    GcValue::Map(ptr) => {
                                        shared_value_opt = None;
                                        gc_result = if let Some(map) = proc.heap.get_map(*ptr) {
                                            if let Some(key) = key_val.to_gc_map_key(&proc.heap) {
                                                map.entries.get(&key).cloned()
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        };
                                    }
                                    GcValue::SharedMap(shared_map) => {
                                        gc_result = None;
                                        shared_value_opt = if let Some(gc_key) = key_val.to_gc_map_key(&proc.heap) {
                                            let shared_key = gc_key.to_shared_key();
                                            shared_map.get(&shared_key).cloned()
                                        } else {
                                            None
                                        };
                                    }
                                    _ => {
                                        return Err(RuntimeError::TypeError {
                                            expected: "Map".to_string(),
                                            found: map_val.type_name(&proc.heap).to_string(),
                                        });
                                    }
                                }
                            }

                            // Convert SharedMapValue to GcValue outside the borrow
                            let result = if let Some(shared_val) = shared_value_opt {
                                let proc = self.get_process_mut(local_id).unwrap();
                                Some(proc.heap.shared_to_gc_value(&shared_val))
                            } else {
                                gc_result
                            };

                            match result {
                                Some(val) => set_reg!(*dst, val),
                                None => return Err(RuntimeError::Panic("Key not found in map".to_string())),
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
    #[allow(dead_code)]
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
