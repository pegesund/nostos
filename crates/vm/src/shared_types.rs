//! Shared types for cross-thread communication and shared data structures.
//!
//! These types are designed to be thread-safe and can be shared between
//! the GC heap (GcValue) and cross-thread messaging (ThreadSafeValue).

use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::collections::{HashMap as StdHashMap, HashSet as StdHashSet};
use imbl::HashMap as ImblHashMap;

use crate::gc::{GcMapKey, GcValue, Heap};
use crate::process::ThreadSafeValue;

// ============================================================================
// Channel types for cross-thread communication
// ============================================================================

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

// ============================================================================
// Debugger types for stepping, breakpoints, and variable inspection
// ============================================================================

/// A breakpoint location - either a function or a line
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Breakpoint {
    /// Break when entering a function (by name)
    Function(String),
    /// Break at a specific line (optional file for disambiguation)
    Line { file: Option<String>, line: usize },
}

impl Breakpoint {
    /// Create a line breakpoint
    pub fn new(line: usize) -> Self {
        Self::Line { file: None, line }
    }

    /// Create a line breakpoint with file
    pub fn with_file(file: String, line: usize) -> Self {
        Self::Line { file: Some(file), line }
    }

    /// Create a function breakpoint
    pub fn function(name: String) -> Self {
        Self::Function(name)
    }
}

/// Step mode for the debugger
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepMode {
    /// Run normally (no stepping)
    Run,
    /// Pause execution (waiting for command)
    Paused,
    /// Execute one instruction then pause
    StepInstruction,
    /// Execute until next source line then pause
    StepLine,
    /// Execute until current function returns then pause
    StepOut,
    /// Execute until next line in current function (step over calls)
    StepOver,
}

/// Commands sent from the debugger to a process
#[derive(Clone, Debug)]
pub enum DebugCommand {
    /// Continue execution
    Continue,
    /// Step one instruction
    StepInstruction,
    /// Step to next source line
    StepLine,
    /// Step over (next line in current function)
    StepOver,
    /// Step out (until current function returns)
    StepOut,
    /// Add a breakpoint
    AddBreakpoint(Breakpoint),
    /// Remove a breakpoint
    RemoveBreakpoint(Breakpoint),
    /// List all breakpoints
    ListBreakpoints,
    /// Print a variable by name
    PrintVariable(String),
    /// Print all local variables
    PrintLocals,
    /// Print local variables for a specific stack frame (0 = current/bottom)
    PrintLocalsForFrame(usize),
    /// Print the call stack
    PrintStack,
}

/// Events sent from a process to the debugger
#[derive(Clone, Debug)]
pub enum DebugEvent {
    /// Process hit a breakpoint
    BreakpointHit {
        pid: u64,
        file: Option<String>,
        line: usize,
        function: String,
    },
    /// Process paused (step completed)
    Paused {
        pid: u64,
        file: Option<String>,
        line: usize,
        function: String,
        /// Source code of the current function (if available)
        source: Option<String>,
        /// Starting line number of the function in the file (for arrow positioning)
        source_start_line: usize,
    },
    /// Process exited
    Exited {
        pid: u64,
        value: Option<String>,
    },
    /// Variable value
    Variable {
        name: String,
        value: String,
        type_name: String,
    },
    /// List of local variables (for current frame)
    Locals {
        variables: Vec<(String, String, String)>, // (name, value, type)
    },
    /// Local variables for a specific stack frame
    LocalsForFrame {
        frame_index: usize,
        variables: Vec<(String, String, String)>, // (name, value, type)
    },
    /// Call stack
    Stack {
        frames: Vec<StackFrame>,
    },
    /// List of breakpoints
    Breakpoints {
        breakpoints: Vec<Breakpoint>,
    },
    /// Error message
    Error {
        message: String,
    },
}

/// A stack frame for debug display
#[derive(Clone, Debug)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// Source file
    pub file: Option<String>,
    /// Line number
    pub line: usize,
    /// Local variable names (for summary)
    pub locals: Vec<String>,
}

/// Type alias for debug command sender
pub type DebugCommandSender = crossbeam::channel::Sender<DebugCommand>;
/// Type alias for debug command receiver
pub type DebugCommandReceiver = crossbeam::channel::Receiver<DebugCommand>;
/// Type alias for debug event sender
pub type DebugEventSender = crossbeam::channel::Sender<DebugEvent>;
/// Type alias for debug event receiver
pub type DebugEventReceiver = crossbeam::channel::Receiver<DebugEvent>;

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

// ============================================================================
// JIT function pointer types
// ============================================================================

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

// ============================================================================
// Sendable types for cross-thread messaging
// ============================================================================

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

impl Hash for SendableMapKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
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
    Map(StdHashMap<SendableMapKey, SendableValue>),
    Set(StdHashSet<SendableMapKey>),
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
                    let items: StdHashSet<SendableMapKey> = set.items
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
                    let entries: StdHashMap<SendableMapKey, SendableValue> = map.entries
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
                let entries: StdHashMap<SendableMapKey, SendableValue> = shared_map
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

    fn gc_map_key_to_sendable(key: &GcMapKey, heap: &Heap) -> Option<SendableMapKey> {
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
                let entries: StdHashMap<SendableMapKey, SendableValue> = map
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

    /// Check if this is a unit value.
    pub fn is_unit(&self) -> bool {
        matches!(self, SendableValue::Unit)
    }

    /// Convert to ThreadSafeValue.
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
            SendableValue::Record(rec) => ThreadSafeValue::Record {
                type_name: rec.type_name.clone(),
                field_names: rec.field_names.clone(),
                fields: rec.fields.iter().map(|v| v.to_thread_safe()).collect(),
                mutable_fields: vec![false; rec.fields.len()],
            },
            SendableValue::Variant(var) => ThreadSafeValue::Variant {
                type_name: Arc::new(var.type_name.clone()),
                constructor: Arc::new(var.constructor.clone()),
                fields: var.fields.iter().map(|v| v.to_thread_safe()).collect(),
            },
            SendableValue::Map(_) => ThreadSafeValue::String(self.display()),
            SendableValue::Set(_) => ThreadSafeValue::String(self.display()),
            SendableValue::Error(msg) => ThreadSafeValue::String(format!("Error: {}", msg)),
        }
    }

    /// Convert to Value (the non-GC version).
    pub fn to_value(&self) -> crate::value::Value {
        use crate::value::Value;

        /// Helper to convert SendableMapKey to MapKey
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
                        fields: fields.iter().map(sendable_key_to_map_key).collect(),
                    }
                }
                SendableMapKey::Variant { type_name, constructor, fields } => {
                    MapKey::Variant {
                        type_name: type_name.clone(),
                        constructor: constructor.clone(),
                        fields: fields.iter().map(sendable_key_to_map_key).collect(),
                    }
                }
            }
        }

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
            SendableValue::Pid(p) => Value::Pid(crate::value::Pid(*p)),
            SendableValue::String(s) => Value::String(Arc::new(s.clone())),
            SendableValue::List(items) => {
                Value::List(Arc::new(items.iter().map(|v| v.to_value()).collect()))
            }
            SendableValue::Tuple(items) => {
                Value::Tuple(Arc::new(items.iter().map(|v| v.to_value()).collect()))
            }
            SendableValue::Record(rec) => Value::Record(Arc::new(crate::value::RecordValue {
                type_name: rec.type_name.clone(),
                field_names: rec.field_names.clone(),
                fields: rec.fields.iter().map(|v| v.to_value()).collect(),
                mutable_fields: vec![false; rec.fields.len()],
            })),
            SendableValue::Variant(var) => Value::Variant(Arc::new(crate::value::VariantValue {
                type_name: Arc::new(var.type_name.clone()),
                constructor: Arc::new(var.constructor.clone()),
                fields: var.fields.iter().map(|v| v.to_value()).collect(),
                named_fields: None,
            })),
            SendableValue::Map(entries) => {
                let map: std::collections::HashMap<crate::value::MapKey, Value> = entries
                    .iter()
                    .map(|(k, v)| (sendable_key_to_map_key(k), v.to_value()))
                    .collect();
                Value::Map(Arc::new(map))
            }
            SendableValue::Set(items) => {
                let set: std::collections::HashSet<crate::value::MapKey> = items
                    .iter()
                    .map(sendable_key_to_map_key)
                    .collect();
                Value::Set(Arc::new(set))
            }
            SendableValue::Error(msg) => Value::String(Arc::new(format!("Error: {}", msg))),
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
            SendableValue::BigInt(bi) => bi.to_string(),
            SendableValue::Decimal(d) => d.to_string(),
            SendableValue::Pid(p) => format!("<pid:{}>", p),
            SendableValue::String(s) => format!("\"{}\"", s),
            SendableValue::List(items) => {
                let items_str: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", items_str.join(", "))
            }
            SendableValue::Tuple(items) => {
                let items_str: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("({})", items_str.join(", "))
            }
            SendableValue::Record(rec) => {
                let fields_str: Vec<String> = rec.field_names.iter().zip(rec.fields.iter())
                    .map(|(n, v)| format!("{}: {}", n, v.display()))
                    .collect();
                format!("{}{{{}}}", rec.type_name, fields_str.join(", "))
            }
            SendableValue::Variant(var) => {
                if var.fields.is_empty() {
                    var.constructor.clone()
                } else {
                    let fields_str: Vec<String> = var.fields.iter().map(|v| v.display()).collect();
                    format!("{}({})", var.constructor, fields_str.join(", "))
                }
            }
            SendableValue::Map(entries) => {
                format!("%{{...{} entries}}", entries.len())
            }
            SendableValue::Set(items) => {
                format!("#{{...{} items}}", items.len())
            }
            SendableValue::Error(msg) => format!("Error: {}", msg),
        }
    }
}

/// Result from running a function, including captured output.
#[derive(Debug)]
pub struct RunResult {
    /// The return value from the function
    pub value: Option<SendableValue>,
    /// Captured output (from println, print, etc.)
    pub output: Vec<String>,
}

// ============================================================================
// SharedMap types (original shared_types.rs content)
// ============================================================================

/// Thread-safe map key for shared maps and cross-thread communication.
/// Supports primitives, strings, records, and variants as keys.
#[derive(Debug, Clone)]
pub enum SharedMapKey {
    Unit,
    Bool(bool),
    Char(char),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    String(String),
    /// Record as key - fields must all be hashable
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<SharedMapKey>,
    },
    /// Variant as key - fields must all be hashable
    Variant {
        type_name: String,
        constructor: String,
        fields: Vec<SharedMapKey>,
    },
}

// Manual implementation of PartialEq for SharedMapKey
impl PartialEq for SharedMapKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SharedMapKey::Unit, SharedMapKey::Unit) => true,
            (SharedMapKey::Bool(a), SharedMapKey::Bool(b)) => a == b,
            (SharedMapKey::Char(a), SharedMapKey::Char(b)) => a == b,
            (SharedMapKey::Int8(a), SharedMapKey::Int8(b)) => a == b,
            (SharedMapKey::Int16(a), SharedMapKey::Int16(b)) => a == b,
            (SharedMapKey::Int32(a), SharedMapKey::Int32(b)) => a == b,
            (SharedMapKey::Int64(a), SharedMapKey::Int64(b)) => a == b,
            (SharedMapKey::UInt8(a), SharedMapKey::UInt8(b)) => a == b,
            (SharedMapKey::UInt16(a), SharedMapKey::UInt16(b)) => a == b,
            (SharedMapKey::UInt32(a), SharedMapKey::UInt32(b)) => a == b,
            (SharedMapKey::UInt64(a), SharedMapKey::UInt64(b)) => a == b,
            (SharedMapKey::String(a), SharedMapKey::String(b)) => a == b,
            (
                SharedMapKey::Record { type_name: tn1, field_names: fn1, fields: f1 },
                SharedMapKey::Record { type_name: tn2, field_names: fn2, fields: f2 },
            ) => tn1 == tn2 && fn1 == fn2 && f1 == f2,
            (
                SharedMapKey::Variant { type_name: tn1, constructor: c1, fields: f1 },
                SharedMapKey::Variant { type_name: tn2, constructor: c2, fields: f2 },
            ) => tn1 == tn2 && c1 == c2 && f1 == f2,
            _ => false,
        }
    }
}

impl Eq for SharedMapKey {}

// Manual implementation of Hash for SharedMapKey
impl Hash for SharedMapKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the discriminant first for different types
        std::mem::discriminant(self).hash(state);
        match self {
            SharedMapKey::Unit => {}
            SharedMapKey::Bool(b) => b.hash(state),
            SharedMapKey::Char(c) => c.hash(state),
            SharedMapKey::Int8(n) => n.hash(state),
            SharedMapKey::Int16(n) => n.hash(state),
            SharedMapKey::Int32(n) => n.hash(state),
            SharedMapKey::Int64(n) => n.hash(state),
            SharedMapKey::UInt8(n) => n.hash(state),
            SharedMapKey::UInt16(n) => n.hash(state),
            SharedMapKey::UInt32(n) => n.hash(state),
            SharedMapKey::UInt64(n) => n.hash(state),
            SharedMapKey::String(s) => s.hash(state),
            SharedMapKey::Record { type_name, field_names, fields } => {
                type_name.hash(state);
                for name in field_names {
                    name.hash(state);
                }
                for field in fields {
                    field.hash(state);
                }
            }
            SharedMapKey::Variant { type_name, constructor, fields } => {
                type_name.hash(state);
                constructor.hash(state);
                for field in fields {
                    field.hash(state);
                }
            }
        }
    }
}

/// Thread-safe value for shared maps.
/// This is a subset of values that can be safely shared between threads.
/// Designed to cover common map value types without complex dependencies.
#[derive(Debug, Clone)]
pub enum SharedMapValue {
    Unit,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Pid(u64),
    String(String),
    Char(char),
    List(Vec<SharedMapValue>),
    Tuple(Vec<SharedMapValue>),
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<SharedMapValue>,
    },
    Variant {
        type_name: String,
        constructor: String,
        fields: Vec<SharedMapValue>,
    },
    /// Nested shared map
    Map(SharedMap),
    /// A set of keys
    Set(Vec<SharedMapKey>),
    /// Int64 typed array
    Int64Array(Vec<i64>),
    /// Float64 typed array
    Float64Array(Vec<f64>),
}

// SharedMapValue is explicitly Send + Sync
unsafe impl Send for SharedMapValue {}
unsafe impl Sync for SharedMapValue {}

/// A shared map that can be efficiently shared across threads.
/// Uses Arc for O(1) cloning and ImblHashMap for O(log n) updates with structural sharing.
pub type SharedMap = Arc<ImblHashMap<SharedMapKey, SharedMapValue>>;

/// Create an empty shared map.
#[inline]
pub fn empty_shared_map() -> SharedMap {
    Arc::new(ImblHashMap::new())
}

impl SharedMapKey {
    /// Display the key for debugging.
    pub fn display(&self) -> String {
        match self {
            SharedMapKey::Unit => "()".to_string(),
            SharedMapKey::Bool(b) => b.to_string(),
            SharedMapKey::Char(c) => format!("'{}'", c),
            SharedMapKey::Int8(i) => i.to_string(),
            SharedMapKey::Int16(i) => i.to_string(),
            SharedMapKey::Int32(i) => i.to_string(),
            SharedMapKey::Int64(i) => i.to_string(),
            SharedMapKey::UInt8(i) => i.to_string(),
            SharedMapKey::UInt16(i) => i.to_string(),
            SharedMapKey::UInt32(i) => i.to_string(),
            SharedMapKey::UInt64(i) => i.to_string(),
            SharedMapKey::String(s) => format!("\"{}\"", s),
            SharedMapKey::Record { type_name, field_names, fields } => {
                let fields_str: Vec<_> = field_names.iter().zip(fields.iter())
                    .map(|(n, v)| format!("{}: {}", n, v.display()))
                    .collect();
                format!("{}{{{}}}", type_name, fields_str.join(", "))
            }
            SharedMapKey::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}.{}", type_name, constructor)
                } else {
                    let fields_str: Vec<_> = fields.iter().map(|v| v.display()).collect();
                    format!("{}.{}({})", type_name, constructor, fields_str.join(", "))
                }
            }
        }
    }
}

impl SharedMapValue {
    /// Display the value for debugging.
    pub fn display(&self) -> String {
        match self {
            SharedMapValue::Unit => "()".to_string(),
            SharedMapValue::Bool(b) => b.to_string(),
            SharedMapValue::Int64(i) => i.to_string(),
            SharedMapValue::Float64(f) => f.to_string(),
            SharedMapValue::Pid(p) => format!("<pid:{}>", p),
            SharedMapValue::String(s) => format!("\"{}\"", s),
            SharedMapValue::Char(c) => format!("'{}'", c),
            SharedMapValue::List(items) => {
                let items_str: Vec<_> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", items_str.join(", "))
            }
            SharedMapValue::Tuple(items) => {
                let items_str: Vec<_> = items.iter().map(|v| v.display()).collect();
                format!("({})", items_str.join(", "))
            }
            SharedMapValue::Record { type_name, field_names, fields } => {
                let fields_str: Vec<_> = field_names.iter().zip(fields.iter())
                    .map(|(n, v)| format!("{}: {}", n, v.display()))
                    .collect();
                format!("{}{{{}}}", type_name, fields_str.join(", "))
            }
            SharedMapValue::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}.{}", type_name, constructor)
                } else {
                    let fields_str: Vec<_> = fields.iter().map(|v| v.display()).collect();
                    format!("{}.{}({})", type_name, constructor, fields_str.join(", "))
                }
            }
            SharedMapValue::Map(map) => format!("%{{...{} entries}}", map.len()),
            SharedMapValue::Set(items) => format!("#{{...{} items}}", items.len()),
            SharedMapValue::Int64Array(arr) => format!("Int64Array[{}]", arr.len()),
            SharedMapValue::Float64Array(arr) => format!("Float64Array[{}]", arr.len()),
        }
    }
}
