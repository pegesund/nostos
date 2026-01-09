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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use imbl::HashMap as ImblHashMap;
use imbl::HashSet as ImblHashSet;
use tokio::sync::oneshot;
use crossbeam_channel; // Added

use crate::gc::{GcConfig, GcValue, Heap, RawGcPtr, InlineOp, GcNativeFn, GcMapKey};
use crate::io_runtime::IoResult;
use crate::shared_types::{SharedMap, SharedMapKey, SharedMapValue};
use crate::value::{FunctionValue, Pid, RefId, Reg};

// ============================================================================
// Profiling Support
// ============================================================================

/// Statistics for a single function.
#[derive(Clone, Debug, Default)]
pub struct FunctionStats {
    /// Number of times this function was called.
    pub call_count: u64,
    /// Total time spent in this function (nanoseconds).
    pub total_time_ns: u64,
    /// Minimum call duration (nanoseconds).
    pub min_time_ns: u64,
    /// Maximum call duration (nanoseconds).
    pub max_time_ns: u64,
}

impl FunctionStats {
    /// Create new stats with zero values.
    pub fn new() -> Self {
        Self {
            call_count: 0,
            total_time_ns: 0,
            min_time_ns: u64::MAX,
            max_time_ns: 0,
        }
    }

    /// Record a call with the given duration.
    pub fn record_call(&mut self, duration_ns: u64) {
        self.call_count += 1;
        self.total_time_ns += duration_ns;
        self.min_time_ns = self.min_time_ns.min(duration_ns);
        self.max_time_ns = self.max_time_ns.max(duration_ns);
    }

    /// Average time per call in nanoseconds.
    pub fn avg_time_ns(&self) -> u64 {
        if self.call_count > 0 {
            self.total_time_ns / self.call_count
        } else {
            0
        }
    }
}

/// Entry on the profiling call stack (for timing nested calls).
#[derive(Clone, Debug)]
pub struct ProfileStackEntry {
    /// Function name.
    pub function_name: String,
    /// When this call started.
    pub start_time: Instant,
}

/// Per-process profiling data.
#[derive(Clone, Debug, Default)]
pub struct ProfileData {
    /// Statistics per function name.
    pub stats: HashMap<String, FunctionStats>,
    /// Stack of active calls (for timing).
    pub call_stack: Vec<ProfileStackEntry>,
}

impl ProfileData {
    /// Create empty profile data.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            call_stack: Vec::new(),
        }
    }

    /// Record function entry.
    pub fn enter_function(&mut self, name: String) {
        self.call_stack.push(ProfileStackEntry {
            function_name: name,
            start_time: Instant::now(),
        });
    }

    /// Record function exit and update stats.
    pub fn exit_function(&mut self) {
        if let Some(entry) = self.call_stack.pop() {
            let duration = entry.start_time.elapsed().as_nanos() as u64;
            self.stats
                .entry(entry.function_name)
                .or_insert_with(FunctionStats::new)
                .record_call(duration);
        }
    }

    /// Merge another ProfileData into this one.
    pub fn merge(&mut self, other: &ProfileData) {
        for (name, other_stats) in &other.stats {
            let stats = self.stats.entry(name.clone()).or_insert_with(FunctionStats::new);
            stats.call_count += other_stats.call_count;
            stats.total_time_ns += other_stats.total_time_ns;
            stats.min_time_ns = stats.min_time_ns.min(other_stats.min_time_ns);
            stats.max_time_ns = stats.max_time_ns.max(other_stats.max_time_ns);
        }
    }

    /// Print a summary of profiling data.
    pub fn print_summary(&self) {
        print!("{}", self.format_summary());
    }

    /// Format a summary of profiling data as a string.
    pub fn format_summary(&self) -> String {
        use std::fmt::Write;
        let mut output = String::new();

        if self.stats.is_empty() {
            return "No profiling data collected.".to_string();
        }

        // Sort by total time descending
        let mut entries: Vec<_> = self.stats.iter().collect();
        entries.sort_by(|a, b| b.1.total_time_ns.cmp(&a.1.total_time_ns));

        writeln!(output, "\n{:=<80}", "").unwrap();
        writeln!(output, "FUNCTION PROFILING SUMMARY").unwrap();
        writeln!(output, "{:=<80}", "").unwrap();
        writeln!(
            output,
            "{:<40} {:>10} {:>12} {:>12}",
            "Function", "Calls", "Total (ms)", "Avg (µs)"
        ).unwrap();
        writeln!(output, "{:-<80}", "").unwrap();

        for (name, stats) in entries {
            let display_name = if name.len() > 38 {
                format!("...{}", &name[name.len() - 35..])
            } else {
                name.clone()
            };
            writeln!(
                output,
                "{:<40} {:>10} {:>12.3} {:>12.3}",
                display_name,
                stats.call_count,
                stats.total_time_ns as f64 / 1_000_000.0,
                stats.avg_time_ns() as f64 / 1_000.0,
            ).unwrap();
        }
        writeln!(output, "{:=<80}", "").unwrap();
        output
    }
}

/// Type alias for IO response receiver
pub type IoReceiver = oneshot::Receiver<IoResult<IoResponseValue>>;

/// Value types that can be returned from IO operations
#[derive(Debug)]
pub enum IoResponseValue {
    /// Unit (for operations like close)
    Unit,
    /// Bytes (for file read, HTTP body)
    Bytes(Vec<u8>),
    /// String (for file read as string)
    String(String),
    /// File handle ID
    FileHandle(u64),
    /// Integer (bytes written, file size)
    Int(i64),
    /// Boolean (for exists checks)
    Bool(bool),
    /// List of strings (for directory listing)
    StringList(Vec<String>),
    /// HTTP response
    HttpResponse {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    },
    /// Optional string (e.g., readline returning None at EOF)
    OptionString(Option<String>),
    /// HTTP server handle
    ServerHandle(u64),
    /// Incoming HTTP request (from Server.accept)
    ServerRequest {
        request_id: u64,
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    },
    /// Process execution result (from Exec.run)
    ExecResult {
        exit_code: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// Process handle (from Exec.spawn)
    ExecHandle(u64),
    /// Process exit code (from Exec.wait)
    ExitCode(i32),
}

/// A call frame on the stack.
#[derive(Clone)]
pub struct CallFrame {
    /// Function being executed
    pub function: Arc<FunctionValue>,
    /// Instruction pointer
    pub ip: usize,
    /// Register file for this frame (GC-managed values)
    pub registers: Vec<GcValue>,
    /// Captured variables (for closures, GC-managed)
    pub captures: Vec<GcValue>,
    /// Return register in caller's frame
    pub return_reg: Option<Reg>,
}

// CallFrame is Send because all its fields are Send:
// - function: Arc<FunctionValue> is Send (FunctionValue is Send)
// - registers/captures: Vec<GcValue> is Send (GcValue is Send)
// - ip, return_reg: primitives
unsafe impl Send for CallFrame {}
unsafe impl Sync for CallFrame {}

impl CallFrame {
    /// Get the current source line number (1-indexed), or 0 if unavailable.
    /// Note: IP is incremented before instruction execution, so we look at ip-1
    /// to find the line number of the instruction that was actually executing.
    pub fn current_line(&self) -> usize {
        // Use ip-1 because IP is incremented before execution
        let actual_ip = if self.ip > 0 { self.ip - 1 } else { 0 };
        if actual_ip < self.function.code.lines.len() {
            self.function.code.lines[actual_ip]
        } else if !self.function.code.lines.is_empty() {
            // If IP is past the end (e.g., after last instruction), use last line
            *self.function.code.lines.last().unwrap()
        } else {
            0
        }
    }
}

/// Format a stack trace from call frames.
/// Returns a human-readable string showing the call stack.
pub fn format_stack_trace(frames: &[CallFrame]) -> String {
    if frames.is_empty() {
        return String::from("  <no stack frames>");
    }

    let mut result = String::new();
    // Print frames from innermost (most recent) to outermost
    for (i, frame) in frames.iter().rev().enumerate() {
        let line = frame.current_line();
        let func_name = &frame.function.name;

        if line > 0 {
            result.push_str(&format!("  {}. {} (line {})\n", i + 1, func_name, line));
        } else {
            result.push_str(&format!("  {}. {}\n", i + 1, func_name));
        }
    }
    result
}

/// Format a stack trace with local variable values (debug mode).
/// Returns a detailed string showing the call stack with variable values.
pub fn format_stack_trace_debug(frames: &[CallFrame]) -> String {
    if frames.is_empty() {
        return String::from("  <no stack frames>");
    }

    let mut result = String::new();
    // Print frames from innermost (most recent) to outermost
    for (i, frame) in frames.iter().rev().enumerate() {
        let line = frame.current_line();
        let func_name = &frame.function.name;

        if line > 0 {
            result.push_str(&format!("  {}. {} (line {})\n", i + 1, func_name, line));
        } else {
            result.push_str(&format!("  {}. {}\n", i + 1, func_name));
        }

        // Show local variables from debug symbols
        if !frame.function.debug_symbols.is_empty() {
            for symbol in &frame.function.debug_symbols {
                let reg_idx = symbol.register as usize;
                if reg_idx < frame.registers.len() {
                    let value = &frame.registers[reg_idx];
                    let value_str = format_value_short(value);
                    result.push_str(&format!("       {} = {}\n", symbol.name, value_str));
                }
            }
        }
    }
    result
}

/// Format a GcValue as a short string (for debug output).
/// Note: Heap-allocated values show type info only (no heap access needed).
fn format_value_short(value: &GcValue) -> String {
    use crate::gc::GcValue;

    match value {
        GcValue::Unit => "()".to_string(),
        GcValue::Bool(b) => b.to_string(),
        GcValue::Char(c) => format!("'{}'", c),
        GcValue::Int8(i) => i.to_string(),
        GcValue::Int16(i) => i.to_string(),
        GcValue::Int32(i) => i.to_string(),
        GcValue::Int64(i) => i.to_string(),
        GcValue::UInt8(i) => i.to_string(),
        GcValue::UInt16(i) => i.to_string(),
        GcValue::UInt32(i) => i.to_string(),
        GcValue::UInt64(i) => i.to_string(),
        GcValue::Float32(f) => format!("{:.4}", f),
        GcValue::Float64(f) => format!("{:.4}", f),
        GcValue::Decimal(d) => format!("<Decimal {:?}>", d),
        GcValue::String(_) => "<String>".to_string(),
        GcValue::List(_) => "<List>".to_string(),
        GcValue::Array(_) => "<Array>".to_string(),
        GcValue::Tuple(_) => "<Tuple>".to_string(),
        GcValue::Function(f) => format!("<fn {}>", f.name),
        GcValue::NativeFunction(f) => format!("<native {}>", f.name),
        GcValue::Closure(_, _) => "<Closure>".to_string(),
        GcValue::Record(_) => "<Record>".to_string(),
        GcValue::Variant(_) => "<Variant>".to_string(),
        GcValue::Int64Array(_) => "<Int64Array>".to_string(),
        GcValue::Float64Array(_) => "<Float64Array>".to_string(),
        GcValue::Pid(p) => format!("<Pid {}>", p),
        GcValue::Ref(r) => format!("<Ref {}>", r),
        GcValue::Map(_) => "<Map>".to_string(),
        GcValue::SharedMap(m) => format!("<SharedMap {} entries>", m.len()),
        GcValue::Set(_) => "<Set>".to_string(),
        GcValue::BigInt(_) => "<BigInt>".to_string(),
        GcValue::Type(t) => format!("<Type {}>", t.name),
        GcValue::Pointer(p) => format!("<Pointer 0x{:x}>", p),
    }
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
    /// Waiting for a message in receive (no timeout).
    Waiting,
    /// Waiting for a message with timeout deadline.
    WaitingTimeout,
    /// Sleeping until wake_time.
    Sleeping,
    /// Waiting for async IO operation to complete.
    /// The actual receiver is stored in Process::io_receiver.
    WaitingIO,
    /// Waiting to acquire an mvar lock.
    /// Contains (mvar_name, is_write_lock).
    WaitingForMvar(String, bool),
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

/// Thread-safe map key for cross-thread communication.
#[derive(Debug, Clone)]
pub enum ThreadSafeMapKey {
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
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<ThreadSafeMapKey>,
    },
    Variant {
        type_name: String,
        constructor: String,
        fields: Vec<ThreadSafeMapKey>,
    },
}

impl PartialEq for ThreadSafeMapKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ThreadSafeMapKey::Unit, ThreadSafeMapKey::Unit) => true,
            (ThreadSafeMapKey::Bool(a), ThreadSafeMapKey::Bool(b)) => a == b,
            (ThreadSafeMapKey::Char(a), ThreadSafeMapKey::Char(b)) => a == b,
            (ThreadSafeMapKey::Int8(a), ThreadSafeMapKey::Int8(b)) => a == b,
            (ThreadSafeMapKey::Int16(a), ThreadSafeMapKey::Int16(b)) => a == b,
            (ThreadSafeMapKey::Int32(a), ThreadSafeMapKey::Int32(b)) => a == b,
            (ThreadSafeMapKey::Int64(a), ThreadSafeMapKey::Int64(b)) => a == b,
            (ThreadSafeMapKey::UInt8(a), ThreadSafeMapKey::UInt8(b)) => a == b,
            (ThreadSafeMapKey::UInt16(a), ThreadSafeMapKey::UInt16(b)) => a == b,
            (ThreadSafeMapKey::UInt32(a), ThreadSafeMapKey::UInt32(b)) => a == b,
            (ThreadSafeMapKey::UInt64(a), ThreadSafeMapKey::UInt64(b)) => a == b,
            (ThreadSafeMapKey::String(a), ThreadSafeMapKey::String(b)) => a == b,
            (
                ThreadSafeMapKey::Record { type_name: tn1, field_names: fn1, fields: f1 },
                ThreadSafeMapKey::Record { type_name: tn2, field_names: fn2, fields: f2 },
            ) => tn1 == tn2 && fn1 == fn2 && f1 == f2,
            (
                ThreadSafeMapKey::Variant { type_name: tn1, constructor: c1, fields: f1 },
                ThreadSafeMapKey::Variant { type_name: tn2, constructor: c2, fields: f2 },
            ) => tn1 == tn2 && c1 == c2 && f1 == f2,
            _ => false,
        }
    }
}

impl Eq for ThreadSafeMapKey {}

impl std::hash::Hash for ThreadSafeMapKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ThreadSafeMapKey::Unit => {}
            ThreadSafeMapKey::Bool(b) => b.hash(state),
            ThreadSafeMapKey::Char(c) => c.hash(state),
            ThreadSafeMapKey::Int8(n) => n.hash(state),
            ThreadSafeMapKey::Int16(n) => n.hash(state),
            ThreadSafeMapKey::Int32(n) => n.hash(state),
            ThreadSafeMapKey::Int64(n) => n.hash(state),
            ThreadSafeMapKey::UInt8(n) => n.hash(state),
            ThreadSafeMapKey::UInt16(n) => n.hash(state),
            ThreadSafeMapKey::UInt32(n) => n.hash(state),
            ThreadSafeMapKey::UInt64(n) => n.hash(state),
            ThreadSafeMapKey::String(s) => s.hash(state),
            ThreadSafeMapKey::Record { type_name, field_names, fields } => {
                type_name.hash(state);
                for name in field_names {
                    name.hash(state);
                }
                for field in fields {
                    field.hash(state);
                }
            }
            ThreadSafeMapKey::Variant { type_name, constructor, fields } => {
                type_name.hash(state);
                constructor.hash(state);
                for field in fields {
                    field.hash(state);
                }
            }
        }
    }
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
    /// A closure with its function and captured values
    Closure {
        function: Arc<FunctionValue>,
        captures: Vec<ThreadSafeValue>,
        capture_names: Vec<String>,
    },
    /// A variant with type, constructor and fields
    Variant {
        type_name: Arc<String>,
        constructor: Arc<String>,
        fields: Vec<ThreadSafeValue>,
    },
    /// A function (already thread-safe via Arc)
    Function(Arc<FunctionValue>),
    /// A native function (already thread-safe via Arc)
    NativeFunction(Arc<GcNativeFn>),
    /// A map with key-value pairs (Arc-wrapped for O(1) sharing)
    Map(SharedMap),
    /// A set of keys
    Set(Vec<ThreadSafeMapKey>),
    /// Int64 typed array (deep copy of data)
    Int64Array(Vec<i64>),
    /// Float64 typed array (deep copy of data)
    Float64Array(Vec<f64>),
}

// ThreadSafeValue is explicitly Send + Sync:
// - All variants contain only Send+Sync types (primitives, String, Arc, Vec)
// - Designed specifically for cross-thread communication
unsafe impl Send for ThreadSafeValue {}
unsafe impl Sync for ThreadSafeValue {}

impl ThreadSafeMapKey {
    /// Convert a GcMapKey to a thread-safe map key.
    pub fn from_gc_map_key(key: &GcMapKey) -> Self {
        match key {
            GcMapKey::Unit => ThreadSafeMapKey::Unit,
            GcMapKey::Bool(b) => ThreadSafeMapKey::Bool(*b),
            GcMapKey::Char(c) => ThreadSafeMapKey::Char(*c),
            GcMapKey::Int8(i) => ThreadSafeMapKey::Int8(*i),
            GcMapKey::Int16(i) => ThreadSafeMapKey::Int16(*i),
            GcMapKey::Int32(i) => ThreadSafeMapKey::Int32(*i),
            GcMapKey::Int64(i) => ThreadSafeMapKey::Int64(*i),
            GcMapKey::UInt8(i) => ThreadSafeMapKey::UInt8(*i),
            GcMapKey::UInt16(i) => ThreadSafeMapKey::UInt16(*i),
            GcMapKey::UInt32(i) => ThreadSafeMapKey::UInt32(*i),
            GcMapKey::UInt64(i) => ThreadSafeMapKey::UInt64(*i),
            GcMapKey::String(s) => ThreadSafeMapKey::String(s.clone()),
            GcMapKey::Record { type_name, field_names, fields } => ThreadSafeMapKey::Record {
                type_name: type_name.clone(),
                field_names: field_names.clone(),
                fields: fields.iter().map(|f| ThreadSafeMapKey::from_gc_map_key(f)).collect(),
            },
            GcMapKey::Variant { type_name, constructor, fields } => ThreadSafeMapKey::Variant {
                type_name: type_name.clone(),
                constructor: constructor.clone(),
                fields: fields.iter().map(|f| ThreadSafeMapKey::from_gc_map_key(f)).collect(),
            },
        }
    }

    /// Convert back to GcMapKey.
    pub fn to_gc_map_key(&self) -> GcMapKey {
        match self {
            ThreadSafeMapKey::Unit => GcMapKey::Unit,
            ThreadSafeMapKey::Bool(b) => GcMapKey::Bool(*b),
            ThreadSafeMapKey::Char(c) => GcMapKey::Char(*c),
            ThreadSafeMapKey::Int8(i) => GcMapKey::Int8(*i),
            ThreadSafeMapKey::Int16(i) => GcMapKey::Int16(*i),
            ThreadSafeMapKey::Int32(i) => GcMapKey::Int32(*i),
            ThreadSafeMapKey::Int64(i) => GcMapKey::Int64(*i),
            ThreadSafeMapKey::UInt8(i) => GcMapKey::UInt8(*i),
            ThreadSafeMapKey::UInt16(i) => GcMapKey::UInt16(*i),
            ThreadSafeMapKey::UInt32(i) => GcMapKey::UInt32(*i),
            ThreadSafeMapKey::UInt64(i) => GcMapKey::UInt64(*i),
            ThreadSafeMapKey::String(s) => GcMapKey::String(s.clone()),
            ThreadSafeMapKey::Record { type_name, field_names, fields } => GcMapKey::Record {
                type_name: type_name.clone(),
                field_names: field_names.clone(),
                fields: fields.iter().map(|f| f.to_gc_map_key()).collect(),
            },
            ThreadSafeMapKey::Variant { type_name, constructor, fields } => GcMapKey::Variant {
                type_name: type_name.clone(),
                constructor: constructor.clone(),
                fields: fields.iter().map(|f| f.to_gc_map_key()).collect(),
            },
        }
    }
}

impl ThreadSafeValue {
    /// Convert a GcValue to a thread-safe value (deep copy).
    pub fn from_gc_value(value: &GcValue, heap: &Heap) -> Option<Self> {
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
            GcValue::List(list) => {
                let items: Option<Vec<_>> = list.items().iter()
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
            GcValue::Closure(ptr, _) => {
                let closure = heap.get_closure(*ptr)?;
                // Recursively convert captures to thread-safe values
                let captures: Option<Vec<_>> = closure.captures.iter()
                    .map(|v| ThreadSafeValue::from_gc_value(v, heap))
                    .collect();
                ThreadSafeValue::Closure {
                    function: closure.function.clone(),
                    captures: captures?,
                    capture_names: closure.capture_names.clone(),
                }
            }
            GcValue::Variant(ptr) => {
                let variant = heap.get_variant(*ptr)?;
                let fields: Option<Vec<_>> = variant.fields.iter()
                    .map(|v| ThreadSafeValue::from_gc_value(v, heap))
                    .collect();
                ThreadSafeValue::Variant {
                    type_name: variant.type_name.clone(),
                    constructor: variant.constructor.clone(),
                    fields: fields?,
                }
            }
            // Functions are already thread-safe (Arc<FunctionValue>)
            GcValue::Function(func) => ThreadSafeValue::Function(func.clone()),
            GcValue::NativeFunction(func) => ThreadSafeValue::NativeFunction(func.clone()),
            // Maps
            GcValue::Map(ptr) => {
                let map = heap.get_map(*ptr)?;
                let shared_map = heap.gc_value_to_shared(&GcValue::Map(*ptr))?;
                if let SharedMapValue::Map(m) = shared_map {
                    ThreadSafeValue::Map(m)
                } else {
                    return None;
                }
            }
            // SharedMap - already in the right format, just clone the Arc (O(1))
            GcValue::SharedMap(map) => {
                ThreadSafeValue::Map(map.clone())
            }
            // Sets
            GcValue::Set(ptr) => {
                let set = heap.get_set(*ptr)?;
                let items: Vec<_> = set.items.iter()
                    .map(|k| ThreadSafeMapKey::from_gc_map_key(k))
                    .collect();
                ThreadSafeValue::Set(items)
            }
            // Typed arrays (deep copy)
            GcValue::Int64Array(ptr) => {
                let arr = heap.get_int64_array(*ptr)?;
                ThreadSafeValue::Int64Array(arr.items.clone())
            }
            GcValue::Float64Array(ptr) => {
                let arr = heap.get_float64_array(*ptr)?;
                ThreadSafeValue::Float64Array(arr.items.clone())
            }
            // Other values cannot be sent safely
            _ => return None,
        })
    }

    /// Convert back to GcValue, allocating on the given heap.
    pub fn to_gc_value(&self, heap: &mut Heap) -> GcValue {
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
                let list = heap.make_list(gc_items);
                GcValue::List(list)
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
            ThreadSafeValue::Closure { function, captures, capture_names } => {
                // Recursively convert captures to GcValue
                let gc_captures: Vec<GcValue> = captures.iter()
                    .map(|v| v.to_gc_value(heap))
                    .collect();
                let inline_op = InlineOp::from_function(function);
                let ptr = heap.alloc_closure(
                    function.clone(),
                    gc_captures,
                    capture_names.clone(),
                );
                GcValue::Closure(ptr, inline_op)
            }
            ThreadSafeValue::Variant { type_name, constructor, fields } => {
                let gc_fields: Vec<GcValue> = fields.iter()
                    .map(|v| v.to_gc_value(heap))
                    .collect();
                let ptr = heap.alloc_variant(
                    type_name.clone(),
                    constructor.clone(),
                    gc_fields,
                );
                GcValue::Variant(ptr)
            }
            ThreadSafeValue::Function(func) => GcValue::Function(func.clone()),
            ThreadSafeValue::NativeFunction(func) => GcValue::NativeFunction(func.clone()),
            ThreadSafeValue::Map(shared_map) => {
                // For message passing: convert to regular Map (deep copy)
                // MVars use SharedMap directly via MvarRead instruction
                let mut entries = ImblHashMap::new();
                for (k, v) in shared_map.iter() {
                    let gc_key = GcMapKey::from_shared_key(k);
                    let gc_val = heap.shared_to_gc_value(v);
                    entries.insert(gc_key, gc_val);
                }
                let ptr = heap.alloc_map(entries);
                GcValue::Map(ptr)
            }
            ThreadSafeValue::Set(items) => {
                let gc_items: ImblHashSet<GcMapKey> = items.iter()
                    .map(|k| k.to_gc_map_key())
                    .collect();
                let ptr = heap.alloc_set(gc_items);
                GcValue::Set(ptr)
            }
            ThreadSafeValue::Int64Array(items) => {
                let ptr = heap.alloc_int64_array(items.clone());
                GcValue::Int64Array(ptr)
            }
            ThreadSafeValue::Float64Array(items) => {
                let ptr = heap.alloc_float64_array(items.clone());
                GcValue::Float64Array(ptr)
            }
        }
    }
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

    /// Channel receiver for incoming messages (thread-safe).
    pub receiver: crossbeam_channel::Receiver<ThreadSafeValue>,

    /// Channel sender (kept for cloning/distribution).
    pub sender: crossbeam_channel::Sender<ThreadSafeValue>,

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

    /// Wake time for sleeping/timed-out processes.
    pub wake_time: Option<Instant>,

    /// Destination register for receive timeout (so check_timers knows where to put Unit).
    pub timeout_dst: Option<u8>,

    /// Output buffer (for testing/REPL).
    pub output: Vec<String>,

    /// When this process was created.
    pub started_at: Instant,

    /// Receiver for pending IO operation (when state is WaitingIO).
    pub io_receiver: Option<IoReceiver>,

    /// Destination register for IO result (when state is WaitingIO).
    pub io_result_reg: Option<Reg>,

    /// MVar locks held by this process: mvar_name -> (is_write, acquisition_depth).
    /// Acquisition depth handles nested calls - only release when depth reaches 0.
    pub held_mvar_locks: HashMap<String, (bool, u32)>,

    /// Profiling data (only populated when profiling is enabled).
    pub profile: Option<ProfileData>,
}

impl Process {
    /// Create a new process with default configuration.
    pub fn new(pid: Pid) -> Self {
        Self::with_gc_config(pid, GcConfig::default())
    }

    /// Create a new process with custom GC configuration.
    pub fn with_gc_config(pid: Pid, gc_config: GcConfig) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self {
            pid,
            heap: Heap::with_config(gc_config),
            frames: Vec::new(),
            register_pool: Vec::new(),
            receiver,
            sender,
            state: ProcessState::Running,
            reductions: REDUCTIONS_PER_SLICE,
            links: Vec::new(),
            monitors: HashMap::new(),
            monitored_by: HashMap::new(),
            handlers: Vec::new(),
            current_exception: None,
            exit_value: None,
            wake_time: None,
            timeout_dst: None,
            output: Vec::new(),
            started_at: Instant::now(),
            io_receiver: None,
            io_result_reg: None,
            held_mvar_locks: HashMap::new(),
            profile: None,
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
        if let Some(safe_msg) = ThreadSafeValue::from_gc_value(&message, source_heap) {
            let _ = self.sender.send(safe_msg);
        }

        // Wake up if waiting for messages
        if self.state == ProcessState::Waiting {
            self.state = ProcessState::Running;
        }
    }

    /// Try to receive a message (simple FIFO for now).
    /// Returns None if mailbox is empty.
    pub fn try_receive(&mut self) -> Option<GcValue> {
        match self.receiver.try_recv() {
            Ok(msg) => Some(msg.to_gc_value(&mut self.heap)),
            Err(_) => None,
        }
    }

    /// Check if mailbox has messages.
    pub fn has_messages(&self) -> bool {
        !self.receiver.is_empty()
    }

    /// Set process to waiting state (blocked in receive).
    pub fn wait_for_message(&mut self) {
        if self.receiver.is_empty() {
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

    // === Garbage Collection ===

    /// Gather all GC roots from this process's state.
    /// Roots include: registers, captures, and mailbox messages.
    pub fn gather_gc_roots(&self) -> Vec<RawGcPtr> {
        let mut roots = Vec::new();

        // Collect from all call frames
        for frame in &self.frames {
            // Registers in this frame
            for reg in &frame.registers {
                roots.extend(reg.gc_pointers());
            }
            // Captured variables (for closures)
            for cap in &frame.captures {
                roots.extend(cap.gc_pointers());
            }
        }

        roots
    }

    /// Run garbage collection if heap threshold exceeded.
    /// Should be called at yield points (when process is about to be preempted).
    pub fn maybe_gc(&mut self) {
        if self.heap.should_collect() {
            // Gather roots from live state
            let roots = self.gather_gc_roots();

            // Set roots and collect
            self.heap.set_roots(roots);
            self.heap.collect();

            // Clear roots after collection (they're only valid during this GC)
            self.heap.clear_roots();
        }
    }

    /// Force garbage collection regardless of threshold.
    /// Useful for testing or when memory pressure is high.
    pub fn force_gc(&mut self) {
        let roots = self.gather_gc_roots();
        self.heap.set_roots(roots);
        self.heap.collect();
        self.heap.clear_roots();
    }

    // === Async IO Support ===

    /// Start waiting for an IO operation.
    /// Sets state to WaitingIO and stores the receiver.
    pub fn start_io_wait(&mut self, receiver: IoReceiver, result_reg: Reg) {
        self.state = ProcessState::WaitingIO;
        self.io_receiver = Some(receiver);
        self.io_result_reg = Some(result_reg);
    }

    /// Check if IO operation has completed.
    /// Returns Some((result, dest_reg)) if done, None if still pending.
    pub fn poll_io(&mut self) -> Option<(IoResult<IoResponseValue>, Reg)> {
        if let Some(ref mut receiver) = self.io_receiver {
            match receiver.try_recv() {
                Ok(result) => {
                    // IO completed - clean up and return result with destination register
                    self.io_receiver = None;
                    let result_reg = self.io_result_reg.take().expect("IO result register should be set");
                    self.state = ProcessState::Running;
                    Some((result, result_reg))
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    // Still waiting
                    None
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    // Sender dropped - IO runtime shut down
                    self.io_receiver = None;
                    let result_reg = self.io_result_reg.take().expect("IO result register should be set");
                    self.state = ProcessState::Running;
                    Some((Err(crate::io_runtime::IoError::Other(
                        "IO runtime closed".to_string(),
                    )), result_reg))
                }
            }
        } else {
            None
        }
    }
}

impl std::fmt::Debug for Process {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Process")
            .field("pid", &self.pid)
            .field("state", &self.state)
            .field("reductions", &self.reductions)
            .field("mailbox_len", &self.receiver.len())
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
        assert!(proc.receiver.is_empty());
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
        let msg = GcValue::Int64(42);

        // Deliver to receiver (deep copy)
        receiver.deliver_message(msg, &sender.heap);

        assert!(receiver.has_messages());
        let received = receiver.try_receive().unwrap();
        assert_eq!(received, GcValue::Int64(42));
        assert!(!receiver.has_messages());
    }

    #[test]
    fn test_waiting_state() {
        let mut proc = Process::new(Pid(1));

        // Empty mailbox -> waiting
        proc.wait_for_message();
        assert_eq!(proc.state, ProcessState::Waiting);

        // Message delivery wakes up
        proc.deliver_message(GcValue::Int64(1), &Heap::new());
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
