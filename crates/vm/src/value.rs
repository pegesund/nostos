//! Value representation for the Nostos VM.
//!
//! Design goals:
//! - JIT-friendly: values can be unboxed in hot paths
//! - Introspectable: runtime type information available
//! - GC-friendly: heap objects clearly identified
//! - Tail-call friendly: closures capture environment properly
//!
//! Current implementation uses a tagged enum. Can be optimized to
//! NaN-boxing later for better performance.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use nostos_extension::GcNativeHandle;
use num_bigint::BigInt;
use rust_decimal::Decimal;

/// Shared register list - makes instruction cloning O(1) instead of O(n).
/// Used for function arguments, list/tuple construction, etc.
/// Uses Arc for thread-safety with ParallelVM.
pub type RegList = Arc<[Reg]>;

/// A runtime value in Nostos.
#[derive(Clone)]
pub enum Value {
    // === Immediate values (unboxed in JIT) ===
    Unit,
    Bool(bool),
    Char(char),

    // === Signed integers ===
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),

    // === Unsigned integers ===
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),

    // === Floating point ===
    Float32(f32),
    Float64(f64),

    // === Arbitrary precision ===
    BigInt(Arc<BigInt>),

    // === Decimal (fixed-point) ===
    Decimal(Decimal),

    // === Heap-allocated values ===
    String(Arc<String>),
    List(Arc<Vec<Value>>),
    Array(Arc<std::sync::RwLock<Vec<Value>>>), // mutable, general
    // Typed arrays for JIT optimization (contiguous memory, no tag checking)
    Int64Array(Arc<std::sync::RwLock<Vec<i64>>>),
    Float64Array(Arc<std::sync::RwLock<Vec<f64>>>),
    Float32Array(Arc<std::sync::RwLock<Vec<f32>>>),
    Tuple(Arc<Vec<Value>>),
    Map(Arc<HashMap<MapKey, Value>>),
    Set(Arc<std::collections::HashSet<MapKey>>),

    // === Structured values ===
    Record(Arc<RecordValue>),
    ReactiveRecord(Arc<ReactiveRecordValue>),
    Variant(Arc<VariantValue>),

    // === Callable values ===
    Function(Arc<FunctionValue>),
    Closure(Arc<ClosureValue>),
    NativeFunction(Arc<NativeFn>),

    // === Concurrency values ===
    Pid(Pid),
    Ref(RefId),

    // === Special ===
    /// Type value for introspection
    Type(Arc<TypeValue>),
    /// Opaque pointer for FFI (legacy)
    Pointer(usize),
    /// GC-managed native handle (extension data with cleanup callback)
    /// Wrapped in Arc so cloning is safe and cleanup only happens once.
    NativeHandle(Arc<GcNativeHandle>),
}

/// Key type for maps and sets (must be hashable).
#[derive(Debug, Clone)]
pub enum MapKey {
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
    // String
    String(Arc<String>),
    // Record as key - fields must all be hashable
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<MapKey>,
    },
    // Variant as key - fields must all be hashable
    Variant {
        type_name: String,
        constructor: String,
        fields: Vec<MapKey>,
    },
}

// Manual implementation of PartialEq for MapKey
impl PartialEq for MapKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MapKey::Unit, MapKey::Unit) => true,
            (MapKey::Bool(a), MapKey::Bool(b)) => a == b,
            (MapKey::Char(a), MapKey::Char(b)) => a == b,
            (MapKey::Int8(a), MapKey::Int8(b)) => a == b,
            (MapKey::Int16(a), MapKey::Int16(b)) => a == b,
            (MapKey::Int32(a), MapKey::Int32(b)) => a == b,
            (MapKey::Int64(a), MapKey::Int64(b)) => a == b,
            (MapKey::UInt8(a), MapKey::UInt8(b)) => a == b,
            (MapKey::UInt16(a), MapKey::UInt16(b)) => a == b,
            (MapKey::UInt32(a), MapKey::UInt32(b)) => a == b,
            (MapKey::UInt64(a), MapKey::UInt64(b)) => a == b,
            (MapKey::String(a), MapKey::String(b)) => a == b,
            (
                MapKey::Record { type_name: tn1, field_names: fn1, fields: f1 },
                MapKey::Record { type_name: tn2, field_names: fn2, fields: f2 },
            ) => tn1 == tn2 && fn1 == fn2 && f1 == f2,
            (
                MapKey::Variant { type_name: tn1, constructor: c1, fields: f1 },
                MapKey::Variant { type_name: tn2, constructor: c2, fields: f2 },
            ) => tn1 == tn2 && c1 == c2 && f1 == f2,
            _ => false,
        }
    }
}

impl Eq for MapKey {}

// Manual implementation of Hash for MapKey
impl std::hash::Hash for MapKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            MapKey::Unit => {}
            MapKey::Bool(b) => b.hash(state),
            MapKey::Char(c) => c.hash(state),
            MapKey::Int8(n) => n.hash(state),
            MapKey::Int16(n) => n.hash(state),
            MapKey::Int32(n) => n.hash(state),
            MapKey::Int64(n) => n.hash(state),
            MapKey::UInt8(n) => n.hash(state),
            MapKey::UInt16(n) => n.hash(state),
            MapKey::UInt32(n) => n.hash(state),
            MapKey::UInt64(n) => n.hash(state),
            MapKey::String(s) => s.hash(state),
            MapKey::Record { type_name, field_names, fields } => {
                type_name.hash(state);
                for name in field_names {
                    name.hash(state);
                }
                for field in fields {
                    field.hash(state);
                }
            }
            MapKey::Variant { type_name, constructor, fields } => {
                type_name.hash(state);
                constructor.hash(state);
                for field in fields {
                    field.hash(state);
                }
            }
        }
    }
}

/// A record value with named fields.
#[derive(Clone)]
pub struct RecordValue {
    /// Type name (e.g., "Point")
    pub type_name: String,
    /// Field names in order
    pub field_names: Vec<String>,
    /// Field values (parallel to field_names)
    pub fields: Vec<Value>,
    /// Which fields are mutable
    pub mutable_fields: Vec<bool>,
}

/// A reactive record with parent tracking for automatic update propagation.
/// Field assignments propagate to all parent containers automatically.
#[derive(Clone)]
pub struct ReactiveRecordValue {
    /// Type name (e.g., "Point")
    pub type_name: String,
    /// Field names in order
    pub field_names: Vec<String>,
    /// Field values (parallel to field_names) - wrapped in RwLock for mutation
    pub fields: Arc<std::sync::RwLock<Vec<Value>>>,
    /// Bitmask of which fields contain reactive records (for .children)
    pub reactive_field_mask: u64,
    /// Parent references: (weak_parent, field_index_in_parent)
    /// Uses Weak to avoid reference cycles
    pub parents: Arc<std::sync::RwLock<Vec<(std::sync::Weak<ReactiveRecordValue>, u16)>>>,
}

impl ReactiveRecordValue {
    /// Create a new reactive record
    pub fn new(type_name: String, field_names: Vec<String>, fields: Vec<Value>, reactive_field_mask: u64) -> Self {
        ReactiveRecordValue {
            type_name,
            field_names,
            fields: Arc::new(std::sync::RwLock::new(fields)),
            reactive_field_mask,
            parents: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// Get a field value by index
    pub fn get_field(&self, index: usize) -> Option<Value> {
        self.fields.read().ok()?.get(index).cloned()
    }

    /// Set a field value by index and propagate to parents
    pub fn set_field(&self, index: usize, value: Value) -> bool {
        if let Ok(mut fields) = self.fields.write() {
            if index < fields.len() {
                fields[index] = value;
                // Note: parent propagation is handled by the VM instruction
                return true;
            }
        }
        false
    }

    /// Add a parent reference
    pub fn add_parent(&self, parent: std::sync::Weak<ReactiveRecordValue>, field_index: u16) {
        if let Ok(mut parents) = self.parents.write() {
            parents.push((parent, field_index));
        }
    }

    /// Remove a specific parent reference
    pub fn remove_parent(&self, parent_ptr: *const ReactiveRecordValue) {
        if let Ok(mut parents) = self.parents.write() {
            parents.retain(|(weak, _)| {
                weak.upgrade().map_or(true, |arc| Arc::as_ptr(&arc) != parent_ptr)
            });
        }
    }

    /// Clean up expired weak references
    pub fn cleanup_parents(&self) {
        if let Ok(mut parents) = self.parents.write() {
            parents.retain(|(weak, _)| weak.strong_count() > 0);
        }
    }

    /// Get list of (parent, field_name) pairs for introspection
    pub fn get_parents(&self) -> Vec<(Arc<ReactiveRecordValue>, String)> {
        let mut result = Vec::new();
        if let Ok(parents) = self.parents.read() {
            for (weak, field_idx) in parents.iter() {
                if let Some(parent) = weak.upgrade() {
                    let field_name = parent.field_names.get(*field_idx as usize)
                        .cloned()
                        .unwrap_or_default();
                    result.push((parent, field_name));
                }
            }
        }
        result
    }

    /// Get list of reactive children for introspection
    pub fn get_children(&self) -> Vec<Arc<ReactiveRecordValue>> {
        let mut result = Vec::new();
        if let Ok(fields) = self.fields.read() {
            for (i, field) in fields.iter().enumerate() {
                if (self.reactive_field_mask & (1 << i)) != 0 {
                    if let Value::ReactiveRecord(child) = field {
                        result.push(child.clone());
                    }
                }
            }
        }
        result
    }
}

/// A variant value (tagged union).
#[derive(Clone)]
pub struct VariantValue {
    /// Type name (e.g., "Option")
    pub type_name: Arc<String>,
    /// Constructor name (e.g., "Some")
    pub constructor: Arc<String>,
    /// Payload (positional fields)
    pub fields: Vec<Value>,
    /// Named fields (if any)
    pub named_fields: Option<HashMap<String, Value>>,
}

/// Debug symbol information for a local variable.
#[derive(Debug, Clone)]
pub struct LocalVarSymbol {
    /// Variable name
    pub name: String,
    /// Register where the variable is stored
    pub register: Reg,
}

/// A compiled function.
/// Uses Arc for thread-safe sharing across worker threads.
pub struct FunctionValue {
    /// Function name
    pub name: String,
    /// Number of parameters
    pub arity: usize,
    /// Parameter names (for introspection)
    pub param_names: Vec<String>,
    /// Bytecode instructions
    pub code: Arc<Chunk>,
    /// Module this function belongs to
    pub module: Option<String>,
    /// Source location for debugging
    pub source_span: Option<(usize, usize)>,
    /// JIT-compiled version (if available)
    pub jit_code: Option<JitFunction>,
    /// Call counter for JIT hot detection (thread-safe for multi-CPU execution)
    pub call_count: AtomicU32,
    /// Debug symbols: local variable names and their registers
    pub debug_symbols: Vec<LocalVarSymbol>,

    // === REPL Introspection Fields ===

    /// Original source code of this function
    pub source_code: Option<Arc<String>>,
    /// Path to source file (or "<repl>" for REPL definitions)
    pub source_file: Option<String>,
    /// Doc comment from source
    pub doc: Option<String>,
    /// Type signature as displayable string (e.g., "Int -> Int -> Bool")
    pub signature: Option<String>,
    /// Parameter types as strings (e.g., ["Int", "String"])
    pub param_types: Vec<String>,
    /// Return type as string
    pub return_type: Option<String>,
}

impl FunctionValue {
    /// Create a simple function value for testing.
    /// REPL introspection fields are set to defaults.
    pub fn new_simple(name: String, arity: usize, param_names: Vec<String>, code: Arc<Chunk>) -> Self {
        Self {
            name,
            arity,
            param_names,
            code,
            module: None,
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols: vec![],
            // REPL introspection fields - default to None/empty for simple construction
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        }
    }
}

impl Clone for FunctionValue {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            arity: self.arity,
            param_names: self.param_names.clone(),
            code: self.code.clone(),
            module: self.module.clone(),
            source_span: self.source_span,
            jit_code: self.jit_code.clone(),
            call_count: AtomicU32::new(self.call_count.load(std::sync::atomic::Ordering::Relaxed)),
            debug_symbols: self.debug_symbols.clone(),
            // REPL introspection fields
            source_code: self.source_code.clone(),
            source_file: self.source_file.clone(),
            doc: self.doc.clone(),
            signature: self.signature.clone(),
            param_types: self.param_types.clone(),
            return_type: self.return_type.clone(),
        }
    }
}

impl std::fmt::Debug for FunctionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionValue")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .field("param_names", &self.param_names)
            .field("module", &self.module)
            .field("source_span", &self.source_span)
            .field("jit_code", &self.jit_code.is_some())
            .finish()
    }
}

/// A closure (function + captured environment).
#[derive(Clone)]
pub struct ClosureValue {
    /// The underlying function
    pub function: Arc<FunctionValue>,
    /// Captured variables from enclosing scope
    pub captures: Vec<Value>,
    /// Captured variable names (for introspection)
    pub capture_names: Vec<String>,
}

/// A native (Rust) function.
pub struct NativeFn {
    pub name: String,
    pub arity: usize,
    pub func: Box<dyn Fn(&[Value]) -> Result<Value, RuntimeError> + Send + Sync>,
}

/// Process ID for concurrency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pid(pub u64);

/// Reference ID for monitors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RefId(pub u64);

/// Type information for introspection.
#[derive(Clone)]
pub struct TypeValue {
    pub name: String,
    pub kind: TypeKind,
    pub fields: Vec<FieldInfo>,
    pub constructors: Vec<ConstructorInfo>,
    /// Derived traits (e.g., ["Hash", "Eq", "Show"])
    pub traits: Vec<String>,

    // === REPL Introspection Fields ===

    /// Original source code of this type definition
    pub source_code: Option<String>,
    /// Path to source file (or "<repl>" for REPL definitions)
    pub source_file: Option<String>,
    /// Doc comment from source
    pub doc: Option<String>,
    /// Type parameters (e.g., ["T", "E"] for Result[T, E])
    pub type_params: Vec<String>,
}

#[derive(Clone)]
pub enum TypeKind {
    Primitive,
    Record { mutable: bool },
    Reactive,
    Variant,
    Alias { target: String },
}

#[derive(Clone)]
pub struct FieldInfo {
    pub name: String,
    pub type_name: String,
    pub mutable: bool,
    pub private: bool,
}

#[derive(Clone)]
pub struct ConstructorInfo {
    pub name: String,
    pub fields: Vec<FieldInfo>,
}

/// Placeholder for JIT-compiled code.
/// Will be replaced with actual Cranelift function pointer.
#[derive(Clone)]
pub struct JitFunction {
    pub ptr: *const u8,
    pub size: usize,
}

// Safety: JIT code pointers are immutable once created
unsafe impl Send for JitFunction {}
unsafe impl Sync for JitFunction {}

/// Runtime errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError {
    #[error("Type error: expected {expected}, got {found}")]
    TypeError { expected: String, found: String },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Index out of bounds: {index} (length {length})")]
    IndexOutOfBounds { index: i64, length: usize },

    #[error("Unknown field: {field} on type {type_name}")]
    UnknownField { type_name: String, field: String },

    #[error("Cannot mutate immutable field: {field}")]
    ImmutableField { field: String },

    #[error("Cannot mutate immutable binding: {name}")]
    ImmutableBinding { name: String },

    #[error("Unknown variable: {0}")]
    UnknownVariable(String),

    #[error("Unknown function: {0}")]
    UnknownFunction(String),

    #[error("Arity mismatch: expected {expected}, got {found}")]
    ArityMismatch { expected: usize, found: usize },

    #[error("Pattern match failed")]
    MatchFailed,

    #[error("Assertion failed: {0}")]
    AssertionFailed(String),

    #[error("Panic: {0}")]
    Panic(String),

    #[error("Stack overflow")]
    StackOverflow,

    #[error("Process not found: {0:?}")]
    ProcessNotFound(Pid),

    #[error("Timeout")]
    Timeout,

    #[error("I/O error: {0}")]
    IOError(String),

    #[error("Interrupted (Ctrl+C)")]
    Interrupted,

    #[error("{error}\n\nStack trace:\n{stack_trace}")]
    WithStackTrace {
        error: Box<RuntimeError>,
        stack_trace: String,
    },
}

impl RuntimeError {
    /// Wrap this error with a stack trace.
    pub fn with_stack_trace(self, stack_trace: String) -> RuntimeError {
        RuntimeError::WithStackTrace {
            error: Box::new(self),
            stack_trace,
        }
    }
    /// Convert this runtime error to an exception Value that can be caught by try/catch.
    /// Returns a record with `type` and `message` fields.
    pub fn to_exception_value(&self) -> Value {
        let (error_type, message) = match self {
            RuntimeError::TypeError { expected, found } => {
                ("TypeError", format!("expected {}, got {}", expected, found))
            }
            RuntimeError::DivisionByZero => {
                ("DivisionByZero", "division by zero".to_string())
            }
            RuntimeError::IndexOutOfBounds { index, length } => {
                ("IndexOutOfBounds", format!("index {} out of bounds (length {})", index, length))
            }
            RuntimeError::UnknownField { type_name, field } => {
                ("UnknownField", format!("unknown field '{}' on type '{}'", field, type_name))
            }
            RuntimeError::ImmutableField { field } => {
                ("ImmutableField", format!("cannot mutate immutable field '{}'", field))
            }
            RuntimeError::ImmutableBinding { name } => {
                ("ImmutableBinding", format!("cannot mutate immutable binding '{}'", name))
            }
            RuntimeError::UnknownVariable(name) => {
                ("UnknownVariable", format!("unknown variable '{}'", name))
            }
            RuntimeError::UnknownFunction(name) => {
                ("UnknownFunction", format!("unknown function '{}'", name))
            }
            RuntimeError::ArityMismatch { expected, found } => {
                ("ArityMismatch", format!("expected {} arguments, got {}", expected, found))
            }
            RuntimeError::MatchFailed => {
                ("MatchFailed", "pattern match failed".to_string())
            }
            RuntimeError::AssertionFailed(msg) => {
                ("AssertionFailed", msg.clone())
            }
            RuntimeError::Panic(msg) => {
                ("Panic", msg.clone())
            }
            RuntimeError::StackOverflow => {
                ("StackOverflow", "stack overflow".to_string())
            }
            RuntimeError::ProcessNotFound(pid) => {
                ("ProcessNotFound", format!("process {:?} not found", pid))
            }
            RuntimeError::Timeout => {
                ("Timeout", "operation timed out".to_string())
            }
            RuntimeError::IOError(msg) => {
                ("IOError", msg.clone())
            }
            RuntimeError::WithStackTrace { error, stack_trace: _stack_trace } => {
                // Delegate to the inner error, but append stack trace to message
                let inner = error.to_exception_value();
                // For now, just return the inner error's value
                // Stack traces are primarily for CLI display, not try/catch
                return inner;
            }
            RuntimeError::Interrupted => {
                ("Interrupted", "execution interrupted (Ctrl+C)".to_string())
            }
        };

        // Create a record-like structure: Error{type: "...", message: "..."}
        let record = RecordValue {
            type_name: "Error".to_string(),
            field_names: vec!["type".to_string(), "message".to_string()],
            fields: vec![
                Value::String(Arc::new(error_type.to_string())),
                Value::String(Arc::new(message)),
            ],
            mutable_fields: vec![false, false],
        };

        Value::Record(Arc::new(record))
    }
}

// Value is Send+Sync: all variants contain only Send+Sync types
// (primitives, Arc, String which are all Send+Sync)
unsafe impl Send for Value {}
unsafe impl Sync for Value {}

/// A chunk of bytecode.
#[derive(Clone, Default)]
pub struct Chunk {
    /// Bytecode instructions
    pub code: Vec<Instruction>,
    /// Constant pool
    pub constants: Vec<Value>,
    /// Line number information for debugging
    pub lines: Vec<usize>,
    /// Local variable names (for debugging/introspection)
    pub locals: Vec<String>,
    /// Number of registers needed
    pub register_count: usize,
}

/// Register index (u8 for compact bytecode).
pub type Reg = u8;

/// Constant pool index.
pub type ConstIdx = u16;

/// Jump offset.
pub type JumpOffset = i16;

/// Bytecode instructions.
///
/// Register-based design for efficient JIT compilation to Cranelift SSA.
/// Explicit tail call instruction for proper TCO.
#[derive(Clone, Debug)]
pub enum Instruction {
    // === Constants and moves ===
    /// Load constant: dst = constants[idx]
    LoadConst(Reg, ConstIdx),
    /// Copy: dst = src
    Move(Reg, Reg),
    /// Load unit: dst = ()
    LoadUnit(Reg),
    /// Load true: dst = true
    LoadTrue(Reg),
    /// Load false: dst = false
    LoadFalse(Reg),

    // === Arithmetic (typed for JIT) ===
    AddInt(Reg, Reg, Reg),      // dst = a + b (Int)
    SubInt(Reg, Reg, Reg),
    MulInt(Reg, Reg, Reg),
    DivInt(Reg, Reg, Reg),
    ModInt(Reg, Reg, Reg),
    NegInt(Reg, Reg),           // dst = -src

    AddFloat(Reg, Reg, Reg),    // dst = a + b (Float)
    SubFloat(Reg, Reg, Reg),
    MulFloat(Reg, Reg, Reg),
    DivFloat(Reg, Reg, Reg),
    NegFloat(Reg, Reg),
    PowFloat(Reg, Reg, Reg),    // dst = a ** b

    // === Comparison ===
    EqInt(Reg, Reg, Reg),       // dst = a == b
    NeInt(Reg, Reg, Reg),
    LtInt(Reg, Reg, Reg),
    LeInt(Reg, Reg, Reg),
    GtInt(Reg, Reg, Reg),
    GeInt(Reg, Reg, Reg),

    EqFloat(Reg, Reg, Reg),
    LtFloat(Reg, Reg, Reg),
    LeFloat(Reg, Reg, Reg),

    EqBool(Reg, Reg, Reg),
    EqStr(Reg, Reg, Reg),

    /// Generic equality (for variants, records)
    Eq(Reg, Reg, Reg),

    // === Logical ===
    Not(Reg, Reg),              // dst = !src
    And(Reg, Reg, Reg),         // dst = a && b (no short-circuit at bytecode level)
    Or(Reg, Reg, Reg),          // dst = a || b

    // === String operations ===
    Concat(Reg, Reg, Reg),      // dst = a ++ b

    // === Collections ===
    /// Create list from registers: dst = [regs...]
    MakeList(Reg, RegList),
    /// Create tuple: dst = (regs...)
    MakeTuple(Reg, RegList),
    /// Create map: dst = %{keys: values}
    MakeMap(Reg, Arc<[(Reg, Reg)]>),
    /// Create set: dst = #{regs...}
    MakeSet(Reg, RegList),
    /// Check if map contains key: dst = map.contains_key(key)
    MapContainsKey(Reg, Reg, Reg),
    /// Get value from map: dst = map[key] (panics if not found)
    MapGet(Reg, Reg, Reg),
    /// Check if set contains value: dst = set.contains(value)
    SetContains(Reg, Reg, Reg),
    /// List cons: dst = [head | tail]
    Cons(Reg, Reg, Reg),
    /// Create Int64List from i64 registers: dst = Int64List[regs...]
    MakeInt64List(Reg, RegList),
    /// List concat: dst = a ++ b
    ListConcat(Reg, Reg, Reg),
    /// Index access: dst = collection[index]
    Index(Reg, Reg, Reg),
    /// Index assignment: collection[index] = value
    IndexSet(Reg, Reg, Reg),
    /// List length: dst = len(list)
    Length(Reg, Reg),

    // === Typed Arrays ===
    /// Create Int64 array: dst = Int64Array of given size, filled with 0
    MakeInt64Array(Reg, Reg),
    /// Create Float64 array: dst = Float64Array of given size, filled with 0.0
    MakeFloat64Array(Reg, Reg),

    // === Tuples ===
    /// Get tuple element: dst = tuple[idx]
    GetTupleField(Reg, Reg, u8),

    // === Records ===
    /// Create record: dst = TypeName{fields...}
    MakeRecord(Reg, ConstIdx, RegList),
    /// Create reactive record: dst = reactive TypeName{fields...}
    MakeReactiveRecord(Reg, ConstIdx, RegList),
    /// Get field: dst = record.field
    GetField(Reg, Reg, ConstIdx),
    /// Set field: record.field = value
    SetField(Reg, ConstIdx, Reg),
    /// Update record: dst = base with new field values
    UpdateRecord(Reg, Reg, ConstIdx, RegList),

    // === Variants ===
    /// Create variant: dst = Constructor(fields...)
    MakeVariant(Reg, ConstIdx, ConstIdx, RegList), // dst, type_idx, ctor_idx, fields
    /// Get variant tag: dst = variant.tag (as int)
    GetTag(Reg, Reg),
    /// Get variant field: dst = variant.fields[idx]
    GetVariantField(Reg, Reg, u8),
    /// Get variant named field: dst = variant.named_fields[name]
    GetVariantFieldByName(Reg, Reg, ConstIdx),

    // === Control flow ===
    /// Unconditional jump
    Jump(JumpOffset),
    /// Jump if true
    JumpIfTrue(Reg, JumpOffset),
    /// Jump if false
    JumpIfFalse(Reg, JumpOffset),

    // === Function calls ===
    /// Call function: dst = func(args...)
    Call(Reg, Reg, RegList),
    /// Tail call (reuse current frame): return func(args...)
    TailCall(Reg, RegList),
    /// Call function by name (looks up in VM's function map) - SLOW, use for dynamic dispatch only
    CallByName(Reg, ConstIdx, RegList),
    /// Tail call by name - SLOW, use for dynamic dispatch only
    TailCallByName(ConstIdx, RegList),
    /// Call function directly by index (no HashMap lookup!)
    CallDirect(Reg, u16, RegList),
    /// Tail call function directly by index (no HashMap lookup!)
    TailCallDirect(u16, RegList),
    /// Call self (recursive call to current function - no lookup needed)
    CallSelf(Reg, RegList),
    /// Tail call self (tail-recursive call to current function)
    TailCallSelf(RegList),
    /// Call native function (string-based lookup)
    CallNative(Reg, ConstIdx, RegList),
    /// Call native function by index (fast path - no string lookup)
    CallNativeIdx(Reg, u16, RegList),
    /// Call extension function (dynamic library): dst = ext_func(args...)
    CallExtension(Reg, ConstIdx, RegList),
    /// Call extension function by index (fast path - no string lookup)
    CallExtensionIdx(Reg, u16, RegList),
    /// Return value from function
    Return(Reg),

    // === Closures ===
    /// Create closure: dst = closure(func_idx, captures...)
    MakeClosure(Reg, ConstIdx, RegList),
    /// Get captured variable: dst = captures[idx]
    GetCapture(Reg, u8),

    // === Pattern matching ===
    /// Test if value matches constructor tag by name (stored in constants)
    /// dst = value.constructor == constants[ctor_idx]
    TestTag(Reg, Reg, ConstIdx),
    /// Test if value is unit: dst = value == ()
    TestUnit(Reg, Reg),
    /// Test if value equals constant: dst = value == const
    TestConst(Reg, Reg, ConstIdx),
    /// Test if list is empty: dst = list == []
    TestNil(Reg, Reg),
    /// Deconstruct list: head, tail = list (fails if empty)
    Decons(Reg, Reg, Reg),
    /// Deconstruct string: head_char, tail_str = string (fails if empty)
    /// head_char is a String containing first character, tail_str is rest of string
    StringDecons(Reg, Reg, Reg),
    /// Test if string is empty: dst = str == ""
    TestEmptyString(Reg, Reg),

    // === Specialized Int64List operations (avoid GcValue boxing) ===
    /// Test if Int64List is empty: dst = list == []
    TestNilInt64(Reg, Reg),
    /// Deconstruct Int64List: head_i64, tail_list = list
    DeconsInt64(Reg, Reg, Reg),
    /// Cons for Int64List: dst = [head | tail]
    ConsInt64(Reg, Reg, Reg),
    /// Create Int64List from range: dst = [n, n-1, ..., 1]
    RangeInt64List(Reg, Reg),
    /// Convert List to Int64List (if all elements are Int64)
    ToInt64List(Reg, Reg),
    /// Sum Int64List directly: dst = sum(list)
    SumInt64List(Reg, Reg),
    /// Test if value is a Map: dst = is_map(val)
    IsMap(Reg, Reg),
    /// Test if value is a Set: dst = is_set(val)
    IsSet(Reg, Reg),

    // === Concurrency ===
    /// Spawn process: dst = spawn(func, args)
    Spawn(Reg, Reg, RegList),
    /// Spawn linked: dst = spawn_link(func, args)
    SpawnLink(Reg, Reg, RegList),
    /// Spawn monitored: (pid, ref) = spawn_monitor(func, args)
    SpawnMonitor(Reg, Reg, Reg, RegList),
    /// Send message: pid <- msg
    Send(Reg, Reg),
    /// Get self PID: dst = self()
    SelfPid(Reg),
    /// Receive message into dst register (handled specially by VM - switches to receive mode)
    Receive(Reg),
    /// Receive with timeout: dst = receive or Unit if timeout (timeout_ms in timeout_reg)
    ReceiveTimeout(Reg, Reg),
    /// Sleep for N milliseconds (value in register)
    Sleep(Reg),
    /// Get VM stats: dst = (spawned, exited, active) process counts
    VmStats(Reg),

    // === Process introspection ===
    /// Get all process IDs: dst = Process.all()
    ProcessAll(Reg),
    /// Get process uptime in ms: dst = Process.time(pid)
    ProcessTime(Reg, Reg),
    /// Check if process is alive: dst = Process.alive(pid)
    ProcessAlive(Reg, Reg),
    /// Get process info record: dst = Process.info(pid)
    ProcessInfo(Reg, Reg),
    /// Kill a process: Process.kill(pid)
    ProcessKill(Reg, Reg),

    // === External process execution ===
    /// Run command and wait: dst = Exec.run(cmd, args)
    ExecRun(Reg, Reg, Reg),
    /// Spawn process with streaming: dst = Exec.spawn(cmd, args)
    ExecSpawn(Reg, Reg, Reg),
    /// Read line from stdout: dst = Exec.readline(handle)
    ExecReadLine(Reg, Reg),
    /// Read line from stderr: dst = Exec.readStderr(handle)
    ExecReadStderr(Reg, Reg),
    /// Write to stdin: Exec.write(handle, data)
    ExecWrite(Reg, Reg, Reg),
    /// Wait for process exit: dst = Exec.wait(handle)
    ExecWait(Reg, Reg),
    /// Kill process: Exec.kill(handle)
    ExecKill(Reg, Reg),

    // === Error handling ===
    /// Push exception handler
    PushHandler(JumpOffset),
    /// Pop exception handler
    PopHandler,
    /// Throw exception
    Throw(Reg),
    /// Get current exception (in catch block)
    GetException(Reg),

    // === Introspection ===
    /// Get type of value: dst = typeof(value)
    TypeOf(Reg, Reg),

    // === Builtin math (compile-time resolved, no runtime dispatch) ===
    /// Absolute value: dst = abs(src) for Int
    AbsInt(Reg, Reg),
    /// Absolute value: dst = abs(src) for Float
    AbsFloat(Reg, Reg),
    /// Square root: dst = sqrt(src) for Float
    SqrtFloat(Reg, Reg),
    /// Minimum of two Ints: dst = min(a, b)
    MinInt(Reg, Reg, Reg),
    /// Maximum of two Ints: dst = max(a, b)
    MaxInt(Reg, Reg, Reg),
    /// Minimum of two Floats: dst = min(a, b)
    MinFloat(Reg, Reg, Reg),
    /// Maximum of two Floats: dst = max(a, b)
    MaxFloat(Reg, Reg, Reg),
    /// Sine: dst = sin(src) for Float (radians)
    SinFloat(Reg, Reg),
    /// Cosine: dst = cos(src) for Float (radians)
    CosFloat(Reg, Reg),
    /// Tangent: dst = tan(src) for Float (radians)
    TanFloat(Reg, Reg),
    /// Floor: dst = floor(src) - round down to Int
    FloorFloat(Reg, Reg),
    /// Ceiling: dst = ceil(src) - round up to Int
    CeilFloat(Reg, Reg),
    /// Round: dst = round(src) - round to nearest Int
    RoundFloat(Reg, Reg),
    /// Natural logarithm: dst = log(src) for Float
    LogFloat(Reg, Reg),
    /// Base-10 logarithm: dst = log10(src) for Float
    Log10Float(Reg, Reg),

    // === Type conversions (compile-time resolved) ===
    /// Int to Float: dst = toFloat(src)
    IntToFloat(Reg, Reg),
    /// Float to Int: dst = toInt(src)
    FloatToInt(Reg, Reg),
    /// Convert to Int8: dst = toInt8(src)
    ToInt8(Reg, Reg),
    /// Convert to Int16: dst = toInt16(src)
    ToInt16(Reg, Reg),
    /// Convert to Int32: dst = toInt32(src)
    ToInt32(Reg, Reg),
    /// Convert to UInt8: dst = toUInt8(src)
    ToUInt8(Reg, Reg),
    /// Convert to UInt16: dst = toUInt16(src)
    ToUInt16(Reg, Reg),
    /// Convert to UInt32: dst = toUInt32(src)
    ToUInt32(Reg, Reg),
    /// Convert to UInt64: dst = toUInt64(src)
    ToUInt64(Reg, Reg),
    /// Convert to Float32: dst = toFloat32(src)
    ToFloat32(Reg, Reg),
    /// Convert to BigInt: dst = toBigInt(src)
    ToBigInt(Reg, Reg),

    // === List operations (compile-time resolved) ===
    /// Head of list: dst = head(list)
    ListHead(Reg, Reg),
    /// Tail of list: dst = tail(list)
    ListTail(Reg, Reg),
    /// Is list empty: dst = isEmpty(list)
    ListIsEmpty(Reg, Reg),
    /// Sum list of integers: dst = listSum(list)
    ListSum(Reg, Reg),
    /// Product of list of integers: dst = listProduct(list)
    ListProduct(Reg, Reg),
    /// Maximum of list of integers: dst = listMax(list)
    ListMax(Reg, Reg),
    /// Minimum of list of integers: dst = listMin(list)
    ListMin(Reg, Reg),
    /// Create list [1..n]: dst = rangeList(n)
    RangeList(Reg, Reg),

    // === Async IO operations ===
    /// Read entire file as string: dst = File.readAll(path)
    /// Suspends process until IO completes
    FileReadAll(Reg, Reg),
    /// Write string to file: dst = File.writeAll(path, content)
    /// Suspends process until IO completes
    FileWriteAll(Reg, Reg, Reg),
    /// HTTP GET request: dst = Http.get(url)
    /// Returns ("ok", HttpResponse{status, headers, body}) or ("error", msg)
    /// Suspends process until request completes
    HttpGet(Reg, Reg),
    /// HTTP POST request: dst = Http.post(url, body)
    HttpPost(Reg, Reg, Reg),
    /// HTTP PUT request: dst = Http.put(url, body)
    HttpPut(Reg, Reg, Reg),
    /// HTTP DELETE request: dst = Http.delete(url)
    HttpDelete(Reg, Reg),
    /// HTTP PATCH request: dst = Http.patch(url, body)
    HttpPatch(Reg, Reg, Reg),
    /// HTTP HEAD request: dst = Http.head(url)
    HttpHead(Reg, Reg),
    /// Generic HTTP request: dst = Http.request(method, url, headers, body)
    /// method: "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"
    /// headers: list of (name, value) tuples
    /// body: string or () for no body
    HttpRequest(Reg, Reg, Reg, Reg, Reg),

    // === File Handle Operations ===
    /// Open file: dst = File.open(path, mode)
    /// mode: "r" (read), "w" (write/create), "a" (append), "rw" (read+write)
    /// Returns (status, handle) where handle is Int
    FileOpen(Reg, Reg, Reg),
    /// Write to file handle: dst = File.write(handle, data)
    /// Returns (status, bytes_written)
    FileWrite(Reg, Reg, Reg),
    /// Read from file handle: dst = File.read(handle, size)
    /// Returns (status, data)
    FileRead(Reg, Reg, Reg),
    /// Read line from file handle: dst = File.readLine(handle)
    /// Returns (status, line) where line is string or None at EOF
    FileReadLine(Reg, Reg),
    /// Flush file handle: dst = File.flush(handle)
    FileFlush(Reg, Reg),
    /// Close file handle: dst = File.close(handle)
    FileClose(Reg, Reg),
    /// Seek in file: dst = File.seek(handle, offset, whence)
    /// whence: "start", "current", "end"
    FileSeek(Reg, Reg, Reg, Reg),

    // === Directory Operations ===
    /// Create directory: dst = Dir.create(path)
    DirCreate(Reg, Reg),
    /// Create directory recursively: dst = Dir.createAll(path)
    DirCreateAll(Reg, Reg),
    /// List directory contents: dst = Dir.list(path)
    /// Returns (status, [entries])
    DirList(Reg, Reg),
    /// Remove empty directory: dst = Dir.remove(path)
    DirRemove(Reg, Reg),
    /// Remove directory recursively: dst = Dir.removeAll(path)
    DirRemoveAll(Reg, Reg),

    // === File Utilities ===
    /// Check if file exists: dst = File.exists(path)
    FileExists(Reg, Reg),
    /// Check if path is directory: dst = Dir.exists(path)
    DirExists(Reg, Reg),
    /// Remove file: dst = File.remove(path)
    FileRemove(Reg, Reg),
    /// Rename/move file: dst = File.rename(old, new)
    FileRename(Reg, Reg, Reg),
    /// Copy file: dst = File.copy(src, dest)
    FileCopy(Reg, Reg, Reg),
    /// Get file size: dst = File.size(path)
    FileSize(Reg, Reg),
    /// Append string to file: dst = File.append(path, content)
    FileAppend(Reg, Reg, Reg),

    // === String Encoding ===
    /// Base64 encode: dst = Base64.encode(string)
    Base64Encode(Reg, Reg),
    /// Base64 decode: dst = Base64.decode(string)
    Base64Decode(Reg, Reg),
    /// URL encode: dst = Url.encode(string)
    UrlEncode(Reg, Reg),
    /// URL decode: dst = Url.decode(string)
    UrlDecode(Reg, Reg),
    /// UTF-8 encode string to bytes: dst = Encoding.toBytes(string)
    Utf8Encode(Reg, Reg),
    /// UTF-8 decode bytes to string: dst = Encoding.fromBytes(bytes)
    Utf8Decode(Reg, Reg),

    // === HTTP Server Operations ===
    /// Bind and start HTTP server: dst = Server.bind(port)
    /// Returns (status, server_handle)
    ServerBind(Reg, Reg),
    /// Accept next HTTP request: dst = Server.accept(server_handle)
    /// Returns (status, HttpRequest{id, method, path, headers, body})
    /// Suspends process until request arrives
    ServerAccept(Reg, Reg),
    /// Respond to HTTP request: dst = Server.respond(request_id, status, headers, body)
    /// Returns (status, ())
    ServerRespond(Reg, Reg, Reg, Reg, Reg),
    /// Close HTTP server: dst = Server.close(server_handle)
    ServerClose(Reg, Reg),

    // === WebSocket Operations ===
    /// Accept WebSocket upgrade: dst = WebSocket.accept(request_id)
    WebSocketAccept(Reg, Reg),
    /// Send message on WebSocket: dst = WebSocket.send(request_id, message)
    WebSocketSend(Reg, Reg, Reg),
    /// Receive message from WebSocket: dst = WebSocket.receive(request_id)
    WebSocketReceive(Reg, Reg),
    /// Close WebSocket: dst = WebSocket.close(request_id)
    WebSocketClose(Reg, Reg),

    // === PostgreSQL Operations ===
    /// Connect to PostgreSQL: dst = Pg.connect(connection_string)
    /// Returns (status, handle) where handle is Int
    PgConnect(Reg, Reg),
    /// Execute query and return rows: dst = Pg.query(handle, query, params)
    /// params is a list of values
    /// Returns (status, rows) where rows is List[List[value]]
    PgQuery(Reg, Reg, Reg, Reg),
    /// Execute statement: dst = Pg.execute(handle, query, params)
    /// Returns (status, affected_rows)
    PgExecute(Reg, Reg, Reg, Reg),
    /// Close PostgreSQL connection: dst = Pg.close(handle)
    PgClose(Reg, Reg),
    /// Begin transaction: dst = Pg.begin(handle)
    PgBegin(Reg, Reg),
    /// Commit transaction: dst = Pg.commit(handle)
    PgCommit(Reg, Reg),
    /// Rollback transaction: dst = Pg.rollback(handle)
    PgRollback(Reg, Reg),
    /// Prepare statement: dst = Pg.prepare(handle, name, query)
    PgPrepare(Reg, Reg, Reg, Reg),
    /// Query with prepared statement: dst = Pg.queryPrepared(handle, name, params)
    PgQueryPrepared(Reg, Reg, Reg, Reg),
    /// Execute with prepared statement: dst = Pg.executePrepared(handle, name, params)
    PgExecutePrepared(Reg, Reg, Reg, Reg),
    /// Deallocate prepared statement: dst = Pg.deallocate(handle, name)
    PgDeallocate(Reg, Reg, Reg),
    /// Create listener connection: dst = Pg.listenConnect(connection_string)
    PgListenConnect(Reg, Reg),
    /// Start listening on channel: dst = Pg.listen(handle, channel)
    PgListen(Reg, Reg, Reg),
    /// Stop listening on channel: dst = Pg.unlisten(handle, channel)
    PgUnlisten(Reg, Reg, Reg),
    /// Send notification: dst = Pg.notify(handle, channel, payload)
    PgNotify(Reg, Reg, Reg, Reg),
    /// Wait for notification: dst = Pg.awaitNotification(handle, timeout_ms)
    PgAwaitNotification(Reg, Reg, Reg),

    // === Time builtins ===
    /// Get current UTC timestamp in milliseconds: dst = Time.now()
    TimeNow(Reg),
    /// Create timestamp from date parts: dst = Time.fromDate(year, month, day)
    TimeFromDate(Reg, Reg, Reg, Reg),
    /// Create time millis from time parts: dst = Time.fromTime(hour, min, sec)
    TimeFromTime(Reg, Reg, Reg, Reg),
    /// Create full timestamp: dst = Time.fromDateTime(year, month, day, hour, min, sec)
    TimeFromDateTime(Reg, Reg, Reg, Reg, Reg, Reg, Reg),
    /// Extract year from timestamp: dst = Time.year(ts)
    TimeYear(Reg, Reg),
    /// Extract month from timestamp: dst = Time.month(ts)
    TimeMonth(Reg, Reg),
    /// Extract day from timestamp: dst = Time.day(ts)
    TimeDay(Reg, Reg),
    /// Extract hour from timestamp: dst = Time.hour(ts)
    TimeHour(Reg, Reg),
    /// Extract minute from timestamp: dst = Time.minute(ts)
    TimeMinute(Reg, Reg),
    /// Extract second from timestamp: dst = Time.second(ts)
    TimeSecond(Reg, Reg),

    // === Type introspection and reflection ===
    /// Get type info by name: dst = Type.info(name_reg)
    /// Returns Json with fields, constructors, etc.
    TypeInfo(Reg, Reg),
    /// Get variant tag: dst = tagOf(value)
    /// Returns tag name for variants, empty string otherwise
    TagOf(Reg, Reg),
    /// Reflect value to Json: dst = reflect(value)
    /// Converts any value to the Json sum type
    Reflect(Reg, Reg),
    /// Construct typed value from Json: dst = construct(type_name, json)
    /// Creates a record or variant from type name string and Json data
    Construct(Reg, Reg, Reg),
    /// Make record from field map: dst = makeRecord[T](fields_map)
    /// fields_map is a Map[String, Json] where keys are field names
    MakeRecordDyn(Reg, Reg, Reg),
    /// Make variant from constructor name and fields: dst = makeVariant[T](ctor_name, fields_map)
    /// fields_map is a Map[String, Json] where keys are field names
    MakeVariantDyn(Reg, Reg, Reg, Reg),
    /// Parse HTTP request params to typed record: dst = requestToType(request, type_name)
    /// Returns Result[T, String] - Ok(record) or Err(error_message)
    RequestToType(Reg, Reg, Reg),

    // === IO/Debug builtins ===
    /// Print value, return string representation: dst = print(value)
    Print(Reg, Reg),
    /// Print value with newline, return unit: println(value)
    Println(Reg),
    /// Panic with message
    Panic(Reg),

    // === Assertions ===
    /// Assert value is true
    Assert(Reg),
    /// Assert two values are equal
    AssertEq(Reg, Reg),

    // === Module-level mutable variables (mvars) ===
    /// Acquire lock on mvar at function entry: name_idx, is_write
    /// Must be released with MvarUnlock before function returns
    MvarLock(ConstIdx, bool),
    /// Release lock on mvar at function exit: name_idx, is_write
    MvarUnlock(ConstIdx, bool),
    /// Read mvar: dst = mvar[name_idx] (assumes lock is already held)
    MvarRead(Reg, ConstIdx),
    /// Write mvar: mvar[name_idx] = src (assumes lock is already held)
    MvarWrite(ConstIdx, Reg),

    // === String Buffer (for efficient HTML rendering) ===
    /// Create a new string buffer: dst = Buffer.new()
    BufferNew(Reg),
    /// Append string to buffer: buffer.append(str)
    BufferAppend(Reg, Reg),
    /// Convert buffer to string: dst = buffer.toString()
    BufferToString(Reg, Reg),

    // === Debug ===
    /// No operation (for alignment/debugging)
    Nop,
    /// Debug print
    DebugPrint(Reg),
}

// Instruction is Send+Sync: all variants contain only primitive types
// (Reg = u8, ConstIdx = u16) and Arc<[...]> which are Send+Sync
unsafe impl Send for Instruction {}
unsafe impl Sync for Instruction {}

impl Chunk {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constant and return its index.
    pub fn add_constant(&mut self, value: Value) -> ConstIdx {
        let idx = self.constants.len();
        self.constants.push(value);
        idx as ConstIdx
    }

    /// Add an instruction and return its offset.
    pub fn emit(&mut self, instr: Instruction, line: usize) -> usize {
        let offset = self.code.len();
        self.code.push(instr);
        self.lines.push(line);
        offset
    }

    /// Patch a jump instruction with the actual offset.
    pub fn patch_jump(&mut self, offset: usize, target: usize) {
        let jump_offset = (target as isize - offset as isize - 1) as JumpOffset;
        match &mut self.code[offset] {
            Instruction::Jump(ref mut off) => *off = jump_offset,
            Instruction::JumpIfTrue(_, ref mut off) => *off = jump_offset,
            Instruction::JumpIfFalse(_, ref mut off) => *off = jump_offset,
            Instruction::PushHandler(ref mut off) => *off = jump_offset,
            _ => panic!("Cannot patch non-jump instruction"),
        }
    }
}

// === Value implementations ===

impl Value {
    /// Get the type name of this value.
    pub fn type_name(&self) -> &str {
        match self {
            Value::Unit => "()",
            Value::Bool(_) => "Bool",
            Value::Char(_) => "Char",
            // Signed integers
            Value::Int8(_) => "Int8",
            Value::Int16(_) => "Int16",
            Value::Int32(_) => "Int32",
            Value::Int64(_) => "Int64",
            // Unsigned integers
            Value::UInt8(_) => "UInt8",
            Value::UInt16(_) => "UInt16",
            Value::UInt32(_) => "UInt32",
            Value::UInt64(_) => "UInt64",
            // Floats
            Value::Float32(_) => "Float32",
            Value::Float64(_) => "Float64",
            // BigInt
            Value::BigInt(_) => "BigInt",
            // Decimal
            Value::Decimal(_) => "Decimal",
            // Collections
            Value::String(_) => "String",
            Value::List(_) => "List",
            Value::Array(_) => "Array",
            Value::Int64Array(_) => "Int64Array",
            Value::Float64Array(_) => "Float64Array",
            Value::Float32Array(_) => "Float32Array",
            Value::Tuple(_) => "Tuple",
            Value::Map(_) => "Map",
            Value::Set(_) => "Set",
            Value::Record(r) => &r.type_name,
            Value::ReactiveRecord(r) => &r.type_name,
            Value::Variant(v) => &v.type_name,
            Value::Function(_) => "Function",
            Value::Closure(_) => "Closure",
            Value::NativeFunction(_) => "NativeFunction",
            Value::Pid(_) => "Pid",
            Value::Ref(_) => "Ref",
            Value::Type(_) => "Type",
            Value::Pointer(_) => "Pointer",
            Value::NativeHandle(_) => "NativeHandle",
        }
    }

    /// Check if this value is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Unit => false,
            _ => true,
        }
    }

    /// Try to convert to MapKey for use in maps/sets.
    pub fn to_map_key(&self) -> Option<MapKey> {
        match self {
            Value::Unit => Some(MapKey::Unit),
            Value::Bool(b) => Some(MapKey::Bool(*b)),
            Value::Char(c) => Some(MapKey::Char(*c)),
            // Signed integers
            Value::Int8(i) => Some(MapKey::Int8(*i)),
            Value::Int16(i) => Some(MapKey::Int16(*i)),
            Value::Int32(i) => Some(MapKey::Int32(*i)),
            Value::Int64(i) => Some(MapKey::Int64(*i)),
            // Unsigned integers
            Value::UInt8(i) => Some(MapKey::UInt8(*i)),
            Value::UInt16(i) => Some(MapKey::UInt16(*i)),
            Value::UInt32(i) => Some(MapKey::UInt32(*i)),
            Value::UInt64(i) => Some(MapKey::UInt64(*i)),
            // String
            Value::String(s) => Some(MapKey::String(s.clone())),
            _ => None,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Char(c) => write!(f, "'{}'", c),
            // Signed integers
            Value::Int8(i) => write!(f, "{}i8", i),
            Value::Int16(i) => write!(f, "{}i16", i),
            Value::Int32(i) => write!(f, "{}i32", i),
            Value::Int64(i) => write!(f, "{}", i),
            // Unsigned integers
            Value::UInt8(i) => write!(f, "{}u8", i),
            Value::UInt16(i) => write!(f, "{}u16", i),
            Value::UInt32(i) => write!(f, "{}u32", i),
            Value::UInt64(i) => write!(f, "{}u64", i),
            // Floats
            Value::Float32(fl) => write!(f, "{}f32", fl),
            Value::Float64(fl) => write!(f, "{}", fl),
            // BigInt
            Value::BigInt(n) => write!(f, "{}n", n),
            // Decimal
            Value::Decimal(d) => write!(f, "{}d", d),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", item)?;
                }
                write!(f, "]")
            }
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", item)?;
                }
                write!(f, ")")
            }
            Value::Record(r) => {
                write!(f, "{}{{", r.type_name)?;
                for (i, (name, val)) in r.field_names.iter().zip(r.fields.iter()).enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {:?}", name, val)?;
                }
                write!(f, "}}")
            }
            Value::ReactiveRecord(r) => {
                write!(f, "reactive {}{{", r.type_name)?;
                if let Ok(fields) = r.fields.read() {
                    for (i, (name, val)) in r.field_names.iter().zip(fields.iter()).enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}: {:?}", name, val)?;
                    }
                }
                write!(f, "}}")
            }
            Value::Variant(v) => {
                write!(f, "{}", v.constructor)?;
                if !v.fields.is_empty() {
                    write!(f, "(")?;
                    for (i, field) in v.fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{:?}", field)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Function(func) => write!(f, "<function {}>", func.name),
            Value::Closure(c) => write!(f, "<closure {}>", c.function.name),
            Value::NativeFunction(n) => write!(f, "<native {}>", n.name),
            Value::Pid(p) => write!(f, "<pid {}>", p.0),
            Value::Ref(r) => write!(f, "<ref {}>", r.0),
            Value::Map(m) => write!(f, "%{{...{} entries}}", m.len()),
            Value::Set(s) => write!(f, "#{{...{} items}}", s.len()),
            Value::Array(a) => write!(f, "Array[{}]", a.read().unwrap().len()),
            Value::Int64Array(a) => write!(f, "Int64Array[{}]", a.read().unwrap().len()),
            Value::Float64Array(a) => write!(f, "Float64Array[{}]", a.read().unwrap().len()),
            Value::Float32Array(a) => write!(f, "Float32Array[{}]", a.read().unwrap().len()),
            Value::Type(t) => write!(f, "<type {}>", t.name),
            Value::Pointer(p) => write!(f, "<ptr 0x{:x}>", p),
            Value::NativeHandle(h) => write!(f, "<native ptr=0x{:x} type={}>", h.ptr, h.type_id),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Char(c) => write!(f, "{}", c),
            // Integers (no suffix in display for Int64, it's the default)
            Value::Int8(i) => write!(f, "{}", i),
            Value::Int16(i) => write!(f, "{}", i),
            Value::Int32(i) => write!(f, "{}", i),
            Value::Int64(i) => write!(f, "{}", i),
            Value::UInt8(i) => write!(f, "{}", i),
            Value::UInt16(i) => write!(f, "{}", i),
            Value::UInt32(i) => write!(f, "{}", i),
            Value::UInt64(i) => write!(f, "{}", i),
            // Floats
            Value::Float32(fl) => write!(f, "{}", fl),
            Value::Float64(fl) => write!(f, "{}", fl),
            // BigInt
            Value::BigInt(n) => write!(f, "{}", n),
            // Decimal
            Value::Decimal(d) => write!(f, "{}", d),
            Value::String(s) => write!(f, "{}", s),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Variant(v) if v.fields.is_empty() => write!(f, "{}", v.constructor),
            Value::Variant(v) => {
                write!(f, "{}(", v.constructor)?;
                for (i, field) in v.fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", field)?;
                }
                write!(f, ")")
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            // Signed integers
            (Value::Int8(a), Value::Int8(b)) => a == b,
            (Value::Int16(a), Value::Int16(b)) => a == b,
            (Value::Int32(a), Value::Int32(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            // Unsigned integers
            (Value::UInt8(a), Value::UInt8(b)) => a == b,
            (Value::UInt16(a), Value::UInt16(b)) => a == b,
            (Value::UInt32(a), Value::UInt32(b)) => a == b,
            (Value::UInt64(a), Value::UInt64(b)) => a == b,
            // Floats
            (Value::Float32(a), Value::Float32(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => a == b,
            // BigInt
            (Value::BigInt(a), Value::BigInt(b)) => a == b,
            // Decimal
            (Value::Decimal(a), Value::Decimal(b)) => a == b,
            // Collections
            (Value::String(a), Value::String(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Variant(a), Value::Variant(b)) => {
                a.type_name == b.type_name
                    && a.constructor == b.constructor
                    && a.fields == b.fields
            }
            (Value::Record(a), Value::Record(b)) => {
                a.type_name == b.type_name && a.fields == b.fields
            }
            (Value::Pid(a), Value::Pid(b)) => a == b,
            (Value::Ref(a), Value::Ref(b)) => a == b,
            _ => false,
        }
    }
}

impl Clone for NativeFn {
    fn clone(&self) -> Self {
        panic!("Cannot clone NativeFn - use Rc<NativeFn> instead")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type_name() {
        assert_eq!(Value::Int64(42).type_name(), "Int64");
        assert_eq!(Value::Int32(42).type_name(), "Int32");
        assert_eq!(Value::UInt8(42).type_name(), "UInt8");
        assert_eq!(Value::Float64(3.14).type_name(), "Float64");
        assert_eq!(Value::Bool(true).type_name(), "Bool");
        assert_eq!(Value::Unit.type_name(), "()");
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value::Int64(42)), "42");
        assert_eq!(format!("{}", Value::Bool(true)), "true");
        assert_eq!(format!("{}", Value::String(Arc::new("hello".to_string()))), "hello");
    }

    #[test]
    fn test_chunk_constants() {
        let mut chunk = Chunk::new();
        let idx = chunk.add_constant(Value::Int64(42));
        assert_eq!(idx, 0);
        assert_eq!(chunk.constants.len(), 1);
    }

    #[test]
    fn test_chunk_emit() {
        let mut chunk = Chunk::new();
        chunk.emit(Instruction::LoadConst(0, 0), 1);
        chunk.emit(Instruction::Return(0), 1);
        assert_eq!(chunk.code.len(), 2);
    }
}
