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
use std::rc::Rc;
use std::cell::{Cell, RefCell};

/// Shared register list - makes instruction cloning O(1) instead of O(n).
/// Used for function arguments, list/tuple construction, etc.
pub type RegList = Rc<[Reg]>;

/// A runtime value in Nostos.
#[derive(Clone)]
pub enum Value {
    // === Immediate values (unboxed in JIT) ===
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),

    // === Heap-allocated values ===
    String(Rc<String>),
    List(Rc<Vec<Value>>),
    Array(Rc<RefCell<Vec<Value>>>), // mutable
    Tuple(Rc<Vec<Value>>),
    Map(Rc<HashMap<MapKey, Value>>),
    Set(Rc<std::collections::HashSet<MapKey>>),

    // === Structured values ===
    Record(Rc<RecordValue>),
    Variant(Rc<VariantValue>),

    // === Callable values ===
    Function(Rc<FunctionValue>),
    Closure(Rc<ClosureValue>),
    NativeFunction(Rc<NativeFn>),

    // === Concurrency values ===
    Pid(Pid),
    Ref(RefId),

    // === Special ===
    /// Type value for introspection
    Type(Rc<TypeValue>),
    /// Opaque pointer for FFI
    Pointer(usize),
}

/// Key type for maps and sets (must be hashable).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum MapKey {
    Unit,
    Bool(bool),
    Int(i64),
    Char(char),
    String(Rc<String>),
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

/// A variant value (tagged union).
#[derive(Clone)]
pub struct VariantValue {
    /// Type name (e.g., "Option")
    pub type_name: String,
    /// Constructor name (e.g., "Some")
    pub constructor: String,
    /// Payload (positional fields)
    pub fields: Vec<Value>,
    /// Named fields (if any)
    pub named_fields: Option<HashMap<String, Value>>,
}

/// A compiled function.
#[derive(Clone)]
pub struct FunctionValue {
    /// Function name
    pub name: String,
    /// Number of parameters
    pub arity: usize,
    /// Parameter names (for introspection)
    pub param_names: Vec<String>,
    /// Bytecode instructions
    pub code: Rc<Chunk>,
    /// Module this function belongs to
    pub module: Option<String>,
    /// Source location for debugging
    pub source_span: Option<(usize, usize)>,
    /// JIT-compiled version (if available)
    pub jit_code: Option<JitFunction>,
    /// Call counter for JIT hot detection (interior mutability for Rc<FunctionValue>)
    pub call_count: Cell<u32>,
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
    pub function: Rc<FunctionValue>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    pub traits: Vec<String>,
}

#[derive(Clone)]
pub enum TypeKind {
    Primitive,
    Record { mutable: bool },
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
}

impl RuntimeError {
    /// Convert this runtime error to an exception Value that can be caught by try/catch.
    /// Returns a record with `type` and `message` fields.
    pub fn to_exception_value(&self) -> Value {
        use std::rc::Rc;

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
        };

        // Create a record-like structure: Error{type: "...", message: "..."}
        let record = RecordValue {
            type_name: "Error".to_string(),
            field_names: vec!["type".to_string(), "message".to_string()],
            fields: vec![
                Value::String(Rc::new(error_type.to_string())),
                Value::String(Rc::new(message)),
            ],
            mutable_fields: vec![false, false],
        };

        Value::Record(Rc::new(record))
    }
}

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
    MakeMap(Reg, Rc<[(Reg, Reg)]>),
    /// Create set: dst = #{regs...}
    MakeSet(Reg, RegList),
    /// List cons: dst = [head | tail]
    Cons(Reg, Reg, Reg),
    /// List concat: dst = a ++ b
    ListConcat(Reg, Reg, Reg),
    /// Index access: dst = collection[index]
    Index(Reg, Reg, Reg),
    /// Index assignment: collection[index] = value
    IndexSet(Reg, Reg, Reg),
    /// List length: dst = len(list)
    Length(Reg, Reg),

    // === Tuples ===
    /// Get tuple element: dst = tuple[idx]
    GetTupleField(Reg, Reg, u8),

    // === Records ===
    /// Create record: dst = TypeName{fields...}
    MakeRecord(Reg, ConstIdx, RegList),
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
    /// Call native function
    CallNative(Reg, ConstIdx, RegList),
    /// Return value from function
    Return(Reg),

    // === Closures ===
    /// Create closure: dst = closure(func_idx, captures...)
    MakeClosure(Reg, ConstIdx, RegList),
    /// Get captured variable: dst = captures[idx]
    GetCapture(Reg, u8),

    // === Pattern matching ===
    /// Test if value matches constructor: dst = value.tag == ctor_idx
    TestTag(Reg, Reg, ConstIdx),
    /// Test if value is unit: dst = value == ()
    TestUnit(Reg, Reg),
    /// Test if value equals constant: dst = value == const
    TestConst(Reg, Reg, ConstIdx),
    /// Test if list is empty: dst = list == []
    TestNil(Reg, Reg),
    /// Deconstruct list: head, tail = list (fails if empty)
    Decons(Reg, Reg, Reg),

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
    /// Receive (handled specially by VM - switches to receive mode)
    Receive,
    /// Set receive timeout
    ReceiveTimeout(Reg),

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

    // === Type conversions (compile-time resolved) ===
    /// Int to Float: dst = toFloat(src)
    IntToFloat(Reg, Reg),
    /// Float to Int: dst = toInt(src)
    FloatToInt(Reg, Reg),

    // === List operations (compile-time resolved) ===
    /// Head of list: dst = head(list)
    ListHead(Reg, Reg),
    /// Tail of list: dst = tail(list)
    ListTail(Reg, Reg),
    /// Is list empty: dst = isEmpty(list)
    ListIsEmpty(Reg, Reg),

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

    // === Debug ===
    /// No operation (for alignment/debugging)
    Nop,
    /// Debug print
    DebugPrint(Reg),
}

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
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Char(_) => "Char",
            Value::String(_) => "String",
            Value::List(_) => "List",
            Value::Array(_) => "Array",
            Value::Tuple(_) => "Tuple",
            Value::Map(_) => "Map",
            Value::Set(_) => "Set",
            Value::Record(r) => &r.type_name,
            Value::Variant(v) => &v.type_name,
            Value::Function(_) => "Function",
            Value::Closure(_) => "Closure",
            Value::NativeFunction(_) => "NativeFunction",
            Value::Pid(_) => "Pid",
            Value::Ref(_) => "Ref",
            Value::Type(_) => "Type",
            Value::Pointer(_) => "Pointer",
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
            Value::Int(i) => Some(MapKey::Int(*i)),
            Value::Char(c) => Some(MapKey::Char(*c)),
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
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Char(c) => write!(f, "'{}'", c),
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
            Value::Array(a) => write!(f, "Array[{}]", a.borrow().len()),
            Value::Type(t) => write!(f, "<type {}>", t.name),
            Value::Pointer(p) => write!(f, "<ptr 0x{:x}>", p),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Char(c) => write!(f, "{}", c),
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
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
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
        assert_eq!(Value::Int(42).type_name(), "Int");
        assert_eq!(Value::Bool(true).type_name(), "Bool");
        assert_eq!(Value::Unit.type_name(), "()");
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value::Int(42)), "42");
        assert_eq!(format!("{}", Value::Bool(true)), "true");
        assert_eq!(format!("{}", Value::String(Rc::new("hello".to_string()))), "hello");
    }

    #[test]
    fn test_chunk_constants() {
        let mut chunk = Chunk::new();
        let idx = chunk.add_constant(Value::Int(42));
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
