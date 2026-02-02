//! Nostos Extension API
//!
//! This crate provides the API for building native Nostos extensions.
//! Extensions are compiled as dynamic libraries (.so/.dylib/.dll) and
//! loaded at runtime.
//!
//! # Example
//!
//! ```rust,ignore
//! use nostos_extension::*;
//!
//! declare_extension!("myext", "0.1.0", register);
//!
//! fn register(reg: &mut ExtRegistry) {
//!     reg.add("MyExt.add", my_add);
//! }
//!
//! fn my_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
//!     let a = args[0].as_f64()?;
//!     let b = args[1].as_f64()?;
//!     Ok(Value::Float(a + b))
//! }
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_char;
use std::future::Future;
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::runtime::Handle as TokioHandle;
use tokio::sync::mpsc;

/// Process ID for message passing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Pid(pub u64);

/// A native handle wrapping any Rust type (Arc-based, cleanup on last drop)
pub struct NativeHandle {
    inner: Arc<dyn Any + Send + Sync>,
    type_name: &'static str,
}

impl NativeHandle {
    pub fn new<T: Any + Send + Sync>(value: T) -> Self {
        NativeHandle {
            inner: Arc::new(value),
            type_name: std::any::type_name::<T>(),
        }
    }

    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }

    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}

impl Clone for NativeHandle {
    fn clone(&self) -> Self {
        NativeHandle {
            inner: Arc::clone(&self.inner),
            type_name: self.type_name,
        }
    }
}

impl std::fmt::Debug for NativeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NativeHandle<{}>", self.type_name)
    }
}

/// Cleanup function type for GC-managed native handles.
/// Called with (ptr, type_id) when the handle is garbage collected.
pub type NativeCleanupFn = fn(ptr: usize, type_id: u64);

/// A GC-managed native handle that stores a raw pointer and type ID.
/// When the GC collects this handle, it calls the cleanup function.
///
/// This is more efficient than NativeHandle for FFI because:
/// - No Arc overhead for reference counting
/// - Extension controls exact cleanup logic
/// - Type ID allows one cleanup function to handle multiple types
///
/// **Important**: Clone creates a non-owning copy (ptr set to 0) that won't
/// trigger cleanup. Only the original handle owns the memory.
pub struct GcNativeHandle {
    /// Raw pointer to native data (as usize for FFI safety).
    /// Set to 0 for non-owning clones.
    pub ptr: usize,
    /// Extension-defined type identifier (e.g., 1=DVector, 2=DMatrix)
    pub type_id: u64,
    /// Cleanup function called by GC when this handle is collected
    pub cleanup: NativeCleanupFn,
}

impl GcNativeHandle {
    /// Create a new GC-managed native handle.
    ///
    /// # Arguments
    /// * `ptr` - Raw pointer to the native data
    /// * `type_id` - Extension-defined type identifier
    /// * `cleanup` - Function to call when GC collects this handle
    ///
    /// # Safety
    /// The cleanup function must correctly free the memory pointed to by ptr
    /// based on the type_id.
    pub fn new(ptr: usize, type_id: u64, cleanup: NativeCleanupFn) -> Self {
        GcNativeHandle { ptr, type_id, cleanup }
    }

    /// Create a handle from a boxed value, returning the handle.
    /// The cleanup function will receive the pointer and type_id.
    pub fn from_boxed<T>(value: Box<T>, type_id: u64, cleanup: NativeCleanupFn) -> Self {
        let ptr = Box::into_raw(value) as usize;
        GcNativeHandle { ptr, type_id, cleanup }
    }

    /// Check if this handle owns the native memory.
    pub fn is_owner(&self) -> bool {
        self.ptr != 0
    }
}

/// Clone creates a non-owning copy. The cloned handle has ptr=0 and won't
/// trigger cleanup when dropped. This is safe because:
/// - Native handles should only exist on one heap (the creating process)
/// - When message passing, the extension should be notified to deep-copy
impl Clone for GcNativeHandle {
    fn clone(&self) -> Self {
        // Create a non-owning clone with ptr=0
        // This prevents double-free when the clone is dropped
        GcNativeHandle {
            ptr: 0, // Non-owning
            type_id: self.type_id,
            cleanup: self.cleanup,
        }
    }
}

impl std::fmt::Debug for GcNativeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcNativeHandle(ptr={:#x}, type_id={}, owner={})",
               self.ptr, self.type_id, self.is_owner())
    }
}

impl Drop for GcNativeHandle {
    fn drop(&mut self) {
        // Only call cleanup if this is the owning handle (ptr != 0)
        if self.ptr != 0 {
            (self.cleanup)(self.ptr, self.type_id);
        }
    }
}

/// Value type for extension arguments and return values.
/// This is a simplified representation that maps to Nostos VM values.
#[derive(Debug, Clone)]
pub enum Value {
    /// Unit value
    Unit,
    /// Boolean
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit float
    Float(f64),
    /// String
    String(Arc<String>),
    /// Byte array
    Bytes(Arc<Vec<u8>>),
    /// List of values
    List(Arc<Vec<Value>>),
    /// Tuple of values
    Tuple(Arc<Vec<Value>>),
    /// Map from values to values
    Map(Arc<HashMap<String, Value>>),
    /// Record with named fields
    Record {
        name: String,
        fields: Vec<(String, Value)>,
    },
    /// Process ID
    Pid(Pid),
    /// Opaque native handle (for holding Rust objects, Arc-based cleanup)
    Native(NativeHandle),
    /// GC-managed native handle (pointer + type_id, GC calls cleanup function)
    /// Wrapped in Arc so cloning is safe and cleanup only happens once.
    GcHandle(Arc<GcNativeHandle>),
    /// None/null value
    None,
}

impl Value {
    // Constructors

    pub fn unit() -> Self {
        Value::Unit
    }

    pub fn bool(b: bool) -> Self {
        Value::Bool(b)
    }

    pub fn int(i: i64) -> Self {
        Value::Int(i)
    }

    pub fn float(f: f64) -> Self {
        Value::Float(f)
    }

    pub fn string(s: impl Into<String>) -> Self {
        Value::String(Arc::new(s.into()))
    }

    pub fn bytes(b: Vec<u8>) -> Self {
        Value::Bytes(Arc::new(b))
    }

    pub fn list(items: Vec<Value>) -> Self {
        Value::List(Arc::new(items))
    }

    pub fn tuple(items: Vec<Value>) -> Self {
        Value::Tuple(Arc::new(items))
    }

    pub fn record(name: impl Into<String>, fields: Vec<(&str, Value)>) -> Self {
        Value::Record {
            name: name.into(),
            fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    pub fn native<T: Any + Send + Sync>(value: T) -> Self {
        Value::Native(NativeHandle::new(value))
    }

    /// Create a GC-managed native handle from a boxed value.
    /// The cleanup function will be called when the GC collects this value.
    pub fn gc_handle<T>(value: Box<T>, type_id: u64, cleanup: NativeCleanupFn) -> Self {
        Value::GcHandle(Arc::new(GcNativeHandle::from_boxed(value, type_id, cleanup)))
    }

    /// Create a GC-managed native handle from raw pointer.
    pub fn gc_handle_raw(ptr: usize, type_id: u64, cleanup: NativeCleanupFn) -> Self {
        Value::GcHandle(Arc::new(GcNativeHandle::new(ptr, type_id, cleanup)))
    }

    pub fn none() -> Self {
        Value::None
    }

    pub fn float_list(values: Vec<f64>) -> Self {
        Value::List(Arc::new(values.into_iter().map(Value::Float).collect()))
    }

    pub fn int_list(values: Vec<i64>) -> Self {
        Value::List(Arc::new(values.into_iter().map(Value::Int).collect()))
    }

    // Extractors

    pub fn as_bool(&self) -> Result<bool, String> {
        match self {
            Value::Bool(b) => Ok(*b),
            _ => Err(format!("Expected Bool, got {:?}", self.type_name())),
        }
    }

    pub fn as_i64(&self) -> Result<i64, String> {
        match self {
            Value::Int(i) => Ok(*i),
            _ => Err(format!("Expected Int, got {:?}", self.type_name())),
        }
    }

    pub fn as_f64(&self) -> Result<f64, String> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(format!("Expected Float, got {:?}", self.type_name())),
        }
    }

    pub fn as_f32(&self) -> Result<f32, String> {
        self.as_f64().map(|f| f as f32)
    }

    pub fn as_string(&self) -> Result<&str, String> {
        match self {
            Value::String(s) => Ok(s.as_str()),
            _ => Err(format!("Expected String, got {:?}", self.type_name())),
        }
    }

    pub fn as_bytes(&self) -> Result<&[u8], String> {
        match self {
            Value::Bytes(b) => Ok(b.as_slice()),
            _ => Err(format!("Expected Bytes, got {:?}", self.type_name())),
        }
    }

    pub fn as_list(&self) -> Result<&[Value], String> {
        match self {
            Value::List(l) => Ok(l.as_slice()),
            _ => Err(format!("Expected List, got {:?}", self.type_name())),
        }
    }

    pub fn as_tuple(&self) -> Result<&[Value], String> {
        match self {
            Value::Tuple(t) => Ok(t.as_slice()),
            _ => Err(format!("Expected Tuple, got {:?}", self.type_name())),
        }
    }

    pub fn as_pid(&self) -> Result<Pid, String> {
        match self {
            Value::Pid(p) => Ok(*p),
            _ => Err(format!("Expected Pid, got {:?}", self.type_name())),
        }
    }

    pub fn as_native<T: Any + Send + Sync>(&self) -> Result<&T, String> {
        match self {
            Value::Native(h) => h
                .downcast_ref::<T>()
                .ok_or_else(|| "Native handle type mismatch".to_string()),
            _ => Err(format!("Expected Native, got {:?}", self.type_name())),
        }
    }

    /// Get the GC-managed native handle's pointer and type_id.
    pub fn as_gc_handle(&self) -> Result<&GcNativeHandle, String> {
        match self {
            Value::GcHandle(h) => Ok(h.as_ref()),
            _ => Err(format!("Expected GcHandle, got {:?}", self.type_name())),
        }
    }

    /// Get the raw pointer from a GC handle, casting to the expected type.
    /// # Safety
    /// Caller must ensure type_id matches the expected type.
    pub unsafe fn as_gc_handle_ptr<T>(&self) -> Result<&T, String> {
        match self {
            Value::GcHandle(h) => Ok(&*(h.ptr as *const T)),
            _ => Err(format!("Expected GcHandle, got {:?}", self.type_name())),
        }
    }

    pub fn as_float_list(&self) -> Result<Vec<f64>, String> {
        let list = self.as_list()?;
        list.iter().map(|v| v.as_f64()).collect()
    }

    pub fn as_int_list(&self) -> Result<Vec<i64>, String> {
        let list = self.as_list()?;
        list.iter().map(|v| v.as_i64()).collect()
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "Unit",
            Value::Bool(_) => "Bool",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::String(_) => "String",
            Value::Bytes(_) => "Bytes",
            Value::List(_) => "List",
            Value::Tuple(_) => "Tuple",
            Value::Map(_) => "Map",
            Value::Record { .. } => "Record",
            Value::Pid(_) => "Pid",
            Value::Native(_) => "Native",
            Value::GcHandle(_) => "GcHandle",
            Value::None => "None",
        }
    }
}

/// Message to send to the Nostos scheduler
#[derive(Debug)]
pub struct ExtMessage {
    pub target: Pid,
    pub value: Value,
}

/// Context passed to extension functions.
/// Provides access to async runtimes and message passing.
#[derive(Clone)]
pub struct ExtContext {
    /// Handle to tokio runtime for async I/O
    tokio_handle: TokioHandle,
    /// Channel to send messages back to Nostos actors
    message_tx: mpsc::UnboundedSender<ExtMessage>,
    /// Caller's process ID
    caller_pid: Pid,
}

impl ExtContext {
    /// Create a new extension context
    pub fn new(
        tokio_handle: TokioHandle,
        message_tx: mpsc::UnboundedSender<ExtMessage>,
        caller_pid: Pid,
    ) -> Self {
        ExtContext {
            tokio_handle,
            message_tx,
            caller_pid,
        }
    }

    /// Spawn an async task on the tokio runtime.
    /// Use this for tokio-compatible async I/O (Kafka, HTTP, async DBs).
    pub fn spawn_async<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.tokio_handle.spawn(future);
    }

    /// Spawn a blocking task on tokio's blocking thread pool.
    /// Use this for blocking I/O or short blocking operations.
    pub fn spawn_blocking<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.tokio_handle.spawn_blocking(f);
    }

    /// Spawn a CPU-intensive task on the rayon thread pool.
    /// Use this for ML inference, image processing, heavy computation.
    pub fn spawn_compute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        rayon::spawn(f);
    }

    /// Send a message to a Nostos actor
    pub fn send(&self, pid: Pid, msg: Value) {
        let _ = self.message_tx.send(ExtMessage { target: pid, value: msg });
    }

    /// Send a message to the caller
    pub fn reply(&self, msg: Value) {
        self.send(self.caller_pid, msg);
    }

    /// Get the caller's PID
    pub fn caller(&self) -> Pid {
        self.caller_pid
    }

    /// Get the tokio runtime handle (for advanced use)
    pub fn tokio_handle(&self) -> &TokioHandle {
        &self.tokio_handle
    }
}

/// Function signature for extension functions
pub type ExtFn = fn(args: &[Value], ctx: &ExtContext) -> Result<Value, String>;

/// Kind of extension-defined type
#[derive(Debug, Clone)]
pub enum ExtTypeKind {
    /// Opaque native handle (Tokenizer, Tensor, etc.)
    Opaque,
    /// Record with named fields: [(field_name, type_name)]
    Record(Vec<(String, String)>),
}

/// Declaration of an extension-defined type
#[derive(Debug, Clone)]
pub struct ExtTypeDecl {
    /// Type name (e.g., "Tensor", "Tokenizer")
    pub name: String,
    /// Kind of type
    pub kind: ExtTypeKind,
}

/// Declaration of an extension function with signature
#[derive(Debug, Clone)]
pub struct ExtFnDecl {
    /// Function name (e.g., "loadTokenizer")
    pub name: String,
    /// Type signature (e.g., "(String) -> Tokenizer")
    pub signature: Option<String>,
    /// Optional documentation
    pub doc: Option<String>,
}

/// Registry for extension functions
pub struct ExtRegistry {
    /// Extension-defined types
    types: Vec<ExtTypeDecl>,
    /// Functions with their declarations
    functions: Vec<(ExtFnDecl, ExtFn)>,
}

impl ExtRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        ExtRegistry {
            types: Vec::new(),
            functions: Vec::new(),
        }
    }

    // === Type declarations ===

    /// Declare an opaque type (native handle wrapper like Tensor, Tokenizer)
    pub fn add_opaque_type(&mut self, name: &str) {
        self.types.push(ExtTypeDecl {
            name: name.to_string(),
            kind: ExtTypeKind::Opaque,
        });
    }

    /// Declare a record type with named fields
    pub fn add_record_type(&mut self, name: &str, fields: &[(&str, &str)]) {
        self.types.push(ExtTypeDecl {
            name: name.to_string(),
            kind: ExtTypeKind::Record(
                fields.iter().map(|(n, t)| (n.to_string(), t.to_string())).collect()
            ),
        });
    }

    /// Get all declared types
    pub fn types(&self) -> &[ExtTypeDecl] {
        &self.types
    }

    // === Function registration ===

    /// Register a function (backwards compatible - no type info)
    pub fn add(&mut self, name: &str, f: ExtFn) {
        self.functions.push((
            ExtFnDecl {
                name: name.to_string(),
                signature: None,
                doc: None,
            },
            f,
        ));
    }

    /// Register a function with type signature
    /// Signature format: "(ParamType1, ParamType2) -> ReturnType"
    pub fn add_fn(&mut self, name: &str, signature: &str, f: ExtFn) {
        self.functions.push((
            ExtFnDecl {
                name: name.to_string(),
                signature: Some(signature.to_string()),
                doc: None,
            },
            f,
        ));
    }

    /// Register a function with type signature and documentation
    pub fn add_fn_doc(&mut self, name: &str, signature: &str, doc: &str, f: ExtFn) {
        self.functions.push((
            ExtFnDecl {
                name: name.to_string(),
                signature: Some(signature.to_string()),
                doc: Some(doc.to_string()),
            },
            f,
        ));
    }

    /// Get all registered functions with their declarations
    pub fn functions_with_decl(&self) -> &[(ExtFnDecl, ExtFn)] {
        &self.functions
    }

    /// Get all registered functions (name and function only, for backwards compat)
    pub fn functions(&self) -> Vec<(String, ExtFn)> {
        self.functions.iter()
            .map(|(decl, f)| (decl.name.clone(), *f))
            .collect()
    }
}

impl Default for ExtRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Signature Parsing
// ============================================================================

/// A parsed type from a signature
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedType {
    /// Simple type like "Int", "String", "Tensor"
    Simple(String),
    /// Generic type like "List[Int]", "Map[String, Int]"
    Generic(String, Vec<ParsedType>),
}

impl ParsedType {
    /// Get the base name (e.g., "List" for List[Int])
    pub fn base_name(&self) -> &str {
        match self {
            ParsedType::Simple(name) => name,
            ParsedType::Generic(name, _) => name,
        }
    }
}

impl std::fmt::Display for ParsedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParsedType::Simple(name) => write!(f, "{}", name),
            ParsedType::Generic(name, args) => {
                write!(f, "{}[", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// A parsed function signature
#[derive(Debug, Clone)]
pub struct ParsedSignature {
    /// Parameter types
    pub params: Vec<ParsedType>,
    /// Return type
    pub return_type: ParsedType,
}

/// Parse a type signature string like "(String, Int) -> Tokenizer"
/// Returns (param_types, return_type)
pub fn parse_signature(sig: &str) -> Result<ParsedSignature, String> {
    let sig = sig.trim();

    // Find the "->" separator
    let arrow_pos = sig.find("->")
        .ok_or_else(|| format!("Invalid signature, missing '->': {}", sig))?;

    let params_str = sig[..arrow_pos].trim();
    let return_str = sig[arrow_pos + 2..].trim();

    // Parse parameters
    let params = if params_str == "()" {
        vec![]
    } else if params_str.starts_with('(') && params_str.ends_with(')') {
        // Remove parens and split by comma (handling nested brackets)
        let inner = &params_str[1..params_str.len()-1];
        parse_type_list(inner)?
    } else {
        // Single param without parens (e.g., "String -> Int")
        vec![parse_type(params_str)?]
    };

    // Parse return type
    let return_type = parse_type(return_str)?;

    Ok(ParsedSignature { params, return_type })
}

/// Parse a comma-separated list of types, handling nested brackets
fn parse_type_list(s: &str) -> Result<Vec<ParsedType>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }

    let mut types = Vec::new();
    let mut current = String::new();
    let mut bracket_depth = 0;

    for ch in s.chars() {
        match ch {
            '[' => {
                bracket_depth += 1;
                current.push(ch);
            }
            ']' => {
                bracket_depth -= 1;
                current.push(ch);
            }
            ',' if bracket_depth == 0 => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    types.push(parse_type(trimmed)?);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    // Don't forget the last type
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        types.push(parse_type(trimmed)?);
    }

    Ok(types)
}

/// Parse a single type like "Int", "List[Int]", or "Map[String, Int]"
fn parse_type(s: &str) -> Result<ParsedType, String> {
    let s = s.trim();

    if let Some(bracket_pos) = s.find('[') {
        // Generic type
        if !s.ends_with(']') {
            return Err(format!("Invalid generic type, missing ']': {}", s));
        }

        let name = s[..bracket_pos].to_string();
        let args_str = &s[bracket_pos + 1..s.len() - 1];
        let args = parse_type_list(args_str)?;

        if args.is_empty() {
            return Err(format!("Generic type must have at least one type argument: {}", s));
        }

        Ok(ParsedType::Generic(name, args))
    } else {
        // Simple type
        if s.is_empty() {
            return Err("Empty type name".to_string());
        }
        Ok(ParsedType::Simple(s.to_string()))
    }
}

/// Extension declaration structure.
/// Extensions must export a static `NOSTOS_EXTENSION` of this type.
#[repr(C)]
pub struct ExtensionDecl {
    /// Extension name (null-terminated C string)
    pub name: *const c_char,
    /// Extension version (null-terminated C string)
    pub version: *const c_char,
    /// Registration function
    pub register: unsafe extern "C" fn(reg: &mut ExtRegistry),
}

// Safety: ExtensionDecl is only accessed from the main thread during loading
unsafe impl Send for ExtensionDecl {}
unsafe impl Sync for ExtensionDecl {}

/// Macro to declare an extension.
///
/// # Example
///
/// ```rust,ignore
/// use nostos_extension::*;
///
/// declare_extension!("myext", "0.1.0", register);
///
/// fn register(reg: &mut ExtRegistry) {
///     reg.add("MyExt.foo", foo_impl);
/// }
/// ```
#[macro_export]
macro_rules! declare_extension {
    ($name:expr, $version:expr, $register:ident) => {
        #[no_mangle]
        pub static NOSTOS_EXTENSION: $crate::ExtensionDecl = $crate::ExtensionDecl {
            name: concat!($name, "\0").as_ptr() as *const std::ffi::c_char,
            version: concat!($version, "\0").as_ptr() as *const std::ffi::c_char,
            register: {
                unsafe extern "C" fn wrapper(reg: &mut $crate::ExtRegistry) {
                    $register(reg)
                }
                wrapper
            },
        };
    };
}

/// Extension manager for loading and calling extension functions.
/// This is used internally by the Nostos VM.
pub struct ExtensionManager {
    /// Loaded libraries (kept to prevent unloading)
    #[cfg(feature = "loader")]
    libraries: Vec<libloading::Library>,
    /// Registered functions
    functions: RwLock<HashMap<String, ExtFn>>,
}

impl ExtensionManager {
    /// Create a new extension manager
    pub fn new() -> Self {
        ExtensionManager {
            #[cfg(feature = "loader")]
            libraries: Vec::new(),
            functions: RwLock::new(HashMap::new()),
        }
    }

    /// Register functions from a registry (used for static registration in tests)
    pub fn register(&self, registry: &ExtRegistry) {
        let mut funcs = self.functions.write();
        for (name, func) in registry.functions() {
            funcs.insert(name, func);
        }
    }

    /// Call an extension function
    pub fn call(
        &self,
        name: &str,
        args: &[Value],
        ctx: &ExtContext,
    ) -> Result<Value, String> {
        let funcs = self.functions.read();
        let func = funcs
            .get(name)
            .ok_or_else(|| format!("Unknown extension function: {}", name))?;
        func(args, ctx)
    }

    /// Check if a function exists
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.read().contains_key(name)
    }

    /// Get all registered function names
    pub fn function_names(&self) -> Vec<String> {
        self.functions.read().keys().cloned().collect()
    }
}

impl Default for ExtensionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
        let a = args[0].as_f64()?;
        let b = args[1].as_f64()?;
        Ok(Value::Float(a + b))
    }

    #[test]
    fn test_registry() {
        let mut reg = ExtRegistry::new();
        reg.add("Test.add", test_add);
        assert_eq!(reg.functions().len(), 1);
    }

    #[test]
    fn test_value_conversions() {
        let v = Value::int(42);
        assert_eq!(v.as_i64().unwrap(), 42);
        assert_eq!(v.as_f64().unwrap(), 42.0);

        let v = Value::float(3.14);
        assert_eq!(v.as_f64().unwrap(), 3.14);

        let v = Value::string("hello");
        assert_eq!(v.as_string().unwrap(), "hello");

        let v = Value::list(vec![Value::int(1), Value::int(2), Value::int(3)]);
        assert_eq!(v.as_int_list().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_native_handle() {
        #[derive(Debug)]
        struct MyData {
            value: i32,
        }

        let v = Value::native(MyData { value: 42 });
        let data = v.as_native::<MyData>().unwrap();
        assert_eq!(data.value, 42);
    }

    #[test]
    fn test_parse_simple_signature() {
        let sig = parse_signature("(String) -> Tokenizer").unwrap();
        assert_eq!(sig.params.len(), 1);
        assert_eq!(sig.params[0], ParsedType::Simple("String".to_string()));
        assert_eq!(sig.return_type, ParsedType::Simple("Tokenizer".to_string()));
    }

    #[test]
    fn test_parse_multi_param_signature() {
        let sig = parse_signature("(Tensor, Tensor) -> Tensor").unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0], ParsedType::Simple("Tensor".to_string()));
        assert_eq!(sig.params[1], ParsedType::Simple("Tensor".to_string()));
        assert_eq!(sig.return_type, ParsedType::Simple("Tensor".to_string()));
    }

    #[test]
    fn test_parse_generic_signature() {
        let sig = parse_signature("(List[Float], List[Int]) -> Tensor").unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0], ParsedType::Generic("List".to_string(), vec![ParsedType::Simple("Float".to_string())]));
        assert_eq!(sig.params[1], ParsedType::Generic("List".to_string(), vec![ParsedType::Simple("Int".to_string())]));
        assert_eq!(sig.return_type, ParsedType::Simple("Tensor".to_string()));
    }

    #[test]
    fn test_parse_no_params_signature() {
        let sig = parse_signature("() -> Int").unwrap();
        assert_eq!(sig.params.len(), 0);
        assert_eq!(sig.return_type, ParsedType::Simple("Int".to_string()));
    }

    #[test]
    fn test_parse_nested_generic() {
        let sig = parse_signature("(Map[String, List[Int]]) -> Unit").unwrap();
        assert_eq!(sig.params.len(), 1);
        let expected = ParsedType::Generic(
            "Map".to_string(),
            vec![
                ParsedType::Simple("String".to_string()),
                ParsedType::Generic("List".to_string(), vec![ParsedType::Simple("Int".to_string())])
            ]
        );
        assert_eq!(sig.params[0], expected);
    }

    #[test]
    fn test_parse_return_generic() {
        let sig = parse_signature("(Tokenizer, String) -> List[Int]").unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.return_type, ParsedType::Generic("List".to_string(), vec![ParsedType::Simple("Int".to_string())]));
    }
}
