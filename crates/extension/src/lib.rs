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

/// A native handle wrapping any Rust type
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
    /// Opaque native handle (for holding Rust objects)
    Native(NativeHandle),
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
                .ok_or_else(|| format!("Native handle type mismatch")),
            _ => Err(format!("Expected Native, got {:?}", self.type_name())),
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

/// Registry for extension functions
pub struct ExtRegistry {
    functions: Vec<(String, ExtFn)>,
}

impl ExtRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        ExtRegistry {
            functions: Vec::new(),
        }
    }

    /// Register a function
    pub fn add(&mut self, name: &str, f: ExtFn) {
        self.functions.push((name.to_string(), f));
    }

    /// Get all registered functions
    pub fn functions(&self) -> &[(String, ExtFn)] {
        &self.functions
    }
}

impl Default for ExtRegistry {
    fn default() -> Self {
        Self::new()
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
            funcs.insert(name.clone(), *func);
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
}
