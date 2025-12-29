//! Extension loading and management for native Nostos extensions.
//!
//! Extensions are dynamic libraries (.so/.dylib/.dll) that export a
//! `NOSTOS_EXTENSION` symbol containing registration information.

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

use libloading::{Library, Symbol};
use parking_lot::RwLock;
use tokio::runtime::Handle as TokioHandle;
use tokio::sync::mpsc;

pub use nostos_extension::{
    ExtContext, ExtFn, ExtMessage, ExtRegistry, ExtensionDecl, NativeHandle, Pid, Value,
};

/// Manages loaded extensions and provides function dispatch.
pub struct ExtensionManager {
    /// Loaded libraries (kept to prevent unloading)
    libraries: RwLock<Vec<Library>>,
    /// Registered functions
    functions: RwLock<HashMap<String, ExtFn>>,
    /// Channel sender for messages back to scheduler
    message_tx: mpsc::UnboundedSender<ExtMessage>,
    /// Channel receiver (handed off to scheduler)
    message_rx: RwLock<Option<mpsc::UnboundedReceiver<ExtMessage>>>,
    /// Tokio runtime handle
    tokio_handle: TokioHandle,
}

impl ExtensionManager {
    /// Create a new extension manager.
    /// Call `take_message_receiver()` to get the receiver for the scheduler.
    pub fn new(tokio_handle: TokioHandle) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        ExtensionManager {
            libraries: RwLock::new(Vec::new()),
            functions: RwLock::new(HashMap::new()),
            message_tx: tx,
            message_rx: RwLock::new(Some(rx)),
            tokio_handle,
        }
    }

    /// Take the message receiver. Should be called once by the scheduler.
    pub fn take_message_receiver(&self) -> Option<mpsc::UnboundedReceiver<ExtMessage>> {
        self.message_rx.write().take()
    }

    /// Load an extension from a dynamic library.
    ///
    /// # Safety
    ///
    /// The library must be a valid Nostos extension compiled with the same
    /// version of `nostos-extension`.
    pub fn load(&self, path: &Path) -> Result<String, String> {
        unsafe {
            let lib = Library::new(path)
                .map_err(|e| format!("Failed to load library {}: {}", path.display(), e))?;

            let decl: Symbol<*const ExtensionDecl> = lib
                .get(b"NOSTOS_EXTENSION")
                .map_err(|e| format!("Failed to find NOSTOS_EXTENSION symbol: {}", e))?;

            let decl = &**decl;

            // Get extension name
            let name = CStr::from_ptr(decl.name)
                .to_str()
                .map_err(|e| format!("Invalid extension name: {}", e))?
                .to_string();

            let version = CStr::from_ptr(decl.version)
                .to_str()
                .map_err(|e| format!("Invalid extension version: {}", e))?;

            // Register functions
            let mut registry = ExtRegistry::new();
            (decl.register)(&mut registry);

            let mut funcs = self.functions.write();
            for (func_name, func) in registry.functions() {
                funcs.insert(func_name.clone(), *func);
            }

            // Keep library loaded
            self.libraries.write().push(lib);

            Ok(format!("Loaded extension {} v{}", name, version))
        }
    }

    /// Register functions directly (for testing without dynamic loading).
    pub fn register(&self, registry: &ExtRegistry) {
        let mut funcs = self.functions.write();
        for (name, func) in registry.functions() {
            funcs.insert(name.clone(), *func);
        }
    }

    /// Call an extension function.
    pub fn call(
        &self,
        name: &str,
        args: &[Value],
        caller_pid: Pid,
    ) -> Result<Value, String> {
        let funcs = self.functions.read();
        let func = funcs
            .get(name)
            .ok_or_else(|| format!("Unknown extension function: {}", name))?;

        let ctx = ExtContext::new(
            self.tokio_handle.clone(),
            self.message_tx.clone(),
            caller_pid,
        );

        func(args, &ctx)
    }

    /// Check if a function exists.
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.read().contains_key(name)
    }

    /// Get all registered function names.
    pub fn function_names(&self) -> Vec<String> {
        self.functions.read().keys().cloned().collect()
    }

    /// Get number of loaded extensions.
    pub fn extension_count(&self) -> usize {
        self.libraries.read().len()
    }
}

/// Convert extension Value to VM Value.
pub fn ext_value_to_vm(v: &Value) -> crate::Value {
    match v {
        Value::Unit => crate::Value::Unit,
        Value::Bool(b) => crate::Value::Bool(*b),
        Value::Int(i) => crate::Value::Int64(*i),
        Value::Float(f) => crate::Value::Float64(*f),
        Value::String(s) => crate::Value::String(Arc::clone(s)),
        Value::Bytes(b) => {
            // Convert to List of UInt8
            let bytes: Vec<crate::Value> = b.iter().map(|byte| crate::Value::UInt8(*byte)).collect();
            crate::Value::List(Arc::new(bytes))
        }
        Value::List(l) => {
            let items: Vec<crate::Value> = l.iter().map(ext_value_to_vm).collect();
            crate::Value::List(Arc::new(items))
        }
        Value::Tuple(t) => {
            let items: Vec<crate::Value> = t.iter().map(ext_value_to_vm).collect();
            crate::Value::Tuple(Arc::new(items))
        }
        Value::Map(m) => {
            let map: std::collections::HashMap<crate::MapKey, crate::Value> = m
                .iter()
                .map(|(k, v)| (crate::MapKey::String(Arc::new(k.clone())), ext_value_to_vm(v)))
                .collect();
            crate::Value::Map(Arc::new(map))
        }
        Value::Record { name, fields } => {
            let field_names: Vec<String> = fields.iter().map(|(k, _)| k.clone()).collect();
            let field_values: Vec<crate::Value> = fields.iter().map(|(_, v)| ext_value_to_vm(v)).collect();
            let mutable_fields: Vec<bool> = vec![false; fields.len()];
            crate::Value::Record(Arc::new(crate::RecordValue {
                type_name: name.clone(),
                field_names,
                fields: field_values,
                mutable_fields,
            }))
        }
        Value::Pid(p) => {
            crate::Value::Pid(crate::Pid(p.0))
        }
        Value::Native(_h) => {
            // For now, we use Pointer to store native handles
            // This needs improvement - perhaps store in a handle map
            crate::Value::Unit
        }
        Value::None => crate::Value::Unit, // VM doesn't have None, use Unit
    }
}

/// Convert VM Value to extension Value.
pub fn vm_value_to_ext(v: &crate::Value) -> Value {
    match v {
        crate::Value::Unit => Value::Unit,
        crate::Value::Bool(b) => Value::Bool(*b),
        crate::Value::Int8(i) => Value::Int(*i as i64),
        crate::Value::Int16(i) => Value::Int(*i as i64),
        crate::Value::Int32(i) => Value::Int(*i as i64),
        crate::Value::Int64(i) => Value::Int(*i),
        crate::Value::UInt8(i) => Value::Int(*i as i64),
        crate::Value::UInt16(i) => Value::Int(*i as i64),
        crate::Value::UInt32(i) => Value::Int(*i as i64),
        crate::Value::UInt64(i) => Value::Int(*i as i64),
        crate::Value::Float32(f) => Value::Float(*f as f64),
        crate::Value::Float64(f) => Value::Float(*f),
        crate::Value::String(s) => Value::String(Arc::clone(s)),
        crate::Value::List(l) => {
            let items: Vec<Value> = l.iter().map(vm_value_to_ext).collect();
            Value::List(Arc::new(items))
        }
        crate::Value::Tuple(t) => {
            let items: Vec<Value> = t.iter().map(vm_value_to_ext).collect();
            Value::Tuple(Arc::new(items))
        }
        crate::Value::Pid(p) => Value::Pid(Pid(p.0)),
        // For other types, return Unit for now
        _ => Value::Unit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    fn test_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
        let a = args[0].as_f64()?;
        let b = args[1].as_f64()?;
        Ok(Value::Float(a + b))
    }

    fn test_multiply(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
        let a = args[0].as_f64()?;
        let b = args[1].as_f64()?;
        Ok(Value::Float(a * b))
    }

    #[test]
    fn test_register_and_call() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let manager = ExtensionManager::new(rt.handle().clone());

        let mut registry = ExtRegistry::new();
        registry.add("Test.add", test_add);
        registry.add("Test.multiply", test_multiply);
        manager.register(&registry);

        assert!(manager.has_function("Test.add"));
        assert!(manager.has_function("Test.multiply"));
        assert!(!manager.has_function("Test.unknown"));

        let result = manager
            .call("Test.add", &[Value::Float(2.0), Value::Float(3.0)], Pid(1))
            .unwrap();
        assert_eq!(result.as_f64().unwrap(), 5.0);

        let result = manager
            .call("Test.multiply", &[Value::Float(4.0), Value::Float(5.0)], Pid(1))
            .unwrap();
        assert_eq!(result.as_f64().unwrap(), 20.0);
    }

    #[test]
    fn test_spawn_compute() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let manager = ExtensionManager::new(rt.handle().clone());
        let mut rx = manager.take_message_receiver().unwrap();

        static COUNTER: AtomicI64 = AtomicI64::new(0);

        fn compute_fn(args: &[Value], ctx: &ExtContext) -> Result<Value, String> {
            let n = args[0].as_i64()?;
            let caller = ctx.caller();
            let ctx_clone = ctx.clone();

            ctx.spawn_compute(move || {
                // Simulate heavy computation
                let result = (1..=n).sum::<i64>();
                COUNTER.store(result, Ordering::SeqCst);
                ctx_clone.send(caller, Value::Int(result));
            });

            Ok(Value::Unit)
        }

        let mut registry = ExtRegistry::new();
        registry.add("Test.compute", compute_fn);
        manager.register(&registry);

        manager
            .call("Test.compute", &[Value::Int(100)], Pid(42))
            .unwrap();

        // Wait for rayon to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Check the message was sent
        let msg = rx.try_recv().unwrap();
        assert_eq!(msg.target.0, 42);
        assert_eq!(msg.value.as_i64().unwrap(), 5050);
    }

    #[test]
    fn test_load_glam_extension() {
        // This test requires the glam extension to be built first
        // Run: cd extensions/glam && cargo build --release
        let glam_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("extensions/glam/target/release/libnostos_glam.so");

        if !glam_path.exists() {
            println!("Skipping test: glam extension not built at {:?}", glam_path);
            return;
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        let manager = ExtensionManager::new(rt.handle().clone());

        // Load the extension
        let result = manager.load(&glam_path);
        assert!(result.is_ok(), "Failed to load glam: {:?}", result);
        println!("Loaded: {}", result.unwrap());

        // Check functions are registered
        assert!(manager.has_function("Glam.vec3"));
        assert!(manager.has_function("Glam.vec3Add"));
        assert!(manager.has_function("Glam.vec3Dot"));
        assert!(manager.has_function("Glam.mat4Identity"));

        // Create a vec3
        let v1 = manager.call(
            "Glam.vec3",
            &[Value::Float(1.0), Value::Float(2.0), Value::Float(3.0)],
            Pid(1)
        ).unwrap();
        println!("v1 = {:?}", v1);

        let v2 = manager.call(
            "Glam.vec3",
            &[Value::Float(4.0), Value::Float(5.0), Value::Float(6.0)],
            Pid(1)
        ).unwrap();

        // Add them
        let sum = manager.call("Glam.vec3Add", &[v1.clone(), v2.clone()], Pid(1)).unwrap();
        println!("v1 + v2 = {:?}", sum);

        // Verify result is (5, 7, 9)
        let tuple = sum.as_tuple().unwrap();
        assert_eq!(tuple[0].as_f64().unwrap(), 5.0);
        assert_eq!(tuple[1].as_f64().unwrap(), 7.0);
        assert_eq!(tuple[2].as_f64().unwrap(), 9.0);

        // Dot product
        let dot = manager.call("Glam.vec3Dot", &[v1, v2], Pid(1)).unwrap();
        assert_eq!(dot.as_f64().unwrap(), 32.0); // 1*4 + 2*5 + 3*6 = 32

        // Test matrix operations
        let identity = manager.call("Glam.mat4Identity", &[], Pid(1)).unwrap();
        println!("identity = {:?}", identity);
        let identity_list = identity.as_list().unwrap();
        assert_eq!(identity_list.len(), 16);
        // First column should be (1, 0, 0, 0)
        assert_eq!(identity_list[0].as_f64().unwrap(), 1.0);
        assert_eq!(identity_list[1].as_f64().unwrap(), 0.0);

        println!("All glam extension tests passed!");
    }
}
