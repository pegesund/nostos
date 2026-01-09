//! Shared types for cross-thread communication and shared data structures.
//!
//! These types are designed to be thread-safe and can be shared between
//! the GC heap (GcValue) and cross-thread messaging (ThreadSafeValue).

use std::sync::Arc;
use imbl::HashMap as ImblHashMap;

/// Thread-safe map key for shared maps and cross-thread communication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
