//! Shared types for cross-thread communication and shared data structures.
//!
//! These types are designed to be thread-safe and can be shared between
//! the GC heap (GcValue) and cross-thread messaging (ThreadSafeValue).

use std::sync::Arc;
use std::hash::{Hash, Hasher};
use imbl::HashMap as ImblHashMap;

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
