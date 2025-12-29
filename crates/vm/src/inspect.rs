//! Value inspection API for interactive browsing.
//!
//! This module provides a uniform way to inspect values, supporting:
//! - Navigation into nested structures via "slots"
//! - Cycle detection
//! - Truncated previews for large values

use std::collections::HashSet;
use std::sync::Arc;

use crate::value::{Value, MapKey};

/// Maximum preview length for strings and collections
const DEFAULT_MAX_PREVIEW_LEN: usize = 50;
const DEFAULT_MAX_ITEMS_PREVIEW: usize = 5;

/// A slot represents a navigable child of a value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Slot {
    /// Field in a record: `.fieldname`
    Field(String),
    /// Index in a list/array/tuple: `[0]`
    Index(usize),
    /// Key in a map: `["key"]` or `[42]`
    MapKey(MapKey),
    /// Variant payload field: `.0`, `.1`, or `.fieldname`
    VariantField(usize),
    /// Named variant field
    VariantNamedField(String),
}

impl Slot {
    /// Format the slot as a path segment
    pub fn to_path_segment(&self) -> String {
        match self {
            Slot::Field(name) => format!(".{}", name),
            Slot::Index(i) => format!("[{}]", i),
            Slot::MapKey(key) => format!("[{}]", key.display()),
            Slot::VariantField(i) => format!(".{}", i),
            Slot::VariantNamedField(name) => format!(".{}", name),
        }
    }
}

impl MapKey {
    /// Display the map key
    pub fn display(&self) -> String {
        match self {
            MapKey::Unit => "()".to_string(),
            MapKey::Bool(b) => b.to_string(),
            MapKey::Char(c) => format!("'{}'", c),
            MapKey::Int8(i) => i.to_string(),
            MapKey::Int16(i) => i.to_string(),
            MapKey::Int32(i) => i.to_string(),
            MapKey::Int64(i) => i.to_string(),
            MapKey::UInt8(i) => i.to_string(),
            MapKey::UInt16(i) => i.to_string(),
            MapKey::UInt32(i) => i.to_string(),
            MapKey::UInt64(i) => i.to_string(),
            MapKey::String(s) => format!("\"{}\"", s),
            MapKey::Record { type_name, field_names, fields } => {
                let fields_str: Vec<_> = field_names.iter().zip(fields.iter())
                    .map(|(n, v)| format!("{}: {}", n, v.display()))
                    .collect();
                format!("{}{{{}}}", type_name, fields_str.join(", "))
            }
            MapKey::Variant { type_name, constructor, fields } => {
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

/// Information about a slot for display purposes
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// The slot itself
    pub slot: Slot,
    /// Type of the value at this slot
    pub value_type: String,
    /// Short preview of the value
    pub preview: String,
    /// Whether this is a leaf (no children)
    pub is_leaf: bool,
}

/// Result of inspecting a value
#[derive(Debug, Clone)]
pub struct InspectResult {
    /// Type name of the value
    pub type_name: String,
    /// Short preview of the entire value
    pub preview: String,
    /// Available slots to navigate into
    pub slots: Vec<SlotInfo>,
    /// Whether this value is a leaf (no children to navigate into)
    pub is_leaf: bool,
    /// Number of total items (for collections)
    pub total_count: Option<usize>,
}

/// Value identity for cycle detection (uses Arc pointer comparison)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl Value {
    /// Get a unique identity for cycle detection.
    /// Returns None for primitives (which can't form cycles).
    pub fn identity(&self) -> Option<ValueId> {
        match self {
            // Heap-allocated values have identity via Arc pointer
            Value::String(s) => Some(ValueId(Arc::as_ptr(s) as usize)),
            Value::List(l) => Some(ValueId(Arc::as_ptr(l) as usize)),
            Value::Array(a) => Some(ValueId(Arc::as_ptr(a) as usize)),
            Value::Int64Array(a) => Some(ValueId(Arc::as_ptr(a) as usize)),
            Value::Float64Array(a) => Some(ValueId(Arc::as_ptr(a) as usize)),
            Value::Tuple(t) => Some(ValueId(Arc::as_ptr(t) as usize)),
            Value::Map(m) => Some(ValueId(Arc::as_ptr(m) as usize)),
            Value::Set(s) => Some(ValueId(Arc::as_ptr(s) as usize)),
            Value::Record(r) => Some(ValueId(Arc::as_ptr(r) as usize)),
            Value::Variant(v) => Some(ValueId(Arc::as_ptr(v) as usize)),
            Value::Closure(c) => Some(ValueId(Arc::as_ptr(c) as usize)),
            // Primitives don't have heap identity
            _ => None,
        }
    }

    /// Check if this value is a leaf (no navigable children)
    pub fn is_leaf(&self) -> bool {
        match self {
            Value::Unit | Value::Bool(_) | Value::Char(_) |
            Value::Int8(_) | Value::Int16(_) | Value::Int32(_) | Value::Int64(_) |
            Value::UInt8(_) | Value::UInt16(_) | Value::UInt32(_) | Value::UInt64(_) |
            Value::Float32(_) | Value::Float64(_) |
            Value::BigInt(_) | Value::Decimal(_) |
            Value::Function(_) | Value::NativeFunction(_) |
            Value::Pid(_) | Value::Ref(_) | Value::Type(_) | Value::Pointer(_) => true,

            Value::String(s) => s.len() <= DEFAULT_MAX_PREVIEW_LEN,
            Value::List(l) => l.is_empty(),
            Value::Array(a) => a.read().map(|g| g.is_empty()).unwrap_or(true),
            Value::Int64Array(a) => a.read().map(|g| g.is_empty()).unwrap_or(true),
            Value::Float64Array(a) => a.read().map(|g| g.is_empty()).unwrap_or(true),
            Value::Float32Array(a) => a.read().map(|g| g.is_empty()).unwrap_or(true),
            Value::Tuple(t) => t.is_empty(),
            Value::Map(m) => m.is_empty(),
            Value::Set(s) => s.is_empty(),
            Value::Record(r) => r.fields.is_empty(),
            Value::Variant(v) => v.fields.is_empty() && v.named_fields.as_ref().map(|nf| nf.is_empty()).unwrap_or(true),
            Value::Closure(_) => false, // Can inspect captured vars
            Value::NativeHandle(_) => true, // Native handles are leaf values
        }
    }

    /// Get a short preview of the value
    pub fn preview(&self, max_len: usize) -> String {
        match self {
            Value::Unit => "()".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Char(c) => format!("'{}'", c),
            Value::Int8(i) => i.to_string(),
            Value::Int16(i) => i.to_string(),
            Value::Int32(i) => i.to_string(),
            Value::Int64(i) => i.to_string(),
            Value::UInt8(i) => i.to_string(),
            Value::UInt16(i) => i.to_string(),
            Value::UInt32(i) => i.to_string(),
            Value::UInt64(i) => i.to_string(),
            Value::Float32(f) => f.to_string(),
            Value::Float64(f) => f.to_string(),
            Value::BigInt(n) => {
                let s = n.to_string();
                if s.len() > max_len {
                    format!("{}...", &s[..max_len-3])
                } else {
                    s
                }
            }
            Value::Decimal(d) => d.to_string(),
            Value::String(s) => {
                if s.len() <= max_len {
                    format!("\"{}\"", s)
                } else {
                    format!("\"{}...\" ({} chars)", &s[..max_len-10], s.len())
                }
            }
            Value::List(l) => {
                if l.is_empty() {
                    "[]".to_string()
                } else if l.len() <= DEFAULT_MAX_ITEMS_PREVIEW {
                    let items: Vec<String> = l.iter().map(|v| v.preview(20)).collect();
                    format!("[{}]", items.join(", "))
                } else {
                    format!("[...] ({} items)", l.len())
                }
            }
            Value::Array(a) => {
                if let Ok(guard) = a.read() {
                    if guard.is_empty() {
                        "#[]".to_string()
                    } else {
                        format!("#[...] ({} items)", guard.len())
                    }
                } else {
                    "#[<locked>]".to_string()
                }
            }
            Value::Int64Array(a) => {
                if let Ok(guard) = a.read() {
                    format!("Int64Array({} items)", guard.len())
                } else {
                    "Int64Array(<locked>)".to_string()
                }
            }
            Value::Float64Array(a) => {
                if let Ok(guard) = a.read() {
                    format!("Float64Array({} items)", guard.len())
                } else {
                    "Float64Array(<locked>)".to_string()
                }
            }
            Value::Float32Array(a) => {
                if let Ok(guard) = a.read() {
                    format!("Float32Array({} items)", guard.len())
                } else {
                    "Float32Array(<locked>)".to_string()
                }
            }
            Value::Tuple(t) => {
                if t.is_empty() {
                    "()".to_string()
                } else if t.len() <= DEFAULT_MAX_ITEMS_PREVIEW {
                    let items: Vec<String> = t.iter().map(|v| v.preview(20)).collect();
                    format!("({})", items.join(", "))
                } else {
                    format!("(...) ({} items)", t.len())
                }
            }
            Value::Map(m) => {
                if m.is_empty() {
                    "{}".to_string()
                } else {
                    format!("{{...}} ({} entries)", m.len())
                }
            }
            Value::Set(s) => {
                if s.is_empty() {
                    "Set{}".to_string()
                } else {
                    format!("Set{{...}} ({} items)", s.len())
                }
            }
            Value::Record(r) => {
                format!("{}{{...}}", r.type_name)
            }
            Value::Variant(v) => {
                if v.fields.is_empty() {
                    v.constructor.to_string()
                } else {
                    format!("{}(...)", v.constructor)
                }
            }
            Value::Function(f) => format!("<fn {}>", f.name),
            Value::Closure(c) => format!("<closure {}>", c.function.name),
            Value::NativeFunction(n) => format!("<native {}>", n.name),
            Value::Pid(p) => format!("<pid {:?}>", p),
            Value::Ref(r) => format!("<ref {:?}>", r),
            Value::Type(t) => format!("<type {}>", t.name),
            Value::Pointer(p) => format!("<ptr 0x{:x}>", p),
            Value::NativeHandle(h) => format!("<native type={}>", h.type_id),
        }
    }

    /// Get the slots (children) of this value
    pub fn get_slots(&self) -> Vec<SlotInfo> {
        match self {
            // Primitives have no slots
            Value::Unit | Value::Bool(_) | Value::Char(_) |
            Value::Int8(_) | Value::Int16(_) | Value::Int32(_) | Value::Int64(_) |
            Value::UInt8(_) | Value::UInt16(_) | Value::UInt32(_) | Value::UInt64(_) |
            Value::Float32(_) | Value::Float64(_) |
            Value::BigInt(_) | Value::Decimal(_) |
            Value::Function(_) | Value::NativeFunction(_) |
            Value::Pid(_) | Value::Ref(_) | Value::Type(_) | Value::Pointer(_) |
            Value::NativeHandle(_) => vec![],

            // String: only has slots if long (for chunked viewing)
            Value::String(_) => vec![],

            // List
            Value::List(l) => {
                l.iter().enumerate().map(|(i, v)| SlotInfo {
                    slot: Slot::Index(i),
                    value_type: v.type_name().to_string(),
                    preview: v.preview(DEFAULT_MAX_PREVIEW_LEN),
                    is_leaf: v.is_leaf(),
                }).collect()
            }

            // Array
            Value::Array(a) => {
                if let Ok(guard) = a.read() {
                    guard.iter().enumerate().map(|(i, v)| SlotInfo {
                        slot: Slot::Index(i),
                        value_type: v.type_name().to_string(),
                        preview: v.preview(DEFAULT_MAX_PREVIEW_LEN),
                        is_leaf: v.is_leaf(),
                    }).collect()
                } else {
                    vec![]
                }
            }

            // Int64Array
            Value::Int64Array(a) => {
                if let Ok(guard) = a.read() {
                    guard.iter().enumerate().map(|(i, v)| SlotInfo {
                        slot: Slot::Index(i),
                        value_type: "Int64".to_string(),
                        preview: v.to_string(),
                        is_leaf: true,
                    }).collect()
                } else {
                    vec![]
                }
            }

            // Float64Array
            Value::Float64Array(a) => {
                if let Ok(guard) = a.read() {
                    guard.iter().enumerate().map(|(i, v)| SlotInfo {
                        slot: Slot::Index(i),
                        value_type: "Float64".to_string(),
                        preview: v.to_string(),
                        is_leaf: true,
                    }).collect()
                } else {
                    vec![]
                }
            }

            // Float32Array
            Value::Float32Array(a) => {
                if let Ok(guard) = a.read() {
                    guard.iter().enumerate().map(|(i, v)| SlotInfo {
                        slot: Slot::Index(i),
                        value_type: "Float32".to_string(),
                        preview: v.to_string(),
                        is_leaf: true,
                    }).collect()
                } else {
                    vec![]
                }
            }

            // Tuple
            Value::Tuple(t) => {
                t.iter().enumerate().map(|(i, v)| SlotInfo {
                    slot: Slot::Index(i),
                    value_type: v.type_name().to_string(),
                    preview: v.preview(DEFAULT_MAX_PREVIEW_LEN),
                    is_leaf: v.is_leaf(),
                }).collect()
            }

            // Map
            Value::Map(m) => {
                m.iter().map(|(k, v)| SlotInfo {
                    slot: Slot::MapKey(k.clone()),
                    value_type: v.type_name().to_string(),
                    preview: v.preview(DEFAULT_MAX_PREVIEW_LEN),
                    is_leaf: v.is_leaf(),
                }).collect()
            }

            // Set - no navigation into elements (they're keys)
            Value::Set(_) => vec![],

            // Record
            Value::Record(r) => {
                r.field_names.iter().zip(r.fields.iter()).map(|(name, v)| SlotInfo {
                    slot: Slot::Field(name.clone()),
                    value_type: v.type_name().to_string(),
                    preview: v.preview(DEFAULT_MAX_PREVIEW_LEN),
                    is_leaf: v.is_leaf(),
                }).collect()
            }

            // Variant
            Value::Variant(v) => {
                let mut slots = Vec::new();

                // Positional fields
                for (i, field) in v.fields.iter().enumerate() {
                    slots.push(SlotInfo {
                        slot: Slot::VariantField(i),
                        value_type: field.type_name().to_string(),
                        preview: field.preview(DEFAULT_MAX_PREVIEW_LEN),
                        is_leaf: field.is_leaf(),
                    });
                }

                // Named fields
                if let Some(named) = &v.named_fields {
                    for (name, field) in named {
                        slots.push(SlotInfo {
                            slot: Slot::VariantNamedField(name.clone()),
                            value_type: field.type_name().to_string(),
                            preview: field.preview(DEFAULT_MAX_PREVIEW_LEN),
                            is_leaf: field.is_leaf(),
                        });
                    }
                }

                slots
            }

            // Closure - show captured environment
            Value::Closure(c) => {
                c.captures.iter().enumerate().map(|(i, v)| SlotInfo {
                    slot: Slot::Index(i),
                    value_type: v.type_name().to_string(),
                    preview: v.preview(DEFAULT_MAX_PREVIEW_LEN),
                    is_leaf: v.is_leaf(),
                }).collect()
            }
        }
    }

    /// Navigate to a child value via a slot
    pub fn get_slot(&self, slot: &Slot) -> Option<Value> {
        match (self, slot) {
            (Value::List(l), Slot::Index(i)) => l.get(*i).cloned(),
            (Value::Array(a), Slot::Index(i)) => a.read().ok()?.get(*i).cloned(),
            (Value::Int64Array(a), Slot::Index(i)) => a.read().ok()?.get(*i).map(|v| Value::Int64(*v)),
            (Value::Float64Array(a), Slot::Index(i)) => a.read().ok()?.get(*i).map(|v| Value::Float64(*v)),
            (Value::Tuple(t), Slot::Index(i)) => t.get(*i).cloned(),
            (Value::Map(m), Slot::MapKey(k)) => m.get(k).cloned(),
            (Value::Record(r), Slot::Field(name)) => {
                r.field_names.iter().position(|n| n == name)
                    .and_then(|i| r.fields.get(i).cloned())
            }
            (Value::Variant(v), Slot::VariantField(i)) => v.fields.get(*i).cloned(),
            (Value::Variant(v), Slot::VariantNamedField(name)) => {
                v.named_fields.as_ref().and_then(|nf| nf.get(name).cloned())
            }
            (Value::Closure(c), Slot::Index(i)) => c.captures.get(*i).cloned(),
            _ => None,
        }
    }

    /// Full inspection of this value
    pub fn inspect(&self) -> InspectResult {
        let total_count = match self {
            Value::List(l) => Some(l.len()),
            Value::Array(a) => a.read().ok().map(|g| g.len()),
            Value::Int64Array(a) => a.read().ok().map(|g| g.len()),
            Value::Float64Array(a) => a.read().ok().map(|g| g.len()),
            Value::Tuple(t) => Some(t.len()),
            Value::Map(m) => Some(m.len()),
            Value::Set(s) => Some(s.len()),
            Value::String(s) => Some(s.len()),
            _ => None,
        };

        InspectResult {
            type_name: self.type_name().to_string(),
            preview: self.preview(DEFAULT_MAX_PREVIEW_LEN),
            slots: self.get_slots(),
            is_leaf: self.is_leaf(),
            total_count,
        }
    }

    /// Inspect with pagination for large collections
    pub fn inspect_paginated(&self, offset: usize, limit: usize) -> InspectResult {
        let mut result = self.inspect();

        // Apply pagination to slots
        if result.slots.len() > limit {
            let end = (offset + limit).min(result.slots.len());
            result.slots = result.slots[offset..end].to_vec();
        }

        result
    }
}

/// Path segment for navigation history
#[derive(Debug, Clone)]
pub struct PathSegment {
    /// The slot navigated through
    pub slot: Slot,
    /// Value identity (for cycle detection)
    pub value_id: Option<ValueId>,
}

/// Inspector state for navigating through values
#[derive(Debug, Clone)]
pub struct Inspector {
    /// Root variable name
    pub var_name: String,
    /// The root value being inspected
    root: Value,
    /// Current navigation path
    path: Vec<PathSegment>,
    /// Visited value identities (for cycle detection)
    visited: HashSet<ValueId>,
    /// Pagination offset for current view
    pub page_offset: usize,
    /// Page size
    pub page_size: usize,
}

impl Inspector {
    /// Create a new inspector for a value
    pub fn new(var_name: String, value: Value) -> Self {
        let mut visited = HashSet::new();
        if let Some(id) = value.identity() {
            visited.insert(id);
        }

        Self {
            var_name,
            root: value,
            path: Vec::new(),
            visited,
            page_offset: 0,
            page_size: 20,
        }
    }

    /// Get the current value being viewed
    pub fn current_value(&self) -> Option<Value> {
        let mut current = self.root.clone();
        for segment in &self.path {
            current = current.get_slot(&segment.slot)?;
        }
        Some(current)
    }

    /// Get the current path as a string
    pub fn path_string(&self) -> String {
        let mut path = self.var_name.clone();
        for segment in &self.path {
            path.push_str(&segment.slot.to_path_segment());
        }
        path
    }

    /// Navigate into a slot
    pub fn navigate_to(&mut self, slot: Slot) -> Result<(), String> {
        let current = self.current_value().ok_or("Invalid path")?;
        let child = current.get_slot(&slot).ok_or("Slot not found")?;

        // Check for cycle
        if let Some(id) = child.identity() {
            if self.visited.contains(&id) {
                return Err(format!("Cycle detected: already visited this value"));
            }
            self.visited.insert(id);
        }

        self.path.push(PathSegment {
            slot,
            value_id: child.identity(),
        });
        self.page_offset = 0; // Reset pagination

        Ok(())
    }

    /// Navigate up one level
    pub fn navigate_up(&mut self) -> bool {
        if let Some(segment) = self.path.pop() {
            // Remove from visited set
            if let Some(id) = segment.value_id {
                self.visited.remove(&id);
            }
            self.page_offset = 0;
            true
        } else {
            false
        }
    }

    /// Navigate to root
    pub fn navigate_to_root(&mut self) {
        self.path.clear();
        self.visited.clear();
        if let Some(id) = self.root.identity() {
            self.visited.insert(id);
        }
        self.page_offset = 0;
    }

    /// Get inspection result for current position
    pub fn inspect_current(&self) -> Option<InspectResult> {
        let current = self.current_value()?;
        Some(current.inspect_paginated(self.page_offset, self.page_size))
    }

    /// Check if a slot would create a cycle
    pub fn would_cycle(&self, slot: &Slot) -> bool {
        if let Some(current) = self.current_value() {
            if let Some(child) = current.get_slot(slot) {
                if let Some(id) = child.identity() {
                    return self.visited.contains(&id);
                }
            }
        }
        false
    }

    /// Get depth (how many levels deep we are)
    pub fn depth(&self) -> usize {
        self.path.len()
    }

    /// Next page (for paginated views)
    pub fn next_page(&mut self) {
        if let Some(current) = self.current_value() {
            let slots = current.get_slots();
            if self.page_offset + self.page_size < slots.len() {
                self.page_offset += self.page_size;
            }
        }
    }

    /// Previous page (for paginated views)
    pub fn prev_page(&mut self) {
        if self.page_offset >= self.page_size {
            self.page_offset -= self.page_size;
        } else {
            self.page_offset = 0;
        }
    }

    /// Jump to specific index
    pub fn jump_to_index(&mut self, index: usize) {
        if let Some(current) = self.current_value() {
            let slots = current.get_slots();
            if index < slots.len() {
                self.page_offset = (index / self.page_size) * self.page_size;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_detection() {
        assert!(Value::Int64(42).is_leaf());
        assert!(Value::Bool(true).is_leaf());
        assert!(Value::String(Arc::new("short".to_string())).is_leaf());
        assert!(Value::List(Arc::new(vec![])).is_leaf());
        assert!(!Value::List(Arc::new(vec![Value::Int64(1)])).is_leaf());
    }

    #[test]
    fn test_preview() {
        assert_eq!(Value::Int64(42).preview(50), "42");
        assert_eq!(Value::Bool(true).preview(50), "true");
        assert_eq!(Value::String(Arc::new("hello".to_string())).preview(50), "\"hello\"");

        let list = Value::List(Arc::new(vec![Value::Int64(1), Value::Int64(2)]));
        assert_eq!(list.preview(50), "[1, 2]");
    }

    #[test]
    fn test_slots() {
        let list = Value::List(Arc::new(vec![Value::Int64(1), Value::Int64(2)]));
        let slots = list.get_slots();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].slot, Slot::Index(0));
        assert_eq!(slots[1].slot, Slot::Index(1));
    }

    #[test]
    fn test_navigation() {
        let inner_list = Value::List(Arc::new(vec![Value::Int64(42)]));
        let outer_list = Value::List(Arc::new(vec![inner_list]));

        let mut inspector = Inspector::new("x".to_string(), outer_list);
        assert_eq!(inspector.path_string(), "x");

        inspector.navigate_to(Slot::Index(0)).unwrap();
        assert_eq!(inspector.path_string(), "x[0]");

        inspector.navigate_to(Slot::Index(0)).unwrap();
        assert_eq!(inspector.path_string(), "x[0][0]");

        let current = inspector.current_value().unwrap();
        assert_eq!(current.preview(50), "42");

        inspector.navigate_up();
        assert_eq!(inspector.path_string(), "x[0]");
    }
}
