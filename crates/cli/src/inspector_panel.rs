//! Inspector panel with tabbed value browsing for the TUI.
#![allow(dead_code)]
#![allow(unused_imports)]

use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2};
use nostos_repl::{InspectEntry, ThreadSafeValue, ThreadSafeMapKey, SharedMapKey, SharedMapValue, SharedMap};
use std::collections::VecDeque;
/// Maximum number of tabs in the inspector
const MAX_TABS: usize = 10;

/// An entry in the inspector (a tab)
#[derive(Clone)]
pub struct InspectorTab {
    /// Name/label of the tab
    pub name: String,
    /// The value being inspected (thread-safe copy)
    pub value: ThreadSafeValue,
    /// Current navigation path (breadcrumb)
    pub path: Vec<String>,
    /// Selected index in current view
    pub selected: usize,
    /// Scroll offset for pagination
    pub scroll_offset: usize,
}

impl InspectorTab {
    fn new(name: String, value: ThreadSafeValue) -> Self {
        Self {
            name,
            value,
            path: Vec::new(),
            selected: 0,
            scroll_offset: 0,
        }
    }

    /// Get the current value being viewed (following the path)
    fn current_value(&self) -> Option<ThreadSafeValue> {
        let mut current = self.value.clone();
        for segment in &self.path {
            current = self.navigate_into(&current, segment)?;
        }
        Some(current)
    }

    /// Navigate into a child by path segment
    fn navigate_into(&self, value: &ThreadSafeValue, segment: &str) -> Option<ThreadSafeValue> {
        match value {
            ThreadSafeValue::List(items) => {
                // Parse index from "[n]"
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).cloned()
            }
            ThreadSafeValue::Tuple(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).cloned()
            }
            ThreadSafeValue::Record { fields, field_names, .. } => {
                // Parse field name from ".name"
                let field_name = segment.trim_start_matches('.');
                let idx = field_names.iter().position(|n| n == field_name)?;
                fields.get(idx).cloned()
            }
            ThreadSafeValue::Map(shared_map) => {
                // For maps, we navigate by index for simplicity
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                shared_map.iter().nth(idx).map(|(_, v)| self.shared_value_to_thread_safe(v))
            }
            ThreadSafeValue::Variant { fields, .. } => {
                let idx: usize = segment.trim_start_matches('.').parse().ok()?;
                fields.get(idx).cloned()
            }
            ThreadSafeValue::Closure { captures, .. } => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                captures.get(idx).cloned()
            }
            ThreadSafeValue::Int64Array(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).map(|v| ThreadSafeValue::Int64(*v))
            }
            ThreadSafeValue::Float64Array(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).map(|v| ThreadSafeValue::Float64(*v))
            }
            ThreadSafeValue::Float32Array(items) => {
                let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                items.get(idx).map(|v| ThreadSafeValue::Float64(*v as f64))
            }
            ThreadSafeValue::Set(items) => {
                 let idx: usize = segment.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
                 // ThreadSafeValue::Set uses Vec<ThreadSafeMapKey>, we need to convert to ThreadSafeValue for inspection
                 items.get(idx).map(|k| self.key_to_value(k))
            }
            _ => None,
        }
    }

    fn key_to_value(&self, key: &ThreadSafeMapKey) -> ThreadSafeValue {
        match key {
            ThreadSafeMapKey::Unit => ThreadSafeValue::Unit,
            ThreadSafeMapKey::Bool(b) => ThreadSafeValue::Bool(*b),
            ThreadSafeMapKey::Char(c) => ThreadSafeValue::Char(*c),
            ThreadSafeMapKey::Int8(i) => ThreadSafeValue::Int64(*i as i64),
            ThreadSafeMapKey::Int16(i) => ThreadSafeValue::Int64(*i as i64),
            ThreadSafeMapKey::Int32(i) => ThreadSafeValue::Int64(*i as i64),
            ThreadSafeMapKey::Int64(i) => ThreadSafeValue::Int64(*i),
            ThreadSafeMapKey::UInt8(i) => ThreadSafeValue::Int64(*i as i64),
            ThreadSafeMapKey::UInt16(i) => ThreadSafeValue::Int64(*i as i64),
            ThreadSafeMapKey::UInt32(i) => ThreadSafeValue::Int64(*i as i64),
            ThreadSafeMapKey::UInt64(i) => ThreadSafeValue::Int64(*i as i64), // Potential overflow if u64 > i64::MAX
            ThreadSafeMapKey::String(s) => ThreadSafeValue::String(s.clone()),
            ThreadSafeMapKey::Record { type_name, field_names, fields } => ThreadSafeValue::Record {
                type_name: type_name.clone(),
                field_names: field_names.clone(),
                fields: fields.iter().map(|f| self.key_to_value(f)).collect(),
                mutable_fields: vec![false; fields.len()],
            },
            ThreadSafeMapKey::Variant { type_name, constructor, fields } => ThreadSafeValue::Variant {
                type_name: std::sync::Arc::new(type_name.clone()),
                constructor: std::sync::Arc::new(constructor.clone()),
                fields: fields.iter().map(|f| self.key_to_value(f)).collect(),
            },
        }
    }

    // ---- SharedMap helpers ----

    fn shared_value_to_thread_safe(&self, value: &SharedMapValue) -> ThreadSafeValue {
        match value {
            SharedMapValue::Unit => ThreadSafeValue::Unit,
            SharedMapValue::Bool(b) => ThreadSafeValue::Bool(*b),
            SharedMapValue::Int64(i) => ThreadSafeValue::Int64(*i),
            SharedMapValue::Float64(f) => ThreadSafeValue::Float64(*f),
            SharedMapValue::Pid(p) => ThreadSafeValue::Pid(*p),
            SharedMapValue::String(s) => ThreadSafeValue::String(s.clone()),
            SharedMapValue::Char(c) => ThreadSafeValue::Char(*c),
            SharedMapValue::List(items) => {
                ThreadSafeValue::List(items.iter().map(|v| self.shared_value_to_thread_safe(v)).collect())
            }
            SharedMapValue::Tuple(items) => {
                ThreadSafeValue::Tuple(items.iter().map(|v| self.shared_value_to_thread_safe(v)).collect())
            }
            SharedMapValue::Record { type_name, field_names, fields } => {
                ThreadSafeValue::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|v| self.shared_value_to_thread_safe(v)).collect(),
                    mutable_fields: vec![false; field_names.len()],
                }
            }
            SharedMapValue::Variant { type_name, constructor, fields } => {
                ThreadSafeValue::Variant {
                    type_name: std::sync::Arc::new(type_name.clone()),
                    constructor: std::sync::Arc::new(constructor.clone()),
                    fields: fields.iter().map(|v| self.shared_value_to_thread_safe(v)).collect(),
                }
            }
            SharedMapValue::Map(map) => ThreadSafeValue::Map(map.clone()),
            SharedMapValue::Set(items) => {
                ThreadSafeValue::Set(items.iter().map(|k| self.shared_key_to_thread_safe_key(k)).collect())
            }
            SharedMapValue::Int64Array(items) => ThreadSafeValue::Int64Array(items.clone()),
            SharedMapValue::Float64Array(items) => ThreadSafeValue::Float64Array(items.clone()),
            SharedMapValue::Float32Array(items) => ThreadSafeValue::Float32Array(items.clone()),
        }
    }

    fn shared_key_to_thread_safe_key(&self, key: &SharedMapKey) -> ThreadSafeMapKey {
        match key {
            SharedMapKey::Unit => ThreadSafeMapKey::Unit,
            SharedMapKey::Bool(b) => ThreadSafeMapKey::Bool(*b),
            SharedMapKey::Char(c) => ThreadSafeMapKey::Char(*c),
            SharedMapKey::Int8(i) => ThreadSafeMapKey::Int8(*i),
            SharedMapKey::Int16(i) => ThreadSafeMapKey::Int16(*i),
            SharedMapKey::Int32(i) => ThreadSafeMapKey::Int32(*i),
            SharedMapKey::Int64(i) => ThreadSafeMapKey::Int64(*i),
            SharedMapKey::UInt8(i) => ThreadSafeMapKey::UInt8(*i),
            SharedMapKey::UInt16(i) => ThreadSafeMapKey::UInt16(*i),
            SharedMapKey::UInt32(i) => ThreadSafeMapKey::UInt32(*i),
            SharedMapKey::UInt64(i) => ThreadSafeMapKey::UInt64(*i),
            SharedMapKey::String(s) => ThreadSafeMapKey::String(s.clone()),
            SharedMapKey::Record { type_name, field_names, fields } => ThreadSafeMapKey::Record {
                type_name: type_name.clone(),
                field_names: field_names.clone(),
                fields: fields.iter().map(|f| self.shared_key_to_thread_safe_key(f)).collect(),
            },
            SharedMapKey::Variant { type_name, constructor, fields } => ThreadSafeMapKey::Variant {
                type_name: type_name.clone(),
                constructor: constructor.clone(),
                fields: fields.iter().map(|f| self.shared_key_to_thread_safe_key(f)).collect(),
            },
        }
    }

    fn shared_key_preview(&self, key: &SharedMapKey) -> String {
        match key {
            SharedMapKey::Unit => "()".to_string(),
            SharedMapKey::Bool(b) => b.to_string(),
            SharedMapKey::Char(c) => format!("'{}'", c),
            SharedMapKey::Int8(n) => n.to_string(),
            SharedMapKey::Int16(n) => n.to_string(),
            SharedMapKey::Int32(n) => n.to_string(),
            SharedMapKey::Int64(n) => n.to_string(),
            SharedMapKey::UInt8(n) => n.to_string(),
            SharedMapKey::UInt16(n) => n.to_string(),
            SharedMapKey::UInt32(n) => n.to_string(),
            SharedMapKey::UInt64(n) => n.to_string(),
            SharedMapKey::String(s) => if s.len() > 20 { format!("\"{}...\"", &s[..17]) } else { format!("\"{}\"", s) },
            SharedMapKey::Record { type_name, .. } => format!("{}{{...}}", type_name),
            SharedMapKey::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}.{}", type_name, constructor)
                } else {
                    format!("{}.{}(...)", type_name, constructor)
                }
            }
        }
    }

    fn shared_type_name(&self, value: &SharedMapValue) -> String {
        match value {
            SharedMapValue::Unit => "Unit".to_string(),
            SharedMapValue::Bool(_) => "Bool".to_string(),
            SharedMapValue::Int64(_) => "Int64".to_string(),
            SharedMapValue::Float64(_) => "Float64".to_string(),
            SharedMapValue::Pid(_) => "Pid".to_string(),
            SharedMapValue::String(_) => "String".to_string(),
            SharedMapValue::Char(_) => "Char".to_string(),
            SharedMapValue::List(items) => format!("List({})", items.len()),
            SharedMapValue::Tuple(items) => format!("Tuple({})", items.len()),
            SharedMapValue::Record { type_name, fields, .. } => format!("{}({})", type_name, fields.len()),
            SharedMapValue::Variant { type_name, constructor, .. } => format!("{}::{}", type_name, constructor),
            SharedMapValue::Map(m) => format!("Map({})", m.len()),
            SharedMapValue::Set(items) => format!("Set({})", items.len()),
            SharedMapValue::Int64Array(items) => format!("Int64Array({})", items.len()),
            SharedMapValue::Float64Array(items) => format!("Float64Array({})", items.len()),
            SharedMapValue::Float32Array(items) => format!("Float32Array({})", items.len()),
        }
    }

    fn shared_preview(&self, value: &SharedMapValue) -> String {
        match value {
            SharedMapValue::Unit => "()".to_string(),
            SharedMapValue::Bool(b) => b.to_string(),
            SharedMapValue::Int64(n) => n.to_string(),
            SharedMapValue::Float64(f) => f.to_string(),
            SharedMapValue::Pid(p) => format!("<{}>", p),
            SharedMapValue::String(s) => if s.len() > 30 { format!("\"{}...\"", &s[..27]) } else { format!("\"{}\"", s) },
            SharedMapValue::Char(c) => format!("'{}'", c),
            SharedMapValue::List(items) if items.is_empty() => "[]".to_string(),
            SharedMapValue::List(items) => format!("[...] ({} items)", items.len()),
            SharedMapValue::Tuple(items) if items.is_empty() => "()".to_string(),
            SharedMapValue::Tuple(items) => format!("(...) ({} items)", items.len()),
            SharedMapValue::Record { type_name, .. } => format!("{}{{...}}", type_name),
            SharedMapValue::Variant { constructor, fields, .. } if fields.is_empty() => constructor.clone(),
            SharedMapValue::Variant { constructor, .. } => format!("{}(...)", constructor),
            SharedMapValue::Map(m) if m.is_empty() => "{}".to_string(),
            SharedMapValue::Map(m) => format!("{{...}} ({} entries)", m.len()),
            SharedMapValue::Set(items) if items.is_empty() => "Set{}".to_string(),
            SharedMapValue::Set(items) => format!("Set{{...}} ({} items)", items.len()),
            SharedMapValue::Int64Array(items) => format!("Int64Array({} items)", items.len()),
            SharedMapValue::Float64Array(items) => format!("Float64Array({} items)", items.len()),
            SharedMapValue::Float32Array(items) => format!("Float32Array({} items)", items.len()),
        }
    }

    fn shared_is_leaf(&self, value: &SharedMapValue) -> bool {
        match value {
            SharedMapValue::Unit | SharedMapValue::Bool(_) | SharedMapValue::Int64(_) |
            SharedMapValue::Float64(_) | SharedMapValue::Pid(_) | SharedMapValue::Char(_) => true,
            SharedMapValue::String(s) => s.len() <= 50,
            SharedMapValue::List(items) => items.is_empty(),
            SharedMapValue::Tuple(items) => items.is_empty(),
            SharedMapValue::Record { fields, .. } => fields.is_empty(),
            SharedMapValue::Variant { fields, .. } => fields.is_empty(),
            SharedMapValue::Map(m) => m.is_empty(),
            SharedMapValue::Set(items) => items.is_empty(),
            SharedMapValue::Int64Array(items) => items.is_empty(),
            SharedMapValue::Float64Array(items) => items.is_empty(),
            SharedMapValue::Float32Array(items) => items.is_empty(),
        }
    }

    fn shared_full_format_key(&self, key: &SharedMapKey) -> String {
        match key {
            SharedMapKey::Unit => "()".to_string(),
            SharedMapKey::Bool(b) => b.to_string(),
            SharedMapKey::Char(c) => format!("{:?}", c),
            SharedMapKey::Int8(n) => n.to_string(),
            SharedMapKey::Int16(n) => n.to_string(),
            SharedMapKey::Int32(n) => n.to_string(),
            SharedMapKey::Int64(n) => n.to_string(),
            SharedMapKey::UInt8(n) => n.to_string(),
            SharedMapKey::UInt16(n) => n.to_string(),
            SharedMapKey::UInt32(n) => n.to_string(),
            SharedMapKey::UInt64(n) => n.to_string(),
            SharedMapKey::String(s) => format!("{:?}", s),
            SharedMapKey::Record { type_name, field_names, fields } => {
                let field_strs: Vec<String> = field_names.iter().zip(fields.iter())
                    .map(|(name, val)| format!("{}: {}", name, self.shared_full_format_key(val)))
                    .collect();
                format!("{}{{{}}}", type_name, field_strs.join(", "))
            }
            SharedMapKey::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}.{}", type_name, constructor)
                } else {
                    let field_strs: Vec<String> = fields.iter()
                        .map(|f| self.shared_full_format_key(f))
                        .collect();
                    format!("{}.{}({})", type_name, constructor, field_strs.join(", "))
                }
            }
        }
    }

    fn shared_full_format(&self, value: &SharedMapValue) -> String {
        match value {
            SharedMapValue::Unit => "()".to_string(),
            SharedMapValue::Bool(b) => b.to_string(),
            SharedMapValue::Int64(n) => n.to_string(),
            SharedMapValue::Float64(f) => f.to_string(),
            SharedMapValue::Pid(p) => format!("<{}>", p),
            SharedMapValue::String(s) => format!("{:?}", s),
            SharedMapValue::Char(c) => format!("{:?}", c),
            SharedMapValue::List(items) => {
                let items_str: Vec<String> = items.iter().map(|v| self.shared_full_format(v)).collect();
                format!("[{}]", items_str.join(", "))
            }
            SharedMapValue::Tuple(items) => {
                let items_str: Vec<String> = items.iter().map(|v| self.shared_full_format(v)).collect();
                format!("({})", items_str.join(", "))
            }
            SharedMapValue::Record { type_name, fields, field_names } => {
                let fields_str: Vec<String> = field_names.iter().zip(fields.iter())
                    .map(|(k, v)| format!("{}: {}", k, self.shared_full_format(v)))
                    .collect();
                format!("{} {{ {} }}", type_name, fields_str.join(", "))
            }
            SharedMapValue::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}::{}", type_name, constructor)
                } else {
                    let fields_str: Vec<String> = fields.iter().map(|v| self.shared_full_format(v)).collect();
                    format!("{}::{}({})", type_name, constructor, fields_str.join(", "))
                }
            }
            SharedMapValue::Map(entries) => {
                let entries_str: Vec<String> = entries.iter()
                    .map(|(k, v)| format!("{} => {}", self.shared_full_format_key(k), self.shared_full_format(v)))
                    .collect();
                format!("Map {{ {} }}", entries_str.join(", "))
            }
            SharedMapValue::Set(items) => {
                let items_str: Vec<String> = items.iter().map(|v| self.shared_full_format_key(v)).collect();
                format!("Set {{ {} }}", items_str.join(", "))
            }
            SharedMapValue::Int64Array(items) => format!("{:?}", items),
            SharedMapValue::Float64Array(items) => format!("{:?}", items),
            SharedMapValue::Float32Array(items) => format!("{:?}", items),
        }
    }

    /// Get slots (children) of a value
    fn get_slots(&self, value: &ThreadSafeValue) -> Vec<(String, String, String, bool)> {
        // Returns: (path_segment, type_name, preview, is_leaf)
        match value {
            ThreadSafeValue::List(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Tuple(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Record { fields, field_names, .. } => {
                field_names.iter().zip(fields.iter()).map(|(name, v)| {
                    (format!(".{}", name), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Map(shared_map) => {
                shared_map.iter().enumerate().map(|(i, (k, v))| {
                    (format!("[{}]", i), format!("{} -> {}", self.shared_key_preview(k), self.shared_type_name(v)), self.shared_preview(v), self.shared_is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Variant { fields, .. } => {
                fields.iter().enumerate().map(|(i, v)| {
                    (format!(".{}", i), self.type_name(v), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Closure { captures, capture_names, .. } => {
                capture_names.iter().zip(captures.iter()).enumerate().map(|(i, (name, v))| {
                    (format!("[{}]", i), format!("{}: {}", name, self.type_name(v)), self.preview(v), self.is_leaf(v))
                }).collect()
            }
            ThreadSafeValue::Int64Array(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), "Int64".to_string(), v.to_string(), true)
                }).collect()
            }
            ThreadSafeValue::Float64Array(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), "Float64".to_string(), v.to_string(), true)
                }).collect()
            }
            ThreadSafeValue::Float32Array(items) => {
                items.iter().enumerate().map(|(i, v)| {
                    (format!("[{}]", i), "Float32".to_string(), v.to_string(), true)
                }).collect()
            }
            ThreadSafeValue::Set(items) => {
                items.iter().enumerate().map(|(i, k)| {
                    (format!("[{}]", i), "Key".to_string(), self.map_key_preview(k), true)
                }).collect()
            }
            _ => Vec::new(),
        }
    }

    fn type_name(&self, value: &ThreadSafeValue) -> String {
        match value {
            ThreadSafeValue::Unit => "Unit".to_string(),
            ThreadSafeValue::Bool(_) => "Bool".to_string(),
            ThreadSafeValue::Int64(_) => "Int64".to_string(),
            ThreadSafeValue::Float64(_) => "Float64".to_string(),
            ThreadSafeValue::Pid(_) => "Pid".to_string(),
            ThreadSafeValue::String(_) => "String".to_string(),
            ThreadSafeValue::Char(_) => "Char".to_string(),
            ThreadSafeValue::List(items) => format!("List({})", items.len()),
            ThreadSafeValue::Tuple(items) => format!("Tuple({})", items.len()),
            ThreadSafeValue::Record { type_name, fields, .. } => format!("{}({})", type_name, fields.len()),
            ThreadSafeValue::Closure { .. } => "Closure".to_string(),
            ThreadSafeValue::Variant { type_name, constructor, .. } => format!("{}::{}", type_name, constructor),
            ThreadSafeValue::Function(f) => format!("Fn({})", f.name),
            ThreadSafeValue::NativeFunction(f) => format!("Native({})", f.name),
            ThreadSafeValue::Map(entries) => format!("Map({})", entries.len()),
            ThreadSafeValue::Set(items) => format!("Set({})", items.len()),
            ThreadSafeValue::Int64Array(items) => format!("Int64Array({})", items.len()),
            ThreadSafeValue::Float64Array(items) => format!("Float64Array({})", items.len()),
            ThreadSafeValue::Float32Array(items) => format!("Float32Array({})", items.len()),
        }
    }

    fn preview(&self, value: &ThreadSafeValue) -> String {
        match value {
            ThreadSafeValue::Unit => "()".to_string(),
            ThreadSafeValue::Bool(b) => b.to_string(),
            ThreadSafeValue::Int64(n) => n.to_string(),
            ThreadSafeValue::Float64(f) => f.to_string(),
            ThreadSafeValue::Pid(p) => format!("<{}>", p),
            ThreadSafeValue::String(s) => {
                // Escape special characters for display
                let escaped = s.replace('\\', "\\\\")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                if escaped.len() > 30 {
                    format!("\"{}...\"", &escaped[..27.min(escaped.len())])
                } else {
                    format!("\"{}\"", escaped)
                }
            }
            ThreadSafeValue::Char(c) => format!("'{}'", c),
            ThreadSafeValue::List(items) if items.is_empty() => "[]".to_string(),
            ThreadSafeValue::List(items) if items.len() <= 3 => {
                let previews: Vec<String> = items.iter().take(3).map(|v| self.preview(v)).collect();
                format!("[{}]", previews.join(", "))
            }
            ThreadSafeValue::List(items) => format!("[...] ({} items)", items.len()),
            ThreadSafeValue::Tuple(items) if items.is_empty() => "()".to_string(),
            ThreadSafeValue::Tuple(items) => format!("(...) ({} items)", items.len()),
            ThreadSafeValue::Record { type_name, .. } => format!("{}{{...}}", type_name),
            ThreadSafeValue::Closure { function, .. } => format!("<closure {}>", function.name),
            ThreadSafeValue::Variant { constructor, fields, .. } if fields.is_empty() => constructor.to_string(),
            ThreadSafeValue::Variant { constructor, .. } => format!("{}(...)", constructor),
            ThreadSafeValue::Function(f) => format!("<fn {}>", f.name),
            ThreadSafeValue::NativeFunction(f) => format!("<native {}>", f.name),
            ThreadSafeValue::Map(entries) if entries.is_empty() => "{}".to_string(),
            ThreadSafeValue::Map(entries) => format!("{{...}} ({} entries)", entries.len()),
            ThreadSafeValue::Set(items) if items.is_empty() => "Set{}".to_string(),
            ThreadSafeValue::Set(items) => format!("Set{{...}} ({} items)", items.len()),
            ThreadSafeValue::Int64Array(items) => format!("Int64Array({} items)", items.len()),
            ThreadSafeValue::Float64Array(items) => format!("Float64Array({} items)", items.len()),
            ThreadSafeValue::Float32Array(items) => format!("Float32Array({} items)", items.len()),
        }
    }

    fn map_key_preview(&self, key: &ThreadSafeMapKey) -> String {
        match key {
            ThreadSafeMapKey::Unit => "()".to_string(),
            ThreadSafeMapKey::Bool(b) => b.to_string(),
            ThreadSafeMapKey::Char(c) => format!("'{}'", c),
            ThreadSafeMapKey::Int8(n) => n.to_string(),
            ThreadSafeMapKey::Int16(n) => n.to_string(),
            ThreadSafeMapKey::Int32(n) => n.to_string(),
            ThreadSafeMapKey::Int64(n) => n.to_string(),
            ThreadSafeMapKey::UInt8(n) => n.to_string(),
            ThreadSafeMapKey::UInt16(n) => n.to_string(),
            ThreadSafeMapKey::UInt32(n) => n.to_string(),
            ThreadSafeMapKey::UInt64(n) => n.to_string(),
            ThreadSafeMapKey::String(s) => {
                if s.len() > 20 {
                    format!("\"{}...\"", &s[..17])
                } else {
                    format!("\"{}\"", s)
                }
            }
            ThreadSafeMapKey::Record { type_name, .. } => format!("{}{{...}}", type_name),
            ThreadSafeMapKey::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}.{}", type_name, constructor)
                } else {
                    format!("{}.{}(...)", type_name, constructor)
                }
            }
        }
    }

    fn is_leaf(&self, value: &ThreadSafeValue) -> bool {
        match value {
            ThreadSafeValue::Unit | ThreadSafeValue::Bool(_) | ThreadSafeValue::Int64(_) |
            ThreadSafeValue::Float64(_) | ThreadSafeValue::Pid(_) | ThreadSafeValue::Char(_) |
            ThreadSafeValue::Function(_) | ThreadSafeValue::NativeFunction(_) => true,
            ThreadSafeValue::String(s) => s.len() <= 50,
            ThreadSafeValue::List(items) => items.is_empty(),
            ThreadSafeValue::Tuple(items) => items.is_empty(),
            ThreadSafeValue::Record { fields, .. } => fields.is_empty(),
            ThreadSafeValue::Closure { captures, .. } => captures.is_empty(),
            ThreadSafeValue::Variant { fields, .. } => fields.is_empty(),
            ThreadSafeValue::Map(entries) => entries.is_empty(),
            ThreadSafeValue::Set(items) => items.is_empty(),
            ThreadSafeValue::Int64Array(items) => items.is_empty(),
            ThreadSafeValue::Float64Array(items) => items.is_empty(),
            ThreadSafeValue::Float32Array(items) => items.is_empty(),
        }
    }

    fn full_format_key(&self, key: &ThreadSafeMapKey) -> String {
        match key {
            ThreadSafeMapKey::Unit => "()".to_string(),
            ThreadSafeMapKey::Bool(b) => b.to_string(),
            ThreadSafeMapKey::Char(c) => format!("{:?}", c),
            ThreadSafeMapKey::Int8(n) => n.to_string(),
            ThreadSafeMapKey::Int16(n) => n.to_string(),
            ThreadSafeMapKey::Int32(n) => n.to_string(),
            ThreadSafeMapKey::Int64(n) => n.to_string(),
            ThreadSafeMapKey::UInt8(n) => n.to_string(),
            ThreadSafeMapKey::UInt16(n) => n.to_string(),
            ThreadSafeMapKey::UInt32(n) => n.to_string(),
            ThreadSafeMapKey::UInt64(n) => n.to_string(),
            ThreadSafeMapKey::String(s) => format!("{:?}", s),
            ThreadSafeMapKey::Record { type_name, field_names, fields } => {
                let field_strs: Vec<String> = field_names.iter().zip(fields.iter())
                    .map(|(name, val)| format!("{}: {}", name, self.full_format_key(val)))
                    .collect();
                format!("{}{{{}}}", type_name, field_strs.join(", "))
            }
            ThreadSafeMapKey::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}.{}", type_name, constructor)
                } else {
                    let field_strs: Vec<String> = fields.iter()
                        .map(|f| self.full_format_key(f))
                        .collect();
                    format!("{}.{}({})", type_name, constructor, field_strs.join(", "))
                }
            }
        }
    }

    /// formatting for clipboard copy - full view
    fn full_format(&self, value: &ThreadSafeValue) -> String {
        match value {
            ThreadSafeValue::Unit => "()".to_string(),
            ThreadSafeValue::Bool(b) => b.to_string(),
            ThreadSafeValue::Int64(n) => n.to_string(),
            ThreadSafeValue::Float64(f) => f.to_string(),
            ThreadSafeValue::Pid(p) => format!("<{}>", p),
            ThreadSafeValue::String(s) => format!("{:?}", s),
            ThreadSafeValue::Char(c) => format!("{:?}", c),
            ThreadSafeValue::List(items) => {
                let items_str: Vec<String> = items.iter().map(|v| self.full_format(v)).collect();
                format!("[{}]", items_str.join(", "))
            }
            ThreadSafeValue::Tuple(items) => {
                let items_str: Vec<String> = items.iter().map(|v| self.full_format(v)).collect();
                format!("({})", items_str.join(", "))
            }
            ThreadSafeValue::Record { type_name, fields, field_names, .. } => {
                let fields_str: Vec<String> = field_names.iter().zip(fields.iter())
                    .map(|(k, v)| format!("{}: {}", k, self.full_format(v)))
                    .collect();
                format!("{} {{ {} }}", type_name, fields_str.join(", "))
            }
            ThreadSafeValue::Closure { function, .. } => format!("<closure {}>", function.name),
            ThreadSafeValue::Variant { type_name, constructor, fields } => {
                if fields.is_empty() {
                    format!("{}::{}", type_name, constructor)
                } else {
                    let fields_str: Vec<String> = fields.iter().map(|v| self.full_format(v)).collect();
                    format!("{}::{}({})", type_name, constructor, fields_str.join(", "))
                }
            }
            ThreadSafeValue::Function(f) => format!("<fn {}>", f.name),
            ThreadSafeValue::NativeFunction(f) => format!("<native {}>", f.name),
            ThreadSafeValue::Map(entries) => {
                 let entries_str: Vec<String> = entries.iter()
                    .map(|(k, v)| format!("{} => {}", self.shared_full_format_key(k), self.shared_full_format(v)))
                    .collect();
                format!("Map {{ {} }}", entries_str.join(", "))
            }
            ThreadSafeValue::Set(items) => {
                 let items_str: Vec<String> = items.iter().map(|v| self.full_format_key(v)).collect();
                 format!("Set {{ {} }}", items_str.join(", "))
            }
            ThreadSafeValue::Int64Array(items) => format!("{:?}", items),
            ThreadSafeValue::Float64Array(items) => format!("{:?}", items),
            ThreadSafeValue::Float32Array(items) => format!("{:?}", items),
        }
    }
}

/// The inspector panel with tabs
pub struct InspectorPanel {
    /// Tabs in order (front = most recent)
    tabs: VecDeque<InspectorTab>,
    /// Currently active tab index
    active_tab: usize,
    /// Visible rows in the value browser
    visible_rows: usize,
}

impl InspectorPanel {
    pub fn new() -> Self {
        Self {
            tabs: VecDeque::new(),
            active_tab: 0,
            visible_rows: 10,
        }
    }

    /// Add or update a tab with the given name and value.
    /// If a tab with this name exists, move it to front and update.
    /// If max tabs reached, remove oldest.
    pub fn add_or_update(&mut self, name: String, value: ThreadSafeValue) {
        // Check if tab exists
        if let Some(pos) = self.tabs.iter().position(|t| t.name == name) {
            // Remove and re-add at front
            self.tabs.remove(pos);
        }

        // Remove oldest if at max
        if self.tabs.len() >= MAX_TABS {
            self.tabs.pop_back();
        }

        // Add at front
        self.tabs.push_front(InspectorTab::new(name, value));
        self.active_tab = 0;
    }

    /// Process incoming inspect entries
    pub fn process_entries(&mut self, entries: Vec<InspectEntry>) {
        for entry in entries {
            self.add_or_update(entry.name, entry.value);
        }
    }

    /// Close the current tab
    pub fn close_current_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.tabs.remove(self.active_tab);
            if self.active_tab >= self.tabs.len() && self.active_tab > 0 {
                self.active_tab -= 1;
            }
        }
    }

    /// Select next tab
    pub fn next_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.tabs.len();
        }
    }

    /// Select previous tab
    pub fn prev_tab(&mut self) {
        if !self.tabs.is_empty() {
            if self.active_tab == 0 {
                self.active_tab = self.tabs.len() - 1;
            } else {
                self.active_tab -= 1;
            }
        }
    }

    /// Get the number of tabs
    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tabs.is_empty()
    }

    /// Get the content of the current tab for clipboard copy
    pub fn get_content(&self) -> String {
        if let Some(tab) = self.tabs.get(self.active_tab) {
            if let Some(current) = tab.current_value() {
                return tab.full_format(&current);
            }
        }
        String::new()
    }

    fn current_tab(&self) -> Option<&InspectorTab> {
        self.tabs.get(self.active_tab)
    }

    fn current_tab_mut(&mut self) -> Option<&mut InspectorTab> {
        self.tabs.get_mut(self.active_tab)
    }

    /// Navigate into selected slot
    fn navigate_into(&mut self) {
        if let Some(tab) = self.current_tab_mut() {
            if let Some(current) = tab.current_value() {
                let slots = tab.get_slots(&current);
                if let Some((path_seg, _, _, is_leaf)) = slots.get(tab.selected) {
                    if !is_leaf {
                        tab.path.push(path_seg.clone());
                        tab.selected = 0;
                        tab.scroll_offset = 0;
                    }
                }
            }
        }
    }

    /// Navigate up (back)
    fn navigate_up(&mut self) {
        if let Some(tab) = self.current_tab_mut() {
            if !tab.path.is_empty() {
                tab.path.pop();
                tab.selected = 0;
                tab.scroll_offset = 0;
            }
        }
    }

    /// Select previous item
    fn select_prev(&mut self) {
        if let Some(tab) = self.current_tab_mut() {
            if let Some(current) = tab.current_value() {
                let slots = tab.get_slots(&current);
                if !slots.is_empty() {
                    if tab.selected == 0 {
                        tab.selected = slots.len() - 1;
                    } else {
                        tab.selected -= 1;
                    }
                    // Update scroll
                    if tab.selected < tab.scroll_offset {
                        tab.scroll_offset = tab.selected;
                    }
                }
            }
        }
    }

    /// Select next item
    fn select_next(&mut self) {
        let visible_rows = self.visible_rows;
        if let Some(tab) = self.current_tab_mut() {
            if let Some(current) = tab.current_value() {
                let slots = tab.get_slots(&current);
                if !slots.is_empty() {
                    tab.selected = (tab.selected + 1) % slots.len();
                    // Update scroll
                    if tab.selected >= tab.scroll_offset + visible_rows {
                        tab.scroll_offset = tab.selected - visible_rows + 1;
                    }
                    if tab.selected < tab.scroll_offset {
                        tab.scroll_offset = tab.selected;
                    }
                }
            }
        }
    }
}

impl View for InspectorPanel {
    fn draw(&self, printer: &Printer) {
        if self.tabs.is_empty() {
            printer.print((1, 1), "No values to inspect");
            printer.print((1, 2), "Use inspect(value, \"name\") in your code");
            return;
        }

        let width = printer.size.x;
        let height = printer.size.y;

        // Draw tab bar at top
        let mut x = 0;
        for (i, tab) in self.tabs.iter().enumerate() {
            let label = if tab.name.len() > 12 {
                format!("{}...", &tab.name[..9])
            } else {
                tab.name.clone()
            };

            let style = if i == self.active_tab {
                ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(255, 255, 0))
            } else {
                ColorStyle::new(Color::Rgb(200, 200, 200), Color::TerminalDefault)
            };

            printer.with_color(style, |p| {
                p.print((x, 0), &format!(" {} ", label));
            });
            x += label.len() + 3;
            if x >= width - 3 {
                break;
            }
        }

        // Draw close hint
        if width > 20 {
            printer.print((width.saturating_sub(8), 0), "[x:close]");
        }

        // Draw separator
        for i in 0..width {
            printer.print((i, 1), "─");
        }

        // Draw current tab content
        if let Some(tab) = self.current_tab() {
            // Draw breadcrumb path
            let path_str = if tab.path.is_empty() {
                tab.name.clone()
            } else {
                format!("{}{}", tab.name, tab.path.join(""))
            };
            let path_display = if path_str.len() > width - 2 {
                format!("...{}", &path_str[path_str.len() - width + 5..])
            } else {
                path_str
            };
            printer.with_color(ColorStyle::new(Color::Rgb(100, 200, 255), Color::TerminalDefault), |p| {
                p.print((0, 2), &path_display);
            });

            // Draw type and preview
            if let Some(current) = tab.current_value() {
                // Define colors
                let color_field = ColorStyle::new(Color::Rgb(80, 200, 255), Color::TerminalDefault); // Light Blue
                let color_type = ColorStyle::new(Color::Rgb(100, 255, 120), Color::TerminalDefault); // Light Green
                let color_value = ColorStyle::new(Color::Rgb(220, 220, 220), Color::TerminalDefault); // Off-white
                let color_punct = ColorStyle::new(Color::Rgb(150, 150, 150), Color::TerminalDefault); // Lighter grey for visibility
                
                // Selection colors
                let bg_selected = Color::Rgb(40, 40, 60);
                let color_field_sel = ColorStyle::new(Color::Rgb(80, 200, 255), bg_selected);
                let color_type_sel = ColorStyle::new(Color::Rgb(100, 255, 120), bg_selected);
                let color_value_sel = ColorStyle::new(Color::Rgb(255, 255, 255), bg_selected);
                let color_punct_sel = ColorStyle::new(Color::Rgb(150, 150, 150), bg_selected);
                let color_prefix_sel = ColorStyle::new(Color::Rgb(255, 255, 0), bg_selected);

                let type_str = tab.type_name(&current);
                let preview = tab.preview(&current);

                // Draw type and preview for root
                printer.print((0, 3), ""); // Clear line start
                printer.with_color(color_type, |p| p.print((0, 3), &type_str));
                printer.with_color(color_punct, |p| p.print((type_str.len(), 3), ": "));
                printer.with_color(color_value, |p| p.print((type_str.len() + 2, 3), &preview));

                // Draw slots
                let slots = tab.get_slots(&current);
                if slots.is_empty() {
                    printer.with_color(color_punct, |p| p.print((0, 5), "(no children to browse)"));
                } else {
                    let start_y = 5;
                    let visible = (height - start_y).min(slots.len() - tab.scroll_offset);

                    for (i, (path_seg, type_name, preview, is_leaf)) in slots.iter()
                        .skip(tab.scroll_offset)
                        .take(visible)
                        .enumerate()
                    {
                        let y = start_y + i;
                        let actual_idx = tab.scroll_offset + i;
                        let selected = actual_idx == tab.selected && printer.focused;
                        
                        // Select styles based on selection state
                        let (s_field, s_type, s_value, s_punct, s_prefix) = if selected {
                            (color_field_sel, color_type_sel, color_value_sel, color_punct_sel, color_prefix_sel)
                        } else {
                            (color_field, color_type, color_value, color_punct, ColorStyle::inherit_parent())
                        };

                        // Draw background if selected
                        if selected {
                             printer.with_color(ColorStyle::new(Color::TerminalDefault, bg_selected), |p| {
                                 for x in 0..width {
                                     p.print((x, y), " ");
                                 }
                             });
                        }

                        let mut x = 0;
                        
                        // Prefix (> or space)
                        let prefix = if selected { "> " } else { "  " };
                        printer.with_color(s_prefix, |p| p.print((x, y), prefix));
                        x += 2;

                        // Field Name
                        printer.with_color(s_field, |p| p.print((x, y), path_seg));
                        x += path_seg.len();

                        // Spacer
                        printer.with_color(s_punct, |p| p.print((x, y), " "));
                        x += 1;

                        // Type Name
                        printer.with_color(s_type, |p| p.print((x, y), type_name));
                        x += type_name.len();

                        // Arrow
                        let arrow = if *is_leaf { " " } else { " -> " };
                        printer.with_color(s_punct, |p| p.print((x, y), arrow));
                        x += arrow.len();

                        // Value Preview
                        // Truncate if too long
                        let max_preview = width.saturating_sub(x);
                        if max_preview > 0 {
                            let display_preview = if preview.len() > max_preview {
                                format!("{}...", &preview[..max_preview.saturating_sub(3)])
                            } else {
                                preview.clone()
                            };
                            printer.with_color(s_value, |p| p.print((x, y), &display_preview));
                        }
                    }

                    // Scroll indicator
                    if slots.len() > visible + tab.scroll_offset {
                         printer.with_color(color_punct, |p| p.print((width.saturating_sub(5), height - 1), "more\u{2193}"));
                    }
                    if tab.scroll_offset > 0 {
                         printer.with_color(color_punct, |p| p.print((width.saturating_sub(5), start_y), "more\u{2191}"));
                    }
                }
            }
        }
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        constraint
    }

    fn take_focus(&mut self, _: Direction) -> Result<EventResult, CannotFocus> {
        Ok(EventResult::Consumed(None))
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        match event {
            // Let Tab propagate for window cycling
            Event::Key(Key::Tab) | Event::Shift(Key::Tab) => {
                EventResult::Ignored
            }
            // Close tab
            Event::Char('x') | Event::Char('X') => {
                self.close_current_tab();
                EventResult::Consumed(None)
            }
            // Navigation within value browser
            Event::Key(Key::Up) => {
                self.select_prev();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                self.select_next();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                self.navigate_into();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
                // If at root level (no path), switch to next tab; otherwise navigate into
                if self.current_tab().map(|t| t.path.is_empty()).unwrap_or(true) {
                    self.next_tab();
                } else {
                    self.navigate_into();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Left) => {
                // If at root level (no path), switch to prev tab; otherwise navigate up
                if self.current_tab().map(|t| t.path.is_empty()).unwrap_or(true) {
                    self.prev_tab();
                } else {
                    self.navigate_up();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Backspace) => {
                self.navigate_up();
                EventResult::Consumed(None)
            }
            // Page navigation
            Event::Key(Key::PageUp) => {
                for _ in 0..self.visible_rows {
                    self.select_prev();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageDown) => {
                for _ in 0..self.visible_rows {
                    self.select_next();
                }
                EventResult::Consumed(None)
            }
            // Ctrl+Y is handled at the tui.rs wrapper level
            _ => EventResult::Ignored,
        }
    }

    fn layout(&mut self, size: Vec2) {
        self.visible_rows = size.y.saturating_sub(6);
    }
}
