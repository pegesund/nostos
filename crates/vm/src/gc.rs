//! Garbage Collector for Nostos
//!
//! A simple mark-and-sweep garbage collector designed for:
//! - Per-process heaps (Erlang-style isolation)
//! - JIT-friendly safepoints
//! - Predictable collection behavior
//!
//! # Design
//!
//! Each process has its own `Heap` containing all GC-managed objects.
//! Objects are allocated from a simple vector with a free list for reuse.
//! Collection uses mark-and-sweep: mark all reachable objects from roots,
//! then sweep unmarked objects back to the free list.
//!
//! # Usage
//!
//! ```ignore
//! let mut heap = Heap::new();
//! let ptr = heap.alloc_string("hello".to_string());
//! heap.add_root(ptr.as_raw());
//! heap.collect(); // Safe - ptr is a root
//! ```

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::value::{
    ClosureValue, FunctionValue, MapKey, RecordValue, RuntimeError, TypeValue, Value, VariantValue,
};

/// Raw index into the heap. Used for type-erased operations.
pub type RawGcPtr = u32;

/// A typed pointer to a GC-managed object.
///
/// This is a lightweight handle (just a u32 index) that can be used
/// to access heap objects. The type parameter ensures type safety
/// at compile time.
///
/// GcPtr is Copy because it's just an index - very cheap to copy.
pub struct GcPtr<T> {
    index: RawGcPtr,
    _marker: PhantomData<*const T>,
}

// Manually implement Copy and Clone to avoid T: Copy bounds
impl<T> Copy for GcPtr<T> {}

impl<T> Clone for GcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> GcPtr<T> {
    /// Check if two pointers point to the same object.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> GcPtr<T> {
    /// Create a new GcPtr from a raw index.
    ///
    /// # Safety
    /// The index must point to a valid object of type T in the heap.
    pub(crate) fn from_raw(index: RawGcPtr) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }

    /// Get the raw index for this pointer.
    pub fn as_raw(&self) -> RawGcPtr {
        self.index
    }
}

impl<T> fmt::Debug for GcPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GcPtr({})", self.index)
    }
}

impl<T> PartialEq for GcPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for GcPtr<T> {}

impl<T> std::hash::Hash for GcPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

/// A GC-managed string.
#[derive(Clone, Debug, PartialEq)]
pub struct GcString {
    pub data: String,
}

/// A GC-managed list (immutable).
#[derive(Clone, Debug)]
pub struct GcList {
    pub items: Vec<GcValue>,
}

/// A GC-managed array (mutable).
#[derive(Clone, Debug)]
pub struct GcArray {
    pub items: Vec<GcValue>,
}

/// A GC-managed tuple.
#[derive(Clone, Debug)]
pub struct GcTuple {
    pub items: Vec<GcValue>,
}

/// A GC-managed map.
#[derive(Clone, Debug)]
pub struct GcMap {
    pub entries: HashMap<GcMapKey, GcValue>,
}

/// A GC-managed set.
#[derive(Clone, Debug)]
pub struct GcSet {
    pub items: std::collections::HashSet<GcMapKey>,
}

/// Keys for GC maps/sets (must be hashable, so only immediate values + strings).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GcMapKey {
    Unit,
    Bool(bool),
    Int(i64),
    Char(char),
    String(GcPtr<GcString>),
}

/// A GC-managed record.
#[derive(Clone, Debug)]
pub struct GcRecord {
    pub type_name: String,
    pub field_names: Vec<String>,
    pub fields: Vec<GcValue>,
    pub mutable_fields: Vec<bool>,
}

/// A GC-managed variant.
#[derive(Clone, Debug)]
pub struct GcVariant {
    pub type_name: String,
    pub constructor: String,
    pub fields: Vec<GcValue>,
}

/// A GC-managed closure.
#[derive(Clone, Debug)]
pub struct GcClosure {
    pub function: Rc<FunctionValue>, // The function being closed over
    pub captures: Vec<GcValue>,
    pub capture_names: Vec<String>,
}

/// A native function that works directly with GC values.
///
/// Unlike `NativeFn` which works with `Value`, this takes GcValues directly
/// and has access to the heap for allocations. This avoids copying values
/// at the native function boundary.
pub struct GcNativeFn {
    pub name: String,
    pub arity: usize,
    /// The function takes GcValue args and a mutable heap reference.
    /// Returns a GcValue directly allocated on the heap.
    pub func: Box<dyn Fn(&[GcValue], &mut Heap) -> Result<GcValue, RuntimeError> + Send + Sync>,
}

impl std::fmt::Debug for GcNativeFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcNativeFn")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish()
    }
}

impl Clone for GcNativeFn {
    fn clone(&self) -> Self {
        panic!("GcNativeFn cannot be cloned - use Rc<GcNativeFn>")
    }
}

/// A value that can contain GC pointers (for tracing).
///
/// This mirrors the Value enum but uses GcPtr for heap objects.
/// Immediate values are stored inline.
#[derive(Clone)]
pub enum GcValue {
    // Immediate values (no GC needed)
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),

    // Heap-allocated values (GC-managed)
    String(GcPtr<GcString>),
    List(GcPtr<GcList>),
    Array(GcPtr<GcArray>),
    Tuple(GcPtr<GcTuple>),
    Map(GcPtr<GcMap>),
    Set(GcPtr<GcSet>),
    Record(GcPtr<GcRecord>),
    Variant(GcPtr<GcVariant>),
    Closure(GcPtr<GcClosure>),

    // Callable values (Rc-managed, not GC'd - code doesn't need collection)
    Function(Rc<FunctionValue>),
    NativeFunction(Rc<GcNativeFn>),

    // Special values
    Pid(u64),
    Ref(u64),
    Type(Rc<TypeValue>),
    Pointer(usize),
}

impl PartialEq for GcValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GcValue::Unit, GcValue::Unit) => true,
            (GcValue::Bool(a), GcValue::Bool(b)) => a == b,
            (GcValue::Int(a), GcValue::Int(b)) => a == b,
            (GcValue::Float(a), GcValue::Float(b)) => a == b,
            (GcValue::Char(a), GcValue::Char(b)) => a == b,
            (GcValue::String(a), GcValue::String(b)) => a == b,
            (GcValue::List(a), GcValue::List(b)) => a == b,
            (GcValue::Array(a), GcValue::Array(b)) => a == b,
            (GcValue::Tuple(a), GcValue::Tuple(b)) => a == b,
            (GcValue::Map(a), GcValue::Map(b)) => a == b,
            (GcValue::Set(a), GcValue::Set(b)) => a == b,
            (GcValue::Record(a), GcValue::Record(b)) => a == b,
            (GcValue::Variant(a), GcValue::Variant(b)) => a == b,
            (GcValue::Closure(a), GcValue::Closure(b)) => a == b,
            (GcValue::Function(a), GcValue::Function(b)) => Rc::ptr_eq(a, b),
            (GcValue::NativeFunction(a), GcValue::NativeFunction(b)) => Rc::ptr_eq(a, b),
            (GcValue::Pid(a), GcValue::Pid(b)) => a == b,
            (GcValue::Ref(a), GcValue::Ref(b)) => a == b,
            (GcValue::Type(a), GcValue::Type(b)) => Rc::ptr_eq(a, b),
            (GcValue::Pointer(a), GcValue::Pointer(b)) => a == b,
            _ => false,
        }
    }
}

impl fmt::Debug for GcValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GcValue::Unit => write!(f, "Unit"),
            GcValue::Bool(b) => write!(f, "Bool({})", b),
            GcValue::Int(i) => write!(f, "Int({})", i),
            GcValue::Float(fl) => write!(f, "Float({})", fl),
            GcValue::Char(c) => write!(f, "Char('{}')", c),
            GcValue::String(ptr) => write!(f, "String({:?})", ptr),
            GcValue::List(ptr) => write!(f, "List({:?})", ptr),
            GcValue::Array(ptr) => write!(f, "Array({:?})", ptr),
            GcValue::Tuple(ptr) => write!(f, "Tuple({:?})", ptr),
            GcValue::Map(ptr) => write!(f, "Map({:?})", ptr),
            GcValue::Set(ptr) => write!(f, "Set({:?})", ptr),
            GcValue::Record(ptr) => write!(f, "Record({:?})", ptr),
            GcValue::Variant(ptr) => write!(f, "Variant({:?})", ptr),
            GcValue::Closure(ptr) => write!(f, "Closure({:?})", ptr),
            GcValue::Function(func) => write!(f, "Function({})", func.name),
            GcValue::NativeFunction(func) => write!(f, "NativeFunction({})", func.name),
            GcValue::Pid(p) => write!(f, "Pid({})", p),
            GcValue::Ref(r) => write!(f, "Ref({})", r),
            GcValue::Type(t) => write!(f, "Type({})", t.name),
            GcValue::Pointer(p) => write!(f, "Pointer(0x{:x})", p),
        }
    }
}

impl GcValue {
    /// Get all GC pointers contained in this value.
    pub fn gc_pointers(&self) -> Vec<RawGcPtr> {
        match self {
            GcValue::Unit
            | GcValue::Bool(_)
            | GcValue::Int(_)
            | GcValue::Float(_)
            | GcValue::Char(_)
            | GcValue::Pid(_)
            | GcValue::Ref(_)
            | GcValue::Function(_)
            | GcValue::NativeFunction(_)
            | GcValue::Type(_)
            | GcValue::Pointer(_) => vec![],

            GcValue::String(ptr) => vec![ptr.as_raw()],
            GcValue::List(ptr) => vec![ptr.as_raw()],
            GcValue::Array(ptr) => vec![ptr.as_raw()],
            GcValue::Tuple(ptr) => vec![ptr.as_raw()],
            GcValue::Map(ptr) => vec![ptr.as_raw()],
            GcValue::Set(ptr) => vec![ptr.as_raw()],
            GcValue::Record(ptr) => vec![ptr.as_raw()],
            GcValue::Variant(ptr) => vec![ptr.as_raw()],
            GcValue::Closure(ptr) => vec![ptr.as_raw()],
        }
    }

    /// Check if this value is an immediate (non-heap) value.
    pub fn is_immediate(&self) -> bool {
        matches!(
            self,
            GcValue::Unit
                | GcValue::Bool(_)
                | GcValue::Int(_)
                | GcValue::Float(_)
                | GcValue::Char(_)
                | GcValue::Pid(_)
                | GcValue::Ref(_)
                | GcValue::Function(_)
                | GcValue::NativeFunction(_)
                | GcValue::Type(_)
                | GcValue::Pointer(_)
        )
    }

    /// Get the type name of this value.
    pub fn type_name<'a>(&'a self, heap: &'a Heap) -> &'a str {
        match self {
            GcValue::Unit => "()",
            GcValue::Bool(_) => "Bool",
            GcValue::Int(_) => "Int",
            GcValue::Float(_) => "Float",
            GcValue::Char(_) => "Char",
            GcValue::String(_) => "String",
            GcValue::List(_) => "List",
            GcValue::Array(_) => "Array",
            GcValue::Tuple(_) => "Tuple",
            GcValue::Map(_) => "Map",
            GcValue::Set(_) => "Set",
            GcValue::Record(ptr) => {
                heap.get_record(*ptr)
                    .map(|r| r.type_name.as_str())
                    .unwrap_or("Record")
            }
            GcValue::Variant(ptr) => {
                heap.get_variant(*ptr)
                    .map(|v| v.type_name.as_str())
                    .unwrap_or("Variant")
            }
            GcValue::Closure(_) => "Closure",
            GcValue::Function(_) => "Function",
            GcValue::NativeFunction(_) => "NativeFunction",
            GcValue::Pid(_) => "Pid",
            GcValue::Ref(_) => "Ref",
            GcValue::Type(_) => "Type",
            GcValue::Pointer(_) => "Pointer",
        }
    }

    /// Check if this value is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            GcValue::Bool(b) => *b,
            GcValue::Unit => false,
            _ => true,
        }
    }

    /// Convert to a map key if possible (for Value-based maps).
    pub fn to_map_key(&self, heap: &Heap) -> Option<MapKey> {
        match self {
            GcValue::Unit => Some(MapKey::Unit),
            GcValue::Bool(b) => Some(MapKey::Bool(*b)),
            GcValue::Int(i) => Some(MapKey::Int(*i)),
            GcValue::Char(c) => Some(MapKey::Char(*c)),
            GcValue::String(ptr) => {
                heap.get_string(*ptr)
                    .map(|s| MapKey::String(Rc::new(s.data.clone())))
            }
            _ => None,
        }
    }

    /// Convert to a GC map key if possible (for GcMap/GcSet).
    pub fn to_gc_map_key(&self) -> Option<GcMapKey> {
        match self {
            GcValue::Unit => Some(GcMapKey::Unit),
            GcValue::Bool(b) => Some(GcMapKey::Bool(*b)),
            GcValue::Int(i) => Some(GcMapKey::Int(*i)),
            GcValue::Char(c) => Some(GcMapKey::Char(*c)),
            GcValue::String(ptr) => Some(GcMapKey::String(*ptr)),
            _ => None,
        }
    }
}

/// The type of a heap object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjectType {
    String,
    List,
    Array,
    Tuple,
    Map,
    Set,
    Record,
    Variant,
    Closure,
}

/// A heap object with GC metadata.
#[derive(Clone)]
pub struct GcObject {
    /// The actual data
    pub data: HeapData,
    /// Mark bit for garbage collection
    pub marked: bool,
    /// Size estimate in bytes (for memory pressure tracking)
    pub size: usize,
}

/// The data stored in a heap object.
#[derive(Clone, Debug)]
pub enum HeapData {
    String(GcString),
    List(GcList),
    Array(GcArray),
    Tuple(GcTuple),
    Map(GcMap),
    Set(GcSet),
    Record(GcRecord),
    Variant(GcVariant),
    Closure(GcClosure),
}

impl HeapData {
    /// Get the type of this heap data.
    pub fn object_type(&self) -> ObjectType {
        match self {
            HeapData::String(_) => ObjectType::String,
            HeapData::List(_) => ObjectType::List,
            HeapData::Array(_) => ObjectType::Array,
            HeapData::Tuple(_) => ObjectType::Tuple,
            HeapData::Map(_) => ObjectType::Map,
            HeapData::Set(_) => ObjectType::Set,
            HeapData::Record(_) => ObjectType::Record,
            HeapData::Variant(_) => ObjectType::Variant,
            HeapData::Closure(_) => ObjectType::Closure,
        }
    }

    /// Get all GC pointers contained in this heap data.
    pub fn gc_pointers(&self) -> Vec<RawGcPtr> {
        match self {
            HeapData::String(_) => vec![],
            HeapData::List(list) => list
                .items
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::Array(arr) => arr
                .items
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::Tuple(tuple) => tuple
                .items
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::Map(map) => {
                let mut ptrs = Vec::new();
                for (k, v) in &map.entries {
                    if let GcMapKey::String(ptr) = k {
                        ptrs.push(ptr.as_raw());
                    }
                    ptrs.extend(v.gc_pointers());
                }
                ptrs
            }
            HeapData::Set(set) => set
                .items
                .iter()
                .filter_map(|k| {
                    if let GcMapKey::String(ptr) = k {
                        Some(ptr.as_raw())
                    } else {
                        None
                    }
                })
                .collect(),
            HeapData::Record(rec) => rec
                .fields
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::Variant(var) => var
                .fields
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::Closure(clo) => clo
                .captures
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
        }
    }

    /// Estimate the size of this object in bytes.
    pub fn estimate_size(&self) -> usize {
        match self {
            HeapData::String(s) => std::mem::size_of::<GcString>() + s.data.len(),
            HeapData::List(l) => {
                std::mem::size_of::<GcList>() + l.items.len() * std::mem::size_of::<GcValue>()
            }
            HeapData::Array(a) => {
                std::mem::size_of::<GcArray>() + a.items.len() * std::mem::size_of::<GcValue>()
            }
            HeapData::Tuple(t) => {
                std::mem::size_of::<GcTuple>() + t.items.len() * std::mem::size_of::<GcValue>()
            }
            HeapData::Map(m) => {
                std::mem::size_of::<GcMap>()
                    + m.entries.len() * (std::mem::size_of::<GcMapKey>() + std::mem::size_of::<GcValue>())
            }
            HeapData::Set(s) => {
                std::mem::size_of::<GcSet>() + s.items.len() * std::mem::size_of::<GcMapKey>()
            }
            HeapData::Record(r) => {
                std::mem::size_of::<GcRecord>()
                    + r.fields.len() * std::mem::size_of::<GcValue>()
                    + r.field_names.iter().map(|s| s.len()).sum::<usize>()
            }
            HeapData::Variant(v) => {
                std::mem::size_of::<GcVariant>()
                    + v.fields.len() * std::mem::size_of::<GcValue>()
                    + v.type_name.len()
                    + v.constructor.len()
            }
            HeapData::Closure(c) => {
                std::mem::size_of::<GcClosure>()
                    + c.captures.len() * std::mem::size_of::<GcValue>()
            }
        }
    }
}

impl fmt::Debug for GcObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GcObject")
            .field("data", &self.data)
            .field("marked", &self.marked)
            .field("size", &self.size)
            .finish()
    }
}

/// Statistics about GC activity.
#[derive(Clone, Debug, Default)]
pub struct GcStats {
    /// Number of collections performed
    pub collections: u64,
    /// Total objects allocated
    pub total_allocated: u64,
    /// Total objects freed
    pub total_freed: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Total bytes freed
    pub total_bytes_freed: u64,
    /// Peak number of live objects
    pub peak_objects: usize,
    /// Peak heap size in bytes
    pub peak_bytes: usize,
}

/// Configuration for the garbage collector.
#[derive(Clone, Debug)]
pub struct GcConfig {
    /// Initial capacity of the heap (number of objects)
    pub initial_capacity: usize,
    /// Bytes allocated before triggering collection
    pub gc_threshold: usize,
    /// Growth factor when heap needs to expand
    pub growth_factor: f64,
    /// Whether to print debug info during collection
    pub debug: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            gc_threshold: 1024 * 1024, // 1MB
            growth_factor: 2.0,
            debug: false,
        }
    }
}

/// A garbage-collected heap.
///
/// This is the main GC structure. Each process should have its own Heap.
pub struct Heap {
    /// Storage for all objects
    objects: Vec<Option<GcObject>>,
    /// Free list (indices of available slots)
    free_list: Vec<RawGcPtr>,
    /// Root set (indices that should not be collected)
    roots: Vec<RawGcPtr>,
    /// Bytes allocated since last collection
    bytes_since_gc: usize,
    /// Configuration
    config: GcConfig,
    /// Statistics
    stats: GcStats,
}

impl Heap {
    /// Create a new heap with default configuration.
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    /// Create a new heap with custom configuration.
    pub fn with_config(config: GcConfig) -> Self {
        Self {
            objects: Vec::with_capacity(config.initial_capacity),
            free_list: Vec::new(),
            roots: Vec::new(),
            bytes_since_gc: 0,
            config,
            stats: GcStats::default(),
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }

    /// Get GC statistics.
    pub fn stats(&self) -> &GcStats {
        &self.stats
    }

    /// Get the number of live objects.
    pub fn live_objects(&self) -> usize {
        self.objects.iter().filter(|o| o.is_some()).count()
    }

    /// Get the total heap capacity.
    pub fn capacity(&self) -> usize {
        self.objects.len()
    }

    /// Allocate a new object on the heap.
    fn alloc(&mut self, data: HeapData) -> RawGcPtr {
        let size = data.estimate_size();
        let obj = GcObject {
            data,
            marked: false,
            size,
        };

        // Update stats
        self.stats.total_allocated += 1;
        self.stats.total_bytes_allocated += size as u64;
        self.bytes_since_gc += size;

        // Try to reuse a free slot
        let index = if let Some(free_idx) = self.free_list.pop() {
            self.objects[free_idx as usize] = Some(obj);
            free_idx
        } else {
            // Allocate new slot
            let idx = self.objects.len() as RawGcPtr;
            self.objects.push(Some(obj));
            idx
        };

        // Update peak stats
        let live = self.live_objects();
        if live > self.stats.peak_objects {
            self.stats.peak_objects = live;
        }

        index
    }

    /// Allocate a string.
    pub fn alloc_string(&mut self, s: String) -> GcPtr<GcString> {
        let data = HeapData::String(GcString { data: s });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a list.
    pub fn alloc_list(&mut self, items: Vec<GcValue>) -> GcPtr<GcList> {
        let data = HeapData::List(GcList { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate an array.
    pub fn alloc_array(&mut self, items: Vec<GcValue>) -> GcPtr<GcArray> {
        let data = HeapData::Array(GcArray { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a tuple.
    pub fn alloc_tuple(&mut self, items: Vec<GcValue>) -> GcPtr<GcTuple> {
        let data = HeapData::Tuple(GcTuple { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a map.
    pub fn alloc_map(&mut self, entries: HashMap<GcMapKey, GcValue>) -> GcPtr<GcMap> {
        let data = HeapData::Map(GcMap { entries });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a set.
    pub fn alloc_set(&mut self, items: std::collections::HashSet<GcMapKey>) -> GcPtr<GcSet> {
        let data = HeapData::Set(GcSet { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a record.
    pub fn alloc_record(
        &mut self,
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<GcValue>,
        mutable_fields: Vec<bool>,
    ) -> GcPtr<GcRecord> {
        let data = HeapData::Record(GcRecord {
            type_name,
            field_names,
            fields,
            mutable_fields,
        });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a variant.
    pub fn alloc_variant(
        &mut self,
        type_name: String,
        constructor: String,
        fields: Vec<GcValue>,
    ) -> GcPtr<GcVariant> {
        let data = HeapData::Variant(GcVariant {
            type_name,
            constructor,
            fields,
        });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a closure.
    pub fn alloc_closure(
        &mut self,
        function: Rc<FunctionValue>,
        captures: Vec<GcValue>,
        capture_names: Vec<String>,
    ) -> GcPtr<GcClosure> {
        let data = HeapData::Closure(GcClosure {
            function,
            captures,
            capture_names,
        });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Get an object by raw pointer.
    pub fn get(&self, ptr: RawGcPtr) -> Option<&GcObject> {
        self.objects.get(ptr as usize).and_then(|o| o.as_ref())
    }

    /// Get a mutable reference to an object.
    pub fn get_mut(&mut self, ptr: RawGcPtr) -> Option<&mut GcObject> {
        self.objects.get_mut(ptr as usize).and_then(|o| o.as_mut())
    }

    /// Get a typed reference to heap data.
    pub fn get_string(&self, ptr: GcPtr<GcString>) -> Option<&GcString> {
        match self.get(ptr.as_raw())?.data {
            HeapData::String(ref s) => Some(s),
            _ => None,
        }
    }

    /// Get a typed reference to list data.
    pub fn get_list(&self, ptr: GcPtr<GcList>) -> Option<&GcList> {
        match self.get(ptr.as_raw())?.data {
            HeapData::List(ref l) => Some(l),
            _ => None,
        }
    }

    /// Get a mutable reference to list data.
    pub fn get_list_mut(&mut self, ptr: GcPtr<GcList>) -> Option<&mut GcList> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::List(ref mut l) => Some(l),
            _ => None,
        }
    }

    /// Get a typed reference to array data.
    pub fn get_array(&self, ptr: GcPtr<GcArray>) -> Option<&GcArray> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Array(ref a) => Some(a),
            _ => None,
        }
    }

    /// Get a mutable reference to array data.
    pub fn get_array_mut(&mut self, ptr: GcPtr<GcArray>) -> Option<&mut GcArray> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::Array(ref mut a) => Some(a),
            _ => None,
        }
    }

    /// Get a typed reference to tuple data.
    pub fn get_tuple(&self, ptr: GcPtr<GcTuple>) -> Option<&GcTuple> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Tuple(ref t) => Some(t),
            _ => None,
        }
    }

    /// Get a typed reference to map data.
    pub fn get_map(&self, ptr: GcPtr<GcMap>) -> Option<&GcMap> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Map(ref m) => Some(m),
            _ => None,
        }
    }

    /// Get a mutable reference to map data.
    pub fn get_map_mut(&mut self, ptr: GcPtr<GcMap>) -> Option<&mut GcMap> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::Map(ref mut m) => Some(m),
            _ => None,
        }
    }

    /// Get a typed reference to set data.
    pub fn get_set(&self, ptr: GcPtr<GcSet>) -> Option<&GcSet> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Set(ref s) => Some(s),
            _ => None,
        }
    }

    /// Get a typed reference to record data.
    pub fn get_record(&self, ptr: GcPtr<GcRecord>) -> Option<&GcRecord> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Record(ref r) => Some(r),
            _ => None,
        }
    }

    /// Get a mutable reference to record data.
    pub fn get_record_mut(&mut self, ptr: GcPtr<GcRecord>) -> Option<&mut GcRecord> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::Record(ref mut r) => Some(r),
            _ => None,
        }
    }

    /// Get a typed reference to variant data.
    pub fn get_variant(&self, ptr: GcPtr<GcVariant>) -> Option<&GcVariant> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Variant(ref v) => Some(v),
            _ => None,
        }
    }

    /// Get a typed reference to closure data.
    pub fn get_closure(&self, ptr: GcPtr<GcClosure>) -> Option<&GcClosure> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Closure(ref c) => Some(c),
            _ => None,
        }
    }

    /// Compare two GcValues for equality by content (not by pointer).
    ///
    /// This is needed because heap-allocated values may have different pointers
    /// even if they contain the same data.
    pub fn gc_values_equal(&self, a: &GcValue, b: &GcValue) -> bool {
        match (a, b) {
            // Immediate values - direct comparison
            (GcValue::Unit, GcValue::Unit) => true,
            (GcValue::Bool(a), GcValue::Bool(b)) => a == b,
            (GcValue::Int(a), GcValue::Int(b)) => a == b,
            (GcValue::Float(a), GcValue::Float(b)) => a == b,
            (GcValue::Char(a), GcValue::Char(b)) => a == b,
            (GcValue::Pid(a), GcValue::Pid(b)) => a == b,

            // Strings - compare by content
            (GcValue::String(a), GcValue::String(b)) => {
                if a == b {
                    return true; // Same pointer
                }
                match (self.get_string(*a), self.get_string(*b)) {
                    (Some(sa), Some(sb)) => sa.data == sb.data,
                    _ => false,
                }
            }

            // Lists - compare elements recursively
            (GcValue::List(a), GcValue::List(b)) => {
                if a == b {
                    return true;
                }
                match (self.get_list(*a), self.get_list(*b)) {
                    (Some(la), Some(lb)) => {
                        if la.items.len() != lb.items.len() {
                            return false;
                        }
                        la.items
                            .iter()
                            .zip(lb.items.iter())
                            .all(|(ia, ib)| self.gc_values_equal(ia, ib))
                    }
                    _ => false,
                }
            }

            // Arrays - compare elements recursively
            (GcValue::Array(a), GcValue::Array(b)) => {
                if a == b {
                    return true;
                }
                match (self.get_array(*a), self.get_array(*b)) {
                    (Some(aa), Some(ab)) => {
                        if aa.items.len() != ab.items.len() {
                            return false;
                        }
                        aa.items
                            .iter()
                            .zip(ab.items.iter())
                            .all(|(ia, ib)| self.gc_values_equal(ia, ib))
                    }
                    _ => false,
                }
            }

            // Tuples - compare elements recursively
            (GcValue::Tuple(a), GcValue::Tuple(b)) => {
                if a == b {
                    return true;
                }
                match (self.get_tuple(*a), self.get_tuple(*b)) {
                    (Some(ta), Some(tb)) => {
                        if ta.items.len() != tb.items.len() {
                            return false;
                        }
                        ta.items
                            .iter()
                            .zip(tb.items.iter())
                            .all(|(ia, ib)| self.gc_values_equal(ia, ib))
                    }
                    _ => false,
                }
            }

            // Records - compare type and field values
            (GcValue::Record(a), GcValue::Record(b)) => {
                if a == b {
                    return true;
                }
                match (self.get_record(*a), self.get_record(*b)) {
                    (Some(ra), Some(rb)) => {
                        ra.type_name == rb.type_name
                            && ra.field_names == rb.field_names
                            && ra.fields.len() == rb.fields.len()
                            && ra
                                .fields
                                .iter()
                                .zip(rb.fields.iter())
                                .all(|(fa, fb)| self.gc_values_equal(fa, fb))
                    }
                    _ => false,
                }
            }

            // Variants - compare type, constructor, and field values
            (GcValue::Variant(a), GcValue::Variant(b)) => {
                if a == b {
                    return true;
                }
                match (self.get_variant(*a), self.get_variant(*b)) {
                    (Some(va), Some(vb)) => {
                        va.type_name == vb.type_name
                            && va.constructor == vb.constructor
                            && va.fields.len() == vb.fields.len()
                            && va
                                .fields
                                .iter()
                                .zip(vb.fields.iter())
                                .all(|(fa, fb)| self.gc_values_equal(fa, fb))
                    }
                    _ => false,
                }
            }

            // Maps - compare entries
            (GcValue::Map(a), GcValue::Map(b)) => {
                if a == b {
                    return true;
                }
                // For now, just compare by pointer since deep comparison of maps is complex
                false
            }

            // Sets - compare entries
            (GcValue::Set(a), GcValue::Set(b)) => {
                if a == b {
                    return true;
                }
                // For now, just compare by pointer since deep comparison of sets is complex
                false
            }

            // Functions and closures - compare by identity (pointer)
            (GcValue::Function(a), GcValue::Function(b)) => Rc::ptr_eq(a, b),
            (GcValue::Closure(a), GcValue::Closure(b)) => a == b,
            (GcValue::NativeFunction(a), GcValue::NativeFunction(b)) => Rc::ptr_eq(a, b),

            // Different types - not equal
            _ => false,
        }
    }

    /// Format a GcValue as a display string.
    /// This is used by native functions that need to print or format values.
    pub fn display_value(&self, value: &GcValue) -> String {
        match value {
            GcValue::Unit => "()".to_string(),
            GcValue::Bool(b) => format!("{}", b),
            GcValue::Int(i) => format!("{}", i),
            GcValue::Float(f) => format!("{}", f),
            GcValue::Char(c) => format!("{}", c),
            GcValue::String(ptr) => {
                if let Some(s) = self.get_string(*ptr) {
                    s.data.clone()
                } else {
                    "<invalid string>".to_string()
                }
            }
            GcValue::List(ptr) => {
                if let Some(list) = self.get_list(*ptr) {
                    let mut result = "[".to_string();
                    for (i, item) in list.items.iter().enumerate() {
                        if i > 0 {
                            result.push_str(", ");
                        }
                        result.push_str(&self.display_value(item));
                    }
                    result.push(']');
                    result
                } else {
                    "<invalid list>".to_string()
                }
            }
            GcValue::Array(ptr) => {
                if let Some(arr) = self.get_array(*ptr) {
                    format!("Array[{}]", arr.items.len())
                } else {
                    "<invalid array>".to_string()
                }
            }
            GcValue::Tuple(ptr) => {
                if let Some(tuple) = self.get_tuple(*ptr) {
                    let mut result = "(".to_string();
                    for (i, item) in tuple.items.iter().enumerate() {
                        if i > 0 {
                            result.push_str(", ");
                        }
                        result.push_str(&self.display_value(item));
                    }
                    result.push(')');
                    result
                } else {
                    "<invalid tuple>".to_string()
                }
            }
            GcValue::Map(ptr) => {
                if let Some(map) = self.get_map(*ptr) {
                    format!("%{{...{} entries}}", map.entries.len())
                } else {
                    "<invalid map>".to_string()
                }
            }
            GcValue::Set(ptr) => {
                if let Some(set) = self.get_set(*ptr) {
                    format!("#{{...{} items}}", set.items.len())
                } else {
                    "<invalid set>".to_string()
                }
            }
            GcValue::Record(ptr) => {
                if let Some(rec) = self.get_record(*ptr) {
                    let mut result = format!("{}{{", rec.type_name);
                    for (i, (name, val)) in rec.field_names.iter().zip(rec.fields.iter()).enumerate()
                    {
                        if i > 0 {
                            result.push_str(", ");
                        }
                        result.push_str(&format!("{}: {}", name, self.display_value(val)));
                    }
                    result.push('}');
                    result
                } else {
                    "<invalid record>".to_string()
                }
            }
            GcValue::Variant(ptr) => {
                if let Some(var) = self.get_variant(*ptr) {
                    if var.fields.is_empty() {
                        var.constructor.clone()
                    } else {
                        let mut result = format!("{}(", var.constructor);
                        for (i, field) in var.fields.iter().enumerate() {
                            if i > 0 {
                                result.push_str(", ");
                            }
                            result.push_str(&self.display_value(field));
                        }
                        result.push(')');
                        result
                    }
                } else {
                    "<invalid variant>".to_string()
                }
            }
            GcValue::Closure(ptr) => {
                if let Some(closure) = self.get_closure(*ptr) {
                    format!("<closure {}>", closure.function.name)
                } else {
                    "<invalid closure>".to_string()
                }
            }
            GcValue::Function(f) => format!("<function {}>", f.name),
            GcValue::NativeFunction(n) => format!("<native {}>", n.name),
            GcValue::Pid(p) => format!("<pid {}>", p),
            GcValue::Ref(r) => format!("<ref {}>", r),
            GcValue::Type(t) => format!("<type {}>", t.name),
            GcValue::Pointer(p) => format!("<ptr 0x{:x}>", p),
        }
    }

    /// Add a root to the root set.
    pub fn add_root(&mut self, ptr: RawGcPtr) {
        if !self.roots.contains(&ptr) {
            self.roots.push(ptr);
        }
    }

    /// Remove a root from the root set.
    pub fn remove_root(&mut self, ptr: RawGcPtr) {
        self.roots.retain(|&r| r != ptr);
    }

    /// Clear all roots.
    pub fn clear_roots(&mut self) {
        self.roots.clear();
    }

    /// Set the entire root set (for VM integration).
    pub fn set_roots(&mut self, roots: Vec<RawGcPtr>) {
        self.roots = roots;
    }

    /// Get the current roots.
    pub fn roots(&self) -> &[RawGcPtr] {
        &self.roots
    }

    /// Check if we should trigger a collection.
    pub fn should_collect(&self) -> bool {
        self.bytes_since_gc >= self.config.gc_threshold
    }

    /// Force a garbage collection.
    pub fn collect(&mut self) {
        self.stats.collections += 1;
        if self.config.debug {
            eprintln!(
                "[GC] Starting collection #{}, {} live objects, {} bytes since last GC",
                self.stats.collections,
                self.live_objects(),
                self.bytes_since_gc
            );
        }

        // Phase 1: Mark
        self.mark_phase();

        // Phase 2: Sweep
        let freed = self.sweep_phase();

        if self.config.debug {
            eprintln!(
                "[GC] Collection complete, freed {} objects, {} now live",
                freed,
                self.live_objects()
            );
        }

        self.bytes_since_gc = 0;
    }

    /// Mark phase: mark all reachable objects starting from roots.
    fn mark_phase(&mut self) {
        // Unmark all objects first
        for obj in self.objects.iter_mut().flatten() {
            obj.marked = false;
        }

        // Mark from roots using a worklist
        let mut worklist: Vec<RawGcPtr> = self.roots.clone();

        while let Some(ptr) = worklist.pop() {
            if let Some(obj) = self.objects.get_mut(ptr as usize).and_then(|o| o.as_mut()) {
                if !obj.marked {
                    obj.marked = true;
                    // Add all referenced objects to worklist
                    worklist.extend(obj.data.gc_pointers());
                }
            }
        }
    }

    /// Sweep phase: free unmarked objects.
    fn sweep_phase(&mut self) -> usize {
        let mut freed = 0;
        let mut bytes_freed = 0;

        for i in 0..self.objects.len() {
            if let Some(ref obj) = self.objects[i] {
                if !obj.marked {
                    bytes_freed += obj.size;
                    freed += 1;
                    self.objects[i] = None;
                    self.free_list.push(i as RawGcPtr);
                }
            }
        }

        self.stats.total_freed += freed;
        self.stats.total_bytes_freed += bytes_freed as u64;

        freed as usize
    }

    /// Collect if threshold exceeded, otherwise do nothing.
    pub fn maybe_collect(&mut self) {
        if self.should_collect() {
            self.collect();
        }
    }

    /// Deep copy a value from another heap into this one.
    ///
    /// This is used for message passing between processes.
    pub fn deep_copy(&mut self, value: &GcValue, source: &Heap) -> GcValue {
        match value {
            // Immediate values are copied directly
            GcValue::Unit => GcValue::Unit,
            GcValue::Bool(b) => GcValue::Bool(*b),
            GcValue::Int(i) => GcValue::Int(*i),
            GcValue::Float(f) => GcValue::Float(*f),
            GcValue::Char(c) => GcValue::Char(*c),
            GcValue::Pid(p) => GcValue::Pid(*p),
            GcValue::Ref(r) => GcValue::Ref(*r),

            // Heap values need recursive deep copy
            GcValue::String(ptr) => {
                if let Some(s) = source.get_string(*ptr) {
                    GcValue::String(self.alloc_string(s.data.clone()))
                } else {
                    GcValue::Unit // Fallback for invalid pointer
                }
            }
            GcValue::List(ptr) => {
                if let Some(list) = source.get_list(*ptr) {
                    let items: Vec<GcValue> = list
                        .items
                        .iter()
                        .map(|v| self.deep_copy(v, source))
                        .collect();
                    GcValue::List(self.alloc_list(items))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Array(ptr) => {
                if let Some(arr) = source.get_array(*ptr) {
                    let items: Vec<GcValue> = arr
                        .items
                        .iter()
                        .map(|v| self.deep_copy(v, source))
                        .collect();
                    GcValue::Array(self.alloc_array(items))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Tuple(ptr) => {
                if let Some(tuple) = source.get_tuple(*ptr) {
                    let items: Vec<GcValue> = tuple
                        .items
                        .iter()
                        .map(|v| self.deep_copy(v, source))
                        .collect();
                    GcValue::Tuple(self.alloc_tuple(items))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Map(ptr) => {
                if let Some(map) = source.get_map(*ptr) {
                    let entries: HashMap<GcMapKey, GcValue> = map
                        .entries
                        .iter()
                        .map(|(k, v)| (self.deep_copy_key(k, source), self.deep_copy(v, source)))
                        .collect();
                    GcValue::Map(self.alloc_map(entries))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Set(ptr) => {
                if let Some(set) = source.get_set(*ptr) {
                    let items: std::collections::HashSet<GcMapKey> = set
                        .items
                        .iter()
                        .map(|k| self.deep_copy_key(k, source))
                        .collect();
                    GcValue::Set(self.alloc_set(items))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Record(ptr) => {
                if let Some(rec) = source.get_record(*ptr) {
                    let fields: Vec<GcValue> = rec
                        .fields
                        .iter()
                        .map(|v| self.deep_copy(v, source))
                        .collect();
                    GcValue::Record(self.alloc_record(
                        rec.type_name.clone(),
                        rec.field_names.clone(),
                        fields,
                        rec.mutable_fields.clone(),
                    ))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Variant(ptr) => {
                if let Some(var) = source.get_variant(*ptr) {
                    let fields: Vec<GcValue> = var
                        .fields
                        .iter()
                        .map(|v| self.deep_copy(v, source))
                        .collect();
                    GcValue::Variant(self.alloc_variant(
                        var.type_name.clone(),
                        var.constructor.clone(),
                        fields,
                    ))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Closure(ptr) => {
                if let Some(clo) = source.get_closure(*ptr) {
                    let captures: Vec<GcValue> = clo
                        .captures
                        .iter()
                        .map(|v| self.deep_copy(v, source))
                        .collect();
                    GcValue::Closure(self.alloc_closure(
                        clo.function.clone(),
                        captures,
                        clo.capture_names.clone(),
                    ))
                } else {
                    GcValue::Unit
                }
            }

            // Rc-based values - these are shared references, just clone them
            // (Functions/NativeFunction/Type are typically static code, not data)
            GcValue::Function(f) => GcValue::Function(f.clone()),
            GcValue::NativeFunction(n) => GcValue::NativeFunction(n.clone()),
            GcValue::Type(t) => GcValue::Type(t.clone()),
            GcValue::Pointer(p) => GcValue::Pointer(*p),
        }
    }

    /// Deep copy a map key.
    fn deep_copy_key(&mut self, key: &GcMapKey, source: &Heap) -> GcMapKey {
        match key {
            GcMapKey::Unit => GcMapKey::Unit,
            GcMapKey::Bool(b) => GcMapKey::Bool(*b),
            GcMapKey::Int(i) => GcMapKey::Int(*i),
            GcMapKey::Char(c) => GcMapKey::Char(*c),
            GcMapKey::String(ptr) => {
                if let Some(s) = source.get_string(*ptr) {
                    GcMapKey::String(self.alloc_string(s.data.clone()))
                } else {
                    GcMapKey::Unit
                }
            }
        }
    }

    /// Clone a value within the same heap (deep copy).
    ///
    /// This creates a completely independent copy of the value.
    /// Used by the `copy` built-in function.
    pub fn clone_value(&mut self, value: &GcValue) -> GcValue {
        match value {
            // Immediate values are copied directly
            GcValue::Unit => GcValue::Unit,
            GcValue::Bool(b) => GcValue::Bool(*b),
            GcValue::Int(i) => GcValue::Int(*i),
            GcValue::Float(f) => GcValue::Float(*f),
            GcValue::Char(c) => GcValue::Char(*c),
            GcValue::Pid(p) => GcValue::Pid(*p),
            GcValue::Ref(r) => GcValue::Ref(*r),

            // Heap values need recursive deep copy
            GcValue::String(ptr) => {
                let data = self.get_string(*ptr).map(|s| s.data.clone());
                if let Some(data) = data {
                    GcValue::String(self.alloc_string(data))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::List(ptr) => {
                let items = self.get_list(*ptr).map(|l| l.items.clone());
                if let Some(items) = items {
                    let cloned: Vec<GcValue> = items.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::List(self.alloc_list(cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Array(ptr) => {
                let items = self.get_array(*ptr).map(|a| a.items.clone());
                if let Some(items) = items {
                    let cloned: Vec<GcValue> = items.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Array(self.alloc_array(cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Tuple(ptr) => {
                let items = self.get_tuple(*ptr).map(|t| t.items.clone());
                if let Some(items) = items {
                    let cloned: Vec<GcValue> = items.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Tuple(self.alloc_tuple(cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Map(ptr) => {
                let entries = self.get_map(*ptr).map(|m| m.entries.clone());
                if let Some(entries) = entries {
                    let cloned: HashMap<GcMapKey, GcValue> = entries
                        .iter()
                        .map(|(k, v)| (self.clone_key(k), self.clone_value(v)))
                        .collect();
                    GcValue::Map(self.alloc_map(cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Set(ptr) => {
                let items = self.get_set(*ptr).map(|s| s.items.clone());
                if let Some(items) = items {
                    let cloned: std::collections::HashSet<GcMapKey> =
                        items.iter().map(|k| self.clone_key(k)).collect();
                    GcValue::Set(self.alloc_set(cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Record(ptr) => {
                let rec_data = self.get_record(*ptr).map(|r| {
                    (r.type_name.clone(), r.field_names.clone(), r.fields.clone(), r.mutable_fields.clone())
                });
                if let Some((type_name, field_names, fields, mutable_fields)) = rec_data {
                    let cloned: Vec<GcValue> = fields.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Record(self.alloc_record(type_name, field_names, cloned, mutable_fields))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Variant(ptr) => {
                let var_data = self.get_variant(*ptr).map(|v| {
                    (v.type_name.clone(), v.constructor.clone(), v.fields.clone())
                });
                if let Some((type_name, constructor, fields)) = var_data {
                    let cloned: Vec<GcValue> = fields.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Variant(self.alloc_variant(type_name, constructor, cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Closure(ptr) => {
                let clo_data = self.get_closure(*ptr).map(|c| {
                    (c.function.clone(), c.captures.clone(), c.capture_names.clone())
                });
                if let Some((function, captures, capture_names)) = clo_data {
                    let cloned: Vec<GcValue> = captures.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Closure(self.alloc_closure(function, cloned, capture_names))
                } else {
                    GcValue::Unit
                }
            }

            // Rc-based values - just clone the Rc (shared reference)
            GcValue::Function(f) => GcValue::Function(f.clone()),
            GcValue::NativeFunction(n) => GcValue::NativeFunction(n.clone()),
            GcValue::Type(t) => GcValue::Type(t.clone()),
            GcValue::Pointer(p) => GcValue::Pointer(*p),
        }
    }

    /// Clone a map key within the same heap.
    fn clone_key(&mut self, key: &GcMapKey) -> GcMapKey {
        match key {
            GcMapKey::Unit => GcMapKey::Unit,
            GcMapKey::Bool(b) => GcMapKey::Bool(*b),
            GcMapKey::Int(i) => GcMapKey::Int(*i),
            GcMapKey::Char(c) => GcMapKey::Char(*c),
            GcMapKey::String(ptr) => {
                let data = self.get_string(*ptr).map(|s| s.data.clone());
                if let Some(data) = data {
                    GcMapKey::String(self.alloc_string(data))
                } else {
                    GcMapKey::Unit
                }
            }
        }
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Heap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Heap")
            .field("live_objects", &self.live_objects())
            .field("capacity", &self.objects.len())
            .field("free_list_size", &self.free_list.len())
            .field("roots", &self.roots.len())
            .field("bytes_since_gc", &self.bytes_since_gc)
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================
// Value <-> GcValue Conversion
// ============================================================

impl Heap {
    /// Convert a Value to a GcValue, allocating heap objects as needed.
    ///
    /// This is used when loading constants from bytecode into registers.
    pub fn value_to_gc(&mut self, value: &Value) -> GcValue {
        match value {
            // Immediate values - direct conversion
            Value::Unit => GcValue::Unit,
            Value::Bool(b) => GcValue::Bool(*b),
            Value::Int(i) => GcValue::Int(*i),
            Value::Float(f) => GcValue::Float(*f),
            Value::Char(c) => GcValue::Char(*c),

            // String - allocate on GC heap
            Value::String(s) => {
                let ptr = self.alloc_string((**s).clone());
                GcValue::String(ptr)
            }

            // List - recursively convert elements
            Value::List(items) => {
                let gc_items: Vec<GcValue> = items.iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_list(gc_items);
                GcValue::List(ptr)
            }

            // Array - recursively convert elements
            Value::Array(arr) => {
                let gc_items: Vec<GcValue> =
                    arr.borrow().iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_array(gc_items);
                GcValue::Array(ptr)
            }

            // Tuple - recursively convert elements
            Value::Tuple(items) => {
                let gc_items: Vec<GcValue> = items.iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_tuple(gc_items);
                GcValue::Tuple(ptr)
            }

            // Map - convert keys and values
            Value::Map(entries) => {
                let gc_entries: HashMap<GcMapKey, GcValue> = entries
                    .iter()
                    .map(|(k, v)| (self.map_key_to_gc(k), self.value_to_gc(v)))
                    .collect();
                let ptr = self.alloc_map(gc_entries);
                GcValue::Map(ptr)
            }

            // Set - convert keys
            Value::Set(items) => {
                let gc_items: std::collections::HashSet<GcMapKey> =
                    items.iter().map(|k| self.map_key_to_gc(k)).collect();
                let ptr = self.alloc_set(gc_items);
                GcValue::Set(ptr)
            }

            // Record - convert fields
            Value::Record(r) => {
                let gc_fields: Vec<GcValue> = r.fields.iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_record(
                    r.type_name.clone(),
                    r.field_names.clone(),
                    gc_fields,
                    r.mutable_fields.clone(),
                );
                GcValue::Record(ptr)
            }

            // Variant - convert fields
            Value::Variant(v) => {
                let gc_fields: Vec<GcValue> = v.fields.iter().map(|f| self.value_to_gc(f)).collect();
                let ptr =
                    self.alloc_variant(v.type_name.clone(), v.constructor.clone(), gc_fields);
                GcValue::Variant(ptr)
            }

            // Closure - convert captures
            Value::Closure(c) => {
                let gc_captures: Vec<GcValue> =
                    c.captures.iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_closure(c.function.clone(), gc_captures, c.capture_names.clone());
                GcValue::Closure(ptr)
            }

            // Function/Type - these are Rc-managed, just clone
            Value::Function(f) => GcValue::Function(f.clone()),
            Value::Type(t) => GcValue::Type(t.clone()),

            // NativeFunction cannot be converted - they use different signatures
            // Native functions should be registered directly as GcNativeFn
            Value::NativeFunction(_) => {
                panic!("Cannot convert Value::NativeFunction to GcValue - use GcNativeFn directly")
            }

            // Special values
            Value::Pid(p) => GcValue::Pid(p.0),
            Value::Ref(r) => GcValue::Ref(r.0),
            Value::Pointer(p) => GcValue::Pointer(*p),
        }
    }

    /// Convert a MapKey to a GcMapKey.
    fn map_key_to_gc(&mut self, key: &MapKey) -> GcMapKey {
        match key {
            MapKey::Unit => GcMapKey::Unit,
            MapKey::Bool(b) => GcMapKey::Bool(*b),
            MapKey::Int(i) => GcMapKey::Int(*i),
            MapKey::Char(c) => GcMapKey::Char(*c),
            MapKey::String(s) => {
                let ptr = self.alloc_string((**s).clone());
                GcMapKey::String(ptr)
            }
        }
    }

    /// Convert a GcValue back to a Value.
    ///
    /// This is used when returning results from the VM.
    pub fn gc_to_value(&self, gc_value: &GcValue) -> Value {
        use crate::value::Pid as ValuePid;
        use crate::value::RefId as ValueRefId;
        use std::cell::RefCell;

        match gc_value {
            // Immediate values - direct conversion
            GcValue::Unit => Value::Unit,
            GcValue::Bool(b) => Value::Bool(*b),
            GcValue::Int(i) => Value::Int(*i),
            GcValue::Float(f) => Value::Float(*f),
            GcValue::Char(c) => Value::Char(*c),

            // String
            GcValue::String(ptr) => {
                let s = self.get_string(*ptr).expect("invalid string pointer");
                Value::String(Rc::new(s.data.clone()))
            }

            // List
            GcValue::List(ptr) => {
                let list = self.get_list(*ptr).expect("invalid list pointer");
                let items: Vec<Value> = list.items.iter().map(|v| self.gc_to_value(v)).collect();
                Value::List(Rc::new(items))
            }

            // Array
            GcValue::Array(ptr) => {
                let arr = self.get_array(*ptr).expect("invalid array pointer");
                let items: Vec<Value> = arr.items.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Array(Rc::new(RefCell::new(items)))
            }

            // Tuple
            GcValue::Tuple(ptr) => {
                let tuple = self.get_tuple(*ptr).expect("invalid tuple pointer");
                let items: Vec<Value> = tuple.items.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Tuple(Rc::new(items))
            }

            // Map
            GcValue::Map(ptr) => {
                let map = self.get_map(*ptr).expect("invalid map pointer");
                let entries: HashMap<MapKey, Value> = map
                    .entries
                    .iter()
                    .map(|(k, v)| (self.gc_map_key_to_value(k), self.gc_to_value(v)))
                    .collect();
                Value::Map(Rc::new(entries))
            }

            // Set
            GcValue::Set(ptr) => {
                let set = self.get_set(*ptr).expect("invalid set pointer");
                let items: std::collections::HashSet<MapKey> =
                    set.items.iter().map(|k| self.gc_map_key_to_value(k)).collect();
                Value::Set(Rc::new(items))
            }

            // Record
            GcValue::Record(ptr) => {
                let record = self.get_record(*ptr).expect("invalid record pointer");
                let fields: Vec<Value> =
                    record.fields.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Record(Rc::new(RecordValue {
                    type_name: record.type_name.clone(),
                    field_names: record.field_names.clone(),
                    fields,
                    mutable_fields: record.mutable_fields.clone(),
                }))
            }

            // Variant
            GcValue::Variant(ptr) => {
                let variant = self.get_variant(*ptr).expect("invalid variant pointer");
                let fields: Vec<Value> =
                    variant.fields.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Variant(Rc::new(VariantValue {
                    type_name: variant.type_name.clone(),
                    constructor: variant.constructor.clone(),
                    fields,
                    named_fields: None,
                }))
            }

            // Closure - we can't fully reconstruct without the function
            GcValue::Closure(ptr) => {
                let closure = self.get_closure(*ptr).expect("invalid closure pointer");
                let captures: Vec<Value> =
                    closure.captures.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Closure(Rc::new(ClosureValue {
                    function: closure.function.clone(),
                    captures,
                    capture_names: closure.capture_names.clone(),
                }))
            }

            // Function/Type - just clone the Rc
            GcValue::Function(f) => Value::Function(f.clone()),
            GcValue::Type(t) => Value::Type(t.clone()),

            // NativeFunction cannot be converted back - different signature
            GcValue::NativeFunction(_) => {
                panic!("Cannot convert GcValue::NativeFunction to Value")
            }

            // Special values
            GcValue::Pid(p) => Value::Pid(ValuePid(*p)),
            GcValue::Ref(r) => Value::Ref(ValueRefId(*r)),
            GcValue::Pointer(p) => Value::Pointer(*p),
        }
    }

    /// Convert a GcMapKey back to a MapKey.
    fn gc_map_key_to_value(&self, key: &GcMapKey) -> MapKey {
        match key {
            GcMapKey::Unit => MapKey::Unit,
            GcMapKey::Bool(b) => MapKey::Bool(*b),
            GcMapKey::Int(i) => MapKey::Int(*i),
            GcMapKey::Char(c) => MapKey::Char(*c),
            GcMapKey::String(ptr) => {
                let s = self.get_string(*ptr).expect("invalid string pointer in map key");
                MapKey::String(Rc::new(s.data.clone()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Basic Allocation Tests
    // ============================================================

    #[test]
    fn test_alloc_string() {
        let mut heap = Heap::new();
        let ptr = heap.alloc_string("hello".to_string());

        let s = heap.get_string(ptr).unwrap();
        assert_eq!(s.data, "hello");
    }

    #[test]
    fn test_alloc_multiple_strings() {
        let mut heap = Heap::new();
        let ptr1 = heap.alloc_string("hello".to_string());
        let ptr2 = heap.alloc_string("world".to_string());
        let ptr3 = heap.alloc_string("foo".to_string());

        assert_eq!(heap.get_string(ptr1).unwrap().data, "hello");
        assert_eq!(heap.get_string(ptr2).unwrap().data, "world");
        assert_eq!(heap.get_string(ptr3).unwrap().data, "foo");
        assert_eq!(heap.live_objects(), 3);
    }

    #[test]
    fn test_alloc_empty_string() {
        let mut heap = Heap::new();
        let ptr = heap.alloc_string(String::new());

        let s = heap.get_string(ptr).unwrap();
        assert_eq!(s.data, "");
    }

    #[test]
    fn test_alloc_list() {
        let mut heap = Heap::new();
        let items = vec![GcValue::Int(1), GcValue::Int(2), GcValue::Int(3)];
        let ptr = heap.alloc_list(items);

        let list = heap.get_list(ptr).unwrap();
        assert_eq!(list.items.len(), 3);
        assert!(matches!(list.items[0], GcValue::Int(1)));
        assert!(matches!(list.items[1], GcValue::Int(2)));
        assert!(matches!(list.items[2], GcValue::Int(3)));
    }

    #[test]
    fn test_alloc_nested_list() {
        let mut heap = Heap::new();

        // Create inner list [1, 2]
        let inner = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(2)]);

        // Create outer list [[1, 2], 3]
        let outer = heap.alloc_list(vec![GcValue::List(inner), GcValue::Int(3)]);

        let outer_list = heap.get_list(outer).unwrap();
        assert_eq!(outer_list.items.len(), 2);

        if let GcValue::List(inner_ptr) = &outer_list.items[0] {
            let inner_list = heap.get_list(*inner_ptr).unwrap();
            assert_eq!(inner_list.items.len(), 2);
        } else {
            panic!("Expected inner list");
        }
    }

    #[test]
    fn test_alloc_array() {
        let mut heap = Heap::new();
        let items = vec![GcValue::Float(1.5), GcValue::Float(2.5)];
        let ptr = heap.alloc_array(items);

        let arr = heap.get_array(ptr).unwrap();
        assert_eq!(arr.items.len(), 2);
    }

    #[test]
    fn test_alloc_tuple() {
        let mut heap = Heap::new();
        let s = heap.alloc_string("hello".to_string());
        let items = vec![GcValue::Int(42), GcValue::String(s), GcValue::Bool(true)];
        let ptr = heap.alloc_tuple(items);

        let tuple = heap.get_tuple(ptr).unwrap();
        assert_eq!(tuple.items.len(), 3);
    }

    #[test]
    fn test_alloc_map() {
        let mut heap = Heap::new();
        let mut entries = HashMap::new();
        entries.insert(GcMapKey::Int(1), GcValue::Bool(true));
        entries.insert(GcMapKey::Int(2), GcValue::Bool(false));
        let ptr = heap.alloc_map(entries);

        let map = heap.get_map(ptr).unwrap();
        assert_eq!(map.entries.len(), 2);
        assert_eq!(map.entries.get(&GcMapKey::Int(1)), Some(&GcValue::Bool(true)));
    }

    #[test]
    fn test_alloc_set() {
        let mut heap = Heap::new();
        let mut items = std::collections::HashSet::new();
        items.insert(GcMapKey::Int(1));
        items.insert(GcMapKey::Int(2));
        items.insert(GcMapKey::Int(3));
        let ptr = heap.alloc_set(items);

        let set = heap.get_set(ptr).unwrap();
        assert_eq!(set.items.len(), 3);
        assert!(set.items.contains(&GcMapKey::Int(1)));
    }

    #[test]
    fn test_alloc_record() {
        let mut heap = Heap::new();
        let ptr = heap.alloc_record(
            "Point".to_string(),
            vec!["x".to_string(), "y".to_string()],
            vec![GcValue::Int(10), GcValue::Int(20)],
            vec![false, false],
        );

        let rec = heap.get_record(ptr).unwrap();
        assert_eq!(rec.type_name, "Point");
        assert_eq!(rec.field_names, vec!["x", "y"]);
        assert_eq!(rec.fields.len(), 2);
    }

    #[test]
    fn test_alloc_variant() {
        let mut heap = Heap::new();
        let ptr = heap.alloc_variant(
            "Option".to_string(),
            "Some".to_string(),
            vec![GcValue::Int(42)],
        );

        let var = heap.get_variant(ptr).unwrap();
        assert_eq!(var.type_name, "Option");
        assert_eq!(var.constructor, "Some");
        assert_eq!(var.fields.len(), 1);
    }

    #[test]
    fn test_alloc_closure() {
        use crate::value::{FunctionValue, Chunk};
        use std::rc::Rc;

        let mut heap = Heap::new();

        // Create a test function
        let func = Rc::new(FunctionValue {
            name: "test_closure".to_string(),
            arity: 2,
            param_names: vec!["x".to_string(), "y".to_string()],
            code: Rc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
        });

        let ptr = heap.alloc_closure(
            func,
            vec![GcValue::Int(10), GcValue::Int(20)],
            vec!["x".to_string(), "y".to_string()],
        );

        let clo = heap.get_closure(ptr).unwrap();
        assert_eq!(clo.function.name, "test_closure");
        assert_eq!(clo.function.arity, 2);
        assert_eq!(clo.captures.len(), 2);
    }

    // ============================================================
    // GC Collection Tests
    // ============================================================

    #[test]
    fn test_gc_collects_unreachable() {
        let mut heap = Heap::new();

        // Allocate some strings without rooting them
        let _ptr1 = heap.alloc_string("garbage1".to_string());
        let _ptr2 = heap.alloc_string("garbage2".to_string());
        let _ptr3 = heap.alloc_string("garbage3".to_string());

        assert_eq!(heap.live_objects(), 3);

        // Collect - all should be freed since no roots
        heap.collect();

        assert_eq!(heap.live_objects(), 0);
        assert_eq!(heap.stats.total_freed, 3);
    }

    #[test]
    fn test_gc_preserves_rooted() {
        let mut heap = Heap::new();

        let ptr1 = heap.alloc_string("keep me".to_string());
        let _ptr2 = heap.alloc_string("garbage".to_string());

        // Root ptr1
        heap.add_root(ptr1.as_raw());

        assert_eq!(heap.live_objects(), 2);
        heap.collect();
        assert_eq!(heap.live_objects(), 1);

        // ptr1 should still be accessible
        assert_eq!(heap.get_string(ptr1).unwrap().data, "keep me");
    }

    #[test]
    fn test_gc_follows_references() {
        let mut heap = Heap::new();

        // Create a list containing strings
        let s1 = heap.alloc_string("item1".to_string());
        let s2 = heap.alloc_string("item2".to_string());
        let list = heap.alloc_list(vec![GcValue::String(s1), GcValue::String(s2)]);

        // Also create some garbage
        let _garbage = heap.alloc_string("garbage".to_string());

        // Only root the list
        heap.add_root(list.as_raw());

        assert_eq!(heap.live_objects(), 4);
        heap.collect();

        // List and its strings should survive, garbage should not
        assert_eq!(heap.live_objects(), 3);

        // Verify data is intact
        let list_data = heap.get_list(list).unwrap();
        if let GcValue::String(ptr) = &list_data.items[0] {
            assert_eq!(heap.get_string(*ptr).unwrap().data, "item1");
        }
    }

    #[test]
    fn test_gc_deeply_nested() {
        let mut heap = Heap::new();

        // Create deeply nested structure
        let s = heap.alloc_string("deep".to_string());
        let l1 = heap.alloc_list(vec![GcValue::String(s)]);
        let l2 = heap.alloc_list(vec![GcValue::List(l1)]);
        let l3 = heap.alloc_list(vec![GcValue::List(l2)]);
        let l4 = heap.alloc_list(vec![GcValue::List(l3)]);

        // Add garbage at various levels
        let _g1 = heap.alloc_string("garbage1".to_string());
        let _g2 = heap.alloc_list(vec![GcValue::Int(1)]);

        heap.add_root(l4.as_raw());

        assert_eq!(heap.live_objects(), 7);
        heap.collect();
        assert_eq!(heap.live_objects(), 5); // l4, l3, l2, l1, s
    }

    #[test]
    fn test_gc_multiple_roots() {
        let mut heap = Heap::new();

        let ptr1 = heap.alloc_string("root1".to_string());
        let ptr2 = heap.alloc_string("root2".to_string());
        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(ptr1.as_raw());
        heap.add_root(ptr2.as_raw());

        heap.collect();

        assert_eq!(heap.live_objects(), 2);
    }

    #[test]
    fn test_gc_remove_root() {
        let mut heap = Heap::new();

        let ptr = heap.alloc_string("temp root".to_string());
        heap.add_root(ptr.as_raw());

        heap.collect();
        assert_eq!(heap.live_objects(), 1);

        heap.remove_root(ptr.as_raw());
        heap.collect();
        assert_eq!(heap.live_objects(), 0);
    }

    #[test]
    fn test_gc_cyclic_references() {
        // This tests that cycles are properly collected
        // We can't actually create cycles with immutable lists,
        // but we can test that the GC handles the structure correctly
        let mut heap = Heap::new();

        // Create a structure that references itself indirectly
        // through an array (which is mutable)
        let _arr = heap.alloc_array(vec![GcValue::Int(1)]);

        // The array is not reachable from any root
        assert_eq!(heap.live_objects(), 1);
        heap.collect();
        assert_eq!(heap.live_objects(), 0);
    }

    #[test]
    fn test_gc_record_with_references() {
        let mut heap = Heap::new();

        let name = heap.alloc_string("Alice".to_string());
        let person = heap.alloc_record(
            "Person".to_string(),
            vec!["name".to_string(), "age".to_string()],
            vec![GcValue::String(name), GcValue::Int(30)],
            vec![false, false],
        );

        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(person.as_raw());
        heap.collect();

        assert_eq!(heap.live_objects(), 2); // person + name string
    }

    #[test]
    fn test_gc_variant_with_references() {
        let mut heap = Heap::new();

        let value = heap.alloc_string("result".to_string());
        let variant = heap.alloc_variant(
            "Result".to_string(),
            "Ok".to_string(),
            vec![GcValue::String(value)],
        );

        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(variant.as_raw());
        heap.collect();

        assert_eq!(heap.live_objects(), 2); // variant + value string
    }

    #[test]
    fn test_gc_closure_with_captures() {
        use crate::value::{FunctionValue, Chunk};
        use std::rc::Rc;

        let mut heap = Heap::new();

        // Create a test function
        let func = Rc::new(FunctionValue {
            name: "test_closure".to_string(),
            arity: 2,
            param_names: vec!["x".to_string(), "y".to_string()],
            code: Rc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
        });

        let x = heap.alloc_string("captured x".to_string());
        let y = heap.alloc_string("captured y".to_string());
        let closure = heap.alloc_closure(
            func,
            vec![GcValue::String(x), GcValue::String(y)],
            vec!["x".to_string(), "y".to_string()],
        );

        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(closure.as_raw());
        heap.collect();

        assert_eq!(heap.live_objects(), 3); // closure + x + y
    }

    #[test]
    fn test_gc_map_with_string_keys() {
        let mut heap = Heap::new();

        let key = heap.alloc_string("my_key".to_string());
        let value = heap.alloc_string("my_value".to_string());

        let mut entries = HashMap::new();
        entries.insert(GcMapKey::String(key), GcValue::String(value));
        entries.insert(GcMapKey::Int(42), GcValue::Bool(true));

        let map = heap.alloc_map(entries);

        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(map.as_raw());
        heap.collect();

        assert_eq!(heap.live_objects(), 3); // map + key + value
    }

    // ============================================================
    // Free List Reuse Tests
    // ============================================================

    #[test]
    fn test_free_list_reuse() {
        let mut heap = Heap::new();

        // Allocate and collect several times
        for _ in 0..10 {
            let _ptr = heap.alloc_string("temp".to_string());
        }

        heap.collect();

        // Free list should have 10 slots
        assert_eq!(heap.free_list.len(), 10);

        // New allocations should reuse free slots
        for _ in 0..5 {
            let _ptr = heap.alloc_string("reused".to_string());
        }

        assert_eq!(heap.free_list.len(), 5);
        assert_eq!(heap.live_objects(), 5);
    }

    #[test]
    fn test_capacity_growth() {
        let config = GcConfig {
            initial_capacity: 2,
            ..Default::default()
        };
        let mut heap = Heap::with_config(config);

        // Allocate more than initial capacity
        let ptr1 = heap.alloc_string("one".to_string());
        let ptr2 = heap.alloc_string("two".to_string());
        let ptr3 = heap.alloc_string("three".to_string());

        heap.add_root(ptr1.as_raw());
        heap.add_root(ptr2.as_raw());
        heap.add_root(ptr3.as_raw());

        assert_eq!(heap.live_objects(), 3);
        assert!(heap.capacity() >= 3);
    }

    // ============================================================
    // Statistics Tests
    // ============================================================

    #[test]
    fn test_stats_tracking() {
        let mut heap = Heap::new();

        let _ptr1 = heap.alloc_string("one".to_string());
        let _ptr2 = heap.alloc_string("two".to_string());

        assert_eq!(heap.stats.total_allocated, 2);
        assert!(heap.stats.total_bytes_allocated > 0);

        heap.collect();

        assert_eq!(heap.stats.collections, 1);
        assert_eq!(heap.stats.total_freed, 2);
    }

    #[test]
    fn test_peak_tracking() {
        let mut heap = Heap::new();

        let ptrs: Vec<_> = (0..100)
            .map(|i| heap.alloc_string(format!("string{}", i)))
            .collect();

        assert_eq!(heap.stats.peak_objects, 100);

        // Root only half
        for ptr in ptrs.iter().take(50) {
            heap.add_root(ptr.as_raw());
        }

        heap.collect();

        // Peak should still be 100
        assert_eq!(heap.stats.peak_objects, 100);
        assert_eq!(heap.live_objects(), 50);
    }

    // ============================================================
    // Deep Copy Tests
    // ============================================================

    #[test]
    fn test_deep_copy_immediate() {
        let heap1 = Heap::new();
        let mut heap2 = Heap::new();

        let value = GcValue::Int(42);
        let copied = heap2.deep_copy(&value, &heap1);

        assert!(matches!(copied, GcValue::Int(42)));
    }

    #[test]
    fn test_deep_copy_string() {
        let mut heap1 = Heap::new();
        let mut heap2 = Heap::new();

        let ptr = heap1.alloc_string("original".to_string());
        let value = GcValue::String(ptr);

        let copied = heap2.deep_copy(&value, &heap1);

        if let GcValue::String(new_ptr) = copied {
            // Verify the data was copied correctly
            assert_eq!(heap2.get_string(new_ptr).unwrap().data, "original");
            // Original still intact
            assert_eq!(heap1.get_string(ptr).unwrap().data, "original");
        } else {
            panic!("Expected string");
        }

        // Verify independent heaps - both have exactly 1 object
        assert_eq!(heap1.live_objects(), 1);
        assert_eq!(heap2.live_objects(), 1);

        // Further verify independence: collecting one doesn't affect the other
        heap1.collect(); // No roots, so the string in heap1 is collected
        assert_eq!(heap1.live_objects(), 0);
        assert_eq!(heap2.live_objects(), 1); // heap2 unaffected
    }

    #[test]
    fn test_deep_copy_nested_list() {
        let mut heap1 = Heap::new();
        let mut heap2 = Heap::new();

        let s = heap1.alloc_string("nested".to_string());
        let inner = heap1.alloc_list(vec![GcValue::String(s), GcValue::Int(42)]);
        let outer = heap1.alloc_list(vec![GcValue::List(inner), GcValue::Bool(true)]);

        let value = GcValue::List(outer);
        let copied = heap2.deep_copy(&value, &heap1);

        if let GcValue::List(new_outer) = copied {
            let outer_list = heap2.get_list(new_outer).unwrap();
            assert_eq!(outer_list.items.len(), 2);

            if let GcValue::List(new_inner) = &outer_list.items[0] {
                let inner_list = heap2.get_list(*new_inner).unwrap();
                assert_eq!(inner_list.items.len(), 2);

                if let GcValue::String(new_s) = &inner_list.items[0] {
                    assert_eq!(heap2.get_string(*new_s).unwrap().data, "nested");
                }
            }
        } else {
            panic!("Expected list");
        }

        // Heap2 should have: outer list, inner list, string = 3 objects
        assert_eq!(heap2.live_objects(), 3);
    }

    #[test]
    fn test_deep_copy_record() {
        let mut heap1 = Heap::new();
        let mut heap2 = Heap::new();

        let name = heap1.alloc_string("Bob".to_string());
        let person = heap1.alloc_record(
            "Person".to_string(),
            vec!["name".to_string(), "age".to_string()],
            vec![GcValue::String(name), GcValue::Int(25)],
            vec![false, false],
        );

        let value = GcValue::Record(person);
        let copied = heap2.deep_copy(&value, &heap1);

        if let GcValue::Record(new_person) = copied {
            let rec = heap2.get_record(new_person).unwrap();
            assert_eq!(rec.type_name, "Person");
            assert_eq!(rec.field_names, vec!["name", "age"]);

            if let GcValue::String(name_ptr) = &rec.fields[0] {
                assert_eq!(heap2.get_string(*name_ptr).unwrap().data, "Bob");
            }
        } else {
            panic!("Expected record");
        }
    }

    #[test]
    fn test_deep_copy_closure() {
        use crate::value::{FunctionValue, Chunk};
        use std::rc::Rc;

        let mut heap1 = Heap::new();
        let mut heap2 = Heap::new();

        // Create a test function for the closure
        let func = Rc::new(FunctionValue {
            name: "test_func".to_string(),
            arity: 2,
            param_names: vec!["x".to_string(), "y".to_string()],
            code: Rc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
        });

        let captured = heap1.alloc_string("captured_value".to_string());
        let closure = heap1.alloc_closure(
            func.clone(),
            vec![GcValue::String(captured), GcValue::Int(10)],
            vec!["x".to_string(), "y".to_string()],
        );

        let value = GcValue::Closure(closure);
        let copied = heap2.deep_copy(&value, &heap1);

        if let GcValue::Closure(new_closure) = copied {
            let clo = heap2.get_closure(new_closure).unwrap();
            assert_eq!(clo.function.name, "test_func");
            assert_eq!(clo.function.arity, 2);
            assert_eq!(clo.captures.len(), 2);

            if let GcValue::String(cap_ptr) = &clo.captures[0] {
                assert_eq!(heap2.get_string(*cap_ptr).unwrap().data, "captured_value");
            }
        } else {
            panic!("Expected closure");
        }
    }

    // ============================================================
    // Threshold and Auto-Collection Tests
    // ============================================================

    #[test]
    fn test_should_collect_threshold() {
        let config = GcConfig {
            gc_threshold: 100, // Very low threshold for testing
            ..Default::default()
        };
        let mut heap = Heap::with_config(config);

        // Initially should not need collection
        assert!(!heap.should_collect());

        // Allocate until threshold exceeded
        for _ in 0..50 {
            let _ptr = heap.alloc_string("test string here".to_string());
        }

        assert!(heap.should_collect());
    }

    #[test]
    fn test_maybe_collect() {
        let config = GcConfig {
            gc_threshold: 100,
            ..Default::default()
        };
        let mut heap = Heap::with_config(config);

        // Allocate a lot (exceeds threshold)
        for _ in 0..50 {
            let _ptr = heap.alloc_string("garbage".to_string());
        }

        let before = heap.stats.collections;
        heap.maybe_collect();
        let after = heap.stats.collections;

        assert_eq!(after, before + 1);
    }

    // ============================================================
    // Edge Cases
    // ============================================================

    #[test]
    fn test_gc_empty_heap() {
        let mut heap = Heap::new();
        heap.collect(); // Should not crash
        assert_eq!(heap.stats.collections, 1);
        assert_eq!(heap.stats.total_freed, 0);
    }

    #[test]
    fn test_gc_all_roots() {
        let mut heap = Heap::new();

        let ptrs: Vec<_> = (0..10)
            .map(|i| heap.alloc_string(format!("str{}", i)))
            .collect();

        for ptr in &ptrs {
            heap.add_root(ptr.as_raw());
        }

        heap.collect();

        assert_eq!(heap.live_objects(), 10);
        assert_eq!(heap.stats.total_freed, 0);
    }

    #[test]
    fn test_gc_no_roots() {
        let mut heap = Heap::new();

        for i in 0..100 {
            let _ptr = heap.alloc_string(format!("str{}", i));
        }

        heap.collect();

        assert_eq!(heap.live_objects(), 0);
        assert_eq!(heap.stats.total_freed, 100);
    }

    #[test]
    fn test_multiple_collections() {
        let mut heap = Heap::new();

        for round in 0..5 {
            // Allocate some objects
            let ptrs: Vec<_> = (0..20)
                .map(|i| heap.alloc_string(format!("round{}_{}", round, i)))
                .collect();

            // Root only half
            for ptr in ptrs.iter().take(10) {
                heap.add_root(ptr.as_raw());
            }

            heap.collect();

            // Clear roots for next round
            heap.clear_roots();
        }

        // After all rounds with no roots, should be empty
        heap.collect();
        assert_eq!(heap.live_objects(), 0);
    }

    #[test]
    fn test_gc_value_is_immediate() {
        assert!(GcValue::Unit.is_immediate());
        assert!(GcValue::Bool(true).is_immediate());
        assert!(GcValue::Int(42).is_immediate());
        assert!(GcValue::Float(3.14).is_immediate());
        assert!(GcValue::Char('x').is_immediate());
        assert!(GcValue::Pid(1).is_immediate());
        assert!(GcValue::Ref(2).is_immediate());

        let ptr: GcPtr<GcString> = GcPtr::from_raw(0);
        assert!(!GcValue::String(ptr).is_immediate());
    }

    #[test]
    fn test_gc_value_gc_pointers() {
        assert!(GcValue::Int(42).gc_pointers().is_empty());

        let ptr: GcPtr<GcString> = GcPtr::from_raw(5);
        let ptrs = GcValue::String(ptr).gc_pointers();
        assert_eq!(ptrs, vec![5]);
    }

    #[test]
    fn test_heap_data_size_estimates() {
        let string_data = HeapData::String(GcString {
            data: "hello world".to_string(),
        });
        assert!(string_data.estimate_size() > 10);

        let list_data = HeapData::List(GcList {
            items: vec![GcValue::Int(1), GcValue::Int(2), GcValue::Int(3)],
        });
        assert!(list_data.estimate_size() > 0);
    }

    // ============================================================
    // Stress Tests
    // ============================================================

    #[test]
    fn test_stress_many_allocations() {
        let mut heap = Heap::new();

        // Allocate 10,000 objects
        let ptrs: Vec<_> = (0..10_000)
            .map(|i| heap.alloc_string(format!("string number {}", i)))
            .collect();

        assert_eq!(heap.live_objects(), 10_000);

        // Root every 10th object
        for ptr in ptrs.iter().step_by(10) {
            heap.add_root(ptr.as_raw());
        }

        heap.collect();

        assert_eq!(heap.live_objects(), 1000);
    }

    #[test]
    fn test_stress_alternating_alloc_collect() {
        let config = GcConfig {
            gc_threshold: 1000,
            ..Default::default()
        };
        let mut heap = Heap::with_config(config);

        let mut keeper = None;

        for i in 0..100 {
            // Allocate batch
            for j in 0..50 {
                let ptr = heap.alloc_string(format!("temp_{}_{}", i, j));
                if j == 25 {
                    keeper = Some(ptr);
                }
            }

            // Keep one object rooted
            if let Some(k) = keeper {
                heap.clear_roots();
                heap.add_root(k.as_raw());
            }

            heap.maybe_collect();
        }

        // Should have at most a few objects (the kept ones)
        assert!(heap.live_objects() <= 50);
    }

    #[test]
    fn test_stress_deep_nesting() {
        let mut heap = Heap::new();

        // Create a very deep list: [[[[...]]]]
        let mut current = heap.alloc_list(vec![GcValue::Int(42)]);

        for _ in 0..100 {
            current = heap.alloc_list(vec![GcValue::List(current)]);
        }

        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(current.as_raw());
        heap.collect();

        // Should have 101 lists (100 wrappers + innermost) + no garbage
        assert_eq!(heap.live_objects(), 101);
    }

    #[test]
    fn test_stress_wide_structure() {
        let mut heap = Heap::new();

        // Create a list with 1000 elements
        let items: Vec<GcValue> = (0..1000)
            .map(|i| {
                let s = heap.alloc_string(format!("item{}", i));
                GcValue::String(s)
            })
            .collect();

        let list = heap.alloc_list(items);

        // Add some garbage
        for i in 0..500 {
            let _g = heap.alloc_string(format!("garbage{}", i));
        }

        heap.add_root(list.as_raw());
        heap.collect();

        // 1000 strings + 1 list
        assert_eq!(heap.live_objects(), 1001);
    }

    // ============================================================
    // gc_values_equal Tests
    // ============================================================

    #[test]
    fn test_gc_values_equal_immediate() {
        let heap = Heap::new();

        // Unit
        assert!(heap.gc_values_equal(&GcValue::Unit, &GcValue::Unit));

        // Bool
        assert!(heap.gc_values_equal(&GcValue::Bool(true), &GcValue::Bool(true)));
        assert!(heap.gc_values_equal(&GcValue::Bool(false), &GcValue::Bool(false)));
        assert!(!heap.gc_values_equal(&GcValue::Bool(true), &GcValue::Bool(false)));

        // Int
        assert!(heap.gc_values_equal(&GcValue::Int(42), &GcValue::Int(42)));
        assert!(!heap.gc_values_equal(&GcValue::Int(42), &GcValue::Int(43)));

        // Float
        assert!(heap.gc_values_equal(&GcValue::Float(3.14), &GcValue::Float(3.14)));
        assert!(!heap.gc_values_equal(&GcValue::Float(3.14), &GcValue::Float(2.71)));

        // Char
        assert!(heap.gc_values_equal(&GcValue::Char('a'), &GcValue::Char('a')));
        assert!(!heap.gc_values_equal(&GcValue::Char('a'), &GcValue::Char('b')));

        // Pid
        assert!(heap.gc_values_equal(&GcValue::Pid(1), &GcValue::Pid(1)));
        assert!(!heap.gc_values_equal(&GcValue::Pid(1), &GcValue::Pid(2)));
    }

    #[test]
    fn test_gc_values_equal_different_types() {
        let heap = Heap::new();

        // Different types should not be equal
        assert!(!heap.gc_values_equal(&GcValue::Int(42), &GcValue::Bool(true)));
        assert!(!heap.gc_values_equal(&GcValue::Unit, &GcValue::Bool(false)));
        assert!(!heap.gc_values_equal(&GcValue::Float(42.0), &GcValue::Int(42)));
    }

    #[test]
    fn test_gc_values_equal_string_same_pointer() {
        let mut heap = Heap::new();

        let ptr = heap.alloc_string("hello".to_string());
        let v1 = GcValue::String(ptr);
        let v2 = GcValue::String(ptr);

        // Same pointer should be equal
        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_string_different_pointers_same_content() {
        let mut heap = Heap::new();

        // Allocate same string twice - different pointers
        let ptr1 = heap.alloc_string("hello".to_string());
        let ptr2 = heap.alloc_string("hello".to_string());

        // Pointers should be different
        assert_ne!(ptr1, ptr2);

        let v1 = GcValue::String(ptr1);
        let v2 = GcValue::String(ptr2);

        // But values should be equal (content comparison)
        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_string_different_content() {
        let mut heap = Heap::new();

        let ptr1 = heap.alloc_string("hello".to_string());
        let ptr2 = heap.alloc_string("world".to_string());

        let v1 = GcValue::String(ptr1);
        let v2 = GcValue::String(ptr2);

        // Different content should not be equal
        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_list_same_content() {
        let mut heap = Heap::new();

        // Create two lists with same content but different allocations
        let list1 = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(2), GcValue::Int(3)]);
        let list2 = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(2), GcValue::Int(3)]);

        // Different pointers
        assert_ne!(list1, list2);

        let v1 = GcValue::List(list1);
        let v2 = GcValue::List(list2);

        // But same content - should be equal
        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_list_different_content() {
        let mut heap = Heap::new();

        let list1 = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(2)]);
        let list2 = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(3)]);

        let v1 = GcValue::List(list1);
        let v2 = GcValue::List(list2);

        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_list_different_length() {
        let mut heap = Heap::new();

        let list1 = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(2)]);
        let list2 = heap.alloc_list(vec![GcValue::Int(1), GcValue::Int(2), GcValue::Int(3)]);

        let v1 = GcValue::List(list1);
        let v2 = GcValue::List(list2);

        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_nested_list_with_strings() {
        let mut heap = Heap::new();

        // Create nested lists with strings - allocate twice
        let s1a = heap.alloc_string("hello".to_string());
        let s1b = heap.alloc_string("world".to_string());
        let inner1 = heap.alloc_list(vec![GcValue::String(s1a), GcValue::String(s1b)]);
        let outer1 = heap.alloc_list(vec![GcValue::List(inner1), GcValue::Int(42)]);

        let s2a = heap.alloc_string("hello".to_string());
        let s2b = heap.alloc_string("world".to_string());
        let inner2 = heap.alloc_list(vec![GcValue::String(s2a), GcValue::String(s2b)]);
        let outer2 = heap.alloc_list(vec![GcValue::List(inner2), GcValue::Int(42)]);

        let v1 = GcValue::List(outer1);
        let v2 = GcValue::List(outer2);

        // Should be equal despite different allocations
        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_tuple() {
        let mut heap = Heap::new();

        let s1 = heap.alloc_string("test".to_string());
        let tuple1 = heap.alloc_tuple(vec![GcValue::Int(1), GcValue::String(s1), GcValue::Bool(true)]);

        let s2 = heap.alloc_string("test".to_string());
        let tuple2 = heap.alloc_tuple(vec![GcValue::Int(1), GcValue::String(s2), GcValue::Bool(true)]);

        let v1 = GcValue::Tuple(tuple1);
        let v2 = GcValue::Tuple(tuple2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_record() {
        let mut heap = Heap::new();

        let name1 = heap.alloc_string("Alice".to_string());
        let rec1 = heap.alloc_record(
            "Person".to_string(),
            vec!["name".to_string(), "age".to_string()],
            vec![GcValue::String(name1), GcValue::Int(30)],
            vec![false, false],
        );

        let name2 = heap.alloc_string("Alice".to_string());
        let rec2 = heap.alloc_record(
            "Person".to_string(),
            vec!["name".to_string(), "age".to_string()],
            vec![GcValue::String(name2), GcValue::Int(30)],
            vec![false, false],
        );

        let v1 = GcValue::Record(rec1);
        let v2 = GcValue::Record(rec2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_record_different_type() {
        let mut heap = Heap::new();

        let rec1 = heap.alloc_record(
            "Person".to_string(),
            vec!["name".to_string()],
            vec![GcValue::Int(1)],
            vec![false],
        );

        let rec2 = heap.alloc_record(
            "User".to_string(),
            vec!["name".to_string()],
            vec![GcValue::Int(1)],
            vec![false],
        );

        let v1 = GcValue::Record(rec1);
        let v2 = GcValue::Record(rec2);

        // Different type names - not equal
        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_variant() {
        let mut heap = Heap::new();

        let var1 = heap.alloc_variant("Option".to_string(), "Some".to_string(), vec![GcValue::Int(42)]);
        let var2 = heap.alloc_variant("Option".to_string(), "Some".to_string(), vec![GcValue::Int(42)]);

        let v1 = GcValue::Variant(var1);
        let v2 = GcValue::Variant(var2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_variant_different_constructor() {
        let mut heap = Heap::new();

        let var1 = heap.alloc_variant("Option".to_string(), "Some".to_string(), vec![GcValue::Int(42)]);
        let var2 = heap.alloc_variant("Option".to_string(), "None".to_string(), vec![]);

        let v1 = GcValue::Variant(var1);
        let v2 = GcValue::Variant(var2);

        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_array() {
        let mut heap = Heap::new();

        let arr1 = heap.alloc_array(vec![GcValue::Float(1.0), GcValue::Float(2.0)]);
        let arr2 = heap.alloc_array(vec![GcValue::Float(1.0), GcValue::Float(2.0)]);

        let v1 = GcValue::Array(arr1);
        let v2 = GcValue::Array(arr2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_empty_list() {
        let mut heap = Heap::new();

        let list1 = heap.alloc_list(vec![]);
        let list2 = heap.alloc_list(vec![]);

        let v1 = GcValue::List(list1);
        let v2 = GcValue::List(list2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }
}
