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
use imbl::HashMap as ImblHashMap;
use imbl::HashSet as ImblHashSet;
use imbl::Vector as ImblVector;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use nostos_extension::GcNativeHandle;
use num_bigint::BigInt;

use crate::shared_types::{SharedMap, SharedMapKey, SharedMapValue};

/// Cached inline operation for closures - avoids pattern matching on every call
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InlineOp {
    #[default]
    None = 0,
    AddInt = 1,
    SubInt = 2,
    MulInt = 3,
}

impl InlineOp {
    /// Compute InlineOp from function code.
    /// Returns Some(op) if the function is a simple 2-arg binary op like (a, b) => a + b
    #[inline]
    pub fn from_function(func: &FunctionValue) -> InlineOp {
        use crate::value::Instruction;
        let instrs = &func.code.code;
        // Check for pattern: BinaryOp(dst, 0, 1); Return(dst)
        if instrs.len() == 2 {
            if let Instruction::Return(ret_reg) = &instrs[1] {
                match &instrs[0] {
                    Instruction::AddInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                        return InlineOp::AddInt;
                    }
                    Instruction::SubInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                        return InlineOp::SubInt;
                    }
                    Instruction::MulInt(op_dst, a, b) if *op_dst == *ret_reg && *a == 0 && *b == 1 => {
                        return InlineOp::MulInt;
                    }
                    _ => {}
                }
            }
        }
        InlineOp::None
    }
}
use rust_decimal::Decimal;

use crate::value::{
    ClosureValue, FunctionValue, MapKey, Pid, ReactiveRecordValue, RecordValue, RuntimeError, TypeValue, Value, VariantValue,
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
    _marker: PhantomData<T>,
}

// Manually implement Copy and Clone to avoid T: Copy bounds
impl<T> Copy for GcPtr<T> {}

impl<T> Clone for GcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// GcPtr is just a u32 index, safe to send between threads.
// The actual data lives in a Heap that is owned by a single process.
unsafe impl<T: Send> Send for GcPtr<T> {}
unsafe impl<T: Sync> Sync for GcPtr<T> {}

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

/// A GC-managed mutable string buffer for efficient string building.
/// Unlike GcString, this is mutable and designed for appending.
#[derive(Clone, Debug)]
pub struct GcBuffer {
    pub data: std::cell::RefCell<String>,
}

impl GcBuffer {
    pub fn new() -> Self {
        GcBuffer {
            data: std::cell::RefCell::new(String::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        GcBuffer {
            data: std::cell::RefCell::new(String::with_capacity(capacity)),
        }
    }

    pub fn append(&self, s: &str) {
        self.data.borrow_mut().push_str(s);
    }

    pub fn to_string(&self) -> String {
        self.data.borrow().clone()
    }
}

/// A GC-managed list (immutable, persistent).
/// Uses imbl::Vector (RRB tree) for O(log n) cons operations.
/// Tail operations use offset tracking for O(1) performance.
#[derive(Clone, Debug)]
pub struct GcList {
    pub data: ImblVector<GcValue>,
    /// Offset into data - allows O(1) tail by just incrementing offset
    pub offset: usize,
}

impl PartialEq for GcList {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl GcList {
    /// Create a new empty list
    #[inline]
    pub fn new() -> Self {
        GcList { data: ImblVector::new(), offset: 0 }
    }

    /// Create from a Vec (consumes the Vec)
    #[inline]
    pub fn from_vec(v: Vec<GcValue>) -> Self {
        GcList { data: v.into_iter().collect(), offset: 0 }
    }

    /// Get items as a Vec (for compatibility - allocates)
    #[inline]
    pub fn items(&self) -> Vec<GcValue> {
        self.data.iter().skip(self.offset).cloned().collect()
    }

    /// Iterate over items without allocating
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &GcValue> {
        self.data.iter().skip(self.offset)
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offset >= self.data.len()
    }

    /// Get length
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    /// Get head element - O(log n) for imbl
    #[inline]
    pub fn head(&self) -> Option<&GcValue> {
        self.data.get(self.offset)
    }

    /// Get head assuming non-empty (caller must ensure)
    #[inline]
    pub fn head_unchecked(&self) -> &GcValue {
        self.data.get(self.offset).unwrap()
    }

    /// Create a tail view - O(1)! Just increment offset.
    #[inline]
    pub fn tail(&self) -> GcList {
        if self.is_empty() {
            GcList::new()
        } else {
            GcList { data: self.data.clone(), offset: self.offset + 1 }
        }
    }

    /// Get tail assuming non-empty (caller must ensure)
    #[inline]
    pub fn tail_unchecked(&self) -> GcList {
        GcList { data: self.data.clone(), offset: self.offset + 1 }
    }

    /// Cons: prepend an element - O(log n) with structural sharing
    #[inline]
    pub fn cons(&self, head: GcValue) -> GcList {
        // Use skip() to create a proper subvector when offset > 0
        // skip() is O(log n) and creates structural sharing
        let mut new_data = if self.offset > 0 {
            self.data.skip(self.offset)
        } else {
            self.data.clone()
        };
        new_data.push_front(head);
        GcList { data: new_data, offset: 0 }
    }

    /// Get element at index - O(log n)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&GcValue> {
        self.data.get(self.offset + index)
    }
}

/// A GC-managed array (mutable, heterogeneous).
#[derive(Clone, Debug)]
pub struct GcArray {
    pub items: Vec<GcValue>,
}

/// A GC-managed typed array of i64 (mutable, homogeneous, JIT-optimized).
#[derive(Clone, Debug)]
pub struct GcInt64Array {
    pub items: Vec<i64>,
}

/// A GC-managed typed array of f64 (mutable, homogeneous, JIT-optimized).
#[derive(Clone, Debug)]
pub struct GcFloat64Array {
    pub items: Vec<f64>,
}

/// A GC-managed typed array of f32 (mutable, homogeneous, for vectors).
#[derive(Clone, Debug)]
pub struct GcFloat32Array {
    pub items: Vec<f32>,
}

/// A specialized immutable list of i64 for fast integer operations.
/// Uses imbl::Vector<i64> for O(log n) cons/tail operations.
/// Avoids GcValue boxing overhead for integer lists.
#[derive(Clone)]
pub struct GcInt64List {
    data: ImblVector<i64>,
    /// Offset into data - allows O(1) tail by just incrementing offset
    offset: usize,
}

impl std::fmt::Debug for GcInt64List {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Int64List[{}]", self.len())
    }
}

impl GcInt64List {
    #[inline]
    pub fn new() -> Self {
        GcInt64List { data: ImblVector::new(), offset: 0 }
    }

    #[inline]
    pub fn from_vec(v: Vec<i64>) -> Self {
        GcInt64List { data: v.into_iter().collect(), offset: 0 }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offset >= self.data.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    #[inline]
    pub fn head(&self) -> Option<i64> {
        self.data.get(self.offset).copied()
    }

    /// Get head assuming non-empty (caller must ensure non-empty)
    #[inline]
    pub fn head_unchecked(&self) -> i64 {
        // Caller ensures list is non-empty, so unwrap is safe
        *self.data.get(self.offset).unwrap()
    }

    #[inline]
    pub fn tail(&self) -> GcInt64List {
        if self.is_empty() {
            GcInt64List::new()
        } else {
            GcInt64List { data: self.data.clone(), offset: self.offset + 1 }
        }
    }

    /// Get tail without bounds check (caller must ensure non-empty)
    #[inline]
    pub fn tail_unchecked(&self) -> GcInt64List {
        GcInt64List { data: self.data.clone(), offset: self.offset + 1 }
    }

    /// O(log n) cons using persistent data structure
    #[inline]
    pub fn cons(&self, head: i64) -> GcInt64List {
        // Use skip() to create a proper subvector when offset > 0
        // skip() is O(log n) and creates structural sharing
        let mut new_data = if self.offset > 0 {
            self.data.skip(self.offset)
        } else {
            self.data.clone()
        };
        new_data.push_front(head);
        GcInt64List { data: new_data, offset: 0 }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.data.iter().skip(self.offset).copied()
    }

    #[inline]
    pub fn sum(&self) -> i64 {
        self.data.iter().skip(self.offset).sum()
    }

    #[inline]
    pub fn product(&self) -> i64 {
        self.data.iter().skip(self.offset).product()
    }

    /// Get element at index - O(log n)
    #[inline]
    pub fn get(&self, index: usize) -> Option<i64> {
        self.data.get(self.offset + index).copied()
    }
}

/// A GC-managed tuple.
#[derive(Clone, Debug)]
pub struct GcTuple {
    pub items: Vec<GcValue>,
}

/// A GC-managed map using persistent data structure for O(log n) updates.
///
/// TODO: Shared heap optimization for cross-process zero-copy sharing.
/// imbl's HashMap is Send+Sync and uses Arc internally. If we introduce a shared
/// heap for immutable data, maps containing only immediate values could be shared
/// across processes without deep copying. This would benefit workloads passing
/// large maps of primitives between processes.
#[derive(Clone, Debug)]
pub struct GcMap {
    pub entries: ImblHashMap<GcMapKey, GcValue>,
}

/// A GC-managed set using persistent data structure for O(log n) updates.
///
/// TODO: Same shared heap optimization applies to sets.
#[derive(Clone, Debug)]
pub struct GcSet {
    pub items: ImblHashSet<GcMapKey>,
}

/// Keys for GC maps/sets - supports primitives, strings, records, and variants.
#[derive(Clone, Debug)]
pub enum GcMapKey {
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
    // String - stored inline for proper equality/hashing
    String(String),
    // Record as key - fields must all be hashable
    Record {
        type_name: String,
        field_names: Vec<String>,
        fields: Vec<GcMapKey>,
    },
    // Variant as key - fields must all be hashable
    Variant {
        type_name: String,
        constructor: String,
        fields: Vec<GcMapKey>,
    },
}

// Manual implementation of PartialEq for GcMapKey
impl PartialEq for GcMapKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GcMapKey::Unit, GcMapKey::Unit) => true,
            (GcMapKey::Bool(a), GcMapKey::Bool(b)) => a == b,
            (GcMapKey::Char(a), GcMapKey::Char(b)) => a == b,
            (GcMapKey::Int8(a), GcMapKey::Int8(b)) => a == b,
            (GcMapKey::Int16(a), GcMapKey::Int16(b)) => a == b,
            (GcMapKey::Int32(a), GcMapKey::Int32(b)) => a == b,
            (GcMapKey::Int64(a), GcMapKey::Int64(b)) => a == b,
            (GcMapKey::UInt8(a), GcMapKey::UInt8(b)) => a == b,
            (GcMapKey::UInt16(a), GcMapKey::UInt16(b)) => a == b,
            (GcMapKey::UInt32(a), GcMapKey::UInt32(b)) => a == b,
            (GcMapKey::UInt64(a), GcMapKey::UInt64(b)) => a == b,
            (GcMapKey::String(a), GcMapKey::String(b)) => a == b,
            (
                GcMapKey::Record { type_name: tn1, field_names: fn1, fields: f1 },
                GcMapKey::Record { type_name: tn2, field_names: fn2, fields: f2 },
            ) => tn1 == tn2 && fn1 == fn2 && f1 == f2,
            (
                GcMapKey::Variant { type_name: tn1, constructor: c1, fields: f1 },
                GcMapKey::Variant { type_name: tn2, constructor: c2, fields: f2 },
            ) => tn1 == tn2 && c1 == c2 && f1 == f2,
            _ => false,
        }
    }
}

impl Eq for GcMapKey {}

// Manual implementation of Hash for GcMapKey
impl std::hash::Hash for GcMapKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            GcMapKey::Unit => {}
            GcMapKey::Bool(b) => b.hash(state),
            GcMapKey::Char(c) => c.hash(state),
            GcMapKey::Int8(n) => n.hash(state),
            GcMapKey::Int16(n) => n.hash(state),
            GcMapKey::Int32(n) => n.hash(state),
            GcMapKey::Int64(n) => n.hash(state),
            GcMapKey::UInt8(n) => n.hash(state),
            GcMapKey::UInt16(n) => n.hash(state),
            GcMapKey::UInt32(n) => n.hash(state),
            GcMapKey::UInt64(n) => n.hash(state),
            GcMapKey::String(s) => s.hash(state),
            GcMapKey::Record { type_name, field_names, fields } => {
                type_name.hash(state);
                for name in field_names {
                    name.hash(state);
                }
                for field in fields {
                    field.hash(state);
                }
            }
            GcMapKey::Variant { type_name, constructor, fields } => {
                type_name.hash(state);
                constructor.hash(state);
                for field in fields {
                    field.hash(state);
                }
            }
        }
    }
}

impl GcMapKey {
    /// Convert a GcMapKey back to a GcValue
    pub fn to_gc_value(&self, heap: &mut Heap) -> GcValue {
        match self {
            GcMapKey::Unit => GcValue::Unit,
            GcMapKey::Bool(b) => GcValue::Bool(*b),
            GcMapKey::Char(c) => GcValue::Char(*c),
            GcMapKey::Int8(n) => GcValue::Int8(*n),
            GcMapKey::Int16(n) => GcValue::Int16(*n),
            GcMapKey::Int32(n) => GcValue::Int32(*n),
            GcMapKey::Int64(n) => GcValue::Int64(*n),
            GcMapKey::UInt8(n) => GcValue::UInt8(*n),
            GcMapKey::UInt16(n) => GcValue::UInt16(*n),
            GcMapKey::UInt32(n) => GcValue::UInt32(*n),
            GcMapKey::UInt64(n) => GcValue::UInt64(*n),
            GcMapKey::String(s) => GcValue::String(heap.alloc_string(s.clone())),
            GcMapKey::Record { type_name, field_names, fields } => {
                let gc_fields: Vec<GcValue> = fields.iter()
                    .map(|f| f.to_gc_value(heap))
                    .collect();
                let mutable_fields = vec![false; field_names.len()];
                GcValue::Record(heap.alloc_record(
                    type_name.clone(),
                    field_names.clone(),
                    gc_fields,
                    mutable_fields,
                ))
            }
            GcMapKey::Variant { type_name, constructor, fields } => {
                let gc_fields: Vec<GcValue> = fields.iter()
                    .map(|f| f.to_gc_value(heap))
                    .collect();
                GcValue::Variant(heap.alloc_variant(
                    Arc::new(type_name.clone()),
                    Arc::new(constructor.clone()),
                    gc_fields,
                ))
            }
        }
    }

    /// Convert to a SharedMapKey for cross-thread sharing.
    pub fn to_shared_key(&self) -> SharedMapKey {
        match self {
            GcMapKey::Unit => SharedMapKey::Unit,
            GcMapKey::Bool(b) => SharedMapKey::Bool(*b),
            GcMapKey::Char(c) => SharedMapKey::Char(*c),
            GcMapKey::Int8(n) => SharedMapKey::Int8(*n),
            GcMapKey::Int16(n) => SharedMapKey::Int16(*n),
            GcMapKey::Int32(n) => SharedMapKey::Int32(*n),
            GcMapKey::Int64(n) => SharedMapKey::Int64(*n),
            GcMapKey::UInt8(n) => SharedMapKey::UInt8(*n),
            GcMapKey::UInt16(n) => SharedMapKey::UInt16(*n),
            GcMapKey::UInt32(n) => SharedMapKey::UInt32(*n),
            GcMapKey::UInt64(n) => SharedMapKey::UInt64(*n),
            GcMapKey::String(s) => SharedMapKey::String(s.clone()),
            GcMapKey::Record { type_name, field_names, fields } => {
                SharedMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|f| f.to_shared_key()).collect(),
                }
            }
            GcMapKey::Variant { type_name, constructor, fields } => {
                SharedMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|f| f.to_shared_key()).collect(),
                }
            }
        }
    }

    /// Convert from a SharedMapKey.
    pub fn from_shared_key(key: &SharedMapKey) -> Self {
        match key {
            SharedMapKey::Unit => GcMapKey::Unit,
            SharedMapKey::Bool(b) => GcMapKey::Bool(*b),
            SharedMapKey::Char(c) => GcMapKey::Char(*c),
            SharedMapKey::Int8(n) => GcMapKey::Int8(*n),
            SharedMapKey::Int16(n) => GcMapKey::Int16(*n),
            SharedMapKey::Int32(n) => GcMapKey::Int32(*n),
            SharedMapKey::Int64(n) => GcMapKey::Int64(*n),
            SharedMapKey::UInt8(n) => GcMapKey::UInt8(*n),
            SharedMapKey::UInt16(n) => GcMapKey::UInt16(*n),
            SharedMapKey::UInt32(n) => GcMapKey::UInt32(*n),
            SharedMapKey::UInt64(n) => GcMapKey::UInt64(*n),
            SharedMapKey::String(s) => GcMapKey::String(s.clone()),
            SharedMapKey::Record { type_name, field_names, fields } => {
                GcMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|f| GcMapKey::from_shared_key(f)).collect(),
                }
            }
            SharedMapKey::Variant { type_name, constructor, fields } => {
                GcMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|f| GcMapKey::from_shared_key(f)).collect(),
                }
            }
        }
    }
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
    pub type_name: Arc<String>,
    pub constructor: Arc<String>,
    pub fields: Vec<GcValue>,
    /// Cached discriminant for fast pattern matching (hash of constructor name)
    pub discriminant: u16,
}

/// Compute discriminant from constructor name (fast hash for pattern matching)
#[inline]
pub fn constructor_discriminant(name: &str) -> u16 {
    // FNV-1a hash truncated to u16 - fast and good distribution
    let mut hash: u32 = 2166136261;
    for byte in name.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash as u16
}

/// A GC-managed BigInt.
#[derive(Clone, Debug)]
pub struct GcBigInt {
    pub value: BigInt,
}

/// A GC-managed closure.
#[derive(Clone, Debug)]
pub struct GcClosure {
    pub function: Arc<FunctionValue>, // The function being closed over
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

    // Floating point
    Float32(f32),
    Float64(f64),

    // Decimal (fixed-point)
    Decimal(Decimal),

    // Heap-allocated values (GC-managed)
    String(GcPtr<GcString>),
    /// Mutable string buffer for efficient string building (e.g., HTML rendering)
    Buffer(GcPtr<GcBuffer>),
    /// Lists are stored inline (not heap-allocated) for O(1) tail without allocation.
    /// The underlying data is in an Arc<Vec>, so cloning is cheap (reference count bump).
    /// GC tracing goes through the Arc to find referenced heap objects.
    List(GcList),
    Array(GcPtr<GcArray>),
    // Typed arrays for JIT optimization (contiguous memory, no tag checking)
    Int64Array(GcPtr<GcInt64Array>),
    Float64Array(GcPtr<GcFloat64Array>),
    Float32Array(GcPtr<GcFloat32Array>),
    /// Specialized list of i64 - avoids GcValue boxing overhead
    Int64List(GcInt64List),
    Tuple(GcPtr<GcTuple>),
    Map(GcPtr<GcMap>),
    /// A shared map from an MVar - Arc-wrapped for O(1) sharing across threads.
    /// Values are converted to GcValue lazily when accessed.
    SharedMap(SharedMap),
    Set(GcPtr<GcSet>),
    Record(GcPtr<GcRecord>),
    /// Reactive record with parent tracking - stored as Arc for shared mutation
    ReactiveRecord(Arc<ReactiveRecordValue>),
    Variant(GcPtr<GcVariant>),
    BigInt(GcPtr<GcBigInt>),
    Closure(GcPtr<GcClosure>, InlineOp),

    // Callable values (Arc-managed, not GC'd - code doesn't need collection)
    Function(Arc<FunctionValue>),
    NativeFunction(Arc<GcNativeFn>),

    // Special values
    Pid(u64),
    Ref(u64),
    Type(Arc<TypeValue>),
    Pointer(usize),

    // Native handle with GC-managed cleanup.
    // Stored directly as Arc (not in heap) - Drop triggers when GcValue is dropped.
    NativeHandle(Arc<GcNativeHandle>),
}

impl Default for GcValue {
    fn default() -> Self {
        GcValue::Unit
    }
}

// GcValue is safe to Send between threads:
// - Immediate values (primitives) are inherently Send
// - GcPtr is just a u32 index, valid only within a specific Heap
// - Each process has its own Heap, so GcValues don't escape their owning process
// - When sending between processes, values are converted to ThreadSafeValue first
unsafe impl Send for GcValue {}
unsafe impl Sync for GcValue {}

// All GC container types are Send+Sync because:
// - They only contain primitives, Strings, Arcs, or other GC types that are Send+Sync
// - Each process has its own Heap, so these values don't escape their owning process
unsafe impl Send for GcString {}
unsafe impl Sync for GcString {}
unsafe impl Send for GcList {}
unsafe impl Sync for GcList {}
unsafe impl Send for GcArray {}
unsafe impl Sync for GcArray {}
unsafe impl Send for GcInt64Array {}
unsafe impl Sync for GcInt64Array {}
unsafe impl Send for GcFloat64Array {}
unsafe impl Sync for GcFloat64Array {}
unsafe impl Send for GcFloat32Array {}
unsafe impl Sync for GcFloat32Array {}
unsafe impl Send for GcTuple {}
unsafe impl Sync for GcTuple {}
unsafe impl Send for GcMap {}
unsafe impl Sync for GcMap {}
unsafe impl Send for GcSet {}
unsafe impl Sync for GcSet {}
unsafe impl Send for GcRecord {}
unsafe impl Sync for GcRecord {}
unsafe impl Send for GcVariant {}
unsafe impl Sync for GcVariant {}
unsafe impl Send for GcBigInt {}
unsafe impl Sync for GcBigInt {}
unsafe impl Send for GcClosure {}
unsafe impl Sync for GcClosure {}
unsafe impl Send for GcMapKey {}
unsafe impl Sync for GcMapKey {}
unsafe impl Send for GcNativeFn {}
unsafe impl Sync for GcNativeFn {}

impl PartialEq for GcValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GcValue::Unit, GcValue::Unit) => true,
            (GcValue::Bool(a), GcValue::Bool(b)) => a == b,
            (GcValue::Char(a), GcValue::Char(b)) => a == b,
            // Signed integers
            (GcValue::Int8(a), GcValue::Int8(b)) => a == b,
            (GcValue::Int16(a), GcValue::Int16(b)) => a == b,
            (GcValue::Int32(a), GcValue::Int32(b)) => a == b,
            (GcValue::Int64(a), GcValue::Int64(b)) => a == b,
            // Unsigned integers
            (GcValue::UInt8(a), GcValue::UInt8(b)) => a == b,
            (GcValue::UInt16(a), GcValue::UInt16(b)) => a == b,
            (GcValue::UInt32(a), GcValue::UInt32(b)) => a == b,
            (GcValue::UInt64(a), GcValue::UInt64(b)) => a == b,
            // Floats
            (GcValue::Float32(a), GcValue::Float32(b)) => a == b,
            (GcValue::Float64(a), GcValue::Float64(b)) => a == b,
            // Decimal
            (GcValue::Decimal(a), GcValue::Decimal(b)) => a == b,
            // Collections
            (GcValue::String(a), GcValue::String(b)) => a == b,
            (GcValue::List(a), GcValue::List(b)) => a == b,
            (GcValue::Array(a), GcValue::Array(b)) => a == b,
            (GcValue::Tuple(a), GcValue::Tuple(b)) => a == b,
            (GcValue::Map(a), GcValue::Map(b)) => a == b,
            (GcValue::Set(a), GcValue::Set(b)) => a == b,
            (GcValue::Record(a), GcValue::Record(b)) => a == b,
            (GcValue::Variant(a), GcValue::Variant(b)) => a == b,
            (GcValue::BigInt(a), GcValue::BigInt(b)) => a == b,
            (GcValue::Closure(a, _), GcValue::Closure(b, _)) => a == b,
            (GcValue::Function(a), GcValue::Function(b)) => Arc::ptr_eq(a, b),
            (GcValue::NativeFunction(a), GcValue::NativeFunction(b)) => Arc::ptr_eq(a, b),
            (GcValue::Pid(a), GcValue::Pid(b)) => a == b,
            (GcValue::Ref(a), GcValue::Ref(b)) => a == b,
            (GcValue::Type(a), GcValue::Type(b)) => Arc::ptr_eq(a, b),
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
            GcValue::Char(c) => write!(f, "Char('{}')", c),
            // Signed integers
            GcValue::Int8(i) => write!(f, "Int8({})", i),
            GcValue::Int16(i) => write!(f, "Int16({})", i),
            GcValue::Int32(i) => write!(f, "Int32({})", i),
            GcValue::Int64(i) => write!(f, "Int64({})", i),
            // Unsigned integers
            GcValue::UInt8(i) => write!(f, "UInt8({})", i),
            GcValue::UInt16(i) => write!(f, "UInt16({})", i),
            GcValue::UInt32(i) => write!(f, "UInt32({})", i),
            GcValue::UInt64(i) => write!(f, "UInt64({})", i),
            // Floats
            GcValue::Float32(fl) => write!(f, "Float32({})", fl),
            GcValue::Float64(fl) => write!(f, "Float64({})", fl),
            // Decimal
            GcValue::Decimal(d) => write!(f, "Decimal({})", d),
            // Collections
            GcValue::String(ptr) => write!(f, "String({:?})", ptr),
            GcValue::List(ptr) => write!(f, "List({:?})", ptr),
            GcValue::Array(ptr) => write!(f, "Array({:?})", ptr),
            GcValue::Int64Array(ptr) => write!(f, "Int64Array({:?})", ptr),
            GcValue::Float64Array(ptr) => write!(f, "Float64Array({:?})", ptr),
            GcValue::Float32Array(ptr) => write!(f, "Float32Array({:?})", ptr),
            GcValue::Tuple(ptr) => write!(f, "Tuple({:?})", ptr),
            GcValue::Map(ptr) => write!(f, "Map({:?})", ptr),
            GcValue::SharedMap(map) => write!(f, "SharedMap({} entries)", map.len()),
            GcValue::Set(ptr) => write!(f, "Set({:?})", ptr),
            GcValue::Record(ptr) => write!(f, "Record({:?})", ptr),
            GcValue::ReactiveRecord(r) => write!(f, "ReactiveRecord({})", r.type_name),
            GcValue::Variant(ptr) => write!(f, "Variant({:?})", ptr),
            GcValue::BigInt(ptr) => write!(f, "BigInt({:?})", ptr),
            GcValue::Closure(ptr, _) => write!(f, "Closure({:?})", ptr),
            GcValue::Function(func) => write!(f, "Function({})", func.name),
            GcValue::NativeFunction(func) => write!(f, "NativeFunction({})", func.name),
            GcValue::Pid(p) => write!(f, "Pid({})", p),
            GcValue::Ref(r) => write!(f, "Ref({})", r),
            GcValue::Type(t) => write!(f, "Type({})", t.name),
            GcValue::Pointer(p) => write!(f, "Pointer(0x{:x})", p),
            GcValue::Int64List(list) => write!(f, "Int64List[{}]", list.len()),
            GcValue::Buffer(_) => write!(f, "Buffer"),
            GcValue::NativeHandle(ptr) => write!(f, "NativeHandle({:?})", ptr),
        }
    }
}

impl GcValue {
    /// Get all GC pointers contained in this value.
    pub fn gc_pointers(&self) -> Vec<RawGcPtr> {
        match self {
            GcValue::Unit
            | GcValue::Bool(_)
            | GcValue::Char(_)
            // Signed integers
            | GcValue::Int8(_)
            | GcValue::Int16(_)
            | GcValue::Int32(_)
            | GcValue::Int64(_)
            // Unsigned integers
            | GcValue::UInt8(_)
            | GcValue::UInt16(_)
            | GcValue::UInt32(_)
            | GcValue::UInt64(_)
            // Floats
            | GcValue::Float32(_)
            | GcValue::Float64(_)
            // Decimal
            | GcValue::Decimal(_)
            // Special
            | GcValue::Pid(_)
            | GcValue::Ref(_)
            | GcValue::Function(_)
            | GcValue::NativeFunction(_)
            | GcValue::Type(_)
            | GcValue::Pointer(_) => vec![],

            // Typed arrays have no GC pointers (contain raw values, not GcValue)
            GcValue::Int64Array(ptr) => vec![ptr.as_raw()],
            GcValue::Float64Array(ptr) => vec![ptr.as_raw()],
            GcValue::Float32Array(ptr) => vec![ptr.as_raw()],

            GcValue::String(ptr) => vec![ptr.as_raw()],
            GcValue::Buffer(ptr) => vec![ptr.as_raw()],
            // Lists are inline, so we trace through all contained values
            GcValue::List(list) => list.items().iter().flat_map(|v| v.gc_pointers()).collect(),
            GcValue::Array(ptr) => vec![ptr.as_raw()],
            GcValue::Tuple(ptr) => vec![ptr.as_raw()],
            GcValue::Map(ptr) => vec![ptr.as_raw()],
            // SharedMap is Arc-managed, no GC pointers
            GcValue::SharedMap(_) => vec![],
            GcValue::Set(ptr) => vec![ptr.as_raw()],
            GcValue::Record(ptr) => vec![ptr.as_raw()],
            // ReactiveRecord is Arc-managed, no GC pointers
            GcValue::ReactiveRecord(_) => vec![],
            GcValue::Variant(ptr) => vec![ptr.as_raw()],
            GcValue::BigInt(ptr) => vec![ptr.as_raw()],
            GcValue::Closure(ptr, _) => vec![ptr.as_raw()],
            // Int64List contains raw i64s, no GC pointers
            GcValue::Int64List(_) => vec![],
            // Native handle stored directly (not in heap), no GC pointers to trace
            GcValue::NativeHandle(_) => vec![],
        }
    }

    /// Check if this value is an immediate (non-heap) value.
    pub fn is_immediate(&self) -> bool {
        matches!(
            self,
            GcValue::Unit
                | GcValue::Bool(_)
                | GcValue::Char(_)
                // Signed integers
                | GcValue::Int8(_)
                | GcValue::Int16(_)
                | GcValue::Int32(_)
                | GcValue::Int64(_)
                // Unsigned integers
                | GcValue::UInt8(_)
                | GcValue::UInt16(_)
                | GcValue::UInt32(_)
                | GcValue::UInt64(_)
                // Floats
                | GcValue::Float32(_)
                | GcValue::Float64(_)
                // Decimal
                | GcValue::Decimal(_)
                // Lists are inline (though they contain refs that need tracing)
                | GcValue::List(_)
                // Special
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
            GcValue::Char(_) => "Char",
            // Signed integers
            GcValue::Int8(_) => "Int8",
            GcValue::Int16(_) => "Int16",
            GcValue::Int32(_) => "Int32",
            GcValue::Int64(_) => "Int64",
            // Unsigned integers
            GcValue::UInt8(_) => "UInt8",
            GcValue::UInt16(_) => "UInt16",
            GcValue::UInt32(_) => "UInt32",
            GcValue::UInt64(_) => "UInt64",
            // Floats
            GcValue::Float32(_) => "Float32",
            GcValue::Float64(_) => "Float64",
            // Decimal
            GcValue::Decimal(_) => "Decimal",
            // Collections
            GcValue::String(_) => "String",
            GcValue::List(_) => "List",
            GcValue::Array(_) => "Array",
            GcValue::Int64Array(_) => "Int64Array",
            GcValue::Float64Array(_) => "Float64Array",
            GcValue::Float32Array(_) => "Float32Array",
            GcValue::Tuple(_) => "Tuple",
            GcValue::Map(_) => "Map",
            GcValue::SharedMap(_) => "Map",
            GcValue::Set(_) => "Set",
            GcValue::Record(ptr) => {
                heap.get_record(*ptr)
                    .map(|r| r.type_name.as_str())
                    .unwrap_or("Record")
            }
            GcValue::ReactiveRecord(r) => r.type_name.as_str(),
            GcValue::Variant(ptr) => {
                heap.get_variant(*ptr)
                    .map(|v| v.type_name.as_str())
                    .unwrap_or("Variant")
            }
            GcValue::BigInt(_) => "BigInt",
            GcValue::Closure(_, _) => "Closure",
            GcValue::Function(_) => "Function",
            GcValue::NativeFunction(_) => "NativeFunction",
            GcValue::Pid(_) => "Pid",
            GcValue::Ref(_) => "Ref",
            GcValue::Type(_) => "Type",
            GcValue::Pointer(_) => "Pointer",
            GcValue::Int64List(_) => "Int64List",
            GcValue::Buffer(_) => "Buffer",
            GcValue::NativeHandle(_) => "NativeHandle",
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
            GcValue::Int64(i) => Some(MapKey::Int64(*i)),
            GcValue::Char(c) => Some(MapKey::Char(*c)),
            GcValue::String(ptr) => {
                heap.get_string(*ptr)
                    .map(|s| MapKey::String(Arc::new(s.data.clone())))
            }
            _ => None,
        }
    }

    /// Convert to a GC map key if possible (for GcMap/GcSet).
    pub fn to_gc_map_key(&self, heap: &Heap) -> Option<GcMapKey> {
        match self {
            GcValue::Unit => Some(GcMapKey::Unit),
            GcValue::Bool(b) => Some(GcMapKey::Bool(*b)),
            GcValue::Char(c) => Some(GcMapKey::Char(*c)),
            GcValue::Int8(n) => Some(GcMapKey::Int8(*n)),
            GcValue::Int16(n) => Some(GcMapKey::Int16(*n)),
            GcValue::Int32(n) => Some(GcMapKey::Int32(*n)),
            GcValue::Int64(n) => Some(GcMapKey::Int64(*n)),
            GcValue::UInt8(n) => Some(GcMapKey::UInt8(*n)),
            GcValue::UInt16(n) => Some(GcMapKey::UInt16(*n)),
            GcValue::UInt32(n) => Some(GcMapKey::UInt32(*n)),
            GcValue::UInt64(n) => Some(GcMapKey::UInt64(*n)),
            GcValue::String(ptr) => {
                heap.get_string(*ptr)
                    .map(|s| GcMapKey::String(s.data.clone()))
            },
            GcValue::Record(ptr) => {
                let record = heap.get_record(*ptr)?;
                // Convert all fields to keys - if any fails, the whole thing fails
                let key_fields: Option<Vec<GcMapKey>> = record.fields.iter()
                    .map(|f| f.to_gc_map_key(heap))
                    .collect();
                key_fields.map(|fields| GcMapKey::Record {
                    type_name: record.type_name.clone(),
                    field_names: record.field_names.clone(),
                    fields,
                })
            }
            GcValue::Variant(ptr) => {
                let variant = heap.get_variant(*ptr)?;
                // Convert all fields to keys - if any fails, the whole thing fails
                let key_fields: Option<Vec<GcMapKey>> = variant.fields.iter()
                    .map(|f| f.to_gc_map_key(heap))
                    .collect();
                key_fields.map(|fields| GcMapKey::Variant {
                    type_name: (*variant.type_name).clone(),
                    constructor: (*variant.constructor).clone(),
                    fields,
                })
            }
            _ => None,
        }
    }
}

/// The type of a heap object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObjectType {
    String,
    Buffer,
    List,
    Array,
    Int64Array,
    Float64Array,
    Float32Array,
    Tuple,
    Map,
    Set,
    Record,
    Variant,
    BigInt,
    Closure,
    NativeHandle,
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
    Buffer(GcBuffer),
    // Note: List is NOT here - lists are stored inline in GcValue for O(1) tail without allocation
    Array(GcArray),
    Int64Array(GcInt64Array),
    Float64Array(GcFloat64Array),
    Float32Array(GcFloat32Array),
    Tuple(GcTuple),
    Map(GcMap),
    Set(GcSet),
    Record(GcRecord),
    Variant(GcVariant),
    BigInt(GcBigInt),
    Closure(GcClosure),
    /// Native handle with GC-managed cleanup
    /// Wrapped in Arc so cloning is safe and cleanup only happens once.
    NativeHandle(Arc<GcNativeHandle>),
}

impl HeapData {
    /// Get the type of this heap data.
    pub fn object_type(&self) -> ObjectType {
        match self {
            HeapData::String(_) => ObjectType::String,
            HeapData::Buffer(_) => ObjectType::Buffer,
            // Note: List is not in HeapData - lists are inline in GcValue
            HeapData::Array(_) => ObjectType::Array,
            HeapData::Int64Array(_) => ObjectType::Int64Array,
            HeapData::Float64Array(_) => ObjectType::Float64Array,
            HeapData::Float32Array(_) => ObjectType::Float32Array,
            HeapData::Tuple(_) => ObjectType::Tuple,
            HeapData::Map(_) => ObjectType::Map,
            HeapData::Set(_) => ObjectType::Set,
            HeapData::Record(_) => ObjectType::Record,
            HeapData::Variant(_) => ObjectType::Variant,
            HeapData::BigInt(_) => ObjectType::BigInt,
            HeapData::Closure(_) => ObjectType::Closure,
            HeapData::NativeHandle(_) => ObjectType::NativeHandle,
        }
    }

    /// Get all GC pointers contained in this heap data.
    pub fn gc_pointers(&self) -> Vec<RawGcPtr> {
        match self {
            HeapData::String(_) => vec![],
            HeapData::Buffer(_) => vec![], // Buffers contain no GC pointers
            // Note: List is not in HeapData - lists are inline in GcValue
            HeapData::Array(arr) => arr
                .items
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            // Typed arrays contain no GC pointers (raw values)
            HeapData::Int64Array(_) => vec![],
            HeapData::Float64Array(_) => vec![],
            HeapData::Float32Array(_) => vec![],
            HeapData::Tuple(tuple) => tuple
                .items
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::Map(map) => {
                let mut ptrs = Vec::new();
                for (_, v) in &map.entries {
                    // Keys are now inline values (String), no pointers
                    ptrs.extend(v.gc_pointers());
                }
                ptrs
            }
            HeapData::Set(_set) => {
                // Keys are inline, no pointers
                vec![]
            }
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
            HeapData::BigInt(_) => vec![],
            HeapData::Closure(clo) => clo
                .captures
                .iter()
                .flat_map(|v| v.gc_pointers())
                .collect(),
            HeapData::NativeHandle(_) => vec![], // Native handles contain no GC pointers
        }
    }

    /// Estimate the size of this object in bytes.
    #[inline]
    pub fn estimate_size(&self) -> usize {
        match self {
            HeapData::String(s) => std::mem::size_of::<GcString>() + s.data.len(),
            // Note: List is not in HeapData - lists are inline in GcValue
            HeapData::Array(a) => {
                std::mem::size_of::<GcArray>() + a.items.len() * std::mem::size_of::<GcValue>()
            }
            HeapData::Int64Array(a) => {
                std::mem::size_of::<GcInt64Array>() + a.items.len() * std::mem::size_of::<i64>()
            }
            HeapData::Float64Array(a) => {
                std::mem::size_of::<GcFloat64Array>() + a.items.len() * std::mem::size_of::<f64>()
            }
            HeapData::Float32Array(a) => {
                std::mem::size_of::<GcFloat32Array>() + a.items.len() * std::mem::size_of::<f32>()
            }
            HeapData::Tuple(t) => {
                std::mem::size_of::<GcTuple>() + t.items.len() * std::mem::size_of::<GcValue>()
            }
            HeapData::Map(m) => {
                std::mem::size_of::<GcMap>()
                    + m.entries.len() * (std::mem::size_of::<GcMapKey>() + std::mem::size_of::<GcValue>())
                    + m.entries.keys().map(|k| match k { GcMapKey::String(s) => s.len(), _ => 0 }).sum::<usize>()
            }
            HeapData::Set(s) => {
                std::mem::size_of::<GcSet>()
                    + s.items.len() * std::mem::size_of::<GcMapKey>()
                    + s.items.iter().map(|k| match k { GcMapKey::String(s) => s.len(), _ => 0 }).sum::<usize>()
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
            HeapData::BigInt(b) => {
                std::mem::size_of::<GcBigInt>() + b.value.to_bytes_le().1.len()
            }
            HeapData::Closure(c) => {
                std::mem::size_of::<GcClosure>()
                    + c.captures.len() * std::mem::size_of::<GcValue>()
            }
            HeapData::Buffer(b) => {
                std::mem::size_of::<GcBuffer>() + b.data.borrow().len()
            }
            HeapData::NativeHandle(_) => {
                // Just the handle struct size, not the native memory (we don't know its size)
                std::mem::size_of::<GcNativeHandle>()
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

impl GcConfig {
    /// Configuration for lightweight spawned processes.
    /// Uses minimal pre-allocation since most workers need very few heap objects.
    pub fn lightweight() -> Self {
        Self {
            initial_capacity: 8, // Minimal pre-allocation
            gc_threshold: 64 * 1024, // 64KB - trigger GC earlier for small heaps
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
    /// Current count of live objects (tracked incrementally for O(1) access)
    live_count: usize,
    /// Configuration
    config: GcConfig,
    /// Statistics
    stats: GcStats,
}

// Heap is safe to Send between threads:
// - Each process has its own Heap instance
// - Heaps are not shared between threads
// - The contained GcObjects are Send (they contain GcValues which are Send)
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

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
            live_count: 0,
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

    /// Get the number of live objects (O(1) - tracked incrementally).
    pub fn live_objects(&self) -> usize {
        self.live_count
    }

    /// Get the total heap capacity.
    pub fn capacity(&self) -> usize {
        self.objects.len()
    }

    /// Get bytes allocated since last GC.
    pub fn bytes_since_gc(&self) -> usize {
        self.bytes_since_gc
    }

    /// Track additional memory for inline structures (like GcList with imbl::Vector).
    /// Call this when creating inline values that consume significant memory.
    #[inline]
    pub fn track_inline_memory(&mut self, bytes: usize) {
        self.bytes_since_gc += bytes;
    }

    /// Allocate a new object on the heap.
    #[inline]
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
        self.live_count += 1;

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

        // Update peak stats (now O(1) since live_count is tracked)
        if self.live_count > self.stats.peak_objects {
            self.stats.peak_objects = self.live_count;
        }

        index
    }

    /// Allocate a string.
    pub fn alloc_string(&mut self, s: String) -> GcPtr<GcString> {
        let data = HeapData::String(GcString { data: s });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a mutable string buffer.
    pub fn alloc_buffer(&mut self) -> GcPtr<GcBuffer> {
        let data = HeapData::Buffer(GcBuffer::new());
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a mutable string buffer with initial capacity.
    pub fn alloc_buffer_with_capacity(&mut self, capacity: usize) -> GcPtr<GcBuffer> {
        let data = HeapData::Buffer(GcBuffer::with_capacity(capacity));
        GcPtr::from_raw(self.alloc(data))
    }

    /// Create a list with memory tracking.
    /// Lists are inline in GcValue but we track their memory for GC triggering.
    #[inline]
    pub fn make_list(&mut self, items: Vec<GcValue>) -> GcList {
        // Track memory: each element is a GcValue, plus imbl::Vector overhead
        // imbl::Vector uses ~64 bytes base + 32 bytes per chunk of 64 elements
        let mem_size = std::mem::size_of::<GcList>()
            + items.len() * std::mem::size_of::<GcValue>()
            + 64; // imbl overhead estimate
        self.track_inline_memory(mem_size);
        GcList::from_vec(items)
    }

    /// Create an empty list (no heap allocation, minimal tracking).
    #[inline]
    pub fn make_empty_list(&mut self) -> GcList {
        self.track_inline_memory(std::mem::size_of::<GcList>());
        GcList::new()
    }

    /// Create a variant value (convenience method).
    pub fn make_variant(&mut self, type_name: &str, constructor: &str, fields: Vec<GcValue>) -> GcValue {
        let ptr = self.alloc_variant(
            Arc::new(type_name.to_string()),
            Arc::new(constructor.to_string()),
            fields,
        );
        GcValue::Variant(ptr)
    }

    /// Create a tuple value (convenience method).
    pub fn make_tuple(&mut self, fields: Vec<GcValue>) -> GcValue {
        // Tuples are represented as records with numeric field names
        let field_names: Vec<String> = (0..fields.len()).map(|i| i.to_string()).collect();
        let mutable_fields = vec![false; fields.len()];
        let ptr = self.alloc_record(
            "Tuple".to_string(),
            field_names,
            fields,
            mutable_fields,
        );
        GcValue::Record(ptr)
    }

    /// Allocate an array.
    pub fn alloc_array(&mut self, items: Vec<GcValue>) -> GcPtr<GcArray> {
        let data = HeapData::Array(GcArray { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a typed i64 array (JIT-optimized).
    pub fn alloc_int64_array(&mut self, items: Vec<i64>) -> GcPtr<GcInt64Array> {
        let data = HeapData::Int64Array(GcInt64Array { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a typed f64 array (JIT-optimized).
    pub fn alloc_float64_array(&mut self, items: Vec<f64>) -> GcPtr<GcFloat64Array> {
        let data = HeapData::Float64Array(GcFloat64Array { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a typed f32 array (for vectors/pgvector).
    pub fn alloc_float32_array(&mut self, items: Vec<f32>) -> GcPtr<GcFloat32Array> {
        let data = HeapData::Float32Array(GcFloat32Array { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a tuple.
    pub fn alloc_tuple(&mut self, items: Vec<GcValue>) -> GcPtr<GcTuple> {
        let data = HeapData::Tuple(GcTuple { items });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a map.
    pub fn alloc_map(&mut self, entries: ImblHashMap<GcMapKey, GcValue>) -> GcPtr<GcMap> {
        let data = HeapData::Map(GcMap { entries });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a set.
    pub fn alloc_set(&mut self, items: ImblHashSet<GcMapKey>) -> GcPtr<GcSet> {
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
        type_name: Arc<String>,
        constructor: Arc<String>,
        fields: Vec<GcValue>,
    ) -> GcPtr<GcVariant> {
        let discriminant = constructor_discriminant(constructor.as_ref());
        let data = HeapData::Variant(GcVariant {
            type_name,
            constructor,
            fields,
            discriminant,
        });
        GcPtr::from_raw(self.alloc(data))
    }

    /// Allocate a closure.
    pub fn alloc_closure(
        &mut self,
        function: Arc<FunctionValue>,
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

    /// Allocate a native handle with GC-managed cleanup.
    /// When this handle is garbage collected, the cleanup function will be called
    /// with (ptr, type_id) to free the native memory.
    pub fn alloc_native_handle(&mut self, handle: Arc<GcNativeHandle>) -> GcPtr<Arc<GcNativeHandle>> {
        let data = HeapData::NativeHandle(handle);
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

    /// Get a typed reference to buffer data.
    pub fn get_buffer(&self, ptr: GcPtr<GcBuffer>) -> Option<&GcBuffer> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Buffer(ref b) => Some(b),
            _ => None,
        }
    }

    // Note: get_list/get_list_mut removed - lists are now inline in GcValue

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

    /// Get a typed reference to i64 array data.
    pub fn get_int64_array(&self, ptr: GcPtr<GcInt64Array>) -> Option<&GcInt64Array> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Int64Array(ref a) => Some(a),
            _ => None,
        }
    }

    /// Get a mutable reference to i64 array data.
    pub fn get_int64_array_mut(&mut self, ptr: GcPtr<GcInt64Array>) -> Option<&mut GcInt64Array> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::Int64Array(ref mut a) => Some(a),
            _ => None,
        }
    }

    /// Get a typed reference to f64 array data.
    pub fn get_float64_array(&self, ptr: GcPtr<GcFloat64Array>) -> Option<&GcFloat64Array> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Float64Array(ref a) => Some(a),
            _ => None,
        }
    }

    /// Get a mutable reference to f64 array data.
    pub fn get_float64_array_mut(&mut self, ptr: GcPtr<GcFloat64Array>) -> Option<&mut GcFloat64Array> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::Float64Array(ref mut a) => Some(a),
            _ => None,
        }
    }

    /// Get a typed reference to f32 array data.
    pub fn get_float32_array(&self, ptr: GcPtr<GcFloat32Array>) -> Option<&GcFloat32Array> {
        match self.get(ptr.as_raw())?.data {
            HeapData::Float32Array(ref a) => Some(a),
            _ => None,
        }
    }

    /// Get a mutable reference to f32 array data.
    pub fn get_float32_array_mut(&mut self, ptr: GcPtr<GcFloat32Array>) -> Option<&mut GcFloat32Array> {
        match self.get_mut(ptr.as_raw())?.data {
            HeapData::Float32Array(ref mut a) => Some(a),
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

    /// Fast unchecked closure access - skips bounds checks and Option handling.
    /// SAFETY: ptr must be a valid closure pointer that hasn't been collected.
    #[inline(always)]
    pub unsafe fn get_closure_unchecked(&self, ptr: GcPtr<GcClosure>) -> &GcClosure {
        let obj = self.objects.get_unchecked(ptr.as_raw() as usize);
        match &obj.as_ref().unwrap_unchecked().data {
            HeapData::Closure(ref c) => c,
            _ => std::hint::unreachable_unchecked(),
        }
    }

    /// Get a typed reference to BigInt data.
    pub fn get_bigint(&self, ptr: GcPtr<GcBigInt>) -> Option<&GcBigInt> {
        match self.get(ptr.as_raw())?.data {
            HeapData::BigInt(ref b) => Some(b),
            _ => None,
        }
    }

    /// Get a typed reference to native handle data.
    pub fn get_native_handle(&self, ptr: GcPtr<Arc<GcNativeHandle>>) -> Option<&Arc<GcNativeHandle>> {
        match self.get(ptr.as_raw())?.data {
            HeapData::NativeHandle(ref h) => Some(h),
            _ => None,
        }
    }

    /// Allocate a BigInt on the heap.
    pub fn alloc_bigint(&mut self, value: BigInt) -> GcPtr<GcBigInt> {
        let ptr = self.alloc(HeapData::BigInt(GcBigInt { value }));
        GcPtr::from_raw(ptr)
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
            (GcValue::Char(a), GcValue::Char(b)) => a == b,
            // Signed integers
            (GcValue::Int8(a), GcValue::Int8(b)) => a == b,
            (GcValue::Int16(a), GcValue::Int16(b)) => a == b,
            (GcValue::Int32(a), GcValue::Int32(b)) => a == b,
            (GcValue::Int64(a), GcValue::Int64(b)) => a == b,
            // Unsigned integers
            (GcValue::UInt8(a), GcValue::UInt8(b)) => a == b,
            (GcValue::UInt16(a), GcValue::UInt16(b)) => a == b,
            (GcValue::UInt32(a), GcValue::UInt32(b)) => a == b,
            (GcValue::UInt64(a), GcValue::UInt64(b)) => a == b,
            // Floats
            (GcValue::Float32(a), GcValue::Float32(b)) => a == b,
            (GcValue::Float64(a), GcValue::Float64(b)) => a == b,
            // Decimal
            (GcValue::Decimal(a), GcValue::Decimal(b)) => a == b,
            // Special
            (GcValue::Pid(a), GcValue::Pid(b)) => a == b,

            // BigInt - compare by content
            (GcValue::BigInt(a), GcValue::BigInt(b)) => {
                if a == b {
                    return true;
                }
                match (self.get_bigint(*a), self.get_bigint(*b)) {
                    (Some(ba), Some(bb)) => ba.value == bb.value,
                    _ => false,
                }
            }

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

            // Lists - compare elements recursively (lists are now inline)
            (GcValue::List(a), GcValue::List(b)) => {
                if a == b {
                    return true;
                }
                if a.len() != b.len() {
                    return false;
                }
                a.items()
                    .iter()
                    .zip(b.items().iter())
                    .all(|(ia, ib)| self.gc_values_equal(ia, ib))
            }

            // Int64List - compare elements directly
            (GcValue::Int64List(a), GcValue::Int64List(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter().zip(b.iter()).all(|(ia, ib)| ia == ib)
            }

            // Cross-type: List vs Int64List (compare element by element)
            (GcValue::List(list), GcValue::Int64List(int_list))
            | (GcValue::Int64List(int_list), GcValue::List(list)) => {
                if list.len() != int_list.len() {
                    return false;
                }
                list.iter().zip(int_list.iter()).all(|(lv, iv)| {
                    matches!(lv, GcValue::Int64(n) if *n == iv)
                })
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
            (GcValue::Function(a), GcValue::Function(b)) => Arc::ptr_eq(a, b),
            (GcValue::Closure(a, _), GcValue::Closure(b, _)) => a == b,
            (GcValue::NativeFunction(a), GcValue::NativeFunction(b)) => Arc::ptr_eq(a, b),

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
            GcValue::Char(c) => format!("{}", c),
            // Signed integers
            GcValue::Int8(i) => format!("{}", i),
            GcValue::Int16(i) => format!("{}", i),
            GcValue::Int32(i) => format!("{}", i),
            GcValue::Int64(i) => format!("{}", i),
            // Unsigned integers
            GcValue::UInt8(i) => format!("{}", i),
            GcValue::UInt16(i) => format!("{}", i),
            GcValue::UInt32(i) => format!("{}", i),
            GcValue::UInt64(i) => format!("{}", i),
            // Floats
            GcValue::Float32(f) => format!("{}", f),
            GcValue::Float64(f) => format!("{}", f),
            // Decimal
            GcValue::Decimal(d) => format!("{}", d),
            // BigInt
            GcValue::BigInt(ptr) => {
                if let Some(big) = self.get_bigint(*ptr) {
                    format!("{}", big.value)
                } else {
                    "<invalid bigint>".to_string()
                }
            }
            GcValue::String(ptr) => {
                if let Some(s) = self.get_string(*ptr) {
                    s.data.clone()
                } else {
                    "<invalid string>".to_string()
                }
            }
            GcValue::List(list) => {
                let mut result = "[".to_string();
                for (i, item) in list.items().iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&self.display_value(item));
                }
                result.push(']');
                result
            }
            GcValue::Array(ptr) => {
                if let Some(arr) = self.get_array(*ptr) {
                    format!("Array[{}]", arr.items.len())
                } else {
                    "<invalid array>".to_string()
                }
            }
            GcValue::Int64Array(ptr) => {
                if let Some(arr) = self.get_int64_array(*ptr) {
                    format!("Int64Array[{}]", arr.items.len())
                } else {
                    "<invalid int64 array>".to_string()
                }
            }
            GcValue::Float64Array(ptr) => {
                if let Some(arr) = self.get_float64_array(*ptr) {
                    format!("Float64Array[{}]", arr.items.len())
                } else {
                    "<invalid float64 array>".to_string()
                }
            }
            GcValue::Float32Array(ptr) => {
                if let Some(arr) = self.get_float32_array(*ptr) {
                    format!("Float32Array[{}]", arr.items.len())
                } else {
                    "<invalid float32 array>".to_string()
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
            GcValue::SharedMap(map) => {
                format!("%{{...{} entries}}", map.len())
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
            GcValue::ReactiveRecord(rec) => {
                let field_count = rec.field_names.len();
                format!("{}{{...{} fields}}", rec.type_name, field_count)
            }
            GcValue::Variant(ptr) => {
                if let Some(var) = self.get_variant(*ptr) {
                    if var.fields.is_empty() {
                        var.constructor.to_string()
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
            GcValue::Closure(ptr, _) => {
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
            GcValue::Int64List(list) => {
                let items: Vec<String> = list.iter().take(10).map(|n| n.to_string()).collect();
                if list.len() > 10 {
                    format!("[{}... ({} more)]", items.join(", "), list.len() - 10)
                } else {
                    format!("[{}]", items.join(", "))
                }
            }
            GcValue::Buffer(buf_ptr) => {
                self.get_buffer(buf_ptr.clone())
                    .map(|b| b.to_string())
                    .unwrap_or_else(|| "<buffer>".to_string())
            }
            GcValue::NativeHandle(h) => {
                format!("<native ptr=0x{:x} type={}>", h.ptr, h.type_id)
            }
        }
    }

    /// Convert a GcValue to a SharedMapValue for cross-thread sharing.
    /// Returns None if the value type cannot be converted (e.g., closures with GC captures).
    pub fn gc_value_to_shared(&self, value: &GcValue) -> Option<SharedMapValue> {
        Some(match value {
            GcValue::Unit => SharedMapValue::Unit,
            GcValue::Bool(b) => SharedMapValue::Bool(*b),
            GcValue::Char(c) => SharedMapValue::Char(*c),
            GcValue::Int8(n) => SharedMapValue::Int64(*n as i64),
            GcValue::Int16(n) => SharedMapValue::Int64(*n as i64),
            GcValue::Int32(n) => SharedMapValue::Int64(*n as i64),
            GcValue::Int64(n) => SharedMapValue::Int64(*n),
            GcValue::UInt8(n) => SharedMapValue::Int64(*n as i64),
            GcValue::UInt16(n) => SharedMapValue::Int64(*n as i64),
            GcValue::UInt32(n) => SharedMapValue::Int64(*n as i64),
            GcValue::UInt64(n) => SharedMapValue::Int64(*n as i64),
            GcValue::Float32(f) => SharedMapValue::Float64(*f as f64),
            GcValue::Float64(f) => SharedMapValue::Float64(*f),
            GcValue::Pid(p) => SharedMapValue::Pid(*p),
            GcValue::String(ptr) => {
                let s = self.get_string(*ptr)?;
                SharedMapValue::String(s.data.clone())
            }
            GcValue::List(list) => {
                let items: Option<Vec<_>> = list.items().iter()
                    .map(|v| self.gc_value_to_shared(v))
                    .collect();
                SharedMapValue::List(items?)
            }
            GcValue::Tuple(ptr) => {
                let tuple = self.get_tuple(*ptr)?;
                let items: Option<Vec<_>> = tuple.items.iter()
                    .map(|v| self.gc_value_to_shared(v))
                    .collect();
                SharedMapValue::Tuple(items?)
            }
            GcValue::Record(ptr) => {
                let rec = self.get_record(*ptr)?;
                let fields: Option<Vec<_>> = rec.fields.iter()
                    .map(|v| self.gc_value_to_shared(v))
                    .collect();
                SharedMapValue::Record {
                    type_name: rec.type_name.clone(),
                    field_names: rec.field_names.clone(),
                    fields: fields?,
                }
            }
            GcValue::Variant(ptr) => {
                let var = self.get_variant(*ptr)?;
                let fields: Option<Vec<_>> = var.fields.iter()
                    .map(|v| self.gc_value_to_shared(v))
                    .collect();
                SharedMapValue::Variant {
                    type_name: var.type_name.to_string(),
                    constructor: var.constructor.to_string(),
                    fields: fields?,
                }
            }
            GcValue::Map(ptr) => {
                let map = self.get_map(*ptr)?;
                let entries: Option<ImblHashMap<SharedMapKey, SharedMapValue>> = map.entries.iter()
                    .map(|(k, v)| {
                        let shared_v = self.gc_value_to_shared(v)?;
                        Some((k.to_shared_key(), shared_v))
                    })
                    .collect();
                SharedMapValue::Map(Arc::new(entries?))
            }
            GcValue::SharedMap(map) => SharedMapValue::Map(map.clone()),
            GcValue::Set(ptr) => {
                let set = self.get_set(*ptr)?;
                let items: Vec<_> = set.items.iter()
                    .map(|k| k.to_shared_key())
                    .collect();
                SharedMapValue::Set(items)
            }
            GcValue::Int64Array(ptr) => {
                let arr = self.get_int64_array(*ptr)?;
                SharedMapValue::Int64Array(arr.items.clone())
            }
            GcValue::Float64Array(ptr) => {
                let arr = self.get_float64_array(*ptr)?;
                SharedMapValue::Float64Array(arr.items.clone())
            }
            GcValue::Float32Array(ptr) => {
                let arr = self.get_float32_array(*ptr)?;
                SharedMapValue::Float32Array(arr.items.clone())
            }
            // Types that can't be shared
            GcValue::Decimal(_) | GcValue::BigInt(_) | GcValue::Array(_) |
            GcValue::Closure(_, _) | GcValue::Function(_) | GcValue::NativeFunction(_) |
            GcValue::Ref(_) | GcValue::Type(_) | GcValue::Pointer(_) |
            GcValue::Int64List(_) | GcValue::Buffer(_) | GcValue::NativeHandle(_) |
            GcValue::ReactiveRecord(_) => {
                return None;
            }
        })
    }

    /// Convert a SharedMapValue to a GcValue, allocating on this heap.
    pub fn shared_to_gc_value(&mut self, value: &SharedMapValue) -> GcValue {
        match value {
            SharedMapValue::Unit => GcValue::Unit,
            SharedMapValue::Bool(b) => GcValue::Bool(*b),
            SharedMapValue::Char(c) => GcValue::Char(*c),
            SharedMapValue::Int64(n) => GcValue::Int64(*n),
            SharedMapValue::Float64(f) => GcValue::Float64(*f),
            SharedMapValue::Pid(p) => GcValue::Pid(*p),
            SharedMapValue::String(s) => {
                let ptr = self.alloc_string(s.clone());
                GcValue::String(ptr)
            }
            SharedMapValue::List(items) => {
                let gc_items: Vec<_> = items.iter()
                    .map(|v| self.shared_to_gc_value(v))
                    .collect();
                GcValue::List(self.make_list(gc_items))
            }
            SharedMapValue::Tuple(items) => {
                let gc_items: Vec<_> = items.iter()
                    .map(|v| self.shared_to_gc_value(v))
                    .collect();
                let ptr = self.alloc_tuple(gc_items);
                GcValue::Tuple(ptr)
            }
            SharedMapValue::Record { type_name, field_names, fields } => {
                let gc_fields: Vec<_> = fields.iter()
                    .map(|v| self.shared_to_gc_value(v))
                    .collect();
                let ptr = self.alloc_record(
                    type_name.clone(),
                    field_names.clone(),
                    gc_fields,
                    vec![false; field_names.len()], // Assume immutable by default
                );
                GcValue::Record(ptr)
            }
            SharedMapValue::Variant { type_name, constructor, fields } => {
                let gc_fields: Vec<_> = fields.iter()
                    .map(|v| self.shared_to_gc_value(v))
                    .collect();
                let ptr = self.alloc_variant(
                    Arc::new(type_name.clone()),
                    Arc::new(constructor.clone()),
                    gc_fields,
                );
                GcValue::Variant(ptr)
            }
            SharedMapValue::Map(map) => {
                // Keep as SharedMap for O(1) sharing - don't convert!
                GcValue::SharedMap(map.clone())
            }
            SharedMapValue::Set(items) => {
                let gc_items: ImblHashSet<GcMapKey> = items.iter()
                    .map(|k| GcMapKey::from_shared_key(k))
                    .collect();
                let ptr = self.alloc_set(gc_items);
                GcValue::Set(ptr)
            }
            SharedMapValue::Int64Array(items) => {
                let ptr = self.alloc_int64_array(items.clone());
                GcValue::Int64Array(ptr)
            }
            SharedMapValue::Float64Array(items) => {
                let ptr = self.alloc_float64_array(items.clone());
                GcValue::Float64Array(ptr)
            }
            SharedMapValue::Float32Array(items) => {
                let ptr = self.alloc_float32_array(items.clone());
                GcValue::Float32Array(ptr)
            }
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
        self.live_count -= freed as usize;

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
            GcValue::Char(c) => GcValue::Char(*c),
            // Signed integers
            GcValue::Int8(i) => GcValue::Int8(*i),
            GcValue::Int16(i) => GcValue::Int16(*i),
            GcValue::Int32(i) => GcValue::Int32(*i),
            GcValue::Int64(i) => GcValue::Int64(*i),
            // Unsigned integers
            GcValue::UInt8(i) => GcValue::UInt8(*i),
            GcValue::UInt16(i) => GcValue::UInt16(*i),
            GcValue::UInt32(i) => GcValue::UInt32(*i),
            GcValue::UInt64(i) => GcValue::UInt64(*i),
            // Floats
            GcValue::Float32(f) => GcValue::Float32(*f),
            GcValue::Float64(f) => GcValue::Float64(*f),
            // Decimal
            GcValue::Decimal(d) => GcValue::Decimal(*d),
            // Special
            GcValue::Pid(p) => GcValue::Pid(*p),
            GcValue::Ref(r) => GcValue::Ref(*r),

            // BigInt - deep copy from source heap
            GcValue::BigInt(ptr) => {
                if let Some(b) = source.get_bigint(*ptr) {
                    GcValue::BigInt(self.alloc_bigint(b.value.clone()))
                } else {
                    GcValue::Unit
                }
            }

            // Heap values need recursive deep copy
            GcValue::String(ptr) => {
                if let Some(s) = source.get_string(*ptr) {
                    GcValue::String(self.alloc_string(s.data.clone()))
                } else {
                    GcValue::Unit // Fallback for invalid pointer
                }
            }
            GcValue::List(list) => {
                let items: Vec<GcValue> = list
                    .items()
                    .iter()
                    .map(|v| self.deep_copy(v, source))
                    .collect();
                GcValue::List(self.make_list(items))
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
            GcValue::Int64Array(ptr) => {
                if let Some(arr) = source.get_int64_array(*ptr) {
                    GcValue::Int64Array(self.alloc_int64_array(arr.items.clone()))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Float64Array(ptr) => {
                if let Some(arr) = source.get_float64_array(*ptr) {
                    GcValue::Float64Array(self.alloc_float64_array(arr.items.clone()))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Float32Array(ptr) => {
                if let Some(arr) = source.get_float32_array(*ptr) {
                    GcValue::Float32Array(self.alloc_float32_array(arr.items.clone()))
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
                    let entries: ImblHashMap<GcMapKey, GcValue> = map
                        .entries
                        .iter()
                        .map(|(k, v)| (self.deep_copy_key(k, source), self.deep_copy(v, source)))
                        .collect();
                    GcValue::Map(self.alloc_map(entries))
                } else {
                    GcValue::Unit
                }
            }
            // SharedMap is Arc-managed, just clone the reference (O(1))
            GcValue::SharedMap(map) => GcValue::SharedMap(map.clone()),
            GcValue::Set(ptr) => {
                if let Some(set) = source.get_set(*ptr) {
                    let items: ImblHashSet<GcMapKey> = set
                        .items
                        .iter()
                        .cloned()
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
            GcValue::ReactiveRecord(rec) => {
                // Deep copy creates a new reactive record with copied fields but empty parents
                if let Ok(fields) = rec.fields.read() {
                    let copied_fields: Vec<Value> = fields.iter().cloned().collect();
                    GcValue::ReactiveRecord(Arc::new(ReactiveRecordValue::new(
                        rec.type_name.clone(),
                        rec.field_names.clone(),
                        copied_fields,
                        rec.reactive_field_mask,
                    )))
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
                        Arc::clone(&var.type_name),
                        Arc::clone(&var.constructor),
                        fields,
                    ))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Closure(ptr, inline_op) => {
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
                    ), *inline_op)
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
            // Int64List - clone (Arc-based, so cheap)
            GcValue::Int64List(list) => GcValue::Int64List(list.clone()),
            // Buffer - deep copy the contents
            GcValue::Buffer(ptr) => {
                if let Some(b) = source.get_buffer(*ptr) {
                    let buf_ptr = self.alloc_buffer();
                    if let Some(new_buf) = self.get_buffer(buf_ptr.clone()) {
                        new_buf.append(&b.to_string());
                    }
                    GcValue::Buffer(buf_ptr)
                } else {
                    GcValue::Unit
                }
            }

            // Native handles cannot be deep copied - they own unique native memory.
            // For message passing, the extension should provide a copy mechanism.
            // We share the Arc - the handle will only be freed when all refs are dropped.
            GcValue::NativeHandle(h) => GcValue::NativeHandle(h.clone())
        }
    }

    /// Deep copy a map key.
    fn deep_copy_key(&mut self, key: &GcMapKey, source: &Heap) -> GcMapKey {
        match key {
            GcMapKey::Unit => GcMapKey::Unit,
            GcMapKey::Bool(b) => GcMapKey::Bool(*b),
            GcMapKey::Char(c) => GcMapKey::Char(*c),
            // Signed integers
            GcMapKey::Int8(i) => GcMapKey::Int8(*i),
            GcMapKey::Int16(i) => GcMapKey::Int16(*i),
            GcMapKey::Int32(i) => GcMapKey::Int32(*i),
            GcMapKey::Int64(i) => GcMapKey::Int64(*i),
            // Unsigned integers
            GcMapKey::UInt8(i) => GcMapKey::UInt8(*i),
            GcMapKey::UInt16(i) => GcMapKey::UInt16(*i),
            GcMapKey::UInt32(i) => GcMapKey::UInt32(*i),
            GcMapKey::UInt64(i) => GcMapKey::UInt64(*i),
            // String
            GcMapKey::String(s) => GcMapKey::String(s.clone()),
            // Record
            GcMapKey::Record { type_name, field_names, fields } => {
                GcMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|f| self.deep_copy_key(f, source)).collect(),
                }
            }
            // Variant
            GcMapKey::Variant { type_name, constructor, fields } => {
                GcMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|f| self.deep_copy_key(f, source)).collect(),
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
            GcValue::Char(c) => GcValue::Char(*c),
            // Signed integers
            GcValue::Int8(i) => GcValue::Int8(*i),
            GcValue::Int16(i) => GcValue::Int16(*i),
            GcValue::Int32(i) => GcValue::Int32(*i),
            GcValue::Int64(i) => GcValue::Int64(*i),
            // Unsigned integers
            GcValue::UInt8(i) => GcValue::UInt8(*i),
            GcValue::UInt16(i) => GcValue::UInt16(*i),
            GcValue::UInt32(i) => GcValue::UInt32(*i),
            GcValue::UInt64(i) => GcValue::UInt64(*i),
            // Floats
            GcValue::Float32(f) => GcValue::Float32(*f),
            GcValue::Float64(f) => GcValue::Float64(*f),
            // Decimal
            GcValue::Decimal(d) => GcValue::Decimal(*d),
            // Special
            GcValue::Pid(p) => GcValue::Pid(*p),
            GcValue::Ref(r) => GcValue::Ref(*r),

            // BigInt - clone from same heap
            GcValue::BigInt(ptr) => {
                let val = self.get_bigint(*ptr).map(|b| b.value.clone());
                if let Some(val) = val {
                    GcValue::BigInt(self.alloc_bigint(val))
                } else {
                    GcValue::Unit
                }
            }

            // Heap values need recursive deep copy
            GcValue::String(ptr) => {
                let data = self.get_string(*ptr).map(|s| s.data.clone());
                if let Some(data) = data {
                    GcValue::String(self.alloc_string(data))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::List(list) => {
                let items = list.items().to_vec();
                let cloned: Vec<GcValue> = items.iter().map(|v| self.clone_value(v)).collect();
                GcValue::List(self.make_list(cloned))
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
            GcValue::Int64Array(ptr) => {
                let items = self.get_int64_array(*ptr).map(|a| a.items.clone());
                if let Some(items) = items {
                    GcValue::Int64Array(self.alloc_int64_array(items))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Float64Array(ptr) => {
                let items = self.get_float64_array(*ptr).map(|a| a.items.clone());
                if let Some(items) = items {
                    GcValue::Float64Array(self.alloc_float64_array(items))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Float32Array(ptr) => {
                let items = self.get_float32_array(*ptr).map(|a| a.items.clone());
                if let Some(items) = items {
                    GcValue::Float32Array(self.alloc_float32_array(items))
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
                    let cloned: ImblHashMap<GcMapKey, GcValue> = entries
                        .iter()
                        .map(|(k, v)| (self.clone_key(k), self.clone_value(v)))
                        .collect();
                    GcValue::Map(self.alloc_map(cloned))
                } else {
                    GcValue::Unit
                }
            }
            // SharedMap is Arc-managed with structural sharing - clone is O(1)
            GcValue::SharedMap(map) => GcValue::SharedMap(map.clone()),
            GcValue::Set(ptr) => {
                let items = self.get_set(*ptr).map(|s| s.items.clone());
                if let Some(items) = items {
                    let cloned: ImblHashSet<GcMapKey> =
                        items.iter().cloned().collect();
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
            GcValue::ReactiveRecord(rec) => {
                // Clone creates a new reactive record with copied fields but empty parents
                if let Ok(fields) = rec.fields.read() {
                    let cloned_fields: Vec<Value> = fields.iter().cloned().collect();
                    GcValue::ReactiveRecord(Arc::new(ReactiveRecordValue::new(
                        rec.type_name.clone(),
                        rec.field_names.clone(),
                        cloned_fields,
                        rec.reactive_field_mask,
                    )))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Variant(ptr) => {
                let var_data = self.get_variant(*ptr).map(|v| {
                    (Arc::clone(&v.type_name), Arc::clone(&v.constructor), v.fields.clone())
                });
                if let Some((type_name, constructor, fields)) = var_data {
                    let cloned: Vec<GcValue> = fields.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Variant(self.alloc_variant(type_name, constructor, cloned))
                } else {
                    GcValue::Unit
                }
            }
            GcValue::Closure(ptr, inline_op) => {
                let clo_data = self.get_closure(*ptr).map(|c| {
                    (c.function.clone(), c.captures.clone(), c.capture_names.clone())
                });
                if let Some((function, captures, capture_names)) = clo_data {
                    let cloned: Vec<GcValue> = captures.iter().map(|v| self.clone_value(v)).collect();
                    GcValue::Closure(self.alloc_closure(function, cloned, capture_names), *inline_op)
                } else {
                    GcValue::Unit
                }
            }

            // Rc-based values - just clone the Rc (shared reference)
            GcValue::Function(f) => GcValue::Function(f.clone()),
            GcValue::NativeFunction(n) => GcValue::NativeFunction(n.clone()),
            GcValue::Type(t) => GcValue::Type(t.clone()),
            GcValue::Pointer(p) => GcValue::Pointer(*p),
            // Int64List - clone (Arc-based, so cheap)
            GcValue::Int64List(list) => GcValue::Int64List(list.clone()),
            // Buffer - create new buffer with same contents
            GcValue::Buffer(ptr) => {
                if let Some(b) = self.get_buffer(*ptr) {
                    let s = b.to_string();
                    let buf_ptr = self.alloc_buffer();
                    if let Some(new_buf) = self.get_buffer(buf_ptr.clone()) {
                        new_buf.append(&s);
                    }
                    GcValue::Buffer(buf_ptr)
                } else {
                    GcValue::Unit
                }
            }
            // Native handle - just clone the Arc (shares ownership)
            GcValue::NativeHandle(h) => GcValue::NativeHandle(h.clone())
        }
    }

    /// Clone a map key within the same heap.
    fn clone_key(&mut self, key: &GcMapKey) -> GcMapKey {
        match key {
            GcMapKey::Unit => GcMapKey::Unit,
            GcMapKey::Bool(b) => GcMapKey::Bool(*b),
            GcMapKey::Char(c) => GcMapKey::Char(*c),
            // Signed integers
            GcMapKey::Int8(i) => GcMapKey::Int8(*i),
            GcMapKey::Int16(i) => GcMapKey::Int16(*i),
            GcMapKey::Int32(i) => GcMapKey::Int32(*i),
            GcMapKey::Int64(i) => GcMapKey::Int64(*i),
            // Unsigned integers
            GcMapKey::UInt8(i) => GcMapKey::UInt8(*i),
            GcMapKey::UInt16(i) => GcMapKey::UInt16(*i),
            GcMapKey::UInt32(i) => GcMapKey::UInt32(*i),
            GcMapKey::UInt64(i) => GcMapKey::UInt64(*i),
            // String
            GcMapKey::String(s) => GcMapKey::String(s.clone()),
            // Record
            GcMapKey::Record { type_name, field_names, fields } => {
                GcMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|f| self.clone_key(f)).collect(),
                }
            }
            // Variant
            GcMapKey::Variant { type_name, constructor, fields } => {
                GcMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|f| self.clone_key(f)).collect(),
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
            Value::Char(c) => GcValue::Char(*c),
            // Signed integers
            Value::Int8(i) => GcValue::Int8(*i),
            Value::Int16(i) => GcValue::Int16(*i),
            Value::Int32(i) => GcValue::Int32(*i),
            Value::Int64(i) => GcValue::Int64(*i),
            // Unsigned integers
            Value::UInt8(i) => GcValue::UInt8(*i),
            Value::UInt16(i) => GcValue::UInt16(*i),
            Value::UInt32(i) => GcValue::UInt32(*i),
            Value::UInt64(i) => GcValue::UInt64(*i),
            // Floats
            Value::Float32(f) => GcValue::Float32(*f),
            Value::Float64(f) => GcValue::Float64(*f),
            // Decimal
            Value::Decimal(d) => GcValue::Decimal(*d),
            // BigInt
            Value::BigInt(b) => GcValue::BigInt(self.alloc_bigint((**b).clone())),

            // String - allocate on GC heap
            Value::String(s) => {
                let ptr = self.alloc_string((**s).clone());
                GcValue::String(ptr)
            }

            // List - recursively convert elements
            Value::List(items) => {
                let gc_items: Vec<GcValue> = items.iter().map(|v| self.value_to_gc(v)).collect();
                let list = self.make_list(gc_items);
                GcValue::List(list)
            }

            // Array - recursively convert elements
            Value::Array(arr) => {
                let gc_items: Vec<GcValue> =
                    arr.read().unwrap().iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_array(gc_items);
                GcValue::Array(ptr)
            }

            // Typed arrays - copy the raw values
            Value::Int64Array(arr) => {
                let items = arr.read().unwrap().clone();
                GcValue::Int64Array(self.alloc_int64_array(items))
            }
            Value::Float64Array(arr) => {
                let items = arr.read().unwrap().clone();
                GcValue::Float64Array(self.alloc_float64_array(items))
            }
            Value::Float32Array(arr) => {
                let items = arr.read().unwrap().clone();
                GcValue::Float32Array(self.alloc_float32_array(items))
            }

            // Tuple - recursively convert elements
            Value::Tuple(items) => {
                let gc_items: Vec<GcValue> = items.iter().map(|v| self.value_to_gc(v)).collect();
                let ptr = self.alloc_tuple(gc_items);
                GcValue::Tuple(ptr)
            }

            // Map - convert keys and values
            Value::Map(entries) => {
                let gc_entries: ImblHashMap<GcMapKey, GcValue> = entries
                    .iter()
                    .map(|(k, v)| (self.map_key_to_gc(k), self.value_to_gc(v)))
                    .collect();
                let ptr = self.alloc_map(gc_entries);
                GcValue::Map(ptr)
            }

            // Set - convert keys
            Value::Set(items) => {
                let gc_items: ImblHashSet<GcMapKey> =
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

            // ReactiveRecord - store Arc directly (preserves parent tracking)
            Value::ReactiveRecord(r) => GcValue::ReactiveRecord(r.clone()),

            // Variant - convert fields
            Value::Variant(v) => {
                let gc_fields: Vec<GcValue> = v.fields.iter().map(|f| self.value_to_gc(f)).collect();
                let ptr =
                    self.alloc_variant(Arc::clone(&v.type_name), Arc::clone(&v.constructor), gc_fields);
                GcValue::Variant(ptr)
            }

            // Closure - convert captures
            Value::Closure(c) => {
                let gc_captures: Vec<GcValue> =
                    c.captures.iter().map(|v| self.value_to_gc(v)).collect();
                let inline_op = InlineOp::from_function(&c.function);
                let ptr = self.alloc_closure(c.function.clone(), gc_captures, c.capture_names.clone());
                GcValue::Closure(ptr, inline_op)
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

            // Native handle - store Arc directly (not in heap)
            Value::NativeHandle(h) => GcValue::NativeHandle(h.clone())
        }
    }

    /// Convert a MapKey to a GcMapKey.
    fn map_key_to_gc(&mut self, key: &MapKey) -> GcMapKey {
        match key {
            MapKey::Unit => GcMapKey::Unit,
            MapKey::Bool(b) => GcMapKey::Bool(*b),
            MapKey::Char(c) => GcMapKey::Char(*c),
            // Signed integers
            MapKey::Int8(i) => GcMapKey::Int8(*i),
            MapKey::Int16(i) => GcMapKey::Int16(*i),
            MapKey::Int32(i) => GcMapKey::Int32(*i),
            MapKey::Int64(i) => GcMapKey::Int64(*i),
            // Unsigned integers
            MapKey::UInt8(i) => GcMapKey::UInt8(*i),
            MapKey::UInt16(i) => GcMapKey::UInt16(*i),
            MapKey::UInt32(i) => GcMapKey::UInt32(*i),
            MapKey::UInt64(i) => GcMapKey::UInt64(*i),
            // String
            MapKey::String(s) => {
                GcMapKey::String((**s).clone())
            }
            // Record
            MapKey::Record { type_name, field_names, fields } => {
                GcMapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|f| self.map_key_to_gc(f)).collect(),
                }
            }
            // Variant
            MapKey::Variant { type_name, constructor, fields } => {
                GcMapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|f| self.map_key_to_gc(f)).collect(),
                }
            }
        }
    }

    /// Convert a GcValue back to a Value.
    ///
    /// This is used when returning results from the VM.
    pub fn gc_to_value(&self, gc_value: &GcValue) -> Value {
        use crate::value::Pid as ValuePid;
        use crate::value::RefId as ValueRefId;
        use std::sync::RwLock;

        match gc_value {
            // Immediate values - direct conversion
            GcValue::Unit => Value::Unit,
            GcValue::Bool(b) => Value::Bool(*b),
            GcValue::Char(c) => Value::Char(*c),
            // Signed integers
            GcValue::Int8(i) => Value::Int8(*i),
            GcValue::Int16(i) => Value::Int16(*i),
            GcValue::Int32(i) => Value::Int32(*i),
            GcValue::Int64(i) => Value::Int64(*i),
            // Unsigned integers
            GcValue::UInt8(i) => Value::UInt8(*i),
            GcValue::UInt16(i) => Value::UInt16(*i),
            GcValue::UInt32(i) => Value::UInt32(*i),
            GcValue::UInt64(i) => Value::UInt64(*i),
            // Floats
            GcValue::Float32(f) => Value::Float32(*f),
            GcValue::Float64(f) => Value::Float64(*f),
            // Decimal
            GcValue::Decimal(d) => Value::Decimal(*d),
            // BigInt
            GcValue::BigInt(ptr) => {
                let b = self.get_bigint(*ptr).expect("invalid bigint pointer");
                Value::BigInt(Arc::new(b.value.clone()))
            }

            // String
            GcValue::String(ptr) => {
                let s = self.get_string(*ptr).expect("invalid string pointer");
                Value::String(Arc::new(s.data.clone()))
            }

            // List
            GcValue::List(list) => {
                let items: Vec<Value> = list.items().iter().map(|v| self.gc_to_value(v)).collect();
                Value::List(Arc::new(items))
            }

            // Int64List - convert back to regular List
            GcValue::Int64List(list) => {
                let items: Vec<Value> = list.iter().map(Value::Int64).collect();
                Value::List(Arc::new(items))
            }

            // Array
            GcValue::Array(ptr) => {
                let arr = self.get_array(*ptr).expect("invalid array pointer");
                let items: Vec<Value> = arr.items.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Array(Arc::new(RwLock::new(items)))
            }

            // Typed arrays
            GcValue::Int64Array(ptr) => {
                let arr = self.get_int64_array(*ptr).expect("invalid int64 array pointer");
                Value::Int64Array(Arc::new(RwLock::new(arr.items.clone())))
            }
            GcValue::Float64Array(ptr) => {
                let arr = self.get_float64_array(*ptr).expect("invalid float64 array pointer");
                Value::Float64Array(Arc::new(RwLock::new(arr.items.clone())))
            }
            GcValue::Float32Array(ptr) => {
                let arr = self.get_float32_array(*ptr).expect("invalid float32 array pointer");
                Value::Float32Array(Arc::new(RwLock::new(arr.items.clone())))
            }

            // Tuple
            GcValue::Tuple(ptr) => {
                let tuple = self.get_tuple(*ptr).expect("invalid tuple pointer");
                let items: Vec<Value> = tuple.items.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Tuple(Arc::new(items))
            }

            // Map
            GcValue::Map(ptr) => {
                let map = self.get_map(*ptr).expect("invalid map pointer");
                let entries: HashMap<MapKey, Value> = map
                    .entries
                    .iter()
                    .map(|(k, v)| (self.gc_map_key_to_value(k), self.gc_to_value(v)))
                    .collect();
                Value::Map(Arc::new(entries))
            }

            // SharedMap - convert entries to Value::Map
            GcValue::SharedMap(map) => {
                let entries: HashMap<MapKey, Value> = map
                    .iter()
                    .map(|(k, v)| (shared_key_to_map_key(k), shared_value_to_value(v)))
                    .collect();
                Value::Map(Arc::new(entries))
            }

            // Set
            GcValue::Set(ptr) => {
                let set = self.get_set(*ptr).expect("invalid set pointer");
                let items: std::collections::HashSet<MapKey> =
                    set.items.iter().map(|k| self.gc_map_key_to_value(k)).collect();
                Value::Set(Arc::new(items))
            }

            // Record
            GcValue::Record(ptr) => {
                let record = self.get_record(*ptr).expect("invalid record pointer");
                let fields: Vec<Value> =
                    record.fields.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Record(Arc::new(RecordValue {
                    type_name: record.type_name.clone(),
                    field_names: record.field_names.clone(),
                    fields,
                    mutable_fields: record.mutable_fields.clone(),
                }))
            }

            // ReactiveRecord - just unwrap the Arc
            GcValue::ReactiveRecord(rec) => Value::ReactiveRecord(rec.clone()),

            // Variant
            GcValue::Variant(ptr) => {
                let variant = self.get_variant(*ptr).expect("invalid variant pointer");
                let fields: Vec<Value> =
                    variant.fields.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Variant(Arc::new(VariantValue {
                    type_name: Arc::clone(&variant.type_name),
                    constructor: Arc::clone(&variant.constructor),
                    fields,
                    named_fields: None,
                }))
            }

            // Closure - we can't fully reconstruct without the function
            GcValue::Closure(ptr, _) => {
                let closure = self.get_closure(*ptr).expect("invalid closure pointer");
                let captures: Vec<Value> =
                    closure.captures.iter().map(|v| self.gc_to_value(v)).collect();
                Value::Closure(Arc::new(ClosureValue {
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
            // Buffer - convert to string
            GcValue::Buffer(ptr) => {
                let s = self.get_buffer(*ptr)
                    .map(|b| b.to_string())
                    .unwrap_or_default();
                Value::String(Arc::new(s))
            }
            // Native handle - convert to Value::NativeHandle
            // Native handle - just clone the Arc
            GcValue::NativeHandle(h) => Value::NativeHandle(h.clone())
        }
    }

    /// Convert a GcMapKey back to a MapKey.
    fn gc_map_key_to_value(&self, key: &GcMapKey) -> MapKey {
        match key {
            GcMapKey::Unit => MapKey::Unit,
            GcMapKey::Bool(b) => MapKey::Bool(*b),
            GcMapKey::Char(c) => MapKey::Char(*c),
            // Signed integers
            GcMapKey::Int8(i) => MapKey::Int8(*i),
            GcMapKey::Int16(i) => MapKey::Int16(*i),
            GcMapKey::Int32(i) => MapKey::Int32(*i),
            GcMapKey::Int64(i) => MapKey::Int64(*i),
            // Unsigned integers
            GcMapKey::UInt8(i) => MapKey::UInt8(*i),
            GcMapKey::UInt16(i) => MapKey::UInt16(*i),
            GcMapKey::UInt32(i) => MapKey::UInt32(*i),
            GcMapKey::UInt64(i) => MapKey::UInt64(*i),
            // String
            GcMapKey::String(s) => {
                MapKey::String(Arc::new(s.clone()))
            }
            // Record
            GcMapKey::Record { type_name, field_names, fields } => {
                MapKey::Record {
                    type_name: type_name.clone(),
                    field_names: field_names.clone(),
                    fields: fields.iter().map(|f| self.gc_map_key_to_value(f)).collect(),
                }
            }
            // Variant
            GcMapKey::Variant { type_name, constructor, fields } => {
                MapKey::Variant {
                    type_name: type_name.clone(),
                    constructor: constructor.clone(),
                    fields: fields.iter().map(|f| self.gc_map_key_to_value(f)).collect(),
                }
            }
        }
    }
}

/// Convert a SharedMapKey to a Value::MapKey for returning results.
fn shared_key_to_map_key(key: &SharedMapKey) -> MapKey {
    match key {
        SharedMapKey::Unit => MapKey::Unit,
        SharedMapKey::Bool(b) => MapKey::Bool(*b),
        SharedMapKey::Char(c) => MapKey::Char(*c),
        SharedMapKey::Int8(n) => MapKey::Int8(*n),
        SharedMapKey::Int16(n) => MapKey::Int16(*n),
        SharedMapKey::Int32(n) => MapKey::Int32(*n),
        SharedMapKey::Int64(n) => MapKey::Int64(*n),
        SharedMapKey::UInt8(n) => MapKey::UInt8(*n),
        SharedMapKey::UInt16(n) => MapKey::UInt16(*n),
        SharedMapKey::UInt32(n) => MapKey::UInt32(*n),
        SharedMapKey::UInt64(n) => MapKey::UInt64(*n),
        SharedMapKey::String(s) => MapKey::String(Arc::new(s.clone())),
        SharedMapKey::Record { type_name, field_names, fields } => {
            MapKey::Record {
                type_name: type_name.clone(),
                field_names: field_names.clone(),
                fields: fields.iter().map(shared_key_to_map_key).collect(),
            }
        }
        SharedMapKey::Variant { type_name, constructor, fields } => {
            MapKey::Variant {
                type_name: type_name.clone(),
                constructor: constructor.clone(),
                fields: fields.iter().map(shared_key_to_map_key).collect(),
            }
        }
    }
}

/// Convert a SharedMapValue to a Value for returning results.
fn shared_value_to_value(value: &SharedMapValue) -> Value {
    match value {
        SharedMapValue::Unit => Value::Unit,
        SharedMapValue::Bool(b) => Value::Bool(*b),
        SharedMapValue::Char(c) => Value::Char(*c),
        SharedMapValue::Int64(n) => Value::Int64(*n),
        SharedMapValue::Float64(f) => Value::Float64(*f),
        SharedMapValue::Pid(p) => Value::Pid(Pid(*p)),
        SharedMapValue::String(s) => Value::String(Arc::new(s.clone())),
        SharedMapValue::List(items) => {
            let values: Vec<Value> = items.iter().map(shared_value_to_value).collect();
            Value::List(Arc::new(values))
        }
        SharedMapValue::Tuple(items) => {
            let values: Vec<Value> = items.iter().map(shared_value_to_value).collect();
            Value::Tuple(Arc::new(values))
        }
        SharedMapValue::Record { type_name, field_names, fields } => {
            let values: Vec<Value> = fields.iter().map(shared_value_to_value).collect();
            Value::Record(Arc::new(RecordValue {
                type_name: type_name.clone(),
                field_names: field_names.clone(),
                fields: values,
                mutable_fields: vec![false; field_names.len()],
            }))
        }
        SharedMapValue::Variant { type_name, constructor, fields } => {
            let values: Vec<Value> = fields.iter().map(shared_value_to_value).collect();
            Value::Variant(Arc::new(VariantValue {
                type_name: Arc::new(type_name.clone()),
                constructor: Arc::new(constructor.clone()),
                fields: values,
                named_fields: None,
            }))
        }
        SharedMapValue::Map(map) => {
            let entries: HashMap<MapKey, Value> = map
                .iter()
                .map(|(k, v)| (shared_key_to_map_key(k), shared_value_to_value(v)))
                .collect();
            Value::Map(Arc::new(entries))
        }
        SharedMapValue::Set(items) => {
            let keys: std::collections::HashSet<MapKey> = items
                .iter()
                .map(shared_key_to_map_key)
                .collect();
            Value::Set(Arc::new(keys))
        }
        SharedMapValue::Int64Array(items) => {
            // Convert to list of Int64 values
            let values: Vec<Value> = items.iter().map(|n| Value::Int64(*n)).collect();
            Value::List(Arc::new(values))
        }
        SharedMapValue::Float64Array(items) => {
            // Convert to list of Float64 values
            let values: Vec<Value> = items.iter().map(|f| Value::Float64(*f)).collect();
            Value::List(Arc::new(values))
        }
        SharedMapValue::Float32Array(items) => {
            // Convert to list of Float32 values
            let values: Vec<Value> = items.iter().map(|f| Value::Float32(*f)).collect();
            Value::List(Arc::new(values))
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
        let items = vec![GcValue::Int64(1), GcValue::Int64(2), GcValue::Int64(3)];
        let list = heap.make_list(items);

        assert_eq!(list.len(), 3);
        assert!(matches!(list.items()[0], GcValue::Int64(1)));
        assert!(matches!(list.items()[1], GcValue::Int64(2)));
        assert!(matches!(list.items()[2], GcValue::Int64(3)));
    }

    #[test]
    fn test_alloc_nested_list() {
        let mut heap = Heap::new();

        // Create inner list [1, 2]
        let inner = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(2)]);

        // Create outer list [[1, 2], 3]
        let outer = heap.make_list(vec![GcValue::List(inner.clone()), GcValue::Int64(3)]);

        assert_eq!(outer.len(), 2);

        if let GcValue::List(inner_list) = &outer.items()[0] {
            assert_eq!(inner_list.len(), 2);
        } else {
            panic!("Expected inner list");
        }
    }

    #[test]
    fn test_alloc_array() {
        let mut heap = Heap::new();
        let items = vec![GcValue::Float64(1.5), GcValue::Float64(2.5)];
        let ptr = heap.alloc_array(items);

        let arr = heap.get_array(ptr).unwrap();
        assert_eq!(arr.items.len(), 2);
    }

    #[test]
    fn test_alloc_tuple() {
        let mut heap = Heap::new();
        let s = heap.alloc_string("hello".to_string());
        let items = vec![GcValue::Int64(42), GcValue::String(s), GcValue::Bool(true)];
        let ptr = heap.alloc_tuple(items);

        let tuple = heap.get_tuple(ptr).unwrap();
        assert_eq!(tuple.items.len(), 3);
    }

    #[test]
    fn test_alloc_map() {
        let mut heap = Heap::new();
        let mut entries = ImblHashMap::new();
        entries.insert(GcMapKey::Int64(1), GcValue::Bool(true));
        entries.insert(GcMapKey::Int64(2), GcValue::Bool(false));
        let ptr = heap.alloc_map(entries);

        let map = heap.get_map(ptr).unwrap();
        assert_eq!(map.entries.len(), 2);
        assert_eq!(map.entries.get(&GcMapKey::Int64(1)), Some(&GcValue::Bool(true)));
    }

    #[test]
    fn test_alloc_set() {
        let mut heap = Heap::new();
        let mut items = ImblHashSet::new();
        items.insert(GcMapKey::Int64(1));
        items.insert(GcMapKey::Int64(2));
        items.insert(GcMapKey::Int64(3));
        let ptr = heap.alloc_set(items);

        let set = heap.get_set(ptr).unwrap();
        assert_eq!(set.items.len(), 3);
        assert!(set.items.contains(&GcMapKey::Int64(1)));
    }

    #[test]
    fn test_alloc_record() {
        let mut heap = Heap::new();
        let ptr = heap.alloc_record(
            "Point".to_string(),
            vec!["x".to_string(), "y".to_string()],
            vec![GcValue::Int64(10), GcValue::Int64(20)],
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
            Arc::new("Option".to_string()),
            Arc::new("Some".to_string()),
            vec![GcValue::Int64(42)],
        );

        let var = heap.get_variant(ptr).unwrap();
        assert_eq!(&*var.type_name, "Option");
        assert_eq!(&*var.constructor, "Some");
        assert_eq!(var.fields.len(), 1);
    }

    #[test]
    fn test_alloc_closure() {
        use crate::value::{FunctionValue, Chunk};
        use std::rc::Rc;

        let mut heap = Heap::new();

        // Create a test function
        let func = Arc::new(FunctionValue {
            name: "test_closure".to_string(),
            arity: 2,
            param_names: vec!["x".to_string(), "y".to_string()],
            code: Arc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: std::sync::atomic::AtomicU32::new(0),
            debug_symbols: vec![],
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None, required_params: None,
        });

        let ptr = heap.alloc_closure(
            func,
            vec![GcValue::Int64(10), GcValue::Int64(20)],
            vec!["x".to_string(), "y".to_string()],
        );

        let clo = heap.get_closure(ptr).unwrap();
        assert_eq!(clo.function.name, "test_closure");
        assert_eq!(clo.function.arity, 2);
        assert_eq!(clo.captures.len(), 2);
    }

    // ============================================================
    // Typed Array Tests
    // ============================================================

    #[test]
    fn test_alloc_int64_array() {
        let mut heap = Heap::new();
        let items = vec![1i64, 2, 3, 4, 5];
        let ptr = heap.alloc_int64_array(items);

        let arr = heap.get_int64_array(ptr).unwrap();
        assert_eq!(arr.items.len(), 5);
        assert_eq!(arr.items[0], 1);
        assert_eq!(arr.items[4], 5);
    }

    #[test]
    fn test_alloc_float64_array() {
        let mut heap = Heap::new();
        let items = vec![1.5f64, 2.5, 3.5];
        let ptr = heap.alloc_float64_array(items);

        let arr = heap.get_float64_array(ptr).unwrap();
        assert_eq!(arr.items.len(), 3);
        assert!((arr.items[0] - 1.5).abs() < f64::EPSILON);
        assert!((arr.items[2] - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_int64_array_mutation() {
        let mut heap = Heap::new();
        let items = vec![0i64; 10];
        let ptr = heap.alloc_int64_array(items);

        // Mutate the array
        {
            let arr = heap.get_int64_array_mut(ptr).unwrap();
            for i in 0..10 {
                arr.items[i] = (i * i) as i64;
            }
        }

        // Verify mutations
        let arr = heap.get_int64_array(ptr).unwrap();
        assert_eq!(arr.items[0], 0);
        assert_eq!(arr.items[1], 1);
        assert_eq!(arr.items[2], 4);
        assert_eq!(arr.items[3], 9);
        assert_eq!(arr.items[9], 81);
    }

    #[test]
    fn test_float64_array_mutation() {
        let mut heap = Heap::new();
        let items = vec![0.0f64; 5];
        let ptr = heap.alloc_float64_array(items);

        // Mutate the array
        {
            let arr = heap.get_float64_array_mut(ptr).unwrap();
            arr.items[0] = 3.14;
            arr.items[1] = 2.71;
            arr.items[2] = 1.41;
        }

        // Verify mutations
        let arr = heap.get_float64_array(ptr).unwrap();
        assert!((arr.items[0] - 3.14).abs() < f64::EPSILON);
        assert!((arr.items[1] - 2.71).abs() < f64::EPSILON);
        assert!((arr.items[2] - 1.41).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_typed_arrays() {
        let mut heap = Heap::new();

        let int_ptr = heap.alloc_int64_array(vec![]);
        let float_ptr = heap.alloc_float64_array(vec![]);

        assert_eq!(heap.get_int64_array(int_ptr).unwrap().items.len(), 0);
        assert_eq!(heap.get_float64_array(float_ptr).unwrap().items.len(), 0);
    }

    #[test]
    fn test_typed_array_gc_survives_when_rooted() {
        let mut heap = Heap::new();

        let int_arr = heap.alloc_int64_array(vec![1, 2, 3]);
        let float_arr = heap.alloc_float64_array(vec![1.0, 2.0]);
        let _garbage = heap.alloc_string("garbage".to_string());

        // Root the typed arrays
        heap.add_root(int_arr.as_raw());
        heap.add_root(float_arr.as_raw());

        assert_eq!(heap.live_objects(), 3);
        heap.collect();
        assert_eq!(heap.live_objects(), 2);

        // Verify data is intact
        assert_eq!(heap.get_int64_array(int_arr).unwrap().items, vec![1, 2, 3]);
        assert_eq!(heap.get_float64_array(float_arr).unwrap().items, vec![1.0, 2.0]);
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
        let list = heap.make_list(vec![GcValue::String(s1), GcValue::String(s2)]);

        // Also create some garbage
        let _garbage = heap.alloc_string("garbage".to_string());

        // NOTE: Lists are now inline (no GC pointer), so we need to root the contained strings
        // We'll root s1 and s2 directly instead
        heap.add_root(s1.as_raw());
        heap.add_root(s2.as_raw());

        assert_eq!(heap.live_objects(), 3);
        heap.collect();

        // Strings should survive, garbage should not
        assert_eq!(heap.live_objects(), 2);

        // Verify data is intact
        if let GcValue::String(ptr) = &list.items()[0] {
            assert_eq!(heap.get_string(*ptr).unwrap().data, "item1");
        }
    }

    #[test]
    fn test_gc_deeply_nested() {
        let mut heap = Heap::new();

        // Create deeply nested structure
        let s = heap.alloc_string("deep".to_string());
        let l1 = heap.make_list(vec![GcValue::String(s)]);
        let l2 = heap.make_list(vec![GcValue::List(l1.clone())]);
        let l3 = heap.make_list(vec![GcValue::List(l2.clone())]);
        let l4 = heap.make_list(vec![GcValue::List(l3.clone())]);

        // Add garbage at various levels
        let _g1 = heap.alloc_string("garbage1".to_string());
        let _g2 = heap.alloc_string("garbage2".to_string());

        // Lists are inline, so we root the string directly
        heap.add_root(s.as_raw());

        assert_eq!(heap.live_objects(), 3);
        heap.collect();
        assert_eq!(heap.live_objects(), 1); // only s survives (lists are inline)
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
        let _arr = heap.alloc_array(vec![GcValue::Int64(1)]);

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
            vec![GcValue::String(name), GcValue::Int64(30)],
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
            Arc::new("Result".to_string()),
            Arc::new("Ok".to_string()),
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
        let func = Arc::new(FunctionValue {
            name: "test_closure".to_string(),
            arity: 2,
            param_names: vec!["x".to_string(), "y".to_string()],
            code: Arc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: std::sync::atomic::AtomicU32::new(0),
            debug_symbols: vec![],
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None, required_params: None,
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

        let value = heap.alloc_string("my_value".to_string());

        let mut entries = ImblHashMap::new();
        // GcMapKey::String stores inline String for proper hashing
        entries.insert(GcMapKey::String("my_key".to_string()), GcValue::String(value));
        entries.insert(GcMapKey::Int64(42), GcValue::Bool(true));

        let map = heap.alloc_map(entries);

        let _garbage = heap.alloc_string("garbage".to_string());

        heap.add_root(map.as_raw());
        heap.collect();

        assert_eq!(heap.live_objects(), 2); // map + value string
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

        let value = GcValue::Int64(42);
        let copied = heap2.deep_copy(&value, &heap1);

        assert!(matches!(copied, GcValue::Int64(42)));
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
        let inner = heap1.make_list(vec![GcValue::String(s), GcValue::Int64(42)]);
        let outer = heap1.make_list(vec![GcValue::List(inner), GcValue::Bool(true)]);

        let value = GcValue::List(outer);
        let copied = heap2.deep_copy(&value, &heap1);

        if let GcValue::List(new_outer) = copied {
            assert_eq!(new_outer.len(), 2);

            if let GcValue::List(new_inner) = &new_outer.items()[0] {
                assert_eq!(new_inner.len(), 2);

                if let GcValue::String(new_s) = &new_inner.items()[0] {
                    assert_eq!(heap2.get_string(*new_s).unwrap().data, "nested");
                }
            }
        } else {
            panic!("Expected list");
        }

        // Heap2 should have: string = 1 object (lists are inline, not on heap)
        assert_eq!(heap2.live_objects(), 1);
    }

    #[test]
    fn test_deep_copy_record() {
        let mut heap1 = Heap::new();
        let mut heap2 = Heap::new();

        let name = heap1.alloc_string("Bob".to_string());
        let person = heap1.alloc_record(
            "Person".to_string(),
            vec!["name".to_string(), "age".to_string()],
            vec![GcValue::String(name), GcValue::Int64(25)],
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
        let func = Arc::new(FunctionValue {
            name: "test_func".to_string(),
            arity: 2,
            param_names: vec!["x".to_string(), "y".to_string()],
            code: Arc::new(Chunk::new()),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: std::sync::atomic::AtomicU32::new(0),
            debug_symbols: vec![],
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None, required_params: None,
        });

        let captured = heap1.alloc_string("captured_value".to_string());
        let closure = heap1.alloc_closure(
            func.clone(),
            vec![GcValue::String(captured), GcValue::Int64(10)],
            vec!["x".to_string(), "y".to_string()],
        );

        let value = GcValue::Closure(closure, InlineOp::None);
        let copied = heap2.deep_copy(&value, &heap1);

        if let GcValue::Closure(new_closure, _) = copied {
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
        assert!(GcValue::Int64(42).is_immediate());
        assert!(GcValue::Float64(3.14).is_immediate());
        assert!(GcValue::Char('x').is_immediate());
        assert!(GcValue::Pid(1).is_immediate());
        assert!(GcValue::Ref(2).is_immediate());

        let ptr: GcPtr<GcString> = GcPtr::from_raw(0);
        assert!(!GcValue::String(ptr).is_immediate());
    }

    #[test]
    fn test_gc_value_gc_pointers() {
        assert!(GcValue::Int64(42).gc_pointers().is_empty());

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

        // Lists are no longer stored in HeapData - they're inline in GcValue
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
        let mut current = heap.make_list(vec![GcValue::Int64(42)]);

        for _ in 0..100 {
            current = heap.make_list(vec![GcValue::List(current)]);
        }

        let _garbage = heap.alloc_string("garbage".to_string());

        // Lists are inline, so we need to root the value itself
        // We can't call add_root on a GcList, so let's just verify collection works
        heap.collect();

        // Should have no live objects (lists are inline, garbage was collected)
        assert_eq!(heap.live_objects(), 0);
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

        let _list = heap.make_list(items);

        // Add some garbage
        for i in 0..500 {
            let _g = heap.alloc_string(format!("garbage{}", i));
        }

        // Lists are inline, so we can't root them directly
        // Without rooting, all strings become garbage
        heap.collect();

        // All strings (both in list and garbage) should be collected
        assert_eq!(heap.live_objects(), 0);
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
        assert!(heap.gc_values_equal(&GcValue::Int64(42), &GcValue::Int64(42)));
        assert!(!heap.gc_values_equal(&GcValue::Int64(42), &GcValue::Int64(43)));

        // Float
        assert!(heap.gc_values_equal(&GcValue::Float64(3.14), &GcValue::Float64(3.14)));
        assert!(!heap.gc_values_equal(&GcValue::Float64(3.14), &GcValue::Float64(2.71)));

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
        assert!(!heap.gc_values_equal(&GcValue::Int64(42), &GcValue::Bool(true)));
        assert!(!heap.gc_values_equal(&GcValue::Unit, &GcValue::Bool(false)));
        assert!(!heap.gc_values_equal(&GcValue::Float64(42.0), &GcValue::Int64(42)));
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

        // Create two lists with same content (inline, no heap allocation)
        let list1 = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(2), GcValue::Int64(3)]);
        let list2 = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(2), GcValue::Int64(3)]);

        let v1 = GcValue::List(list1.clone());
        let v2 = GcValue::List(list2.clone());

        // Same content - should be equal
        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_list_different_content() {
        let mut heap = Heap::new();

        let list1 = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(2)]);
        let list2 = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(3)]);

        let v1 = GcValue::List(list1);
        let v2 = GcValue::List(list2);

        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_list_different_length() {
        let mut heap = Heap::new();

        let list1 = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(2)]);
        let list2 = heap.make_list(vec![GcValue::Int64(1), GcValue::Int64(2), GcValue::Int64(3)]);

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
        let inner1 = heap.make_list(vec![GcValue::String(s1a), GcValue::String(s1b)]);
        let outer1 = heap.make_list(vec![GcValue::List(inner1), GcValue::Int64(42)]);

        let s2a = heap.alloc_string("hello".to_string());
        let s2b = heap.alloc_string("world".to_string());
        let inner2 = heap.make_list(vec![GcValue::String(s2a), GcValue::String(s2b)]);
        let outer2 = heap.make_list(vec![GcValue::List(inner2), GcValue::Int64(42)]);

        let v1 = GcValue::List(outer1);
        let v2 = GcValue::List(outer2);

        // Should be equal despite different allocations
        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_tuple() {
        let mut heap = Heap::new();

        let s1 = heap.alloc_string("test".to_string());
        let tuple1 = heap.alloc_tuple(vec![GcValue::Int64(1), GcValue::String(s1), GcValue::Bool(true)]);

        let s2 = heap.alloc_string("test".to_string());
        let tuple2 = heap.alloc_tuple(vec![GcValue::Int64(1), GcValue::String(s2), GcValue::Bool(true)]);

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
            vec![GcValue::String(name1), GcValue::Int64(30)],
            vec![false, false],
        );

        let name2 = heap.alloc_string("Alice".to_string());
        let rec2 = heap.alloc_record(
            "Person".to_string(),
            vec!["name".to_string(), "age".to_string()],
            vec![GcValue::String(name2), GcValue::Int64(30)],
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
            vec![GcValue::Int64(1)],
            vec![false],
        );

        let rec2 = heap.alloc_record(
            "User".to_string(),
            vec!["name".to_string()],
            vec![GcValue::Int64(1)],
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

        let var1 = heap.alloc_variant(Arc::new("Option".to_string()), Arc::new("Some".to_string()), vec![GcValue::Int64(42)]);
        let var2 = heap.alloc_variant(Arc::new("Option".to_string()), Arc::new("Some".to_string()), vec![GcValue::Int64(42)]);

        let v1 = GcValue::Variant(var1);
        let v2 = GcValue::Variant(var2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_variant_different_constructor() {
        let mut heap = Heap::new();

        let var1 = heap.alloc_variant(Arc::new("Option".to_string()), Arc::new("Some".to_string()), vec![GcValue::Int64(42)]);
        let var2 = heap.alloc_variant(Arc::new("Option".to_string()), Arc::new("None".to_string()), vec![]);

        let v1 = GcValue::Variant(var1);
        let v2 = GcValue::Variant(var2);

        assert!(!heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_array() {
        let mut heap = Heap::new();

        let arr1 = heap.alloc_array(vec![GcValue::Float64(1.0), GcValue::Float64(2.0)]);
        let arr2 = heap.alloc_array(vec![GcValue::Float64(1.0), GcValue::Float64(2.0)]);

        let v1 = GcValue::Array(arr1);
        let v2 = GcValue::Array(arr2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }

    #[test]
    fn test_gc_values_equal_empty_list() {
        let mut heap = Heap::new();

        let list1 = heap.make_list(vec![]);
        let list2 = heap.make_list(vec![]);

        let v1 = GcValue::List(list1);
        let v2 = GcValue::List(list2);

        assert!(heap.gc_values_equal(&v1, &v2));
    }
}
