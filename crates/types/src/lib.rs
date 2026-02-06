//! Type system for the Nostos programming language.
//!
//! This module provides:
//! - Type representation (Type enum)
//! - Type environment (TypeEnv)
//! - Type inference (infer module)
//! - Type checking (check module)
//! - Type errors (TypeError)

use std::collections::HashMap;
use thiserror::Error;

pub mod infer;
pub mod check;
pub mod mono;

/// Unique identifier for type variables during inference.
pub type TypeVarId = u32;

/// A type in the Nostos type system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    // === Primitives ===
    // Signed integers
    Int8,
    Int16,
    Int32,
    Int64,
    Int, // Alias for Int64
    // Unsigned integers
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    // Floating point
    Float32,
    Float64,
    Float, // Alias for Float64
    // Arbitrary precision
    BigInt,
    Decimal,
    // Other primitives
    Bool,
    Char,
    String,
    Unit,
    Never, // Bottom type (functions that don't return)

    // === Type Variables (for inference and generics) ===
    /// Unification variable (gets resolved during inference)
    Var(TypeVarId),
    /// Named type parameter (e.g., T in List[T])
    TypeParam(String),

    // === Compound Types ===
    /// Tuple: (Int, String, Bool)
    Tuple(Vec<Type>),
    /// List: List[T]
    List(Box<Type>),
    /// Array: Array[T]
    Array(Box<Type>),
    /// Map: Map[K, V]
    Map(Box<Type>, Box<Type>),
    /// Set: Set[T]
    Set(Box<Type>),

    // === Structural Types ===
    /// Record: {x: Int, y: Float}
    Record(RecordType),
    /// Variant/Sum type: None | Some(T)
    Variant(VariantType),

    // === Function Types ===
    /// Function: (Int, Int) -> Int
    Function(FunctionType),

    // === Named Types ===
    /// Reference to a named type: Point, Option[Int], Result[T, E]
    Named {
        name: String,
        args: Vec<Type>,
    },

    // === Special ===
    /// IO monad wrapper: IO[T]
    IO(Box<Type>),
    /// Process ID type
    Pid,
    /// Reference type (for monitors)
    Ref,
}

/// A record type with named fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordType {
    /// Name of the record type (if named, e.g., Point)
    pub name: Option<String>,
    /// Fields with their types
    pub fields: Vec<(String, Type, bool)>, // (name, type, is_mutable)
}

/// A variant (sum) type with multiple constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantType {
    /// Name of the variant type (e.g., Option, Result)
    pub name: String,
    /// Type parameters
    pub params: Vec<String>,
    /// Constructors
    pub constructors: Vec<Constructor>,
}

/// A constructor in a variant type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constructor {
    /// Unit constructor: None, Nil
    Unit(String),
    /// Positional fields: Some(T), Cons(T, List[T])
    Positional(String, Vec<Type>),
    /// Named fields: Circle{radius: Float}
    Named(String, Vec<(String, Type)>),
}

/// A function type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    /// Type parameters with their constraints
    pub type_params: Vec<TypeParam>,
    /// Parameter types
    pub params: Vec<Type>,
    /// Return type
    pub ret: Box<Type>,
    /// Number of required parameters (without defaults).
    /// If None, all parameters are required.
    pub required_params: Option<usize>,
}

impl FunctionType {
    /// Find the maximum type variable ID in this function type, or None if no type variables.
    pub fn max_var_id(&self) -> Option<TypeVarId> {
        let param_max = self.params.iter().filter_map(|t| t.max_var_id()).max();
        let ret_max = self.ret.max_var_id();
        match (param_max, ret_max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

/// A type parameter with optional constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeParam {
    pub name: String,
    /// Trait constraints: T: Eq + Show
    pub constraints: Vec<String>,
}

/// A trait definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDef {
    pub name: String,
    /// Supertrait requirements: Ord: Eq
    pub supertraits: Vec<String>,
    /// Required methods (no default impl)
    pub required: Vec<TraitMethod>,
    /// Methods with default implementations
    pub defaults: Vec<TraitMethod>,
}

/// A method in a trait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitMethod {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret: Type,
}

/// A trait implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitImpl {
    /// The trait being implemented
    pub trait_name: String,
    /// The type implementing the trait
    pub for_type: Type,
    /// Constraints on type parameters
    pub constraints: Vec<(String, Vec<String>)>,
}

/// Type checking errors.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TypeError {
    #[error("type mismatch: expected {expected}, found {found}")]
    Mismatch { expected: String, found: String },

    #[error("Unknown identifier: {0}")]
    UnknownIdent(String),

    #[error("Unknown type: {0}")]
    UnknownType(String),

    #[error("Unknown constructor: {0}")]
    UnknownConstructor(String),

    #[error("Wrong number of arguments: expected {expected}, found {found}")]
    ArityMismatch { expected: usize, found: usize },

    #[error("Wrong number of type arguments: expected {expected}, found {found}")]
    TypeArityMismatch { expected: usize, found: usize },

    #[error("Cannot unify types: {0} and {1}")]
    UnificationFailed(String, String),

    #[error("Infinite type: {0} contains itself")]
    InfiniteType(String),

    #[error("Missing trait implementation: {ty} does not implement {trait_name}")]
    MissingTraitImpl { ty: String, trait_name: String },

    #[error("Missing trait method: {method} not implemented for {ty}")]
    MissingTraitMethod { ty: String, method: String },

    #[error("Non-exhaustive patterns: {missing:?} not covered")]
    NonExhaustive { missing: Vec<String> },

    #[error("Duplicate field: {0}")]
    DuplicateField(String),

    #[error("Missing field: {0}")]
    MissingField(String),

    #[error("Extra field: {0}")]
    ExtraField(String),

    #[error("Field {field} is not mutable")]
    ImmutableField { field: String },

    #[error("Cannot modify immutable binding: {0}")]
    ImmutableBinding(String),

    #[error("Type {0} is not callable")]
    NotCallable(String),

    #[error("Type {0} is not indexable")]
    NotIndexable(String),

    #[error("Type {ty} has no field {field}")]
    NoSuchField { ty: String, field: String },

    #[error("Ambiguous type: cannot infer type for {0}")]
    AmbiguousType(String),

    #[error("Type {receiver_type} has no method {method}")]
    UndefinedMethod { method: String, receiver_type: String },

    #[error("Recursive type without indirection: {0}")]
    InvalidRecursiveType(String),

    #[error("Private field {field} cannot be accessed outside module {module}")]
    PrivateField { field: String, module: String },

    #[error("Cannot send type {0} between processes (not serializable)")]
    NotSerializable(String),

    #[error("Unreachable pattern: {0}")]
    UnreachablePattern(String),

    #[error("Cannot coerce {from} to {to}")]
    InvalidCoercion { from: String, to: String },

    #[error("Type mismatch: cannot unify a type with a type containing itself ({0} in {1})")]
    OccursCheck(String, String),

    #[error("Wildcard '_' is only valid in pattern contexts, not as a standalone expression")]
    InvalidWildcard(nostos_syntax::Span),

    #[error("Type annotation required: {param} must have the same type, but used as {type1} and {type2}")]
    AnnotationRequired {
        func: String,
        param: String,
        type1: String,
        type2: String,
    },

    #[error("Type mismatch in argument {arg_index} of `{function_name}`: expected `{expected}`, found `{found}`")]
    ArgumentTypeMismatch {
        function_name: String,
        arg_index: usize,
        expected: String,
        found: String,
    },
}

/// A type error with optional source span for precise error reporting.
#[derive(Debug, Clone)]
pub struct TypeErrorWithSpan {
    pub error: TypeError,
    pub span: Option<nostos_syntax::Span>,
}

impl TypeErrorWithSpan {
    pub fn new(error: TypeError, span: Option<nostos_syntax::Span>) -> Self {
        Self { error, span }
    }

    pub fn without_span(error: TypeError) -> Self {
        Self { error, span: None }
    }
}

impl std::fmt::Display for TypeErrorWithSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for TypeErrorWithSpan {}

/// Type environment for tracking bindings and definitions.
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    /// Variable bindings: name -> (type, is_mutable)
    pub bindings: HashMap<String, (Type, bool)>,
    /// Type definitions: name -> TypeDef
    pub types: HashMap<String, TypeDef>,
    /// Type aliases: short name -> qualified name (e.g., "Option" -> "stdlib.list.Option")
    pub type_aliases: HashMap<String, String>,
    /// Trait definitions: name -> TraitDef
    pub traits: HashMap<String, TraitDef>,
    /// Trait implementations
    pub impls: Vec<TraitImpl>,
    /// Function signatures: name -> FunctionType
    pub functions: HashMap<String, FunctionType>,
    /// Index: base function name -> all function keys with that base (for O(1) lookups)
    pub functions_by_base: HashMap<String, std::collections::HashSet<String>>,
    /// Function aliases: short name -> qualified name (e.g., "query" -> "stdlib.pool.query")
    /// Used to resolve imports during type inference
    pub function_aliases: HashMap<String, String>,
    /// Function parameter names: function name -> list of param names.
    /// Used for matching named arguments in UFCS method calls.
    pub function_param_names: HashMap<String, Vec<String>>,
    /// Current module name
    pub current_module: Option<String>,
    /// Type variable counter for fresh variables
    pub next_var: TypeVarId,
    /// Substitution map for type inference
    pub substitution: HashMap<TypeVarId, Type>,
}

/// A type definition (record, variant, alias).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDef {
    /// Record type definition
    Record {
        params: Vec<TypeParam>,
        fields: Vec<(String, Type, bool)>,
        is_mutable: bool,
    },
    /// Variant type definition
    Variant {
        params: Vec<TypeParam>,
        constructors: Vec<Constructor>,
    },
    /// Type alias
    Alias {
        params: Vec<TypeParam>,
        target: Type,
    },
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a fresh type variable.
    pub fn fresh_var(&mut self) -> Type {
        let id = self.next_var;
        self.next_var += 1;
        Type::Var(id)
    }

    /// Look up a binding.
    pub fn lookup(&self, name: &str) -> Option<&(Type, bool)> {
        self.bindings.get(name)
    }

    /// Add a binding.
    pub fn bind(&mut self, name: String, ty: Type, mutable: bool) {
        self.bindings.insert(name, (ty, mutable));
    }

    /// Look up a type definition.
    /// First checks for type aliases, then looks up the resolved name.
    pub fn lookup_type(&self, name: &str) -> Option<&TypeDef> {
        // First try direct lookup
        if let Some(def) = self.types.get(name) {
            return Some(def);
        }
        // Try resolving through aliases
        if let Some(qualified) = self.type_aliases.get(name) {
            return self.types.get(qualified);
        }
        None
    }

    /// Resolve a type name through aliases.
    /// Returns the qualified name if an alias exists, otherwise returns the original name.
    pub fn resolve_type_name(&self, name: &str) -> String {
        self.type_aliases.get(name).cloned().unwrap_or_else(|| name.to_string())
    }

    /// Add a type alias (e.g., "Option" -> "stdlib.list.Option").
    pub fn add_type_alias(&mut self, short_name: String, qualified_name: String) {
        self.type_aliases.insert(short_name, qualified_name);
    }

    /// Insert a function and update the index for O(1) base name lookups.
    pub fn insert_function(&mut self, name: String, func_type: FunctionType) {
        // Extract base name (before the /) for indexing
        let base_name = name.split('/').next().unwrap_or(&name).to_string();
        self.functions_by_base
            .entry(base_name)
            .or_default()
            .insert(name.clone());
        self.functions.insert(name, func_type);
    }

    /// Look up a function by name with arity-aware resolution.
    /// First tries arity-qualified name (e.g., `foo/_` for 1 arg), then falls back to exact match.
    /// Also handles optional parameters by trying higher arities.
    /// For typed overloads (e.g., `add/Int,Int`), searches all functions with matching prefix.
    /// This is essential for resolving overloaded functions by call arity.
    pub fn lookup_function_with_arity(&self, name: &str, arity: usize) -> Option<&FunctionType> {
        // Don't apply arity resolution if name already has a slash (already qualified)
        if !name.contains('/') {
            // First try exact arity match with wildcards
            let arity_suffix = if arity == 0 {
                "/".to_string()
            } else {
                format!("/{}", vec!["_"; arity].join(","))
            };
            let qualified_name = format!("{}{}", name, arity_suffix);
            if let Some(ft) = self.functions.get(&qualified_name) {
                return Some(ft);
            }

            // Try higher arities for functions with optional params
            // (provided arity >= required_params)
            for extra in 1..=10 {
                let total_arity = arity + extra;
                let arity_suffix = format!("/{}", vec!["_"; total_arity].join(","));
                let qualified_name = format!("{}{}", name, arity_suffix);
                if let Some(ft) = self.functions.get(&qualified_name) {
                    // Check if this function accepts our arity (has enough defaults)
                    let min_required = ft.required_params.unwrap_or(ft.params.len());
                    if arity >= min_required && arity <= ft.params.len() {
                        return Some(ft);
                    }
                }
            }

            // Try to find ANY typed overload with matching prefix and arity
            // This handles cases like `add/Int,Int` and `add/String,String`
            // Use O(1) index lookup instead of iterating all functions
            let prefix = format!("{}/", name);
            if let Some(keys) = self.functions_by_base.get(name) {
                for fn_name in keys {
                    if let Some(ft) = self.functions.get(fn_name) {
                        if fn_name.starts_with(&prefix) && ft.params.len() == arity {
                            // Check that this isn't a wildcard entry (those were checked above)
                            // Wildcard entries have "_" in the suffix
                            let suffix = &fn_name[prefix.len()..];
                            if !suffix.contains('_') || suffix.split(',').any(|p| p != "_") {
                                return Some(ft);
                            }
                        }
                    }
                }
            }
        }
        // Fall back to exact match
        self.functions.get(name)
    }

    /// Look up ALL functions matching name and arity (for overload resolution).
    /// Returns all typed overloads like `add/Int,Int` and `add/String,String`.
    /// The caller should try each one to find the best match for argument types.
    pub fn lookup_all_functions_with_arity(&self, name: &str, arity: usize) -> Vec<&FunctionType> {
        let mut results = Vec::new();

        // Try to resolve the name via function_aliases (imports)
        let resolved_name = if !name.contains('.') {
            if let Some(qualified) = self.function_aliases.get(name) {
                qualified.as_str()
            } else {
                name
            }
        } else {
            name
        };

        if !resolved_name.contains('/') {
            // First check wildcard entry (untyped overload)
            let arity_suffix = if arity == 0 {
                "/".to_string()
            } else {
                format!("/{}", vec!["_"; arity].join(","))
            };
            let qualified_name = format!("{}{}", resolved_name, arity_suffix);
            if let Some(ft) = self.functions.get(&qualified_name) {
                results.push(ft);
            }

            // Check higher arities for functions with optional params
            for extra in 1..=10 {
                let total_arity = arity + extra;
                let arity_suffix = format!("/{}", vec!["_"; total_arity].join(","));
                let qualified_name = format!("{}{}", resolved_name, arity_suffix);
                if let Some(ft) = self.functions.get(&qualified_name) {
                    let min_required = ft.required_params.unwrap_or(ft.params.len());
                    if arity >= min_required && arity <= ft.params.len() {
                        results.push(ft);
                    }
                }
            }

            // Collect ALL typed overloads with matching prefix and arity
            // Use O(1) index lookup instead of iterating all functions
            let prefix = format!("{}/", resolved_name);
            if let Some(keys) = self.functions_by_base.get(resolved_name) {
                for fn_name in keys {
                    if let Some(ft) = self.functions.get(fn_name) {
                        if fn_name.starts_with(&prefix) && ft.params.len() == arity {
                            let suffix = &fn_name[prefix.len()..];
                            // Only include typed entries (not wildcard entries already checked)
                            if !suffix.chars().all(|c| c == '_' || c == ',') {
                                results.push(ft);
                            }
                        }
                    }
                }
            }
        }

        // Check exact match - but only if arity is compatible
        if let Some(ft) = self.functions.get(resolved_name) {
            if !results.contains(&ft) {
                // Only add if the provided arity matches the function's parameter count
                // or if the arity is within the valid range for optional parameters
                let min_required = ft.required_params.unwrap_or(ft.params.len());
                if arity >= min_required && arity <= ft.params.len() {
                    results.push(ft);
                }
            }
        }

        results
    }

    /// Look up a function by name, ignoring arity.
    /// Used to check if a function exists even when called with wrong number of args.
    pub fn lookup_function_any_arity(&self, name: &str) -> Option<&FunctionType> {
        // Try to resolve the name via function_aliases (imports)
        let resolved_name = if !name.contains('.') {
            if let Some(qualified) = self.function_aliases.get(name) {
                qualified.as_str()
            } else {
                name
            }
        } else {
            name
        };

        if !resolved_name.contains('/') {
            // Try to find any function with this base name
            if let Some(keys) = self.functions_by_base.get(resolved_name) {
                for fn_name in keys {
                    if let Some(ft) = self.functions.get(fn_name) {
                        return Some(ft);
                    }
                }
            }
        }
        // Fall back to exact match
        self.functions.get(resolved_name)
    }

    /// Look up the field types for a variant constructor.
    /// Given a type name (e.g., "Expr") and constructor name (e.g., "Add"),
    /// returns the concrete field types from the type definition.
    pub fn lookup_variant_field_types(&self, type_name: &str, ctor_name: &str) -> Option<Vec<Type>> {
        if let Some(TypeDef::Variant { constructors, .. }) = self.types.get(type_name) {
            for ctor in constructors {
                match ctor {
                    Constructor::Unit(name) if name == ctor_name => {
                        return Some(vec![]);
                    }
                    Constructor::Positional(name, field_types) if name == ctor_name => {
                        return Some(field_types.clone());
                    }
                    Constructor::Named(name, fields) if name == ctor_name => {
                        return Some(fields.iter().map(|(_, ty)| ty.clone()).collect());
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Given a type name and constructor name, returns the named field info (name, type) pairs.
    /// Returns None for unit or positional constructors.
    pub fn lookup_variant_named_fields(&self, type_name: &str, ctor_name: &str) -> Option<Vec<(String, Type)>> {
        if let Some(TypeDef::Variant { constructors, .. }) = self.types.get(type_name) {
            for ctor in constructors {
                if let Constructor::Named(name, fields) = ctor {
                    if name == ctor_name {
                        return Some(fields.clone());
                    }
                }
            }
        }
        None
    }

    /// Add a type definition.
    pub fn define_type(&mut self, name: String, def: TypeDef) {
        self.types.insert(name, def);
    }

    /// Look up a trait definition.
    pub fn lookup_trait(&self, name: &str) -> Option<&TraitDef> {
        self.traits.get(name)
    }

    /// Check if a type implements a trait.
    pub fn implements(&self, ty: &Type, trait_name: &str) -> bool {
        // First check exact matches from registered implementations
        if self.impls.iter().any(|i| {
            i.trait_name == trait_name && self.types_match(&i.for_type, ty)
        }) {
            return true;
        }

        // Try flexible matching for module-qualified types
        // If ty is "Vec" and impl is for "nalgebra.Vec", they should match
        // Also check if trait names match with qualification
        if self.impls.iter().any(|i| {
            self.trait_names_match(&i.trait_name, trait_name) &&
            self.types_match_flexible(&i.for_type, ty)
        }) {
            return true;
        }

        // Handle parameterized types (Option, Result, List, Map, Set, etc.)
        // These implement Eq and Show if their element types do
        match ty {
            Type::Named { name, args } => {
                // Eq and Show are auto-derived for all user-defined types (variants/records)
                // as long as their type arguments also implement the trait
                match trait_name {
                    "Eq" | "Show" => {
                        // Check if all type arguments implement the trait
                        args.is_empty() || args.iter().all(|arg| self.implements(arg, trait_name))
                    }
                    "Hash" => {
                        // Hash is auto-derived for container types if their element types implement it
                        let container_types = ["Option", "Result", "Json"];
                        container_types.contains(&name.as_str()) &&
                            (args.is_empty() || args.iter().all(|arg| self.implements(arg, trait_name)))
                    }
                    _ => {
                        // Other traits (Num, Ord, etc.) are NOT auto-derived for container types
                        false
                    }
                }
            }
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) => {
                match trait_name {
                    "Eq" | "Show" => self.implements(elem, trait_name),
                    // Lists and Arrays support Concat (++), Sets do not
                    "Concat" => matches!(ty, Type::List(_) | Type::Array(_)),
                    // Lists/Arrays/Sets are NOT hashable - they can't be Set elements or Map keys
                    _ => false,
                }
            }
            Type::Map(key, val) => {
                match trait_name {
                    "Eq" | "Show" => {
                        self.implements(key, trait_name) && self.implements(val, trait_name)
                    }
                    // Maps are NOT hashable
                    _ => false,
                }
            }
            Type::Tuple(elems) => {
                match trait_name {
                    "Eq" | "Show" | "Hash" => elems.iter().all(|e| self.implements(e, trait_name)),
                    _ => false,
                }
            }
            // Variant and Record types auto-derive Eq, Show, and Hash
            Type::Variant(_) | Type::Record(_) => {
                matches!(trait_name, "Eq" | "Show" | "Hash")
            }
            // Numeric types implement Eq, Show, Num, Ord, and Hash
            Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
            Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
            Type::Float | Type::Float32 | Type::Float64 |
            Type::BigInt | Type::Decimal => {
                matches!(trait_name, "Eq" | "Show" | "Num" | "Ord" | "Hash")
            }
            // String implements Eq, Show, Ord, Hash, and Concat
            Type::String => {
                matches!(trait_name, "Eq" | "Show" | "Ord" | "Hash" | "Concat")
            }
            // Bool implements Eq, Show, and Hash (not Ord - bools can't be compared with <)
            Type::Bool => {
                matches!(trait_name, "Eq" | "Show" | "Hash")
            }
            // Char implements Eq, Show, Hash, and Ord (chars are compared lexicographically at runtime)
            Type::Char => {
                matches!(trait_name, "Eq" | "Show" | "Hash" | "Ord")
            }
            // Unit implements Eq, Show, and Hash
            Type::Unit => {
                matches!(trait_name, "Eq" | "Show" | "Hash")
            }
            // Pid only implements Eq and Show (not hashable)
            Type::Pid => {
                matches!(trait_name, "Eq" | "Show")
            }
            _ => false,
        }
    }

    /// Check if a type DEFINITELY does not implement a trait, even with unresolved vars.
    /// Returns true only when we can be certain - treats Vars conservatively (assumes
    /// they might implement the trait). Used during deferred constraint checking where
    /// types may be partially resolved.
    pub fn definitely_not_implements(&self, ty: &Type, trait_name: &str) -> bool {
        match ty {
            Type::Var(_) => false, // Unknown - might implement
            Type::Function(_) => true, // Functions never implement any trait
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) => {
                match trait_name {
                    "Eq" | "Show" => self.definitely_not_implements(elem, trait_name),
                    "Num" | "Ord" | "Hash" => true, // Containers never implement Num/Ord/Hash
                    _ => false,
                }
            }
            Type::Map(key, val) => {
                match trait_name {
                    "Eq" | "Show" => {
                        self.definitely_not_implements(key, trait_name) ||
                        self.definitely_not_implements(val, trait_name)
                    }
                    "Num" | "Ord" | "Hash" => true, // Maps never implement Num/Ord/Hash
                    _ => false,
                }
            }
            Type::Tuple(elems) => {
                match trait_name {
                    "Eq" | "Show" | "Hash" => {
                        elems.iter().any(|e| self.definitely_not_implements(e, trait_name))
                    }
                    "Num" | "Ord" => true, // Tuples never implement Num/Ord
                    _ => false,
                }
            }
            Type::Named { args, .. } => {
                match trait_name {
                    "Eq" | "Show" => {
                        // Named types auto-derive Eq/Show. Fails if any arg definitely fails.
                        args.iter().any(|arg| self.definitely_not_implements(arg, trait_name))
                    }
                    "Num" | "Ord" => {
                        // Check if there's an explicit implementation
                        !self.implements(ty, trait_name)
                    }
                    _ => false,
                }
            }
            _ => {
                // For fully concrete types, defer to implements()
                if !ty.has_any_type_var() {
                    !self.implements(ty, trait_name)
                } else {
                    false
                }
            }
        }
    }

    /// Check if two types match (simple equality for now).
    fn types_match(&self, a: &Type, b: &Type) -> bool {
        a == b
    }

    /// Check if two trait names match, considering module qualification.
    /// "nalgebra.Num" matches "Num" and vice versa.
    fn trait_names_match(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        // Extract unqualified parts and compare
        let a_short = a.rsplit('.').next().unwrap_or(a);
        let b_short = b.rsplit('.').next().unwrap_or(b);
        a_short == b_short
    }

    /// Flexible type matching that handles module-qualified names.
    /// "nalgebra.Vec" matches "Vec" if they're the same underlying type.
    #[allow(clippy::only_used_in_recursion)]
    fn types_match_flexible(&self, a: &Type, b: &Type) -> bool {
        if a == b {
            return true;
        }
        match (a, b) {
            (Type::Named { name: a_name, args: a_args }, Type::Named { name: b_name, args: b_args }) => {
                // If args don't match in length, they're different
                if a_args.len() != b_args.len() {
                    return false;
                }
                // Extract unqualified names and compare
                let a_short = a_name.rsplit('.').next().unwrap_or(a_name);
                let b_short = b_name.rsplit('.').next().unwrap_or(b_name);
                if a_short != b_short {
                    return false;
                }
                // Check args recursively
                a_args.iter().zip(b_args.iter()).all(|(x, y)| self.types_match_flexible(x, y))
            }
            // For other type variants, use exact matching
            _ => false,
        }
    }

    /// Apply the current substitution to a type.
    pub fn apply_subst(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(resolved) = self.substitution.get(id) {
                    self.apply_subst(resolved)
                } else {
                    ty.clone()
                }
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|t| self.apply_subst(t)).collect())
            }
            Type::List(elem) => Type::List(Box::new(self.apply_subst(elem))),
            Type::Array(elem) => Type::Array(Box::new(self.apply_subst(elem))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.apply_subst(k)),
                Box::new(self.apply_subst(v)),
            ),
            Type::Set(elem) => Type::Set(Box::new(self.apply_subst(elem))),
            Type::Record(rec) => Type::Record(RecordType {
                name: rec.name.clone(),
                fields: rec
                    .fields
                    .iter()
                    .map(|(n, t, m)| (n.clone(), self.apply_subst(t), *m))
                    .collect(),
            }),
            Type::Function(f) => Type::Function(FunctionType { required_params: f.required_params,
                type_params: f.type_params.clone(),
                params: f.params.iter().map(|t| self.apply_subst(t)).collect(),
                ret: Box::new(self.apply_subst(&f.ret)),
            }),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|t| self.apply_subst(t)).collect(),
            },
            Type::IO(inner) => Type::IO(Box::new(self.apply_subst(inner))),
            _ => ty.clone(),
        }
    }
}

impl Type {
    /// Check if this type contains the given type variable.
    pub fn contains_var(&self, var_id: TypeVarId) -> bool {
        match self {
            Type::Var(id) => *id == var_id,
            Type::Tuple(elems) => elems.iter().any(|t| t.contains_var(var_id)),
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) | Type::IO(elem) => {
                elem.contains_var(var_id)
            }
            Type::Map(k, v) => k.contains_var(var_id) || v.contains_var(var_id),
            Type::Record(rec) => rec.fields.iter().any(|(_, t, _)| t.contains_var(var_id)),
            Type::Function(f) => {
                f.params.iter().any(|t| t.contains_var(var_id)) || f.ret.contains_var(var_id)
            }
            Type::Named { args, .. } => args.iter().any(|t| t.contains_var(var_id)),
            _ => false,
        }
    }

    /// Check if this type contains any type variable (resolved or not).
    pub fn has_any_type_var(&self) -> bool {
        match self {
            Type::Var(_) => true,
            Type::Tuple(elems) => elems.iter().any(|t| t.has_any_type_var()),
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) | Type::IO(elem) => {
                elem.has_any_type_var()
            }
            Type::Map(k, v) => k.has_any_type_var() || v.has_any_type_var(),
            Type::Record(rec) => rec.fields.iter().any(|(_, t, _)| t.has_any_type_var()),
            Type::Function(f) => {
                f.params.iter().any(|t| t.has_any_type_var()) || f.ret.has_any_type_var()
            }
            Type::Named { args, .. } => args.iter().any(|t| t.has_any_type_var()),
            _ => false,
        }
    }

    /// Check if this type is fully concrete (no type variables or type parameters).
    /// A concrete type can be used in type annotations.
    pub fn is_concrete(&self) -> bool {
        match self {
            Type::Var(_) | Type::TypeParam(_) => false,
            Type::Tuple(elems) => elems.iter().all(|t| t.is_concrete()),
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) | Type::IO(elem) => {
                elem.is_concrete()
            }
            Type::Map(k, v) => k.is_concrete() && v.is_concrete(),
            Type::Record(rec) => rec.fields.iter().all(|(_, t, _)| t.is_concrete()),
            Type::Function(f) => {
                f.params.iter().all(|t| t.is_concrete()) && f.ret.is_concrete()
            }
            Type::Named { args, .. } => args.iter().all(|t| t.is_concrete()),
            Type::Variant(var) => var.constructors.iter().all(|c| match c {
                Constructor::Unit(_) => true,
                Constructor::Positional(_, types) => types.iter().all(|t| t.is_concrete()),
                Constructor::Named(_, fields) => fields.iter().all(|(_, t)| t.is_concrete()),
            }),
            // All primitives are concrete
            _ => true,
        }
    }

    /// Find the maximum type variable ID in this type, or None if no type variables.
    pub fn max_var_id(&self) -> Option<TypeVarId> {
        match self {
            Type::Var(id) => Some(*id),
            Type::Tuple(elems) => elems.iter().filter_map(|t| t.max_var_id()).max(),
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) | Type::IO(elem) => {
                elem.max_var_id()
            }
            Type::Map(k, v) => {
                let k_max = k.max_var_id();
                let v_max = v.max_var_id();
                match (k_max, v_max) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            }
            Type::Record(rec) => rec.fields.iter().filter_map(|(_, t, _)| t.max_var_id()).max(),
            Type::Function(f) => f.max_var_id(),
            Type::Named { args, .. } => args.iter().filter_map(|t| t.max_var_id()).max(),
            _ => None,
        }
    }

    /// Pretty print a type.
    pub fn display(&self) -> String {
        match self {
            // Signed integers
            Type::Int8 => "Int8".to_string(),
            Type::Int16 => "Int16".to_string(),
            Type::Int32 => "Int32".to_string(),
            Type::Int64 => "Int64".to_string(),
            Type::Int => "Int".to_string(),
            // Unsigned integers
            Type::UInt8 => "UInt8".to_string(),
            Type::UInt16 => "UInt16".to_string(),
            Type::UInt32 => "UInt32".to_string(),
            Type::UInt64 => "UInt64".to_string(),
            // Floating point
            Type::Float32 => "Float32".to_string(),
            Type::Float64 => "Float64".to_string(),
            Type::Float => "Float".to_string(),
            // Arbitrary precision
            Type::BigInt => "BigInt".to_string(),
            Type::Decimal => "Decimal".to_string(),
            // Other primitives
            Type::Bool => "Bool".to_string(),
            Type::Char => "Char".to_string(),
            Type::String => "String".to_string(),
            Type::Unit => "()".to_string(),
            Type::Never => "Never".to_string(),
            Type::Var(id) => format!("?{}", id),
            Type::TypeParam(name) => name.clone(),
            Type::Tuple(elems) => {
                let inner: Vec<_> = elems.iter().map(|t| t.display()).collect();
                format!("({})", inner.join(", "))
            }
            Type::List(elem) => format!("List[{}]", elem.display()),
            Type::Array(elem) => format!("Array[{}]", elem.display()),
            Type::Map(k, v) => format!("Map[{}, {}]", k.display(), v.display()),
            Type::Set(elem) => format!("Set[{}]", elem.display()),
            Type::Record(rec) => {
                if let Some(name) = &rec.name {
                    name.clone()
                } else {
                    let fields: Vec<_> = rec
                        .fields
                        .iter()
                        .map(|(n, t, _)| format!("{}: {}", n, t.display()))
                        .collect();
                    format!("{{{}}}", fields.join(", "))
                }
            }
            Type::Variant(var) => var.name.clone(),
            Type::Function(f) => {
                let params: Vec<_> = f.params.iter().map(|t| t.display()).collect();
                format!("({}) -> {}", params.join(", "), f.ret.display())
            }
            Type::Named { name, args } => {
                // Use short name for display (e.g., "Option" not "stdlib.list.Option")
                let short_name = name.rsplit('.').next().unwrap_or(name);
                if args.is_empty() {
                    short_name.to_string()
                } else {
                    let args_str: Vec<_> = args.iter().map(|t| t.display()).collect();
                    format!("{}[{}]", short_name, args_str.join(", "))
                }
            }
            Type::IO(inner) => format!("IO[{}]", inner.display()),
            Type::Pid => "Pid".to_string(),
            Type::Ref => "Ref".to_string(),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// Initialize a type environment with standard types and traits.
pub fn standard_env() -> TypeEnv {
    let mut env = TypeEnv::new();

    // Standard types
    env.define_type(
        "Option".to_string(),
        TypeDef::Variant {
            params: vec![TypeParam {
                name: "T".to_string(),
                constraints: vec![],
            }],
            constructors: vec![
                Constructor::Unit("None".to_string()),
                Constructor::Positional("Some".to_string(), vec![Type::TypeParam("T".to_string())]),
            ],
        },
    );

    env.define_type(
        "Result".to_string(),
        TypeDef::Variant {
            params: vec![
                TypeParam {
                    name: "T".to_string(),
                    constraints: vec![],
                },
                TypeParam {
                    name: "E".to_string(),
                    constraints: vec![],
                },
            ],
            constructors: vec![
                Constructor::Positional("Ok".to_string(), vec![Type::TypeParam("T".to_string())]),
                Constructor::Positional("Err".to_string(), vec![Type::TypeParam("E".to_string())]),
            ],
        },
    );

    env.define_type(
        "List".to_string(),
        TypeDef::Variant {
            params: vec![TypeParam {
                name: "T".to_string(),
                constraints: vec![],
            }],
            constructors: vec![
                Constructor::Unit("Nil".to_string()),
                Constructor::Positional(
                    "Cons".to_string(),
                    vec![
                        Type::TypeParam("T".to_string()),
                        Type::Named {
                            name: "List".to_string(),
                            args: vec![Type::TypeParam("T".to_string())],
                        },
                    ],
                ),
            ],
        },
    );

    // Standard traits
    env.traits.insert(
        "Eq".to_string(),
        TraitDef {
            name: "Eq".to_string(),
            supertraits: vec![],
            required: vec![TraitMethod {
                name: "==".to_string(),
                params: vec![
                    ("self".to_string(), Type::TypeParam("Self".to_string())),
                    ("other".to_string(), Type::TypeParam("Self".to_string())),
                ],
                ret: Type::Bool,
            }],
            defaults: vec![TraitMethod {
                name: "!=".to_string(),
                params: vec![
                    ("self".to_string(), Type::TypeParam("Self".to_string())),
                    ("other".to_string(), Type::TypeParam("Self".to_string())),
                ],
                ret: Type::Bool,
            }],
        },
    );

    env.traits.insert(
        "Ord".to_string(),
        TraitDef {
            name: "Ord".to_string(),
            supertraits: vec!["Eq".to_string()],
            required: vec![TraitMethod {
                name: "compare".to_string(),
                params: vec![
                    ("self".to_string(), Type::TypeParam("Self".to_string())),
                    ("other".to_string(), Type::TypeParam("Self".to_string())),
                ],
                ret: Type::Named {
                    name: "Ordering".to_string(),
                    args: vec![],
                },
            }],
            defaults: vec![],
        },
    );

    env.traits.insert(
        "Show".to_string(),
        TraitDef {
            name: "Show".to_string(),
            supertraits: vec![],
            required: vec![TraitMethod {
                name: "show".to_string(),
                params: vec![("self".to_string(), Type::TypeParam("Self".to_string()))],
                ret: Type::String,
            }],
            defaults: vec![],
        },
    );

    env.traits.insert(
        "Hash".to_string(),
        TraitDef {
            name: "Hash".to_string(),
            supertraits: vec![],
            required: vec![TraitMethod {
                name: "hash".to_string(),
                params: vec![("self".to_string(), Type::TypeParam("Self".to_string()))],
                ret: Type::Int,
            }],
            defaults: vec![],
        },
    );

    // Primitive trait implementations
    for ty in ["Int", "Float", "Bool", "Char", "String"] {
        env.impls.push(TraitImpl {
            trait_name: "Eq".to_string(),
            for_type: match ty {
                "Int" => Type::Int,
                "Float" => Type::Float,
                "Bool" => Type::Bool,
                "Char" => Type::Char,
                "String" => Type::String,
                _ => unreachable!(),
            },
            constraints: vec![],
        });
        env.impls.push(TraitImpl {
            trait_name: "Show".to_string(),
            for_type: match ty {
                "Int" => Type::Int,
                "Float" => Type::Float,
                "Bool" => Type::Bool,
                "Char" => Type::Char,
                "String" => Type::String,
                _ => unreachable!(),
            },
            constraints: vec![],
        });
    }

    // Unit implements Eq and Show
    env.impls.push(TraitImpl {
        trait_name: "Eq".to_string(),
        for_type: Type::Unit,
        constraints: vec![],
    });
    env.impls.push(TraitImpl {
        trait_name: "Show".to_string(),
        for_type: Type::Unit,
        constraints: vec![],
    });

    for ty in ["Int", "Float", "Char", "String"] {
        env.impls.push(TraitImpl {
            trait_name: "Ord".to_string(),
            for_type: match ty {
                "Int" => Type::Int,
                "Float" => Type::Float,
                "Char" => Type::Char,
                "String" => Type::String,
                _ => unreachable!(),
            },
            constraints: vec![],
        });
    }

    for ty in ["Int", "Bool", "Char", "String"] {
        env.impls.push(TraitImpl {
            trait_name: "Hash".to_string(),
            for_type: match ty {
                "Int" => Type::Int,
                "Bool" => Type::Bool,
                "Char" => Type::Char,
                "String" => Type::String,
                _ => unreachable!(),
            },
            constraints: vec![],
        });
    }

    // Num trait for arithmetic operations
    env.traits.insert(
        "Num".to_string(),
        TraitDef {
            name: "Num".to_string(),
            supertraits: vec![],
            required: vec![
                TraitMethod {
                    name: "+".to_string(),
                    params: vec![
                        ("self".to_string(), Type::TypeParam("Self".to_string())),
                        ("other".to_string(), Type::TypeParam("Self".to_string())),
                    ],
                    ret: Type::TypeParam("Self".to_string()),
                },
                TraitMethod {
                    name: "-".to_string(),
                    params: vec![
                        ("self".to_string(), Type::TypeParam("Self".to_string())),
                        ("other".to_string(), Type::TypeParam("Self".to_string())),
                    ],
                    ret: Type::TypeParam("Self".to_string()),
                },
                TraitMethod {
                    name: "*".to_string(),
                    params: vec![
                        ("self".to_string(), Type::TypeParam("Self".to_string())),
                        ("other".to_string(), Type::TypeParam("Self".to_string())),
                    ],
                    ret: Type::TypeParam("Self".to_string()),
                },
                TraitMethod {
                    name: "/".to_string(),
                    params: vec![
                        ("self".to_string(), Type::TypeParam("Self".to_string())),
                        ("other".to_string(), Type::TypeParam("Self".to_string())),
                    ],
                    ret: Type::TypeParam("Self".to_string()),
                },
            ],
            defaults: vec![],
        },
    );

    // Num implementations for numeric types
    for ty in [Type::Int, Type::Float, Type::Int8, Type::Int16, Type::Int32,
               Type::UInt8, Type::UInt16, Type::UInt32, Type::UInt64,
               Type::Float32, Type::BigInt, Type::Decimal] {
        env.impls.push(TraitImpl {
            trait_name: "Num".to_string(),
            for_type: ty,
            constraints: vec![],
        });
    }

    // Built-in functions with trait constraints
    // println: Show a => a -> ()
    env.insert_function(
        "println".to_string(),
        FunctionType { required_params: None,
            type_params: vec![TypeParam {
                name: "a".to_string(),
                constraints: vec!["Show".to_string()],
            }],
            params: vec![Type::TypeParam("a".to_string())],
            ret: Box::new(Type::Unit),
        },
    );

    // print: Show a => a -> ()
    env.insert_function(
        "print".to_string(),
        FunctionType { required_params: None,
            type_params: vec![TypeParam {
                name: "a".to_string(),
                constraints: vec!["Show".to_string()],
            }],
            params: vec![Type::TypeParam("a".to_string())],
            ret: Box::new(Type::Unit),
        },
    );

    // show: Show a => a -> String
    env.insert_function(
        "show".to_string(),
        FunctionType { required_params: None,
            type_params: vec![TypeParam {
                name: "a".to_string(),
                constraints: vec!["Show".to_string()],
            }],
            params: vec![Type::TypeParam("a".to_string())],
            ret: Box::new(Type::String),
        },
    );

    // inspect: a -> String -> () (for TUI debugging - sends value to inspector panel)
    env.insert_function(
        "inspect".to_string(),
        FunctionType { required_params: None,
            type_params: vec![TypeParam {
                name: "a".to_string(),
                constraints: vec![],  // No constraints - any value can be inspected
            }],
            params: vec![Type::TypeParam("a".to_string()), Type::String],
            ret: Box::new(Type::Unit),
        },
    );

    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_display() {
        assert_eq!(Type::Int.display(), "Int");
        assert_eq!(Type::List(Box::new(Type::Int)).display(), "List[Int]");
        assert_eq!(
            Type::Map(Box::new(Type::String), Box::new(Type::Int)).display(),
            "Map[String, Int]"
        );
        assert_eq!(
            Type::Tuple(vec![Type::Int, Type::String]).display(),
            "(Int, String)"
        );
        assert_eq!(
            Type::Function(FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int, Type::Int],
                ret: Box::new(Type::Int),
            })
            .display(),
            "(Int, Int) -> Int"
        );
    }

    #[test]
    fn test_standard_env() {
        let env = standard_env();

        // Check Option type exists
        let option = env.lookup_type("Option").unwrap();
        match option {
            TypeDef::Variant { constructors, .. } => {
                assert_eq!(constructors.len(), 2);
            }
            _ => panic!("Option should be a variant"),
        }

        // Check traits exist
        assert!(env.lookup_trait("Eq").is_some());
        assert!(env.lookup_trait("Ord").is_some());
        assert!(env.lookup_trait("Show").is_some());

        // Check implementations
        assert!(env.implements(&Type::Int, "Eq"));
        assert!(env.implements(&Type::Int, "Ord"));
        assert!(env.implements(&Type::String, "Show"));
    }

    #[test]
    fn test_fresh_var() {
        let mut env = TypeEnv::new();
        let v1 = env.fresh_var();
        let v2 = env.fresh_var();
        assert_ne!(v1, v2);
    }
}
