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

/// Unique identifier for type variables during inference.
pub type TypeVarId = u32;

/// A type in the Nostos type system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    // === Primitives ===
    Int,
    Float,
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
    #[error("Type mismatch: expected {expected}, found {found}")]
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

    #[error("Occurs check failed: {0} appears in {1}")]
    OccursCheck(String, String),
}

/// Type environment for tracking bindings and definitions.
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    /// Variable bindings: name -> (type, is_mutable)
    pub bindings: HashMap<String, (Type, bool)>,
    /// Type definitions: name -> TypeDef
    pub types: HashMap<String, TypeDef>,
    /// Trait definitions: name -> TraitDef
    pub traits: HashMap<String, TraitDef>,
    /// Trait implementations
    pub impls: Vec<TraitImpl>,
    /// Function signatures: name -> FunctionType
    pub functions: HashMap<String, FunctionType>,
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
    pub fn lookup_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
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
        self.impls.iter().any(|i| {
            i.trait_name == trait_name && self.types_match(&i.for_type, ty)
        })
    }

    /// Check if two types match (simple equality for now).
    fn types_match(&self, a: &Type, b: &Type) -> bool {
        a == b
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
            Type::Function(f) => Type::Function(FunctionType {
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

    /// Pretty print a type.
    pub fn display(&self) -> String {
        match self {
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
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
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<_> = args.iter().map(|t| t.display()).collect();
                    format!("{}[{}]", name, args_str.join(", "))
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
            Type::Function(FunctionType {
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
