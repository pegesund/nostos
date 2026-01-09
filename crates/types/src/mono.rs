//! Monomorphization for Nostos.
//!
//! This module handles the generation of specialized function variants
//! for polymorphic functions. For example, a function like:
//!
//! ```nostos
//! double(x: Num) -> Num = x + x
//! ```
//!
//! Will generate specialized variants like:
//! - `double$I` for Int arguments
//! - `double$F` for Float arguments

use crate::Type;
use std::collections::HashMap;

/// Result of monomorphization analysis.
#[derive(Debug, Clone, Default)]
pub struct MonomorphizationPlan {
    /// Mapping from call site spans to the mangled function name to use.
    /// Key: (start, end) of call expression span
    pub call_rewrites: HashMap<(usize, usize), String>,
}

impl MonomorphizationPlan {
    /// Create a new empty plan.
    pub fn new() -> Self {
        Self {
            call_rewrites: HashMap::new(),
        }
    }

    /// Get the mangled name for a call at the given span, if it needs rewriting.
    pub fn get_rewritten_name(&self, start: usize, end: usize) -> Option<&str> {
        self.call_rewrites.get(&(start, end)).map(|s| s.as_str())
    }

    /// Add a call rewrite.
    pub fn add_rewrite(&mut self, start: usize, end: usize, mangled_name: String) {
        self.call_rewrites.insert((start, end), mangled_name);
    }
}

/// Generate a type suffix for a concrete type.
/// Used to create mangled function names.
pub fn type_suffix(ty: &Type) -> String {
    match ty {
        // Signed integers
        Type::Int8 => "I8".to_string(),
        Type::Int16 => "I16".to_string(),
        Type::Int32 => "I32".to_string(),
        Type::Int64 => "I64".to_string(),
        Type::Int => "I".to_string(),
        // Unsigned integers
        Type::UInt8 => "U8".to_string(),
        Type::UInt16 => "U16".to_string(),
        Type::UInt32 => "U32".to_string(),
        Type::UInt64 => "U64".to_string(),
        // Floats
        Type::Float32 => "F32".to_string(),
        Type::Float64 => "F64".to_string(),
        Type::Float => "F".to_string(),
        // Arbitrary precision
        Type::BigInt => "BI".to_string(),
        Type::Decimal => "D".to_string(),
        // Other primitives
        Type::Bool => "B".to_string(),
        Type::Char => "C".to_string(),
        Type::String => "S".to_string(),
        Type::Unit => "U".to_string(),
        Type::Never => "N".to_string(),
        // Type variables
        Type::Var(id) => format!("v{}", id),
        Type::TypeParam(name) => format!("T{}", name),
        // Compound types
        Type::Tuple(elems) => {
            let inner: String = elems.iter().map(type_suffix).collect();
            format!("Tup{}", inner)
        }
        Type::List(inner) => format!("L{}", type_suffix(inner)),
        Type::Array(inner) => format!("A{}", type_suffix(inner)),
        Type::Map(k, v) => format!("M{}_{}", type_suffix(k), type_suffix(v)),
        Type::Set(inner) => format!("Set{}", type_suffix(inner)),
        // Structural types
        Type::Record(rec) => {
            format!("R{}", rec.name.as_ref().map(|s| s.as_str()).unwrap_or("anon"))
        }
        Type::Variant(var) => {
            format!("V{}", var.name.as_str())
        }
        // Function type
        Type::Function(ft) => {
            let args_suffix: String = ft.params.iter().map(type_suffix).collect();
            format!("Fn{}_{}", args_suffix, type_suffix(&ft.ret))
        }
        // Named type
        Type::Named { name, args } => {
            if args.is_empty() {
                format!("N{}", name)
            } else {
                let args_suffix: String = args.iter().map(type_suffix).collect();
                format!("N{}_{}", name, args_suffix)
            }
        }
        // Special types
        Type::IO(inner) => format!("IO{}", type_suffix(inner)),
        Type::Pid => "P".to_string(),
        Type::Ref => "Ref".to_string(),
    }
}

/// Generate a mangled function name from the original name and argument types.
pub fn mangle_name(name: &str, arg_types: &[Type]) -> String {
    if arg_types.is_empty() {
        name.to_string()
    } else {
        let suffix: String = arg_types.iter().map(type_suffix).collect::<Vec<_>>().join("_");
        format!("{}${}", name, suffix)
    }
}

/// Functions that should NOT be monomorphized (builtins, etc.)
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "println" | "print" | "show" | "length" | "head" | "tail" | "empty"
            | "append" | "reverse" | "concat" | "map" | "filter" | "fold" | "foldl" | "foldr"
            | "range" | "zip" | "unzip" | "take" | "drop" | "nth" | "contains"
            | "keys" | "values" | "get" | "insert" | "remove" | "merge"
            | "floor" | "ceil" | "round" | "abs" | "sqrt" | "pow" | "sin" | "cos" | "tan"
            | "log" | "exp" | "min" | "max" | "mod" | "div"
            | "throw" | "self" | "spawn" | "send" | "receive" | "sleep"
            | "trunc" | "toFloat" | "toInt" | "toString"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_suffix() {
        assert_eq!(type_suffix(&Type::Int), "I");
        assert_eq!(type_suffix(&Type::Float), "F");
        assert_eq!(type_suffix(&Type::Bool), "B");
        assert_eq!(type_suffix(&Type::String), "S");
        assert_eq!(type_suffix(&Type::List(Box::new(Type::Int))), "LI");
    }

    #[test]
    fn test_mangle_name() {
        assert_eq!(
            mangle_name("double", &[Type::Int]),
            "double$I"
        );
        assert_eq!(
            mangle_name("double", &[Type::Float]),
            "double$F"
        );
        assert_eq!(
            mangle_name("add", &[Type::Int, Type::Int]),
            "add$I_I"
        );
    }
}
