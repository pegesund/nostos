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

use crate::infer::InferenceResult;
use crate::Type;
use std::collections::{HashMap, HashSet};

/// A specialized function instance with a mangled name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionInstance {
    /// Original function name
    pub original_name: String,
    /// Mangled name with type suffix (e.g., "double$I")
    pub mangled_name: String,
    /// Concrete argument types for this instantiation
    pub arg_types: Vec<Type>,
    /// Concrete return type for this instantiation
    pub ret_type: Type,
}

/// Result of monomorphization analysis.
#[derive(Debug, Clone)]
pub struct MonomorphizationPlan {
    /// All function instances that need to be generated.
    /// Key: (original_name, arg_types), Value: FunctionInstance
    pub instances: HashMap<(String, Vec<Type>), FunctionInstance>,
    /// Mapping from call site spans to the mangled function name to use.
    /// Key: (start, end) of call expression span
    pub call_rewrites: HashMap<(usize, usize), String>,
}

impl MonomorphizationPlan {
    /// Create a new empty plan.
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            call_rewrites: HashMap::new(),
        }
    }

    /// Get the mangled name for a call at the given span, if it needs rewriting.
    pub fn get_rewritten_name(&self, start: usize, end: usize) -> Option<&str> {
        self.call_rewrites.get(&(start, end)).map(|s| s.as_str())
    }

    /// Get all unique function instances that need to be generated.
    pub fn unique_instances(&self) -> impl Iterator<Item = &FunctionInstance> {
        self.instances.values()
    }
}

impl Default for MonomorphizationPlan {
    fn default() -> Self {
        Self::new()
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
fn is_builtin(name: &str) -> bool {
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

/// Check if a function needs monomorphization based on its call sites.
/// A function needs monomorphization if it's called with different type instantiations.
fn needs_monomorphization(
    func_name: &str,
    infer_result: &InferenceResult,
) -> bool {
    if is_builtin(func_name) {
        return false;
    }

    // Check if there are multiple distinct instantiations of this function
    let instantiations = infer_result.function_instantiations();
    let matching: Vec<_> = instantiations
        .keys()
        .filter(|(name, _)| name == func_name)
        .collect();

    // If there are multiple distinct type instantiations, we need monomorphization
    matching.len() > 1
}

/// Analyze the inference results and create a monomorphization plan.
pub fn create_monomorphization_plan(
    infer_result: &InferenceResult,
    polymorphic_functions: &HashSet<String>,
) -> MonomorphizationPlan {
    let mut plan = MonomorphizationPlan::new();

    // Group call sites by function and type instantiation
    for call_site in &infer_result.call_sites {
        let func_name = &call_site.func_name;

        // Skip builtins
        if is_builtin(func_name) {
            continue;
        }

        // Only monomorphize if the function is polymorphic or has multiple instantiations
        let should_mono = polymorphic_functions.contains(func_name)
            || needs_monomorphization(func_name, infer_result);

        if should_mono {
            let key = (func_name.clone(), call_site.arg_types.clone());
            let mangled = mangle_name(func_name, &call_site.arg_types);

            // Record the instance if we haven't seen it
            plan.instances.entry(key).or_insert_with(|| FunctionInstance {
                original_name: func_name.clone(),
                mangled_name: mangled.clone(),
                arg_types: call_site.arg_types.clone(),
                ret_type: call_site.ret_type.clone(),
            });

            // Record the call rewrite
            plan.call_rewrites.insert(
                (call_site.span.start, call_site.span.end),
                mangled,
            );
        }
    }

    plan
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
