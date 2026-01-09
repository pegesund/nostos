//! Type checking for Nostos.
//!
//! This module walks the AST and:
//! 1. Resolves type definitions
//! 2. Checks type annotations
//! 3. Infers types for expressions
//! 4. Checks pattern exhaustiveness
//! 5. Validates trait implementations

use crate::{Type, TypeError, TypeEnv, TypeDef, FunctionType, Constructor};

/// Check that a type definition is well-formed.
pub fn check_type_def(env: &TypeEnv, def: &TypeDef) -> Result<(), TypeError> {
    match def {
        TypeDef::Record { params: _, fields, is_mutable: _ } => {
            // Check no duplicate fields
            let mut seen = std::collections::HashSet::new();
            for (name, _, _) in fields {
                if !seen.insert(name) {
                    return Err(TypeError::DuplicateField(name.clone()));
                }
            }
            // Check field types are valid
            for (_, ty, _) in fields {
                check_type_exists(env, ty)?;
            }
            Ok(())
        }
        TypeDef::Variant { params: _, constructors } => {
            // Check no duplicate constructor names
            let mut seen = std::collections::HashSet::new();
            for ctor in constructors {
                let name = match ctor {
                    Constructor::Unit(n) | Constructor::Positional(n, _) | Constructor::Named(n, _) => n,
                };
                if !seen.insert(name) {
                    return Err(TypeError::DuplicateField(name.clone()));
                }
            }
            Ok(())
        }
        TypeDef::Alias { params: _, target } => {
            check_type_exists(env, target)
        }
    }
}

/// Check that a type reference is valid.
pub fn check_type_exists(env: &TypeEnv, ty: &Type) -> Result<(), TypeError> {
    match ty {
        Type::Named { name, args } => {
            if env.lookup_type(name).is_none() {
                // Check if it's a primitive
                if !["Int", "Float", "Bool", "Char", "String", "Pid", "Ref"].contains(&name.as_str()) {
                    return Err(TypeError::UnknownType(name.clone()));
                }
            }
            for arg in args {
                check_type_exists(env, arg)?;
            }
            Ok(())
        }
        Type::TypeParam(_) => Ok(()), // Type params are checked elsewhere
        Type::Tuple(elems) => {
            for elem in elems {
                check_type_exists(env, elem)?;
            }
            Ok(())
        }
        Type::List(elem) | Type::Array(elem) | Type::Set(elem) | Type::IO(elem) => {
            check_type_exists(env, elem)
        }
        Type::Map(k, v) => {
            check_type_exists(env, k)?;
            check_type_exists(env, v)
        }
        Type::Record(rec) => {
            for (_, field_ty, _) in &rec.fields {
                check_type_exists(env, field_ty)?;
            }
            Ok(())
        }
        Type::Function(f) => {
            for param in &f.params {
                check_type_exists(env, param)?;
            }
            check_type_exists(env, &f.ret)
        }
        _ => Ok(()), // Primitives and variables are always valid
    }
}

/// Check that a function signature is well-formed.
pub fn check_function_sig(env: &TypeEnv, sig: &FunctionType) -> Result<(), TypeError> {
    // Check type parameter constraints refer to valid traits
    for param in &sig.type_params {
        for constraint in &param.constraints {
            if env.lookup_trait(constraint).is_none() {
                return Err(TypeError::UnknownType(format!("trait {}", constraint)));
            }
        }
    }

    // Check parameter types
    for param_ty in &sig.params {
        check_type_exists(env, param_ty)?;
    }

    // Check return type
    check_type_exists(env, &sig.ret)?;

    Ok(())
}

/// Check that patterns in a match are exhaustive.
pub fn check_exhaustive(env: &TypeEnv, ty: &Type, patterns: &[PatternKind]) -> Result<(), TypeError> {
    // For now, just check that there's a wildcard or we cover all constructors
    // A full implementation would use exhaustiveness checking algorithms

    if patterns.iter().any(|p| matches!(p, PatternKind::Wildcard)) {
        return Ok(());
    }

    match ty {
        Type::Named { name, .. } => {
            if let Some(TypeDef::Variant { constructors, .. }) = env.lookup_type(name) {
                let covered: std::collections::HashSet<_> = patterns
                    .iter()
                    .filter_map(|p| {
                        if let PatternKind::Constructor(n, _) = p {
                            Some(n.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();

                let all: std::collections::HashSet<_> = constructors
                    .iter()
                    .map(|c| match c {
                        Constructor::Unit(n) | Constructor::Positional(n, _) | Constructor::Named(n, _) => n.as_str(),
                    })
                    .collect();

                let missing: Vec<_> = all.difference(&covered).map(|s| s.to_string()).collect();
                if !missing.is_empty() {
                    return Err(TypeError::NonExhaustive { missing });
                }
            }
        }
        Type::Bool => {
            let has_true = patterns.iter().any(|p| matches!(p, PatternKind::Bool(true)));
            let has_false = patterns.iter().any(|p| matches!(p, PatternKind::Bool(false)));
            if !has_true || !has_false {
                let mut missing = vec![];
                if !has_true { missing.push("true".to_string()); }
                if !has_false { missing.push("false".to_string()); }
                return Err(TypeError::NonExhaustive { missing });
            }
        }
        _ => {
            // For other types (Int, String, etc.), we'd need a wildcard
            return Err(TypeError::NonExhaustive { missing: vec!["_".to_string()] });
        }
    }

    Ok(())
}

/// Simplified pattern representation for exhaustiveness checking.
#[derive(Debug, Clone)]
pub enum PatternKind {
    Wildcard,
    Variable(String),
    Constructor(String, Vec<PatternKind>),
    Tuple(Vec<PatternKind>),
    Literal(LiteralKind),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone)]
pub enum LiteralKind {
    Int(i64),
    Float(String),
    String(String),
    Char(char),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard_env;

    #[test]
    fn test_check_type_exists_primitives() {
        let env = standard_env();
        assert!(check_type_exists(&env, &Type::Int).is_ok());
        assert!(check_type_exists(&env, &Type::Float).is_ok());
        assert!(check_type_exists(&env, &Type::String).is_ok());
    }

    #[test]
    fn test_check_type_exists_unknown() {
        let env = standard_env();
        let ty = Type::Named { name: "Unknown".to_string(), args: vec![] };
        assert!(matches!(check_type_exists(&env, &ty), Err(TypeError::UnknownType(_))));
    }

    #[test]
    fn test_check_type_exists_option() {
        let env = standard_env();
        let ty = Type::Named { name: "Option".to_string(), args: vec![Type::Int] };
        assert!(check_type_exists(&env, &ty).is_ok());
    }

    #[test]
    fn test_exhaustive_bool() {
        let env = standard_env();

        // Exhaustive
        let patterns = vec![PatternKind::Bool(true), PatternKind::Bool(false)];
        assert!(check_exhaustive(&env, &Type::Bool, &patterns).is_ok());

        // Non-exhaustive
        let patterns = vec![PatternKind::Bool(true)];
        assert!(matches!(check_exhaustive(&env, &Type::Bool, &patterns), Err(TypeError::NonExhaustive { .. })));
    }

    #[test]
    fn test_exhaustive_wildcard() {
        let env = standard_env();
        let patterns = vec![PatternKind::Wildcard];
        assert!(check_exhaustive(&env, &Type::Int, &patterns).is_ok());
    }
}
