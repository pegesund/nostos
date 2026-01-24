//! Type inference for Nostos.
//!
//! Implements Hindley-Milner type inference with extensions for:
//! - Records (structural typing)
//! - Variants (sum types)
//! - Traits (type class constraints)
//!
//! The algorithm works in two phases:
//! 1. Generate constraints by walking the AST
//! 2. Solve constraints through unification

use crate::{Constructor, FunctionType, RecordType, Type, TypeDef, TypeError, TypeEnv};
use nostos_syntax::ast::{
    BinOp, Binding, CallArg, Expr, FnClause, FnDef, Item, MatchArm, Module, Pattern, RecordField,
    RecordPatternField, Span, Stmt, TypeExpr, UnaryOp, VariantPatternFields,
};
use std::collections::HashMap;

/// A type constraint generated during inference.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two types must be equal (with optional span for error reporting)
    Equal(Type, Type, Option<Span>),
    /// A type must implement a trait
    HasTrait(Type, String),
    /// A type must have a field
    HasField(Type, String, Type),
}

/// A pending method call to be type-checked after constraint solving.
#[derive(Debug, Clone)]
pub struct PendingMethodCall {
    pub receiver_ty: Type,
    pub method_name: String,
    pub arg_types: Vec<Type>,
    pub ret_ty: Type,
    pub span: Option<Span>,
}

/// Type inference context.
pub struct InferCtx<'a> {
    pub env: &'a mut TypeEnv,
    pub constraints: Vec<Constraint>,
    /// Trait bounds on type variables: maps type var ID to set of required traits
    pub trait_bounds: HashMap<u32, Vec<String>>,
    /// Name of the function currently being inferred (for handling recursive calls)
    current_function: Option<String>,
    /// Pending method calls to check after solve() - for UFCS type checking
    pending_method_calls: Vec<PendingMethodCall>,
    /// Span of the most recent error during constraint solving
    last_error_span: Option<Span>,
    /// Current span being processed (for propagating to nested unifications)
    current_constraint_span: Option<Span>,
    /// Inferred types for each expression, keyed by Span.
    /// After solve(), these contain resolved (substituted) types.
    pub expr_types: HashMap<Span, Type>,
    /// TypeParam name -> Type mappings during unification.
    /// Ensures that the same TypeParam (e.g., "a") maps to the same type variable
    /// throughout a single unification session.
    type_param_mappings: HashMap<String, Type>,
}

impl<'a> InferCtx<'a> {
    pub fn new(env: &'a mut TypeEnv) -> Self {
        Self {
            env,
            constraints: Vec::new(),
            trait_bounds: HashMap::new(),
            current_function: None,
            pending_method_calls: Vec::new(),
            last_error_span: None,
            current_constraint_span: None,
            expr_types: HashMap::new(),
            type_param_mappings: HashMap::new(),
        }
    }

    /// Add a trait bound to a type variable
    pub fn add_trait_bound(&mut self, var_id: u32, trait_name: String) {
        let bounds = self.trait_bounds.entry(var_id).or_default();
        if !bounds.contains(&trait_name) {
            bounds.push(trait_name);
        }
    }

    /// Get trait bounds for a type variable.
    /// Follows the substitution chain to find all bounds on equivalent type variables.
    pub fn get_trait_bounds(&self, var_id: u32) -> Vec<&String> {
        let mut result = Vec::new();

        // Check bounds directly on this var
        if let Some(bounds) = self.trait_bounds.get(&var_id) {
            result.extend(bounds.iter());
        }

        // Also check bounds on any type variable that maps TO this one
        // (i.e., look through substitution backwards)
        for (&source_id, source_bounds) in &self.trait_bounds {
            if source_id == var_id {
                continue; // Already added
            }
            // Check if source_id eventually resolves to var_id
            let resolved = self.env.apply_subst(&Type::Var(source_id));
            if let Type::Var(resolved_id) = resolved {
                if resolved_id == var_id {
                    result.extend(source_bounds.iter());
                }
            }
        }

        result
    }

    /// Generate a fresh type variable.
    pub fn fresh(&mut self) -> Type {
        self.env.fresh_var()
    }

    /// Add an equality constraint (without span information).
    pub fn unify(&mut self, t1: Type, t2: Type) {
        self.constraints.push(Constraint::Equal(t1, t2, None));
    }

    /// Add an equality constraint with span information for precise error reporting.
    pub fn unify_at(&mut self, t1: Type, t2: Type, span: Span) {
        self.constraints.push(Constraint::Equal(t1, t2, Some(span)));
    }

    /// Add a trait constraint.
    pub fn require_trait(&mut self, ty: Type, trait_name: &str) {
        self.constraints
            .push(Constraint::HasTrait(ty, trait_name.to_string()));
    }

    /// Add a field constraint.
    pub fn require_field(&mut self, ty: Type, field: &str, field_ty: Type) {
        self.constraints
            .push(Constraint::HasField(ty, field.to_string(), field_ty));
    }

    /// Uncurry a curried function type into an uncurried one.
    /// For example, `b -> a -> b` (curried) becomes `(b, a) -> b` (uncurried).
    /// Returns the flattened parameter list and the final return type.
    fn uncurry_function_type(ft: &FunctionType) -> (Vec<Type>, Type) {
        let mut params = ft.params.clone();
        let mut ret = ft.ret.as_ref().clone();

        // While the return type is also a function, flatten it
        while let Type::Function(inner_ft) = ret {
            params.extend(inner_ft.params.clone());
            ret = inner_ft.ret.as_ref().clone();
        }

        (params, ret)
    }

    /// Get the base type name from a resolved type (for UFCS method lookup).
    /// Returns None for type variables or types without a clear base name.
    fn get_type_name(&self, ty: &Type) -> Option<String> {
        // First, resolve any type variables
        let resolved = self.env.apply_subst(ty);
        match resolved {
            Type::String => Some("String".to_string()),
            Type::Int | Type::Int64 => Some("Int".to_string()),
            Type::Int8 => Some("Int8".to_string()),
            Type::Int16 => Some("Int16".to_string()),
            Type::Int32 => Some("Int32".to_string()),
            Type::UInt8 => Some("UInt8".to_string()),
            Type::UInt16 => Some("UInt16".to_string()),
            Type::UInt32 => Some("UInt32".to_string()),
            Type::UInt64 => Some("UInt64".to_string()),
            Type::Float | Type::Float64 => Some("Float".to_string()),
            Type::Float32 => Some("Float32".to_string()),
            Type::BigInt => Some("BigInt".to_string()),
            Type::Decimal => Some("Decimal".to_string()),
            Type::Bool => Some("Bool".to_string()),
            Type::Char => Some("Char".to_string()),
            Type::List(_) => Some("List".to_string()),
            Type::Array(_) => Some("Array".to_string()),
            Type::Map(_, _) => Some("Map".to_string()),
            Type::Set(_) => Some("Set".to_string()),
            Type::Named { name, .. } => Some(name),
            Type::Pid => Some("Pid".to_string()),
            Type::Ref => Some("Ref".to_string()),
            Type::IO(_) => Some("IO".to_string()),
            Type::Tuple(_) => Some("Tuple".to_string()),
            // Variant types can have methods
            Type::Variant(vt) => Some(vt.name.clone()),
            // Type variables can't be looked up
            Type::Var(_) | Type::TypeParam(_) => None,
            // These don't have methods
            Type::Unit | Type::Never | Type::Function(_) | Type::Record(_) => None,
        }
    }

    /// Find the best matching overload for a function call based on argument types.
    /// Returns the INDEX of the overload whose parameter types best match the (resolved) argument types.
    /// This is the key to proper overload resolution in HM type inference.
    fn find_best_overload_idx(&self, overloads: &[&FunctionType], arg_types: &[Type]) -> Option<usize> {
        // Resolve argument types through current substitution
        let resolved_args: Vec<Type> = arg_types.iter()
            .map(|t| self.env.apply_subst(t))
            .collect();

        // Try each overload and score how well it matches
        let mut best_match: Option<(usize, usize)> = None;  // (index, score)

        for (idx, &overload) in overloads.iter().enumerate() {
            if overload.params.len() != arg_types.len() {
                continue;
            }

            let mut score = 0;
            let mut compatible = true;

            for (param_ty, arg_ty) in overload.params.iter().zip(resolved_args.iter()) {
                match self.types_compatible(param_ty, arg_ty) {
                    Some(s) => score += s,
                    None => {
                        compatible = false;
                        break;
                    }
                }
            }

            if compatible {
                // Higher score = better match
                // Prefer exact matches over generic matches
                if best_match.is_none() || score > best_match.unwrap().1 {
                    best_match = Some((idx, score));
                }
            }
        }

        best_match.map(|(idx, _)| idx)
    }

    /// Check if a parameter type is compatible with an argument type.
    /// Returns Some(score) if compatible (higher = better match), None if incompatible.
    fn types_compatible(&self, param_ty: &Type, arg_ty: &Type) -> Option<usize> {
        // Resolve both types through substitution
        let param = self.env.apply_subst(param_ty);
        let arg = self.env.apply_subst(arg_ty);

        match (&param, &arg) {
            // Exact match - highest score
            _ if param == arg => Some(100),

            // Type variable in param position - can match anything (generic function)
            (Type::Var(_), _) | (Type::TypeParam(_), _) => Some(10),

            // Type variable in arg position - can match anything (not yet resolved)
            (_, Type::Var(_)) | (_, Type::TypeParam(_)) => Some(50),

            // Same base type with potentially different type args
            (Type::List(p), Type::List(a)) => {
                self.types_compatible(p, a).map(|s| s / 2 + 50)
            }
            (Type::Array(p), Type::Array(a)) => {
                self.types_compatible(p, a).map(|s| s / 2 + 50)
            }
            (Type::Map(pk, pv), Type::Map(ak, av)) => {
                let k_compat = self.types_compatible(pk, ak)?;
                let v_compat = self.types_compatible(pv, av)?;
                Some((k_compat + v_compat) / 2 + 30)
            }
            (Type::Set(p), Type::Set(a)) => {
                self.types_compatible(p, a).map(|s| s / 2 + 50)
            }

            // Named types must match by name
            (Type::Named { name: pn, .. }, Type::Named { name: an, .. }) => {
                if pn == an { Some(90) } else { None }
            }

            // Variant types must match by name
            (Type::Variant(pv), Type::Variant(av)) => {
                if pv.name == av.name { Some(90) } else { None }
            }

            // Tuples must have same length and compatible elements
            (Type::Tuple(pt), Type::Tuple(at)) => {
                if pt.len() != at.len() {
                    return None;
                }
                let mut total = 0;
                for (p, a) in pt.iter().zip(at.iter()) {
                    total += self.types_compatible(p, a)?;
                }
                Some(total / pt.len().max(1) + 50)
            }

            // Function types
            (Type::Function(pf), Type::Function(af)) => {
                if pf.params.len() != af.params.len() {
                    return None;
                }
                let mut total = 0;
                for (p, a) in pf.params.iter().zip(af.params.iter()) {
                    total += self.types_compatible(p, a)?;
                }
                total += self.types_compatible(&pf.ret, &af.ret)?;
                Some(total / (pf.params.len() + 1).max(1) + 30)
            }

            // Incompatible types
            _ => None,
        }
    }

    /// Instantiate a polymorphic function type.
    /// Replaces type parameters AND existing Var types with fresh type variables.
    /// This ensures each call site gets its own type variables for proper unification.
    pub fn instantiate_function(&mut self, func_ty: &FunctionType) -> Type {
        // Collect all existing Var IDs in the function type
        let mut var_ids: Vec<u32> = Vec::new();
        self.collect_var_ids(&Type::Function(func_ty.clone()), &mut var_ids);

        // Create mapping from old Var IDs to fresh ones
        let mut var_subst: HashMap<u32, Type> = HashMap::new();
        for var_id in var_ids {
            if !var_subst.contains_key(&var_id) {
                var_subst.insert(var_id, self.fresh());
            }
        }

        // Also handle explicit type parameters
        let mut param_subst: HashMap<String, Type> = HashMap::new();
        for type_param in &func_ty.type_params {
            let fresh_var = self.fresh();

            // Add trait constraints for this fresh variable
            if let Type::Var(var_id) = fresh_var {
                for constraint in &type_param.constraints {
                    self.add_trait_bound(var_id, constraint.clone());
                }
            }

            param_subst.insert(type_param.name.clone(), fresh_var);
        }

        // Substitute both Var IDs and type parameters
        let instantiated_params: Vec<Type> = func_ty.params
            .iter()
            .map(|p| self.freshen_type(p, &var_subst, &param_subst))
            .collect();
        let instantiated_ret = self.freshen_type(&func_ty.ret, &var_subst, &param_subst);

        Type::Function(FunctionType { required_params: func_ty.required_params,
            type_params: vec![], // Instantiated function has no type params
            params: instantiated_params,
            ret: Box::new(instantiated_ret),
        })
    }

    /// Collect all Var IDs in a type
    fn collect_var_ids(&self, ty: &Type, ids: &mut Vec<u32>) {
        match ty {
            Type::Var(id) => {
                if !ids.contains(id) {
                    ids.push(*id);
                }
            }
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) => {
                self.collect_var_ids(elem, ids);
            }
            Type::Map(k, v) => {
                self.collect_var_ids(k, ids);
                self.collect_var_ids(v, ids);
            }
            Type::Tuple(elems) => {
                for elem in elems {
                    self.collect_var_ids(elem, ids);
                }
            }
            Type::Function(ft) => {
                for p in &ft.params {
                    self.collect_var_ids(p, ids);
                }
                self.collect_var_ids(&ft.ret, ids);
            }
            _ => {}
        }
    }

    /// Freshen a type by replacing Var IDs and TypeParams with fresh variables
    fn freshen_type(&self, ty: &Type, var_subst: &HashMap<u32, Type>, param_subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Var(id) => var_subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
            Type::TypeParam(name) => param_subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
            Type::List(elem) => Type::List(Box::new(self.freshen_type(elem, var_subst, param_subst))),
            Type::Array(elem) => Type::Array(Box::new(self.freshen_type(elem, var_subst, param_subst))),
            Type::Set(elem) => Type::Set(Box::new(self.freshen_type(elem, var_subst, param_subst))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.freshen_type(k, var_subst, param_subst)),
                Box::new(self.freshen_type(v, var_subst, param_subst)),
            ),
            Type::Tuple(elems) => Type::Tuple(
                elems.iter().map(|e| self.freshen_type(e, var_subst, param_subst)).collect()
            ),
            Type::Function(ft) => Type::Function(FunctionType { required_params: ft.required_params,
                type_params: vec![],
                params: ft.params.iter().map(|p| self.freshen_type(p, var_subst, param_subst)).collect(),
                ret: Box::new(self.freshen_type(&ft.ret, var_subst, param_subst)),
            }),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.freshen_type(a, var_subst, param_subst)).collect(),
            },
            Type::IO(inner) => Type::IO(Box::new(self.freshen_type(inner, var_subst, param_subst))),
            Type::Record(rec) => Type::Record(RecordType {
                name: rec.name.clone(),
                fields: rec.fields.iter()
                    .map(|(n, t, m)| (n.clone(), self.freshen_type(t, var_subst, param_subst), *m))
                    .collect(),
            }),
            _ => ty.clone(),
        }
    }

    /// Solve all constraints through unification.
    /// Uses a fixed-point iteration with a maximum iteration limit to avoid infinite loops
    /// when constraints reference unresolved type variables.
    /// The span of the most recent error during constraint solving.
    /// This is used to provide precise error locations.
    pub fn last_error_span(&self) -> Option<Span> {
        self.last_error_span
    }

    pub fn solve(&mut self) -> Result<(), TypeError> {
        const MAX_ITERATIONS: usize = 1000;
        let mut iteration = 0;
        let mut deferred_count = 0;

        while let Some(constraint) = self.constraints.pop() {
            iteration += 1;
            if iteration > MAX_ITERATIONS {
                // Too many iterations - likely an infinite loop due to unresolved constraints
                // Return success but leave type variables unresolved
                return Ok(());
            }

            match constraint {
                Constraint::Equal(t1, t2, span) => {
                    // Track the most recent constraint span for error reporting
                    // Only update if this constraint has a span - preserve previous span otherwise
                    // This helps with constraints added during inference that don't have spans
                    if span.is_some() {
                        self.current_constraint_span = span;
                    }
                    // Use this constraint's span if it has one, otherwise use the tracked span
                    let error_span = span.or(self.current_constraint_span);
                    if let Err(e) = self.unify_types(&t1, &t2) {
                        self.last_error_span = error_span;
                        return Err(e);
                    }
                    deferred_count = 0; // Made progress
                }
                Constraint::HasTrait(ty, trait_name) => {
                    let resolved = self.env.apply_subst(&ty);
                    match &resolved {
                        Type::Var(var_id) => {
                            // Track trait bound on the type variable
                            self.add_trait_bound(*var_id, trait_name);
                            deferred_count = 0; // Made progress (recorded the bound)
                        }
                        _ => {
                            // Concrete type - check if it implements the trait
                            if !self.env.implements(&resolved, &trait_name) {
                                return Err(TypeError::MissingTraitImpl {
                                    ty: resolved.display(),
                                    trait_name,
                                });
                            }
                            deferred_count = 0; // Made progress
                        }
                    }
                }
                Constraint::HasField(ty, field, expected_ty) => {
                    let resolved = self.env.apply_subst(&ty);
                    match &resolved {
                        Type::Record(rec) => {
                            if let Some((_, actual_ty, _)) =
                                rec.fields.iter().find(|(n, _, _)| n == &field)
                            {
                                self.unify_types(actual_ty, &expected_ty)?;
                            } else {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                });
                            }
                            deferred_count = 0; // Made progress
                        }
                        Type::Named { name, .. } => {
                            // Check if this is a known record type
                            if let Some(crate::TypeDef::Record { fields, .. }) = self.env.lookup_type(name).cloned() {
                                if let Some((_, actual_ty, _)) =
                                    fields.iter().find(|(n, _, _)| n == &field)
                                {
                                    self.unify_types(&actual_ty, &expected_ty)?;
                                    deferred_count = 0; // Made progress
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: resolved.display(),
                                        field,
                                    });
                                }
                            } else {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                });
                            }
                        }
                        Type::Var(_) => {
                            // Defer: we don't know the type yet
                            deferred_count += 1;
                            if deferred_count > self.constraints.len() + 1 {
                                // We've gone around the entire queue without progress
                                // Drop this constraint to avoid infinite loop
                                continue;
                            }
                            self.constraints
                                .push(Constraint::HasField(resolved, field, expected_ty));
                        }
                        Type::Tuple(elems) => {
                            // Handle tuple field access like .0, .1, etc.
                            if let Ok(idx) = field.parse::<usize>() {
                                if idx < elems.len() {
                                    self.unify_types(&elems[idx], &expected_ty)?;
                                    deferred_count = 0; // Made progress
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: resolved.display(),
                                        field,
                                    });
                                }
                            } else {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                });
                            }
                        }
                        _ => {
                            return Err(TypeError::NoSuchField {
                                ty: resolved.display(),
                                field,
                            });
                        }
                    }
                }
            }
        }

        // Post-solve: check pending method calls now that types are resolved
        self.check_pending_method_calls()?;

        // Apply substitution to all stored expression types
        self.finalize_expr_types();

        Ok(())
    }

    /// Apply the current substitution to all stored expression types.
    /// This should be called after solve() to get the resolved types.
    /// Also resolves any remaining TypeParams using type_param_mappings.
    pub fn finalize_expr_types(&mut self) {
        let resolved: HashMap<Span, Type> = self.expr_types
            .iter()
            .map(|(span, ty)| (*span, self.apply_full_subst(ty)))
            .collect();
        self.expr_types = resolved;
    }

    /// Get the resolved type for an expression by its span.
    /// Returns None if the expression wasn't inferred or if types haven't been finalized.
    pub fn get_expr_type(&self, span: &Span) -> Option<&Type> {
        self.expr_types.get(span)
    }

    /// Get all expression types (for transferring to the compiler).
    pub fn take_expr_types(&mut self) -> HashMap<Span, Type> {
        std::mem::take(&mut self.expr_types)
    }

    /// Check pending method calls after constraint solving.
    /// This enables UFCS type checking for cases where the receiver type
    /// wasn't known during initial inference (e.g., status from Server.bind).
    fn check_pending_method_calls(&mut self) -> Result<(), TypeError> {
        // Take ownership of pending calls to avoid borrow issues
        let pending = std::mem::take(&mut self.pending_method_calls);

        for call in pending {
            // Resolve the receiver type now that constraints are solved
            let resolved_receiver = self.env.apply_subst(&call.receiver_ty);

            // Try to get the type name
            if let Some(type_name) = self.get_type_name(&resolved_receiver) {
                let qualified_name = format!("{}.{}", type_name, call.method_name);

                // Look up the function by qualified name first
                let fn_type_opt = self.env.functions.get(&qualified_name).cloned().or_else(|| {
                    // Fallback: for List methods, try unqualified lookup
                    // This handles multi-param methods like map, fold, filter which aren't
                    // registered with "List." prefix to avoid eager unification at call site.
                    // Here (post-inference), we can safely do full type checking.
                    if type_name == "List" {
                        self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                            // Verify it's actually a list method (first param is [a])
                            matches!(ft.params.first(), Some(Type::List(_)))
                        })
                    } else {
                        None
                    }
                });

                if let Some(fn_type) = fn_type_opt {
                    let func_ty = self.instantiate_function(&fn_type);
                    if let Type::Function(ft) = func_ty {
                        // Check arity (accounting for optional parameters)
                        let min_args = ft.required_params.unwrap_or(ft.params.len());
                        let max_args = ft.params.len();
                        let provided = call.arg_types.len();
                        if provided < min_args || provided > max_args {
                            // Set error span for precise error location
                            self.last_error_span = call.span;
                            return Err(TypeError::ArityMismatch {
                                expected: if min_args == max_args { max_args } else { min_args },
                                found: provided,
                            });
                        }

                        // Resolve arg types and check against params
                        for (param_ty, arg_ty) in ft.params.iter().zip(call.arg_types.iter()) {
                            let resolved_arg = self.env.apply_subst(arg_ty);
                            let resolved_param = self.env.apply_subst(param_ty);

                            // Handle function type mismatches between curried and uncurried forms
                            if let (Type::Function(expected_fn), Type::Function(actual_fn)) =
                                (&resolved_param, &resolved_arg)
                            {
                                // Case 1: Expected is curried (e.g., b -> a -> b), actual is uncurried (e.g., (b, a) -> b)
                                // This is common for fold-like functions where the signature is parsed as curried
                                // but the lambda is written as uncurried
                                let expected_is_curried = matches!(*expected_fn.ret, Type::Function(_));

                                if expected_is_curried && actual_fn.params.len() > 1 {
                                    // Uncurry the expected type and compare
                                    let (expected_params, expected_ret) = Self::uncurry_function_type(expected_fn);

                                    // Check parameter counts match
                                    if expected_params.len() == actual_fn.params.len() {
                                        // Unify each parameter
                                        for (expected_p, actual_p) in expected_params.iter().zip(actual_fn.params.iter()) {
                                            self.unify_types(actual_p, expected_p)?;
                                        }
                                        // Unify return types
                                        self.unify_types(&actual_fn.ret, &expected_ret)?;
                                        continue; // Skip the normal unification below
                                    }
                                }

                                // Case 2: Expected takes 1 tuple and actual takes multiple
                                // In Nostos, (a, b) => creates a 2-param lambda, but map on [(Int,Int)]
                                // expects (tuple) -> b. Skip unification for these cases to avoid false errors.
                                let expected_single_tuple = expected_fn.params.len() == 1
                                    && matches!(expected_fn.params.first(), Some(Type::Tuple(_)));
                                let actual_multi = actual_fn.params.len() > 1;
                                if expected_single_tuple && actual_multi {
                                    continue; // Skip unification - runtime handles conversion
                                }
                            }

                            self.unify_types(&resolved_arg, &resolved_param)?;
                        }

                        // Unify return type
                        let resolved_ret = self.env.apply_subst(&call.ret_ty);
                        self.unify_types(&resolved_ret, &ft.ret)?;
                    }
                } else {
                    // Method not found for this type in the inference environment
                    // For known builtin methods that only work on List/String, we can
                    // definitively report an error if called on a primitive.
                    // For unknown methods, assume they might be trait methods.
                    let list_only_methods = [
                        "length", "len", "head", "tail", "init", "last", "nth",
                        "push", "pop", "get", "set", "slice", "concat", "reverse", "sort",
                        "map", "filter", "fold", "any", "all", "find", "position",
                        "unique", "flatten", "zip", "unzip", "take", "drop",
                        "empty", "isEmpty", "sum", "product", "indexOf", "sortBy",
                        "intersperse", "spanList", "groupBy", "transpose", "pairwise",
                        "isSorted", "isSortedBy", "enumerate",
                    ];
                    let string_methods = ["split", "trim", "trimStart", "trimEnd",
                                           "toUpper", "toLower", "startsWith", "endsWith",
                                           "contains", "replace", "chars"];

                    let is_list_only = list_only_methods.contains(&call.method_name.as_str());
                    let is_string_method = string_methods.contains(&call.method_name.as_str());
                    let is_primitive = matches!(type_name.as_str(), "Int" | "Float" | "Bool" | "Char" |
                                                           "Int8" | "Int16" | "Int32" | "Int64" |
                                                           "UInt8" | "UInt16" | "UInt32" | "UInt64" |
                                                           "Float32" | "Float64" | "BigInt" | "Decimal");

                    // Report error for list-only methods on non-List types
                    if is_list_only && is_primitive {
                        self.last_error_span = call.span;
                        return Err(TypeError::UndefinedMethod {
                            method: call.method_name.clone(),
                            receiver_type: type_name.to_string(),
                        });
                    }

                    // Report error for string methods on non-String primitives
                    if is_string_method && is_primitive && type_name.as_str() != "String" {
                        self.last_error_span = call.span;
                        return Err(TypeError::UndefinedMethod {
                            method: call.method_name.clone(),
                            receiver_type: type_name.to_string(),
                        });
                    }

                    // For other methods, don't report - might be trait implementations
                }
            } else {
                // Type is still not resolved to a concrete name
                // Check for known methods on resolved primitive types
                let resolved = self.env.apply_subst(&call.receiver_ty);
                let list_only_methods = [
                    "length", "len", "head", "tail", "init", "last", "nth",
                    "push", "pop", "get", "set", "slice", "concat", "reverse", "sort",
                    "map", "filter", "fold", "any", "all", "find", "position",
                    "unique", "flatten", "zip", "unzip", "take", "drop",
                    "empty", "isEmpty", "sum", "product", "indexOf", "sortBy",
                    "intersperse", "spanList", "groupBy", "transpose", "pairwise",
                    "isSorted", "isSortedBy", "enumerate",
                ];

                let is_list_only = list_only_methods.contains(&call.method_name.as_str());
                let is_primitive = matches!(&resolved,
                    Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                    Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                    Type::Float | Type::Float32 | Type::Float64 |
                    Type::Bool | Type::Char | Type::BigInt | Type::Decimal
                );

                if is_list_only && is_primitive {
                    let type_name = resolved.display();
                    self.last_error_span = call.span;
                    return Err(TypeError::UndefinedMethod {
                        method: call.method_name.clone(),
                        receiver_type: type_name,
                    });
                }
            }
        }

        Ok(())
    }

    /// Apply substitution and resolve TypeParams to their mapped types.
    /// This is the complete type resolution that should be used after solve().
    pub fn apply_full_subst(&self, ty: &Type) -> Type {
        // First apply the main substitution
        let ty = self.env.apply_subst(ty);
        // Then resolve any TypeParams
        self.resolve_type_params(&ty)
    }

    /// Recursively resolve TypeParams in a type using type_param_mappings.
    /// If a TypeParam isn't mapped yet, it stays as-is (for finalize_expr_types).
    fn resolve_type_params(&self, ty: &Type) -> Type {
        match ty {
            Type::TypeParam(name) => {
                if let Some(mapped) = self.type_param_mappings.get(name) {
                    self.resolve_type_params(mapped)
                } else {
                    ty.clone()
                }
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|t| self.resolve_type_params(t)).collect())
            }
            Type::List(elem) => Type::List(Box::new(self.resolve_type_params(elem))),
            Type::Array(elem) => Type::Array(Box::new(self.resolve_type_params(elem))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.resolve_type_params(k)),
                Box::new(self.resolve_type_params(v)),
            ),
            Type::Set(elem) => Type::Set(Box::new(self.resolve_type_params(elem))),
            Type::Function(f) => Type::Function(FunctionType {
                type_params: f.type_params.clone(),
                params: f.params.iter().map(|t| self.resolve_type_params(t)).collect(),
                ret: Box::new(self.resolve_type_params(&f.ret)),
                required_params: f.required_params,
            }),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|t| self.resolve_type_params(t)).collect(),
            },
            Type::IO(inner) => Type::IO(Box::new(self.resolve_type_params(inner))),
            Type::Record(rec) => Type::Record(RecordType {
                name: rec.name.clone(),
                fields: rec.fields.iter()
                    .map(|(n, t, m)| (n.clone(), self.resolve_type_params(t), *m))
                    .collect(),
            }),
            _ => ty.clone(),
        }
    }

    /// Convert all TypeParams in a type to type variables.
    /// Uses existing mappings if available, creates fresh vars for new TypeParams.
    /// This is used for recursive function calls to avoid TypeParams in the type system.
    fn instantiate_type_params(&mut self, ty: &Type) -> Type {
        match ty {
            Type::TypeParam(name) => {
                if let Some(mapped) = self.type_param_mappings.get(name).cloned() {
                    mapped
                } else {
                    let fresh = self.fresh();
                    self.type_param_mappings.insert(name.clone(), fresh.clone());
                    fresh
                }
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|t| self.instantiate_type_params(t)).collect())
            }
            Type::List(elem) => Type::List(Box::new(self.instantiate_type_params(elem))),
            Type::Array(elem) => Type::Array(Box::new(self.instantiate_type_params(elem))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.instantiate_type_params(k)),
                Box::new(self.instantiate_type_params(v)),
            ),
            Type::Set(elem) => Type::Set(Box::new(self.instantiate_type_params(elem))),
            Type::Function(f) => Type::Function(FunctionType {
                type_params: f.type_params.clone(),
                params: f.params.iter().map(|t| self.instantiate_type_params(t)).collect(),
                ret: Box::new(self.instantiate_type_params(&f.ret)),
                required_params: f.required_params,
            }),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|t| self.instantiate_type_params(t)).collect(),
            },
            Type::IO(inner) => Type::IO(Box::new(self.instantiate_type_params(inner))),
            Type::Record(rec) => Type::Record(RecordType {
                name: rec.name.clone(),
                fields: rec.fields.iter()
                    .map(|(n, t, m)| (n.clone(), self.instantiate_type_params(t), *m))
                    .collect(),
            }),
            _ => ty.clone(),
        }
    }

    /// Unify two types, updating the substitution.
    fn unify_types(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = self.apply_full_subst(t1);
        let t2 = self.apply_full_subst(t2);

        match (&t1, &t2) {
            // Same type - nothing to do
            _ if t1 == t2 => Ok(()),

            // Float and Float64 are aliases and should unify
            (Type::Float, Type::Float64) | (Type::Float64, Type::Float) => Ok(()),

            // Int and Int64 are aliases and should unify
            (Type::Int, Type::Int64) | (Type::Int64, Type::Int) => Ok(()),

            // Variable on the left
            (Type::Var(id), _) => {
                if t2.contains_var(*id) {
                    return Err(TypeError::OccursCheck(
                        format!("?{}", id),
                        t2.display(),
                    ));
                }
                // If t2 is also a Var, merge their trait bounds
                if let Type::Var(id2) = t2 {
                    // Merge trait bounds from id2 into id
                    let bounds2 = self.trait_bounds.get(&id2).cloned().unwrap_or_default();
                    for bound in bounds2 {
                        self.add_trait_bound(*id, bound);
                    }
                }
                self.env.substitution.insert(*id, t2);
                Ok(())
            }

            // Variable on the right
            (_, Type::Var(id)) => {
                if t1.contains_var(*id) {
                    return Err(TypeError::OccursCheck(
                        format!("?{}", id),
                        t1.display(),
                    ));
                }
                // If t1 is also a Var, merge their trait bounds
                if let Type::Var(id1) = t1 {
                    // Merge trait bounds from id1 into id
                    let bounds1 = self.trait_bounds.get(&id1).cloned().unwrap_or_default();
                    for bound in bounds1 {
                        self.add_trait_bound(*id, bound);
                    }
                }
                self.env.substitution.insert(*id, t1);
                Ok(())
            }

            // TypeParam on the left - treat like a type variable
            // This allows generic functions to be called with concrete types
            // IMPORTANT: Create a fresh type variable for the TypeParam so apply_subst works
            (Type::TypeParam(name), _) => {
                if let Some(existing) = self.type_param_mappings.get(name).cloned() {
                    // We've seen this TypeParam before - unify with existing type variable
                    self.unify_types(&existing, &t2)
                } else {
                    // First time seeing this TypeParam - create a fresh type variable
                    let fresh_var = self.fresh();
                    self.type_param_mappings.insert(name.clone(), fresh_var.clone());
                    // Unify the fresh variable with the target type
                    self.unify_types(&fresh_var, &t2)
                }
            }

            // TypeParam on the right - treat like a type variable
            (_, Type::TypeParam(name)) => {
                if let Some(existing) = self.type_param_mappings.get(name).cloned() {
                    // We've seen this TypeParam before - unify with existing type variable
                    self.unify_types(&t1, &existing)
                } else {
                    // First time seeing this TypeParam - create a fresh type variable
                    let fresh_var = self.fresh();
                    self.type_param_mappings.insert(name.clone(), fresh_var.clone());
                    // Unify the fresh variable with the target type
                    self.unify_types(&t1, &fresh_var)
                }
            }

            // Tuples
            (Type::Tuple(elems1), Type::Tuple(elems2)) => {
                if elems1.len() != elems2.len() {
                    return Err(TypeError::ArityMismatch {
                        expected: elems1.len(),
                        found: elems2.len(),
                    });
                }
                for (e1, e2) in elems1.iter().zip(elems2.iter()) {
                    self.unify_types(e1, e2)?;
                }
                Ok(())
            }

            // Lists
            (Type::List(elem1), Type::List(elem2)) => self.unify_types(elem1, elem2),

            // Arrays
            (Type::Array(elem1), Type::Array(elem2)) => self.unify_types(elem1, elem2),

            // Sets
            (Type::Set(elem1), Type::Set(elem2)) => self.unify_types(elem1, elem2),

            // Maps
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                self.unify_types(k1, k2)?;
                self.unify_types(v1, v2)
            }

            // Functions
            (Type::Function(f1), Type::Function(f2)) => {
                // Handle optional parameters: allow unifying when one has more params
                // if the extra params have defaults
                let (func_with_more, func_with_less) = if f1.params.len() >= f2.params.len() {
                    (f1, f2)
                } else {
                    (f2, f1)
                };

                // Check if the arity difference is acceptable
                let min_required = func_with_more.required_params.unwrap_or(func_with_more.params.len());
                let max_params = func_with_more.params.len();
                let provided = func_with_less.params.len();

                // Allow the call if provided args are between min_required and max_params
                if provided < min_required || provided > max_params {
                    // But if required_params is None on the larger function, it might just
                    // be an HM inference function without complete info - be lenient
                    if func_with_more.required_params.is_some() || provided > max_params {
                        return Err(TypeError::ArityMismatch {
                            expected: min_required,
                            found: provided,
                        });
                    }
                    // Otherwise, allow it - the compiler will catch real errors
                }

                // Unify the params that were provided
                for (p1, p2) in func_with_more.params.iter().zip(func_with_less.params.iter()) {
                    self.unify_types(p1, p2)?;
                }
                self.unify_types(&f1.ret, &f2.ret)
            }

            // Records (structural)
            (Type::Record(rec1), Type::Record(rec2)) => {
                // Build field maps
                let fields1: HashMap<_, _> = rec1
                    .fields
                    .iter()
                    .map(|(n, t, m)| (n.clone(), (t.clone(), *m)))
                    .collect();
                let fields2: HashMap<_, _> = rec2
                    .fields
                    .iter()
                    .map(|(n, t, m)| (n.clone(), (t.clone(), *m)))
                    .collect();

                // Check all fields in rec1 exist in rec2
                for (name, (ty1, _)) in &fields1 {
                    if let Some((ty2, _)) = fields2.get(name) {
                        self.unify_types(ty1, ty2)?;
                    } else {
                        return Err(TypeError::MissingField(name.clone()));
                    }
                }

                // Check no extra fields in rec2
                for name in fields2.keys() {
                    if !fields1.contains_key(name) {
                        return Err(TypeError::ExtraField(name.clone()));
                    }
                }

                Ok(())
            }

            // Named types - resolve aliases before comparing
            (
                Type::Named { name: n1, args: a1 },
                Type::Named { name: n2, args: a2 },
            ) => {
                // Resolve type aliases (e.g., "RNode" -> "stdlib.rhtml.RNode")
                let resolved1 = self.env.resolve_type_name(n1);
                let resolved2 = self.env.resolve_type_name(n2);

                if resolved1 != resolved2 {
                    return Err(TypeError::UnificationFailed(t1.display(), t2.display()));
                }
                if a1.len() != a2.len() {
                    return Err(TypeError::TypeArityMismatch {
                        expected: a1.len(),
                        found: a2.len(),
                    });
                }
                for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                    self.unify_types(arg1, arg2)?;
                }
                Ok(())
            }

            // IO
            (Type::IO(inner1), Type::IO(inner2)) => self.unify_types(inner1, inner2),

            // Typed arrays are compatible with List for type inference purposes
            // Int64Array <-> List[Int] or List[Int64]
            (Type::Named { name, args }, Type::List(elem))
            | (Type::List(elem), Type::Named { name, args })
                if name == "Int64Array" && args.is_empty() =>
            {
                // Int64Array is compatible with List[Int] or List[Int64]
                match elem.as_ref() {
                    Type::Int | Type::Int64 => Ok(()),
                    Type::Var(_) => {
                        // Constrain the element type to Int
                        self.unify_types(elem, &Type::Int)
                    }
                    _ => Err(TypeError::UnificationFailed(t1.display(), t2.display())),
                }
            }
            // Float64Array <-> List[Float] or List[Float64]
            (Type::Named { name, args }, Type::List(elem))
            | (Type::List(elem), Type::Named { name, args })
                if name == "Float64Array" && args.is_empty() =>
            {
                // Float64Array is compatible with List[Float] or List[Float64]
                match elem.as_ref() {
                    Type::Float | Type::Float64 => Ok(()),
                    Type::Var(_) => {
                        // Constrain the element type to Float
                        self.unify_types(elem, &Type::Float)
                    }
                    _ => Err(TypeError::UnificationFailed(t1.display(), t2.display())),
                }
            }

            // Named "List" unifies with Type::List - handles user-defined List types
            (Type::Named { name, args }, Type::List(elem))
            | (Type::List(elem), Type::Named { name, args })
                if (name == "List" || name.ends_with(".List")) && args.len() == 1 =>
            {
                self.unify_types(&args[0], elem)
            }

            // Named "Set" unifies with Type::Set
            (Type::Named { name, args }, Type::Set(elem))
            | (Type::Set(elem), Type::Named { name, args })
                if (name == "Set" || name.ends_with(".Set")) && args.len() == 1 =>
            {
                self.unify_types(&args[0], elem)
            }

            // Named "Array" unifies with Type::Array
            (Type::Named { name, args }, Type::Array(elem))
            | (Type::Array(elem), Type::Named { name, args })
                if (name == "Array" || name.ends_with(".Array")) && args.len() == 1 =>
            {
                self.unify_types(&args[0], elem)
            }

            // Named "Map" unifies with Type::Map
            (Type::Named { name, args }, Type::Map(k, v))
            | (Type::Map(k, v), Type::Named { name, args })
                if (name == "Map" || name.ends_with(".Map")) && args.len() == 2 =>
            {
                self.unify_types(&args[0], k)?;
                self.unify_types(&args[1], v)
            }

            // Mismatch
            _ => Err(TypeError::UnificationFailed(t1.display(), t2.display())),
        }
    }

    // =========================================================================
    // AST Type Inference
    // =========================================================================

    /// Convert an AST TypeExpr to an internal Type.
    pub fn type_from_ast(&mut self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Name(ident) => {
                let name = &ident.node;
                match name.as_str() {
                    // Integer types
                    "Int" => Type::Int,
                    "Int8" => Type::Int8,
                    "Int16" => Type::Int16,
                    "Int32" => Type::Int32,
                    "Int64" => Type::Int64,
                    "UInt8" => Type::UInt8,
                    "UInt16" => Type::UInt16,
                    "UInt32" => Type::UInt32,
                    "UInt64" => Type::UInt64,
                    // Float types
                    "Float" => Type::Float,
                    "Float32" => Type::Float32,
                    "Float64" => Type::Float64,
                    // Arbitrary precision
                    "BigInt" => Type::BigInt,
                    "Decimal" => Type::Decimal,
                    // Other primitives
                    "Bool" => Type::Bool,
                    "Char" => Type::Char,
                    "String" => Type::String,
                    "Pid" => Type::Pid,
                    "Ref" => Type::Ref,
                    "Never" => Type::Never,
                    _ => Type::Named {
                        // Resolve through type aliases (e.g., Option -> stdlib.list.Option)
                        name: self.env.resolve_type_name(name),
                        args: vec![],
                    }
                }
            }
            TypeExpr::Generic(name, args) => {
                let type_args: Vec<_> = args.iter().map(|a| self.type_from_ast(a)).collect();
                let name_str = &name.node;
                match name_str.as_str() {
                    "List" if type_args.len() == 1 => Type::List(Box::new(type_args[0].clone())),
                    "Array" if type_args.len() == 1 => Type::Array(Box::new(type_args[0].clone())),
                    "Set" if type_args.len() == 1 => Type::Set(Box::new(type_args[0].clone())),
                    "Map" if type_args.len() == 2 => Type::Map(
                        Box::new(type_args[0].clone()),
                        Box::new(type_args[1].clone()),
                    ),
                    "IO" if type_args.len() == 1 => Type::IO(Box::new(type_args[0].clone())),
                    _ => Type::Named {
                        // Resolve through type aliases (e.g., Option -> stdlib.list.Option)
                        name: self.env.resolve_type_name(name_str),
                        args: type_args,
                    },
                }
            }
            TypeExpr::Function(params, ret) => {
                let param_types: Vec<_> = params.iter().map(|p| self.type_from_ast(p)).collect();
                let ret_type = self.type_from_ast(ret);
                Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_type),
                })
            }
            TypeExpr::Record(fields) => {
                let field_types: Vec<_> = fields
                    .iter()
                    .map(|(name, ty)| (name.node.clone(), self.type_from_ast(ty), false))
                    .collect();
                Type::Record(RecordType {
                    name: None,
                    fields: field_types,
                })
            }
            TypeExpr::Tuple(elems) => {
                let elem_types: Vec<_> = elems.iter().map(|e| self.type_from_ast(e)).collect();
                Type::Tuple(elem_types)
            }
            TypeExpr::Unit => Type::Unit,
        }
    }

    /// Infer the type of an expression and store it for later retrieval.
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        let ty = self.infer_expr_inner(expr)?;
        // Store the inferred type keyed by the expression's span
        self.expr_types.insert(expr.span(), ty.clone());
        Ok(ty)
    }

    /// Internal implementation of type inference for an expression.
    fn infer_expr_inner(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match expr {
            // Literals - integers
            Expr::Int(_, _) => Ok(Type::Int),
            Expr::Int8(_, _) => Ok(Type::Int8),
            Expr::Int16(_, _) => Ok(Type::Int16),
            Expr::Int32(_, _) => Ok(Type::Int32),
            // Unsigned integers
            Expr::UInt8(_, _) => Ok(Type::UInt8),
            Expr::UInt16(_, _) => Ok(Type::UInt16),
            Expr::UInt32(_, _) => Ok(Type::UInt32),
            Expr::UInt64(_, _) => Ok(Type::UInt64),
            // Floats
            Expr::Float(_, _) => Ok(Type::Float),
            Expr::Float32(_, _) => Ok(Type::Float32),
            // Arbitrary precision
            Expr::BigInt(_, _) => Ok(Type::BigInt),
            Expr::Decimal(_, _) => Ok(Type::Decimal),
            // Other literals
            Expr::Bool(_, _) => Ok(Type::Bool),
            Expr::Char(_, _) => Ok(Type::Char),
            Expr::String(_, _) => Ok(Type::String),
            Expr::Unit(_) => Ok(Type::Unit),

            // Wildcard is only valid in pattern contexts, not as a standalone expression
            Expr::Wildcard(span) => Err(TypeError::InvalidWildcard(*span)),

            // Variable lookup
            Expr::Var(ident) => {
                let name = &ident.node;
                // First check if it's a constructor
                if let Some(ty) = self.lookup_constructor(name) {
                    return Ok(ty);
                }
                // Then check bindings
                if let Some((ty, _)) = self.env.lookup(name) {
                    Ok(ty.clone())
                } else if let Some(sig) = self.env.functions.get(name).cloned() {
                    // For recursive calls (calling the function being inferred),
                    // DON'T instantiate with completely fresh vars - use consistent mappings
                    // for TypeParams so that constraints are preserved
                    let is_recursive = self.current_function.as_ref() == Some(name);
                    if is_recursive {
                        // Convert TypeParams to type variables using consistent mappings
                        // This ensures the same TypeParam (e.g., "a") always maps to the same var
                        let instantiated = self.instantiate_type_params(&Type::Function(sig));
                        Ok(instantiated)
                    } else {
                        // Instantiate polymorphic functions with fresh type variables
                        Ok(self.instantiate_function(&sig))
                    }
                } else {
                    Err(TypeError::UnknownIdent(name.clone()))
                }
            }

            // Binary operations
            Expr::BinOp(left, op, right, _) => self.infer_binop(left, *op, right),

            // Unary operations
            Expr::UnaryOp(op, operand, _) => {
                let operand_ty = self.infer_expr(operand)?;
                match op {
                    UnaryOp::Neg => {
                        // Neg works on Int or Float
                        let result_ty = self.fresh();
                        self.unify(operand_ty.clone(), result_ty.clone());
                        // We need to check it's numeric at solve time
                        Ok(result_ty)
                    }
                    UnaryOp::Not => {
                        self.unify(operand_ty, Type::Bool);
                        Ok(Type::Bool)
                    }
                }
            }

            // Function call (with optional type args)
            Expr::Call(func, _type_args, args, call_span) => {
                // Check if any arguments are named - if so, skip strict overload resolution
                // because FunctionType doesn't store parameter names and we can't reorder
                let has_named_args = args.iter().any(|arg| matches!(arg, CallArg::Named(_, _)));

                // Infer argument types first (needed for overload resolution)
                let mut arg_types = Vec::new();
                for arg in args {
                    let expr = match arg {
                        CallArg::Positional(e) | CallArg::Named(_, e) => e,
                    };
                    arg_types.push(self.infer_expr(expr)?);
                }

                // Special handling for simple variable function calls: try all overloads
                let func_ty = if let Expr::Var(ident) = func.as_ref() {
                    let name = &ident.node;
                    // First check if it's a local binding (lambdas, let-bound functions)
                    if let Some((ty, _)) = self.env.lookup(name) {
                        ty.clone()
                    } else {
                        // Get ALL overloads and find the best match based on argument types
                        // Clone to avoid borrow issues with instantiate_function
                        let mut overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(name, args.len())
                            .into_iter().cloned().collect();

                        // Also check for type-qualified versions (e.g., String.length for length(s) where s: String)
                        // This enables UFCS-style resolution for bare function calls
                        if args.len() == 1 && !arg_types.is_empty() {
                            let resolved_arg_ty = self.env.apply_subst(&arg_types[0]);
                            if let Some(type_name) = self.get_type_name(&resolved_arg_ty) {
                                let qualified_name = format!("{}.{}", type_name, name);
                                let qualified_overloads: Vec<FunctionType> = self.env
                                    .lookup_all_functions_with_arity(&qualified_name, 1)
                                    .into_iter().cloned().collect();
                                overloads.extend(qualified_overloads);
                            }
                        }

                        if !overloads.is_empty() {
                            let is_recursive = self.current_function.as_ref() == Some(name);
                            // Find best overload index first to avoid borrow issues
                            // Skip strict matching if named args present (can't reorder without param names)
                            let best_idx = if has_named_args {
                                Some(0) // Use first/wildcard overload for named args
                            } else {
                                let overload_refs: Vec<&FunctionType> = overloads.iter().collect();
                                self.find_best_overload_idx(&overload_refs, &arg_types)
                            };
                            let sig = overloads[best_idx.unwrap_or(0)].clone();
                            if is_recursive {
                                // Convert TypeParams to type variables using consistent mappings
                                // This ensures the same TypeParam (e.g., "a") always maps to the same var
                                self.instantiate_type_params(&Type::Function(sig))
                            } else {
                                self.instantiate_function(&sig)
                            }
                        } else if let Some(ty) = self.lookup_constructor(name) {
                            // Could be a constructor call
                            ty
                        } else {
                            return Err(TypeError::UnknownIdent(name.clone()));
                        }
                    }
                } else if let Expr::FieldAccess(base_expr, field, _) = func.as_ref() {
                    // Special handling for qualified function calls like good.addf(1, 2)
                    if let Expr::Var(base_ident) = base_expr.as_ref() {
                        let qualified_name = format!("{}.{}", base_ident.node, field.node);
                        // Clone to avoid borrow issues
                        let overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(&qualified_name, args.len())
                            .into_iter().cloned().collect();
                        if !overloads.is_empty() {
                            // Find best overload index first to avoid borrow issues
                            // Skip strict matching if named args present
                            let best_idx = if has_named_args {
                                Some(0)
                            } else {
                                let overload_refs: Vec<&FunctionType> = overloads.iter().collect();
                                self.find_best_overload_idx(&overload_refs, &arg_types)
                            };
                            let sig = overloads[best_idx.unwrap_or(0)].clone();
                            self.instantiate_function(&sig)
                        } else {
                            // Fall back to inferring as regular field access
                            self.infer_expr(func)?
                        }
                    } else {
                        self.infer_expr(func)?
                    }
                } else {
                    // For non-variable function expressions, infer normally
                    self.infer_expr(func)?
                };

                // When named arguments are present, we can't do positional type checking
                // because FunctionType doesn't store parameter names. Skip strict unification
                // and just extract the return type - the compiler will verify types after
                // it reorders arguments based on parameter names.
                if has_named_args {
                    if let Type::Function(ft) = &func_ty {
                        return Ok((*ft.ret).clone());
                    }
                    // If not a function type, fall through to normal unification
                    // which will report an appropriate error
                }

                let ret_ty = self.fresh();
                let expected_func_ty = Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: arg_types,
                    ret: Box::new(ret_ty.clone()),
                });

                // Use unify_at with the call span for precise error reporting
                self.unify_at(func_ty, expected_func_ty, *call_span);
                Ok(ret_ty)
            }

            // Lambda
            Expr::Lambda(params, body, _) => {
                // Save current bindings
                let saved_bindings = self.env.bindings.clone();

                let mut param_types = Vec::new();
                for param in params {
                    let param_ty = self.fresh();
                    self.infer_pattern(param, &param_ty)?;
                    param_types.push(param_ty);
                }

                let body_ty = self.infer_expr(body)?;

                // Restore bindings
                self.env.bindings = saved_bindings;

                Ok(Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(body_ty),
                }))
            }

            // If expression
            Expr::If(cond, then_branch, else_branch, _) => {
                let cond_ty = self.infer_expr(cond)?;
                self.unify(cond_ty, Type::Bool);

                let then_ty = self.infer_expr(then_branch)?;
                let else_ty = self.infer_expr(else_branch)?;
                self.unify(then_ty.clone(), else_ty);

                Ok(then_ty)
            }

            // Match expression
            Expr::Match(scrutinee, arms, _) => {
                let scrutinee_ty = self.infer_expr(scrutinee)?;
                let result_ty = self.fresh();

                for arm in arms {
                    self.infer_match_arm(arm, &scrutinee_ty, &result_ty)?;
                }

                Ok(result_ty)
            }

            // Tuple
            Expr::Tuple(elems, _) => {
                let mut elem_types = Vec::new();
                for elem in elems {
                    elem_types.push(self.infer_expr(elem)?);
                }
                Ok(Type::Tuple(elem_types))
            }

            // List
            Expr::List(elems, tail, _) => {
                let elem_ty = self.fresh();

                for elem in elems {
                    let ty = self.infer_expr(elem)?;
                    self.unify(ty, elem_ty.clone());
                }

                if let Some(tail_expr) = tail {
                    let tail_ty = self.infer_expr(tail_expr)?;
                    self.unify(tail_ty, Type::List(Box::new(elem_ty.clone())));
                }

                Ok(Type::List(Box::new(elem_ty)))
            }

            // Map
            Expr::Map(pairs, _) => {
                let key_ty = self.fresh();
                let val_ty = self.fresh();

                for (k, v) in pairs {
                    let k_ty = self.infer_expr(k)?;
                    let v_ty = self.infer_expr(v)?;
                    self.unify(k_ty, key_ty.clone());
                    self.unify(v_ty, val_ty.clone());
                }

                self.require_trait(key_ty.clone(), "Hash");
                self.require_trait(key_ty.clone(), "Eq");

                Ok(Type::Map(Box::new(key_ty), Box::new(val_ty)))
            }

            // Set
            Expr::Set(elems, _) => {
                let elem_ty = self.fresh();

                for elem in elems {
                    let ty = self.infer_expr(elem)?;
                    self.unify(ty, elem_ty.clone());
                }

                self.require_trait(elem_ty.clone(), "Hash");
                self.require_trait(elem_ty.clone(), "Eq");

                Ok(Type::Set(Box::new(elem_ty)))
            }

            // Record construction
            Expr::Record(name, fields, _) => {
                let type_name = &name.node;
                // Resolve type name through aliases to get qualified name
                let resolved_type_name = self.env.resolve_type_name(type_name);

                // Check for record update pattern: first field is positional (base), rest have named fields
                // E.g., Point(p, x: 10) means "update p with x = 10"
                if fields.len() >= 2 {
                    if let RecordField::Positional(base_expr) = &fields[0] {
                        // Check if any subsequent field is named
                        let has_named = fields[1..].iter().any(|f| matches!(f, RecordField::Named(_, _)));
                        if has_named {
                            // This is a record update - infer base type and require it matches the record type
                            let base_ty = self.infer_expr(base_expr)?;
                            let expected_ty = Type::Named {
                                name: resolved_type_name.clone(),
                                args: vec![],
                            };
                            self.unify(base_ty, expected_ty.clone());

                            // Infer and check types of named update fields
                            if let Some(TypeDef::Record { fields: def_fields, .. }) = self.env.lookup_type(&resolved_type_name).cloned() {
                                for field in &fields[1..] {
                                    if let RecordField::Named(fname, expr) = field {
                                        let ty = self.infer_expr(expr)?;
                                        if let Some((_, fty, _)) = def_fields.iter().find(|(n, _, _)| n == &fname.node) {
                                            self.unify(ty, fty.clone());
                                        } else {
                                            return Err(TypeError::NoSuchField {
                                                ty: resolved_type_name.clone(),
                                                field: fname.node.clone(),
                                            });
                                        }
                                    }
                                }
                            }

                            return Ok(expected_ty);
                        }
                    }
                }

                if let Some(TypeDef::Record {
                    fields: def_fields, ..
                }) = self.env.lookup_type(&resolved_type_name).cloned()
                {
                    let mut provided = HashMap::new();
                    for field in fields {
                        match field {
                            RecordField::Positional(expr) => {
                                // Positional args match in order
                                let ty = self.infer_expr(expr)?;
                                let idx = provided.len();
                                if idx < def_fields.len() {
                                    let (fname, fty, _) = &def_fields[idx];
                                    self.unify(ty, fty.clone());
                                    provided.insert(fname.clone(), ());
                                }
                            }
                            RecordField::Named(fname, expr) => {
                                let ty = self.infer_expr(expr)?;
                                if let Some((_, fty, _)) =
                                    def_fields.iter().find(|(n, _, _)| n == &fname.node)
                                {
                                    self.unify(ty, fty.clone());
                                    provided.insert(fname.node.clone(), ());
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: resolved_type_name.clone(),
                                        field: fname.node.clone(),
                                    });
                                }
                            }
                        }
                    }

                    Ok(Type::Named {
                        name: resolved_type_name.clone(),
                        args: vec![],
                    })
                } else if let Some(ctor_ty) = self.lookup_constructor(&resolved_type_name) {
                    // This is a variant constructor call (e.g., Some(42), Good("hello"), Nil)
                    if fields.is_empty() {
                        // Unit constructor (no arguments) - ctor_ty is just the result type
                        Ok(ctor_ty)
                    } else {
                        // Positional/named constructor - ctor_ty is a Function type
                        // Infer argument types and unify with constructor parameter types
                        let mut arg_types = Vec::new();
                        for field in fields {
                            let expr = match field {
                                RecordField::Positional(e) => e,
                                RecordField::Named(_, e) => e,
                            };
                            arg_types.push(self.infer_expr(expr)?);
                        }

                        let ret_ty = self.fresh();
                        let expected_func_ty = Type::Function(FunctionType {
                            required_params: None,
                            type_params: vec![],
                            params: arg_types,
                            ret: Box::new(ret_ty.clone()),
                        });

                        // Unify constructor type with expected - this will catch type mismatches
                        self.unify(ctor_ty, expected_func_ty);
                        Ok(ret_ty)
                    }
                } else {
                    Err(TypeError::UnknownType(resolved_type_name.clone()))
                }
            }

            // Field access
            Expr::FieldAccess(expr, field, _) => {
                // Check if this is a qualified function name like "Panel.show"
                // In this case, look up "Panel.show" in the functions environment
                if let Expr::Var(base_ident) = expr.as_ref() {
                    let qualified_name = format!("{}.{}", base_ident.node, field.node);
                    if let Some(fn_type) = self.env.functions.get(&qualified_name).cloned() {
                        // Return the function type, instantiated with fresh type variables
                        return Ok(self.instantiate_function(&fn_type));
                    }
                }

                // Regular field access on a value
                let expr_ty = self.infer_expr(expr)?;
                let field_ty = self.fresh();
                self.require_field(expr_ty, &field.node, field_ty.clone());
                Ok(field_ty)
            }

            // Index access
            Expr::Index(container, index, span) => {
                let container_ty = self.infer_expr(container)?;
                let index_ty = self.infer_expr(index)?;
                let elem_ty = self.fresh();

                // Apply substitutions to get resolved container type
                let resolved_container = self.env.apply_subst(&container_ty);

                // Check if container is a custom Named type - allow indexing without List unification
                // The compiler will dispatch to {typeLower}Get function
                match &resolved_container {
                    Type::Named { .. } | Type::Var(_) => {
                        // Custom type or unresolved type variable - allow Int indexing
                        // The compiler will handle dispatch to typeGet function
                        self.unify(index_ty, Type::Int);
                        Ok(elem_ty)
                    }
                    _ => {
                        // Container could be List, Array, Map, or String
                        let list_ty = Type::List(Box::new(elem_ty.clone()));

                        // For now, assume Int index for lists/arrays
                        // This is simplified - a full impl would handle Map[K, V]
                        self.unify(index_ty, Type::Int);

                        // Try to unify with List or Array
                        // We'll handle this via constraint solving
                        self.constraints.push(Constraint::Equal(
                            container_ty,
                            list_ty,
                            Some(*span),
                        ));

                        Ok(elem_ty)
                    }
                }
            }

            // Block
            Expr::Block(stmts, _) => {
                let saved_bindings = self.env.bindings.clone();
                let mut result_ty = Type::Unit;

                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i == stmts.len() - 1;
                    match stmt {
                        Stmt::Expr(expr) => {
                            let ty = self.infer_expr(expr)?;
                            if is_last {
                                result_ty = ty;
                            }
                        }
                        Stmt::Let(binding) => {
                            self.infer_binding(binding)?;
                        }
                        Stmt::Assign(target, expr, _) => {
                            let expr_ty = self.infer_expr(expr)?;
                            match target {
                                nostos_syntax::ast::AssignTarget::Var(ident) => {
                                    if let Some((var_ty, mutable)) = self.env.lookup(&ident.node) {
                                        if !mutable {
                                            return Err(TypeError::ImmutableBinding(
                                                ident.node.clone(),
                                            ));
                                        }
                                        self.unify(expr_ty, var_ty.clone());
                                    } else {
                                        return Err(TypeError::UnknownIdent(ident.node.clone()));
                                    }
                                }
                                nostos_syntax::ast::AssignTarget::Field(obj, field) => {
                                    let obj_ty = self.infer_expr(obj)?;
                                    self.require_field(obj_ty, &field.node, expr_ty);
                                }
                                nostos_syntax::ast::AssignTarget::Index(obj, idx) => {
                                    let _obj_ty = self.infer_expr(obj)?;
                                    let _idx_ty = self.infer_expr(idx)?;
                                    // Index assignment - simplified for now
                                }
                            }
                        }
                    }
                }

                self.env.bindings = saved_bindings;
                Ok(result_ty)
            }

            // Method call
            Expr::MethodCall(receiver, method, args, call_span) => {
                // Check if this is a qualified function call like "Panel.show(id)"
                // where receiver is Record("Panel", [], _) and method is "show"
                if let Expr::Record(type_ident, fields, _) = receiver.as_ref() {
                    if fields.is_empty() {
                        // This is a namespace-like call: Panel.show(id)
                        let qualified_name = format!("{}.{}", type_ident.node, method.node);
                        if let Some(fn_type) = self.env.functions.get(&qualified_name).cloned() {
                            // Found the function - infer argument types and unify
                            let mut arg_types = Vec::new();
                            for arg in args {
                                let expr = match arg {
                                    CallArg::Positional(e) | CallArg::Named(_, e) => e,
                                };
                                arg_types.push(self.infer_expr(expr)?);
                            }

                            // Instantiate the function type
                            let func_ty = self.instantiate_function(&fn_type);
                            if let Type::Function(ft) = func_ty {
                                // Check arity (accounting for optional parameters)
                                let min_args = ft.required_params.unwrap_or(ft.params.len());
                                let max_args = ft.params.len();
                                if arg_types.len() < min_args || arg_types.len() > max_args {
                                    return Err(TypeError::ArityMismatch {
                                        expected: if min_args == max_args { max_args } else { min_args },
                                        found: arg_types.len(),
                                    });
                                }
                                // Use unify_at with call span for precise error reporting
                                for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                                    self.unify_at(arg_ty.clone(), param_ty.clone(), *call_span);
                                }
                                return Ok(*ft.ret);
                            }
                        }
                    }
                }

                // Check if receiver is a module name: module.func(args)
                // This handles qualified function calls like good.addf(1, 2)
                if let Expr::Var(module_ident) = receiver.as_ref() {
                    let qualified_name = format!("{}.{}", module_ident.node, method.node);
                    // Use arity-aware lookup since function keys include arity suffix
                    if let Some(fn_type) = self.env.lookup_function_with_arity(&qualified_name, args.len()).cloned() {
                        // Found the function - infer argument types and unify
                        let mut arg_types = Vec::new();
                        for arg in args {
                            let expr = match arg {
                                CallArg::Positional(e) | CallArg::Named(_, e) => e,
                            };
                            arg_types.push(self.infer_expr(expr)?);
                        }

                        // Instantiate the function type
                        let func_ty = self.instantiate_function(&fn_type);
                        if let Type::Function(ft) = func_ty {
                            // Check arity (accounting for optional parameters)
                            let min_args = ft.required_params.unwrap_or(ft.params.len());
                            let max_args = ft.params.len();
                            if arg_types.len() < min_args || arg_types.len() > max_args {
                                self.last_error_span = Some(*call_span);
                                return Err(TypeError::ArityMismatch {
                                    expected: if min_args == max_args { max_args } else { min_args },
                                    found: arg_types.len(),
                                });
                            }
                            // Use unify_at with call span for precise error reporting
                            for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                                self.unify_at(arg_ty.clone(), param_ty.clone(), *call_span);
                            }
                            return Ok(*ft.ret);
                        }
                    }
                }

                // Regular method call on a value - use UFCS lookup
                let receiver_ty = self.infer_expr(receiver)?;
                let mut arg_types = vec![receiver_ty.clone()];
                for arg in args {
                    let expr = match arg {
                        CallArg::Positional(e) | CallArg::Named(_, e) => e,
                    };
                    arg_types.push(self.infer_expr(expr)?);
                }

                // Create a fresh return type variable
                let ret_ty = self.fresh();

                // Try immediate UFCS lookup if receiver type is already resolved
                if let Some(type_name) = self.get_type_name(&receiver_ty) {
                    let qualified_name = format!("{}.{}", type_name, method.node);
                    // Only look up qualified names (e.g., List.get, List.head)
                    // We DON'T fall back to unqualified names here because that would
                    // cause eager unification for multi-param methods like map/fold,
                    // which breaks nested pattern inference in lambdas.
                    // Unregistered methods use deferred checking via pending_method_calls.
                    if let Some(fn_type) = self.env.functions.get(&qualified_name).cloned() {
                        // Found the function - instantiate and unify
                        let func_ty = self.instantiate_function(&fn_type);
                        if let Type::Function(ft) = func_ty {
                            // Check arity (including receiver as first arg, accounting for optional parameters)
                            let min_args = ft.required_params.unwrap_or(ft.params.len());
                            let max_args = ft.params.len();
                            if arg_types.len() < min_args || arg_types.len() > max_args {
                                self.last_error_span = Some(*call_span);
                                return Err(TypeError::ArityMismatch {
                                    expected: if min_args == max_args { max_args } else { min_args },
                                    found: arg_types.len(),
                                });
                            }
                            // Unify argument types with parameter types (with call span)
                            for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                                self.unify_at(arg_ty.clone(), param_ty.clone(), *call_span);
                            }
                            // Unify return type
                            self.unify_at(ret_ty.clone(), *ft.ret, *call_span);
                            return Ok(ret_ty);
                        }
                    }
                }

                // Record method call for post-solve UFCS type checking
                // This handles cases where receiver type is not yet resolved
                self.pending_method_calls.push(PendingMethodCall {
                    receiver_ty: receiver_ty.clone(),
                    method_name: method.node.clone(),
                    arg_types: arg_types.clone(),
                    ret_ty: ret_ty.clone(),
                    span: Some(*call_span),
                });

                Ok(ret_ty)
            }

            // Try expression (error propagation)
            Expr::Try_(inner, _) => {
                let inner_ty = self.infer_expr(inner)?;
                let ok_ty = self.fresh();
                let err_ty = self.fresh();

                // inner should be Result[T, E]
                let result_type_name = self.env.resolve_type_name("Result");
                self.unify(
                    inner_ty,
                    Type::Named {
                        name: result_type_name,
                        args: vec![ok_ty.clone(), err_ty],
                    },
                );

                Ok(ok_ty)
            }

            // Do block (IO monad)
            Expr::Do(stmts, _) => {
                let saved_bindings = self.env.bindings.clone();
                let mut result_ty = Type::Unit;

                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i == stmts.len() - 1;
                    match stmt {
                        nostos_syntax::ast::DoStmt::Bind(pat, expr) => {
                            let expr_ty = self.infer_expr(expr)?;
                            // expr should have type IO[T]
                            let inner_ty = self.fresh();
                            self.unify(expr_ty, Type::IO(Box::new(inner_ty.clone())));
                            self.infer_pattern(pat, &inner_ty)?;
                        }
                        nostos_syntax::ast::DoStmt::Expr(expr) => {
                            let ty = self.infer_expr(expr)?;
                            if is_last {
                                // Last expr should be IO[T]
                                let inner_ty = self.fresh();
                                self.unify(ty, Type::IO(Box::new(inner_ty.clone())));
                                result_ty = inner_ty;
                            }
                        }
                    }
                }

                self.env.bindings = saved_bindings;
                Ok(Type::IO(Box::new(result_ty)))
            }

            // Spawn
            Expr::Spawn(_, func, args, _) => {
                let func_ty = self.infer_expr(func)?;
                let mut arg_types = Vec::new();
                for arg in args {
                    arg_types.push(self.infer_expr(arg)?);
                }

                let ret_ty = self.fresh();
                let expected = Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: arg_types,
                    ret: Box::new(ret_ty),
                });
                self.unify(func_ty, expected);

                Ok(Type::Pid)
            }

            // Send
            Expr::Send(pid, msg, _) => {
                let pid_ty = self.infer_expr(pid)?;
                let _msg_ty = self.infer_expr(msg)?;
                self.unify(pid_ty, Type::Pid);
                Ok(Type::Unit)
            }

            // Receive
            Expr::Receive(arms, timeout, _) => {
                let result_ty = self.fresh();
                let msg_ty = self.fresh();

                for arm in arms {
                    self.infer_match_arm(arm, &msg_ty, &result_ty)?;
                }

                if let Some((timeout_expr, timeout_body)) = timeout {
                    let timeout_ty = self.infer_expr(timeout_expr)?;
                    self.unify(timeout_ty, Type::Int);
                    let body_ty = self.infer_expr(timeout_body)?;
                    self.unify(body_ty, result_ty.clone());
                }

                Ok(result_ty)
            }

            // Record update
            Expr::RecordUpdate(name, base, fields, _) => {
                let resolved_name = self.env.resolve_type_name(&name.node);
                let base_ty = self.infer_expr(base)?;
                let expected_ty = Type::Named {
                    name: resolved_name,
                    args: vec![],
                };
                self.unify(base_ty, expected_ty.clone());

                for field in fields {
                    if let RecordField::Named(fname, expr) = field {
                        let field_ty = self.infer_expr(expr)?;
                        self.require_field(expected_ty.clone(), &fname.node, field_ty);
                    }
                }

                Ok(expected_ty)
            }

            // Try/catch
            // Note: The catch block can return a different type than the try block
            // because exceptions are dynamic. We return a fresh type variable that
            // is NOT unified with either branch, allowing the type to be inferred
            // from usage context (e.g., assignment target or function argument).
            Expr::Try(body, catch_arms, finally, _) => {
                // Infer try body type but don't unify with result
                let _body_ty = self.infer_expr(body)?;

                // Exception type is always String (thrown messages)
                let err_ty = Type::String;

                // Infer catch arms but don't unify with result
                for arm in catch_arms {
                    let catch_result_ty = self.fresh();
                    self.infer_match_arm(arm, &err_ty, &catch_result_ty)?;
                }

                if let Some(finally_expr) = finally {
                    let _ = self.infer_expr(finally_expr)?;
                }

                // Return a fresh type - the actual type depends on which branch executes
                // at runtime. The type checker can't determine this statically.
                Ok(self.fresh())
            }

            // Quote/Splice - macro-related
            Expr::Quote(inner, _) => {
                let _ = self.infer_expr(inner)?;
                Ok(Type::Named {
                    name: "Expr".to_string(),
                    args: vec![],
                })
            }
            Expr::Splice(inner, _) => {
                let inner_ty = self.infer_expr(inner)?;
                self.unify(
                    inner_ty,
                    Type::Named {
                        name: "Expr".to_string(),
                        args: vec![],
                    },
                );
                // Result depends on context
                Ok(self.fresh())
            }

            // While loop: condition must be Bool, returns Unit
            Expr::While(cond, body, _) => {
                let cond_ty = self.infer_expr(cond)?;
                self.unify(cond_ty, Type::Bool);
                let _ = self.infer_expr(body)?;
                Ok(Type::Unit)
            }

            // For loop: iterates from start to end (both Int), returns Unit
            Expr::For(var, start, end, body, _) => {
                let start_ty = self.infer_expr(start)?;
                let end_ty = self.infer_expr(end)?;
                self.unify(start_ty, Type::Int);
                self.unify(end_ty, Type::Int);
                // Add loop variable to scope
                let saved_bindings = self.env.bindings.clone();
                self.env.bind(var.node.clone(), Type::Int, false);
                let _ = self.infer_expr(body)?;
                self.env.bindings = saved_bindings;
                Ok(Type::Unit)
            }

            // Break: optional value, returns Never (doesn't produce a value normally)
            Expr::Break(value, _) => {
                if let Some(val) = value {
                    let _ = self.infer_expr(val)?;
                }
                Ok(Type::Unit) // Break doesn't continue execution normally
            }

            // Continue: returns Never (doesn't produce a value normally)
            Expr::Continue(_) => Ok(Type::Unit),

            // Return: optional value, returns Never (doesn't produce a value normally)
            Expr::Return(value, _) => {
                if let Some(val) = value {
                    let _ = self.infer_expr(val)?;
                }
                Ok(Type::Unit) // Return doesn't continue execution normally
            }
        }
    }

    /// Infer types for a binary operation.
    fn infer_binop(&mut self, left: &Expr, op: BinOp, right: &Expr) -> Result<Type, TypeError> {
        let left_ty = self.infer_expr(left)?;
        let right_ty = self.infer_expr(right)?;

        match op {
            // Arithmetic: both operands numeric, with implicit Int->Float coercion
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                // Apply substitutions to get resolved types
                let resolved_left = self.env.apply_subst(&left_ty);
                let resolved_right = self.env.apply_subst(&right_ty);

                // Check for Int/Float mixing - result is Float due to implicit coercion
                match (&resolved_left, &resolved_right) {
                    (Type::Int, Type::Float) | (Type::Float, Type::Int) |
                    (Type::Int, Type::Float64) | (Type::Float64, Type::Int) |
                    (Type::Int64, Type::Float) | (Type::Float, Type::Int64) |
                    (Type::Int64, Type::Float64) | (Type::Float64, Type::Int64) => {
                        // Mixed Int/Float arithmetic - result is Float
                        Ok(Type::Float)
                    }
                    (Type::Int, Type::BigInt) | (Type::BigInt, Type::Int) |
                    (Type::Int64, Type::BigInt) | (Type::BigInt, Type::Int64) => {
                        // Mixed Int/BigInt arithmetic - result is BigInt
                        Ok(Type::BigInt)
                    }
                    // Scalar operations: custom type OP numeric type
                    // e.g., Vec * Float, Vec + Int - compiler generates scalar function calls
                    (Type::Named { .. }, Type::Float | Type::Int | Type::Float64 | Type::Int64) |
                    (Type::Named { .. }, Type::Var(_)) => {
                        // Custom type with numeric scalar - require Num trait and return the custom type
                        // The compiler will dispatch to {typeLower}{Op}Scalar function
                        self.require_trait(left_ty.clone(), "Num");
                        Ok(left_ty)
                    }
                    // Handle type variable on left with numeric on right - defer unification
                    // This allows Vec (as Var) * Float to succeed when the Var is later resolved to Vec
                    (Type::Var(_), Type::Float | Type::Int | Type::Float64 | Type::Int64) => {
                        // Don't unify - let the compiler handle scalar dispatch if left resolves to Named
                        self.require_trait(left_ty.clone(), "Num");
                        Ok(left_ty)
                    }
                    _ => {
                        // Same type or type variables - standard unification
                        self.unify(left_ty.clone(), right_ty);
                        self.require_trait(left_ty.clone(), "Num");
                        Ok(left_ty)
                    }
                }
            }

            // Comparison: both operands same type with Ord, result Bool
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                self.unify(left_ty.clone(), right_ty);
                self.require_trait(left_ty, "Ord");
                Ok(Type::Bool)
            }

            // Equality: both operands same type with Eq, result Bool
            BinOp::Eq | BinOp::NotEq => {
                self.unify(left_ty.clone(), right_ty);
                self.require_trait(left_ty, "Eq");
                Ok(Type::Bool)
            }

            // Logical: both operands Bool, result Bool
            BinOp::And | BinOp::Or => {
                self.unify(left_ty, Type::Bool);
                self.unify(right_ty, Type::Bool);
                Ok(Type::Bool)
            }

            // Concat: lists or strings
            BinOp::Concat => {
                // List concatenation allows heterogeneous types (design decision)
                // [String] ++ [Int] produces a mixed list at runtime
                // We just check both are lists, but don't unify their element types
                match (&left_ty, &right_ty) {
                    (Type::List(_), Type::List(_)) => {
                        // Return a list with a fresh type variable
                        // since we can't know statically what types the mixed list contains
                        Ok(Type::List(Box::new(self.fresh())))
                    }
                    _ => {
                        // If not both lists, fall back to old behavior (unify)
                        // This handles cases like string concatenation
                        self.unify(left_ty.clone(), right_ty);
                        Ok(left_ty)
                    }
                }
            }

            // Cons: h :: t means prepend h to list t
            // left is the element, right is the list
            BinOp::Cons => {
                // right should be a List of left's type
                let expected_list = Type::List(Box::new(left_ty));
                self.unify(right_ty, expected_list.clone());
                Ok(expected_list)
            }

            // Pipe: f |> g means g(f)
            BinOp::Pipe => {
                let result_ty = self.fresh();
                let expected_func = Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: vec![left_ty],
                    ret: Box::new(result_ty.clone()),
                });
                self.unify(right_ty, expected_func);
                Ok(result_ty)
            }
        }
    }

    /// Infer types for a pattern, binding variables in the environment.
    pub fn infer_pattern(&mut self, pat: &Pattern, expected: &Type) -> Result<(), TypeError> {
        match pat {
            Pattern::Wildcard(_) => Ok(()),

            Pattern::Var(ident) => {
                self.env.bind(ident.node.clone(), expected.clone(), false);
                Ok(())
            }

            Pattern::Int(_, _) => {
                self.unify(expected.clone(), Type::Int);
                Ok(())
            }

            Pattern::Int8(_, _) => {
                self.unify(expected.clone(), Type::Int8);
                Ok(())
            }

            Pattern::Int16(_, _) => {
                self.unify(expected.clone(), Type::Int16);
                Ok(())
            }

            Pattern::Int32(_, _) => {
                self.unify(expected.clone(), Type::Int32);
                Ok(())
            }

            Pattern::UInt8(_, _) => {
                self.unify(expected.clone(), Type::UInt8);
                Ok(())
            }

            Pattern::UInt16(_, _) => {
                self.unify(expected.clone(), Type::UInt16);
                Ok(())
            }

            Pattern::UInt32(_, _) => {
                self.unify(expected.clone(), Type::UInt32);
                Ok(())
            }

            Pattern::UInt64(_, _) => {
                self.unify(expected.clone(), Type::UInt64);
                Ok(())
            }

            Pattern::BigInt(_, _) => {
                self.unify(expected.clone(), Type::BigInt);
                Ok(())
            }

            Pattern::Float(_, _) => {
                self.unify(expected.clone(), Type::Float);
                Ok(())
            }

            Pattern::Float32(_, _) => {
                self.unify(expected.clone(), Type::Float32);
                Ok(())
            }

            Pattern::Decimal(_, _) => {
                self.unify(expected.clone(), Type::Decimal);
                Ok(())
            }

            Pattern::String(_, _) => {
                self.unify(expected.clone(), Type::String);
                Ok(())
            }

            Pattern::Char(_, _) => {
                self.unify(expected.clone(), Type::Char);
                Ok(())
            }

            Pattern::Bool(_, _) => {
                self.unify(expected.clone(), Type::Bool);
                Ok(())
            }

            Pattern::Unit(_) => {
                self.unify(expected.clone(), Type::Unit);
                Ok(())
            }

            Pattern::Tuple(elems, _) => {
                let elem_types: Vec<_> = elems.iter().map(|_| self.fresh()).collect();
                self.unify(expected.clone(), Type::Tuple(elem_types.clone()));
                for (elem, ty) in elems.iter().zip(elem_types.iter()) {
                    self.infer_pattern(elem, ty)?;
                }
                Ok(())
            }

            Pattern::List(list_pat, _) => {
                let elem_ty = self.fresh();
                self.unify(expected.clone(), Type::List(Box::new(elem_ty.clone())));

                match list_pat {
                    nostos_syntax::ast::ListPattern::Empty => {}
                    nostos_syntax::ast::ListPattern::Cons(elems, tail) => {
                        for elem in elems {
                            self.infer_pattern(elem, &elem_ty)?;
                        }
                        if let Some(tail_pat) = tail {
                            self.infer_pattern(tail_pat, &Type::List(Box::new(elem_ty)))?;
                        }
                    }
                }
                Ok(())
            }

            Pattern::StringCons(string_pat, _) => {
                // String cons pattern matches against strings
                self.unify(expected.clone(), Type::String);

                match string_pat {
                    nostos_syntax::ast::StringPattern::Empty => {}
                    nostos_syntax::ast::StringPattern::Cons(_, tail_pat) => {
                        // tail_pat binds to the rest of the string
                        self.infer_pattern(tail_pat, &Type::String)?;
                    }
                }
                Ok(())
            }

            Pattern::Record(fields, _) => {
                for field in fields {
                    match field {
                        RecordPatternField::Punned(name) => {
                            let field_ty = self.fresh();
                            self.require_field(expected.clone(), &name.node, field_ty.clone());
                            self.env.bind(name.node.clone(), field_ty, false);
                        }
                        RecordPatternField::Named(name, pat) => {
                            let field_ty = self.fresh();
                            self.require_field(expected.clone(), &name.node, field_ty.clone());
                            self.infer_pattern(pat, &field_ty)?;
                        }
                        RecordPatternField::Rest(_) => {}
                    }
                }
                Ok(())
            }

            Pattern::Variant(name, fields, _) => {
                // Look up constructor
                if let Some(ctor_ty) = self.lookup_constructor(&name.node) {
                    match fields {
                        VariantPatternFields::Unit => {
                            self.unify(expected.clone(), ctor_ty);
                        }
                        VariantPatternFields::Positional(pats) => {
                            // Constructor should be a function from fields to the type
                            if let Type::Function(f) = &ctor_ty {
                                // Unify expected with the constructor's return type
                                self.unify(expected.clone(), (*f.ret).clone());

                                // Try to get concrete field types from the constructor's return type
                                // This enables proper type narrowing through pattern matching
                                // The return type is the variant type (e.g., Expr for Add constructor)
                                // But only for NON-PARAMETRIC types - parametric types like Tree[T]
                                // need to use the substituted params from the constructor function.
                                let concrete_field_types = if let Type::Named { name: type_name, args } = &*f.ret {
                                    // Only use concrete field types for non-parametric types
                                    // Parametric types need the substituted f.params instead
                                    if args.is_empty() {
                                        self.env.lookup_variant_field_types(type_name, &name.node)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };

                                if let Some(field_types) = concrete_field_types {
                                    // Use concrete field types from the type definition
                                    // This is key for type narrowing: when matching Add(e1, e2) on Expr,
                                    // we know e1 and e2 are Expr, not just type variables
                                    for (pat, field_ty) in pats.iter().zip(field_types.iter()) {
                                        self.infer_pattern(pat, field_ty)?;
                                    }
                                } else {
                                    // Use constructor params (already substituted for parametric types)
                                    for (pat, param_ty) in pats.iter().zip(f.params.iter()) {
                                        self.infer_pattern(pat, param_ty)?;
                                    }
                                }
                            } else {
                                self.unify(expected.clone(), ctor_ty);
                            }
                        }
                        VariantPatternFields::Named(fields) => {
                            for field in fields {
                                match field {
                                    RecordPatternField::Punned(fname) => {
                                        let field_ty = self.fresh();
                                        self.env.bind(fname.node.clone(), field_ty, false);
                                    }
                                    RecordPatternField::Named(fname, pat) => {
                                        let field_ty = self.fresh();
                                        self.infer_pattern(pat, &field_ty)?;
                                        let _ = fname;
                                    }
                                    RecordPatternField::Rest(_) => {}
                                }
                            }
                            self.unify(expected.clone(), ctor_ty);
                        }
                    }
                    Ok(())
                } else {
                    Err(TypeError::UnknownConstructor(name.node.clone()))
                }
            }

            Pattern::Or(pats, _) => {
                for pat in pats {
                    self.infer_pattern(pat, expected)?;
                }
                Ok(())
            }

            Pattern::Pin(expr, _) => {
                let pin_ty = self.infer_expr(expr)?;
                self.unify(expected.clone(), pin_ty);
                Ok(())
            }

            Pattern::Map(entries, _) => {
                let key_ty = self.fresh();
                let val_ty = self.fresh();
                self.unify(expected.clone(), Type::Map(Box::new(key_ty.clone()), Box::new(val_ty.clone())));

                for (key_expr, val_pat) in entries {
                    let k_ty = self.infer_expr(key_expr)?;
                    self.unify(k_ty, key_ty.clone());
                    self.infer_pattern(val_pat, &val_ty)?;
                }
                Ok(())
            }

            Pattern::Set(elements, _) => {
                let elem_ty = self.fresh();
                self.unify(expected.clone(), Type::Set(Box::new(elem_ty.clone())));

                for elem_pat in elements {
                    self.infer_pattern(elem_pat, &elem_ty)?;
                }
                Ok(())
            }
        }
    }

    /// Look up a constructor and return its type.
    fn lookup_constructor(&mut self, name: &str) -> Option<Type> {
        // Search through all types for this constructor
        // Sort type names to prioritize user-defined types over stdlib types
        // User types (no dots or not starting with "stdlib.") come first
        let types_clone = self.env.types.clone();
        let mut type_names: Vec<_> = types_clone.keys().collect();
        type_names.sort_by(|a, b| {
            // Only stdlib. prefix indicates stdlib - user types can have module prefixes too
            let a_is_stdlib = a.starts_with("stdlib.");
            let b_is_stdlib = b.starts_with("stdlib.");
            match (a_is_stdlib, b_is_stdlib) {
                (false, true) => std::cmp::Ordering::Less,    // User types first
                (true, false) => std::cmp::Ordering::Greater, // Stdlib types last
                _ => a.cmp(b),  // Same category: alphabetical
            }
        });
        for type_name in type_names {
            let def = types_clone.get(type_name).unwrap();
            // Check if this is a record type with the same name as the constructor
            // Record types can be constructed using their type name as a constructor
            if type_name == name {
                if let TypeDef::Record { params, fields, .. } = def {
                    // Create fresh type variables for each type parameter
                    let substitution: HashMap<String, Type> = params
                        .iter()
                        .map(|p| (p.name.clone(), self.fresh()))
                        .collect();

                    // Get the type args for the result type
                    let type_args: Vec<Type> = params
                        .iter()
                        .map(|p| substitution.get(&p.name).cloned().unwrap_or_else(|| Type::TypeParam(p.name.clone())))
                        .collect();

                    let result_ty = Type::Named {
                        name: type_name.clone(),
                        args: type_args,
                    };

                    // Get field types in order for positional constructor
                    let field_types: Vec<Type> = fields
                        .iter()
                        .map(|(_, ty, _)| Self::substitute_type_params(ty, &substitution))
                        .collect();

                    if field_types.is_empty() {
                        return Some(result_ty);
                    } else {
                        return Some(Type::Function(FunctionType { required_params: None,
                            type_params: vec![],
                            params: field_types,
                            ret: Box::new(result_ty),
                        }));
                    }
                }
            }

            if let TypeDef::Variant { params, constructors } = def {
                for ctor in constructors {
                    let matches = match ctor {
                        Constructor::Unit(n) => n == name,
                        Constructor::Positional(n, _) => n == name,
                        Constructor::Named(n, _) => n == name,
                    };
                    if !matches {
                        continue;
                    }

                    // Create fresh type variables for each type parameter
                    let substitution: HashMap<String, Type> = params
                        .iter()
                        .map(|p| (p.name.clone(), self.fresh()))
                        .collect();

                    // Apply substitution to get instantiated types
                    let instantiate = |ty: &Type| -> Type {
                        Self::substitute_type_params(ty, &substitution)
                    };

                    // Get the type args for the result type
                    let type_args: Vec<Type> = params
                        .iter()
                        .map(|p| substitution.get(&p.name).cloned().unwrap_or_else(|| Type::TypeParam(p.name.clone())))
                        .collect();

                    let result_ty = Type::Named {
                        name: type_name.clone(),
                        args: type_args,
                    };

                    match ctor {
                        Constructor::Unit(_) => {
                            return Some(result_ty);
                        }
                        Constructor::Positional(_, ctor_params) => {
                            if ctor_params.is_empty() {
                                return Some(result_ty);
                            } else {
                                let instantiated_params: Vec<_> = ctor_params
                                    .iter()
                                    .map(instantiate)
                                    .collect();
                                return Some(Type::Function(FunctionType { required_params: None,
                                    type_params: vec![],
                                    params: instantiated_params,
                                    ret: Box::new(result_ty),
                                }));
                            }
                        }
                        Constructor::Named(_, fields) => {
                            let instantiated_params: Vec<_> = fields
                                .iter()
                                .map(|(_, ty)| instantiate(ty))
                                .collect();
                            if instantiated_params.is_empty() {
                                return Some(result_ty);
                            } else {
                                return Some(Type::Function(FunctionType { required_params: None,
                                    type_params: vec![],
                                    params: instantiated_params,
                                    ret: Box::new(result_ty),
                                }));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Substitute type parameters in a type with concrete types.
    fn substitute_type_params(ty: &Type, subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::TypeParam(name) => {
                subst.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|t| Self::substitute_type_params(t, subst)).collect())
            }
            Type::List(elem) => {
                Type::List(Box::new(Self::substitute_type_params(elem, subst)))
            }
            Type::Array(elem) => {
                Type::Array(Box::new(Self::substitute_type_params(elem, subst)))
            }
            Type::Set(elem) => {
                Type::Set(Box::new(Self::substitute_type_params(elem, subst)))
            }
            Type::Map(k, v) => {
                Type::Map(
                    Box::new(Self::substitute_type_params(k, subst)),
                    Box::new(Self::substitute_type_params(v, subst)),
                )
            }
            Type::Function(f) => {
                Type::Function(FunctionType { required_params: None,
                    type_params: f.type_params.clone(),
                    params: f.params.iter().map(|t| Self::substitute_type_params(t, subst)).collect(),
                    ret: Box::new(Self::substitute_type_params(&f.ret, subst)),
                })
            }
            Type::Named { name, args } => {
                // Handle type parameters that were parsed as Named types with no args
                // (e.g., "T" in "Tree[T]" gets parsed as Named { name: "T", args: [] })
                if args.is_empty() {
                    if let Some(replacement) = subst.get(name) {
                        return replacement.clone();
                    }
                }
                Type::Named {
                    name: name.clone(),
                    args: args.iter().map(|t| Self::substitute_type_params(t, subst)).collect(),
                }
            }
            Type::IO(inner) => {
                Type::IO(Box::new(Self::substitute_type_params(inner, subst)))
            }
            Type::Record(rec) => {
                Type::Record(RecordType {
                    name: rec.name.clone(),
                    fields: rec.fields.iter().map(|(n, t, m)| {
                        (n.clone(), Self::substitute_type_params(t, subst), *m)
                    }).collect(),
                })
            }
            _ => ty.clone(),
        }
    }

    /// Infer types for a match arm.
    fn infer_match_arm(
        &mut self,
        arm: &MatchArm,
        scrutinee_ty: &Type,
        result_ty: &Type,
    ) -> Result<(), TypeError> {
        let saved_bindings = self.env.bindings.clone();

        // Infer pattern against scrutinee
        self.infer_pattern(&arm.pattern, scrutinee_ty)?;

        // Check guard if present
        if let Some(guard) = &arm.guard {
            let guard_ty = self.infer_expr(guard)?;
            self.unify(guard_ty, Type::Bool);
        }

        // Infer body
        let body_ty = self.infer_expr(&arm.body)?;
        self.unify(body_ty, result_ty.clone());

        self.env.bindings = saved_bindings;
        Ok(())
    }

    /// Infer types for a binding.
    pub fn infer_binding(&mut self, binding: &Binding) -> Result<Type, TypeError> {
        let value_ty = self.infer_expr(&binding.value)?;

        // If there's a type annotation, unify with it
        if let Some(ty_expr) = &binding.ty {
            let annotated_ty = self.type_from_ast(ty_expr);
            self.unify(value_ty.clone(), annotated_ty);
        }

        // Bind pattern
        self.infer_pattern(&binding.pattern, &value_ty)?;

        Ok(value_ty)
    }

    /// Infer types for a function definition.
    pub fn infer_function(&mut self, func: &FnDef) -> Result<FunctionType, TypeError> {
        let name = &func.name.node;

        if func.clauses.is_empty() {
            return Err(TypeError::ArityMismatch {
                expected: 1,
                found: 0,
            });
        }

        // Set current function for recursive call detection
        let saved_current = self.current_function.take();
        self.current_function = Some(name.clone());

        // Get pre-registered type if available (for recursive calls to use the same vars)
        // Try plain name, wildcard qualified name, and any typed overloads
        let arity = func.clauses.first().map(|c| c.params.len()).unwrap_or(0);
        let arity_suffix = if arity == 0 {
            "/".to_string()
        } else {
            format!("/{}", vec!["_"; arity].join(","))
        };
        let qualified_name = format!("{}{}", name, arity_suffix);
        let pre_registered = self.env.functions.get(name).cloned()
            .or_else(|| self.env.functions.get(&qualified_name).cloned())
            .or_else(|| {
                // Also check typed overloads (e.g., "showCounter/Counter")
                let all_overloads = self.env.lookup_all_functions_with_arity(name, arity);
                all_overloads.first().cloned().cloned()
            });

        // Infer the first clause
        let (clause_params, clause_ret) = self.infer_clause(&func.clauses[0])?;

        // If we have a pre-registered type, unify with its vars to connect recursive calls
        let (mut param_types, mut ret_ty) = if let Some(ref pre_reg) = pre_registered {
            // Unify clause params with pre-registered params
            if clause_params.len() == pre_reg.params.len() {
                for (cp, pp) in clause_params.iter().zip(pre_reg.params.iter()) {
                    self.unify(cp.clone(), pp.clone());
                }
            }
            // Unify clause return with pre-registered return
            self.unify(clause_ret.clone(), (*pre_reg.ret).clone());
            // Use pre-registered types (they're now unified)
            (pre_reg.params.clone(), (*pre_reg.ret).clone())
        } else {
            (clause_params, clause_ret)
        };

        // Infer remaining clauses and unify their types with the first
        for clause in func.clauses.iter().skip(1) {
            let (clause_params, clause_ret) = self.infer_clause(clause)?;

            // Unify parameter types
            if clause_params.len() != param_types.len() {
                return Err(TypeError::ArityMismatch {
                    expected: param_types.len(),
                    found: clause_params.len(),
                });
            }
            for i in 0..param_types.len() {
                self.unify(param_types[i].clone(), clause_params[i].clone());
            }
            // Update param_types with unified types
            for i in 0..param_types.len() {
                param_types[i] = self.env.apply_subst(&param_types[i]);
            }

            // Unify return types
            self.unify(ret_ty.clone(), clause_ret);
            ret_ty = self.env.apply_subst(&ret_ty);
        }

        // Count required parameters (those without defaults)
        let required_count = func.clauses[0].params.iter()
            .filter(|p| p.default.is_none())
            .count();
        let has_defaults = required_count < func.clauses[0].params.len();

        // Note: Type variables in param_types/ret_ty will be resolved after solve()
        // is called in check_module - see the post-processing step there

        let func_ty = FunctionType {
            type_params: vec![],
            params: param_types,
            ret: Box::new(ret_ty),
            required_params: if has_defaults { Some(required_count) } else { None },
        };

        // Register function in environment
        self.env.insert_function(name.clone(), func_ty.clone());

        // Restore previous current function
        self.current_function = saved_current;

        Ok(func_ty)
    }

    /// Infer types for a function clause.
    fn infer_clause(&mut self, clause: &FnClause) -> Result<(Vec<Type>, Type), TypeError> {
        let saved_bindings = self.env.bindings.clone();

        let mut param_types = Vec::new();
        for param in &clause.params {
            let param_ty = if let Some(ty_expr) = &param.ty {
                self.type_from_ast(ty_expr)
            } else {
                self.fresh()
            };
            self.infer_pattern(&param.pattern, &param_ty)?;
            param_types.push(param_ty);
        }

        // Check guard if present
        if let Some(guard) = &clause.guard {
            let guard_ty = self.infer_expr(guard)?;
            self.unify(guard_ty, Type::Bool);
        }

        // Infer body
        let body_ty = self.infer_expr(&clause.body)?;

        // If there's a return type annotation, unify with it
        let ret_ty = if let Some(ret_expr) = &clause.return_type {
            let annotated = self.type_from_ast(ret_expr);
            self.unify(body_ty, annotated.clone());
            annotated
        } else {
            body_ty
        };

        self.env.bindings = saved_bindings;
        Ok((param_types, ret_ty))
    }

    /// Infer types for a module.
    pub fn infer_module(&mut self, module: &Module) -> Result<(), TypeError> {
        // First pass: collect type and function definitions
        for item in &module.items {
            match item {
                Item::TypeDef(td) => {
                    self.register_type_def(td)?;
                }
                Item::FnDef(fd) => {
                    // Register a placeholder for recursive calls
                    let ret_ty = self.fresh();
                    let param_types: Vec<_> = fd.clauses.first().map(|c| {
                        c.params.iter().map(|_| self.fresh()).collect()
                    }).unwrap_or_default();

                    // Count required parameters (those without defaults)
                    let required_count = fd.clauses.first().map(|c| {
                        c.params.iter().filter(|p| p.default.is_none()).count()
                    }).unwrap_or(0);
                    let total_params = param_types.len();
                    let has_defaults = required_count < total_params;

                    self.env.insert_function(
                        fd.name.node.clone(),
                        FunctionType {
                            type_params: vec![],
                            params: param_types,
                            ret: Box::new(ret_ty),
                            required_params: if has_defaults { Some(required_count) } else { None },
                        },
                    );
                }
                _ => {}
            }
        }

        // Second pass: infer function bodies
        for item in &module.items {
            match item {
                Item::FnDef(fd) => {
                    self.infer_function(fd)?;
                }
                Item::Binding(binding) => {
                    self.infer_binding(binding)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Register a type definition in the environment.
    fn register_type_def(
        &mut self,
        td: &nostos_syntax::ast::TypeDef,
    ) -> Result<(), TypeError> {
        let name = td.name.node.clone();
        let params: Vec<_> = td
            .type_params
            .iter()
            .map(|p| crate::TypeParam {
                name: p.name.node.clone(),
                constraints: p.constraints.iter().map(|c| c.node.clone()).collect(),
            })
            .collect();

        let def = match &td.body {
            nostos_syntax::ast::TypeBody::Record(fields) => TypeDef::Record {
                params,
                fields: fields
                    .iter()
                    .map(|f| (f.name.node.clone(), self.type_from_ast(&f.ty), !f.private))
                    .collect(),
                is_mutable: td.mutable,
            },
            nostos_syntax::ast::TypeBody::Variant(variants) => TypeDef::Variant {
                params,
                constructors: variants
                    .iter()
                    .map(|v| {
                        let vname = v.name.node.clone();
                        match &v.fields {
                            nostos_syntax::ast::VariantFields::Unit => Constructor::Unit(vname),
                            nostos_syntax::ast::VariantFields::Positional(types) => {
                                Constructor::Positional(
                                    vname,
                                    types.iter().map(|t| self.type_from_ast(t)).collect(),
                                )
                            }
                            nostos_syntax::ast::VariantFields::Named(fields) => Constructor::Named(
                                vname,
                                fields
                                    .iter()
                                    .map(|f| (f.name.node.clone(), self.type_from_ast(&f.ty)))
                                    .collect(),
                            ),
                        }
                    })
                    .collect(),
            },
            nostos_syntax::ast::TypeBody::Alias(ty) => TypeDef::Alias {
                params,
                target: self.type_from_ast(ty),
            },
            nostos_syntax::ast::TypeBody::Empty => {
                return Ok(()); // Never type
            }
        };

        self.env.define_type(name, def);
        Ok(())
    }
}

/// Convenience function to infer and solve types for an expression.
pub fn infer_expr_type(env: &mut TypeEnv, expr: &Expr) -> Result<Type, TypeError> {
    let mut ctx = InferCtx::new(env);
    let ty = ctx.infer_expr(expr)?;
    ctx.solve()?;
    Ok(ctx.env.apply_subst(&ty))
}

/// Convenience function to type-check a module.
pub fn check_module(env: &mut TypeEnv, module: &Module) -> Result<(), TypeError> {
    let mut ctx = InferCtx::new(env);
    ctx.infer_module(module)?;
    ctx.solve()?;

    // After solving constraints, apply substitution to all registered functions
    // to replace type variables with their resolved concrete types
    let function_names: Vec<String> = ctx.env.functions.keys().cloned().collect();
    for name in function_names {
        if let Some(ft) = ctx.env.functions.get(&name).cloned() {
            let resolved_params: Vec<Type> = ft.params.iter()
                .map(|p| ctx.env.apply_subst(p))
                .collect();
            let resolved_ret = ctx.env.apply_subst(&ft.ret);
            ctx.env.insert_function(name, FunctionType {
                type_params: ft.type_params,
                params: resolved_params,
                ret: Box::new(resolved_ret),
                required_params: ft.required_params,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nostos_syntax::ast::Span;
    use nostos_syntax::CallArg;

    // Helper to create a spanned identifier
    fn ident(name: &str) -> nostos_syntax::ast::Ident {
        nostos_syntax::ast::Spanned::new(name.to_string(), Span::default())
    }

    // Helper to create expression with default span
    fn span() -> Span {
        Span::default()
    }

    // =========================================================================
    // Unification Tests
    // =========================================================================

    #[test]
    fn test_unify_same() {
        let mut env = TypeEnv::new();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(Type::Int, Type::Int);
        assert!(ctx.solve().is_ok());
    }

    #[test]
    fn test_unify_var() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(var.clone(), Type::Int);
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var), Type::Int);
    }

    #[test]
    fn test_unify_mismatch() {
        let mut env = TypeEnv::new();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(Type::Int, Type::String);
        assert!(matches!(ctx.solve(), Err(TypeError::UnificationFailed(_, _))));
    }

    #[test]
    fn test_unify_lists() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(
            Type::List(Box::new(var.clone())),
            Type::List(Box::new(Type::Int)),
        );
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var), Type::Int);
    }

    #[test]
    fn test_occurs_check() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        // Try to unify ?0 with List[?0] - should fail
        ctx.unify(var.clone(), Type::List(Box::new(var.clone())));
        assert!(matches!(ctx.solve(), Err(TypeError::OccursCheck(_, _))));
    }

    #[test]
    fn test_unify_tuples() {
        let mut env = TypeEnv::new();
        let var1 = env.fresh_var();
        let var2 = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);
        ctx.unify(
            Type::Tuple(vec![var1.clone(), var2.clone()]),
            Type::Tuple(vec![Type::Int, Type::String]),
        );
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var1), Type::Int);
        assert_eq!(ctx.env.apply_subst(&var2), Type::String);
    }

    #[test]
    fn test_unify_functions() {
        let mut env = TypeEnv::new();
        let var = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);

        let func1 = Type::Function(FunctionType { required_params: None,
            type_params: vec![],
            params: vec![Type::Int],
            ret: Box::new(var.clone()),
        });
        let func2 = Type::Function(FunctionType { required_params: None,
            type_params: vec![],
            params: vec![Type::Int],
            ret: Box::new(Type::Bool),
        });

        ctx.unify(func1, func2);
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var), Type::Bool);
    }

    // =========================================================================
    // Literal Expression Tests
    // =========================================================================

    #[test]
    fn test_infer_int_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Int(42, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_float_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Float(3.14, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Float);
    }

    #[test]
    fn test_infer_bool_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Bool(true, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_string_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_unit_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Unit(span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Unit);
    }

    // =========================================================================
    // Variable Tests
    // =========================================================================

    #[test]
    fn test_infer_variable() {
        let mut env = TypeEnv::new();
        env.bind("x".to_string(), Type::Int, false);
        let expr = Expr::Var(ident("x"));
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_unknown_variable() {
        let mut env = TypeEnv::new();
        let expr = Expr::Var(ident("unknown"));
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnknownIdent(_))));
    }

    // =========================================================================
    // Binary Operation Tests
    // =========================================================================

    #[test]
    fn test_infer_addition() {
        let mut env = crate::standard_env();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Add,
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_comparison() {
        let mut env = crate::standard_env();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Lt,
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_equality() {
        let mut env = crate::standard_env();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Eq,
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_logical() {
        let mut env = TypeEnv::new();
        let expr = Expr::BinOp(
            Box::new(Expr::Bool(true, span())),
            BinOp::And,
            Box::new(Expr::Bool(false, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_type_mismatch_binop() {
        let mut env = crate::standard_env();
        let expr = Expr::BinOp(
            Box::new(Expr::Int(1, span())),
            BinOp::Add,
            Box::new(Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        // Should fail because Int and String cannot be unified
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Unary Operation Tests
    // =========================================================================

    #[test]
    fn test_infer_negation() {
        let mut env = TypeEnv::new();
        let expr = Expr::UnaryOp(UnaryOp::Neg, Box::new(Expr::Int(42, span())), span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_not() {
        let mut env = TypeEnv::new();
        let expr = Expr::UnaryOp(UnaryOp::Not, Box::new(Expr::Bool(true, span())), span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    // =========================================================================
    // Tuple Tests
    // =========================================================================

    #[test]
    fn test_infer_tuple() {
        let mut env = TypeEnv::new();
        let expr = Expr::Tuple(
            vec![
                Expr::Int(1, span()),
                Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span()),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Tuple(vec![Type::Int, Type::String]));
    }

    // =========================================================================
    // List Tests
    // =========================================================================

    #[test]
    fn test_infer_empty_list() {
        let mut env = TypeEnv::new();
        let expr = Expr::List(vec![], None, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        // Empty list has a fresh variable element type
        assert!(matches!(ty, Type::List(_)));
    }

    #[test]
    fn test_infer_int_list() {
        let mut env = TypeEnv::new();
        let expr = Expr::List(
            vec![Expr::Int(1, span()), Expr::Int(2, span()), Expr::Int(3, span())],
            None,
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::List(Box::new(Type::Int)));
    }

    #[test]
    fn test_infer_heterogeneous_list_fails() {
        let mut env = TypeEnv::new();
        let expr = Expr::List(
            vec![
                Expr::Int(1, span()),
                Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span()),
            ],
            None,
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Lambda Tests
    // =========================================================================

    #[test]
    fn test_infer_lambda_identity() {
        let mut env = TypeEnv::new();
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("x"))],
            Box::new(Expr::Var(ident("x"))),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 1);
            // Return type should equal param type
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_lambda_add() {
        let mut env = TypeEnv::new();
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("x"))],
            Box::new(Expr::BinOp(
                Box::new(Expr::Var(ident("x"))),
                BinOp::Add,
                Box::new(Expr::Int(1, span())),
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params, vec![Type::Int]);
            assert_eq!(*f.ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    // =========================================================================
    // Function Call Tests
    // =========================================================================

    #[test]
    fn test_infer_function_call() {
        let mut env = TypeEnv::new();
        // Register a function: f : Int -> Bool
        env.insert_function(
            "f".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Bool),
            },
        );
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("f"))),
            vec![],
            vec![CallArg::Positional(Expr::Int(42, span()))],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Bool);
    }

    #[test]
    fn test_infer_function_call_wrong_arg_type() {
        let mut env = TypeEnv::new();
        // Register a function: f : Int -> Bool
        env.insert_function(
            "f".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Bool),
            },
        );
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("f"))),
            vec![],
            vec![CallArg::Positional(Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span()))],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // If Expression Tests
    // =========================================================================

    #[test]
    fn test_infer_if() {
        let mut env = TypeEnv::new();
        let expr = Expr::If(
            Box::new(Expr::Bool(true, span())),
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_if_branches_must_match() {
        let mut env = TypeEnv::new();
        let expr = Expr::If(
            Box::new(Expr::Bool(true, span())),
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    #[test]
    fn test_infer_if_condition_must_be_bool() {
        let mut env = TypeEnv::new();
        let expr = Expr::If(
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::Int(1, span())),
            Box::new(Expr::Int(2, span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Match Expression Tests
    // =========================================================================

    #[test]
    fn test_infer_match() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::Int(1, span())),
            vec![
                MatchArm {
                    pattern: Pattern::Int(0, span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("zero".to_string()), span()),
                    span: span(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard(span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("other".to_string()), span()),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_match_with_binding() {
        let mut env = crate::standard_env();
        let expr = Expr::Match(
            Box::new(Expr::Int(42, span())),
            vec![MatchArm {
                pattern: Pattern::Var(ident("n")),
                guard: None,
                body: Expr::BinOp(
                    Box::new(Expr::Var(ident("n"))),
                    BinOp::Add,
                    Box::new(Expr::Int(1, span())),
                    span(),
                ),
                span: span(),
            }],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Block Tests
    // =========================================================================

    #[test]
    fn test_infer_block() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![
                Stmt::Expr(Expr::Int(1, span())),
                Stmt::Expr(Expr::String(nostos_syntax::ast::StringLit::Plain("result".to_string()), span())),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_block_with_let() {
        let mut env = crate::standard_env();
        let expr = Expr::Block(
            vec![
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: None,
                    value: Expr::Int(42, span()),
                    span: span(),
                }),
                Stmt::Expr(Expr::BinOp(
                    Box::new(Expr::Var(ident("x"))),
                    BinOp::Add,
                    Box::new(Expr::Int(1, span())),
                    span(),
                )),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Pattern Tests
    // =========================================================================

    #[test]
    fn test_infer_tuple_pattern() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::Tuple(
                vec![Expr::Int(1, span()), Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())],
                span(),
            )),
            vec![MatchArm {
                pattern: Pattern::Tuple(
                    vec![Pattern::Var(ident("a")), Pattern::Var(ident("b"))],
                    span(),
                ),
                guard: None,
                body: Expr::Var(ident("a")),
                span: span(),
            }],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_list_pattern() {
        let mut env = TypeEnv::new();
        let expr = Expr::Match(
            Box::new(Expr::List(
                vec![Expr::Int(1, span()), Expr::Int(2, span())],
                None,
                span(),
            )),
            vec![
                MatchArm {
                    pattern: Pattern::List(
                        nostos_syntax::ast::ListPattern::Cons(
                            vec![Pattern::Var(ident("h"))],
                            Some(Box::new(Pattern::Var(ident("t")))),
                        ),
                        span(),
                    ),
                    guard: None,
                    body: Expr::Var(ident("h")),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Constructor Tests
    // =========================================================================

    #[test]
    fn test_infer_option_none() {
        let mut env = crate::standard_env();
        let expr = Expr::Var(ident("None"));
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        // None has type Option[?T] for some unknown T
        if let Type::Named { name, args } = ty {
            assert_eq!(name, "Option");
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected Option type, got {:?}", ty);
        }
    }

    #[test]
    fn test_infer_option_some() {
        let mut env = crate::standard_env();
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("Some"))),
            vec![],
            vec![CallArg::Positional(Expr::Int(42, span()))],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        // Some(42) should have type Option (with Int inner type)
        assert!(matches!(ty, Type::Named { name, .. } if name == "Option"));
    }

    // =========================================================================
    // Pipe Operator Tests
    // =========================================================================

    #[test]
    fn test_infer_pipe() {
        let mut env = TypeEnv::new();
        // Register a function: double : Int -> Int
        env.insert_function(
            "double".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        let expr = Expr::BinOp(
            Box::new(Expr::Int(21, span())),
            BinOp::Pipe,
            Box::new(Expr::Var(ident("double"))),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    // =========================================================================
    // Type Annotation Tests
    // =========================================================================

    #[test]
    fn test_infer_with_annotation() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: Some(TypeExpr::Name(ident("Int"))),
                    value: Expr::Int(42, span()),
                    span: span(),
                }),
                Stmt::Expr(Expr::Var(ident("x"))),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_annotation_mismatch() {
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![Stmt::Let(Binding {
                mutable: false,
                pattern: Pattern::Var(ident("x")),
                ty: Some(TypeExpr::Name(ident("String"))),
                value: Expr::Int(42, span()),
                span: span(),
            })],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))));
    }

    // =========================================================================
    // Map and Set Tests
    // =========================================================================

    #[test]
    fn test_infer_map() {
        let mut env = crate::standard_env();
        let expr = Expr::Map(
            vec![
                (
                    Expr::String(nostos_syntax::ast::StringLit::Plain("a".to_string()), span()),
                    Expr::Int(1, span()),
                ),
                (
                    Expr::String(nostos_syntax::ast::StringLit::Plain("b".to_string()), span()),
                    Expr::Int(2, span()),
                ),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(
            ty,
            Type::Map(Box::new(Type::String), Box::new(Type::Int))
        );
    }

    #[test]
    fn test_infer_set() {
        let mut env = crate::standard_env();
        let expr = Expr::Set(
            vec![Expr::Int(1, span()), Expr::Int(2, span()), Expr::Int(3, span())],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Set(Box::new(Type::Int)));
    }

    // =========================================================================
    // Advanced Type Inference Tests
    // =========================================================================

    #[test]
    fn test_infer_higher_order_function() {
        // map : (a -> b) -> List[a] -> List[b]
        // map(f, xs) should infer correct types
        let mut env = TypeEnv::new();
        // Create a lambda: f => xs => ... (simplified)
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("f"))],
            Box::new(Expr::Lambda(
                vec![Pattern::Var(ident("xs"))],
                Box::new(Expr::Var(ident("xs"))), // Just return xs for now
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 1);
            if let Type::Function(inner) = *f.ret {
                assert_eq!(inner.params.len(), 1);
            } else {
                panic!("Expected nested function type");
            }
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_curried_function() {
        // Curried addition with concrete constraint: x => y => x + y + 0
        // The 0 forces the type to be Int
        let mut env = crate::standard_env();
        let expr = Expr::Lambda(
            vec![Pattern::Var(ident("x"))],
            Box::new(Expr::Lambda(
                vec![Pattern::Var(ident("y"))],
                Box::new(Expr::BinOp(
                    Box::new(Expr::BinOp(
                        Box::new(Expr::Var(ident("x"))),
                        BinOp::Add,
                        Box::new(Expr::Var(ident("y"))),
                        span(),
                    )),
                    BinOp::Add,
                    Box::new(Expr::Int(0, span())), // Constrains to Int
                    span(),
                )),
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 1);
            assert_eq!(f.params[0], Type::Int); // Now constrained to Int
            if let Type::Function(inner) = *f.ret {
                assert_eq!(inner.params.len(), 1);
                assert_eq!(inner.params[0], Type::Int);
                assert_eq!(*inner.ret, Type::Int);
            } else {
                panic!("Expected nested function type");
            }
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_nested_list() {
        let mut env = TypeEnv::new();
        // [[1, 2], [3, 4]]
        let inner1 = Expr::List(
            vec![Expr::Int(1, span()), Expr::Int(2, span())],
            None,
            span(),
        );
        let inner2 = Expr::List(
            vec![Expr::Int(3, span()), Expr::Int(4, span())],
            None,
            span(),
        );
        let expr = Expr::List(vec![inner1, inner2], None, span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::List(Box::new(Type::List(Box::new(Type::Int)))));
    }

    #[test]
    fn test_infer_match_with_guard() {
        let mut env = crate::standard_env();
        // match x { n when n > 0 -> "positive", _ -> "non-positive" }
        env.bind("x".to_string(), Type::Int, false);
        let expr = Expr::Match(
            Box::new(Expr::Var(ident("x"))),
            vec![
                MatchArm {
                    pattern: Pattern::Var(ident("n")),
                    guard: Some(Expr::BinOp(
                        Box::new(Expr::Var(ident("n"))),
                        BinOp::Gt,
                        Box::new(Expr::Int(0, span())),
                        span(),
                    )),
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("positive".to_string()), span()),
                    span: span(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard(span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("non-positive".to_string()), span()),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_lambda_with_multiple_params() {
        let mut env = crate::standard_env();
        // (a, b, c) => a + b + c + 0  (0 constrains to Int)
        let expr = Expr::Lambda(
            vec![
                Pattern::Var(ident("a")),
                Pattern::Var(ident("b")),
                Pattern::Var(ident("c")),
            ],
            Box::new(Expr::BinOp(
                Box::new(Expr::BinOp(
                    Box::new(Expr::BinOp(
                        Box::new(Expr::Var(ident("a"))),
                        BinOp::Add,
                        Box::new(Expr::Var(ident("b"))),
                        span(),
                    )),
                    BinOp::Add,
                    Box::new(Expr::Var(ident("c"))),
                    span(),
                )),
                BinOp::Add,
                Box::new(Expr::Int(0, span())), // Constrains to Int
                span(),
            )),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params.len(), 3);
            assert_eq!(f.params, vec![Type::Int, Type::Int, Type::Int]);
            assert_eq!(*f.ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_nested_if() {
        let mut env = TypeEnv::new();
        // if true then (if false then 1 else 2) else 3
        let expr = Expr::If(
            Box::new(Expr::Bool(true, span())),
            Box::new(Expr::If(
                Box::new(Expr::Bool(false, span())),
                Box::new(Expr::Int(1, span())),
                Box::new(Expr::Int(2, span())),
                span(),
            )),
            Box::new(Expr::Int(3, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_function_returning_function() {
        let mut env = TypeEnv::new();
        // Register: make_adder : Int -> (Int -> Int)
        env.insert_function(
            "make_adder".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: vec![Type::Int],
                    ret: Box::new(Type::Int),
                })),
            },
        );
        // make_adder(5)
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("make_adder"))),
            vec![],
            vec![CallArg::Positional(Expr::Int(5, span()))],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Function(f) = ty {
            assert_eq!(f.params, vec![Type::Int]);
            assert_eq!(*f.ret, Type::Int);
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_infer_chained_pipe() {
        let mut env = TypeEnv::new();
        // Register functions
        env.insert_function(
            "add_one".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        env.insert_function(
            "to_string".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::String),
            },
        );
        // 5 |> add_one |> to_string
        let expr = Expr::BinOp(
            Box::new(Expr::BinOp(
                Box::new(Expr::Int(5, span())),
                BinOp::Pipe,
                Box::new(Expr::Var(ident("add_one"))),
                span(),
            )),
            BinOp::Pipe,
            Box::new(Expr::Var(ident("to_string"))),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_polymorphic_function() {
        // Test that we can call monomorphic functions
        let mut env = TypeEnv::new();
        // id_int : Int -> Int (monomorphic version)
        env.insert_function(
            "id_int".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        // id_int(42) should be Int
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("id_int"))),
            vec![],
            vec![CallArg::Positional(Expr::Int(42, span()))],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_complex_block() {
        let mut env = crate::standard_env();
        // {
        //   let x = 1
        //   let y = 2
        //   let z = x + y
        //   z * 2
        // }
        let expr = Expr::Block(
            vec![
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: None,
                    value: Expr::Int(1, span()),
                    span: span(),
                }),
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("y")),
                    ty: None,
                    value: Expr::Int(2, span()),
                    span: span(),
                }),
                Stmt::Let(Binding {
                    mutable: false,
                    pattern: Pattern::Var(ident("z")),
                    ty: None,
                    value: Expr::BinOp(
                        Box::new(Expr::Var(ident("x"))),
                        BinOp::Add,
                        Box::new(Expr::Var(ident("y"))),
                        span(),
                    ),
                    span: span(),
                }),
                Stmt::Expr(Expr::BinOp(
                    Box::new(Expr::Var(ident("z"))),
                    BinOp::Mul,
                    Box::new(Expr::Int(2, span())),
                    span(),
                )),
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Int);
    }

    #[test]
    fn test_infer_or_pattern() {
        let mut env = TypeEnv::new();
        // match x { 0 | 1 | 2 -> "small", _ -> "big" }
        env.bind("x".to_string(), Type::Int, false);
        let expr = Expr::Match(
            Box::new(Expr::Var(ident("x"))),
            vec![
                MatchArm {
                    pattern: Pattern::Or(
                        vec![
                            Pattern::Int(0, span()),
                            Pattern::Int(1, span()),
                            Pattern::Int(2, span()),
                        ],
                        span(),
                    ),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("small".to_string()), span()),
                    span: span(),
                },
                MatchArm {
                    pattern: Pattern::Wildcard(span()),
                    guard: None,
                    body: Expr::String(nostos_syntax::ast::StringLit::Plain("big".to_string()), span()),
                    span: span(),
                },
            ],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::String);
    }

    #[test]
    fn test_infer_result_ok() {
        let mut env = crate::standard_env();
        // Ok(42)
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("Ok"))),
            vec![],
            vec![CallArg::Positional(Expr::Int(42, span()))],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Named { name, args } = ty {
            assert_eq!(name, "Result");
            assert_eq!(args.len(), 2); // Result[T, E]
        } else {
            panic!("Expected Result type");
        }
    }

    #[test]
    fn test_infer_result_err() {
        let mut env = crate::standard_env();
        // Err("error")
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("Err"))),
            vec![],
            vec![CallArg::Positional(Expr::String(nostos_syntax::ast::StringLit::Plain("error".to_string()), span()))],
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        if let Type::Named { name, args } = ty {
            assert_eq!(name, "Result");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected Result type");
        }
    }

    #[test]
    fn test_infer_float_arithmetic() {
        let mut env = crate::standard_env();
        // 3.14 * 2.0 + 1.0
        let expr = Expr::BinOp(
            Box::new(Expr::BinOp(
                Box::new(Expr::Float(3.14, span())),
                BinOp::Mul,
                Box::new(Expr::Float(2.0, span())),
                span(),
            )),
            BinOp::Add,
            Box::new(Expr::Float(1.0, span())),
            span(),
        );
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Float);
    }

    #[test]
    fn test_infer_mixed_int_float_coercion() {
        let mut env = crate::standard_env();
        // 3.14 + 1 (with int-to-float coercion, this should succeed and return Float)
        let expr = Expr::BinOp(
            Box::new(Expr::Float(3.14, span())),
            BinOp::Add,
            Box::new(Expr::Int(1, span())),
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(result.is_ok(), "Mixed Int/Float should coerce to Float");
        assert_eq!(result.unwrap(), Type::Float);
    }

    #[test]
    fn test_infer_char_literal() {
        let mut env = TypeEnv::new();
        let expr = Expr::Char('x', span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        assert_eq!(ty, Type::Char);
    }

    #[test]
    fn test_infer_deeply_nested_tuple() {
        let mut env = TypeEnv::new();
        // ((1, 2), (3, 4))
        let inner1 = Expr::Tuple(
            vec![Expr::Int(1, span()), Expr::Int(2, span())],
            span(),
        );
        let inner2 = Expr::Tuple(
            vec![Expr::Int(3, span()), Expr::Int(4, span())],
            span(),
        );
        let expr = Expr::Tuple(vec![inner1, inner2], span());
        let ty = infer_expr_type(&mut env, &expr).unwrap();
        let expected = Type::Tuple(vec![
            Type::Tuple(vec![Type::Int, Type::Int]),
            Type::Tuple(vec![Type::Int, Type::Int]),
        ]);
        assert_eq!(ty, expected);
    }

    #[test]
    fn test_infer_unify_nested_types() {
        let mut env = TypeEnv::new();
        let var1 = env.fresh_var();
        let var2 = env.fresh_var();
        let mut ctx = InferCtx::new(&mut env);

        // Unify List[Tuple[?0, ?1]] with List[Tuple[Int, String]]
        ctx.unify(
            Type::List(Box::new(Type::Tuple(vec![var1.clone(), var2.clone()]))),
            Type::List(Box::new(Type::Tuple(vec![Type::Int, Type::String]))),
        );
        assert!(ctx.solve().is_ok());
        assert_eq!(ctx.env.apply_subst(&var1), Type::Int);
        assert_eq!(ctx.env.apply_subst(&var2), Type::String);
    }

    #[test]
    fn test_infer_function_arity_mismatch() {
        let mut env = TypeEnv::new();
        env.insert_function(
            "add".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int, Type::Int],
                ret: Box::new(Type::Int),
            },
        );
        // add(1) - missing one argument
        let expr = Expr::Call(
            Box::new(Expr::Var(ident("add"))),
            vec![],
            vec![CallArg::Positional(Expr::Int(1, span()))],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        assert!(matches!(result, Err(TypeError::ArityMismatch { .. })));
    }
}
