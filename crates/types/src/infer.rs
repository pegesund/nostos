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
use std::collections::{HashMap, HashSet};

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
    /// Named argument info: (name, index in arg_types) for each named arg.
    /// Used to correctly match named args to function parameters during UFCS resolution.
    pub named_args: Vec<(String, usize)>,
    pub ret_ty: Type,
    pub span: Option<Span>,
}

/// Error type for incomplete type resolution.
#[derive(Debug, Clone)]
pub enum UnresolvedTypeError {
    /// Type contains unresolved type variables
    UnresolvedVar(Type),
    /// Type contains leaked type parameters (TypeParam not resolved to concrete type)
    LeakedTypeParam(Type),
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
    /// Tracks type variables that came from unannotated function parameters.
    /// Maps var ID to (function name, parameter name) for better error messages.
    unannotated_param_vars: HashMap<u32, (String, String)>,
    /// Tracks expressions where types couldn't be fully resolved (contain TypeParams after finalization).
    /// This is informational - code generation should handle these via fallback paths.
    pub unresolved_type_params: Vec<(Span, Type, String)>, // (span, type, param_name)
    /// Type parameters currently in scope (from function type parameters like [T: Hash]).
    /// Used by type_from_ast to distinguish type parameters from named types.
    current_type_params: HashSet<String>,
    /// Trait constraints for type parameters in scope (e.g., "T" -> ["Sizeable", "Show"]).
    /// Used by unify_types to propagate trait bounds when creating fresh vars for TypeParams.
    current_type_param_constraints: HashMap<String, Vec<String>>,
    /// Whether solve() completed normally (all constraints processed).
    /// False if solve() hit MAX_ITERATIONS limit.
    solve_completed: bool,
    /// Deferred HasTrait constraints: the ORIGINAL type from the constraint
    /// (before substitution) and the trait name. Used in post-solve to retry
    /// constraints that were deferred because the type was still a Var.
    /// Unlike trait_bounds (which includes merged bounds from variable unification),
    /// this tracks only direct HasTrait constraints, avoiding false positives.
    deferred_has_trait: Vec<(Type, String)>,
    /// Deferred HasField constraints: type, field name, expected field type.
    /// These are saved when the type is still unresolved (Var) so they can be
    /// re-checked after check_pending_method_calls resolves more types.
    deferred_has_field: Vec<(Type, String, Type)>,
    /// Deferred length/len checks: the argument type and call span.
    /// length() only works on collections (List, String, Map, Set, arrays).
    deferred_length_checks: Vec<(Type, Span)>,
    /// Deferred concat (++) checks: (left_type, right_type, span).
    /// ++ only works on String ++ String or List[T] ++ List[T].
    deferred_concat_checks: Vec<(Type, Type, Span)>,
    /// Deferred Set/Map return type checks: function call return types that may
    /// resolve to Set[X] or Map[K, V], requiring Hash+Eq on element/key type.
    deferred_collection_ret_checks: Vec<(Type, Span)>,
    /// Deferred function call argument checks: (param_type, arg_type, span).
    /// After solve(), checks that resolved types are structurally compatible
    /// (e.g., List vs Map is always an error regardless of type variables).
    deferred_fn_call_checks: Vec<(Type, Type, Span)>,
    /// Deferred trait bound checks from function calls: (arg_type, trait_name, span).
    /// When a generic function with trait bounds is called (e.g., equal[T: Eq]),
    /// and an argument is unified with a constrained type param, record the arg type
    /// and required trait. After solve(), verify that the resolved arg type satisfies
    /// the trait - particularly catches function types passed to constrained generics
    /// (functions never implement Eq/Ord/Num).
    deferred_generic_trait_checks: Vec<(Type, String, Span)>,
    /// Deferred default parameter checks: (default_value_type, param_type, span).
    /// Instead of immediately unifying default values with param types (which
    /// over-constrains generic params in batch inference), defer the check
    /// until after solve() resolves types through body usage.
    deferred_default_param_checks: Vec<(Type, Type, Span)>,
    /// Deferred method-on-var checks: (receiver_type, method_name, arg_types, ret_type, span).
    /// When a method call is resolved on a type variable (e.g., x.length() where x is Var),
    /// record it so the constraint can be propagated through function signatures.
    /// At the call site, the method will be re-validated on the concrete type.
    deferred_method_on_var: Vec<(Type, String, Vec<Type>, Type, Option<Span>)>,
    /// Deferred method existence checks from HasMethod constraints in instantiate_function.
    /// After solve() resolves the type var, we check if the resolved type supports the method.
    /// Format: (type_var, method_name, span).
    deferred_method_existence_checks: Vec<(Type, String, Option<Span>)>,
    /// Deferred indirect call checks: (func_type_var, arg_types, span).
    /// When a Call expression's func_ty is a Var (e.g., curried calls like `adder(1)("hello")`),
    /// we can't extract param types at inference time. After solve(), resolve func_ty to a
    /// Function type and check each param vs arg for concrete type mismatches.
    deferred_indirect_call_checks: Vec<(Type, Vec<Type>, Span)>,
    /// Deferred typed binding checks: (value_type, annotation_type, span).
    /// When a binding has a type annotation (e.g., `b: Box[Int] = expr`), record
    /// both types. After solve(), verify the resolved types match.
    deferred_typed_binding_checks: Vec<(Type, Type, Span)>,
    /// Deferred branch type checks: (branch_type, result_type, span).
    /// When if-else or match branches produce values, record each branch type
    /// paired with the shared result type variable. After solve(), check for
    /// structural container mismatches (e.g., List vs Set in different branches).
    deferred_branch_type_checks: Vec<(Type, Type, Span)>,
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
            unannotated_param_vars: HashMap::new(),
            unresolved_type_params: Vec::new(),
            current_type_params: HashSet::new(),
            current_type_param_constraints: HashMap::new(),
            solve_completed: false,
            deferred_has_trait: Vec::new(),
            deferred_has_field: Vec::new(),
            deferred_length_checks: Vec::new(),
            deferred_concat_checks: Vec::new(),
            deferred_collection_ret_checks: Vec::new(),
            deferred_fn_call_checks: Vec::new(),
            deferred_generic_trait_checks: Vec::new(),
            deferred_default_param_checks: Vec::new(),
            deferred_method_on_var: Vec::new(),
            deferred_method_existence_checks: Vec::new(),
            deferred_indirect_call_checks: Vec::new(),
            deferred_typed_binding_checks: Vec::new(),
            deferred_branch_type_checks: Vec::new(),
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

    /// Get unresolved HasField constraints (field access on still-generic type vars).
    /// Used to propagate field requirements from generic function bodies to call sites.
    pub fn get_deferred_has_field(&self) -> &[(Type, String, Type)] {
        &self.deferred_has_field
    }

    /// Get deferred method-on-var calls (method calls on still-generic type vars).
    /// Used to propagate method requirements from generic function bodies to call sites.
    pub fn get_deferred_method_on_var(&self) -> &[(Type, String, Vec<Type>, Type, Option<Span>)] {
        &self.deferred_method_on_var
    }

    /// Look up the Var type that a TypeParam name maps to.
    pub fn get_type_param_mapping(&self, name: &str) -> Option<Type> {
        self.type_param_mappings.get(name).cloned()
    }

    /// Generate a fresh type variable.
    pub fn fresh(&mut self) -> Type {
        self.env.fresh_var()
    }

    /// Extract a parameter name from a pattern for error messages.
    fn extract_pattern_name(pattern: &Pattern) -> String {
        match pattern {
            Pattern::Var(ident) => ident.node.clone(),
            Pattern::Wildcard(_) => "_".to_string(),
            Pattern::Tuple(pats, _) => {
                let names: Vec<_> = pats.iter().map(Self::extract_pattern_name).collect();
                format!("({})", names.join(", "))
            }
            Pattern::Record(fields, _) => {
                let names: Vec<_> = fields.iter().map(|f| match f {
                    RecordPatternField::Punned(ident) => ident.node.clone(),
                    RecordPatternField::Named(name, _) => name.node.clone(),
                    RecordPatternField::Rest(_) => "_".to_string(),
                }).collect();
                format!("{{{}}}", names.join(", "))
            }
            _ => "<pattern>".to_string(),
        }
    }

    /// Check if a unification failure is because function parameters that share a type variable
    /// are being used with incompatible concrete types. This suggests type annotations are needed.
    fn check_annotation_required(&self, t1: &Type, t2: &Type, _original_error: &TypeError) -> Option<TypeError> {
        // When unifying function types, check if the same type variable appears in multiple
        // parameter positions with conflicting concrete types.
        //
        // Pattern: t1 = (a -> b, a, a) -> (b, b)  [function has shared type var 'a' for params]
        //          t2 = (f, Int, String) -> r     [call site has different concrete types]
        //
        // The shared type var 'a' can't be both Int and String.

        if let (Type::Function(f1), Type::Function(f2)) = (t1, t2) {
            // Skip for functions with multiple declared type params (e.g., pair[A, B]).
            // HM inference may have merged distinct type params (A=B) through if/else
            // branches that swap them (e.g., `if swap then (b,a) else (a,b)`).
            // The "shared var" is actually two different declared type params that HM
            // collapsed, so the conflict is a false positive.
            if f1.type_params.len() >= 2 {
                return None;
            }

            // Check if f1 (the declared function type) has shared type variables among params
            // that f2 (the call site) tries to instantiate with different concrete types

            // Find type variables in f1 params that appear multiple times
            // Look at the RAW types (before substitution) to find shared vars
            let mut var_positions: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
            for (i, param) in f1.params.iter().enumerate() {
                // Extract var IDs from the raw param type (not resolved)
                let mut var_ids = Vec::new();
                self.collect_var_ids(param, &mut var_ids);
                for var_id in var_ids {
                    var_positions.entry(var_id).or_default().push(i);
                }
            }

            // For vars that appear in multiple positions, check if f2 has conflicting types
            for (_, positions) in var_positions {
                if positions.len() < 2 {
                    continue;
                }

                // Get the concrete types at these positions in f2
                // Only consider positions where the raw param IS the var (not contains the var)
                let mut concrete_types: Vec<(usize, String)> = Vec::new();
                for &pos in &positions {
                    if let (Some(f1_param), Some(f2_param)) = (f1.params.get(pos), f2.params.get(pos)) {
                        // Only include if the f1 param is directly a Type::Var (not a function containing a var)
                        if matches!(f1_param, Type::Var(_)) {
                            let resolved = self.apply_full_subst(f2_param);
                            // Skip if still a type variable
                            if !matches!(resolved, Type::Var(_)) {
                                concrete_types.push((pos, resolved.display()));
                            }
                        }
                    }
                }

                // Check for conflicts among the concrete types
                // Only report as annotation-needed if the types are both "simple" (primitives)
                // Complex types (functions, etc.) indicate a different kind of error
                if concrete_types.len() >= 2 {
                    let is_simple_type = |s: &str| {
                        matches!(s, "Int" | "String" | "Float" | "Bool" | "Char" | "()" |
                                 "Int8" | "Int16" | "Int32" | "Int64" |
                                 "UInt8" | "UInt16" | "UInt32" | "UInt64" |
                                 "Float32" | "Float64" | "BigInt" | "Decimal")
                    };

                    let first = &concrete_types[0].1;
                    for (pos, ty) in &concrete_types[1..] {
                        // Only report if both types are simple primitives
                        // Complex types like functions indicate a different error
                        if ty != first && is_simple_type(first) && is_simple_type(ty) {
                            // Found a conflict: same type variable, different simple types
                            return Some(TypeError::AnnotationRequired {
                                func: "<function>".to_string(),
                                param: format!("parameters {} and {}", concrete_types[0].0 + 1, pos + 1),
                                type1: first.clone(),
                                type2: ty.clone(),
                            });
                        }
                    }
                }
            }
        }

        None
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

    /// Peek at pending constraints to see what a type variable will resolve to.
    /// This is used during pattern matching (e.g., for scalar dispatch) to look ahead
    /// at constraints that will be solved later.
    /// Returns the concrete type if found in pending constraints, otherwise returns the input type.
    fn peek_pending_resolution(&self, ty: &Type) -> Type {
        if let Type::Var(var_id) = ty {
            // Search through pending constraints for ones that bind this variable
            for constraint in &self.constraints {
                if let Constraint::Equal(t1, t2, _) = constraint {
                    // Check if this constraint binds our variable to a concrete type
                    if let Type::Var(id1) = t1 {
                        if *id1 == *var_id {
                            // Found Var(var_id) = t2
                            if matches!(t2, Type::Named { .. } | Type::Record(_) | Type::Variant(_)) {
                                return t2.clone();
                            }
                        }
                    }
                    if let Type::Var(id2) = t2 {
                        if *id2 == *var_id {
                            // Found t1 = Var(var_id)
                            if matches!(t1, Type::Named { .. } | Type::Record(_) | Type::Variant(_)) {
                                return t1.clone();
                            }
                        }
                    }

                    // Special case: Function types with return type constraints
                    // e.g., Function(...ret: Named) = Function(...ret: Var(24))
                    if let (Type::Function(ft1), Type::Function(ft2)) = (t1, t2) {
                        if let Type::Var(ret_id) = ft2.ret.as_ref() {
                            if *ret_id == *var_id
                                && matches!(ft1.ret.as_ref(), Type::Named { .. } | Type::Record(_) | Type::Variant(_))
                            {
                                return ft1.ret.as_ref().clone();
                            }
                        }
                        if let Type::Var(ret_id) = ft1.ret.as_ref() {
                            if *ret_id == *var_id
                                && matches!(ft2.ret.as_ref(), Type::Named { .. } | Type::Record(_) | Type::Variant(_))
                            {
                                return ft2.ret.as_ref().clone();
                            }
                        }
                    }
                }
            }
        }
        ty.clone()
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
            Type::Named { name, .. } => {
                // Normalize qualified names like "stdlib.list.Option" to "Option"
                let short = name.rsplit('.').next().unwrap_or(&name);
                Some(short.to_string())
            }
            Type::Pid => Some("Pid".to_string()),
            Type::Ref => Some("Ref".to_string()),
            Type::IO(_) => Some("IO".to_string()),
            Type::Tuple(_) => Some("Tuple".to_string()),
            // Variant types can have methods
            Type::Variant(vt) => Some(vt.name.clone()),
            // Type variables can't be looked up - type is unknown
            Type::Var(_) | Type::TypeParam(_) => None,
            // Never type - bottom type from error()/throw(), allow method calls in dead code
            Type::Never => None,
            // These are concrete types that don't support collection/string methods.
            // Return type names so check_pending_method_calls can report proper errors.
            Type::Unit => Some("Unit".to_string()),
            Type::Function(_) => Some("Function".to_string()),
            Type::Record(_) => Some("Record".to_string()),
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
            var_subst.entry(var_id).or_insert_with(|| self.fresh());
        }

        // Also handle explicit type parameters.
        // Two passes: first create fresh vars for ALL params (so HasField can reference any param),
        // then process constraints.
        // IMPORTANT: Use the var_subst fresh vars when a TypeParam maps to a known Var ID,
        // so that HasField result types are unified with the return type's vars.
        // e.g., TypeParam "b" maps to Var(2), and var_subst[2] = ?fresh_100.
        // HasField(0,b) must use ?fresh_100 so it connects to the return type.
        let mut param_subst: HashMap<String, Type> = HashMap::new();
        for type_param in &func_ty.type_params {
            // Try to find existing var_subst entry for this type param
            // TypeParam name "a" → Var(1), "b" → Var(2), etc.
            let existing_var = if type_param.name.len() == 1 {
                let ch = type_param.name.chars().next().unwrap();
                if ch.is_ascii_lowercase() {
                    let orig_id = (ch as u32) - ('a' as u32) + 1;
                    var_subst.get(&orig_id).cloned()
                } else {
                    None
                }
            } else {
                None
            };
            let fresh_var = existing_var.unwrap_or_else(|| self.fresh());
            param_subst.insert(type_param.name.clone(), fresh_var);
        }

        // Second pass: add trait constraints for each type parameter's fresh variable.
        for type_param in &func_ty.type_params {
            let fresh_var = param_subst.get(&type_param.name).cloned().unwrap();

            // Only push HasTrait for well-known traits that the type system can check
            // (Eq, Ord, Num, Concat, Hash, Show). User-defined traits are stored in
            // trait_bounds for signature propagation but not actively checked via HasTrait
            // because the type environment may not have all trait impls registered.
            if let Type::Var(var_id) = &fresh_var {
                for constraint in &type_param.constraints {
                    // Handle HasMethod constraints with optional arg types and return param:
                    //   HasMethod(name) - simple existence check
                    //   HasMethod(name|arg1|arg2) - with concrete/param arg types
                    //   HasMethod(name,retparam) - with return type param (no arg types)
                    //   HasMethod(name|arg1,retparam) - with both arg types and return param
                    // Arg types enable checking that callers provide compatible types.
                    // e.g., hasString(xs) = xs.contains("hello") → HasMethod(contains|String) a
                    if let Some(inner) = constraint.strip_prefix("HasMethod(").and_then(|s| s.strip_suffix(')')) {
                        // Split on comma for return param (comma is AFTER any |args)
                        let (method_and_args, result_param_opt) = if let Some(comma_pos) = inner.find(',') {
                            (&inner[..comma_pos], Some(inner[comma_pos + 1..].trim()))
                        } else {
                            (inner, None)
                        };
                        // Split method_and_args on | for arg types
                        let parts: Vec<&str> = method_and_args.split('|').collect();
                        let method_name = parts[0];
                        let encoded_arg_types = &parts[1..]; // may be empty

                        // Parse arg types from their string representations
                        let parsed_arg_types: Vec<Type> = encoded_arg_types.iter().map(|s| {
                            Self::parse_simple_type(s, &param_subst, &var_subst)
                        }).collect();

                        if result_param_opt.is_some() || !parsed_arg_types.is_empty() {
                            // Has return param or arg types - push as PendingMethodCall
                            let result_ty = if let Some(result_param) = result_param_opt {
                                param_subst.get(result_param)
                                    .cloned()
                                    .or_else(|| {
                                        if result_param.len() == 1 {
                                            let ch = result_param.chars().next().unwrap();
                                            if ch.is_ascii_lowercase() {
                                                let orig_id = (ch as u32) - ('a' as u32) + 1;
                                                var_subst.get(&orig_id).cloned()
                                            } else { None }
                                        } else { None }
                                    })
                                    .unwrap_or_else(|| self.fresh())
                            } else {
                                self.fresh()
                            };
                            self.pending_method_calls.push(PendingMethodCall {
                                receiver_ty: fresh_var.clone(),
                                method_name: method_name.to_string(),
                                arg_types: parsed_arg_types,
                                named_args: vec![],
                                ret_ty: result_ty,
                                span: None,
                            });
                        } else {
                            // HasMethod(name) - simple existence check
                            self.deferred_method_existence_checks.push((
                                fresh_var.clone(),
                                method_name.to_string(),
                                None,
                            ));
                        }
                        // Also store in trait_bounds for the transfer section
                        self.add_trait_bound(*var_id, constraint.clone());
                    // Handle HasField constraints: "HasField(fieldname)" or "HasField(fieldname,resultparam)"
                    // The resultparam variant links the field result to a type param in the return type,
                    // enabling proper type flow through generic field access (e.g., swap(p) = (p.1, p.0)).
                    } else if let Some(inner) = constraint.strip_prefix("HasField(").and_then(|s| s.strip_suffix(')')) {
                        let (field_name, field_ty) = if let Some(comma_pos) = inner.find(',') {
                            let fname = &inner[..comma_pos];
                            let result_param = inner[comma_pos + 1..].trim();
                            // Look up the result type param's fresh var.
                            // First check param_subst (type params with constraints),
                            // then fall back to var_subst (type params only in return type).
                            let result_ty = param_subst.get(result_param)
                                .cloned()
                                .or_else(|| {
                                    if result_param.len() == 1 {
                                        let ch = result_param.chars().next().unwrap();
                                        if ch.is_ascii_lowercase() {
                                            let orig_id = (ch as u32) - ('a' as u32) + 1;
                                            var_subst.get(&orig_id).cloned()
                                        } else { None }
                                    } else { None }
                                })
                                .unwrap_or_else(|| self.fresh());
                            (fname.to_string(), result_ty)
                        } else {
                            (inner.to_string(), self.fresh())
                        };
                        self.constraints.push(Constraint::HasField(
                            fresh_var.clone(),
                            field_name,
                            field_ty,
                        ));
                        // Also store in trait_bounds so transfer section can copy to var_subst vars
                        self.add_trait_bound(*var_id, constraint.clone());
                    } else {
                        self.add_trait_bound(*var_id, constraint.clone());
                        if matches!(constraint.as_str(), "Eq" | "Ord" | "Num" | "Concat" | "Hash" | "Show") {
                            self.constraints.push(Constraint::HasTrait(
                                fresh_var.clone(),
                                constraint.clone(),
                            ));
                        }
                    }
                }
            }
        }

        // CRITICAL FIX: Transfer trait bounds from param_subst to the CORRESPONDING var_subst vars.
        // When a function signature uses Var(N) instead of TypeParam("a") in the params,
        // the fresh var created in var_subst won't have the trait bounds. Copy the bounds
        // from the constrained type parameter to its corresponding var.
        // The mapping is: TypeParam "a" → Var(1), "b" → Var(2), etc. (from type_name_to_type)
        // So look up that original Var ID in var_subst to find the fresh var.
        for (tp_name, param_fresh_var) in &param_subst {
            if let Type::Var(param_var_id) = param_fresh_var {
                let bounds = self.trait_bounds.get(param_var_id).cloned().unwrap_or_default();
                if bounds.is_empty() {
                    continue;
                }
                // Map type param name to original Var ID: "a" → 1, "b" → 2, etc.
                let original_var_id = if tp_name.len() == 1 {
                    let ch = tp_name.chars().next().unwrap();
                    if ch.is_ascii_lowercase() {
                        Some((ch as u32) - ('a' as u32) + 1)
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Find the corresponding fresh var in var_subst
                if let Some(orig_id) = original_var_id {
                    if let Some(var_subst_var) = var_subst.get(&orig_id) {
                        if let Type::Var(var_subst_id) = var_subst_var {
                            for bound in &bounds {
                                if let Some(inner) = bound.strip_prefix("HasMethod(").and_then(|s| s.strip_suffix(')')) {
                                    // Parse same format as main section: name|arg1|arg2,retparam
                                    let (method_and_args, result_param_opt) = if let Some(comma_pos) = inner.find(',') {
                                        (&inner[..comma_pos], Some(inner[comma_pos + 1..].trim()))
                                    } else {
                                        (inner, None)
                                    };
                                    let parts: Vec<&str> = method_and_args.split('|').collect();
                                    let method_name = parts[0];
                                    let encoded_arg_types = &parts[1..];
                                    let parsed_arg_types: Vec<Type> = encoded_arg_types.iter().map(|s| {
                                        Self::parse_simple_type(s, &param_subst, &var_subst)
                                    }).collect();

                                    if result_param_opt.is_some() || !parsed_arg_types.is_empty() {
                                        let result_ty = if let Some(result_param) = result_param_opt {
                                            param_subst.get(result_param)
                                                .cloned()
                                                .or_else(|| {
                                                    if result_param.len() == 1 {
                                                        let ch = result_param.chars().next().unwrap();
                                                        if ch.is_ascii_lowercase() {
                                                            let orig_id = (ch as u32) - ('a' as u32) + 1;
                                                            var_subst.get(&orig_id).cloned()
                                                        } else { None }
                                                    } else { None }
                                                })
                                                .unwrap_or_else(|| self.fresh())
                                        } else {
                                            self.fresh()
                                        };
                                        self.pending_method_calls.push(PendingMethodCall {
                                            receiver_ty: var_subst_var.clone(),
                                            method_name: method_name.to_string(),
                                            arg_types: parsed_arg_types,
                                            named_args: vec![],
                                            ret_ty: result_ty,
                                            span: None,
                                        });
                                    } else {
                                        self.deferred_method_existence_checks.push((
                                            var_subst_var.clone(),
                                            method_name.to_string(),
                                            None,
                                        ));
                                    }
                                } else if let Some(inner) = bound.strip_prefix("HasField(").and_then(|s| s.strip_suffix(')')) {
                                    let (field_name, field_ty) = if let Some(comma_pos) = inner.find(',') {
                                        let fname = &inner[..comma_pos];
                                        let result_param = inner[comma_pos + 1..].trim();
                                        let result_ty = param_subst.get(result_param)
                                            .cloned()
                                            .or_else(|| {
                                                if result_param.len() == 1 {
                                                    let ch = result_param.chars().next().unwrap();
                                                    if ch.is_ascii_lowercase() {
                                                        let orig_id = (ch as u32) - ('a' as u32) + 1;
                                                        var_subst.get(&orig_id).cloned()
                                                    } else { None }
                                                } else { None }
                                            })
                                            .unwrap_or_else(|| self.fresh());
                                        (fname.to_string(), result_ty)
                                    } else {
                                        (inner.to_string(), self.fresh())
                                    };
                                    self.constraints.push(Constraint::HasField(
                                        var_subst_var.clone(),
                                        field_name,
                                        field_ty,
                                    ));
                                } else {
                                    self.add_trait_bound(*var_subst_id, bound.clone());
                                    if matches!(bound.as_str(), "Eq" | "Ord" | "Num" | "Concat" | "Hash" | "Show") {
                                        self.constraints.push(Constraint::HasTrait(
                                            var_subst_var.clone(),
                                            bound.clone(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Substitute both Var IDs and type parameters
        let instantiated_params: Vec<Type> = func_ty.params
            .iter()
            .map(|p| self.freshen_type(p, &var_subst, &param_subst))
            .collect();
        let instantiated_ret = self.freshen_type(&func_ty.ret, &var_subst, &param_subst);

        Type::Function(FunctionType { required_params: func_ty.required_params,
            type_params: func_ty.type_params.clone(), // Preserve for check_annotation_required
            params: instantiated_params,
            ret: Box::new(instantiated_ret),
        })
    }

    /// Collect all Var IDs in a type
    #[allow(clippy::only_used_in_recursion)]
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
            Type::Named { args, .. } => {
                for arg in args {
                    self.collect_var_ids(arg, ids);
                }
            }
            Type::Record(rec) => {
                for (_, field_ty, _) in &rec.fields {
                    self.collect_var_ids(field_ty, ids);
                }
            }
            Type::IO(inner) => {
                self.collect_var_ids(inner, ids);
            }
            _ => {}
        }
    }

    /// Parse a simple type string from a signature encoding back into a Type.
    /// Handles primitives (Int, String, Bool, etc.), single-letter type params,
    /// and basic compound types (List[X], (X, Y)).
    fn parse_simple_type(s: &str, param_subst: &HashMap<String, Type>, var_subst: &HashMap<u32, Type>) -> Type {
        match s.trim() {
            "Int" => Type::Int,
            "String" => Type::String,
            "Bool" => Type::Bool,
            "Float" => Type::Float,
            "Char" => Type::Char,
            "()" => Type::Unit,
            s if s.starts_with("List[") && s.ends_with(']') => {
                let inner = &s[5..s.len() - 1];
                Type::List(Box::new(Self::parse_simple_type(inner, param_subst, var_subst)))
            }
            s if s.len() == 1 && s.chars().next().unwrap().is_ascii_lowercase() => {
                let ch = s.chars().next().unwrap();
                // Look up in param_subst first, then var_subst
                param_subst.get(s).cloned()
                    .or_else(|| {
                        let orig_id = (ch as u32) - ('a' as u32) + 1;
                        var_subst.get(&orig_id).cloned()
                    })
                    .unwrap_or(Type::String) // Should not happen in practice
            }
            _ => {
                // Unknown type string - return a Named type as fallback
                Type::Named { name: s.to_string(), args: vec![] }
            }
        }
    }

    /// Freshen a type by replacing Var IDs and TypeParams with fresh variables
    #[allow(clippy::only_used_in_recursion)]
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
            Type::Named { name, args } => {
                // Handle type parameters that were stored as Named types with no args
                // (e.g., "T" might be Named { name: "T", args: [] } instead of TypeParam("T"))
                if args.is_empty() {
                    if let Some(replacement) = param_subst.get(name) {
                        return replacement.clone();
                    }
                }
                Type::Named {
                    name: name.clone(),
                    args: args.iter().map(|a| self.freshen_type(a, var_subst, param_subst)).collect(),
                }
            }
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

    pub fn pending_method_calls_count(&self) -> usize {
        self.pending_method_calls.len()
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Check deferred function call arg/param pairs for structural type mismatches.
    /// This should be called after solve() (even if solve() returned an error) to catch
    /// cases like passing a Map where a List is expected. Returns the first Mismatch error
    /// found, or None if all checks pass.
    ///
    /// Only reports mismatches where the PARAM type (from the function signature) is a
    /// concrete collection type (List, Map, Set, Tuple). This avoids false positives from
    /// generic functions like assert_eq(a, a) where the param type is a type variable.
    pub fn check_fn_call_mismatches(&mut self) -> Option<TypeError> {
        for (param_ty, arg_ty, span) in &self.deferred_fn_call_checks {
            let resolved_param = self.env.apply_subst(param_ty);
            let resolved_arg = self.env.apply_subst(arg_ty);

            // Only check when param type is a CONCRETE collection type (not a type variable).
            // This means the function explicitly expects a specific collection type.
            let param_is_concrete_collection = matches!(&resolved_param,
                Type::List(_) | Type::Map(_, _) | Type::Set(_) | Type::Tuple(_));
            if !param_is_concrete_collection {
                continue;
            }

            // Skip if arg is still a type variable (not yet resolved)
            if matches!(&resolved_arg, Type::Var(_) | Type::TypeParam(_)) {
                continue;
            }

            // Check structural mismatch: different collection base types
            // Also check element-level mismatches within containers
            let is_structural_mismatch = |param: &Type, arg: &Type| -> bool {
                match (param, arg) {
                    // Tuple vs non-tuple is always a mismatch
                    (Type::Tuple(_), Type::Int | Type::Float | Type::Bool |
                     Type::String | Type::Char | Type::Unit) => true,
                    (Type::Int | Type::Float | Type::Bool |
                     Type::String | Type::Char | Type::Unit, Type::Tuple(_)) => true,
                    // Different-sized tuples
                    (Type::Tuple(a), Type::Tuple(b)) => a.len() != b.len(),
                    _ => false,
                }
            };
            let is_mismatch = match (&resolved_param, &resolved_arg) {
                // Same container type: check element-level structural mismatch
                (Type::List(inner_p), Type::List(inner_a)) => {
                    let rp = self.env.apply_subst(inner_p);
                    let ra = self.env.apply_subst(inner_a);
                    // Skip if either element is still a type variable
                    if matches!(&rp, Type::Var(_)) || matches!(&ra, Type::Var(_)) {
                        false
                    } else {
                        is_structural_mismatch(&rp, &ra)
                    }
                }
                (Type::Map(_, _), Type::Map(_, _)) => false,
                (Type::Set(_), Type::Set(_)) => false,
                (Type::Tuple(a), Type::Tuple(b)) => a.len() != b.len(),
                // Param is a collection but arg is something different
                (Type::List(_), _) | (Type::Map(_, _), _) |
                (Type::Set(_), _) | (Type::Tuple(_), _) => true,
                _ => false,
            };

            if is_mismatch {
                self.last_error_span = Some(*span);
                return Some(TypeError::Mismatch {
                    expected: resolved_param.display(),
                    found: resolved_arg.display(),
                });
            }
        }
        None
    }

    /// Check deferred indirect call mismatches (curried calls, higher-order returns).
    /// When func_ty is a Var at inference time (e.g., `adder(1)("hello")` where
    /// adder(1) returns a Var), we defer the check. After solve(), resolve func_ty
    /// to a Function type and check each param vs arg for concrete type mismatches.
    pub fn check_indirect_call_mismatches(&mut self) -> Option<TypeError> {
        for (func_ty, arg_types, span) in &self.deferred_indirect_call_checks {
            let resolved_func = self.env.apply_subst(func_ty);
            if let Type::Function(ft) = &resolved_func {
                for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                    let resolved_param = self.env.apply_subst(param_ty);
                    let resolved_arg = self.env.apply_subst(arg_ty);

                    // Skip if either is still a type variable (not yet resolved)
                    if matches!(&resolved_param, Type::Var(_) | Type::TypeParam(_)) { continue; }
                    if matches!(&resolved_arg, Type::Var(_) | Type::TypeParam(_)) { continue; }

                    // Check concrete type mismatch: both are concrete and differ
                    let is_concrete_simple = |t: &Type| -> bool {
                        matches!(t, Type::Int | Type::Float | Type::Bool |
                                    Type::String | Type::Char | Type::Unit)
                    };
                    let is_collection = |t: &Type| -> bool {
                        matches!(t, Type::List(_) | Type::Map(_, _) | Type::Set(_) | Type::Tuple(_))
                    };

                    // Simple type vs anything different
                    if is_concrete_simple(&resolved_param) && resolved_param != resolved_arg {
                        self.last_error_span = Some(*span);
                        return Some(TypeError::Mismatch {
                            expected: resolved_param.display(),
                            found: resolved_arg.display(),
                        });
                    }

                    // Collection type vs incompatible type
                    if is_collection(&resolved_param) {
                        let compatible = match (&resolved_param, &resolved_arg) {
                            (Type::List(_), Type::List(_)) => true,
                            (Type::Map(_, _), Type::Map(_, _)) => true,
                            (Type::Set(_), Type::Set(_)) => true,
                            (Type::Tuple(a), Type::Tuple(b)) => a.len() == b.len(),
                            _ => false,
                        };
                        if !compatible {
                            self.last_error_span = Some(*span);
                            return Some(TypeError::Mismatch {
                                expected: resolved_param.display(),
                                found: resolved_arg.display(),
                            });
                        }
                    }

                    // Named type vs different type
                    if let Type::Named { name: ref name_p, .. } = &resolved_param {
                        match &resolved_arg {
                            Type::Named { name: ref name_a, .. } if name_a != name_p => {
                                self.last_error_span = Some(*span);
                                return Some(TypeError::Mismatch {
                                    expected: resolved_param.display(),
                                    found: resolved_arg.display(),
                                });
                            }
                            Type::Named { .. } => {} // Same name, compatible
                            _ => {
                                // Named vs primitive/collection
                                self.last_error_span = Some(*span);
                                return Some(TypeError::Mismatch {
                                    expected: resolved_param.display(),
                                    found: resolved_arg.display(),
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Check deferred typed binding annotations.
    /// After solve(), verify that resolved value types match their annotations.
    /// This catches mismatches like `b: Box[Int] = Box(value: "hello")` that
    /// the batch unify might have silently dropped.
    pub fn check_typed_binding_mismatches(&mut self) -> Option<TypeError> {
        for (value_ty, ann_ty, span) in &self.deferred_typed_binding_checks {
            let resolved_value = self.env.apply_subst(value_ty);
            let resolved_ann = self.env.apply_subst(ann_ty);

            // Check structural container mismatches even with unresolved type vars.
            // E.g., List[?25] vs Set[Int] - different container types, always an error.
            let is_structural_mismatch = match (&resolved_value, &resolved_ann) {
                (Type::List(_), Type::Set(_)) | (Type::Set(_), Type::List(_)) => true,
                (Type::List(_), Type::Map(_, _)) | (Type::Map(_, _), Type::List(_)) => true,
                (Type::Set(_), Type::Map(_, _)) | (Type::Map(_, _), Type::Set(_)) => true,
                (Type::List(_), Type::Named { .. }) | (Type::Named { .. }, Type::List(_)) => true,
                (Type::Set(_), Type::Named { .. }) | (Type::Named { .. }, Type::Set(_)) => true,
                (Type::Map(_, _), Type::Named { .. }) | (Type::Named { .. }, Type::Map(_, _)) => true,
                _ => false,
            };
            if is_structural_mismatch {
                self.last_error_span = Some(*span);
                return Some(TypeError::Mismatch {
                    expected: resolved_ann.display(),
                    found: resolved_value.display(),
                });
            }

            // Skip if either is still unresolved (for non-structural checks)
            if resolved_value.has_any_type_var() || resolved_ann.has_any_type_var() {
                continue;
            }

            // Check parameterized types: generic Named types (Box[Int], Option[String])
            // and builtin container types (Set[Int], Map[K,V], List[T]).
            // Normal unification handles these but its errors may get dropped as
            // UnificationFailed. This deferred check surfaces mismatches that slip through.
            let is_parameterized = matches!(&resolved_ann,
                Type::Named { args, .. } if !args.is_empty())
                || matches!(&resolved_ann, Type::Set(_) | Type::Map(_, _) | Type::List(_));
            if !is_parameterized {
                continue;
            }

            // Compare display strings to avoid false positives from internal Type representation
            // differences (e.g., two Option[Val] with different internal ids but same semantics)
            let val_str = resolved_value.display();
            let ann_str = resolved_ann.display();
            if val_str != ann_str {
                self.last_error_span = Some(*span);
                return Some(TypeError::Mismatch {
                    expected: ann_str,
                    found: val_str,
                });
            }
        }
        None
    }

    /// Check deferred branch type mismatches from if-else and match expressions.
    /// After solve(), resolve both branch type and result type, check if they're
    /// structurally different container types (e.g., List vs Set in different branches).
    /// Returns Mismatch error that bypasses string-based filters in compile.rs.
    pub fn check_branch_type_mismatches(&mut self) -> Option<TypeError> {
        for (type_a, type_b, span) in &self.deferred_branch_type_checks {
            let resolved_a = self.env.apply_subst(type_a);
            let resolved_b = self.env.apply_subst(type_b);

            // Check structural container mismatches
            let is_structural_mismatch = match (&resolved_a, &resolved_b) {
                (Type::List(_), Type::Set(_)) | (Type::Set(_), Type::List(_)) => true,
                (Type::List(_), Type::Map(_, _)) | (Type::Map(_, _), Type::List(_)) => true,
                (Type::Set(_), Type::Map(_, _)) | (Type::Map(_, _), Type::Set(_)) => true,
                _ => false,
            };
            if is_structural_mismatch {
                self.last_error_span = Some(*span);
                return Some(TypeError::Mismatch {
                    expected: resolved_a.display(),
                    found: resolved_b.display(),
                });
            }
        }
        None
    }

    /// Check deferred generic function trait bounds.
    /// When a generic function with trait bounds is called (e.g., equal[T: Eq]),
    /// fresh vars get trait bounds. After solve(), verify that the resolved types
    /// satisfy those bounds. This specifically catches function types passed to
    /// constrained generics - functions never implement Eq/Ord/Num.
    ///
    /// Unlike the general trait_bounds check (which skips unresolved vars),
    /// this targets ONLY bounds from explicit type parameters, so it can safely
    /// check function types even when inner vars are unresolved.
    pub fn check_generic_trait_bounds(&mut self) -> Option<TypeError> {
        // Only check builtin traits here. User-defined traits are checked through
        // the trait system's method dispatch. The implements() method on TypeEnv
        // may not have user-defined trait impls registered in the type_check_fn context.
        let builtin_traits = ["Eq", "Ord", "Num", "Hash", "Show", "Concat"];
        for (arg_ty, trait_name, span) in &self.deferred_generic_trait_checks {
            if !builtin_traits.contains(&trait_name.as_str()) {
                continue;
            }
            let resolved = self.env.apply_subst(arg_ty);
            // Skip unresolved type variables - can't check yet
            if let Type::Var(_) = &resolved {
                continue;
            }
            // Skip type parameters - they represent generic params whose bounds
            // are checked at the caller's call site, not here
            if let Type::TypeParam(_) = &resolved {
                continue;
            }
            // Skip types that still contain unresolved vars (partially resolved)
            if resolved.has_any_type_var() {
                continue;
            }
            // Skip Named types that look like unresolved type variables (e.g., "?429")
            // These arise from stale substitutions in pending_fn_signatures
            if let Type::Named { name, .. } = &resolved {
                if name.starts_with('?') {
                    continue;
                }
            }
            // Check if the concrete type implements the required trait
            if !self.env.implements(&resolved, trait_name) {
                self.last_error_span = Some(*span);
                // Use Mismatch to bypass the string-based error filters in compile.rs
                // that would suppress MissingTraitImpl with type variables
                return Some(TypeError::Mismatch {
                    expected: format!("type implementing {}", trait_name),
                    found: resolved.display(),
                });
            }
        }
        None
    }

    pub fn pending_method_calls_debug(&self) -> Vec<(String, String)> {
        self.pending_method_calls.iter().map(|c| {
            (c.method_name.clone(), format!("{:?}", c.receiver_ty))
        }).collect()
    }

    pub fn solve(&mut self) -> Result<(), TypeError> {
        const MAX_ITERATIONS: usize = 1000;
        let mut iteration = 0;
        let mut deferred_count = 0;

        while let Some(constraint) = self.constraints.pop() {
            iteration += 1;
            if iteration > MAX_ITERATIONS {
                // Too many iterations - likely an infinite loop due to unresolved constraints
                // solve_completed stays false to indicate incomplete resolution
                #[cfg(debug_assertions)]
                {
                    eprintln!(
                        "[TYPE-INVARIANT] solve() hit MAX_ITERATIONS ({}) - {} constraints may be unresolved",
                        MAX_ITERATIONS,
                        self.constraints.len()
                    );
                    // Warn about pending method calls that may not be properly validated
                    if !self.pending_method_calls.is_empty() {
                        eprintln!(
                            "[TYPE-INVARIANT] {} pending method calls may have incomplete type checking",
                            self.pending_method_calls.len()
                        );
                    }
                }
                // Still finalize what we have, but mark as incomplete
                // Note: check_pending_method_calls may produce spurious errors since
                // constraint solving is incomplete - use .ok() to ignore them
                self.check_pending_method_calls().ok();
                self.finalize_expr_types();
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
                        // Check if this unification failure involves a parameter that needs annotation
                        match self.check_annotation_required(&t1, &t2, &e) {
                            Some(better_error) => return Err(better_error),
                            None => {
                                // check_annotation_required returns None in two cases:
                                // 1. Not a function type unification (no annotation relevant)
                                // 2. Multi-type-param function where HM merged distinct type
                                //    params - false positive, skip the error
                                // Case 2 is detected by checking if either side is a Function
                                // with 2+ type_params AND the inner error is a simple type
                                // mismatch (not a structural mismatch like Function vs List).
                                let is_multi_tp_false_positive = match (&t1, &t2) {
                                    (Type::Function(f), _) | (_, Type::Function(f)) => {
                                        if f.type_params.len() >= 2 {
                                            // Only suppress simple type mismatches (e.g., Int vs String
                                            // from merged type params). Don't suppress structural
                                            // mismatches (Function vs List, etc.) which are real errors.
                                            match &e {
                                                TypeError::UnificationFailed(a, b) => {
                                                    let is_structural = a.contains("->") || b.contains("->")
                                                        || a.contains("List[") || b.contains("List[")
                                                        || a.contains("Map[") || b.contains("Map[")
                                                        || a.contains("Set[") || b.contains("Set[");
                                                    !is_structural
                                                }
                                                _ => false,
                                            }
                                        } else {
                                            false
                                        }
                                    }
                                    _ => false,
                                };
                                if !is_multi_tp_false_positive {
                                    return Err(e);
                                }
                                // Multi-type-param false positive - skip this constraint
                            }
                        }
                    }
                    deferred_count = 0; // Made progress
                }
                Constraint::HasTrait(ty, trait_name) => {
                    let resolved = self.env.apply_subst(&ty);
                    match &resolved {
                        Type::Var(var_id) => {
                            // Track trait bound on the type variable
                            self.add_trait_bound(*var_id, trait_name.clone());
                            // Also record the original constraint for post-solve retry.
                            // Unlike trait_bounds (which get merged during variable
                            // unification), this preserves the original type so we can
                            // re-check after all substitutions are finalized.
                            self.deferred_has_trait.push((ty, trait_name));
                            deferred_count = 0; // Made progress (recorded the bound)
                        }
                        _ => {
                            // Check if type implements the trait
                            if self.env.implements(&resolved, &trait_name) {
                                deferred_count = 0; // Made progress
                            } else if matches!(&resolved, Type::Function(_)) {
                                // Function types NEVER implement standard traits (Num, Concat,
                                // Eq, Ord, etc.) regardless of their param/return types.
                                // Check this BEFORE has_any_type_var because function types
                                // often have unresolved param/return type vars, but that
                                // doesn't change the fact that functions can't be compared,
                                // added, sorted, etc.
                                if resolved.has_any_type_var() {
                                    // Function type with unresolved vars - defer to post-solve.
                                    // In batch inference, type variables from different functions
                                    // can get confused, leading to false positives (e.g. a Num
                                    // constraint from one function landing on a callback type
                                    // from another). Deferring gives more time for types to
                                    // resolve and avoids these false positives.
                                    self.deferred_has_trait.push((ty, trait_name));
                                    deferred_count = 0;
                                } else {
                                    // Fully concrete function type - definitely an error.
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved.display(),
                                        trait_name,
                                    });
                                }
                            } else if resolved.has_any_type_var() {
                                // Type still contains unresolved variables (e.g. List[?a])
                                // For traits like Eq/Show that depend on element types, defer.
                                // But for traits like Num that container types NEVER implement
                                // regardless of element types, report immediately.
                                let outer_is_container = matches!(&resolved,
                                    Type::List(_) | Type::Map(_, _) | Type::Set(_) |
                                    Type::Array(_) | Type::Tuple(_));
                                if outer_is_container && matches!(trait_name.as_str(), "Num" | "Ord") {
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved.display(),
                                        trait_name,
                                    });
                                }
                                // Save for post-solve retry - type vars may resolve later
                                self.deferred_has_trait.push((ty, trait_name));
                                deferred_count = 0;
                            } else {
                                // Fully concrete type that doesn't implement the trait
                                return Err(TypeError::MissingTraitImpl {
                                    ty: resolved.display(),
                                    trait_name,
                                });
                            }
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
                        Type::Named { name, args } => {
                            // Check if this is a known record or variant type
                            if let Some(typedef) = self.env.lookup_type(name).cloned() {
                                match &typedef {
                                    crate::TypeDef::Record { params, fields, .. } => {
                                        if let Some((_, actual_ty, _)) =
                                            fields.iter().find(|(n, _, _)| n == &field)
                                        {
                                            // Build substitution from type params to type args
                                            // e.g., for Wrapper[Int], substitute T -> Int
                                            let subst: HashMap<String, Type> = params.iter()
                                                .map(|p| p.name.clone())
                                                .zip(args.iter().cloned())
                                                .collect();
                                            let substituted_ty = Self::substitute_type_params(actual_ty, &subst);
                                            self.unify_types(&substituted_ty, &expected_ty)?;
                                            deferred_count = 0; // Made progress
                                        } else {
                                            return Err(TypeError::NoSuchField {
                                                ty: resolved.display(),
                                                field,
                                            });
                                        }
                                    }
                                    crate::TypeDef::Variant { params, constructors } => {
                                        // Only allow field access on single-constructor variants
                                        // (which act like records). Multi-constructor variants
                                        // require pattern matching since the VM can't resolve
                                        // field names across different constructors.
                                        if constructors.len() == 1 {
                                            if let Some(Constructor::Named(_, fields)) = constructors.first() {
                                                if let Some((_, actual_ty)) = fields.iter().find(|(n, _)| n == &field) {
                                                    let subst: HashMap<String, Type> = params.iter()
                                                        .map(|p| p.name.clone())
                                                        .zip(args.iter().cloned())
                                                        .collect();
                                                    let substituted_ty = Self::substitute_type_params(actual_ty, &subst);
                                                    self.unify_types(&substituted_ty, &expected_ty)?;
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
                                // Save this constraint for post-method-call re-check
                                // Don't drop it - it may become checkable after
                                // check_pending_method_calls resolves more types
                                self.deferred_has_field.push((ty, field, expected_ty));
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
                        Type::List(_) => {
                            // Lists don't support field access (.0, .1, etc.) -
                            // that's only for tuples. Use list.get(idx) instead.
                            return Err(TypeError::NoSuchField {
                                ty: resolved.display(),
                                field,
                            });
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

        // Post-solve: retry deferred HasTrait constraints now that type variables
        // may have been resolved through unification.
        // This catches cases like Circle(5) + Square(3) where the constructor return
        // type (Shape) was still a Var when the HasTrait(Num) constraint was first
        // processed, but is now resolved after all Equal constraints are solved.
        //
        // We use deferred_has_trait (which tracks the ORIGINAL constraint types)
        // rather than trait_bounds (which includes merged bounds from variable
        // unification that may create false positives).
        for (ty, trait_name) in &self.deferred_has_trait.clone() {
            let resolved = self.env.apply_subst(ty);
            match &resolved {
                Type::Var(_) => {} // Still unresolved, skip
                Type::Function(_) => {
                    // Function types NEVER implement standard traits (Eq, Ord, Num, etc.)
                    // regardless of their parameter/return types. So even with unresolved
                    // type vars in the function signature, this is always an error.
                    // Display the type, substituting what we can.
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: trait_name.clone(),
                    });
                }
                _ => {
                    if !self.env.implements(&resolved, trait_name) {
                        if resolved.has_any_type_var() {
                            // Type still has unresolved vars. Use definitive check
                            // that handles vars conservatively - only report if we
                            // can be CERTAIN the type fails (e.g., contains a Function
                            // type which never implements Eq/Num/Ord/etc.)
                            if !self.env.definitely_not_implements(&resolved, trait_name) {
                                continue;
                            }
                            // Fall through to report the definitive error
                        }
                        // Type is fully concrete (or definitely doesn't implement).
                        self.last_error_span = None;
                        let display_ty = if let Type::Named { name, args, .. } = &resolved {
                            if args.iter().any(|a| a.has_any_type_var()) {
                                name.rsplit('.').next().unwrap_or(name).to_string()
                            } else {
                                resolved.display()
                            }
                        } else {
                            resolved.display()
                        };
                        return Err(TypeError::MissingTraitImpl {
                            ty: display_ty,
                            trait_name: trait_name.clone(),
                        });
                    }
                }
            }
        }

        // Post-solve: check pending method calls now that types are resolved
        self.check_pending_method_calls()?;

        // Post-method-call: re-check deferred HasTrait constraints.
        // Method call resolution (above) may have resolved type variables that were
        // Vars during the first deferred_has_trait pass. For example, fold's lambda
        // params get unified with List element types during check_pending_method_calls.
        for (ty, trait_name) in &self.deferred_has_trait.clone() {
            let resolved = self.env.apply_subst(ty);
            match &resolved {
                Type::Var(_) => {} // Still unresolved, skip
                Type::Function(_) => {
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: trait_name.clone(),
                    });
                }
                _ => {
                    if !self.env.implements(&resolved, trait_name) {
                        if resolved.has_any_type_var() {
                            if !self.env.definitely_not_implements(&resolved, trait_name) {
                                continue;
                            }
                        }
                        self.last_error_span = None;
                        let display_ty = if let Type::Named { name, args, .. } = &resolved {
                            if args.iter().any(|a| a.has_any_type_var()) {
                                name.rsplit('.').next().unwrap_or(name).to_string()
                            } else {
                                resolved.display()
                            }
                        } else {
                            resolved.display()
                        };
                        return Err(TypeError::MissingTraitImpl {
                            ty: display_ty,
                            trait_name: trait_name.clone(),
                        });
                    }
                }
            }
        }

        // Post-method-call: re-check deferred HasField constraints now that method calls
        // may have resolved more type variables (e.g., lambda parameter types unified
        // with list element types)
        // Process deferred HasField constraints in a loop - each pass may resolve
        // type vars that enable the next pass (e.g., o.inner.value needs inner resolved first)
        let mut max_passes = 5;
        loop {
            max_passes -= 1;
            if max_passes == 0 { break; }
            let deferred = std::mem::take(&mut self.deferred_has_field);
            if deferred.is_empty() { break; }
            let mut still_deferred = Vec::new();
            let mut made_progress = false;
        for (ty, field, expected_ty) in deferred {
            let resolved = self.env.apply_subst(&ty);
            if !matches!(&resolved, Type::Var(_)) {
                made_progress = true;
            }
            match &resolved {
                Type::Record(rec) => {
                    if let Some((_, actual_ty, _)) = rec.fields.iter().find(|(n, _, _)| n == &field) {
                        self.unify_types(actual_ty, &expected_ty)?;
                    } else {
                        return Err(TypeError::NoSuchField {
                            ty: resolved.display(),
                            field,
                        });
                    }
                }
                Type::Named { name, args } => {
                    if let Some(typedef) = self.env.lookup_type(name).cloned() {
                        match &typedef {
                            crate::TypeDef::Record { params, fields, .. } => {
                                if let Some((_, actual_ty, _)) = fields.iter().find(|(n, _, _)| n == &field) {
                                    let subst: HashMap<String, Type> = params.iter()
                                        .map(|p| p.name.clone())
                                        .zip(args.iter().cloned())
                                        .collect();
                                    let substituted_ty = Self::substitute_type_params(actual_ty, &subst);
                                    self.unify_types(&substituted_ty, &expected_ty)?;
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: resolved.display(),
                                        field,
                                    });
                                }
                            }
                            crate::TypeDef::Variant { params, constructors } => {
                                // Only allow field access on single-constructor variants
                                if constructors.len() == 1 {
                                    if let Some(Constructor::Named(_, fields)) = constructors.first() {
                                        if let Some((_, actual_ty)) = fields.iter().find(|(n, _)| n == &field) {
                                            let subst: HashMap<String, Type> = params.iter()
                                                .map(|p| p.name.clone())
                                                .zip(args.iter().cloned())
                                                .collect();
                                            let substituted_ty = Self::substitute_type_params(actual_ty, &subst);
                                            self.unify_types(&substituted_ty, &expected_ty)?;
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
                    } else {
                        return Err(TypeError::NoSuchField {
                            ty: resolved.display(),
                            field,
                        });
                    }
                }
                Type::Tuple(elems) => {
                    if let Ok(idx) = field.parse::<usize>() {
                        if idx < elems.len() {
                            self.unify_types(&elems[idx], &expected_ty)?;
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
                Type::List(_) => {
                    // Lists don't support field access (.0, .1, etc.) -
                    // that's only for tuples. Use list.get(idx) instead.
                    // Must match the solve() HasField handler behavior.
                    return Err(TypeError::NoSuchField {
                        ty: resolved.display(),
                        field,
                    });
                }
                Type::Var(_) => {
                    // Still unresolved - re-defer for another pass
                    still_deferred.push((ty, field, expected_ty));
                }
                _ => {
                    // Other types don't support field access
                    return Err(TypeError::NoSuchField {
                        ty: resolved.display(),
                        field,
                    });
                }
            }
        }
            // If we made progress or still have deferred constraints, continue
            if still_deferred.is_empty() {
                break;
            }
            if !made_progress {
                // No progress - these vars are truly unresolved, stop retrying
                // Preserve them for later inspection (e.g., encoding in function signatures)
                self.deferred_has_field = still_deferred;
                break;
            }
            self.deferred_has_field = still_deferred;
        } // end loop

        // Post-HasField: re-check deferred HasTrait constraints a third time.
        // The deferred_has_field loop above may have resolved type variables that
        // intermediate HasField results depend on. For example:
        //   getFirst(pairs) = { pair = pairs.head(); pair.0 ++ " items" }
        // The signature encodes HasField(0,b) on list element type and Concat on b.
        // After deferred_has_field resolves pair.0 → Int, the Concat bound on b (now Int)
        // becomes checkable. Without this third pass, the error goes undetected.
        for (ty, trait_name) in &self.deferred_has_trait.clone() {
            let resolved = self.env.apply_subst(ty);
            match &resolved {
                Type::Var(_) => {} // Still unresolved, skip
                Type::Function(_) => {
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: trait_name.clone(),
                    });
                }
                _ => {
                    if !self.env.implements(&resolved, trait_name) {
                        if resolved.has_any_type_var() {
                            if !self.env.definitely_not_implements(&resolved, trait_name) {
                                continue;
                            }
                        }
                        self.last_error_span = None;
                        let display_ty = if let Type::Named { name, args, .. } = &resolved {
                            if args.iter().any(|a| a.has_any_type_var()) {
                                name.rsplit('.').next().unwrap_or(name).to_string()
                            } else {
                                resolved.display()
                            }
                        } else {
                            resolved.display()
                        };
                        return Err(TypeError::MissingTraitImpl {
                            ty: display_ty,
                            trait_name: trait_name.clone(),
                        });
                    }
                }
            }
        }

        // Post-method-call: check length/len calls require types that VM supports
        // VM supports: List, String, Tuple, Float64Array, Int64Array
        // Note: Map and Set should use .size() instead
        for (arg_ty, _span) in std::mem::take(&mut self.deferred_length_checks) {
            let resolved = self.env.apply_subst(&arg_ty);
            let is_valid_for_length = match &resolved {
                Type::List(_) | Type::String | Type::Tuple(_) | Type::Array(_) => true,
                Type::Named { name, .. } => {
                    let short = name.rsplit('.').next().unwrap_or(name);
                    matches!(short, "Float64Array" | "Int64Array" | "Float32Array" | "Buffer")
                }
                Type::Var(_) => {
                    // Still unresolved - propagate as HasMethod("length") so the
                    // constraint reaches call sites through generic function signatures.
                    // Without this, getLen(x) = length(x) then getLen(42) would not
                    // catch that Int doesn't support length.
                    self.deferred_method_on_var.push((
                        resolved.clone(),
                        "length".to_string(),
                        vec![],
                        Type::Int,
                        Some(_span),
                    ));
                    true
                }
                _ => false,
            };
            if !is_valid_for_length {
                // Special error message for Map/Set to guide users
                let hint = match &resolved {
                    Type::Map(_, _) => " (use .size() for Map)",
                    Type::Set(_) => " (use .size() for Set)",
                    _ => "",
                };
                return Err(TypeError::Mismatch {
                    expected: format!("List, String, Tuple, or Array{}", hint),
                    found: resolved.display(),
                });
            }
        }

        // Pre-concat: check default parameter values against resolved param types.
        // This MUST run before deferred_concat_checks so that default types are
        // substituted into param vars, allowing concat to see concrete types.
        // e.g., process(items = [1,2,3]) = items ++ " done"
        // → items stays as Var with Concat bound, default is List[Int]
        // → substitute ?X = List[Int], then concat check sees List[Int] ++ String → error
        for (default_ty, param_ty, span) in std::mem::take(&mut self.deferred_default_param_checks) {
            let resolved_default = self.env.apply_subst(&default_ty);
            let resolved_param = self.env.apply_subst(&param_ty);
            if !resolved_default.has_any_type_var() && !resolved_param.has_any_type_var() {
                // Both fully concrete - simple equality check
                if resolved_default != resolved_param {
                    self.last_error_span = Some(span);
                    return Err(TypeError::Mismatch {
                        expected: resolved_param.display(),
                        found: resolved_default.display(),
                    });
                }
            } else if !resolved_default.has_any_type_var() {
                // Default is concrete but param still has unresolved vars.
                // Check if param's trait bounds are satisfied by the default type.
                if let Type::Var(var_id) = &resolved_param {
                    let bounds: Vec<String> = self.get_trait_bounds(*var_id)
                        .into_iter().cloned().collect();
                    for bound in &bounds {
                        if !self.env.implements(&resolved_default, bound) {
                            self.last_error_span = Some(span);
                            return Err(TypeError::Mismatch {
                                expected: format!("type implementing {}", bound),
                                found: resolved_default.display(),
                            });
                        }
                    }
                    // Substitute the default type into the param var so that downstream
                    // checks (concat, etc.) see concrete types instead of Var.
                    self.env.substitution.insert(*var_id, resolved_default);
                }
            } else if resolved_default.has_any_type_var() {
                // Default has unresolved type vars (e.g., [] → List[?T], None → Option[?T],
                // x => x → ?A -> ?B). Even though inner types are unknown, the OUTER
                // constructor may be known to never implement certain traits.
                // e.g., List[?T] never implements Num, Function never implements anything.
                if let Type::Var(var_id) = &resolved_param {
                    let bounds: Vec<String> = self.get_trait_bounds(*var_id)
                        .into_iter().cloned().collect();
                    for bound in &bounds {
                        if self.env.definitely_not_implements(&resolved_default, bound) {
                            self.last_error_span = Some(span);
                            return Err(TypeError::Mismatch {
                                expected: format!("type implementing {}", bound),
                                found: resolved_default.display(),
                            });
                        }
                    }
                } else if !resolved_param.has_any_type_var() {
                    // Param is concrete but default has type vars.
                    // Outer type constructors must match for the types to be compatible.
                    // e.g., List[?Y] vs Int → mismatch, List[?Y] vs List[Int] → OK
                    let mismatch = match (&resolved_default, &resolved_param) {
                        (Type::List(_), Type::List(_)) => false,
                        (Type::List(_), _) => true,
                        (Type::Tuple(_), Type::Tuple(_)) => false,
                        (Type::Tuple(_), _) => true,
                        (Type::Function(_), Type::Function(_)) => false,
                        (Type::Function(_), _) => true,
                        (Type::Named { name: a, .. }, Type::Named { name: b, .. }) => a != b,
                        (Type::Named { .. }, _) => true,
                        _ => false, // Can't determine, skip
                    };
                    if mismatch {
                        self.last_error_span = Some(span);
                        return Err(TypeError::Mismatch {
                            expected: resolved_param.display(),
                            found: resolved_default.display(),
                        });
                    }
                }
            }
        }

        // Post-method-call: check deferred concat (++) operations
        // ++ only works on String ++ String or List[T] ++ List[T].
        // When operands were type variables (e.g., from deferred field access),
        // we deferred the check. Now that types are resolved, verify they're valid.
        for (left_ty, right_ty, _span) in std::mem::take(&mut self.deferred_concat_checks) {
            let resolved_left = self.env.apply_subst(&left_ty);
            let resolved_right = self.env.apply_subst(&right_ty);
            // Check each operand - must be String, List, or still-unresolved Var
            for resolved in [&resolved_left, &resolved_right] {
                let is_valid = matches!(
                    resolved,
                    Type::String | Type::List(_) | Type::Var(_)
                );
                if !is_valid {
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: "Concat".to_string(),
                    });
                }
            }
            // If both resolved to concrete types, verify they're compatible
            match (&resolved_left, &resolved_right) {
                (Type::List(_), Type::String) | (Type::String, Type::List(_)) => {
                    return Err(TypeError::Mismatch {
                        expected: resolved_left.display(),
                        found: resolved_right.display(),
                    });
                }
                (Type::List(a), Type::List(b)) => {
                    let _ = self.unify_types(a, b);
                }
                (Type::String, Type::String) => {}
                _ => {
                    // At least one is still Var or they match - OK
                }
            }
        }

        // Post-solve: check function call return types that resolved to Set[X] or Map[K, V].
        // Set/Map require their element/key types to implement Hash+Eq.
        // We couldn't check at call time because unification was deferred.
        // Check Hash (stricter than Eq) - Lists implement Eq but not Hash,
        // so checking Hash catches Set[List[Int]] which would fail at runtime.
        for (ret_ty, span) in std::mem::take(&mut self.deferred_collection_ret_checks) {
            let resolved = self.env.apply_subst(&ret_ty);
            match &resolved {
                Type::Set(elem) => {
                    let resolved_elem = self.env.apply_subst(elem);
                    // Check Hash first (stricter), then Eq as fallback
                    if self.env.definitely_not_implements(&resolved_elem, "Hash") {
                        self.last_error_span = Some(span);
                        return Err(TypeError::MissingTraitImpl {
                            ty: resolved_elem.display(),
                            trait_name: "Eq".to_string(), // User-facing: say Eq (Hash is internal)
                        });
                    }
                    if self.env.definitely_not_implements(&resolved_elem, "Eq") {
                        self.last_error_span = Some(span);
                        return Err(TypeError::MissingTraitImpl {
                            ty: resolved_elem.display(),
                            trait_name: "Eq".to_string(),
                        });
                    }
                }
                Type::Map(key, _) => {
                    let resolved_key = self.env.apply_subst(key);
                    if self.env.definitely_not_implements(&resolved_key, "Hash") {
                        self.last_error_span = Some(span);
                        return Err(TypeError::MissingTraitImpl {
                            ty: resolved_key.display(),
                            trait_name: "Eq".to_string(),
                        });
                    }
                    if self.env.definitely_not_implements(&resolved_key, "Eq") {
                        self.last_error_span = Some(span);
                        return Err(TypeError::MissingTraitImpl {
                            ty: resolved_key.display(),
                            trait_name: "Eq".to_string(),
                        });
                    }
                }
                _ => {}
            }
        }

        // Post-method-call: re-check trait bounds on type variables that may have been
        // resolved by check_pending_method_calls. For example, xs.zip(ys).map(pair => pair + 1)
        // - during solve(), ?pair has a Num bound but is still unresolved
        // - check_pending_method_calls resolves zip, so ?pair becomes Tuple(Int, String)
        // - now we need to verify Tuple(Int, String) doesn't satisfy Num
        for (&var_id, bounds) in &self.trait_bounds.clone() {
            let resolved = self.env.apply_subst(&Type::Var(var_id));
            // Skip if still contains unresolved type variables - can't definitively
            // check trait compliance when inner types are unknown
            if self.contains_unresolved_var(&resolved) {
                continue;
            }
            match &resolved {
                Type::Var(_) => {} // Still unresolved, skip
                Type::List(_) | Type::Map(_, _) | Type::Set(_) | Type::Array(_) | Type::Tuple(_) => {
                    // Container and tuple types never implement Num, Ord, Concat
                    for bound in bounds {
                        // Skip HasField/HasMethod - handled by dedicated constraint handlers
                        if bound.starts_with("HasField(") || bound.starts_with("HasMethod(") {
                            continue;
                        }
                        if !self.env.implements(&resolved, bound) {
                            return Err(TypeError::MissingTraitImpl {
                                ty: resolved.display(),
                                trait_name: bound.clone(),
                            });
                        }
                    }
                }
                Type::Named { name, .. } => {
                    // Only check container types (Option, Result) - they clearly
                    // can't implement Num/Ord. Other Named types might have Num bounds
                    // from merged trait_bounds (variable unification) which are false positives.
                    let short = name.rsplit('.').next().unwrap_or(name);
                    if ["Option", "Result"].contains(&short) {
                        for bound in bounds {
                            if bound.starts_with("HasField(") || bound.starts_with("HasMethod(") {
                                continue;
                            }
                            if !self.env.implements(&resolved, bound) {
                                return Err(TypeError::MissingTraitImpl {
                                    ty: resolved.display(),
                                    trait_name: bound.clone(),
                                });
                            }
                        }
                    }
                }
                Type::Function(_) => {
                    // Function types never implement any standard trait (Eq, Ord, Num, Concat, etc.)
                    // This catches user-defined generic functions with trait bounds
                    // e.g., equals[T: Eq](a: T, b: T) = a == b called with function arguments
                    for bound in bounds {
                        if bound.starts_with("HasField(") || bound.starts_with("HasMethod(") {
                            continue;
                        }
                        if !self.env.implements(&resolved, bound) {
                            return Err(TypeError::MissingTraitImpl {
                                ty: resolved.display(),
                                trait_name: bound.clone(),
                            });
                        }
                    }
                }
                Type::String | Type::Bool | Type::Char | Type::Unit => {
                    // Check Num and Ord bounds on primitive types.
                    // - Bool/Char/Unit don't implement Num or Ord
                    // - String doesn't implement Num
                    // Only check Num/Ord (not Eq/Show which are auto-derived and cause
                    // false positives from merged trait_bounds during variable unification).
                    for bound in bounds {
                        if bound.starts_with("HasField(") || bound.starts_with("HasMethod(") {
                            continue;
                        }
                        if matches!(bound.as_str(), "Num" | "Ord") && !self.env.implements(&resolved, bound) {
                            return Err(TypeError::MissingTraitImpl {
                                ty: resolved.display(),
                                trait_name: bound.clone(),
                            });
                        }
                    }
                }
                _ => {} // Numeric primitives (Int, Float, etc.) implement Num
            }
        }

        // Process deferred method existence checks from HasMethod constraints.
        // These come from instantiate_function when a generic function has a HasMethod
        // bound (e.g., applyLen(x) = x.length() → HasMethod(length) a => a -> b).
        // After solve(), the type var should be resolved; check if the method exists.
        for (ty, method_name, span) in std::mem::take(&mut self.deferred_method_existence_checks) {
            let resolved = self.env.apply_subst(&ty);
            // For unresolved type vars, re-record as deferred_method_on_var so that
            // try_hm_inference can propagate the HasMethod constraint into this
            // function's signature. This enables transitive propagation:
            // g(x) = x.length() → HasMethod(length); f(x) = g(x) → also HasMethod(length)
            if matches!(&resolved, Type::Var(_)) {
                let dummy_ret = self.fresh();
                self.deferred_method_on_var.push((
                    ty,
                    method_name,
                    vec![],
                    dummy_ret,
                    span,
                ));
                continue;
            }

            // Types that definitely don't support collection/string methods
            let has_no_methods = matches!(&resolved,
                Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                Type::Float | Type::Float32 | Type::Float64 |
                Type::Bool | Type::Char | Type::BigInt | Type::Decimal |
                Type::Unit | Type::Function(_)
            );

            if has_no_methods {
                let type_display = resolved.display();
                self.last_error_span = span;
                return Err(TypeError::UndefinedMethod {
                    method: method_name,
                    receiver_type: type_display,
                });
            }
        }

        // Verify post-solve invariants (debug mode only)
        self.verify_post_solve_invariants();

        // Apply substitution to all stored expression types
        self.finalize_expr_types();

        // Verify post-finalize invariants (debug mode only)
        self.verify_post_finalize_invariants();

        // Mark solve as successfully completed
        self.solve_completed = true;

        Ok(())
    }

    /// Check if solve() completed normally (all constraints processed).
    /// Returns false if solve() hit the MAX_ITERATIONS limit.
    pub fn solve_completed(&self) -> bool {
        self.solve_completed
    }

    /// Apply the current substitution to all stored expression types.
    /// This should be called after solve() to get the resolved types.
    /// Also resolves any remaining TypeParams using type_param_mappings.
    /// Tracks any expressions with unresolved type parameters in `unresolved_type_params`.
    pub fn finalize_expr_types(&mut self) {
        self.unresolved_type_params.clear();

        let mut resolved_types = HashMap::new();
        let mut unresolved = Vec::new();

        for (span, ty) in &self.expr_types {
            let resolved_ty = self.apply_full_subst(ty);

            // Track expressions that still have unresolved TypeParams
            if self.contains_type_param(&resolved_ty) {
                if let Some(param_name) = self.extract_type_param_name(&resolved_ty) {
                    unresolved.push((*span, resolved_ty.clone(), param_name));
                }
            }

            resolved_types.insert(*span, resolved_ty);
        }

        self.expr_types = resolved_types;
        self.unresolved_type_params = unresolved;
    }

    /// Extract the first TypeParam name from a type (for error reporting).
    fn extract_type_param_name(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::TypeParam(name) => Some(name.clone()),
            Type::Var(id) => {
                if let Some(resolved) = self.env.substitution.get(id) {
                    self.extract_type_param_name(resolved)
                } else {
                    None
                }
            }
            Type::List(elem) => self.extract_type_param_name(elem),
            Type::Array(elem) => self.extract_type_param_name(elem),
            Type::Map(k, v) => self.extract_type_param_name(k).or_else(|| self.extract_type_param_name(v)),
            Type::Set(elem) => self.extract_type_param_name(elem),
            Type::Tuple(elems) => elems.iter().find_map(|e| self.extract_type_param_name(e)),
            Type::Function(ft) => {
                ft.params.iter().find_map(|p| self.extract_type_param_name(p))
                    .or_else(|| self.extract_type_param_name(&ft.ret))
            }
            Type::Named { args, .. } => args.iter().find_map(|a| self.extract_type_param_name(a)),
            Type::IO(inner) => self.extract_type_param_name(inner),
            Type::Record(rec) => rec.fields.iter().find_map(|(_, t, _)| self.extract_type_param_name(t)),
            Type::Variant(var) => var.constructors.iter().find_map(|c| match c {
                Constructor::Unit(_) => None,
                Constructor::Positional(_, types) => types.iter().find_map(|t| self.extract_type_param_name(t)),
                Constructor::Named(_, fields) => fields.iter().find_map(|(_, t)| self.extract_type_param_name(t)),
            }),
            _ => None,
        }
    }

    /// Generate warning messages for expressions with unresolved type parameters.
    /// Returns a list of (span, message) pairs for each unresolved expression.
    pub fn unresolved_type_param_warnings(&self) -> Vec<(Span, String)> {
        self.unresolved_type_params.iter().map(|(span, ty, param_name)| {
            let type_display = ty.display();
            let msg = format!(
                "Type parameter '{}' could not be inferred (type is '{}'). Consider adding a type annotation.",
                param_name, type_display
            );
            (*span, msg)
        }).collect()
    }

    /// Check if there are any unresolved type parameters.
    pub fn has_unresolved_type_params(&self) -> bool {
        !self.unresolved_type_params.is_empty()
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

    /// For a deferred method call on a Var (generic param), try to determine the
    /// method's return type by checking all known implementations. If ALL implementations
    /// return the same concrete type (e.g., isEmpty → Bool, length → Int), we can
    /// constrain the return type even when the receiver type is unknown.
    /// This catches errors like `f(x) = x.isEmpty() + 1` where isEmpty returns Bool
    /// but the result is used as a number.
    fn try_constrain_deferred_method_return(
        &mut self,
        method_name: &str,
        ret_ty: &Type,
        receiver_ty: &Type,
        span: Option<Span>,
    ) -> Result<(), TypeError> {
        // First, check if the receiver has trait bounds that define this method.
        // If so, use the trait's return type (which is authoritative) and skip
        // the BUILTINS lookup (which could find unrelated methods with the same name,
        // e.g., user-defined `toInt` vs BUILTINS `String.toInt`).
        let resolved_receiver = self.env.apply_subst(receiver_ty);

        // If receiver is a TypeParam (e.g., T in `f[T: SomeTrait](x: T)`),
        // skip the BUILTINS lookup entirely. TypeParam method resolution is handled
        // by the trait bounds check further in check_pending_method_calls.
        if matches!(&resolved_receiver, Type::TypeParam(_)) {
            return Ok(());
        }

        if let Type::Var(var_id) = &resolved_receiver {
            let bounds = self.get_trait_bounds(*var_id);
            for trait_name in &bounds {
                if let Some(trait_def) = self.env.traits.get(*trait_name).cloned() {
                    let method_opt = trait_def.required.iter()
                        .chain(trait_def.defaults.iter())
                        .find(|m| m.name == method_name);
                    if let Some(method) = method_opt {
                        // Found method in trait - use trait's return type
                        let mut trait_ret = method.ret.clone();
                        // Replace Self with the receiver type
                        if let Type::TypeParam(ref name) = trait_ret {
                            if name == "Self" {
                                trait_ret = resolved_receiver.clone();
                            }
                        }
                        if !trait_ret.has_any_type_var() && !matches!(trait_ret, Type::TypeParam(_)) {
                            if let Err(_) = self.unify_types(ret_ty, &trait_ret) {
                                let resolved_call_ret = self.env.apply_subst(ret_ty);
                                self.last_error_span = span;
                                return Err(TypeError::Mismatch {
                                    expected: format!("{} (return type of .{}())", trait_ret.display(), method_name),
                                    found: format!("{}", resolved_call_ret.display()),
                                });
                            }
                        }
                        // Trait method found - don't fall through to BUILTINS lookup
                        return Ok(());
                    }
                }
            }
        }

        // No trait defines this method. Look up all known BUILTINS implementations.
        // Only constrain if the method is found on at least 2 different types to avoid
        // false positives from name collisions (e.g., user-defined `toInt` vs String.toInt).
        let prefixes = [
            "List", "String", "Map", "Set",
            "stdlib.list", "stdlib.string", "stdlib.map", "stdlib.set",
        ];
        let mut concrete_return_types: Vec<Type> = Vec::new();
        let mut source_count = 0u32;

        for prefix in &prefixes {
            let qualified = format!("{}.{}", prefix, method_name);
            if let Some(ft) = self.env.functions.get(&qualified) {
                let ret = &*ft.ret;
                if !ret.has_any_type_var() {
                    source_count += 1;
                    if !concrete_return_types.contains(ret) {
                        concrete_return_types.push(ret.clone());
                    }
                }
            }
        }

        // Also check unqualified entry (generic BUILTINS)
        if let Some(ft) = self.env.functions.get(method_name) {
            let ret = &*ft.ret;
            if !ret.has_any_type_var() {
                source_count += 1;
                if !concrete_return_types.contains(ret) {
                    concrete_return_types.push(ret.clone());
                }
            }
        }

        // Only constrain if found on 2+ sources AND all agree on the return type.
        // This avoids false positives from single-type methods that might collide
        // with user-defined trait methods (e.g., String.toInt vs custom ToInt trait).
        if source_count >= 2
            && !concrete_return_types.is_empty()
            && concrete_return_types.iter().all(|r| r == &concrete_return_types[0])
        {
            let common_ret = concrete_return_types[0].clone();
            let resolved_call_ret = self.env.apply_subst(ret_ty);

            if let Err(_) = self.unify_types(ret_ty, &common_ret) {
                self.last_error_span = span;
                return Err(TypeError::Mismatch {
                    expected: format!("{} (return type of .{}())", common_ret.display(), method_name),
                    found: format!("{}", resolved_call_ret.display()),
                });
            }
        }

        Ok(())
    }

    /// Check pending method calls after constraint solving.
    /// This enables UFCS type checking for cases where the receiver type
    /// wasn't known during initial inference (e.g., status from Server.bind).
    fn check_pending_method_calls(&mut self) -> Result<(), TypeError> {
        // Take ownership of pending calls to avoid borrow issues
        let mut pending = std::mem::take(&mut self.pending_method_calls);

        // Iteratively process pending calls. Inner calls may have unresolved
        // receiver types that become resolved after outer calls are processed.
        // Example: xs.map(inner => inner.map(x => ...)) - the inner map's
        // receiver type is only known after the outer map is checked.
        let max_iterations = 5;
        for iteration in 0..max_iterations {
            let mut deferred = Vec::new();
            let mut made_progress = false;

            for call in pending {
                // Resolve the receiver type now that constraints are solved
                let resolved_receiver = self.env.apply_subst(&call.receiver_ty);

                // Try to get the type name
                let type_name_opt = self.get_type_name(&resolved_receiver);

                // If receiver is still a type variable, defer for later iteration
                // (but not on the final iteration - let it fall through to fallback handling).
                // Exception: if the method is a known list method, don't defer - we can
                // infer the receiver type from the method name (handled in the else branch).
                if type_name_opt.is_none() && matches!(&resolved_receiver, Type::Var(_))
                    && iteration < max_iterations - 1
                {
                    // Truly list-only methods that can be used to infer receiver type.
                    // Excludes methods shared with String/Map/Set (length, len, contains,
                    // isEmpty, get, set, concat, empty).
                    let can_infer_from_method = matches!(call.method_name.as_str(),
                        "map" | "filter" | "fold" | "flatMap" | "any" | "all" | "find" |
                        "sort" | "sortBy" | "head" | "tail" | "init" | "last" |
                        "reverse" | "sum" | "product" | "zip" | "unzip" | "take" | "drop" |
                        "unique" | "flatten" | "position" | "indexOf" |
                        "push" | "pop" | "nth" | "slice" |
                        "scanl" | "foldl" | "foldr" | "enumerate" | "intersperse" |
                        "spanList" | "groupBy" | "transpose" | "pairwise" | "isSorted" |
                        "isSortedBy" | "maximum" | "minimum" | "takeWhile" | "dropWhile" |
                        "partition" | "zipWith"
                    );
                    if !can_infer_from_method {
                        // Before deferring, try to constrain the return type from known
                        // method signatures. This catches errors like `f(x) = x.isEmpty() + 1`
                        // where isEmpty returns Bool but the result is used as Int.
                        self.try_constrain_deferred_method_return(
                            &call.method_name, &call.ret_ty, &call.receiver_ty, call.span,
                        )?;
                        deferred.push(call);
                        continue;
                    }
                }

                // If receiver is still a Var but method is list-only, infer receiver as List.
                // This ensures the method call gets processed (lookup, param/return unification)
                // instead of being silently dropped when type_name_opt is None.
                let type_name_opt = if type_name_opt.is_none() && matches!(&resolved_receiver, Type::Var(_)) {
                    let can_infer_from_method = matches!(call.method_name.as_str(),
                        "map" | "filter" | "fold" | "flatMap" | "any" | "all" | "find" |
                        "sort" | "sortBy" | "head" | "tail" | "init" | "last" |
                        "reverse" | "sum" | "product" | "zip" | "unzip" | "take" | "drop" |
                        "unique" | "flatten" | "position" | "indexOf" |
                        "push" | "pop" | "nth" | "slice" |
                        "scanl" | "foldl" | "foldr" | "enumerate" | "intersperse" |
                        "spanList" | "groupBy" | "transpose" | "pairwise" | "isSorted" |
                        "isSortedBy" | "maximum" | "minimum" | "takeWhile" | "dropWhile" |
                        "partition" | "zipWith"
                    );
                    if can_infer_from_method {
                        // Unify receiver with List[?X] so type info flows properly
                        let elem = self.fresh();
                        let list_ty = Type::List(Box::new(elem));
                        let _ = self.unify_types(&resolved_receiver, &list_ty);
                        Some("List".to_string())
                    } else {
                        type_name_opt
                    }
                } else {
                    type_name_opt
                };

                if let Some(type_name) = type_name_opt {
                let qualified_name = format!("{}.{}", type_name, call.method_name);

                // Look up the function by qualified name first
                let arity = call.arg_types.len();
                let fn_type_opt = self.env.functions.get(&qualified_name).cloned().or_else(|| {
                    // Try stdlib qualified name first (has real types from source),
                    // then fall back to BUILTINS unqualified name (generic types).
                    let stdlib_module = match type_name.as_str() {
                        "List" => Some("list"),
                        "Map" => Some("map"),
                        "Set" => Some("set"),
                        "String" => Some("string"),
                        _ => None,
                    };
                    // Try stdlib lookup: stdlib.{module}.{method} with arity
                    if let Some(module) = stdlib_module {
                        let stdlib_name = format!("stdlib.{}.{}", module, call.method_name);
                        if let Some(ft) = self.env.lookup_function_with_arity(&stdlib_name, arity) {
                            return Some(ft.clone());
                        }
                    }
                    // Fallback: try unqualified name (finds BUILTINS entries)
                    // Note: Also accept Type::Var as first param - generic BUILTINS functions
                    // like `length: a -> Int` have type variables, not concrete types.
                    match type_name.as_str() {
                        "List" => {
                            self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                matches!(ft.params.first(), Some(Type::List(_)) | Some(Type::Var(_)))
                            })
                        }
                        "Map" => self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                            matches!(ft.params.first(), Some(Type::Map(_, _)) | Some(Type::Var(_)))
                        }),
                        "Set" => self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                            matches!(ft.params.first(), Some(Type::Set(_)) | Some(Type::Var(_)))
                        }),
                        "String" => self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                            matches!(ft.params.first(), Some(Type::String) | Some(Type::Var(_)))
                        }),
                        "Option" => {
                            // Option methods are registered as optXxx in stdlib
                            // Support both shorthand (.map) and direct (.optMap) names
                            let opt_name = match call.method_name.as_str() {
                                "map" | "optMap" => Some("optMap"),
                                "flatMap" | "optFlatMap" => Some("optFlatMap"),
                                "unwrap" | "optUnwrap" => Some("optUnwrap"),
                                "unwrapOr" | "optUnwrapOr" => Some("optUnwrapOr"),
                                "isSome" | "optIsSome" => Some("optIsSome"),
                                "isNone" | "optIsNone" => Some("optIsNone"),
                                _ => None,
                            };
                            opt_name.and_then(|name| self.env.functions.get(name).cloned())
                        }
                        "Result" => {
                            // Result methods are registered as resXxx in stdlib
                            // Support both shorthand (.map) and direct (.resMap) names
                            let res_name = match call.method_name.as_str() {
                                "map" | "resMap" => Some("resMap"),
                                "mapErr" | "resMapErr" => Some("resMapErr"),
                                "flatMap" | "resFlatMap" => Some("resFlatMap"),
                                "unwrap" | "resUnwrap" => Some("resUnwrap"),
                                "unwrapOr" | "resUnwrapOr" => Some("resUnwrapOr"),
                                "isOk" | "resIsOk" => Some("resIsOk"),
                                "isErr" | "resIsErr" => Some("resIsErr"),
                                "toOption" | "resToOption" => Some("resToOption"),
                                _ => None,
                            };
                            res_name.and_then(|name| self.env.functions.get(name).cloned())
                        }
                        _ => None,
                    }
                });

                // Cross-type UFCS lookup: when {ReceiverType}.{method} is not found,
                // try other type-prefixed entries (e.g., String.join when receiver is List).
                // This is the general UFCS pattern: methods may be registered under a
                // different type prefix but still accept the receiver type as first param.
                let fn_type_opt = if fn_type_opt.is_some() {
                    fn_type_opt
                } else {
                    let resolved_receiver = self.env.apply_subst(&call.receiver_ty);
                    let cross_prefixes = ["List", "String", "Map", "Set"];
                    let mut cross_found = None;
                    for prefix in &cross_prefixes {
                        if *prefix == type_name.as_str() { continue; }
                        let cross_name = format!("{}.{}", prefix, call.method_name);
                        if let Some(ft) = self.env.functions.get(&cross_name) {
                            // Check if first param structurally matches receiver type
                            if let Some(first_param) = ft.params.first() {
                                let compatible = match (&resolved_receiver, first_param) {
                                    (Type::List(_), Type::List(_)) => true,
                                    (Type::Map(_, _), Type::Map(_, _)) => true,
                                    (Type::Set(_), Type::Set(_)) => true,
                                    (Type::String, Type::String) => true,
                                    (_, Type::Var(_)) => true,
                                    _ => false,
                                };
                                if compatible {
                                    cross_found = Some(ft.clone());
                                    break;
                                }
                            }
                        }
                    }
                    cross_found
                };

                if let Some(fn_type) = fn_type_opt {
                    let func_ty = self.instantiate_function(&fn_type);
                    if let Type::Function(ft) = func_ty {
                        // Check arity (accounting for optional parameters)
                        // Note: arg_types includes receiver as first element, matching UFCS params
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

                        // Pre-check: methods that require numeric element types
                        // This must run BEFORE unification to produce MissingTraitImpl errors
                        // (which aren't filtered) instead of UnificationFailed errors (which
                        // can be incorrectly filtered by the try/catch mismatch heuristic).
                        if matches!(call.method_name.as_str(), "sum" | "product") {
                            let resolved_recv = self.env.apply_subst(&call.receiver_ty);
                            if let Type::List(elem) = &resolved_recv {
                                let resolved_elem = self.env.apply_subst(elem);
                                // Check if element type is definitely non-numeric.
                                // Numeric types: Int*, UInt*, Float*, BigInt, Decimal
                                // Type variables (Var/TypeParam) might be numeric, so allow them.
                                let is_non_numeric = matches!(&resolved_elem,
                                    Type::String | Type::Bool | Type::Char | Type::Unit | Type::Never |
                                    Type::Tuple(_) | Type::List(_) | Type::Map(_, _) | Type::Set(_) |
                                    Type::Array(_) | Type::Record(_) | Type::Function(_) |
                                    Type::Named { .. });
                                if is_non_numeric {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved_elem.display(),
                                        trait_name: "Num".to_string(),
                                    });
                                }
                            }
                        }

                        // Pre-check: sort/maximum/minimum/isSorted require Ord on element types
                        if matches!(call.method_name.as_str(), "sort" | "maximum" | "minimum" | "isSorted") {
                            let resolved_recv = self.env.apply_subst(&call.receiver_ty);
                            if let Type::List(elem) = &resolved_recv {
                                let resolved_elem = self.env.apply_subst(elem);
                                // Types that don't implement Ord: tuples, lists, maps, sets,
                                // records, functions, Bool, user-defined types
                                let is_non_orderable = matches!(&resolved_elem,
                                    Type::Bool | Type::Unit | Type::Never |
                                    Type::Tuple(_) | Type::List(_) | Type::Map(_, _) | Type::Set(_) |
                                    Type::Array(_) | Type::Record(_) | Type::Function(_) |
                                    Type::Named { .. });
                                if is_non_orderable {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved_elem.display(),
                                        trait_name: "Ord".to_string(),
                                    });
                                }
                            }
                        }

                        // Pre-check: contains/indexOf/unique require Eq on element types
                        if matches!(call.method_name.as_str(), "contains" | "indexOf" | "unique") {
                            let resolved_recv = self.env.apply_subst(&call.receiver_ty);
                            if let Type::List(elem) = &resolved_recv {
                                let resolved_elem = self.env.apply_subst(elem);
                                if self.env.definitely_not_implements(&resolved_elem, "Eq") {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved_elem.display(),
                                        trait_name: "Eq".to_string(),
                                    });
                                }
                            }
                        }

                        // Pre-check: groupBy key function return type must implement Eq
                        // e.g., [1,2,3].groupBy(x => (x) => x) should fail because
                        // function types don't implement Eq
                        if call.method_name == "groupBy" {
                            // arg_types[0] is receiver, arg_types[1] is key function
                            if let Some(key_fn_ty) = call.arg_types.get(1) {
                                let resolved_key_fn = self.env.apply_subst(key_fn_ty);
                                if let Type::Function(ft) = &resolved_key_fn {
                                    let key_ty = self.env.apply_subst(&ft.ret);
                                    if self.env.definitely_not_implements(&key_ty, "Eq") {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::MissingTraitImpl {
                                            ty: key_ty.display(),
                                            trait_name: "Eq".to_string(),
                                        });
                                    }
                                }
                            }
                        }

                        // Pre-check: unzip requires list of tuples
                        if call.method_name == "unzip" {
                            let resolved_recv = self.env.apply_subst(&call.receiver_ty);
                            if let Type::List(elem) = &resolved_recv {
                                let resolved_elem = self.env.apply_subst(elem);
                                if !matches!(&resolved_elem, Type::Tuple(_) | Type::Var(_) | Type::TypeParam(_)) {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::Mismatch {
                                        expected: "List of tuples".to_string(),
                                        found: format!("List[{}]", resolved_elem.display()),
                                    });
                                }
                            }
                        }

                        // Pre-check: map/filter lambda param must match list element type
                        // The lambda body may have eagerly unified the param with a wrong type
                        // (e.g., p + 1 unifies p=Var with Int, but list has Tuple elements).
                        // Detect this mismatch here with a MissingTraitImpl error that
                        // bypasses the tuple error filter in compile.rs.
                        if matches!(call.method_name.as_str(), "map" | "filter") {
                            let resolved_recv = self.env.apply_subst(&call.receiver_ty);
                            if let Type::List(elem) = &resolved_recv {
                                let resolved_elem = self.env.apply_subst(elem);
                                // Check if element is a non-numeric type
                                let is_non_numeric = matches!(&resolved_elem,
                                    Type::Tuple(_) | Type::Record(_) | Type::Named { .. });
                                if is_non_numeric {
                                    // Check if lambda param was unified to a different (numeric) type
                                    if let Some(arg_ty) = call.arg_types.get(1) {
                                        let resolved_arg = self.env.apply_subst(arg_ty);
                                        if let Type::Function(lambda_fn) = &resolved_arg {
                                            if let Some(lambda_param) = lambda_fn.params.first() {
                                                let resolved_lambda_param = self.env.apply_subst(lambda_param);
                                                // If lambda param resolved to a primitive but element is
                                                // Tuple/Record, this means arithmetic was applied to non-numeric type
                                                if matches!(&resolved_lambda_param,
                                                    Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                                    Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                                    Type::Float | Type::Float32 | Type::Float64 |
                                                    Type::BigInt | Type::Decimal)
                                                {
                                                    self.last_error_span = call.span;
                                                    return Err(TypeError::MissingTraitImpl {
                                                        ty: resolved_elem.display(),
                                                        trait_name: "Num".to_string(),
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Reorder arguments if there are named args that need to be
                        // matched to specific parameter positions (not just positionally).
                        let reordered_args: Vec<Type> = if !call.named_args.is_empty() {
                            // Look up param names to map named args to correct positions
                            let param_names_opt = self.env.function_param_names.get(&call.method_name)
                                .or_else(|| {
                                    // Try with qualified name patterns
                                    let qualified = format!("{}/", call.method_name);
                                    self.env.function_param_names.get(&qualified)
                                });
                            if let Some(param_names) = param_names_opt {
                                let mut result: Vec<Option<Type>> = vec![None; ft.params.len()];
                                // Named arg indices in call.arg_types
                                let named_indices: std::collections::HashSet<usize> =
                                    call.named_args.iter().map(|(_, idx)| *idx).collect();
                                // Place receiver (always first)
                                if let Some(recv) = call.arg_types.first() {
                                    result[0] = Some(recv.clone());
                                }
                                // Place positional args (skip receiver at index 0, skip named args)
                                let mut param_idx = 1;
                                for (i, arg_ty) in call.arg_types.iter().enumerate().skip(1) {
                                    if named_indices.contains(&i) {
                                        continue;
                                    }
                                    while param_idx < result.len() && result[param_idx].is_some() {
                                        param_idx += 1;
                                    }
                                    if param_idx < result.len() {
                                        result[param_idx] = Some(arg_ty.clone());
                                        param_idx += 1;
                                    }
                                }
                                // Place named args at their correct parameter positions
                                for (name, arg_idx) in &call.named_args {
                                    if let Some(pos) = param_names.iter().position(|n| n == name) {
                                        if pos < result.len() {
                                            if let Some(arg_ty) = call.arg_types.get(*arg_idx) {
                                                result[pos] = Some(arg_ty.clone());
                                            }
                                        }
                                    }
                                }
                                // Fill remaining with fresh type vars (for defaulted params)
                                result.into_iter().map(|opt| opt.unwrap_or_else(|| self.fresh())).collect()
                            } else {
                                // No param names available - fall back to positional matching
                                call.arg_types.clone()
                            }
                        } else {
                            call.arg_types.clone()
                        };

                        // Resolve arg types and check against params
                        for (param_ty, arg_ty) in ft.params.iter().zip(reordered_args.iter()) {
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
                                        // Unify return types - catch structural mismatches
                                        match self.unify_types(&actual_fn.ret, &expected_ret) {
                                            Ok(()) => {}
                                            Err(TypeError::UnificationFailed(ref a, ref b)) => {
                                                let resolved_ret = self.env.apply_subst(&actual_fn.ret);
                                                let resolved_expected_ret = self.env.apply_subst(&expected_ret);
                                                let ret_is_simple = matches!(&resolved_ret,
                                                    Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                                    Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                                    Type::Float | Type::Float32 | Type::Float64 |
                                                    Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                                    Type::Unit);
                                                let expected_is_wrapper = matches!(&resolved_expected_ret,
                                                    Type::Named { .. } | Type::List(_) | Type::Set(_) |
                                                    Type::Map(_, _) | Type::Tuple(_));
                                                if ret_is_simple && expected_is_wrapper {
                                                    self.last_error_span = call.span;
                                                    return Err(TypeError::Mismatch {
                                                        expected: resolved_expected_ret.display(),
                                                        found: resolved_ret.display(),
                                                    });
                                                }
                                                // Also handle wrapper vs simple (e.g., List vs Int)
                                                let ret_is_wrapper = matches!(&resolved_ret,
                                                    Type::Named { .. } | Type::List(_) | Type::Set(_) |
                                                    Type::Map(_, _) | Type::Tuple(_));
                                                let expected_is_simple = matches!(&resolved_expected_ret,
                                                    Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                                    Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                                    Type::Float | Type::Float32 | Type::Float64 |
                                                    Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                                    Type::Unit);
                                                if ret_is_wrapper && expected_is_simple {
                                                    self.last_error_span = call.span;
                                                    return Err(TypeError::Mismatch {
                                                        expected: resolved_expected_ret.display(),
                                                        found: resolved_ret.display(),
                                                    });
                                                }
                                                // Both-concrete case
                                                if !a.contains('?') && !b.contains('?') {
                                                    self.last_error_span = call.span;
                                                    return Err(TypeError::Mismatch {
                                                        expected: b.clone(),
                                                        found: a.clone(),
                                                    });
                                                }
                                            }
                                            Err(_) => {}
                                        }
                                        continue; // Skip the normal unification below
                                    }
                                }

                                // Case 2: Expected takes 1 tuple and actual takes multiple
                                // In Nostos, (a, b) => creates a 2-param lambda, but map on [(Int,Int)]
                                // expects (tuple) -> b. Instead of skipping, unify the tuple elements
                                // with the lambda parameters to catch type mismatches like (Int, String) => a + b.
                                if expected_fn.params.len() == 1 {
                                    if let Some(Type::Tuple(tuple_elems)) = expected_fn.params.first() {
                                        if tuple_elems.len() == actual_fn.params.len() {
                                            // Unify each tuple element with the corresponding lambda param
                                            for (tuple_elem, lambda_param) in tuple_elems.iter().zip(actual_fn.params.iter()) {
                                                let resolved_tuple_elem = self.env.apply_subst(tuple_elem);
                                                let resolved_lambda_param = self.env.apply_subst(lambda_param);
                                                match self.unify_types(&resolved_lambda_param, &resolved_tuple_elem) {
                                                    Ok(()) => {}
                                                    Err(TypeError::UnificationFailed(ref a, ref b)) => {
                                                        // Check if both types are concrete (no type vars)
                                                        if !a.contains('?') && !b.contains('?') {
                                                            self.last_error_span = call.span;
                                                            return Err(TypeError::Mismatch {
                                                                expected: b.clone(),
                                                                found: a.clone(),
                                                            });
                                                        }
                                                    }
                                                    Err(_) => {}
                                                }
                                            }
                                            // Unify return types - catch structural mismatches
                                            match self.unify_types(&actual_fn.ret, &expected_fn.ret) {
                                                Ok(()) => {}
                                                Err(TypeError::UnificationFailed(ref a, ref b)) => {
                                                    // Check structural incompatibility: simple type vs wrapper
                                                    let resolved_ret = self.env.apply_subst(&actual_fn.ret);
                                                    let resolved_expected_ret = self.env.apply_subst(&expected_fn.ret);
                                                    let ret_is_simple = matches!(&resolved_ret,
                                                        Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                                        Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                                        Type::Float | Type::Float32 | Type::Float64 |
                                                        Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                                        Type::Unit);
                                                    let expected_is_wrapper = matches!(&resolved_expected_ret,
                                                        Type::Named { .. } | Type::List(_) | Type::Set(_) |
                                                        Type::Map(_, _) | Type::Tuple(_));
                                                    if ret_is_simple && expected_is_wrapper {
                                                        self.last_error_span = call.span;
                                                        return Err(TypeError::Mismatch {
                                                            expected: resolved_expected_ret.display(),
                                                            found: resolved_ret.display(),
                                                        });
                                                    }
                                                    // Also check both-concrete case
                                                    if !a.contains('?') && !b.contains('?') {
                                                        self.last_error_span = call.span;
                                                        return Err(TypeError::Mismatch {
                                                            expected: b.clone(),
                                                            found: a.clone(),
                                                        });
                                                    }
                                                }
                                                Err(_) => {}
                                            }
                                            continue; // Done with this param
                                        }
                                    }
                                }
                            }

                            // Convert UnificationFailed to Mismatch so it passes through
                            // the error filter in compile.rs. Only convert when BOTH types
                            // are fully resolved (no type variables), to avoid false positives
                            // from intermediate unification during constraint solving.
                            //
                            // Special case for function types: when comparing lambdas, check if
                            // the PARAMETER types conflict even if return types have variables.
                            // E.g., (Int) -> ?1 vs (String) -> ?2 should error on Int/String mismatch.
                            // Also check if return type structures conflict (e.g., Int vs Option[?1]).
                            match self.unify_types(&resolved_arg, &resolved_param) {
                                Ok(()) => {}
                                Err(TypeError::UnificationFailed(ref a, ref b)) => {
                                    // Structural mismatch: the argument has a fundamentally
                                    // different type structure than the parameter expects.
                                    // These mismatches are always errors regardless of type variables
                                    // inside the expected type (e.g., List[?b] with ?b unresolved).

                                    // Check: any non-Var, non-Function type passed where Function expected
                                    if matches!(&resolved_param, Type::Function(_))
                                        && !matches!(&resolved_arg, Type::Function(_) | Type::Var(_) | Type::TypeParam(_))
                                    {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::Mismatch {
                                            expected: "function".to_string(),
                                            found: resolved_arg.display(),
                                        });
                                    }

                                    // Check: concrete primitive/structural type where different structure expected
                                    let arg_is_concrete_non_var = !matches!(&resolved_arg, Type::Var(_) | Type::TypeParam(_));
                                    if arg_is_concrete_non_var {
                                        let param_is_list = matches!(&resolved_param, Type::List(_));
                                        let arg_is_list = matches!(&resolved_arg, Type::List(_));
                                        let param_is_map = matches!(&resolved_param, Type::Map(_, _));
                                        let arg_is_map = matches!(&resolved_arg, Type::Map(_, _));

                                        // Non-list where List expected
                                        if param_is_list && !arg_is_list {
                                            self.last_error_span = call.span;
                                            return Err(TypeError::Mismatch {
                                                expected: "List".to_string(),
                                                found: resolved_arg.display(),
                                            });
                                        }
                                        // Non-map where Map expected
                                        if param_is_map && !arg_is_map {
                                            self.last_error_span = call.span;
                                            return Err(TypeError::Mismatch {
                                                expected: "Map".to_string(),
                                                found: resolved_arg.display(),
                                            });
                                        }
                                    }

                                    // Check for function parameter type conflicts
                                    if let (Type::Function(arg_fn), Type::Function(param_fn)) =
                                        (&resolved_arg, &resolved_param)
                                    {
                                        // Check if any parameter types conflict (both resolved to concrete types)
                                        for (arg_p, param_p) in arg_fn.params.iter().zip(param_fn.params.iter()) {
                                            if !arg_p.has_any_type_var() && !param_p.has_any_type_var() {
                                                if self.unify_types(arg_p, param_p).is_err() {
                                                    self.last_error_span = call.span;
                                                    return Err(TypeError::Mismatch {
                                                        expected: param_p.display(),
                                                        found: arg_p.display(),
                                                    });
                                                }
                                            }
                                        }

                                        // Check if return type structures are incompatible
                                        // E.g., Int cannot unify with Option[?1] even though Option has a var
                                        // Apply substitution because unify_types may have partially succeeded
                                        // (setting type vars) before failing on the return type.
                                        let arg_ret = self.env.apply_subst(&arg_fn.ret);
                                        let param_ret = self.env.apply_subst(&param_fn.ret);
                                        let arg_is_concrete_simple = matches!(arg_ret,
                                            Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                            Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                            Type::Float | Type::Float32 | Type::Float64 |
                                            Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                            Type::Unit);
                                        let param_is_wrapper = matches!(param_ret,
                                            Type::Named { .. } | Type::List(_) | Type::Set(_) |
                                            Type::Map(_, _) | Type::Tuple(_));
                                        if arg_is_concrete_simple && param_is_wrapper {
                                            self.last_error_span = call.span;
                                            return Err(TypeError::Mismatch {
                                                expected: param_ret.display(),
                                                found: arg_ret.display(),
                                            });
                                        }
                                        // Also check reverse: wrapper return where simple expected
                                        let arg_is_wrapper = matches!(&arg_ret,
                                            Type::Named { .. } | Type::List(_) | Type::Set(_) |
                                            Type::Map(_, _) | Type::Tuple(_));
                                        let param_is_simple = matches!(&param_ret,
                                            Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                            Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                            Type::Float | Type::Float32 | Type::Float64 |
                                            Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                            Type::Unit);
                                        if arg_is_wrapper && param_is_simple {
                                            self.last_error_span = call.span;
                                            return Err(TypeError::Mismatch {
                                                expected: param_ret.display(),
                                                found: arg_ret.display(),
                                            });
                                        }
                                    }
                                    // General case: only error if both types in the error are fully resolved.
                                    // Check the error message strings (a and b) instead of the original
                                    // resolved types, because unify_types may have updated the substitution
                                    // for type variables that appeared in resolved_arg/resolved_param.
                                    let a_has_var = a.contains('?');
                                    let b_has_var = b.contains('?');
                                    if !a_has_var && !b_has_var {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::Mismatch {
                                            expected: b.clone(),
                                            found: a.clone(),
                                        });
                                    }
                                    // Structural mismatch: primitive vs wrapper can never unify
                                    // regardless of type variables (e.g., Int vs List[?1] from
                                    // [1,2,3].flatten() where flatten expects List[List[a]]).
                                    let is_prim = |s: &str| matches!(s,
                                        "Int" | "Int8" | "Int16" | "Int32" | "Int64" |
                                        "UInt8" | "UInt16" | "UInt32" | "UInt64" |
                                        "Float" | "Float32" | "Float64" |
                                        "BigInt" | "Decimal" | "String" | "Bool" | "Char" | "()");
                                    let is_wrap = |s: &str| s.starts_with("List[") || s.starts_with('[')
                                        || s.starts_with("Map[") || s.starts_with("Set[")
                                        || (s.starts_with('(') && s.contains(','));
                                    if (is_prim(a) && is_wrap(b)) || (is_wrap(a) && is_prim(b)) {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::Mismatch {
                                            expected: resolved_param.display(),
                                            found: resolved_arg.display(),
                                        });
                                    }
                                }
                                Err(TypeError::ArityMismatch { expected, found }) => {
                                    // Function arity mismatch (e.g., passing 2-arg function to map)
                                    self.last_error_span = call.span;
                                    return Err(TypeError::ArityMismatch { expected, found });
                                }
                                Err(_) => {} // Other errors - ignore for now
                            }
                        }

                        // Unify return type
                        let resolved_ret = self.env.apply_subst(&call.ret_ty);
                        let resolved_ft_ret = self.env.apply_subst(&ft.ret);
                        match self.unify_types(&resolved_ret, &ft.ret) {
                            Ok(()) => {}
                            Err(TypeError::UnificationFailed(ref a, ref b)) => {
                                // Report error if both types are fully resolved
                                if !resolved_ret.has_any_type_var() && !resolved_ft_ret.has_any_type_var() {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::Mismatch {
                                        expected: b.clone(),
                                        found: a.clone(),
                                    });
                                }
                                // Also report error for structural incompatibility even with type vars:
                                // E.g., List[?X] cannot unify with Int regardless of what ?X resolves to.
                                // One is a wrapper type (List, Option, etc.) and other is a simple type.
                                let ret_is_simple = matches!(&resolved_ret,
                                    Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                    Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                    Type::Float | Type::Float32 | Type::Float64 |
                                    Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                    Type::Unit);
                                let expected_is_simple = matches!(&resolved_ft_ret,
                                    Type::Int | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 |
                                    Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 |
                                    Type::Float | Type::Float32 | Type::Float64 |
                                    Type::BigInt | Type::Decimal | Type::String | Type::Bool | Type::Char |
                                    Type::Unit);
                                let ret_is_wrapper = matches!(&resolved_ret,
                                    Type::List(_) | Type::Set(_) | Type::Map(_, _) |
                                    Type::Tuple(_) | Type::Named { .. } | Type::Array(_));
                                let expected_is_wrapper = matches!(&resolved_ft_ret,
                                    Type::List(_) | Type::Set(_) | Type::Map(_, _) |
                                    Type::Tuple(_) | Type::Named { .. } | Type::Array(_));
                                // Simple vs wrapper or wrapper vs simple is always an error
                                if (ret_is_simple && expected_is_wrapper) ||
                                   (ret_is_wrapper && expected_is_simple) {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::Mismatch {
                                        expected: resolved_ft_ret.display(),
                                        found: resolved_ret.display(),
                                    });
                                }
                                // Different wrapper types are always incompatible
                                // (List cannot be Map, Map cannot be Set, etc.)
                                // This catches cases where a variable was bound to the wrong
                                // container type by another function's constraint during solve().
                                let is_container_mismatch = matches!(
                                    (&resolved_ret, &resolved_ft_ret),
                                    (Type::List(_), Type::Map(_, _)) | (Type::Map(_, _), Type::List(_)) |
                                    (Type::List(_), Type::Set(_)) | (Type::Set(_), Type::List(_)) |
                                    (Type::Map(_, _), Type::Set(_)) | (Type::Set(_), Type::Map(_, _)) |
                                    (Type::Tuple(_), Type::List(_)) | (Type::List(_), Type::Tuple(_)) |
                                    (Type::Tuple(_), Type::Map(_, _)) | (Type::Map(_, _), Type::Tuple(_)) |
                                    (Type::Tuple(_), Type::Set(_)) | (Type::Set(_), Type::Tuple(_))
                                );
                                if is_container_mismatch {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::Mismatch {
                                        expected: resolved_ft_ret.display(),
                                        found: resolved_ret.display(),
                                    });
                                }
                            }
                            Err(_) => {} // Other errors - may resolve later
                        }

                        made_progress = true;
                    }
                } else {
                    // Method not found for this type in the inference environment.
                    // For known builtin methods that only work on List/String, report an error.
                    // Note: "get" and "set" are excluded — also used on MVar/reactive types.
                    let list_only_methods = [
                        "length", "len", "head", "tail", "init", "last", "nth",
                        "push", "pop", "slice", "concat", "reverse", "sort",
                        "map", "filter", "fold", "any", "all", "find", "position",
                        "unique", "flatten", "zip", "unzip", "take", "drop",
                        "empty", "isEmpty", "sum", "product", "indexOf", "sortBy",
                        "intersperse", "spanList", "groupBy", "transpose", "pairwise",
                        "isSorted", "isSortedBy", "enumerate", "maximum", "minimum",
                        "takeWhile", "dropWhile", "partition", "zipWith", "flatMap",
                        "join",
                    ];
                    let string_methods = ["split", "trim", "trimStart", "trimEnd",
                                           "toUpper", "toLower", "startsWith", "endsWith",
                                           "contains", "replace", "chars"];

                    let is_list_only = list_only_methods.contains(&call.method_name.as_str());
                    let is_string_method = string_methods.contains(&call.method_name.as_str());

                    if is_list_only || is_string_method {
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

                // Check if the receiver is a type variable with trait bounds
                // If so, look up the method in those traits
                if let Type::Var(var_id) = &resolved {
                    let bounds = self.get_trait_bounds(*var_id);
                    let mut found_in_trait = false;

                    for trait_name in bounds {
                        // Look up the trait definition
                        if let Some(trait_def) = self.env.traits.get(trait_name).cloned() {
                            // Check if the trait provides this method
                            let method_opt = trait_def.required.iter()
                                .chain(trait_def.defaults.iter())
                                .find(|m| m.name == call.method_name);

                            if let Some(method) = method_opt {
                                // Found the method in a trait bound!
                                // Unify the return type
                                let mut ret_type = method.ret.clone();
                                // Replace Self type param with the receiver type
                                if let Type::TypeParam(ref name) = ret_type {
                                    if name == "Self" {
                                        ret_type = resolved.clone();
                                    }
                                }
                                self.unify_types(&call.ret_ty, &ret_type)?;
                                found_in_trait = true;
                                break;
                            }
                        }
                    }

                    if found_in_trait {
                        continue; // Method found in trait, move to next pending call
                    }
                }

                let list_only_methods = [
                    "length", "len", "head", "tail", "init", "last", "nth",
                    "push", "pop", "get", "set", "slice", "concat", "reverse", "sort",
                    "map", "filter", "fold", "any", "all", "find", "position",
                    "unique", "flatten", "zip", "unzip", "take", "drop",
                    "empty", "isEmpty", "sum", "product", "indexOf", "sortBy",
                    "flatMap", "scanl", "foldl", "foldr",
                    "intersperse", "spanList", "groupBy", "transpose", "pairwise",
                    "isSorted", "isSortedBy", "enumerate", "maximum", "minimum",
                    "takeWhile", "dropWhile", "partition", "zipWith",
                ];

                let is_list_only = list_only_methods.contains(&call.method_name.as_str());

                // When receiver is a type variable and method is a known list-only method,
                // infer that the receiver must be a List type. This enables type
                // propagation through generic wrapper functions like:
                //   applyFilter(xs, pred) = xs.filter(pred)
                // Without this, xs stays as a type variable and pred's constraint
                // (must return Bool) is never propagated to the inferred signature.
                // Use a narrower list than list_only_methods - only methods that are
                // truly unique to Lists (not shared with String/Map/Set).
                let infer_list_methods = [
                    "map", "filter", "fold", "flatMap", "any", "all", "find",
                    "sort", "sortBy", "head", "tail", "init", "last",
                    "reverse", "sum", "product", "zip", "unzip", "take", "drop",
                    "unique", "flatten", "position", "indexOf",
                    "push", "pop", "nth", "slice",
                    "scanl", "foldl", "foldr", "enumerate", "intersperse",
                    "spanList", "groupBy", "transpose", "pairwise", "isSorted",
                    "isSortedBy", "maximum", "minimum", "takeWhile", "dropWhile",
                    "partition", "zipWith",
                ];
                let can_infer_list = infer_list_methods.contains(&call.method_name.as_str());
                if matches!(&resolved, Type::Var(_)) && can_infer_list {
                    let elem_ty = self.fresh();
                    let list_ty = Type::List(Box::new(elem_ty));
                    let _ = self.unify_types(&resolved, &list_ty);

                    // Now look up the method with the inferred List type
                    let qualified_name = format!("List.{}", call.method_name);
                    let arity = call.arg_types.len();
                    let fn_type_opt = self.env.functions.get(&qualified_name).cloned()
                        .or_else(|| {
                            let stdlib_name = format!("stdlib.list.{}", call.method_name);
                            self.env.lookup_function_with_arity(&stdlib_name, arity).cloned()
                        })
                        .or_else(|| {
                            self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                matches!(ft.params.first(), Some(Type::List(_)) | Some(Type::Var(_)))
                            })
                        });

                    if let Some(fn_type) = fn_type_opt {
                        let func_ty = self.instantiate_function(&fn_type);
                        if let Type::Function(ft) = func_ty {
                            let min_args = ft.required_params.unwrap_or(ft.params.len());
                            let max_args = ft.params.len();
                            if call.arg_types.len() >= min_args && call.arg_types.len() <= max_args {
                                for (param_ty, arg_ty) in ft.params.iter().zip(call.arg_types.iter()) {
                                    let _ = self.unify_types(arg_ty, param_ty);
                                }
                                if let Err(_) = self.unify_types(&call.ret_ty, &ft.ret) {
                                    // Check for structural container mismatch (e.g., List vs Map)
                                    let resolved_ret = self.env.apply_subst(&call.ret_ty);
                                    let resolved_expected = self.env.apply_subst(&ft.ret);
                                    let is_container_mismatch = matches!(
                                        (&resolved_ret, &resolved_expected),
                                        (Type::List(_), Type::Map(_, _)) | (Type::Map(_, _), Type::List(_)) |
                                        (Type::List(_), Type::Set(_)) | (Type::Set(_), Type::List(_)) |
                                        (Type::Map(_, _), Type::Set(_)) | (Type::Set(_), Type::Map(_, _))
                                    );
                                    if is_container_mismatch {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::Mismatch {
                                            expected: resolved_expected.display(),
                                            found: resolved_ret.display(),
                                        });
                                    }
                                }
                                made_progress = true;
                            }
                        }
                    }

                    continue;
                }

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

                // Record method calls on still-unresolved Vars for signature propagation.
                // These will be encoded as HasMethod(name) in the function signature so that
                // call sites can validate the method when the type becomes concrete.
                if matches!(&resolved, Type::Var(_)) {
                    self.deferred_method_on_var.push((
                        call.receiver_ty.clone(),
                        call.method_name.clone(),
                        call.arg_types.clone(),
                        call.ret_ty.clone(),
                        call.span,
                    ));
                }
            }
            } // end for call in pending

            // If no deferred calls remain or no progress was made, stop iterating
            if deferred.is_empty() || !made_progress {
                // Before breaking, record any deferred method calls on still-unresolved
                // Vars for signature propagation. These will be encoded as HasMethod(name)
                // in the function signature so call sites can validate them.
                for call in &deferred {
                    let resolved = self.env.apply_subst(&call.receiver_ty);
                    if matches!(&resolved, Type::Var(_)) {
                        self.deferred_method_on_var.push((
                            call.receiver_ty.clone(),
                            call.method_name.clone(),
                            call.arg_types.clone(),
                            call.ret_ty.clone(),
                            call.span,
                        ));
                    }
                }
                break;
            }

            // Continue iterating with deferred calls
            pending = deferred;
        } // end iteration loop

        Ok(())
    }

    /// Apply substitution and resolve TypeParams to their mapped types.
    /// This is the complete type resolution that should be used after solve().
    pub fn apply_full_subst(&self, ty: &Type) -> Type {
        // First apply the main substitution
        let substituted = self.env.apply_subst(ty);
        // Then resolve any TypeParams
        self.resolve_type_params(&substituted)
    }

    /// Apply full substitution and verify the result is fully resolved.
    /// Returns Err with the problematic part if resolution is incomplete.
    pub fn resolve_type_fully(&self, ty: &Type) -> Result<Type, UnresolvedTypeError> {
        let resolved = self.apply_full_subst(ty);

        // Check for unresolved type variables
        if self.contains_unresolved_var(&resolved) {
            return Err(UnresolvedTypeError::UnresolvedVar(resolved));
        }

        // Check for leaked TypeParams (optional - some callers may allow this)
        if self.contains_type_param(&resolved) {
            return Err(UnresolvedTypeError::LeakedTypeParam(resolved));
        }

        Ok(resolved)
    }

    /// Check if a type contains unresolved TypeParams (not just type variables).
    /// Used to detect when type parameters have leaked into resolved types.
    pub fn contains_type_param(&self, ty: &Type) -> bool {
        match ty {
            Type::TypeParam(_) => true,
            Type::Var(id) => {
                // Check if this var resolves to a TypeParam
                if let Some(resolved) = self.env.substitution.get(id) {
                    self.contains_type_param(resolved)
                } else {
                    false
                }
            }
            Type::List(elem) => self.contains_type_param(elem),
            Type::Array(elem) => self.contains_type_param(elem),
            Type::Map(k, v) => self.contains_type_param(k) || self.contains_type_param(v),
            Type::Set(elem) => self.contains_type_param(elem),
            Type::Tuple(elems) => elems.iter().any(|e| self.contains_type_param(e)),
            Type::Function(ft) => {
                ft.params.iter().any(|p| self.contains_type_param(p))
                    || self.contains_type_param(&ft.ret)
            }
            Type::Named { args, .. } => args.iter().any(|a| self.contains_type_param(a)),
            Type::IO(inner) => self.contains_type_param(inner),
            Type::Record(rec) => rec.fields.iter().any(|(_, t, _)| self.contains_type_param(t)),
            Type::Variant(var) => var.constructors.iter().any(|c| match c {
                Constructor::Unit(_) => false,
                Constructor::Positional(_, types) => types.iter().any(|t| self.contains_type_param(t)),
                Constructor::Named(_, fields) => fields.iter().any(|(_, t)| self.contains_type_param(t)),
            }),
            _ => false, // Primitives never contain TypeParams
        }
    }

    /// Check if a type is fully resolved (no type variables or type parameters).
    /// This is stricter than contains_type_param - it also rejects unresolved Vars.
    pub fn is_type_fully_resolved(&self, ty: &Type) -> bool {
        match ty {
            Type::TypeParam(_) => false,
            Type::Var(id) => {
                // Check if this var has been resolved
                if let Some(resolved) = self.env.substitution.get(id) {
                    self.is_type_fully_resolved(resolved)
                } else {
                    false // Unresolved type variable
                }
            }
            Type::List(elem) => self.is_type_fully_resolved(elem),
            Type::Array(elem) => self.is_type_fully_resolved(elem),
            Type::Map(k, v) => self.is_type_fully_resolved(k) && self.is_type_fully_resolved(v),
            Type::Set(elem) => self.is_type_fully_resolved(elem),
            Type::Tuple(elems) => elems.iter().all(|e| self.is_type_fully_resolved(e)),
            Type::Function(ft) => {
                ft.params.iter().all(|p| self.is_type_fully_resolved(p))
                    && self.is_type_fully_resolved(&ft.ret)
            }
            Type::Named { args, .. } => args.iter().all(|a| self.is_type_fully_resolved(a)),
            Type::IO(inner) => self.is_type_fully_resolved(inner),
            Type::Record(rec) => rec.fields.iter().all(|(_, t, _)| self.is_type_fully_resolved(t)),
            Type::Variant(var) => var.constructors.iter().all(|c| match c {
                Constructor::Unit(_) => true,
                Constructor::Positional(_, types) => types.iter().all(|t| self.is_type_fully_resolved(t)),
                Constructor::Named(_, fields) => fields.iter().all(|(_, t)| self.is_type_fully_resolved(t)),
            }),
            _ => true, // Primitives are always resolved
        }
    }

    /// Check if a type contains unresolved type variables (Type::Var not in substitution).
    /// This is different from contains_type_param which checks for TypeParams.
    pub fn contains_unresolved_var(&self, ty: &Type) -> bool {
        match ty {
            Type::Var(id) => {
                if let Some(resolved) = self.env.substitution.get(id) {
                    self.contains_unresolved_var(resolved)
                } else {
                    true // Unresolved type variable
                }
            }
            Type::TypeParam(_) => false, // TypeParams are not unresolved Vars
            Type::List(elem) => self.contains_unresolved_var(elem),
            Type::Array(elem) => self.contains_unresolved_var(elem),
            Type::Map(k, v) => self.contains_unresolved_var(k) || self.contains_unresolved_var(v),
            Type::Set(elem) => self.contains_unresolved_var(elem),
            Type::Tuple(elems) => elems.iter().any(|e| self.contains_unresolved_var(e)),
            Type::Function(ft) => {
                ft.params.iter().any(|p| self.contains_unresolved_var(p))
                    || self.contains_unresolved_var(&ft.ret)
            }
            Type::Named { args, .. } => args.iter().any(|a| self.contains_unresolved_var(a)),
            Type::IO(inner) => self.contains_unresolved_var(inner),
            Type::Record(rec) => rec.fields.iter().any(|(_, t, _)| self.contains_unresolved_var(t)),
            Type::Variant(var) => var.constructors.iter().any(|c| match c {
                Constructor::Unit(_) => false,
                Constructor::Positional(_, types) => types.iter().any(|t| self.contains_unresolved_var(t)),
                Constructor::Named(_, fields) => fields.iter().any(|(_, t)| self.contains_unresolved_var(t)),
            }),
            _ => false, // Primitives never contain Vars
        }
    }

    /// Store an expression's inferred type, with collision detection in debug mode.
    /// In debug builds, warns if the same span already has a different type stored.
    fn store_expr_type(&mut self, span: Span, ty: Type) {
        #[cfg(debug_assertions)]
        {
            if let Some(existing) = self.expr_types.get(&span) {
                // Only warn if the types are actually different
                if existing != &ty {
                    eprintln!(
                        "[TYPE-INVARIANT] Span collision detected at {:?}:\n  existing: {}\n  new: {}",
                        span,
                        existing.display(),
                        ty.display()
                    );
                }
            }
        }
        self.expr_types.insert(span, ty);
    }

    /// Verify type system invariants after constraint solving (before finalization).
    /// In debug builds, logs warnings for any violations.
    /// Returns the number of violations found.
    #[cfg(debug_assertions)]
    pub fn verify_post_solve_invariants(&self) -> usize {
        let mut violations = 0;

        // Check that pending_method_calls is empty (should be checked by now)
        if !self.pending_method_calls.is_empty() {
            eprintln!(
                "[TYPE-INVARIANT] {} pending method calls remain after solve",
                self.pending_method_calls.len()
            );
            violations += self.pending_method_calls.len();
        }

        // Check that all constraints have been processed
        if !self.constraints.is_empty() {
            eprintln!(
                "[TYPE-INVARIANT] {} constraints remain after solve",
                self.constraints.len()
            );
            violations += self.constraints.len();
        }

        violations
    }

    /// Verify type system invariants after finalization.
    /// In debug builds, logs warnings for any violations.
    /// Returns the number of violations found.
    #[cfg(debug_assertions)]
    pub fn verify_post_finalize_invariants(&self) -> usize {
        let mut violations = 0;

        // Check all expr_types are fully resolved (no unresolved Vars)
        for (span, ty) in &self.expr_types {
            if self.contains_unresolved_var(ty) {
                eprintln!(
                    "[TYPE-INVARIANT] Unresolved type variable in expr_types at {:?}: {}",
                    span,
                    ty.display()
                );
                violations += 1;
            }
        }

        // Report any expressions with leaked TypeParams (informational, not necessarily a bug)
        if !self.unresolved_type_params.is_empty() {
            eprintln!(
                "[TYPE-INVARIANT] {} expressions have unresolved TypeParams after finalization:",
                self.unresolved_type_params.len()
            );
            for (span, ty, param) in &self.unresolved_type_params {
                eprintln!("  - {:?}: {} (param: {})", span, ty.display(), param);
            }
            // Note: This is tracked but not counted as violation - it's handled by fallback paths
        }

        violations
    }

    /// No-op version for release builds
    #[cfg(not(debug_assertions))]
    pub fn verify_post_solve_invariants(&self) -> usize { 0 }

    /// No-op version for release builds
    #[cfg(not(debug_assertions))]
    pub fn verify_post_finalize_invariants(&self) -> usize { 0 }

    /// Maximum recursion depth for type resolution functions.
    /// Prevents stack overflow from circular type references.
    const MAX_TYPE_RESOLUTION_DEPTH: usize = 100;

    /// Recursively resolve TypeParams in a type using type_param_mappings.
    /// If a TypeParam isn't mapped yet, it stays as-is (for finalize_expr_types).
    pub fn resolve_type_params(&self, ty: &Type) -> Type {
        self.resolve_type_params_with_depth(ty, 0)
    }

    /// Internal implementation with depth tracking to prevent stack overflow.
    fn resolve_type_params_with_depth(&self, ty: &Type, depth: usize) -> Type {
        // Prevent stack overflow from circular type references
        if depth > Self::MAX_TYPE_RESOLUTION_DEPTH {
            // Log warning in all builds - this indicates potential circular types
            eprintln!("[TYPE WARNING] resolve_type_params exceeded max depth ({}) for: {}",
                Self::MAX_TYPE_RESOLUTION_DEPTH, ty.display());
            return ty.clone();
        }

        match ty {
            Type::TypeParam(name) => {
                if let Some(mapped) = self.type_param_mappings.get(name) {
                    // Apply substitution to the mapped type (might be a Var that needs resolving)
                    let substituted = self.env.apply_subst(mapped);
                    self.resolve_type_params_with_depth(&substituted, depth + 1)
                } else {
                    ty.clone()
                }
            }
            Type::Var(id) => {
                // Apply substitution to resolve type variables
                if let Some(resolved) = self.env.substitution.get(id) {
                    self.resolve_type_params_with_depth(resolved, depth + 1)
                } else {
                    ty.clone()
                }
            }
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|t| self.resolve_type_params_with_depth(t, depth + 1)).collect())
            }
            Type::List(elem) => Type::List(Box::new(self.resolve_type_params_with_depth(elem, depth + 1))),
            Type::Array(elem) => Type::Array(Box::new(self.resolve_type_params_with_depth(elem, depth + 1))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.resolve_type_params_with_depth(k, depth + 1)),
                Box::new(self.resolve_type_params_with_depth(v, depth + 1)),
            ),
            Type::Set(elem) => Type::Set(Box::new(self.resolve_type_params_with_depth(elem, depth + 1))),
            Type::Function(f) => Type::Function(FunctionType {
                type_params: f.type_params.clone(),
                params: f.params.iter().map(|t| self.resolve_type_params_with_depth(t, depth + 1)).collect(),
                ret: Box::new(self.resolve_type_params_with_depth(&f.ret, depth + 1)),
                required_params: f.required_params,
            }),
            Type::Named { name, args } => {
                // Handle type parameters that were stored as Named types with no args
                // (e.g., "T" might be Named { name: "T", args: [] } instead of TypeParam("T"))
                if args.is_empty() {
                    if let Some(mapped) = self.type_param_mappings.get(name) {
                        let substituted = self.env.apply_subst(mapped);
                        return self.resolve_type_params_with_depth(&substituted, depth + 1);
                    }
                }
                Type::Named {
                    name: name.clone(),
                    args: args.iter().map(|t| self.resolve_type_params_with_depth(t, depth + 1)).collect(),
                }
            }
            Type::IO(inner) => Type::IO(Box::new(self.resolve_type_params_with_depth(inner, depth + 1))),
            Type::Record(rec) => Type::Record(RecordType {
                name: rec.name.clone(),
                fields: rec.fields.iter()
                    .map(|(n, t, m)| (n.clone(), self.resolve_type_params_with_depth(t, depth + 1), *m))
                    .collect(),
            }),
            _ => ty.clone(),
        }
    }

    /// Convert all TypeParams in a type to type variables.
    /// Uses existing mappings if available, creates fresh vars for new TypeParams.
    /// This is used for recursive function calls to avoid TypeParams in the type system.
    fn instantiate_type_params(&mut self, ty: &Type) -> Type {
        self.instantiate_type_params_with_depth(ty, 0)
    }

    /// Internal implementation with depth tracking to prevent stack overflow.
    fn instantiate_type_params_with_depth(&mut self, ty: &Type, depth: usize) -> Type {
        // Prevent stack overflow from circular type references
        if depth > Self::MAX_TYPE_RESOLUTION_DEPTH {
            // Log warning in all builds - this indicates potential circular types
            eprintln!("[TYPE WARNING] instantiate_type_params exceeded max depth ({}) for: {}",
                Self::MAX_TYPE_RESOLUTION_DEPTH, ty.display());
            return ty.clone();
        }

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
                Type::Tuple(elems.iter().map(|t| self.instantiate_type_params_with_depth(t, depth + 1)).collect())
            }
            Type::List(elem) => Type::List(Box::new(self.instantiate_type_params_with_depth(elem, depth + 1))),
            Type::Array(elem) => Type::Array(Box::new(self.instantiate_type_params_with_depth(elem, depth + 1))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.instantiate_type_params_with_depth(k, depth + 1)),
                Box::new(self.instantiate_type_params_with_depth(v, depth + 1)),
            ),
            Type::Set(elem) => Type::Set(Box::new(self.instantiate_type_params_with_depth(elem, depth + 1))),
            Type::Function(f) => Type::Function(FunctionType {
                type_params: f.type_params.clone(),
                params: f.params.iter().map(|t| self.instantiate_type_params_with_depth(t, depth + 1)).collect(),
                ret: Box::new(self.instantiate_type_params_with_depth(&f.ret, depth + 1)),
                required_params: f.required_params,
            }),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|t| self.instantiate_type_params_with_depth(t, depth + 1)).collect(),
            },
            Type::IO(inner) => Type::IO(Box::new(self.instantiate_type_params_with_depth(inner, depth + 1))),
            Type::Record(rec) => Type::Record(RecordType {
                name: rec.name.clone(),
                fields: rec.fields.iter()
                    .map(|(n, t, m)| (n.clone(), self.instantiate_type_params_with_depth(t, depth + 1), *m))
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
                    // Add trait bounds from the TypeParam's constraints (for propagation into lambdas)
                    // Clone constraints to avoid borrow issues
                    let constraints = self.current_type_param_constraints.get(name).cloned();
                    if let (Type::Var(var_id), Some(constraints)) = (fresh_var, constraints) {
                        for constraint in constraints {
                            self.add_trait_bound(var_id, constraint);
                        }
                        let fresh_var = Type::Var(var_id);
                        self.type_param_mappings.insert(name.clone(), fresh_var.clone());
                        self.unify_types(&fresh_var, &t2)
                    } else {
                        let fresh_var = self.fresh();
                        self.type_param_mappings.insert(name.clone(), fresh_var.clone());
                        self.unify_types(&fresh_var, &t2)
                    }
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
                    // Add trait bounds from the TypeParam's constraints (for propagation into lambdas)
                    // Clone constraints to avoid borrow issues
                    let constraints = self.current_type_param_constraints.get(name).cloned();
                    if let (Type::Var(var_id), Some(constraints)) = (fresh_var, constraints) {
                        for constraint in constraints {
                            self.add_trait_bound(var_id, constraint);
                        }
                        let fresh_var = Type::Var(var_id);
                        self.type_param_mappings.insert(name.clone(), fresh_var.clone());
                        self.unify_types(&t1, &fresh_var)
                    } else {
                        let fresh_var = self.fresh();
                        self.type_param_mappings.insert(name.clone(), fresh_var.clone());
                        self.unify_types(&t1, &fresh_var)
                    }
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
                // When required_params is None, ALL params are required (no defaults)
                if provided < min_required || provided > max_params {
                    return Err(TypeError::ArityMismatch {
                        expected: min_required,
                        found: provided,
                    });
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
            _ => {
                Err(TypeError::UnificationFailed(t1.display(), t2.display()))
            }
        }
    }

    // =========================================================================
    // AST Type Inference
    // =========================================================================

    /// Convert an AST TypeExpr to an internal Type.
    /// Uses the current_type_params field to identify type parameters.
    /// Prefer `type_from_ast_with_params` for explicit type parameter passing.
    pub fn type_from_ast(&mut self, ty: &TypeExpr) -> Type {
        // Clone to avoid borrow issues
        let type_params = self.current_type_params.clone();
        self.type_from_ast_with_params(ty, &type_params)
    }

    /// Convert an AST TypeExpr to an internal Type with explicit type parameters.
    /// This is the preferred version - it avoids mutable state for tracking type params.
    pub fn type_from_ast_with_params(&mut self, ty: &TypeExpr, type_params: &HashSet<String>) -> Type {
        match ty {
            TypeExpr::Name(ident) => {
                let name = &ident.node;
                // Check if this is a type parameter in the explicit scope
                if type_params.contains(name) {
                    return Type::TypeParam(name.clone());
                }
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
                let type_args: Vec<_> = args.iter().map(|a| self.type_from_ast_with_params(a, type_params)).collect();
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
                let param_types: Vec<_> = params.iter().map(|p| self.type_from_ast_with_params(p, type_params)).collect();
                let ret_type = self.type_from_ast_with_params(ret, type_params);
                Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_type),
                })
            }
            TypeExpr::Record(fields) => {
                let field_types: Vec<_> = fields
                    .iter()
                    .map(|(name, ty)| (name.node.clone(), self.type_from_ast_with_params(ty, type_params), false))
                    .collect();
                Type::Record(RecordType {
                    name: None,
                    fields: field_types,
                })
            }
            TypeExpr::Tuple(elems) => {
                let elem_types: Vec<_> = elems.iter().map(|e| self.type_from_ast_with_params(e, type_params)).collect();
                Type::Tuple(elem_types)
            }
            TypeExpr::Unit => Type::Unit,
        }
    }

    /// Infer the type of an expression and store it for later retrieval.
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        let ty = self.infer_expr_inner(expr)?;
        // Store the inferred type keyed by the expression's span
        // Uses store_expr_type for collision detection in debug builds
        self.store_expr_type(expr.span(), ty.clone());
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
            Expr::BinOp(left, op, right, span) => self.infer_binop(left, *op, right, *span),

            // Unary operations
            Expr::UnaryOp(op, operand, _) => {
                let operand_ty = self.infer_expr(operand)?;
                match op {
                    UnaryOp::Neg => {
                        // Neg requires numeric type (Num trait)
                        self.require_trait(operand_ty.clone(), "Num");
                        Ok(operand_ty)
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
                let mut arg_spans = Vec::new();
                for arg in args {
                    let expr = match arg {
                        CallArg::Positional(e) | CallArg::Named(_, e) => e,
                    };
                    arg_spans.push(expr.span());
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
                            // Constructor call - check arity
                            match &ty {
                                Type::Function(ft) => {
                                    // Constructor with fields - check arg count
                                    if ft.params.len() != args.len() {
                                        return Err(TypeError::ArityMismatch {
                                            expected: ft.params.len(),
                                            found: args.len(),
                                        });
                                    }
                                }
                                _ => {
                                    // Unit constructor (no fields) - should not be called with args
                                    if !args.is_empty() {
                                        return Err(TypeError::ArityMismatch {
                                            expected: 0,
                                            found: args.len(),
                                        });
                                    }
                                }
                            }
                            ty
                        } else {
                            // Function not found with this arity - check if it exists with different arity
                            if let Some(ft) = self.env.lookup_function_any_arity(name) {
                                return Err(TypeError::ArityMismatch {
                                    expected: ft.params.len(),
                                    found: args.len(),
                                });
                            }
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
                        // For multi-type-param functions (e.g., pair[A, B]),
                        // HM inference may have merged distinct type params into the
                        // same Var, causing the return type to have merged vars too.
                        // Return a fresh var to avoid false positive type errors.
                        if ft.type_params.len() >= 2 {
                            return Ok(self.fresh());
                        }
                        return Ok((*ft.ret).clone());
                    }
                    // If not a function type, fall through to normal unification
                    // which will report an appropriate error
                }

                let ret_ty = self.fresh();
                let arg_types_clone = arg_types.clone();
                let expected_func_ty = Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: arg_types,
                    ret: Box::new(ret_ty.clone()),
                });

                // Store param/arg type pairs for post-solve structural mismatch checking.
                // We need to compare resolved types AFTER solve() to catch cases like
                // passing a Map where a List is expected.
                if let Type::Function(ref ft) = func_ty {
                    for (param_ty, arg_ty) in ft.params.iter().zip(arg_types_clone.iter()) {
                        self.deferred_fn_call_checks.push((param_ty.clone(), arg_ty.clone(), *call_span));
                    }
                    // Record trait bounds from constrained type params for post-solve checking.
                    // When a param IS a Var with trait bounds (from instantiate_function),
                    // record (arg_type, trait_name, span) for later verification.
                    // This catches cases like equal[T: Eq] called with function args.
                    // NOTE: Only check top-level Vars, NOT Vars inside containers (e.g.,
                    // List[Var(id)]). For container cases, the HasTrait constraint from
                    // instantiate_function handles checking via solve()/deferred_has_trait.
                    for (param_ty, arg_ty) in ft.params.iter().zip(arg_types_clone.iter()) {
                        if let Type::Var(var_id) = param_ty {
                            if let Some(bounds) = self.trait_bounds.get(var_id) {
                                for bound in bounds.clone() {
                                    self.deferred_generic_trait_checks.push((arg_ty.clone(), bound, *call_span));
                                }
                            }
                        }
                    }
                } else {
                    // func_ty is not directly a Function (likely a Var from curried calls,
                    // e.g., `adder(1)("hello")` where the inner call returns a Var).
                    // Defer the check: after solve(), resolve func_ty and check param/arg types.
                    self.deferred_indirect_call_checks.push((func_ty.clone(), arg_types_clone.clone(), *call_span));
                }

                // Use unify_at with the call span for precise error reporting
                self.unify_at(func_ty, expected_func_ty, *call_span);

                // Emit trait constraints for known builtin functions that have
                // trait requirements not encoded in their generic signatures.
                // E.g., sort: [a] -> [a] should require Ord a, but BUILTINS
                // signatures don't support trait bounds.
                if let Expr::Var(ident) = func.as_ref() {
                    let fn_name = ident.node.as_str();
                    if matches!(fn_name, "sort" | "maximum" | "minimum" | "isSorted"
                                       | "sum" | "product") {
                        // These functions take a list as the first argument.
                        // Extract the element type from the resolved first arg.
                        if let Some(first_arg) = arg_types_clone.first() {
                            let resolved_arg = self.env.apply_subst(first_arg);
                            if let Type::List(elem) = &resolved_arg {
                                let resolved_elem = self.env.apply_subst(elem);
                                let trait_name = if matches!(fn_name, "sum" | "product") {
                                    "Num"
                                } else {
                                    "Ord"
                                };
                                self.require_trait(resolved_elem, trait_name);
                            }
                        }
                    }

                    // min/max/clamp require Ord on the argument type directly
                    if matches!(fn_name, "min" | "max" | "clamp") {
                        if let Some(first_arg) = arg_types_clone.first() {
                            let resolved_arg = self.env.apply_subst(first_arg);
                            self.require_trait(resolved_arg, "Ord");
                        }
                    }

                    // abs/pow require Num on the argument type directly
                    if matches!(fn_name, "abs" | "pow") {
                        if let Some(first_arg) = arg_types_clone.first() {
                            let resolved_arg = self.env.apply_subst(first_arg);
                            self.require_trait(resolved_arg, "Num");
                        }
                    }

                    // length/len only work on collections (List, String, Map, Set, arrays).
                    // Save for deferred checking after types are resolved.
                    if matches!(fn_name, "length" | "len") {
                        if let Some(first_arg) = arg_types_clone.first() {
                            self.deferred_length_checks.push((first_arg.clone(), *call_span));
                        }
                    }
                }

                // Save return type for post-solve Set/Map Hash+Eq checking.
                // If the return type resolves to Set[X] or Map[K, V], we need
                // Hash+Eq on X or K. Can't check now because unify_at is deferred.
                self.deferred_collection_ret_checks.push((ret_ty.clone(), *call_span));

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
            Expr::If(cond, then_branch, else_branch, span) => {
                let cond_ty = self.infer_expr(cond)?;
                self.unify(cond_ty, Type::Bool);

                let then_ty = self.infer_expr(then_branch)?;
                let else_ty = self.infer_expr(else_branch)?;
                self.unify(then_ty.clone(), else_ty.clone());

                // Record branch types for post-solve structural container mismatch check
                self.deferred_branch_type_checks.push((then_ty.clone(), else_ty, *span));

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

                            // For generic record types, get params and create substitution from base type args
                            if let Some(TypeDef::Record { params, fields: def_fields, .. }) = self.env.lookup_type(&resolved_type_name).cloned() {
                                // Create substitution from base type's args (if it's a Named type)
                                let resolved_base = self.env.apply_subst(&base_ty);
                                let substitution: HashMap<String, Type> = if let Type::Named { args, .. } = &resolved_base {
                                    params.iter().zip(args.iter())
                                        .map(|(p, a)| (p.name.clone(), a.clone()))
                                        .collect()
                                } else {
                                    // Create fresh vars if base type isn't resolved yet
                                    params.iter()
                                        .map(|p| (p.name.clone(), self.fresh()))
                                        .collect()
                                };

                                let type_args: Vec<Type> = params.iter()
                                    .map(|p| substitution.get(&p.name).cloned().unwrap_or_else(|| Type::TypeParam(p.name.clone())))
                                    .collect();

                                let expected_ty = Type::Named {
                                    name: resolved_type_name.clone(),
                                    args: type_args,
                                };
                                self.unify(base_ty, expected_ty.clone());

                                // Infer and check types of named update fields
                                for field in &fields[1..] {
                                    if let RecordField::Named(fname, expr) = field {
                                        let ty = self.infer_expr(expr)?;
                                        if let Some((_, fty, _)) = def_fields.iter().find(|(n, _, _)| n == &fname.node) {
                                            let substituted_fty = Self::substitute_type_params(fty, &substitution);
                                            self.unify(ty, substituted_fty);
                                        } else {
                                            return Err(TypeError::NoSuchField {
                                                ty: resolved_type_name.clone(),
                                                field: fname.node.clone(),
                                            });
                                        }
                                    }
                                }

                                return Ok(expected_ty);
                            }
                        }
                    }
                }

                if let Some(TypeDef::Record {
                    params, fields: def_fields, ..
                }) = self.env.lookup_type(&resolved_type_name).cloned()
                {
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

                    // If no fields provided and record has required fields,
                    // treat this as a reference to the constructor function, not a construction
                    let has_required_fields = def_fields.iter().any(|(_, _, has_default)| !has_default);
                    if fields.is_empty() && has_required_fields {
                        // Return constructor function type: (field_types...) -> RecordType
                        let field_types: Vec<Type> = def_fields.iter()
                            .map(|(_, ty, _)| Self::substitute_type_params(ty, &substitution))
                            .collect();
                        let result_ty = Type::Named {
                            name: resolved_type_name.clone(),
                            args: type_args,
                        };
                        return Ok(Type::Function(FunctionType {
                            required_params: None,
                            type_params: vec![],
                            params: field_types,
                            ret: Box::new(result_ty),
                        }));
                    }

                    let mut provided = HashMap::new();
                    let mut positional_count = 0;
                    for field in fields {
                        match field {
                            RecordField::Positional(expr) => {
                                // Positional args match in order
                                let ty = self.infer_expr(expr)?;
                                let idx = positional_count;
                                positional_count += 1;
                                if idx < def_fields.len() {
                                    let (fname, fty, _) = &def_fields[idx];
                                    // Apply type parameter substitution to field type
                                    let substituted_fty = Self::substitute_type_params(fty, &substitution);
                                    self.unify(ty, substituted_fty);
                                    provided.insert(fname.clone(), ());
                                } else {
                                    // Too many positional arguments
                                    return Err(TypeError::ArityMismatch {
                                        expected: def_fields.len(),
                                        found: positional_count,
                                    });
                                }
                            }
                            RecordField::Named(fname, expr) => {
                                let ty = self.infer_expr(expr)?;
                                if let Some((_, fty, _)) =
                                    def_fields.iter().find(|(n, _, _)| n == &fname.node)
                                {
                                    // Apply type parameter substitution to field type
                                    let substituted_fty = Self::substitute_type_params(fty, &substitution);
                                    self.unify(ty, substituted_fty);
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

                    // Check that all required fields are provided
                    // (fields without defaults are required)
                    for (fname, _, has_default) in &def_fields {
                        if !has_default && !provided.contains_key(fname) {
                            return Err(TypeError::MissingField(fname.clone()));
                        }
                    }

                    Ok(Type::Named {
                        name: resolved_type_name.clone(),
                        args: type_args,
                    })
                } else if let Some(ctor_ty) = self.lookup_constructor(&resolved_type_name) {
                    // This is a variant constructor call (e.g., Some(42), Good("hello"), Nil)
                    if fields.is_empty() {
                        // Unit constructor (no arguments) - ctor_ty is just the result type
                        Ok(ctor_ty)
                    } else if !matches!(&ctor_ty, Type::Function(_)) {
                        // Unit constructor called with arguments - error
                        return Err(TypeError::ArityMismatch {
                            expected: 0,
                            found: fields.len(),
                        });
                    } else {
                        // Positional/named constructor - ctor_ty is a Function type
                        // For named fields, validate field names against the constructor definition
                        let has_named_fields = fields.iter().any(|f| matches!(f, RecordField::Named(_, _)));
                        if has_named_fields {
                            // Look up the constructor definition to get field names AND return type
                            // sharing the same fresh type vars (so field constraints flow to return type)
                            if let Some((ctor_fields, shared_ret_ty)) = self.lookup_constructor_named_fields_with_ret(&resolved_type_name) {
                                // Validate and type-check named fields
                                let mut provided = HashMap::new();
                                let mut positional_count = 0;
                                for field in fields {
                                    match field {
                                        RecordField::Named(fname, expr) => {
                                            let ty = self.infer_expr(expr)?;
                                            if let Some((_, expected_field_ty)) = ctor_fields.iter().find(|(n, _)| n == &fname.node) {
                                                self.unify(ty, expected_field_ty.clone());
                                                provided.insert(fname.node.clone(), ());
                                            } else {
                                                return Err(TypeError::NoSuchField {
                                                    ty: resolved_type_name.clone(),
                                                    field: fname.node.clone(),
                                                });
                                            }
                                        }
                                        RecordField::Positional(expr) => {
                                            let ty = self.infer_expr(expr)?;
                                            if positional_count < ctor_fields.len() {
                                                let (_, expected_field_ty) = &ctor_fields[positional_count];
                                                self.unify(ty, expected_field_ty.clone());
                                            }
                                            positional_count += 1;
                                        }
                                    }
                                }
                                // Use the return type that shares fresh vars with field types
                                return Ok(shared_ret_ty);
                            }
                        }

                        // Positional args or fallback - infer arg types and unify
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
                            params: arg_types.clone(),
                            ret: Box::new(ret_ty.clone()),
                        });

                        // Unify constructor type with expected - this will catch type mismatches
                        self.unify(ctor_ty.clone(), expected_func_ty);

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
                    Type::Map(key_ty, val_ty) => {
                        // Map[K,V] indexing: m[key] returns V, index must be K
                        self.constraints.push(Constraint::Equal(
                            index_ty,
                            (**key_ty).clone(),
                            Some(*span),
                        ));
                        self.constraints.push(Constraint::Equal(
                            elem_ty.clone(),
                            (**val_ty).clone(),
                            Some(*span),
                        ));
                        Ok(elem_ty)
                    }
                    Type::Set(set_elem_ty) => {
                        // Set[T] indexing: s[elem] returns Bool (contains check)
                        self.constraints.push(Constraint::Equal(
                            index_ty,
                            (**set_elem_ty).clone(),
                            Some(*span),
                        ));
                        self.constraints.push(Constraint::Equal(
                            elem_ty.clone(),
                            Type::Bool,
                            Some(*span),
                        ));
                        Ok(elem_ty)
                    }
                    _ => {
                        // Container is List, Array, or String
                        let list_ty = Type::List(Box::new(elem_ty.clone()));

                        self.unify(index_ty, Type::Int);

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
                            // Check if this is an mvar reassignment disguised as a let binding.
                            // The parser generates Stmt::Let for `x = expr` even when x is an mvar,
                            // because it doesn't distinguish local bindings from mvar assignments.
                            // If the binding target is a simple Var that's already bound as mutable
                            // (i.e., it's an mvar), unify the value type with the mvar's declared type.
                            let mvar_type = if let Pattern::Var(ident) = &binding.pattern {
                                self.env.lookup(&ident.node)
                                    .filter(|(_, mutable)| *mutable)
                                    .map(|(ty, _)| ty.clone())
                            } else {
                                None
                            };
                            if let Some(ref var_ty) = mvar_type {
                                // Only check concrete types - bare container names like Map without
                                // type params can't meaningfully unify with parameterized versions.
                                let is_checkable = !var_ty.has_any_type_var() && !matches!(var_ty,
                                    Type::Named { args, .. } if args.is_empty());
                                if is_checkable {
                                    let value_ty = self.infer_expr(&binding.value)?;
                                    self.unify(value_ty, var_ty.clone());
                                } else {
                                    self.infer_binding(binding)?;
                                }
                            } else {
                                self.infer_binding(binding)?;
                            }
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
                                for (i, (param_ty, arg_ty)) in ft.params.iter().zip(arg_types.iter()).enumerate() {
                                    self.unify_at(arg_ty.clone(), param_ty.clone(), *call_span);
                                    // Record for post-solve structural mismatch checking
                                    self.deferred_fn_call_checks.push((param_ty.clone(), arg_ty.clone(), *call_span));
                                    // Pre-check: if param expects List[Tuple(...)], verify the arg
                                    // list literal contains tuple elements. The post-solve check can't
                                    // detect this because LIFO constraint processing corrupts the arg type.
                                    if let Type::List(inner_p) = param_ty {
                                        if matches!(inner_p.as_ref(), Type::Tuple(_)) {
                                            if let Some(arg_expr) = args.get(i) {
                                                let expr = match arg_expr {
                                                    CallArg::Positional(e) | CallArg::Named(_, e) => e,
                                                };
                                                if let Expr::List(elems, _, _) = expr {
                                                    if !elems.is_empty() {
                                                        let has_non_tuple = elems.iter().any(|e| {
                                                            !matches!(e, Expr::Tuple(_, _))
                                                        });
                                                        if has_non_tuple {
                                                            self.last_error_span = Some(*call_span);
                                                            // Use the inferred element type for a clear message
                                                            let inferred_elem = self.env.apply_subst(arg_ty);
                                                            let found_str = inferred_elem.display();
                                                            return Err(TypeError::Mismatch {
                                                                expected: "List of key-value tuples".to_string(),
                                                                found: found_str,
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                // Save return type for post-solve Set/Map Hash+Eq checking
                                self.deferred_collection_ret_checks.push(((*ft.ret).clone(), *call_span));
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
                                // Record for post-solve structural mismatch checking
                                self.deferred_fn_call_checks.push((param_ty.clone(), arg_ty.clone(), *call_span));
                            }
                            // Save return type for post-solve Set/Map Hash+Eq checking
                            self.deferred_collection_ret_checks.push(((*ft.ret).clone(), *call_span));
                            return Ok(*ft.ret);
                        }
                    }
                }

                // Regular method call on a value - use UFCS lookup
                let receiver_ty = self.infer_expr(receiver)?;
                let mut arg_types = vec![receiver_ty.clone()];
                let mut named_args = Vec::new();
                for arg in args {
                    let (expr, is_named) = match arg {
                        CallArg::Named(name, e) => {
                            named_args.push((name.node.clone(), arg_types.len()));
                            (e, true)
                        }
                        CallArg::Positional(e) => (e, false),
                    };
                    let _ = is_named;
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
                    named_args,
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
            Expr::Quote(_inner, _) => {
                // Quote captures code as data - don't type-check the inner expression
                // since it may contain splices (~expr) which have different semantics
                Ok(Type::Named {
                    name: "Expr".to_string(),
                    args: vec![],
                })
            }
            Expr::Splice(inner, _) => {
                // Splice can take any expression - it will be evaluated at template expansion time
                // The result type is Expr (it becomes part of the AST being built)
                let _ = self.infer_expr(inner)?;
                Ok(Type::Named {
                    name: "Expr".to_string(),
                    args: vec![],
                })
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
    fn infer_binop(&mut self, left: &Expr, op: BinOp, right: &Expr, span: Span) -> Result<Type, TypeError> {
        let left_ty = self.infer_expr(left)?;
        let right_ty = self.infer_expr(right)?;

        match op {
            // Arithmetic: both operands numeric, with implicit Int->Float coercion
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                // Apply substitutions to get resolved types
                let resolved_left = self.env.apply_subst(&left_ty);
                let resolved_right = self.env.apply_subst(&right_ty);

                // For scalar dispatch, peek at pending constraints to see through type variables
                // This allows patterns like `(Named, Float)` to match even when the Named type
                // is currently represented as Var(x) that will resolve to Named during solving
                let left_for_dispatch = self.peek_pending_resolution(&resolved_left);

                // Check for Int/Float mixing - result is Float due to implicit coercion
                match (&left_for_dispatch, &resolved_right) {
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
                    // We use left_for_dispatch which peeks at pending constraints, so this pattern
                    // matches even when the left type is currently Var(x) but will resolve to Named
                    (Type::Named { .. }, Type::Float | Type::Int | Type::Float64 | Type::Int64) |
                    (Type::Named { .. }, Type::Var(_)) |
                    (Type::Record(_), Type::Float | Type::Int | Type::Float64 | Type::Int64) |
                    (Type::Record(_), Type::Var(_)) => {
                        // Custom type with numeric scalar - require Num trait and return the custom type
                        // The compiler will dispatch to {typeLower}{Op}Scalar function
                        self.require_trait(left_ty.clone(), "Num");
                        Ok(left_ty)
                    }
                    // Type variable + concrete numeric → unify for lambda parameter inference
                    // This case handles lambdas where the parameter type needs to be inferred
                    (Type::Var(_), Type::Float | Type::Int | Type::Float64 | Type::Int64) => {
                        self.unify(left_ty.clone(), right_ty.clone());
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
                // List concatenation must have matching element types
                // [String] ++ [Int] is a type error

                // Resolve any type variables first to see actual types
                let resolved_left = self.env.apply_subst(&left_ty);
                let resolved_right = self.env.apply_subst(&right_ty);

                match (&resolved_left, &resolved_right) {
                    // Both are concrete list types - try to unify element types
                    // but allow heterogeneous lists (Nostos supports mixed-type lists)
                    (Type::List(elem_left), Type::List(elem_right)) => {
                        // Try to unify element types (useful when one has type vars)
                        // If both are concrete but different, allow it - Nostos
                        // supports heterogeneous lists like Python
                        let _ = self.unify_types(elem_left, elem_right);
                        Ok(resolved_left.clone())
                    }
                    // Left is list, right is type variable - constrain right to be SOME list
                    (Type::List(_), Type::Var(id)) => {
                        // Constrain the Var to be a list with a fresh element type.
                        // This keeps the parameter generic (allowing heterogeneous lists)
                        // while the RESULT preserves the known element type from the left
                        // operand (needed for recursive functions like collect(n) = [show(n)] ++ collect(n-1)
                        // to correctly infer Int -> [String] instead of Int -> [a]).
                        let fresh_elem = self.fresh();
                        self.env.substitution.insert(*id, Type::List(Box::new(fresh_elem)));
                        Ok(resolved_left.clone())
                    }
                    // Right is list, left is type variable - constrain left to be SOME list
                    (Type::Var(id), Type::List(_)) => {
                        let fresh_elem = self.fresh();
                        self.env.substitution.insert(*id, Type::List(Box::new(fresh_elem)));
                        Ok(resolved_right.clone())
                    }
                    // String ++ String is valid concatenation
                    (Type::String, Type::String) => {
                        Ok(Type::String)
                    }
                    // String ++ Var or Var ++ String - defer check until Var resolves
                    // (the Var may come from a deferred field access that hasn't resolved yet)
                    (Type::String, Type::Var(_)) | (Type::Var(_), Type::String) => {
                        // Add Concat trait bound to the Var side so it propagates
                        // through function signatures (e.g., addStr(x) = x ++ "!")
                        let var_ty = if matches!(&resolved_left, Type::Var(_)) { &left_ty } else { &right_ty };
                        self.require_trait(var_ty.clone(), "Concat");
                        self.deferred_concat_checks.push((left_ty.clone(), right_ty.clone(), span));
                        Ok(Type::String)
                    }
                    // List ++ String or String ++ List is always a type error
                    (Type::List(_), Type::String) | (Type::String, Type::List(_)) => {
                        return Err(TypeError::UnificationFailed(
                            resolved_left.display(),
                            resolved_right.display(),
                        ));
                    }
                    // Both are type variables - defer check until types resolve
                    (Type::Var(_), Type::Var(_)) => {
                        self.unify(left_ty.clone(), right_ty.clone());
                        // Add Concat trait bound so it propagates through wrapper signatures
                        self.require_trait(left_ty.clone(), "Concat");
                        // Store for deferred check after type resolution
                        self.deferred_concat_checks.push((left_ty.clone(), right_ty.clone(), span));
                        Ok(left_ty)
                    }
                    _ => {
                        // Concrete non-string, non-list type used with ++
                        // Report error on whichever side is not String/List
                        let bad_type = if !matches!(&resolved_left, Type::String | Type::List(_) | Type::Var(_)) {
                            &resolved_left
                        } else {
                            &resolved_right
                        };
                        return Err(TypeError::MissingTraitImpl {
                            ty: bad_type.display(),
                            trait_name: "Concat".to_string(),
                        });
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
                    params: vec![left_ty.clone()],
                    ret: Box::new(result_ty.clone()),
                });
                self.unify(right_ty, expected_func);

                // Pipe behaves like a function call. Apply the same deferred
                // checks that Expr::Call does (e.g., length on non-collections).
                if let Expr::Var(ident) = right {
                    if matches!(ident.node.as_str(), "length" | "len") {
                        self.deferred_length_checks.push((left_ty, span));
                    }
                }

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
                                    // Validate pattern arity matches constructor field count
                                    if pats.len() != field_types.len() {
                                        return Err(TypeError::ArityMismatch {
                                            expected: field_types.len(),
                                            found: pats.len(),
                                        });
                                    }
                                    // Use concrete field types from the type definition
                                    // This is key for type narrowing: when matching Add(e1, e2) on Expr,
                                    // we know e1 and e2 are Expr, not just type variables
                                    for (pat, field_ty) in pats.iter().zip(field_types.iter()) {
                                        self.infer_pattern(pat, field_ty)?;
                                    }
                                } else {
                                    // Validate pattern arity matches constructor param count
                                    if pats.len() != f.params.len() {
                                        return Err(TypeError::ArityMismatch {
                                            expected: f.params.len(),
                                            found: pats.len(),
                                        });
                                    }
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
                            // Constructor should be a function from fields to the type
                            // For named patterns, we need to unify with the return type, not the function type
                            if let Type::Function(f) = &ctor_ty {
                                // Unify expected with the constructor's return type
                                self.unify(expected.clone(), (*f.ret).clone());

                                // Try to get the constructor's named fields for validation
                                let ctor_fields: Option<Vec<(String, Type)>> =
                                    if let Type::Named { name: type_name, .. } = &*f.ret {
                                        self.env.lookup_variant_named_fields(type_name, &name.node)
                                    } else {
                                        None
                                    };

                                for field in fields {
                                    match field {
                                        RecordPatternField::Punned(fname) => {
                                            // Validate field name exists if we have constructor info
                                            let field_ty = if let Some(ref ctor_fields) = ctor_fields {
                                                if let Some((_, ty)) = ctor_fields.iter().find(|(n, _)| n == &fname.node) {
                                                    ty.clone()
                                                } else {
                                                    return Err(TypeError::NoSuchField {
                                                        ty: name.node.clone(),
                                                        field: fname.node.clone(),
                                                    });
                                                }
                                            } else {
                                                self.fresh()
                                            };
                                            self.env.bind(fname.node.clone(), field_ty, false);
                                        }
                                        RecordPatternField::Named(fname, pat) => {
                                            // Validate field name exists if we have constructor info
                                            let field_ty = if let Some(ref ctor_fields) = ctor_fields {
                                                if let Some((_, ty)) = ctor_fields.iter().find(|(n, _)| n == &fname.node) {
                                                    ty.clone()
                                                } else {
                                                    return Err(TypeError::NoSuchField {
                                                        ty: name.node.clone(),
                                                        field: fname.node.clone(),
                                                    });
                                                }
                                            } else {
                                                self.fresh()
                                            };
                                            self.infer_pattern(pat, &field_ty)?;
                                        }
                                        RecordPatternField::Rest(_) => {}
                                    }
                                }
                            } else {
                                // Unit constructor case - unify directly
                                self.unify(expected.clone(), ctor_ty);
                            }
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

            Pattern::Range(_, _, _, _) => {
                // Range patterns match integers
                self.unify(expected.clone(), Type::Int);
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
            // Priority: user types first, then built-in types, then stdlib types.
            // Built-in types (List, Option, Result) are registered in standard_env()
            // and should be lower priority than user-defined types so that user
            // constructor names (e.g., Nil in a user-defined MyList) take precedence.
            let builtin_types = ["List", "Option", "Result"];
            let a_is_stdlib = a.starts_with("stdlib.");
            let b_is_stdlib = b.starts_with("stdlib.");
            let a_is_builtin = !a_is_stdlib && builtin_types.contains(&a.as_str());
            let b_is_builtin = !b_is_stdlib && builtin_types.contains(&b.as_str());
            let a_priority = if a_is_stdlib { 2 } else if a_is_builtin { 1 } else { 0 };
            let b_priority = if b_is_stdlib { 2 } else if b_is_builtin { 1 } else { 0 };
            a_priority.cmp(&b_priority).then(a.cmp(b))
        });
        for type_name in type_names {
            let def = types_clone.get(type_name).expect("type_name from types_clone keys");
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
                                let func_ty = Type::Function(FunctionType { required_params: None,
                                    type_params: vec![],
                                    params: instantiated_params.clone(),
                                    ret: Box::new(result_ty.clone()),
                                });
                                return Some(func_ty);
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

    /// Look up a variant constructor's named fields (with instantiated types).
    /// Returns None if the constructor is not a Named variant or not found.
    fn lookup_constructor_named_fields(&mut self, name: &str) -> Option<Vec<(String, Type)>> {
        self.lookup_constructor_named_fields_with_ret(name).map(|(fields, _)| fields)
    }

    /// Look up a variant constructor's named fields AND return type, using shared fresh vars.
    fn lookup_constructor_named_fields_with_ret(&mut self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        let types_clone = self.env.types.clone();
        for (type_name, def) in &types_clone {
            if let TypeDef::Variant { params, constructors } = def {
                for ctor in constructors {
                    if let Constructor::Named(ctor_name, fields) = ctor {
                        if ctor_name == name {
                            let substitution: HashMap<String, Type> = params
                                .iter()
                                .map(|p| (p.name.clone(), self.fresh()))
                                .collect();
                            let field_types = fields.iter().map(|(fname, fty)| {
                                (fname.clone(), Self::substitute_type_params(fty, &substitution))
                            }).collect();
                            let type_args: Vec<Type> = params
                                .iter()
                                .map(|p| substitution.get(&p.name).cloned().unwrap_or_else(|| Type::TypeParam(p.name.clone())))
                                .collect();
                            let ret_ty = Type::Named {
                                name: type_name.clone(),
                                args: type_args,
                            };
                            return Some((field_types, ret_ty));
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
        self.unify(body_ty.clone(), result_ty.clone());

        // Record branch type for post-solve structural container mismatch check
        let arm_span = arm.body.span();
        self.deferred_branch_type_checks.push((body_ty, result_ty.clone(), arm_span));

        self.env.bindings = saved_bindings;
        Ok(())
    }

    /// Infer types for a binding.
    pub fn infer_binding(&mut self, binding: &Binding) -> Result<Type, TypeError> {
        let value_ty = self.infer_expr(&binding.value)?;

        // If there's a type annotation, unify with it
        if let Some(ty_expr) = &binding.ty {
            let annotated_ty = self.type_from_ast(ty_expr);
            self.unify(value_ty.clone(), annotated_ty.clone());
            // Also record for deferred check (batch unify errors are dropped)
            self.deferred_typed_binding_checks.push((
                value_ty.clone(),
                annotated_ty,
                binding.span,
            ));
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

        // Clear type parameter mappings to prevent cross-function pollution.
        // Each function gets fresh type variable mappings for its type parameters.
        self.type_param_mappings.clear();

        // Set type parameters in scope (e.g., [T: Hash] -> {"T"})
        // Also track their trait constraints for propagation into lambdas
        let saved_type_params = std::mem::take(&mut self.current_type_params);
        let saved_type_param_constraints = std::mem::take(&mut self.current_type_param_constraints);
        for tp in &func.type_params {
            self.current_type_params.insert(tp.name.node.clone());
            // Store constraints for this type param (e.g., T: Sizeable -> ["Sizeable"])
            let constraints: Vec<String> = tp.constraints.iter().map(|b| b.node.clone()).collect();
            if !constraints.is_empty() {
                self.current_type_param_constraints.insert(tp.name.node.clone(), constraints);
            }
        }

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
            for param_type in &mut param_types {
                *param_type = self.env.apply_subst(param_type);
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

        // Convert AST type params to types module TypeParams
        // This allows instantiate_function to properly replace TypeParam("T") with fresh vars
        let type_params: Vec<crate::TypeParam> = func.type_params.iter()
            .map(|tp| crate::TypeParam {
                name: tp.name.node.clone(),
                constraints: tp.constraints.iter().map(|c| c.node.clone()).collect(),
            })
            .collect();

        // Note: Type variables in param_types/ret_ty will be resolved after solve()
        // is called in check_module - see the post-processing step there

        let func_ty = FunctionType {
            type_params,
            params: param_types,
            ret: Box::new(ret_ty),
            required_params: if has_defaults { Some(required_count) } else { None },
        };

        // Register function in environment
        self.env.insert_function(name.clone(), func_ty.clone());

        // Restore previous current function and type parameters
        self.current_function = saved_current;
        self.current_type_params = saved_type_params;
        self.current_type_param_constraints = saved_type_param_constraints;

        Ok(func_ty)
    }

    /// Infer types for a function clause.
    fn infer_clause(&mut self, clause: &FnClause) -> Result<(Vec<Type>, Type), TypeError> {
        let saved_bindings = self.env.bindings.clone();
        let func_name = self.current_function.clone().unwrap_or_default();

        let mut param_types = Vec::new();
        for param in &clause.params {
            let param_ty = if let Some(ty_expr) = &param.ty {
                self.type_from_ast(ty_expr)
            } else {
                // Create fresh type variable for unannotated parameter
                let fresh_ty = self.fresh();
                // Track this var for better error messages if it causes conflicts
                if let Type::Var(var_id) = fresh_ty {
                    let param_name = Self::extract_pattern_name(&param.pattern);
                    self.unannotated_param_vars.insert(var_id, (func_name.clone(), param_name));
                }
                fresh_ty
            };
            self.infer_pattern(&param.pattern, &param_ty)?;

            // Defer type-check of default value against parameter type.
            // We can't unify immediately because in batch inference (shared InferCtx),
            // unifying would over-constrain generic params (e.g., f = helper would lock
            // f to helper's concrete type, preventing other function args).
            // Instead, infer the default's type and check compatibility after solve().
            // If infer_expr fails (e.g., function reference not in scope during
            // per-function inference), skip the check - the body constrains the param type.
            if let Some(default_expr) = &param.default {
                if let Ok(default_ty) = self.infer_expr(default_expr) {
                    let span = default_expr.span();
                    self.deferred_default_param_checks.push((default_ty, param_ty.clone(), span));
                }
            }

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
        // First pass: collect type definitions, function signatures, and mvar types
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
                Item::MvarDef(mvar) => {
                    // Mvar has a required type annotation - register it early so functions can see it
                    let annotated_ty = self.type_from_ast(&mvar.ty);
                    self.env.bind(mvar.name.node.clone(), annotated_ty, true);
                }
                _ => {}
            }
        }

        // Second pass: infer function bodies and mvar initializers
        for item in &module.items {
            match item {
                Item::FnDef(fd) => {
                    self.infer_function(fd)?;
                }
                Item::Binding(binding) => {
                    self.infer_binding(binding)?;
                }
                Item::MvarDef(mvar) => {
                    // Verify mvar initializer matches the annotated type
                    let annotated_ty = self.type_from_ast(&mvar.ty);
                    let value_ty = self.infer_expr(&mvar.value)?;
                    self.unify(value_ty, annotated_ty);
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

        // Set up current_type_params so type_from_ast recognizes type parameters
        // e.g., for "type Tree[T] = Node(T, Tree[T])", "T" should become TypeParam("T")
        let saved_type_params = std::mem::take(&mut self.current_type_params);
        for p in &params {
            self.current_type_params.insert(p.name.clone());
        }

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
                // Restore type params before early return
                self.current_type_params = saved_type_params;
                return Ok(()); // Never type
            }
        };

        // Restore previous type params
        self.current_type_params = saved_type_params;

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
    use nostos_syntax::ast::{Span, Visibility};
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
                    visibility: Visibility::Private,
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
                    visibility: Visibility::Private,
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
                visibility: Visibility::Private,
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

    #[test]
    fn test_infer_lambda_return_type_annotation_mismatch() {
        // f: () -> Int = () => "hello"
        // The annotation says return Int, but lambda body returns String
        let mut env = TypeEnv::new();
        let expr = Expr::Block(
            vec![Stmt::Let(Binding {
                visibility: Visibility::Private,
                mutable: false,
                pattern: Pattern::Var(ident("f")),
                ty: Some(TypeExpr::Function(
                    vec![],
                    Box::new(TypeExpr::Name(ident("Int"))),
                )),
                value: Expr::Lambda(
                    vec![],
                    Box::new(Expr::String(nostos_syntax::ast::StringLit::Plain("hello".to_string()), span())),
                    span(),
                ),
                span: span(),
            })],
            span(),
        );
        let result = infer_expr_type(&mut env, &expr);
        // This should detect that Int != String
        assert!(matches!(result, Err(TypeError::UnificationFailed(_, _))),
            "Expected UnificationFailed error for lambda return type mismatch, got: {:?}", result);
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
                    visibility: Visibility::Private,
                    mutable: false,
                    pattern: Pattern::Var(ident("x")),
                    ty: None,
                    value: Expr::Int(1, span()),
                    span: span(),
                }),
                Stmt::Let(Binding {
                    visibility: Visibility::Private,
                    mutable: false,
                    pattern: Pattern::Var(ident("y")),
                    ty: None,
                    value: Expr::Int(2, span()),
                    span: span(),
                }),
                Stmt::Let(Binding {
                    visibility: Visibility::Private,
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

