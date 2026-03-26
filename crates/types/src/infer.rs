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

use crate::{Constructor, FunctionType, RecordType, Type, TypeDef, TypeError, TypeEnv, TypeParam};
use nostos_syntax::ast::{
    BinOp, Binding, CallArg, Expr, FnClause, FnDef, Item, MatchArm, Module, Pattern, RecordField,
    RecordPatternField, Span, Stmt, TypeExpr, UnaryOp, VariantPatternFields,
};

/// Check if an expression contains a reference to a variable with the given name.
/// Used to detect recursive lambda bindings (e.g., `f = (x) => ... f(x-1) ...`).
fn expr_references_name(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Var(ident) => ident.node == name,
        Expr::BinOp(l, _, r, _) | Expr::Index(l, r, _) | Expr::Send(l, r, _) => {
            expr_references_name(l, name) || expr_references_name(r, name)
        }
        Expr::UnaryOp(_, e, _) | Expr::FieldAccess(e, _, _) | Expr::Try_(e, _)
        | Expr::Quote(e, _) | Expr::Splice(e, _)
        | Expr::TypeAscription(e, _, _) => expr_references_name(e, name),
        Expr::Call(callee, _, args, _) => {
            expr_references_name(callee, name)
                || args.iter().any(|a| call_arg_references_name(a, name))
        }
        Expr::MethodCall(recv, _, args, _) => {
            expr_references_name(recv, name)
                || args.iter().any(|a| call_arg_references_name(a, name))
        }
        Expr::Lambda(_, body, _) => expr_references_name(body, name),
        Expr::If(cond, then_, else_, _) => {
            expr_references_name(cond, name)
                || expr_references_name(then_, name)
                || expr_references_name(else_, name)
        }
        Expr::Match(scrutinee, arms, _) | Expr::Try(scrutinee, arms, _, _) => {
            expr_references_name(scrutinee, name)
                || arms.iter().any(|arm| expr_references_name(&arm.body, name)
                    || arm.guard.as_ref().map_or(false, |g| expr_references_name(g, name)))
        }
        Expr::Receive(arms, timeout, _) => {
            arms.iter().any(|arm| expr_references_name(&arm.body, name)
                || arm.guard.as_ref().map_or(false, |g| expr_references_name(g, name)))
            || timeout.as_ref().map_or(false, |(t, b)| {
                expr_references_name(t, name) || expr_references_name(b, name)
            })
        }
        Expr::Tuple(exprs, _) | Expr::List(exprs, _, _) | Expr::Set(exprs, _) => {
            exprs.iter().any(|e| expr_references_name(e, name))
        }
        Expr::Map(pairs, _) => {
            pairs.iter().any(|(k, v)| expr_references_name(k, name) || expr_references_name(v, name))
        }
        Expr::Block(stmts, _) => stmts.iter().any(|s| stmt_references_name(s, name)),
        Expr::Record(_, fields, _) => {
            fields.iter().any(|f| record_field_references_name(f, name))
        }
        Expr::RecordUpdate(_, base, fields, _) => {
            expr_references_name(base, name)
                || fields.iter().any(|f| record_field_references_name(f, name))
        }
        Expr::While(cond, body, _) => {
            expr_references_name(cond, name) || expr_references_name(body, name)
        }
        Expr::For(_, start, end, body, _) => {
            expr_references_name(start, name)
                || expr_references_name(end, name)
                || expr_references_name(body, name)
        }
        Expr::Spawn(_, callee, args, _) => {
            expr_references_name(callee, name)
                || args.iter().any(|a| expr_references_name(a, name))
        }
        Expr::Do(do_stmts, _) => {
            do_stmts.iter().any(|ds| match ds {
                nostos_syntax::ast::DoStmt::Bind(_, e) | nostos_syntax::ast::DoStmt::Expr(e) => {
                    expr_references_name(e, name)
                }
            })
        }
        Expr::Return(Some(e), _) | Expr::Break(Some(e), _) => expr_references_name(e, name),
        // Literals and other terminals don't reference names
        _ => false,
    }
}

fn call_arg_references_name(arg: &CallArg, name: &str) -> bool {
    match arg {
        CallArg::Positional(e) | CallArg::Named(_, e) => expr_references_name(e, name),
    }
}

fn record_field_references_name(field: &RecordField, name: &str) -> bool {
    match field {
        RecordField::Positional(e) | RecordField::Named(_, e) => expr_references_name(e, name),
    }
}

fn stmt_references_name(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::Expr(e) => expr_references_name(e, name),
        Stmt::Let(b) => expr_references_name(&b.value, name),
        Stmt::Assign(_, e, _) => expr_references_name(e, name),
        Stmt::LocalFnDef(fn_def) => fn_def.clauses.iter().any(|c| expr_references_name(&c.body, name)),
    }
}
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// List methods that uniquely identify the receiver as a List type.
/// Excludes methods shared with String (reverse, take, drop) and
/// methods shared with Option/Result (map, flatMap).
static EXCLUSIVE_LIST_METHODS: LazyLock<HashSet<&str>> = LazyLock::new(|| {
    HashSet::from([
        "filter", "fold", "any", "all", "find",
        "sort", "sortBy", "head", "tail", "init", "last",
        "sum", "product", "zip", "unzip",
        "unique", "flatten", "position",
        // Note: "indexOf" is NOT here - it's also valid on String (String.indexOf returns Int)
        // so it doesn't uniquely identify a List receiver.
        "push", "pop", "nth", "slice",
        "scanl", "foldl", "foldr", "enumerate", "intersperse",
        "spanList", "groupBy", "transpose", "pairwise", "isSorted",
        "isSortedBy", "maximum", "minimum", "takeWhile", "dropWhile",
        "partition", "zipWith",
    ])
});

/// Map methods that uniquely identify the receiver as a Map type.
/// Note: "insert" is Map-exclusive only when called with 3 args (receiver + key + value);
/// with 2 args it's a Set method.
static EXCLUSIVE_MAP_METHODS: LazyLock<HashSet<&str>> = LazyLock::new(|| {
    HashSet::from(["lookup", "keys", "values", "getOrThrow", "toList"])
});

/// String methods that uniquely identify the receiver as a String type.
/// These methods exist ONLY on String (not on List/Map/Set/Int/Float), so when an
/// unresolved receiver calls one of these methods, we can infer the receiver is String
/// and properly constrain the return type (e.g., chars() -> [Char]).
/// NOTE: toInt/toFloat/parseInt/parseFloat are NOT included because they also exist
/// on Int/Float as numeric conversion methods.
static EXCLUSIVE_STRING_METHODS: LazyLock<HashSet<&str>> = LazyLock::new(|| {
    HashSet::from([
        "chars", "toUpper", "toLower", "trim", "trimStart", "trimEnd",
        "startsWith", "endsWith", "replace", "replaceAll",
        "repeat", "padStart", "padEnd", "lines", "words",
        "lastIndexOf", "substring",
    ])
});

/// Option method aliasing: short method name → stdlib function name.
/// Supports both shorthand (.map) and direct (.optMap) names.
static OPTION_METHOD_ALIASES: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    HashMap::from([
        ("map", "optMap"), ("optMap", "optMap"),
        ("flatMap", "optFlatMap"), ("optFlatMap", "optFlatMap"),
        ("unwrap", "optUnwrap"), ("optUnwrap", "optUnwrap"),
        ("unwrapOr", "optUnwrapOr"), ("optUnwrapOr", "optUnwrapOr"),
        ("isSome", "optIsSome"), ("optIsSome", "optIsSome"),
        ("isNone", "optIsNone"), ("optIsNone", "optIsNone"),
    ])
});

/// Result method aliasing: short method name → stdlib function name.
/// Supports both shorthand (.map) and direct (.resMap) names.
static RESULT_METHOD_ALIASES: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    HashMap::from([
        ("map", "resMap"), ("resMap", "resMap"),
        ("mapErr", "resMapErr"), ("resMapErr", "resMapErr"),
        ("flatMap", "resFlatMap"), ("resFlatMap", "resFlatMap"),
        ("unwrap", "resUnwrap"), ("resUnwrap", "resUnwrap"),
        ("unwrapOr", "resUnwrapOr"), ("resUnwrapOr", "resUnwrapOr"),
        ("isOk", "resIsOk"), ("resIsOk", "resIsOk"),
        ("isErr", "resIsErr"), ("resIsErr", "resIsErr"),
        ("toOption", "resToOption"), ("resToOption", "resToOption"),
    ])
});

fn is_exclusive_list_method(name: &str) -> bool {
    EXCLUSIVE_LIST_METHODS.contains(name)
}

fn is_exclusive_map_method(name: &str) -> bool {
    EXCLUSIVE_MAP_METHODS.contains(name)
}

fn is_exclusive_string_method(name: &str) -> bool {
    EXCLUSIVE_STRING_METHODS.contains(name)
}

fn resolve_option_method_alias(name: &str) -> Option<&'static str> {
    OPTION_METHOD_ALIASES.get(name).copied()
}

fn resolve_result_method_alias(name: &str) -> Option<&'static str> {
    RESULT_METHOD_ALIASES.get(name).copied()
}

/// Check if a type contains any unresolved type variables (Var or TypeParam).
/// Used to avoid false positive structural mismatch errors when HM inference
/// has only partially resolved types (e.g., higher-order function parameters).
fn has_unresolved_vars(ty: &Type) -> bool {
    match ty {
        Type::Var(_) | Type::TypeParam(_) => true,
        Type::List(inner) | Type::Set(inner) | Type::IO(inner) | Type::Array(inner) => {
            has_unresolved_vars(inner)
        }
        Type::Map(k, v) => has_unresolved_vars(k) || has_unresolved_vars(v),
        Type::Tuple(elems) => elems.iter().any(|e| has_unresolved_vars(e)),
        Type::Named { args, .. } => args.iter().any(|a| has_unresolved_vars(a)),
        Type::Function(ft) => {
            ft.params.iter().any(|p| has_unresolved_vars(p)) || has_unresolved_vars(&ft.ret)
        }
        _ => false,
    }
}

/// Check if a type contains any "external anchor" for let-polymorphism analysis.
/// An external anchor is a concrete type (Int, String, Named, etc.) or a type variable
/// with ID less than `vars_before` (i.e., from the outer scope before lambda inference).
/// This is used to determine whether an Equal constraint pins lambda-internal vars
/// to external types, preventing generalization.
fn type_has_external_anchor(ty: &Type, vars_before: u32) -> bool {
    match ty {
        Type::Var(id) => *id < vars_before,
        Type::TypeParam(_) => true, // TypeParams are external
        Type::Int | Type::Float | Type::String | Type::Bool | Type::Unit
        | Type::Char => true,
        Type::Named { args, .. } => {
            // A named type with no type args is concrete (e.g., Color).
            // A named type with all-internal type args (e.g., Option[?N]) is not external.
            args.is_empty() || args.iter().any(|a| type_has_external_anchor(a, vars_before))
        }
        Type::List(inner) | Type::Set(inner) | Type::IO(inner) | Type::Array(inner) => {
            type_has_external_anchor(inner, vars_before)
        }
        Type::Map(k, v) => type_has_external_anchor(k, vars_before) || type_has_external_anchor(v, vars_before),
        Type::Tuple(elems) => elems.iter().any(|e| type_has_external_anchor(e, vars_before)),
        Type::Function(ft) => {
            ft.params.iter().any(|p| type_has_external_anchor(p, vars_before))
                || type_has_external_anchor(&ft.ret, vars_before)
        }
        _ => false,
    }
}

/// A type constraint generated during inference.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two types must be equal (with optional span for error reporting)
    Equal(Type, Type, Option<Span>),
    /// A type must implement a trait (with optional span for error reporting)
    HasTrait(Type, String, Option<Span>),
    /// A type must have a field (with optional span for error reporting)
    HasField(Type, String, Type, Option<Span>),
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
    /// (before substitution), the trait name, and optional span for error reporting.
    /// Used in post-solve to retry constraints that were deferred because the type
    /// was still a Var.
    /// Unlike trait_bounds (which includes merged bounds from variable unification),
    /// this tracks only direct HasTrait constraints, avoiding false positives.
    deferred_has_trait: Vec<(Type, String, Option<Span>)>,
    /// Deferred HasField constraints: type, field name, expected field type, optional span.
    /// These are saved when the type is still unresolved (Var) so they can be
    /// re-checked after check_pending_method_calls resolves more types.
    deferred_has_field: Vec<(Type, String, Type, Option<Span>)>,
    /// Deferred length/len checks: the argument type and call span.
    /// length() only works on collections (List, String, Map, Set, arrays).
    deferred_length_checks: Vec<(Type, Span)>,
    /// Deferred concat (++) checks: (left_type, right_type, span).
    /// ++ only works on String ++ String or List[T] ++ List[T].
    deferred_concat_checks: Vec<(Type, Type, Span)>,
    /// Var IDs that are the result type of a HasField constraint (field access).
    /// When `x.name` creates HasField(?X, "name", ?Y), ?Y's ID is recorded here.
    /// In BinOp::Add, we avoid eagerly unifying field-result Vars with Int,
    /// so the Num trait bound can be checked after HasField resolves the actual type.
    field_result_vars: HashSet<u32>,
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
    #[allow(clippy::type_complexity)]
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
    /// Deferred index access checks: (container_type, index_type, elem_type, span, literal_index).
    /// When an Index expression's container is an unresolved Var, defer the check
    /// until after solve() resolves the container type. This correctly handles
    /// Map[K,V] indexing where m["key"] should return V, not assume List indexing.
    /// The optional literal_index stores the compile-time integer index for tuple access.
    deferred_index_checks: Vec<(Type, Type, Type, Span, Option<i64>)>,
    /// Known implicit conversion function names (e.g., "tensorFromList").
    /// Populated by the compiler from functions matching the naming convention.
    known_implicit_fns: HashSet<String>,
    /// Implicit conversions detected during solve(): (span, conversion_fn_name).
    /// When a type mismatch is resolved by an implicit conversion, the span
    /// of the argument expression and the conversion function name are recorded.
    pub implicit_conversions: Vec<(Span, String)>,
    /// Direct clause return type mappings: pre_reg_ret_var_id → clause_ret_type.
    /// When infer_function unifies a clause's return type with a pre-registered
    /// return type variable, the mapping is stored here as a fallback for enrichment.
    /// This is necessary because solve() may exit early on errors from OTHER
    /// functions, leaving the deferred constraint unprocessed and the pre-reg
    /// ret var unresolved in the substitution.
    pub clause_ret_types: HashMap<u32, Type>,
    /// Direct mapping from pre-registered param Var IDs to their resolved param types.
    /// Used as fallback during enrichment when solve() exits early, similar to clause_ret_types.
    pub clause_param_types: HashMap<u32, Type>,
    /// Deferred overload calls: when multiple overloads tie because arg types are
    /// unresolved Vars, defer the overload selection until after constraint solving.
    /// Format: (overloads, arg_types, return_type_var, span).
    #[allow(clippy::type_complexity)]
    deferred_overload_calls: Vec<(Vec<FunctionType>, Vec<Type>, Type, Span)>,
    /// Var IDs created by instantiate_function (from named function references).
    /// These are safe to freshen in instantiate_local_binding for let-polymorphism,
    /// because they don't have body constraints in the current inference scope.
    /// Vars from lambda body inference are NOT in this set and should not be freshened.
    polymorphic_vars: HashSet<u32>,
    /// Internal Equal constraints from let-bound lambda bodies that involve
    /// only polymorphic (generalizable) vars. When instantiate_local_binding
    /// freshens these vars, it replicates these constraints with fresh copies
    /// so that `(a, b) => a + b; addUp(1, "hello")` catches the type mismatch.
    polymorphic_body_constraints: Vec<(Type, Type)>,
    /// Internal pending method calls from let-bound lambda bodies that involve
    /// polymorphic (generalizable) vars. When instantiate_local_binding
    /// freshens these vars, it replicates these method calls with fresh vars
    /// so that `first = (xs) => xs.head()` can be used at multiple types.
    polymorphic_body_method_calls: Vec<PendingMethodCall>,
    /// Internal deferred overload calls from let-bound lambda bodies that involve
    /// polymorphic vars. These must be replicated with fresh vars during
    /// instantiation, otherwise the deferred resolution during solve() would
    /// lock the original vars to a concrete type. E.g., `showAny = (x) => show(x)`
    /// where show has overloads for Int, Bool, String etc.
    #[allow(clippy::type_complexity)]
    polymorphic_body_deferred_overloads: Vec<(Vec<FunctionType>, Vec<Type>, Type, Span)>,
    /// Freshened var pairs from instantiate_local_binding: (original_var_id, fresh_var_id, call_span).
    /// After solve(), if original and fresh both resolve to concrete but different types,
    /// the freshening was incorrect (let-polymorphism was applied to a monomorphic binding).
    freshened_binding_vars: Vec<(u32, u32, Span)>,
    /// The declared return type of the function currently being inferred.
    /// Used to disambiguate constructor names when multiple types have the same constructor.
    /// E.g., if both IntTree and StrTree have a `Node` constructor, and the function returns
    /// `StrTree`, prefer `StrTree.Node` over `IntTree.Node`.
    current_declared_return_type: Option<String>,
    /// Stack of type variables for the current loop's break value type.
    /// When a while/for loop is entered, a fresh type var is pushed.
    /// `break value` unifies its value type with the top of this stack.
    /// The while/for loop's return type is this var (resolved after solve).
    loop_break_type_stack: Vec<Type>,
    /// Stack of type variables for the current function's return type.
    /// When a function clause is entered, a fresh type var is pushed.
    /// `return expr` unifies its value type with the top of this stack,
    /// so early returns contribute to the function's inferred return type.
    fn_return_type_stack: Vec<Type>,
    /// Stack of booleans tracking whether any `return` statement was encountered
    /// in the current function clause. Paired 1:1 with fn_return_type_stack.
    /// When true, the function's return type should come from fn_return_type_stack
    /// rather than from the body's fallthrough type.
    fn_has_return_stack: Vec<bool>,
}

use crate::is_structural_mismatch;

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
            field_result_vars: HashSet::new(),
            deferred_collection_ret_checks: Vec::new(),
            deferred_fn_call_checks: Vec::new(),
            deferred_generic_trait_checks: Vec::new(),
            deferred_default_param_checks: Vec::new(),
            deferred_method_on_var: Vec::new(),
            deferred_method_existence_checks: Vec::new(),
            deferred_indirect_call_checks: Vec::new(),
            deferred_typed_binding_checks: Vec::new(),
            deferred_branch_type_checks: Vec::new(),
            deferred_index_checks: Vec::new(),
            deferred_overload_calls: Vec::new(),
            known_implicit_fns: HashSet::new(),
            implicit_conversions: Vec::new(),
            clause_ret_types: HashMap::new(),
            clause_param_types: HashMap::new(),
            polymorphic_vars: HashSet::new(),
            polymorphic_body_constraints: Vec::new(),
            polymorphic_body_method_calls: Vec::new(),
            polymorphic_body_deferred_overloads: Vec::new(),
            freshened_binding_vars: Vec::new(),
            current_declared_return_type: None,
            loop_break_type_stack: Vec::new(),
            fn_return_type_stack: Vec::new(),
            fn_has_return_stack: Vec::new(),
        }
    }


    /// Set the known implicit conversion function names.
    /// These follow the naming convention: {targetTypeLower}From{SourceTypeShort}
    /// e.g., "tensorFromList" converts List[Float] -> Tensor.
    pub fn set_known_implicit_fns(&mut self, fns: HashSet<String>) {
        self.known_implicit_fns = fns;
    }

    /// Try to find an implicit conversion function for a type mismatch.
    /// Returns the function name (e.g., "tensorFromList") if a valid conversion exists.
    ///
    /// Convention: {targetTypeLower}From{sourceTypeShort}
    ///   List[Float] -> Tensor:  tensorFromList
    ///   List[Int]   -> Tensor:  tensorFromIntList
    ///   Int         -> Tensor:  tensorFromInt
    fn find_implicit_conversion(&self, t1: &Type, t2: &Type) -> Option<String> {
        if self.known_implicit_fns.is_empty() {
            return None;
        }

        // Try both directions: (target, source) and (source, target)
        self.try_conversion(t1, t2)
            .or_else(|| self.try_conversion(t2, t1))
    }

    /// Try to find a conversion function from `source` to `target`.
    fn try_conversion(&self, target: &Type, source: &Type) -> Option<String> {
        let target_name = match target {
            Type::Named { name, .. } => name.clone(),
            _ => return None,
        };
        let target_lower = {
            let mut s = target_name.clone();
            if let Some(first) = s.get_mut(..1) {
                first.make_ascii_lowercase();
            }
            s
        };

        let source_short = match source {
            Type::List(elem) => {
                // List[Float] -> "List", List[Int] -> "IntList"
                match elem.as_ref() {
                    Type::Int | Type::Int64 => "IntList".to_string(),
                    Type::Float | Type::Float64 => "List".to_string(),
                    Type::String => "StringList".to_string(),
                    Type::Bool => "BoolList".to_string(),
                    _ => "List".to_string(),
                }
            }
            Type::Int | Type::Int64 => "Int".to_string(),
            Type::Float | Type::Float64 => "Float".to_string(),
            Type::String => "String".to_string(),
            Type::Bool => "Bool".to_string(),
            _ => return None,
        };

        let fn_name = format!("{}From{}", target_lower, source_short);
        if self.known_implicit_fns.contains(&fn_name) {
            Some(fn_name)
        } else {
            None
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

    /// Get trait bounds for Named types that were unified with Vars carrying trait bounds.
    /// When annotations use `p: Pair[a, b]`, HM inference stores `a` as `Named("a")`.
    /// If `a` is used with operations requiring traits (e.g., `<` needing Ord),
    /// the trait bound is on the Var that was unified with Named("a"), not directly on it.
    /// This method returns (named_type_name, trait_name) pairs for such cases.
    pub fn get_trait_bounds_for_named_type_params(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();
        for (&var_id, bounds) in &self.trait_bounds {
            let resolved = self.env.apply_subst(&Type::Var(var_id));
            if let Type::Named { name, args } = &resolved {
                // Only for single lowercase letter names with no args (type params from annotations)
                if args.is_empty() && name.len() == 1 {
                    let ch = name.chars().next().unwrap();
                    if ch.is_ascii_lowercase() {
                        for bound in bounds {
                            result.push((name.clone(), bound.clone()));
                        }
                    }
                }
            }
        }
        result
    }

    /// Get unresolved HasField constraints (field access on still-generic type vars).
    /// Used to propagate field requirements from generic function bodies to call sites.
    pub fn get_deferred_has_field(&self) -> &[(Type, String, Type, Option<Span>)] {
        &self.deferred_has_field
    }

    /// Get deferred method-on-var calls (method calls on still-generic type vars).
    /// Used to propagate method requirements from generic function bodies to call sites.
    #[allow(clippy::type_complexity)]
    pub fn get_deferred_method_on_var(&self) -> &[(Type, String, Vec<Type>, Type, Option<Span>)] {
        &self.deferred_method_on_var
    }

    /// Look up the Var type that a TypeParam name maps to.
    pub fn get_type_param_mapping(&self, name: &str) -> Option<Type> {
        self.type_param_mappings.get(name).cloned()
    }

    /// Get deferred HasTrait constraints (type, trait_name, span triples).
    /// Used by try_hm_inference to discover trait bounds on TypeParams
    /// that were deferred during solve() rather than stored in trait_bounds.
    pub fn get_deferred_has_trait(&self) -> &[(Type, String, Option<Span>)] {
        &self.deferred_has_trait
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
        // Eagerly instantiate TypeParams to function-scoped fresh vars before storing.
        // Without this, TypeParam("a") from different functions shares the same name
        // and gets merged by type_param_mappings during solve(), causing spurious
        // type errors like "expected (a, Stack[a]), found Int".
        let t1 = if self.contains_type_param(&t1) { self.instantiate_type_params(&t1) } else { t1 };
        let t2 = if self.contains_type_param(&t2) { self.instantiate_type_params(&t2) } else { t2 };
        self.constraints.push(Constraint::Equal(t1, t2, None));
    }

    /// Add an equality constraint with span information for precise error reporting.
    pub fn unify_at(&mut self, t1: Type, t2: Type, span: Span) {
        // Eagerly instantiate TypeParams to function-scoped fresh vars before storing.
        // Same reasoning as unify() above.
        let t1 = if self.contains_type_param(&t1) { self.instantiate_type_params(&t1) } else { t1 };
        let t2 = if self.contains_type_param(&t2) { self.instantiate_type_params(&t2) } else { t2 };
        self.constraints.push(Constraint::Equal(t1, t2, Some(span)));
    }

    /// Add a trait constraint.
    pub fn require_trait(&mut self, ty: Type, trait_name: &str) {
        self.constraints
            .push(Constraint::HasTrait(ty, trait_name.to_string(), None));
    }

    /// Add a trait constraint with span information for precise error reporting.
    pub fn require_trait_at(&mut self, ty: Type, trait_name: &str, span: Span) {
        self.constraints
            .push(Constraint::HasTrait(ty, trait_name.to_string(), Some(span)));
    }

    /// Add a field constraint.
    pub fn require_field(&mut self, ty: Type, field: &str, field_ty: Type) {
        self.require_field_at(ty, field, field_ty, None);
    }

    /// Add a field constraint with span for precise error reporting.
    pub fn require_field_at(&mut self, ty: Type, field: &str, field_ty: Type, span: Option<Span>) {
        // Track the result Var ID so BinOp::Add can avoid eager unification
        // on field-result variables (prevents ?Y=Int before HasField resolves ?Y=String)
        if let Type::Var(id) = &field_ty {
            self.field_result_vars.insert(*id);
        }
        self.constraints
            .push(Constraint::HasField(ty, field.to_string(), field_ty, span));
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
    /// Returns (index, is_ambiguous): the INDEX of the best-matching overload,
    /// and whether the match was ambiguous (multiple overloads tied at a low score,
    /// typically because arg types are unresolved Vars).
    fn find_best_overload_idx(&self, overloads: &[&FunctionType], arg_types: &[Type]) -> (Option<usize>, bool) {
        // Resolve argument types through current substitution
        let resolved_args: Vec<Type> = arg_types.iter()
            .map(|t| self.env.apply_subst(t))
            .collect();

        // Try each overload and score how well it matches
        let mut best_match: Option<(usize, usize)> = None;  // (index, score)
        let mut num_tied = 0usize;

        for (idx, &overload) in overloads.iter().enumerate() {
            let min_required = overload.required_params.unwrap_or(overload.params.len());
            if arg_types.len() < min_required || arg_types.len() > overload.params.len() {
                continue;
            }

            let mut score = 0;
            let mut compatible = true;

            // Only compare provided args (optional params may be omitted)
            for (param_ty, arg_ty) in overload.params.iter().take(arg_types.len()).zip(resolved_args.iter()) {
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
                // When scores are tied, prefer overloads with fewer type_params.
                // This ensures builtin functions (no type_params, e.g., range: Int -> Int -> [Int])
                // win over stdlib functions with the same name but generic return types
                // (e.g., stdlib.validation.range: Concat a => Int -> Int -> ((a, String) -> Option[String])).
                let is_better = match best_match {
                    Some((_, prev_score)) if score > prev_score => {
                        num_tied = 1;
                        true
                    }
                    Some((prev_idx, prev_score)) if score == prev_score => {
                        num_tied += 1;
                        // Tiebreaker: prefer overloads with fewer type_params (more concrete)
                        overload.type_params.len() < overloads[prev_idx].type_params.len()
                    }
                    Some(_) => false,
                    None => {
                        num_tied = 1;
                        true
                    }
                };
                if is_better {
                    best_match = Some((idx, score));
                }
            }
        }

        // Ambiguous if multiple overloads tied AND the best score is not perfect
        // AND the overloads have conflicting concrete types at some parameter position.
        // Generic overloads (TypeParam/Var params) don't conflict and shouldn't trigger deferral.
        let has_concrete_conflict = if num_tied > 1 {
            let arg_count = arg_types.len();
            let mut conflicting = false;
            // Helper: check if a type is fully generic (only contains TypeParam/Var leaves)
            fn is_fully_generic(ty: &Type) -> bool {
                match ty {
                    Type::Var(_) | Type::TypeParam(_) => true,
                    Type::List(inner) | Type::Set(inner) | Type::Array(inner) | Type::IO(inner) => is_fully_generic(inner),
                    Type::Map(k, v) => is_fully_generic(k) && is_fully_generic(v),
                    Type::Tuple(elems) => elems.iter().all(is_fully_generic),
                    Type::Named { args, .. } => args.iter().all(is_fully_generic),
                    Type::Function(ft) => ft.params.iter().all(is_fully_generic) && is_fully_generic(&ft.ret),
                    _ => false,
                }
            }
            for pos in 0..arg_count {
                let mut concrete_types: Vec<String> = Vec::new();
                for overload in overloads {
                    if pos < overload.params.len() {
                        let param = &overload.params[pos];
                        // Skip fully generic types (TypeParam/Var or containers of them).
                        // e.g., List[a] and List[T] are both generic and don't conflict.
                        if !is_fully_generic(param) {
                            concrete_types.push(param.display());
                        }
                    }
                }
                if concrete_types.len() >= 2 && concrete_types.windows(2).any(|w| w[0] != w[1]) {
                    conflicting = true;
                    break;
                }
            }
            conflicting
        } else {
            false
        };
        // When conflicting concrete overloads are tied, treat as ambiguous unless
        // the score is perfect (100 per arg = all types exactly resolved).
        // Score 75 = container(Var) match (e.g., List(Var) vs List(Int)), score 50 = bare Var.
        let is_ambiguous = num_tied > 1
            && best_match.map(|(_, s)| s < 100 * arg_types.len()).unwrap_or(false)
            && has_concrete_conflict;
        (best_match.map(|(idx, _)| idx), is_ambiguous)
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

            // Named types must match by name (handle module-qualified names)
            (Type::Named { name: pn, args: pa }, Type::Named { name: an, args: aa }) => {
                // Check exact match or base-name match for module-qualified types
                // e.g., "Line" matches "types.Line" and vice versa
                let names_match = pn == an || {
                    let pbase = pn.rsplit('.').next().unwrap_or(pn);
                    let abase = an.rsplit('.').next().unwrap_or(an);
                    pbase == abase
                };
                if names_match {
                    if pa.is_empty() && aa.is_empty() {
                        Some(90)
                    } else if pa.len() == aa.len() {
                        let mut total = 0;
                        for (p, a) in pa.iter().zip(aa.iter()) {
                            total += self.types_compatible(p, a)?;
                        }
                        Some(total / pa.len().max(1) + 50)
                    } else {
                        None
                    }
                } else {
                    None
                }
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

        // Propagate trait bounds from var_bounds (discovered during inference)
        // to the fresh vars. This enables trait bound propagation through call chains.
        for (old_id, bound) in &func_ty.var_bounds {
            if let Some(Type::Var(new_id)) = var_subst.get(old_id) {
                self.add_trait_bound(*new_id, bound.clone());
                self.constraints.push(Constraint::HasTrait(
                    Type::Var(*new_id),
                    bound.clone(),
                    None,
                ));
            }
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

        // Also create fresh vars for any TypeParam names used in params/ret but NOT in type_params.
        // This happens when a type param has no constraints (e.g., "b" in "HasField(0,b) a => a -> b"
        // where only "a" has constraints). Without this, TypeParam("b") would not be substituted,
        // disconnecting the return type from HasField results.
        fn collect_type_param_names(ty: &Type, names: &mut Vec<String>) {
            match ty {
                Type::TypeParam(name) => {
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
                Type::Function(ft) => {
                    for p in &ft.params {
                        collect_type_param_names(p, names);
                    }
                    collect_type_param_names(&ft.ret, names);
                }
                Type::List(e) | Type::Array(e) | Type::Set(e) | Type::IO(e) => {
                    collect_type_param_names(e, names);
                }
                Type::Map(k, v) => {
                    collect_type_param_names(k, names);
                    collect_type_param_names(v, names);
                }
                Type::Tuple(elems) => {
                    for e in elems {
                        collect_type_param_names(e, names);
                    }
                }
                Type::Named { args, .. } => {
                    for a in args {
                        collect_type_param_names(a, names);
                    }
                }
                Type::Record(rec) => {
                    for (_, t, _) in &rec.fields {
                        collect_type_param_names(t, names);
                    }
                }
                _ => {}
            }
        }
        let mut all_tp_names = Vec::new();
        for p in &func_ty.params {
            collect_type_param_names(p, &mut all_tp_names);
        }
        collect_type_param_names(&func_ty.ret, &mut all_tp_names);
        for tp_name in &all_tp_names {
            if !param_subst.contains_key(tp_name) {
                let fresh = self.fresh();
                param_subst.insert(tp_name.clone(), fresh);
            }
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
                        let mut qvar_map: HashMap<u32, Type> = HashMap::new();
                        let mut parsed_arg_types: Vec<Type> = Vec::new();
                        for s in encoded_arg_types {
                            parsed_arg_types.push(Self::parse_simple_type(s, &param_subst, &var_subst, &mut self.env.next_var, &mut qvar_map));
                        }

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
                        // Track field result var so BinOp::Add doesn't eagerly unify it
                        if let Type::Var(fid) = &field_ty {
                            self.field_result_vars.insert(*fid);
                        }
                        self.constraints.push(Constraint::HasField(
                            fresh_var.clone(),
                            field_name,
                            field_ty,
                            None,
                        ));
                        // Also store in trait_bounds so transfer section can copy to var_subst vars
                        self.add_trait_bound(*var_id, constraint.clone());
                    } else {
                        self.add_trait_bound(*var_id, constraint.clone());
                        // Push HasTrait for well-known builtin traits AND for user-defined traits
                        // that are registered in the type environment. This allows generic functions
                        // with user-defined trait bounds (e.g., [a: Transformer]) to validate that
                        // callers provide types implementing the trait.
                        let is_builtin_trait = matches!(constraint.as_str(), "Eq" | "Ord" | "Num" | "Concat" | "Hash" | "Show");
                        let is_known_user_trait = self.env.traits.contains_key(constraint.as_str())
                            || self.env.traits.keys().any(|k| k.ends_with(&format!(".{}", constraint)));
                        if is_builtin_trait || is_known_user_trait {
                            self.constraints.push(Constraint::HasTrait(
                                fresh_var.clone(),
                                constraint.clone(),
                                None,
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
                            // Skip if param_subst already points to the same var as var_subst.
                            // This means the main type_params loop (above) already processed
                            // constraints for this var — creating duplicate HasMethod/HasField
                            // PendingMethodCalls would cause double type wrapping (e.g., List[List[Val]]
                            // instead of List[Val]) because both get processed independently in
                            // check_pending_method_calls.
                            if param_var_id == var_subst_id {
                                continue;
                            }
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
                                    let mut qvar_map2: HashMap<u32, Type> = HashMap::new();
                                    let mut parsed_arg_types: Vec<Type> = Vec::new();
                                    for s in encoded_arg_types {
                                        parsed_arg_types.push(Self::parse_simple_type(s, &param_subst, &var_subst, &mut self.env.next_var, &mut qvar_map2));
                                    }

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
                                    // Track field result var so BinOp::Add doesn't eagerly unify it
                                    if let Type::Var(fid) = &field_ty {
                                        self.field_result_vars.insert(*fid);
                                    }
                                    self.constraints.push(Constraint::HasField(
                                        var_subst_var.clone(),
                                        field_name,
                                        field_ty,
                                        None,
                                    ));
                                } else {
                                    self.add_trait_bound(*var_subst_id, bound.clone());
                                    if matches!(bound.as_str(), "Eq" | "Ord" | "Num" | "Concat" | "Hash" | "Show") {
                                        self.constraints.push(Constraint::HasTrait(
                                            var_subst_var.clone(),
                                            bound.clone(),
                                            None,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Record all fresh Var IDs from instantiation as polymorphic.
        // These can safely be freshened in instantiate_local_binding because
        // they don't have body constraints in the current inference scope.
        for fresh_ty in var_subst.values().chain(param_subst.values()) {
            if let Type::Var(fresh_id) = fresh_ty {
                self.polymorphic_vars.insert(*fresh_id);
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
            var_bounds: vec![],
        })
    }

    /// Apply a variable substitution map to a type, replacing Var(id) with the
    /// mapped type if present. Used to replicate body constraints with fresh vars.
    fn apply_var_subst(&self, ty: &Type, subst: &HashMap<u32, Type>) -> Type {
        match ty {
            Type::Var(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
            Type::Function(ft) => Type::Function(FunctionType {
                params: ft.params.iter().map(|p| self.apply_var_subst(p, subst)).collect(),
                ret: Box::new(self.apply_var_subst(&ft.ret, subst)),
                type_params: ft.type_params.clone(),
                var_bounds: ft.var_bounds.clone(),
                required_params: ft.required_params,
            }),
            Type::List(inner) => Type::List(Box::new(self.apply_var_subst(inner, subst))),
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| self.apply_var_subst(e, subst)).collect()),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.apply_var_subst(a, subst)).collect(),
            },
            Type::Map(k, v) => Type::Map(
                Box::new(self.apply_var_subst(k, subst)),
                Box::new(self.apply_var_subst(v, subst)),
            ),
            Type::Set(inner) => Type::Set(Box::new(self.apply_var_subst(inner, subst))),
            Type::IO(inner) => Type::IO(Box::new(self.apply_var_subst(inner, subst))),
            Type::Array(inner) => Type::Array(Box::new(self.apply_var_subst(inner, subst))),
            _ => ty.clone(),
        }
    }

    /// Instantiate a local function binding for let-polymorphism.
    /// When a local binding has a function type with unresolved type variables
    /// (e.g., `always42 = constFn(42)` giving `?a -> Int`), each use of that
    /// binding gets fresh copies of the type variables. This prevents the first
    /// call from locking the parameter type for all subsequent calls.
    /// `exclude_binding` is the name of the binding being looked up, which should
    /// be excluded from the scope scan to avoid counting its own Vars as "shared".
    fn instantiate_local_binding(&mut self, ty: &Type, _exclude_binding: &str, call_span: Option<Span>) -> Type {
        let resolved = self.env.apply_subst(ty);

        // Only freshen types with unresolved Vars
        if !resolved.has_any_type_var() {
            return resolved;
        }

        // Must be a function type for let-polymorphism to apply
        if !matches!(resolved, Type::Function(_)) {
            return resolved;
        }

        // Collect all Var IDs in the resolved type
        let mut var_ids = Vec::new();
        self.collect_var_ids(&resolved, &mut var_ids);

        if var_ids.is_empty() {
            return resolved;
        }

        // Only freshen Vars that came from instantiate_function (polymorphic vars).
        // Vars from lambda body inference have pending constraints in the current
        // scope and must NOT be freshened, or those constraints get disconnected.
        let mut var_subst: HashMap<u32, Type> = HashMap::new();
        for var_id in &var_ids {
            if self.polymorphic_vars.contains(var_id) {
                let fresh = self.fresh();
                // Copy trait bounds from old var to fresh var, and push
                // HasTrait constraints so the solver will check them
                if let Type::Var(fresh_id) = fresh {
                    if let Some(bounds) = self.trait_bounds.get(var_id).cloned() {
                        for bound in &bounds {
                            self.add_trait_bound(fresh_id, bound.clone());
                        }
                        // Also push HasTrait constraints for the fresh var so
                        // deferred_has_trait processing will validate them.
                        // Without this, trait bounds on let-poly vars are silently
                        // dropped (e.g., `addUp = (a,b) => a+b; addUp(1,"hi")` gives
                        // runtime error instead of compile error).
                        for bound in &bounds {
                            self.constraints.push(Constraint::HasTrait(
                                Type::Var(fresh_id),
                                bound.clone(),
                                None,
                            ));
                        }
                    }
                    // Mark the fresh var as polymorphic too, so chained
                    // let-bindings (f = g where g = id) work correctly
                    self.polymorphic_vars.insert(fresh_id);
                }
                var_subst.insert(*var_id, fresh);
            }
        }

        if var_subst.is_empty() {
            return resolved; // All vars are shared with environment
        }

        // Replicate internal body constraints from the lambda with fresh var copies.
        // E.g., if `(a, b) => a + b` had Equal(?a, ?b) from the `+` operator,
        // generate Equal(?fresh_a, ?fresh_b) so type checking catches `addUp(1, "hello")`.
        //
        // IMPORTANT: Body constraints, PMCs, and deferred overloads may reference
        // polymorphic vars that are NOT in the function type signature. E.g., for
        // `g = (y) => f(y)` where f is let-poly, the body constraint has ?f_fresh.
        // For `transform = (xs) => xs.map(x => [x])`, the PMC contains inner lambda
        // vars ?x and ?elem. These must be freshened on each instantiation.
        // We extend var_subst with fresh vars for any such "hidden" polymorphic vars
        // found across ALL body state (constraints, PMCs, deferred overloads).
        {
            let mut extra_var_ids = Vec::new();
            // Collect from body constraints
            for (a, b) in &self.polymorphic_body_constraints {
                self.collect_var_ids(a, &mut extra_var_ids);
                self.collect_var_ids(b, &mut extra_var_ids);
            }
            // Collect from body method calls
            for pmc in &self.polymorphic_body_method_calls {
                self.collect_var_ids(&pmc.receiver_ty, &mut extra_var_ids);
                for arg in &pmc.arg_types {
                    self.collect_var_ids(arg, &mut extra_var_ids);
                }
                self.collect_var_ids(&pmc.ret_ty, &mut extra_var_ids);
            }
            // Collect from body deferred overloads
            for (_, arg_types, ret_ty, _) in &self.polymorphic_body_deferred_overloads {
                for arg in arg_types {
                    self.collect_var_ids(arg, &mut extra_var_ids);
                }
                self.collect_var_ids(ret_ty, &mut extra_var_ids);
            }
            for extra_id in extra_var_ids {
                if !var_subst.contains_key(&extra_id) && self.polymorphic_vars.contains(&extra_id) {
                    let fresh = self.fresh();
                    if let Type::Var(fresh_id) = fresh {
                        // Copy trait bounds
                        if let Some(bounds) = self.trait_bounds.get(&extra_id).cloned() {
                            for bound in &bounds {
                                self.add_trait_bound(fresh_id, bound.clone());
                            }
                            for bound in &bounds {
                                self.constraints.push(Constraint::HasTrait(
                                    Type::Var(fresh_id),
                                    bound.clone(),
                                    None,
                                ));
                            }
                        }
                        self.polymorphic_vars.insert(fresh_id);
                    }
                    var_subst.insert(extra_id, fresh);
                }
            }
        }

        // Replicate body constraints with fresh vars
        let body_constraints = self.polymorphic_body_constraints.clone();
        for (a, b) in &body_constraints {
            let fresh_a = self.apply_var_subst(a, &var_subst);
            let fresh_b = self.apply_var_subst(b, &var_subst);
            if fresh_a != *a || fresh_b != *b {
                self.constraints.push(Constraint::Equal(fresh_a, fresh_b, call_span));
            }
        }

        // Record freshened var pairs for post-solve verification.
        // If the original var resolves to a concrete type after solve(),
        // but the fresh var was used with a different type, we have a
        // type error that let-polymorphism obscured.
        if let Some(span) = call_span {
            for (orig_id, fresh_ty) in &var_subst {
                if let Type::Var(fresh_id) = fresh_ty {
                    self.freshened_binding_vars.push((*orig_id, *fresh_id, span));
                }
            }
        }

        // Duplicate internal pending method calls with fresh vars.
        // E.g., if `first = (xs) => xs.head()` had a PMC with receiver ?xs,
        // generate a new PMC with fresh ?xs so that multiple calls to `first`
        // at different types each get their own method call validation.
        let body_pmcs = self.polymorphic_body_method_calls.clone();
        for pmc in &body_pmcs {
            let fresh_receiver = self.apply_var_subst(&pmc.receiver_ty, &var_subst);
            let fresh_args: Vec<Type> = pmc.arg_types.iter()
                .map(|a| self.apply_var_subst(a, &var_subst))
                .collect();
            let fresh_ret = self.apply_var_subst(&pmc.ret_ty, &var_subst);
            // Only push if something was actually substituted (avoid duplicates)
            if fresh_receiver != pmc.receiver_ty || fresh_ret != pmc.ret_ty
                || fresh_args != pmc.arg_types {
                self.pending_method_calls.push(PendingMethodCall {
                    receiver_ty: fresh_receiver,
                    method_name: pmc.method_name.clone(),
                    arg_types: fresh_args,
                    named_args: pmc.named_args.clone(),
                    ret_ty: fresh_ret,
                    span: pmc.span,
                });
            }
        }

        // Duplicate deferred overload calls with fresh vars.
        // E.g., if `showAny = (x) => show(x)` had a deferred overload for show
        // with arg_types=[?x], generate a new deferred overload with [?fresh_x]
        // so each call to showAny gets its own overload resolution.
        let body_docs = self.polymorphic_body_deferred_overloads.clone();
        for (overloads, arg_types, ret_ty, span) in &body_docs {
            let fresh_args: Vec<Type> = arg_types.iter()
                .map(|a| self.apply_var_subst(a, &var_subst))
                .collect();
            let fresh_ret = self.apply_var_subst(ret_ty, &var_subst);
            if fresh_args != *arg_types || fresh_ret != *ret_ty {
                self.deferred_overload_calls.push((
                    overloads.clone(),
                    fresh_args,
                    fresh_ret,
                    *span,
                ));
            }
        }

        let empty_param_subst: HashMap<String, Type> = HashMap::new();
        self.freshen_type(&resolved, &var_subst, &empty_param_subst)
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

    /// Collect unique TypeParam names from a type.
    fn collect_type_param_names(ty: &Type, names: &mut Vec<String>) {
        match ty {
            Type::TypeParam(name) => {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
            Type::List(elem) | Type::Array(elem) | Type::Set(elem) | Type::IO(elem) => {
                Self::collect_type_param_names(elem, names);
            }
            Type::Map(k, v) => {
                Self::collect_type_param_names(k, names);
                Self::collect_type_param_names(v, names);
            }
            Type::Tuple(elems) => {
                for elem in elems {
                    Self::collect_type_param_names(elem, names);
                }
            }
            Type::Function(ft) => {
                for p in &ft.params {
                    Self::collect_type_param_names(p, names);
                }
                Self::collect_type_param_names(&ft.ret, names);
            }
            Type::Named { args, .. } => {
                for arg in args {
                    Self::collect_type_param_names(arg, names);
                }
            }
            Type::Record(rec) => {
                for (_, field_ty, _) in &rec.fields {
                    Self::collect_type_param_names(field_ty, names);
                }
            }
            _ => {}
        }
    }

    /// Parse a simple type string from a signature encoding back into a Type.
    /// Handles primitives (Int, String, Bool, etc.), single-letter type params,
    /// and basic compound types (List[X], (X, Y)).
    fn parse_simple_type(
        s: &str,
        param_subst: &HashMap<String, Type>,
        var_subst: &HashMap<u32, Type>,
        next_var: &mut u32,
        qvar_map: &mut HashMap<u32, Type>,
    ) -> Type {
        let s = s.trim();

        // Primitives
        match s {
            "Int" => return Type::Int,
            "String" => return Type::String,
            "Bool" => return Type::Bool,
            "Float" => return Type::Float,
            "Char" => return Type::Char,
            "()" => return Type::Unit,
            _ => {}
        }

        // List[X]
        if s.starts_with("List[") && s.ends_with(']') {
            let inner = &s[5..s.len() - 1];
            return Type::List(Box::new(Self::parse_simple_type(inner, param_subst, var_subst, next_var, qvar_map)));
        }

        // Map[K, V]
        if s.starts_with("Map[") && s.ends_with(']') {
            let inner = &s[4..s.len() - 1];
            // Split on comma at depth 0
            let mut depth = 0i32;
            let mut split_pos = None;
            for (i, ch) in inner.char_indices() {
                match ch {
                    '[' | '(' => depth += 1,
                    ']' | ')' => depth -= 1,
                    ',' if depth == 0 => { split_pos = Some(i); break; }
                    _ => {}
                }
            }
            if let Some(pos) = split_pos {
                let k = Self::parse_simple_type(inner[..pos].trim(), param_subst, var_subst, next_var, qvar_map);
                let v = Self::parse_simple_type(inner[pos+1..].trim(), param_subst, var_subst, next_var, qvar_map);
                return Type::Map(Box::new(k), Box::new(v));
            }
        }

        // Set[X]
        if s.starts_with("Set[") && s.ends_with(']') {
            let inner = &s[4..s.len() - 1];
            return Type::Set(Box::new(Self::parse_simple_type(inner, param_subst, var_subst, next_var, qvar_map)));
        }

        // Generic named types: Name[Args] (e.g., Option[Int], Result[Int, String])
        // Must come after List/Map/Set to avoid shadowing built-in types
        if let Some(bracket_pos) = s.find('[') {
            if s.ends_with(']') {
                let name = &s[..bracket_pos];
                let args_str = &s[bracket_pos + 1..s.len() - 1];
                // Parse comma-separated args at depth 0
                let mut args = Vec::new();
                let mut current = String::new();
                let mut depth = 0i32;
                for ch in args_str.chars() {
                    match ch {
                        '[' | '(' => { depth += 1; current.push(ch); }
                        ']' | ')' => { depth -= 1; current.push(ch); }
                        ',' if depth == 0 => {
                            if !current.trim().is_empty() {
                                args.push(Self::parse_simple_type(current.trim(), param_subst, var_subst, next_var, qvar_map));
                            }
                            current.clear();
                        }
                        _ => current.push(ch),
                    }
                }
                if !current.trim().is_empty() {
                    args.push(Self::parse_simple_type(current.trim(), param_subst, var_subst, next_var, qvar_map));
                }
                return Type::Named { name: name.to_string(), args };
            }
        }

        // Handle ?N type variable references (from first-pass inference embedded in HasMethod args)
        if let Some(stripped) = s.strip_prefix('?') {
            if let Ok(id) = stripped.parse::<u32>() {
                return qvar_map.entry(id).or_insert_with(|| {
                    // Use post-increment to match fresh_var() convention:
                    // return current value, then increment. Pre-increment caused
                    // Var ID collisions where parse_simple_type and fresh_var()
                    // would return the same ID (e.g., both return Var(29)).
                    let v = *next_var;
                    *next_var += 1;
                    Type::Var(v)
                }).clone();
            }
        }

        // Handle parenthesized expressions - could be function types or grouped types
        if s.starts_with('(') && s.ends_with(')') && s.len() > 2 {
            let inner = &s[1..s.len() - 1];
            // Check if outer parens wrap the entire expression (balanced)
            let mut depth = 0i32;
            let mut is_wrapped = true;
            for ch in inner.chars() {
                match ch {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => {
                        depth -= 1;
                        if depth < 0 { is_wrapped = false; break; }
                    }
                    _ => {}
                }
            }
            if is_wrapped && depth == 0 {
                return Self::parse_simple_type(inner, param_subst, var_subst, next_var, qvar_map);
            }
        }

        // Handle function types: find "->" at depth 0
        {
            let mut depth = 0i32;
            let bytes = s.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                match bytes[i] {
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth = (depth - 1).max(0),
                    b'-' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'>' => {
                        let params_str = s[..i].trim();
                        let ret_str = s[i + 2..].trim();

                        let params = if params_str == "()" || params_str.is_empty() {
                            vec![]
                        } else if params_str.starts_with('(') && params_str.ends_with(')') {
                            // Params in parens - split by comma at depth 0
                            let inner_params = &params_str[1..params_str.len() - 1];
                            let mut parts = Vec::new();
                            let mut current = std::string::String::new();
                            let mut d = 0i32;
                            for ch in inner_params.chars() {
                                match ch {
                                    '(' | '[' | '{' => { d += 1; current.push(ch); }
                                    ')' | ']' | '}' => { d -= 1; current.push(ch); }
                                    ',' if d == 0 => {
                                        if !current.trim().is_empty() {
                                            parts.push(Self::parse_simple_type(current.trim(), param_subst, var_subst, next_var, qvar_map));
                                        }
                                        current.clear();
                                    }
                                    _ => current.push(ch),
                                }
                            }
                            if !current.trim().is_empty() {
                                parts.push(Self::parse_simple_type(current.trim(), param_subst, var_subst, next_var, qvar_map));
                            }
                            parts
                        } else {
                            vec![Self::parse_simple_type(params_str, param_subst, var_subst, next_var, qvar_map)]
                        };

                        let ret = Self::parse_simple_type(ret_str, param_subst, var_subst, next_var, qvar_map);

                        return Type::Function(FunctionType {
                            type_params: vec![],
                            params,
                            ret: Box::new(ret),
                            required_params: None,
                            var_bounds: vec![],
                        });
                    }
                    _ => {}
                }
                i += 1;
            }
        }

        // Single lowercase letter → type param
        if s.len() == 1 && s.chars().next().unwrap().is_ascii_lowercase() {
            let ch = s.chars().next().unwrap();
            return param_subst.get(s).cloned()
                .or_else(|| {
                    let orig_id = (ch as u32) - ('a' as u32) + 1;
                    var_subst.get(&orig_id).cloned()
                })
                .unwrap_or(Type::String); // Should not happen in practice
        }

        // Fallback: Named type
        Type::Named { name: s.to_string(), args: vec![] }
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
                var_bounds: vec![],
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
            // Expand type aliases (e.g., `type Values = [Int]` → List[Int])
            let resolved_param = self.deep_expand_aliases(&resolved_param);
            let resolved_arg = self.deep_expand_aliases(&resolved_arg);

            // Only check when param type is a CONCRETE collection type (not a type variable).
            // This means the function explicitly expects a specific collection type.
            let param_is_concrete_collection = matches!(&resolved_param,
                Type::List(_) | Type::Map(_, _) | Type::Set(_) | Type::Tuple(_));
            if !param_is_concrete_collection {
                continue;
            }

            // Skip if arg is still a type variable or CONTAINS unresolved type variables.
            // When HM inference fails (e.g., for higher-order functions like zipWith),
            // types may be partially resolved with TypeParams/Vars inside Function types.
            // We can't make structural mismatch claims about incompletely resolved types.
            if has_unresolved_vars(&resolved_arg) {
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
                    // Expand type aliases (e.g., `type Values = [Int]` → List[Int])
                    let resolved_param = self.deep_expand_aliases(&resolved_param);
                    let resolved_arg = self.deep_expand_aliases(&resolved_arg);

                    // Skip if either is still a type variable (not yet resolved)
                    if matches!(&resolved_param, Type::Var(_) | Type::TypeParam(_)) { continue; }
                    if matches!(&resolved_arg, Type::Var(_) | Type::TypeParam(_)) { continue; }

                    // Skip if either is a single-letter Named type (leaked type parameter).
                    // This happens when function annotations use Pair[A, B] with cross-module types:
                    // A and B become Named { name: "A", args: [] } rather than TypeParam.
                    let is_named_type_param = |t: &Type| matches!(t,
                        Type::Named { name, args }
                        if args.is_empty() && name.len() == 1
                            && name.starts_with(|c: char| c.is_alphabetic())
                    );
                    if is_named_type_param(&resolved_param) || is_named_type_param(&resolved_arg) { continue; }

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

    /// Check freshened binding vars for post-solve type conflicts.
    /// When instantiate_local_binding freshens polymorphic vars for let-bindings,
    /// the freshened vars may end up with different types than the originals after
    /// solve() processes deferred constraints. This catches cases like:
    ///   g = wrap(identity(inc))  -- original vars resolve to Int after solve
    ///   g("hello")               -- freshened vars resolve to String
    /// Without this check, let-polymorphism incorrectly allows the second call.
    pub fn check_freshened_binding_conflicts(&mut self) -> Option<TypeError> {
        for (orig_id, fresh_id, span) in &self.freshened_binding_vars {
            let resolved_orig = self.env.apply_subst(&Type::Var(*orig_id));
            let resolved_fresh = self.env.apply_subst(&Type::Var(*fresh_id));
            // Deep expand type aliases for transparent comparison
            let resolved_orig = self.deep_expand_aliases(&resolved_orig);
            let resolved_fresh = self.deep_expand_aliases(&resolved_fresh);

            // Skip if either is still unresolved (truly polymorphic)
            if resolved_orig.has_any_type_var() || resolved_fresh.has_any_type_var() {
                continue;
            }

            // Both are concrete - if they differ, the freshening was incorrect
            if resolved_orig != resolved_fresh {
                self.last_error_span = Some(*span);
                return Some(TypeError::Mismatch {
                    expected: resolved_orig.display(),
                    found: resolved_fresh.display(),
                });
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
            // Deep expand type aliases for transparent comparison (handles List[Score] vs List[Int])
            let resolved_value = self.deep_expand_aliases(&resolved_value);
            let resolved_ann = self.deep_expand_aliases(&resolved_ann);

            // Check structural container mismatches even with unresolved type vars.
            // E.g., List[?25] vs Set[Int] - different container types, always an error.
            let is_structural_mismatch = matches!((&resolved_value, &resolved_ann),
                (Type::List(_), Type::Set(_)) | (Type::Set(_), Type::List(_)) |
                (Type::List(_), Type::Map(_, _)) | (Type::Map(_, _), Type::List(_)) |
                (Type::Set(_), Type::Map(_, _)) | (Type::Map(_, _), Type::Set(_)) |
                (Type::List(_), Type::Named { .. }) | (Type::Named { .. }, Type::List(_)) |
                (Type::Set(_), Type::Named { .. }) | (Type::Named { .. }, Type::Set(_)) |
                (Type::Map(_, _), Type::Named { .. }) | (Type::Named { .. }, Type::Map(_, _))
            );
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

            // Check if types are compatible (accounting for aliases like Float/Float64, Int/Int64)
            let types_compatible = resolved_value == resolved_ann
                || matches!((&resolved_value, &resolved_ann),
                    (Type::Float, Type::Float64) | (Type::Float64, Type::Float) |
                    (Type::Int, Type::Int64) | (Type::Int64, Type::Int));
            if !types_compatible {
                // For Named types, also resolve aliases (e.g., module.Type vs Type)
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
            let is_structural_mismatch = matches!((&resolved_a, &resolved_b),
                (Type::List(_), Type::Set(_)) | (Type::Set(_), Type::List(_)) |
                (Type::List(_), Type::Map(_, _)) | (Type::Map(_, _), Type::List(_)) |
                (Type::Set(_), Type::Map(_, _)) | (Type::Map(_, _), Type::Set(_))
            );
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
            // Expand type aliases before checking
            let resolved = self.expand_type_alias(&resolved).unwrap_or(resolved);
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
        // Collect the first unification error instead of returning early.
        // This allows subsequent constraints to still be processed, building
        // a more complete substitution for enrichment and the second pass.
        let mut first_unification_error: Option<TypeError> = None;

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
                        // TypeParams are polymorphic type parameters from function signatures
                        // (e.g., T in `double[T: Num](x: T)`). In batch inference, these
                        // can't be unified with concrete types - that happens during
                        // monomorphization. Skip unification failures involving TypeParams.
                        let t1r_pre = self.env.apply_subst(&t1);
                        let t2r_pre = self.env.apply_subst(&t2);
                        if matches!(&t1r_pre, Type::TypeParam(_)) || matches!(&t2r_pre, Type::TypeParam(_)) {
                            deferred_count = 0;
                            continue;
                        }

                        // Different-arity tuple types in pattern matching: allow match arms to
                        // have tuples of different lengths (e.g., (a, b) in one arm, (a, b, c)
                        // in another). This is valid in Nostos - the arms are mutually exclusive
                        // at runtime. Skip this constraint instead of failing.
                        if matches!(&e, TypeError::ArityMismatch { .. }) {
                            let t1r_full = self.apply_full_subst(&t1);
                            let t2r_full = self.apply_full_subst(&t2);
                            if matches!((&t1r_full, &t2r_full), (Type::Tuple(_), Type::Tuple(_))) {
                                deferred_count = 0;
                                continue; // Different-arity tuples coexist in match arms
                            }
                        }

                        // Check for implicit conversion before reporting error.
                        // If one side is a Named type (e.g., Tensor) and the other is List,
                        // look for a conversion function like tensorFromList.
                        let t1r = self.apply_full_subst(&t1);
                        let t2r = self.apply_full_subst(&t2);
                        if let Some(conv_fn) = self.find_implicit_conversion(&t1r, &t2r) {
                            if let Some(s) = span {
                                self.implicit_conversions.push((s, conv_fn));
                            }
                            deferred_count = 0;
                            continue; // Skip error, accept the conversion
                        }

                        self.last_error_span = error_span;

                        // Determine if this is a structural mismatch (e.g., List[X] vs Int,
                        // Function vs String). Structural mismatches from call chains are
                        // almost certainly real errors. For these, we save the error and
                        // CONTINUE solving so the substitution is more complete for enrichment.
                        // Simple type mismatches (Int vs String) may be from heterogeneous
                        // containers or overloading, so we return early (old behavior).
                        let mut structural = is_structural_mismatch(&t1r, &t2r);
                        let mut e = e;

                        // When top-level types are both Functions (from call constraints),
                        // the mismatch may be nested in param/return positions.
                        // E.g., filter(pred, map(f, xs)) with swapped args produces
                        // Function((List[?], ? -> Bool) -> ...) vs Function((?, List[?]) -> ...)
                        // where inner param types List vs Function are structurally mismatched.
                        if !structural {
                            if let (Type::Function(f1), Type::Function(f2)) = (&t1r, &t2r) {
                                for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                                    let rp1 = self.apply_full_subst(p1);
                                    let rp2 = self.apply_full_subst(p2);
                                    if is_structural_mismatch(&rp1, &rp2) {
                                        structural = true;
                                        e = TypeError::StructuralMismatch(rp1, rp2);
                                        break;
                                    }
                                }
                                if !structural {
                                    let rr1 = self.apply_full_subst(&f1.ret);
                                    let rr2 = self.apply_full_subst(&f2.ret);
                                    if is_structural_mismatch(&rr1, &rr2) {
                                        structural = true;
                                        e = TypeError::StructuralMismatch(rr1, rr2);
                                    }
                                }
                            }
                        }

                        // Check if this unification failure involves a parameter that needs annotation
                        match self.check_annotation_required(&t1, &t2, &e) {
                            Some(better_error) => {
                                if structural {
                                    let has_tp = self.contains_type_param(&t1r)
                                        || self.contains_type_param(&t2r);
                                    if !has_tp && first_unification_error.is_none() {
                                        first_unification_error = Some(better_error);
                                    }
                                } else {
                                    return Err(better_error);
                                }
                            },
                            None => {
                                // check_annotation_required returns None in two cases:
                                // 1. Not a function type unification (no annotation relevant)
                                // 2. Multi-type-param function where HM merged distinct type
                                //    params - false positive, skip the error
                                // Case 2 is detected by checking if either side is a Function
                                // with 2+ type_params AND the inner error is a simple type
                                // mismatch (not a structural mismatch like Function vs List).
                                let is_multi_tp_false_positive = match (&t1, &t2) {
                                    (Type::Function(f_def), Type::Function(f_call))
                                    | (Type::Function(f_call), Type::Function(f_def))
                                        if f_def.type_params.len() >= 2 =>
                                    {
                                        // Multi-type-param functions (e.g., pair[A, B]) may produce
                                        // false positive type errors when HM merges distinct type
                                        // params through if/else branches that swap them.
                                        // Suppress ONLY when both conflicting types appear in the
                                        // call-site args (indicating HM param merging). If one type
                                        // comes from an external constraint (e.g., another function's
                                        // param annotation), it's a real error.
                                        // Structural mismatches are always real, never false positives
                                        if structural {
                                            false
                                        } else {
                                            match &e {
                                                TypeError::UnificationFailed(a, b) => {
                                                    let call_params: Vec<Type> = f_call.params.iter()
                                                        .map(|p| self.apply_full_subst(p))
                                                        .collect();
                                                    let a_in_args = call_params.iter().any(|p| p == a);
                                                    let b_in_args = call_params.iter().any(|p| p == b);
                                                    a_in_args && b_in_args
                                                }
                                                _ => false,
                                            }
                                        }
                                    }
                                    _ => false,
                                };
                                if !is_multi_tp_false_positive {
                                    if structural {
                                        // Skip structural mismatches where either side contains
                                        // TypeParams (e.g., Tuple[(TypeParam a, Named Stack[TypeParam a])]
                                        // vs Int). These arise from uninstantiated generic function
                                        // signatures during batch inference and are false positives.
                                        // The TypeParam check at line ~2480 only catches top-level
                                        // TypeParams; here we catch TypeParams nested inside containers.
                                        let has_tp = self.contains_type_param(&t1r)
                                            || self.contains_type_param(&t2r);
                                        if !has_tp && first_unification_error.is_none() {
                                            first_unification_error = Some(e);
                                        }
                                    } else {
                                        return Err(e);
                                    }
                                }
                                // Multi-type-param false positive - skip this constraint
                            }
                        }
                    }
                    deferred_count = 0; // Made progress
                }
                Constraint::HasTrait(ty, trait_name, span) => {
                    let resolved = self.env.apply_subst(&ty);
                    // Expand type aliases so e.g. `type Score = Int` makes Score implement Num
                    let resolved = self.expand_type_alias(&resolved).unwrap_or(resolved);
                    // Update span tracking when a span is present, for precise error reporting
                    if let Some(s) = span {
                        self.current_constraint_span = Some(s);
                        self.last_error_span = Some(s);
                    }
                    match &resolved {
                        Type::Var(var_id) => {
                            // Track trait bound on the type variable
                            self.add_trait_bound(*var_id, trait_name.clone());
                            // Also record the original constraint for post-solve retry.
                            // Unlike trait_bounds (which get merged during variable
                            // unification), this preserves the original type so we can
                            // re-check after all substitutions are finalized.
                            self.deferred_has_trait.push((ty, trait_name, span));
                            deferred_count = 0; // Made progress (recorded the bound)
                        }
                        Type::TypeParam(_) => {
                            // TypeParams represent polymorphic type variables from function
                            // signatures (e.g., T in `double[T: Num](x: T)`). The trait
                            // bound is declared on the function, so this is always valid.
                            // Defer to post-solve - the TypeParam will be replaced with
                            // concrete types during monomorphization.
                            //
                            // ALSO store the trait bound on the original Var ID (if the
                            // original `ty` was a Var that resolved to TypeParam via subst).
                            // This is needed for try_hm_inference to discover usage-based
                            // trait bounds (e.g., Eq from `x == y` in `contains[T]`) and
                            // include them in the function's signature string.
                            if let Type::Var(orig_var_id) = &ty {
                                self.add_trait_bound(*orig_var_id, trait_name.clone());
                            }
                            self.deferred_has_trait.push((ty, trait_name, span));
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
                                    self.deferred_has_trait.push((ty, trait_name, span));
                                    deferred_count = 0;
                                } else {
                                    // Fully concrete function type - definitely an error.
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved.display(),
                                        trait_name,
                                        resolved_type: Some(resolved.clone()),
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
                                        resolved_type: Some(resolved.clone()),
                                    });
                                }
                                // Save for post-solve retry - type vars may resolve later
                                self.deferred_has_trait.push((ty, trait_name, span));
                                deferred_count = 0;
                            } else {
                                // Fully concrete type that doesn't implement the trait
                                return Err(TypeError::MissingTraitImpl {
                                    ty: resolved.display(),
                                    trait_name,
                                    resolved_type: Some(resolved.clone()),
                                });
                            }
                        }
                    }
                }
                Constraint::HasField(ty, field, expected_ty, field_span) => {
                    // When the field name starts with uppercase, it's a namespace qualifier
                    // used in chaining like `val.Map.insert(k, v)` which the compiler rewrites
                    // to `Map.insert(val, k, v)`. Skip the HasField constraint silently so
                    // type inference doesn't block compilation for this valid UFCS pattern.
                    // The compiler's codegen handles the actual dispatch.
                    if field.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        // Namespace qualifier - suppress HasField constraint, don't error
                        continue;
                    }
                    // If we have a span for this field access, set it as the last error span
                    // so NoSuchField errors point to the specific field access expression.
                    if let Some(s) = field_span {
                        self.last_error_span = Some(s);
                    }
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
                                    resolved_type: Some(resolved.clone()),
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
                                                resolved_type: Some(resolved.clone()),
                                            });
                                        }
                                    }
                                    crate::TypeDef::Variant { params, constructors } => {
                                        // Only allow field access on single-constructor variants
                                        // (which act like records). Multi-constructor variants
                                        // require pattern matching since the VM can't resolve
                                        // field names across different constructors.
                                        if constructors.len() == 1 {
                                            match constructors.first() {
                                            Some(Constructor::Named(_, fields)) => {
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
                                                        resolved_type: Some(resolved.clone()),
                                                    });
                                                }
                                            }
                                            Some(Constructor::Positional(_, field_types)) => {
                                                // Support numeric field access: .0, .1, .2, etc.
                                                if let Ok(idx) = field.parse::<usize>() {
                                                    if let Some(actual_ty) = field_types.get(idx) {
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
                                                            resolved_type: Some(resolved.clone()),
                                                        });
                                                    }
                                                } else {
                                                    return Err(TypeError::NoSuchField {
                                                        ty: resolved.display(),
                                                        field,
                                                        resolved_type: Some(resolved.clone()),
                                                    });
                                                }
                                            }
                                            _ => {
                                                return Err(TypeError::NoSuchField {
                                                    ty: resolved.display(),
                                                    field,
                                                    resolved_type: Some(resolved.clone()),
                                                });
                                            }
                                            }
                                        } else {
                                            return Err(TypeError::NoSuchField {
                                                ty: resolved.display(),
                                                field,
                                                resolved_type: Some(resolved.clone()),
                                            });
                                        }
                                    }
                                    _ => {
                                        return Err(TypeError::NoSuchField {
                                            ty: resolved.display(),
                                            field,
                                            resolved_type: Some(resolved.clone()),
                                        });
                                    }
                                }
                            } else {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                    resolved_type: Some(resolved.clone()),
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
                                self.deferred_has_field.push((ty, field, expected_ty, field_span));
                                continue;
                            }
                            self.constraints
                                .push(Constraint::HasField(resolved, field, expected_ty, field_span));
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
                                        resolved_type: Some(resolved.clone()),
                                    });
                                }
                            } else {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                    resolved_type: Some(resolved.clone()),
                                });
                            }
                        }
                        Type::List(_) => {
                            // Lists don't support field access (.0, .1, etc.) -
                            // that's only for tuples. Use list.get(idx) instead.
                            return Err(TypeError::NoSuchField {
                                ty: resolved.display(),
                                field,
                                resolved_type: Some(resolved.clone()),
                            });
                        }
                        _ => {
                            return Err(TypeError::NoSuchField {
                                ty: resolved.display(),
                                field,
                                resolved_type: Some(resolved.clone()),
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
        for (ty, trait_name, span) in &self.deferred_has_trait.clone() {
            let resolved = self.env.apply_subst(ty);
            // Expand type aliases so e.g. `type Score = Int` makes Score implement Num
            let resolved = self.expand_type_alias(&resolved).unwrap_or(resolved);
            // Update span tracking if this constraint has a span
            if let Some(s) = span {
                self.last_error_span = Some(*s);
            }
            match &resolved {
                Type::Var(_) | Type::TypeParam(_) => {} // Still unresolved or polymorphic, skip
                // Named types with single lowercase letter and no args are type params
                // leaked through apply_subst (which doesn't call resolve_type_params_with_depth).
                // Treat them as polymorphic — don't reject, just skip.
                Type::Named { name, args } if args.is_empty() && name.len() == 1
                    && name.starts_with(|c: char| c.is_ascii_lowercase()) => {}
                Type::Function(_) => {
                    // Function types NEVER implement standard traits (Eq, Ord, Num, etc.)
                    // regardless of their parameter/return types. So even with unresolved
                    // type vars in the function signature, this is always an error.
                    // Display the type, substituting what we can.
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: trait_name.clone(),
                        resolved_type: Some(resolved.clone()),
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
                        // last_error_span is already set above from the stored span if available.
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
                            resolved_type: Some(resolved.clone()),
                        });
                    }
                }
            }
        }

        // Post-solve: resolve deferred overload calls now that type variables
        // (e.g., lambda params from map/filter) may be resolved.
        // Process iteratively: if an overload's arg types are still unresolved Vars,
        // re-queue it. This handles cascading deferrals (e.g., magnitude deferred
        // because map hadn't resolved the lambda param type yet).
        let mut deferred_overloads = std::mem::take(&mut self.deferred_overload_calls);
        for _pass in 0..3 {
            if deferred_overloads.is_empty() { break; }
            let mut still_deferred = Vec::new();
            for (overloads, arg_types, ret_ty, span) in deferred_overloads {
                let overload_refs: Vec<&FunctionType> = overloads.iter().collect();
                let (best_idx, is_ambiguous) = self.find_best_overload_idx(&overload_refs, &arg_types);
                if is_ambiguous {
                    // Still ambiguous - re-queue for next pass
                    still_deferred.push((overloads, arg_types, ret_ty, span));
                    continue;
                }
                if let Some(idx) = best_idx {
                    let sig = overloads[idx].clone();
                    let func_ty = self.instantiate_function(&sig);
                    if let Type::Function(ft) = func_ty {
                        for (arg_ty, param_ty) in arg_types.iter().zip(ft.params.iter()) {
                            if std::env::var("NOSTOS_DEBUG_DEFERRED").is_ok() {
                                let ra = self.env.apply_subst(arg_ty);
                                let rp = self.env.apply_subst(param_ty);
                                eprintln!("[3-PASS UNIFY] {} <-> {} (overload[{}] ret={})",
                                    ra.display(), rp.display(), idx, ft.ret.display());
                            }
                            self.unify_types(arg_ty, param_ty)?;
                        }
                        self.unify_types(&ret_ty, &ft.ret)?;
                    }
                }
            }
            deferred_overloads = still_deferred;
        }

        // Sequential deferred overload resolution: when chained calls have all-Var args
        // (e.g., filter(pred, map(f, xs))), try committing overloads sequentially so
        // constraints from resolving one call (map) flow into checking the next (filter).
        if !deferred_overloads.is_empty() {
            let saved_subst = self.env.substitution.clone();
            let saved_bounds = self.trait_bounds.clone();
            let mut sequential_error: Option<TypeError> = None;

            for (overloads, arg_types, ret_ty, _span) in &deferred_overloads {
                if std::env::var("NOSTOS_DEBUG_DEFERRED").is_ok() {
                    eprintln!("[DEFERRED] {} overloads, arg_types={:?}", overloads.len(),
                        arg_types.iter().map(|t| self.env.apply_subst(t).display()).collect::<Vec<_>>());
                    for (i, ov) in overloads.iter().enumerate() {
                        eprintln!("  overload[{}]: params={:?} ret={}", i,
                            ov.params.iter().map(|p| p.display()).collect::<Vec<_>>(),
                            ov.ret.display());
                    }
                }
                let resolved_args: Vec<Type> = arg_types.iter()
                    .map(|t| self.env.apply_subst(t))
                    .collect();

                let mut call_resolved = false;
                for sig in overloads {
                    let saved_inner = self.env.substitution.clone();
                    let saved_bounds_inner = self.trait_bounds.clone();
                    let func_ty = self.instantiate_function(sig);

                    let ok = if let Type::Function(ref ft) = func_ty {
                        let mut ok = true;
                        for (a, p) in resolved_args.iter().zip(ft.params.iter()) {
                            if self.unify_types(a, p).is_err() {
                                ok = false;
                                break;
                            }
                        }
                        if ok && self.unify_types(ret_ty, &ft.ret).is_err() {
                            ok = false;
                        }
                        ok
                    } else { false };

                    if ok {
                        call_resolved = true;
                        break; // Keep substitution — constraints flow to next call
                    } else {
                        self.env.substitution = saved_inner;
                        self.trait_bounds = saved_bounds_inner;
                    }
                }

                if !call_resolved {
                    // All overloads failed. Check for structural mismatch using
                    // the current substitution (which has constraints from prior calls).
                    let re_resolved: Vec<Type> = arg_types.iter()
                        .map(|t| self.env.apply_subst(t))
                        .collect();
                    if let Some(sig) = overloads.first() {
                        let saved_check = self.env.substitution.clone();
                        let saved_bounds_check = self.trait_bounds.clone();
                        let func_ty = self.instantiate_function(sig);
                        if let Type::Function(ref ft) = func_ty {
                            for (a, p) in re_resolved.iter().zip(ft.params.iter()) {
                                let ra = self.env.apply_subst(a);
                                let rp = self.env.apply_subst(p);
                                // Use is_structural_mismatch on actual Type values
                                if is_structural_mismatch(&ra, &rp) {
                                    sequential_error = Some(TypeError::StructuralMismatch(
                                        ra.clone(), rp.clone(),
                                    ));
                                    break;
                                }
                            }
                        }
                        self.env.substitution = saved_check;
                        self.trait_bounds = saved_bounds_check;
                    }
                    if sequential_error.is_some() { break; }
                }
            }

            // Restore original state — this was a validation pass
            self.env.substitution = saved_subst;
            self.trait_bounds = saved_bounds;

            if let Some(err) = sequential_error {
                return Err(err);
            }
        }

        // Post-solve: check pending method calls now that types are resolved
        self.check_pending_method_calls()?;

        // Post-method-call: re-check deferred HasTrait constraints.
        // Method call resolution (above) may have resolved type variables that were
        // Vars during the first deferred_has_trait pass. For example, fold's lambda
        // params get unified with List element types during check_pending_method_calls.
        for (ty, trait_name, span) in &self.deferred_has_trait.clone() {
            let resolved = self.env.apply_subst(ty);
            // Expand type aliases
            let resolved = self.expand_type_alias(&resolved).unwrap_or(resolved);
            // Update span tracking if this constraint has a span
            if let Some(s) = span {
                self.last_error_span = Some(*s);
            }
            match &resolved {
                Type::Var(_) | Type::TypeParam(_) => {} // Still unresolved or polymorphic, skip
                // Named type params leaked through apply_subst — skip
                Type::Named { name, args } if args.is_empty() && name.len() == 1
                    && name.starts_with(|c: char| c.is_ascii_lowercase()) => {}
                Type::Function(_) => {
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: trait_name.clone(),
                        resolved_type: Some(resolved.clone()),
                    });
                }
                _ => {
                    if !self.env.implements(&resolved, trait_name) {
                        if resolved.has_any_type_var()
                            && !self.env.definitely_not_implements(&resolved, trait_name) {
                            continue;
                        }
                        // last_error_span is already set above from the stored span if available.
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
                            resolved_type: Some(resolved.clone()),
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
        for (ty, field, expected_ty, field_span) in deferred {
            // Set the field's span so error reporting points to the specific access
            if let Some(s) = field_span {
                self.last_error_span = Some(s);
            }
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
                            resolved_type: Some(resolved.clone()),
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
                                        resolved_type: Some(resolved.clone()),
                                    });
                                }
                            }
                            crate::TypeDef::Variant { params, constructors } => {
                                // Only allow field access on single-constructor variants
                                if constructors.len() == 1 {
                                    match constructors.first() {
                                    Some(Constructor::Named(_, fields)) => {
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
                                                resolved_type: Some(resolved.clone()),
                                            });
                                        }
                                    }
                                    Some(Constructor::Positional(_, field_types)) => {
                                        // Support numeric field access: .0, .1, .2, etc.
                                        if let Ok(idx) = field.parse::<usize>() {
                                            if let Some(actual_ty) = field_types.get(idx) {
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
                                                    resolved_type: Some(resolved.clone()),
                                                });
                                            }
                                        } else {
                                            return Err(TypeError::NoSuchField {
                                                ty: resolved.display(),
                                                field,
                                                resolved_type: Some(resolved.clone()),
                                            });
                                        }
                                    }
                                    _ => {
                                        return Err(TypeError::NoSuchField {
                                            ty: resolved.display(),
                                            field,
                                            resolved_type: Some(resolved.clone()),
                                        });
                                    }
                                    }
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: resolved.display(),
                                        field,
                                        resolved_type: Some(resolved.clone()),
                                    });
                                }
                            }
                            _ => {
                                return Err(TypeError::NoSuchField {
                                    ty: resolved.display(),
                                    field,
                                    resolved_type: Some(resolved.clone()),
                                });
                            }
                        }
                    } else {
                        return Err(TypeError::NoSuchField {
                            ty: resolved.display(),
                            field,
                            resolved_type: Some(resolved.clone()),
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
                                resolved_type: Some(resolved.clone()),
                            });
                        }
                    } else {
                        return Err(TypeError::NoSuchField {
                            ty: resolved.display(),
                            field,
                            resolved_type: Some(resolved.clone()),
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
                        resolved_type: Some(resolved.clone()),
                    });
                }
                Type::Var(_) => {
                    // Still unresolved - re-defer for another pass
                    still_deferred.push((ty, field, expected_ty, field_span));
                }
                _ => {
                    // Other types don't support field access
                    return Err(TypeError::NoSuchField {
                        ty: resolved.display(),
                        field,
                        resolved_type: Some(resolved.clone()),
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
        for (ty, trait_name, span) in &self.deferred_has_trait.clone() {
            let resolved = self.env.apply_subst(ty);
            // Expand type aliases
            let resolved = self.expand_type_alias(&resolved).unwrap_or(resolved);
            // Update span tracking if this constraint has a span
            if let Some(s) = span {
                self.last_error_span = Some(*s);
            }
            match &resolved {
                Type::Var(_) | Type::TypeParam(_) => {} // Still unresolved or polymorphic, skip
                // Named type params leaked through apply_subst — skip
                Type::Named { name, args } if args.is_empty() && name.len() == 1
                    && name.starts_with(|c: char| c.is_ascii_lowercase()) => {}
                Type::Function(_) => {
                    return Err(TypeError::MissingTraitImpl {
                        ty: resolved.display(),
                        trait_name: trait_name.clone(),
                        resolved_type: Some(resolved.clone()),
                    });
                }
                _ => {
                    if !self.env.implements(&resolved, trait_name) {
                        if resolved.has_any_type_var()
                            && !self.env.definitely_not_implements(&resolved, trait_name) {
                            continue;
                        }
                        // last_error_span is already set above from the stored span if available.
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
                            resolved_type: Some(resolved.clone()),
                        });
                    }
                }
            }
        }

        // Process deferred index access checks.
        // When Index expressions had unresolved container types (Var), we deferred them.
        // Now that solve() has run, resolve the container and add proper constraints.
        for (container_ty, index_ty, elem_ty, _span, literal_index) in std::mem::take(&mut self.deferred_index_checks) {
            let resolved = self.env.apply_subst(&container_ty);
            match &resolved {
                Type::Map(key_ty, val_ty) => {
                    // Map[K,V] indexing: index must be K, result is V
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), key_ty);
                    let _ = self.unify_types(&self.env.apply_subst(&elem_ty), val_ty);
                }
                Type::Set(set_elem_ty) => {
                    // Set[T] indexing: index must be T, result is Bool
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), set_elem_ty);
                    let _ = self.unify_types(&self.env.apply_subst(&elem_ty), &Type::Bool);
                }
                Type::List(list_elem_ty) => {
                    // List[T] indexing: index must be Int, result is T
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), &Type::Int);
                    let _ = self.unify_types(&self.env.apply_subst(&elem_ty), list_elem_ty);
                }
                Type::Tuple(elems) => {
                    // Tuple indexing: index must be Int
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), &Type::Int);
                    // If literal index available, resolve to specific element type
                    if let Some(n) = literal_index {
                        let idx = n as usize;
                        if idx < elems.len() {
                            let _ = self.unify_types(&self.env.apply_subst(&elem_ty), &elems[idx]);
                        }
                    }
                }
                Type::String => {
                    // String indexing: index must be Int, result is String
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), &Type::Int);
                    let _ = self.unify_types(&self.env.apply_subst(&elem_ty), &Type::String);
                }
                Type::Array(arr_elem_ty) => {
                    // Typed array indexing: index must be Int, result is element type
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), &Type::Int);
                    let _ = self.unify_types(&self.env.apply_subst(&elem_ty), &arr_elem_ty);
                }
                Type::Named { .. } => {
                    // Custom type indexing: index must be Int
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), &Type::Int);
                }
                Type::Var(_) => {
                    // Container type is still unresolved. Check if the index type
                    // is Int (suggesting List access like xs[0]) vs a type variable
                    // (suggesting Map access like m[k]).
                    let resolved_idx = self.env.apply_subst(&index_ty);
                    if matches!(&resolved_idx, Type::Int) && literal_index.is_none() {
                        // Index is Int from a variable expression (not a literal) -
                        // assume List. Use unify_types directly (not constraint push)
                        // because the main constraint loop has already exited. This
                        // ensures the container var gets unified with List[elem],
                        // linking element type to trait bounds (e.g., Num from `xs[i] + 1`).
                        //
                        // When literal_index is Some, we DON'T assume List because the
                        // container could be a Tuple (e.g., kv[0], kv[1] on a (String, Int)).
                        // Tuples support literal integer indexing just like Lists. In batch
                        // inference, the container may remain Var because instantiate_function
                        // creates independent vars that aren't linked to the function body's
                        // HasMethod constraints. Forcing List here would wrongly unify
                        // heterogeneous tuple elements to a single type.
                        let list_ty = Type::List(Box::new(elem_ty.clone()));
                        let _ = self.unify_types(&container_ty, &list_ty);
                    }
                    // If index is non-Int (e.g., a type variable), leave unresolved -
                    // could be Map[K,V] indexing where we can't assume List.
                }
                _ => {
                    // Default: assume List-like (String, Array, etc.)
                    // Use unify_types directly (not constraint push) because the main
                    // constraint loop has already exited.
                    let _ = self.unify_types(&self.env.apply_subst(&index_ty), &Type::Int);
                    let list_ty = Type::List(Box::new(elem_ty.clone()));
                    let _ = self.unify_types(&container_ty, &list_ty);
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
                // Both fully concrete - use unify_types for comparison to handle
                // module-qualified Named types (e.g., "Cfg.Config" vs "Config")
                if self.unify_types(&resolved_default, &resolved_param).is_err() {
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
                        resolved_type: Some(resolved.clone()),
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
                    // Apply same heterogeneous-list logic as in BinOp::Concat handler:
                    // only unify element types when one is a Var (propagation) or same type.
                    // Skip merging two independent Vars or two different concrete types.
                    let ra = self.env.apply_subst(a);
                    let rb = self.env.apply_subst(b);
                    let should_unify = match (&ra, &rb) {
                        _ if ra == rb => true,
                        (Type::Var(_), Type::Var(_)) => false,
                        (Type::Var(_), _) | (_, Type::Var(_)) => true,
                        _ => false,
                    };
                    if should_unify {
                        let _ = self.unify_types(a, b);
                    }
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
                            resolved_type: Some(resolved_elem.clone()),
                        });
                    }
                    if self.env.definitely_not_implements(&resolved_elem, "Eq") {
                        self.last_error_span = Some(span);
                        return Err(TypeError::MissingTraitImpl {
                            ty: resolved_elem.display(),
                            trait_name: "Eq".to_string(),
                            resolved_type: Some(resolved_elem.clone()),
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
                            resolved_type: Some(resolved_key.clone()),
                        });
                    }
                    if self.env.definitely_not_implements(&resolved_key, "Eq") {
                        self.last_error_span = Some(span);
                        return Err(TypeError::MissingTraitImpl {
                            ty: resolved_key.display(),
                            trait_name: "Eq".to_string(),
                            resolved_type: Some(resolved_key.clone()),
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
                                resolved_type: Some(resolved.clone()),
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
                                    resolved_type: Some(resolved.clone()),
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
                                resolved_type: Some(resolved.clone()),
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
                                resolved_type: Some(resolved.clone()),
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

        // If a unification error was collected (but not returned early), return it now.
        // The substitution and post-solve checks are more complete than if we had
        // returned early, enabling better enrichment and second-pass error detection.
        if let Some(err) = first_unification_error {
            return Err(err);
        }

        Ok(())
    }

    /// Check if solve() completed normally (all constraints processed).
    /// Returns false if solve() hit the MAX_ITERATIONS limit.
    pub fn solve_completed(&self) -> bool {
        self.solve_completed
    }

    /// Build a local substitution for a specific set of seed Var IDs by tracing
    /// their dependencies through remaining unprocessed constraints.
    /// Returns a map from Var ID to resolved Type, without modifying the global substitution.
    pub fn resolve_vars_locally(&self, seed_vars: &[u32]) -> HashMap<u32, Type> {
        if seed_vars.is_empty() { return HashMap::new(); }

        // Collect all constraints into a local pool for mini-solve
        let mut local_subst: HashMap<u32, Type> = HashMap::new();

        // Track which vars we care about (seed + transitively discovered)
        let mut relevant_vars: std::collections::HashSet<u32> = seed_vars.iter().cloned().collect();

        for _ in 0..5 {
            let mut new_bindings: Vec<(u32, Type)> = Vec::new();
            for constraint in &self.constraints {
                if let Constraint::Equal(ref a, ref b, _) = constraint {
                    let a = self.apply_local_subst(a, &local_subst);
                    let b = self.apply_local_subst(b, &local_subst);
                    Self::extract_local_bindings(&a, &b, &relevant_vars, &mut new_bindings);
                }
            }
            let mut added = 0;
            // Track new vars for next iteration
            fn collect_var_ids(ty: &Type, ids: &mut std::collections::HashSet<u32>) {
                match ty {
                    Type::Var(id) => { ids.insert(*id); }
                    Type::List(inner) | Type::Set(inner) | Type::IO(inner) | Type::Array(inner) => {
                        collect_var_ids(inner, ids);
                    }
                    Type::Map(k, v) => { collect_var_ids(k, ids); collect_var_ids(v, ids); }
                    Type::Tuple(elems) => { for e in elems { collect_var_ids(e, ids); } }
                    Type::Function(ft) => {
                        for p in &ft.params { collect_var_ids(p, ids); }
                        collect_var_ids(&ft.ret, ids);
                    }
                    Type::Named { args, .. } => { for a in args { collect_var_ids(a, ids); } }
                    _ => {}
                }
            }
            for (var_id, ty) in new_bindings {
                if let std::collections::hash_map::Entry::Vacant(entry) = local_subst.entry(var_id) {
                    collect_var_ids(&ty, &mut relevant_vars);
                    entry.insert(ty);
                    added += 1;
                }
            }
            if added == 0 { break; }
        }
        local_subst
    }

    pub fn apply_local_subst(&self, ty: &Type, local: &HashMap<u32, Type>) -> Type {
        self.apply_local_subst_inner(ty, local, 0)
    }

    fn apply_local_subst_inner(&self, ty: &Type, local: &HashMap<u32, Type>, depth: usize) -> Type {
        if depth > 20 { return ty.clone(); } // cycle guard
        let t = self.env.apply_subst(ty);
        match &t {
            Type::Var(id) => {
                if let Some(resolved) = local.get(id) {
                    self.apply_local_subst_inner(resolved, local, depth + 1)
                } else {
                    t
                }
            }
            Type::List(inner) => Type::List(Box::new(self.apply_local_subst_inner(inner, local, depth))),
            Type::Set(inner) => Type::Set(Box::new(self.apply_local_subst_inner(inner, local, depth))),
            Type::IO(inner) => Type::IO(Box::new(self.apply_local_subst_inner(inner, local, depth))),
            Type::Array(inner) => Type::Array(Box::new(self.apply_local_subst_inner(inner, local, depth))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.apply_local_subst_inner(k, local, depth)),
                Box::new(self.apply_local_subst_inner(v, local, depth)),
            ),
            Type::Tuple(elems) => Type::Tuple(
                elems.iter().map(|e| self.apply_local_subst_inner(e, local, depth)).collect(),
            ),
            Type::Function(ft) => Type::Function(FunctionType {
                type_params: ft.type_params.clone(),
                params: ft.params.iter().map(|p| self.apply_local_subst_inner(p, local, depth)).collect(),
                ret: Box::new(self.apply_local_subst_inner(&ft.ret, local, depth)),
                required_params: ft.required_params,
                var_bounds: vec![],
            }),
            Type::Named { name: n, args } => Type::Named {
                name: n.clone(),
                args: args.iter().map(|a| self.apply_local_subst_inner(a, local, depth)).collect(),
            },
            _ => t,
        }
    }

    fn extract_local_bindings(
        a: &Type, b: &Type,
        relevant: &std::collections::HashSet<u32>,
        bindings: &mut Vec<(u32, Type)>,
    ) {
        match (a, b) {
            (Type::Var(id), ty) if !matches!(ty, Type::Var(_)) => {
                if relevant.contains(id) {
                    bindings.push((*id, ty.clone()));
                }
            }
            (ty, Type::Var(id)) if !matches!(ty, Type::Var(_)) => {
                if relevant.contains(id) {
                    bindings.push((*id, ty.clone()));
                }
            }
            (Type::Var(id1), Type::Var(id2)) if id1 != id2 => {
                if relevant.contains(id1) || relevant.contains(id2) {
                    if relevant.contains(id1) {
                        bindings.push((*id1, Type::Var(*id2)));
                    }
                    if relevant.contains(id2) {
                        bindings.push((*id2, Type::Var(*id1)));
                    }
                }
            }
            (Type::Function(f1), Type::Function(f2)) if f1.params.len() == f2.params.len() => {
                for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                    Self::extract_local_bindings(p1, p2, relevant, bindings);
                }
                Self::extract_local_bindings(&f1.ret, &f2.ret, relevant, bindings);
            }
            (Type::List(a), Type::List(b)) | (Type::Set(a), Type::Set(b))
            | (Type::IO(a), Type::IO(b)) | (Type::Array(a), Type::Array(b)) => {
                Self::extract_local_bindings(a, b, relevant, bindings);
            }
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                Self::extract_local_bindings(k1, k2, relevant, bindings);
                Self::extract_local_bindings(v1, v2, relevant, bindings);
            }
            (Type::Tuple(e1), Type::Tuple(e2)) if e1.len() == e2.len() => {
                for (a, b) in e1.iter().zip(e2.iter()) {
                    Self::extract_local_bindings(a, b, relevant, bindings);
                }
            }
            (Type::Named { args: a1, .. }, Type::Named { args: a2, .. }) if a1.len() == a2.len() => {
                for (a, b) in a1.iter().zip(a2.iter()) {
                    Self::extract_local_bindings(a, b, relevant, bindings);
                }
            }
            _ => {}
        }
    }

    /// After solve() exits early, supplement the substitution for specific Var IDs
    /// (typically those from pending_fn_signatures) from remaining unprocessed
    /// constraints. Only processes constraints involving the target Var IDs and
    /// their transitive dependencies, avoiding contamination of unrelated vars.
    pub fn supplement_substitution_for_vars(&mut self, target_vars: &std::collections::HashSet<u32>) {
        if target_vars.is_empty() { return; }

        // Iteratively resolve: each pass may discover transitive bindings
        for _ in 0..5 {
            let mut new_bindings: Vec<(u32, Type)> = Vec::new();
            for constraint in &self.constraints {
                if let Constraint::Equal(ref a, ref b, _) = constraint {
                    self.collect_var_bindings_targeted(a, b, target_vars, &mut new_bindings);
                }
            }
            let mut added = 0;
            for (var_id, ty) in new_bindings {
                if let std::collections::hash_map::Entry::Vacant(entry) = self.env.substitution.entry(var_id) {
                    entry.insert(ty);
                    added += 1;
                }
            }
            if added == 0 {
                break;
            }
        }
    }

    fn collect_var_bindings_targeted(
        &self, a: &Type, b: &Type,
        target_vars: &std::collections::HashSet<u32>,
        bindings: &mut Vec<(u32, Type)>,
    ) {
        let a = self.env.apply_subst(a);
        let b = self.env.apply_subst(b);
        match (&a, &b) {
            (Type::Var(id), ty) if !matches!(ty, Type::Var(_)) => {
                // Only collect if this var is in our target set
                if target_vars.contains(id) {
                    bindings.push((*id, ty.clone()));
                }
                // Also collect inner Var→concrete bindings for transitive resolution
                if let Type::Var(inner_id) = ty {
                    if target_vars.contains(inner_id) {
                        bindings.push((*inner_id, Type::Var(*id)));
                    }
                }
            }
            (ty, Type::Var(id)) if !matches!(ty, Type::Var(_)) => {
                if target_vars.contains(id) {
                    bindings.push((*id, ty.clone()));
                }
            }
            (Type::Var(id1), Type::Var(id2)) if id1 != id2 => {
                // Var→Var: only if at least one is a target
                if target_vars.contains(id1) {
                    bindings.push((*id1, Type::Var(*id2)));
                }
                if target_vars.contains(id2) {
                    bindings.push((*id2, Type::Var(*id1)));
                }
            }
            (Type::Function(f1), Type::Function(f2)) if f1.params.len() == f2.params.len() => {
                for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                    self.collect_var_bindings_targeted(p1, p2, target_vars, bindings);
                }
                self.collect_var_bindings_targeted(&f1.ret, &f2.ret, target_vars, bindings);
            }
            (Type::List(a), Type::List(b)) | (Type::Set(a), Type::Set(b))
            | (Type::IO(a), Type::IO(b)) | (Type::Array(a), Type::Array(b)) => {
                self.collect_var_bindings_targeted(a, b, target_vars, bindings);
            }
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                self.collect_var_bindings_targeted(k1, k2, target_vars, bindings);
                self.collect_var_bindings_targeted(v1, v2, target_vars, bindings);
            }
            (Type::Tuple(e1), Type::Tuple(e2)) if e1.len() == e2.len() => {
                for (a, b) in e1.iter().zip(e2.iter()) {
                    self.collect_var_bindings_targeted(a, b, target_vars, bindings);
                }
            }
            (Type::Named { args: a1, .. }, Type::Named { args: a2, .. }) if a1.len() == a2.len() => {
                for (a, b) in a1.iter().zip(a2.iter()) {
                    self.collect_var_bindings_targeted(a, b, target_vars, bindings);
                }
            }
            _ => {}
        }
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
                        if !trait_ret.has_any_type_var() && !matches!(trait_ret, Type::TypeParam(_))
                            && self.unify_types(ret_ty, &trait_ret).is_err() {
                            let resolved_call_ret = self.env.apply_subst(ret_ty);
                            self.last_error_span = span;
                            return Err(TypeError::Mismatch {
                                expected: format!("{} (return type of .{}())", trait_ret.display(), method_name),
                                found: resolved_call_ret.display().to_string(),
                            });
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

            if self.unify_types(ret_ty, &common_ret).is_err() {
                self.last_error_span = span;
                return Err(TypeError::Mismatch {
                    expected: format!("{} (return type of .{}())", common_ret.display(), method_name),
                    found: resolved_call_ret.display().to_string(),
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
        let pending = std::mem::take(&mut self.pending_method_calls);
        // Stable-partition: process calls with already-resolved receiver types first,
        // then unresolved ones. This ensures chained calls (a.zip().map()) process
        // in order, while lambda inner calls (w.reverse() inside map) are deferred.
        // Within each group, preserve creation order to handle chains correctly.
        let mut resolved_first = Vec::new();
        let mut unresolved_later = Vec::new();
        for call in pending {
            let resolved_receiver = self.env.apply_subst(&call.receiver_ty);
            if matches!(&resolved_receiver, Type::Var(_)) {
                unresolved_later.push(call);
            } else {
                resolved_first.push(call);
            }
        }
        resolved_first.append(&mut unresolved_later);
        let mut pending = resolved_first;

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
                // Expand type aliases so e.g. `type Nums = [Int]` resolves to List for method dispatch
                let resolved_receiver = self.deep_expand_aliases(&resolved_receiver);

                // Try to get the type name
                let type_name_opt = self.get_type_name(&resolved_receiver);

                // If receiver is still a type variable, defer for later iteration
                // (but not on the final iteration - let it fall through to fallback handling).
                // Exception: if the method is a known list method, don't defer - we can
                // infer the receiver type from the method name (handled in the else branch).
                if type_name_opt.is_none() && matches!(&resolved_receiver, Type::Var(_))
                    && iteration < max_iterations - 1
                {
                    // Methods that uniquely identify the receiver type.
                    // NOTE: "reverse", "take", "drop" are NOT exclusive - String also has them
                    let is_list = is_exclusive_list_method(&call.method_name);
                    // "map" and "flatMap" are shared between List, Option, and Result.
                    // Only assume List if the return type has already been resolved to List
                    // by other constraints. Otherwise defer to allow more iterations.
                    let is_shared_method = matches!(call.method_name.as_str(), "map" | "flatMap");
                    let assume_list_for_shared = if is_shared_method && iteration > 0 {
                        let resolved_ret = self.env.apply_subst(&call.ret_ty);
                        matches!(resolved_ret, Type::List(_))
                    } else {
                        false
                    };
                    let is_list_method = is_list || assume_list_for_shared;
                    // Map-only methods (not on Set/List/String):
                    // insert with 3 arg_types (receiver + key + value) → Map; 2 arg_types (receiver + elem) → Set
                    let is_map_method = is_exclusive_map_method(&call.method_name)
                        || (call.method_name == "insert" && call.arg_types.len() == 3);
                    // String-exclusive methods: can infer receiver is String
                    let is_string_method = is_exclusive_string_method(&call.method_name);
                    let can_infer_from_method = is_list_method || is_map_method || is_string_method;
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

                // If receiver is still a Var but method uniquely identifies a type,
                // infer the receiver type. This ensures the method call gets processed
                // (lookup, param/return unification) instead of being silently dropped.
                let type_name_opt = if type_name_opt.is_none() && matches!(&resolved_receiver, Type::Var(_)) {
                    // Last-resort inference: if receiver is still Var after all iterations,
                    // assume List for methods that uniquely identify list operations.
                    // NOTE: "reverse", "take", "drop" are NOT exclusive - String also has them
                    let is_list = is_exclusive_list_method(&call.method_name);
                    // "map" and "flatMap" are shared between List, Option, and Result.
                    // Only assume List if the return type has already been resolved to List
                    // by other constraints (e.g., from fold/filter/sort in the same chain).
                    // Otherwise, leave the receiver unresolved for monomorphization to handle
                    // (e.g., `addResults(r1, r2) = r1.flatMap(a => r2.map(b => a + b))`
                    // should dispatch to Result.flatMap/map, not List.flatMap/map).
                    let is_shared_method = matches!(call.method_name.as_str(), "map" | "flatMap");
                    let assume_list_for_shared = if is_shared_method {
                        let resolved_ret = self.env.apply_subst(&call.ret_ty);
                        matches!(resolved_ret, Type::List(_))
                    } else {
                        false
                    };
                    let is_list_method = is_list || assume_list_for_shared;
                    let is_map_method = is_exclusive_map_method(&call.method_name)
                        || (call.method_name == "insert" && call.arg_types.len() == 3);
                    let is_string_method = is_exclusive_string_method(&call.method_name);
                    if is_list_method {
                        // Unify receiver with List[?X] so type info flows properly
                        let elem = self.fresh();
                        let list_ty = Type::List(Box::new(elem));
                        let _ = self.unify_types(&resolved_receiver, &list_ty);
                        Some("List".to_string())
                    } else if is_map_method {
                        // Unify receiver with Map[?K, ?V] so type info flows properly
                        let key = self.fresh();
                        let val = self.fresh();
                        let map_ty = Type::Map(Box::new(key), Box::new(val));
                        let _ = self.unify_types(&resolved_receiver, &map_ty);
                        Some("Map".to_string())
                    } else if is_string_method {
                        // Unify receiver with String so type info flows properly
                        // (e.g., s.chars() must return [Char], not be free to match [String])
                        let _ = self.unify_types(&resolved_receiver, &Type::String);
                        Some("String".to_string())
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
                            // Try unqualified first, then qualified "List.method"
                            self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                matches!(ft.params.first(), Some(Type::List(_)) | Some(Type::Var(_)))
                            }).or_else(|| {
                                let qualified = format!("List.{}", call.method_name);
                                self.env.functions.get(&qualified).cloned()
                            })
                        }
                        "Map" => {
                            // Try unqualified first, then qualified "Map.method"
                            self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                matches!(ft.params.first(), Some(Type::Map(_, _)) | Some(Type::Var(_)))
                            }).or_else(|| {
                                let qualified = format!("Map.{}", call.method_name);
                                self.env.functions.get(&qualified).cloned()
                            })
                        }
                        "Set" => {
                            // Try unqualified first, then qualified "Set.method"
                            self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                matches!(ft.params.first(), Some(Type::Set(_)) | Some(Type::Var(_)))
                            }).or_else(|| {
                                let qualified = format!("Set.{}", call.method_name);
                                self.env.functions.get(&qualified).cloned()
                            })
                        }
                        "String" => {
                            // Try unqualified first, then qualified "String.method"
                            self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                matches!(ft.params.first(), Some(Type::String) | Some(Type::Var(_)))
                            }).or_else(|| {
                                let qualified = format!("String.{}", call.method_name);
                                self.env.functions.get(&qualified).cloned()
                            })
                        }
                        "Option" => {
                            // Option methods are registered as optXxx in stdlib
                            resolve_option_method_alias(&call.method_name)
                                .and_then(|name| self.env.functions.get(name).cloned())
                        }
                        "Result" => {
                            // Result methods are registered as resXxx in stdlib
                            resolve_result_method_alias(&call.method_name)
                                .and_then(|name| self.env.functions.get(name).cloned())
                        }
                        // For numeric types (Float, Int, etc.), fall back to generic builtins.
                        // E.g., Float.toInt() should resolve to generic `toInt: a -> Int` (returns Int),
                        // not String.toInt (returns Option[Int]).
                        "Float" | "Float32" | "Float64" |
                        "Int" | "Int8" | "Int16" | "Int32" | "Int64" |
                        "UInt8" | "UInt16" | "UInt32" | "UInt64" |
                        "BigInt" | "Decimal" => {
                            // Numeric types don't support list-only or string-only methods.
                            // Don't look them up — let the error branch below catch these.
                            let list_and_string_methods = [
                                "length", "len", "head", "tail", "init", "last", "nth",
                                "push", "pop", "slice", "concat", "reverse", "sort",
                                "map", "filter", "fold", "any", "all", "find", "position",
                                "unique", "flatten", "zip", "unzip", "take", "drop",
                                "empty", "isEmpty", "sum", "product", "indexOf", "sortBy",
                                "intersperse", "spanList", "groupBy", "transpose", "pairwise",
                                "isSorted", "isSortedBy", "enumerate", "maximum", "minimum",
                                "takeWhile", "dropWhile", "partition", "zipWith", "flatMap",
                                "join", "split", "trim", "trimStart", "trimEnd",
                                "toUpper", "toLower", "startsWith", "endsWith",
                                "contains", "replace", "chars",
                            ];
                            if list_and_string_methods.contains(&call.method_name.as_str()) {
                                None
                            } else {
                                // Look up generic unqualified builtin (accepts any type via Var param)
                                self.env.functions.get(&call.method_name).cloned().filter(|ft| {
                                    matches!(ft.params.first(), Some(Type::Var(_)))
                                })
                            }
                        }
                        _ => None,
                    }
                });

                // Fallback: arity-based lookup for custom type trait methods.
                // Trait impl methods are registered as {TypeName}.{method}/_,_,...
                // in trait_method_ufcs_signatures by compile_trait_impl_inner.
                let fn_type_opt = if fn_type_opt.is_some() {
                    fn_type_opt
                } else {
                    self.env.lookup_function_with_arity(&qualified_name, arity).cloned()
                };

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
                                let compatible = matches!((&resolved_receiver, first_param),
                                    (Type::List(_), Type::List(_)) |
                                    (Type::Map(_, _), Type::Map(_, _)) |
                                    (Type::Set(_), Type::Set(_)) |
                                    (Type::String, Type::String) |
                                    (_, Type::Var(_))
                                );
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
                        // HasMethod-sourced calls (from signature constraints) don't include
                        // receiver in arg_types. These calls exist for method existence
                        // checking and return type linking, not for full UFCS dispatch.
                        // Finding fn_type_opt above already confirms the method exists.
                        // Skip full dispatch to avoid over-constraining type vars through
                        // cross-context unification (e.g., forcing a generic param to be List
                        // because length's first param is [a]).
                        // BUT: still unify non-receiver params to propagate trait bounds.
                        // e.g., hasItem(xs, item) = xs.contains(item) — the Eq bound from
                        // contains' element param must flow to hasItem's item param.
                        if call.span.is_none() {
                            // HasMethod-sourced calls from signature constraints.
                            // call.arg_types[0] is the receiver; normally we skip unifying it
                            // to avoid forcing an unresolved receiver to a specific collection type
                            // (e.g., applyLen(x) = x.length() shouldn't force x to be List).
                            // BUT: when the receiver IS already resolved (from the calling context),
                            // we MUST unify it with the method's first param to establish element type
                            // connections. Example: myMapper(lst, f) = lst.map(f) called with
                            // List[(String, Int)] — unifying the receiver links elem=(String, Int)
                            // so the callback parameter f gets the correct element type.
                            if !call.arg_types.is_empty() && !ft.params.is_empty() {
                                let resolved_first = self.env.apply_subst(&call.arg_types[0]);
                                if !matches!(&resolved_first, Type::Var(_)) {
                                    let _ = self.unify_types(&call.arg_types[0], &ft.params[0]);
                                }
                            }
                            // Unify non-receiver args to propagate callback types.
                            // This enables cross-module dispatch: myMapper(xs, f) = xs.map(f)
                            // where the callback type f must flow through to link element types.
                            // BUT: skip unification when the resolved arg is a Function type
                            // whose params contain collection types (List/Set/Map). This avoids
                            // cross-pollution when HasMethod callback args share vars with the
                            // expression tree and those vars were wrapped by last-resort List
                            // inference, causing double-wrapping (e.g., List[List[Val]]).
                            if call.arg_types.len() > 1 && ft.params.len() > 1 {
                                for (call_arg, method_param) in call.arg_types[1..].iter().zip(ft.params[1..].iter()) {
                                    let resolved_arg = self.apply_full_subst(call_arg);
                                    let has_collection_in_fn_params = if let Type::Function(ref ft_inner) = resolved_arg {
                                        ft_inner.params.iter().any(|p| matches!(p, Type::List(_) | Type::Set(_) | Type::Map(_, _)))
                                    } else {
                                        false
                                    };
                                    if !has_collection_in_fn_params {
                                        let _ = self.unify_types(call_arg, method_param);
                                    }
                                }
                            }
                            // Also unify return type to flow result type info
                            let _ = self.unify_types(&call.ret_ty, &ft.ret);
                            // Drain any HasTrait constraints pushed by instantiate_function
                            // and add them to deferred_has_trait so they get checked in the
                            // post-method-call retry pass. The main constraint loop has
                            // already finished, so self.constraints won't be processed again.
                            let remaining = std::mem::take(&mut self.constraints);
                            for constraint in remaining {
                                if let Constraint::HasTrait(ty, trait_name, span) = constraint {
                                    self.deferred_has_trait.push((ty, trait_name, span));
                                } else {
                                    // Put back non-HasTrait constraints
                                    self.constraints.push(constraint);
                                }
                            }
                            made_progress = true;
                            continue;
                        }

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
                                // Expand type aliases before checking (e.g., `type Score = Int` → Int)
                                let resolved_elem = self.deep_expand_aliases(&resolved_elem);
                                // Check if element type is definitely non-numeric.
                                // Numeric types: Int*, UInt*, Float*, BigInt, Decimal
                                // Type variables (Var/TypeParam) might be numeric, so allow them.
                                let is_non_numeric = match &resolved_elem {
                                    Type::String | Type::Bool | Type::Char | Type::Unit | Type::Never |
                                    Type::Tuple(_) | Type::List(_) | Type::Map(_, _) | Type::Set(_) |
                                    Type::Array(_) | Type::Record(_) | Type::Function(_) => true,
                                    // Named types are non-numeric unless they're leaked type params
                                    // (single lowercase letter via apply_subst path)
                                    Type::Named { name, args } => {
                                        !(args.is_empty() && name.len() == 1
                                          && name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false))
                                    }
                                    _ => false,
                                };
                                if is_non_numeric {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved_elem.display(),
                                        trait_name: "Num".to_string(),
                                        resolved_type: Some(resolved_elem.clone()),
                                    });
                                }
                                // Add Num trait bound on type variables so it propagates
                                // through wrapper functions (e.g., mySum(xs) = xs.sum())
                                if let Type::Var(var_id) = &resolved_elem {
                                    self.add_trait_bound(*var_id, "Num".to_string());
                                }
                            }
                        }

                        // Pre-check: sort/maximum/minimum/isSorted require Ord on element types
                        if matches!(call.method_name.as_str(), "sort" | "maximum" | "minimum" | "isSorted") {
                            let resolved_recv = self.env.apply_subst(&call.receiver_ty);
                            if let Type::List(elem) = &resolved_recv {
                                let resolved_elem = self.env.apply_subst(elem);
                                // Types that don't implement Ord: tuples, lists, maps, sets,
                                // records, functions, Bool, user-defined types.
                                // TypeParam is generic — Ord bound checked at call sites, not definition.
                                // Named types are non-orderable unless they're leaked type params
                                // (single lowercase letter via apply_subst path).
                                let is_non_orderable = match &resolved_elem {
                                    Type::Bool | Type::Unit | Type::Never |
                                    Type::Tuple(_) | Type::List(_) | Type::Map(_, _) | Type::Set(_) |
                                    Type::Array(_) | Type::Record(_) | Type::Function(_) => true,
                                    Type::Named { .. } => {
                                        // Use the type environment to check if this Named type
                                        // implements Ord. User-defined variants/records auto-derive
                                        // Ord as long as their type args also implement Ord.
                                        // (Single-lowercase-letter names are type params that also
                                        // pass through implements() returning false conservatively,
                                        // but they are caught by the Var branch below.)
                                        !self.env.implements(&resolved_elem, "Ord")
                                    }
                                    _ => false,
                                };
                                if is_non_orderable {
                                    self.last_error_span = call.span;
                                    return Err(TypeError::MissingTraitImpl {
                                        ty: resolved_elem.display(),
                                        trait_name: "Ord".to_string(),
                                        resolved_type: Some(resolved_elem.clone()),
                                    });
                                }
                                // Add Ord trait bound on type variables so it propagates
                                // through wrapper functions (e.g., mySort(xs) = xs.sort())
                                if let Type::Var(var_id) = &resolved_elem {
                                    self.add_trait_bound(*var_id, "Ord".to_string());
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
                                        resolved_type: Some(resolved_elem.clone()),
                                    });
                                }
                                // Add Eq trait bound on type variables so it propagates
                                // through wrapper functions (e.g., myUnique(xs) = xs.unique())
                                if let Type::Var(var_id) = &resolved_elem {
                                    self.add_trait_bound(*var_id, "Eq".to_string());
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
                                            resolved_type: Some(key_ty.clone()),
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
                                // Expand type aliases before checking (e.g., `type Id = Int` → Int)
                                let resolved_elem = self.deep_expand_aliases(&resolved_elem);
                                // Check if element is a non-numeric type
                                let is_non_numeric = matches!(&resolved_elem,
                                    Type::Tuple(_) | Type::Record(_) | Type::Named { .. });
                                if is_non_numeric {
                                    // Check if lambda param was unified to a different (numeric) type
                                    if let Some(arg_ty) = call.arg_types.get(1) {
                                        let resolved_arg = self.env.apply_subst(arg_ty);
                                        if let Type::Function(lambda_fn) = &resolved_arg {
                                            // Skip this check when the lambda has more params than
                                            // the expected callback (1 param). This means tuple
                                            // destructuring: (a, b) => a + b where the element type
                                            // is (Int, Int). The lambda params map to the tuple
                                            // elements, not the whole tuple. The Case 2 handler
                                            // below will correctly unify the tuple elements with
                                            // the lambda params.
                                            let is_tuple_destructuring = if let Type::Tuple(elems) = &resolved_elem {
                                                lambda_fn.params.len() == elems.len() && lambda_fn.params.len() > 1
                                            } else {
                                                false
                                            };
                                            if !is_tuple_destructuring {
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
                                                            resolved_type: Some(resolved_elem.clone()),
                                                        });
                                                    }
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
                            // Look up param names to map named args to correct positions.
                            // Try multiple key patterns: bare method name, qualified with type,
                            // and with arity suffix. Trait UFCS methods are registered under
                            // "{Type}.{method}" keys, so we must also try the qualified name.
                            let param_names_opt = self.env.function_param_names.get(&call.method_name)
                                .or_else(|| {
                                    // Try with qualified name: "{TypeName}.{method}"
                                    self.env.function_param_names.get(&qualified_name)
                                })
                                .or_else(|| {
                                    // Try with arity suffix patterns
                                    let qualified = format!("{}/", call.method_name);
                                    self.env.function_param_names.get(&qualified)
                                })
                                .or_else(|| {
                                    // Try any key ending with ".{method}" - handles trait-qualified
                                    // keys like "App.Configurable.configure" when we look for
                                    // "App.configure" or just "configure".
                                    // For cross-module types, the key may be
                                    // "lib.App.lib.Configurable.configure" where type_name is "App"
                                    // (short name from get_type_name), so also check
                                    // contains(".{type_name}.").
                                    let method_suffix = format!(".{}", call.method_name);
                                    let type_dot = format!(".{}.", type_name);
                                    self.env.function_param_names.iter()
                                        .find(|(k, _)| k.ends_with(&method_suffix)
                                            && (k.starts_with(&type_name) || k.contains(&type_dot)))
                                        .map(|(_, v)| v)
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
                                            Err(TypeError::UnificationFailed(..)) => {
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
                                                // Both-concrete case: use resolved Type values instead of error strings
                                                if !resolved_ret.has_any_type_var() && !resolved_expected_ret.has_any_type_var() {
                                                    self.last_error_span = call.span;
                                                    return Err(TypeError::Mismatch {
                                                        expected: resolved_expected_ret.display(),
                                                        found: resolved_ret.display(),
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
                                                    Err(TypeError::UnificationFailed(..)) => {
                                                        // Check if both types are concrete (no type vars)
                                                        // Re-resolve after unify_types may have updated substitution
                                                        let fresh_elem = self.env.apply_subst(tuple_elem);
                                                        let fresh_param = self.env.apply_subst(lambda_param);
                                                        if !fresh_elem.has_any_type_var() && !fresh_param.has_any_type_var() {
                                                            self.last_error_span = call.span;
                                                            return Err(TypeError::Mismatch {
                                                                expected: fresh_param.display(),
                                                                found: fresh_elem.display(),
                                                            });
                                                        }
                                                    }
                                                    Err(_) => {}
                                                }
                                            }
                                            // Unify return types - catch structural mismatches
                                            match self.unify_types(&actual_fn.ret, &expected_fn.ret) {
                                                Ok(()) => {}
                                                Err(TypeError::UnificationFailed(..)) => {
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
                                                    // Also check both-concrete case using resolved Type values
                                                    if !resolved_ret.has_any_type_var() && !resolved_expected_ret.has_any_type_var() {
                                                        self.last_error_span = call.span;
                                                        return Err(TypeError::Mismatch {
                                                            expected: resolved_expected_ret.display(),
                                                            found: resolved_ret.display(),
                                                        });
                                                    }
                                                }
                                                Err(_) => {}
                                            }
                                            continue; // Done with this param
                                        }
                                    }
                                }

                                // Case 3: Expected takes 1 unresolved type var and actual takes multiple
                                // This handles: f(xs) = { xs.map((a, b) => ...) } where xs is polymorphic.
                                // The receiver type isn't yet resolved to List[(T,U)], so the expected
                                // callback param is a type var. Treat the multi-param lambda as if it
                                // destructures a tuple, unifying the var with Tuple(params).
                                if expected_fn.params.len() == 1 && actual_fn.params.len() > 1 {
                                    let resolved_expected_param = self.env.apply_subst(&expected_fn.params[0]);
                                    if matches!(resolved_expected_param, Type::Var(_) | Type::TypeParam(_)) {
                                        // Unify the type var with a tuple of the actual lambda params
                                        let actual_param_types: Vec<Type> = actual_fn.params.iter()
                                            .map(|p| self.env.apply_subst(p))
                                            .collect();
                                        let tuple_ty = Type::Tuple(actual_param_types);
                                        let _ = self.unify_types(&resolved_expected_param, &tuple_ty);
                                        // Unify return types
                                        let _ = self.unify_types(&actual_fn.ret, &expected_fn.ret);
                                        continue; // Skip normal arity-check unification
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
                                Err(TypeError::UnificationFailed(..)) => {
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
                                            if !arg_p.has_any_type_var() && !param_p.has_any_type_var()
                                                && self.unify_types(arg_p, param_p).is_err() {
                                                self.last_error_span = call.span;
                                                return Err(TypeError::Mismatch {
                                                    expected: param_p.display(),
                                                    found: arg_p.display(),
                                                });
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
                                    // General case: re-resolve types after unify_types may have
                                    // updated the substitution, then check using actual Type values.
                                    let fresh_arg = self.env.apply_subst(arg_ty);
                                    let fresh_param = self.env.apply_subst(param_ty);
                                    if !fresh_arg.has_any_type_var() && !fresh_param.has_any_type_var() {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::Mismatch {
                                            expected: fresh_param.display(),
                                            found: fresh_arg.display(),
                                        });
                                    }
                                    // Structural mismatch: different type kinds can never unify
                                    // regardless of type variables (e.g., Int vs List[?1] from
                                    // [1,2,3].flatten() where flatten expects List[List[a]]).
                                    if is_structural_mismatch(&fresh_arg, &fresh_param) {
                                        self.last_error_span = call.span;
                                        return Err(TypeError::Mismatch {
                                            expected: fresh_param.display(),
                                            found: fresh_arg.display(),
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
                                        expected: b.display(),
                                        found: a.display(),
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

                    // Tuples support length/len at runtime
                    let is_tuple_length = type_name == "Tuple"
                        && matches!(call.method_name.as_str(), "length" | "len");

                    if (is_list_only || is_string_method) && !is_tuple_length {
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
                // NOTE: "reverse", "take", "drop" are NOT unique to List - String also has them
                let infer_list_methods = [
                    "filter", "fold", "any", "all", "find",
                    "sort", "sortBy", "head", "tail", "init", "last",
                    "sum", "product", "zip", "unzip",
                    "unique", "flatten", "position",
                    // Note: "indexOf" is NOT here because it's also valid on String.
                    // Assuming List for indexOf would incorrectly type String.indexOf calls
                    // when the receiver is still a type variable.
                    "push", "pop", "nth", "slice",
                    "scanl", "foldl", "foldr", "enumerate", "intersperse",
                    "spanList", "groupBy", "transpose", "pairwise", "isSorted",
                    "isSortedBy", "maximum", "minimum", "takeWhile", "dropWhile",
                    "partition", "zipWith",
                ];
                // "map" and "flatMap" are shared between List, Option, and Result.
                // Only assume List if the return type has already been resolved to List.
                let is_shared_method = matches!(call.method_name.as_str(), "map" | "flatMap");
                let assume_list_for_shared = if is_shared_method {
                    let resolved_ret = self.env.apply_subst(&call.ret_ty);
                    matches!(resolved_ret, Type::List(_))
                } else {
                    false
                };
                let can_infer_list = infer_list_methods.contains(&call.method_name.as_str())
                    || assume_list_for_shared;
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
                                if self.unify_types(&call.ret_ty, &ft.ret).is_err() {
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
                // Try to resolve remaining deferred calls where the receiver is still
                // an unresolved Var by searching for unique trait method implementations.
                // For example, if getPrice is only implemented for Item, we can infer
                // that the receiver must be Item. This enables type inference for lambdas
                // like `item => item.getPrice()` where the parameter has no annotation.
                // IMPORTANT: Only do this for Var receivers, NOT TypeParam receivers,
                // to avoid interfering with generic function type parameters.
                let mut newly_resolved = Vec::new();
                let mut still_deferred = Vec::new();
                for call in deferred {
                    let resolved = self.env.apply_subst(&call.receiver_ty);
                    if !matches!(&resolved, Type::Var(_)) {
                        still_deferred.push(call);
                        continue;
                    }
                    // Check if the bare method name exists as a generic function
                    // (e.g., "show" with signature "a -> String"). If so, it works on
                    // any type and we should NOT infer a specific receiver type.
                    // Check if this method belongs to a trait that has implementations
                    // for multiple types (e.g., Show has impls for Int, String, Bool, etc.).
                    // If so, we cannot infer a unique receiver type from the method name alone.
                    let is_widely_implemented = {
                        // Check ALL traits that define this method. If ANY of them
                        // has 2+ implementations, the method is widely available and
                        // we should not infer a specific receiver type.
                        self.env.traits.values().any(|td| {
                            let has_method = td.required.iter().chain(td.defaults.iter())
                                .any(|m| m.name == call.method_name);
                            if has_method {
                                let impl_count = self.env.impls.iter()
                                    .filter(|imp| imp.trait_name == td.name)
                                    .count();
                                impl_count >= 2
                            } else {
                                false
                            }
                        })
                    };
                    if is_widely_implemented {
                        still_deferred.push(call);
                        continue;
                    }
                    let method_suffix = format!(".{}", call.method_name);
                    let mut candidate_type: Option<String> = None;
                    let mut multiple = false;
                    for (fn_name, fn_type) in self.env.functions.iter() {
                        let base = if let Some(slash_pos) = fn_name.find('/') {
                            &fn_name[..slash_pos]
                        } else {
                            fn_name.as_str()
                        };
                        if let Some(prefix) = base.strip_suffix(&method_suffix) {
                            if !prefix.contains('.') && !prefix.is_empty()
                                && prefix.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                            {
                                // Verify this is actually a UFCS method (first param is the
                                // receiver type) and not a module-qualified function (e.g.,
                                // Exec.run has String as first param, not Exec). Module functions
                                // should NOT be used to infer receiver types since they're not
                                // methods on values of that type.
                                let is_ufcs_method = fn_type.params.first().is_some_and(|first_param| {
                                    match first_param {
                                        Type::Named { name, .. } => {
                                            let base_name = name.split('.').last().unwrap_or(name);
                                            base_name == prefix
                                        }
                                        Type::Var(_) | Type::TypeParam(_) => true, // Generic first param is OK
                                        Type::List(_) => prefix == "List",
                                        Type::Map(_, _) => prefix == "Map",
                                        Type::Set(_) => prefix == "Set",
                                        Type::String => prefix == "String",
                                        _ => false,
                                    }
                                });
                                if !is_ufcs_method {
                                    continue; // Skip module functions
                                }
                                if candidate_type.as_ref() != Some(&prefix.to_string()) {
                                    if candidate_type.is_some() {
                                        multiple = true;
                                        break;
                                    }
                                    candidate_type = Some(prefix.to_string());
                                }
                            }
                        }
                    }
                    // Before committing to the unique candidate, check if there's a
                    // generic (unqualified) version of this method. Generic builtins
                    // like `toFloat: a -> Float` work on ANY type, so the existence
                    // of `String.toFloat` doesn't mean the receiver must be String.
                    if !multiple && candidate_type.is_some() {
                        if let Some(generic_fn) = self.env.functions.get(&call.method_name) {
                            if generic_fn.params.first().is_some_and(|p| matches!(p, Type::Var(_) | Type::TypeParam(_))) {
                                // Generic version exists — don't infer receiver type
                                candidate_type = None;
                            }
                        }
                    }
                    // Before committing to the unique candidate, check if any
                    // user-defined record type has a function-typed field with the
                    // same name as the method. If so, the method call could be a
                    // field access + call (e.g., `obj.send(x)` where `send` is a
                    // function-typed field), and we should NOT infer the receiver
                    // type from the UFCS candidate.
                    if !multiple && candidate_type.is_some() {
                        let has_fn_field = self.env.types.values().any(|td| {
                            if let TypeDef::Record { fields, .. } = td {
                                fields.iter().any(|(fname, ftype, _)| {
                                    fname == &call.method_name && matches!(ftype, Type::Function(_))
                                })
                            } else {
                                false
                            }
                        });
                        if has_fn_field {
                            // Ambiguous: could be UFCS method OR function-typed field
                            candidate_type = None;
                        }
                    }
                    if !multiple {
                        if let Some(type_name) = candidate_type {
                            // Found unique type - unify receiver with it
                            let param_count = self.env.types.get(&type_name)
                                .map(|td| match td {
                                    crate::TypeDef::Record { params, .. } |
                                    crate::TypeDef::Variant { params, .. } => params.len(),
                                    crate::TypeDef::Alias { .. } => 0,
                                })
                                .unwrap_or(0);
                            let type_args: Vec<Type> = (0..param_count).map(|_| self.fresh()).collect();
                            let named_ty = Type::Named { name: type_name.clone(), args: type_args };
                            let _ = self.unify_types(&resolved, &named_ty);
                            newly_resolved.push(call);
                            continue;
                        }
                    }
                    still_deferred.push(call);
                }

                // If we resolved some calls, process them like normal method calls
                if !newly_resolved.is_empty() {
                    for call in newly_resolved {
                        let resolved_receiver = self.env.apply_subst(&call.receiver_ty);
                        if let Some(type_name) = self.get_type_name(&resolved_receiver) {
                            let qualified_name = format!("{}.{}", type_name, call.method_name);
                            let arity = call.arg_types.len();
                            let fn_type_opt = self.env.functions.get(&qualified_name).cloned()
                                .or_else(|| self.env.lookup_function_with_arity(&qualified_name, arity).cloned());
                            if let Some(fn_type) = fn_type_opt {
                                let func_ty = self.instantiate_function(&fn_type);
                                if let Type::Function(ft) = func_ty {
                                    for (param_ty, arg_ty) in ft.params.iter().zip(call.arg_types.iter()) {
                                        let _ = self.unify_types(arg_ty, param_ty);
                                    }
                                    let _ = self.unify_types(&call.ret_ty, &ft.ret);
                                }
                            }
                        }
                    }
                }

                // Before breaking, record any still-deferred method calls on unresolved
                // Vars for signature propagation. These will be encoded as HasMethod(name)
                // in the function signature so call sites can validate them.
                for call in &still_deferred {
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

            // Pick up any newly added pending method calls from instantiate_function.
            // When processing a user-defined UFCS call (e.g., .double()), instantiate_function
            // may create new PendingMethodCall entries from the function's HasMethod constraints.
            // These were pushed to self.pending_method_calls (which was taken at the start).
            // We must include them in the next iteration so their return types get resolved,
            // enabling chained UFCS calls (e.g., [1,2,3].double().myReverse()).
            let newly_added = std::mem::take(&mut self.pending_method_calls);
            if !newly_added.is_empty() {
                made_progress = true;
                deferred.extend(newly_added);
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
            // Circular type reference - bail out silently
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
                    let result = self.resolve_type_params_with_depth(resolved, depth + 1);
                    // Fix: Batch inference can contaminate shared Var IDs through
                    // type_param_mappings, causing Var(1) to resolve to Named("a")
                    // instead of TypeParam("a"). Convert these back at the source.
                    if let Type::Named { ref name, ref args } = result {
                        if args.is_empty() && name.len() == 1
                           && name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
                           && !self.type_param_mappings.contains_key(name.as_str()) {
                            return Type::TypeParam(name.clone());
                        }
                    }
                    result
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
                var_bounds: vec![],
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
            // Circular type reference - bail out silently
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
                var_bounds: vec![],
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
    /// Expand a Named type if it refers to a TypeDef::Alias, returning the target type.
    /// For parameterized aliases like `type Ids[a] = List[a]`, substitutes type params.
    fn expand_type_alias(&self, ty: &Type) -> Option<Type> {
        if let Type::Named { name, args } = ty {
            // Resolve the name through type_aliases first
            let resolved_name = self.env.resolve_type_name(name);
            if let Some(TypeDef::Alias { params, target }) = self.env.types.get(&resolved_name) {
                // Substitute type parameters if the alias is parameterized
                if params.is_empty() || args.is_empty() {
                    return Some(target.clone());
                }
                // Build substitution map: param name -> provided arg
                let mut result = target.clone();
                for (param, arg) in params.iter().zip(args.iter()) {
                    result = self.substitute_type_param_in_type(&result, &param.name, arg);
                }
                return Some(result);
            }
        }
        None
    }

    /// Recursively expand all type aliases in a type, including inside containers.
    /// Returns the type with all aliases resolved, or the original type unchanged.
    fn deep_expand_aliases(&self, ty: &Type) -> Type {
        // First try top-level expansion
        if let Some(expanded) = self.expand_type_alias(ty) {
            return self.deep_expand_aliases(&expanded);
        }
        // Then recurse into sub-types
        match ty {
            Type::List(inner) => Type::List(Box::new(self.deep_expand_aliases(inner))),
            Type::Set(inner) => Type::Set(Box::new(self.deep_expand_aliases(inner))),
            Type::Array(inner) => Type::Array(Box::new(self.deep_expand_aliases(inner))),
            Type::IO(inner) => Type::IO(Box::new(self.deep_expand_aliases(inner))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.deep_expand_aliases(k)),
                Box::new(self.deep_expand_aliases(v)),
            ),
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| self.deep_expand_aliases(e)).collect()),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.deep_expand_aliases(a)).collect(),
            },
            Type::Function(ft) => Type::Function(FunctionType {
                params: ft.params.iter().map(|p| self.deep_expand_aliases(p)).collect(),
                ret: Box::new(self.deep_expand_aliases(&ft.ret)),
                type_params: ft.type_params.clone(),
                required_params: ft.required_params,
                var_bounds: ft.var_bounds.clone(),
            }),
            _ => ty.clone(),
        }
    }

    /// Substitute a named type parameter in a type with a concrete type.
    fn substitute_type_param_in_type(&self, ty: &Type, param_name: &str, replacement: &Type) -> Type {
        match ty {
            Type::TypeParam(name) if name == param_name => replacement.clone(),
            Type::Named { name, args } if args.is_empty() && name == param_name => replacement.clone(),
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.substitute_type_param_in_type(a, param_name, replacement)).collect(),
            },
            Type::List(inner) => Type::List(Box::new(self.substitute_type_param_in_type(inner, param_name, replacement))),
            Type::Array(inner) => Type::Array(Box::new(self.substitute_type_param_in_type(inner, param_name, replacement))),
            Type::Set(inner) => Type::Set(Box::new(self.substitute_type_param_in_type(inner, param_name, replacement))),
            Type::IO(inner) => Type::IO(Box::new(self.substitute_type_param_in_type(inner, param_name, replacement))),
            Type::Map(k, v) => Type::Map(
                Box::new(self.substitute_type_param_in_type(k, param_name, replacement)),
                Box::new(self.substitute_type_param_in_type(v, param_name, replacement)),
            ),
            Type::Tuple(elems) => Type::Tuple(
                elems.iter().map(|e| self.substitute_type_param_in_type(e, param_name, replacement)).collect(),
            ),
            Type::Function(ft) => Type::Function(FunctionType {
                params: ft.params.iter().map(|p| self.substitute_type_param_in_type(p, param_name, replacement)).collect(),
                ret: Box::new(self.substitute_type_param_in_type(&ft.ret, param_name, replacement)),
                type_params: ft.type_params.clone(),
                required_params: ft.required_params,
                var_bounds: ft.var_bounds.clone(),
            }),
            other => other.clone(),
        }
    }

    fn unify_types(&mut self, t1: &Type, t2: &Type) -> Result<(), TypeError> {
        let t1 = self.apply_full_subst(t1);
        let t2 = self.apply_full_subst(t2);

        // Expand type aliases before matching (e.g., `type Id = Int` → Int)
        let t1 = if let Some(expanded) = self.expand_type_alias(&t1) { expanded } else { t1 };
        let t2 = if let Some(expanded) = self.expand_type_alias(&t2) { expanded } else { t2 };

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
                // f1 is the "definition" side (lhs), f2 is the "call" side (rhs).
                // We always produce errors as "expected: f1 param count, found: f2 param count"
                // so the message reads "expected N (from definition), found M (from call)".
                let (func_with_more, func_with_less, f1_is_more) = if f1.params.len() >= f2.params.len() {
                    (f1, f2, true)
                } else {
                    (f2, f1, false)
                };

                // Check if the arity difference is acceptable
                let min_required = func_with_more.required_params.unwrap_or(func_with_more.params.len());
                let max_params = func_with_more.params.len();
                let provided = func_with_less.params.len();

                // Allow the call if provided args are between min_required and max_params
                // When required_params is None, ALL params are required (no defaults)
                if provided < min_required || provided > max_params {
                    // Error direction: "expected <definition params>, found <call args>"
                    // f1 is the definition (lhs of unify), f2 is what was provided (rhs).
                    let (expected, found) = if f1_is_more {
                        // f1 has more params (definition expects more than call provided)
                        (min_required, provided)
                    } else {
                        // f2 has more params (call provided more than definition expects)
                        // f1 is the definition, so expected = f1.params (= func_with_less.params)
                        (provided, min_required)
                    };
                    return Err(TypeError::ArityMismatch { expected, found });
                }

                // Unify the params that were provided.
                // When a param unification fails, check for implicit conversion before propagating.
                for (p1, p2) in func_with_more.params.iter().zip(func_with_less.params.iter()) {
                    if let Err(_e) = self.unify_types(p1, p2) {
                        // Try implicit conversion: resolve the types and check
                        let p1r = self.apply_full_subst(p1);
                        let p2r = self.apply_full_subst(p2);
                        if let Some(conv_fn) = self.find_implicit_conversion(&p1r, &p2r) {
                            // Record the conversion (span will be associated at the solve level)
                            if let Some(span) = self.current_constraint_span {
                                self.implicit_conversions.push((span, conv_fn));
                            }
                            // Continue without error — the conversion will be applied in codegen
                            continue;
                        }
                        // No conversion found — propagate the error
                        return Err(_e);
                    }
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
                    return Err(TypeError::UnificationFailed(t1.clone(), t2.clone()));
                }
                if a1.len() != a2.len() {
                    // When one side has 0 type args and the other doesn't,
                    // treat the 0-args side as having implicit fresh type variables.
                    // This handles bare type annotations like `p: Pair` for `Pair[A, B]`.
                    if a1.is_empty() && !a2.is_empty() {
                        // a1 is bare (e.g., Pair), a2 has args (e.g., Pair[Int, Int])
                        // Just accept - the bare type is compatible with any instantiation
                        return Ok(());
                    }
                    if a2.is_empty() && !a1.is_empty() {
                        // Same but reversed
                        return Ok(());
                    }
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
                    _ => Err(TypeError::UnificationFailed(t1.clone(), t2.clone())),
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
                    _ => Err(TypeError::UnificationFailed(t1.clone(), t2.clone())),
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

            // Named type that is actually a built-in primitive (e.g., Named("String") vs Type::String)
            // This can happen when cross-module signatures create Named types for primitives
            (Type::Named { name, args }, other) | (other, Type::Named { name, args })
                if args.is_empty() =>
            {
                // First try to expand type aliases (e.g., types.Token -> String for cross-module aliases)
                let named_ty = Type::Named { name: name.clone(), args: args.clone() };
                if let Some(expanded) = self.expand_type_alias(&named_ty) {
                    if expanded != named_ty {
                        return self.unify_types(&expanded, other);
                    }
                }
                // Try to resolve the Named type to a primitive and re-unify
                let resolved_name = self.env.resolve_type_name(name);
                let primitive = match resolved_name.as_str() {
                    "String" => Some(Type::String),
                    "Int" | "Int64" | "PgConn" => Some(Type::Int),
                    "Float" | "Float64" => Some(Type::Float),
                    "Bool" => Some(Type::Bool),
                    "Char" => Some(Type::Char),
                    "Int8" => Some(Type::Int8),
                    "Int16" => Some(Type::Int16),
                    "Int32" => Some(Type::Int32),
                    "UInt8" => Some(Type::UInt8),
                    "UInt16" => Some(Type::UInt16),
                    "UInt32" => Some(Type::UInt32),
                    "UInt64" => Some(Type::UInt64),
                    "Float32" => Some(Type::Float32),
                    "BigInt" => Some(Type::BigInt),
                    "Decimal" => Some(Type::Decimal),
                    "Pid" => Some(Type::Pid),
                    "Ref" => Some(Type::Ref),
                    "()" | "Unit" => Some(Type::Unit),
                    _ => None,
                };
                if let Some(prim) = primitive {
                    if &prim == other {
                        Ok(())
                    } else {
                        // Re-unify the resolved primitive with the other type
                        self.unify_types(&prim, other)
                    }
                } else {
                    Err(TypeError::UnificationFailed(t1.clone(), t2.clone()))
                }
            }

            // Named type with args that might be a type alias (e.g., Queue[Int] -> ([Int], [Int]))
            // Try expanding the Named type alias before giving up
            (Type::Named { .. }, other) | (other, Type::Named { .. }) => {
                let (named, non_named) = match (&t1, &t2) {
                    (Type::Named { .. }, _) => (&t1, &t2),
                    _ => (&t2, &t1),
                };
                if let Some(expanded) = self.expand_type_alias(named) {
                    if &expanded != named {
                        return self.unify_types(&expanded, non_named);
                    }
                }
                Err(TypeError::UnificationFailed(t1.clone(), t2.clone()))
            }

            // Mismatch
            _ => {
                Err(TypeError::UnificationFailed(t1.clone(), t2.clone()))
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
                    "Int" | "PgConn" => Type::Int,
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
                    _ => {
                        // Single lowercase letters are implicit type variables
                        // (e.g., xs: [a] means a is a type parameter).
                        // This is consistent with type_name_to_type("a") -> Var(1).
                        if name.len() == 1 && name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) {
                            let var_id = (name.chars().next().unwrap() as u32) - ('a' as u32) + 1;
                            Type::Var(var_id)
                        } else if name.len() == 1 && name.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false) {
                            // Single uppercase letters (A, B, T, etc.) are implicitly treated as
                            // type parameters when they are not known concrete types.
                            // This handles annotations like `fn(p: Pair[A, B])` where A and B
                            // are implicit type parameters rather than explicit `[A, B]` declarations.
                            let resolved = self.env.resolve_type_name(name);
                            if self.env.types.contains_key(&resolved) || resolved != name.as_str() {
                                // It's a known type (either directly or via alias) - use as Named
                                Type::Named {
                                    name: resolved,
                                    args: vec![],
                                }
                            } else {
                                // Not a known type - treat as implicit type parameter
                                Type::TypeParam(name.clone())
                            }
                        } else {
                            Type::Named {
                                // Resolve through type aliases (e.g., Option -> stdlib.list.Option)
                                name: self.env.resolve_type_name(name),
                                args: vec![],
                            }
                        }
                    }
                }
            }
            TypeExpr::Generic(name, args) => {
                let type_args: Vec<_> = args.iter().map(|a| self.type_from_ast_with_params(a, type_params)).collect();
                let name_str = &name.node;
                // Check if this name is a user-defined type that shadows a builtin name
                // (e.g., `type List[a]` in a user module shadows the stdlib List)
                if matches!(name_str.as_str(), "List" | "Array" | "Set" | "Map" | "IO") {
                    let resolved = self.env.resolve_type_name(name_str);
                    if resolved != name_str.as_str() {
                        return Type::Named {
                            name: resolved,
                            args: type_args,
                        };
                    }
                }
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
                    var_bounds: vec![],
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
                    let ty = ty.clone();
                    // Let-polymorphism: when a local binding with a polymorphic function type
                    // is referenced as a value (e.g., passed as HOF argument like `xs.map(double)`),
                    // freshen its type variables just like we do for direct calls.
                    // Without this, `double = (x) => x + x; [1,2,3].map(double)` would lock
                    // ?N to Int, preventing later use with other types.
                    Ok(self.instantiate_local_binding(&ty, name, Some(ident.span)))
                } else if let Some(sig) = self.env.functions.get(name).cloned() {
                    // Check if there are multiple typed overloads for this function name.
                    // When a function reference like `double` is used as a HOF argument
                    // (e.g., `xs.map(double)`), we need to let the calling context determine
                    // which overload to use via unification, rather than committing to whichever
                    // overload was registered first under the bare name.
                    let typed_overload_count = self.env.functions_by_base.get(name)
                        .map_or(0, |keys| {
                            let prefix = format!("{}/", name);
                            keys.iter().filter(|k| {
                                if !k.starts_with(&prefix) { return false; }
                                let suffix = &k[prefix.len()..];
                                // Exclude wildcard entries like "name/_" or "name/_,_"
                                !suffix.chars().all(|c| c == '_' || c == ',')
                            }).count()
                        });

                    if typed_overload_count > 1 {
                        // Multiple typed overloads exist - return a fresh function type
                        // so the calling context can constrain the params/return via unification.
                        // The compilation phase will then use the resolved types to select
                        // the correct overload.
                        let arity = sig.params.len();
                        let fresh_params: Vec<Type> = (0..arity).map(|_| self.fresh()).collect();
                        let fresh_ret = self.fresh();
                        Ok(Type::Function(FunctionType {
                            type_params: vec![],
                            params: fresh_params,
                            ret: Box::new(fresh_ret),
                            required_params: sig.required_params,
                            var_bounds: vec![],
                        }))
                    } else {
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
                        let ty = ty.clone();
                        // Let-polymorphism: if the local binding is a function type
                        // with unresolved type variables, create fresh copies on each use.
                        // This prevents polymorphic bindings like `always42 = constFn(42)`
                        // from locking their param types to the first call's arguments.
                        self.instantiate_local_binding(&ty, name, Some(*call_span))
                    } else {
                        // Get ALL overloads and find the best match based on argument types
                        // Clone to avoid borrow issues with instantiate_function
                        let mut overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(name, args.len())
                            .into_iter().cloned().collect();

                        // Also check for type-qualified versions (e.g., String.length for length(s) where s: String)
                        // This enables UFCS-style resolution for bare function calls.
                        // BUT: only add UFCS overloads if there are NO existing direct overloads.
                        // If the user has defined `parseInt`, `String.parseInt` should NOT be added
                        // as a competing overload — user-defined functions always shadow builtin methods.
                        if args.len() == 1 && !arg_types.is_empty() && overloads.is_empty() {
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
                            let (best_idx, is_ambiguous) = if has_named_args {
                                (Some(0), false) // Use first/wildcard overload for named args
                            } else {
                                let overload_refs: Vec<&FunctionType> = overloads.iter().collect();
                                self.find_best_overload_idx(&overload_refs, &arg_types)
                            };

                            // If ambiguous (multiple overloads tied because args are Vars),
                            // defer overload selection until after constraint solving.
                            // Return early to skip normal func_ty unification.
                            if is_ambiguous && overloads.len() > 1 && !is_recursive {
                                let ret_ty = self.fresh();
                                self.deferred_overload_calls.push((
                                    overloads.clone(),
                                    arg_types.clone(),
                                    ret_ty.clone(),
                                    *call_span,
                                ));
                                return Ok(ret_ty);
                            }
                            let mut sig = overloads[best_idx.unwrap_or(0)].clone();
                            // Trim params to match call arity when function has default params
                            if sig.params.len() > args.len() {
                                let min_required = sig.required_params.unwrap_or(sig.params.len());
                                if args.len() >= min_required {
                                    sig.params.truncate(args.len());
                                    sig.required_params = None;
                                }
                            }
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
                    // Also handle module names parsed as unit Record: Expr::Record("Module", [])
                    // Recursively extract qualified name to support nested modules:
                    // Outer.Inner.func(...) => base_expr = FieldAccess(Var("Outer"), "Inner")
                    let base_name_opt = Self::extract_qualified_name(base_expr);
                    if let Some(ref base_name) = base_name_opt {
                        let qualified_name = format!("{}.{}", base_name, field.node);
                        // Clone to avoid borrow issues
                        let overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(&qualified_name, args.len())
                            .into_iter().cloned().collect();
                        if !overloads.is_empty() {
                            // Find best overload index first to avoid borrow issues
                            // Skip strict matching if named args present
                            let (best_idx, is_ambiguous) = if has_named_args {
                                (Some(0), false)
                            } else {
                                let overload_refs: Vec<&FunctionType> = overloads.iter().collect();
                                self.find_best_overload_idx(&overload_refs, &arg_types)
                            };

                            if is_ambiguous && overloads.len() > 1 {
                                let ret_ty = self.fresh();
                                self.deferred_overload_calls.push((
                                    overloads.clone(),
                                    arg_types.clone(),
                                    ret_ty.clone(),
                                    *call_span,
                                ));
                                return Ok(ret_ty);
                            }
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
                        // same Var (from if/else branches that swap them), causing
                        // the return type to carry merged vars. Return a fresh var
                        // to avoid false positive type errors.
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
                    var_bounds: vec![],
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
                self.unify_at(func_ty.clone(), expected_func_ty, *call_span);

                // Eagerly unify the return type when func_ty returns a Function.
                // This makes curried function returns immediately available via
                // apply_subst for instantiate_local_binding, enabling let-polymorphism:
                //   always42 = constFn(42)  -- ret_ty resolves to (b -> Int) immediately
                //   always42("hello")       -- instantiate_local_binding sees Function type
                // Only for Function returns to avoid changing error messages for simple
                // calls like f() + 1 where f() returns a non-function type.
                if let Type::Function(ref ft) = func_ty {
                    if matches!(&*ft.ret, Type::Function(_)) {
                        let _ = self.unify_types(&ret_ty, &ft.ret);

                        // When the call has concrete arguments that constrain the return
                        // type's vars, those vars are no longer truly polymorphic.
                        // E.g., wrap(inc) where inc: Int->Int constrains wrap's return
                        // type vars to Int. If we leave them as polymorphic,
                        // instantiate_local_binding will freshen them, disconnecting
                        // the type constraint and letting g("hello") pass unchecked.
                        //
                        // Only remove polymorphic status for vars that appear in BOTH
                        // the function's params AND its return type. Vars that appear
                        // only in the return type (e.g., constFn(42) returns _ => 42,
                        // the _ parameter is not constrained) should stay polymorphic.
                        //
                        // Resolve args through apply_subst to handle nested calls
                        // like wrap(identity(inc)) where the arg is a Var that resolves
                        // to a concrete type through eager return-type unification.
                        let has_concrete_arg = arg_types_clone.iter().any(|arg| {
                            let resolved = self.env.apply_subst(arg);
                            match &resolved {
                                Type::Var(_) | Type::TypeParam(_) => false,
                                Type::Function(_) => {
                                    // A function arg is "concrete" if it has no vars,
                                    // or has at least one non-polymorphic var.
                                    let mut inner_ids = Vec::new();
                                    self.collect_var_ids(&resolved, &mut inner_ids);
                                    if inner_ids.is_empty() {
                                        true // No vars → fully concrete (e.g., Int -> Int)
                                    } else {
                                        inner_ids.iter().any(|id| !self.polymorphic_vars.contains(id))
                                    }
                                }
                                _ => true,
                            }
                        });
                        if has_concrete_arg {
                            // Collect var IDs from the param types (constrained by args)
                            let mut param_var_ids = Vec::new();
                            for p in &ft.params {
                                self.collect_var_ids(p, &mut param_var_ids);
                            }
                            // Only de-polymorphize return vars that also appear in params.
                            // Collect from entire return type, not just Function returns,
                            // to handle cases like identity(inc) where ret is a Var.
                            let mut ret_var_ids = Vec::new();
                            self.collect_var_ids(&ft.ret, &mut ret_var_ids);
                            for var_id in ret_var_ids {
                                if param_var_ids.contains(&var_id) {
                                    self.polymorphic_vars.remove(&var_id);
                                }
                            }
                        }
                    }
                }

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

                // Push a new return-type tracking frame for this lambda.
                // This ensures that `return` inside a lambda:
                // 1. Contributes to the lambda's return type (not the outer function's)
                // 2. Doesn't corrupt the outer function's fn_has_return_stack flag
                let lambda_ret_var = self.fresh();
                self.fn_return_type_stack.push(lambda_ret_var.clone());
                self.fn_has_return_stack.push(false);

                let body_ty = self.infer_expr(body)?;

                // Pop the lambda's return-type tracking frame
                self.fn_return_type_stack.pop();
                let lambda_has_return = self.fn_has_return_stack.pop().unwrap_or(false);

                // If the lambda body has `return` statements and the body type is Unit
                // (because the last statement was a return), use lambda_ret_var instead.
                let effective_body_ty = if lambda_has_return {
                    match &body_ty {
                        Type::Unit => lambda_ret_var,
                        _ => {
                            self.unify(lambda_ret_var, body_ty.clone());
                            body_ty
                        }
                    }
                } else {
                    body_ty
                };

                // Restore bindings
                self.env.bindings = saved_bindings;

                Ok(Type::Function(FunctionType { required_params: None,
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(effective_body_ty),
                    var_bounds: vec![],
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

                // Detect conflicting tuple arities in arms. If present, we use an unbound
                // fresh type variable as the "effective scrutinee type" for pattern inference
                // in those arms. This prevents the scrutinee from being bound to any specific
                // tuple arity, allowing dispatch functions that match different-arity tuples.
                let tuple_arities: Vec<Option<usize>> = arms.iter().map(|arm| {
                    if let Pattern::Tuple(elems, _) = &arm.pattern {
                        Some(elems.len())
                    } else {
                        None
                    }
                }).collect();
                let has_conflicting_tuple_arities = {
                    let defined_arities: Vec<usize> = tuple_arities.iter().flatten().copied().collect();
                    let first = defined_arities.first().copied();
                    first.is_some() && defined_arities.iter().any(|&a| Some(a) != first)
                };

                // Pre-scan: if the scrutinee type is unresolved, look for any pattern in
                // the arms that uniquely identifies the variant type. This disambiguates
                // constructors with the same name across types.
                // E.g., if StrLeaf only belongs to StrTree, and we see a StrLeaf arm,
                // we know Node patterns in other arms should use StrTree.Node.
                let saved_match_declared_return = self.current_declared_return_type.clone();
                {
                    let resolved_scrutinee = self.env.apply_subst(&scrutinee_ty);
                    if !matches!(&resolved_scrutinee, Type::Named { .. }) {
                        // Scrutinee type is still unresolved - scan arm patterns for unique constructors
                        let types_snap = self.env.types.clone();
                        'prescan: for arm in arms.iter() {
                            if let Pattern::Variant(pat_name, _, _) = &arm.pattern {
                                let ctor_search = &pat_name.node;
                                // Find all types that have this constructor
                                let mut owning_types: Vec<String> = Vec::new();
                                for (tn, def) in &types_snap {
                                    if let crate::TypeDef::Variant { constructors, .. } = def {
                                        for ctor in constructors {
                                            let cn = match ctor {
                                                crate::Constructor::Unit(n) => n.as_str(),
                                                crate::Constructor::Positional(n, _) => n.as_str(),
                                                crate::Constructor::Named(n, _) => n.as_str(),
                                            };
                                            if cn == ctor_search.as_str() {
                                                owning_types.push(tn.clone());
                                                break;
                                            }
                                        }
                                    }
                                }
                                // If exactly one type owns this constructor, use it to disambiguate
                                if owning_types.len() == 1 {
                                    let tn = &owning_types[0];
                                    let base = tn.rsplit('.').next().unwrap_or(tn.as_str());
                                    self.current_declared_return_type = Some(base.to_string());
                                    break 'prescan;
                                }
                            }
                        }
                    }
                }

                for arm in arms {
                    // If this arm is a tuple pattern and there are conflicting arities,
                    // infer the pattern against a fresh variable (not the scrutinee_ty),
                    // so the scrutinee is not constrained to a specific arity.
                    let arm_scrutinee_ty = if has_conflicting_tuple_arities {
                        if let Pattern::Tuple(_, _) = &arm.pattern {
                            self.fresh()
                        } else {
                            scrutinee_ty.clone()
                        }
                    } else {
                        scrutinee_ty.clone()
                    };
                    self.infer_match_arm(arm, &arm_scrutinee_ty, &result_ty)?;
                }
                // Restore the declared return type hint after processing all match arms.
                self.current_declared_return_type = saved_match_declared_return;

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
            Expr::Record(name, fields, record_span) => {
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

                            // Helper: given params and the base_ty, build substitution and expected_ty,
                            // then type-check the named update fields against def_fields.
                            // Returns Ok(expected_ty) if handled, or falls through.
                            let typedef = self.env.lookup_type(&resolved_type_name).cloned();

                            // Extract (params, def_fields) from either Record or single-Named-constructor Variant
                            let record_info: Option<(Vec<TypeParam>, Vec<(String, Type, bool)>)> = match &typedef {
                                Some(TypeDef::Record { params, fields: def_fields, .. }) => {
                                    Some((params.clone(), def_fields.clone()))
                                }
                                Some(TypeDef::Variant { params, constructors }) => {
                                    // If it's a single Named constructor with the same name as the type
                                    // (i.e., `type Config = Config { ... }`), treat it as a record update target
                                    if constructors.len() == 1 {
                                        if let Constructor::Named(ctor_name, named_fields) = &constructors[0] {
                                            let short_type_name = resolved_type_name.split('.').last().unwrap_or(&resolved_type_name);
                                            if ctor_name == short_type_name || &resolved_type_name == ctor_name {
                                                // Convert Named fields to the same format as Record fields
                                                let def_fields: Vec<(String, Type, bool)> = named_fields.iter()
                                                    .map(|(n, t)| (n.clone(), t.clone(), false))
                                                    .collect();
                                                Some((params.clone(), def_fields))
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            };

                            if let Some((params, def_fields)) = record_info {
                                // For generic record types, get params and create substitution from base type args
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
                                            // unify(expected, actual) so error says "expected field_type, found arg_type"
                                            self.unify(substituted_fty, ty);
                                        } else {
                                            return Err(TypeError::NoSuchField {
                                                ty: resolved_type_name.clone(),
                                                field: fname.node.clone(),
                                                resolved_type: None,
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
                            var_bounds: vec![],
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
                                    // unify(expected, actual) so error says "expected field_type, found arg_type"
                                    self.unify(substituted_fty, ty);
                                    provided.insert(fname.clone(), ());
                                } else {
                                    // Too many positional arguments
                                    self.last_error_span = Some(*record_span);
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
                                    // unify(expected, actual) so error says "expected field_type, found arg_type"
                                    self.unify(substituted_fty, ty);
                                    provided.insert(fname.node.clone(), ());
                                } else {
                                    return Err(TypeError::NoSuchField {
                                        ty: resolved_type_name.clone(),
                                        field: fname.node.clone(),
                                        resolved_type: None,
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
                                                    resolved_type: None,
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

                        // Check arity explicitly to produce correct "expected N, found M" message
                        if let Type::Function(ref ctor_fn) = ctor_ty {
                            let expected_params = ctor_fn.required_params.unwrap_or(ctor_fn.params.len());
                            if arg_types.len() != expected_params {
                                // Set error span to the constructor call site, not the outer function definition
                                self.last_error_span = Some(*record_span);
                                return Err(TypeError::ArityMismatch {
                                    expected: expected_params,
                                    found: arg_types.len(),
                                });
                            }
                        }

                        let ret_ty = self.fresh();
                        let expected_func_ty = Type::Function(FunctionType {
                            required_params: None,
                            type_params: vec![],
                            params: arg_types.clone(),
                            ret: Box::new(ret_ty.clone()),
                            var_bounds: vec![],
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
            Expr::FieldAccess(expr, field, field_access_span) => {
                // Extract the base name from Var, unit Record, or nested FieldAccess.
                // Supports nested modules: Outer.Inner.func => "Outer.Inner"
                let base_name = Self::extract_qualified_name(expr);

                // Check if this is a qualified function name like "Panel.show"
                // In this case, look up "Panel.show" in the functions environment
                if let Some(ref base) = base_name {
                    let qualified_name = format!("{}.{}", base, field.node);
                    if let Some(fn_type) = self.env.functions.get(&qualified_name).cloned() {
                        // Return the function type, instantiated with fresh type variables
                        return Ok(self.instantiate_function(&fn_type));
                    }
                    // Fallback: try arity-suffixed lookup for module-qualified functions.
                    // Functions are registered as "Module.func/_" (with arity suffix) but
                    // FieldAccess references use bare "Module.func". Search for any arity
                    // match so module functions can be used as first-class values and in
                    // pipeline operators (|>).
                    {
                        let prefix = format!("{}/", qualified_name);
                        let found = self.env.functions.iter()
                            .find(|(k, _)| k.starts_with(&prefix))
                            .map(|(_, v)| v.clone());
                        if let Some(fn_type) = found {
                            return Ok(self.instantiate_function(&fn_type));
                        }
                    }
                    // Also check if this is a module-qualified constructor (e.g., MyModule.Blank)
                    // Unit variant constructors don't appear in env.functions since they take no args,
                    // but they should be resolvable via lookup_constructor using the field name.
                    // We check both the qualified name and the unqualified field name.
                    if let Some(ty) = self.lookup_constructor(&qualified_name) {
                        return Ok(ty);
                    }
                    if let Some(ty) = self.lookup_constructor(&field.node) {
                        return Ok(ty);
                    }
                    // Check if this is a module-qualified binding (e.g., Constants.greeting)
                    // Module bindings are registered in env.bindings with their qualified name
                    if let Some((ty, _)) = self.env.lookup(&qualified_name) {
                        return Ok(ty.clone());
                    }
                }

                // Regular field access on a value
                let expr_ty = self.infer_expr(expr)?;
                let field_ty = self.fresh();

                // Eagerly resolve field type when base type is already concrete.
                // This ensures overload selection has resolved arg types
                // (e.g. `showThing(l.start)` knows arg is Point, not a Var).
                let resolved_base = self.env.apply_subst(&expr_ty);
                match &resolved_base {
                    Type::Named { name, args } => {
                        if let Some(typedef) = self.env.lookup_type(name).cloned() {
                            match &typedef {
                                crate::TypeDef::Record { params, fields, .. } => {
                                    if let Some((_, actual_ty, _)) =
                                        fields.iter().find(|(n, _, _)| n == &field.node)
                                    {
                                        let subst: std::collections::HashMap<String, Type> = params.iter()
                                            .map(|p| p.name.clone())
                                            .zip(args.iter().cloned())
                                            .collect();
                                        let substituted = Self::substitute_type_params(actual_ty, &subst);
                                        let _ = self.unify_types(&field_ty, &substituted);
                                    }
                                }
                                crate::TypeDef::Variant { params, constructors } => {
                                    if constructors.len() == 1 {
                                        if let Some(crate::Constructor::Named(_, fields)) = constructors.first() {
                                            if let Some((_, actual_ty)) =
                                                fields.iter().find(|(n, _)| n == &field.node)
                                            {
                                                let subst: std::collections::HashMap<String, Type> = params.iter()
                                                    .map(|p| p.name.clone())
                                                    .zip(args.iter().cloned())
                                                    .collect();
                                                let substituted = Self::substitute_type_params(actual_ty, &subst);
                                                let _ = self.unify_types(&field_ty, &substituted);
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Type::Record(rec) => {
                        if let Some((_, actual_ty, _)) =
                            rec.fields.iter().find(|(n, _, _)| n == &field.node)
                        {
                            let _ = self.unify_types(&field_ty, actual_ty);
                        }
                    }
                    _ => {}
                }

                self.require_field_at(expr_ty, &field.node, field_ty.clone(), Some(*field_access_span));
                Ok(field_ty)
            }

            // Index access
            Expr::Index(container, index, span) => {
                let container_ty = self.infer_expr(container)?;
                let index_ty = self.infer_expr(index)?;
                let elem_ty = self.fresh();

                // Extract literal integer index for tuple element type resolution
                let literal_index = match index.as_ref() {
                    Expr::Int(n, _) => Some(*n),
                    _ => None,
                };

                // Apply substitutions to get resolved container type
                let resolved_container = self.env.apply_subst(&container_ty);

                // Check if container is a custom Named type - allow indexing without List unification
                // The compiler will dispatch to {typeLower}Get function
                match &resolved_container {
                    Type::Var(_) => {
                        // Unresolved type variable - defer the index check until after solve()
                        // This correctly handles cases like m["key"] where m is a Map returned
                        // from a function call whose return type hasn't been resolved yet.
                        self.deferred_index_checks.push((container_ty.clone(), index_ty, elem_ty.clone(), *span, literal_index));
                        Ok(elem_ty)
                    }
                    Type::Named { .. } => {
                        // Custom type - allow Int indexing
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
                    Type::Tuple(elems) => {
                        // Tuple indexing: t[i] where i is Int
                        self.unify(index_ty, Type::Int);
                        // If index is a literal integer, resolve to the specific element type
                        // This enables heterogeneous tuple access: t[0]: String, t[1]: Int
                        if let Some(n) = literal_index {
                            let idx = n as usize;
                            if idx < elems.len() {
                                self.constraints.push(Constraint::Equal(
                                    elem_ty.clone(),
                                    elems[idx].clone(),
                                    Some(*span),
                                ));
                            }
                        }
                        Ok(elem_ty)
                    }
                    Type::String => {
                        // String indexing: s[i] returns String (single character)
                        self.unify(index_ty, Type::Int);
                        self.constraints.push(Constraint::Equal(
                            elem_ty.clone(),
                            Type::String,
                            Some(*span),
                        ));
                        Ok(elem_ty)
                    }
                    Type::Array(arr_elem_ty) => {
                        // Typed array indexing: arr[i] returns element type
                        self.unify(index_ty, Type::Int);
                        self.constraints.push(Constraint::Equal(
                            elem_ty.clone(),
                            (**arr_elem_ty).clone(),
                            Some(*span),
                        ));
                        Ok(elem_ty)
                    }
                    _ => {
                        // Container is List (default assumption)
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
                                // Infer the value expression type
                                let value_ty = self.infer_expr(&binding.value)?;
                                // Unify immediately (may be caught by solve)
                                self.unify(value_ty.clone(), var_ty.clone());
                                // Also register as deferred typed binding check so that
                                // after solve() resolves type vars, the mismatch is caught.
                                // This is needed because batch unify errors are often dropped.
                                self.deferred_typed_binding_checks.push((
                                    value_ty,
                                    var_ty.clone(),
                                    binding.span,
                                ));
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
                                    let obj_ty = self.infer_expr(obj)?;
                                    let idx_ty = self.infer_expr(idx)?;
                                    // Constrain value type based on container type
                                    let resolved_obj = self.env.apply_subst(&obj_ty);
                                    match &resolved_obj {
                                        Type::Map(key_ty, val_ty) => {
                                            self.unify(idx_ty, (**key_ty).clone());
                                            self.unify(expr_ty, (**val_ty).clone());
                                        }
                                        Type::List(elem_ty) => {
                                            self.unify(idx_ty, Type::Int);
                                            self.unify(expr_ty, (**elem_ty).clone());
                                        }
                                        _ => {
                                            // Unknown container type - no constraint
                                        }
                                    }
                                }
                            }
                        }
                        Stmt::LocalFnDef(fn_def) => {
                            // Local function definition with default params.
                            // Infer the function type and bind the name to it.
                            if let Some(clause) = fn_def.clauses.first() {
                                let mut param_tys = Vec::new();
                                for param in &clause.params {
                                    let ty = self.fresh();
                                    if let nostos_syntax::ast::Pattern::Var(ident) = &param.pattern {
                                        self.env.bind(ident.node.clone(), ty.clone(), false);
                                    }
                                    param_tys.push(ty);
                                }
                                let body_ty = self.infer_expr(&clause.body).unwrap_or_else(|_| self.fresh());
                                let required_count = clause.params.iter()
                                    .filter(|p| p.default.is_none())
                                    .count();
                                let required_params = if required_count < clause.params.len() {
                                    Some(required_count)
                                } else {
                                    None
                                };
                                let fn_ty = Type::Function(FunctionType {
                                    params: param_tys,
                                    ret: Box::new(body_ty),
                                    type_params: vec![],
                                    required_params,
                                    var_bounds: vec![],
                                });
                                self.env.bind(fn_def.name.node.clone(), fn_ty, false);
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
                        // Use arity-aware lookup to find functions registered with arity suffixes
                        // (e.g., "EqCheck.hasDup/_" for a module function with 1 param).
                        // Direct self.env.functions.get() misses these because module function
                        // keys always include the arity suffix.
                        let overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(&qualified_name, args.len())
                            .into_iter().cloned().collect();
                        if let Some(fn_type) = overloads.into_iter().next() {
                            // Found the function - infer argument types and unify
                            let has_named_args = args.iter().any(|a| matches!(a, CallArg::Named(_, _)));
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
                                // When named args are present, skip positional type checking.
                                // The compiler will reorder args by name at code generation time.
                                if has_named_args {
                                    self.deferred_collection_ret_checks.push(((*ft.ret).clone(), *call_span));
                                    return Ok(*ft.ret);
                                }
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
                                // unify_at(expected=param, actual=arg) so error says "expected param_type, found arg_type"
                                for (i, (param_ty, arg_ty)) in ft.params.iter().zip(arg_types.iter()).enumerate() {
                                    self.unify_at(param_ty.clone(), arg_ty.clone(), *call_span);
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
                    // Use arity-aware lookup to find ALL overloads (not just the first one).
                    // This is critical for functions with multiple overloads like:
                    //   add(a: Int, b: Int) = a + b
                    //   add(a: String, b: String) = a ++ b
                    // Without overload resolution, the wrong overload might be selected,
                    // causing spurious type errors.
                    let overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(&qualified_name, args.len())
                        .into_iter().cloned().collect();
                    if !overloads.is_empty() {
                        // Infer argument types first (needed for overload resolution)
                        let has_named_args = args.iter().any(|a| matches!(a, CallArg::Named(_, _)));
                        let mut arg_types = Vec::new();
                        for arg in args {
                            let expr = match arg {
                                CallArg::Positional(e) | CallArg::Named(_, e) => e,
                            };
                            arg_types.push(self.infer_expr(expr)?);
                        }

                        // Select the best overload based on argument types
                        let fn_type = if overloads.len() == 1 {
                            overloads.into_iter().next().unwrap()
                        } else {
                            let overload_refs: Vec<&FunctionType> = overloads.iter().collect();
                            let (best_idx, is_ambiguous) = if has_named_args {
                                (Some(0), false) // Use first overload for named args; positional check skipped
                            } else {
                                self.find_best_overload_idx(&overload_refs, &arg_types)
                            };
                            if is_ambiguous {
                                // Defer overload resolution until types are better known
                                let ret_ty = self.fresh();
                                self.deferred_overload_calls.push((
                                    overloads.clone(),
                                    arg_types.clone(),
                                    ret_ty.clone(),
                                    *call_span,
                                ));
                                return Ok(ret_ty);
                            }
                            overloads[best_idx.unwrap_or(0)].clone()
                        };

                        // Instantiate the function type
                        let func_ty = self.instantiate_function(&fn_type);
                        if let Type::Function(ft) = func_ty {
                            // When named args are present, skip positional type checking.
                            // The compiler will reorder args by name at code generation time.
                            // Return the function's return type directly.
                            if has_named_args {
                                self.deferred_collection_ret_checks.push(((*ft.ret).clone(), *call_span));
                                return Ok(*ft.ret);
                            }
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
                            // NOTE: param_ty FIRST, arg_ty SECOND so errors say "expected param, found arg"
                            for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                                self.unify_at(param_ty.clone(), arg_ty.clone(), *call_span);
                                // Record for post-solve structural mismatch checking
                                self.deferred_fn_call_checks.push((param_ty.clone(), arg_ty.clone(), *call_span));
                            }
                            // Save return type for post-solve Set/Map Hash+Eq checking
                            self.deferred_collection_ret_checks.push(((*ft.ret).clone(), *call_span));
                            return Ok(*ft.ret);
                        }
                    }
                }

                // Check if receiver is a nested module path: Outer.Inner.func(args)
                // This handles deeply qualified function calls via FieldAccess chain
                if let Some(receiver_base) = Self::extract_qualified_name(receiver) {
                    let qualified_name = format!("{}.{}", receiver_base, method.node);
                    let overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(&qualified_name, args.len())
                        .into_iter().cloned().collect();
                    if let Some(fn_type) = overloads.into_iter().next() {
                        let has_named_args = args.iter().any(|a| matches!(a, CallArg::Named(_, _)));
                        let mut arg_types = Vec::new();
                        for arg in args {
                            let expr = match arg {
                                CallArg::Positional(e) | CallArg::Named(_, e) => e,
                            };
                            arg_types.push(self.infer_expr(expr)?);
                        }
                        let func_ty = self.instantiate_function(&fn_type);
                        if let Type::Function(ft) = func_ty {
                            // When named args are present, skip positional type checking.
                            // The compiler will reorder args by name at code generation time.
                            if has_named_args {
                                self.deferred_collection_ret_checks.push(((*ft.ret).clone(), *call_span));
                                return Ok(*ft.ret);
                            }
                            let min_args = ft.required_params.unwrap_or(ft.params.len());
                            let max_args = ft.params.len();
                            if arg_types.len() < min_args || arg_types.len() > max_args {
                                self.last_error_span = Some(*call_span);
                                return Err(TypeError::ArityMismatch {
                                    expected: if min_args == max_args { max_args } else { min_args },
                                    found: arg_types.len(),
                                });
                            }
                            // NOTE: param_ty FIRST, arg_ty SECOND so errors say "expected param, found arg"
                            for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                                self.unify_at(param_ty.clone(), arg_ty.clone(), *call_span);
                                self.deferred_fn_call_checks.push((param_ty.clone(), arg_ty.clone(), *call_span));
                            }
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

                // Try immediate UFCS lookup if receiver type is already resolved.
                // Skip when there are named args - let the deferred check_pending_method_calls
                // handle reordering them to the correct parameter positions.
                if named_args.is_empty() {
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
                            // NOTE: param_ty FIRST, arg_ty SECOND so errors say "expected param, found arg"
                            for (param_ty, arg_ty) in ft.params.iter().zip(arg_types.iter()) {
                                self.unify_at(param_ty.clone(), arg_ty.clone(), *call_span);
                            }
                            // Unify return type
                            self.unify_at(ret_ty.clone(), *ft.ret, *call_span);
                            return Ok(ret_ty);
                        }
                    }
                }
                } // end named_args.is_empty() guard

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

            // Try expression (error propagation via ?)
            // At runtime, ? is exception propagation (catch+rethrow), not Result unwrapping.
            // The inner expression's value passes through if no exception is thrown.
            Expr::Try_(inner, _) => {
                let inner_ty = self.infer_expr(inner)?;
                Ok(inner_ty)
            }

            // Do block (IO monad)
            Expr::Do(stmts, _) => {
                // Do blocks compile like regular blocks at the bytecode level.
                // IO[T] is purely a type-level annotation; at runtime T and IO[T] are identical.
                // Treat bind statements (x = expr) as let-bindings and expression statements
                // like block statements — result type is the type of the last expression.
                let saved_bindings = self.env.bindings.clone();
                let mut result_ty = Type::Unit;

                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i == stmts.len() - 1;
                    match stmt {
                        nostos_syntax::ast::DoStmt::Bind(pat, expr) => {
                            let expr_ty = self.infer_expr(expr)?;
                            self.infer_pattern(pat, &expr_ty)?;
                            if is_last {
                                result_ty = expr_ty;
                            }
                        }
                        nostos_syntax::ast::DoStmt::Expr(expr) => {
                            let ty = self.infer_expr(expr)?;
                            if is_last {
                                result_ty = ty;
                            }
                        }
                    }
                }

                self.env.bindings = saved_bindings;
                Ok(result_ty)
            }

            // Spawn
            Expr::Spawn(_, func, args, _) => {
                // Lambda and Var are already callable - unify with function type.
                // Everything else (Block, Call, Send, etc.) gets auto-wrapped in a
                // thunk by the compiler, so we just infer the inner expression type.
                let is_already_callable = matches!(func.as_ref(),
                    Expr::Lambda(_, _, _) | Expr::Var(_));

                if is_already_callable {
                    // func should be a callable (Lambda or Var pointing to a function)
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
                        var_bounds: vec![],
                    });
                    self.unify(func_ty, expected);
                } else {
                    // Auto-wrapped: just infer the expression type
                    let _inner_ty = self.infer_expr(func)?;
                    for arg in args {
                        let _ = self.infer_expr(arg)?;
                    }
                }

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

            // Try/catch - unify try body with catch arm types (like if/then/else)
            Expr::Try(body, catch_arms, finally, _) => {
                let _body_ty = self.infer_expr(body)?;

                // Exception type is a fresh variable since throw() can throw any type
                let err_ty = self.fresh();

                // try/catch is dynamically typed - branches can return different types
                // (like Python). Do NOT unify try body with catch arm types.
                for arm in catch_arms {
                    let catch_result_ty = self.fresh();
                    self.infer_match_arm(arm, &err_ty, &catch_result_ty)?;
                }

                if let Some(finally_expr) = finally {
                    let _ = self.infer_expr(finally_expr)?;
                }

                // Return a fresh type - the actual type depends on which branch runs
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

            // Type ascription: `expr` constrained to be `Type`
            // Used for lambda return type annotations: `(x: Int) -> RetType => body`
            Expr::TypeAscription(inner, ty_expr, _) => {
                let inner_ty = self.infer_expr(inner)?;
                let ascribed_ty = self.type_from_ast(ty_expr);
                self.unify(inner_ty, ascribed_ty.clone());
                Ok(ascribed_ty)
            }

            // While loop: condition must be Bool, returns Unit or break-value type
            Expr::While(cond, body, _) => {
                let cond_ty = self.infer_expr(cond)?;
                self.unify(cond_ty, Type::Bool);
                // Push a fresh type var for any break value
                let break_tv = self.fresh();
                self.loop_break_type_stack.push(break_tv.clone());
                let _ = self.infer_expr(body)?;
                self.loop_break_type_stack.pop();
                // The while loop returns the break type var (may be Unit if no break with value)
                Ok(break_tv)
            }

            // For loop: iterates from start to end (both Int), returns Unit or break-value type
            Expr::For(var, start, end, body, _) => {
                let start_ty = self.infer_expr(start)?;
                let end_ty = self.infer_expr(end)?;
                self.unify(start_ty, Type::Int);
                self.unify(end_ty, Type::Int);
                // Add loop variable to scope
                let saved_bindings = self.env.bindings.clone();
                self.env.bind(var.node.clone(), Type::Int, false);
                // Push a fresh type var for any break value
                let break_tv = self.fresh();
                self.loop_break_type_stack.push(break_tv.clone());
                let _ = self.infer_expr(body)?;
                self.loop_break_type_stack.pop();
                self.env.bindings = saved_bindings;
                Ok(break_tv)
            }

            // Break: optional value, returns Never (doesn't produce a value normally)
            Expr::Break(value, _) => {
                if let Some(val) = value {
                    let val_ty = self.infer_expr(val)?;
                    // Unify with the current loop's break type
                    if let Some(break_tv) = self.loop_break_type_stack.last().cloned() {
                        self.unify(break_tv, val_ty);
                    }
                } else {
                    // bare break: unify loop break type with Unit
                    if let Some(break_tv) = self.loop_break_type_stack.last().cloned() {
                        self.unify(break_tv, Type::Unit);
                    }
                }
                Ok(Type::Unit) // Break doesn't continue execution normally
            }

            // Continue: returns Never (doesn't produce a value normally)
            Expr::Continue(_) => Ok(Type::Unit),

            // Return: optional value, returns Never (doesn't produce a value normally)
            Expr::Return(value, _) => {
                // Mark that this function has a `return` statement
                if let Some(flag) = self.fn_has_return_stack.last_mut() {
                    *flag = true;
                }
                if let Some(val) = value {
                    let val_ty = self.infer_expr(val)?;
                    // Unify with the current function's return type variable so that
                    // early returns contribute to the inferred return type.
                    if let Some(ret_tv) = self.fn_return_type_stack.last().cloned() {
                        self.unify(ret_tv, val_ty);
                    }
                } else {
                    // bare return: unify function return type with Unit
                    if let Some(ret_tv) = self.fn_return_type_stack.last().cloned() {
                        self.unify(ret_tv, Type::Unit);
                    }
                }
                Ok(Type::Unit) // Return doesn't continue execution normally
            }
        }
    }

    /// Infer types for a binary operation.
    fn infer_binop(&mut self, left: &Expr, op: BinOp, right: &Expr, span: Span) -> Result<Type, TypeError> {
        // For Pipe operator, handle RHS Call specially: a |> f(b) => f(a, b)
        if op == BinOp::Pipe {
            if let Expr::Call(func, type_args, call_args, call_span) = right {
                let _left_ty = self.infer_expr(left)?;
                // Build new args: [piped_value, ...original_args]
                let mut new_args = vec![CallArg::Positional(left.clone())];
                new_args.extend(call_args.iter().cloned());
                // Create a synthetic Call expression with the piped value prepended
                let new_call = Expr::Call(func.clone(), type_args.clone(), new_args, *call_span);
                return self.infer_expr(&new_call);
            }
        }

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
                    // Type variable + concrete numeric
                    (Type::Var(id), Type::Float | Type::Int | Type::Float64 | Type::Int64) => {
                        if self.field_result_vars.contains(id) {
                            // This Var is from a field access (HasField result) - DON'T unify.
                            // Premature ?Y=Int would lock the wrong type before HasField
                            // resolves ?Y to the actual field type (e.g., String).
                            // The Num trait bound will be captured by enrichment and
                            // checked at call sites after HasField resolution.
                            self.require_trait(left_ty.clone(), "Num");
                            Ok(left_ty)
                        } else {
                            // Normal Var (lambda param, etc.) - unify to infer the type
                            self.unify(left_ty.clone(), right_ty.clone());
                            self.require_trait(left_ty.clone(), "Num");
                            Ok(left_ty)
                        }
                    }
                    (Type::Float | Type::Int | Type::Float64 | Type::Int64, Type::Var(id)) => {
                        if self.field_result_vars.contains(id) {
                            self.require_trait(right_ty.clone(), "Num");
                            Ok(right_ty)
                        } else {
                            self.unify(left_ty.clone(), right_ty.clone());
                            self.require_trait(left_ty.clone(), "Num");
                            Ok(left_ty)
                        }
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
                self.require_trait_at(left_ty, "Ord", span);
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
                // Expand type aliases (e.g., `type Name = String`)
                let resolved_left = self.expand_type_alias(&resolved_left).unwrap_or(resolved_left);
                let resolved_right = self.expand_type_alias(&resolved_right).unwrap_or(resolved_right);

                match (&resolved_left, &resolved_right) {
                    // Both are concrete list types.
                    // Nostos supports heterogeneous lists (like Python), so mismatched
                    // element types are allowed. We attempt unification only when both
                    // element types are the SAME variable (already linked) or when one
                    // of them is a type variable that can absorb the other's concrete type.
                    // Critically, we must NOT merge two independent type variables here:
                    // doing so causes `["hello"] ++ [1, 2, 3]` to fail because both element
                    // vars get merged, then later one gets constrained to String and the
                    // other to Int, producing a spurious "expected String, found Int" error.
                    (Type::List(elem_left), Type::List(elem_right)) => {
                        let resolved_el = self.env.apply_subst(elem_left);
                        let resolved_er = self.env.apply_subst(elem_right);
                        // Only unify element types when they would clearly benefit from it:
                        // - Both are the same type (no-op, harmless)
                        // - Exactly one is a type variable (propagate concrete type into var)
                        // Do NOT unify two independent type variables or two different concrete
                        // types, as that would prevent heterogeneous list concatenation.
                        let should_unify = match (&resolved_el, &resolved_er) {
                            _ if resolved_el == resolved_er => true,    // Same type - safe no-op
                            (Type::Var(_), Type::Var(_)) => false,      // Two independent vars - skip
                            (Type::Var(_), _) | (_, Type::Var(_)) => true, // One var - propagate
                            _ => false,                                 // Two different concrete types - skip
                        };
                        if should_unify {
                            let _ = self.unify_types(elem_left, elem_right);
                        }
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
                        Err(TypeError::UnificationFailed(
                            resolved_left.clone(),
                            resolved_right.clone(),
                        ))
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
                        Err(TypeError::MissingTraitImpl {
                            ty: bad_type.display(),
                            trait_name: "Concat".to_string(),
                            resolved_type: Some(bad_type.clone()),
                        })
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
                    var_bounds: vec![],
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
                // StringCons is ambiguous: ["x" | rest] could match a String (char decomposition)
                // or a List[String] (list element match). Check existing constraints to decide.
                // Look through existing constraints to see if expected type is already
                // constrained to be a List type.
                // Follow the chain of Equal constraints to find what expected resolves to.
                // We need to handle Var(a) = Var(b) = List[String] chains.
                let is_constrained_as_list = {
                    let mut is_list = matches!(expected, Type::List(_));
                    if !is_list {
                        // Collect all var IDs that are transitively equal to expected
                        let mut var_ids: Vec<u32> = Vec::new();
                        if let Type::Var(id) = expected {
                            var_ids.push(*id);
                        }
                        let mut changed = true;
                        while changed && !is_list {
                            changed = false;
                            for c in &self.constraints {
                                if let Constraint::Equal(t1, t2, _) = c {
                                    // Check if either side is one of our tracked vars
                                    let t1_is_tracked = if let Type::Var(id) = t1 { var_ids.contains(id) } else { false };
                                    let t2_is_tracked = if let Type::Var(id) = t2 { var_ids.contains(id) } else { false };

                                    if t1_is_tracked {
                                        if matches!(t2, Type::List(_)) { is_list = true; break; }
                                        if let Type::Var(id2) = t2 {
                                            if !var_ids.contains(id2) { var_ids.push(*id2); changed = true; }
                                        }
                                    }
                                    if t2_is_tracked {
                                        if matches!(t1, Type::List(_)) { is_list = true; break; }
                                        if let Type::Var(id1) = t1 {
                                            if !var_ids.contains(id1) { var_ids.push(*id1); changed = true; }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    is_list
                };

                if is_constrained_as_list {
                    // Expected type is a list - treat as list pattern with string literal elements
                    let elem_ty = self.fresh();
                    self.unify(expected.clone(), Type::List(Box::new(elem_ty.clone())));

                    match string_pat {
                        nostos_syntax::ast::StringPattern::Empty => {}
                        nostos_syntax::ast::StringPattern::Cons(strings, tail_pat) => {
                            for _ in strings {
                                self.unify(elem_ty.clone(), Type::String);
                            }
                            self.infer_pattern(tail_pat, &Type::List(Box::new(elem_ty)))?;
                        }
                    }
                } else {
                    // Default: treat as string prefix matching
                    self.unify(expected.clone(), Type::String);
                    match string_pat {
                        nostos_syntax::ast::StringPattern::Empty => {}
                        nostos_syntax::ast::StringPattern::Cons(_, tail_pat) => {
                            self.infer_pattern(tail_pat, &Type::String)?;
                        }
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
                // Look up constructor, using the expected type (from the match scrutinee) to
                // disambiguate when multiple types have the same constructor name.
                // E.g., if expected = StrTree, prefer StrTree.Node over IntTree.Node.
                // Apply substitution first to resolve Var types that may have been solved already.
                let saved_declared_return_for_pat = self.current_declared_return_type.take();
                let resolved_expected = self.env.apply_subst(expected);
                if let Type::Named { name: expected_type_name, .. } = &resolved_expected {
                    // Resolved expected type: use it to disambiguate the constructor.
                    let base_name = expected_type_name.rsplit('.').next().unwrap_or(expected_type_name.as_str());
                    self.current_declared_return_type = Some(base_name.to_string());
                } else if saved_declared_return_for_pat.is_some() {
                    // Expected type is still an unresolved variable, but we have a hint from
                    // the surrounding context (e.g. infer_match_arm set it from the result type).
                    // Keep using that hint so lookup_constructor can disambiguate.
                    self.current_declared_return_type = saved_declared_return_for_pat.clone();
                }
                let ctor_ty_opt = self.lookup_constructor(&name.node);
                self.current_declared_return_type = saved_declared_return_for_pat;
                if let Some(ctor_ty) = ctor_ty_opt {
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

                                // Try to get the constructor's named fields for validation.
                                // IMPORTANT: f.params already contains INSTANTIATED types (fresh vars
                                // from lookup_constructor). We must use those, not raw types from the
                                // typedef, to correctly link field types to the constructor's type params.
                                //
                                // For Record types (e.g., `type Box[a] = Box { val: a }`):
                                //   f.params = [Var(N)]  (the instantiated 'a')
                                //   f.ret = Box[Var(N)]
                                //   lookup_variant_named_fields returns None (it only handles Variants)
                                //   -> We use record field names paired with f.params by index.
                                //
                                // For Variant Named constructors (e.g., `type T = Ctor { field: a }`):
                                //   lookup_variant_named_fields returns raw TypeParam types (NOT fresh vars)
                                //   -> We must use f.params by index too.
                                let ctor_fields: Option<Vec<(String, Type)>> =
                                    if let Type::Named { name: type_name, .. } = &*f.ret {
                                        // First try Variant named fields (for name lookup only)
                                        let variant_fields = self.env.lookup_variant_named_fields(type_name, &name.node);
                                        if let Some(variant_field_defs) = variant_fields {
                                            // Variant: use field names from typedef, but types from f.params
                                            Some(variant_field_defs.iter().zip(f.params.iter())
                                                .map(|((fname, _), instantiated_ty)| (fname.clone(), instantiated_ty.clone()))
                                                .collect())
                                        } else {
                                            // Record type: use record field names paired with f.params
                                            self.env.lookup_record_field_names(type_name)
                                                .map(|field_names| {
                                                    field_names.iter().zip(f.params.iter())
                                                        .map(|(fname, instantiated_ty)| (fname.clone(), instantiated_ty.clone()))
                                                        .collect()
                                                })
                                        }
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
                                                        resolved_type: None,
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
                                                        resolved_type: None,
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

            Pattern::TypeAnnotated(inner_pat, type_expr, _) => {
                // Convert the type annotation to a Type and unify with the expected type.
                // This enforces that when the lambda is called, the argument must match the annotation.
                let annotated_ty = self.type_from_ast(type_expr);
                self.unify(expected.clone(), annotated_ty);
                self.infer_pattern(inner_pat, expected)?;
                Ok(())
            }
        }
    }

    /// Look up a constructor and return its type.
    /// Extract a fully-qualified module path from nested FieldAccess expressions.
    /// E.g., FieldAccess(FieldAccess(Var("Outer"), "Inner"), "func") => Some("Outer.Inner.func")
    fn extract_qualified_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(ident) => Some(ident.node.clone()),
            Expr::Record(ident, fields, _) if fields.is_empty() => Some(ident.node.clone()),
            Expr::FieldAccess(base, field, _) => {
                Self::extract_qualified_name(base).map(|base_name| {
                    format!("{}.{}", base_name, field.node)
                })
            }
            _ => None,
        }
    }

    fn lookup_constructor(&mut self, name: &str) -> Option<Type> {
        // Search through all types for this constructor.
        // Sort type names to prioritize user-defined types over stdlib types,
        // and types from the current module/declared return type over others.
        let types_clone = self.env.types.clone();
        let mut type_names: Vec<_> = types_clone.keys().collect();
        let current_module = self.env.current_module.clone();
        let declared_return = self.current_declared_return_type.clone();

        // First pass: collect all type names that have a matching constructor.
        // Use this to detect ambiguity and prefer based on declared return type.
        fn ctor_name_matches_type(type_name: &str, ctor_name: &str, search_name: &str) -> bool {
            if ctor_name != search_name {
                // Also check qualified name matching
                if let Some(dot_pos) = search_name.rfind('.') {
                    let module_prefix = &search_name[..dot_pos];
                    let local_name = &search_name[dot_pos + 1..];
                    return local_name == ctor_name && type_name.starts_with(&format!("{}.", module_prefix));
                }
                return false;
            }
            true
        }

        let mut matching_type_names: Vec<String> = Vec::new();
        for (tn, def) in &types_clone {
            let local_tn = tn.rsplit('.').next().unwrap_or(tn);
            // Record type with matching name (for record types)
            let record_matches = if tn.as_str() == name {
                true
            } else if let Some(dot_pos) = tn.rfind('.') {
                &tn[dot_pos + 1..] == name
            } else {
                false
            };
            if record_matches {
                if matches!(def, TypeDef::Record { .. }) {
                    matching_type_names.push(tn.clone());
                }
            }
            // Check variant constructors
            if let TypeDef::Variant { constructors, .. } = def {
                for ctor in constructors {
                    let ctor_name = match ctor {
                        Constructor::Unit(n) => n.as_str(),
                        Constructor::Positional(n, _) => n.as_str(),
                        Constructor::Named(n, _) => n.as_str(),
                    };
                    if ctor_name_matches_type(tn, ctor_name, name) {
                        matching_type_names.push(tn.clone());
                        break;
                    }
                }
            }
        }

        // If multiple types have this constructor and we have a declared return type,
        // try to pick the one that matches the declared return type.
        // This disambiguates when e.g. both `IntTree` and `StrTree` have `Node` constructors.
        let preferred_type: Option<String> = if matching_type_names.len() > 1 {
            if let Some(ref ret_ty_name) = declared_return {
                // Normalize: strip module prefix for comparison
                let ret_base = ret_ty_name.rsplit('.').next().unwrap_or(ret_ty_name.as_str());
                // Priority order: exact name match > non-stdlib base match > stdlib base match
                // This ensures user-defined types shadow stdlib types with the same base name.
                // (e.g., user-defined `ValidationResult` shadows `stdlib.validation.ValidationResult`)
                let exact_match = matching_type_names.iter()
                    .find(|tn| tn.as_str() == ret_ty_name.as_str())
                    .cloned();
                let found = if exact_match.is_some() {
                    exact_match
                } else {
                    // Among base-name matches, prefer non-stdlib types
                    let non_stdlib = matching_type_names.iter()
                        .find(|tn| {
                            let tn_base = tn.rsplit('.').next().unwrap_or(tn.as_str());
                            (tn_base == ret_base) && !tn.starts_with("stdlib.")
                        })
                        .cloned();
                    if non_stdlib.is_some() {
                        non_stdlib
                    } else {
                        matching_type_names.iter()
                            .find(|tn| {
                                let tn_base = tn.rsplit('.').next().unwrap_or(tn.as_str());
                                tn_base == ret_base || tn.as_str() == ret_ty_name.as_str()
                            })
                            .cloned()
                    }
                };
                found
            } else {
                None
            }
        } else {
            None
        };

        type_names.sort_by(|a, b| {
            // Priority: preferred type (from declared return type) first,
            // then current module's types, then other user types, then built-in types, then stdlib types.
            // This ensures that within a single file, Node(v, StrLeaf, StrLeaf) in a function
            // returning StrTree finds StrTree.Node, not IntTree.Node.
            let builtin_types = ["List", "Option", "Result"];
            let a_is_stdlib = a.starts_with("stdlib.");
            let b_is_stdlib = b.starts_with("stdlib.");
            let a_is_builtin = !a_is_stdlib && builtin_types.contains(&a.as_str());
            let b_is_builtin = !b_is_stdlib && builtin_types.contains(&b.as_str());
            let a_is_current = current_module.as_ref().map(|m| a.starts_with(m.as_str())).unwrap_or(false);
            let b_is_current = current_module.as_ref().map(|m| b.starts_with(m.as_str())).unwrap_or(false);
            let a_is_preferred = preferred_type.as_ref().map(|p| a.as_str() == p.as_str()).unwrap_or(false);
            let b_is_preferred = preferred_type.as_ref().map(|p| b.as_str() == p.as_str()).unwrap_or(false);
            // Lower number = higher priority (sorted ascending)
            let a_priority = if a_is_preferred { 0 } else if a_is_current { 1 } else if !a_is_stdlib && !a_is_builtin { 2 } else if a_is_builtin { 3 } else { 4 };
            let b_priority = if b_is_preferred { 0 } else if b_is_current { 1 } else if !b_is_stdlib && !b_is_builtin { 2 } else if b_is_builtin { 3 } else { 4 };
            a_priority.cmp(&b_priority).then(a.cmp(b))
        });
        for type_name in type_names {
            let def = types_clone.get(type_name).expect("type_name from types_clone keys");
            // Check if this is a record type with the same name as the constructor
            // Record types can be constructed using their type name as a constructor
            // Also match cross-module: type_name "Errors.AppError" matches constructor "AppError"
            let record_matches = if type_name == name {
                true
            } else if let Some(dot_pos) = type_name.rfind('.') {
                &type_name[dot_pos + 1..] == name
            } else {
                false
            };
            if record_matches {
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
                            var_bounds: vec![],
                        }));
                    }
                }
            }

            if let TypeDef::Variant { params, constructors } = def {
                for ctor in constructors {
                    let ctor_name = match ctor {
                        Constructor::Unit(n) => n.as_str(),
                        Constructor::Positional(n, _) => n.as_str(),
                        Constructor::Named(n, _) => n.as_str(),
                    };
                    // Match by exact name, or by local part of a qualified name
                    // e.g., "Shapes.Circle" matches constructor "Circle" in type "Shapes.Shape"
                    let matches = if ctor_name == name {
                        true
                    } else if let Some(dot_pos) = name.rfind('.') {
                        let module_prefix = &name[..dot_pos];
                        let local_name = &name[dot_pos + 1..];
                        local_name == ctor_name && type_name.starts_with(&format!("{}.", module_prefix))
                    } else {
                        false
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
                                    var_bounds: vec![],
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
                                    var_bounds: vec![],
                                }));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Look up a variant constructor's named fields AND return type, using shared fresh vars.
    fn lookup_constructor_named_fields_with_ret(&mut self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        let types_clone = self.env.types.clone();
        for (type_name, def) in &types_clone {
            if let TypeDef::Variant { params, constructors } = def {
                for ctor in constructors {
                    if let Constructor::Named(ctor_name, fields) = ctor {
                        // Match by exact name, or by local part of a qualified name
                        let name_matches = ctor_name == name || {
                            if let Some(dot_pos) = name.rfind('.') {
                                let module_prefix = &name[..dot_pos];
                                let local_name = &name[dot_pos + 1..];
                                local_name == ctor_name && type_name.starts_with(&format!("{}.", module_prefix))
                            } else {
                                false
                            }
                        };
                        if name_matches {
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
    pub fn substitute_type_params(ty: &Type, subst: &HashMap<String, Type>) -> Type {
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
                    var_bounds: vec![],
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

        // After pattern inference, set current_declared_return_type so that constructor
        // expressions in the arm body can be disambiguated when multiple variant types
        // share the same constructor name (e.g., IntTree.Node vs StrTree.Node).
        // Priority: use result_ty if resolved (it's the function return type), otherwise scrutinee_ty.
        let saved_arm_declared_return = self.current_declared_return_type.take();
        {
            let resolved_result = self.env.apply_subst(result_ty);
            let resolved_scrutinee = self.env.apply_subst(scrutinee_ty);
            let hint_name = if let Type::Named { name: n, .. } = &resolved_result {
                Some(n.rsplit('.').next().unwrap_or(n.as_str()).to_string())
            } else if let Type::Named { name: n, .. } = &resolved_scrutinee {
                Some(n.rsplit('.').next().unwrap_or(n.as_str()).to_string())
            } else {
                saved_arm_declared_return.clone()
            };
            self.current_declared_return_type = hint_name;
        }

        // Check guard if present
        if let Some(guard) = &arm.guard {
            let guard_ty = self.infer_expr(guard)?;
            self.unify(guard_ty, Type::Bool);
        }

        // Save whether a `return` was already seen before this arm, so we can detect
        // if THIS arm body contains a `return` statement (making it diverge).
        let had_return_before = self.fn_has_return_stack.last().copied().unwrap_or(false);

        // Infer body
        let body_ty = self.infer_expr(&arm.body)?;

        // Check if this arm body introduced a `return` statement (i.e., the arm diverges).
        let arm_has_return = self.fn_has_return_stack.last().copied().unwrap_or(false) && !had_return_before;

        // If the arm body diverges via `return`, body_ty is Unit (the return stmt type),
        // but the arm is actually compatible with any result type (it never falls through).
        // Skip the unification to avoid false type mismatches with non-Unit arms.
        if !arm_has_return || body_ty != Type::Unit {
            self.unify(body_ty.clone(), result_ty.clone());
        }

        // Restore the declared return type after body inference.
        self.current_declared_return_type = saved_arm_declared_return;

        // Record branch type for post-solve structural container mismatch check.
        // Skip diverging arms to avoid false positives.
        let arm_span = arm.body.span();
        if !arm_has_return || body_ty != Type::Unit {
            self.deferred_branch_type_checks.push((body_ty, result_ty.clone(), arm_span));
        }

        self.env.bindings = saved_bindings;
        Ok(())
    }

    /// Infer types for a binding.
    pub fn infer_binding(&mut self, binding: &Binding) -> Result<Type, TypeError> {
        // Snapshot constraint counts before inferring value, for let-polymorphism analysis
        let constraints_before = self.constraints.len();
        let pending_methods_before = self.pending_method_calls.len();
        let deferred_overloads_before = self.deferred_overload_calls.len();
        // Snapshot var counter to distinguish lambda-internal vars from outer-scope vars
        let vars_before = self.env.next_var;

        // For recursive lambda bindings: pre-register the binding name with a fresh type
        // variable so that recursive references within the lambda body can be resolved.
        // Without this, recursive references fail with UnknownIdent, which causes all
        // constraints from the lambda body (including match type mismatches) to be lost.
        // IMPORTANT: Only do this for lambdas that actually reference their own name
        // (recursive lambdas). Non-recursive lambdas must NOT be pre-registered because
        // it would break let-polymorphism (the fresh var gets unified to a single type,
        // preventing polymorphic use at multiple types).
        let pre_registered_var = if !binding.mutable && matches!(&binding.value, Expr::Lambda(_, _, _)) {
            if let Pattern::Var(ident) = &binding.pattern {
                if expr_references_name(&binding.value, &ident.node) {
                    let fresh_ty = self.fresh();
                    self.env.bind(ident.node.clone(), fresh_ty.clone(), false);
                    Some((ident.node.clone(), fresh_ty))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let value_ty = self.infer_expr(&binding.value)?;

        // If we pre-registered, unify the fresh var with the actual inferred type
        if let Some((_name, fresh_ty)) = &pre_registered_var {
            self.unify(fresh_ty.clone(), value_ty.clone());
        }

        // Let-polymorphism for local lambdas: when a let binding's value is a lambda,
        // mark its free type variables as polymorphic so that each use of the binding
        // gets fresh copies. This enables `id = (x) => x; a = id(42); b = id("hi")`.
        // Only do this for non-mutable bindings (mutable vars should keep a fixed type).
        // IMPORTANT: Only generalize vars that are NOT constrained by the lambda body
        // against external types. A constraint that only relates lambda-internal vars
        // (e.g., from `[x]` creating Equal(?param, ?elem_ty)) does NOT prevent
        // generalization. Only constraints that pin a var to a concrete type or an
        // outer-scope variable prevent generalization. This allows `wrap = (x) => [x]`
        // to be used polymorphically.
        if !binding.mutable && matches!(&binding.value, Expr::Lambda(_, _, _)) {
            if let Type::Function(_) = &value_ty {
                let mut var_ids = Vec::new();
                self.collect_var_ids(&value_ty, &mut var_ids);

                // Collect vars that are constrained against external types/vars.
                // A constraint is "externally anchored" if it involves a concrete type
                // (Int, String, Named, etc.) or a var from before the lambda (id < vars_before).
                // Constraints that only relate lambda-internal vars to each other
                // (e.g., Equal(?param, ?elem_ty) from `[x]`) don't prevent generalization.
                let mut constrained_vars: HashSet<u32> = HashSet::new();
                // Collect internal-only Equal constraints for transitive propagation
                let mut internal_equals: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
                // Collect internal HasTrait constraints to register as trait_bounds
                // for generalized vars (so instantiate_local_binding can copy them)
                let mut internal_has_traits: Vec<(u32, String)> = Vec::new();
                // Helper: analyze a pair of types for external anchoring.
                // Decomposes Function-vs-Function equalities into param/ret pairs
                // so that a concrete return type doesn't prevent generalization of params.
                // E.g., Equal(Function(?25 -> String), Function(?24 -> ?26)) decomposes into
                // Equal(?25, ?24) [internal] and Equal(String, ?26) [external: constrains ?26].
                let mut analyze_constraint_pair = |a: &Type, b: &Type, constrained_vars: &mut HashSet<u32>, internal_equals: &mut Vec<(Vec<u32>, Vec<u32>)>| {
                    // If both sides are Function types, decompose into component constraints
                    if let (Type::Function(ft_a), Type::Function(ft_b)) = (a, b) {
                        // Analyze each param pair separately
                        for (pa, pb) in ft_a.params.iter().zip(ft_b.params.iter()) {
                            let pa_ext = type_has_external_anchor(pa, vars_before);
                            let pb_ext = type_has_external_anchor(pb, vars_before);
                            if pa_ext || pb_ext {
                                let mut ids = Vec::new();
                                self.collect_var_ids(pa, &mut ids);
                                self.collect_var_ids(pb, &mut ids);
                                for id in &ids { constrained_vars.insert(*id); }
                            } else {
                                let mut a_ids = Vec::new();
                                let mut b_ids = Vec::new();
                                self.collect_var_ids(pa, &mut a_ids);
                                self.collect_var_ids(pb, &mut b_ids);
                                if !a_ids.is_empty() || !b_ids.is_empty() {
                                    internal_equals.push((a_ids, b_ids));
                                }
                            }
                        }
                        // Analyze return type separately
                        let ra_ext = type_has_external_anchor(&ft_a.ret, vars_before);
                        let rb_ext = type_has_external_anchor(&ft_b.ret, vars_before);
                        if ra_ext || rb_ext {
                            let mut ids = Vec::new();
                            self.collect_var_ids(&ft_a.ret, &mut ids);
                            self.collect_var_ids(&ft_b.ret, &mut ids);
                            for id in &ids { constrained_vars.insert(*id); }
                        } else {
                            let mut a_ids = Vec::new();
                            let mut b_ids = Vec::new();
                            self.collect_var_ids(&ft_a.ret, &mut a_ids);
                            self.collect_var_ids(&ft_b.ret, &mut b_ids);
                            if !a_ids.is_empty() || !b_ids.is_empty() {
                                internal_equals.push((a_ids, b_ids));
                            }
                        }
                        return;
                    }
                    // Non-function: treat as a single constraint
                    let a_external = type_has_external_anchor(a, vars_before);
                    let b_external = type_has_external_anchor(b, vars_before);
                    if a_external || b_external {
                        let mut ids = Vec::new();
                        self.collect_var_ids(a, &mut ids);
                        self.collect_var_ids(b, &mut ids);
                        for id in &ids { constrained_vars.insert(*id); }
                    } else {
                        let mut a_ids = Vec::new();
                        let mut b_ids = Vec::new();
                        self.collect_var_ids(a, &mut a_ids);
                        self.collect_var_ids(b, &mut b_ids);
                        if !a_ids.is_empty() || !b_ids.is_empty() {
                            internal_equals.push((a_ids, b_ids));
                        }
                    }
                };

                for constraint in &self.constraints[constraints_before..] {
                    match constraint {
                        Constraint::Equal(a, b, _) => {
                            analyze_constraint_pair(a, b, &mut constrained_vars, &mut internal_equals);
                        }
                        Constraint::HasTrait(ty, trait_name, _span) => {
                            // HasTrait constrains only if it involves external types/vars.
                            // A HasTrait(?X, "Num") where ?X is internal to the lambda
                            // should NOT prevent generalization - the trait bound gets
                            // copied to fresh vars in instantiate_local_binding.
                            // This enables let-poly for `double = (x) => x + x` (Num constraint).
                            if type_has_external_anchor(ty, vars_before) {
                                let mut ids = Vec::new();
                                self.collect_var_ids(ty, &mut ids);
                                for id in ids {
                                    constrained_vars.insert(id);
                                }
                            } else {
                                // Internal HasTrait: record for trait_bounds registration
                                // so instantiate_local_binding can copy the bound to fresh vars
                                let mut ids = Vec::new();
                                self.collect_var_ids(ty, &mut ids);
                                for id in &ids {
                                    internal_has_traits.push((*id, trait_name.clone()));
                                }
                            }
                        }
                        Constraint::HasField(ty, _, field_ty, _) => {
                            // HasField constrains against a specific field name (external)
                            let mut ids = Vec::new();
                            self.collect_var_ids(ty, &mut ids);
                            self.collect_var_ids(field_ty, &mut ids);
                            for id in ids {
                                constrained_vars.insert(id);
                            }
                        }
                    }
                }
                // Also check pending method calls added during lambda inference.
                // If a PMC receiver is purely internal (all vars >= vars_before), we can
                // generalize it - the PMC will be duplicated with fresh vars in
                // instantiate_local_binding. If it involves external vars, it constrains
                // those vars and prevents generalization.
                let mut internal_pmcs: Vec<usize> = Vec::new(); // indices into pending_method_calls
                for (pmc_idx, pmc) in self.pending_method_calls[pending_methods_before..].iter().enumerate() {
                    let mut ids = Vec::new();
                    self.collect_var_ids(&pmc.receiver_ty, &mut ids);
                    // Also collect IDs from arg types and return type
                    for arg in &pmc.arg_types {
                        self.collect_var_ids(arg, &mut ids);
                    }
                    self.collect_var_ids(&pmc.ret_ty, &mut ids);
                    let is_fully_internal = ids.iter().all(|id| *id >= vars_before);
                    if is_fully_internal {
                        // This PMC is fully internal - can be generalized
                        internal_pmcs.push(pending_methods_before + pmc_idx);
                    } else {
                        // External anchor - prevents generalization of involved vars
                        let mut recv_ids = Vec::new();
                        self.collect_var_ids(&pmc.receiver_ty, &mut recv_ids);
                        for id in recv_ids {
                            constrained_vars.insert(id);
                        }
                    }
                }

                // Transitive propagation: if an internal Equal constraint connects
                // a constrained var to an unconstrained one, the unconstrained one
                // also becomes constrained. E.g., if ?N+1 is constrained (has .length()
                // method call) and Equal(?N, ?N+1) exists, then ?N is also constrained.
                // Repeat until fixed point.
                let mut changed = true;
                while changed {
                    changed = false;
                    for (a_ids, b_ids) in &internal_equals {
                        let a_has = a_ids.iter().any(|id| constrained_vars.contains(id));
                        let b_has = b_ids.iter().any(|id| constrained_vars.contains(id));
                        if a_has || b_has {
                            for id in a_ids.iter().chain(b_ids.iter()) {
                                if constrained_vars.insert(*id) {
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                for var_id in var_ids {
                    if !constrained_vars.contains(&var_id) {
                        self.polymorphic_vars.insert(var_id);
                    }
                }

                // Also mark internal vars from PMCs and body constraints as polymorphic
                // if they're not externally constrained. These "hidden" vars (not in the
                // function type signature) must be freshened on each instantiation.
                // E.g., for `transform = (xs) => xs.map(x => [x])`, the inner lambda's
                // param ?V2 and list element ?V3 are not in transform's type signature
                // but must be freshened for each call to transform.
                let mut extra_poly_vars: Vec<u32> = Vec::new();
                for pmc_idx in &internal_pmcs {
                    let pmc = &self.pending_method_calls[*pmc_idx];
                    let mut ids = Vec::new();
                    self.collect_var_ids(&pmc.receiver_ty, &mut ids);
                    for arg in &pmc.arg_types {
                        self.collect_var_ids(arg, &mut ids);
                    }
                    self.collect_var_ids(&pmc.ret_ty, &mut ids);
                    for id in ids {
                        if id >= vars_before && !constrained_vars.contains(&id) {
                            extra_poly_vars.push(id);
                        }
                    }
                }
                // Also check internal Equal constraints for hidden vars
                for constraint in &self.constraints[constraints_before..] {
                    if let Constraint::Equal(a, b, _) = constraint {
                        let a_ext = type_has_external_anchor(a, vars_before);
                        let b_ext = type_has_external_anchor(b, vars_before);
                        if !a_ext && !b_ext {
                            let mut ids = Vec::new();
                            self.collect_var_ids(a, &mut ids);
                            self.collect_var_ids(b, &mut ids);
                            for id in ids {
                                if id >= vars_before && !constrained_vars.contains(&id) {
                                    extra_poly_vars.push(id);
                                }
                            }
                        }
                    }
                }
                for id in extra_poly_vars {
                    self.polymorphic_vars.insert(id);
                }

                // Register internal HasTrait constraints as trait_bounds for
                // generalized vars, so instantiate_local_binding can copy them
                // to fresh vars. This ensures `double = (x) => x + x` correctly
                // propagates the Num bound when freshened, catching
                // `double("hello")` at compile time.
                for (var_id, trait_name) in &internal_has_traits {
                    if self.polymorphic_vars.contains(var_id) {
                        self.add_trait_bound(*var_id, trait_name.clone());
                    }
                }

                // Record internal Equal constraints that involve ONLY polymorphic vars.
                // When instantiate_local_binding freshens these vars, it replicates
                // these constraints with fresh copies. Without this,
                // `addUp = (a,b) => a + b; addUp(1, "hello")` would not catch
                // the Int/String mismatch because fresh_a and fresh_b would be
                // unconstrained relative to each other.
                // Only record constraints where ALL var IDs in BOTH sides are polymorphic
                // (to avoid mixing in constraints involving outer scope vars).
                for constraint in &self.constraints[constraints_before..] {
                    if let Constraint::Equal(a, b, _) = constraint {
                        let mut a_ids = Vec::new();
                        let mut b_ids = Vec::new();
                        self.collect_var_ids(a, &mut a_ids);
                        self.collect_var_ids(b, &mut b_ids);
                        let all_poly = a_ids.iter().chain(b_ids.iter())
                            .all(|id| self.polymorphic_vars.contains(id));
                        if all_poly && (!a_ids.is_empty() || !b_ids.is_empty()) {
                            self.polymorphic_body_constraints.push((a.clone(), b.clone()));
                        }
                    }
                }

                // Record internal pending method calls as polymorphic_body_method_calls
                // so that instantiate_local_binding can duplicate them with fresh vars.
                // This enables `first = (xs) => xs.head()` to be used at multiple types.
                // Record PMCs where ANY involved var is polymorphic. Non-polymorphic vars
                // stay fixed during duplication (apply_var_subst only replaces vars in
                // var_subst). This handles cases like `addOne = (xs) => xs.map(x => x+1)`
                // where the callback's param is fixed to Int but the receiver is polymorphic.
                for pmc_idx in &internal_pmcs {
                    let pmc = &self.pending_method_calls[*pmc_idx];
                    let mut ids = Vec::new();
                    self.collect_var_ids(&pmc.receiver_ty, &mut ids);
                    for arg in &pmc.arg_types {
                        self.collect_var_ids(arg, &mut ids);
                    }
                    self.collect_var_ids(&pmc.ret_ty, &mut ids);
                    let any_poly = ids.iter().any(|id| self.polymorphic_vars.contains(id));
                    if any_poly {
                        self.polymorphic_body_method_calls.push(pmc.clone());
                    }
                }

                // Record internal deferred overload calls as polymorphic_body_deferred_overloads
                // so that instantiate_local_binding can duplicate them with fresh vars.
                // This enables `showAny = (x) => show(x)` to be used at multiple types.
                // Without this, the deferred overload resolution during solve() locks the
                // original vars to a concrete type, preventing polymorphic use.
                // Also remove the originals from deferred_overload_calls to prevent
                // solve() from resolving the polymorphic vars to a concrete type.
                let mut indices_to_remove = Vec::new();
                for (idx, doc) in self.deferred_overload_calls[deferred_overloads_before..].iter().enumerate() {
                    let (_, arg_types, ret_ty, _) = doc;
                    let mut ids = Vec::new();
                    for arg in arg_types {
                        self.collect_var_ids(arg, &mut ids);
                    }
                    self.collect_var_ids(ret_ty, &mut ids);
                    let any_poly = ids.iter().any(|id| self.polymorphic_vars.contains(id));
                    if any_poly {
                        self.polymorphic_body_deferred_overloads.push(doc.clone());
                        indices_to_remove.push(deferred_overloads_before + idx);
                    }
                }
                // Remove in reverse order to maintain indices
                for idx in indices_to_remove.into_iter().rev() {
                    self.deferred_overload_calls.remove(idx);
                }
            }
        }

        // If there's a type annotation, unify with it.
        // Order: annotated_ty first, value_ty second so that unification failures report
        // "expected <annotation>, found <actual>" in the correct direction.
        if let Some(ty_expr) = &binding.ty {
            let annotated_ty = self.type_from_ast(ty_expr);
            self.unify(annotated_ty.clone(), value_ty.clone());
            // Also record for deferred check (batch unify errors are dropped)
            self.deferred_typed_binding_checks.push((
                value_ty.clone(),
                annotated_ty,
                binding.span,
            ));
        }

        // Bind pattern
        self.infer_pattern(&binding.pattern, &value_ty)?;

        // If this is a mutable binding (var x = ...), re-register as mutable
        // so that subsequent reassignments can be type-checked against the original type.
        if binding.mutable {
            if let Pattern::Var(ident) = &binding.pattern {
                let ty = if let Some(ty_expr) = &binding.ty {
                    // Use the annotated type for mutable bindings
                    self.type_from_ast(ty_expr)
                } else {
                    value_ty.clone()
                };
                self.env.bind(ident.node.clone(), ty, true);
            }
        }

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
        // Collect declared param types for overload matching.
        // This prevents cross-contamination in batch inference when multiple overloads
        // (e.g., add(Int,Int) and add(String,String)) share the same bare name.
        let declared_param_types: Vec<Option<Type>> = func.clauses.first()
            .map(|c| c.params.iter().map(|p| {
                p.ty.as_ref().map(|ty_expr| self.type_from_ast(ty_expr))
            }).collect())
            .unwrap_or_default();

        // Helper: check if a function type's params are compatible with declared param types
        let check_param_compat = |ft: &FunctionType| -> bool {
            if ft.params.len() != declared_param_types.len() { return false; }
            for (op, dp) in ft.params.iter().zip(declared_param_types.iter()) {
                if let Some(declared) = dp {
                    // Both are concrete types - must match
                    if !matches!(op, Type::Var(_)) && op != declared {
                        return false;
                    }
                }
            }
            true
        };

        // Build typed qualified name for overload lookup (e.g., "inc/String" for inc(x: String)).
        // This is needed when multiple overloads exist (e.g., inc/_  and inc/String).
        // The wildcard qualified_name ("inc/_") would find the wrong overload for typed functions.
        let typed_qualified_name: Option<String> = {
            let any_typed = declared_param_types.iter().any(|dp| dp.is_some());
            if any_typed {
                let parts: Vec<String> = declared_param_types.iter().map(|dp| {
                    match dp {
                        Some(ty) => ty.display(),
                        None => "_".to_string(),
                    }
                }).collect();
                Some(format!("{}/{}", name, parts.join(",")))
            } else {
                None
            }
        };

        #[allow(clippy::redundant_closure)]
        let pre_registered = if let Some(ref tqn) = typed_qualified_name {
            // Try typed qualified name first (e.g., "inc/String") to avoid cross-contamination
            // with other overloads (e.g., "inc/_" which may have Num constraints).
            self.env.functions.get(tqn.as_str()).cloned()
                .filter(|ft| check_param_compat(ft))
                .or_else(|| self.env.functions.get(&qualified_name).cloned()
                    .filter(|ft| check_param_compat(ft)))
                .or_else(|| self.env.functions.get(name).cloned()
                    .filter(|ft| check_param_compat(ft)))
        } else {
            self.env.functions.get(name).cloned()
                .filter(|ft| check_param_compat(ft))
                .or_else(|| self.env.functions.get(&qualified_name).cloned()
                    .filter(|ft| check_param_compat(ft)))
        }
            .or_else(|| {
                // Check typed overloads (e.g., "showCounter/Counter")
                // Clone to avoid borrow conflicts with self
                let all_overloads: Vec<FunctionType> = self.env.lookup_all_functions_with_arity(name, arity)
                    .into_iter().cloned().collect();
                if all_overloads.len() <= 1 {
                    all_overloads.into_iter().next()
                } else {
                    // Multiple overloads: match by declared param types to find the
                    // correct pre-registered entry. This prevents cross-contamination
                    // between overloads (e.g., add(Int,Int) picking up add(String,String)'s
                    // return type var during batch inference).
                    let declared_param_types: Vec<Type> = func.clauses.first()
                        .map(|c| c.params.iter().map(|p| {
                            if let Some(ty_expr) = &p.ty {
                                self.type_from_ast(ty_expr)
                            } else {
                                Type::Var(u32::MAX) // sentinel for untyped
                            }
                        }).collect())
                        .unwrap_or_default();

                    // Find the overload whose concrete param types match
                    let mut best: Option<FunctionType> = None;
                    for overload in &all_overloads {
                        if overload.params.len() != declared_param_types.len() {
                            continue;
                        }
                        let mut matches = true;
                        for (op, dp) in overload.params.iter().zip(declared_param_types.iter()) {
                            // Skip comparison for untyped params or type vars
                            if matches!(dp, Type::Var(id) if *id == u32::MAX) { continue; }
                            if matches!(op, Type::Var(_)) { continue; }
                            if op != dp {
                                matches = false;
                                break;
                            }
                        }
                        if matches {
                            best = Some(overload.clone());
                            break;
                        }
                    }
                    best.or_else(|| all_overloads.into_iter().next())
                }
            });

        // For functions with explicit type parameters (e.g., myMap[A, B]),
        // replace TypeParam types in the pre-registered signature with fresh Vars
        // and store these in type_param_mappings. This prevents occurs check failures
        // in recursive generic multi-clause functions, where TypeParam mappings
        // shared across clauses would create cyclic unification constraints.
        if !func.type_params.is_empty() && func.clauses.len() > 1 {
            if let Some(ref pre_reg) = pre_registered {
                let func_ty = Type::Function(pre_reg.clone());
                if func_ty.has_type_params() {
                    let mut tp_names = Vec::new();
                    Self::collect_type_param_names(&func_ty, &mut tp_names);
                    let mut tp_subst: HashMap<String, Type> = HashMap::new();
                    for tp_name in &tp_names {
                        let fresh = self.fresh();
                        // Apply trait constraints to the fresh var
                        if let Type::Var(var_id) = fresh {
                            if let Some(constraints) = self.current_type_param_constraints.get(tp_name) {
                                for constraint in constraints.clone() {
                                    self.add_trait_bound(var_id, constraint);
                                }
                            }
                        }
                        tp_subst.insert(tp_name.clone(), fresh);
                    }
                    // Replace TypeParam types with fresh vars in the pre-registered signature
                    let substituted = Self::substitute_type_params(&func_ty, &tp_subst);
                    if let Type::Function(sub_sig) = substituted {
                        // Update the environment entry so recursive calls see Var types
                        self.env.insert_function(name.clone(), sub_sig);
                        if self.env.functions.contains_key(&qualified_name) {
                            if let Some(updated) = self.env.functions.get(name).cloned() {
                                self.env.insert_function(qualified_name.clone(), updated);
                            }
                        }
                    }
                    // Store the mappings so type_from_ast resolves TypeParam to the same vars
                    for (tp_name, fresh_var) in &tp_subst {
                        self.type_param_mappings.insert(tp_name.clone(), fresh_var.clone());
                    }
                }
            }
        }

        // Infer the first clause
        let (clause_params, clause_ret) = self.infer_clause(&func.clauses[0])?;
        // If we have a pre-registered type, unify with its vars to connect recursive calls
        let (mut param_types, mut ret_ty, discovered_var_bounds) = if let Some(ref pre_reg) = pre_registered {
            // Unify clause params with pre-registered params
            if clause_params.len() == pre_reg.params.len() {
                for (cp, pp) in clause_params.iter().zip(pre_reg.params.iter()) {
                    self.unify(cp.clone(), pp.clone());
                }
            }
            // Unify clause return with pre-registered return
            self.unify(clause_ret.clone(), (*pre_reg.ret).clone());
            // Store direct mapping from pre-reg ret var to clause ret type.
            // This serves as a fallback for enrichment when solve() exits early
            // due to errors in OTHER functions, leaving this constraint unprocessed.
            if let Type::Var(ret_var_id) = &*pre_reg.ret {
                self.clause_ret_types.insert(*ret_var_id, clause_ret.clone());
            }

            // Store direct mapping from pre-reg param Var IDs to clause param types.
            // This serves as a fallback for enrichment when solve() exits early.
            for (pre_param, clause_param) in pre_reg.params.iter().zip(clause_params.iter()) {
                if let Type::Var(param_var_id) = pre_param {
                    self.clause_param_types.insert(*param_var_id, clause_param.clone());
                }
            }

            // Propagate trait bounds to the function's param/ret vars.
            // During batch inference, trait bounds from called functions (e.g., a(x)=x+1
            // has Num on a's param) need to reach the caller's param vars (e.g., b(x)=a(x)).
            // Build equivalence classes from Equal(Var,Var) constraints and collect
            // HasTrait bounds, then push HasTrait constraints on clause param/ret vars
            // so instantiate_function can find them when another function calls this one.
            fn collect_vars_for_bounds(ty: &Type, ids: &mut Vec<u32>) {
                match ty {
                    Type::Var(id) => { if !ids.contains(id) { ids.push(*id); } }
                    Type::List(inner) | Type::Array(inner) | Type::Set(inner) | Type::IO(inner) => {
                        collect_vars_for_bounds(inner, ids);
                    }
                    Type::Map(k, v) => { collect_vars_for_bounds(k, ids); collect_vars_for_bounds(v, ids); }
                    Type::Tuple(elems) => { for e in elems { collect_vars_for_bounds(e, ids); } }
                    Type::Function(ft) => {
                        for p in &ft.params { collect_vars_for_bounds(p, ids); }
                        collect_vars_for_bounds(&ft.ret, ids);
                    }
                    Type::Named { args, .. } => { for a in args { collect_vars_for_bounds(a, ids); } }
                    _ => {}
                }
            }
            if !clause_params.is_empty() {
                // Collect var IDs from clause params and ret
                let mut fn_var_ids: Vec<u32> = Vec::new();
                for p in &clause_params { collect_vars_for_bounds(p, &mut fn_var_ids); }
                collect_vars_for_bounds(&clause_ret, &mut fn_var_ids);
                // Also collect from pre-registered type vars so bounds are pushed onto
                // the vars stored in the env (which instantiate_function will see).
                if let Some(ref pre_reg) = pre_registered {
                    for p in &pre_reg.params { collect_vars_for_bounds(p, &mut fn_var_ids); }
                    collect_vars_for_bounds(&pre_reg.ret, &mut fn_var_ids);
                }

                if !fn_var_ids.is_empty() {
                    // Build union-find from Equal constraints, decomposing structured types.
                    // Equal(Function(p1->r1), Function(p2->r2)) implies Equal(p1,p2) and Equal(r1,r2).
                    // This is needed to track trait bounds through function call chains:
                    // b(x)=a(x) creates Equal(Function([?B],?RB), Function([?A],?RA))
                    // which implies Equal(?B, ?A) and Equal(?RB, ?RA).
                    let mut parent: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
                    fn uf_find2(parent: &mut std::collections::HashMap<u32, u32>, x: u32) -> u32 {
                        let p = *parent.get(&x).unwrap_or(&x);
                        if p == x { return x; }
                        let root = uf_find2(parent, p);
                        parent.insert(x, root);
                        root
                    }
                    fn uf_union(parent: &mut std::collections::HashMap<u32, u32>, a: u32, b: u32) {
                        let ra = uf_find2(parent, a);
                        let rb = uf_find2(parent, b);
                        if ra != rb { parent.insert(ra, rb); }
                    }
                    fn decompose_equal(t1: &Type, t2: &Type, parent: &mut std::collections::HashMap<u32, u32>) {
                        match (t1, t2) {
                            (Type::Var(a), Type::Var(b)) => {
                                uf_union(parent, *a, *b);
                            }
                            (Type::Function(f1), Type::Function(f2)) => {
                                for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                                    decompose_equal(p1, p2, parent);
                                }
                                decompose_equal(&f1.ret, &f2.ret, parent);
                            }
                            (Type::List(a), Type::List(b)) | (Type::Set(a), Type::Set(b)) | (Type::IO(a), Type::IO(b)) => {
                                decompose_equal(a, b, parent);
                            }
                            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                                decompose_equal(k1, k2, parent);
                                decompose_equal(v1, v2, parent);
                            }
                            (Type::Tuple(e1), Type::Tuple(e2)) if e1.len() == e2.len() => {
                                for (a, b) in e1.iter().zip(e2.iter()) {
                                    decompose_equal(a, b, parent);
                                }
                            }
                            (Type::Named { args: a1, .. }, Type::Named { args: a2, .. }) if a1.len() == a2.len() => {
                                for (a, b) in a1.iter().zip(a2.iter()) {
                                    decompose_equal(a, b, parent);
                                }
                            }
                            _ => {}
                        }
                    }
                    for constraint in &self.constraints {
                        if let Constraint::Equal(ref a, ref b, _) = constraint {
                            decompose_equal(a, b, &mut parent);
                        }
                    }

                    // Collect HasTrait bounds grouped by equivalence class root
                    let mut root_bounds: std::collections::HashMap<u32, Vec<String>> = std::collections::HashMap::new();
                    for constraint in &self.constraints {
                        if let Constraint::HasTrait(Type::Var(var_id), trait_name, _span) = constraint {
                            let root = uf_find2(&mut parent, *var_id);
                            let bounds = root_bounds.entry(root).or_default();
                            if !bounds.contains(trait_name) {
                                bounds.push(trait_name.clone());
                            }
                        }
                    }
                    for (&var_id, bounds) in &self.trait_bounds {
                        let root = uf_find2(&mut parent, var_id);
                        let root_b = root_bounds.entry(root).or_default();
                        for b in bounds {
                            if !root_b.contains(b) {
                                root_b.push(b.clone());
                            }
                        }
                    }

                    // For each fn var, push HasTrait constraints from its equivalence class
                    for &var_id in &fn_var_ids {
                        let root = uf_find2(&mut parent, var_id);
                        if let Some(bounds) = root_bounds.get(&root) {
                            for bound in bounds {
                                if bound.starts_with("HasMethod(") || bound.starts_with("HasField(") {
                                    continue;
                                }
                                // Only push if this exact constraint isn't already queued
                                let already_exists = self.constraints.iter().any(|c| {
                                    matches!(c, Constraint::HasTrait(Type::Var(vid), tn, _) if *vid == var_id && tn == bound)
                                });
                                if !already_exists {
                                    self.constraints.push(Constraint::HasTrait(
                                        Type::Var(var_id),
                                        bound.clone(),
                                        None,
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            // Collect discovered bounds for pre-registered vars to store on FunctionType.
            // This enables instantiate_function to propagate bounds through call chains.
            let mut pre_reg_var_ids: Vec<u32> = Vec::new();
            for p in &pre_reg.params { collect_vars_for_bounds(p, &mut pre_reg_var_ids); }
            collect_vars_for_bounds(&pre_reg.ret, &mut pre_reg_var_ids);

            let mut discovered_bounds: Vec<(u32, String)> = Vec::new();
            for &var_id in &pre_reg_var_ids {
                if let Some(bounds) = self.trait_bounds.get(&var_id) {
                    for bound in bounds {
                        if !bound.starts_with("HasMethod(") && !bound.starts_with("HasField(") {
                            discovered_bounds.push((var_id, bound.clone()));
                        }
                    }
                }
                // Also check constraints pushed by the union-find propagation above
                for constraint in &self.constraints {
                    if let Constraint::HasTrait(Type::Var(vid), tn, _) = constraint {
                        if *vid == var_id && !tn.starts_with("HasMethod(") && !tn.starts_with("HasField(")
                            && !discovered_bounds.iter().any(|(id, b)| *id == var_id && b == tn)
                        {
                            discovered_bounds.push((var_id, tn.clone()));
                        }
                    }
                }
            }

            // Use pre-registered types (they're now unified)
            (pre_reg.params.clone(), (*pre_reg.ret).clone(), discovered_bounds)
        } else {
            (clause_params, clause_ret, vec![])
        };

        // Pre-scan: detect param positions where clauses have structurally incompatible patterns.
        // This covers tuple patterns that differ in arity or nested structure across clauses.
        // For such positions, skip cross-clause type unification and use a fresh type variable.
        let arity = func.clauses[0].params.len();
        let mut conflicting_tuple_arity_positions: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Helper: compute a "structural signature" of a pattern for comparison.
        // Returns None for non-structural patterns (vars, wildcards, literals),
        // or Some(string) describing the tuple structure.
        fn pattern_structure(p: &Pattern) -> Option<String> {
            match p {
                Pattern::Tuple(elems, _) => {
                    let inner: Vec<String> = elems.iter().map(|e| {
                        pattern_structure(e).unwrap_or_else(|| "_".to_string())
                    }).collect();
                    Some(format!("({})", inner.join(",")))
                }
                _ => None,
            }
        }

        for i in 0..arity {
            let structures: Vec<Option<String>> = func.clauses.iter()
                .filter_map(|c| c.params.get(i).map(|p| pattern_structure(&p.pattern)))
                .collect();
            // If any two clauses have different structural signatures, mark as conflicting
            let first = &structures[0];
            for s in structures.iter().skip(1) {
                if s != first {
                    conflicting_tuple_arity_positions.insert(i);
                    break;
                }
            }
        }

        // Pre-compute: for each param position, does this FnDef have a mix of
        // typed AND untyped clauses? If so, it contains multiple typed overloads
        // grouped together, and we must not unify params across overload boundaries.
        // Example: inc(x) = x+1 grouped with inc(x: String) = x++"!" - these are
        // separate dispatches; their param types must not be unified.
        let mut mixed_type_annotation_positions: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for i in 0..arity {
            let has_typed = func.clauses.iter().any(|c| c.params.get(i).map_or(false, |p| p.ty.is_some()));
            let has_untyped = func.clauses.iter().any(|c| c.params.get(i).map_or(false, |p| p.ty.is_none()));
            if has_typed && has_untyped {
                mixed_type_annotation_positions.insert(i);
            }
        }

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

            // Determine if this clause is part of a different typed dispatch branch.
            // A clause is a separate typed overload if ALL params at mixed positions have
            // explicit type annotations that DIFFER from the accumulated type.
            // This prevents cross-contamination like: inc(x:String)=x++"!" acquiring Num bounds
            // from inc(x)=x+1 when both end up in the same FnDef due to grouping.
            let is_separate_typed_overload = if mixed_type_annotation_positions.is_empty() {
                false
            } else {
                // Check if this clause's explicit type annotations at mixed positions
                // differ from accumulated param types.
                let mut has_concrete_type_mismatch = false;
                for &i in &mixed_type_annotation_positions {
                    let accumulated = self.env.apply_subst(&param_types[i]);
                    let accumulated_is_concrete = !matches!(&accumulated, Type::Var(_));
                    if accumulated_is_concrete {
                        if let Some(p) = clause.params.get(i) {
                            if p.ty.is_none() {
                                // This clause has NO type annotation at a mixed position where
                                // the accumulated type is already concrete: it's the untyped branch.
                                has_concrete_type_mismatch = true;
                                break;
                            }
                            // Has explicit type - check if it matches
                            let explicit_ty = self.type_from_ast(p.ty.as_ref().unwrap());
                            if accumulated != explicit_ty {
                                has_concrete_type_mismatch = true;
                                break;
                            }
                        }
                    }
                }
                has_concrete_type_mismatch
            };

            if is_separate_typed_overload {
                // This clause belongs to a different typed dispatch branch - skip param unification
                // to prevent cross-contamination between distinct overloads.
                // Return types are also skipped (different overloads return independently).
                continue;
            }

            for i in 0..param_types.len() {
                // If this param position has conflicting tuple structures across clauses,
                // do NOT unify them. Different-structure tuple dispatch is valid at runtime
                // (mutually exclusive patterns), but unifying would force conflicting sizes.
                if !conflicting_tuple_arity_positions.contains(&i) {
                    self.unify(param_types[i].clone(), clause_params[i].clone());
                }
            }
            // Update param_types with unified types
            for param_type in &mut param_types {
                *param_type = self.env.apply_subst(param_type);
            }

            // Unify return types
            self.unify(ret_ty.clone(), clause_ret);
            ret_ty = self.env.apply_subst(&ret_ty);
        }

        // For param positions with conflicting tuple arities, replace the registered param type
        // with a fresh unconstrained type variable. This allows the caller to pass any tuple
        // (the runtime dispatch picks the right clause). Without this, the signature would
        // be locked to one specific tuple arity and calls with other arities would fail type check.
        for i in &conflicting_tuple_arity_positions {
            if *i < param_types.len() {
                param_types[*i] = self.fresh();
            }
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
            var_bounds: discovered_var_bounds,
        };

        // Register function in environment (bare name)
        self.env.insert_function(name.clone(), func_ty.clone());

        // Also update the arity-qualified entry (e.g., "a/_") so that
        // lookup_all_functions_with_arity finds the version with var_bounds.
        let arity = func_ty.params.len();
        let arity_suffix = if arity == 0 {
            "/".to_string()
        } else {
            format!("/{}", vec!["_"; arity].join(","))
        };
        let qualified_name = format!("{}{}", name, arity_suffix);
        if self.env.functions.contains_key(&qualified_name) {
            self.env.insert_function(qualified_name, func_ty.clone());
        }

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

        // Set the declared return type hint before inferring the body.
        // This allows lookup_constructor to prefer constructors from the declared return type
        // when multiple types have constructors with the same name.
        // E.g., if both IntTree and StrTree have `Node`, and the function returns StrTree,
        // `Node(v, StrLeaf, StrLeaf)` in the body should resolve to StrTree.Node.
        let saved_declared_return = self.current_declared_return_type.take();
        if let Some(ret_expr) = &clause.return_type {
            // Extract the base type name from the return type annotation.
            // For TypeExpr::Name or TypeExpr::Generic, get the outermost type name.
            let ret_type_name: Option<String> = match ret_expr {
                TypeExpr::Name(ident) => Some(ident.node.clone()),
                TypeExpr::Generic(ident, _) => Some(ident.node.clone()),
                _ => None,
            };
            self.current_declared_return_type = ret_type_name;
        }

        // Push a fresh type variable for this function's return type.
        // `return expr` inside the body will unify with this var, so early
        // returns contribute to the inferred return type rather than being lost.
        let fn_ret_var = self.fresh();
        self.fn_return_type_stack.push(fn_ret_var.clone());
        self.fn_has_return_stack.push(false);

        // Infer body
        let body_ty = self.infer_expr(&clause.body)?;

        // Pop the function return type var and the has-return flag.
        self.fn_return_type_stack.pop();
        let has_return = self.fn_has_return_stack.pop().unwrap_or(false);

        // Restore the declared return type after body inference.
        self.current_declared_return_type = saved_declared_return;

        // The actual return type is the join of:
        //  - body_ty: the type of the last expression (Unit when body ends with `return`)
        //  - fn_ret_var: unified with all `return expr` values inside the body
        //
        // If has_return is true and body_ty is Unit (body ends with a `return` statement),
        // use fn_ret_var as the return type so early returns are not lost.
        // If has_return is true and body_ty is NOT Unit, unify fn_ret_var with body_ty.
        // If has_return is false, use body_ty as usual (fn_ret_var is an unbound fresh var).
        let effective_body_ty = if has_return {
            match &body_ty {
                Type::Unit => fn_ret_var.clone(),
                _ => {
                    // Unify: any `return expr` values must match the fallthrough type
                    self.unify(fn_ret_var, body_ty.clone());
                    body_ty
                }
            }
        } else {
            body_ty
        };

        // If there's a return type annotation, unify with it.
        // Order: unify(annotated, effective_body_ty) so errors say "expected <annotated>, found <body_type>"
        // which is the natural reading: the annotation is what's expected, the body is what was found.
        let ret_ty = if let Some(ret_expr) = &clause.return_type {
            let annotated = self.type_from_ast(ret_expr);
            self.unify(annotated.clone(), effective_body_ty);
            annotated
        } else {
            effective_body_ty
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
                            var_bounds: vec![],
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
                var_bounds: vec![],
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
            var_bounds: vec![],
        });
        let func2 = Type::Function(FunctionType { required_params: None,
            type_params: vec![],
            params: vec![Type::Int],
            ret: Box::new(Type::Bool),
            var_bounds: vec![],
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
                var_bounds: vec![],
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
                var_bounds: vec![],
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
                var_bounds: vec![],
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
                    var_bounds: vec![],
                })),
                var_bounds: vec![],
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
                var_bounds: vec![],
            },
        );
        env.insert_function(
            "to_string".to_string(),
            FunctionType { required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::String),
                var_bounds: vec![],
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
                var_bounds: vec![],
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
                var_bounds: vec![],
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

