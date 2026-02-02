//! REPL session management with transactional semantics.
//!
//! The session is always in a valid, runnable state. Changes that would
//! introduce errors are rejected and the previous state is preserved.

use std::collections::{HashMap, HashSet};

use nostos_syntax::ast::{CallArg, Expr, FnDef, MatchArm, Pattern, Stmt, TypeDef};
use thiserror::Error;

use crate::CallGraph;

/// Errors that can occur in the REPL session.
#[derive(Debug, Clone, Error)]
pub enum ReplError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Type error in '{name}': {message}")]
    TypeError { name: String, message: String },

    #[error("Compilation error: {0}")]
    CompileError(String),

    #[error("Unknown function: {0}")]
    UnknownFunction(String),

    #[error("Change rejected: would break {count} function(s)")]
    WouldBreak {
        count: usize,
        errors: Vec<FunctionError>,
    },

    #[error("Cannot remove '{name}': would break {count} function(s)")]
    RemoveWouldBreak {
        name: String,
        count: usize,
        errors: Vec<FunctionError>,
    },
}

/// An error in a specific function.
#[derive(Debug, Clone)]
pub struct FunctionError {
    pub name: String,
    pub errors: Vec<String>,
}

/// A definition in the REPL session.
/// In transactional mode, all definitions are always valid.
#[derive(Debug, Clone)]
pub struct Definition {
    /// Original source code
    pub source: String,

    /// Parsed function definition
    pub fn_def: FnDef,

    /// Functions this definition calls (dependencies)
    pub dependencies: HashSet<String>,
}

impl Definition {
    /// Create a new definition from source and AST.
    pub fn new(source: String, fn_def: FnDef) -> Self {
        let dependencies = extract_dependencies_from_fn(&fn_def);
        Self {
            source,
            fn_def,
            dependencies,
        }
    }
}

/// Result of a successful define operation.
#[derive(Debug, Clone)]
pub struct DefineSuccess {
    /// The function that was defined
    pub name: String,

    /// Whether this was a redefinition
    pub was_redefinition: bool,

    /// Functions that were re-checked (for redefinitions)
    pub rechecked: Vec<String>,
}

/// Result of trying a define operation (without committing).
#[derive(Debug, Clone)]
pub struct TryDefineResult {
    /// The function name
    pub name: String,

    /// Whether this would be a redefinition
    pub would_be_redefinition: bool,

    /// Errors in the new definition itself
    pub self_errors: Vec<String>,

    /// Functions that would break and their errors
    pub would_break: Vec<FunctionError>,
}

impl TryDefineResult {
    /// Check if this define would succeed.
    pub fn is_ok(&self) -> bool {
        self.self_errors.is_empty() && self.would_break.is_empty()
    }

    /// Get all errors.
    pub fn all_errors(&self) -> Vec<FunctionError> {
        let mut errors = Vec::new();
        if !self.self_errors.is_empty() {
            errors.push(FunctionError {
                name: self.name.clone(),
                errors: self.self_errors.clone(),
            });
        }
        errors.extend(self.would_break.clone());
        errors
    }
}

/// The REPL session state.
///
/// This session is transactional: all definitions are always valid,
/// and changes that would introduce errors are rejected.
#[derive(Debug, Clone, Default)]
pub struct ReplSession {
    /// All definitions by name (always valid)
    definitions: HashMap<String, Definition>,

    /// The call graph tracking dependencies
    call_graph: CallGraph,
}

impl ReplSession {
    /// Create a new empty REPL session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Try a define operation without committing.
    ///
    /// Returns what would happen if this definition were applied.
    /// Use this to preview errors before committing.
    pub fn try_define(&self, _source: &str, fn_def: &FnDef) -> TryDefineResult {
        let name = fn_def.name.node.clone();
        let would_be_redefinition = self.definitions.contains_key(&name);
        let dependencies = extract_dependencies_from_fn(fn_def);

        // Check for errors in the new definition itself
        let mut self_errors = vec![];
        for dep in &dependencies {
            // Skip self-references (recursion is fine)
            if dep == &name {
                continue;
            }
            if !self.definitions.contains_key(dep) {
                self_errors.push(format!("Unknown function: {}", dep));
            }
        }

        // If the definition itself has errors, no need to check dependents
        if !self_errors.is_empty() {
            return TryDefineResult {
                name,
                would_be_redefinition,
                self_errors,
                would_break: vec![],
            };
        }

        // For redefinitions, check if any dependents would break
        // (In a full implementation, this would involve type checking)
        let would_break = vec![];

        TryDefineResult {
            name,
            would_be_redefinition,
            self_errors,
            would_break,
        }
    }

    /// Define a new function or redefine an existing one.
    ///
    /// This is transactional: if the definition would cause errors
    /// (in itself or in dependents), the change is rejected.
    ///
    /// Returns `Ok(DefineSuccess)` if the change was committed,
    /// or `Err(ReplError)` if it was rejected.
    pub fn define(&mut self, source: String, fn_def: FnDef) -> Result<DefineSuccess, ReplError> {
        let result = self.try_define(&source, &fn_def);

        if !result.is_ok() {
            let errors = result.all_errors();
            return Err(ReplError::WouldBreak {
                count: errors.len(),
                errors,
            });
        }

        // Commit the change
        let name = fn_def.name.node.clone();
        let was_redefinition = self.definitions.contains_key(&name);
        let dependencies = extract_dependencies_from_fn(&fn_def);

        // Update call graph
        self.call_graph.update(&name, dependencies);

        // Store definition
        let def = Definition::new(source, fn_def);
        self.definitions.insert(name.clone(), def);

        // Get rechecked functions (for reporting)
        let rechecked = if was_redefinition {
            let dependents = self.call_graph.transitive_dependents(&name);
            self.call_graph.topological_sort(&dependents)
        } else {
            vec![]
        };

        Ok(DefineSuccess {
            name,
            was_redefinition,
            rechecked,
        })
    }

    /// Force a definition even if it would cause errors.
    ///
    /// This bypasses the transactional check. Use with caution!
    /// Returns the errors that were introduced.
    pub fn force_define(&mut self, source: String, fn_def: FnDef) -> (DefineSuccess, Vec<FunctionError>) {
        let result = self.try_define(&source, &fn_def);
        let errors = result.all_errors();

        let name = fn_def.name.node.clone();
        let was_redefinition = self.definitions.contains_key(&name);
        let dependencies = extract_dependencies_from_fn(&fn_def);

        // Update call graph
        self.call_graph.update(&name, dependencies);

        // Store definition
        let def = Definition::new(source, fn_def);
        self.definitions.insert(name.clone(), def);

        let rechecked = if was_redefinition {
            let dependents = self.call_graph.transitive_dependents(&name);
            self.call_graph.topological_sort(&dependents)
        } else {
            vec![]
        };

        (
            DefineSuccess {
                name,
                was_redefinition,
                rechecked,
            },
            errors,
        )
    }

    /// Try removing a function without committing.
    ///
    /// Returns the list of functions that would break.
    pub fn try_remove(&self, name: &str) -> Vec<FunctionError> {
        if !self.definitions.contains_key(name) {
            return vec![];
        }

        let dependents = self.call_graph.transitive_dependents(name);
        let sorted = self.call_graph.topological_sort(&dependents);

        // All dependents would break since they depend on the removed function
        sorted
            .into_iter()
            .map(|dep_name| {
                // Find which dependencies would be missing
                let dep_deps = self.call_graph.direct_dependencies(&dep_name);
                let missing: Vec<String> = dep_deps
                    .iter()
                    .filter(|d| *d == name || dependents.contains(*d))
                    .map(|d| format!("Unknown function: {}", d))
                    .collect();

                FunctionError {
                    name: dep_name,
                    errors: if missing.is_empty() {
                        vec![format!("Depends on removed function: {}", name)]
                    } else {
                        missing
                    },
                }
            })
            .collect()
    }

    /// Remove a function from the session.
    ///
    /// This is transactional: if removing would break other functions,
    /// the change is rejected.
    pub fn remove(&mut self, name: &str) -> Result<(), ReplError> {
        let would_break = self.try_remove(name);

        if !would_break.is_empty() {
            return Err(ReplError::RemoveWouldBreak {
                name: name.to_string(),
                count: would_break.len(),
                errors: would_break,
            });
        }

        // Safe to remove
        self.definitions.remove(name);
        self.call_graph.remove(name);
        Ok(())
    }

    /// Force remove a function even if it would break dependents.
    ///
    /// Returns the functions that were broken.
    pub fn force_remove(&mut self, name: &str) -> Vec<FunctionError> {
        let would_break = self.try_remove(name);

        // Remove anyway
        self.definitions.remove(name);
        self.call_graph.remove(name);

        // Also remove the now-broken dependents
        for err in &would_break {
            self.definitions.remove(&err.name);
            self.call_graph.remove(&err.name);
        }

        would_break
    }

    /// Get all function names in the session.
    pub fn functions(&self) -> Vec<&str> {
        self.definitions.keys().map(|s| s.as_str()).collect()
    }

    /// Get the direct dependents of a function.
    pub fn dependents(&self, name: &str) -> HashSet<String> {
        self.call_graph.direct_dependents(name)
    }

    /// Get the direct dependencies of a function.
    pub fn dependencies(&self, name: &str) -> HashSet<String> {
        self.call_graph.direct_dependencies(name)
    }

    /// Check if a function is defined.
    pub fn is_defined(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    /// Get a definition by name.
    pub fn get(&self, name: &str) -> Option<&Definition> {
        self.definitions.get(name)
    }

    /// Get the call graph (for inspection).
    pub fn call_graph(&self) -> &CallGraph {
        &self.call_graph
    }
}

/// Extract function dependencies from a function definition.
/// This includes:
/// - Functions called in the body
/// - Functions referenced in guards
/// - Template functions used as decorators (@decorator)
pub fn extract_dependencies_from_fn(fn_def: &FnDef) -> HashSet<String> {
    let mut deps = HashSet::new();

    // Track decorator/template dependencies
    // When a function uses @decorator, it depends on the decorator template
    for decorator in &fn_def.decorators {
        deps.insert(decorator.name.node.clone());
        // Also track dependencies in decorator arguments
        for arg in &decorator.args {
            extract_dependencies_from_expr(arg, &mut deps, &HashSet::new());
        }
    }

    // Collect all parameter names - these are local variables, not modules
    let mut local_vars: HashSet<String> = HashSet::new();
    for clause in &fn_def.clauses {
        for param in &clause.params {
            extract_pattern_names(&param.pattern, &mut local_vars);
        }
    }
    for clause in &fn_def.clauses {
        extract_dependencies_from_expr(&clause.body, &mut deps, &local_vars);
        if let Some(guard) = &clause.guard {
            extract_dependencies_from_expr(guard, &mut deps, &local_vars);
        }
    }
    deps
}

/// Extract dependencies from a type definition.
/// This tracks template functions used as decorators on types (@derive, @withGetters, etc.)
pub fn extract_dependencies_from_type(type_def: &TypeDef) -> HashSet<String> {
    let mut deps = HashSet::new();

    // Track decorator/template dependencies
    // When a type uses @decorator, it depends on the decorator template
    for decorator in &type_def.decorators {
        deps.insert(decorator.name.node.clone());
        // Also track dependencies in decorator arguments
        for arg in &decorator.args {
            extract_dependencies_from_expr(arg, &mut deps, &HashSet::new());
        }
    }

    deps
}

/// Extract variable names from a pattern (for parameter and let bindings)
fn extract_pattern_names(pattern: &Pattern, names: &mut HashSet<String>) {
    use nostos_syntax::ast::{ListPattern, RecordPatternField, VariantPatternFields};
    match pattern {
        Pattern::Var(ident) => {
            names.insert(ident.node.clone());
        }
        Pattern::Tuple(patterns, _) => {
            for p in patterns {
                extract_pattern_names(p, names);
            }
        }
        Pattern::Record(fields, _) => {
            for field in fields {
                match field {
                    RecordPatternField::Punned(ident) => {
                        names.insert(ident.node.clone());
                    }
                    RecordPatternField::Named(_, p) => {
                        extract_pattern_names(p, names);
                    }
                    RecordPatternField::Rest(_) => {}
                }
            }
        }
        Pattern::Variant(_, fields, _) => {
            match fields {
                VariantPatternFields::Unit => {}
                VariantPatternFields::Positional(patterns) => {
                    for p in patterns {
                        extract_pattern_names(p, names);
                    }
                }
                VariantPatternFields::Named(named_fields) => {
                    for field in named_fields {
                        match field {
                            RecordPatternField::Punned(ident) => {
                                names.insert(ident.node.clone());
                            }
                            RecordPatternField::Named(_, p) => {
                                extract_pattern_names(p, names);
                            }
                            RecordPatternField::Rest(_) => {}
                        }
                    }
                }
            }
        }
        Pattern::List(list_pat, _) => {
            match list_pat {
                ListPattern::Empty => {}
                ListPattern::Cons(patterns, rest) => {
                    for p in patterns {
                        extract_pattern_names(p, names);
                    }
                    if let Some(rest_pat) = rest {
                        extract_pattern_names(rest_pat, names);
                    }
                }
            }
        }
        Pattern::Wildcard(_) | Pattern::Or(_, _) | Pattern::Pin(_, _) => {}
        // Literal patterns don't introduce names
        Pattern::Int(_, _) | Pattern::Int8(_, _) | Pattern::Int16(_, _) | Pattern::Int32(_, _) |
        Pattern::UInt8(_, _) | Pattern::UInt16(_, _) | Pattern::UInt32(_, _) | Pattern::UInt64(_, _) |
        Pattern::Float(_, _) | Pattern::Float32(_, _) | Pattern::Decimal(_, _) | Pattern::BigInt(_, _) |
        Pattern::String(_, _) | Pattern::Char(_, _) | Pattern::Bool(_, _) | Pattern::Unit(_) |
        Pattern::StringCons(_, _) | Pattern::Map(_, _) | Pattern::Set(_, _) | Pattern::Range(_, _, _, _) => {}
    }
}

/// Extract function references from an expression.
/// Try to extract a qualified name from an expression like `foo.bar.baz`
/// Returns None if the expression is not a simple qualified name chain
fn try_extract_qualified_name(expr: &Expr) -> Option<String> {
    try_extract_qualified_name_inner(expr, true)
}

/// Helper that tracks whether we're at the root level.
/// At root level, we check if the base looks like a module name vs local variable.
/// Single-letter lowercase identifiers (p, x, y, n, m, etc.) are likely local variables.
/// For method calls on local variables (p.describe), we return None.
fn try_extract_qualified_name_inner(expr: &Expr, is_root: bool) -> Option<String> {
    match expr {
        Expr::Var(ident) => {
            let name = &ident.node;
            if is_root {
                // At root level, check if this looks like a local variable name.
                // Single-letter lowercase identifiers are almost always local variables.
                // Module names are typically longer (good, List, helper, etc.)
                let is_likely_local_var = name.len() == 1
                    && name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false);

                if is_likely_local_var {
                    // Single-letter lowercase - likely a local variable, not a module
                    None
                } else {
                    Some(name.clone())
                }
            } else {
                Some(name.clone())
            }
        }
        Expr::FieldAccess(base, field, _) => {
            // Only capture qualified module.function, not method calls on values
            // When inside a FieldAccess, the base is NOT root level anymore -
            // we're building a qualified name, so single-letter names like `c`
            // in `c.getval` should be captured as module names.
            if let Some(base_name) = try_extract_qualified_name_inner(base, false) {
                Some(format!("{}.{}", base_name, field.node))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_dependencies_from_expr(expr: &Expr, deps: &mut HashSet<String>, local_vars: &HashSet<String>) {
    match expr {
        // NOTE: We intentionally do NOT capture Expr::Var references here.
        // Var could be a local variable (x, y, result, etc.) or a function name.
        // Function calls are captured via Expr::Call and Expr::MethodCall.
        // Capturing all Vars would incorrectly treat local bindings as dependencies.
        Expr::Var(_) => {
            // Don't capture plain variable references - they're usually locals
        }
        Expr::Call(callee, _type_args, args, _) => {
            // For calls, try to capture the full qualified name (e.g., good.multiply)
            // For simple function calls (Var), always capture the name - it's a function call,
            // not a local variable reference.
            match callee.as_ref() {
                Expr::Var(ident) => {
                    // Simple function call like f() or a() - always capture
                    deps.insert(ident.node.clone());
                }
                Expr::FieldAccess(base, _, _) => {
                    // Module.function call like c.getval() or lib.helper()
                    // Only capture if the base is NOT a known local variable
                    if let Expr::Var(base_ident) = base.as_ref() {
                        if !local_vars.contains(&base_ident.node) {
                            if let Some(qualified) = try_extract_qualified_name(callee) {
                                deps.insert(qualified);
                            }
                        }
                    } else {
                        // Complex base expression
                        if let Some(qualified) = try_extract_qualified_name(callee) {
                            deps.insert(qualified);
                        } else {
                            extract_dependencies_from_expr(callee, deps, local_vars);
                        }
                    }
                }
                _ => {
                    // Other complex callees - try qualified name extraction
                    if let Some(qualified) = try_extract_qualified_name(callee) {
                        deps.insert(qualified);
                    } else {
                        extract_dependencies_from_expr(callee, deps, local_vars);
                    }
                }
            }
            for arg in args {
                let expr = match arg {
                    CallArg::Positional(e) | CallArg::Named(_, e) => e,
                };
                // Special case: if the argument is a simple Var, it might be a function
                // being passed as a value (higher-order function). Capture it if it's
                // not a known local variable.
                if let Expr::Var(ident) = expr {
                    let name = &ident.node;
                    if !local_vars.contains(name) {
                        deps.insert(name.clone());
                    }
                } else {
                    extract_dependencies_from_expr(expr, deps, local_vars);
                }
            }
        }
        Expr::BinOp(left, _, right, _) => {
            extract_dependencies_from_expr(left, deps, local_vars);
            extract_dependencies_from_expr(right, deps, local_vars);
        }
        Expr::UnaryOp(_, operand, _) => {
            extract_dependencies_from_expr(operand, deps, local_vars);
        }
        Expr::If(cond, then_branch, else_branch, _) => {
            extract_dependencies_from_expr(cond, deps, local_vars);
            extract_dependencies_from_expr(then_branch, deps, local_vars);
            extract_dependencies_from_expr(else_branch, deps, local_vars);
        }
        Expr::Match(scrutinee, arms, _) => {
            extract_dependencies_from_expr(scrutinee, deps, local_vars);
            for arm in arms {
                extract_dependencies_from_match_arm(arm, deps, local_vars);
            }
        }
        Expr::Lambda(params, body, _) => {
            // Lambda parameters are local to the lambda body
            let mut lambda_locals = local_vars.clone();
            for param in params {
                extract_pattern_names(param, &mut lambda_locals);
            }
            extract_dependencies_from_expr(body, deps, &lambda_locals);
        }
        Expr::Block(stmts, _) => {
            let mut block_locals = local_vars.clone();
            for stmt in stmts {
                extract_dependencies_from_stmt(stmt, deps, &mut block_locals);
            }
        }
        Expr::Tuple(elems, _) => {
            for elem in elems {
                extract_dependencies_from_expr(elem, deps, local_vars);
            }
        }
        Expr::List(elems, tail, _) => {
            for elem in elems {
                extract_dependencies_from_expr(elem, deps, local_vars);
            }
            if let Some(tail) = tail {
                extract_dependencies_from_expr(tail, deps, local_vars);
            }
        }
        Expr::Record(_, fields, _) => {
            for field in fields {
                match field {
                    nostos_syntax::ast::RecordField::Positional(e) => {
                        extract_dependencies_from_expr(e, deps, local_vars);
                    }
                    nostos_syntax::ast::RecordField::Named(_, e) => {
                        extract_dependencies_from_expr(e, deps, local_vars);
                    }
                }
            }
        }
        Expr::RecordUpdate(_, base, fields, _) => {
            extract_dependencies_from_expr(base, deps, local_vars);
            for field in fields {
                match field {
                    nostos_syntax::ast::RecordField::Positional(e) => {
                        extract_dependencies_from_expr(e, deps, local_vars);
                    }
                    nostos_syntax::ast::RecordField::Named(_, e) => {
                        extract_dependencies_from_expr(e, deps, local_vars);
                    }
                }
            }
        }
        Expr::FieldAccess(base, _field, _) => {
            // Don't capture standalone field access as dependencies.
            // Record field access like `p.name` is NOT a function call.
            // Qualified function calls like `module.function(args)` are handled
            // by the Expr::Call handler which uses try_extract_qualified_name.
            extract_dependencies_from_expr(base, deps, local_vars);
        }
        Expr::Index(base, index, _) => {
            extract_dependencies_from_expr(base, deps, local_vars);
            extract_dependencies_from_expr(index, deps, local_vars);
        }
        Expr::MethodCall(receiver, method, args, _) => {
            // For method calls like `good.multiply()` or `c.getval()`, if receiver is a simple Var,
            // capture as dependency ONLY if receiver is NOT a known local variable.
            if let Expr::Var(receiver_name) = receiver.as_ref() {
                if !local_vars.contains(&receiver_name.node) {
                    let qualified = format!("{}.{}", receiver_name.node, method.node);
                    deps.insert(qualified);
                }
            } else {
                extract_dependencies_from_expr(receiver, deps, local_vars);
            }
            for arg in args {
                let expr = match arg {
                    CallArg::Positional(e) | CallArg::Named(_, e) => e,
                };
                // Special case for function references passed as arguments
                if let Expr::Var(ident) = expr {
                    let name = &ident.node;
                    let is_likely_local = name.len() == 1
                        && name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false);
                    if !is_likely_local {
                        deps.insert(name.clone());
                    }
                } else {
                    extract_dependencies_from_expr(expr, deps, local_vars);
                }
            }
        }
        Expr::Map(pairs, _) => {
            for (k, v) in pairs {
                extract_dependencies_from_expr(k, deps, local_vars);
                extract_dependencies_from_expr(v, deps, local_vars);
            }
        }
        Expr::Set(elems, _) => {
            for elem in elems {
                extract_dependencies_from_expr(elem, deps, local_vars);
            }
        }
        Expr::Try(body, arms, finally, _) => {
            extract_dependencies_from_expr(body, deps, local_vars);
            for arm in arms {
                extract_dependencies_from_match_arm(arm, deps, local_vars);
            }
            if let Some(finally) = finally {
                extract_dependencies_from_expr(finally, deps, local_vars);
            }
        }
        Expr::Try_(inner, _) => {
            extract_dependencies_from_expr(inner, deps, local_vars);
        }
        Expr::Quote(inner, _) => {
            extract_dependencies_from_expr(inner, deps, local_vars);
        }
        Expr::Splice(inner, _) => {
            extract_dependencies_from_expr(inner, deps, local_vars);
        }
        Expr::Do(stmts, _) => {
            let mut do_locals = local_vars.clone();
            for stmt in stmts {
                match stmt {
                    nostos_syntax::ast::DoStmt::Bind(pat, e) => {
                        extract_dependencies_from_expr(e, deps, &do_locals);
                        extract_pattern_names(pat, &mut do_locals);
                    }
                    nostos_syntax::ast::DoStmt::Expr(e) => {
                        extract_dependencies_from_expr(e, deps, &do_locals);
                    }
                }
            }
        }
        Expr::Receive(arms, timeout, _) => {
            for arm in arms {
                extract_dependencies_from_match_arm(arm, deps, local_vars);
            }
            if let Some((duration, body)) = timeout {
                extract_dependencies_from_expr(duration, deps, local_vars);
                extract_dependencies_from_expr(body, deps, local_vars);
            }
        }
        Expr::Spawn(_, callee, args, _) => {
            extract_dependencies_from_expr(callee, deps, local_vars);
            for arg in args {
                extract_dependencies_from_expr(arg, deps, local_vars);
            }
        }
        Expr::Send(target, msg, _) => {
            extract_dependencies_from_expr(target, deps, local_vars);
            extract_dependencies_from_expr(msg, deps, local_vars);
        }
        Expr::String(lit, _) => {
            if let nostos_syntax::ast::StringLit::Interpolated(parts) = lit {
                for part in parts {
                    if let nostos_syntax::ast::StringPart::Expr(e) = part {
                        extract_dependencies_from_expr(e, deps, local_vars);
                    }
                }
            }
        }
        // Loop expressions
        Expr::While(cond, body, _) => {
            extract_dependencies_from_expr(cond, deps, local_vars);
            extract_dependencies_from_expr(body, deps, local_vars);
        }
        Expr::For(var, start, end, body, _) => {
            extract_dependencies_from_expr(start, deps, local_vars);
            extract_dependencies_from_expr(end, deps, local_vars);
            // The loop variable is local to the body
            let mut for_locals = local_vars.clone();
            for_locals.insert(var.node.clone());
            extract_dependencies_from_expr(body, deps, &for_locals);
        }
        Expr::Break(value, _) => {
            if let Some(val) = value {
                extract_dependencies_from_expr(val, deps, local_vars);
            }
        }
        Expr::Continue(_) => {}
        Expr::Return(value, _) => {
            if let Some(val) = value {
                extract_dependencies_from_expr(val, deps, local_vars);
            }
        }
        // Literals don't have dependencies
        Expr::Int(_, _)
        | Expr::Int8(_, _)
        | Expr::Int16(_, _)
        | Expr::Int32(_, _)
        | Expr::UInt8(_, _)
        | Expr::UInt16(_, _)
        | Expr::UInt32(_, _)
        | Expr::UInt64(_, _)
        | Expr::BigInt(_, _)
        | Expr::Float(_, _)
        | Expr::Float32(_, _)
        | Expr::Decimal(_, _)
        | Expr::Bool(_, _)
        | Expr::Char(_, _)
        | Expr::Unit(_)
        | Expr::Wildcard(_) => {}
    }
}

/// Extract dependencies from a match arm.
fn extract_dependencies_from_match_arm(arm: &MatchArm, deps: &mut HashSet<String>, local_vars: &HashSet<String>) {
    // Pattern bindings in match arms are local to the arm
    let mut arm_locals = local_vars.clone();
    extract_pattern_names(&arm.pattern, &mut arm_locals);
    if let Some(guard) = &arm.guard {
        extract_dependencies_from_expr(guard, deps, &arm_locals);
    }
    extract_dependencies_from_expr(&arm.body, deps, &arm_locals);
}

/// Extract dependencies from a statement.
/// Also updates local_vars with any bindings introduced by the statement.
fn extract_dependencies_from_stmt(stmt: &Stmt, deps: &mut HashSet<String>, local_vars: &mut HashSet<String>) {
    match stmt {
        Stmt::Expr(e) => {
            extract_dependencies_from_expr(e, deps, local_vars);
        }
        Stmt::Let(binding) => {
            extract_dependencies_from_expr(&binding.value, deps, local_vars);
            // Add the binding pattern names to local_vars for subsequent statements
            extract_pattern_names(&binding.pattern, local_vars);
        }
        Stmt::Assign(_, e, _) => {
            extract_dependencies_from_expr(e, deps, local_vars);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nostos_syntax::ast::{FnClause, Span, Spanned, Visibility};

    /// Helper to create a simple function definition.
    fn make_fn(name: &str, body: Expr) -> FnDef {
        FnDef {
            visibility: Visibility::Public,
            doc: None,
            decorators: vec![],
            name: Spanned::new(name.to_string(), Span::default()),
            type_params: vec![],
            clauses: vec![FnClause {
                params: vec![],
                guard: None,
                return_type: None,
                body,
                span: Span::default(),
            }],
            is_template: false,
            span: Span::default(),
        }
    }

    /// Helper to create a function with parameters.
    fn make_fn_with_params(name: &str, param_names: &[&str], body: Expr) -> FnDef {
        use nostos_syntax::ast::FnParam;
        let params = param_names.iter().map(|p| {
            FnParam {
                pattern: Pattern::Var(Spanned::new(p.to_string(), Span::default())),
                ty: None,
                default: None,
            }
        }).collect();
        FnDef {
            visibility: Visibility::Public,
            doc: None,
            decorators: vec![],
            name: Spanned::new(name.to_string(), Span::default()),
            type_params: vec![],
            clauses: vec![FnClause {
                params,
                guard: None,
                return_type: None,
                body,
                span: Span::default(),
            }],
            is_template: false,
            span: Span::default(),
        }
    }

    /// Helper to create an integer literal expression.
    fn int_expr(n: i64) -> Expr {
        Expr::Int(n, Span::default())
    }

    /// Helper to create a variable reference expression.
    fn var_expr(name: &str) -> Expr {
        Expr::Var(Spanned::new(name.to_string(), Span::default()))
    }

    /// Helper to create a call expression.
    fn call_expr(name: &str, args: Vec<Expr>) -> Expr {
        use nostos_syntax::ast::CallArg;
        let call_args: Vec<CallArg> = args.into_iter().map(CallArg::Positional).collect();
        Expr::Call(Box::new(var_expr(name)), vec![], call_args, Span::default())
    }

    /// Helper to create a binary operation.
    fn binop_expr(left: Expr, op: nostos_syntax::ast::BinOp, right: Expr) -> Expr {
        Expr::BinOp(Box::new(left), op, Box::new(right), Span::default())
    }

    #[test]
    fn test_simple_definition() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo = make_fn("foo", int_expr(42));
        let result = session.define("foo() = 42".to_string(), foo);

        assert!(result.is_ok());
        let success = result.unwrap();
        assert_eq!(success.name, "foo");
        assert!(!success.was_redefinition);
        assert!(session.is_defined("foo"));
    }

    #[test]
    fn test_definition_with_dependency() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo = make_fn("foo", int_expr(42));
        session.define("foo() = 42".to_string(), foo).unwrap();

        // bar() = foo() + 1
        let bar_body = binop_expr(
            call_expr("foo", vec![]),
            nostos_syntax::ast::BinOp::Add,
            int_expr(1),
        );
        let bar = make_fn("bar", bar_body);
        let result = session.define("bar() = foo() + 1".to_string(), bar);

        assert!(result.is_ok());
        assert!(session.dependencies("bar").contains("foo"));
        assert!(session.dependents("foo").contains("bar"));
    }

    #[test]
    fn test_reject_unknown_dependency() {
        let mut session = ReplSession::new();

        // bar() = foo() - foo doesn't exist, should be rejected
        let bar = make_fn("bar", call_expr("foo", vec![]));
        let result = session.define("bar() = foo()".to_string(), bar);

        assert!(result.is_err());
        assert!(!session.is_defined("bar")); // Not committed
    }

    #[test]
    fn test_try_define_preview() {
        let session = ReplSession::new();

        // Try defining bar() = foo() without foo existing
        let bar = make_fn("bar", call_expr("foo", vec![]));
        let result = session.try_define("bar() = foo()", &bar);

        assert!(!result.is_ok());
        assert!(!result.self_errors.is_empty());
        assert!(result.self_errors[0].contains("foo"));
    }

    #[test]
    fn test_self_recursion_allowed() {
        let mut session = ReplSession::new();

        // fact() = fact() - self-recursion should be allowed
        let fact = make_fn("fact", call_expr("fact", vec![]));
        let result = session.define("fact() = fact()".to_string(), fact);

        assert!(result.is_ok());
        assert!(session.is_defined("fact"));
    }

    #[test]
    fn test_remove_rejects_if_dependents() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo = make_fn("foo", int_expr(42));
        session.define("foo() = 42".to_string(), foo).unwrap();

        // bar() = foo()
        let bar = make_fn("bar", call_expr("foo", vec![]));
        session.define("bar() = foo()".to_string(), bar).unwrap();

        // Try to remove foo - should fail because bar depends on it
        let result = session.remove("foo");

        assert!(result.is_err());
        assert!(session.is_defined("foo")); // Still there
        assert!(session.is_defined("bar")); // Still there
    }

    #[test]
    fn test_remove_leaf_function() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo = make_fn("foo", int_expr(42));
        session.define("foo() = 42".to_string(), foo).unwrap();

        // bar() = foo()
        let bar = make_fn("bar", call_expr("foo", vec![]));
        session.define("bar() = foo()".to_string(), bar).unwrap();

        // Remove bar - should succeed (no dependents)
        let result = session.remove("bar");

        assert!(result.is_ok());
        assert!(session.is_defined("foo"));
        assert!(!session.is_defined("bar"));
    }

    #[test]
    fn test_force_define_with_errors() {
        let mut session = ReplSession::new();

        // Force define bar() = foo() without foo
        let bar = make_fn("bar", call_expr("foo", vec![]));
        let (success, errors) = session.force_define("bar() = foo()".to_string(), bar);

        assert_eq!(success.name, "bar");
        assert!(!errors.is_empty());
        // Definition was committed despite errors
        assert!(session.is_defined("bar"));
    }

    #[test]
    fn test_force_remove_with_dependents() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo = make_fn("foo", int_expr(42));
        session.define("foo() = 42".to_string(), foo).unwrap();

        // bar() = foo()
        let bar = make_fn("bar", call_expr("foo", vec![]));
        session.define("bar() = foo()".to_string(), bar).unwrap();

        // Force remove foo - should remove bar too
        let broken = session.force_remove("foo");

        assert!(!session.is_defined("foo"));
        assert!(!session.is_defined("bar")); // Also removed
        assert!(broken.iter().any(|e| e.name == "bar"));
    }

    #[test]
    fn test_try_remove_preview() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo = make_fn("foo", int_expr(42));
        session.define("foo() = 42".to_string(), foo).unwrap();

        // bar() = foo()
        let bar = make_fn("bar", call_expr("foo", vec![]));
        session.define("bar() = foo()".to_string(), bar).unwrap();

        // Preview what would break
        let would_break = session.try_remove("foo");

        assert!(!would_break.is_empty());
        assert!(would_break.iter().any(|e| e.name == "bar"));
        // But nothing actually removed
        assert!(session.is_defined("foo"));
        assert!(session.is_defined("bar"));
    }

    #[test]
    fn test_redefinition_success() {
        let mut session = ReplSession::new();

        // foo() = 42
        let foo1 = make_fn("foo", int_expr(42));
        session.define("foo() = 42".to_string(), foo1).unwrap();

        // Redefine foo() = 100
        let foo2 = make_fn("foo", int_expr(100));
        let result = session.define("foo() = 100".to_string(), foo2);

        assert!(result.is_ok());
        let success = result.unwrap();
        assert!(success.was_redefinition);
    }

    #[test]
    fn test_chain_dependency_remove() {
        let mut session = ReplSession::new();

        // a() = 1
        session.define("a() = 1".to_string(), make_fn("a", int_expr(1))).unwrap();

        // b() = a()
        session.define("b() = a()".to_string(), make_fn("b", call_expr("a", vec![]))).unwrap();

        // c() = b()
        session.define("c() = b()".to_string(), make_fn("c", call_expr("b", vec![]))).unwrap();

        // Try to remove a - should fail (b and c depend on it transitively)
        let would_break = session.try_remove("a");
        assert_eq!(would_break.len(), 2); // b and c

        // Remove c first (leaf)
        assert!(session.remove("c").is_ok());

        // Remove b (now leaf)
        assert!(session.remove("b").is_ok());

        // Now can remove a
        assert!(session.remove("a").is_ok());
    }

    #[test]
    fn test_functions_list() {
        let mut session = ReplSession::new();

        session.define("foo() = 1".to_string(), make_fn("foo", int_expr(1))).unwrap();
        session.define("bar() = 2".to_string(), make_fn("bar", int_expr(2))).unwrap();

        let funcs = session.functions();
        assert_eq!(funcs.len(), 2);
        assert!(funcs.contains(&"foo"));
        assert!(funcs.contains(&"bar"));
    }

    #[test]
    fn test_extract_qualified_call_dependency() {
        use nostos_syntax::ast::CallArg;

        // Create AST for: main() = good.multiply(2, 3)
        // This is: Call(FieldAccess(Var("good"), "multiply"), [2, 3])
        let good_var = Expr::Var(Spanned::new("good".to_string(), Span::default()));
        let field_access = Expr::FieldAccess(
            Box::new(good_var),
            Spanned::new("multiply".to_string(), Span::default()),
            Span::default(),
        );
        let call = Expr::Call(
            Box::new(field_access),
            vec![],
            vec![CallArg::Positional(int_expr(2)), CallArg::Positional(int_expr(3))],
            Span::default(),
        );

        let main_fn = make_fn("main", call);
        let deps = extract_dependencies_from_fn(&main_fn);

        println!("Dependencies extracted: {:?}", deps);

        // Should contain "good.multiply", NOT just "good"
        assert!(deps.contains("good.multiply"),
            "Should extract 'good.multiply' as dependency, but got: {:?}", deps);
        assert!(!deps.contains("good"),
            "Should NOT have 'good' alone in deps (should be 'good.multiply'), got: {:?}", deps);
    }

    #[test]
    fn test_local_var_method_call_not_dependency() {
        // Create AST for: main(p) = p.describe()
        // This is a method call on parameter 'p', NOT a module call
        // It should NOT be extracted as a dependency
        let p_var = Expr::Var(Spanned::new("p".to_string(), Span::default()));
        let method_call = Expr::MethodCall(
            Box::new(p_var),
            Spanned::new("describe".to_string(), Span::default()),
            vec![],
            Span::default(),
        );

        // Create function with 'p' as a parameter
        let main_fn = make_fn_with_params("main", &["p"], method_call);
        let deps = extract_dependencies_from_fn(&main_fn);

        println!("Dependencies extracted for p.describe(): {:?}", deps);

        // Should NOT contain "p.describe" - p is a parameter, not a module
        assert!(!deps.contains("p.describe"),
            "Should NOT extract 'p.describe' as dependency (p is a parameter). Got: {:?}", deps);
        assert!(!deps.contains("p"),
            "Should NOT have 'p' in deps. Got: {:?}", deps);
    }

    #[test]
    fn test_use_statement_parsing() {
        // Test that use statements are parsed correctly
        let code = "use lib.helper\nmain() = helper(5)";
        let (parsed, errors) = nostos_syntax::parse(code);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let module = parsed.expect("Parse should succeed");

        println!("Module items: {:?}", module.items.len());
        for (i, item) in module.items.iter().enumerate() {
            println!("Item {}: {:?}", i, item);
        }

        // Check the use statement
        let use_stmt = match &module.items[0] {
            nostos_syntax::Item::Use(u) => u,
            other => panic!("Expected Use, got {:?}", other),
        };

        println!("Use stmt path: {:?}", use_stmt.path);
        println!("Use stmt imports: {:?}", use_stmt.imports);

        // Path should be ["lib"]
        assert_eq!(use_stmt.path.len(), 1, "Path should have 1 component");
        assert_eq!(use_stmt.path[0].node, "lib", "Path should be 'lib'");

        // Imports should be Named with "helper"
        match &use_stmt.imports {
            nostos_syntax::UseImports::Named(items) => {
                assert_eq!(items.len(), 1, "Should have 1 import");
                assert_eq!(items[0].name.node, "helper", "Import name should be 'helper'");
            }
            other => panic!("Expected Named imports, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_qualified_call_from_parser() {
        // Parse the actual source code and check what AST the parser produces
        let code = "main() = good.multiply(2, 3)";
        let (parsed, errors) = nostos_syntax::parse(code);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let module = parsed.expect("Parse should succeed");

        println!("Parsed module items: {:?}", module.items.len());

        // Get the function definition
        let fn_def = match &module.items[0] {
            nostos_syntax::Item::FnDef(f) => f,
            other => panic!("Expected FnDef, got {:?}", other),
        };

        // Print the AST structure
        println!("Function body: {:?}", fn_def.clauses[0].body);

        let deps = extract_dependencies_from_fn(&fn_def);
        println!("Dependencies extracted from parsed code: {:?}", deps);

        // Should contain "good.multiply", NOT just "good"
        assert!(deps.contains("good.multiply"),
            "Should extract 'good.multiply' as dependency from parsed code, but got: {:?}", deps);
    }

    #[test]
    fn test_record_field_access_not_dependency() {
        // Test that record field access like p.name is NOT treated as a dependency
        let code = r#"
type Person = { name: String, age: Int }

main() = {
    p = Person(name: "test", age: 10)
    n = p.name
    a = p.age
    n
}
"#;
        let (parsed, errors) = nostos_syntax::parse(code);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let module = parsed.expect("Parse should succeed");

        // Find the main function
        let fn_def = module.items.iter().find_map(|item| {
            if let nostos_syntax::Item::FnDef(fd) = item {
                if fd.name.node == "main" { Some(fd) } else { None }
            } else {
                None
            }
        }).expect("Should find main function");

        let deps = extract_dependencies_from_fn(fn_def);
        println!("Dependencies extracted from record field access code: {:?}", deps);

        // Should NOT contain p.name or p.age - these are record field accesses, not function calls
        assert!(!deps.iter().any(|d| d.contains("name")),
            "Should NOT have 'name' in deps (it's a field access, not a function call), got: {:?}", deps);
        assert!(!deps.iter().any(|d| d.contains("age")),
            "Should NOT have 'age' in deps (it's a field access, not a function call), got: {:?}", deps);
        assert!(!deps.iter().any(|d| d.starts_with("p.")),
            "Should NOT have 'p.anything' in deps, got: {:?}", deps);
    }
}
