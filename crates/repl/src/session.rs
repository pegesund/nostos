//! REPL session management with transactional semantics.
//!
//! The session is always in a valid, runnable state. Changes that would
//! introduce errors are rejected and the previous state is preserved.

use std::collections::{HashMap, HashSet};

use nostos_syntax::ast::{Expr, FnDef, MatchArm, Stmt};
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
pub fn extract_dependencies_from_fn(fn_def: &FnDef) -> HashSet<String> {
    let mut deps = HashSet::new();
    for clause in &fn_def.clauses {
        extract_dependencies_from_expr(&clause.body, &mut deps);
        if let Some(guard) = &clause.guard {
            extract_dependencies_from_expr(guard, &mut deps);
        }
    }
    deps
}

/// Extract function references from an expression.
fn extract_dependencies_from_expr(expr: &Expr, deps: &mut HashSet<String>) {
    match expr {
        Expr::Var(ident) => {
            deps.insert(ident.node.clone());
        }
        Expr::Call(callee, args, _) => {
            extract_dependencies_from_expr(callee, deps);
            for arg in args {
                extract_dependencies_from_expr(arg, deps);
            }
        }
        Expr::BinOp(left, _, right, _) => {
            extract_dependencies_from_expr(left, deps);
            extract_dependencies_from_expr(right, deps);
        }
        Expr::UnaryOp(_, operand, _) => {
            extract_dependencies_from_expr(operand, deps);
        }
        Expr::If(cond, then_branch, else_branch, _) => {
            extract_dependencies_from_expr(cond, deps);
            extract_dependencies_from_expr(then_branch, deps);
            extract_dependencies_from_expr(else_branch, deps);
        }
        Expr::Match(scrutinee, arms, _) => {
            extract_dependencies_from_expr(scrutinee, deps);
            for arm in arms {
                extract_dependencies_from_match_arm(arm, deps);
            }
        }
        Expr::Lambda(_, body, _) => {
            extract_dependencies_from_expr(body, deps);
        }
        Expr::Block(stmts, _) => {
            for stmt in stmts {
                extract_dependencies_from_stmt(stmt, deps);
            }
        }
        Expr::Tuple(elems, _) => {
            for elem in elems {
                extract_dependencies_from_expr(elem, deps);
            }
        }
        Expr::List(elems, tail, _) => {
            for elem in elems {
                extract_dependencies_from_expr(elem, deps);
            }
            if let Some(tail) = tail {
                extract_dependencies_from_expr(tail, deps);
            }
        }
        Expr::Record(_, fields, _) => {
            for field in fields {
                match field {
                    nostos_syntax::ast::RecordField::Positional(e) => {
                        extract_dependencies_from_expr(e, deps);
                    }
                    nostos_syntax::ast::RecordField::Named(_, e) => {
                        extract_dependencies_from_expr(e, deps);
                    }
                }
            }
        }
        Expr::RecordUpdate(_, base, fields, _) => {
            extract_dependencies_from_expr(base, deps);
            for field in fields {
                match field {
                    nostos_syntax::ast::RecordField::Positional(e) => {
                        extract_dependencies_from_expr(e, deps);
                    }
                    nostos_syntax::ast::RecordField::Named(_, e) => {
                        extract_dependencies_from_expr(e, deps);
                    }
                }
            }
        }
        Expr::FieldAccess(base, _, _) => {
            extract_dependencies_from_expr(base, deps);
        }
        Expr::Index(base, index, _) => {
            extract_dependencies_from_expr(base, deps);
            extract_dependencies_from_expr(index, deps);
        }
        Expr::MethodCall(receiver, _, args, _) => {
            extract_dependencies_from_expr(receiver, deps);
            for arg in args {
                extract_dependencies_from_expr(arg, deps);
            }
        }
        Expr::Map(pairs, _) => {
            for (k, v) in pairs {
                extract_dependencies_from_expr(k, deps);
                extract_dependencies_from_expr(v, deps);
            }
        }
        Expr::Set(elems, _) => {
            for elem in elems {
                extract_dependencies_from_expr(elem, deps);
            }
        }
        Expr::Try(body, arms, finally, _) => {
            extract_dependencies_from_expr(body, deps);
            for arm in arms {
                extract_dependencies_from_match_arm(arm, deps);
            }
            if let Some(finally) = finally {
                extract_dependencies_from_expr(finally, deps);
            }
        }
        Expr::Try_(inner, _) => {
            extract_dependencies_from_expr(inner, deps);
        }
        Expr::Quote(inner, _) => {
            extract_dependencies_from_expr(inner, deps);
        }
        Expr::Splice(inner, _) => {
            extract_dependencies_from_expr(inner, deps);
        }
        Expr::Do(stmts, _) => {
            for stmt in stmts {
                match stmt {
                    nostos_syntax::ast::DoStmt::Bind(_, e) => {
                        extract_dependencies_from_expr(e, deps);
                    }
                    nostos_syntax::ast::DoStmt::Expr(e) => {
                        extract_dependencies_from_expr(e, deps);
                    }
                }
            }
        }
        Expr::Receive(arms, timeout, _) => {
            for arm in arms {
                extract_dependencies_from_match_arm(arm, deps);
            }
            if let Some((duration, body)) = timeout {
                extract_dependencies_from_expr(duration, deps);
                extract_dependencies_from_expr(body, deps);
            }
        }
        Expr::Spawn(_, callee, args, _) => {
            extract_dependencies_from_expr(callee, deps);
            for arg in args {
                extract_dependencies_from_expr(arg, deps);
            }
        }
        Expr::Send(target, msg, _) => {
            extract_dependencies_from_expr(target, deps);
            extract_dependencies_from_expr(msg, deps);
        }
        Expr::String(lit, _) => {
            if let nostos_syntax::ast::StringLit::Interpolated(parts) = lit {
                for part in parts {
                    if let nostos_syntax::ast::StringPart::Expr(e) = part {
                        extract_dependencies_from_expr(e, deps);
                    }
                }
            }
        }
        // Loop expressions
        Expr::While(cond, body, _) => {
            extract_dependencies_from_expr(cond, deps);
            extract_dependencies_from_expr(body, deps);
        }
        Expr::For(_, start, end, body, _) => {
            extract_dependencies_from_expr(start, deps);
            extract_dependencies_from_expr(end, deps);
            extract_dependencies_from_expr(body, deps);
        }
        Expr::Break(value, _) => {
            if let Some(val) = value {
                extract_dependencies_from_expr(val, deps);
            }
        }
        Expr::Continue(_) => {}
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
fn extract_dependencies_from_match_arm(arm: &MatchArm, deps: &mut HashSet<String>) {
    if let Some(guard) = &arm.guard {
        extract_dependencies_from_expr(guard, deps);
    }
    extract_dependencies_from_expr(&arm.body, deps);
}

/// Extract dependencies from a statement.
fn extract_dependencies_from_stmt(stmt: &Stmt, deps: &mut HashSet<String>) {
    match stmt {
        Stmt::Expr(e) => {
            extract_dependencies_from_expr(e, deps);
        }
        Stmt::Let(binding) => {
            extract_dependencies_from_expr(&binding.value, deps);
        }
        Stmt::Assign(_, e, _) => {
            extract_dependencies_from_expr(e, deps);
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
            name: Spanned::new(name.to_string(), Span::default()),
            type_params: vec![],
            clauses: vec![FnClause {
                params: vec![],
                guard: None,
                return_type: None,
                body,
                span: Span::default(),
            }],
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
        Expr::Call(Box::new(var_expr(name)), args, Span::default())
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
}
