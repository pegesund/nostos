// Test for function call validation
// Run with: cargo test --release -p nostos-compiler compile_check

use std::collections::HashSet;
use nostos_syntax::{parse, offset_to_line_col};
use nostos_syntax::ast::{Expr, Item, Stmt};
use nostos_compiler::Compiler;

fn check_module_compiles_standalone(content: &str) -> Result<(), String> {
    let (module_opt, errors) = parse(content);

    if !errors.is_empty() {
        return Err(format!("Parse error: {:?}", errors));
    }

    let module = match module_opt {
        Some(m) => m,
        None => return Err("Failed to parse module".to_string()),
    };

    let mut known_functions: HashSet<String> = HashSet::new();

    for name in Compiler::get_builtin_names() {
        known_functions.insert(name.to_string());
    }

    for item in &module.items {
        if let Item::FnDef(fn_def) = item {
            known_functions.insert(fn_def.name.node.clone());
        }
    }

    fn get_call_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(ident) => Some(ident.node.clone()),
            Expr::FieldAccess(base, field, _) => {
                if let Expr::Var(module_ident) = base.as_ref() {
                    Some(format!("{}.{}", module_ident.node, field.node))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn collect_calls_stmt(stmt: &Stmt, calls: &mut Vec<(String, usize)>) {
        match stmt {
            Stmt::Expr(expr) => collect_calls(expr, calls),
            Stmt::Let(binding) => collect_calls(&binding.value, calls),
            Stmt::Assign(_, expr, _) => collect_calls(expr, calls),
        }
    }

    fn collect_calls(expr: &Expr, calls: &mut Vec<(String, usize)>) {
        match expr {
            Expr::Call(callee, _, args, span) => {
                if let Some(name) = get_call_name(callee) {
                    calls.push((name, span.start));
                }
                collect_calls(callee, calls);
                for arg in args {
                    collect_calls(arg, calls);
                }
            }
            Expr::MethodCall(receiver, method, args, span) => {
                // Check if receiver is a module/type name
                let module_name = match receiver.as_ref() {
                    Expr::Var(ident) => Some(&ident.node),
                    Expr::Record(ident, fields, _) if fields.is_empty() => Some(&ident.node),
                    _ => None,
                };
                if let Some(name) = module_name {
                    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let call_name = format!("{}.{}", name, method.node);
                        calls.push((call_name, span.start));
                    }
                }
                collect_calls(receiver, calls);
                for arg in args {
                    collect_calls(arg, calls);
                }
            }
            Expr::BinOp(left, _, right, _) => {
                collect_calls(left, calls);
                collect_calls(right, calls);
            }
            Expr::If(cond, then_branch, else_branch, _) => {
                collect_calls(cond, calls);
                collect_calls(then_branch, calls);
                collect_calls(else_branch, calls);
            }
            Expr::Block(stmts, _) => {
                for stmt in stmts {
                    collect_calls_stmt(stmt, calls);
                }
            }
            Expr::Lambda(_, body, _) => {
                collect_calls(body, calls);
            }
            Expr::Tuple(exprs, _) => {
                for e in exprs {
                    collect_calls(e, calls);
                }
            }
            _ => {}
        }
    }

    let mut all_calls = Vec::new();
    for item in &module.items {
        if let Item::FnDef(fn_def) = item {
            for clause in &fn_def.clauses {
                collect_calls(&clause.body, &mut all_calls);
            }
        }
        if let Item::Binding(binding) = item {
            collect_calls(&binding.value, &mut all_calls);
        }
    }

    for (call_name, offset) in all_calls {
        if known_functions.contains(&call_name) {
            continue;
        }

        if !call_name.contains('.') && call_name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
            continue;
        }

        let (line, _col) = offset_to_line_col(content, offset);
        return Err(format!("line {}: unknown function `{}`", line, call_name));
    }

    Ok(())
}

#[test]
fn test_valid_server_close() {
    let code = r#"
main() = {
  (status, server) = Server.bind(8888)
  Server.close(server)
  status
}
"#;
    assert!(check_module_compiles_standalone(code).is_ok());
}

#[test]
fn test_invalid_server_closex() {
    let code = r#"
main() = {
  (status, server) = Server.bind(8888)
  Server.closex(server)
  status
}
"#;
    let result = check_module_compiles_standalone(code);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Server.closex"));
}

#[test]
fn test_simple_invalid_method() {
    let code = "main() = Server.closex(42)";
    let result = check_module_compiles_standalone(code);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Server.closex"));
}

#[test]
fn test_builtins_include_server() {
    let builtins = Compiler::get_builtin_names();
    assert!(builtins.contains(&"Server.close"));
    assert!(builtins.contains(&"Server.bind"));
}
