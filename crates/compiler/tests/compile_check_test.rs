// Test for function call validation
// Run with: cargo test --release -p nostos-compiler --test compile_check_test

use std::collections::HashMap;
use std::collections::HashSet;
use nostos_syntax::{parse, offset_to_line_col};
use nostos_syntax::ast::{Expr, Item, Stmt, Pattern};
use nostos_compiler::Compiler;

/// Infer type from an expression
fn infer_expr_type(expr: &Expr, local_types: &HashMap<String, String>) -> Option<String> {
    match expr {
        Expr::Int(_, _) => Some("Int".to_string()),
        Expr::Float(_, _) => Some("Float".to_string()),
        Expr::String(_, _) => Some("String".to_string()),
        Expr::Bool(_, _) => Some("Bool".to_string()),
        Expr::Char(_, _) => Some("Char".to_string()),
        Expr::List(_, _, _) => Some("List".to_string()),
        Expr::Map(_, _) => Some("Map".to_string()),
        Expr::Set(_, _) => Some("Set".to_string()),
        Expr::Var(ident) => local_types.get(&ident.node).cloned(),
        Expr::Record(name, _, _) => Some(name.node.clone()),
        _ => None,
    }
}

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

    // Collect calls with local type context
    fn collect_calls_stmt(
        stmt: &Stmt,
        calls: &mut Vec<(String, usize)>,
        local_types: &mut HashMap<String, String>,
        known_functions: &HashSet<String>,
    ) {
        match stmt {
            Stmt::Expr(expr) => collect_calls(expr, calls, local_types, known_functions),
            Stmt::Let(binding) => {
                // Infer type from value and track it
                if let Some(ty) = infer_expr_type(&binding.value, local_types) {
                    if let Pattern::Var(ident) = &binding.pattern {
                        local_types.insert(ident.node.clone(), ty);
                    }
                }
                collect_calls(&binding.value, calls, local_types, known_functions);
            }
            Stmt::Assign(_, expr, _) => collect_calls(expr, calls, local_types, known_functions),
        }
    }

    fn collect_calls(
        expr: &Expr,
        calls: &mut Vec<(String, usize)>,
        local_types: &mut HashMap<String, String>,
        known_functions: &HashSet<String>,
    ) {
        match expr {
            Expr::Call(callee, _, args, span) => {
                if let Some(name) = get_call_name(callee) {
                    calls.push((name, span.start));
                }
                collect_calls(callee, calls, local_types, known_functions);
                for arg in args {
                    collect_calls(arg, calls, local_types, known_functions);
                }
            }
            Expr::MethodCall(receiver, method, args, span) => {
                // First, try to get the receiver type
                let receiver_type = match receiver.as_ref() {
                    // Capitalized name with no fields = module/type name (e.g., Server)
                    Expr::Record(ident, fields, _) if fields.is_empty() => {
                        Some(ident.node.clone())
                    }
                    // Variable - look up its type
                    Expr::Var(ident) => {
                        local_types.get(&ident.node).cloned()
                    }
                    // Try to infer from expression
                    _ => infer_expr_type(receiver, local_types),
                };

                if let Some(recv_type) = receiver_type {
                    let call_name = format!("{}.{}", recv_type, method.node);
                    calls.push((call_name, span.start));
                }

                collect_calls(receiver, calls, local_types, known_functions);
                for arg in args {
                    collect_calls(arg, calls, local_types, known_functions);
                }
            }
            Expr::BinOp(left, _, right, _) => {
                collect_calls(left, calls, local_types, known_functions);
                collect_calls(right, calls, local_types, known_functions);
            }
            Expr::If(cond, then_branch, else_branch, _) => {
                collect_calls(cond, calls, local_types, known_functions);
                collect_calls(then_branch, calls, local_types, known_functions);
                collect_calls(else_branch, calls, local_types, known_functions);
            }
            Expr::Block(stmts, _) => {
                for stmt in stmts {
                    collect_calls_stmt(stmt, calls, local_types, known_functions);
                }
            }
            Expr::Lambda(_, body, _) => {
                collect_calls(body, calls, local_types, known_functions);
            }
            Expr::Tuple(exprs, _) => {
                for e in exprs {
                    collect_calls(e, calls, local_types, known_functions);
                }
            }
            _ => {}
        }
    }

    let mut all_calls = Vec::new();
    let mut local_types = HashMap::new();
    for item in &module.items {
        if let Item::FnDef(fn_def) = item {
            for clause in &fn_def.clauses {
                local_types.clear();
                collect_calls(&clause.body, &mut all_calls, &mut local_types, &known_functions);
            }
        }
        if let Item::Binding(binding) = item {
            collect_calls(&binding.value, &mut all_calls, &mut local_types, &known_functions);
        }
    }

    for (call_name, offset) in all_calls {
        if known_functions.contains(&call_name) {
            continue;
        }

        // Skip unqualified lowercase names (local function calls)
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

#[test]
fn test_int_method_call_invalid() {
    // me = 42 (Int), so me.xxx() should check for Int.xxx
    let code = "main() = { me = 42; me.xxx() }";
    let result = check_module_compiles_standalone(code);
    println!("Result: {:?}", result);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Int.xxx"));
}

#[test]
fn test_string_method_call_valid() {
    // s = "hello" (String), s.length() should check for String.length which exists
    let code = r#"main() = { s = "hello"; s.length() }"#;
    let result = check_module_compiles_standalone(code);
    println!("Result: {:?}", result);
    assert!(result.is_ok());
}

#[test]
fn test_string_method_call_invalid() {
    // s = "hello" (String), s.xxx() should check for String.xxx which doesn't exist
    let code = r#"main() = { s = "hello"; s.xxx() }"#;
    let result = check_module_compiles_standalone(code);
    println!("Result: {:?}", result);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("String.xxx"));
}
