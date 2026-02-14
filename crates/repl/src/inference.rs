//! Shared type inference for autocomplete.
//!
//! This module contains ALL type inference logic used by:
//! - The LSP server (crates/lsp/) for VS Code completions and diagnostics
//! - The TUI file editor (crates/cli/src/editor.rs) for in-editor completions
//! - The TUI REPL panel (crates/cli/src/tui.rs) for REPL completions
//!
//! There is ONE implementation. All consumers call into this module.
//! Do NOT duplicate this logic elsewhere.

use std::collections::HashMap;
use crate::ReplEngine;

// ---------------------------------------------------------------------------
// High-level entry point
// ---------------------------------------------------------------------------

/// Infer the type of the expression before a dot, given the full document context.
///
/// This is the single entry point for dot-completion type inference.
/// It combines: extract local bindings → handle lambda context → infer expression type.
///
/// Used by the LSP server, TUI editor, and TUI REPL panel.
pub fn infer_dot_receiver_type(
    content: &str,
    cursor_line: usize,
    before_dot: &str,
    engine: Option<&ReplEngine>,
) -> Option<String> {
    let mut local_vars = extract_local_bindings(content, cursor_line, engine);

    // Add lambda parameters visible at cursor position
    extract_lambda_params_to_local_vars(before_dot, &mut local_vars);

    // Extract just the expression part if before_dot contains an assignment
    let expr_to_infer = if let Some(eq_pos) = before_dot.rfind('=') {
        let before_eq = &before_dot[..eq_pos];
        if !before_eq.ends_with('!') && !before_eq.ends_with('=')
            && !before_eq.ends_with('<') && !before_eq.ends_with('>')
        {
            before_dot[eq_pos + 1..].trim()
        } else {
            before_dot
        }
    } else {
        before_dot
    };

    // Extract the receiver expression (handles literals, brackets, etc.)
    let receiver_expr = extract_receiver_expression(expr_to_infer);

    // Extract the last identifier for variable lookup
    let identifier = before_dot
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .last()
        .unwrap_or("");

    // Try literal type first
    if let Some(lt) = detect_literal_type(receiver_expr) {
        return Some(lt.to_string());
    }

    // Try local variable lookup
    if let Some(ty) = local_vars.get(identifier) {
        return Some(ty.clone());
    }

    // Try field access type (e.g., self.name.)
    if let Some(engine) = engine {
        if let Some(field_type) = infer_field_access_type(before_dot, identifier, &local_vars, engine, content) {
            return Some(field_type);
        }
    }

    // Try indexed list literal (e.g., [["a","b"]][0][0])
    if let Some(idx_literal_type) = infer_indexed_list_literal_type(expr_to_infer) {
        return Some(idx_literal_type);
    }

    // Try index expression (e.g., arr[0])
    if let Some(idx_type) = infer_index_expr_type(expr_to_infer, &local_vars) {
        return Some(idx_type);
    }

    // Try function call return type
    if let Some(func_ret_type) = infer_rhs_type(expr_to_infer, engine, &local_vars) {
        return Some(func_ret_type);
    }

    // Fall back to engine's general expression type inference
    if let Some(engine) = engine {
        return engine.infer_expression_type(expr_to_infer, &local_vars);
    }

    None
}

// ---------------------------------------------------------------------------
// Local binding extraction
// ---------------------------------------------------------------------------

/// Resolve unresolved function parameters by scanning the ENTIRE file for call sites.
///
/// When `handleRequest(req, mainPid)` has generic param types, we search the file for
/// calls like `handleRequest(Server.accept(server), pid)` and infer the argument types.
fn resolve_params_from_call_sites(
    content: &str,
    fn_name: &str,
    unresolved: &[(usize, String)],  // (param_index, param_name)
    engine: Option<&ReplEngine>,
    bindings: &mut HashMap<String, String>,
) {
    let call_pattern = format!("{}(", fn_name);

    for line in content.lines() {
        let trimmed = line.trim();

        // Look for lines containing calls to this function
        let Some(call_pos) = trimmed.find(&call_pattern) else { continue };

        // Skip the function definition line itself: `name(params) = ...`
        let before_call = trimmed[..call_pos].trim();
        if before_call.is_empty() || before_call == "pub" {
            // Could be the definition line — check if it has `= ` after the params
            let after_call = &trimmed[call_pos + fn_name.len()..];
            if let Some(close_paren) = after_call.find(')') {
                let after_close = after_call[close_paren + 1..].trim();
                if after_close.starts_with('=') && !after_close.starts_with("==") {
                    continue; // This is the definition, not a call
                }
            }
        }

        // Extract arguments from the call
        let args_start = call_pos + call_pattern.len();
        let args_rest = &trimmed[args_start..];

        // Find the matching close paren (handle nested parens)
        let mut depth = 1;
        let mut end_pos = None;
        for (i, ch) in args_rest.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end_pos = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end_pos) = end_pos else { continue };
        let args_str = &args_rest[..end_pos];

        // Split arguments by comma (respecting nesting)
        let args = split_call_args(args_str);

        // Build bindings for the call site's scope (to resolve argument types)
        // We do a lightweight scan of the enclosing function for this call site
        let call_site_bindings = extract_call_site_bindings(content, trimmed, engine);

        for &(param_idx, ref param_name) in unresolved {
            if bindings.contains_key(param_name) {
                continue; // Already resolved
            }
            if let Some(arg_expr) = args.get(param_idx) {
                let arg = arg_expr.trim();
                // Try to infer the argument type using the call site's bindings
                if let Some(ty) = infer_rhs_type(arg, engine, &call_site_bindings) {
                    bindings.insert(param_name.clone(), ty);
                }
            }
        }

        // If all params resolved, stop searching
        if unresolved.iter().all(|(_, name)| bindings.contains_key(name)) {
            break;
        }
    }
}

/// Split function call arguments by comma, respecting nested parens/brackets.
pub fn split_call_args(args: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let mut in_string = false;

    let chars: Vec<char> = args.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '"' && (i == 0 || chars[i - 1] != '\\') {
            in_string = !in_string;
            current.push(c);
            continue;
        }
        if in_string {
            current.push(c);
            continue;
        }
        match c {
            '(' => { paren_depth += 1; current.push(c); }
            ')' => { paren_depth -= 1; current.push(c); }
            '[' => { bracket_depth += 1; current.push(c); }
            ']' => { bracket_depth -= 1; current.push(c); }
            ',' if paren_depth == 0 && bracket_depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Build bindings for the scope surrounding a call site line.
/// Finds the enclosing function and extracts bindings from it.
fn extract_call_site_bindings(
    content: &str,
    target_line: &str,
    engine: Option<&ReplEngine>,
) -> HashMap<String, String> {
    let mut bindings = HashMap::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == target_line {
            break;
        }

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Extract simple bindings: x = expr
        if let Some(eq_pos) = trimmed.find('=') {
            let before_eq = trimmed[..eq_pos].trim();
            let after_eq_start = eq_pos + 1;
            if after_eq_start < trimmed.len() && !trimmed[after_eq_start..].starts_with('=') {
                let after_eq = trimmed[after_eq_start..].trim();

                if !before_eq.is_empty()
                    && before_eq.chars().next().map_or(false, |c| c.is_lowercase())
                    && before_eq.chars().all(|c| c.is_alphanumeric() || c == '_')
                {
                    if let Some(ty) = infer_rhs_type(after_eq, engine, &bindings) {
                        bindings.insert(before_eq.to_string(), ty);
                    }
                }
            }
        }
    }

    bindings
}

/// Scan source code up to a given line and extract local variable bindings with their inferred types.
///
/// Handles:
/// - Simple bindings: `x = expr`
/// - Type-annotated bindings: `x: Type = expr`
/// - Mvar declarations: `mvar name: Type = expr`
/// - Trait impl `self` parameter: inside `TypeName: TraitName ... end` blocks
/// - Function parameters: `name(param1, param2) = ...` with call-site type resolution
pub fn extract_local_bindings(
    content: &str,
    up_to_line: usize,
    engine: Option<&ReplEngine>,
) -> HashMap<String, String> {
    let mut bindings = HashMap::new();

    // Track trait implementation context for `self` type inference
    let mut current_impl_type: Option<String> = None;

    // Track function parameter names so we can remove them when entering a new function scope
    let mut fn_param_names: Vec<String> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let is_current_line = line_num == up_to_line;
        if line_num > up_to_line {
            break;
        }

        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Check for trait impl header: "TypeName: TraitName" (no =)
        if !trimmed.contains('=') {
            if trimmed == "end" {
                current_impl_type = None;
                continue;
            }

            if let Some(colon_pos) = trimmed.find(':') {
                let before_colon = trimmed[..colon_pos].trim();
                let after_colon = trimmed[colon_pos + 1..].trim();

                let type_name = before_colon.split('[').next().unwrap_or(before_colon).trim();
                if !type_name.is_empty()
                    && type_name.chars().next().map_or(false, |c| c.is_uppercase())
                    && !after_colon.is_empty()
                    && after_colon.chars().next().map_or(false, |c| c.is_uppercase())
                    && after_colon.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '[' || c == ']' || c == ',')
                {
                    current_impl_type = Some(before_colon.to_string());
                    continue;
                }
            }
        }

        // Detect function definition: name(param1, param2, ...) = ...
        // Extract parameters as local bindings within the function body.
        if trimmed.contains('(') && trimmed.contains('=') {
            let check = if trimmed.starts_with("pub ") { &trimmed[4..] } else { trimmed };

            if let Some(paren_start) = check.find('(') {
                let name_part = check[..paren_start].trim();

                if !name_part.is_empty()
                    && name_part.chars().next().map_or(false, |c| c.is_lowercase() || c == '_')
                    && name_part.chars().all(|c| c.is_alphanumeric() || c == '_')
                {
                    if let Some(paren_end_rel) = check[paren_start..].find(')') {
                        let paren_end = paren_start + paren_end_rel;
                        let after_paren = check[paren_end + 1..].trim();

                        if after_paren.starts_with('=') && !after_paren.starts_with("==") {
                            // New function scope — remove previous function's params
                            for name in &fn_param_names {
                                bindings.remove(name);
                            }
                            fn_param_names.clear();

                            let fn_name = name_part;
                            let params_str = &check[paren_start + 1..paren_end];

                            // Get parameter types from engine (compiler-inferred)
                            let param_types = engine.and_then(|e| e.get_function_params(fn_name));

                            // Collect param names for call-site resolution
                            let mut unresolved_params: Vec<(usize, String)> = Vec::new();

                            for (i, param) in params_str.split(',').enumerate() {
                                let param = param.trim();
                                if param.is_empty() || param == "_" {
                                    continue;
                                }

                                // Handle type annotation: param: Type
                                let (pname, explicit_type) = if let Some(colon_pos) = param.find(':') {
                                    let n = param[..colon_pos].trim();
                                    let t = param[colon_pos + 1..].trim();
                                    (n, if t.is_empty() { None } else { Some(t.to_string()) })
                                } else {
                                    (param, None)
                                };

                                if pname.is_empty() || !pname.chars().all(|c| c.is_alphanumeric() || c == '_') {
                                    continue;
                                }

                                fn_param_names.push(pname.to_string());

                                let final_type = explicit_type.or_else(|| {
                                    param_types.as_ref().and_then(|pts| {
                                        pts.get(i).and_then(|(_, ptype, _, _)| {
                                            // Skip unresolved types (?) and single-letter type variables
                                            if !ptype.is_empty()
                                                && !ptype.contains('?')
                                                && !(ptype.len() == 1 && ptype.chars().next().map_or(false, |c| c.is_lowercase()))
                                            {
                                                Some(ptype.clone())
                                            } else {
                                                None
                                            }
                                        })
                                    })
                                });

                                if let Some(ty) = final_type {
                                    bindings.insert(pname.to_string(), ty);
                                } else {
                                    unresolved_params.push((i, pname.to_string()));
                                }
                            }

                            // If we have unresolved parameters, scan the entire file for
                            // call sites to infer argument types
                            if !unresolved_params.is_empty() {
                                resolve_params_from_call_sites(
                                    content,
                                    fn_name,
                                    &unresolved_params,
                                    engine,
                                    &mut bindings,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Detect `self` parameter in trait impl methods
        if let Some(ref impl_type) = current_impl_type {
            if let Some(paren_pos) = trimmed.find('(') {
                let params_start = paren_pos + 1;
                if let Some(params_end) = trimmed[params_start..].find(')') {
                    let params = &trimmed[params_start..params_start + params_end];
                    let first_param = params.split(',').next().unwrap_or("").trim();
                    if first_param == "self" {
                        bindings.insert("self".to_string(), impl_type.clone());
                    }
                }
            }
        }

        // Skip binding extraction for current line (only detect self above)
        if is_current_line {
            continue;
        }

        // Mvar declarations: "mvar name: Type = expr"
        if trimmed.starts_with("mvar ") {
            let rest = trimmed[5..].trim();
            if let Some(colon_pos) = rest.find(':') {
                let var_name = rest[..colon_pos].trim();
                let after_colon = rest[colon_pos + 1..].trim();
                if let Some(eq_pos) = after_colon.find('=') {
                    let type_name = after_colon[..eq_pos].trim();
                    if !var_name.is_empty() && !type_name.is_empty() {
                        bindings.insert(var_name.to_string(), type_name.to_string());
                    }
                }
            }
            continue;
        }

        // Simple bindings: "x = expr" or "x:Type = expr"
        // Also tuple destructuring: "(a, b, c) = expr"
        if let Some(eq_pos) = trimmed.find('=') {
            let before_eq = trimmed[..eq_pos].trim();
            let after_eq_start = eq_pos + 1;
            if after_eq_start < trimmed.len() && !trimmed[after_eq_start..].starts_with('=') {
                let after_eq = trimmed[after_eq_start..].trim();

                // Tuple destructuring: (a, b, c) = expr
                if before_eq.starts_with('(') && before_eq.ends_with(')') {
                    let inner = &before_eq[1..before_eq.len() - 1];
                    let names: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                    let all_valid = names.iter().all(|n| {
                        !n.is_empty()
                            && (n.chars().next().map_or(false, |c| c.is_lowercase() || c == '_'))
                            && n.chars().all(|c| c.is_alphanumeric() || c == '_')
                    });

                    if all_valid && names.len() >= 2 {
                        // Try to infer the RHS tuple type
                        if let Some(rhs_type) = infer_rhs_type(after_eq, engine, &bindings) {
                            // Parse tuple type: (Type1, Type2, Type3)
                            let element_types = parse_tuple_element_types(&rhs_type);
                            if element_types.len() == names.len() {
                                for (name, ty) in names.iter().zip(element_types.iter()) {
                                    bindings.insert(name.to_string(), ty.clone());
                                }
                            }
                        }
                    }
                } else {
                    // Simple binding
                    let (var_name, explicit_type) = if let Some(colon_pos) = before_eq.find(':') {
                        let name = before_eq[..colon_pos].trim();
                        let ty = before_eq[colon_pos + 1..].trim();
                        (name, Some(ty.to_string()))
                    } else {
                        (before_eq, None)
                    };

                    if !var_name.is_empty()
                        && var_name.chars().next().map_or(false, |c| c.is_lowercase())
                        && var_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        let final_type = if let Some(ty) = explicit_type {
                            Some(ty)
                        } else {
                            infer_rhs_type(after_eq, engine, &bindings)
                        };

                        if let Some(ty) = final_type {
                            bindings.insert(var_name.to_string(), ty);
                        }
                    }
                }
            }
        }
    }

    bindings
}

// ---------------------------------------------------------------------------
// RHS type inference (main dispatcher)
// ---------------------------------------------------------------------------

/// Infer the type of an expression on the right-hand side of a binding.
///
/// This is the main dispatcher that tries method chains, index expressions,
/// literals, record/variant constructors, function calls, etc.
pub fn infer_rhs_type(
    expr: &str,
    engine: Option<&ReplEngine>,
    current_bindings: &HashMap<String, String>,
) -> Option<String> {
    let trimmed = expr.trim();

    // Variable reference: look up in current bindings
    if !trimmed.is_empty()
        && trimmed.chars().next().map_or(false, |c| c.is_lowercase() || c == '_')
        && trimmed.chars().all(|c| c.is_alphanumeric() || c == '_')
    {
        if let Some(ty) = current_bindings.get(trimmed) {
            return Some(ty.clone());
        }
    }

    // Method chain or field access: x.method().field, server.accept(), req.path
    if trimmed.contains('.') {
        if let Some(inferred) = infer_method_chain_type(trimmed, current_bindings, engine) {
            return Some(inferred);
        }
    }

    // Index expression: arr[0][0]
    if trimmed.contains('[') && !trimmed.starts_with('[') {
        if let Some(inferred) = infer_index_expr_type(trimmed, current_bindings) {
            return Some(inferred);
        }
    }

    // List literals (possibly indexed)
    if trimmed.starts_with('[') {
        if let Some(indexed_type) = infer_indexed_list_literal_type(trimmed) {
            return Some(indexed_type);
        }
        return infer_list_type(trimmed);
    }

    if trimmed.starts_with('"') {
        return Some("String".to_string());
    }
    if trimmed.starts_with("%{") {
        return Some("Map".to_string());
    }
    if trimmed.starts_with("#{") {
        return Some("Set".to_string());
    }

    // Tuple literals: (42, "hello")
    if trimmed.starts_with('(') && trimmed.ends_with(')') && trimmed.contains(',') {
        if let Some(tuple_type) = infer_tuple_type(trimmed) {
            return Some(tuple_type);
        }
    }

    // Record/Variant construction: TypeName(field: value) or ConstructorName(value)
    if let Some(first_char) = trimmed.chars().next() {
        if first_char.is_uppercase() {
            let name: String = trimmed.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();

            if !name.is_empty() {
                let rest = trimmed[name.len()..].trim_start();
                let is_construction = rest.starts_with('(');

                if let Some(engine) = engine {
                    if let Some(type_name) = engine.get_type_for_constructor(&name) {
                        return Some(type_name);
                    }

                    let types = engine.get_types();
                    if types.contains(&name) {
                        return Some(name);
                    }
                    for registered_type in &types {
                        let type_base = registered_type.rsplit('.').next().unwrap_or(registered_type);
                        if type_base == name {
                            return Some(registered_type.clone());
                        }
                    }
                }

                if is_construction {
                    if rest.contains(':') {
                        return Some(name);
                    } else {
                        return Some(name);
                    }
                }
            }
        }
    }

    // Numeric literals
    if trimmed.chars().all(|c| c.is_ascii_digit() || c == '-') && !trimmed.is_empty() {
        return Some("Int".to_string());
    }
    if trimmed.contains('.') && trimmed.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-') {
        return Some("Float".to_string());
    }

    // Function call: Module.func(...) or func(...)
    if let Some(paren_pos) = trimmed.find('(') {
        let func_part = trimmed[..paren_pos].trim();
        let args_part = &trimmed[paren_pos..];
        if let Some(engine) = engine {
            if let Some(sig) = engine.get_function_signature(func_part) {
                if let Some(arrow_pos) = sig.rfind("->") {
                    let ret_type = sig[arrow_pos + 2..].trim();

                    if ret_type.len() == 1 && ret_type.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                        if let Some(first_arg_type) = infer_first_arg_type(args_part, current_bindings) {
                            return Some(first_arg_type);
                        }
                    }

                    return Some(ret_type.to_string());
                }
            }
        }
    }

    None
}

/// Parse tuple element types from a tuple type string like "(Int, Int, Int)".
/// Returns the individual element types, or empty vec if not a tuple.
fn parse_tuple_element_types(type_str: &str) -> Vec<String> {
    let trimmed = type_str.trim();
    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return Vec::new();
    }
    let inner = &trimmed[1..trimmed.len() - 1];

    // Split by comma respecting nested parens/brackets
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in inner.chars() {
        match ch {
            '(' | '[' => { depth += 1; current.push(ch); }
            ')' | ']' => { depth -= 1; current.push(ch); }
            ',' if depth == 0 => {
                let t = current.trim().to_string();
                if !t.is_empty() {
                    result.push(t);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let t = current.trim().to_string();
    if !t.is_empty() {
        result.push(t);
    }
    result
}

// ---------------------------------------------------------------------------
// Literal type inference
// ---------------------------------------------------------------------------

/// Detect the type of a literal expression (string, list, map, set, int, float).
pub fn detect_literal_type(expr: &str) -> Option<&'static str> {
    let trimmed = expr.trim();

    if trimmed.starts_with('"') || trimmed.starts_with('\'') {
        return Some("String");
    }

    if trimmed.starts_with('[') {
        // Check if this is an indexed list literal
        let mut depth = 0;
        let mut list_end = None;
        for (i, c) in trimmed.chars().enumerate() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        list_end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        if let Some(end_idx) = list_end {
            let after_list = &trimmed[end_idx + 1..];
            if after_list.starts_with('[') {
                return None; // Indexed list literal — handled elsewhere
            }
        }

        return Some("List");
    }

    if trimmed.starts_with("%{") {
        return Some("Map");
    }

    if trimmed.starts_with("#{") {
        return Some("Set");
    }

    let num_part = trimmed.strip_prefix('-').unwrap_or(trimmed);
    if !num_part.is_empty() && num_part.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        if num_part.contains('.') {
            return Some("Float");
        }
        return Some("Int");
    }

    None
}

/// Infer the type of a list literal, handling nested lists.
/// e.g., `[[0,1]]` → `List[List[Int]]`, `[1,2,3]` → `List[Int]`
pub fn infer_list_type(expr: &str) -> Option<String> {
    let trimmed = expr.trim();

    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return None;
    }

    let inner = trimmed[1..trimmed.len() - 1].trim();

    if inner.is_empty() {
        return Some("List".to_string());
    }

    let first_elem = extract_first_list_element(inner)?;
    let first_trimmed = first_elem.trim();

    let elem_type = if first_trimmed.starts_with('[') {
        infer_list_type(first_trimmed)?
    } else if first_trimmed.starts_with('"') {
        "String".to_string()
    } else if first_trimmed.parse::<i64>().is_ok() {
        "Int".to_string()
    } else if first_trimmed.parse::<f64>().is_ok() {
        "Float".to_string()
    } else if first_trimmed.chars().next().map_or(false, |c| c.is_uppercase()) {
        let name: String = first_trimmed.chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            name
        } else {
            return Some("List".to_string());
        }
    } else {
        return Some("List".to_string());
    };

    Some(format!("List[{}]", elem_type))
}

/// Extract the first element from a list interior, handling nested brackets.
pub fn extract_first_list_element(inner: &str) -> Option<String> {
    let mut depth = 0;
    let mut end_pos = inner.len();

    for (i, c) in inner.chars().enumerate() {
        match c {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth -= 1,
            ',' if depth == 0 => {
                end_pos = i;
                break;
            }
            _ => {}
        }
    }

    Some(inner[..end_pos].to_string())
}

/// Infer the type of an indexed list literal expression.
/// e.g., `[["a","b"]][0]` → `List[String]`, `[["a","b"]][0][0]` → `String`
pub fn infer_indexed_list_literal_type(expr: &str) -> Option<String> {
    let trimmed = expr.trim();

    if !trimmed.starts_with('[') {
        return None;
    }

    let mut depth = 0;
    let mut list_end = None;

    for (i, c) in trimmed.chars().enumerate() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    list_end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    let list_end = list_end?;
    let after_list = &trimmed[list_end + 1..];
    if !after_list.starts_with('[') {
        return None;
    }

    let index_count = after_list.matches('[').count();
    if index_count == 0 {
        return None;
    }

    let list_literal = &trimmed[..=list_end];
    let base_type = infer_list_type(list_literal)?;

    let mut current_type = base_type;
    for _ in 0..index_count {
        if current_type.starts_with("List[") && current_type.ends_with(']') {
            current_type = current_type
                .strip_prefix("List[")?
                .strip_suffix(']')?
                .to_string();
        } else if current_type == "List" {
            return None;
        } else {
            return Some(current_type);
        }
    }

    Some(current_type)
}

/// Infer the type of a tuple literal like `(42, "hello", true)` → `(Int, String, Bool)`
pub fn infer_tuple_type(expr: &str) -> Option<String> {
    let trimmed = expr.trim();

    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return None;
    }

    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return Some("()".to_string());
    }

    let mut elements = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for c in inner.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                elements.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        elements.push(current.trim().to_string());
    }

    let mut types = Vec::new();
    for elem in elements {
        let elem_type = if elem.starts_with('"') {
            "String".to_string()
        } else if elem.starts_with('[') {
            infer_list_type(&elem).unwrap_or_else(|| "List".to_string())
        } else if elem.starts_with('(') && elem.contains(',') {
            infer_tuple_type(&elem).unwrap_or_else(|| "Tuple".to_string())
        } else if elem == "true" || elem == "false" {
            "Bool".to_string()
        } else if elem.parse::<i64>().is_ok() {
            "Int".to_string()
        } else if elem.parse::<f64>().is_ok() {
            "Float".to_string()
        } else if elem.chars().next().map_or(false, |c| c.is_uppercase()) {
            elem.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect()
        } else {
            "Unknown".to_string()
        };
        types.push(elem_type);
    }

    Some(format!("({})", types.join(", ")))
}

// ---------------------------------------------------------------------------
// Method chain type inference
// ---------------------------------------------------------------------------

/// Infer the type of a method chain expression like `[["a","b"]].get(0).get(0)`.
///
/// Handles both method calls (`obj.method(args)`) and field access (`obj.field`).
/// When the static lookup table fails, delegates to the engine for resolution
/// of builtin types, UFCS methods, field access, trait methods, etc.
pub fn infer_method_chain_type(
    expr: &str,
    local_vars: &HashMap<String, String>,
    engine: Option<&ReplEngine>,
) -> Option<String> {
    let trimmed = expr.trim();
    let mut current_type: Option<String> = None;
    let mut remaining = trimmed;

    // Find the base expression (before first dot-access)
    let mut depth = 0;
    let mut base_end = 0;
    let chars: Vec<char> = remaining.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        match c {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth -= 1,
            '.' if depth == 0 => {
                let after_dot: String = chars[i+1..].iter().collect();
                if after_dot.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                    base_end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    if base_end == 0 {
        if trimmed.starts_with('[') {
            return infer_list_type(trimmed);
        } else if trimmed.starts_with('"') {
            return Some("String".to_string());
        } else if let Some(ty) = local_vars.get(trimmed) {
            return Some(ty.clone());
        }
        return None;
    }

    let base_expr = &remaining[..base_end];
    remaining = &remaining[base_end..];

    if base_expr.starts_with('[') {
        current_type = infer_list_type(base_expr);
    } else if base_expr.starts_with('"') {
        current_type = Some("String".to_string());
    } else if let Some(ty) = local_vars.get(base_expr.trim()) {
        current_type = Some(ty.clone());
    } else if base_expr.trim().chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        // Uppercase identifier = type name (e.g., Server.bind(), WebSocket.connect())
        current_type = Some(base_expr.trim().to_string());
    }

    // Process each part in the chain (method calls and field accesses)
    while !remaining.is_empty() && remaining.starts_with('.') {
        remaining = &remaining[1..]; // skip the dot

        if remaining.is_empty() {
            break;
        }

        // Extract the name (alphabetic + digits + _)
        let name_end = remaining
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(remaining.len());
        let name = &remaining[..name_end];

        if name.is_empty() {
            break;
        }

        let after_name = &remaining[name_end..];

        let advance_to;
        if after_name.starts_with('(') {
            // Method call: name(args) — find matching close paren
            let paren_start = name_end;
            let mut depth = 0;
            let mut close_paren = None;
            for (i, c) in remaining[paren_start..].chars().enumerate() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            close_paren = Some(paren_start + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let close_paren = close_paren?;
            advance_to = close_paren + 1;
        } else {
            // Field access: just the name, no parens
            advance_to = name_end;
        }

        // Resolve the type of this part
        if let Some(ref recv_type) = current_type {
            // Try static table first (fast, no engine needed)
            let resolved = infer_method_return_type_static(recv_type, name);
            if resolved.is_some() {
                current_type = resolved;
            } else if let Some(engine) = engine {
                // Fall back to engine for builtin types, UFCS, field access, etc.
                current_type = engine.get_method_return_type(recv_type, name);
            } else {
                current_type = None;
            }
        } else {
            return None;
        }

        remaining = &remaining[advance_to..];
    }

    current_type
}

/// Infer the return type of a method call based on receiver type.
/// This is a static lookup table — no engine needed.
pub fn infer_method_return_type_static(receiver_type: &str, method_name: &str) -> Option<String> {
    // Generic methods
    match method_name {
        "show" => return Some("String".to_string()),
        "hash" => return Some("Int".to_string()),
        "copy" => return Some(receiver_type.to_string()),
        _ => {}
    }

    let (base_type, elem_type) = if receiver_type.starts_with("List[") && receiver_type.ends_with(']') {
        ("List", Some(&receiver_type[5..receiver_type.len()-1]))
    } else if receiver_type.starts_with("Option[") && receiver_type.ends_with(']') {
        ("Option", Some(&receiver_type[7..receiver_type.len()-1]))
    } else {
        (receiver_type, None)
    };

    match base_type {
        "List" => {
            match method_name {
                "filter" | "take" | "drop" | "reverse" | "sort" | "unique" |
                "takeWhile" | "dropWhile" | "init" | "tail" | "push" | "remove" |
                "removeAt" | "insertAt" | "set" | "slice" => {
                    if let Some(elem) = elem_type {
                        Some(format!("List[{}]", elem))
                    } else {
                        Some("List".to_string())
                    }
                }
                "get" | "head" | "last" | "nth" | "find" | "sum" | "product" |
                "maximum" | "minimum" => {
                    elem_type.map(|e| e.to_string())
                }
                "any" | "all" | "contains" | "isEmpty" => Some("Bool".to_string()),
                "length" | "len" | "count" | "indexOf" => Some("Int".to_string()),
                "first" | "safeHead" | "safeLast" => {
                    elem_type.map(|e| format!("Option[{}]", e))
                }
                "map" | "flatMap" => Some("List".to_string()),
                "enumerate" => {
                    if let Some(elem) = elem_type {
                        Some(format!("List[(Int, {})]", elem))
                    } else {
                        Some("List".to_string())
                    }
                }
                "flatten" => {
                    if let Some(elem) = elem_type {
                        if elem.starts_with("List[") {
                            Some(elem.to_string())
                        } else {
                            Some(format!("List[{}]", elem))
                        }
                    } else {
                        Some("List".to_string())
                    }
                }
                _ => None,
            }
        }
        "String" => {
            match method_name {
                "chars" => Some("List[Char]".to_string()),
                "lines" | "words" | "split" => Some("List[String]".to_string()),
                "trim" | "trimStart" | "trimEnd" | "toUpper" | "toLower" |
                "replace" | "replaceAll" | "substring" | "repeat" |
                "padStart" | "padEnd" | "reverse" => Some("String".to_string()),
                "length" | "indexOf" | "lastIndexOf" => Some("Int".to_string()),
                "contains" | "startsWith" | "endsWith" | "isEmpty" => Some("Bool".to_string()),
                _ => None,
            }
        }
        "Option" => {
            match method_name {
                "unwrap" | "getOrElse" => elem_type.map(|e| e.to_string()),
                "isSome" | "isNone" => Some("Bool".to_string()),
                "map" => Some("Option".to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Index expression type inference
// ---------------------------------------------------------------------------

/// Infer type of an index expression like `g2[0]` or `g2[0][0]`.
/// If `g2` has type `List[List[String]]`, then `g2[0]` → `List[String]`, `g2[0][0]` → `String`.
pub fn infer_index_expr_type(expr: &str, local_vars: &HashMap<String, String>) -> Option<String> {
    let trimmed = expr.trim();

    if !trimmed.contains('[') {
        return None;
    }

    let first_bracket = trimmed.find('[')?;
    let base_var = trimmed[..first_bracket].trim();

    if base_var.is_empty() {
        return None;
    }

    let base_type = local_vars.get(base_var)?;
    let index_count = trimmed.matches('[').count();

    let mut current_type = base_type.clone();
    for _ in 0..index_count {
        if current_type.starts_with("List[") && current_type.ends_with(']') {
            current_type = current_type
                .strip_prefix("List[")?
                .strip_suffix(']')?
                .to_string();
        } else if current_type == "List" {
            return None;
        } else {
            return None;
        }
    }

    Some(current_type)
}

// ---------------------------------------------------------------------------
// Field access type inference
// ---------------------------------------------------------------------------

/// Infer the type of a field access like `self.age` where `self` is in local_vars.
///
/// Handles chained completions like `self.age.` by:
/// 1. Finding `self` in local_vars → `Person`
/// 2. Finding `age` field in `Person` → `Int`
pub fn infer_field_access_type(
    before_dot: &str,
    field_name: &str,
    local_vars: &HashMap<String, String>,
    engine: &ReplEngine,
    document_content: &str,
) -> Option<String> {
    let pattern = format!(".{}", field_name);
    let field_start = before_dot.rfind(&pattern)?;

    let before_field = &before_dot[..field_start];
    let base_var = before_field
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .last()?;

    let base_type = local_vars.get(base_var)?;

    // Tuple element access: t.0, t.1, etc.
    if base_type.starts_with('(') && base_type.ends_with(')') {
        if let Ok(index) = field_name.parse::<usize>() {
            let inner = &base_type[1..base_type.len()-1];
            let mut element_types = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for c in inner.chars() {
                match c {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current.push(c);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current.push(c);
                    }
                    ',' if depth == 0 => {
                        element_types.push(current.trim().to_string());
                        current = String::new();
                    }
                    _ => current.push(c),
                }
            }
            if !current.trim().is_empty() {
                element_types.push(current.trim().to_string());
            }

            if index < element_types.len() {
                return Some(element_types[index].clone());
            }
        }
    }

    // Engine field lookup
    if let Some(field_type) = engine.get_field_type(base_type, field_name) {
        return Some(field_type);
    }

    // Fallback: extract from source
    let fields = extract_type_fields_from_source(document_content, base_type);
    for field in fields {
        if let Some(colon_pos) = field.find(':') {
            let name = field[..colon_pos].trim();
            let ty = field[colon_pos + 1..].trim();
            if name == field_name {
                return Some(ty.to_string());
            }
        }
    }

    None
}

/// Extract record type fields directly from source code.
/// Works even when the file has parse errors elsewhere.
pub fn extract_type_fields_from_source(content: &str, type_name: &str) -> Vec<String> {
    let mut fields = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("type ") {
            let rest = trimmed[5..].trim();
            let def_type_name = rest.split(|c| c == '=' || c == '[')
                .next()
                .unwrap_or("")
                .trim();

            if def_type_name == type_name {
                if let Some(brace_start) = trimmed.find('{') {
                    let after_brace = &trimmed[brace_start + 1..];
                    if let Some(brace_end) = after_brace.find('}') {
                        let fields_str = &after_brace[..brace_end];
                        for field in fields_str.split(',') {
                            let field_trimmed = field.trim();
                            if !field_trimmed.is_empty() {
                                fields.push(field_trimmed.to_string());
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    fields
}

// ---------------------------------------------------------------------------
// Lambda parameter type inference
// ---------------------------------------------------------------------------

/// Extract all visible lambda parameters from the prefix and add them to local_vars.
/// Enables field access completion on lambda params like `people.map(p => p.age.)`.
pub fn extract_lambda_params_to_local_vars(
    prefix: &str,
    local_vars: &mut HashMap<String, String>,
) {
    let mut pos = 0;
    let chars: Vec<char> = prefix.chars().collect();

    while pos < chars.len() {
        if pos + 1 < chars.len() && chars[pos] == '=' && chars[pos + 1] == '>' {
            let arrow_pos = pos;

            let mut param_end = arrow_pos;
            while param_end > 0 && chars[param_end - 1].is_whitespace() {
                param_end -= 1;
            }

            let mut param_start = param_end;
            while param_start > 0 && (chars[param_start - 1].is_alphanumeric() || chars[param_start - 1] == '_') {
                param_start -= 1;
            }

            if param_start < param_end {
                let param_name: String = chars[param_start..param_end].iter().collect();

                if !local_vars.contains_key(&param_name) {
                    let mut paren_pos = param_start;
                    while paren_pos > 0 && chars[paren_pos - 1] != '(' {
                        paren_pos -= 1;
                    }

                    if paren_pos > 0 {
                        let before_paren: String = chars[..paren_pos - 1].iter().collect();
                        let before_paren = before_paren.trim_end();

                        if let Some(dot_pos) = before_paren.rfind('.') {
                            let method_name = before_paren[dot_pos + 1..].trim();
                            let mut receiver_expr = before_paren[..dot_pos].trim();

                            if let Some(arrow_idx) = receiver_expr.rfind("=>") {
                                receiver_expr = receiver_expr[arrow_idx + 2..].trim();
                            }

                            if let Some(receiver_type) = infer_method_chain_type(receiver_expr, local_vars, None) {
                                if let Some(param_type) = infer_lambda_param_type_for_method(&receiver_type, method_name) {
                                    local_vars.insert(param_name, param_type);
                                }
                            }
                        }
                    }
                }
            }

            pos += 2;
        } else {
            pos += 1;
        }
    }
}

/// Infer the type of a lambda parameter from context.
/// For `yy.map(m => m.` where `yy` is a `List`, returns the element type.
pub fn infer_lambda_param_type(
    full_prefix: &str,
    before_dot: &str,
    local_vars: &HashMap<String, String>,
) -> Option<String> {
    infer_lambda_param_type_recursive(full_prefix, before_dot, local_vars, 0)
}

/// Recursive helper for lambda parameter type inference.
fn infer_lambda_param_type_recursive(
    full_prefix: &str,
    before_dot: &str,
    local_vars: &HashMap<String, String>,
    depth: usize,
) -> Option<String> {
    if depth > 5 {
        return None;
    }

    let param_name = before_dot
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .last()?;

    let lambda_pattern = format!("{} =>", param_name);
    let alt_pattern1 = format!("{}=>", param_name);
    let alt_pattern2 = format!("{} =", param_name);
    let alt_pattern3 = format!("{}=", param_name);

    let arrow_pos = full_prefix.rfind(&lambda_pattern)
        .or_else(|| full_prefix.rfind(&alt_pattern1))
        .or_else(|| full_prefix.rfind(&alt_pattern2))
        .or_else(|| full_prefix.rfind(&alt_pattern3))?;

    let before_lambda = &full_prefix[..arrow_pos];

    let mut paren_depth: i32 = 0;
    let mut method_call_start = None;
    for (i, c) in before_lambda.chars().rev().enumerate() {
        match c {
            ')' | ']' | '}' => paren_depth += 1,
            '(' => {
                if paren_depth == 0 {
                    method_call_start = Some(before_lambda.len() - i - 1);
                    break;
                }
                paren_depth -= 1;
            }
            '[' | '{' => paren_depth = (paren_depth - 1).max(0),
            _ => {}
        }
    }

    let paren_pos = method_call_start?;
    let before_paren = before_lambda[..paren_pos].trim();

    let dot_pos = before_paren.rfind('.')?;
    let method_name = before_paren[dot_pos + 1..].trim();
    let receiver_expr = before_paren[..dot_pos].trim();

    let receiver_type = infer_method_chain_type(receiver_expr, local_vars, None)?;

    infer_lambda_param_type_for_method(&receiver_type, method_name)
}

/// Infer the type of a lambda parameter based on receiver type and method name.
/// e.g., `List[Int].map` → lambda param is `Int`.
pub fn infer_lambda_param_type_for_method(receiver_type: &str, method_name: &str) -> Option<String> {
    // List methods
    if receiver_type.starts_with("List") || receiver_type.starts_with('[') || receiver_type == "List" {
        let element_type = if receiver_type.starts_with("List[") {
            receiver_type.strip_prefix("List[")?.strip_suffix(']')?.to_string()
        } else if receiver_type.starts_with('[') && receiver_type.ends_with(']') {
            receiver_type[1..receiver_type.len()-1].to_string()
        } else {
            "Int".to_string()
        };

        match method_name {
            "map" | "filter" | "each" | "any" | "all" | "find" | "takeWhile" | "dropWhile" |
            "partition" | "span" | "sortBy" | "groupBy" | "count" => {
                return Some(element_type);
            }
            "fold" | "foldl" | "foldr" => {
                return Some(element_type);
            }
            "zipWith" => {
                return Some(element_type);
            }
            _ => {}
        }
    }

    // Option methods
    if receiver_type.starts_with("Option") || receiver_type == "Option" {
        let inner_type = if receiver_type.starts_with("Option[") && receiver_type.ends_with(']') {
            receiver_type[7..receiver_type.len()-1].to_string()
        } else if receiver_type.starts_with("Option ") {
            receiver_type.strip_prefix("Option ")?.to_string()
        } else {
            "a".to_string()
        };

        match method_name {
            "map" | "flatMap" | "filter" => return Some(inner_type),
            _ => {}
        }
    }

    // Result methods
    if receiver_type.starts_with("Result") || receiver_type == "Result" {
        let (ok_type, err_type) = if receiver_type.starts_with("Result[") && receiver_type.ends_with(']') {
            let inner = &receiver_type[7..receiver_type.len()-1];
            let mut depth = 0;
            let mut comma_pos = None;
            for (i, c) in inner.chars().enumerate() {
                match c {
                    '[' | '(' | '{' => depth += 1,
                    ']' | ')' | '}' => depth -= 1,
                    ',' if depth == 0 => {
                        comma_pos = Some(i);
                        break;
                    }
                    _ => {}
                }
            }
            if let Some(pos) = comma_pos {
                (inner[..pos].trim().to_string(), inner[pos+1..].trim().to_string())
            } else {
                ("a".to_string(), "e".to_string())
            }
        } else {
            ("a".to_string(), "e".to_string())
        };

        match method_name {
            "map" => return Some(ok_type),
            "mapErr" => return Some(err_type),
            _ => {}
        }
    }

    // Map methods
    if receiver_type.starts_with("Map") || receiver_type == "Map" {
        match method_name {
            "map" | "filter" | "each" => {
                return Some("(k, v)".to_string());
            }
            _ => {}
        }
    }

    // Set methods
    if receiver_type.starts_with("Set") || receiver_type == "Set" {
        let element_type = if receiver_type.starts_with("Set[") {
            receiver_type.strip_prefix("Set[")?.strip_suffix(']')?.to_string()
        } else {
            "a".to_string()
        };

        match method_name {
            "map" | "filter" | "each" | "any" | "all" => return Some(element_type),
            _ => {}
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract and infer the type of the first argument in a function call.
fn infer_first_arg_type(args_str: &str, bindings: &HashMap<String, String>) -> Option<String> {
    let trimmed = args_str.trim();
    if !trimmed.starts_with('(') {
        return None;
    }

    let inner = &trimmed[1..];
    let mut depth = 0;
    let mut end_pos = 0;

    for (i, c) in inner.chars().enumerate() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                if depth == 0 {
                    end_pos = i;
                    break;
                }
                depth -= 1;
            }
            ',' if depth == 0 => {
                end_pos = i;
                break;
            }
            _ => {}
        }
    }

    if end_pos == 0 {
        end_pos = inner.find(')').unwrap_or(inner.len());
    }

    let first_arg = inner[..end_pos].trim();

    if first_arg.is_empty() {
        return None;
    }

    if first_arg.chars().all(|c| c.is_ascii_digit() || c == '-') && !first_arg.is_empty() {
        return Some("Int".to_string());
    }
    if first_arg.contains('.') && first_arg.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-') {
        return Some("Float".to_string());
    }
    if first_arg.starts_with('"') {
        return Some("String".to_string());
    }
    if first_arg.starts_with('[') {
        return infer_list_type(first_arg);
    }

    if let Some(ty) = bindings.get(first_arg) {
        return Some(ty.clone());
    }

    None
}

/// Extract the receiver expression before a dot, handling brackets and parens.
pub fn extract_receiver_expression(text: &str) -> &str {
    let chars: Vec<char> = text.chars().collect();
    let mut i = chars.len();
    let mut depth = 0;
    let mut in_string = false;
    let mut string_char = '"';

    while i > 0 {
        i -= 1;
        let c = chars[i];

        if in_string {
            if c == string_char {
                let mut escapes = 0;
                let mut j = i;
                while j > 0 && chars[j - 1] == '\\' {
                    escapes += 1;
                    j -= 1;
                }
                if escapes % 2 == 0 {
                    in_string = false;
                }
            }
            continue;
        }

        match c {
            '"' | '\'' => {
                in_string = true;
                string_char = c;
            }
            ')' | ']' | '}' => depth += 1,
            '(' | '[' | '{' => {
                if depth > 0 {
                    depth -= 1;
                } else {
                    return &text[i..];
                }
            }
            _ if depth == 0 => {
                if !c.is_alphanumeric() && c != '_' && c != '.' {
                    return &text[i + 1..];
                }
            }
            _ => {}
        }
    }

    text
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_literal_type() {
        assert_eq!(detect_literal_type("\"hello\""), Some("String"));
        assert_eq!(detect_literal_type("[1,2,3]"), Some("List"));
        assert_eq!(detect_literal_type("%{a: 1}"), Some("Map"));
        assert_eq!(detect_literal_type("#{1,2}"), Some("Set"));
        assert_eq!(detect_literal_type("42"), Some("Int"));
        assert_eq!(detect_literal_type("3.14"), Some("Float"));
        assert_eq!(detect_literal_type("foo"), None);
    }

    #[test]
    fn test_infer_list_type() {
        assert_eq!(infer_list_type("[1,2,3]"), Some("List[Int]".to_string()));
        assert_eq!(infer_list_type("[\"a\",\"b\"]"), Some("List[String]".to_string()));
        assert_eq!(infer_list_type("[[1,2],[3,4]]"), Some("List[List[Int]]".to_string()));
        assert_eq!(infer_list_type("[]"), Some("List".to_string()));
    }

    #[test]
    fn test_infer_tuple_type() {
        assert_eq!(infer_tuple_type("(42, \"hello\")"), Some("(Int, String)".to_string()));
        assert_eq!(infer_tuple_type("(true, 1, 3.14)"), Some("(Bool, Int, Float)".to_string()));
    }

    #[test]
    fn test_infer_method_return_type() {
        assert_eq!(infer_method_return_type_static("List[Int]", "filter"), Some("List[Int]".to_string()));
        assert_eq!(infer_method_return_type_static("List[Int]", "head"), Some("Int".to_string()));
        assert_eq!(infer_method_return_type_static("String", "chars"), Some("List[Char]".to_string()));
        assert_eq!(infer_method_return_type_static("String", "length"), Some("Int".to_string()));
    }

    #[test]
    fn test_infer_lambda_param_type_for_method() {
        assert_eq!(infer_lambda_param_type_for_method("List[Int]", "map"), Some("Int".to_string()));
        assert_eq!(infer_lambda_param_type_for_method("List[String]", "filter"), Some("String".to_string()));
        assert_eq!(infer_lambda_param_type_for_method("Option[Int]", "map"), Some("Int".to_string()));
    }

    #[test]
    fn test_extract_local_bindings_simple() {
        let content = "x = 42\ny = \"hello\"\nserver = Server.bind(8080)";
        let bindings = extract_local_bindings(content, 10, None);
        assert_eq!(bindings.get("x"), Some(&"Int".to_string()));
        assert_eq!(bindings.get("y"), Some(&"String".to_string()));
    }

    #[test]
    fn test_extract_local_bindings_typed() {
        let content = "x: Int = 42\ny: String = \"hello\"";
        let bindings = extract_local_bindings(content, 10, None);
        assert_eq!(bindings.get("x"), Some(&"Int".to_string()));
        assert_eq!(bindings.get("y"), Some(&"String".to_string()));
    }

    #[test]
    fn test_infer_method_chain() {
        let mut vars = HashMap::new();
        vars.insert("nums".to_string(), "List[Int]".to_string());
        assert_eq!(infer_method_chain_type("nums.filter(x => x > 0)", &vars, None), Some("List[Int]".to_string()));
    }

    #[test]
    fn test_extract_receiver_expression() {
        assert_eq!(extract_receiver_expression("x"), "x");
        assert_eq!(extract_receiver_expression("[1,2,3]"), "[1,2,3]");
        assert_eq!(extract_receiver_expression("a + [1,2,3]"), "[1,2,3]");
    }
}
