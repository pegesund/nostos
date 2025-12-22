//! Integration tests that run .nos test files.
//!
//! Test files should have a `# expect: <value>` comment at the top to specify
//! the expected result of running main().

use nostos_compiler::compile::{compile_module, compile_module_with_stdlib, MvarInitValue};
use nostos_syntax::parse;
use nostos_vm::value::Value;
use nostos_vm::async_vm::{AsyncVM, AsyncConfig};
use nostos_vm::process::ThreadSafeValue;
use std::fs;
use std::path::Path;

/// Convert MvarInitValue to ThreadSafeValue for VM registration.
fn mvar_init_to_thread_safe(init: &MvarInitValue) -> ThreadSafeValue {
    match init {
        MvarInitValue::Unit => ThreadSafeValue::Unit,
        MvarInitValue::Bool(b) => ThreadSafeValue::Bool(*b),
        MvarInitValue::Int(n) => ThreadSafeValue::Int64(*n),
        MvarInitValue::Float(f) => ThreadSafeValue::Float64(*f),
        MvarInitValue::String(s) => ThreadSafeValue::String(s.clone()),
        MvarInitValue::Char(c) => ThreadSafeValue::Char(*c),
        MvarInitValue::EmptyList => ThreadSafeValue::List(vec![]),
        MvarInitValue::IntList(ints) => ThreadSafeValue::List(
            ints.iter().map(|n| ThreadSafeValue::Int64(*n)).collect()
        ),
        MvarInitValue::FloatList(floats) => ThreadSafeValue::List(
            floats.iter().map(|f| ThreadSafeValue::Float64(*f)).collect()
        ),
        MvarInitValue::BoolList(bools) => ThreadSafeValue::List(
            bools.iter().map(|b| ThreadSafeValue::Bool(*b)).collect()
        ),
        MvarInitValue::StringList(strings) => ThreadSafeValue::List(
            strings.iter().map(|s| ThreadSafeValue::String(s.clone())).collect()
        ),
        MvarInitValue::Tuple(items) => ThreadSafeValue::Tuple(
            items.iter().map(mvar_init_to_thread_safe).collect()
        ),
        MvarInitValue::List(items) => ThreadSafeValue::List(
            items.iter().map(mvar_init_to_thread_safe).collect()
        ),
        MvarInitValue::Record(type_name, fields) => {
            let field_names: Vec<String> = fields.iter().map(|(name, _)| name.clone()).collect();
            let values: Vec<ThreadSafeValue> = fields.iter()
                .map(|(_, val)| mvar_init_to_thread_safe(val))
                .collect();
            ThreadSafeValue::Record {
                type_name: type_name.clone(),
                field_names,
                fields: values,
                mutable_fields: vec![false; fields.len()],
            }
        }
        MvarInitValue::EmptyMap => {
            ThreadSafeValue::Map(nostos_vm::empty_shared_map())
        }
        MvarInitValue::Map(entries) => {
            let mut map = imbl::HashMap::new();
            for (k, v) in entries {
                if let Some(key) = mvar_init_to_shared_key(k) {
                    let value = mvar_init_to_shared_value(v);
                    map.insert(key, value);
                }
            }
            ThreadSafeValue::Map(std::sync::Arc::new(map))
        }
    }
}

fn mvar_init_to_shared_key(init: &MvarInitValue) -> Option<nostos_vm::SharedMapKey> {
    use nostos_vm::SharedMapKey;
    match init {
        MvarInitValue::Unit => Some(SharedMapKey::Unit),
        MvarInitValue::Bool(b) => Some(SharedMapKey::Bool(*b)),
        MvarInitValue::Int(n) => Some(SharedMapKey::Int64(*n)),
        MvarInitValue::String(s) => Some(SharedMapKey::String(s.clone())),
        MvarInitValue::Char(c) => Some(SharedMapKey::Char(*c)),
        _ => None,
    }
}

fn mvar_init_to_shared_value(init: &MvarInitValue) -> nostos_vm::SharedMapValue {
    use nostos_vm::SharedMapValue;
    match init {
        MvarInitValue::Unit => SharedMapValue::Unit,
        MvarInitValue::Bool(b) => SharedMapValue::Bool(*b),
        MvarInitValue::Int(n) => SharedMapValue::Int64(*n),
        MvarInitValue::Float(f) => SharedMapValue::Float64(*f),
        MvarInitValue::String(s) => SharedMapValue::String(s.clone()),
        MvarInitValue::Char(c) => SharedMapValue::Char(*c),
        MvarInitValue::EmptyList | MvarInitValue::IntList(_) | MvarInitValue::FloatList(_)
        | MvarInitValue::BoolList(_) | MvarInitValue::StringList(_) => {
            SharedMapValue::List(vec![]) // Simplified for tests
        }
        MvarInitValue::List(items) => SharedMapValue::List(
            items.iter().map(mvar_init_to_shared_value).collect()
        ),
        MvarInitValue::Tuple(items) => SharedMapValue::Tuple(
            items.iter().map(mvar_init_to_shared_value).collect()
        ),
        MvarInitValue::Record(type_name, fields) => {
            let field_names: Vec<String> = fields.iter().map(|(name, _)| name.clone()).collect();
            let values: Vec<SharedMapValue> = fields.iter()
                .map(|(_, val)| mvar_init_to_shared_value(val))
                .collect();
            SharedMapValue::Record { type_name: type_name.clone(), field_names, fields: values }
        }
        MvarInitValue::EmptyMap => SharedMapValue::Map(nostos_vm::empty_shared_map()),
        MvarInitValue::Map(entries) => {
            let mut map = imbl::HashMap::new();
            for (k, v) in entries {
                if let Some(key) = mvar_init_to_shared_key(k) {
                    map.insert(key, mvar_init_to_shared_value(v));
                }
            }
            SharedMapValue::Map(std::sync::Arc::new(map))
        }
    }
}

/// Parse expected value from test file comments.
/// Looks for `# expect: <value>` line.
fn parse_expected(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# expect:") {
            return Some(trimmed["# expect:".len()..].trim().to_string());
        }
    }
    None
}

/// Parse all expected error strings from test file comments.
/// Looks for all `# expect_error: <value>` lines.
fn parse_expected_errors(source: &str) -> Vec<String> {
    let mut errors = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# expect_error:") {
            errors.push(trimmed["# expect_error:".len()..].trim().to_string());
        }
    }
    errors
}

/// Convert Value to string for comparison.
fn value_to_string(value: &Value) -> String {
    match value {
        // Signed integers
        Value::Int8(n) => format!("{}i8", n),
        Value::Int16(n) => format!("{}i16", n),
        Value::Int32(n) => format!("{}i32", n),
        Value::Int64(n) => n.to_string(),
        // Unsigned integers
        Value::UInt8(n) => format!("{}u8", n),
        Value::UInt16(n) => format!("{}u16", n),
        Value::UInt32(n) => format!("{}u32", n),
        Value::UInt64(n) => format!("{}u64", n),
        // Floats
        Value::Float32(f) => format!("{}f32", f),
        Value::Float64(f) => f.to_string(),
        // BigInt and Decimal
        Value::BigInt(n) => format!("{}n", n),
        Value::Decimal(d) => format!("{}d", d),
        // Other types
        Value::Bool(b) => b.to_string(),
        Value::String(s) => s.to_string(),
        Value::Char(c) => format!("'{}'", c),
        Value::Unit => "()".to_string(),
        Value::List(items) => {
            let items_str: Vec<String> = items.iter().map(value_to_string).collect();
            format!("[{}]", items_str.join(", "))
        }
        Value::Tuple(items) => {
            let items_str: Vec<String> = items.iter().map(value_to_string).collect();
            format!("({})", items_str.join(", "))
        }
        Value::Record(rec) => {
            // Format as TypeName{field1: val1, field2: val2}
            let fields_str: Vec<String> = rec.field_names.iter()
                .zip(rec.fields.iter())
                .map(|(name, val)| format!("{}: {}", name, value_to_string(val)))
                .collect();
            format!("{}{{{}}}", rec.type_name, fields_str.join(", "))
        }
        Value::Variant(var) => {
            if var.fields.is_empty() {
                var.constructor.to_string()
            } else {
                let fields_str: Vec<String> = var.fields.iter().map(value_to_string).collect();
                format!("{}({})", var.constructor, fields_str.join(", "))
            }
        }
        _ => format!("{:?}", value),
    }
}

/// Find the stdlib directory relative to the workspace root.
fn find_stdlib_path() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
    workspace_root.join("stdlib")
}

/// Compile and run a Nostos source file using AsyncVM, returning a Value.
fn run_nos_source(source: &str) -> Result<Value, String> {
    // Parse
    let (module_opt, errors) = parse(source);
    if !errors.is_empty() {
        return Err(format!("Parse error: {:?}", errors));
    }
    let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;

    // Compile with stdlib
    let stdlib_path = find_stdlib_path();
    let compiler = compile_module_with_stdlib(&module, source, &stdlib_path).map_err(|e| format!("Compile error: {:?}", e))?;

    // Create AsyncVM
    let config = AsyncConfig::default();
    let mut vm = AsyncVM::new(config);
    vm.register_default_natives();

    for (name, func) in compiler.get_all_functions().iter() {
        vm.register_function(name, func.clone());
    }
    vm.set_function_list(compiler.get_function_list());
    for (name, type_val) in compiler.get_vm_types().iter() {
        vm.register_type(name, type_val.clone());
    }

    // Register mvars (module-level mutable variables)
    for (name, info) in compiler.get_mvars() {
        let initial_value = mvar_init_to_thread_safe(&info.initial_value);
        vm.register_mvar(name, initial_value);
    }

    // Run and convert result (function names include signature suffix, e.g. "main/")
    let result = vm.run("main/")
        .map_err(|e| format!("Runtime error: {:?}", e))?;

    Ok(result.to_value())
}

/// Compile and run a Nostos source file using AsyncVM (returns string display).
fn run_nos_source_gc(source: &str) -> Result<String, String> {
    // Parse
    let (module_opt, errors) = parse(source);
    if !errors.is_empty() {
        return Err(format!("Parse error: {:?}", errors));
    }
    let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;

    // Compile with stdlib
    let stdlib_path = find_stdlib_path();
    let compiler = compile_module_with_stdlib(&module, source, &stdlib_path).map_err(|e| format!("Compile error: {:?}", e))?;

    // Create AsyncVM
    let config = AsyncConfig::default();
    let mut vm = AsyncVM::new(config);
    vm.register_default_natives();

    for (name, func) in compiler.get_all_functions().iter() {
        vm.register_function(name, func.clone());
    }
    vm.set_function_list(compiler.get_function_list());
    for (name, type_val) in compiler.get_vm_types().iter() {
        vm.register_type(name, type_val.clone());
    }

    // Register mvars (module-level mutable variables)
    for (name, info) in compiler.get_mvars() {
        let initial_value = mvar_init_to_thread_safe(&info.initial_value);
        vm.register_mvar(name, initial_value);
    }

    // Run and get display string (function names include signature suffix, e.g. "main/")
    let result = vm.run("main/")
        .map_err(|e| format!("Runtime error: {:?}", e))?;

    Ok(result.display())
}

/// Run a single test file.
fn run_test_file(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    // Check if this is an error test (uses # expect_error:)
    let expected_errors = parse_expected_errors(&source);
    if !expected_errors.is_empty() {
        // This test expects an error
        match run_nos_source(&source) {
            Ok(result) => {
                return Err(format!(
                    "{}: Expected error containing {:?}, but got success: {}",
                    path.display(),
                    expected_errors,
                    value_to_string(&result)
                ));
            }
            Err(error_msg) => {
                // Check that all expected strings are in the error
                for expected in &expected_errors {
                    if !error_msg.contains(expected) {
                        return Err(format!(
                            "{}: Error missing expected string '{}'. Got: {}",
                            path.display(),
                            expected,
                            error_msg
                        ));
                    }
                }
                return Ok(());
            }
        }
    }

    let expected = parse_expected(&source)
        .ok_or_else(|| format!("{}: Missing '# expect:' or '# expect_error:' comment", path.display()))?;

    let result = run_nos_source(&source)
        .map_err(|e| format!("{}: {}", path.display(), e))?;
    let actual = value_to_string(&result);

    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{}: Expected {}, got {}",
            path.display(),
            expected,
            actual
        ))
    }
}

/// Run a single test file for concurrency tests (checks Pid output).
fn run_test_file_concurrent(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let expected = parse_expected(&source)
        .ok_or_else(|| format!("{}: Missing '# expect:' comment", path.display()))?;

    let actual = run_nos_source_gc(&source)?;

    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{}: Expected {}, got {}",
            path.display(),
            expected,
            actual
        ))
    }
}

/// Find all .nos files in a directory recursively.
fn find_nos_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_nos_files(&path));
            } else if path.extension().map(|e| e == "nos").unwrap_or(false) {
                files.push(path);
            }
        }
    }
    files
}

#[test]
#[ignore] // This test hangs - use individual category tests instead (see CLAUDE.md)
fn test_all_nos_files() {
    // Find the tests directory relative to the workspace root
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
    let tests_dir = workspace_root.join("tests");

    if !tests_dir.exists() {
        panic!("Tests directory not found: {}", tests_dir.display());
    }

    let files = find_nos_files(&tests_dir);
    if files.is_empty() {
        panic!("No .nos test files found in {}", tests_dir.display());
    }

    let mut failures = Vec::new();
    let mut passed = 0;

    for file in &files {
        // Skip concurrency tests - they're tested separately with the concurrent runner
        if file.to_str().map(|s| s.contains("/concurrency/")).unwrap_or(false) {
            println!("SKIP: {} (uses concurrent runner)", file.display());
            continue;
        }

        match run_test_file(file) {
            Ok(()) => {
                passed += 1;
                println!("PASS: {}", file.display());
            }
            Err(e) => {
                failures.push(e.clone());
                println!("FAIL: {}", e);
            }
        }
    }

    println!("\n{} passed, {} failed", passed, failures.len());

    if !failures.is_empty() {
        panic!("Test failures:\n{}", failures.join("\n"));
    }
}

// Individual test modules for each category
mod basics {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("basics").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn literal_int() { run_category_test("literal_int"); }

    #[test]
    fn literal_bool() { run_category_test("literal_bool"); }

    #[test]
    fn literal_string() { run_category_test("literal_string"); }

    #[test]
    fn literal_float() { run_category_test("literal_float"); }

    #[test]
    fn literal_char() { run_category_test("literal_char"); }

    #[test]
    fn literal_unit() { run_category_test("literal_unit"); }

    #[test]
    fn literal_hex() { run_category_test("literal_hex"); }

    #[test]
    fn literal_binary() { run_category_test("literal_binary"); }

    #[test]
    fn underscore_in_numbers() { run_category_test("underscore_in_numbers"); }
}

mod arithmetic {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("arithmetic").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn add() { run_category_test("add"); }

    #[test]
    fn mul() { run_category_test("mul"); }

    #[test]
    fn precedence() { run_category_test("precedence"); }

    #[test]
    fn sub() { run_category_test("sub"); }

    #[test]
    fn div() { run_category_test("div"); }

    #[test]
    fn modulo() { run_category_test("modulo"); }

    #[test]
    fn power() { run_category_test("power"); }

    #[test]
    fn negation() { run_category_test("negation"); }

    #[test]
    fn float_add() { run_category_test("float_add"); }

    #[test]
    fn float_mul() { run_category_test("float_mul"); }

    #[test]
    fn complex_expr() { run_category_test("complex_expr"); }
}

mod functions {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("functions").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn simple_call() { run_category_test("simple_call"); }

    #[test]
    fn recursion() { run_category_test("recursion"); }

    #[test]
    fn higher_order() { run_category_test("higher_order"); }

    #[test]
    fn lambda() { run_category_test("lambda"); }

    #[test]
    fn lambda_multi_arg() { run_category_test("lambda_multi_arg"); }

    #[test]
    fn closure() { run_category_test("closure"); }

    #[test]
    fn tail_recursion() { run_category_test("tail_recursion"); }

    #[test]
    fn multi_clause() { run_category_test("multi_clause"); }

    #[test]
    fn guards() { run_category_test("guards"); }

    #[test]
    fn mutual_recursion() { run_category_test("mutual_recursion"); }

    #[test]
    fn fibonacci() { run_category_test("fibonacci"); }
}

mod types {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("types").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn record() { run_category_test("record"); }

    #[test]
    fn variant() { run_category_test("variant"); }

    #[test]
    fn record_field() { run_category_test("record_field"); }

    #[test]
    fn variant_match() { run_category_test("variant_match"); }

    #[test]
    fn result() { run_category_test("result"); }

    #[test]
    fn recursive_list() { run_category_test("recursive_list"); }
}

mod patterns {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("patterns").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn match_int() { run_category_test("match_int"); }

    #[test]
    fn list() { run_category_test("list"); }

    #[test]
    fn tuple() { run_category_test("tuple"); }

    #[test]
    fn wildcard() { run_category_test("wildcard"); }

    #[test]
    fn nested() { run_category_test("nested"); }

    #[test]
    fn variant() { run_category_test("variant"); }

    #[test]
    fn list_head_tail() { run_category_test("list_head_tail"); }

    #[test]
    fn list_recursive() { run_category_test("list_recursive"); }

    #[test]
    fn comprehensive_map_set() { run_category_test("comprehensive_map_set"); }

    #[test]
    fn map_advanced() { run_category_test("map_advanced"); }

    #[test]
    fn function_args_map_set() { run_category_test("function_args_map_set"); }
}

mod traits {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("traits").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn basic() { run_category_test("basic"); }

    #[test]
    fn record_trait() { run_category_test("record_trait"); }

    #[test]
    fn multiple_impls() { run_category_test("multiple_impls"); }
}

mod comparison {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("comparison").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn equal() { run_category_test("equal"); }

    #[test]
    fn not_equal() { run_category_test("not_equal"); }

    #[test]
    fn less_than() { run_category_test("less_than"); }

    #[test]
    fn greater_than() { run_category_test("greater_than"); }

    #[test]
    fn less_equal() { run_category_test("less_equal"); }

    #[test]
    fn greater_equal() { run_category_test("greater_equal"); }
}

mod logical {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("logical").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn and() { run_category_test("and"); }

    #[test]
    fn or() { run_category_test("or"); }

    #[test]
    fn not() { run_category_test("not"); }

    #[test]
    fn complex() { run_category_test("complex"); }
}

mod control_flow {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("control_flow").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn if_else() { run_category_test("if_else"); }

    #[test]
    fn nested_if() { run_category_test("nested_if"); }

    #[test]
    fn match_simple() { run_category_test("match_simple"); }

    #[test]
    fn match_guard() { run_category_test("match_guard"); }

    #[test]
    fn block() { run_category_test("block"); }

    #[test]
    fn nested_block() { run_category_test("nested_block"); }
}

mod strings {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("strings").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn concat() { run_category_test("concat"); }

    #[test]
    fn interpolation() { run_category_test("interpolation"); }

    #[test]
    fn escape() { run_category_test("escape"); }
}

mod collections {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("collections").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn tuple() { run_category_test("tuple"); }

    #[test]
    fn list_literal() { run_category_test("list_literal"); }

    #[test]
    fn list_empty() { run_category_test("list_empty"); }

    #[test]
    fn list_cons() { run_category_test("list_cons"); }
}

mod concurrency {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("concurrency").join(format!("{}.nos", name));

        // Use concurrent runner for concurrency tests (returns Pid format)
        if let Err(e) = run_test_file_concurrent(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn self_pid() { run_category_test("self_pid"); }

    #[test]
    fn spawn_simple() { run_category_test("spawn_simple"); }

    #[test]
    fn spawn_compute() { run_category_test("spawn_compute"); }

    #[test]
    fn message_tuple() { run_category_test("message_tuple"); }

    #[test]
    fn message_variant() { run_category_test("message_variant"); }

    #[test]
    fn multiple_spawns() { run_category_test("multiple_spawns"); }
}

/// Tests for source code display (multi-clause functions)
mod source_display {
    use super::*;

    #[test]
    fn multi_clause_source_includes_all_clauses() {
        let source = r#"
# Multi-clause function
filter(_, []) = []
filter(pred, [x | xs]) = if pred(x) then [x | filter(pred, xs)] else filter(pred, xs)

main() = filter(x => x > 2, [1,2,3,4,5])
"#;

        let (module_opt, errors) = parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let module = module_opt.expect("No module");

        let compiler = compile_module(&module, source).expect("Compile failed");

        // Get source for filter - should include BOTH clauses
        let filter_source = compiler.get_all_function_sources("filter")
            .expect("No source found for filter");

        println!("=== Filter source ===\n{}", filter_source);

        // Verify both clauses are present
        assert!(filter_source.contains("filter(_, [])"),
            "Missing first clause (empty list base case). Source:\n{}", filter_source);
        assert!(filter_source.contains("filter(pred, [x | xs])"),
            "Missing second clause (recursive case). Source:\n{}", filter_source);
    }

    #[test]
    fn multiple_clauses_with_different_patterns() {
        let source = r#"
# Multiple clauses with different patterns
describe([]) = "empty"
describe([x]) = "single"
describe([x, y | rest]) = "many"

main() = describe([1,2,3])
"#;

        let (module_opt, errors) = parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let module = module_opt.expect("No module");

        let compiler = compile_module(&module, source).expect("Compile failed");

        let describe_source = compiler.get_all_function_sources("describe")
            .expect("No source found for describe");

        println!("=== Describe source ===\n{}", describe_source);

        // Verify all three clauses are present
        assert!(describe_source.contains("describe([])"),
            "Missing empty list clause. Source:\n{}", describe_source);
        assert!(describe_source.contains("describe([x])"),
            "Missing single element clause. Source:\n{}", describe_source);
        assert!(describe_source.contains("describe([x, y | rest])"),
            "Missing multiple elements clause. Source:\n{}", describe_source);
    }
}

mod file_io {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("file_io").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn read_write_basic() { run_category_test("read_write_basic"); }

    #[test]
    fn read_write_unicode() { run_category_test("read_write_unicode"); }

    #[test]
    fn read_write_multiline() { run_category_test("read_write_multiline"); }

    #[test]
    fn read_write_empty() { run_category_test("read_write_empty"); }

    #[test]
    fn read_nonexistent() { run_category_test("read_nonexistent"); }

    #[test]
    fn append_basic() { run_category_test("append_basic"); }

    #[test]
    fn append_multiple() { run_category_test("append_multiple"); }

    #[test]
    fn append_creates_file() { run_category_test("append_creates_file"); }

    #[test]
    fn dir_create() { run_category_test("dir_create"); }

    #[test]
    fn dir_create_all() { run_category_test("dir_create_all"); }

    #[test]
    fn dir_list() { run_category_test("dir_list"); }

    #[test]
    fn dir_remove() { run_category_test("dir_remove"); }

    #[test]
    fn dir_remove_all() { run_category_test("dir_remove_all"); }

    #[test]
    fn dir_exists() { run_category_test("dir_exists"); }

    #[test]
    fn file_exists() { run_category_test("file_exists"); }

    #[test]
    fn file_size() { run_category_test("file_size"); }

    #[test]
    fn file_copy() { run_category_test("file_copy"); }

    #[test]
    fn file_rename() { run_category_test("file_rename"); }

    #[test]
    fn file_remove() { run_category_test("file_remove"); }

    #[test]
    fn handle_write_read() { run_category_test("handle_write_read"); }

    #[test]
    fn handle_read() { run_category_test("handle_read"); }

    #[test]
    fn handle_readline() { run_category_test("handle_readline"); }

    #[test]
    fn handle_seek() { run_category_test("handle_seek"); }

    #[test]
    fn handle_append_mode() { run_category_test("handle_append_mode"); }

    #[test]
    fn handle_flush() { run_category_test("handle_flush"); }
}

mod json {
    use super::*;

    fn run_category_test(name: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
        let file = workspace_root.join("tests").join("json").join(format!("{}.nos", name));

        if let Err(e) = run_test_file(&file) {
            panic!("{}", e);
        }
    }

    #[test]
    fn json_to_type_all_primitives() { run_category_test("json_to_type_all_primitives"); }

    #[test]
    fn json_to_type_deeply_nested() { run_category_test("json_to_type_deeply_nested"); }

    #[test]
    fn json_to_type_from_jsonStringify() { run_category_test("json_to_type_from_jsonStringify"); }

    #[test]
    fn json_to_type_large_int() { run_category_test("json_to_type_large_int"); }

    #[test]
    fn json_to_type_many_variants() { run_category_test("json_to_type_many_variants"); }

    #[test]
    fn json_to_type_negative_float() { run_category_test("json_to_type_negative_float"); }

    #[test]
    fn json_to_type_nested_record() { run_category_test("json_to_type_nested_record"); }

    #[test]
    fn json_to_type_record_bool() { run_category_test("json_to_type_record_bool"); }

    #[test]
    fn json_to_type_record_empty_string() { run_category_test("json_to_type_record_empty_string"); }

    #[test]
    fn json_to_type_record_false() { run_category_test("json_to_type_record_false"); }

    #[test]
    fn json_to_type_record_float() { run_category_test("json_to_type_record_float"); }

    #[test]
    fn json_to_type_record_in_variant() { run_category_test("json_to_type_record_in_variant"); }

    #[test]
    fn json_to_type_record_multiple_fields() { run_category_test("json_to_type_record_multiple_fields"); }

    #[test]
    fn json_to_type_record_negative_int() { run_category_test("json_to_type_record_negative_int"); }

    #[test]
    fn json_to_type_record_simple() { run_category_test("json_to_type_record_simple"); }

    #[test]
    fn json_to_type_record_string() { run_category_test("json_to_type_record_string"); }

    #[test]
    fn json_to_type_record_zero() { run_category_test("json_to_type_record_zero"); }

    #[test]
    fn json_to_type_roundtrip_record() { run_category_test("json_to_type_roundtrip_record"); }

    #[test]
    fn json_to_type_roundtrip_variant() { run_category_test("json_to_type_roundtrip_variant"); }

    #[test]
    fn json_to_type_special_chars() { run_category_test("json_to_type_special_chars"); }

    #[test]
    fn json_to_type_unicode_string() { run_category_test("json_to_type_unicode_string"); }

    #[test]
    fn json_to_type_variant_bool_payload() { run_category_test("json_to_type_variant_bool_payload"); }

    #[test]
    fn json_to_type_variant_error() { run_category_test("json_to_type_variant_error"); }

    #[test]
    fn json_to_type_variant_extract_value() { run_category_test("json_to_type_variant_extract_value"); }

    #[test]
    fn json_to_type_variant_float_payload() { run_category_test("json_to_type_variant_float_payload"); }

    #[test]
    fn json_to_type_variant_in_record() { run_category_test("json_to_type_variant_in_record"); }

    #[test]
    fn json_to_type_variant_mixed() { run_category_test("json_to_type_variant_mixed"); }

    #[test]
    fn json_to_type_variant_multi_field() { run_category_test("json_to_type_variant_multi_field"); }

    #[test]
    fn json_to_type_variant_single_field() { run_category_test("json_to_type_variant_single_field"); }

    #[test]
    fn json_to_type_variant_some() { run_category_test("json_to_type_variant_some"); }

    #[test]
    fn json_to_type_variant_string_payload() { run_category_test("json_to_type_variant_string_payload"); }

    #[test]
    fn json_to_type_variant_three_cases() { run_category_test("json_to_type_variant_three_cases"); }

    #[test]
    fn json_to_type_variant_three_fields() { run_category_test("json_to_type_variant_three_fields"); }

    #[test]
    fn json_to_type_variant_unit() { run_category_test("json_to_type_variant_unit"); }

    #[test]
    fn json_to_type_int8() { run_category_test("json_to_type_int8"); }

    #[test]
    fn json_to_type_uint8() { run_category_test("json_to_type_uint8"); }

    #[test]
    fn json_to_type_uint16() { run_category_test("json_to_type_uint16"); }

    #[test]
    fn json_to_type_uint32() { run_category_test("json_to_type_uint32"); }

    #[test]
    fn json_to_type_uint64() { run_category_test("json_to_type_uint64"); }

    #[test]
    fn json_to_type_tuple() { run_category_test("json_to_type_tuple"); }

    #[test]
    fn json_to_type_tuple_three() { run_category_test("json_to_type_tuple_three"); }

    #[test]
    fn json_to_type_error_missing_field() { run_category_test("json_to_type_error_missing_field"); }

    #[test]
    fn json_to_type_error_unknown_constructor() { run_category_test("json_to_type_error_unknown_constructor"); }

    #[test]
    fn json_to_type_error_unknown_type() { run_category_test("json_to_type_error_unknown_type"); }

    #[test]
    fn json_to_type_error_variant_missing_field() { run_category_test("json_to_type_error_variant_missing_field"); }

    #[test]
    fn json_to_type_error_variant_multiple_keys() { run_category_test("json_to_type_error_variant_multiple_keys"); }

    #[test]
    fn json_to_type_error_wrong_json_type() { run_category_test("json_to_type_error_wrong_json_type"); }
}
