//! Integration tests that run .nos test files.
//!
//! Test files should have a `# expect: <value>` comment at the top to specify
//! the expected result of running main().

use nostos_compiler::compile::compile_module;
use nostos_syntax::parse;
use nostos_vm::value::Value;
use nostos_vm::parallel::{ParallelVM, ParallelConfig};
use std::fs;
use std::path::Path;

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
        Value::String(s) => format!("\"{}\"", s),
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
            let fields_str: Vec<String> = rec.fields.iter().map(value_to_string).collect();
            format!("{}({})", rec.type_name, fields_str.join(", "))
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

/// Compile and run a Nostos source file using ParallelVM, returning a Value.
fn run_nos_source(source: &str) -> Result<Value, String> {
    // Parse
    let (module_opt, errors) = parse(source);
    if !errors.is_empty() {
        return Err(format!("Parse error: {:?}", errors));
    }
    let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;

    // Compile
    let compiler = compile_module(&module, source).map_err(|e| format!("Compile error: {:?}", e))?;

    // Create ParallelVM with single thread for deterministic tests
    let config = ParallelConfig {
        num_threads: 1,
        ..Default::default()
    };
    let mut vm = ParallelVM::new(config);
    vm.register_default_natives();

    for (name, func) in compiler.get_all_functions() {
        vm.register_function(name, func.clone());
    }
    vm.set_function_list(compiler.get_function_list());
    for (name, type_val) in compiler.get_vm_types() {
        vm.register_type(&name, type_val);
    }

    // Get main function
    let main_func = compiler.get_function("main")
        .ok_or_else(|| "No main function".to_string())?;

    // Run and convert result
    let result = vm.run(main_func)
        .map_err(|e| format!("Runtime error: {:?}", e))?;
    result.value
        .map(|v| v.to_value())
        .ok_or_else(|| "No result returned".to_string())
}

/// Compile and run a Nostos source file using ParallelVM (returns string display).
fn run_nos_source_gc(source: &str) -> Result<String, String> {
    // Parse
    let (module_opt, errors) = parse(source);
    if !errors.is_empty() {
        return Err(format!("Parse error: {:?}", errors));
    }
    let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;

    // Compile
    let compiler = compile_module(&module, source).map_err(|e| format!("Compile error: {:?}", e))?;

    // Create ParallelVM with single thread for deterministic tests
    let config = ParallelConfig {
        num_threads: 1,
        ..Default::default()
    };
    let mut vm = ParallelVM::new(config);
    vm.register_default_natives();

    for (name, func) in compiler.get_all_functions() {
        vm.register_function(name, func.clone());
    }
    vm.set_function_list(compiler.get_function_list());
    for (name, type_val) in compiler.get_vm_types() {
        vm.register_type(&name, type_val);
    }

    // Get main function
    let main_func = compiler.get_function("main")
        .ok_or_else(|| "No main function".to_string())?;

    // Run and get display string
    let result = vm.run(main_func)
        .map_err(|e| format!("Runtime error: {:?}", e))?;

    // Get output from println and return value
    let mut output = String::new();
    for line in &result.output {
        output.push_str(line);
        output.push('\n');
    }

    match result.value {
        Some(value) => Ok(output + &value.display()),
        None => Ok(output + "()"),
    }
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
