//! Integration tests that run .nos test files.
//!
//! Test files should have a `# expect: <value>` comment at the top to specify
//! the expected result of running main().

use nostos_compiler::compile::compile_module;
use nostos_syntax::parse;
use nostos_vm::{value::Value, VM};
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

/// Convert Value to string for comparison.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
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
                var.constructor.clone()
            } else {
                let fields_str: Vec<String> = var.fields.iter().map(value_to_string).collect();
                format!("{}({})", var.constructor, fields_str.join(", "))
            }
        }
        _ => format!("{:?}", value),
    }
}

/// Compile and run a Nostos source file, returning the result of main().
fn run_nos_source(source: &str) -> Result<Value, String> {
    // Parse
    let (module_opt, errors) = parse(source);
    if !errors.is_empty() {
        return Err(format!("Parse error: {:?}", errors));
    }
    let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;

    // Compile
    let compiler = compile_module(&module).map_err(|e| format!("Compile error: {:?}", e))?;

    // Create VM and load functions/types
    let mut vm = VM::new();
    for (name, func) in compiler.get_all_functions() {
        vm.functions.insert(name.clone(), func.clone());
    }
    for (name, type_val) in compiler.get_vm_types() {
        vm.types.insert(name, type_val);
    }

    // Run main
    if vm.functions.contains_key("main") {
        vm.call("main", vec![]).map_err(|e| format!("Runtime error: {:?}", e))
    } else {
        Err("No main function".to_string())
    }
}

/// Run a single test file.
fn run_test_file(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let expected = parse_expected(&source)
        .ok_or_else(|| format!("{}: Missing '# expect:' comment", path.display()))?;

    let result = run_nos_source(&source)?;
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
