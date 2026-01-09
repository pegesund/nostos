//! Nostos CLI - Command-line interface for running Nostos programs.

use nostos_compiler::compile::compile_module;
use nostos_syntax::parse;
use nostos_vm::VM;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: nostos <file.nos> [args...]");
        eprintln!();
        eprintln!("Run a Nostos program file.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --help     Show this help message");
        eprintln!("  --version  Show version information");
        return ExitCode::FAILURE;
    }

    let file_path = &args[1];

    if file_path == "--help" || file_path == "-h" {
        println!("Usage: nostos <file.nos> [args...]");
        println!();
        println!("Run a Nostos program file.");
        println!();
        println!("Options:");
        println!("  --help     Show this help message");
        println!("  --version  Show version information");
        return ExitCode::SUCCESS;
    }

    if file_path == "--version" || file_path == "-v" {
        println!("nostos {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    // Read the source file
    let source = match fs::read_to_string(file_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", file_path, e);
            return ExitCode::FAILURE;
        }
    };

    // Parse
    let (module_opt, errors) = parse(&source);
    if !errors.is_empty() {
        eprintln!("Parse errors in '{}':", file_path);
        for error in &errors {
            eprintln!("  {:?}", error);
        }
        return ExitCode::FAILURE;
    }

    let module = match module_opt {
        Some(m) => m,
        None => {
            eprintln!("Failed to parse '{}'", file_path);
            return ExitCode::FAILURE;
        }
    };

    // Compile
    let compiler = match compile_module(&module) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Compile error in '{}': {:?}", file_path, e);
            return ExitCode::FAILURE;
        }
    };

    // Create VM and load functions/types
    let mut vm = VM::new();
    for (name, func) in compiler.get_all_functions() {
        vm.functions.insert(name.clone(), func.clone());
    }
    for (name, type_val) in compiler.get_vm_types() {
        vm.types.insert(name, type_val);
    }

    // Run main
    if !vm.functions.contains_key("main") {
        eprintln!("Error: No 'main' function found in '{}'", file_path);
        return ExitCode::FAILURE;
    }

    match vm.call("main", vec![]) {
        Ok(result) => {
            // Only print the result if it's not Unit
            if !matches!(result, nostos_vm::value::Value::Unit) {
                println!("{}", result);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            ExitCode::FAILURE
        }
    }
}
