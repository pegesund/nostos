//! Nostos CLI - Command-line interface for running Nostos programs.

use nostos_compiler::compile::compile_module;
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::gc::GcValue;
use nostos_vm::runtime::Runtime;
use nostos_vm::VM;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: nostos [options] <file.nos> [args...]");
        eprintln!();
        eprintln!("Run a Nostos program file.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --runtime  Use the multi-process runtime instead of simple VM");
        eprintln!("  --help     Show this help message");
        eprintln!("  --version  Show version information");
        return ExitCode::FAILURE;
    }

    // Parse options
    let mut use_runtime = false;
    let mut file_idx = 1;

    for (i, arg) in args.iter().enumerate().skip(1) {
        if arg == "--runtime" {
            use_runtime = true;
            file_idx = i + 1;
        } else if arg.starts_with("--") || arg.starts_with("-") {
            // Handle other flags
            if arg == "--help" || arg == "-h" {
                println!("Usage: nostos [options] <file.nos> [args...]");
                println!();
                println!("Run a Nostos program file.");
                println!();
                println!("Options:");
                println!("  --runtime  Use the multi-process runtime instead of simple VM");
                println!("  --help     Show this help message");
                println!("  --version  Show version information");
                return ExitCode::SUCCESS;
            }
            if arg == "--version" || arg == "-v" {
                println!("nostos {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
        } else {
            file_idx = i;
            break;
        }
    }

    if file_idx >= args.len() {
        eprintln!("Error: No input file specified");
        return ExitCode::FAILURE;
    }

    let file_path = &args[file_idx];

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
        let source_errors = parse_errors_to_source_errors(&errors);
        eprint_errors(&source_errors, file_path, &source);
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
            let source_error = e.to_source_error();
            source_error.eprint(file_path, &source);
            return ExitCode::FAILURE;
        }
    };

    if use_runtime {
        // Use multi-process runtime
        let mut runtime = Runtime::new();

        for (name, func) in compiler.get_all_functions() {
            runtime.register_function(&name, func.clone());
        }
        // Set function list for direct indexed calls (CallDirect)
        runtime.set_function_list(compiler.get_function_list());
        for (name, type_val) in compiler.get_vm_types() {
            runtime.register_type(&name, type_val);
        }

        // Get main function
        let main_func = match compiler.get_all_functions().get("main") {
            Some(func) => func.clone(),
            None => {
                eprintln!("Error: No 'main' function found in '{}'", file_path);
                return ExitCode::FAILURE;
            }
        };

        runtime.spawn_initial(main_func);

        match runtime.run() {
            Ok(result) => {
                if let Some(val) = result {
                    if !matches!(val, GcValue::Unit) {
                        println!("{:?}", val);
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Runtime error: {}", e);
                ExitCode::FAILURE
            }
        }
    } else {
        // Use simple VM
        let mut vm = VM::new();
        for (name, func) in compiler.get_all_functions() {
            vm.functions.insert(name.clone(), func.clone());
        }
        // Populate function_list for direct indexed calls (CallDirect)
        vm.function_list = compiler.get_function_list();
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
}
