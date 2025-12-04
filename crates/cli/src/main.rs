//! Nostos CLI - Command-line interface for running Nostos programs.

use nostos_compiler::compile::compile_module;
use nostos_jit::{JitCompiler, JitConfig};
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::gc::GcValue;
use nostos_vm::runtime::Runtime;
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
        eprintln!("  --help     Show this help message");
        eprintln!("  --version  Show version information");
        return ExitCode::FAILURE;
    }

    // Parse options
    let mut file_idx = 1;

    for (i, arg) in args.iter().enumerate().skip(1) {
        if arg.starts_with("--") || arg.starts_with("-") {
            if arg == "--help" || arg == "-h" {
                println!("Usage: nostos [options] <file.nos> [args...]");
                println!();
                println!("Run a Nostos program file.");
                println!();
                println!("Options:");
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

    // Create runtime and load functions/types
    let mut runtime = Runtime::new();

    for (name, func) in compiler.get_all_functions() {
        runtime.register_function(&name, func.clone());
    }
    let function_list = compiler.get_function_list();
    runtime.set_function_list(function_list.clone());
    for (name, type_val) in compiler.get_vm_types() {
        runtime.register_type(&name, type_val);
    }

    // JIT compile suitable functions
    if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
        for idx in 0..function_list.len() {
            // Queue all functions for JIT compilation
            jit.queue_compilation(idx as u16);
        }
        // Process the queue
        if let Ok(compiled) = jit.process_queue(&function_list) {
            if compiled > 0 {
                // Register JIT functions with the runtime
                for (idx, _func) in function_list.iter().enumerate() {
                    if let Some(jit_fn) = jit.get_int_function(idx as u16) {
                        runtime.register_jit_int_function(idx as u16, jit_fn);
                    }
                }
            }
        }
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
}
