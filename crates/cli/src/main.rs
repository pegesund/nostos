//! Nostos CLI - Command-line interface for running Nostos programs.

use nostos_compiler::compile::compile_module;
use nostos_jit::{JitCompiler, JitConfig};
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::gc::GcValue;
use nostos_vm::runtime::Runtime;
use nostos_vm::scheduler::Scheduler;
use nostos_vm::worker::{WorkerPool, WorkerPoolConfig};
use nostos_vm::parallel::{ParallelVM, ParallelConfig};
use nostos_vm::value::RuntimeError;
use std::env;
use std::fs;
use std::process::ExitCode;

/// Format a runtime error as JSON for debugger integration.
fn format_error_json(error: &RuntimeError, file_path: &str) -> String {
    // Extract the error type, message, and stack trace
    let (error_type, message, stack_trace) = extract_error_info(error);

    // Build JSON manually (avoid adding serde dependency just for this)
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"file\": {},\n", json_string(file_path)));
    json.push_str(&format!("  \"error_type\": {},\n", json_string(&error_type)));
    json.push_str(&format!("  \"message\": {},\n", json_string(&message)));
    json.push_str("  \"stack_trace\": [\n");

    for (i, frame) in stack_trace.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"function\": {},\n", json_string(&frame.0)));
        json.push_str(&format!("      \"line\": {}\n", frame.1));
        json.push_str("    }");
        if i < stack_trace.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }

    json.push_str("  ]\n");
    json.push_str("}");
    json
}

/// Extract error type, message, and stack frames from a RuntimeError.
fn extract_error_info(error: &RuntimeError) -> (String, String, Vec<(String, usize)>) {
    match error {
        RuntimeError::WithStackTrace { error: inner, stack_trace } => {
            let (error_type, message) = get_error_type_and_message(inner);
            let frames = parse_stack_trace(stack_trace);
            (error_type, message, frames)
        }
        _ => {
            let (error_type, message) = get_error_type_and_message(error);
            (error_type, message, vec![])
        }
    }
}

/// Get the error type name and message from a RuntimeError.
fn get_error_type_and_message(error: &RuntimeError) -> (String, String) {
    match error {
        RuntimeError::TypeError { expected, found } => {
            ("TypeError".to_string(), format!("expected {}, got {}", expected, found))
        }
        RuntimeError::DivisionByZero => {
            ("DivisionByZero".to_string(), "Division by zero".to_string())
        }
        RuntimeError::IndexOutOfBounds { index, length } => {
            ("IndexOutOfBounds".to_string(), format!("Index {} out of bounds (length {})", index, length))
        }
        RuntimeError::UnknownField { type_name, field } => {
            ("UnknownField".to_string(), format!("Unknown field '{}' on type '{}'", field, type_name))
        }
        RuntimeError::ImmutableField { field } => {
            ("ImmutableField".to_string(), format!("Cannot mutate immutable field '{}'", field))
        }
        RuntimeError::ImmutableBinding { name } => {
            ("ImmutableBinding".to_string(), format!("Cannot mutate immutable binding '{}'", name))
        }
        RuntimeError::UnknownVariable(name) => {
            ("UnknownVariable".to_string(), format!("Unknown variable '{}'", name))
        }
        RuntimeError::UnknownFunction(name) => {
            ("UnknownFunction".to_string(), format!("Unknown function '{}'", name))
        }
        RuntimeError::ArityMismatch { expected, found } => {
            ("ArityMismatch".to_string(), format!("Expected {} arguments, got {}", expected, found))
        }
        RuntimeError::MatchFailed => {
            ("MatchFailed".to_string(), "Pattern match failed".to_string())
        }
        RuntimeError::AssertionFailed(msg) => {
            ("AssertionFailed".to_string(), msg.clone())
        }
        RuntimeError::Panic(msg) => {
            ("Panic".to_string(), msg.clone())
        }
        RuntimeError::StackOverflow => {
            ("StackOverflow".to_string(), "Stack overflow".to_string())
        }
        RuntimeError::ProcessNotFound(pid) => {
            ("ProcessNotFound".to_string(), format!("Process not found: {:?}", pid))
        }
        RuntimeError::Timeout => {
            ("Timeout".to_string(), "Timeout".to_string())
        }
        RuntimeError::IOError(msg) => {
            ("IOError".to_string(), msg.clone())
        }
        RuntimeError::WithStackTrace { error, .. } => {
            get_error_type_and_message(error)
        }
    }
}

/// Parse a formatted stack trace string into (function_name, line_number) pairs.
fn parse_stack_trace(trace: &str) -> Vec<(String, usize)> {
    let mut frames = Vec::new();
    for line in trace.lines() {
        let line = line.trim();
        // Format: "1. function_name (line N)" or "1. function_name"
        if let Some(rest) = line.strip_prefix(|c: char| c.is_ascii_digit()) {
            let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
            let rest = rest.trim();

            if let Some(paren_pos) = rest.find(" (line ") {
                let func_name = rest[..paren_pos].to_string();
                let line_part = &rest[paren_pos + 7..];
                if let Some(end_paren) = line_part.find(')') {
                    if let Ok(line_num) = line_part[..end_paren].parse::<usize>() {
                        frames.push((func_name, line_num));
                        continue;
                    }
                }
            }
            // No line number
            frames.push((rest.to_string(), 0));
        }
    }
    frames
}

/// Escape a string for JSON output.
fn json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

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
    let mut enable_jit = true;
    let mut json_errors = false;
    let mut debug_mode = false;
    let mut parallel_workers: Option<usize> = None;
    let mut parallel_affinity: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") || arg.starts_with("-") {
            if arg == "--help" || arg == "-h" {
                println!("Usage: nostos [options] <file.nos> [args...]");
                println!();
                println!("Run a Nostos program file.");
                println!();
                println!("Options:");
                println!("  --help           Show this help message");
                println!("  --version        Show version information");
                println!("  --no-jit         Disable JIT compilation (for debugging)");
                println!("  --debug          Show local variable values in stack traces");
                println!("  --json-errors    Output errors as JSON (for debugger integration)");
                println!("  --parallel [N]   Enable parallel execution with N workers (default: all CPUs)");
                println!("  --parallel-affinity [N]  Parallel with CPU affinity (processes stay on spawning thread)");
                return ExitCode::SUCCESS;
            }
            if arg == "--version" || arg == "-v" {
                println!("nostos {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            if arg == "--no-jit" {
                enable_jit = false;
                i += 1;
                continue;
            }
            if arg == "--debug" {
                debug_mode = true;
                i += 1;
                continue;
            }
            if arg == "--json-errors" {
                json_errors = true;
                i += 1;
                continue;
            }
            if arg == "--parallel" {
                // Check if next arg is a number
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        parallel_workers = Some(n);
                        i += 2;
                        continue;
                    }
                }
                // No number specified, use 0 (auto-detect)
                parallel_workers = Some(0);
                i += 1;
                continue;
            }
            if arg == "--parallel-affinity" {
                // Check if next arg is a number
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        parallel_affinity = Some(n);
                        i += 2;
                        continue;
                    }
                }
                // No number specified, use 0 (auto-detect)
                parallel_affinity = Some(0);
                i += 1;
                continue;
            }
            i += 1;
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
    let compiler = match compile_module(&module, &source) {
        Ok(c) => c,
        Err(e) => {
            let source_error = e.to_source_error();
            source_error.eprint(file_path, &source);
            return ExitCode::FAILURE;
        }
    };

    // Create runtime and load functions/types
    let mut runtime = Runtime::new();
    runtime.set_debug_mode(debug_mode);

    for (name, func) in compiler.get_all_functions() {
        runtime.register_function(&name, func.clone());
    }
    let function_list = compiler.get_function_list();
    runtime.set_function_list(function_list.clone());
    for (name, type_val) in compiler.get_vm_types() {
        runtime.register_type(&name, type_val);
    }

    // JIT compile suitable functions (unless --no-jit was specified)
    if enable_jit {
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
                        // Register pure numeric JIT functions
                        if let Some(jit_fn) = jit.get_int_function(idx as u16) {
                            runtime.register_jit_int_function(idx as u16, jit_fn);
                        }
                        // Register loop-based array JIT functions
                        if let Some(jit_fn) = jit.get_loop_int64_array_function(idx as u16) {
                            runtime.register_jit_loop_array_function(idx as u16, jit_fn);
                        }
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

    // Run with either parallel WorkerPool, ParallelVM (affinity), or single-threaded Runtime
    if let Some(num_threads) = parallel_affinity {
        // Parallel execution with CPU affinity (new ParallelVM)
        let config = ParallelConfig {
            num_threads,
            ..Default::default()
        };

        let mut vm = ParallelVM::new(config);

        // Register default native functions
        vm.register_default_natives();

        // Register functions
        for (name, func) in compiler.get_all_functions() {
            vm.register_function(&name, func.clone());
        }

        // Set function list for CallDirect
        vm.set_function_list(compiler.get_function_list());

        // Register types
        for (name, type_val) in compiler.get_vm_types() {
            vm.register_type(&name, type_val);
        }

        // JIT compile suitable functions (unless --no-jit was specified)
        if enable_jit {
            if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
                for idx in 0..function_list.len() {
                    jit.queue_compilation(idx as u16);
                }
                if let Ok(compiled) = jit.process_queue(&function_list) {
                    if compiled > 0 {
                        for (idx, _func) in function_list.iter().enumerate() {
                            if let Some(jit_fn) = jit.get_int_function(idx as u16) {
                                vm.register_jit_int_function(idx as u16, jit_fn);
                            }
                            if let Some(jit_fn) = jit.get_loop_int64_array_function(idx as u16) {
                                vm.register_jit_loop_array_function(idx as u16, jit_fn);
                            }
                        }
                    }
                }
            }
        }

        match vm.run(main_func) {
            Ok(Some(val)) => {
                if !val.is_unit() {
                    println!("{}", val.display());
                }
                return ExitCode::SUCCESS;
            }
            Ok(None) => return ExitCode::SUCCESS,
            Err(e) => {
                if json_errors {
                    println!("{}", format_error_json(&e, file_path));
                } else {
                    eprintln!("Runtime error in {}:", file_path);
                    eprintln!("{}", e);
                }
                return ExitCode::FAILURE;
            }
        }
    }

    let result = if let Some(num_workers) = parallel_workers {
        // Parallel execution with work-stealing WorkerPool
        let scheduler = Scheduler::new();

        // Register functions on scheduler
        for (name, func) in compiler.get_all_functions() {
            scheduler.functions.write().insert(name.clone(), func.clone());
        }

        let config = WorkerPoolConfig {
            num_workers,
            enable_jit,
            ..Default::default()
        };

        let pool = WorkerPool::new(scheduler, config);
        pool.spawn_initial(main_func);
        let result = pool.run();
        pool.shutdown();
        result
    } else {
        // Single-threaded execution
        runtime.spawn_initial(main_func);
        runtime.run()
    };

    match result {
        Ok(result) => {
            if let Some(val) = result {
                if !matches!(val, GcValue::Unit) {
                    // Use display_result to get human-readable output (matches ParallelVM)
                    println!("{}", runtime.display_result(&val));
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if json_errors {
                println!("{}", format_error_json(&e, file_path));
            } else {
                eprintln!("Runtime error in {}:", file_path);
                eprintln!("{}", e);
            }
            ExitCode::FAILURE
        }
    }
}
