//! Nostos CLI - Command-line interface for running Nostos programs.

mod repl;
mod tui;
mod editor;
mod custom_views;
mod repl_panel;
mod autocomplete;
mod inspector_panel;

use nostos_compiler::compile::{compile_module, Compiler};
use nostos_jit::{JitCompiler, JitConfig};
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::parallel::{ParallelVM, ParallelConfig};
use nostos_vm::value::RuntimeError;
use std::env;
use std::fs;
use std::process::ExitCode;

use repl::{Repl, ReplConfig};

/// Recursively visit directories and find .nos files
fn visit_dirs(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else {
                if let Some(ext) = path.extension() {
                    if ext == "nos" {
                        files.push(path);
                    }
                }
            }
        }
    }
    Ok(())
}

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

/// Run the REPL with optional initial files
fn run_repl(args: &[String]) -> ExitCode {
    let mut config = ReplConfig::default();
    let mut files_to_load = Vec::new();

    // Parse repl-specific options
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--no-jit" {
            config.enable_jit = false;
            i += 1;
        } else if arg == "--threads" && i + 1 < args.len() {
            if let Ok(n) = args[i + 1].parse::<usize>() {
                config.num_threads = n;
            }
            i += 2;
        } else if arg.starts_with("-") {
            eprintln!("Unknown option: {}", arg);
            i += 1;
        } else {
            // Treat as file to load
            files_to_load.push(arg.clone());
            i += 1;
        }
    }

    let mut repl = Repl::new(config);

    // Load stdlib
    if let Err(e) = repl.load_stdlib() {
        eprintln!("Warning: Failed to load stdlib: {}", e);
    }

    // Load any specified files
    for file in files_to_load {
        if let Err(e) = repl.load_file(&file) {
            eprintln!("Error loading {}: {}", file, e);
            return ExitCode::FAILURE;
        }
        println!("Loaded {}", file);
    }

    // Run the REPL
    match repl.run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("REPL error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: nostos [options] <command|file.nos> [args...]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  repl       Start the interactive REPL");
        eprintln!();
        eprintln!("Run a Nostos program file or start the REPL.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --help     Show this help message");
        eprintln!("  --version  Show version information");
        return ExitCode::FAILURE;
    }

    // Check for repl subcommand
    if args.len() >= 2 {
        if args[1] == "repl" {
            return run_repl(&args[2..]);
        }
        if args[1] == "tui" {
            return tui::run_tui(&args[2..]);
        }
    }

    // Parse options
    let mut file_idx = 1;
    let mut enable_jit = true;
    let mut json_errors = false;
    let mut debug_mode = false;
    let mut num_threads: usize = 0; // 0 = auto-detect

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") || arg.starts_with("-") {
            if arg == "--help" || arg == "-h" {
                println!("Usage: nostos [options] <command|file.nos> [args...]");
                println!();
                println!("Commands:");
                println!("  repl           Start the interactive REPL");
                println!();
                println!("Run a Nostos program file or start the REPL.");
                println!();
                println!("Options:");
                println!("  --help           Show this help message");
                println!("  --version        Show version information");
                println!("  --no-jit         Disable JIT compilation (for debugging)");
                println!("  --debug          Show local variable values in stack traces");
                println!("  --json-errors    Output errors as JSON (for debugger integration)");
                println!("  --threads N      Use N worker threads (default: all CPUs)");
                println!();
                println!("REPL usage:");
                println!("  nostos repl              Start interactive REPL");
                println!("  nostos repl file.nos     Load file and start REPL");
                println!("  nostos repl < script.nos Run script non-interactively");
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
            if arg == "--threads" || arg == "--parallel" || arg == "--parallel-affinity" {
                // Set number of worker threads
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        num_threads = n;
                        i += 2;
                        continue;
                    }
                }
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
        eprintln!("Use 'nostos repl' to start the interactive REPL");
        return ExitCode::FAILURE;
    }

    let file_path_arg = &args[file_idx];
    let input_path = std::path::Path::new(file_path_arg);

    // Check if input is directory or file
    let mut source_files = Vec::new();
    let project_root;

    if input_path.is_dir() {
        project_root = input_path;
        match visit_dirs(input_path, &mut source_files) {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error scanning directory '{}': {}", file_path_arg, e);
                return ExitCode::FAILURE;
            }
        }
        if source_files.is_empty() {
            eprintln!("No .nos files found in '{}'", file_path_arg);
            return ExitCode::FAILURE;
        }
    } else {
        project_root = input_path.parent().unwrap_or(std::path::Path::new("."));
        source_files.push(input_path.to_path_buf());
    }

    // Initialize empty compiler
    let mut compiler = Compiler::new_empty();

    // Load stdlib
    let stdlib_candidates = vec![
        std::path::PathBuf::from("stdlib"),
        std::path::PathBuf::from("../stdlib"),
    ];
    
    let mut stdlib_path = std::path::PathBuf::from("stdlib");
    let mut found_stdlib = false;
    
    for path in stdlib_candidates {
        if path.is_dir() {
            stdlib_path = path;
            found_stdlib = true;
            break;
        }
    }
    
    if !found_stdlib {
        // Try relative to executable
        if let Ok(mut p) = std::env::current_exe() {
             p.pop(); // remove binary name
             p.pop(); // remove release/debug
             p.pop(); // remove target
             p.push("stdlib");
             if p.is_dir() {
                 stdlib_path = p;
                 found_stdlib = true;
             }
        }
    }

    if found_stdlib {
        let mut stdlib_files = Vec::new();
        visit_dirs(&stdlib_path, &mut stdlib_files).ok();

        for path in &stdlib_files {
             let source = fs::read_to_string(path).expect("Failed to read stdlib file");
             let (module_opt, _) = parse(&source);
             if let Some(module) = module_opt {
                 let relative = path.strip_prefix(&stdlib_path).unwrap();
                 let mut components: Vec<String> = relative.components()
                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                    .collect();
                 if let Some(last) = components.last_mut() {
                    if last.ends_with(".nos") {
                        *last = last.trim_end_matches(".nos").to_string();
                    }
                 }
                 
                 compiler.add_module(&module, components, std::sync::Arc::new(source.clone()), path.to_str().unwrap().to_string()).expect("Failed to compile stdlib");
             }
        }
    }

    // Process each file
    for path in &source_files {
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading file '{}': {}", path.display(), e);
                return ExitCode::FAILURE;
            }
        };

        let (module_opt, errors) = parse(&source);
        if !errors.is_empty() {
            let source_errors = parse_errors_to_source_errors(&errors);
            eprint_errors(&source_errors, path.to_str().unwrap_or("unknown"), &source);
            return ExitCode::FAILURE;
        }

        let module = match module_opt {
            Some(m) => m,
            None => {
                eprintln!("Failed to parse '{}'", path.display());
                return ExitCode::FAILURE;
            }
        };

        // Determine module path based on file location relative to project root
        // For single file, module path is empty (top-level)
        let module_path = if input_path.is_dir() {
            let relative = path.strip_prefix(project_root).unwrap();
            let mut components: Vec<String> = relative.components()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect();

            // Remove .nos extension from last component
            if let Some(last) = components.last_mut() {
                if last.ends_with(".nos") {
                    *last = last.trim_end_matches(".nos").to_string();
                }
            }
            components
        } else {
            vec![]
        };

        // Add to compiler with source tracking
        if let Err(e) = compiler.add_module(&module, module_path, std::sync::Arc::new(source.clone()), path.to_str().unwrap_or("unknown").to_string()) {
            let source_error = e.to_source_error();
            source_error.eprint(path.to_str().unwrap_or("unknown"), &source);
            return ExitCode::FAILURE;
        }
    }

    // Compile all bodies
    if let Err((e, filename, source)) = compiler.compile_all() {
        let source_error = e.to_source_error();
        source_error.eprint(&filename, &source);
        return ExitCode::FAILURE;
    }

    // Resolve entry point (function names now include signature, main has no params so it's "main/")
    let entry_point_name = if input_path.is_dir() {
        // Check for main.main/ or main/ (with signature suffix)
        let funcs = compiler.get_all_functions();
        if funcs.contains_key("main.main/") {
            "main.main/".to_string()
        } else if funcs.contains_key("main/") {
            "main/".to_string()
        } else if funcs.keys().any(|k| k.starts_with("main.main/")) {
            // Find the main.main function with any signature (should be 0-arity for entry point)
            funcs.keys()
                .find(|k| k.starts_with("main.main/"))
                .cloned()
                .unwrap_or_else(|| "main.main/".to_string())
        } else if funcs.keys().any(|k| k.starts_with("main/")) {
            funcs.keys()
                .find(|k| k.starts_with("main/"))
                .cloned()
                .unwrap_or_else(|| "main/".to_string())
        } else {
             eprintln!("Error: No 'main.main' or 'main' function found in project.");
             return ExitCode::FAILURE;
        }
    } else {
        "main/".to_string()
    };

    // Get main function
    let main_func = match compiler.get_function(&entry_point_name) {
        Some(func) => func,
        None => {
            eprintln!("Error: Entry point '{}' not found", entry_point_name);
            return ExitCode::FAILURE;
        }
    };

    // Create ParallelVM (always use parallel execution)
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
    let function_list = compiler.get_function_list(); // Reuse variable name or shadow
    if enable_jit {
        if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
            for idx in 0..function_list.len() {
                jit.queue_compilation(idx as u16);
            }
            if let Ok(compiled) = jit.process_queue(&function_list) {
                if compiled > 0 {
                    for (idx, _func) in function_list.iter().enumerate() {
                        // Register pure numeric JIT functions by arity
                        if let Some(jit_fn) = jit.get_int_function_0(idx as u16) {
                            vm.register_jit_int_function_0(idx as u16, jit_fn);
                        }
                        if let Some(jit_fn) = jit.get_int_function(idx as u16) {
                            vm.register_jit_int_function(idx as u16, jit_fn);
                        }
                        if let Some(jit_fn) = jit.get_int_function_2(idx as u16) {
                            vm.register_jit_int_function_2(idx as u16, jit_fn);
                        }
                        if let Some(jit_fn) = jit.get_int_function_3(idx as u16) {
                            vm.register_jit_int_function_3(idx as u16, jit_fn);
                        }
                        if let Some(jit_fn) = jit.get_int_function_4(idx as u16) {
                            vm.register_jit_int_function_4(idx as u16, jit_fn);
                        }
                        // Register loop-based array JIT functions
                        if let Some(jit_fn) = jit.get_loop_int64_array_function(idx as u16) {
                            vm.register_jit_loop_array_function(idx as u16, jit_fn);
                        }
                    }
                }
            }
        }
    }

    // Run the program
    match vm.run(main_func) {
        Ok(result) => {
            // Output is already printed to stdout by the VM
            // Print return value if not unit
            if let Some(val) = result.value {
                if !val.is_unit() {
                    println!("{}", val.display());
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if json_errors {
                println!("{}", format_error_json(&e, file_path_arg));
            } else {
                eprintln!("Runtime error:");
                eprintln!("{}", e);
            }
            ExitCode::FAILURE
        }
    }
}
