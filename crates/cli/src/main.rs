//! Nostos CLI - Command-line interface for running Nostos programs.

mod repl;
mod tui;
mod editor;
mod custom_views;
mod repl_panel;
mod autocomplete;
mod inspector_panel;
mod nostos_panel;
mod debug_panel;
mod git_panel;

use nostos_compiler::compile::{compile_module, Compiler, MvarInitValue};
use nostos_jit::{JitCompiler, JitConfig};
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::async_vm::{AsyncVM, AsyncConfig};
use nostos_vm::process::ThreadSafeValue;
use nostos_vm::value::RuntimeError;
use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::atomic::Ordering;

use repl::{Repl, ReplConfig};

/// Recursively visit directories and find .nos files
fn visit_dirs(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                // Skip .nostos directory - it contains per-definition files for REPL/TUI
                if let Some(name) = path.file_name() {
                    if name == ".nostos" {
                        continue;
                    }
                }
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
        RuntimeError::Interrupted => {
            ("Interrupted".to_string(), "Execution interrupted (Ctrl+C)".to_string())
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

/// Convert MvarInitValue to ThreadSafeValue for VM registration
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
        MvarInitValue::StringList(strings) => ThreadSafeValue::List(
            strings.iter().map(|s| ThreadSafeValue::String(s.clone())).collect()
        ),
        MvarInitValue::FloatList(floats) => ThreadSafeValue::List(
            floats.iter().map(|f| ThreadSafeValue::Float64(*f)).collect()
        ),
        MvarInitValue::BoolList(bools) => ThreadSafeValue::List(
            bools.iter().map(|b| ThreadSafeValue::Bool(*b)).collect()
        ),
        MvarInitValue::Tuple(items) => ThreadSafeValue::Tuple(
            items.iter().map(|item| mvar_init_to_thread_safe(item)).collect()
        ),
        MvarInitValue::List(items) => ThreadSafeValue::List(
            items.iter().map(|item| mvar_init_to_thread_safe(item)).collect()
        ),
        MvarInitValue::Record(type_name, fields) => {
            // Convert to record with field names and values
            let field_names: Vec<String> = fields.iter()
                .enumerate()
                .map(|(i, (name, _))| {
                    if name.is_empty() {
                        format!("_{}", i) // Positional fields
                    } else {
                        name.clone()
                    }
                })
                .collect();
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
            use nostos_vm::{SharedMapKey, SharedMapValue};
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

/// Convert MvarInitValue to SharedMapKey (for map keys)
fn mvar_init_to_shared_key(init: &MvarInitValue) -> Option<nostos_vm::SharedMapKey> {
    use nostos_vm::SharedMapKey;
    match init {
        MvarInitValue::Unit => Some(SharedMapKey::Unit),
        MvarInitValue::Bool(b) => Some(SharedMapKey::Bool(*b)),
        MvarInitValue::Int(n) => Some(SharedMapKey::Int64(*n)),
        MvarInitValue::String(s) => Some(SharedMapKey::String(s.clone())),
        MvarInitValue::Char(c) => Some(SharedMapKey::Char(*c)),
        _ => None, // Other types can't be map keys
    }
}

/// Convert MvarInitValue to SharedMapValue (for map values)
fn mvar_init_to_shared_value(init: &MvarInitValue) -> nostos_vm::SharedMapValue {
    use nostos_vm::SharedMapValue;
    match init {
        MvarInitValue::Unit => SharedMapValue::Unit,
        MvarInitValue::Bool(b) => SharedMapValue::Bool(*b),
        MvarInitValue::Int(n) => SharedMapValue::Int64(*n),
        MvarInitValue::Float(f) => SharedMapValue::Float64(*f),
        MvarInitValue::String(s) => SharedMapValue::String(s.clone()),
        MvarInitValue::Char(c) => SharedMapValue::Char(*c),
        MvarInitValue::EmptyList => SharedMapValue::List(vec![]),
        MvarInitValue::IntList(ints) => SharedMapValue::List(
            ints.iter().map(|n| SharedMapValue::Int64(*n)).collect()
        ),
        MvarInitValue::FloatList(floats) => SharedMapValue::List(
            floats.iter().map(|f| SharedMapValue::Float64(*f)).collect()
        ),
        MvarInitValue::BoolList(bools) => SharedMapValue::List(
            bools.iter().map(|b| SharedMapValue::Bool(*b)).collect()
        ),
        MvarInitValue::StringList(strings) => SharedMapValue::List(
            strings.iter().map(|s| SharedMapValue::String(s.clone())).collect()
        ),
        MvarInitValue::List(items) => SharedMapValue::List(
            items.iter().map(|item| mvar_init_to_shared_value(item)).collect()
        ),
        MvarInitValue::Tuple(items) => SharedMapValue::Tuple(
            items.iter().map(|item| mvar_init_to_shared_value(item)).collect()
        ),
        MvarInitValue::Record(type_name, fields) => {
            let field_names: Vec<String> = fields.iter()
                .enumerate()
                .map(|(i, (name, _))| if name.is_empty() { format!("_{}", i) } else { name.clone() })
                .collect();
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
                    let value = mvar_init_to_shared_value(v);
                    map.insert(key, value);
                }
            }
            SharedMapValue::Map(std::sync::Arc::new(map))
        }
    }
}

/// Run program using the tokio-based AsyncVM.
fn run_with_async_vm(
    compiler: &Compiler,
    entry_point_name: &str,
    profiling_enabled: bool,
    enable_jit: bool,
) -> ExitCode {
    let config = AsyncConfig {
        profiling_enabled,
        ..AsyncConfig::default()
    };
    let mut vm = AsyncVM::new(config);

    // Register default native functions
    vm.register_default_natives();

    // Register all functions from compiler
    for (name, func) in compiler.get_all_functions().iter() {
        vm.register_function(name, func.clone());
    }

    // Set function list for indexed calls
    let func_list = compiler.get_function_list();
    vm.set_function_list(func_list);

    // Register all types
    for (name, type_val) in compiler.get_vm_types().iter() {
        vm.register_type(name, type_val.clone());
    }

    // Register mvars (module-level mutable variables)
    for (name, info) in compiler.get_mvars() {
        let initial_value = mvar_init_to_thread_safe(&info.initial_value);
        vm.register_mvar(name, initial_value);
    }

    // JIT compile suitable functions (unless --no-jit was specified)
    let function_list = compiler.get_function_list();
    if enable_jit {
    if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
        for idx in 0..function_list.len() {
            jit.queue_compilation(idx as u16);
        }
        if let Ok(compiled) = jit.process_queue(&function_list) {
            if compiled > 0 {
                for (idx, _func) in function_list.iter().enumerate() {
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
                    if let Some(jit_fn) = jit.get_loop_int64_array_function(idx as u16) {
                        vm.register_jit_loop_array_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_recursive_array_fill_function(idx as u16) {
                        vm.register_jit_array_fill_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_recursive_array_sum_function(idx as u16) {
                        vm.register_jit_array_sum_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_list_sum_function(idx as u16) {
                        vm.register_jit_list_sum_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_list_sum_tr_function(idx as u16) {
                        vm.register_jit_list_sum_tr_function(idx as u16, jit_fn);
                    }
                }
            }
        }
    }
    }

    // Set up Ctrl+C handler for interrupting running code
    let interrupt_handle = vm.get_interrupt_handle();
    if let Err(e) = ctrlc::set_handler(move || {
        interrupt_handle.interrupt.store(true, Ordering::SeqCst);
    }) {
        eprintln!("Warning: Could not set Ctrl+C handler: {}", e);
    }

    // Run the program
    match vm.run(entry_point_name) {
        Ok(result) => {
            if !result.is_unit() {
                println!("{}", result.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) if e.contains("Interrupted") => {
            eprintln!("Interrupted");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
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
    let mut profiling_enabled = false; // Enable function call profiling

    let mut i = 1;
    let mut file_idx: Option<usize> = None;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") || arg.starts_with("-") {
            if arg == "--help" || arg == "-h" {
                println!("Usage: nostos [options] <command|file.nos> [options...]");
                println!();
                println!("Commands:");
                println!("  repl           Start the interactive REPL");
                println!();
                println!("Run a Nostos program file or start the REPL.");
                println!();
                println!("Options (can appear before or after the file):");
                println!("  --help           Show this help message");
                println!("  --version        Show version information");
                println!("  --no-jit         Disable JIT compilation (for debugging)");
                println!("  --debug          Show local variable values in stack traces");
                println!("  --json-errors    Output errors as JSON (for debugger integration)");
                println!("  --threads N      Use N worker threads (default: all CPUs)");
                println!("  --profile        Enable function call profiling (JIT functions show as [JIT])");
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
            if arg == "--profile" {
                profiling_enabled = true;
                // JIT profiling is now supported - JIT functions show as "[JIT] function_name"
                i += 1;
                continue;
            }
            i += 1;
        } else {
            // First non-flag argument is the file
            if file_idx.is_none() {
                file_idx = Some(i);
            }
            i += 1;
        }
    }

    let file_idx = match file_idx {
        Some(idx) => idx,
        None => {
            eprintln!("Error: No input file specified");
            eprintln!("Use 'nostos repl' to start the interactive REPL");
            return ExitCode::FAILURE;
        }
    };

    let file_path_arg = &args[file_idx];
    let input_path = std::path::Path::new(file_path_arg);

    // Check if input is directory or file
    let mut source_files = Vec::new();
    let project_root;

    if input_path.is_dir() {
        project_root = input_path;

        // Check for main.nos in the directory
        let main_file = input_path.join("main.nos");
        if !main_file.exists() {
            eprintln!("Error: No 'main.nos' found in directory '{}'", file_path_arg);
            eprintln!("Projects must have a main.nos file with a main() function.");
            return ExitCode::FAILURE;
        }

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

        // Track all stdlib function names for prelude imports
        let mut stdlib_functions: Vec<(String, String)> = Vec::new();

        for file_path in &stdlib_files {
             let source = fs::read_to_string(file_path).expect("Failed to read stdlib file");
             let (module_opt, _) = parse(&source);
             if let Some(module) = module_opt {
                 // Build module path: stdlib.list, stdlib.json, etc.
                 let relative = file_path.strip_prefix(&stdlib_path).unwrap();
                 let mut components: Vec<String> = vec!["stdlib".to_string()];
                 for component in relative.components() {
                     let s = component.as_os_str().to_string_lossy().to_string();
                     if s.ends_with(".nos") {
                         components.push(s.trim_end_matches(".nos").to_string());
                     } else {
                         components.push(s);
                     }
                 }
                 let module_prefix = components.join(".");

                 // Collect function names from this module for prelude imports
                 for item in &module.items {
                     if let nostos_syntax::ast::Item::FnDef(fn_def) = item {
                         let local_name = fn_def.name.node.clone();
                         let qualified_name = format!("{}.{}", module_prefix, local_name);
                         stdlib_functions.push((local_name, qualified_name));
                     }
                 }

                 compiler.add_module(&module, components, std::sync::Arc::new(source.clone()), file_path.to_str().unwrap().to_string()).expect("Failed to compile stdlib");
             }
        }

        // Register prelude imports so stdlib functions are available without prefix
        for (local_name, qualified_name) in stdlib_functions {
            compiler.add_prelude_import(local_name, qualified_name);
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

    // Compile all bodies (includes mvar safety check)
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

    // Run with AsyncVM
    run_with_async_vm(&compiler, &entry_point_name, profiling_enabled, enable_jit)
}
