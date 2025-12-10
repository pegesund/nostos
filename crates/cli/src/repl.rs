//! Interactive REPL for Nostos
//!
//! Provides Haskell-like introspection with Forth/Lisp flexibility:
//! - `:help` - Show available commands
//! - `:quit` - Exit the REPL
//! - `:load <file>` - Load a file
//! - `:reload` - Reload previously loaded files
//! - `:browse [module]` - List functions (optionally in a module)
//! - `:info <name>` - Show info about a function/type
//! - `:view <name>` - Show source code
//! - `:type <expr>` - Show the type of an expression
//! - `:deps <name>` - Show what a function depends on
//! - `:rdeps <name>` - Show what depends on a function

use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

use nostos_compiler::compile::Compiler;
use nostos_jit::{JitCompiler, JitConfig};
use nostos_repl::CallGraph;
use nostos_syntax::ast::Item;
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::parallel::{ParallelVM, ParallelConfig};

/// REPL configuration
pub struct ReplConfig {
    pub enable_jit: bool,
    pub num_threads: usize,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            enable_jit: true,
            num_threads: 0, // auto-detect
        }
    }
}

/// The REPL state
pub struct Repl {
    compiler: Compiler,
    vm: ParallelVM,
    loaded_files: Vec<PathBuf>,
    config: ReplConfig,
    stdlib_path: Option<PathBuf>,
    call_graph: CallGraph,
    eval_counter: u64,
}

impl Repl {
    /// Create a new REPL instance
    pub fn new(config: ReplConfig) -> Self {
        let compiler = Compiler::new_empty();
        let vm_config = ParallelConfig {
            num_threads: config.num_threads,
            ..Default::default()
        };
        let mut vm = ParallelVM::new(vm_config);
        vm.register_default_natives();

        Self {
            compiler,
            vm,
            loaded_files: Vec::new(),
            config,
            stdlib_path: None,
            call_graph: CallGraph::new(),
            eval_counter: 0,
        }
    }

    /// Load the standard library
    pub fn load_stdlib(&mut self) -> Result<(), String> {
        let stdlib_candidates = vec![
            PathBuf::from("stdlib"),
            PathBuf::from("../stdlib"),
        ];

        let mut stdlib_path = None;

        for path in stdlib_candidates {
            if path.is_dir() {
                stdlib_path = Some(path);
                break;
            }
        }

        if stdlib_path.is_none() {
            // Try relative to executable
            if let Ok(mut p) = std::env::current_exe() {
                p.pop(); // remove binary name
                p.pop(); // remove release/debug
                p.pop(); // remove target
                p.push("stdlib");
                if p.is_dir() {
                    stdlib_path = Some(p);
                }
            }
        }

        if let Some(path) = &stdlib_path {
            let mut stdlib_files = Vec::new();
            visit_dirs(path, &mut stdlib_files)?;

            for file_path in &stdlib_files {
                let source = fs::read_to_string(file_path)
                    .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e))?;
                let (module_opt, _) = parse(&source);
                if let Some(module) = module_opt {
                    let relative = file_path.strip_prefix(path).unwrap();
                    let mut components: Vec<String> = relative.components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                        .collect();
                    if let Some(last) = components.last_mut() {
                        if last.ends_with(".nos") {
                            *last = last.trim_end_matches(".nos").to_string();
                        }
                    }

                    self.compiler.add_module(&module, components, Arc::new(source.clone()), file_path.to_str().unwrap().to_string())
                        .map_err(|e| format!("Failed to compile stdlib: {}", e))?;
                }
            }
            self.stdlib_path = stdlib_path;
        }

        Ok(())
    }

    /// Run the main REPL loop
    pub fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let is_interactive = stdin.is_terminal();

        if is_interactive {
            println!("Nostos REPL v{}", env!("CARGO_PKG_VERSION"));
            println!("Type :help for available commands, :quit to exit");
            println!();
        }

        // Buffer for multi-line input
        let mut input_buffer = String::new();
        let mut in_multiline = false;

        loop {
            // Print prompt
            if is_interactive {
                if in_multiline {
                    print!("... ");
                } else {
                    print!("nos> ");
                }
                stdout.flush()?;
            }

            // Read line
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => {
                    // EOF
                    if is_interactive {
                        println!();
                    }
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    break;
                }
            }

            let line = line.trim_end();

            // Handle multi-line input
            if in_multiline {
                if line.is_empty() {
                    // Empty line ends multi-line input
                    in_multiline = false;
                    let input = std::mem::take(&mut input_buffer);
                    self.process_input(&input);
                } else {
                    input_buffer.push_str(line);
                    input_buffer.push('\n');
                }
                continue;
            }

            // Check for commands
            if line.starts_with(':') {
                match self.handle_command(line) {
                    CommandResult::Continue => continue,
                    CommandResult::Quit => break,
                    CommandResult::Error(msg) => {
                        eprintln!("{}", msg);
                        continue;
                    }
                }
            }

            // Check if line ends with backslash (multi-line continuation)
            if line.ends_with('\\') {
                in_multiline = true;
                input_buffer = line[..line.len()-1].to_string();
                input_buffer.push('\n');
                continue;
            }

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Process input (expression or definition)
            self.process_input(line);
        }

        Ok(())
    }

    /// Handle a REPL command
    fn handle_command(&mut self, line: &str) -> CommandResult {
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        let cmd = parts[0];
        let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

        match cmd {
            ":quit" | ":q" | ":exit" => CommandResult::Quit,

            ":help" | ":h" | ":?" => {
                self.show_help();
                CommandResult::Continue
            }

            ":load" | ":l" => {
                if args.is_empty() {
                    return CommandResult::Error("Usage: :load <file.nos>".to_string());
                }
                match self.load_file(args) {
                    Ok(()) => {
                        println!("Loaded {}", args);
                        CommandResult::Continue
                    }
                    Err(e) => CommandResult::Error(e),
                }
            }

            ":reload" | ":r" => {
                match self.reload_files() {
                    Ok(count) => {
                        println!("Reloaded {} file(s)", count);
                        CommandResult::Continue
                    }
                    Err(e) => CommandResult::Error(e),
                }
            }

            ":browse" | ":b" => {
                self.browse(if args.is_empty() { None } else { Some(args) });
                CommandResult::Continue
            }

            ":info" | ":i" => {
                if args.is_empty() {
                    return CommandResult::Error("Usage: :info <name>".to_string());
                }
                self.show_info(args);
                CommandResult::Continue
            }

            ":view" | ":v" => {
                if args.is_empty() {
                    return CommandResult::Error("Usage: :view <name>".to_string());
                }
                self.show_source(args);
                CommandResult::Continue
            }

            ":type" | ":t" => {
                if args.is_empty() {
                    return CommandResult::Error("Usage: :type <expression>".to_string());
                }
                self.show_type(args);
                CommandResult::Continue
            }

            ":deps" | ":d" => {
                if args.is_empty() {
                    return CommandResult::Error("Usage: :deps <function>".to_string());
                }
                self.show_deps(args);
                CommandResult::Continue
            }

            ":rdeps" | ":rd" => {
                if args.is_empty() {
                    return CommandResult::Error("Usage: :rdeps <function>".to_string());
                }
                self.show_rdeps(args);
                CommandResult::Continue
            }

            ":functions" | ":fns" => {
                self.list_functions();
                CommandResult::Continue
            }

            ":types" => {
                self.list_types();
                CommandResult::Continue
            }

            ":traits" => {
                self.list_traits();
                CommandResult::Continue
            }

            _ => CommandResult::Error(format!("Unknown command: {}. Type :help for available commands.", cmd)),
        }
    }

    /// Show help text
    fn show_help(&self) {
        println!("Commands:");
        println!("  :help, :h, :?        Show this help");
        println!("  :quit, :q, :exit     Exit the REPL");
        println!("  :load <file>, :l     Load a Nostos file");
        println!("  :reload, :r          Reload previously loaded files");
        println!("  :browse [module], :b List functions (optionally in a module)");
        println!("  :info <name>, :i     Show info about a function or type");
        println!("  :view <name>, :v     Show source code of a function");
        println!("  :type <expr>, :t     Show the type of an expression");
        println!("  :deps <fn>, :d       Show dependencies of a function");
        println!("  :rdeps <fn>, :rd     Show reverse dependencies (what calls this)");
        println!("  :functions, :fns     List all functions");
        println!("  :types               List all types");
        println!("  :traits              List all traits");
        println!();
        println!("Input:");
        println!("  <expression>         Evaluate an expression");
        println!("  <name>(...) = ...    Define a function");
        println!("  type <Name> = ...    Define a type");
        println!();
        println!("Use \\ at end of line for multi-line input");
    }

    /// Load a file into the REPL
    pub fn load_file(&mut self, path_str: &str) -> Result<(), String> {
        let path = PathBuf::from(path_str);

        if !path.exists() {
            return Err(format!("File not found: {}", path_str));
        }

        let source = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let (module_opt, errors) = parse(&source);
        if !errors.is_empty() {
            let source_errors = parse_errors_to_source_errors(&errors);
            eprint_errors(&source_errors, path_str, &source);
            return Err("Parse errors".to_string());
        }

        let module = module_opt.ok_or("Failed to parse file")?;

        // Add to compiler
        self.compiler.add_module(&module, vec![], Arc::new(source.clone()), path_str.to_string())
            .map_err(|e| format!("Compilation error: {}", e))?;

        // Compile all bodies
        if let Err((e, filename, source)) = self.compiler.compile_all() {
            let source_error = e.to_source_error();
            source_error.eprint(&filename, &source);
            return Err("Compilation error".to_string());
        }

        // Update VM with new functions
        self.sync_vm();

        // Track loaded file for reload
        if !self.loaded_files.contains(&path) {
            self.loaded_files.push(path);
        }

        Ok(())
    }

    /// Reload all previously loaded files
    fn reload_files(&mut self) -> Result<usize, String> {
        let files = self.loaded_files.clone();
        let count = files.len();

        // Reset compiler (but keep stdlib)
        self.compiler = Compiler::new_empty();
        self.call_graph = CallGraph::new();
        if let Err(e) = self.load_stdlib() {
            eprintln!("Warning: Failed to reload stdlib: {}", e);
        }

        self.loaded_files.clear();

        for path in files {
            let path_str = path.to_string_lossy().to_string();
            self.load_file(&path_str)?;
        }

        Ok(count)
    }

    /// Browse functions, optionally filtered by module
    fn browse(&self, module_filter: Option<&str>) {
        let mut functions: Vec<_> = self.compiler.get_function_names()
            .into_iter()
            .filter(|name| {
                if let Some(filter) = module_filter {
                    name.starts_with(filter)
                } else {
                    true
                }
            })
            .collect();

        functions.sort();

        if functions.is_empty() {
            if let Some(filter) = module_filter {
                println!("No functions found in module '{}'", filter);
            } else {
                println!("No functions defined");
            }
            return;
        }

        for name in functions {
            if let Some(sig) = self.compiler.get_function_signature(name) {
                println!("  {} :: {}", name, sig);
            } else {
                println!("  {}", name);
            }
        }
    }

    /// Show info about a function or type
    fn show_info(&self, name: &str) {
        // Try as function first
        if let Some(fn_def) = self.compiler.get_fn_def(name) {
            println!("{}  (function)", name);

            // Signature
            let sig = fn_def.signature();
            if !sig.is_empty() && sig != "?" {
                println!("  :: {}", sig);
            }

            // Doc comment
            if let Some(doc) = &fn_def.doc {
                println!();
                for line in doc.lines() {
                    println!("  {}", line);
                }
            }

            // Module/file info
            if let Some(source) = self.compiler.get_function_source(name) {
                let lines = source.lines().count();
                println!();
                println!("  Defined in {} lines", lines);
            }

            // Dependencies
            let deps = self.call_graph.direct_dependencies(name);
            if !deps.is_empty() {
                println!();
                let mut deps_vec: Vec<_> = deps.iter().collect();
                deps_vec.sort();
                println!("  Depends on: {}", deps_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
            }

            return;
        }

        // Try as type
        if let Some(type_def) = self.compiler.get_type_def(name) {
            println!("{}  (type)", type_def.full_name());

            // Type body
            let body = type_def.body_string();
            if !body.is_empty() {
                println!("  = {}", body);
            }

            // Doc comment
            if let Some(doc) = &type_def.doc {
                println!();
                for line in doc.lines() {
                    println!("  {}", line);
                }
            }

            // Derived traits
            if !type_def.deriving.is_empty() {
                let traits: Vec<_> = type_def.deriving.iter().map(|t| t.node.as_str()).collect();
                println!();
                println!("  Deriving: {}", traits.join(", "));
            }

            return;
        }

        // Try as trait
        let implementors = self.compiler.get_trait_implementors(name);
        if !implementors.is_empty() {
            println!("{}  (trait)", name);
            println!();
            println!("  Implemented by:");
            for ty in implementors {
                println!("    - {}", ty);
            }
            return;
        }

        println!("Not found: {}", name);
    }

    /// Show source code of a function
    fn show_source(&self, name: &str) {
        if let Some(source) = self.compiler.get_function_source(name) {
            println!("{}", source);
        } else if let Some(type_def) = self.compiler.get_type_def(name) {
            // Reconstruct type definition
            let mut output = String::new();
            if type_def.visibility == nostos_syntax::ast::Visibility::Private {
                output.push_str("private ");
            }
            output.push_str("type ");
            output.push_str(&type_def.full_name());

            let body = type_def.body_string();
            if !body.is_empty() {
                output.push_str(" = ");
                output.push_str(&body);
            }

            if !type_def.deriving.is_empty() {
                output.push_str(" deriving ");
                let traits: Vec<_> = type_def.deriving.iter().map(|t| t.node.as_str()).collect();
                output.push_str(&traits.join(", "));
            }

            println!("{}", output);
        } else {
            println!("Not found: {}", name);
        }
    }

    /// Show type of an expression (placeholder - needs type inference)
    fn show_type(&self, _expr: &str) {
        // TODO: Implement type inference for expressions
        println!("Type inference not yet implemented");
    }

    /// Show dependencies of a function
    fn show_deps(&self, name: &str) {
        let deps = self.call_graph.direct_dependencies(name);
        if deps.is_empty() {
            println!("{} has no dependencies", name);
        } else {
            println!("{} depends on:", name);
            let mut deps_vec: Vec<_> = deps.iter().collect();
            deps_vec.sort();
            for dep in deps_vec {
                println!("  {}", dep);
            }
        }
    }

    /// Show reverse dependencies (what calls this function)
    fn show_rdeps(&self, name: &str) {
        let rdeps = self.call_graph.direct_dependents(name);
        if rdeps.is_empty() {
            println!("{} is not called by any function", name);
        } else {
            println!("{} is called by:", name);
            let mut rdeps_vec: Vec<_> = rdeps.iter().collect();
            rdeps_vec.sort();
            for rdep in rdeps_vec {
                println!("  {}", rdep);
            }
        }
    }

    /// List all functions
    fn list_functions(&self) {
        let mut functions: Vec<_> = self.compiler.get_function_names().into_iter().collect();
        functions.sort();

        if functions.is_empty() {
            println!("No functions defined");
        } else {
            println!("Functions ({}):", functions.len());
            for name in functions {
                println!("  {}", name);
            }
        }
    }

    /// List all types
    fn list_types(&self) {
        let mut types: Vec<_> = self.compiler.get_type_names().into_iter().collect();
        types.sort();

        if types.is_empty() {
            println!("No types defined");
        } else {
            println!("Types ({}):", types.len());
            for name in types {
                println!("  {}", name);
            }
        }
    }

    /// List all traits
    fn list_traits(&self) {
        let mut traits: Vec<_> = self.compiler.get_trait_names().into_iter().collect();
        traits.sort();

        if traits.is_empty() {
            println!("No traits defined");
        } else {
            println!("Traits ({}):", traits.len());
            for name in traits {
                let implementors = self.compiler.get_trait_implementors(name);
                if implementors.is_empty() {
                    println!("  {}", name);
                } else {
                    println!("  {} ({})", name, implementors.len());
                }
            }
        }
    }

    /// Check if module has function or type definitions
    fn has_definitions(module: &nostos_syntax::Module) -> bool {
        for item in &module.items {
            match item {
                Item::FnDef(_) | Item::TypeDef(_) => return true,
                _ => {}
            }
        }
        false
    }

    /// Get function definitions from module
    fn get_fn_defs(module: &nostos_syntax::Module) -> Vec<&nostos_syntax::ast::FnDef> {
        module.items.iter().filter_map(|item| {
            if let Item::FnDef(fn_def) = item {
                Some(fn_def)
            } else {
                None
            }
        }).collect()
    }

    /// Get type definitions from module
    fn get_type_defs(module: &nostos_syntax::Module) -> Vec<&nostos_syntax::ast::TypeDef> {
        module.items.iter().filter_map(|item| {
            if let Item::TypeDef(type_def) = item {
                Some(type_def)
            } else {
                None
            }
        }).collect()
    }

    /// Process user input (expression or definition)
    fn process_input(&mut self, input: &str) {
        // Try to parse as a module (which includes definitions and expressions)
        let (module_opt, errors) = parse(input);

        // If parsing failed or no definitions, try as expression
        if !errors.is_empty() || module_opt.as_ref().map(|m| !Self::has_definitions(m)).unwrap_or(true) {
            // Try to evaluate as expression by wrapping in a temporary function
            self.try_eval_expression(input);
            return;
        }

        let module = match module_opt {
            Some(m) => m,
            None => {
                eprintln!("Failed to parse input");
                return;
            }
        };

        // Check if this is a function/type definition or an expression
        if Self::has_definitions(&module) {
            // Definition(s) - add to compiler
            if let Err(e) = self.compiler.add_module(&module, vec![], Arc::new(input.to_string()), "<repl>".to_string()) {
                eprintln!("Error: {}", e);
                return;
            }

            // Compile
            if let Err((e, filename, source)) = self.compiler.compile_all() {
                let source_error = e.to_source_error();
                source_error.eprint(&filename, &source);
                return;
            }

            // Sync VM
            self.sync_vm();

            // Report what was defined
            for fn_def in Self::get_fn_defs(&module) {
                let name = &fn_def.name.node;
                if let Some(sig) = self.compiler.get_function_signature(name) {
                    println!("{} :: {}", name, sig);
                } else {
                    println!("Defined {}", name);
                }
            }
            for type_def in Self::get_type_defs(&module) {
                println!("Defined type {}", type_def.full_name());
            }
        }
    }

    /// Try to evaluate input as an expression
    fn try_eval_expression(&mut self, input: &str) {
        // Use a unique name for each evaluation to avoid caching issues
        self.eval_counter += 1;
        let eval_name = format!("__repl_eval_{}__", self.eval_counter);

        // Wrap in a temporary function and execute
        let wrapper = format!("{}() = {}", eval_name, input);
        let (wrapper_module_opt, errors) = parse(&wrapper);

        if !errors.is_empty() {
            let source_errors = parse_errors_to_source_errors(&errors);
            eprint_errors(&source_errors, "<repl>", &wrapper);
            return;
        }

        let wrapper_module = match wrapper_module_opt {
            Some(m) => m,
            None => {
                eprintln!("Failed to parse expression");
                return;
            }
        };

        // Add wrapper function temporarily
        if let Err(e) = self.compiler.add_module(&wrapper_module, vec![], Arc::new(wrapper.clone()), "<repl>".to_string()) {
            eprintln!("Error: {}", e);
            return;
        }

        // Compile
        if let Err((e, filename, source)) = self.compiler.compile_all() {
            let source_error = e.to_source_error();
            source_error.eprint(&filename, &source);
            return;
        }

        // Sync VM and execute
        self.sync_vm();

        if let Some(func) = self.compiler.get_all_functions().get(&eval_name) {
            match self.vm.run(func.clone()) {
                Ok(Some(val)) => {
                    if !val.is_unit() {
                        println!("{}", val.display());
                    }
                }
                Ok(None) => {
                    // VM returned None, try direct call if possible
                }
                Err(e) => {
                    eprintln!("Runtime error: {}", e);
                }
            }
        } else {
            eprintln!("Internal error: evaluation function not found");
        }
    }

    /// Sync the VM with the current compiler state
    fn sync_vm(&mut self) {
        // Register all functions
        for (name, func) in self.compiler.get_all_functions() {
            self.vm.register_function(&name, func.clone());
        }

        // Set function list
        self.vm.set_function_list(self.compiler.get_function_list());

        // Register types
        for (name, type_val) in self.compiler.get_vm_types() {
            self.vm.register_type(&name, type_val);
        }

        // JIT compile if enabled
        if self.config.enable_jit {
            let function_list = self.compiler.get_function_list();
            if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
                for idx in 0..function_list.len() {
                    jit.queue_compilation(idx as u16);
                }
                if let Ok(compiled) = jit.process_queue(&function_list) {
                    if compiled > 0 {
                        for idx in 0..function_list.len() {
                            if let Some(jit_fn) = jit.get_int_function_0(idx as u16) {
                                self.vm.register_jit_int_function_0(idx as u16, jit_fn);
                            }
                            if let Some(jit_fn) = jit.get_int_function(idx as u16) {
                                self.vm.register_jit_int_function(idx as u16, jit_fn);
                            }
                            if let Some(jit_fn) = jit.get_int_function_2(idx as u16) {
                                self.vm.register_jit_int_function_2(idx as u16, jit_fn);
                            }
                            if let Some(jit_fn) = jit.get_int_function_3(idx as u16) {
                                self.vm.register_jit_int_function_3(idx as u16, jit_fn);
                            }
                            if let Some(jit_fn) = jit.get_int_function_4(idx as u16) {
                                self.vm.register_jit_int_function_4(idx as u16, jit_fn);
                            }
                            if let Some(jit_fn) = jit.get_loop_int64_array_function(idx as u16) {
                                self.vm.register_jit_loop_array_function(idx as u16, jit_fn);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Result of handling a command
enum CommandResult {
    Continue,
    Quit,
    Error(String),
}

/// Recursively visit directories and find .nos files
fn visit_dirs(dir: &std::path::Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else if let Some(ext) = path.extension() {
                if ext == "nos" {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}
