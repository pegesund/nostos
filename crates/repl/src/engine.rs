//! Core REPL engine logic (UI-agnostic).

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use nostos_compiler::compile::Compiler;
use nostos_jit::{JitCompiler, JitConfig};
use crate::CallGraph;
use nostos_syntax::ast::{Item, Pattern};
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::parallel::{ParallelVM, ParallelConfig};

/// An item in the browser
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BrowserItem {
    Module(String),
    Function { name: String, signature: String },
    Type { name: String },
    Trait { name: String },
    Variable { name: String, mutable: bool },
}

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

/// Information about a REPL variable binding
#[derive(Clone)]
struct VarBinding {
    /// The thunk function name (e.g., "__repl_var_x__")
    thunk_name: String,
    /// Whether the variable was declared with `var` (mutable)
    mutable: bool,
    /// Type annotation if provided
    type_annotation: Option<String>,
}

/// The REPL state
pub struct ReplEngine {
    compiler: Compiler,
    vm: ParallelVM,
    loaded_files: Vec<PathBuf>,
    config: ReplConfig,
    stdlib_path: Option<PathBuf>,
    call_graph: CallGraph,
    eval_counter: u64,
    /// Current active module (default: "repl")
    current_module: String,
    /// Module name -> source file path (for file-backed modules)
    module_sources: HashMap<String, PathBuf>,
    /// Variable bindings: name -> VarBinding
    var_bindings: HashMap<String, VarBinding>,
    /// Counter for unique variable thunk names
    var_counter: u64,
}

impl ReplEngine {
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
            current_module: "repl".to_string(),
            module_sources: HashMap::new(),
            var_bindings: HashMap::new(),
            var_counter: 0,
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
            // Format errors to string
            let _source_errors = parse_errors_to_source_errors(&errors);
            // eprint_errors writes to stderr. We want to capture it or return error string.
            // For now, return generic error, but ideally we should format it.
            // But source_errors.eprint prints.
            // Let's just return "Parse errors". The UI can't easily see details unless we change API.
            // Note: `Repl` in CLI printed errors.
            return Err("Parse errors (check console/logs)".to_string());
        }

        let module = module_opt.ok_or("Failed to parse file")?;

        // Derive module name
        let module_name = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

        // Extract top-level bindings from the module before compilation
        // We need to create thunk functions for each binding so they're accessible
        let mut bindings_to_add: Vec<(String, bool, String)> = Vec::new();
        for item in &module.items {
            if let Item::Binding(binding) = item {
                // Extract variable name from pattern (only simple variable patterns)
                if let Pattern::Var(ident) = &binding.pattern {
                    let name = ident.node.clone();
                    let mutable = binding.mutable;
                    // Extract the expression source text using its span
                    let expr_span = binding.value.span();
                    let expr_text = if expr_span.end <= source.len() {
                        source[expr_span.start..expr_span.end].to_string()
                    } else {
                        // Fallback: shouldn't happen, but just in case
                        continue;
                    };
                    bindings_to_add.push((name, mutable, expr_text));
                }
            }
        }

        // Add to compiler
        self.compiler.add_module(&module, vec![], Arc::new(source.clone()), path_str.to_string())
            .map_err(|e| format!("Compilation error: {}", e))?;

        // Compile all bodies
        if let Err((e, _filename, _source)) = self.compiler.compile_all() {
            return Err(format!("Compilation error: {}", e));
        }

        // Now create thunk functions for each binding
        for (name, mutable, expr_text) in bindings_to_add {
            self.var_counter += 1;
            let thunk_name = format!("__file_var_{}_{}", name, self.var_counter);

            let wrapper = format!("{}() = {}", thunk_name, expr_text);
            let (wrapper_module_opt, errors) = parse(&wrapper);

            if errors.is_empty() {
                if let Some(wrapper_module) = wrapper_module_opt {
                    if self.compiler.add_module(&wrapper_module, vec![], Arc::new(wrapper.clone()), "<file-binding>".to_string()).is_ok() {
                        if self.compiler.compile_all().is_ok() {
                            self.var_bindings.insert(name.clone(), VarBinding {
                                thunk_name,
                                mutable,
                                type_annotation: None,
                            });
                        }
                    }
                }
            }
        }

        // Update VM
        self.sync_vm();

        // Track loaded file
        if !self.loaded_files.contains(&path) {
            self.loaded_files.push(path.clone());
        }

        // Track module source
        if !module_name.is_empty() {
            self.module_sources.insert(module_name, path);
        }

        Ok(())
    }

    /// Reload all previously loaded files
    pub fn reload_files(&mut self) -> Result<usize, String> {
        let files = self.loaded_files.clone();
        let count = files.len();

        // Reset compiler (but keep stdlib)
        self.compiler = Compiler::new_empty();
        self.call_graph = CallGraph::new();
        if let Err(e) = self.load_stdlib() {
            return Err(format!("Warning: Failed to reload stdlib: {}", e));
        }

        self.loaded_files.clear();

        for path in files {
            let path_str = path.to_string_lossy().to_string();
            self.load_file(&path_str)?;
        }

        Ok(count)
    }

    /// Check if input is a variable binding
    pub fn is_var_binding(input: &str) -> Option<(String, bool, String)> {
        let input = input.trim();

        // Check for "var name = expr" pattern
        if input.starts_with("var ") {
            let rest = input[4..].trim();
            if let Some(eq_pos) = rest.find('=') {
                let name = rest[..eq_pos].trim();
                let expr = rest[eq_pos + 1..].trim();
                if !name.contains('(') && !name.is_empty() && !expr.is_empty() {
                    return Some((name.to_string(), true, expr.to_string()));
                }
            }
        }

        // Check for "name = expr" pattern
        if let Some(eq_pos) = input.find('=') {
            let before_eq = if eq_pos > 0 { input.chars().nth(eq_pos - 1) } else { None };
            let after_eq = input.chars().nth(eq_pos + 1);

            let is_comparison = matches!(before_eq, Some('!' | '<' | '>' | '='))
                || matches!(after_eq, Some('=') | Some('>'));

            if eq_pos > 0 && !is_comparison {
                let name = input[..eq_pos].trim();
                let expr = input[eq_pos + 1..].trim();
                if !name.contains('(') && !name.is_empty() && !expr.is_empty() {
                    if let Some(first_char) = name.chars().next() {
                        if first_char.is_lowercase() || first_char == '_' {
                            return Some((name.to_string(), false, expr.to_string()));
                        }
                    }
                }
            }
        }

        None
    }

    /// Define a variable binding
    fn define_var(&mut self, name: &str, mutable: bool, expr: &str) -> Result<String, String> {
        self.var_counter += 1;
        let thunk_name = format!("__repl_var_{}_{}", name, self.var_counter);

        let wrapper = format!("{}() = {}", thunk_name, expr);
        let (wrapper_module_opt, errors) = parse(&wrapper);

        if !errors.is_empty() {
            return Err("Parse error in variable definition".to_string());
        }

        let wrapper_module = wrapper_module_opt.ok_or("Failed to parse expression")?;

        self.compiler.add_module(&wrapper_module, vec![], Arc::new(wrapper.clone()), "<repl>".to_string())
            .map_err(|e| format!("Error: {}", e))?;

        if let Err((e, _, _)) = self.compiler.compile_all() {
            return Err(format!("Compilation error: {}", e));
        }

        self.sync_vm();

        self.var_bindings.insert(name.to_string(), VarBinding {
            thunk_name,
            mutable,
            type_annotation: None,
        });

        let mutability = if mutable { "var " } else { "" };
        Ok(format!("{}{} = {}", mutability, name, expr))
    }

    /// Process input (eval or define). Returns output string or error.
    pub fn eval(&mut self, input: &str) -> Result<String, String> {
        let input = input.trim();
        // Check for variable binding
        if let Some((name, mutable, expr)) = Self::is_var_binding(input) {
            return self.define_var(&name, mutable, &expr);
        }

        let (module_opt, errors) = parse(input);

        if !errors.is_empty() || module_opt.as_ref().map(|m| !Self::has_definitions(m)).unwrap_or(true) {
            return self.eval_expression_inner(input);
        }

        let module = module_opt.ok_or("Failed to parse input")?;

        if Self::has_definitions(&module) {
            let module_path = if self.current_module == "repl" {
                vec![]
            } else {
                self.current_module.split('.').map(String::from).collect()
            };

            self.compiler.add_module(&module, module_path.clone(), Arc::new(input.to_string()), "<repl>".to_string())
                .map_err(|e| format!("Error: {}", e))?;

            if let Err((e, _, _)) = self.compiler.compile_all() {
                return Err(format!("Compilation error: {}", e));
            }

            self.sync_vm();

            let prefix = if module_path.is_empty() {
                String::new()
            } else {
                format!("{}.", module_path.join("."))
            };

            let mut output = String::new();
            let mut defined_fns = HashSet::new();
            for fn_def in Self::get_fn_defs(&module) {
                defined_fns.insert(fn_def.name.node.clone());
            }

            let all_functions = self.compiler.get_function_names();

            for name in defined_fns {
                let base_name = format!("{}{}", prefix, name);
                
                // Find matching keys in all_functions (overloads)
                let mut matches = Vec::new();
                for key in &all_functions {
                    if key == &base_name || key.starts_with(&format!("{}/", base_name)) {
                        matches.push(key.clone());
                    }
                }
                
                if matches.is_empty() {
                     output.push_str(&format!("Defined {}\n", base_name));
                } else {
                    for key in matches {
                        if let Some(sig) = self.compiler.get_function_signature(&key) {
                             output.push_str(&format!("{} :: {}\n", name, sig)); 
                        } else {
                             output.push_str(&format!("Defined {}\n", key));
                        }
                    }
                }
            }
            for type_def in Self::get_type_defs(&module) {
                let type_name = type_def.full_name();
                output.push_str(&format!("Defined type {}{}\n", prefix, type_name));
            }
            return Ok(output.trim_end().to_string());
        }
        
        Ok(String::new())
    }

    fn eval_expression_inner(&mut self, input: &str) -> Result<String, String> {
        self.eval_counter += 1;
        let eval_name = format!("__repl_eval_{}__", self.eval_counter);

        let bindings_preamble = if self.var_bindings.is_empty() {
            String::new()
        } else {
            let bindings: Vec<String> = self.var_bindings
                .iter()
                .map(|(name, binding)| format!("{} = {}()", name, binding.thunk_name))
                .collect();
            bindings.join("\n    ") + "\n    "
        };

        let wrapper = if bindings_preamble.is_empty() {
            format!("{}() = {}", eval_name, input)
        } else {
            format!("{}() = {{\n    {}{}\n}}", eval_name, bindings_preamble, input)
        };
        
        let (wrapper_module_opt, errors) = parse(&wrapper);

        if !errors.is_empty() {
            return Err(format!("Parse error: {:?}", errors));
        }

        let wrapper_module = wrapper_module_opt.ok_or("Failed to parse expression")?;

        self.compiler.add_module(&wrapper_module, vec![], Arc::new(wrapper.clone()), "<repl>".to_string())
            .map_err(|e| format!("Error: {}", e))?;

        if let Err((e, _, _)) = self.compiler.compile_all() {
            return Err(format!("Compilation error: {}", e));
        }

        self.sync_vm();

        if let Some(func) = self.compiler.get_function(&eval_name) {
            match self.vm.run(func.clone()) {
                Ok(result) => {
                    let mut output = String::new();

                    // Add captured output (from println, print, etc.)
                    for line in &result.output {
                        output.push_str(line);
                        output.push('\n');
                    }

                    // Add return value (if not Unit)
                    if let Some(val) = result.value {
                        if !val.is_unit() {
                            output.push_str(&val.display());
                        }
                    }

                    Ok(output.trim_end().to_string())
                }
                Err(e) => Err(format!("Runtime error: {}", e)),
            }
        } else {
            Err("Internal error: evaluation function not found".to_string())
        }
    }

    fn sync_vm(&mut self) {
        for (name, func) in self.compiler.get_all_functions() {
            self.vm.register_function(&name, func.clone());
        }
        self.vm.set_function_list(self.compiler.get_function_list());
        for (name, type_val) in self.compiler.get_vm_types() {
            self.vm.register_type(&name, type_val);
        }

        if self.config.enable_jit {
            let function_list = self.compiler.get_function_list();
            if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
                for idx in 0..function_list.len() {
                    jit.queue_compilation(idx as u16);
                }
                if let Ok(compiled) = jit.process_queue(&function_list) {
                    if compiled > 0 {
                        // Register JIT functions (same as in CLI)
                        // This logic is duplicated, ideally should be shared/helper
                        for idx in 0..function_list.len() {
                            if let Some(jit_fn) = jit.get_int_function_0(idx as u16) {
                                self.vm.register_jit_int_function_0(idx as u16, jit_fn);
                            }
                            // ... (abbreviated for now, assume single function call if refactored)
                        }
                    }
                }
            }
        }
    }

    // Helpers
    fn has_definitions(module: &nostos_syntax::Module) -> bool {
        for item in &module.items {
            match item {
                Item::FnDef(_) | Item::TypeDef(_) => return true,
                _ => {}
            }
        }
        false
    }

    fn get_fn_defs(module: &nostos_syntax::Module) -> Vec<&nostos_syntax::ast::FnDef> {
        module.items.iter().filter_map(|item| {
            if let Item::FnDef(fn_def) = item {
                Some(fn_def)
            } else {
                None
            }
        }).collect()
    }

    fn get_type_defs(module: &nostos_syntax::Module) -> Vec<&nostos_syntax::ast::TypeDef> {
        module.items.iter().filter_map(|item| {
            if let Item::TypeDef(type_def) = item {
                Some(type_def)
            } else {
                None
            }
        }).collect()
    }
    
    // Introspection methods returning String instead of printing
    pub fn get_functions(&self) -> Vec<String> {
        let mut functions: Vec<_> = self.compiler.get_function_names().into_iter().map(String::from).collect();
        functions.sort();
        functions
    }

    pub fn get_types(&self) -> Vec<String> {
        let mut types: Vec<_> = self.compiler.get_type_names().into_iter().map(String::from).collect();
        types.sort();
        types
    }

    pub fn browse(&self, module_filter: Option<&str>) -> String {
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
                return format!("No functions found in module '{}'", filter);
            } else {
                return "No functions defined".to_string();
            }
        }

        let mut output = String::new();
        for name in functions {
            if let Some(sig) = self.compiler.get_function_signature(name) {
                output.push_str(&format!("  {} :: {}\n", name, sig));
            } else {
                output.push_str(&format!("  {}\n", name));
            }
        }
        output.trim_end().to_string()
    }

    pub fn get_info(&self, name: &str) -> String {
        // Try as function first
        if let Some(fn_def) = self.compiler.get_fn_def(name) {
            let mut output = format!("{}  (function)\n", name);

            // Signature
            let sig = fn_def.signature();
            if !sig.is_empty() && sig != "?" {
                output.push_str(&format!("  :: {}\n", sig));
            }

            // Doc comment
            if let Some(doc) = &fn_def.doc {
                output.push('\n');
                for line in doc.lines() {
                    output.push_str(&format!("  {}\n", line));
                }
            }

            // Module/file info
            if let Some(source) = self.compiler.get_function_source(name) {
                let lines = source.lines().count();
                output.push('\n');
                output.push_str(&format!("  Defined in {} lines\n", lines));
            }

            // Dependencies
            let deps = self.call_graph.direct_dependencies(name);
            if !deps.is_empty() {
                output.push('\n');
                let mut deps_vec: Vec<_> = deps.iter().collect();
                deps_vec.sort();
                output.push_str(&format!("  Depends on: {}\n", deps_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
            }

            return output;
        }

        // Try as type
        if let Some(type_def) = self.compiler.get_type_def(name) {
            let mut output = format!("{}  (type)\n", type_def.full_name());

            // Type body
            let body = type_def.body_string();
            if !body.is_empty() {
                output.push_str(&format!("  = {}\n", body));
            }

            // Doc comment
            if let Some(doc) = &type_def.doc {
                output.push('\n');
                for line in doc.lines() {
                    output.push_str(&format!("  {}\n", line));
                }
            }

            // Derived traits
            if !type_def.deriving.is_empty() {
                let traits: Vec<_> = type_def.deriving.iter().map(|t| t.node.as_str()).collect();
                output.push('\n');
                output.push_str(&format!("  Deriving: {}\n", traits.join(", ")));
            }

            return output;
        }

        // Try as trait
        let implementors = self.compiler.get_trait_implementors(name);
        if !implementors.is_empty() {
            let mut output = format!("{}  (trait)\n\n  Implemented by:\n", name);
            for ty in implementors {
                output.push_str(&format!("    - {}\n", ty));
            }
            return output;
        }

        format!("Not found: {}", name)
    }

    pub fn get_source(&self, name: &str) -> String {
        if let Some(source) = self.compiler.get_function_source(name) {
            return source.to_string();
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

            return output;
        }
        format!("Not found: {}", name)
    }

    pub fn get_deps(&self, name: &str) -> String {
        let deps = self.call_graph.direct_dependencies(name);
        if deps.is_empty() {
            return format!("{} has no dependencies", name);
        }
        let mut output = format!("{} depends on:\n", name);
        let mut deps_vec: Vec<_> = deps.iter().collect();
        deps_vec.sort();
        for dep in deps_vec {
            output.push_str(&format!("  {}\n", dep));
        }
        output.trim_end().to_string()
    }

    pub fn get_rdeps(&self, name: &str) -> String {
        let rdeps = self.call_graph.direct_dependents(name);
        if rdeps.is_empty() {
            return format!("{} is not called by any function", name);
        }
        let mut output = format!("{} is called by:\n", name);
        let mut rdeps_vec: Vec<_> = rdeps.iter().collect();
        rdeps_vec.sort();
        for rdep in rdeps_vec {
            output.push_str(&format!("  {}\n", rdep));
        }
        output.trim_end().to_string()
    }

    pub fn get_traits(&self) -> Vec<String> {
        let mut traits: Vec<_> = self.compiler.get_trait_names().into_iter().map(String::from).collect();
        traits.sort();
        traits
    }

    pub fn switch_module(&mut self, module_name: &str) -> String {
        let module_exists = self.compiler.module_exists(module_name);

        if module_exists {
            self.current_module = module_name.to_string();
            let mut msg = format!("Switched to module '{}'", module_name);
            if let Some(path) = self.module_sources.get(module_name) {
                msg.push_str(&format!("\nSource file: {}", path.display()));
            }
            msg
        } else if module_name == "repl" {
            self.current_module = "repl".to_string();
            "Switched to module 'repl'".to_string()
        } else {
            self.current_module = module_name.to_string();
            format!("Created new module '{}' (not file-backed)", module_name)
        }
    }

    pub fn get_current_module(&self) -> &str {
        &self.current_module
    }

    pub fn get_vars(&self) -> String {
        if self.var_bindings.is_empty() {
            return "No variable bindings".to_string();
        }

        let mut output = format!("Variable bindings ({}):\n", self.var_bindings.len());
        let mut names: Vec<_> = self.var_bindings.keys().collect();
        names.sort();
        for name in names {
            let binding = &self.var_bindings[name];
            let mutability = if binding.mutable { "var " } else { "" };
            let type_str = binding.type_annotation.as_ref()
                .map(|t| format!(": {}", t))
                .unwrap_or_default();
            output.push_str(&format!("  {}{}{}\n", mutability, name, type_str));
        }
        output.trim_end().to_string()
    }

    /// Get browser items at a given module path
    /// If path is empty, returns top-level modules and items
    pub fn get_browser_items(&self, path: &[String]) -> Vec<BrowserItem> {
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{}.", path.join("."))
        };
        let prefix_len = prefix.len();

        let mut modules: BTreeSet<String> = BTreeSet::new();
        let mut functions: BTreeSet<(String, String)> = BTreeSet::new();
        let mut types: BTreeSet<String> = BTreeSet::new();
        let mut traits: BTreeSet<String> = BTreeSet::new();

        // Process functions
        for name in self.compiler.get_function_names() {
            // Skip internal names
            if name.starts_with("__") {
                continue;
            }

            // Extract base name (without signature suffix)
            let base_name = if let Some(slash_pos) = name.find('/') {
                &name[..slash_pos]
            } else {
                name
            };

            if prefix.is_empty() {
                // At root level
                if let Some(dot_pos) = base_name.find('.') {
                    // Has a module prefix - extract first component
                    modules.insert(base_name[..dot_pos].to_string());
                } else {
                    // Direct function at root
                    let sig = self.compiler.get_function_signature(name).unwrap_or_default();
                    functions.insert((base_name.to_string(), sig));
                }
            } else if base_name.starts_with(&prefix) {
                // Under our path
                let rest = &base_name[prefix_len..];
                if let Some(dot_pos) = rest.find('.') {
                    // Has more path components - it's a submodule
                    modules.insert(rest[..dot_pos].to_string());
                } else {
                    // Direct function in this module
                    let sig = self.compiler.get_function_signature(name).unwrap_or_default();
                    functions.insert((rest.to_string(), sig));
                }
            }
        }

        // Process types
        for name in self.compiler.get_type_names() {
            if prefix.is_empty() {
                if let Some(dot_pos) = name.find('.') {
                    modules.insert(name[..dot_pos].to_string());
                } else {
                    types.insert(name.to_string());
                }
            } else if name.starts_with(&prefix) {
                let rest = &name[prefix_len..];
                if let Some(dot_pos) = rest.find('.') {
                    modules.insert(rest[..dot_pos].to_string());
                } else {
                    types.insert(rest.to_string());
                }
            }
        }

        // Process traits
        for name in self.compiler.get_trait_names() {
            if prefix.is_empty() {
                if let Some(dot_pos) = name.find('.') {
                    modules.insert(name[..dot_pos].to_string());
                } else {
                    traits.insert(name.to_string());
                }
            } else if name.starts_with(&prefix) {
                let rest = &name[prefix_len..];
                if let Some(dot_pos) = rest.find('.') {
                    modules.insert(rest[..dot_pos].to_string());
                } else {
                    traits.insert(rest.to_string());
                }
            }
        }

        // Build result list: modules first, then variables (at root), then types, traits, functions
        let mut items = Vec::new();

        // Modules first
        for m in modules {
            items.push(BrowserItem::Module(m));
        }

        // Variables second (at root level only) - so user's REPL bindings are visible
        if path.is_empty() {
            let mut var_names: Vec<_> = self.var_bindings.keys().collect();
            var_names.sort();
            for name in var_names {
                let binding = &self.var_bindings[name];
                items.push(BrowserItem::Variable {
                    name: name.clone(),
                    mutable: binding.mutable,
                });
            }
        }

        // Types
        for t in types {
            items.push(BrowserItem::Type { name: t });
        }

        // Traits
        for t in traits {
            items.push(BrowserItem::Trait { name: t });
        }

        // Functions last
        for (name, sig) in functions {
            items.push(BrowserItem::Function { name, signature: sig });
        }

        items
    }

    /// Get the full qualified name for a browser item at a given path
    pub fn get_full_name(&self, path: &[String], item: &BrowserItem) -> String {
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{}.", path.join("."))
        };
        match item {
            BrowserItem::Module(name) => format!("{}{}", prefix, name),
            BrowserItem::Function { name, .. } => format!("{}{}", prefix, name),
            BrowserItem::Type { name } => format!("{}{}", prefix, name),
            BrowserItem::Trait { name } => format!("{}{}", prefix, name),
            BrowserItem::Variable { name, .. } => name.clone(),
        }
    }

    /// Get the value of a variable by evaluating its thunk (as string)
    pub fn get_var_value(&mut self, name: &str) -> Option<String> {
        if let Some(binding) = self.var_bindings.get(name) {
            let thunk_call = format!("{}()", binding.thunk_name);
            match self.eval(&thunk_call) {
                Ok(value) => Some(value),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    /// Get the raw Value of a variable for inspection
    pub fn get_var_value_raw(&mut self, name: &str) -> Option<nostos_vm::Value> {
        let binding = self.var_bindings.get(name)?.clone();
        let func = self.compiler.get_function(&binding.thunk_name)?;

        match self.vm.run(func) {
            Ok(result) => result.value.map(|v| v.to_value()),
            Err(_) => None,
        }
    }

    /// Check if a variable is mutable
    pub fn is_var_mutable(&self, name: &str) -> bool {
        self.var_bindings.get(name).map(|b| b.mutable).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_binding_detection() {
        // Test is_var_binding
        assert_eq!(ReplEngine::is_var_binding("a = 10"), Some(("a".to_string(), false, "10".to_string())));
        assert_eq!(ReplEngine::is_var_binding("var x = 5"), Some(("x".to_string(), true, "5".to_string())));
        assert_eq!(ReplEngine::is_var_binding("foo = bar + 1"), Some(("foo".to_string(), false, "bar + 1".to_string())));
        // Not variable bindings
        assert_eq!(ReplEngine::is_var_binding("a == 10"), None);
        assert_eq!(ReplEngine::is_var_binding("f(x) = x + 1"), None);
    }

    #[test]
    fn test_var_bindings_in_browser() {
        let config = ReplConfig { enable_jit: false, num_threads: 1 };
        let mut engine = ReplEngine::new(config);

        // Define some variables
        let result = engine.eval("a = 10");
        println!("eval a = 10: {:?}", result);

        let result = engine.eval("b = 20");
        println!("eval b = 20: {:?}", result);

        // Check var_bindings directly
        println!("var_bindings.len() = {}", engine.var_bindings.len());
        for (k, v) in &engine.var_bindings {
            println!("  {} -> {} (mutable={})", k, v.thunk_name, v.mutable);
        }

        // Get browser items at root
        let items = engine.get_browser_items(&[]);
        println!("Browser items count: {}", items.len());

        let vars: Vec<_> = items.iter().filter(|item| matches!(item, BrowserItem::Variable { .. })).collect();
        println!("Variable items: {:?}", vars);

        assert!(engine.var_bindings.contains_key("a"), "var_bindings should contain 'a'");
        assert!(engine.var_bindings.contains_key("b"), "var_bindings should contain 'b'");
        assert!(vars.len() >= 2, "Browser should show at least 2 variables, got {}", vars.len());
    }
}

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
