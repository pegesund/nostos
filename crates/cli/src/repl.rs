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

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::borrow::Cow;

use nostos_compiler::compile::Compiler;
use nostos_jit::{JitCompiler, JitConfig};
use nostos_repl::CallGraph;
use nostos_syntax::ast::Item;
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_syntax::lexer::{Token, lex};
use nostos_vm::parallel::{ParallelVM, ParallelConfig};

use reedline::{
    Reedline, Signal, FileBackedHistory, Highlighter, StyledText, 
    Prompt, PromptEditMode, PromptHistorySearch, Completer, Suggestion, Span,
    ReedlineMenu, ColumnarMenu, MenuBuilder,
    default_emacs_keybindings, KeyModifiers, KeyCode, ReedlineEvent, Emacs
};
use nu_ansi_term::{Color, Style};

/// Syntax highlighter for Nostos
pub struct NostosHighlighter;

impl Highlighter for NostosHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();
        let mut last_idx = 0;

        for (token, span) in lex(line) {
            // Add whitespace/skipped text before token
            if span.start > last_idx {
                styled.push((Style::new(), line[last_idx..span.start].to_string()));
            }

            let style = match token {
                // Keywords
                Token::Type | Token::Var | Token::Mvar | Token::If | Token::Then | Token::Else |
                Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
                Token::Use | Token::Private | Token::Pub | Token::SelfKw | Token::SelfType |
                Token::Try | Token::Catch | Token::Finally | Token::Do |
                Token::While | Token::For | Token::To | Token::Break | Token::Continue |
                Token::Spawn | Token::SpawnLink | Token::SpawnMonitor | Token::Receive | Token::After |
                Token::Panic | Token::Extern | Token::From | Token::Test | Token::Deriving | Token::Quote =>
                    Style::new().fg(Color::Magenta).bold(),

                // Boolean literals
                Token::True | Token::False => Style::new().fg(Color::Yellow).bold(),

                // Numeric literals
                Token::Int(_) | Token::HexInt(_) | Token::BinInt(_) | 
                Token::Int8(_) | Token::Int16(_) | Token::Int32(_) |
                Token::UInt8(_) | Token::UInt16(_) | Token::UInt32(_) | Token::UInt64(_) |
                Token::BigInt(_) | Token::Float(_) | Token::Float32(_) | Token::Decimal(_) => 
                    Style::new().fg(Color::Yellow),

                // String/Char literals
                Token::String(_) | Token::Char(_) => Style::new().fg(Color::Green),

                // Comments (Hash is start of set literal or comment? Lexer handles comments separately in logos skip)
                // Wait, logos skips comments, so `lex` iterator won't yield them as tokens?
                // `nostos_syntax::lexer::lex` filters `tok.ok()`.
                // If logos skips, they are not yielded.
                // We need to handle gaps as potential comments or whitespace.
                // But we don't know if it's a comment or whitespace from `lex`.
                // However, highlighting comments is nice.
                // If `lex` skips comments, we can't style them easily unless we scan gaps.
                // For now, gaps are unstyled (default).
                
                // Operators
                Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent | Token::StarStar |
                Token::EqEq | Token::NotEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq |
                Token::AndAnd | Token::OrOr | Token::Bang | Token::PlusPlus | Token::PipeRight |
                Token::Eq | Token::PlusEq | Token::MinusEq | Token::StarEq | Token::SlashEq |
                Token::LeftArrow | Token::RightArrow | Token::FatArrow | Token::Caret | Token::Dollar | Token::Question =>
                    Style::new().fg(Color::Rgb(255, 165, 0)), // Orange - visible on dark backgrounds

                // Delimiters
                Token::LParen | Token::RParen | Token::LBracket | Token::RBracket |
                Token::LBrace | Token::RBrace | Token::Comma | Token::Colon | Token::ColonColon | Token::Dot |
                Token::Pipe | Token::Hash =>
                    Style::new().fg(Color::White),

                // Identifiers
                Token::UpperIdent(_) => Style::new().fg(Color::Yellow), // Types/Constructors
                Token::LowerIdent(_) => Style::new().fg(Color::White),  // Variables/Functions
                
                Token::Underscore => Style::new().fg(Color::Rgb(150, 150, 150)), // Lighter gray - visible on dark backgrounds
                Token::Newline => Style::new(),
                Token::Comment | Token::MultiLineComment => Style::new().fg(Color::Rgb(150, 150, 150)), // Lighter gray
            };

            styled.push((style, line[span.clone()].to_string()));
            last_idx = span.end;
        }

        // Add remaining text
        if last_idx < line.len() {
            styled.push((Style::new(), line[last_idx..].to_string()));
        }

        styled
    }
}

#[derive(Clone)]
struct NostosPrompt {
    left: String,
}

impl NostosPrompt {
    fn new(left: String) -> Self {
        Self { left }
    }
}

impl Prompt for NostosPrompt {
    fn render_prompt_left(&self) -> Cow<str> {
        Cow::Borrowed(&self.left)
    }

    fn render_prompt_right(&self) -> Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<str> {
        Cow::Borrowed(".. ")
    }

    fn render_prompt_history_search_indicator(&self, _history_search: PromptHistorySearch) -> Cow<str> {
        Cow::Borrowed("(reverse-search) ")
    }
}

struct NostosCompleter;

impl Completer for NostosCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let (start, word) = find_word_at_pos(line, pos);
        let span = Span::new(start, pos);

        if word.starts_with(':') {
            let commands = vec![
                ":help", ":quit", ":exit", ":load", ":reload", ":browse", ":info",
                ":view", ":type", ":deps", ":rdeps", ":functions", ":types",
                ":traits", ":module", ":vars", ":bindings",
                ":h", ":q", ":l", ":r", ":b", ":i", ":v", ":t", ":d", ":rd", ":fns", ":m"
            ];
            
            commands.iter()
                .filter(|cmd| cmd.starts_with(word))
                .map(|cmd| Suggestion {
                    value: cmd.to_string(),
                    description: None,
                    style: None,
                    extra: None,
                    span,
                    append_whitespace: true,
                })
                .collect()
        } else if !word.is_empty() {
            // Keywords
            let keywords = vec![
                "type", "var", "if", "then", "else", "match", "when", "trait", "module", "end",
                "use", "private", "pub", "self", "Self", "try", "catch", "finally", "do",
                "while", "for", "to", "break", "continue", "spawn", "receive", "after", "panic",
                "extern", "test", "deriving", "quote", "true", "false"
            ];

            keywords.iter()
                .filter(|kw| kw.starts_with(word))
                .map(|kw| Suggestion {
                    value: kw.to_string(),
                    description: None,
                    style: None,
                    extra: None,
                    span,
                    append_whitespace: true,
                })
                .collect()
        } else {
            vec![]
        }
    }
}

fn find_word_at_pos(line: &str, pos: usize) -> (usize, &str) {
    let start = line[..pos].rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
        .map(|i| i + 1)
        .unwrap_or(0);
    (start, &line[start..pos])
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

/// The REPL state
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

pub struct Repl {
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

            // Track all stdlib function names for prelude imports
            let mut stdlib_functions: Vec<(String, String)> = Vec::new();

            for file_path in &stdlib_files {
                let source = fs::read_to_string(file_path)
                    .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e))?;
                let (module_opt, _) = parse(&source);
                if let Some(module) = module_opt {
                    // Build module path: stdlib.list, stdlib.json, etc.
                    let relative = file_path.strip_prefix(path).unwrap();
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

                    self.compiler.add_module(&module, components, Arc::new(source.clone()), file_path.to_str().unwrap().to_string())
                        .map_err(|e| format!("Failed to compile stdlib: {}", e))?;
                }
            }

            // Register prelude imports so stdlib functions are available without prefix
            for (local_name, qualified_name) in stdlib_functions {
                self.compiler.add_prelude_import(local_name, qualified_name);
            }

            // Compile all stdlib functions to populate source_code fields
            if let Err((e, _, _)) = self.compiler.compile_all() {
                return Err(format!("Failed to compile stdlib: {}", e));
            }

            self.stdlib_path = stdlib_path;
        }

        Ok(())
    }

    /// Run the main REPL loop
    pub fn run(&mut self) -> io::Result<()> {
        println!("Nostos REPL v{}", env!("CARGO_PKG_VERSION"));
        println!("Type :help for available commands, :quit to exit");
        println!();

        // Setup history
        let history = Box::new(
            FileBackedHistory::with_file(1000, ".nostos_history".into())
                .unwrap_or_else(|_| FileBackedHistory::new(1000).expect("Error creating history"))
        );

        // Create completion menu
        let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));

        // Setup keybindings
        let mut keybindings = default_emacs_keybindings();
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_string()),
                ReedlineEvent::MenuNext,
            ]),
        );
        let edit_mode = Box::new(Emacs::new(keybindings));

        // Create Reedline instance with highlighter, history, completer and menu
        let mut line_editor = Reedline::create()
            .with_history(history)
            .with_highlighter(Box::new(NostosHighlighter))
            .with_completer(Box::new(NostosCompleter))
            .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
            .with_edit_mode(edit_mode);

        // Buffer for multi-line input
        let mut input_buffer = String::new();
        let mut in_multiline = false;

        loop {
            // Set prompt based on state
            let p: Box<dyn Prompt> = if in_multiline {
                Box::new(NostosPrompt::new("... ".to_string()))
            } else {
                let prompt = if self.current_module == "repl" {
                    "nos> ".to_string()
                } else {
                    format!("{}> ", self.current_module)
                };
                Box::new(NostosPrompt::new(prompt))
            };

            let sig = line_editor.read_line(&*p);

            match sig {
                Ok(Signal::Success(line)) => {
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
                Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                    println!("Bye!");
                    break;
                }
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    break;
                }
            }
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

            ":module" | ":m" => {
                if args.is_empty() {
                    // Show current module
                    println!("Current module: {}", self.current_module);
                    if let Some(path) = self.module_sources.get(&self.current_module) {
                        println!("Source file: {}", path.display());
                    }
                } else {
                    // Switch to module
                    self.switch_module(args);
                }
                CommandResult::Continue
            }

            ":vars" | ":bindings" => {
                self.list_vars();
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
        println!("  :module [name], :m   Show/switch current module");
        println!("  :browse [module], :b List functions (optionally in a module)");
        println!("  :info <name>, :i     Show info about a function or type");
        println!("  :view <name>, :v     Show source code of a function");
        println!("  :type <expr>, :t     Show the type of an expression");
        println!("  :deps <fn>, :d       Show dependencies of a function");
        println!("  :rdeps <fn>, :rd     Show reverse dependencies (what calls this)");
        println!("  :functions, :fns     List all functions");
        println!("  :types               List all types");
        println!("  :traits              List all traits");
        println!("  :vars, :bindings     List variable bindings");
        println!();
        println!("Input:");
        println!("  <expression>         Evaluate an expression");
        println!("  <name> = <expr>      Bind a variable (immutable)");
        println!("  var <name> = <expr>  Bind a mutable variable");
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

        // Derive module name from file path (e.g., "foo/bar.nos" -> "bar")
        let module_name = path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

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
            self.loaded_files.push(path.clone());
        }

        // Track module source for file sync
        if !module_name.is_empty() {
            self.module_sources.insert(module_name, path);
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
        use nostos_compiler::compile::Compiler;

        // Try as function first - get all variants if overloaded
        let all_defs = self.compiler.get_all_fn_defs(name);

        if !all_defs.is_empty() {
            let is_overloaded = all_defs.len() > 1;

            for (full_name, fn_def) in &all_defs {
                let display_name = Compiler::function_name_display(full_name);
                println!("{}  (function)", display_name);

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
                if let Some(source) = self.compiler.get_function_source(full_name) {
                    let lines = source.lines().count();
                    println!();
                    println!("  Defined in {} lines", lines);
                }

                // Dependencies
                let deps = self.call_graph.direct_dependencies(full_name);
                if !deps.is_empty() {
                    println!();
                    let mut deps_vec: Vec<_> = deps.iter().collect();
                    deps_vec.sort();
                    let deps_display: Vec<_> = deps_vec.iter()
                        .map(|s| Compiler::function_name_display(s))
                        .collect();
                    println!("  Depends on: {}", deps_display.join(", "));
                }

                // Add separator between overloaded variants
                if is_overloaded {
                    println!();
                }
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
        // Get all variants of an overloaded function
        let variants = self.compiler.get_function_variants(name);
        if !variants.is_empty() {
            let is_overloaded = variants.len() > 1;
            for (full_name, display_name) in variants {
                if let Some(source) = self.compiler.get_function_source(&full_name) {
                    if is_overloaded {
                        println!("# {}", display_name);
                    }
                    println!("{}", source);
                    if is_overloaded {
                        println!();
                    }
                }
            }
            return;
        }

        // Try single function by exact name
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

    /// Show type of an expression or function
    fn show_type(&self, expr: &str) {
        let name = expr.trim();

        // First try as a function name
        if let Some(sig) = self.compiler.get_function_signature(name) {
            println!("{} :: {}", name, sig);
            return;
        }

        // Try with current module prefix
        if !self.current_module.is_empty() {
            let qualified_name = format!("{}::{}", self.current_module, name);
            if let Some(sig) = self.compiler.get_function_signature(&qualified_name) {
                println!("{} :: {}", name, sig);
                return;
            }
        }

        // Not found
        println!("Unknown: {}", name);
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
        let mut functions = self.compiler.get_function_names_display();
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

    /// Switch to a different module
    fn switch_module(&mut self, module_name: &str) {
        // Check if the module exists in the compiler (loaded from file)
        let module_exists = self.compiler.module_exists(module_name);

        if module_exists {
            self.current_module = module_name.to_string();
            println!("Switched to module '{}'", module_name);
            if let Some(path) = self.module_sources.get(module_name) {
                println!("Source file: {}", path.display());
            }
        } else if module_name == "repl" {
            // Always allow switching to repl module
            self.current_module = "repl".to_string();
            println!("Switched to module 'repl'");
        } else {
            // Create a new virtual module
            self.current_module = module_name.to_string();
            println!("Created new module '{}' (not file-backed)", module_name);
        }
    }

    /// List all variable bindings
    fn list_vars(&self) {
        if self.var_bindings.is_empty() {
            println!("No variable bindings");
            return;
        }

        println!("Variable bindings ({}):", self.var_bindings.len());
        let mut names: Vec<_> = self.var_bindings.keys().collect();
        names.sort();
        for name in names {
            let binding = &self.var_bindings[name];
            let mutability = if binding.mutable { "var " } else { "" };
            let type_str = binding.type_annotation.as_ref()
                .map(|t| format!(": {}", t))
                .unwrap_or_default();
            println!("  {}{}{}", mutability, name, type_str);
        }
    }

    /// Check if input is a variable binding (e.g., "x = 5" or "var y = 10")
    fn is_var_binding(input: &str) -> Option<(String, bool, String)> {
        let input = input.trim();

        // Check for "var name = expr" pattern
        if input.starts_with("var ") {
            let rest = input[4..].trim();
            if let Some(eq_pos) = rest.find('=') {
                let name = rest[..eq_pos].trim();
                let expr = rest[eq_pos + 1..].trim();
                // Make sure it's not a function definition (no parentheses in name)
                if !name.contains('(') && !name.is_empty() && !expr.is_empty() {
                    return Some((name.to_string(), true, expr.to_string()));
                }
            }
        }

        // Check for "name = expr" pattern (not a function definition)
        if let Some(eq_pos) = input.find('=') {
            // Make sure it's not ==, !=, <=, >=, or => operators
            let before_eq = if eq_pos > 0 { input.chars().nth(eq_pos - 1) } else { None };
            let after_eq = input.chars().nth(eq_pos + 1);

            let is_comparison = matches!(before_eq, Some('!' | '<' | '>' | '='))
                || matches!(after_eq, Some('=') | Some('>'));

            if eq_pos > 0 && !is_comparison {
                let name = input[..eq_pos].trim();
                let expr = input[eq_pos + 1..].trim();
                // Make sure it's not a function definition (no parentheses before =)
                // and the name is a valid identifier (lowercase start)
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
    fn define_var(&mut self, name: &str, mutable: bool, expr: &str) -> bool {
        self.var_counter += 1;
        let thunk_name = format!("__repl_var_{}_{}", name, self.var_counter);

        // Create a thunk function that returns the expression value
        let wrapper = format!("{}() = {}", thunk_name, expr);
        let (wrapper_module_opt, errors) = parse(&wrapper);

        if !errors.is_empty() {
            let source_errors = parse_errors_to_source_errors(&errors);
            eprint_errors(&source_errors, "<repl>", &wrapper);
            return false;
        }

        let wrapper_module = match wrapper_module_opt {
            Some(m) => m,
            None => {
                eprintln!("Failed to parse expression");
                return false;
            }
        };

        // Add the thunk function
        if let Err(e) = self.compiler.add_module(&wrapper_module, vec![], Arc::new(wrapper.clone()), "<repl>".to_string()) {
            eprintln!("Error: {}", e);
            return false;
        }

        // Compile
        if let Err((e, filename, source)) = self.compiler.compile_all() {
            let source_error = e.to_source_error();
            source_error.eprint(&filename, &source);
            return false;
        }

        // Sync VM
        self.sync_vm();

        // Store the binding
        self.var_bindings.insert(name.to_string(), VarBinding {
            thunk_name,
            mutable,
            type_annotation: None,
        });

        // Report the binding
        let mutability = if mutable { "var " } else { "" };
        println!("{}{} = {}", mutability, name, expr);

        true
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

    /// Preprocess input to support semicolons as statement separators.
    /// This allows multi-clause function definitions like: `f(0) = 0; f(n) = n * f(n-1)`
    fn preprocess_input(input: &str) -> String {
        // Replace semicolons with newlines, but be careful not to replace semicolons inside strings
        let mut result = String::with_capacity(input.len());
        let mut in_string = false;
        let mut escape_next = false;

        for ch in input.chars() {
            if escape_next {
                result.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => {
                    result.push(ch);
                    escape_next = true;
                }
                '"' => {
                    in_string = !in_string;
                    result.push(ch);
                }
                ';' if !in_string => {
                    result.push('\n');
                }
                _ => {
                    result.push(ch);
                }
            }
        }

        result
    }

    /// Process user input (expression or definition)
    fn process_input(&mut self, input: &str) {
        // Preprocess: convert semicolons to newlines (for multi-clause definitions)
        let input = Self::preprocess_input(input);
        let input = input.as_str();

        // Check for variable binding first (e.g., "x = 5" or "var y = 10")
        if let Some((name, mutable, expr)) = Self::is_var_binding(input) {
            self.define_var(&name, mutable, &expr);
            return;
        }

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
            // Determine module path based on current module
            let module_path = if self.current_module == "repl" {
                vec![]
            } else {
                // Split module name by . for nested modules
                self.current_module.split('.').map(String::from).collect()
            };

            // Definition(s) - add to compiler
            if let Err(e) = self.compiler.add_module(&module, module_path.clone(), Arc::new(input.to_string()), "<repl>".to_string()) {
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

            // Report what was defined (with module prefix if applicable)
            let prefix = if module_path.is_empty() {
                String::new()
            } else {
                format!("{}.", module_path.join("."))
            };

            for fn_def in Self::get_fn_defs(&module) {
                let name = &fn_def.name.node;
                let qualified_name = format!("{}{}", prefix, name);
                if let Some(sig) = self.compiler.get_function_signature(&qualified_name) {
                    println!("{} :: {}", qualified_name, sig);
                } else {
                    println!("Defined {}", qualified_name);
                }
            }
            for type_def in Self::get_type_defs(&module) {
                let type_name = type_def.full_name();
                println!("Defined type {}{}", prefix, type_name);
            }
        }
    }

    /// Try to evaluate input as an expression
    fn try_eval_expression(&mut self, input: &str) {
        if let Some(result) = self.eval_expression_inner(input) {
            println!("{}", result);
        }
    }

    /// Inner evaluation that returns the result as a string (for testing)
    fn eval_expression_inner(&mut self, input: &str) -> Option<String> {
        // Use a unique name for each evaluation to avoid caching issues
        self.eval_counter += 1;
        let eval_name = format!("__repl_eval_{}__", self.eval_counter);

        // Build variable bindings preamble
        let bindings_preamble = if self.var_bindings.is_empty() {
            String::new()
        } else {
            let bindings: Vec<String> = self.var_bindings
                .iter()
                .map(|(name, binding)| format!("{} = {}()", name, binding.thunk_name))
                .collect();
            bindings.join("\n    ") + "\n    "
        };

        // Wrap in a temporary function with variable bindings injected
        let wrapper = if bindings_preamble.is_empty() {
            format!("{}() = {}", eval_name, input)
        } else {
            format!("{}() = {{\n    {}{}\n}}", eval_name, bindings_preamble, input)
        };
        let (wrapper_module_opt, errors) = parse(&wrapper);

        if !errors.is_empty() {
            let source_errors = parse_errors_to_source_errors(&errors);
            eprint_errors(&source_errors, "<repl>", &wrapper);
            return None;
        }

        let wrapper_module = match wrapper_module_opt {
            Some(m) => m,
            None => {
                eprintln!("Failed to parse expression");
                return None;
            }
        };

        // Add wrapper function temporarily
        if let Err(e) = self.compiler.add_module(&wrapper_module, vec![], Arc::new(wrapper.clone()), "<repl>".to_string()) {
            eprintln!("Error: {}", e);
            return None;
        }

        // Compile
        if let Err((e, filename, source)) = self.compiler.compile_all() {
            let source_error = e.to_source_error();
            source_error.eprint(&filename, &source);
            return None;
        }

        // Sync VM and execute
        self.sync_vm();

        if let Some(func) = self.compiler.get_function(&eval_name) {
            match self.vm.run(func.clone()) {
                Ok(result) => {
                    // Output is already printed to stdout by the VM
                    // Return the value for display if not Unit
                    if let Some(val) = result.value {
                        if !val.is_unit() {
                            return Some(val.display());
                        }
                    }
                    return None;
                }
                Err(e) => {
                    eprintln!("Runtime error: {}", e);
                    return None;
                }
            }
        } else {
            eprintln!("Internal error: evaluation function not found");
            None
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

    // ===== Public API for testing =====

    /// Evaluate input and return the result as a string (for testing)
    /// This handles variable bindings, function definitions, and expressions.
    pub fn eval(&mut self, input: &str) -> Option<String> {
        // Preprocess: convert semicolons to newlines (for multi-clause definitions)
        let input = Self::preprocess_input(input);
        let input = input.as_str();

        // Check for variable binding first (e.g., "x = 5" or "var y = 10")
        if let Some((name, mutable, expr)) = Self::is_var_binding(input) {
            if self.define_var(&name, mutable, &expr) {
                return Some(format!("{} = <bound>", name));
            }
            return None;
        }

        // Try to parse as a module (which includes definitions and expressions)
        let (module_opt, errors) = parse(input);

        // If parsing failed or no definitions, try as expression
        if !errors.is_empty() || module_opt.as_ref().map(|m| !Self::has_definitions(m)).unwrap_or(true) {
            return self.eval_expression_inner(input);
        }

        let module = match module_opt {
            Some(m) => m,
            None => return None,
        };

        // Definition(s) - add to compiler
        let module_path = if self.current_module == "repl" {
            vec![]
        } else {
            self.current_module.split('.').map(String::from).collect()
        };

        if self.compiler.add_module(&module, module_path.clone(), Arc::new(input.to_string()), "<repl>".to_string()).is_err() {
            return None;
        }

        if self.compiler.compile_all().is_err() {
            return None;
        }

        self.sync_vm();

        // Return a summary of what was defined
        let prefix = if module_path.is_empty() {
            String::new()
        } else {
            format!("{}.", module_path.join("."))
        };

        let mut defined = Vec::new();
        for fn_def in Self::get_fn_defs(&module) {
            defined.push(format!("{}{}()", prefix, fn_def.name.node));
        }
        for type_def in Self::get_type_defs(&module) {
            defined.push(format!("type {}{}", prefix, type_def.full_name()));
        }

        if defined.is_empty() {
            None
        } else {
            Some(format!("Defined: {}", defined.join(", ")))
        }
    }

    /// Get all variable binding names
    pub fn get_var_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.var_bindings.keys().cloned().collect();
        names.sort();
        names
    }

    /// Check if a variable exists
    pub fn has_var(&self, name: &str) -> bool {
        self.var_bindings.contains_key(name)
    }

    /// Get the current module name
    pub fn get_current_module(&self) -> &str {
        &self.current_module
    }

    /// Set the current module (for testing)
    pub fn set_module(&mut self, name: &str) {
        self.switch_module(name);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_repl() -> Repl {
        let config = ReplConfig {
            enable_jit: false, // Disable JIT for faster tests
            num_threads: 1,
        };
        let mut repl = Repl::new(config);
        // Load stdlib for tests
        let _ = repl.load_stdlib();
        repl
    }

    #[test]
    fn test_simple_expression() {
        let mut repl = create_test_repl();
        assert_eq!(repl.eval("1 + 2"), Some("3".to_string()));
        assert_eq!(repl.eval("10 * 5"), Some("50".to_string()));
    }

    #[test]
    fn test_variable_binding() {
        let mut repl = create_test_repl();

        // Bind a variable
        let result = repl.eval("x = 42");
        assert!(result.is_some());
        assert!(repl.has_var("x"));
        assert_eq!(repl.get_var_names(), vec!["x"]);

        // Use the variable
        assert_eq!(repl.eval("x"), Some("42".to_string()));
        assert_eq!(repl.eval("x + 8"), Some("50".to_string()));
    }

    #[test]
    fn test_multiple_variables() {
        let mut repl = create_test_repl();

        repl.eval("a = 10");
        repl.eval("b = 20");
        repl.eval("c = 30");

        assert_eq!(repl.get_var_names(), vec!["a", "b", "c"]);
        assert_eq!(repl.eval("a + b + c"), Some("60".to_string()));
    }

    #[test]
    fn test_var_mutable_binding() {
        let mut repl = create_test_repl();

        // Bind with var keyword
        let result = repl.eval("var count = 100");
        assert!(result.is_some());
        assert!(repl.has_var("count"));

        // Verify value
        assert_eq!(repl.eval("count"), Some("100".to_string()));
    }

    #[test]
    fn test_variable_in_expression() {
        let mut repl = create_test_repl();

        repl.eval("x = 5");
        repl.eval("y = 10");

        // Variables should be accessible in expressions
        assert_eq!(repl.eval("x * y"), Some("50".to_string()));
        assert_eq!(repl.eval("x + y + 5"), Some("20".to_string()));
    }

    #[test]
    fn test_function_definition() {
        let mut repl = create_test_repl();

        // Define a function
        let result = repl.eval("double(n) = n * 2");
        assert!(result.is_some());

        // Call the function
        assert_eq!(repl.eval("double(21)"), Some("42".to_string()));
    }

    #[test]
    fn test_function_with_variables() {
        let mut repl = create_test_repl();

        // Define a variable
        repl.eval("factor = 3");

        // Variable should be accessible in expression (but not in function body directly)
        assert_eq!(repl.eval("factor * 10"), Some("30".to_string()));
    }

    #[test]
    fn test_module_switching() {
        let mut repl = create_test_repl();

        assert_eq!(repl.get_current_module(), "repl");

        repl.set_module("MyModule");
        assert_eq!(repl.get_current_module(), "MyModule");

        repl.set_module("repl");
        assert_eq!(repl.get_current_module(), "repl");
    }

    #[test]
    fn test_is_var_binding_detection() {
        // Should detect variable bindings
        assert!(Repl::is_var_binding("x = 5").is_some());
        assert!(Repl::is_var_binding("var y = 10").is_some());
        assert!(Repl::is_var_binding("myVar = 1 + 2").is_some());

        // Should NOT detect these as variable bindings
        assert!(Repl::is_var_binding("foo(x) = x * 2").is_none()); // function def
        assert!(Repl::is_var_binding("1 + 2").is_none()); // expression
        assert!(Repl::is_var_binding("x == 5").is_none()); // comparison
        assert!(Repl::is_var_binding("Type = Something").is_none()); // starts with uppercase
    }

    #[test]
    fn test_rebind_variable() {
        let mut repl = create_test_repl();

        repl.eval("x = 10");
        assert_eq!(repl.eval("x"), Some("10".to_string()));

        // Rebind to a new value
        repl.eval("x = 20");
        assert_eq!(repl.eval("x"), Some("20".to_string()));
    }

    #[test]
    fn test_string_variable() {
        let mut repl = create_test_repl();

        repl.eval(r#"name = "hello""#);
        assert!(repl.has_var("name"));
        // Note: string display doesn't include quotes
        assert_eq!(repl.eval("name"), Some("hello".to_string()));
    }

    #[test]
    fn test_list_variable() {
        let mut repl = create_test_repl();

        repl.eval("nums = [1, 2, 3]");
        assert!(repl.has_var("nums"));
        assert_eq!(repl.eval("nums"), Some("[1, 2, 3]".to_string()));
    }

    #[test]
    fn test_function_signature_type_variables() {
        let mut repl = create_test_repl();

        // Define an untyped function
        let result = repl.eval("madd(x, y) = x + y");
        assert!(result.is_some());
        // The signature should use type variables (a, b, c) instead of ?
        let result_str = result.unwrap();
        assert!(result_str.contains("madd()"), "Should define madd function: {}", result_str);

        // Function should work
        assert_eq!(repl.eval("madd(3, 4)"), Some("7".to_string()));
    }

    #[test]
    fn test_function_signature_with_types() {
        let mut repl = create_test_repl();

        // Define a typed function
        let result = repl.eval("typed_add(x: Int, y: Int) -> Int = x + y");
        assert!(result.is_some());

        // Function should work
        assert_eq!(repl.eval("typed_add(10, 20)"), Some("30".to_string()));
    }

    #[test]
    fn test_function_redefinition_basic() {
        let mut repl = create_test_repl();

        // Define function A
        repl.eval("get_value() = 10");
        assert_eq!(repl.eval("get_value()"), Some("10".to_string()));

        // Redefine function A with a different value
        repl.eval("get_value() = 42");
        assert_eq!(repl.eval("get_value()"), Some("42".to_string()));
    }

    #[test]
    fn test_function_redefinition_updates_callers() {
        let mut repl = create_test_repl();

        // Define function A that returns a constant
        repl.eval("base() = 10");

        // Define function B that calls A
        repl.eval("derived() = base() * 2");

        // Initially B should return 10 * 2 = 20
        assert_eq!(repl.eval("derived()"), Some("20".to_string()));

        // Redefine A to return a different value
        repl.eval("base() = 100");

        // Now B should return 100 * 2 = 200
        assert_eq!(repl.eval("derived()"), Some("200".to_string()));
    }

    #[test]
    fn test_function_redefinition_chain() {
        let mut repl = create_test_repl();

        // Create a chain: A -> B -> C
        repl.eval("a() = 5");
        repl.eval("b() = a() + 1");  // b = 6
        repl.eval("c() = b() * 2");  // c = 12

        assert_eq!(repl.eval("c()"), Some("12".to_string()));

        // Redefine A
        repl.eval("a() = 10");

        // Now c should be (10 + 1) * 2 = 22
        assert_eq!(repl.eval("c()"), Some("22".to_string()));
    }

    #[test]
    fn test_function_redefinition_zero_arg_in_parameterized() {
        let mut repl = create_test_repl();

        // Define a zero-arg function that a parameterized function uses
        repl.eval("get_mult() = 2");

        // Define a parameterized function that uses it
        repl.eval("scale(x) = x * get_mult()");

        assert_eq!(repl.eval("scale(5)"), Some("10".to_string()));

        // Redefine the zero-arg function
        repl.eval("get_mult() = 3");

        // scale should pick up the new get_mult
        assert_eq!(repl.eval("scale(5)"), Some("15".to_string()));
    }

    #[test]
    fn test_function_redefinition_multiple_callers() {
        let mut repl = create_test_repl();

        // Define base function
        repl.eval("factor() = 2");

        // Define multiple functions that use it
        repl.eval("double(x) = x * factor()");
        repl.eval("add_factor(x) = x + factor()");

        assert_eq!(repl.eval("double(10)"), Some("20".to_string()));
        assert_eq!(repl.eval("add_factor(10)"), Some("12".to_string()));

        // Redefine factor
        repl.eval("factor() = 5");

        // Both callers should use the new definition
        assert_eq!(repl.eval("double(10)"), Some("50".to_string()));
        assert_eq!(repl.eval("add_factor(10)"), Some("15".to_string()));
    }

    #[test]
    fn test_function_redefinition_with_recursion_wrapper() {
        let mut repl = create_test_repl();

        // Define a simple non-recursive helper
        repl.eval("multiplier() = 2");

        // Define a function that uses the helper
        repl.eval("apply_mult(x) = x * multiplier()");

        assert_eq!(repl.eval("apply_mult(5)"), Some("10".to_string()));

        // Redefine the helper
        repl.eval("multiplier() = 3");

        // apply_mult should use the new multiplier
        assert_eq!(repl.eval("apply_mult(5)"), Some("15".to_string()));
    }

    #[test]
    fn test_function_redefinition_with_closure() {
        let mut repl = create_test_repl();

        // Define a constant provider
        repl.eval("get_multiplier() = 2");

        // Define a function that creates a closure-like behavior
        repl.eval("scale(x) = x * get_multiplier()");

        assert_eq!(repl.eval("scale(5)"), Some("10".to_string()));

        // Change the multiplier
        repl.eval("get_multiplier() = 10");

        // Scale should now use new multiplier
        assert_eq!(repl.eval("scale(5)"), Some("50".to_string()));
    }

    #[test]
    fn test_function_redefinition_preserves_unrelated() {
        let mut repl = create_test_repl();

        // Define multiple independent functions
        repl.eval("foo() = 1");
        repl.eval("bar() = 2");
        repl.eval("baz() = 3");

        // Define a function that uses only foo
        repl.eval("use_foo() = foo() + 100");

        assert_eq!(repl.eval("use_foo()"), Some("101".to_string()));
        assert_eq!(repl.eval("bar()"), Some("2".to_string()));
        assert_eq!(repl.eval("baz()"), Some("3".to_string()));

        // Redefine foo
        repl.eval("foo() = 50");

        // use_foo should update
        assert_eq!(repl.eval("use_foo()"), Some("150".to_string()));

        // bar and baz should be unchanged
        assert_eq!(repl.eval("bar()"), Some("2".to_string()));
        assert_eq!(repl.eval("baz()"), Some("3".to_string()));
    }

    #[test]
    fn test_parameterized_function_redefinition_direct() {
        let mut repl = create_test_repl();

        // Define a function with parameters
        repl.eval("double(x) = x * 2");
        assert_eq!(repl.eval("double(5)"), Some("10".to_string()));

        // Redefine it to triple
        repl.eval("double(x) = x * 3");

        // Direct call should use new definition
        assert_eq!(repl.eval("double(5)"), Some("15".to_string()));
    }

    #[test]
    fn test_recursive_function_safe() {
        let mut repl = create_test_repl();

        // Define multi-clause function using semicolons
        repl.eval("countdown(0) = 0; countdown(n) = countdown(n - 1)");

        // Should terminate and return 0
        assert_eq!(repl.eval("countdown(5)"), Some("0".to_string()));

        // Define a function that uses countdown
        repl.eval("count_from(n) = countdown(n)");
        assert_eq!(repl.eval("count_from(3)"), Some("0".to_string()));
    }

    #[test]
    fn test_recursive_function_accumulator() {
        let mut repl = create_test_repl();

        // Define sum using semicolons
        repl.eval("sum_to(0) = 0; sum_to(n) = n + sum_to(n - 1)");

        // sum_to(5) = 5 + 4 + 3 + 2 + 1 + 0 = 15
        assert_eq!(repl.eval("sum_to(5)"), Some("15".to_string()));
    }

    #[test]
    fn test_semicolon_in_string_preserved() {
        let mut repl = create_test_repl();

        // Semicolons inside strings should NOT be converted to newlines
        repl.eval(r#"greeting() = "Hello; World""#);
        assert_eq!(repl.eval("greeting()"), Some("Hello; World".to_string()));
    }

    #[test]
    fn test_multiple_definitions_semicolon() {
        let mut repl = create_test_repl();

        // Multiple independent definitions separated by semicolons
        repl.eval("first() = 1; second() = 2; third() = 3");

        assert_eq!(repl.eval("first()"), Some("1".to_string()));
        assert_eq!(repl.eval("second()"), Some("2".to_string()));
        assert_eq!(repl.eval("third()"), Some("3".to_string()));
    }
}
