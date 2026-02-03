//! Autocomplete engine for REPL and Editor
//!
//! Provides completion candidates based on:
//! - Function names
//! - Type names
//! - Module names
//! - Variable names
//! - Type fields (after `.` on a type instance)
//! - Module members (after `.` on a module name)
//! - Module context (functions in same module)
//! - Imports (use statements)

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Get file path completions for a partial path.
///
/// Returns a list of completion items for files and directories that match the partial path.
/// Prioritizes .nos files and directories, but shows all files.
pub fn get_file_completions(partial_path: &str) -> Vec<CompletionItem> {
    let mut completions = Vec::new();

    // Determine the directory to list and the prefix to match
    let (dir_path, file_prefix) = if partial_path.is_empty() {
        // Empty path - list current directory
        (Path::new("."), "")
    } else if partial_path.ends_with('/') || partial_path.ends_with(std::path::MAIN_SEPARATOR) {
        // Path ends with separator - list that directory
        (Path::new(partial_path), "")
    } else {
        // Path has a partial filename - split into dir and prefix
        let path = Path::new(partial_path);
        match (path.parent(), path.file_name()) {
            (Some(parent), Some(name)) => {
                let parent_str = if parent.as_os_str().is_empty() { "." } else { parent.to_str().unwrap_or(".") };
                (Path::new(parent_str), name.to_str().unwrap_or(""))
            }
            _ => (Path::new("."), partial_path),
        }
    };

    // Read directory entries
    let entries = match std::fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(_) => return completions,
    };

    // Collect matching entries
    for entry in entries.flatten() {
        let file_name = match entry.file_name().into_string() {
            Ok(name) => name,
            Err(_) => continue,
        };

        // Skip hidden files (starting with .)
        if file_name.starts_with('.') {
            continue;
        }

        // Check if it matches the prefix (case-insensitive)
        if !file_prefix.is_empty() && !file_name.to_lowercase().starts_with(&file_prefix.to_lowercase()) {
            continue;
        }

        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let is_nos_file = file_name.ends_with(".nos");

        // Build the completion text (just the filename, not the full path)
        let completion_text = if is_dir {
            format!("{}/", file_name)
        } else {
            file_name.clone()
        };

        let kind = if is_dir {
            CompletionKind::Directory
        } else {
            CompletionKind::File
        };

        // Build label with indicator
        let label = if is_dir {
            format!("{}  [dir]", file_name)
        } else if is_nos_file {
            format!("{}  [nos]", file_name)
        } else {
            file_name.clone()
        };

        completions.push(CompletionItem {
            text: completion_text,
            label,
            kind,
            doc: None,
        });
    }

    // Sort: directories first, then .nos files, then other files
    completions.sort_by(|a, b| {
        let a_is_dir = matches!(a.kind, CompletionKind::Directory);
        let b_is_dir = matches!(b.kind, CompletionKind::Directory);
        let a_is_nos = a.text.ends_with(".nos");
        let b_is_nos = b.text.ends_with(".nos");

        match (a_is_dir, b_is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => match (a_is_nos, b_is_nos) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.text.to_lowercase().cmp(&b.text.to_lowercase()),
            }
        }
    });

    completions
}

/// Parse `use` statements from source code and extract imported names.
///
/// Handles patterns like:
/// - `use Math.add` -> ["Math.add"]
/// - `use Math.{add, sub}` -> ["Math.add", "Math.sub"]
/// - `use IO.{print, println}` -> ["IO.print", "IO.println"]
pub fn parse_imports(source: &str) -> Vec<String> {
    let mut imports = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("use ") {
            continue;
        }

        let rest = trimmed[4..].trim();

        // Check for brace syntax: use Module.{a, b, c}
        if let Some(brace_start) = rest.find(".{") {
            let module = &rest[..brace_start];
            if let Some(brace_end) = rest.find('}') {
                let members = &rest[brace_start + 2..brace_end];
                for member in members.split(',') {
                    let member = member.trim();
                    if !member.is_empty() {
                        imports.push(format!("{}.{}", module, member));
                    }
                }
            }
        } else {
            // Simple import: use Module.function
            // Strip trailing comments or other text
            let import_path = rest.split_whitespace().next().unwrap_or(rest);
            if import_path.contains('.') {
                imports.push(import_path.to_string());
            }
        }
    }

    imports
}

/// Extract the module name from a qualified function name.
/// E.g., "Math.Trig.sin" -> Some("Math.Trig"), "add" -> None
#[cfg(test)]
pub fn extract_module(qualified_name: &str) -> Option<&str> {
    qualified_name.rfind('.').map(|pos| &qualified_name[..pos])
}

/// Extract module name from an editor name.
/// Handles both:
/// - Qualified function names: "utils.bar" -> Some("utils")
/// - File paths: "utils.nos" or "/path/to/utils.nos" -> Some("utils")
/// - Simple names: "add" -> None
pub fn extract_module_from_editor_name(name: &str) -> Option<String> {
    if name.ends_with(".nos") {
        // File path: extract base name without extension
        name.rsplit(['/', '\\']).next()
            .and_then(|f| f.strip_suffix(".nos"))
            .filter(|m| !m.is_empty())
            .map(|s| s.to_string())
    } else {
        // Qualified name like "utils.bar" -> extract module "utils"
        name.rfind('.').map(|dot_pos| name[..dot_pos].to_string())
    }
}

/// A completion candidate
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionItem {
    /// The text to insert (just the identifier, not the full path)
    pub text: String,
    /// Display label (may include type info or signature)
    pub label: String,
    /// Kind of completion
    pub kind: CompletionKind,
    /// Doc comment for the item (if available)
    pub doc: Option<String>,
}

/// Kind of completion item
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompletionKind {
    Function,
    PublicFunction,  // Public (exported) function - shown in a distinct color
    Type,
    Module,
    Field,
    Variable,
    Constructor,
    Method,
    File,
    Directory,
}

impl CompletionKind {
    /// Short prefix for display
    pub fn prefix(&self) -> &'static str {
        match self {
            CompletionKind::Function => "fn",
            CompletionKind::PublicFunction => "pub",
            CompletionKind::Type => "type",
            CompletionKind::Module => "mod",
            CompletionKind::Field => "field",
            CompletionKind::Variable => "var",
            CompletionKind::Constructor => "ctor",
            CompletionKind::Method => "method",
            CompletionKind::File => "file",
            CompletionKind::Directory => "dir",
        }
    }

    /// Color for display (RGB values)
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            CompletionKind::Function => (100, 200, 255),   // Light blue (private fn)
            CompletionKind::PublicFunction => (100, 255, 100), // Green (public fn)
            CompletionKind::Type => (255, 200, 100),       // Orange
            CompletionKind::Module => (200, 200, 200),     // Gray
            CompletionKind::Field => (150, 255, 150),      // Light green
            CompletionKind::Variable => (255, 150, 255),   // Pink
            CompletionKind::Constructor => (255, 255, 100), // Yellow
            CompletionKind::Method => (100, 220, 200),     // Cyan
            CompletionKind::File => (200, 255, 200),       // Light green
            CompletionKind::Directory => (100, 150, 255),  // Blue
        }
    }
}

/// Source of completion data - implemented by ReplEngine wrapper
pub trait CompletionSource {
    /// Get all function names (fully qualified, e.g., "Math.sin", "add")
    fn get_functions(&self) -> Vec<String>;

    /// Get all type names (e.g., "Point", "List")
    fn get_types(&self) -> Vec<String>;

    /// Get all variable names in scope
    fn get_variables(&self) -> Vec<String>;

    /// Get fields for a type (returns field names)
    fn get_type_fields(&self, type_name: &str) -> Vec<String>;

    /// Check if a function is public (exported)
    fn is_function_public(&self, name: &str) -> bool;

    /// Get constructors for a variant type
    fn get_type_constructors(&self, type_name: &str) -> Vec<String>;

    /// Get the signature for a function (e.g., "Int -> Int -> Int")
    fn get_function_signature(&self, name: &str) -> Option<String>;

    /// Get the doc comment for a function
    fn get_function_doc(&self, name: &str) -> Option<String>;

    /// Get the type of a variable (for field access completion)
    fn get_variable_type(&self, var_name: &str) -> Option<String>;

    /// Get UFCS methods for a type (functions whose first parameter matches the type)
    /// Returns (local_name, signature, doc) tuples
    fn get_ufcs_methods_for_type(&self, type_name: &str) -> Vec<(String, String, Option<String>)>;
}

/// Completion context - what kind of completion is needed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionContext {
    /// General identifier completion (functions, types, variables, modules)
    Identifier { prefix: String },
    /// Dot completion after a module name (e.g., "Math.")
    ModuleMember { module: String, prefix: String },
    /// Dot completion after a type/variable (field access)
    FieldAccess { receiver: String, prefix: String },
    /// File path completion (inside :load "...")
    FilePath { partial_path: String },
}

/// Autocomplete engine
pub struct Autocomplete {
    /// Cached module names (extracted from function/type prefixes)
    pub modules: HashSet<String>,
    /// Function name -> signatures
    functions: HashMap<String, Vec<String>>,
    /// Type names
    types: HashSet<String>,
}

impl Autocomplete {
    pub fn new() -> Self {
        Self {
            modules: HashSet::new(),
            functions: HashMap::new(),
            types: HashSet::new(),
        }
    }

    /// Update the autocomplete cache from a completion source
    pub fn update_from_source(&mut self, source: &dyn CompletionSource) {
        self.modules.clear();
        self.functions.clear();
        self.types.clear();

        // Process functions
        for name in source.get_functions() {
            // Extract module from qualified name (e.g., "Math.sin" -> "Math")
            if let Some(dot_pos) = name.rfind('.') {
                let module = &name[..dot_pos];
                self.modules.insert(module.to_string());
            }

            // Store function (strip signature suffix like "/Int,Int")
            let base_name = name.split('/').next().unwrap_or(&name);
            self.functions.entry(base_name.to_string()).or_default();
        }

        // Process types
        for name in source.get_types() {
            self.types.insert(name.clone());
            // Types can also define modules (e.g., "Geometry.Point")
            if let Some(dot_pos) = name.rfind('.') {
                let module = &name[..dot_pos];
                self.modules.insert(module.to_string());
            }
        }
    }

    /// Parse the completion context from input text and cursor position
    pub fn parse_context(&self, line: &str, cursor_col: usize) -> CompletionContext {
        // Get text before cursor
        let before_cursor: String = line.chars().take(cursor_col).collect();

        // Check for file path context first (inside :load "...")
        if let Some(partial_path) = self.extract_load_path(&before_cursor) {
            return CompletionContext::FilePath { partial_path };
        }

        // Find the start of the current identifier/expression
        let (_expr_start, expr) = self.extract_expression(&before_cursor);

        // Check if there's a dot in the expression
        if let Some(dot_pos) = expr.rfind('.') {
            let receiver = &expr[..dot_pos];
            let prefix = &expr[dot_pos + 1..];

            // Check if receiver is a known module
            if self.modules.contains(receiver) || self.is_upper_ident(receiver) {
                return CompletionContext::ModuleMember {
                    module: receiver.to_string(),
                    prefix: prefix.to_string(),
                };
            }

            // Otherwise assume field access
            return CompletionContext::FieldAccess {
                receiver: receiver.to_string(),
                prefix: prefix.to_string(),
            };
        }

        CompletionContext::Identifier { prefix: expr.to_string() }
    }

    /// Check if cursor is inside a :load "..." command and extract the partial path
    fn extract_load_path(&self, before_cursor: &str) -> Option<String> {
        // Look for :load " pattern followed by text without closing quote
        // Pattern: :load "partial_path  (no closing quote before cursor)
        let trimmed = before_cursor.trim_start();

        // Check for :load command
        if !trimmed.starts_with(":load ") {
            return None;
        }

        let after_load = &trimmed[6..]; // Skip ":load "
        let after_load = after_load.trim_start();

        // Must start with a quote
        if !after_load.starts_with('"') {
            return None;
        }

        let path_part = &after_load[1..]; // Skip the opening quote

        // If there's a closing quote, we're not inside the string
        if path_part.contains('"') {
            return None;
        }

        // Return the partial path (everything after the opening quote)
        Some(path_part.to_string())
    }

    /// Extract the expression being completed from text before cursor
    fn extract_expression<'a>(&self, text: &'a str) -> (usize, &'a str) {
        // Walk backwards to find expression start
        // Handle: identifiers, dots, method calls (...), and literal expressions ([...], "...", %{...}, #{...})
        let chars: Vec<char> = text.chars().collect();
        let mut start = chars.len();
        let mut i = chars.len();

        while i > 0 {
            i -= 1;
            let c = chars[i];

            if c.is_alphanumeric() || c == '_' || c == '.' {
                start = i;
            } else if c == ')' {
                // Method call arguments - find matching ( and continue
                if let Some(open) = self.find_matching_bracket(&chars, i, '(', ')') {
                    start = open;
                    i = open;
                } else {
                    break;
                }
            } else if c == ']' {
                // List literal - find matching [
                if let Some(open) = self.find_matching_bracket(&chars, i, '[', ']') {
                    start = open;
                    i = open;
                } else {
                    break;
                }
            } else if c == '}' {
                // Map or Set literal - find matching { and check for %{ or #{
                if let Some(open) = self.find_matching_bracket(&chars, i, '{', '}') {
                    if open > 0 && (chars[open - 1] == '%' || chars[open - 1] == '#') {
                        start = open - 1;
                        i = open - 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else if c == '"' {
                // String literal - find matching opening quote
                if let Some(open) = self.find_string_start(&chars, i, '"') {
                    start = open;
                    i = open;
                } else {
                    break;
                }
            } else if c == '\'' {
                // Single-quoted string literal - find matching opening quote
                if let Some(open) = self.find_string_start(&chars, i, '\'') {
                    start = open;
                    i = open;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Convert char index to byte index
        let byte_start = text.char_indices()
            .nth(start)
            .map(|(i, _)| i)
            .unwrap_or(text.len());

        (start, &text[byte_start..])
    }

    /// Find the matching opening bracket for a closing bracket at position `close`
    fn find_matching_bracket(&self, chars: &[char], close: usize, open_char: char, close_char: char) -> Option<usize> {
        let mut depth = 1;
        let mut i = close;

        while i > 0 {
            i -= 1;
            if chars[i] == close_char {
                depth += 1;
            } else if chars[i] == open_char {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Find the opening quote for a string literal ending at position `close`
    fn find_string_start(&self, chars: &[char], close: usize, quote_char: char) -> Option<usize> {
        let mut i = close;

        while i > 0 {
            i -= 1;
            if chars[i] == quote_char {
                // Check if escaped
                let mut backslashes = 0;
                let mut j = i;
                while j > 0 && chars[j - 1] == '\\' {
                    backslashes += 1;
                    j -= 1;
                }
                // If even number of backslashes, this is the opening quote
                if backslashes % 2 == 0 {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Check if string starts with uppercase (likely a module/type name)
    fn is_upper_ident(&self, s: &str) -> bool {
        s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
    }

    /// Get completion candidates for a given context
    pub fn get_completions(
        &self,
        context: &CompletionContext,
        source: &dyn CompletionSource,
    ) -> Vec<CompletionItem> {
        self.get_completions_with_context(context, source, None, &[])
    }

    /// Get completion candidates with module context awareness
    ///
    /// - `current_module`: The module being edited (e.g., "Math" when editing "Math.add")
    /// - `imports`: List of imported names (e.g., ["IO.print", "List.map"])
    pub fn get_completions_with_context(
        &self,
        context: &CompletionContext,
        source: &dyn CompletionSource,
        current_module: Option<&str>,
        imports: &[String],
    ) -> Vec<CompletionItem> {
        match context {
            CompletionContext::Identifier { prefix } => {
                self.complete_identifier_with_context(prefix, source, current_module, imports)
            }
            CompletionContext::ModuleMember { module, prefix } => {
                self.complete_module_member(module, prefix, source)
            }
            CompletionContext::FieldAccess { receiver, prefix } => {
                self.complete_field_access(receiver, prefix, source)
            }
            CompletionContext::FilePath { partial_path } => {
                // File path completions are handled separately via get_file_completions
                get_file_completions(partial_path)
            }
        }
    }

    /// Format a function label with optional signature
    fn format_function_label(name: &str, full_name: &str, suffix: Option<&str>, source: &dyn CompletionSource) -> String {
        let sig = source.get_function_signature(full_name);
        match (sig, suffix) {
            (Some(s), Some(suf)) => format!("{} :: {} {}", name, s, suf),
            (Some(s), None) => format!("{} :: {}", name, s),
            (None, Some(suf)) => format!("{} {}", name, suf),
            (None, None) => name.to_string(),
        }
    }

    /// Complete identifier with module context and imports
    fn complete_identifier_with_context(
        &self,
        prefix: &str,
        source: &dyn CompletionSource,
        current_module: Option<&str>,
        imports: &[String],
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        let prefix_lower = prefix.to_lowercase();
        let mut seen = HashSet::new();

        // Build set of imported short names -> full names
        let mut imported_names: HashMap<String, String> = HashMap::new();
        for import in imports {
            // import could be "Math.add" - short name is "add"
            if let Some(short) = import.rsplit('.').next() {
                imported_names.insert(short.to_string(), import.clone());
            }
        }

        // Add matching modules
        for module in &self.modules {
            if module.to_lowercase().starts_with(&prefix_lower) {
                if seen.insert(module.clone()) {
                    items.push(CompletionItem {
                        text: module.clone(),
                        label: module.clone(),
                        kind: CompletionKind::Module,
                        doc: None,
                    });
                }
            }
        }

        // Add matching types
        for type_name in &self.types {
            let base = type_name.rsplit('.').next().unwrap_or(type_name);
            if base.to_lowercase().starts_with(&prefix_lower) {
                if seen.insert(base.to_string()) {
                    items.push(CompletionItem {
                        text: base.to_string(),
                        label: type_name.clone(),
                        kind: CompletionKind::Type,
                        doc: None,
                    });
                }
            }
        }

        // Add functions from current module (can use without prefix)
        if let Some(module) = current_module {
            let module_prefix = format!("{}.", module);
            for func_name in self.functions.keys() {
                if func_name.starts_with(&module_prefix) {
                    let short_name = &func_name[module_prefix.len()..];
                    // Only direct members (no nested modules)
                    if !short_name.contains('.') && short_name.to_lowercase().starts_with(&prefix_lower) {
                        if seen.insert(short_name.to_string()) {
                            let label = Self::format_function_label(short_name, func_name, Some("(mod)"), source);
                            let doc = source.get_function_doc(func_name);
                            let kind = if source.is_function_public(func_name) {
                                CompletionKind::PublicFunction
                            } else {
                                CompletionKind::Function
                            };
                            items.push(CompletionItem {
                                text: short_name.to_string(),
                                label,
                                kind,
                                doc,
                            });
                        }
                    }
                }
            }
        }

        // Add imported functions (can use with short name)
        for (short_name, full_name) in &imported_names {
            if short_name.to_lowercase().starts_with(&prefix_lower) {
                if seen.insert(short_name.clone()) {
                    let label = Self::format_function_label(short_name, full_name, Some("(imp)"), source);
                    let doc = source.get_function_doc(full_name);
                    let kind = if source.is_function_public(full_name) {
                        CompletionKind::PublicFunction
                    } else {
                        CompletionKind::Function
                    };
                    items.push(CompletionItem {
                        text: short_name.clone(),
                        label,
                        kind,
                        doc,
                    });
                }
            }
        }

        // Add top-level functions (not module-qualified)
        for func_name in self.functions.keys() {
            if !func_name.contains('.') {
                if func_name.to_lowercase().starts_with(&prefix_lower) {
                    if seen.insert(func_name.clone()) {
                        let label = Self::format_function_label(func_name, func_name, None, source);
                        let doc = source.get_function_doc(func_name);
                        let kind = if source.is_function_public(func_name) {
                            CompletionKind::PublicFunction
                        } else {
                            CompletionKind::Function
                        };
                        items.push(CompletionItem {
                            text: func_name.clone(),
                            label,
                            kind,
                            doc,
                        });
                    }
                }
            }
        }

        // Add matching variables with type signatures
        for var_name in source.get_variables() {
            if var_name.to_lowercase().starts_with(&prefix_lower) {
                if seen.insert(var_name.clone()) {
                    let label = if let Some(var_type) = source.get_variable_type(&var_name) {
                        format!("{} :: {}", var_name, var_type)
                    } else {
                        var_name.clone()
                    };
                    items.push(CompletionItem {
                        text: var_name.clone(),
                        label,
                        kind: CompletionKind::Variable,
                        doc: None,
                    });
                }
            }
        }

        // Sort: exact prefix match first, then alphabetically
        items.sort_by(|a, b| {
            let a_exact = a.text.starts_with(prefix);
            let b_exact = b.text.starts_with(prefix);
            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.text.cmp(&b.text),
            }
        });

        items
    }

    /// Complete members of a module (functions, types, submodules)
    fn complete_module_member(
        &self,
        module: &str,
        prefix: &str,
        source: &dyn CompletionSource,
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        let prefix_lower = prefix.to_lowercase();
        let module_prefix = format!("{}.", module);

        // Add functions in this module
        for func_name in self.functions.keys() {
            if func_name.starts_with(&module_prefix) {
                let member = &func_name[module_prefix.len()..];
                // Only direct members (no more dots)
                if !member.contains('.') && member.to_lowercase().starts_with(&prefix_lower) {
                    let label = Self::format_function_label(member, func_name, None, source);
                    let doc = source.get_function_doc(func_name);
                    let kind = if source.is_function_public(func_name) {
                        CompletionKind::PublicFunction
                    } else {
                        CompletionKind::Function
                    };
                    items.push(CompletionItem {
                        text: member.to_string(),
                        label,
                        kind,
                        doc,
                    });
                }
            }
        }

        // Add types in this module
        for type_name in &self.types {
            if type_name.starts_with(&module_prefix) {
                let member = &type_name[module_prefix.len()..];
                if !member.contains('.') && member.to_lowercase().starts_with(&prefix_lower) {
                    items.push(CompletionItem {
                        text: member.to_string(),
                        label: type_name.clone(),
                        kind: CompletionKind::Type,
                        doc: None,
                    });
                }
            }
        }

        // Add submodules
        for submodule in &self.modules {
            if submodule.starts_with(&module_prefix) {
                let member = &submodule[module_prefix.len()..];
                // Only direct submodules
                let direct_member = member.split('.').next().unwrap_or(member);
                if direct_member.to_lowercase().starts_with(&prefix_lower) {
                    // Avoid duplicates
                    if !items.iter().any(|i| i.text == direct_member) {
                        items.push(CompletionItem {
                            text: direct_member.to_string(),
                            label: format!("{}.{}", module, direct_member),
                            kind: CompletionKind::Module,
                            doc: None,
                        });
                    }
                }
            }
        }

        items.sort_by(|a, b| a.text.cmp(&b.text));
        items
    }

    /// Check if a type name is a numeric type
    fn is_numeric_type(type_name: &str) -> bool {
        matches!(
            type_name,
            "Int" | "Int8" | "Int16" | "Int32" | "Int64"
                | "UInt8" | "UInt16" | "UInt32" | "UInt64"
                | "Float" | "Float32" | "Float64"
                | "BigInt"
        )
    }

    /// Get available methods for a builtin type
    /// Returns (method_name, signature, docstring)
    fn get_builtin_methods(type_name: &str) -> Vec<(&'static str, &'static str, &'static str)> {
        // Strip trait bounds prefix (e.g., "Eq a, Hash a => Map[a, b]" -> "Map[a, b]")
        let base_type = if let Some(arrow_pos) = type_name.find("=>") {
            type_name[arrow_pos + 2..].trim()
        } else {
            type_name
        };

        let methods: Vec<(&'static str, &'static str, &'static str)> = if base_type.starts_with("Map") || base_type == "Map" {
            vec![
                ("get", "(key) -> value", "Get the value associated with a key"),
                ("insert", "(key, value) -> Map", "Insert a key-value pair, returning a new map"),
                ("remove", "(key) -> Map", "Remove a key, returning a new map"),
                ("contains", "(key) -> Bool", "Check if the map contains a key"),
                ("keys", "() -> List", "Get all keys as a list"),
                ("values", "() -> List", "Get all values as a list"),
                ("size", "() -> Int", "Get the number of key-value pairs"),
                ("isEmpty", "() -> Bool", "Check if the map is empty"),
                ("merge", "(other) -> Map", "Merge two maps, with other's values taking precedence"),
            ]
        } else if base_type.starts_with("Set") || base_type == "Set" {
            vec![
                ("contains", "(elem) -> Bool", "Check if the set contains an element"),
                ("insert", "(elem) -> Set", "Insert an element, returning a new set"),
                ("remove", "(elem) -> Set", "Remove an element, returning a new set"),
                ("size", "() -> Int", "Get the number of elements"),
                ("isEmpty", "() -> Bool", "Check if the set is empty"),
                ("union", "(other) -> Set", "Return the union of two sets"),
                ("intersection", "(other) -> Set", "Return the intersection of two sets"),
                ("difference", "(other) -> Set", "Return elements in this set but not in other"),
                ("toList", "() -> List", "Convert the set to a list"),
            ]
        } else if base_type == "String" {
            vec![
                ("length", "() -> Int", "Get the length of the string"),
                ("chars", "() -> List", "Get the characters as a list"),
                ("toInt", "() -> Option Int", "Parse as an integer, None if invalid"),
                ("toFloat", "() -> Option Float", "Parse as a float, None if invalid"),
                ("trim", "() -> String", "Remove leading and trailing whitespace"),
                ("trimStart", "() -> String", "Remove leading whitespace"),
                ("trimEnd", "() -> String", "Remove trailing whitespace"),
                ("toUpper", "() -> String", "Convert to uppercase"),
                ("toLower", "() -> String", "Convert to lowercase"),
                ("contains", "(substr) -> Bool", "Check if the string contains a substring"),
                ("startsWith", "(prefix) -> Bool", "Check if the string starts with a prefix"),
                ("endsWith", "(suffix) -> Bool", "Check if the string ends with a suffix"),
                ("replace", "(from, to) -> String", "Replace first occurrence of a substring"),
                ("replaceAll", "(from, to) -> String", "Replace all occurrences of a substring"),
                ("indexOf", "(substr) -> Int", "Find the index of a substring (-1 if not found)"),
                ("lastIndexOf", "(substr) -> Int", "Find the last index of a substring"),
                ("substring", "(start, end) -> String", "Extract a substring by indices"),
                ("repeat", "(n) -> String", "Repeat the string n times"),
                ("padStart", "(len, pad) -> String", "Pad the start to reach length"),
                ("padEnd", "(len, pad) -> String", "Pad the end to reach length"),
                ("reverse", "() -> String", "Reverse the string"),
                ("lines", "() -> List", "Split into lines"),
                ("words", "() -> List", "Split into words"),
                ("isEmpty", "() -> Bool", "Check if the string is empty"),
            ]
        } else if base_type.starts_with("List") || base_type == "List" || base_type.starts_with('[') {
            vec![
                // Builtins
                ("length", "() -> Int", "Get the number of elements"),
                ("head", "() -> a", "Get the first element"),
                ("tail", "() -> List", "Get all elements except the first"),
                ("isEmpty", "() -> Bool", "Check if the list is empty"),
                // Core transformations
                ("map", "(f) -> List", "Apply a function to each element"),
                ("filter", "(pred) -> List", "Keep elements that satisfy the predicate"),
                ("each", "(f) -> ()", "Apply function to each element for side effects"),
                ("fold", "(acc, f) -> a", "Left fold with accumulator"),
                ("foldr", "(acc, f) -> a", "Right fold with accumulator"),
                // Predicates
                ("any", "(pred) -> Bool", "Check if any element satisfies the predicate"),
                ("all", "(pred) -> Bool", "Check if all elements satisfy the predicate"),
                ("contains", "(elem) -> Bool", "Check if the list contains an element"),
                // Search
                ("find", "(pred) -> Option", "Find the first element satisfying the predicate"),
                ("position", "(pred) -> Option Int", "Find the index of first matching element"),
                ("indexOf", "(elem) -> Option Int", "Find index of first occurrence"),
                // Element access
                ("last", "() -> a", "Get the last element"),
                ("init", "() -> List", "Get all elements except the last"),
                ("get", "(n) -> a", "Get element at index (0-based)"),
                ("nth", "(n) -> a", "Get element at index (0-based)"),
                // Modification
                ("set", "(idx, val) -> List", "Set element at index, returns new list"),
                ("push", "(elem) -> List", "Append element to end of list"),
                ("pop", "() -> (List, a)", "Remove and return last element"),
                ("remove", "(elem) -> List", "Remove first occurrence of element"),
                ("removeAt", "(idx) -> List", "Remove element at index"),
                ("insertAt", "(idx, elem) -> List", "Insert element at index"),
                // Slicing
                ("take", "(n) -> List", "Take the first n elements"),
                ("drop", "(n) -> List", "Drop the first n elements"),
                ("slice", "(start, stop) -> List", "Get sublist from start to stop (exclusive)"),
                ("takeWhile", "(pred) -> List", "Take elements while predicate is true"),
                ("dropWhile", "(pred) -> List", "Drop elements while predicate is true"),
                ("splitAt", "(n) -> (List, List)", "Split at index n"),
                ("partition", "(pred) -> (List, List)", "Split into (matching, non-matching)"),
                // Ordering
                ("reverse", "() -> List", "Reverse the list"),
                ("sort", "() -> List", "Sort the list"),
                ("sortBy", "(cmp) -> List", "Sort with custom comparator"),
                ("isSorted", "() -> Bool", "Check if list is sorted"),
                // Combining
                ("concat", "(other) -> List", "Concatenate two lists"),
                ("flatten", "() -> List", "Flatten a list of lists"),
                ("zip", "(other) -> List", "Zip two lists into pairs"),
                ("zipWith", "(other, f) -> List", "Zip two lists using a function"),
                ("unzip", "() -> (List, List)", "Split list of pairs into two lists"),
                ("interleave", "(other) -> List", "Interleave elements from two lists"),
                ("intersperse", "(sep) -> List", "Insert separator between elements"),
                // Grouping
                ("unique", "() -> List", "Remove duplicate elements"),
                ("group", "() -> List", "Group consecutive equal elements"),
                ("groupBy", "(keyFn) -> List", "Group by key function"),
                // Aggregation
                ("count", "(pred) -> Int", "Count elements satisfying the predicate"),
                ("sum", "() -> a", "Sum all elements"),
                ("product", "() -> a", "Product of all elements"),
                ("maximum", "() -> a", "Get the maximum element"),
                ("minimum", "() -> a", "Get the minimum element"),
                // Scan
                ("scanl", "(acc, f) -> List", "Left scan with accumulator"),
                // Other
                ("replicate", "(n, val) -> List", "Create list of n copies"),
                ("transpose", "() -> List", "Transpose list of lists"),
                ("pairwise", "(f) -> List", "Apply function to adjacent pairs"),
            ]
        } else if base_type == "Tuple" || base_type.starts_with('(') {
            vec![
                ("length", "() -> Int", "Get the number of elements in the tuple"),
            ]
        } else if base_type == "Option" || base_type.starts_with("Option ") || base_type.starts_with("Option[") {
            vec![
                ("isSome", "() -> Bool", "Check if this is Some"),
                ("isNone", "() -> Bool", "Check if this is None"),
                ("unwrap", "() -> a", "Get the value or panic if None"),
                ("unwrapOr", "(default) -> a", "Get the value or return default if None"),
                ("map", "(f) -> Option", "Apply function to value if Some"),
                ("flatMap", "(f) -> Option", "Apply function returning Option if Some"),
                ("filter", "(pred) -> Option", "Return None if predicate fails"),
                ("getOrElse", "(f) -> a", "Get value or compute default"),
            ]
        } else if base_type == "Result" || base_type.starts_with("Result ") || base_type.starts_with("Result[") {
            vec![
                ("isOk", "() -> Bool", "Check if this is Ok"),
                ("isErr", "() -> Bool", "Check if this is Err"),
                ("unwrap", "() -> a", "Get the value or panic if Err"),
                ("unwrapOr", "(default) -> a", "Get the value or return default if Err"),
                ("unwrapErr", "() -> e", "Get the error or panic if Ok"),
                ("map", "(f) -> Result", "Apply function to value if Ok"),
                ("mapErr", "(f) -> Result", "Apply function to error if Err"),
                ("flatMap", "(f) -> Result", "Apply function returning Result if Ok"),
                ("getOrElse", "(f) -> a", "Get value or compute from error"),
            ]
        } else if base_type == "Float64Array" {
            vec![
                ("fromList", "(List Float) -> Float64Array", "Create array from a list of floats"),
                ("length", "() -> Int", "Get the number of elements"),
                ("get", "(Int) -> Float", "Get element at index"),
                ("set", "(Int, Float) -> Float64Array", "Set element at index, returns new array"),
                ("toList", "() -> List Float", "Convert to a list of floats"),
                ("make", "(Int, Float) -> Float64Array", "Create array of size with default value"),
            ]
        } else if base_type == "Int64Array" {
            vec![
                ("fromList", "(List Int) -> Int64Array", "Create array from a list of integers"),
                ("length", "() -> Int", "Get the number of elements"),
                ("get", "(Int) -> Int", "Get element at index"),
                ("set", "(Int, Int) -> Int64Array", "Set element at index, returns new array"),
                ("toList", "() -> List Int", "Convert to a list of integers"),
                ("make", "(Int, Int) -> Int64Array", "Create array of size with default value"),
            ]
        } else if base_type == "Float32Array" {
            vec![
                ("fromList", "(List Float) -> Float32Array", "Create array from a list of floats"),
                ("length", "() -> Int", "Get the number of elements"),
                ("get", "(Int) -> Float", "Get element at index"),
                ("set", "(Int, Float) -> Float32Array", "Set element at index, returns new array"),
                ("toList", "() -> List Float", "Convert to a list of floats"),
                ("make", "(Int, Float) -> Float32Array", "Create array of size with default value"),
            ]
        } else if base_type == "Buffer" {
            vec![
                ("new", "() -> Buffer", "Create a new empty buffer"),
                ("append", "(String) -> Buffer", "Append a string to the buffer"),
                ("toString", "() -> String", "Convert buffer contents to string"),
            ]
        } else if base_type == "Uuid" {
            vec![
                ("v4", "() -> String", "Generate a random UUID v4"),
                ("isValid", "(String) -> Bool", "Check if string is a valid UUID"),
            ]
        } else if base_type == "Crypto" {
            vec![
                ("sha256", "(String) -> String", "Compute SHA-256 hash, returns hex string"),
                ("sha512", "(String) -> String", "Compute SHA-512 hash, returns hex string"),
                ("md5", "(String) -> String", "Compute MD5 hash (insecure), returns hex string"),
                ("bcryptHash", "(String, Int) -> String", "Hash password with bcrypt"),
                ("bcryptVerify", "(String, String) -> Bool", "Verify password against bcrypt hash"),
                ("randomBytes", "(Int) -> String", "Generate n random bytes as hex string"),
            ]
        } else if base_type == "Runtime" {
            vec![
                ("threadCount", "() -> Int", "Get number of available CPU threads"),
                ("uptimeMs", "() -> Int", "Get milliseconds since program started"),
                ("memoryKb", "() -> Int", "Get current process memory usage in KB"),
                ("pid", "() -> Int", "Get current process ID"),
                ("loadAvg", "() -> (Float, Float, Float)", "Get 1, 5, 15 minute load averages"),
                ("numThreads", "() -> Int", "Get number of OS threads in process"),
                ("tokioWorkers", "() -> Int", "Get number of tokio worker threads"),
                ("blockingThreads", "() -> Int", "Get number of tokio blocking threads"),
            ]
        } else if Self::is_numeric_type(base_type) {
            // Type conversion methods available on all numeric types
            vec![
                ("asInt8", "() -> Int8", "Convert to Int8"),
                ("asInt16", "() -> Int16", "Convert to Int16"),
                ("asInt32", "() -> Int32", "Convert to Int32"),
                ("asInt64", "() -> Int64", "Convert to Int64"),
                ("asInt", "() -> Int", "Convert to Int (alias for asInt64)"),
                ("asUInt8", "() -> UInt8", "Convert to UInt8"),
                ("asUInt16", "() -> UInt16", "Convert to UInt16"),
                ("asUInt32", "() -> UInt32", "Convert to UInt32"),
                ("asUInt64", "() -> UInt64", "Convert to UInt64"),
                ("asFloat32", "() -> Float32", "Convert to Float32"),
                ("asFloat64", "() -> Float64", "Convert to Float64"),
                ("asFloat", "() -> Float", "Convert to Float (alias for asFloat64)"),
                ("asBigInt", "() -> BigInt", "Convert to BigInt"),
            ]
        } else {
            // For all other types, return generic builtins only
            vec![
                ("show", "() -> String", "Convert to string representation"),
                ("hash", "() -> Int", "Get hash code"),
                ("copy", "() -> Self", "Create a copy of the value"),
            ]
        };

        // Add generic builtins that work on any type
        // These are appended to the type-specific methods
        let mut all_methods = methods;
        let generic_builtins: Vec<(&'static str, &'static str, &'static str)> = vec![
            ("show", "() -> String", "Convert to string representation"),
            ("hash", "() -> Int", "Get hash code"),
            ("copy", "() -> Self", "Create a copy of the value"),
        ];

        for builtin in generic_builtins {
            if !all_methods.iter().any(|(name, _, _)| *name == builtin.0) {
                all_methods.push(builtin);
            }
        }

        all_methods
    }

    /// Detect the type of a literal expression
    fn detect_literal_type(expr: &str) -> Option<&'static str> {
        let trimmed = expr.trim();

        // String literal: "..." or '...'
        if trimmed.starts_with('"') || trimmed.starts_with('\'') {
            return Some("String");
        }

        // List literal: [...] or starts with [
        if trimmed.starts_with('[') {
            return Some("List");
        }

        // Map literal: %{...} or starts with %{
        if trimmed.starts_with("%{") {
            return Some("Map");
        }

        // Set literal: #{...} or starts with #{
        if trimmed.starts_with("#{") {
            return Some("Set");
        }

        // Tuple literal: (x, y, ...) - must have comma to distinguish from parenthesized expr
        if trimmed.starts_with('(') && trimmed.contains(',') {
            return Some("Tuple");
        }

        // Numeric literals - check for typed suffixes first
        // Handle negative numbers by stripping the minus sign
        let num_part = trimmed.strip_prefix('-').unwrap_or(trimmed);
        if !num_part.is_empty() && (num_part.chars().next().unwrap().is_ascii_digit() || num_part.starts_with('.')) {
            // Check for typed integer suffixes
            if num_part.ends_with("i8") { return Some("Int8"); }
            if num_part.ends_with("i16") { return Some("Int16"); }
            if num_part.ends_with("i32") { return Some("Int32"); }
            if num_part.ends_with("i64") { return Some("Int64"); }
            if num_part.ends_with("u8") { return Some("UInt8"); }
            if num_part.ends_with("u16") { return Some("UInt16"); }
            if num_part.ends_with("u32") { return Some("UInt32"); }
            if num_part.ends_with("u64") { return Some("UInt64"); }
            if num_part.ends_with("f32") { return Some("Float32"); }
            if num_part.ends_with("f64") { return Some("Float64"); }
            if num_part.ends_with('n') { return Some("BigInt"); }

            // Check if it looks like a float (contains decimal point)
            if num_part.contains('.') {
                return Some("Float");
            }

            // Plain integer literal
            if num_part.chars().all(|c| c.is_ascii_digit() || c == '_') {
                return Some("Int");
            }
        }

        None
    }

    /// Complete fields of a type
    fn complete_field_access(
        &self,
        receiver: &str,
        prefix: &str,
        source: &dyn CompletionSource,
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        let prefix_lower = prefix.to_lowercase();

        // Try to find the type of the receiver
        let type_name = if self.is_upper_ident(receiver) {
            // Uppercase - it's a type name directly
            receiver.to_string()
        } else if let Some(literal_type) = Self::detect_literal_type(receiver) {
            // Literal expression like [1,2,3], "hello", %{...}, #{...}
            literal_type.to_string()
        } else if let Some(chain_type) = self.infer_method_chain_type(receiver, source) {
            // Method chain like aa.append("ss") - infer return type
            chain_type
        } else if let Some(call_type) = self.infer_function_call_type(receiver, source) {
            // Direct function call like greet(Person("petter")) - infer return type
            call_type
        } else {
            // Lowercase - try to look up the variable's type
            if let Some(var_type) = source.get_variable_type(receiver) {
                var_type
            } else {
                receiver.to_string()
            }
        };

        // Get methods for builtin types (Map, Set, String, List)
        for (method_name, signature, docstring) in Self::get_builtin_methods(&type_name) {
            if method_name.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: method_name.to_string(),
                    label: format!("{}{}", method_name, signature),
                    kind: CompletionKind::Method,
                    doc: Some(docstring.to_string()),
                });
            }
        }

        // Get fields from known type definitions
        for field in source.get_type_fields(&type_name) {
            if field.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: field.clone(),
                    label: field,
                    kind: CompletionKind::Field,
                    doc: None,
                });
            }
        }

        // If the type looks like an anonymous record "{ field1: Type1, field2: Type2 }",
        // extract fields from it
        if type_name.starts_with('{') && type_name.ends_with('}') {
            for field in Self::extract_record_fields(&type_name) {
                if field.to_lowercase().starts_with(&prefix_lower) {
                    if !items.iter().any(|i| i.text == field) {
                        items.push(CompletionItem {
                            text: field.clone(),
                            label: field,
                            kind: CompletionKind::Field,
                            doc: None,
                        });
                    }
                }
            }
        }

        // Get constructors (for variant types)
        for ctor in source.get_type_constructors(&type_name) {
            if ctor.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: ctor.clone(),
                    label: ctor,
                    kind: CompletionKind::Constructor,
                    doc: None,
                });
            }
        }

        // Get UFCS methods (functions whose first parameter matches the type)
        // This enables v.vecLen() style completions for extension types
        for (method_name, signature, doc) in source.get_ufcs_methods_for_type(&type_name) {
            if method_name.to_lowercase().starts_with(&prefix_lower) {
                // Skip if we already have this method from builtins
                if !items.iter().any(|i| i.text == method_name) {
                    items.push(CompletionItem {
                        text: method_name.clone(),
                        label: format!("{}{}", method_name, signature),
                        kind: CompletionKind::Method,
                        doc,
                    });
                }
            }
        }

        items.sort_by(|a, b| a.text.cmp(&b.text));
        items
    }

    /// Infer the return type of a method chain like "aa.append(\"ss\")"
    fn infer_method_chain_type(&self, receiver: &str, source: &dyn CompletionSource) -> Option<String> {
        // Look for method call pattern: base.method(...)
        // Find the last method call by looking for .method( pattern
        let mut depth: i32 = 0;
        let mut last_dot_before_paren = None;

        for (i, c) in receiver.chars().enumerate() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = (depth - 1).max(0),
                '.' if depth == 0 => last_dot_before_paren = Some(i),
                _ => {}
            }
        }

        let dot_pos = last_dot_before_paren?;
        let base = &receiver[..dot_pos];
        let method_part = &receiver[dot_pos + 1..];

        // Extract just the method name (before parentheses)
        let method_name = method_part.split('(').next()?.trim();

        // Get the type of the base expression
        let base_type = if let Some(var_type) = source.get_variable_type(base) {
            var_type
        } else if let Some(chain_type) = self.infer_method_chain_type(base, source) {
            // Recursive: base is also a chain
            chain_type
        } else {
            return None;
        };

        // Look up the return type of base_type.method_name
        // First try builtin types
        if let Some(ret_type) = Self::get_method_return_type(&base_type, method_name) {
            return Some(ret_type);
        }

        // Then try user-defined functions (UFCS)
        // Look for a function named method_name that could take base_type as first arg
        if let Some(sig) = source.get_function_signature(method_name) {
            // Signature format: "Type1 -> Type2 -> ReturnType" or "ReturnType" (for 0-arg)
            // Extract the return type (last part after ->)
            if let Some(arrow_pos) = sig.rfind("->") {
                let ret_type = sig[arrow_pos + 2..].trim();
                if !ret_type.is_empty() {
                    return Some(ret_type.to_string());
                }
            } else if !sig.contains("->") {
                // No arrows means it's just the return type
                return Some(sig.trim().to_string());
            }
        }

        None
    }

    /// Infer the return type of a direct function call like "greet(Person(\"petter\"))"
    fn infer_function_call_type(&self, receiver: &str, source: &dyn CompletionSource) -> Option<String> {
        // Check if receiver matches pattern: identifier(...)
        // The identifier should start the string and be followed by (
        let receiver = receiver.trim();

        // Find the first (
        let paren_pos = receiver.find('(')?;
        if paren_pos == 0 {
            return None; // Starts with ( - not a function call
        }

        // Extract potential function name
        let func_name = &receiver[..paren_pos];

        // Function name should be a valid identifier (letters, digits, underscores, starting with letter)
        if func_name.is_empty() || !func_name.chars().next()?.is_alphabetic() {
            return None;
        }
        if !func_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }

        // Verify the parentheses are balanced (i.e., this is a complete function call)
        let after_func = &receiver[paren_pos..];
        let mut depth = 0;
        for c in after_func.chars() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
        }
        if depth != 0 {
            return None; // Unbalanced - not a complete function call
        }

        // Handle generic builtins with known return types
        match func_name {
            "show" => return Some("String".to_string()),
            "hash" => return Some("Int".to_string()),
            "copy" => {
                // copy returns the same type as its argument
                // Extract the argument and infer its type
                let args_str = &receiver[paren_pos + 1..receiver.len() - 1];
                if let Some(lit_type) = Self::detect_literal_type(args_str) {
                    return Some(lit_type.to_string());
                }
                // Try to infer as variable type
                let arg_trimmed = args_str.trim();
                if let Some(var_type) = source.get_variable_type(arg_trimmed) {
                    return Some(var_type);
                }
                // Try to infer as nested function call
                if let Some(call_type) = self.infer_function_call_type(args_str, source) {
                    return Some(call_type);
                }
                // Fallback: check if it's a method chain
                if let Some(chain_type) = self.infer_method_chain_type(args_str, source) {
                    return Some(chain_type);
                }
            }
            _ => {}
        }

        // Look up the function signature
        if let Some(sig) = source.get_function_signature(func_name) {
            // Signature format: "Type1 -> Type2 -> ReturnType" or "ReturnType" (for 0-arg)
            // Extract the return type (last part after ->)
            if let Some(arrow_pos) = sig.rfind("->") {
                let ret_type = sig[arrow_pos + 2..].trim();
                if !ret_type.is_empty() {
                    return Some(ret_type.to_string());
                }
            } else if !sig.contains("->") {
                // No arrows means it's just the return type
                return Some(sig.trim().to_string());
            }
        }

        None
    }

    /// Get the return type of a method on a type
    fn get_method_return_type(type_name: &str, method_name: &str) -> Option<String> {
        // Buffer methods
        if type_name == "Buffer" {
            return match method_name {
                "append" => Some("Buffer".to_string()),
                "toString" => Some("String".to_string()),
                _ => None,
            };
        }

        // Float64Array methods
        if type_name == "Float64Array" {
            return match method_name {
                "length" => Some("Int".to_string()),
                "get" => Some("Float".to_string()),
                "set" | "slice" | "map" | "fill" => Some("Float64Array".to_string()),
                "toList" => Some("List".to_string()),
                "sum" | "min" | "max" | "mean" => Some("Float".to_string()),
                _ => None,
            };
        }

        // Int64Array methods
        if type_name == "Int64Array" {
            return match method_name {
                "length" | "sum" | "min" | "max" => Some("Int".to_string()),
                "get" => Some("Int".to_string()),
                "set" | "slice" | "map" | "fill" => Some("Int64Array".to_string()),
                "toList" => Some("List".to_string()),
                _ => None,
            };
        }

        // Float32Array methods
        if type_name == "Float32Array" {
            return match method_name {
                "length" => Some("Int".to_string()),
                "get" => Some("Float".to_string()),
                "set" | "slice" | "map" | "fill" => Some("Float32Array".to_string()),
                "toList" => Some("List".to_string()),
                "sum" | "min" | "max" | "mean" => Some("Float".to_string()),
                _ => None,
            };
        }

        // String methods
        if type_name == "String" {
            return match method_name {
                "length" => Some("Int".to_string()),
                "toUpper" | "toLower" | "trim" | "trimStart" | "trimEnd" |
                "replace" | "replaceAll" | "substring" | "repeat" |
                "padStart" | "padEnd" | "reverse" => Some("String".to_string()),
                "contains" | "startsWith" | "endsWith" | "isEmpty" => Some("Bool".to_string()),
                "split" | "chars" | "lines" | "words" => Some("List".to_string()),
                "indexOf" | "lastIndexOf" | "toInt" => Some("Int".to_string()),
                "toFloat" => Some("Float".to_string()),
                _ => None,
            };
        }

        // List methods
        if type_name == "List" || type_name.starts_with("List[") || type_name.starts_with("[") {
            return match method_name {
                "length" | "count" => Some("Int".to_string()),
                "isEmpty" => Some("Bool".to_string()),
                "map" | "filter" | "take" | "drop" | "reverse" | "sort" |
                "append" | "concat" | "flatten" => Some(type_name.to_string()),
                _ => None,
            };
        }

        // Map methods
        if type_name == "Map" || type_name.starts_with("Map[") {
            return match method_name {
                "size" => Some("Int".to_string()),
                "isEmpty" | "contains" => Some("Bool".to_string()),
                "insert" | "remove" | "merge" | "union" | "intersection" | "difference" => Some(type_name.to_string()),
                "keys" | "values" | "toList" => Some("List".to_string()),
                _ => None,
            };
        }

        // Set methods
        if type_name == "Set" || type_name.starts_with("Set[") {
            return match method_name {
                "size" => Some("Int".to_string()),
                "isEmpty" | "contains" | "isSubset" | "isProperSubset" => Some("Bool".to_string()),
                "insert" | "remove" | "union" | "intersection" | "difference" |
                "symmetricDifference" => Some(type_name.to_string()),
                "toList" => Some("List".to_string()),
                _ => None,
            };
        }

        // Option methods
        if type_name == "Option" || type_name.starts_with("Option ") || type_name.starts_with("Option[") {
            return match method_name {
                "isSome" | "isNone" => Some("Bool".to_string()),
                "map" | "flatMap" => Some(type_name.to_string()),
                // unwrap and unwrapOr return the inner type, but we can't easily extract it
                // Return "a" as placeholder - the actual type depends on the Option's type parameter
                _ => None,
            };
        }

        // Result methods
        if type_name == "Result" || type_name.starts_with("Result ") || type_name.starts_with("Result[") {
            return match method_name {
                "isOk" | "isErr" => Some("Bool".to_string()),
                "map" | "mapErr" => Some(type_name.to_string()),
                // unwrap and unwrapOr return the inner type
                _ => None,
            };
        }

        // Generic builtins that work on any type
        match method_name {
            "show" => return Some("String".to_string()),
            "hash" => return Some("Int".to_string()),
            "copy" => return Some(type_name.to_string()),
            _ => {}
        }

        None
    }

    /// Extract field names from an anonymous record type like "{ exitCode: Int, stdout: String }"
    fn extract_record_fields(type_str: &str) -> Vec<String> {
        let mut fields = Vec::new();

        // Strip outer braces and whitespace
        let inner = type_str.trim()
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .unwrap_or("")
            .trim();

        if inner.is_empty() {
            return fields;
        }

        // Parse field definitions, handling nested types
        let mut depth = 0;
        let mut current_field = String::new();

        for c in inner.chars() {
            match c {
                '{' | '(' | '[' => {
                    depth += 1;
                    current_field.push(c);
                }
                '}' | ')' | ']' => {
                    depth -= 1;
                    current_field.push(c);
                }
                ',' if depth == 0 => {
                    if let Some(field_name) = Self::extract_field_name(&current_field) {
                        fields.push(field_name);
                    }
                    current_field.clear();
                }
                _ => current_field.push(c),
            }
        }

        // Don't forget the last field
        if let Some(field_name) = Self::extract_field_name(&current_field) {
            fields.push(field_name);
        }

        fields
    }

    /// Extract field name from a field definition like "exitCode: Int"
    fn extract_field_name(field_def: &str) -> Option<String> {
        let trimmed = field_def.trim();
        if let Some(colon_pos) = trimmed.find(':') {
            let name = trimmed[..colon_pos].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        None
    }

    /// Apply a completion: returns the text to insert and cursor adjustment
    #[cfg(test)]
    pub fn apply_completion(
        &self,
        _context: &CompletionContext,
        item: &CompletionItem,
    ) -> (String, usize) {
        let text = &item.text;
        let cursor_advance = text.len();
        (text.clone(), cursor_advance)
    }
}

impl Default for Autocomplete {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock completion source for testing
    struct MockSource {
        functions: Vec<String>,
        types: Vec<String>,
        variables: Vec<String>,
        type_fields: HashMap<String, Vec<String>>,
        type_constructors: HashMap<String, Vec<String>>,
        variable_types: HashMap<String, String>,
        function_signatures: HashMap<String, String>,
    }

    impl MockSource {
        fn new() -> Self {
            Self {
                functions: Vec::new(),
                types: Vec::new(),
                variables: Vec::new(),
                type_fields: HashMap::new(),
                type_constructors: HashMap::new(),
                variable_types: HashMap::new(),
                function_signatures: HashMap::new(),
            }
        }

        fn with_functions(mut self, funcs: &[&str]) -> Self {
            self.functions = funcs.iter().map(|s| s.to_string()).collect();
            self
        }

        fn with_types(mut self, types: &[&str]) -> Self {
            self.types = types.iter().map(|s| s.to_string()).collect();
            self
        }

        fn with_variables(mut self, vars: &[&str]) -> Self {
            self.variables = vars.iter().map(|s| s.to_string()).collect();
            self
        }

        fn with_type_fields(mut self, type_name: &str, fields: &[&str]) -> Self {
            self.type_fields.insert(
                type_name.to_string(),
                fields.iter().map(|s| s.to_string()).collect(),
            );
            self
        }

        fn with_type_constructors(mut self, type_name: &str, ctors: &[&str]) -> Self {
            self.type_constructors.insert(
                type_name.to_string(),
                ctors.iter().map(|s| s.to_string()).collect(),
            );
            self
        }

        fn with_variable_type(mut self, var_name: &str, var_type: &str) -> Self {
            self.variable_types.insert(var_name.to_string(), var_type.to_string());
            self
        }

        fn with_function_signature(mut self, func_name: &str, signature: &str) -> Self {
            self.function_signatures.insert(func_name.to_string(), signature.to_string());
            self
        }
    }

    impl CompletionSource for MockSource {
        fn get_functions(&self) -> Vec<String> {
            self.functions.clone()
        }

        fn get_types(&self) -> Vec<String> {
            self.types.clone()
        }

        fn get_variables(&self) -> Vec<String> {
            self.variables.clone()
        }

        fn get_type_fields(&self, type_name: &str) -> Vec<String> {
            self.type_fields.get(type_name).cloned().unwrap_or_default()
        }

        fn is_function_public(&self, _name: &str) -> bool {
            // MockSource treats all functions as public by default
            true
        }

        fn get_type_constructors(&self, type_name: &str) -> Vec<String> {
            self.type_constructors.get(type_name).cloned().unwrap_or_default()
        }

        fn get_function_signature(&self, name: &str) -> Option<String> {
            self.function_signatures.get(name).cloned()
        }

        fn get_function_doc(&self, _name: &str) -> Option<String> {
            // MockSource doesn't have doc comments
            None
        }

        fn get_variable_type(&self, var_name: &str) -> Option<String> {
            self.variable_types.get(var_name).cloned()
        }

        fn get_ufcs_methods_for_type(&self, _type_name: &str) -> Vec<(String, String, Option<String>)> {
            // MockSource doesn't have UFCS methods
            Vec::new()
        }
    }

    #[test]
    fn test_parse_context_simple_identifier() {
        let ac = Autocomplete::new();

        let ctx = ac.parse_context("pri", 3);
        assert_eq!(ctx, CompletionContext::Identifier { prefix: "pri".to_string() });
    }

    #[test]
    fn test_parse_context_after_space() {
        let ac = Autocomplete::new();

        let ctx = ac.parse_context("let x = foo", 11);
        assert_eq!(ctx, CompletionContext::Identifier { prefix: "foo".to_string() });
    }

    #[test]
    fn test_parse_context_module_member() {
        let mut ac = Autocomplete::new();
        ac.modules.insert("Math".to_string());

        let ctx = ac.parse_context("Math.s", 6);
        assert_eq!(ctx, CompletionContext::ModuleMember {
            module: "Math".to_string(),
            prefix: "s".to_string()
        });
    }

    #[test]
    fn test_parse_context_module_member_no_prefix() {
        let mut ac = Autocomplete::new();
        ac.modules.insert("Math".to_string());

        let ctx = ac.parse_context("Math.", 5);
        assert_eq!(ctx, CompletionContext::ModuleMember {
            module: "Math".to_string(),
            prefix: "".to_string()
        });
    }

    #[test]
    fn test_parse_context_field_access() {
        let ac = Autocomplete::new();

        // point is lowercase, so it's field access not module
        let ctx = ac.parse_context("point.x", 7);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "point".to_string(),
            prefix: "x".to_string()
        });
    }

    #[test]
    fn test_parse_context_list_literal() {
        let ac = Autocomplete::new();

        // [1,2,3]. should be parsed as field access with list literal receiver
        let ctx = ac.parse_context("[1,2,3].", 8);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        });

        // [1,2,3].ma should match "map" prefix
        let ctx = ac.parse_context("[1,2,3].ma", 10);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "ma".to_string()
        });
    }

    #[test]
    fn test_parse_context_string_literal() {
        let ac = Autocomplete::new();

        let ctx = ac.parse_context("\"hello\".", 8);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "\"hello\"".to_string(),
            prefix: "".to_string()
        });
    }

    #[test]
    fn test_parse_context_map_literal() {
        let ac = Autocomplete::new();

        let ctx = ac.parse_context("%{\"a\": 1}.", 10);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "%{\"a\": 1}".to_string(),
            prefix: "".to_string()
        });
    }

    #[test]
    fn test_parse_context_set_literal() {
        let ac = Autocomplete::new();

        let ctx = ac.parse_context("#{1,2,3}.", 9);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "#{1,2,3}".to_string(),
            prefix: "".to_string()
        });
    }

    #[test]
    fn test_complete_identifier_functions() {
        let source = MockSource::new()
            .with_functions(&["print", "println", "parse", "add"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "pr".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "print"));
        assert!(items.iter().any(|i| i.text == "println"));
    }

    #[test]
    fn test_complete_identifier_types() {
        let source = MockSource::new()
            .with_types(&["Point", "Person", "List", "Map"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "P".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "Point"));
        assert!(items.iter().any(|i| i.text == "Person"));
    }

    #[test]
    fn test_complete_identifier_variables() {
        let source = MockSource::new()
            .with_variables(&["count", "counter", "name"]);

        let ac = Autocomplete::new();

        let ctx = CompletionContext::Identifier { prefix: "cou".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "count"));
        assert!(items.iter().any(|i| i.text == "counter"));
    }

    #[test]
    fn test_complete_identifier_modules() {
        let source = MockSource::new()
            .with_functions(&["Math.sin", "Math.cos", "IO.print"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "M".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "Math" && i.kind == CompletionKind::Module));
    }

    #[test]
    fn test_complete_module_member_functions() {
        let source = MockSource::new()
            .with_functions(&["Math.sin", "Math.cos", "Math.sqrt", "Math.abs", "IO.print"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::ModuleMember {
            module: "Math".to_string(),
            prefix: "s".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "sin"));
        assert!(items.iter().any(|i| i.text == "sqrt"));
    }

    #[test]
    fn test_complete_module_member_all() {
        let source = MockSource::new()
            .with_functions(&["Math.sin", "Math.cos"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::ModuleMember {
            module: "Math".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "sin"));
        assert!(items.iter().any(|i| i.text == "cos"));
    }

    #[test]
    fn test_complete_module_member_types() {
        let source = MockSource::new()
            .with_types(&["Geometry.Point", "Geometry.Line", "Other.Thing"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::ModuleMember {
            module: "Geometry".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "Point" && i.kind == CompletionKind::Type));
        assert!(items.iter().any(|i| i.text == "Line" && i.kind == CompletionKind::Type));
    }

    #[test]
    fn test_complete_field_access() {
        let source = MockSource::new()
            .with_type_fields("Point", &["x", "y", "z"]);

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "Point".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have fields x, y, z plus generic builtins (show, hash, copy)
        assert!(items.iter().any(|i| i.text == "x" && i.kind == CompletionKind::Field));
        assert!(items.iter().any(|i| i.text == "y" && i.kind == CompletionKind::Field));
        assert!(items.iter().any(|i| i.text == "z" && i.kind == CompletionKind::Field));
    }

    #[test]
    fn test_complete_field_access_with_prefix() {
        let source = MockSource::new()
            .with_type_fields("Person", &["name", "age", "address", "nickname"]);

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "Person".to_string(),
            prefix: "n".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "name"));
        assert!(items.iter().any(|i| i.text == "nickname"));
    }

    #[test]
    fn test_complete_variable_field_access() {
        // Test completing fields on a variable with a known type
        let source = MockSource::new()
            .with_variables(&["result"])
            .with_variable_type("result", "{ exitCode: Int, stdout: String, stderr: String }");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "result".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have fields exitCode, stdout, stderr plus generic builtins
        assert!(items.iter().any(|i| i.text == "exitCode"));
        assert!(items.iter().any(|i| i.text == "stdout"));
        assert!(items.iter().any(|i| i.text == "stderr"));
    }

    #[test]
    fn test_complete_variable_field_access_with_prefix() {
        // Test prefix matching on variable fields
        let source = MockSource::new()
            .with_variables(&["result"])
            .with_variable_type("result", "{ exitCode: Int, stdout: String, stderr: String }");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "result".to_string(),
            prefix: "st".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2, "Should find stdout and stderr: {:?}", items);
        assert!(items.iter().any(|i| i.text == "stdout"));
        assert!(items.iter().any(|i| i.text == "stderr"));
    }

    #[test]
    fn test_complete_constructors() {
        let source = MockSource::new()
            .with_type_constructors("Option", &["Some", "None"]);

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "Option".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have constructors Some, None plus generic builtins
        assert!(items.iter().any(|i| i.text == "Some" && i.kind == CompletionKind::Constructor));
        assert!(items.iter().any(|i| i.text == "None" && i.kind == CompletionKind::Constructor));
    }

    #[test]
    fn test_case_insensitive_matching() {
        let source = MockSource::new()
            .with_functions(&["println", "Printf", "PRINT_ALL"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "print".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_exact_prefix_sorted_first() {
        let source = MockSource::new()
            .with_functions(&["map", "mapValues", "Map", "mapper"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "map".to_string() };
        let items = ac.get_completions(&ctx, &source);

        // "map" should come before "Map" because it's an exact case match
        assert!(!items.is_empty());
        assert_eq!(items[0].text, "map");
    }

    #[test]
    fn test_nested_modules() {
        let source = MockSource::new()
            .with_functions(&[
                "Net.Http.get",
                "Net.Http.post",
                "Net.Tcp.connect",
            ]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // Completing "Net." should show Http and Tcp
        let ctx = CompletionContext::ModuleMember {
            module: "Net".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have Http and Tcp as submodules
        assert!(items.iter().any(|i| i.text == "Http" && i.kind == CompletionKind::Module));
        assert!(items.iter().any(|i| i.text == "Tcp" && i.kind == CompletionKind::Module));
    }

    #[test]
    fn test_apply_completion() {
        let ac = Autocomplete::new();

        let ctx = CompletionContext::Identifier { prefix: "pr".to_string() };
        let item = CompletionItem {
            text: "println".to_string(),
            label: "println".to_string(),
            kind: CompletionKind::Function,
            doc: None,
        };

        let (text, advance) = ac.apply_completion(&ctx, &item);
        assert_eq!(text, "println");
        assert_eq!(advance, 7);
    }

    #[test]
    fn test_empty_prefix() {
        let source = MockSource::new()
            .with_functions(&["add", "sub"])
            .with_types(&["Int", "String"])
            .with_variables(&["x", "y"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "".to_string() };
        let items = ac.get_completions(&ctx, &source);

        // Should return all items
        assert_eq!(items.len(), 6);
    }

    #[test]
    fn test_no_matches() {
        let source = MockSource::new()
            .with_functions(&["add", "sub"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::Identifier { prefix: "xyz".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.is_empty());
    }

    #[test]
    fn test_update_from_source_clears_old_data() {
        let source1 = MockSource::new()
            .with_functions(&["old_func"]);

        let source2 = MockSource::new()
            .with_functions(&["new_func"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source1);
        ac.update_from_source(&source2);

        let ctx = CompletionContext::Identifier { prefix: "".to_string() };
        let items = ac.get_completions(&ctx, &source2);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "new_func");
    }

    // Tests for parse_imports

    #[test]
    fn test_parse_imports_simple() {
        let source = "use Math.add";
        let imports = parse_imports(source);
        assert_eq!(imports, vec!["Math.add"]);
    }

    #[test]
    fn test_parse_imports_multiple_lines() {
        let source = r#"
            use Math.add
            use IO.print
        "#;
        let imports = parse_imports(source);
        assert_eq!(imports, vec!["Math.add", "IO.print"]);
    }

    #[test]
    fn test_parse_imports_brace_syntax() {
        let source = "use Math.{add, sub, mul}";
        let imports = parse_imports(source);
        assert_eq!(imports, vec!["Math.add", "Math.sub", "Math.mul"]);
    }

    #[test]
    fn test_parse_imports_brace_with_spaces() {
        let source = "use Math.{ add , sub }";
        let imports = parse_imports(source);
        assert_eq!(imports, vec!["Math.add", "Math.sub"]);
    }

    #[test]
    fn test_parse_imports_mixed() {
        let source = r#"
            use Math.{sin, cos}
            use IO.println
            use List.map
        "#;
        let imports = parse_imports(source);
        assert_eq!(imports, vec!["Math.sin", "Math.cos", "IO.println", "List.map"]);
    }

    #[test]
    fn test_parse_imports_ignores_non_use() {
        let source = r#"
            // use this is a comment
            add(x, y) = x + y
            use Math.add
        "#;
        let imports = parse_imports(source);
        assert_eq!(imports, vec!["Math.add"]);
    }

    #[test]
    fn test_extract_module() {
        assert_eq!(extract_module("Math.add"), Some("Math"));
        assert_eq!(extract_module("Math.Trig.sin"), Some("Math.Trig"));
        assert_eq!(extract_module("add"), None);
    }

    // Tests for module-aware completion

    #[test]
    fn test_complete_with_current_module() {
        let source = MockSource::new()
            .with_functions(&["Math.add", "Math.sub", "Math.mul", "IO.print"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // When in Math module, Math functions should be available by short name
        let ctx = CompletionContext::Identifier { prefix: "a".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("Math"), &[]);

        // Should find "add" from same module
        assert!(items.iter().any(|i| i.text == "add"));
    }

    #[test]
    fn test_complete_with_imports() {
        let source = MockSource::new()
            .with_functions(&["Math.add", "Math.sub", "IO.print", "top_level"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // With IO.print imported, "print" should be available
        let ctx = CompletionContext::Identifier { prefix: "pr".to_string() };
        let imports = vec!["IO.print".to_string()];
        let items = ac.get_completions_with_context(&ctx, &source, None, &imports);

        // Should find "print" from imports
        assert!(items.iter().any(|i| i.text == "print"));
        // Label should contain "(imp)" suffix for imported functions
        assert!(items.iter().any(|i| i.label.contains("(imp)")));
    }

    #[test]
    fn test_complete_with_module_and_imports() {
        let source = MockSource::new()
            .with_functions(&["Math.add", "Math.sub", "IO.print", "List.map"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // In Math module with IO.print imported
        let ctx = CompletionContext::Identifier { prefix: "".to_string() };
        let imports = vec!["IO.print".to_string()];
        let items = ac.get_completions_with_context(&ctx, &source, Some("Math"), &imports);

        // Should find: add, sub (from Math), print (imported), Math, IO, List (modules)
        assert!(items.iter().any(|i| i.text == "add"));
        assert!(items.iter().any(|i| i.text == "sub"));
        assert!(items.iter().any(|i| i.text == "print"));
        assert!(items.iter().any(|i| i.text == "Math" && i.kind == CompletionKind::Module));
        assert!(items.iter().any(|i| i.text == "IO" && i.kind == CompletionKind::Module));
    }

    // Tests for lowercase module names (user modules)

    #[test]
    fn test_complete_lowercase_module_same_module_functions() {
        // Scenario: editing utils.nos which has foo and bar
        // When typing in bar's body, foo should be available
        let source = MockSource::new()
            .with_functions(&["utils.foo", "utils.bar", "utils.helper", "other.something"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // In utils module, typing "f" should complete to "foo"
        let ctx = CompletionContext::Identifier { prefix: "f".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("utils"), &[]);

        assert!(items.iter().any(|i| i.text == "foo"), "foo should be available in same module");
    }

    #[test]
    fn test_complete_lowercase_module_all_same_module_functions() {
        let source = MockSource::new()
            .with_functions(&["utils.foo", "utils.bar", "utils.baz", "math.add"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // In utils module, empty prefix should show all utils functions
        let ctx = CompletionContext::Identifier { prefix: "".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("utils"), &[]);

        assert!(items.iter().any(|i| i.text == "foo"), "foo should be available");
        assert!(items.iter().any(|i| i.text == "bar"), "bar should be available");
        assert!(items.iter().any(|i| i.text == "baz"), "baz should be available");
        // math.add should NOT be directly available without prefix
        assert!(!items.iter().any(|i| i.text == "add"), "add should NOT be directly available");
    }

    #[test]
    fn test_complete_lowercase_module_prefix_match() {
        let source = MockSource::new()
            .with_functions(&["utils.process", "utils.parse", "utils.print_debug", "other.parse"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // In utils module, typing "p" should show process, parse, print_debug
        let ctx = CompletionContext::Identifier { prefix: "p".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("utils"), &[]);

        assert_eq!(items.iter().filter(|i| i.kind == CompletionKind::Function || i.kind == CompletionKind::PublicFunction).count(), 3);
        assert!(items.iter().any(|i| i.text == "process"));
        assert!(items.iter().any(|i| i.text == "parse"));
        assert!(items.iter().any(|i| i.text == "print_debug"));
    }

    #[test]
    fn test_complete_module_member_lowercase() {
        // When typing "utils." should show utils functions
        let source = MockSource::new()
            .with_functions(&["utils.foo", "utils.bar", "math.add"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::ModuleMember {
            module: "utils".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "foo"));
        assert!(items.iter().any(|i| i.text == "bar"));
    }

    #[test]
    fn test_complete_module_member_lowercase_with_prefix() {
        let source = MockSource::new()
            .with_functions(&["utils.foo", "utils.filter", "utils.bar"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        let ctx = CompletionContext::ModuleMember {
            module: "utils".to_string(),
            prefix: "f".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "foo"));
        assert!(items.iter().any(|i| i.text == "filter"));
    }

    #[test]
    fn test_parse_context_lowercase_module() {
        let mut ac = Autocomplete::new();
        ac.modules.insert("utils".to_string());

        // typing "utils.f" should be parsed as module member
        let ctx = ac.parse_context("utils.f", 7);
        assert_eq!(ctx, CompletionContext::ModuleMember {
            module: "utils".to_string(),
            prefix: "f".to_string()
        });
    }

    #[test]
    fn test_parse_context_lowercase_module_no_prefix() {
        let mut ac = Autocomplete::new();
        ac.modules.insert("utils".to_string());

        let ctx = ac.parse_context("utils.", 6);
        assert_eq!(ctx, CompletionContext::ModuleMember {
            module: "utils".to_string(),
            prefix: "".to_string()
        });
    }

    #[test]
    fn test_modules_extracted_from_lowercase_functions() {
        let source = MockSource::new()
            .with_functions(&["utils.foo", "math.add", "io.print"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // Modules should be extracted from function names
        assert!(ac.modules.contains("utils"));
        assert!(ac.modules.contains("math"));
        assert!(ac.modules.contains("io"));
    }

    #[test]
    fn test_complete_shows_modules_as_completions() {
        let source = MockSource::new()
            .with_functions(&["utils.foo", "math.add", "io.print"]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // Typing "u" should suggest "utils" as a module
        let ctx = CompletionContext::Identifier { prefix: "u".to_string() };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "utils" && i.kind == CompletionKind::Module));
    }

    #[test]
    fn test_realistic_editor_scenario() {
        // Simulating: editing utils.nos, file contains foo() and bar()
        // User is typing in bar's body and wants to call foo
        let source = MockSource::new()
            .with_functions(&[
                "utils.foo",
                "utils.bar",
                "utils.helper",
                "Math.sin",  // Foreign library (uppercase)
                "Math.cos",
            ]);

        let mut ac = Autocomplete::new();
        ac.update_from_source(&source);

        // Scenario 1: typing "fo" should complete to "foo"
        let ctx = CompletionContext::Identifier { prefix: "fo".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("utils"), &[]);
        assert!(items.iter().any(|i| i.text == "foo"), "foo from same module should be suggested");

        // Scenario 2: typing "he" should complete to "helper"
        let ctx = CompletionContext::Identifier { prefix: "he".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("utils"), &[]);
        assert!(items.iter().any(|i| i.text == "helper"), "helper from same module should be suggested");

        // Scenario 3: typing "Math." should show Math functions
        let ctx = CompletionContext::ModuleMember {
            module: "Math".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "sin"));
        assert!(items.iter().any(|i| i.text == "cos"));

        // Scenario 4: typing "M" should suggest Math module
        let ctx = CompletionContext::Identifier { prefix: "M".to_string() };
        let items = ac.get_completions_with_context(&ctx, &source, Some("utils"), &[]);
        assert!(items.iter().any(|i| i.text == "Math" && i.kind == CompletionKind::Module));
    }

    // Tests for extract_module_from_editor_name

    #[test]
    fn test_extract_module_from_editor_name_qualified() {
        // Qualified function names: "utils.bar" -> "utils"
        assert_eq!(extract_module_from_editor_name("utils.bar"), Some("utils".to_string()));
        assert_eq!(extract_module_from_editor_name("utils.foo"), Some("utils".to_string()));
        assert_eq!(extract_module_from_editor_name("Math.sin"), Some("Math".to_string()));
        assert_eq!(extract_module_from_editor_name("Net.Http.get"), Some("Net.Http".to_string()));
    }

    #[test]
    fn test_extract_module_from_editor_name_file_path() {
        // File paths: "utils.nos" -> "utils"
        assert_eq!(extract_module_from_editor_name("utils.nos"), Some("utils".to_string()));
        assert_eq!(extract_module_from_editor_name("math.nos"), Some("math".to_string()));
        assert_eq!(extract_module_from_editor_name("/path/to/utils.nos"), Some("utils".to_string()));
        assert_eq!(extract_module_from_editor_name("src/modules/math.nos"), Some("math".to_string()));
    }

    #[test]
    fn test_extract_module_from_editor_name_simple() {
        // Simple names without module: "add" -> None
        assert_eq!(extract_module_from_editor_name("add"), None);
        assert_eq!(extract_module_from_editor_name("foo"), None);
    }

    #[test]
    fn test_extract_module_from_editor_name_edge_cases() {
        // Edge cases
        assert_eq!(extract_module_from_editor_name(""), None);
        assert_eq!(extract_module_from_editor_name(".nos"), None);  // Empty module name
    }

    // Tests for method completion on builtin types

    #[test]
    fn test_complete_map_methods() {
        let source = MockSource::new()
            .with_variables(&["m"])
            .with_variable_type("m", "Map");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "m".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have Map methods
        assert!(items.iter().any(|i| i.text == "get" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "insert" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "contains" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "keys" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "size" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_string_methods() {
        let source = MockSource::new()
            .with_variables(&["s"])
            .with_variable_type("s", "String");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "s".to_string(),
            prefix: "to".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have String methods starting with "to"
        assert!(items.iter().any(|i| i.text == "toUpper"));
        assert!(items.iter().any(|i| i.text == "toLower"));
        assert!(items.iter().any(|i| i.text == "toInt"));
        assert!(items.iter().any(|i| i.text == "toFloat"));
    }

    #[test]
    fn test_complete_list_methods() {
        let source = MockSource::new()
            .with_variables(&["lst"])
            .with_variable_type("lst", "List");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "lst".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have List methods
        assert!(items.iter().any(|i| i.text == "map" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "filter" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "fold" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "reverse" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_set_methods() {
        let source = MockSource::new()
            .with_variables(&["set1"])
            .with_variable_type("set1", "Set");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "set1".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have Set methods
        assert!(items.iter().any(|i| i.text == "union" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "intersection" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "difference" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "contains" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_methods_with_signature() {
        let source = MockSource::new()
            .with_variables(&["m"])
            .with_variable_type("m", "Map");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "m".to_string(),
            prefix: "ins".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have insert method with signature in label
        assert!(items.len() == 1);
        assert_eq!(items[0].text, "insert");
        assert!(items[0].label.contains("(key, value) -> Map"));
    }

    // Tests for literal type detection

    #[test]
    fn test_complete_list_literal_methods() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // [1,2,3]. should suggest List methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "map" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "filter" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_string_literal_methods() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // "hello". should suggest String methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "\"hello\"".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "toUpper" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "trim" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_map_literal_methods() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // %{"key": "value"}. should suggest Map methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "%{\"key\": \"value\"}".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "get" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "insert" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_set_literal_methods() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // #{1,2,3}. should suggest Set methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "#{1,2,3}".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "union" && i.kind == CompletionKind::Method));
        assert!(items.iter().any(|i| i.text == "contains" && i.kind == CompletionKind::Method));
    }

    #[test]
    fn test_complete_tuple_literal_methods() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // (1, "hello", 3.14). should suggest Tuple methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "(1, \"hello\", 3.14)".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert!(items.iter().any(|i| i.text == "length" && i.kind == CompletionKind::Method),
            "Tuple should have length method, got: {:?}", items);
    }

    #[test]
    fn test_complete_tuple_literal_with_prefix() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // (1, 2).le should suggest length
        let ctx = CompletionContext::FieldAccess {
            receiver: "(1, 2)".to_string(),
            prefix: "le".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 1, "Should only match 'length', got: {:?}", items);
        assert_eq!(items[0].text, "length");
    }

    #[test]
    fn test_tuple_literal_detection() {
        // Tuple must have comma to distinguish from parenthesized expression
        assert_eq!(Autocomplete::detect_literal_type("(1, 2)"), Some("Tuple"));
        assert_eq!(Autocomplete::detect_literal_type("(1, 2, 3)"), Some("Tuple"));
        assert_eq!(Autocomplete::detect_literal_type("(\"a\", \"b\")"), Some("Tuple"));

        // Single element in parens is NOT a tuple
        assert_eq!(Autocomplete::detect_literal_type("(1)"), None);

        // Other literals should still work
        assert_eq!(Autocomplete::detect_literal_type("[1, 2]"), Some("List"));
        assert_eq!(Autocomplete::detect_literal_type("\"hello\""), Some("String"));
    }

    #[test]
    fn test_numeric_literal_detection() {
        // Plain integers
        assert_eq!(Autocomplete::detect_literal_type("42"), Some("Int"));
        assert_eq!(Autocomplete::detect_literal_type("0"), Some("Int"));
        assert_eq!(Autocomplete::detect_literal_type("-5"), Some("Int"));
        assert_eq!(Autocomplete::detect_literal_type("1_000_000"), Some("Int"));

        // Floats
        assert_eq!(Autocomplete::detect_literal_type("3.14"), Some("Float"));
        assert_eq!(Autocomplete::detect_literal_type("-2.5"), Some("Float"));
        assert_eq!(Autocomplete::detect_literal_type("0.0"), Some("Float"));

        // Typed integers
        assert_eq!(Autocomplete::detect_literal_type("42i8"), Some("Int8"));
        assert_eq!(Autocomplete::detect_literal_type("42i16"), Some("Int16"));
        assert_eq!(Autocomplete::detect_literal_type("42i32"), Some("Int32"));
        assert_eq!(Autocomplete::detect_literal_type("42i64"), Some("Int64"));
        assert_eq!(Autocomplete::detect_literal_type("42u8"), Some("UInt8"));
        assert_eq!(Autocomplete::detect_literal_type("42u16"), Some("UInt16"));
        assert_eq!(Autocomplete::detect_literal_type("42u32"), Some("UInt32"));
        assert_eq!(Autocomplete::detect_literal_type("42u64"), Some("UInt64"));

        // Typed floats
        assert_eq!(Autocomplete::detect_literal_type("3.14f32"), Some("Float32"));
        assert_eq!(Autocomplete::detect_literal_type("3.14f64"), Some("Float64"));

        // BigInt
        assert_eq!(Autocomplete::detect_literal_type("1000000n"), Some("BigInt"));
    }

    #[test]
    fn test_numeric_literal_autocomplete() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // 42.as should suggest asInt8, asInt16, etc.
        let ctx = CompletionContext::FieldAccess {
            receiver: "42".to_string(),
            prefix: "as".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should have all as* conversion methods
        assert!(items.iter().any(|i| i.text == "asInt8"), "Should suggest asInt8");
        assert!(items.iter().any(|i| i.text == "asInt32"), "Should suggest asInt32");
        assert!(items.iter().any(|i| i.text == "asFloat64"), "Should suggest asFloat64");
    }

    #[test]
    fn test_complete_literal_with_prefix() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // [1,2,3].ma should suggest map
        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "ma".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // Should match "map" and "maximum"
        assert_eq!(items.len(), 2);
        assert!(items.iter().any(|i| i.text == "map"));
        assert!(items.iter().any(|i| i.text == "maximum"));
    }

    #[test]
    fn test_full_flow_list_literal_completions() {
        // Test the complete flow from parse_context to get_completions
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // Simulate typing "[1,2,3]." and pressing Tab
        let line = "[1,2,3].";
        let cursor_pos = line.len(); // cursor is at position 8

        // Step 1: parse_context should extract the literal
        let ctx = ac.parse_context(line, cursor_pos);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        });

        // Step 2: get_completions should return List methods
        let items = ac.get_completions(&ctx, &source);
        assert!(!items.is_empty(), "Should have completions for list literal");
        assert!(items.iter().any(|i| i.text == "map"), "Should suggest map");
        assert!(items.iter().any(|i| i.text == "filter"), "Should suggest filter");
        assert!(items.iter().any(|i| i.text == "fold"), "Should suggest fold");
    }

    #[test]
    fn test_full_flow_string_literal_completions() {
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "\"hello\".";
        let cursor_pos = line.len();

        let ctx = ac.parse_context(line, cursor_pos);
        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "\"hello\"".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(!items.is_empty(), "Should have completions for string literal");
        assert!(items.iter().any(|i| i.text == "trim"), "Should suggest trim");
        assert!(items.iter().any(|i| i.text == "toUpper"), "Should suggest toUpper");
    }

    // ============================================================
    // Comprehensive autocomplete tests for REPL and editor
    // ============================================================

    #[test]
    fn test_nested_list_literal_completions() {
        // Nested lists should still be recognized as List type
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "[[1,2], [3,4]].";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[[1,2], [3,4]]".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "flatten"), "Nested list should suggest flatten");
        assert!(items.iter().any(|i| i.text == "map"), "Nested list should suggest map");
    }

    #[test]
    fn test_map_literal_with_prefix() {
        // Map literal with partial method name
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "%{\"a\": 1}.ge";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "%{\"a\": 1}".to_string(),
            prefix: "ge".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert_eq!(items.len(), 1, "Should only match 'get'");
        assert_eq!(items[0].text, "get");
    }

    #[test]
    fn test_set_literal_with_prefix() {
        // Set literal with partial method name
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "#{1, 2, 3}.con";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "#{1, 2, 3}".to_string(),
            prefix: "con".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert_eq!(items.len(), 1, "Should only match 'contains'");
        assert_eq!(items[0].text, "contains");
    }

    #[test]
    fn test_string_with_escaped_quotes() {
        // String containing escaped quotes
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = r#""hello \"world\""."#;
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: r#""hello \"world\"""#.to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "length"), "Should suggest string methods");
    }

    #[test]
    fn test_list_after_assignment() {
        // List literal after assignment operator
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "x = [1,2,3].";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "map"), "Should suggest List methods after assignment");
    }

    #[test]
    fn test_empty_list_literal() {
        // Empty list should still get completions
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "[].";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[]".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(!items.is_empty(), "Empty list should still have completions");
        assert!(items.iter().any(|i| i.text == "map"));
    }

    #[test]
    fn test_empty_string_literal() {
        // Empty string should still get completions
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "\"\".";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "\"\"".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "isEmpty"), "Empty string should suggest isEmpty");
    }

    #[test]
    fn test_empty_map_literal() {
        // Empty map should still get completions
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "%{}.";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "%{}".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "isEmpty"), "Empty map should suggest isEmpty");
        assert!(items.iter().any(|i| i.text == "insert"), "Empty map should suggest insert");
    }

    #[test]
    fn test_empty_set_literal() {
        // Empty set should still get completions
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "#{}.";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "#{}".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "isEmpty"), "Empty set should suggest isEmpty");
        assert!(items.iter().any(|i| i.text == "insert"), "Empty set should suggest insert");
    }

    #[test]
    fn test_literal_in_function_call() {
        // List literal as function argument
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "foo([1,2,3].";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "map"), "Should work inside function call");
    }

    #[test]
    fn test_case_insensitive_prefix_matching() {
        // Prefix matching should be case insensitive
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // Uppercase prefix should match lowercase methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "MAP".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "map"), "MAP should match map");

        // Mixed case prefix
        let ctx = CompletionContext::FieldAccess {
            receiver: "\"hello\"".to_string(),
            prefix: "ToUp".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "toUpper"), "ToUp should match toUpper");
    }

    #[test]
    fn test_method_signature_in_label() {
        // Completions should include method signatures
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "fold".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        // "fold" matches both "fold" and "foldr"
        assert_eq!(items.len(), 2);
        let fold_item = items.iter().find(|i| i.text == "fold").unwrap();
        assert!(fold_item.label.contains("(acc, f)"), "Label should contain signature");
    }

    #[test]
    fn test_all_list_methods_present() {
        // Verify all expected List methods are available
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        let expected_methods = vec![
            "map", "filter", "fold", "foldr", "any", "all", "find",
            "take", "drop", "reverse", "sort", "concat", "flatten",
            "unique", "zip", "partition", "contains", "maximum", "minimum"
        ];

        for method in expected_methods {
            assert!(items.iter().any(|i| i.text == method),
                "List should have method: {}", method);
        }
    }

    #[test]
    fn test_all_string_methods_present() {
        // Verify all expected String methods are available
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "\"test\"".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        let expected_methods = vec![
            "length", "chars", "trim", "trimStart", "trimEnd",
            "toUpper", "toLower", "contains", "startsWith", "endsWith",
            "replace", "indexOf", "substring", "repeat", "reverse", "isEmpty"
        ];

        for method in expected_methods {
            assert!(items.iter().any(|i| i.text == method),
                "String should have method: {}", method);
        }
    }

    #[test]
    fn test_all_map_methods_present() {
        // Verify all expected Map methods are available
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "%{}".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        let expected_methods = vec![
            "get", "insert", "remove", "contains", "keys", "values",
            "size", "isEmpty", "merge"
        ];

        for method in expected_methods {
            assert!(items.iter().any(|i| i.text == method),
                "Map should have method: {}", method);
        }
    }

    #[test]
    fn test_all_set_methods_present() {
        // Verify all expected Set methods are available
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "#{}".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        let expected_methods = vec![
            "contains", "insert", "remove", "size", "isEmpty",
            "union", "intersection", "difference", "toList"
        ];

        for method in expected_methods {
            assert!(items.iter().any(|i| i.text == method),
                "Set should have method: {}", method);
        }
    }

    #[test]
    fn test_completion_kind_is_method() {
        // All builtin type completions should have Method kind
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        for item in &items {
            assert_eq!(item.kind, CompletionKind::Method,
                "Completion '{}' should have Method kind", item.text);
        }
    }

    #[test]
    fn test_no_duplicate_completions() {
        // Should not have duplicate method names
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        let mut seen = std::collections::HashSet::new();
        for item in &items {
            assert!(seen.insert(&item.text),
                "Duplicate completion found: {}", item.text);
        }
    }

    #[test]
    fn test_complex_nested_expression() {
        // Complex nested structure
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "%{\"users\": [{\"name\": \"Alice\"}]}.";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "%{\"users\": [{\"name\": \"Alice\"}]}".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "get"), "Complex map should suggest get");
    }

    #[test]
    fn test_chained_method_completions() {
        // After a method call, should still suggest methods
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // [1,2,3].map(x => x). should suggest List methods
        let line = "[1,2,3].map(x => x).";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3].map(x => x)".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);

        // Should still get List methods because [1,2,3].map(...) returns a List
        assert!(items.iter().any(|i| i.text == "filter"),
            "Chained list expression should suggest filter");
    }

    #[test]
    fn test_multiple_chained_methods() {
        // Multiple chained method calls
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "[1,2,3].map(x => x * 2).filter(x => x > 2).";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3].map(x => x * 2).filter(x => x > 2)".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "fold"),
            "Multiple chained methods should still suggest List methods");
    }

    #[test]
    fn test_chained_string_methods() {
        // Chained string methods
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "\"hello\".trim().toUpper().";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "\"hello\".trim().toUpper()".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "toLower"),
            "Chained string methods should suggest String methods");
    }

    #[test]
    fn test_chained_map_methods() {
        // Chained map methods
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "%{\"a\": 1}.insert(\"b\", 2).";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "%{\"a\": 1}.insert(\"b\", 2)".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "get"),
            "Chained map methods should suggest Map methods");
    }

    #[test]
    fn test_chained_set_methods() {
        // Chained set methods
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "#{1,2}.insert(3).union(#{4}).";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "#{1,2}.insert(3).union(#{4})".to_string(),
            prefix: "".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert!(items.iter().any(|i| i.text == "contains"),
            "Chained set methods should suggest Set methods");
    }

    #[test]
    fn test_chained_with_prefix() {
        // Chained methods with partial method name
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let line = "[1,2,3].map(x => x).fil";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "[1,2,3].map(x => x)".to_string(),
            prefix: "fil".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "filter");
    }

    #[test]
    fn test_completions_include_docstrings() {
        // Method completions should include docstrings
        let source = MockSource::new();
        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "[1,2,3]".to_string(),
            prefix: "map".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "map");
        assert!(items[0].doc.is_some(), "Should have a docstring");
        assert!(items[0].doc.as_ref().unwrap().contains("Apply"),
            "Docstring should describe the method");
    }

    #[test]
    fn test_all_builtin_methods_have_docstrings() {
        // Every builtin method should have a docstring
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // Test List methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "[]".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        for item in &items {
            assert!(item.doc.is_some(),
                "List method '{}' should have a docstring", item.text);
            assert!(!item.doc.as_ref().unwrap().is_empty(),
                "List method '{}' docstring should not be empty", item.text);
        }

        // Test String methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "\"\"".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        for item in &items {
            assert!(item.doc.is_some(),
                "String method '{}' should have a docstring", item.text);
        }

        // Test Map methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "%{}".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        for item in &items {
            assert!(item.doc.is_some(),
                "Map method '{}' should have a docstring", item.text);
        }

        // Test Set methods
        let ctx = CompletionContext::FieldAccess {
            receiver: "#{}".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);
        for item in &items {
            assert!(item.doc.is_some(),
                "Set method '{}' should have a docstring", item.text);
        }
    }

    #[test]
    fn test_function_call_return_type_completions() {
        // When typing greet(Person("petter"))., should get String methods
        // because greet returns String
        let source = MockSource::new()
            .with_function_signature("greet", "Person -> String");
        let ac = Autocomplete::new();

        // Test the context parsing
        let line = "greet(Person(\"petter\")).";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "greet(Person(\"petter\"))".to_string(),
            prefix: "".to_string()
        });

        // Should get String methods
        let items = ac.get_completions(&ctx, &source);
        println!("Completions for greet(Person(\"petter\")).: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());
        assert!(items.iter().any(|i| i.text == "toUpper"),
            "Should suggest String method 'toUpper' since greet returns String");
        assert!(items.iter().any(|i| i.text == "length"),
            "Should suggest String method 'length' since greet returns String");
    }

    #[test]
    fn test_function_call_chain_return_type() {
        // Test chaining: greet(Person("petter")).toUpper().
        let source = MockSource::new()
            .with_function_signature("greet", "Person -> String");
        let ac = Autocomplete::new();

        let line = "greet(Person(\"petter\")).toUpper().";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "greet(Person(\"petter\")).toUpper()".to_string(),
            prefix: "".to_string()
        });

        // toUpper returns String, so should still get String methods
        let items = ac.get_completions(&ctx, &source);
        println!("Completions for toUpper chain: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());
        assert!(items.iter().any(|i| i.text == "length"),
            "Should suggest String methods after toUpper()");
    }

    #[test]
    fn test_function_call_with_prefix_filter() {
        // Test prefix filtering: greet(Person("petter")).to
        let source = MockSource::new()
            .with_function_signature("greet", "Person -> String");
        let ac = Autocomplete::new();

        let line = "greet(Person(\"petter\")).to";
        let ctx = ac.parse_context(line, line.len());

        assert_eq!(ctx, CompletionContext::FieldAccess {
            receiver: "greet(Person(\"petter\"))".to_string(),
            prefix: "to".to_string()
        });

        let items = ac.get_completions(&ctx, &source);
        println!("Completions with 'to' prefix: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());
        // Should only get methods starting with 'to'
        assert!(items.iter().any(|i| i.text == "toUpper"));
        assert!(items.iter().any(|i| i.text == "toLower"));
        assert!(items.iter().any(|i| i.text == "toInt"));
        // Should NOT have methods not starting with 'to'
        assert!(!items.iter().any(|i| i.text == "length"));
    }

    /// Wrapper to implement CompletionSource for ReplEngine (same as in repl_panel.rs)
    struct ReplEngineSource<'a> {
        engine: &'a nostos_repl::ReplEngine,
    }

    impl<'a> CompletionSource for ReplEngineSource<'a> {
        fn get_functions(&self) -> Vec<String> {
            self.engine.get_functions()
        }

        fn get_types(&self) -> Vec<String> {
            self.engine.get_types()
        }

        fn get_variables(&self) -> Vec<String> {
            self.engine.get_variables()
        }

        fn get_type_fields(&self, type_name: &str) -> Vec<String> {
            self.engine.get_type_fields(type_name)
        }

        fn is_function_public(&self, name: &str) -> bool {
            self.engine.is_function_public(name)
        }

        fn get_type_constructors(&self, type_name: &str) -> Vec<String> {
            self.engine.get_type_constructors(type_name)
        }

        fn get_function_signature(&self, name: &str) -> Option<String> {
            self.engine.get_function_signature(name)
        }

        fn get_function_doc(&self, name: &str) -> Option<String> {
            self.engine.get_function_doc(name)
        }

        fn get_variable_type(&self, var_name: &str) -> Option<String> {
            self.engine.get_variable_type(var_name)
        }

        fn get_ufcs_methods_for_type(&self, type_name: &str) -> Vec<(String, String, Option<String>)> {
            self.engine.get_ufcs_methods_for_type(type_name)
        }
    }

    #[test]
    fn test_function_call_with_real_repl_engine() {
        // Integration test: use real ReplEngine like TUI-REPL does
        // This tests the exact same path as typing in the TUI-REPL
        use nostos_repl::{ReplEngine, ReplConfig};

        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.load_stdlib().expect("Failed to load stdlib");

        // Define a type and function in the REPL
        engine.eval("type Person = { name: String }").expect("Failed to define Person type");
        engine.eval("greet(p: Person) -> String = \"Hello \" ++ p.name").expect("Failed to define greet function");

        // Verify the function signature is available
        let sig = engine.get_function_signature("greet");
        println!("greet signature: {:?}", sig);
        assert!(sig.is_some(), "greet should have a signature");
        assert!(sig.as_ref().unwrap().contains("String"), "greet should return String");

        // Now test autocomplete
        let source = ReplEngineSource { engine: &engine };
        let ac = Autocomplete::new();

        let line = "greet(Person(\"petter\")).";
        let ctx = ac.parse_context(line, line.len());
        println!("Context: {:?}", ctx);

        let items = ac.get_completions(&ctx, &source);
        println!("Completions: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        assert!(items.iter().any(|i| i.text == "toUpper"),
            "Should suggest String method 'toUpper' for greet() return type");
        assert!(items.iter().any(|i| i.text == "length"),
            "Should suggest String method 'length' for greet() return type");
    }

    #[test]
    fn test_string_literal_method_chain() {
        // "aa".length() should infer that "aa" is a String
        // and length() returns Int
        let source = MockSource::new();
        let ac = Autocomplete::new();

        // First test: "aa".length(). should give Int methods
        let line = "\"aa\".length().";
        let ctx = ac.parse_context(line, line.len());
        println!("Context for string.length(): {:?}", ctx);

        let items = ac.get_completions(&ctx, &source);
        println!("Completions for \"aa\".length().: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());
        // Int doesn't have many builtin methods, but we should get something or at least not crash
    }

    #[test]
    fn test_show_builtin_returns_string() {
        // show() is a builtin that returns String for any type
        // "aa".length().show(). should give String methods
        use nostos_repl::{ReplEngine, ReplConfig};

        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.load_stdlib().expect("Failed to load stdlib");

        // Verify show signature
        let show_sig = engine.get_function_signature("show");
        println!("show signature: {:?}", show_sig);
        assert!(show_sig.is_some(), "show should have a signature");
        assert!(show_sig.as_ref().unwrap().contains("String"), "show should return String");

        let source = ReplEngineSource { engine: &engine };
        let ac = Autocomplete::new();

        let line = "\"aa\".length().show().";
        let ctx = ac.parse_context(line, line.len());
        println!("Context for show(): {:?}", ctx);

        let items = ac.get_completions(&ctx, &source);
        println!("Completions for \"aa\".length().show().: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        assert!(items.iter().any(|i| i.text == "toUpper"),
            "Should suggest String method 'toUpper' after show()");
    }

    #[test]
    fn test_ufcs_user_function_autocomplete() {
        // p.greet() should work via UFCS and autocomplete should show String methods
        use nostos_repl::{ReplEngine, ReplConfig};

        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.load_stdlib().expect("Failed to load stdlib");

        // Define a type and function
        engine.eval("type Person = { name: String }").expect("Failed to define Person type");
        engine.eval("greet(p: Person) -> String = \"Hello \" ++ p.name").expect("Failed to define greet");

        // Create a variable of type Person
        engine.eval("p = Person(\"Alice\")").expect("Failed to create p");

        let source = ReplEngineSource { engine: &engine };
        let ac = Autocomplete::new();

        // Test: p.greet(). should show String methods
        let line = "p.greet().";
        let ctx = ac.parse_context(line, line.len());
        println!("Context for p.greet(): {:?}", ctx);

        let items = ac.get_completions(&ctx, &source);
        println!("Completions for p.greet().: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        assert!(items.iter().any(|i| i.text == "toUpper"),
            "Should suggest String method 'toUpper' for p.greet() UFCS call");
    }

    #[test]
    fn test_nested_generic_function_calls() {
        // Test nested generic calls like show(show(123)).
        use nostos_repl::{ReplEngine, ReplConfig};

        let mut engine = ReplEngine::new(ReplConfig::default());
        engine.load_stdlib().expect("Failed to load stdlib");

        let source = ReplEngineSource { engine: &engine };
        let ac = Autocomplete::new();

        // show(123). - show returns String
        let line = "show(123).";
        let ctx = ac.parse_context(line, line.len());
        let items = ac.get_completions(&ctx, &source);
        println!("show(123).: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());
        assert!(items.iter().any(|i| i.text == "toUpper"),
            "show(123). should suggest String methods");

        // show(show(123)). - nested show, still returns String
        let line2 = "show(show(123)).";
        let ctx2 = ac.parse_context(line2, line2.len());
        let items2 = ac.get_completions(&ctx2, &source);
        println!("show(show(123)).: {:?}", items2.iter().map(|i| &i.text).collect::<Vec<_>>());
        assert!(items2.iter().any(|i| i.text == "toUpper"),
            "show(show(123)). should suggest String methods");

        // hash(show(123)). - hash returns Int
        let line3 = "hash(show(123)).";
        let ctx3 = ac.parse_context(line3, line3.len());
        let items3 = ac.get_completions(&ctx3, &source);
        println!("hash(show(123)).: {:?}", items3.iter().map(|i| &i.text).collect::<Vec<_>>());
        // Int doesn't have many methods, but show should work
        assert!(items3.iter().any(|i| i.text == "show"),
            "hash(show(123)). should suggest show for Int");

        // copy("hello"). - copy returns same type (String)
        let line4 = "copy(\"hello\").";
        let ctx4 = ac.parse_context(line4, line4.len());
        let items4 = ac.get_completions(&ctx4, &source);
        println!("copy(\"hello\").: {:?}", items4.iter().map(|i| &i.text).collect::<Vec<_>>());
        assert!(items4.iter().any(|i| i.text == "toUpper"),
            "copy(\"hello\"). should suggest String methods");
    }

    #[test]
    fn test_builtin_function_return_type_autocomplete() {
        // Test that builtin functions like range(1,10) show correct return type
        let mut source = MockSource::new();
        // Register range with its signature from BUILTINS
        source.function_signatures.insert("range".to_string(), "Int -> Int -> [Int]".to_string());

        let ac = Autocomplete::new();

        // range(1,10). should suggest List methods
        let line = "range(1,10).";
        let ctx = ac.parse_context(line, line.len());
        println!("Context: {:?}", ctx);

        let items = ac.get_completions(&ctx, &source);
        println!("range(1,10).: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        assert!(!items.is_empty(), "range(1,10). should have completions");
        assert!(items.iter().any(|i| i.text == "map"),
            "range(1,10). should suggest map for List");
        assert!(items.iter().any(|i| i.text == "filter"),
            "range(1,10). should suggest filter for List");
    }

    #[test]
    fn test_variable_with_numeric_type_autocomplete() {
        // Test that a variable with a numeric type (Int32) gets autocomplete suggestions
        let source = MockSource::new()
            .with_variables(&["a"])
            .with_variable_type("a", "Int32");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "a".to_string(),
            prefix: "as".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("Items for a.as: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have type conversion methods
        assert!(!items.is_empty(), "a.as should have completions when a is Int32");
        assert!(items.iter().any(|i| i.text == "asInt64"),
            "Int32 variable should have asInt64 method");
        assert!(items.iter().any(|i| i.text == "asFloat"),
            "Int32 variable should have asFloat method");
    }

    #[test]
    fn test_is_numeric_type() {
        // Test the is_numeric_type helper function
        assert!(Autocomplete::is_numeric_type("Int"));
        assert!(Autocomplete::is_numeric_type("Int8"));
        assert!(Autocomplete::is_numeric_type("Int16"));
        assert!(Autocomplete::is_numeric_type("Int32"));
        assert!(Autocomplete::is_numeric_type("Int64"));
        assert!(Autocomplete::is_numeric_type("UInt8"));
        assert!(Autocomplete::is_numeric_type("UInt16"));
        assert!(Autocomplete::is_numeric_type("UInt32"));
        assert!(Autocomplete::is_numeric_type("UInt64"));
        assert!(Autocomplete::is_numeric_type("Float"));
        assert!(Autocomplete::is_numeric_type("Float32"));
        assert!(Autocomplete::is_numeric_type("Float64"));
        assert!(Autocomplete::is_numeric_type("BigInt"));

        // Non-numeric types
        assert!(!Autocomplete::is_numeric_type("String"));
        assert!(!Autocomplete::is_numeric_type("Bool"));
        assert!(!Autocomplete::is_numeric_type("List"));
        assert!(!Autocomplete::is_numeric_type("Map"));
    }

    #[test]
    fn test_option_type_autocomplete() {
        // Test that Option type gets autocomplete suggestions
        let source = MockSource::new()
            .with_variables(&["maybe_value"])
            .with_variable_type("maybe_value", "Option Int");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "maybe_value".to_string(),
            prefix: "".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("Option completions: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have Option-specific methods
        assert!(!items.is_empty(), "Option should have completions");
        assert!(items.iter().any(|i| i.text == "isSome"), "Option should have isSome method");
        assert!(items.iter().any(|i| i.text == "isNone"), "Option should have isNone method");
        assert!(items.iter().any(|i| i.text == "unwrap"), "Option should have unwrap method");
        assert!(items.iter().any(|i| i.text == "unwrapOr"), "Option should have unwrapOr method");
        assert!(items.iter().any(|i| i.text == "map"), "Option should have map method");
    }

    #[test]
    fn test_result_type_autocomplete() {
        // Test that Result type gets autocomplete suggestions
        let source = MockSource::new()
            .with_variables(&["result"])
            .with_variable_type("result", "Result Int String");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "result".to_string(),
            prefix: "".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("Result completions: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have Result-specific methods
        assert!(!items.is_empty(), "Result should have completions");
        assert!(items.iter().any(|i| i.text == "isOk"), "Result should have isOk method");
        assert!(items.iter().any(|i| i.text == "isErr"), "Result should have isErr method");
        assert!(items.iter().any(|i| i.text == "unwrap"), "Result should have unwrap method");
        assert!(items.iter().any(|i| i.text == "map"), "Result should have map method");
        assert!(items.iter().any(|i| i.text == "mapErr"), "Result should have mapErr method");
    }

    #[test]
    fn test_option_method_chain_type() {
        // Test that Option.isSome() returns Bool for further chaining
        // Test that the method return type is tracked correctly
        let ret = Autocomplete::get_method_return_type("Option Int", "isSome");
        assert_eq!(ret, Some("Bool".to_string()), "isSome should return Bool");

        let ret = Autocomplete::get_method_return_type("Option Int", "map");
        assert_eq!(ret, Some("Option Int".to_string()), "map should return Option");
    }

    #[test]
    fn test_result_method_chain_type() {
        // Test that Result method return types work
        let ret = Autocomplete::get_method_return_type("Result Int String", "isOk");
        assert_eq!(ret, Some("Bool".to_string()), "isOk should return Bool");

        let ret = Autocomplete::get_method_return_type("Result Int String", "map");
        assert_eq!(ret, Some("Result Int String".to_string()), "map should return Result");

        let ret = Autocomplete::get_method_return_type("Result Int String", "mapErr");
        assert_eq!(ret, Some("Result Int String".to_string()), "mapErr should return Result");
    }

    #[test]
    fn test_option_with_bracket_syntax() {
        // Test Option[Int] syntax (alternative to Option Int)
        let source = MockSource::new()
            .with_variables(&["opt"])
            .with_variable_type("opt", "Option[Int]");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "opt".to_string(),
            prefix: "is".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("Option[Int] completions with 'is' prefix: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have isSome and isNone with bracket syntax
        assert!(items.iter().any(|i| i.text == "isSome"), "Option[Int] should have isSome");
        assert!(items.iter().any(|i| i.text == "isNone"), "Option[Int] should have isNone");
    }

    #[test]
    fn test_nested_generic_type_autocomplete() {
        // Test List[Option Int] - nested generics
        let source = MockSource::new()
            .with_variables(&["list_opts"])
            .with_variable_type("list_opts", "List[Option Int]");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "list_opts".to_string(),
            prefix: "".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("List[Option Int] completions: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have List methods (map, filter, etc.)
        assert!(items.iter().any(|i| i.text == "map"), "List[Option Int] should have map");
        assert!(items.iter().any(|i| i.text == "filter"), "List[Option Int] should have filter");
        assert!(items.iter().any(|i| i.text == "head"), "List[Option Int] should have head");
    }

    #[test]
    fn test_map_with_generic_type() {
        // Test Map[String, Int] gets correct methods
        let source = MockSource::new()
            .with_variables(&["ages"])
            .with_variable_type("ages", "Map[String, Int]");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "ages".to_string(),
            prefix: "".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("Map[String, Int] completions: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have Map methods
        assert!(items.iter().any(|i| i.text == "get"), "Map should have get");
        assert!(items.iter().any(|i| i.text == "insert"), "Map should have insert");
        assert!(items.iter().any(|i| i.text == "keys"), "Map should have keys");
        assert!(items.iter().any(|i| i.text == "values"), "Map should have values");
        assert!(items.iter().any(|i| i.text == "size"), "Map should have size");
    }

    #[test]
    fn test_set_with_generic_type() {
        // Test Set[String] gets correct methods
        let source = MockSource::new()
            .with_variables(&["names"])
            .with_variable_type("names", "Set[String]");

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "names".to_string(),
            prefix: "".to_string(),
        };
        let items = ac.get_completions(&ctx, &source);

        println!("Set[String] completions: {:?}", items.iter().map(|i| &i.text).collect::<Vec<_>>());

        // Should have Set methods
        assert!(items.iter().any(|i| i.text == "contains"), "Set should have contains");
        assert!(items.iter().any(|i| i.text == "insert"), "Set should have insert");
        assert!(items.iter().any(|i| i.text == "union"), "Set should have union");
        assert!(items.iter().any(|i| i.text == "intersection"), "Set should have intersection");
        assert!(items.iter().any(|i| i.text == "toList"), "Set should have toList");
    }

}
