//! Autocomplete engine for REPL
//!
//! Provides completion candidates based on:
//! - Function names
//! - Type names
//! - Module names
//! - Variable names
//! - Type fields (after `.` on a type instance)
//! - Module members (after `.` on a module name)

use std::collections::{HashMap, HashSet};

/// A completion candidate
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionItem {
    /// The text to insert (just the identifier, not the full path)
    pub text: String,
    /// Display label (may include type info or signature)
    pub label: String,
    /// Kind of completion
    pub kind: CompletionKind,
}

/// Kind of completion item
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompletionKind {
    Function,
    Type,
    Module,
    Field,
    Variable,
    Constructor,
}

impl CompletionKind {
    /// Short prefix for display
    pub fn prefix(&self) -> &'static str {
        match self {
            CompletionKind::Function => "fn",
            CompletionKind::Type => "type",
            CompletionKind::Module => "mod",
            CompletionKind::Field => "field",
            CompletionKind::Variable => "var",
            CompletionKind::Constructor => "ctor",
        }
    }

    /// Color for display (RGB values)
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            CompletionKind::Function => (100, 200, 255),   // Light blue
            CompletionKind::Type => (255, 200, 100),       // Orange
            CompletionKind::Module => (200, 200, 200),     // Gray
            CompletionKind::Field => (150, 255, 150),      // Light green
            CompletionKind::Variable => (255, 150, 255),   // Pink
            CompletionKind::Constructor => (255, 255, 100), // Yellow
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

    /// Get constructors for a variant type
    fn get_type_constructors(&self, type_name: &str) -> Vec<String>;
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
}

/// Autocomplete engine
pub struct Autocomplete {
    /// Cached module names (extracted from function/type prefixes)
    modules: HashSet<String>,
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

        // Find the start of the current identifier/expression
        let (expr_start, expr) = self.extract_expression(&before_cursor);

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

    /// Extract the expression being completed from text before cursor
    fn extract_expression<'a>(&self, text: &'a str) -> (usize, &'a str) {
        // Walk backwards to find expression start
        // Valid expression chars: alphanumeric, underscore, dot
        let chars: Vec<char> = text.chars().collect();
        let mut start = chars.len();

        for i in (0..chars.len()).rev() {
            let c = chars[i];
            if c.is_alphanumeric() || c == '_' || c == '.' {
                start = i;
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
        match context {
            CompletionContext::Identifier { prefix } => {
                self.complete_identifier(prefix, source)
            }
            CompletionContext::ModuleMember { module, prefix } => {
                self.complete_module_member(module, prefix, source)
            }
            CompletionContext::FieldAccess { receiver, prefix } => {
                self.complete_field_access(receiver, prefix, source)
            }
        }
    }

    /// Complete a general identifier
    fn complete_identifier(
        &self,
        prefix: &str,
        source: &dyn CompletionSource,
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        let prefix_lower = prefix.to_lowercase();

        // Add matching modules
        for module in &self.modules {
            if module.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: module.clone(),
                    label: module.clone(),
                    kind: CompletionKind::Module,
                });
            }
        }

        // Add matching types
        for type_name in &self.types {
            // Only match the base name (after last dot)
            let base = type_name.rsplit('.').next().unwrap_or(type_name);
            if base.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: base.to_string(),
                    label: type_name.clone(),
                    kind: CompletionKind::Type,
                });
            }
        }

        // Add matching functions (only top-level, not module-qualified)
        for func_name in self.functions.keys() {
            if !func_name.contains('.') {
                if func_name.to_lowercase().starts_with(&prefix_lower) {
                    items.push(CompletionItem {
                        text: func_name.clone(),
                        label: func_name.clone(),
                        kind: CompletionKind::Function,
                    });
                }
            }
        }

        // Add matching variables
        for var_name in source.get_variables() {
            if var_name.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: var_name.clone(),
                    label: var_name,
                    kind: CompletionKind::Variable,
                });
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
        _source: &dyn CompletionSource,
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
                    items.push(CompletionItem {
                        text: member.to_string(),
                        label: func_name.clone(),
                        kind: CompletionKind::Function,
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
                        });
                    }
                }
            }
        }

        items.sort_by(|a, b| a.text.cmp(&b.text));
        items
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
        // For now, if receiver looks like a type name (uppercase), get its fields
        // This is a simplification - full type inference would be better

        let type_name = if self.is_upper_ident(receiver) {
            receiver.to_string()
        } else {
            // Could try to look up variable type here
            receiver.to_string()
        };

        // Get fields
        for field in source.get_type_fields(&type_name) {
            if field.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: field.clone(),
                    label: field,
                    kind: CompletionKind::Field,
                });
            }
        }

        // Get constructors (for variant types)
        for ctor in source.get_type_constructors(&type_name) {
            if ctor.to_lowercase().starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    text: ctor.clone(),
                    label: ctor,
                    kind: CompletionKind::Constructor,
                });
            }
        }

        items.sort_by(|a, b| a.text.cmp(&b.text));
        items
    }

    /// Apply a completion: returns the text to insert and cursor adjustment
    pub fn apply_completion(
        &self,
        context: &CompletionContext,
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
    }

    impl MockSource {
        fn new() -> Self {
            Self {
                functions: Vec::new(),
                types: Vec::new(),
                variables: Vec::new(),
                type_fields: HashMap::new(),
                type_constructors: HashMap::new(),
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

        fn get_type_constructors(&self, type_name: &str) -> Vec<String> {
            self.type_constructors.get(type_name).cloned().unwrap_or_default()
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

        assert_eq!(items.len(), 3);
        assert!(items.iter().all(|i| i.kind == CompletionKind::Field));
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
    fn test_complete_constructors() {
        let source = MockSource::new()
            .with_type_constructors("Option", &["Some", "None"]);

        let ac = Autocomplete::new();

        let ctx = CompletionContext::FieldAccess {
            receiver: "Option".to_string(),
            prefix: "".to_string()
        };
        let items = ac.get_completions(&ctx, &source);

        assert_eq!(items.len(), 2);
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
}
