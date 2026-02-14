use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle, Style};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2, Rect};
use nostos_syntax::lexer::{Token, lex};
use nostos_syntax::{parse, parse_errors_to_source_errors, offset_to_line_col};
use std::collections::VecDeque;
use std::rc::Rc;
use std::cell::RefCell;
use std::time::Instant;
use nostos_repl::ReplEngine;
use nostos_repl::inference;

use crate::autocomplete::{Autocomplete, CompletionContext, CompletionItem, CompletionSource, parse_imports, extract_module_from_editor_name};
use crate::tui::copy_to_system_clipboard;

/// Compile status for the editor
#[derive(Clone, Debug)]
pub enum CompileStatus {
    /// Not yet checked
    Unknown,
    /// Code parses and compiles OK
    Ok,
    /// Just compiled successfully (shows "Compiled ✓", reverts to Ok on next edit)
    Compiled,
    /// Parse error (syntax)
    ParseError(String),
    /// Compile error (type error, etc.)
    CompileError(String),
}

/// Wrapper to implement CompletionSource for ReplEngine with local variable inference
struct EditorCompletionSource<'a> {
    engine: &'a ReplEngine,
    /// Current buffer content for local variable inference
    buffer_content: &'a str,
    /// Current module name for qualified function lookups
    module_name: Option<&'a str>,
}

// Type inference functions are in the shared nostos_repl::inference module

impl<'a> EditorCompletionSource<'a> {
    /// Extract local variable names from the buffer
    fn extract_local_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();

        for line in self.buffer_content.lines() {
            let trimmed = line.trim();

            // Skip comments
            if trimmed.starts_with('#') {
                continue;
            }

            // Function definition: foo(param1, param2: Type) = ...
            // Extract parameters as local variables
            if let Some(paren_start) = trimmed.find('(') {
                if let Some(paren_end) = trimmed[paren_start..].find(')') {
                    let after_paren = trimmed[paren_start + paren_end + 1..].trim_start();
                    // Check if this is a function definition (has = after params)
                    if after_paren.starts_with('=') || after_paren.starts_with("->") {
                        let params = &trimmed[paren_start + 1..paren_start + paren_end];
                        for param in params.split(',') {
                            let param = param.trim();
                            // Handle "name: Type" or just "name"
                            let name = param.split(':').next().unwrap_or(param).trim();
                            if !name.is_empty() && name != "_" && name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                                vars.push(name.to_string());
                            }
                        }
                    }
                }
            }

            // Tuple destructuring: (a, b, ...) = expr
            if trimmed.starts_with('(') {
                if let Some(close_paren) = trimmed.find(')') {
                    let after = trimmed[close_paren + 1..].trim_start();
                    if after.starts_with('=') && !after.starts_with("==") {
                        let pattern = &trimmed[1..close_paren];
                        for name in pattern.split(',') {
                            let name = name.trim();
                            if !name.is_empty() && name != "_" && name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                                vars.push(name.to_string());
                            }
                        }
                    }
                }
                continue;
            }

            // Simple binding: varname = expr (not ==, not function def)
            if let Some(eq_pos) = trimmed.find('=') {
                let lhs = trimmed[..eq_pos].trim();
                let after_eq = &trimmed[eq_pos + 1..];

                // Skip == and function definitions (contains parens before =)
                if after_eq.starts_with('=') || lhs.contains('(') {
                    continue;
                }

                // Handle "var name" or just "name"
                let name = lhs.strip_prefix("var ").unwrap_or(lhs).trim();

                // Must be valid identifier (lowercase start, no spaces)
                if !name.is_empty() && !name.contains(' ') && name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                    vars.push(name.to_string());
                }
            }
        }

        // Remove duplicates
        vars.sort();
        vars.dedup();
        vars
    }

    /// Build local variable type map from buffer for expression type inference
    fn build_local_vars(&self) -> std::collections::HashMap<String, String> {
        let mut local_vars = std::collections::HashMap::new();
        // Add REPL variables
        for var_name in self.engine.get_variables() {
            if let Some(var_type) = self.engine.get_variable_type(&var_name) {
                local_vars.insert(var_name, var_type);
            }
        }
        // Add local variables inferred from buffer using shared inference module
        let buffer_bindings = inference::extract_local_bindings(
            self.buffer_content,
            self.buffer_content.lines().count(),
            Some(self.engine),
        );
        for (name, ty) in buffer_bindings {
            if !local_vars.contains_key(&name) {
                local_vars.insert(name, ty);
            }
        }
        local_vars
    }
}

impl<'a> CompletionSource for EditorCompletionSource<'a> {
    fn get_functions(&self) -> Vec<String> {
        self.engine.get_functions()
    }

    fn get_types(&self) -> Vec<String> {
        self.engine.get_types()
    }

    fn get_variables(&self) -> Vec<String> {
        // Combine REPL variables with local variables from buffer
        let mut vars = self.engine.get_variables();
        vars.extend(self.extract_local_variables());
        vars.sort();
        vars.dedup();
        vars
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
        // First try REPL variables
        if let Some(t) = self.engine.get_variable_type(var_name) {
            return Some(t);
        }

        // Then try local variable inference from buffer using shared inference module
        let bindings = inference::extract_local_bindings(
            self.buffer_content,
            self.buffer_content.lines().count(),
            Some(self.engine),
        );
        bindings.get(var_name).cloned()
    }

    fn get_ufcs_methods_for_type(&self, type_name: &str) -> Vec<(String, String, Option<String>)> {
        self.engine.get_ufcs_methods_for_type(type_name)
    }

    fn get_builtin_methods_for_type(&self, type_name: &str) -> Vec<(String, String, String)> {
        nostos_repl::ReplEngine::get_builtin_methods_for_type(type_name)
            .into_iter()
            .map(|(n, s, d)| (n.to_string(), s.to_string(), d.to_string()))
            .collect()
    }

    fn infer_expression_type(&self, expr: &str) -> Option<String> {
        let local_vars = self.build_local_vars();
        self.engine.infer_expression_type(expr, &local_vars)
    }

    fn get_trait_methods_for_type(&self, type_name: &str) -> Vec<(String, String, Option<String>)> {
        self.engine.get_trait_methods_for_type(type_name)
    }
}

/// Maximum items to show in autocomplete popup
const AC_MAX_VISIBLE: usize = 10;

/// Signature help state (shows function parameters when typing '(')
struct SignatureHelpState {
    /// Whether signature help popup is visible
    active: bool,
    /// Function name being called
    function_name: String,
    /// Parameter info: (name, type, is_optional, default_preview)
    params: Vec<(String, String, bool, Option<String>)>,
    /// Full signature string (for builtins that don't have param details)
    signature: Option<String>,
    /// Current parameter index (based on comma count)
    current_param: usize,
}

impl SignatureHelpState {
    fn new() -> Self {
        Self {
            active: false,
            function_name: String::new(),
            params: Vec::new(),
            signature: None,
            current_param: 0,
        }
    }

    fn reset(&mut self) {
        self.active = false;
        self.function_name.clear();
        self.params.clear();
        self.signature = None;
        self.current_param = 0;
    }
}

/// Autocomplete state for the editor
struct AutocompleteState {
    /// Whether autocomplete popup is visible
    active: bool,
    /// Current completion candidates
    candidates: Vec<CompletionItem>,
    /// Selected index in candidates
    selected: usize,
    /// Scroll offset for pagination
    scroll_offset: usize,
    /// The context (prefix info)
    context: Option<CompletionContext>,
}

impl AutocompleteState {
    fn new() -> Self {
        Self {
            active: false,
            candidates: Vec::new(),
            selected: 0,
            scroll_offset: 0,
            context: None,
        }
    }

    fn reset(&mut self) {
        self.active = false;
        self.candidates.clear();
        self.selected = 0;
        self.scroll_offset = 0;
        self.context = None;
    }

    /// Move selection up, with pagination
    fn select_prev(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        if self.selected == 0 {
            self.selected = self.candidates.len() - 1;
            // Scroll to show selection
            if self.selected >= AC_MAX_VISIBLE {
                self.scroll_offset = self.selected - AC_MAX_VISIBLE + 1;
            }
        } else {
            self.selected -= 1;
            // Scroll up if selection goes above visible area
            if self.selected < self.scroll_offset {
                self.scroll_offset = self.selected;
            }
        }
    }

    /// Move selection down, with pagination
    fn select_next(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        self.selected = (self.selected + 1) % self.candidates.len();
        if self.selected == 0 {
            // Wrapped around to top
            self.scroll_offset = 0;
        } else if self.selected >= self.scroll_offset + AC_MAX_VISIBLE {
            // Scroll down to keep selection visible
            self.scroll_offset = self.selected - AC_MAX_VISIBLE + 1;
        }
    }

    /// Page down (move by AC_MAX_VISIBLE items)
    fn page_down(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        let new_selected = (self.selected + AC_MAX_VISIBLE).min(self.candidates.len() - 1);
        self.selected = new_selected;
        // Adjust scroll to show selection
        if self.selected >= self.scroll_offset + AC_MAX_VISIBLE {
            self.scroll_offset = self.selected - AC_MAX_VISIBLE + 1;
        }
    }

    /// Page up (move by AC_MAX_VISIBLE items)
    fn page_up(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        self.selected = self.selected.saturating_sub(AC_MAX_VISIBLE);
        // Adjust scroll to show selection
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
    }
}

pub struct CodeEditor {
    content: Vec<String>,
    cursor: (usize, usize), // col, row (0-indexed)
    scroll_offset: (usize, usize), // horizontal, vertical scroll offset
    last_size: Vec2,
    engine: Option<Rc<RefCell<ReplEngine>>>, // For autocomplete
    saved_content: String, // Content when last saved (for dirty checking)
    /// Autocomplete engine
    autocomplete: Autocomplete,
    /// Autocomplete state
    ac_state: AutocompleteState,
    /// Signature help state
    sig_help: SignatureHelpState,
    /// Module name being edited (e.g., "Math" when editing "Math.add")
    module_name: Option<String>,
    /// Function name being edited (the simple name, e.g., "main" or "add")
    function_name: Option<String>,
    /// Whether editor is in read-only mode (for eval'd functions)
    read_only: bool,
    /// Current compile status
    compile_status: CompileStatus,
    /// Line number when we last checked compile status
    last_check_line: usize,
    /// Content hash when we last checked (to detect changes)
    last_check_content: String,
    /// Time of last edit (for debounced full compile)
    last_edit_time: Option<Instant>,
    /// Whether we need a full compile check (after debounce)
    needs_full_compile: bool,
    /// Whether this is a file: mode editor (standalone file, not part of a module)
    is_file_mode: bool,
    /// Selection anchor point (col, row). When Some, selection spans from anchor to cursor.
    selection_anchor: Option<(usize, usize)>,
    /// Undo stack (snapshots of previous states)
    undo_stack: VecDeque<EditorSnapshot>,
    /// Redo stack (snapshots of undone states)
    redo_stack: VecDeque<EditorSnapshot>,
}

const MAX_UNDO_DEPTH: usize = 100;

#[derive(Clone)]
struct EditorSnapshot {
    content: Vec<String>,
    cursor: (usize, usize),
    selection_anchor: Option<(usize, usize)>,
}

impl CodeEditor {
    pub fn new(text: String) -> Self {
        let content: Vec<String> = text.lines().map(String::from).collect();
        let content = if content.is_empty() { vec![String::new()] } else { content };
        let saved_content = content.join("\n");
        Self {
            content,
            cursor: (0, 0),
            scroll_offset: (0, 0),
            last_size: Vec2::zero(),
            engine: None,
            saved_content,
            autocomplete: Autocomplete::new(),
            ac_state: AutocompleteState::new(),
            sig_help: SignatureHelpState::new(),
            module_name: None,
            function_name: None,
            read_only: false,
            compile_status: CompileStatus::Unknown,
            last_check_line: 0,
            last_check_content: String::new(),
            last_edit_time: None,
            needs_full_compile: false,
            is_file_mode: false,
            selection_anchor: None,
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
        }
    }

    /// Set file mode (for standalone file editors)
    pub fn with_file_mode(mut self, is_file: bool) -> Self {
        self.is_file_mode = is_file;
        self
    }

    /// Perform a quick parse check on the current content
    fn check_parse(&mut self) {
        let content = self.content.join("\n");

        // Don't recheck if content hasn't changed
        if content == self.last_check_content {
            return;
        }

        self.last_check_content = content.clone();

        // Try to parse the content
        let (module_opt, errors) = parse(&content);

        if !errors.is_empty() {
            // Convert raw parse errors to SourceErrors
            let source_errors = parse_errors_to_source_errors(&errors);
            if let Some(first_error) = source_errors.first() {
                let (line, _col) = offset_to_line_col(&content, first_error.span.start);
                let error_msg = format!("line {}: {}", line, first_error.message);
                self.compile_status = CompileStatus::ParseError(error_msg);
            } else {
                self.compile_status = CompileStatus::ParseError("Parse error".to_string());
            }
        } else if module_opt.is_some() {
            // Parse OK - run full compile check immediately
            self.needs_full_compile = true;
            self.check_compile();
        } else {
            self.compile_status = CompileStatus::ParseError("Unknown parse error".to_string());
        }
    }

    /// Perform a full compile check (called after debounce or on save)
    fn check_compile(&mut self) {
        if !self.needs_full_compile {
            return;
        }

        let engine = match &self.engine {
            Some(e) => e,
            None => return,
        };

        let content = self.content.join("\n");

        // Try to compile via the engine
        let eng = engine.borrow();

        // Determine the module name for compile context
        // For file: mode, use empty string to avoid prepending module context
        // If module_name is not set but we have a function_name, try to look up the module
        let module_name = if self.is_file_mode {
            String::new() // Standalone file - no module context
        } else {
            match &self.module_name {
                Some(m) => m.clone(),
                None => {
                    // Try to find the module for this function
                    if let Some(ref fn_name) = self.function_name {
                        eng.get_function_module(fn_name).unwrap_or_default()
                    } else {
                        String::new()
                    }
                }
            }
        };

        let result = eng.check_module_compiles(&module_name, &content);
        match result {
            Ok(()) => {
                self.compile_status = CompileStatus::Ok;
            }
            Err(error) => {
                self.compile_status = CompileStatus::CompileError(error.clone());
            }
        }

        self.needs_full_compile = false;
    }

    /// Called when cursor moves to a different line
    fn on_line_change(&mut self, old_line: usize, new_line: usize) {
        if old_line != new_line {
            // Trigger parse check when leaving a line
            self.check_parse();
            self.last_check_line = new_line;
        }
    }

    /// Called on each edit to track timing for debounced compile
    fn on_edit(&mut self) {
        self.last_edit_time = Some(Instant::now());
        // Revert "Compiled ✓" back to plain "✓" on next edit
        if matches!(self.compile_status, CompileStatus::Compiled) {
            self.compile_status = CompileStatus::Ok;
        }
    }

    /// Set the function name being edited (e.g., "utils.bar" or "Math.add")
    /// This extracts the module context for autocomplete
    pub fn with_function_name(mut self, name: &str) -> Self {
        self.module_name = extract_module_from_editor_name(name);
        // Store the simple function name (part after last dot, or full name)
        self.function_name = Some(name.rsplit('.').next().unwrap_or(name).to_string());
        self
    }

    /// Set read-only mode (for viewing eval'd functions)
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Adjust scroll to keep cursor visible
    fn ensure_cursor_visible(&mut self) {
        let (cx, cy) = self.cursor;
        let (sx, sy) = self.scroll_offset;
        let width = self.last_size.x.saturating_sub(1); // Leave room for cursor at edge
        let height = self.last_size.y;

        if width == 0 || height == 0 {
            return;
        }

        // Horizontal scrolling
        if cx < sx {
            self.scroll_offset.0 = cx;
        } else if cx >= sx + width {
            self.scroll_offset.0 = cx.saturating_sub(width) + 1;
        }

        // Vertical scrolling
        if cy < sy {
            self.scroll_offset.1 = cy;
        } else if cy >= sy + height {
            self.scroll_offset.1 = cy.saturating_sub(height) + 1;
        }
    }

    pub fn with_engine(mut self, engine: Rc<RefCell<ReplEngine>>) -> Self {
        // Initialize autocomplete from engine
        {
            let eng = engine.borrow();
            let source = EditorCompletionSource {
                engine: &eng,
                buffer_content: "",
                module_name: self.module_name.as_deref(),
            };
            self.autocomplete.update_from_source(&source);
        }
        self.engine = Some(engine);
        self
    }

    pub fn get_content(&self) -> String {
        self.content.join("\n")
    }

    /// Check if the editor has unsaved changes
    pub fn is_dirty(&self) -> bool {
        self.get_content() != self.saved_content
    }

    /// Mark the current content as saved
    pub fn mark_saved(&mut self) {
        self.saved_content = self.get_content();
    }

    /// Whether there is an active text selection
    pub fn has_selection(&self) -> bool {
        self.selection_anchor.is_some()
    }

    /// Clear the current selection
    pub fn clear_selection(&mut self) {
        self.selection_anchor = None;
    }

    /// Start or continue a selection. Sets anchor if not already set.
    fn ensure_selection_anchor(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
    }

    /// Get ordered selection range: (start_col, start_row), (end_col, end_row)
    /// Start is always before end in document order.
    pub fn selection_range(&self) -> Option<((usize, usize), (usize, usize))> {
        let anchor = self.selection_anchor?;
        let cursor = self.cursor;
        // Order by row first, then column
        if anchor.1 < cursor.1 || (anchor.1 == cursor.1 && anchor.0 <= cursor.0) {
            Some((anchor, cursor))
        } else {
            Some((cursor, anchor))
        }
    }

    /// Get the selected text, if any
    pub fn get_selected_text(&self) -> Option<String> {
        let ((start_col, start_row), (end_col, end_row)) = self.selection_range()?;
        if start_row == end_row {
            // Single line selection
            let line = &self.content[start_row];
            let start_byte = Self::char_to_byte_idx(line, start_col);
            let end_byte = Self::char_to_byte_idx(line, end_col);
            Some(line[start_byte..end_byte].to_string())
        } else {
            // Multi-line selection
            let mut result = String::new();
            // First line: from start_col to end
            let first_line = &self.content[start_row];
            let start_byte = Self::char_to_byte_idx(first_line, start_col);
            result.push_str(&first_line[start_byte..]);
            // Middle lines: full lines
            for row in (start_row + 1)..end_row {
                result.push('\n');
                result.push_str(&self.content[row]);
            }
            // Last line: from start to end_col
            result.push('\n');
            let last_line = &self.content[end_row];
            let end_byte = Self::char_to_byte_idx(last_line, end_col);
            result.push_str(&last_line[..end_byte]);
            Some(result)
        }
    }

    /// Delete the selected text and place cursor at the start of the selection
    pub fn delete_selection(&mut self) {
        let ((start_col, start_row), (end_col, end_row)) = match self.selection_range() {
            Some(r) => r,
            None => return,
        };

        if start_row == end_row {
            // Single line: remove characters between start and end
            let line = &self.content[start_row];
            let start_byte = Self::char_to_byte_idx(line, start_col);
            let end_byte = Self::char_to_byte_idx(line, end_col);
            let mut new_line = line[..start_byte].to_string();
            new_line.push_str(&line[end_byte..]);
            self.content[start_row] = new_line;
        } else {
            // Multi-line: keep start of first line + end of last line, remove middle lines
            let first_line = &self.content[start_row];
            let last_line = &self.content[end_row];
            let start_byte = Self::char_to_byte_idx(first_line, start_col);
            let end_byte = Self::char_to_byte_idx(last_line, end_col);
            let mut new_line = first_line[..start_byte].to_string();
            new_line.push_str(&last_line[end_byte..]);
            // Remove lines from end_row down to start_row+1, then replace start_row
            for _ in (start_row + 1)..=end_row {
                self.content.remove(start_row + 1);
            }
            self.content[start_row] = new_line;
        }

        // Place cursor at start of deleted region
        self.cursor = (start_col, start_row);
        self.selection_anchor = None;
    }

    /// Insert text at cursor position (handles multi-line text for paste)
    pub fn insert_text(&mut self, text: &str) {
        // Normalize line endings (handle \r\n from clipboard)
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let lines: Vec<&str> = normalized.split('\n').collect();
        if lines.is_empty() {
            return;
        }

        if lines.len() == 1 {
            // Single line insert
            let line = &self.content[self.cursor.1];
            let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
            let mut new_line = line[..byte_idx].to_string();
            new_line.push_str(lines[0]);
            new_line.push_str(&line[byte_idx..]);
            self.content[self.cursor.1] = new_line;
            self.cursor.0 += lines[0].chars().count();
        } else {
            // Multi-line insert
            let current_line = &self.content[self.cursor.1];
            let byte_idx = Self::char_to_byte_idx(current_line, self.cursor.0);
            let before = current_line[..byte_idx].to_string();
            let after = current_line[byte_idx..].to_string();

            // First line: before + first paste line
            self.content[self.cursor.1] = format!("{}{}", before, lines[0]);

            // Middle lines
            for (i, line) in lines[1..lines.len()-1].iter().enumerate() {
                self.content.insert(self.cursor.1 + 1 + i, line.to_string());
            }

            // Last line: last paste line + after
            let last_paste_line = lines[lines.len() - 1];
            let new_last = format!("{}{}", last_paste_line, after);
            let insert_row = self.cursor.1 + lines.len() - 1;
            if insert_row < self.content.len() {
                self.content.insert(insert_row, new_last);
            } else {
                self.content.push(new_last);
            }

            // Position cursor at end of pasted text
            self.cursor.1 = self.cursor.1 + lines.len() - 1;
            self.cursor.0 = last_paste_line.chars().count();
        }
    }

    /// Save current state to undo stack (call before any edit operation)
    pub fn save_snapshot(&mut self) {
        let snapshot = EditorSnapshot {
            content: self.content.clone(),
            cursor: self.cursor,
            selection_anchor: self.selection_anchor,
        };
        self.undo_stack.push_back(snapshot);
        if self.undo_stack.len() > MAX_UNDO_DEPTH {
            self.undo_stack.pop_front();
        }
        // Any new edit clears the redo stack
        self.redo_stack.clear();
    }

    /// Undo the last edit operation
    fn undo(&mut self) {
        if let Some(snapshot) = self.undo_stack.pop_back() {
            // Save current state to redo stack
            self.redo_stack.push_back(EditorSnapshot {
                content: self.content.clone(),
                cursor: self.cursor,
                selection_anchor: self.selection_anchor,
            });
            self.content = snapshot.content;
            self.cursor = snapshot.cursor;
            self.selection_anchor = snapshot.selection_anchor;
        }
    }

    /// Redo the last undone operation
    fn redo(&mut self) {
        if let Some(snapshot) = self.redo_stack.pop_back() {
            // Save current state to undo stack (without clearing redo)
            self.undo_stack.push_back(EditorSnapshot {
                content: self.content.clone(),
                cursor: self.cursor,
                selection_anchor: self.selection_anchor,
            });
            self.content = snapshot.content;
            self.cursor = snapshot.cursor;
            self.selection_anchor = snapshot.selection_anchor;
        }
    }

    /// Set the compile error status directly (for Ctrl+E compile feedback)
    pub fn set_compile_error(&mut self, error: Option<String>) {
        match error {
            Some(msg) => self.compile_status = CompileStatus::CompileError(msg),
            None => self.compile_status = CompileStatus::Ok,
        }
    }

    /// Mark as successfully compiled (shows "Compiled ✓" until next edit)
    pub fn set_compiled_ok(&mut self) {
        self.compile_status = CompileStatus::Compiled;
    }

    /// Get the number of characters in a line
    fn line_char_count(&self, line_idx: usize) -> usize {
        self.content[line_idx].chars().count()
    }

    /// Convert character index to byte index for a line
    fn char_to_byte_idx(line: &str, char_idx: usize) -> usize {
        line.char_indices()
            .nth(char_idx)
            .map(|(i, _)| i)
            .unwrap_or(line.len())
    }

    fn fix_cursor_x(&mut self) {
        let line_char_len = self.line_char_count(self.cursor.1);
        if self.cursor.0 > line_char_len {
            self.cursor.0 = line_char_len;
        }
    }

    /// Extract line number from compile status error message
    /// Error messages are formatted as "line N: message"
    fn get_error_line(&self) -> Option<usize> {
        let error_msg = match &self.compile_status {
            CompileStatus::ParseError(msg) | CompileStatus::CompileError(msg) => msg,
            _ => return None,
        };

        // Parse "line N:" prefix
        if error_msg.starts_with("line ") {
            if let Some(colon_pos) = error_msg.find(':') {
                let num_str = &error_msg[5..colon_pos];
                return num_str.parse().ok();
            }
        }
        None
    }

    /// Jump to a specific line number (1-indexed)
    pub fn jump_to_line(&mut self, line: usize) {
        // Line numbers are 1-indexed, cursor row is 0-indexed
        let target_row = line.saturating_sub(1);
        if target_row < self.content.len() {
            self.cursor.1 = target_row;
            self.cursor.0 = 0; // Go to start of line
            self.ac_state.reset();
            self.ensure_cursor_visible();
        }
    }

    fn insert_char(&mut self, c: char) {
        let line = &mut self.content[self.cursor.1];
        let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
        if byte_idx >= line.len() {
            line.push(c);
        } else {
            line.insert(byte_idx, c);
        }
        self.cursor.0 += 1;

        // Handle signature help triggers
        match c {
            '(' => self.trigger_signature_help(),
            ',' => self.update_signature_help_param(),
            ')' => self.close_signature_help(),
            _ => {}
        }

        // Update autocomplete after each character
        self.update_autocomplete();
    }

    /// Trigger signature help when '(' is typed
    fn trigger_signature_help(&mut self) {
        let engine = match &self.engine {
            Some(e) => e,
            None => return,
        };

        let line = &self.content[self.cursor.1];
        // Get text before cursor (before the '(' we just inserted)
        let before_paren: String = line.chars().take(self.cursor.0.saturating_sub(1)).collect();

        // Extract the function name before the '('
        let func_name: String = before_paren
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
            .collect::<String>()
            .chars()
            .rev()
            .collect();

        if func_name.is_empty() {
            return;
        }

        let eng = engine.borrow();

        // Try to get function params
        if let Some(params) = eng.get_function_params(&func_name) {
            self.sig_help.active = true;
            self.sig_help.function_name = func_name;
            self.sig_help.params = params;
            self.sig_help.signature = None;
            self.sig_help.current_param = 0;
        } else if let Some(sig) = eng.get_function_signature(&func_name) {
            // Fall back to signature string for builtins
            self.sig_help.active = true;
            self.sig_help.function_name = func_name;
            self.sig_help.params.clear();
            self.sig_help.signature = Some(sig);
            self.sig_help.current_param = 0;
        }
    }

    /// Update current parameter position when ',' is typed
    fn update_signature_help_param(&mut self) {
        if self.sig_help.active {
            self.sig_help.current_param += 1;
        }
    }

    /// Close signature help when ')' is typed
    fn close_signature_help(&mut self) {
        self.sig_help.reset();
    }

    /// Find matching bracket position for bracket at or before cursor
    /// Returns (row, col) of matching bracket, or None if no match
    fn find_matching_bracket(&self) -> Option<(usize, usize)> {
        let line = &self.content[self.cursor.1];
        let chars: Vec<char> = line.chars().collect();

        // Check char at cursor position
        let char_at_cursor = chars.get(self.cursor.0).copied();

        // Determine bracket type and search direction
        let (open, close, search_forward) = match char_at_cursor {
            Some('(') => ('(', ')', true),
            Some('[') => ('[', ']', true),
            Some('{') => ('{', '}', true),
            Some(')') => ('(', ')', false),
            Some(']') => ('[', ']', false),
            Some('}') => ('{', '}', false),
            _ => return None,
        };

        if search_forward {
            self.find_closing_bracket(self.cursor.1, self.cursor.0, open, close)
        } else {
            self.find_opening_bracket(self.cursor.1, self.cursor.0, open, close)
        }
    }

    /// Search forward for closing bracket, starting after (row, col)
    fn find_closing_bracket(&self, start_row: usize, start_col: usize, open: char, close: char) -> Option<(usize, usize)> {
        let mut depth = 1;
        let mut row = start_row;
        let mut col = start_col + 1; // Start after the opening bracket

        while row < self.content.len() {
            let line: Vec<char> = self.content[row].chars().collect();

            while col < line.len() {
                let c = line[col];
                if c == open {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                    if depth == 0 {
                        return Some((row, col));
                    }
                }
                col += 1;
            }

            row += 1;
            col = 0;
        }
        None
    }

    /// Search backward for opening bracket, starting before (row, col)
    fn find_opening_bracket(&self, start_row: usize, start_col: usize, open: char, close: char) -> Option<(usize, usize)> {
        let mut depth = 1;
        let mut row = start_row;
        let mut col = start_col;

        loop {
            if col > 0 {
                col -= 1;
                let line: Vec<char> = self.content[row].chars().collect();
                let c = line[col];

                if c == close {
                    depth += 1;
                } else if c == open {
                    depth -= 1;
                    if depth == 0 {
                        return Some((row, col));
                    }
                }
            } else if row > 0 {
                row -= 1;
                col = self.content[row].chars().count();
            } else {
                break;
            }
        }
        None
    }

    fn insert_newline(&mut self) {
        let line = &mut self.content[self.cursor.1];
        let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
        let new_line_content = if byte_idx < line.len() {
            line.split_off(byte_idx)
        } else {
            String::new()
        };
        self.content.insert(self.cursor.1 + 1, new_line_content);
        self.cursor.1 += 1;
        self.cursor.0 = 0;
        self.ac_state.reset();
    }

    fn backspace(&mut self) {
        if self.cursor.0 > 0 {
            let line = &mut self.content[self.cursor.1];
            // Find byte index of character before cursor
            let byte_idx = Self::char_to_byte_idx(line, self.cursor.0 - 1);
            // Find the character at that position and remove it
            if let Some(c) = line[byte_idx..].chars().next() {
                line.replace_range(byte_idx..byte_idx + c.len_utf8(), "");
            }
            self.cursor.0 -= 1;
            self.update_autocomplete();
        } else if self.cursor.1 > 0 {
            let current_line = self.content.remove(self.cursor.1);
            self.cursor.1 -= 1;
            let prev_line = &mut self.content[self.cursor.1];
            self.cursor.0 = prev_line.chars().count();
            prev_line.push_str(&current_line);
            self.ac_state.reset();
        }
    }

    fn delete(&mut self) {
        let line_char_len = self.line_char_count(self.cursor.1);
        if self.cursor.0 < line_char_len {
            let line = &mut self.content[self.cursor.1];
            let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
            // Find the character at that position and remove it
            if let Some(c) = line[byte_idx..].chars().next() {
                line.replace_range(byte_idx..byte_idx + c.len_utf8(), "");
            }
            self.update_autocomplete();
        } else if self.cursor.1 < self.content.len() - 1 {
            let next_line = self.content.remove(self.cursor.1 + 1);
            let current_line = &mut self.content[self.cursor.1];
            current_line.push_str(&next_line);
        }
    }

    /// Update autocomplete candidates based on current input
    fn update_autocomplete(&mut self) {
        let engine = match &self.engine {
            Some(e) => e,
            None => {
                self.ac_state.reset();
                return;
            }
        };

        let line = &self.content[self.cursor.1];
        let context = self.autocomplete.parse_context(line, self.cursor.0);

        // Parse imports from the current content
        let full_content = self.content.join("\n");
        let imports = parse_imports(&full_content);

        // Get completions with module context and imports
        // Use EditorCompletionSource which can infer local variable types from buffer
        let eng = engine.borrow();
        let source = EditorCompletionSource {
            engine: &eng,
            buffer_content: &full_content,
            module_name: self.module_name.as_deref(),
        };

        let candidates = self.autocomplete.get_completions_with_context(
            &context,
            &source,
            self.module_name.as_deref(),
            &imports,
        );

        // Check if cursor is inside a function call (after '(' or ',')
        // In this case, show completions even with empty prefix
        let inside_function_call = if self.cursor.0 > 0 {
            let before_cursor: String = line.chars().take(self.cursor.0).collect();
            let trimmed = before_cursor.trim_end();
            trimmed.ends_with('(') || trimmed.ends_with(',')
        } else {
            false
        };

        // Only show popup if we have candidates and a non-empty prefix (or dot completion, or inside function call)
        let show_popup = match &context {
            CompletionContext::Identifier { prefix } => {
                !candidates.is_empty() && (!prefix.is_empty() || inside_function_call)
            }
            CompletionContext::ModuleMember { .. } => !candidates.is_empty(),
            CompletionContext::FieldAccess { .. } => !candidates.is_empty(),
            CompletionContext::FilePath { .. } => false, // File path completion not used in editor
        };

        if show_popup {
            self.ac_state.active = true;
            self.ac_state.candidates = candidates;
            self.ac_state.selected = 0;
            self.ac_state.context = Some(context);
        } else {
            self.ac_state.reset();
        }
    }

    /// Show all functions in the current module (Ctrl+F)
    fn show_module_functions(&mut self) {
        let engine = match &self.engine {
            Some(e) => e,
            None => return,
        };

        let module = match &self.module_name {
            Some(m) => m.clone(),
            None => return,
        };

        let eng = engine.borrow();
        let source = EditorCompletionSource {
            engine: &eng,
            buffer_content: "",
            module_name: Some(&module),
        };

        // Get all functions from the current module
        let mut candidates: Vec<CompletionItem> = source.get_functions()
            .into_iter()
            .filter_map(|func| {
                // Check if function belongs to current module
                if let Some(dot_pos) = func.rfind('.') {
                    let func_module = &func[..dot_pos];
                    if func_module == module {
                        let func_suffix = &func[dot_pos + 1..];
                        let short_name = func_suffix.split('/').next().unwrap_or(func_suffix);
                        // Get the base function name (without signature) for signature lookup
                        let base_name = format!("{}.{}", func_module, short_name);
                        let label = if let Some(sig) = source.get_function_signature(&func) {
                            format!("{} :: {}", short_name, sig)
                        } else if let Some(sig) = source.get_function_signature(&base_name) {
                            format!("{} :: {}", short_name, sig)
                        } else {
                            short_name.to_string()
                        };
                        let doc = source.get_function_doc(&func)
                            .or_else(|| source.get_function_doc(&base_name));
                        return Some(CompletionItem {
                            text: short_name.to_string(),
                            label,
                            kind: crate::autocomplete::CompletionKind::Function,
                            doc,
                        });
                    }
                }
                None
            })
            .collect();

        candidates.sort_by(|a, b| a.text.cmp(&b.text));
        candidates.dedup_by(|a, b| a.text == b.text);

        if !candidates.is_empty() {
            self.ac_state.active = true;
            self.ac_state.candidates = candidates;
            self.ac_state.selected = 0;
            // Use empty identifier context so Enter will just insert the function name
            self.ac_state.context = Some(CompletionContext::Identifier { prefix: String::new() });
        }
    }

    /// Accept the currently selected completion
    fn accept_completion(&mut self) {
        if !self.ac_state.active || self.ac_state.candidates.is_empty() {
            return;
        }

        let item = &self.ac_state.candidates[self.ac_state.selected];
        let context = self.ac_state.context.as_ref().unwrap();

        // Calculate how much to replace
        let prefix_len = match context {
            CompletionContext::Identifier { prefix } => prefix.len(),
            CompletionContext::ModuleMember { prefix, .. } => prefix.len(),
            CompletionContext::FieldAccess { prefix, .. } => prefix.len(),
            CompletionContext::FilePath { .. } => return, // File path completion not used in editor
        };

        // Replace prefix with completion text
        let line = &mut self.content[self.cursor.1];
        let cursor_byte = Self::char_to_byte_idx(line, self.cursor.0);
        let prefix_start_byte = Self::char_to_byte_idx(line, self.cursor.0 - prefix_len);

        line.replace_range(prefix_start_byte..cursor_byte, &item.text);

        // Update cursor position
        self.cursor.0 = self.cursor.0 - prefix_len + item.text.chars().count();

        self.ac_state.reset();
    }

    /// Draw the autocomplete popup
    fn draw_autocomplete(&self, printer: &Printer) {
        if !self.ac_state.active || self.ac_state.candidates.is_empty() {
            return;
        }

        // Calculate cursor screen position
        let cursor_screen_x = self.cursor.0.saturating_sub(self.scroll_offset.0);
        let cursor_screen_y = self.cursor.1.saturating_sub(self.scroll_offset.1);

        let total_items = self.ac_state.candidates.len();
        let visible_count = AC_MAX_VISIBLE.min(total_items);
        let scroll_offset = self.ac_state.scroll_offset;

        // Calculate popup width based on visible items
        let popup_width = self.ac_state.candidates.iter()
            .skip(scroll_offset)
            .take(visible_count)
            .map(|c| c.label.len() + 8) // +8 for kind prefix "[type] "
            .max()
            .unwrap_or(20)
            .max(20) // Minimum width for status line
            .min(printer.size.x.saturating_sub(2));

        // Calculate popup height
        let popup_height = visible_count + 1; // +1 for status line

        // Position popup: prefer below cursor, but show above if not enough space
        let space_below = printer.size.y.saturating_sub(cursor_screen_y + 1);
        let space_above = cursor_screen_y;

        let (popup_y, show_above) = if space_below >= popup_height {
            // Enough space below
            (cursor_screen_y + 1, false)
        } else if space_above >= popup_height {
            // Show above cursor
            (cursor_screen_y.saturating_sub(popup_height), true)
        } else {
            // Not enough space either way, prefer the side with more space
            if space_below >= space_above {
                (cursor_screen_y + 1, false)
            } else {
                (cursor_screen_y.saturating_sub(popup_height.min(space_above)), true)
            }
        };

        // Clamp popup_x so the popup doesn't go off the right edge of the screen
        let popup_x = if cursor_screen_x + popup_width > printer.size.x {
            printer.size.x.saturating_sub(popup_width)
        } else {
            cursor_screen_x
        };

        // Limit visible items to available space
        let max_visible = if show_above {
            space_above.min(visible_count)
        } else {
            space_below.saturating_sub(1).min(visible_count) // -1 for status line
        };

        // Draw popup background
        let bg_style = Style::from(ColorStyle::new(
            Color::Rgb(200, 200, 200),
            Color::Rgb(40, 40, 60)
        ));

        // Draw visible items
        for (display_idx, item) in self.ac_state.candidates.iter()
            .skip(scroll_offset)
            .take(max_visible)
            .enumerate()
        {
            let y = popup_y + display_idx;
            if y >= printer.size.y {
                break;
            }

            let actual_idx = scroll_offset + display_idx;
            let is_selected = actual_idx == self.ac_state.selected;
            let (r, g, b) = item.kind.color();
            let kind_color = Color::Rgb(r, g, b);
            let style = if is_selected {
                Style::from(ColorStyle::new(
                    Color::Rgb(0, 0, 0),
                    kind_color
                ))
            } else {
                Style::from(ColorStyle::new(
                    kind_color,
                    Color::Rgb(40, 40, 60)
                ))
            };

            // Format: [fn] name
            let prefix = format!("[{}] ", item.kind.prefix());
            let display = format!("{}{}", prefix, &item.label);
            let display_truncated: String = display.chars().take(popup_width).collect();
            let padded = format!("{:width$}", display_truncated, width = popup_width);

            printer.with_style(style, |p| {
                p.print((popup_x, y), &padded);
            });
        }

        // Show status line with pagination info
        let status_y = popup_y + max_visible;
        if status_y < printer.size.y {
            let current_page = scroll_offset / AC_MAX_VISIBLE + 1;
            let total_pages = (total_items + AC_MAX_VISIBLE - 1) / AC_MAX_VISIBLE;
            let status = if total_pages > 1 {
                format!("[{}/{}] PgUp/PgDn", current_page, total_pages)
            } else {
                format!("[{} items]", total_items)
            };
            let padded_status = format!("{:width$}", status, width = popup_width);
            printer.with_style(bg_style, |p| {
                p.print((popup_x, status_y), &padded_status);
            });
        }

        // Show doc comment for selected item (if available)
        let selected_item = &self.ac_state.candidates[self.ac_state.selected];
        if let Some(ref doc) = selected_item.doc {
            let doc_y = status_y + 1;
            if doc_y < printer.size.y {
                let doc_style = Style::from(ColorStyle::new(
                    Color::Rgb(180, 180, 180),
                    Color::Rgb(30, 30, 45)
                ));
                // Show first line of doc (truncated to fit)
                let first_line = doc.lines().next().unwrap_or("");
                let doc_display: String = first_line.chars().take(popup_width).collect();
                let padded_doc = format!("{:width$}", doc_display, width = popup_width);
                printer.with_style(doc_style, |p| {
                    p.print((popup_x, doc_y), &padded_doc);
                });
            }
        }
    }

    /// Draw compile status indicator at top-right corner
    fn draw_compile_status(&self, printer: &Printer) {
        let (indicator, color) = match &self.compile_status {
            CompileStatus::Unknown => ("?", Color::Rgb(128, 128, 128)),
            CompileStatus::Ok => ("✓", Color::Rgb(0, 255, 0)),
            CompileStatus::Compiled => ("Compiled ✓", Color::Rgb(0, 255, 0)),
            CompileStatus::ParseError(msg) => {
                // Truncate message for display
                let short_msg: String = msg.chars().take(40).collect();
                let display = format!("✗ {}", short_msg);
                // We'll handle this case specially below
                let style = Style::from(ColorStyle::new(
                    Color::Rgb(255, 100, 100),
                    Color::Rgb(40, 0, 0)
                ));
                let x = printer.size.x.saturating_sub(display.len() + 1);
                printer.with_style(style, |p| {
                    p.print((x, 0), &display);
                });
                return;
            }
            CompileStatus::CompileError(msg) => {
                let short_msg: String = msg.chars().take(40).collect();
                let display = format!("✗ {}", short_msg);
                let style = Style::from(ColorStyle::new(
                    Color::Rgb(255, 150, 50),
                    Color::Rgb(40, 20, 0)
                ));
                let x = printer.size.x.saturating_sub(display.len() + 1);
                printer.with_style(style, |p| {
                    p.print((x, 0), &display);
                });
                return;
            }
        };

        // Draw simple indicator
        let style = Style::from(ColorStyle::new(color, Color::Rgb(30, 30, 30)));
        let x = printer.size.x.saturating_sub(indicator.len() + 1);
        printer.with_style(style, |p| {
            p.print((x, 0), indicator);
        });
    }

    /// Draw signature help popup above the cursor
    fn draw_signature_help(&self, printer: &Printer) {
        if !self.sig_help.active {
            return;
        }

        // Calculate cursor screen position
        let screen_x = self.cursor.0.saturating_sub(self.scroll_offset.0);
        let screen_y = self.cursor.1.saturating_sub(self.scroll_offset.1);

        // Position popup above cursor if possible
        let popup_y = if screen_y > 0 { screen_y - 1 } else { screen_y + 1 };

        // Build the signature help text
        let text = if !self.sig_help.params.is_empty() {
            // Show detailed params with current param highlighted
            let params_str: Vec<String> = self.sig_help.params.iter().enumerate()
                .map(|(i, (name, typ, is_opt, _default))| {
                    let opt_marker = if *is_opt { "?" } else { "" };
                    if i == self.sig_help.current_param {
                        format!("[{}{}:{}]", name, opt_marker, typ)
                    } else {
                        format!("{}{}:{}", name, opt_marker, typ)
                    }
                })
                .collect();
            format!("{}({})", self.sig_help.function_name, params_str.join(", "))
        } else if let Some(ref sig) = self.sig_help.signature {
            // Show simple signature
            format!("{} :: {}", self.sig_help.function_name, sig)
        } else {
            return;
        };

        // Draw background
        let width = text.chars().count().min(printer.size.x.saturating_sub(screen_x));
        let bg_style = Style::from(ColorStyle::new(Color::Rgb(200, 200, 200), Color::Rgb(60, 60, 60)));
        printer.with_style(bg_style, |p| {
            p.print((screen_x, popup_y), &text[..text.len().min(width * 4)]); // approximate char count
        });
    }
}

impl View for CodeEditor {
    fn draw(&self, printer: &Printer) {
        let (scroll_x, scroll_y) = self.scroll_offset;
        let view_height = printer.size.y;
        let view_width = printer.size.x;

        // Draw only visible lines
        for screen_y in 0..view_height {
            let line_idx = scroll_y + screen_y;
            if line_idx >= self.content.len() {
                break;
            }

            let line = &self.content[line_idx];

            // First draw the entire line in white as a base (handles unrecognized chars like unclosed quotes)
            let base_style = Style::from(Color::Rgb(255, 255, 255));
            let visible_line: String = line.chars()
                .skip(scroll_x)
                .take(view_width)
                .collect();
            printer.with_style(base_style, |p| {
                p.print((0, screen_y), &visible_line);
            });

            // Syntax highlighting with scroll offset
            for (token, span) in lex(line) {
                let color = match token {
                    Token::Type | Token::Var | Token::If | Token::Then | Token::Else |
                    Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
                    Token::Use | Token::Private | Token::Pub | Token::SelfKw | Token::SelfType |
                    Token::Try | Token::Catch | Token::Finally | Token::Do |
                    Token::While | Token::For | Token::To | Token::Break | Token::Continue |
                    Token::Spawn | Token::SpawnLink | Token::SpawnMonitor | Token::Receive | Token::After |
                    Token::Panic | Token::Extern | Token::From | Token::Test | Token::Quote =>
                        Color::Rgb(255, 0, 255),

                    Token::True | Token::False |
                    Token::Int(_) | Token::HexInt(_) | Token::BinInt(_) |
                    Token::Int8(_) | Token::Int16(_) | Token::Int32(_) |
                    Token::UInt8(_) | Token::UInt16(_) | Token::UInt32(_) | Token::UInt64(_) |
                    Token::BigInt(_) | Token::Float(_) | Token::Float32(_) | Token::Decimal(_) =>
                        Color::Rgb(255, 255, 0),

                    Token::String(_) | Token::SingleQuoteString(_) | Token::Char(_) => Color::Rgb(0, 255, 0),

                    Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent | Token::StarStar |
                    Token::EqEq | Token::NotEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq |
                    Token::AndAnd | Token::OrOr | Token::Bang | Token::PlusPlus | Token::PipeRight |
                    Token::Eq | Token::PlusEq | Token::MinusEq | Token::StarEq | Token::SlashEq |
                    Token::LeftArrow | Token::RightArrow | Token::FatArrow | Token::Caret | Token::Dollar | Token::Question |
                    Token::LParen | Token::RParen | Token::LBracket | Token::RBracket |
                    Token::LBrace | Token::RBrace | Token::Comma | Token::Colon | Token::Dot |
                    Token::Pipe | Token::Hash =>
                        Color::Rgb(255, 165, 0),

                    Token::UpperIdent(_) => Color::Rgb(255, 255, 0),
                    Token::LowerIdent(_) => Color::Rgb(255, 255, 255),

                    Token::Underscore => Color::Rgb(150, 150, 150), // Lighter gray - visible on dark backgrounds
                    Token::Newline => Color::Rgb(255, 255, 255),
                    Token::Comment | Token::MultiLineComment => Color::Rgb(150, 150, 150), // Lighter gray
                    _ => Color::Rgb(255, 255, 255),
                };

                let style = Style::from(color);

                // Calculate screen position with horizontal scroll
                let token_start = span.start;
                let token_end = span.end;

                // Skip tokens entirely before visible area
                if token_end <= scroll_x {
                    continue;
                }

                // Skip tokens entirely after visible area
                if token_start >= scroll_x + view_width {
                    continue;
                }

                // Calculate visible portion of token
                let visible_start = token_start.saturating_sub(scroll_x);
                let text_start = if token_start < scroll_x { scroll_x - token_start } else { 0 };
                let text = &line[span.clone()];

                if text_start < text.len() {
                    let visible_text: String = text.chars().skip(text_start).take(view_width - visible_start).collect();

                    printer.with_style(style, |p| {
                        p.print((visible_start, screen_y), &visible_text);
                    });
                }
            }
        }

        // Draw selection highlight
        if let Some(((sel_start_col, sel_start_row), (sel_end_col, sel_end_row))) = self.selection_range() {
            let sel_style = Style::from(ColorStyle::new(
                Color::Rgb(255, 255, 255),  // White foreground
                Color::Rgb(50, 80, 160),    // Blue background
            ));

            for screen_y in 0..view_height {
                let line_idx = scroll_y + screen_y;
                if line_idx >= self.content.len() {
                    break;
                }

                // Skip lines outside selection
                if line_idx < sel_start_row || line_idx > sel_end_row {
                    continue;
                }

                let line = &self.content[line_idx];
                let line_chars: usize = line.chars().count();

                // Determine the selection range within this line (in char indices)
                let sel_col_start = if line_idx == sel_start_row { sel_start_col } else { 0 };
                let sel_col_end = if line_idx == sel_end_row { sel_end_col } else { line_chars };

                if sel_col_start >= sel_col_end && line_idx == sel_start_row && line_idx == sel_end_row {
                    continue; // Empty selection on single line
                }

                // Render each selected character with highlight
                for col in sel_col_start..sel_col_end {
                    if col < scroll_x || col >= scroll_x + view_width {
                        continue; // Off-screen
                    }
                    let screen_x_pos = col - scroll_x;
                    let ch = line.chars().nth(col).unwrap_or(' ');
                    printer.with_style(sel_style, |p| {
                        p.print((screen_x_pos, screen_y), &ch.to_string());
                    });
                }

                // If selection extends beyond line end on non-last selected line, highlight the trailing space
                if line_idx != sel_end_row && line_chars >= scroll_x && line_chars < scroll_x + view_width {
                    let screen_x_pos = line_chars - scroll_x;
                    printer.with_style(sel_style, |p| {
                        p.print((screen_x_pos, screen_y), " ");
                    });
                }
            }
        }

        // Draw matching bracket highlight
        if let Some((match_row, match_col)) = self.find_matching_bracket() {
            let bracket_style = Style::from(ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(0, 255, 255))); // Cyan background

            // Highlight the bracket at cursor
            if self.cursor.1 >= scroll_y && self.cursor.1 < scroll_y + view_height {
                if self.cursor.0 >= scroll_x && self.cursor.0 < scroll_x + view_width {
                    let screen_x = self.cursor.0 - scroll_x;
                    let screen_y = self.cursor.1 - scroll_y;
                    let char_at_cursor = self.content[self.cursor.1].chars().nth(self.cursor.0).unwrap_or(' ');
                    printer.with_style(bracket_style, |p| {
                        p.print((screen_x, screen_y), &char_at_cursor.to_string());
                    });
                }
            }

            // Highlight the matching bracket
            if match_row >= scroll_y && match_row < scroll_y + view_height {
                if match_col >= scroll_x && match_col < scroll_x + view_width {
                    let screen_x = match_col - scroll_x;
                    let screen_y = match_row - scroll_y;
                    let char_at_match = self.content[match_row].chars().nth(match_col).unwrap_or(' ');
                    printer.with_style(bracket_style, |p| {
                        p.print((screen_x, screen_y), &char_at_match.to_string());
                    });
                }
            }
        }

        // Draw cursor (simple block) - adjusted for scroll
        // cursor.0 is a character index, not byte index
        if self.cursor.1 >= scroll_y && self.cursor.1 < scroll_y + view_height {
            if self.cursor.0 >= scroll_x && self.cursor.0 < scroll_x + view_width {
                let screen_x = self.cursor.0 - scroll_x;
                let screen_y = self.cursor.1 - scroll_y;

                let cursor_style = Style::from(ColorStyle::new(Color::Rgb(0,0,0), Color::Rgb(0, 255, 0)));
                printer.with_style(cursor_style, |p| {
                    let line_char_count = self.content[self.cursor.1].chars().count();
                    let char_at_cursor = if self.cursor.1 < self.content.len() && self.cursor.0 < line_char_count {
                        self.content[self.cursor.1].chars().nth(self.cursor.0).unwrap_or(' ')
                    } else {
                        ' '
                    };
                    p.print((screen_x, screen_y), &char_at_cursor.to_string());
                });
            }
        }

        // Draw signature help popup (above cursor)
        self.draw_signature_help(printer);

        // Draw autocomplete popup
        self.draw_autocomplete(printer);

        // Draw compile status indicator at top-right
        self.draw_compile_status(printer);
    }

    fn layout(&mut self, size: Vec2) {
        self.last_size = size;
        self.ensure_cursor_visible();
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        // We handle scrolling internally, so just accept whatever size we're given
        constraint
    }

    fn important_area(&self, _view_size: Vec2) -> Rect {
        // Return rect around cursor position on screen (after scroll adjustment)
        let screen_x = self.cursor.0.saturating_sub(self.scroll_offset.0);
        let screen_y = self.cursor.1.saturating_sub(self.scroll_offset.1);
        Rect::from_point(Vec2::new(screen_x, screen_y))
    }

    fn take_focus(&mut self, _: Direction) -> Result<EventResult, CannotFocus> {
        Ok(EventResult::Consumed(None))
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        // Track line before event for compile status checking
        let old_line = self.cursor.1;

        // Check if this is an editing event (for tracking edit time)
        let is_edit = matches!(event,
            Event::Char(_) | Event::Key(Key::Enter) | Event::Key(Key::Backspace) |
            Event::Key(Key::Del) | Event::Key(Key::Tab) | Event::CtrlChar('k')
        );

        // In read-only mode, ignore all editing events but allow navigation
        if self.read_only {
            match event {
                // Navigation is allowed
                Event::Key(Key::Left) | Event::Key(Key::Right) |
                Event::Key(Key::Up) | Event::Key(Key::Down) |
                Event::Key(Key::Home) | Event::Key(Key::End) |
                Event::Key(Key::PageUp) | Event::Key(Key::PageDown) |
                Event::Key(Key::Esc) | Event::CtrlChar('a') => { /* fall through to normal handling */ }
                // Editing is blocked
                Event::Char(_) | Event::Key(Key::Enter) | Event::Key(Key::Backspace) |
                Event::Key(Key::Del) | Event::Key(Key::Tab) | Event::CtrlChar('k') => {
                    return EventResult::Consumed(None); // Consume but do nothing
                }
                _ => { /* other events fall through */ }
            }
        }

        let result = match event {
            // Tab: accept completion or cycle through candidates
            Event::Key(Key::Tab) => {
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    // If only one candidate, accept it
                    if self.ac_state.candidates.len() == 1 {
                        self.save_snapshot();
                        self.accept_completion();
                    } else {
                        // Cycle to next
                        self.ac_state.select_next();
                    }
                    return EventResult::Consumed(None);
                }
                // Regular tab inserts spaces
                self.save_snapshot();
                self.insert_char(' ');
                self.insert_char(' ');
                self.insert_char(' ');
                self.insert_char(' ');
                EventResult::Consumed(None)
            }
            // Shift+Tab: cycle backwards
            Event::Shift(Key::Tab) => {
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.ac_state.select_prev();
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            Event::Char(c) => {
                self.save_snapshot();
                if self.has_selection() {
                    self.delete_selection();
                }
                self.insert_char(c);
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                // If autocomplete is active, accept the selection
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.save_snapshot();
                    self.accept_completion();
                    return EventResult::Consumed(None);
                }
                self.save_snapshot();
                if self.has_selection() {
                    self.delete_selection();
                }
                self.insert_newline();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Esc) => {
                // If autocomplete is active, close it
                if self.ac_state.active {
                    self.ac_state.reset();
                    return EventResult::Consumed(None);
                }
                // Clear selection
                if self.has_selection() {
                    self.clear_selection();
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            Event::Key(Key::Backspace) => {
                self.save_snapshot();
                if self.has_selection() {
                    self.delete_selection();
                } else {
                    self.backspace();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Del) => {
                self.save_snapshot();
                if self.has_selection() {
                    self.delete_selection();
                } else {
                    self.delete();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Left) => {
                self.ac_state.reset();
                self.clear_selection();
                if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                } else if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.cursor.0 = self.line_char_count(self.cursor.1);
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
                self.ac_state.reset();
                self.clear_selection();
                let line_char_len = self.line_char_count(self.cursor.1);
                if self.cursor.0 < line_char_len {
                    self.cursor.0 += 1;
                } else if self.cursor.1 < self.content.len() - 1 {
                    self.cursor.1 += 1;
                    self.cursor.0 = 0;
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Up) => {
                // If autocomplete is active, navigate up
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.ac_state.select_prev();
                    return EventResult::Consumed(None);
                }
                self.clear_selection();
                if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.fix_cursor_x();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                // If autocomplete is active, navigate down
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.ac_state.select_next();
                    return EventResult::Consumed(None);
                }
                self.clear_selection();
                if self.cursor.1 < self.content.len() - 1 {
                    self.cursor.1 += 1;
                    self.fix_cursor_x();
                }
                EventResult::Consumed(None)
            }
            // Shift+Arrow: extend selection
            Event::Shift(Key::Left) => {
                self.ac_state.reset();
                self.ensure_selection_anchor();
                if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                } else if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.cursor.0 = self.line_char_count(self.cursor.1);
                }
                EventResult::Consumed(None)
            }
            Event::Shift(Key::Right) => {
                self.ac_state.reset();
                self.ensure_selection_anchor();
                let line_char_len = self.line_char_count(self.cursor.1);
                if self.cursor.0 < line_char_len {
                    self.cursor.0 += 1;
                } else if self.cursor.1 < self.content.len() - 1 {
                    self.cursor.1 += 1;
                    self.cursor.0 = 0;
                }
                EventResult::Consumed(None)
            }
            Event::Shift(Key::Up) => {
                self.ac_state.reset();
                self.ensure_selection_anchor();
                if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.fix_cursor_x();
                }
                EventResult::Consumed(None)
            }
            Event::Shift(Key::Down) => {
                self.ac_state.reset();
                self.ensure_selection_anchor();
                if self.cursor.1 < self.content.len() - 1 {
                    self.cursor.1 += 1;
                    self.fix_cursor_x();
                }
                EventResult::Consumed(None)
            }
            Event::Shift(Key::Home) => {
                self.ac_state.reset();
                self.ensure_selection_anchor();
                self.cursor.0 = 0;
                EventResult::Consumed(None)
            }
            Event::Shift(Key::End) => {
                self.ac_state.reset();
                self.ensure_selection_anchor();
                self.cursor.0 = self.line_char_count(self.cursor.1);
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageUp) => {
                // If autocomplete is active, page up
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.ac_state.page_up();
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            Event::Key(Key::PageDown) => {
                // If autocomplete is active, page down
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.ac_state.page_down();
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            Event::Key(Key::Home) => {
                self.ac_state.reset();
                self.clear_selection();
                self.cursor.0 = 0;
                EventResult::Consumed(None)
            }
            Event::Key(Key::End) => {
                self.ac_state.reset();
                self.clear_selection();
                self.cursor.0 = self.line_char_count(self.cursor.1);
                EventResult::Consumed(None)
            }
            // Ctrl+Space to trigger autocomplete manually
            Event::CtrlChar(' ') => {
                self.update_autocomplete();
                // Show even with empty prefix
                if !self.ac_state.candidates.is_empty() {
                    self.ac_state.active = true;
                }
                EventResult::Consumed(None)
            }
            // Ctrl+F to show all functions in current module
            Event::CtrlChar('f') => {
                self.show_module_functions();
                EventResult::Consumed(None)
            }
            // Ctrl+A to go to start of line
            Event::CtrlChar('a') => {
                self.ac_state.reset();
                self.clear_selection();
                self.cursor.0 = 0;
                EventResult::Consumed(None)
            }
            // Ctrl+E to go to end of line (common readline behavior)
            Event::CtrlChar('e') => {
                self.ac_state.reset();
                self.clear_selection();
                self.cursor.0 = self.line_char_count(self.cursor.1);
                EventResult::Consumed(None)
            }
            // Ctrl+Z to undo
            Event::CtrlChar('z') => {
                self.undo();
                EventResult::Consumed(None)
            }
            // Ctrl+R to redo
            Event::CtrlChar('r') => {
                self.redo();
                EventResult::Consumed(None)
            }
            // Ctrl+K to delete current line
            Event::CtrlChar('k') => {
                self.save_snapshot();
                self.ac_state.reset();
                if self.content.len() > 1 {
                    self.content.remove(self.cursor.1);
                    // Adjust cursor if it was on the last line
                    if self.cursor.1 >= self.content.len() {
                        self.cursor.1 = self.content.len() - 1;
                    }
                    // Ensure cursor column is within bounds
                    let line_len = self.line_char_count(self.cursor.1);
                    if self.cursor.0 > line_len {
                        self.cursor.0 = line_len;
                    }
                } else {
                    // Only one line - clear it instead of removing
                    self.content[0].clear();
                    self.cursor.0 = 0;
                }
                EventResult::Consumed(None)
            }
            // Ctrl+X to cut selection to clipboard
            Event::CtrlChar('x') => {
                if let Some(text) = self.get_selected_text() {
                    self.save_snapshot();
                    let _ = copy_to_system_clipboard(&text);
                    self.delete_selection();
                }
                EventResult::Consumed(None)
            }
            // Alt+E to jump to error line
            Event::AltChar('e') | Event::AltChar('E') => {
                if let Some(line) = self.get_error_line() {
                    // Line numbers in errors are 1-indexed, cursor row is 0-indexed
                    let target_row = line.saturating_sub(1);
                    if target_row < self.content.len() {
                        self.cursor.1 = target_row;
                        self.cursor.0 = 0; // Go to start of line
                        self.ac_state.reset();
                    }
                }
                EventResult::Consumed(None)
            }
            _ => EventResult::Ignored,
        };

        // After any consumed event, ensure cursor stays visible
        if matches!(result, EventResult::Consumed(_)) {
            self.ensure_cursor_visible();

            // Track edits for debounced compile
            if is_edit {
                self.on_edit();
            }

            // Check for line change (triggers parse check)
            if self.cursor.1 != old_line {
                self.on_line_change(old_line, self.cursor.1);
            }
        }

        result
    }
}
