//! Interactive REPL panel with syntax highlighting and autocomplete
//!
//! Provides a notebook-style REPL where each input/output pair is displayed
//! in a scrollable view with syntax highlighting.

use cursive::event::{Callback, Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle, Style};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Cursive, Printer, Vec2, Rect};
use nostos_syntax::lexer::{Token, lex};
use std::rc::Rc;
use std::cell::RefCell;
use nostos_repl::ReplEngine;
use nostos_vm::ThreadedEvalHandle;
use std::fs;
use std::env;
use std::path::PathBuf;
use std::io::Write;

use crate::autocomplete::{Autocomplete, CompletionContext, CompletionItem, CompletionKind, CompletionSource};

/// Wrapper to implement CompletionSource for ReplEngine
struct EngineCompletionSource<'a> {
    engine: &'a ReplEngine,
}

impl<'a> CompletionSource for EngineCompletionSource<'a> {
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
}

/// A single REPL entry (input + output)
#[derive(Clone)]
pub struct ReplEntry {
    /// The input code
    input: Vec<String>,
    /// The output/result (None if not yet evaluated)
    output: Option<ReplOutput>,
}

/// Output from evaluating an entry
#[derive(Clone)]
pub enum ReplOutput {
    /// Successful evaluation with result
    Success(String),
    /// Error during evaluation
    Error(String),
    /// Definition (no visible result, just confirmation)
    Definition(String),
}

impl ReplEntry {
    fn new() -> Self {
        Self {
            input: vec![String::new()],
            output: None,
        }
    }

    /// Total height in lines (input lines + output lines)
    fn height(&self) -> usize {
        let input_height = self.input.len();
        let output_height = match &self.output {
            None => 0,
            Some(ReplOutput::Success(s)) | Some(ReplOutput::Error(s)) | Some(ReplOutput::Definition(s)) => {
                if s.is_empty() { 0 } else { s.lines().count().max(1) }
            }
        };
        input_height + output_height
    }
}

/// Autocomplete state
struct AutocompleteState {
    /// Whether autocomplete popup is visible
    active: bool,
    /// Current completion candidates
    candidates: Vec<CompletionItem>,
    /// Selected index in candidates
    selected: usize,
    /// Scroll offset (first visible item index)
    scroll_offset: usize,
    /// The context (prefix info)
    context: Option<CompletionContext>,
}

/// Maximum visible items in autocomplete popup
const AC_MAX_VISIBLE: usize = 10;

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

    /// Move selection up, adjusting scroll if needed
    fn select_prev(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        if self.selected == 0 {
            self.selected = self.candidates.len() - 1;
            // Scroll to show the last item
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

    /// Move selection down, adjusting scroll if needed
    fn select_next(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        self.selected = (self.selected + 1) % self.candidates.len();
        if self.selected == 0 {
            // Wrapped to beginning
            self.scroll_offset = 0;
        } else if self.selected >= self.scroll_offset + AC_MAX_VISIBLE {
            // Scroll down if selection goes below visible area
            self.scroll_offset = self.selected - AC_MAX_VISIBLE + 1;
        }
    }
}

/// Result of evaluating input - whether to close the panel
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EvalResult {
    Continue,
    Exit,
}

/// Interactive REPL panel
pub struct ReplPanel {
    /// History of evaluated entries
    history: Vec<ReplEntry>,
    /// Current input being edited
    current: ReplEntry,
    /// Cursor position in current input (col, row)
    cursor: (usize, usize),
    /// Scroll offset (horizontal, vertical)
    scroll_offset: (usize, usize),
    /// Last known size
    last_size: Vec2,
    /// Engine for evaluation
    engine: Rc<RefCell<ReplEngine>>,
    /// REPL instance number (for display)
    instance_id: usize,
    /// Autocomplete engine
    autocomplete: Autocomplete,
    /// Autocomplete state
    ac_state: AutocompleteState,
    /// History navigation index (None = current edit buffer)
    history_index: Option<usize>,
    /// Stashed current input when navigating history
    stash_current: Option<Vec<String>>,
    /// Persistent command history (separate from display history)
    command_history: Vec<Vec<String>>,
    /// Whether an async evaluation is in progress
    eval_in_progress: bool,
    /// Handle for async evaluation (owned by this panel, includes independent interrupt)
    eval_handle: Option<ThreadedEvalHandle>,
}

impl ReplPanel {
    pub fn new(engine: Rc<RefCell<ReplEngine>>, instance_id: usize) -> Self {
        let mut autocomplete = Autocomplete::new();
        // Initialize autocomplete from engine
        {
            let eng = engine.borrow();
            let source = EngineCompletionSource { engine: &eng };
            autocomplete.update_from_source(&source);
        }

        Self {
            history: Vec::new(),
            current: ReplEntry::new(),
            cursor: (0, 0),
            scroll_offset: (0, 0),
            last_size: Vec2::zero(),
            engine,
            instance_id,
            autocomplete,

            ac_state: AutocompleteState::new(),
            history_index: None,
            stash_current: None,
            command_history: Self::load_history_from_disk(),
            eval_in_progress: false,
            eval_handle: None,
        }
    }

    /// Check if all braces/brackets/parens are balanced in the current input
    fn has_balanced_delimiters(&self) -> bool {
        let mut brace_count = 0i32;
        let mut bracket_count = 0i32;
        let mut paren_count = 0i32;
        let mut in_string = false;
        let mut prev_char = '\0';

        for line in &self.current.input {
            for ch in line.chars() {
                // Simple string detection (not handling escape sequences perfectly)
                if ch == '"' && prev_char != '\\' {
                    in_string = !in_string;
                }

                if !in_string {
                    match ch {
                        '{' => brace_count += 1,
                        '}' => brace_count -= 1,
                        '[' => bracket_count += 1,
                        ']' => bracket_count -= 1,
                        '(' => paren_count += 1,
                        ')' => paren_count -= 1,
                        _ => {}
                    }
                }
                prev_char = ch;
            }
        }

        brace_count == 0 && bracket_count == 0 && paren_count == 0
    }

    /// Evaluate the current input
    /// Returns Exit if :exit command was entered
    fn evaluate(&mut self) -> EvalResult {
        let input_text = self.current.input.join("\n");
        let trimmed = input_text.trim();

        if trimmed.is_empty() {
            return EvalResult::Continue;
        }

        // Check for :exit command
        if trimmed == ":exit" || trimmed == ":quit" || trimmed == ":q" {
            return EvalResult::Exit;
        }

        // For REPL commands (: prefix), use synchronous eval (they're quick)
        if trimmed.starts_with(':') {
            let result = self.engine.borrow_mut().eval(&input_text);
            self.finish_eval_with_result(result);
            return EvalResult::Continue;
        }

        // Try async evaluation for expressions
        let async_result = self.engine.borrow_mut().start_eval_async(&input_text);
        match async_result {
            Ok(handle) => {
                // Async eval started - store handle and mark as in progress
                self.eval_handle = Some(handle);
                self.eval_in_progress = true;
                self.current.output = Some(ReplOutput::Definition("Evaluating...".to_string()));
            }
            Err(e) if e == "Use eval() for commands" => {
                // This shouldn't happen since we handle : above, but fallback to sync
                let result = self.engine.borrow_mut().eval(&input_text);
                self.finish_eval_with_result(result);
            }
            Err(e) => {
                // Compile-time error - show it immediately
                self.current.output = Some(ReplOutput::Error(e));
                self.finalize_entry();
            }
        }

        EvalResult::Continue
    }

    /// Complete evaluation with a result (used for sync eval and polling)
    fn finish_eval_with_result(&mut self, result: Result<String, String>) {
        // Drain any output from spawned processes
        let spawned_output = self.engine.borrow().drain_output();

        self.current.output = Some(match result {
            Ok(mut output) => {
                // Append any output from spawned processes
                for line in spawned_output {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&line);
                }
                if output.is_empty() || output.starts_with("Defined") || output.starts_with("Type") {
                    ReplOutput::Definition(output)
                } else {
                    ReplOutput::Success(output)
                }
            }
            Err(e) => ReplOutput::Error(e),
        });

        self.finalize_entry();
    }

    /// Finalize the current entry (move to history, reset state)
    fn finalize_entry(&mut self) {
        // Move current to history and start fresh
        self.history.push(self.current.clone());
        self.command_history.push(self.current.input.clone());
        self.current = ReplEntry::new();
        self.cursor = (0, 0);

        self.history_index = None;
        self.stash_current = None;
        self.eval_in_progress = false;
        self.eval_handle = None;

        // Auto-save history to disk
        self.save_history_to_disk();

        // Update autocomplete cache after evaluation (new definitions may be available)
        {
            let eng = self.engine.borrow();
            let source = EngineCompletionSource { engine: &eng };
            self.autocomplete.update_from_source(&source);
        }

        // Scroll to bottom
        self.scroll_to_bottom();
    }

    /// Poll for async evaluation result. Returns true if result was received.
    pub fn poll_eval_result(&mut self) -> bool {
        if !self.eval_in_progress {
            return false;
        }

        let handle = match &self.eval_handle {
            Some(h) => h,
            None => return false,
        };

        match handle.try_recv() {
            Ok(Ok(result)) => {
                let output = if result.is_unit() {
                    String::new()
                } else {
                    result.display()
                };
                self.finish_eval_with_result(Ok(output));
                true
            }
            Ok(Err(e)) => {
                let err_msg = if e.contains("Interrupted") {
                    "Interrupted".to_string()
                } else {
                    format!("Runtime error: {}", e)
                };
                self.finish_eval_with_result(Err(err_msg));
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.finish_eval_with_result(Err("Evaluation thread crashed".to_string()));
                true
            }
        }
    }

    /// Cancel the current async evaluation
    pub fn cancel_eval(&mut self) {
        if let Some(handle) = &self.eval_handle {
            handle.cancel();
            // The poll will pick up the interrupted result
        }
    }

    /// Check if an evaluation is in progress
    pub fn is_eval_in_progress(&self) -> bool {
        self.eval_in_progress
    }

    /// Calculate total content height
    fn total_height(&self) -> usize {
        let history_height: usize = self.history.iter().map(|e| e.height()).sum();
        let current_height = self.current.input.len();
        history_height + current_height
    }

    /// Scroll to make cursor visible
    fn ensure_cursor_visible(&mut self) {
        let view_height = self.last_size.y;
        let view_width = self.last_size.x.saturating_sub(5); // Account for prompt

        if view_height == 0 || view_width == 0 {
            return;
        }

        // Calculate cursor's absolute Y position
        let history_height: usize = self.history.iter().map(|e| e.height()).sum();
        let cursor_abs_y = history_height + self.cursor.1;

        // Vertical scrolling
        if cursor_abs_y < self.scroll_offset.1 {
            self.scroll_offset.1 = cursor_abs_y;
        } else if cursor_abs_y >= self.scroll_offset.1 + view_height {
            self.scroll_offset.1 = cursor_abs_y.saturating_sub(view_height) + 1;
        }

        // Horizontal scrolling (account for prompt width)
        let cursor_x = self.cursor.0;
        if cursor_x < self.scroll_offset.0 {
            self.scroll_offset.0 = cursor_x;
        } else if cursor_x >= self.scroll_offset.0 + view_width {
            self.scroll_offset.0 = cursor_x.saturating_sub(view_width) + 1;
        }
    }

    /// Scroll to bottom
    fn scroll_to_bottom(&mut self) {
        let total = self.total_height();
        let view_height = self.last_size.y;
        if total > view_height {
            self.scroll_offset.1 = total.saturating_sub(view_height);
        }
    }

    fn line_char_count(&self, line_idx: usize) -> usize {
        self.current.input[line_idx].chars().count()
    }

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

    fn insert_char(&mut self, c: char) {
        let line = &mut self.current.input[self.cursor.1];
        let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
        line.insert(byte_idx, c);
        self.cursor.0 += 1;

        // Update autocomplete after each character
        self.update_autocomplete();
    }

    fn insert_newline(&mut self) {
        let line = &mut self.current.input[self.cursor.1];
        let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
        let rest = line[byte_idx..].to_string();
        line.truncate(byte_idx);
        self.current.input.insert(self.cursor.1 + 1, rest);
        self.cursor.1 += 1;
        self.cursor.0 = 0;
        self.ac_state.reset();
    }

    fn backspace(&mut self) {
        if self.cursor.0 > 0 {
            let line = &mut self.current.input[self.cursor.1];
            let char_count = line.chars().count();
            if self.cursor.0 <= char_count {
                let byte_idx = Self::char_to_byte_idx(line, self.cursor.0 - 1);
                let c = line[byte_idx..].chars().next().unwrap();
                line.replace_range(byte_idx..byte_idx + c.len_utf8(), "");
            }
            self.cursor.0 -= 1;
            self.update_autocomplete();
        } else if self.cursor.1 > 0 {
            let current_line = self.current.input.remove(self.cursor.1);
            self.cursor.1 -= 1;
            self.cursor.0 = self.line_char_count(self.cursor.1);
            self.current.input[self.cursor.1].push_str(&current_line);
            self.ac_state.reset();
        }
    }

    fn delete(&mut self) {
        let line_char_len = self.line_char_count(self.cursor.1);
        if self.cursor.0 < line_char_len {
            let line = &mut self.current.input[self.cursor.1];
            let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
            if let Some(c) = line[byte_idx..].chars().next() {
                line.replace_range(byte_idx..byte_idx + c.len_utf8(), "");
            }
            self.update_autocomplete();
        } else if self.cursor.1 < self.current.input.len() - 1 {
            let next_line = self.current.input.remove(self.cursor.1 + 1);
            self.current.input[self.cursor.1].push_str(&next_line);
        }
    }

    /// Update autocomplete candidates based on current input
    fn update_autocomplete(&mut self) {
        let line = &self.current.input[self.cursor.1];
        let context = self.autocomplete.parse_context(line, self.cursor.0);

        // Get completions
        let eng = self.engine.borrow();
        let source = EngineCompletionSource { engine: &eng };
        let candidates = self.autocomplete.get_completions(&context, &source);

        // Only show popup if we have candidates and a non-empty prefix (or dot completion)
        let show_popup = match &context {
            CompletionContext::Identifier { prefix } => !prefix.is_empty() && !candidates.is_empty(),
            CompletionContext::ModuleMember { .. } => !candidates.is_empty(),
            CompletionContext::FieldAccess { .. } => !candidates.is_empty(),
        };

        if show_popup {
            self.ac_state.active = true;
            self.ac_state.candidates = candidates;
            self.ac_state.selected = 0;
            self.ac_state.scroll_offset = 0;
            self.ac_state.context = Some(context);
        } else {
            self.ac_state.reset();
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
        };

        // Replace prefix with completion text
        let line = &mut self.current.input[self.cursor.1];
        let cursor_byte = Self::char_to_byte_idx(line, self.cursor.0);
        let prefix_start_byte = Self::char_to_byte_idx(line, self.cursor.0 - prefix_len);

        line.replace_range(prefix_start_byte..cursor_byte, &item.text);

        // Update cursor position
        self.cursor.0 = self.cursor.0 - prefix_len + item.text.chars().count();

        self.ac_state.reset();
    }

    /// Draw a line with syntax highlighting
    fn draw_highlighted_line(&self, printer: &Printer, line: &str, y: usize, x_offset: usize) {
        let scroll_x = self.scroll_offset.0;
        let view_width = printer.size.x.saturating_sub(x_offset);

        // First draw the entire line in white as a base (handles unrecognized chars like unclosed quotes)
        let base_style = Style::from(Color::Rgb(255, 255, 255));
        let visible_line: String = line.chars()
            .skip(scroll_x)
            .take(view_width)
            .collect();
        printer.with_style(base_style, |p| {
            p.print((x_offset, y), &visible_line);
        });

        // Then overlay syntax highlighting for recognized tokens
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

                Token::String(_) | Token::Char(_) => Color::Rgb(0, 255, 0),

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
            let token_start = span.start;
            let token_end = span.end;

            if token_end <= scroll_x || token_start >= scroll_x + view_width {
                continue;
            }

            let visible_start = token_start.saturating_sub(scroll_x);
            let text_start = if token_start < scroll_x { scroll_x - token_start } else { 0 };
            let text = &line[span.clone()];

            if text_start < text.len() {
                let visible_text: String = text.chars()
                    .skip(text_start)
                    .take(view_width.saturating_sub(visible_start))
                    .collect();

                printer.with_style(style, |p| {
                    p.print((x_offset + visible_start, y), &visible_text);
                });
            }
        }
    }

    /// Draw the autocomplete popup
    fn draw_autocomplete(&self, printer: &Printer, cursor_screen_y: usize) {
        if !self.ac_state.active || self.ac_state.candidates.is_empty() {
            return;
        }

        let prompt_width = 4;
        let total_candidates = self.ac_state.candidates.len();
        let visible_count = AC_MAX_VISIBLE.min(total_candidates);
        let scroll_offset = self.ac_state.scroll_offset;

        // Calculate popup width based on visible items
        let popup_width = self.ac_state.candidates.iter()
            .skip(scroll_offset)
            .take(visible_count)
            .map(|c| c.label.len() + 6) // +6 for kind prefix "[fn] "
            .max()
            .unwrap_or(20)
            .min(printer.size.x.saturating_sub(prompt_width + 2));

        // Position popup below cursor line
        let popup_y = cursor_screen_y + 1;
        let popup_x = prompt_width + self.cursor.0.saturating_sub(self.scroll_offset.0);

        // Draw popup background
        let bg_style = Style::from(ColorStyle::new(
            Color::Rgb(200, 200, 200),
            Color::Rgb(40, 40, 60)
        ));

        // Draw visible items (starting from scroll_offset)
        for (display_idx, item) in self.ac_state.candidates.iter()
            .skip(scroll_offset)
            .take(visible_count)
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

        // Show scroll indicator if there are hidden items
        let status_y = popup_y + visible_count;
        let items_above = scroll_offset;
        let items_below = total_candidates.saturating_sub(scroll_offset + visible_count);

        if items_above > 0 || items_below > 0 {
            let indicator = if items_above > 0 && items_below > 0 {
                format!("↑{} ↓{} ({}/{})", items_above, items_below, self.ac_state.selected + 1, total_candidates)
            } else if items_above > 0 {
                format!("↑{} ({}/{})", items_above, self.ac_state.selected + 1, total_candidates)
            } else {
                format!("↓{} ({}/{})", items_below, self.ac_state.selected + 1, total_candidates)
            };
            printer.with_style(bg_style, |p| {
                p.print((popup_x, status_y), &indicator);
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
}

impl View for ReplPanel {
    fn draw(&self, printer: &Printer) {
        let scroll_y = self.scroll_offset.1;
        let view_height = printer.size.y;
        let prompt_width = 4; // ">>> " or "... "

        let mut screen_y: usize = 0;
        let mut content_y: usize = 0;

        // Draw prompt style
        let prompt_style = Style::from(Color::Rgb(100, 200, 100));
        let output_success_style = Style::from(Color::Rgb(100, 255, 100));
        let output_error_style = Style::from(Color::Rgb(255, 100, 100));
        let output_def_style = Style::from(Color::Rgb(150, 150, 150));

        // Track cursor screen position for autocomplete popup
        let mut cursor_screen_y = 0;

        // Draw history entries
        for entry in &self.history {
            // Draw input lines
            for (i, line) in entry.input.iter().enumerate() {
                if content_y >= scroll_y && screen_y < view_height {
                    // Draw prompt
                    let prompt = if i == 0 { ">>> " } else { "... " };
                    printer.with_style(prompt_style, |p| {
                        p.print((0, screen_y), prompt);
                    });

                    // Draw highlighted code
                    self.draw_highlighted_line(printer, line, screen_y, prompt_width);
                    screen_y += 1;
                }
                content_y += 1;
            }

            // Draw output
            if let Some(ref output) = entry.output {
                let (text, style) = match output {
                    ReplOutput::Success(s) => (s.as_str(), output_success_style),
                    ReplOutput::Error(s) => (s.as_str(), output_error_style),
                    ReplOutput::Definition(s) => (s.as_str(), output_def_style),
                };

                if !text.is_empty() {
                    for line in text.lines() {
                        if content_y >= scroll_y && screen_y < view_height {
                            printer.with_style(style, |p| {
                                p.print((0, screen_y), line);
                            });
                            screen_y += 1;
                        }
                        content_y += 1;
                    }
                }
            }
        }

        // Draw current input
        for (i, line) in self.current.input.iter().enumerate() {
            if content_y >= scroll_y && screen_y < view_height {
                // Draw prompt
                let prompt = if i == 0 { ">>> " } else { "... " };
                printer.with_style(prompt_style, |p| {
                    p.print((0, screen_y), prompt);
                });

                // Draw highlighted code
                self.draw_highlighted_line(printer, line, screen_y, prompt_width);

                // Draw cursor if on this line
                if i == self.cursor.1 {
                    cursor_screen_y = screen_y;
                    let cursor_screen_x = prompt_width + self.cursor.0.saturating_sub(self.scroll_offset.0);
                    if cursor_screen_x < printer.size.x {
                        let cursor_style = Style::from(ColorStyle::new(
                            Color::Rgb(0, 0, 0),
                            Color::Rgb(0, 255, 0)
                        ));
                        printer.with_style(cursor_style, |p| {
                            let char_at_cursor = if self.cursor.0 < line.chars().count() {
                                line.chars().nth(self.cursor.0).unwrap_or(' ')
                            } else {
                                ' '
                            };
                            p.print((cursor_screen_x, screen_y), &char_at_cursor.to_string());
                        });
                    }
                }

                screen_y += 1;
            }
            content_y += 1;
        }

        // Draw autocomplete popup
        self.draw_autocomplete(printer, cursor_screen_y);

        // Draw scrollbar if needed
        let total_height = self.total_height();
        if total_height > view_height && printer.size.x > 1 {
            let scrollbar_x = printer.size.x - 1;
            let scrollbar_height = (view_height * view_height / total_height).max(1);
            let scrollbar_pos = scroll_y * view_height / total_height;

            for y in 0..view_height {
                let ch = if y >= scrollbar_pos && y < scrollbar_pos + scrollbar_height {
                    '█'
                } else {
                    '░'
                };
                printer.print((scrollbar_x, y), &ch.to_string());
            }
        }
    }

    fn layout(&mut self, size: Vec2) {
        self.last_size = size;
        self.ensure_cursor_visible();
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        constraint
    }

    fn important_area(&self, _view_size: Vec2) -> Rect {
        let history_height: usize = self.history.iter().map(|e| e.height()).sum();
        let cursor_abs_y = history_height + self.cursor.1;
        let screen_x = 4 + self.cursor.0.saturating_sub(self.scroll_offset.0);
        let screen_y = cursor_abs_y.saturating_sub(self.scroll_offset.1);
        Rect::from_point(Vec2::new(screen_x, screen_y))
    }

    fn take_focus(&mut self, _: Direction) -> Result<EventResult, CannotFocus> {
        Ok(EventResult::Consumed(None))
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        // During evaluation, only Escape can cancel - block all other input
        if self.eval_in_progress {
            if let Event::Key(Key::Esc) = event {
                self.cancel_eval();
                return EventResult::Consumed(None);
            }
            // Block all other input during evaluation
            return EventResult::Consumed(None);
        }

        match event {
            // Tab: accept completion or cycle through candidates
            Event::Key(Key::Tab) => {
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    // If only one candidate, accept it
                    if self.ac_state.candidates.len() == 1 {
                        self.accept_completion();
                    } else {
                        // Cycle to next (with scroll support)
                        self.ac_state.select_next();
                    }
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            // Shift+Tab: cycle backwards
            Event::Shift(Key::Tab) => {
                if self.ac_state.active && self.ac_state.candidates.len() > 1 {
                    self.ac_state.select_prev();
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            Event::Char(c) => {
                self.insert_char(c);
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                // If autocomplete is active, accept the selection
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.accept_completion();
                    self.ensure_cursor_visible();
                    return EventResult::Consumed(None);
                }

                // Check for multiline continuation
                let current_line = &self.current.input[self.cursor.1];
                let trimmed = current_line.trim_end();

                // Continue multiline if:
                // 1. Current line ends with continuation chars ({ do \)
                // 2. OR delimiters are not balanced (we're inside a block/array/etc)
                let needs_continuation = trimmed.ends_with('\\')
                    || trimmed.ends_with('{')
                    || trimmed.ends_with("do")
                    || !self.has_balanced_delimiters();

                if needs_continuation {
                    self.insert_newline();
                } else {
                    // Evaluate and check for :exit
                    if self.evaluate() == EvalResult::Exit {
                        // Return a callback to close this REPL panel
                        let id = self.instance_id;
                        return EventResult::Consumed(Some(Callback::from_fn(move |s| {
                            crate::tui::close_repl_panel(s, id);
                        })));
                    }
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Esc) => {
                // If autocomplete is active, close it
                if self.ac_state.active {
                    self.ac_state.reset();
                    return EventResult::Consumed(None);
                }
                EventResult::Ignored
            }
            Event::Key(Key::Backspace) => {
                self.backspace();
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Del) => {
                self.delete();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Left) => {
                self.ac_state.reset();
                if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                } else if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.cursor.0 = self.line_char_count(self.cursor.1);
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
                self.ac_state.reset();
                if self.cursor.0 < self.line_char_count(self.cursor.1) {
                    self.cursor.0 += 1;
                } else if self.cursor.1 < self.current.input.len() - 1 {
                    self.cursor.1 += 1;
                    self.cursor.0 = 0;
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }

            Event::Key(Key::Up) => {
                // If autocomplete is active, navigate up
                if self.ac_state.active && self.ac_state.candidates.len() > 1 {
                    self.ac_state.select_prev();
                    return EventResult::Consumed(None);
                }
                
                // History navigation
                if self.cursor.1 == 0 {
                    // At top line - navigate history backend
                    let handled = if self.command_history.is_empty() {
                         false
                    } else if let Some(idx) = self.history_index {
                        if idx > 0 {
                            self.load_history(Some(idx - 1));
                            true
                        } else {
                            // Already at oldest
                            true
                        }
                    } else {
                        // Stash current and go to latest
                        self.stash_current = Some(self.current.input.clone());
                        self.load_history(Some(self.command_history.len() - 1));
                        true
                    };
                    
                    if handled {
                        return EventResult::Consumed(None);
                    }
                }

                // Otherwise normal cursor movement
                if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.fix_cursor_x();
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                // If autocomplete is active, navigate down
                if self.ac_state.active && self.ac_state.candidates.len() > 1 {
                    self.ac_state.select_next();
                    return EventResult::Consumed(None);
                }

                // History navigation
                let last_line = self.current.input.len().saturating_sub(1);
                if self.cursor.1 == last_line {
                    // At bottom line - navigate history forward
                    let handled = if let Some(idx) = self.history_index {
                        if idx < self.command_history.len() - 1 {
                            self.load_history(Some(idx + 1));
                            true
                        } else {
                            // Restore stashed
                            self.load_history(None);
                            true
                        }
                    } else {
                        false
                    };

                    if handled {
                        return EventResult::Consumed(None);
                    }
                }

                // Otherwise normal cursor movement
                if self.cursor.1 < self.current.input.len() - 1 {
                    self.cursor.1 += 1;
                    self.fix_cursor_x();
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Home) => {
                self.ac_state.reset();
                self.cursor.0 = 0;
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::End) => {
                self.ac_state.reset();
                self.cursor.0 = self.line_char_count(self.cursor.1);
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageUp) => {
                let page = self.last_size.y.saturating_sub(1);
                self.scroll_offset.1 = self.scroll_offset.1.saturating_sub(page);
                EventResult::Consumed(None)
            }
            Event::Key(Key::PageDown) => {
                let page = self.last_size.y.saturating_sub(1);
                let max_scroll = self.total_height().saturating_sub(self.last_size.y);
                self.scroll_offset.1 = (self.scroll_offset.1 + page).min(max_scroll);
                EventResult::Consumed(None)
            }
            // Alt+Enter for force newline
            Event::AltChar('\n') | Event::AltChar('\r') => {
                self.insert_newline();
                self.ensure_cursor_visible();
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
            Event::CtrlChar('s') => {
                self.save_history_to_disk();
                EventResult::Consumed(None)
            }
            _ => EventResult::Ignored,
        }
    }
}

/// Get current REPL content as text (for copy)
impl ReplPanel {
    pub fn get_content(&self) -> String {
        let mut result = String::new();

        for entry in &self.history {
            for (i, line) in entry.input.iter().enumerate() {
                let prompt = if i == 0 { ">>> " } else { "... " };
                result.push_str(prompt);
                result.push_str(line);
                result.push('\n');
            }
            if let Some(ref output) = entry.output {
                let text = match output {
                    ReplOutput::Success(s) | ReplOutput::Error(s) | ReplOutput::Definition(s) => s,
                };
                if !text.is_empty() {
                    result.push_str(text);
                    if !text.ends_with('\n') {
                        result.push('\n');
                    }
                }
            }
        }

        // Current input
        for (i, line) in self.current.input.iter().enumerate() {
            let prompt = if i == 0 { ">>> " } else { "... " };
            result.push_str(prompt);
            result.push_str(line);
            result.push('\n');
        }

        result
    }

    pub fn get_instance_id(&self) -> usize {
        self.instance_id
    }

    /// Get the history for preservation across rebuilds
    pub fn get_history(&self) -> Vec<ReplEntry> {
        self.history.clone()
    }

    /// Set the history (restore after rebuild)
    pub fn set_history(&mut self, history: Vec<ReplEntry>) {
        self.history = history;
    }

    /// Get the command history for preservation across rebuilds (used for arrow-key navigation)
    pub fn get_command_history(&self) -> Vec<Vec<String>> {
        self.command_history.clone()
    }

    /// Set the command history (restore after rebuild)
    pub fn set_command_history(&mut self, command_history: Vec<Vec<String>>) {
        self.command_history = command_history;
    }

    /// Take the eval state (handle, in_progress flag, and current input) for preservation across rebuilds.
    /// Returns None if no eval is in progress.
    pub fn take_eval_state(&mut self) -> Option<(nostos_vm::ThreadedEvalHandle, Vec<String>)> {
        if self.eval_in_progress {
            self.eval_in_progress = false;
            let handle = self.eval_handle.take();
            let input = self.current.input.clone();
            handle.map(|h| (h, input))
        } else {
            None
        }
    }

    /// Restore eval state after a rebuild.
    pub fn restore_eval_state(&mut self, handle: nostos_vm::ThreadedEvalHandle, input: Vec<String>) {
        self.eval_handle = Some(handle);
        self.eval_in_progress = true;
        self.current.input = input;
        self.current.output = Some(ReplOutput::Definition("Evaluating...".to_string()));
    }

    /// Load history entry by index (or restore stashed if None)
    fn load_history(&mut self, index: Option<usize>) {
        self.history_index = index;

        if let Some(idx) = index {
            if idx < self.command_history.len() {
                self.current.input = self.command_history[idx].clone();
            }
        } else {
            // Restore stashed or empty
            if let Some(stashed) = self.stash_current.take() {
                self.current.input = stashed;
            } else {
                self.current.input = vec![String::new()];
            }
        }

        // Move cursor to end
        self.cursor.1 = self.current.input.len().saturating_sub(1);
        self.cursor.0 = self.line_char_count(self.cursor.1);
        self.ensure_cursor_visible();
        
        // Reset autocomplete
        self.ac_state.reset();
    }

    fn get_history_file_path() -> PathBuf {
        PathBuf::from(env::var("HOME").unwrap_or_else(|_| ".".to_string()))
            .join(".nostos_history")
    }

    fn load_history_from_disk() -> Vec<Vec<String>> {
        let path = Self::get_history_file_path();
        if !path.exists() {
            return Vec::new();
        }

        if let Ok(content) = fs::read_to_string(path) {
            let mut history = Vec::new();
            let mut current_entry = Vec::new();

            for line in content.lines() {
                if line == "# --- ENTRY ---" {
                    if !current_entry.is_empty() {
                        history.push(current_entry);
                        current_entry = Vec::new();
                    }
                } else {
                    current_entry.push(line.to_string());
                }
            }
            if !current_entry.is_empty() {
                history.push(current_entry);
            }
            history
        } else {
            Vec::new()
        }
    }

    fn save_history_to_disk(&self) {
        let path = Self::get_history_file_path();
        
        if let Ok(mut file) = fs::File::create(path) {
            for entry in &self.command_history {
                if let Err(_) = writeln!(file, "# --- ENTRY ---") { break; }
                for line in entry {
                    if let Err(_) = writeln!(file, "{}", line) { break; }
                }
            }
        }
    }
}
