use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle, Style};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2, Rect};
use nostos_syntax::lexer::{Token, lex};
use std::rc::Rc;
use std::cell::RefCell;
use nostos_repl::ReplEngine;

use crate::autocomplete::{Autocomplete, CompletionContext, CompletionItem, CompletionSource, parse_imports, extract_module_from_editor_name};

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

/// Maximum items to show in autocomplete popup
const AC_MAX_VISIBLE: usize = 10;

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
    /// Module name being edited (e.g., "Math" when editing "Math.add")
    module_name: Option<String>,
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
            module_name: None,
        }
    }

    /// Set the function name being edited (e.g., "utils.bar" or "Math.add")
    /// This extracts the module context for autocomplete
    pub fn with_function_name(mut self, name: &str) -> Self {
        self.module_name = extract_module_from_editor_name(name);
        eprintln!("[Editor] with_function_name({:?}) -> module_name={:?}", name, self.module_name);
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
            let source = EngineCompletionSource { engine: &eng };
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

    fn insert_char(&mut self, c: char) {
        let line = &mut self.content[self.cursor.1];
        let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
        if byte_idx >= line.len() {
            line.push(c);
        } else {
            line.insert(byte_idx, c);
        }
        self.cursor.0 += 1;

        // Update autocomplete after each character
        self.update_autocomplete();
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
                eprintln!("[AC] No engine available");
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
        let eng = engine.borrow();
        let source = EngineCompletionSource { engine: &eng };

        // Debug: log available functions
        let funcs = source.get_functions();
        eprintln!("[AC] module_name={:?}, context={:?}, funcs_count={}, imports={:?}",
            self.module_name, context, funcs.len(), imports);
        if funcs.len() < 20 {
            eprintln!("[AC] functions: {:?}", funcs);
        }
        eprintln!("[AC] modules in autocomplete: {:?}", self.autocomplete.modules);

        let candidates = self.autocomplete.get_completions_with_context(
            &context,
            &source,
            self.module_name.as_deref(),
            &imports,
        );

        eprintln!("[AC] candidates: {} items", candidates.len());
        for c in candidates.iter().take(5) {
            eprintln!("[AC]   - {} ({:?})", c.text, c.kind);
        }

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
        let source = EngineCompletionSource { engine: &eng };

        // Get all functions from the current module
        let mut candidates: Vec<CompletionItem> = source.get_functions()
            .into_iter()
            .filter_map(|func| {
                // Check if function belongs to current module
                if let Some(dot_pos) = func.rfind('.') {
                    let func_module = &func[..dot_pos];
                    if func_module == module {
                        let short_name = func[dot_pos + 1..].split('/').next().unwrap_or(&func[dot_pos + 1..]);
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

        eprintln!("[Editor] Ctrl+F: module={}, found {} functions", module, candidates.len());

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

        // Position popup below cursor line
        let popup_y = cursor_screen_y + 1;
        let popup_x = cursor_screen_x;

        // Draw popup background
        let bg_style = Style::from(ColorStyle::new(
            Color::Rgb(200, 200, 200),
            Color::Rgb(40, 40, 60)
        ));

        // Draw visible items
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

        // Show status line with pagination info
        let status_y = popup_y + visible_count;
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

            // Syntax highlighting with scroll offset
            for (token, span) in lex(line) {
                let color = match token {
                    Token::Type | Token::Var | Token::If | Token::Then | Token::Else |
                    Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
                    Token::Use | Token::Private | Token::Pub | Token::SelfKw | Token::SelfType |
                    Token::Try | Token::Catch | Token::Finally | Token::Do |
                    Token::While | Token::For | Token::To | Token::Break | Token::Continue |
                    Token::Spawn | Token::SpawnLink | Token::SpawnMonitor | Token::Receive | Token::After |
                    Token::Panic | Token::Extern | Token::From | Token::Test | Token::Deriving | Token::Quote =>
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

                    Token::Underscore => Color::Rgb(100, 100, 100),
                    Token::Newline => Color::Rgb(255, 255, 255),
                    Token::Comment | Token::MultiLineComment => Color::Rgb(128, 128, 128),
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

        // Draw autocomplete popup
        self.draw_autocomplete(printer);
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
        let result = match event {
            // Tab: accept completion or cycle through candidates
            Event::Key(Key::Tab) => {
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    // If only one candidate, accept it
                    if self.ac_state.candidates.len() == 1 {
                        self.accept_completion();
                    } else {
                        // Cycle to next
                        self.ac_state.select_next();
                    }
                    return EventResult::Consumed(None);
                }
                // Regular tab inserts spaces
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
                self.insert_char(c);
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                // If autocomplete is active, accept the selection
                if self.ac_state.active && !self.ac_state.candidates.is_empty() {
                    self.accept_completion();
                    return EventResult::Consumed(None);
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
                EventResult::Ignored
            }
            Event::Key(Key::Backspace) => {
                self.backspace();
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
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
                self.ac_state.reset();
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
                // Otherwise normal cursor movement
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
                // Otherwise normal cursor movement
                if self.cursor.1 < self.content.len() - 1 {
                    self.cursor.1 += 1;
                    self.fix_cursor_x();
                }
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
                self.cursor.0 = 0;
                EventResult::Consumed(None)
            }
            Event::Key(Key::End) => {
                self.ac_state.reset();
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
            _ => EventResult::Ignored,
        };

        // After any consumed event, ensure cursor stays visible
        if matches!(result, EventResult::Consumed(_)) {
            self.ensure_cursor_visible();
        }

        result
    }
}
