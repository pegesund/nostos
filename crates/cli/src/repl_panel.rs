//! Interactive REPL panel with syntax highlighting
//!
//! Provides a notebook-style REPL where each input/output pair is displayed
//! in a scrollable view with syntax highlighting.

use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle, Style};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2, Rect};
use nostos_syntax::lexer::{Token, lex};
use std::rc::Rc;
use std::cell::RefCell;
use nostos_repl::ReplEngine;

/// A single REPL entry (input + output)
#[derive(Clone)]
struct ReplEntry {
    /// The input code
    input: Vec<String>,
    /// The output/result (None if not yet evaluated)
    output: Option<ReplOutput>,
}

/// Output from evaluating an entry
#[derive(Clone)]
enum ReplOutput {
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

    fn from_input(input: &str) -> Self {
        let lines: Vec<String> = input.lines().map(String::from).collect();
        Self {
            input: if lines.is_empty() { vec![String::new()] } else { lines },
            output: None,
        }
    }

    /// Total height in lines (input lines + output lines + separator)
    fn height(&self) -> usize {
        let input_height = self.input.len();
        let output_height = match &self.output {
            None => 0,
            Some(ReplOutput::Success(s)) | Some(ReplOutput::Error(s)) | Some(ReplOutput::Definition(s)) => {
                if s.is_empty() { 0 } else { s.lines().count().max(1) }
            }
        };
        // 1 line for prompt prefix on first input line
        input_height + output_height
    }
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
}

impl ReplPanel {
    pub fn new(engine: Rc<RefCell<ReplEngine>>, instance_id: usize) -> Self {
        Self {
            history: Vec::new(),
            current: ReplEntry::new(),
            cursor: (0, 0),
            scroll_offset: (0, 0),
            last_size: Vec2::zero(),
            engine,
            instance_id,
        }
    }

    /// Get the prompt string for a line
    fn prompt(&self, is_continuation: bool) -> &'static str {
        if is_continuation { "... " } else { ">>> " }
    }

    /// Evaluate the current input
    fn evaluate(&mut self) {
        let input_text = self.current.input.join("\n");
        if input_text.trim().is_empty() {
            return;
        }

        // Evaluate using the engine
        let result = self.engine.borrow_mut().eval(&input_text);

        self.current.output = Some(match result {
            Ok(output) => {
                if output.is_empty() || output.starts_with("Defined") || output.starts_with("Type") {
                    ReplOutput::Definition(output)
                } else {
                    ReplOutput::Success(output)
                }
            }
            Err(e) => ReplOutput::Error(e),
        });

        // Move current to history and start fresh
        self.history.push(self.current.clone());
        self.current = ReplEntry::new();
        self.cursor = (0, 0);

        // Scroll to bottom
        self.scroll_to_bottom();
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
    }

    fn insert_newline(&mut self) {
        let line = &mut self.current.input[self.cursor.1];
        let byte_idx = Self::char_to_byte_idx(line, self.cursor.0);
        let rest = line[byte_idx..].to_string();
        line.truncate(byte_idx);
        self.current.input.insert(self.cursor.1 + 1, rest);
        self.cursor.1 += 1;
        self.cursor.0 = 0;
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
        } else if self.cursor.1 > 0 {
            let current_line = self.current.input.remove(self.cursor.1);
            self.cursor.1 -= 1;
            self.cursor.0 = self.line_char_count(self.cursor.1);
            self.current.input[self.cursor.1].push_str(&current_line);
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
        } else if self.cursor.1 < self.current.input.len() - 1 {
            let next_line = self.current.input.remove(self.cursor.1 + 1);
            self.current.input[self.cursor.1].push_str(&next_line);
        }
    }

    /// Draw a line with syntax highlighting
    fn draw_highlighted_line(&self, printer: &Printer, line: &str, y: usize, x_offset: usize) {
        let scroll_x = self.scroll_offset.0;
        let view_width = printer.size.x.saturating_sub(x_offset);

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
        match event {
            Event::Char(c) => {
                self.insert_char(c);
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                // Shift+Enter or if line ends with \ or { -> insert newline
                // Otherwise evaluate
                let current_line = &self.current.input[self.cursor.1];
                let trimmed = current_line.trim_end();

                if trimmed.ends_with('\\') || trimmed.ends_with('{') || trimmed.ends_with("do") {
                    self.insert_newline();
                } else {
                    self.evaluate();
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
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
                if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.fix_cursor_x();
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                if self.cursor.1 < self.current.input.len() - 1 {
                    self.cursor.1 += 1;
                    self.fix_cursor_x();
                }
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::Home) => {
                self.cursor.0 = 0;
                self.ensure_cursor_visible();
                EventResult::Consumed(None)
            }
            Event::Key(Key::End) => {
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
}
