use cursive::event::{Event, EventResult, Key};
use cursive::theme::{Color, ColorStyle, Style};
use cursive::view::{View, CannotFocus};
use cursive::direction::Direction;
use cursive::{Printer, Vec2, Rect};
use nostos_syntax::lexer::{Token, lex};
use std::rc::Rc;
use std::cell::RefCell;
use nostos_repl::ReplEngine;

pub struct CodeEditor {
    content: Vec<String>,
    cursor: (usize, usize), // col, row (0-indexed)
    scroll_offset: (usize, usize), // horizontal, vertical scroll offset
    last_size: Vec2,
    engine: Option<Rc<RefCell<ReplEngine>>>, // For autocomplete
    saved_content: String, // Content when last saved (for dirty checking)
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
        }
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
        } else if self.cursor.1 > 0 {
            let current_line = self.content.remove(self.cursor.1);
            self.cursor.1 -= 1;
            let prev_line = &mut self.content[self.cursor.1];
            self.cursor.0 = prev_line.chars().count();
            prev_line.push_str(&current_line);
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
        } else if self.cursor.1 < self.content.len() - 1 {
            let next_line = self.content.remove(self.cursor.1 + 1);
            let current_line = &mut self.content[self.cursor.1];
            current_line.push_str(&next_line);
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
            Event::Char(c) => {
                self.insert_char(c);
                EventResult::Consumed(None)
            }
            Event::Key(Key::Enter) => {
                self.insert_newline();
                EventResult::Consumed(None)
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
                if self.cursor.0 > 0 {
                    self.cursor.0 -= 1;
                } else if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.cursor.0 = self.line_char_count(self.cursor.1);
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
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
                if self.cursor.1 > 0 {
                    self.cursor.1 -= 1;
                    self.fix_cursor_x();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Down) => {
                if self.cursor.1 < self.content.len() - 1 {
                    self.cursor.1 += 1;
                    self.fix_cursor_x();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Home) => {
                self.cursor.0 = 0;
                EventResult::Consumed(None)
            }
            Event::Key(Key::End) => {
                self.cursor.0 = self.line_char_count(self.cursor.1);
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
