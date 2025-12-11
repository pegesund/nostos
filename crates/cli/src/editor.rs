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
    last_size: Vec2,
    engine: Option<Rc<RefCell<ReplEngine>>>, // For autocomplete
}

impl CodeEditor {
    pub fn new(text: String) -> Self {
        let content: Vec<String> = text.lines().map(String::from).collect();
        let content = if content.is_empty() { vec![String::new()] } else { content };
        Self {
            content,
            cursor: (0, 0),
            last_size: Vec2::zero(),
            engine: None,
        }
    }

    pub fn with_engine(mut self, engine: Rc<RefCell<ReplEngine>>) -> Self {
        self.engine = Some(engine);
        self
    }

    pub fn get_content(&self) -> String {
        self.content.join("\n")
    }

    fn fix_cursor_x(&mut self) {
        let line_len = self.content[self.cursor.1].len();
        if self.cursor.0 > line_len {
            self.cursor.0 = line_len;
        }
    }

    fn insert_char(&mut self, c: char) {
        let line = &mut self.content[self.cursor.1];
        if self.cursor.0 >= line.len() {
            line.push(c);
        } else {
            line.insert(self.cursor.0, c);
        }
        self.cursor.0 += 1;
    }

    fn insert_newline(&mut self) {
        let line = &mut self.content[self.cursor.1];
        let new_line_content = if self.cursor.0 < line.len() {
            line.split_off(self.cursor.0)
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
            line.remove(self.cursor.0 - 1);
            self.cursor.0 -= 1;
        } else if self.cursor.1 > 0 {
            let current_line = self.content.remove(self.cursor.1);
            self.cursor.1 -= 1;
            let prev_line = &mut self.content[self.cursor.1];
            self.cursor.0 = prev_line.len();
            prev_line.push_str(&current_line);
        }
    }

    fn delete(&mut self) {
        let line_len = self.content[self.cursor.1].len();
        if self.cursor.0 < line_len {
            let line = &mut self.content[self.cursor.1];
            line.remove(self.cursor.0);
        } else if self.cursor.1 < self.content.len() - 1 {
            let next_line = self.content.remove(self.cursor.1 + 1);
            let current_line = &mut self.content[self.cursor.1];
            current_line.push_str(&next_line);
        }
    }
}

impl View for CodeEditor {
    fn draw(&self, printer: &Printer) {
        // Draw content within the printer's viewport
        for (i, line) in self.content.iter().enumerate() {
            // Only draw lines that are likely visible? Printer handles clipping.
            // But iteration is cheap for small files.
            
            // Syntax highlighting
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
                    _ => Color::Rgb(255, 255, 255),
                };

                let style = Style::from(color);
                
                // Draw token content
                printer.with_style(style, |p| {
                    p.print((span.start, i), &line[span.clone()]);
                });
            }
        }

        // Draw cursor (simple block)
        // Cursor position is relative to content start (0,0) because ScrollView handles offset
        let cx = self.cursor.0;
        let cy = self.cursor.1;
        
        let cursor_style = Style::from(ColorStyle::new(Color::Rgb(0,0,0), Color::Rgb(0, 255, 0)));
        printer.with_style(cursor_style, |p| {
            let char_at_cursor = if self.cursor.1 < self.content.len() && self.cursor.0 < self.content[self.cursor.1].len() {
                self.content[self.cursor.1].chars().nth(self.cursor.0).unwrap_or(' ')
            } else {
                ' '
            };
            p.print((cx, cy), &char_at_cursor.to_string());
        });
    }

    fn layout(&mut self, size: Vec2) {
        self.last_size = size;
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        constraint
    }

    fn important_area(&self, _view_size: Vec2) -> Rect {
        // Return rect around cursor so ScrollView follows it
        Rect::from_point(Vec2::new(self.cursor.0, self.cursor.1))
    }

    fn take_focus(&mut self, _: Direction) -> Result<EventResult, CannotFocus> {
        Ok(EventResult::Consumed(None))
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        match event {
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
                    self.cursor.0 = self.content[self.cursor.1].len();
                }
                EventResult::Consumed(None)
            }
            Event::Key(Key::Right) => {
                let line_len = self.content[self.cursor.1].len();
                if self.cursor.0 < line_len {
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
            _ => EventResult::Ignored,
        }
    }
}
