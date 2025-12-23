//! TUI implementation for Nostos

use cursive::Cursive;
use cursive::traits::*;
use cursive::views::{Dialog, EditView, LinearLayout, ScrollView, TextView, OnEventView, SelectView, Panel, BoxedView};
use cursive::theme::{Color, PaletteColor, Theme, BorderStyle, Style, ColorStyle};
use cursive::view::Resizable;
use cursive::utils::markup::StyledString;
use cursive::event::{Event, EventResult, EventTrigger, Key};
use nostos_repl::{ReplEngine, ReplConfig, BrowserItem, SaveCompileResult, CompileStatus, SearchResult, PanelInfo, PanelState};
use nostos_vm::PanelCommand;
use nostos_vm::{Value, Inspector, Slot, SlotInfo};
use nostos_syntax::lexer::{Token, lex};
use nostos_syntax::parse;
use std::cell::RefCell;
use std::rc::Rc;
use std::process::ExitCode;
use std::io::Write;

use crate::repl_panel::ReplPanel;
use crate::inspector_panel::InspectorPanel;
use crate::nostos_panel::NostosPanel;
use crate::debug_panel::{DebugPanel, DebugPanelCommand};

/// Debug logging - enable for troubleshooting
#[allow(unused)]
fn debug_log(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/nostos_tui_debug.log")
    {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{}] {}", timestamp, msg);
    }
}

/// Apply syntax highlighting to source code, returning a StyledString
fn syntax_highlight_code(source: &str) -> StyledString {
    use nostos_syntax::lexer::Token;

    let mut styled = StyledString::new();

    for line in source.lines() {
        let mut pos = 0;  // Track position to preserve whitespace
        for (token, span) in lex(line) {
            // Add any whitespace/characters between previous token and this one
            if span.start > pos {
                styled.append_plain(&line[pos..span.start]);
            }

            let color = match token {
                Token::Type | Token::Var | Token::If | Token::Then | Token::Else |
                Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
                Token::Use | Token::Private | Token::Pub | Token::SelfKw | Token::SelfType |
                Token::Try | Token::Catch | Token::Finally | Token::Do |
                Token::While | Token::For | Token::To | Token::Break | Token::Continue |
                Token::Spawn | Token::SpawnLink | Token::SpawnMonitor | Token::Receive | Token::After |
                Token::Panic | Token::Extern | Token::From | Token::Test | Token::Quote =>
                    Color::Rgb(255, 0, 255),  // Keywords: magenta

                Token::True | Token::False |
                Token::Int(_) | Token::HexInt(_) | Token::BinInt(_) |
                Token::Int8(_) | Token::Int16(_) | Token::Int32(_) |
                Token::UInt8(_) | Token::UInt16(_) | Token::UInt32(_) | Token::UInt64(_) |
                Token::BigInt(_) | Token::Float(_) | Token::Float32(_) | Token::Decimal(_) =>
                    Color::Rgb(255, 255, 0),  // Numbers/booleans: yellow

                Token::String(_) | Token::SingleQuoteString(_) | Token::Char(_) => Color::Rgb(0, 255, 0),  // Strings: green

                Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent | Token::StarStar |
                Token::EqEq | Token::NotEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq |
                Token::AndAnd | Token::OrOr | Token::Bang | Token::PlusPlus | Token::PipeRight |
                Token::Eq | Token::PlusEq | Token::MinusEq | Token::StarEq | Token::SlashEq |
                Token::LeftArrow | Token::RightArrow | Token::FatArrow | Token::Caret | Token::Dollar | Token::Question |
                Token::LParen | Token::RParen | Token::LBracket | Token::RBracket |
                Token::LBrace | Token::RBrace | Token::Comma | Token::Colon | Token::Dot |
                Token::Pipe | Token::Hash =>
                    Color::Rgb(255, 165, 0),  // Operators: orange

                Token::UpperIdent(_) => Color::Rgb(255, 255, 0),  // Types: yellow
                Token::LowerIdent(_) => Color::Rgb(255, 255, 255),  // Identifiers: white

                Token::Underscore => Color::Rgb(150, 150, 150), // Lighter gray - visible on dark backgrounds
                Token::Comment | Token::MultiLineComment => Color::Rgb(150, 150, 150), // Lighter gray
                _ => Color::Rgb(255, 255, 255),
            };

            let text = &line[span.start..span.end];
            styled.append_styled(text, Style::from(color));
            pos = span.end;
        }
        // Add any trailing content on the line
        if pos < line.len() {
            styled.append_plain(&line[pos..]);
        }
        styled.append_plain("\n");
    }

    styled
}

/// Format compile error for human-readable display
/// Cleans up token names and provides friendly messages
fn format_compile_error(error: &str) -> String {
    let mut result = error.to_string();

    // Replace common token patterns with friendly names
    let replacements = [
        // Parser tokens
        ("expected one of `", "expected "),
        ("`, `", "`, `"),
        ("` or `", "` or `"),
        ("found `EOF`", "found end of input"),
        ("found `Newline`", "found end of line"),
        ("Newline", "newline"),
        ("LParen", "'('"),
        ("RParen", "')'"),
        ("LBrace", "'{'"),
        ("RBrace", "'}'"),
        ("LBracket", "'['"),
        ("RBracket", "']'"),
        ("Comma", "','"),
        ("Colon", "':'"),
        ("Semicolon", "';'"),
        ("Dot", "'.'"),
        ("Arrow", "'->'"),
        ("FatArrow", "'=>'"),
        ("Eq", "'='"),
        ("EqEq", "'=='"),
        ("NotEq", "'!='"),
        ("Lt", "'<'"),
        ("Gt", "'>'"),
        ("LtEq", "'<='"),
        ("GtEq", "'>='"),
        ("Plus", "'+'"),
        ("Minus", "'-'"),
        ("Star", "'*'"),
        ("Slash", "'/'"),
        ("Percent", "'%'"),
        ("AndAnd", "'&&'"),
        ("OrOr", "'||'"),
        ("Bang", "'!'"),
        ("Pipe", "'|'"),
        ("Underscore", "'_'"),
        ("EOF", "end of input"),
    ];

    for (from, to) in replacements {
        result = result.replace(from, to);
    }

    // If error is still very cryptic, provide a generic message
    if result.contains("Token") || result.contains("Parse") && result.len() > 200 {
        // Try to extract just the line info
        if let Some(line_info) = extract_line_info(&result) {
            return format!("Syntax error at {}", line_info);
        }
        return "Syntax error in source code".to_string();
    }

    // Limit length and clean up
    if result.len() > 300 {
        result = result.chars().take(300).collect::<String>() + "...";
    }

    result
}

/// Extract line/column info from error message if present
fn extract_line_info(error: &str) -> Option<String> {
    // Look for patterns like "line 5", "at line 5", "5:10", etc.
    let error_lower = error.to_lowercase();

    // Try to find "line X" pattern
    if let Some(line_pos) = error_lower.find("line ") {
        let after_line = &error[line_pos + 5..];
        let line_num: String = after_line.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !line_num.is_empty() {
            // Check for column after
            let after_num = &after_line[line_num.len()..];
            if let Some(col_start) = after_num.to_lowercase().find("col") {
                let after_col = &after_num[col_start..];
                // Skip "col" or "column"
                let num_start = after_col.find(|c: char| c.is_ascii_digit());
                if let Some(start) = num_start {
                    let col_num: String = after_col[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
                    if !col_num.is_empty() {
                        return Some(format!("line {}, column {}", line_num, col_num));
                    }
                }
            }
            return Some(format!("line {}", line_num));
        }
    }

    // Try "X:Y" pattern (line:column) - look for digit:digit
    for (i, c) in error.chars().enumerate() {
        if c == ':' && i > 0 {
            // Check if there are digits before and after
            let before: String = error[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect::<String>().chars().rev().collect();
            let after: String = error[i+1..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if !before.is_empty() && !after.is_empty() {
                return Some(format!("line {}, column {}", before, after));
            }
        }
    }

    None
}

/// Extract definition names from source code
fn extract_definition_names(source: &str) -> Vec<String> {
    use nostos_syntax::ast::Item;

    // Strip together directives before parsing
    let code_only: String = source
        .lines()
        .filter(|line| !line.trim().starts_with("# together "))
        .collect::<Vec<_>>()
        .join("\n");

    let (parsed, _errors) = parse(&code_only);
    let mut names = Vec::new();

    if let Some(module) = parsed {
        for item in &module.items {
            match item {
                Item::FnDef(fn_def) => names.push(fn_def.name.node.clone()),
                Item::TypeDef(type_def) => names.push(type_def.name.node.clone()),
                Item::TraitDef(trait_def) => names.push(trait_def.name.node.clone()),
                Item::Binding(binding) => {
                    if let nostos_syntax::ast::Pattern::Var(ident) = &binding.pattern {
                        names.push(ident.node.clone());
                    }
                }
                _ => {}
            }
        }
    }

    names
}

use crate::editor::CodeEditor;
use crate::custom_views::{ActiveWindow, FocusableConsole};

mod nomouse_backend {
    //! Custom crossterm backend without mouse capture enabled.
    //! This allows normal terminal text selection and copy/paste to work.

    use crossterm::{
        cursor,
        event::{
            DisableBracketedPaste, EnableBracketedPaste,
            Event as CEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
            MouseButton, MouseEvent as CMouseEvent, MouseEventKind,
        },
        execute, queue,
        style::{Attribute, Color, Print, SetAttribute, SetBackgroundColor, SetForegroundColor},
        terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use cursive::backend::Backend;
    use cursive::event::{Event, Key, MouseButton as CursiveMouseButton, MouseEvent};
    use cursive::theme;
    use cursive::Vec2;
    use std::io::{self, BufWriter, Stdout, Write};

    pub struct NoMouseBackend {
        writer: BufWriter<Stdout>,
        #[allow(dead_code)]
        current_style: theme::ColorPair,
    }

    impl NoMouseBackend {
        pub fn init() -> io::Result<Box<dyn Backend>> {
            terminal::enable_raw_mode()?;
            let mut stdout = io::stdout();

            // Note: We intentionally skip EnableMouseCapture here
            execute!(
                stdout,
                EnterAlternateScreen,
                EnableBracketedPaste,
                cursor::Hide
            )?;

            Ok(Box::new(Self {
                writer: BufWriter::new(stdout),
                current_style: theme::ColorPair {
                    front: theme::Color::TerminalDefault,
                    back: theme::Color::TerminalDefault,
                },
            }))
        }
    }

    impl Drop for NoMouseBackend {
        fn drop(&mut self) {
            let _ = self.writer.flush();
            let _ = execute!(
                io::stdout(),
                LeaveAlternateScreen,
                DisableBracketedPaste,
                cursor::Show
            );
            let _ = terminal::disable_raw_mode();
        }
    }

    impl Backend for NoMouseBackend {
        fn poll_event(&mut self) -> Option<Event> {
            self.writer.flush().ok();

            match crossterm::event::poll(std::time::Duration::from_millis(10)) {
                Ok(true) => match crossterm::event::read() {
                    Ok(CEvent::Key(key_event)) => translate_key_event(key_event),
                    Ok(CEvent::Mouse(mouse_event)) => translate_mouse_event(mouse_event),
                    Ok(CEvent::Resize(_, _)) => Some(Event::WindowResize),
                    Ok(CEvent::Paste(_text)) => None, // Paste events not supported in this cursive version
                    _ => None,
                },
                _ => None,
            }
        }

        fn set_title(&mut self, title: String) {
            let _ = execute!(self.writer, crossterm::terminal::SetTitle(title));
        }

        fn refresh(&mut self) {
            let _ = self.writer.flush();
        }

        fn has_colors(&self) -> bool {
            true
        }

        fn screen_size(&self) -> Vec2 {
            terminal::size().map(|(w, h)| Vec2::new(w as usize, h as usize)).unwrap_or(Vec2::new(80, 24))
        }

        fn print_at(&self, pos: Vec2, text: &str) {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let _ = queue!(
                handle,
                cursor::MoveTo(pos.x as u16, pos.y as u16),
                Print(text)
            );
        }

        fn print_at_rep(&self, pos: Vec2, repetitions: usize, text: &str) {
            if repetitions > 0 {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                let _ = queue!(handle, cursor::MoveTo(pos.x as u16, pos.y as u16));
                for _ in 0..repetitions {
                    let _ = queue!(handle, Print(text));
                }
            }
        }

        fn clear(&self, color: theme::Color) {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let _ = queue!(
                handle,
                SetBackgroundColor(translate_color(color)),
                terminal::Clear(terminal::ClearType::All)
            );
        }

        fn set_color(&self, color_pair: theme::ColorPair) -> theme::ColorPair {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let _ = queue!(
                handle,
                SetForegroundColor(translate_color(color_pair.front)),
                SetBackgroundColor(translate_color(color_pair.back))
            );
            color_pair
        }

        fn set_effect(&self, effect: theme::Effect) {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let attr = match effect {
                theme::Effect::Bold => Attribute::Bold,
                theme::Effect::Italic => Attribute::Italic,
                theme::Effect::Strikethrough => Attribute::CrossedOut,
                theme::Effect::Underline => Attribute::Underlined,
                theme::Effect::Reverse => Attribute::Reverse,
                theme::Effect::Blink => Attribute::SlowBlink,
                theme::Effect::Dim => Attribute::Dim,
                theme::Effect::Simple => Attribute::Reset,
            };
            let _ = queue!(handle, SetAttribute(attr));
        }

        fn unset_effect(&self, effect: theme::Effect) {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let attr = match effect {
                theme::Effect::Bold => Attribute::NormalIntensity,
                theme::Effect::Italic => Attribute::NoItalic,
                theme::Effect::Strikethrough => Attribute::NotCrossedOut,
                theme::Effect::Underline => Attribute::NoUnderline,
                theme::Effect::Reverse => Attribute::NoReverse,
                theme::Effect::Blink => Attribute::NoBlink,
                theme::Effect::Dim => Attribute::NormalIntensity,
                theme::Effect::Simple => Attribute::Reset,
            };
            let _ = queue!(handle, SetAttribute(attr));
        }

        fn name(&self) -> &str {
            "crossterm-nomouse"
        }
    }

    fn translate_color(color: theme::Color) -> Color {
        match color {
            theme::Color::TerminalDefault => Color::Reset,
            theme::Color::Dark(theme::BaseColor::Black) => Color::Black,
            theme::Color::Dark(theme::BaseColor::Red) => Color::DarkRed,
            theme::Color::Dark(theme::BaseColor::Green) => Color::DarkGreen,
            theme::Color::Dark(theme::BaseColor::Yellow) => Color::DarkYellow,
            theme::Color::Dark(theme::BaseColor::Blue) => Color::DarkBlue,
            theme::Color::Dark(theme::BaseColor::Magenta) => Color::DarkMagenta,
            theme::Color::Dark(theme::BaseColor::Cyan) => Color::DarkCyan,
            theme::Color::Dark(theme::BaseColor::White) => Color::Grey,
            theme::Color::Light(theme::BaseColor::Black) => Color::DarkGrey,
            theme::Color::Light(theme::BaseColor::Red) => Color::Red,
            theme::Color::Light(theme::BaseColor::Green) => Color::Green,
            theme::Color::Light(theme::BaseColor::Yellow) => Color::Yellow,
            theme::Color::Light(theme::BaseColor::Blue) => Color::Blue,
            theme::Color::Light(theme::BaseColor::Magenta) => Color::Magenta,
            theme::Color::Light(theme::BaseColor::Cyan) => Color::Cyan,
            theme::Color::Light(theme::BaseColor::White) => Color::White,
            theme::Color::Rgb(r, g, b) => Color::Rgb { r, g, b },
            theme::Color::RgbLowRes(r, g, b) => Color::AnsiValue(16 + 36 * r + 6 * g + b),
        }
    }

    fn translate_key_event(event: KeyEvent) -> Option<Event> {
        // Only process key press events (not release or repeat for most keys)
        if event.kind != KeyEventKind::Press {
            return None;
        }

        let mods = event.modifiers;
        let shift = mods.contains(KeyModifiers::SHIFT);
        let ctrl = mods.contains(KeyModifiers::CONTROL);
        let alt = mods.contains(KeyModifiers::ALT);

        match event.code {
            KeyCode::Char(c) if ctrl => Some(Event::CtrlChar(c)),
            KeyCode::Char(c) if alt => Some(Event::AltChar(c)),
            KeyCode::Char(c) => Some(Event::Char(c)),
            KeyCode::Enter => Some(Event::Key(Key::Enter)),
            KeyCode::Backspace => Some(Event::Key(Key::Backspace)),
            KeyCode::Tab if shift => Some(Event::Shift(Key::Tab)),
            KeyCode::Tab => Some(Event::Key(Key::Tab)),
            KeyCode::Esc => Some(Event::Key(Key::Esc)),
            KeyCode::Left => Some(Event::Key(Key::Left)),
            KeyCode::Right => Some(Event::Key(Key::Right)),
            KeyCode::Up => Some(Event::Key(Key::Up)),
            KeyCode::Down => Some(Event::Key(Key::Down)),
            KeyCode::Home => Some(Event::Key(Key::Home)),
            KeyCode::End => Some(Event::Key(Key::End)),
            KeyCode::PageUp => Some(Event::Key(Key::PageUp)),
            KeyCode::PageDown => Some(Event::Key(Key::PageDown)),
            KeyCode::Delete => Some(Event::Key(Key::Del)),
            KeyCode::Insert => Some(Event::Key(Key::Ins)),
            KeyCode::F(n) => Some(Event::Key(Key::from_f(n))),
            _ => None,
        }
    }

    fn translate_mouse_event(event: CMouseEvent) -> Option<Event> {
        let pos = cursive::Vec2::new(event.column as usize, event.row as usize);
        let event = match event.kind {
            MouseEventKind::Down(MouseButton::Left) => MouseEvent::Press(CursiveMouseButton::Left),
            MouseEventKind::Down(MouseButton::Right) => MouseEvent::Press(CursiveMouseButton::Right),
            MouseEventKind::Down(MouseButton::Middle) => MouseEvent::Press(CursiveMouseButton::Middle),
            MouseEventKind::Up(MouseButton::Left) => MouseEvent::Release(CursiveMouseButton::Left),
            MouseEventKind::Up(MouseButton::Right) => MouseEvent::Release(CursiveMouseButton::Right),
            MouseEventKind::Up(MouseButton::Middle) => MouseEvent::Release(CursiveMouseButton::Middle),
            MouseEventKind::ScrollUp => MouseEvent::WheelUp,
            MouseEventKind::ScrollDown => MouseEvent::WheelDown,
            _ => return None,
        };
        Some(Event::Mouse { offset: cursive::Vec2::zero(), position: pos, event })
    }
}

struct TuiState {
    open_editors: Vec<String>,
    open_repls: Vec<usize>,  // REPL instance IDs
    next_repl_id: usize,
    active_window_idx: usize,
    engine: Rc<RefCell<ReplEngine>>,
    inspector_open: bool,
    console_open: bool,
    nostos_panel_open: bool,
    debug_panel_open: bool,
    /// Currently open panel info (if nostos_panel_open is true) - old API
    current_panel: Option<PanelInfo>,
    /// Currently open panel ID (if nostos_panel_open is true) - new API
    current_panel_id: Option<u64>,
}

pub fn run_tui(args: &[String]) -> ExitCode {
    // Use default cursive backend
    let mut siv = cursive::default();

    // Custom theme
    let mut theme = Theme::default();
    theme.borders = BorderStyle::Simple;
    theme.palette[PaletteColor::Background] = Color::TerminalDefault;
    theme.palette[PaletteColor::View] = Color::TerminalDefault;
    theme.palette[PaletteColor::Primary] = Color::Rgb(0, 255, 0); // Green text
    theme.palette[PaletteColor::TitlePrimary] = Color::Rgb(255, 255, 255); // White titles
    theme.palette[PaletteColor::Highlight] = Color::Rgb(60, 60, 100); // Dark blue highlight background
    theme.palette[PaletteColor::HighlightText] = Color::Rgb(255, 255, 255); // White text on highlight
    theme.palette[PaletteColor::Secondary] = Color::TerminalDefault;
    siv.set_theme(theme);

    // Initialize REPL engine
    let config = ReplConfig::default();
    let mut engine = ReplEngine::new(config);
    if let Err(e) = engine.load_stdlib() {
        eprintln!("Failed to load stdlib: {}", e);
    }

    // Load files or directories
    for arg in args {
        if !arg.starts_with('-') {
            let path = std::path::Path::new(arg);
            if path.is_dir() {
                // Load directory with SourceManager
                if let Err(e) = engine.load_directory(arg) {
                    eprintln!("Error loading directory {}: {}", arg, e);
                }
            } else {
                // Load single file
                if let Err(e) = engine.load_file(arg) {
                    eprintln!("Error loading {}: {}", arg, e);
                }
            }
        }
    }

    let engine = Rc::new(RefCell::new(engine));

    // Initialize State
    siv.set_user_data(Rc::new(RefCell::new(TuiState {
        open_editors: Vec::new(),
        open_repls: Vec::new(),
        next_repl_id: 1,
        active_window_idx: 0,
        engine: engine.clone(),
        inspector_open: false,
        console_open: true,
        nostos_panel_open: false,
        debug_panel_open: false,
        current_panel: None,
        current_panel_id: None,
    })));

    // Components

    // 1. REPL Log (Console) - will be added to workspace
    // Wrap in FocusableConsole so it can receive focus and show yellow border
    let repl_log = FocusableConsole::new(
        TextView::new(format!(
            "Nostos TUI v{}\nType :help for commands\n\n",
            env!("CARGO_PKG_VERSION")
        ))
        .scrollable()
        .scroll_x(true)
    ).with_name("repl_log");

    // Wrap console with OnEventView for Ctrl+Y copy (same pattern as Ctrl+S on editor)
    let repl_log_with_events = OnEventView::new(repl_log)
        .on_event(Event::CtrlChar('y'), |s| {
            if let Some(text) = s.call_on_name("repl_log", |view: &mut FocusableConsole| {
                view.get_content()
            }) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        });

    // 2. Workspace - LinearLayout for equal window distribution
    // Console is smaller - use max_width to limit size
    let console_view = ActiveWindow::new(repl_log_with_events, "Console").max_width(60);

    let layout = LinearLayout::horizontal()
        .child(console_view)
        .full_width()
        .full_height();

    siv.add_fullscreen_layer(layout);

    // Focus repl_log at start
    siv.focus_name("repl_log").ok();

    // Global Shift+Tab for window cycling (this one needs to be global)
    siv.set_on_pre_event(Event::Shift(Key::Tab), |s| {
        cycle_window(s);
    });

    // Ctrl+Left/Right for window navigation
    siv.set_on_pre_event(Event::Ctrl(Key::Left), |s| {
        cycle_window_backward(s);
    });
    siv.set_on_pre_event(Event::Ctrl(Key::Right), |s| {
        cycle_window(s);
    });

    // Global Ctrl+B to open browser from anywhere
    siv.set_on_pre_event(Event::CtrlChar('b'), |s| {
        open_browser(s);
    });

    // Global Ctrl+R to open new REPL panel
    siv.set_on_pre_event(Event::CtrlChar('r'), |s| {
        open_repl_panel(s);
    });

    // Global Alt+I to toggle inspector panel (Ctrl+I conflicts with Tab in terminals)
    siv.set_on_pre_event(Event::AltChar('i'), |s| {
        toggle_inspector(s);
    });

    // Also try F9 as backup
    siv.set_on_pre_event(Event::Key(Key::F9), |s| {
        toggle_inspector(s);
    });

    // Global Alt+C to toggle console
    siv.set_on_pre_event(Event::AltChar('c'), |s| {
        toggle_console(s);
    });

    // Global Ctrl+H to show help
    siv.set_on_pre_event(Event::CtrlChar('h'), |s| {
        show_help_dialog(s);
    });

    // Global Ctrl+D to toggle debug panel
    siv.set_on_pre_event(Event::CtrlChar('d'), |s| {
        toggle_debug_panel(s);
    });

    // Debug commands - F-keys only (letter keys handled via REPL :c/:n/:s/:o commands)
    // F5 - Continue
    siv.set_on_pre_event(Event::Key(Key::F5), |s| {
        send_debug_command(s, nostos_vm::shared_types::DebugCommand::Continue);
    });

    // F10 - Step Over
    siv.set_on_pre_event(Event::Key(Key::F10), |s| {
        send_debug_command(s, nostos_vm::shared_types::DebugCommand::StepOver);
    });

    // F11 - Step In
    siv.set_on_pre_event(Event::Key(Key::F11), |s| {
        send_debug_command(s, nostos_vm::shared_types::DebugCommand::StepLine);
    });

    // Dynamic global keybindings - panels register via Panel.registerHotkey() from Nostos code
    // When Alt+<letter> is pressed, check if any hotkey callback is registered
    // Skip letters that have dedicated handlers (i=inspector, c=console)
    for c in 'a'..='z' {
        if c == 'i' || c == 'c' {
            continue;  // These have dedicated handlers above
        }
        let engine_for_key = engine.clone();
        let key_char = c;
        siv.set_on_pre_event(Event::AltChar(c), move |s| {
            let key_str = format!("alt+{}", key_char);

            // First, drain any pending panel commands to register hotkeys
            engine_for_key.borrow_mut().drain_panel_commands();

            // Check for new API hotkey callback
            let callback = engine_for_key.borrow().get_hotkey_callback(&key_str).cloned();
            if let Some(callback_fn) = callback {
                // Call the callback function
                let call_expr = format!("{}()", callback_fn);
                match engine_for_key.borrow_mut().eval(&call_expr) {
                    Ok(_) => {}
                    Err(e) => {
                        log_to_repl(s, &format!("Hotkey error: {}", e));
                    }
                }

                // Process any panel commands generated by the callback (like Panel.show)
                process_panel_commands(s, engine_for_key.clone());
                return;
            }

            // Fall back to old API: check if a panel is registered for this key
            let panel_info = engine_for_key.borrow().get_panel_for_key(&key_str).cloned();

            if let Some(info) = panel_info {
                open_registered_panel(s, engine_for_key.clone(), info);
            }
        });
    }

    // Set up auto-refresh for async evaluation polling
    // Refresh at 30 FPS to poll for eval results
    siv.set_autorefresh(true);
    siv.set_fps(30);

    // Handle refresh events to poll for async eval results and debug events
    siv.set_on_pre_event(Event::Refresh, move |s| {
        poll_repl_evals(s);
        poll_debug_events(s);
        poll_debug_panel_commands(s);
        // Poll for println output and show in REPL panels during debugging
        poll_debug_output(s);
    });

    siv.run();
    ExitCode::SUCCESS
}

fn log_to_repl(s: &mut Cursive, text: &str) {
    s.call_on_name("repl_log", |view: &mut FocusableConsole| {
        view.append(&format!("{}\n", text));
    });
}

/// Poll for output (println) from any VM process and log to console.
#[allow(dead_code)]
fn poll_output(s: &mut Cursive, engine: &Rc<RefCell<ReplEngine>>) {
    let output = engine.borrow().drain_output();
    for line in output {
        log_to_repl(s, &line);
    }
}

/// Poll for println output during debugging and show in REPL panel.
/// Only drains output if there's an active debug session to avoid stealing
/// output from regular evals that will drain it in finish_eval_with_result.
fn poll_debug_output(s: &mut Cursive) {
    // Only drain output during debugging - regular evals drain in finish_eval_with_result
    let debug_panel_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().debug_panel_open
    }).unwrap_or(false);

    if !debug_panel_open {
        return;
    }

    // Get engine to drain output
    let engine = match s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().engine.clone()
    }) {
        Some(e) => e,
        None => return,
    };

    let output = engine.borrow().drain_output();
    if output.is_empty() {
        return;
    }

    debug_log(&format!("poll_debug_output: got {} lines", output.len()));

    // Send to all open REPLs (same pattern as poll_repl_evals which works)
    let repl_ids: Vec<usize> = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().open_repls.clone()
    }).unwrap_or_default();

    for repl_id in repl_ids {
        let panel_id = format!("repl_panel_{}", repl_id);
        let output_clone = output.clone();
        s.call_on_name(&panel_id, |panel: &mut ReplPanel| {
            debug_log(&format!("poll_debug_output: appending to repl {}", repl_id));
            panel.append_debug_output(&output_clone);
        });
    }
}

/// Poll for async evaluation results from all REPL panels.
fn poll_repl_evals(s: &mut Cursive) {
    let repl_ids: Vec<usize> = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().open_repls.clone()
    }).unwrap_or_default();

    for repl_id in repl_ids {
        let panel_id = format!("repl_panel_{}", repl_id);
        s.call_on_name(&panel_id, |panel: &mut ReplPanel| {
            panel.poll_eval_result();
        });
    }
}

/// Poll for inspect entries and update the inspector panel if open.
fn poll_inspect_entries(s: &mut Cursive, engine: &Rc<RefCell<ReplEngine>>) {
    let entries = engine.borrow().drain_inspect_entries();
    if entries.is_empty() {
        return;
    }

    // Check if inspector is open
    let inspector_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().inspector_open
    }).unwrap_or(false);

    if inspector_open {
        // Update the existing panel
        s.call_on_name("inspector_panel", |panel: &mut InspectorPanel| {
            panel.process_entries(entries);
        });
    }
    // If inspector is closed, entries are simply discarded
    // (they were already drained from the queue)
}

/// Poll for debug panel commands (from pending_command field set by key events).
fn poll_debug_panel_commands(s: &mut Cursive) {
    // Check if debug panel has a pending command
    let cmd = s.call_on_name("debug_panel", |panel: &mut DebugPanel| {
        panel.take_pending_command()
    }).flatten();

    if let Some(cmd) = cmd {
        debug_log(&format!("poll_debug_panel_commands: got command {:?}", cmd));
        let debug_cmd = match cmd {
            DebugPanelCommand::Continue => nostos_vm::shared_types::DebugCommand::Continue,
            DebugPanelCommand::StepOver => nostos_vm::shared_types::DebugCommand::StepOver,
            DebugPanelCommand::StepIn => nostos_vm::shared_types::DebugCommand::StepLine,
            DebugPanelCommand::StepOut => nostos_vm::shared_types::DebugCommand::StepOut,
            DebugPanelCommand::Stop => {
                // For now, just continue to let it finish
                nostos_vm::shared_types::DebugCommand::Continue
            }
        };
        send_debug_command(s, debug_cmd);
    }

    // Check if debug panel needs locals for a specific frame
    let frame_request = s.call_on_name("debug_panel", |panel: &mut DebugPanel| {
        panel.take_pending_locals_request()
    }).flatten();

    if let Some(frame_index) = frame_request {
        debug_log(&format!("poll_debug_panel_commands: requesting locals for frame {}", frame_index));
        send_locals_for_frame_command(s, frame_index);
    }
}

/// Send a request for locals for a specific frame.
fn send_locals_for_frame_command(s: &mut Cursive, frame_index: usize) {
    let repl_ids: Vec<usize> = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().open_repls.clone()
    }).unwrap_or_default();

    for repl_id in repl_ids {
        let panel_id = format!("repl_panel_{}", repl_id);
        let sent = s.call_on_name(&panel_id, |panel: &mut ReplPanel| {
            if let Some(session) = panel.get_debug_session() {
                let _ = session.send(nostos_vm::shared_types::DebugCommand::PrintLocalsForFrame(frame_index));
                true
            } else {
                false
            }
        }).unwrap_or(false);

        if sent {
            return;
        }
    }
}

/// Send a debug command to the active debug session (if any).
fn send_debug_command(s: &mut Cursive, command: nostos_vm::shared_types::DebugCommand) {
    debug_log(&format!("send_debug_command: {:?}", command));
    let repl_ids: Vec<usize> = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().open_repls.clone()
    }).unwrap_or_default();

    // Find the first REPL panel with an active debug session and send the command
    for repl_id in repl_ids {
        let panel_id = format!("repl_panel_{}", repl_id);
        let sent = s.call_on_name(&panel_id, |panel: &mut ReplPanel| {
            if let Some(session) = panel.get_debug_session() {
                debug_log(&format!("send_debug_command: found session, sending {:?}", command));
                let _ = session.send(command.clone());
                // Also request stack and locals after the command
                let _ = session.send(nostos_vm::shared_types::DebugCommand::PrintStack);
                let _ = session.send(nostos_vm::shared_types::DebugCommand::PrintLocals);
                true
            } else {
                debug_log(&format!("send_debug_command: no session for repl {}", repl_id));
                false
            }
        }).unwrap_or(false);

        if sent {
            return;
        }
    }
    debug_log("send_debug_command: no active debug session found");
}

/// Poll for debug events from REPL panels and update the debug panel.
fn poll_debug_events(s: &mut Cursive) {
    use crate::debug_panel::{DebugPanel, DebugState};
    use nostos_vm::shared_types::DebugEvent;

    let repl_ids: Vec<usize> = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().open_repls.clone()
    }).unwrap_or_default();

    let mut events: Vec<DebugEvent> = Vec::new();

    // Collect debug events from all REPL panels
    for repl_id in repl_ids {
        let panel_id = format!("repl_panel_{}", repl_id);
        s.call_on_name(&panel_id, |panel: &mut ReplPanel| {
            while let Some(event) = panel.poll_debug_event() {
                events.push(event);
            }
        });
    }

    if events.is_empty() {
        return;
    }

    // Open the debug panel FIRST if not already open (so events can be processed)
    let debug_panel_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().debug_panel_open
    }).unwrap_or(false);

    if !debug_panel_open {
        // Auto-open debug panel when debugging starts
        s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
            state.borrow_mut().debug_panel_open = true;
        });
        rebuild_workspace(s);
        s.focus_name("debug_panel").ok();
    }

    // NOW update debug panel with events (panel exists now)
    for event in &events {
        debug_log(&format!("poll_debug_events: processing {:?}", event));
    }
    for event in events {
        s.call_on_name("debug_panel", |panel: &mut DebugPanel| {
            match event {
                DebugEvent::Paused { function, file, line, source, source_start_line, .. } => {
                    debug_log(&format!("poll_debug_events: Paused in {} at line {}, source={:?}, source_start_line={}", function, line, source, source_start_line));
                    panel.on_paused(function, file, line, source, source_start_line);
                }
                DebugEvent::BreakpointHit { function, file, line, .. } => {
                    debug_log(&format!("poll_debug_events: BreakpointHit in {} at line {}", function, line));
                    // BreakpointHit doesn't have source, pass None and default start line
                    panel.on_paused(function, file, line, None, 1);
                }
                DebugEvent::Exited { value, .. } => {
                    debug_log(&format!("poll_debug_events: Exited with {:?}", value));
                    panel.on_finished(value);
                }
                DebugEvent::Stack { frames } => {
                    debug_log(&format!("poll_debug_events: Stack with {} frames", frames.len()));
                    panel.set_stack(frames);
                }
                DebugEvent::Locals { variables } => {
                    debug_log(&format!("poll_debug_events: Locals with {} variables", variables.len()));
                    panel.set_locals(variables);
                }
                DebugEvent::LocalsForFrame { frame_index, variables } => {
                    debug_log(&format!("poll_debug_events: LocalsForFrame {} with {} variables", frame_index, variables.len()));
                    panel.set_locals_for_frame(frame_index, variables);
                }
                _ => {}
            }
        });
    }
}

/// Rebuild the workspace layout based on current windows
/// Uses LinearLayout for equal window distribution
/// Navigation: Ctrl+Left/Right to move between windows
fn rebuild_workspace(s: &mut Cursive) {
    let (editor_names, repl_ids, engine, inspector_open, console_open, nostos_panel_open, debug_panel_open, current_panel, current_panel_id) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let state = state.borrow();
        (state.open_editors.clone(), state.open_repls.clone(), state.engine.clone(), state.inspector_open, state.console_open, state.nostos_panel_open, state.debug_panel_open, state.current_panel.clone(), state.current_panel_id)
    }).unwrap();

    // Get console content BEFORE clearing workspace
    let repl_log_content: String = s.call_on_name("repl_log", |view: &mut FocusableConsole| {
        view.get_content()
    }).unwrap_or_default();

    // Preserve REPL panel state before clearing (histories, eval handles, and debug sessions)
    let mut repl_histories: std::collections::HashMap<usize, (Vec<crate::repl_panel::ReplEntry>, Vec<Vec<String>>)> = std::collections::HashMap::new();
    let mut repl_eval_handles: std::collections::HashMap<usize, (nostos_vm::ThreadedEvalHandle, Vec<String>)> = std::collections::HashMap::new();
    let mut repl_debug_sessions: std::collections::HashMap<usize, (nostos_vm::DebugSession, Vec<String>)> = std::collections::HashMap::new();
    for &repl_id in &repl_ids {
        let panel_id = format!("repl_panel_{}", repl_id);
        if let Some((history, cmd_history, eval_state, debug_state)) = s.call_on_name(&panel_id, |panel: &mut ReplPanel| {
            let h = panel.get_history();
            let ch = panel.get_command_history();
            let es = panel.take_eval_state();
            let ds = panel.take_debug_state();
            (h, ch, es, ds)
        }) {
            repl_histories.insert(repl_id, (history, cmd_history));
            if let Some((handle, input)) = eval_state {
                repl_eval_handles.insert(repl_id, (handle, input));
            }
            if let Some((session, input)) = debug_state {
                repl_debug_sessions.insert(repl_id, (session, input));
            }
        }
    }

    // Preserve debug panel state before clearing
    let debug_panel_state: Option<(crate::debug_panel::DebugState, Vec<nostos_vm::shared_types::StackFrame>, std::collections::HashMap<usize, Vec<(String, String, String)>>)> =
        if debug_panel_open {
            s.call_on_name("debug_panel", |panel: &mut DebugPanel| {
                panel.take_state()
            })
        } else {
            None
        };

    // Remove old layer and create fresh one
    s.pop_layer();

    // Create console view
    let repl_log = FocusableConsole::new(
        TextView::new(repl_log_content).scrollable().scroll_x(true)
    ).with_name("repl_log");

    let repl_log_with_events = OnEventView::new(repl_log)
        .on_event(Event::CtrlChar('y'), |s| {
            if let Some(text) = s.call_on_name("repl_log", |view: &mut FocusableConsole| {
                view.get_content()
            }) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('w'), |s| {
            s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                state.borrow_mut().console_open = false;
            });
            rebuild_workspace(s);
        });

    // Console is smaller - use max_width to limit size
    let console = ActiveWindow::new(repl_log_with_events, "Console").max_width(60);

    // Collect all window views
    let mut windows: Vec<Box<dyn View>> = Vec::new();

    if console_open {
        windows.push(Box::new(console));
    }

    if inspector_open {
        let inspector_view = create_inspector_view(&engine);
        windows.push(Box::new(inspector_view));
    }

    if debug_panel_open {
        let debug_view = create_debug_view(&engine);
        windows.push(Box::new(debug_view));
    }

    // Restore debug panel state after window is added but before layout is finalized
    // (We need to do this after adding to windows because call_on_name requires the view to be in the tree)
    let saved_debug_state = debug_panel_state;

    for name in &editor_names {
        let read_only = engine.borrow().is_eval_function(name);
        let editor_view = create_editor_view(s, &engine, name, read_only);
        windows.push(Box::new(editor_view));
    }

    for &repl_id in &repl_ids {
        let eval_state = repl_eval_handles.remove(&repl_id);
        let debug_state = repl_debug_sessions.remove(&repl_id);
        let repl_view = create_repl_panel_view(&engine, repl_id, repl_histories.remove(&repl_id), eval_state, debug_state);
        windows.push(Box::new(repl_view));
    }

    // Nostos panel (if open)
    let nostos_view: Option<Box<dyn View>> = if nostos_panel_open {
        if let Some(panel_id) = current_panel_id {
            Some(Box::new(create_nostos_panel_view_by_id(&engine, panel_id)))
        } else if let Some(ref info) = current_panel {
            Some(Box::new(create_nostos_panel_view(&engine, info)))
        } else {
            None
        }
    } else {
        None
    };

    if windows.is_empty() && nostos_view.is_none() {
        // Empty workspace - show hint
        let layout = LinearLayout::vertical()
            .child(TextView::new("Workspace empty. Ctrl+B for Browser, Ctrl+R for REPL.")
                .center()
                .full_width()
                .full_height());
        s.add_fullscreen_layer(layout);
        return;
    }

    // Build layout with LinearLayout for equal distribution
    // Split into two rows if more than 6 windows
    let mut layout = LinearLayout::vertical();

    if windows.len() <= 6 {
        // Single row
        let mut row = LinearLayout::horizontal();
        for window in windows {
            row.add_child(window);
        }
        layout.add_child(row.full_width().full_height());
    } else {
        // Two rows - split evenly
        let mid = (windows.len() + 1) / 2; // First row gets extra if odd
        let mut row1 = LinearLayout::horizontal();
        let mut row2 = LinearLayout::horizontal();

        for (i, window) in windows.into_iter().enumerate() {
            if i < mid {
                row1.add_child(window);
            } else {
                row2.add_child(window);
            }
        }

        layout.add_child(row1.full_width().full_height());
        layout.add_child(row2.full_width().full_height());
    }

    // Add nostos panel as separate row if open
    if let Some(nostos) = nostos_view {
        layout.add_child(nostos);
    }

    s.add_fullscreen_layer(layout);

    // Restore debug panel state after layer is added
    if let Some((state, stack, frame_locals)) = saved_debug_state {
        s.call_on_name("debug_panel", |panel: &mut DebugPanel| {
            panel.restore_state(state, stack, frame_locals);
        });
    }
}

/// Create a REPL panel view
fn create_repl_panel_view(
    engine: &Rc<RefCell<ReplEngine>>,
    repl_id: usize,
    histories: Option<(Vec<crate::repl_panel::ReplEntry>, Vec<Vec<String>>)>,
    eval_state: Option<(nostos_vm::ThreadedEvalHandle, Vec<String>)>,
    debug_state: Option<(nostos_vm::DebugSession, Vec<String>)>,
) -> impl View {
    let mut panel = ReplPanel::new(engine.clone(), repl_id);
    if let Some((history, command_history)) = histories {
        panel.set_history(history);
        panel.set_command_history(command_history);
    }
    if let Some((handle, input)) = eval_state {
        panel.restore_eval_state(handle, input);
    }
    if let Some((session, input)) = debug_state {
        panel.restore_debug_state(session, input);
    }
    let panel_id = format!("repl_panel_{}", repl_id);
    let panel_id_copy = panel_id.clone();
    let panel_id_close = repl_id;

    let panel_with_events = OnEventView::new(panel.with_name(&panel_id))
        .on_event(Event::CtrlChar('y'), move |s| {
            // Copy REPL content to clipboard
            if let Some(text) = s.call_on_name(&panel_id_copy, |v: &mut ReplPanel| v.get_content()) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('w'), move |s| {
            close_repl_panel(s, panel_id_close);
        })
        .on_event(Key::Esc, move |s| {
            close_repl_panel(s, panel_id_close);
        });

    ActiveWindow::new(panel_with_events, &format!("REPL #{}", repl_id)).full_width()
}

/// Create the Nostos panel view from PanelInfo
fn create_nostos_panel_view(engine: &Rc<RefCell<ReplEngine>>, info: &PanelInfo) -> impl View {
    let panel = NostosPanel::new(
        engine.clone(),
        &info.view_fn,
        &info.key_handler_fn,
        &info.title
    );

    let panel_with_name = panel.with_name("nostos_mvar_panel");

    // Only need OnEventView for close actions (these affect TUI state, not Nostos state)
    let panel_with_events = OnEventView::new(panel_with_name)
        .on_event(Event::CtrlChar('w'), |s| {
            close_nostos_panel(s);
        })
        .on_event(Key::Esc, |s| {
            close_nostos_panel(s);
        });

    let title = info.title.clone();
    ActiveWindow::new(panel_with_events, &title).full_width()
}

/// Create the Nostos panel view from panel ID (new API)
fn create_nostos_panel_view_by_id(engine: &Rc<RefCell<ReplEngine>>, panel_id: u64) -> impl View {
    // Get panel state
    let (title, content, key_handler_fn) = engine.borrow().get_panel_state(panel_id)
        .map(|s| (s.title.clone(), s.content.clone(), s.key_handler_fn.clone()))
        .unwrap_or_else(|| (format!("Panel {}", panel_id), String::new(), None));

    // Create a simple text view for the content
    let content_view = TextView::new(&content).scrollable();

    // Wrap in an OnEventView to handle keyboard input
    let engine_for_keys = engine.clone();
    let key_handler = key_handler_fn.clone();

    let panel_with_events = OnEventView::new(content_view)
        .on_pre_event_inner(EventTrigger::any(), move |view, event| {
            // Handle close events
            match event {
                Event::CtrlChar('w') | Event::Key(Key::Esc) => {
                    return Some(EventResult::with_cb(|s| close_nostos_panel(s)));
                }
                _ => {}
            }

            // Convert event to key name for handler
            let key_name: Option<String> = match event {
                Event::Key(Key::Up) => Some("up".to_string()),
                Event::Key(Key::Down) => Some("down".to_string()),
                Event::Key(Key::Left) => Some("left".to_string()),
                Event::Key(Key::Right) => Some("right".to_string()),
                Event::Key(Key::Enter) => Some("enter".to_string()),
                Event::Key(Key::Backspace) => Some("backspace".to_string()),
                Event::Key(Key::Tab) => Some("tab".to_string()),
                Event::Key(Key::Del) => Some("delete".to_string()),
                Event::Key(Key::Home) => Some("home".to_string()),
                Event::Key(Key::End) => Some("end".to_string()),
                Event::Key(Key::PageUp) => Some("pageup".to_string()),
                Event::Key(Key::PageDown) => Some("pagedown".to_string()),
                Event::Char(c) => Some(c.to_string()),
                Event::CtrlChar(c) => Some(format!("ctrl+{}", c)),
                Event::AltChar(c) => Some(format!("alt+{}", c)),
                _ => None,
            };

            if let (Some(key), Some(handler_fn)) = (key_name, key_handler.as_ref()) {
                // Call the handler function directly (no parsing/compiling - much faster)
                let _ = engine_for_keys.borrow_mut().call_function_with_string_arg(handler_fn, key);

                // Refresh the panel content after handling key
                let commands = engine_for_keys.borrow_mut().drain_panel_commands();
                for cmd in commands {
                    if let PanelCommand::SetContent { id, content: new_content } = cmd {
                        if id == panel_id {
                            view.get_inner_mut().set_content(new_content);
                        }
                    }
                }

                return Some(EventResult::Consumed(None));
            }

            None
        });

    let panel_with_name = panel_with_events.with_name("nostos_mvar_panel");
    ActiveWindow::new(panel_with_name, &title).full_width()
}

/// Close the Nostos panel
fn close_nostos_panel(s: &mut Cursive) {
    s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        state.nostos_panel_open = false;
        state.current_panel = None;
        state.current_panel_id = None;
    });
    rebuild_workspace(s);
    log_to_repl(s, "Closed panel");
}

/// Open a registered panel
/// Panel must have been registered via Panel.register() from Nostos code
fn open_registered_panel(s: &mut Cursive, _engine: Rc<RefCell<ReplEngine>>, info: PanelInfo) {
    // Check if already open
    let already_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().nostos_panel_open
    }).unwrap();

    if already_open {
        log_to_repl(s, "A panel is already open");
        return;
    }

    let title = info.title.clone();
    let key = info.key.clone();

    // Mark panel as open and store info
    s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        state.nostos_panel_open = true;
        state.current_panel = Some(info);
    });
    rebuild_workspace(s);

    // Focus the panel
    s.focus_name("nostos_mvar_panel").ok();
    log_to_repl(s, &format!("Opened '{}' ({}). Ctrl+W/ESC to close.", title, key));
}

/// Open a panel by ID (new Panel.* API)
fn open_panel_by_id(s: &mut Cursive, engine: Rc<RefCell<ReplEngine>>, panel_id: u64) {
    // Check if already open
    let already_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().nostos_panel_open
    }).unwrap();

    if already_open {
        // If it's the same panel, just refresh content; otherwise close old and open new
        let same_panel = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
            state.borrow().current_panel_id == Some(panel_id)
        }).unwrap();

        if same_panel {
            // Just refresh the workspace to update content
            rebuild_workspace(s);
            return;
        } else {
            // Close the current panel first
            s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                let mut state = state.borrow_mut();
                state.nostos_panel_open = false;
                state.current_panel = None;
                state.current_panel_id = None;
            });
        }
    }

    // Get panel title for logging
    let title = engine.borrow().get_panel_state(panel_id)
        .map(|s| s.title.clone())
        .unwrap_or_else(|| format!("Panel {}", panel_id));

    // Mark panel as open
    s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        state.nostos_panel_open = true;
        state.current_panel_id = Some(panel_id);
    });
    rebuild_workspace(s);

    // Focus the panel
    s.focus_name("nostos_mvar_panel").ok();
    log_to_repl(s, &format!("Opened '{}'. Ctrl+W/ESC to close.", title));
}

/// Process pending panel commands from the engine
fn process_panel_commands(s: &mut Cursive, engine: Rc<RefCell<ReplEngine>>) {
    // Drain all pending panel commands
    let commands = engine.borrow_mut().drain_panel_commands();

    let mut needs_rebuild = false;

    for cmd in commands {
        match cmd {
            PanelCommand::Show { id } => {
                open_panel_by_id(s, engine.clone(), id);
                needs_rebuild = false; // open_panel_by_id already rebuilds
            }
            PanelCommand::Hide { id } => {
                // Check if this is the currently open panel
                let is_current = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                    state.borrow().current_panel_id == Some(id)
                }).unwrap();

                if is_current {
                    s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                        let mut state = state.borrow_mut();
                        state.nostos_panel_open = false;
                        state.current_panel_id = None;
                    });
                    needs_rebuild = true;
                }
            }
            PanelCommand::SetContent { id, .. } => {
                // Check if this is the currently open panel
                let is_current = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                    state.borrow().current_panel_id == Some(id)
                }).unwrap();

                if is_current {
                    needs_rebuild = true;
                }
            }
            // Create, OnKey, RegisterHotkey are already handled by drain_panel_commands
            _ => {}
        }
    }

    if needs_rebuild {
        rebuild_workspace(s);
    }
}

/// Open a new REPL panel
fn open_repl_panel(s: &mut Cursive) {
    let can_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        // Limit total windows (Console + editors + REPLs) to 12 (2 rows of 6)
        let total = 1 + state.open_editors.len() + state.open_repls.len();
        if total >= 12 {
            return Err("Max 12 windows");
        }
        let id = state.next_repl_id;
        state.next_repl_id += 1;
        state.open_repls.push(id);
        Ok(id)
    }).unwrap();

    match can_open {
        Ok(id) => {
            rebuild_workspace(s);
            // Focus the new REPL panel
            let panel_id = format!("repl_panel_{}", id);
            s.focus_name(&panel_id).ok();
            log_to_repl(s, &format!("Opened REPL #{}", id));
        }
        Err(msg) => {
            log_to_repl(s, msg);
        }
    }
}

/// Close a REPL panel by ID
pub fn close_repl_panel(s: &mut Cursive, repl_id: usize) {
    let was_removed = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if let Some(idx) = state.open_repls.iter().position(|&id| id == repl_id) {
            state.open_repls.remove(idx);
            true
        } else {
            false
        }
    }).unwrap();

    if was_removed {
        rebuild_workspace(s);
        // Focus something sensible
        if let Err(_) = s.focus_name("input") {
             // Fallback if input is gone (it is)
             let _ = s.focus_name("repl_log");
        }
        log_to_repl(s, &format!("Closed REPL #{}", repl_id));
    }
}

/// Toggle the console visibility
fn toggle_console(s: &mut Cursive) {
    let console_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        state.console_open = !state.console_open;
        state.console_open
    }).unwrap();

    rebuild_workspace(s);

    if console_open {
        s.focus_name("repl_log").ok();
    } else {
        // Focus first available window
        let (first_editor, first_repl) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
            let state = state.borrow();
            (
                state.open_editors.first().cloned(),
                state.open_repls.first().cloned()
            )
        }).unwrap();

        if let Some(name) = first_editor {
            let editor_id = format!("editor_{}", name);
            s.focus_name(&editor_id).ok();
        } else if let Some(id) = first_repl {
            let panel_id = format!("repl_panel_{}", id);
            s.focus_name(&panel_id).ok();
        }
    }
}

/// Toggle the inspector panel
fn toggle_inspector(s: &mut Cursive) {
    let inspector_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        state.inspector_open = !state.inspector_open;
        state.inspector_open
    }).unwrap();

    if inspector_open {
        // rebuild_workspace will create the inspector panel and drain entries
        rebuild_workspace(s);
        s.focus_name("inspector_panel").ok();
        log_to_repl(s, "Inspector panel opened (Alt+I or :ins to close)");
    } else {
        rebuild_workspace(s);
        // Focus fallback if nothing else
        s.focus_name("repl_log").ok();
        log_to_repl(s, "Inspector panel closed");
    }
}

/// Toggle the debug panel
fn toggle_debug_panel(s: &mut Cursive) {
    let debug_panel_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        state.debug_panel_open = !state.debug_panel_open;
        state.debug_panel_open
    }).unwrap();

    if debug_panel_open {
        rebuild_workspace(s);
        s.focus_name("debug_panel").ok();
        log_to_repl(s, "Debug panel opened - use c/n/s/o keys when focused (Ctrl+D to close)");
    } else {
        rebuild_workspace(s);
        s.focus_name("repl_log").ok();
        log_to_repl(s, "Debug panel closed");
    }
}

/// Create the debug panel view
fn create_debug_view(engine: &Rc<RefCell<ReplEngine>>) -> impl View {
    use crate::debug_panel::DebugPanel;

    let mut panel = DebugPanel::new();

    // Sync breakpoints from engine
    for bp in engine.borrow().get_breakpoints() {
        panel.add_breakpoint(bp);
    }

    let panel_with_events = OnEventView::new(panel.with_name("debug_panel"))
        .on_event(Event::CtrlChar('d'), |s| {
            toggle_debug_panel(s);
        });

    ActiveWindow::new(panel_with_events, "Debug")
}

/// Show help dialog with all keybindings
fn show_help_dialog(s: &mut Cursive) {
    use cursive::theme::{Effect, Style, ColorStyle, BaseColor, Color};

    let mut styled = StyledString::new();
    let bold = Style::from(Effect::Bold);
    let header_style = Style::from(ColorStyle::new(
        Color::Light(BaseColor::Yellow),
        Color::TerminalDefault,
    )).combine(Effect::Bold);
    let key_style = Style::from(ColorStyle::new(
        Color::Light(BaseColor::Cyan),
        Color::TerminalDefault,
    ));

    // Title
    styled.append_plain("\n");
    styled.append_styled("                 NOSTOS KEYBINDINGS\n", header_style);
    styled.append_plain("\n");

    // Helper to add a section
    fn add_section(styled: &mut StyledString, title: &str, bindings: &[(&str, &str)],
                   header_style: Style, key_style: Style) {
        styled.append_styled(format!("  {}\n", title), header_style);
        styled.append_plain("  ─────────────────────────────────────────────\n");
        for (key, desc) in bindings {
            styled.append_plain("  ");
            styled.append_styled(format!("{:<14}", key), key_style);
            styled.append_plain(format!("  {}\n", desc));
        }
        styled.append_plain("\n");
    }

    add_section(&mut styled, "GLOBAL", &[
        ("Ctrl+H", "Show this help"),
        ("Ctrl+B", "Open file browser"),
        ("Ctrl+R", "Open new REPL panel"),
        ("Alt+I / F9", "Toggle inspector panel"),
        ("Alt+C", "Toggle console"),
        ("Alt+T", "Open Nostos test panel"),
        ("Shift+Tab", "Cycle focus between windows"),
    ], header_style, key_style);

    add_section(&mut styled, "REPL PANEL", &[
        ("Enter", "Execute code"),
        ("Alt+Enter", "Insert newline (multiline)"),
        ("Tab", "Autocomplete / cycle suggestions"),
        ("Shift+Tab", "Cycle suggestions backward"),
        ("Ctrl+Space", "Show autocomplete menu"),
        ("Ctrl+S", "Save session to file"),
        ("Ctrl+Y", "Copy output to clipboard"),
        ("Ctrl+W", "Close REPL panel"),
        ("Up/Down", "Navigate history / autocomplete"),
        ("PageUp/Down", "Scroll output"),
        ("Esc", "Cancel autocomplete"),
    ], header_style, key_style);

    add_section(&mut styled, "CODE EDITOR", &[
        ("Ctrl+S", "Save file"),
        ("Ctrl+G", "Go to definition"),
        ("Ctrl+F", "Search in file"),
        ("Ctrl+Y", "Copy to clipboard"),
        ("Ctrl+W", "Close editor"),
        ("Tab", "Indent / autocomplete"),
        ("Shift+Tab", "Dedent / cycle backward"),
        ("Ctrl+Space", "Show autocomplete menu"),
        ("PageUp/Down", "Scroll"),
        ("Esc", "Cancel autocomplete / close"),
    ], header_style, key_style);

    add_section(&mut styled, "CONSOLE", &[
        ("Ctrl+Y", "Copy content to clipboard"),
        ("Ctrl+W", "Close console"),
    ], header_style, key_style);

    add_section(&mut styled, "INSPECTOR PANEL", &[
        ("Tab", "Switch between tabs"),
        ("Up/Down", "Navigate values"),
        ("Enter/Right", "Drill into value"),
        ("Left/Bksp", "Go up one level"),
        ("PageUp/Down", "Scroll list"),
        ("Ctrl+Y", "Copy value to clipboard"),
    ], header_style, key_style);

    add_section(&mut styled, "FILE BROWSER (Ctrl+B)", &[
        ("Enter", "Open selected file"),
        ("n", "Create new file"),
        ("r", "Rename file"),
        ("d", "Delete file"),
        ("e", "Show compile errors"),
        ("g", "Show call graph"),
        ("Left/Right", "Navigate lists"),
        ("Esc", "Close browser"),
    ], header_style, key_style);

    styled.append_plain("          Press Esc or Enter to close\n");

    let text_view = TextView::new(styled);
    let scroll_view = ScrollView::new(text_view)
        .scroll_x(false)
        .scroll_y(true);

    let dialog = Dialog::around(scroll_view)
        .title("Help")
        .fixed_size((52, 32));

    let dialog_with_keys = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| { s.pop_layer(); })
        .on_event(Key::Enter, |s| { s.pop_layer(); })
        .on_event(Event::CtrlChar('h'), |s| { s.pop_layer(); });

    s.add_layer(dialog_with_keys);
}

/// Create the inspector panel view
fn create_inspector_view(engine: &Rc<RefCell<ReplEngine>>) -> impl View {
    let panel = InspectorPanel::new();

    // Add any pending entries
    let entries = engine.borrow().drain_inspect_entries();
    let mut panel = panel;
    if !entries.is_empty() {
        panel.process_entries(entries);
    }

    let panel_with_events = OnEventView::new(panel.with_name("inspector_panel"))
        .on_event(Event::CtrlChar('y'), |s| {
            // Copy inspector content to clipboard
            if let Some(text) = s.call_on_name("inspector_panel", |view: &mut InspectorPanel| {
                view.get_content()
            }) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('w'), |s| {
            // Close inspector with Ctrl+W
            s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                state.borrow_mut().inspector_open = false;
            });
            rebuild_workspace(s);
            s.focus_name("repl_log").ok();
            log_to_repl(s, "Inspector panel closed");
        })
        .on_event(Key::Esc, |s| {
            // Close inspector with Esc
            s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                state.borrow_mut().inspector_open = false;
            });
            rebuild_workspace(s);
            s.focus_name("repl_log").ok();
            log_to_repl(s, "Inspector panel closed");
        });

    ActiveWindow::new(panel_with_events, "Inspector").full_width()
}

/// Create an editor view for a given name
/// If read_only is true, the editor will be in view-only mode (for eval'd functions)
fn create_editor_view(_s: &mut Cursive, engine: &Rc<RefCell<ReplEngine>>, name: &str, read_only: bool) -> impl View {
    let source = engine.borrow().get_source(name);
    let is_new_definition = source.starts_with("Not found");
    let source = if is_new_definition && !read_only {
        // Use simple name (without module prefix) for the placeholder
        let simple_name = name.rsplit('.').next().unwrap_or(name);
        format!("{}() = {{\n    \n}}", simple_name)
    } else if is_new_definition {
        // For read-only but not found, show a message
        format!("# Source not available for: {}", name)
    } else {
        source
    };

    // Extract the expected simple name (without module prefix)
    let expected_simple_name = name.rsplit('.').next().unwrap_or(name).to_string();

    let editor = CodeEditor::new(source)
        .with_function_name(name)  // Set module context for autocomplete
        .with_engine(engine.clone())
        .with_read_only(read_only);
    let editor_id = format!("editor_{}", name);

    let name_for_save = name.to_string();
    let name_for_close = name.to_string();
    let name_for_close_w = name.to_string();
    let name_for_close_w_save = name.to_string();
    let name_for_close_w_discard = name.to_string();
    let engine_save = engine.clone();
    let engine_save_w = engine.clone();
    let editor_id_save = editor_id.clone();
    let editor_id_copy = editor_id.clone();
    let editor_id_dirty = editor_id.clone();
    let editor_id_save_w = editor_id.clone();
    let name_for_graph = name.to_string();
    let engine_for_graph = engine.clone();
    let name_for_eval = name.to_string();
    let engine_eval = engine.clone();
    let editor_id_eval = editor_id.clone();

    // Ctrl+S to save, Ctrl+W to close, Ctrl+Y to copy, Ctrl+G for graph, Ctrl+E to compile, Esc to close
    let editor_with_events = OnEventView::new(editor.with_name(&editor_id))
        .on_event(Event::CtrlChar('y'), move |s| {
            // Copy editor content to clipboard (Ctrl+Y)
            if let Some(text) = s.call_on_name(&editor_id_copy, |v: &mut CodeEditor| v.get_content()) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('w'), move |s| {
            // Check if editor has unsaved changes
            let is_dirty = s.call_on_name(&editor_id_dirty, |v: &mut CodeEditor| v.is_dirty())
                .unwrap_or(false);

            if is_dirty {
                // Show confirmation dialog
                let name_save = name_for_close_w_save.clone();
                let name_discard = name_for_close_w_discard.clone();
                let editor_id_for_save = editor_id_save_w.clone();
                let engine_for_save = engine_save_w.clone();

                s.add_layer(
                    Dialog::text("Unsaved changes. What do you want to do?")
                        .title("Close Editor")
                        .button("Save", move |s| {
                            s.pop_layer(); // Remove dialog
                            // Save the content
                            let content = s.call_on_name(&editor_id_for_save, |v: &mut CodeEditor| v.get_content());
                            if let Some(content) = content {
                                match engine_for_save.borrow_mut().eval(&content) {
                                    Ok(output) => {
                                        if output.is_empty() {
                                            log_to_repl(s, &format!("Saved {}", name_save));
                                        } else {
                                            log_to_repl(s, &output);
                                        }
                                        close_editor(s, &name_save);
                                    }
                                    Err(e) => {
                                        s.add_layer(Dialog::info(format!("Error: {}", e)));
                                    }
                                }
                            }
                        })
                        .button("Discard", move |s| {
                            s.pop_layer(); // Remove dialog
                            close_editor(s, &name_discard);
                        })
                        .button("Cancel", |s| {
                            s.pop_layer(); // Just close dialog
                        })
                );
            } else {
                // No unsaved changes, close directly
                close_editor(s, &name_for_close_w);
            }
        })
        .on_event(Event::CtrlChar('g'), move |s| {
             let name = name_for_graph.clone();
             let engine = engine_for_graph.clone();
             show_call_graph_dialog(s, engine, name);
        })
        .on_event(Event::CtrlChar('s'), move |s| {
            debug_log(&format!("Ctrl+S pressed for: {}", name_for_save));
            let content = match s.call_on_name(&editor_id_save, |v: &mut CodeEditor| v.get_content()) {
                Some(c) => c,
                None => {
                    debug_log(&format!("ERROR: Could not find editor {}", editor_id_save));
                    log_to_repl(s, &format!("Error: Could not find editor {}", editor_id_save));
                    return;
                }
            };
            debug_log(&format!("Content length: {} chars", content.len()));

            // Extract actual definition names from the content
            let actual_names = extract_definition_names(&content);
            debug_log(&format!("Actual definition names in content: {:?}", actual_names));

            // Check if this is a rename (existing definition, expected name not in actual names)
            let was_renamed = !is_new_definition && !actual_names.contains(&expected_simple_name);
            if was_renamed {
                debug_log(&format!("Detected rename: expected '{}' not found in {:?}", expected_simple_name, actual_names));
            }

            let mut engine = engine_save.borrow_mut();

            // Check if this is a metadata file
            if engine.is_metadata(&name_for_save) {
                debug_log(&format!("Saving as METADATA: {}", name_for_save));
                // Metadata files only need to be saved, not evaluated
                match engine.save_metadata(&name_for_save, &content) {
                    Ok(()) => {
                        debug_log(&format!("Metadata saved OK: {}", name_for_save));
                        drop(engine);
                        log_to_repl(s, &format!("Saved {}", name_for_save));
                        close_editor(s, &name_for_save);
                    }
                    Err(e) => {
                        debug_log(&format!("Metadata save ERROR: {}", e));
                        drop(engine);
                        s.add_layer(Dialog::info(format!("Error: {}", e)));
                    }
                }
                return;
            }

            debug_log(&format!("Saving as DEFINITION: {}", name_for_save));
            // Strip together directives before eval (compiler doesn't understand them)
            let eval_content: String = content
                .lines()
                .filter(|line| !line.trim().starts_with("# together "))
                .collect::<Vec<_>>()
                .join("\n");
            debug_log(&format!("Eval content (stripped together): {} chars", eval_content.len()));

            // Try to compile first, then decide whether to save
            if engine.has_source_manager() {
                debug_log(&format!("Trying to compile: {}", name_for_save));
                let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine.eval_in_module(&eval_content, Some(&name_for_save))
                }));

                let compile_result = match compile_result {
                    Ok(r) => r,
                    Err(panic_info) => {
                        let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        debug_log(&format!("eval_in_module PANIC: {}", panic_msg));
                        Err(format!("Internal error (panic): {}", panic_msg))
                    }
                };

                match compile_result {
                    Ok(output) => {
                        // Compilation succeeded - save to disk
                        debug_log(&format!("Compilation OK, saving to disk"));
                        let save_result = engine.save_group_source(&name_for_save, &content);

                        match save_result {
                            Ok(saved_names) => {
                                // Mark as compiled - use actual_names since saved_names may be empty if no file changes
                                let module_prefix = if let Some(dot_pos) = name_for_save.rfind('.') {
                                    format!("{}.", &name_for_save[..dot_pos])
                                } else {
                                    String::new()
                                };

                                // Update status for all names in the content (not just saved ones)
                                // This ensures we clear error status even when file didn't change
                                let names_to_update = if actual_names.is_empty() {
                                    // Fall back to the expected name if we couldn't parse actual names
                                    vec![expected_simple_name.clone()]
                                } else {
                                    actual_names.clone()
                                };
                                for name in &names_to_update {
                                    let qualified = format!("{}{}", module_prefix, name);
                                    debug_log(&format!("Setting compile status to Compiled for: {}", qualified));
                                    engine.set_compile_status(&qualified, CompileStatus::Compiled);
                                }

                                if saved_names.is_empty() {
                                    log_to_repl(s, &format!("Saved {} (no changes)", name_for_save));
                                } else if saved_names.len() == 1 {
                                    log_to_repl(s, &format!("Saved {} (committed)", saved_names[0]));
                                } else {
                                    log_to_repl(s, &format!("Saved {} definitions: {} (committed)",
                                        saved_names.len(), saved_names.join(", ")));
                                }

                                drop(engine);
                                close_editor_and_browse(s, &name_for_save);

                                // Show rename warning if applicable
                                if was_renamed {
                                    let new_names = if actual_names.is_empty() {
                                        "(no definitions found)".to_string()
                                    } else {
                                        actual_names.join(", ")
                                    };
                                    s.add_layer(
                                        Dialog::text(format!(
                                            "Definition '{}' was renamed to '{}'.\n\n\
                                            The original '{}' still exists and can be deleted from the browser if needed.",
                                            expected_simple_name, new_names, expected_simple_name
                                        ))
                                        .title("Definition Renamed")
                                        .button("OK", |s| { s.pop_layer(); })
                                    );
                                }
                            }
                            Err(e) => {
                                drop(engine);
                                s.add_layer(Dialog::info(format!("Save failed: {}", e)));
                            }
                        }
                    }
                    Err(compile_error) => {
                        // Compilation failed - offer to save anyway
                        debug_log(&format!("Compilation failed: {}", compile_error));
                        drop(engine);

                        // Format error message for human readability
                        let friendly_error = format_compile_error(&compile_error);

                        // Clone values for the closure
                        let error_for_status = compile_error.clone();
                        let name_for_dialog = name_for_save.clone();
                        let content_for_dialog = content.clone();
                        let actual_names_for_dialog = actual_names.clone();
                        let expected_simple_name_for_dialog = expected_simple_name.clone();
                        let was_renamed_for_dialog = was_renamed;

                        s.add_layer(
                            Dialog::text(format!(
                                "Compilation failed:\n\n{}\n\nSave anyway (with errors)?",
                                friendly_error
                            ))
                            .title("Compile Error")
                            .button("Save Anyway", move |s| {
                                s.pop_layer();
                                // Save despite errors
                                let engine = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
                                    state.borrow().engine.clone()
                                }).unwrap();
                                let mut engine = engine.borrow_mut();

                                let save_result = engine.save_group_source(&name_for_dialog, &content_for_dialog);
                                match save_result {
                                    Ok(saved_names) => {
                                        // Mark as having compile errors
                                        let module_prefix = if let Some(dot_pos) = name_for_dialog.rfind('.') {
                                            format!("{}.", &name_for_dialog[..dot_pos])
                                        } else {
                                            String::new()
                                        };
                                        for name in &saved_names {
                                            let qualified = format!("{}{}", module_prefix, name);
                                            engine.set_compile_status(&qualified, CompileStatus::CompileError(error_for_status.clone()));
                                            engine.mark_dependents_stale(&qualified, &format!("{} has errors", qualified));
                                        }

                                        if saved_names.is_empty() {
                                            log_to_repl(s, &format!("Saved {} (no changes)", name_for_dialog));
                                        } else if saved_names.len() == 1 {
                                            log_to_repl(s, &format!("Saved {} (with errors)", saved_names[0]));
                                        } else {
                                            log_to_repl(s, &format!("Saved {} definitions: {} (with errors)",
                                                saved_names.len(), saved_names.join(", ")));
                                        }

                                        drop(engine);
                                        close_editor_and_browse(s, &name_for_dialog);

                                        // Show rename warning if applicable
                                        if was_renamed_for_dialog {
                                            let new_names = if actual_names_for_dialog.is_empty() {
                                                "(no definitions found)".to_string()
                                            } else {
                                                actual_names_for_dialog.join(", ")
                                            };
                                            s.add_layer(
                                                Dialog::text(format!(
                                                    "Definition '{}' was renamed to '{}'.\n\n\
                                                    The original '{}' still exists and can be deleted from the browser if needed.",
                                                    expected_simple_name_for_dialog, new_names, expected_simple_name_for_dialog
                                                ))
                                                .title("Definition Renamed")
                                                .button("OK", |s| { s.pop_layer(); })
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        drop(engine);
                                        s.add_layer(Dialog::info(format!("Save failed: {}", e)));
                                    }
                                }
                            })
                            .dismiss_button("Cancel")  // Esc or click Cancel to close
                        );
                    }
                }
            } else {
                // No source manager - just try to compile without saving
                debug_log("No source manager, just compiling");
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine.eval_in_module(&eval_content, Some(&name_for_save))
                }));

                match result {
                    Ok(Ok(output)) => {
                        drop(engine);
                        if output.is_empty() {
                            log_to_repl(s, &format!("Evaluated {}", name_for_save));
                        } else {
                            log_to_repl(s, &output);
                        }
                        close_editor_and_browse(s, &name_for_save);
                    }
                    Ok(Err(e)) => {
                        drop(engine);
                        s.add_layer(Dialog::info(format!("Error: {}", e)));
                    }
                    Err(panic_info) => {
                        let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        drop(engine);
                        s.add_layer(Dialog::info(format!("Internal error: {}", panic_msg)));
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('e'), move |s| {
            // Ctrl+E: Save and compile without closing the editor
            debug_log(&format!("Ctrl+E pressed for: {}", name_for_eval));
            let content = match s.call_on_name(&editor_id_eval, |v: &mut CodeEditor| v.get_content()) {
                Some(c) => c,
                None => {
                    debug_log(&format!("ERROR: Could not find editor {}", editor_id_eval));
                    log_to_repl(s, &format!("Error: Could not find editor {}", editor_id_eval));
                    return;
                }
            };

            // Extract actual definition names from the content
            let actual_names = extract_definition_names(&content);
            debug_log(&format!("Ctrl+E: definition names: {:?}", actual_names));

            let mut engine = engine_eval.borrow_mut();

            // Check if this is a metadata file
            if engine.is_metadata(&name_for_eval) {
                debug_log(&format!("Ctrl+E: Saving metadata: {}", name_for_eval));
                match engine.save_metadata(&name_for_eval, &content) {
                    Ok(()) => {
                        drop(engine);
                        log_to_repl(s, &format!("Saved {}", name_for_eval));
                        // Mark editor as not dirty
                        s.call_on_name(&editor_id_eval, |v: &mut CodeEditor| v.mark_saved());
                    }
                    Err(e) => {
                        drop(engine);
                        log_to_repl(s, &format!("Error saving {}: {}", name_for_eval, e));
                    }
                }
                return;
            }

            // Strip together directives before eval
            let eval_content: String = content
                .lines()
                .filter(|line| !line.trim().starts_with("# together "))
                .collect::<Vec<_>>()
                .join("\n");

            if engine.has_source_manager() {
                // Time the compilation
                let start = std::time::Instant::now();

                let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine.eval_in_module(&eval_content, Some(&name_for_eval))
                }));

                let elapsed = start.elapsed();
                let elapsed_ms = elapsed.as_millis();

                let compile_result = match compile_result {
                    Ok(r) => r,
                    Err(panic_info) => {
                        let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        debug_log(&format!("Ctrl+E eval_in_module PANIC: {}", panic_msg));
                        Err(format!("Internal error (panic): {}", panic_msg))
                    }
                };

                match compile_result {
                    Ok(_output) => {
                        // Compilation succeeded - save to disk
                        debug_log(&format!("Ctrl+E: Compilation OK, saving to disk"));
                        let save_result = engine.save_group_source(&name_for_eval, &content);

                        match save_result {
                            Ok(_saved_names) => {
                                // Mark as compiled
                                let module_prefix = if let Some(dot_pos) = name_for_eval.rfind('.') {
                                    format!("{}.", &name_for_eval[..dot_pos])
                                } else {
                                    String::new()
                                };

                                // Build qualified names for logging
                                let qualified_names: Vec<String> = actual_names.iter()
                                    .map(|n| format!("{}{}", module_prefix, n))
                                    .collect();

                                // Update compile status for all definitions
                                for qualified in &qualified_names {
                                    engine.set_compile_status(qualified, CompileStatus::Compiled);
                                }

                                drop(engine);

                                // Mark editor as not dirty
                                s.call_on_name(&editor_id_eval, |v: &mut CodeEditor| v.mark_saved());

                                // Log success message with function names and timing
                                let names_str = if qualified_names.is_empty() {
                                    name_for_eval.clone()
                                } else {
                                    qualified_names.join(", ")
                                };
                                log_to_repl(s, &format!("Compiled: {} ({}ms)", names_str, elapsed_ms));
                            }
                            Err(e) => {
                                drop(engine);
                                log_to_repl(s, &format!("Save failed: {}", e));
                            }
                        }
                    }
                    Err(compile_error) => {
                        // Compilation failed - set inline error status (no popup)
                        debug_log(&format!("Ctrl+E: Compilation failed: {}", compile_error));

                        // Mark definitions as having compile errors
                        let module_prefix = if let Some(dot_pos) = name_for_eval.rfind('.') {
                            format!("{}.", &name_for_eval[..dot_pos])
                        } else {
                            String::new()
                        };

                        for name in &actual_names {
                            let qualified = format!("{}{}", module_prefix, name);
                            engine.set_compile_status(&qualified, CompileStatus::CompileError(compile_error.clone()));
                        }

                        drop(engine);

                        // Update the editor's compile error display inline
                        s.call_on_name(&editor_id_eval, |v: &mut CodeEditor| {
                            v.set_compile_error(Some(compile_error.clone()));
                        });

                        log_to_repl(s, &format!("Compile error ({}ms)", elapsed_ms));
                    }
                }
            } else {
                // No source manager - just try to compile
                debug_log("Ctrl+E: No source manager, just compiling");
                let start = std::time::Instant::now();

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine.eval_in_module(&eval_content, Some(&name_for_eval))
                }));

                let elapsed = start.elapsed();
                let elapsed_ms = elapsed.as_millis();

                match result {
                    Ok(Ok(_output)) => {
                        drop(engine);
                        let names_str = if actual_names.is_empty() {
                            name_for_eval.clone()
                        } else {
                            actual_names.join(", ")
                        };
                        log_to_repl(s, &format!("Compiled: {} ({}ms)", names_str, elapsed_ms));
                    }
                    Ok(Err(compile_error)) => {
                        drop(engine);
                        s.call_on_name(&editor_id_eval, |v: &mut CodeEditor| {
                            v.set_compile_error(Some(compile_error));
                        });
                        log_to_repl(s, &format!("Compile error ({}ms)", elapsed_ms));
                    }
                    Err(panic_info) => {
                        let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        drop(engine);
                        log_to_repl(s, &format!("Internal error: {}", panic_msg));
                    }
                }
            }
        })
        .on_event(Key::Esc, move |s| {
            close_editor(s, &name_for_close);
        })
        .on_pre_event_inner(Event::Shift(Key::Tab), |_, _| {
            // Consume the event - don't let it propagate
            // Global callback will handle window cycling
            Some(EventResult::Consumed(None))
        });

    // Wrap in ActiveWindow with title - add [VIEW ONLY] for read-only editors
    let title = if read_only {
        format!("{} [VIEW ONLY]", name)
    } else {
        name.to_string()
    };
    ActiveWindow::new(editor_with_events.full_height(), &title).full_width()
}

fn open_editor(s: &mut Cursive, name: &str) {
    debug_log(&format!("open_editor called for: {}", name));
    let name_owned = name.to_string();

    // Check limit and add to state
    let can_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if state.open_editors.contains(&name_owned) {
            return Err("Editor already open");
        }
        if state.open_editors.len() >= 11 { // 11 editors + 1 console = 12 windows max
            return Err("Max 12 windows");
        }
        state.open_editors.push(name_owned.clone());
        Ok(())
    }).unwrap();

    if let Err(msg) = can_open {
        log_to_repl(s, msg);
        return;
    }

    // Rebuild the entire workspace layout
    rebuild_workspace(s);

    // Focus the new editor
    let editor_id = format!("editor_{}", name);
    eprintln!("[TUI] Focusing editor: {}", editor_id);
    if let Err(e) = s.focus_name(&editor_id) {
        eprintln!("[TUI] Failed to focus {}: {:?}", editor_id, e);
    } else {
        eprintln!("[TUI] Successfully focused {}", editor_id);
    }
}

fn close_editor(s: &mut Cursive, name: &str) {
    let was_removed = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if let Some(idx) = state.open_editors.iter().position(|x| x == name) {
            state.open_editors.remove(idx);
            true
        } else {
            false
        }
    }).unwrap();

    if was_removed {
        // Rebuild the entire workspace layout
        rebuild_workspace(s);

        // Focus fallback
        s.focus_name("repl_log").ok();
    }
}

/// Close the editor and open the browser at the module containing the definition
fn close_editor_and_browse(s: &mut Cursive, name: &str) {
    // Extract module path from qualified name (e.g., "utils.foo" -> ["utils"])
    let parts: Vec<&str> = name.split('.').collect();
    let module_path: Vec<String> = if parts.len() > 1 {
        parts[..parts.len()-1].iter().map(|s| s.to_string()).collect()
    } else {
        vec![]
    };

    let was_removed = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if let Some(idx) = state.open_editors.iter().position(|x| x == name) {
            state.open_editors.remove(idx);
            true
        } else {
            false
        }
    }).unwrap();

    if was_removed {
        // Rebuild the entire workspace layout
        rebuild_workspace(s);

        // Get engine and open browser at the module path
        let engine = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
            state.borrow().engine.clone()
        }).unwrap();

        show_browser_dialog(s, engine, module_path);
    }
}

/// Open the module browser dialog
fn open_browser(s: &mut Cursive) {
    // Get engine reference and current editor info
    let (engine, open_editors, active_idx) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let state = state.borrow();
        (state.engine.clone(), state.open_editors.clone(), state.active_window_idx)
    }).unwrap();

    // Try to find the module path from the currently active editor
    let mut path: Vec<String> = vec![];

    // Build window list to find what's at active_idx
    // Order: console (if open), editors, repls, inspector, nostos_panel
    let console_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        state.borrow().console_open
    }).unwrap_or(true);

    let console_offset = if console_open { 1 } else { 0 };

    // Check if active window is an editor
    if active_idx >= console_offset && active_idx < console_offset + open_editors.len() {
        let editor_idx = active_idx - console_offset;
        if let Some(editor_name) = open_editors.get(editor_idx) {
            // Extract module path from function name (e.g., "myModule.myFunc" -> ["myModule"])
            if let Some(dot_pos) = editor_name.rfind('.') {
                let module_part = &editor_name[..dot_pos];
                path = module_part.split('.').map(|s| s.to_string()).collect();
            }
        }
    }

    show_browser_dialog(s, engine, path);
}

/// Show the browser dialog at a given path
fn show_browser_dialog(s: &mut Cursive, engine: Rc<RefCell<ReplEngine>>, path: Vec<String>) {
    // Get browser items with mutable borrow (syncs dynamic functions)
    let items = engine.borrow_mut().get_browser_items(&path);
    // Now borrow immutably for the rest
    let engine_ref = engine.borrow();

    // Build title with status summary
    let mut title = if path.is_empty() {
        "Browse".to_string()
    } else {
        format!("Browse: {}", path.join("."))
    };

    // Add status summary if there are problems
    if let Some(status) = engine_ref.get_status_summary() {
        title = format!("{} {}", title, status);
    }

    // Get current module prefix for qualified names
    let module_prefix = if path.is_empty() {
        String::new()
    } else {
        format!("{}.", path.join("."))
    };

    let mut select = SelectView::<BrowserItem>::new();

    // Add ".." entry if not at root
    if !path.is_empty() {
        select.add_item("📁 ..", BrowserItem::Module("..".to_string()));
    }

    // Add all items with appropriate icons and status indicators
    for item in &items {
        let label: StyledString = match item {
            BrowserItem::Module(name) => {
                // Check if this module has any errors
                let mut module_path = path.clone();
                module_path.push(name.clone());
                let has_errors = engine_ref.module_has_problems(&module_path);
                if has_errors {
                    StyledString::plain(format!("📁 {} 🔴", name))
                } else {
                    StyledString::plain(format!("📁 {}", name))
                }
            }
            BrowserItem::Type { name, eval_created } => {
                let eval_prefix = if *eval_created { "[eval] " } else { "" };
                let mut styled = StyledString::plain(format!("📦 {}{}", eval_prefix, name));
                if *eval_created {
                    styled = StyledString::styled(
                        styled.source(),
                        Style::from(Color::Rgb(100, 180, 200))  // Muted cyan for eval
                    );
                }
                styled
            }
            BrowserItem::Trait { name, eval_created } => {
                let eval_prefix = if *eval_created { "[eval] " } else { "" };
                let mut styled = StyledString::plain(format!("🔷 {}{}", eval_prefix, name));
                if *eval_created {
                    styled = StyledString::styled(
                        styled.source(),
                        Style::from(Color::Rgb(100, 180, 200))  // Muted cyan for eval
                    );
                }
                styled
            }
            BrowserItem::Function { name, signature, doc, eval_created } => {
                // Check compile status for this function
                let qualified_name = format!("{}{}", module_prefix, name);
                let status_indicator = match engine_ref.get_compile_status(&qualified_name) {
                    Some(CompileStatus::CompileError(_)) => " 🔴",
                    Some(CompileStatus::Stale { .. }) => " 🟡",
                    Some(CompileStatus::NotCompiled) => " [?]",
                    _ => "",
                };
                // Build styled string with doc comment in different color
                let mut styled = StyledString::new();
                // Add [eval] prefix for eval-created functions (view-only)
                let eval_prefix = if *eval_created { "[eval] " } else { "" };
                if signature.is_empty() {
                    styled.append_plain(format!("ƒ  {}{}{}", eval_prefix, name, status_indicator));
                } else {
                    styled.append_plain(format!("ƒ  {}{} :: {}{}", eval_prefix, name, signature, status_indicator));
                }
                // Style eval functions differently (muted cyan)
                if *eval_created {
                    styled = StyledString::styled(
                        styled.source(),
                        Style::from(Color::Rgb(100, 180, 200))  // Muted cyan for eval
                    );
                }
                // Add doc comment in a muted green/cyan color
                if let Some(doc_text) = doc.as_ref().and_then(|d| d.lines().next()) {
                    styled.append_styled(
                        format!(" # {}", doc_text),
                        Style::from(Color::Rgb(120, 180, 140))  // Muted green
                    );
                }
                styled
            }
            BrowserItem::Variable { name, mutable, eval_created, is_mvar, type_name } => {
                let eval_prefix = if *eval_created { "[eval] " } else { "" };
                let type_suffix = type_name.as_ref()
                    .map(|t| format!(": {}", t))
                    .unwrap_or_default();
                let text = if *is_mvar {
                    format!("⚡ mvar {}{}{}", eval_prefix, name, type_suffix)
                } else if *mutable {
                    format!("⚡ var {}{}{}", eval_prefix, name, type_suffix)
                } else {
                    format!("📌 {}{}{}", eval_prefix, name, type_suffix)
                };
                let mut styled = StyledString::plain(text);
                if *eval_created {
                    styled = StyledString::styled(
                        styled.source(),
                        Style::from(Color::Rgb(100, 180, 200))  // Muted cyan for eval
                    );
                }
                styled
            }
            BrowserItem::Metadata { .. } => StyledString::plain("⚙  _meta (together directives)"),
        };
        select.add_item(label, item.clone());
    }
    drop(engine_ref);

    // Handle selection change - update preview pane with syntax highlighting
    let path_for_preview = path.clone();
    let engine_for_preview = engine.clone();
    select.set_on_select(move |s: &mut Cursive, item: &BrowserItem| {
        let engine = engine_for_preview.clone();
        let current_path = path_for_preview.clone();

        let preview_styled: StyledString = match item {
            BrowserItem::Function { .. } => {
                let full_name = engine.borrow().get_full_name(&current_path, item);
                let source = engine.borrow().get_source(&full_name);
                syntax_highlight_code(&source)
            }
            BrowserItem::Type { .. } => {
                let full_name = engine.borrow().get_full_name(&current_path, item);
                let source = engine.borrow().get_source(&full_name);
                syntax_highlight_code(&source)
            }
            BrowserItem::Trait { .. } => {
                let full_name = engine.borrow().get_full_name(&current_path, item);
                let source = engine.borrow().get_source(&full_name);
                syntax_highlight_code(&source)
            }
            BrowserItem::Module(name) if name == ".." => {
                StyledString::plain("(parent directory)")
            }
            BrowserItem::Module(name) => {
                StyledString::plain(format!("Module: {}", name))
            }
            BrowserItem::Variable { name, .. } => {
                // Try REPL binding first, then mvar with qualified name
                let mut eng = engine.borrow_mut();

                // First try REPL variable binding
                if let Some(val) = eng.get_var_value_raw(name) {
                    let source = format!("{} = {}", name, val);
                    syntax_highlight_code(&source)
                } else {
                    // Try as mvar - build qualified name
                    let qualified = if current_path.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", current_path.join("."), name)
                    };
                    if let Some(val_str) = eng.get_mvar_value_string(&qualified) {
                        let source = format!("{} = {}", name, val_str);
                        syntax_highlight_code(&source)
                    } else {
                        // Mvar exists but value not available (not yet initialized?)
                        StyledString::plain(format!("{} (mvar - run code to initialize)", name))
                    }
                }
            }
            BrowserItem::Metadata { module } => {
                let full_name = format!("{}._meta", module);
                let source = engine.borrow().get_source(&full_name);
                syntax_highlight_code(&source)
            }
        };

        // Update preview with styled content
        s.call_on_name("browser_preview", |v: &mut TextView| {
            v.set_content(preview_styled);
        });
    });

    // Handle submit (Enter key)
    let path_for_select = path.clone();
    let engine_for_select = engine.clone();
    select.set_on_submit(move |s, item: &BrowserItem| {
        let engine = engine_for_select.clone();
        let mut new_path = path_for_select.clone();

        match item {
            BrowserItem::Module(name) if name == ".." => {
                // Go up one level
                s.pop_layer();
                new_path.pop();
                show_browser_dialog(s, engine, new_path);
            }
            BrowserItem::Module(name) => {
                // Drill into module
                s.pop_layer();
                new_path.push(name.clone());
                show_browser_dialog(s, engine, new_path);
            }
            BrowserItem::Function { name, .. } => {
                // Open function in editor
                let full_name = engine.borrow().get_full_name(&new_path, item);
                debug_log(&format!("Browser: selected Function: {} -> full_name: {}", name, full_name));
                s.pop_layer();
                open_editor(s, &full_name);
            }
            BrowserItem::Type { name, .. } => {
                // Open type in editor directly (preview pane shows info)
                let full_name = engine.borrow().get_full_name(&new_path, item);
                debug_log(&format!("Browser: selected Type: {} -> full_name: {}", name, full_name));
                s.pop_layer();
                open_editor(s, &full_name);
            }
            BrowserItem::Trait { name, .. } => {
                // Open trait in editor directly (preview pane shows info)
                let full_name = engine.borrow().get_full_name(&new_path, item);
                debug_log(&format!("Browser: selected Trait: {} -> full_name: {}", name, full_name));
                s.pop_layer();
                open_editor(s, &full_name);
            }
            BrowserItem::Variable { name, .. } => {
                // Get variable value and open in inspector
                // Try REPL binding first, then mvar with qualified name
                let var_name = name.clone();
                let value_opt = {
                    let mut eng = engine.borrow_mut();
                    if let Some(val) = eng.get_var_value_raw(&var_name) {
                        Some(val)
                    } else {
                        // Build qualified name for mvar
                        let qualified = if new_path.is_empty() {
                            var_name.clone()
                        } else {
                            format!("{}.{}", new_path.join("."), var_name)
                        };
                        eng.get_mvar_value_raw(&qualified)
                    }
                };
                if let Some(value) = value_opt {
                    open_inspector(s, &var_name, value);
                } else {
                    log_to_repl(s, &format!("Unable to evaluate variable: {}", var_name));
                }
            }
            BrowserItem::Metadata { module } => {
                // Open metadata editor
                let full_name = format!("{}._meta", module);
                debug_log(&format!("Browser: selected Metadata for module: {}", module));
                s.pop_layer();
                open_editor(s, &full_name);
            }
        }
    });

    // Wrap in scroll view for long lists
    let select_scroll = select
        .with_name("browser_select")
        .scrollable()
        .fixed_size((45, 20));

    // Create preview pane (read-only code view)
    // Text wraps automatically for readability
    let preview = TextView::new("Select an item to preview")
        .with_name("browser_preview");
    let preview_scroll = preview
        .scrollable()
        .fixed_size((60, 20));

    // Create horizontal split: browser on left, preview on right
    let split_view = LinearLayout::horizontal()
        .child(Panel::new(select_scroll).title("Items"))
        .child(Panel::new(preview_scroll).title("Preview"));

    // Create dialog with navigation hints
    let dialog = Dialog::around(
        LinearLayout::vertical()
            .child(split_view)
            .child(TextView::new("Enter: Open | a: All | n: New | r: Rename | d: Delete | e: Error | g: Graph | Ctrl+F: Search | Esc: Close"))
    )
    .title(&title);

    // Wrap in OnEventView for keyboard navigation
    let path_for_back = path.clone();
    let path_for_new = path.clone();
    let path_for_rename = path.clone();
    let path_for_delete = path.clone();
    let path_for_error = path.clone();
    let path_for_search = path.clone();
    let engine_for_back = engine.clone();
    let engine_for_rename = engine.clone();
    let engine_for_delete = engine.clone();
    let engine_for_error = engine.clone();
    let engine_for_search = engine.clone();
    let path_for_graph = path.clone();
    let engine_for_graph = engine.clone();
    let dialog_with_keys = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| {
            s.pop_layer();
        })
        .on_event(Key::Left, move |s| {
            if !path_for_back.is_empty() {
                let engine = engine_for_back.clone();
                let mut new_path = path_for_back.clone();
                s.pop_layer();
                new_path.pop();
                show_browser_dialog(s, engine, new_path);
            }
        })
        .on_event('g', move |s| {
             let engine = engine_for_graph.clone();
             let path = path_for_graph.clone();

             let selected = s.call_on_name("browser_select", |v: &mut SelectView<BrowserItem>| {
                 v.selection().map(|rc| (*rc).clone())
             }).flatten();

             if let Some(item) = selected {
                 if let BrowserItem::Function { .. } = item {
                     let full_name = engine.borrow().get_full_name(&path, &item);
                     // Keep browser open underneath
                     show_call_graph_dialog(s, engine, full_name);
                 } else {
                     log_to_repl(s, "Call graph only available for functions");
                 }
             }
        })
        .on_event('a', {
            let engine = engine.clone();
            let path = path.clone();
            move |s| {
                // Show all module source in a view-only dialog
                let source = engine.borrow_mut().get_module_source(&path);
                let module_name = if path.is_empty() {
                    "root".to_string()
                } else {
                    path.join(".")
                };
                show_module_view_dialog(s, &module_name, &source);
            }
        })
        .on_event('n', move |s| {
            // Show dialog to enter new function name
            let path_for_create = path_for_new.clone();
            debug_log(&format!("New function dialog opened, current path: {:?}", path_for_create));

            let edit_view = EditView::new()
                .on_submit(move |s, name| {
                    let name = name.trim();
                    if name.is_empty() {
                        return;
                    }
                    // Build full name with module path
                    let full_name = if path_for_create.is_empty() {
                        name.to_string()
                    } else {
                        format!("{}.{}", path_for_create.join("."), name)
                    };
                    debug_log(&format!("Creating new function: name='{}', full_name='{}'", name, full_name));
                    s.pop_layer(); // Remove new function dialog
                    s.pop_layer(); // Remove browser dialog
                    open_editor(s, &full_name);
                })
                .with_name("new_func_name")
                .fixed_width(30);

            let dialog = Dialog::new()
                .title("New Function (Enter to create, Esc to cancel)")
                .content(
                    LinearLayout::vertical()
                        .child(TextView::new("Function name:"))
                        .child(edit_view)
                );

            let dialog_with_esc = OnEventView::new(dialog)
                .on_event(Key::Esc, |s| { s.pop_layer(); });

            s.add_layer(dialog_with_esc);
        })
        .on_event(Event::Char('r'), move |s| {
            // Get selected item from browser for rename
            let selected = s.call_on_name("browser_select", |v: &mut SelectView<BrowserItem>| {
                v.selection().map(|rc| (*rc).clone())
            }).flatten();

            if let Some(item) = selected {
                // Only allow renaming functions (for now)
                let name = match &item {
                    BrowserItem::Function { name, .. } => name.clone(),
                    _ => {
                        log_to_repl(s, "Only functions can be renamed");
                        return;
                    }
                };

                // Build full name
                let full_name = if path_for_rename.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", path_for_rename.join("."), name)
                };

                let engine_for_rename_submit = engine_for_rename.clone();
                let path_for_refresh = path_for_rename.clone();
                let old_name = full_name.clone();

                // Show dialog to enter new name
                let edit_view = EditView::new()
                    .content(&name) // Pre-fill with current name
                    .on_submit(move |s, new_name| {
                        let new_name = new_name.trim();
                        if new_name.is_empty() || new_name == name {
                            s.pop_layer();
                            return;
                        }

                        s.pop_layer(); // Remove rename dialog

                        // Perform rename
                        let rename_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            engine_for_rename_submit.borrow_mut().rename_definition(&old_name, new_name)
                        }));

                        match rename_result {
                            Ok(Ok((new_qualified, affected_callers))) => {
                                let mut msg = format!("Renamed '{}' to '{}'", old_name, new_qualified);
                                if !affected_callers.is_empty() {
                                    msg.push_str(&format!("\nAffected callers (need updating): {:?}", affected_callers));
                                }
                                log_to_repl(s, &msg);
                                // Refresh browser at the same path
                                s.pop_layer(); // Remove browser
                                show_browser_dialog(s, engine_for_rename_submit.clone(), path_for_refresh.clone());
                            }
                            Ok(Err(e)) => {
                                s.add_layer(Dialog::info(format!("Rename error: {}", e)));
                            }
                            Err(_) => {
                                s.add_layer(Dialog::info("Internal error during rename"));
                            }
                        }
                    })
                    .with_name("new_name")
                    .fixed_width(30);

                let dialog = Dialog::new()
                    .title(format!("Rename '{}' (Enter to rename, Esc to cancel)", full_name))
                    .content(
                        LinearLayout::vertical()
                            .child(TextView::new("New name:"))
                            .child(edit_view)
                    );

                let dialog_with_esc = OnEventView::new(dialog)
                    .on_event(Key::Esc, |s| { s.pop_layer(); });

                s.add_layer(dialog_with_esc);
            }
        })
        .on_event(Event::Char('d'), move |s| {
            // Get selected item from browser
            let selected = s.call_on_name("browser_select", |v: &mut SelectView<BrowserItem>| {
                v.selection().map(|rc| (*rc).clone())
            }).flatten();

            if let Some(item) = selected {
                // Only allow deleting functions, types, traits, variables
                let (name, kind) = match &item {
                    BrowserItem::Function { name, .. } => (name.clone(), "function"),
                    BrowserItem::Type { name, .. } => (name.clone(), "type"),
                    BrowserItem::Trait { name, .. } => (name.clone(), "trait"),
                    BrowserItem::Variable { name, .. } => (name.clone(), "variable"),
                    _ => return, // Can't delete modules or metadata
                };

                // Build full name
                let full_name = if path_for_delete.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", path_for_delete.join("."), name)
                };

                let engine_confirm = engine_for_delete.clone();
                let path_for_refresh = path_for_delete.clone();

                // Show confirmation dialog
                s.add_layer(
                    Dialog::text(format!("Delete {} '{}'?\n\nThis cannot be undone.", kind, full_name))
                        .title("Confirm Delete")
                        .button("Delete", move |s| {
                            s.pop_layer(); // Remove confirm dialog
                            // Perform delete - scope the borrow to avoid RefCell panic
                            // Also wrap in catch_unwind for stability
                            let delete_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                engine_confirm.borrow_mut().delete_definition(&full_name)
                            }));
                            let delete_result = match delete_result {
                                Ok(result) => result,
                                Err(_) => Err("Internal error during delete".to_string()),
                            };
                            match delete_result {
                                Ok(()) => {
                                    log_to_repl(s, &format!("Deleted {}", full_name));
                                    // Refresh browser at the same path
                                    s.pop_layer(); // Remove browser
                                    show_browser_dialog(s, engine_confirm.clone(), path_for_refresh.clone());
                                }
                                Err(e) => {
                                    s.add_layer(Dialog::info(format!("Error: {}", e)));
                                }
                            }
                        })
                        .button("Cancel", |s| { s.pop_layer(); })
                );
            }
        })
        .on_event(Event::Char('e'), move |s| {
            // Show error message for selected item
            let selected = s.call_on_name("browser_select", |v: &mut SelectView<BrowserItem>| {
                v.selection().map(|rc| (*rc).clone())
            }).flatten();

            if let Some(item) = selected {
                // Only functions can have errors
                if let BrowserItem::Function { name, .. } = &item {
                    let full_name = if path_for_error.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", path_for_error.join("."), name)
                    };

                    if let Some(error_msg) = engine_for_error.borrow().get_error_message(&full_name) {
                        s.add_layer(
                            Dialog::text(error_msg)
                                .title(format!("Error: {}", full_name))
                                .button("OK", |s| { s.pop_layer(); })
                        );
                    } else {
                        log_to_repl(s, &format!("{} has no errors", full_name));
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('f'), move |s| {
            // Open search dialog
            show_search_dialog(s, engine_for_search.clone(), path_for_search.clone());
        });

    s.add_layer(dialog_with_keys);
}

/// Show a read-only view of module source code with syntax highlighting
fn show_module_view_dialog(s: &mut Cursive, module_name: &str, source: &str) {
    // Apply syntax highlighting
    let styled_content = syntax_highlight_code(source);

    // Create a TextView with the styled content
    let text_view = TextView::new(styled_content);

    // Wrap in ScrollView with both horizontal and vertical scrolling
    let scroll_view = text_view
        .scrollable()
        .scroll_x(true)  // Enable horizontal scrolling
        .scroll_y(true); // Enable vertical scrolling

    // Create dialog with fixed size to show scrollbars
    let dialog = Dialog::around(
        scroll_view.fixed_size((80, 30))
    )
    .title(format!("Module: {} [Ctrl+Y to copy]", module_name))
    .button("Close", |s| { s.pop_layer(); });

    // Capture source for Ctrl+Y copy
    let source_for_copy = source.to_string();

    // Wrap in OnEventView for Esc to close and Ctrl+Y to copy
    let dialog_with_events = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| { s.pop_layer(); })
        .on_event(Event::CtrlChar('y'), move |s| {
            match copy_to_system_clipboard(&source_for_copy) {
                Ok(_) => log_to_repl(s, &format!("Copied {} chars to clipboard", source_for_copy.len())),
                Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
            }
        });

    s.add_layer(dialog_with_events);
}

/// Show search dialog for searching within a module
fn show_search_dialog(s: &mut Cursive, engine: Rc<RefCell<ReplEngine>>, path: Vec<String>) {
    let module_name = if path.is_empty() {
        "all modules".to_string()
    } else {
        path.join(".")
    };

    // Create search input
    let engine_for_search = engine.clone();
    let path_for_search = path.clone();

    let edit_view = EditView::new()
        .on_submit(move |s, query| {
            if query.trim().is_empty() {
                return;
            }

            // Perform search
            let results = engine_for_search.borrow().search_in_module(&path_for_search, query);

            if results.is_empty() {
                log_to_repl(s, &format!("No results found for '{}'", query));
                return;
            }

            // Show results dialog
            show_search_results_dialog(s, engine_for_search.clone(), results, query.to_string());
        })
        .with_name("search_input")
        .fixed_width(40);

    let dialog = Dialog::new()
        .title(format!("Search in {} [Enter to search, Esc to cancel]", module_name))
        .content(
            LinearLayout::vertical()
                .child(TextView::new("Search query:"))
                .child(edit_view)
        );

    let dialog_with_esc = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| { s.pop_layer(); });

    s.add_layer(dialog_with_esc);
}

/// Show search results dialog
fn show_search_results_dialog(s: &mut Cursive, engine: Rc<RefCell<ReplEngine>>, results: Vec<SearchResult>, query: String) {
    // Close the search input dialog
    s.pop_layer();

    let result_count = results.len();
    let mut select = SelectView::<SearchResult>::new();

    for result in results {
        // Build styled label with highlighted match
        let mut styled = StyledString::new();

        // Function name in cyan
        styled.append_styled(
            &result.function_name,
            Style::from(Color::Rgb(100, 200, 255))
        );

        if result.line_number == 0 {
            // Name match - just show the function name with highlight
            styled.append_plain(" (name match)");
        } else {
            // Line match - show line number and content with highlight
            styled.append_plain(&format!(":{} ", result.line_number));

            // Show line content with match highlighted
            let line = &result.line_content;
            let start = result.match_start;
            let end = result.match_end;

            // Before match
            if start > 0 {
                let before: String = line.chars().take(start).collect();
                styled.append_plain(&before);
            }

            // The match itself - highlighted in yellow
            let match_text: String = line.chars().skip(start).take(end - start).collect();
            styled.append_styled(
                &match_text,
                Style::from(ColorStyle::new(Color::Rgb(0, 0, 0), Color::Rgb(255, 255, 0)))
            );

            // After match
            let after: String = line.chars().skip(end).collect();
            if !after.is_empty() {
                styled.append_plain(&after);
            }
        }

        select.add_item(styled, result);
    }

    // Handle selection - open function in editor
    let engine_for_submit = engine.clone();
    select.set_on_submit(move |s, result: &SearchResult| {
        let func_name = result.function_name.clone();
        s.pop_layer(); // Close results dialog
        s.pop_layer(); // Close browser dialog
        open_editor(s, &func_name);
    });

    let select_scroll = select
        .with_name("search_results")
        .scrollable()
        .fixed_size((80, 20));

    let dialog = Dialog::around(select_scroll)
        .title(format!("Search results for '{}' ({} hits)", query, result_count))
        .button("Close", |s| { s.pop_layer(); });

    let dialog_with_esc = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| { s.pop_layer(); });

    s.add_layer(dialog_with_esc);
}

/// Open the value inspector dialog
fn open_inspector(s: &mut Cursive, var_name: &str, value: Value) {
    let inspector = Rc::new(RefCell::new(Inspector::new(var_name.to_string(), value)));
    show_inspector_dialog(s, inspector);
}

/// Show the inspector dialog at the current navigation position
fn show_inspector_dialog(s: &mut Cursive, inspector: Rc<RefCell<Inspector>>) {
    let inspector_borrow = inspector.borrow();
    let path_str = inspector_borrow.path_string();

    let inspect_result = match inspector_borrow.inspect_current() {
        Some(r) => r,
        None => {
            drop(inspector_borrow);
            log_to_repl(s, "Error: Invalid inspection path");
            return;
        }
    };

    // Build title with type info
    let title = format!("Inspector: {} :: {}", path_str, inspect_result.type_name);

    // Build the content
    let mut select = SelectView::<Slot>::new();

    // Show ".." to go up if not at root
    if inspector_borrow.depth() > 0 {
        select.add_item("⬆️  ..", Slot::Field("..".to_string()));
    }

    // If it's a leaf, show the full value
    if inspect_result.is_leaf {
        drop(inspector_borrow);

        let content = if let Some(count) = inspect_result.total_count {
            format!("{}\n\n({} items)", inspect_result.preview, count)
        } else {
            inspect_result.preview.clone()
        };

        let dialog = Dialog::around(
            LinearLayout::vertical()
                .child(TextView::new(format!("Path: {}", path_str)))
                .child(TextView::new(format!("Type: {}", inspect_result.type_name)))
                .child(TextView::new(""))
                .child(TextView::new(content).scrollable().fixed_height(15))
                .child(TextView::new("Left: Go up | Esc: Close"))
        )
        .title("Inspector");

        let inspector_for_back = inspector.clone();
        let dialog_with_keys = OnEventView::new(dialog)
            .on_event(Key::Esc, |s| {
                s.pop_layer();
            })
            .on_event(Key::Left, move |s| {
                s.pop_layer();
                if inspector_for_back.borrow_mut().navigate_up() {
                    show_inspector_dialog(s, inspector_for_back.clone());
                }
            });

        s.add_layer(dialog_with_keys);
        return;
    }

    // Show slots
    let page_offset = inspector_borrow.page_offset;
    let page_size = inspector_borrow.page_size;
    let total_slots = inspect_result.slots.len();

    for slot_info in &inspect_result.slots {
        let slot = slot_info.slot.clone();
        let is_cycle = inspector_borrow.would_cycle(&slot);

        let icon = if is_cycle {
            "🔄"
        } else if slot_info.is_leaf {
            "📄"
        } else {
            "📁"
        };

        let label = match &slot_info.slot {
            Slot::Field(name) => format!("{} .{}: {} = {}", icon, name, slot_info.value_type, slot_info.preview),
            Slot::Index(i) => format!("{} [{}]: {} = {}", icon, i, slot_info.value_type, slot_info.preview),
            Slot::MapKey(k) => format!("{} [{}]: {} = {}", icon, k.display(), slot_info.value_type, slot_info.preview),
            Slot::VariantField(i) => format!("{} .{}: {} = {}", icon, i, slot_info.value_type, slot_info.preview),
            Slot::VariantNamedField(name) => format!("{} .{}: {} = {}", icon, name, slot_info.value_type, slot_info.preview),
        };

        select.add_item(label, slot);
    }

    drop(inspector_borrow);

    // Handle selection
    let inspector_for_select = inspector.clone();
    select.set_on_submit(move |s, slot: &Slot| {
        // Handle ".." navigation
        if matches!(slot, Slot::Field(name) if name == "..") {
            s.pop_layer();
            inspector_for_select.borrow_mut().navigate_up();
            show_inspector_dialog(s, inspector_for_select.clone());
            return;
        }

        // Check for cycle
        if inspector_for_select.borrow().would_cycle(slot) {
            log_to_repl(s, "Cannot navigate: cycle detected");
            return;
        }

        // Navigate into the slot
        s.pop_layer();
        if let Err(e) = inspector_for_select.borrow_mut().navigate_to(slot.clone()) {
            log_to_repl(s, &format!("Navigation error: {}", e));
        }
        show_inspector_dialog(s, inspector_for_select.clone());
    });

    let select_scroll = select
        .with_name("inspector_select")
        .scrollable()
        .fixed_size((70, 18));

    // Pagination info
    let pagination_info = if total_slots > page_size {
        format!("Showing {}-{} of {} | PgUp/PgDn: Navigate pages",
            page_offset + 1,
            (page_offset + page_size).min(total_slots),
            total_slots)
    } else {
        "Enter/Right: Drill in | Left: Go up | Esc: Close".to_string()
    };

    let dialog = Dialog::around(
        LinearLayout::vertical()
            .child(TextView::new(format!("Path: {}", path_str)))
            .child(TextView::new(format!("Type: {} {}", inspect_result.type_name,
                inspect_result.total_count.map(|c| format!("({} items)", c)).unwrap_or_default())))
            .child(TextView::new(""))
            .child(select_scroll)
            .child(TextView::new(pagination_info))
    )
    .title(&title);

    // Wrap for keyboard navigation
    let inspector_for_back = inspector.clone();
    let inspector_for_pgup = inspector.clone();
    let inspector_for_pgdn = inspector.clone();

    let dialog_with_keys = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| {
            // Close inspector, browser underneath will be revealed
            s.pop_layer();
        })
        .on_event(Key::Left, move |s| {
            s.pop_layer();
            if inspector_for_back.borrow_mut().navigate_up() {
                show_inspector_dialog(s, inspector_for_back.clone());
            }
        })
        .on_event(Key::PageUp, move |s| {
            inspector_for_pgup.borrow_mut().prev_page();
            s.pop_layer();
            show_inspector_dialog(s, inspector_for_pgup.clone());
        })
        .on_event(Key::PageDown, move |s| {
            inspector_for_pgdn.borrow_mut().next_page();
            s.pop_layer();
            show_inspector_dialog(s, inspector_for_pgdn.clone());
        });

    s.add_layer(dialog_with_keys);
}

/// Open a variable viewer in the workspace
fn open_variable_viewer(s: &mut Cursive, name: &str, mutable: bool, value: &str) {
    let viewer_name = format!("var_{}", name);

    // Check limit and add to state
    let can_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if state.open_editors.contains(&viewer_name) {
            return Err("Viewer already open");
        }
        if state.open_editors.len() >= 11 {
            return Err("Max 12 windows");
        }
        state.open_editors.push(viewer_name.clone());
        Ok(())
    }).unwrap();

    if let Err(msg) = can_open {
        log_to_repl(s, msg);
        return;
    }

    // Format the display text
    let var_kind = if mutable { "var" } else { "val" };
    let display_text = format!("{} {} = {}", var_kind, name, value);

    // Rebuild workspace with the new viewer
    rebuild_workspace_with_viewer(s, &viewer_name, &display_text);

    // Focus the new viewer
    s.focus_name(&viewer_name).ok();
}

/// Rebuild workspace including a variable viewer
fn rebuild_workspace_with_viewer(s: &mut Cursive, viewer_name: &str, viewer_content: &str) {
    let (editor_names, engine) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let state = state.borrow();
        (state.open_editors.clone(), state.engine.clone())
    }).unwrap();

    // Get console content BEFORE clearing workspace
    let repl_log_content: String = s.call_on_name("repl_log", |view: &mut FocusableConsole| {
        view.get_content()
    }).unwrap_or_default();

    // Remove old workspace content
    s.call_on_name("workspace", |ws: &mut LinearLayout| {
        ws.clear();
    });

    // Total windows = Console + editors/viewers
    let total_windows = 1 + editor_names.len();

    let repl_log = FocusableConsole::new(
        TextView::new(repl_log_content).scrollable().scroll_x(true)
    ).with_name("repl_log");

    let repl_log_with_events = OnEventView::new(repl_log)
        .on_event(Event::CtrlChar('y'), |s| {
            if let Some(text) = s.call_on_name("repl_log", |view: &mut FocusableConsole| {
                view.get_content()
            }) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        });

    // Console is smaller and shrinks first - use max_width to limit size
    let console = ActiveWindow::new(repl_log_with_events, "Console").max_width(60);

    if total_windows <= 3 {
        let mut row = LinearLayout::horizontal().child(console);

        for name in &editor_names {
            if name == viewer_name {
                // Create viewer
                let view = create_variable_viewer(name, viewer_content);
                row.add_child(view);
            } else if name.starts_with("var_") {
                // Existing viewer - skip for now (would need stored content)
                continue;
            } else {
                let read_only = engine.borrow().is_eval_function(name);
                let editor_view = create_editor_view(s, &engine, name, read_only);
                row.add_child(editor_view);
            }
        }

        s.call_on_name("workspace", |ws: &mut LinearLayout| {
            ws.add_child(row.full_width().full_height().with_name("workspace_row_0"));
        });
    } else {
        let mut row0 = LinearLayout::horizontal().child(console);
        let mut row1 = LinearLayout::horizontal();

        for (i, name) in editor_names.iter().enumerate() {
            let read_only = engine.borrow().is_eval_function(name);
            let view: Box<dyn View> = if name == viewer_name {
                Box::new(create_variable_viewer(name, viewer_content))
            } else if name.starts_with("var_") {
                continue;
            } else {
                Box::new(create_editor_view(s, &engine, name, read_only))
            };

            if i < 2 {
                row0.add_child(view);
            } else {
                row1.add_child(view);
            }
        }

        s.call_on_name("workspace", |ws: &mut LinearLayout| {
            ws.add_child(row0.full_width().full_height().with_name("workspace_row_0"));
            ws.add_child(row1.full_width().full_height().with_name("workspace_row_1"));
        });
    }
}

/// Create a read-only variable viewer
fn create_variable_viewer(viewer_name: &str, content: &str) -> impl View {
    // Extract the variable name from viewer_name (remove "var_" prefix)
    let display_name = viewer_name.strip_prefix("var_").unwrap_or(viewer_name);

    let text_view = TextView::new(content)
        .scrollable()
        .with_name(viewer_name);

    let name_for_close = viewer_name.to_string();
    let name_for_close_w = viewer_name.to_string();
    let viewer_id_copy = viewer_name.to_string();

    // Ctrl+W to close, Ctrl+Y to copy, Esc to close
    let viewer_with_events = OnEventView::new(text_view)
        .on_event(Event::CtrlChar('y'), move |s| {
            if let Some(view) = s.call_on_name(&viewer_id_copy, |v: &mut ScrollView<TextView>| {
                v.get_inner().get_content().source().to_string()
            }) {
                if !view.is_empty() {
                    match copy_to_system_clipboard(&view) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", view.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        })
        .on_event(Event::CtrlChar('w'), move |s| {
            close_viewer(s, &name_for_close_w);
        })
        .on_event(Key::Esc, move |s| {
            close_viewer(s, &name_for_close);
        })
        .on_pre_event_inner(Event::Shift(Key::Tab), |_, _| {
            Some(EventResult::Consumed(None))
        });

    ActiveWindow::new(viewer_with_events.full_height(), display_name).full_width()
}

/// Close a variable viewer
fn close_viewer(s: &mut Cursive, viewer_name: &str) {
    let was_removed = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if let Some(idx) = state.open_editors.iter().position(|x| x == viewer_name) {
            state.open_editors.remove(idx);
            true
        } else {
            false
        }
    }).unwrap();

    if was_removed {
        rebuild_workspace(s);
        s.focus_name("repl_log").ok();
    }
}

fn cycle_window(s: &mut Cursive) {
    let target = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        let mut windows = Vec::new();

        // Console (always first if open)
        if state.console_open {
            windows.push("repl_log".to_string());
        }

        // Editors
        for name in &state.open_editors {
            windows.push(format!("editor_{}", name));
        }

        // REPL panels
        for id in &state.open_repls {
            windows.push(format!("repl_panel_{}", id));
        }

        // Inspector
        if state.inspector_open {
            windows.push("inspector_panel".to_string());
        }

        // Nostos panel
        if state.nostos_panel_open {
            windows.push("nostos_mvar_panel".to_string());
        }

        if windows.is_empty() {
            return "repl_log".to_string();
        }

        state.active_window_idx = (state.active_window_idx + 1) % windows.len();
        windows[state.active_window_idx].clone()
    }).unwrap();

    s.focus_name(&target).ok();
}

fn cycle_window_backward(s: &mut Cursive) {
    let target = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        let mut windows = Vec::new();

        // Console (always first if open)
        if state.console_open {
            windows.push("repl_log".to_string());
        }

        // Editors
        for name in &state.open_editors {
            windows.push(format!("editor_{}", name));
        }

        // REPL panels
        for id in &state.open_repls {
            windows.push(format!("repl_panel_{}", id));
        }

        // Inspector
        if state.inspector_open {
            windows.push("inspector_panel".to_string());
        }

        // Nostos panel
        if state.nostos_panel_open {
            windows.push("nostos_mvar_panel".to_string());
        }

        if windows.is_empty() {
            return "repl_log".to_string();
        }

        // Go backward with wrapping
        if state.active_window_idx == 0 {
            state.active_window_idx = windows.len() - 1;
        } else {
            state.active_window_idx -= 1;
        }
        windows[state.active_window_idx].clone()
    }).unwrap();

    s.focus_name(&target).ok();
}

fn close_active_editor(s: &mut Cursive) {
    // Get active window info
    let (active_idx, editor_names) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let state = state.borrow();
        (state.active_window_idx, state.open_editors.clone())
    }).unwrap_or((0, vec![]));

    // Build window list (same order as cycle_window)
    let mut windows = vec!["repl_log".to_string()];
    for name in &editor_names {
        windows.push(format!("editor_{}", name));
    }

    let focused_window = windows.get(active_idx % windows.len()).cloned().unwrap_or_default();

    // Only close if it's an editor window
    if focused_window.starts_with("editor_") {
        let name = focused_window.strip_prefix("editor_").unwrap().to_string();
        close_editor(s, &name);
    }
}

fn copy_focused_window(s: &mut Cursive) {
    // Get the currently active window from tracked index
    let (active_idx, editor_names) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let state = state.borrow();
        (state.active_window_idx, state.open_editors.clone())
    }).unwrap_or((0, vec![]));

    // Build window list (same order as cycle_window)
    let mut windows = vec!["repl_log".to_string()];
    for name in &editor_names {
        windows.push(format!("editor_{}", name));
    }

    let focused_window = windows.get(active_idx % windows.len()).cloned().unwrap_or_default();

    // Get content based on focused window
    let content: Option<String> = if focused_window == "repl_log" {
        s.call_on_name("repl_log", |view: &mut FocusableConsole| {
            view.get_content()
        })
    } else if focused_window.starts_with("editor_") {
        s.call_on_name(&focused_window, |view: &mut crate::editor::CodeEditor| {
            view.get_content()
        })
    } else {
        None
    };

    // Copy to clipboard silently (no logging to avoid console resize)
    if let Some(text) = content {
        if !text.is_empty() {
            let _ = copy_to_system_clipboard(&text);
        }
    }
}

/// Copy text to system clipboard using xclip (X11) or wl-copy (Wayland)
fn copy_to_system_clipboard(text: &str) -> Result<String, String> {
    use std::process::{Command, Stdio};
    use std::io::Write;

    // Try wl-copy first (Wayland), then xclip (X11)
    let commands = [
        ("wl-copy", vec![]),
        ("xclip", vec!["-selection", "clipboard"]),
        ("xsel", vec!["--clipboard", "--input"]),
    ];

    for (cmd, args) in &commands {
        if let Ok(mut child) = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                if stdin.write_all(text.as_bytes()).is_ok() {
                    drop(stdin);
                    if let Ok(status) = child.wait() {
                        if status.success() {
                            return Ok(cmd.to_string());
                        }
                    }
                }
            }
        }
    }

    Err("No clipboard tool found (tried wl-copy, xclip, xsel)".to_string())
}

fn style_input(text: &str) -> StyledString {
    let mut styled = StyledString::new();
    styled.append_styled("nos> ", Color::Rgb(0, 255, 0));

    let mut last_idx = 0;
    for (token, span) in lex(text) {
        if span.start > last_idx {
            styled.append_plain(&text[last_idx..span.start]);
        }

        let color = match token {
            Token::Type | Token::Var | Token::Mvar | Token::If | Token::Then | Token::Else |
            Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
            Token::Use | Token::Import | Token::Private | Token::Pub | Token::SelfKw | Token::SelfType |
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
            Token::LBrace | Token::RBrace | Token::Comma | Token::Semicolon | Token::Colon | Token::ColonColon | Token::Dot |
            Token::Pipe | Token::Hash =>
                Color::Rgb(255, 165, 0),

            Token::UpperIdent(_) => Color::Rgb(255, 255, 0),
            Token::LowerIdent(_) => Color::Rgb(255, 255, 255),

            Token::Underscore => Color::Rgb(150, 150, 150), // Lighter gray - visible on dark backgrounds
            Token::Newline => Color::TerminalDefault,
            Token::Comment | Token::MultiLineComment => Color::Rgb(150, 150, 150), // Lighter gray
        };

        styled.append_styled(&text[span.clone()], color);
        last_idx = span.end;
    }

    if last_idx < text.len() {
        styled.append_plain(&text[last_idx..]);
    }

    styled
}

fn handle_command(engine: &mut ReplEngine, line: &str) -> String {
    let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0];
    let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ":help" | ":h" | ":?" => {
            "Commands:
  :help, :h            Show this help
  :quit, :q            Exit
  :edit <name>, :e     Edit function
  :load <file>, :l     Load file
  :reload, :r          Reload files
  :browse, :b          Open module browser
  :info <name>, :i     Show info
  :view <name>, :v     Show source
  :type <expr>, :t     Show type (not impl)
  :deps <fn>, :d       Show dependencies
  :rdeps <fn>, :rd     Show reverse deps
  :functions, :fns     List functions
  :types               List types
  :traits              List traits
  :vars                List variables
  :copy, :cp           Copy console to clipboard
  :w, :write           Write module files from defs

Keyboard shortcuts:
  Shift+Tab            Cycle between windows
  Ctrl+Y               Copy focused window to clipboard
  Ctrl+W               Close active editor
  Ctrl+S               Save editor (when in editor)
  Ctrl+G               View call graph (when in editor)
  Esc                  Close editor (when in editor)".to_string()
        }
        ":load" | ":l" => {
            if args.is_empty() { return "Usage: :load <file.nos>".to_string(); }
            match engine.load_file(args) {
                Ok(_) => format!("Loaded {}", args),
                Err(e) => e,
            }
        }
        ":reload" | ":r" => {
            match engine.reload_files() {
                Ok(count) => format!("Reloaded {} file(s)", count),
                Err(e) => e,
            }
        }
        ":browse" | ":b" => engine.browse(if args.is_empty() { None } else { Some(args) }),
        ":info" | ":i" => if args.is_empty() { "Usage: :info <name>".to_string() } else { engine.get_info(args) },
        ":view" | ":v" => if args.is_empty() { "Usage: :view <name>".to_string() } else { engine.get_source(args) },
        ":type" | ":t" => {
             let type_info = engine.get_info(args);
             if !type_info.starts_with("Not found") {
                 type_info
             } else {
                 "Type inference for expressions not yet implemented. Use :info <name> to see types of functions/definitions.".to_string()
             }
        },
        ":deps" | ":d" => if args.is_empty() { "Usage: :deps <name>".to_string() } else { engine.get_deps(args) },
        ":rdeps" | ":rd" => if args.is_empty() { "Usage: :rdeps <name>".to_string() } else { engine.get_rdeps(args) },
        ":functions" | ":fns" => {
            let fns = engine.get_functions();
            if fns.is_empty() { "No functions defined".to_string() } else { fns.join("\n") }
        }
        ":types" => {
            let ts = engine.get_types();
            if ts.is_empty() { "No types defined".to_string() } else { ts.join("\n") }
        }
        ":traits" => {
            let ts = engine.get_traits();
            if ts.is_empty() { "No traits defined".to_string() } else { ts.join("\n") }
        }
        ":module" | ":m" => if args.is_empty() {
            format!("Current module: {}", engine.get_current_module())
        } else {
            engine.switch_module(args)
        },
        ":vars" | ":bindings" => engine.get_vars(),
        ":w" | ":write" => {
            match engine.write_module_files() {
                Ok(0) => "No changes to write".to_string(),
                Ok(1) => "1 module file written".to_string(),
                Ok(n) => format!("{} module files written", n),
                Err(e) => e,
            }
        }
        _ => format!("Unknown command: {}", cmd),
    }
}

/// Show call graph dialog for a function
fn show_call_graph_dialog(s: &mut Cursive, engine: Rc<RefCell<ReplEngine>>, name: String) {
    let engine_ref = engine.borrow();

    // Get Callers (Dependents) - Only project functions
    let mut callers: Vec<String> = engine_ref.call_graph.direct_dependents(&name)
        .into_iter()
        .filter(|n| engine_ref.is_project_function(n))
        .collect();
    callers.sort();

    // Get Callees (Dependencies) - Only project functions
    let mut callees: Vec<String> = engine_ref.call_graph.direct_dependencies(&name)
        .into_iter()
        .filter(|n| engine_ref.is_project_function(n))
        .collect();
    callees.sort();

    // Create SelectViews
    let mut callers_view = SelectView::new();
    for caller in &callers {
        callers_view.add_item(caller.clone(), caller.clone());
    }

    let mut callees_view = SelectView::new();
    for callee in &callees {
        callees_view.add_item(callee.clone(), callee.clone());
    }

    drop(engine_ref); // Release borrow

    // Shared event handler for selection (open editor)
    let handler = move |s: &mut Cursive, selected: &String| {
        s.pop_layer(); // Close graph dialog
        open_editor(s, selected);
    };

    // Shared event handler for navigation (g key - recursive graph)
    let engine_nav = engine.clone();
    let nav_handler = move |s: &mut Cursive, selected: &String| {
         s.pop_layer(); // Close current graph dialog
         show_call_graph_dialog(s, engine_nav.clone(), selected.clone());
    };

    // Bind handlers
    callers_view.set_on_submit(handler.clone());
    callees_view.set_on_submit(handler);

    // Wrap in scroll views
    let callers_scroll = callers_view.with_name("graph_callers").scrollable().fixed_size((40, 20));
    let callees_scroll = callees_view.with_name("graph_callees").scrollable().fixed_size((40, 20));

    // Layout
    let layout = LinearLayout::horizontal()
        .child(Panel::new(callers_scroll).title("Callers (Incoming)"))
        .child(Panel::new(callees_scroll).title("Callees (Outgoing)"));

    // Dialog
    let dialog = Dialog::around(layout)
        .title(format!("Call Graph: {}", name));

    // Add navigation keys
    let dialog_with_events = OnEventView::new(dialog)
        .on_event(Key::Esc, |s| { s.pop_layer(); })
        .on_event(Key::Left, |s| { s.focus_name("graph_callers").ok(); })
        .on_event(Key::Right, |s| { s.focus_name("graph_callees").ok(); })
        .on_event('g', move |s| {
            // Check which view has focus and get selection
            let selected = if s.focus_name("graph_callers").is_ok() {
                 s.call_on_name("graph_callers", |v: &mut SelectView<String>| v.selection()).flatten()
            } else {
                 s.call_on_name("graph_callees", |v: &mut SelectView<String>| v.selection()).flatten()
            };

            if let Some(sel) = selected {
                 nav_handler(s, &sel);
            }
        });

    s.add_layer(dialog_with_events);
}
