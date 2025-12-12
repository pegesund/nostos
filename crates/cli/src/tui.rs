//! TUI implementation for Nostos

use cursive::Cursive;
use cursive::traits::*;
use cursive::views::{Dialog, EditView, LinearLayout, ScrollView, TextView, OnEventView};
use cursive::theme::{Color, PaletteColor, Theme, BorderStyle};
use cursive::view::Resizable;
use cursive::utils::markup::StyledString;
use cursive::event::{Event, EventResult, Key};
use nostos_repl::{ReplEngine, ReplConfig};
use nostos_syntax::lexer::{Token, lex};
use std::cell::RefCell;
use std::rc::Rc;
use std::process::ExitCode;
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
    active_window_idx: usize,
    engine: Rc<RefCell<ReplEngine>>,
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
    theme.palette[PaletteColor::Highlight] = Color::Rgb(255, 255, 0); // Yellow highlight
    theme.palette[PaletteColor::Secondary] = Color::TerminalDefault;
    siv.set_theme(theme);

    // Initialize REPL engine
    let config = ReplConfig::default();
    let mut engine = ReplEngine::new(config);
    if let Err(e) = engine.load_stdlib() {
        eprintln!("Failed to load stdlib: {}", e);
    }

    // Load files
    for arg in args {
        if !arg.starts_with('-') {
            if let Err(e) = engine.load_file(arg) {
                eprintln!("Error loading {}: {}", arg, e);
            }
        }
    }

    let engine = Rc::new(RefCell::new(engine));

    // Initialize State
    siv.set_user_data(Rc::new(RefCell::new(TuiState {
        open_editors: Vec::new(),
        active_window_idx: 0,
        engine: engine.clone(),
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

    // 2. Workspace - starts with just Console (full width and height)
    let workspace = LinearLayout::vertical()
        .child(
            LinearLayout::horizontal()
                .child(ActiveWindow::new(repl_log_with_events, "Console").full_width())
                .full_width()
                .full_height()
                .with_name("workspace_row_0")
        )
        .with_name("workspace");

    // 3. Input
    let engine_clone = engine.clone();
    let input_view = EditView::new()
        .on_submit(move |s, text| {
            let text = text.trim();
            if text.is_empty() { return; }
            let input_text = text.to_string();

            s.call_on_name("input", |view: &mut EditView| {
                view.set_content("");
            });

            if input_text == ":quit" || input_text == ":q" || input_text == ":exit" {
                s.quit();
                return;
            }

            // Handle :edit
            if input_text.starts_with(":edit") || input_text.starts_with(":e ") {
                let parts: Vec<&str> = input_text.splitn(2, char::is_whitespace).collect();
                let name = parts.get(1).map(|s| s.trim()).unwrap_or("");
                if name.is_empty() {
                    log_to_repl(s, "Usage: :edit <name>");
                } else {
                    open_editor(s, name);
                }
                return;
            }

            // Handle :copy
            if input_text == ":copy" || input_text == ":cp" {
                copy_focused_window(s);
                return;
            }

            // Handle :debug - show what command was received
            if input_text == ":debug" {
                log_to_repl(s, "Debug: TUI commands are working");
                return;
            }

            // Echo
            s.call_on_name("repl_log", |view: &mut FocusableConsole| {
                view.append_styled(style_input(&input_text));
                view.append("\n");
            });

            // Eval
            let result = if input_text.starts_with(':') {
                handle_command(&mut engine_clone.borrow_mut(), &input_text)
            } else {
                match engine_clone.borrow_mut().eval(&input_text) {
                    Ok(output) => {
                        // Debug: log to file to see what's returned
                        let _ = std::fs::write("/var/tmp/nostos_eval_result.txt", &output);
                        output
                    }
                    Err(e) => format!("Error: {}", e),
                }
            };

            // Always log something to confirm the flow works
            if result.is_empty() {
                log_to_repl(s, "(no output)");
            } else {
                log_to_repl(s, &result);
            }
        })
        .with_name("input");

    // Wrap input with OnEventView for Ctrl+Y copy (same pattern as Ctrl+S on editor)
    let input_with_events = OnEventView::new(input_view)
        .on_event(Event::CtrlChar('y'), |s| {
            if let Some(text) = s.call_on_name("input", |view: &mut EditView| {
                view.get_content().to_string()
            }) {
                if !text.is_empty() {
                    match copy_to_system_clipboard(&text) {
                        Ok(_) => log_to_repl(s, &format!("Copied {} chars", text.len())),
                        Err(e) => log_to_repl(s, &format!("Copy failed: {}", e)),
                    }
                }
            }
        });

    // Root Layout - Input always has full width
    let root_layout = LinearLayout::vertical()
        .child(workspace.full_width().full_height())
        .child(ActiveWindow::new(input_with_events, "Input").full_width().fixed_height(3));

    siv.add_layer(root_layout);

    // Focus input at start
    siv.focus_name("input").ok();

    // Global Shift+Tab for window cycling (this one needs to be global)
    siv.set_on_pre_event(Event::Shift(Key::Tab), |s| {
        cycle_window(s);
    });

    siv.run();
    ExitCode::SUCCESS
}

fn log_to_repl(s: &mut Cursive, text: &str) {
    s.call_on_name("repl_log", |view: &mut FocusableConsole| {
        view.append(&format!("{}\n", text));
    });
}

/// Rebuild the workspace layout based on current windows
/// Layout rules:
/// - 1-3 windows: single row, equal width
/// - 4-6 windows: two rows, top row has 3 windows, bottom row has rest
fn rebuild_workspace(s: &mut Cursive) {
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

    // Total windows = Console + editors
    let total_windows = 1 + editor_names.len();

    let repl_log = FocusableConsole::new(
        TextView::new(repl_log_content).scrollable()
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

    let console = ActiveWindow::new(repl_log_with_events, "Console").full_width();

    if total_windows <= 3 {
        // Single row layout
        let mut row = LinearLayout::horizontal().child(console);

        for name in &editor_names {
            let editor_view = create_editor_view(s, &engine, name);
            row.add_child(editor_view);
        }

        s.call_on_name("workspace", |ws: &mut LinearLayout| {
            ws.add_child(row.full_width().full_height().with_name("workspace_row_0"));
        });
    } else {
        // Two row layout: first 2 editors + console on top, rest on bottom
        let mut row0 = LinearLayout::horizontal().child(console);
        let mut row1 = LinearLayout::horizontal();

        for (i, name) in editor_names.iter().enumerate() {
            let editor_view = create_editor_view(s, &engine, name);
            if i < 2 {
                row0.add_child(editor_view);
            } else {
                row1.add_child(editor_view);
            }
        }

        s.call_on_name("workspace", |ws: &mut LinearLayout| {
            ws.add_child(row0.full_width().full_height().with_name("workspace_row_0"));
            ws.add_child(row1.full_width().full_height().with_name("workspace_row_1"));
        });
    }
}

/// Create an editor view for a given name
fn create_editor_view(_s: &mut Cursive, engine: &Rc<RefCell<ReplEngine>>, name: &str) -> impl View {
    let source = engine.borrow().get_source(name);
    let source = if source.starts_with("Not found") {
        format!("{}() = {{\n    \n}}", name)
    } else {
        source
    };

    let editor = CodeEditor::new(source).with_engine(engine.clone());
    let editor_id = format!("editor_{}", name);

    let name_for_save = name.to_string();
    let name_for_close = name.to_string();
    let name_for_close_w = name.to_string();
    let engine_save = engine.clone();
    let editor_id_save = editor_id.clone();
    let editor_id_copy = editor_id.clone();

    // Ctrl+S to save, Ctrl+W to close, Ctrl+Y to copy, Esc to close
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
            // Close this editor
            close_editor(s, &name_for_close_w);
        })
        .on_event(Event::CtrlChar('s'), move |s| {
            let content = match s.call_on_name(&editor_id_save, |v: &mut CodeEditor| v.get_content()) {
                Some(c) => c,
                None => {
                    log_to_repl(s, &format!("Error: Could not find editor {}", editor_id_save));
                    return;
                }
            };

            match engine_save.borrow_mut().eval(&content) {
                Ok(output) => {
                    if output.is_empty() {
                        log_to_repl(s, &format!("Saved {} (no definitions found)", name_for_save));
                    } else {
                        log_to_repl(s, &output);
                    }
                    close_editor(s, &name_for_save);
                }
                Err(e) => {
                    s.add_layer(Dialog::info(format!("Error: {}", e)));
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

    // Wrap in ActiveWindow with just the function name as title
    ActiveWindow::new(editor_with_events.full_height(), name).full_width()
}

fn open_editor(s: &mut Cursive, name: &str) {
    let name_owned = name.to_string();

    // Check limit and add to state
    let can_open = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        if state.open_editors.contains(&name_owned) {
            return Err("Editor already open");
        }
        if state.open_editors.len() >= 5 { // 5 editors + 1 console = 6 windows max
            return Err("Max 6 windows");
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
    s.focus_name(&editor_id).ok();
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

        // Focus input
        s.focus_name("input").ok();
    }
}

fn cycle_window(s: &mut Cursive) {
    let target = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let mut state = state.borrow_mut();
        let mut windows = vec!["input".to_string(), "repl_log".to_string()];

        for name in &state.open_editors {
            windows.push(format!("editor_{}", name));
        }

        state.active_window_idx = (state.active_window_idx + 1) % windows.len();
        windows[state.active_window_idx].clone()
    }).unwrap();

    // Now console is focusable via FocusableConsole wrapper
    s.focus_name(&target).ok();
}

fn close_active_editor(s: &mut Cursive) {
    // Get active window info
    let (active_idx, editor_names) = s.with_user_data(|state: &mut Rc<RefCell<TuiState>>| {
        let state = state.borrow();
        (state.active_window_idx, state.open_editors.clone())
    }).unwrap_or((0, vec![]));

    // Build window list (same order as cycle_window)
    let mut windows = vec!["input".to_string(), "repl_log".to_string()];
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
    let mut windows = vec!["input".to_string(), "repl_log".to_string()];
    for name in &editor_names {
        windows.push(format!("editor_{}", name));
    }

    let focused_window = windows.get(active_idx % windows.len()).cloned().unwrap_or_default();

    // Get content based on focused window
    let content: Option<String> = if focused_window == "input" {
        s.call_on_name("input", |view: &mut EditView| {
            view.get_content().to_string()
        })
    } else if focused_window == "repl_log" {
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
            Token::Newline => Color::TerminalDefault,
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
  :browse [mod], :b    List functions
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

Keyboard shortcuts:
  Shift+Tab            Cycle between windows
  Ctrl+Y               Copy focused window to clipboard
  Ctrl+W               Close active editor
  Ctrl+S               Save editor (when in editor)
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
        _ => format!("Unknown command: {}", cmd),
    }
}
