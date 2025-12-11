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
use crate::custom_views::ActiveWindow;

struct TuiState {
    open_editors: Vec<String>,
    active_window_idx: usize,
    engine: Rc<RefCell<ReplEngine>>,
}

pub fn run_tui(args: &[String]) -> ExitCode {
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
    let repl_log = TextView::new(format!("Nostos TUI v{}\nType :help for commands\n\n", env!("CARGO_PKG_VERSION")))
        .scrollable()
        .with_name("repl_log");

    // 2. Workspace - starts with just Console (full width and height)
    let workspace = LinearLayout::vertical()
        .child(
            LinearLayout::horizontal()
                .child(ActiveWindow::new(repl_log, "Console").full_width())
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

            // Echo
            s.call_on_name("repl_log", |view: &mut ScrollView<TextView>| {
                let text_view = view.get_inner_mut();
                text_view.append(style_input(&input_text));
                text_view.append("\n");
                view.scroll_to_bottom();
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

    // Root Layout - Input always has full width
    let root_layout = LinearLayout::vertical()
        .child(workspace.full_width().full_height())
        .child(ActiveWindow::new(input_view, "Input").full_width().fixed_height(3));

    siv.add_layer(root_layout);

    // Global Window Cycling with Shift+Tab
    siv.add_global_callback(Event::Shift(Key::Tab), cycle_window);

    siv.run();
    ExitCode::SUCCESS
}

fn log_to_repl(s: &mut Cursive, text: &str) {
    s.call_on_name("repl_log", |view: &mut ScrollView<TextView>| {
        let text_view = view.get_inner_mut();
        text_view.append(format!("{}\n", text));
        view.scroll_to_bottom();
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
    let repl_log_content: String = s.call_on_name("repl_log", |view: &mut ScrollView<TextView>| {
        view.get_inner().get_content().source().to_string()
    }).unwrap_or_default();

    // Remove old workspace content
    s.call_on_name("workspace", |ws: &mut LinearLayout| {
        ws.clear();
    });

    // Total windows = Console + editors
    let total_windows = 1 + editor_names.len();

    let repl_log = TextView::new(repl_log_content)
        .scrollable()
        .with_name("repl_log");
    let console = ActiveWindow::new(repl_log, "Console").full_width();

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
    let engine_save = engine.clone();
    let editor_id_save = editor_id.clone();

    // Ctrl+S to save, Esc to close
    let editor_with_events = OnEventView::new(editor.with_name(&editor_id))
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

    s.focus_name(&target).ok();
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
  :vars                List variables".to_string()
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
