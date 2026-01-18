//! REPL Connect Client - connect to a running TUI REPL server
//!
//! Usage: `nostos connect -p <port>`
//!
//! Features:
//! - Readline-style line editing (reedline)
//! - Command history with persistence
//! - Syntax highlighting for Nostos code
//! - Autocomplete via server requests

use std::borrow::Cow;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use reedline::{
    Reedline, Signal, Prompt, PromptHistorySearch, PromptHistorySearchStatus,
    FileBackedHistory, Highlighter, StyledText, Completer, Suggestion, Span,
    ColumnarMenu, ReedlineMenu, KeyCode, KeyModifiers, ReedlineEvent,
    default_emacs_keybindings, MenuBuilder,
};
use nu_ansi_term::{Color, Style};
use nostos_syntax::lexer::{Token, lex};

/// Monotonically increasing command ID
static COMMAND_ID: AtomicU64 = AtomicU64::new(1);

/// Parse command-line arguments for connect
pub fn run_connect(args: &[String]) -> ExitCode {
    let mut port: Option<u16> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--port" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<u16>() {
                        Ok(p) => port = Some(p),
                        Err(_) => {
                            eprintln!("Error: Invalid port number '{}'", args[i + 1]);
                            return ExitCode::FAILURE;
                        }
                    }
                    i += 2;
                } else {
                    eprintln!("Error: -p requires a port number");
                    return ExitCode::FAILURE;
                }
            }
            "--help" | "-h" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            _ => {
                // Try to parse as port number directly
                if port.is_none() {
                    if let Ok(p) = args[i].parse::<u16>() {
                        port = Some(p);
                    } else {
                        eprintln!("Error: Unknown argument '{}'", args[i]);
                        return ExitCode::FAILURE;
                    }
                }
                i += 1;
            }
        }
    }

    let port = match port {
        Some(p) => p,
        None => {
            eprintln!("Error: Port number required");
            eprintln!("Usage: nostos connect -p <port>");
            return ExitCode::FAILURE;
        }
    };

    connect_to_server(port)
}

fn print_help() {
    eprintln!("Connect to a running REPL server");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    nostos connect -p <port>");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("    -p, --port <PORT>    Port to connect to");
    eprintln!("    -h, --help           Show this help");
    eprintln!();
    eprintln!("COMMANDS (after connecting):");
    eprintln!("    :load <file>         Load a .nos file or directory");
    eprintln!("    :reload              Reload all loaded files");
    eprintln!("    :status              Show compilation status");
    eprintln!("    :eval <expr>         Evaluate an expression");
    eprintln!("    :compile <file>      Compile a file (check for errors)");
    eprintln!("    :quit                Disconnect from server");
    eprintln!();
    eprintln!("FEATURES:");
    eprintln!("    - Arrow keys for line editing");
    eprintln!("    - Up/Down for command history");
    eprintln!("    - Tab for autocomplete");
    eprintln!("    - Syntax highlighting");
    eprintln!();
    eprintln!("EXAMPLE:");
    eprintln!("    # Terminal 1: Start REPL with server");
    eprintln!("    nostos repl --serve 7878");
    eprintln!();
    eprintln!("    # Terminal 2: Connect to it");
    eprintln!("    nostos connect -p 7878");
}

// ============================================================================
// Nostos Syntax Highlighter
// ============================================================================

struct NostosHighlighter;

impl Highlighter for NostosHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();

        // If it's a command, highlight differently
        if line.starts_with(':') {
            styled.push((Style::new().fg(Color::Cyan).bold(), line.to_string()));
            return styled;
        }

        let mut pos = 0;
        for (token, span) in lex(line) {
            // Add any whitespace/characters between previous token and this one
            if span.start > pos {
                styled.push((Style::new(), line[pos..span.start].to_string()));
            }

            let style = match token {
                Token::Type | Token::Var | Token::If | Token::Then | Token::Else |
                Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
                Token::Use | Token::Private | Token::Pub | Token::SelfKw | Token::SelfType |
                Token::Try | Token::Catch | Token::Finally | Token::Do |
                Token::While | Token::For | Token::To | Token::Break | Token::Continue |
                Token::Spawn | Token::SpawnLink | Token::SpawnMonitor | Token::Receive | Token::After |
                Token::Panic | Token::Extern | Token::From | Token::Test | Token::Quote =>
                    Style::new().fg(Color::Magenta).bold(),

                Token::True | Token::False |
                Token::Int(_) | Token::HexInt(_) | Token::BinInt(_) |
                Token::Int8(_) | Token::Int16(_) | Token::Int32(_) |
                Token::UInt8(_) | Token::UInt16(_) | Token::UInt32(_) | Token::UInt64(_) |
                Token::BigInt(_) | Token::Float(_) | Token::Float32(_) | Token::Decimal(_) =>
                    Style::new().fg(Color::Yellow),

                Token::String(_) | Token::SingleQuoteString(_) | Token::Char(_) =>
                    Style::new().fg(Color::Green),

                Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent | Token::StarStar |
                Token::EqEq | Token::NotEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq |
                Token::AndAnd | Token::OrOr | Token::Bang | Token::PlusPlus | Token::PipeRight |
                Token::Eq | Token::PlusEq | Token::MinusEq | Token::StarEq | Token::SlashEq |
                Token::LeftArrow | Token::RightArrow | Token::FatArrow | Token::Caret | Token::Dollar | Token::Question |
                Token::LParen | Token::RParen | Token::LBracket | Token::RBracket |
                Token::LBrace | Token::RBrace | Token::Comma | Token::Colon | Token::Dot |
                Token::Pipe | Token::Hash =>
                    Style::new().fg(Color::LightBlue),

                Token::UpperIdent(_) => Style::new().fg(Color::Yellow),
                Token::LowerIdent(_) => Style::new().fg(Color::White),

                Token::Comment | Token::MultiLineComment => Style::new().fg(Color::DarkGray),
                _ => Style::new(),
            };

            let text = &line[span.start..span.end];
            styled.push((style, text.to_string()));
            pos = span.end;
        }

        // Add any trailing content
        if pos < line.len() {
            styled.push((Style::new(), line[pos..].to_string()));
        }

        styled
    }
}

// ============================================================================
// Nostos Prompt
// ============================================================================

struct NostosPrompt;

impl Prompt for NostosPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("nostos> ")
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _prompt_mode: reedline::PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(&self, history_search: PromptHistorySearch) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!("({}reverse-search: {}) ", prefix, history_search.term))
    }
}

// ============================================================================
// Completer that queries the server
// ============================================================================

struct ServerCompleter {
    stream: Arc<Mutex<TcpStream>>,
}

impl Completer for ServerCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        // For commands starting with :, provide command completions
        if line.starts_with(':') {
            let commands = [":load", ":reload", ":status", ":eval", ":compile", ":quit", ":help"];
            return commands.iter()
                .filter(|c| c.starts_with(line))
                .map(|c| Suggestion {
                    value: c.to_string(),
                    description: None,
                    style: None,
                    extra: None,
                    span: Span::new(0, pos),
                    append_whitespace: true,
                })
                .collect();
        }

        // For code, send completion request to server
        let json = format!(
            r#"{{"id":0,"cmd":"complete","code":"{}","pos":{}}}"#,
            escape_json_string(line),
            pos
        );

        if let Ok(mut stream) = self.stream.lock() {
            // Clone for reading
            if let Ok(reader_stream) = stream.try_clone() {
                let mut reader = BufReader::new(reader_stream);

                // Send request
                if writeln!(stream, "{}", json).is_ok() {
                    let _ = stream.flush();

                    // Read response
                    let mut response = String::new();
                    if reader.read_line(&mut response).is_ok() {
                        return parse_completion_response(&response, line, pos);
                    }
                }
            }
        }

        Vec::new()
    }
}

/// Find where the current identifier/word starts
fn find_word_start(line: &str, pos: usize) -> usize {
    let prefix = if pos <= line.len() { &line[..pos] } else { line };

    // Check if we're completing after a dot
    if let Some(dot_pos) = prefix.rfind('.') {
        return dot_pos + 1; // Start after the dot
    }

    // Find the start of the current identifier
    prefix.rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0)
}

fn parse_completion_response(json: &str, line: &str, pos: usize) -> Vec<Suggestion> {
    // Parse completions from server response
    // Expected format: {"id":0,"status":"ok","output":"","completions":["foo","bar"]}
    let mut suggestions = Vec::new();

    let word_start = find_word_start(line, pos);

    let completions_pattern = r#""completions":["#;
    if let Some(start) = json.find(completions_pattern) {
        let rest = &json[start + completions_pattern.len()..];
        if let Some(end) = rest.find(']') {
            let array_content = &rest[..end];
            // Parse simple string array
            for item in array_content.split(',') {
                let item = item.trim().trim_matches('"');
                if !item.is_empty() {
                    suggestions.push(Suggestion {
                        value: item.to_string(),
                        description: None,
                        style: None,
                        extra: None,
                        span: Span::new(word_start, pos),
                        append_whitespace: false,
                    });
                }
            }
        }
    }

    suggestions
}

// ============================================================================
// Main connection logic
// ============================================================================

fn connect_to_server(port: u16) -> ExitCode {
    let addr = format!("127.0.0.1:{}", port);

    let stream = match TcpStream::connect(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: Could not connect to {}: {}", addr, e);
            eprintln!("Make sure the REPL server is running with: nostos repl --serve {}", port);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Connected to REPL server at {}", addr);
    eprintln!("Type :help for commands, :quit to disconnect");
    eprintln!();

    // Set up history file
    let history_path = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("nostos")
        .join("connect_history.txt");

    // Ensure directory exists
    if let Some(parent) = history_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let history = FileBackedHistory::with_file(1000, history_path)
        .expect("Failed to create history");

    // Create completer with shared stream
    let stream_for_complete = Arc::new(Mutex::new(stream.try_clone().expect("Failed to clone stream")));
    let completer = Box::new(ServerCompleter { stream: stream_for_complete });

    // Create completion menu
    let completion_menu = Box::new(
        ColumnarMenu::default()
            .with_name("completion_menu")
    );

    // Create keybindings with Tab for completion
    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu("completion_menu".to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );

    // Create reedline with all features
    let mut line_editor = Reedline::create()
        .with_history(Box::new(history))
        .with_highlighter(Box::new(NostosHighlighter))
        .with_completer(completer)
        .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
        .with_edit_mode(Box::new(reedline::Emacs::new(keybindings)));

    let prompt = NostosPrompt;

    let mut reader = BufReader::new(stream.try_clone().expect("Failed to clone stream"));
    let mut writer = stream;

    loop {
        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Handle local commands
                if line == ":quit" || line == ":q" || line == ":exit" {
                    eprintln!("Disconnected.");
                    break;
                }

                if line == ":help" || line == ":h" || line == "?" {
                    print_client_help();
                    continue;
                }

                // Parse and send command
                let (cmd, args) = parse_input(line);
                let json = format_command(&cmd, &args);

                // Send to server
                if let Err(e) = writeln!(writer, "{}", json) {
                    eprintln!("Error sending command: {}", e);
                    break;
                }
                writer.flush().ok();

                // Read response
                let mut response = String::new();
                match reader.read_line(&mut response) {
                    Ok(0) => {
                        eprintln!("Server disconnected.");
                        break;
                    }
                    Ok(_) => {
                        print_response(&response);
                    }
                    Err(e) => {
                        eprintln!("Error reading response: {}", e);
                        break;
                    }
                }
            }
            Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                eprintln!("\nDisconnected.");
                break;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    ExitCode::SUCCESS
}

fn print_client_help() {
    eprintln!("Commands:");
    eprintln!("  :load <path>    Load a .nos file or directory");
    eprintln!("  :reload         Reload all loaded files");
    eprintln!("  :status         Show compilation status");
    eprintln!("  :eval <expr>    Evaluate an expression");
    eprintln!("  :compile <file> Compile a file and show errors");
    eprintln!("  :quit           Disconnect from server");
    eprintln!("  :help           Show this help");
    eprintln!();
    eprintln!("Keyboard shortcuts:");
    eprintln!("  Up/Down         Navigate command history");
    eprintln!("  Ctrl+R          Search history");
    eprintln!("  Tab             Autocomplete");
    eprintln!("  Ctrl+C/D        Disconnect");
    eprintln!();
    eprintln!("You can also type code directly to evaluate it.");
}

/// Parse user input into command and arguments
fn parse_input(line: &str) -> (String, String) {
    if line.starts_with(':') {
        // Command
        let parts: Vec<&str> = line[1..].splitn(2, ' ').collect();
        let cmd = parts[0].to_string();
        let args = if parts.len() > 1 { parts[1].to_string() } else { String::new() };
        (cmd, args)
    } else {
        // Direct code evaluation
        ("eval".to_string(), line.to_string())
    }
}

/// Format a command as JSON for the server
fn format_command(cmd: &str, args: &str) -> String {
    let id = COMMAND_ID.fetch_add(1, Ordering::SeqCst);

    // Determine the appropriate key for the args
    let arg_key = match cmd {
        "load" | "compile" => "file",
        "eval" => "code",
        _ => "args",
    };

    // Escape the args for JSON
    let escaped_args = escape_json_string(args);

    if args.is_empty() {
        format!(r#"{{"id":{},"cmd":"{}"}}"#, id, cmd)
    } else {
        format!(r#"{{"id":{},"cmd":"{}","{}":"{}"}}"#, id, cmd, arg_key, escaped_args)
    }
}

/// Escape a string for JSON
fn escape_json_string(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\t' => result.push_str("\\t"),
            '\r' => result.push_str("\\r"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

/// Parse and print a JSON response from the server
fn print_response(json: &str) {
    // Simple JSON parsing without serde
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        eprintln!("Invalid response: {}", json);
        return;
    }

    // Extract fields manually
    let status = extract_json_field(json, "status");
    let output = extract_json_field(json, "output");
    let errors = extract_json_array(json, "errors");

    // Print based on status
    match status.as_str() {
        "ok" => {
            if !output.is_empty() {
                println!("{}", unescape_json_string(&output));
            }
        }
        "error" => {
            eprintln!("Error: {}", unescape_json_string(&output));
            if !errors.is_empty() {
                for error in &errors {
                    let file = extract_json_field(error, "file");
                    let line = extract_json_field(error, "line");
                    let message = extract_json_field(error, "message");
                    eprintln!("  {}:{}: {}", file, line, unescape_json_string(&message));
                }
            }
        }
        _ => {
            println!("{}", unescape_json_string(&output));
        }
    }
}

/// Extract a string field from JSON (simple parser)
fn extract_json_field(json: &str, field: &str) -> String {
    let pattern = format!(r#""{}":"#, field);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        // Check if value is a string (starts with ")
        if rest.starts_with('"') {
            // Find the closing quote, handling escaped quotes
            let mut end = 1;
            let chars: Vec<char> = rest.chars().collect();
            while end < chars.len() {
                if chars[end] == '"' && (end == 0 || chars[end - 1] != '\\') {
                    break;
                }
                end += 1;
            }
            return rest[1..end].to_string();
        }
        // Numeric value
        let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
        return rest[..end].to_string();
    }
    String::new()
}

/// Extract an array field from JSON (simple parser)
fn extract_json_array(json: &str, field: &str) -> Vec<String> {
    let pattern = format!(r#""{}":["#, field);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        if let Some(end) = rest.find(']') {
            let array_content = &rest[..end];
            // Split by },{ to get individual objects
            let mut result = Vec::new();
            let mut depth = 0;
            let mut current = String::new();
            for c in array_content.chars() {
                match c {
                    '{' => {
                        depth += 1;
                        current.push(c);
                    }
                    '}' => {
                        depth -= 1;
                        current.push(c);
                        if depth == 0 {
                            result.push(current.clone());
                            current.clear();
                        }
                    }
                    ',' if depth == 0 => {
                        // Skip comma between objects
                    }
                    _ => {
                        if depth > 0 {
                            current.push(c);
                        }
                    }
                }
            }
            return result;
        }
    }
    Vec::new()
}

/// Unescape JSON string
fn unescape_json_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('u') => {
                    // Unicode escape \uXXXX
                    let mut hex = String::new();
                    for _ in 0..4 {
                        if let Some(h) = chars.next() {
                            hex.push(h);
                        }
                    }
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(code) {
                            result.push(ch);
                        }
                    }
                }
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}
