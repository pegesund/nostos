//! REPL Server - allows remote connections to the TUI REPL
//!
//! When `nostos repl --serve <port>` is used, the TUI also listens on a TCP port
//! for JSON commands from remote clients (e.g., `nostos connect -p <port>`).

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Sender, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;

/// Command sent from a remote client
#[derive(Debug, Clone)]
pub struct ServerCommand {
    pub id: u64,
    pub cmd: String,
    pub args: String,
    /// Position for completion requests
    pub pos: Option<usize>,
    /// Channel to send response back to the client
    pub response_tx: Sender<ServerResponse>,
}

/// Response sent back to a remote client
#[derive(Debug, Clone)]
pub struct ServerResponse {
    pub id: u64,
    pub status: String,
    pub output: String,
    pub errors: Vec<ServerError>,
    /// Completions for autocomplete requests
    pub completions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ServerError {
    pub file: String,
    pub line: u32,
    pub message: String,
}

/// Parse a JSON command from a client
fn parse_command(line: &str, response_tx: Sender<ServerResponse>) -> Option<ServerCommand> {
    // Simple JSON parsing without serde
    // Expected format: {"id": 1, "cmd": "load", "file": "main.nos"}
    // or: {"id": 2, "cmd": "eval", "code": "1 + 2"}
    // or: {"id": 3, "cmd": "reload"}

    let line = line.trim();
    if !line.starts_with('{') || !line.ends_with('}') {
        return None;
    }

    let inner = &line[1..line.len()-1];

    let mut id: u64 = 0;
    let mut cmd = String::new();
    let mut args = String::new();
    let mut pos: Option<usize> = None;

    // Parse key-value pairs
    for part in inner.split(',') {
        let part = part.trim();
        if let Some(colon_pos) = part.find(':') {
            let key = part[..colon_pos].trim().trim_matches('"');
            let value = part[colon_pos + 1..].trim().trim_matches('"');

            match key {
                "id" => {
                    id = value.parse().unwrap_or(0);
                }
                "cmd" => {
                    cmd = value.to_string();
                }
                "file" | "code" | "expr" => {
                    // Handle escaped strings
                    args = unescape_json_string(value);
                }
                "pos" => {
                    pos = value.parse().ok();
                }
                _ => {}
            }
        }
    }

    if cmd.is_empty() {
        return None;
    }

    Some(ServerCommand {
        id,
        cmd,
        args,
        pos,
        response_tx,
    })
}

/// Unescape JSON string (basic implementation)
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

/// Format a response as JSON
fn format_response(response: &ServerResponse) -> String {
    let errors_json: Vec<String> = response.errors.iter().map(|e| {
        format!(r#"{{"file":"{}","line":{},"message":"{}"}}"#,
            escape_json_string(&e.file),
            e.line,
            escape_json_string(&e.message))
    }).collect();

    let completions_json: Vec<String> = response.completions.iter()
        .map(|c| format!(r#""{}""#, escape_json_string(c)))
        .collect();

    format!(
        r#"{{"id":{},"status":"{}","output":"{}","errors":[{}],"completions":[{}]}}"#,
        response.id,
        escape_json_string(&response.status),
        escape_json_string(&response.output),
        errors_json.join(","),
        completions_json.join(",")
    )
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

/// Handle a single client connection
fn handle_client(
    stream: TcpStream,
    command_tx: Sender<ServerCommand>,
    client_id: u64,
) {
    let peer_addr = stream.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "unknown".to_string());
    eprintln!("REPL Server: Client {} connected from {}", client_id, peer_addr);

    let mut reader = BufReader::new(stream.try_clone().expect("Failed to clone stream"));
    let mut writer = stream;

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF - client disconnected
                eprintln!("REPL Server: Client {} disconnected", client_id);
                break;
            }
            Ok(_) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Create a channel for the response
                let (response_tx, response_rx) = mpsc::channel();

                // Parse and send command
                if let Some(cmd) = parse_command(line, response_tx) {
                    if command_tx.send(cmd).is_err() {
                        eprintln!("REPL Server: Failed to send command to TUI");
                        break;
                    }

                    // Wait for response
                    match response_rx.recv() {
                        Ok(response) => {
                            let json = format_response(&response);
                            if writeln!(writer, "{}", json).is_err() {
                                eprintln!("REPL Server: Failed to send response to client {}", client_id);
                                break;
                            }
                            let _ = writer.flush();
                        }
                        Err(_) => {
                            eprintln!("REPL Server: Failed to receive response");
                            break;
                        }
                    }
                } else {
                    // Invalid command
                    let error_response = ServerResponse {
                        id: 0,
                        status: "error".to_string(),
                        output: "Invalid JSON command".to_string(),
                        errors: vec![],
                        completions: vec![],
                    };
                    let json = format_response(&error_response);
                    let _ = writeln!(writer, "{}", json);
                    let _ = writer.flush();
                }
            }
            Err(e) => {
                eprintln!("REPL Server: Error reading from client {}: {}", client_id, e);
                break;
            }
        }
    }
}

/// Start the REPL server on the given port
/// Returns a receiver for commands that should be processed by the TUI
pub fn start_server(port: u16) -> Result<Receiver<ServerCommand>, String> {
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr)
        .map_err(|e| format!("Failed to bind to {}: {}", addr, e))?;

    eprintln!("REPL Server: Listening on {}", addr);

    let (command_tx, command_rx) = mpsc::channel();

    // Spawn server thread
    thread::spawn(move || {
        let mut client_id = 0u64;

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    client_id += 1;
                    let command_tx = command_tx.clone();

                    // Spawn a thread for each client
                    thread::spawn(move || {
                        handle_client(stream, command_tx, client_id);
                    });
                }
                Err(e) => {
                    eprintln!("REPL Server: Error accepting connection: {}", e);
                }
            }
        }
    });

    Ok(command_rx)
}
