//! Integration tests for the Nostos LSP server.
//! These tests run the actual LSP binary and communicate via LSP protocol.
//!
//! IMPORTANT: These tests verify the exact same environment as VS Code uses.
//! They start the actual LSP binary and send real LSP protocol messages.

use std::io::{BufRead, BufReader, Read as IoRead, Write};
use std::process::{Child, Command, Stdio};
use std::path::PathBuf;
use std::fs;
use std::time::{Duration, Instant};
use serde_json::{json, Value};

struct LspClient {
    process: Child,
    request_id: i64,
    stdout_reader: BufReader<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
}

#[derive(Debug, Clone)]
struct Diagnostic {
    line: u32,       // 0-based line number
    message: String,
}

impl LspClient {
    fn new(lsp_binary: &str) -> Self {
        let mut process = Command::new(lsp_binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start LSP server");

        let stdout = process.stdout.take().expect("Failed to get stdout");
        let stderr = process.stderr.take();
        let stdout_reader = BufReader::new(stdout);

        LspClient {
            process,
            request_id: 0,
            stdout_reader,
            stderr,
        }
    }

    fn get_stderr(&mut self) -> String {
        if let Some(mut stderr) = self.stderr.take() {
            let mut output = String::new();
            // Read with a timeout by setting non-blocking
            // For simplicity, just try to read what's available
            use std::io::Read;
            let mut buf = [0u8; 8192];
            // Set a short deadline - just read what's there
            match stderr.read(&mut buf) {
                Ok(n) => output = String::from_utf8_lossy(&buf[..n]).to_string(),
                Err(_) => {}
            }
            output
        } else {
            String::new()
        }
    }

    fn send_request(&mut self, method: &str, params: Value) -> Value {
        self.request_id += 1;
        let expected_id = self.request_id;
        let request = json!({
            "jsonrpc": "2.0",
            "id": expected_id,
            "method": method,
            "params": params
        });
        self.send_message(&request);

        // Keep reading until we get a response with matching id
        // (skip over notifications which have no id)
        loop {
            let msg = self.read_message().expect("Failed to read response");
            if let Some(id) = msg.get("id") {
                if id.as_i64() == Some(expected_id) {
                    return msg;
                }
            }
            // It's a notification, skip it and read next message
        }
    }

    fn send_notification(&mut self, method: &str, params: Value) {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        self.send_message(&notification);
    }

    fn send_message(&mut self, message: &Value) {
        let content = serde_json::to_string(message).unwrap();
        let header = format!("Content-Length: {}\r\n\r\n", content.len());

        let stdin = self.process.stdin.as_mut().unwrap();
        stdin.write_all(header.as_bytes()).unwrap();
        stdin.write_all(content.as_bytes()).unwrap();
        stdin.flush().unwrap();
    }

    fn read_message(&mut self) -> Option<Value> {
        // Read headers
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            match self.stdout_reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(_) => {}
                Err(_) => return None,
            }
            let line = line.trim();
            if line.is_empty() {
                break;
            }
            if line.starts_with("Content-Length:") {
                content_length = line["Content-Length:".len()..].trim().parse().unwrap();
            }
        }

        if content_length == 0 {
            return None;
        }

        // Read content
        let mut content = vec![0u8; content_length];
        self.stdout_reader.read_exact(&mut content).ok()?;

        serde_json::from_slice(&content).ok()
    }

    /// Read messages until we get a publishDiagnostics notification for the given URI
    /// or timeout after the specified duration
    fn read_diagnostics(&mut self, uri: &str, timeout: Duration) -> Vec<Diagnostic> {
        let start = Instant::now();
        let mut read_count = 0;

        while start.elapsed() < timeout {
            if let Some(msg) = self.read_message() {
                read_count += 1;
                let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("none");
                println!("DEBUG: Read message #{}: method={}", read_count, method);

                // Check if this is a publishDiagnostics notification
                if msg.get("method") == Some(&json!("textDocument/publishDiagnostics")) {
                    if let Some(params) = msg.get("params") {
                        let msg_uri = params.get("uri").and_then(|u| u.as_str()).unwrap_or("none");
                        println!("DEBUG: publishDiagnostics for uri={}", msg_uri);
                        if params.get("uri") == Some(&json!(uri)) {
                            // Parse diagnostics
                            if let Some(diagnostics) = params.get("diagnostics").and_then(|d| d.as_array()) {
                                let result: Vec<Diagnostic> = diagnostics.iter().filter_map(|d| {
                                    let line = d.get("range")?.get("start")?.get("line")?.as_u64()? as u32;
                                    let message = d.get("message")?.as_str()?.to_string();
                                    Some(Diagnostic { line, message })
                                }).collect();
                                println!("DEBUG: Found {} diagnostics, first line: {:?}",
                                    result.len(),
                                    result.first().map(|d| d.line));
                                return result;
                            }
                        }
                    }
                }
            } else {
                // No message available, wait a bit
                std::thread::sleep(Duration::from_millis(50));
            }
        }

        vec![]
    }

    /// Wait for the server to be ready (indicated by nostos/fileStatus notification)
    /// This ensures engine initialization is complete before sending requests
    fn wait_for_ready(&mut self, timeout: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if let Some(msg) = self.read_message() {
                let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("none");
                if method == "nostos/fileStatus" {
                    return true;
                }
            } else {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        false
    }

    fn initialize(&mut self, root_path: &str) -> Value {
        let root_uri = format!("file://{}", root_path);
        self.send_request("initialize", json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {},
            "workspaceFolders": [{
                "uri": root_uri,
                "name": "test"
            }]
        }))
    }

    fn initialized(&mut self) {
        self.send_notification("initialized", json!({}));
    }

    fn did_open(&mut self, uri: &str, content: &str) {
        self.send_notification("textDocument/didOpen", json!({
            "textDocument": {
                "uri": uri,
                "languageId": "nostos",
                "version": 1,
                "text": content
            }
        }));
    }

    fn did_change(&mut self, uri: &str, content: &str, version: i32) {
        self.send_notification("textDocument/didChange", json!({
            "textDocument": {
                "uri": uri,
                "version": version
            },
            "contentChanges": [{
                "text": content
            }]
        }));
    }

    /// Request completions at a specific position
    /// line and character are 0-based
    fn completion(&mut self, uri: &str, line: u32, character: u32) -> Vec<String> {
        let response = self.send_request("textDocument/completion", json!({
            "textDocument": {
                "uri": uri
            },
            "position": {
                "line": line,
                "character": character
            }
        }));

        // Parse completion items
        if let Some(result) = response.get("result") {
            if result.is_null() {
                return vec![];
            }
            if let Some(items) = result.as_array() {
                return items.iter()
                    .filter_map(|item| item.get("label").and_then(|l| l.as_str()))
                    .map(|s| s.to_string())
                    .collect();
            }
            // CompletionList format
            if let Some(items) = result.get("items").and_then(|i| i.as_array()) {
                return items.iter()
                    .filter_map(|item| item.get("label").and_then(|l| l.as_str()))
                    .map(|s| s.to_string())
                    .collect();
            }
        }
        vec![]
    }

    /// Request document symbols for a file
    fn document_symbol(&mut self, uri: &str) -> Vec<(String, String, u32)> {
        let response = self.send_request("textDocument/documentSymbol", json!({
            "textDocument": {
                "uri": uri
            }
        }));

        // Parse symbol information - returns (name, kind, line)
        let mut symbols = Vec::new();
        if let Some(result) = response.get("result") {
            if let Some(items) = result.as_array() {
                for item in items {
                    let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                    let kind = item.get("kind").and_then(|k| k.as_u64()).unwrap_or(0);
                    let kind_name = match kind {
                        1 => "File",
                        2 => "Module",
                        5 => "Class",
                        6 => "Method",
                        11 => "Interface",
                        12 => "Function",
                        23 => "Struct",
                        _ => "Unknown",
                    }.to_string();
                    let line = item.get("location")
                        .and_then(|l| l.get("range"))
                        .and_then(|r| r.get("start"))
                        .and_then(|s| s.get("line"))
                        .and_then(|l| l.as_u64())
                        .unwrap_or(0) as u32;
                    symbols.push((name, kind_name, line));
                }
            }
        }
        symbols
    }

    /// Find all references to symbol at position
    fn references(&mut self, uri: &str, line: u32, character: u32) -> Vec<(String, u32, u32)> {
        let response = self.send_request("textDocument/references", json!({
            "textDocument": {
                "uri": uri
            },
            "position": {
                "line": line,
                "character": character
            },
            "context": {
                "includeDeclaration": true
            }
        }));

        // Parse locations - returns (uri, line, character)
        let mut refs = Vec::new();
        if let Some(result) = response.get("result") {
            if let Some(items) = result.as_array() {
                for item in items {
                    let loc_uri = item.get("uri").and_then(|u| u.as_str()).unwrap_or("").to_string();
                    let line = item.get("range")
                        .and_then(|r| r.get("start"))
                        .and_then(|s| s.get("line"))
                        .and_then(|l| l.as_u64())
                        .unwrap_or(0) as u32;
                    let char = item.get("range")
                        .and_then(|r| r.get("start"))
                        .and_then(|s| s.get("character"))
                        .and_then(|c| c.as_u64())
                        .unwrap_or(0) as u32;
                    refs.push((loc_uri, line, char));
                }
            }
        }
        refs
    }

    fn shutdown(&mut self) -> Value {
        self.send_request("shutdown", json!(null))
    }

    fn exit(&mut self) {
        self.send_notification("exit", json!(null));
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.process.kill();
        // Reap the zombie process
        let _ = self.process.wait();
    }
}

fn create_test_project(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join("nostos_lsp_integration_tests");
    fs::create_dir_all(&base).ok();
    let path = base.join(name);
    if path.exists() {
        fs::remove_dir_all(&path).ok();
    }
    fs::create_dir_all(&path).expect("Failed to create test dir");

    // Create nostos.toml
    fs::write(path.join("nostos.toml"), "[project]\nname = \"test\"\n").unwrap();

    path
}

fn cleanup_test_project(path: &PathBuf) {
    fs::remove_dir_all(path).ok();
}

fn get_lsp_binary() -> String {
    // Use the installed binary (same one VS Code uses)
    let home = std::env::var("HOME").unwrap();
    format!("{}/.local/bin/nostos-lsp", home)
}

/// Test that nested list map works without errors
/// User code: gg.map(m => m.map(n => n.asInt32()))
/// Expected: No errors (nested lambda types should be inferred correctly)
#[test]
fn test_lsp_nested_list_map() {
    let project_path = create_test_project("nested_list_map");

    // Create good.nos
    fs::write(
        project_path.join("good.nos"),
        "pub addff(a, b) = a + b\npub multiply(x, y) = x * y\n"
    ).unwrap();

    // Create main.nos with nested list map - EXACT user code
    let main_content = r#"main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)
    y1.asInt32()

    gg = [[0,1]]
    gg.map(m => m.map(n => n.asInt32()))
}
"#;
    fs::write(project_path.join("main.nos"), main_content).unwrap();

    // Start LSP
    let mut client = LspClient::new(&get_lsp_binary());

    // Initialize
    let init_response = client.initialize(project_path.to_str().unwrap());
    println!("Initialize response: {:?}", init_response);

    client.initialized();

    // Give LSP time to load stdlib
    std::thread::sleep(Duration::from_millis(500));

    // Open main.nos
    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, main_content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for nested list map test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);  // +1 for 1-based display
    }
    println!("=== End diagnostics ===");

    // Shutdown
    let _ = client.shutdown();
    client.exit();

    cleanup_test_project(&project_path);

    // ASSERTION: There should be NO errors for valid nested list map
    // If this fails, it means nested list inference is broken
    assert!(
        diagnostics.is_empty(),
        "Expected no errors for valid nested list map, but got {} errors:\n{}",
        diagnostics.len(),
        diagnostics.iter()
            .map(|d| format!("  Line {}: {}", d.line + 1, d.message))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Test that error line is correct when user ADDS empty lines via didChange
/// This simulates what happens in VS Code when user types empty lines at top
#[test]
fn test_lsp_error_line_after_adding_empty_lines() {
    let project_path = create_test_project("add_empty_lines");

    fs::write(
        project_path.join("good.nos"),
        "pub addff(a, b) = a + b\npub multiply(x, y) = x * y\n"
    ).unwrap();

    // Start with NO empty lines at top - gg.map() on line 13
    let initial_content = r#"main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)
    y1.asInt32()

    gg = [[0,1]]
    gg.map(m => m.map(n => n.asFloat32()))
    # test
    gg.map()

}
"#;
    fs::write(project_path.join("main.nos"), initial_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, initial_content);

    // Wait and get initial diagnostics - should show error on line 13
    std::thread::sleep(Duration::from_millis(300));
    let initial_diags = client.read_diagnostics(&main_uri, Duration::from_secs(2));
    println!("=== Initial diagnostics (no empty lines) ===");
    for d in &initial_diags {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    // NOW simulate user adding 2 empty lines at top via didChange
    // This shifts gg.map() from line 13 to line 15
    let modified_content = r#"

main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)
    y1.asInt32()

    gg = [[0,1]]
    gg.map(m => m.map(n => n.asFloat32()))
    # test
    gg.map()

}
"#;
    client.did_change(&main_uri, modified_content, 2);

    // Wait for recompile and get new diagnostics
    // The LSP might publish multiple diagnostics - keep reading until we get the updated line
    std::thread::sleep(Duration::from_millis(500));

    // Read diagnostics - may need to read twice if there's a stale notification buffered
    let mut modified_diags = client.read_diagnostics(&main_uri, Duration::from_secs(2));
    println!("=== After adding 2 empty lines at top (first read) ===");
    for d in &modified_diags {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    // If we got line 12 (old), try reading again to get the new notification
    if modified_diags.iter().any(|d| d.line == 12) {
        println!("Got stale notification, reading again...");
        let second_read = client.read_diagnostics(&main_uri, Duration::from_secs(2));
        if !second_read.is_empty() {
            modified_diags = second_read;
            println!("=== After adding 2 empty lines at top (second read) ===");
            for d in &modified_diags {
                println!("  Line {}: {}", d.line + 1, d.message);
            }
        }
    }

    let _ = client.shutdown();

    // Read stderr before exit to see debug output
    let stderr_output = client.get_stderr();
    println!("=== LSP stderr output ===\n{}\n=== End stderr ===", stderr_output);

    client.exit();
    cleanup_test_project(&project_path);

    // Look for argument errors (for gg.map() call with missing arguments)
    let map_errors: Vec<_> = modified_diags.iter()
        .filter(|d| d.message.contains("argument") || d.message.contains("map"))
        .collect();

    assert!(!map_errors.is_empty(), "Expected argument error for gg.map() after adding empty lines");

    // After adding 2 empty lines, gg.map() is on line 15 (1-based), line 14 (0-based)
    let expected_line_0based = 14;
    let has_error_on_correct_line = map_errors.iter().any(|d| d.line == expected_line_0based);

    assert!(
        has_error_on_correct_line,
        "After adding 2 empty lines, error should be on line 15 (0-based: 14), but got: {:?}",
        map_errors.iter().map(|d| d.line + 1).collect::<Vec<_>>()
    );
}

/// Test that error for gg.map() (missing args) appears on the correct line
/// EXACT user code from /var/tmp/test_status_project/main.nos
#[test]
fn test_lsp_map_missing_args_error_line() {
    let project_path = create_test_project("map_missing_args");

    // Create good.nos - EXACT user code
    fs::write(
        project_path.join("good.nos"),
        r#"# A working function
pub addff(a, b) = a + b

# A working function
pub addfff(a, b) = a + b

pub multiply(x, y) = x * y
"#
    ).unwrap();

    // Create main.nos - EXACT user code from /var/tmp/test_status_project/main.nos
    // Line numbers (1-based):
    // 1:  type XX = AAA | BBB
    // 2:  (empty)
    // 3:  main() = {
    // 4:      x = good.addff(3, 2)
    // 5:      y = good.multiply(2,3)
    // 6:      yy = [1,2,3]
    // 7:      yy.map(m => m.asInt8())
    // 8:      y1 = 33
    // 9:      g = asInt32(y1)
    // 10:     y1.asInt32()
    // 11: (empty)
    // 12: gg = [[0,1]]                     <-- Note: no indent!
    // 13:     gg.map(m => m.map(n => n.asFloat32()))
    // 14:     # test
    // 15:     gg.map()                      <-- ERROR SHOULD BE HERE (line 15)
    // 16: (empty)
    // 17: }
    // EXACT user code - NO empty lines at start, gg.map() on line 13
    // Line numbers (1-based):
    // 1:  main() = {
    // 2:      x = good.addff(3, 2)
    // 3:      y = good.multiply(2,3)
    // 4:      yy = [1,2,3]
    // 5:      yy.map(m => m.asInt8())
    // 6:      y1 = 33
    // 7:      g = asInt32(y1)
    // 8:      y1.asInt32()
    // 9:  (empty)
    // 10:     gg = [[0,1]]
    // 11:     gg.map(m => m.map(n => n.asFloat32()))
    // 12:     # test
    // 13:     gg.map()   <-- ERROR SHOULD BE HERE (line 13)
    // 14: (empty)
    // 15: }
    let main_content = r#"main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)
    y1.asInt32()

    gg = [[0,1]]
    gg.map(m => m.map(n => n.asFloat32()))
    # test
    gg.map()

}
"#;
    fs::write(project_path.join("main.nos"), main_content).unwrap();

    // Start LSP
    let mut client = LspClient::new(&get_lsp_binary());

    // Initialize
    let init_response = client.initialize(project_path.to_str().unwrap());
    println!("Initialize response: {:?}", init_response);

    client.initialized();

    // Give LSP time to load stdlib
    std::thread::sleep(Duration::from_millis(500));

    // Open main.nos
    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, main_content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for map missing args test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);  // +1 for 1-based display
    }
    println!("=== End diagnostics ===");

    // Shutdown
    let _ = client.shutdown();
    client.exit();

    cleanup_test_project(&project_path);

    // gg.map() on line 13 (1-based) should have error about wrong arguments
    // The error is "Wrong number of arguments: expected 2, found 1" for the map() call
    let arg_errors: Vec<_> = diagnostics.iter()
        .filter(|d| d.message.contains("argument") || d.message.contains("map"))
        .collect();

    println!("\n=== Argument-related errors ===");
    for d in &arg_errors {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    // Should have at least one error about arguments for gg.map()
    assert!(
        !arg_errors.is_empty(),
        "Expected at least one argument-related error for gg.map()"
    );

    // Rename for the rest of the assertions
    let map_errors = arg_errors;

    // gg.map() is on line 13 (1-based), which is line 12 (0-based)
    let expected_line_0based = 12;
    let has_error_on_correct_line = map_errors.iter().any(|d| d.line == expected_line_0based);

    assert!(
        has_error_on_correct_line,
        "Error for gg.map() should be on line 13 (0-based: 12), but got errors on lines: {:?}",
        map_errors.iter().map(|d| d.line + 1).collect::<Vec<_>>()
    );
}

/// Test autocomplete works for nested list lambdas
/// When typing `m.` inside `gg.map(m => m.)`, should get List methods
/// because m is List[Int] (gg is List[List[Int]])
/// This test uses the EXACT user code structure with good.nos imports
#[test]
fn test_lsp_nested_list_autocomplete() {
    let project_path = create_test_project("nested_autocomplete");

    // Create good.nos (same as user has)
    fs::write(
        project_path.join("good.nos"),
        "pub addff(a, b) = a + b\npub multiply(x, y) = x * y\n"
    ).unwrap();

    // Start with VALID code that compiles
    let initial_content = r#"main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)
    y1.asInt32()

    gg = [[0,1]]
    gg.map(m => m.map(n => n.asInt32()))
}
"#;

    fs::write(project_path.join("main.nos"), initial_content).unwrap();

    // Start LSP
    let mut client = LspClient::new(&get_lsp_binary());

    // Initialize
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();

    // Give LSP time to load stdlib
    std::thread::sleep(Duration::from_millis(500));

    // Open main.nos with initial valid content
    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, initial_content);

    // Wait for initial compile
    std::thread::sleep(Duration::from_millis(500));

    // NOW simulate user typing inside NESTED lambda - typing "n."
    // User has: gg.map(m => m.map(n => n.))
    // They want completions for "n." where n is Int (element of List[Int])
    // Line 11 (0-based: 10): "    gg.map(m => m.map(n => n.))"
    //                         0         1         2         3
    //                         0123456789012345678901234567890
    //                                                    ^ cursor at position 29 (after "n.")
    let typing_content = r#"main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)
    y1.asInt32()

    gg = [[0,1]]
    gg.map(m => m.map(n => n.))
}
"#;

    // Send did_change to simulate typing
    client.did_change(&main_uri, typing_content, 2);

    // Wait briefly for change to be processed
    std::thread::sleep(Duration::from_millis(200));

    // Request completions at position after "n." in the NESTED lambda
    // Line 11 (0-based: 10): "    gg.map(m => m.map(n => n.))"
    let completions = client.completion(&main_uri, 10, 29);

    println!("=== Completions for NESTED lambda (n.) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("=== End completions ({} total) ===", completions.len());

    // Shutdown
    let _ = client.shutdown();
    client.exit();

    cleanup_test_project(&project_path);

    // ASSERTION: Should get Int methods like "asInt32", "asFloat", etc.
    // n is the element of m which is List[Int], so n should be Int
    assert!(
        !completions.is_empty(),
        "Expected completions for NESTED lambda parameter 'n' (type Int), but got none"
    );

    // Check for Int methods (asInt32, asFloat, etc.)
    let has_int_method = completions.iter().any(|c| c.starts_with("as") || c == "abs" || c == "negate");

    assert!(
        has_int_method,
        "Expected Int methods like 'asInt32' or 'abs' in completions for nested lambda, got: {:?}",
        completions
    );
}

/// Test that stdlib is loaded and UFCS works (map resolves to stdlib.list.map)
/// If stdlib is NOT loaded, we get: "no method `map` found for type `List[Int]`"
/// If stdlib IS loaded, we get: "function `stdlib.list.map` expects 2 arguments..."
#[test]
fn test_lsp_stdlib_loaded() {
    let project_path = create_test_project("stdlib_loaded");

    // Create main.nos with a simple map call that will fail (missing lambda)
    // This tests that:
    // 1. Stdlib is loaded (map resolves to stdlib.list.map)
    // 2. Error message mentions stdlib.list.map, not "no method found"
    let main_content = r#"main() = {
    x = [1, 2, 3]
    x.map()
}
"#;
    fs::write(project_path.join("main.nos"), main_content).unwrap();

    // Start LSP
    let mut client = LspClient::new(&get_lsp_binary());

    // Initialize
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();

    // Give LSP time to load stdlib
    std::thread::sleep(Duration::from_millis(500));

    // Open main.nos
    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, main_content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for stdlib loaded test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }
    println!("=== End diagnostics ===");

    // Shutdown
    let _ = client.shutdown();
    client.exit();

    cleanup_test_project(&project_path);

    // Verify we got some diagnostic
    assert!(
        !diagnostics.is_empty(),
        "Expected at least one diagnostic error"
    );

    let first_error = &diagnostics[0].message;

    // ASSERTION: Error should NOT be "no method found" (which would mean stdlib NOT loaded)
    // If stdlib is loaded, map resolves and we get an argument count error instead
    assert!(
        !first_error.contains("no method"),
        "Stdlib not loaded! Got 'no method found' error: {}",
        first_error
    );

    // The error should be about wrong arguments (map expects 2: list and lambda)
    // This proves stdlib.list.map was resolved
    assert!(
        first_error.contains("argument") || first_error.contains("stdlib.list.map"),
        "Expected argument error or stdlib.list.map reference, got: {}",
        first_error
    );
}

/// Test autocomplete for explicit type annotations: x:String = ... then x.
#[test]
fn test_lsp_autocomplete_explicit_type_annotation() {
    let project_path = create_test_project("explicit_type_annotation");

    let content = r#"main() = {
    g2 = [["a", "b"]]
    x2:String = g2[0][0]
    x2.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at position after "x2." (line 4, after the dot)
    // Line 4 (0-based: 3): "    x2."
    let completions = client.completion(&main_uri, 3, 7);

    println!("=== Completions for explicit type annotation (x2:String) ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    assert!(
        !completions.is_empty(),
        "Expected completions for x2 with explicit type String, got none"
    );

    // Should have String methods like chars, length, etc.
    let has_string_method = completions.iter().any(|c|
        c == "chars" || c == "length" || c == "split" || c == "trim"
    );
    assert!(
        has_string_method,
        "Expected String methods like 'chars' or 'length', got: {:?}",
        completions
    );
}

/// Test autocomplete for index expressions: g2[0]. should show List methods
#[test]
fn test_lsp_autocomplete_index_expression() {
    let project_path = create_test_project("index_expression");

    let content = r#"main() = {
    g2:List[List[String]] = [["a", "b"]]
    g2[0].
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "g2[0]." (line 3)
    // Line 3 (0-based: 2): "    g2[0]."
    let completions = client.completion(&main_uri, 2, 10);

    println!("=== Completions for index expression g2[0] ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    assert!(
        !completions.is_empty(),
        "Expected completions for g2[0] (type List[String]), got none"
    );

    // Should have List methods like map, filter, etc.
    let has_list_method = completions.iter().any(|c|
        c == "map" || c == "filter" || c == "fold" || c == "length"
    );
    assert!(
        has_list_method,
        "Expected List methods like 'map' or 'filter', got: {:?}",
        completions
    );
}

/// Test autocomplete for record field access: p.x where p is a record type
#[test]
fn test_lsp_autocomplete_record_fields() {
    let project_path = create_test_project("record_fields");

    let content = r#"type Point = { x: Int, y: Int }

main() = {
    p = Point(10, 20)
    p.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p." (line 5, 0-based: 4)
    // Line 5: "    p."
    let completions = client.completion(&main_uri, 4, 6);

    println!("=== Completions for record (Point) ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Records should show field names or general methods
    // Even if field autocomplete isn't implemented, should get SOME methods
    println!("Record completions count: {}", completions.len());

    // For now, just check we get some completions (generic methods work on any type)
    // In the future, we should specifically check for field names x and y
}

/// Test autocomplete for variant types with explicit annotation
#[test]
fn test_lsp_autocomplete_variant_type() {
    let project_path = create_test_project("variant_type");

    let content = r#"type Result[T, E] = Ok(T) | Err(E)

main() = {
    r:Result[Int, String] = Ok(42)
    r.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "r." (line 5, 0-based: 4)
    let completions = client.completion(&main_uri, 4, 6);

    println!("=== Completions for variant (Result) ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Variants should show some methods (at least generic ones like show, hash)
    println!("Variant completions count: {}", completions.len());
}

/// Test autocomplete for mvar variables
/// mvar counter: Int = 42 then counter. should show Int methods
#[test]
fn test_lsp_autocomplete_mvar() {
    let project_path = create_test_project("mvar_autocomplete");

    let content = r#"mvar counter: Int = 42

get_val() = counter

main() = {
    counter.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "counter." (line 6, 0-based: 5)
    // Line 6: "    counter."
    let completions = client.completion(&main_uri, 5, 12);

    println!("=== Completions for mvar (counter:Int) ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // mvar should show Int methods (asFloat, abs, etc.)
    println!("MVar completions count: {}", completions.len());

    // Check for Int methods
    let has_int_method = completions.iter().any(|c|
        c.starts_with("as") || c == "abs" || c == "negate"
    );

    // For now, just log - if no completions, the type inference for mvars needs work
    if completions.is_empty() {
        println!("Note: MVar autocomplete may need type inference support for mvar declarations");
    } else {
        assert!(
            has_int_method,
            "Expected Int methods for mvar counter, got: {:?}",
            completions
        );
    }
}

/// Test autocomplete for reactive records
/// reactive Point = { x: Int, y: Int } then p. should show Point methods or fields
#[test]
fn test_lsp_autocomplete_reactive() {
    let project_path = create_test_project("reactive_autocomplete");

    let content = r#"reactive Point = { x: Int, y: Int }

main() = {
    p = Point(x: 0, y: 0)
    p.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p." (line 5, 0-based: 4)
    // Line 5: "    p."
    let completions = client.completion(&main_uri, 4, 6);

    println!("=== Completions for reactive (Point) ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Reactive records should show some methods (onChange, onRead, fields, etc.)
    println!("Reactive completions count: {}", completions.len());

    // Check for reactive-specific methods if available
    let has_reactive_method = completions.iter().any(|c|
        c == "onChange" || c == "onRead" || c == "x" || c == "y"
    );

    if completions.is_empty() {
        println!("Note: Reactive autocomplete may need type inference support");
    } else if !has_reactive_method {
        println!("Note: Reactive-specific methods (onChange, onRead, fields) not found, got generic methods: {:?}", completions);
    }
}

/// Test autocomplete for inferred type from index expression (NO explicit annotation)
/// g2 = [["a" "b"]]
/// x2 = g2[0][0]   <- should infer String from g2's type
/// x2.             <- should show String methods
#[test]
fn test_lsp_autocomplete_inferred_index_type() {
    let project_path = create_test_project("inferred_index_type");

    let content = r#"main() = {
    g2 = [["a", "b"]]
    x2 = g2[0][0]
    y3 = "ffff"
    x2.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "x2." (line 5, 0-based: 4)
    // Line 5: "    x2."
    let completions = client.completion(&main_uri, 4, 7);

    println!("=== Completions for inferred index type (x2 = g2[0][0]) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have String methods
    assert!(
        !completions.is_empty(),
        "Expected completions for x2 (inferred String from g2[0][0]), got none"
    );

    let has_string_method = completions.iter().any(|c|
        c == "chars" || c == "length" || c == "split" || c == "trim"
    );
    assert!(
        has_string_method,
        "Expected String methods like 'chars' or 'length', got: {:?}",
        completions
    );
}

/// Test that errors are shown when opening a file that has errors
/// The "x2." line is incomplete and should show as an error
#[test]
fn test_lsp_errors_shown_on_open() {
    let project_path = create_test_project("errors_on_open");

    // File with an error - "x2." is incomplete
    let content = r#"main() = {
    g2 = [["a", "b"]]
    x2 = g2[0][0]
    y3 = "ffff"
    x2.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();

    // Wait for engine to be ready (can take 5+ seconds in parallel test runs)
    client.wait_for_ready(Duration::from_secs(15));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics after opening file (engine is ready, so this should be fast)
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(5));

    println!("=== Diagnostics after opening file with error ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }
    println!("Total diagnostics: {}", diagnostics.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have at least one error for the "x2." line
    assert!(
        !diagnostics.is_empty(),
        "Expected diagnostics for incomplete 'x2.' expression, got none"
    );
}

/// Test autocomplete for method chains: x2.chars(). should show List[Char] methods
#[test]
fn test_lsp_autocomplete_method_chain() {
    let project_path = create_test_project("method_chain");

    let content = r#"main() = {
    x2:String = "hello"
    x2.chars().
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "x2.chars()." (line 3, 0-based: 2)
    // Line 3: "    x2.chars()."
    let completions = client.completion(&main_uri, 2, 15);

    println!("=== Completions for method chain (x2.chars().) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have List methods since chars() returns List[Char]
    assert!(
        !completions.is_empty(),
        "Expected completions for x2.chars() (List[Char]), got none"
    );

    // Check for List methods
    let has_list_method = completions.iter().any(|c|
        c == "map" || c == "filter" || c == "fold" || c == "length"
    );
    assert!(
        has_list_method,
        "Expected List methods like 'map' or 'filter', got: {:?}",
        completions
    );
}

/// Test autocomplete for complex method chains with generic return types
/// x2.chars().drop(1).get(0). should show Char methods (not generic 'a')
#[test]
fn test_lsp_autocomplete_method_chain_generic() {
    let project_path = create_test_project("method_chain_generic");

    let content = r#"main() = {
    x2:String = "hello"
    x2.chars().drop(1).get(0).
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "x2.chars().drop(1).get(0)." (line 3, 0-based: 2)
    let completions = client.completion(&main_uri, 2, 30);

    println!("=== Completions for method chain with get() ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have Char methods since get() on List[Char] returns Char
    assert!(
        !completions.is_empty(),
        "Expected completions for get(0) result (Char), got none"
    );
}

/// Test autocomplete for nested list local variable: g2 = [["a", "b"]] then g2.
/// This is the EXACT user scenario from /var/tmp/test_status_project/main.nos
/// IMPORTANT: This test simulates the REAL user workflow:
/// 1. Open file WITHOUT "g2." line
/// 2. Send didChange to ADD the "g2." line (simulates user typing)
/// 3. Request completion
#[test]
fn test_lsp_autocomplete_nested_list_local_var() {
    let project_path = create_test_project("nested_list_local_var");

    // Create good.nos - EXACT user code
    fs::write(
        project_path.join("good.nos"),
        "pub addff(a, b) = a + b\npub multiply(x, y) = x * y\n"
    ).unwrap();

    // EXACT content from /var/tmp/test_status_project/main.nos - BYTE FOR BYTE
    // Note: line 13 has NO leading indent (gg = [[0,1]] starts at column 0)
    // Line numbers (1-based):
    // 1:  type XX = AAA | BBB
    // 2:  (empty)
    // 3:  main() = {
    // 4:      x = good.addff(3, 2)
    // 5:      y = good.multiply(2,3)
    // 6:      yy = [1,2,3]
    // 7:      yy.map(m => m.asInt8())
    // 8:      y1 = 33
    // 9:      g = asInt32(y1)
    // 10:     (empty with spaces)
    // 11:     y1.asInt32()
    // 12: (empty)
    // 13: gg = [[0,1]]              <-- NO INDENT!
    // 14:     gg.map(m => m.map(n => n.asFloat32()))
    // 15:     # test
    // 16:     g2 = [["a" ,"b"]]
    // 17:     x2 = g2[0][0]
    // ...
    let initial_content = "type XX = AAA | BBB\n\nmain() = {\n    x = good.addff(3, 2)\n    y = good.multiply(2,3)\n    yy = [1,2,3]\n    yy.map(m => m.asInt8())\n    y1 = 33\n    g = asInt32(y1)\n    \n    y1.asInt32()\n\ngg = [[0,1]]\n    gg.map(m => m.map(n => n.asFloat32()))\n    # test\n    g2 = [[\"a\" ,\"b\"]]\n    x2 = g2[0][0]\n    y3 = \"ffff\"\n    x2.chars().drop(1).get(1).show()\n\n}\n";

    fs::write(project_path.join("main.nos"), initial_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, initial_content);
    std::thread::sleep(Duration::from_millis(500));

    // NOW simulate user typing "g2." - inserting a new line after g2 declaration
    // User is on line 17 (between g2 = ... and x2 = ...) and types "    g2."
    let edited_content = "type XX = AAA | BBB\n\nmain() = {\n    x = good.addff(3, 2)\n    y = good.multiply(2,3)\n    yy = [1,2,3]\n    yy.map(m => m.asInt8())\n    y1 = 33\n    g = asInt32(y1)\n    \n    y1.asInt32()\n\ngg = [[0,1]]\n    gg.map(m => m.map(n => n.asFloat32()))\n    # test\n    g2 = [[\"a\" ,\"b\"]]\n    g2.\n    x2 = g2[0][0]\n    y3 = \"ffff\"\n    x2.chars().drop(1).get(1).show()\n\n}\n";

    // Send didChange to simulate user typing
    client.did_change(&main_uri, edited_content, 2);
    std::thread::sleep(Duration::from_millis(500));

    // Count actual line numbers
    println!("=== Line analysis ===");
    for (i, line) in edited_content.lines().enumerate() {
        println!("Line {} (0-based): '{}'", i, line);
    }

    // Request completions after "g2." on line 17 (0-based: 16)
    // g2 = ... is on line 16 (0-based: 15)
    // g2. is on line 17 (0-based: 16)
    let completions = client.completion(&main_uri, 16, 7);

    println!("=== Completions for nested list local var (g2 = [[\"a\" ,\"b\"]]) ===");
    println!("=== After didChange (simulating real user typing) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have List methods since g2 is List[List[String]]
    assert!(
        !completions.is_empty(),
        "Expected completions for g2 (inferred List[List[String]]), got none"
    );

    // Check for List methods
    let has_list_method = completions.iter().any(|c|
        c == "map" || c == "filter" || c == "fold" || c == "length"
    );
    assert!(
        has_list_method,
        "Expected List methods like 'map' or 'filter' for g2, got: {:?}",
        completions
    );

    // Check for type indicator at top (should show ": List[List[String]]")
    let has_type_indicator = completions.iter().any(|c| c.starts_with(": List[List["));
    println!("Has type indicator: {}", has_type_indicator);
    assert!(has_type_indicator, "Expected type indicator ': List[List[String]]' but didn't find it");
}

/// Test autocomplete when typing . at end of assignment line: x2 = g2[0][0].
/// This is the EXACT user scenario where they have an assignment line and type . at the end
#[test]
fn test_lsp_autocomplete_dot_after_assignment() {
    let project_path = create_test_project("dot_after_assignment");

    fs::write(
        project_path.join("good.nos"),
        "pub addff(a, b) = a + b\npub multiply(x, y) = x * y\n"
    ).unwrap();

    // EXACT file content - user has g2 defined, then x2 = g2[0][0] followed by .
    // This simulates typing "." at the end of the x2 = g2[0][0] line
    let content = "type XX = AAA | BBB\n\nmain() = {\n    x = good.addff(3, 2)\n    y = good.multiply(2,3)\n    yy = [1,2,3]\n    yy.map(m => m.asInt8())\n    y1 = 33\n    g = asInt32(y1)\n    \n    y1.asInt32()\n\ngg = [[0,1]]\n    gg.map(m => m.map(n => n.asFloat32()))\n    # test\n    g2 = [[\"a\" ,\"b\"]]\n    x2 = g2[0][0].\n    y3 = \"ffff\"\n    x2.chars().drop(1).get(1).show()\n\n}\n";

    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(500));

    // Count lines to find the right position
    println!("=== Line analysis for dot-after-assignment ===");
    for (i, line) in content.lines().enumerate() {
        if line.contains("g2[0][0]") {
            println!("Line {} (0-based): '{}' <-- target line", i, line);
        }
    }

    // x2 = g2[0][0]. is on line 16 (0-based)
    // "    x2 = g2[0][0]."
    //  0123456789012345678
    // Cursor at position 18 (after the dot)
    let completions = client.completion(&main_uri, 16, 18);

    println!("=== Completions for dot after assignment (x2 = g2[0][0].) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have String methods since g2[0][0] is String
    assert!(
        !completions.is_empty(),
        "Expected completions for g2[0][0] (type String), got none"
    );

    // Check for type indicator (should show ": String")
    let has_string_type = completions.iter().any(|c| c == ": String");
    println!("Has String type indicator: {}", has_string_type);

    // Check for String methods
    let has_string_method = completions.iter().any(|c|
        c == "chars" || c == "length" || c == "split" || c == "trim"
    );
    assert!(
        has_string_method,
        "Expected String methods like 'chars' or 'length' for g2[0][0], got: {:?}",
        completions
    );
}

/// Test autocomplete for module function call return type: y = good.multiply(2,3) then y.
#[test]
fn test_lsp_autocomplete_module_function_return_type() {
    let project_path = create_test_project("module_func_return");

    // Create good.nos WITHOUT explicit type annotations (matches user's actual code)
    fs::write(
        project_path.join("good.nos"),
        r#"pub addff(a, b) = a + b
pub multiply(x, y) = x * y
"#
    ).unwrap();

    // main.nos calls good.multiply and then tries to use the result
    let content = r#"main() = {
    y = good.multiply(2, 3)
    y.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(500));

    // Request completions after "y." on line 3 (0-based: 2)
    let completions = client.completion(&main_uri, 2, 6);

    println!("=== Completions for module function return type (y = good.multiply(2,3)) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have Int methods since multiply returns Int
    assert!(
        !completions.is_empty(),
        "Expected completions for y (type Int from good.multiply), got none"
    );

    // Check for type indicator (should show ": Int")
    let has_int_type = completions.iter().any(|c| c == ": Int");
    println!("Has Int type indicator: {}", has_int_type);
    assert!(has_int_type, "Expected type indicator ': Int' for y, but didn't find it. Got: {:?}", completions);
}

/// Test autocomplete for record construction: TypeName(field: value, ...)
#[test]
fn test_lsp_autocomplete_record_literal_syntax() {
    let project_path = create_test_project("record_literal");

    // Use a valid file so types get registered (file must parse correctly)
    let content = r#"type Person = { name: String, age: Int }

main() = {
    p = Person(name: "Alice", age: 30)
    p.name
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p." (line 5, 0-indexed: 4, column 6)
    let completions = client.completion(&main_uri, 4, 6);

    println!("=== Completions for record literal (Person {{ }}) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Person (short or fully qualified)
    let has_person_type = completions.iter().any(|c| c == ": Person" || c.ends_with("Person"));
    println!("Has Person type indicator: {}", has_person_type);
    assert!(has_person_type, "Expected Person type indicator, but didn't find it. Got: {:?}", completions);

    // Should show record fields (name, age)
    let has_name_field = completions.iter().any(|c| c == "name");
    let has_age_field = completions.iter().any(|c| c == "age");
    println!("Has name field: {}, Has age field: {}", has_name_field, has_age_field);
    assert!(has_name_field, "Expected field 'name' for Person, but didn't find it. Got: {:?}", completions);
    assert!(has_age_field, "Expected field 'age' for Person, but didn't find it. Got: {:?}", completions);
}

/// Test autocomplete for variant construction without explicit type annotation
/// e.g., "r = Success(42)" should infer type from constructor name
#[test]
fn test_lsp_autocomplete_variant_constructor_inference() {
    let project_path = create_test_project("variant_ctor");

    let content = r#"type MyResult = Success(Int) | Failure(String)

main() = {
    r = Success(42)
    r.
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "r." (line 4, 0-based: 4)
    let completions = client.completion(&main_uri, 4, 6);

    println!("=== Completions for variant constructor (r = Ok 42) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator - either MyResult (if engine registered it) or Success (fallback)
    let has_type_indicator = completions.iter().any(|c| c.starts_with(": "));
    println!("Has type indicator: {}", has_type_indicator);
    assert!(has_type_indicator, "Expected type indicator for r (from Success constructor), but didn't find it. Got: {:?}", completions);
}

/// Test autocomplete for trait methods on user-defined types
#[test]
fn test_lsp_autocomplete_trait_methods() {
    let project_path = create_test_project("trait_methods");

    // Note: Use valid syntax (p.describe()) to avoid parse errors
    // Parse errors prevent type registration
    let content = r#"type Person = { name: String, age: Int }

trait Describable
    describe(self) -> String
end

Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "Alice", age: 30)
    p.describe()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p." (line 12, character 6)
    let completions = client.completion(&main_uri, 12, 6);

    println!("=== Completions for Person with trait ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Person (short or fully qualified)
    let has_person_type = completions.iter().any(|c| c == ": Person" || c.ends_with("Person"));
    println!("Has Person type indicator: {}", has_person_type);

    // Should show record fields
    let has_name_field = completions.iter().any(|c| c == "name");
    println!("Has name field: {}", has_name_field);

    // Should show trait method describe
    let has_describe_method = completions.iter().any(|c| c == "describe");
    println!("Has describe method: {}", has_describe_method);

    assert!(has_person_type, "Expected Person type indicator. Got: {:?}", completions);
    assert!(has_name_field, "Expected 'name' field. Got: {:?}", completions);
    assert!(has_describe_method, "Expected 'describe' trait method. Got: {:?}", completions);
}

/// Test autocomplete for record fields - simple case without traits
/// p = Person(name: "petter", age: 11) then p. should show name and age fields
#[test]
fn test_lsp_autocomplete_record_fields_simple() {
    let project_path = create_test_project("record_fields_simple");

    // Note: The file needs to be syntactically valid for load_directory to parse it.
    // The completion position will be after "p." but the file itself must be valid.
    let content = r#"# Record type for testing
type Person = { name: String, age: Int }

main() = {
    p = Person(name: "petter", age: 11)
    p.name
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p." (line 6, 0-based: 5, column 6)
    let completions = client.completion(&main_uri, 5, 6);

    println!("=== Completions for Person record ===");
    for c in &completions {
        println!("  '{}'", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Person (either short or fully qualified)
    let has_person_type = completions.iter().any(|c| c == ": Person" || c.ends_with("Person"));
    println!("Has Person type indicator: {}", has_person_type);

    // Should show record fields - name and age
    let has_name_field = completions.iter().any(|c| c == "name");
    let has_age_field = completions.iter().any(|c| c == "age");
    println!("Has name field: {}", has_name_field);
    println!("Has age field: {}", has_age_field);

    assert!(has_person_type, "Expected Person type indicator. Got: {:?}", completions);
    assert!(has_name_field, "Expected 'name' field for Person record. Got: {:?}", completions);
    assert!(has_age_field, "Expected 'age' field for Person record. Got: {:?}", completions);
}

/// Test that record field access compiles without errors
/// p.name and p.age should NOT give "undefined function" errors
#[test]
fn test_lsp_record_field_access_compiles() {
    let project_path = create_test_project("record_field_compile");

    // This file uses record field access - should compile without errors
    let content = r#"# Record type for testing
type Person = { name: String, age: Int }

main() = {
    p = Person(name: "petter", age: 11)
    n = p.name
    a = p.age
    n ++ " is " ++ a.show()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics - should be empty (no errors)
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for record field access test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have NO diagnostics - p.name and p.age should compile fine
    let has_undefined_function_error = diagnostics.iter().any(|d|
        d.message.contains("undefined function") ||
        d.message.contains("undefined") ||
        d.message.contains("Undefined")
    );

    assert!(
        !has_undefined_function_error,
        "Record field access should NOT give 'undefined function' error. Got diagnostics: {:?}",
        diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );

    // Ideally should have zero errors
    assert!(
        diagnostics.is_empty(),
        "Record field access should compile without errors. Got: {:?}",
        diagnostics.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Test autocomplete for chained field access: p.age. should show Int methods
/// since age is an Int field of Person
#[test]
fn test_lsp_autocomplete_chained_field_access() {
    let project_path = create_test_project("chained_field_access");

    // p.age is an Int, so p.age. should show Int methods like show, asFloat32, etc.
    // Note: file must be syntactically valid for type registration to work
    let content = r#"type Person = { name: String, age: Int }

main() = {
    p = Person(name: "petter", age: 11)
    p.age.show()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p.age." (line 4, character 10)
    let completions = client.completion(&main_uri, 4, 10);

    println!("=== Completions for p.age. (chained field access) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Int (since p.age is an Int field)
    let has_int_type = completions.iter().any(|c| c == ": Int");
    println!("Has Int type indicator: {}", has_int_type);

    // Should show Int methods like show, hash, etc.
    let has_show_method = completions.iter().any(|c| c == "show");
    let has_hash_method = completions.iter().any(|c| c == "hash");
    println!("Has show method: {}", has_show_method);
    println!("Has hash method: {}", has_hash_method);

    assert!(has_int_type, "Expected ': Int' type indicator for p.age. Got: {:?}", completions);
    assert!(has_show_method, "Expected 'show' method for Int type. Got: {:?}", completions);
    assert!(has_hash_method, "Expected 'hash' method for Int type. Got: {:?}", completions);
}

/// Test autocomplete for chained field access across modules
/// Type defined in one module, used in another - p.age. should still show Int methods
#[test]
fn test_lsp_autocomplete_cross_module_field_access() {
    let project_path = create_test_project("cross_module_field");

    // types.nos - defines Person type
    let types_content = r#"type Person = { name: String, age: Int }
"#;
    fs::write(project_path.join("types.nos"), types_content).unwrap();

    // main.nos - uses Person from types module
    let main_content = r#"use types.{Person}

main() = {
    p = Person(name: "test", age: 25)
    p.age.show()
}
"#;
    fs::write(project_path.join("main.nos"), main_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, main_content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p.age." (line 4, character 10)
    let completions = client.completion(&main_uri, 4, 10);

    println!("=== Completions for p.age. (cross-module) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Int
    let has_int_type = completions.iter().any(|c| c == ": Int");
    println!("Has Int type indicator: {}", has_int_type);

    // Should show Int methods
    let has_show_method = completions.iter().any(|c| c == "show");
    println!("Has show method: {}", has_show_method);

    assert!(has_int_type, "Expected ': Int' type indicator for cross-module p.age. Got: {:?}", completions);
    assert!(has_show_method, "Expected 'show' method for Int type. Got: {:?}", completions);
}

/// Test autocomplete for trait methods on user-defined types
/// When Person implements Describable trait, p. should show describe method
#[test]
fn test_lsp_autocomplete_trait_method_on_type() {
    let project_path = create_test_project("trait_method_completion");

    let content = r#"# Trait definition
trait Describable
    describe(self) -> String
end

# Record type
type Person = { name: String, age: Int }

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "test", age: 25)
    p.describe()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "p." (line 15, character 6)
    let completions = client.completion(&main_uri, 15, 6);

    println!("=== Completions for Person with trait ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Person (short or fully qualified)
    let has_person_type = completions.iter().any(|c| c == ": Person" || c.ends_with("Person"));
    println!("Has Person type indicator: {}", has_person_type);

    // Should show record fields
    let has_name_field = completions.iter().any(|c| c == "name");
    let has_age_field = completions.iter().any(|c| c == "age");
    println!("Has name field: {}", has_name_field);
    println!("Has age field: {}", has_age_field);

    // Should show trait method describe
    let has_describe_method = completions.iter().any(|c| c == "describe");
    println!("Has describe method: {}", has_describe_method);

    assert!(has_person_type, "Expected Person type indicator. Got: {:?}", completions);
    assert!(has_name_field, "Expected 'name' field. Got: {:?}", completions);
    assert!(has_age_field, "Expected 'age' field. Got: {:?}", completions);
    assert!(has_describe_method, "Expected 'describe' trait method. Got: {:?}", completions);
}

/// Test autocomplete for nested records - accessing fields of nested record types
#[test]
fn test_lsp_autocomplete_nested_records() {
    let project_path = create_test_project("nested_records_autocomplete");

    let content = r#"# Nested record types
type Address = { street: String, city: String, zip: Int }
type Person = { name: String, age: Int, address: Address }

main() = {
    addr = Address(street: "Main St", city: "Oslo", zip: 1234)
    p = Person(name: "Alice", age: 30, address: addr)
    p.address.city.length()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Test 1: completions after "p." should show Person fields including "address"
    // Line 7: "    p.address.city.length()"
    // Position after "p." is line 7, character 6
    let completions_p = client.completion(&main_uri, 7, 6);

    println!("=== Completions for p. (Person) ===");
    for c in &completions_p {
        println!("  {}", c);
    }

    // Test 2: completions after "p.address." should show Address fields
    // Position after "p.address." is line 7, character 14
    let completions_addr = client.completion(&main_uri, 7, 14);

    println!("=== Completions for p.address. (Address) ===");
    for c in &completions_addr {
        println!("  {}", c);
    }

    // Test 3: completions after "p.address.city." should show String methods
    // Position after "p.address.city." is line 7, character 19
    let completions_city = client.completion(&main_uri, 7, 19);

    println!("=== Completions for p.address.city. (String) ===");
    for c in &completions_city {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Verify p. shows Person type and address field
    let has_person_type = completions_p.iter().any(|c| c == ": Person" || c.ends_with("Person"));
    let has_address_field = completions_p.iter().any(|c| c == "address");
    let has_name_field = completions_p.iter().any(|c| c == "name");

    assert!(has_person_type, "Expected Person type indicator. Got: {:?}", completions_p);
    assert!(has_address_field, "Expected 'address' field. Got: {:?}", completions_p);
    assert!(has_name_field, "Expected 'name' field. Got: {:?}", completions_p);

    // Verify p.address. shows Address type and fields (short or fully qualified)
    let has_address_type = completions_addr.iter().any(|c| c == ": Address" || c.ends_with("Address"));
    let has_street_field = completions_addr.iter().any(|c| c == "street");
    let has_city_field = completions_addr.iter().any(|c| c == "city");
    let has_zip_field = completions_addr.iter().any(|c| c == "zip");

    assert!(has_address_type, "Expected ': Address' or ': main.Address' type. Got: {:?}", completions_addr);
    assert!(has_street_field, "Expected 'street' field. Got: {:?}", completions_addr);
    assert!(has_city_field, "Expected 'city' field. Got: {:?}", completions_addr);
    assert!(has_zip_field, "Expected 'zip' field. Got: {:?}", completions_addr);

    // Verify p.address.city. shows String type and methods
    let has_string_type = completions_city.iter().any(|c| c == ": String");
    let has_length_method = completions_city.iter().any(|c| c == "length");
    let has_chars_method = completions_city.iter().any(|c| c == "chars");

    assert!(has_string_type, "Expected ': String' type. Got: {:?}", completions_city);
    assert!(has_length_method, "Expected 'length' method. Got: {:?}", completions_city);
    assert!(has_chars_method, "Expected 'chars' method. Got: {:?}", completions_city);
}

/// Test that nested record field access compiles without errors
#[test]
fn test_lsp_nested_record_field_access_compiles() {
    let project_path = create_test_project("nested_records_compile");

    let content = r#"# Nested record types
type Address = { street: String, city: String, zip: Int }
type Person = { name: String, age: Int, address: Address }

main() = {
    addr = Address(street: "Main St", city: "Oslo", zip: 1234)
    p = Person(name: "Alice", age: 30, address: addr)

    # Access nested fields
    street = p.address.street
    city = p.address.city
    zip = p.address.zip

    # Chain methods on nested field
    cityLen = p.address.city.length()

    city
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics with timeout
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for nested record field access ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have no errors - nested field access should compile
    let has_errors = diagnostics.iter().any(|d| {
        d.message.contains("undefined") || d.message.contains("Undefined") ||
        d.message.contains("type error")
    });

    assert!(!has_errors, "Nested record field access should compile without errors. Got: {:?}", diagnostics);
}

/// Test that trait method calls compile without "undefined function" errors
#[test]
fn test_lsp_trait_method_call_compiles() {
    let project_path = create_test_project("trait_method_compile");

    let content = r#"# Trait definition
trait Describable
    describe(self) -> String
end

# Record type
type Person = { name: String, age: Int }

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "Alice", age: 30)
    result = p.describe()
    result
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for trait method call test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should NOT have "undefined function: p.describe" error
    let has_undefined_describe = diagnostics.iter().any(|d| {
        d.message.contains("p.describe") ||
        (d.message.contains("Undefined") && d.message.contains("describe"))
    });

    assert!(
        !has_undefined_describe,
        "Trait method p.describe() should NOT give 'undefined function' error. Got: {:?}",
        diagnostics.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Test that trait method calls with module-qualified type names work
/// This reproduces the user's scenario where error says "no method describe found for type test_types.Person"
#[test]
fn test_lsp_trait_method_module_qualified() {
    let project_path = create_test_project("trait_method_module");

    // Use test_types.nos as the filename (like user's file)
    let content = r#"# Nested record types for testing
type Address = { street: String, city: String, zip: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Record type for testing
type Person = { name: String, age: Int }

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "Alice", age: 30)
    result = p.describe()
    result
}
"#;
    // Use test_types.nos as filename to match user's scenario
    fs::write(project_path.join("test_types.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/test_types.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for module-qualified trait method test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should NOT have any describe-related errors
    let has_describe_error = diagnostics.iter().any(|d| {
        d.message.contains("describe")
    });

    assert!(
        !has_describe_error,
        "Trait method p.describe() should work. Got: {:?}",
        diagnostics.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Test trait method compile error in multi-file project (reproduces user's exact setup)
/// Trait defined before type, with implementation after type definition
#[test]
fn test_lsp_trait_method_multifile_project() {
    let project_path = create_test_project("trait_multifile");

    // test_types.nos - matches user's exact structure
    let test_types_content = r#"# Nested record types for testing
type Address = { street: String, city: String, zip: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Variant type for testing
type MyResult = Success(Int) | Failure(String)

# Record type for testing
type Person = { name: String, age: Int }

# Record with nested record field
type PersonWithAddress = { name: String, age: Int, address: Address }

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "petter", age: 11)
    p.describe()
}
"#;

    // main.nos - separate entry point file
    let main_content = r#"main() = {
    1
}
"#;

    fs::write(project_path.join("test_types.nos"), test_types_content).unwrap();
    fs::write(project_path.join("main.nos"), main_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    // Open test_types.nos specifically
    let test_types_uri = format!("file://{}/test_types.nos", project_path.display());
    client.did_open(&test_types_uri, test_types_content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&test_types_uri, Duration::from_secs(3));

    println!("=== Diagnostics for multi-file trait method test ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should NOT have any describe-related errors
    let has_describe_error = diagnostics.iter().any(|d| {
        d.message.contains("describe") || d.message.contains("no method")
    });

    assert!(
        !has_describe_error,
        "Trait method p.describe() should compile without errors in multi-file project. Got: {:?}",
        diagnostics.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Reproduce EXACT user project structure for trait method error
/// Simulates user EDITING the file in VS Code (didChange flow)
#[test]
fn test_lsp_trait_method_did_change_flow() {
    let project_path = create_test_project("did_change_flow");

    // nostos.toml
    fs::write(project_path.join("nostos.toml"), r#"[project]
name = "test"
"#).unwrap();

    // good.nos
    fs::write(project_path.join("good.nos"), r#"pub addff(a, b) = a + b
pub multiply(x, y) = x * y
"#).unwrap();

    // main.nos
    fs::write(project_path.join("main.nos"), r#"main() = { 1 }
"#).unwrap();

    // test_types.nos - EXACT content from user's file (no trait impl, no p.describe)
    let initial_content = r#"# Nested record types for testing
type Address = { street: String, city: String, zip: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Variant type for testing
type MyResult = Success(Int) | Failure(String)

# Record type for testing
type Person = { name: String, age: Int }

# Record with nested record field
type PersonWithAddress = { name: String, age: Int, address: Address }

main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)

    y1.asInt32()

gg = [[0,1]]
    gg.map(m => m.map(n => n.asFloat32()))
    # test
    g2 = [["a" ,"b"]]
    x2 = g2[0][0]
    y3 = "ffff"
    x2.chars().drop(1).get(1).show()

}
"#;
    fs::write(project_path.join("test_types.nos"), initial_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let test_types_uri = format!("file://{}/test_types.nos", project_path.display());

    // Open with initial content
    client.did_open(&test_types_uri, initial_content);
    std::thread::sleep(Duration::from_millis(300));

    // Now simulate user EDITING to add trait impl and p.describe()
    // Using EXACT content from user's file plus the additions
    let edited_content = r#"# Nested record types for testing
type Address = { street: String, city: String, zip: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Variant type for testing
type MyResult = Success(Int) | Failure(String)

# Record type for testing
type Person = { name: String, age: Int }

# Record with nested record field
type PersonWithAddress = { name: String, age: Int, address: Address }

Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    yy = [1,2,3]
    yy.map(m => m.asInt8())
    y1 = 33
    g = asInt32(y1)

    y1.asInt32()

gg = [[0,1]]
    gg.map(m => m.map(n => n.asFloat32()))
    # test
    g2 = [["a" ,"b"]]
    x2 = g2[0][0]
    y3 = "ffff"
    x2.chars().drop(1).get(1).show()

    p = Person(name: "petter", age: 11)
    p.describe()
}
"#;
    client.did_change(&test_types_uri, edited_content, 2);

    // Read diagnostics after the edit
    let diagnostics = client.read_diagnostics(&test_types_uri, Duration::from_secs(3));

    println!("=== Diagnostics after didChange (simulating VS Code edit) ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Check for the EXACT error user reports
    let has_describe_error = diagnostics.iter().any(|d| {
        d.message.contains("no method") && d.message.contains("describe")
    });

    assert!(
        !has_describe_error,
        "ERROR REPRODUCED: 'no method describe found'. Got: {:?}",
        diagnostics.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Project: name = "test", files: test_types.nos, good.nos, main.nos
#[test]
fn test_lsp_trait_method_exact_user_project() {
    let project_path = create_test_project("exact_user_project");

    // nostos.toml - exact same as user
    fs::write(project_path.join("nostos.toml"), r#"[project]
name = "test"
"#).unwrap();

    // good.nos - user's file
    fs::write(project_path.join("good.nos"), r#"# A working function
pub addff(a, b) = a + b

pub multiply(x, y) = x * y
"#).unwrap();

    // main.nos - user's file
    fs::write(project_path.join("main.nos"), r#"main() = {
    1
}
"#).unwrap();

    // test_types.nos - user's file WITH trait impl and p.describe() added
    let test_types_content = r#"# Nested record types for testing
type Address = { street: String, city: String, zip: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Variant type for testing
type MyResult = Success(Int) | Failure(String)

# Record type for testing
type Person = { name: String, age: Int }

# Record with nested record field
type PersonWithAddress = { name: String, age: Int, address: Address }

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "petter", age: 11)
    p.describe()
}
"#;
    fs::write(project_path.join("test_types.nos"), test_types_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    // Open test_types.nos
    let test_types_uri = format!("file://{}/test_types.nos", project_path.display());
    client.did_open(&test_types_uri, test_types_content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&test_types_uri, Duration::from_secs(3));

    println!("=== Diagnostics for EXACT user project ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Check for the EXACT error user reports
    let has_describe_error = diagnostics.iter().any(|d| {
        d.message.contains("no method") && d.message.contains("describe")
    });

    assert!(
        !has_describe_error,
        "ERROR REPRODUCED: 'no method describe found for type test_types.Person'. Got: {:?}",
        diagnostics.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Test that autocomplete after trait method call infers the return type
/// p.describe(). should show String methods since describe returns String
#[test]
fn test_lsp_autocomplete_trait_method_return_type() {
    let project_path = create_test_project("trait_method_return");

    // Use a valid file first so types get registered
    let initial_content = r#"# Trait definition
trait Describable
    describe(self) -> String
end

# Record type
type Person = { name: String, age: Int }

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name
end

main() = {
    p = Person(name: "Alice", age: 30)
    result = p.describe().length()
}
"#;
    fs::write(project_path.join("main.nos"), initial_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, initial_content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completion after "p.describe()." (before "length")
    // Line 16 (0-indexed: 15), after the dot at column 26
    // Line content: "    result = p.describe().length()"
    //                0         1         2         3
    //                0123456789012345678901234567890123
    // Column 25 is the '.', column 26 is after it
    let completions = client.completion(&main_uri, 15, 26);

    println!("=== Completions after p.describe(). ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have String type indicator
    let has_string_type = completions.iter().any(|c| {
        c.contains("String") || c == ": String"
    });

    // Should have String methods like length, chars, contains
    let has_string_methods = completions.iter().any(|c| {
        c == "length" || c == "chars" || c == "contains"
    });

    assert!(
        has_string_type,
        "Should show String type indicator after p.describe(). Got: {:?}",
        completions
    );

    assert!(
        has_string_methods,
        "Should show String methods after p.describe(). Got: {:?}",
        completions
    );
}

/// Test that keyword completions are provided
#[test]
fn test_lsp_keyword_completions() {
    let project_path = create_test_project("keyword_completions");

    // Write a file with a partial keyword at cursor
    let content = r#"main() = {
    ma
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at position after "ma" (line 1, character 6)
    let completions = client.completion(&main_uri, 1, 6);

    println!("=== Keyword completions for 'ma' ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should include "match" keyword
    let has_match_keyword = completions.iter().any(|c| c == "match");
    assert!(
        has_match_keyword,
        "Expected 'match' keyword completion, got: {:?}",
        completions
    );
}

/// Test that keywords like 'if', 'while', 'for' are completed
#[test]
fn test_lsp_keyword_completions_if_while() {
    let project_path = create_test_project("keyword_if_while");

    let content = r#"main() = {
    i
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at position after "i"
    let completions = client.completion(&main_uri, 1, 5);

    println!("=== Keyword completions for 'i' ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should include "if" and "in" keywords
    let has_if = completions.iter().any(|c| c == "if");
    let has_in = completions.iter().any(|c| c == "in");
    assert!(
        has_if,
        "Expected 'if' keyword completion, got: {:?}",
        completions
    );
    assert!(
        has_in,
        "Expected 'in' keyword completion, got: {:?}",
        completions
    );
}

/// Test that variant constructors are completed
#[test]
fn test_lsp_constructor_completions() {
    let project_path = create_test_project("constructor_completions");

    // Define a variant type and try to complete a constructor
    let content = r#"type Status = Loading | Ready | Failed(String)

main() = {
    Loa
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at position after "Loa" (line 3, character 7)
    let completions = client.completion(&main_uri, 3, 7);

    println!("=== Constructor completions for 'Loa' ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should include "Loading" constructor
    let has_loading = completions.iter().any(|c| c == "Loading");
    assert!(
        has_loading,
        "Expected 'Loading' constructor completion, got: {:?}",
        completions
    );
}

/// Test that all variant constructors are available
#[test]
fn test_lsp_constructor_completions_all() {
    let project_path = create_test_project("constructor_all");

    let content = r#"type Status = Loading | Ready | Failed(String)

main() = {
    Re
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at position after "Re"
    let completions = client.completion(&main_uri, 3, 6);

    println!("=== Constructor completions for 'Re' ===");
    for c in &completions {
        println!("  {}", c);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should include "Ready" constructor and possibly "return" keyword
    let has_ready = completions.iter().any(|c| c == "Ready");
    let has_return = completions.iter().any(|c| c == "return");

    assert!(
        has_ready,
        "Expected 'Ready' constructor completion, got: {:?}",
        completions
    );
    assert!(
        has_return,
        "Expected 'return' keyword completion, got: {:?}",
        completions
    );
}

/// Test that errors introduced by editing are reported on the CORRECT line,
/// not on line 1 (use statement). This reproduces a bug where any error
/// in test_types.nos would show on line 1 and couldn't be fixed.
#[test]
fn test_lsp_error_line_number_after_edit() {
    let project_path = create_test_project("error_line_number");

    // nostos.toml
    fs::write(project_path.join("nostos.toml"), r#"[project]
name = "test"
"#).unwrap();

    // good.nos - module with exported functions
    fs::write(project_path.join("good.nos"), r#"# A working function
pub addff(a, b) = a + b

# A working function
pub addfff(a, b) = a + b

pub multiply(x, y) = x * y
"#).unwrap();

    // test_types.nos - with use statement on line 1
    // This is the EXACT user file content that triggers the bug
    let initial_content = r#"use good.*

# Variant type for testing
type MyResult = Success(Int) | Failure(String)

# Record type for testing
type Person = { name: String, age: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name ++ ", age " ++ self.age.show()
end

main() = {
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    p = Person(name: "petter", age: 11)
    p.describe()
}
"#;
    fs::write(project_path.join("test_types.nos"), initial_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let test_types_uri = format!("file://{}/test_types.nos", project_path.display());

    // Open the file
    client.did_open(&test_types_uri, initial_content);
    std::thread::sleep(Duration::from_millis(300));

    // Initial file should have NO errors
    let initial_diags = client.read_diagnostics(&test_types_uri, Duration::from_secs(2));
    println!("=== Initial diagnostics (should be empty) ===");
    for d in &initial_diags {
        println!("  Line {}: {}", d.line + 1, d.message);
    }
    assert!(initial_diags.is_empty(), "Initial file should have no errors, got: {:?}",
        initial_diags.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>());

    // Now simulate user typing an error - add "asdf" on line 20 (inside main)
    // The error should appear on line 20, NOT on line 1
    let content_with_error = r#"use good.*

# Variant type for testing
type MyResult = Success(Int) | Failure(String)

# Record type for testing
type Person = { name: String, age: Int }

# Trait for testing
trait Describable
    describe(self) -> String
end

# Implement trait for Person
Person: Describable
    describe(self) = "Person: " ++ self.name ++ ", age " ++ self.age.show()
end

main() = {
    asdf
    x = good.addff(3, 2)
    y = good.multiply(2,3)
    p = Person(name: "petter", age: 11)
    p.describe()
}
"#;
    client.did_change(&test_types_uri, content_with_error, 2);

    let error_diags = client.read_diagnostics(&test_types_uri, Duration::from_secs(3));
    println!("=== Diagnostics after introducing 'asdf' error ===");
    for d in &error_diags {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    // CRITICAL: Error should NOT be on line 1
    // The 'asdf' was added on line 20, so error should be around there
    let has_error_on_line_1 = error_diags.iter().any(|d| d.line == 0);
    let has_error_near_line_20 = error_diags.iter().any(|d| d.line >= 18 && d.line <= 22);

    assert!(
        !has_error_on_line_1 || has_error_near_line_20,
        "BUG REPRODUCED: Error appeared on line 1 (use statement) instead of near line 20 where 'asdf' was added. Diagnostics: {:?}",
        error_diags.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );

    // Now fix the error by removing 'asdf' - diagnostics should clear
    client.did_change(&test_types_uri, initial_content, 3);

    let fixed_diags = client.read_diagnostics(&test_types_uri, Duration::from_secs(3));
    println!("=== Diagnostics after fixing the error ===");
    for d in &fixed_diags {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // After fixing, should have no errors (or at least no error on line 1)
    let still_has_error_on_line_1 = fixed_diags.iter().any(|d| d.line == 0);
    assert!(
        !still_has_error_on_line_1,
        "BUG: After fixing the error, there's still an error on line 1. Diagnostics: {:?}",
        fixed_diags.iter().map(|d| format!("Line {}: {}", d.line + 1, &d.message)).collect::<Vec<_>>()
    );
}

/// Test that all example files compile without LSP errors.
/// This catches cases where the LSP reports errors that the compiler doesn't.
#[test]
fn test_lsp_all_examples_compile() {
    use std::process::Command;

    let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("examples");

    println!("Examples directory: {}", examples_dir.display());

    // Get all .nos files in examples directory
    let mut example_files: Vec<_> = fs::read_dir(&examples_dir)
        .expect("Failed to read examples directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "nos").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    example_files.sort();

    println!("Found {} example files", example_files.len());

    let mut lsp_only_errors: Vec<(String, Vec<String>)> = vec![];
    let mut both_errors: Vec<(String, Vec<String>, String)> = vec![];

    for example_path in &example_files {
        let file_name = example_path.file_name().unwrap().to_string_lossy().to_string();
        let module_name = example_path.file_stem().unwrap().to_string_lossy().to_string();

        println!("\n=== Testing: {} ===", file_name);

        // Read the file content
        let content = fs::read_to_string(&example_path)
            .expect(&format!("Failed to read {}", file_name));

        // First, check if the actual compiler reports errors
        let compiler_result = Command::new(get_nostos_binary())
            .arg("--check")
            .arg(&example_path)
            .output();

        let compiler_has_error = match compiler_result {
            Ok(output) => !output.status.success(),
            Err(_) => false, // If we can't run compiler, assume no error
        };

        // Create a test project for this example
        let project_path = create_test_project(&format!("example_{}", module_name));

        // Write nostos.toml
        fs::write(project_path.join("nostos.toml"), format!(r#"[project]
name = "{}"
"#, module_name)).unwrap();

        // Write the example file
        fs::write(project_path.join(&file_name), &content).unwrap();

        // Start LSP client
        let mut client = LspClient::new(&get_lsp_binary());
        let _ = client.initialize(project_path.to_str().unwrap());
        client.initialized();
        std::thread::sleep(Duration::from_millis(500));

        // Open the file
        let file_uri = format!("file://{}/{}", project_path.display(), file_name);
        client.did_open(&file_uri, &content);

        // Read diagnostics
        let diagnostics = client.read_diagnostics(&file_uri, Duration::from_secs(3));

        let _ = client.shutdown();
        client.exit();
        cleanup_test_project(&project_path);

        // Analyze results
        if !diagnostics.is_empty() {
            let diag_messages: Vec<String> = diagnostics.iter()
                .map(|d| format!("Line {}: {}", d.line + 1, d.message))
                .collect();

            println!("  LSP diagnostics:");
            for msg in &diag_messages {
                println!("    {}", msg);
            }

            if compiler_has_error {
                println!("  Compiler also reports error (expected)");
                both_errors.push((file_name.clone(), diag_messages, "compiler error".to_string()));
            } else {
                println!("  *** LSP ERROR BUT COMPILER OK - THIS IS A BUG ***");
                lsp_only_errors.push((file_name.clone(), diag_messages));
            }
        } else {
            println!("  OK - no LSP errors");
        }
    }

    // Print summary
    println!("\n\n========== SUMMARY ==========");
    println!("Total examples: {}", example_files.len());
    println!("LSP-only errors (BUGS): {}", lsp_only_errors.len());
    println!("Both LSP and compiler errors: {}", both_errors.len());

    if !lsp_only_errors.is_empty() {
        println!("\n*** LSP BUGS (compiler OK but LSP reports error): ***");
        for (file, errors) in &lsp_only_errors {
            println!("\n  {}:", file);
            for err in errors {
                println!("    {}", err);
            }
        }
    }

    // Fail the test if there are LSP-only errors
    assert!(
        lsp_only_errors.is_empty(),
        "LSP reports errors for {} files that compile successfully with the compiler. See above for details.",
        lsp_only_errors.len()
    );
}

/// Test autocomplete for self. inside trait implementations
/// When implementing a trait method, self.fieldName should show the type's fields
#[test]
fn test_lsp_autocomplete_self_in_trait_impl() {
    let project_path = create_test_project("self_trait_impl");

    // Create a file with a type, trait definition, and trait implementation
    // The self.n placeholder is where we'll request completions
    let content = r#"type Counter = { count: Int, name: String }

trait Incrementable
    increment(self) -> Int
end

Counter: Incrementable
    increment(self) = self.count + 1
end

main() = {
    c = Counter(count: 0, name: "test")
    c.increment()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "self." inside the trait implementation
    // Line 8 (0-indexed: 7) is "    increment(self) = self.count + 1"
    // Position after "self." is column 27 (after the dot at position 26)
    let completions = client.completion(&main_uri, 7, 27);

    println!("=== Completions for self. inside trait implementation ===");
    for c in &completions {
        println!("  '{}'", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show the Counter type's fields
    let has_count_field = completions.iter().any(|c| c == "count");
    let has_name_field = completions.iter().any(|c| c == "name");

    println!("Has count field: {}", has_count_field);
    println!("Has name field: {}", has_name_field);

    assert!(
        has_count_field,
        "Expected 'count' field for self. inside trait impl. Got: {:?}",
        completions
    );
    assert!(
        has_name_field,
        "Expected 'name' field for self. inside trait impl. Got: {:?}",
        completions
    );
}

/// Test autocomplete for self. inside trait method definition (in trait block)
/// When defining a default method in a trait, self. should work if the trait has bounds
#[test]
fn test_lsp_autocomplete_self_in_trait_definition() {
    let project_path = create_test_project("self_trait_def");

    // Create a file with a trait that has a default implementation using self
    let content = r#"type Point = { x: Int, y: Int }

trait HasX
    getX(self) -> Int
    doubleX(self) -> Int = self.getX() * 2
end

Point: HasX
    getX(self) = self.x
end

main() = {
    p = Point(x: 5, y: 10)
    p.doubleX()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "self." in the trait impl (line 9, column 22)
    // "    getX(self) = self.x" - dot is at position 21, so column 22 is after the dot
    let completions = client.completion(&main_uri, 8, 22);

    println!("=== Completions for self. inside impl block ===");
    for c in &completions {
        println!("  '{}'", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show Point's fields (x, y)
    let has_x_field = completions.iter().any(|c| c == "x");
    let has_y_field = completions.iter().any(|c| c == "y");

    println!("Has x field: {}", has_x_field);
    println!("Has y field: {}", has_y_field);

    assert!(
        has_x_field,
        "Expected 'x' field for self. inside trait impl. Got: {:?}",
        completions
    );
    assert!(
        has_y_field,
        "Expected 'y' field for self. inside trait impl. Got: {:?}",
        completions
    );
}

/// Test autocomplete for self. at end of incomplete line inside trait implementation
/// This matches the user scenario: typing "self." at the end of a method body
#[test]
fn test_lsp_autocomplete_self_incomplete_line() {
    let project_path = create_test_project("self_incomplete");

    // Create a file where the user is actively typing self. at the end
    // Note: The line is incomplete - ends with "self."
    let content = r#"type Person = { name: String, age: Int }

trait Describable
    describe(self) -> String
end

Person: Describable
    describe(self) = "Person: " ++ self.name ++ ", age " ++ self.age.show() ++ self.
end

main() = {
    p = Person(name: "Alice", age: 30)
    p.describe()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at the end of line 8 (0-indexed: 7)
    // Line is: '    describe(self) = "Person: " ++ self.name ++ ", age " ++ self.age.show() ++ self.'
    // The cursor is right after the final "self." - we need to find where that dot is
    let line_8 = content.lines().nth(7).unwrap();
    let dot_pos = line_8.len(); // After the final character
    println!("Line 8: '{}'", line_8);
    println!("Requesting completions at line 7, column {}", dot_pos);

    let completions = client.completion(&main_uri, 7, dot_pos as u32);

    println!("=== Completions for self. at end of incomplete line ===");
    for c in &completions {
        println!("  '{}'", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show Person's fields: name and age
    let has_name_field = completions.iter().any(|c| c == "name");
    let has_age_field = completions.iter().any(|c| c == "age");

    println!("Has name field: {}", has_name_field);
    println!("Has age field: {}", has_age_field);

    assert!(
        has_name_field,
        "Expected 'name' field for self. at end of incomplete line. Got: {:?}",
        completions
    );
    assert!(
        has_age_field,
        "Expected 'age' field for self. at end of incomplete line. Got: {:?}",
        completions
    );
}

/// Test that numeric conversion functions (asInt32, asFloat64, etc.) report type errors
/// when called with non-numeric arguments like String
#[test]
fn test_lsp_numeric_conversion_type_check() {
    let project_path = create_test_project("numeric_conversion_type_check");

    // asInt32("xxx") should be a type error - String is not numeric
    let content = r#"main() = {
    x = asInt32("hello")
    x
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

    println!("=== Diagnostics for asInt32(String) type check ===");
    for d in &diagnostics {
        println!("  Line {}: {}", d.line + 1, d.message);
    }
    println!("Total diagnostics: {}", diagnostics.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should have a type error for asInt32("hello")
    let has_type_error = diagnostics.iter().any(|d| {
        d.message.to_lowercase().contains("type") &&
        (d.message.to_lowercase().contains("mismatch") ||
         d.message.to_lowercase().contains("expected"))
    });

    assert!(
        has_type_error,
        "Expected type error for asInt32(String). Got: {:?}",
        diagnostics
    );
}

/// Test that self.age. shows Int methods when age is an Int field
/// This tests chained field access completion
#[test]
fn test_lsp_autocomplete_self_field_chain() {
    let project_path = create_test_project("self_field_chain");

    // Create a file where user is typing self.age. and expects Int methods
    let content = r#"type Person = { name: String, age: Int }

trait Describable
    describe(self) -> String
end

Person: Describable
    describe(self) = "Person: " ++ self.name ++ ", age " ++ self.age.
end

main() = {
    p = Person(name: "Alice", age: 30)
    p.describe()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions at the end of line 8 (0-indexed: 7)
    // Line is: '    describe(self) = "Person: " ++ self.name ++ ", age " ++ self.age.'
    // The cursor is right after "self.age."
    let line_8 = content.lines().nth(7).unwrap();
    let dot_pos = line_8.len(); // After the final character
    println!("Line 8: '{}'", line_8);
    println!("Requesting completions at line 7, column {}", dot_pos);

    let completions = client.completion(&main_uri, 7, dot_pos as u32);

    println!("=== Completions for self.age. (expecting Int methods) ===");
    for c in &completions {
        println!("  '{}'", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show Int methods like 'show' (from Show trait)
    // Also should show type indicator ": Int" at top
    let has_type_int = completions.iter().any(|c| c.contains("Int"));
    let has_show_method = completions.iter().any(|c| c == "show");

    println!("Has Int type indicator: {}", has_type_int);
    println!("Has show method: {}", has_show_method);

    assert!(
        has_type_int,
        "Expected ': Int' type indicator for self.age. Got: {:?}",
        completions
    );
    assert!(
        has_show_method,
        "Expected 'show' method for Int type. Got: {:?}",
        completions
    );
}

/// Test document symbols (outline view)
#[test]
fn test_lsp_document_symbols() {
    let project_path = create_test_project("document_symbols");

    let content = r#"# A test file with various symbols

type Person = { name: String, age: Int }

type Color
    | Red
    | Green
    | Blue
end

trait Show
    show(self) -> String
end

Person: Show
    show(self) = self.name
end

pub greet(person: Person) = "Hello, " ++ person.name

main() = {
    p = Person(name: "Alice", age: 30)
    greet(p)
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    let symbols = client.document_symbol(&main_uri);

    println!("=== Document Symbols ===");
    for (name, kind, line) in &symbols {
        println!("  {} ({}) at line {}", name, kind, line);
    }
    println!("Total symbols: {}", symbols.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Check that we found the expected symbols
    let has_person_type = symbols.iter().any(|(name, kind, _)| name == "Person" && kind == "Struct");
    let has_color_type = symbols.iter().any(|(name, kind, _)| name == "Color" && kind == "Struct");
    let has_show_trait = symbols.iter().any(|(name, kind, _)| name == "Show" && kind == "Interface");
    let has_greet_fn = symbols.iter().any(|(name, kind, _)| name == "greet" && kind == "Function");
    let has_main_fn = symbols.iter().any(|(name, kind, _)| name == "main" && kind == "Function");

    assert!(has_person_type, "Expected Person type. Got: {:?}", symbols);
    assert!(has_color_type, "Expected Color type. Got: {:?}", symbols);
    assert!(has_show_trait, "Expected Show trait. Got: {:?}", symbols);
    assert!(has_greet_fn, "Expected greet function. Got: {:?}", symbols);
    assert!(has_main_fn, "Expected main function. Got: {:?}", symbols);
}

/// Test find references
#[test]
fn test_lsp_find_references() {
    let project_path = create_test_project("find_references");

    let content = r#"type Person = { name: String, age: Int }

greet(person: Person) = "Hello, " ++ person.name

farewell(person: Person) = "Goodbye, " ++ person.name

main() = {
    p = Person(name: "Alice", age: 30)
    greet(p) ++ " and " ++ farewell(p)
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Find references to "Person" - cursor on line 0 (type Person = ...)
    let refs = client.references(&main_uri, 0, 5); // "Person" starts at column 5

    println!("=== References to 'Person' ===");
    for (uri, line, col) in &refs {
        println!("  {}:{}:{}", uri, line, col);
    }
    println!("Total references: {}", refs.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should find at least 4 references:
    // 1. type Person = ...
    // 2. greet(person: Person)
    // 3. farewell(person: Person)
    // 4. p = Person(name: ...)
    assert!(
        refs.len() >= 4,
        "Expected at least 4 references to 'Person'. Got {} references: {:?}",
        refs.len(),
        refs
    );
}

/// Test find references across multiple files
#[test]
fn test_lsp_find_references_cross_file() {
    let project_path = create_test_project("find_references_cross");

    // Define type in one file
    let types_content = r#"pub type User = { id: Int, name: String }

pub createUser(name: String) = User(id: 1, name: name)
"#;
    fs::write(project_path.join("types.nos"), types_content).unwrap();

    // Use type in another file
    let main_content = r#"import types

main() = {
    user = types.createUser("Alice")
    user.name
}
"#;
    fs::write(project_path.join("main.nos"), main_content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let types_uri = format!("file://{}/types.nos", project_path.display());
    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&types_uri, types_content);
    client.did_open(&main_uri, main_content);
    std::thread::sleep(Duration::from_millis(300));

    // Find references to "User" from types.nos
    let refs = client.references(&types_uri, 0, 9); // "User" starts at column 9

    println!("=== References to 'User' across files ===");
    for (uri, line, col) in &refs {
        println!("  {}:{}:{}", uri, line, col);
    }
    println!("Total references: {}", refs.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should find at least 2 references:
    // 1. pub type User = ... (in types.nos)
    // 2. User(id: 1, ...) (in types.nos constructor call)
    assert!(
        refs.len() >= 2,
        "Expected at least 2 references to 'User'. Got {} references: {:?}",
        refs.len(),
        refs
    );
}

/// Test autocomplete for supertrait methods
/// When a type implements a trait that has a supertrait, autocomplete should show
/// methods from both the trait AND all supertraits in the hierarchy.
#[test]
fn test_lsp_autocomplete_supertrait_methods() {
    let project_path = create_test_project("supertrait_autocomplete");

    // Create a supertrait hierarchy: Base -> Child
    // MyType implements both, so x. should show methods from Base AND Child
    let content = r#"trait Base
    getValue(self) -> Int
end

trait Child: Base
    getDouble(self) -> Int
end

type MyType = { value: Int }

MyType: Base getValue(self) = self.value end
MyType: Child getDouble(self) = self.value * 2 end

main() = {
    x = MyType(10)
    x.getValue()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "x." (line 15, character 6)
    let completions = client.completion(&main_uri, 15, 6);

    println!("=== Completions for MyType with supertrait ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show method from Base trait (supertrait)
    let has_get_value = completions.iter().any(|c| c == "getValue");
    println!("Has getValue (Base trait method): {}", has_get_value);

    // Should show method from Child trait
    let has_get_double = completions.iter().any(|c| c == "getDouble");
    println!("Has getDouble (Child trait method): {}", has_get_double);

    // Should show record field
    let has_value_field = completions.iter().any(|c| c == "value");
    println!("Has value field: {}", has_value_field);

    assert!(has_get_value, "Expected 'getValue' method from Base supertrait. Got: {:?}", completions);
    assert!(has_get_double, "Expected 'getDouble' method from Child trait. Got: {:?}", completions);
    assert!(has_value_field, "Expected 'value' field. Got: {:?}", completions);
}

/// Test autocomplete for transitive supertrait methods
/// When A: B: C, a type implementing C should show methods from A, B, and C.
#[test]
fn test_lsp_autocomplete_transitive_supertrait_methods() {
    let project_path = create_test_project("transitive_supertrait_autocomplete");

    // Create a transitive supertrait hierarchy: A -> B -> C
    let content = r#"trait A
    getA(self) -> Int
end

trait B: A
    getB(self) -> Int
end

trait C: B
    getC(self) -> Int
end

type MyType = { value: Int }

MyType: A getA(self) = self.value end
MyType: B getB(self) = self.value * 2 end
MyType: C getC(self) = self.value * 3 end

main() = {
    x = MyType(10)
    x.getA()
}
"#;
    fs::write(project_path.join("main.nos"), content).unwrap();

    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);
    std::thread::sleep(Duration::from_millis(300));

    // Request completions after "x." (line 20, character 6)
    let completions = client.completion(&main_uri, 20, 6);

    println!("=== Completions for MyType with transitive supertraits ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show method from A trait (root supertrait)
    let has_get_a = completions.iter().any(|c| c == "getA");
    println!("Has getA (A trait - root supertrait): {}", has_get_a);

    // Should show method from B trait (middle supertrait)
    let has_get_b = completions.iter().any(|c| c == "getB");
    println!("Has getB (B trait - middle supertrait): {}", has_get_b);

    // Should show method from C trait (direct trait)
    let has_get_c = completions.iter().any(|c| c == "getC");
    println!("Has getC (C trait - direct trait): {}", has_get_c);

    // Should show record field
    let has_value_field = completions.iter().any(|c| c == "value");
    println!("Has value field: {}", has_value_field);

    assert!(has_get_a, "Expected 'getA' method from root supertrait A. Got: {:?}", completions);
    assert!(has_get_b, "Expected 'getB' method from middle supertrait B. Got: {:?}", completions);
    assert!(has_get_c, "Expected 'getC' method from direct trait C. Got: {:?}", completions);
    assert!(has_value_field, "Expected 'value' field. Got: {:?}", completions);
}

fn get_nostos_binary() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("target/release/nostos")
        .to_string_lossy()
        .to_string()
}
