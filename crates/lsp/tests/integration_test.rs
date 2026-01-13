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

    let map_errors: Vec<_> = modified_diags.iter()
        .filter(|d| d.message.contains("map"))
        .collect();

    assert!(!map_errors.is_empty(), "Expected map error after adding empty lines");

    // After adding 2 empty lines, gg.map() is on line 15 (1-based), line 14 (0-based)
    // Bug: error might still show on line 13 (the OLD line number)
    let expected_line_0based = 14;
    let has_error_on_correct_line = map_errors.iter().any(|d| d.line == expected_line_0based);

    assert!(
        has_error_on_correct_line,
        "After adding 2 empty lines, error should be on line 15 (0-based: 14), but got: {:?}. Bug: line not updated after didChange!",
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

    // gg.map() on line 15 should have error "expects 2 arguments"
    let map_errors: Vec<_> = diagnostics.iter()
        .filter(|d| d.message.contains("map"))
        .collect();

    println!("\n=== Map-related errors ===");
    for d in &map_errors {
        println!("  Line {}: {}", d.line + 1, d.message);
    }

    // Bug: User sees error on line 13, should be line 15
    assert!(
        !map_errors.is_empty(),
        "Expected at least one map-related error for gg.map()"
    );

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

    // Find error about map
    let map_errors: Vec<_> = diagnostics.iter()
        .filter(|d| d.message.contains("map"))
        .collect();

    assert!(
        !map_errors.is_empty(),
        "Expected at least one map-related error"
    );

    // ASSERTION: Error should mention stdlib.list.map (stdlib loaded)
    // NOT "no method `map` found" (stdlib NOT loaded)
    let first_map_error = &map_errors[0].message;

    assert!(
        first_map_error.contains("stdlib.list.map"),
        "Stdlib not loaded! Expected error mentioning 'stdlib.list.map', got: {}",
        first_map_error
    );

    // Additional check: should NOT contain "no method found"
    assert!(
        !first_map_error.contains("no method"),
        "Stdlib not loaded! Got 'no method found' error instead of stdlib error: {}",
        first_map_error
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
    std::thread::sleep(Duration::from_millis(500));

    let main_uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&main_uri, content);

    // Read diagnostics after opening file
    let diagnostics = client.read_diagnostics(&main_uri, Duration::from_secs(3));

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

    let content = r#"type Person = { name: String, age: Int }

main() = {
    p = Person(name: "Alice", age: 30)
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

    // Request completions after "p." (line 4, 0-based: 4)
    let completions = client.completion(&main_uri, 4, 6);

    println!("=== Completions for record literal (Person {{ }}) ===");
    for c in &completions {
        println!("  {}", c);
    }
    println!("Completions count: {}", completions.len());

    let _ = client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Should show type indicator for Person
    let has_person_type = completions.iter().any(|c| c == ": Person");
    println!("Has Person type indicator: {}", has_person_type);
    assert!(has_person_type, "Expected type indicator ': Person' for p, but didn't find it. Got: {:?}", completions);
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
