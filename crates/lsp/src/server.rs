#![allow(dead_code)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::match_result_ok)]
#![allow(clippy::map_entry)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::manual_strip)]
#![allow(clippy::question_mark)]
#![allow(clippy::manual_pattern_char_comparison)]
#![allow(clippy::writeln_empty_string)]
#![allow(clippy::needless_borrows_for_generic_args)]

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::io::Write;

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use nostos_repl::{ReplEngine, ReplConfig};
use tower_lsp::lsp_types::notification::Notification;

/// Custom notification for file status updates (for VS Code file decorations)
pub struct FileStatusNotification;

impl Notification for FileStatusNotification {
    type Params = serde_json::Value;
    const METHOD: &'static str = "nostos/fileStatus";
}

fn log(msg: &str) {
    eprintln!("{}", msg);
    std::io::stderr().flush().ok();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/nostos_lsp.log")
    {
        let _ = writeln!(f, "{}", msg);
    }
}

/// Information about a binding extracted from a line (for inlay hints)
struct BindingInfo {
    name: String,
    name_end: usize,   // Column where name ends (for hint placement)
    rhs_start: usize,  // Column where RHS expression starts
}

pub struct NostosLanguageServer {
    client: Client,
    engine: Mutex<Option<ReplEngine>>,
    /// Map from URI to document content (for unsaved changes)
    documents: DashMap<Url, String>,
    /// Map from URI to error diagnostics from recompile_file (to preserve when adding stale warnings)
    file_errors: DashMap<Url, Vec<Diagnostic>>,
    /// Root path of the workspace
    root_path: Mutex<Option<PathBuf>>,
    /// Flag to prevent double initialization
    initializing: AtomicBool,
    /// Files that have been modified but not yet compiled (dirty)
    dirty_files: DashMap<String, bool>,
}

impl NostosLanguageServer {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            engine: Mutex::new(None),
            documents: DashMap::new(),
            file_errors: DashMap::new(),
            root_path: Mutex::new(None),
            initializing: AtomicBool::new(false),
            dirty_files: DashMap::new(),
        }
    }

    /// Initialize the ReplEngine with the workspace
    fn init_engine(&self, root_path: &PathBuf) {
        let config = ReplConfig {
            enable_jit: false,
            num_threads: 1,
        };

        // Use consolidated initialization
        match ReplEngine::init_with_project(config, Some(root_path)) {
            Ok(engine) => {
                if engine.get_prelude_imports_count() == 0 {
                    eprintln!("Warning: Stdlib loaded but 0 prelude imports registered - stdlib may not have been found");
                }
                *self.engine.lock().unwrap() = Some(engine);
                *self.root_path.lock().unwrap() = Some(root_path.clone());
            }
            Err(e) => {
                eprintln!("Warning: Failed to initialize engine: {}", e);
                // Create engine without project loading
                let fallback_config = ReplConfig { enable_jit: false, num_threads: 1 };
                let mut engine = ReplEngine::new(fallback_config);
                let _ = engine.load_stdlib();
                *self.engine.lock().unwrap() = Some(engine);
                *self.root_path.lock().unwrap() = Some(root_path.clone());
            }
        }
    }

    /// Send file status notification to VS Code for file decorations
    /// Status: "ok" (green), "error" (red), "stale" (blue), "dirty" (yellow)
    async fn send_file_status_notification(&self) {
        // Collect all data synchronously first, then send notification
        let params = {
            let root_path = self.root_path.lock().unwrap().clone();
            let Some(root) = root_path else { return };

            let mut file_statuses: Vec<serde_json::Value> = Vec::new();

            // Get compile status for all modules
            let engine_guard = self.engine.lock().unwrap();
            if let Some(engine) = engine_guard.as_ref() {
                // Collect all .nos files in the project
                if let Ok(entries) = std::fs::read_dir(&root) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().map(|e| e == "nos").unwrap_or(false) {
                            let file_path = path.to_string_lossy().to_string();
                            let module_name = path.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown");

                            // Check if file is dirty (modified but not compiled)
                            let is_dirty = self.dirty_files.contains_key(&file_path);

                            // Get compile status
                            let status = if is_dirty {
                                "dirty"
                            } else {
                                // Check compile status from engine
                                let mut has_error = false;
                                let mut is_stale = false;

                                for (fn_name, compile_status) in engine.get_all_compile_status_detailed() {
                                    if fn_name.starts_with(&format!("{}.", module_name)) || fn_name == module_name {
                                        match compile_status {
                                            nostos_repl::CompileStatus::CompileError(_) => has_error = true,
                                            nostos_repl::CompileStatus::Stale { .. } => is_stale = true,
                                            _ => {}
                                        }
                                    }
                                }

                                if has_error {
                                    "error"
                                } else if is_stale {
                                    "stale"
                                } else {
                                    "ok"
                                }
                            };

                            file_statuses.push(serde_json::json!({
                                "path": file_path,
                                "status": status
                            }));
                        }
                    }
                }
            }
            // engine_guard dropped here

            serde_json::json!({
                "files": file_statuses
            })
        };

        // Send custom notification (after all locks are released)
        self.client.send_notification::<FileStatusNotification>(params).await;
    }

    /// Publish diagnostics for a specific file (currently unused, kept for future use)
    #[allow(dead_code)]
    async fn publish_file_diagnostics(&self, uri: &Url, file_path: &str) {
        // Try to get file content for better error location
        let file_content = std::fs::read_to_string(file_path).ok();
        let content_ref = file_content.as_deref();

        let diagnostics = {
            let engine_guard = self.engine.lock().unwrap();
            let Some(engine) = engine_guard.as_ref() else {
                return;
            };

            if engine.file_has_errors(file_path) {
                // Get more specific error info from compile status
                let module_name = std::path::Path::new(file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                // Check function-level status for this module
                let mut errors = Vec::new();
                for (fn_name, status_str) in engine.get_all_compile_status() {
                    if fn_name.starts_with(&format!("{}.", module_name)) || fn_name == module_name {
                        if status_str.starts_with("Error:") {
                            let (line, message) = Self::parse_error_location(&status_str, content_ref);
                            errors.push(Diagnostic {
                                range: Range {
                                    start: Position { line, character: 0 },
                                    end: Position { line, character: 100 },
                                },
                                severity: Some(DiagnosticSeverity::ERROR),
                                message,
                                source: Some("nostos".to_string()),
                                ..Default::default()
                            });
                        } else if status_str.starts_with("Stale:") {
                            errors.push(Diagnostic {
                                range: Range {
                                    start: Position { line: 0, character: 0 },
                                    end: Position { line: 0, character: 0 },
                                },
                                severity: Some(DiagnosticSeverity::WARNING),
                                message: status_str,
                                source: Some("nostos".to_string()),
                                ..Default::default()
                            });
                        }
                    }
                }
                errors
            } else {
                vec![]
            }
        };

        self.client.publish_diagnostics(uri.clone(), diagnostics, None).await;
    }

    /// Check a file for errors without modifying the live compiler state.
    /// Used for real-time analysis while the user is typing.
    async fn check_file(&self, uri: &Url, content: &str) {
        let file_path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return,
        };

        let file_path_str = file_path.to_string_lossy().to_string();
        eprintln!("Checking file (analysis only): {}", file_path_str);

        // Wait for engine to be initialized (async wait doesn't block the tokio runtime)
        // Engine init can take several seconds as it compiles stdlib
        let mut attempts = 0;
        while attempts < 50 {
            {
                let guard = self.engine.lock().unwrap();
                if guard.is_some() {
                    break;
                }
            }
            attempts += 1;
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        // Extract module name from file path
        let module_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("main")
            .to_string();

        // Use check_module_compiles which does NOT modify the engine state
        // This is a read-only analysis for real-time feedback
        let result = {
            let engine_guard = self.engine.lock().unwrap();
            let Some(engine) = engine_guard.as_ref() else {
                eprintln!("check_file: engine not ready yet after waiting, skipping check");
                return;
            };

            engine.check_module_compiles(&module_name, content)
        };

        eprintln!("Check result for {}: {:?}", module_name, result);

        // Build diagnostics from the check result
        let error_diagnostics = if let Err(e) = &result {
            let (line, message) = Self::parse_error_location(e, Some(content));
            vec![Diagnostic {
                range: Range {
                    start: Position { line, character: 0 },
                    end: Position { line, character: 100 },
                },
                severity: Some(DiagnosticSeverity::ERROR),
                message,
                source: Some("nostos".to_string()),
                ..Default::default()
            }]
        } else {
            vec![]
        };

        // Store error diagnostics for this file
        self.file_errors.insert(uri.clone(), error_diagnostics.clone());

        self.client.publish_diagnostics(uri.clone(), error_diagnostics, None).await;
    }

    /// Recompile a file and publish updated diagnostics.
    /// This modifies the live compiler state - use for commits.
    /// Returns Ok(()) if compilation succeeded, Err(message) if there were errors.
    async fn recompile_file(&self, uri: &Url, content: &str) -> std::result::Result<(), String> {
        let file_path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return Err("Invalid file path".to_string()),
        };

        let file_path_str = file_path.to_string_lossy().to_string();
        eprintln!("Committing file to live system: {}", file_path_str);

        // Extract module name from file path
        let module_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("main")
            .to_string();

        // Use recompile_module_with_content which properly handles function additions/deletions/changes
        // using source hashing to detect what actually changed
        let result = {
            let mut engine_guard = self.engine.lock().unwrap();
            let Some(engine) = engine_guard.as_mut() else {
                return Err("Engine not initialized".to_string());
            };

            engine.recompile_module_with_content(&module_name, content)
        };

        eprintln!("Compile result for {}: {:?}", module_name, result);

        // Publish diagnostics based on result AND actual compile status
        // (recompile_module_with_content might return Ok("No changes detected") even if
        // the function has an error from a previous compilation)
        let error_diagnostics = {
            let mut errors = vec![];

            // First check the return value for direct errors
            if let Err(e) = &result {
                let (line, message) = Self::parse_error_location(e, Some(content));
                errors.push(Diagnostic {
                    range: Range {
                        start: Position { line, character: 0 },
                        end: Position { line, character: 100 },
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    message,
                    source: Some("nostos".to_string()),
                    ..Default::default()
                });
            }

            // Also check the actual compile status from the engine
            // This catches errors that persist even when "no changes detected"
            {
                let engine_guard = self.engine.lock().unwrap();
                if let Some(engine) = engine_guard.as_ref() {
                    for (fn_name, status) in engine.get_all_compile_status_detailed() {
                        if fn_name.starts_with(&format!("{}.", module_name)) || fn_name == module_name {
                            if let nostos_repl::CompileStatus::CompileError(e) = status {
                                let (line, message) = Self::parse_error_location(&e, Some(content));
                                // Avoid duplicate errors
                                if !errors.iter().any(|d| d.message == message) {
                                    errors.push(Diagnostic {
                                        range: Range {
                                            start: Position { line, character: 0 },
                                            end: Position { line, character: 100 },
                                        },
                                        severity: Some(DiagnosticSeverity::ERROR),
                                        message,
                                        source: Some("nostos".to_string()),
                                        ..Default::default()
                                    });
                                }
                            }
                        }
                    }
                }
            }

            errors
        };

        // Store error diagnostics for this file (so we can merge with stale warnings later)
        self.file_errors.insert(uri.clone(), error_diagnostics.clone());

        self.client.publish_diagnostics(uri.clone(), error_diagnostics.clone(), None).await;

        // When a file compiles successfully, recompile other open files
        // (they might depend on the changed file and now compile successfully)
        if error_diagnostics.is_empty() {
            self.recompile_other_open_files(uri).await;
            Ok(())
        } else {
            // Return the first error message for the commit command to display
            let first_error = error_diagnostics.first().map(|d| d.message.clone()).unwrap_or_else(|| "Unknown error".to_string());
            Err(first_error)
        }
    }

    /// Recompile all open files except the one that was just compiled
    async fn recompile_other_open_files(&self, exclude_uri: &Url) {
        let open_docs: Vec<(Url, String)> = self.documents.iter()
            .filter(|entry| entry.key() != exclude_uri)
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        for (uri, content) in open_docs {
            eprintln!("Recompiling dependent file: {}", uri);

            let file_path = match uri.to_file_path() {
                Ok(p) => p,
                Err(_) => continue,
            };

            let module_name = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("main")
                .to_string();

            let result = {
                let mut engine_guard = self.engine.lock().unwrap();
                let Some(engine) = engine_guard.as_mut() else {
                    continue;
                };

                engine.recompile_module_with_content(&module_name, &content)
            };

            let error_diagnostics = match &result {
                Ok(_) => vec![],
                Err(e) => {
                    let (line, message) = Self::parse_error_location(e, Some(&content));
                    vec![Diagnostic {
                        range: Range {
                            start: Position { line, character: 0 },
                            end: Position { line, character: 100 },
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        message,
                        source: Some("nostos".to_string()),
                        ..Default::default()
                    }]
                }
            };

            self.file_errors.insert(uri.clone(), error_diagnostics.clone());
            self.client.publish_diagnostics(uri, error_diagnostics, None).await;
        }
    }

    /// Publish diagnostics for all known files in the workspace (currently unused, kept for future use)
    #[allow(dead_code)]
    async fn publish_all_file_diagnostics(&self) {
        let file_paths: Vec<String> = {
            let engine_guard = self.engine.lock().unwrap();
            let Some(engine) = engine_guard.as_ref() else {
                return;
            };

            // Get all module sources (file paths)
            engine.get_module_source_paths()
        };

        for file_path in file_paths {
            let path = PathBuf::from(&file_path);
            let uri = match Url::from_file_path(&path) {
                Ok(u) => u,
                Err(_) => continue,
            };

            let is_open = self.documents.contains_key(&uri);

            // For open documents, only publish stale warnings (they get accurate errors from recompile_file)
            // For closed documents, publish everything
            self.publish_file_diagnostics_filtered(&uri, &file_path, is_open).await;
        }
    }

    /// Publish diagnostics for a file, optionally filtering out errors (for open documents)
    #[allow(dead_code)]
    async fn publish_file_diagnostics_filtered(&self, uri: &Url, file_path: &str, is_open: bool) {
        // Get document content for line detection - try open document first, then read from disk
        let doc_content = self.documents.get(uri)
            .map(|d| d.clone())
            .or_else(|| std::fs::read_to_string(file_path).ok());

        let stale_warnings = {
            let engine_guard = self.engine.lock().unwrap();
            let Some(engine) = engine_guard.as_ref() else {
                return;
            };

            let module_name = std::path::Path::new(file_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let mut warnings = Vec::new();

            for (fn_name, status) in engine.get_all_compile_status_detailed() {
                if fn_name.starts_with(&format!("{}.", module_name)) || fn_name == module_name {
                    if let nostos_repl::CompileStatus::Stale { reason, depends_on } = status {
                        // Try to find the line where the missing function is called
                        let line = if let Some(ref content) = doc_content {
                            Self::find_function_call_line(content, &depends_on)
                        } else {
                            0
                        };

                        warnings.push(Diagnostic {
                            range: Range {
                                start: Position { line, character: 0 },
                                end: Position { line, character: 100 },
                            },
                            severity: Some(DiagnosticSeverity::WARNING),
                            message: format!("Stale: {}", reason),
                            source: Some("nostos".to_string()),
                            ..Default::default()
                        });
                    }
                }
            }
            warnings
        };

        // Get errors from engine status (only for CLOSED files)
        // Open files get accurate errors directly from recompile_file - don't republish
        // stale errors from compile_status which may have outdated line numbers
        let mut diagnostics = if is_open {
            // Open files: skip errors from compile_status - they were already published
            // with correct line numbers directly from recompile_file()
            vec![]
        } else {
            // Closed files: get errors from compile_status
            let engine_guard = self.engine.lock().unwrap();
            if let Some(engine) = engine_guard.as_ref() {
                let module_name = std::path::Path::new(file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                let mut errors = Vec::new();
                for (fn_name, status_str) in engine.get_all_compile_status() {
                    if fn_name.starts_with(&format!("{}.", module_name)) || fn_name == module_name {
                        if status_str.starts_with("Error:") {
                            let (line, message) = Self::parse_error_location(&status_str, doc_content.as_deref());
                            errors.push(Diagnostic {
                                range: Range {
                                    start: Position { line, character: 0 },
                                    end: Position { line, character: 100 },
                                },
                                severity: Some(DiagnosticSeverity::ERROR),
                                message,
                                source: Some("nostos".to_string()),
                                ..Default::default()
                            });
                        }
                    }
                }
                errors
            } else {
                vec![]
            }
        };

        // Add stale warnings
        diagnostics.extend(stale_warnings);

        self.client.publish_diagnostics(uri.clone(), diagnostics, None).await;
    }

    /// Parse error message to extract line number
    /// Formats:
    /// - From check_module_compiles: "line N: message"
    /// - From engine status: "Error: file:line: message" or "Error: message"
    /// - "<repl>:N: message" or "file:N: message"
    fn parse_error_location(error: &str, content: Option<&str>) -> (u32, String) {
        eprintln!("Parsing error: {}", error);

        // Strip "Error: " prefix if present
        let error = error.strip_prefix("Error: ").unwrap_or(error);

        // Format 1: "line N: message"
        if let Some(rest) = error.strip_prefix("line ") {
            if let Some(colon_pos) = rest.find(':') {
                if let Ok(line) = rest[..colon_pos].trim().parse::<u32>() {
                    let message = rest[colon_pos + 1..].trim().to_string();
                    eprintln!("Parsed line {} message: {}", line, message);
                    return (line.saturating_sub(1), message); // LSP lines are 0-indexed
                }
            }
        }

        // Format 2: "<repl>:N: message" or "file:N: message" (e.g., "/path/to/file.nos:42: error message")
        if let Some(colon_pos) = error.find(':') {
            let after_first = &error[colon_pos + 1..];
            if let Some(second_colon) = after_first.find(':') {
                if let Ok(line) = after_first[..second_colon].trim().parse::<u32>() {
                    let message = after_first[second_colon + 1..].trim().to_string();
                    eprintln!("Parsed (format 2) line {} message: {}", line, message);
                    return (line.saturating_sub(1), message);
                }
            }
        }

        // At this point, error_msg is the message without "Error: " prefix
        let error_msg = error;

        // Format 4: "Undefined function: X" - search for X in content
        if let Some(fn_name) = error_msg.strip_prefix("Undefined function: ") {
            if let Some(content) = content {
                let fn_name = fn_name.trim();
                let line = Self::find_function_call_line(content, &[fn_name.to_string()]);
                eprintln!("Found undefined function {} at line {}", fn_name, line);
                return (line, error_msg.to_string());
            }
        }

        // Format 5: Extract function name from error like "cannot resolve trait method `X`"
        if let Some(content) = content {
            // Try to extract backtick-quoted identifiers
            let re_pattern: Vec<&str> = error_msg.match_indices('`').collect::<Vec<_>>()
                .chunks(2)
                .filter_map(|pair| {
                    if pair.len() == 2 {
                        let start = pair[0].0 + 1;
                        let end = pair[1].0;
                        Some(&error_msg[start..end])
                    } else {
                        None
                    }
                })
                .collect();

            if !re_pattern.is_empty() {
                let search_terms: Vec<String> = re_pattern.iter().map(|s| s.to_string()).collect();
                let line = Self::find_function_call_line(content, &search_terms);
                eprintln!("Found function reference {:?} at line {}", re_pattern, line);
                return (line, error_msg.to_string());
            }
        }

        // Format 6: "Wrong number of arguments" - search for empty function calls
        // like .map() or .filter() that are likely missing arguments
        // NOTE: This is a fallback heuristic. The type checker should ideally provide
        // correct spans, but currently has a known limitation with span tracking.
        if error_msg.contains("Wrong number of arguments") {
            if let Some(content) = content {
                // Search for patterns like .X() with empty parentheses
                let line = Self::find_empty_call_line(content);
                if line > 0 {
                    eprintln!("Found likely empty call at line {}", line);
                    return (line - 1, error_msg.to_string()); // Convert to 0-based
                }
            }
        }

        // Fallback: no line number found
        eprintln!("Could not parse line number, using line 0");
        (0, error_msg.to_string())
    }

    /// Find the line number where a function from depends_on is called
    fn find_function_call_line(content: &str, depends_on: &[String]) -> u32 {
        for (line_num, line) in content.lines().enumerate() {
            // Skip comment lines
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.is_empty() {
                continue;
            }

            for dep in depends_on {
                // Extract the function name from qualified name (e.g., "good.addx" -> "addx")
                let fn_name = dep.rsplit('.').next().unwrap_or(dep);
                // Look for actual function call (with parenthesis) to avoid matching variable names
                let call_pattern = format!("{}(", fn_name);
                let qualified_call = format!("{}(", dep);
                if line.contains(&call_pattern) || line.contains(&qualified_call) || line.contains(dep) {
                    eprintln!("Found function call {} on line {}: {}", fn_name, line_num, line);
                    return line_num as u32;
                }
            }
        }
        0 // Default to line 0 if not found
    }

    /// Find the line number (1-based) where an empty method call is likely causing an arity error
    /// Looks for patterns like .map() .filter() etc with empty parentheses
    fn find_empty_call_line(content: &str) -> u32 {
        // Common higher-order functions that take at least one argument
        let hot_functions = ["map", "filter", "fold", "reduce", "flatMap", "forEach", "any", "all"];

        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            // Skip comment lines
            if trimmed.starts_with('#') || trimmed.is_empty() {
                continue;
            }

            // Look for .function() with empty parentheses
            for func in &hot_functions {
                let pattern = format!(".{}()", func);
                if line.contains(&pattern) {
                    eprintln!("Found empty call {} at line {}: {}", func, line_num + 1, line);
                    return (line_num + 1) as u32; // Return 1-based line number
                }
            }
        }
        0 // Not found
    }
}

// Increment BUILD_ID manually when making changes to easily verify binary is updated
const LSP_VERSION: &str = env!("CARGO_PKG_VERSION");
const LSP_BUILD_ID: &str = "2026-01-13-show-inferred-type";

#[tower_lsp::async_trait]
impl LanguageServer for NostosLanguageServer {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        log(&format!("Nostos LSP v{} (build: {})", LSP_VERSION, LSP_BUILD_ID));
        log("initialize() called - fast path");

        // Store workspace root for lazy initialization - do NOT block here
        // Heavy init will happen on first file open or in initialized() notification
        if let Some(root_uri) = params.root_uri {
            if let Ok(path) = root_uri.to_file_path() {
                eprintln!("Workspace root: {:?}", path);
                *self.root_path.lock().unwrap() = Some(path);
            }
        } else if let Some(folders) = params.workspace_folders {
            if let Some(folder) = folders.first() {
                if let Ok(path) = folder.uri.to_file_path() {
                    eprintln!("Workspace folder: {:?}", path);
                    *self.root_path.lock().unwrap() = Some(path);
                }
            }
        }

        log("initialize() returning response (engine will init lazily)");

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                // Sync full document on change
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                // Autocomplete support
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                // Hover support for type information
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                // Go to definition support
                definition_provider: Some(OneOf::Left(true)),
                // Signature help for function calls
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                // Document symbols (outline view)
                document_symbol_provider: Some(OneOf::Left(true)),
                // Find references
                references_provider: Some(OneOf::Left(true)),
                // Inlay hints disabled for now - needs more work
                // inlay_hint_provider: Some(OneOf::Right(InlayHintServerCapabilities::Options(
                //     InlayHintOptions {
                //         resolve_provider: Some(false),
                //         work_done_progress_options: Default::default(),
                //     }
                // ))),
                // Don't advertise commands here - the extension registers them
                // and forwards via workspace/executeCommand. Advertising them
                // causes vscode-languageclient to also try registering them,
                // leading to "command already exists" errors.
                execute_command_provider: None,
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "nostos-lsp".to_string(),
                version: Some("0.1.0".to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        log("initialized() notification received");

        // Get path before any await - must not hold MutexGuard across await
        let path_opt = self.root_path.lock().unwrap().clone();

        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "initialized(): path_opt={:?}", path_opt);
        }

        // Do the heavy engine initialization in a blocking thread pool
        // This prevents blocking the async message loop
        if let Some(path) = path_opt {
            // Set initializing flag to prevent double init from did_open
            if self.initializing.swap(true, Ordering::SeqCst) {
                eprintln!("Engine initialization already in progress, skipping");
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                    use std::io::Write;
                    let _ = writeln!(f, "initialized(): Skipping - already initializing");
                }
                return;
            }

            eprintln!("Starting engine initialization for {:?}...", path);
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                use std::io::Write;
                let _ = writeln!(f, "initialized(): Starting spawn_blocking for {:?}", path);
            }
            let start = std::time::Instant::now();

            // Use spawn_blocking to run in a separate thread pool, returning the engine
            let init_result = tokio::task::spawn_blocking(move || {
                let config = ReplConfig {
                    enable_jit: false,
                    num_threads: 1,
                };
                let mut engine = ReplEngine::new(config);

                // Load stdlib
                if let Err(e) = engine.load_stdlib() {
                    eprintln!("Warning: Failed to load stdlib: {}", e);
                }

                // Load the project directory
                if let Err(e) = engine.load_directory(path.to_str().unwrap_or(".")) {
                    eprintln!("Warning: Failed to load directory: {}", e);
                }

                engine
            }).await;

            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                use std::io::Write;
                let _ = writeln!(f, "initialized(): spawn_blocking returned, init_result.is_ok={}", init_result.is_ok());
            }

            match init_result {
                Ok(engine) => {
                    // Collect errors before storing engine
                    let error_defs = engine.get_error_definitions();
                    let root = self.root_path.lock().unwrap().clone();

                    *self.engine.lock().unwrap() = Some(engine);
                    eprintln!("Engine initialized in {:?}", start.elapsed());
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                        use std::io::Write;
                        let _ = writeln!(f, "initialized(): Engine set to Some, elapsed={:?}", start.elapsed());
                    }

                    // Publish diagnostics for files with errors at startup
                    if !error_defs.is_empty() {
                        eprintln!("Publishing {} startup errors...", error_defs.len());
                        for (fn_name, error_msg) in error_defs {
                            // Extract module name from function name (e.g., "main.main" -> "main")
                            let module_name = if let Some(dot_pos) = fn_name.find('.') {
                                &fn_name[..dot_pos]
                            } else {
                                &fn_name[..]
                            };

                            // Build file path from module name
                            if let Some(ref root_path) = root {
                                let file_path = root_path.join(format!("{}.nos", module_name));
                                if file_path.exists() {
                                    let uri = Url::from_file_path(&file_path).ok();
                                    if let Some(uri) = uri {
                                        // Read file content for better error location
                                        let content = std::fs::read_to_string(&file_path).ok();
                                        let (line, message) = Self::parse_error_location(&error_msg, content.as_deref());

                                        let diagnostic = Diagnostic {
                                            range: Range {
                                                start: Position { line, character: 0 },
                                                end: Position { line, character: 100 },
                                            },
                                            severity: Some(DiagnosticSeverity::ERROR),
                                            message,
                                            source: Some("nostos".to_string()),
                                            ..Default::default()
                                        };

                                        // Store in file_errors for persistence
                                        self.file_errors.insert(uri.clone(), vec![diagnostic.clone()]);

                                        // Publish to client
                                        self.client.publish_diagnostics(uri, vec![diagnostic], None).await;
                                        eprintln!("Published startup error for {}", file_path.display());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Engine initialization failed: {}", e);
                }
            }

            self.initializing.store(false, Ordering::SeqCst);

            // Send initial file status notification
            self.send_file_status_notification().await;
        }

        eprintln!("Nostos LSP fully initialized!");
        eprintln!("Server is now ready and waiting for requests...");
    }

    async fn shutdown(&self) -> Result<()> {
        log("!!! SHUTDOWN REQUEST RECEIVED !!!");

        // Persist module cache on shutdown
        if let Some(ref mut engine) = *self.engine.lock().unwrap() {
            match engine.persist_module_cache() {
                Ok(0) => {} // No modules to persist
                Ok(n) => eprintln!("Persisted {} module(s) to cache", n),
                Err(e) => eprintln!("Warning: Failed to persist module cache: {}", e),
            }
        }

        Ok(())
    }

    async fn execute_command(&self, params: ExecuteCommandParams) -> Result<Option<serde_json::Value>> {
        eprintln!("Execute command: {}", params.command);

        match params.command.as_str() {
            "nostos.buildCache" => {
                let result = if let Some(ref mut engine) = *self.engine.lock().unwrap() {
                    match engine.persist_module_cache() {
                        Ok(n) => format!("Built cache: {} module(s) persisted", n),
                        Err(e) => format!("Failed to build cache: {}", e),
                    }
                } else {
                    "No project loaded".to_string()
                };

                self.client.show_message(MessageType::INFO, &result).await;
                Ok(Some(serde_json::json!({ "message": result })))
            }
            "nostos.clearCache" => {
                let result = if let Some(ref mut engine) = *self.engine.lock().unwrap() {
                    match engine.clear_all_caches() {
                        Ok(()) => "Cache cleared successfully".to_string(),
                        Err(e) => format!("Failed to clear cache: {}", e),
                    }
                } else {
                    "No project loaded".to_string()
                };

                self.client.show_message(MessageType::INFO, &result).await;
                Ok(Some(serde_json::json!({ "message": result })))
            }
            "nostos.commit" => {
                // Commit a specific file to the live system
                // Expects argument: file URI
                let uri_str = params.arguments.first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if uri_str.is_empty() {
                    self.client.show_message(MessageType::WARNING, "No file specified for commit").await;
                    return Ok(Some(serde_json::json!({ "error": "No file specified" })));
                }

                let uri = match Url::parse(uri_str) {
                    Ok(u) => u,
                    Err(e) => {
                        let msg = format!("Invalid URI: {}", e);
                        self.client.show_message(MessageType::ERROR, &msg).await;
                        return Ok(Some(serde_json::json!({ "error": msg })));
                    }
                };

                // Get content from our document cache
                if let Some(content) = self.documents.get(&uri) {
                    match self.recompile_file(&uri, &content.clone()).await {
                        Ok(()) => {
                            // Clear dirty flag on successful commit
                            if let Ok(file_path) = uri.to_file_path() {
                                self.dirty_files.remove(&file_path.to_string_lossy().to_string());
                            }
                            // Send updated file status notification
                            self.send_file_status_notification().await;

                            let msg = format!("Committed to live system: {}", uri.path());
                            self.client.show_message(MessageType::INFO, &msg).await;
                            eprintln!("{}", msg);
                            Ok(Some(serde_json::json!({ "success": true, "message": msg })))
                        }
                        Err(error) => {
                            let msg = format!("Commit failed: {}", error);
                            self.client.show_message(MessageType::ERROR, &msg).await;
                            eprintln!("{}", msg);
                            Ok(Some(serde_json::json!({ "success": false, "error": error })))
                        }
                    }
                } else {
                    let msg = "File not open in editor";
                    self.client.show_message(MessageType::WARNING, msg).await;
                    Ok(Some(serde_json::json!({ "success": false, "error": msg })))
                }
            }
            "nostos.commitAll" => {
                // Commit all open files to the live system
                let open_docs: Vec<(Url, String)> = self.documents.iter()
                    .map(|entry| (entry.key().clone(), entry.value().clone()))
                    .collect();

                let mut success_count = 0;
                let mut error_count = 0;
                let mut errors: Vec<String> = vec![];

                for (uri, content) in &open_docs {
                    match self.recompile_file(uri, content).await {
                        Ok(()) => {
                            success_count += 1;
                            // Clear dirty flag on successful commit
                            if let Ok(file_path) = uri.to_file_path() {
                                self.dirty_files.remove(&file_path.to_string_lossy().to_string());
                            }
                        }
                        Err(e) => {
                            error_count += 1;
                            errors.push(format!("{}: {}", uri.path(), e));
                        }
                    }
                }

                // Send updated file status notification
                self.send_file_status_notification().await;

                if error_count == 0 {
                    let msg = format!("Committed {} file(s) to live system", success_count);
                    self.client.show_message(MessageType::INFO, &msg).await;
                    eprintln!("{}", msg);
                    Ok(Some(serde_json::json!({ "success": true, "message": msg, "count": success_count })))
                } else {
                    let msg = format!("Commit failed: {} error(s), {} succeeded", error_count, success_count);
                    self.client.show_message(MessageType::ERROR, &msg).await;
                    eprintln!("{}", msg);
                    for err in &errors {
                        eprintln!("  {}", err);
                    }
                    Ok(Some(serde_json::json!({ "success": false, "error": msg, "errors": errors })))
                }
            }
            "nostos.eval" => {
                // Evaluate an expression in the REPL
                // Expects argument: expression string
                let expr = params.arguments.first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if expr.is_empty() {
                    return Ok(Some(serde_json::json!({
                        "success": false,
                        "error": "No expression provided"
                    })));
                }

                eprintln!("REPL eval: {}", expr);

                // We need to run eval in a separate thread because the VM creates
                // its own tokio runtime, and we can't nest runtimes.
                let (tx, rx) = std::sync::mpsc::channel();
                let expr_owned = expr.to_string();

                // Take engine out of mutex, run eval in separate thread, put it back
                let engine_opt = {
                    let mut engine_guard = self.engine.lock().unwrap();
                    engine_guard.take()
                };

                let Some(mut engine) = engine_opt else {
                    return Ok(Some(serde_json::json!({
                        "success": false,
                        "error": "Engine not initialized"
                    })));
                };

                std::thread::spawn(move || {
                    // Debug: log known modules
                    let known_mods = engine.get_known_modules();
                    eprintln!("REPL eval thread - known modules: {:?}", known_mods);
                    // Use eval_with_capture to capture println output
                    let result = engine.eval_with_capture(&expr_owned);
                    let _ = tx.send((engine, result));
                });

                // Wait for result (with timeout)
                match rx.recv_timeout(std::time::Duration::from_secs(30)) {
                    Ok((engine, result)) => {
                        // Put engine back
                        {
                            let mut engine_guard = self.engine.lock().unwrap();
                            *engine_guard = Some(engine);
                        }

                        match result {
                            Ok((output, captured)) => {
                                eprintln!("REPL result: {}, captured: {}", output, captured);
                                // Combine captured output with result
                                let full_output = if captured.is_empty() {
                                    output
                                } else if output.is_empty() || output == "()" {
                                    captured.trim_end().to_string()
                                } else {
                                    format!("{}\n{}", captured.trim_end(), output)
                                };
                                Ok(Some(serde_json::json!({
                                    "success": true,
                                    "result": full_output
                                })))
                            }
                            Err(error) => {
                                eprintln!("REPL error: {}", error);
                                Ok(Some(serde_json::json!({
                                    "success": false,
                                    "error": error
                                })))
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("REPL eval timeout or thread panic");
                        Ok(Some(serde_json::json!({
                            "success": false,
                            "error": "Evaluation timed out or failed"
                        })))
                    }
                }
            }
            "nostos.replComplete" => {
                // Get completions for REPL input
                // Expects arguments: [text, cursorPosition]
                let text = params.arguments.first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let cursor_pos = params.arguments.get(1)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(text.len() as u64) as usize;

                eprintln!("REPL complete: text='{}', cursor={}", text, cursor_pos);

                let completions = {
                    let engine_guard = self.engine.lock().unwrap();
                    let Some(engine) = engine_guard.as_ref() else {
                        return Ok(Some(serde_json::json!({
                            "completions": []
                        })));
                    };

                    self.get_repl_completions(engine, text, cursor_pos)
                };

                Ok(Some(serde_json::json!({
                    "completions": completions
                })))
            }
            _ => {
                eprintln!("Unknown command: {}", params.command);
                Ok(None)
            }
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        eprintln!("Opened: {}", params.text_document.uri);

        let uri = params.text_document.uri;
        let content = params.text_document.text;

        // If engine is not initialized and no init is in progress, try lazy init
        // This handles the case where a single file is opened without opening a folder
        {
            let engine_guard = self.engine.lock().unwrap();
            let needs_init = engine_guard.is_none();
            drop(engine_guard);

            // Check if initialization is already in progress (from initialized() handler)
            let init_in_progress = self.initializing.load(Ordering::SeqCst);

            if needs_init && !init_in_progress {
                if let Ok(file_path) = uri.to_file_path() {
                    if let Some(parent) = file_path.parent() {
                        // Try to set the initializing flag
                        if !self.initializing.swap(true, Ordering::SeqCst) {
                            eprintln!("Lazy init: using file's parent as root: {:?}", parent);
                            self.init_engine(&parent.to_path_buf());
                            self.initializing.store(false, Ordering::SeqCst);
                        } else {
                            eprintln!("Skipping lazy init - initialization already in progress");
                        }
                    }
                }
            } else if needs_init && init_in_progress {
                eprintln!("Waiting for background initialization to complete...");
            }
        }

        // Store document content
        self.documents.insert(uri.clone(), content.clone());

        // Check for errors (analysis only)
        // The file was already loaded via load_directory, so we just need to show current state
        self.check_file(&uri, &content).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;

        // Get the full content (we're using FULL sync)
        if let Some(change) = params.content_changes.into_iter().last() {
            let content = change.text;

            // Update stored content
            self.documents.insert(uri.clone(), content.clone());

            // Mark file as dirty (modified but not compiled)
            if let Ok(file_path) = uri.to_file_path() {
                self.dirty_files.insert(file_path.to_string_lossy().to_string(), true);
                // Send file status notification
                self.send_file_status_notification().await;
            }

            // Check for errors (analysis only, no state change)
            // Real-time feedback while typing
            self.check_file(&uri, &content).await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        eprintln!("Saved: {} (use Ctrl+Alt+C to commit to live)", params.text_document.uri);

        // Save does NOT commit to live system
        // User must explicitly use nostos.commit command (Ctrl+Alt+C)
        // This allows saving work-in-progress without affecting the running system
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        eprintln!("Closed: {}", params.text_document.uri);

        // Remove from our document cache
        self.documents.remove(&params.text_document.uri);
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        eprintln!("Completion request at {:?}", position);

        // Wait for engine to be initialized (async wait doesn't block the tokio runtime)
        // Engine init can take several seconds as it compiles stdlib
        let mut attempts = 0;
        while attempts < 50 {
            {
                let guard = self.engine.lock().unwrap();
                if guard.is_some() {
                    break;
                }
            }
            attempts += 1;
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        // Get the line up to cursor
        let lines: Vec<&str> = content.lines().collect();
        let line_num = position.line as usize;
        if line_num >= lines.len() {
            return Ok(None);
        }

        let line = lines[line_num];
        let cursor_char = position.character as usize;
        let prefix = if cursor_char <= line.len() {
            &line[..cursor_char]
        } else {
            line
        };

        eprintln!("Line prefix: '{}'", prefix);

        // Extract local variable bindings from the document (lines before cursor)
        let engine_guard = self.engine.lock().unwrap();
        let engine_ref = engine_guard.as_ref();
        let local_vars = Self::extract_local_bindings(&content, line_num, engine_ref);
        drop(engine_guard);
        eprintln!("Local vars: {:?}", local_vars);

        // Determine completion context
        let items = if let Some(dot_pos) = prefix.rfind('.') {
            // After a dot - could be module or UFCS call
            let before_dot = prefix[..dot_pos].trim();
            eprintln!("After dot, before: '{}'", before_dot);

            // Check if we're inside a lambda - look for lambda parameter context
            // Pattern: "receiver.method(param =>" where we're completing "param."
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                    let _ = writeln!(f, "DEBUG: Checking lambda context. prefix='{}', before_dot='{}'", prefix, before_dot);
                    let _ = writeln!(f, "DEBUG: local_vars={:?}", local_vars);
                }
            }
            let lambda_type = Self::infer_lambda_param_type(prefix, before_dot, &local_vars);
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                    let _ = writeln!(f, "DEBUG: Lambda type result: {:?}", lambda_type);
                }
            }
            if let Some(ref lt) = lambda_type {
                eprintln!("Detected lambda parameter with type: {}", lt);
            }

            self.get_dot_completions(before_dot, &local_vars, lambda_type.as_deref(), &content)
        } else {
            // General identifier completion
            let partial = prefix.split(|c: char| !c.is_alphanumeric() && c != '_')
                .last()
                .unwrap_or("");
            eprintln!("Identifier completion, partial: '{}'", partial);
            self.get_identifier_completions(partial)
        };

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        eprintln!("Hover request at {:?}", position);

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        // Get the word at cursor position
        let lines: Vec<&str> = content.lines().collect();
        let line_num = position.line as usize;
        if line_num >= lines.len() {
            return Ok(None);
        }

        let line = lines[line_num];
        let cursor = position.character as usize;

        // Extract the word/expression at cursor
        let (word, word_start, word_end) = Self::extract_word_at_cursor(line, cursor);
        if word.is_empty() {
            return Ok(None);
        }

        eprintln!("Hover word: '{}' at {}..{}", word, word_start, word_end);

        // Compute byte offset for HM type lookup
        let byte_offset = Self::line_col_to_byte_offset(&content, position.line as usize, word_start);

        // Get local bindings for type inference
        let engine_guard = self.engine.lock().unwrap();
        let engine_ref = engine_guard.as_ref();
        let local_vars = Self::extract_local_bindings(&content, line_num + 1, engine_ref);

        // Try HM-inferred type first (uses position-based lookup)
        let hm_type = if let Some(engine) = engine_ref {
            // Use file_id 0 since that's what the parser uses by default
            engine.get_inferred_type_at_position(0, byte_offset)
        } else {
            None
        };

        // Try to get type/signature information
        let hover_info = if let Some(ty) = hm_type {
            // Got an HM-inferred type - use it
            Some(format!("```nostos\n{}: {}\n```\n*(inferred)*", word, ty))
        } else if let Some(engine) = engine_ref {
            self.get_hover_info(engine, &word, &local_vars, line)
        } else {
            None
        };

        drop(engine_guard);

        if let Some(info) = hover_info {
            let range = Range {
                start: Position { line: position.line, character: word_start as u32 },
                end: Position { line: position.line, character: word_end as u32 },
            };

            Ok(Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: info,
                }),
                range: Some(range),
            }))
        } else {
            Ok(None)
        }
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = &params.text_document.uri;
        let range = params.range;

        eprintln!("Inlay hint request for range {:?}", range);

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return Ok(None);
        };

        // Find all simple bindings in the visible range (lines with "name = value")
        // and get their types from HM inference
        let mut hints = Vec::new();
        let mut seen_lines = std::collections::HashSet::new();

        let start_line = range.start.line as usize;
        let end_line = range.end.line as usize;

        for (line_idx, line) in content.lines().enumerate() {
            if line_idx < start_line || line_idx > end_line {
                continue;
            }

            // Skip if already processed this line
            if seen_lines.contains(&line_idx) {
                continue;
            }

            // Look for simple binding pattern: "name = value" (not function def)
            if let Some(binding) = Self::extract_binding_from_line(line) {
                // Calculate byte offset for the binding's RHS
                let line_start = Self::line_col_to_byte_offset(&content, line_idx, 0);
                let rhs_start = line_start + binding.rhs_start;

                // Get type at the RHS position
                if let Some(ty) = engine.get_inferred_type_at_position(0, rhs_start) {
                    // Skip unresolved types (containing ?)
                    if !ty.contains('?') && Self::should_show_type_hint(&ty) {
                        let position = Position {
                            line: line_idx as u32,
                            character: binding.name_end as u32,
                        };
                        hints.push(InlayHint {
                            position,
                            label: InlayHintLabel::String(format!(": {}", ty)),
                            kind: Some(InlayHintKind::TYPE),
                            text_edits: None,
                            tooltip: None,
                            padding_left: Some(false),
                            padding_right: Some(true),
                            data: None,
                        });
                        seen_lines.insert(line_idx);
                        eprintln!("Inlay hint: {} : {} at line {}", binding.name, ty, line_idx);
                    }
                }
            }
        }

        drop(engine_guard);

        if hints.is_empty() {
            Ok(None)
        } else {
            Ok(Some(hints))
        }
    }

    async fn goto_definition(&self, params: GotoDefinitionParams) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        eprintln!("Goto definition at {:?}", position);

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        // Get the word at cursor position
        let lines: Vec<&str> = content.lines().collect();
        let line_num = position.line as usize;
        if line_num >= lines.len() {
            return Ok(None);
        }

        let line = lines[line_num];
        let cursor = position.character as usize;

        let (word, _, _) = Self::extract_word_at_cursor(line, cursor);
        if word.is_empty() {
            return Ok(None);
        }

        eprintln!("Goto definition for: '{}'", word);

        // Try to find the definition
        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return Ok(None);
        };

        // Check if it's a function
        if let Some(source_file) = engine.get_function_source_file(&word) {
            // Try to find the line in the source file
            if let Some(file_content) = std::fs::read_to_string(&source_file).ok() {
                if let Some(line_num) = Self::find_definition_line(&file_content, &word) {
                    if let Ok(target_uri) = Url::from_file_path(&source_file) {
                        return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                            uri: target_uri,
                            range: Range {
                                start: Position { line: line_num, character: 0 },
                                end: Position { line: line_num, character: 0 },
                            },
                        })));
                    }
                }
            }
        }

        // Check if it's a type
        if let Some(type_def_info) = engine.get_type_definition_location(&word) {
            if let Ok(target_uri) = Url::from_file_path(&type_def_info.0) {
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: target_uri,
                    range: Range {
                        start: Position { line: type_def_info.1, character: 0 },
                        end: Position { line: type_def_info.1, character: 0 },
                    },
                })));
            }
        }

        Ok(None)
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        let lines: Vec<&str> = content.lines().collect();
        let line_num = position.line as usize;
        if line_num >= lines.len() {
            return Ok(None);
        }

        let line = lines[line_num];
        let cursor = position.character as usize;
        let prefix = if cursor <= line.len() { &line[..cursor] } else { line };

        // Find the function name before the opening paren
        let (fn_name, active_param) = Self::extract_function_call_context(prefix);
        if fn_name.is_empty() {
            return Ok(None);
        }

        eprintln!("Signature help for: '{}', param {}", fn_name, active_param);

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return Ok(None);
        };

        // Get function signature
        if let Some(sig) = engine.get_function_signature(&fn_name) {
            // Get parameter info
            let params_info = engine.get_function_params(&fn_name);
            let doc = engine.get_function_doc(&fn_name);

            let parameters: Vec<ParameterInformation> = if let Some(params) = params_info {
                params.iter().map(|(name, ty, has_default, default_val)| {
                    let label = if *has_default {
                        if let Some(def) = default_val {
                            format!("{}: {} = {}", name, ty, def)
                        } else {
                            format!("{}: {} = ?", name, ty)
                        }
                    } else {
                        format!("{}: {}", name, ty)
                    };
                    ParameterInformation {
                        label: ParameterLabel::Simple(label),
                        documentation: None,
                    }
                }).collect()
            } else {
                vec![]
            };

            let signature = SignatureInformation {
                label: format!("{}: {}", fn_name, sig),
                documentation: doc.map(|d| Documentation::String(d)),
                parameters: Some(parameters),
                active_parameter: Some(active_param as u32),
            };

            return Ok(Some(SignatureHelp {
                signatures: vec![signature],
                active_signature: Some(0),
                active_parameter: Some(active_param as u32),
            }));
        }

        // Try builtin
        if let Some(sig) = nostos_compiler::Compiler::get_builtin_signature(&fn_name) {
            let doc = nostos_compiler::Compiler::get_builtin_doc(&fn_name);

            let signature = SignatureInformation {
                label: format!("{}: {}", fn_name, sig),
                documentation: doc.map(|d| Documentation::String(d.to_string())),
                parameters: None,
                active_parameter: Some(active_param as u32),
            };

            return Ok(Some(SignatureHelp {
                signatures: vec![signature],
                active_signature: Some(0),
                active_parameter: Some(active_param as u32),
            }));
        }

        Ok(None)
    }

    async fn document_symbol(&self, params: DocumentSymbolParams) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        eprintln!("Document symbols for: {:?}", uri);

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        let symbols = Self::extract_document_symbols(&content);

        if symbols.is_empty() {
            Ok(None)
        } else {
            Ok(Some(DocumentSymbolResponse::Flat(symbols)))
        }
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        eprintln!("Find references at {:?}", position);

        // Get document content
        let content = match self.documents.get(uri) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };

        // Get the word at cursor position
        let lines: Vec<&str> = content.lines().collect();
        let line_num = position.line as usize;
        if line_num >= lines.len() {
            return Ok(None);
        }

        let line = lines[line_num];
        let cursor = position.character as usize;

        let (word, _, _) = Self::extract_word_at_cursor(line, cursor);
        if word.is_empty() {
            return Ok(None);
        }

        eprintln!("Find references for: '{}'", word);

        // Get all documents and search for references
        let mut locations = Vec::new();

        // Search in all open documents
        for entry in self.documents.iter() {
            let doc_uri = entry.key();
            let doc_content = entry.value();

            Self::find_references_in_content(&word, doc_uri, doc_content, &mut locations);
        }

        // Also search in project files from engine
        let engine_guard = self.engine.lock().unwrap();
        if let Some(engine) = engine_guard.as_ref() {
            // Get all loaded module source paths
            for file_path in engine.get_module_source_paths() {
                // Skip if already searched (open document)
                if let Ok(file_uri) = Url::from_file_path(&file_path) {
                    if self.documents.contains_key(&file_uri) {
                        continue;
                    }

                    // Read file content
                    if let Ok(file_content) = std::fs::read_to_string(&file_path) {
                        Self::find_references_in_content(&word, &file_uri, &file_content, &mut locations);
                    }
                }
            }
        }

        if locations.is_empty() {
            Ok(None)
        } else {
            Ok(Some(locations))
        }
    }
}

impl NostosLanguageServer {
    /// Extract local variable bindings from document content up to a certain line
    /// Returns a map of variable name -> inferred type (e.g., "y" -> "Int")
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn extract_local_bindings(content: &str, up_to_line: usize, engine: Option<&nostos_repl::ReplEngine>) -> std::collections::HashMap<String, String> {
        let mut bindings = std::collections::HashMap::new();

        // Track trait implementation context for `self` type inference
        // Pattern: "TypeName: TraitName" starts impl block, "end" closes it
        let mut current_impl_type: Option<String> = None;

        for (line_num, line) in content.lines().enumerate() {
            // Process up to AND including the current line for self detection
            let is_current_line = line_num == up_to_line;
            if line_num > up_to_line {
                break;
            }

            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Check for trait implementation header: "TypeName: TraitName" or "TypeName[T]: TraitName"
            // Pattern: starts with uppercase, has colon, ends with trait name (no =)
            if !trimmed.contains('=') {
                // Check for "end" keyword that closes trait impl block
                if trimmed == "end" {
                    current_impl_type = None;
                    continue;
                }

                // Check for trait impl pattern: "TypeName: TraitName" or "TypeName[T]: TraitName"
                if let Some(colon_pos) = trimmed.find(':') {
                    let before_colon = trimmed[..colon_pos].trim();
                    let after_colon = trimmed[colon_pos + 1..].trim();

                    // Check if before_colon starts with uppercase (type name)
                    // and after_colon is a valid trait name (starts with uppercase, no special chars except brackets)
                    let type_name = before_colon.split('[').next().unwrap_or(before_colon).trim();
                    if !type_name.is_empty()
                        && type_name.chars().next().map_or(false, |c| c.is_uppercase())
                        && !after_colon.is_empty()
                        && after_colon.chars().next().map_or(false, |c| c.is_uppercase())
                        && after_colon.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '[' || c == ']' || c == ',')
                    {
                        current_impl_type = Some(before_colon.to_string());
                        continue;
                    }
                }
            }

            // If we're in a trait impl, check for method definitions with `self` parameter
            if let Some(ref impl_type) = current_impl_type {
                // Method pattern: "methodName(self, ...) = ..." or "methodName(self) = ..."
                if let Some(paren_pos) = trimmed.find('(') {
                    let params_start = paren_pos + 1;
                    if let Some(params_end) = trimmed[params_start..].find(')') {
                        let params = &trimmed[params_start..params_start + params_end];
                        // Check if first parameter is `self`
                        let first_param = params.split(',').next().unwrap_or("").trim();
                        if first_param == "self" {
                            // Add self binding with the implementing type
                            bindings.insert("self".to_string(), impl_type.clone());
                        }
                    }
                }
            }

            // Skip regular binding extraction for the current line
            // (we only want to detect self parameter on current line)
            if is_current_line {
                continue;
            }

            // Check for mvar declarations: "mvar name: Type = expr"
            if trimmed.starts_with("mvar ") {
                let rest = trimmed[5..].trim(); // Skip "mvar "
                if let Some(colon_pos) = rest.find(':') {
                    let var_name = rest[..colon_pos].trim();
                    let after_colon = rest[colon_pos + 1..].trim();
                    // Find the = sign to separate type from initial value
                    if let Some(eq_pos) = after_colon.find('=') {
                        let type_name = after_colon[..eq_pos].trim();
                        if !var_name.is_empty() && !type_name.is_empty() {
                            eprintln!("Extracted mvar binding: {} : {}", var_name, type_name);
                            bindings.insert(var_name.to_string(), type_name.to_string());
                        }
                    }
                }
                continue;
            }

            // Look for simple bindings: "x = expr" or "x:Type = expr"
            if let Some(eq_pos) = trimmed.find('=') {
                // Make sure it's not == or other operators
                let before_eq = trimmed[..eq_pos].trim();
                let after_eq_start = eq_pos + 1;
                if after_eq_start < trimmed.len() && !trimmed[after_eq_start..].starts_with('=') {
                    let after_eq = trimmed[after_eq_start..].trim();

                    // Check for type annotation: "x:Type" or "x : Type"
                    let (var_name, explicit_type) = if let Some(colon_pos) = before_eq.find(':') {
                        let name = before_eq[..colon_pos].trim();
                        let ty = before_eq[colon_pos + 1..].trim();
                        (name, Some(ty.to_string()))
                    } else {
                        (before_eq, None)
                    };

                    // Check if var_name is a simple identifier (variable name)
                    if !var_name.is_empty()
                        && var_name.chars().next().map_or(false, |c| c.is_lowercase())
                        && var_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        // Use explicit type annotation if available, otherwise infer from RHS
                        let final_type = if let Some(ty) = explicit_type {
                            eprintln!("Extracted binding with explicit type: {} : {}", var_name, ty);
                            Some(ty)
                        } else {
                            // Pass current bindings so we can resolve index expressions like g2[0][0]
                            Self::infer_rhs_type(after_eq, engine, &bindings)
                        };

                        if let Some(ty) = final_type {
                            bindings.insert(var_name.to_string(), ty);
                        }
                    }
                }
            }
        }

        bindings
    }

    /// Infer the type of an expression on the right-hand side of a binding
    pub(crate) fn infer_rhs_type(expr: &str, engine: Option<&nostos_repl::ReplEngine>, current_bindings: &std::collections::HashMap<String, String>) -> Option<String> {
        let trimmed = expr.trim();

        // Check for method chain expressions like [["a","b"]].get(0).get(0) or x.chars().drop(1)
        if trimmed.contains('.') && trimmed.contains('(') {
            if let Some(inferred) = Self::infer_method_chain_type(trimmed, current_bindings) {
                eprintln!("Inferred method chain type: {} -> {}", trimmed, inferred);
                return Some(inferred);
            }
        }

        // Check for index expressions like g2[0][0] - use current bindings to resolve
        if trimmed.contains('[') && !trimmed.starts_with('[') {
            // This looks like an index expression (not a list literal)
            if let Some(inferred) = Self::infer_index_expr_type(trimmed, current_bindings) {
                eprintln!("Inferred index expression type: {} -> {}", trimmed, inferred);
                return Some(inferred);
            }
        }

        // List literals - analyze element type recursively
        // Handle both plain list literals [1,2,3] and indexed literals [["a","b"]][0][0]
        if trimmed.starts_with('[') {
            // Check if this is a list literal followed by index operations
            // e.g., [["a","b"]][0][0] or [[1,2]][0]
            if let Some(indexed_type) = Self::infer_indexed_list_literal_type(trimmed) {
                eprintln!("Inferred indexed list literal type: {} -> {}", trimmed, indexed_type);
                return Some(indexed_type);
            }
            // Plain list literal without indexing
            return Self::infer_list_type(trimmed);
        }
        if trimmed.starts_with('"') {
            return Some("String".to_string());
        }
        if trimmed.starts_with("%{") {
            return Some("Map".to_string());
        }
        if trimmed.starts_with("#{") {
            return Some("Set".to_string());
        }

        // Record/Variant construction: TypeName(field: value, ...) or ConstructorName(value)
        // Both start with uppercase letter followed by parentheses
        if let Some(first_char) = trimmed.chars().next() {
            if first_char.is_uppercase() {
                // Extract the type/constructor name (before parenthesis or space)
                let name: String = trimmed.chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();

                if !name.is_empty() {
                    // Check if it's followed by ( - indicates construction call
                    let rest = trimmed[name.len()..].trim_start();
                    let is_construction = rest.starts_with('(');

                    if let Some(engine) = engine {
                        // First check if it's a variant constructor (e.g., Success -> MyResult)
                        if let Some(type_name) = engine.get_type_for_constructor(&name) {
                            eprintln!("Inferred variant construction type: {} from constructor {}", type_name, name);
                            return Some(type_name);
                        }

                        // Otherwise check if it's a record type name directly (e.g., Person -> Person)
                        // Record types are their own constructors
                        let types = engine.get_types();
                        // First try exact match
                        if types.contains(&name) {
                            eprintln!("Inferred record construction type: {}", name);
                            return Some(name);
                        }
                        // Then try to find a qualified type that ends with this name
                        // (e.g., "Person" matches "module.Person")
                        for registered_type in &types {
                            let type_base = registered_type.rsplit('.').next().unwrap_or(registered_type);
                            if type_base == name {
                                eprintln!("Inferred record construction type (qualified): {}", registered_type);
                                return Some(registered_type.clone());
                            }
                        }
                    }

                    // Fallback: If it looks like a record construction (TypeName(field: value))
                    // and the type isn't registered yet, assume the type name equals the constructor name
                    if is_construction {
                        if rest.contains(':') {
                            // Pattern: TypeName(field: value, ...) - the colon indicates named field (record)
                            eprintln!("Inferred record type from construction pattern: {}", name);
                            return Some(name);
                        } else {
                            // Pattern: ConstructorName(value) - could be a variant constructor
                            // Without engine, we can't determine parent type, so just use the constructor name
                            // This is a best-effort fallback
                            eprintln!("Inferred type from constructor pattern (fallback): {}", name);
                            return Some(name);
                        }
                    }
                }
            }
        }

        // Numeric literals
        if trimmed.chars().all(|c| c.is_ascii_digit() || c == '-') && !trimmed.is_empty() {
            return Some("Int".to_string());
        }
        if trimmed.contains('.') && trimmed.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-') {
            return Some("Float".to_string());
        }

        // Try to infer type from function call: Module.func(...) or func(...)
        if let Some(paren_pos) = trimmed.find('(') {
            let func_part = trimmed[..paren_pos].trim();
            let args_part = &trimmed[paren_pos..];
            if let Some(engine) = engine {
                // Try to get the return type of the function
                if let Some(sig) = engine.get_function_signature(func_part) {
                    // Parse return type from signature like "(Int, Int) -> Int" or "Num a => a -> a -> a"
                    if let Some(arrow_pos) = sig.rfind("->") {
                        let ret_type = sig[arrow_pos + 2..].trim();

                        // If return type is a type variable (single lowercase letter), try to infer from arguments
                        if ret_type.len() == 1 && ret_type.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                            // Extract first argument and infer its type
                            if let Some(first_arg_type) = Self::infer_first_arg_type(args_part, current_bindings) {
                                return Some(first_arg_type);
                            }
                        }

                        return Some(ret_type.to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract and infer the type of the first argument in a function call
    fn infer_first_arg_type(args_str: &str, bindings: &std::collections::HashMap<String, String>) -> Option<String> {
        // args_str looks like "(arg1, arg2, ...)" or "(arg1)"
        let trimmed = args_str.trim();
        if !trimmed.starts_with('(') {
            return None;
        }

        // Find the first argument (handle nested parens/brackets)
        let inner = &trimmed[1..]; // Skip opening paren
        let mut depth = 0;
        let mut end_pos = 0;

        for (i, c) in inner.chars().enumerate() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    if depth == 0 {
                        end_pos = i;
                        break;
                    }
                    depth -= 1;
                }
                ',' if depth == 0 => {
                    end_pos = i;
                    break;
                }
                _ => {}
            }
        }

        if end_pos == 0 {
            // Single argument or find the closing paren
            end_pos = inner.find(')').unwrap_or(inner.len());
        }

        let first_arg = inner[..end_pos].trim();

        // Infer type of first argument
        if first_arg.is_empty() {
            return None;
        }

        // Check if it's a numeric literal
        if first_arg.chars().all(|c| c.is_ascii_digit() || c == '-') && !first_arg.is_empty() {
            return Some("Int".to_string());
        }
        if first_arg.contains('.') && first_arg.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-') {
            return Some("Float".to_string());
        }
        if first_arg.starts_with('"') {
            return Some("String".to_string());
        }
        if first_arg.starts_with('[') {
            return Self::infer_list_type(first_arg);
        }

        // Check if it's a known binding
        if let Some(ty) = bindings.get(first_arg) {
            return Some(ty.clone());
        }

        None
    }

    /// Infer the type of an indexed list literal expression
    /// e.g., [["a","b"]][0] -> List[String], [["a","b"]][0][0] -> String
    fn infer_indexed_list_literal_type(expr: &str) -> Option<String> {
        let trimmed = expr.trim();

        // Must start with '[' (list literal)
        if !trimmed.starts_with('[') {
            return None;
        }

        // Find the matching closing bracket for the list literal part
        // We need to handle nested brackets like [["a","b"]]
        let mut depth = 0;
        let mut list_end = None;

        for (i, c) in trimmed.chars().enumerate() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        list_end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        let list_end = list_end?;

        // Check if there are index operations after the list literal
        let after_list = &trimmed[list_end + 1..];
        if !after_list.starts_with('[') {
            // No index operations - this is just a plain list literal
            return None;
        }

        // Count how many index operations there are (each [...] after the list literal)
        let index_count = after_list.matches('[').count();

        if index_count == 0 {
            return None;
        }

        // Get the list literal part and infer its type
        let list_literal = &trimmed[..=list_end];
        let base_type = Self::infer_list_type(list_literal)?;

        // Unwrap one level of List[...] for each index operation
        let mut current_type = base_type;
        for _ in 0..index_count {
            if current_type.starts_with("List[") && current_type.ends_with(']') {
                current_type = current_type
                    .strip_prefix("List[")?
                    .strip_suffix(']')?
                    .to_string();
            } else if current_type == "List" {
                // Generic List without element type - can't infer further
                return None;
            } else {
                // Not a List type - this is the final element type
                return Some(current_type);
            }
        }

        Some(current_type)
    }

    /// Infer the type of a list literal, handling nested lists
    /// e.g., [[0,1]] -> List[List[Int]], [1,2,3] -> List[Int]
    fn infer_list_type(expr: &str) -> Option<String> {
        let trimmed = expr.trim();

        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return None;
        }

        let inner = trimmed[1..trimmed.len() - 1].trim();

        if inner.is_empty() {
            return Some("List".to_string());
        }

        // Find the first element (handle nested brackets)
        let first_elem = Self::extract_first_list_element(inner)?;
        let first_trimmed = first_elem.trim();

        // Recursively infer element type
        let elem_type = if first_trimmed.starts_with('[') {
            // Nested list
            Self::infer_list_type(first_trimmed)?
        } else if first_trimmed.starts_with('"') {
            "String".to_string()
        } else if first_trimmed.parse::<i64>().is_ok() {
            "Int".to_string()
        } else if first_trimmed.parse::<f64>().is_ok() {
            "Float".to_string()
        } else {
            // Unknown element type
            return Some("List".to_string());
        };

        Some(format!("List[{}]", elem_type))
    }

    /// Extract the first element from a list interior, handling nested brackets
    fn extract_first_list_element(inner: &str) -> Option<String> {
        let mut depth = 0;
        let mut end_pos = inner.len();

        for (i, c) in inner.chars().enumerate() {
            match c {
                '[' | '(' | '{' => depth += 1,
                ']' | ')' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    end_pos = i;
                    break;
                }
                _ => {}
            }
        }

        Some(inner[..end_pos].to_string())
    }

    /// Infer the type of a lambda parameter from context
    /// For "yy.map(m => m." where yy is a List, returns "Int" (element type)
    /// Handles nested lambdas like "gg.map(m => m.map(n => n." by recursively inferring types
    fn infer_lambda_param_type(
        full_prefix: &str,
        before_dot: &str,
        local_vars: &std::collections::HashMap<String, String>,
    ) -> Option<String> {
        Self::infer_lambda_param_type_recursive(full_prefix, before_dot, local_vars, 0)
    }

    /// Recursive helper for lambda parameter type inference
    /// depth limits recursion to prevent infinite loops
    fn infer_lambda_param_type_recursive(
        full_prefix: &str,
        before_dot: &str,
        local_vars: &std::collections::HashMap<String, String>,
        depth: usize,
    ) -> Option<String> {
        // Limit recursion depth
        if depth > 5 {
            return None;
        }

        // Debug log helper
        let log = |msg: &str| {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                let _ = writeln!(f, "LAMBDA[{}]: {}", depth, msg);
            }
        };

        log(&format!("full_prefix='{}', before_dot='{}'", full_prefix, before_dot));

        // Extract the identifier we're completing (e.g., "m" from "m.")
        let param_name = before_dot
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .last()?;

        log(&format!("param_name='{}'", param_name));

        // Look for lambda arrow pattern: "param =>" or "param=" before the current position
        let lambda_pattern = format!("{} =>", param_name);
        let alt_pattern1 = format!("{}=>", param_name);
        let alt_pattern2 = format!("{} =", param_name);
        let alt_pattern3 = format!("{}=", param_name);

        log(&format!("Looking for patterns: '{}', '{}', '{}', '{}'", lambda_pattern, alt_pattern1, alt_pattern2, alt_pattern3));

        let arrow_pos = full_prefix.rfind(&lambda_pattern)
            .or_else(|| full_prefix.rfind(&alt_pattern1))
            .or_else(|| full_prefix.rfind(&alt_pattern2))
            .or_else(|| full_prefix.rfind(&alt_pattern3))?;

        log(&format!("arrow_pos={}", arrow_pos));

        // Now look backwards from arrow_pos to find the method call context
        let before_lambda = &full_prefix[..arrow_pos];

        // Find the opening paren that contains this lambda
        let mut paren_depth: i32 = 0;
        let mut method_call_start = None;
        for (i, c) in before_lambda.chars().rev().enumerate() {
            match c {
                ')' | ']' | '}' => paren_depth += 1,
                '(' => {
                    if paren_depth == 0 {
                        method_call_start = Some(before_lambda.len() - i - 1);
                        break;
                    }
                    paren_depth -= 1;
                }
                '[' | '{' => paren_depth = (paren_depth - 1).max(0),
                _ => {}
            }
        }

        let paren_pos = method_call_start?;
        log(&format!("paren_pos={}", paren_pos));
        let before_paren = before_lambda[..paren_pos].trim();
        log(&format!("before_paren='{}'", before_paren));

        // Find the method name and receiver: "receiver.method"
        // For nested lambdas like "m.map", we need the LAST dot
        let dot_pos = before_paren.rfind('.')?;
        let method_name = before_paren[dot_pos + 1..].trim();
        let receiver_expr = before_paren[..dot_pos].trim();

        log(&format!("receiver_expr='{}', method='{}'", receiver_expr, method_name));

        // Extract the receiver variable name (the last identifier)
        let receiver_var = receiver_expr
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .last()
            .unwrap_or(receiver_expr);

        log(&format!("receiver_var='{}'", receiver_var));

        // Try to get receiver type from local_vars first
        let receiver_type = if let Some(t) = local_vars.get(receiver_var) {
            log(&format!("Found receiver_var '{}' in local_vars: {}", receiver_var, t));
            Some(t.clone())
        } else if let Some(t) = Self::infer_literal_type(receiver_expr) {
            log(&format!("Inferred literal type for '{}': {}", receiver_expr, t));
            Some(t)
        } else {
            // receiver_var is NOT in local_vars - it might be a lambda param itself!
            // Check if receiver_var appears as a lambda parameter earlier in the prefix
            log(&format!("receiver_var '{}' not found, checking if it's a lambda param", receiver_var));

            // Build a fake "before_dot" to recursively infer the receiver's type
            // We need to find where receiver_var is defined as a lambda param
            let receiver_before_dot = receiver_var.to_string();

            // Recursively infer the type of the receiver
            Self::infer_lambda_param_type_recursive(
                before_paren,  // Look in the context before this call
                &receiver_before_dot,
                local_vars,
                depth + 1
            )
        };

        let receiver_type = receiver_type?;
        log(&format!("receiver_type='{}'", receiver_type));

        // Infer lambda parameter type based on receiver type and method
        let result = Self::infer_lambda_param_type_for_method(&receiver_type, method_name);
        log(&format!("infer_lambda_param_type_for_method result: {:?}", result));
        result
    }

    /// Infer the type of a lambda parameter based on receiver type and method name
    fn infer_lambda_param_type_for_method(receiver_type: &str, method_name: &str) -> Option<String> {
        // For List methods, the lambda parameter is often the element type
        if receiver_type.starts_with("List") || receiver_type.starts_with('[') || receiver_type == "List" {
            // Extract element type from List[X] or [X]
            let element_type = if receiver_type.starts_with("List[") {
                receiver_type.strip_prefix("List[")?.strip_suffix(']')?.to_string()
            } else if receiver_type.starts_with('[') && receiver_type.ends_with(']') {
                receiver_type[1..receiver_type.len()-1].to_string()
            } else {
                // Generic List without element type - assume Int for [1,2,3] style literals
                "Int".to_string()
            };

            // Methods where lambda param is element type
            match method_name {
                "map" | "filter" | "each" | "any" | "all" | "find" | "takeWhile" | "dropWhile" |
                "partition" | "span" | "sortBy" | "groupBy" | "count" => {
                    return Some(element_type);
                }
                "fold" | "foldl" | "foldr" => {
                    // For fold, second param of lambda is element type
                    // First param is accumulator - can't easily infer
                    return Some(element_type);
                }
                "zipWith" => {
                    // zipWith takes (a, b) -> c, complex to infer
                    return Some(element_type);
                }
                _ => {}
            }
        }

        // For Option methods
        if receiver_type.starts_with("Option") || receiver_type == "Option" {
            let inner_type = if receiver_type.starts_with("Option ") {
                receiver_type.strip_prefix("Option ")?.to_string()
            } else {
                "a".to_string() // Generic
            };

            match method_name {
                "map" | "flatMap" | "filter" => return Some(inner_type),
                _ => {}
            }
        }

        // For Result methods
        if receiver_type.starts_with("Result") || receiver_type == "Result" {
            match method_name {
                "map" => return Some("a".to_string()), // Ok value
                "mapErr" => return Some("e".to_string()), // Err value
                _ => {}
            }
        }

        // For Map methods
        if receiver_type.starts_with("Map") || receiver_type == "Map" {
            match method_name {
                "map" | "filter" | "each" => {
                    // Map iteration gives (key, value) pairs
                    return Some("(k, v)".to_string());
                }
                _ => {}
            }
        }

        // For Set methods
        if receiver_type.starts_with("Set") || receiver_type == "Set" {
            let element_type = if receiver_type.starts_with("Set[") {
                receiver_type.strip_prefix("Set[")?.strip_suffix(']')?.to_string()
            } else {
                "a".to_string()
            };

            match method_name {
                "map" | "filter" | "each" | "any" | "all" => return Some(element_type),
                _ => {}
            }
        }

        None
    }

    /// Infer type from a literal expression
    fn infer_literal_type(expr: &str) -> Option<String> {
        let trimmed = expr.trim();
        // Use the recursive list type inference
        if trimmed.starts_with('[') {
            return Self::infer_list_type(trimmed);
        }
        if trimmed.starts_with('"') {
            return Some("String".to_string());
        }
        if trimmed.parse::<i64>().is_ok() {
            return Some("Int".to_string());
        }
        if trimmed.parse::<f64>().is_ok() {
            return Some("Float".to_string());
        }
        None
    }

    /// Extract record type fields directly from source code
    /// This works even when the file has parse errors elsewhere
    /// Pattern: "type TypeName = { field1: Type1, field2: Type2 }"
    /// Extract document symbols (functions, types, traits) from source content
    fn extract_document_symbols(content: &str) -> Vec<SymbolInformation> {
        let mut symbols = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Type definitions: "type Foo = ..." or "type Foo[T] = ..."
            if trimmed.starts_with("type ") {
                let rest = &trimmed[5..];
                let name = rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                if !name.is_empty() {
                    #[allow(deprecated)]
                    symbols.push(SymbolInformation {
                        name: name.to_string(),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: Url::parse("file:///").unwrap(), // Will be replaced
                            range: Range {
                                start: Position { line: line_num as u32, character: 0 },
                                end: Position { line: line_num as u32, character: line.len() as u32 },
                            },
                        },
                        container_name: None,
                    });
                }
            }
            // Trait definitions: "trait Foo" or "trait Foo[T]"
            else if trimmed.starts_with("trait ") {
                let rest = &trimmed[6..];
                let name = rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                if !name.is_empty() {
                    #[allow(deprecated)]
                    symbols.push(SymbolInformation {
                        name: name.to_string(),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: Url::parse("file:///").unwrap(), // Will be replaced
                            range: Range {
                                start: Position { line: line_num as u32, character: 0 },
                                end: Position { line: line_num as u32, character: line.len() as u32 },
                            },
                        },
                        container_name: None,
                    });
                }
            }
            // Function definitions: "foo(...) = ..." or "pub foo(...) = ..."
            // But not inside trait impl blocks (indented), and not trait impl headers
            else if !line.starts_with(' ') && !line.starts_with('\t') {
                // Skip trait implementation headers: "TypeName: TraitName" or "TypeName[T]: TraitName"
                if trimmed.contains(':') && !trimmed.contains('(') && !trimmed.contains('=') {
                    continue;
                }

                // Look for function pattern: name(...) = or pub name(...) =
                let check_line = if trimmed.starts_with("pub ") {
                    &trimmed[4..]
                } else {
                    trimmed
                };

                // Must have ( and = and not start with keyword
                if check_line.contains('(') && check_line.contains('=') {
                    let fn_name = check_line.split('(')
                        .next()
                        .unwrap_or("")
                        .trim();

                    // Skip if it looks like a keyword or invalid
                    if !fn_name.is_empty()
                        && fn_name.chars().next().map(|c| c.is_alphabetic() || c == '_').unwrap_or(false)
                        && !["if", "else", "match", "let", "type", "trait", "end", "import"].contains(&fn_name)
                    {
                        #[allow(deprecated)]
                        symbols.push(SymbolInformation {
                            name: fn_name.to_string(),
                            kind: SymbolKind::FUNCTION,
                            tags: None,
                            deprecated: None,
                            location: Location {
                                uri: Url::parse("file:///").unwrap(), // Will be replaced
                                range: Range {
                                    start: Position { line: line_num as u32, character: 0 },
                                    end: Position { line: line_num as u32, character: line.len() as u32 },
                                },
                            },
                            container_name: None,
                        });
                    }
                }
            }
        }

        symbols
    }

    /// Find all references to a word in content and add to locations
    fn find_references_in_content(word: &str, uri: &Url, content: &str, locations: &mut Vec<Location>) {
        for (line_num, line) in content.lines().enumerate() {
            // Find all occurrences of word in line
            let mut search_start = 0;
            while let Some(pos) = line[search_start..].find(word) {
                let actual_pos = search_start + pos;

                // Check word boundaries - must not be part of a larger identifier
                let before_ok = actual_pos == 0
                    || !line.chars().nth(actual_pos - 1).map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
                let after_ok = actual_pos + word.len() >= line.len()
                    || !line.chars().nth(actual_pos + word.len()).map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);

                if before_ok && after_ok {
                    locations.push(Location {
                        uri: uri.clone(),
                        range: Range {
                            start: Position { line: line_num as u32, character: actual_pos as u32 },
                            end: Position { line: line_num as u32, character: (actual_pos + word.len()) as u32 },
                        },
                    });
                }

                search_start = actual_pos + word.len();
            }
        }
    }

    fn extract_type_fields_from_source(content: &str, type_name: &str) -> Vec<String> {
        let mut fields = Vec::new();

        // Look for type definition pattern: "type TypeName = { ... }"
        // Handle both single-line and multi-line definitions
        for line in content.lines() {
            let trimmed = line.trim();

            // Check for record type definition start
            // Pattern: "type Person = { name: String, age: Int }"
            // or: "type Person = {"
            if trimmed.starts_with("type ") {
                let rest = &trimmed[5..].trim();

                // Extract type name (before = or [)
                let def_type_name = rest.split(|c| c == '=' || c == '[')
                    .next()
                    .unwrap_or("")
                    .trim();

                if def_type_name == type_name {
                    // Found the type definition - extract fields from { ... }
                    if let Some(brace_start) = trimmed.find('{') {
                        // Find the matching }
                        let after_brace = &trimmed[brace_start + 1..];
                        if let Some(brace_end) = after_brace.find('}') {
                            // Extract fields between braces
                            let fields_str = &after_brace[..brace_end];
                            for field in fields_str.split(',') {
                                let field_trimmed = field.trim();
                                if !field_trimmed.is_empty() {
                                    fields.push(field_trimmed.to_string());
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        fields
    }

    /// Infer type of an index expression like g2[0] or g2[0][0]
    /// If g2 has type List[List[String]], then:
    ///   g2[0] -> List[String]
    ///   g2[0][0] -> String
    /// Infer the type of a field access like `self.age` where `self` is in local_vars
    /// and `age` is a field of the type of `self`.
    ///
    /// This handles chained completions like `self.age.` where we need to find:
    /// 1. `self` in local_vars -> `Person`
    /// 2. `age` field in `Person` -> `Int`
    /// 3. Show methods for `Int`
    fn infer_field_access_type(
        before_dot: &str,
        field_name: &str,
        local_vars: &std::collections::HashMap<String, String>,
        engine: &nostos_repl::ReplEngine,
        document_content: &str,
    ) -> Option<String> {
        // Look for pattern: something.field_name at the end of before_dot
        // e.g., before_dot = "... self.age", field_name = "age"
        // We need to find "self" and get its type, then get the type of "age" field

        // Find where .field_name appears at the end
        let pattern = format!(".{}", field_name);
        let field_start = before_dot.rfind(&pattern)?;

        // Get everything before the .field_name
        let before_field = &before_dot[..field_start];

        // Extract the base variable - it's the last identifier before the field access
        // E.g., from "... self" we want "self"
        let base_var = before_field
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .last()?;

        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "infer_field_access_type: before_dot='{}', field_name='{}', base_var='{}'", before_dot, field_name, base_var);
        }

        // Get the type of the base variable
        let base_type = local_vars.get(base_var)?;

        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "infer_field_access_type: base_var='{}' has type '{}'", base_var, base_type);
        }

        // Now look up the field type in the base type
        // First try engine.get_field_type
        if let Some(field_type) = engine.get_field_type(base_type, field_name) {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                use std::io::Write;
                let _ = writeln!(f, "infer_field_access_type: found field '{}' with type '{}' via engine", field_name, field_type);
            }
            return Some(field_type);
        }

        // If engine lookup fails (e.g., file has parse errors), try extracting from source
        let fields = Self::extract_type_fields_from_source(document_content, base_type);
        for field in fields {
            // Fields are in "name: Type" format
            if let Some(colon_pos) = field.find(':') {
                let name = field[..colon_pos].trim();
                let ty = field[colon_pos + 1..].trim();
                if name == field_name {
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                        use std::io::Write;
                        let _ = writeln!(f, "infer_field_access_type: found field '{}' with type '{}' from source", field_name, ty);
                    }
                    return Some(ty.to_string());
                }
            }
        }

        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "infer_field_access_type: could not find field '{}' in type '{}'", field_name, base_type);
        }

        None
    }

    /// Infer the type of a method chain expression like [["a","b"]].get(0).get(0)
    fn infer_method_chain_type(expr: &str, local_vars: &std::collections::HashMap<String, String>) -> Option<String> {
        let trimmed = expr.trim();

        // Split the expression into base and method calls
        // We need to handle nested brackets and parentheses properly
        let mut current_type: Option<String> = None;
        let mut remaining = trimmed;

        // First, find the base expression (before the first method call)
        // Handle cases like: [["a","b"]].get(0) or x.method() or "hello".chars()

        // Find the start of method calls by looking for .methodName( pattern
        // But we need to skip over any . that's part of a float literal or inside brackets

        let mut depth = 0;
        let mut base_end = 0;
        let chars: Vec<char> = remaining.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            match c {
                '[' | '(' | '{' => depth += 1,
                ']' | ')' | '}' => depth -= 1,
                '.' if depth == 0 => {
                    // Check if this starts a method call (followed by identifier and paren)
                    let after_dot: String = chars[i+1..].iter().collect();
                    if after_dot.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                        base_end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if base_end == 0 {
            // No method call found
            return None;
        }

        let base_expr = &remaining[..base_end];
        remaining = &remaining[base_end..];

        // Infer the base type
        if base_expr.starts_with('[') {
            current_type = Self::infer_list_type(base_expr);
        } else if base_expr.starts_with('"') {
            current_type = Some("String".to_string());
        } else if let Some(ty) = local_vars.get(base_expr.trim()) {
            current_type = Some(ty.clone());
        }

        // Now process each method call
        while !remaining.is_empty() && remaining.starts_with('.') {
            remaining = &remaining[1..]; // Skip the dot

            // Find the method name and arguments
            let paren_pos = remaining.find('(')?;
            let method_name = &remaining[..paren_pos];

            // Find matching closing paren
            let mut depth = 0;
            let mut close_paren = None;
            for (i, c) in remaining[paren_pos..].chars().enumerate() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            close_paren = Some(paren_pos + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            let close_paren = close_paren?;

            // Apply the method to get the new type
            if let Some(ref recv_type) = current_type {
                current_type = Self::infer_method_return_type(recv_type, method_name);
            } else {
                return None;
            }

            // Move past this method call
            remaining = &remaining[close_paren + 1..];
        }

        current_type
    }

    /// Infer the return type of a method call based on receiver type
    fn infer_method_return_type(receiver_type: &str, method_name: &str) -> Option<String> {
        // Generic methods that work on any type
        match method_name {
            "show" => return Some("String".to_string()),
            "hash" => return Some("Int".to_string()),
            "copy" => return Some(receiver_type.to_string()),
            _ => {}
        }

        // Extract base type and element type for parameterized types
        let (base_type, elem_type) = if receiver_type.starts_with("List[") && receiver_type.ends_with(']') {
            ("List", Some(&receiver_type[5..receiver_type.len()-1]))
        } else if receiver_type.starts_with("Option[") && receiver_type.ends_with(']') {
            ("Option", Some(&receiver_type[7..receiver_type.len()-1]))
        } else {
            (receiver_type, None)
        };

        match base_type {
            "List" => {
                match method_name {
                    // Methods that preserve List type
                    "filter" | "take" | "drop" | "reverse" | "sort" | "unique" |
                    "takeWhile" | "dropWhile" | "init" | "tail" | "push" | "remove" |
                    "removeAt" | "insertAt" | "set" | "slice" => {
                        if let Some(elem) = elem_type {
                            Some(format!("List[{}]", elem))
                        } else {
                            Some("List".to_string())
                        }
                    }
                    // Methods that return element type
                    "get" | "head" | "last" | "nth" | "find" | "sum" | "product" |
                    "maximum" | "minimum" => {
                        elem_type.map(|e| e.to_string())
                    }
                    // Methods that return Bool
                    "any" | "all" | "contains" | "isEmpty" => Some("Bool".to_string()),
                    // Methods that return Int
                    "length" | "len" | "count" | "indexOf" => Some("Int".to_string()),
                    // Methods that return Option[element]
                    "first" | "safeHead" | "safeLast" => {
                        elem_type.map(|e| format!("Option[{}]", e))
                    }
                    // map transforms element type - can't infer without lambda
                    "map" | "flatMap" => Some("List".to_string()),
                    "enumerate" => {
                        if let Some(elem) = elem_type {
                            Some(format!("List[(Int, {})]", elem))
                        } else {
                            Some("List".to_string())
                        }
                    }
                    "flatten" => {
                        // If element is List[X], result is List[X]
                        if let Some(elem) = elem_type {
                            if elem.starts_with("List[") {
                                Some(elem.to_string())
                            } else {
                                Some(format!("List[{}]", elem))
                            }
                        } else {
                            Some("List".to_string())
                        }
                    }
                    _ => None,
                }
            }
            "String" => {
                match method_name {
                    "chars" => Some("List[Char]".to_string()),
                    "lines" | "words" | "split" => Some("List[String]".to_string()),
                    "trim" | "trimStart" | "trimEnd" | "toUpper" | "toLower" |
                    "replace" | "replaceAll" | "substring" | "repeat" |
                    "padStart" | "padEnd" | "reverse" => Some("String".to_string()),
                    "length" | "indexOf" | "lastIndexOf" => Some("Int".to_string()),
                    "contains" | "startsWith" | "endsWith" | "isEmpty" => Some("Bool".to_string()),
                    _ => None,
                }
            }
            "Option" => {
                match method_name {
                    "unwrap" | "getOrElse" => elem_type.map(|e| e.to_string()),
                    "isSome" | "isNone" => Some("Bool".to_string()),
                    "map" => Some("Option".to_string()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn infer_index_expr_type(expr: &str, local_vars: &std::collections::HashMap<String, String>) -> Option<String> {
        let trimmed = expr.trim();

        // Check if expression contains index access
        if !trimmed.contains('[') {
            return None;
        }

        // Find the base variable (everything before the first '[')
        let first_bracket = trimmed.find('[')?;
        let base_var = trimmed[..first_bracket].trim();

        if base_var.is_empty() {
            return None;
        }

        // Get the base variable's type
        let base_type = local_vars.get(base_var)?;

        // Count the number of index operations
        let index_count = trimmed.matches('[').count();

        // "Unwrap" the List type for each index operation
        let mut current_type = base_type.clone();
        for _ in 0..index_count {
            // Strip one level of List[...]
            if current_type.starts_with("List[") && current_type.ends_with(']') {
                current_type = current_type
                    .strip_prefix("List[")?
                    .strip_suffix(']')?
                    .to_string();
            } else if current_type == "List" {
                // Generic List without element type - can't infer further
                return None;
            } else {
                // Not a List type, can't index further
                return None;
            }
        }

        eprintln!("Inferred index expr type for '{}': {}", expr, current_type);
        Some(current_type)
    }

    /// Get completions after a dot (module functions or UFCS methods)
    fn get_dot_completions(&self, before_dot: &str, local_vars: &std::collections::HashMap<String, String>, lambda_param_type: Option<&str>, document_content: &str) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                use std::io::Write;
                let _ = writeln!(f, "ERROR: engine is None in get_dot_completions (after async wait)!");
            }
            return items;
        };

        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "get_dot_completions: engine available, before_dot='{}', lambda_param_type={:?}", before_dot, lambda_param_type);
        }

        // If we have a lambda parameter type, use that directly
        if let Some(param_type) = lambda_param_type {
            eprintln!("Using lambda param type: {}", param_type);

            // Add type indicator at top of list
            items.push(CompletionItem {
                label: format!(": {}", param_type),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                detail: Some("Lambda parameter type".to_string()),
                documentation: Some(Documentation::String(format!("Parameter type: {}", param_type))),
                sort_text: Some("!0".to_string()), // Sort first
                filter_text: Some("".to_string()), // Don't filter this item
                ..Default::default()
            });

            let mut seen = std::collections::HashSet::new();

            for (method_name, signature, doc) in nostos_repl::ReplEngine::get_builtin_methods_for_type(param_type) {
                if !seen.insert(method_name.to_string()) {
                    continue;
                }
                items.push(CompletionItem {
                    label: method_name.to_string(),
                    kind: Some(CompletionItemKind::METHOD),
                    detail: Some(signature.to_string()),
                    documentation: Some(Documentation::String(doc.to_string())),
                    ..Default::default()
                });
            }

            for (method_name, signature, doc) in engine.get_ufcs_methods_for_type(param_type) {
                if !seen.insert(method_name.clone()) {
                    continue;
                }
                items.push(CompletionItem {
                    label: method_name,
                    kind: Some(CompletionItemKind::METHOD),
                    detail: Some(signature),
                    documentation: doc.map(|d| Documentation::String(d)),
                    ..Default::default()
                });
            }

            return items;
        }

        // Extract the identifier before the dot
        let identifier = before_dot
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .last()
            .unwrap_or("");

        eprintln!("Identifier before dot: '{}'", identifier);

        // Debug log
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
            use std::io::Write;
            let _ = writeln!(f, "get_dot_completions: identifier='{}', local_vars={:?}", identifier, local_vars);
        }

        // Check if it's a known module name (capitalize first letter to match module convention)
        let potential_module = if !identifier.is_empty() {
            let mut chars: Vec<char> = identifier.chars().collect();
            chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
            chars.into_iter().collect::<String>()
        } else {
            String::new()
        };

        // Check if it matches a module
        let known_modules: Vec<String> = engine.get_functions()
            .iter()
            .filter_map(|f| f.split('.').next().map(|s| s.to_string()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // IMPORTANT: Check if identifier is a local variable FIRST, before checking modules
        // This prevents local vars like "p" from being confused with module functions like Html.p
        let is_local_var = local_vars.contains_key(identifier);

        if !is_local_var && (known_modules.contains(&potential_module) || known_modules.contains(&identifier.to_string())) {
            // It's a module - show module functions
            let module_name = if known_modules.contains(&potential_module) {
                &potential_module
            } else {
                identifier
            };

            eprintln!("Found module: {}", module_name);

            for fn_name in engine.get_functions() {
                if fn_name.starts_with(&format!("{}.", module_name)) {
                    let short_name = fn_name.strip_prefix(&format!("{}.", module_name))
                        .unwrap_or(&fn_name);

                    // Skip internal functions (starting with underscore)
                    if short_name.starts_with('_') {
                        continue;
                    }

                    let signature = engine.get_function_signature(&fn_name);
                    let doc = engine.get_function_doc(&fn_name);

                    items.push(CompletionItem {
                        label: short_name.to_string(),
                        kind: Some(CompletionItemKind::FUNCTION),
                        detail: signature,
                        documentation: doc.map(|d| Documentation::String(d)),
                        ..Default::default()
                    });
                }
            }
        } else {
            // Not a module - try to infer the type and show methods
            eprintln!("Not a module, trying to infer type of: '{}'", before_dot);

            // Extract just the expression part if before_dot contains an assignment
            // e.g., "x2 = g2[0][0]" -> "g2[0][0]"
            let expr_to_infer = if let Some(eq_pos) = before_dot.rfind('=') {
                // Make sure it's not == or !=
                let before_eq = &before_dot[..eq_pos];
                if !before_eq.ends_with('!') && !before_eq.ends_with('=') && !before_eq.ends_with('<') && !before_eq.ends_with('>') {
                    before_dot[eq_pos + 1..].trim()
                } else {
                    before_dot
                }
            } else {
                before_dot
            };
            eprintln!("Expression to infer: '{}'", expr_to_infer);

            // Extract the full receiver expression (handles literals like [1,2,3], "hello", etc.)
            let receiver_expr = Self::extract_receiver_expression(expr_to_infer);
            eprintln!("Receiver expression: '{}'", receiver_expr);

            // First check for literal types (string, list, etc.)
            let literal_type = Self::detect_literal_type(receiver_expr);
            if let Some(lt) = literal_type {
                eprintln!("Detected literal type: {}", lt);
            }

            // IMPORTANT: First check if the last identifier (extracted earlier) is a simple
            // variable in local_vars. This handles cases like "self.name ++ self." where
            // the expression is complex but we just need the type of "self".
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                use std::io::Write;
                let _ = writeln!(f, "About to check identifier '{}' in local_vars, receiver_expr='{}', literal_type={:?}", identifier, receiver_expr, literal_type);
            }
            let inferred_type = if let Some(lt) = literal_type {
                // Use literal type directly
                Some(lt.to_string())
            } else if let Some(ty) = local_vars.get(identifier) {
                eprintln!("Found identifier '{}' directly in local_vars with type: {}", identifier, ty);
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                    use std::io::Write;
                    let _ = writeln!(f, "Found identifier '{}' in local_vars with type: {}", identifier, ty);
                }
                Some(ty.clone())
            } else if let Some(field_type) = Self::infer_field_access_type(before_dot, identifier, local_vars, engine, document_content) {
                // Check if this is a field access like self.age where self is in local_vars
                eprintln!("Inferred field access type for '{}': {}", identifier, field_type);
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                    use std::io::Write;
                    let _ = writeln!(f, "Inferred field access type for '{}': {}", identifier, field_type);
                }
                Some(field_type)
            } else if let Some(idx_literal_type) = Self::infer_indexed_list_literal_type(expr_to_infer) {
                // Check if it's an indexed list literal like [["a","b"]][0][0]
                eprintln!("Inferred indexed list literal type: {}", idx_literal_type);
                Some(idx_literal_type)
            } else if let Some(idx_type) = Self::infer_index_expr_type(expr_to_infer, local_vars) {
                // Check if it's an index expression like g2[0] or g2[0][0]
                eprintln!("Inferred index expression type: {}", idx_type);
                Some(idx_type)
            } else if let Some(func_ret_type) = Self::infer_rhs_type(expr_to_infer, Some(engine), local_vars) {
                // Check if it's a function call like good.addff(3,2)
                eprintln!("Inferred function call return type: {}", func_ret_type);
                Some(func_ret_type)
            } else {
                // Use the engine's general expression type inference
                // which handles method chains, index expressions, and local bindings
                engine.infer_expression_type(expr_to_infer, &local_vars)
            };
            eprintln!("Inferred type: {:?}", inferred_type);

            // Add type indicator at top of list
            if let Some(ref type_name) = inferred_type {
                items.push(CompletionItem {
                    label: format!(": {}", type_name),
                    kind: Some(CompletionItemKind::TYPE_PARAMETER),
                    detail: Some("Inferred type".to_string()),
                    documentation: Some(Documentation::String(format!("Expression type: {}", type_name))),
                    sort_text: Some("!0".to_string()), // Sort first (! comes before letters)
                    filter_text: Some("".to_string()), // Don't filter this item
                    ..Default::default()
                });
            }

            let mut seen = std::collections::HashSet::new();

            if let Some(ref type_name) = inferred_type {
                // Show builtin methods for the inferred type
                for (method_name, signature, doc) in nostos_repl::ReplEngine::get_builtin_methods_for_type(type_name) {
                    if !seen.insert(method_name.to_string()) {
                        continue;
                    }

                    items.push(CompletionItem {
                        label: method_name.to_string(),
                        kind: Some(CompletionItemKind::METHOD),
                        detail: Some(signature.to_string()),
                        documentation: Some(Documentation::String(doc.to_string())),
                        ..Default::default()
                    });
                }

                // Also add UFCS methods from user-defined functions
                for (method_name, signature, doc) in engine.get_ufcs_methods_for_type(type_name) {
                    if !seen.insert(method_name.clone()) {
                        continue;
                    }

                    items.push(CompletionItem {
                        label: method_name,
                        kind: Some(CompletionItemKind::METHOD),
                        detail: Some(signature),
                        documentation: doc.map(|d| Documentation::String(d)),
                        ..Default::default()
                    });
                }

                // Add trait methods implemented for the type
                for (method_name, signature, doc) in engine.get_trait_methods_for_type(type_name) {
                    if !seen.insert(method_name.clone()) {
                        continue;
                    }

                    items.push(CompletionItem {
                        label: method_name,
                        kind: Some(CompletionItemKind::METHOD),
                        detail: Some(signature),
                        documentation: doc.map(|d| Documentation::String(d)),
                        ..Default::default()
                    });
                }

                // Add record fields for the type
                let all_types = engine.get_types();
                let fields = engine.get_type_fields(type_name);
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                    use std::io::Write;
                    let _ = writeln!(f, "");
                    let _ = writeln!(f, "=== get_dot_completions for type '{}' ===", type_name);
                    let _ = writeln!(f, "Total types count: {}", all_types.len());
                    // Filter to show user types (non-stdlib)
                    let user_types: Vec<_> = all_types.iter().filter(|t| !t.starts_with("stdlib.")).collect();
                    let _ = writeln!(f, "User types (non-stdlib): {:?}", user_types);
                    let _ = writeln!(f, "Fields for '{}': {:?}", type_name, fields);
                }

                // Also try with different name formats if direct lookup fails
                let fields = if fields.is_empty() {
                    // Try looking up with possible module prefixes
                    let mut found_fields = Vec::new();
                    for t in &all_types {
                        if t.ends_with(&format!(".{}", type_name)) || t == type_name {
                            found_fields = engine.get_type_fields(t);
                            if !found_fields.is_empty() {
                                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                                    use std::io::Write;
                                    let _ = writeln!(f, "Found fields via '{}': {:?}", t, found_fields);
                                }
                                break;
                            }
                        }
                    }

                    // If still no fields found, try extracting directly from source code
                    // This works even when the file has parse errors
                    if found_fields.is_empty() {
                        found_fields = Self::extract_type_fields_from_source(document_content, type_name);
                        if !found_fields.is_empty() {
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
                                use std::io::Write;
                                let _ = writeln!(f, "Found fields from source for '{}': {:?}", type_name, found_fields);
                            }
                        }
                    }

                    found_fields
                } else {
                    fields
                };

                for field in fields {
                    if !seen.insert(field.clone()) {
                        continue;
                    }

                    // Parse field name and type from "name: Type" format
                    let (field_name, field_type) = if let Some(colon_pos) = field.find(':') {
                        (field[..colon_pos].trim().to_string(), Some(field[colon_pos + 1..].trim().to_string()))
                    } else {
                        (field.clone(), None)
                    };

                    items.push(CompletionItem {
                        label: field_name,
                        kind: Some(CompletionItemKind::FIELD),
                        detail: field_type,
                        documentation: Some(Documentation::String(format!("Field of {}", type_name))),
                        sort_text: Some("!1".to_string()), // Sort after type indicator but before methods
                        ..Default::default()
                    });
                }
            }

            // If no type inferred, show generic methods
            if inferred_type.is_none() {
                for (method_name, signature, doc) in nostos_repl::ReplEngine::get_builtin_methods_for_type("Unknown") {
                    items.push(CompletionItem {
                        label: method_name.to_string(),
                        kind: Some(CompletionItemKind::METHOD),
                        detail: Some(signature.to_string()),
                        documentation: Some(Documentation::String(doc.to_string())),
                        ..Default::default()
                    });
                }
            }
        }

        items
    }

    /// Get completions for identifier at cursor position
    fn get_identifier_completions(&self, partial: &str) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        let partial_lower = partial.to_lowercase();

        // Add keywords first (always available, even without engine)
        let keywords = [
            ("if", "Conditional expression"),
            ("then", "Then branch of conditional"),
            ("else", "Else branch of conditional"),
            ("match", "Pattern matching expression"),
            ("with", "Match arm separator"),
            ("end", "End of block"),
            ("type", "Type definition"),
            ("trait", "Trait definition"),
            ("reactive", "Reactive type definition"),
            ("use", "Import module"),
            ("var", "Mutable variable"),
            ("mvar", "Module-level mutable variable"),
            ("while", "While loop"),
            ("for", "For loop"),
            ("true", "Boolean true"),
            ("false", "Boolean false"),
            ("and", "Logical and"),
            ("or", "Logical or"),
            ("not", "Logical not"),
            ("in", "In expression (for loops)"),
            ("do", "Do expression"),
            ("return", "Return from function"),
            ("break", "Break from loop"),
            ("continue", "Continue to next iteration"),
        ];

        for (kw, doc) in keywords {
            if partial.is_empty() || kw.starts_with(&partial_lower) {
                items.push(CompletionItem {
                    label: kw.to_string(),
                    kind: Some(CompletionItemKind::KEYWORD),
                    detail: Some("keyword".to_string()),
                    documentation: Some(Documentation::String(doc.to_string())),
                    insert_text: Some(kw.to_string()),
                    ..Default::default()
                });
            }
        }

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return items;
        };

        // Add functions
        let mut seen_functions = std::collections::HashSet::new();
        for fn_name in engine.get_functions() {
            // Only show simple names (not module.function format) or match on full name
            // Also strip /signature suffix used for overloaded functions
            let display_name = fn_name.rsplit('.').next().unwrap_or(&fn_name);
            let display_name = display_name.split('/').next().unwrap_or(display_name);

            // Skip duplicates (from overloaded functions)
            if !seen_functions.insert(display_name.to_string()) {
                continue;
            }

            if partial.is_empty() || display_name.to_lowercase().starts_with(&partial_lower) {
                let signature = engine.get_function_signature(&fn_name);
                let doc = engine.get_function_doc(&fn_name);

                // Use short name for label, full name in detail
                let detail = if fn_name.contains('.') {
                    Some(format!("{} ({})", signature.unwrap_or_default(), fn_name))
                } else {
                    signature
                };

                items.push(CompletionItem {
                    label: display_name.to_string(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    detail,
                    documentation: doc.map(|d| Documentation::String(d)),
                    insert_text: Some(display_name.to_string()),
                    ..Default::default()
                });
            }
        }

        // Add types and their constructors
        for type_name in engine.get_types() {
            let display_name = type_name.rsplit('.').next().unwrap_or(&type_name);

            if partial.is_empty() || display_name.to_lowercase().starts_with(&partial_lower) {
                items.push(CompletionItem {
                    label: display_name.to_string(),
                    kind: Some(CompletionItemKind::CLASS),
                    detail: Some(type_name.clone()),
                    insert_text: Some(display_name.to_string()),
                    ..Default::default()
                });
            }

            // Add variant constructors for this type
            for ctor_name in engine.get_type_constructors(&type_name) {
                if partial.is_empty() || ctor_name.to_lowercase().starts_with(&partial_lower) {
                    items.push(CompletionItem {
                        label: ctor_name.clone(),
                        kind: Some(CompletionItemKind::ENUM_MEMBER),
                        detail: Some(format!("constructor of {}", display_name)),
                        insert_text: Some(ctor_name),
                        ..Default::default()
                    });
                }
            }
        }

        // Limit results if too many
        if items.len() > 200 {
            items.truncate(200);
        }

        items
    }

    /// Detect the type of a literal expression
    fn detect_literal_type(expr: &str) -> Option<&'static str> {
        let trimmed = expr.trim();

        // String literal
        if trimmed.starts_with('"') || trimmed.starts_with('\'') {
            return Some("String");
        }

        // List literal - but NOT indexed list literals like [["a","b"]][0][0]
        // Those need to be handled by infer_indexed_list_literal_type instead
        if trimmed.starts_with('[') {
            // Check if this is an indexed list literal by finding the matching bracket
            // and seeing if there's an index operation after it
            let mut depth = 0;
            let mut list_end = None;
            for (i, c) in trimmed.chars().enumerate() {
                match c {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            list_end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(end_idx) = list_end {
                let after_list = &trimmed[end_idx + 1..];
                if after_list.starts_with('[') {
                    // This is an indexed list literal - return None so it falls through
                    // to infer_indexed_list_literal_type
                    return None;
                }
            }

            return Some("List");
        }

        // Map literal
        if trimmed.starts_with("%{") {
            return Some("Map");
        }

        // Set literal
        if trimmed.starts_with("#{") {
            return Some("Set");
        }

        // Numeric literals
        let num_part = trimmed.strip_prefix('-').unwrap_or(trimmed);
        if !num_part.is_empty() && num_part.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            if num_part.contains('.') {
                return Some("Float");
            }
            return Some("Int");
        }

        None
    }

    /// Generate insertText for a function/method with parameter placeholders
    /// e.g., "add" with signature "(Int, Int) -> Int" becomes "add(,)"
    /// For 0-arity functions, returns "name()"
    fn generate_function_insert_text(name: &str, signature: Option<&str>) -> String {
        let arity = signature.map(|sig| Self::count_parameters(sig)).unwrap_or(0);
        if arity == 0 {
            format!("{}()", name)
        } else {
            let commas = ",".repeat(arity.saturating_sub(1));
            format!("{}({})", name, commas)
        }
    }

    /// Count the number of parameters in a function signature
    /// Handles signatures like "(Int, Int) -> Int" or "Int -> Int -> Int"
    fn count_parameters(signature: &str) -> usize {
        // First, strip trait bounds like "Num a => ..."
        let sig = if let Some(pos) = signature.find("=>") {
            signature[pos + 2..].trim()
        } else {
            signature.trim()
        };

        // Check if it's a tuple-style signature "(a, b, c) -> result"
        if sig.starts_with('(') {
            if let Some(paren_end) = sig.find(')') {
                let params = &sig[1..paren_end];
                if params.trim().is_empty() {
                    return 0;
                }
                // Count commas, handling nested types like List[Int]
                let mut count = 1;
                let mut depth = 0;
                for c in params.chars() {
                    match c {
                        '[' | '(' | '{' => depth += 1,
                        ']' | ')' | '}' => depth -= 1,
                        ',' if depth == 0 => count += 1,
                        _ => {}
                    }
                }
                return count;
            }
        }

        // Arrow-style signature "a -> b -> c" (curried)
        // Count arrows, the result is one less than number of parts
        let mut count = 0;
        let mut depth = 0;
        let mut chars = sig.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '[' | '(' | '{' => depth += 1,
                ']' | ')' | '}' => depth -= 1,
                '-' if depth == 0 => {
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        count += 1;
                    }
                }
                _ => {}
            }
        }
        // For "a -> b -> c", count=2, meaning 2 params (a and b), result is c
        count
    }

    /// Generate insertText for a type constructor with named fields
    /// e.g., "Person" with fields ["name: String", "age: Int"] becomes "Person(name: , age: )"
    fn generate_type_insert_text(name: &str, fields: &[String]) -> String {
        if fields.is_empty() {
            name.to_string()
        } else {
            let field_names: Vec<&str> = fields.iter()
                .map(|f| f.split(':').next().unwrap_or("").trim())
                .collect();
            format!("{}({})", name, field_names.iter().map(|f| format!("{}: ", f)).collect::<Vec<_>>().join(", "))
        }
    }

    /// Extract the expression before the dot, handling brackets and parens
    fn extract_receiver_expression(text: &str) -> &str {
        let chars: Vec<char> = text.chars().collect();
        let mut i = chars.len();
        let mut depth = 0;
        let mut in_string = false;
        let mut string_char = '"';

        while i > 0 {
            i -= 1;
            let c = chars[i];

            // Handle string literals (scan backwards through them)
            if in_string {
                if c == string_char {
                    // Check for escape
                    let mut escapes = 0;
                    let mut j = i;
                    while j > 0 && chars[j - 1] == '\\' {
                        escapes += 1;
                        j -= 1;
                    }
                    if escapes % 2 == 0 {
                        // This is the opening quote - include it in the expression
                        in_string = false;
                    }
                }
                continue;
            }

            match c {
                '"' | '\'' => {
                    // Start of string (we're going backwards, so this is the closing quote)
                    in_string = true;
                    string_char = c;
                }
                ')' | ']' | '}' => depth += 1,
                '(' | '[' | '{' => {
                    if depth > 0 {
                        depth -= 1;
                    } else {
                        return &text[i..];
                    }
                }
                _ if depth == 0 => {
                    if !c.is_alphanumeric() && c != '_' && c != '.' {
                        return &text[i + 1..];
                    }
                }
                _ => {}
            }
        }

        text
    }

    /// Get completions for REPL input
    /// Returns a list of completion items as JSON-serializable values
    fn get_repl_completions(&self, engine: &nostos_repl::ReplEngine, text: &str, cursor_pos: usize) -> Vec<serde_json::Value> {
        let mut items = Vec::new();

        // Get the text up to cursor
        let text_to_cursor = if cursor_pos <= text.len() {
            &text[..cursor_pos]
        } else {
            text
        };

        // Check if we're inside a type constructor call like "Person(" or "module.Person("
        // Look for pattern: (Module.)?TypeName( where cursor is after the paren
        if let Some(paren_pos) = text_to_cursor.rfind('(') {
            let before_paren = text_to_cursor[..paren_pos].trim_end();
            let after_paren = &text_to_cursor[paren_pos + 1..];

            // Only trigger if after_paren is empty or starts with whitespace/identifier
            // Don't trigger in middle of function calls with args already present
            let in_field_context = after_paren.is_empty()
                || after_paren.chars().all(|c| c.is_whitespace())
                || (after_paren.trim_start().is_empty());

            if in_field_context {
                // Extract type name (could be "Person" or "module.Person")
                let type_expr_start = before_paren.rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let type_expr = &before_paren[type_expr_start..];

                // Check if it looks like a type (starts with uppercase)
                let base_name = if let Some(dot_idx) = type_expr.rfind('.') {
                    &type_expr[dot_idx + 1..]
                } else {
                    type_expr
                };

                if !base_name.is_empty() && base_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    // This looks like a type constructor - check if we have this type
                    let known_types = engine.get_types();
                    let matching_type = known_types.iter()
                        .find(|t| {
                            // Match "Person" or "module.Person"
                            *t == type_expr || t.ends_with(&format!(".{}", type_expr)) || t.ends_with(&format!(".{}", base_name))
                        });

                    if let Some(full_type_name) = matching_type {
                        let fields = engine.get_type_fields(full_type_name);
                        if !fields.is_empty() {
                            eprintln!("REPL constructor completion: type='{}', fields={:?}", full_type_name, fields);

                            for field_info in &fields {
                                // field_info is "name: type" format
                                let field_name = field_info.split(':').next().unwrap_or(field_info).trim();
                                items.push(serde_json::json!({
                                    "label": format!("{}: ", field_name),
                                    "kind": "field",
                                    "detail": field_info,
                                    "insertText": format!("{}: ", field_name),
                                    "replaceStart": paren_pos + 1,
                                    "replaceEnd": cursor_pos
                                }));
                            }

                            // Return early with constructor field completions
                            if !items.is_empty() {
                                return items;
                            }
                        }
                    }
                }
            }
        }

        // Check if we're completing after a dot
        if let Some(dot_pos) = text_to_cursor.rfind('.') {
            let before_dot = &text_to_cursor[..dot_pos];
            let partial_after = &text_to_cursor[dot_pos + 1..];

            eprintln!("REPL dot completion: before='{}', partial='{}'", before_dot, partial_after);
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_repl_complete.log") {
                    let _ = writeln!(f, "=== REPL DOT COMPLETION ===");
                    let _ = writeln!(f, "text_to_cursor: '{}'", text_to_cursor);
                    let _ = writeln!(f, "before_dot: '{}'", before_dot);
                    let _ = writeln!(f, "partial_after: '{}'", partial_after);
                }
            }

            // Extract the full receiver expression (handles [1,2,3], "hello", etc.)
            let expr = Self::extract_receiver_expression(before_dot);
            eprintln!("REPL receiver expr: '{}'", expr);
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_repl_complete.log") {
                    let _ = writeln!(f, "expr (after extract_receiver_expression): '{}'", expr);
                }
            }

            // Check if it's a module name (uppercase identifier)
            let known_modules = engine.get_known_modules();
            let is_module = known_modules.iter().any(|m| m == expr || m.eq_ignore_ascii_case(expr));

            // First, check if we're inside a lambda and completing a lambda parameter
            // Build local_vars from REPL variable types for lambda inference
            let mut local_vars = std::collections::HashMap::new();
            for var_name in engine.get_variables() {
                if let Some(var_type) = engine.get_variable_type(&var_name) {
                    local_vars.insert(var_name, var_type);
                }
            }

            // Try lambda parameter inference first
            let lambda_type = Self::infer_lambda_param_type(text_to_cursor, before_dot, &local_vars);
            if let Some(ref lt) = lambda_type {
                eprintln!("REPL lambda param type: {}", lt);
            }

            if is_module {
                // Module completion - show functions and types from this module
                let module_name = known_modules.iter()
                    .find(|m| m == &expr || m.eq_ignore_ascii_case(expr))
                    .map(|s| s.as_str())
                    .unwrap_or(expr);

                let partial_lower = partial_after.to_lowercase();

                // Add functions from the module
                for fn_name in engine.get_functions() {
                    if fn_name.starts_with(&format!("{}.", module_name)) {
                        let short_name = fn_name.strip_prefix(&format!("{}.", module_name))
                            .unwrap_or(&fn_name);

                        // Strip arity suffix like "/2" or "/_,_"
                        let display_name = if let Some(pos) = short_name.find('/') {
                            &short_name[..pos]
                        } else {
                            short_name
                        };

                        if display_name.starts_with('_') {
                            continue;
                        }

                        if partial_after.is_empty() || display_name.to_lowercase().starts_with(&partial_lower) {
                            let signature = engine.get_function_signature(&fn_name);
                            let insert_text = Self::generate_function_insert_text(display_name, signature.as_deref());
                            items.push(serde_json::json!({
                                "label": display_name,
                                "kind": "function",
                                "detail": signature,
                                "insertText": insert_text,
                                "replaceStart": dot_pos + 1,
                                "replaceEnd": cursor_pos
                            }));
                        }
                    }
                }

                // Add types from the module
                for type_name in engine.get_types() {
                    if type_name.starts_with(&format!("{}.", module_name)) {
                        let short_name = type_name.strip_prefix(&format!("{}.", module_name))
                            .unwrap_or(&type_name);

                        if short_name.starts_with('_') {
                            continue;
                        }

                        if partial_after.is_empty() || short_name.to_lowercase().starts_with(&partial_lower) {
                            let fields = engine.get_type_fields(&type_name);
                            let insert_text = Self::generate_type_insert_text(short_name, &fields);
                            items.push(serde_json::json!({
                                "label": short_name,
                                "kind": "type",
                                "detail": format!("type {}", type_name),
                                "insertText": insert_text,
                                "replaceStart": dot_pos + 1,
                                "replaceEnd": cursor_pos
                            }));
                        }
                    }
                }
            } else {
                // Try to infer type in order:
                // 0. Use lambda parameter type if we're in a lambda
                // 1. Check if it's a simple identifier that's a REPL variable
                // 2. Check for literal types ([1,2], "hello", etc.)
                // 3. Use engine's expression type inference
                let is_simple_ident = expr.chars().all(|c| c.is_alphanumeric() || c == '_');
                {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_repl_complete.log") {
                        let _ = writeln!(f, "is_simple_ident: {}", is_simple_ident);
                        let _ = writeln!(f, "lambda_type: {:?}", lambda_type);
                    }
                }

                let type_name = lambda_type.or_else(|| {
                    if is_simple_ident {
                        // Check REPL variable type first
                        if let Some(var_type) = engine.get_variable_type(expr) {
                            eprintln!("REPL variable type for '{}': {}", expr, var_type);
                            Some(var_type)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }).or_else(|| {
                    // Check for literal types
                    let literal_type = Self::detect_literal_type(expr);
                    {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_repl_complete.log") {
                            let _ = writeln!(f, "detect_literal_type('{}') = {:?}", expr, literal_type);
                        }
                    }
                    literal_type.map(|s| {
                        eprintln!("REPL detected literal type: {}", s);
                        s.to_string()
                    })
                }).or_else(|| {
                    // Try engine's type inference for complex expressions
                    engine.infer_expression_type(expr, &local_vars)
                });

                if let Some(ref type_name) = type_name {
                    eprintln!("REPL inferred type: {}", type_name);
                    {
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_repl_complete.log") {
                            let _ = writeln!(f, "FINAL type_name: {}", type_name);
                            let methods = nostos_repl::ReplEngine::get_builtin_methods_for_type(type_name);
                            let _ = writeln!(f, "get_builtin_methods_for_type returned {} methods", methods.len());
                            for (m, s, d) in methods.iter().take(3) {
                                let _ = writeln!(f, "  - {} : {} -> {}", m, s, d);
                            }
                        }
                    }
                }

                if let Some(type_name) = type_name {

                    let partial_lower = partial_after.to_lowercase();
                    let mut seen = std::collections::HashSet::new();

                    // Get imported names to filter UFCS methods from non-imported modules
                    let imported_names = engine.get_imported_names();

                    // Add builtin methods
                    for (method_name, signature, doc) in nostos_repl::ReplEngine::get_builtin_methods_for_type(&type_name) {
                        if !seen.insert(method_name.to_string()) {
                            continue;
                        }
                        if partial_after.is_empty() || method_name.to_lowercase().starts_with(&partial_lower) {
                            let insert_text = Self::generate_function_insert_text(method_name, Some(signature));
                            items.push(serde_json::json!({
                                "label": method_name,
                                "kind": "method",
                                "detail": signature,
                                "documentation": doc,
                                "insertText": insert_text,
                                "replaceStart": dot_pos + 1,
                                "replaceEnd": cursor_pos
                            }));
                        }
                    }

                    // Add UFCS methods (only from imported modules)
                    for (method_name, signature, doc) in engine.get_ufcs_methods_for_type(&type_name) {
                        // CRITICAL: Only show UFCS methods if they're imported
                        // This prevents stdlib.html.a, stdlib.html.div, etc. from appearing
                        // when user types "xx". on a string without importing stdlib.html
                        if !imported_names.contains(method_name.as_str()) {
                            continue;
                        }

                        if !seen.insert(method_name.clone()) {
                            continue;
                        }
                        if partial_after.is_empty() || method_name.to_lowercase().starts_with(&partial_lower) {
                            let insert_text = Self::generate_function_insert_text(&method_name, Some(&signature));
                            items.push(serde_json::json!({
                                "label": method_name,
                                "kind": "method",
                                "detail": signature,
                                "documentation": doc,
                                "insertText": insert_text,
                                "replaceStart": dot_pos + 1,
                                "replaceEnd": cursor_pos
                            }));
                        }
                    }

                    // Add trait methods
                    for (method_name, signature, doc) in engine.get_trait_methods_for_type(&type_name) {
                        // Only show trait methods if they're imported
                        if !imported_names.contains(method_name.as_str()) {
                            continue;
                        }

                        if !seen.insert(method_name.clone()) {
                            continue;
                        }
                        if partial_after.is_empty() || method_name.to_lowercase().starts_with(&partial_lower) {
                            let insert_text = Self::generate_function_insert_text(&method_name, Some(&signature));
                            items.push(serde_json::json!({
                                "label": method_name,
                                "kind": "method",
                                "detail": signature,
                                "documentation": doc,
                                "insertText": insert_text,
                                "replaceStart": dot_pos + 1,
                                "replaceEnd": cursor_pos
                            }));
                        }
                    }

                    // Add type fields
                    for field_info in engine.get_type_fields(&type_name) {
                        // field_info is "name: type" format
                        let field_name = field_info.split(':').next().unwrap_or(&field_info).trim();
                        if partial_after.is_empty() || field_name.to_lowercase().starts_with(&partial_lower) {
                            items.push(serde_json::json!({
                                "label": field_name,
                                "kind": "field",
                                "detail": field_info,
                                "insertText": field_name,
                                "replaceStart": dot_pos + 1,
                                "replaceEnd": cursor_pos
                            }));
                        }
                    }
                }
            }
        } else {
            // Identifier completion (no dot)
            let word_start = text_to_cursor
                .rfind(|c: char| !c.is_alphanumeric() && c != '_')
                .map(|i| i + 1)
                .unwrap_or(0);
            let partial = &text_to_cursor[word_start..];
            let partial_lower = partial.to_lowercase();

            eprintln!("REPL identifier completion: partial='{}'", partial);

            // Add keywords
            let keywords = [
                ("if", "Conditional"), ("then", "Then branch"), ("else", "Else branch"),
                ("match", "Pattern matching"), ("type", "Type definition"), ("true", "Boolean"),
                ("false", "Boolean"), ("and", "Logical and"), ("or", "Logical or"),
            ];
            for (kw, doc) in keywords {
                if partial.is_empty() || kw.starts_with(&partial_lower) {
                    items.push(serde_json::json!({
                        "label": kw,
                        "kind": "keyword",
                        "detail": doc,
                        "insertText": kw,
                        "replaceStart": word_start,
                        "replaceEnd": cursor_pos
                    }));
                }
            }

            // Add functions (limit to avoid overwhelming UI)
            // Use a set to avoid duplicate display names
            let mut seen_functions = std::collections::HashSet::new();
            let mut count = 0;

            // Get imported names to know which functions can be used without module prefix
            let imported_names = engine.get_imported_names();

            for fn_name in engine.get_functions() {
                if count >= 50 {
                    break;
                }

                // Get local name (without module prefix) and strip arity suffix
                let local_name = fn_name.rsplit('.').next().unwrap_or(&fn_name);
                let local_name = local_name.split('/').next().unwrap_or(local_name);

                // Get qualified name (with module prefix) without arity suffix
                let qualified_name = fn_name.split('/').next().unwrap_or(&fn_name);

                // Check if this function is imported (can be used without module prefix)
                let is_imported = imported_names.contains(local_name);

                // IMPORTANT: Only show functions that are imported OR when user is typing a qualified path
                // Don't pollute autocomplete with stdlib.html.a, stdlib.json.parse, etc. unless imported
                let is_qualified_completion = partial.contains('.') || qualified_name.starts_with(&partial);

                if !is_imported && !is_qualified_completion {
                    // Skip non-imported functions unless user is explicitly typing qualified path
                    continue;
                }

                // Use local name if imported, otherwise qualified name
                let (display_name, insert_name) = if is_imported {
                    (local_name.to_string(), local_name.to_string())
                } else {
                    (qualified_name.to_string(), qualified_name.to_string())
                };

                // Skip if we've already added this display name
                if !seen_functions.insert(display_name.clone()) {
                    continue;
                }

                if partial.is_empty() || local_name.to_lowercase().starts_with(&partial_lower)
                    || qualified_name.to_lowercase().starts_with(&partial_lower) {
                    let signature = engine.get_function_signature(&fn_name);
                    let insert_text = Self::generate_function_insert_text(&insert_name, signature.as_deref());
                    items.push(serde_json::json!({
                        "label": display_name,
                        "kind": "function",
                        "detail": signature,
                        "insertText": insert_text,
                        "replaceStart": word_start,
                        "replaceEnd": cursor_pos
                    }));
                    count += 1;
                }
            }

            // Add types (avoid duplicates)
            let mut seen_types = std::collections::HashSet::new();
            for type_name in engine.get_types() {
                let local_name = type_name.rsplit('.').next().unwrap_or(&type_name);

                // Check if this type is imported (can be used without module prefix)
                let is_imported = imported_names.contains(local_name);

                // Only show types that are imported OR when user is typing a qualified path
                let is_qualified_completion = partial.contains('.') || type_name.starts_with(&partial);

                if !is_imported && !is_qualified_completion {
                    // Skip non-imported types unless user is explicitly typing qualified path
                    continue;
                }

                // Use local name if imported, otherwise qualified name
                let (display_name, insert_name) = if is_imported {
                    (local_name.to_string(), local_name.to_string())
                } else {
                    (type_name.clone(), type_name.clone())
                };

                // Skip if we've already added this display name
                if !seen_types.insert(display_name.clone()) {
                    continue;
                }

                if partial.is_empty() || local_name.to_lowercase().starts_with(&partial_lower)
                    || type_name.to_lowercase().starts_with(&partial_lower) {
                    let fields = engine.get_type_fields(&type_name);
                    let insert_text = Self::generate_type_insert_text(&insert_name, &fields);
                    items.push(serde_json::json!({
                        "label": display_name,
                        "kind": "type",
                        "detail": type_name,
                        "insertText": insert_text,
                        "replaceStart": word_start,
                        "replaceEnd": cursor_pos
                    }));
                }
            }

            // Add modules
            for module_name in engine.get_known_modules() {
                if partial.is_empty() || module_name.to_lowercase().starts_with(&partial_lower) {
                    items.push(serde_json::json!({
                        "label": &module_name,
                        "kind": "module",
                        "detail": format!("module {}", module_name),
                        "insertText": &module_name,
                        "replaceStart": word_start,
                        "replaceEnd": cursor_pos
                    }));
                }
            }
        }

        // Limit results
        if items.len() > 100 {
            items.truncate(100);
        }

        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_repl_complete.log") {
                let _ = writeln!(f, "RETURNING {} completion items", items.len());
                if !items.is_empty() {
                    let _ = writeln!(f, "First item: {}", serde_json::to_string(&items[0]).unwrap_or_default());
                }
                let _ = writeln!(f, "=== END ===\n");
            }
        }

        items
    }

    /// Convert line/column (0-based) to byte offset in content
    fn line_col_to_byte_offset(content: &str, line: usize, col: usize) -> usize {
        let mut offset = 0;
        for (i, line_content) in content.lines().enumerate() {
            if i == line {
                return offset + col;
            }
            offset += line_content.len() + 1; // +1 for newline
        }
        offset
    }

    /// Convert byte offset to LSP Position (line/column)
    fn byte_offset_to_position(content: &str, byte_offset: usize) -> Position {
        let mut line = 0u32;
        let mut col = 0u32;
        let mut current_offset = 0;

        for ch in content.chars() {
            if current_offset >= byte_offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
            current_offset += ch.len_utf8();
        }

        Position { line, character: col }
    }

    /// Extract binding information from a line if it's a simple binding.
    /// Returns None for function definitions, comments, etc.
    fn extract_binding_from_line(line: &str) -> Option<BindingInfo> {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }

        // Skip function definitions (they have parentheses before =)
        // Pattern: name(...) = or name[...](...) =
        if let Some(eq_pos) = trimmed.find('=') {
            let before_eq = &trimmed[..eq_pos];
            if before_eq.contains('(') {
                return None;
            }
        }

        // Find the '=' in the original line (not trimmed)
        let eq_pos = line.find('=')?;

        // Check this isn't == or != or <= or >=
        if eq_pos > 0 {
            let prev = line.as_bytes().get(eq_pos - 1);
            if prev == Some(&b'!') || prev == Some(&b'<') || prev == Some(&b'>') {
                return None;
            }
        }
        if line.as_bytes().get(eq_pos + 1) == Some(&b'=') {
            return None;
        }

        // Extract the identifier before =
        let before_eq = line[..eq_pos].trim();

        // Should be a simple identifier (alphanumeric + underscore, possibly with type annotation)
        // Handle "name" or "name: Type"
        let name = if let Some(colon_pos) = before_eq.find(':') {
            before_eq[..colon_pos].trim()
        } else {
            before_eq
        };

        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }

        // If there's already a type annotation, don't add a hint
        if before_eq.contains(':') {
            return None;
        }

        // Calculate positions
        let leading_spaces = line.len() - line.trim_start().len();
        let name_end = leading_spaces + name.len();
        let rhs_start = eq_pos + 1;
        // Skip whitespace after =
        let rhs_start = rhs_start + line[rhs_start..].len() - line[rhs_start..].trim_start().len();

        Some(BindingInfo {
            name: name.to_string(),
            name_end,
            rhs_start,
        })
    }

    /// Check if a span position represents a binding (variable assignment).
    /// Returns Some((name, name_end_offset)) if it's a binding, None otherwise.
    fn is_binding_position(content: &str, span_start: usize, _span_end: usize) -> Option<(String, usize)> {
        // Look backwards from span_start to find the beginning of the line
        let line_start = content[..span_start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = content[span_start..].find('\n').map(|i| span_start + i).unwrap_or(content.len());
        let line = &content[line_start..line_end];

        // Parse the line to find binding patterns like "name = value"
        // Skip lines that start with common non-binding patterns
        let trimmed = line.trim();

        // Skip comments
        if trimmed.starts_with('#') {
            return None;
        }

        // Skip function definitions (they have parentheses before =)
        // Pattern: name(...) = or name[...](...) =
        if let Some(eq_pos) = trimmed.find('=') {
            let before_eq = &trimmed[..eq_pos];
            // If there's a '(' before '=' it's likely a function definition
            if before_eq.contains('(') {
                return None;
            }
        }

        // Look for pattern: identifier = value (simple binding)
        // The span should be part of the value side of the binding
        let relative_start = span_start - line_start;

        // Find the '=' in the line
        if let Some(eq_pos) = line.find('=') {
            // Check this isn't == or != or <= or >=
            if eq_pos > 0 && (line.as_bytes().get(eq_pos - 1) == Some(&b'!')
                || line.as_bytes().get(eq_pos - 1) == Some(&b'<')
                || line.as_bytes().get(eq_pos - 1) == Some(&b'>')) {
                return None;
            }
            if line.as_bytes().get(eq_pos + 1) == Some(&b'=') {
                return None;
            }

            // Check if our span is on the value side (after =)
            if relative_start > eq_pos {
                // Extract the identifier before =
                let before_eq = line[..eq_pos].trim();
                // Should be a simple identifier (alphanumeric + underscore)
                if !before_eq.is_empty() && before_eq.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    let name = before_eq.to_string();
                    // The hint should appear right after the identifier name
                    let name_end = line_start + line[..eq_pos].trim_end().len();
                    return Some((name, name_end));
                }
            }
        }

        None
    }

    /// Determine if we should show an inlay hint for this type.
    /// Skip obvious types like simple literals.
    fn should_show_type_hint(ty: &str) -> bool {
        // Show hints for complex types
        // Skip if it's just a basic type that's usually obvious from context
        // For now, show all types - users can configure this in VS Code settings
        // In the future, could skip Int/Float/String/Bool for simple literals
        !ty.is_empty()
    }

    /// Extract the word/identifier at the cursor position
    /// Returns (word, start_index, end_index)
    fn extract_word_at_cursor(line: &str, cursor: usize) -> (String, usize, usize) {
        let chars: Vec<char> = line.chars().collect();
        let cursor = cursor.min(chars.len());

        // Find start of word (including dots for qualified names)
        let mut start = cursor;
        while start > 0 {
            let c = chars[start - 1];
            if c.is_alphanumeric() || c == '_' || c == '.' {
                start -= 1;
            } else {
                break;
            }
        }

        // Find end of word
        let mut end = cursor;
        while end < chars.len() {
            let c = chars[end];
            if c.is_alphanumeric() || c == '_' {
                end += 1;
            } else {
                break;
            }
        }

        let word: String = chars[start..end].iter().collect();
        (word, start, end)
    }

    /// Get hover information for a word
    fn get_hover_info(
        &self,
        engine: &nostos_repl::ReplEngine,
        word: &str,
        local_vars: &std::collections::HashMap<String, String>,
        _line: &str,
    ) -> Option<String> {
        // Check if it's a local variable
        if let Some(ty) = local_vars.get(word) {
            return Some(format!("```nostos\n{}: {}\n```\n*(local variable)*", word, ty));
        }

        // Check if it's a function
        if let Some(sig) = engine.get_function_signature(word) {
            let mut info = format!("```nostos\n{}: {}\n```", word, sig);
            if let Some(doc) = engine.get_function_doc(word) {
                info.push_str(&format!("\n\n{}", doc));
            }
            return Some(info);
        }

        // Try with module prefix stripped
        let simple_name = word.rsplit('.').next().unwrap_or(word);
        if simple_name != word {
            if let Some(sig) = engine.get_function_signature(simple_name) {
                let mut info = format!("```nostos\n{}: {}\n```", simple_name, sig);
                if let Some(doc) = engine.get_function_doc(simple_name) {
                    info.push_str(&format!("\n\n{}", doc));
                }
                return Some(info);
            }
        }

        // Check if it's a builtin
        if let Some(sig) = nostos_compiler::Compiler::get_builtin_signature(word) {
            let mut info = format!("```nostos\n{}: {}\n```\n*(builtin)*", word, sig);
            if let Some(doc) = nostos_compiler::Compiler::get_builtin_doc(word) {
                info.push_str(&format!("\n\n{}", doc));
            }
            return Some(info);
        }

        // Check if it's a type
        for type_name in engine.get_types() {
            let short = type_name.rsplit('.').next().unwrap_or(&type_name);
            if short == word || type_name == word {
                // Get type info
                let fields = engine.get_type_fields(&type_name);
                let constructors = engine.get_type_constructors(&type_name);

                let mut info = format!("```nostos\ntype {}\n```", short);

                if !fields.is_empty() {
                    info.push_str("\n\n**Fields:**\n");
                    for field in fields {
                        info.push_str(&format!("- `{}`\n", field));
                    }
                }

                if !constructors.is_empty() {
                    info.push_str("\n\n**Constructors:**\n");
                    for ctor in constructors {
                        info.push_str(&format!("- `{}`\n", ctor));
                    }
                }

                return Some(info);
            }
        }

        // Try to infer expression type
        if let Some(ty) = engine.infer_expression_type(word, local_vars) {
            return Some(format!("```nostos\n{}: {}\n```\n*(inferred)*", word, ty));
        }

        None
    }

    /// Find the line number where a function is defined
    fn find_definition_line(content: &str, fn_name: &str) -> Option<u32> {
        let simple_name = fn_name.rsplit('.').next().unwrap_or(fn_name);
        let pattern = format!("{}(", simple_name);
        let pattern2 = format!("{} (", simple_name);

        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            // Look for function definition: name(...) = or name (...) =
            if (trimmed.starts_with(&pattern) || trimmed.starts_with(&pattern2))
                && trimmed.contains('=')
            {
                return Some(line_num as u32);
            }
        }
        None
    }

    /// Extract function name and active parameter index from a partial call expression
    /// e.g., "foo(1, 2," -> ("foo", 2)
    fn extract_function_call_context(prefix: &str) -> (String, usize) {
        // Find the last unmatched opening paren
        let mut paren_depth = 0;
        let mut last_open_paren = None;

        for (i, c) in prefix.char_indices() {
            match c {
                '(' => {
                    if paren_depth == 0 {
                        last_open_paren = Some(i);
                    }
                    paren_depth += 1;
                }
                ')' => paren_depth = (paren_depth - 1).max(0),
                _ => {}
            }
        }

        let Some(paren_pos) = last_open_paren else {
            return (String::new(), 0);
        };

        // Get function name before the paren
        let before_paren = prefix[..paren_pos].trim();
        let fn_name = before_paren
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
            .last()
            .unwrap_or("")
            .to_string();

        // Count commas to determine active parameter
        let after_paren = &prefix[paren_pos + 1..];
        let mut comma_count = 0;
        let mut depth = 0;

        for c in after_paren.chars() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = (depth - 1).max(0),
                ',' if depth == 0 => comma_count += 1,
                _ => {}
            }
        }

        (fn_name, comma_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_parameters_tuple_style() {
        // Empty params
        assert_eq!(NostosLanguageServer::count_parameters("() -> Int"), 0);

        // Single param
        assert_eq!(NostosLanguageServer::count_parameters("(Int) -> Int"), 1);

        // Two params
        assert_eq!(NostosLanguageServer::count_parameters("(Int, Int) -> Int"), 2);

        // Three params
        assert_eq!(NostosLanguageServer::count_parameters("(Int, String, Bool) -> Int"), 3);

        // With nested types
        assert_eq!(NostosLanguageServer::count_parameters("(List[Int], Map[String, Int]) -> Bool"), 2);

        // With trait bounds
        assert_eq!(NostosLanguageServer::count_parameters("Num a => (a, a) -> a"), 2);
    }

    #[test]
    fn test_count_parameters_arrow_style() {
        // a -> b (one param)
        assert_eq!(NostosLanguageServer::count_parameters("Int -> Int"), 1);

        // a -> b -> c (two params)
        assert_eq!(NostosLanguageServer::count_parameters("Int -> Int -> Int"), 2);

        // With nested types
        assert_eq!(NostosLanguageServer::count_parameters("List[Int] -> Map[String, Int] -> Bool"), 2);
    }

    #[test]
    fn test_generate_function_insert_text() {
        // Zero arity
        assert_eq!(
            NostosLanguageServer::generate_function_insert_text("foo", Some("() -> Int")),
            "foo()"
        );

        // One param
        assert_eq!(
            NostosLanguageServer::generate_function_insert_text("bar", Some("(Int) -> Int")),
            "bar()"
        );

        // Two params
        assert_eq!(
            NostosLanguageServer::generate_function_insert_text("add", Some("(Int, Int) -> Int")),
            "add(,)"
        );

        // Three params
        assert_eq!(
            NostosLanguageServer::generate_function_insert_text("func", Some("(Int, String, Bool) -> Int")),
            "func(,,)"
        );

        // No signature (defaults to 0 arity)
        assert_eq!(
            NostosLanguageServer::generate_function_insert_text("unknown", None),
            "unknown()"
        );
    }

    #[test]
    fn test_generate_type_insert_text() {
        // No fields
        assert_eq!(
            NostosLanguageServer::generate_type_insert_text("Unit", &[]),
            "Unit"
        );

        // One field
        assert_eq!(
            NostosLanguageServer::generate_type_insert_text("Wrapper", &["value: Int".to_string()]),
            "Wrapper(value: )"
        );

        // Two fields
        assert_eq!(
            NostosLanguageServer::generate_type_insert_text("Person", &["name: String".to_string(), "age: Int".to_string()]),
            "Person(name: , age: )"
        );

        // Three fields
        assert_eq!(
            NostosLanguageServer::generate_type_insert_text("Point3D", &["x: Float".to_string(), "y: Float".to_string(), "z: Float".to_string()]),
            "Point3D(x: , y: , z: )"
        );
    }

    #[test]
    fn test_strip_arity_suffix_from_function_name() {
        // Verify that the helper strips arity suffix properly
        let fn_name = "module.addff/_,_";
        let short_name = fn_name.strip_prefix("module.").unwrap();

        // Strip arity suffix
        let display_name = if let Some(pos) = short_name.find('/') {
            &short_name[..pos]
        } else {
            short_name
        };

        assert_eq!(display_name, "addff", "Should strip arity suffix /_,_");

        // Test with simple numeric arity
        let fn_name2 = "module.foo/2";
        let short_name2 = fn_name2.strip_prefix("module.").unwrap();
        let display_name2 = if let Some(pos) = short_name2.find('/') {
            &short_name2[..pos]
        } else {
            short_name2
        };
        assert_eq!(display_name2, "foo", "Should strip arity suffix /2");

        // Test without arity suffix
        let fn_name3 = "module.bar";
        let short_name3 = fn_name3.strip_prefix("module.").unwrap();
        let display_name3 = if let Some(pos) = short_name3.find('/') {
            &short_name3[..pos]
        } else {
            short_name3
        };
        assert_eq!(display_name3, "bar", "Should keep name as-is without arity suffix");
    }

    #[test]
    fn test_repl_module_type_completion_detection() {
        // Test that module.TypeName pattern is detected correctly
        // The actual completion logic requires a running engine, but we can test the detection pattern

        let type_expr = "module.Person";

        // Check if it looks like a type (base name starts with uppercase)
        let base_name = if let Some(dot_idx) = type_expr.rfind('.') {
            &type_expr[dot_idx + 1..]
        } else {
            type_expr
        };

        assert_eq!(base_name, "Person");
        assert!(base_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
            "Type names should start with uppercase");

        // Test plain type
        let type_expr2 = "Person";
        let base_name2 = if let Some(dot_idx) = type_expr2.rfind('.') {
            &type_expr2[dot_idx + 1..]
        } else {
            type_expr2
        };
        assert_eq!(base_name2, "Person");
        assert!(base_name2.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));

        // Test lowercase function (should not be treated as type)
        let func_expr = "module.calculate";
        let base_name3 = if let Some(dot_idx) = func_expr.rfind('.') {
            &func_expr[dot_idx + 1..]
        } else {
            func_expr
        };
        assert_eq!(base_name3, "calculate");
        assert!(!base_name3.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
            "Function names start with lowercase, should not be treated as type");
    }

    #[test]
    fn test_constructor_field_context_detection() {
        // Test the logic for detecting when we're inside a constructor call

        // Case 1: Just opened paren "Person("
        let text = "Person(";
        let paren_pos = text.rfind('(').unwrap();
        let after_paren = &text[paren_pos + 1..];
        let in_field_context = after_paren.is_empty()
            || after_paren.chars().all(|c| c.is_whitespace())
            || after_paren.trim_start().is_empty();
        assert!(in_field_context, "Just after '(' should be field context");

        // Case 2: Space after paren "Person( "
        let text2 = "Person( ";
        let paren_pos2 = text2.rfind('(').unwrap();
        let after_paren2 = &text2[paren_pos2 + 1..];
        let in_field_context2 = after_paren2.is_empty()
            || after_paren2.chars().all(|c| c.is_whitespace())
            || after_paren2.trim_start().is_empty();
        assert!(in_field_context2, "After '( ' should be field context");

        // Case 3: With module prefix "test_types.Person("
        let text3 = "test_types.Person(";
        let paren_pos3 = text3.rfind('(').unwrap();
        let before_paren3 = text3[..paren_pos3].trim_end();
        let type_expr_start = before_paren3.rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
            .map(|i| i + 1)
            .unwrap_or(0);
        let type_expr = &before_paren3[type_expr_start..];
        assert_eq!(type_expr, "test_types.Person", "Should extract full type expression");
    }

    #[test]
    fn test_infer_indexed_list_literal_type() {
        // Single index on nested list: [["a","b"]][0] -> List[String]
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type(r#"[["a","b"]][0]"#),
            Some("List[String]".to_string())
        );

        // Double index on nested list: [["a","b"]][0][0] -> String
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type(r#"[["a","b"]][0][0]"#),
            Some("String".to_string())
        );

        // Single index on simple list: [1,2,3][0] -> Int
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type("[1,2,3][0]"),
            Some("Int".to_string())
        );

        // Double index on doubly nested list: [[[1,2]]][0][0] -> List[Int]
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type("[[[1,2]]][0][0]"),
            Some("List[Int]".to_string())
        );

        // Triple index on doubly nested list: [[[1,2]]][0][0][0] -> Int
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type("[[[1,2]]][0][0][0]"),
            Some("Int".to_string())
        );

        // Plain list literal without index should return None (not handled here)
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type("[1,2,3]"),
            None
        );

        // Variable access (not a list literal) should return None
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type("x[0]"),
            None
        );

        // Empty list with index
        assert_eq!(
            NostosLanguageServer::infer_indexed_list_literal_type("[][0]"),
            None  // Can't determine element type of empty list
        );
    }

    /// Test that function call return types are correctly inferred for autocomplete
    /// When user has `x = good.addff(3, 2)` and types `x.`, should get Int methods
    #[test]
    fn test_function_call_return_type_in_bindings() {
        use std::fs;
        use nostos_repl::{ReplEngine, ReplConfig};

        let temp_dir = tempfile::tempdir().unwrap();
        fs::write(temp_dir.path().join("nostos.toml"), "[project]\nname = \"test\"").unwrap();

        // Create good.nos module with typed functions
        let good_content = "pub addff(a: Int, b: Int) -> Int = a + b\npub multiply(x: Int, y: Int) -> Int = x * y\n";
        fs::write(temp_dir.path().join("good.nos"), good_content).unwrap();

        // Create main.nos - simulates file where user will type x.
        let main_content = r#"use good.*
main() = {
    x = good.addff(3, 2)
    y = good.multiply(2, 3)
    x
}
"#;
        fs::write(temp_dir.path().join("main.nos"), main_content).unwrap();

        // Create engine and load project
        let config = ReplConfig { enable_jit: false, num_threads: 1 };
        let mut engine = ReplEngine::new(config);
        engine.load_stdlib().ok();
        engine.load_directory(temp_dir.path().to_str().unwrap()).unwrap();

        // Test extract_local_bindings - simulating what happens when user is on line 4
        // Line 0: use good.*
        // Line 1: main() = {
        // Line 2:     x = good.addff(3, 2)
        // Line 3:     y = good.multiply(2, 3)
        // Line 4:     x   <- user types x. here
        let bindings = NostosLanguageServer::extract_local_bindings(main_content, 4, Some(&engine));

        println!("Extracted bindings: {:?}", bindings);

        // x should be inferred as Int from good.addff return type
        assert!(bindings.contains_key("x"), "Should have binding for x, got: {:?}", bindings);
        let x_type = bindings.get("x").unwrap();
        println!("x type: {}", x_type);
        // Type could be "Int" or "a" (polymorphic) depending on inference
        assert!(x_type == "Int" || x_type == "a",
            "x should be Int or polymorphic 'a', got: {}", x_type);

        // y should also be inferred
        assert!(bindings.contains_key("y"), "Should have binding for y, got: {:?}", bindings);
        let y_type = bindings.get("y").unwrap();
        println!("y type: {}", y_type);
        assert!(y_type == "Int" || y_type == "a",
            "y should be Int or polymorphic 'a', got: {}", y_type);
    }

    /// Test the full autocomplete flow when user types `x.` on a new line
    #[test]
    fn test_autocomplete_after_function_call_assignment() {
        use std::fs;
        use nostos_repl::{ReplEngine, ReplConfig};

        let temp_dir = tempfile::tempdir().unwrap();
        fs::write(temp_dir.path().join("nostos.toml"), "[project]\nname = \"test\"").unwrap();

        // Create good.nos module with typed functions
        let good_content = "pub addff(a: Int, b: Int) -> Int = a + b\n";
        fs::write(temp_dir.path().join("good.nos"), good_content).unwrap();

        // Create main.nos - user is on line 4 typing "x."
        // Line 0: use good.*
        // Line 1: main() = {
        // Line 2:     x = good.addff(3, 2)
        // Line 3:     x.   <- cursor here
        // Line 4: }
        let main_content = r#"use good.*
main() = {
    x = good.addff(3, 2)
    x.
}
"#;
        fs::write(temp_dir.path().join("main.nos"), main_content).unwrap();

        // Create engine and load project
        let config = ReplConfig { enable_jit: false, num_threads: 1 };
        let mut engine = ReplEngine::new(config);
        engine.load_stdlib().ok();
        engine.load_directory(temp_dir.path().to_str().unwrap()).unwrap();

        // Simulate the autocomplete flow
        // 1. Extract local bindings up to line 3 (where x. is typed)
        let bindings = NostosLanguageServer::extract_local_bindings(main_content, 3, Some(&engine));
        println!("Bindings: {:?}", bindings);
        assert!(bindings.contains_key("x"), "Should have x binding");
        assert_eq!(bindings.get("x").unwrap(), "Int", "x should be Int");

        // 2. Simulate what get_dot_completions does
        // before_dot for line "    x." would be "    x" after trimming prefix
        let before_dot = "x";

        // Extract identifier (simulating line 2957-2960)
        let identifier = before_dot
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .last()
            .unwrap_or("");
        println!("Identifier: '{}'", identifier);
        assert_eq!(identifier, "x", "Should extract 'x' as identifier");

        // Look up in bindings
        let inferred_type = bindings.get(identifier);
        println!("Inferred type for '{}': {:?}", identifier, inferred_type);
        assert!(inferred_type.is_some(), "Should find type for x");
        assert_eq!(inferred_type.unwrap(), "Int", "x should be Int");

        // Verify methods would be returned for Int
        let methods = ReplEngine::get_builtin_methods_for_type("Int");
        println!("Int methods count: {}", methods.len());
        assert!(methods.len() > 0, "Int should have methods");

        // Check some expected methods exist
        let method_names: Vec<&str> = methods.iter().map(|(n, _, _)| *n).collect();
        println!("Int methods: {:?}", method_names);
        assert!(method_names.contains(&"abs"), "Int should have abs method");
    }

    /// Test autocomplete on function call result directly: good.addff(3,2).
    /// Should show Int methods since addff returns Int
    #[test]
    fn test_autocomplete_on_function_call_result() {
        use std::fs;
        use nostos_repl::{ReplEngine, ReplConfig};

        let temp_dir = tempfile::tempdir().unwrap();
        fs::write(temp_dir.path().join("nostos.toml"), "[project]\nname = \"test\"").unwrap();

        // Create good.nos module with typed function
        let good_content = "pub addff(a: Int, b: Int) -> Int = a + b\n";
        fs::write(temp_dir.path().join("good.nos"), good_content).unwrap();

        // Create main.nos - user types good.addff(3,2). on line 2
        let main_content = r#"use good.*
main() = {
    good.addff(3,2).
}
"#;
        fs::write(temp_dir.path().join("main.nos"), main_content).unwrap();

        // Create engine and load project
        let config = ReplConfig { enable_jit: false, num_threads: 1 };
        let mut engine = ReplEngine::new(config);
        engine.load_stdlib().ok();
        engine.load_directory(temp_dir.path().to_str().unwrap()).unwrap();

        // The expression before dot is "good.addff(3,2)"
        let expr = "good.addff(3,2)";

        // Test infer_rhs_type directly - this should return Int
        let bindings = std::collections::HashMap::new();
        let inferred = NostosLanguageServer::infer_rhs_type(expr, Some(&engine), &bindings);
        println!("Inferred type for '{}': {:?}", expr, inferred);
        assert!(inferred.is_some(), "Should infer type for function call");
        assert_eq!(inferred.unwrap(), "Int", "good.addff(3,2) should return Int");

        // Verify methods would be returned for Int
        let methods = ReplEngine::get_builtin_methods_for_type("Int");
        assert!(methods.len() > 0, "Int should have methods for autocomplete");
    }

    /// Test with untyped (polymorphic) function - should infer from arguments
    #[test]
    fn test_autocomplete_on_polymorphic_function_call() {
        use std::fs;
        use nostos_repl::{ReplEngine, ReplConfig};

        let temp_dir = tempfile::tempdir().unwrap();
        fs::write(temp_dir.path().join("nostos.toml"), "[project]\nname = \"test\"").unwrap();

        // Create good.nos with UNTYPED function (like user's actual file)
        let good_content = "pub addff(a, b) = a + b\n";
        fs::write(temp_dir.path().join("good.nos"), good_content).unwrap();

        let main_content = "main() = good.addff(3,2)\n";
        fs::write(temp_dir.path().join("main.nos"), main_content).unwrap();

        let config = ReplConfig { enable_jit: false, num_threads: 1 };
        let mut engine = ReplEngine::new(config);
        engine.load_stdlib().ok();
        engine.load_directory(temp_dir.path().to_str().unwrap()).unwrap();

        // Check signature - should be polymorphic "Num a => a -> a -> a"
        let sig = engine.get_function_signature("good.addff");
        println!("Signature for untyped good.addff: {:?}", sig);
        assert!(sig.is_some(), "Should have signature");

        // Test infer_rhs_type - with Int arguments, should infer Int
        let expr = "good.addff(3,2)";
        let bindings = std::collections::HashMap::new();
        let inferred = NostosLanguageServer::infer_rhs_type(expr, Some(&engine), &bindings);
        println!("Inferred type for '{}': {:?}", expr, inferred);
        assert!(inferred.is_some(), "Should infer type for polymorphic function call");
        // Should be Int (inferred from first argument 3)
        assert_eq!(inferred.unwrap(), "Int", "Should infer Int from arguments");
    }
}
