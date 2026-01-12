use std::path::PathBuf;
use std::sync::Mutex;

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use nostos_repl::{ReplEngine, ReplConfig};

pub struct NostosLanguageServer {
    client: Client,
    engine: Mutex<Option<ReplEngine>>,
    /// Map from URI to document content (for unsaved changes)
    documents: DashMap<Url, String>,
    /// Map from URI to error diagnostics from recompile_file (to preserve when adding stale warnings)
    file_errors: DashMap<Url, Vec<Diagnostic>>,
    /// Root path of the workspace
    root_path: Mutex<Option<PathBuf>>,
}

impl NostosLanguageServer {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            engine: Mutex::new(None),
            documents: DashMap::new(),
            file_errors: DashMap::new(),
            root_path: Mutex::new(None),
        }
    }

    /// Initialize the ReplEngine with the workspace
    fn init_engine(&self, root_path: &PathBuf) {
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
        if let Err(e) = engine.load_directory(root_path.to_str().unwrap_or(".")) {
            eprintln!("Warning: Failed to load directory: {}", e);
        }

        *self.engine.lock().unwrap() = Some(engine);
        *self.root_path.lock().unwrap() = Some(root_path.clone());
    }

    /// Publish diagnostics for a specific file
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

    /// Recompile a file and publish updated diagnostics
    async fn recompile_file(&self, uri: &Url, content: &str) {
        let file_path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return,
        };

        let file_path_str = file_path.to_string_lossy().to_string();
        eprintln!("Recompiling file: {}", file_path_str);

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
                return;
            };

            engine.recompile_module_with_content(&module_name, content)
        };

        eprintln!("Compile result for {}: {:?}", module_name, result);

        // Debug: print all compile statuses
        {
            let engine_guard = self.engine.lock().unwrap();
            if let Some(engine) = engine_guard.as_ref() {
                eprintln!("=== All compile statuses ===");
                for (name, status) in engine.get_all_compile_status() {
                    eprintln!("  {} -> {}", name, status);
                }
                eprintln!("=== End statuses ===");
            }
        }

        // Publish diagnostics based on result AND actual compile status
        // (recompile_module_with_content might return Ok("No changes detected") even if
        // the function has an error from a previous compilation)
        let error_diagnostics = {
            let mut errors = vec![];

            // First check the return value for direct errors
            if let Err(e) = &result {
                let (line, message) = Self::parse_error_location(e, Some(content));
                eprintln!("Publishing error for {} at line {}: {}", module_name, line, message);
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
                                    eprintln!("Publishing status error for {} at line {}: {}", fn_name, line, message);
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
        }

        // Update diagnostics for all files (stale warnings, errors for closed files)
        self.publish_all_file_diagnostics().await;
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

    /// Publish diagnostics for all known files in the workspace
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

        // Get errors from engine status (works for both open and closed files)
        // This is more reliable than file_errors since URIs might not match exactly
        let mut diagnostics = {
            let engine_guard = self.engine.lock().unwrap();
            if let Some(engine) = engine_guard.as_ref() {
                let module_name = std::path::Path::new(file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                let mut errors = Vec::new();
                eprintln!("DEBUG publish_file_diagnostics_filtered: checking module '{}' (file: {})", module_name, file_path);
                for (fn_name, status_str) in engine.get_all_compile_status() {
                    if fn_name.starts_with(&format!("{}.", module_name)) || fn_name == module_name {
                        eprintln!("DEBUG: {} -> {}", fn_name, status_str);
                        if status_str.starts_with("Error:") {
                            let (line, message) = Self::parse_error_location(&status_str, doc_content.as_deref());
                            eprintln!("DEBUG: Adding error diagnostic at line {}: {}", line, message);
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

        eprintln!("Publishing {} diagnostics for {} (is_open={})", diagnostics.len(), file_path, is_open);
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
}

// Increment BUILD_ID manually when making changes to easily verify binary is updated
const LSP_VERSION: &str = env!("CARGO_PKG_VERSION");
const LSP_BUILD_ID: &str = "2026-01-12-l";

#[tower_lsp::async_trait]
impl LanguageServer for NostosLanguageServer {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        eprintln!("Nostos LSP v{} (build: {})", LSP_VERSION, LSP_BUILD_ID);
        eprintln!("Initializing Nostos LSP...");

        // Get workspace root
        if let Some(root_uri) = params.root_uri {
            if let Ok(path) = root_uri.to_file_path() {
                eprintln!("Workspace root: {:?}", path);
                self.init_engine(&path);
            }
        } else if let Some(folders) = params.workspace_folders {
            if let Some(folder) = folders.first() {
                if let Ok(path) = folder.uri.to_file_path() {
                    eprintln!("Workspace folder: {:?}", path);
                    self.init_engine(&path);
                }
            }
        }

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                // Sync full document on change
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                // We'll add more capabilities later:
                // - completion_provider
                // - hover_provider
                // - definition_provider
                // - references_provider
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "nostos-lsp".to_string(),
                version: Some("0.1.0".to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        eprintln!("Nostos LSP initialized!");

        // Publish initial diagnostics for all files
        self.publish_all_file_diagnostics().await;
    }

    async fn shutdown(&self) -> Result<()> {
        eprintln!("Shutting down Nostos LSP...");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        eprintln!("Opened: {}", params.text_document.uri);

        let uri = params.text_document.uri;
        let content = params.text_document.text;

        // Debug: show what content VS Code sent
        eprintln!("DEBUG did_open content:\n{}", content);

        // Store document content
        self.documents.insert(uri.clone(), content.clone());

        // Compile and publish diagnostics
        self.recompile_file(&uri, &content).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;

        // Get the full content (we're using FULL sync)
        if let Some(change) = params.content_changes.into_iter().last() {
            let content = change.text;

            // Update stored content
            self.documents.insert(uri.clone(), content.clone());

            // Recompile and publish diagnostics
            self.recompile_file(&uri, &content).await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        eprintln!("Saved: {}", params.text_document.uri);

        // On save, we might want to do a full recompile
        // For now, the did_change handler keeps things up to date
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        eprintln!("Closed: {}", params.text_document.uri);

        // Remove from our document cache
        self.documents.remove(&params.text_document.uri);
    }
}
