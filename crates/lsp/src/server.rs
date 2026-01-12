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
const LSP_BUILD_ID: &str = "2026-01-12-u";

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
                // Autocomplete support
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                // We'll add more capabilities later:
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

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        eprintln!("Completion request at {:?}", position);

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
            self.get_dot_completions(before_dot, &local_vars)
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
}

impl NostosLanguageServer {
    /// Extract local variable bindings from document content up to a certain line
    /// Returns a map of variable name -> inferred type (e.g., "y" -> "Int")
    fn extract_local_bindings(content: &str, up_to_line: usize, engine: Option<&nostos_repl::ReplEngine>) -> std::collections::HashMap<String, String> {
        let mut bindings = std::collections::HashMap::new();

        for (line_num, line) in content.lines().enumerate() {
            if line_num >= up_to_line {
                break;
            }

            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Look for simple bindings: "x = expr" or "x = Module.func(...)"
            if let Some(eq_pos) = trimmed.find('=') {
                // Make sure it's not == or other operators
                let before_eq = trimmed[..eq_pos].trim();
                let after_eq_start = eq_pos + 1;
                if after_eq_start < trimmed.len() && !trimmed[after_eq_start..].starts_with('=') {
                    let after_eq = trimmed[after_eq_start..].trim();

                    // Check if before_eq is a simple identifier (variable name)
                    if !before_eq.is_empty()
                        && before_eq.chars().next().map_or(false, |c| c.is_lowercase())
                        && before_eq.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        // Try to infer the type from the RHS
                        let inferred_type = Self::infer_rhs_type(after_eq, engine);
                        if let Some(ty) = inferred_type {
                            eprintln!("Extracted binding: {} = {} (type: {})", before_eq, after_eq, ty);
                            bindings.insert(before_eq.to_string(), ty);
                        }
                    }
                }
            }
        }

        bindings
    }

    /// Infer the type of an expression on the right-hand side of a binding
    fn infer_rhs_type(expr: &str, engine: Option<&nostos_repl::ReplEngine>) -> Option<String> {
        let trimmed = expr.trim();

        // Literals
        if trimmed.starts_with('[') {
            return Some("List".to_string());
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
            if let Some(engine) = engine {
                // Try to get the return type of the function
                if let Some(sig) = engine.get_function_signature(func_part) {
                    // Parse return type from signature like "(Int, Int) -> Int"
                    if let Some(arrow_pos) = sig.rfind("->") {
                        let ret_type = sig[arrow_pos + 2..].trim();
                        eprintln!("Inferred type from func call {}: {}", func_part, ret_type);
                        return Some(ret_type.to_string());
                    }
                }
            }
        }

        None
    }

    /// Get completions after a dot (module functions or UFCS methods)
    fn get_dot_completions(&self, before_dot: &str, local_vars: &std::collections::HashMap<String, String>) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return items;
        };

        // Extract the identifier before the dot
        let identifier = before_dot
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .last()
            .unwrap_or("");

        eprintln!("Identifier before dot: '{}'", identifier);

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

        if known_modules.contains(&potential_module) || known_modules.contains(&identifier.to_string()) {
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

            // First check if it's a local variable we know about
            // Extract just the identifier (handle cases like "    yy" -> "yy")
            let var_name = before_dot.split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|s| !s.is_empty())
                .last()
                .unwrap_or(before_dot);

            let inferred_type = if let Some(var_type) = local_vars.get(var_name) {
                eprintln!("Found local var '{}' with type: {}", var_name, var_type);
                Some(var_type.clone())
            } else {
                // Try to infer the type of the expression from the engine
                engine.infer_expression_type(before_dot)
            };
            eprintln!("Inferred type: {:?}", inferred_type);

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

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return items;
        };

        let partial_lower = partial.to_lowercase();

        // Add functions
        for fn_name in engine.get_functions() {
            // Only show simple names (not module.function format) or match on full name
            let display_name = fn_name.rsplit('.').next().unwrap_or(&fn_name);

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

        // Add types
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
        }

        // Limit results if too many
        if items.len() > 100 {
            items.truncate(100);
        }

        items
    }
}
