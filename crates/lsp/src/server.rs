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
        } else if engine.get_prelude_imports_count() == 0 {
            eprintln!("Warning: Stdlib loaded but 0 prelude imports registered - stdlib may not have been found");
        }

        // Load the project directory
        if let Err(e) = engine.load_directory(root_path.to_str().unwrap_or(".")) {
            eprintln!("Warning: Failed to load directory: {}", e);
        }

        *self.engine.lock().unwrap() = Some(engine);
        *self.root_path.lock().unwrap() = Some(root_path.clone());
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
                eprintln!("LSP DEBUG: Raw error string: {}", e);
                let (line, message) = Self::parse_error_location(e, Some(content));
                eprintln!("LSP DEBUG: Publishing error for {} at line {} (0-based): {}", module_name, line, message);
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

        // Note: Don't call publish_all_file_diagnostics here. It would publish
        // empty lists for open files (to avoid stale errors) which could
        // overwrite the real error we just published if the client processes
        // notifications in order.
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
const LSP_BUILD_ID: &str = "2026-01-13-i-line-fix";

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

        // Don't publish diagnostics here - each file gets proper diagnostics
        // when opened via did_open. Publishing here would push stale line numbers
        // from compile_status before files are opened.
    }

    async fn shutdown(&self) -> Result<()> {
        eprintln!("Shutting down Nostos LSP...");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        eprintln!("Opened: {}", params.text_document.uri);

        let uri = params.text_document.uri;
        let content = params.text_document.text;

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

            self.get_dot_completions(before_dot, &local_vars, lambda_type.as_deref())
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
                            let inferred = Self::infer_rhs_type(after_eq, engine, &bindings);
                            if let Some(ref ty) = inferred {
                                eprintln!("Extracted binding: {} = {} (inferred type: {})", var_name, after_eq, ty);
                            }
                            inferred
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
    fn infer_rhs_type(expr: &str, engine: Option<&nostos_repl::ReplEngine>, current_bindings: &std::collections::HashMap<String, String>) -> Option<String> {
        let trimmed = expr.trim();

        // Check for index expressions like g2[0][0] - use current bindings to resolve
        if trimmed.contains('[') && !trimmed.starts_with('[') {
            // This looks like an index expression (not a list literal)
            if let Some(inferred) = Self::infer_index_expr_type(trimmed, current_bindings) {
                eprintln!("Inferred index expression type: {} -> {}", trimmed, inferred);
                return Some(inferred);
            }
        }

        // List literals - analyze element type recursively
        if trimmed.starts_with('[') {
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

    /// Infer the type of a list literal, handling nested lists
    /// e.g., [[0,1]] -> List[List[Int]], [1,2,3] -> List[Int]
    fn infer_list_type(expr: &str) -> Option<String> {
        let trimmed = expr.trim();

        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return None;
        }

        let inner = &trimmed[1..trimmed.len() - 1].trim();

        if inner.is_empty() {
            return Some("List".to_string());
        }

        // Find the first element (handle nested brackets)
        let first_elem = Self::extract_first_list_element(inner)?;
        let first_trimmed = first_elem.trim();

        eprintln!("infer_list_type: inner='{}', first_elem='{}'", inner, first_trimmed);

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
            let receiver_before_dot = format!("{}", receiver_var);

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

    /// Infer type of an index expression like g2[0] or g2[0][0]
    /// If g2 has type List[List[String]], then:
    ///   g2[0] -> List[String]
    ///   g2[0][0] -> String
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
    fn get_dot_completions(&self, before_dot: &str, local_vars: &std::collections::HashMap<String, String>, lambda_param_type: Option<&str>) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return items;
        };

        // If we have a lambda parameter type, use that directly
        if let Some(param_type) = lambda_param_type {
            eprintln!("Using lambda param type: {}", param_type);
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

            // First check if it's an index expression like g2[0] or g2[0][0]
            let inferred_type = if let Some(idx_type) = Self::infer_index_expr_type(before_dot, local_vars) {
                eprintln!("Inferred index expression type: {}", idx_type);
                Some(idx_type)
            } else {
                // Check if it's a local variable we know about
                // Extract just the identifier (handle cases like "    yy" -> "yy")
                let var_name = before_dot.split(|c: char| !c.is_alphanumeric() && c != '_')
                    .filter(|s| !s.is_empty())
                    .last()
                    .unwrap_or(before_dot);

                // Use the engine's general expression type inference
                // which handles method chains, index expressions, and local bindings
                engine.infer_expression_type(before_dot, &local_vars)
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
