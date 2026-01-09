//! Async IO Runtime for Nostos
//!
//! Provides non-blocking IO operations (file, HTTP, etc.) that integrate
//! with the process scheduler. Processes can initiate IO operations and
//! yield while waiting for completion.
//!
//! Architecture:
//! - Single tokio runtime shared across all worker threads
//! - Worker threads send IO requests via channel
//! - Each request includes a oneshot channel for the response
//! - Process state becomes WaitingIO until response arrives

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::process::IoResponseValue;

/// Unique handle for an open file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileHandle(pub u64);

/// Unique handle for an HTTP client (for connection reuse)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HttpClientHandle(pub u64);

/// Unique handle for an HTTP server
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerHandle(pub u64);

/// File open mode
#[derive(Debug, Clone, Copy)]
pub enum FileMode {
    Read,
    Write,
    Append,
    ReadWrite,
}

/// HTTP method
#[derive(Debug, Clone)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
}

/// HTTP request configuration
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub timeout_ms: Option<u64>,
}

/// HTTP response
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Result type for IO operations
pub type IoResult<T> = Result<T, IoError>;

/// IO error type
#[derive(Debug, Clone)]
pub enum IoError {
    FileNotFound(String),
    PermissionDenied(String),
    IoError(String),
    InvalidHandle,
    HttpError(String),
    Timeout,
    InvalidUrl(String),
    ConnectionFailed(String),
    Other(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::FileNotFound(p) => write!(f, "File not found: {}", p),
            IoError::PermissionDenied(p) => write!(f, "Permission denied: {}", p),
            IoError::IoError(e) => write!(f, "IO error: {}", e),
            IoError::InvalidHandle => write!(f, "Invalid handle"),
            IoError::HttpError(e) => write!(f, "HTTP error: {}", e),
            IoError::Timeout => write!(f, "Operation timed out"),
            IoError::InvalidUrl(u) => write!(f, "Invalid URL: {}", u),
            IoError::ConnectionFailed(e) => write!(f, "Connection failed: {}", e),
            IoError::Other(e) => write!(f, "{}", e),
        }
    }
}

/// Response type for all IO requests
pub type IoResponse = oneshot::Sender<IoResult<IoResponseValue>>;

/// Seek whence position
#[derive(Debug, Clone, Copy)]
pub enum SeekWhence {
    Start,
    Current,
    End,
}

/// Request sent from worker thread to IO runtime
pub enum IoRequest {
    // File operations - convenience for one-shot file ops
    FileReadToString {
        path: PathBuf,
        response: IoResponse,
    },
    FileWriteAll {
        path: PathBuf,
        data: Vec<u8>,
        response: IoResponse,
    },
    FileAppend {
        path: PathBuf,
        data: Vec<u8>,
        response: IoResponse,
    },

    // File handle operations
    FileOpen {
        path: PathBuf,
        mode: FileMode,
        response: IoResponse,
    },
    FileWrite {
        handle: u64,
        data: Vec<u8>,
        response: IoResponse,
    },
    FileRead {
        handle: u64,
        size: usize,
        response: IoResponse,
    },
    FileReadLine {
        handle: u64,
        response: IoResponse,
    },
    FileFlush {
        handle: u64,
        response: IoResponse,
    },
    FileClose {
        handle: u64,
        response: IoResponse,
    },
    FileSeek {
        handle: u64,
        offset: i64,
        whence: SeekWhence,
        response: IoResponse,
    },

    // Directory operations
    DirCreate {
        path: PathBuf,
        response: IoResponse,
    },
    DirCreateAll {
        path: PathBuf,
        response: IoResponse,
    },
    DirList {
        path: PathBuf,
        response: IoResponse,
    },
    DirRemove {
        path: PathBuf,
        response: IoResponse,
    },
    DirRemoveAll {
        path: PathBuf,
        response: IoResponse,
    },

    // File utilities
    FileExists {
        path: PathBuf,
        response: IoResponse,
    },
    DirExists {
        path: PathBuf,
        response: IoResponse,
    },
    FileRemove {
        path: PathBuf,
        response: IoResponse,
    },
    FileRename {
        old_path: PathBuf,
        new_path: PathBuf,
        response: IoResponse,
    },
    FileCopy {
        src_path: PathBuf,
        dest_path: PathBuf,
        response: IoResponse,
    },
    FileSize {
        path: PathBuf,
        response: IoResponse,
    },

    // HTTP operations
    HttpGet {
        url: String,
        response: IoResponse,
    },
    HttpRequest {
        request: HttpRequest,
        response: IoResponse,
    },

    // HTTP Server operations
    ServerBind {
        port: u16,
        response: IoResponse,
    },
    ServerAccept {
        handle: u64,
        response: IoResponse,
    },
    ServerRespond {
        request_id: u64,
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        response: IoResponse,
    },
    ServerClose {
        handle: u64,
        response: IoResponse,
    },

    // External process operations
    /// Run a command and wait for completion (captures stdout/stderr)
    ExecRun {
        command: String,
        args: Vec<String>,
        response: IoResponse,
    },
    /// Spawn a process with streaming I/O (returns handle)
    ExecSpawn {
        command: String,
        args: Vec<String>,
        response: IoResponse,
    },
    /// Read a line from spawned process stdout
    ExecReadLine {
        handle: u64,
        response: IoResponse,
    },
    /// Read a line from spawned process stderr
    ExecReadStderr {
        handle: u64,
        response: IoResponse,
    },
    /// Write to spawned process stdin
    ExecWrite {
        handle: u64,
        data: Vec<u8>,
        response: IoResponse,
    },
    /// Wait for spawned process to exit (returns exit code)
    ExecWait {
        handle: u64,
        response: IoResponse,
    },
    /// Kill a spawned process
    ExecKill {
        handle: u64,
        response: IoResponse,
    },

    // Shutdown
    Shutdown,
}

/// Open file state (held by the IO runtime)
struct OpenFile {
    file: tokio::fs::File,
    path: PathBuf,
    #[allow(dead_code)]
    mode: FileMode,
}

/// The IO runtime - runs on a separate tokio runtime
pub struct IoRuntime {
    /// Tokio runtime handle
    runtime: Runtime,
    /// Channel to send requests to the runtime
    request_tx: mpsc::UnboundedSender<IoRequest>,
    /// Next file handle ID
    next_handle: AtomicU64,
    /// HTTP client for connection pooling
    http_client: reqwest::Client,
}

impl IoRuntime {
    /// Create a new IO runtime
    pub fn new() -> Self {
        let runtime = Runtime::new().expect("Failed to create tokio runtime");
        let (request_tx, request_rx) = mpsc::unbounded_channel();

        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(10)
            .build()
            .expect("Failed to create HTTP client");

        let io_runtime = IoRuntime {
            runtime,
            request_tx,
            next_handle: AtomicU64::new(1),
            http_client: http_client.clone(),
        };

        // Spawn the request handler
        let client = http_client;
        io_runtime.runtime.spawn(async move {
            Self::run_handler(request_rx, client).await;
        });

        io_runtime
    }

    /// Get sender for IO requests (clone for each worker thread)
    pub fn request_sender(&self) -> mpsc::UnboundedSender<IoRequest> {
        self.request_tx.clone()
    }

    /// Generate a new unique file handle
    pub fn next_file_handle(&self) -> FileHandle {
        FileHandle(self.next_handle.fetch_add(1, Ordering::Relaxed))
    }

    /// The main handler loop running on tokio
    async fn run_handler(
        mut request_rx: mpsc::UnboundedReceiver<IoRequest>,
        http_client: reqwest::Client,
    ) {
        use std::io::SeekFrom;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, AsyncBufReadExt};
        use tokio::sync::Mutex;

        // Track open file handles - use BufReader for line-by-line reading support
        let mut open_files: HashMap<u64, BufReader<tokio::fs::File>> = HashMap::new();
        let mut next_handle: u64 = 1;

        // HTTP Server state
        // Request type: (request_id, method, path, headers, body)
        type ServerRequest = (u64, String, String, Vec<(String, String)>, Vec<u8>);
        type ServerRequestTx = mpsc::UnboundedSender<ServerRequest>;
        type ServerRequestRx = mpsc::UnboundedReceiver<ServerRequest>;

        // Maps server handle -> shared receiver for incoming requests
        // Multiple ServerAccept calls can wait on the same receiver concurrently
        let server_request_receivers: Arc<Mutex<HashMap<u64, Arc<Mutex<ServerRequestRx>>>>> = Arc::new(Mutex::new(HashMap::new()));

        // Maps request_id -> oneshot sender for the response
        type ResponseSender = oneshot::Sender<(u16, Vec<(String, String)>, Vec<u8>)>;
        let pending_responses: Arc<Mutex<HashMap<u64, ResponseSender>>> = Arc::new(Mutex::new(HashMap::new()));
        let mut next_server_handle: u64 = 1;
        let mut next_request_id: Arc<std::sync::atomic::AtomicU64> = Arc::new(std::sync::atomic::AtomicU64::new(1));

        // Spawned process state
        // Each spawned process has optional stdin, buffered stdout/stderr readers
        use tokio::process::{Child, ChildStdin, ChildStdout, ChildStderr};
        struct SpawnedProcess {
            child: Child,
            stdin: Option<ChildStdin>,
            stdout: Option<BufReader<ChildStdout>>,
            stderr: Option<BufReader<ChildStderr>>,
        }
        let spawned_processes: Arc<Mutex<HashMap<u64, SpawnedProcess>>> = Arc::new(Mutex::new(HashMap::new()));
        let mut next_process_handle: u64 = 1;

        while let Some(request) = request_rx.recv().await {
            match request {
                IoRequest::Shutdown => break,

                IoRequest::FileReadToString { path, response } => {
                    let result = Self::handle_file_read_to_string(&path).await;
                    let _ = response.send(result.map(IoResponseValue::String));
                }

                IoRequest::FileWriteAll { path, data, response } => {
                    let result = Self::handle_file_write_all(&path, &data).await;
                    let _ = response.send(result.map(|_| IoResponseValue::Unit));
                }

                IoRequest::FileAppend { path, data, response } => {
                    let result = tokio::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                        .await;
                    let result = match result {
                        Ok(mut file) => {
                            match file.write_all(&data).await {
                                Ok(()) => Ok(IoResponseValue::Unit),
                                Err(e) => Err(IoError::IoError(e.to_string())),
                            }
                        }
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                // File handle operations
                IoRequest::FileOpen { path, mode, response } => {
                    use tokio::fs::OpenOptions;
                    let file_result = match mode {
                        FileMode::Read => OpenOptions::new().read(true).open(&path).await,
                        FileMode::Write => OpenOptions::new()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .open(&path)
                            .await,
                        FileMode::Append => OpenOptions::new()
                            .write(true)
                            .create(true)
                            .append(true)
                            .open(&path)
                            .await,
                        FileMode::ReadWrite => OpenOptions::new()
                            .read(true)
                            .write(true)
                            .create(true)
                            .open(&path)
                            .await,
                    };
                    let result = match file_result {
                        Ok(file) => {
                            let handle = next_handle;
                            next_handle += 1;
                            open_files.insert(handle, BufReader::new(file));
                            Ok(IoResponseValue::FileHandle(handle))
                        }
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileWrite { handle, data, response } => {
                    let result = match open_files.get_mut(&handle) {
                        Some(buf_reader) => {
                            // Access inner file for writing
                            match buf_reader.get_mut().write(&data).await {
                                Ok(n) => Ok(IoResponseValue::Int(n as i64)),
                                Err(e) => Err(IoError::IoError(e.to_string())),
                            }
                        }
                        None => Err(IoError::InvalidHandle),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileRead { handle, size, response } => {
                    let result = match open_files.get_mut(&handle) {
                        Some(file) => {
                            let mut buf = vec![0u8; size];
                            match file.read(&mut buf).await {
                                Ok(n) => {
                                    buf.truncate(n);
                                    Ok(IoResponseValue::Bytes(buf))
                                }
                                Err(e) => Err(IoError::IoError(e.to_string())),
                            }
                        }
                        None => Err(IoError::InvalidHandle),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileReadLine { handle, response } => {
                    let result = match open_files.get_mut(&handle) {
                        Some(file) => {
                            let mut line = String::new();
                            match file.read_line(&mut line).await {
                                Ok(0) => Ok(IoResponseValue::OptionString(None)), // EOF
                                Ok(_) => Ok(IoResponseValue::OptionString(Some(line))),
                                Err(e) => Err(IoError::IoError(e.to_string())),
                            }
                        }
                        None => Err(IoError::InvalidHandle),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileFlush { handle, response } => {
                    let result = match open_files.get_mut(&handle) {
                        Some(buf_reader) => {
                            // Access inner file for flushing
                            match buf_reader.get_mut().flush().await {
                                Ok(()) => Ok(IoResponseValue::Unit),
                                Err(e) => Err(IoError::IoError(e.to_string())),
                            }
                        }
                        None => Err(IoError::InvalidHandle),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileClose { handle, response } => {
                    let result = match open_files.remove(&handle) {
                        Some(file) => {
                            drop(file);
                            Ok(IoResponseValue::Unit)
                        }
                        None => Err(IoError::InvalidHandle),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileSeek { handle, offset, whence, response } => {
                    let result = match open_files.get_mut(&handle) {
                        Some(buf_reader) => {
                            let seek_from = match whence {
                                SeekWhence::Start => SeekFrom::Start(offset as u64),
                                SeekWhence::Current => SeekFrom::Current(offset),
                                SeekWhence::End => SeekFrom::End(offset),
                            };
                            // BufReader.seek properly invalidates the buffer
                            match buf_reader.seek(seek_from).await {
                                Ok(pos) => Ok(IoResponseValue::Int(pos as i64)),
                                Err(e) => Err(IoError::IoError(e.to_string())),
                            }
                        }
                        None => Err(IoError::InvalidHandle),
                    };
                    let _ = response.send(result);
                }

                // Directory operations
                IoRequest::DirCreate { path, response } => {
                    let result = match tokio::fs::create_dir(&path).await {
                        Ok(()) => Ok(IoResponseValue::Unit),
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::DirCreateAll { path, response } => {
                    let result = match tokio::fs::create_dir_all(&path).await {
                        Ok(()) => Ok(IoResponseValue::Unit),
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::DirList { path, response } => {
                    let result = match tokio::fs::read_dir(&path).await {
                        Ok(mut entries) => {
                            let mut names = Vec::new();
                            while let Ok(Some(entry)) = entries.next_entry().await {
                                if let Ok(name) = entry.file_name().into_string() {
                                    names.push(name);
                                }
                            }
                            Ok(IoResponseValue::StringList(names))
                        }
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::DirRemove { path, response } => {
                    let result = match tokio::fs::remove_dir(&path).await {
                        Ok(()) => Ok(IoResponseValue::Unit),
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::DirRemoveAll { path, response } => {
                    let result = match tokio::fs::remove_dir_all(&path).await {
                        Ok(()) => Ok(IoResponseValue::Unit),
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                // File utilities
                IoRequest::FileExists { path, response } => {
                    let exists = tokio::fs::try_exists(&path).await.unwrap_or(false);
                    let _ = response.send(Ok(IoResponseValue::Bool(exists)));
                }

                IoRequest::DirExists { path, response } => {
                    let result = match tokio::fs::metadata(&path).await {
                        Ok(meta) => Ok(IoResponseValue::Bool(meta.is_dir())),
                        Err(_) => Ok(IoResponseValue::Bool(false)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileRemove { path, response } => {
                    let result = match tokio::fs::remove_file(&path).await {
                        Ok(()) => Ok(IoResponseValue::Unit),
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileRename { old_path, new_path, response } => {
                    let result = match tokio::fs::rename(&old_path, &new_path).await {
                        Ok(()) => Ok(IoResponseValue::Unit),
                        Err(e) => Err(Self::convert_io_error(e, &old_path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileCopy { src_path, dest_path, response } => {
                    let result = match tokio::fs::copy(&src_path, &dest_path).await {
                        Ok(bytes) => Ok(IoResponseValue::Int(bytes as i64)),
                        Err(e) => Err(Self::convert_io_error(e, &src_path)),
                    };
                    let _ = response.send(result);
                }

                IoRequest::FileSize { path, response } => {
                    let result = match tokio::fs::metadata(&path).await {
                        Ok(meta) => Ok(IoResponseValue::Int(meta.len() as i64)),
                        Err(e) => Err(Self::convert_io_error(e, &path)),
                    };
                    let _ = response.send(result);
                }

                // HTTP operations - spawned as tasks to avoid blocking the IO loop
                IoRequest::HttpGet { url, response } => {
                    let client = http_client.clone();
                    tokio::spawn(async move {
                        let result = Self::handle_http_get(&client, &url).await;
                        let _ = response.send(result.map(|resp| IoResponseValue::HttpResponse {
                            status: resp.status,
                            headers: resp.headers,
                            body: resp.body,
                        }));
                    });
                }

                IoRequest::HttpRequest { request, response } => {
                    let client = http_client.clone();
                    tokio::spawn(async move {
                        let result = Self::handle_http_request(&client, request).await;
                        let _ = response.send(result.map(|resp| IoResponseValue::HttpResponse {
                            status: resp.status,
                            headers: resp.headers,
                            body: resp.body,
                        }));
                    });
                }

                // HTTP Server operations
                IoRequest::ServerBind { port, response } => {
                    use axum::{
                        extract::Request,
                        routing::any,
                        Router,
                    };

                    let handle = next_server_handle;
                    next_server_handle += 1;

                    // Create channel for incoming requests - wrap receiver in Arc<Mutex<>> for sharing
                    let (req_tx, req_rx) = mpsc::unbounded_channel();
                    let shared_rx = Arc::new(Mutex::new(req_rx));
                    server_request_receivers.lock().await.insert(handle, shared_rx);

                    // Clone shared state for the axum handler
                    let pending = pending_responses.clone();
                    let req_id_gen = next_request_id.clone();

                    // Create the router
                    let app = Router::new().fallback(any(move |request: Request| {
                        let req_tx = req_tx.clone();
                        let pending = pending.clone();
                        let req_id_gen = req_id_gen.clone();
                        async move {
                            // Extract request parts
                            let method = request.method().to_string();
                            let path = request.uri().path().to_string();
                            let headers: Vec<(String, String)> = request
                                .headers()
                                .iter()
                                .map(|(k, v)| {
                                    (k.to_string(), v.to_str().unwrap_or("").to_string())
                                })
                                .collect();

                            // Read body
                            let body = match axum::body::to_bytes(request.into_body(), usize::MAX).await {
                                Ok(bytes) => bytes.to_vec(),
                                Err(_) => vec![],
                            };

                            // Generate request ID and create response channel
                            let request_id = req_id_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let (resp_tx, resp_rx) = oneshot::channel();

                            // Store the response channel
                            pending.lock().await.insert(request_id, resp_tx);

                            // Send request to the Nostos process
                            let _ = req_tx.send((request_id, method, path, headers, body));

                            // Wait for response from Nostos
                            match resp_rx.await {
                                Ok((status, headers, body)) => {
                                    let mut response = axum::response::Response::builder()
                                        .status(status);
                                    for (key, value) in headers {
                                        response = response.header(key, value);
                                    }
                                    response.body(axum::body::Body::from(body)).unwrap()
                                }
                                Err(_) => {
                                    // Request was abandoned
                                    axum::response::Response::builder()
                                        .status(500)
                                        .body(axum::body::Body::from("Internal Server Error"))
                                        .unwrap()
                                }
                            }
                        }
                    }));

                    // Spawn the server
                    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
                    let listener_result = tokio::net::TcpListener::bind(addr).await;

                    match listener_result {
                        Ok(listener) => {
                            tokio::spawn(async move {
                                axum::serve(listener, app).await.ok();
                            });
                            let _ = response.send(Ok(IoResponseValue::ServerHandle(handle)));
                        }
                        Err(e) => {
                            server_request_receivers.lock().await.remove(&handle);
                            let _ = response.send(Err(IoError::IoError(format!("Failed to bind: {}", e))));
                        }
                    }
                }

                IoRequest::ServerAccept { handle, response } => {
                    // Get the shared receiver for this server handle
                    let receivers = server_request_receivers.clone();
                    tokio::spawn(async move {
                        // Get the shared receiver (briefly lock the map)
                        let shared_rx = {
                            let map = receivers.lock().await;
                            map.get(&handle).cloned()
                        };

                        match shared_rx {
                            Some(rx) => {
                                // Lock the receiver and wait for next request
                                // Only one waiter at a time can hold this lock
                                let result = rx.lock().await.recv().await;
                                match result {
                                    Some((request_id, method, path, headers, body)) => {
                                        let _ = response.send(Ok(IoResponseValue::ServerRequest {
                                            request_id,
                                            method,
                                            path,
                                            headers,
                                            body,
                                        }));
                                    }
                                    None => {
                                        let _ = response.send(Err(IoError::Other("Server closed".to_string())));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::ServerRespond { request_id, status, headers, body, response } => {
                    let mut pending = pending_responses.lock().await;
                    match pending.remove(&request_id) {
                        Some(resp_tx) => {
                            let _ = resp_tx.send((status, headers, body));
                            let _ = response.send(Ok(IoResponseValue::Unit));
                        }
                        None => {
                            let _ = response.send(Err(IoError::Other("Request not found or already responded".to_string())));
                        }
                    }
                }

                IoRequest::ServerClose { handle, response } => {
                    // Remove the receiver - this will cause the server to stop accepting
                    // Note: The actual TCP listener continues but new requests will have nowhere to go
                    server_request_receivers.lock().await.remove(&handle);
                    let _ = response.send(Ok(IoResponseValue::Unit));
                }

                // External process operations
                IoRequest::ExecRun { command, args, response } => {
                    // Spawn and run in background task to avoid blocking
                    tokio::spawn(async move {
                        use tokio::process::Command;
                        let result = Command::new(&command)
                            .args(&args)
                            .output()
                            .await;

                        let result = match result {
                            Ok(output) => Ok(IoResponseValue::ExecResult {
                                exit_code: output.status.code().unwrap_or(-1),
                                stdout: output.stdout,
                                stderr: output.stderr,
                            }),
                            Err(e) => Err(IoError::IoError(format!("Failed to execute {}: {}", command, e))),
                        };
                        let _ = response.send(result);
                    });
                }

                IoRequest::ExecSpawn { command, args, response } => {
                    use tokio::process::Command;
                    use std::process::Stdio;

                    let child_result = Command::new(&command)
                        .args(&args)
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn();

                    match child_result {
                        Ok(mut child) => {
                            let handle = next_process_handle;
                            next_process_handle += 1;

                            let stdin = child.stdin.take();
                            let stdout = child.stdout.take().map(BufReader::new);
                            let stderr = child.stderr.take().map(BufReader::new);

                            spawned_processes.lock().await.insert(handle, SpawnedProcess {
                                child,
                                stdin,
                                stdout,
                                stderr,
                            });

                            let _ = response.send(Ok(IoResponseValue::ExecHandle(handle)));
                        }
                        Err(e) => {
                            let _ = response.send(Err(IoError::IoError(format!("Failed to spawn {}: {}", command, e))));
                        }
                    }
                }

                IoRequest::ExecReadLine { handle, response } => {
                    let processes = spawned_processes.clone();
                    tokio::spawn(async move {
                        let mut procs = processes.lock().await;
                        match procs.get_mut(&handle) {
                            Some(proc) => {
                                match &mut proc.stdout {
                                    Some(stdout) => {
                                        let mut line = String::new();
                                        match stdout.read_line(&mut line).await {
                                            Ok(0) => {
                                                let _ = response.send(Ok(IoResponseValue::OptionString(None)));
                                            }
                                            Ok(_) => {
                                                let _ = response.send(Ok(IoResponseValue::OptionString(Some(line))));
                                            }
                                            Err(e) => {
                                                let _ = response.send(Err(IoError::IoError(e.to_string())));
                                            }
                                        }
                                    }
                                    None => {
                                        let _ = response.send(Err(IoError::Other("Process stdout not available".to_string())));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::ExecReadStderr { handle, response } => {
                    let processes = spawned_processes.clone();
                    tokio::spawn(async move {
                        let mut procs = processes.lock().await;
                        match procs.get_mut(&handle) {
                            Some(proc) => {
                                match &mut proc.stderr {
                                    Some(stderr) => {
                                        let mut line = String::new();
                                        match stderr.read_line(&mut line).await {
                                            Ok(0) => {
                                                let _ = response.send(Ok(IoResponseValue::OptionString(None)));
                                            }
                                            Ok(_) => {
                                                let _ = response.send(Ok(IoResponseValue::OptionString(Some(line))));
                                            }
                                            Err(e) => {
                                                let _ = response.send(Err(IoError::IoError(e.to_string())));
                                            }
                                        }
                                    }
                                    None => {
                                        let _ = response.send(Err(IoError::Other("Process stderr not available".to_string())));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::ExecWrite { handle, data, response } => {
                    let processes = spawned_processes.clone();
                    tokio::spawn(async move {
                        let mut procs = processes.lock().await;
                        match procs.get_mut(&handle) {
                            Some(proc) => {
                                match &mut proc.stdin {
                                    Some(stdin) => {
                                        match stdin.write_all(&data).await {
                                            Ok(()) => {
                                                let _ = response.send(Ok(IoResponseValue::Unit));
                                            }
                                            Err(e) => {
                                                let _ = response.send(Err(IoError::IoError(e.to_string())));
                                            }
                                        }
                                    }
                                    None => {
                                        let _ = response.send(Err(IoError::Other("Process stdin not available".to_string())));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::ExecWait { handle, response } => {
                    let processes = spawned_processes.clone();
                    tokio::spawn(async move {
                        // Take ownership of the process to wait on it
                        let proc_opt = {
                            let mut procs = processes.lock().await;
                            procs.remove(&handle)
                        };

                        match proc_opt {
                            Some(mut proc) => {
                                match proc.child.wait().await {
                                    Ok(status) => {
                                        let _ = response.send(Ok(IoResponseValue::ExitCode(status.code().unwrap_or(-1))));
                                    }
                                    Err(e) => {
                                        let _ = response.send(Err(IoError::IoError(e.to_string())));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::ExecKill { handle, response } => {
                    let mut procs = spawned_processes.lock().await;
                    match procs.get_mut(&handle) {
                        Some(proc) => {
                            match proc.child.kill().await {
                                Ok(()) => {
                                    // Remove the process after killing
                                    procs.remove(&handle);
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(e.to_string())));
                                }
                            }
                        }
                        None => {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    }
                }
            }
        }
    }

    async fn handle_file_open(
        path: &PathBuf,
        mode: FileMode,
        next_handle: &mut u64,
    ) -> IoResult<(FileHandle, OpenFile)> {
        use tokio::fs::OpenOptions;

        let file = match mode {
            FileMode::Read => OpenOptions::new().read(true).open(path).await,
            FileMode::Write => OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .await,
            FileMode::Append => OpenOptions::new()
                .write(true)
                .create(true)
                .append(true)
                .open(path)
                .await,
            FileMode::ReadWrite => OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(path)
                .await,
        };

        match file {
            Ok(file) => {
                let handle = FileHandle(*next_handle);
                *next_handle += 1;
                Ok((handle, OpenFile {
                    file,
                    path: path.clone(),
                    mode,
                }))
            }
            Err(e) => Err(Self::convert_io_error(e, path)),
        }
    }

    async fn handle_file_read(file: &mut tokio::fs::File, size: usize) -> IoResult<Vec<u8>> {
        use tokio::io::AsyncReadExt;
        let mut buf = vec![0u8; size];
        match file.read(&mut buf).await {
            Ok(n) => {
                buf.truncate(n);
                Ok(buf)
            }
            Err(e) => Err(IoError::IoError(e.to_string())),
        }
    }

    async fn handle_file_read_all(file: &mut tokio::fs::File) -> IoResult<Vec<u8>> {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        match file.read_to_end(&mut buf).await {
            Ok(_) => Ok(buf),
            Err(e) => Err(IoError::IoError(e.to_string())),
        }
    }

    async fn handle_file_read_line(file: &mut tokio::fs::File) -> IoResult<Option<String>> {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Ok(0) => Ok(None), // EOF
            Ok(_) => Ok(Some(line)),
            Err(e) => Err(IoError::IoError(e.to_string())),
        }
    }

    async fn handle_file_write(file: &mut tokio::fs::File, data: &[u8]) -> IoResult<usize> {
        use tokio::io::AsyncWriteExt;
        match file.write(data).await {
            Ok(n) => Ok(n),
            Err(e) => Err(IoError::IoError(e.to_string())),
        }
    }

    async fn handle_file_read_to_string(path: &PathBuf) -> IoResult<String> {
        match tokio::fs::read_to_string(path).await {
            Ok(s) => Ok(s),
            Err(e) => Err(Self::convert_io_error(e, path)),
        }
    }

    async fn handle_file_write_all(path: &PathBuf, data: &[u8]) -> IoResult<()> {
        match tokio::fs::write(path, data).await {
            Ok(()) => Ok(()),
            Err(e) => Err(Self::convert_io_error(e, path)),
        }
    }

    async fn handle_http_get(client: &reqwest::Client, url: &str) -> IoResult<HttpResponse> {
        let result = client.get(url).send().await;
        match result {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let headers: Vec<(String, String)> = resp
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                match resp.bytes().await {
                    Ok(body) => Ok(HttpResponse {
                        status,
                        headers,
                        body: body.to_vec(),
                    }),
                    Err(e) => Err(IoError::HttpError(e.to_string())),
                }
            }
            Err(e) => {
                if e.is_timeout() {
                    Err(IoError::Timeout)
                } else if e.is_connect() {
                    Err(IoError::ConnectionFailed(e.to_string()))
                } else {
                    Err(IoError::HttpError(e.to_string()))
                }
            }
        }
    }

    async fn handle_http_request(
        client: &reqwest::Client,
        request: HttpRequest,
    ) -> IoResult<HttpResponse> {
        let method = match request.method {
            HttpMethod::Get => reqwest::Method::GET,
            HttpMethod::Post => reqwest::Method::POST,
            HttpMethod::Put => reqwest::Method::PUT,
            HttpMethod::Delete => reqwest::Method::DELETE,
            HttpMethod::Patch => reqwest::Method::PATCH,
            HttpMethod::Head => reqwest::Method::HEAD,
        };

        let mut req_builder = client.request(method, &request.url);

        for (key, value) in request.headers {
            req_builder = req_builder.header(key, value);
        }

        if let Some(body) = request.body {
            req_builder = req_builder.body(body);
        }

        if let Some(timeout_ms) = request.timeout_ms {
            req_builder = req_builder.timeout(std::time::Duration::from_millis(timeout_ms));
        }

        let result = req_builder.send().await;

        match result {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let headers: Vec<(String, String)> = resp
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                match resp.bytes().await {
                    Ok(body) => Ok(HttpResponse {
                        status,
                        headers,
                        body: body.to_vec(),
                    }),
                    Err(e) => Err(IoError::HttpError(e.to_string())),
                }
            }
            Err(e) => {
                if e.is_timeout() {
                    Err(IoError::Timeout)
                } else if e.is_connect() {
                    Err(IoError::ConnectionFailed(e.to_string()))
                } else {
                    Err(IoError::HttpError(e.to_string()))
                }
            }
        }
    }

    fn convert_io_error(e: std::io::Error, path: &PathBuf) -> IoError {
        use std::io::ErrorKind;
        match e.kind() {
            ErrorKind::NotFound => IoError::FileNotFound(path.display().to_string()),
            ErrorKind::PermissionDenied => IoError::PermissionDenied(path.display().to_string()),
            _ => IoError::IoError(e.to_string()),
        }
    }

    /// Shutdown the IO runtime
    pub fn shutdown(&self) {
        let _ = self.request_tx.send(IoRequest::Shutdown);
    }
}

impl Default for IoRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IoRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_file_read_to_string() {
        let io = IoRuntime::new();
        let tx = io.request_sender();

        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("nostos_test_io.txt");
        std::fs::write(&path, "Hello, Nostos!\n").unwrap();

        // Send read request
        let (resp_tx, resp_rx) = oneshot::channel();
        tx.send(IoRequest::FileReadToString {
            path: path.clone(),
            response: resp_tx,
        }).unwrap();

        // Block on response (for testing only)
        let result = io.runtime.block_on(async {
            resp_rx.await.unwrap()
        });

        // Clean up
        let _ = std::fs::remove_file(&path);

        assert!(result.is_ok());
        match result.unwrap() {
            IoResponseValue::String(s) => assert_eq!(s.trim(), "Hello, Nostos!"),
            _ => panic!("Expected String response"),
        }
    }

    #[test]
    #[ignore]
    fn test_http_get() {
        let io = IoRuntime::new();
        let tx = io.request_sender();

        let (resp_tx, resp_rx) = oneshot::channel();
        tx.send(IoRequest::HttpGet {
            url: "https://httpbin.org/get".to_string(),
            response: resp_tx,
        }).unwrap();

        let result = io.runtime.block_on(async {
            resp_rx.await.unwrap()
        });

        assert!(result.is_ok());
        match result.unwrap() {
            IoResponseValue::HttpResponse { status, .. } => assert_eq!(status, 200),
            _ => panic!("Expected HttpResponse"),
        }
    }
}
