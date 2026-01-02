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
use std::sync::Arc;

use chrono::Timelike;
use std::future::poll_fn;
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot, Mutex as TokioMutex};
use tokio_postgres::{AsyncMessage, Client as PgClient, NoTls, Statement};
use postgres_native_tls::MakeTlsConnector;
use native_tls::TlsConnector;
use deadpool_postgres::{Manager as PgPoolManager, Pool as PgPool, Runtime as DeadpoolRuntime, ManagerConfig, RecyclingMethod};
use futures::{SinkExt, StreamExt};
use thirtyfour::prelude::*;

use crate::process::{IoResponseValue, PgValue};

/// Wrapper for pgvector binary format
/// Format: 2 bytes dim (u16 big-endian), 2 bytes unused, n*4 bytes f32 (big-endian)
struct PgVector(Vec<f32>);

impl<'a> tokio_postgres::types::FromSql<'a> for PgVector {
    fn from_sql(
        _ty: &tokio_postgres::types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        if raw.len() < 4 {
            return Err("pgvector data too short".into());
        }
        // Dimension is first 2 bytes, big-endian
        let dim = u16::from_be_bytes([raw[0], raw[1]]) as usize;
        let expected_len = 4 + dim * 4;
        if raw.len() < expected_len {
            return Err(format!("pgvector data too short: expected {} bytes, got {}", expected_len, raw.len()).into());
        }
        let mut floats = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = 4 + i * 4;
            let bytes = [raw[offset], raw[offset+1], raw[offset+2], raw[offset+3]];
            floats.push(f32::from_be_bytes(bytes));
        }
        Ok(PgVector(floats))
    }

    fn accepts(ty: &tokio_postgres::types::Type) -> bool {
        ty.name() == "vector"
    }
}

/// Pooled connection wrapper that holds the connection object
/// When dropped, the connection returns to the pool automatically
struct PgPooledConnection(deadpool_postgres::Object);

impl PgPooledConnection {
    /// Get a reference to the underlying client
    fn client(&self) -> &PgClient {
        self.0.as_ref()
    }
}

/// PostgreSQL connection with prepared statements (pooled or direct)
enum PgConnection {
    /// Direct connection (legacy, non-pooled)
    Direct {
        client: PgClient,
        prepared: HashMap<String, Statement>,
    },
    /// Pooled connection
    Pooled {
        conn: PgPooledConnection,
        prepared: HashMap<String, Statement>,
    },
}

impl PgConnection {
    fn client(&self) -> &PgClient {
        match self {
            PgConnection::Direct { client, .. } => client,
            PgConnection::Pooled { conn, .. } => conn.client(),
        }
    }

    fn prepared(&self) -> &HashMap<String, Statement> {
        match self {
            PgConnection::Direct { prepared, .. } => prepared,
            PgConnection::Pooled { prepared, .. } => prepared,
        }
    }

    fn prepared_mut(&mut self) -> &mut HashMap<String, Statement> {
        match self {
            PgConnection::Direct { prepared, .. } => prepared,
            PgConnection::Pooled { prepared, .. } => prepared,
        }
    }
}

/// Connection pools keyed by connection string
type PgPools = Arc<TokioMutex<HashMap<String, PgPool>>>;

/// Listener connection for LISTEN/NOTIFY
/// Uses a dedicated non-pooled connection with a notification channel
struct PgListenerConnection {
    client: PgClient,
    notification_rx: TokioMutex<mpsc::UnboundedReceiver<(String, String)>>,
}

/// Create a TLS connector for PostgreSQL
fn create_tls_connector() -> Result<MakeTlsConnector, native_tls::Error> {
    let tls_connector = TlsConnector::builder()
        .danger_accept_invalid_certs(true) // For self-signed certs
        .min_protocol_version(Some(native_tls::Protocol::Tlsv10)) // Allow older TLS
        .build()?;
    Ok(MakeTlsConnector::new(tls_connector))
}

/// Check if a connection string requires TLS
fn requires_tls(connection_string: &str) -> bool {
    connection_string.contains("sslmode=require") ||
    connection_string.contains("sslmode=prefer") ||
    connection_string.contains("sslmode=verify-ca") ||
    connection_string.contains("sslmode=verify-full")
}

/// Unique handle for an open file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileHandle(pub u64);

/// Unique handle for an HTTP client (for connection reuse)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HttpClientHandle(pub u64);

/// Unique handle for an HTTP server
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerHandle(pub u64);

/// Unique handle for a Postgres connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgHandle(pub u64);

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

/// PostgreSQL parameter types
#[derive(Debug, Clone)]
pub enum PgParam {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    /// Timestamp as milliseconds since Unix epoch (UTC)
    Timestamp(i64),
    /// JSON/JSONB data as string
    Json(String),
    /// Vector data (for pgvector extension)
    Vector(Vec<f32>),
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
    PgError(String),
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
            IoError::PgError(e) => write!(f, "PostgreSQL error: {}", e),
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

    // WebSocket operations
    /// Accept a WebSocket upgrade (sends 101 response and waits for connection)
    WebSocketAccept {
        request_id: u64,
        response: IoResponse,
    },
    /// Send a message on a WebSocket connection
    WebSocketSend {
        request_id: u64,
        message: String,
        response: IoResponse,
    },
    /// Receive a message from a WebSocket connection
    WebSocketReceive {
        request_id: u64,
        response: IoResponse,
    },
    /// Close a WebSocket connection
    WebSocketClose {
        request_id: u64,
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

    // PostgreSQL operations
    /// Connect to a PostgreSQL database
    PgConnect {
        connection_string: String,
        response: IoResponse,
    },
    /// Execute a query and return rows
    PgQuery {
        handle: u64,
        query: String,
        params: Vec<PgParam>,
        response: IoResponse,
    },
    /// Execute a statement (INSERT, UPDATE, DELETE) and return affected rows
    PgExecute {
        handle: u64,
        query: String,
        params: Vec<PgParam>,
        response: IoResponse,
    },
    /// Close a PostgreSQL connection
    PgClose {
        handle: u64,
        response: IoResponse,
    },
    /// Begin a transaction
    PgBegin {
        handle: u64,
        response: IoResponse,
    },
    /// Commit a transaction
    PgCommit {
        handle: u64,
        response: IoResponse,
    },
    /// Rollback a transaction
    PgRollback {
        handle: u64,
        response: IoResponse,
    },
    /// Prepare a statement
    PgPrepare {
        handle: u64,
        name: String,
        query: String,
        response: IoResponse,
    },
    /// Execute a prepared query (returns rows)
    PgQueryPrepared {
        handle: u64,
        name: String,
        params: Vec<PgParam>,
        response: IoResponse,
    },
    /// Execute a prepared statement (returns affected rows)
    PgExecutePrepared {
        handle: u64,
        name: String,
        params: Vec<PgParam>,
        response: IoResponse,
    },
    /// Deallocate a prepared statement
    PgDeallocate {
        handle: u64,
        name: String,
        response: IoResponse,
    },
    /// Create a dedicated listener connection for LISTEN/NOTIFY
    PgListenConnect {
        connection_string: String,
        response: IoResponse,
    },
    /// Start listening on a channel (LISTEN)
    PgListen {
        handle: u64,
        channel: String,
        response: IoResponse,
    },
    /// Stop listening on a channel (UNLISTEN)
    PgUnlisten {
        handle: u64,
        channel: String,
        response: IoResponse,
    },
    /// Wait for a notification with timeout
    PgAwaitNotification {
        handle: u64,
        timeout_ms: u64,
        response: IoResponse,
    },
    /// Send a notification (NOTIFY)
    PgNotify {
        handle: u64,
        channel: String,
        payload: String,
        response: IoResponse,
    },

    // Selenium WebDriver operations
    /// Connect to WebDriver
    SeleniumConnect {
        webdriver_url: String,
        response: IoResponse,
    },
    /// Navigate to URL
    SeleniumGoto {
        driver_handle: u64,
        url: String,
        response: IoResponse,
    },
    /// Click element by CSS selector
    SeleniumClick {
        driver_handle: u64,
        selector: String,
        response: IoResponse,
    },
    /// Get text content by CSS selector
    SeleniumText {
        driver_handle: u64,
        selector: String,
        response: IoResponse,
    },
    /// Send keys to element
    SeleniumSendKeys {
        driver_handle: u64,
        selector: String,
        text: String,
        response: IoResponse,
    },
    /// Execute JavaScript
    SeleniumExecuteJs {
        driver_handle: u64,
        script: String,
        response: IoResponse,
    },
    /// Execute JavaScript with args
    SeleniumExecuteJsWithArgs {
        driver_handle: u64,
        script: String,
        args: Vec<String>,
        response: IoResponse,
    },
    /// Wait for element
    SeleniumWaitFor {
        driver_handle: u64,
        selector: String,
        timeout_ms: u64,
        response: IoResponse,
    },
    /// Get element attribute
    SeleniumGetAttribute {
        driver_handle: u64,
        selector: String,
        attribute: String,
        response: IoResponse,
    },
    /// Check if element exists
    SeleniumExists {
        driver_handle: u64,
        selector: String,
        response: IoResponse,
    },
    /// Close WebDriver
    SeleniumClose {
        driver_handle: u64,
        response: IoResponse,
    },

    // Shutdown
    Shutdown,
}

/// Open file state (held by the IO runtime)
#[allow(dead_code)]
struct OpenFile {
    file: tokio::fs::File,
    path: PathBuf,
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
    #[allow(dead_code)]
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
        // Request type: (request_id, method, path, headers, body, query_params, cookies, form_params, is_websocket)
        type ServerRequest = (u64, String, String, Vec<(String, String)>, Vec<u8>, Vec<(String, String)>, Vec<(String, String)>, Vec<(String, String)>, bool);
        #[allow(dead_code)]
        type ServerRequestTx = mpsc::UnboundedSender<ServerRequest>;
        type ServerRequestRx = mpsc::UnboundedReceiver<ServerRequest>;

        // Maps server handle -> shared receiver for incoming requests
        // Multiple ServerAccept calls can wait on the same receiver concurrently
        let server_request_receivers: Arc<Mutex<HashMap<u64, Arc<Mutex<ServerRequestRx>>>>> = Arc::new(Mutex::new(HashMap::new()));

        // Maps server handle -> abort handle for the server task
        // Used to properly shut down the server and release the port
        let server_abort_handles: Arc<Mutex<HashMap<u64, tokio::task::AbortHandle>>> = Arc::new(Mutex::new(HashMap::new()));

        // Maps request_id -> oneshot sender for the response
        type ResponseSender = oneshot::Sender<(u16, Vec<(String, String)>, Vec<u8>)>;
        let pending_responses: Arc<Mutex<HashMap<u64, ResponseSender>>> = Arc::new(Mutex::new(HashMap::new()));
        let mut next_server_handle: u64 = 1;
        let next_request_id: Arc<std::sync::atomic::AtomicU64> = Arc::new(std::sync::atomic::AtomicU64::new(1));

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

        // PostgreSQL connection state (with prepared statements)
        let pg_connections: Arc<Mutex<HashMap<u64, PgConnection>>> = Arc::new(Mutex::new(HashMap::new()));
        let mut next_pg_handle: u64 = 1;

        // PostgreSQL connection pools (keyed by connection string)
        let pg_pools: PgPools = Arc::new(TokioMutex::new(HashMap::new()));

        // PostgreSQL listener connections (for LISTEN/NOTIFY)
        let pg_listeners: Arc<TokioMutex<HashMap<u64, PgListenerConnection>>> = Arc::new(TokioMutex::new(HashMap::new()));
        let mut next_listener_handle: u64 = 1;

        // WebSocket connection storage
        // Maps request_id -> (sender, receiver) for active WebSocket connections
        use tokio_tungstenite::WebSocketStream;
        use tokio_tungstenite::tungstenite::Message as WsMessage;
        type WsStream = WebSocketStream<hyper_util::rt::TokioIo<hyper::upgrade::Upgraded>>;
        type WsSender = futures::stream::SplitSink<WsStream, WsMessage>;
        type WsReceiver = futures::stream::SplitStream<WsStream>;
        let ws_connections: Arc<TokioMutex<HashMap<u64, (WsSender, WsReceiver)>>> = Arc::new(TokioMutex::new(HashMap::new()));

        // Pending WebSocket upgrades (stores the OnUpgrade handle until accept is called)
        use hyper::upgrade::OnUpgrade;
        let pending_ws_upgrades: Arc<TokioMutex<HashMap<u64, OnUpgrade>>> = Arc::new(TokioMutex::new(HashMap::new()));

        // Selenium WebDriver connections
        let selenium_drivers: Arc<TokioMutex<HashMap<u64, WebDriver>>> = Arc::new(TokioMutex::new(HashMap::new()));
        let mut next_selenium_handle: u64 = 1;

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
                    let pending_upgrades = pending_ws_upgrades.clone();
                    let ws_conns = ws_connections.clone();

                    // Create the router
                    let app = Router::new().fallback(any(move |mut request: Request| {
                        let req_tx = req_tx.clone();
                        let pending = pending.clone();
                        let req_id_gen = req_id_gen.clone();
                        let pending_upgrades = pending_upgrades.clone();
                        let ws_conns = ws_conns.clone();
                        async move {
                            // Extract request parts
                            let method = request.method().to_string();
                            let path = request.uri().path().to_string();

                            // Parse query parameters
                            let query_params: Vec<(String, String)> = request
                                .uri()
                                .query()
                                .map(|q| {
                                    q.split('&')
                                        .filter_map(|pair| {
                                            let mut parts = pair.splitn(2, '=');
                                            let key = parts.next()?;
                                            let value = parts.next().unwrap_or("");
                                            // URL decode key and value
                                            use percent_encoding::percent_decode_str;
                                            let key = percent_decode_str(key).decode_utf8_lossy().into_owned();
                                            let value = percent_decode_str(value).decode_utf8_lossy().into_owned();
                                            Some((key, value))
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();

                            let headers: Vec<(String, String)> = request
                                .headers()
                                .iter()
                                .map(|(k, v)| {
                                    (k.to_string(), v.to_str().unwrap_or("").to_string())
                                })
                                .collect();

                            // Parse cookies from Cookie header
                            let cookies: Vec<(String, String)> = headers
                                .iter()
                                .filter(|(k, _)| k.eq_ignore_ascii_case("cookie"))
                                .flat_map(|(_, v)| {
                                    v.split(';').filter_map(|cookie| {
                                        let cookie = cookie.trim();
                                        let mut parts = cookie.splitn(2, '=');
                                        let name = parts.next()?.trim();
                                        let value = parts.next().unwrap_or("").trim();
                                        if name.is_empty() {
                                            None
                                        } else {
                                            Some((name.to_string(), value.to_string()))
                                        }
                                    })
                                })
                                .collect();

                            // Check if this is a WebSocket upgrade request
                            let is_websocket = headers.iter().any(|(k, v)| {
                                k.eq_ignore_ascii_case("upgrade") && v.eq_ignore_ascii_case("websocket")
                            }) && headers.iter().any(|(k, v)| {
                                k.eq_ignore_ascii_case("connection") && v.to_lowercase().contains("upgrade")
                            });

                            // Get sec-websocket-key for WebSocket handshake
                            let ws_key = headers.iter()
                                .find(|(k, _)| k.eq_ignore_ascii_case("sec-websocket-key"))
                                .map(|(_, v)| v.clone());

                            // For WebSocket requests, extract the OnUpgrade handle BEFORE reading body
                            let on_upgrade = if is_websocket {
                                request.extensions_mut().remove::<hyper::upgrade::OnUpgrade>()
                            } else {
                                None
                            };

                            // Read body (empty for WebSocket requests)
                            let body = if is_websocket {
                                vec![]
                            } else {
                                match axum::body::to_bytes(request.into_body(), usize::MAX).await {
                                    Ok(bytes) => bytes.to_vec(),
                                    Err(_) => vec![],
                                }
                            };

                            // Parse form body if Content-Type is application/x-www-form-urlencoded
                            let form_params: Vec<(String, String)> = {
                                let content_type = headers.iter()
                                    .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
                                    .map(|(_, v)| v.as_str())
                                    .unwrap_or("");

                                if content_type.starts_with("application/x-www-form-urlencoded") {
                                    if let Ok(body_str) = std::str::from_utf8(&body) {
                                        use percent_encoding::percent_decode_str;
                                        body_str.split('&')
                                            .filter_map(|pair| {
                                                let mut parts = pair.splitn(2, '=');
                                                let key = parts.next()?;
                                                let value = parts.next().unwrap_or("");
                                                let key = percent_decode_str(key).decode_utf8_lossy().into_owned();
                                                let value = percent_decode_str(value).decode_utf8_lossy().into_owned();
                                                Some((key, value))
                                            })
                                            .collect()
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    vec![]
                                }
                            };

                            // Generate request ID and create response channel
                            let request_id = req_id_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let (resp_tx, resp_rx) = oneshot::channel();

                            // Store the response channel
                            pending.lock().await.insert(request_id, resp_tx);

                            // Store OnUpgrade for WebSocket requests
                            if let Some(upgrade) = on_upgrade {
                                pending_upgrades.lock().await.insert(request_id, upgrade);
                            }

                            // Send request to the Nostos process
                            let _ = req_tx.send((request_id, method, path, headers, body, query_params, cookies, form_params, is_websocket));

                            // Wait for response from Nostos
                            match resp_rx.await {
                                Ok((status, resp_headers, body)) => {
                                    // Check if this is a WebSocket upgrade response (status 101)
                                    if status == 101 && is_websocket {
                                        // Complete the WebSocket upgrade
                                        if let Some(upgrade) = pending_upgrades.lock().await.remove(&request_id) {
                                            // Calculate the WebSocket accept key
                                            let accept_key = if let Some(ref key) = ws_key {
                                                use sha1::{Sha1, Digest};
                                                let mut hasher = Sha1::new();
                                                hasher.update(key.as_bytes());
                                                hasher.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
                                                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, hasher.finalize())
                                            } else {
                                                String::new()
                                            };

                                            // Spawn task to complete upgrade and store WebSocket
                                            let ws_conns = ws_conns.clone();
                                            tokio::spawn(async move {
                                                match upgrade.await {
                                                    Ok(upgraded) => {
                                                        // Create WebSocket from upgraded connection using tokio-tungstenite
                                                        use tokio_tungstenite::WebSocketStream;
                                                        let io = hyper_util::rt::TokioIo::new(upgraded);
                                                        let ws = WebSocketStream::from_raw_socket(
                                                            io,
                                                            tokio_tungstenite::tungstenite::protocol::Role::Server,
                                                            None,
                                                        ).await;
                                                        let (sender, receiver) = ws.split();
                                                        ws_conns.lock().await.insert(request_id, (sender, receiver));
                                                    }
                                                    Err(e) => {
                                                        eprintln!("WebSocket upgrade failed: {}", e);
                                                    }
                                                }
                                            });

                                            // Return 101 response with WebSocket headers
                                            return axum::response::Response::builder()
                                                .status(101)
                                                .header("Upgrade", "websocket")
                                                .header("Connection", "Upgrade")
                                                .header("Sec-WebSocket-Accept", accept_key)
                                                .body(axum::body::Body::empty())
                                                .unwrap();
                                        }
                                    }

                                    // Normal HTTP response
                                    let mut response = axum::response::Response::builder()
                                        .status(status);
                                    for (key, value) in resp_headers {
                                        response = response.header(key, value);
                                    }
                                    response.body(axum::body::Body::from(body)).unwrap()
                                }
                                Err(_) => {
                                    // Request was abandoned - clean up pending upgrade if any
                                    pending_upgrades.lock().await.remove(&request_id);
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
                            // Spawn the server and store its abort handle for cleanup
                            let server_task = tokio::spawn(async move {
                                axum::serve(listener, app).await.ok();
                            });
                            server_abort_handles.lock().await.insert(handle, server_task.abort_handle());
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
                                    Some((request_id, method, path, headers, body, query_params, cookies, form_params, is_websocket)) => {
                                        let _ = response.send(Ok(IoResponseValue::ServerRequest {
                                            request_id,
                                            method,
                                            path,
                                            headers,
                                            body,
                                            query_params,
                                            cookies,
                                            form_params,
                                            is_websocket,
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
                    // Remove the receiver and abort the server task to release the port
                    server_request_receivers.lock().await.remove(&handle);
                    if let Some(abort_handle) = server_abort_handles.lock().await.remove(&handle) {
                        abort_handle.abort();
                    }
                    let _ = response.send(Ok(IoResponseValue::Unit));
                }

                // WebSocket operations
                IoRequest::WebSocketAccept { request_id, response } => {
                    // Send 101 response to trigger WebSocket upgrade
                    let mut pending = pending_responses.lock().await;
                    match pending.remove(&request_id) {
                        Some(resp_tx) => {
                            // Send empty 101 response - the axum handler will add WebSocket headers
                            let _ = resp_tx.send((101, vec![], vec![]));
                            drop(pending);

                            // Wait for WebSocket connection to be established
                            let ws_conns = ws_connections.clone();
                            tokio::spawn(async move {
                                // Poll for connection with timeout
                                for _ in 0..100 {
                                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                                    if ws_conns.lock().await.contains_key(&request_id) {
                                        let _ = response.send(Ok(IoResponseValue::Int(request_id as i64)));
                                        return;
                                    }
                                }
                                let _ = response.send(Err(IoError::IoError("WebSocket upgrade timeout".to_string())));
                            });
                        }
                        None => {
                            let _ = response.send(Err(IoError::Other("Request not found or not a WebSocket request".to_string())));
                        }
                    }
                }

                IoRequest::WebSocketSend { request_id, message, response } => {
                    let ws_conns = ws_connections.clone();
                    tokio::spawn(async move {
                        let mut conns = ws_conns.lock().await;
                        if let Some((sender, _)) = conns.get_mut(&request_id) {
                            match sender.send(WsMessage::Text(message.into())).await {
                                Ok(()) => {
                                    // Flush to ensure message is sent immediately
                                    use futures::SinkExt;
                                    if let Err(e) = sender.flush().await {
                                        let _ = response.send(Err(IoError::IoError(format!("WebSocket flush failed: {}", e))));
                                        return;
                                    }
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("WebSocket send failed: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::IoError("WebSocket connection not found".to_string())));
                        }
                    });
                }

                IoRequest::WebSocketReceive { request_id, response } => {
                    let ws_conns = ws_connections.clone();
                    tokio::spawn(async move {
                        // Take the connection out of the map to avoid holding the lock during recv
                        let conn = {
                            let mut conns = ws_conns.lock().await;
                            conns.remove(&request_id)
                        };
                        // Lock is now released - safe to await on receiver
                        if let Some((sender, mut receiver)) = conn {
                            match receiver.next().await {
                                Some(Ok(msg)) => {
                                    let text: String = match msg {
                                        WsMessage::Text(t) => t.to_string(),
                                        WsMessage::Binary(b) => String::from_utf8_lossy(&b).to_string(),
                                        WsMessage::Close(_) => {
                                            let _ = response.send(Err(IoError::IoError("WebSocket closed".to_string())));
                                            return;
                                        }
                                        _ => String::new(),
                                    };
                                    // Put the connection back for future operations
                                    ws_conns.lock().await.insert(request_id, (sender, receiver));
                                    let _ = response.send(Ok(IoResponseValue::String(text)));
                                }
                                Some(Err(e)) => {
                                    let _ = response.send(Err(IoError::IoError(format!("WebSocket receive failed: {}", e))));
                                }
                                None => {
                                    let _ = response.send(Err(IoError::IoError("WebSocket connection closed".to_string())));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::IoError("WebSocket connection not found".to_string())));
                        }
                    });
                }

                IoRequest::WebSocketClose { request_id, response } => {
                    let ws_conns = ws_connections.clone();
                    tokio::spawn(async move {
                        let mut conns = ws_conns.lock().await;
                        if let Some((mut sender, _)) = conns.remove(&request_id) {
                            let _ = sender.send(WsMessage::Close(None)).await;
                            let _ = sender.close().await;
                            let _ = response.send(Ok(IoResponseValue::Unit));
                        } else {
                            let _ = response.send(Ok(IoResponseValue::Unit)); // Already closed, that's OK
                        }
                    });
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

                // PostgreSQL operations (with connection pooling)
                IoRequest::PgConnect { connection_string, response } => {
                    let pg_conns = pg_connections.clone();
                    let pools = pg_pools.clone();
                    let handle = next_pg_handle;
                    next_pg_handle += 1;

                    tokio::spawn(async move {
                        let use_tls = requires_tls(&connection_string);

                        // Get or create pool for this connection string
                        let pool_result = {
                            let mut pools_guard = pools.lock().await;
                            if !pools_guard.contains_key(&connection_string) {
                                // Parse connection string using tokio_postgres (handles both URL and key=value formats)
                                let pg_config: tokio_postgres::Config = match connection_string.parse() {
                                    Ok(c) => c,
                                    Err(e) => {
                                        let _ = response.send(Err(IoError::PgError(format!("Invalid connection string: {}", e))));
                                        return;
                                    }
                                };

                                let mgr_config = ManagerConfig {
                                    recycling_method: RecyclingMethod::Fast,
                                };

                                let pool_create_result: Result<PgPool, IoError> = if use_tls {
                                    match create_tls_connector() {
                                        Ok(tls) => {
                                            let mgr = PgPoolManager::from_config(pg_config, tls, mgr_config);
                                            PgPool::builder(mgr)
                                                .runtime(DeadpoolRuntime::Tokio1)
                                                .build()
                                                .map_err(|e| IoError::PgError(format!("Pool creation error: {}", e)))
                                        }
                                        Err(e) => Err(IoError::PgError(format!("TLS error: {}", e))),
                                    }
                                } else {
                                    let mgr = PgPoolManager::from_config(pg_config, NoTls, mgr_config);
                                    PgPool::builder(mgr)
                                        .runtime(DeadpoolRuntime::Tokio1)
                                        .build()
                                        .map_err(|e| IoError::PgError(format!("Pool creation error: {}", e)))
                                };

                                match pool_create_result {
                                    Ok(pool) => {
                                        pools_guard.insert(connection_string.clone(), pool);
                                        Ok(())
                                    }
                                    Err(e) => Err(e),
                                }
                            } else {
                                Ok(())
                            }
                        };

                        if let Err(e) = pool_result {
                            let _ = response.send(Err(e));
                            return;
                        }

                        // Get connection from pool
                        let conn_result = {
                            let pools_guard = pools.lock().await;
                            let pool = pools_guard.get(&connection_string).unwrap();
                            pool.get().await
                                .map(PgPooledConnection)
                                .map_err(|e| IoError::PgError(format!("Pool get error: {}", e)))
                        };

                        match conn_result {
                            Ok(pooled_conn) => {
                                let pg_conn = PgConnection::Pooled {
                                    conn: pooled_conn,
                                    prepared: HashMap::new(),
                                };
                                pg_conns.lock().await.insert(handle, pg_conn);
                                let _ = response.send(Ok(IoResponseValue::PgHandle(handle)));
                            }
                            Err(e) => {
                                let _ = response.send(Err(e));
                            }
                        }
                    });
                }

                IoRequest::PgQuery { handle, query, params, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        match conns.get(&handle) {
                            Some(conn) => {
                                // Build params - we need to keep ownership of the boxed values
                                let result = Self::execute_pg_query(conn.client(), &query, &params).await;
                                match result {
                                    Ok(rows) => {
                                        let _ = response.send(Ok(IoResponseValue::PgRows(rows)));
                                    }
                                    Err(e) => {
                                        let _ = response.send(Err(IoError::PgError(e)));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::PgExecute { handle, query, params, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        match conns.get(&handle) {
                            Some(conn) => {
                                let result = Self::execute_pg_execute(conn.client(), &query, &params).await;
                                match result {
                                    Ok(count) => {
                                        let _ = response.send(Ok(IoResponseValue::PgAffected(count)));
                                    }
                                    Err(e) => {
                                        let _ = response.send(Err(IoError::PgError(e)));
                                    }
                                }
                            }
                            None => {
                                let _ = response.send(Err(IoError::InvalidHandle));
                            }
                        }
                    });
                }

                IoRequest::PgClose { handle, response } => {
                    let mut conns = pg_connections.lock().await;
                    if conns.remove(&handle).is_some() {
                        let _ = response.send(Ok(IoResponseValue::Unit));
                    } else {
                        let _ = response.send(Err(IoError::InvalidHandle));
                    }
                }

                IoRequest::PgBegin { handle, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get(&handle) {
                            match conn.client().batch_execute("BEGIN").await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(e.to_string())));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgCommit { handle, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get(&handle) {
                            match conn.client().batch_execute("COMMIT").await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(e.to_string())));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgRollback { handle, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get(&handle) {
                            match conn.client().batch_execute("ROLLBACK").await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(e.to_string())));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgPrepare { handle, name, query, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let mut conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get_mut(&handle) {
                            match conn.client().prepare(&query).await {
                                Ok(stmt) => {
                                    conn.prepared_mut().insert(name, stmt);
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(e.to_string())));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgQueryPrepared { handle, name, params, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get(&handle) {
                            if let Some(stmt) = conn.prepared().get(&name) {
                                let result = Self::execute_pg_query_prepared(conn.client(), stmt, &params).await;
                                match result {
                                    Ok(rows) => {
                                        let _ = response.send(Ok(IoResponseValue::PgRows(rows)));
                                    }
                                    Err(e) => {
                                        let _ = response.send(Err(IoError::PgError(e)));
                                    }
                                }
                            } else {
                                let _ = response.send(Err(IoError::PgError(format!("Prepared statement '{}' not found", name))));
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgExecutePrepared { handle, name, params, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get(&handle) {
                            if let Some(stmt) = conn.prepared().get(&name) {
                                let result = Self::execute_pg_execute_prepared(conn.client(), stmt, &params).await;
                                match result {
                                    Ok(count) => {
                                        let _ = response.send(Ok(IoResponseValue::PgAffected(count)));
                                    }
                                    Err(e) => {
                                        let _ = response.send(Err(IoError::PgError(e)));
                                    }
                                }
                            } else {
                                let _ = response.send(Err(IoError::PgError(format!("Prepared statement '{}' not found", name))));
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgDeallocate { handle, name, response } => {
                    let pg_conns = pg_connections.clone();

                    tokio::spawn(async move {
                        let mut conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get_mut(&handle) {
                            if conn.prepared_mut().remove(&name).is_some() {
                                let _ = response.send(Ok(IoResponseValue::Unit));
                            } else {
                                let _ = response.send(Err(IoError::PgError(format!("Prepared statement '{}' not found", name))));
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                // LISTEN/NOTIFY operations
                IoRequest::PgListenConnect { connection_string, response } => {
                    let listeners = pg_listeners.clone();
                    let handle = next_listener_handle;
                    next_listener_handle += 1;

                    tokio::spawn(async move {
                        let use_tls = requires_tls(&connection_string);

                        // Create channel for notifications
                        let (tx, rx) = mpsc::unbounded_channel();

                        // Create a direct (non-pooled) connection for listening
                        // Handle TLS and non-TLS separately due to different types
                        let client_result = if use_tls {
                            match create_tls_connector() {
                                Ok(tls) => {
                                    match tokio_postgres::connect(&connection_string, tls).await {
                                        Ok((client, mut connection)) => {
                                            let tx = tx.clone();
                                            // Spawn connection handler using poll_message
                                            tokio::spawn(async move {
                                                loop {
                                                    let message = poll_fn(|cx| connection.poll_message(cx)).await;
                                                    match message {
                                                        Some(Ok(AsyncMessage::Notification(n))) => {
                                                            let _ = tx.send((
                                                                n.channel().to_string(),
                                                                n.payload().to_string(),
                                                            ));
                                                        }
                                                        Some(Ok(_)) => {}
                                                        Some(Err(_)) | None => break,
                                                    }
                                                }
                                            });
                                            Ok(client)
                                        }
                                        Err(e) => Err(IoError::PgError(format!("Connection error: {}", e))),
                                    }
                                }
                                Err(e) => Err(IoError::PgError(format!("TLS error: {}", e))),
                            }
                        } else {
                            match tokio_postgres::connect(&connection_string, NoTls).await {
                                Ok((client, mut connection)) => {
                                    let tx = tx.clone();
                                    // Spawn connection handler using poll_message
                                    tokio::spawn(async move {
                                        loop {
                                            let message = poll_fn(|cx| connection.poll_message(cx)).await;
                                            match message {
                                                Some(Ok(AsyncMessage::Notification(n))) => {
                                                    let _ = tx.send((
                                                        n.channel().to_string(),
                                                        n.payload().to_string(),
                                                    ));
                                                }
                                                Some(Ok(_)) => {}
                                                Some(Err(_)) | None => break,
                                            }
                                        }
                                    });
                                    Ok(client)
                                }
                                Err(e) => Err(IoError::PgError(format!("Connection error: {}", e))),
                            }
                        };

                        match client_result {
                            Ok(client) => {
                                let listener_conn = PgListenerConnection {
                                    client,
                                    notification_rx: TokioMutex::new(rx),
                                };
                                listeners.lock().await.insert(handle, listener_conn);
                                let _ = response.send(Ok(IoResponseValue::PgHandle(handle)));
                            }
                            Err(e) => {
                                let _ = response.send(Err(e));
                            }
                        }
                    });
                }

                IoRequest::PgListen { handle, channel, response } => {
                    let listeners = pg_listeners.clone();

                    tokio::spawn(async move {
                        let listeners_guard = listeners.lock().await;
                        if let Some(conn) = listeners_guard.get(&handle) {
                            // Send LISTEN command
                            let query = format!("LISTEN {}", Self::quote_identifier(&channel));
                            match conn.client.execute(&query, &[]).await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(format!("LISTEN error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgUnlisten { handle, channel, response } => {
                    let listeners = pg_listeners.clone();

                    tokio::spawn(async move {
                        let listeners_guard = listeners.lock().await;
                        if let Some(conn) = listeners_guard.get(&handle) {
                            // Send UNLISTEN command
                            let query = format!("UNLISTEN {}", Self::quote_identifier(&channel));
                            match conn.client.execute(&query, &[]).await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(format!("UNLISTEN error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgAwaitNotification { handle, timeout_ms, response } => {
                    let listeners = pg_listeners.clone();

                    tokio::spawn(async move {
                        // Wait for notification with timeout
                        // We hold the listeners lock for the entire operation to keep the reference valid
                        let listeners_guard = listeners.lock().await;
                        if let Some(conn) = listeners_guard.get(&handle) {
                            let mut rx = conn.notification_rx.lock().await;

                            let timeout = tokio::time::Duration::from_millis(timeout_ms);
                            match tokio::time::timeout(timeout, rx.recv()).await {
                                Ok(Some((channel, payload))) => {
                                    let _ = response.send(Ok(IoResponseValue::PgNotificationOption(
                                        Some((channel, payload))
                                    )));
                                }
                                Ok(None) => {
                                    // Channel closed (connection dropped)
                                    let _ = response.send(Ok(IoResponseValue::PgNotificationOption(None)));
                                }
                                Err(_) => {
                                    // Timeout
                                    let _ = response.send(Ok(IoResponseValue::PgNotificationOption(None)));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::PgNotify { handle, channel, payload, response } => {
                    let pg_conns = pg_connections.clone();
                    let listeners = pg_listeners.clone();

                    tokio::spawn(async move {
                        // First try regular connections, then listener connections
                        let conns = pg_conns.lock().await;
                        if let Some(conn) = conns.get(&handle) {
                            let query = format!(
                                "NOTIFY {}, {}",
                                Self::quote_identifier(&channel),
                                Self::quote_string(&payload)
                            );
                            match conn.client().execute(&query, &[]).await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(format!("NOTIFY error: {}", e))));
                                }
                            }
                            return;
                        }
                        drop(conns);

                        // Try listener connections
                        let listeners_guard = listeners.lock().await;
                        if let Some(conn) = listeners_guard.get(&handle) {
                            let query = format!(
                                "NOTIFY {}, {}",
                                Self::quote_identifier(&channel),
                                Self::quote_string(&payload)
                            );
                            match conn.client.execute(&query, &[]).await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::PgError(format!("NOTIFY error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                // === Selenium WebDriver Operations ===
                IoRequest::SeleniumConnect { webdriver_url, response } => {
                    let drivers = selenium_drivers.clone();
                    let handle = next_selenium_handle;
                    next_selenium_handle += 1;

                    tokio::spawn(async move {
                        match WebDriver::new(&webdriver_url, DesiredCapabilities::chrome()).await {
                            Ok(driver) => {
                                drivers.lock().await.insert(handle, driver);
                                let _ = response.send(Ok(IoResponseValue::Int(handle as i64)));
                            }
                            Err(e) => {
                                let _ = response.send(Err(IoError::IoError(format!("Selenium connect error: {}", e))));
                            }
                        }
                    });
                }

                IoRequest::SeleniumGoto { driver_handle, url, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            match driver.goto(&url).await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium goto error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumClick { driver_handle, selector, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            // Use JavaScript click for more reliable clicking
                            let script = format!(
                                r#"var el = document.querySelector("{}"); if (el) {{ el.click(); return true; }} return false;"#,
                                selector.replace('"', r#"\""#)
                            );
                            match driver.execute(&script, vec![]).await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium click error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumText { driver_handle, selector, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            let script = format!(
                                r#"return document.querySelector("{}")?.textContent || ''"#,
                                selector.replace('"', r#"\""#)
                            );
                            match driver.execute(&script, vec![]).await {
                                Ok(result) => {
                                    let text = result.json().as_str().unwrap_or("").to_string();
                                    let _ = response.send(Ok(IoResponseValue::String(text)));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium text error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumSendKeys { driver_handle, selector, text, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            match driver.find(By::Css(&selector)).await {
                                Ok(elem) => {
                                    match elem.send_keys(&text).await {
                                        Ok(_) => {
                                            let _ = response.send(Ok(IoResponseValue::Unit));
                                        }
                                        Err(e) => {
                                            let _ = response.send(Err(IoError::IoError(format!("Selenium sendKeys error: {}", e))));
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium find error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumExecuteJs { driver_handle, script, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            match driver.execute(&script, vec![]).await {
                                Ok(result) => {
                                    let text = result.json().to_string();
                                    let _ = response.send(Ok(IoResponseValue::String(text)));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium executeJs error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumExecuteJsWithArgs { driver_handle, script, args, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            // Convert args to serde_json::Value
                            let json_args: Vec<serde_json::Value> = args.iter()
                                .map(|s| serde_json::Value::String(s.clone()))
                                .collect();
                            match driver.execute(&script, json_args).await {
                                Ok(result) => {
                                    let text = result.json().to_string();
                                    let _ = response.send(Ok(IoResponseValue::String(text)));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium executeJs error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumWaitFor { driver_handle, selector, timeout_ms, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            let timeout = std::time::Duration::from_millis(timeout_ms);
                            let start = std::time::Instant::now();

                            loop {
                                let script = format!(
                                    r#"return document.querySelector("{}") !== null"#,
                                    selector.replace('"', r#"\""#)
                                );
                                match driver.execute(&script, vec![]).await {
                                    Ok(result) => {
                                        if result.json().as_bool().unwrap_or(false) {
                                            let _ = response.send(Ok(IoResponseValue::Bool(true)));
                                            return;
                                        }
                                    }
                                    Err(_) => {}
                                }

                                if start.elapsed() >= timeout {
                                    let _ = response.send(Ok(IoResponseValue::Bool(false)));
                                    return;
                                }

                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumGetAttribute { driver_handle, selector, attribute, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            let script = format!(
                                r#"var el = document.querySelector("{}"); return el ? el.getAttribute("{}") : null"#,
                                selector.replace('"', r#"\""#),
                                attribute.replace('"', r#"\""#)
                            );
                            match driver.execute(&script, vec![]).await {
                                Ok(result) => {
                                    let text = result.json().as_str().unwrap_or("").to_string();
                                    let _ = response.send(Ok(IoResponseValue::String(text)));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium getAttribute error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumExists { driver_handle, selector, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.get(&driver_handle) {
                            let script = format!(
                                r#"return document.querySelector("{}") !== null"#,
                                selector.replace('"', r#"\""#)
                            );
                            match driver.execute(&script, vec![]).await {
                                Ok(result) => {
                                    let exists = result.json().as_bool().unwrap_or(false);
                                    let _ = response.send(Ok(IoResponseValue::Bool(exists)));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium exists error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }

                IoRequest::SeleniumClose { driver_handle, response } => {
                    let drivers = selenium_drivers.clone();

                    tokio::spawn(async move {
                        let mut drivers_guard = drivers.lock().await;
                        if let Some(driver) = drivers_guard.remove(&driver_handle) {
                            match driver.quit().await {
                                Ok(_) => {
                                    let _ = response.send(Ok(IoResponseValue::Unit));
                                }
                                Err(e) => {
                                    let _ = response.send(Err(IoError::IoError(format!("Selenium close error: {}", e))));
                                }
                            }
                        } else {
                            let _ = response.send(Err(IoError::InvalidHandle));
                        }
                    });
                }
            }
        }
    }

    /// Quote a PostgreSQL identifier (for LISTEN/UNLISTEN channel names)
    fn quote_identifier(s: &str) -> String {
        format!("\"{}\"", s.replace('"', "\"\""))
    }

    /// Quote a PostgreSQL string literal (for NOTIFY payload)
    fn quote_string(s: &str) -> String {
        format!("'{}'", s.replace('\'', "''"))
    }

    /// Convert a row column value to PgValue
    fn row_value_to_pg_value(row: &tokio_postgres::Row, idx: usize) -> PgValue {
        use tokio_postgres::types::Type;

        let col_type = row.columns()[idx].type_();

        // Try each type in order of likelihood
        match *col_type {
            Type::BOOL => {
                match row.try_get::<_, bool>(idx) {
                    Ok(v) => PgValue::Bool(v),
                    Err(_) => PgValue::Null,
                }
            }
            Type::INT2 => {
                match row.try_get::<_, i16>(idx) {
                    Ok(v) => PgValue::Int(v as i64),
                    Err(_) => PgValue::Null,
                }
            }
            Type::INT4 => {
                match row.try_get::<_, i32>(idx) {
                    Ok(v) => PgValue::Int(v as i64),
                    Err(_) => PgValue::Null,
                }
            }
            Type::INT8 => {
                match row.try_get::<_, i64>(idx) {
                    Ok(v) => PgValue::Int(v),
                    Err(_) => PgValue::Null,
                }
            }
            Type::FLOAT4 => {
                match row.try_get::<_, f32>(idx) {
                    Ok(v) => PgValue::Float(v as f64),
                    Err(_) => PgValue::Null,
                }
            }
            Type::FLOAT8 => {
                match row.try_get::<_, f64>(idx) {
                    Ok(v) => PgValue::Float(v),
                    Err(_) => PgValue::Null,
                }
            }
            Type::TEXT | Type::VARCHAR | Type::NAME | Type::CHAR | Type::BPCHAR => {
                match row.try_get::<_, String>(idx) {
                    Ok(v) => PgValue::String(v),
                    Err(_) => PgValue::Null,
                }
            }
            Type::BYTEA => {
                match row.try_get::<_, Vec<u8>>(idx) {
                    Ok(v) => PgValue::Bytes(v),
                    Err(_) => PgValue::Null,
                }
            }
            Type::TIMESTAMP => {
                // TIMESTAMP without timezone - stored as NaiveDateTime
                use chrono::NaiveDateTime;
                match row.try_get::<_, NaiveDateTime>(idx) {
                    Ok(v) => PgValue::Timestamp(v.and_utc().timestamp_millis()),
                    Err(_) => PgValue::Null,
                }
            }
            Type::TIMESTAMPTZ => {
                // TIMESTAMP WITH TIME ZONE - stored as DateTime<Utc>
                use chrono::{DateTime, Utc};
                match row.try_get::<_, DateTime<Utc>>(idx) {
                    Ok(v) => PgValue::Timestamp(v.timestamp_millis()),
                    Err(_) => PgValue::Null,
                }
            }
            Type::DATE => {
                // DATE - stored as NaiveDate, convert to millis at midnight UTC
                use chrono::NaiveDate;
                match row.try_get::<_, NaiveDate>(idx) {
                    Ok(v) => {
                        let midnight = v.and_hms_opt(0, 0, 0).unwrap();
                        PgValue::Timestamp(midnight.and_utc().timestamp_millis())
                    }
                    Err(_) => PgValue::Null,
                }
            }
            Type::TIME => {
                // TIME - stored as NaiveTime, convert to millis since midnight
                use chrono::NaiveTime;
                match row.try_get::<_, NaiveTime>(idx) {
                    Ok(v) => {
                        let millis = v.num_seconds_from_midnight() as i64 * 1000
                            + (v.nanosecond() / 1_000_000) as i64;
                        PgValue::Timestamp(millis)
                    }
                    Err(_) => PgValue::Null,
                }
            }
            Type::TIMETZ => {
                // TIME WITH TIME ZONE - try as string for now (complex type)
                match row.try_get::<_, String>(idx) {
                    Ok(v) => PgValue::String(v),
                    Err(_) => PgValue::Null,
                }
            }
            Type::JSON | Type::JSONB => {
                // JSON/JSONB - get as string representation
                match row.try_get::<_, serde_json::Value>(idx) {
                    Ok(v) => PgValue::Json(v.to_string()),
                    Err(_) => PgValue::Null,
                }
            }
            _ => {
                // Check for vector type by name (pgvector extension)
                if col_type.name() == "vector" {
                    // Vector type - decode pgvector binary format directly
                    // pgvector format: 2 bytes dim (u16), 2 bytes unused, n*4 bytes f32 (big endian)
                    match row.try_get::<_, PgVector>(idx) {
                        Ok(v) => PgValue::Vector(v.0),
                        Err(_) => PgValue::Null,
                    }
                } else {
                    // For other types, try to get as string
                    match row.try_get::<_, String>(idx) {
                        Ok(v) => PgValue::String(v),
                        Err(_) => PgValue::Null,
                    }
                }
            }
        }
    }

    /// Add type casts to SQL based on param types
    fn add_type_casts(query: &str, params: &[PgParam]) -> String {
        let mut result = query.to_string();
        // Replace $N with $N::type based on param type (in reverse order to not mess up indices)
        for (i, param) in params.iter().enumerate().rev() {
            let placeholder = format!("${}", i + 1);
            let typed_placeholder = match param {
                PgParam::Null => format!("{}::text", placeholder), // NULL needs a type for PostgreSQL
                PgParam::Bool(_) => format!("{}::bool", placeholder),
                PgParam::Int(_) => format!("{}::int8", placeholder),
                PgParam::Float(_) => format!("{}::float8", placeholder),
                PgParam::String(_) => format!("{}::text", placeholder),
                PgParam::Timestamp(_) => format!("{}::timestamptz", placeholder),
                PgParam::Json(_) => format!("{}::jsonb", placeholder),
                PgParam::Vector(_) => format!("{}::text::vector", placeholder),
            };
            // Only replace if not already typed (check for ::)
            let check_pattern = format!("{}::", placeholder);
            if !result.contains(&check_pattern) {
                result = result.replace(&placeholder, &typed_placeholder);
            }
        }
        result
    }

    /// Execute a Postgres query with parameters
    async fn execute_pg_query(
        client: &PgClient,
        query: &str,
        params: &[PgParam],
    ) -> Result<Vec<Vec<PgValue>>, String> {
        let rows = if params.is_empty() {
            client.query(query, &[]).await
        } else {
            // Add type casts based on param types
            let typed_query = Self::add_type_casts(query, params);

            // Prepare statement
            let stmt = match client.prepare(&typed_query).await {
                Ok(s) => s,
                Err(e) => return Err(e.to_string()),
            };

            // Build typed params based on what PostgreSQL expects
            let typed_params = Self::build_typed_params(params, stmt.params());
            let param_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
                typed_params.iter().map(|p| &**p as &(dyn tokio_postgres::types::ToSql + Sync)).collect();

            client.query(&stmt, &param_refs).await
        };

        match rows {
            Ok(rows) => {
                let result: Vec<Vec<PgValue>> = rows.iter().map(|row| {
                    (0..row.len()).map(|i| Self::row_value_to_pg_value(row, i)).collect()
                }).collect();
                Ok(result)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Execute a Postgres statement with parameters
    async fn execute_pg_execute(
        client: &PgClient,
        query: &str,
        params: &[PgParam],
    ) -> Result<u64, String> {
        let count = if params.is_empty() {
            client.execute(query, &[]).await
        } else {
            // Add type casts based on param types
            let typed_query = Self::add_type_casts(query, params);

            let stmt = match client.prepare(&typed_query).await {
                Ok(s) => s,
                Err(e) => return Err(e.to_string()),
            };

            let typed_params = Self::build_typed_params(params, stmt.params());
            let param_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
                typed_params.iter().map(|p| &**p as &(dyn tokio_postgres::types::ToSql + Sync)).collect();

            client.execute(&stmt, &param_refs).await
        };

        count.map_err(|e| e.to_string())
    }

    /// Execute a prepared query with parameters
    async fn execute_pg_query_prepared(
        client: &PgClient,
        stmt: &Statement,
        params: &[PgParam],
    ) -> Result<Vec<Vec<PgValue>>, String> {
        let typed_params = Self::build_typed_params(params, stmt.params());
        let param_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
            typed_params.iter().map(|p| &**p as &(dyn tokio_postgres::types::ToSql + Sync)).collect();

        match client.query(stmt, &param_refs).await {
            Ok(rows) => {
                let result: Vec<Vec<PgValue>> = rows.iter().map(|row| {
                    (0..row.len()).map(|i| Self::row_value_to_pg_value(row, i)).collect()
                }).collect();
                Ok(result)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Execute a prepared statement with parameters
    async fn execute_pg_execute_prepared(
        client: &PgClient,
        stmt: &Statement,
        params: &[PgParam],
    ) -> Result<u64, String> {
        let typed_params = Self::build_typed_params(params, stmt.params());
        let param_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
            typed_params.iter().map(|p| &**p as &(dyn tokio_postgres::types::ToSql + Sync)).collect();

        client.execute(stmt, &param_refs).await.map_err(|e| e.to_string())
    }

    /// Build typed parameters that match PostgreSQL's expected types
    fn build_typed_params(params: &[PgParam], types: &[tokio_postgres::types::Type]) -> Vec<Box<dyn tokio_postgres::types::ToSql + Sync + Send>> {
        use tokio_postgres::types::Type;

        params.iter().enumerate().map(|(i, p)| {
            let expected_type = types.get(i);
            match p {
                PgParam::Null => {
                    // Use Option<String> for NULL - works with any type
                    Box::new(None::<String>) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                }
                PgParam::Bool(b) => {
                    Box::new(*b) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                }
                PgParam::Int(i) => {
                    // Convert to the expected integer type, use text for unknown
                    match expected_type {
                        Some(t) if *t == Type::INT2 => Box::new(*i as i16) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        Some(t) if *t == Type::INT4 => Box::new(*i as i32) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        Some(t) if *t == Type::INT8 => Box::new(*i) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        Some(t) if *t == Type::FLOAT4 => Box::new(*i as f32) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        Some(t) if *t == Type::FLOAT8 => Box::new(*i as f64) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        // For TEXT, VARCHAR, or UNKNOWN types, send as string
                        _ => Box::new(i.to_string()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                    }
                }
                PgParam::Float(f) => {
                    match expected_type {
                        Some(t) if *t == Type::FLOAT4 => Box::new(*f as f32) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        Some(t) if *t == Type::FLOAT8 => Box::new(*f) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        // For unknown types, send as string
                        _ => Box::new(f.to_string()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                    }
                }
                PgParam::String(s) => {
                    // Check if expected type is JSON/JSONB
                    match expected_type {
                        Some(t) if *t == Type::JSON || *t == Type::JSONB => {
                            // Parse string as JSON and send as serde_json::Value
                            match serde_json::from_str::<serde_json::Value>(s) {
                                Ok(v) => Box::new(v) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                                Err(_) => Box::new(s.clone()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                            }
                        }
                        _ => {
                            // Strings can be used for any type - PostgreSQL will convert
                            Box::new(s.clone()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                        }
                    }
                }
                PgParam::Timestamp(millis) => {
                    use chrono::{DateTime, NaiveTime};
                    // Convert millis since epoch to appropriate chrono type based on expected type
                    match expected_type {
                        Some(t) if *t == Type::DATE => {
                            // Convert millis to NaiveDate
                            let dt = DateTime::from_timestamp_millis(*millis)
                                .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                            Box::new(dt.date_naive()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                        }
                        Some(t) if *t == Type::TIME => {
                            // Millis since midnight to NaiveTime
                            let secs = (*millis / 1000) as u32;
                            let nanos = ((*millis % 1000) * 1_000_000) as u32;
                            let time = NaiveTime::from_num_seconds_from_midnight_opt(secs, nanos)
                                .unwrap_or_else(|| NaiveTime::from_hms_opt(0, 0, 0).unwrap());
                            Box::new(time) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                        }
                        Some(t) if *t == Type::TIMESTAMP => {
                            // Convert to NaiveDateTime (no timezone)
                            let dt = DateTime::from_timestamp_millis(*millis)
                                .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                            Box::new(dt.naive_utc()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                        }
                        _ => {
                            // Default to TIMESTAMPTZ (DateTime<Utc>)
                            let dt = DateTime::from_timestamp_millis(*millis)
                                .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
                            Box::new(dt) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                        }
                    }
                }
                PgParam::Json(s) => {
                    // Send JSON as serde_json::Value for proper JSONB handling
                    match serde_json::from_str::<serde_json::Value>(s) {
                        Ok(v) => Box::new(v) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                        // Fallback to string if parsing fails
                        Err(_) => Box::new(s.clone()) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>,
                    }
                }
                PgParam::Vector(v) => {
                    // Send vector as text representation for pgvector
                    // Format: "[1.0, 2.0, 3.0]"
                    let text = format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","));
                    Box::new(text) as Box<dyn tokio_postgres::types::ToSql + Sync + Send>
                }
            }
        }).collect()
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    async fn handle_file_read_all(file: &mut tokio::fs::File) -> IoResult<Vec<u8>> {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        match file.read_to_end(&mut buf).await {
            Ok(_) => Ok(buf),
            Err(e) => Err(IoError::IoError(e.to_string())),
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
