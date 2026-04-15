# LSP Completion Speedup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce dot-completion latency in nostos-lsp from multi-second delays to sub-200ms response times.

**Architecture:** The root cause is lock contention: every keystroke triggers `didChange` which locks the engine mutex for a full parse+typecheck (`check_module_compiles`), blocking concurrent completion requests. We fix this in three phases: (1) debounce `didChange` to stop competing for the lock, (2) remove debug logging noise, (3) optimize the engine API methods called during completion.

**Tech Stack:** Rust, tower-lsp 0.20, tokio (multi-thread), dashmap

**Codebase:** `~/git/wego/nostos/crates/lsp/`

---

## Phase 1: Debounce `didChange` processing

The single biggest win. Currently every keystroke locks the engine twice (once for `send_file_status_notification`, once for `check_file`). When the user types `state.`, the completion request has to wait for the `check_file` from the `.` keystroke to finish.

### Task 1: Debounce `check_file` in `didChange`

**Files:**
- Modify: `crates/lsp/src/server.rs` (struct `NostosLanguageServer`, `did_change` method)

- [ ] **Step 1: Add a debounce token field to the server struct**

At `crates/lsp/src/server.rs:42`, add a `check_debounce` field to `NostosLanguageServer`:

```rust
use tokio::sync::Notify;
use std::sync::atomic::AtomicU64;

pub struct NostosLanguageServer {
    client: Client,
    engine: Mutex<Option<ReplEngine>>,
    documents: DashMap<Url, String>,
    file_errors: DashMap<Url, Vec<Diagnostic>>,
    root_path: Mutex<Option<PathBuf>>,
    initializing: AtomicBool,
    dirty_files: DashMap<String, bool>,
    check_generation: AtomicU64,  // <-- add this
}
```

And initialize it in `new()`:

```rust
check_generation: AtomicU64::new(0),
```

- [ ] **Step 2: Replace the synchronous `check_file` call in `did_change` with a debounced version**

At `crates/lsp/src/server.rs:1074`, change the `did_change` method. Replace the direct call to `self.check_file(&uri, &content).await` with a debounced pattern:

```rust
async fn did_change(&self, params: DidChangeTextDocumentParams) {
    let uri = params.text_document.uri;

    if let Some(change) = params.content_changes.into_iter().last() {
        let content = change.text;

        // Update stored content
        self.documents.insert(uri.clone(), content.clone());

        // Mark file as dirty
        if let Ok(file_path) = uri.to_file_path() {
            self.dirty_files.insert(file_path.to_string_lossy().to_string(), true);
        }

        // Bump generation to cancel any pending check
        let gen = self.check_generation.fetch_add(1, Ordering::SeqCst) + 1;

        // Wait 300ms, then check if we're still the latest keystroke
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        if self.check_generation.load(Ordering::SeqCst) != gen {
            return; // A newer keystroke arrived, skip this check
        }

        // Send file status notification (debounced together with check)
        self.send_file_status_notification().await;

        // Now do the actual check
        self.check_file(&uri, &content).await;
    }
}
```

- [ ] **Step 3: Update imports**

At the top of `crates/lsp/src/server.rs`, make sure `AtomicU64` is imported alongside `AtomicBool`:

```rust
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
```

- [ ] **Step 4: Build and verify**

Run:
```bash
cd ~/git/wego/nostos && cargo build -p nostos-lsp
```

Expected: compiles with no errors.

- [ ] **Step 5: Manual test**

1. In the intellij-nostos project, run `gradle runIde`.
2. Open `~/git/wego/trivium-client/nostos`.
3. Open `example.nos`, go to line 116, type `state.`.
4. Observe that completions arrive noticeably faster (no `check_file` contention).
5. Verify that error diagnostics still appear after you stop typing for ~300ms.

- [ ] **Step 6: Commit**

```bash
cd ~/git/wego/nostos
git add crates/lsp/src/server.rs
git commit -m "Debounce check_file in didChange to avoid lock contention with completion"
```

---

## Phase 2: Replace ad-hoc logging with the `log` crate

There are **89 `eprintln!` calls** and **16 file writes to `/tmp/nostos_lsp_debug.log`** in `server.rs`, plus dual-logging (stderr + file) in both `main.rs` and `server.rs`. These are development artifacts that pollute stderr and do synchronous file I/O on every request.

Instead of deleting all logging, we migrate to the `log` crate with compile-time level filtering. In release builds, `debug!()` and `trace!()` calls are **stripped out entirely by the compiler** — zero overhead. In debug builds, they remain active for development.

### Task 2: Add `log` crate with compile-time filtering

**Files:**
- Modify: `crates/lsp/Cargo.toml`

- [ ] **Step 1: Add `log` and `env_logger` dependencies**

Add to `[dependencies]` in `crates/lsp/Cargo.toml`:

```toml
log = { version = "0.4", features = ["release_max_level_info"] }
env_logger = "0.11"
```

The `release_max_level_info` feature means: in release builds, all `debug!()` and `trace!()` calls are compiled to nothing. In debug builds, all levels are active.

- [ ] **Step 2: Initialize `env_logger` in `main.rs`**

Replace the contents of `crates/lsp/src/main.rs`:

```rust
#![allow(
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::unnecessary_map_or,
    clippy::type_complexity,
    clippy::ptr_arg,
    clippy::needless_lifetimes,
    clippy::needless_borrows_for_generic_args,
    clippy::clone_on_copy,
    clippy::collapsible_match,
    clippy::redundant_pattern_matching,
    clippy::manual_strip,
    clippy::match_result_ok,
    clippy::manual_pattern_char_comparison,
    clippy::writeln_empty_string,
    dead_code
)]

use tower_lsp::{LspService, Server};

mod server;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    )
    .target(env_logger::Target::Stderr)
    .init();

    // Set up panic handler
    std::panic::set_hook(Box::new(|panic_info| {
        log::error!("LSP PANIC: {}", panic_info);
    }));

    log::info!("Starting Nostos LSP server... PID={}", std::process::id());

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(server::NostosLanguageServer::new);

    Server::new(stdin, stdout, socket).serve(service).await;
}
```

This removes the custom `log` function, the file logging, and the log file cleanup.

- [ ] **Step 3: Build and verify**

Run:
```bash
cd ~/git/wego/nostos && cargo build -p nostos-lsp
```

Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
cd ~/git/wego/nostos
git add crates/lsp/Cargo.toml crates/lsp/src/main.rs
git commit -m "Add log crate with compile-time filtering for release builds"
```

### Task 3: Migrate `server.rs` from `eprintln!` to `log` macros

**Files:**
- Modify: `crates/lsp/src/server.rs`

- [ ] **Step 1: Remove the custom `log` function and add `use log`**

Remove the `log` function at `server.rs:23-33` and the `use std::io::Write;` import at line 4. Add at the top of the file:

```rust
use log::{info, warn, debug, trace};
```

Replace all calls to `log("...")` with `info!("...")`.

- [ ] **Step 2: Remove all `/tmp/nostos_lsp_debug.log` file writes**

Search for all 16 blocks matching this pattern and remove them entirely:

```rust
if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_lsp_debug.log") {
    use std::io::Write;
    let _ = writeln!(f, ...);
}
```

Remove each block including surrounding braces. Some are multi-line writes — remove the entire block.

- [ ] **Step 3: Convert `eprintln!` calls to appropriate log levels**

Apply this mapping to all remaining `eprintln!` calls:

| Content pattern | Log level | Rationale |
|---|---|---|
| `"Warning:"`, `"Failed"` | `warn!(...)` | Actionable problems |
| `"Engine initialized"`, `"Starting engine"`, server lifecycle | `info!(...)` | Useful in production |
| Request tracing (`"Completion request"`, `"Hover request"`, `"Saved:"`) | `debug!(...)` | Useful during development |
| Detailed inference (`"Line prefix"`, `"Local vars"`, `"Inferred type"`, `"Identifier before dot"`, `"Found module"`, etc.) | `trace!(...)` | Only needed for deep debugging |

Concretely:
- `eprintln!("Warning: ...")` → `warn!("...")`
- `eprintln!("Engine initialized in {:?}", ...)` → `info!("Engine initialized in {:?}", ...)`
- `eprintln!("Completion request at {:?}", ...)` → `debug!("Completion request at {:?}", ...)`
- `eprintln!("Line prefix: '{}'", ...)` → `trace!("Line prefix: '{}'", ...)`
- `eprintln!("Local vars: {:?}", ...)` → `trace!("Local vars: {:?}", ...)`
- `eprintln!("Inferred type: {:?}", ...)` → `trace!("Inferred type: {:?}", ...)`
- And so on for all ~89 `eprintln!` calls.

All `trace!()` and `debug!()` calls will be **compiled to nothing** in release builds due to `release_max_level_info`.

- [ ] **Step 4: Build and verify**

Run:
```bash
cd ~/git/wego/nostos && cargo build -p nostos-lsp
```

Expected: compiles with no errors. Zero `eprintln!` calls remaining.

Verify:
```bash
grep -c 'eprintln!' crates/lsp/src/server.rs
```

Expected: `0`

- [ ] **Step 5: Commit**

```bash
cd ~/git/wego/nostos
git add crates/lsp/src/server.rs
git commit -m "Migrate server.rs from eprintln to log crate macros"
```

---

## Phase 3: Replace engine init polling with notification

### Task 4: Use `tokio::sync::Notify` instead of polling loop

**Files:**
- Modify: `crates/lsp/src/server.rs`

- [ ] **Step 1: Add `Notify` field to `NostosLanguageServer`**

Add to the struct at line 42:

```rust
use tokio::sync::Notify;

pub struct NostosLanguageServer {
    // ... existing fields ...
    engine_ready: Notify,  // <-- add this
}
```

Initialize in `new()`:

```rust
engine_ready: Notify::new(),
```

- [ ] **Step 2: Signal readiness after engine initialization**

In `initialized()` (around line 707), after `*self.engine.lock().unwrap() = Some(engine);`, add:

```rust
self.engine_ready.notify_waiters();
```

Also in `init_engine()` (around line 83), after `*self.engine.lock().unwrap() = Some(engine);`, add the same:

```rust
self.engine_ready.notify_waiters();
```

And the fallback path at line 92:

```rust
self.engine_ready.notify_waiters();
```

- [ ] **Step 3: Replace polling loops with `Notify::notified()`**

Find the two polling loops (in `check_file` around line 181 and in `completion` around line 1118). They both look like:

```rust
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
```

Replace each with:

```rust
// Fast path: check if engine is already ready
{
    let guard = self.engine.lock().unwrap();
    if guard.is_none() {
        drop(guard);
        // Wait for engine initialization (with timeout)
        tokio::select! {
            _ = self.engine_ready.notified() => {}
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                return; // or return Ok(None) for completion
            }
        }
    }
}
```

Note: the `completion` method should `return Ok(None)` on timeout, while `check_file` should just `return`.

- [ ] **Step 4: Build and verify**

Run:
```bash
cd ~/git/wego/nostos && cargo build -p nostos-lsp
```

- [ ] **Step 5: Commit**

```bash
cd ~/git/wego/nostos
git add crates/lsp/src/server.rs
git commit -m "Replace engine init polling with tokio::sync::Notify"
```

---

## Phase 4: Reduce lock hold time during completion

### Task 5: Extract data upfront and drop lock before building CompletionItems

**Files:**
- Modify: `crates/lsp/src/server.rs` (`get_dot_completions`)

- [ ] **Step 1: Restructure `get_dot_completions` to minimize lock hold time**

The current implementation holds the engine lock (line 2470) for the entire function — through all the type lookups, field queries, UFCS method scanning, trait method collection, and CompletionItem construction. Instead, extract all needed data under the lock, drop it, then build CompletionItems without the lock.

At `crates/lsp/src/server.rs:2467`, restructure the function. The key change is to collect raw data tuples under the lock, then drop the lock, then build `CompletionItem` structs:

```rust
fn get_dot_completions(
    &self,
    before_dot: &str,
    local_vars: &std::collections::HashMap<String, String>,
    lambda_param_type: Option<&str>,
    document_content: &str,
) -> Vec<CompletionItem> {
    // Collect raw data under the lock
    let raw_data = {
        let engine_guard = self.engine.lock().unwrap();
        let Some(engine) = engine_guard.as_ref() else {
            return Vec::new();
        };

        // ... all engine queries go here ...
        // Return a struct or tuple with the data needed
        // (fields, builtin_methods, ufcs_methods, trait_methods, inferred_type, etc.)
    };
    // Lock is dropped here

    // Build CompletionItems from raw_data without holding the lock
    // ...
}
```

This is a significant refactor of a ~450-line function. The data extraction under the lock should collect:
- `inferred_type: Option<String>`
- `fields: Vec<String>`
- `builtin_methods: Vec<(&'static str, &'static str, &'static str)>`
- `ufcs_methods: Vec<(String, String, Option<String>)>`
- `trait_methods: Vec<(String, String, Option<String>)>`
- For module completions: `module_functions: Vec<(String, Option<String>, Option<String>)>` (name, signature, doc)

The CompletionItem construction loop (which is pure data formatting, no engine calls) happens after the lock is released.

- [ ] **Step 2: Build and verify**

Run:
```bash
cd ~/git/wego/nostos && cargo build -p nostos-lsp
```

- [ ] **Step 3: Manual test**

Repeat the test from Task 1 Step 5. Verify dot-completion still works correctly.

- [ ] **Step 4: Commit**

```bash
cd ~/git/wego/nostos
git add crates/lsp/src/server.rs
git commit -m "Minimize engine lock hold time during dot completion"
```

---

## Phase 5: Avoid redundant `get_functions()` calls

### Task 6: Cache `get_functions()` result within a single completion request

**Files:**
- Modify: `crates/lsp/src/server.rs` (`get_dot_completions`)

- [ ] **Step 1: Call `get_functions()` once and reuse the result**

In `get_dot_completions`, `engine.get_functions()` is called twice (line 2608 for module name extraction, line 2629 for iterating module functions). Each call clones all function names, sorts, and dedups them.

Under the lock (as structured in Task 5), call it once:

```rust
let all_functions = engine.get_functions();
```

Then use `all_functions` for both the module name extraction:
```rust
let known_modules: Vec<String> = all_functions.iter()
    .filter_map(|f| f.split('.').next().map(|s| s.to_string()))
    .collect::<std::collections::HashSet<_>>()
    .into_iter()
    .collect();
```

And for iterating module functions:
```rust
for fn_name in &all_functions {
    if fn_name.starts_with(&format!("{}.", module_name)) {
        // ...
    }
}
```

- [ ] **Step 2: Build and verify**

Run:
```bash
cd ~/git/wego/nostos && cargo build -p nostos-lsp
```

- [ ] **Step 3: Commit**

```bash
cd ~/git/wego/nostos
git add crates/lsp/src/server.rs
git commit -m "Call get_functions() once per completion request instead of twice"
```

---

## Summary

| Task | What | Expected impact |
|------|------|-----------------|
| 1 | Debounce `didChange` check_file | **High** — eliminates lock contention from typing |
| 2 | Add `log` crate with compile-time filtering | **Low** — sets up infrastructure |
| 3 | Migrate `eprintln!` to `log` macros | **Medium** — removes sync I/O in hot path; zero overhead in release builds |
| 4 | Replace polling with Notify | **Medium** — faster first completion after startup |
| 5 | Minimize lock hold time | **Medium** — reduces window for contention |
| 6 | Deduplicate `get_functions()` | **Low** — avoids redundant alloc+sort |

Tasks 1-3 are safe, isolated changes. Tasks 4-6 touch more code but are still localized to the LSP crate.

### Future work (not in this plan)

These require deeper changes in the `nostos-repl` or `nostos-compiler` crates:

- **Build a UFCS index** in the compiler (`HashMap<type_name, Vec<fn_name>>`) to replace the O(N) scan in `get_ufcs_methods_for_type`
- **Cache compiled stdlib to disk** using the existing `ModuleCache` infrastructure to reduce startup from ~7s to milliseconds
- **Switch to `tokio::sync::RwLock`** so completion can run concurrently with `check_file` (both take `&self` on the engine)
- **Optimize `extract_local_bindings`** to avoid O(L²) nested file scanning
