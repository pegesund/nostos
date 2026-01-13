# LSP Testing Guide - READ THIS BEFORE EVERY FIX

## The Problem

The LSP has MULTIPLE code paths that get triggered in different scenarios:

1. **Initial load**: `load_directory()` → `compile_all_collecting_errors()`
2. **No changes edit**: `recompile_module_with_content()` → "No changes detected" path
3. **Actual edit**: `recompile_module_with_content()` → Recompilation + post-validation path
4. **check_module_compiles**: Uses a SEPARATE `check_compiler` instance

**CRITICAL**: You MUST test ALL relevant code paths! A bug might only appear in ONE of them.

## The Correct Testing Pattern

To test LSP validation correctly, you MUST simulate the user's actual workflow:

```rust
#[test]
fn test_lsp_scenario() {
    let temp_path = create_temp_dir_lsp("test_name");

    // 1. Create files WITHOUT the problematic code
    let initial_content = r#"main() = {
    x = 1
    y = 2
}
"#;
    fs::write(temp_path.join("main.nos"), initial_content).expect("write");

    // 2. Create engine with LSP-IDENTICAL config
    let config = ReplConfig {
        enable_jit: false,
        num_threads: 1,
    };
    let mut engine = ReplEngine::new(config);
    engine.load_stdlib().expect("Failed to load stdlib");
    engine.load_directory(temp_path.to_str().unwrap()).expect("load");

    // 3. NOW simulate user EDITING and adding the problematic code
    let edited_content = r#"main() = {
    x = 1
    y = 2
    y.map(m => m.something())  // <-- The new line being tested
}
"#;

    // 4. Call recompile_module_with_content with CHANGED content
    let result = engine.recompile_module_with_content("main", edited_content);

    // 5. Assert the result
    assert!(result.is_ok(), "Expected Ok but got: {:?}", result);
}
```

## Why This Matters

### WRONG approach (what I kept doing):
```rust
// Create file WITH the problematic code
let content = "main() = { y.map(m => m.foo()) }";
fs::write(path, content);

// Load it
engine.load_directory(path);

// Call recompile with SAME content
engine.recompile_module_with_content("main", content);
// Returns "No changes detected" - DOESN'T TEST THE ACTUAL CODE PATH!
```

### RIGHT approach:
```rust
// Create file WITHOUT the problematic code
let initial = "main() = { x = 1 }";
fs::write(path, initial);

// Load it
engine.load_directory(path);

// Now simulate EDIT with CHANGED content
let edited = "main() = { x = 1; y.map(m => m.foo()) }";
engine.recompile_module_with_content("main", edited);
// This triggers ACTUAL recompilation and post-validation!
```

## Common Bugs This Catches

1. **Duplicate validation arrays**: `recompile_module_with_content` has TWO `generic_builtins` arrays:
   - One for "no changes" path
   - One for "after compilation" path
   - If they differ, bugs only appear when content actually changes!

2. **State not preserved**: Some state might exist after initial load but not after recompilation.

3. **Different compiler instances**: Initial load uses main compiler, some validation uses isolated compilers.

## Test Locations

### 1. Unit Tests (engine.rs)
For testing internal engine behavior:
```
crates/repl/src/engine.rs - module lsp_show_tests
```

Run with:
```bash
cargo test --release -p nostos-repl lsp_show_tests -- --nocapture
```

### 2. Integration Tests (PREFERRED for LSP behavior)
For testing actual LSP behavior as VS Code sees it:
```
crates/lsp/tests/integration_test.rs
```

Run with:
```bash
cargo test --release -p nostos-lsp --test integration_test -- --nocapture
```

**Why integration tests are better:**
- Run the actual `nostos-lsp` binary via stdin/stdout
- Test real JSON-RPC message flow (didOpen, didChange, publishDiagnostics)
- Catch notification ordering bugs that unit tests miss
- Test the full LSP lifecycle including initialization

**Integration test pattern:**
```rust
#[test]
fn test_lsp_scenario() {
    let project_path = create_test_project("test_name");

    // Create test files
    fs::write(project_path.join("main.nos"), "main() = { ... }").unwrap();

    // Start LSP client (spawns actual binary)
    let mut client = LspClient::new(&get_lsp_binary());
    let _ = client.initialize(project_path.to_str().unwrap());
    client.initialized();

    // Open file
    let uri = format!("file://{}/main.nos", project_path.display());
    client.did_open(&uri, content);

    // Read diagnostics
    let diagnostics = client.read_diagnostics(&uri, Duration::from_secs(2));

    // Test didChange
    client.did_change(&uri, new_content, version);
    let updated = client.read_diagnostics(&uri, Duration::from_secs(2));

    // Cleanup
    client.shutdown();
    client.exit();
    cleanup_test_project(&project_path);

    // Assert
    assert!(diagnostics.iter().any(|d| d.line == expected_line));
}
```

**Important:** Integration tests use `~/.local/bin/nostos-lsp`. After changes:
```bash
cargo build --release -p nostos-lsp && cp ./target/release/nostos-lsp ~/.local/bin/
```

## Required Tests

Create tests for ALL these scenarios:

1. **Initial load test** (`test_map_at_initial_load`):
   - File has problematic code FROM THE START
   - Tests `load_directory` → `compile_all_collecting_errors`

2. **Edit test** (`test_map_with_asint8`):
   - Initial file WITHOUT problematic code
   - Then call `recompile_module_with_content` WITH problematic code
   - Tests actual recompilation path

3. **check_module_compiles test** (`test_check_module_compiles_with_map`):
   - Uses separate compiler instance
   - Tests the editor's live validation path

## CRITICAL: Stdlib Must Be Found

The LSP server runs from the **workspace directory** (e.g., `/var/tmp/test_status_project`), NOT from the Nostos source directory. This means `load_stdlib()` must be able to find the stdlib from ANY working directory.

### The Problem

Tests run from the Nostos source tree where `stdlib/` exists at `../../stdlib`. But the installed LSP binary at `~/.local/bin/nostos-lsp` runs from arbitrary directories.

If stdlib is not found:
- `load_stdlib()` returns `Ok(())` but registers **0 prelude imports**
- Functions like `map`, `filter`, `fold` are NOT available
- UFCS calls like `yy.map(...)` fail with "no method found"

### Stdlib Search Paths

`load_stdlib()` searches these paths in order:
1. `stdlib` (current directory)
2. `../stdlib`, `../../stdlib` (parent directories)
3. `~/.nostos/stdlib` (user installation)
4. `~/dev/rust/nostos/stdlib` (common dev location)
5. `~/dev/rust/nostos_duplicate/stdlib` (this repo)
6. Relative to executable path

### Verifying Stdlib Loaded

Check the LSP output for:
```
LSP: Stdlib loaded successfully, N prelude imports registered
LSP: 'map' is in prelude imports
```

If you see `0 prelude imports`, stdlib was NOT found!

### Installing Stdlib for LSP

For the installed LSP binary to work, either:
1. Copy stdlib to `~/.nostos/stdlib`:
   ```bash
   mkdir -p ~/.nostos
   cp -r stdlib ~/.nostos/
   ```
2. Or ensure the LSP runs from a directory where stdlib is findable

## Checklist Before Every LSP Fix

- [ ] I have read this document
- [ ] My test creates initial files WITHOUT the problematic code
- [ ] My test simulates an EDIT by calling `recompile_module_with_content` with CHANGED content
- [ ] My test uses `ReplConfig { enable_jit: false, num_threads: 1 }` (LSP-identical config)
- [ ] My test calls `load_stdlib()` before `load_directory()`
- [ ] **My test verifies stdlib loaded** (check prelude imports count > 0)
- [ ] My test verifies the ACTUAL error the user sees, not a different error
- [ ] I ran the test and it FAILS before my fix, PASSES after
- [ ] I tested ALL THREE scenarios (initial load, edit, check_module_compiles)
