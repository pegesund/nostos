- do not make new vms yourself, work with the exising ones. the parallell one is the main target, so prefer fixing this first. Single threaded is for experimentation and validation.
- remember we only work with/benchmark the paralell vm. The single threaded is only kept for experiments.
- build release, not debug
- remember that comments in nostos are with # and NOT with //

## CRITICAL: LSP Testing - READ docs/LSP_TESTING.md FIRST

**BEFORE fixing ANY LSP issue, READ `docs/LSP_TESTING.md`!**

The LSP has multiple code paths. Tests that load files and call `recompile_module_with_content` with the SAME content will return "No changes detected" and NOT test the actual recompilation path!

**Correct pattern:**
1. Create files WITHOUT the problematic code
2. Load directory
3. Call `recompile_module_with_content` with CHANGED content (simulating user edit)

This is the ONLY way to test the actual code path users experience.

## CRITICAL: Do NOT Run Full Test Suite Constantly

**STOP running `cd tests && ./runall.sh` after every small change!**

- Full test suite takes several minutes - this wastes enormous time
- Only run full suite when: explicitly asked, or before final commit
- For focused changes: just `cargo build --release` and let user test manually
- For unit tests: run only the specific test (e.g., `cargo test --release -p nostos-repl test_name`)
- The user will tell you when to run the full suite

## CRITICAL: Update LSP Binary After Building

**VS Code uses the LSP binary from PATH, NOT from ./target/release/**

The LSP binary is installed at: `~/.local/bin/nostos-lsp`

After making changes to the LSP server:
```bash
# Build the LSP
cargo build --release -p nostos-lsp

# Copy to PATH (VS Code won't see changes otherwise!)
cp ./target/release/nostos-lsp ~/.local/bin/nostos-lsp

# If "Text file busy" error, kill VS Code first:
pkill -9 code
cp ./target/release/nostos-lsp ~/.local/bin/nostos-lsp

# Then restart VS Code
```

**ALWAYS copy after building** - otherwise you'll waste hours debugging with the old binary!

## CRITICAL: Debug Output Must Go To FILE, Not Console

**TUI takes over the terminal - eprintln!/println! output DISAPPEARS!**

- When debugging TUI issues, write to a file: `/tmp/nostos_debug.log`
- Use: `use std::io::Write; let mut f = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nostos_debug.log").unwrap(); writeln!(f, "debug: {}", value).unwrap();`
- NEVER use eprintln! or println! for TUI debugging - it will not be visible
- Console output only works for non-TUI (script mode) debugging
- Before asking user to test: clear the log with `rm -f /tmp/nostos_debug.log`
- After user tests: read the log yourself with `Read` tool - don't ask user to cat it

## CRITICAL: Never Hide Bugs - Fix The Code

**THIS IS THE MOST IMPORTANT RULE. VIOLATING THIS IS UNACCEPTABLE.**

When a test fails:
1. **FIX THE CODE** - The test is correct, the code is broken
2. **NEVER change test expectations** to match broken behavior
3. **NEVER remove tests** because they expose bugs
4. **NEVER claim "all tests pass"** without actually running ALL tests

Tests exist to catch bugs. When they fail, they are doing their job. The fix is ALWAYS to fix the underlying code, not to hide the bug by modifying or removing the test.

**Examples of what NOT to do:**
- Changing `# expect: 100` to `# expect: 0` because code returns wrong value - NO!
- Removing a test file because it exposes a bug - NO!
- Saying "all tests pass" after only running 159 of 648 tests - NO!

**What to do instead:**
- Investigate WHY the test fails
- Find and fix the bug in the compiler/VM/runtime
- Verify the test passes with the fix
- Only then commit

## Commit Early and Often

**IMPORTANT: Commit working code immediately to avoid losing it.**

- When a feature or fix is working, commit it RIGHT AWAY
- Do not wait until "everything is done"
- Session context can be lost at any time
- Uncommitted changes will be lost if stash/checkout operations are needed
- Small, frequent commits are better than losing hours of work

**DANGER: git stash operations**
- Before doing `git stash`, commit any important work first
- `git stash` only saves staged AND unstaged changes that git knows about
- If files were modified in a previous session but not committed, they may be lost
- Prefer committing to a branch over stashing

## CRITICAL: Implement Features in Nostos, NOT Rust

**READ THIS CAREFULLY - THIS HAS BEEN VIOLATED 4+ TIMES:**

When asked to implement stdlib features or language functionality:
1. ALWAYS implement in Nostos code (in `stdlib/*.nos` files)
2. DO NOT implement in Rust unless there is absolutely no other option
3. If Nostos lacks required language features, FIX THE LANGUAGE first
4. Use existing builtins as building blocks

**Available builtins for reflection/construction:**
- `typeInfo(typeName)` - returns Map with type metadata (kind, fields, constructors)
- `makeRecordByName(typeName, fieldsMap)` - construct record from Map of field values
- `makeVariantByName(typeName, ctorName, fieldsMap)` - construct variant
- `Map.get`, `Map.insert`, `Map.isEmpty` - Map operations
- `fromJsonValue(typeName, json)` - convert parsed Json to typed value (in stdlib/json.nos)

**Common mistakes to AVOID:**
- Writing Rust functions for things that can be done in Nostos - NO
- Saying "I'll implement this as a builtin" when it can be stdlib - NO

**The correct approach:**
- Implement in `stdlib/*.nos` using existing builtins
- Test with `./target/release/nostos testfile.nos`
- Only add Rust builtins for things that truly require low-level access

**This is a new programming language. If a task requires features Nostos doesn't have, add those features to the language rather than falling back to Rust.**

## Test System

### Test File Format
Tests are `.nos` files in the `tests/` directory with special comment headers:
- `# expect: <value>` - Expected return value from main()
- `# expect_error: <string>` - Test expects compilation/runtime error containing this string

Example success test:
```nostos
# expect: 42
main() = 21 * 2
```

Example error test:
```nostos
# expect_error: type mismatch
main() = "hello" + 5
```

### Test Runner Location
Test runner is at: `crates/compiler/tests/nos_files.rs`

### Test Categories (directories in tests/)
arithmetic, basics, bindings, collections, comparison, concurrency, control_flow,
deriving, edge_cases, error_reporting, exceptions, functions, gc, io, logical,
memory, modules, mvar, nos, numerics, patterns, patterns_stress, stack, strings,
trait_bounds, traits, typed_bindings, type_errors, types, type_system

### Running Tests

**CRITICAL: Use runall.sh - DO NOT CHANGE THIS PROCEDURE**

The ONLY correct way to run all tests is:
```bash
cd tests && ./runall.sh
```

This script:
- Runs all 581+ tests in parallel with proper timeouts
- Handles concurrency tests correctly
- Shows pass/fail summary

**DO NOT:**
- Run `cargo test` for .nos files (will hang on concurrency tests)
- Create new test runner scripts
- Change the testing procedure in any way

**For single file testing:**
```bash
./target/release/nostos path/to/file.nos
```

**For REPL/autocomplete unit tests:**
```bash
cargo test --release -p nostos-repl test_name -- --nocapture
```

**Tests that may hang (avoid in batch runs):**
- `tests/timeout/` - 2 tests that intentionally test timeout behavior

**Tests requiring external services:**
- `tests/postgres/` - Requires PostgreSQL running at localhost with user/password: postgres/postgres
  - Run manually: `for f in tests/postgres/*.nos; do ./target/release/nostos "$f"; done`

**Test counts:**
- 490 tests in regular categories (all pass)
- 2 timeout tests in `tests/timeout/` (rapid_spawn_die, running_average)
- 5 postgres tests in `tests/postgres/` (require PostgreSQL)

**Common stdlib name collisions to avoid in tests:**
- `get` -> use `getValue` (stdlib.list.get)
- `set` -> use `setValue` (stdlib.list.set)
- `reverse` -> use `reverseList` (stdlib.list.reverse)
- `count` -> use `countList` (stdlib.list.count)
- `flatten` -> use `flattenList` (stdlib.list.flatten)
- `interleave` -> use custom name (stdlib.list.interleave)

### Testing Editor/REPL Compile Checking

When working on the TUI editor's compile checking (live error detection), **do NOT debug by manually testing in the TUI**. Instead, write Rust unit tests that call the same function the editor uses.

**Location:** `crates/repl/src/engine.rs` - module `check_module_tests`

**How to test:**
```rust
#[test]
fn test_your_scenario() {
    let engine = ReplEngine::new(ReplConfig::default());
    let code = "main() = { x = [1]; x.map(y => y * 2).xxx() }";
    let result = engine.check_module_compiles("", code);
    println!("Result: {:?}", result);
    assert!(result.is_err(), "Expected error");
    assert!(result.unwrap_err().contains("List.xxx"));
}
```

**Run tests:**
```bash
cargo test --release -p nostos-repl check_module_tests -- --nocapture
```

**Why this approach:**
1. Tests use the exact same `check_module_compiles` function as the TUI editor
2. Fast iteration - no need to start/stop the TUI
3. Can add debug output with `println!` and `--nocapture`
4. Tests document expected behavior and prevent regressions

**Key function:** `ReplEngine::check_module_compiles(&self, module_name: &str, content: &str) -> Result<(), String>`

## CRITICAL: LSP Testing - DO NOT Ask User To Test Manually

**Every time you ask the user to test LSP in VS Code, you waste their time and create whack-a-mole bugs!**

### The Problem
The LSP uses multiple code paths that must work together:
1. `load_directory` - initial project load
2. `recompile_module_with_content` - when files are edited
3. `publish_file_diagnostics_filtered` - sending errors to VS Code

Fixing one path often breaks another. Manual testing catches ONE scenario, then next change breaks previous fixes.

### The Solution: Automated LSP Integration Tests

**ALWAYS write tests that simulate the EXACT editor flow before making LSP changes.**

**Location:** `crates/repl/src/engine.rs` - module `lsp_integration_tests`

**Test pattern:**
```rust
#[test]
fn test_lsp_scenario() {
    // 1. Create temp directory with multiple .nos files
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("good.nos"), "pub add(a: Int, b: Int) = a + b").unwrap();
    std::fs::write(dir.path().join("main.nos"), "main() = good.add(1, \"bad\")").unwrap();

    // 2. Create engine and load directory (simulates VS Code opening project)
    let mut engine = ReplEngine::new(ReplConfig::default());
    engine.load_directory(dir.path().to_str().unwrap()).unwrap();

    // 3. Check compile status (what VS Code sees)
    let status = engine.get_all_compile_status();
    assert!(status.iter().any(|(_, s)| s.contains("Error")), "Should detect type error");

    // 4. Simulate edit (user changes file)
    let new_content = "main() = good.add(1, 2)"; // Fixed
    engine.recompile_module_with_content("main", new_content, dir.path().to_str().unwrap()).unwrap();

    // 5. Verify error is gone
    let status = engine.get_all_compile_status();
    assert!(!status.iter().any(|(n, s)| n.contains("main") && s.contains("Error")));
}
```

**Scenarios that MUST be tested:**
1. Initial load detects type errors
2. Fixing error clears diagnostic
3. Reintroducing error shows diagnostic again
4. Changing function signature in module A marks module B as stale/error
5. Renaming function in module A shows "undefined function" in module B
6. Line numbers are correct in all scenarios

**Run LSP tests:**
```bash
cargo test --release -p nostos-repl lsp_integration_tests -- --nocapture
```

**Workflow:**
1. FIRST: Write failing test for the bug/feature
2. THEN: Fix the code
3. VERIFY: Test passes
4. ONLY THEN: Ask user to test if needed (rare)

### VS Code Binary Update
After LSP changes:
```bash
cargo build --release -p nostos-lsp && cp ./target/release/nostos-lsp ~/.local/bin/nostos-lsp
```

## BUILTINS Array Notes

- Float64Array, Int64Array, Float32Array methods are in BUILTINS for type inference
- Buffer methods are NOT in BUILTINS (causes type conflicts with html.nos) but are handled via:
  - Direct UFCS dispatch in compile.rs (Buffer.append, Buffer.toString)
  - Expression pattern detection in engine.rs/repl.rs for type inference