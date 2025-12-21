- do not make new vms yourself, work with the exising ones. the parallell one is the main target, so prefer fixing this first. Single threaded is for experimentation and validation.
- remember we only work with/benchmark the paralell vm. The single threaded is only kept for experiments.
- build release, not debug
- remember that comments in nostos are with # and NOT with //

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

**IMPORTANT: DO NOT run `cargo test` or `test_all_nos_files` - they WILL HANG!**

Many tests use spawn/receive (concurrency) and will hang indefinitely without timeout protection.

**Safe ways to run tests:**
1. Run specific test categories with short timeout:
   ```bash
   timeout 30 cargo test --release -p nostos-compiler basics
   timeout 30 cargo test --release -p nostos-compiler arithmetic
   timeout 30 cargo test --release -p nostos-compiler functions
   timeout 30 cargo test --release -p nostos-compiler types
   ```

2. Run examples with the script (has built-in timeouts):
   ```bash
   ./scripts/run_examples.sh
   ```

3. Test a specific .nos file directly:
   ```bash
   ./target/release/nostos path/to/file.nos
   ```

4. Run non-hanging categories in batch:
   ```bash
   timeout 60 cargo test --release -p nostos-compiler -- basics arithmetic functions types patterns comparison logical control_flow strings collections traits
   ```

**Run all safe tests with timeout:**
```bash
for f in $(find tests -name "*.nos" ! -path "*/timeout/*"); do
    timeout 5 ./target/release/nostos "$f" >/dev/null 2>&1 || echo "FAIL: $f"
done
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