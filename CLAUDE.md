- do not make new vms yourself, work with the exising ones. the parallell one is the main target, so prefer fixing this first. Single threaded is for experimentation and validation.
- remember we only work with/benchmark the paralell vm. The single threaded is only kept for experiments.
- build release, not debug
- remember that comments in nostos are with # and NOT with //

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
for f in $(find tests -name "*.nos" ! -path "*/timeout/*" ! -path "*/flaky/*"); do
    timeout 5 ./target/release/nostos "$f" >/dev/null 2>&1 || echo "FAIL: $f"
done
```

**Tests that may hang (avoid in batch runs):**
- `tests/timeout/` - 2 tests that actually timeout
- `tests/flaky/` - 10 race condition tests (may fail intermittently)

**Test counts:**
- 484 safe tests in regular categories (all pass)
- 2 true timeout tests in `tests/timeout/` (rapid_spawn_die, running_average)
- 10 flaky race condition tests in `tests/flaky/` (known to fail due to non-atomic mvar operations)

**Common stdlib name collisions to avoid in tests:**
- `get` -> use `getValue` (stdlib.list.get)
- `set` -> use `setValue` (stdlib.list.set)
- `reverse` -> use `reverseList` (stdlib.list.reverse)
- `count` -> use `countList` (stdlib.list.count)
- `flatten` -> use `flattenList` (stdlib.list.flatten)
- `interleave` -> use custom name (stdlib.list.interleave)