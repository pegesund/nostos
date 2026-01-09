- do not make new vms yourself, work with the exising ones. the parallell one is the main target, so prefer fixing this first. Single threaded is for experimentation and validation.
- remember we only work with/benchmark the paralell vm. The single threaded is only kept for experiments.
- build release, not debug
- remember that comments in nostos are with # and NOT with //

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

**PENDING TASK - jsonToTypeByName in Nostos:**
The function `jsonToTypeByName(typeName, json)` must be implemented in `stdlib/json.nos` using:
1. `typeInfo(typeName)` to get type metadata
2. Pattern match on kind ("record" or "variant")
3. Extract field values from Json and convert to correct types
4. Use `makeRecordByName` or `makeVariantByName` to construct the result

DO NOT touch `crates/compiler/src/compile.rs` for this. Write Nostos code.

**Common mistakes to AVOID:**
- Adding new match arms in compile.rs for "jsonToTypeByName" - NO
- Writing Rust functions that walk JSON and construct types - NO
- Saying "I'll implement this as a builtin" - NO
- Starting to write Rust code then asking if it's okay - NO

**The correct approach:**
- Open `stdlib/json.nos`
- Write Nostos functions using existing builtins
- Test with `./target/release/nostos testfile.nos`

**If something doesn't work in stdlib (e.g., throw, Map.insert):**
- First verify the builtin exists and works in regular .nos files
- If it works in tests but not stdlib, fix the stdlib loading mechanism
- If the builtin doesn't exist, add it as a minimal primitive
- NEVER use "stdlib doesn't support X" as an excuse to implement in Rust

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