- do not make new vms yourself, work with the exising ones. the parallell one is the main target, so prefer fixing this first. Single threaded is for experimentation and validation.
- remember we only work with/benchmark the paralell vm. The single threaded is only kept for experiments.
- build release, not debug
- remember that comments in nostos are with # and NOT with //

## Running Tests

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

**Tests that WILL hang (avoid these):**
- `test_all_nos_files` - loops through all tests including concurrent ones
- Any test in: `/concurrency/`, `/mvar/concurrent/`, `/gc/gc_concurrent*`, `/io/`
- Tests with `spawn`, `receive`, or `send` in the name