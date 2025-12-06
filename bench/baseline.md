# Nostos Benchmark Baseline

Recorded: 2025-12-06

## System Info
- Platform: Linux 6.14.0-113033-tuxedo
- Rust: release build

## Summary Table

| Benchmark | Nostos JIT | Nostos no-JIT | Python | JIT Speedup | vs Python |
|-----------|------------|---------------|--------|-------------|-----------|
| Fibonacci (fib 35) | 0.07s | 10.12s | 0.77s | **145x** | **11x faster** |
| Array Sum (1M × 100) | 0.09s | 5.12s | 1.79s | **57x** | **20x faster** |
| Array Write (100k × 100) | 0.05s | 1.36s | 0.99s | **27x** | **20x faster** |

## Detailed Results

### Fibonacci (fib(35))
Recursive fibonacci - tests function call overhead and JIT for numeric functions.

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel, JIT) | 0.07s | Single-clause if-then-else JITs |
| Nostos (parallel, no-JIT) | 10.12s | Interpreted only |
| Python 3 | 0.77s | - |

### Array Sum (1M elements × 100 iterations)
Tests loop array JIT for Int64Array operations.

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel, JIT) | 0.09s | Loop array JIT compiles to native |
| Nostos (parallel, no-JIT) | 5.12s | Interpreted loop |
| Python 3 | 1.79s | Native list iteration |

### Array Write Loop (100k elements × 100 iterations)
Tests while-loop JIT for array element writes.

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel, JIT) | 0.05s | Loop JIT for array writes |
| Nostos (parallel, no-JIT) | 1.36s | Interpreted loop |
| Python 3 | 0.99s | - |

## Notes

- JIT provides **27-145x speedup** over interpreted execution
- Nostos with JIT is **11-20x faster than Python**
- Single-clause functions with `if-then-else` compile to numeric JIT
- While loops over Int64Array compile to loop array JIT
- Multi-clause pattern matching (e.g., `fib(0)=0; fib(1)=1; fib(n)=...`) does not JIT compile
