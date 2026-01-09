# Nostos Benchmark Baseline

Recorded: 2025-12-06 (updated with InlineOp closure optimization)

## System Info
- Platform: Linux 6.14.0-116036-tuxedo
- Rust: release build

## Summary Table

| Benchmark | Nostos | Python | Comparison |
|-----------|--------|--------|------------|
| Fibonacci (fib 35) | 0.15s | 0.77s | **5x faster** |
| Array Write (100k × 100) | 0.29s | 0.99s | **3x faster** |
| List Sum fold (1k × 1000) | 0.12s | 0.13s | **~same** |
| List Sum recursive (1k × 1000) | 0.12s | 0.11s | **~same** |
| List Sum native (1k × 100) | 0.07s | - | - |
| List Sum simple (10k × 1000) | 0.06s | - | - |

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

### List Sum Fold (1k elements × 1000 iterations)
Tests recursive fold over linked list with lambda closure.

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel) | 0.12s | InlineOp optimization for closure calls |
| Python 3 | 0.13s | TCO with sys.setrecursionlimit |

### List Sum Recursive (1k elements × 1000 iterations)
Tests recursive sum without lambda (direct recursion).

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel) | 0.12s | Direct tail recursion |
| Python 3 | 0.11s | TCO with sys.setrecursionlimit |

### List Sum Native (1k elements × 100 iterations)
Uses native `listSum` function.

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel) | 0.07s | Native Rust implementation |

## Notes

- Nostos with JIT is **3-5x faster than Python** for numeric benchmarks
- InlineOp optimization caches closure operation type at creation time
- This avoids heap lookup and pattern matching on every closure call
- List fold with lambda is now **~same speed as Python** (was 2x slower before InlineOp)
- Lists use O(1) tail via Arc sharing for functional programming patterns
