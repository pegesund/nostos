# Nostos Benchmark Baseline

Recorded: 2025-12-06 (with InlineOp closure optimization)

## System Info
- Platform: Linux 6.14.0-116036-tuxedo
- CPU: Intel Core Ultra 7 155H
- Rust: release build
- Note: Absolute times vary with CPU frequency; ratios are stable

## Summary Table

| Benchmark | Nostos | Python | Ratio |
|-----------|--------|--------|-------|
| Fibonacci (fib 35) | 0.15s | 1.56s | **10x faster** |
| List Sum fold (1k × 1000) | 0.13s | 0.13s | **~same** |
| List Sum recursive (1k × 1000) | 0.12s | 0.11s | **~same** |

## Detailed Results

### Fibonacci (fib(35))
Recursive fibonacci - tests function call overhead and JIT.

| Runtime | Ratio | Notes |
|---------|-------|-------|
| Nostos (parallel) | **10x faster** | JIT compiles numeric functions |
| Python 3 | baseline | Recursive interpreter |

### List Sum Fold (1k elements × 1000 iterations)
Tests recursive fold over linked list with lambda closure `(a, b) => a + b`.

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel) | 0.13s | InlineOp optimization for closure calls |
| Python 3 | 0.13s | TCO with sys.setrecursionlimit |

**Key insight**: With InlineOp, closure calls (0.13s) are almost identical to
direct recursion (0.12s), proving the closure overhead has been eliminated.

### List Sum Recursive (1k elements × 1000 iterations)
Tests recursive sum without lambda (direct TCO recursion).

| Runtime | Time | Notes |
|---------|------|-------|
| Nostos (parallel) | 0.12s | Direct tail recursion |
| Python 3 | 0.11s | TCO with sys.setrecursionlimit |

## InlineOp Optimization

The `InlineOp` optimization caches the closure operation type at creation time:

```rust
pub enum InlineOp {
    None,
    AddInt,  // (a, b) => a + b
    SubInt,  // (a, b) => a - b
    MulInt,  // (a, b) => a * b
}
```

This avoids heap lookup and pattern matching on every closure call.
Before: fold was 2x slower than Python. After: ~same speed.

## Notes

- Nostos is **10x faster than Python** for recursive numeric code (JIT)
- Closure calls are now as fast as direct function calls (InlineOp)
- Lists use O(1) tail via Arc sharing for functional programming patterns
