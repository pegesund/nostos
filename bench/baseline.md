# Nostos Benchmark Baseline

Recorded: 2025-12-06

## System Info
- Platform: Linux 6.14.0-113033-tuxedo
- Rust: release build

## Fibonacci (fib(35))

| Runtime | Time | vs Python | vs Ruby |
|---------|------|-----------|---------|
| Nostos (regular VM) | 0.054s | 14x faster | 12x faster |
| Nostos (parallel VM) | 0.073s | 10.5x faster | 9x faster |
| Python 3 | 0.769s | - | - |
| Ruby | 0.652s | - | - |

## Array Write (100 iterations x 100k elements)

| Runtime | Time | vs Python | vs Ruby |
|---------|------|-----------|---------|
| Nostos loop (regular VM) | 0.050s | 19x faster | 17x faster |
| Nostos loop (parallel VM) | 0.051s | 19x faster | 17x faster |
| Nostos recursive | 0.386s | 2.5x faster | 2.2x faster |
| Python 3 | 0.972s | - | - |
| Ruby | 0.860s | - | - |

## Array Sum JIT (1M elements x 10 calls)

| Runtime | Time | JIT Speedup |
|---------|------|-------------|
| Nostos (parallel VM, JIT) | 0.074s | **8x faster** |
| Nostos (no JIT) | 0.593s | baseline |

## Array Sum Single (10M elements, 1 call)

| Runtime | Time | JIT Speedup |
|---------|------|-------------|
| Nostos (parallel VM, JIT) | 0.547s | **2x faster** |
| Nostos (no JIT) | 1.098s | baseline |

## Notes

- JIT compilation works for single-clause functions with `if-then-else`
- Multi-clause pattern matching functions (e.g., `fib(0)=0; fib(1)=1; fib(n)=...`) do not JIT compile due to `TestConst` instruction
- Loop JIT provides significant speedup for array operations
- Loop array JIT compiles Int64Array sum functions to native code (8x speedup)
- Single-call array sum shows 2x speedup because array filling dominates runtime
- Both regular and parallel VMs achieve similar performance for compute-bound tasks
