# Nostos Benchmark Baseline

Recorded: 2025-12-05

## System Info
- Platform: Linux 6.14.0-113033-tuxedo
- Rust: release build

## Fibonacci (fib(35))

| Runtime | Time | vs Python | vs Ruby |
|---------|------|-----------|---------|
| Nostos (regular VM) | 0.059s | 13x faster | 11x faster |
| Nostos (parallel VM) | 0.053s | 14.5x faster | 12.3x faster |
| Python 3 | 0.769s | - | - |
| Ruby | 0.652s | - | - |

## Array Write (100 iterations x 100k elements)

| Runtime | Time | vs Python | vs Ruby |
|---------|------|-----------|---------|
| Nostos loop (regular VM) | 0.039s | 24x faster | 22x faster |
| Nostos loop (parallel VM) | 0.040s | 24x faster | 21x faster |
| Nostos recursive | 0.339s | 2.9x faster | 2.5x faster |
| Python 3 | 0.972s | - | - |
| Ruby | 0.860s | - | - |

## Notes

- JIT compilation works for single-clause functions with `if-then-else`
- Multi-clause pattern matching functions (e.g., `fib(0)=0; fib(1)=1; fib(n)=...`) do not JIT compile due to `TestConst` instruction
- Loop JIT provides significant speedup for array operations (9.6x vs recursive)
- Both regular and parallel VMs achieve similar performance for compute-bound tasks
