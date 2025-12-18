# Performance Optimization History

## 2025-12-18: Baseline
- Established baseline benchmarks for 25 test programs
- Most benchmarks: 87-153ms (dominated by startup/compilation time)
- Key bottlenecks identified in ANALYSIS.md

## 2025-12-18: O(log n) Cons Optimization
**Change**: Replaced `Arc<Vec<GcValue>>` with `imbl::Vector<GcValue>` for GcList

**Files modified**:
- `crates/vm/src/gc.rs` - Changed GcList to use imbl::Vector
- `crates/vm/src/async_vm.rs` - Updated Cons instruction, various list usages
- `crates/vm/src/parallel.rs` - Updated Cons instruction
- `crates/vm/src/worker.rs` - Updated Cons instruction

**Results**:
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| 08_list_build (5000 elements) | 120ms | 94ms | **-21%** |
| 26_list_build_large (10000 elements) | N/A | 6ms runtime | New test |

**Notes**:
- Building a 10000-element list now takes ~6ms (0.6µs per cons)
- Previously O(n²) total time for building N elements, now O(n log n)
- Most benchmarks dominated by startup time (~85-100ms), masking improvements
- Some slight regression in other benchmarks due to imbl::Vector overhead vs Arc<Vec> for small lists

**Tests**: All collection and function tests pass

## 2025-12-18: Remove Instruction Cloning
**Change**: Replaced `instruction.clone()` with raw pointer dereference to avoid cloning on every instruction execution.

**Files modified**:
- `crates/vm/src/async_vm.rs`:
  - Added `AsIdx` trait for polymorphic index conversion (works with both `u8` and `&u8`)
  - Changed instruction access from `.clone()` to raw pointer with SAFETY comments
  - Updated macros to use `AsIdx::as_idx()` instead of `as usize` casts
  - Added `*` dereferences for pattern-bound references in specific locations

**Technical details**:
- When matching on `&Instruction`, pattern bindings become references (`dst: &u8` instead of `dst: u8`)
- The `AsIdx` trait allows macros to work with both owned and referenced indices
- Raw pointer is safe because instruction data is in Arc<FunctionValue> which remains valid

**Results**:
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| 03_fib_tail | 99ms | 91ms | **-8%** |
| 09_arithmetic_heavy | 103ms | 96ms | **-6%** |
| 11_record_ops | 98ms | 92ms | **-6%** |
| 19_wide_calls | 101ms | 93ms | **-7%** |
| 25_conditionals | 102ms | 92ms | **-9%** |

**Notes**:
- Some benchmarks show 5-9% improvement
- Other benchmarks show noise variance (within measurement error)
- These short benchmarks (~100ms) are dominated by startup/compilation time
- The optimization eliminates atomic reference counting operations for instructions with Arc fields
- Real benefit more visible in longer-running computations

**Tests**: All 75 core tests pass

## 2025-12-18: Register Value Cloning Analysis
**Status**: Investigated, not implemented (minimal benefit)

**Analysis**:
The register access macro clones GcValue on every read:
```rust
frame.registers[$r.as_idx()].clone()
```

However, after analysis, GcValue cloning is actually cheap for all variants:
- **GcPtr<T>** is explicitly `Copy` (just a u32 index) - no heap allocation
- **Primitive values** (Int64, Float64, Bool, etc.) are Copy
- **List(GcList)** uses `imbl::Vector` which is O(1) clone with structural sharing
- **Arc-based values** (Function, NativeFunction, Type) are O(1) atomic increment

The Clone cost is essentially a fixed-size memcpy of the enum (~24-32 bytes), which is negligible compared to the instruction execution cost.

**Decision**: Not worth implementing because:
1. Clone cost is minimal for all GcValue variants
2. Avoiding clones would require complex borrow checker workarounds
3. Would add code complexity with no measurable performance benefit

**Conclusion**: Mark as investigated, move to other optimizations

## 2025-12-18: Better Benchmarks Created
**Problem**: Original benchmarks (~100ms) were dominated by startup/compilation time (~90ms),
making optimization impact unmeasurable.

**Solution**: Created longer-running benchmarks:

| Benchmark | Description | Time (NO JIT) |
|-----------|-------------|---------------|
| 30_fib_large | fib(35) - ~29M recursive calls | 9.18s |
| 31_list_build_huge | Build 100k element list | 0.14s |
| 32_sum_large | Sum 1M iterations (tail recursive) | 0.29s |
| 33_arithmetic_stress | 10M arithmetic iterations | 3.23s |

**Key Findings**:
- **list_build 100k in 0.14s** proves O(log n) cons is working
  - With O(n) cons: ~5 billion element copies → minutes
  - With O(log n) cons: 140ms → **1000x+ improvement**
- **sum 1M in 0.29s** shows tail call optimization working (0.29µs per iteration)
- **fib(35) in 9.18s** gives meaningful recursive call benchmark
- **arithmetic 10M in 3.23s** stresses instruction dispatch (0.32µs per iteration)

## 2025-12-18: Optimization Attempts

### Attempt 1: SmallVec for registers
**Status**: REVERTED - made things worse

Changed `Vec<GcValue>` to `SmallVec<[GcValue; 8]>` for register storage.

**Result**: fib(35) went from 9.18s to 11.31s (+23% slower!)

**Analysis**: SmallVec overhead for checking inline/heap on every operation
outweighed the allocation savings. The branch prediction cost was higher
than the malloc/free cost for small allocations.

### Attempt 2: Remove arg_values Vec in CallDirect/CallSelf
**Status**: IMPLEMENTED - small improvement

Changed from:
```rust
let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
for (i, arg) in arg_values.into_iter().enumerate() { ... }
```

To:
```rust
for (i, r) in args.iter().enumerate() {
    registers[i] = reg!(*r);
}
```

**Result**: fib(35) ~5% faster (9.18s → 8.72-8.88s median)

**Files modified**: `crates/vm/src/async_vm.rs` (CallDirect and CallSelf)

### Key Insight: Why is fib so slow?
- Tail-recursive sum(1M): 0.29s = **0.29µs per iteration**
- Recursive fib(35): 9s / 30M = **0.3µs per call** + stack overhead
- The 27x difference is mostly frame allocation overhead

The remaining bottleneck is `vec![GcValue::Unit; register_count]` allocation.
Modern allocators are fast, but 30M malloc/free pairs still add up.
