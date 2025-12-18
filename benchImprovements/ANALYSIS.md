# Interpreter Performance Analysis

## Executive Summary

Based on profiling 25 benchmark programs, I've identified several significant performance bottlenecks in the AsyncVM interpreter. The most critical issue is **O(n) list cons operations** which makes list-heavy code quadratically slow. Other major issues include excessive cloning of instructions and values on every operation.

## Profiling Results Summary

| Benchmark | Calls | Total Time | Avg per Call | Notes |
|-----------|-------|------------|--------------|-------|
| 02_fib_recursive | 21,890 | 109ms | 5µs | Reasonable for recursive |
| 03_fib_tail | 1 (main only) | 0.06ms | - | Tail calls bypass profiling |
| 04_sum_recursive | 5,000 | 9,119ms | 1,824µs | **EXTREMELY SLOW** |
| 06_list_map | 501 | 367ms | 732µs | Very slow list ops |
| 07_list_filter | 251 | 72ms | 287µs | Slow list ops |
| 08_list_build | 1 | 35ms | - | Building 5000-element list |
| 10_pattern_match | 1,000 | 715ms | 715µs | List traversal bottleneck |
| 18_nested_calls | 11,000 | 3ms | 0.1-2.8µs | Reasonable call overhead |
| 22_ackermann | 85,865 | 18,419ms | 214µs | Stack-heavy benchmark |

### Key Observations

1. **List operations are catastrophically slow**: 732µs per map call, 287µs per filter call
2. **Function call overhead is reasonable**: ~0.1-5µs per call for simple functions (identity: 0.04µs)
3. **Tail calls work but bypass profiling**: fib_tail shows only main() because tail calls reuse frames

### Profiler Note: Inclusive Timing

The profiler measures **inclusive time** (time in function including all children). For deeply recursive functions like sum_to(5000), this compounds:
- sum_to(1) measures just itself
- sum_to(2) measures itself + sum_to(1)
- sum_to(5000) measures itself + all 4999 children

This explains the 1,824µs "average" for sum_to — it's not the actual per-call cost but the inclusive time divided by calls. The real per-call overhead is ~1-2µs, which is reasonable but still adds up for 5000 nested calls.

---

## CRITICAL ISSUES

### 1. O(n) List Cons Operation — **HIGHEST PRIORITY**

**Location**: `crates/vm/src/async_vm.rs:1746-1757`

```rust
Cons(dst, head, tail) => {
    let head_val = reg!(head);
    let tail_val = reg!(tail);
    if let GcValue::List(tail_list) = tail_val {
        let mut items = vec![head_val];
        items.extend(tail_list.items().iter().cloned());  // O(n) COPY!
        let new_list = self.heap.make_list(items);
        set_reg!(dst, GcValue::List(new_list));
    }
}
```

**Problem**: Every cons operation copies the ENTIRE tail list. Building a list of N elements becomes O(n²).

**Impact**: This explains why:
- `list_map` is 732µs per call (each recursive call conses, copying growing list)
- `list_build` takes 35ms to build 5000 elements
- Pattern matching on lists is slow (decons + cons patterns)

**Solution Options**:
1. **Linked list with Arc nodes**: `Arc<ConsCell>` where `ConsCell = { head: GcValue, tail: Arc<ConsCell> }`
2. **Prepend buffer**: Keep a prepend buffer that's only copied on access
3. **Rope-like structure**: Tree of chunks for O(log n) operations

**Estimated Impact**: 10-100x speedup for list-heavy code

---

### 2. Instruction Cloning on Every Step — **HIGH PRIORITY**

**Location**: `crates/vm/src/async_vm.rs:511-514`

```rust
let instruction = {
    let frame = self.frames.last().unwrap();
    frame.function.code.code[ip].clone()
};
```

**Problem**: Every instruction is cloned before dispatch. Instructions contain:
- `RegList` = `Arc<[Reg]>` — atomic refcount bump
- Other variants with heap data

**Impact**: Adds ~50-100ns overhead per instruction from atomic operations.

**Solution**: Use reference or index-based dispatch:
```rust
let instruction = &self.frames.last().unwrap().function.code.code[ip];
// Or: store instruction pointer and use unsafe raw pointer
```

**Estimated Impact**: 2-5x speedup on instruction dispatch

---

### 3. Register Value Cloning — **HIGH PRIORITY**

**Location**: `crates/vm/src/async_vm.rs:520-525`

```rust
macro_rules! reg {
    ($r:expr) => {{
        let frame = self.frames.last().unwrap();
        frame.registers[$r as usize].clone()  // CLONES VALUE
    }};
}
```

**Problem**: Every register READ clones the value. GcValue variants like:
- `GcValue::List(GcList)` — contains `Arc<Vec<GcValue>>`
- `GcValue::String(GcPtr<GcString>)` — heap pointer
- These require atomic refcount operations on clone

**Impact**: 3-10 clones per instruction × ~20-50ns each = 60-500ns overhead per instruction

**Solution**: Use references where possible, only clone when escaping:
```rust
macro_rules! reg_ref {
    ($r:expr) => {{
        &self.frames.last().unwrap().registers[$r as usize]
    }};
}
```

**Estimated Impact**: 2-4x speedup in value-heavy code

---

### 4. Repeated Frame Lookup — **MODERATE PRIORITY**

**Location**: Throughout step() function

**Problem**: Every register access calls `self.frames.last().unwrap()`:
- Bounds check on Vec
- Pointer dereference
- Called 5-10 times per instruction

**Solution**: Cache frame pointer at start of instruction:
```rust
let frame = self.frames.last_mut().unwrap();
// Use frame directly for all register ops in this instruction
```

**Estimated Impact**: 10-20% speedup

---

### 5. Per-Instruction Overhead in Run Loop — **LOW PRIORITY**

**Location**: `crates/vm/src/async_vm.rs:461-489`

```rust
loop {
    if self.shared.shutdown.load(Ordering::SeqCst) { ... }  // Atomic load
    if self.frames.is_empty() { ... }                        // Check
    match self.step().await {
        Ok(StepResult::Continue) => {
            self.maybe_yield().await;  // Counter check
        }
        ...
    }
}
```

**Problem**: 3 checks per instruction add up.

**Solution**:
- Check shutdown every 1000 instructions
- Move frame empty check to StepResult::Finished handling
- Inline the step() function

**Estimated Impact**: 5-10% speedup

---

## RECOMMENDED OPTIMIZATION ORDER

### Phase 1: Low-Hanging Fruit (High Impact, Moderate Effort)

1. **Fix Cons to be O(1)** — Transform list-heavy code from O(n²) to O(n)
   - Implement proper persistent list structure
   - Test with list_map, list_filter, list_build benchmarks
   - Expected: 10-100x speedup for list operations

2. **Avoid instruction cloning** — Use references
   - Change step() to work with instruction references
   - May need unsafe code for lifetime management
   - Expected: 2-5x speedup overall

3. **Avoid register cloning for reads** — Return references
   - Add reg_ref! macro that returns &GcValue
   - Update instruction handlers to use references where possible
   - Expected: 2-4x speedup

### Phase 2: Moderate Effort Optimizations

4. **Cache current frame** — Avoid repeated lookup
   - Store raw pointer to current frame
   - Update on call/return only
   - Expected: 10-20% speedup

5. **Batch shutdown checks** — Reduce per-instruction overhead
   - Check every N instructions
   - Expected: 5-10% speedup

6. **Inline hot paths** — Use #[inline(always)] strategically
   - Profile to identify hottest instruction handlers
   - Inline small, frequently-called functions
   - Expected: 5-15% speedup

### Phase 3: Architectural Changes (High Effort, High Reward)

7. **Threaded dispatch** — Use computed goto equivalent
   - Replace match with jump table
   - Requires unsafe Rust
   - Expected: 20-50% speedup

8. **NaN boxing** — Pack values into 64 bits
   - Eliminate heap allocations for common types
   - Major refactor of GcValue
   - Expected: 2-5x speedup for numeric code

9. **Register-based allocation** — Avoid stack Vec operations
   - Pre-allocate register arrays
   - Use fixed-size arrays where possible
   - Expected: 10-30% speedup

---

## QUICK WINS CHECKLIST

- [x] Fix O(n) cons — **DONE** (used imbl::Vector for O(log n) cons)
- [x] Remove instruction clone — **DONE** (used raw pointer to avoid clone)
- [x] Remove register clone — **ANALYZED** (GcPtr is Copy, minimal benefit)
- [ ] Cache frame pointer — 10-20%
- [ ] Batch shutdown checks — 5-10%
- [ ] Add profiling for tail calls — debugging aid

## MEASUREMENT METHODOLOGY

To validate improvements, re-run these benchmarks:
```bash
./target/release/nostos --profile benchImprovements/04_sum_recursive.nos
./target/release/nostos --profile benchImprovements/06_list_map.nos
./target/release/nostos --profile benchImprovements/08_list_build.nos
./target/release/nostos --profile benchImprovements/22_ackermann.nos
```

Target improvements:
- sum_recursive: from 1,824µs/call to <10µs/call
- list_map: from 732µs/call to <10µs/call
- list_build: from 35ms to <1ms
- ackermann: from 214µs/call to <50µs/call
