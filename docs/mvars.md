# Module-Level Mutable Variables (Mvars)

Mvars are thread-safe, module-level mutable variables that can be safely accessed from multiple concurrent processes. They provide shared mutable state with compile-time safety guarantees to prevent deadlocks and race conditions.

## Table of Contents

1. [User Guide](#user-guide)
   - [Declaring Mvars](#declaring-mvars)
   - [Reading and Writing](#reading-and-writing)
   - [The Safe Pattern](#the-safe-pattern)
   - [Common Patterns](#common-patterns)
   - [Error Messages](#error-messages)
2. [Implementation Details](#implementation-details)
   - [Architecture Overview](#architecture-overview)
   - [Function-Level Locking](#function-level-locking)
   - [Compile-Time Safety Analysis](#compile-time-safety-analysis)
   - [Lock Ordering](#lock-ordering)
   - [Instruction Set](#instruction-set)

---

## User Guide

### Declaring Mvars

Mvars are declared at module level using the `mvar` keyword with a type annotation:

```nostos
mvar counter: Int = 0
mvar name: String = "default"
mvar flag: Bool = false
mvar balance: Float = 100.0
mvar items: List[Int] = []
```

**Requirements:**
- Type annotation is required (for thread-safe shared state)
- Initial value must be a constant literal
- Mvars are module-scoped (visible within the module)

### Reading and Writing

Mvars are accessed like regular variables, but **only within functions**:

```nostos
mvar counter: Int = 0

# Read the mvar
get_counter() = counter

# Write to the mvar
set_counter(n) = {
    counter = n
    counter
}

# Read and write in same function (atomic operation)
increment() = {
    counter = counter + 1
    counter
}
```

### The Safe Pattern

Nostos enforces a **safe pattern** at compile time to prevent deadlocks:

> **Rule:** Functions that access an mvar cannot call other functions that access the same mvar.

This means mvar-accessing functions must be "leaf" functions.

#### Safe Examples

```nostos
mvar counter: Int = 0

# SAFE: Leaf function that accesses mvar
increment() = {
    counter = counter + 1
    counter
}

# SAFE: Leaf function that accesses mvar
get_counter() = counter

# SAFE: main doesn't access counter directly, only calls leaf functions
main() = {
    increment()
    increment()
    get_counter()  # Returns 2
}
```

```nostos
mvar a: Int = 0
mvar b: Int = 0

# SAFE: Different mvars in different functions
inc_a() = { a = a + 1; a }
inc_b() = { b = b + 1; b }

# SAFE: Calls functions accessing different mvars
main() = {
    inc_a()
    inc_b()
}
```

#### Unsafe Examples (Compile-Time Errors)

```nostos
mvar counter: Int = 0

increment() = {
    counter = counter + 1
    counter
}

# UNSAFE: Reads counter AND calls increment (which writes counter)
bad_function() = {
    increment()
    counter      # Error! Can't access mvar here
}
```

```nostos
mvar value: Int = 0

get_value() = value

# UNSAFE: Writes value AND calls get_value (which reads value)
update() = {
    value = get_value() + 1  # Error!
    value
}
```

```nostos
mvar counter: Int = 0

# UNSAFE: Recursive function accessing mvar
recursive_count(n) = {
    if n <= 0 then counter
    else {
        counter = counter + 1
        recursive_count(n - 1)  # Error! Recursive call with mvar access
    }
}
```

### Common Patterns

#### Counter Pattern

```nostos
mvar counter: Int = 0

increment() = { counter = counter + 1; counter }
decrement() = { counter = counter - 1; counter }
get() = counter
reset() = { counter = 0; counter }

main() = {
    increment()
    increment()
    get()  # Returns 2
}
```

#### State Machine Pattern

```nostos
mvar state: Int = 0

get_state() = state

next_state() = {
    s = state
    state = match s
        0 -> 1
        1 -> 2
        2 -> 0
        _ -> 0
    end
    state
}
```

#### Accumulator Pattern

```nostos
mvar total: Int = 0
mvar count: Int = 0

add_value(v) = {
    total = total + v
    count = count + 1
    total
}

get_total() = total
get_count() = count
```

#### Flag Pattern

```nostos
mvar enabled: Bool = false

is_enabled() = enabled

enable() = { enabled = true; enabled }
disable() = { enabled = false; enabled }
toggle() = { enabled = !enabled; enabled }
```

### Error Messages

When you violate the safe pattern, you'll see:

```
Error: Mvar safety violation detected:
  - Potential deadlock: `bad_function` reads mvar `counter` and calls `increment` which writes to the same mvar

Functions that access an mvar cannot call other functions that access
the same mvar. This prevents deadlocks with function-level locking.

To fix: Restructure your code so that mvar-accessing functions are
'leaf' functions that don't call other mvar-accessing functions.
```

---

## Implementation Details

### Architecture Overview

The mvar system consists of three main components:

1. **Compile-Time Analysis** (`crates/compiler/src/compile.rs`)
   - Analyzes AST to find all mvar accesses in each function
   - Detects unsafe patterns (caller-callee conflicts, recursion)
   - Determines lock types (read vs write) for each function

2. **Lock Instructions** (`crates/vm/src/value.rs`)
   - `MvarLock(name_idx, is_write)` - Acquire lock at function entry
   - `MvarUnlock(name_idx, is_write)` - Release lock at function exit
   - `MvarRead(dst, name_idx)` - Read mvar (assumes lock held)
   - `MvarWrite(name_idx, src)` - Write mvar (assumes lock held)

3. **Runtime Locking** (`crates/vm/src/parallel.rs`)
   - Uses `parking_lot::RwLock` for each mvar
   - Implements raw lock/unlock for manual control
   - Thread-safe value conversion between local and shared heaps

### Function-Level Locking

Unlike per-instruction locking (which has race conditions), Nostos uses **function-level locking**:

```
┌─────────────────────────────────────────────────┐
│ increment() = {                                 │
│   MvarLock("counter", write)    ← Lock at entry │
│   ...                                           │
│   counter = counter + 1         ← Atomic!       │
│   ...                                           │
│   MvarUnlock("counter", write)  ← Unlock at exit│
│   Return                                        │
│ }                                               │
└─────────────────────────────────────────────────┘
```

This ensures that operations like `counter = counter + 1` are atomic - no other process can modify `counter` between the read and write.

### Compile-Time Safety Analysis

The compiler performs these checks:

1. **AST Pre-Analysis**: Before compiling a function, analyze its body to find all mvar accesses:

```rust
fn analyze_fn_def_mvar_access(&self, def: &FnDef) -> (HashSet<String>, HashSet<String>) {
    // Returns (reads, writes) for all mvars accessed in function
}
```

2. **Build Call Graph**: Track which functions call which other functions:

```rust
fn_calls: HashMap<String, HashSet<String>>  // fn_name -> called functions
fn_mvar_accesses: HashMap<String, (HashSet<String>, HashSet<String>)>  // fn_name -> (reads, writes)
```

3. **Deadlock Detection**: Check for violations:

```rust
fn check_mvar_deadlocks(&self) -> Vec<String> {
    // For each function that accesses mvars:
    //   For each function it calls (transitively):
    //     If callee accesses same mvar -> ERROR
    //   If function calls itself (recursion) -> ERROR
}
```

### Lock Ordering

When a function accesses multiple mvars, locks are acquired in **alphabetical order** to prevent deadlocks:

```nostos
mvar x: Int = 0
mvar y: Int = 0
mvar z: Int = 0

swap_all() = {
    # Compiler emits locks in order: x, y, z
    temp = x
    x = y
    y = z
    z = temp
}
```

Generated instructions:
```
MvarLock("x", write)    # Always in alphabetical order
MvarLock("y", write)
MvarLock("z", write)
... body ...
MvarUnlock("z", write)  # Reverse order for unlock
MvarUnlock("y", write)
MvarUnlock("x", write)
Return
```

This prevents the classic deadlock scenario:
- Process A: Lock X, then try to lock Y
- Process B: Lock Y, then try to lock X
- Result: Deadlock!

With sorted ordering, both processes always lock X first, then Y.

### Instruction Set

#### MvarLock(name_idx, is_write)

Acquires a lock on the mvar at function entry.

```rust
MvarLock(name_idx, is_write) => {
    let name = &constants[*name_idx as usize];  // Get mvar name from constants
    let var = self.shared.mvars.get(name)?;      // Get RwLock
    unsafe {
        if *is_write {
            var.raw().lock_exclusive();  // Write lock
        } else {
            var.raw().lock_shared();     // Read lock
        }
    }
}
```

#### MvarUnlock(name_idx, is_write)

Releases the lock before function returns.

```rust
MvarUnlock(name_idx, is_write) => {
    let var = self.shared.mvars.get(name)?;
    unsafe {
        if *is_write {
            var.raw().unlock_exclusive();
        } else {
            var.raw().unlock_shared();
        }
    }
}
```

#### MvarRead(dst, name_idx)

Reads the mvar value into a register. **Assumes lock is already held.**

```rust
MvarRead(dst, name_idx) => {
    let var = self.shared.mvars.get(name)?;
    let value = unsafe { &*var.data_ptr() };  // Direct pointer access
    // Convert from shared heap to local heap
    let local_value = self.convert_to_local(value);
    registers[*dst] = local_value;
}
```

#### MvarWrite(name_idx, src)

Writes a register value to the mvar. **Assumes write lock is already held.**

```rust
MvarWrite(name_idx, src) => {
    let var = self.shared.mvars.get(name)?;
    let value = &registers[*src];
    // Convert from local heap to shared heap (thread-safe copy)
    let safe_value = self.convert_to_shared(value);
    unsafe { *var.data_ptr() = safe_value; }
}
```

### Thread Safety

The mvar system ensures thread safety through:

1. **RwLock**: Each mvar is wrapped in `parking_lot::RwLock<Value>`
2. **Value Conversion**: Values are deep-copied between local and shared heaps
3. **Lock Ordering**: Consistent acquisition order prevents deadlocks
4. **Compile-Time Checks**: Unsafe patterns are rejected before runtime

### Performance Considerations

- **Function-level locking**: Locks are held for the entire function body, which may reduce concurrency but ensures atomicity
- **Read vs Write locks**: Multiple readers can access simultaneously; writers have exclusive access
- **Lock ordering overhead**: Minimal - just sorting a small list of mvar names at compile time

---

## Testing

The mvar implementation has comprehensive tests in `tests/mvar/`:

- `tests/mvar/*.nos` - Basic safe patterns (13 tests)
- `tests/mvar/reject/*.nos` - Compile-time rejection (19 tests)
- `tests/mvar/concurrent/*.nos` - Multithreaded tests (16 tests)
- `tests/mvar/edge_cases/*.nos` - Edge cases (13 tests)

Run all tests:
```bash
for f in tests/mvar/**/*.nos; do ./target/release/nostos "$f"; done
```
