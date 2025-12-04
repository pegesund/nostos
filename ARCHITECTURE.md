# Nostos Runtime Architecture

This document outlines the planned architecture for GC, concurrency, and JIT compilation in Nostos.

## Design Philosophy

**Erlang-style shared-nothing concurrency** with:
- Lightweight processes with private heaps
- Message passing for communication
- Per-process garbage collection
- Implicit async (no function coloring)
- Preemptive scheduling via reduction counting

**Cranelift for JIT** compilation when the interpreter becomes a bottleneck.

---

## Current State

### Value Representation (`crates/vm/src/value.rs`)

```
Immediate (unboxed):     Heap-allocated (Rc<T>):
├── Unit                 ├── String
├── Bool                 ├── List
├── Int (i64)            ├── Array (mutable)
├── Float (f64)          ├── Tuple
└── Char                 ├── Map
                         ├── Set
                         ├── Record
                         ├── Variant
                         ├── Function
                         ├── Closure
                         └── NativeFunction
```

**Good:**
- Clear immediate vs heap separation (JIT-friendly)
- Pid/RefId types exist for concurrency
- JitFunction placeholder already defined
- Concurrency bytecode instructions defined (Spawn, Send, Receive, etc.)

**Needs work:**
- Uses `Rc` (single-threaded, no GC)
- Concurrency instructions not implemented
- No GC - relies on Rust's RAII (leaks cycles)

---

## Phase 1: Garbage Collection Foundation

### Goal
Replace `Rc<T>` with GC-managed heap objects.

### Approach: Custom Mark-and-Sweep

We will write our own GC for several reasons:
1. **Educational** - understand the system fully
2. **Control** - optimize for Nostos semantics
3. **Per-process heaps** - Erlang-style requires process-local GC
4. **Simplicity** - basic mark-sweep is ~500-1000 lines

### Design

```rust
/// A GC-managed pointer
pub struct GcPtr<T> {
    index: u32,        // Index into heap
    _marker: PhantomData<T>,
}

/// Per-process heap
pub struct Heap {
    /// Object storage
    objects: Vec<Option<GcObject>>,
    /// Free list for allocation
    free_list: Vec<u32>,
    /// Root set (registers, stack, globals)
    roots: Vec<u32>,
    /// Bytes allocated since last GC
    bytes_since_gc: usize,
    /// Threshold to trigger GC
    gc_threshold: usize,
}

/// A heap object with GC metadata
pub struct GcObject {
    /// The actual value
    value: HeapValue,
    /// Mark bit for GC
    marked: bool,
    /// Size in bytes (for memory pressure)
    size: usize,
}

/// Heap-allocated value variants
pub enum HeapValue {
    String(String),
    List(Vec<GcPtr<Value>>),
    Record { fields: Vec<GcPtr<Value>>, ... },
    Closure { captures: Vec<GcPtr<Value>>, ... },
    // etc.
}
```

### GC Cycle

```
1. Mark Phase:
   - Start from roots (registers, stack, globals)
   - Recursively mark all reachable objects

2. Sweep Phase:
   - Iterate all objects
   - Free unmarked objects
   - Add to free list

3. Triggering:
   - After N bytes allocated
   - On process yield (reduction boundary)
```

### Value Representation Change

```rust
// Before (current)
pub enum Value {
    Int(i64),
    String(Rc<String>),      // Rc
    List(Rc<Vec<Value>>),    // Rc
    ...
}

// After (with GC)
pub enum Value {
    Int(i64),
    String(GcPtr<StringObj>),  // GC pointer
    List(GcPtr<ListObj>),      // GC pointer
    ...
}
```

### Tasks

- [ ] Define `Heap` struct with allocation/collection
- [ ] Define `GcPtr<T>` smart pointer
- [ ] Convert `Value` variants from `Rc` to `GcPtr`
- [ ] Add root tracking to VM (registers, stack)
- [ ] Implement mark phase
- [ ] Implement sweep phase
- [ ] Add GC trigger points (allocation threshold, function calls)
- [ ] Update all code creating heap values

---

## Phase 2: Process Abstraction

### Goal
Introduce lightweight processes with private heaps.

### Design

```rust
/// A lightweight process
pub struct Process {
    /// Unique process ID
    pub pid: Pid,
    /// Private heap (GC-managed)
    pub heap: Heap,
    /// Call stack
    pub frames: Vec<CallFrame>,
    /// Mailbox for messages
    pub mailbox: VecDeque<Value>,
    /// Process state
    pub state: ProcessState,
    /// Reduction counter
    pub reductions: usize,
    /// Links to other processes
    pub links: HashSet<Pid>,
    /// Monitors
    pub monitors: HashMap<RefId, Pid>,
}

pub enum ProcessState {
    Running,
    Waiting,      // In receive, waiting for message
    Suspended,    // Yielded, ready to run
    Exited(Value), // Terminated with value
}

/// Process scheduler
pub struct Scheduler {
    /// All processes
    processes: HashMap<Pid, Process>,
    /// Run queue (ready to execute)
    run_queue: VecDeque<Pid>,
    /// Waiting processes (in receive)
    waiting: HashSet<Pid>,
    /// Next PID to assign
    next_pid: u64,
}
```

### Message Passing

Messages are **deep copied** between process heaps:

```rust
impl Process {
    fn send(&mut self, target: Pid, msg: &Value, scheduler: &mut Scheduler) {
        // Deep copy msg into target's heap
        let target_proc = scheduler.get_mut(target);
        let copied = target_proc.heap.deep_copy(msg, &self.heap);
        target_proc.mailbox.push_back(copied);

        // Wake target if waiting
        if target_proc.state == ProcessState::Waiting {
            target_proc.state = ProcessState::Suspended;
            scheduler.run_queue.push_back(target);
        }
    }
}
```

### Preemptive Scheduling

Reduction counting at:
- Function calls
- Loop back-edges
- Allocation

```rust
const REDUCTIONS_PER_SLICE: usize = 2000;

impl Process {
    fn should_yield(&mut self) -> bool {
        self.reductions += 1;
        self.reductions >= REDUCTIONS_PER_SLICE
    }
}
```

### Tasks

- [ ] Define `Process` struct
- [ ] Define `Scheduler` struct
- [ ] Implement `spawn` instruction
- [ ] Implement `send` with deep copy
- [ ] Implement `receive` with pattern matching
- [ ] Implement reduction counting
- [ ] Implement round-robin scheduling
- [ ] Add process linking and monitoring

---

## Phase 3: Concurrency Primitives

### Syntax

```
# Spawn a process
pid = spawn(() -> heavy_work())

# Spawn with link (crash propagation)
pid = spawn_link(() -> risky_work())

# Send message
send(pid, {self(), :request, data})

# Receive with pattern matching
result = receive
    {from, :response, data} -> data
    {from, :error, err} -> handle_error(err)
    after 5000 -> timeout_error()
end

# Higher-level: async/await style
future = async expensive_computation()
result = await(future)
```

### Standard Library

```
# Parallel map
pmap : (List[a], a -> b) -> List[b]
pmap(items, f) = {
    pids = map(items, item -> spawn(() -> f(item)))
    map(pids, pid -> await(pid))
}

# Task module (like Elixir)
Task.async : (() -> a) -> Task[a]
Task.await : Task[a] -> a
Task.yield : (Task[a], Int) -> Option[a]
```

---

## Phase 4: JIT Compilation with Cranelift

### Prerequisites
- Stable GC with safepoints
- Working concurrency model

### Design

```
┌─────────────────────────────────────────┐
│              Compilation Tiers          │
├─────────────────────────────────────────┤
│ Tier 0: Interpreter (current VM)        │
│         - All code starts here          │
│         - Profiles hot functions        │
├─────────────────────────────────────────┤
│ Tier 1: Baseline JIT (Cranelift)        │
│         - Fast compilation              │
│         - Moderate optimization         │
│         - For warm functions            │
├─────────────────────────────────────────┤
│ Tier 2: Optimizing JIT (future)         │
│         - Aggressive optimization       │
│         - Inlining, specialization      │
│         - For hot functions             │
└─────────────────────────────────────────┘
```

### Safepoints

JIT code must include safepoints for:
1. **GC** - known points where collection can occur
2. **Preemption** - yield for scheduler
3. **Deoptimization** - fall back to interpreter

Locations:
- Function prologues
- Loop back-edges
- Allocation sites
- Call sites

### Stack Maps

JIT must report which stack slots/registers hold GC pointers:

```rust
pub struct StackMap {
    /// Instruction offset where this map applies
    pub offset: usize,
    /// Which slots contain GC pointers
    pub gc_slots: Vec<u32>,
    /// Which registers contain GC pointers
    pub gc_regs: Vec<u8>,
}
```

### Tasks

- [ ] Add function call counting (for hot detection)
- [ ] Integrate Cranelift crate
- [ ] Implement bytecode → Cranelift IR translation
- [ ] Generate stack maps
- [ ] Emit safepoint checks
- [ ] Implement code cache management
- [ ] Add deoptimization support

---

## Dependencies to Add

```toml
# Phase 1: GC
# (no external deps - custom implementation)

# Phase 2: Concurrency
crossbeam = "0.8"          # Lock-free data structures
parking_lot = "0.12"       # Fast mutexes

# Phase 4: JIT
cranelift-codegen = "0.95"
cranelift-frontend = "0.95"
cranelift-module = "0.95"
cranelift-jit = "0.95"
```

---

## Migration Strategy

### Step 1: GC without breaking changes
- Implement GC alongside current Rc system
- Feature flag: `--features gc`
- Gradually migrate tests

### Step 2: Process abstraction
- Current VM becomes "process execution engine"
- Single process works exactly as before
- Add multi-process support incrementally

### Step 3: Parallel execution
- Add OS thread pool for scheduler
- Work-stealing between threads
- Each thread runs processes

### Step 4: JIT
- Hot functions JIT-compiled
- Cold functions stay interpreted
- Transparent to user code

---

## Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │           Scheduler             │
                    │  (N OS threads, work-stealing)  │
                    └─────────────────────────────────┘
                           │         │         │
              ┌────────────┼─────────┼─────────┼────────────┐
              │            │         │         │            │
         ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
         │Process 1│  │Process 2│  │Process 3│  │Process N│
         ├─────────┤  ├─────────┤  ├─────────┤  ├─────────┤
         │ Mailbox │  │ Mailbox │  │ Mailbox │  │ Mailbox │
         │ Stack   │  │ Stack   │  │ Stack   │  │ Stack   │
         │ Heap    │  │ Heap    │  │ Heap    │  │ Heap    │
         │ GC      │  │ GC      │  │ GC      │  │ GC      │
         └─────────┘  └─────────┘  └─────────┘  └─────────┘
              │            │           │            │
              └────────────┴─────┬─────┴────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Shared Code Cache     │
                    │  (JIT-compiled, r/o)    │
                    │      [Cranelift]        │
                    └─────────────────────────┘
```

---

## Open Questions

1. **Immutable sharing optimization?**
   - Could share immutable values between processes without copying
   - Requires tracking mutability at value level
   - Trade-off: complexity vs performance

2. **Binary/ByteArray handling?**
   - Large binaries are expensive to copy
   - Consider reference-counted binaries (like Erlang)
   - Separate "shared binary heap"?

3. **Native function integration?**
   - Native functions currently use `Rc<NativeFn>`
   - Need to decide: per-process or shared?
   - Thread-safety concerns

4. **Selective receive optimization?**
   - Erlang optimizes `receive` with reference matching
   - Worth implementing?

---

## References

- [BEAM VM Internals](http://beam-wisdoms.clau.se/en/latest/)
- [Cranelift Documentation](https://cranelift.readthedocs.io/)
- [mmtk-core](https://github.com/mmtk/mmtk-core)
- [Writing a GC in Rust](https://rust-unofficial.github.io/too-many-lists/)
