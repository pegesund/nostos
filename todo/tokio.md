# Future: Migrate to Tokio Task-Based Process Scheduling

## Summary

Currently, processes (lightweight threads) are scheduled by a custom work-stealing scheduler.
Migrating to tokio tasks would simplify MVar locking, remove custom scheduler code, and
leverage tokio's highly optimized runtime.

## Current Architecture

```
WorkerPool
 ├── Worker Thread 1 ─┬── Process A (Running)
 │                    ├── Process B (Waiting)
 │                    └── Process C (Suspended)
 ├── Worker Thread 2 ─┬── Process D
 │                    └── Process E
 └── Shared Scheduler ─── work-stealing queues (crossbeam)

Custom State Machine (ProcessState):
 Running → Waiting → Running → Exited
 Manual yield, manual wake-up, manual scheduling
```

## Proposed Tokio Architecture

```
Tokio Runtime
 └── Each Process = Tokio Task
      ├── mailbox.recv().await   (natural yield)
      ├── mvar.read().await      (natural yield)
      └── sleep().await          (natural yield)

Tokio handles:
 ✓ Work stealing
 ✓ Task scheduling
 ✓ Async I/O
 ✓ Wake-up on channel/lock availability
```

## Benefits

1. **MVar locking becomes trivial** - just `.read().await` / `.write().await`
2. **No custom scheduler maintenance** - tokio handles work stealing
3. **No ProcessState enum** - task is running or awaiting
4. **No manual wake-up logic** - channels/locks wake tasks automatically
5. **No timer management** - `tokio::time::sleep().await`
6. **Better performance** - tokio is highly optimized
7. **Already using tokio** - for I/O, so natural fit

## Key Code Changes

### 1. Process Struct (Simplified)

```rust
// BEFORE: Manual state machine
pub struct Process {
    pub state: ProcessState,  // Running, Waiting, WaitingIO, etc.
    pub wake_time: Option<Instant>,
    // ... lots of state tracking
}

// AFTER: Tokio handles state
pub struct Process {
    pub pid: Pid,
    pub heap: Heap,
    pub frames: Vec<CallFrame>,
    pub mailbox: tokio::sync::mpsc::Receiver<Message>,
    pub held_mvar_locks: HashMap<String, bool>,
    // No state machine needed - task is running or awaiting
}
```

### 2. Spawn Instruction

```rust
// BEFORE: Add to scheduler queue
Instruction::Spawn(dst, func_reg, arg_regs) => {
    let pid = self.scheduler.allocate_pid();
    let process = Process::new(pid, func, args);
    self.scheduler.add_process(process);
    self.local_queue.push(pid);
}

// AFTER: tokio::spawn
Instruction::Spawn(dst, func_reg, arg_regs) => {
    let pid = allocate_pid();
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    register_mailbox(pid, tx);

    tokio::spawn(async move {
        let mut process = Process::new(pid, func, args, rx);
        run_process(&mut process).await
    });
}
```

### 3. Receive Instruction

```rust
// BEFORE: State machine + scheduler
Instruction::Receive(dst) => {
    if let Some(msg) = proc.mailbox.try_recv() {
        proc.regs[dst] = msg;
    } else {
        proc.state = ProcessState::Waiting;
        return ProcessResult::Waiting;
    }
}

// AFTER: Just await
Instruction::Receive(dst) => {
    let msg = proc.mailbox.recv().await?;  // Yields task naturally!
    proc.regs[dst] = msg;
}
```

### 4. MVar Access

```rust
// BEFORE: parking_lot blocking or try_lock
MvarRead(dst, name) => {
    let guard = mvar.read();  // Blocks OS thread!
}

// AFTER: tokio async
MvarRead(dst, name) => {
    let guard = mvar.read().await;  // Yields task, doesn't block thread!
}
```

### 5. Main Execution Loop

```rust
// AFTER: Async loop with natural yield
async fn run_process(process: &mut Process) -> Result<GcValue, RuntimeError> {
    loop {
        match instruction {
            Receive(dst) => {
                let msg = process.mailbox.recv().await?;
                process.regs[dst] = msg;
            }
            MvarRead(dst, name) => {
                let guard = mvars.get(name).read().await;
                process.regs[dst] = guard.to_gc_value(&mut process.heap);
            }
            // Pure instructions stay sync
            Add(dst, a, b) => {
                process.regs[dst] = process.regs[a] + process.regs[b];
            }
        }
    }
}
```

## Files Affected

| Component | Lines | Change Needed |
|-----------|-------|---------------|
| `worker.rs` | 3,694 | **Replace** - becomes thin wrapper around tokio spawn |
| `process.rs` | 945 | **Moderate** - remove state machine, keep heap/frames |
| `parallel.rs` | 9,828 | **Significant** - instruction execution becomes async |
| `gc.rs` | 3,709 | **Moderate** - work with async boundaries |
| `scheduler.rs` | ~500 | **Remove** - tokio handles this |
| **Total** | ~18k | ~40-50% touch |

## Challenges

### 1. GC During Await

```rust
// Problem: Process yields while holding heap references
async fn run_process(process: &mut Process) {
    let value = process.heap.alloc_string("hello");
    let msg = mailbox.recv().await;  // YIELD - can GC run here?
    // Is 'value' still valid?
}
```

Solutions:
- Pin heap during await (prevent GC from moving)
- GC only at explicit safe points (between instructions)
- Copy values to stack before await

### 2. Reduction Counting / Fairness

Currently we count "reductions" to ensure fair scheduling. With tokio:
- Could use `tokio::task::yield_now().await` periodically
- Or rely on tokio's cooperative scheduling
- May need custom yield points for CPU-bound code

### 3. Debugging

- Tokio stack traces are less clear than our explicit ProcessState
- May want to keep some state tracking for debugging/introspection

## Implementation Plan

### Phase 1: Prototype (2-3 days)
- Create async version of instruction execution loop
- Test with simple program (no spawn/receive)

### Phase 2: Convert Spawn/Receive (2-3 days)
- Replace Process state machine with tokio channels
- Spawn creates tokio task
- Receive uses `.recv().await`

### Phase 3: Async MVar Locks (1 day)
- Change `parking_lot::RwLock` to `tokio::sync::RwLock`
- Update MvarRead/MvarWrite to use `.await`

### Phase 4: Handle GC (3-5 days)
- Ensure GC works correctly across await points
- Test with concurrent programs that allocate heavily

### Phase 5: Cleanup (1 day)
- Remove old scheduler code
- Remove ProcessState enum
- Simplify worker.rs

### Phase 6: Testing (2-3 days)
- Run full test suite
- Performance benchmarks
- Fix edge cases

**Total estimate: ~2 weeks**

## MVar Considerations with Tokio

With tokio::sync::RwLock:

1. **Read locks CAN be held across await** - multiple readers OK, parking_lot allows cross-thread unlock
2. **Write locks SHOULD NOT be held across await** - blocks all other access, potential liveness issues
3. **No deadlock from lock ordering** - tokio's async locks don't block OS threads

The compile-time analysis (transitive write propagation) becomes less critical for correctness,
but still useful for:
- Performance hints (warn about holding locks across yields)
- Detecting potential liveness issues

## Alternative Considered: try_lock + Retry

Before committing to full tokio migration, we implemented a simpler approach:
- Use `try_read()` / `try_write()` (non-blocking)
- If lock unavailable, set process state to `WaitingForMvar` and yield
- Scheduler retries waiting processes

This works with current architecture but:
- Requires polling/retry (less efficient)
- Still needs manual scheduler logic
- MVar-waiting processes compete with message-waiting processes

The tokio migration is cleaner long-term but requires more upfront work.

## Decision

Implement try_lock + retry first (quick win), then migrate to tokio as a separate project.
