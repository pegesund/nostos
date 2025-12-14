# Process Introspection & Monitoring

Future feature: ability to inspect running processes, their state, variables, and resource usage.

## Current State

### What exists:

1. **PID-to-thread mapping** (`SharedState.pid_to_thread: DashMap<Pid, u16>`)
   - Tracks which thread owns each PID
   - Used for message routing
   - Effectively a "registry" of live PIDs

2. **Per-thread process tables** (each worker has `processes: HashMap<u64, Process>`)
   - Process state is thread-local
   - Contains: registers, stack frames, heap, mailbox, state
   - Not accessible from outside the worker thread

3. **Process struct contains:**
   - `frames: Vec<Frame>` - call stack with registers
   - `heap: Heap` - all allocated values
   - `mailbox: VecDeque<GcValue>` - pending messages
   - `state: ProcessState` - Running, Waiting, WaitingIO, Dead
   - `output: Vec<String>` - captured println output

### What's missing:

- No centralized "list all processes" API
- No CPU time / reduction counting per process
- No way to inspect a remote process's variables
- No process hierarchy tracking (parent/children)
- No memory usage stats per process

## Reference: Erlang's Approach

```erlang
erlang:processes()                    % -> [Pid]
erlang:process_info(Pid)              % -> [{status, running}, {heap_size, 1234}, ...]
erlang:process_info(Pid, reductions)  % -> {reductions, 42000}
```

## Design Considerations

### Challenge 1: Thread locality

Process state lives in worker threads. To inspect a process on thread 3 from the main thread, we'd need either:
- **Message-passing**: send "inspect request" to thread, thread responds with snapshot
- **Shared memory**: make some process state Arc-wrapped and readable cross-thread

### Challenge 2: What to expose?

| Field | Difficulty | Notes |
|-------|------------|-------|
| PID, status, parent PID | Easy | Already tracked |
| Message queue length | Easy | `mailbox.len()` |
| Heap size / memory | Medium | Need to track allocations |
| CPU time / reductions | Medium | Need to count instructions |
| Local variables | Hard | Need to map register numbers back to names |

### Challenge 3: Performance impact

- Counting reductions adds overhead to every instruction
- Tracking memory per-process adds overhead to every allocation
- Could make it optional (debug/profile mode)

## Proposed Implementation

### Phase 1: Basic Process Registry

1. Add `processes()` builtin - queries all threads for their PID lists
2. Add `process_info(pid)` builtin - returns basic info:
   ```
   { pid: Pid, status: String, mailbox_len: Int }
   ```

### Phase 2: Resource Tracking

1. Add `reductions: u64` counter to Process struct
2. Increment on each instruction (or each slice)
3. Add `heap_size()` method to Heap
4. Extend `process_info` to include:
   ```
   { pid, status, mailbox_len, reductions, heap_size }
   ```

### Phase 3: Process Hierarchy

1. Track `parent: Option<Pid>` in Process
2. Track `children: Vec<Pid>` or use links/monitors
3. Add `spawn_link`, `spawn_monitor` variants
4. Extend `process_info` with parent/children

### Phase 4: TUI Process Monitor

Add a "Process Monitor" panel (like htop for Nostos):

```
┌─ Process Monitor ────────────────────────────────────┐
│ PID    Status     Reductions   Heap    MQ   Function │
│ <0>    running    142,857      12KB    0    main     │
│ <1>    waiting    89,234       8KB     3    worker   │
│ <2>    io_wait    45,123       4KB     0    reader   │
│ <3>    running    12,456       2KB     0    compute  │
└──────────────────────────────────────────────────────┘
```

Features:
- Live refresh
- Sort by column
- Select process to inspect details
- Kill process option
- Show message queue contents

### Phase 5: Variable Inspection (Advanced)

To show local variables for a process:
1. Compiler needs to emit debug info (var name -> register mapping)
2. Store debug info in function metadata
3. When inspecting, look up current function's debug info
4. Map register values to variable names

## Files to Modify

- `crates/vm/src/process.rs` - add reductions counter, parent tracking
- `crates/vm/src/parallel.rs` - add process query instructions, cross-thread inspection
- `crates/vm/src/gc.rs` - add heap_size() method
- `crates/compiler/src/compile.rs` - emit debug info for variables
- `crates/cli/src/tui.rs` - add process monitor panel
