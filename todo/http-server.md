# HTTP Server Architecture Issues and Fixes

## Summary

Two issues with current HTTP server examples:
1. **Memory leak**: Tail recursion inside try-catch is not optimized
2. **Mutex bottleneck**: Multiple workers compete for single channel receiver

## Issue 1: Tail Recursion in Try-Catch Not Optimized

### The Problem

The compiler explicitly disables tail call optimization inside try blocks (`compile.rs:7992`):

```rust
let try_result = self.compile_expr_tail(try_expr, false)?;  // is_tail = false
```

This is necessary because the exception handler frame must remain on the stack.

### Current Pattern (all examples use this)

```nostos
# From examples/http_server.nos, http_server_example.nos, http_server_test.nos
acceptLoop(server) = {
    result = try {
        req = Server.accept(server)
        handleRequest(req)
        acceptLoop(server)  # RECURSIVE CALL INSIDE TRY - NOT TAIL OPTIMIZED!
    } catch { e -> () }
    result
}
```

### Memory Impact

Tested with 10 million iterations:

| Version | Memory | Time |
|---------|--------|------|
| Plain tail recursion | 22 MB | 1.0s |
| Tail call in try-catch | 11 GB | 7.1s |

Each call adds ~1.1 KB stack frame that never gets freed. The VM uses heap-allocated frames so it doesn't crash - it just consumes memory until OOM.

### The Fix

Move recursion OUTSIDE the try block:

```nostos
acceptLoop(server) = {
    continue = try {
        req = Server.accept(server)
        handleRequest(req)
        true
    } catch { e -> false }

    if continue then acceptLoop(server)  # TRUE tail call - outside try!
    else ()
}
```

## Issue 2: Mutex Bottleneck in Worker Distribution

### Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RUST LAYER (axum)                        │
│                                                                 │
│  HTTP Request 1 ──▶ axum task 1 ──┐                             │
│  HTTP Request 2 ──▶ axum task 2 ──┼──▶ mpsc channel             │
│  HTTP Request 3 ──▶ axum task 3 ──┘         │                   │
│                                             ▼                   │
│                              Arc<Mutex<Receiver>>               │
└─────────────────────────────────────────────────────────────────┘
                                              │
┌─────────────────────────────────────────────────────────────────┐
│                       NOSTOS LAYER                              │
│                                                                 │
│  Worker 1 ──┐                                                   │
│  Worker 2 ──┼──▶ rx.lock().await.recv().await                   │
│  Worker 3 ──┤         │            │                            │
│  ...        │         │            └── Wait for request         │
│  Worker 8 ──┘         └── MUTEX: only ONE can wait at a time!   │
└─────────────────────────────────────────────────────────────────┘
```

### The Problem

From `io_runtime.rs` ServerAccept handler:

```rust
let result = rx.lock().await.recv().await;
//             ^^^^^^^^^^^^
//             Only one worker can hold this lock
```

- Axum spawns a task per request (good, handles thousands concurrently)
- All requests go into ONE mpsc channel
- Nostos workers compete for a mutex on the receiver
- **Only one worker at a time can actually wait for requests**
- Others block on the mutex, not on HTTP connections

## Recommended Solution: Spawn Per Request

Instead of N workers with recursive loops, use single acceptor that spawns per request:

```nostos
# Single acceptor, spawns handler per request
acceptLoop(server) = {
    continue = try {
        req = Server.accept(server)
        spawn { handleRequest(req) }  # New lightweight process per request
        true
    } catch { e -> false }

    if continue then acceptLoop(server)  # Proper tail call
    else ()
}

main() = {
    server = Server.bind(8080)
    acceptLoop(server)
}
```

### Benefits

1. **No memory leak** - Each handler is short-lived, exits after response
2. **No mutex contention** - Single acceptor, no competing workers
3. **Scales to thousands** - Tokio tasks are cheap (~few KB each)
4. **Proper tail recursion** - Recursion outside try-catch
5. **Natural backpressure** - If handlers slow down, accept queue grows
6. **Erlang/OTP pattern** - Proven at scale (WhatsApp, Discord, etc.)

### How It Works

```
HTTP Request ──▶ axum task ──▶ channel ──▶ ONE Nostos acceptor
                    │                              │
                    │                        spawn { handle(req1) }
                    │                        spawn { handle(req2) }
                    │                        spawn { handle(req3) }
                    │                              │
                    ◀──── oneshot response ◀───────┘
```

- Axum: 1 task per HTTP connection (already implemented)
- Nostos: 1 acceptor + 1 spawned process per request
- Tokio multiplexes all tasks across CPU cores
- No contention anywhere

## Files to Update

1. `examples/http_server.nos` - Multi-worker example
2. `examples/http_server_example.nos` - Simple example
3. `examples/http_server_test.nos` - Test server
4. `docs/tutorial/17_async_io_http.md` - Tutorial (if it shows the bad pattern)

## Testing Plan

1. Create new example with spawn-per-request pattern
2. Load test with `wrk` or `ab`:
   ```bash
   wrk -t4 -c100 -d10s http://localhost:8080/
   ```
3. Compare throughput and memory usage vs old pattern
4. Verify no memory growth over extended run

## Future Considerations

### Could We Fix Tail Calls in Try-Catch?

Theoretically possible but complex:
- Would need to transform `try { ... f() }` into continuation-passing style
- Exception handler would need to be stored separately
- Significant compiler complexity for edge case

The spawn-per-request pattern is simpler and more idiomatic.

### Alternative: Work-Stealing Queue

Instead of mutex on receiver, could use crossbeam work-stealing queue:
- Each worker has local queue
- Workers steal from others when idle
- No central contention point

But spawn-per-request is simpler and matches the Erlang model we're emulating.
