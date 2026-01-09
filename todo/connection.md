# Connecting TUI to Running Nostos Process

## Overview

This document outlines how to connect an external TUI/client to an already running Nostos process, enabling remote REPL, debugging, and monitoring capabilities.

## Use Cases

1. **Remote REPL** - Send code to execute in a running process, get results back
2. **Debugging** - Inspect state, set breakpoints, step through code
3. **Monitoring** - Watch variables, process stats, logs, spawned processes
4. **Hot reloading** - Push new function definitions to running process
5. **Admin console** - Control/inspect long-running servers

## Available Building Blocks

### Already Implemented

| Mechanism | Location | Description |
|-----------|----------|-------------|
| HTTP Server | `crates/vm/src/io_runtime.rs` | Full HTTP server with `Server.bind`, `Server.accept`, `Server.respond` |
| HTTP Client | `crates/vm/src/io_runtime.rs` | `Http.get`, `Http.post`, etc. |
| PostgreSQL LISTEN/NOTIFY | `crates/vm/src/io_runtime.rs` | Pub/sub via database |
| Process Messaging | `crates/vm/src/async_vm.rs` | `spawn`, `<-`, `receive` (intra-VM only) |
| JSON parsing | `stdlib/json.nos` | `Json.parse`, `Json.stringify` |

### Not Yet Implemented

| Mechanism | Effort | Notes |
|-----------|--------|-------|
| WebSocket | Medium | Would need new builtins in io_runtime.rs |
| Unix Domain Sockets | Medium | Platform-specific, fast for local |
| Raw TCP Sockets | Medium | Lower level than HTTP |

## Recommended Architecture

### Phase 1: HTTP-based Debug Server (Use existing builtins)

```
┌─────────────────┐         HTTP/JSON         ┌─────────────────────────┐
│  TUI / Client   │  ──────────────────────>  │  Running Nostos Process │
│                 │  <──────────────────────  │  + DebugServer module   │
└─────────────────┘                           └─────────────────────────┘
```

The running Nostos process imports a debug server module that exposes HTTP endpoints.

#### Protocol Design

```
POST /eval
  Request:  { "code": "1 + 2" }
  Response: { "id": "eval-123", "status": "pending" }

GET /eval/{id}
  Response: { "status": "complete", "result": "3", "type": "Int" }
        or: { "status": "running", "output": ["print line 1", "print line 2"] }
        or: { "status": "error", "error": "undefined variable: x" }

GET /state
  Response: {
    "mvars": { "counter": 42, "config": {...} },
    "processes": [
      { "pid": 1, "state": "running", "function": "main" },
      { "pid": 2, "state": "waiting", "function": "serverLoop" }
    ]
  }

GET /processes
  Response: [
    { "pid": 1, "state": "running", "reductions": 15000 },
    { "pid": 2, "state": "waiting", "mailbox_size": 0 }
  ]

POST /interrupt/{eval_id}
  Response: { "status": "interrupted" }
```

#### Handling Long-Running Evaluations

Problem: `POST /eval` with code that takes 30 seconds blocks the HTTP response.

Solution: **Async eval pattern**
1. `POST /eval` returns immediately with an `eval_id`
2. Evaluation runs in spawned process
3. Client polls `GET /eval/{id}` for result
4. Or: Long-polling with timeout (server holds connection until result ready)

```nostos
# Pseudo-code for eval endpoint handler
handleEval(request) = {
  code = Json.parse(request.body).code
  evalId = generateId()

  # Store result destination
  evalResults.insert(evalId, { status: "pending", output: [] })

  # Spawn evaluation in background
  spawn {
    result = try {
      Ok(eval(code))  # Need eval builtin or workaround
    } catch e {
      Err(e.message)
    }
    evalResults.update(evalId, { status: "complete", result: result })
  }

  respond(200, Json.stringify({ id: evalId, status: "pending" }))
}
```

### Phase 2: WebSocket Support (Future)

Add WebSocket builtins for true bidirectional streaming:

```nostos
# Server side
ws = WebSocket.upgrade(httpRequest)
WebSocket.send(ws, message)
received = WebSocket.receive(ws)  # Blocks until message

# Client side
ws = WebSocket.connect("ws://localhost:8888/debug")
WebSocket.send(ws, { type: "eval", code: "1 + 1" })
result = WebSocket.receive(ws)
```

Benefits:
- Push results as they happen (no polling)
- Stream print output in real-time
- Lower latency than HTTP round-trips
- Natural fit for REPL interaction

## Implementation Plan

### Step 1: Create `stdlib/debug_server.nos`

```nostos
# stdlib/debug_server.nos

# Start debug server on given port
startDebugServer(port: Int) = {
  server = Server.bind(port)
  spawn { debugServerLoop(server) }
  print("Debug server listening on port " ++ show(port))
}

debugServerLoop(server: ServerHandle) = {
  request = Server.accept(server)
  spawn { handleRequest(request) }  # Handle each request in own process
  debugServerLoop(server)
}

handleRequest(request: HttpRequest) = {
  route = parseRoute(request.path)
  response = match route {
    Route.Eval -> handleEval(request)
    Route.EvalResult(id) -> handleEvalResult(id)
    Route.State -> handleState()
    Route.Processes -> handleProcesses()
    _ -> { status: 404, body: "Not found" }
  }
  Server.respond(request.id, response.status, defaultHeaders(), response.body)
}

# ... endpoint handlers ...
```

### Step 2: Add Required Builtins (if needed)

The key missing piece is **dynamic code evaluation**. Options:

1. **String eval builtin** - `eval(codeString)` returns result
   - Security concern, but useful for REPL
   - Would need to be added in Rust

2. **Pre-registered command handlers** - No dynamic eval
   - Process registers handlers: `registerHandler("reload", fn)`
   - Safer but less flexible

3. **Hot module loading** - Load new .nos files at runtime
   - `loadModule("path/to/module.nos")`
   - Adds functions/types to running process

For Phase 1 without dynamic eval, focus on:
- State inspection (mvars, processes)
- Pre-defined commands
- Module reloading

### Step 3: Create TUI Client

Options:

**A. Rust TUI (extend existing)**
- Add "Connect to remote" mode to existing TUI
- HTTP client calls to debug server
- Reuse existing UI components

**B. Nostos TUI (new)**
- Write TUI entirely in Nostos using HTTP client
- Would need terminal/curses builtins (not yet available)
- More dogfooding but bigger effort

**C. Web UI**
- Debug server serves HTML/JS interface
- Browser-based debugging
- No additional client needed

Recommendation: Start with **A** (extend existing Rust TUI) for quickest path.

### Step 4: Protocol Refinements

Add authentication:
```
POST /auth
  Request:  { "token": "secret-token" }
  Response: { "session": "sess-abc123" }

# All subsequent requests include:
Header: X-Debug-Session: sess-abc123
```

Add streaming output:
```
GET /eval/{id}/stream
  Response: Server-Sent Events stream

  event: output
  data: {"line": "Processing item 1..."}

  event: output
  data: {"line": "Processing item 2..."}

  event: complete
  data: {"result": "Done", "type": "String"}
```

## Alternative: PostgreSQL as Message Bus

For scenarios where HTTP isn't ideal (firewalls, multiple clients):

```
┌───────────┐     ┌────────────┐     ┌─────────────────┐
│   TUI 1   │────>│            │<────│ Nostos Process  │
├───────────┤     │ PostgreSQL │     │ (LISTEN on      │
│   TUI 2   │────>│            │     │  'debug_cmd')   │
└───────────┘     └────────────┘     └─────────────────┘
                        │
              commands table
              results table
              LISTEN/NOTIFY
```

```nostos
# In running process
main() = {
  conn = Pg.listenConnect(connStr)
  Pg.listen(conn, "debug_cmd")
  debugLoop(conn)
}

debugLoop(conn) = {
  notification = Pg.awaitNotification(conn, 30000)
  match notification {
    Some((channel, payload)) -> {
      result = handleCommand(Json.parse(payload))
      Pg.notify(conn, "debug_result", Json.stringify(result))
    }
    None -> ()  # Timeout, continue waiting
  }
  debugLoop(conn)
}
```

Pros:
- Multiple TUIs can connect
- Commands/results are persisted
- Works across networks/firewalls
- Already fully implemented

Cons:
- Requires PostgreSQL running
- Higher latency
- Overkill for local development

## Open Questions

1. **Dynamic eval** - Do we add `eval(string)` builtin? Security implications?

2. **State serialization** - How to serialize arbitrary Nostos values to JSON?
   - Records/variants: Use existing `toJson` derive
   - Functions: Just show type signature
   - Processes: Show pid and state

3. **Breakpoint integration** - How does remote debugging interact with existing DebugSession?
   - Needs hooks in VM to pause/resume from external signal

4. **Hot reloading scope** - When loading new code:
   - Replace existing functions?
   - Add new functions only?
   - What about type changes?

5. **Multi-process debugging** - Running process may have many spawned processes
   - Which one receives eval commands?
   - How to target specific process?

## Minimal Viable Implementation

For a quick proof-of-concept without new builtins:

1. Running process starts HTTP server
2. Exposes `/state` endpoint showing mvars
3. Exposes `/command/{name}` for pre-registered handlers
4. TUI connects via HTTP, displays state, sends commands

```nostos
# Example: Controllable server with debug endpoint

var requestCount = 0

main() = {
  # Start main server
  server = Server.bind(8080)
  spawn { serverLoop(server) }

  # Start debug server
  debug = Server.bind(9999)
  spawn { debugLoop(debug) }

  # Keep main alive
  receive { _ -> () }
}

debugLoop(debug) = {
  req = Server.accept(debug)
  response = match req.path {
    "/state" -> Json.stringify({
      requestCount: requestCount,
      uptime: getUptime()
    })
    "/reset" -> {
      requestCount := 0
      "Counter reset"
    }
    _ -> "Unknown command"
  }
  Server.respond(req.id, 200, [], response)
  debugLoop(debug)
}
```

This works TODAY with no new code needed.

## Summary

| Approach | Effort | Flexibility | Best For |
|----------|--------|-------------|----------|
| HTTP + pre-defined commands | Low | Low | Quick MVP, monitoring |
| HTTP + dynamic eval | Medium | High | Full remote REPL |
| WebSocket | Medium | High | Real-time streaming |
| PostgreSQL bus | Low | Medium | Multi-client, persistence |

**Recommended path:**
1. Start with HTTP + pre-defined commands (works today)
2. Add dynamic eval builtin when needed
3. Add WebSocket for better streaming experience
4. Consider PostgreSQL for production monitoring scenarios
