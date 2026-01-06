# RWeb Thread Messaging Strategy

## Problem

In rweb, each WebSocket connection has its own session process that blocks on `WebSocket.recv(ws)`. External processes (e.g., background workers, other user sessions) cannot push messages to a connected client because the session process is blocked and can't receive from its mailbox.

## Current Architecture

```
Browser ◄──► WebSocket ◄──► Session Process (blocks on recv)
                                    │
                                    └── owns full WebSocket handle
```

- One process per connection
- Session loop: `loop { msg = WebSocket.recv(ws); handle(msg); respond(ws) }`
- No way for external processes to send to the browser

## Proposed Solution: Split WebSocket with Shared Writer

Split the WebSocket into read and write halves at connection time. Session process owns read-half, write-half is shared.

```
                              ┌─────────────────────┐
Browser ◄───────────────────► │     WebSocket       │
                              └─────────┬───────────┘
                                        │ split
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                ┌─────────────────┐         ┌─────────────────────┐
                │   Read Half     │         │ Write Half          │
                │ (session only)  │         │ Arc<Mutex<Writer>>  │
                └─────────────────┘         └─────────────────────┘
                         │                             │
                         │                    ┌────────┴────────┐
                         ▼                    │                 │
                ┌─────────────────┐    Session process    External processes
                │ Session Process │    (responses)        (push notifications)
                │ blocks on recv  │
                └─────────────────┘
```

## API Change

Current:
```nostos
rwebHandleRequest(request, sessionHandler)
# returns: pid (or unit)
```

New:
```nostos
rwebCreateSession(request, sessionHandler)
# returns: {pid: ProcessId, writer: WebSocketWriter}
```

External processes can then:
```nostos
WebSocket.send(writer, Json.encode({type: "push", event: "notification", data: ...}))
```

## Implementation Steps

### 1. Rust: Split WebSocket in builtin

In the WebSocket accept/upgrade code:
- Use `ws_stream.split()` to get `(write_half, read_half)`
- Wrap write-half in `Arc<Mutex<SplitSink<...>>>`
- Create two resource handles: one for read, one for write
- Or: single handle that internally manages both, with thread-safe send

### 2. Rust: WebSocket.send() thread safety

Ensure `WebSocket.send(writer, data)` builtin:
- Locks the mutex
- Sends the message
- Releases lock
- Works from any process/task

### 3. Nostos: Update rweb.nos

Modify session creation to return both pid and writer:
```nostos
rwebCreateSession(request, handler) = {
  {read, write} = WebSocket.split(ws)
  pid = spawn(() => sessionLoop(read, handler))
  {pid: pid, writer: write}
}
```

### 4. Nostos: Session loop uses writer

Session loop receives writer as parameter, uses it for responses:
```nostos
sessionLoop(reader, writer, handler) = {
  loop {
    msg = WebSocket.recv(reader)
    response = handler(msg)
    WebSocket.send(writer, response)
  }
}
```

### 5. Browser: Handle push messages

Update rweb.js to distinguish push from response:
```javascript
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "push") {
    dispatchPushEvent(msg);
  } else {
    handleResponse(msg);
  }
}
```

## Benefits

- No blocking Mvar
- No polling in session loop
- External processes can push anytime
- Minimal changes to existing code
- Thread-safe by design (mutex serializes writes)

## Open Questions

1. Should writer be a new type (`WebSocketWriter`) or reuse existing `WebSocket` type?
2. Error handling when connection closes while external process holds writer reference
3. Message format convention for push vs response (suggest `{type: "push", ...}`)
