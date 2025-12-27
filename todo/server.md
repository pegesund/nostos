# Web Server Enhancements

## Features to Implement

### 1. Parameter Parsing (GET/POST)
- [x] Parse query string parameters from URL (e.g., `/search?q=hello&page=1`)
- [x] Parse POST body parameters (form-urlencoded)
- [ ] Parse JSON POST bodies
- [x] Expose as `HttpRequest.queryParams` (List of tuples)
- [x] Expose as `HttpRequest.formParams` (List of tuples)

### 2. Cookie Handling
- [x] Parse incoming cookies from request headers
- [x] Expose as `HttpRequest.cookies` (List of tuples)
- [ ] Helper to set cookies in response: `Server.setCookie(name, value, options)`
- [ ] Cookie options: expires, maxAge, path, domain, secure, httpOnly

### 3. Routing (Rust-side for speed)
- [x] Pattern-based routing with path parameters (e.g., `/users/:id/posts/:postId`)
- [x] `Server.matchPath(path, pattern)` returns list of extracted params or empty if no match
- [ ] Method-based routing (GET, POST, PUT, DELETE, etc.) - use req.method in Nostos
- [ ] Multiple pattern matching (can use list of patterns in Nostos)

### 4. WebSockets
- [x] WebSocket upgrade handling (via isWebSocket field and status 101 response)
- [x] `WebSocket.send(requestId, message)` -> () (uses request ID as handle)
- [x] `WebSocket.receive(requestId)` -> String (blocking, throws on close)
- [x] `WebSocket.close(requestId)` -> ()

To accept a WebSocket connection, respond with status 101. The connection is then
available via the request ID. Example:
```
handle(req) = {
    if req.isWebSocket then {
        # Accept the WebSocket upgrade
        Server.respond(req.id, 101, [], "")

        # Now we can send/receive
        msg = WebSocket.receive(req.id)
        WebSocket.send(req.id, "Echo: " ++ msg)
        WebSocket.close(req.id)
    } else {
        Server.respond(req.id, 200, [], "Hello!")
    }
}
```

## Implementation Order
1. Parameter parsing (foundational, needed for other features)
2. Cookie handling (builds on parameter parsing patterns)
3. Routing (requires understanding request flow)
4. WebSockets (most complex, separate protocol)

## Testing Strategy
- Spawn server on test port
- Make HTTP client requests to test endpoints
- Shutdown server via /shutdown endpoint
- See examples/http_server.nos for pattern

## Current Server Builtins
- `Server.bind(port)` -> Int (server handle)
- `Server.accept(server)` -> HttpRequest
- `Server.respond(reqId, status, headers, body)` -> ()
- `Server.close(server)` -> ()
- `Server.matchPath(path, pattern)` -> [(String, String)] # NEW: route matching

## WebSocket Builtins
- `WebSocket.send(requestId, message)` -> () # Send text message
- `WebSocket.receive(requestId)` -> String   # Receive message (blocking)
- `WebSocket.close(requestId)` -> ()         # Close connection

## HttpRequest Record
- id: Int
- method: String
- path: String
- headers: [(String, String)]
- body: String
- queryParams: [(String, String)]  # NEW: parsed from URL query string
- cookies: [(String, String)]      # NEW: parsed from Cookie header
- formParams: [(String, String)]   # NEW: parsed from form-urlencoded POST body
- isWebSocket: Bool                # NEW: true if this is a WebSocket upgrade request

## Notes
- Implement in Rust for speed
- Expose all functionality to Nostos
- Test each feature incrementally
