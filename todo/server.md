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
- [ ] WebSocket upgrade handling
- [ ] `Server.acceptWebSocket(req)` -> WebSocket handle
- [ ] `WebSocket.send(ws, message)`
- [ ] `WebSocket.receive(ws)` -> message (blocking)
- [ ] `WebSocket.close(ws)`

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

## HttpRequest Record
- id: Int
- method: String
- path: String
- headers: [(String, String)]
- body: String

## Notes
- Implement in Rust for speed
- Expose all functionality to Nostos
- Test each feature incrementally
