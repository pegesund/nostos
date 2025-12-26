# Asynchronous I/O & HTTP

Nostos provides fully asynchronous I/O operations for network and file access. You write code as if it were synchronous and blocking, but the runtime transparently handles scheduling other processes while I/O is in progress, ensuring high concurrency and responsiveness.

## HTTP Client

The built-in `Http` module allows making various web requests. These are non-blocking, meaning your program won't freeze while waiting for a response.

```nostos
fetch_ip() = {
    # Perform a GET request to get public IP
    # I/O operations throw exceptions on error - use try/catch to handle
    try {
        resp = Http.get("https://httpbin.org/ip")
        println("Your IP: " ++ resp.body)
    } catch { e ->
        println("Error fetching IP: " ++ e)
    }
}

main() = {
    fetch_ip()
    println("Client operations are non-blocking!")
}
```

## Parallel HTTP Requests

Leverage Nostos's lightweight processes to perform many HTTP requests concurrently. The runtime ensures efficient use of resources without complex async/await keywords.

```nostos
timed_request_worker(parent, id) = {
    println("Worker " ++ show(id) ++ " starting request...")
    # This HTTP call yields the process, allowing others to run
    # Exceptions are caught and reported
    try {
        data = Http.get("https://httpbin.org/delay/1")
        println("Worker " ++ show(id) ++ " done: status " ++ show(data.status))
        parent <- ("done", id)
    } catch { e ->
        println("Worker " ++ show(id) ++ " error: " ++ e)
        parent <- ("done", id)
    }
}

main() = {
    me = self()
    println("Spawning 5 HTTP requests with 1-second server-side delay each.")
    println("Total time will be ~1-2 seconds, not 5 seconds, due to parallelism.")

    # Spawn 5 workers; all requests are initiated almost simultaneously
    for i = 1 to 5 {
        spawn { timed_request_worker(me, i) }
    }

    # Collect results from all workers
    for i = 1 to 5 {
        receive { ("done", id) -> println("Request from worker " ++ show(id) ++ " completed.") }
    }
    println("All parallel requests finished!")
}
```

## HTTP Server

Build scalable web services with Nostos's non-blocking HTTP server. Spawn multiple worker processes to handle incoming requests concurrently on the same port.

```nostos
# Simple request handler
handle_request(req) = {
    path = req.path
    if path == "/" then {
        Server.respond(req.id, 200, [("Content-Type", "text/plain")], "Hello from Nostos!")
    } else if path == "/echo" then {
        Server.respond(req.id, 200, [("Content-Type", "text/plain")], req.body)
    } else {
        Server.respond(req.id, 404, [("Content-Type", "text/plain")], "Not Found")
    }
}

# Worker process continuously accepts and handles requests
server_worker(server_socket) = {
    # Server.accept yields if no connection is available, allowing other processes to run
    # Throws exception when server is closed
    result = try {
        request = Server.accept(server_socket)
        handle_request(request)
        server_worker(server_socket)  # Loop to accept next request
    } catch { e ->
        ()  # Exit when server is closed or on error
    }
    result
}

main() = {
    # Bind to port 8080 - throws on error
    try {
        server = Server.bind(8080)
        println("HTTP Server listening on port 8080. Spawning workers...")
        # Spawn multiple worker processes to handle incoming connections
        for i = 1 to 4 {
            spawn { server_worker(server) }
        }
        # Keep the main process alive (e.g., waiting for a shutdown signal)
        receive { _ -> () }
    } catch { e ->
        println("Failed to bind server: " ++ e)
    }
}
```
