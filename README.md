# Nostos

> *In Greek, Nostos (νόστος) means a hero's homecoming or return from a long journey.*

After wandering through callback hell, fighting with async/await, and battling race conditions—Nostos welcomes you home to a place where concurrent code is simple, readable, and just works.

**⚠️ Early Stage Software:** It is early days and version 1.0 is not ready. For brave developers and early adopters only.

---

## What Makes Nostos Different

### Code That Reads Like Poetry

```nos
# Pattern matching flows naturally
fibonacci(0) = 0
fibonacci(1) = 1
fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)

# Lists destructure elegantly
sum([]) = 0
sum([head | tail]) = head + sum(tail)

# Quicksort in 3 lines
quicksort([]) = []
quicksort([pivot | rest]) =
    quicksort(rest.filter(x => x < pivot)) ++ [pivot] ++ quicksort(rest.filter(x => x >= pivot))

# Pipes chain beautifully
result = users
    .filter(u => u.active)
    .map(u => u.name)
    .join(", ")
```

### Non-Blocking by Default

Every I/O operation yields automatically. No `async`, no `await`, no colored functions. Your code looks synchronous but runs concurrently.

```nos
# These HTTP requests run in parallel—no special syntax needed
fetchAll(urls) = urls.map(url => Http.get(url))

# Spawn 100,000 processes without breaking a sweat
main() = {
    pids = range(1, 100001).map(i => spawn(() => worker(i)))
    println("Spawned " ++ show(pids.length()) ++ " processes")
}
```

### Reactive Records

State changes propagate automatically. Build reactive UIs without the ceremony.

```nos
reactive Counter = { value: Int }

main() = {
    counter = Counter(0)

    # Changes trigger re-renders automatically
    counter.value = counter.value + 1
}
```

### Pattern Matching Rocks

Describe the shape you expect, not the steps to extract it. Pattern matching handles recursive data structures elegantly:

```nos
# A binary tree: either empty or a node with value and children
type Tree[T] = Empty | Node(T, Tree[T], Tree[T])

# Count nodes - pattern matching makes it trivial
size(Empty) = 0
size(Node(_, left, right)) = 1 + size(left) + size(right)

# Sum all values
sum(Empty) = 0
sum(Node(val, left, right)) = val + sum(left) + sum(right)

# Search with guards
contains(Empty, _) = false
contains(Node(val, _, _), target) when val == target = true
contains(Node(_, left, right), target) =
    contains(left, target) || contains(right, target)

# Build and query a tree
main() = {
    tree = Node(5,
        Node(3, Node(1, Empty, Empty), Empty),
        Node(8, Node(6, Empty, Empty), Node(9, Empty, Empty))
    )

    println("Size: " ++ show(size(tree)))      # 6
    println("Sum: " ++ show(sum(tree)))        # 32
    println("Has 6? " ++ show(contains(tree, 6)))  # true
}
```

The compiler ensures you handle every case. Add a new tree variant? The compiler reminds you to update every function. No forgotten cases, no runtime surprises.

### Living Development Environment

The REPL isn't just for experiments—it's your development cockpit:

- **Live reload**: Change code, see results instantly
- **Autocomplete**: Context-aware suggestions as you type
- **Inline errors**: Know what's wrong before you run
- **State inspection**: Peek inside running processes

```bash
$ nostos
Nostos REPL v0.1.0
>>> users = [{ name: "Alice", age: 30 }, { name: "Bob", age: 25 }]
>>> users.filter(u => u.age > 28).map(u => u.name)
["Alice"]
```

### VS Code That Understands Your Code

Not just syntax highlighting—true understanding:

- **Real-time error checking** as you type
- **Go to definition** across modules
- **Smart autocomplete** with type inference
- **Integrated REPL** in your editor
- **File status badges** showing compile state at a glance

---

## Some Batteries Included

### PostgreSQL

Query your database with minimal friction:

```nos
main() = {
    conn = Pg.connect("host=localhost dbname=mydb user=postgres password=secret")

    # Parameterized queries prevent SQL injection
    # Params: () for none, scalar for one, tuple/list for multiple
    rows = Pg.query(conn, "SELECT name, email FROM users WHERE age > $1 AND active = $2", (18, true))

    rows.map(row => println(row.0 ++ ": " ++ row.1))

    Pg.close(conn)
}
```

No slow, complex, or scary ORM. Just plain queries and safe types, powered by introspection.

**Typed Results** — Map query results to typed records:

```nos
use stdlib.db.{query}

type User = { name: String, email: String }

main() = {
    conn = Pg.connect("host=localhost dbname=mydb user=postgres password=secret")

    # Column order in SELECT must match field order in type
    users: List[User] = query[User](conn, "SELECT name, email FROM users", ())

    # Now use field names instead of positional access!
    users.map(u => println(u.name ++ ": " ++ u.email))

    Pg.close(conn)
}
```

**Features:**
- **Transactions** - `Pg.begin()`, `Pg.commit()`, `Pg.rollback()`
- **Connection pooling** - Automatic per-connection-string pooling
- **Prepared statements** - `Pg.prepare()`, `Pg.queryPrepared()`, `Pg.executePrepared()`
- **Vector search (pgvector)** - Native Float32Array support for embeddings
- **JSON/JSONB** - Direct JSON value support in queries
- **LISTEN/NOTIFY** - Real-time change notifications with `Pg.listen()`, `Pg.notify()`
- **Binary types** - Int, Float, Bool, String, arrays, and custom types
- **TLS/SSL** - Secure connections to cloud providers (Supabase, Neon, etc.)

### Reactive Web (RWeb)

Full-stack reactive web apps with server-side rendering and automatic DOM diffing:

```nos
use stdlib.rweb.*

reactive Todo = { text: String, done: Bool }
reactive State = { todos: List[Todo] }

sessionSetup(writerId) = {
    state = State([])

    renderPage = () => RHtml(div([
        h1("Todo App"),
        component("list", () => RHtml(
            ul(state.todos.map(t => li(t.text)))
        )),
        input(type: "text", dataAction: "add")
    ]))

    onAction = (action, params) => match action {
        "add" -> { state.todos = state.todos ++ [Todo(params.text, false)] }
        _ -> ()
    }

    (renderPage, onAction)
}

main() = startRWeb(8080, "Todos", sessionSetup)
```

### HTTP Server & Client

Production-ready networking:

```nos
# Server
handle(req) = match req.path {
    "/api/users" -> jsonResponse(getUsers()),
    "/health" -> textResponse("OK"),
    _ -> notFound()
}

main() = Server.start(8080, handle, workers: 8)

# Client
response = Http.get("https://api.example.com/data")
data = json.parse(response.body)
```

### TCP Sockets

Low-level when you need it:

```nos
server = Tcp.listen(9000)
client = Tcp.accept(server)
Tcp.send(client, "Hello!")
message = Tcp.receive(client)
```

### WebSockets

Real-time bidirectional communication:

```nos
# Server
handle(req) = {
    if WebSocket.isUpgrade(req) then {
        ws = WebSocket.accept(req.id)
        WebSocket.send(ws, "Welcome!")
        message = WebSocket.recv(ws)
        WebSocket.close(ws)
    } else respond404(req)
}

# Client
ws = WebSocket.connect("wss://echo.websocket.org")
WebSocket.send(ws, "Hello!")
response = WebSocket.recv(ws)
```

### Cryptography

Secure hashing and password storage:

```nos
# Hashing
sha = Crypto.sha256("password123")  # Hex string
sha512 = Crypto.sha512("data")

# Password hashing (bcrypt)
hash = Crypto.bcryptHash("password", 12)  # Cost factor 12
valid = Crypto.bcryptVerify("password", hash)  # true

# Random bytes for tokens/keys
token = Crypto.randomBytes(32)  # 32 random bytes as hex
```

### Regular Expressions

Pattern matching and text processing:

```nos
# Match and find
if Regex.matches("hello123", "\\d+") then println("Has numbers")
Regex.find("Price: $42.99", "\\$[0-9.]+")  # Some("$42.99")

# Replace
Regex.replace("hello world", "world", "Nostos")  # "hello Nostos"
Regex.replaceAll("a1b2c3", "\\d", "X")  # "aXbXcX"

# Split and capture
words = Regex.split("one,two,three", ",")  # ["one", "two", "three"]
```

### File I/O

Read and write files with ease:

```nos
# Simple read/write
content = File.readAll("config.txt")
File.writeAll("output.txt", "Hello, world!")
File.append("log.txt", "Error: connection failed\n")

# Streaming for large files
handle = File.open("large.dat", "r")
line = File.readLine(handle)
File.close(handle)
```

### JSON

Parse, generate, and transform:

```nos
data = json.parse('{"name": "Alice", "scores": [95, 87, 92]}')
name = data.get("name")  # "Alice"

output = json.stringify(#{ "status": "ok", "count": 42 })
```

### HTML Templating

Type-safe templates with built-in parameter names for common attributes:

```nos
use stdlib.html.{Html, render}

# Common attributes like class, id, style are built-in parameters
page(title, content) = Html(
    div(class: "container", id: "main", [
        header([
            h1(class: "title", title),
            nav(class: "nav", [
                a(href: "/home", "Home"),
                a(href: "/about", "About")
            ])
        ]),
        div(class: "content", content),
        button(
            "Submit",
            btnType: "submit",
            class: "btn-primary",
            dataAction: "submit-form"
        ),
        input(
            inputType: "email",
            placeholder: "Enter email",
            class: "input",
            name: "email"
        ),
        footer(class: "footer", [ p("© 2024") ])
    ])
)

html = render(page("Welcome", [p("Hello, world!")]))
```

Built-in parameters include `class`, `id`, `style`, `href`, `inputType`, `btnType`, `dataAction`, and more. Use `attrs: [("custom", "value")]` for non-standard attributes.

### Logging

Structured logging with levels:

```nos
import logging

log.info("Server started", port: 8080)
log.error("Connection failed", error: err, retry: 3)
```

---

## FFI: Extend With Native Code

When you need raw performance or existing libraries, Nostos extensions bridge to native code seamlessly:

```nos
# Load a native extension
import glam  # Linear algebra via Rust's glam crate

main() = {
    v1 = glam.vec3(1.0, 2.0, 3.0)
    v2 = glam.vec3(4.0, 5.0, 6.0)

    dot = v1.dot(v2)
    cross = v1.cross(v2)
    normalized = v1.normalize()
}
```

Extensions are Rust crates that expose functions to Nostos. The type system ensures safe interop:

```toml
# nostos.toml
[extensions]
glam = { git = "https://github.com/pegesund/nostos-glam" }
nalgebra = { git = "https://github.com/pegesund/nostos-nalgebra" }
```

### Available Extensions

| Extension | Description | Repository |
|-----------|-------------|------------|
| **glam** | Fast linear algebra (vectors, matrices, quaternions) | [nostos-glam](https://github.com/pegesund/nostos-glam) |
| **nalgebra** | Dynamic vectors and matrices for scientific computing | [nostos-nalgebra](https://github.com/pegesund/nostos-nalgebra) |
| **redis** | Redis client for caching and pub/sub | [nostos-redis](https://github.com/pegesund/nostos-redis) |
| **candle** | Machine learning with Hugging Face's Candle | [nostos-candle](https://github.com/pegesund/nostos-candle) |

### Nostlets (TUI Plugins)

Extend the REPL with custom panels:

| Nostlet | Description | Repository |
|---------|-------------|------------|
| **runtime-stats** | Live CPU, memory, and process statistics | [nostos-runtime-stats](https://github.com/pegesund/nostos-runtime-stats) |
| **community** | Community-contributed panels | [nostos-nostlets](https://github.com/pegesund/nostos-nostlets) |

---

## The Type System Stays Out of Your Way

Strong static typing with inference that actually works:

```nos
# Types are inferred...
numbers = [1, 2, 3]           # List[Int]
doubled = numbers.map(x => x * 2)  # List[Int]

# ...but you can be explicit when it helps
parseConfig(path: String) -> Result[Config, String] = {
    content = readFile(path)
    json.parse(content).mapErr(e => "Parse error: " ++ e)
}
```

### Traits for Polymorphism

```nos
trait Drawable
    draw(self) -> String
end

type Circle = { radius: Float }
type Square = { side: Float }

Circle: Drawable
    draw(self) = "●"
end

Square: Drawable
    draw(self) = "■"
end

# Works with any Drawable
render[T: Drawable](shapes: List[T]) = shapes.map(s => s.draw()).join(" ")
```

### Supertraits Build Hierarchies

```nos
trait Printable
    toString(self) -> String
end

trait Serializable: Printable  # Requires Printable
    toJson(self) -> String
end
```

---

## Getting Started

```bash
# Clone and build
git clone https://github.com/pegesund/nostos.git
cd nostos
cargo build --release

# Run the REPL
./target/release/nostos

# Run a program
./target/release/nostos examples/hello_server.nos

# Install VS Code extension
cd editors/vscode && npm install && npm run package
code --install-extension nostos-*.vsix
```

---

## Architecture

Built in Rust for reliability and performance:

| Crate | Purpose |
|-------|---------|
| `compiler` | Lexer, parser, type inference |
| `vm` | Register-based bytecode interpreter |
| `jit` | Cranelift-powered JIT compilation |
| `scheduler` | Lightweight process runtime |
| `repl` | Interactive environment & LSP |
| `lsp` | Language server for editors |

### Under the Hood

- **[Tokio](https://tokio.rs/) runtime** — All async I/O is powered by Tokio, the industry-standard async runtime for Rust. File operations, networking, timers, and process scheduling all run on Tokio's work-stealing thread pool.

- **[imbl](https://docs.rs/imbl/) persistent data structures** — Lists, maps, and sets are immutable by default, using structural sharing for efficient updates. No defensive copying, no surprise mutations. Functional programming without the performance penalty.

- **Hindley-Milner type inference** — Full type inference means you rarely write type annotations, but the compiler catches errors at compile time. Generics, traits, and higher-order functions all work seamlessly.

- **Register-based VM** — Unlike stack-based VMs, our register-based design reduces instruction count and enables better JIT optimization. Hot paths are compiled to native code via [Cranelift](https://cranelift.dev/).

- **[tokio-postgres](https://docs.rs/tokio-postgres/) + [deadpool](https://docs.rs/deadpool-postgres/)** — Native async PostgreSQL driver with connection pooling. No ORM overhead, just fast queries with automatic pooling.

- **[Axum](https://docs.rs/axum/) + [Reqwest](https://docs.rs/reqwest/)** — HTTP server and client built on Tokio. Production-ready with automatic keep-alive, connection pooling, and TLS support via [native-tls](https://docs.rs/native-tls/).

- **[tokio-tungstenite](https://docs.rs/tokio-tungstenite/)** — WebSocket client and server with TLS support. Real-time bidirectional communication with automatic frame handling.

- **[regex](https://docs.rs/regex/)** — Fast, safe regular expressions. Guaranteed linear time performance with no catastrophic backtracking.

- **[bcrypt](https://docs.rs/bcrypt/) + [sha2](https://docs.rs/sha2/)** — Cryptographic hashing for passwords and data integrity. Industry-standard algorithms with safe defaults.

- **[thirtyfour](https://docs.rs/thirtyfour/)** — Browser automation via Selenium WebDriver. End-to-end testing and web scraping with a clean async API.

- **Lightweight processes** — Each Nostos process is ~2KB of overhead. Spawn millions of them. They're scheduled cooperatively on the Tokio runtime, yielding at I/O boundaries.

- **Tail call optimization** — Recursive functions that return a function call directly are optimized to loops, enabling elegant recursive algorithms without stack overflow.

- **Tracing garbage collector** — Automatic memory management with a concurrent GC that minimizes pause times.

- **Built-in profiler** — Measure execution time with `:profile expr` in the REPL. Compare implementations, find bottlenecks, and watch JIT warmup effects in real time.

- **Interactive debugger** — Set breakpoints with `:debug fn`, step through calls, inspect arguments. Debug without leaving your REPL flow.

```bash
nostos> :profile fib(35)
Result: 9227465
Time: 42.3ms

nostos> :debug factorial
nostos> factorial(5)
[BREAK] factorial(5)
  Press Enter to continue, 'c' to skip remaining...
```

---

## Functional First, Pragmatism Always

Nostos is designed to **save you time**. That's the only metric that matters. We prefer functional patterns because they're often clearer and safer, but we won't force you into contortions when a mutable variable just makes sense.

### Immutable by Default

Data structures are immutable by default. Updates create new versions with structural sharing—fast and safe:

```nos
# Immutable list operations
numbers = [1, 2, 3]
doubled = numbers.map(x => x * 2)  # Creates new list
filtered = numbers.filter(x => x > 1)  # Original unchanged

# Immutable maps
config = %{"port": 8080, "host": "localhost"}
updated = Map.insert(config, "debug", true)  # New map, config unchanged
```

This prevents entire classes of bugs. No surprise mutations, no defensive copying needed.

### Mutable Variables When You Need Them

Sometimes a loop with a mutable accumulator is clearer than a fold. We're okay with that:

```nos
# Functional style - elegant for simple cases
sumFunctional(numbers) = numbers.reduce(0, (acc, x) => acc + x)

# Imperative style - clearer for complex logic
sumImperative(numbers) = {
    mut total = 0
    mut count = 0

    for n in numbers {
        if n > 0 then {
            total = total + n
            count = count + 1
        } else ()
    }

    if count > 0 then total / count else 0
}
```

Use `mut` when it makes your intent clearer. The data structures themselves remain immutable—only local variables can be reassigned.

### Thread-Safe Globals with mvars

Need shared state across processes? Use `mvar` for thread-safe global variables:

```nos
# Declare a thread-safe global counter
mvar requestCount: Int = 0

# Safe to update from any process
handleRequest(req) = {
    requestCount = requestCount + 1  # Atomic update
    println("Request #" ++ show(requestCount))
    respondOk(req)
}

# Multiple processes can safely increment
main() = {
    server = Server.bind(8080)
    # Each request spawns a process, all safely update the counter
    Server.accept_loop(server, req => spawn { handleRequest(req) })
}
```

`mvar` uses locks internally, so updates are atomic. Use them sparingly—message passing is usually better—but they're there when you need simple shared counters or caches.

### Choose What Reads Best

Here's the same task both ways. Pick the style that's clearest for your use case:

```nos
# Functional: great for transformations
processItems(items) =
    items
        .filter(item => item.active)
        .map(item => item.name.toUpper())
        .sort()

# Imperative: better when you need early exit or complex conditions
processItemsImperative(items) = {
    mut result = []

    for item in items {
        if !item.active then continue else ()

        # Complex condition that would be awkward in filter
        if item.score > 100 && item.age < 18 then {
            result = result ++ [item.name.toUpper()]
        } else ()

        # Early exit when we have enough
        if result.length() >= 10 then break else ()
    }

    result.sort()
}
```

Both are valid Nostos. Use whichever makes the code easier to understand. **Your time matters more than purity.**

---

## Philosophy

Nostos believes that:

- **Concurrency should be simple.** Message passing between lightweight processes beats shared memory and locks.
- **I/O should never block.** Every operation yields, letting thousands of processes share a thread pool.
- **Types should help, not hinder.** Inference handles the common case; annotations clarify intent.
- **Development should be interactive.** The REPL and live reload keep you in flow.
- **Batteries should be included.** HTTP, PostgreSQL, JSON, WebSockets—ready when you are.

---

*Welcome home.*
