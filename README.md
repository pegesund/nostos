# Nostos

> *In Greek, Nostos (νόστος) means a hero's homecoming or return from a long journey.*

After wandering through callback hell, fighting with async/await, and battling race conditions—Nostos welcomes you home to a place where concurrent code is simple, readable, and just works.

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

## Batteries Included

### PostgreSQL

Query your database with minimal friction:

```nos
import postgres

main() = {
    db = postgres.connect("localhost", "mydb", "user", "pass")

    users = db.query("SELECT * FROM users WHERE active = $1", [true])

    users.forEach(u => println(u.name))
}
```

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

### JSON

Parse, generate, and transform:

```nos
data = json.parse('{"name": "Alice", "scores": [95, 87, 92]}')
name = data.get("name")  # "Alice"

output = json.stringify(#{ "status": "ok", "count": 42 })
```

### HTML Templating

Type-safe templates that compose:

```nos
page(title, content) = html([
    head([ title(title) ]),
    body([
        header([ h1(title) ]),
        main(content),
        footer([ p("© 2024") ])
    ])
])
```

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

- **Immutable persistent data structures** — Lists, maps, and sets are immutable by default, using structural sharing for efficient updates. No defensive copying, no surprise mutations. Functional programming without the performance penalty.

- **Hindley-Milner type inference** — Full type inference means you rarely write type annotations, but the compiler catches errors at compile time. Generics, traits, and higher-order functions all work seamlessly.

- **Register-based VM** — Unlike stack-based VMs, our register-based design reduces instruction count and enables better JIT optimization. Hot paths are compiled to native code via [Cranelift](https://cranelift.dev/).

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

## Philosophy

Nostos believes that:

- **Concurrency should be simple.** Message passing between lightweight processes beats shared memory and locks.
- **I/O should never block.** Every operation yields, letting thousands of processes share a thread pool.
- **Types should help, not hinder.** Inference handles the common case; annotations clarify intent.
- **Development should be interactive.** The REPL and live reload keep you in flow.
- **Batteries should be included.** HTTP, PostgreSQL, JSON, WebSockets—ready when you are.

---

*Welcome home.*
