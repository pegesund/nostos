# Nostos

> *Nostos (νόστος): Greek for "homecoming" — the return journey. The language you keep coming back to.*

**Nostos** is a minimal, fast, type-safe programming language designed for high-concurrency applications and live development. It combines the fault-tolerant actor model of Erlang with the expressiveness and type safety of modern functional languages, all powered by a high-performance register-based VM written in Rust.

## Key Features

*   **⚡ Lightweight Concurrency**: Spawns hundreds of thousands of isolated processes (actors) with minimal overhead. Uses message passing for safe communication.
*   **🧩 Pattern Matching**: Expressive pattern matching in function definitions, `match` expressions, and message reception.
*   **🛡️ Type Safety**: Strong, static structural typing with type inference. No nulls (uses `Option`), no exceptions (uses `Result` or supervision trees).
*   **🎭 Traits**: Flexible polymorphism through traits, allowing you to define shared behavior for different types.
*   **🚀 High Performance**: 
    *   Register-based VM with typed bytecode.
    *   **JIT Compilation** using [Cranelift](https://cranelift.readthedocs.io/) for hot paths.
    *   Typed arrays for efficient numeric computation.
*   **🔍 Introspection & Live Dev**: Full runtime introspection of types and functions. Hot code reloading and image saving (Smalltalk-style) planned.
*   **🧹 Garbage Collection**: Per-process garbage collection ensures short pauses and system responsiveness.

## Examples

### 1. Pattern Matching & Recursion

Nostos supports multiple function clauses and efficient tail recursion.

```nos
# Naive recursive fibonacci
fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)

# Tail-recursive with pattern matching
fib_fast(n) = fib_helper(n, 0, 1)

fib_helper(0, a, _) = a
fib_helper(n, a, b) = fib_helper(n - 1, b, a + b)
```

### 2. Massive Concurrency

Spawn lightweight processes and communicate via messages.

```nos
# Worker: receives a value and the parent pid, echoes it back
worker(parent, secret) = {
    parent <- secret
    ()
}

main() = {
    me = self()
    n = 100000
    
    # Spawn 100,000 processes
    for i = 0 to n {
        spawn { worker(me, 42) }
    }

    # Collect responses
    correct = 0
    for i = 0 to n {
        correct = receive
            42 -> correct + 1
            _  -> correct
        end
    }
    
    println("Received " ++ show(correct) ++ " responses!")
}
```

### 3. Traits & Polymorphism

Define shared behavior for different data types.

```nos
trait Shape
    area(self) -> Float
    name(self) -> String
end

type Circle = { radius: Float }
type Rectangle = { width: Float, height: Float }

Circle: Shape
    area(self) = 3.14159 * self.radius * self.radius
    name(self) = "Circle"
end

Rectangle: Shape
    area(self) = self.width * self.height
    name(self) = "Rectangle"
end
```

## Architecture

Nostos is built in **Rust** and organized as a workspace of crates:

*   `crates/compiler`: Lexer, parser, and AST lowering.
*   `crates/vm`: The register-based interpreter and runtime.
*   `crates/jit`: JIT compiler backend using Cranelift.
*   `crates/scheduler`: Process scheduler and concurrency primitives.
*   `crates/types`: Type checker and inference engine.

## Getting Started

To build and run Nostos, you need a standard Rust toolchain.

```bash
# Clone the repository
git clone https://github.com/yourusername/nostos.git
cd nostos

# Run an example
cargo run --release -- examples/fibonacci.nos

# Run the REPL
cargo run --release
```

## Status

Nostos is currently in **active development**. Syntax and features are subject to change.
