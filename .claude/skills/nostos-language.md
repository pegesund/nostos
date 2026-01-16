# Nostos Language Reference

A functional, concurrent programming language with Erlang-style actors and Hindley-Milner type inference.

## 1. Syntax Quick Reference

### Comments
```nostos
# This is a comment (NOT //)
```

### Functions
```nostos
# Simple function
add(a, b) = a + b

# With type annotations
add(a: Int, b: Int) -> Int = a + b

# With block body
process(x) = {
    y = x * 2
    y + 1
}

# Pattern matching in function args
myHead([h | _]) = h
myHead([]) = 0

# Multiple clauses (like Erlang)
fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)
```

### Lambdas
```nostos
# Single parameter
x => x * 2

# Multiple parameters (NOT curried)
(a, b) => a + b

# No parameters
() => 42

# Capture variables
multiplier = 10
f = x => x * multiplier
```

### Operators
| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Arithmetic |
| `%` | Modulo |
| `==`, `!=` | Equality |
| `<`, `>`, `<=`, `>=` | Comparison |
| `&&`, `\|\|`, `!` | Logical |
| `++` | String concatenation |
| `\|` | List cons: `[h \| t]` |
| `<-` | Send message: `pid <- msg` |

### Control Flow
```nostos
# If expression (always returns value)
if x > 0 then "positive" else "non-positive"

# Multi-line if
if x > 0 then {
    println("positive")
    1
} else {
    println("negative")
    -1
}

# Match expression
match value {
    0 -> "zero"
    1 -> "one"
    n -> "other: " ++ show(n)
}

# Pattern match on variants
match option {
    Some(x) -> x
    None -> default
}
```

### Blocks
```nostos
# Blocks return last expression
result = {
    a = 1
    b = 2
    a + b  # returns 3
}

# Semicolons for side effects
{
    println("hello");
    println("world");
    42
}
```

## 2. Type System

### Primitive Types
| Type | Description | Examples |
|------|-------------|----------|
| `Int` | 64-bit integer | `42`, `-17`, `1_000_000` |
| `Float` | 64-bit float | `3.14`, `-0.5` |
| `Bool` | Boolean | `true`, `false` |
| `String` | UTF-8 string | `"hello"` |
| `Char` | Single character | `'a'`, `'\n'` |
| `Unit` | No value | `()` |

### Sized Numerics
`Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `BigInt`

### Collections
```nostos
# List
[1, 2, 3]           # Type: [Int]
["a", "b"]          # Type: [String]

# Tuple
(1, "hello", true)  # Type: (Int, String, Bool)

# Map
{| "a" => 1, "b" => 2 |}  # Type: Map[String, Int]

# Set
{< 1, 2, 3 >}       # Type: Set[Int]
```

### Function Types
```nostos
# Function taking Int and String, returning Bool
(Int, String) -> Bool

# IMPORTANT: NOT curried! Use tuples for multi-param callbacks
[a] -> b -> ((b, a) -> b) -> b   # fold signature
```

### Type Annotations
```nostos
# On bindings
x: Int = 42

# On function parameters
greet(name: String) -> String = "Hello, " ++ name

# On function return (optional - inferred)
add(a: Int, b: Int) -> Int = a + b
```

## 3. Defining Types

### Records
```nostos
# Definition
type Point = { x: Int, y: Int }

# Construction
p = Point(10, 20)
p = Point { x: 10, y: 20 }  # Named fields

# Field access
p.x  # 10

# Update (returns new record)
p2 = p { x: 15 }
```

### Variants (Enums)
```nostos
# Definition
type Option[T] = Some(T) | None

type Result[T, E] = Ok(T) | Err(E)

type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])

# Construction
opt = Some(42)
err = Err("failed")

# Pattern matching
match opt {
    Some(x) -> x
    None -> 0
}
```

### Generic Records
```nostos
type Pair[A, B] = { first: A, second: B }

p = Pair(1, "hello")  # Pair[Int, String]
```

### Traits
```nostos
# Definition
trait Drawable
    draw(self) -> String
end

# Implementation
type Circle = { radius: Int }

Circle: Drawable
    draw(self) = "Circle(" ++ show(self.radius) ++ ")"
end

# Usage
c = Circle(5)
c.draw()  # "Circle(5)"
```

### Built-in Traits
| Trait | Methods | Description |
|-------|---------|-------------|
| `Eq` | `eq(self, other)` | Equality |
| `Ord` | `lt`, `lte`, `gt`, `gte` | Ordering |
| `Num` | `add`, `sub`, `mul`, `div` | Arithmetic |
| `Show` | `show(self)` | String conversion |
| `Hash` | `hash(self)` | Hash code |
| `Copy` | - | Deep copy |

## 4. Module System

### File = Module
Each `.nos` file is a module. File name = module name.

### Exports
```nostos
# math.nos
pub add(a, b) = a + b      # Public function
pub type Point = { x: Int, y: Int }  # Public type

helper(x) = x * 2          # Private (no pub)
```

### Imports
```nostos
# Import module
use math

# Use qualified
math.add(1, 2)

# Functions from stdlib are auto-imported
```

## 5. Standard Library Highlights

### List Operations
| Function | Signature | Example |
|----------|-----------|---------|
| `map` | `[a] -> (a -> b) -> [b]` | `[1,2,3].map(x => x * 2)` |
| `filter` | `[a] -> (a -> Bool) -> [a]` | `[1,2,3].filter(x => x > 1)` |
| `fold` | `[a] -> b -> ((b,a) -> b) -> b` | `[1,2,3].fold(0, (acc,x) => acc + x)` |
| `flatMap` | `[a] -> (a -> [b]) -> [b]` | `[1,2].flatMap(x => [x, x])` |
| `head` | `[a] -> a` | `[1,2,3].head()` |
| `tail` | `[a] -> [a]` | `[1,2,3].tail()` |
| `take` | `[a] -> Int -> [a]` | `[1,2,3].take(2)` |
| `drop` | `[a] -> Int -> [a]` | `[1,2,3].drop(1)` |
| `find` | `[a] -> (a -> Bool) -> Option[a]` | `[1,2,3].find(x => x > 1)` |
| `any` | `[a] -> (a -> Bool) -> Bool` | `[1,2,3].any(x => x > 2)` |
| `all` | `[a] -> (a -> Bool) -> Bool` | `[1,2,3].all(x => x > 0)` |
| `zip` | `[a] -> [b] -> [(a,b)]` | `[1,2].zip(["a","b"])` |
| `sort` | `[a] -> [a]` | `[3,1,2].sort()` |
| `reverse` | `[a] -> [a]` | `[1,2,3].reverse()` |
| `length` | `[a] -> Int` | `[1,2,3].length()` |
| `isEmpty` | `[a] -> Bool` | `[].isEmpty()` |
| `concat` | `[a] -> [a] -> [a]` | `[1,2].concat([3,4])` |
| `unique` | `[a] -> [a]` | `[1,1,2].unique()` |
| `sum` | `[a] -> a` | `[1,2,3].sum()` |
| `range` | `Int -> Int -> [Int]` | `range(1, 5)` |

### String Operations
| Function | Example |
|----------|---------|
| `String.length(s)` | `"hello".length()` → `5` |
| `String.toUpper(s)` | `"hello".toUpper()` → `"HELLO"` |
| `String.toLower(s)` | `"HELLO".toLower()` → `"hello"` |
| `String.trim(s)` | `"  hi  ".trim()` → `"hi"` |
| `String.split(s, d)` | `"a,b,c".split(",")` → `["a","b","c"]` |
| `String.contains(s, sub)` | `"hello".contains("ell")` → `true` |
| `String.startsWith(s, p)` | `"hello".startsWith("he")` → `true` |
| `String.replace(s, f, t)` | `"hello".replace("l", "x")` → `"hexlo"` |
| `String.substring(s, i, j)` | `"hello".substring(1, 4)` → `"ell"` |

### Map Operations
| Function | Example |
|----------|---------|
| `Map.insert(m, k, v)` | `m.insert("key", 42)` |
| `Map.get(m, k)` | `m.get("key")` |
| `Map.contains(m, k)` | `m.contains("key")` |
| `Map.remove(m, k)` | `m.remove("key")` |
| `Map.keys(m)` | `m.keys()` |
| `Map.values(m)` | `m.values()` |
| `Map.size(m)` | `m.size()` |
| `Map.fromList(pairs)` | `Map.fromList([("a", 1)])` |

### Set Operations
| Function | Example |
|----------|---------|
| `Set.insert(s, x)` | `s.insert(42)` |
| `Set.contains(s, x)` | `s.contains(42)` |
| `Set.remove(s, x)` | `s.remove(42)` |
| `Set.union(a, b)` | `a.union(b)` |
| `Set.intersection(a, b)` | `a.intersection(b)` |
| `Set.fromList(xs)` | `Set.fromList([1,2,3])` |

## 6. Concurrency

### Spawn Process
```nostos
# Spawn returns Pid
pid = spawn { some_function() }

# Spawn with closure
pid = spawn {
    x = compute_something()
    parent <- x
}
```

### Message Passing
```nostos
# Send message
pid <- "hello"
send(pid, "hello")  # Equivalent

# Receive with pattern matching
msg = receive {
    "ping" -> "pong"
    n: Int -> n * 2
    other -> other
}
```

### Self and Parent Communication
```nostos
main() = {
    me = self()
    spawn { me <- 42 }
    result = receive { x -> x }
    println(result)  # 42
}
```

### MVars (Shared Mutable State)
```nostos
# Declare mvar at module level
mvar counter: Int = 0

# Atomically update
increment() = {
    counter = counter + 1
    counter
}

# Read
getCount() = counter
```

## 7. Error Handling

### Exceptions
```nostos
# Throw
throw("Something went wrong")

# Try/catch
result = try {
    risky_operation()
} catch {
    e -> "Error: " ++ e
}
```

### Result Type
```nostos
# From stdlib
type Result[T, E] = Ok(T) | Err(E)

# Usage
safeDivide(a, b) =
    if b == 0 then Err("division by zero")
    else Ok(a / b)

match safeDivide(10, 2) {
    Ok(n) -> println(n)
    Err(e) -> println("Error: " ++ e)
}
```

### Option Type
```nostos
# From stdlib
type Option[T] = Some(T) | None

# Usage
findFirst(list, pred) =
    match list {
        [] -> None
        [x | xs] -> if pred(x) then Some(x) else findFirst(xs, pred)
    }
```

## 8. Pattern Matching

### List Patterns
```nostos
# Empty list
[]

# Head and tail
[h | t]

# Specific elements
[a, b, c]

# Fixed head, rest
[first, second | rest]
```

### Tuple Patterns
```nostos
(a, b) = (1, 2)
(x, _, z) = (1, 2, 3)  # Ignore middle
```

### Record Patterns
```nostos
type Person = { name: String, age: Int }

greet({ name: n, age: a }) = "Hello " ++ n ++ ", age " ++ show(a)
greet(Person { name: n }) = "Hello " ++ n  # Partial match
```

### Guard Clauses
```nostos
classify(n) | n < 0 = "negative"
classify(n) | n == 0 = "zero"
classify(n) = "positive"
```

## 9. I/O and Side Effects

### Printing
```nostos
println("Hello")        # With newline
print("No newline")     # Without newline
eprintln("Error!")      # To stderr
show(42)                # Convert to string: "42"
```

### Files
```nostos
content = File.readAll("/path/to/file")
File.writeAll("/path/to/file", "content")
File.append("/path/to/file", "more")
exists = File.exists("/path/to/file")
```

### HTTP
```nostos
response = Http.get("https://example.com")
# response has: status, headers, body
```

## 10. Common Idioms

### Pipeline Style
```nostos
result = data
    .filter(x => x > 0)
    .map(x => x * 2)
    .fold(0, (acc, x) => acc + x)
```

### Recursion (No Loops)
```nostos
# Use recursion instead of for/while
sumList([]) = 0
sumList([x | xs]) = x + sumList(xs)

# Or use fold
sumList(list) = list.fold(0, (acc, x) => acc + x)
```

### Builder Pattern with Named Args
```nostos
type Config = { host: String, port: Int, debug: Bool }

defaultConfig() = Config { host: "localhost", port: 8080, debug: false }

# Update specific fields
myConfig = defaultConfig() { port: 3000, debug: true }
```

## 11. Common Mistakes to Avoid

| Wrong | Right | Why |
|-------|-------|-----|
| `// comment` | `# comment` | Nostos uses `#` for comments |
| `(b -> a -> b)` | `((b, a) -> b)` | No currying - use tuple params |
| `get(list, i)` | Use `getValue` or `nth` | `get` shadows stdlib |
| `set(list, i, v)` | Use `setValue` | `set` shadows stdlib |
| `reverse(list)` | Use `reverseList` | `reverse` shadows stdlib |
| `"a" + "b"` | `"a" ++ "b"` | Use `++` for strings |
| `for x in list` | `list.map(...)` | No loops - use map/fold/recursion |

## 12. Testing

### Test File Format
```nostos
# expect: 42
main() = 21 * 2
```

### Error Tests
```nostos
# expect_error: type mismatch
main() = 5 + "hello"
```

### Running Tests
```bash
# Single file
./target/release/nostos path/to/test.nos

# All tests
cd tests && ./runall.sh
```

## 13. Builtin Modules Reference

### File Operations
`File.readAll`, `File.writeAll`, `File.append`, `File.exists`, `File.remove`, `File.copy`, `File.size`

### Directory Operations
`Dir.create`, `Dir.createAll`, `Dir.list`, `Dir.remove`, `Dir.removeAll`, `Dir.exists`

### Time
`Time.now`, `Time.format`, `Time.parse`, `Time.year`, `Time.month`, `Time.day`, `Time.hour`, `Time.minute`, `Time.second`

### Random
`Random.int(min, max)`, `Random.float()`, `Random.bool()`, `Random.choice(list)`, `Random.shuffle(list)`

### Environment
`Env.get(name)`, `Env.set(name, val)`, `Env.cwd()`, `Env.home()`, `Env.args()`, `Env.platform()`

### Path
`Path.join(a, b)`, `Path.dirname(p)`, `Path.basename(p)`, `Path.extension(p)`, `Path.normalize(p)`

### Regex
`Regex.matches(str, pattern)`, `Regex.find(str, pattern)`, `Regex.findAll(str, pattern)`, `Regex.replace(str, pattern, replacement)`

### JSON
`Json.parse(str)`, `Json.stringify(val)`

### Crypto
`Crypto.sha256(str)`, `Crypto.sha512(str)`, `Crypto.bcryptHash(pass, cost)`, `Crypto.bcryptVerify(pass, hash)`

### UUID
`Uuid.v4()`, `Uuid.isValid(str)`

### Process Introspection
`Process.all()`, `Process.alive(pid)`, `Process.info(pid)`, `Process.kill(pid)`

### PostgreSQL
`Pg.connect(url)`, `Pg.query(h, sql, params)`, `Pg.execute(h, sql, params)`, `Pg.transaction(h, fn)`, `Pg.close(h)`

### HTTP Server
`Server.bind(port)`, `Server.accept(h)`, `Server.respond(reqId, status, headers, body)`, `Server.close(h)`

### TCP
`Tcp.connect(host, port)`, `Tcp.listen(port)`, `Tcp.accept(listener)`, `Tcp.read(sock, n)`, `Tcp.write(sock, data)`, `Tcp.close(sock)`

### WebSocket
`WebSocket.isUpgrade(req)`, `WebSocket.accept(reqId)`, `WebSocket.send(ws, msg)`, `WebSocket.recv(ws)`, `WebSocket.close(ws)`
