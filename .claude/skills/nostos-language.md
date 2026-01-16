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

---

## 14. Complete Example Programs

### Example 1: Fibonacci (Recursion & Tail Recursion)

```nostos
# Fibonacci Sequence
# Demonstrates recursion and pattern matching

# Naive recursive fibonacci - O(2^n) time
fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)

# Tail-recursive fibonacci - O(n) time
fib_fast(n) = fib_helper(n, 0, 1)

fib_helper(0, a, _) = a
fib_helper(n, a, b) = fib_helper(n - 1, b, a + b)

# Generate first n fibonacci numbers as a list
fibs(0) = []
fibs(n) = fibs_helper(n, [])

fibs_helper(0, acc) = acc
fibs_helper(n, acc) = fibs_helper(n - 1, [fib_fast(n - 1) | acc])

main() = {
    assert_eq(55, fib_fast(10))
    assert_eq([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], fibs(10))
    println("Fibonacci tests passed!")
    0
}
```

### Example 2: Ping-Pong (Message Passing)

```nostos
# Ping-Pong: Two processes communicating via messages

type Message = Ping(Pid, Int) | Pong(Int) | Done(Int) | Stop

# Ping process: sends Ping, waits for Pong, counts down
ping(pong_pid, 0, parent) = {
    parent <- Done(0)
    ()
}
ping(pong_pid, n, parent) = {
    pong_pid <- Ping(self(), n)
    receive {
        Pong(m) -> ping(pong_pid, m - 1, parent)
    }
}

# Pong process: waits for Ping, sends Pong back
pong() = receive {
    Ping(sender, n) -> {
        sender <- Pong(n)
        pong()
    }
    Stop -> ()
}

main() = {
    me = self()
    pong_pid = spawn { pong() }
    spawn { ping(pong_pid, 5, me) }

    result = receive {
        Done(n) -> {
            pong_pid <- Stop
            n
        }
    }

    assert_eq(0, result)
    println("Ping-pong test passed!")
    0
}
```

### Example 3: Counter Server (Stateful Actor)

```nostos
# Counter Server: OTP GenServer pattern with state via recursion

type Message = Inc(Pid) | Dec(Pid) | Get(Pid) | Stop | Ok(Int) | Value(Int)

# Counter loop maintains state as function argument
counter_loop(state) = receive {
    Inc(sender) -> {
        sender <- Ok(state + 1)
        counter_loop(state + 1)
    }
    Dec(sender) -> {
        sender <- Ok(state - 1)
        counter_loop(state - 1)
    }
    Get(sender) -> {
        sender <- Value(state)
        counter_loop(state)
    }
    Stop -> state
}

# Start a counter with initial value
start_counter(initial) = spawn { counter_loop(initial) }

# Client operations
increment(counter) = {
    counter <- Inc(self())
    receive { Ok(val) -> val }
}

decrement(counter) = {
    counter <- Dec(self())
    receive { Ok(val) -> val }
}

get_value(counter) = {
    counter <- Get(self())
    receive { Value(val) -> val }
}

stop(counter) = counter <- Stop

main() = {
    c = start_counter(0)
    assert_eq(0, get_value(c))
    assert_eq(1, increment(c))
    assert_eq(2, increment(c))
    assert_eq(3, increment(c))
    assert_eq(2, decrement(c))
    stop(c)
    println("Counter server tests passed!")
    0
}
```

### Example 4: Worker Pool (Distributed Work)

```nostos
# Worker Pool: Distribute tasks across multiple workers

type Message = Task(Pid, Int) | Result(Int) | Stop

# Worker process
worker(pool) = receive {
    Task(client, work) -> {
        result = work()
        client <- Result(result)
        worker(pool)
    }
    Stop -> ()
}

# Start n workers
start_workers(0, _) = []
start_workers(n, pool) = [spawn { worker(pool) } | start_workers(n - 1, pool)]

# Submit task to first worker
submit_task([w | _], client, work) = w <- Task(client, work)

# Work functions
fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)

compute_fib() = fib(10)

main() = {
    me = self()
    workers = start_workers(2, me)

    submit_task(workers, me, compute_fib)
    result = receive { Result(r) -> r }

    assert_eq(55, result)
    println("Worker pool tests passed!")
    0
}
```

### Example 5: Binary Tree (Recursive Data Types)

```nostos
# Binary Tree: Recursive algebraic data type

type Tree[T] = Leaf | Node(T, Tree[T], Tree[T])

# Calculate tree depth
depth(Leaf) = 0
depth(Node(_, left, right)) = 1 + maxVal(depth(left), depth(right))

maxVal(a, b) = if a > b then a else b

# Sum all values in tree
sum_tree(Leaf) = 0
sum_tree(Node(v, left, right)) = v + sum_tree(left) + sum_tree(right)

# Count nodes
countNodes(Leaf) = 0
countNodes(Node(_, left, right)) = 1 + countNodes(left) + countNodes(right)

# In-order traversal to list
inorder(Leaf) = []
inorder(Node(v, left, right)) = inorder(left).concat([v | inorder(right)])

# Map a function over tree
map_tree(_, Leaf) = Leaf
map_tree(f, Node(v, left, right)) = Node(f(v), map_tree(f, left), map_tree(f, right))

main() = {
    # Create:    4
    #           / \
    #          2   6
    #         / \ / \
    #        1  3 5  7
    tree = Node(4,
        Node(2, Node(1, Leaf, Leaf), Node(3, Leaf, Leaf)),
        Node(6, Node(5, Leaf, Leaf), Node(7, Leaf, Leaf))
    )

    assert_eq(28, sum_tree(tree))           # 1+2+3+4+5+6+7
    assert_eq(7, countNodes(tree))
    assert_eq(3, depth(tree))
    assert_eq([1, 2, 3, 4, 5, 6, 7], inorder(tree))

    doubled = map_tree(x => x * 2, tree)
    assert_eq(56, sum_tree(doubled))

    println("Binary tree tests passed!")
    0
}
```

### Example 6: List Operations (Functional Programming)

```nostos
# List Operations: Higher-order functions and method chaining

# Custom implementations (stdlib provides these too)
myMap([], _) = []
myMap([h | t], f) = [f(h) | myMap(t, f)]

myFilter([], _) = []
myFilter([h | t], p) = if p(h) then [h | myFilter(t, p)] else myFilter(t, p)

myFold([], acc, _) = acc
myFold([h | t], acc, f) = myFold(t, f(acc, h), f)

# Helper functions
listSum(lst) = lst.fold(0, (a, b) => a + b)
listProduct(lst) = lst.fold(1, (a, b) => a * b)
rev(lst) = lst.fold([], (acc, x) => [x | acc])

main() = {
    nums = [1, 2, 3, 4, 5]

    # Method chaining with stdlib
    doubled = nums.map(x => x * 2)
    assert_eq([2, 4, 6, 8, 10], doubled)

    evens = nums.filter(x => x % 2 == 0)
    assert_eq([2, 4], evens)

    # Chained operations
    result = nums.map(x => x * 2).filter(x => x > 4)
    assert_eq([6, 8, 10], result)

    assert_eq(15, listSum(nums))
    assert_eq(120, listProduct(nums))
    assert_eq([5, 4, 3, 2, 1], rev(nums))

    # Complex pipeline
    complexResult = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        .filter(x => x % 2 == 0)      # [2, 4, 6, 8, 10]
        .map(x => x * x)              # [4, 16, 36, 64, 100]
        .fold(0, (a, b) => a + b)     # 220
    assert_eq(220, complexResult)

    println("List operations tests passed!")
    0
}
```

### Example 7: Pattern Matching (Guards & ADTs)

```nostos
# Pattern Matching: Guards, variants, and list patterns

type Shape =
    | Circle(Float)
    | Rectangle(Float, Float)
    | Triangle(Float, Float, Float)

# Match on variant constructors
area(Circle(r)) = 3.14159 * r * r
area(Rectangle(w, h)) = w * h
area(Triangle(a, b, c)) = {
    s = (a + b + c) / 2.0
    s * (s - a) * (s - b) * (s - c)
}

describe(Circle(_)) = "circle"
describe(Rectangle(_, _)) = "rectangle"
describe(Triangle(_, _, _)) = "triangle"

# Pattern matching with guards
classify_number(n) when n > 0 = "positive"
classify_number(n) when n < 0 = "negative"
classify_number(0) = "zero"

# Match on tuple
compare(a, b) = match (a > b, a < b) {
    (true, _) -> "greater"
    (_, true) -> "less"
    _ -> "equal"
}

# List patterns
describe_list([]) = "empty"
describe_list([_]) = "singleton"
describe_list([_, _]) = "pair"
describe_list([_, _ | _]) = "many"

main() = {
    circle = Circle(5.0)
    rect = Rectangle(3.0, 4.0)

    assert_eq(12.0, area(rect))
    assert_eq("circle", describe(circle))

    assert_eq("positive", classify_number(42))
    assert_eq("negative", classify_number(-5))
    assert_eq("zero", classify_number(0))

    assert_eq("greater", compare(10, 5))
    assert_eq("less", compare(3, 8))
    assert_eq("equal", compare(5, 5))

    assert_eq("empty", describe_list([]))
    assert_eq("singleton", describe_list([1]))
    assert_eq("pair", describe_list([1, 2]))
    assert_eq("many", describe_list([1, 2, 3, 4]))

    println("Pattern matching tests passed!")
    0
}
```

### Example 8: Traits (Polymorphism)

```nostos
# Traits: Define behavior for multiple types

type Circle = { radius: Float }
type Rectangle = { width: Float, height: Float }

# Define a trait
trait Shape
    area(self) -> Float
    perimeter(self) -> Float
    name(self) -> String
end

# Implement for Circle
Circle: Shape
    area(self) = 3.14159 * self.radius * self.radius
    perimeter(self) = 2.0 * 3.14159 * self.radius
    name(self) = "Circle(r=" ++ show(self.radius) ++ ")"
end

# Implement for Rectangle
Rectangle: Shape
    area(self) = self.width * self.height
    perimeter(self) = 2.0 * (self.width + self.height)
    name(self) = "Rectangle(" ++ show(self.width) ++ "x" ++ show(self.height) ++ ")"
end

main() = {
    circle = Circle { radius: 5.0 }
    rect = Rectangle { width: 4.0, height: 6.0 }

    # Call trait methods
    assert_eq(78.53975, circle.area())
    assert_eq(31.4159, circle.perimeter())

    assert_eq(24.0, rect.area())
    assert_eq(20.0, rect.perimeter())

    println(circle.name())  # Circle(r=5.0)
    println(rect.name())    # Rectangle(4.0x6.0)

    total = circle.area() + rect.area()
    assert_eq(102.53975, total)

    println("Traits tests passed!")
    0
}
```

### Example 9: HTTP Server (Web)

```nostos
# Simple HTTP Server

handleRequest(server, req, main_pid) = {
    path = req.path

    if path == "/shutdown" then {
        main_pid <- ("shutdown", ())
        Server.respond(req.id, 200, [("Content-Type", "text/plain")], "Shutting down...")
    }
    else {
        Server.respond(req.id, 200, [("Content-Type", "text/plain")], "Hello, World!")
    }
}

workerLoop(server, main_pid) = {
    result = try {
        req = Server.accept(server)
        handleRequest(server, req, main_pid)
        workerLoop(server, main_pid)
    } catch { e -> () }
    result
}

spawnWorkers(server, main_pid, 0) = ()
spawnWorkers(server, main_pid, n) = {
    spawn { workerLoop(server, main_pid) }
    spawnWorkers(server, main_pid, n - 1)
}

main() = {
    main_pid = self()

    result = try {
        server = Server.bind(8080)
        println("Server running on http://localhost:8080")
        println("Hit /shutdown to stop")

        spawnWorkers(server, main_pid, 8)

        receive {
            ("shutdown", _) -> {
                Server.close(server)
                println("Server shutdown complete.")
            }
        }
    } catch { e ->
        println("Error: " ++ e)
    }
    0
}
```

---

## 15. UFCS (Uniform Function Call Syntax)

Nostos uses **receiver-first UFCS** - method calls are transformed to function calls with the receiver as the first argument.

### How It Works

```nostos
# These are equivalent:
[1, 2, 3].map(x => x * 2)
map([1, 2, 3], x => x * 2)

# Method chaining works via UFCS
[1, 2, 3].filter(x => x > 1).map(x => x * 2)
# Equivalent to:
map(filter([1, 2, 3], x => x > 1), x => x * 2)
```

### Implications

1. **Any function can be called as a method** if its first parameter matches the receiver type
2. **Stdlib functions** are designed with receiver-first for chaining: `[a] -> (a -> b) -> [b]`
3. **Custom functions** work the same way:

```nostos
# Define a function
double(x: Int) = x * 2

# Call as method
result = 5.double()  # 10

# With multiple params
add(a: Int, b: Int) = a + b
result = 5.add(3)    # 8 - same as add(5, 3)
```

### UFCS in BUILTINS Signatures

BUILTIN signatures show receiver-first order:

| Signature | Method Call | Function Call |
|-----------|-------------|---------------|
| `[a] -> (a -> b) -> [b]` | `list.map(f)` | `map(list, f)` |
| `[a] -> b -> ((b, a) -> b) -> b` | `list.fold(init, f)` | `fold(list, init, f)` |
| `String -> String -> Bool` | `s.contains(sub)` | `String.contains(s, sub)` |

---

## 16. Common Error Messages

### Type Mismatch
```
Error: type mismatch: expected `Int`, found `String`
```
**Cause:** Using incompatible types together (e.g., `5 + "hello"`)
**Fix:** Use correct types or convert: `show(5) ++ "hello"` for string concat

### Unknown Variable
```
Error: unknown variable `unknownVar`
```
**Cause:** Using a variable that hasn't been defined
**Fix:** Define the variable first, or check spelling

### Unknown Method
```
Error: no method `unknownMethod` found for type `List[Int]`
```
**Cause:** Calling a method that doesn't exist for the type
**Fix:** Check available methods for the type, verify spelling

### Wrong Arity
```
Error: function `add` expects 2 arguments, but 1 was provided
```
**Cause:** Calling a function with wrong number of arguments
**Fix:** Provide all required arguments

### Immutable Reassignment
```
Error: cannot reassign immutable variable 'x'; use 'var' to declare a mutable variable
```
**Cause:** Trying to reassign a value bound with `=`
**Fix:** Use `var x = 5` then `x = 10`, or use different names

### Non-Exhaustive Match (Runtime)
```
Runtime error: No clause matched for function 'describe/_'
```
**Cause:** Pattern match doesn't cover all cases
**Fix:** Add missing cases or a wildcard `_ -> ...`

### String Does Not Implement Num
```
Error: String does not implement Num
```
**Cause:** Using arithmetic operators on strings
**Fix:** Use `++` for concatenation, or convert to numbers first

### Index Out of Bounds (Runtime)
```
Runtime error: Index out of bounds
```
**Cause:** Accessing list/array at invalid index
**Fix:** Check bounds before access, use `find` for safe lookup

---

## 17. Project Structure

### Directory Layout
```
nostos/
├── crates/                 # Rust source code
│   ├── cli/               # TUI, REPL, editor
│   ├── compiler/          # Parser, compiler, BUILTINS
│   ├── jit/               # JIT compilation
│   ├── lsp/               # Language Server Protocol
│   ├── repl/              # REPL engine, compile checking
│   ├── source/            # Source file management
│   ├── syntax/            # Lexer, parser, AST
│   ├── types/             # Type inference, checking
│   └── vm/                # Virtual machine, GC, scheduler
├── stdlib/                 # Standard library (Nostos code)
├── examples/               # Example programs
├── tests/                  # Test files (.nos)
└── target/release/         # Built binaries
```

### Key Crates

| Crate | Purpose | Key Files |
|-------|---------|-----------|
| `compiler` | Compiles AST to bytecode | `compile.rs` (BUILTINS defined here) |
| `types` | HM type inference | `infer.rs` (type inference), `check.rs` |
| `syntax` | Parsing | `lexer.rs`, `parser.rs`, `ast.rs` |
| `vm` | Execution | `async_vm.rs`, `gc.rs`, `scheduler.rs` |
| `repl` | Interactive mode | `engine.rs` (compile checking for LSP) |
| `lsp` | VS Code support | `server.rs` |

### Builtin vs Stdlib

**Builtins** (in `crates/compiler/src/compile.rs`):
- Implemented in Rust
- Low-level operations: `+`, `-`, `spawn`, `receive`
- I/O: `File.*`, `Http.*`, `Server.*`, `Pg.*`
- Collections: `map`, `filter`, `fold` (optimized)
- Type operations: `show`, `typeOf`, `reflect`

**Stdlib** (in `stdlib/*.nos`):
- Implemented in Nostos
- Higher-level abstractions
- Can be modified without recompiling

### Stdlib Files

| File | Contents |
|------|----------|
| `list.nos` | Option, Result, list operations (many also as builtins) |
| `json.nos` | JSON parser, Json type, serialization |
| `html.nos` | HTML DSL for building HTML |
| `server.nos` | HTTP server helpers (getParam, respondText, etc.) |
| `validation.nos` | Form validation framework |
| `pool.nos` | PostgreSQL connection pooling |
| `traits.nos` | Built-in trait definitions (Eq, Ord, Num, Show, Hash, Copy) |
| `io.nos` | I/O handle types (FileHandle, ServerHandle, etc.) |
| `url.nos` | URL parsing utilities |
| `logging.nos` | Logging framework |
| `ws.nos` | WebSocket helpers |
| `rhtml.nos` | Reactive HTML components |
| `rweb.nos` | Reactive web framework |

---

## 18. Type Inference Edge Cases

Nostos uses Hindley-Milner type inference. Most types are inferred automatically, but some cases need help.

### What HM Infers Well

```nostos
# Literals - type is obvious
x = 42          # Int
s = "hello"     # String
b = true        # Bool

# Function calls - types flow from arguments
result = [1, 2, 3].map(x => x * 2)  # [Int]

# Generic functions - instantiated at call site
identity(x) = x
n = identity(42)      # Int
s = identity("hi")    # String

# Variant constructors
opt = Some(42)        # Option[Int] inferred from 42
```

### When Annotations Help

**Empty collections:**
```nostos
# Type is ambiguous without context
xs: [Int] = []
m: Map[String, Int] = {| |}
```

**Numeric literals in generic contexts:**
```nostos
# 0 could be Int, Float, etc.
# Usually fine, but annotation can clarify
initial: Float = 0.0
```

**Complex function signatures:**
```nostos
# For documentation and error messages
transform(items: [a], f: a -> b) -> [b] = items.map(f)
```

### What Works Without Annotations

```nostos
# Type flows through the expression
result = [1, 2, 3]
    .filter(x => x > 1)     # x inferred as Int
    .map(x => x * 2)        # still Int
    .fold(0, (a, b) => a + b)  # accumulator Int

# Generic functions just work
first([x | _]) = x
head = first([1, 2, 3])  # Int

# Variants with data
data = Some(42)
match data {
    Some(n) -> n * 2     # n is Int
    None -> 0
}
```

### Type Errors Caught at Compile Time

```nostos
# Mismatched types in operations
bad = 5 + "hello"           # Error: expected Int, found String

# Wrong callback signature
bad = [1,2,3].fold(0, (acc, x) => acc ++ show(x))
# Error: fold accumulator must match return type

# Incompatible branch types
bad = if true then 42 else "hello"  # Error: branches must match
```

### Not Caught Until Runtime

```nostos
# Non-exhaustive patterns (warning but compiles)
describe(Red) = "red"
describe(Green) = "green"
# Blue case missing - runtime error if called with Blue

# Division by zero
x = 10 / 0  # Runtime error

# Index out of bounds
x = [1, 2, 3].nth(10)  # Runtime error
```

---

## 19. Data Processing Pipeline Example

A complete example showing records, variants, traits, and pipelines together:

```nostos
# Data Processing Pipeline
# Demonstrates: records, variants, traits, pipelines, error handling

# --- Types ---

type User = {
    id: Int,
    name: String,
    email: String,
    age: Int,
    active: Bool
}

type ProcessingError =
    | ValidationError(String)
    | TransformError(String)

type ProcessResult = Ok(User) | Err(ProcessingError)

# --- Validation Trait ---

trait Validate
    validate(self) -> ProcessResult
end

User: Validate
    validate(self) = {
        if self.name.isEmpty() then
            Err(ValidationError("Name cannot be empty"))
        else if self.age < 0 then
            Err(ValidationError("Age cannot be negative"))
        else if !self.email.contains("@") then
            Err(ValidationError("Invalid email format"))
        else
            Ok(self)
    }
end

# --- Processing Functions ---

# Parse raw data into User
parseUser(data: (Int, String, String, Int, Bool)) -> User = {
    (id, name, email, age, active) = data
    User { id: id, name: name, email: email, age: age, active: active }
}

# Transform: normalize email to lowercase
normalizeEmail(user: User) -> User =
    user { email: user.email.toLower() }

# Transform: anonymize for export
anonymize(user: User) -> User =
    user { email: "***@***", name: user.name.take(1) ++ "***" }

# Filter: only active adults
isActiveAdult(user: User) -> Bool =
    user.active && user.age >= 18

# Summarize: extract key info
summarize(user: User) -> String =
    user.name ++ " (" ++ show(user.age) ++ ")"

# --- Pipeline ---

processUsers(rawData: [(Int, String, String, Int, Bool)]) -> [String] = {
    rawData
        .map(parseUser)                    # Parse raw tuples to Users
        .map(normalizeEmail)               # Normalize emails
        .filter(isActiveAdult)             # Keep active adults only
        .filter(u => u.validate().isOk())  # Keep only valid users
        .map(summarize)                    # Extract summaries
}

# Helper for Result
isOk(Ok(_)) = true
isOk(Err(_)) = false

main() = {
    # Sample data: (id, name, email, age, active)
    rawData = [
        (1, "Alice", "ALICE@example.com", 30, true),
        (2, "Bob", "bob@test.com", 17, true),      # Too young
        (3, "Charlie", "charlie@work.com", 25, false),  # Not active
        (4, "", "empty@test.com", 40, true),       # Empty name - invalid
        (5, "Diana", "DIANA@CORP.COM", 28, true),
        (6, "Eve", "invalid-email", 35, true)      # Bad email - invalid
    ]

    results = processUsers(rawData)

    # Should only have Alice and Diana
    assert_eq(2, results.length())
    assert_eq("Alice (30)", results.nth(0))
    assert_eq("Diana (28)", results.nth(1))

    println("Results: " ++ show(results))
    println("Data processing pipeline tests passed!")
    0
}
```
