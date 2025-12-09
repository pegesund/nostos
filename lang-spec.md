# Nostos

A minimal, fast, type-safe programming language with full introspection and live development capabilities.

*Nostos (νόστος): Greek for "homecoming" — the return journey. The language you keep coming back to.*

## Design Goals

- **Minimal syntax** — Clean, readable, low ceremony
- **Type-safe** — Structural typing with inference, simple generics
- **Fast** — Register-based VM with typed bytecode, designed for JIT
- **Concurrent** — Lightweight processes, message passing, fault tolerance
- **Introspective** — Full reflection on types, functions, traits at runtime
- **Live** — Modify code at runtime, save/load images (Smalltalk-style)
- **Functional-first** — Immutable by default, pattern matching, first-class functions
- **Object feel** — Dot syntax via UFCS, no classes or inheritance

## Implementation

- **Language**: Rust
- **VM**: Custom register-based bytecode interpreter
- **JIT**: Cranelift (optional, for hot paths)
- **GC**: Custom precise GC (type-aware) or Boehm initially

---

## Syntax Specification

### Algebraic Data Types

Nostos has full support for algebraic data types: sum types (variants), product types (records), newtypes, and type aliases.

#### Sum Types (Variants)

```
# Unit variants (no data)
type Direction = North | South | East | West

# Single-field variants
type Option[T] = None | Some(T)

# Multi-field variants (positional)
type List[T] = Nil | Cons(T, List[T])

# Named fields in variants
type Shape =
  | Circle{radius: Float}
  | Rectangle{width: Float, height: Float}
  | Point

# Mixed positional and named
type Expr =
  | Lit(Int)
  | Var(String)
  | BinOp{op: Op, left: Expr, right: Expr}
  | Call{func: String, args: List[Expr]}
```

#### Pattern Matching on Variants

```
# Unit variants
turn(North) = East
turn(East) = South
turn(South) = West
turn(West) = North

# Positional fields
match opt
  None -> "empty"
  Some(x) -> "got: " ++ x.show()

# Named fields
match shape
  Circle{radius: r} -> pi * r * r
  Rectangle{width: w, height: h} -> w * h
  Point -> 0.0

# Partial named (order doesn't matter)
match shape
  Rectangle{height: h} -> h    # ignores width

# Positional shorthand for named (must be in order)
match shape
  Circle(r) -> pi * r * r
  Rectangle(w, h) -> w * h
```

#### Product Types (Records)

```
# Immutable record
type Point = {x: Float, y: Float}

# Mutable record
var type Buffer = {data: Array[Int], len: Int}

# Records with many fields
type User = {
  id: Int,
  name: String,
  email: String,
  active: Bool
}

# Construction
p = Point(3.0, 4.0)
p = Point(x: 3.0, y: 4.0)      # named

# Field access
p.x
p.y

# Update (creates new record)
q = Point(p, x: 10.0)
```

#### Newtypes

Single-variant types for type safety:

```
# Define newtype
type UserId = UserId(Int)
type Email = Email(String)
type Meters = Meters(Float)
type Seconds = Seconds(Float)

# Construction
id = UserId(42)
dist = Meters(100.0)

# Type safety - can't mix
getUser(UserId(id)) = ...
getUser(42)                    # Error: expected UserId, got Int
getUser(UserId(42))            # Ok

# Unwrap in patterns
speed(Meters(d), Seconds(t)) = d / t

# Explicit unwrap
id.value                       # => 42
dist.value                     # => 100.0
```

#### Type Aliases

Transparent aliases for convenience:

```
# Alias for complex types
type StringResult = Result[String, Error]
type Handler = (Request) -> Response
type Predicate[T] = (T) -> Bool
type UserMap = Map[String, User]

# Aliases are interchangeable with original
f(r: StringResult) = ...
f(Ok("hello"))                 # works, Ok("hello") is StringResult

# Useful for documentation
type Milliseconds = Int
type Port = Int
sleep(duration: Milliseconds) = ...
connect(port: Port) = ...
```

#### Recursive Types

```
type List[T] = Nil | Cons(T, List[T])

type Tree[T] = 
  | Leaf(T)
  | Node{left: Tree[T], value: T, right: Tree[T]}

type Json =
  | JsonNull
  | JsonBool(Bool)
  | JsonNumber(Float)
  | JsonString(String)
  | JsonArray(List[Json])
  | JsonObject(Map[String, Json])
```

#### Generic Constraints

```
type Map[K: Eq, V] = ...

type SortedList[T: Ord] = Empty | Sorted(T, SortedList[T])

type Cacheable[T: Eq + Hash] = Cached{key: Int, value: T}
```

#### Empty Type

Type with no values (for functions that never return):

```
type Never = 

# Functions that never return
loop() -> Never = loop()
panic(msg: String) -> Never = ...

# Useful in variants
type Result[T, E] = Ok(T) | Err(E)

# A result that can't fail
safeOp() -> Result[Int, Never] = Ok(42)
```

### Types (Summary)

```
# Records (product types)
type Point = {x: Float, y: Float}
var type Buffer = {data: Array[Int], len: Int}

# Variants (sum types)
type Option[T] = None | Some(T)
type Shape = Circle{radius: Float} | Rectangle{w: Float, h: Float}

# Newtypes
type UserId = UserId(Int)

# Type aliases
type Handler = (Request) -> Response

# Generic constraints
type Map[K: Eq, V] = ...
```

### Primitives

```
# Integers
x = 42
y = -17
hex = 0xFF
binary = 0b1010
big = 1_000_000

# Floats
pi = 3.14159
small = 1.2e-10

# Booleans
yes = true
no = false

# Strings (double quotes, interpolation with ${})
name = "world"
greeting = "Hello, ${name}!"
multiline = "
  This is a
  multi-line string
"

# Characters
c = 'a'
newline = '\n'

# Unit (like void, but a value)
nothing = ()
```

### Comments

```
# Single line comment

#* 
  Multi-line
  comment 
*#

## Documentation comment (for introspection)
## Computes the fibonacci number
fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)
```

### Collections

```
# Lists (linked, immutable)
items = [1, 2, 3, 4, 5]
empty = []
[head | tail] = items          # destructure

# Arrays (indexed, mutable)
arr = Array.new(10, 0)         # size 10, filled with 0
arr[0] = 42
x = arr[5]

# Maps (immutable by default)
ages = %{"alice": 30, "bob": 25}
ages.get("alice")              # => Some(30)
ages.get("unknown")            # => None
updated = ages.put("carol", 28)

# Or use records for known keys (preferred, typed)
type Config = {host: String, port: Int}
config = Config("localhost", 8080)

# Sets
seen = #{1, 2, 3}
seen.contains?(2)              # => true
more = seen.add(4)

# Tuples (fixed size, mixed types)
pair = (1, "hello")
triple = (true, 3.14, "yes")
(a, b) = pair                  # destructure
```

### Nil and Option

No null/nil in Nostos. Use Option for optional values:

```
# Option type
type Option[T] = None | Some(T)

find(list, pred) -> Option[T] = ...

# Must handle both cases
match find(users, isAdmin)
  Some(admin) -> greet(admin)
  None -> println("no admin found")

# Helpers
maybe.map(x => x * 2)          # None if None, Some(result) if Some
maybe.flatMap(x => lookup(x))
maybe.getOr(default)
maybe.get()                    # panics if None
```

### Error Handling

Use Result for recoverable errors, panic for bugs, try/catch at boundaries:

```
# Result type for expected errors
type Result[T, E] = Ok(T) | Err(E)

readFile(path) -> Result[String, IoError] = ...

# Pattern match
match readFile("data.txt")
  Ok(content) -> process(content)
  Err(e) -> println("Error: " ++ e.show())

# Chain operations
readFile("config.json")
  .map(parse)
  .flatMap(validate)
  .mapErr(e => ConfigError(e))

# Propagate with ?
loadConfig(path) -> Result[Config, Error] = {
  content = readFile(path)?     # returns early if Err
  parsed = parse(content)?
  Ok(parsed)
}

# Panic for bugs (unrecoverable)
panic("this should never happen")
assert(x > 0, "x must be positive")
```

### try/catch

Define error types, then pattern match:

```
# Error types
type Error =
  | AssertionError(String)
  | ArithmeticError(String)
  | IoError(IoReason)
  | Timeout

type IoReason =
  | NotFound(String)
  | PermissionDenied(String)
  | ConnectionFailed(String)

# Pattern matching on errors
try {
  operation()
} catch
  AssertionError(msg) -> log("assertion failed: ${msg}")
  ArithmeticError(_) -> log("math error")
  IoError(NotFound(path)) -> log("file not found: ${path}")
  IoError(reason) -> log("IO error: ${reason}")
  e -> panic(e)                      # re-raise unknown
end

# With finally
try {
  file = open("data.txt")
  process(file)
} catch
  IoError(reason) -> log("IO error: ${reason}")
  e -> panic(e)
finally
  file.close()
end

# Assign result
result = try {
  compute()
} catch
  _ -> fallback()
end

# In concurrent code: uncaught panic crashes process
# Supervisor handles restart
```

### When to use what

| Mechanism | Use for |
|-----------|---------|
| `Result[T, E]` | Expected errors (IO, parsing, validation) |
| `?` operator | Propagate Result, early return on Err |
| `panic(msg)` | Bugs, invariant violations |
| `try/catch` | Boundaries (FFI, untrusted code, cleanup) |
| Supervisor | Restart crashed processes |

### Boolean Operators

Short-circuit evaluation:

```
# && stops if first is false
valid = notEmpty(s) && isNumeric(s)

# || stops if first is true
name = providedName || "anonymous"

# ! negation
if !done then continue()
```

### Bindings

Nostos has two kinds of bindings: **immutable** (default) and **mutable** (with `var`).

#### Immutable Bindings

Immutable bindings cannot be reassigned. Attempting to rebind an immutable variable to a *different* value causes a runtime assertion error. However, rebinding to the *same* value succeeds (pattern matching semantics):

```
# Immutable (default)
x = 5
x = 5                          # OK: same value, acts as assertion
x = 6                          # RUNTIME ERROR: Assertion failed: 5 != 6

# Useful for asserting invariants
result = compute()
result = expected_value        # Fails if compute() returned something else
```

#### Mutable Bindings

Use `var` to create a mutable binding that can be reassigned:

```
# Mutable
var count = 0
count = count + 1              # ok
count += 1                     # ok (sugar)

# Mutable can shadow immutable
x = 10                         # immutable
var x = 15                     # new mutable binding shadows the old one
x = 20                         # ok, x is now mutable
```

#### Function Parameters

Function parameters are immutable by default:

```
add_one(x) = {
    x = x + 1                  # RUNTIME ERROR: x is immutable
    x
}
```

#### Pattern Matching with Existing Variables

When an immutable variable appears in a pattern, it acts as a **constraint** rather than a new binding. The pattern only matches if the value equals the existing variable's value:

```
# Variable as constraint in match
x = 5
match (5, 10)
    (x, y) -> y * 2            # Matches! x must equal 5, binds y to 10
end
# Result: 20

match (3, 10)
    (x, y) -> y * 2            # Doesn't match! 3 != x (which is 5)
    _ -> 0
end
# Result: 0

# This enables Erlang-style constraint matching
expected = 200
match response
    {status: expected, body: b} -> process(b)  # Only if status == 200
    {status: s, body: b} -> handle_error(s, b)
end
```

#### Guard Fallthrough

When a pattern matches but its guard fails, execution falls through to try the next arm:

```
classify(n) = match n
    x when x > 10 -> "big"
    x when x > 3 -> "medium"   # Tried if first guard fails
    _ -> "small"
end

classify(5)   # "medium" - first pattern matches but guard fails, tries second
classify(1)   # "small" - both guards fail, falls through to wildcard
```

#### Destructuring

```
(a, b) = getTuple()
{x, y} = point                 # punning: {x, y} means {x: x, y: y}
{name: n, age: a} = person
[head | tail] = items
Some(value) = maybeValue       # runtime error if None
```

#### Pin Operator

The pin operator `^` explicitly asserts equality with an existing variable in patterns (same as using an immutable variable directly):

```
expected = 200
{status: ^expected, body: data} = response
```

#### Fresh Variables Per Match Arm

Each match arm has its own scope for pattern-bound variables. Variables bound in one arm don't leak to other arms:

```
match value
    x -> x * 2      # x bound here
    y -> y + 1      # y is independent, x not visible
end
```

### Functions and Pattern Matching

Pattern matching is the core of the language. Patterns can appear directly in function parameters with full flexibility:

```
# Simple function
add(a, b) = a + b

# Literals in patterns
fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)

# Multiple literal patterns
f(2, x) = x * 2
f(3, x) = x * 3
f(_, x) = x

# List patterns with literals
f([2 | 3 | t]) = t             # matches list starting with 2, 3
f([0 | t]) = t                 # matches list starting with 0
f([h | t]) = h                 # matches any non-empty list
f([]) = Nil                    # matches empty list

# Multiple list elements
sum([]) = 0
sum([x]) = x
sum([x | y | rest]) = x + y + sum(rest)

# Guards
abs(n) when n >= 0 = n
abs(n) = -n

max(a, b) when a >= b = a
max(a, b) = b

# Deep patterns
getName({name: n, _}) = n
getX({x, y}) = x

# Variant patterns
unwrap(Some(x)) = x
unwrap(None) = error("empty")

map(f, None) = None
map(f, Some(x)) = Some(f(x))

map(f, []) = []
map(f, [h | t]) = [f(h) | map(f, t)]

# Nested patterns
processResponse({status: 200, body: {data: [first | _]}}) = first
processResponse({status: code, _}) = error("status: " ++ code.show())

# Pin operator in patterns
matchValue(^expected, value) = value == expected

# Combining literals, destructuring, and guards
process(0, _) = "zero first"
process(_, 0) = "zero second"
process(a, b) when a == b = "equal"
process(a, b) = "different"

# Type annotations (optional, inferred)
length(p: Point) -> Float = sqrt(p.x ** 2 + p.y ** 2)
```

### Lambdas

```
double = x => x * 2
add = (a, b) => a + b
items.map(x => x * 2).filter(x => x > 0)
```

### Blocks

Multi-expression bodies use braces, last expression is returned:

```
# Single expression, no braces
double(x) = x * 2

# Multi-expression, braces required
process(data) = {
  validated = validate(data)
  transformed = transform(validated)
  save(transformed)
}

# Multi-line lambdas
h = x => {
  y = x * 2
  y + 1
}

# Inline blocks
result = {
  x = compute()
  y = transform(x)
  x + y
}

# Iteration with multi-line body
items.each(x => {
  processed = transform(x)
  println(processed)
})
```

Scoping uses shadowing (like Rust/ML):

```
x = 5
result = {
  x = 10             # shadows outer x
  x + 1              # => 11
}
# x still 5 here
```

### Methods via UFCS

Any function taking `T` as first argument can be called as method on `T`:

```
# Define as function
length(p: Point) = sqrt(p.x ** 2 + p.y ** 2)
scale(p: Point, s: Float) = Point(p.x * s, p.y * s)

# Call as method
point.length()
point.scale(2.0).length()

# Chaining
items
  .filter(x => x > 0)
  .map(square)
  .take(10)
```

### Record Operations

```
# Construction
p = Point(3.0, 4.0)
p = Point(x: 3.0, y: 4.0)      # named fields

# Field access
p.x
p.y

# Update (creates new record)
q = Point(p, x: 10.0)          # copy p, override x
r = Point(p, x: 0.0, y: 0.0)   # copy p, override x and y

# Mutation (only for var type)
var type MutPoint = {x: Float, y: Float}
p = MutPoint(3.0, 4.0)
p.x = 88.0                     # ok
p.y += 1.0                     # ok
```

### Traits

```
# Define trait
trait Show
  show(self) -> String

trait Eq
  ==(self, other: Self) -> Bool
  !=(self, other) = !(self == other)    # default impl

trait Ord: Eq                            # extends Eq
  compare(self, other: Self) -> Ordering
  <(self, other) = self.compare(other) == Less
  >(self, other) = self.compare(other) == Greater

# Implement trait
Point: Show
  show(self) = "(" ++ self.x.show() ++ ", " ++ self.y.show() ++ ")"

Point: Eq
  ==(self, other) = self.x == other.x && self.y == other.y

# Generic implementation
Option[T]: Show when T: Show
  show(None) = "None"
  show(Some(x)) = "Some(" ++ x.show() ++ ")"
```

### Control Flow

```
# If expression
max(a, b) = if a > b then a else b

# Multi-line if
result = if condition then
  doSomething()
  computeValue()
else
  alternative()

# Match expression
describe(n) = match n
  0 -> "zero"
  1 -> "one"
  _ -> "many"

# Match with guards
classify(n) = match n
  0 -> "zero"
  n when n > 0 -> "positive"
  _ -> "negative"

# Usually prefer function clauses over match
describe(0) = "zero"
describe(1) = "one"
describe(_) = "many"
```

### Modules

```
module Geometry

  type Point = {x: Float, y: Float}
  
  origin = Point(0.0, 0.0)
  
  distance(a: Point, b: Point) = 
    sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

end

# Usage
p = Geometry.Point(3.0, 4.0)
Geometry.distance(p, Geometry.origin)

# Import
use Geometry.{Point, distance}
```

### Visibility

Module items (functions, types, nested modules) are **private by default** and can be made public with the `pub` keyword:

```
module Math

  # Private helper (not accessible outside module)
  square(x) = x * x

  # Public function (accessible from outside)
  pub hypotenuse(a, b) = sqrt(square(a) + square(b))

  # Public type
  pub type Point = {x: Float, y: Float}

  # Public nested module
  pub module Trig
    pub sin(x) = ...
    cos(x) = ...                   # private within Trig
  end

end

# Usage from outside
Math.hypotenuse(3.0, 4.0)          # ok - pub function
Math.square(5)                      # ERROR - private function
p = Math.Point(1.0, 2.0)           # ok - pub type
Math.Trig.sin(0.5)                 # ok - pub function in pub module
Math.Trig.cos(0.5)                 # ERROR - cos is private
```

#### Private Record Fields

Record fields can also be marked as private:

```
module Account

  # Private field
  type T = {private balance: Float, name: String}

  # Public functions provide controlled access
  pub new(name, initial) = T(balance: initial, name: name)
  pub deposit(self: T, amount) = T(self, balance: self.balance + amount)
  pub balance(self: T) = self.balance

end

# Usage
acc = Account.new("Alice", 100.0)
acc.balance()                      # ok - uses public accessor
acc.balance                        # ERROR - field is private
```

### Operators

Standard precedence (unlike Smalltalk):

```
2 + 3 * 4                      # 14, not 20

# Precedence (high to low):
# **                           (exponentiation, right-assoc)
# * / %                        (multiplicative)
# + -                          (additive)
# ++ --                        (concatenation)
# == != < > <= >=              (comparison)
# &&                           (logical and)
# ||                           (logical or)
```

---

## Introspection

Everything is inspectable at runtime.

### Value Introspection

```
point = Point(3.0, 4.0)

typeOf(point)                  # => Point
point.fields()                 # => ["x", "y"]
point.get("x")                 # => 3.0
point.set("x", 5.0)            # => Point(5.0, 4.0) (if mutable)
point.toMap()                  # => {x: 3.0, y: 4.0}
```

### Type Introspection

```
Point.name                     # => "Point"
Point.fields                   # => [{name: "x", type: Float, mutable: false}, ...]
Point.implements               # => [Show, Eq]
Point.mutable                  # => false
Point.new({x: 1.0, y: 2.0})    # construct dynamically
```

### Function Introspection

```
fib.name                       # => "fib"
fib.arity                      # => 1
fib.params                     # => [{name: "n", type: Int}]
fib.returnType                 # => Int
fib.clauses                    # => [Clause(...), Clause(...), Clause(...)]
fib.source                     # => "fib(0) = 0\nfib(1) = 1\n..."
fib.ast                        # => full AST
fib.calls                      # => [fib, +, -]
fib.callers                    # => functions that call fib
```

### Trait Introspection

```
Show.name                      # => "Show"
Show.required                  # => [{name: "show", sig: Self -> String}]
Show.implementations           # => [Point, Int, Float, String, ...]
Show.implementedBy?(Point)     # => true
```

### Module Introspection

```
Geometry.types                 # => [Point, Line, Circle]
Geometry.functions             # => [distance, intersect, ...]
Geometry.exports               # => [Point, distance]
```

### Search and Query

```
# Find all functions that take a Point
Functions.where(f => f.params.any(p => p.type == Point))

# Find all types that implement Show
Types.where(t => t.implements?(Show))

# Find all functions matching a signature
Functions.matching(Int -> Int)
Functions.matching((Point, Float) -> Point)
```

### AST Manipulation

```
ast = quote(x + y * 2)
ast.walk(node => println(node.type))
ast.transform(node => match node
  Var(name) -> Var(name.uppercase())
  other -> other
)
```

---

## Live Development

### Eval and Quote

```
eval("1 + 2 * 3")              # => 7

ast = quote(x + y * 2)
ast.eval({x: 1, y: 3})         # => 7

# Splice values
y = 10
ast = quote(x + ${y})          # => BinOp("+", Var("x"), Lit(10))
```

### Runtime Modification

```
# Define new function
Functions.eval("double(x) = x * 2")

# Redefine existing function
Functions.define("length",
  params: [{name: "p", type: Point}],
  body: quote(p.x.abs() + p.y.abs())
)

# Add trait implementation at runtime
Traits.implement(Show, for: Point,
  show: p => "⟨" ++ p.x.show() ++ ", " ++ p.y.show() ++ "⟩"
)

# Create type at runtime
Types.define("Vec3",
  fields: [{name: "x", type: Float},
           {name: "y", type: Float},
           {name: "z", type: Float}]
)
```

### Image Save/Load

```
Image.save("snapshot.img")
Image.load("snapshot.img")
```

### Watchers

```
fib.onCall(args => println("fib called with " ++ args.show()))
Point.onCreate(p => println("new point: " ++ p.show()))
```

---

## Concurrency

Actor-based concurrency with lightweight processes. See `nostos-concurrency.md` for full specification.

### Quick Overview

```
# Define message types
type Request = GetStatus(Pid) | SetValue(Int, Pid)
type Response = Status(Int) | Ok

# Spawn process
pid = spawn(worker)
pid = spawn_link(worker)       # linked (dies together)

# Send message
pid <- GetStatus(self())

# Receive with pattern matching
receive
  Status(n) -> handle(n)
  Ok -> done()
after 5000 ->
  timeout()
end

# Named processes
register("logger", pid)
lookup("logger") <- LogMessage("hello")

# Supervision
children = [
  Child(id: "worker1", start: worker1),
  Child(id: "worker2", start: worker2)
]
Supervisor.start(children, strategy: OneForOne)
```

### Why Actors Fit Nostos

- **Immutability** — Messages copied safely, no races
- **Pattern matching** — `receive` uses same patterns as functions
- **Type safety** — Message protocols are typed
- **Fault tolerance** — Let it crash, supervise, restart
- **Introspection** — Processes are inspectable values
- **Hot reload** — Update code per process

---

## FFI (Foreign Function Interface)

Calling external code (C, Rust):

```
# Declare external function
extern sqrt(x: Float) -> Float from "libm"
extern puts(s: CString) -> Int from "libc"

# Use normally
result = sqrt(2.0)

# Structs map to C layout
extern type FILE

extern fopen(path: CString, mode: CString) -> Ptr[FILE] from "libc"
extern fclose(file: Ptr[FILE]) -> Int from "libc"

# Callbacks (Nostos function to C)
extern qsort(
  base: Ptr[Void],
  count: Int,
  size: Int,
  compare: Fn[(Ptr[Void], Ptr[Void]) -> Int]
) from "libc"
```

### Safety

```
# FFI code is unsafe, must be wrapped
# Use safe wrappers for user code

module File
  
  open(path: String) -> Result[Handle, IoError] = {
    ptr = fopen(path.toCString(), "r".toCString())
    if ptr.isNull() then
      Err(IoError.new("cannot open"))
    else
      Ok(Handle(ptr))
  }

end
```

---

## IO

IO is explicit, not hidden:

```
# IO actions return IO[T]
readLine() -> IO[String]
println(s: String) -> IO[()]
readFile(path: String) -> IO[Result[String, IoError]]

# Chain with flatMap
greet() -> IO[()] = {
  println("What is your name?")
    .flatMap(_ => readLine())
    .flatMap(name => println("Hello, ${name}!"))
}

# Or use do-notation sugar
greet() -> IO[()] = do
  println("What is your name?")
  name = readLine()
  println("Hello, ${name}!")
end

# Run at top level
main() = greet().run()
```

### Practical Compromise

For pragmatic use, IO can be implicit in impure contexts:

```
# In REPL or scripts, IO is automatic
println("hello")               # just works

# In pure modules, mark as IO
module Pure
  compute(x) = x * 2           # pure, no IO
end

module App
  run() -> IO[()] = {          # must declare IO
    println("Starting...")
    result = Pure.compute(21)
    println("Result: ${result}")
  }
end
```

---

## Testing

Built-in test framework:

```
module Math.Test

test "addition" = {
  assert(add(2, 3) == 5)
  assert(add(-1, 1) == 0)
}

test "fib computes correctly" = {
  assert(fib(0) == 0)
  assert(fib(1) == 1)
  assert(fib(10) == 55)
}

test "fib fails on negative" = {
  assertPanic({ fib(-1) })
}

test "async process" = {
  pid = spawn(worker)
  pid <- {ping, self()}
  
  assertReceive({pong, _}, timeout: 1000)
}

end

# Run tests
$ nostos test
$ nostos test Math.Test
$ nostos test --filter "fib"
```

### Assertions

```
assert(condition)
assert(condition, "message")
assertEqual(actual, expected)
assertNotEqual(a, b)
assertMatch(value, pattern)
assertPanic(block)
assertReceive(pattern, timeout: ms)
```

---

## Virtual Machine

### Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                        VM                              │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Type Table   │  │ Fn Table     │  │ Trait Table  │ │
│  │ (metadata)   │  │ (bytecode)   │  │ (vtables)    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │                 Scheduler                         │ │
│  │   Run queue | Wait queue | Work stealing         │ │
│  └──────────────────────────────────────────────────┘ │
│         │              │              │               │
│    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐         │
│    │ Process │    │ Process │    │ Process │   ...   │
│    │ Mailbox │    │ Mailbox │    │ Mailbox │         │
│    │ Regs    │    │ Regs    │    │ Regs    │         │
│    │ Heap    │    │ Heap    │    │ Heap    │         │
│    └────┬────┘    └────┬────┘    └────┬────┘         │
│         │              │              │               │
│    Local GC       Local GC       Local GC            │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │               JIT (Cranelift)                     │ │
│  │   Hot functions compiled to native                │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Register-based bytecode** — Maps cleanly to Cranelift's SSA form for easy JIT compilation
- **Typed operations** — Specialized instructions (AddInt, AddFloat) avoid runtime type checks
- **Unboxed primitives** — Int, Float, Bool stored directly, no heap allocation
- **Lightweight processes** — Green threads with per-process heaps
- **Per-process GC** — Isolated collection, no global pauses
- **Preemptive scheduling** — Reduction counting for fair scheduling
- **Metadata preserved** — Full introspection info kept separate from execution

---

## Rust Implementation

### Project Structure

```
nostos/
├── Cargo.toml
├── crates/
│   ├── syntax/              # Parsing
│   │   ├── lexer.rs
│   │   ├── parser.rs
│   │   └── ast.rs
│   │
│   ├── types/               # Type checking
│   │   ├── infer.rs
│   │   └── check.rs
│   │
│   ├── compiler/            # AST -> Bytecode
│   │   ├── lower.rs
│   │   └── bytecode.rs
│   │
│   ├── vm/                  # Interpreter
│   │   ├── interp.rs
│   │   ├── value.rs
│   │   ├── gc.rs
│   │   └── introspect.rs
│   │
│   ├── scheduler/           # Process scheduler
│   │   ├── process.rs
│   │   ├── mailbox.rs
│   │   └── supervisor.rs
│   │
│   ├── jit/                 # Cranelift JIT
│   │   └── compile.rs
│   │
│   └── repl/                # Interactive shell
│       └── main.rs
│
└── std/                     # Standard library
    ├── core.ns              # Primitives, operators
    ├── option.ns            # Option type
    ├── result.ns            # Result type
    ├── list.ns              # List functions
    ├── array.ns             # Array functions
    ├── map.ns               # Map functions
    ├── set.ns               # Set functions
    ├── string.ns            # String functions
    ├── io.ns                # IO operations
    ├── process.ns           # Concurrency
    ├── supervisor.ns        # Supervision trees
    └── test.ns              # Testing framework
```

### Implementation Phases

**Phase 1: Minimal Interpreter**
- Lexer and parser for core syntax
- AST-walking interpreter (no bytecode yet)
- Basic types: Int, Float, Bool, String, records, variants
- Pattern matching in function heads
- Simple REPL

**Phase 2: Bytecode VM**
- Compile AST to register-based bytecode
- Stack of frames with typed register arrays
- Proper GC (start with Boehm or simple mark-sweep)

**Phase 3: Type System**
- Structural type inference
- Generics with constraints
- Trait resolution

**Phase 4: Concurrency**
- Lightweight processes with mailboxes
- Scheduler with run/wait queues
- Per-process heaps and GC
- Links, monitors, supervision

**Phase 5: Introspection**
- Type/function/trait metadata at runtime
- Dynamic field access
- AST quoting and eval

**Phase 6: Optimization**
- Hot function detection
- Cranelift JIT for hot paths
- Inline caching for trait dispatch

**Phase 7: Image System**
- Serialize full VM state
- Save/load snapshots
- Live code modification

---

## Summary

| Feature | Choice |
|---------|--------|
| Typing | Structural, inferred, with simple generics |
| ADTs | Sum types (variants), product types (records), newtypes, aliases |
| Functions | Pattern matching in heads, multi-clause |
| Polymorphism | Traits (no inheritance) |
| Methods | UFCS (dot syntax on any function) |
| Mutation | `var` for bindings, `var type` for records |
| Visibility | Private by default, `pub` for public |
| Concurrency | Actors, message passing, supervision |
| Metaprogramming | Full introspection, eval, quote, live modification |
| VM | Register-based, typed bytecode |
| JIT | Cranelift |
| GC | Precise (type-aware), per-process heaps |
