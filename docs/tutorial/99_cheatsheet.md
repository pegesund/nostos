# Nostos Tutorial Cheat Sheet

Quick reference for concepts covered in the tutorials.

---

## 01: Basics

```nostos
# Comments
x = 42                    # Immutable binding
var y = 0                 # Mutable local variable
y = y + 1                 # Reassign mutable

# Types
Int Float Bool String Char ()

# Operators
+ - * / %                 # Arithmetic
== != < <= > >=           # Comparison
&& || !                   # Logical
++                        # String concat

# Print
println("Hello")
print("no newline")
show(42)                  # Any -> String
```

---

## 02: Functions

```nostos
# Definition
add(a: Int, b: Int) -> Int = a + b
greet(name) = "Hello, " ++ name    # Types inferred

# Block body
process(x) = {
    y = x * 2
    y + 1                 # Last expression returned
}

# Lambdas
x => x * 2
(a, b) => a + b

# UFCS (method-style calls)
5.add(3)                  # Same as add(5, 3)

# Pattern matching in definitions
factorial(0) = 1
factorial(n) = n * factorial(n - 1)
```

---

## 03: Pattern Matching

```nostos
# Match expression
match value {
    0 -> "zero",
    1 -> "one",
    n if n > 0 -> "positive",
    _ -> "negative"
}

# Destructuring
(a, b) = tuple
[h | t] = list            # Head and tail
{ name, age } = person

# Variant patterns
match opt {
    Some(x) -> use(x),
    None -> default
}

# Nested patterns
match data {
    (Some(x), [h | _]) -> x + h,
    _ -> 0
}
```

---

## 04: Lists & Tuples

```nostos
# Lists
list = [1, 2, 3]
list[0]                   # Access (may panic)
list.get(0)               # Safe access -> Option
[h | t] = list            # Destructure

# Common operations
list.map(x => x * 2)
list.filter(x => x > 0)
list.fold(0, (acc, x) => acc + x)
list.length()
list.push(4)              # Returns new list
list.append([4, 5])

# Tuples
pair = (1, "hello")
(a, b) = pair             # Destructure
triple = (1, 2, 3)
```

---

## 05: Maps & Sets

```nostos
# Maps
map = %{"a": 1, "b": 2}
map["a"]                  # Access (may panic)
map.get("a")              # Safe -> Option
map.insert("c", 3)        # Returns new map
map.remove("a")
map.keys()
map.values()

# Sets
set = Set.from([1, 2, 2, 3])  # {1, 2, 3}
set.contains(2)           # true
set.insert(4)
a.union(b)
a.intersection(b)
```

---

## 06: Typed Arrays

```nostos
# Fixed-type arrays for performance
Int64Array.new(1000)
Float64Array.from([1.0, 2.0, 3.0])

arr[0]                    # Access
arr.set(0, value)         # Mutate in place
arr.length()
arr.slice(start, end)
```

---

## 07: Type System

```nostos
# Records
type Person = { name: String, age: Int }
p = Person("Alice", 30)
p.name

# Variants (sum types)
type Color = Red | Green | Blue
type Option[T] = Some(T) | None
type Result[T, E] = Ok(T) | Err(E)

# Variants with data
type Shape = Circle(Float) | Rectangle(Float, Float)

# Generic types
type Box[T] = { value: T }
type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
```

---

## 08: Traits

```nostos
# Define trait
trait Show {
    show(self) -> String
}

# Implement for type
impl Show for Person {
    show(self) = self.name ++ " (" ++ show(self.age) ++ ")"
}

# Use
person.show()
```

---

## 09: Built-in Traits

```nostos
# Eq - equality comparison
impl Eq for MyType { eq(self, other) = ... }
a == b

# Ord - ordering
impl Ord for MyType { cmp(self, other) = ... }
a < b

# Hash - for Map/Set keys
impl Hash for MyType { hash(self) = ... }

# Default - default value
impl Default for MyType { default() = MyType(...) }
```

---

## 10: Trait Bounds

```nostos
# Generic function with bounds
printAll[T: Show](items: List[T]) = {
    items.map(x => println(x.show()))
}

# Multiple bounds
compare[T: Eq + Ord](a: T, b: T) = ...
```

---

## 11: Error Handling

```nostos
# Option
Some(42)  None
opt.map(f)
opt.unwrapOr(default)
opt.isSome()  opt.isNone()

# Result
Ok(value)  Err(error)
res.map(f)  res.mapErr(f)
res.unwrapOr(default)
res.isOk()  res.isErr()

# Try/catch
try {
    riskyOperation()
} catch {
    "specific" -> handle(),
    e -> handleGeneric(e)
}

# Throw
throw("error message")
```

---

## 12: Modules

```nostos
# math.nos
pub add(a, b) = a + b     # Public
helper(x) = x * 2         # Private

# main.nos
import math
math.add(1, 2)

# Selective import
import math.{add, subtract}
```

---

## 13: Standard Library

```nostos
# String
s.length()  s.toLower()  s.toUpper()
s.split(",")  s.trim()  s.contains("x")
s.replace("a", "b")  s.substring(0, 5)

# List
list.map(f)  list.filter(f)  list.fold(init, f)
list.find(f)  list.any(f)  list.all(f)
list.sort()  list.reverse()  list.take(n)

# Math
abs(x)  min(a, b)  max(a, b)
sqrt(x)  pow(x, n)  floor(x)  ceil(x)
```

---

## 14: JSON

```nostos
import json

# Parse
json.parse('{"name": "Alice"}')  # Result[Json, String]

# Decode to type
type User = { name: String, age: Int }
json.decode[User]('{"name": "Alice", "age": 30}')

# Encode
json.encode(user)         # -> String

# Access Json values
j["field"]
j[0]                      # Array index
```

---

## 15: Concurrency

```nostos
# Spawn process
pid = spawn(() => work())

# Message passing
send(pid, message)
msg = receive()

# Receive with patterns
receive {
    "ping" -> "pong",
    ("data", x) -> process(x)
}

# Receive with timeout
receiveTimeout(1000) { msg -> handle(msg) }
```

---

## 16: Async Runtime

```nostos
# Spawn and await
handle = spawn(() => computation())
result = await(handle)

# Sleep
sleep(1000)               # Milliseconds

# Parallel operations
results = items.parMap(f)
```

---

## 17: Async I/O & HTTP

```nostos
import http

# GET
response = http.get(url)
response.status           # Int
response.body             # String

# POST
http.post(url,
    body: data,
    headers: %{"Content-Type": "application/json"}
)

# File I/O
import file
file.read("path")         # Result[String, String]
file.write("path", content)
```

---

## 18: Reflection

```nostos
# Type info at runtime
info = typeInfo("Person")
info["kind"]              # "record"
info["fields"]            # List of field info

# Dynamic construction
makeRecordByName("Person", %{"name": "Alice", "age": 30})
makeVariantByName("Option", "Some", %{"0": 42})
```

---

## 19: Debugging & Profiling

```nostos
# Debug print
debug(value)              # Prints with type info

# Assertions
assert_eq(expected, actual)

# Timing
start = now()
# ... work ...
elapsed = now() - start
```

---

## 20: HTML Templating

```nostos
import html

# Build HTML
div(class: "container") {
    h1() { "Title" }
    p() { "Content" }
    ul() {
        items.map(i => li() { i })
    }
}

# Render to string
html.render(element)
```

---

## 21: Mutability

```nostos
# Immutable (default)
x = 42                    # Cannot reassign

# Local mutable
var counter = 0
counter = counter + 1

# Module-level mutable
mvar globalState: Int = 0

# Mutable in closures
makeCounter() = {
    var count = 0
    () => {
        count = count + 1
        count
    }
}
```

---

## 22: Templates (Metaprogramming)

```nostos
# Basic template
template double(fn) = quote {
    result = ~fn.body
    result * 2
}

@double
getValue() = 21           # Returns 42

# Function metadata
~fn.name                  # Function name
~fn.body                  # Function body AST
~fn.params                # Parameters list

# Type decorator
template withGetters(typeDef) = quote {
    ~typeDef
    ~typeDef.fields.map(f =>
        eval("get_" ++ f.name ++ "(r) = r." ++ f.name))
}

# Code generation
eval("code as string")    # Parse string as code
gensym("prefix")          # Unique identifier

# Compile-time conditionals
~if condition {
    quote { ... }
} else {
    quote { ... }
}

# Try/catch in templates
template withFallback(fn, fallback) = quote {
    try { ~fn.body } catch { _ -> ~fallback }
}
```

---

## Quick Syntax Reference

| Syntax | Meaning |
|--------|---------|
| `x = 42` | Immutable binding |
| `var x = 42` | Mutable local |
| `mvar x: T = v` | Mutable module-level |
| `f(a, b)` | Function call |
| `a.f(b)` | UFCS call (same as f(a, b)) |
| `x => x * 2` | Lambda |
| `[1, 2, 3]` | List |
| `%{"a": 1}` | Map |
| `(a, b)` | Tuple |
| `Type { field: v }` | Record construction |
| `match x { ... }` | Pattern match |
| `if c then a else b` | Conditional |
| `type T = ...` | Type definition |
| `trait T { ... }` | Trait definition |
| `impl T for U { ... }` | Trait implementation |
| `@decorator` | Apply template |
| `~expr` | Splice in template |
| `quote { ... }` | Capture AST |
