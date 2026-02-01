# Nostos Language Reference

Complete reference for the Nostos programming language. This document combines all skill documentation into a single file.

## Table of Contents

1. [Language Overview](#nostos-language-overview)
2. [Gotchas & Common Mistakes](#nostos-gotchas--common-mistakes)
3. [Standard Library Reference](#nostos-standard-library-reference)
4. [End-to-End Examples](#nostos-end-to-end-examples)
5. [Quick Reference Cheat Sheet](#nostos-quick-reference-cheat-sheet)
6. [Basics](#nostos-basics)
7. [Collections](#collections-in-nostos)
8. [Concurrency](#concurrency-in-nostos)
9. [Control Flow](#control-flow-in-nostos)
10. [Database](#database-access-in-nostos)
11. [Error Handling](#error-handling-in-nostos)
12. [File I/O](#file-io-in-nostos)
13. [Functions](#functions-in-nostos)
14. [HTTP](#http-in-nostos)
15. [JSON](#json-in-nostos)
16. [Lists](#lists-in-nostos)
17. [Modules](#modules-in-nostos)
18. [Reactive Programming](#reactive-programming-in-nostos)
19. [Strings](#strings-in-nostos)
20. [Templates & Metaprogramming](#templates--metaprogramming-in-nostos)
21. [Testing](#testing-in-nostos)
22. [Traits](#traits-in-nostos)
23. [Types](#types-in-nostos)
24. [WebSockets](#websockets-in-nostos)

---

# Nostos Language Overview

Nostos is a modern functional programming language with a focus on expressiveness, safety, and metaprogramming.

## Key Characteristics

**Expression-based**: Everything is an expression that returns a value. No statements.
```nostos
# if/else returns a value
result = if x > 0 { "positive" } else { "non-positive" }

# Blocks return their last expression
value = {
    a = 10
    b = 20
    a + b  # This is returned
}
```

**Algebraic Data Types**: First-class sum types (variants) and product types (records).
```nostos
type Result[T, E] = Ok(T) | Err(E)
type Person = { name: String, age: Int }
```

**Pattern Matching**: Destructure data elegantly.
```nostos
match result {
    Ok(value) -> "Got: " ++ show(value),
    Err(msg) -> "Error: " ++ msg
}
```

**UFCS (Uniform Function Call Syntax)**: Call any function as a method.
```nostos
# These are equivalent:
double(5)
5.double()

# Enables fluent chaining:
[1, 2, 3].map(x => x * 2).filter(x => x > 2).sum()
```

**Compile-time Metaprogramming**: Templates generate code at compile time.
```nostos
template logged(fn) = quote {
    println("Calling " ++ ~fn.name)
    ~fn.body
}

@logged
compute() = 42  # Prints "Calling compute" when called
```

**Hindley-Milner Type Inference**: Types are inferred but can be annotated.
```nostos
# Compiler infers: add(Int, Int) -> Int
add(a, b) = a + b

# Explicit when needed
identity[T](x: T) -> T = x
```

## Compared to Other Languages

| Feature | Nostos | Similar to |
|---------|--------|------------|
| Algebraic types | `type Option[T] = Some(T) \| None` | Rust, Haskell, OCaml |
| Pattern matching | `match x { ... }` | Rust, Scala, Elixir |
| Lambdas | `x => x * 2` | JavaScript, Kotlin |
| String concat | `"a" ++ "b"` | Haskell, Elixir |
| List literals | `[1, 2, 3]` | Python, JavaScript |
| Map literals | `%{"a": 1}` | Elixir (similar) |
| Immutable by default | `x = 5` | Rust, Scala |
| UFCS | `5.double()` | D, Nim |
| Templates/macros | `@decorator` | Rust (proc macros), Elixir |
| Traits | `trait Show { show() }` | Rust, Scala, Haskell (typeclasses) |

## What Nostos is Good For

- **Scripting and automation** - Concise syntax, fast startup
- **Data transformation** - Pattern matching, list operations
- **Web services** - Built-in HTTP server/client, JSON support
- **DSLs** - Templates enable domain-specific abstractions
- **Learning FP** - Clean syntax without ceremony

## Program Structure

```nostos
# Imports (optional)
import json
import http

# Type definitions
type User = { id: Int, name: String }

# Constants
const API_URL = "https://api.example.com"

# Functions
fetchUser(id: Int) -> Result[User, String] = {
    # ...
}

# Entry point
main() = {
    match fetchUser(1) {
        Ok(user) -> println("Hello, " ++ user.name),
        Err(e) -> println("Error: " ++ e)
    }
}
```

## Module System

```nostos
# math.nos
pub add(a: Int, b: Int) = a + b      # Public
helper(x: Int) = x * 2               # Private (no pub)

# main.nos
import math
main() = math.add(1, 2)              # Use qualified name
```

## Concurrency Model

Nostos uses lightweight tasks (green threads) with message passing:

```nostos
import concurrent

main() = {
    # Spawn concurrent task
    handle = spawn(() => {
        sleep(100)
        42
    })

    # Wait for result
    result = await(handle)  # 42
}
```

## Error Handling

Two mechanisms: `Result` types for expected errors, exceptions for unexpected ones.

```nostos
# Result for expected failures
parseNumber(s: String) -> Result[Int, String] = {
    # ...
}

# Exceptions for unexpected failures
riskyOperation() = {
    if somethingWrong {
        throw("Unexpected error")
    }
    result
}

# Catch exceptions
safe() = try {
    riskyOperation()
} catch {
    e -> "Caught: " ++ e
}
```

---

# Nostos Gotchas & Common Mistakes

Things that trip people up when learning Nostos.

## Syntax Differences

### Comments use `#`, not `//`
```nostos
# Correct: hash for comments
// Wrong: this is NOT a comment, it's a syntax error
```

### String concatenation is `++`, not `+`
```nostos
# Correct
"Hello" ++ " " ++ "World"

# Wrong - type error (+ is for numbers)
"Hello" + " " + "World"
```

### No semicolons
```nostos
# Correct - expressions separated by newlines or in blocks
x = 1
y = 2

# Wrong - semicolons are syntax errors
x = 1;
y = 2;
```

### Commas in match arms
```nostos
# Correct - commas between match arms
match x {
    1 -> "one",
    2 -> "two",
    _ -> "other"
}

# Wrong - no commas causes parse error
match x {
    1 -> "one"
    2 -> "two"
}
```

### `then` required in single-line if
```nostos
# Multi-line if (no then needed)
if x > 0 {
    "positive"
} else {
    "non-positive"
}

# Single-line if REQUIRES then
if x > 0 then "positive" else "non-positive"

# Wrong - missing then
if x > 0 "positive" else "non-positive"
```

## Variable Mutability

### `var` vs `mvar` - different scopes
```nostos
# var = local mutable variable (inside functions)
process() = {
    var counter = 0
    counter = counter + 1
    counter
}

# mvar = module-level mutable variable (top-level)
mvar globalCounter: Int = 0

increment() = {
    globalCounter = globalCounter + 1
}

# Wrong - var at module level
var badGlobal = 0  # Error!

# Wrong - mvar inside function
process() = {
    mvar x = 0  # Error!
}
```

### Immutable by default
```nostos
# This creates an immutable binding
x = 42
x = 43  # Error! Cannot reassign

# Use var for mutability
var x = 42
x = 43  # OK
```

## Try/Catch Syntax

### Catch uses pattern matching, not variable binding
```nostos
# Correct - pattern matching syntax
try {
    riskyOperation()
} catch {
    "specific error" -> handleSpecific(),
    e -> handleGeneric(e)  # Catch-all pattern
}

# Wrong - this is NOT Nostos syntax
try {
    riskyOperation()
} catch (e) {
    handleError(e)
}
```

### Catch arms need commas
```nostos
# Correct
try { x } catch {
    "error1" -> handle1(),
    "error2" -> handle2(),
    _ -> handleOther()
}

# Wrong - missing commas
try { x } catch {
    "error1" -> handle1()
    "error2" -> handle2()
}
```

## Type System

### Generic syntax uses `[]`, not `<>`
```nostos
# Correct
List[Int]
Map[String, Int]
Option[T]

# Wrong - angle brackets are comparison operators
List<Int>  # Parsed as: List < Int > (comparison!)
```

### Type field access uses `.ty`, not `.type`
```nostos
# In templates, accessing field types:
template example(typeDef) = quote {
    ~typeDef.fields.map(f =>
        # Correct - use .ty
        eval(f.name ++ ": " ++ f.ty)
    )
}

# Wrong - type is a keyword
f.type  # Error!
```

### Records need type definitions
```nostos
# Correct - define type first
type Person = { name: String, age: Int }
p = Person("Alice", 30)

# Wrong - anonymous records don't exist
p = { name: "Alice", age: 30 }  # Error!
```

## Functions

### Single-expression functions don't need braces
```nostos
# Both are correct:
add(a, b) = a + b
add(a, b) = { a + b }

# But multi-statement needs braces
process(x) = {
    y = x * 2
    y + 1
}
```

### Return is implicit (last expression)
```nostos
# Correct - last expression is returned
calculate(x) = {
    y = x * 2
    z = y + 1
    z  # This is returned
}

# Explicit return exists but rarely needed
earlyExit(x) = {
    if x < 0 { return 0 } else { () }
    x * 2
}
```

### Pattern matching in function definitions
```nostos
# Multiple clauses with patterns
factorial(0) = 1
factorial(n) = n * factorial(n - 1)

# List patterns
sum([]) = 0
sum([h | t]) = h + sum(t)

# The clauses are tried in order
```

## Lists and Collections

### List cons pattern is `[h | t]`, not `h::t`
```nostos
# Correct
match list {
    [] -> "empty",
    [h | t] -> "head: " ++ show(h)
}

# Wrong - not Nostos syntax
match list {
    h::t -> "head: " ++ show(h)
}
```

### Map literals use `%{}`
```nostos
# Correct
myMap = %{"a": 1, "b": 2}

# Wrong - this is a block, not a map
myMap = {"a": 1, "b": 2}  # Error!
```

### Indexing with `[]` vs `.get()`
```nostos
# Direct indexing (may panic)
list[0]       # First element, panics if empty
map["key"]    # Value for key, panics if missing

# Safe access with Option
list.get(0)   # Some(first) or None
map.get("key") # Some(value) or None
```

## Common Runtime Errors

### Forgetting to handle None/Err
```nostos
# This will panic if list is empty
first = list[0]

# Safe alternative
first = match list.get(0) {
    Some(x) -> x,
    None -> defaultValue
}
```

### Integer division truncates
```nostos
5 / 2   # Returns 2, not 2.5

# For float division, use floats
5.0 / 2.0  # Returns 2.5
```

### String comparison is case-sensitive
```nostos
"Hello" == "hello"  # false

# For case-insensitive comparison
"Hello".toLower() == "hello".toLower()  # true
```

## Templates

### `~` splices AST, not values
```nostos
template example(fn) = quote {
    # ~fn.body inserts the AST of the function body
    result = ~fn.body
    result * 2
}

# The splice happens at compile time, not runtime
```

### eval() parses strings as code
```nostos
template makeFn(name) = quote {
    # eval turns a string into code
    ~eval(~name ++ "() = 42")
}

# This generates: myFunc() = 42
@makeFn("myFunc")
type Dummy = Dummy {}
```

### gensym for unique names
```nostos
template safe(fn) = quote {
    # Without gensym, variable names might collide
    ~gensym("tmp") = ~fn.body
    ~gensym("tmp")  # Different name each time!
}
```

---

# Nostos Standard Library Reference

Quick reference for methods on built-in types.

## String

```nostos
s = "Hello, World!"

# Length and access
s.length()              # 13
s.charAt(0)             # 'H'
s.substring(0, 5)       # "Hello"

# Case conversion
s.toLower()             # "hello, world!"
s.toUpper()             # "HELLO, WORLD!"

# Search
s.contains("World")     # true
s.startsWith("Hello")   # true
s.endsWith("!")         # true
s.indexOf("o")          # Some(4)
s.lastIndexOf("o")      # Some(8)

# Manipulation
s.trim()                # Remove whitespace from ends
s.trimStart()           # Remove leading whitespace
s.trimEnd()             # Remove trailing whitespace
s.replace("World", "Nostos")  # "Hello, Nostos!"
s.replaceAll("l", "L")  # "HeLLo, WorLd!"

# Split and join
"a,b,c".split(",")      # ["a", "b", "c"]
["a", "b", "c"].join("-")  # "a-b-c"

# Conversion
"42".parseInt()         # Some(42) or None
"3.14".parseFloat()     # Some(3.14) or None
show(42)                # "42" (any value to string)

# Concatenation
"Hello" ++ " " ++ "World"  # "Hello World"

# Characters
s.chars()               # List of Char
s.bytes()               # List of Int (UTF-8 bytes)
```

## List[T]

```nostos
list = [1, 2, 3, 4, 5]

# Length and access
list.length()           # 5
list[0]                 # 1 (panics if out of bounds)
list.get(0)             # Some(1) or None
list.first()            # Some(1) or None
list.last()             # Some(5) or None

# Add/remove
list.push(6)            # [1, 2, 3, 4, 5, 6]
list.append([6, 7])     # [1, 2, 3, 4, 5, 6, 7]
list.prepend(0)         # [0, 1, 2, 3, 4, 5]
[h | t] = list          # h = 1, t = [2, 3, 4, 5]

# Transform
list.map(x => x * 2)    # [2, 4, 6, 8, 10]
list.filter(x => x > 2) # [3, 4, 5]
list.flatMap(x => [x, x])  # [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

# Reduce
list.fold(0, (acc, x) => acc + x)  # 15
list.reduce((a, b) => a + b)       # 15 (no initial value)
list.sum()              # 15 (for numeric lists)
list.product()          # 120 (for numeric lists)

# Search
list.find(x => x > 3)   # Some(4)
list.any(x => x > 3)    # true
list.all(x => x > 0)    # true
list.contains(3)        # true
list.indexOf(3)         # Some(2)

# Sort and reverse
list.sort()             # [1, 2, 3, 4, 5]
list.sortBy(x => -x)    # [5, 4, 3, 2, 1]
list.reverse()          # [5, 4, 3, 2, 1]

# Slice
list.take(3)            # [1, 2, 3]
list.drop(2)            # [3, 4, 5]
list.slice(1, 4)        # [2, 3, 4]
list.takeWhile(x => x < 4)  # [1, 2, 3]
list.dropWhile(x => x < 3)  # [3, 4, 5]

# Combine
list.zip([10, 20, 30])  # [(1, 10), (2, 20), (3, 30)]
[[1, 2], [3, 4]].flatten()  # [1, 2, 3, 4]

# Check
list.isEmpty()          # false
list.nonEmpty()         # true

# Convert
list.toSet()            # Set with unique elements
```

## Map[K, V]

```nostos
map = %{"a": 1, "b": 2, "c": 3}

# Access
map["a"]                # 1 (panics if missing)
map.get("a")            # Some(1) or None
map.getOrElse("x", 0)   # 0 (default if missing)

# Modify (returns new map)
map.insert("d", 4)      # %{"a": 1, "b": 2, "c": 3, "d": 4}
map.remove("a")         # %{"b": 2, "c": 3}
map.update("a", v => v + 10)  # %{"a": 11, "b": 2, "c": 3}

# Query
map.contains("a")       # true
map.size()              # 3
map.isEmpty()           # false

# Iterate
map.keys()              # ["a", "b", "c"]
map.values()            # [1, 2, 3]
map.entries()           # [("a", 1), ("b", 2), ("c", 3)]

# Transform
map.map((k, v) => (k, v * 2))     # %{"a": 2, "b": 4, "c": 6}
map.filter((k, v) => v > 1)       # %{"b": 2, "c": 3}
map.mapValues(v => v * 2)         # %{"a": 2, "b": 4, "c": 6}

# Merge
map.merge(%{"c": 30, "d": 4})     # %{"a": 1, "b": 2, "c": 30, "d": 4}
```

## Set[T]

```nostos
set = Set.from([1, 2, 3, 2, 1])  # {1, 2, 3}

# Modify (returns new set)
set.insert(4)           # {1, 2, 3, 4}
set.remove(2)           # {1, 3}

# Query
set.contains(2)         # true
set.size()              # 3
set.isEmpty()           # false

# Set operations
a = Set.from([1, 2, 3])
b = Set.from([2, 3, 4])
a.union(b)              # {1, 2, 3, 4}
a.intersection(b)       # {2, 3}
a.difference(b)         # {1}
a.isSubsetOf(b)         # false

# Convert
set.toList()            # [1, 2, 3]
```

## Option[T]

```nostos
some = Some(42)
none = None

# Check
some.isSome()           # true
none.isNone()           # true

# Extract
some.unwrap()           # 42 (panics if None)
none.unwrapOr(0)        # 0
some.getOrElse(0)       # 42

# Transform
some.map(x => x * 2)    # Some(84)
none.map(x => x * 2)    # None
some.flatMap(x => Some(x + 1))  # Some(43)
some.filter(x => x > 50)        # None

# Convert
some.toList()           # [42]
none.toList()           # []
some.okOr("error")      # Ok(42)
none.okOr("error")      # Err("error")

# Pattern match
match opt {
    Some(x) -> "got " ++ show(x),
    None -> "nothing"
}
```

## Result[T, E]

```nostos
ok = Ok(42)
err = Err("failed")

# Check
ok.isOk()               # true
err.isErr()             # true

# Extract
ok.unwrap()             # 42 (panics if Err)
err.unwrapErr()         # "failed"
ok.unwrapOr(0)          # 42
err.unwrapOr(0)         # 0

# Transform
ok.map(x => x * 2)      # Ok(84)
err.map(x => x * 2)     # Err("failed")
ok.mapErr(e => "Error: " ++ e)   # Ok(42)
err.mapErr(e => "Error: " ++ e)  # Err("Error: failed")
ok.flatMap(x => Ok(x + 1))       # Ok(43)

# Convert
ok.ok()                 # Some(42)
err.ok()                # None
ok.err()                # None
err.err()               # Some("failed")

# Pattern match
match result {
    Ok(value) -> "success: " ++ show(value),
    Err(e) -> "error: " ++ e
}
```

## Int / Float

```nostos
# Int operations
42.abs()                # 42
(-42).abs()             # 42
10.max(20)              # 20
10.min(20)              # 10
10.clamp(5, 15)         # 10
2.pow(10)               # 1024

# Float operations
3.14.floor()            # 3.0
3.14.ceil()             # 4.0
3.14.round()            # 3.0
3.14.abs()              # 3.14
(-1.5).abs()            # 1.5
4.0.sqrt()              # 2.0
2.0.pow(3.0)            # 8.0

# Conversion
42.toFloat()            # 42.0
3.14.toInt()            # 3
42.toString()           # "42"

# Ranges
(1..5)                  # [1, 2, 3, 4] (exclusive end)
(1..=5)                 # [1, 2, 3, 4, 5] (inclusive end)
```

## Char

```nostos
c = 'A'

c.isAlpha()             # true
c.isDigit()             # false
c.isAlphanumeric()      # true
c.isWhitespace()        # false
c.isUpper()             # true
c.isLower()             # false
c.toLower()             # 'a'
c.toUpper()             # 'A'
c.toInt()               # 65 (ASCII/Unicode value)
Char.fromInt(65)        # 'A'
```

## Tuple

```nostos
t = (1, "hello", true)

# Access by destructuring
(a, b, c) = t           # a = 1, b = "hello", c = true

# Two-element tuples have .0 and .1
pair = (10, 20)
pair.0                  # 10
pair.1                  # 20

# Swap
(10, 20).swap()         # (20, 10)
```

## Common Utility Functions

```nostos
# Printing
print("no newline")
println("with newline")
println("Value: " ++ show(42))

# Assertions (panic on failure)
assert_eq(expected, actual)
assert_eq(4, 2 + 2)

# Type conversion
show(anyValue)          # Convert to String
"42".parseInt()         # String to Int
"3.14".parseFloat()     # String to Float

# Comparison
min(a, b)               # Smaller of two
max(a, b)               # Larger of two

# Ranges
range(0, 5)             # [0, 1, 2, 3, 4]
range(0, 10, 2)         # [0, 2, 4, 6, 8] (with step)
```

## File I/O

```nostos
import file

# Read
content = file.read("path.txt")           # Result[String, String]
lines = file.readLines("path.txt")        # Result[List[String], String]
bytes = file.readBytes("path.bin")        # Result[List[Int], String]

# Write
file.write("path.txt", "content")         # Result[(), String]
file.writeLines("path.txt", ["a", "b"])   # Result[(), String]
file.append("path.txt", "more")           # Result[(), String]

# Check
file.exists("path.txt")                   # Bool
file.isFile("path.txt")                   # Bool
file.isDir("path")                        # Bool

# Directory
file.listDir(".")                         # Result[List[String], String]
file.createDir("newdir")                  # Result[(), String]
```

## JSON

```nostos
import json

# Parse
json.parse('{"name": "Alice", "age": 30}')  # Result[Json, String]

# Access Json values
j = json.parse('{"items": [1, 2, 3]}').unwrap()
j["items"]              # Json array
j["items"][0]           # Json number

# Convert to typed value
type Person = { name: String, age: Int }
json.decode[Person]('{"name": "Alice", "age": 30}')  # Result[Person, String]

# Encode
json.encode(Person("Alice", 30))  # '{"name":"Alice","age":30}'
```

## HTTP

```nostos
import http

# GET request
response = http.get("https://api.example.com/data")  # Result[Response, String]
body = response.unwrap().body                        # String

# POST with JSON
response = http.post("https://api.example.com/users",
    body: '{"name": "Alice"}',
    headers: %{"Content-Type": "application/json"}
)

# Response fields
response.status         # Int (200, 404, etc.)
response.body           # String
response.headers        # Map[String, String]
```

## Concurrency

```nostos
import concurrent

# Spawn task
handle = spawn(() => expensiveComputation())

# Wait for result
result = await(handle)

# Sleep
sleep(1000)             # Milliseconds

# Parallel map
results = [1, 2, 3].parMap(x => compute(x))

# Channels
ch = channel()
spawn(() => ch.send(42))
value = ch.receive()    # 42
```

---

# Nostos End-to-End Examples

Complete working examples for common tasks.

## Read JSON File and Extract Data

```nostos
import file
import json

type User = { id: Int, name: String, email: String }
type UsersFile = { users: List[User] }

main() = {
    # Read and parse JSON file
    content = match file.read("users.json") {
        Ok(s) -> s,
        Err(e) -> {
            println("Failed to read file: " ++ e)
            return ()
        }
    }

    # Parse JSON into typed structure
    data = match json.decode[UsersFile](content) {
        Ok(d) -> d,
        Err(e) -> {
            println("Failed to parse JSON: " ++ e)
            return ()
        }
    }

    # Process the data
    activeUsers = data.users.filter(u => u.email.contains("@"))

    println("Found " ++ show(activeUsers.length()) ++ " users:")
    activeUsers.map(u => println("  - " ++ u.name ++ " <" ++ u.email ++ ">"))
}
```

**Sample users.json:**
```json
{
  "users": [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"}
  ]
}
```

## HTTP Request and Parse Response

```nostos
import http
import json

type Post = { id: Int, title: String, body: String, userId: Int }

# Fetch a single post
fetchPost(id: Int) -> Result[Post, String] = {
    response = http.get("https://jsonplaceholder.typicode.com/posts/" ++ show(id))
    match response {
        Ok(r) -> {
            if r.status == 200 {
                json.decode[Post](r.body)
            } else {
                Err("HTTP " ++ show(r.status))
            }
        },
        Err(e) -> Err(e)
    }
}

# Fetch multiple posts concurrently
fetchPosts(ids: List[Int]) -> List[Result[Post, String]] = {
    ids.parMap(id => fetchPost(id))
}

main() = {
    # Single request
    match fetchPost(1) {
        Ok(post) -> println("Title: " ++ post.title),
        Err(e) -> println("Error: " ++ e)
    }

    # Multiple concurrent requests
    results = fetchPosts([1, 2, 3, 4, 5])
    successes = results.filter(r => r.isOk()).map(r => r.unwrap())
    println("Fetched " ++ show(successes.length()) ++ " posts")
}
```

## POST Request with JSON Body

```nostos
import http
import json

type CreateUser = { name: String, email: String }
type UserResponse = { id: Int, name: String, email: String }

createUser(user: CreateUser) -> Result[UserResponse, String] = {
    body = json.encode(user)
    response = http.post(
        "https://api.example.com/users",
        body: body,
        headers: %{
            "Content-Type": "application/json",
            "Authorization": "Bearer " ++ getApiKey()
        }
    )
    match response {
        Ok(r) -> {
            if r.status >= 200 && r.status < 300 {
                json.decode[UserResponse](r.body)
            } else {
                Err("HTTP " ++ show(r.status) ++ ": " ++ r.body)
            }
        },
        Err(e) -> Err(e)
    }
}

getApiKey() = env("API_KEY").unwrapOr("default-key")

main() = {
    newUser = CreateUser(name: "Alice", email: "alice@example.com")
    match createUser(newUser) {
        Ok(created) -> println("Created user with ID: " ++ show(created.id)),
        Err(e) -> println("Failed: " ++ e)
    }
}
```

## Simple CLI Tool

```nostos
import file
import env

# Word count tool: count lines, words, chars in files

countFile(path: String) -> Result[(Int, Int, Int), String] = {
    match file.read(path) {
        Ok(content) -> {
            lines = content.split("\n").length()
            words = content.split(" ").flatMap(s => s.split("\n")).filter(s => s.length() > 0).length()
            chars = content.length()
            Ok((lines, words, chars))
        },
        Err(e) -> Err(e)
    }
}

printUsage() = {
    println("Usage: wc <file1> [file2] ...")
    println("Count lines, words, and characters in files")
}

main() = {
    args = env.args()  # Get command line arguments

    if args.length() < 2 {
        printUsage()
        return ()
    }

    files = args.drop(1)  # Skip program name

    var totalLines = 0
    var totalWords = 0
    var totalChars = 0

    files.map(path => {
        match countFile(path) {
            Ok((lines, words, chars)) -> {
                println(show(lines) ++ "\t" ++ show(words) ++ "\t" ++ show(chars) ++ "\t" ++ path)
                totalLines = totalLines + lines
                totalWords = totalWords + words
                totalChars = totalChars + chars
            },
            Err(e) -> println("Error reading " ++ path ++ ": " ++ e)
        }
    })

    if files.length() > 1 {
        println(show(totalLines) ++ "\t" ++ show(totalWords) ++ "\t" ++ show(totalChars) ++ "\ttotal")
    } else {
        ()
    }
}
```

## Simple HTTP Server

```nostos
import http.server
import json

type Todo = { id: Int, title: String, done: Bool }

mvar todos: List[Todo] = [
    Todo(1, "Learn Nostos", false),
    Todo(2, "Build something", false)
]

mvar nextId: Int = 3

handleRequest(req: Request) -> Response = {
    match (req.method, req.path) {
        ("GET", "/todos") -> {
            Response(
                status: 200,
                body: json.encode(todos),
                headers: %{"Content-Type": "application/json"}
            )
        },

        ("POST", "/todos") -> {
            match json.decode[{title: String}](req.body) {
                Ok(data) -> {
                    newTodo = Todo(nextId, data.title, false)
                    nextId = nextId + 1
                    todos = todos.push(newTodo)
                    Response(
                        status: 201,
                        body: json.encode(newTodo),
                        headers: %{"Content-Type": "application/json"}
                    )
                },
                Err(e) -> Response(status: 400, body: "Invalid JSON: " ++ e)
            }
        },

        ("DELETE", path) -> {
            if path.startsWith("/todos/") {
                idStr = path.substring(7, path.length())
                match idStr.parseInt() {
                    Some(id) -> {
                        todos = todos.filter(t => t.id != id)
                        Response(status: 204, body: "")
                    },
                    None -> Response(status: 400, body: "Invalid ID")
                }
            } else {
                Response(status: 404, body: "Not found")
            }
        },

        _ -> Response(status: 404, body: "Not found")
    }
}

main() = {
    println("Server running on http://localhost:8080")
    http.server.run(8080, handleRequest)
}
```

## Data Pipeline with Error Handling

```nostos
import file
import json

type RawRecord = { id: String, value: String, timestamp: String }
type ProcessedRecord = { id: Int, value: Float, timestamp: Int }

# Parse a raw record, returning errors for invalid data
parseRecord(raw: RawRecord) -> Result[ProcessedRecord, String] = {
    id = match raw.id.parseInt() {
        Some(n) -> n,
        None -> return Err("Invalid id: " ++ raw.id)
    }

    value = match raw.value.parseFloat() {
        Some(f) -> f,
        None -> return Err("Invalid value: " ++ raw.value)
    }

    timestamp = match raw.timestamp.parseInt() {
        Some(t) -> t,
        None -> return Err("Invalid timestamp: " ++ raw.timestamp)
    }

    Ok(ProcessedRecord(id, value, timestamp))
}

# Process a file, collecting successes and errors
processFile(path: String) -> (List[ProcessedRecord], List[String]) = {
    content = file.read(path).unwrapOr("[]")
    records = json.decode[List[RawRecord]](content).unwrapOr([])

    results = records.map(r => parseRecord(r))

    successes = results.filter(r => r.isOk()).map(r => r.unwrap())
    errors = results.filter(r => r.isErr()).map(r => r.unwrapErr())

    (successes, errors)
}

main() = {
    (records, errors) = processFile("data.json")

    println("Processed " ++ show(records.length()) ++ " records")

    if errors.nonEmpty() {
        println("Errors:")
        errors.map(e => println("  - " ++ e))
    } else {
        ()
    }

    # Calculate statistics
    if records.nonEmpty() {
        values = records.map(r => r.value)
        avg = values.sum() / values.length().toFloat()
        println("Average value: " ++ show(avg))
    } else {
        ()
    }
}
```

## WebSocket Client

```nostos
import websocket
import json

type Message = { type: String, content: String }

main() = {
    ws = websocket.connect("wss://echo.websocket.org")

    # Send a message
    msg = Message(type: "greeting", content: "Hello!")
    ws.send(json.encode(msg))

    # Receive response
    response = ws.receive()
    println("Received: " ++ response)

    # Close connection
    ws.close()
}
```

## Concurrent Task Processing

```nostos
import concurrent

type Task = { id: Int, data: String }
type TaskResult = { id: Int, result: String, success: Bool }

# Simulate processing a task
processTask(task: Task) -> TaskResult = {
    # Simulate work
    sleep(100)

    if task.data.length() > 0 {
        TaskResult(task.id, "Processed: " ++ task.data.toUpper(), true)
    } else {
        TaskResult(task.id, "Empty data", false)
    }
}

main() = {
    tasks = [
        Task(1, "hello"),
        Task(2, "world"),
        Task(3, ""),
        Task(4, "nostos"),
        Task(5, "rocks")
    ]

    # Process all tasks concurrently
    println("Processing " ++ show(tasks.length()) ++ " tasks...")

    results = tasks.parMap(t => processTask(t))

    successes = results.filter(r => r.success)
    failures = results.filter(r => !r.success)

    println("Completed: " ++ show(successes.length()) ++ " succeeded, " ++
            show(failures.length()) ++ " failed")

    successes.map(r => println("  [" ++ show(r.id) ++ "] " ++ r.result))
}
```

## Configuration with Defaults

```nostos
import file
import json
import env

type Config = {
    host: String,
    port: Int,
    debug: Bool,
    maxConnections: Int
}

defaultConfig = Config(
    host: "localhost",
    port: 8080,
    debug: false,
    maxConnections: 100
)

# Load config from file, falling back to defaults
loadConfig(path: String) -> Config = {
    # Try file first
    fileConfig = match file.read(path) {
        Ok(content) -> json.decode[Config](content).ok(),
        Err(_) -> None
    }

    match fileConfig {
        Some(c) -> c,
        None -> {
            # Fall back to environment variables
            Config(
                host: env("HOST").unwrapOr(defaultConfig.host),
                port: env("PORT").flatMap(s => s.parseInt()).unwrapOr(defaultConfig.port),
                debug: env("DEBUG").map(s => s == "true").unwrapOr(defaultConfig.debug),
                maxConnections: env("MAX_CONN").flatMap(s => s.parseInt()).unwrapOr(defaultConfig.maxConnections)
            )
        }
    }
}

main() = {
    config = loadConfig("config.json")

    println("Starting server with config:")
    println("  Host: " ++ config.host)
    println("  Port: " ++ show(config.port))
    println("  Debug: " ++ show(config.debug))
    println("  Max connections: " ++ show(config.maxConnections))
}
```

---

# Nostos Quick Reference Cheat Sheet

One-page syntax and patterns reference.

## Basics

```nostos
# Comments start with hash
x = 42              # Immutable binding
var y = 0           # Mutable local variable
mvar z: Int = 0     # Mutable module-level variable
const PI = 3.14     # Compile-time constant
```

## Types

```nostos
Int Float Bool String Char ()     # Primitives
List[T] Map[K,V] Set[T]           # Collections
Option[T] Result[T,E]             # Common patterns
(A, B, C)                         # Tuples

type Point = { x: Int, y: Int }   # Record
type Color = Red | Green | Blue   # Variants
type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
```

## Literals

```nostos
42  -17  1_000_000  0xFF  0b1010  # Int
3.14  -0.5  1.0e10                # Float
true  false                       # Bool
"hello"  'c'  '\n'                # String, Char
[1, 2, 3]                         # List
%{"a": 1, "b": 2}                 # Map
(1, "two", true)                  # Tuple
```

## Operators

```nostos
+  -  *  /  %                     # Arithmetic
==  !=  <  <=  >  >=              # Comparison
&&  ||  !                         # Logical
++                                # String concat
|>                                # Pipe (x |> f  =  f(x))
```

## Functions

```nostos
# Definition
add(a: Int, b: Int) -> Int = a + b
greet(name) = "Hello, " ++ name   # Types inferred

# Lambdas
x => x * 2
(a, b) => a + b

# Call
add(1, 2)
1.add(2)                          # UFCS

# Pattern matching in definition
factorial(0) = 1
factorial(n) = n * factorial(n - 1)
```

## Control Flow

```nostos
# If expression
if x > 0 { "positive" } else { "non-positive" }
if x > 0 then "positive" else "non-positive"

# Match
match value {
    0 -> "zero",
    n if n > 0 -> "positive",
    _ -> "negative"
}

# Loops
while condition { body }
for item in list { body }
(1..10).map(i => process(i))      # Prefer this
```

## Pattern Matching

```nostos
# Literals
match x { 1 -> "one", 2 -> "two", _ -> "other" }

# Destructuring
(a, b) = tuple
{ name, age } = person
[h | t] = list                    # Head and tail
[a, b, c] = threeList             # Exact length

# Variants
match opt {
    Some(x) -> use(x),
    None -> default
}

# Guards
match n {
    x if x > 0 -> "positive",
    x if x < 0 -> "negative",
    _ -> "zero"
}
```

## Collections

```nostos
# List
[1, 2, 3].map(x => x * 2)         # [2, 4, 6]
[1, 2, 3].filter(x => x > 1)      # [2, 3]
[1, 2, 3].fold(0, (a,b) => a+b)   # 6
list.length()  list[0]  list.get(0)

# Map
map = %{"a": 1, "b": 2}
map["a"]  map.get("a")
map.insert("c", 3)
map.keys()  map.values()

# Set
Set.from([1, 2, 2, 3])            # {1, 2, 3}
set.contains(x)  set.insert(x)
a.union(b)  a.intersection(b)
```

## Option & Result

```nostos
# Option
Some(42)  None
opt.map(f)  opt.flatMap(f)
opt.unwrap()  opt.unwrapOr(default)
opt.isSome()  opt.isNone()

# Result
Ok(value)  Err(error)
res.map(f)  res.mapErr(f)
res.unwrap()  res.unwrapOr(default)
res.isOk()  res.isErr()
```

## Error Handling

```nostos
# Try/catch
try {
    riskyOperation()
} catch {
    "specific" -> handleSpecific(),
    e -> handleGeneric(e)
}

# Throw
throw("error message")

# Early return
if bad { return Err("failed") } else { () }
```

## Modules

```nostos
# math.nos
pub add(a, b) = a + b             # Public
helper(x) = x * 2                 # Private

# main.nos
import math
math.add(1, 2)
```

## Traits

```nostos
trait Show {
    show(self) -> String
}

impl Show for Point {
    show(self) = "(" ++ show(self.x) ++ "," ++ show(self.y) ++ ")"
}
```

## Templates

```nostos
template logged(fn) = quote {
    println(">>> " ++ ~fn.name)
    ~fn.body
}

@logged
compute() = 42

# Key operations
quote { code }                    # Capture AST
~expr                             # Splice AST
eval("code string")               # Parse string as code
gensym("prefix")                  # Unique identifier
~fn.name  ~fn.body  ~fn.params    # Function metadata
~typeDef.name  ~typeDef.fields    # Type metadata
```

## Concurrency

```nostos
import concurrent

handle = spawn(() => computation())
result = await(handle)
sleep(1000)                       # Milliseconds
list.parMap(f)                    # Parallel map
```

## I/O

```nostos
# Print
print("no newline")
println("with newline")
show(42)                          # Any to String

# Files
import file
file.read("path")                 # Result[String, String]
file.write("path", content)       # Result[(), String]

# HTTP
import http
http.get(url)
http.post(url, body: data, headers: %{...})

# JSON
import json
json.parse(str)                   # Result[Json, String]
json.decode[Type](str)            # Result[Type, String]
json.encode(value)                # String
```

## Common Patterns

```nostos
# Pipeline
data
    .filter(x => x.valid)
    .map(x => transform(x))
    .fold(init, combine)

# Safe unwrap with default
value = opt.unwrapOr(default)

# Error propagation
match operation() {
    Ok(x) -> continue(x),
    Err(e) -> return Err(e)
}

# Builder pattern
Config()
    .withHost("localhost")
    .withPort(8080)
    .build()
```

## Comparison to Other Languages

| Nostos | Python | Rust | JavaScript |
|--------|--------|------|------------|
| `#` comment | `#` comment | `//` comment | `//` comment |
| `"a" ++ "b"` | `"a" + "b"` | `format!()` | `"a" + "b"` |
| `[1,2,3]` | `[1,2,3]` | `vec![1,2,3]` | `[1,2,3]` |
| `%{"a":1}` | `{"a":1}` | `HashMap` | `{a:1}` |
| `x => x*2` | `lambda x: x*2` | `\|x\| x*2` | `x => x*2` |
| `List[Int]` | `list[int]` | `Vec<i32>` | N/A |
| `Some(x)` | `x` or `None` | `Some(x)` | `x` or `null` |
| `match {}` | `match` (3.10+) | `match {}` | `switch` |

---

# Nostos Basics

## Comments

```nostos
# This is a single-line comment
# Comments use the hash symbol
```

## Literals

```nostos
# Integers
42
-17
1_000_000       # Underscores for readability
0xFF            # Hexadecimal
0b1010          # Binary

# Floats
3.14
-0.5
1.0e10          # Scientific notation

# Booleans
true
false

# Strings (double or single quotes)
"Hello, World!"
'{"key": "value"}'   # Useful for JSON

# Characters
'a'
'\n'            # Newline
'\t'            # Tab

# Unit (empty value, like void)
()
```

## Variables

```nostos
# Immutable binding (default)
x = 42
name = "Alice"

# Mutable variable (use sparingly)
var counter = 0
counter = counter + 1

# Type annotations (optional, inferred)
x: Int = 42
name: String = "Alice"
```

## Basic Types

| Type | Description | Example |
|------|-------------|---------|
| `Int` | 64-bit integer | `42` |
| `Float` | 64-bit float | `3.14` |
| `Bool` | Boolean | `true`, `false` |
| `String` | UTF-8 string | `"hello"` |
| `Char` | Single character | `'a'` |
| `()` | Unit type | `()` |

## Operators

```nostos
# Arithmetic
1 + 2       # Addition
5 - 3       # Subtraction
4 * 2       # Multiplication
10 / 3      # Integer division
10 % 3      # Modulo

# Comparison
x == y      # Equal
x != y      # Not equal
x < y       # Less than
x <= y      # Less or equal
x > y       # Greater than
x >= y      # Greater or equal

# Logical
a && b      # And
a || b      # Or
!a          # Not

# String concatenation
"Hello" ++ " " ++ "World"   # "Hello World"
```

## Printing

```nostos
print("No newline")
println("With newline")

# Convert to string with show()
println("Value: " ++ show(42))
```

## Hello World

```nostos
# Every program needs a main() function
main() = {
    println("Hello, World!")
}

# Or single expression
main() = println("Hello, World!")
```

## Constants

```nostos
# Module-level constants (evaluated at compile time)
const PI = 3.14159
const MAX_SIZE = 1000
const GREETING = "Hello"

main() = println(GREETING ++ ", PI is " ++ show(PI))
```

## Assertions

```nostos
# assert_eq checks equality, panics on failure
assert_eq(4, 2 + 2)
assert_eq("hello", "hel" ++ "lo")

# Useful in tests
main() = {
    assert_eq(42, 6 * 7)
    println("All assertions passed!")
}
```

---

# Collections in Nostos

## Maps (Dictionaries)

```nostos
# Create a map with %{ }
ages = %{"alice": 30, "bob": 25, "carol": 35}

# Type annotation
scores: Map[String, Int] = %{"math": 95, "english": 87}

# Empty map
empty: Map[String, Int] = %{}
```

## Map Index Syntax

Maps support convenient bracket syntax for get and set:

```nostos
m = %{"name": "Alice", "age": 30}

# Get value (returns value or Unit if not found)
name = m["name"]        # "Alice"
age = m["age"]          # 30

# Set value (updates variable with new map)
m["city"] = "Oslo"      # m now has 3 keys
m["age"] = 31           # Update existing key

# Works with any key type
ids = %{1: "one", 2: "two"}
ids[3] = "three"        # Add new entry
val = ids[1]            # "one"
```

Since maps are immutable, `m["key"] = value` is equivalent to `m = Map.insert(m, "key", value)`.

## Map Operations

```nostos
m = %{"a": 1, "b": 2, "c": 3}

# Get value (returns Option)
m.get("a")              # Some(1)
m.get("z")              # None

# Get with default
m.getOrDefault("z", 0)  # 0

# Insert (returns new map)
m2 = m.insert("d", 4)   # %{"a": 1, "b": 2, "c": 3, "d": 4}

# Remove (returns new map)
m3 = m.remove("a")      # %{"b": 2, "c": 3}

# Update value
m4 = m.update("a", x => x + 10)  # %{"a": 11, "b": 2, "c": 3}

# Check existence
m.contains("a")         # true
m.contains("z")         # false

# Size
m.size()                # 3
m.isEmpty()             # false
```

## Iterating Maps

```nostos
m = %{"a": 1, "b": 2, "c": 3}

# Get keys as list
m.keys()                # ["a", "b", "c"]

# Get values as list
m.values()              # [1, 2, 3]

# Get entries as list of tuples
m.entries()             # [("a", 1), ("b", 2), ("c", 3)]

# Map over values
m.mapValues(v => v * 2) # %{"a": 2, "b": 4, "c": 6}

# Filter entries
m.filter((k, v) => v > 1)  # %{"b": 2, "c": 3}

# Fold
m.fold(0, (acc, k, v) => acc + v)  # 6
```

## Merging Maps

```nostos
m1 = %{"a": 1, "b": 2}
m2 = %{"b": 20, "c": 3}

# Merge (right side wins on conflict)
m1.merge(m2)            # %{"a": 1, "b": 20, "c": 3}

# Merge with custom conflict resolution
m1.mergeWith(m2, (v1, v2) => v1 + v2)  # %{"a": 1, "b": 22, "c": 3}
```

## Sets

```nostos
# Create a set with #{ }
colors = #{"red", "green", "blue"}

# Type annotation
numbers: Set[Int] = #{1, 2, 3, 4, 5}

# Empty set
empty: Set[String] = #{}
```

## Set Operations

```nostos
s = #{1, 2, 3, 4, 5}

# Add element (returns new set)
s2 = s.insert(6)        # #{1, 2, 3, 4, 5, 6}

# Remove element (returns new set)
s3 = s.remove(1)        # #{2, 3, 4, 5}

# Check membership
s.contains(3)           # true
s.contains(10)          # false

# Size
s.size()                # 5
s.isEmpty()             # false

# Convert to list
s.toList()              # [1, 2, 3, 4, 5]
```

## Set Index Syntax

Sets support bracket syntax for membership checking:

```nostos
s = #{1, 2, 3, 4, 5}

# Check membership (returns Bool)
s[3]                    # true
s[10]                   # false

# Use in conditions
if s[3] then {
    println("3 is in the set")
}

# Combine checks
s[1] && !s[100]         # true

# With variables
elem = 3
s[elem]                 # true
```

This is equivalent to `s.contains(elem)` but more concise.

## Set Math Operations

```nostos
a = #{1, 2, 3, 4}
b = #{3, 4, 5, 6}

# Union
a.union(b)              # #{1, 2, 3, 4, 5, 6}

# Intersection
a.intersection(b)       # #{3, 4}

# Difference
a.difference(b)         # #{1, 2}

# Subset check
#{1, 2}.isSubset(a)     # true
a.isSuperset(#{1, 2})   # true
```

## Typed Arrays

```nostos
# Efficient numeric arrays

# Float64Array
floats = Float64Array.new(100)          # 100 zeros
floats = Float64Array.from([1.0, 2.0, 3.0])

# Int64Array
ints = Int64Array.new(100)
ints = Int64Array.from([1, 2, 3, 4, 5])

# Float32Array (for GPU compatibility)
f32 = Float32Array.new(100)
```

## Typed Array Operations

```nostos
arr = Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0])

# Index access
arr[0]                  # 1.0
arr[2]                  # 3.0

# Index assignment (returns new array in immutable mode)
arr[0] = 10.0           # [10.0, 2.0, 3.0, 4.0, 5.0]

# Length
arr.length()            # 5

# Map (returns new typed array)
arr.map(x => x * 2)     # [2.0, 4.0, 6.0, 8.0, 10.0]

# Fold/reduce
arr.fold(0.0, (acc, x) => acc + x)  # 15.0

# Sum (optimized)
arr.sum()               # 15.0

# Slice
arr.slice(1, 4)         # [2.0, 3.0, 4.0]
```

## Buffer Type

```nostos
# Growable byte buffer

buf = Buffer.new()
buf = buf.append("Hello")
buf = buf.append(" World")
buf.toString()          # "Hello World"

# Useful for building strings or binary data
buildCsv(rows: List[List[String]]) -> String = {
    buf = Buffer.new()
    rows.forEach(row => {
        buf = buf.append(row.join(","))
        buf = buf.append("\n")
    })
    buf.toString()
}
```

## Tuples

```nostos
# Fixed-size, mixed-type collection
point = (10, 20)
named = ("Alice", 30, true)

# Destructuring
(x, y) = point
(name, age, active) = named

# Access by pattern matching
getFirst((a, _, _)) = a
getSecond((_, b, _)) = b

# Nested tuples
nested = ((1, 2), (3, 4))
((a, b), (c, d)) = nested
```

## Converting Between Collections

```nostos
# List to Set (removes duplicates)
[1, 2, 2, 3, 3, 3].toSet()      # #{1, 2, 3}

# Set to List
#{1, 2, 3}.toList()             # [1, 2, 3]

# List of pairs to Map
[("a", 1), ("b", 2)].toMap()    # %{"a": 1, "b": 2}

# Map to list of pairs
%{"a": 1, "b": 2}.entries()     # [("a", 1), ("b", 2)]

# List to typed array
[1.0, 2.0, 3.0].toFloat64Array()

# Typed array to list
Float64Array.from([1.0, 2.0]).toList()
```

## Common Patterns

```nostos
# Count occurrences
countWords(words: List[String]) -> Map[String, Int] =
    words.fold(%{}, (acc, word) => {
        count = acc.getOrDefault(word, 0)
        acc.insert(word, count + 1)
    })

# Group by key
groupBy(items: List[T], keyFn: T -> K) -> Map[K, List[T]] =
    items.fold(%{}, (acc, item) => {
        key = keyFn(item)
        existing = acc.getOrDefault(key, [])
        acc.insert(key, existing ++ [item])
    })

# Index lookup table
createIndex(items: List[T], keyFn: T -> K) -> Map[K, T] =
    items.fold(%{}, (acc, item) => acc.insert(keyFn(item), item))

# Deduplicate while preserving order
dedupe(items: List[T]) -> List[T] = {
    (result, _) = items.fold(([], #{}), ((acc, seen), item) => {
        if seen.contains(item) then (acc, seen)
        else (acc ++ [item], seen.insert(item))
    })
    result
}
```

## Performance Considerations

```nostos
# Maps and Sets: O(log n) for most operations
# - Use for lookup-heavy workloads
# - Immutable, returns new collection on modification

# Lists: O(n) for index access, O(1) for head/cons
# - Good for sequential processing
# - Use fold/map instead of index loops

# Typed Arrays: O(1) index access
# - Use for numeric computation
# - More memory efficient than List[Float]
# - Good for interop with FFI

# Choose based on:
# - Access pattern (random vs sequential)
# - Data type (numeric vs mixed)
# - Mutability needs
```

---

# Concurrency in Nostos

## Spawning Processes

```nostos
# spawn creates a lightweight process
pid = spawn(() => {
    println("Hello from spawned process!")
})

# Process runs concurrently with main
println("Main continues immediately")

# Spawn with argument
worker(id: Int) = {
    println("Worker " ++ show(id) ++ " started")
}

spawn(() => worker(1))
spawn(() => worker(2))
```

## Message Passing

```nostos
# Processes communicate via messages
# self() returns current process ID

# Send a message
send(pid, "Hello!")

# Receive a message (blocks until received)
msg = receive()

# Basic ping-pong
main() = {
    parent = self()

    child = spawn(() => {
        msg = receive()
        send(parent, "Got: " ++ msg)
    })

    send(child, "Hello")
    reply = receive()
    println(reply)      # "Got: Hello"
}
```

## Receive with Pattern Matching

```nostos
# Receive with pattern matching
result = receive {
    "ping" -> "pong",
    ("add", a, b) -> show(a + b),
    ("quit") -> {
        println("Shutting down")
        "bye"
    },
    other -> "Unknown: " ++ show(other)
}
```

## Receive with Timeout

```nostos
# Receive with timeout (milliseconds)
result = receiveTimeout(1000) {
    msg -> "Got: " ++ msg
}

# Returns Option - None if timeout
match result {
    Some(value) -> println(value),
    None -> println("Timed out!")
}
```

## MVar (Mutable Variable)

```nostos
# MVar is a synchronized mutable container
# Can be empty or full

# Create empty MVar
mv = MVar.new()

# Create MVar with initial value
mv = MVar.newWith(42)

# Put value (blocks if full)
mv.put(100)

# Take value (blocks if empty)
value = mv.take()

# Read without removing
value = mv.read()

# Try operations (non-blocking, returns Option)
result = mv.tryTake()
result = mv.tryPut(42)
```

## Worker Pool Pattern

```nostos
# Process pool for parallel work

workerLoop(id: Int) = {
    match receive() {
        ("task", data, replyTo) -> {
            result = processData(data)
            send(replyTo, ("result", id, result))
            workerLoop(id)
        },
        "stop" -> ()
    }
}

createWorkers(n: Int) -> List[Pid] = {
    range(1, n + 1).map(id => spawn(() => workerLoop(id)))
}

main() = {
    workers = createWorkers(4)
    me = self()

    # Distribute work
    tasks = [1, 2, 3, 4, 5, 6, 7, 8]
    tasks.zip(cycle(workers)).forEach((task, worker) => {
        send(worker, ("task", task, me))
    })

    # Collect results
    results = tasks.map(_ => receive())
    println(show(results))

    # Stop workers
    workers.forEach(w => send(w, "stop"))
}
```

## Parallel Map

```nostos
# Parallel map over a list
parallelMap(items: List[T], f: T -> R) -> List[R] = {
    me = self()

    # Spawn a process for each item
    pids = items.map(item => spawn(() => {
        result = f(item)
        send(me, result)
    }))

    # Collect results (in order)
    items.map(_ => receive())
}

# Usage
squares = parallelMap([1, 2, 3, 4, 5], x => x * x)
```

## Process Linking

```nostos
# Link processes - if one dies, the other is notified
spawnLink(() => {
    # If this crashes, parent receives exit message
    riskyOperation()
})

# Handle exit messages
result = receive {
    ("EXIT", pid, reason) -> "Process " ++ show(pid) ++ " exited: " ++ reason,
    normalMsg -> handleNormal(normalMsg)
}
```

## Supervisors

```nostos
# Restart failed processes
supervisor(childFn) = {
    child = spawnLink(childFn)

    match receive() {
        ("EXIT", _, _) -> {
            println("Child crashed, restarting...")
            supervisor(childFn)
        },
        msg -> {
            send(child, msg)
            supervisor(childFn)
        }
    }
}
```

## Ring Benchmark

```nostos
# Classic concurrency benchmark: message ring

ringNode(next: Pid) = {
    match receive() {
        0 -> send(next, 0),
        n -> {
            send(next, n - 1)
            ringNode(next)
        }
    }
}

main() = {
    n = 1000    # Ring size
    m = 10000   # Messages

    # Create ring of processes
    first = self()
    last = range(1, n).fold(first, (prev, _) => {
        spawn(() => ringNode(prev))
    })

    # Connect last to first
    send(last, m)

    # Wait for completion
    receive()
    println("Ring complete!")
}
```

## Async Sleep

```nostos
# Sleep without blocking other processes
sleep(1000)     # Sleep 1 second

# Delayed message
sendAfter(pid, msg, delay) = {
    spawn(() => {
        sleep(delay)
        send(pid, msg)
    })
}

# Timer pattern
startTimer(duration: Int, callback: () -> ()) = {
    spawn(() => {
        sleep(duration)
        callback()
    })
}
```

## Channel Pattern

```nostos
# Implement channels using MVars
type Channel[T] = { queue: MVar[List[T]] }

newChannel() -> Channel[T] = Channel(MVar.newWith([]))

channelSend(ch: Channel[T], value: T) = {
    items = ch.queue.take()
    ch.queue.put(items ++ [value])
}

channelReceive(ch: Channel[T]) -> T = {
    items = ch.queue.take()
    match items {
        [h | t] -> {
            ch.queue.put(t)
            h
        },
        [] -> {
            ch.queue.put([])
            # Wait and retry
            sleep(10)
            channelReceive(ch)
        }
    }
}
```

## Best Practices

```nostos
# 1. Prefer message passing over shared state
# Good: send(worker, data)
# Avoid: shared mutable variables

# 2. Use timeouts to prevent deadlocks
receiveTimeout(5000) { ... }

# 3. Handle process failures
spawnLink for supervised processes

# 4. Keep messages small
# Send IDs, not large data structures

# 5. Use MVars for simple synchronization
counter = MVar.newWith(0)
increment() = {
    n = counter.take()
    counter.put(n + 1)
}
```

## See Also

- **error_handling.md** - Handle exceptions in spawned processes with try/catch
- **03_examples.md** - End-to-end concurrent task processing example
- **templates.md** - Use `@retry` template pattern for flaky concurrent operations

---

# Control Flow in Nostos

## If/Then/Else

```nostos
# Expression form (returns a value)
max = if a > b then a else b

# Statement form with blocks
result = if condition then {
    doSomething()
    value1
} else {
    doOther()
    value2
}

# Nested conditionals
grade = if score >= 90 then "A"
        else if score >= 80 then "B"
        else if score >= 70 then "C"
        else "F"

# If without else (returns unit)
if debug then
    println("Debug mode")
else
    ()
```

## Pattern Matching (match)

```nostos
# Basic match
describe(n: Int) = match n {
    0 -> "zero",
    1 -> "one",
    _ -> "other"
}

# Match on variants
type Option[T] = Some(T) | None

getValue(opt: Option[Int]) = match opt {
    Some(x) -> x,
    None -> 0
}

# Match with guards
classify(n: Int) = match n {
    x if x < 0 -> "negative",
    0 -> "zero",
    x if x > 100 -> "large",
    _ -> "positive"
}

# Match on tuples
handlePoint((x, y)) = match (x, y) {
    (0, 0) -> "origin",
    (0, _) -> "on y-axis",
    (_, 0) -> "on x-axis",
    _ -> "elsewhere"
}

# Match on lists
describe(lst) = match lst {
    [] -> "empty",
    [x] -> "single: " ++ show(x),
    [x, y] -> "pair",
    [h | t] -> "head: " ++ show(h) ++ ", tail has " ++ show(t.length())
}

# Match on records
type Person = { name: String, age: Int }

greet(p: Person) = match p {
    { name: "Alice", age } -> "Hi Alice, you're " ++ show(age),
    { name, age } if age < 18 -> "Hello young " ++ name,
    { name, _ } -> "Hello " ++ name
}
```

## While Loops

```nostos
# Basic while loop
main() = {
    var i = 0
    while i < 5 {
        println(show(i))
        i = i + 1
    }
}

# Sum with while
sumTo(n: Int) -> Int = {
    var sum = 0
    var i = 1
    while i <= n {
        sum = sum + i
        i = i + 1
    }
    sum
}

# Early exit with return
findIndex(items: List[Int], target: Int) -> Int = {
    var i = 0
    while i < items.length() {
        if items[i] == target then
            return i
        else
            ()
        i = i + 1
    }
    -1
}
```

## Functional Iteration (Preferred)

```nostos
# Instead of while loops, prefer functional style:

# Map over a list
doubled = [1, 2, 3].map(x => x * 2)

# Filter elements
evens = [1, 2, 3, 4].filter(x => x % 2 == 0)

# Reduce/fold
sum = [1, 2, 3].fold(0, (acc, x) => acc + x)

# Range-based iteration
# Create a range and process
range(1, 5).map(x => x * x)     # [1, 4, 9, 16]
range(1, 10).filter(x => x % 2 == 0)  # [2, 4, 6, 8]

# forEach for side effects
[1, 2, 3].forEach(x => println(show(x)))
```

## Early Return

```nostos
# Return exits the function immediately
checkAge(age: Int) -> String = {
    if age < 0 then
        return "Invalid age"
    else
        ()

    if age < 18 then
        return "Minor"
    else
        ()

    "Adult"
}

# Works in nested contexts
process(items: List[Int]) -> Int = {
    var total = 0
    var i = 0
    while i < items.length() {
        item = items[i]
        if item < 0 then
            return -1  # Error: negative found
        else
            ()
        total = total + item
        i = i + 1
    }
    total
}
```

## Blocks

```nostos
# Blocks are expressions, return last value
result = {
    x = 10
    y = 20
    x + y   # This is returned
}

# Nested blocks
outer = {
    a = {
        temp = 5
        temp * 2
    }
    b = {
        temp = 3
        temp * 3
    }
    a + b   # 10 + 9 = 19
}
```

## Match as Expression

```nostos
# Match returns a value
status = match code {
    200 -> "OK",
    404 -> "Not Found",
    500 -> "Server Error",
    _ -> "Unknown"
}

# Used inline
println("Status: " ++ match code { 200 -> "OK", _ -> "Error" })
```

## Combining Control Flow

```nostos
processItems(items: List[Int]) -> String = {
    if items.length() == 0 then
        return "Empty list"
    else
        ()

    var result = ""
    var i = 0
    while i < items.length() {
        item = items[i]
        category = match item {
            x if x < 0 -> "negative",
            0 -> "zero",
            x if x > 100 -> "large",
            _ -> "normal"
        }
        result = result ++ category ++ " "
        i = i + 1
    }
    result
}
```

---

# Database Operations in Nostos

## PostgreSQL Connection

```nostos
# Connect to PostgreSQL
conn = Pg.connect("host=localhost dbname=mydb user=postgres password=secret")

# Or with more options
conn = Pg.connect("host=localhost port=5432 dbname=mydb user=postgres password=secret sslmode=prefer")

# Close connection when done
Pg.close(conn)
```

## Basic Queries

```nostos
# Query returns list of tuples
rows = Pg.query(conn, "SELECT name, email FROM users", [])

# Access columns positionally
rows.map(row => println(row.0 ++ ": " ++ row.1))

# Parameterized queries (prevent SQL injection)
rows = Pg.query(conn, "SELECT * FROM users WHERE active = $1 AND age > $2", [true, 18])

# Single row access
firstRow = head(rows)
name = firstRow.0
email = firstRow.1
```

## Execute (Non-Query Operations)

```nostos
# INSERT, UPDATE, DELETE don't return rows
Pg.execute(conn, "INSERT INTO users (name, email) VALUES ($1, $2)", ["Alice", "alice@example.com"])

Pg.execute(conn, "UPDATE users SET active = $1 WHERE id = $2", [true, 42])

Pg.execute(conn, "DELETE FROM users WHERE id = $1", [42])
```

## Typed Results with Introspection

Map query results to typed records using the `stdlib.db` module:

```nostos
use stdlib.db.{rowsToRecords, rowToRecord, queryAs}

type User = { id: Int, name: String, email: String }

main() = {
    conn = Pg.connect("host=localhost user=postgres password=postgres")

    # Query and map to typed records
    rows = Pg.query(conn, "SELECT id, name, email FROM users", [])
    users: List[User] = rowsToRecords("User", rows)

    # Now use field names instead of positional access
    users.map(u => println(u.name ++ " <" ++ u.email ++ ">"))

    # Filter by field
    active = users.filter(u => u.id > 10)

    # Map by field
    emails = users.map(u => u.email)

    Pg.close(conn)
}
```

**Important:** Column order in SELECT must match field order in the type definition.

## Type Conversions

PostgreSQL types map to Nostos types:

| PostgreSQL | Nostos |
|------------|--------|
| INTEGER, BIGINT | Int |
| REAL, DOUBLE | Float |
| TEXT, VARCHAR | String |
| BOOLEAN | Bool |
| JSON, JSONB | String (parse with jsonParse) |

## Transactions

```nostos
# Begin transaction
Pg.execute(conn, "BEGIN", [])

# Do work
Pg.execute(conn, "INSERT INTO orders (user_id, amount) VALUES ($1, $2)", [1, 100])
Pg.execute(conn, "UPDATE users SET balance = balance - $1 WHERE id = $2", [100, 1])

# Commit (or ROLLBACK on error)
Pg.execute(conn, "COMMIT", [])
```

## Connection Pooling

For production apps, use a connection pool:

```nostos
use stdlib.pool.*

# Create pool with max 10 connections
pool = Pool.create(10, () => Pg.connect("host=localhost user=postgres password=postgres"))

# Get connection from pool
conn = pool.acquire()

# Use connection
rows = Pg.query(conn, "SELECT * FROM users", [])

# Return to pool
pool.release(conn)
```

## Error Handling

```nostos
main() = {
    result = try {
        conn = Pg.connect("host=localhost user=postgres password=wrong")
        Pg.query(conn, "SELECT * FROM users", [])
    } catch e {
        println("Database error: " ++ e)
        []  # Return empty list on error
    }
    result
}
```

## Prepared Statements

For repeated queries, prepare once and execute many:

```nostos
# Prepare statement
Pg.execute(conn, "PREPARE get_user AS SELECT * FROM users WHERE id = $1", [])

# Execute prepared statement multiple times
user1 = Pg.query(conn, "EXECUTE get_user(1)", [])
user2 = Pg.query(conn, "EXECUTE get_user(2)", [])
user3 = Pg.query(conn, "EXECUTE get_user(3)", [])

# Deallocate when done
Pg.execute(conn, "DEALLOCATE get_user", [])
```

## Listen/Notify (Pub/Sub)

```nostos
# In subscriber process
conn1 = Pg.connect("host=localhost user=postgres password=postgres")
Pg.execute(conn1, "LISTEN my_channel", [])

# Wait for notification (blocking)
notification = Pg.waitForNotification(conn1)
println("Got: " ++ notification)

# In publisher process
conn2 = Pg.connect("host=localhost user=postgres password=postgres")
Pg.execute(conn2, "NOTIFY my_channel, 'hello'", [])
```

---

# Error Handling in Nostos

## Option Type

```nostos
type Option[T] = Some(T) | None

# Representing missing values
findUser(id: Int) -> Option[User] = {
    if id == 1 then Some(User("Alice"))
    else None
}

# Pattern matching
result = match findUser(1) {
    Some(user) -> "Found: " ++ user.name,
    None -> "Not found"
}
```

## Option Methods

```nostos
opt = Some(42)
none: Option[Int] = None

# Map - transform the value if present
opt.map(x => x * 2)         # Some(84)
none.map(x => x * 2)        # None

# getOrElse - extract with default
opt.getOrElse(0)            # 42
none.getOrElse(0)           # 0

# isSome / isNone
opt.isSome()                # true
none.isNone()               # true

# flatMap - for chained optionals
opt.flatMap(x => if x > 0 then Some(x) else None)
```

## Result Type

```nostos
type Result[T, E] = Ok(T) | Err(E)

# Representing success or failure
parseNumber(s: String) -> Result[Int, String] = {
    if s.all(c => c.isDigit()) then
        Ok(s.parseInt())
    else
        Err("Invalid number: " ++ s)
}

# Pattern matching
result = match parseNumber("42") {
    Ok(n) -> "Parsed: " ++ show(n),
    Err(e) -> "Error: " ++ e
}
```

## Result Methods

```nostos
ok: Result[Int, String] = Ok(42)
err: Result[Int, String] = Err("failed")

# Map - transform success value
ok.map(x => x * 2)          # Ok(84)
err.map(x => x * 2)         # Err("failed")

# mapErr - transform error value
ok.mapErr(e => "Error: " ++ e)   # Ok(42)
err.mapErr(e => "Error: " ++ e)  # Err("Error: failed")

# getOrElse
ok.getOrElse(0)             # 42
err.getOrElse(0)            # 0

# isOk / isErr
ok.isOk()                   # true
err.isErr()                 # true
```

## Chaining Results

```nostos
# Sequential operations that may fail
processData(input: String) -> Result[Int, String] = {
    match parseNumber(input) {
        Ok(n) -> {
            if n > 0 then Ok(n * 2)
            else Err("Number must be positive")
        },
        Err(e) -> Err(e)
    }
}

# Using flatMap for cleaner chaining
process(input: String) -> Result[Int, String] =
    parseNumber(input)
        .flatMap(n => if n > 0 then Ok(n) else Err("Must be positive"))
        .map(n => n * 2)
```

## Try/Catch for Exceptions

```nostos
# Some operations can throw exceptions
# Use try/catch to handle them

result = try {
    riskyOperation()
} catch e {
    "Error occurred: " ++ e
}

# Catch specific patterns
result = try {
    parseAndProcess(input)
} catch {
    "parse error" -> "Invalid input format",
    "not found" -> "Resource not found",
    e -> "Unknown error: " ++ e
}
```

## Throwing Exceptions

```nostos
# Throw an exception
divide(a: Int, b: Int) -> Int = {
    if b == 0 then
        throw "Division by zero"
    else
        a / b
}

# Will be caught by try/catch
result = try {
    divide(10, 0)
} catch e {
    0  # Default value on error
}
```

## Converting Between Types

```nostos
# Option to Result
optToResult(opt: Option[T], err: E) -> Result[T, E] = match opt {
    Some(x) -> Ok(x),
    None -> Err(err)
}

# Result to Option (loses error info)
resultToOpt(res: Result[T, E]) -> Option[T] = match res {
    Ok(x) -> Some(x),
    Err(_) -> None
}

# Example
findUser(id).optToResult("User not found")
```

## Early Return with Pattern Matching

```nostos
# Extract or return early
processUser(id: Int) -> Result[String, String] = {
    user = match findUser(id) {
        Some(u) -> u,
        None -> return Err("User not found")
    }

    profile = match getProfile(user) {
        Some(p) -> p,
        None -> return Err("Profile not found")
    }

    Ok(profile.summary)
}
```

## Collecting Results

```nostos
# Process list, collect all errors or all successes
processAll(items: List[String]) -> Result[List[Int], List[String]] = {
    results = items.map(parseNumber)
    errors = results.filter(r => r.isErr()).map(r => match r { Err(e) -> e, _ -> "" })

    if errors.isEmpty() then
        Ok(results.map(r => match r { Ok(x) -> x, _ -> 0 }))
    else
        Err(errors)
}
```

## Best Practices

```nostos
# 1. Prefer Result over exceptions for expected errors
parseConfig(path: String) -> Result[Config, String]  # Good
# vs throwing exceptions                              # Avoid

# 2. Use Option for "might not exist"
findById(id: Int) -> Option[User]

# 3. Use Result for "might fail with reason"
saveUser(user: User) -> Result[(), String]

# 4. Provide context in errors
Err("Failed to parse config at line " ++ show(line) ++ ": " ++ reason)

# 5. Use early return for cleaner error handling
processRequest(req: Request) -> Result[Response, Error] = {
    user = match authenticate(req) {
        Ok(u) -> u,
        Err(e) -> return Err(AuthError(e))
    }

    data = match fetchData(user) {
        Ok(d) -> d,
        Err(e) -> return Err(DataError(e))
    }

    Ok(Response(data))
}
```

## Custom Error Types

```nostos
# Define specific error variants
type AppError =
    | NotFound(String)
    | InvalidInput(String)
    | NetworkError(String)
    | DatabaseError(String)

# Use in Result
fetchUser(id: Int) -> Result[User, AppError] = {
    if id < 0 then
        Err(InvalidInput("ID must be positive"))
    else if id > 1000 then
        Err(NotFound("User " ++ show(id)))
    else
        Ok(User("User" ++ show(id)))
}

# Handle specific errors
match fetchUser(id) {
    Ok(user) -> handleUser(user),
    Err(NotFound(msg)) -> show404(msg),
    Err(InvalidInput(msg)) -> show400(msg),
    Err(e) -> show500(show(e))
}
```

## See Also

- **types.md** - Defining custom error types with variants
- **concurrency.md** - Error handling in spawned processes, process linking for crash handling
- **templates.md** - `@withFallback` and `@retry` patterns for automatic error recovery
- **02_stdlib_reference.md** - Full Option and Result method reference

---

# File I/O in Nostos

## Reading Files

```nostos
# Read entire file as string
content = File.readAll("config.txt")
println(content)

# Read file as lines
lines = File.readLines("data.txt")
lines.each(line => println(line))

# Read with error handling
content = try {
    File.readAll("maybe-missing.txt")
} catch e {
    println("File error: " ++ e)
    ""  # Default empty string
}
```

## Writing Files

```nostos
# Write string to file (creates or overwrites)
File.writeAll("output.txt", "Hello, World!")

# Write lines
lines = ["Line 1", "Line 2", "Line 3"]
File.writeAll("output.txt", lines.join("\n"))

# Append to file
File.append("log.txt", "New log entry\n")
```

## File Handles (Streaming)

```nostos
# Open file for reading
handle = File.open("large.txt", "r")

# Read line by line
line = File.readLine(handle)
while line != "" {
    processLine(line)
    line = File.readLine(handle)
}

# Close when done
File.close(handle)

# Open modes: "r" (read), "w" (write), "a" (append)
```

## Safe File Operations

```nostos
# Read with automatic cleanup
withFile(path: String, mode: String, fn: Handle -> T) -> T = {
    handle = File.open(path, mode)
    try {
        fn(handle)
    } finally {
        File.close(handle)
    }
}

# Usage
result = withFile("data.txt", "r", handle => {
    lines = []
    line = File.readLine(handle)
    while line != "" {
        lines = lines ++ [line]
        line = File.readLine(handle)
    }
    lines
})
```

## File Existence and Info

```nostos
# Check if file exists
if File.exists("config.json") then {
    config = File.readAll("config.json")
} else {
    config = "{}"
}

# Get file size
size = File.size("data.bin")
println("File size: " ++ show(size) ++ " bytes")
```

## Directory Operations

```nostos
# List files in directory
files = File.listDir("./data")
files.each(f => println(f))

# Create directory
File.mkdir("./output")

# Check if path is directory
if File.isDir("./data") then {
    processDirectory("./data")
}
```

## Path Operations

```nostos
# Join paths
fullPath = Path.join("data", "users", "alice.json")
# "data/users/alice.json"

# Get filename from path
name = Path.filename("/home/user/doc.txt")  # "doc.txt"

# Get directory from path
dir = Path.dirname("/home/user/doc.txt")    # "/home/user"

# Get file extension
ext = Path.extension("photo.jpg")           # "jpg"
```

## Binary Files

```nostos
# Read binary data
bytes = File.readBytes("image.png")

# Write binary data
File.writeBytes("output.bin", bytes)

# Work with typed arrays for efficiency
data = Float64Array.from([1.0, 2.0, 3.0])
File.writeBytes("floats.bin", data.toBytes())
```

## CSV Processing

```nostos
# Read CSV
parseCsv(content: String) -> List[List[String]] = {
    content.split("\n")
        .filter(line => line != "")
        .map(line => line.split(","))
}

csv = File.readAll("data.csv")
rows = parseCsv(csv)

# With header
header = rows[0]
data = rows.tail()

# Access by column name
getColumn(rows, name: String) = {
    idx = header.indexOf(name)
    data.map(row => row[idx])
}

names = getColumn(rows, "name")
```

## JSON Configuration Files

```nostos
type Config = {
    host: String,
    port: Int,
    debug: Bool
}

loadConfig(path: String) -> Config = {
    content = File.readAll(path)
    fromJson[Config](content)
}

saveConfig(path: String, config: Config) = {
    content = json.stringifyPretty(toJson(config))
    File.writeAll(path, content)
}

# Usage
config = loadConfig("config.json")
config.port = 9000
saveConfig("config.json", config)
```

## Log File Pattern

```nostos
mvar logHandle: Option[Handle] = None

initLog(path: String) = {
    logHandle = Some(File.open(path, "a"))
}

log(level: String, message: String) = {
    match logHandle {
        Some(handle) -> {
            timestamp = formatTime(currentTimeMillis())
            line = timestamp ++ " [" ++ level ++ "] " ++ message ++ "\n"
            File.write(handle, line)
        }
        None -> ()
    }
}

closeLog() = {
    match logHandle {
        Some(handle) -> {
            File.close(handle)
            logHandle = None
        }
        None -> ()
    }
}

# Usage
initLog("app.log")
log("INFO", "Application started")
log("ERROR", "Something went wrong")
closeLog()
```

## Temporary Files

```nostos
# Create temp file
tempPath = File.tempFile("prefix", ".txt")
File.writeAll(tempPath, "temporary data")

# Use temp file
data = processFile(tempPath)

# Clean up
File.delete(tempPath)
```

## File Watching Pattern

```nostos
watchFile(path: String, onChange: String -> ()) = {
    lastContent = File.readAll(path)

    loop() = {
        sleep(1000)  # Check every second
        currentContent = File.readAll(path)
        if currentContent != lastContent then {
            onChange(currentContent)
            lastContent = currentContent
        }
        loop()
    }
    spawn(loop)
}

# Usage
watchFile("config.json", newContent => {
    println("Config changed!")
    reloadConfig(newContent)
})
```

## Parallel File Processing

```nostos
# Process multiple files in parallel
processFiles(paths: List[String]) -> List[Result] = {
    me = self()

    paths.each((path, i) => spawn(() => {
        result = try {
            content = File.readAll(path)
            Ok(processContent(content))
        } catch e {
            Err(e)
        }
        send(me, (i, result))
    }))

    # Collect in order
    results = paths.map(_ => receive())
    results.sortBy(r => r.0).map(r => r.1)
}
```

## Error Recovery

```nostos
# Retry on transient errors
readWithRetry(path: String, maxRetries: Int) -> Option[String] = {
    attempt(retries) = {
        if retries <= 0 then None
        else {
            try {
                Some(File.readAll(path))
            } catch e {
                println("Read failed: " ++ e ++ ", retrying...")
                sleep(100)
                attempt(retries - 1)
            }
        }
    }
    attempt(maxRetries)
}
```

---

# Functions in Nostos

## Basic Function Definition

```nostos
# Single expression (no braces needed)
add(a: Int, b: Int) -> Int = a + b

# Multi-statement with block
greet(name: String) -> String = {
    prefix = "Hello, "
    prefix ++ name ++ "!"
}

# Type inference (return type can be omitted)
double(x) = x * 2

# Unit return (for side effects)
logMessage(msg: String) = println(msg)
```

## Calling Functions

```nostos
result = add(2, 3)          # 5
message = greet("Alice")    # "Hello, Alice!"

# UFCS (Uniform Function Call Syntax)
# First argument can be receiver
5.double()                  # Same as double(5)
"Alice".greet()             # Same as greet("Alice")
```

## Named and Default Parameters

```nostos
# Named parameters
connect(host: String, port: Int, timeout: Int) =
    println("Connecting to " ++ host ++ ":" ++ show(port))

# Call with named arguments
connect(host: "localhost", port: 8080, timeout: 30)

# Default values
greet(name: String, greeting: String = "Hello") =
    greeting ++ ", " ++ name ++ "!"

greet("Alice")              # "Hello, Alice!"
greet("Alice", "Hi")        # "Hi, Alice!"
```

## Closures (Anonymous Functions)

```nostos
# Lambda syntax
double = x => x * 2
add = (a, b) => a + b

# Multi-statement closure
process = x => {
    y = x * 2
    y + 1
}

# Used with higher-order functions
[1, 2, 3].map(x => x * 2)           # [2, 4, 6]
[1, 2, 3].filter(x => x > 1)        # [2, 3]
[1, 2, 3].fold(0, (acc, x) => acc + x)  # 6
```

## Closures Capturing Variables

```nostos
makeCounter() = {
    var count = 0
    () => {
        count = count + 1
        count
    }
}

counter = makeCounter()
counter()   # 1
counter()   # 2
counter()   # 3
```

## Recursion

```nostos
# Simple recursion
factorial(0) = 1
factorial(n) = n * factorial(n - 1)

# Tail recursion (optimized)
factorialTail(n) = go(n, 1)
go(0, acc) = acc
go(n, acc) = go(n - 1, n * acc)

# Recursive list processing
sum([]) = 0
sum([h | t]) = h + sum(t)
```

## Generic Functions

```nostos
# Type parameter in brackets
identity[T](x: T) -> T = x

# Multiple type parameters
pair[A, B](a: A, b: B) -> (A, B) = (a, b)

# With trait bounds
printAll[T: Show](items: List[T]) =
    items.map(x => x.show()).join(", ")
```

## Early Return

```nostos
findFirst(items: List[Int], target: Int) -> Int = {
    var i = 0
    while i < items.length() {
        if items[i] == target then
            return i
        else
            ()
        i = i + 1
    }
    -1  # Not found
}
```

## Function Composition

```nostos
# Compose functions
double(x) = x * 2
addOne(x) = x + 1

# Manual composition
composed(x) = addOne(double(x))

# Using pipe-style with UFCS
result = 5.double().addOne()    # 11

# Method chaining
[1, 2, 3]
    .map(x => x * 2)
    .filter(x => x > 2)
    .fold(0, (a, b) => a + b)
```

## Higher-Order Functions

```nostos
# Function taking a function
applyTwice(f, x) = f(f(x))

applyTwice(x => x * 2, 3)   # 12

# Function returning a function
multiplier(n) = x => x * n

triple = multiplier(3)
triple(4)   # 12

# Common patterns
[1, 2, 3].map(x => x * 2)           # Transform each
[1, 2, 3].filter(x => x > 1)        # Keep matching
[1, 2, 3].fold(0, (a, b) => a + b)  # Reduce to one
[1, 2, 3].any(x => x > 2)           # true if any match
[1, 2, 3].all(x => x > 0)           # true if all match
[1, 2, 3].find(x => x > 1)          # Some(2)
```

## See Also

- **templates.md** - Transform functions at compile time with decorators (`@logged`, `@retry`)
- **traits.md** - Generic functions with trait bounds (`[T: Show]`)
- **02_stdlib_reference.md** - Full list of collection methods (map, filter, fold, etc.)
- **03_examples.md** - Functions in real-world contexts

---

# HTTP in Nostos

## HTTP Client

```nostos
# Simple GET request
response = Http.get("https://api.example.com/users")
println(response.body)

# With headers
response = Http.get("https://api.example.com/users", %{
    "Authorization": "Bearer token123",
    "Accept": "application/json"
})

# POST with JSON body
response = Http.post("https://api.example.com/users",
    json.stringify(%{"name": "Alice", "email": "alice@example.com"}),
    %{"Content-Type": "application/json"}
)

# Other HTTP methods
Http.put(url, body, headers)
Http.patch(url, body, headers)
Http.delete(url, headers)
```

## Response Handling

```nostos
response = Http.get("https://api.example.com/data")

# Check status
if response.status == 200 then {
    data = json.parse(response.body)
    processData(data)
} else {
    println("Error: " ++ show(response.status))
}

# Response fields
response.status      # Int: HTTP status code
response.body        # String: response body
response.headers     # Map[String, String]: response headers
```

## HTTP Server

```nostos
use stdlib.server.*

# Basic server
handler(req) = {
    match req.path {
        "/" -> respondText(req, "Hello, World!")
        "/health" -> respondText(req, "OK")
        _ -> respond404(req)
    }
}

main() = serve(8080, handler)
```

## Request Object

```nostos
handler(req) = {
    # Request fields
    req.path        # String: URL path
    req.method      # String: GET, POST, etc.
    req.body        # String: request body
    req.headers     # Map[String, String]: request headers
    req.query       # Map[String, String]: query parameters
    req.id          # Int: unique request ID for responding
}
```

## Response Helpers

```nostos
use stdlib.server.*

handler(req) = {
    # Text response
    respondText(req, "Hello")

    # JSON response
    respondJson(req, %{"status": "ok", "count": 42})

    # HTML response
    respondHtml(req, "<h1>Hello</h1>")

    # Custom status and headers
    respond(req, 201, %{"X-Custom": "value"}, "Created")

    # Error responses
    respond404(req)
    respond500(req, "Internal error")
}
```

## Route Matching

```nostos
use stdlib.server.*

handler(req) = match req.path {
    "/" -> respondText(req, "Home")
    "/api/users" -> handleUsers(req)
    "/api/posts" -> handlePosts(req)
    path when path.startsWith("/static/") -> serveStatic(req)
    _ -> respond404(req)
}

# With path parameters (manual parsing)
handleUserById(req) = {
    # /users/123 -> extract "123"
    parts = req.path.split("/")
    userId = parts[2].toInt()
    user = findUser(userId)
    respondJson(req, user)
}
```

## Query Parameters

```nostos
# URL: /search?q=nostos&limit=10
handler(req) = {
    query = req.query.get("q").getOrElse("")
    limit = req.query.get("limit").map(s => s.toInt()).getOrElse(20)

    results = search(query, limit)
    respondJson(req, results)
}
```

## JSON API Pattern

```nostos
use stdlib.server.*

type User = { id: Int, name: String, email: String }

# GET /users
getUsers(req) = {
    users = fetchAllUsers()
    respondJson(req, users)
}

# POST /users
createUser(req) = {
    data = json.parse(req.body)
    user = User(
        id: generateId(),
        name: data["name"],
        email: data["email"]
    )
    saveUser(user)
    respond(req, 201, %{}, json.stringify(user))
}

# Router
handler(req) = match (req.method, req.path) {
    ("GET", "/users") -> getUsers(req)
    ("POST", "/users") -> createUser(req)
    _ -> respond404(req)
}
```

## Middleware Pattern

```nostos
# Logging middleware
withLogging(handler) = req => {
    println(req.method ++ " " ++ req.path)
    start = currentTimeMillis()
    result = handler(req)
    elapsed = currentTimeMillis() - start
    println("Completed in " ++ show(elapsed) ++ "ms")
    result
}

# Auth middleware
withAuth(handler) = req => {
    token = req.headers.get("Authorization")
    match token {
        Some(t) when isValidToken(t) -> handler(req)
        _ -> respond(req, 401, %{}, "Unauthorized")
    }
}

# Compose middlewares
main() = {
    handler = withLogging(withAuth(apiHandler))
    serve(8080, handler)
}
```

## Error Handling

```nostos
handler(req) = {
    try {
        data = processRequest(req)
        respondJson(req, data)
    } catch {
        "not found" -> respond404(req)
        "unauthorized" -> respond(req, 401, %{}, "Unauthorized")
        e -> respond500(req, "Error: " ++ e)
    }
}
```

## Concurrent Requests

```nostos
# Fetch multiple URLs in parallel
fetchAll(urls: List[String]) -> List[String] = {
    me = self()

    # Spawn request for each URL
    urls.each((url, i) => spawn(() => {
        response = Http.get(url)
        send(me, (i, response.body))
    }))

    # Collect results in order
    results = urls.map(_ => receive())
    results.sortBy(r => r.0).map(r => r.1)
}
```

## Graceful Shutdown

```nostos
mvar running: Bool = true

handler(req) = match req.path {
    "/shutdown" -> {
        running = false
        respondText(req, "Shutting down...")
    }
    _ -> respondText(req, "Hello")
}

serverLoop(handle) = {
    if running then {
        req = Server.accept(handle)
        spawn(() => handler(req))
        serverLoop(handle)
    } else {
        Server.close(handle)
    }
}

main() = {
    handle = Server.bind(8080)
    println("Server running on :8080")
    serverLoop(handle)
}
```

## CORS Headers

```nostos
withCors(handler) = req => {
    corsHeaders = %{
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }

    if req.method == "OPTIONS" then
        respond(req, 204, corsHeaders, "")
    else {
        response = handler(req)
        # Add CORS headers to response
        response
    }
}
```

---

# JSON in Nostos

## Parsing JSON

```nostos
# Parse JSON string to dynamic value
data = json.parse('{"name": "Alice", "age": 30}')

# Access fields with bracket syntax
name = data["name"]     # "Alice"
age = data["age"]       # 30

# Nested access
config = json.parse('{"server": {"host": "localhost", "port": 8080}}')
host = config["server"]["host"]  # "localhost"

# Arrays
items = json.parse('[1, 2, 3, 4, 5]')
first = items[0]        # 1
```

## Generating JSON

```nostos
# Convert map to JSON string
data = %{"name": "Bob", "active": true, "scores": [95, 87, 92]}
jsonStr = json.stringify(data)
# '{"name":"Bob","active":true,"scores":[95,87,92]}'

# Pretty print
prettyJson = json.stringifyPretty(data)
```

## Type-Safe JSON with Records

```nostos
use stdlib.json.*

type User = { name: String, email: String, age: Int }

# Parse JSON directly to typed record
jsonStr = '{"name": "Alice", "email": "alice@example.com", "age": 30}'
user: User = fromJson[User](jsonStr)

println(user.name)   # "Alice"
println(user.age)    # 30

# Convert record to JSON
userJson = toJson(user)
```

## Handling Variants

```nostos
type Status = Active | Inactive(String) | Pending { since: Int }

# Variant serialization
statusJson = toJson(Active)           # {"variant": "Active"}
statusJson = toJson(Inactive("left")) # {"variant": "Inactive", "value": "left"}
statusJson = toJson(Pending { since: 1234 })
# {"variant": "Pending", "fields": {"since": 1234}}

# Parse back to variant
status: Status = fromJson[Status]('{"variant": "Active"}')
```

## Nested Records

```nostos
type Address = { street: String, city: String }
type Person = { name: String, address: Address }

jsonStr = '''
{
    "name": "Alice",
    "address": {
        "street": "123 Main St",
        "city": "Oslo"
    }
}
'''

person: Person = fromJson[Person](jsonStr)
println(person.address.city)  # "Oslo"
```

## Lists of Records

```nostos
type Product = { id: Int, name: String, price: Float }

jsonStr = '''
[
    {"id": 1, "name": "Widget", "price": 9.99},
    {"id": 2, "name": "Gadget", "price": 19.99}
]
'''

products: List[Product] = fromJson[List[Product]](jsonStr)
products.each(p => println(p.name ++ ": $" ++ show(p.price)))
```

## Optional Fields

```nostos
type Config = {
    host: String,
    port: Int,
    debug: Option[Bool]  # Optional field
}

# JSON without optional field
config1: Config = fromJson[Config]('{"host": "localhost", "port": 8080}')
config1.debug  # None

# JSON with optional field
config2: Config = fromJson[Config]('{"host": "localhost", "port": 8080, "debug": true}')
config2.debug  # Some(true)
```

## Error Handling

```nostos
# Safe parsing
result = try {
    data = json.parse(inputStr)
    Some(data)
} catch e {
    println("Parse error: " ++ e)
    None
}

# Type-safe parsing with Result
parseUser(jsonStr: String) -> Result[User, String] = {
    try {
        Ok(fromJson[User](jsonStr))
    } catch e {
        Err("Failed to parse user: " ++ e)
    }
}
```

## Dynamic JSON Access

```nostos
# Check field existence
data = json.parse('{"name": "Alice"}')
hasEmail = data.contains("email")  # false

# Get with default
email = data.get("email").getOrElse("no-email@example.com")

# Iterate over object fields
data = json.parse('{"a": 1, "b": 2, "c": 3}')
data.keys().each(key => println(key ++ ": " ++ show(data[key])))
```

## JSON Path-like Access

```nostos
# Deep access helper
getPath(data, path: List[String]) = {
    path.fold(data, (current, key) => current[key])
}

config = json.parse('''
{
    "database": {
        "connection": {
            "host": "localhost"
        }
    }
}
''')

host = getPath(config, ["database", "connection", "host"])
# "localhost"
```

## Building JSON Dynamically

```nostos
# Start with empty map
builder = %{}

# Add fields
builder["name"] = "Alice"
builder["age"] = 30
builder["tags"] = ["developer", "nostos"]

# Convert to JSON string
json.stringify(builder)
```

## JSON API Response Pattern

```nostos
type ApiResponse[T] = {
    success: Bool,
    data: Option[T],
    error: Option[String]
}

successResponse(data) = ApiResponse {
    success: true,
    data: Some(data),
    error: None
}

errorResponse(msg: String) = ApiResponse {
    success: false,
    data: None,
    error: Some(msg)
}

# In handler
handler(req) = {
    result = try {
        data = processRequest(req)
        successResponse(data)
    } catch e {
        errorResponse(e)
    }
    respondJson(req, toJson(result))
}
```

## JSON Transformation

```nostos
# Transform JSON structure
transformUser(data) = %{
    "fullName": data["firstName"] ++ " " ++ data["lastName"],
    "contact": %{
        "email": data["email"],
        "phone": data["phone"]
    }
}

input = json.parse('{"firstName": "Alice", "lastName": "Smith", "email": "a@b.com", "phone": "123"}')
output = transformUser(input)
json.stringify(output)
```

## Round-Trip Testing

```nostos
# Verify JSON encode/decode preserves data
testRoundTrip(value: T) -> Bool = {
    encoded = toJson(value)
    decoded: T = fromJson[T](encoded)
    decoded == value
}

# Test
user = User { name: "Alice", email: "a@b.com", age: 30 }
assert(testRoundTrip(user))
```

## JSON with Reflection

```nostos
# Get type info at runtime
info = typeInfo("User")
fields = info["fields"]  # List of field definitions

# Dynamic record construction from JSON
buildFromJson(typeName: String, jsonStr: String) = {
    data = json.parse(jsonStr)
    makeRecordByName(typeName, data)
}
```

---

# Lists in Nostos

## Creating Lists

```nostos
# Empty list
empty = []

# List with elements
numbers = [1, 2, 3, 4, 5]
strings = ["a", "b", "c"]
mixed = [1, "two", true]   # Heterogeneous (not recommended)

# Type annotation
nums: List[Int] = [1, 2, 3]
```

## Cons Operator (Prepend)

```nostos
# | prepends element to list
list = [1 | [2, 3]]         # [1, 2, 3]
list = [0 | [1, 2, 3]]      # [0, 1, 2, 3]

# Building lists
addFront(x, lst) = [x | lst]
addFront(0, [1, 2])         # [0, 1, 2]
```

## Pattern Matching on Lists

```nostos
# Match empty vs non-empty
describe([]) = "empty"
describe([h | t]) = "head: " ++ show(h)

# Match specific lengths
handleList([]) = "empty"
handleList([x]) = "single: " ++ show(x)
handleList([x, y]) = "pair: " ++ show(x) ++ ", " ++ show(y)
handleList(_) = "many elements"

# Extract head and tail
firstTwo([a, b | _]) = (a, b)
```

## Basic Operations

```nostos
lst = [1, 2, 3, 4, 5]

# Length
lst.length()            # 5

# Access by index (0-based)
lst[0]                  # 1
lst[2]                  # 3
lst.get(0)              # 1

# Head and tail
lst.head()              # Some(1)
lst.tail()              # [2, 3, 4, 5]
lst.last()              # Some(5)

# Check empty
lst.isEmpty()           # false
[].isEmpty()            # true
```

## Transformations

```nostos
# Map: transform each element
[1, 2, 3].map(x => x * 2)           # [2, 4, 6]
["a", "b"].map(s => s.toUpper())    # ["A", "B"]

# Filter: keep matching elements
[1, 2, 3, 4].filter(x => x % 2 == 0)    # [2, 4]
["apple", "banana", "apricot"].filter(s => s.startsWith("a"))  # ["apple", "apricot"]

# Fold/Reduce: combine into single value
[1, 2, 3, 4].fold(0, (acc, x) => acc + x)   # 10 (sum)
[1, 2, 3, 4].fold(1, (acc, x) => acc * x)   # 24 (product)
```

## More List Methods

```nostos
# Take/Drop
[1, 2, 3, 4, 5].take(3)     # [1, 2, 3]
[1, 2, 3, 4, 5].drop(2)     # [3, 4, 5]

# Reverse
[1, 2, 3].reverse()         # [3, 2, 1]

# Concatenation
[1, 2] ++ [3, 4]            # [1, 2, 3, 4]
[[1, 2], [3, 4]].flatten()  # [1, 2, 3, 4]

# Sort
[3, 1, 4, 1, 5].sort()      # [1, 1, 3, 4, 5]

# Unique
[1, 2, 2, 3, 3, 3].unique() # [1, 2, 3]
```

## Searching

```nostos
lst = [10, 20, 30, 40]

# Find first match
lst.find(x => x > 15)       # Some(20)
lst.find(x => x > 100)      # None

# Check existence
lst.any(x => x > 30)        # true
lst.all(x => x > 5)         # true
lst.contains(20)            # true

# Index of
lst.indexOf(30)             # 2
lst.indexOf(99)             # -1
```

## FlatMap

```nostos
# Map then flatten
[[1, 2], [3, 4]].flatMap(lst => lst.map(x => x * 2))
# [2, 4, 6, 8]

# Useful for optional results
users.flatMap(u => u.email)  # Skips None values
```

## Zipping

```nostos
# Combine two lists element-wise
zip([1, 2, 3], ["a", "b", "c"])  # [(1, "a"), (2, "b"), (3, "c")]

# Zip with function
zipWith((a, b) => a + b, [1, 2], [10, 20])  # [11, 22]
```

## Partitioning

```nostos
# Split by predicate
(evens, odds) = [1, 2, 3, 4, 5].partition(x => x % 2 == 0)
# evens = [2, 4], odds = [1, 3, 5]

# Group by
words = ["apple", "banana", "apricot", "blueberry"]
words.groupBy(w => w.charAt(0))
# {'a': ["apple", "apricot"], 'b': ["banana", "blueberry"]}
```

## Recursive List Functions

```nostos
# Sum
sum([]) = 0
sum([h | t]) = h + sum(t)

# Length
len([]) = 0
len([_ | t]) = 1 + len(t)

# Map (manual implementation)
myMap([], _) = []
myMap([h | t], f) = [f(h) | myMap(t, f)]

# Filter (manual)
myFilter([], _) = []
myFilter([h | t], p) =
    if p(h) then [h | myFilter(t, p)]
    else myFilter(t, p)
```

## Method Chaining

```nostos
# Fluent API style
result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .filter(x => x % 2 == 0)      # [2, 4, 6, 8, 10]
    .map(x => x * x)              # [4, 16, 36, 64, 100]
    .take(3)                      # [4, 16, 36]
    .fold(0, (a, b) => a + b)     # 56

# Processing data
orders
    .filter(o => o.status == "completed")
    .map(o => o.total)
    .fold(0.0, (a, b) => a + b)
```

## Ranges

```nostos
# Create a range
range(1, 5)         # [1, 2, 3, 4]
range(0, 10)        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# With step
rangeStep(0, 10, 2) # [0, 2, 4, 6, 8]

# Common patterns
range(1, n + 1).map(x => x * x)  # Squares from 1 to n
range(0, n).fold(1, (acc, _) => acc * base)  # Power
```

## ForEach (Side Effects)

```nostos
# When you need side effects
[1, 2, 3].forEach(x => println(show(x)))

# vs map which returns values
results = [1, 2, 3].map(x => {
    println("Processing: " ++ show(x))
    x * 2
})
```

---

# Modules in Nostos

## File Structure

```
project/
├── main.nos          # Entry point with main()
├── utils.nos         # Module file
├── models/
│   ├── user.nos      # Nested module
│   └── order.nos
└── nostos.toml       # Optional project config
```

## Importing Modules

```nostos
# Import a module (file in same directory)
import utils

# Use qualified names
utils.helper()
utils.Config

# Import nested module
import models.user

models.user.createUser("Alice")
```

## Using Imports

```nostos
# Import and bring names into scope
import utils
use utils.*             # Import all public names

helper()                # Can use directly now
Config                  # Type is available

# Selective import
use utils.{helper, Config}

# Alias imports
import models.user as u
u.createUser("Alice")
```

## Public vs Private

```nostos
# In utils.nos:

# Public function (accessible from other modules)
pub helper() = "I'm public"

# Public type
pub type Config = { name: String, value: Int }

# Private function (only in this module)
internalHelper() = "I'm private"

# Private type
type InternalState = { data: List[Int] }
```

## Exporting

```nostos
# Everything marked `pub` is exported

# Public function
pub greet(name: String) = "Hello, " ++ name

# Public type
pub type User = { name: String, email: String }

# Public constant
pub const MAX_USERS = 100

# Public trait
pub trait Serializable
    serialize(self) -> String
end
```

## Module Example

```nostos
# math_utils.nos

pub const PI = 3.14159

pub square(x: Float) = x * x

pub cube(x: Float) = x * x * x

pub circleArea(radius: Float) = PI * square(radius)

# Private helper
clamp(x, min, max) = {
    if x < min then min
    else if x > max then max
    else x
}

pub clampedSquare(x: Float) = square(clamp(x, 0.0, 100.0))
```

```nostos
# main.nos

import math_utils
use math_utils.*

main() = {
    println("PI = " ++ show(PI))
    println("5^2 = " ++ show(square(5.0)))
    println("Circle area = " ++ show(circleArea(3.0)))
}
```

## Standard Library Imports

```nostos
# The stdlib is auto-imported, but you can be explicit
import json
import io
import url
import logging

# Using stdlib
json.parse('{"key": "value"}')
io.readFile("data.txt")
```

## Circular Dependencies

```nostos
# Nostos handles circular imports
# But prefer avoiding them for clarity

# a.nos
import b
pub aFunc() = b.bFunc() + 1

# b.nos
import a
pub bFunc() = 10
# Avoid calling a.aFunc() here - would cause infinite loop
```

## Project Configuration (nostos.toml)

```toml
[project]
name = "myproject"
version = "0.1.0"

[dependencies]
# External packages (from GitHub)
nalgebra = { git = "https://github.com/user/nalgebra-nos" }

[extensions]
# Native extensions
glam = { version = "0.1.0" }
```

## Visibility Rules

```nostos
# Public items are accessible from anywhere
pub type PublicType = { data: Int }
pub publicFunc() = 42

# Private items only in same module
type PrivateType = { secret: String }
privateFunc() = "hidden"

# Trait implementations follow the type's visibility
PublicType: Show
    show(self) = show(self.data)
end
```

## Nested Modules

```nostos
# models/user.nos
pub type User = { id: Int, name: String }

pub createUser(name: String) -> User = User(0, name)

pub validateEmail(email: String) -> Bool =
    email.contains("@")
```

```nostos
# main.nos
import models.user
use models.user.{User, createUser}

main() = {
    user = createUser("Alice")
    println(user.name)
}
```

## Re-exporting

```nostos
# lib.nos - re-export from submodules

import models.user
import models.order
import utils

# Re-export for convenient access
pub use models.user.User
pub use models.order.Order
pub use utils.helper
```

```nostos
# main.nos
import lib
use lib.*

# Can now use User, Order, helper directly
user = User(1, "Alice")
```

## Module Initialization

```nostos
# Code at module level runs when imported
# Use for initialization

# config.nos
println("Config module loaded")  # Runs on import

pub const DEBUG = true
pub var connectionCount = 0      # Module-level mutable state
```

## Best Practices

```nostos
# 1. One module = one responsibility
# Good: user.nos for user-related code
# Bad: utils.nos with everything

# 2. Explicit exports
pub type User = ...     # Document public API

# 3. Use qualified names for clarity
import json
json.parse(data)        # Clear where parse comes from

# 4. Group related imports
import models.user
import models.order
import models.product

use models.user.User
use models.order.Order
use models.product.Product
```

---

# Reactive Programming in Nostos

## Reactive Records

Reactive records automatically track changes to their fields:

```nostos
# Define a reactive type
reactive Counter = { value: Int }

main() = {
    counter = Counter { value: 0 }

    # Register change listener
    counter.onChange("value", (old, new) => {
        println("Value changed: " ++ show(old) ++ " -> " ++ show(new))
    })

    # Changes trigger the callback
    counter.value = 1   # "Value changed: 0 -> 1"
    counter.value = 5   # "Value changed: 1 -> 5"
}
```

## Multiple Callbacks

```nostos
reactive State = { count: Int, name: String }

main() = {
    state = State { count: 0, name: "initial" }

    # Multiple callbacks on same field
    state.onChange("count", (_, new) => println("Logger: count = " ++ show(new)))
    state.onChange("count", (_, new) => updateUI(new))

    # Callbacks on different fields
    state.onChange("name", (_, new) => println("Name changed to: " ++ new))

    state.count = 42    # Both count callbacks fire
    state.name = "updated"  # Name callback fires
}
```

## Change History Tracking

```nostos
reactive Document = { title: String, content: String }

mvar history: List[(String, String, String)] = []

trackChanges(doc: Document) = {
    doc.onChange("title", (old, new) => {
        history = history ++ [("title", old, new)]
    })
    doc.onChange("content", (old, new) => {
        history = history ++ [("content", old, new)]
    })
}

main() = {
    doc = Document { title: "Untitled", content: "" }
    trackChanges(doc)

    doc.title = "My Document"
    doc.content = "Hello, world!"
    doc.title = "My Great Document"

    # Print change history
    history.each((field, old, new) => {
        println(field ++ ": '" ++ old ++ "' -> '" ++ new ++ "'")
    })
}
```

## Reactive Variants

```nostos
reactive Status = Idle | Loading | Success(String) | Error(String)

main() = {
    status = Idle

    # Track state transitions
    status.onChange((old, new) => {
        println("Status: " ++ show(old) ++ " -> " ++ show(new))
    })

    status.set(Loading)           # "Status: Idle -> Loading"
    status.set(Success("Done"))   # "Status: Loading -> Success(Done)"
}
```

## State Machine Pattern

```nostos
reactive ConnectionState =
    | Disconnected
    | Connecting
    | Connected { sessionId: String }
    | Reconnecting { attempt: Int }

handleConnection(state: ConnectionState) = {
    state.onChange((old, new) => {
        match new {
            Connected { sessionId } -> {
                println("Connected! Session: " ++ sessionId)
                startHeartbeat()
            }
            Disconnected -> {
                println("Disconnected")
                stopHeartbeat()
            }
            Reconnecting { attempt } -> {
                println("Reconnecting... attempt " ++ show(attempt))
            }
            _ -> ()
        }
    })
}
```

## RWeb: Reactive Web Framework

```nostos
use stdlib.rweb.*

# Define reactive state for the page
reactive AppState = { count: Int, message: String }

# Session setup function
sessionSetup(writerId) = {
    state = AppState { count: 0, message: "Welcome!" }

    # Render function - called on state changes
    renderPage = () => RHtml(div([
        h1(state.message),
        p("Count: " ++ show(state.count)),
        button("Increment", dataAction: "increment"),
        button("Reset", dataAction: "reset")
    ]))

    # Action handler
    onAction = (action, params) => match action {
        "increment" -> { state.count = state.count + 1 }
        "reset" -> { state.count = 0 }
        _ -> ()
    }

    (renderPage, onAction)
}

main() = startRWeb(8080, "Counter App", sessionSetup)
```

## Component Pattern

```nostos
use stdlib.rweb.*

# Reusable counter component
counterComponent(state, fieldName: String) = {
    count = state[fieldName]

    component("counter-" ++ fieldName, () => RHtml(div(class: "counter", [
        span(show(count)),
        button("+", dataAction: "inc-" ++ fieldName),
        button("-", dataAction: "dec-" ++ fieldName)
    ])))
}

# Use in page
renderPage = () => RHtml(div([
    h1("Multi-Counter"),
    counterComponent(state, "counter1"),
    counterComponent(state, "counter2"),
    counterComponent(state, "counter3")
]))
```

## Form Handling

```nostos
use stdlib.rweb.*

reactive FormState = {
    username: String,
    email: String,
    errors: Map[String, String]
}

validate(state: FormState) = {
    errors = %{}
    if state.username.length() < 3 then {
        errors["username"] = "Username too short"
    }
    if !state.email.contains("@") then {
        errors["email"] = "Invalid email"
    }
    state.errors = errors
    errors.isEmpty()
}

sessionSetup(writerId) = {
    state = FormState { username: "", email: "", errors: %{} }

    renderPage = () => RHtml(form([
        div([
            label("Username:"),
            input(name: "username", value: state.username),
            errorMsg(state.errors["username"])
        ]),
        div([
            label("Email:"),
            input(name: "email", value: state.email),
            errorMsg(state.errors["email"])
        ]),
        button("Submit", dataAction: "submit")
    ]))

    onAction = (action, params) => match action {
        "submit" -> {
            state.username = params["username"]
            state.email = params["email"]
            if validate(state) then {
                saveUser(state.username, state.email)
            }
        }
        _ -> ()
    }

    (renderPage, onAction)
}
```

## Parent/Child Introspection

```nostos
reactive Parent = { children: List[Child] }
reactive Child = { name: String, value: Int }

main() = {
    parent = Parent { children: [] }
    child1 = Child { name: "first", value: 10 }
    child2 = Child { name: "second", value: 20 }

    parent.children = [child1, child2]

    # Check what holds a reference to child1
    parents = child1.parents
    println("child1 is held by " ++ show(parents.length()) ++ " parent(s)")

    # Get all children
    allChildren = parent.children
    println("Parent has " ++ show(allChildren.length()) ++ " children")
}
```

## Computed Values Pattern

```nostos
reactive Cart = { items: List[CartItem] }
reactive CartItem = { name: String, price: Float, quantity: Int }

# Computed total (not reactive, but derived)
cartTotal(cart: Cart) -> Float = {
    cart.items.fold(0.0, (sum, item) => {
        sum + (item.price * item.quantity.toFloat())
    })
}

# Re-render when items change
cart.onChange("items", (_, _) => {
    total = cartTotal(cart)
    updateTotalDisplay(total)
})
```

## Debounced Updates

```nostos
reactive SearchState = { query: String, results: List[String] }

# Debounce search to avoid too many API calls
debounce(ms: Int, fn: () -> ()) = {
    mvar timer: Option[Pid] = None

    () => {
        # Cancel previous timer
        match timer {
            Some(pid) -> send(pid, "cancel")
            None -> ()
        }

        # Start new timer
        me = self()
        newTimer = spawn(() => {
            match receiveTimeout(ms) {
                None -> send(me, "fire")  # Timeout = execute
                Some("cancel") -> ()       # Cancelled
            }
        })
        timer = Some(newTimer)
    }
}

main() = {
    state = SearchState { query: "", results: [] }
    debouncedSearch = debounce(300, () => {
        state.results = searchApi(state.query)
    })

    state.onChange("query", (_, _) => debouncedSearch())
}
```

## Undo/Redo Pattern

```nostos
reactive Editor = { content: String }

type EditorHistory = {
    past: List[String],
    future: List[String]
}

mvar history: EditorHistory = EditorHistory { past: [], future: [] }

trackHistory(editor: Editor) = {
    editor.onChange("content", (old, new) => {
        history = EditorHistory {
            past: history.past ++ [old],
            future: []  # Clear redo stack on new edit
        }
    })
}

undo(editor: Editor) = {
    match history.past {
        [] -> ()
        [..rest, last] -> {
            current = editor.content
            editor.content = last  # This won't trigger onChange for history
            history = EditorHistory {
                past: rest,
                future: [current] ++ history.future
            }
        }
    }
}

redo(editor: Editor) = {
    match history.future {
        [] -> ()
        [next, ..rest] -> {
            current = editor.content
            editor.content = next
            history = EditorHistory {
                past: history.past ++ [current],
                future: rest
            }
        }
    }
}
```

---

# Strings in Nostos

## String Literals

```nostos
# Double quotes
greeting = "Hello, World!"

# Single quotes (useful for embedded double quotes)
json = '{"name": "Alice", "age": 30}'

# Escape sequences
newline = "Line 1\nLine 2"
tab = "Col1\tCol2"
quote = "She said \"Hello\""
backslash = "C:\\path\\file"
```

## String Concatenation

```nostos
# Use ++ operator
full = "Hello" ++ " " ++ "World"

# Building strings
name = "Alice"
age = 30
message = "Name: " ++ name ++ ", Age: " ++ show(age)

# Multi-part concatenation
result = "Part 1" ++
         " Part 2" ++
         " Part 3"
```

## Converting to String

```nostos
# show() converts any value to string
show(42)           # "42"
show(3.14)         # "3.14"
show(true)         # "true"
show([1, 2, 3])    # "[1, 2, 3]"

# Common pattern
println("Value: " ++ show(someValue))
```

## String Methods

```nostos
s = "Hello, World!"

# Length
s.length()          # 13

# Case conversion
s.toUpper()         # "HELLO, WORLD!"
s.toLower()         # "hello, world!"

# Substring
s.substring(0, 5)   # "Hello"
s.substring(7, 12)  # "World"

# Contains/starts/ends
s.contains("World")     # true
s.startsWith("Hello")   # true
s.endsWith("!")         # true

# Finding
s.indexOf("World")      # 7 (or -1 if not found)

# Trimming whitespace
"  hello  ".trim()      # "hello"
"  hello  ".trimStart() # "hello  "
"  hello  ".trimEnd()   # "  hello"
```

## Split and Join

```nostos
# Split string into list
"a,b,c".split(",")          # ["a", "b", "c"]
"hello world".split(" ")     # ["hello", "world"]

# Join list into string
["a", "b", "c"].join(",")   # "a,b,c"
["hello", "world"].join(" ") # "hello world"

# Useful pattern
csv = "1,2,3,4,5"
numbers = csv.split(",").map(s => parseInt(s))
```

## Replace

```nostos
# Replace first occurrence
"hello world".replace("world", "there")  # "hello there"

# Replace all occurrences
"ababa".replaceAll("a", "x")  # "xbxbx"
```

## Character Access

```nostos
s = "Hello"

# Get character at index
s.charAt(0)         # 'H'
s.charAt(4)         # 'o'

# Get as list of characters
s.chars()           # ['H', 'e', 'l', 'l', 'o']
```

## String Comparison

```nostos
"abc" == "abc"      # true
"abc" != "def"      # true
"abc" < "abd"       # true (lexicographic)
"ABC" < "abc"       # true (uppercase < lowercase)
```

## Parsing Strings

```nostos
# Parse to Int
"42".parseInt()     # 42

# Parse to Float
"3.14".parseFloat() # 3.14

# Safe parsing (returns Result)
"42".tryParseInt()      # Ok(42)
"invalid".tryParseInt() # Err("...")
```

## Multiline Strings

```nostos
# Use regular strings with \n
multiline = "Line 1\nLine 2\nLine 3"

# Or concatenation for readability
poem = "Roses are red,\n" ++
       "Violets are blue,\n" ++
       "Nostos is fun,\n" ++
       "And so are you!"
```

## String Building Pattern

```nostos
# Building a string incrementally
buildReport(items: List[String]) -> String = {
    var result = "Report:\n"
    var i = 0
    while i < items.length() {
        result = result ++ "- " ++ items[i] ++ "\n"
        i = i + 1
    }
    result
}

# Functional style (preferred)
buildReport(items: List[String]) -> String =
    "Report:\n" ++ items.map(s => "- " ++ s).join("\n")
```

## URL Encoding

```nostos
import url

# Encode special characters
url.encode("hello world")   # "hello%20world"
url.encode("a=b&c=d")       # "a%3Db%26c%3Dd"

# Decode
url.decode("hello%20world") # "hello world"
```

## Common Patterns

```nostos
# Check if empty
s.length() == 0
s == ""

# Default for empty
name = if input == "" then "Anonymous" else input

# Format number with padding
padLeft(s: String, len: Int, char: String) -> String = {
    if s.length() >= len then s
    else padLeft(char ++ s, len, char)
}

padLeft(show(42), 5, "0")   # "00042"

# Repeat string
repeat(s: String, n: Int) -> String = {
    if n <= 0 then ""
    else s ++ repeat(s, n - 1)
}

repeat("ab", 3)   # "ababab"
```

---

# Templates & Metaprogramming in Nostos

Templates let you write code that generates code at compile time. This is powerful for eliminating boilerplate, creating DSLs, and building reusable patterns.

## Core Concepts

**Templates** are compile-time functions that manipulate code as data:
- `quote { code }` - captures code as AST (not executed, just stored)
- `~expr` - splices an AST value into a quote (inserts the code)
- `@decorator` - applies a template to a function or type

## Function Decorators

Decorators wrap or transform functions at compile time:

```nostos
# Double the return value of any function
template double(fn) = quote {
    result = ~fn.body    # splice in the original function body
    result * 2
}

@double
getValue() = 21

main() = getValue()  # Returns 42 (21 * 2)
```

**Available function metadata:**
- `~fn.name` - function name as String
- `~fn.body` - the function body (AST)
- `~fn.params` - list of parameters with name/type
- `~fn.returnType` - return type as String

## Type Decorators

Generate code based on type structure:

```nostos
# Auto-generate getters for all fields
template withGetters(typeDef) = quote {
    ~typeDef
    ~typeDef.fields.map(f =>
        eval("get_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ") = r." ++ f.name))
}

@withGetters
type Point = Point { x: Int, y: Int }

main() = {
    p = Point(x: 10, y: 20)
    get_x(p) + get_y(p)  # 30
}
```

**Type introspection:**
- `~typeDef.name` - type name as String
- `~typeDef.fields` - list of fields (for single-constructor types)
- `~typeDef.fields[i].name` - field name at index
- `~typeDef.fields[i].ty` - field type at index (note: `ty` not `type`)

## Code Generation with eval

`eval("code")` parses a string as code at compile time:

```nostos
# Generate a function dynamically
template makeAdder(typeDef, amount) = quote {
    ~typeDef
    ~eval("add" ++ ~amount ++ "(x: Int) = x + " ++ ~amount)
}

@makeAdder("10")
type Config = Config {}

main() = add10(5)  # 15
```

## Compile-Time Conditionals

Use `~if` for conditional code generation:

```nostos
template maybeLog(fn, shouldLog) = quote {
    ~if ~shouldLog {
        quote {
            println("Calling " ++ ~fn.name)
            ~fn.body
        }
    } else {
        quote { ~fn.body }
    }
}

@maybeLog(true)
debug() = 42      # prints "Calling debug", returns 42

@maybeLog(false)
release() = 42    # just returns 42, no logging
```

## Exception Handling in Templates

Add try/catch for error handling patterns:

```nostos
# Wrap any function with a fallback value
template withFallback(fn, fallback) = quote {
    try {
        ~fn.body
    } catch {
        _ -> ~fallback
    }
}

@withFallback("error")
risky() = throw("oops")

main() = risky()  # Returns "error" instead of throwing
```

## Retry Pattern

Retry failed operations multiple times:

```nostos
template retry3(fn) = quote {
    try { ~fn.body } catch {
        _ -> try { ~fn.body } catch {
            _ -> ~fn.body
        }
    }
}

mvar attempts: Int = 0

@retry3
flaky() = {
    attempts = attempts + 1
    if attempts < 3 { throw("not ready") }
    "success"
}

main() = flaky()  # Succeeds on 3rd attempt
```

## Unique Identifiers with gensym

Avoid naming collisions in generated code:

```nostos
template withHelper(typeDef) = quote {
    ~typeDef
    ~eval(~gensym("helper") ++ "() = 42")
}

@withHelper
type A = A {}

@withHelper
type B = B {}

# Generates helper_0() and helper_1() - no collision!
main() = helper_0() + helper_1()  # 84
```

## Parameter Access with param()

Reference function parameters with `~param(n)`:

```nostos
template validatePositive(fn) = quote {
    if ~param(0) <= 0 {
        throw(~fn.name ++ ": must be positive")
    }
    ~fn.body
}

@validatePositive
sqrt(n: Int) = n * n

main() = sqrt(5)    # 25
# sqrt(-1) would throw "sqrt: must be positive"
```

The `~param(n)` shorthand is equivalent to `~toVar(fn.params[n].name)`.

## Compile-Time Computation

Execute code at compile time with `comptime`:

```nostos
# String syntax
template withDefault(fn, expr) = quote {
    value = ~comptime(~expr)  # evaluated at compile time
    ~fn.body + value
}

@withDefault("21 * 2")
add(x: Int) = x

main() = add(0)  # 42

# Block syntax
template computed(fn, useSquare) = quote {
    ~comptime({
        base = 10
        if ~useSquare { base * base } else { base * 2 }
    })
}
```

## Practical Patterns

### Logging Decorator
```nostos
template logged(fn) = quote {
    println(">>> " ++ ~fn.name)
    result = ~fn.body
    println("<<< " ++ ~fn.name)
    result
}
```

### Builder Pattern
```nostos
template builder(typeDef) = quote {
    ~typeDef
    ~typeDef.fields.map(f =>
        eval("with_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ", v: " ++ f.ty ++ ") = " ++
             ~typeDef.name ++ "(" ++ f.name ++ ": v)"))
}

@builder
type Config = Config { timeout: Int, debug: Bool }

# Generates: with_timeout(r, v), with_debug(r, v)
```

### Feature Flags
```nostos
template featureFlag(fn, enabled, msg) = quote {
    ~if ~enabled {
        quote { ~fn.body }
    } else {
        quote { throw(~msg) }
    }
}

@featureFlag(true, "Beta disabled")
betaFeature() = "works!"

@featureFlag(false, "Experimental disabled")
experimental() = "never runs"
```

## Key Points for Code Generation

1. **Templates run at compile time** - no runtime overhead
2. **`quote` captures code** - doesn't execute it
3. **`~` splices values** - inserts AST into quotes
4. **`eval` parses strings** - for dynamic function names
5. **`gensym` creates unique names** - prevents collisions
6. **`toVar` references parameters** - for validation patterns
7. **`~if` generates conditionally** - compile-time branching
8. **`comptime` executes early** - pre-compute values
9. **try/catch works in templates** - for error handling patterns

## See Also

- **types.md** - Type definitions that templates can introspect (`~typeDef.fields`)
- **traits.md** - Templates can generate trait implementations
- **functions.md** - Function syntax that templates transform (`~fn.body`, `~fn.params`)
- **error_handling.md** - Try/catch syntax used in template error patterns

---

# Testing in Nostos

## Basic Assertions

```nostos
# Assert condition is true
assert(1 + 1 == 2)

# Assert with message
assert(user.age >= 18, "User must be adult")

# Assert equality
assert_eq(actual, expected)
assert_eq(sum([1, 2, 3]), 6)

# Assert not equal
assert_ne(a, b)
```

## Test File Format

Test files use special comments to specify expected behavior:

```nostos
# expect: 42
# This test expects main() to return 42

main() = 21 * 2
```

```nostos
# expect_error: type mismatch
# This test expects a compilation error containing "type mismatch"

main() = "hello" + 5
```

## Organizing Tests

```nostos
# Group related assertions
testArithmetic() = {
    assert_eq(1 + 1, 2)
    assert_eq(10 - 3, 7)
    assert_eq(4 * 5, 20)
    assert_eq(15 / 3, 5)
    println("Arithmetic tests passed!")
}

testStrings() = {
    assert_eq("hello" ++ " world", "hello world")
    assert_eq("abc".length(), 3)
    assert_eq("HELLO".toLower(), "hello")
    println("String tests passed!")
}

main() = {
    testArithmetic()
    testStrings()
    0
}
```

## Testing with Setup/Teardown

```nostos
# Setup helper
withTestData(test: List[Int] -> T) -> T = {
    data = [1, 2, 3, 4, 5]  # Setup
    result = test(data)
    # Cleanup (if needed)
    result
}

# Usage
main() = {
    withTestData(data => {
        assert_eq(data.length(), 5)
        assert_eq(data.sum(), 15)
    })
    0
}
```

## Testing Option/Result

```nostos
# Test Option values
testOption() = {
    some = Some(42)
    none: Option[Int] = None

    assert(some.isSome())
    assert(none.isNone())
    assert_eq(some.getOrElse(0), 42)
    assert_eq(none.getOrElse(0), 0)
}

# Test Result values
testResult() = {
    ok: Result[Int, String] = Ok(42)
    err: Result[Int, String] = Err("failed")

    assert(ok.isOk())
    assert(err.isErr())
    assert_eq(ok.getOrElse(0), 42)
    assert_eq(err.getOrElse(0), 0)
}
```

## Testing Exceptions

```nostos
# Test that exception is thrown
testThrows() = {
    threw = try {
        divide(10, 0)
        false
    } catch _ {
        true
    }
    assert(threw, "Should have thrown")
}

# Test exception message
testErrorMessage() = {
    message = try {
        parseNumber("abc")
        ""
    } catch e {
        e
    }
    assert(message.contains("invalid"), "Should mention 'invalid'")
}
```

## Property-Based Testing Pattern

```nostos
# Test properties hold for many inputs
testProperty(name: String, gen: () -> T, prop: T -> Bool, iterations: Int) = {
    for i = 0 to iterations {
        value = gen()
        if !prop(value) then {
            println("Property '" ++ name ++ "' failed for: " ++ show(value))
            assert(false)
        }
    }
    println("Property '" ++ name ++ "' passed " ++ show(iterations) ++ " tests")
}

# Usage
main() = {
    # Test that reversing a list twice gives original
    testProperty(
        "reverse-reverse",
        () => randomList(10),
        list => list.reverse().reverse() == list,
        100
    )
    0
}
```

## Testing Async/Concurrent Code

```nostos
# Test with timeout
testWithTimeout(name: String, timeoutMs: Int, test: () -> T) = {
    me = self()
    spawn(() => {
        result = test()
        send(me, ("done", result))
    })

    match receiveTimeout(timeoutMs) {
        Some(("done", result)) -> {
            println("Test '" ++ name ++ "' passed")
            result
        }
        None -> {
            println("Test '" ++ name ++ "' timed out!")
            assert(false)
        }
    }
}

# Test message passing
testMessagePassing() = {
    parent = self()
    child = spawn(() => {
        msg = receive()
        send(parent, msg * 2)
    })
    send(child, 21)
    result = receive()
    assert_eq(result, 42)
}
```

## Mocking Pattern

```nostos
# Define interface as functions
type Database = {
    query: String -> List[Row],
    execute: String -> Int
}

# Real implementation
realDb = Database {
    query: sql => Pg.query(conn, sql, []),
    execute: sql => Pg.execute(conn, sql, [])
}

# Mock implementation
mockDb = Database {
    query: _ => [("Alice", 30), ("Bob", 25)],
    execute: _ => 1
}

# Function uses interface
getUsers(db: Database) = db.query("SELECT * FROM users")

# Test with mock
testGetUsers() = {
    users = getUsers(mockDb)
    assert_eq(users.length(), 2)
}
```

## Test Fixtures

```nostos
# Create test data
createTestUser(id: Int) = User {
    id: id,
    name: "User" ++ show(id),
    email: "user" ++ show(id) ++ "@test.com"
}

createTestUsers(n: Int) = range(1, n + 1).map(createTestUser)

# Usage in tests
testUserFiltering() = {
    users = createTestUsers(10)
    adults = users.filter(u => u.id > 5)
    assert_eq(adults.length(), 5)
}
```

## Testing HTTP Handlers

```nostos
# Create mock request
mockRequest(path: String, method: String, body: String) = {
    Request {
        path: path,
        method: method,
        body: body,
        headers: %{},
        query: %{},
        id: 0
    }
}

# Capture response
type CapturedResponse = { status: Int, body: String }
mvar capturedResponse: Option[CapturedResponse] = None

mockRespond(req, status: Int, body: String) = {
    capturedResponse = Some(CapturedResponse { status, body })
}

# Test handler
testGetUsersEndpoint() = {
    req = mockRequest("/users", "GET", "")
    handler(req)  # Uses mockRespond

    match capturedResponse {
        Some(resp) -> {
            assert_eq(resp.status, 200)
            assert(resp.body.contains("Alice"))
        }
        None -> assert(false, "No response captured")
    }
}
```

## Benchmark Pattern

```nostos
benchmark(name: String, iterations: Int, fn: () -> T) = {
    start = currentTimeMillis()
    for i = 0 to iterations {
        fn()
    }
    elapsed = currentTimeMillis() - start
    perOp = elapsed / iterations
    println(name ++ ": " ++ show(elapsed) ++ "ms total, " ++ show(perOp) ++ "ms/op")
}

# Usage
main() = {
    benchmark("list append", 10000, () => {
        [1, 2, 3] ++ [4, 5, 6]
    })

    benchmark("map insert", 10000, () => {
        %{"a": 1}.insert("b", 2)
    })
    0
}
```

## Test Runner Pattern

```nostos
type TestResult = Pass(String) | Fail(String, String)

runTest(name: String, test: () -> ()) -> TestResult = {
    try {
        test()
        Pass(name)
    } catch e {
        Fail(name, e)
    }
}

runSuite(tests: List[(String, () -> ())]) = {
    results = tests.map((name, test) => runTest(name, test))

    passed = results.filter(r => match r { Pass(_) -> true, _ -> false }).length()
    failed = results.filter(r => match r { Fail(_, _) -> true, _ -> false }).length()

    results.each(r => match r {
        Pass(name) -> println("[PASS] " ++ name)
        Fail(name, err) -> println("[FAIL] " ++ name ++ ": " ++ err)
    })

    println("\n" ++ show(passed) ++ " passed, " ++ show(failed) ++ " failed")
    failed == 0
}

# Usage
main() = {
    success = runSuite([
        ("arithmetic", testArithmetic),
        ("strings", testStrings),
        ("options", testOption)
    ])
    if success then 0 else 1
}
```

---

# Traits in Nostos

## Defining Traits

```nostos
# Basic trait definition
trait Show
    show(self) -> String
end

trait Eq
    eq(self, other: Self) -> Bool
end

# Trait with multiple methods
trait Comparable
    compare(self, other: Self) -> Int
    lt(self, other: Self) -> Bool
    gt(self, other: Self) -> Bool
end
```

## Implementing Traits

```nostos
type Person = { name: String, age: Int }

# Implement Show for Person
Person: Show
    show(self) = self.name ++ " (age " ++ show(self.age) ++ ")"
end

# Implement Eq for Person
Person: Eq
    eq(self, other) = self.name == other.name && self.age == other.age
end

# Use the trait methods
alice = Person("Alice", 30)
bob = Person("Bob", 25)

alice.show()        # "Alice (age 30)"
alice.eq(bob)       # false
```

## Supertraits (Trait Inheritance)

```nostos
# Base trait
trait Displayable
    display(self) -> String
end

# Child trait requires Displayable
trait Formattable: Displayable
    format(self, prefix: String) -> String
end

type Item = { id: Int, name: String }

# Must implement Displayable FIRST
Item: Displayable
    display(self) = self.name
end

# Then can implement Formattable
Item: Formattable
    format(self, prefix) = prefix ++ ": " ++ self.display()
end
```

## Multiple Supertraits

```nostos
trait Printable
    print(self) -> String
end

trait Sortable
    sortKey(self) -> Int
end

# Requires both
trait Listable: Printable, Sortable
    listItem(self) -> String
end
```

## Trait Bounds on Functions

```nostos
# Function accepting any Show type
printItem[T: Show](item: T) = println(item.show())

# Multiple bounds with +
printAndCompare[T: Show + Eq](a: T, b: T) = {
    println(a.show())
    println(b.show())
    a.eq(b)
}

# Alternative syntax with when
process(item: T) when T: Show = item.show()
```

## Generic Trait Implementations

```nostos
type Box[T] = { value: T }

# Implement Show for Box when T has Show
Box[T]: Show when T: Show
    show(self) = "Box(" ++ self.value.show() ++ ")"
end

# Now works for any Box[T] where T: Show
intBox = Box(42)
intBox.show()       # "Box(42)"
```

## Operator Overloading via Traits

```nostos
# Num trait for arithmetic operators
trait Num
    add(self, other: Self) -> Self
    sub(self, other: Self) -> Self
    mul(self, other: Self) -> Self
    div(self, other: Self) -> Self
end

type Vec2 = { x: Int, y: Int }

Vec2: Num
    add(self, other) = Vec2(self.x + other.x, self.y + other.y)
    sub(self, other) = Vec2(self.x - other.x, self.y - other.y)
    mul(self, other) = Vec2(self.x * other.x, self.y * other.y)
    div(self, other) = Vec2(self.x / other.x, self.y / other.y)
end

# Now operators work!
v1 = Vec2(1, 2)
v2 = Vec2(3, 4)
v3 = v1 + v2        # Vec2(4, 6)
```

## Index Trait

```nostos
trait Index
    index(self, i: Int) -> Float
end

trait IndexMut
    indexMut(self, i: Int, value: Float) -> Self
end

type Vector = { data: List[Float] }

Vector: Index
    index(self, i) = self.data[i]
end

Vector: IndexMut
    indexMut(self, i, value) = Vector(self.data.set(i, value))
end

# Now bracket notation works
v = Vector([1.0, 2.0, 3.0])
v[0]            # 1.0
v[1] = 5.0      # Vector([1.0, 5.0, 3.0])
```

## Heterogeneous Collections with Sum Types

```nostos
trait Drawable
    draw(self) -> String
end

type Circle = { radius: Float }
type Square = { side: Float }

Circle: Drawable
    draw(self) = "Circle(r=" ++ show(self.radius) ++ ")"
end

Square: Drawable
    draw(self) = "Square(s=" ++ show(self.side) ++ ")"
end

# Sum type wrapper
type Shape = C(Circle) | S(Square)

Shape: Drawable
    draw(self) = match self { C(c) -> c.draw(), S(s) -> s.draw() }
end

# Now heterogeneous list works!
shapes: List[Shape] = [C(Circle(1.0)), S(Square(2.0))]
shapes.map(s => s.draw())   # ["Circle(r=1)", "Square(s=2)"]
```

## Built-in Traits

```nostos
# Show - convert to string
trait Show
    show(self) -> String
end

# Eq - equality comparison
trait Eq
    eq(self, other: Self) -> Bool
end

# Ord - ordering comparison
trait Ord
    lt(self, other: Self) -> Bool
    gt(self, other: Self) -> Bool
    lte(self, other: Self) -> Bool
    gte(self, other: Self) -> Bool
end

# Hash - for use in Maps/Sets
trait Hash
    hash(self) -> Int
end

# Copy - value can be copied (default for simple types)
trait Copy end

# Default - has a default value
trait Default
    default() -> Self
end
```

## Deriving Traits

```nostos
# Automatically derive common traits
type Point = { x: Int, y: Int }
    deriving (Show, Eq, Hash, Copy)

# Now these work automatically
p = Point(1, 2)
p.show()            # "Point { x: 1, y: 2 }"
p.eq(Point(1, 2))   # true
p.hash()            # some hash value
```

---

# Types in Nostos

## Built-in Types

```nostos
Int         # 64-bit integer
Float       # 64-bit floating point
Bool        # true or false
String      # UTF-8 string
Char        # Single character
()          # Unit type (empty value)
List[T]     # List of elements
(A, B)      # Tuple
Map[K, V]   # Map/dictionary
Set[T]      # Set
```

## Type Annotations

```nostos
# Variables (usually inferred)
x: Int = 42
name: String = "Alice"
items: List[Int] = [1, 2, 3]

# Function parameters and return
add(a: Int, b: Int) -> Int = a + b

# Complex types
data: Map[String, List[Int]] = %{"a": [1, 2], "b": [3, 4]}
```

## Record Types

```nostos
# Define a record type
type Person = { name: String, age: Int }

# Create instance
alice = Person("Alice", 30)
bob = Person(name: "Bob", age: 25)

# Access fields
alice.name      # "Alice"
alice.age       # 30

# Records are immutable - create new with changes
older = Person(alice.name, alice.age + 1)
```

## Variant Types (Sum Types)

```nostos
# Define variants
type Color = Red | Green | Blue

# Variants with data
type Shape = Circle(Float) | Rectangle(Float, Float) | Point

# Use pattern matching
describe(c: Color) = match c {
    Red -> "red",
    Green -> "green",
    Blue -> "blue"
}

area(s: Shape) = match s {
    Circle(r) -> 3.14159 * r * r,
    Rectangle(w, h) -> w * h,
    Point -> 0.0
}

# Common pattern: Option
type Option[T] = Some(T) | None

# Common pattern: Result
type Result[T, E] = Ok(T) | Err(E)
```

## Variant with Named Fields

```nostos
# Variants can have named fields
type Event =
    | Click { x: Int, y: Int }
    | KeyPress { key: Char, modifiers: List[String] }
    | Scroll { delta: Int }

# Pattern match with field names
handle(e: Event) = match e {
    Click { x, y } -> "Clicked at " ++ show(x) ++ "," ++ show(y),
    KeyPress { key, _ } -> "Pressed " ++ show(key),
    Scroll { delta } -> "Scrolled " ++ show(delta)
}
```

## Generic Types

```nostos
# Generic record
type Box[T] = { value: T }

intBox: Box[Int] = Box(42)
strBox: Box[String] = Box("hello")

# Generic variant
type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])

# Multiple type parameters
type Pair[A, B] = { first: A, second: B }
type Either[L, R] = Left(L) | Right(R)
```

## Tuples

```nostos
# Tuple types
point: (Int, Int) = (10, 20)
triple: (String, Int, Bool) = ("hello", 42, true)

# Access by index (pattern matching)
(x, y) = point
(name, age, active) = triple

# Tuple in function return
divmod(a: Int, b: Int) -> (Int, Int) = (a / b, a % b)
(quotient, remainder) = divmod(17, 5)
```

## Nested Types

```nostos
type Address = { street: String, city: String, zip: String }
type Company = { name: String, address: Address }
type Employee = { name: String, company: Company }

# Accessing nested fields
emp = Employee("Alice", Company("Acme", Address("123 Main", "NYC", "10001")))
emp.company.address.city    # "NYC"
```

## Recursive Types

```nostos
# Linked list
type LinkedList[T] = Nil | Cons(T, LinkedList[T])

# Binary tree
type BinaryTree[T] = Empty | Node(T, BinaryTree[T], BinaryTree[T])

# JSON-like structure
type Json =
    | JsonNull
    | JsonBool(Bool)
    | JsonNumber(Float)
    | JsonString(String)
    | JsonArray(List[Json])
    | JsonObject(Map[String, Json])
```

## Type Inference

```nostos
# Types are inferred when possible
x = 42                  # x: Int
y = "hello"             # y: String
z = [1, 2, 3]           # z: List[Int]

# Inference through functions
double(x) = x * 2       # Inferred: Int -> Int
greet(name) = "Hi " ++ name  # Inferred: String -> String

# Sometimes annotation needed
identity(x: T) -> T = x  # Generic needs explicit parameter
```

## Working with Option

```nostos
type Option[T] = Some(T) | None

# Creating
found: Option[Int] = Some(42)
missing: Option[Int] = None

# Pattern match
getValue(opt: Option[Int]) -> Int = match opt {
    Some(x) -> x,
    None -> 0
}

# Common methods
Some(42).map(x => x * 2)        # Some(84)
None.map(x => x * 2)            # None
Some(42).getOrElse(0)           # 42
None.getOrElse(0)               # 0
```

## Working with Result

```nostos
type Result[T, E] = Ok(T) | Err(E)

# Creating
success: Result[Int, String] = Ok(42)
failure: Result[Int, String] = Err("not found")

# Pattern match
handle(r: Result[Int, String]) = match r {
    Ok(value) -> "Got: " ++ show(value),
    Err(msg) -> "Error: " ++ msg
}

# Common methods
Ok(42).map(x => x * 2)          # Ok(84)
Err("fail").map(x => x * 2)     # Err("fail")
Ok(42).mapErr(e => "Error: " ++ e)  # Ok(42)
```

## Constructing Records/Variants

```nostos
type Person = { name: String, age: Int }

# Positional
p1 = Person("Alice", 30)

# Named (order doesn't matter)
p2 = Person(age: 25, name: "Bob")

# Variants
type Status = Active | Inactive(String)

s1 = Active
s2 = Inactive("vacation")
```

## See Also

- **templates.md** - Generate code based on type structure (`~typeDef.fields`)
- **traits.md** - Add behavior to types with trait implementations
- **error_handling.md** - `Option[T]` and `Result[T, E]` patterns
- **02_stdlib_reference.md** - Methods available on Option, Result, and collections

---

# WebSockets in Nostos

## WebSocket Server

```nostos
use stdlib.server.*

handler(req) = {
    if WebSocket.isUpgrade(req) then {
        ws = WebSocket.accept(req.id)
        handleWebSocket(ws)
    } else {
        respondText(req, "WebSocket endpoint")
    }
}

handleWebSocket(ws) = {
    msg = WebSocket.recv(ws)
    WebSocket.send(ws, "Echo: " ++ msg)
    handleWebSocket(ws)  # Loop for more messages
}

main() = serve(8080, handler)
```

## WebSocket Client

```nostos
# Connect to WebSocket server
ws = WebSocket.connect("wss://echo.websocket.org")

# Send message
WebSocket.send(ws, "Hello, WebSocket!")

# Receive message
response = WebSocket.recv(ws)
println("Got: " ++ response)

# Close connection
WebSocket.close(ws)
```

## Message Types

```nostos
# Text messages (default)
WebSocket.send(ws, "Hello")
msg = WebSocket.recv(ws)  # String

# JSON messages
data = %{"type": "chat", "text": "Hello"}
WebSocket.send(ws, json.stringify(data))

received = json.parse(WebSocket.recv(ws))
msgType = received["type"]
```

## Chat Server Pattern

```nostos
mvar clients: List[WebSocket] = []

broadcast(message: String) = {
    clients.each(ws => {
        try { WebSocket.send(ws, message) }
        catch _ { () }  # Ignore send errors
    })
}

handleClient(ws) = {
    # Add to client list
    clients = clients ++ [ws]

    loop() = {
        try {
            msg = WebSocket.recv(ws)
            broadcast(msg)
            loop()
        } catch _ {
            # Remove on disconnect
            clients = clients.filter(c => c != ws)
        }
    }
    loop()
}

handler(req) = {
    if WebSocket.isUpgrade(req) then {
        ws = WebSocket.accept(req.id)
        spawn(() => handleClient(ws))
    } else {
        respondText(req, "Chat server")
    }
}

main() = serve(8080, handler)
```

## Typed Message Protocol

```nostos
type ClientMessage =
    | Join { username: String }
    | Chat { text: String }
    | Leave

type ServerMessage =
    | Welcome { users: List[String] }
    | UserJoined { username: String }
    | Message { from: String, text: String }
    | UserLeft { username: String }

parseClientMessage(raw: String) -> ClientMessage = {
    data = json.parse(raw)
    match data["type"] {
        "join" -> Join { username: data["username"] }
        "chat" -> Chat { text: data["text"] }
        "leave" -> Leave
    }
}

sendServerMessage(ws, msg: ServerMessage) = {
    json = match msg {
        Welcome { users } -> %{"type": "welcome", "users": users}
        UserJoined { username } -> %{"type": "joined", "username": username}
        Message { from, text } -> %{"type": "message", "from": from, "text": text}
        UserLeft { username } -> %{"type": "left", "username": username}
    }
    WebSocket.send(ws, json.stringify(json))
}
```

## Room-Based Chat

```nostos
type Room = { name: String, clients: List[WebSocket] }

mvar rooms: Map[String, Room] = %{}

joinRoom(roomName: String, ws: WebSocket) = {
    room = rooms[roomName].getOrElse(Room { name: roomName, clients: [] })
    updatedRoom = Room { name: roomName, clients: room.clients ++ [ws] }
    rooms[roomName] = updatedRoom
}

broadcastToRoom(roomName: String, message: String) = {
    match rooms[roomName] {
        Some(room) -> room.clients.each(ws => WebSocket.send(ws, message))
        None -> ()
    }
}

leaveRoom(roomName: String, ws: WebSocket) = {
    match rooms[roomName] {
        Some(room) -> {
            updated = Room { name: roomName, clients: room.clients.filter(c => c != ws) }
            rooms[roomName] = updated
        }
        None -> ()
    }
}
```

## Ping/Pong Heartbeat

```nostos
handleWithHeartbeat(ws) = {
    lastPing = currentTimeMillis()

    # Spawn heartbeat checker
    spawn(() => {
        while true {
            sleep(30000)  # Check every 30s
            if currentTimeMillis() - lastPing > 60000 then {
                WebSocket.close(ws)
                break
            }
        }
    })

    loop() = {
        msg = WebSocket.recv(ws)
        if msg == "ping" then {
            lastPing = currentTimeMillis()
            WebSocket.send(ws, "pong")
        } else {
            handleMessage(ws, msg)
        }
        loop()
    }
    loop()
}
```

## Binary Data

```nostos
# Send binary data as base64
sendBinary(ws, data: List[Int]) = {
    encoded = base64Encode(data)
    WebSocket.send(ws, encoded)
}

# Receive and decode
receiveBinary(ws) -> List[Int] = {
    encoded = WebSocket.recv(ws)
    base64Decode(encoded)
}
```

## Error Handling

```nostos
handleWebSocket(ws) = {
    try {
        loop() = {
            msg = WebSocket.recv(ws)
            response = processMessage(msg)
            WebSocket.send(ws, response)
            loop()
        }
        loop()
    } catch {
        "connection closed" -> println("Client disconnected")
        e -> println("WebSocket error: " ++ e)
    }
}
```

## Reconnection (Client)

```nostos
connectWithRetry(url: String, maxRetries: Int) -> Option[WebSocket] = {
    attempt(retries) = {
        if retries <= 0 then None
        else {
            try {
                ws = WebSocket.connect(url)
                Some(ws)
            } catch _ {
                println("Connection failed, retrying in 1s...")
                sleep(1000)
                attempt(retries - 1)
            }
        }
    }
    attempt(maxRetries)
}
```

## Broadcast with Sender Filtering

```nostos
mvar clientMap: Map[Int, WebSocket] = %{}
mvar nextId: Int = 0

addClient(ws: WebSocket) -> Int = {
    id = nextId
    nextId = nextId + 1
    clientMap[id] = ws
    id
}

broadcastExcept(senderId: Int, message: String) = {
    clientMap.entries().each((id, ws) => {
        if id != senderId then {
            try { WebSocket.send(ws, message) }
            catch _ { clientMap = clientMap.remove(id) }
        }
    })
}
```

---

