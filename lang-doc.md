# Nostos Language Reference

This document serves as a quick reference for the Nostos programming language syntax and semantics.

## Basics

### Comments
```nos
# Single-line comment
```

### Literals
```nos
42              # Int
3.14            # Float
true            # Bool
false           # Bool
'a'             # Char
"hello"         # String
()              # Unit
0xFF            # Hex
0b1010          # Binary
1_000_000       # Underscores in numbers
```

### Blocks
Blocks use curly braces with statements separated by **newlines** or **commas**:
```nos
main() = {
    x = 10
    y = 20
    x + y       # Last expression is returned
}

# Or with commas (equivalent):
main() = { x = 10, y = 20, x + y }
```

**Note**: Newlines are significant - each statement should be on its own line or separated by a comma.

## Functions

### Single-clause Functions
```nos
double(x) = x * 2
add(a, b) = a + b
```

### Multi-clause Functions (Pattern Matching)
```nos
fact(0) = 1
fact(n) = n * fact(n - 1)

fib(0) = 0
fib(1) = 1
fib(n) = fib(n - 1) + fib(n - 2)
```

### Functions with Guards
```nos
abs(n) when n >= 0 = n
abs(n) = -n

max(a, b) when a > b = a
max(a, b) = b
```

### Lambda Expressions
```nos
double = x => x * 2
add = (a, b) => a + b
apply = (f, x) => f(x)
```

## Types

### Type Definitions
```nos
# Variant types (sum types)
type Option[T] = Some(T) | None

type Result[T, E] = Ok(T) | Err(E)

type List[T] = Nil | Cons(T, List[T])

# Record types
type Point = Point { x: Int, y: Int }

type Person = Person { name: String, age: Int }
```

### Using Variants
```nos
type Color = Red | Green | Blue | RGB(Int, Int, Int)

describe(Red) = "red"
describe(Green) = "green"
describe(Blue) = "blue"
describe(RGB(r, g, b)) = "custom color"
```

### Using Records
```nos
type Point = Point { x: Int, y: Int }

# Create record
p = Point { x: 10, y: 20 }

# Access field
get_x(Point { x, y }) = x
```

## Collections

### Lists
```nos
empty = []
numbers = [1, 2, 3, 4, 5]

# Cons operator (::) - prepend element to list
list1 = 1 :: [2, 3]           # [1, 2, 3]
list2 = 1 :: 2 :: 3 :: []     # [1, 2, 3] (right-associative)

# List cons pattern matching
head([x | _]) = x
tail([_ | xs]) = xs

# Recursive list processing
sum([]) = 0
sum([x | xs]) = x + sum(xs)

length([]) = 0
length([_ | xs]) = 1 + length(xs)
```

### Tuples
```nos
pair = (1, 2)
triple = ("hello", 42, true)

# Tuple pattern matching
first((a, _)) = a
second((_, b)) = b
```

### Sets
Sets are unordered collections of unique elements. Use the `#{}` syntax:
```nos
# Empty set
empty = #{}

# Set with elements - duplicates are automatically removed
numbers = #{1, 2, 3, 2, 1}   # Results in #{1, 2, 3}

# Set of strings
tags = #{"urgent", "review", "bug"}

# Boolean set
bools = #{true, false}
```

Sets are useful for representing unique collections and removing duplicates.

### Maps
Maps (dictionaries) provide key-value storage with O(1) lookup. Use the `%{key: value}` syntax:
```nos
# Empty map
empty = %{}

# Map with string keys
person = %{"name": "Alice", "age": 30, "city": "Paris"}

# Map with integer keys
squares = %{1: 1, 2: 4, 3: 9, 4: 16}

# Map with mixed value types
config = %{"debug": true, "port": 8080, "host": "localhost"}
```

Keys must be hashable (strings, numbers, booleans).

### Typed Arrays
Typed arrays are contiguous arrays of a single numeric type. They are optimized for efficient numeric computation and JIT compilation.

```nos
# Int64Array - array of 64-bit integers
arr = newInt64Array(10)       # Create array of 10 zeros
arr[0] = 42                   # Set element by index
x = arr[0]                    # Get element by index
len = length(arr)             # Get array length

# Float64Array - array of 64-bit floats
floats = newFloat64Array(5)   # Create array of 5 zeros
floats[0] = 3.14
floats[1] = 2.71
y = floats[0]

# Float32Array - array of 32-bit floats (ideal for pgvector)
vec = Float32Array.fromList([1.0, 2.0, 3.0])
len = Float32Array.length(vec)
val = Float32Array.get(vec, 0)
vec2 = Float32Array.set(vec, 0, 99.0)
lst = Float32Array.toList(vec)
zeros = Float32Array.make(5, 0.0)   # 5 elements, all 0.0
```

Example: Sum an array
```nos
sumArray(arr, i, acc) =
    if i >= length(arr) then acc
    else sumArray(arr, i + 1, acc + arr[i])

main() = {
    arr = newInt64Array(10)
    fillRange(arr, 0)           # Fill with 1, 2, 3, ..., 10
    sumArray(arr, 0, 0)         # Returns 55
}

fillRange(arr, i) =
    if i >= length(arr) then ()
    else {
        arr[i] = i + 1
        fillRange(arr, i + 1)
    }
```

**Note**: Typed arrays store raw numeric values without boxing, making them efficient for numeric algorithms. Unlike regular lists, they support O(1) indexed access and mutation.

## Pattern Matching

### Match Expression
```nos
describe(n) = match n {
    0 -> "zero"
    1 -> "one"
    _ -> "many"
}

# With guards
abs(n) = match n {
    x when x >= 0 -> x
    x -> -x
}
```

### Patterns in Function Definitions
```nos
# Literal patterns
is_zero(0) = true
is_zero(_) = false

# List patterns
is_empty([]) = true
is_empty(_) = false

# Tuple patterns
swap((a, b)) = (b, a)

# Variant patterns
unwrap(Some(x)) = x
unwrap(None) = panic("None")

# Nested patterns
first_of_first([[x | _] | _]) = x

# String patterns - match string prefixes
greet(["hello" | rest]) = "Hi! " ++ rest
greet(["bye" | rest]) = "Goodbye! " ++ rest
greet(_) = "I don't understand"

# Single character string patterns
first_char(["h" | _]) = "starts with h"
first_char(["a" | _]) = "starts with a"
first_char(_) = "other"
```

## Control Flow

### If Expression
```nos
max(a, b) = if a > b then a else b

sign(n) = if n > 0 then 1 else if n < 0 then -1 else 0
```

### Match Expression
```nos
classify(n) = match n {
    0 -> "zero"
    1 -> "one"
    2 -> "two"
    _ -> "many"
}
```

### Loops

#### While Loop
```nos
# while condition { body }
countdown(n) = {
    i = n
    while i > 0 {
        println(i)
        i = i - 1
    }
}

# Sum an array with while loop
sumArray(arr) = {
    total = 0
    i = 0
    n = length(arr)
    while i < n {
        total = total + arr[i]
        i = i + 1
    }
    total
}
```

#### For Loop
```nos
# for var = start to end { body }
# Iterates from start up to (but not including) end
printRange(a, b) = {
    for i = a to b {
        println(i)
    }
}

# Fill array with squares
fillSquares(arr) = {
    n = length(arr)
    for i = 0 to n {
        arr[i] = i * i
    }
}
```

#### Break and Continue
```nos
# break exits the loop early, optionally with a value
findFirst(arr, target) = {
    result = -1
    for i = 0 to length(arr) {
        if arr[i] == target then {
            result = i
            break
        }
    }
    result
}

# continue skips to the next iteration
sumPositive(arr) = {
    total = 0
    for i = 0 to length(arr) {
        if arr[i] < 0 then continue
        total = total + arr[i]
    }
    total
}
```

## Exception Handling

Nostos provides try/catch for exception handling. Exceptions can be any value type.

### Throw
```nos
# Throw an exception with any value
throw("error message")
throw(42)
throw(ErrorRecord { code: 404, msg: "not found" })
```

### Try/Catch
```nos
# Basic try/catch
result = try { risky_operation() }
    catch { e -> handle_error(e) }

# The caught value becomes the result if an exception occurs
safe_divide(a, b) =
    if b == 0 then throw("division by zero")
    else a / b

result = try { safe_divide(10, 0) } catch { e -> e }
# result = "division by zero"
```

### Pattern Matching in Catch
```nos
# Handle different exception types differently
classify_error(code) =
    try {
        if code == 1 then throw("not_found")
        else if code == 2 then throw("forbidden")
        else "ok"
    }
    catch {
        "not_found" -> "Error: Not found"
        "forbidden" -> "Error: Access denied"
        other -> "Error: Unknown"
    }
```

### Nested Try/Catch
```nos
# Inner exceptions can be caught by outer handlers
nested_example() =
    try {
        try { throw("inner error") }
        catch { "other" -> "handled other" }
        # "inner error" doesn't match, propagates up
    }
    catch { e -> "outer caught: " ++ e }
```

### Safe Wrapper Pattern
```nos
# Use a thunk (zero-arg function) to delay evaluation
safe_call(risky_fn, default) =
    try { risky_fn() }
    catch { _ -> default }

# Usage
result = safe_call(() => dangerous_operation(), -1)
```

## Higher-Order Functions

```nos
# Function as parameter
apply(f, x) = f(x)
twice(f, x) = f(f(x))

# Function as return value
add(n) = x => x + n
add5 = add(5)

# Common patterns (using :: cons operator for building results)
map(_, []) = []
map(f, [x | xs]) = f(x) :: map(f, xs)

filter(_, []) = []
filter(p, [x | xs]) = if p(x) then x :: filter(p, xs) else filter(p, xs)

foldl(_, acc, []) = acc
foldl(f, acc, [x | xs]) = foldl(f, f(acc, x), xs)
```

## Traits

```nos
# Define a trait
trait Show[T] {
    show(T) -> String
}

# Implement for a type
impl Show[Int] {
    show(n) = int_to_string(n)
}

impl Show[Bool] {
    show(true) = "true"
    show(false) = "false"
}
```

## Builtin Traits

All types in Nostos automatically have implementations for the core traits `Hash`, `Eq`, `Show`, and `Copy`. No special syntax is needed - these capabilities are always available.

### Available Traits

| Trait | Function | Description |
|-------|----------|-------------|
| `Hash` | `hash(x) -> Int` | Hash function for maps/sets |
| `Show` | `show(x) -> String` | String representation |
| `Eq` | `==`, `!=` operators | Equality comparison |
| `Copy` | `copy(x) -> T` | Deep copy |

### Examples

```nos
# All types automatically support Hash, Show, Eq, Copy
type Color = Red | Green | Blue
type Result = Ok(Int) | Err(String)
type Address = { street: String, city: String, zip: Int }

# Nested types work automatically
type Email = Email(String)
type Contact = { name: String, email: Email }
```

### Using Builtin Traits

```nos
type Person = { name: String, age: Int }

main() = {
    alice = Person("Alice", 30)
    bob = Person("Bob", 25)

    # Using Eq
    println("alice == alice: " ++ show(alice == alice))  # true
    println("alice == bob: " ++ show(alice == bob))      # false

    # Using Show
    println("Person: " ++ show(alice))  # Person{name: Alice, age: 30}

    # Using Hash
    println("Hash: " ++ show(hash(alice)))

    # Using Copy
    alice_copy = copy(alice)
    println("Copy equals original: " ++ show(alice == alice_copy))  # true
}
```

## Trait Bounds

Trait bounds let you write generic functions that require certain capabilities from their type parameters. This enables compile-time checking that types support the operations you need.

### Basic Syntax

```nos
# Function with trait bound: T must implement Hash
hashable[T: Hash](x: T) -> Int = hash(x)

# Multiple constraints with +
hash_and_show[T: Hash + Show](x: T) -> String = {
    h = hash(x)
    "hash(" ++ show(x) ++ ") = " ++ show(h)
}
```

### Multiple Type Parameters

Each type parameter can have its own constraints:

```nos
# Two parameters, each with a constraint
compare_hashes[T: Hash, U: Hash](x: T, y: U) -> Bool = hash(x) == hash(y)

# Mixed: one constrained, one unconstrained
show_first[T: Show, U](x: T, y: U) -> String = show(x)
```

### Examples

```nos
# Generic equality check
are_equal[T: Eq](x: T, y: T) -> Bool = x == y

# Generic hash-based lookup
lookup[K: Eq, V](items: List, key: K) -> V = match items {
    [] -> panic("Key not found")
    [(k, v) | rest] -> if k == key then v else lookup(rest, key)
}

# Works with any type that implements Eq (all types do)
type UserId = UserId(Int)

main() = {
    # All types implement Eq, Hash, Show
    println(show(are_equal(42, 42)))      # true
    println(show(are_equal("a", "b")))    # false

    # Custom types automatically have these traits
    println(show(are_equal(UserId(1), UserId(1))))  # true
}
```

### Primitive Type Traits

Built-in primitive types automatically implement common traits:

| Type | Implements |
|------|------------|
| `Int` | `Hash`, `Eq`, `Show`, `Copy` |
| `Float` | `Eq`, `Show`, `Copy` |
| `Bool` | `Hash`, `Eq`, `Show`, `Copy` |
| `Char` | `Hash`, `Eq`, `Show`, `Copy` |
| `String` | `Hash`, `Eq`, `Show`, `Copy` |

### Compile-Time Checking

Trait bounds are checked at compile time. This is most useful for custom traits that you define yourself. If you try to use a type that doesn't implement the required trait, you get a compile error:

```nos
trait Printable
    toText(x) -> String

printable[T: Printable](x: T) -> String = toText(x)

type MyType = MyType(Int)  # No Printable impl

main() = {
    x = MyType(42)
    printable(x)  # Compile error: MyType does not implement trait `Printable`
}
```

Note: All types automatically implement the builtin traits `Hash`, `Eq`, `Show`, and `Copy`.

## Concurrency (Erlang-style)

### Spawning Processes
```nos
# spawn creates a new process and returns its Pid
child_func(parent) = {
    parent <- 42,    # Send message to parent
    ()
}

main() = {
    me = self(),                    # Get current process's Pid
    spawn(() => child_func(me)),    # Spawn child process
    receive { n -> n }              # Receive message, return it
}
```

### Send Operator (`<-`)
```nos
# pid <- message
parent <- 42              # Send integer
parent <- (x, y)          # Send tuple
parent <- Result(value)   # Send variant
```

### Receive Expression
```nos
# Basic receive
receive { msg -> handle(msg) }

# Pattern matching in receive
receive {
    (a, b) -> a + b
    n -> n * 2
}

# Receive with timeout (returns timeout_value if no message within N ms)
receive {
    msg -> handle(msg)
    after 1000 -> timeout_value
}

# Multiple receives
receive {
    a -> receive { b -> a + b }
}
```

### Sleep Function
```nos
# Sleep for N milliseconds
sleep(1000)   # Sleeps for 1 second

# Sleep allows other processes to run while waiting
worker(id, delay, parent) = {
    sleep(delay)
    parent <- id
    ()
}
```

### Complete Example
```nos
worker(parent, n) = {
    result = n * n,
    parent <- result,
    ()
}

main() = {
    me = self(),
    spawn(() => worker(me, 5)),
    spawn(() => worker(me, 10)),
    receive {
        a -> receive { b -> a + b }    # Returns 25 + 100 = 125
    }
}
```

## Async I/O

Nostos provides non-blocking I/O operations that integrate seamlessly with the process scheduler. When a process performs I/O, it yields to allow other processes to run, then resumes when the operation completes.

### File I/O

Nostos provides comprehensive file I/O operations with both high-level convenience functions and low-level handle-based operations.

#### High-Level File Operations

```nos
# Read entire file as string
(status, content) = File.readAll("/path/to/file.txt")

# Write string to file (creates or overwrites)
(status, result) = File.writeAll("/path/to/file.txt", "Hello, World!")

# Check if file exists
(status, exists) = File.exists("/path/to/file.txt")

# Get file size in bytes
(status, size) = File.size("/path/to/file.txt")

# Remove a file
(status, result) = File.remove("/path/to/file.txt")

# Rename/move a file
(status, result) = File.rename("/old/path.txt", "/new/path.txt")

# Copy a file
(status, result) = File.copy("/source.txt", "/dest.txt")
```

#### File Handle Operations

For more control, use file handles with explicit open/close:

```nos
# Open a file - modes: "r" (read), "w" (write), "a" (append), "rw" (read/write)
(status, handle) = File.open("/path/to/file.txt", "w")

# Write to file
(status, result) = File.write(handle, "Hello, World!\n")

# Flush buffered data to disk
(status, result) = File.flush(handle)

# Close the file
(status, result) = File.close(handle)

# Reading with handles
(status, handle) = File.open("/path/to/file.txt", "r")
(status, data) = File.read(handle, 100)      # Read up to 100 bytes
(status, line) = File.readLine(handle)       # Read one line

# Seek to position - whence: "start", "current", "end"
(status, newPos) = File.seek(handle, 10, "start")    # Seek to byte 10
(status, newPos) = File.seek(handle, -5, "current")  # Move back 5 bytes
(status, newPos) = File.seek(handle, 0, "end")       # Seek to end

File.close(handle)
```

#### Directory Operations

```nos
# Create a directory
(status, result) = Dir.create("/path/to/dir")

# Create nested directories (like mkdir -p)
(status, result) = Dir.createAll("/path/to/nested/dirs")

# Check if directory exists
(status, exists) = Dir.exists("/path/to/dir")

# List directory contents
(status, files) = Dir.list("/path/to/dir")

# Remove an empty directory
(status, result) = Dir.remove("/path/to/dir")

# Remove directory and all contents recursively
(status, result) = Dir.removeAll("/path/to/dir")
```

#### Complete Example

```nos
main() = {
    # Create a directory
    Dir.create("/tmp/myapp")

    # Write some files
    File.writeAll("/tmp/myapp/config.txt", "setting=value")
    File.writeAll("/tmp/myapp/data.txt", "some data")

    # List the directory
    (s, files) = Dir.list("/tmp/myapp")
    println("Files: ")
    println(files)

    # Read a file
    (s, content) = File.readAll("/tmp/myapp/config.txt")
    println("Config: " ++ content)

    # Clean up
    Dir.removeAll("/tmp/myapp")
}
```

### HTTP Client

The HTTP client supports all standard HTTP verbs and returns a response record with `status`, `headers`, and `body` fields.

```nos
# HTTP GET request
(status, resp) = Http.get("https://api.example.com/data")

# HTTP POST with body
(status, resp) = Http.post("https://api.example.com/data", "{\"key\": \"value\"}")

# HTTP PUT
(status, resp) = Http.put("https://api.example.com/resource", "{\"update\": \"data\"}")

# HTTP DELETE
(status, resp) = Http.delete("https://api.example.com/resource/123")

# HTTP PATCH
(status, resp) = Http.patch("https://api.example.com/resource", "{\"partial\": \"update\"}")

# HTTP HEAD (returns headers only, no body)
(status, resp) = Http.head("https://api.example.com/data")

# Generic HTTP request with custom headers
headers = [("Content-Type", "application/json"), ("Authorization", "Bearer token")]
(status, resp) = Http.request("POST", "https://api.example.com/data", headers, "{\"data\": 1}")
```

Response record fields:
- `status`: HTTP status code (Int, e.g., 200)
- `headers`: List of (name, value) tuples
- `body`: Response body as String

```nos
main() = {
    (status, resp) = Http.get("https://httpbin.org/ip")
    println("Status: " ++ show(resp.status))
    println("Body: " ++ resp.body)
}
```

### Parallel I/O

I/O operations run concurrently with other processes. Multiple I/O operations can be in flight simultaneously:

```nos
# Spawn multiple HTTP requests in parallel
httpWorker(parent, id) = {
    (status, data) = Http.get("https://httpbin.org/delay/1")
    parent <- ("done", id, data.status)
}

main() = {
    me = self()

    # All 5 requests run in parallel (~1-2s total, not 5s)
    spawn(() => httpWorker(me, 1))
    spawn(() => httpWorker(me, 2))
    spawn(() => httpWorker(me, 3))
    spawn(() => httpWorker(me, 4))
    spawn(() => httpWorker(me, 5))

    # Collect results...
}
```

### Error Handling

I/O operations return result tuples:
- Success: `("ok", value)` for Http.get, or the content directly for File operations
- Error: `("error", message)` when operations fail

```nos
main() = {
    result = File.readAll("/nonexistent/file.txt")
    # result will be an error if file doesn't exist
}
```

## String Encoding

Nostos provides utilities for encoding and decoding strings.

### Base64

```nos
# Encode string to Base64
encoded = Base64.encode("Hello, World!")
# encoded = "SGVsbG8sIFdvcmxkIQ=="

# Decode Base64 back to string (returns result tuple)
(status, decoded) = Base64.decode(encoded)
# status = "ok", decoded = "Hello, World!"
```

### URL Encoding

```nos
# Encode string for use in URLs
encoded = Url.encode("Hello World! How are you?")
# encoded = "Hello%20World%21%20How%20are%20you%3F"

# Decode URL-encoded string (returns result tuple)
(status, decoded) = Url.decode(encoded)
# status = "ok", decoded = "Hello World! How are you?"
```

### UTF-8 Byte Operations

```nos
# Convert string to list of UTF-8 bytes
bytes = Encoding.toBytes("Hi")
# bytes = [72, 105]

# Convert bytes back to string (returns result tuple)
(status, str) = Encoding.fromBytes([72, 105])
# status = "ok", str = "Hi"
```

## JSON

Nostos provides comprehensive JSON support for parsing, serialization, and typed deserialization.

### Parsing JSON

```nos
# Parse a JSON string into a Json value
json = jsonParse("{\"name\": \"Alice\", \"age\": 30}")

# The Json type is a variant:
# type Json = Null | Bool(Bool) | Number(Float) | String(String)
#           | Array(List[Json]) | Object(List[(String, Json)])
```

### Stringify JSON

```nos
# Convert Json back to a string
json = jsonParse("{\"x\": 1, \"y\": 2}")
str = jsonStringify(json)  # "{\"x\":1,\"y\":2}"
```

### Typed Deserialization with fromJson

The `fromJson[T](json)` builtin converts JSON to typed Nostos values. This is the recommended way to work with JSON data.

```nos
# Define your types
type Person = { name: String, age: Int }

# Parse and convert to typed value
json = jsonParse("{\"name\": \"Alice\", \"age\": 30}")
person = fromJson[Person](json)

println(person.name)  # "Alice"
println(person.age)   # 30
```

Note: `jsonToType[T]` is an alias for `fromJson[T]` and works identically.

### Supported Types

`fromJson` supports all Nostos types:

```nos
# All numeric types
type Sizes = {
    i8: Int8, i16: Int16, i32: Int32, i64: Int,
    u8: UInt8, u16: UInt16, u32: UInt32, u64: UInt64,
    f32: Float32, f64: Float
}

# Records with any field types
type Point = { x: Int, y: Int }
type Config = { name: String, enabled: Bool, value: Float }

# Sum types (variants)
type Result = Ok(Int) | Err(String)
type Option = Some(String) | None

# Nested types
type Address = { city: String, zip: Int }
type Person = { name: String, address: Address }

# Tuples (represented as JSON arrays)
type Pair = { data: (Int, String) }
```

### JSON Format for Variants

Variants use a JSON object with the constructor name as the key. All fields use `_0`, `_1`, etc. for positional values:

```nos
type Result = Ok(Int) | Err(String)

# Ok(42) is represented as: {"Ok": {"_0": 42}}
# Err("fail") is represented as: {"Err": {"_0": "fail"}}

json = jsonParse("{\"Ok\": {\"_0\": 42}}")
result = fromJson[Result](json)  # Ok(42)

# Multi-field variants use _0, _1, _2...:
type Point = Coord(Int, Int)
# Coord(10, 20) is: {"Coord": {"_0": 10, "_1": 20}}

# Unit variants (no payload) use null or empty object:
type Status = Active | Pending | Done
# Active is: {"Active": null} or {"Active": {}}
```

### Error Handling

`fromJson` throws catchable exceptions on errors:

```nos
type Person = { name: String, age: Int }

main() = {
    # Missing required field
    json = jsonParse("{\"name\": \"Alice\"}")

    try {
        person = fromJson[Person](json)
        "success"
    } catch { e -> "Error: " ++ e }
    # Returns: "Error: Missing field: age"
}
```

Common errors:
- `Missing field: <name>` - Required field not in JSON
- `Unknown constructor: <name>` - Variant constructor doesn't exist
- `Unknown type: <name>` - Type not defined
- `Expected Json Object, found <type>` - Wrong JSON structure

### Round-Trip Example

```nos
type User = { id: Int, name: String, active: Bool }

main() = {
    # Create a user
    user = User { id: 1, name: "Bob", active: true }

    # Convert to JSON using reflect
    json = reflect(user)
    jsonStr = jsonStringify(json)

    # Parse back to typed value
    parsed = jsonParse(jsonStr)
    user2 = fromJson[User](parsed)

    user == user2  # true
}
```

### Advanced: Type Introspection and Dynamic Construction

Nostos provides low-level builtins for type introspection and dynamic value construction.

#### typeInfo - Get Type Metadata

The `typeInfo(typeName)` builtin returns type metadata as a native Map:

```nos
type Person = { name: String, age: Int }

main() = {
    info = typeInfo("Person")

    # Returns Map with keys:
    # "name" -> "Person"
    # "kind" -> "record" or "variant"
    # "fields" -> List of Maps with "name" and "type"
    # "constructors" -> List of Maps (for variants)

    kind = Map.get(info, "kind")
    println(kind)  # "record"
}
```

#### makeRecordByName and makeVariantByName - Dynamic Construction

Construct typed values dynamically when the type name is known at runtime:

```nos
type Person = { name: String, age: Int }
type Result = Ok(Int) | Err(String)

main() = {
    # Construct a record dynamically
    fields = %{"name": "Alice", "age": 30}
    person = makeRecordByName("Person", fields)
    println(person.name)  # "Alice"

    # Construct a variant dynamically
    result = makeVariantByName("Result", "Ok", %{"_0": 42})
    # result is Ok(42)
}
```

#### fromJsonValue - Dynamic JSON Conversion

For scenarios where the type name is determined at runtime, use `fromJsonValue` from the stdlib:

```nos
import stdlib.json

# When type name is known at runtime
typeName = "Person"
json = jsonParse("{\"name\": \"Charlie\", \"age\": 35}")
person = stdlib.json.fromJsonValue(typeName, json)
```

These builtins enable metaprogramming scenarios like:
- Generic serialization/deserialization libraries
- ORM-style database mapping
- Dynamic form generation from type schemas

## UUID

Generate and validate UUIDs:

```nos
# Generate a random UUID v4
id = Uuid.v4()
# id = "550e8400-e29b-41d4-a716-446655440000"

# Check if a string is a valid UUID
valid = Uuid.isValid(id)         # true
invalid = Uuid.isValid("not-uuid")  # false
```

## Cryptography

Cryptographic hashing and password functions:

### Hashing

```nos
# SHA-256 hash (returns hex string)
hash = Crypto.sha256("hello")
# "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

# SHA-512 hash
hash512 = Crypto.sha512("hello")

# MD5 hash (for compatibility, not secure)
md5 = Crypto.md5("hello")
# "5d41402abc4b2a76b9719d911017c592"
```

### Password Hashing

```nos
# Hash a password with bcrypt (cost 10 is recommended for production)
hash = Crypto.bcryptHash("mypassword", 10)
# "$2b$10$..."

# Verify a password against a hash
valid = Crypto.bcryptVerify("mypassword", hash)  # true
wrong = Crypto.bcryptVerify("wrongpass", hash)   # false
```

### Random Bytes

```nos
# Generate cryptographically secure random bytes (hex string)
randomHex = Crypto.randomBytes(16)
# "3fb7afff58ee14ac2dfcf0d791ded729"
```

## Regular Expressions

Pattern matching with regex:

```nos
# Check if string matches pattern
matches = Regex.matches("hello123", "[a-z]+[0-9]+")  # true

# Find first match
found = Regex.find("hello123world456", "[0-9]+")  # "123"

# Find all matches
all = Regex.findAll("a1b2c3", "[0-9]")  # ["1", "2", "3"]

# Replace first match
replaced = Regex.replace("hello world", "world", "there")
# "hello there"

# Replace all matches
replacedAll = Regex.replaceAll("a1b2c3", "[0-9]", "X")
# "aXbXcX"

# Split by pattern
parts = Regex.split("a,b;c:d", "[,;:]")  # ["a", "b", "c", "d"]

# Get capture groups
caps = Regex.captures("John Doe", "([A-Z][a-z]+) ([A-Z][a-z]+)")
# ["John Doe", "John", "Doe"]
```

## Environment Variables

Access and modify environment variables:

```nos
# Get an environment variable (returns Option)
home = Env.get("HOME")  # Some("/home/user")

# Set an environment variable
Env.set("MY_VAR", "value")

# Remove an environment variable
Env.remove("MY_VAR")

# Get all environment variables as list of tuples
all = Env.all()  # [("HOME", "/home/user"), ("PATH", "..."), ...]

# Get current working directory
cwd = Env.cwd()

# Set current working directory
Env.setCwd("/tmp")

# Get home directory
home = Env.home()  # Some("/home/user")

# Get command-line arguments
args = Env.args()  # ["arg1", "arg2", ...]

# Get platform name
platform = Env.platform()  # "linux", "macos", or "windows"
```

## Path Operations

File path manipulation:

```nos
# Join path components
path = Path.join("/home", "user")  # "/home/user"

# Get directory name
dir = Path.dirname("/home/user/file.txt")  # "/home/user"

# Get file name
name = Path.basename("/home/user/file.txt")  # "file.txt"

# Get file extension
ext = Path.extension("file.txt")  # "txt"

# Replace file extension
newPath = Path.withExtension("file.txt", "md")  # "file.md"

# Normalize path
normalized = Path.normalize("/home/../home/./user")  # "/home/user"

# Check if path is absolute
isAbs = Path.isAbsolute("/home/user")  # true
isRel = Path.isRelative("./file.txt")  # true

# Split into components
parts = Path.split("/home/user/file.txt")  # ["", "home", "user", "file.txt"]
```

## Random Numbers

Generate random values:

```nos
# Random integer in range [min, max] (inclusive)
n = Random.int(1, 100)

# Random float in [0.0, 1.0)
f = Random.float()

# Random boolean
b = Random.bool()

# Pick random element from list
item = Random.choice(["a", "b", "c"])

# Shuffle a list
shuffled = Random.shuffle([1, 2, 3, 4, 5])

# Generate random bytes (list of 0-255 values)
bytes = Random.bytes(8)  # [42, 128, 3, ...]
```

## Time Functions

Work with timestamps and dates:

```nos
# Get current Unix timestamp in milliseconds
now = Time.now()

# Get current Unix timestamp in seconds
nowSecs = Time.nowSecs()

# Format timestamp with strftime pattern
formatted = Time.format(now, "%Y-%m-%d %H:%M:%S")
# "2024-01-15 14:30:45"

# Format as UTC
utc = Time.formatUtc(now, "%Y-%m-%d %H:%M:%S")

# Parse time string (returns Option Int)
parsed = Time.parse("2024-01-15", "%Y-%m-%d")

# Extract components from timestamp
year = Time.year(now)    # 2024
month = Time.month(now)  # 1-12
day = Time.day(now)      # 1-31
hour = Time.hour(now)    # 0-23
minute = Time.minute(now) # 0-59
second = Time.second(now) # 0-59
weekday = Time.weekday(now)  # 0=Sunday, 6=Saturday

# Get timezone info
tz = Time.timezone()        # "Europe/Oslo"
offset = Time.timezoneOffset()  # Minutes from UTC

# Create timestamp from date/time
ts = Time.fromDate(2024, 1, 15)  # Midnight UTC
ts = Time.fromDateTime(2024, 1, 15, 14, 30, 0)
```

## String Functions

String manipulation (all strings are UTF-8):

```nos
# Length in characters
len = String.length("hello")  # 5

# Convert to/from char list
chars = String.chars("hi")  # ['h', 'i']
str = String.from_chars(['h', 'i'])  # "hi"

# Parsing
num = String.toInt("42")     # Some(42)
f = String.toFloat("3.14")   # Some(3.14)

# Trimming
trimmed = String.trim("  hello  ")     # "hello"
left = String.trimStart("  hello")     # "hello"
right = String.trimEnd("hello  ")      # "hello"

# Case conversion
upper = String.toUpper("hello")  # "HELLO"
lower = String.toLower("HELLO")  # "hello"

# Searching
has = String.contains("hello", "ell")   # true
starts = String.startsWith("hello", "he")  # true
ends = String.endsWith("hello", "lo")   # true
idx = String.indexOf("hello", "l")      # 2
last = String.lastIndexOf("hello", "l") # 3

# Manipulation
sub = String.substring("hello", 1, 4)  # "ell"
rep = String.replace("hello", "l", "L")  # "heLlo"
repAll = String.replaceAll("hello", "l", "L")  # "heLLo"
repeated = String.repeat("ab", 3)  # "ababab"
padded = String.padStart("42", 5, "0")  # "00042"
padEnd = String.padEnd("hi", 5, "!")   # "hi!!!"
rev = String.reverse("hello")  # "olleh"

# Splitting
lines = String.lines("a\nb\nc")  # ["a", "b", "c"]
words = String.words("a b c")   # ["a", "b", "c"]

# Check empty
empty = String.isEmpty("")  # true
```

## List Functions

Common list operations:

```nos
# Basic operations
len = List.length([1, 2, 3])  # 3
h = List.head([1, 2, 3])      # 1
t = List.tail([1, 2, 3])      # [2, 3]
last = List.last([1, 2, 3])   # 3
init = List.init([1, 2, 3])   # [1, 2]
nth = List.nth([1, 2, 3], 1)  # 2

# Modification
pushed = List.push([1, 2], 3)  # [1, 2, 3]
popped = List.pop([1, 2, 3])   # [1, 2]
sliced = List.slice([1, 2, 3, 4], 1, 3)  # [2, 3]
concat = List.concat([1, 2], [3, 4])  # [1, 2, 3, 4]
rev = List.reverse([1, 2, 3])  # [3, 2, 1]
sorted = List.sort([3, 1, 2])  # [1, 2, 3]

# Higher-order functions
mapped = List.map([1, 2, 3], x => x * 2)  # [2, 4, 6]
filtered = List.filter([1, 2, 3, 4], x => x > 2)  # [3, 4]
folded = List.fold([1, 2, 3], 0, (acc, x) => acc + x)  # 6

# Searching
anyMatch = List.any([1, 2, 3], x => x > 2)  # true
allMatch = List.all([1, 2, 3], x => x > 0)  # true
found = List.find([1, 2, 3], x => x > 1)    # 2
pos = List.position([1, 2, 3], x => x > 1)  # 1

# Other
unique = List.unique([1, 2, 2, 3])  # [1, 2, 3]
flat = List.flatten([[1, 2], [3, 4]])  # [1, 2, 3, 4]
zipped = List.zip([1, 2], ["a", "b"])  # [(1, "a"), (2, "b")]
taken = List.take([1, 2, 3, 4], 2)  # [1, 2]
dropped = List.drop([1, 2, 3, 4], 2)  # [3, 4]
ranged = List.range(1, 5)  # [1, 2, 3, 4]
replicated = List.replicate(3, "x")  # ["x", "x", "x"]

# Aggregation
sum = List.sum([1, 2, 3])       # 6
product = List.product([2, 3, 4])  # 24
```

## Map Functions

Dictionary/hash map operations:

```nos
# Create a map
m = %{"a": 1, "b": 2}

# Insert (returns new map)
m2 = Map.insert(m, "c", 3)  # %{"a": 1, "b": 2, "c": 3}

# Remove (returns new map)
m3 = Map.remove(m2, "b")  # %{"a": 1, "c": 3}

# Get value (returns Option)
val = Map.get(m, "a")  # Some(1)
missing = Map.get(m, "z")  # None

# Check key exists
has = Map.contains(m, "a")  # true

# Get all keys/values
keys = Map.keys(m)    # ["a", "b"]
vals = Map.values(m)  # [1, 2]

# Size
size = Map.size(m)  # 2
empty = Map.isEmpty(%{})  # true
```

## Set Functions

Unordered unique collections:

```nos
# Create a set
s = #{1, 2, 3}

# Insert (returns new set)
s2 = Set.insert(s, 4)  # #{1, 2, 3, 4}

# Remove (returns new set)
s3 = Set.remove(s, 2)  # #{1, 3}

# Check membership
has = Set.contains(s, 2)  # true

# Size
size = Set.size(s)  # 3
empty = Set.isEmpty(#{})  # true

# Convert to list
lst = Set.toList(s)  # [1, 2, 3]

# Set operations
union = Set.union(#{1, 2}, #{2, 3})  # #{1, 2, 3}
inter = Set.intersection(#{1, 2, 3}, #{2, 3, 4})  # #{2, 3}
diff = Set.difference(#{1, 2, 3}, #{2})  # #{1, 3}
```

## Math Functions

Mathematical operations:

```nos
# Basic math
sqrt = Math.sqrt(16.0)   # 4.0
abs = Math.abs(-5)       # 5
min = Math.min(3, 5)     # 3
max = Math.max(3, 5)     # 5
pow = Math.pow(2.0, 3.0) # 8.0

# Rounding
floor = Math.floor(3.7)  # 3.0
ceil = Math.ceil(3.2)    # 4.0
round = Math.round(3.5)  # 4.0

# Logarithms
log = Math.log(10.0)     # Natural log
log10 = Math.log10(100.0)  # 2.0

# Trigonometry
sin = Math.sin(0.0)      # 0.0
cos = Math.cos(0.0)      # 1.0
tan = Math.tan(0.0)      # 0.0
```

## PostgreSQL Database

Connect to and query PostgreSQL databases.

### Basic Connection

```nos
# Connect to local database
handle = Pg.connect("host=localhost user=postgres password=secret dbname=mydb")

# Execute query with parameters
rows = Pg.query(handle, "SELECT id, name FROM users WHERE age > $1", [18])
# Returns list of lists: [[1, "Alice"], [2, "Bob"]]

# Execute statement (INSERT, UPDATE, DELETE)
affected = Pg.execute(handle, "INSERT INTO users (name, age) VALUES ($1, $2)", ["Charlie", 25])
# Returns number of affected rows

# Close connection
Pg.close(handle)
```

### Cloud Database Providers

Nostos supports TLS connections to cloud PostgreSQL providers. Use `sslmode=require` in the connection string:

```nos
# Supabase
# Note: URL-encode special characters in password (e.g., comma -> %2C, ! -> %21)
handle = Pg.connect("postgresql://postgres:YourPass%21@db.xxxxx.supabase.co:5432/postgres?sslmode=require")

# Neon
handle = Pg.connect("postgresql://user:pass@ep-xxx-pooler.region.aws.neon.tech/neondb?sslmode=require")

# Any PostgreSQL with TLS
handle = Pg.connect("host=db.example.com user=app password=secret dbname=prod sslmode=require")
```

### Connection Pooling

Nostos automatically pools PostgreSQL connections for efficiency. Connections to the same database URL are reused from a pool rather than creating new connections each time:

```nos
# First connection creates a pool for this URL
conn1 = Pg.connect("postgresql://user:pass@localhost/mydb")
Pg.query(conn1, "SELECT 1", [])
Pg.close(conn1)  # Returns connection to pool

# Second connection reuses from pool (fast, no new connection overhead)
conn2 = Pg.connect("postgresql://user:pass@localhost/mydb")
Pg.query(conn2, "SELECT 2", [])
Pg.close(conn2)

# Multiple simultaneous connections are supported
connA = Pg.connect("postgresql://user:pass@localhost/mydb")
connB = Pg.connect("postgresql://user:pass@localhost/mydb")
connC = Pg.connect("postgresql://user:pass@localhost/mydb")
# ... use all three ...
Pg.close(connA)
Pg.close(connB)
Pg.close(connC)
```

Pooling benefits:
- **Performance**: Reuses existing connections instead of creating new ones
- **Automatic**: No configuration needed, works transparently
- **Per-URL pools**: Each unique connection string gets its own pool
- **TLS support**: Works with both local (non-TLS) and cloud (TLS) databases

### Transactions

```nos
Pg.begin(handle)
Pg.execute(handle, "UPDATE accounts SET balance = balance - 100 WHERE id = $1", [1])
Pg.execute(handle, "UPDATE accounts SET balance = balance + 100 WHERE id = $1", [2])
Pg.commit(handle)  # or Pg.rollback(handle) to cancel

# Or use Pg.transaction for automatic commit/rollback:
Pg.transaction(handle, () => {
    Pg.execute(handle, "UPDATE accounts SET balance = balance - 100 WHERE id = $1", [1])
    Pg.execute(handle, "UPDATE accounts SET balance = balance + 100 WHERE id = $1", [2])
})
```

### Prepared Statements

```nos
Pg.prepare(handle, "get_user", "SELECT * FROM users WHERE id = $1")
rows = Pg.queryPrepared(handle, "get_user", [42])
Pg.deallocate(handle, "get_user")
```

### JSON Support

JSON and JSONB columns are fully supported:

```nos
# Insert JSON data - pass Json variant or string
data = Object([("name", String("Alice")), ("age", Number(30.0))])
Pg.execute(handle, "INSERT INTO docs (data) VALUES ($1)", [data])

# Or as a string (automatically parsed as JSON for JSONB columns)
Pg.execute(handle, "INSERT INTO docs (data) VALUES ($1)", ["{\"key\": \"value\"}"])

# Query returns JSON as string - parse with jsonParse if needed
rows = Pg.query(handle, "SELECT data FROM docs", [])
jsonStr = head(head(rows))  # e.g., "{\"name\":\"Alice\",\"age\":30}"
```

### Vector Support (pgvector)

For AI/ML applications using the pgvector extension. Uses `Float32Array` which matches pgvector's native f32 format:

```nos
# Insert vector using Float32Array (recommended - native pgvector format)
embedding = Float32Array.fromList([0.1, 0.2, 0.3])
Pg.execute(handle, "INSERT INTO items (embedding) VALUES ($1)", [embedding])

# Float64Array also works (automatically converted to f32)
embedding64 = Float64Array.fromList([0.1, 0.2, 0.3])
Pg.execute(handle, "INSERT INTO items (embedding) VALUES ($1)", [embedding64])

# Query returns vectors as Float32Array
rows = Pg.query(handle, "SELECT embedding FROM items", [])
vec = head(head(rows))  # Float32Array
len = Float32Array.length(vec)
first = Float32Array.get(vec, 0)
lst = Float32Array.toList(vec)  # Convert to list

# L2 distance similarity search
query = Float32Array.fromList([0.1, 0.2, 0.3])
rows = Pg.query(handle, "SELECT id, embedding <-> $1 as distance FROM items ORDER BY distance LIMIT 10", [query])

# Cosine similarity search
rows = Pg.query(handle, "SELECT id, 1 - (embedding <=> $1) as similarity FROM items ORDER BY similarity DESC LIMIT 10", [query])
```

### Error Handling

PostgreSQL errors are thrown as exceptions:

```nos
result = try {
    conn = Pg.connect("host=invalid")
    "connected"
} catch { e ->
    "failed: " ++ show(e)  # e.g., (connection_error, PostgreSQL error: ...)
}
```

## HTTP Server

Create HTTP servers:

```nos
# Bind server to port
handle = Server.bind(8080)

# Accept incoming request (blocking)
# Returns: { id: Int, method: String, path: String, headers: [(String, String)], body: String }
request = Server.accept(handle)

# Send response
headers = [("Content-Type", "text/plain")]
Server.respond(request.id, 200, headers, "Hello, World!")

# Close server
Server.close(handle)
```

### Simple Web Server Example

```nos
handleRequest(req) = match req.path {
    "/" -> (200, "Welcome!")
    "/api/hello" -> (200, "{\"message\": \"Hello!\"}")
    _ -> (404, "Not found")
}

serverLoop(handle) = {
    req = Server.accept(handle)
    (status, body) = handleRequest(req)
    Server.respond(req.id, status, [("Content-Type", "text/plain")], body)
    serverLoop(handle)
}

main() = {
    handle = Server.bind(8080)
    println("Server running on port 8080")
    serverLoop(handle)
}
```

## Command-Line Interface

### Running Programs
```bash
nostos <file.nos>           # Run a program
nostos --help               # Show help
nostos --version            # Show version
```

### Options
```bash
nostos --no-jit <file.nos>       # Disable JIT compilation
nostos --debug <file.nos>        # Show local variables in stack traces
nostos --json-errors <file.nos>  # Output errors as JSON (for debugger integration)
```

### Debug Mode
When `--debug` is enabled, stack traces include local variable values:
```
Runtime error in /tmp/test.nos:
Panic: Index 10 out of bounds

Stack trace:
  1. crash (line 5)
       sum = 142
       arr = <List>
       x = 42
       y = 100
  2. main (line 11)
       a = 42
       b = 100
```

Heap-allocated values show type placeholders (`<List>`, `<Record>`, `<Closure>`, etc.) for safety.

### JSON Error Format
When `--json-errors` is enabled, runtime errors are output as JSON:
```json
{
  "file": "/path/to/file.nos",
  "error_type": "IndexOutOfBounds",
  "message": "Index 10 out of bounds (length 3)",
  "stack_trace": [
    { "function": "inner", "line": 3 },
    { "function": "middle", "line": 7 },
    { "function": "main", "line": 12 }
  ]
}
```

## Keywords

The following are reserved keywords:
- `if`, `then`, `else`
- `match`, `try`, `catch`
- `while`, `for`, `to`, `break`, `continue`
- `receive`, `after`
- `spawn`, `spawn_link`, `spawn_monitor`
- `self`
- `true`, `false`
- `type`, `trait`, `end`
- `when`

**Note**: `end` is used for trait definitions and cannot be used as a variable name.

## Known Limitations

### Nested List Patterns in Variants
Nested list patterns inside variant constructors don't work correctly:
```nos
# This does NOT work correctly:
flatten(List([h | t])) = ...    # [h | t] pattern fails inside List()

# Workaround: extract the inner list first
get_inner(List(lst)) = lst

flatten(lst) = match get_inner(lst) {
    [] -> []
    [h | t] -> ...
}
```

### Module-Level Value Bindings
Module-level value bindings without function syntax are not supported:
```nos
# This does NOT work:
pi = 3.14159

# Use zero-argument functions or define in main:
pi() = 3.14159
# or
main() = {
    pi = 3.14159,
    ...
}
```

