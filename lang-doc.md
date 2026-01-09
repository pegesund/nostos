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
describe(n) = match n
    0 -> "zero"
    1 -> "one"
    _ -> "many"
end

# With guards
abs(n) = match n
    x when x >= 0 -> x
    x -> -x
end
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
```

## Control Flow

### If Expression
```nos
max(a, b) = if a > b then a else b

sign(n) = if n > 0 then 1 else if n < 0 then -1 else 0
```

### Match Expression
```nos
classify(n) = match n
    0 -> "zero"
    1 -> "one"
    2 -> "two"
    _ -> "many"
end
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

## Higher-Order Functions

```nos
# Function as parameter
apply(f, x) = f(x)
twice(f, x) = f(f(x))

# Function as return value
add(n) = x => x + n
add5 = add(5)

# Common patterns
map(_, []) = []
map(f, [x | xs]) = [f(x) | map(f, xs)]

filter(_, []) = []
filter(p, [x | xs]) = if p(x) then [x | filter(p, xs)] else filter(p, xs)

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
    receive
        n -> n                      # Receive message, return it
    end
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
receive
    msg -> handle(msg)
end

# Pattern matching in receive
receive
    (a, b) -> a + b
    n -> n * 2
end

# Multiple receives
receive
    a -> receive
        b -> a + b
    end
end
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
    receive
        a -> receive
            b -> a + b    # Returns 25 + 100 = 125
        end
    end
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
nostos --no-jit <file.nos>       # Disable JIT compilation (for debugging)
nostos --json-errors <file.nos>  # Output errors as JSON (for debugger integration)
```

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
- `match`, `end`
- `while`, `for`, `to`, `break`, `continue`
- `receive`
- `spawn`, `spawn_link`, `spawn_monitor`
- `self`
- `true`, `false`
- `type`, `trait`, `impl`
- `when`

**Note**: `end` is a keyword and cannot be used as a variable name (use `stop`, `limit`, etc. instead).

## Known Limitations

### Nested List Patterns in Variants
Nested list patterns inside variant constructors don't work correctly:
```nos
# This does NOT work correctly:
flatten(List([h | t])) = ...    # [h | t] pattern fails inside List()

# Workaround: extract the inner list first
get_inner(List(lst)) = lst

flatten(lst) = match get_inner(lst)
    [] -> []
    [h | t] -> ...
end
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

