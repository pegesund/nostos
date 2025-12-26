# Language Basics

Nostos syntax is designed to be minimal and readable. It draws inspiration from Ruby, Rust, and Elixir.

## Getting Started

Run Nostos programs from the command line:

```bash
# Run a program
nostos myprogram.nos

# Start the interactive REPL
nostos

# Show help
nostos --help
```

### Command Line Options

- `--threads N` - Use N worker threads (default: all CPUs)
- `--no-jit` - Disable JIT compilation (for debugging)
- `--debug` - Show local variables in stack traces
- `--profile` - Enable function call profiling
- `--json-errors` - Output errors as JSON (for IDE integration)

### Hello World

Every Nostos program starts with a `main()` function:

```nostos
# hello.nos
main() = {
    println("Hello, World!")
}
```

## Literals

```nostos
# Numbers
42              # Int
3.14            # Float
1_000_000       # Underscores for readability
0xFF            # Hex

# Boolean
true
false

# Strings & Chars
"Hello World"   # String (double-quoted)
'{"key": "val"}'# String (single-quoted, for embedding double quotes)
'a'             # Char (single char in single quotes)

# Unit (Empty value)
()
```

## Comments

Use `#` for single-line comments.

```nostos
# This is a comment
x = 10 # Comment at end of line
```

## Bindings: Immutable & Mutable

By default, bindings in Nostos are **immutable**. Use `var` to create mutable bindings.

```nostos
# Immutable binding (default)
x = 10
x = 10      # OK: same value (acts as assertion)
# x = 20    # ERROR: Assertion failed: 10 != 20

# Mutable binding
var count = 0
count = count + 1   # OK
```

**Key insight:** Rebinding an immutable variable to the *same* value succeeds (useful for assertions), but rebinding to a *different* value causes a runtime error. This is pattern matching semantics.

```nostos
# Mutable can shadow immutable
x = 10              # immutable
var x = 15          # new mutable binding shadows the old one
x = 20              # OK, x is now mutable
```

## Typed Bindings

Nostos supports optional type annotations using Java-style syntax, where the type comes before the variable name. Types are inferred when not specified.

```nostos
# With explicit types
Int x = 42
Float pi = 3.14159
String name = "Alice"

# Mutable with type
var Int counter = 0
counter = counter + 1

# Generic types
List[Int] numbers = [1, 2, 3]

# Without types (inferred)
y = 100             # inferred as Int
greeting = "Hello"  # inferred as String
```

## Blocks & Expressions

Everything in Nostos is an expression. Blocks are defined with `{}`. The last expression in a block is the return value.

```nostos
main() = {
    x = 10
    y = 20

    # The result of x + y is returned automatically
    x + y
}
```

## Control Flow

### If-Then-Else

Conditionals use `if...then...else` syntax. They are expressions that return values.

```nostos
# Simple conditional expression
max(a, b) = if a > b then a else b

# Multi-line with blocks
abs(x) = if x < 0 then {
    -x
} else {
    x
}

# Chained conditions
grade(score) = if score >= 90 then "A"
    else if score >= 80 then "B"
    else if score >= 70 then "C"
    else "F"
```

### For Loops

Use `for i = start to end` to iterate over a range.

```nostos
main() = {
    var sum = 0
    for i = 1 to 10 {
        sum = sum + i
    }
    sum  # Returns 45 (1+2+...+9, excludes 10)
}
```

### While Loops

Use `while condition { ... }` for condition-based loops.

```nostos
main() = {
    var sum = 0
    var i = 1
    while i <= 10 {
        sum = sum + i
        i = i + 1
    }
    sum  # Returns 55
}
```

### Break and Continue

Use `break` to exit a loop early and `continue` to skip to the next iteration.

```nostos
# Find first occurrence
findFirst(arr, target) = {
    var result = -1
    for i = 0 to arr.length() {
        if arr[i] == target then {
            result = i
            break
        }
    }
    result
}

# Sum only positive numbers
sumPositive(arr) = {
    var total = 0
    for i = 0 to arr.length() {
        if arr[i] < 0 then continue
        total = total + arr[i]
    }
    total
}
```

## Assertions

Use `assert` and `assert_eq` to verify conditions during development and testing.

```nostos
main() = {
    x = 5

    # Assert a condition is true
    assert(x > 0)
    assert(x < 10)

    # Assert two values are equal
    assert_eq(x * 2, 10)
    assert_eq([1, 2, 3], [1, 2, 3])

    # Useful in tests
    result = some_function()
    assert_eq(result, expected_value)
}
```

**Tip:** Assertions throw exceptions when they fail, making them ideal for writing tests and validating assumptions in your code.
