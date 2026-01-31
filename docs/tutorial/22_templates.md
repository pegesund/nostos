# Templates & Metaprogramming

Nostos provides a compile-time metaprogramming system based on templates. Templates are functions that manipulate code as data, enabling powerful abstractions like decorators, code generation, and domain-specific languages.

## Core Concepts

The template system has three key components:

1. **Quote** - Capture code as AST (Abstract Syntax Tree) data
2. **Splice** - Insert AST values back into code
3. **Templates** - Compile-time functions that transform code

## Quote: Capturing Code as Data

The `quote` keyword captures an expression as an AST value instead of evaluating it.

```nostos
main() = {
    # Capture the expression "1 + 2" as data
    ast = quote(1 + 2)
    println(ast)  # Prints the AST representation

    # Quote works with any expression
    complex = quote(x.map(y => y * 2).filter(z => z > 10))
    println(complex)
}
```

### Block Quote Syntax

For multi-line expressions, use the block form:

```nostos
main() = {
    ast = quote {
        x = 10
        y = 20
        x + y
    }
    println(ast)
}
```

The quoted expression is not evaluated - it becomes a runtime value representing the code structure.

## Splice: Inserting AST Values

The `~` operator (splice) inserts an AST value into quoted code:

```nostos
main() = {
    # Create an AST for a number
    num = quote(42)

    # Splice it into a larger expression
    expr = quote(~num * 2)
    # expr represents: 42 * 2

    println(expr)
}
```

Splice is the inverse of quote - it takes AST data and makes it part of the code being constructed.

## Template Functions

Templates are compile-time functions declared with the `template` keyword. They receive AST values and return transformed AST.

```nostos
# A template that wraps an expression to double its result
template double(body) = quote {
    result = ~body
    result * 2
}

# Use the template directly
main() = {
    value = double(21)  # Expands to: result = 21; result * 2
    println(value)      # 42
}
```

### How Templates Work

1. When a template is called, arguments are converted to AST values
2. The template body executes at compile time
3. Splices substitute the argument AST into the quoted result
4. The resulting AST replaces the template call in the compiled code

### Templates with Multiple Parameters

```nostos
# Template that creates a range check
template inRange(value, min, max) = quote {
    v = ~value
    v >= ~min && v <= ~max
}

main() = {
    x = 50
    if inRange(x, 0, 100) {
        println("In range!")
    }
}
```

## Decorators

Decorators are a convenient syntax for applying templates to function definitions. Use `@decorator` before a function:

```nostos
template double(body) = quote {
    result = ~body
    result * 2
}

@double
getValue() = 21

main() = getValue()  # Returns 42
```

The decorator receives the function body as its first argument and returns the transformed body.

### Decorators with Arguments

Decorators can take additional arguments after the implicit body parameter:

```nostos
# Multiply result by a factor
template multiply(body, factor) = quote {
    result = ~body
    result * ~factor
}

# Add a fixed amount
template add(body, amount) = quote {
    result = ~body
    result + ~amount
}

@add(35)
@multiply(10)
compute() = 7  # 7 * 10 = 70, then + 35 = 105

main() = compute()  # Returns 105
```

### Multiple Decorators

When stacking decorators, they apply bottom-up (innermost first):

```nostos
template double(body) = quote {
    result = ~body
    result * 2
}

template triple(body) = quote {
    result = ~body
    result * 3
}

# Applied as: triple(double(7))
# First double: 7 * 2 = 14
# Then triple: 14 * 3 = 42
@triple
@double
getValue() = 7

main() = getValue()  # 42
```

## Practical Examples

### Logging Decorator

```nostos
template logged(body) = quote {
    println("Function called")
    result = ~body
    println("Function returned")
    result
}

@logged
add(a: Int, b: Int) = a + b

main() = {
    result = add(1, 2)
    println(result)
}
# Output:
# Function called
# Function returned
# 3
```

### Timing Decorator

```nostos
template timed(body) = quote {
    start = Time.now()
    result = ~body
    elapsed = Time.now() - start
    println("Elapsed: " ++ elapsed.toString() ++ "ms")
    result
}

@timed
slowComputation() = {
    # Simulate work
    sum = 0
    i = 0
    while i < 1000000 {
        sum = sum + i
        i = i + 1
    }
    sum
}

main() = slowComputation()
```

### Default Value Decorator

```nostos
template withDefault(body, default) = quote {
    result = ~body
    if result == () { ~default } else { result }
}

@withDefault(0)
parseNumber(s: String) = {
    # Returns () on parse failure
    Int.parse(s)
}

main() = {
    println(parseNumber("42"))    # 42
    println(parseNumber("abc"))   # 0 (default)
}
```

### Retry Decorator

```nostos
template retry(body, times) = quote {
    attempts = 0
    result = ()
    success = false
    while attempts < ~times && !success {
        attempts = attempts + 1
        result = try { ~body } catch { () }
        success = result != ()
    }
    result
}

@retry(3)
fetchData() = {
    # Might fail, will retry up to 3 times
    Http.get("https://api.example.com/data")
}
```

## Template Patterns

### Code Generation

Templates can generate repetitive code:

```nostos
template getter(field) = quote {
    self.~field
}

template setter(field) = quote {
    self.~field = value
    self
}
```

### Validation

```nostos
template validate(body, predicate, message) = quote {
    value = ~body
    if !(~predicate(value)) {
        throw(~message)
    }
    value
}

@validate(x => x > 0, "must be positive")
getPositive() = readInt()
```

### Memoization

```nostos
template memoize(body) = quote {
    # Note: simplified example
    cached = None
    if cached == None {
        cached = Some(~body)
    }
    cached.unwrap()
}
```

## AST Types

When working with templates, the AST values have these kinds:

- **Literals**: Int, Float, String, Bool, Char, Unit
- **Variables**: Var(name)
- **Operations**: BinOp, UnaryOp
- **Calls**: Call, MethodCall
- **Access**: FieldAccess, Index
- **Structures**: Lambda, Block, If, Match
- **Collections**: List, Tuple, Record, Map
- **Patterns**: PatternVar, PatternLit, PatternConstructor
- **Definitions**: FnDef, Let

## Best Practices

1. **Keep templates simple** - Complex logic is harder to debug at compile time
2. **Use meaningful names** - Template parameters should describe their role
3. **Document behavior** - Explain what transformation the template performs
4. **Test templates** - Create test cases for each template
5. **Prefer composition** - Build complex behavior from simple templates

## Comparison with Other Languages

| Feature | Nostos | Rust | Lisp |
|---------|--------|------|------|
| Syntax | `template`, `quote`, `~` | `macro_rules!`, proc macros | `defmacro`, `'`, `,` |
| Hygiene | Automatic | Manual | Manual |
| Phase | Compile-time | Compile-time | Various |
| AST Access | Direct | TokenStream | S-expressions |

## Limitations

- Templates execute at compile time, not runtime
- Spliced values must be valid AST
- Templates cannot access runtime values
- Recursive templates must terminate

Templates provide a powerful way to extend the language without modifying the compiler. Use them to create domain-specific abstractions and eliminate boilerplate code.
