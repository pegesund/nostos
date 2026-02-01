# Templates & Metaprogramming

Nostos provides a compile-time metaprogramming system based on templates.

## Core Concepts

1. **Quote** - Capture code as AST data: `quote { expr }`
2. **Splice** - Insert AST values: `~expr`
3. **Templates** - Compile-time functions: `template name(params) = ...`

## Example 1: Simple Function Wrapper

The most basic use - wrap a function body with extra behavior:

```nostos
template double(fn) = quote {
    result = ~fn.body
    result * 2
}

@double
getValue() = 21

main() = getValue()  # Returns 42
```

The decorator receives the full function and uses `~fn.body` to splice in the original body.

## Example 2: Logging Decorator

Add automatic logging to any function:

```nostos
template logged(fn) = quote {
    println(">>> Entering " ++ ~fn.name)
    result = ~fn.body
    println("<<< Exiting " ++ ~fn.name)
    result
}

@logged
add(a: Int, b: Int) = a + b

main() = add(10, 20)
# Output:
# >>> Entering add
# <<< Exiting add
# 30
```

Available function metadata:
- `~fn.name` - Function name as String
- `~fn.params` - List of Maps with keys: `"name"` (String), `"type"` (String), `"ty"` (String)
- `~fn.body` - Function body AST
- `~fn.returnType` - Return type as String

Note: For parameters, both `p.get("type")` and `p.get("ty")` return the type string. The `"ty"` key exists for consistency with type field access.

## Example 3: Parameter Validation

Use `~param(n)` to reference function parameters:

```nostos
template validatePositive(fn) = quote {
    if ~param(0) <= 0 {
        panic(~fn.name ++ ": first argument must be positive")
    }
    ~fn.body
}

@validatePositive
square(n: Int) = n * n

main() = square(5)  # Returns 25
# square(-5) would panic: "square: first argument must be positive"
```

The `~param(n)` shorthand references the n-th parameter of the decorated function. It's equivalent to `~toVar(fn.params[n].name)`.

## Example 4: Auto-Generate Getters

Generate accessor functions for all fields of a type:

```nostos
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

This generates `get_x(r: Point) = r.x` and `get_y(r: Point) = r.y`.

## Example 5: Fluent Builder Pattern

Generate setter methods for immutable updates:

```nostos
template builder(typeDef) = quote {
    ~typeDef
    ~typeDef.fields.map(f =>
        eval("with_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ", v: " ++ f.ty ++ ") = " ++
             ~typeDef.name ++ "(" ++ f.name ++ ": v)"))
}

@builder
type Config = Config { timeout: Int }

main() = {
    c = Config(timeout: 100)
    c2 = with_timeout(c, 200)
    c2.timeout  # 200
}
```

## Example 6: Auto-Generate toString

Create a string representation of any type:

```nostos
template stringify(typeDef) = quote {
    ~typeDef
    ~eval("toString(v: " ++ ~typeDef.name ++ ") = \"" ++ ~typeDef.name ++
          "(\" ++ show(v." ++ ~typeDef.fields[0].name ++
          ") ++ \", \" ++ show(v." ++ ~typeDef.fields[1].name ++ ") ++ \")\"")
}

@stringify
type Point = Point { x: Int, y: Int }

main() = {
    p = Point(x: 10, y: 20)
    toString(p)  # "Point(10, 20)"
}
```

## Example 7: Conditional Code Generation

Use `~if` to generate different code based on compile-time values:

```nostos
template maybeDouble(fn, shouldDouble) = quote {
    result = ~fn.body
    ~if ~shouldDouble { quote(result * 2) } else { quote(result) }
}

@maybeDouble(true)
getValue() = 21

@maybeDouble(false)
getOther() = 100

main() = getValue() + getOther()  # 42 + 100 = 142
```

### Compile-Time Arithmetic and Comparisons

Template arguments support arithmetic and comparison operations that evaluate at compile time:

```nostos
template multiply(fn, factor) = quote {
    ~fn.body * ~factor
}

@multiply(2 + 3)  # factor evaluates to 5 at compile time
getValue() = 10

main() = getValue()  # Returns 50
```

Comparisons also work at compile time:

```nostos
template maybeDouble(fn, threshold) = quote {
    ~if ~threshold > 10 then ~fn.body * 2 else ~fn.body
}

@maybeDouble(15)  # 15 > 10, so body is doubled
getValue() = 21

@maybeDouble(5)   # 5 <= 10, so body unchanged
getOther() = 21

main() = getValue() + getOther()  # 42 + 21 = 63
```

Supported compile-time operations:
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`

## Example 8: Feature Flags

Enable/disable features at compile time:

```nostos
template featureFlag(fn, enabled, errorMsg) = quote {
    ~if ~enabled {
        quote { ~fn.body }
    } else {
        quote { panic(~errorMsg) }
    }
}

@featureFlag(true, "Beta feature disabled")
betaFeature() = "Active!"

@featureFlag(false, "Experimental feature disabled")
experimentalFeature() = "Never runs"

main() = betaFeature()  # "Active!"
```

## Example 9: Unique Names with Gensym

Avoid naming collisions with `gensym`. The `gensym` function generates unique identifiers that increment for each call:

```nostos
template withHelper(typeDef) = quote {
    ~typeDef
    ~eval(~gensym("helper") ++ "() = 42")
}

@withHelper
type A = A {}

@withHelper
type B = B {}

# Generates: helper_0() and helper_1()
main() = helper_0() + helper_1()  # 84
```

### Gensym with Variable Bindings

You can use `gensym` inside `eval` to generate unique variable bindings:

```nostos
template wrapValue(fn) = quote {
    ~eval(~gensym("v") ++ " = 42")
}

@wrapValue
getValue() = 0  # Original body replaced

main() = getValue()  # Returns 42
```

This generates code like `v_0 = 42` and returns the bound value. This is useful for creating temporary variables in generated code without risking name collisions with user code.

## Example 10: Compile-Time Computation

Use `comptime` to execute code at compile time:

### String syntax
```nostos
template computed(fn, multiplier) = quote {
    defaultValue = ~comptime("21 * " ++ ~multiplier)
    ~fn.body + defaultValue
}

@computed("2")
getValue(x: Int) = x

main() = getValue(0)  # 42
```

### Block syntax
```nostos
template computed(fn, useSquare) = quote {
    result = ~comptime({
        base = 10
        if ~useSquare { base * base } else { base * 2 }
    })
    result
}

@computed(true)
getSquare() = 0

main() = getSquare()  # 100
```

## Example 11: Exception Handling

Use `try/catch` in templates to add error handling:

```nostos
template withFallback(fn, fallback) = quote {
    try {
        ~fn.body
    } catch {
        _ -> ~fallback
    }
}

@withFallback("error occurred")
riskyOperation() = {
    throw("something went wrong")
}

@withFallback(0)
safeComputation() = 42

main() = {
    safeComputation()  # Returns 42
    # riskyOperation() would return "error occurred"
}
```

### Retry Pattern

Build a retry decorator that attempts the operation multiple times:

```nostos
# Fixed 3-retry version
template retry3(fn) = quote {
    try {
        ~fn.body
    } catch {
        _ -> try {
            ~fn.body
        } catch {
            _ -> ~fn.body  # Last attempt, let it throw
        }
    }
}

mvar attempts: Int = 0

@retry3
unreliableService() = {
    attempts = attempts + 1
    if attempts < 3 { throw("temporary failure") }
    "success on attempt " ++ show(attempts)
}

main() = unreliableService()  # "success on attempt 3"
```

For parameterized retry counts, use compile-time conditionals:

```nostos
template retry(fn, times) = quote {
    ~if ~times == 1 {
        quote { ~fn.body }
    } else {
        ~if ~times == 2 {
            quote {
                try { ~fn.body } catch { _ -> ~fn.body }
            }
        } else {
            quote {
                try { ~fn.body } catch { _ ->
                    try { ~fn.body } catch { _ -> ~fn.body }
                }
            }
        }
    }
}

@retry(3)
flaky() = someUnreliableCall()
```

## Type Introspection Reference

For type decorators:
- `~typeDef.name` - Type name as String
- `~typeDef.fields` - List of fields (for single-constructor types like `type Point = Point { x: Int }`)
- `~typeDef.fields[i].name` - Field name at index i
- `~typeDef.fields[i].ty` - Field type at index i (note: `ty`, not `type`)
- `~typeDef.typeParams` - Generic type parameters

## Template Functions Reference

| Function | Purpose | Example |
|----------|---------|---------|
| `quote { ... }` | Capture code as AST | `quote { x + 1 }` |
| `~expr` | Splice AST value | `~fn.body` |
| `eval("code")` | Parse and compile string as code | `eval("foo() = 42")` |
| `param(n)` | Reference n-th function parameter | `param(0)` → first parameter as variable |
| `toVar(string)` | Convert string to variable reference | `toVar(fn.params[0].get("name"))` |
| `gensym("prefix")` | Generate unique identifier | `gensym("temp")` → `"temp_0"` |
| `comptime("code")` | Execute code at compile time | `comptime("1 + 2")` → `3` |
| `comptime({ block })` | Execute block at compile time | `comptime({ if x { 1 } else { 2 } })` |
| `try { } catch { }` | Exception handling in generated code | `try { ~fn.body } catch { _ -> fallback }` |

### Compile-Time Operations

These operations evaluate at compile time when used in template arguments:

| Operation | Example | Result |
|-----------|---------|--------|
| Arithmetic | `@multiply(2 + 3)` | `factor` = 5 |
| Comparison | `~if ~n > 10` | Boolean at compile time |
| String concat | `~gensym("x") ++ "_val"` | `"x_0_val"` |

## Best Practices

1. **Start simple** - Get basic templates working before adding complexity
2. **Use `eval` for code generation** - When you need to generate function definitions dynamically
3. **Use `gensym` to avoid collisions** - When generating helper functions
4. **Test incrementally** - Verify each generated function works
5. **Document what's generated** - Comment the expected output
