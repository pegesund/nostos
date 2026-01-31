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

## Example 2: Accessing Function Metadata

Templates can inspect function name, parameters, and return type:

```nostos
template debugCall(fn) = quote {
    println("Calling: " ++ ~fn.name)
    result = ~fn.body
    println("Done: " ++ ~fn.name)
    result
}

@debugCall
add(a: Int, b: Int) = a + b

main() = add(1, 2)
# Output:
# Calling: add
# Done: add
# 3
```

Available fields:
- `~fn.name` - Function name as String
- `~fn.params` - List of {name, type} records
- `~fn.body` - Function body AST
- `~fn.returnType` - Return type as String

## Example 3: Conditional Code Generation

Use `~if` to generate different code based on compile-time values:

```nostos
template cached(fn, useCache) = quote {
    ~if ~useCache {
        quote {
            # With caching
            key = ~fn.name
            cached = Cache.get(key)
            if cached != () { cached }
            else {
                result = ~fn.body
                Cache.set(key, result)
                result
            }
        }
    } else {
        quote { ~fn.body }
    }
}

@cached(true)
expensiveCalc() = computeSomething()

@cached(false)
cheapCalc() = 1 + 1
```

## Example 4: Type Decorator with Getter Generation

Generate accessor functions for all fields of a type:

```nostos
template withGetters(typeDef) = quote {
    ~typeDef
    # For single-constructor types, .fields returns field list directly
    ~typeDef.fields.map(f =>
        eval("get_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ") = r." ++ f.name)
    )
}

@withGetters
type Point = Point { x: Int, y: Int }

main() = {
    p = Point(x: 10, y: 20)
    get_x(p) + get_y(p)  # 30
}
```

This generates `get_x(r: Point) = r.x` and `get_y(r: Point) = r.y`.

## Example 5: Builder Pattern with Setters

Generate both getters and setters:

```nostos
template withAccessors(typeDef) = quote {
    ~typeDef
    # Generate getters
    ~typeDef.fields.map(f =>
        eval("get_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ") = r." ++ f.name)
    )
    # Generate setters (return new instance with just one field changed)
    # Note: f.ty gives field type (not f.type which is a keyword)
    ~typeDef.fields.map(f =>
        eval("set_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ", v: " ++ f.ty ++ ") = " ++
             ~typeDef.name ++ "(" ++ f.name ++ ": v)")
    )
}

@withAccessors
type Config = Config { timeout: Int, retries: Int }

main() = {
    c = Config(timeout: 30, retries: 3)
    c2 = set_timeout(c, 60)
    get_timeout(c2)  # 60
}
```

## Example 6: Compile-Time Feature Flags

Use templates to enable/disable features at compile time:

```nostos
# Compile-time check - generates different code based on flag
template featureFlag(fn, enabled, errorMsg) = quote {
    ~if ~enabled {
        quote { ~fn.body }
    } else {
        quote { panic(~errorMsg) }
    }
}

@featureFlag(true, "Beta feature disabled")
betaFeature() = "Beta feature is active!"

@featureFlag(false, "Experimental feature disabled")
experimentalFeature() = "Never runs"

main() = betaFeature()  # Works: "Beta feature is active!"
# experimentalFeature() would panic: "Experimental feature disabled"
```

## Example 7: Runtime Parameter Validation with toVar

Use `toVar` to convert a parameter name string into a variable reference:

```nostos
# toVar converts fn.params[0].name (String) to a variable reference
template nonNegative(fn) = quote {
    if ~toVar(fn.params[0].name) < 0 {
        panic("Value must be non-negative")
    }
    ~fn.body
}

@nonNegative
mySqrt(n: Int) = n * n

main() = {
    mySqrt(4)    # Returns 16
    # mySqrt(-5) # Would panic: "Value must be non-negative"
}
```

The `toVar` function takes a String and returns a Var AST node, allowing you to reference variables dynamically in generated code.

## Example 8: Unique Variable Names with Gensym

Use gensym to avoid naming collisions in generated code:

```nostos
template withTempVar(typeDef) = quote {
    ~typeDef
    ~eval(~gensym("temp") ++ "_helper() = 42")
}

@withTempVar
type A = A {}

@withTempVar
type B = B {}

# Generates: temp_0_helper() and temp_1_helper()
# No naming collision!
```

## Example 9: Compile-Time Code Execution

Use `comptime` to execute arbitrary Nostos code at compile time. Two syntaxes are supported:

### String syntax: `comptime("code")`

```nostos
template withComputedDefault(fn, multiplier) = quote {
    # Compute at compile time: 21 * 2 = 42
    defaultValue = ~comptime("21 * " ++ ~multiplier)
    ~fn.body + defaultValue
}

@withComputedDefault("2")
getValue(x: Int) = x

main() = getValue(0)  # Returns 42
```

### Block syntax: `comptime({ block })`

```nostos
template computed(fn, useSquare) = quote {
    result = ~comptime({
        base = 10
        if ~useSquare {
            base * base
        } else {
            base * 2
        }
    })
    result
}

@computed(true)
getSquare() = 0

main() = getSquare()  # 10 * 10 = 100
```

The `comptime` function:
- String syntax: Takes a String that gets evaluated
- Block syntax: Takes a block `({ ... })` that gets serialized and evaluated
- Executes the code at compile time
- Splices the result into the template

This is useful for pre-computing values, lookup tables, or any computation that can be done once at compile time.

**Note:** `comptime` executes in a minimal environment without stdlib. Use it for basic arithmetic, string operations, and simple computations.

## Type Introspection Reference

For type decorators:
- `~typeDef.name` - Type name as String
- `~typeDef.fields` - For single-constructor types like `type Point = Point { x: Int, y: Int }`,
  returns the fields directly. For true variants with multiple constructors, returns a list of constructors.
- `~typeDef.typeParams` - Generic type parameters

Each field has:
- `f.name` - Field name as String
- `f.ty` - Field type as String (note: use `ty`, not `type` which is a keyword)

## AST Types Reference

**Literals:** `Int`, `Float`, `String`, `Bool`, `Char`, `Unit`

**Identifiers:** `Var`

**Expressions:** `BinOp`, `UnaryOp`, `Call`, `MethodCall`, `FieldAccess`, `Index`, `Lambda`, `Block`, `If`, `Match`, `Let`

**Collections:** `List`, `Tuple`, `Record`, `Map`

**Patterns:** `PatternWildcard`, `PatternVar`, `PatternLit`, `PatternTuple`, `PatternList`, `PatternConstructor`

**Definitions:** `FnDef`, `TypeDef`, `TraitImpl`, `Items`

**Template-specific:** `Splice`

## Best Practices

1. **Start simple** - Get basic templates working before adding complexity
2. **Test incrementally** - Verify each generated function works
3. **Use meaningful names** - `fn`, `typeDef` are conventional for decorators
4. **Document what's generated** - Comment the expected output
5. **Handle edge cases** - What if there are no fields? No parameters?

See the full tutorial at: https://nostos-lang.org/tutorial/22_templates.html
