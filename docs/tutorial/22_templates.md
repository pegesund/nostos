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
    ~typeDef.fields[0].fields.map(f =>
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
    ~typeDef.fields[0].fields.map(f =>
        eval("get_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ") = r." ++ f.name)
    )
    # Generate setters (return new instance)
    ~typeDef.fields[0].fields.map(f =>
        eval("set_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ", v: " ++ f.type ++ ") = " ++
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

## Example 6: Validation Decorators

Add runtime validation to functions:

```nostos
# Runtime validation - checks parameter at runtime
template nonNegative(fn) = quote {
    param = ~fn.params[0].name
    if param < 0 {
        panic("Value must be non-negative")
    }
    ~fn.body
}

@nonNegative
sqrt(n: Int) = {
    # Body only runs if n >= 0
    n  # placeholder for actual sqrt
}

main() = {
    println(sqrt(16))   # Works: 16
    println(sqrt(-1))   # Panics: "Value must be non-negative"
}
```

Compile-time conditional validation:

```nostos
# Compile-time check - generates different code based on flag
template validated(fn, check, errorMsg) = quote {
    ~if ~check {
        quote { ~fn.body }
    } else {
        quote { panic(~errorMsg) }
    }
}

@validated(true, "Feature disabled")
enabledFeature() = "This works"

@validated(false, "Feature disabled")
disabledFeature() = "Never runs"

main() = {
    println(enabledFeature())   # Works: "This works"
    println(disabledFeature())  # Panics: "Feature disabled"
}
```

## Example 7: Unique Variable Names with Gensym

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

## Type Introspection Reference

For type decorators:
- `~typeDef.name` - Type name as String
- `~typeDef.fields` - Constructors (for variants) or fields (for records)
- `~typeDef.fields[0].fields` - Fields of first constructor
- `~typeDef.typeParams` - Generic type parameters

Each field has:
- `f.name` - Field name
- `f.type` - Field type as String

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
