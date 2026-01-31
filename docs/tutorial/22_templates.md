# Templates & Metaprogramming

Nostos provides a compile-time metaprogramming system based on templates.

## Core Concepts

1. **Quote** - Capture code as AST data: `quote { expr }`
2. **Splice** - Insert AST values: `~expr`
3. **Templates** - Compile-time functions: `template name(params) = ...`

## Template Functions

```nostos
template double(body) = quote {
    result = ~body
    result * 2
}

main() = double(21)  # Returns 42
```

## Decorators

Use `@decorator` syntax on functions:

```nostos
@double
getValue() = 21

main() = getValue()  # Returns 42
```

## Type Decorators

Templates can be applied to type definitions:

```nostos
template withGetter(typeDef) = quote {
    ~typeDef
    ~eval("getX(r: " ++ ~typeDef.name ++ ") = r.x")
}

@withGetter
type Point = Point { x: Int, y: Int }

main() = {
    p = Point(x: 42, y: 10)
    getX(p)  # Returns 42
}
```

## Compile-Time Introspection

- `~typeDef.name` - Type name as String
- `~typeDef.fields` - List of {name, type} records
- `~typeDef.typeParams` - Type parameters as List

## Compile-Time Eval

Generate code from strings:

```nostos
~eval("functionName() = 42")
```

See the full tutorial at: https://nostos-lang.org/tutorial/22_templates.html
