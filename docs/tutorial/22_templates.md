# Templates & Metaprogramming

Nostos provides a compile-time metaprogramming system based on templates.

## Core Concepts

1. **Quote** - Capture code as AST data: `quote { expr }`
2. **Splice** - Insert AST values: `~expr`
3. **Templates** - Compile-time functions: `template name(params) = ...`

## Template Functions

```nostos
template double(fn) = quote {
    result = ~fn.body
    result * 2
}

main() = double(21)  # Returns 42
```

## Decorators

Use `@decorator` syntax on functions. The decorator receives the full function definition:

```nostos
template double(fn) = quote {
    result = ~fn.body
    result * 2
}

@double
getValue() = 21

main() = getValue()  # Returns 42
```

## Function Introspection

Function decorators receive full function info:

- `~fn.name` - Function name as String
- `~fn.params` - List of {name, type} records
- `~fn.body` - Function body AST
- `~fn.returnType` - Return type as String (empty if not specified)

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

## Type Introspection

- `~typeDef.name` - Type name as String
- `~typeDef.fields` - List of constructors (for variants) or fields (for records)
- `~typeDef.typeParams` - Type parameters as List

For variant types like `type Point = Point { x: Int, y: Int }`:
- `~typeDef.fields[0].fields` - Fields of the first constructor

## Compile-Time Eval

Generate code from strings:

```nostos
~eval("functionName() = 42")
```

## Compile-Time Iteration

Use `.map()` to generate multiple items:

```nostos
template withGetters(typeDef) = quote {
    ~typeDef
    ~typeDef.fields[0].fields.map(f => eval("get_" ++ f.name ++ "(r: " ++ ~typeDef.name ++ ") = r." ++ f.name))
}

@withGetters
type Point = Point { x: Int, y: Int }
# Generates: get_x(r: Point) and get_y(r: Point)
```

## Compile-Time Conditionals

Use `~if` for conditional code generation:

```nostos
template maybeDouble(fn, shouldDouble) = quote {
    result = ~fn.body
    ~if ~shouldDouble { quote(result * 2) } else { quote(result) }
}

@maybeDouble(true)
getValue() = 21  # Returns 42

@maybeDouble(false)
getOther() = 100  # Returns 100
```

## Gensym (Unique Identifiers)

Generate unique names to avoid collisions:

```nostos
template genFunc(typeDef) = quote {
    ~typeDef
    ~eval(~gensym("helper") ++ "() = 42")
}
# Generates: helper_0() = 42
```

## AST Types

When working with templates, AST values have these kinds:

**Literals:**
- `Int`, `Float`, `String`, `Bool`, `Char`, `Unit`

**Identifiers:**
- `Var` - variable reference

**Expressions:**
- `BinOp` - binary operations (+, -, *, /, ++, etc.)
- `UnaryOp` - unary operations (-, !)
- `Call` - function calls
- `MethodCall` - method calls (receiver.method(args))
- `FieldAccess` - field access (expr.field)
- `Index` - index access (expr[index])
- `Lambda` - lambda expressions (|params| body)
- `Block` - blocks ({ stmts; result })
- `If` - conditionals
- `Match` - pattern matching
- `Let` - let bindings

**Collections:**
- `List` - list literals [a, b, c]
- `Tuple` - tuple literals (a, b, c)
- `Record` - record literals { field: value }
- `Map` - map literals %{ key: value }

**Patterns:**
- `PatternWildcard` - wildcard (_)
- `PatternVar` - variable pattern
- `PatternLit` - literal pattern
- `PatternTuple` - tuple pattern
- `PatternList` - list pattern with optional rest
- `PatternConstructor` - constructor pattern

**Template-specific:**
- `Splice` - splice marker (~expr)

**Definitions:**
- `FnDef` - function definition
- `TypeDef` - type definition (Record, Variant, or Alias)
- `TraitImpl` - trait implementation
- `Items` - multiple top-level items

See the full tutorial at: https://nostos-lang.org/tutorial/22_templates.html
