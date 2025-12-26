# Reflection & Eval

Nostos provides powerful metaprogramming capabilities including runtime type introspection, dynamic value construction, and code evaluation.

## Type Introspection with typeInfo

The `typeInfo(typeName)` builtin returns type metadata as a Map, allowing you to inspect types at runtime.

```nostos
type Person = { name: String, age: Int }
type Status = Active | Pending | Done

main() = {
    # Get record type info
    personInfo = typeInfo("Person")
    println(personInfo.get("kind"))    # "record"
    println(personInfo.get("fields"))  # List of field metadata

    # Get variant type info
    statusInfo = typeInfo("Status")
    println(statusInfo.get("kind"))    # "variant"
    println(statusInfo.get("constructors"))  # List of constructors

    # Unknown types return empty map
    unknownInfo = typeInfo("NoSuchType")
    println(unknownInfo.isEmpty())  # true
}
```

### typeInfo Return Value

- `"name"` - Type name as string
- `"kind"` - "record" or "variant"
- `"fields"` - List of Maps with "name" and "type" (for records)
- `"constructors"` - List of Maps with constructor info (for variants)

## Getting Variant Tags with tagOf

The `tagOf(value)` builtin returns the constructor name of a variant value, or an empty string for non-variants.

```nostos
type Result = Ok(Int) | Err(String)
type Color = Red | Green | Blue

main() = {
    result = Ok(42)
    println(tagOf(result))  # "Ok"

    err = Err("failed")
    println(tagOf(err))     # "Err"

    color = Red
    println(tagOf(color))   # "Red"

    # Non-variants return empty string
    println(tagOf(42))      # ""
    println(tagOf("hello")) # ""
}
```

## Dynamic Value Construction

Construct typed values dynamically using `makeRecord` and `makeVariant`.

### With Type Parameters

Use type parameters when the type is known at compile time.

```nostos
use stdlib.json.{String, Number}

type Person = { name: String, age: Int }
type Result = Ok(Int) | Err(String)

main() = {
    # Construct a record from a Map[String, Json]
    fields = %{"name": String("Alice"), "age": Number(30.0)}
    person = makeRecord[Person](fields)
    println(person.name)  # "Alice"
    println(person.age)   # 30

    # Construct a variant
    ok_fields = %{"_0": Number(42.0)}
    result = makeVariant[Result]("Ok", ok_fields)
    # result is Ok(42)
}
```

### With String Type Names

Use string-based versions when the type name is determined at runtime.

```nostos
use stdlib.json.{jsonParse, fromJsonValue, String, Number}

type Person = { name: String, age: Int }

main() = {
    # Type name as runtime string
    typeName = "Person"
    fields = %{"name": String("Bob"), "age": Number(25.0)}
    person = makeRecordByName(typeName, fields)

    # Same for variants
    result = makeVariantByName("Result", "Ok", %{"_0": Number(100.0)})

    # JSON to typed value using stdlib
    json = jsonParse('{"name": "Charlie", "age": 35}')
    person2 = fromJsonValue("Person", json)
}
```

## Dynamic Code Evaluation

The `eval(code)` builtin compiles and executes Nostos code at runtime, returning the result as a string.

```nostos
main() = {
    # Evaluate simple expressions
    result = eval("1 + 2 * 3")
    println(result)  # "7"

    # Define and call functions
    eval("add(a, b) = a + b")
    result2 = eval("add(10, 20)")
    println(result2)  # "30"

    # Works with any Nostos code
    eval("type Point = { x: Int, y: Int }")
    result3 = eval("Point(1, 2)")
    println(result3)  # "Point { x: 1, y: 2 }"
}
```

### Performance Note

`eval` compiles code at runtime, which has overhead. For performance-critical code, prefer pre-compiled functions. Use `eval` for:

- REPL-like interfaces
- Dynamic configuration
- Plugin systems
- Prototyping and experimentation

## Use Cases

- **Generic serialization:** Build JSON/YAML/XML serializers that work with any type
- **ORM mapping:** Map database rows to typed records dynamically
- **Form generation:** Generate UI forms from type schemas
- **Plugin systems:** Load and execute code at runtime
- **REPL tools:** Build interactive development environments
