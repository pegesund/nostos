# Type System

Nostos features a powerful and flexible type system based on Algebraic Data Types (ADTs), structural typing, and type inference. It emphasizes safety and expressiveness without the need for verbose type annotations.

## Algebraic Data Types (ADTs)

ADTs are fundamental for modeling complex data. Nostos supports two primary forms: Sum Types (Variants) and Product Types (Records).

### Sum Types (Variants)

Variants allow a type to be one of several possibilities. They are perfect for modeling states or mutually exclusive data.

```nostos
# An Option type can be Some(T) or None
type Option[T] = Some(T) | None

# A Result type can be Ok(T) or Err(E)
type Result[T, E] = Ok(T) | Err(E)

# A Shape can be a Circle, Rectangle, or Triangle
type Shape =
  | Circle{radius: Float}
  | Rectangle{width: Float, height: Float}
  | Triangle{base: Float, height: Float}

main() = {
    my_circle = Circle{radius: 10.0}
    my_rect = Rectangle{width: 5.0, height: 8.0}

    # Variants are used extensively with pattern matching
    area(Circle{radius}) = 3.14 * radius * radius
    area(Rectangle{width, height}) = width * height
    area(Triangle{base, height}) = 0.5 * base * height

    println(area(my_circle)) # Calculates circle area
    println(area(my_rect))   # Calculates rectangle area
}
```

### Product Types (Records)

Records are analogous to structs or objects, grouping named fields together. They can be immutable by default or explicitly mutable with `var type`.

```nostos
# Immutable Point record
type Point = {x: Float, y: Float}

# Mutable Buffer record
var type Buffer = {data: List[Int], size: Int}

main() = {
    p = Point(x: 10.0, y: 20.0) # Named field construction
    x_val = p.x                 # Field access

    # For mutable records (var type)
    buf = Buffer(data: [1, 2], size: 2)
    buf.size = 3                # Direct mutation is allowed
}
```

### Record Update (Functional Update)

Create a new record with some fields updated while copying the rest from an existing record. The original record is not modified.

```nostos
type Point = {x: Int, y: Int, z: Int}

main() = {
    p = Point(1, 2, 3)

    # Syntax: Type(base, field: newValue, ...)
    p2 = Point(p, x: 10)        # {x: 10, y: 2, z: 3}
    p3 = Point(p, x: 10, z: 30) # {x: 10, y: 2, z: 30}

    # Works with qualified type names from modules
    use stdlib.rhtml.*
    result = RHtml(div([span("Hello")]))
    updated = stdlib.rhtml.RHtmlResult(result, deps: newDeps)
}
```

### Newtypes

Newtypes wrap an existing type to provide type-safety and prevent accidental misuse. They are single-field variants.

```nostos
type UserId = UserId(Int)
type Email = Email(String)

process_user(id: UserId, email: Email) = {
    println("Processing user " ++ show(id) ++ " with email " ++ show(email))
}

main() = {
    user_id = UserId(123)
    user_email = Email("test@example.com")

    process_user(user_id, user_email)

    # This would be a type error:
    # process_user(123, "test@example.com")

    # Unwrap the value (usually through pattern matching)
    (UserId(id_val)) = user_id
    println("Raw user ID: " ++ show(id_val))
}
```

### Type Aliases

Type aliases provide alternative names for existing types, improving readability without creating a new distinct type.

```nostos
type UserMap = Map[String, String] # Alias for a map of String to String
type Milliseconds = Int

# Now you can use Milliseconds for better clarity
sleep(duration: Milliseconds) = {
    # ... implementation ...
}

main() = {
    config_map: UserMap = %{"host": "localhost", "port": "8080"}
    sleep(1000) # Represents 1000 milliseconds
}
```

## Generics

Nostos supports generics, allowing you to write functions and types that work with any type, or types constrained by traits, without sacrificing type safety.

```nostos
# The Option type is generic over T
type Option[T] = Some(T) | None

# A generic identity function
identity[T](val: T) -> T = val

# A generic list type
type List[T] = Nil | Cons(T, List[T])

main() = {
    int_opt = Some(42)
    str_opt = Some("hello")

    int_id = identity(123)       # int_id is 123, type inferred as Int
    string_id = identity("world") # string_id is "world", type inferred as String
}
```
