# Traits

Traits in Nostos provide a powerful mechanism for polymorphism, allowing you to define shared behavior that different types can implement. They are similar to interfaces in Java or Go, or type classes in Haskell.

## Defining a Trait

A trait defines a set of functions that a type must implement. It specifies the function signatures without providing an implementation.

```nostos
# Define a 'Show' trait for types that can be converted to a string
trait Show
    show(self) -> String
end

# Define an 'Eq' trait for types that can be compared for equality
trait Eq
    eq(self, other: Self) -> Bool
end
```

## Implementing a Trait for a Type

Use the `Type: Trait` syntax to implement a trait for a specific type. You must provide an implementation for all functions defined in the trait.

```nostos
type Point = {x: Float, y: Float}

# Implement the Show trait for Point
Point: Show
    show(self) = "(" ++ self.x.show() ++ ", " ++ self.y.show() ++ ")"
end

# Implement the Eq trait for Point
Point: Eq
    eq(self, other) = self.x == other.x && self.y == other.y
end

main() = {
    p1 = Point(1.0, 2.0)
    p2 = Point(1.0, 2.0)
    p3 = Point(3.0, 4.0)

    println(p1.show()) # Prints "(1.0, 2.0)"
    is_equal1 = p1.eq(p2) # true
    is_equal2 = p1.eq(p3) # false
}
```

## Generic Traits and Implementations

Traits can be generic, and you can implement them for generic types, often with constraints on the type parameters.

```nostos
type Option[T] = Some(T) | None

# Implement Show for Option[T], but only if T itself implements Show
Option[T]: Show when T: Show
    show(None) = "None"
    show(Some(x)) = "Some(" ++ x.show() ++ ")"
end

main() = {
    opt_int = Some(42)
    opt_none = None

    println(opt_int.show())  # Prints "Some(42)"
    println(opt_none.show()) # Prints "None"
}
```

## Using Traits in Functions

Functions can accept arguments that implement a specific trait, allowing them to work polymorphically with any type that satisfies the trait's contract.

```nostos
# A function that can print any 'Show'able thing
print_showable(item: T) when T: Show = {
    println("Item: " ++ item.show())
}

main() = {
    p = Point(5.0, 10.0)
    num = 123
    text = "Hello Traits!"

    print_showable(p)     # Prints "Item: (5.0, 10.0)"
    print_showable(num)   # Prints "Item: 123" (Int has a built-in Show impl)
    print_showable(text)  # Prints "Item: Hello Traits!" (String has a built-in Show impl)
}
```

## Defining Your Own Traits

Beyond standard traits like Show and Eq, you can define traits that capture domain-specific behavior.

### Example: Serializable Trait

```nostos
# Define a trait for serialization
trait Serializable
    serialize(self) -> String
    deserialize(data: String) -> Self
end

type Config = { host: String, port: Int, debug: Bool }

Config: Serializable
    serialize(self) = self.host ++ ":" ++ self.port.show() ++ ":" ++ self.debug.show()

    deserialize(data) = {
        parts = data.split(":")
        Config(parts.get(0), parts.get(1).parseInt(), parts.get(2) == "true")
    }
end

main() = {
    config = Config("localhost", 8080, true)
    saved = config.serialize()          # "localhost:8080:true"
    loaded = Config.deserialize(saved)  # Config back from string
}
```

### Example: Drawable Trait

```nostos
# Define a trait for drawable shapes
trait Drawable
    draw(self) -> String
    area(self) -> Float
    perimeter(self) -> Float
end

type Circle = { radius: Float }
type Rectangle = { width: Float, height: Float }

Circle: Drawable
    draw(self) = "O (radius=" ++ self.radius.show() ++ ")"
    area(self) = 3.14159 * self.radius * self.radius
    perimeter(self) = 2.0 * 3.14159 * self.radius
end

Rectangle: Drawable
    draw(self) = "[" ++ self.width.show() ++ "x" ++ self.height.show() ++ "]"
    area(self) = self.width * self.height
    perimeter(self) = 2.0 * (self.width + self.height)
end

# Works with any Drawable
print_shape_info(shape: T) when T: Drawable = {
    println(shape.draw())
    println("  Area: " ++ shape.area().show())
}

main() = {
    circle = Circle(5.0)
    rect = Rectangle(4.0, 3.0)

    print_shape_info(circle)
    print_shape_info(rect)
}
```

### When to Define Custom Traits

- **Shared behavior:** Multiple types need the same set of operations
- **Polymorphism:** You want functions to work with any type that has certain capabilities
- **Decoupling:** Separate "what" from "how" - define interface, implement details per type
- **Testing:** Mock implementations can satisfy the same trait as real ones

## Operator Overloading

Operators in Nostos can be overloaded for custom types by implementing the appropriate trait.

**Operator to Trait Mapping:**
- `+`, `-`, `*`, `/` -> `Num` trait (`add`, `sub`, `mul`, `div`)
- `<`, `>`, `<=`, `>=` -> `Ord` trait (`lt`, `gt`, `lte`, `gte`)
- `==`, `!=` -> `Eq` trait (`eq`, `neq`)

```nostos
# Define a 2D vector type
type Vec2 = { x: Int, y: Int }

# Define the Num trait for arithmetic operations
trait Num
    add(self, other: Self) -> Self
    sub(self, other: Self) -> Self
    mul(self, other: Self) -> Self
    div(self, other: Self) -> Self
end

# Implement Num for Vec2 to enable operator overloading
Vec2: Num
    add(self, other: Vec2) -> Vec2 = Vec2(self.x + other.x, self.y + other.y)
    sub(self, other: Vec2) -> Vec2 = Vec2(self.x - other.x, self.y - other.y)
    mul(self, other: Vec2) -> Vec2 = Vec2(self.x * other.x, self.y * other.y)
    div(self, other: Vec2) -> Vec2 = Vec2(self.x / other.x, self.y / other.y)
end

main() = {
    v1 = Vec2(1, 2)
    v2 = Vec2(3, 4)

    # Now you can use operators directly!
    v3 = v1 + v2          # Calls Vec2.Num.add
    println(v3.x)         # Prints 4
    println(v3.y)         # Prints 6

    # Chained operations work too
    v4 = v1 + v2 + v1     # (1+3+1, 2+4+2) = (5, 8)
}
```

**Note:** Operator overloading is resolved at compile time with zero runtime overhead.
