# Builtin Traits

In Nostos, all types automatically have implementations for the core traits `Hash`, `Eq`, `Show`, and `Copy`. No special syntax is needed - these capabilities are always available for every type you define.

## Available Builtin Traits

Every type in Nostos automatically supports these four traits:

| Trait | Function | Description |
|-------|----------|-------------|
| Hash | `hash(x) -> Int` | Hash function for use in maps/sets |
| Show | `show(x) -> String` | String representation for debugging |
| Eq | `==`, `!=` operators | Equality comparison |
| Copy | `copy(x) -> T` | Deep copy of values |

## Basic Usage

Just define your types - all traits are available automatically.

```nostos
# All types automatically have Hash, Show, Eq, Copy
type Point = Point(Int, Int)
type Color = Red | Green | Blue
type Person = { name: String, age: Int }

main() = {
    p = Point(3, 4)

    # Using Show
    println(show(p))           # Point(3, 4)

    # Using Eq
    println(show(p == Point(3, 4)))  # true

    # Using Hash
    println(show(hash(p)))     # some integer

    # Using Copy
    p2 = copy(p)
    println(show(p == p2))     # true
}
```

## Variant Types

All variant (sum) types have builtin traits - from simple enums to complex variants with fields.

```nostos
# Simple enum-like variants
type Color = Red | Green | Blue

# Variants with fields
type Result = Ok(Int) | Err(String)

# Complex variants
type Expression = Number(Int) | Variable(String) | Add(Expression, Expression)

main() = {
    # Using Show
    println(show(Red))           # "Red"
    println(show(Ok(42)))        # "Ok(42)"

    # Using Eq
    println(show(Red == Red))    # true
    println(show(Red == Blue))   # false
}
```

## Record Types

Records with named fields also have all builtin traits. The implementations use all fields.

```nostos
type Person = { name: String, age: Int }

main() = {
    alice = Person("Alice", 30)
    bob = Person("Bob", 25)
    alice2 = Person("Alice", 30)

    # Show displays all fields
    println(show(alice))  # Person{name: Alice, age: 30}

    # Eq compares all fields
    println(show(alice == alice2))  # true
    println(show(alice == bob))     # false

    # Copy creates an independent copy
    copied = copy(alice)
    println(show(copied == alice))  # true
}
```

## Nested Types

When a type contains another type, the inner type's trait methods are called automatically. Composition works naturally.

```nostos
# Inner types
type Email = Email(String)
type Address = { street: String, city: String, zip: Int }

# Outer type that uses them
type Contact = { name: String, email: Email, address: Address }

main() = {
    email = Email("alice@example.com")
    addr = Address("123 Main St", "Springfield", 12345)
    contact = Contact("Alice", email, addr)

    # Show recursively shows nested types
    println(show(contact))

    # Eq recursively compares nested types
    contact2 = Contact("Alice", Email("alice@example.com"), Address("123 Main St", "Springfield", 12345))
    println(show(contact == contact2))  # true
}
```

## Hash Consistency

The builtin `Hash` trait ensures that equal values always produce equal hash codes - a requirement for correct behavior in maps and sets.

```nostos
type UserId = UserId(Int)

main() = {
    u1 = UserId(42)
    u2 = UserId(42)
    u3 = UserId(99)

    # Equal values have equal hashes
    println(show(hash(u1) == hash(u2)))  # true

    # Different values typically have different hashes
    println(show(hash(u1) == hash(u3)))  # false (usually)

    # This makes UserId usable as a map key
}
```

## Practical Example: Custom Map Key

Here's a practical example using builtin traits to create a type suitable for use as a map key.

```nostos
type UserId = UserId(Int)
type UserData = { name: String, score: Int }

# Simple association list lookup using builtin Eq
lookup([], key) = "not found"
lookup([(k, v) | rest], key) = if k == key then v else lookup(rest, key)

main() = {
    users = [(UserId(1), UserData("Alice", 100)),
             (UserId(2), UserData("Bob", 85)),
             (UserId(3), UserData("Charlie", 92))]

    # lookup uses the builtin == from Eq
    result = lookup(users, UserId(2))
    println(show(result))  # UserData{name: Bob, score: 85}
}
```
