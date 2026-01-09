# Trait Bounds

Trait bounds let you write generic functions that require certain capabilities from their type parameters. This enables **compile-time checking** that types support the operations you need, catching errors before your code runs.

## Basic Syntax

Add trait bounds after a type parameter using a colon. The syntax `[T: Trait]` means "T must implement Trait".

```nostos
# T must implement Hash
hashable[T: Hash](x: T) -> Int = hash(x)

# T must implement Eq
are_equal[T: Eq](x: T, y: T) -> Bool = x == y

# T must implement Show
describe[T: Show](x: T) -> String = "Value: " ++ show(x)

main() = {
    # Works with any type implementing the required trait
    println(show(hashable(42)))           # Int implements Hash
    println(show(are_equal("a", "a")))    # String implements Eq
    println(describe(true))               # Bool implements Show
}
```

## Multiple Constraints

Use `+` to require a type to implement multiple traits.

```nostos
# T must implement both Hash AND Show
hash_and_show[T: Hash + Show](x: T) -> String = {
    h = hash(x)
    "hash(" ++ show(x) ++ ") = " ++ show(h)
}

# T must implement Hash, Eq, AND Show
full_info[T: Hash + Eq + Show](x: T, y: T) -> String = {
    eq_str = if x == y then "equal" else "not equal"
    show(x) ++ " and " ++ show(y) ++ " are " ++ eq_str
}

main() = {
    println(hash_and_show(42))
    # Output: hash(42) = 7298839827432

    println(full_info("hello", "hello"))
    # Output: hello and hello are equal
}
```

## Multiple Type Parameters

Each type parameter can have its own constraints, separated by commas.

```nostos
# Two parameters with the same bound
compare_hashes[T: Hash, U: Hash](x: T, y: U) -> Bool = hash(x) == hash(y)

# Different bounds on each parameter
show_hash[T: Show, U: Hash](x: T, y: U) -> String = {
    show(x) ++ " has hash peer with hash " ++ show(hash(y))
}

# Mixed: one constrained, one unconstrained
show_first[T: Show, U](x: T, y: U) -> String = show(x)

main() = {
    # compare_hashes works with different types
    println(show(compare_hashes(42, 42)))     # true (same Int)
    println(show(compare_hashes("a", "a")))   # true (same String)
}
```

## Compile-Time Checking

Trait bounds are checked at compile time. If you try to use a type that doesn't implement the required trait, you get a helpful error message.

```nostos
# Define a custom trait
trait Printable
    toText(x) -> String
end

printable[T: Printable](x: T) -> String = toText(x)

type MyType = MyType(Int)  # No Printable impl

main() = {
    x = MyType(42)
    printable(x)  # Compile error!
}

# Error: MyType does not implement trait `Printable`
```

**Note:** All types automatically have the builtin traits `Hash`, `Eq`, `Show`, and `Copy`. Trait bounds are most useful for custom traits.

## Primitive Type Traits

Built-in primitive types automatically implement common traits:

| Type | Implements |
|------|------------|
| Int | Hash, Eq, Show, Copy |
| Float | Eq, Show, Copy |
| Bool | Hash, Eq, Show, Copy |
| Char | Hash, Eq, Show, Copy |
| String | Hash, Eq, Show, Copy |

## Practical Example: Generic Lookup

Here's a practical example: a generic lookup function that works with any key type that implements `Eq`.

```nostos
# Generic lookup - K must implement Eq for comparison
type NotFound = NotFound
type Found = Found(V)

lookup[K: Eq, V](items: List, key: K) = match items {
    [] -> NotFound
    [(k, v) | rest] -> if k == key then Found(v) else lookup(rest, key)
}

# Custom types - all have builtin Eq
type UserId = UserId(Int)
type Email = Email(String)

main() = {
    # Works with UserId keys
    users = [(UserId(1), "Alice"), (UserId(2), "Bob")]
    result1 = lookup(users, UserId(1))
    println(show(result1))  # Found(Alice)

    # Works with Email keys
    emails = [(Email("a@b.com"), 100), (Email("x@y.com"), 200)]
    result2 = lookup(emails, Email("x@y.com"))
    println(show(result2))  # Found(200)
}
```

## Bounds with Higher-Order Functions

Trait bounds work seamlessly with higher-order functions and lambdas.

```nostos
# Apply a function and show the result
apply_and_show[T: Show, U: Show](f: T -> U, x: T) -> String = {
    result = f(x)
    show(x) ++ " -> " ++ show(result)
}

main() = {
    double = x => x * 2
    println(apply_and_show(double, 21))
    # Output: 21 -> 42

    negate = b => if b then false else true
    println(apply_and_show(negate, true))
    # Output: true -> false
}
```

## Summary

1. **Basic bounds:** `[T: Trait]` requires T to implement Trait
2. **Multiple traits:** Use `+` to combine: `[T: Hash + Eq]`
3. **Multiple params:** Each can have bounds: `[T: Hash, U: Show]`
4. **Compile-time safety:** Errors caught before runtime
5. **Builtin traits:** All types have `Hash`, `Eq`, `Show`, `Copy` automatically
