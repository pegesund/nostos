# Pattern Matching

Pattern matching is the heart of Nostos. It's not just a feature - it's a way of thinking about data. Instead of asking "what type is this?" and then digging inside, you describe the *shape* you expect and the values fall into place.

## The "Aha!" Moment

Here's a database query. See how we name the columns right in the pattern:

```nostos
# Query returns rows as tuples
rows = Pg.query(conn, "SELECT id, name, email FROM users", [])

# Pattern matching gives each column a name - no row[0], row[1] needed!
rows.each((id, name, email) => {
    println(name ++ " <" ++ email ++ ">")
})
```

No index lookups. No parsing. The structure of your data becomes the structure of your code.

Even better - match on the entire row, including country:

```nostos
# Pattern match on the whole tuple, branching on country
rows = Pg.query(conn, "SELECT id, amount, country FROM orders", [])

rows.each(row => match row {
    (id, amount, "NO") -> println("Order #" ++ show(id) ++ ": " ++ show(amount * 1.25) ++ " (incl. 25% Norwegian VAT)")
    (id, amount, _) -> println("Order #" ++ show(id) ++ ": " ++ show(amount))
})
```

The country is matched as part of the tuple pattern - Norwegian orders get special handling, others fall through to the default branch.

## Why Pattern Matching?

Traditional languages make you *ask* about your data: Is it null? What type? How long? Pattern matching lets you *declare* what shape you expect, and the language handles the rest.

### Key Benefits

- **Exhaustiveness checking:** The compiler warns if you forget a case
- **Destructuring built-in:** Extract values while matching structure
- **No null pointer exceptions:** Handle None/Some explicitly
- **Self-documenting:** The pattern shows exactly what data shape is expected
- **Composable:** Combine patterns with guards for complex logic

## Real-World Example: Parsing Commands

Pattern matching shines when handling structured data. Here's how you might parse user commands:

```nostos
# Command type with different variants
type Command =
    | Help
    | Quit
    | Load(String)
    | Save(String)
    | Search(String, Int)

# Handle each command - the compiler ensures all cases are covered
execute(Help) = println("Available commands: help, quit, load, save, search")
execute(Quit) = println("Goodbye!")
execute(Load(filename)) = println("Loading " ++ filename ++ "...")
execute(Save(filename)) = println("Saving to " ++ filename ++ "...")
execute(Search(query, limit)) = println("Searching for '" ++ query ++ "' (max " ++ show(limit) ++ " results)")

main() = {
    execute(Help)
    execute(Load("data.json"))
    execute(Search("nostos", 10))
    execute(Quit)
}
```

Each function clause handles one command variant. If you add a new command to the type, the compiler will remind you to handle it everywhere. No more forgotten cases!

## Patterns in Function Definitions

You can define multiple function clauses, each with a pattern. The first matching pattern is executed.

```nostos
# Literal patterns
is_zero(0) = true
is_zero(_) = false

# List patterns (head and tail)
first_element([x | _]) = x
rest_elements([_ | xs]) = xs

# Tuple patterns
swap((a, b)) = (b, a)

main() = {
    println(is_zero(0))   # true
    println(is_zero(5))   # false
    println(first_element([1, 2, 3])) # 1
    println(swap(("hello", 123))) # (123, "hello")
}
```

## The `match` Expression

For more complex pattern matching or when matching against a single value, the `match` expression is invaluable.

```nostos
describe(n) = match n {
    0 -> "zero"
    1 -> "one"
    _ -> "many" # Wildcard pattern for any other value
}

main() = {
    println(describe(0)) # "zero"
    println(describe(1)) # "one"
    println(describe(5)) # "many"
}
```

## Patterns with Guards

Guards allow you to add conditional logic to patterns, making them even more powerful.

```nostos
classify(n) = match n {
    x when x > 0 -> "positive"
    0 -> "zero"
    _ -> "negative"
}

main() = {
    println(classify(10)) # "positive"
    println(classify(0))  # "zero"
    println(classify(-3)) # "negative"
}
```

## Guard Fallthrough

When a pattern matches but its guard fails, execution **falls through** to try the next arm. This is important when you have multiple arms with guards on the same pattern.

```nostos
size_category(n) = match n {
    x when x > 100 -> "huge"
    x when x > 10 -> "big"
    x when x > 0 -> "small"
    _ -> "zero or negative"
}

main() = {
    println(size_category(50))   # "big" - first guard fails, second matches
    println(size_category(5))    # "small" - first two guards fail
    println(size_category(-1))   # "zero or negative"
}
```

## Variable Constraints in Patterns

When an **existing immutable variable** appears in a pattern, it acts as a **constraint** rather than creating a new binding. The pattern only matches if the value equals the variable's current value.

```nostos
main() = {
    expected = 5

    # x acts as a constraint - must equal 5
    result1 = match (5, 10) {
        (expected, y) -> y * 2   # Matches! expected == 5
    }
    println(result1)  # 20

    # This won't match the first arm
    result2 = match (3, 10) {
        (expected, y) -> y * 2   # Doesn't match! 3 != expected
        (_, y) -> y + 1          # Falls through to this
    }
    println(result2)  # 11
}
```

This is especially useful for matching on expected values like status codes:

```nostos
handle_response(status, body) = {
    ok_status = 200

    match status {
        ok_status -> "Success: " ++ body    # Only if status == 200
        404 -> "Not found"
        s -> "Error: " ++ show(s)
    }
}
```

## Destructuring Records and Variants

You can destructure custom data types to extract their internal values.

```nostos
type Point = { x: Int, y: Int }
type Option[T] = Some(T) | None

get_x_coord({ x, _ }) = x # Punning: {x, _} is short for {x: x, _}
unwrap_option(Some(val)) = val

main() = {
    p = Point(10, 20)
    x_val = get_x_coord(p)  # x_val is 10

    opt_val = Some("hello")
    unwrapped = unwrap_option(opt_val) # unwrapped is "hello"

    # Using match for comprehensive handling
    description = match opt_val {
        Some(s) -> "Contains: " ++ s
        None -> "Is empty"
    }
    println(description) # "Contains: hello"
}
```

## Maps and Sets

Nostos supports pattern matching on Maps and Sets.

```nostos
# Matching on a Map
check_config(config) = match config {
    %{"debug": true, "level": lvl} -> "Debug mode level " ++ show(lvl)
    %{"debug": false} -> "Production mode"
    _ -> "Unknown config"
}

# Matching on a Set
check_access(roles) = match roles {
    #{"admin"} -> "Access granted"
    #{"editor", "viewer"} -> "Partial access"
    _ -> "Access denied"
}

main() = {
    conf = %{"debug": true, "level": 3, "other": "ignored"}
    println(check_config(conf))  # "Debug mode level 3"

    user_roles = #{"user", "admin", "guest"}
    println(check_access(user_roles)) # "Access granted"
}
```

## The Pin Operator (`^`)

The pin operator `^` in a pattern asserts that a value must match an *existing* variable's value.

```nostos
expected_status = 200

handle_response(response) = match response {
    { status: ^expected_status, body: data } -> "Success: " ++ data
    { status: s, body: data } -> "Error " ++ show(s) ++ ": " ++ data
}

main() = {
    response1 = { status: 200, body: "Data received" }
    response2 = { status: 404, body: "Not Found" }

    println(handle_response(response1)) # "Success: Data received"
    println(handle_response(response2)) # "Error 404: Not Found"
}
```

## Pattern Matching with Recursive Types

Pattern matching truly shines with recursive data structures like trees:

```nostos
# A binary tree: either empty or a node with value and children
type Tree[T] = Empty | Node(T, Tree[T], Tree[T])

# Count nodes in the tree
size(Empty) = 0
size(Node(_, left, right)) = 1 + size(left) + size(right)

# Sum all values (for numeric trees)
sum(Empty) = 0
sum(Node(val, left, right)) = val + sum(left) + sum(right)

# Check if a value exists in the tree
contains(Empty, _) = false
contains(Node(val, _, _), target) when val == target = true
contains(Node(_, left, right), target) = contains(left, target) || contains(right, target)

main() = {
    tree = Node(5,
        Node(3, Node(1, Empty, Empty), Empty),
        Node(8, Node(6, Empty, Empty), Node(9, Empty, Empty))
    )

    assert_eq(size(tree), 6)
    assert_eq(sum(tree), 32)
    println("Tree operations passed!")
}
```

**Tip:** Pattern matching encourages you to think about your data as *shapes* rather than as objects with methods. This often leads to simpler, more composable code.
