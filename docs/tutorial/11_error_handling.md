# Error Handling

Nostos provides multiple ways to handle errors. For most everyday code, `try/catch` with `throw` is the simplest approach. For more structured error handling, you can use the `Option` and `Result` types.

## try/catch - The Everyday Approach

The simplest way to handle errors in Nostos is `try/catch`. Wrap code that might fail in a `try` block, and handle errors in the `catch` block.

```nostos
# Basic try/catch
main() = {
    result = try {
        data = File.readAll("/etc/config.txt")
        println("Got: " ++ data)
        data
    } catch { e ->
        println("Failed: " ++ e)
        "default value"
    }
}

# Multiple operations in one try block
process_request(url) = {
    try {
        response = Http.get(url)
        parsed = Json.parse(response.body)
        save_to_db(parsed)
        "success"
    } catch { e ->
        log_error(e)
        "failed"
    }
}
```

All I/O operations (files, network, database) throw on error, so you'll use `try/catch` frequently.

## throw - Raising Errors

Use `throw` to raise exceptions that can be caught with `try/catch`. You can throw any value.

```nostos
# Throw with a string message
safe_divide(a, b) =
    if b == 0 then throw("division by zero")
    else a / b

# Throw with a custom error type
type ApiError = { code: Int, msg: String }

fetch_user(id) =
    if id < 0 then throw(ApiError(404, "not found"))
    else "User " ++ id.show()

main() = {
    result = try { safe_divide(10, 0) } catch { e -> "Error: " ++ e }
    println(result)  # "Error: division by zero"
}
```

## panic - Bugs and Impossible States

Use `panic` for truly unrecoverable situations - programmer errors, violated invariants, or bugs that should never happen.

```nostos
# Panic on invariant violation
validate_positive(n) =
    if n <= 0 then panic("Expected a positive number, got " ++ n.show())
    else n

# Pattern matching exhaustiveness
unwrap(Some(x)) = x
unwrap(None) = panic("called unwrap on None")

main() = {
    valid = validate_positive(5)
    # This will cause a panic and terminate the process
    # invalid = validate_positive(-1)
}
```

### throw vs panic

- `throw` - Expected errors that callers should handle (invalid input, network failure, file not found)
- `panic` - Bugs and invariant violations that should never occur in correct code

## Option - When Values Might Be Absent

Nostos does not have `null` or `nil`. Instead, it uses the `Option[T]` type to represent the possible absence of a value.

```nostos
type Option[T] = Some(T) | None

find_item(list, target) =
    if list.contains?(target) then Some(target)
    else None

main() = {
    my_list = [1, 2, 3]
    found = find_item(my_list, 2)

    # Handle Option using pattern matching
    result = match found {
        Some(val) -> "Found: " ++ val.show()
        None -> "Not found"
    }
    println(result) # "Found: 2"
}
```

## Result - Typed Error Handling

For operations that can fail in a way you might want to recover from, use `Result[T, E]`.

```nostos
type Result[T, E] = Ok(T) | Err(E)

# Custom error type
type DivideByZeroError = DivideByZeroError{message: String}

safe_divide(numerator, denominator) =
    if denominator == 0 then Err(DivideByZeroError{message: "Cannot divide by zero"})
    else Ok(numerator / denominator)

main() = {
    result = match safe_divide(10, 2) {
        Ok(val) -> "Result: " ++ val.show()
        Err(err) -> "Error: " ++ err.message
    }
    println(result) # "Result: 5"
}
```

## The ? Operator - Propagating Results

The `?` operator provides a convenient shorthand for propagating errors from functions that return `Result`.

```nostos
calculate_complex(a, b, c) = {
    val1 = safe_divide(a, b)?   # Returns Err if b is 0
    val2 = safe_divide(val1, c)? # Returns Err if c is 0
    Ok(val1 + val2)
}

main() = {
    res1 = calculate_complex(10, 2, 1) # Ok(7)
    res2 = calculate_complex(10, 0, 1) # Err (from first call)
    res3 = calculate_complex(10, 2, 0) # Err (from second call)
}
```

## When to Use What

| Situation | Use |
|-----------|-----|
| I/O operations (file, network, DB) | `try/catch` |
| Validation errors | `throw` + `try/catch` |
| Value might be absent | `Option` |
| Need typed error information | `Result` |
| Bugs / impossible states | `panic` |

**Practical tip:** Start with `try/catch`. It's simple and handles most cases.
