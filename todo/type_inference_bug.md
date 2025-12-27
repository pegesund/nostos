# Type Inference Bug: Generic Helper Functions with Concrete Types

## Issue
The type inference fails to unify generic types from helper functions with concrete types from builtins.

## Reproduction

This **fails**:
```nostos
getVal(params) = {
    (_, v) = head(params)
    v
}

main() = {
    params = Server.matchPath("/users/42", "/users/:id")
    if length(params) == 1 then {
        id = getVal(params)  # ERROR: Cannot unify List[(a, b)] and List[(String, String)]
        println("Got id: " ++ id)
    } else {
        println("No match")
    }
    0
}
```

Error: `Cannot unify types: List[(a, b)] and List[(String, String)]`

This **works** (inline deconstruction):
```nostos
main() = {
    params = Server.matchPath("/users/42", "/users/:id")
    if length(params) == 1 then {
        (_, id) = head(params)  # OK - inline works fine
        println("Got id: " ++ id)
    } else {
        println("No match")
    }
    0
}
```

## Root Cause
The helper function `getVal` infers a generic type `List[(a, b)]` for its parameter. When called with the result of `Server.matchPath` (which returns `List[(String, String)]`), the type checker cannot unify the generic type with the concrete type.

## Additional Notes

### Parser Bug: `receive` as keyword after dot
Related issue discovered: The word `receive` is treated as a keyword even after a dot, so `WebSocket.receive` fails to parse. We renamed the builtin to `WebSocket.recv` as a workaround.

This may affect other builtins that contain reserved words.

## Expected Behavior
The generic type `List[(a, b)]` should unify with `List[(String, String)]` by binding `a = String` and `b = String`.

## Workaround
Inline the tuple deconstruction instead of using helper functions:

```nostos
# Instead of using a helper:
# userId = getVal(params)

# Inline deconstruction works:
(_, userId) = head(params)
```

## Affected Features
- `Server.matchPath` - used for URL routing with path parameters
- `req.queryParams`, `req.cookies`, `req.formParams` - all return `List[(String, String)]`
- Any similar pattern where a helper function extracts values from typed tuples

## Priority
Medium - has workaround, but makes code less clean/reusable.
