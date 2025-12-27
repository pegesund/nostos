# Type Inference Bug: Generic Helper Functions with Concrete Types

## Status: FIXED

The bug was fixed by updating `parse_type_args` in `crates/compiler/src/compile.rs` to handle parentheses when splitting type arguments, not just square brackets.

## Original Issue

The type inference failed to unify generic types from helper functions with concrete types from builtins.

## Reproduction

This previously **failed** but now **works**:
```nostos
getVal(params) = {
    (_, v) = head(params)
    v
}

main() = {
    params = Server.matchPath("/users/42", "/users/:id")
    if length(params) == 1 then {
        id = getVal(params)  # Now works correctly!
        println("Got id: " ++ id)
    } else {
        println("No match")
    }
    0
}
```

## Root Cause

The `parse_type_args` function only tracked square bracket depth `[]` when parsing type arguments, but not parenthesis depth `()`. This caused tuple types like `(a, b)` inside generic types like `List[(a, b)]` to be incorrectly split on the comma.

When parsing `List[(a, b)]`:
- **Before fix**: Split args on comma -> `[(a` and `b)]` -> Parsed as two separate `Named` types
- **After fix**: Correctly parses `(a, b)` as a single tuple type argument

## The Fix

In `crates/compiler/src/compile.rs`, the `parse_type_args` function was updated:

```rust
// Before: Only tracked []
'[' => { depth += 1; ... }
']' => { depth -= 1; ... }

// After: Tracks both [] and ()
'[' | '(' => { depth += 1; ... }
']' | ')' => { depth -= 1; ... }
```

## Related Note

### Parser Bug: `receive` as keyword after dot
The word `receive` is treated as a keyword even after a dot, so `WebSocket.receive` fails to parse. We renamed the builtin to `WebSocket.recv` as a workaround.

This may affect other builtins that contain reserved words.
