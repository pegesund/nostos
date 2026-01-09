# Nostos Improvement Ideas

Ideas and potential improvements for the language.

## Language Features

### Deriving for Traits
Auto-generate trait implementations for common traits.

```nostos
# Current: must implement manually
type Point = { x: Int, y: Int }
Point: Show show(self) = "Point(" ++ show(self.x) ++ ", " ++ show(self.y) ++ ")" end
Point: Eq eq(self, other) = self.x == other.x && self.y == other.y end

# Desired: auto-derive
type Point = { x: Int, y: Int } deriving (Show, Eq, Hash)
```

**Priority**: Medium - reduces boilerplate significantly

### Trait Bounds on Generics
Constrain generic type parameters.

```nostos
# Current: no constraints
sort(xs: List[a]) -> List[a]  # How do we compare elements?

# Desired: require Ord trait
sort[a: Ord](xs: List[a]) -> List[a]
```

**Priority**: Medium - enables safer generic code

### Multiline Type Definitions
Allow record/variant types to span multiple lines.

```nostos
# Current: must be single line
type App = { state: State, render: () -> Html, handle: (Action) -> () }

# Desired: multiline for readability
type App = {
    state: State,
    render: () -> Html,
    handle: (Action) -> ()
}
```

**Priority**: Low - cosmetic, single-line works

### Forward Declarations
Enable mutual recursion between functions.

```nostos
# Current: order matters, mutual recursion tricky
# Desired: declare before define
declare isEven: Int -> Bool
declare isOdd: Int -> Bool

isEven(n) = if n == 0 then true else isOdd(n - 1)
isOdd(n) = if n == 0 then false else isEven(n - 1)
```

**Priority**: Low - rarely needed, workarounds exist

## Standard Library

### String.drop and String.take
Convenient string slicing.

```nostos
# Current workaround
rest = String.substring(s, 5, String.length(s))

# Desired
rest = String.drop(s, 5)
first5 = String.take(s, 5)
```

**Priority**: Low - substring works fine

### Set Type
Unique collection type.

```nostos
# Desired
s = Set.from([1, 2, 3, 2, 1])  # {1, 2, 3}
Set.contains(s, 2)             # true
Set.insert(s, 4)               # {1, 2, 3, 4}
```

**Priority**: Medium - useful for many algorithms

### Result Type for Error Handling
Alternative to exceptions for recoverable errors.

```nostos
# Desired
type Result[T, E] = Ok(T) | Err(E)

parseNumber(s: String) -> Result[Int, String]
parseNumber(s) = match String.toInt(s) {
    Some(n) -> Ok(n)
    None() -> Err("Invalid number: " ++ s)
}
```

**Priority**: Low - exceptions work, Option exists

## Performance

### JIT Compilation
Currently partial - expand to more operations.

**Priority**: Medium - significant performance gains possible

### Shared Heap for Cross-Process Data
Zero-copy sharing between processes for immutable data.

```nostos
# Currently data is copied when sent between processes
# Desired: large immutable data shared without copying
```

**Priority**: Low - current copying works, optimize later

## Developer Experience

### Better Error Messages
More helpful compile-time error messages with suggestions.

```
# Current
Type mismatch: expected Int, got String

# Desired
Type mismatch at line 42:
    result = x + "hello"
                 ^^^^^^^
Expected: Int (to match left side of +)
Found: String

Hint: Use `show(x) ++ "hello"` to concatenate strings
```

**Priority**: High - significantly improves usability

### LSP Go-to-Definition
Jump to function/type definitions from usage.

**Priority**: Medium - improves IDE experience

### REPL History Persistence
Save REPL history between sessions.

**Priority**: Low - nice to have

## Known Issues

### Callback cur_frame Staleness
Some callback patterns in the VM may have stale frame references.

**Location**: `crates/vm/src/async_vm.rs`
**Status**: Needs investigation

## Completed

- ✅ Native JSON parsing (Json.parse, Json.stringify, Json.escapeString)
- ✅ Traits with custom implementations
- ✅ Math functions (sin, cos, sqrt, pow, log, etc.)
- ✅ Reactive web framework (RWeb)
- ✅ WebSocket support with split read/write
- ✅ MVar atomic update operation
- ✅ PostgreSQL with connection pooling
- ✅ String.split and String.join
