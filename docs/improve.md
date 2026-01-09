# Nostos Improvement Ideas

Ideas and potential improvements for the language.

## Language Features

### Custom Trait Method Dispatch in Generics
Currently trait bounds work for built-in traits (Hash, Show, Eq, Copy) but custom trait methods can't be called on generic types.

```nostos
# Works: built-in traits
showable[T: Show](x: T) -> String = show(x)

# Doesn't work: custom trait methods
trait Describable
    describe(self) -> String
end
describeIt[T: Describable](x: T) -> String = x.describe()  # Error
```

**Priority**: Medium - would enable more flexible generic code

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

## Standard Library

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

**Implemented:**
- ✅ "Did you mean?" suggestions for method typos (e.g., `s.lenght()` suggests `length`)
- ✅ Source code context with line numbers and highlights
- ✅ UFCS explanation in method-not-found errors
- ✅ Type mismatch hints (e.g., `show(42) ++ " items"` for Int+String)
- ✅ Trait implementation hints for Num errors (String/Bool + arithmetic)
- ✅ Exhaustive pattern match checking (detects missing cases for Bool, Option, variants)

**Priority**: Low - core features complete

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
- ✅ String.drop and String.take
- ✅ Mutual recursion (two-phase compilation - no forward declarations needed)
- ✅ Auto-derived traits (Show, Eq, Hash, Copy work automatically for all types)
- ✅ Trait bounds on generics (`[T: Eq]`, `[T: Hash + Eq]` for built-in traits)
