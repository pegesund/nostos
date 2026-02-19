# Type Inference Probe Results

Systematic probing of the Nostos compiler's type inference, focusing on
multi-file projects and two-phase compilation.

## Session 1 (2026-02-19)

### Probes 1-51: All Passed
Covered:
- Cross-module generic functions with trait bounds
- Cross-module generic types (variants, records)
- Cross-module trait implementations
- Diamond dependency patterns
- Alphabetical ordering stress tests for two-phase compilation
- Recursive generic types
- Higher-order functions and closures across modules
- Pipeline/state monad patterns
- Nested generics (List[Option[Int]])
- 5-module dependency chains

### Probe 52: **BUG FOUND** - Ambiguous UFCS dispatch on polymorphic receivers
**Problem**: `storeGet(store, key) = store.get(key)` dispatched to `List.nth`
instead of `Map.get` when called with a Map argument.

**Root cause**: When receiver type is a type variable, the compiler fell
through all type-based dispatch tables and used the unqualified `get` builtin
(which is the List version `[a] -> Int -> a`).

**Fix**: Added `is_ambiguous_builtin_method()` check. Methods that exist on
multiple builtin types (`get`, `set`, `contains`) now trigger
`UnresolvedTraitMethod` when the receiver is a type variable, forcing
monomorphization with concrete types.

**Commit**: `823b885` - Fix ambiguous UFCS method dispatch on polymorphic receivers

### Probes 53-86: All Passed (34 additional probes after fix)
Covered:
- Generic `contains` on Map, Set, String (monomorphization creates correct variants)
- Same generic function called with different types at different call sites
- Generic `get` working for both Map and List at different call sites
- Cross-module generic Map utils (lookup, register, etc.)
- Cross-module types + Map method dispatch
- Reverse alphabetical module ordering (types in 'z' module, funcs in 'a')
- Builder pattern with Map operations across modules
- Higher-order function chains (compose, applyTwice)
- Option chaining through generic wrappers
- Generic pair mapping, Either type, currying
- 4-module diamond dependency with variant types
- 5-module dependency chain with aggregation functions
- Registry pattern with generic CRUD on maps across modules
- Error detection still works (wrong types, nonexistent methods)

## Summary

| Session | Probes before error | Bug found | Fixed | Total probes |
|---------|-------------------|-----------|-------|-------------|
| 1       | 51                | Ambiguous UFCS dispatch | Yes (823b885) | 86 |
