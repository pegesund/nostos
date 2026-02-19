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

### Probes 87-91: All Passed (5 additional probes)
Covered continuation of multi-file and generic patterns.

### Probe 92: **BUG FOUND** - Cross-module overload resolution and return type inference
**Problem**: Multiple related issues with cross-module generic functions.

**Fix**: Commits `be3cfa5`, `71eefa7`, `6457d99`, `6738f69` - Fix cross-module
overload resolution, return type inference for overloaded functions, and type var
ID collisions.

### Probes 93-126: All Passed (34 additional probes after fix)

### Probe 127: **BUG FOUND** - Polymorphic Map method wrappers fail on chained calls
**Problem**: `storeSet(s, k, v) = s.insert(k, v)` followed by
`storeGetOpt(m2, "name")` where m2 came from storeSet - failed with
"cannot resolve trait method `lookup` without type information".

**Root causes** (3 interconnected issues):
1. Map method receiver inference used wrong arg count (arg_types includes
   receiver, so Map.insert has 3 not 2)
2. Polymorphic functions returned early without computing HM signatures,
   so the base function's type parameter relationships were lost
3. Map.insert signature linked value type too tightly, causing
   UnificationFailed for heterogeneous map operations

**Fix**: Commit `1fd9c9c` - Fix type inference for polymorphic Map method wrappers

### Probes 128-148: All Passed (21 additional probes after fix)
Covered:
- Map transform with lookup/match/insert chain through generic wrapper
- Generic pipeline applying function to list elements
- Set wrapper with fold accumulator
- Record types with field access
- Generic fold wrapper, nested Option chaining
- Chained filter/map through wrappers
- Multi-file: generic Box type, Map operations, transform pipeline
- Multi-file: trait implementations, import chains (3+ modules)
- Multi-file: shapes with traits, state management
- Multi-file: recursive variant types (AST evaluator)
- Multi-file: generic Stack container
- Multi-file: cross-module compose/applyTo
- Multi-file: diamond dependency pattern

### Probe 149: **BUG FOUND** - HM inference failure for functions using toFloat
**Problem**: `safeSqrt(n) = if n < 0 then None else Some(sqrt(toFloat(n)))`
failed when used as argument to a higher-order function. Error: "type mismatch:
expected Int, found Option[?28]".

**Root cause**: The `toFloat` numeric conversion function (and all `to*` variants
like `toInt`, `toInt32`, etc.) were handled as special cases in the compiler but
NOT registered in the BUILTINS array. When `try_hm_inference` ran for `safeSqrt`,
it couldn't find `toFloat` in the environment, causing inference to fail. The
fallback AST-based signature `a -> a` was used, making `safeSqrt` look like an
identity function instead of `Int -> Option[Float]`.

**Fix**: Added all `to*` numeric conversion builtins to the BUILTINS array with
proper signatures (e.g., `toFloat: a -> Float`, `toInt: a -> Int`). Excluded
these from the builtin shadowing check since user code may define trait methods
with these names.

### Probes 150: Passed (1 additional probe after fix)
Covered:
- Multi-file: mapReduce pattern with map and fold across modules

## Summary

| Session | Probes before error | Bug found | Fixed | Total probes |
|---------|-------------------|-----------|-------|-------------|
| 1       | 51                | Ambiguous UFCS dispatch | Yes (823b885) | 86 |
| 2       | 5 (87-91)         | Cross-module overload resolution | Yes (be3cfa5+) | 126 |
| 3       | 34 (93-126)       | Polymorphic Map method wrappers | Yes (1fd9c9c) | 130 |
| 4       | 19 (131-149)      | Missing toFloat in BUILTINS | Yes | 150 |
