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

### Probes 150-162: All Passed (13 additional probes after fix)
Covered:
- Multi-file: mapReduce pattern with map and fold across modules
- toInt/toFloat conversion in polymorphic wrappers
- Multi-file: cross-module converters with toFloat
- Multi-file: higher-order function application with named functions
- Multi-file: Option chaining with float operations (safeDivF + safeLog)
- Multi-file: accumulate/sum/product patterns
- Multi-file: Result-like variant types with bind/mapResult
- Multi-file: function composition chains (compose, applyBoth)
- Multi-file: recursive tree types with sumTree
- Multi-file: Map config pattern (getConfig/setConfig)
- Multi-file: recursive list operations (myMap/myFilter/myFold)
- Multi-file: currying with closures (add, mul, applyBoth)
- Multi-file: string operations with mapWords

### Probe 163: **BUG FOUND** - Method calls on unknown receiver type not deferred
**Problem**: `lookupOr(m, key, default) = match m.lookup(key) { ... }` failed
with "no method `lookup` found for type `unknown`".

**Root cause**: When the receiver type is "unknown" (polymorphic parameter),
the UFCS error handler only treated it as `UnresolvedTraitMethod` if the
function had explicit type params OR `current_fn_generic_hm` was set. Plain
polymorphic functions like `lookupOr(m, key, default)` had neither, so the
error was reported as `TypeError` instead of being deferred to monomorphization.

**Fix**: Always treat method calls on "unknown" receiver type as
`UnresolvedTraitMethod`. If we don't know the receiver type, we can't
determine whether the method exists — defer to monomorphization where
concrete types are available.

### Probes 164-175: All Passed (12 additional probes after fix)
Covered:
- Multi-file: trait dispatch with measureAll on homogeneous list
- Multi-file: state management with list-based state (addToState, runOps)
- Multi-file: generic pair with zipWith
- Multi-file: pipeline operations (pipe, pipe2, pipe3)
- Multi-file: counter variant type with increment
- Multi-file: validator with predicate list
- Multi-file: 3-module chain (types/ops/main)
- Multi-file: event system with handler dispatch
- Multi-file: cache pattern with Map lookup/insert
- Multi-file: math utilities using toFloat
- Multi-file: list partition/span with filter
- Multi-file: Option map/getOrElse/flatMap utilities

### Probe 176: **BUG FOUND** - Trait methods on generic types with function params
**Problem**: `Wrap: Transformable transform(self, f) = match self { Wrap(v) -> Wrap(f(v)) } end`
where `Wrap[a]` is generic — calling `w.transform(x => x * 5)` failed with
"type mismatch: expected Int, found (Int) -> Int".

**Root cause**: Two interrelated issues:
1. In `type_check_fn`, the compiled function's signature was registered via
   `trait_method_type_aliases` under the bare name `"Wrap.transform"`. This
   signature had `Var(1)` for BOTH the generic type param `a` AND the method
   param `f`, causing them to be unified (so `f: Int` instead of `f: (Int)->Int`).
2. The UFCS signature (which correctly used distinct var IDs: `Var(1)` for `a`
   and `Var(101)` for `f`) was registered later but with a `contains_key` guard,
   so it didn't overwrite the wrong one.
3. Additionally, `infer_return_type_from_method_body` returned `Named("Wrap", [])`
   (no type args) for generic types because `find_type_for_constructor` only
   returned the bare type name.

**Fix**: (1) UFCS signatures now always overwrite in `type_check_fn` and are
registered under both arity-suffixed and bare names. (2) Return type inference
now includes type params for generic constructors.

**Commit**: `6449963` - Fix type inference for trait methods on generic types with function params

### Probes 177-229: All Passed (53 additional probes after fix)
Covered:
- Multi-file: diamond dependency with generic variant types
- Multi-file: cross-module higher-order functions (apply, compose)
- Multi-file: polymorphic wrappers (Box, Pair, Identity)
- Multi-file: multi-param generic types with trait methods
- Multi-file: chained trait method calls
- Multi-file: trait methods as HOF arguments
- Multi-file: multiple trait impls on same type
- Multi-file: Functor-style map on generic types
- Multi-file: nested generic types (List[Option[T]])
- Multi-file: cross-module fold with generic accumulator
- Multi-file: state threading patterns
- Multi-file: recursive types with tree fold
- Multi-file: validator with function types (Validator[a])
- Multi-file: complex lambda chains across modules
- Multi-file: string operations, pipeline composition
- Multi-file: Map operations through generic wrappers

### Probe 230: **BUG FOUND** - Lambda param types not propagated from HM inference
**Problem**: `foldr(pairs, %{}, (pair, acc) => match pair { (k, v) -> acc.insert(k, v) })`
failed with "cannot resolve trait method `insert` without type information".
Also, the binding `m = foldr(...)` got type `"U"` (type parameter name from foldr's
signature) instead of `Map`, so subsequent `m.keys()` also failed.

**Root causes** (2 issues):
1. `get_function_param_types("foldr")` returns empty because stdlib functions
   loaded from bytecode cache don't have AST entries in `fn_asts`. Without
   expected param types, lambda arguments aren't compiled with type information,
   so the lambda parameter `acc` has no type during method dispatch.
2. `expr_type_name` handled `Type::Named` for partially-resolved HM types but
   NOT `Type::Map`, `Type::Set`, or `Type::List` (which are separate enum variants).
   So when HM inference resolved a type to `Map[String, ?121]`, `expr_type_name`
   fell through to pattern-based inference which returned the type parameter `"U"`.

**Fix**: (1) In `compile_arg_with_expected_type`, when no expected type is available
from the function signature but HM inference has resolved the lambda's type, extract
parameter types from the HM-inferred function type. (2) In `expr_type_name`, handle
`Type::Map`, `Type::Set`, and `Type::List` alongside `Type::Named` for partially
resolved types — the base type is known and sufficient for method dispatch.

## Summary

| Session | Probes before error | Bug found | Fixed | Total probes |
|---------|-------------------|-----------|-------|-------------|
| 1       | 51                | Ambiguous UFCS dispatch | Yes (823b885) | 86 |
| 2       | 5 (87-91)         | Cross-module overload resolution | Yes (be3cfa5+) | 126 |
| 3       | 34 (93-126)       | Polymorphic Map method wrappers | Yes (1fd9c9c) | 130 |
| 4       | 19 (131-149)      | Missing toFloat in BUILTINS | Yes (afe8f0a) | 150 |
| 5       | 13 (150-162)      | Unknown receiver method deferral | Yes (27f6044) | 165 |
| 6       | 11 (165-175)      | Trait method on generic type + fn params | Yes (6449963) | 176 |
| 7       | 54 (177-230)      | Lambda param types from HM inference | Yes (68b495e) | 230 |
