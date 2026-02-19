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

### Probes 231-232: All Passed (2 additional probes after fix)

### Probe 233: **BUG FOUND** - Cross-module same-named functions resolve to wrong module
**Problem**: When two modules export functions with the same name (e.g.,
`alpha.transform(x) = x * 2` and `beta.transform(x) = x + 100`), qualified
calls like `beta.transform(5)` returned `10` (alpha's result) instead of `105`.

**Root cause**: In `resolve_function_call`, after finding direct candidates via
prefix scan (e.g., `beta.transform/_`), an import fallback unconditionally added
candidates from other modules (e.g., `alpha.transform/_` via imports). Both
candidates scored identically, and the alphabetical tiebreaker picked `alpha`
over `beta`.

**Fix**: Added `candidates.is_empty()` guard to the import fallback lookup.
Only search imports for candidates when no direct candidates were found from
the prefix scan.

**Commit**: `f80384b` - Fix cross-module qualified call resolving to wrong module

### Probes 234-240: All Passed (7 additional probes after fix)

## Session 9 (2026-02-19)

### Probes 241-258: All Passed (18 probes)
Covered:
- Multi-file: generic wrapper types with method chains
- Multi-file: variant constructors with named/positional fields
- Multi-file: generic fold/reduce patterns across modules
- Multi-file: recursive types (tree, list) with generic operations
- Multi-file: pipeline builders with type-safe chaining
- Multi-file: nested generic types (Option[List[Int]], etc.)
- Multi-file: trait methods returning generic types
- Multi-file: function composition with type propagation

### Probe 259: **BUG FOUND** - Variant pattern matching wrong type for non-parametric fields
**Problem**: `type Outcome[a] = Success(a) | Failure(String)` — matching
`Outcome[Int]` against `Failure(m)` incorrectly assigned `m` type `Int`
instead of `String`.

**Root cause**: The pattern binding type extraction assumed "single type
arg and single pattern field → they match". This is wrong when the variant's
field uses a concrete type (String) rather than a type parameter (a).

**Fix**: Look up the constructor's actual field types from the type definition,
build a substitution map from type params to type args, and only substitute
type parameters while keeping concrete types unchanged. Applied to both
string-based and structural extraction paths.

**Commit**: `fa3b230` - Fix variant pattern matching assigning wrong types for non-parametric fields

### Probes 260-279: All Passed (20 probes after fix)
Covered:
- Multi-file: generic functions operating on variant types
- Multi-file: cross-module pattern matching with destructuring
- Multi-file: higher-order functions with variant return types
- Multi-file: Result type with map/flatMap operations
- Multi-file: generic container operations (push, pop, peek)
- Multi-file: cross-module type aliases and re-exports

### Probe 280: **BUG FOUND** - Cross-module polymorphic return type not resolved
**Problem**: `getFirst(p: Pair[a, b]) = p.first` in a types module, called
from main as `types.getFirst(p)` then `length(first)` — failed with "type
mismatch: expected List, String, Tuple, or Array, found a".

**Root cause**: Batch inference shares fixed Var IDs across functions (Var(1)
for 'a', Var(2) for 'b' from type_name_to_type). During batch solving, these
shared Vars get unified with Named("a") from type definition field accesses.
When apply_full_subst resolves getFirst's signature, Var(1) becomes Named("a")
instead of staying as a type variable. Phase 3's collect_all_vars doesn't
detect Named("a") as a type parameter, so no TypeParam conversion happens.
The pending signature ends up with type_params: [] and ret: Named("a"),
which the HM inference can't instantiate or unify with concrete types.

**Fix**: Added normalize_leaked_type_params step after apply_full_subst in
the enrichment pipeline. Detects Named { name, args: [] } where name is a
single lowercase letter not matching any known type, and converts it back to
the corresponding Var(id). This allows Phase 3 to properly convert them to
TypeParams.

**Commit**: `8416dc8` - Fix cross-module polymorphic return type leak in batch type enrichment

### Probes 281-350: All Passed (70 additional probes after fix)
Covered:
- Cross-module generic Box/Wrapper with positional and named fields
- Cross-module Either type with multiple type params
- Polymorphic identity function at multiple call sites
- Cross-module fold, map/filter pipelines
- Cross-module recursive types (Tree, rose tree Node)
- Cross-module function composition, currying, closures
- 3-module Result operations chain
- Cross-module generic Stack container
- Cross-module variant pattern matching (Shape, Cmd)
- 3-module diamond dependency with variant types
- Cross-module trait implementations (in same and separate modules)
- Alphabetical ordering stress (ztypes module)
- Trait on generic type (Box: Mappable)
- Cross-module pipeline transformations
- UFCS .map() in cross-module functions
- sort/unique wrappers, median calculation
- Lambda with pattern match on cross-module type
- 5-module dependency chain
- Chained trait method calls on generic type
- List of functions pipeline
- Cross-module string interpolation
- Cross-module histogram with Map accumulator fold
- Generic function extract + method call (like probe 280 pattern)
- Error detection (type mismatch correctly caught)
- Multiple call sites with same generic function, different types
- Function-returning generic functions

### Probe 351: **BUG FOUND** - Type param substitution corrupts type names
**Problem**: `type Registry[a] = Registry(Map[String, a])` — pattern matching
`Registry(m)` then calling `m.insert(k, v)` failed with "no method `insert`
found for type `M?36p[String, ?36]`".

**Root cause**: Three locations in compile.rs used naive `string.replace(param, concrete)`
to substitute type parameters in type strings. When param is "a" and the type string
is "Map[String, a]", the replacement turned it into "M?36p[String, ?36]" because the
"a" inside "Map" was also replaced.

**Fix**: Replaced naive `.replace()` with the existing `substitute_single_type_param()`
function which does word-boundary-aware substitution (only replacing type params that
appear in syntactic positions — after brackets, between commas, etc.).

**Commit**: `f426c75` - Fix type param substitution corrupting type names containing param letters

### Probes 352-358: All Passed (7 probes)
Covered:
- Continuation of multi-file generic patterns after type param substitution fix

### Probe 359: **BUG FOUND** - Type variable letter collision with annotation type params
**Problem**: `mapPair2(p: Pair[a, b], f, g) = Pair(f(fst2(p)), g(snd2(p)))` —
when `f` and `g` change the return type (e.g., `toString` and `length`), the
call failed with "type mismatch: expected String, found Int".

**Root cause**: When building HM signatures, `Named("a")` from annotations and
`Var(?f_ret)` from inference both formatted to letter "a". The signature became
`Pair[a, b] -> (a -> a) -> (b -> b) -> Pair[a, b]` instead of the correct
`Pair[a, b] -> (a -> c) -> (b -> d) -> Pair[c, d]`, over-constraining return types.

**Fix**: Added `collect_named_type_param_letters()` to find letters already used
by Named types from annotations. When assigning letters to Var IDs, those reserved
letters are skipped to avoid collisions.

**Commit**: `34ce960` - Fix type variable letter collision with annotation type param names

### Probes 360-389: All Passed (30 probes after fix)
Covered:
- Annotated vs unannotated generic functions with type-changing HOFs
- Cross-module annotated generic functions
- Pipeline operations with annotated params
- BST operations (insert, inorder traversal)
- Generic fold/reduce with annotated accumulator types
- Recursive annotated generic functions

### Probe 390: **BUG FOUND** - Annotation type params fail trait bound checks
**Problem**: `bInsert(t: BTree[a], val) = ... val < x ...` failed with
"type parameter 'a' must have 'Ord' trait bound" even though the same function
works without annotations.

**Root cause**: Three interrelated issues:
1. `solve()` in HM inference rejects `Named("a")` with `MissingTraitImpl` because
   deferred_has_trait treats it as a concrete type lacking trait impls
2. Trait bounds on Vars unified with `Named("a")` aren't included in signature
3. Second-pass check requires explicit `[a: Ord]` even when HM can infer the bound

**Fix**: (1) Skip single-char lowercase Named types in all three deferred_has_trait
retry passes. (2) Add `get_trait_bounds_for_named_type_params()` to collect trait
bounds from Vars resolved to Named type params. (3) Only require explicit trait
bound declarations when user declared type params explicitly.

**Commit**: `fe13951` - Fix annotation type params failing trait bound checks

### Probes 391-398: All Passed (8 probes)
Covered:
- Annotated function with inferred Num trait bound (addDouble)
- Multiple annotated params from same type, returning List[a]
- Cross-module annotated generic function (wrapInList, doubleWrap)
- Nested generic type with annotations (Option[List[a]])
- Annotated recursive function on generic type (myLen)
- Cross-module generic record type (Box[a] with makeBox/unbox)
- Annotated function returning different generic type than input
- Two annotated functions composed together

### Probe 399: **BUG FOUND** - Comparison ops on annotation type params emit Int-specific instructions
**Problem**: `clampedAdd(x: a, y: a, maxVal: a) -> a = if x + y > maxVal then maxVal else x + y`
crashed with "Panic: GtInt: expected Int64" when called with Float arguments.

**Root cause**: `is_current_type_param()` only recognized uppercase single-letter type params
(A, B, C...), not lowercase ones (a, b, c...) from annotations. So for `x: a`, the type "a"
wasn't recognized as a type parameter, and `>` was compiled as `GtInt` instead of generic `Gt`.

**Fix**: Extend the single-letter type param check in `is_current_type_param()` to include
both uppercase and lowercase letters.

**Commit**: `ff5e74e` - Fix comparison ops on lowercase annotation type params

### Probes 400-486: All Passed (87 probes after fix, including parallel agent 421-486)
Covered:
- Cross-module 3-module chain with annotated generic functions
- Multi-file mutual recursion with imports
- Multi-file with overloaded function names across modules
- Multi-file with trait impl in separate module (both alpha orders)
- Multi-file with generic type in one module, trait impl in another, use in third
- Multi-file with 4+ modules in a chain
- Multi-file where module names affect compilation order (z before a)
- Cross-module pattern matching on imported variant types
- Cross-module closures capturing imported types
- Cross-module generic record types with field access and transform
- Cross-module Option chaining, Result types
- Cross-module trait implementations with comparisons
- Cross-module middleware chain pattern
- Cross-module with deeply nested module chains (6 modules)
- Generic function instantiation with different types cross-module

### Probe 487: **BUG FOUND** - Trait method `-> self` return type not resolved to concrete type
**Problem**: Trait method with `-> self` return type doesn't resolve on chained calls.
`c.stepForward().stepForward()` fails because the return type of the first call is
"self" (unresolved) instead of "Counter".

**Root cause**: `type_expr_to_string()` returns lowercase "self" for `-> self` return types.
Downstream code uses `.replace("Self", impl_type)` to substitute the concrete type, which
fails because "self" != "Self" (case mismatch).

**Fix**: Normalize "self" to "Self" at all three TraitMethodInfo creation points.

**Commit**: `4a1f3dc` - Fix trait method return type 'self' not normalized to 'Self'

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
| 8       | 2 (231-232)       | Cross-module same-name resolution | Yes (f80384b) | 240 |
| 9       | 18 (241-258)      | Variant pattern type for non-param fields | Yes (fa3b230) | 259 |
| 9b      | 20 (260-279)      | Cross-module polymorphic return type | Yes (8416dc8) | 280 |
| 10      | 70 (281-350)      | Type param substitution corrupts type names | Yes (f426c75) | 351 |
| 11      | 7 (352-358)       | Type var letter collision with annotations | Yes (34ce960) | 359 |
| 11b     | 30 (360-389)      | Annotation type params fail trait bounds | Yes (fe13951) | 390 |
| 12      | 8 (391-398)       | Comparison ops on annotation type params | Yes (ff5e74e) | 399 |
| 12b     | 87 (400-486)      | Trait method `-> self` not normalized | Yes (4a1f3dc) | 487 |
