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

### Probes 488-519: All Passed (32 probes)
Covered:
- Trait method returns generic type (Option[Self], List[Self])
- Multi-file trait methods returning Self on generic types
- Multi-file trait impls on multiple types (Circle, Rectangle) in same module
- Multi-file trait with multiple methods (area, perimeter)
- Deeply nested multi-file chains with trait method calls
- Cross-module trait impl with same method name
- String trait impl (measureLen)
- Cross-module trait hierarchies

### Probe 520: **BUG FOUND** - UFCS key includes type args for concrete generic trait impls
**Problem**: `Either[Int, String]: Mappable` with `mapLeft(self, f)` method — calling
`e1.mapLeft(x => x * 5)` fails to find the method.

**Root cause**: When registering trait method UFCS signatures, the type name included
type args (e.g., `Either[Int, String].mapLeft/_,_`). But `get_type_name()` returns just
the base name (`Either`), so `check_pending_method_calls` looks up `Either.mapLeft/_,_`
which doesn't match.

**Fix**: Strip type args from both `unqualified_type_name` and `qualified_type_name`
when forming UFCS keys. E.g., `Either[Int, String]` → `Either`.

**Commit**: `7087c27` - Strip type args from UFCS key when registering trait method signatures

### Probes 521-586: All Passed (66 probes, parallel agent)
Covered:
- Generic functions with multiple type params in nested types
- Cross-module callbacks returning generic types
- Pattern matching on nested variant types (Option[List[Int]])
- Cross-module recursive Tree[a] traversal
- Higher-order currying with type params
- Multi-file Map operations through generic wrappers
- Mutual recursion with annotations
- Newtype wrapper with trait impl
- Lambda with complex pattern match on variant type
- Generic function at 3+ different type instantiations
- Closure capturing (by-value semantics)
- Cross-module variant construction and matching
- Generic Either type with map/bind operations
- Multi-type-param types (Triple[a,b,c])
- Recursive variant traversal (custom List2)
- Option chaining transformations
- Polymorphic list operations (myLength, myReverse)
- Pattern matching with or-patterns
- Multi-file generic Stack container
- Deeply nested variant match (3 levels)
- Nested lambda with captured variables
- Tuple patterns in various contexts

### Probe 590: **BUG FOUND** - Cross-module trait method lookup fails for impls on builtin/generic types
**Problem**: `List[Int]: Summable` with method `total(self)` in module `types.nos` —
calling `nums.total()` from `main.nos` fails with "no method `total` found for type `List[Int]`".

**Root cause**: Two issues:
1. `type_traits` map was only keyed on the qualified type name (e.g., `"types.List[Int]"`)
   but `find_trait_method` tried short names like `"List"`, `"List[Int]"` — none matched.
2. Even after finding the trait, function name construction used the base type name `"List"`
   but the actual function was registered with type args (e.g., `"types.List[Int].types.Summable.total"`).

**Fix**: (1) Also register `type_traits` entries under the short base type name without
type args and module prefix. (2) In `find_trait_method`, also try constructing function
names with the original type name including type args.

**Commit**: `55919ec` - Fix cross-module trait method lookup for impls on builtin/generic types

### Probes 591-780: All Passed (190 probes, parallel agents)
Covered:
- Cross-module trait impl on String (shout method)
- Multiple trait impls in same module (Int, String)
- Trait impl separated: type in module A, trait in B, impl in C
- Trait with multiple methods (double, triple)
- Type def + trait impl in same module (Shape: HasArea)
- Type from one module, trait impl in another
- Re-export patterns through intermediate modules
- Generic types (Box[a], Maybe[a], Wrapper[a]) across modules
- Higher-order functions, closures, currying across modules
- Forward references (two-phase: type/trait defined after use)
- Mutual recursion (isEven/isOdd)
- Deeply nested module chains (4-5 levels: a→b→c→d→main)
- Recursive types (Tree, IntList) across modules
- Variant with 4+ constructors
- Generic wrapper with map-like operations (mapBox, unbox)
- Set/Map operations across modules
- Record types with multiple fields
- Pipeline patterns (filter/map/fold chains)
- Chained trait method calls returning Self
- Pattern matching on nested variants
- Cross-module exceptions and try/catch
- Trait impls on Bool, String, List[Int], List[String], Option[Int], Result[Int, String]
- Trait on imported generic type with concrete type arg (Pair[Int]: Summable)
- Multiple traits on same type from different modules
- Variant with both positional and named fields
- Three levels of type nesting (A, B { inner: A }, C { inner: B })
- Polymorphic functions called with different types at multiple call sites
- MVar usage across modules
- Spawn/await across modules
- Complex 5-module project (model + traits + impls + utils + main)

### Probe 782: **BUG FOUND** - UFCS key collision for multiple concrete generic trait impls
**Problem**: `Holder[Int]: Measurable` and `Holder[String]: Measurable` both register UFCS
signatures under the same key `"Holder.measure/_"`. The second overwrites the first, so
calling `.measure()` on the "wrong" instantiation fails with "expected String, found Int".

**Root cause**: When type args are stripped from the UFCS key (fix from probe 520), the
concrete self type in the signature retains the specific type args. Multiple impls
overwrite each other. E.g., `Holder.measure/_` stores `Holder[String] -> Int`, then
`Holder(5).measure()` tries to unify `Holder[Int]` with `Holder[String]`.

**Fix**: For concrete generic impls (where impl specifies type args like `Holder[Int]`),
use fresh type variables for the type args in the UFCS self parameter. So the signature
becomes `Holder[?X] -> Int` which works for any instantiation.

**Commit**: `293fc89` - Generalize UFCS self type for concrete generic trait impls

## Session 16 (2026-02-20)

### Probes 783-788: All Passed
Covered:
- Trait impls on user-defined generic types (single-file and cross-module)
- Cross-module trait with multiple methods
- Multi-module pipeline patterns with traits

### Probes 789-790: **BUG FOUND** - Trait impl on parameterized types produces wrong function key
**Problem**: `List[a]: Countable` with `count2(self)` compiles but fails at runtime with
`Function not found: List.Countable.count2/_`. Same for cross-module version.

**Root cause**: Function key was built as `"List[a].Countable.count2/List[a]"` (using the
full type name with type args), but `find_trait_method` looks up by base name
`"List.Countable.count2/"` prefix. The `[a]` suffix in the key prevented the prefix match.

**Fix**: Strip type args from the type name when building the function key in both the
forward declaration and body compilation passes. Also register `type_traits` under the
module-qualified base name (e.g., `"types.Either2"`) in addition to the full name and
short base.

### Probe 827: **BUG FOUND** - Cross-module trait impl on concrete generic not found
**Problem**: `Either2[Int, String]: Describable` defined cross-module. Calling
`Left2(42).describe()` fails with `no method 'describe' found for type
'types.Either2[Int, ?209]'` because `b` is unresolved from `Left2(42)`.

**Root cause**: `type_traits` was registered under `"types.Either2[Int, String]"` and
`"Either2"` but NOT under `"types.Either2"` (the module-qualified base without type args),
which is what `find_trait_method` looks up.

**Fix**: Part of same commit - also register `type_traits` under the module-qualified
base name without type args.

**Commit**: `ff46dc8` - Fix trait impl on parameterized types

### Probes 791-830, 831-880: All Passed (90 probes)
Covered:
- Polymorphic function instantiation, higher-order functions
- Recursive variants with trait impls
- Multi-module patterns: diamond dependencies, game entities, 3-4 module chains
- Record types with trait impls, sortBy, fold patterns
- Cross-module generic accumulators, mutual recursion
- Two traits on same type from different modules

### Probes 881-930: All Passed (50 probes)
Covered:
- Nested generic types (Option[List[Int]], Option[Option[Int]])
- Cross-module trait impl variants
- Record types with generic field access
- Pipeline patterns with annotated types
- Recursive generic functions on custom types

### Probes 931-980: All Passed (50 probes + 80 variants)
Covered:
- Nested generics: `Option[List[Int]]`, `Option[Option[Int]]`, `Result[List[Int], String]`
- Generic type definitions with pattern matching (Wrapper, Pair, Tree)
- Method chaining: `.map().filter().fold()` chains
- Higher-order functions: compose, curried functions, partial application
- Cross-module: generic Option/Box functions, re-export patterns (A->B->C)
- Cross-module: overloaded functions, trait impls for shapes
- Edge cases: conditional returning generic type, fold with different acc type
- Extreme cases: 80 additional variant/edge/fragile/extreme probes all passed

### Probes 981-1030: All Passed (50 probes)
Covered:
- Multiple concrete trait impls on same generic type (Pair[Int,Int] and Pair[String,String])
- Trait impl on nested generic (Option[List[Int]])
- Trait on builtin generic (List[Int])
- Two-phase compilation ordering: reverse alphabetical, forward references, circular deps
- Trait/type/impl across 3-4 different modules
- Complex generic instantiation: nested generics, currying, function composition
- Pattern matching: nested Option[Option[Int]], list patterns, recursive expression trees
- Cross-module supertrait relationships
- Monadic bind chains with pipeline operator

### Probes 1031-1080: All Passed (50 probes)
Covered:
- Closures: capturing generic functions, records, nested closures, cross-module closures
- Method chaining: `.map(f).filter(g).map(h)` with type changes, `.sort().map()`, `.flatMap()`
- Recursive types across modules: Expr evaluator, mutual recursion, generic Tree[a]
- Polymorphic functions: 4+ different type instantiations, passed as HOF arguments
- Complex trait+generic: two traits on same type, recursive trait methods, cross-module impls
- Action/state reducer pattern with variant match + record + closure + fold

### Probes 1081-1130: All Passed (50 probes)
Covered:
- Trait impls with Self return type on GENERIC types (Box[a]: Clonable, clone preserves type)
- Chained `.clone().clone()` on generic types
- Complex 5-6 module projects with chain and diamond dependencies
- Lambda type inference through 3+ indirection levels
- Cross-module HOF with lambdas, nested/curried lambdas
- Tuple destructuring in lambdas, lambdas calling trait methods
- Map literal construction, keys/values/toList/fromList
- Fold accumulating into Map, record pipelines
- Error detection: missing methods, type mismatches, missing imports, arity errors

### Probes 1131-1180: All Passed (50 probes)
Covered:
- Empty list operations ([].length(), [] ++ [1,2,3], [].map(...))
- Let polymorphism (top-level generic functions work polymorphically)
- Type narrowing through pattern match, cross-module generic inference
- Chained identity calls, empty Map insert/lookup chains
- 5-module trait/type/impl separation across different modules
- Trait impl body calling functions from third modules
- Two traits on same type, variant trait impls matching all constructors
- Generic Option piped to other generic functions, List[Option[Int]] filtering
- Generic recursive types (RoseTree) with fold
- Cross-module overloaded dispatch (same name, different types/arities)
- 5-step method chains, deeply nested generic types
- Functions with 5+ differently-typed parameters
- 4-module data transformation pipeline

### Probes 1181-1230: All Passed (50 targeted probes)
Covered (deliberately targeting previously-buggy code paths):
- Trait dispatch key resolution: List[a]/Option[Int] trait impls, cross-module 4-module chains
- Two traits on same List[Int], trait method taking List param, chain of 3 trait calls
- Recursive trait methods on generic variants, cross-module Option[Int] impls
- Two-phase compilation: reverse alphabetical ordering, mutual function deps, 5-module projects
- Module re-exports, private function visibility, mixed qualified/wildcard imports
- 5-level deep module chains (A->B->C->D->main)
- Lambda/closure inference: cross-module HOF, Map fold accumulator, nested lambdas
- Record creation in map, type-transforming pipeline lambdas
- Type var unification: same generic fn at different types, trait bounds + concrete types
- Recursive generic functions, closure captured vars constraining type variables

### Probes 1231-1280: All Passed (50 targeted probes)
Covered (targeting specific known-tricky code paths):
- Trait Self return type on generics: Box[a]: Mappable with mapSelf -> Self
- Chained Self-returning methods (.appendVal(1).appendVal(2).appendVal(3))
- Two traits with -> Self on same type, Option[Self] return, List[Self] return
- Cross-module Self on record types, Self on variant types, Self with 3 type params
- Variant pattern matching type propagation: Result[String,Int], Either[List[Int],String]
- Nested match on Option + inner variant, 3+ field constructors, unit + param ctors
- Cross-module pattern matching, recursive Tree matching, method call result matching
- Polymorphic function recompilation: Int+String in same body, swapped type param orders
- Cross-module chained generics, polymorphic fn as HOF argument, recursive cross-module
- Constraint solver ordering: trait bound after type resolve, method chain intermediate types
- map(f).sort() where f changes type, empty list type from later constraint

### Probes 1281-1330: Error Detection Probes (25 false-positive + 25 false-negative)
**Focus**: Testing that valid code is NOT rejected (false positives) and invalid code IS rejected (false negatives).

**False-positive tests (25/25 passed)**: Valid code correctly accepted:
- Nested generic types, lambda inference, cross-module imports
- Trait method calls, pattern matching, pipeline chains
- Higher-order functions, recursive types, closures

**False-negative tests (22/24 passed, 1 bug found, 1 design question)**:

### Probe 1325: **BUG FOUND** - `use NonExistent.*` silently succeeds
**Problem**: `use NonExistent.*` doesn't produce an error. The compiler silently ignores
imports from modules that don't exist, which means typos in module names go undetected.

**Root cause**: `compile_use_stmt` didn't validate that the referenced module actually
exists before processing the import. It would simply find no functions/types with the
module prefix and import nothing, silently.

**Fix**: Added module existence validation at the start of `compile_use_stmt`. Checks
`known_modules`, `function_visibility`, and `type_visibility` for the module path. If
none contain the module, returns `CompileError::TypeError` with "module `X` not found".

**Tests**: All 1436 tests pass after the fix.

### Probes 1331-1380: All Passed (50 probes)
Covered:
- Lambda returning lambda (higher-order closures, currying patterns)
- Method chains returning different types at each step
- Type inference through complex if/match chains
- Generic functions with empty collections
- Partial application patterns
- Pipeline operator with functions
- Mutual recursion across modules (isEven/isOdd)
- Cross-module closures returning generic types
- Deeply nested pattern matching on generic variants
- Recursive types with multiple type params
- Import shadowing (first import wins for wildcards)
- Nested generics (Option[Option[Int]], Result[Int, String])
- Closures with captures from multiple scopes
- Cross-module types with builder patterns

### Probe 1413: **BUG FOUND** - Option/Result variant type_name mismatch in native functions
**Problem**: `Map.lookup(m, "a") == Some(1)` returns `false` even though both display
as `Some(1)`. Also `Map.lookup(m, "z") == None` returns `false`.

**Root cause**: Native VM functions (Map.lookup, Regex.find, Regex.captures) create Option
variants with type_name `"Option"`, while compiled Nostos code creates them with the
fully qualified `"stdlib.list.Option"`. The variant equality check (`gc_values_equal`)
compares `type_name` fields, so they don't match.

Same issue affects Result: `make_ok_variant`/`make_err_variant` used `"Result"` instead
of `"stdlib.list.Result"`.

**Fix**: Changed all native function Option/Result variant creation in async_vm.rs to
use the fully qualified type names: `"stdlib.list.Option"` and `"stdlib.list.Result"`.
7 occurrences fixed across Map.lookup, Regex.find, Regex.captures, parse_string_to_type,
make_ok_variant, and make_err_variant.

**Tests**: All 1436 tests pass after the fix.

### Probes 1381-1430: All Passed (50 probes, after fix)
Covered:
- Recursive expression evaluator with cross-module variant type
- Nested record field access
- Map.lookup with Option equality (now works after fix)
- Triple[A,B,C] with mapFirst
- Nested lambdas capturing outer scope
- Cross-module record pattern matching
- Filter/map comprehension chains
- Generic fold producing string
- Recursive variant type (MyList[T]) with myLen/myMap
- Mutual recursion with variant pattern matching
- Cross-module trait impl in main module
- Complex fold producing records
- Multiple type params with mapBoth
- Cross-module exception throwing and catching

### Probes 1431-1480: All Passed (50 probes)
Covered:
- Cross-module record types with pattern matching (field access + destructuring)
- Deeply nested cross-module chains (4-5 level A->B->C->D->main)
- Two traits on same type, both imported cross-module
- Trait methods taking trait-bounded arguments
- Map.lookup equality (Some/None comparison works after fix)
- Result type from try/catch, cross-module exceptions
- Cross-module closures capturing imported functions
- Nested variant matching (2-3 levels: Some(Ok(x)), Some(Some(Some(v))))
- Default parameters in function signatures
- List operations on cross-module types (.map().filter().sum())
- Pattern matching on imported record fields
- Multiple imports from same module (use M.{f, g, h})
- Qualified calls mixed with unqualified in same function
- Recursive variant types cross-module (Expr tree evaluator)

### Probes 1481-1530: All Passed (50 probes)
Covered:
- Type-changing fold (accumulator type differs from element type)
- Function returning function returning function (3-level closures)
- Complex pipeline with type changes at each step
- Match on string literal patterns
- List of variant values with map/filter on constructors
- Nested match with destructuring on generic variants
- Higher-order: generic fn taking generic fn as arg
- Cross-module diamond dependency with shared types
- Cross-module recursive types (Expr = Lit | Add | Mul)
- Generic function composition
- Equality comparison on Option (verifying fix)
- Generic functions with Num/Eq constraints
- Mutual recursion patterns
- Tuple construction and element access
- Chained method calls (filter.map.fold)

### Probes 1531-1580: All Passed (50 probes - complex integration)
Covered:
- Trait impl + generic type + cross-module + pattern match (combined)
- 5-module projects (types, traits, impls, utils, main)
- Generic accumulator fold over cross-module variant list (AST evaluator)
- Trait method calling another trait method on same self
- Cross-module trait impl using functions from third module
- Lambda capturing cross-module type and using methods
- Record with Option fields, variant with named field constructors
- sortBy with cross-module record type
- Recursive data structures (BST, Peano, custom lists)
- Cross-module UFCS method chains (x.addTo().mulBy().extract())
- 4-module employee filter/aggregate pipeline
- 5-module expression evaluator with environment

### Probes 1581-1630: All Passed (50 probes - regression + boundary)
Covered:
- **Regressions verified**: Map.lookup equality, use NonExistent.* error,
  trait impl on List[a], trait -> Self on generic, cross-module polymorphic return,
  annotation type params + trait bounds
- **Empty list**: [].length(), [].map(f), [].fold(init, f), [].filter(f)
- **Single-element**: [42].head(), [42].tail(), [42].map(f)
- **Boundary**: 10+ chained method calls, 6-8 parameter functions,
  5-7 level nested if/else, 8-branch match, 3 trait impls on same type
- **Cross-module re-export chain**: A -> B -> C
- **Error detection**: private function access, wrong constructor arity

### Probes 1631-1680: All Passed (50 probes - creative patterns)
Covered:
- Recursive descent parser pattern (cross-module variant AST)
- State machine with variant types and transitions
- Builder pattern with method chaining on records
- Interpreter pattern: Expr type with environment Map
- Generic container with multiple operations (push, pop, peek, size)
- Cross-module visitor pattern on variant types
- Fold-based computations on deeply nested types
- Generic pair/triple types with mapFirst, mapSecond
- Complex pipeline with type changes at every step
- Lambda factories (functions returning lambdas based on args)
- Cross-module trait hierarchies (3 traits, 2 types)
- Recursive tree operations (depth, flatten, map)
- Sort/filter on records with field access
- Higher-order closures with multiple captures

### Probes 1681-1730: All Passed (50 probes - multi-file two-phase compilation)
Covered:
- Type defined in alphabetically-last module (z_ prefix), used from a_ prefix
- Transitive import chains (3-4 levels deep: A -> B -> C -> D)
- Diamond imports (two modules both importing shared module)
- Trait in separate module from type and impl (3-file separation)
- Module with only types vs only functions
- Generic types/functions cross-module (Box[a], MyPair[a,b], Either[a,b])
- Selective vs wildcard imports (`use lib.{valA, valB}`)
- Public/private visibility (private helper, public function)
- Cross-module type used in 4+ modules (Token type in 5 modules)
- Recursive types cross-module (Tree, MyList, BinOp AST)
- Higher-order functions cross-module (applyTwice, addN, factories)
- Complex 5-file projects (type, trait, impl, helpers, main)

### Probes 1731-1780: All Passed (50 probes - type inference edge cases)
Covered:
- Type variable escape through closures (lambda captures, nested closures, stored lambdas)
- Constraint ordering stress (trait bounds before type, retroactive chain constraining)
- Cross-module generic instantiation (5+ types, aliased imports, transitive generics)
- Record/variant inference (generic fields, update patterns, two records same field name)
- Method resolution (.map on List vs Option, chain on literal, methods on match/if results)
- Error handling integration (try/catch returning Option, nested Result[Option[Int]])

### Probes 1781-1830: All Passed (50 probes - advanced multi-file)
Covered:
- Circular module dependencies (mutual A<->B, 3-cycle A->B->C->A, diamond+circular)
- Module importing itself, re-export circular patterns
- Shadowing/name resolution (local shadows import, lambda param shadows outer, import shadows builtin)
- Two wildcard imports with same name, qualified vs unqualified access
- Complex pattern matching (guards, nested tuples, string literals, Bool exhaustive)
- Nested variant patterns (Some(Ok(x)), Some(Err(e)), None)
- Default parameters (0-3 args, complex defaults, cross-module, named args)
- Concurrency (spawn/await, MVar generic, multiple spawns different types)
- Collections (Set operations, nested Map of Lists, List of Maps)

### Probes 1831-1880: All Passed (50 probes - tricky HM inference)
Covered:
- Polymorphic recursion (recursive list/tree processing, mutual recursion, fix-point patterns)
- Higher-rank polymorphism stress (identity at two types, generic args, function lists, compose)
- Complex data flow (5+ assignments, deeply nested match, shared captures, chained generics)
- Cross-module trait dispatch (same-name methods, chained traits, qualified paths, nested traits)
- Operators (comparison chains, arithmetic with generics, show(), equality on custom types)
- Edge cases (wildcard match, 50+ char fn names, Unicode, 10+ nested calls, 8 params)

### Probes 1881-1930: All Passed (50 probes - combinatorial multi-file)
Covered:
- Module ordering permutations (z_types/a_impl, 5-module chains, re-exports, trait-only modules)
- Forward reference stress (main calls later module, mutual trait impls, circular imports)
- Visibility/access control (public/private, opaque types, facade patterns, selective imports)
- Complex generic types (3 type params, nested generics, phantom types, recursive generic Stack)
- Error detection accuracy (10 tests: private access, missing imports, type mismatches, wrong arity)

### Probe 1932: **BUG FOUND** - Generic functions with custom trait bounds fail at runtime
**Problem**: `describe[T: Describable](item: T) = item.desc()` compiles but fails at
runtime with `Function not found: T.Describable.desc/_`. The trait method dispatch
emits a call to a placeholder function name using the type parameter instead of triggering
monomorphization.

**Root cause**: In `find_trait_method()`, when the type name is a type parameter (like `T`
with trait bound `Describable`), it returns a placeholder string `"T.Describable.desc"`.
The UFCS dispatch code then emitted a function call to this placeholder, which doesn't
exist at runtime.

**Fix**: Added `is_type_concrete()` checks in two locations:
1. Method-call trait dispatch (`.method()` syntax) - returns `UnresolvedTraitMethod` error
2. Function-call trait dispatch (`method(obj)` syntax) - returns `UnresolvedTraitMethod` error
Both trigger monomorphization to specialize the function with concrete types.

### Probe 1936: **BUG FOUND** - Lambda parameter types not propagated from HM inference
**Problem**: Inside lambda bodies, user-defined trait methods can't be resolved because
the lambda parameter types aren't available. E.g., `items.foldl(0, (acc, item) => acc + item.score())`
fails because `item`'s type (inferred from `List[Item]`) isn't propagated into the lambda.

**Root cause**: The `Expr::Lambda` compilation path used `compile_lambda` which doesn't
set `local_types` for parameters. The alternative `compile_lambda_with_types` does set
types, but was only called from specific paths.

**Fix**: Modified `Expr::Lambda` compilation to check `inferred_expr_types` for the
lambda expression's span. If HM inference resolved the lambda's function type with concrete
parameter types, use `compile_lambda_with_types` instead of `compile_lambda`.

**Commit**: (both fixes in same commit)

**Tests**: All 1436 tests + 14 postgres tests pass after the fix.

### Probes 1931-1980: All Passed (50 probes, after fixes)
Covered:
- Cross-module trait + generics + pattern matching (combined 3-feature tests)
- Lambda + cross-module + type inference stress (fold, sortBy, captured functions)
- PostgreSQL (basic query, parameterized, insert/select, transactions, sequences)
- Real-world patterns (config parser, calculator, state machine, event system, template engine)
- Regression prevention (all 10 previously-found bug patterns re-tested)

### Probes 1981-2030: All Passed (50 probes - adversarial)
Covered:
- Deeply nested generics (Option[Option[Option[Int]]], List[Option[Result[Int,String]]])
- String processing (split/map, chars/filter, replace, trim, toInt/toFloat)
- Exception handling (nested try/catch, exceptions in lambdas, custom error types)
- Numeric edge cases (large ints, float precision, negative modulo, conversions)
- Advanced collections (zip, zipWith, take/drop, any/all, find, groupBy, flatten)

### Probe 2074: **BUG FOUND** - Pattern-matched values from generic stdlib types lose type info
**Problem**: Calling a trait method on a value extracted via pattern matching from a generic
stdlib type (e.g., `Some(entry)` where scrutinee is `Option[Entry]`) fails because `entry`
gets type `"T"` instead of `"Entry"`.

**Root cause**: When stdlib types are loaded from bytecode cache, `type_defs` (AST-level
type definitions with type parameter names) is not populated. The pattern binding type
extraction relied solely on `type_defs` to map type parameters (like `T` in `Some(T)`)
to concrete types (like `Entry`). Without the mapping, variables were assigned the raw
type parameter name.

**Fix**: Added `type_param_names: Vec<String>` field to `TypeInfo` struct. Populated from
`TypeDef.type_params` during source compilation and `TypeValue.type_params` during cache
loading. The pattern binding extraction falls back to `TypeInfo.type_param_names` when
`type_defs` doesn't have the entry.

**Commit**: `696bb26`

### Probes 2031-2080: All Passed (50 probes, after fix)
Covered:
- Trait methods inside lambdas (map, filter, fold, nested, captured, chained)
- Generic functions with trait bounds (Num, Eq, Ord, custom, two bounds, recursive)
- Trait impls with complex bodies (pattern match, closures, builder, cross-module)
- Multi-module trait dispatch (3-4 module chains, fold over trait values, pipeline)
- Edge cases (if/else returning trait types, Map.lookup + trait method, tuple + trait)

### Probes 2081-2130: All Passed (50 probes - comprehensive multi-file)
Covered:
- Module chain stress (6-module chain, fan-in, fan-out, star, diamond+complex)
- Compilation order (reverse alphabetical, numbers in names, 6-level deep chains)
- Cross-module pattern matching (nested variants, guards, exhaustive, Option[ModuleType])
- Cross-module exceptions (3-module propagation, custom error types, Result chaining)
- Real-world apps (todo, math vectors, game entities, interpreter, config, event bus, router, ORM)

### Probe 2141: **BUG FOUND** - Trait methods with default parameters fail on UFCS calls
**Problem**: `n.add()` where `add(self, x: Int = 5)` has a default parameter fails with
"Wrong number of arguments: expected 2, found 1".

**Root cause**: Three locations in compile.rs needed fixes:
1. Trait impl forward declaration `FunctionValue` had `required_params: None`
2. UFCS signature `FunctionType` had `required_params: None`
3. Trait dispatch argument compilation didn't fill in default values for missing arguments

**Fix**: Compute `required_params` from method clause parameters at both registration
points. Also fill in default values at the trait dispatch argument compilation path.

**Commit**: `dd92d6c`

### Probes 2131-2180: All Passed (50 probes, after fix)
Covered:
- Cross-module default parameters (function call defaults, module value defaults, overloaded)
- Trait method default params (single-file and cross-module, multiple types, pattern matching)
- Generic constructors in complex positions (Some in map, nested Some, fold accumulator, pipeline)
- Method chaining on trait method results (.map().filter(), Option match, cross-module)
- Closures capturing trait-bounded values (nested closures, cross-module, combined with defaults)

### Probes 2181-2230: All Passed (50 probes - adversarial corners)
Covered:
- Mutual recursion + generics (isEven/isOdd, Tree processing, cross-module, expression eval)
- Complex pattern match + type propagation (Result[Option[Int]], nested 3 layers, 4+ constructors)
- Higher-order functions + trait bounds (applyTwice, compose, filter+map chain, named fn args)
- Record field access chains (3-level deep, cross-module, generic records, map over fields)
- Pipeline operator stress (5+ steps, type changes, cross-module, feeding into match)

### Probe 2239: **BUG FOUND** - Named arguments in UFCS trait method calls assigned to wrong positions
**Problem**: `s.adjusted(scale: 3)` where `adjusted(self, offset: Int = 0, scale: Int = 1)`
assigns `3` to `offset` instead of `scale`, returning `(5+3)*1=8` instead of `(5+0)*3=15`.

**Root cause**: Two UFCS code paths in compile.rs compiled arguments purely positionally
without checking for `CallArg::Named` variants. Module-qualified calls and trait method
dispatch both ignored named argument names.

**Fix**: Added named-arg reordering logic to both UFCS paths, matching the approach already
used in `compile_call()`. For trait method UFCS, accounts for `self` being at position 0.

**Commit**: `8e7fbbb`

### Probes 2231-2280: All Passed (50 probes, after fix)
Covered:
- Trait method default param regression (single-file, cross-module, multiple types, chained calls)
- Type inference through let bindings (lambda binding, conditional, HOF, nested closures)
- Recursive variant types with methods (binary tree, linked list, expression eval, cross-module)
- Complex fold patterns (record accumulator, Map building, different accumulator type, nested folds)
- Mixed feature stress (trait+generic+cross-module, default+match+pipeline, closure+trait+fold)

### Probes 2281-2330: All Passed (50 probes - dispatch ordering + instantiation)
Covered:
- Cross-module trait dispatch ordering (3-file separation, multiple traits, supertrait, chained)
- Generic function instantiation (4+ types, generic calling generic, composition, cross-module)
- Pattern matching + type narrowing (wildcard, recursive, list patterns, guard conditions)
- String processing (split+map+fold, chars+filter, comparison chains, method chaining)
- Concurrency (MVar with generic types, multiple spawns, channel-like patterns)

### Probes 2331-2380: All Passed (50 probes - named args, traits, generics)
Covered:
- Named args in regular/cross-module/trait calls, reordering, skipping middle defaults
- Trait inheritance/composition (supertrait chains, diamond, cross-module, same method names)
- Generic record types (Box[a], Pair[a,b], nested generics, cross-module)
- Error propagation (try/catch, nested, cross-module, in lambda/fold)
- Complex type instantiation (List[Option[Int]], Option[List[String]], Result[List[Int], String])

### Probes 2381-2430: All Passed (50 probes - multi-module, real-world)
Covered:
- Multi-file 4-5 module projects (chain, fan, diamond, re-export, generic types)
- Trait impls on builtin types (Int, String, Bool, Float, custom variants)
- Complex lambda/closure patterns (currying, triple-nested, composition, closure capture)
- Operator overloading via Num/Ord/Eq traits (Money, Vec3, Rational, Complex numbers)
- Real-world application patterns (calculator, JSON builder, state machine, interpreter, registry)

### Probes 2431-2480: All Passed (50 probes - deep adversarial)
Covered:
- Type variable escape in closures (capture + generic fn, closure factories, nested closures)
- Constraint solver ordering stress (unconstrained vars, delayed resolution, chain filtering)
- Cross-module type alias-like patterns (generic types, records, variants, recursive types)
- Complex method resolution (same name on different types, chained calls, pattern-matched values)
- Edge cases in pattern matching (wildcard nested, tuples, 3+ deep constructors, guard-like)

### Probes 2481-2530: All Passed (50 probes - two-phase compilation stress)
Covered:
- Forward references across modules (later alphabetical, diamond, 5-module cascade)
- Visibility edge cases (private type, public constructor, access denied errors, selective import)
- Module naming (numbers, single-letter, underscores, 10 modules, type-name match)
- Recompilation and cache (value change, arity change, add/remove exports, cache invalidation)
- Import system stress (5-item selective, 8-export wildcard, re-export, duplicate import)

### Probes 2531-2630: All Passed (100 probes - deep generic patterns)
Covered:
- Deep generic types (Option[List[Int]], Result[List[String], Int], nested 3-deep)
- Cross-module generic function chains with multiple type params
- HOF with recursive match-based patterns
- Lambda factories and closures with trait method calls
- Complex multi-module pipeline patterns
- Generic type instantiation at 4+ different types
- Cross-module trait impls on generic types

### Probes 2631-2680: All Passed (50 probes - multi-file adversarial)
Covered:
- Cross-module generic types with trait methods and UFCS chains
- Diamond dependency patterns with generic functions
- Nested lambdas with type inference across modules
- Two-phase compilation alphabetical sensitivity (aaa depends on zzz)
- Trait impl in alphabetically-earlier module than type def
- Chain of generic function calls across 3 modules
- Cross-module overloaded function resolution
- Recursive generic functions across modules
- Multiple trait impls on same type from different modules
- 4-module chains with type flow

### Probe 2681: **BUG FOUND** - Widely-implemented trait method inferred to wrong type
**Problem**: `makeFormatter() = x => x.show()` fails with "type mismatch: expected Panel,
found Int". The unique-type inference (from probe 2608 fix) incorrectly resolves the
receiver to `Panel` because `Panel.show` is the only explicit UFCS entry found.

**Root cause**: The unique-type inference searched `env.functions` for `{TypeName}.{method}`
patterns and found only `Panel.show` for the `show` method. But `show` is a method of the
`Show` trait which is implemented for ALL primitive types (Int, Float, Bool, Char, String)
via auto-derived impls. These auto-derived impls don't create UFCS entries in env.functions,
so the search incorrectly identifies Panel as the unique implementing type.

**Fix**: Before doing unique-type inference, check if the method belongs to a trait that has
2+ implementations registered in `env.impls`. This covers all widely-implemented traits
(Show, Hash, Eq, Ord) whose methods work on many types. The check is fully general - it
uses trait definition and implementation data rather than hardcoded method names.

**Commit**: `423a2a5` - Skip unique-type inference for methods from widely-implemented traits

### Probes 2682-2710: Named args in trait UFCS not resolved during HM inference
**Problem**: Cross-module trait method with named/default parameters called via UFCS
(e.g., `s.doSetup(port: 3000)`) assigns named args to wrong positions because param
names aren't found during HM inference.

**Root cause**: `check_pending_method_calls` in infer.rs looked up param names under
the bare method name (e.g., `"setup"`) but trait UFCS methods register param names under
the qualified name (e.g., `"Server.setup"`).

**Fix**: Two changes:
1. In infer.rs: Also try looking up param names under `qualified_name` when reordering named args
2. In compile.rs: Register param names for trait UFCS signatures under the qualified base name

**Commit**: `f94f6b0`

### Probes 2731-2780: All Passed (50 probes - trait interaction + widely-implemented traits)
Covered:
- show()/hash() on generic params in lambdas, HOFs, stored in variables
- User-defined traits with single and multiple impls in lambdas and pipelines
- Cross-module traits with two-phase compilation (trait defined alphabetically after impl)
- Recursive data structures (tree traversal) with pattern matching
- Nested generics (Option[List[Int]], nested records 3 levels deep)
- Pipeline patterns (chained map/filter/fold with trait methods)
- Multi-file projects up to 5 modules with cross-module trait definitions and usage
- Generic functions (identity, HOF callbacks across modules)

### Probes 2781-2830: All Passed (50 probes - complex patterns)
Covered:
- Generic function instantiated at 5+ different types
- Lambda through 3 levels of indirection (compose, applyTwice)
- Pattern matching on nested Option[Option[Int]], Result inside Option
- Chained method calls (map->filter->length, map->filter->map->sum)
- Cross-module generic instantiation at 4+ types, 3-module dependency chains
- Two-phase compilation: trait in z_ module, impl in a_ module
- Cross-module mutual recursion (isEven/isOdd), concurrent spawn + message passing
- Recursive generic data structures (trees, linked lists, stacks)
- Function factories returning generic list transformers
- sortBy with custom comparator, multiple trait impls on same type cross-module

### Probes 2831-2880: All Passed (50 probes - single-impl inference + error detection)
Covered:
- Single-impl trait lambda inference (correct unique-type resolution)
- Multi-impl trait lambda correctly errors ("cannot resolve trait method")
- Builtin shadowing prevention (show, eval, contains)
- Recursive generic Tree with pattern matching, deep tree depth computation
- Generic Pair swap, pipeline operator chains
- Type mismatch error detection, missing trait impl errors
- Cross-module variant functions, 5-module dependency chains, diamond dependency
- Cross-module single-impl trait lambda inference
- Wrong number of args error detection, undefined variable error (with suggestions)
- Mixed return type error detection
- Complex 5-module project (types, traits, filters, stats, main)

### Probes 2881-2930: All Passed (50 probes - trait dispatch + re-exports)
Covered:
- Pattern match + map with variant type, cross-module re-export chain (A->B->C)
- Generic fn calling generic fn (applyTwice), nested folds on list of lists
- Two-phase trait impls on generic types (z_types, a_traits, m_impl) - 4 module project
- Generic function composition, method chain on freshly constructed list
- show() on user-defined unit variant types, spawn with Option type
- Diamond dependency with shared type, default parameters cross-module
- Recursive variant type across modules (Tree), trait method in map over list
- 5-module projects with complex dependency graphs
- Cross-module generic Either type, recursive generic variant stack
- Trait impl on generic type in alphabetically-first file (two-phase compilation)

### Probes 2931-2980: All Passed (50 probes - adversarial inference)
Covered:
- Type inference through multiple let bindings (map/filter/length chains)
- Cross-module overloaded functions same name different types
- Nested generic type construction Some(Ok([1,2,3])) with pattern matching
- Lambda with multi-step body, generic function as argument to generic function
- Mutable variable with chained map/filter operations
- Pipeline operator with cross-module functions
- Cross-module trait impl ordering (impl in alphabetically-first module)
- 5-module diamond dependency pattern, 4-module transitive imports
- Private vs public visibility enforcement, selective imports with curly braces
- Cartesian product via nested flatMap/map, mutual recursion cross-module
- Generic combineWith with 3 type vars cross-module

### Probes 2981-3030: All Passed (50 probes - limits of type system)
Covered:
- Higher-order functions returning functions (makeAdder, compose)
- List of functions with map application
- Triple type with 3 type params (mapFirst/mapSecond/mapThird)
- Manual fold via recursive overloading, recursive expression evaluation
- Deeply nested method chains (map/filter/flatMap)
- Cross-module diamond/chain dependencies (5 modules deep)
- Cross-module generic type wrapper (wrap/unwrap/doubleWrapped)
- Phantom type patterns (Tagged[u] with unit types) - single and cross-module
- Match with guards (when clauses), try/catch across 3 modules
- Cross-module reverse alphabetical ordering stress (aaa depends on zzz)
- Trait method chaining (show on result of map)
- Mutable state with mvar

### Probes 3031-3080: All Passed (50 probes - fragile area targeting)
Covered:
- Trait UFCS with multiple string params across modules
- Trait with 4+ params, trait method taking function param
- Single-impl vs multi-impl trait methods in lambda context (boundary cases)
- Pattern matching on cross-module generic variant
- Cross-module function with Num trait bound type params
- Three-module trait + generic + pattern matching combined
- Method chains (map/filter/sortBy) on cross-module types
- Deeply nested function application with generics
- Cross-module recursive variant (MathExpr) with evaluate
- Lambda capturing cross-module function
- Complex 5-module project (types, validators, transforms, formatter)
- Cross-module trait with supertrait, trait impl on variant type
- Recursive binary tree (sum + depth), multi-module algebra/geometry projects
- Generic pair operations (swap, mapFirst, mapBoth with 4 type params)

### Probes 3081-3130: All Passed (50 probes - inference corners)
Covered:
- Method call on match result (arithmetic, string, list)
- Option pipeline (map chain, unwrapOr)
- fold with show() in accumulator, fold with separator logic
- Lambda returning tuple and triple tuple
- Empty collection operations (map/fold/filter on empty typed lists)
- Polymorphic identity at 5 types, polymorphic pair at multiple types
- Nested record field access, generic record type (Box[a])
- Cross-module polymorphic function, record type, stdlib Result
- Recursive tree type with sumTree, recursive expression evaluator (AST)
- Generic stack type (push/peek), function shadowing across modules

### Probe 3165: **BUG FOUND** - Multi-clause function dispatch ignores definition order
**Problem**: `myTake(xs, 0) = []` followed by `myTake([x|rest], n) = [x] ++ myTake(rest, n-1)`
incorrectly matches clause 2 when called with `myTake([1,2,3], 0)`, causing infinite recursion.

**Root cause**: The compiler sorted clauses by first-parameter pattern specificity
(literal=0, constructor=1, variable=2) at two locations in compile.rs (lines 4908-4944
and 34386-34413). This reordered `myTake(xs, 0)` (variable first param, specificity=2)
AFTER `myTake([x|rest], n)` (list pattern, specificity=1), breaking the standard
top-to-bottom definition order semantics.

**Fix**: Removed pattern specificity sorting. Kept only type-annotation-count sorting
(needed for typed overloads like `f(x: Int)` before `f(x)`). Uses stable sort so
clauses with equal type annotation counts preserve definition order.

**Commit**: `14be7fb`

### Probes 3131-3180: All Passed (50 probes - real-world patterns)
Covered:
- Simple and complex AST with recursive evaluation (IntLit, BoolLit, BinOp)
- State machine with transitions (Idle, Running, Done)
- Generic stack, queue implementations
- Config record with nested defaults, event system with handlers
- Graph operations (adjacency list), parser combinator patterns
- Mini type checker with recursive types
- Data pipeline (validate, transform, aggregate, format)
- Serialization of variant types, closures capturing state
- Builder pattern, curried functions, multi-module interpreter

### Probes 3181-3230: All Passed (50 probes - type inference torture)
Covered:
- Generic function as both direct call and UFCS
- Chained user-defined trait methods (transform().describe())
- Lambda returning tuple with mixed types (Int, Int, String)
- Cross-module generic functions with 3+ type params (zip3)
- Pattern match inside lambda inside fold
- Nested generics 3+ deep (Option[List[Result[Int, String]]])
- Multiple trait impls in same expression (Point, Circle, Rect)
- Generic record Box[a] with UFCS methods
- Type error detection (mismatch, undefined function, wrong arg type)
- Recursive variant types (Tree, Expr evaluator)
- Cross-module function factories, generic Either variant

### Probes 3231-3280: All Passed (50 probes - compilation edge cases)
Covered:
- Long function bodies (10+ bindings)
- Deeply nested if/else (5-7 levels, 9 branches)
- Functions with 6-8 parameters
- Multiple sequential match expressions
- Cross-module 6-module project (model/validate/format/stats/search/main)
- Recursive types (linked list, binary tree, expression tree)
- Deeply nested pattern matching (Option[Option[Option[Int]]])
- Generic types (Box[a], Pair[a,b], Either[a,b])
- Complex string formatting (show/concat/join chains)
- UFCS method chaining on custom records
- Overloaded functions with dispatch by parameter type

### Probes 3281-3380: All Passed (100 probes)
Covered:
- Multi-clause function dispatch ordering (regression verified)
- Long function bodies (10+ bindings), deeply nested if/else
- Functions with 6-8 parameters, multiple sequential match expressions
- Cross-module 6-module projects (model/validate/format/stats/search/main)
- Recursive types (linked list, binary tree, expression tree)
- Generic types (Box[a], Pair[a,b], Either[a,b])
- Complex string formatting chains
- Overloaded functions with dispatch by parameter type

### Probe ~3340: **BUG FOUND** - JIT bool detection fails for if/then/else returning Bool
**Problem**: `isPos(n: Int) = if n > 0 then true else false` returns `Int(1)` instead
of `Bool(true)`. The function works correctly with `--no-jit`, confirming it's a JIT issue.

**Root cause**: The JIT's `function_returns_bool` analysis only tracked the LAST `Return`
instruction. In functions with `if/then/else` in tail position, `compile_if` emits `Return`
for both branches (with bool registers), but `compile_fn_def` appends a dead `Return(dst)`
where `dst` was allocated but never assigned a bool value. The analysis saw only this last
dead Return and concluded the function doesn't return bool, so the JIT's int 0/1 result
wasn't converted back to `Bool(true)`/`Bool(false)`.

**Fix**: Changed `function_returns_bool` to check ALL Return instructions instead of just
the last one. If any Return instruction returns from a bool-producing register (LoadTrue,
LoadFalse, comparison, etc.), the function is marked as bool-returning. Dead-code Returns
with unassigned registers are safely ignored since reachable Returns will match.

**Commit**: `1c1754a`

### Probes 3381-3430: All Passed (see sessions 40-40b)

### Probes 3431-3511: All Passed (see session 40b)

### Probe 3536: **BUG FOUND** - Mutable variable aliasing: let-bindings share register
**Problem**: `var b = 1; temp = b; b = 2; (temp, b)` returns `(2, 2)` instead of `(1, 2)`.
A let-binding (`temp = b`) that references a mutable variable (`var b`) shares the same
register. When `b` is later mutated, `temp` also changes because they alias the same register.

**Root cause**: In `compile_binding`, the RHS expression `b` compiles to `info.reg` (the
register number of the mutable variable). The new binding `temp` is then mapped to the
same register. When `b = 2` emits `Move(reg, new_value)`, both `b` and `temp` see the
new value because they share the register.

This breaks all temp-variable patterns (swap, fibonacci, accumulator snapshots, etc.):
- `var a = 0; var b = 1; temp = b; b = a + b; a = temp` (fibonacci) produced powers of 2
  instead of fibonacci numbers because `temp` aliased `b`
- `temp = a; a = b; b = temp` (swap) didn't work correctly

**Fix**: After compiling the RHS in `compile_binding`, check if `value_reg` belongs to
any mutable local. If so, allocate a fresh register and emit a `Move` instruction to
copy the value. This ensures the new binding captures the value at binding time rather
than referencing the mutable cell.

**Commit**: `016bc3f`

### Probes 3512-3597: All Passed (~86 probes after fix, including 20 multi-file projects)
Covered:
- Mutable variable value capture (snapshot, swap, fibonacci, chain of snapshots)
- Nested mutable variable scopes (inner/outer for loops)
- Mutable accumulator patterns (for loop, string building, list reversal)
- Iterative factorial, nested var mutation
- Cross-module generic types (Pair, Shape, Stack, Counter, Tree)
- Cross-module trait definitions and implementations (Describable, HasArea)
- Three-module chains (types -> logic -> main)
- Two-phase compilation stress (zzz_types/aaa_user alphabetical ordering)
- Cross-module variant types (Shape, Maybe, Result, BTree)
- Cross-module HOF (applyAll, function composition, validators)
- Cross-module record types with builder patterns (Config, GameState, User)
- Cross-module recursive types (BTree insert/inorder/size)
- Cross-module mutable var in for loops (GameState.simulate)
- Map operations, Option chaining, try/catch
- flatMap, unique, sort, zip, fold patterns
- String interpolation, pattern matching with guards
- Deeply nested if-else, multi-clause functions
- Complex fold patterns (scanl, running totals, type-changing folds)

## Session 42 (2026-02-20)

### Probes 3598-3617: Single-file probes (20 probes, all passed or test writing errors)
Covered:
- Generic swap function, fibonacci with var mutation
- filterMap, record update syntax (not supported - test errors)
- Ok/Err construction, sortBy, list comprehension patterns
- String.join, take/drop argument ordering

### Probes proj116-proj125: Multi-file probes (10 projects)
- proj116: Generic Stack container (passed after naming fix)
- proj117: **BUG FOUND** - cross-module generic type annotation collision
- proj118-proj125: Various multi-file patterns (most passed after syntax fixes)

### Probe proj117: **BUG FOUND** - AST-based signature reuses annotation type param letters
**Problem**: `mapPair(p: Pair[a, b], f) = match p { Pair(x, y) -> Pair(f(x), f(y)) }`
in module `zzz_types`, called from `main` as `mapPair(makePair(1, 2), x => x * 2)` -
failed with "type mismatch: expected Int, found (Int) -> Int" on `makePair(1, 2)`.

**Root cause**: When `try_hm_inference` fails for `mapPair` (returns None), the
fallback `def.signature()` in ast.rs generates type variable letters starting at 'a'.
Since the annotation `p: Pair[a, b]` already uses 'a' and 'b', the untyped param `f`
gets letter 'a' - colliding with Pair's first type parameter. The resulting signature
`Pair[a, b] -> a -> b` makes `parse_signature_string` map both the Pair's first arg
and the `f` param to Var(1). When main calls `mapPair(Pair[Int,Int], lambda)`, the
lambda's type (Int)->Int conflicts with Int for the shared Var(1).

**Fix**: Collect all lowercase single-letter type params used in annotations and skip
them when assigning type variables to untyped parameters and return types. Now `f`
gets letter 'c' instead of 'a', giving the correct `Pair[a, b] -> c -> d` signature.

**Commit**: `f9e6346` - Fix cross-module type variable collision in AST-based signature generation

## Session 43 (2026-02-20)

### Probes 3629-3713: All Passed (~85 probes - intensive multi-file + single-file)
Covered:
- Multi-file: generic types (Box[a], Pair[a,b], Triple[a,b,c], Quad[a,b,c,d]) across modules
- Multi-file: HOF with generic type params (mapBox, filterGroup, applyAll)
- Multi-file: multiple modules importing same generic type (ops + main)
- Multi-file: 3-module chains (types -> ops -> compute -> main)
- Multi-file: function returning function cross-module (pairMaker)
- Multi-file: nested generic types (Labeled[a] with Box, Option, List)
- Multi-file: recursive generic types (Tree[a] with treeMap, treeSum)
- Multi-file: variant types without type params (Shape, Color, Expr AST)
- Multi-file: two unrelated generic types from same module
- Multi-file: generic type with list field + standalone usage (Accumulator[a])
- Multi-file: concrete type specialization (Container[Int], Container[String])
- Two-phase compilation: zzz_ types / aaa_ ops (reverse alphabetical ordering)
- Two-phase compilation: trait in zzz_ module, impl in aaa_ module
- Two-phase compilation: type AND trait in zzz_ module, impl in aaa_ module
- Two-phase compilation: generic type with 4 params, partial annotation cross-module
- Cross-module traits: pub trait + impl in separate modules (Describable, Scorer, Measurable)
- Cross-module traits: Scalable with Self return type on record types
- Cross-module traits: multiple impls (Int + String) in same module
- Cross-module: record types with generic type fields (Container104)
- Cross-module: function taking multiple different generic types as arguments
- Cross-module: generic function returning list of generic type (mapSlots)
- Cross-module: 4-module projects (types, ops, mul/add, main)
- Cross-module: zipWith combining two lists into list of generic pairs
- Single-file: nested map chains, lambda capture type propagation
- Single-file: function composition (compose, apply)
- Single-file: tuple destructuring from function returns
- Single-file: polymorphic identity function at 3 types
- Single-file: fold with different accumulator types (String, List)
- Single-file: method call on generic match result
- Single-file: chained filter/map/take operations
- Single-file: mutual recursion (isEven/isOdd)
- Single-file: option pattern matching (safeDivide with filter/map)
- Single-file: n-ary tree type with recursive sumTree
- Single-file: record operations (Vec2, Vec3)
- Single-file: function values in list applied sequentially via fold
- Single-file: try/catch with custom variant Result type

**No compiler bugs found in 85 probes.** The multi-file compilation, two-phase
compilation ordering, cross-module generic types, and trait system are all very solid.

## Session 44 (2026-02-20)

### Probes 3714-3793: All Passed (~80 probes - multi-file + single-file adversarial)
Covered (70 multi-file projects + 10 single-file probes):
- **Recursive generic types cross-module**: Tree44[a] with insert, depth, mapTree, sumLeaves
- **Nested generic types**: Maybe44[Pair44[a,b]] with findPair, mapMaybe, withDefault
- **HOF cross-module**: transformBoxes, chainTransform on Box44[a]
- **Multiple trait impls cross-module**: Showable44 + Combinable44 on Wrapper44[a]
- **Recursive variant pattern matching**: Expr44 (Num, Add, Mul, Neg) with eval + simplify
- **Triple type with 3 type params**: mapFirst, mapAll, tripleToList cross-module
- **Pipeline chains with generic Result type**: mapResult, flatMapResult, processInput
- **Mutual recursion**: isEven/isOdd via Peano naturals cross-module
- **4-module projects**: Shape44 with separate area and perimeter trait impl modules
- **Complex 5-module projects**: Animal44 with traits, impls, utils (fold, bestAnimal)
- **Generic record types**: Point44/Rect44 with nested field access chains cross-module
- **Record types with default parameters**: Config44 with makeConfig(host:, port:, verbose:)
- **Exception handling cross-module**: throw/catch with custom AppError44 variant
- **Deeply nested JSON-like recursive type**: JNull, JNum, JStr, JArr, JObj with count
- **Function-in-type pattern**: Fn44[a,b] with apply, compose, pipeline
- **Trait with multiple methods**: Geometry44 (perimeter, area, describe) cross-module
- **Generic Either type**: Left44/Right44 with isLeft, mapRight, partitionEithers
- **Nested record field access**: Wrapper44.outer.inner.value (3 levels deep)
- **Closure-returning functions**: boxMapper, boxExtractor, boxWrapper
- **BST cross-module**: BTree44[a] with insert, contains, inorder, treeSize, treeDepth
- **Lazy evaluation pattern**: Lazy44[a] with force, defer, mapLazy, zipLazy, sequence
- **Nested generic types**: Collection44[Id44[a]] with addItem, findById, mapColl
- **Polymorphic recompilation**: Box44 used at Int and String in different modules
- **Tuple operations**: sortBy with complex comparators, fold into tuples
- **State machine pattern**: State44 (Idle, Running, Done, Error) transitions cross-module
- **Predicate combinators**: andPred, orPred, notPred with filterWith cross-module
- **6-module expression evaluator**: Token44, Expr44 across lexer, parser, eval modules
- **Cross-module Map operations**: entriesToMap, mapValues using %{} empty map
- **Single-file**: nested lambdas, currying, chained methods, recursive list type, sortBy, Map ops

**No compiler bugs found in ~80 probes.** Combined with session 43 (~85 probes),
this is ~165 consecutive probes across two sessions with zero compiler bugs.
The multi-file compilation system, two-phase ordering, cross-module generic types,
trait implementations, UFCS dispatch, and HM inference are all extremely solid.

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
| 13      | 32 (488-519)      | UFCS key includes type args | Yes (7087c27) | 520 |
| 13b     | 66 (521-586)      | (none - clean run) | N/A | 586 |
| 14      | 3 (587-589)       | Cross-module trait on builtin types | Yes (55919ec) | 590 |
| 14b     | 190 (591-780)     | (none - clean run) | N/A | 780 |
| 15      | 1 (781)           | UFCS key collision for multiple concrete generic impls | Yes (293fc89) | 782 |
| 16      | 6 (783-788)       | Trait impl on List[a] wrong fn key + cross-module Either2 | Yes (ff46dc8) | 789 |
| 16b     | 40 (791-830)      | (none - clean run) | N/A | 830 |
| 16c     | 50 (831-880)      | (none - clean run) | N/A | 880 |
| 17      | 50 (881-930)      | (none - clean run) | N/A | 930 |
| 17b     | 50+80 (931-980)   | (none - clean run) | N/A | 980 |
| 18      | 50 (981-1030)     | (none - clean run) | N/A | 1030 |
| 18b     | 50 (1031-1080)    | (none - clean run) | N/A | 1080 |
| 19      | 50 (1081-1130)    | (none - clean run) | N/A | 1130 |
| 19b     | 50 (1131-1180)    | (none - clean run) | N/A | 1180 |
| 20      | 50 (1181-1230)    | (none - targeted) | N/A | 1230 |
| 20b     | 50 (1231-1280)    | (none - targeted) | N/A | 1280 |
| 21      | 44 (1281-1324)    | `use NonExistent.*` silently succeeds | Yes (2b3d467) | 1330 |
| 22      | 50 (1331-1380)    | (none - clean run) | N/A | 1380 |
| 22b     | 32 (1381-1412)    | Option/Result type_name mismatch + record pattern match | Yes (f7fcf9c, d327f94) | 1430 |
| 23      | 50 (1431-1480)    | (none - clean run) | N/A | 1480 |
| 23b     | 50 (1481-1530)    | (none - clean run) | N/A | 1530 |
| 24      | 50 (1531-1580)    | (none - integration) | N/A | 1580 |
| 24b     | 50 (1581-1630)    | (none - regression+boundary) | N/A | 1630 |
| 25      | 50 (1631-1680)    | (none - creative patterns) | N/A | 1680 |
| 25b     | 50 (1681-1730)    | (none - multi-file two-phase) | N/A | 1730 |
| 26      | 50 (1731-1780)    | (none - type inference edge cases) | N/A | 1780 |
| 26b     | 50 (1781-1830)    | (none - advanced multi-file) | N/A | 1830 |
| 27      | 50 (1831-1880)    | (none - tricky HM inference) | N/A | 1880 |
| 27b     | 50 (1881-1930)    | (none - combinatorial multi-file) | N/A | 1930 |
| 28      | 1 (1931)          | Generic fn with trait bound + lambda param types | Yes | 1936 |
| 28b     | 44 (1937-1980)    | (none - after fix) | N/A | 1980 |
| 28c     | 50 (1981-2030)    | (none - adversarial) | N/A | 2030 |
| 29      | 43 (2031-2073)    | Pattern-matched generic stdlib type loses type info | Yes (696bb26) | 2074 |
| 29b     | 6 (2075-2080)     | (none - after fix) | N/A | 2080 |
| 29c     | 50 (2081-2130)    | (none - multi-file stress) | N/A | 2130 |
| 30      | 10 (2131-2140)    | Trait method default params fail on UFCS calls | Yes (dd92d6c) | 2141 |
| 30b     | 39 (2142-2180)    | (none - after fix) | N/A | 2180 |
| 30c     | 50 (2181-2230)    | (none - adversarial corners) | N/A | 2230 |
| 31      | 8 (2231-2238)     | Named args in UFCS trait method calls | Yes (8e7fbbb) | 2239 |
| 31b     | 41 (2240-2280)    | (none - after fix) | N/A | 2280 |
| 31c     | 50 (2281-2330)    | (none - dispatch+instantiation) | N/A | 2330 |
| 32      | 50 (2331-2380)    | (none - named args, traits, generics) | N/A | 2380 |
| 32b     | 50 (2381-2430)    | (none - multi-module, real-world) | N/A | 2430 |
| 33      | 50 (2431-2480)    | (none - deep adversarial) | N/A | 2480 |
| 33b     | 50 (2481-2530)    | (none - two-phase compilation stress) | N/A | 2530 |
| 34      | 50 (2531-2580)    | (none - deep generic patterns) | N/A | 2580 |
| 34b     | 27 (2581-2607)    | Lambda returning trait method call on untyped param | Yes (e0166e7) | 2608 |
| 34c     | 22 (2609-2630)    | (none - after fix) | N/A | 2630 |
| 35      | 50 (2631-2680)    | (none - clean run) | N/A | 2680 |
| 35b     | 0 (2681)          | Widely-implemented trait method (.show()) inferred to wrong type | Yes (423a2a5) | 2681 |
| 35c     | ~25 (2682-2707)   | Named args in trait UFCS not resolved during HM inference | Yes (f94f6b0) | ~2710 |
| 35d     | 50 (2731-2780)    | (none - clean run) | N/A | 2780 |
| 36      | 50 (2781-2830)    | (none - complex patterns) | N/A | 2830 |
| 36b     | 50 (2831-2880)    | (none - single-impl inference + errors) | N/A | 2880 |
| 37      | 50 (2881-2930)    | (none - trait dispatch + re-exports) | N/A | 2930 |
| 37b     | 50 (2931-2980)    | (none - adversarial inference) | N/A | 2980 |
| 37c     | 50 (2981-3030)    | (none - type system limits) | N/A | 3030 |
| 37d     | 50 (3031-3080)    | (none - fragile area targeting) | N/A | 3080 |
| 38      | 50 (3081-3130)    | (none - inference corners) | N/A | 3130 |
| 38b     | 34 (3131-3164)    | Multi-clause dispatch ignores definition order | Yes (14be7fb) | 3165 |
| 38c     | 15 (3166-3180)    | (none - after fix) | N/A | 3180 |
| 38d     | 50 (3181-3230)    | (none - type inference torture) | N/A | 3230 |
| 38e     | 50 (3231-3280)    | (none - compilation edge cases) | N/A | 3280 |
| 39      | 100 (3281-3380)   | (none - clean run) | N/A | 3380 |
| 39b     | ~60 (3340)        | JIT bool detection for if/then/else returning Bool | Yes (1c1754a) | ~3340 |
| 40      | ~48 (3381-3428)   | Function-typed field call collides with builtin module method (e.g., `cb.run(x)` on record → Exec.run) | Yes (3838ac8) | ~3430 |
| 40b     | ~80 (3431-3511+30 multi-file) | Function-typed field `send` collides with UFCS method `WebSocket.send` (same class as 40, but true UFCS not module fn) | Yes (0caf5ea) | ~3511 |
| 40c     | ~25 (3512-3536)   | Mutable variable aliasing: let-bindings share register with var | Yes (016bc3f) | ~3536 |
| 40d     | ~61 (3537-3597+20 multi-file) | (none - clean run after var fix) | N/A | ~3597 |
| 41      | ~20 single + 10 multi (3598-3628) | Cross-module generic type annotation collision in AST-based signature | Yes (f9e6346) | ~3628 |
| 43      | ~85 (3629-3713) | (none - clean run) | N/A | ~3713 |
| 44      | ~80 (3714-3793) | (none - clean run, multi-file + single-file) | N/A | ~3793 |
| 44b     | ~10 (3794-3803) | UFCS dispatch: `words.map(w => w.reverse())` resolves to List.reverse instead of String.reverse inside lambdas | Yes (9e59563) | ~3803 |
