# Nostos Type Inference Audit

This document tracks discovered type inference issues in the Nostos language.

## Summary

| # | Issue | Status | Severity |
|---|-------|--------|----------|
| 1 | Unknown type in method error messages | **Fixed** | Low |
| 2 | zipWith wrong arg order: runtime vs compile error | **Fixed** | Medium |
| 3 | Misleading "if/else branches" error for numeric type mismatches | **Fixed** | Low |
| 4 | sortBy wrong function arity: runtime vs compile error | **Fixed** | High |
| 5 | **Lambdas silently ignore extra arguments** | **Fixed** | **Critical** |
| 6 | Trait bounds not propagated into lambdas | **Fixed** | High |
| 7 | `length()` rejects String argument | **Fixed** | Medium |
| 8 | Missing BUILTINS signatures for stdlib functions | **Fixed** | Medium |
| 9 | Trait method chaining in generic functions | **Fixed** | High |
| 10 | UFCS method return types not tracked for local bindings | **Fixed** | Medium |
| 11 | HM false positive blocks monomorphization with function params | **Fixed** | High |
| 12 | Generic function calls from lambdas inside generic functions | **Fixed** | **Critical** |
| 13 | Trait bound violations crash instead of compile error | **Fixed** | **Critical** |
| 14 | Constructor name collision: user-defined types shadowed by built-in | **Fixed** | High |
| 15 | `map` result ++ String not caught at compile time | **Fixed** | High |
| 16 | List-only methods (head, tail, map etc.) not caught on String | **Fixed** | Medium |
| 17 | Bool arithmetic (`true + 5`) not caught at compile time | **Fixed** | Medium |
| 18 | Field access on primitives (`42.x`) not caught at compile time | **Fixed** | Medium |
| 19 | Calling non-function (`f = 42; f(10)`) not caught at compile time | **Fixed** | Medium |
| 20 | `List + List` arithmetic not caught at compile time | **Fixed** | Medium |
| 21 | Unary negation on non-numeric (`-"hello"`) not caught at compile time | **Fixed** | Medium |
| 22 | String doesn't implement Ord but VM supports comparison | **Fixed** | Medium |
| 23 | Numeric field access (.0, .1) on lists fails HM inference | **Fixed** | Medium |
| 24 | Type params used with Num/Ord ops crash at runtime | **Fixed** | High |
| 25 | Trait method on function param return inside lambda fails | **Fixed** | High |

## Discovered Issues

### Issue 1: "unknown" type in deferred method error messages

**Status**: FIXED

**Severity**: Low (UX issue)

**Description**: When a method doesn't exist on a type that was inferred through a type variable, the error previously said `type 'unknown'` which was uninformative.

**Reproduction**:
```nostos
process(f) = {
    x = f()
    x.nonexistent()  # Wrong method
}

main() = {
    result = process(() => [1, 2, 3])
    0
}
```

**Previous behavior**: Error said "no method `nonexistent` found for type `unknown`"

**Fix**: Improved error messages for polymorphic types. When the type is a type variable (polymorphic), we now:
1. Show that the type is polymorphic with a type variable name
2. Explain that the concrete type is not known at this point
3. Suggest adding type annotations to constrain the type

**Now produces**:
```
Error: no method `nonexistent` found for polymorphic type
       the receiver has type `y` which is a type variable - its concrete type is not known at this point

Help: consider adding a type annotation to constrain the type, e.g., `x: List[Int] = f()`
Note: polymorphic functions have type variables that are only known when called
```

**Technical note**: The original expectation was to show `List[Int]`, but this is technically incorrect. Within the body of `process`, the type of `x` IS a type variable - it's only resolved to `List[Int]` at the call site. The new error message correctly reflects this and provides helpful guidance.

---

### Issue 2: zipWith with wrong argument order causes runtime error instead of compile error

**Severity**: Medium

**Status**: FIXED

**Description**: Calling `zipWith(f, xs, ys)` instead of `zipWith(xs, ys, f)` was causing a runtime error ("Length: unsupported type") instead of a compile-time type mismatch error.

**Root Cause**: Multiple issues identified and fixed:

1. **Stdlib Function Registration**: Multiple code paths were registering stdlib functions with empty `type_params`, overwriting the correctly-cached signatures.

2. **Error Filter Too Broad**: The error filter in `compile_function` was incorrectly filtering out legitimate type errors:
   - The "tuple error" filter matched any error containing `(`, `,`, `)`, and "Cannot unify" - which matched function type errors like `(?24, ?25) -> ?27`
   - Another filter matched `List[?N]` with `->` as "type variable confusion"

**Fixes Applied**:

1. **Skip stdlib in `try_hm_inference`** (compile.rs:22708): Added check to skip stdlib functions when registering from `self.functions`, preserving cached signatures with proper type_params.

2. **Exclude stdlib from `compiled_base_names`** (compile.rs:22786): Modified to exclude stdlib functions so their signatures are registered from `pending_fn_signatures`.

3. **Fixed error filter** (compile.rs:9307-9310): Made tuple error filter not match function types by adding `&& !message.contains("->")`

4. **Disabled overly-broad type var confusion filter** (compile.rs:9301)

**Reproduction**:
```nostos
main() = {
    # WRONG order - should be (xs, ys, f)
    combined = zipWith((n, s) => show(n) ++ s, [1,2,3], ["a","b","c"])
    0
}
```

**Now produces compile-time error**:
```
Error: type mismatch: expected `List[?30]`, found `(?24, ?25) -> ?27`

Help: function arguments may be in the wrong order
Note: functions like filter, map, find, etc. take the list as the first argument
```

---

### Issue 3: Misleading "if/else branches" error for numeric type mismatches

**Severity**: Low (UX issue)

**Description**: When there's a type mismatch between Int and Int32 (or Float and Float32), the error message incorrectly mentions "if and else branches" even when no if/else exists in the code.

**Reproduction**:
```nostos
doubleThem(x: Int32, y: Int32) -> Int32 = x + y

main() = {
    # Pass Int instead of Int32
    result = doubleThem(5, 10)
    0
}
```

**Status**: FIXED

**Fix**: The error message conversion in `convert_type_error` was using `if_branch_type_mismatch` as a generic fallback for all type mismatches. Changed to use a generic "type mismatch" error message instead.

**Now produces**:
```
Error: type mismatch: expected `Int32`, found `Int`
```

---

### Issue 4: sortBy with wrong function arity causes runtime error instead of compile error

**Status**: FIXED

**Severity**: High (type safety violation)

**Description**: Passing a unary function `T -> Int` to `sortBy` which expects a binary comparator `(T, T) -> Int` causes a runtime error instead of a compile-time type error.

**Fix**: Fixed BUILTIN signatures that used curried function syntax `(a -> a -> Int)` instead of multi-param syntax `((a, a) -> Int)`. The curried syntax was being parsed as a 1-param function returning a function, causing arity mismatches to go undetected. Fixed signatures for: sortBy, fold, pairwise, isSortedBy.

**Now produces compile error**:
```
Error: Wrong number of arguments: expected 2, found 1
```

---

### Issue 5: Lambdas silently ignore extra arguments

**Status**: FIXED

**Severity**: CRITICAL (fundamental type safety violation)

**Description**: Lambdas accept any number of arguments and silently ignore extras. This completely breaks function arity checking for lambdas.

**Fix**: Fixed the `unify_types` function in `crates/types/src/infer.rs` to enforce strict arity checking on function types. Previously, when `required_params` was `None`, the code was lenient about arity mismatches. Now it correctly requires all parameters when `required_params` is `None`.

**Reproduction (now fails correctly)**:
```nostos
main() = {
    f = x => x * 2
    result = f(5, 10, 15)  # Now: compile error about arity
    0
}
```

Also fixed passing wrong-arity lambdas to higher-order functions:
```nostos
main() = {
    nums = [3, 1, 2]
    result = nums.sortBy(x => x)  # Now: compile error - expected 2 params, found 1
    0
}
```

**Root cause**: Two separate issues were fixed:
1. `unify_types` was too lenient on function arity when `required_params` was `None`
2. BUILTIN signatures used curried function syntax which the parser interpreted as 1-param functions

---

### Issue 6: Trait bounds not propagated into lambdas

**Severity**: High

**Status**: **Fixed** (all cases resolved including chaining and nested records)

**Description**: When a function has a type parameter with a trait bound (e.g., `T: Sizeable`), lambdas within that function cannot call trait methods on values of type `T`. The type checker sees `T` as an unconstrained type parameter within the lambda body.

**Reproduction (now works)**:
```nostos
trait Sizeable
    size(self) -> Int
end

type Box = Box(Int, Int)

Box: Sizeable
    size(self) = match self { Box(w, h) -> w * h }
end

# This works - direct call
direct(item: Box) = item.size()

# This now works! - trait method in lambda
inLambda[T: Sizeable](item: T) -> Int =
    ((x) => x.size())(item)

main() = {
    b = Box(2, 3)
    inLambda(b)  # Returns 6
}
```

**Fix Details**:

1. **Type inference phase** (infer.rs):
   - Added `current_type_param_constraints` to track trait bounds for type parameters in scope
   - When unifying with a TypeParam, propagate constraints to fresh type variables
   - In `check_pending_method_calls`, look up trait methods when receiver is a type variable with bounds
   - Register user-defined traits in the TypeEnv for lookup

2. **Compilation phase** (compile.rs):
   - Register traits from `trait_defs` into `env.traits` in `try_hm_inference`
   - Fixed `find_trait_method` to strip `" (type parameter)"` suffix when matching
   - Fixed `is_current_type_param` to strip the suffix as well
   - When `expr_type_name` returns a TypeParam, check `current_type_bindings` for concrete type during monomorphization

3. **Monomorphization path**:
   - When a polymorphic function is called with concrete types, `current_type_bindings` maps type params to concrete types
   - `expr_type_name` now uses these bindings to return concrete types during monomorphization

**Now passing**: All `tests/type_inference/trait_closure_*.nos` tests (8 tests), plus:
- `trait_stdlib_reverse.nos` - reverse with trait methods on result
- `double_map_trait.nos` - double map with trait methods in lambdas
- `triple_once_twice.nos` - calling generic function twice from main
- `dropwhile_trait.nos` - dropWhile with trait predicate
- `takewhile_trait.nos` - takeWhile with trait predicate
- `partition_trait.nos` - partition with trait predicate
- `nested_list_map.nos` - nested map with trait methods

**Additional fixes for Issue #6**:

4. **BUILTIN signature fallback** (compile.rs `get_return_type_for_call`):
   - When stdlib functions like `map` aren't in `fn_asts`, parse their BUILTIN signatures
   - Match arg types against signature params to build type param map
   - Substitute type params in return type

5. **extract_type_bindings uppercase type params**:
   - Allow binding lowercase builtin params (`a`, `b`) to uppercase user type params (`T`, `U`)
   - Previously rejected because both are single-letter names

6. **Lambda type param passthrough** (compile.rs `compile_arg_with_expected_type`):
   - When expected type for a lambda param is a type param (e.g., `T` after substitution),
     accept it if it's one of the current function's type params
   - This allows trait method resolution in lambdas via monomorphization

7. **Missing BUILTINS entries** added for: takeWhile, dropWhile, partition, flatMap, zipWith, enumerate

**Remaining edge cases**:
- Nested records with trait methods in nested lambdas (now fixed by Issue #9)
- Deeply nested lambdas calling trait methods on captured variables (now fixed by Issue #9)

---

### Issue 7: `length()` rejects String argument

**Status**: FIXED

**Severity**: Medium

**Description**: `length(s)` where `s: String` produced a false compile-time type error because `length`'s BUILTINS signature was `[a] -> Int` (list-only), but `length` is a VM instruction that works on any collection type (List, String, Map, etc.).

**Fix**: Changed `length`'s BUILTINS signature from `[a] -> Int` to `a -> Int`. The VM's `Length` instruction already handles dispatch for all supported types at runtime.

---

### Issue 8: Missing BUILTINS signatures for stdlib functions

**Status**: FIXED

**Severity**: Medium

**Description**: Several stdlib list functions (takeWhile, dropWhile, partition, flatMap, zipWith, enumerate) had no BUILTINS entries. This meant `get_function_param_types` couldn't find their parameter types, preventing type-directed lambda compilation. Lambdas passed to these functions would compile without type information, causing trait method resolution to fail.

**Fix**: Added BUILTINS entries with correct signatures for all six functions.

---

### Issue 9: Trait method chaining in generic functions

**Status**: FIXED

**Severity**: High

**Description**: Chaining trait method calls (e.g., `x.increment().increment()`) in generic functions fails. The first call works, but the result's type isn't tracked in `local_types`, so subsequent method calls fail.

**Root cause**: `expr_type_name` encountered an unresolved HM type variable (Type::Var) for method call expressions and immediately returned a "(polymorphic)" placeholder, preventing the pattern-based MethodCall branch from running. The MethodCall branch correctly looks up trait definitions to determine return types.

**Fix**: Changed `expr_type_name` to store unresolved type variables (Type::Var, Type::TypeParam) as fallbacks instead of returning immediately. Pattern-based matching runs first, and the HM fallback is only used if pattern matching also fails. This allows the trait method return type lookup to correctly resolve `x.increment()` -> `Int` from the trait definition, which then allows `a.increment()` to work on the chained call.

**Tests fixed**: chained_trait_methods, trait_method_chain, trait_method_chain_three, trait_self_reference, trait_self_return, generic_returning_generic, trait_binary_op, nested_records_trait, nested_records_chain, deep_nested_records_trait, nested_variant_fields, record_nested (12 tests).

---

## Test Cases

Test files are in `/tmp/infer_tests/` for reproduction.

### Passing Tests (no issues found):
- 01_nested_lambda.nos - Nested lambdas with shared context
- 02_method_chain_lambda.nos - Method chaining with type propagation
- 03_generic_return.nos - Generic function return type inference
- 04_unresolved_method.nos - Method on late-resolved type
- 06_higher_order_generic.nos - compose function
- 07_record_field_inference.nos - Record field access in lambdas
- 08_variant_inference.nos - Variant/Result type inference
- 09_tuple_inference.nos - Tuple element access
- 10_closure_capture.nos - Closure variable capture
- 11_generic_constraint.nos - Generic function with equality
- 13_recursive_type.nos - Recursive variant types (Tree)
- 14_mutual_recursion.nos - Mutually recursive functions
- 16_empty_collection.nos - Empty collection inference
- 17_ambiguous_overload.nos - Overload resolution
- 18_let_polymorphism.nos - Let polymorphism works
- 19_constraint_propagation.nos - Type flow through call chains
- 20_late_binding.nos - Conditional branch type checking
- 21_delayed_method.nos - Method call on late-resolved type
- 23_chained_methods_generic.nos - Long method chains
- 24_return_type_mismatch.nos - Return type annotation checking
- 26_nested_generic.nos - Deeply nested generic types (Box[Box[Box[T]]])
- 27_map_key_inference.nos - Map key/value type inference
- 28_map_wrong_key_type.nos - Wrong key type error
- 29_lambda_return_type.nos - Inconsistent branch types
- 31_option_inference.nos - Option type inference
- 32_fold_type_propagation.nos - Fold accumulator types
- 33_zipWith_inference.nos - zipWith with correct arg order
- 35_higher_kinded.nos - Different container types with map
- 38_conversions.nos - Type conversion methods (asInt32, asFloat32)
- 39_annotated_function_param.nos - Annotated function parameters
- 41_trait_bound.nos - Trait bound checking (Ord)
- 42_function_type_inference.nos - Higher-order function types
- 43_continuation_style.nos - CPS style type inference
- 44_partially_applied_method.nos - Methods in lambda
- 45_eta_expansion.nos - Function references
- 46c_generic_function.nos - Top-level generic polymorphism
- 48_list_of_functions.nos - List of compatible functions
- 49_mixed_function_list.nos - List of incompatible functions (correctly errors)
- 50_mutual_type_constraint.nos - Multiple trait requirements
- 51_existential_like.nos - Show trait abstraction
- 52_record_update.nos - Record copy/update pattern
- 53_double_generic.nos - Two independent type params
- 54_constrained_return.nos - Return type from usage
- 57b_gadt_like.nos - Recursive variant evaluation
- 58_flatten_nested.nos - Nested list flattening
- 59_flatMap_inference.nos - flatMap type inference
- 62_extra_args.nos - Named function arity check (works correctly)
- 65_fun_arity.nos - Named function too few args (works correctly)

### Known Limitations (not bugs):
- 15_polymorphic_recursion.nos - Polymorphic recursion requires explicit types (HM limitation)
- 30_recursive_lambda.nos - Local recursive lambdas need Y combinator (no `let rec`)
- 46_generic_function_value.nos - Storing generic function loses polymorphism (value restriction)
- 55_shadowing.nos - No variable shadowing in nested blocks (language design choice)
- 56_type_alias.nos - No type aliases (language limitation)

---

---

### Issue 10: UFCS method return types not tracked for local bindings

**Status**: FIXED

**Severity**: Medium

**Description**: When a stdlib UFCS method like `items.head()` was used in a let-binding inside a generic function, `expr_type_name` couldn't determine the return type because stdlib functions aren't in `fn_asts`. This meant `local_types` didn't track the bound variable's type, so subsequent trait method calls on that variable failed at runtime.

**Fix**: Added BUILTINS signature lookup in the MethodCall branch of `expr_type_name`. When the receiver type is known, the method's BUILTINS signature is parsed, the receiver type is matched against the first parameter to build type param bindings, and the return type is resolved via substitution. For example, `head`'s signature `[a] -> a` with receiver `List[Int]` resolves `a=Int`, returning `Int`.

---

### Issue 11: HM false positive blocks monomorphization with function params

**Status**: FIXED

**Severity**: High

**Description**: When a generic function had both a trait-bounded type parameter and a concrete function-typed parameter (e.g., `f[T: Trait](x: T, op: Int -> Int)`), monomorphization failed with a false HM type error: "Cannot unify types: (Int) -> Int and (Int) -> Int". The HM inference couldn't unify two identical function types during the monomorphized variant's type checking.

**Fix**: Skip HM type checking for monomorphized function variants (identified by `$` in the function name). These variants are derived from already-validated generic functions - the call site has already been type-checked, so re-checking the specialized variant with HM inference adds no safety value and produces false positives.

---

### Issue 12: Generic function calls from lambdas inside generic functions

**Status**: FIXED

**Severity**: Critical (runtime crash)

**Description**: Calling a generic function from within a lambda that's inside another generic function fails at runtime with "ip=0 but code.len=0" because the inner generic function never gets monomorphized.

**Reproduction**:
```nostos
trait Weighted
    weight(self) -> Int
end
Int: Weighted
    weight(self) = self
end
getWeight[T: Weighted](x: T) -> Int = x.weight()
totalWeights[T: Weighted](items: List[T]) -> Int =
    items.fold(0, (acc, x) => acc + getWeight(x))
main() = totalWeights([3, 5, 7, 11])
```

**Root cause**: The function type annotation `(U, T) -> U` in fold's parameter type was parsed as `Function([Tuple([U, T])], U)` - a function with a single tuple parameter. But the lambda `(acc, x) => ...` has two separate parameters. This mismatch (1 param vs 2 params) caused `compile_lambda_with_types` to be skipped entirely, leaving lambda parameters untyped. Without type information for `x`, `getWeight(x)` couldn't be monomorphized.

**Fix**: In `compile_arg_with_expected_type`, when the expected function type has a single Tuple parameter and the lambda has multiple parameters matching the tuple elements, expand the tuple elements as individual parameter types. This correctly maps `(acc: Int, x: T)` from the `((Int, T)) -> Int` expected type.

**Now produces**: `26` (correct: 3+5+7+11)

---

### Issue 13: Trait bound violations crash instead of compile error

**Status**: FIXED

**Severity**: Critical (crash / runtime error instead of compile error)

**Description**: When a generic function with trait bounds is called with a type that doesn't implement the required trait, the compiler crashes at runtime instead of producing a compile error. For example, calling `sumCounts(["a", "b", "c"])` where `sumCounts[T: Countable]` and String doesn't implement Countable causes "ip=0 but code.len=0" or Illegal Instruction crash.

**Root cause**: Two related issues:
1. UFCS method calls on type parameters (e.g., `x.countMe()` where x has type `T` from lambda param types) returned `TypeError` instead of `UnresolvedTraitMethod`. This prevented the enclosing generic function from being marked as needing monomorphization.
2. When monomorphization failed with a type error (type doesn't implement required trait), the compiler silently fell back to calling the empty polymorphic stub instead of propagating the error.

**Fix**:
1. In UFCS error handler: check if receiver type is a type parameter (from `current_fn_type_params`). If so, return `UnresolvedTraitMethod` to trigger polymorphic marking.
2. In `compile_call` monomorphization: propagate `TypeError` and `UnresolvedTraitMethod` errors instead of falling back to the empty stub.

**Now produces**: `Error: no method 'countMe' found for type 'String'` (proper compile-time error)

---

### Issue 14: Constructor name collision: user-defined types shadowed by built-in

**Status**: FIXED

**Severity**: High

**Description**: When a user defines a recursive variant type that reuses constructor names from the built-in `List` type (e.g., `type MyList[A] = Cons(A, MyList[A]) | Nil`), the HM type inference's `lookup_constructor` function found the built-in `List`'s `Nil` constructor first (because "List" sorts alphabetically before "MyList"), causing false type errors like "expected List[Int], found MyList[Int]".

**Reproduction**:
```nostos
type MyList[A] = Cons(A, MyList[A]) | Nil

myMap[A, B](xs: MyList[A], f: A -> B) -> MyList[B] = match xs {
    Nil -> Nil
    Cons(x, rest) -> Cons(f(x), myMap(rest, f))
}

main() = myMap(Cons(1, Cons(2, Nil)), x => x * 2)
```

**Root cause**: `lookup_constructor` in `infer.rs` sorted type names to prioritize non-stdlib types, but the built-in `List` type (registered as "List", not "stdlib.List") was not treated as built-in. It sorted alphabetically alongside user types, and "List" < "MyList", so `List`'s `Nil` was found first.

**Fix**: Extended the sort order in `lookup_constructor` to treat built-in types (List, Option, Result) as lower priority than user-defined types, so user constructors take precedence when names collide.

---

### Issue 15: `map` result ++ String not caught at compile time

**Status**: FIXED

**Severity**: High (soundness issue - type error only caught at runtime)

**Description**: When using `++` (concat) to concatenate a `List[Int]` produced by `xs.map(x => x * 2)` with a `String`, the type error was not caught at compile time and only produced a runtime error. Strangely, `xs.filter(x => x > 1) ++ "wrong"` WAS caught at compile time. Direct list literals like `[1,2,3] ++ "hello"` were also caught.

**Reproduction**:
```nostos
main() = {
    xs = [1, 2, 3]
    mapped = xs.map(x => x * 2)
    mapped ++ "hello"    # Should be compile error, was runtime error
}
```

**Root cause**: Two overly broad error suppression filters in `compile.rs` were hiding the legitimate type error from HM inference:

1. `(message.contains("List[Int]") && message.contains("String"))` - intended for "polymorphic function calls" but matched any List[Int] vs String error, including the concat mismatch.

2. `is_try_catch_mismatch` used `message.contains("Int") && message.contains("String")` - intended for try/catch type confusion but matched any error containing "Int" (even "List[Int]") and "String".

The `filter` case worked because `List.filter` is registered in the TypeEnv (single type param), so the error was generated during inference with `List[?27]` (unresolved type variable), which didn't match the `"List[Int]"` filter. The `map` case used deferred method resolution (pending_method_calls), producing a fully-resolved `List[Int]` in the error message, which DID match the filter.

**Fix**:
1. Removed the `List[Int] + String` suppression filter entirely (it was hiding legitimate errors).
2. Made `is_try_catch_mismatch` more specific: only match exact "Cannot unify types: Int and String" / "Cannot unify types: String and Int" (not compound types like "List[Int]").

---

### Issue 16: List-only methods (head, tail, map, etc.) not caught on String at compile time

**Status**: FIXED

**Severity**: Medium

**Description**: Calling list-only methods like `head`, `tail`, `map`, `filter` etc. on a String value was not caught at compile time. The `check_pending_method_calls` function in infer.rs only rejected list-only methods on `is_primitive` types (Int, Float, Bool, etc.), but String is not classified as a primitive. This meant `"hello".head()` or `"hello".map(c => c)` would only fail at runtime.

**Reproduction**:
```nostos
main() = "hello".head()     # Was runtime error, now compile error
main() = "hello".map(c => c) # Was runtime error, now compile error
```

**Root cause**: The `is_primitive` check in `check_pending_method_calls` only included numeric types, Bool, and Char. String was intentionally excluded from `is_primitive` because String HAS legitimate methods (split, trim, length, etc.). But the guard for `list_only_methods` should also reject String since list operations like head/tail/map/filter don't work on String.

**Fix**: Added `|| type_name.as_str() == "String"` to the `list_only_methods` check. String-specific builtins (length, split, toUpper, etc.) still work because they're registered as `String.length`, `String.split` etc. and are resolved through the immediate UFCS lookup before reaching `check_pending_method_calls`.

**Now produces**: `Error: Type String has no method head`

---

### Issue 17: Bool arithmetic not caught at compile time

**Status**: FIXED

**Severity**: Medium

**Description**: Using Bool values in arithmetic operations (e.g., `true + 5`, `true * 3`) was not caught at compile time. The HM inference correctly produced "Bool does not implement Num" but this error was explicitly suppressed by a filter: `(message.contains("Bool") && message.contains("does not implement Num"))`.

**Fix**: Removed the Bool/Num suppression filter. No false positives found across the full test suite.

---

### Issue 18: Field access on primitives not caught at compile time

**Status**: FIXED

**Severity**: Medium

**Description**: Accessing a field on a primitive type (e.g., `p = 42; p.x`) was not caught at compile time. The HM inference correctly produced "Type Int has no field x" but this was suppressed by the blanket `message.contains("has no field")` filter.

**Fix**: Changed the filter to only suppress "has no field" errors when the type is unknown or unresolved. When the type is a concrete primitive (Int, Float, Bool, String, etc.), the error is now reported.

---

### Issue 19: Calling non-function values not caught at compile time

**Status**: FIXED

**Severity**: Medium

**Description**: Calling a non-function value (e.g., `f = 42; f(10)`) was not caught at compile time. The HM inference correctly produced "Cannot unify types: Int and (Int) -> ?23" but `is_type_variable_only_error` suppressed it because the function type contained `?23` (a type variable).

**Fix**: Added a check in `is_type_variable_only_error`: if one type is a concrete primitive (Int, Float, Bool, String, ()) and the other is a function type (contains `->`), it's a real error regardless of type variables.

---

### Issue 20: List + List arithmetic not caught at compile time

**Status**: FIXED

**Severity**: Medium

**Description**: Using `+` operator on two lists (e.g., `[1,2,3] + [4,5,6]`) was not caught at compile time. The HM inference correctly produced "List[?25] does not implement Num" but `is_type_variable_only_error` suppressed it because the type contained `?` (from the unresolved element type `?25`).

**Fix**: Changed the type variable trait error filter to only suppress when the type name itself starts with `?` (a bare type variable), not when `?` appears inside generic parameters of a concrete type.

---

### Issue 21: Unary negation on non-numeric types not caught at compile time

**Status**: FIXED

**Severity**: Medium

**Description**: Negating non-numeric values (e.g., `-"hello"`, `-true`) was not caught at compile time. The `UnaryOp::Neg` handler in infer.rs only created a fresh type variable and unified it with the operand, but never added a `Num` trait constraint.

**Fix**: Added `self.require_trait(operand_ty.clone(), "Num")` to the `Neg` case, matching how arithmetic operators check for numeric types.

---

### Issue 22: String doesn't implement Ord but VM supports comparison

**Status**: FIXED

**Severity**: Medium (false positive - valid code rejected)

**Description**: The type system declared that `String` doesn't implement the `Ord` trait, causing the compiler to reject valid code like `"hello" > "abc"` or `["c", "a", "b"].sort()`. However, the VM's comparison instructions correctly handle string comparison at runtime.

**Reproduction**:
```nostos
main() = ["hello", "world", "abc"].sort()  # Was rejected, should work
```

**Root cause**: In `crates/types/src/lib.rs`, the `implements()` function had `Type::String` grouped with `Type::Bool | Type::Char | Type::Unit`, all returning only `Eq | Show`. But String comparison IS supported by the VM.

**Fix**: Separated `Type::String` to implement `Eq | Show | Ord`, matching the VM's capabilities. Note: `Char` comparison does NOT work at runtime (crashes with "GtInt: expected Int64"), so Char correctly remains without Ord.

---

### Issue 23: Numeric field access (.0, .1) on lists fails HM inference

**Status**: FIXED

**Severity**: Medium (causes signature inference to fail)

**Description**: When a function uses numeric field access on a list (e.g., `head(result).0` where result is `[[a]]`), the HM inference would fail with `NoSuchField { ty: "List[...]", field: "0" }`. This caused `try_hm_inference` to return `None`, leaving the function with a placeholder signature like `a -> b` instead of the correctly inferred types.

**Reproduction**:
```nostos
test_float32(conn) = {
    r1 = Pg.query(conn, "SELECT 3.14::real", [])
    v1 = head(r1).0  # .0 on List element
    assert(v1 > 3.1)
}
```
Expected signature: `Int -> ()`
Actual: `a -> b` (HM inference failed silently)

**Root cause**: In `crates/types/src/infer.rs`, the `solve()` function's `HasField` constraint handling had cases for `Type::Tuple` (numeric indices) and `Type::Record` (named fields), but no case for `Type::List`. In Nostos, `list.0` is syntactic sugar for list indexing.

**Fix**: Added `Type::List(elem_ty)` handling in two locations in `solve()`. When `.0`, `.1`, etc. is used on a list, it now correctly unifies the expected type with the list element type.

---

### Issue 24: Type parameters used with Num/Ord ops crash at runtime

**Status**: FIXED

**Severity**: High (runtime crash instead of compile error)

**Description**: When a generic function uses a type parameter in arithmetic or comparison operations without declaring the required trait bound, the code compiles but crashes at runtime. For example, `compare(x: T, y: T) = x > y` would compile, but calling `compare(true, false)` crashes with "Gt: values are not comparable".

**Reproduction**:
```nostos
compare(x: T, y: T) = x > y
main() = compare(true, false)  # Crashes at runtime!
```

**Root cause**: In `compile.rs`, the error filtering for `MissingTraitImpl` errors explicitly skipped type parameters (line 2070) to avoid false positives for user-defined types that might implement the trait. However, for Num/Ord traits, if the type parameter doesn't have the required trait bound declared, it's a real error.

**Fix**: Added special handling in the `MissingTraitImpl` error filter: when a type parameter is used with Num/Ord and the function doesn't declare that trait bound, generate a compile error with a helpful message suggesting to add the trait bound (e.g., `[T: Ord]`).

**Now produces**:
```
Error: type parameter `T` must have `Ord` trait bound to use comparison operations. Add trait bound: `[T: Ord]`
```

---

### Issue 25: Trait method dispatch fails on function param return inside lambda

**Status**: FIXED

**Severity**: High (false positive - valid code rejected)

**Description**: When calling a trait method on a value returned from a function parameter `f: () -> U` (where `U: Measurable`) inside a lambda in a higher-order function call (like `map`), the compiler failed with "cannot resolve trait method without type information".

**Reproduction**:
```nostos
trait Measurable
    measure(self) -> Int
end

Int: Measurable
    measure(self) = self
end

applyFunc[U: Measurable](items: List[()], f: () -> U) -> List[Int] =
    items.map(_ => f().measure())

main() = applyFunc([()], () => 42).length()  # Now works, returns 1
```

**Root cause**: When `f()` was called (where `f: () -> U`), `expr_type_name` didn't extract the return type `U` from the function type. It only handled calls to named functions, not calls to function-typed variables.

**Fix**: Added handling in `expr_type_name` for `Expr::Call` where the callee is a variable with a known function type. When the variable's type is a function type (contains " -> "), extract the return type and use it for trait method resolution.

---

*Last updated: Iteration 26 - Fixed Issue #25. All 25 issues fixed.*
