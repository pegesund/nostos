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
| 6 | Trait bounds not propagated into lambdas | **Partially Fixed** | High |

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

**Status**: **Partially Fixed** (basic case works, some edge cases remain)

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

**Now passing**: All `tests/type_inference/trait_closure_*.nos` tests (8 tests)

**Remaining edge cases**: Some complex scenarios like:
- Nested records with trait methods (e.g., `items.head().level2.level3.triple()`)
- Deeply nested lambdas calling trait methods
- Some stdlib interactions with trait bounds (dropWhile, takeWhile, partition)

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

*Last updated: Iteration 8 - Partially fixed Issue #6 (trait bounds propagation into lambdas). Basic cases now work, all trait_closure_* tests pass. 5.5 of 6 issues fixed.*
