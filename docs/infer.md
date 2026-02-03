# Nostos Type Inference Audit

This document tracks discovered type inference issues in the Nostos language.

## Summary

| # | Issue | Status | Severity |
|---|-------|--------|----------|
| 1 | Unknown type in method error messages | **Fixed** | Low |
| 2 | zipWith wrong arg order: runtime vs compile error | Open | Medium |
| 3 | Misleading "if/else branches" error for numeric type mismatches | **Fixed** | Low |
| 4 | sortBy wrong function arity: runtime vs compile error | **Fixed** | High |
| 5 | **Lambdas silently ignore extra arguments** | **Fixed** | **Critical** |

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

**Status**: ROOT CAUSE IDENTIFIED (requires cache fix)

**Description**: Calling `zipWith(f, xs, ys)` instead of `zipWith(xs, ys, f)` causes a runtime error ("Length: unsupported type") instead of a compile-time type mismatch error.

**Root Cause**: Two separate issues identified:

1. **Stdlib Cache Issue (PRIMARY)**: When stdlib loads from the bytecode cache (`~/.nostos/cache/stdlib/`), function signatures are NOT registered with their `type_params`. The cache stores compiled bytecode but doesn't store/restore the type parameter information needed for HM type inference. This means `instantiate_function` can't properly freshen type variables like `T`, `U`, `V` in generic signatures like `List[T]`.

2. **Type Param Propagation**: When `type_params` is empty in FunctionType, `freshen_type` can't replace `TypeParam("T")` or `Named { name: "T", args: [] }` with fresh type variables, so `List[T]` incorrectly unifies with a function type.

**Fix Required**: Modify `try_load_stdlib_from_cache` in `crates/cli/src/main.rs` to also register function signatures with type_params into `pending_fn_signatures`. Currently, only the compiled bytecode is loaded, but type signatures for HM inference are missing.

**Reproduction**:
```nostos
main() = {
    # WRONG order - should be (xs, ys, f)
    combined = zipWith((n, s) => show(n) ++ s, [1,2,3], ["a","b","c"])
    0
}
```

**Expected**: Compile-time error about type mismatch (function vs List)

**Actual**: Runtime error: "Panic: Length: unsupported type"

**Impact**: Type errors slip through to runtime, violating type safety guarantees.

**Technical details**: The `instantiate_function` in `infer.rs` creates `param_subst` from `func_ty.type_params`. When type_params is properly populated, TypeParam/Named type parameters inside List get replaced with fresh vars. When type_params is empty (due to various code paths not preserving it), the replacement doesn't happen and unification between List[T] and Function succeeds (because the element T is never constrained).

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

## Known Stdlib Issues

### json.nos: convertNumber function missing type annotation

The `convertNumber(n, typeName)` function in `stdlib/json.nos` doesn't have a type annotation for `n`. With stricter type checking, calling `n.asInt64()` on a polymorphic type now correctly fails at compile time. This was previously undetected because the stdlib was loaded from cache (bypassing type inference).

**Fix needed**: Add type annotation `n: Float` or use a trait constraint.

---

*Last updated: Iteration 5 - Fixed Issue #1 (improved error messages for polymorphic types). Issues #1, #3, #4, #5 now fixed. Issue #2 root cause identified: stdlib cache doesn't preserve type_params for HM inference.*
