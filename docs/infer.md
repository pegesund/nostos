# Nostos Type Inference Audit

This document tracks discovered type inference issues in the Nostos language.

## Summary

| # | Issue | Status | Severity |
|---|-------|--------|----------|
| 1 | Unknown type in method error messages | Open | Low |
| 2 | zipWith wrong arg order: runtime vs compile error | Open | Medium |

## Discovered Issues

### Issue 1: "unknown" type in deferred method error messages

**Severity**: Low (UX issue)

**Description**: When a method doesn't exist on a type that was inferred through a type variable, the error says `type 'unknown'` instead of showing the actual resolved type.

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

**Expected**: Error should say "no method `nonexistent` found for type `List[Int]`"

**Actual**: Error says "no method `nonexistent` found for type `unknown`"

**Impact**: Makes debugging harder when working with generic/higher-order code.

---

### Issue 2: zipWith with wrong argument order causes runtime error instead of compile error

**Severity**: Medium

**Description**: Calling `zipWith(f, xs, ys)` instead of `zipWith(xs, ys, f)` causes a runtime error ("Length: unsupported type") instead of a compile-time type mismatch error.

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

### Known Limitations (not bugs):
- 15_polymorphic_recursion.nos - Polymorphic recursion requires explicit types (HM limitation)
- 30_recursive_lambda.nos - Local recursive lambdas need Y combinator (no `let rec`)

---

*Last updated: Iteration 1*
