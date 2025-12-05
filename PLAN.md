# Plan: Expanded Numeric Types

## Overview
Add comprehensive numeric type support to Nostos:
- **Signed integers**: Int8, Int16, Int32, Int64 (current Int becomes Int64)
- **Unsigned integers**: UInt8, UInt16, UInt32, UInt64
- **Floats**: Float32, Float64 (current Float becomes Float64)
- **Arbitrary precision**: BigInt

## Naming Convention
- Capitalized names: `Int64`, `Float32`, `UInt8`
- Type aliases for convenience: `Int = Int64`, `Float = Float64`, `Byte = UInt8`

## Literal Syntax
- `42` → Int64 (default)
- `42i8`, `42i16`, `42i32` → Int8, Int16, Int32
- `42u8`, `42u16`, `42u32`, `42u64` → unsigned
- `3.14` → Float64 (default)
- `3.14f32` → Float32
- `42n` → BigInt

## Implementation Steps

### Phase 1: Value Representation
**Files:** `crates/vm/src/value.rs`, `crates/vm/src/gc.rs`

1. Update `Value` enum with all numeric variants
2. Update `GcValue` enum to match
3. Add BigInt support (using `num-bigint` crate)
4. Update type_name(), Debug, Display, PartialEq impls

### Phase 2: Lexer & Parser
**Files:** `crates/syntax/src/lexer.rs`, `crates/syntax/src/parser.rs`, `crates/syntax/src/ast.rs`

1. Add numeric suffixes to lexer (i8, i16, i32, u8, u16, u32, u64, f32, n)
2. Parse numeric literals with type suffixes
3. Add new type names to the type expression parser

### Phase 3: Type System
**Files:** `crates/compiler/src/types.rs` (if exists), `crates/compiler/src/compile.rs`

1. Define type hierarchy for numeric types
2. Add type checking for numeric operations
3. Implement explicit conversion functions: `toInt8()`, `toUInt32()`, `toFloat32()`, etc.
4. Generate appropriate instructions based on operand types

### Phase 4: Instructions
**Files:** `crates/vm/src/value.rs` (Instruction enum)

Strategy: Use polymorphic instructions with runtime type dispatch (keeps instruction count manageable).

1. Update arithmetic instructions to handle all numeric types
2. Add conversion instructions: `ConvertInt8`, `ConvertUInt32`, `ConvertFloat32`, etc.
3. Add overflow-checked variants for integer arithmetic (optional)

### Phase 5: Runtime
**Files:** `crates/vm/src/runtime.rs`, `crates/vm/src/gc.rs`

1. Implement arithmetic for all numeric types
2. Handle type mismatches gracefully
3. Implement BigInt operations
4. Update GC to handle BigInt values

### Phase 6: JIT
**Files:** `crates/jit/src/lib.rs`

1. Add Cranelift types: I8, I16, I32, I64, F32, F64
2. Support JIT for pure functions of any single numeric type
3. Add type specialization - generate optimized code for each numeric type
4. Extend function signature support beyond just `fn(i64) -> i64`

### Phase 7: Builtins & Standard Library
**Files:** `crates/compiler/src/compile.rs` (builtins)

1. Add conversion builtins: `toInt8()`, `toInt16()`, etc.
2. Add overflow-checking arithmetic: `addChecked()`, `mulChecked()`
3. Add BigInt operations: `bigAdd()`, `bigMul()`, `bigPow()`

## Type Conversion Rules (Explicit Only)

Conversions require explicit function calls:
- `toInt8(x)`, `toInt16(x)`, `toInt32(x)`, `toInt64(x)`
- `toUInt8(x)`, `toUInt16(x)`, `toUInt32(x)`, `toUInt64(x)`
- `toFloat32(x)`, `toFloat64(x)`
- `toBigInt(x)`

Narrowing conversions may truncate or fail:
- Overflow behavior: wrap (default) or error (with checked variants)

## Value Enum Changes

```rust
pub enum Value {
    Unit,
    Bool(bool),
    // Signed integers
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    // Unsigned integers
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    // Floats
    Float32(f32),
    Float64(f64),
    // Arbitrary precision
    BigInt(Rc<num_bigint::BigInt>),
    // ... rest unchanged (Char, String, List, etc.)
}
```

## GcValue Enum Changes

```rust
pub enum GcValue {
    Unit,
    Bool(bool),
    // Signed integers
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    // Unsigned integers
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    // Floats
    Float32(f32),
    Float64(f64),
    // Arbitrary precision (heap allocated)
    BigInt(GcPtr<GcBigInt>),
    // ... rest unchanged
}
```

## Migration Path

1. Current `Int` becomes alias for `Int64`
2. Current `Float` becomes alias for `Float64`
3. Existing code continues to work unchanged
4. New code can use specific types when needed

## Dependencies

Add to `Cargo.toml`:
```toml
num-bigint = "0.4"
num-traits = "0.2"
```

## Testing Strategy

1. Unit tests for each numeric type's arithmetic
2. Overflow tests for each integer size
3. Conversion tests (valid and invalid)
4. BigInt edge cases (very large numbers)
5. JIT tests for each numeric type
6. Integration tests with mixed numeric types
