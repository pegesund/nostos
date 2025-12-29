# Nostos FFI (Foreign Function Interface)

This document explains how Nostos integrates with native Rust extensions, allowing high-performance libraries like `nalgebra` to be used from Nostos code.

## Overview

The FFI system has three main components:

1. **Nostos Interface Module** (`.nos` file) - Defines types and function signatures in Nostos
2. **Rust Extension Library** (`.so`/`.dylib`) - Implements the native functions
3. **Extension Manager** - Loads and dispatches calls to native functions

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   Nostos Code       │     │   Extension Library  │
│   (nalgebra.nos)    │────▶│   (libnostos_*.so)   │
│                     │     │                      │
│  type Vec = {...}   │     │  fn dvecAdd(...)     │
│  pub vec(data) =    │     │  fn dvecDot(...)     │
│    __native__       │     │  fn dmatMul(...)     │
└─────────────────────┘     └──────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────┐
│              Extension Manager                   │
│  - Loads .so/.dylib files                       │
│  - Maintains function registry                  │
│  - Indexed dispatch for performance             │
└─────────────────────────────────────────────────┘
```

## The nalgebra Extension

The `nalgebra` extension provides dynamic-sized vector and matrix operations. Here's the complete API:

### Types

```nostos
type Vec = { data: List }   # Dynamic vector
type Mat = { data: List }   # Dynamic matrix
```

Both types implement the `Num` trait for operator overloading (`+`, `-`, `*`, `/`).

### Vec Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `vec(data)` | `List -> Vec` | Create vector from list |
| `vecZeros(n)` | `Int -> Vec` | Create n-length zero vector |
| `vecOnes(n)` | `Int -> Vec` | Create n-length ones vector |
| `vecLen(v)` | `Vec -> Int` | Get vector length |
| `vecGet(v, i)` | `Vec -> Int -> Float` | Get element at index |
| `vecSum(v)` | `Vec -> Float` | Sum of all elements |
| `vecMin(v)` | `Vec -> Float` | Minimum element |
| `vecMax(v)` | `Vec -> Float` | Maximum element |
| `vecDot(a, b)` | `Vec -> Vec -> Float` | Dot product |
| `vecNorm(v)` | `Vec -> Float` | Euclidean norm |
| `vecNormalize(v)` | `Vec -> Vec` | Unit vector |
| `vecScale(v, s)` | `Vec -> Float -> Vec` | Scalar multiplication |
| `vecDistance(a, b)` | `Vec -> Vec -> Float` | Euclidean distance |

### Mat Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `mat(data)` | `List -> Mat` | Create matrix from nested list |
| `matIdentity(n)` | `Int -> Mat` | Create n×n identity matrix |
| `matZeros(r, c)` | `Int -> Int -> Mat` | Create r×c zero matrix |
| `matOnes(r, c)` | `Int -> Int -> Mat` | Create r×c ones matrix |
| `matRows(m)` | `Mat -> Int` | Get row count |
| `matCols(m)` | `Mat -> Int` | Get column count |
| `matGet(m, r, c)` | `Mat -> Int -> Int -> Float` | Get element |
| `matTranspose(m)` | `Mat -> Mat` | Transpose matrix |
| `matTrace(m)` | `Mat -> Float` | Sum of diagonal |
| `matDeterminant(m)` | `Mat -> Float` | Matrix determinant |
| `matInverse(m)` | `Mat -> Mat` | Matrix inverse |
| `matScale(m, s)` | `Mat -> Float -> Mat` | Scalar multiplication |
| `matPow(m, n)` | `Mat -> Int -> Mat` | Matrix power |
| `matMulVec(m, v)` | `Mat -> Vec -> Vec` | Matrix-vector multiply |
| `matGetRow(m, r)` | `Mat -> Int -> Vec` | Extract row as vector |
| `matGetCol(m, c)` | `Mat -> Int -> Vec` | Extract column as vector |

### Operator Overloading

```nostos
v1 = vec([1, 2, 3])
v2 = vec([4, 5, 6])

v1 + v2   # Vec{data: [5, 7, 9]}
v1 - v2   # Vec{data: [-3, -3, -3]}
v1 * v2   # Component-wise: Vec{data: [4, 10, 18]}

m1 = mat([[1, 2], [3, 4]])
m2 = mat([[5, 6], [7, 8]])

m1 + m2   # Matrix addition
m1 * m2   # Matrix multiplication
m1 / m2   # m1 * inverse(m2)
```

### UFCS Method Syntax

All functions can be called as methods via UFCS:

```nostos
v = vec([1, 2, 3, 4, 5])
v.vecLen()        # 5
v.vecNorm()       # 7.416...
v.vecNormalize()  # unit vector
v.vecSum()        # 15.0
```

## Creating Your Own Extension

### Step 1: Define the Nostos Interface

Create a `.nos` file that defines types and declares native functions:

```nostos
# mylib.nos

type MyType = { data: List }

# Use __native__("ExtensionName.functionName", args...) to call native code
pub myFunc(x: MyType) -> MyType = MyType(__native__("MyLib.process", x.data))
```

### Step 2: Implement the Rust Extension

The native library exports functions that the VM can call. The extension registers its functions with fully qualified names.

### Step 3: Function Naming Convention

Native function names follow this pattern:

```
module.function_name/param_type1,param_type2
```

Examples:
- `nalgebra.vec/_` - `vec` function with untyped parameter
- `nalgebra.vecDot/Vec,Vec` - `vecDot` with two `Vec` parameters
- `nalgebra.Vec.nalgebra.Num.add/_,Vec` - trait method `add` on `Vec` implementing `Num`

### Step 4: Build and Load

```bash
# Build the extension
cargo build --release

# The library will be at target/release/libnostos_mylib.so (Linux)
# or target/release/libnostos_mylib.dylib (macOS)
```

In Nostos:
```nostos
import mylib  # Loads both .nos interface and .so library
```

## REPL Usage

After loading a project with extensions:

```
>>> use nalgebra.*
imported: vec, vecDot, vecNorm, vecLen, mat, matMul, ...

>>> v1 = vec([1, 2, 3])
v1 = vec([1, 2, 3])

>>> v2 = vec([4, 5, 6])
v2 = vec([4, 5, 6])

>>> v1 + v2
nalgebra.Vec{data: [5, 7, 9]}

>>> v1.vecDot(v2)
32.0

>>> v1.vecNorm()
3.7416573867739413

>>> m = mat([[1, 2], [3, 4]])
>>> m.matDeterminant()
-2.0

>>> m.matInverse()
nalgebra.Mat{data: [[-2, 1], [1.5, -0.5]]}
```

Key points:
- Use explicit `use module.*` to import extension functions
- Types are properly qualified (e.g., `nalgebra.Vec`)
- Operators work via trait implementations
- UFCS methods appear in autocomplete (e.g., `v.vecLen()`)

## How It Works Internally

### 1. Compilation Phase

When the compiler sees `__native__("Name", args)`:
1. Records the function as a native call
2. Generates `CallExtension` or `CallExtensionIdx` instruction
3. Stores the qualified function name for runtime lookup

### 2. Extension Loading

When `import module` is executed:
1. Finds the `.nos` interface file
2. Finds the matching `.so`/`.dylib` library
3. Calls `dlopen` to load the library
4. Calls the `nostos_extension_init` entry point
5. Registers all exported functions with their indices

### 3. Runtime Dispatch

Two dispatch modes for performance:

**Indexed Dispatch (fast path):**
```
CallExtensionIdx { index: 42, arity: 2 }
```
- Used when extension index is known at compile time
- Direct array lookup, no string comparison

**Named Dispatch (fallback):**
```
CallExtension { name: "Nalgebra.dvecAdd", arity: 2 }
```
- Used for dynamic calls or first invocation
- Hash map lookup by function name

### 4. Type Qualification

The REPL maintains type qualification for proper operator dispatch:

1. `vec([1,2,3])` returns type `nalgebra.Vec` (not just `Vec`)
2. When evaluating `v1 + v2`, the VM knows both are `nalgebra.Vec`
3. Looks up trait implementation `nalgebra.Vec: nalgebra.Num`
4. Finds and calls `nalgebra.Vec.nalgebra.Num.add`

## Files Involved

| File | Purpose |
|------|---------|
| `crates/vm/src/extension.rs` | ExtensionManager, function registry |
| `crates/vm/src/worker.rs` | CallExtension/CallExtensionIdx handling |
| `crates/compiler/src/compile.rs` | `__native__` compilation |
| `crates/repl/src/engine.rs` | REPL imports, type qualification |

## Performance Considerations

1. **Indexed dispatch**: Pre-computed indices avoid hash lookups
2. **Inline execution**: Fast native calls run inline (no thread spawning)
3. **Blocking calls**: Long-running operations can be marked to run on I/O thread pool

## Current Limitations

1. Extensions must be compiled for the same platform
2. No automatic memory management across FFI boundary
3. Complex types (closures, processes) cannot cross FFI boundary directly

## Future Improvements

- [ ] Automatic binding generation from Rust types
- [ ] Support for async native functions
- [ ] Cross-platform extension distribution
- [ ] Generic type support in native functions
