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
│  type Vec = {...}   │     │  fn vec_new(...)     │
│  pub vec(data) =    │     │  fn vec_add(...)     │
│    __native__       │     │  fn vec_dot(...)     │
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

## Creating an Extension

### Step 1: Define the Nostos Interface

Create a `.nos` file that defines types and declares native functions:

```nostos
# nalgebra.nos - Vector/Matrix math library

# Type definition - wraps native data
type Vec = { data: List }

# Trait implementation for operator overloading
trait Num
    add(self, other: Self) -> Self
    sub(self, other: Self) -> Self
    mul(self, other: Self) -> Self
    div(self, other: Self) -> Self
end

Vec: Num
    add(self, other: Vec) -> Vec = __native__
    sub(self, other: Vec) -> Vec = __native__
    mul(self, other: Vec) -> Vec = __native__
    div(self, other: Vec) -> Vec = __native__
end

# Constructor - calls native implementation
pub vec(data: List) -> Vec = __native__

# Other operations
pub vecDot(a: Vec, b: Vec) -> Float = __native__
pub vecNorm(v: Vec) -> Float = __native__
pub vecNormalize(v: Vec) -> Vec = __native__
```

Key points:
- Use `__native__` as the function body to mark native implementations
- Use `pub` for functions that should be importable
- Trait implementations enable operator overloading (`+`, `-`, `*`, `/`)

### Step 2: Implement the Rust Extension

Create a Rust library with the `nostos-extension` crate:

```rust
// src/lib.rs
use nostos_extension::{
    extension_init, ExtensionFunction, NostosValue, NostosError
};

// Initialize the extension - called when library is loaded
extension_init! {
    name: "nalgebra",
    version: "0.1.0",
    functions: [
        ("nalgebra.vec/_", vec_new),
        ("nalgebra.Vec.nalgebra.Num.add/_,Vec", vec_add),
        ("nalgebra.Vec.nalgebra.Num.sub/_,Vec", vec_sub),
        ("nalgebra.vecDot/Vec,Vec", vec_dot),
        ("nalgebra.vecNorm/Vec", vec_norm),
    ]
}

fn vec_new(args: &[NostosValue]) -> Result<NostosValue, NostosError> {
    // args[0] is the List passed to vec()
    let list = args[0].as_list()?;
    let data: Vec<f64> = list.iter()
        .map(|v| v.as_float())
        .collect::<Result<_, _>>()?;

    // Return a record with the data field
    Ok(NostosValue::record("Vec", vec![
        ("data", NostosValue::list(data.into_iter().map(NostosValue::float).collect()))
    ]))
}

fn vec_add(args: &[NostosValue]) -> Result<NostosValue, NostosError> {
    let a = extract_vec_data(&args[0])?;
    let b = extract_vec_data(&args[1])?;

    let result: Vec<f64> = a.iter().zip(b.iter())
        .map(|(x, y)| x + y)
        .collect();

    Ok(NostosValue::record("Vec", vec![
        ("data", NostosValue::list(result.into_iter().map(NostosValue::float).collect()))
    ]))
}
```

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

# The library will be at target/release/libnostos_nalgebra.so (Linux)
# or target/release/libnostos_nalgebra.dylib (macOS)
```

In Nostos:
```nostos
import nalgebra  # Loads both .nos interface and .so library
```

## REPL Usage

After loading a project with extensions:

```
>>> use nalgebra.*
imported: vec, vecDot, vecNorm, Vec.nalgebra.Num.add, ...

>>> v1 = vec([1, 2, 3])
v1 = vec([1, 2, 3])

>>> v2 = vec([4, 5, 6])
v2 = vec([4, 5, 6])

>>> v1 + v2
nalgebra.Vec{data: [5, 7, 9]}

>>> vecDot(v1, v2)
32.0
```

Key points:
- Use explicit `use module.*` to import extension functions
- Types are properly qualified (e.g., `nalgebra.Vec`)
- Operators work via trait implementations

## How It Works Internally

### 1. Compilation Phase

When the compiler sees `__native__`:
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
CallExtension { name: "nalgebra.vec/_", arity: 1 }
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
