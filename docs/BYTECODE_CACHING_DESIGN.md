# Bytecode Caching Design Document

## Executive Summary

This document outlines the strategy for implementing bytecode caching in Nostos to dramatically reduce compilation time. The key insight is that **stdlib rarely changes** but is compiled on every run, taking ~1.5s. With caching, we can reduce startup to ~0.1-0.2s.

## Current Compilation Flow

```
User runs program
    ↓
Parse all .nos files (stdlib + user) → ~0.1s
    ↓
Type inference (HM) for all functions → ~0.3s
    ↓
Compile to bytecode (all functions) → ~1.3s
    ↓
Execute → varies
```

**Problem**: Steps 2-3 repeat identical work for stdlib on every run.

## Proposed Flow with Caching

```
User runs program
    ↓
Check cache validity (hash comparison) → ~0.01s
    ↓
If valid: Load cached bytecode → ~0.05s
If invalid: Recompile changed modules → varies
    ↓
Compile user code only → ~0.1s
    ↓
Execute
```

---

## Module Dependency Model

### The Dependency Graph

Modules form a directed acyclic graph (DAG) based on imports:

```
stdlib.list ← user.utils ← user.main
     ↑              ↑
stdlib.json ←───────┘
```

**Key Properties:**
1. If A imports B, then A depends on B
2. If B changes, A may need recompilation
3. Stdlib modules have no user dependencies

### What "Depends" Means

Module A depends on Module B if:
- A uses `use B.*` or `use B.{func1, func2}`
- A calls any function from B (even without explicit import)
- A uses any type defined in B

### Tracking Dependencies

The compiler already tracks this via:
- `imports: HashMap<String, String>` - local name → qualified name
- `imported_modules: HashSet<(Vec<String>, String)>` - (importer, imported)
- `CallGraph` - function-level call dependencies

---

## Invalidation Strategy

### The Core Question

When file X changes, which cached modules must be invalidated?

### Three Levels of Change

**Level 1: No Change (fastest)**
- File hash unchanged
- Cache is fully valid
- Action: Load cached bytecode

**Level 2: Body-Only Change (medium)**
- Function body changed, but signature unchanged
- Example: `add(x: Int, y: Int) -> Int = x + y` becomes `x + y + 0`
- Action: Recompile this module only
- Dependents: Keep cached (signature same, calls still valid)

**Level 3: Signature Change (full invalidation)**
- Function signature changed (params, return type, removed, renamed)
- Example: `add(x: Int, y: Int) -> Int` becomes `add(x: Int) -> Int`
- Action: Recompile this module AND all dependents
- Dependents must recompile because call sites may be invalid

### Signature-Based Invalidation Algorithm

```
function invalidate(changed_module):
    old_sigs = cache.get_signatures(changed_module)
    new_sigs = parse_and_extract_signatures(changed_module)

    # Find what changed
    removed = old_sigs.keys() - new_sigs.keys()
    changed = {f for f in old_sigs ∩ new_sigs if old_sigs[f] != new_sigs[f]}

    if removed or changed:
        # Signature change - invalidate dependents
        for dependent in get_dependents(changed_module):
            if uses_any(dependent, removed ∪ changed):
                mark_stale(dependent)
                invalidate(dependent)  # Recursive

    # Always recompile the changed module itself
    mark_for_recompile(changed_module)
```

### Transitive Invalidation

If A depends on B, and B depends on C:
- C changes signature → B invalidated → A invalidated
- B changes signature → A invalidated, C untouched
- A changes → Only A recompiles

This uses the existing `CallGraph::transitive_dependents()` function.

---

## Cache Structure

### Directory Layout

```
.nostos-cache/
├── manifest.json           # Global state
├── stdlib/
│   ├── list.cache         # Compiled bytecode
│   ├── list.sig           # Exported signatures
│   ├── json.cache
│   └── json.sig
└── user/
    ├── main.cache
    └── utils.cache
```

### Manifest Format

```json
{
  "format_version": 1,
  "compiler_version": "0.1.0",
  "created": "2024-01-20T12:00:00Z",
  "modules": {
    "stdlib.list": {
      "source_path": "stdlib/list.nos",
      "source_hash": "sha256:abc123...",
      "cache_path": "stdlib/list.cache",
      "signature_path": "stdlib/list.sig",
      "compiled_at": "2024-01-20T12:00:00Z"
    }
  },
  "dependency_graph": {
    "stdlib.list": [],
    "stdlib.json": ["stdlib.list"],
    "user.main": ["stdlib.list", "stdlib.json"]
  }
}
```

### Cache File Format (.cache)

Binary format containing:

```
Header:
  magic: [u8; 4] = "NOSC"  # Nostos Cache
  version: u32
  function_count: u32
  type_count: u32

Functions[]:
  name_len: u32
  name: [u8; name_len]
  arity: u32
  param_count: u32
  params[]: string
  chunk: SerializedChunk
  metadata: FunctionMetadata

Types[]:
  name_len: u32
  name: [u8; name_len]
  kind: u8  # Record=0, Variant=1, Alias=2
  type_data: ...
```

### Signature File Format (.sig)

```json
{
  "functions": {
    "map": {
      "type_params": ["T", "U"],
      "params": ["List[T]", "(T) -> U"],
      "return": "List[U]"
    },
    "filter": {
      "type_params": ["T"],
      "params": ["List[T]", "(T) -> Bool"],
      "return": "List[T]"
    }
  },
  "types": {
    "Option": {
      "kind": "variant",
      "type_params": ["T"],
      "constructors": ["Some(T)", "None"]
    }
  }
}
```

---

## What to Serialize

### Must Serialize (for bytecode execution)

| Field | Type | Notes |
|-------|------|-------|
| `Chunk.code` | `Vec<Instruction>` | The bytecode |
| `Chunk.constants` | `Vec<Value>` | Constant pool |
| `Chunk.lines` | `Vec<usize>` | Debug info |
| `Chunk.locals` | `Vec<String>` | Debug info |
| `Chunk.register_count` | `usize` | Execution |
| `FunctionValue.name` | `String` | Lookup key |
| `FunctionValue.arity` | `usize` | Call validation |
| `FunctionValue.param_names` | `Vec<String>` | Introspection |
| `FunctionValue.module` | `Option<String>` | Organization |

### Do NOT Serialize (regenerate at runtime)

| Field | Reason |
|-------|--------|
| `jit_code` | JIT compiles fresh on hot paths |
| `call_count` | Reset on each run |
| `source_code` | Can reload from file if needed |
| Type inference state | Regenerate for user code |

### Instruction Serialization

All `Instruction` variants use only:
- `Reg` (u8) - register indices
- `ConstIdx` (u16) - constant pool indices
- `JumpOffset` (i16) - jump offsets
- `Arc<[Reg]>` - register lists

This is trivially serializable as tagged binary.

### Value Serialization (constant pool)

**Serializable:**
- Primitives: Unit, Bool, Char, Int*, UInt*, Float*, BigInt, Decimal
- Strings: `Arc<String>` → length + bytes
- Collections: List, Tuple, Map, Set (recursive)

**NOT in constant pool (constructed at runtime):**
- Function, Closure, NativeFunction
- Record, Variant (data values)
- Mutable arrays
- Process IDs, References

---

## Implementation Phases

### Phase 1: Stdlib-Only Caching (Recommended Start)

**Goal**: Cache stdlib, always recompile user code

**Benefits**:
- Biggest performance win (~1.5s → ~0.1s)
- Simplest invalidation (stdlib rarely changes)
- No cross-module dependency tracking needed

**Implementation**:
1. Add `--build-cache` flag to compile and cache stdlib
2. On normal run, load stdlib from cache
3. Compile user code as usual

### Phase 2: User Code Caching

**Goal**: Cache user modules too

**Challenges**:
- User code changes frequently
- Need dependency tracking between user modules
- Incremental recompilation

**Implementation**:
1. Track module dependencies in manifest
2. On change, use signature comparison
3. Invalidate transitively

### Phase 3: Granular Invalidation

**Goal**: Minimize recompilation

**Features**:
- Function-level caching
- Signature-based invalidation
- Parallel recompilation of independent modules

---

## Serialization Library Choice

### Option A: bincode (Recommended)

```rust
// Add to Cargo.toml
bincode = "1.3"
serde = { version = "1.0", features = ["derive"] }

// Usage
#[derive(Serialize, Deserialize)]
struct CachedChunk { ... }

let bytes = bincode::serialize(&chunk)?;
let chunk: CachedChunk = bincode::deserialize(&bytes)?;
```

**Pros**: Fast, compact, well-maintained
**Cons**: Non-human-readable

### Option B: postcard (Embedded-friendly)

More compact than bincode, good for constrained environments.

### Option C: Custom Binary

Full control, no dependencies, but more code to maintain.

**Recommendation**: Start with bincode for simplicity.

---

## Risk Mitigation

### Cache Corruption

- Include checksum in each cache file
- On checksum mismatch, delete and recompile
- Never trust cache blindly

### Compiler Version Mismatch

- Store compiler version in manifest
- If version differs, invalidate entire cache
- Consider bytecode format version separately

### Concurrent Access

- Use file locking when writing cache
- Or atomic rename: write to `.cache.tmp`, rename to `.cache`

### Disk Space

- Estimate: ~100KB per stdlib module, ~1MB total
- Add `--clear-cache` flag
- Consider LRU eviction for user code cache

---

## Performance Expectations

| Scenario | Current | With Caching |
|----------|---------|--------------|
| Cold start (no cache) | ~1.7s | ~1.7s + cache write |
| Warm start (cached stdlib) | ~1.7s | ~0.2s |
| User code change | ~1.7s | ~0.3s |
| Stdlib change | ~1.7s | ~1.8s (invalidate + rebuild) |

**Expected improvement**: 5-10x faster typical startup.

---

## Next Steps

1. **Add serde/bincode to vm crate** - derive Serialize/Deserialize
2. **Create cache directory structure** - `.nostos-cache/`
3. **Implement Phase 1** - stdlib-only caching
4. **Add --build-cache flag** - for explicit cache building
5. **Test invalidation** - modify stdlib, verify recompile
6. **Implement Phase 2** - user code caching with dependencies
