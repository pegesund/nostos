# Bytecode Cache Analysis - Phase 1 Investigation

## Question: Does `check_module_compiles` (Phase 1) use cache?

**Answer: NO - Phase 1 does NOT use bytecode cache**

### Evidence

1. **`check_module_compiles` implementation** (engine.rs:5884-6300)
   - Builds `known_functions` set by querying **compiler state**, not cache
   - Line 5998: `self.compiler.get_known_modules()`
   - Line 6021: `self.compiler.get_module_public_functions()`
   - Line 6058: `self.compiler.get_function_names()`
   - Line 6076: `self.compiler.get_all_types()`
   - **No cache lookups found in check_module_compiles**

2. **Cache methods exist but aren't called in Phase 1**
   - `get_cached_module()` exists (engine.rs:8906)
   - But grep shows NO calls to this in check_module_compiles

3. **`load_directory` doesn't use cache either**
   - Line 4793: Calls `compiler.add_module()` directly
   - No cache lookup before compiling
   - Compiles every module from source every time

## Critical Finding: Cache is Currently Unused!

The bytecode cache infrastructure exists but appears to NOT be integrated into:
- `load_directory` - loads project modules
- `check_module_compiles` - Phase 1 type checking
- `recompile_module_with_content` - Phase 2 actual compilation

This explains why:
- Startup is always slow (2s instead of 0.1s)
- No performance benefit from caching
- Cache files are created but never loaded

## Where Cache SHOULD Be Used

### Scenario 1: Project Load (load_directory)
```rust
// CURRENT (engine.rs:4793):
self.compiler.add_module(&module, components, source, path);

// SHOULD BE:
let source_hash = compute_hash(&source);
if let Some(cached) = self.get_cached_module(&module_name, &source_hash) {
    // Use cached bytecode, skip compilation
    load_from_cache(cached);
} else {
    // Compile and cache
    self.compiler.add_module(&module, components, source, path);
    self.store_cached_module(&module_name, &source_hash, compiled_data);
}
```

### Scenario 2: Phase 1 Type Checking (check_module_compiles)
```rust
// Phase 1 must validate dependencies are fresh BEFORE using cached types

// SHOULD:
1. Compute source hash
2. Check if cache exists AND is valid:
   - Source hash matches
   - ALL imported modules have matching signatures
   - Compiler version matches
3. If cache valid: use cached type information
4. If cache stale: recompile to get fresh types
```

### Scenario 3: Two-Phase with Dependencies

**TUI Editor Scenario:**
1. User opens file A (imports B)
2. Phase 1 on A: checks types using cached B signatures
3. User externally edits B, changes function signature
4. Phase 1 on A again: MUST detect B changed!
   - Current: Would use stale cached types from B
   - Correct: Detect B's source hash changed, invalidate A's cache

## Missing Features

### 1. Dependency Signature Validation
```rust
// Cache needs to store:
struct CachedModule {
    source_hash: String,
    function_signatures: HashMap<String, FunctionSignature>,
    dependency_signatures: HashMap<String, HashMap<String, FunctionSignature>>,
    //                      ^module_name  ^fn_name     ^signature
}

// On cache load:
fn validate_dependencies(cache: &CachedModule) -> Result<(), String> {
    for (dep_module, expected_sigs) in &cache.dependency_signatures {
        let current_cache = load_module_cache(dep_module)?;
        for (fn_name, expected_sig) in expected_sigs {
            let actual_sig = current_cache.function_signatures.get(fn_name)?;
            if actual_sig != expected_sig {
                return Err(format!("{}.{} signature changed", dep_module, fn_name));
            }
        }
    }
    Ok(())
}
```

### 2. Transitive Invalidation
When C changes → B cache stale → A cache stale

Requires dependency graph in manifest:
```rust
struct CacheManifest {
    dependency_graph: HashMap<String, Vec<String>>,
    // "module_a" -> ["module_b", "module_c"]
}

fn invalidate_transitive(module: &str, manifest: &mut CacheManifest) {
    let mut to_invalidate = vec![module];
    let mut invalidated = HashSet::new();

    while let Some(m) = to_invalidate.pop() {
        if invalidated.insert(m) {
            // Find all modules that depend on m
            for (dependent, deps) in &manifest.dependency_graph {
                if deps.contains(&m) {
                    to_invalidate.push(dependent);
                }
            }
        }
    }

    // Delete cache files for all invalidated modules
    for m in invalidated {
        delete_cache(m);
    }
}
```

### 3. Compiler State vs Cache Consistency

**Problem:** Compiler state (types, functions) might differ from cache

**Solution:** When loading from cache, must also populate compiler state:
```rust
fn load_from_cache(cached: &CachedModule) -> Result<(), String> {
    // 1. Load bytecode into VM
    vm.load_functions(&cached.functions);

    // 2. Populate compiler state so check_module_compiles works
    compiler.register_types(&cached.types);
    compiler.register_function_signatures(&cached.function_signatures);
    compiler.register_exports(&cached.exports);

    // 3. Update call graph for dependency tracking
    call_graph.add_dependencies(&cached.module_name, &cached.dependencies);
}
```

## Next Steps

1. ✅ **Test 1-3 created** - Document expected behavior
2. **Add cache loading to load_directory** - Use cache on project load
3. **Add dependency validation** - Check imported signatures match
4. **Add cache to check_module_compiles** - Phase 1 uses cached types
5. **Add two-phase tests** - Verify Phase 1 detects stale dependencies

## Test Coverage Needed

- [ ] Project load uses cache (fast startup)
- [ ] Phase 1 validates dependency signatures
- [ ] Phase 1 detects when imported module changed
- [ ] Two-phase: edit B, Phase 1 on A detects change
- [ ] Transitive: change C, invalidates B and A
- [ ] Compiler version mismatch rejects cache
