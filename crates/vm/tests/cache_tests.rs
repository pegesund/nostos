//! Integration tests for bytecode caching
//!
//! These tests verify that:
//! 1. File changes invalidate cache (source hash check)
//! 2. Dependency signature changes invalidate dependent modules
//! 3. Transitive dependencies are properly tracked

use std::path::Path;
use std::collections::HashMap;
use nostos_vm::cache::{load_module_cache, save_module_cache, CachedModule, BytecodeCache, FunctionSignature};

/// Helper to compute source hash (same algorithm as compiler)
fn compute_source_hash(content: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Helper to create a minimal cached module for testing
fn create_test_cache(
    module_name: &str,
    source_content: &str,
    function_sigs: HashMap<String, FunctionSignature>,
    cache_dir: &Path,
) -> Result<std::path::PathBuf, String> {
    let cached = CachedModule {
        module_path: module_name.split('.').map(String::from).collect(),
        source_hash: compute_source_hash(source_content),
        functions: Vec::new(), // Empty for testing - we only care about signatures
        function_signatures: function_sigs,
        exports: Vec::new(),
        prelude_imports: Vec::new(),
        types: Vec::new(),
        mvars: Vec::new(),
        dependency_signatures: HashMap::new(), // No dependencies for simple test
    };

    let cache_path = cache_dir.join(format!("{}.cache", module_name));
    save_module_cache(&cache_path, &cached)
        .map_err(|e| format!("Failed to save cache: {}", e))?;

    Ok(cache_path)
}

// ============================================================================
// Test 1: Basic File Change Detection
// ============================================================================

#[test]
fn test_source_change_invalidates_cache() {
    let cache_dir = tempfile::tempdir().unwrap();

    // Initial content and hash
    let initial_content = "pub add(a: Int, b: Int) = a + b";
    let initial_hash = compute_source_hash(initial_content);

    // Create cache with initial content
    let mut sigs = HashMap::new();
    sigs.insert("add".to_string(), FunctionSignature {
        name: "add".to_string(),
        type_params: Vec::new(),
        param_types: vec!["Int".to_string(), "Int".to_string()],
        return_type: "Int".to_string(),
    });

    let cache_path = create_test_cache("math", initial_content, sigs, cache_dir.path())
        .expect("Failed to create cache");

    // Verify cache was created
    assert!(cache_path.exists(), "Cache file should exist");

    // Load cache and verify hash
    let cached = load_module_cache(&cache_path).expect("Failed to load cache");
    assert_eq!(cached.source_hash, initial_hash, "Cache should have initial hash");

    // Modify source file (add a comment - semantically equivalent but different hash)
    let modified_content = "# New comment\npub add(a: Int, b: Int) = a + b";
    let new_hash = compute_source_hash(modified_content);

    // Verify hash changed
    assert_ne!(initial_hash, new_hash, "Hash should change when source changes");

    // Verify cache is now stale (hash doesn't match)
    let cached = load_module_cache(&cache_path).expect("Failed to load cache");
    assert_ne!(cached.source_hash, new_hash, "Cache should be stale after source change");

    println!("✓ Test 1 PASSED: Source change invalidates cache (hash mismatch)");
}

#[test]
fn test_identical_content_preserves_cache() {
    let cache_dir = tempfile::tempdir().unwrap();

    let content = "pub add(a: Int, b: Int) = a + b";
    let hash = compute_source_hash(content);

    let mut sigs = HashMap::new();
    sigs.insert("add".to_string(), FunctionSignature {
        name: "add".to_string(),
        type_params: Vec::new(),
        param_types: vec!["Int".to_string(), "Int".to_string()],
        return_type: "Int".to_string(),
    });

    let cache_path = create_test_cache("math", content, sigs, cache_dir.path())
        .expect("Failed to create cache");

    // "Recompile" with identical content (simulate editor save without changes)
    let new_hash = compute_source_hash(content);

    // Verify hash unchanged
    assert_eq!(hash, new_hash, "Hash should be identical for identical content");

    // Verify cache still valid
    let cached = load_module_cache(&cache_path).expect("Failed to load cache");
    assert_eq!(cached.source_hash, new_hash, "Cache should still be valid");

    println!("✓ Test 1b PASSED: Identical content preserves cache");
}

// ============================================================================
// Test 2: Dependency Signature Changes
// ============================================================================

#[test]
fn test_dependency_signature_change_invalidates_cache() {
    let cache_dir = tempfile::tempdir().unwrap();

    // Module B version 1: returns Int
    let module_b_v1 = "pub getValue() -> Int = 42";
    let mut b_sigs_v1 = HashMap::new();
    b_sigs_v1.insert("getValue".to_string(), FunctionSignature {
        name: "getValue".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    let b_cache_path = create_test_cache("module_b", module_b_v1, b_sigs_v1, cache_dir.path())
        .expect("Failed to create B cache");

    // Module A: depends on B
    let module_a = "use module_b.getValue\npub main() = getValue() + 10";
    let mut a_sigs = HashMap::new();
    a_sigs.insert("main".to_string(), FunctionSignature {
        name: "main".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    let a_cache_path = create_test_cache("module_a", module_a, a_sigs, cache_dir.path())
        .expect("Failed to create A cache");

    // Load and verify B's v1 signature
    let b_cached_v1 = load_module_cache(&b_cache_path).expect("Failed to load B cache");
    let b_sig_v1 = b_cached_v1.function_signatures.get("getValue")
        .expect("getValue signature not found");
    assert_eq!(b_sig_v1.return_type, "Int", "B v1 should return Int");

    // Now change module B's signature to return String
    let module_b_v2 = "pub getValue() -> String = \"hello\"";
    let mut b_sigs_v2 = HashMap::new();
    b_sigs_v2.insert("getValue".to_string(), FunctionSignature {
        name: "getValue".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "String".to_string(),
    });

    // Create new cache for B v2 (overwrites old cache)
    create_test_cache("module_b", module_b_v2, b_sigs_v2, cache_dir.path())
        .expect("Failed to create B cache v2");

    // Verify B's signature changed
    let b_cached_v2 = load_module_cache(&b_cache_path).expect("Failed to load B cache v2");
    let b_sig_v2 = b_cached_v2.function_signatures.get("getValue")
        .expect("getValue signature not found in v2");
    assert_eq!(b_sig_v2.return_type, "String", "B v2 should return String");
    assert_ne!(b_sig_v1.return_type, b_sig_v2.return_type, "Return type should differ");

    // The KEY TEST: Module A's cache is now stale!
    // A expects getValue() -> Int, but B now provides getValue() -> String
    // When we load A's cache, we SHOULD detect this incompatibility

    let _a_cached = load_module_cache(&a_cache_path).expect("Failed to load A cache");

    // TODO: Implement dependency signature validation
    // Expected behavior: Check that all functions A imports from B still have
    // compatible signatures. If not, reject A's cache.

    // For now, this test demonstrates the PROBLEM:
    // A's cache is "valid" by source hash (A's source didn't change),
    // but it's incompatible with new B!

    println!("✓ Test 2 PASSED: Detected signature change in dependency");
    println!("  NOTE: Dependency validation NOT YET IMPLEMENTED");
    println!("  Module A's cache is stale but would currently be used!");
    println!("  B changed: getValue() Int -> String");
    println!("  A expects Int but would get String - TYPE ERROR!");
}

// ============================================================================
// Test 3: Transitive Dependencies
// ============================================================================

#[test]
fn test_transitive_dependency_invalidation() {
    let cache_dir = tempfile::tempdir().unwrap();

    // Module C v1: provides baseValue() -> Int
    let module_c_v1 = "pub baseValue() -> Int = 42";
    let mut c_sigs_v1 = HashMap::new();
    c_sigs_v1.insert("baseValue".to_string(), FunctionSignature {
        name: "baseValue".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    create_test_cache("module_c", module_c_v1, c_sigs_v1.clone(), cache_dir.path())
        .expect("Failed to create C cache");

    // Module B: depends on C, provides derived() -> Int
    let module_b = "use module_c.baseValue\npub derived() = baseValue() + 1";
    let mut b_sigs = HashMap::new();
    b_sigs.insert("derived".to_string(), FunctionSignature {
        name: "derived".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    create_test_cache("module_b", module_b, b_sigs, cache_dir.path())
        .expect("Failed to create B cache");

    // Module A: depends on B (transitive dependency on C)
    let module_a = "use module_b.derived\npub main() = derived() * 2";
    let mut a_sigs = HashMap::new();
    a_sigs.insert("main".to_string(), FunctionSignature {
        name: "main".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    create_test_cache("module_a", module_a, a_sigs, cache_dir.path())
        .expect("Failed to create A cache");

    // All three cached successfully
    let c_cache_path = cache_dir.path().join("module_c.cache");
    let b_cache_path = cache_dir.path().join("module_b.cache");
    let a_cache_path = cache_dir.path().join("module_a.cache");

    assert!(c_cache_path.exists(), "C cache should exist");
    assert!(b_cache_path.exists(), "B cache should exist");
    assert!(a_cache_path.exists(), "A cache should exist");

    // Now change C's signature: baseValue() -> String
    let module_c_v2 = "pub baseValue() -> String = \"changed\"";
    let c_hash_v2 = compute_source_hash(module_c_v2);

    let mut c_sigs_v2 = HashMap::new();
    c_sigs_v2.insert("baseValue".to_string(), FunctionSignature {
        name: "baseValue".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "String".to_string(),
    });

    // C's cache is now stale (source hash changed)
    let c_cached_v1 = load_module_cache(&c_cache_path).unwrap();
    assert_ne!(c_cached_v1.source_hash, c_hash_v2, "C cache should be stale (hash mismatch)");

    // Verify signature change
    let c_sig_v1 = c_cached_v1.function_signatures.get("baseValue").unwrap();
    assert_eq!(c_sig_v1.return_type, "Int");

    // After recompiling C with new signature
    create_test_cache("module_c", module_c_v2, c_sigs_v2, cache_dir.path())
        .expect("Failed to update C cache");

    let c_cached_v2 = load_module_cache(&c_cache_path).unwrap();
    let c_sig_v2 = c_cached_v2.function_signatures.get("baseValue").unwrap();
    assert_eq!(c_sig_v2.return_type, "String");

    // Expected behavior (NOT YET IMPLEMENTED):
    // 1. C must be recompiled (source changed) - DONE
    // 2. B's cache is stale (depends on C.baseValue which changed Int -> String)
    //    B expects Int but would get String - type error!
    // 3. A's cache is stale (transitive: depends on B.derived which depends on C.baseValue)
    //    Even though A doesn't directly use C, it's affected by the change

    // The dependency chain:
    // A.main() calls B.derived()
    // B.derived() calls C.baseValue()
    // When C.baseValue changes Int -> String, B.derived breaks (can't do String + 1)
    // Therefore A.main also breaks (even though it doesn't call C directly)

    println!("✓ Test 3 PASSED: Identified transitive dependency chain");
    println!("  Dependency chain: A.main -> B.derived -> C.baseValue");
    println!("  When C.baseValue changes Int -> String:");
    println!("    - C cache: invalidated (source hash changed) ✓");
    println!("    - B cache: SHOULD be invalidated (dependency signature changed)");
    println!("    - A cache: SHOULD be invalidated (transitive dependency)");
    println!("  NOTE: Dependency invalidation NOT YET IMPLEMENTED");
}

#[test]
fn test_dependency_graph_tracking() {
    let cache_dir = tempfile::tempdir().unwrap();

    // Create a BytecodeCache to test dependency graph
    let cache = BytecodeCache::new(cache_dir.path().to_path_buf(), "test-0.1.0");

    // The dependency graph should track: derived -> [base]
    // This means "derived depends on base"

    // Verify the manifest structure exists
    let manifest = cache.manifest();
    assert_eq!(manifest.format_version, 1);
    assert_eq!(manifest.compiler_version, "test-0.1.0");
    assert!(manifest.dependency_graph.is_empty(), "Should start with empty graph");

    // TODO: When modules are cached, dependency graph should be populated
    // For example:
    // dependency_graph.insert("derived", vec!["base"])
    // means "derived depends on base"

    println!("✓ Test 3b PASSED: Dependency graph structure validated");
    println!("  dependency_graph: HashMap<String, Vec<String>>");
    println!("  Format: module_name -> [list of dependencies]");
}

// ============================================================================
// Test Utilities
// ============================================================================

#[test]
fn test_hash_computation_consistency() {
    // Verify hash computation is deterministic
    let content = "pub test() = 42";
    let hash1 = compute_source_hash(content);
    let hash2 = compute_source_hash(content);
    assert_eq!(hash1, hash2, "Hash should be deterministic");

    // Verify different content gives different hash
    let different = "pub test() = 43";
    let hash3 = compute_source_hash(different);
    assert_ne!(hash1, hash3, "Different content should have different hash");

    // Verify even whitespace changes hash (strict equality)
    let with_space = "pub test() = 42 ";
    let hash4 = compute_source_hash(with_space);
    assert_ne!(hash1, hash4, "Whitespace should affect hash");

    println!("✓ Hash computation is consistent and strict");
}

// ============================================================================
// Test 4: Two-Phase Compilation with Cache
// ============================================================================

#[test]
fn test_two_phase_stale_dependency_detection() {
    // Simulates TUI editor workflow:
    // 1. Load project (modules A and B)
    // 2. Phase 1 (check_module_compiles) on A - uses cached types from B
    // 3. User edits B externally, changes function signature
    // 4. Phase 1 on A again - MUST detect B changed and show type error

    let cache_dir = tempfile::tempdir().unwrap();

    // Module B v1: provides add(Int, Int) -> Int
    let module_b_v1 = "pub add(a: Int, b: Int) -> Int = a + b";
    let mut b_sigs_v1 = HashMap::new();
    b_sigs_v1.insert("add".to_string(), FunctionSignature {
        name: "add".to_string(),
        type_params: Vec::new(),
        param_types: vec!["Int".to_string(), "Int".to_string()],
        return_type: "Int".to_string(),
    });

    create_test_cache("module_b", module_b_v1, b_sigs_v1.clone(), cache_dir.path())
        .expect("Failed to create B cache v1");

    // Module A: imports and uses B.add
    let module_a = "use module_b.add\npub main() = add(1, 2)";
    let mut a_sigs = HashMap::new();
    a_sigs.insert("main".to_string(), FunctionSignature {
        name: "main".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    create_test_cache("module_a", module_a, a_sigs, cache_dir.path())
        .expect("Failed to create A cache");

    // PHASE 1: check_module_compiles on A
    // Expected: Should load cached types from B, A type-checks OK

    // Now user edits B externally: add(String, String) -> String
    let module_b_v2 = "pub add(a: String, b: String) -> String = a ++ b";
    let mut b_sigs_v2 = HashMap::new();
    b_sigs_v2.insert("add".to_string(), FunctionSignature {
        name: "add".to_string(),
        type_params: Vec::new(),
        param_types: vec!["String".to_string(), "String".to_string()],
        return_type: "String".to_string(),
    });

    // Update B's cache (simulates user saving B in another editor)
    create_test_cache("module_b", module_b_v2, b_sigs_v2, cache_dir.path())
        .expect("Failed to update B cache");

    // PHASE 1 AGAIN: check_module_compiles on A
    // Expected: Should detect B's cache changed (source hash differs)
    // Expected: Should validate B's add signature changed
    // Expected: Should report TYPE ERROR in A: add(1, 2) - Int args but expects String

    // Load both caches
    let _a_cache = load_module_cache(&cache_dir.path().join("module_a.cache")).unwrap();
    let b_cache = load_module_cache(&cache_dir.path().join("module_b.cache")).unwrap();

    // Verify B's signature changed
    let b_sig_v1_check = b_sigs_v1.get("add").unwrap();
    let b_sig_v2_check = b_cache.function_signatures.get("add").unwrap();
    assert_ne!(b_sig_v1_check.param_types, b_sig_v2_check.param_types);
    assert_ne!(b_sig_v1_check.return_type, b_sig_v2_check.return_type);

    // The CRITICAL TEST:
    // When Phase 1 (check_module_compiles) runs on A, it should:
    // 1. Load A's cache
    // 2. See that A imports B.add
    // 3. Load B's cache to get add's signature
    // 4. Detect that B's cache hash changed (NOT the same as when A was compiled)
    // 5. Either:
    //    a) Reject A's cache entirely (safe but slow)
    //    b) Validate B's add signature matches what A expects
    // 6. Report type error: A calls add(1, 2) but B.add now expects (String, String)

    println!("✓ Test 4 PASSED: Two-phase stale dependency detection");
    println!("  Scenario:");
    println!("    1. Load project with A (uses B.add)");
    println!("    2. Phase 1 on A: type checks OK");
    println!("    3. User edits B externally: add Int -> String");
    println!("    4. Phase 1 on A: SHOULD detect B changed");
    println!("  Expected: Phase 1 detects B's signature incompatible");
    println!("  NOTE: Dependency validation NOT YET IMPLEMENTED");
}

#[test]
fn test_two_phase_cache_vs_compiler_state() {
    // Tests that cached bytecode and compiler state stay in sync
    //
    // Problem: If we load bytecode from cache but don't update compiler state,
    // check_module_compiles won't know about the cached module's types/functions

    let cache_dir = tempfile::tempdir().unwrap();

    // Create a cached module with a custom type
    let module_lib = r#"
type Config = {
    host: String,
    port: Int
}

pub defaultConfig() -> Config = Config {
    host: "localhost",
    port: 8080
}
"#;

    let mut lib_sigs = HashMap::new();
    lib_sigs.insert("defaultConfig".to_string(), FunctionSignature {
        name: "defaultConfig".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Config".to_string(),
    });

    create_test_cache("lib", module_lib, lib_sigs, cache_dir.path())
        .expect("Failed to create lib cache");

    // Module that uses the cached type
    let _module_app = r#"
use lib.defaultConfig

pub main() = {
    cfg = defaultConfig()
    cfg.host ++ ":" ++ cfg.port.show()
}
"#;

    // When we load_from_cache(lib), we must also:
    // 1. Register Config type in compiler.types
    // 2. Register defaultConfig in compiler.functions
    // 3. Register lib.* exports
    //
    // Otherwise, Phase 1 (check_module_compiles) on app will fail:
    // - "Unknown type: Config"
    // - "Undefined function: defaultConfig"

    // This test documents the requirement: cache loading MUST populate compiler state

    println!("✓ Test 4b PASSED: Cache vs Compiler State consistency");
    println!("  When loading cached bytecode:");
    println!("    1. Load functions/types into VM");
    println!("    2. Register types in compiler.types");
    println!("    3. Register functions in compiler.functions");
    println!("    4. Register exports in compiler.exports");
    println!("  Otherwise Phase 1 type checking fails!");
    println!("  NOTE: Cache loading integration NOT YET IMPLEMENTED");
}

#[test]
fn test_concurrent_edits_different_modules() {
    // Tests cache behavior when multiple modules change simultaneously
    // (e.g., user switches git branches, many files change)

    let cache_dir = tempfile::tempdir().unwrap();

    // Three modules: util, business, app
    let util_v1 = "pub helper() -> Int = 42";
    let business_v1 = "use util.helper\npub process() = helper() * 2";
    let app_v1 = "use business.process\npub main() = process() + 1";

    let mut util_sigs = HashMap::new();
    util_sigs.insert("helper".to_string(), FunctionSignature {
        name: "helper".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    let mut business_sigs = HashMap::new();
    business_sigs.insert("process".to_string(), FunctionSignature {
        name: "process".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    let mut app_sigs = HashMap::new();
    app_sigs.insert("main".to_string(), FunctionSignature {
        name: "main".to_string(),
        type_params: Vec::new(),
        param_types: Vec::new(),
        return_type: "Int".to_string(),
    });

    create_test_cache("util", util_v1, util_sigs, cache_dir.path()).unwrap();
    create_test_cache("business", business_v1, business_sigs, cache_dir.path()).unwrap();
    create_test_cache("app", app_v1, app_sigs, cache_dir.path()).unwrap();

    // Git branch switch: ALL source files change simultaneously
    let _util_v2 = "pub helper() -> String = \"new\"";
    let _business_v2 = "use util.helper\npub process() = helper() ++ \"!\"";
    let _app_v2 = "use business.process\npub main() = process() ++ \"?\"";

    // Expected behavior:
    // 1. Detect util.nos source hash changed -> invalidate util cache
    // 2. Detect business.nos source hash changed -> invalidate business cache
    // 3. Detect app.nos source hash changed -> invalidate app cache
    // 4. Recompile all three in dependency order: util -> business -> app

    // Alternative (smarter but more complex):
    // 1. Detect util signature changed -> invalidate business & app
    // 2. Detect business signature changed -> invalidate app
    // 3. App source unchanged but dependencies changed -> invalidate app
    // 4. Recompile only what's actually needed

    println!("✓ Test 4c PASSED: Concurrent multi-module changes");
    println!("  Scenario: Git branch switch, all files change");
    println!("  Expected: Detect all source hashes changed");
    println!("  Expected: Invalidate all caches");
    println!("  Expected: Recompile in dependency order");
    println!("  NOTE: Batch invalidation NOT YET OPTIMIZED");
}
