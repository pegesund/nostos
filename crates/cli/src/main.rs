//! Nostos CLI - Command-line interface for running Nostos programs.
//!
//! Note: We use the system allocator (not mimalloc) to ensure compatibility
//! with native extensions, which also use the system allocator. Using different
//! allocators across DLL boundaries causes crashes when memory is allocated
//! by one allocator and freed by another.

mod tui;
mod editor;
mod custom_views;
mod repl_panel;
mod autocomplete;
mod inspector_panel;
mod nostos_panel;
mod debug_panel;
mod git_panel;
mod tutorial;
mod packages;
mod server;
mod connect;

use nostos_compiler::compile::{Compiler, MvarInitValue};
use nostos_jit::{JitCompiler, JitConfig};
use nostos_syntax::{parse, parse_errors_to_source_errors, eprint_errors};
use nostos_vm::async_vm::{AsyncVM, AsyncConfig};
use nostos_vm::cache::{BytecodeCache, CachedModule, CachedMvar, CachedMvarValue, function_to_cached_with_fn_list, compute_file_hash};
use nostos_vm::process::ThreadSafeValue;
use nostos_vm::value::RuntimeError;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

/// Recursively visit directories and find .nos files
fn visit_dirs(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                // Skip .nostos directory - it contains per-definition files for REPL/TUI
                if let Some(name) = path.file_name() {
                    if name == ".nostos" {
                        continue;
                    }
                }
                visit_dirs(&path, files)?;
            } else {
                if let Some(ext) = path.extension() {
                    if ext == "nos" {
                        files.push(path);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Get the path to the bytecode cache directory
fn get_cache_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".nostos").join("cache")
    } else {
        PathBuf::from(".nostos-cache")
    }
}

/// Get the cache directory for extension modules
fn get_extension_cache_dir(ext_name: &str) -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".nostos").join("cache").join("extensions").join(ext_name)
    } else {
        PathBuf::from(".nostos-cache").join("extensions").join(ext_name)
    }
}

/// Result of extension cache loading
struct ExtensionCacheLoadResult {
    functions_loaded: usize,
}

/// Try to load an extension module from cache.
/// Returns Some if cache is valid and was loaded successfully.
fn try_load_extension_from_cache(
    compiler: &mut Compiler,
    ext_name: &str,
    ext_dir: &std::path::Path,
) -> Option<ExtensionCacheLoadResult> {
    use nostos_vm::cache::{CachedFunction, cached_to_function, cached_to_function_with_resolver};
    use nostos_vm::value::Value;

    let cache_dir = get_extension_cache_dir(ext_name);

    // Check if cache exists
    if !cache_dir.exists() {
        return None;
    }

    // Load cache manager
    let cache = BytecodeCache::new(cache_dir.clone(), env!("CARGO_PKG_VERSION"));

    if !cache.has_cache() {
        return None;
    }

    // List extension wrapper files (excluding test files)
    let files = match nostos_packages::PackageManager::list_extension_wrapper_files(ext_dir) {
        Ok(f) => f,
        Err(_) => return None,
    };

    if files.is_empty() {
        return None;
    }

    // Check if all modules have valid cache
    let mut modules_to_load: Vec<(String, PathBuf)> = Vec::new();

    for file_path in &files {
        // Module name is ext_name (e.g., "glam")
        let module_name = ext_name.to_string();

        if !cache.is_module_valid(&module_name, file_path) {
            return None;
        }

        modules_to_load.push((module_name, file_path.clone()));
    }

    // All modules have valid cache - load them
    let mut all_cached_functions: Vec<CachedFunction> = Vec::new();
    let mut all_types: Vec<nostos_vm::value::TypeValue> = Vec::new();
    let mut all_mvars: Vec<CachedMvar> = Vec::new();

    for (module_name, _file_path) in &modules_to_load {
        match cache.load_module(module_name) {
            Ok(cached_module) => {
                all_cached_functions.extend(cached_module.functions);
                all_types.extend(cached_module.types);
                all_mvars.extend(cached_module.mvars);
            }
            Err(_) => {
                return None;
            }
        }
    }

    // Register the module so that `use ext_name.*` works
    compiler.register_known_module(ext_name);

    // Build a map of function names to their cached data for resolution
    let cached_fn_map: std::collections::HashMap<String, &CachedFunction> = all_cached_functions
        .iter()
        .map(|f| (f.name.clone(), f))
        .collect();

    // Convert functions with resolver and register them
    let mut loaded_count = 0;
    for cached_fn in &all_cached_functions {
        let func = cached_to_function_with_resolver(cached_fn, |name| {
            // First check if already registered
            if let Some(existing) = compiler.get_function(name) {
                return Some(Value::Function(existing));
            }
            // Otherwise, look up in cached functions and convert (without resolver - base case)
            if let Some(cached) = cached_fn_map.get(name) {
                let converted = cached_to_function(cached);
                Some(Value::Function(std::sync::Arc::new(converted)))
            } else {
                None
            }
        });
        compiler.register_external_function(&cached_fn.name, std::sync::Arc::new(func));
        // Also register visibility so `use ext_name.*` works
        compiler.register_function_visibility(&cached_fn.name, nostos_syntax::ast::Visibility::Public);
        loaded_count += 1;
    }

    // Register types
    for type_val in &all_types {
        compiler.register_external_type(&type_val.name, &std::sync::Arc::new(type_val.clone()));
    }

    // Register mvars from cache
    for cached_mvar in &all_mvars {
        let initial_value = cached_to_mvar_init(&cached_mvar.initial_value);
        compiler.register_mvar_with_info(
            &cached_mvar.name,
            cached_mvar.type_name.clone(),
            initial_value,
        );
    }

    if loaded_count > 0 {
        Some(ExtensionCacheLoadResult {
            functions_loaded: loaded_count,
        })
    } else {
        None
    }
}

/// Save an extension module to cache after compiling
fn save_extension_to_cache(
    compiler: &Compiler,
    ext_name: &str,
    source_path: &std::path::Path,
    source_hash: &str,
) -> Result<(), String> {
    let cache_dir = get_extension_cache_dir(ext_name);

    // Create cache directory
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create extension cache directory: {}", e))?;

    // Initialize cache manager
    let mut cache = BytecodeCache::new(cache_dir, env!("CARGO_PKG_VERSION"));

    // Get the function list for CallDirect → CallByName conversion
    let function_list = compiler.get_function_list_names();

    // Get all functions, types, and mvars from compiler
    let functions = compiler.get_all_functions();
    let all_types = compiler.get_all_types();
    let all_mvars = compiler.get_mvars();

    // Module prefix for filtering
    let module_prefix = format!("{}.", ext_name);

    // Collect functions for this extension module
    let mut cached_functions = Vec::new();
    for (func_name, func) in functions {
        if func_name.starts_with(&module_prefix) || func.module.as_deref() == Some(ext_name) {
            if let Some(cached) = function_to_cached_with_fn_list(func, function_list) {
                cached_functions.push(cached);
            }
        }
    }

    // Collect types for this module
    let module_types: Vec<nostos_vm::value::TypeValue> = all_types.iter()
        .filter(|(type_name, _)| type_name.starts_with(&module_prefix))
        .map(|(_, type_val)| (**type_val).clone())
        .collect();

    // Collect mvars for this module
    let module_mvars: Vec<CachedMvar> = all_mvars.iter()
        .filter(|(mvar_name, _)| mvar_name.starts_with(&module_prefix))
        .map(|(name, info)| CachedMvar {
            name: name.clone(),
            type_name: info.type_name.clone(),
            initial_value: mvar_init_to_cached(&info.initial_value),
        })
        .collect();

    let cached_module = CachedModule {
        module_path: vec![ext_name.to_string()],
        source_hash: source_hash.to_string(),
        functions: cached_functions,
        function_signatures: std::collections::HashMap::new(),
        exports: Vec::new(),
        prelude_imports: Vec::new(),
        types: module_types,
        mvars: module_mvars,
        dependency_signatures: std::collections::HashMap::new(),
    };

    // Save the module
    cache.save_module(ext_name, source_path.to_str().unwrap_or("unknown"), &cached_module)?;

    // Save manifest
    cache.save_manifest()?;

    Ok(())
}

/// Get the cache directory for source package modules
fn get_package_cache_dir(pkg_name: &str) -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".nostos").join("cache").join("packages").join(pkg_name)
    } else {
        PathBuf::from(".nostos-cache").join("packages").join(pkg_name)
    }
}

/// Result of package cache loading
struct PackageCacheLoadResult {
    modules_loaded: usize,
    functions_loaded: usize,
}

/// Try to load a source package from cache.
/// Returns Some if cache is valid for all modules and was loaded successfully.
fn try_load_package_from_cache(
    compiler: &mut Compiler,
    pkg_name: &str,
    package_path: &std::path::Path,
) -> Option<PackageCacheLoadResult> {
    use nostos_vm::cache::{CachedFunction, cached_to_function, cached_to_function_with_resolver};
    use nostos_vm::value::Value;

    let cache_dir = get_package_cache_dir(pkg_name);

    // Check if cache exists
    if !cache_dir.exists() {
        return None;
    }

    // Load cache manager
    let cache = BytecodeCache::new(cache_dir.clone(), env!("CARGO_PKG_VERSION"));

    if !cache.has_cache() {
        return None;
    }

    // List package files
    let files = match nostos_packages::PackageManager::list_package_files(package_path) {
        Ok(f) => f,
        Err(_) => return None,
    };

    if files.is_empty() {
        return None;
    }

    // Check if all modules have valid cache
    let mut modules_to_load: Vec<(String, PathBuf)> = Vec::new();

    for file_path in &files {
        let file_stem = file_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let module_name = format!("{}.{}", pkg_name, file_stem);

        if !cache.is_module_valid(&module_name, file_path) {
            return None;
        }

        modules_to_load.push((module_name, file_path.clone()));
    }

    // All modules have valid cache - load them
    let mut all_cached_functions: Vec<CachedFunction> = Vec::new();
    let mut all_types: Vec<nostos_vm::value::TypeValue> = Vec::new();
    let mut all_mvars: Vec<CachedMvar> = Vec::new();
    let mut modules_loaded = 0;

    for (module_name, _file_path) in &modules_to_load {
        match cache.load_module(module_name) {
            Ok(cached_module) => {
                all_cached_functions.extend(cached_module.functions);
                all_types.extend(cached_module.types);
                all_mvars.extend(cached_module.mvars);
                modules_loaded += 1;
            }
            Err(_) => {
                return None;
            }
        }
    }

    // Register the package and all module names so imports work
    compiler.register_known_module(pkg_name);
    for (module_name, _) in &modules_to_load {
        compiler.register_known_module(module_name);
    }

    // Build a map of function names to their cached data for resolution
    let cached_fn_map: std::collections::HashMap<String, &CachedFunction> = all_cached_functions
        .iter()
        .map(|f| (f.name.clone(), f))
        .collect();

    // Convert functions with resolver and register them
    let mut functions_loaded = 0;
    for cached_fn in &all_cached_functions {
        let func = cached_to_function_with_resolver(cached_fn, |name| {
            // First check if already registered
            if let Some(existing) = compiler.get_function(name) {
                return Some(Value::Function(existing));
            }
            // Otherwise, look up in cached functions and convert (without resolver - base case)
            if let Some(cached) = cached_fn_map.get(name) {
                let converted = cached_to_function(cached);
                Some(Value::Function(std::sync::Arc::new(converted)))
            } else {
                None
            }
        });
        compiler.register_external_function(&cached_fn.name, std::sync::Arc::new(func));
        // Also register visibility so `use pkg.*` works
        compiler.register_function_visibility(&cached_fn.name, nostos_syntax::ast::Visibility::Public);
        functions_loaded += 1;
    }

    // Register types
    for type_val in &all_types {
        compiler.register_external_type(&type_val.name, &std::sync::Arc::new(type_val.clone()));
    }

    // Register mvars from cache
    for cached_mvar in &all_mvars {
        let initial_value = cached_to_mvar_init(&cached_mvar.initial_value);
        compiler.register_mvar_with_info(
            &cached_mvar.name,
            cached_mvar.type_name.clone(),
            initial_value,
        );
    }

    if functions_loaded > 0 || modules_loaded > 0 {
        Some(PackageCacheLoadResult {
            modules_loaded,
            functions_loaded,
        })
    } else {
        None
    }
}

/// Save a source package to cache after compiling.
/// Saves each module in the package separately.
fn save_package_to_cache(
    compiler: &Compiler,
    pkg_name: &str,
    module_infos: &[(String, PathBuf, String)], // (module_name, source_path, source_hash)
) -> Result<(), String> {
    let cache_dir = get_package_cache_dir(pkg_name);

    // Create cache directory
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create package cache directory: {}", e))?;

    // Initialize cache manager
    let mut cache = BytecodeCache::new(cache_dir, env!("CARGO_PKG_VERSION"));

    // Get the function list for CallDirect → CallByName conversion
    let function_list = compiler.get_function_list_names();

    // Get all functions, types, and mvars from compiler
    let functions = compiler.get_all_functions();
    let all_types = compiler.get_all_types();
    let all_mvars = compiler.get_mvars();

    // Save each module separately
    for (module_name, source_path, source_hash) in module_infos {
        let module_prefix = format!("{}.", module_name);

        // Collect functions for this module
        let mut cached_functions = Vec::new();
        for (func_name, func) in functions {
            if func_name.starts_with(&module_prefix) || func.module.as_deref() == Some(module_name.as_str()) {
                if let Some(cached) = function_to_cached_with_fn_list(func, function_list) {
                    cached_functions.push(cached);
                }
            }
        }

        // Collect types for this module
        let module_types: Vec<nostos_vm::value::TypeValue> = all_types.iter()
            .filter(|(type_name, _)| type_name.starts_with(&module_prefix))
            .map(|(_, type_val)| (**type_val).clone())
            .collect();

        // Collect mvars for this module
        let module_mvars: Vec<CachedMvar> = all_mvars.iter()
            .filter(|(mvar_name, _)| mvar_name.starts_with(&module_prefix))
            .map(|(name, info)| CachedMvar {
                name: name.clone(),
                type_name: info.type_name.clone(),
                initial_value: mvar_init_to_cached(&info.initial_value),
            })
            .collect();

        let cached_module = CachedModule {
            module_path: module_name.split('.').map(|s| s.to_string()).collect(),
            source_hash: source_hash.clone(),
            functions: cached_functions,
            function_signatures: std::collections::HashMap::new(),
            exports: Vec::new(),
            prelude_imports: Vec::new(),
            types: module_types,
            mvars: module_mvars,
            dependency_signatures: std::collections::HashMap::new(),
        };

        // Save the module
        cache.save_module(module_name, source_path.to_str().unwrap_or("unknown"), &cached_module)?;
    }

    // Save manifest
    cache.save_manifest()?;

    Ok(())
}

/// Get the project-local cache directory
fn get_project_cache_dir(project_root: &std::path::Path) -> PathBuf {
    project_root.join(".nostos-cache")
}

/// Result of project cache loading
struct ProjectCacheLoadResult {
    modules_loaded: usize,
    functions_loaded: usize,
}

/// Try to load all project modules from cache.
/// Returns Some only if ALL modules have valid cache (conservative approach).
/// This avoids complex dependency tracking between project modules.
fn try_load_project_from_cache(
    compiler: &mut Compiler,
    project_root: &std::path::Path,
    source_files: &[PathBuf],
) -> Option<ProjectCacheLoadResult> {
    use nostos_vm::cache::{CachedFunction, cached_to_function, cached_to_function_with_resolver};
    use nostos_vm::value::Value;

    // Only cache for directory projects, not single files
    if source_files.len() <= 1 {
        return None;
    }

    let cache_dir = get_project_cache_dir(project_root);

    // Check if cache exists
    if !cache_dir.exists() {
        return None;
    }

    // Load cache manager
    let cache = BytecodeCache::new(cache_dir.clone(), env!("CARGO_PKG_VERSION"));

    if !cache.has_cache() {
        return None;
    }

    // Check if ALL modules have valid cache
    let mut modules_to_load: Vec<(String, PathBuf)> = Vec::new();

    for file_path in source_files {
        let relative = match file_path.strip_prefix(project_root) {
            Ok(r) => r,
            Err(_) => return None,
        };

        let mut components: Vec<String> = relative.components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();

        // Remove .nos extension from last component
        if let Some(last) = components.last_mut() {
            if last.ends_with(".nos") {
                *last = last.trim_end_matches(".nos").to_string();
            }
        }

        let module_name = components.join(".");

        // If any module is invalid, don't use cache at all
        if !cache.is_module_valid(&module_name, file_path) {
            return None;
        }

        modules_to_load.push((module_name, file_path.clone()));
    }

    // All modules have valid cache - load them
    let mut all_cached_functions: Vec<CachedFunction> = Vec::new();
    let mut all_types: Vec<nostos_vm::value::TypeValue> = Vec::new();
    let mut all_mvars: Vec<CachedMvar> = Vec::new();
    let mut modules_loaded = 0;

    for (module_name, _file_path) in &modules_to_load {
        match cache.load_module(module_name) {
            Ok(cached_module) => {
                all_cached_functions.extend(cached_module.functions);
                all_types.extend(cached_module.types);
                all_mvars.extend(cached_module.mvars);
                modules_loaded += 1;
            }
            Err(_) => {
                return None;
            }
        }
    }

    // Register all module names
    for (module_name, _) in &modules_to_load {
        if !module_name.is_empty() {
            compiler.register_known_module(module_name);
        }
    }

    // Build a map of function names to their cached data for resolution
    let cached_fn_map: std::collections::HashMap<String, &CachedFunction> = all_cached_functions
        .iter()
        .map(|f| (f.name.clone(), f))
        .collect();

    // Convert functions with resolver and register them
    let mut functions_loaded = 0;
    for cached_fn in &all_cached_functions {
        let func = cached_to_function_with_resolver(cached_fn, |name| {
            // First check if already registered
            if let Some(existing) = compiler.get_function(name) {
                return Some(Value::Function(existing));
            }
            // Otherwise, look up in cached functions and convert (without resolver - base case)
            if let Some(cached) = cached_fn_map.get(name) {
                let converted = cached_to_function(cached);
                Some(Value::Function(std::sync::Arc::new(converted)))
            } else {
                None
            }
        });
        compiler.register_external_function(&cached_fn.name, std::sync::Arc::new(func));
        // Register visibility for project functions (they're implicitly public within the project)
        compiler.register_function_visibility(&cached_fn.name, nostos_syntax::ast::Visibility::Public);
        functions_loaded += 1;
    }

    // Register types
    for type_val in &all_types {
        compiler.register_external_type(&type_val.name, &std::sync::Arc::new(type_val.clone()));
    }

    // Register mvars from cache
    for cached_mvar in &all_mvars {
        let initial_value = cached_to_mvar_init(&cached_mvar.initial_value);
        compiler.register_mvar_with_info(
            &cached_mvar.name,
            cached_mvar.type_name.clone(),
            initial_value,
        );
    }

    if functions_loaded > 0 || modules_loaded > 0 {
        Some(ProjectCacheLoadResult {
            modules_loaded,
            functions_loaded,
        })
    } else {
        None
    }
}

/// Save all project modules to cache after compiling.
fn save_project_to_cache(
    compiler: &Compiler,
    project_root: &std::path::Path,
    module_infos: &[(String, PathBuf, String)], // (module_name, source_path, source_hash)
) -> Result<(), String> {
    let cache_dir = get_project_cache_dir(project_root);

    // Create cache directory
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create project cache directory: {}", e))?;

    // Initialize cache manager
    let mut cache = BytecodeCache::new(cache_dir, env!("CARGO_PKG_VERSION"));

    // Get the function list for CallDirect → CallByName conversion
    let function_list = compiler.get_function_list_names();

    // Get all functions, types, and mvars from compiler
    let functions = compiler.get_all_functions();
    let all_types = compiler.get_all_types();
    let all_mvars = compiler.get_mvars();

    // Save each module separately
    for (module_name, source_path, source_hash) in module_infos {
        // For top-level module (empty name), use special handling
        let module_prefix = if module_name.is_empty() {
            String::new()
        } else {
            format!("{}.", module_name)
        };

        // Collect functions for this module
        let mut cached_functions = Vec::new();
        for (func_name, func) in functions {
            let belongs_to_module = if module_name.is_empty() {
                // Top-level functions don't have a module prefix
                !func_name.contains('.') || func.module.is_none()
            } else {
                func_name.starts_with(&module_prefix) || func.module.as_deref() == Some(module_name.as_str())
            };

            if belongs_to_module {
                if let Some(cached) = function_to_cached_with_fn_list(func, function_list) {
                    cached_functions.push(cached);
                }
            }
        }

        // Collect types for this module
        let module_types: Vec<nostos_vm::value::TypeValue> = all_types.iter()
            .filter(|(type_name, _)| {
                if module_name.is_empty() {
                    !type_name.contains('.')
                } else {
                    type_name.starts_with(&module_prefix)
                }
            })
            .map(|(_, type_val)| (**type_val).clone())
            .collect();

        // Collect mvars for this module
        let module_mvars: Vec<CachedMvar> = all_mvars.iter()
            .filter(|(mvar_name, _)| {
                if module_name.is_empty() {
                    !mvar_name.contains('.')
                } else {
                    mvar_name.starts_with(&module_prefix)
                }
            })
            .map(|(name, info)| CachedMvar {
                name: name.clone(),
                type_name: info.type_name.clone(),
                initial_value: mvar_init_to_cached(&info.initial_value),
            })
            .collect();

        // Use a special name for top-level module
        let cache_module_name = if module_name.is_empty() {
            "__main__".to_string()
        } else {
            module_name.clone()
        };

        let cached_module = CachedModule {
            module_path: if module_name.is_empty() {
                vec![]
            } else {
                module_name.split('.').map(|s| s.to_string()).collect()
            },
            source_hash: source_hash.clone(),
            functions: cached_functions,
            function_signatures: std::collections::HashMap::new(),
            exports: Vec::new(),
            prelude_imports: Vec::new(),
            types: module_types,
            mvars: module_mvars,
            dependency_signatures: std::collections::HashMap::new(),
        };

        // Save the module
        cache.save_module(&cache_module_name, source_path.to_str().unwrap_or("unknown"), &cached_module)?;
    }

    // Save manifest
    cache.save_manifest()?;

    Ok(())
}

/// Find the stdlib directory
fn find_stdlib_path() -> Option<PathBuf> {
    let candidates = vec![
        PathBuf::from("stdlib"),
        PathBuf::from("../stdlib"),
    ];

    for path in candidates {
        if path.is_dir() {
            return Some(path);
        }
    }

    // Try relative to executable
    if let Ok(mut p) = std::env::current_exe() {
        p.pop(); // remove binary name
        p.pop(); // remove release/debug
        p.pop(); // remove target
        p.push("stdlib");
        if p.is_dir() {
            return Some(p);
        }
    }

    // Try home directory locations
    if let Some(home) = dirs::home_dir() {
        let home_stdlib = home.join(".nostos").join("stdlib");
        if home_stdlib.is_dir() {
            return Some(home_stdlib);
        }
    }

    None
}

/// Try to load stdlib from cache. Returns true if cache was valid and loaded.
/// Result of cache loading: prelude imports
struct CacheLoadResult {
    prelude_imports: Vec<(String, String)>,
}

fn try_load_stdlib_from_cache(
    compiler: &mut Compiler,
    stdlib_path: &std::path::Path,
) -> Option<CacheLoadResult> {
    use nostos_vm::cache::{CachedFunction, cached_to_function, cached_to_function_with_resolver};
    use nostos_vm::value::Value;

    let cache_dir = get_cache_dir();

    // Check if cache exists
    if !cache_dir.exists() {
        return None;
    }

    // Load cache manager
    let cache = BytecodeCache::new(cache_dir.clone(), env!("CARGO_PKG_VERSION"));

    if !cache.has_cache() {
        return None;
    }

    // Collect stdlib files to check cache validity
    let mut stdlib_files = Vec::new();
    if visit_dirs(stdlib_path, &mut stdlib_files).is_err() {
        return None;
    }

    // Check if all modules have valid cache
    let mut all_valid = true;
    let mut modules_to_load: Vec<(String, PathBuf)> = Vec::new();

    for file_path in &stdlib_files {
        let relative = match file_path.strip_prefix(stdlib_path) {
            Ok(r) => r,
            Err(_) => return None,
        };

        let mut components: Vec<String> = vec!["stdlib".to_string()];
        for component in relative.components() {
            let s = component.as_os_str().to_string_lossy().to_string();
            if s.ends_with(".nos") {
                components.push(s.trim_end_matches(".nos").to_string());
            } else {
                components.push(s);
            }
        }
        let module_name = components.join(".");

        if !cache.is_module_valid(&module_name, file_path) {
            all_valid = false;
            break;
        }

        modules_to_load.push((module_name, file_path.clone()));
    }

    if !all_valid {
        return None;
    }

    // All modules have valid cache - load them
    let mut all_cached_functions: Vec<CachedFunction> = Vec::new();
    let mut all_prelude_imports: Vec<(String, String)> = Vec::new();
    let mut all_types: Vec<nostos_vm::value::TypeValue> = Vec::new();
    let mut all_mvars: Vec<CachedMvar> = Vec::new();
    for (module_name, _file_path) in &modules_to_load {
        match cache.load_module(module_name) {
            Ok(cached_module) => {
                all_cached_functions.extend(cached_module.functions);
                all_prelude_imports.extend(cached_module.prelude_imports);
                all_types.extend(cached_module.types);
                all_mvars.extend(cached_module.mvars);
            }
            Err(_) => {
                // Cache file couldn't be loaded - invalidate
                return None;
            }
        }
    }

    // Build a map of function names to their cached data for resolution
    let cached_fn_map: std::collections::HashMap<String, &CachedFunction> = all_cached_functions
        .iter()
        .map(|f| (f.name.clone(), f))
        .collect();

    // Convert functions with resolver and register them
    let mut loaded_count = 0;
    for cached_fn in &all_cached_functions {
        let func = cached_to_function_with_resolver(cached_fn, |name| {
            // First check if already registered
            if let Some(existing) = compiler.get_function(name) {
                return Some(Value::Function(existing));
            }
            // Otherwise, look up in cached functions and convert (without resolver - base case)
            if let Some(cached) = cached_fn_map.get(name) {
                let converted = cached_to_function(cached);
                Some(Value::Function(std::sync::Arc::new(converted)))
            } else {
                None
            }
        });
        compiler.register_external_function(&cached_fn.name, std::sync::Arc::new(func));

        // Register visibility for use statement wildcard imports
        let visibility = if cached_fn.is_public {
            nostos_syntax::ast::Visibility::Public
        } else {
            nostos_syntax::ast::Visibility::Private
        };
        compiler.register_function_visibility(&cached_fn.name, visibility);

        loaded_count += 1;
    }

    // Register types
    for type_val in &all_types {
        compiler.register_external_type(&type_val.name, &std::sync::Arc::new(type_val.clone()));
    }

    // Register mvars from cache
    for cached_mvar in &all_mvars {
        let initial_value = cached_to_mvar_init(&cached_mvar.initial_value);
        compiler.register_mvar_with_info(
            &cached_mvar.name,
            cached_mvar.type_name.clone(),
            initial_value,
        );
    }

    // Parse stdlib files that have default parameters or generic functions in parallel using threads
    // Generic functions need ASTs for monomorphization at call sites
    let modules_with_defaults = ["stdlib.html", "stdlib.rhtml", "stdlib.rweb", "stdlib.server", "stdlib.html_parser", "stdlib.db"];
    let modules_to_parse: Vec<_> = modules_to_load.iter()
        .filter(|(name, _)| modules_with_defaults.iter().any(|m| name.starts_with(m)))
        .map(|(name, path)| (name.clone(), path.clone()))
        .collect();

    // Parse files in parallel using threads
    let handles: Vec<_> = modules_to_parse.into_iter()
        .map(|(module_name, file_path)| {
            std::thread::spawn(move || {
                fs::read_to_string(&file_path).ok().and_then(|source| {
                    let (module_opt, _) = parse(&source);
                    module_opt.map(|m| (module_name, m))
                })
            })
        })
        .collect();

    // Collect results and register ASTs
    for handle in handles {
        if let Ok(Some((module_name, module))) = handle.join() {
            for item in &module.items {
                if let nostos_syntax::ast::Item::FnDef(fn_def) = item {
                    compiler.register_external_fn_ast(&module_name, fn_def);
                }
            }
        }
    }

    if loaded_count > 0 {
        Some(CacheLoadResult {
            prelude_imports: all_prelude_imports,
        })
    } else {
        None
    }
}

/// Build and save the stdlib bytecode cache
fn build_stdlib_cache() -> ExitCode {
    println!("Building stdlib bytecode cache...");

    let stdlib_path = match find_stdlib_path() {
        Some(p) => p,
        None => {
            eprintln!("Error: Could not find stdlib directory");
            return ExitCode::FAILURE;
        }
    };

    println!("Found stdlib at: {}", stdlib_path.display());

    // Collect all stdlib files
    let mut stdlib_files = Vec::new();
    if let Err(e) = visit_dirs(&stdlib_path, &mut stdlib_files) {
        eprintln!("Error scanning stdlib: {}", e);
        return ExitCode::FAILURE;
    }

    println!("Found {} stdlib files", stdlib_files.len());

    // Initialize compiler
    let mut compiler = Compiler::new_empty();

    // Read CORE_MODULES to determine which modules should be auto-imported
    let core_modules_path = stdlib_path.join("CORE_MODULES");
    let core_modules: std::collections::HashSet<String> = if core_modules_path.exists() {
        match fs::read_to_string(&core_modules_path) {
            Ok(content) => content
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(|s| s.to_string())
                .collect(),
            Err(e) => {
                eprintln!("Warning: Could not read CORE_MODULES file: {}", e);
                eprintln!("All stdlib modules will be auto-imported");
                std::collections::HashSet::new()
            }
        }
    } else {
        eprintln!("Warning: CORE_MODULES file not found, all stdlib modules will be auto-imported");
        std::collections::HashSet::new()
    };

    if !core_modules.is_empty() {
        println!("Core modules (auto-imported): {}", core_modules.iter().cloned().collect::<Vec<_>>().join(", "));
    }

    // Track module info for cache
    let mut modules: Vec<(String, String, PathBuf, Vec<(String, String)>, std::collections::HashMap<String, bool>)> = Vec::new(); // (module_name, source_hash, path, prelude_imports, function_visibility)

    // Track function names for prelude imports (same as main execution path)
    let mut stdlib_functions: Vec<(String, String)> = Vec::new();

    // Parse and add all stdlib modules
    for file_path in &stdlib_files {
        let source = match fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", file_path.display(), e);
                return ExitCode::FAILURE;
            }
        };

        let (module_opt, errors) = parse(&source);
        if !errors.is_empty() {
            eprintln!("Parse errors in {}", file_path.display());
            return ExitCode::FAILURE;
        }

        let module = match module_opt {
            Some(m) => m,
            None => continue,
        };

        // Build module path
        let relative = file_path.strip_prefix(&stdlib_path).unwrap();
        let mut components: Vec<String> = vec!["stdlib".to_string()];
        for component in relative.components() {
            let s = component.as_os_str().to_string_lossy().to_string();
            if s.ends_with(".nos") {
                components.push(s.trim_end_matches(".nos").to_string());
            } else {
                components.push(s);
            }
        }
        let module_name = components.join(".");

        // Compute source hash
        let source_hash = match compute_file_hash(file_path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Error hashing {}: {}", file_path.display(), e);
                return ExitCode::FAILURE;
            }
        };

        // Determine if this module should be auto-imported (is in core modules list)
        // Extract the module name without "stdlib." prefix
        let module_short_name = module_name.strip_prefix("stdlib.").unwrap_or(&module_name);
        let is_core_module = core_modules.is_empty() || core_modules.contains(module_short_name);

        // Collect function and type names
        // Note: ALL stdlib functions are added to stdlib_functions for compilation (so they can reference each other)
        // But only CORE module imports are saved to module_prelude_imports (for auto-import)
        let mut module_prelude_imports = Vec::new();
        let mut function_visibility = std::collections::HashMap::new();
        for item in &module.items {
            match item {
                nostos_syntax::ast::Item::FnDef(fn_def) => {
                    let local_name = fn_def.name.node.clone();
                    let qualified_name = format!("{}.{}", module_name, local_name);
                    let is_public = matches!(fn_def.visibility, nostos_syntax::ast::Visibility::Public);

                    // Track visibility for later use in cache
                    function_visibility.insert(qualified_name.clone(), is_public);

                    // All functions added for compilation (so stdlib functions can call each other)
                    stdlib_functions.push((local_name.clone(), qualified_name.clone()));

                    // Only core modules added to prelude (for auto-import to user code)
                    if is_core_module {
                        module_prelude_imports.push((local_name, qualified_name));
                    }
                }
                nostos_syntax::ast::Item::TypeDef(type_def) => {
                    if matches!(type_def.visibility, nostos_syntax::ast::Visibility::Public) {
                        let local_name = type_def.name.node.clone();
                        let qualified_name = format!("{}.{}", module_name, local_name);

                        // All types added for compilation
                        stdlib_functions.push((local_name.clone(), qualified_name.clone()));

                        // Only core modules added to prelude
                        if is_core_module {
                            module_prelude_imports.push((local_name, qualified_name));
                        }
                    }
                }
                _ => {}
            }
        }

        modules.push((module_name, source_hash, file_path.clone(), module_prelude_imports, function_visibility));

        if let Err(e) = compiler.add_module(&module, components, std::sync::Arc::new(source), file_path.to_str().unwrap().to_string()) {
            eprintln!("Error adding module {}: {}", file_path.display(), e);
            return ExitCode::FAILURE;
        }
    }

    // Register prelude imports so stdlib functions can reference each other
    for (local_name, qualified_name) in stdlib_functions {
        compiler.add_prelude_import(local_name, qualified_name);
    }

    // Compile all functions
    println!("Compiling stdlib functions...");
    if let Err((e, filename, source)) = compiler.compile_all() {
        let source_error = e.to_source_error();
        source_error.eprint(&filename, &source);
        return ExitCode::FAILURE;
    }

    // Get all compiled functions
    let functions = compiler.get_all_functions();
    println!("Compiled {} functions", functions.len());

    // Create cache directory
    let cache_dir = get_cache_dir();
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        eprintln!("Error creating cache directory: {}", e);
        return ExitCode::FAILURE;
    }

    // Initialize cache manager
    let mut cache = BytecodeCache::new(cache_dir.clone(), env!("CARGO_PKG_VERSION"));

    // Get the function list for CallDirect → CallByName conversion
    let function_list = compiler.get_function_list_names();

    // Get all types from compiler
    let all_types = compiler.get_all_types();

    // Get all mvars from compiler
    let all_mvars = compiler.get_mvars();

    // Group functions by module and save
    let mut saved_count = 0;
    for (module_name, source_hash, path, prelude_imports, fn_visibility) in &modules {
        let module_prefix = format!("{}.", module_name);

        // Collect functions for this module
        let mut cached_functions = Vec::new();
        for (func_name, func) in functions {
            if func_name.starts_with(&module_prefix) || func.module.as_deref() == Some(module_name.as_str()) {
                // Use function_to_cached_with_fn_list to convert CallDirect to CallByName
                if let Some(mut cached) = function_to_cached_with_fn_list(func, function_list) {
                    // Update visibility from AST
                    // Strip signature from function name (e.g., "stdlib.rhtml.h1/List,..." -> "stdlib.rhtml.h1")
                    let base_name = func_name.split('/').next().unwrap_or(func_name);
                    if let Some(&is_public) = fn_visibility.get(base_name) {
                        cached.is_public = is_public;
                    }
                    cached_functions.push(cached);
                }
            }
        }

        // Collect types for this module
        let module_types: Vec<nostos_vm::value::TypeValue> = all_types.iter()
            .filter(|(type_name, _)| type_name.starts_with(&module_prefix))
            .map(|(_, type_val)| (**type_val).clone())
            .collect();

        // Collect mvars for this module
        let module_mvars: Vec<CachedMvar> = all_mvars.iter()
            .filter(|(mvar_name, _)| mvar_name.starts_with(&module_prefix))
            .map(|(name, info)| CachedMvar {
                name: name.clone(),
                type_name: info.type_name.clone(),
                initial_value: mvar_init_to_cached(&info.initial_value),
            })
            .collect();

        let cached_module = CachedModule {
            module_path: module_name.split('.').map(|s| s.to_string()).collect(),
            source_hash: source_hash.clone(),
            functions: cached_functions,
            function_signatures: std::collections::HashMap::new(),
            exports: Vec::new(),
            prelude_imports: prelude_imports.clone(),
            types: module_types,
            mvars: module_mvars,
            dependency_signatures: std::collections::HashMap::new(),
        };

        // Save even modules with no functions (for types/traits-only modules)
        if let Err(e) = cache.save_module(module_name, path.to_str().unwrap(), &cached_module) {
            eprintln!("Error saving cache for {}: {}", module_name, e);
        } else {
            saved_count += 1;
        }
    }

    // Save manifest
    if let Err(e) = cache.save_manifest() {
        eprintln!("Error saving cache manifest: {}", e);
        return ExitCode::FAILURE;
    }

    println!("Saved {} module caches to {}", saved_count, cache_dir.display());
    println!("Cache built successfully!");

    ExitCode::SUCCESS
}

/// Clear the bytecode cache
fn clear_bytecode_cache() -> ExitCode {
    let cache_dir = get_cache_dir();

    if !cache_dir.exists() {
        println!("Cache directory does not exist: {}", cache_dir.display());
        return ExitCode::SUCCESS;
    }

    match fs::remove_dir_all(&cache_dir) {
        Ok(_) => {
            println!("Cleared bytecode cache at {}", cache_dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error clearing cache: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Format a runtime error as JSON for debugger integration.
fn format_error_json(error: &RuntimeError, file_path: &str) -> String {
    // Extract the error type, message, and stack trace
    let (error_type, message, stack_trace) = extract_error_info(error);

    // Build JSON manually (avoid adding serde dependency just for this)
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"file\": {},\n", json_string(file_path)));
    json.push_str(&format!("  \"error_type\": {},\n", json_string(&error_type)));
    json.push_str(&format!("  \"message\": {},\n", json_string(&message)));
    json.push_str("  \"stack_trace\": [\n");

    for (i, frame) in stack_trace.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"function\": {},\n", json_string(&frame.0)));
        json.push_str(&format!("      \"line\": {}\n", frame.1));
        json.push_str("    }");
        if i < stack_trace.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }

    json.push_str("  ]\n");
    json.push_str("}");
    json
}

/// Extract error type, message, and stack frames from a RuntimeError.
fn extract_error_info(error: &RuntimeError) -> (String, String, Vec<(String, usize)>) {
    match error {
        RuntimeError::WithStackTrace { error: inner, stack_trace } => {
            let (error_type, message) = get_error_type_and_message(inner);
            let frames = parse_stack_trace(stack_trace);
            (error_type, message, frames)
        }
        _ => {
            let (error_type, message) = get_error_type_and_message(error);
            (error_type, message, vec![])
        }
    }
}

/// Get the error type name and message from a RuntimeError.
fn get_error_type_and_message(error: &RuntimeError) -> (String, String) {
    match error {
        RuntimeError::TypeError { expected, found } => {
            ("TypeError".to_string(), format!("expected {}, got {}", expected, found))
        }
        RuntimeError::DivisionByZero => {
            ("DivisionByZero".to_string(), "Division by zero".to_string())
        }
        RuntimeError::IndexOutOfBounds { index, length } => {
            ("IndexOutOfBounds".to_string(), format!("Index {} out of bounds (length {})", index, length))
        }
        RuntimeError::UnknownField { type_name, field } => {
            ("UnknownField".to_string(), format!("Unknown field '{}' on type '{}'", field, type_name))
        }
        RuntimeError::ImmutableField { field } => {
            ("ImmutableField".to_string(), format!("Cannot mutate immutable field '{}'", field))
        }
        RuntimeError::ImmutableBinding { name } => {
            ("ImmutableBinding".to_string(), format!("Cannot mutate immutable binding '{}'", name))
        }
        RuntimeError::UnknownVariable(name) => {
            ("UnknownVariable".to_string(), format!("Unknown variable '{}'", name))
        }
        RuntimeError::UnknownFunction(name) => {
            ("UnknownFunction".to_string(), format!("Unknown function '{}'", name))
        }
        RuntimeError::ArityMismatch { expected, found } => {
            ("ArityMismatch".to_string(), format!("Expected {} arguments, got {}", expected, found))
        }
        RuntimeError::MatchFailed => {
            ("MatchFailed".to_string(), "Pattern match failed".to_string())
        }
        RuntimeError::AssertionFailed(msg) => {
            ("AssertionFailed".to_string(), msg.clone())
        }
        RuntimeError::Panic(msg) => {
            ("Panic".to_string(), msg.clone())
        }
        RuntimeError::StackOverflow => {
            ("StackOverflow".to_string(), "Stack overflow".to_string())
        }
        RuntimeError::ProcessNotFound(pid) => {
            ("ProcessNotFound".to_string(), format!("Process not found: {:?}", pid))
        }
        RuntimeError::Timeout => {
            ("Timeout".to_string(), "Timeout".to_string())
        }
        RuntimeError::IOError(msg) => {
            ("IOError".to_string(), msg.clone())
        }
        RuntimeError::WithStackTrace { error, .. } => {
            get_error_type_and_message(error)
        }
        RuntimeError::Interrupted => {
            ("Interrupted".to_string(), "Execution interrupted (Ctrl+C)".to_string())
        }
    }
}

/// Parse a formatted stack trace string into (function_name, line_number) pairs.
fn parse_stack_trace(trace: &str) -> Vec<(String, usize)> {
    let mut frames = Vec::new();
    for line in trace.lines() {
        let line = line.trim();
        // Format: "1. function_name (line N)" or "1. function_name"
        if let Some(rest) = line.strip_prefix(|c: char| c.is_ascii_digit()) {
            let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
            let rest = rest.trim();

            if let Some(paren_pos) = rest.find(" (line ") {
                let func_name = rest[..paren_pos].to_string();
                let line_part = &rest[paren_pos + 7..];
                if let Some(end_paren) = line_part.find(')') {
                    if let Ok(line_num) = line_part[..end_paren].parse::<usize>() {
                        frames.push((func_name, line_num));
                        continue;
                    }
                }
            }
            // No line number
            frames.push((rest.to_string(), 0));
        }
    }
    frames
}

/// Escape a string for JSON output.
fn json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

/// Run the REPL - now just launches the TUI
fn run_repl(args: &[String]) -> ExitCode {
    // The REPL command now just launches the TUI
    tui::run_tui(args)
}

/// Convert MvarInitValue to ThreadSafeValue for VM registration
fn mvar_init_to_thread_safe(init: &MvarInitValue) -> ThreadSafeValue {
    match init {
        MvarInitValue::Unit => ThreadSafeValue::Unit,
        MvarInitValue::Bool(b) => ThreadSafeValue::Bool(*b),
        MvarInitValue::Int(n) => ThreadSafeValue::Int64(*n),
        MvarInitValue::Float(f) => ThreadSafeValue::Float64(*f),
        MvarInitValue::String(s) => ThreadSafeValue::String(s.clone()),
        MvarInitValue::Char(c) => ThreadSafeValue::Char(*c),
        MvarInitValue::EmptyList => ThreadSafeValue::List(vec![]),
        MvarInitValue::IntList(ints) => ThreadSafeValue::List(
            ints.iter().map(|n| ThreadSafeValue::Int64(*n)).collect()
        ),
        MvarInitValue::StringList(strings) => ThreadSafeValue::List(
            strings.iter().map(|s| ThreadSafeValue::String(s.clone())).collect()
        ),
        MvarInitValue::FloatList(floats) => ThreadSafeValue::List(
            floats.iter().map(|f| ThreadSafeValue::Float64(*f)).collect()
        ),
        MvarInitValue::BoolList(bools) => ThreadSafeValue::List(
            bools.iter().map(|b| ThreadSafeValue::Bool(*b)).collect()
        ),
        MvarInitValue::Tuple(items) => ThreadSafeValue::Tuple(
            items.iter().map(|item| mvar_init_to_thread_safe(item)).collect()
        ),
        MvarInitValue::List(items) => ThreadSafeValue::List(
            items.iter().map(|item| mvar_init_to_thread_safe(item)).collect()
        ),
        MvarInitValue::Record(type_name, fields) => {
            // Convert to record with field names and values
            let field_names: Vec<String> = fields.iter()
                .enumerate()
                .map(|(i, (name, _))| {
                    if name.is_empty() {
                        format!("_{}", i) // Positional fields
                    } else {
                        name.clone()
                    }
                })
                .collect();
            let values: Vec<ThreadSafeValue> = fields.iter()
                .map(|(_, val)| mvar_init_to_thread_safe(val))
                .collect();
            ThreadSafeValue::Record {
                type_name: type_name.clone(),
                field_names,
                fields: values,
                mutable_fields: vec![false; fields.len()],
            }
        }
        MvarInitValue::EmptyMap => {
            ThreadSafeValue::Map(nostos_vm::empty_shared_map())
        }
        MvarInitValue::Map(entries) => {
            use nostos_vm::{SharedMapKey, SharedMapValue};
            let mut map = imbl::HashMap::new();
            for (k, v) in entries {
                if let Some(key) = mvar_init_to_shared_key(k) {
                    let value = mvar_init_to_shared_value(v);
                    map.insert(key, value);
                }
            }
            ThreadSafeValue::Map(std::sync::Arc::new(map))
        }
    }
}

/// Convert MvarInitValue to SharedMapKey (for map keys)
fn mvar_init_to_shared_key(init: &MvarInitValue) -> Option<nostos_vm::SharedMapKey> {
    use nostos_vm::SharedMapKey;
    match init {
        MvarInitValue::Unit => Some(SharedMapKey::Unit),
        MvarInitValue::Bool(b) => Some(SharedMapKey::Bool(*b)),
        MvarInitValue::Int(n) => Some(SharedMapKey::Int64(*n)),
        MvarInitValue::String(s) => Some(SharedMapKey::String(s.clone())),
        MvarInitValue::Char(c) => Some(SharedMapKey::Char(*c)),
        _ => None, // Other types can't be map keys
    }
}

/// Convert MvarInitValue to SharedMapValue (for map values)
fn mvar_init_to_shared_value(init: &MvarInitValue) -> nostos_vm::SharedMapValue {
    use nostos_vm::SharedMapValue;
    match init {
        MvarInitValue::Unit => SharedMapValue::Unit,
        MvarInitValue::Bool(b) => SharedMapValue::Bool(*b),
        MvarInitValue::Int(n) => SharedMapValue::Int64(*n),
        MvarInitValue::Float(f) => SharedMapValue::Float64(*f),
        MvarInitValue::String(s) => SharedMapValue::String(s.clone()),
        MvarInitValue::Char(c) => SharedMapValue::Char(*c),
        MvarInitValue::EmptyList => SharedMapValue::List(vec![]),
        MvarInitValue::IntList(ints) => SharedMapValue::List(
            ints.iter().map(|n| SharedMapValue::Int64(*n)).collect()
        ),
        MvarInitValue::FloatList(floats) => SharedMapValue::List(
            floats.iter().map(|f| SharedMapValue::Float64(*f)).collect()
        ),
        MvarInitValue::BoolList(bools) => SharedMapValue::List(
            bools.iter().map(|b| SharedMapValue::Bool(*b)).collect()
        ),
        MvarInitValue::StringList(strings) => SharedMapValue::List(
            strings.iter().map(|s| SharedMapValue::String(s.clone())).collect()
        ),
        MvarInitValue::List(items) => SharedMapValue::List(
            items.iter().map(|item| mvar_init_to_shared_value(item)).collect()
        ),
        MvarInitValue::Tuple(items) => SharedMapValue::Tuple(
            items.iter().map(|item| mvar_init_to_shared_value(item)).collect()
        ),
        MvarInitValue::Record(type_name, fields) => {
            let field_names: Vec<String> = fields.iter()
                .enumerate()
                .map(|(i, (name, _))| if name.is_empty() { format!("_{}", i) } else { name.clone() })
                .collect();
            let values: Vec<SharedMapValue> = fields.iter()
                .map(|(_, val)| mvar_init_to_shared_value(val))
                .collect();
            SharedMapValue::Record { type_name: type_name.clone(), field_names, fields: values }
        }
        MvarInitValue::EmptyMap => SharedMapValue::Map(nostos_vm::empty_shared_map()),
        MvarInitValue::Map(entries) => {
            let mut map = imbl::HashMap::new();
            for (k, v) in entries {
                if let Some(key) = mvar_init_to_shared_key(k) {
                    let value = mvar_init_to_shared_value(v);
                    map.insert(key, value);
                }
            }
            SharedMapValue::Map(std::sync::Arc::new(map))
        }
    }
}

/// Convert MvarInitValue to CachedMvarValue for cache storage.
fn mvar_init_to_cached(init: &MvarInitValue) -> CachedMvarValue {
    match init {
        MvarInitValue::Unit => CachedMvarValue::Unit,
        MvarInitValue::Bool(b) => CachedMvarValue::Bool(*b),
        MvarInitValue::Int(n) => CachedMvarValue::Int(*n),
        MvarInitValue::Float(f) => CachedMvarValue::Float(*f),
        MvarInitValue::String(s) => CachedMvarValue::String(s.clone()),
        MvarInitValue::Char(c) => CachedMvarValue::Char(*c),
        MvarInitValue::EmptyList => CachedMvarValue::EmptyList,
        MvarInitValue::IntList(ints) => CachedMvarValue::IntList(ints.clone()),
        MvarInitValue::StringList(strings) => CachedMvarValue::StringList(strings.clone()),
        MvarInitValue::FloatList(floats) => CachedMvarValue::FloatList(floats.clone()),
        MvarInitValue::BoolList(bools) => CachedMvarValue::BoolList(bools.clone()),
        MvarInitValue::Tuple(items) => CachedMvarValue::Tuple(
            items.iter().map(mvar_init_to_cached).collect()
        ),
        MvarInitValue::List(items) => CachedMvarValue::List(
            items.iter().map(mvar_init_to_cached).collect()
        ),
        MvarInitValue::Record(type_name, fields) => CachedMvarValue::Record(
            type_name.clone(),
            fields.iter().map(|(name, val)| (name.clone(), mvar_init_to_cached(val))).collect()
        ),
        MvarInitValue::EmptyMap => CachedMvarValue::EmptyMap,
        MvarInitValue::Map(entries) => CachedMvarValue::Map(
            entries.iter().map(|(k, v)| (mvar_init_to_cached(k), mvar_init_to_cached(v))).collect()
        ),
    }
}

/// Convert CachedMvarValue back to MvarInitValue for cache loading.
fn cached_to_mvar_init(cached: &CachedMvarValue) -> MvarInitValue {
    match cached {
        CachedMvarValue::Unit => MvarInitValue::Unit,
        CachedMvarValue::Bool(b) => MvarInitValue::Bool(*b),
        CachedMvarValue::Int(n) => MvarInitValue::Int(*n),
        CachedMvarValue::Float(f) => MvarInitValue::Float(*f),
        CachedMvarValue::String(s) => MvarInitValue::String(s.clone()),
        CachedMvarValue::Char(c) => MvarInitValue::Char(*c),
        CachedMvarValue::EmptyList => MvarInitValue::EmptyList,
        CachedMvarValue::IntList(ints) => MvarInitValue::IntList(ints.clone()),
        CachedMvarValue::StringList(strings) => MvarInitValue::StringList(strings.clone()),
        CachedMvarValue::FloatList(floats) => MvarInitValue::FloatList(floats.clone()),
        CachedMvarValue::BoolList(bools) => MvarInitValue::BoolList(bools.clone()),
        CachedMvarValue::Tuple(items) => MvarInitValue::Tuple(
            items.iter().map(cached_to_mvar_init).collect()
        ),
        CachedMvarValue::List(items) => MvarInitValue::List(
            items.iter().map(cached_to_mvar_init).collect()
        ),
        CachedMvarValue::Record(type_name, fields) => MvarInitValue::Record(
            type_name.clone(),
            fields.iter().map(|(name, val)| (name.clone(), cached_to_mvar_init(val))).collect()
        ),
        CachedMvarValue::EmptyMap => MvarInitValue::EmptyMap,
        CachedMvarValue::Map(entries) => MvarInitValue::Map(
            entries.iter().map(|(k, v)| (cached_to_mvar_init(k), cached_to_mvar_init(v))).collect()
        ),
    }
}

/// Run program using the tokio-based AsyncVM.
fn run_with_async_vm(
    compiler: &Compiler,
    entry_point_name: &str,
    profiling_enabled: bool,
    enable_jit: bool,
    ext_mgr: Option<std::sync::Arc<nostos_vm::ExtensionManager>>,
) -> ExitCode {
    let config = AsyncConfig {
        profiling_enabled,
        ..AsyncConfig::default()
    };
    let mut vm = AsyncVM::new(config);

    // Register default native functions
    vm.register_default_natives();

    // Set extension manager if provided
    if let Some(ext_mgr) = ext_mgr {
        vm.set_extension_manager(ext_mgr);
    }

    // Register all functions from compiler
    for (name, func) in compiler.get_all_functions().iter() {
        vm.register_function(name, func.clone());
    }

    // Set function list for indexed calls
    let func_list = compiler.get_function_list();
    vm.set_function_list(func_list);

    // Register all types
    for (name, type_val) in compiler.get_vm_types().iter() {
        vm.register_type(name, type_val.clone());
    }

    // Register mvars (module-level mutable variables)
    for (name, info) in compiler.get_mvars() {
        let initial_value = mvar_init_to_thread_safe(&info.initial_value);
        vm.register_mvar(name, initial_value);
    }

    // JIT compile suitable functions (unless --no-jit was specified)
    let function_list = compiler.get_function_list();
    if enable_jit {
    if let Ok(mut jit) = JitCompiler::new(JitConfig::default()) {
        for idx in 0..function_list.len() {
            jit.queue_compilation(idx as u16);
        }
        if let Ok(compiled) = jit.process_queue(&function_list) {
            if compiled > 0 {
                for (idx, _func) in function_list.iter().enumerate() {
                    if let Some(jit_fn) = jit.get_int_function_0(idx as u16) {
                        vm.register_jit_int_function_0(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_int_function(idx as u16) {
                        vm.register_jit_int_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_int_function_2(idx as u16) {
                        vm.register_jit_int_function_2(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_int_function_3(idx as u16) {
                        vm.register_jit_int_function_3(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_int_function_4(idx as u16) {
                        vm.register_jit_int_function_4(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_loop_int64_array_function(idx as u16) {
                        vm.register_jit_loop_array_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_recursive_array_fill_function(idx as u16) {
                        vm.register_jit_array_fill_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_recursive_array_sum_function(idx as u16) {
                        vm.register_jit_array_sum_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_list_sum_function(idx as u16) {
                        vm.register_jit_list_sum_function(idx as u16, jit_fn);
                    }
                    if let Some(jit_fn) = jit.get_list_sum_tr_function(idx as u16) {
                        vm.register_jit_list_sum_tr_function(idx as u16, jit_fn);
                    }
                }
            }
        }
    }
    }

    // Set up Ctrl+C handler - exit immediately since IO operations may block
    // and not check the interrupt flag
    if let Err(e) = ctrlc::set_handler(move || {
        eprintln!("\nInterrupted");
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        eprintln!("Warning: Could not set Ctrl+C handler: {}", e);
    }

    // Run the program
    match vm.run(entry_point_name) {
        Ok(result) => {
            if !result.is_unit() {
                println!("{}", result.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) if e.contains("Interrupted") => {
            eprintln!("Interrupted");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            ExitCode::FAILURE
        }
    }
}

const REGISTRY_URL: &str = "https://raw.githubusercontent.com/pegesund/nostos/master/nostlets-registry.json";

/// Run the nostlet subcommand
fn run_nostlet_command(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: nostos nostlet <command>");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  list              List available nostlets from registry");
        eprintln!("  install <name>    Install a nostlet to ~/.nostos/nostlets/");
        eprintln!("  installed         List locally installed nostlets");
        return ExitCode::FAILURE;
    }

    match args[0].as_str() {
        "list" => nostlet_list(),
        "install" => {
            if args.len() < 2 {
                eprintln!("Usage: nostos nostlet install <name>");
                return ExitCode::FAILURE;
            }
            nostlet_install(&args[1])
        }
        "installed" => nostlet_installed(),
        _ => {
            eprintln!("Unknown nostlet command: {}", args[0]);
            eprintln!("Use 'nostos nostlet' for help");
            ExitCode::FAILURE
        }
    }
}

/// Run the init subcommand - create a new project
fn run_init_command(args: &[String]) -> ExitCode {
    // Get project name from args or use current directory name
    let (project_dir, project_name) = if args.is_empty() {
        // Use current directory
        let cwd = match env::current_dir() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: Could not get current directory: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let name = cwd.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("my-project")
            .to_string();
        (cwd, name)
    } else {
        // Create new directory with given name
        let dir = std::path::PathBuf::from(&args[0]);
        let name = dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&args[0])
            .to_string();
        (dir, name)
    };

    // Create directory if it doesn't exist
    if !project_dir.exists() {
        if let Err(e) = fs::create_dir_all(&project_dir) {
            eprintln!("Error: Could not create directory '{}': {}", project_dir.display(), e);
            return ExitCode::FAILURE;
        }
        println!("Created directory: {}", project_dir.display());
    }

    // Check if nostos.toml already exists
    let config_path = project_dir.join("nostos.toml");
    if config_path.exists() {
        eprintln!("Error: nostos.toml already exists in {}", project_dir.display());
        return ExitCode::FAILURE;
    }

    // Create nostos.toml
    let config_content = format!(r#"[project]
name = "{}"
version = "0.1.0"
"#, project_name);

    if let Err(e) = fs::write(&config_path, &config_content) {
        eprintln!("Error: Could not create nostos.toml: {}", e);
        return ExitCode::FAILURE;
    }
    println!("Created: {}", config_path.display());

    // Create main.nos
    let main_path = project_dir.join("main.nos");
    if !main_path.exists() {
        let main_content = r#"# Main entry point

main() = {
    println("Hello from Nostos!")
    0
}
"#;
        if let Err(e) = fs::write(&main_path, main_content) {
            eprintln!("Error: Could not create main.nos: {}", e);
            return ExitCode::FAILURE;
        }
        println!("Created: {}", main_path.display());
    }

    println!();
    println!("Project '{}' initialized!", project_name);
    println!();
    println!("Run your project with:");
    println!("    nostos {}/", project_dir.display());

    ExitCode::SUCCESS
}

/// Run the extension subcommand
fn run_extension_command(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Manage native Rust extensions for Nostos");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("    nostos extension <command>");
        eprintln!();
        eprintln!("COMMANDS:");
        eprintln!("    install <git-url>   Install extension from GitHub repository");
        eprintln!("    list                List installed extensions");
        eprintln!("    remove <name>       Remove an installed extension");
        eprintln!();
        eprintln!("EXAMPLES:");
        eprintln!("    nostos extension install https://github.com/pegesund/nostos-nalgebra");
        eprintln!("    nostos extension list");
        eprintln!();
        eprintln!("After installation, use extensions with:");
        eprintln!("    nostos --use nalgebra myprogram.nos");
        return ExitCode::FAILURE;
    }

    match args[0].as_str() {
        "install" => {
            if args.len() < 2 {
                eprintln!("Usage: nostos extension install <git-url>");
                eprintln!();
                eprintln!("Example:");
                eprintln!("    nostos extension install https://github.com/pegesund/nostos-nalgebra");
                return ExitCode::FAILURE;
            }
            extension_install(&args[1])
        }
        "list" => extension_list(),
        "remove" => {
            if args.len() < 2 {
                eprintln!("Usage: nostos extension remove <name>");
                return ExitCode::FAILURE;
            }
            extension_remove(&args[1])
        }
        _ => {
            eprintln!("Unknown extension command: {}", args[0]);
            eprintln!("Use 'nostos extension' for help");
            ExitCode::FAILURE
        }
    }
}

/// Run the cache subcommand (clear, build, info)
fn run_cache_command(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Usage: nostos cache <command>");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("    clear [target]    Clear cached bytecode");
        eprintln!("    build             Build cache for current project");
        eprintln!("    info              Show cache information");
        eprintln!();
        eprintln!("Clear targets:");
        eprintln!("    all               Clear all caches (default)");
        eprintln!("    stdlib            Clear stdlib cache");
        eprintln!("    extensions        Clear all extension caches");
        eprintln!("    packages          Clear all package caches");
        eprintln!("    project           Clear current project cache");
        return ExitCode::FAILURE;
    }

    match args[0].as_str() {
        "clear" => {
            let target = args.get(1).map(|s| s.as_str()).unwrap_or("all");
            cache_clear(target)
        }
        "build" => cache_build(),
        "info" => cache_info(),
        _ => {
            eprintln!("Unknown cache command: {}", args[0]);
            eprintln!("Use 'nostos cache' for help");
            ExitCode::FAILURE
        }
    }
}

/// Clear cache based on target
fn cache_clear(target: &str) -> ExitCode {
    match target {
        "all" => {
            let cache_dir = get_cache_dir();
            let project_cache = std::path::PathBuf::from(".nostos-cache");

            let mut cleared_any = false;

            // Clear global cache (~/.nostos/cache/)
            if cache_dir.exists() {
                match fs::remove_dir_all(&cache_dir) {
                    Ok(_) => {
                        println!("Cleared global cache: {}", cache_dir.display());
                        cleared_any = true;
                    }
                    Err(e) => eprintln!("Error clearing global cache: {}", e),
                }
            }

            // Clear project cache (.nostos-cache/)
            if project_cache.exists() {
                match fs::remove_dir_all(&project_cache) {
                    Ok(_) => {
                        println!("Cleared project cache: {}", project_cache.display());
                        cleared_any = true;
                    }
                    Err(e) => eprintln!("Error clearing project cache: {}", e),
                }
            }

            if !cleared_any {
                println!("No caches to clear");
            }
            ExitCode::SUCCESS
        }
        "stdlib" => {
            let cache_dir = get_cache_dir();
            // Only clear stdlib files, not extensions/packages subdirs
            let manifest = cache_dir.join("manifest.bin");
            let mut cleared = false;

            if manifest.exists() {
                if let Err(e) = fs::remove_file(&manifest) {
                    eprintln!("Error removing manifest: {}", e);
                }
            }

            // Remove .cache files but keep extensions/ and packages/ dirs
            if cache_dir.exists() {
                if let Ok(entries) = fs::read_dir(&cache_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() && path.extension().map(|e| e == "cache").unwrap_or(false) {
                            if let Err(e) = fs::remove_file(&path) {
                                eprintln!("Error removing {}: {}", path.display(), e);
                            } else {
                                cleared = true;
                            }
                        }
                    }
                }
            }

            if cleared || manifest.exists() {
                println!("Cleared stdlib cache");
            } else {
                println!("Stdlib cache already empty");
            }
            ExitCode::SUCCESS
        }
        "extensions" => {
            let ext_cache_dir = get_cache_dir().join("extensions");
            if ext_cache_dir.exists() {
                match fs::remove_dir_all(&ext_cache_dir) {
                    Ok(_) => println!("Cleared extension caches: {}", ext_cache_dir.display()),
                    Err(e) => {
                        eprintln!("Error clearing extension caches: {}", e);
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                println!("No extension caches to clear");
            }
            ExitCode::SUCCESS
        }
        "packages" => {
            let pkg_cache_dir = get_cache_dir().join("packages");
            if pkg_cache_dir.exists() {
                match fs::remove_dir_all(&pkg_cache_dir) {
                    Ok(_) => println!("Cleared package caches: {}", pkg_cache_dir.display()),
                    Err(e) => {
                        eprintln!("Error clearing package caches: {}", e);
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                println!("No package caches to clear");
            }
            ExitCode::SUCCESS
        }
        "project" => {
            let project_cache = std::path::PathBuf::from(".nostos-cache");
            if project_cache.exists() {
                match fs::remove_dir_all(&project_cache) {
                    Ok(_) => println!("Cleared project cache: {}", project_cache.display()),
                    Err(e) => {
                        eprintln!("Error clearing project cache: {}", e);
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                println!("No project cache to clear (not in a project directory)");
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("Unknown clear target: {}", target);
            eprintln!("Valid targets: all, stdlib, extensions, packages, project");
            ExitCode::FAILURE
        }
    }
}

/// Build/warm cache for current project
fn cache_build() -> ExitCode {
    // Check if we're in a project directory
    let current_dir = std::env::current_dir().unwrap_or_default();

    // Look for nostos.toml or main.nos
    let has_toml = current_dir.join("nostos.toml").exists();
    let has_main = current_dir.join("main.nos").exists();

    if !has_toml && !has_main {
        eprintln!("Not in a Nostos project directory");
        eprintln!("(no nostos.toml or main.nos found)");
        return ExitCode::FAILURE;
    }

    println!("Building cache for project: {}", current_dir.display());
    println!("Run 'nostos .' to build the cache");
    println!("(the cache is built automatically on first run)");
    ExitCode::SUCCESS
}

/// Show cache information
fn cache_info() -> ExitCode {
    use nostos_vm::cache::BytecodeCache;

    let cache_dir = get_cache_dir();
    let project_cache = std::path::PathBuf::from(".nostos-cache");

    println!("Cache locations:");
    println!();

    // Stdlib cache
    println!("Stdlib cache: {}", cache_dir.display());
    if cache_dir.exists() {
        let manifest_path = cache_dir.join("manifest.bin");
        if manifest_path.exists() {
            let cache = BytecodeCache::new(cache_dir.clone(), env!("CARGO_PKG_VERSION"));
            let module_count = cache.manifest().modules.len();
            if module_count > 0 {
                println!("  Status: {} modules cached", module_count);
                let total_size = calculate_dir_size(&cache_dir);
                println!("  Size: {}", format_size(total_size));
            } else {
                println!("  Status: empty");
            }
        } else {
            println!("  Status: empty");
        }
    } else {
        println!("  Status: not created");
    }
    println!();

    // Extension caches
    let ext_cache_dir = cache_dir.join("extensions");
    println!("Extension caches: {}", ext_cache_dir.display());
    if ext_cache_dir.exists() {
        if let Ok(entries) = fs::read_dir(&ext_cache_dir) {
            let mut count = 0;
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let ext_name = entry.file_name().to_string_lossy().to_string();
                    let ext_dir = entry.path();
                    let manifest_path = ext_dir.join("manifest.bin");
                    if manifest_path.exists() {
                        let cache = BytecodeCache::new(ext_dir.clone(), env!("CARGO_PKG_VERSION"));
                        let module_count = cache.manifest().modules.len();
                        if module_count > 0 {
                            let size = calculate_dir_size(&ext_dir);
                            println!("  {}: {} modules ({})", ext_name, module_count, format_size(size));
                            count += 1;
                        }
                    }
                }
            }
            if count == 0 {
                println!("  (empty)");
            }
        }
    } else {
        println!("  (not created)");
    }
    println!();

    // Package caches
    let pkg_cache_dir = cache_dir.join("packages");
    println!("Package caches: {}", pkg_cache_dir.display());
    if pkg_cache_dir.exists() {
        if let Ok(entries) = fs::read_dir(&pkg_cache_dir) {
            let mut count = 0;
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let pkg_name = entry.file_name().to_string_lossy().to_string();
                    let pkg_dir = entry.path();
                    let manifest_path = pkg_dir.join("manifest.bin");
                    if manifest_path.exists() {
                        let cache = BytecodeCache::new(pkg_dir.clone(), env!("CARGO_PKG_VERSION"));
                        let module_count = cache.manifest().modules.len();
                        if module_count > 0 {
                            let size = calculate_dir_size(&pkg_dir);
                            println!("  {}: {} modules ({})", pkg_name, module_count, format_size(size));
                            count += 1;
                        }
                    }
                }
            }
            if count == 0 {
                println!("  (empty)");
            }
        }
    } else {
        println!("  (not created)");
    }
    println!();

    // Project cache
    println!("Project cache: {}", project_cache.display());
    if project_cache.exists() {
        let manifest_path = project_cache.join("manifest.bin");
        if manifest_path.exists() {
            let cache = BytecodeCache::new(project_cache.clone(), env!("CARGO_PKG_VERSION"));
            let module_count = cache.manifest().modules.len();
            if module_count > 0 {
                let size = calculate_dir_size(&project_cache);
                println!("  Status: {} modules cached ({})", module_count, format_size(size));
            } else {
                println!("  Status: empty");
            }
        } else {
            println!("  Status: empty");
        }
    } else {
        println!("  Status: not in a cached project");
    }

    ExitCode::SUCCESS
}

/// Calculate total size of a directory
fn calculate_dir_size(path: &std::path::Path) -> u64 {
    let mut size = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                size += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                size += calculate_dir_size(&path);
            }
        }
    }
    size
}

/// Format size in human-readable form
fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Install an extension from a git URL
fn extension_install(git_url: &str) -> ExitCode {
    // Extract repo name from URL
    let repo_name = git_url
        .trim_end_matches(".git")
        .rsplit('/')
        .next()
        .unwrap_or("unknown");

    println!("Installing extension from {}...", git_url);

    // Create extension dependency
    let dep = packages::ExtensionDep {
        git: git_url.to_string(),
        branch: None,
        tag: None,
        version: None,
    };

    // Fetch the extension
    let ext_dir = match packages::fetch_extension(repo_name, &dep) {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Error fetching extension: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Build the extension
    match packages::build_extension(&ext_dir) {
        Ok(lib_path) => {
            println!();
            println!("Extension installed successfully!");
            println!("  Location: {}", ext_dir.display());
            println!("  Library:  {}", lib_path.display());
            println!();
            println!("Use it with:");
            println!("    nostos --use {} yourprogram.nos", repo_name.strip_prefix("nostos-").unwrap_or(repo_name));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error building extension: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// List installed extensions
fn extension_list() -> ExitCode {
    let ext_dir = packages::extensions_cache_dir();

    if !ext_dir.exists() {
        println!("No extensions installed.");
        println!();
        println!("Install extensions with:");
        println!("    nostos extension install <git-url>");
        return ExitCode::SUCCESS;
    }

    let entries = match fs::read_dir(&ext_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error reading extensions directory: {}", e);
            return ExitCode::FAILURE;
        }
    };

    println!("Installed extensions in {}:", ext_dir.display());
    println!("{:-<60}", "");

    let mut found = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy();

            // Check if it has a built library
            let release_dir = path.join("target").join("release");
            let has_lib = release_dir.exists() && fs::read_dir(&release_dir)
                .map(|entries| entries.flatten().any(|e| {
                    let p = e.path();
                    p.extension().map(|ext| ext == "so" || ext == "dylib").unwrap_or(false)
                }))
                .unwrap_or(false);

            let status = if has_lib { "built" } else { "not built" };
            let use_name = name.strip_prefix("nostos-").unwrap_or(&name);

            println!("  {} [{}]", name, status);
            println!("      Use with: nostos --use {} <file.nos>", use_name);
            println!();
            found = true;
        }
    }

    if !found {
        println!("  (none)");
    }

    ExitCode::SUCCESS
}

/// Remove an installed extension
fn extension_remove(name: &str) -> ExitCode {
    let ext_dir = packages::extensions_cache_dir();

    // Try both with and without nostos- prefix
    let candidates = vec![
        ext_dir.join(format!("nostos-{}", name)),
        ext_dir.join(name),
    ];

    for path in &candidates {
        if path.exists() {
            println!("Removing extension at {}...", path.display());
            if let Err(e) = fs::remove_dir_all(path) {
                eprintln!("Error removing extension: {}", e);
                return ExitCode::FAILURE;
            }
            println!("Extension '{}' removed.", name);
            return ExitCode::SUCCESS;
        }
    }

    eprintln!("Extension '{}' not found.", name);
    eprintln!("Tried:");
    for c in &candidates {
        eprintln!("  - {}", c.display());
    }
    ExitCode::FAILURE
}

/// Fetch the nostlet registry from GitHub
fn fetch_registry() -> Result<serde_json::Value, String> {
    let response = ureq::get(REGISTRY_URL)
        .call()
        .map_err(|e| format!("Failed to fetch registry: {}", e))?;

    let body = response
        .into_string()
        .map_err(|e| format!("Failed to read registry: {}", e))?;

    serde_json::from_str(&body).map_err(|e| format!("Failed to parse registry: {}", e))
}

/// List available nostlets from registry
fn nostlet_list() -> ExitCode {
    println!("Fetching nostlet registry...");

    let registry = match fetch_registry() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let nostlets = match registry.get("nostlets").and_then(|n| n.as_array()) {
        Some(n) => n,
        None => {
            eprintln!("Error: Invalid registry format");
            return ExitCode::FAILURE;
        }
    };

    if nostlets.is_empty() {
        println!("No nostlets available in registry.");
        return ExitCode::SUCCESS;
    }

    println!();
    println!("Available nostlets:");
    println!("{:-<60}", "");

    for nostlet in nostlets {
        let name = nostlet.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
        let desc = nostlet.get("description").and_then(|d| d.as_str()).unwrap_or("");
        let author = nostlet.get("author").and_then(|a| a.as_str()).unwrap_or("unknown");

        println!("  {} - {}", name, desc);
        println!("    by {}", author);
        println!();
    }

    println!("Install with: nostos nostlet install <name>");
    ExitCode::SUCCESS
}

/// Install a nostlet from registry
fn nostlet_install(name: &str) -> ExitCode {
    println!("Fetching nostlet registry...");

    let registry = match fetch_registry() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let nostlets = match registry.get("nostlets").and_then(|n| n.as_array()) {
        Some(n) => n,
        None => {
            eprintln!("Error: Invalid registry format");
            return ExitCode::FAILURE;
        }
    };

    // Find the nostlet by name
    let nostlet = nostlets.iter().find(|n| {
        n.get("name").and_then(|nm| nm.as_str()) == Some(name)
    });

    let nostlet = match nostlet {
        Some(n) => n,
        None => {
            eprintln!("Error: Nostlet '{}' not found in registry", name);
            eprintln!("Use 'nostos nostlet list' to see available nostlets");
            return ExitCode::FAILURE;
        }
    };

    let url = match nostlet.get("url").and_then(|u| u.as_str()) {
        Some(u) => u,
        None => {
            eprintln!("Error: Nostlet '{}' has no download URL", name);
            return ExitCode::FAILURE;
        }
    };

    println!("Downloading {} from {}...", name, url);

    // Fetch the nostlet file
    let response = match ureq::get(url).call() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Failed to download nostlet: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let content = match response.into_string() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: Failed to read nostlet: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Create ~/.nostos/nostlets/ directory if it doesn't exist
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => {
            eprintln!("Error: Could not determine home directory");
            return ExitCode::FAILURE;
        }
    };

    let nostlets_dir = home.join(".nostos").join("nostlets");
    if let Err(e) = fs::create_dir_all(&nostlets_dir) {
        eprintln!("Error: Failed to create nostlets directory: {}", e);
        return ExitCode::FAILURE;
    }

    // Write the nostlet file
    let file_path = nostlets_dir.join(format!("{}.nos", name));
    if let Err(e) = fs::write(&file_path, &content) {
        eprintln!("Error: Failed to write nostlet file: {}", e);
        return ExitCode::FAILURE;
    }

    println!("Installed {} to {}", name, file_path.display());
    println!("Open the TUI and press Ctrl+N to use it!");
    ExitCode::SUCCESS
}

/// List locally installed nostlets
fn nostlet_installed() -> ExitCode {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => {
            eprintln!("Error: Could not determine home directory");
            return ExitCode::FAILURE;
        }
    };

    let nostlets_dir = home.join(".nostos").join("nostlets");

    if !nostlets_dir.exists() {
        println!("No nostlets installed.");
        println!("Use 'nostos nostlet list' to see available nostlets");
        return ExitCode::SUCCESS;
    }

    let entries = match fs::read_dir(&nostlets_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error reading nostlets directory: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let mut found = false;
    println!("Installed nostlets in {}:", nostlets_dir.display());
    println!("{:-<60}", "");

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "nos").unwrap_or(false) {
            if let Some(name) = path.file_stem() {
                println!("  {}", name.to_string_lossy());
                found = true;
            }
        }
    }

    if !found {
        println!("  (none)");
    }

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: nostos [options] <command|file.nos> [args...]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  init        Create a new Nostos project");
        eprintln!("  repl        Start the interactive REPL");
        eprintln!("  tui         Start the TUI editor");
        eprintln!("  connect     Connect to a running REPL server");
        eprintln!("  extension   Manage native Rust extensions");
        eprintln!("  nostlet     Manage nostlets (pure Nostos plugins)");
        eprintln!();
        eprintln!("Run a Nostos program file or start the REPL.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --help      Show detailed help message");
        eprintln!("  --version   Show version information");
        return ExitCode::FAILURE;
    }

    // Check for subcommands
    if args.len() >= 2 {
        if args[1] == "repl" {
            return run_repl(&args[2..]);
        }
        if args[1] == "tui" {
            return tui::run_tui(&args[2..]);
        }
        if args[1] == "init" {
            return run_init_command(&args[2..]);
        }
        if args[1] == "nostlet" {
            return run_nostlet_command(&args[2..]);
        }
        if args[1] == "extension" {
            return run_extension_command(&args[2..]);
        }
        if args[1] == "cache" {
            return run_cache_command(&args[2..]);
        }
        if args[1] == "connect" {
            return connect::run_connect(&args[2..]);
        }
    }

    // Parse options
    let mut file_idx = 1;
    let mut enable_jit = true;
    let mut json_errors = false;
    let mut debug_mode = false;
    let mut num_threads: usize = 0; // 0 = auto-detect
    let mut profiling_enabled = false; // Enable function call profiling
    let mut extension_paths: Vec<String> = Vec::new(); // Extension library paths
    let mut use_extensions: Vec<String> = Vec::new(); // Extensions to load by name from ~/.nostos/extensions/
    let mut bin_name: Option<String> = None; // Binary entry point name from [[bin]] in nostos.toml

    let mut i = 1;
    let mut file_idx: Option<usize> = None;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") || arg.starts_with("-") {
            if arg == "--help" || arg == "-h" {
                println!("Nostos - A functional programming language with native extensions");
                println!();
                println!("USAGE:");
                println!("    nostos <file.nos>              Run a single file");
                println!("    nostos <directory/>            Run a project (needs main.nos)");
                println!("    nostos <dir/> --bin NAME       Run specific entry point from project");
                println!("    nostos --use <ext> <file.nos>  Run with an extension");
                println!("    nostos repl                    Start interactive TUI/REPL");
                println!();
                println!("EXAMPLES:");
                println!("    nostos hello.nos                       # Run a program");
                println!("    nostos myproject/                      # Run a project");
                println!("    nostos myproject/ --bin server         # Run 'server' entry point");
                println!("    nostos --use nalgebra script.nos       # Use nalgebra extension");
                println!("    nostos --profile slow_program.nos      # Profile for performance");
                println!();
                println!("EXTENSIONS:");
                println!("    --use NAME        Load installed extension from ~/.nostos/extensions/");
                println!("                      Example: --use nalgebra, --use redis");
                println!("    --extension PATH  Load extension directly from .so/.dylib file");
                println!();
                println!("    Projects can also declare extensions in nostos.toml:");
                println!("        [extensions]");
                println!("        nalgebra = {{ git = \"https://github.com/user/nostos-nalgebra\" }}");
                println!();
                println!("ENTRY POINTS:");
                println!("    --bin NAME, -b    Run specific entry point from [[bin]] in nostos.toml");
                println!("                      Example: --bin server, -b cli");
                println!();
                println!("    Define entry points in nostos.toml:");
                println!("        [[bin]]");
                println!("        name = \"server\"");
                println!("        entry = \"server.main\"");
                println!("        default = true");
                println!();
                println!("PERFORMANCE:");
                println!("    --threads N       Use N worker threads (default: all CPUs)");
                println!("    --profile         Show function call timing after execution");
                println!("    --no-jit          Disable JIT compilation");
                println!();
                println!("DEBUGGING:");
                println!("    --debug           Show local variables in stack traces");
                println!("    --json-errors     Output errors as JSON (for IDE integration)");
                println!();
                println!("COMMANDS:");
                println!("    init [name]       Create a new project (in current dir or new dir)");
                println!("    repl              Start the interactive TUI with editor and REPL");
                println!("    tui               Same as repl");
                println!("    extension install Install a native extension from GitHub");
                println!("    extension list    List installed extensions");
                println!("    nostlet list      List available nostlets from registry");
                println!("    nostlet install   Install a nostlet plugin");
                println!();
                println!("MORE INFO:");
                println!("    --help            Show this help");
                println!("    --build-cache     Build stdlib bytecode cache");
                println!("    --clear-cache     Clear the bytecode cache");
                println!("    --version         Show version");
                println!();
                println!("Documentation: https://pegesund.github.io/nostos/tutorial/24_command_line.html");
                return ExitCode::SUCCESS;
            }
            if arg == "--version" || arg == "-v" {
                println!("nostos {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            if arg == "--no-jit" {
                enable_jit = false;
                i += 1;
                continue;
            }
            if arg == "--build-cache" {
                // Build and save stdlib bytecode cache
                return build_stdlib_cache();
            }
            if arg == "--clear-cache" {
                // Clear the bytecode cache
                return clear_bytecode_cache();
            }
            if arg == "--debug" {
                debug_mode = true;
                i += 1;
                continue;
            }
            if arg == "--json-errors" {
                json_errors = true;
                i += 1;
                continue;
            }
            if arg == "--threads" || arg == "--parallel" || arg == "--parallel-affinity" {
                // Set number of worker threads
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        num_threads = n;
                        i += 2;
                        continue;
                    }
                }
                i += 1;
                continue;
            }
            if arg == "--profile" {
                profiling_enabled = true;
                // JIT profiling is now supported - JIT functions show as "[JIT] function_name"
                i += 1;
                continue;
            }
            if arg == "--extension" || arg == "-e" {
                // Load extension from shared library
                if i + 1 < args.len() {
                    extension_paths.push(args[i + 1].clone());
                    i += 2;
                    continue;
                } else {
                    eprintln!("Error: --extension requires a path argument");
                    return ExitCode::FAILURE;
                }
            }
            if arg == "--use" || arg == "-u" {
                // Load installed extension by name from ~/.nostos/extensions/
                if i + 1 < args.len() {
                    use_extensions.push(args[i + 1].clone());
                    i += 2;
                    continue;
                } else {
                    eprintln!("Error: --use requires an extension name");
                    return ExitCode::FAILURE;
                }
            }
            if arg == "--bin" || arg == "-b" {
                // Specify which binary entry point to run (from [[bin]] in nostos.toml)
                if i + 1 < args.len() {
                    bin_name = Some(args[i + 1].clone());
                    i += 2;
                    continue;
                } else {
                    eprintln!("Error: --bin requires a binary name");
                    return ExitCode::FAILURE;
                }
            }
            i += 1;
        } else {
            // First non-flag argument is the file
            if file_idx.is_none() {
                file_idx = Some(i);
            }
            i += 1;
        }
    }

    let file_idx = match file_idx {
        Some(idx) => idx,
        None => {
            eprintln!("Error: No input file specified");
            eprintln!("Use 'nostos repl' to start the interactive REPL");
            return ExitCode::FAILURE;
        }
    };

    let file_path_arg = &args[file_idx];
    let input_path = std::path::Path::new(file_path_arg);

    // Look for nostos.toml and auto-load extensions
    let search_dir = if input_path.is_dir() {
        std::fs::canonicalize(input_path).unwrap_or_else(|_| input_path.to_path_buf())
    } else {
        let parent = input_path.parent().unwrap_or(std::path::Path::new("."));
        std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf())
    };

    // Store extension module directories for loading later
    let mut extension_module_dirs: Vec<(String, std::path::PathBuf)> = Vec::new();

    // Load project config for [[bin]] entries
    let project_config: Option<nostos_source::ProjectConfig> = if let Some(config_path) = packages::find_config(&search_dir) {
        // Load extensions from old [extensions] section (for backwards compatibility)
        match packages::parse_config(&config_path) {
            Ok(config) => {
                if !config.extensions.is_empty() {
                    match packages::fetch_and_build_all(&config) {
                        Ok(results) => {
                            for result in results {
                                extension_paths.push(result.library_path.to_string_lossy().to_string());
                                extension_module_dirs.push((result.name.clone(), result.module_dir));
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load extensions from [extensions]: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to parse nostos.toml extensions: {}", e);
            }
        }

        // Also load extensions from new [dependencies] section (extension = true)
        let config_dir = config_path.parent().unwrap_or(std::path::Path::new("."));
        if let Ok(manifest) = nostos_packages::PackageManager::load_manifest(config_dir) {
            let pkg_manager = nostos_packages::PackageManager::new();
            for (name, dep) in &manifest.dependencies {
                if dep.is_extension() {
                    match pkg_manager.ensure_extension(name, dep) {
                        Ok(result) => {
                            extension_paths.push(result.library_path.to_string_lossy().to_string());
                            extension_module_dirs.push((result.name.clone(), result.module_dir));
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load extension '{}' from [dependencies]: {}", name, e);
                        }
                    }
                }
            }
        }

        // Load full project config for [[bin]] entries
        match nostos_source::ProjectConfig::load(&config_path) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                eprintln!("Warning: Failed to load project config: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Resolve --use extensions from ~/.nostos/extensions/
    for ext_name in &use_extensions {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => {
                eprintln!("Error: Could not determine home directory");
                return ExitCode::FAILURE;
            }
        };

        // Try different naming conventions for the extension directory
        let ext_dir_candidates = vec![
            home.join(".nostos").join("extensions").join(format!("nostos-{}", ext_name)),
            home.join(".nostos").join("extensions").join(ext_name),
        ];

        let ext_dir = ext_dir_candidates.iter().find(|p| p.exists());
        let ext_dir = match ext_dir {
            Some(d) => d.clone(),
            None => {
                eprintln!("Error: Extension '{}' not found in ~/.nostos/extensions/", ext_name);
                eprintln!("Tried:");
                for c in &ext_dir_candidates {
                    eprintln!("  - {}", c.display());
                }
                eprintln!();
                eprintln!("Install extensions with: nostos nostlet install <name>");
                eprintln!("Or use --extension <path> to load a .so file directly");
                return ExitCode::FAILURE;
            }
        };

        // Find the shared library (.so on Linux, .dylib on macOS)
        let lib_candidates = vec![
            ext_dir.join("target").join("release").join(format!("lib{}.so", ext_name.replace("-", "_"))),
            ext_dir.join("target").join("release").join(format!("libnostos_{}.so", ext_name)),
            ext_dir.join("target").join("release").join(format!("lib{}.dylib", ext_name.replace("-", "_"))),
            ext_dir.join("target").join("release").join(format!("libnostos_{}.dylib", ext_name)),
        ];

        let lib_path = lib_candidates.iter().find(|p| p.exists());
        let lib_path = match lib_path {
            Some(p) => p.clone(),
            None => {
                eprintln!("Error: Extension '{}' library not found. Build it first:", ext_name);
                eprintln!("  cd {} && cargo build --release", ext_dir.display());
                return ExitCode::FAILURE;
            }
        };

        extension_paths.push(lib_path.to_string_lossy().to_string());
        extension_module_dirs.push((ext_name.clone(), ext_dir));
    }

    // Check if input is directory or file
    let mut source_files = Vec::new();
    let project_root;

    if input_path.is_dir() {
        project_root = input_path;

        // Check for main.nos in the directory (unless project has [[bin]] entries)
        let main_file = input_path.join("main.nos");
        let has_bin_entries = project_config.as_ref().map(|c| c.has_bins()).unwrap_or(false);
        if !main_file.exists() && !has_bin_entries {
            eprintln!("Error: No 'main.nos' found in directory '{}'", file_path_arg);
            eprintln!("Projects must have a main.nos file with a main() function,");
            eprintln!("or define entry points with [[bin]] in nostos.toml.");
            return ExitCode::FAILURE;
        }

        match visit_dirs(input_path, &mut source_files) {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error scanning directory '{}': {}", file_path_arg, e);
                return ExitCode::FAILURE;
            }
        }
        if source_files.is_empty() {
            eprintln!("No .nos files found in '{}'", file_path_arg);
            return ExitCode::FAILURE;
        }
    } else {
        project_root = input_path.parent().unwrap_or(std::path::Path::new("."));
        source_files.push(input_path.to_path_buf());
    }

    // Initialize empty compiler
    let mut compiler = Compiler::new_empty();

    // Load extensions once - used for both getting indices AND execution
    // We keep the runtime alive for the duration of the program
    let _ext_runtime: Option<tokio::runtime::Runtime>;
    let ext_mgr: Option<std::sync::Arc<nostos_vm::ExtensionManager>>;

    if !extension_paths.is_empty() {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime for extensions");
        let mgr = std::sync::Arc::new(nostos_vm::ExtensionManager::new(rt.handle().clone()));

        for path in &extension_paths {
            let ext_path = std::path::Path::new(path);
            match mgr.load(ext_path) {
                Ok(msg) => eprintln!("{}", msg),
                Err(e) => {
                    eprintln!("Error loading extension '{}': {}", path, e);
                    return ExitCode::FAILURE;
                }
            }
        }

        // Set extension indices on compiler for CallExtensionIdx optimization
        compiler.set_extension_indices(mgr.get_all_function_indices());
        // Set extension signatures for type inference
        compiler.set_extension_signatures(mgr.get_all_function_decls());
        // Set extension types (opaque types like Tensor, Tokenizer)
        compiler.set_extension_types(mgr.get_all_types());

        // Keep runtime and manager alive
        _ext_runtime = Some(rt);
        ext_mgr = Some(mgr);
    } else {
        _ext_runtime = None;
        ext_mgr = None;
    }

    // Load stdlib
    let stdlib_candidates = vec![
        std::path::PathBuf::from("stdlib"),
        std::path::PathBuf::from("../stdlib"),
    ];
    
    let mut stdlib_path = std::path::PathBuf::from("stdlib");
    let mut found_stdlib = false;
    
    for path in stdlib_candidates {
        if path.is_dir() {
            stdlib_path = path;
            found_stdlib = true;
            break;
        }
    }
    
    if !found_stdlib {
        // Try relative to executable
        if let Ok(mut p) = std::env::current_exe() {
             p.pop(); // remove binary name
             p.pop(); // remove release/debug
             p.pop(); // remove target
             p.push("stdlib");
             if p.is_dir() {
                 stdlib_path = p;
                 found_stdlib = true;
             }
        }
    }

    if found_stdlib {
        let mut stdlib_files = Vec::new();
        visit_dirs(&stdlib_path, &mut stdlib_files).ok();

        // Try to load stdlib from cache
        let cache_result = try_load_stdlib_from_cache(&mut compiler, &stdlib_path);

        if let Some(cache_data) = cache_result {
            // Cache is valid - use prelude imports from cache, skip parsing
            for (local_name, qualified_name) in cache_data.prelude_imports {
                compiler.add_prelude_import(local_name, qualified_name);
            }
        } else {
            // No cache - parse all stdlib files
            // No cache - parse all stdlib files
            let mut stdlib_functions: Vec<(String, String)> = Vec::new();

            for (idx, file_path) in stdlib_files.iter().enumerate() {
                 let file_id = (idx + 1) as u32;
                 let source = fs::read_to_string(file_path).expect("Failed to read stdlib file");
                 let (module_opt, _) = parse(&source);
                 if let Some(mut module) = module_opt {
                     // Set file_id on all spans in the module for unique span identification
                     module.set_file_id(file_id);
                     // Build module path: stdlib.list, stdlib.json, etc.
                     let relative = file_path.strip_prefix(&stdlib_path).unwrap();
                     let mut components: Vec<String> = vec!["stdlib".to_string()];
                     for component in relative.components() {
                         let s = component.as_os_str().to_string_lossy().to_string();
                         if s.ends_with(".nos") {
                             components.push(s.trim_end_matches(".nos").to_string());
                         } else {
                             components.push(s);
                         }
                     }
                     let module_prefix = components.join(".");

                     // Collect function and type names from this module for prelude imports
                     for item in &module.items {
                         match item {
                             nostos_syntax::ast::Item::FnDef(fn_def) => {
                                 let local_name = fn_def.name.node.clone();
                                 let qualified_name = format!("{}.{}", module_prefix, local_name);
                                 stdlib_functions.push((local_name, qualified_name));
                             }
                             nostos_syntax::ast::Item::TypeDef(type_def) => {
                                 // Add public types to the prelude (like Option, Result)
                                 if matches!(type_def.visibility, nostos_syntax::ast::Visibility::Public) {
                                     let local_name = type_def.name.node.clone();
                                     let qualified_name = format!("{}.{}", module_prefix, local_name);
                                     stdlib_functions.push((local_name, qualified_name));
                                 }
                             }
                             _ => {}
                         }
                     }

                     // No cache - full compilation
                     compiler.add_module(&module, components, std::sync::Arc::new(source.clone()), file_path.to_str().unwrap().to_string()).expect("Failed to compile stdlib");
                 }
            }

            // Register prelude imports so stdlib functions are available without prefix
            for (local_name, qualified_name) in stdlib_functions {
                compiler.add_prelude_import(local_name, qualified_name);
            }
        } // end of else (no cache)
    }

    // Load extension modules (.nos wrapper files from extension repos)
    // Track extensions that need caching after compilation
    let mut extensions_to_cache: Vec<(String, PathBuf, String)> = Vec::new(); // (name, first_file_path, source_hash)

    for (ext_name, ext_dir) in &extension_module_dirs {
        // Try to load from cache first
        if let Some(cache_result) = try_load_extension_from_cache(&mut compiler, ext_name, ext_dir) {
            eprintln!("Loaded extension '{}' from cache ({} functions)", ext_name, cache_result.functions_loaded);
            continue;
        }

        // Cache miss - parse and compile
        if let Ok(files) = nostos_packages::PackageManager::list_extension_wrapper_files(ext_dir) {
            let mut first_path: Option<PathBuf> = None;
            let mut combined_hash = String::new();

            for path in files {
                // Compute hash for cache invalidation
                if let Ok(hash) = compute_file_hash(&path) {
                    combined_hash.push_str(&hash);
                }

                if first_path.is_none() {
                    first_path = Some(path.clone());
                }

                if let Ok(source) = fs::read_to_string(&path) {
                    let (module_opt, _errors) = parse(&source);
                    if let Some(module) = module_opt {
                        // Module path is just the extension name (e.g., "glam")
                        let module_path = vec![ext_name.clone()];

                        if let Err(e) = compiler.add_module(&module, module_path, std::sync::Arc::new(source.clone()), path.to_str().unwrap().to_string()) {
                            eprintln!("Warning: Failed to load extension module {}: {}", path.display(), e);
                        }
                    }
                }
            }

            // Mark for caching after compile_all
            if let Some(path) = first_path {
                extensions_to_cache.push((ext_name.clone(), path, combined_hash));
            }
        }
    }

    // Load package dependencies from nostos.toml (source packages only - extensions already loaded)
    // Track packages that need caching after compilation
    let mut packages_to_cache: Vec<(String, Vec<(String, PathBuf, String)>)> = Vec::new(); // (pkg_name, module_infos)

    if let Ok(manifest) = nostos_packages::PackageManager::load_manifest(project_root) {
        // Filter to only source packages (extensions already loaded earlier)
        let source_pkgs: Vec<_> = manifest.dependencies.iter()
            .filter(|(_, dep)| !dep.is_extension())
            .collect();

        if !source_pkgs.is_empty() {
            eprintln!("Loading {} source package dependencies...", source_pkgs.len());
            let pkg_manager = nostos_packages::PackageManager::new();

            for (name, dep) in source_pkgs {
                match pkg_manager.ensure_dependency(name, dep) {
                    Ok(package_path) => {
                        // Try to load from cache first
                        if let Some(cache_result) = try_load_package_from_cache(&mut compiler, name, &package_path) {
                            eprintln!("  Loaded package '{}' from cache ({} modules, {} functions)",
                                name, cache_result.modules_loaded, cache_result.functions_loaded);
                            continue;
                        }

                        // Cache miss - parse and compile all files
                        let mut module_infos: Vec<(String, PathBuf, String)> = Vec::new();

                        if let Ok(files) = nostos_packages::PackageManager::list_package_files(&package_path) {
                            for file_path in files {
                                // Compute hash for cache
                                let source_hash = compute_file_hash(&file_path).unwrap_or_default();

                                if let Ok(source) = fs::read_to_string(&file_path) {
                                    let (module_opt, errors) = parse(&source);
                                    if !errors.is_empty() {
                                        eprintln!("Warning: Parse errors in package file {:?}", file_path);
                                        continue;
                                    }

                                    if let Some(module) = module_opt {
                                        // Module path: package_name.file_name (without .nos)
                                        let file_stem = file_path.file_stem()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("unknown");
                                        let module_path = vec![name.clone(), file_stem.to_string()];
                                        let module_name = format!("{}.{}", name, file_stem);

                                        // Register module with compiler
                                        compiler.register_known_module(&module_name);

                                        // Add module
                                        if let Err(e) = compiler.add_module(
                                            &module,
                                            module_path,
                                            std::sync::Arc::new(source.clone()),
                                            file_path.to_string_lossy().to_string(),
                                        ) {
                                            eprintln!("Warning: Failed to compile package module {}: {}", module_name, e);
                                            continue;
                                        }

                                        // Track for caching
                                        module_infos.push((module_name.clone(), file_path.clone(), source_hash));

                                        eprintln!("  Loaded: {}", module_name);
                                    }
                                }
                            }
                        }

                        // Mark package for caching after compile_all
                        if !module_infos.is_empty() {
                            packages_to_cache.push((name.clone(), module_infos));
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to fetch package {}: {}", name, e);
                    }
                }
            }
        }
    }

    // Try to load project modules from cache (only for directory projects with multiple files)
    let mut project_modules_to_cache: Vec<(String, PathBuf, String)> = Vec::new();
    let project_cache_used = if input_path.is_dir() && source_files.len() > 1 {
        if let Some(cache_result) = try_load_project_from_cache(&mut compiler, project_root, &source_files) {
            eprintln!("Loaded project from cache ({} modules, {} functions)",
                cache_result.modules_loaded, cache_result.functions_loaded);
            true
        } else {
            false
        }
    } else {
        false
    };

    // Process each file (skip if loaded from cache)
    if !project_cache_used {
        for path in &source_files {
            let source = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error reading file '{}': {}", path.display(), e);
                    return ExitCode::FAILURE;
                }
            };

            // Compute hash for caching
            let source_hash = compute_file_hash(path).unwrap_or_default();

            let (module_opt, errors) = parse(&source);
            if !errors.is_empty() {
                let source_errors = parse_errors_to_source_errors(&errors);
                eprint_errors(&source_errors, path.to_str().unwrap_or("unknown"), &source);
                return ExitCode::FAILURE;
            }

            let module = match module_opt {
                Some(m) => m,
                None => {
                    eprintln!("Failed to parse '{}'", path.display());
                    return ExitCode::FAILURE;
                }
            };

            // Determine module path based on file location relative to project root
            // For single file, module path is empty (top-level)
            let module_path = if input_path.is_dir() {
                let relative = path.strip_prefix(project_root).unwrap();
                let mut components: Vec<String> = relative.components()
                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                    .collect();

                // Remove .nos extension from last component
                if let Some(last) = components.last_mut() {
                    if last.ends_with(".nos") {
                        *last = last.trim_end_matches(".nos").to_string();
                    }
                }
                components
            } else {
                vec![]
            };

            // Track for caching (for directory projects)
            if input_path.is_dir() {
                let module_name = module_path.join(".");
                project_modules_to_cache.push((module_name, path.clone(), source_hash));
            }

            // Add to compiler with source tracking
            if let Err(e) = compiler.add_module(&module, module_path, std::sync::Arc::new(source.clone()), path.to_str().unwrap_or("unknown").to_string()) {
                let source_error = e.to_source_error();
                source_error.eprint(path.to_str().unwrap_or("unknown"), &source);
                return ExitCode::FAILURE;
            }
        }
    }

    // Compile all bodies (includes mvar safety check) - skip if loaded from cache
    if !project_cache_used {
        if let Err((e, filename, source)) = compiler.compile_all() {
            let source_error = e.to_source_error();
            source_error.eprint(&filename, &source);
            return ExitCode::FAILURE;
        }
    }

    // Save extension modules to cache (after successful compile_all)
    for (ext_name, source_path, source_hash) in &extensions_to_cache {
        if let Err(e) = save_extension_to_cache(&compiler, ext_name, source_path, source_hash) {
            eprintln!("Warning: Failed to cache extension '{}': {}", ext_name, e);
        }
    }

    // Save source package modules to cache (after successful compile_all)
    for (pkg_name, module_infos) in &packages_to_cache {
        if let Err(e) = save_package_to_cache(&compiler, pkg_name, module_infos) {
            eprintln!("Warning: Failed to cache package '{}': {}", pkg_name, e);
        }
    }

    // Save project modules to cache (after successful compile_all)
    if !project_cache_used && !project_modules_to_cache.is_empty() {
        if let Err(e) = save_project_to_cache(&compiler, project_root, &project_modules_to_cache) {
            eprintln!("Warning: Failed to cache project: {}", e);
        }
    }

    // Resolve entry point (function names now include signature, main has no params so it's "main/")
    let entry_point_name = if input_path.is_dir() {
        let funcs = compiler.get_all_functions();

        // Helper to find function with signature suffix
        let find_func = |base: &str| -> Option<String> {
            let with_slash = format!("{}/", base);
            if funcs.contains_key(&with_slash) {
                Some(with_slash)
            } else {
                funcs.keys()
                    .find(|k| k.starts_with(&with_slash))
                    .cloned()
            }
        };

        // Check if --bin was specified
        if let Some(ref name) = bin_name {
            if let Some(ref cfg) = project_config {
                if let Some(bin_entry) = cfg.get_bin(name) {
                    // Convert "module.func" to "module.func/"
                    match find_func(&bin_entry.entry) {
                        Some(f) => f,
                        None => {
                            eprintln!("Error: Entry point '{}' for bin '{}' not found", bin_entry.entry, name);
                            return ExitCode::FAILURE;
                        }
                    }
                } else {
                    eprintln!("Error: No [[bin]] entry named '{}' in nostos.toml", name);
                    if cfg.has_bins() {
                        eprintln!("Available bins: {}", cfg.bin_names().join(", "));
                    }
                    return ExitCode::FAILURE;
                }
            } else {
                eprintln!("Error: --bin requires a nostos.toml with [[bin]] entries");
                return ExitCode::FAILURE;
            }
        }
        // Check for default bin in project config
        else if let Some(ref cfg) = project_config {
            if let Some(default_bin) = cfg.get_default_bin() {
                match find_func(&default_bin.entry) {
                    Some(f) => f,
                    None => {
                        eprintln!("Error: Default entry point '{}' not found", default_bin.entry);
                        return ExitCode::FAILURE;
                    }
                }
            } else if cfg.has_bins() {
                // Has bins but none is default - require --bin
                eprintln!("Error: Project has [[bin]] entries but none is marked as default.");
                eprintln!("Use --bin NAME to specify which to run. Available: {}", cfg.bin_names().join(", "));
                return ExitCode::FAILURE;
            } else {
                // No bins defined, fall back to main.main or main
                find_func("main.main")
                    .or_else(|| find_func("main"))
                    .unwrap_or_else(|| {
                        eprintln!("Error: No 'main.main' or 'main' function found in project.");
                        std::process::exit(1);
                    })
            }
        }
        // No project config, fall back to main.main or main
        else {
            find_func("main.main")
                .or_else(|| find_func("main"))
                .unwrap_or_else(|| {
                    eprintln!("Error: No 'main.main' or 'main' function found in project.");
                    std::process::exit(1);
                })
        }
    } else {
        "main/".to_string()
    };

    // Get main function
    let main_func = match compiler.get_function(&entry_point_name) {
        Some(func) => func,
        None => {
            eprintln!("Error: Entry point '{}' not found", entry_point_name);
            return ExitCode::FAILURE;
        }
    };

    // Run with AsyncVM
    run_with_async_vm(&compiler, &entry_point_name, profiling_enabled, enable_jit, ext_mgr)
}
