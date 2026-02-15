//! Bytecode caching for faster startup
//!
//! This module provides serialization and deserialization of compiled bytecode
//! to avoid recompiling unchanged modules on every run.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::value::Instruction;

// ============================================================================
// Serializable Value (for constant pools)
// ============================================================================

/// Serializable version of Value for constant pools.
/// Only includes types that can appear in constant pools.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CachedValue {
    Unit,
    Bool(bool),
    Char(char),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    BigInt(String), // Serialize as string
    Decimal(String), // Serialize as string
    String(String),
    List(Vec<CachedValue>),
    Tuple(Vec<CachedValue>),
    // Function references stored as names (for named stdlib functions)
    FunctionRef(String),
    // Inline function (for lambdas that can't be looked up by name)
    InlineFunction(Box<CachedFunction>),
    // Pre-computed record template for fast MakeRecordCached
    RecordTemplate {
        type_name: String,
        field_names: Vec<String>,
        mutable_fields: Vec<bool>,
        discriminant: u16,
    },
    // Pre-computed variant template for fast MakeVariantCached
    VariantTemplate {
        type_name: String,
        constructor: String,
        discriminant: u16,
    },
}

// ============================================================================
// Serializable Chunk
// ============================================================================

/// Serializable version of Chunk
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedChunk {
    pub code: Vec<Instruction>,
    pub constants: Vec<CachedValue>,
    pub lines: Vec<usize>,
    pub locals: Vec<String>,
    pub register_count: usize,
}

// ============================================================================
// Cached Function
// ============================================================================

/// Cached function data - everything needed to reconstruct a FunctionValue
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedFunction {
    pub name: String,
    pub arity: usize,
    pub param_names: Vec<String>,
    pub code: CachedChunk,
    pub module: Option<String>,
    pub source_span: Option<(usize, usize)>,
    pub debug_symbols: Vec<(String, u8, Option<usize>, Option<usize>)>, // (name, register, start, end)
    pub source_file: Option<String>,
    pub doc: Option<String>,
    pub signature: Option<String>,
    pub param_types: Vec<String>,
    pub return_type: Option<String>,
    pub required_params: Option<usize>,
    /// Cached default parameter values (for params after required_params).
    /// None means no default or complex default that requires AST parsing.
    #[serde(default)]
    pub default_values: Vec<Option<CachedValue>>,
    /// Whether this function is public (true) or private (false).
    #[serde(default)]
    pub is_public: bool,
}

// ============================================================================
// Function Signature (for invalidation)
// ============================================================================

/// Function signature for dependency tracking
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub type_params: Vec<String>,
    pub param_types: Vec<String>,
    pub return_type: String,
}

// ============================================================================
// Cached Mvar
// ============================================================================

/// Cached mvar (module-level mutable variable) definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedMvar {
    pub name: String,
    pub type_name: String,
    pub initial_value: CachedMvarValue,
}

/// Serializable mvar initial value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CachedMvarValue {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    EmptyList,
    IntList(Vec<i64>),
    StringList(Vec<String>),
    FloatList(Vec<f64>),
    BoolList(Vec<bool>),
    Tuple(Vec<CachedMvarValue>),
    List(Vec<CachedMvarValue>),
    Record(String, Vec<(String, CachedMvarValue)>),
    EmptyMap,
    Map(Vec<(CachedMvarValue, CachedMvarValue)>),
}

// ============================================================================
// Module Cache
// ============================================================================

/// Cached module containing all compiled functions and types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedModule {
    pub module_path: Vec<String>,
    pub source_hash: String,
    pub functions: Vec<CachedFunction>,
    pub function_signatures: HashMap<String, FunctionSignature>,
    pub exports: Vec<String>,
    /// Prelude imports: (local_name, qualified_name)
    pub prelude_imports: Vec<(String, String)>,
    /// Type definitions from this module
    pub types: Vec<TypeValue>,
    /// Module-level mutable variables (mvars)
    #[serde(default)]
    pub mvars: Vec<CachedMvar>,
    /// Dependency function signatures for validation
    /// Maps: dependency_module_name -> (function_name -> expected_signature)
    /// Used to detect when imported functions change signatures
    #[serde(default)]
    pub dependency_signatures: HashMap<String, HashMap<String, FunctionSignature>>,
}

use crate::value::TypeValue;

// ============================================================================
// Cache Manifest
// ============================================================================

/// Global cache manifest tracking all cached modules
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheManifest {
    pub format_version: u32,
    pub compiler_version: String,
    pub modules: HashMap<String, ModuleCacheInfo>,
    pub dependency_graph: HashMap<String, Vec<String>>,
}

/// Info about a cached module
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleCacheInfo {
    pub source_path: String,
    pub source_hash: String,
    pub cache_path: String,
}

impl CacheManifest {
    pub fn new(compiler_version: &str) -> Self {
        Self {
            format_version: 1,
            compiler_version: compiler_version.to_string(),
            modules: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }
}

// ============================================================================
// Value Conversion
// ============================================================================

use crate::value::Value;
use num_bigint::BigInt;
use rust_decimal::Decimal;
use std::sync::Arc;

impl CachedValue {
    /// Convert a Value to CachedValue for constant pool serialization
    pub fn from_value(value: &Value) -> Option<Self> {
        Self::from_value_with_fn_list(value, &[])
    }

    /// Convert a Value to CachedValue, converting CallDirect → CallByName in inline functions.
    /// The function_list maps function indices to their names.
    pub fn from_value_with_fn_list(value: &Value, function_list: &[String]) -> Option<Self> {
        match value {
            Value::Unit => Some(CachedValue::Unit),
            Value::Bool(b) => Some(CachedValue::Bool(*b)),
            Value::Char(c) => Some(CachedValue::Char(*c)),
            Value::Int8(i) => Some(CachedValue::Int8(*i)),
            Value::Int16(i) => Some(CachedValue::Int16(*i)),
            Value::Int32(i) => Some(CachedValue::Int32(*i)),
            Value::Int64(i) => Some(CachedValue::Int64(*i)),
            Value::UInt8(i) => Some(CachedValue::UInt8(*i)),
            Value::UInt16(i) => Some(CachedValue::UInt16(*i)),
            Value::UInt32(i) => Some(CachedValue::UInt32(*i)),
            Value::UInt64(i) => Some(CachedValue::UInt64(*i)),
            Value::Float32(f) => Some(CachedValue::Float32(*f)),
            Value::Float64(f) => Some(CachedValue::Float64(*f)),
            Value::BigInt(n) => Some(CachedValue::BigInt(n.to_string())),
            Value::Decimal(d) => Some(CachedValue::Decimal(d.to_string())),
            Value::String(s) => Some(CachedValue::String((**s).clone())),
            Value::List(items) => {
                let cached: Option<Vec<_>> = items.iter().map(|v| CachedValue::from_value_with_fn_list(v, function_list)).collect();
                cached.map(CachedValue::List)
            }
            Value::Tuple(items) => {
                let cached: Option<Vec<_>> = items.iter().map(|v| CachedValue::from_value_with_fn_list(v, function_list)).collect();
                cached.map(CachedValue::Tuple)
            }
            Value::Function(f) => {
                // If it's a lambda (anonymous function), cache it inline
                if f.name == "<lambda>" || f.name.contains(".<lambda>") {
                    if function_list.is_empty() {
                        function_to_cached(f).map(|cached_fn| CachedValue::InlineFunction(Box::new(cached_fn)))
                    } else {
                        function_to_cached_with_fn_list(f, function_list).map(|cached_fn| CachedValue::InlineFunction(Box::new(cached_fn)))
                    }
                } else {
                    Some(CachedValue::FunctionRef(f.name.clone()))
                }
            }
            Value::RecordTemplate(t) => Some(CachedValue::RecordTemplate {
                type_name: t.type_name.to_string(),
                field_names: t.field_names.to_vec(),
                mutable_fields: t.mutable_fields.to_vec(),
                discriminant: t.discriminant,
            }),
            Value::VariantTemplate(t) => Some(CachedValue::VariantTemplate {
                type_name: t.type_name.to_string(),
                constructor: t.constructor.to_string(),
                discriminant: t.discriminant,
            }),
            // Other types (closures, records, variants) are not stored in constant pools
            _ => None,
        }
    }

    /// Convert CachedValue back to Value
    pub fn to_value(&self) -> Value {
        match self {
            CachedValue::Unit => Value::Unit,
            CachedValue::Bool(b) => Value::Bool(*b),
            CachedValue::Char(c) => Value::Char(*c),
            CachedValue::Int8(i) => Value::Int8(*i),
            CachedValue::Int16(i) => Value::Int16(*i),
            CachedValue::Int32(i) => Value::Int32(*i),
            CachedValue::Int64(i) => Value::Int64(*i),
            CachedValue::UInt8(i) => Value::UInt8(*i),
            CachedValue::UInt16(i) => Value::UInt16(*i),
            CachedValue::UInt32(i) => Value::UInt32(*i),
            CachedValue::UInt64(i) => Value::UInt64(*i),
            CachedValue::Float32(f) => Value::Float32(*f),
            CachedValue::Float64(f) => Value::Float64(*f),
            CachedValue::BigInt(s) => {
                let n: BigInt = s.parse().unwrap_or_default();
                Value::BigInt(Arc::new(n))
            }
            CachedValue::Decimal(s) => {
                let d: Decimal = s.parse().unwrap_or_default();
                Value::Decimal(d)
            }
            CachedValue::String(s) => Value::String(Arc::new(s.clone())),
            CachedValue::List(items) => {
                let vals: Vec<Value> = items.iter().map(|v| v.to_value()).collect();
                Value::List(Arc::new(vals))
            }
            CachedValue::Tuple(items) => {
                let vals: Vec<Value> = items.iter().map(|v| v.to_value()).collect();
                Value::Tuple(Arc::new(vals))
            }
            CachedValue::FunctionRef(_name) => {
                // Function refs need to be resolved at load time
                // Return Unit as placeholder - caller should handle this
                Value::Unit
            }
            CachedValue::InlineFunction(cached_fn) => {
                // Inline functions (lambdas) are fully cached
                let func = cached_to_function(cached_fn);
                Value::Function(Arc::new(func))
            }
            CachedValue::RecordTemplate { type_name, field_names, mutable_fields, discriminant } => {
                Value::RecordTemplate(Arc::new(crate::value::RecordTemplate {
                    type_name: Arc::from(type_name.as_str()),
                    field_names: Arc::from(field_names.clone()),
                    mutable_fields: Arc::from(mutable_fields.clone()),
                    discriminant: *discriminant,
                }))
            }
            CachedValue::VariantTemplate { type_name, constructor, discriminant } => {
                Value::VariantTemplate(Arc::new(crate::value::VariantTemplate {
                    type_name: Arc::from(type_name.as_str()),
                    constructor: Arc::from(constructor.as_str()),
                    discriminant: *discriminant,
                }))
            }
        }
    }

    /// Convert to Value with function reference resolution
    pub fn to_value_with_resolver<F>(&self, resolver: &F) -> Value
    where
        F: Fn(&str) -> Option<Value>,
    {
        match self {
            CachedValue::FunctionRef(name) => {
                // Try to resolve the function reference
                if let Some(func) = resolver(name) {
                    func
                } else {
                    // Function not found - return Unit (will cause runtime error)
                    Value::Unit
                }
            }
            CachedValue::InlineFunction(cached_fn) => {
                // Inline functions (lambdas) - convert WITH resolver using boxed trait object
                // to avoid recursive type expansion
                #[allow(clippy::type_complexity)]
                let boxed_resolver: Box<dyn Fn(&str) -> Option<Value> + '_> = Box::new(|name| resolver(name));
                let func = cached_to_function_with_boxed_resolver(cached_fn, &boxed_resolver);
                Value::Function(Arc::new(func))
            }
            // For non-function values, use the regular conversion
            _ => self.to_value(),
        }
    }

    /// Convert to Value with boxed function reference resolution
    /// This avoids recursive type expansion when inline functions contain other inline functions
    #[allow(clippy::type_complexity)]
    pub fn to_value_with_boxed_resolver(&self, resolver: &Box<dyn Fn(&str) -> Option<Value> + '_>) -> Value {
        match self {
            CachedValue::FunctionRef(name) => {
                // Try to resolve the function reference
                if let Some(func) = resolver(name) {
                    func
                } else {
                    // Function not found - return Unit (will cause runtime error)
                    Value::Unit
                }
            }
            CachedValue::InlineFunction(cached_fn) => {
                // Inline functions (lambdas) - convert WITH resolver
                let func = cached_to_function_with_boxed_resolver(cached_fn, resolver);
                Value::Function(Arc::new(func))
            }
            // For non-function values, use the regular conversion
            _ => self.to_value(),
        }
    }
}

// ============================================================================
// Chunk Conversion
// ============================================================================

use crate::value::Chunk;

impl CachedChunk {
    /// Convert a Chunk to CachedChunk
    pub fn from_chunk(chunk: &Chunk) -> Option<Self> {
        let constants: Option<Vec<_>> = chunk.constants.iter()
            .map(CachedValue::from_value)
            .collect();

        Some(CachedChunk {
            code: chunk.code.clone(),
            constants: constants?,
            lines: chunk.lines.clone(),
            locals: chunk.locals.clone(),
            register_count: chunk.register_count,
        })
    }

    /// Convert a Chunk to CachedChunk, converting CallDirect/TailCallDirect to CallByName/TailCallByName.
    /// This is needed because CallDirect uses function indices that are assigned at compile time,
    /// but when loading from cache, functions get different indices.
    pub fn from_chunk_with_function_list(chunk: &Chunk, function_list: &[String]) -> Option<Self> {
        let mut constants: Vec<CachedValue> = chunk.constants.iter()
            .map(|v| CachedValue::from_value_with_fn_list(v, function_list))
            .collect::<Option<Vec<_>>>()?;

        // Convert instructions, replacing CallDirect with CallByName
        let code: Vec<Instruction> = chunk.code.iter().map(|instr| {
            match instr {
                Instruction::CallDirect(dst, func_idx, args) => {
                    // Look up function name from index
                    if let Some(func_name) = function_list.get(*func_idx as usize) {
                        // Add function name to constants
                        let name_const_idx = constants.len() as u16;
                        constants.push(CachedValue::String(func_name.clone()));
                        Instruction::CallByName(*dst, name_const_idx, args.clone())
                    } else {
                        // Function index not found - keep original (will fail at runtime anyway)
                        instr.clone()
                    }
                }
                Instruction::TailCallDirect(func_idx, args) => {
                    // Look up function name from index
                    if let Some(func_name) = function_list.get(*func_idx as usize) {
                        // Add function name to constants
                        let name_const_idx = constants.len() as u16;
                        constants.push(CachedValue::String(func_name.clone()));
                        Instruction::TailCallByName(name_const_idx, args.clone())
                    } else {
                        // Function index not found - keep original
                        instr.clone()
                    }
                }
                _ => instr.clone(),
            }
        }).collect();

        Some(CachedChunk {
            code,
            constants,
            lines: chunk.lines.clone(),
            locals: chunk.locals.clone(),
            register_count: chunk.register_count,
        })
    }

    /// Convert CachedChunk back to Chunk
    pub fn to_chunk(&self) -> Chunk {
        Chunk {
            code: self.code.clone(),
            constants: self.constants.iter().map(|v| v.to_value()).collect(),
            lines: self.lines.clone(),
            locals: self.locals.clone(),
            register_count: self.register_count,
        }
    }

    /// Convert CachedChunk back to Chunk with function reference resolution
    pub fn to_chunk_with_resolver<F>(&self, resolver: F) -> Chunk
    where
        F: Fn(&str) -> Option<Value>,
    {
        Chunk {
            code: self.code.clone(),
            constants: self.constants.iter().map(|v| v.to_value_with_resolver(&resolver)).collect(),
            lines: self.lines.clone(),
            locals: self.locals.clone(),
            register_count: self.register_count,
        }
    }

    /// Convert CachedChunk back to Chunk with boxed function reference resolution
    /// This avoids recursive type expansion when inline functions contain other inline functions
    #[allow(clippy::type_complexity)]
    pub fn to_chunk_with_boxed_resolver(&self, resolver: &Box<dyn Fn(&str) -> Option<Value> + '_>) -> Chunk {
        Chunk {
            code: self.code.clone(),
            constants: self.constants.iter().map(|v| v.to_value_with_boxed_resolver(resolver)).collect(),
            lines: self.lines.clone(),
            locals: self.locals.clone(),
            register_count: self.register_count,
        }
    }
}

// ============================================================================
// Cache I/O
// ============================================================================

use std::fs;
use std::path::Path;

/// Save a module cache to disk
pub fn save_module_cache(path: &Path, module: &CachedModule) -> Result<(), String> {
    let bytes = bincode::serialize(module)
        .map_err(|e| format!("Failed to serialize module: {}", e))?;

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;
    }

    fs::write(path, bytes)
        .map_err(|e| format!("Failed to write cache file: {}", e))?;

    Ok(())
}

/// Load a module cache from disk
pub fn load_module_cache(path: &Path) -> Result<CachedModule, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("Failed to read cache file: {}", e))?;

    bincode::deserialize(&bytes)
        .map_err(|e| format!("Failed to deserialize module: {}", e))
}

/// Save the cache manifest
pub fn save_manifest(path: &Path, manifest: &CacheManifest) -> Result<(), String> {
    let bytes = bincode::serialize(manifest)
        .map_err(|e| format!("Failed to serialize manifest: {}", e))?;

    fs::write(path, bytes)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;

    Ok(())
}

/// Load the cache manifest
pub fn load_manifest(path: &Path) -> Result<CacheManifest, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("Failed to read manifest: {}", e))?;

    bincode::deserialize(&bytes)
        .map_err(|e| format!("Failed to deserialize manifest: {}", e))
}

// ============================================================================
// Cache Invalidation
// ============================================================================

/// Compute SHA256 hash of file contents
pub fn compute_file_hash(path: &Path) -> Result<String, String> {
    use sha2::{Sha256, Digest};

    let contents = fs::read(path)
        .map_err(|e| format!("Failed to read file for hashing: {}", e))?;

    let mut hasher = Sha256::new();
    hasher.update(&contents);
    let result = hasher.finalize();

    Ok(format!("{:x}", result))
}

/// Check if a cached module is still valid
pub fn is_cache_valid(
    manifest: &CacheManifest,
    module_name: &str,
    source_path: &Path,
    compiler_version: &str,
) -> bool {
    // Check compiler version
    if manifest.compiler_version != compiler_version {
        return false;
    }

    // Check if module exists in manifest
    let info = match manifest.modules.get(module_name) {
        Some(info) => info,
        None => return false,
    };

    // Check source hash
    let current_hash = match compute_file_hash(source_path) {
        Ok(h) => h,
        Err(_) => return false,
    };

    info.source_hash == current_hash
}

/// Get modules that depend on a given module (for transitive invalidation)
pub fn get_dependents(
    manifest: &CacheManifest,
    module_name: &str,
) -> Vec<String> {
    let mut dependents = Vec::new();

    for (name, deps) in &manifest.dependency_graph {
        if deps.contains(&module_name.to_string()) {
            dependents.push(name.clone());
        }
    }

    dependents
}

/// Get all modules that need to be invalidated when a module changes
pub fn get_transitive_invalidations(
    manifest: &CacheManifest,
    changed_module: &str,
) -> Vec<String> {
    let mut to_invalidate = vec![changed_module.to_string()];
    let mut i = 0;

    while i < to_invalidate.len() {
        let module = &to_invalidate[i].clone();
        let dependents = get_dependents(manifest, module);

        for dep in dependents {
            if !to_invalidate.contains(&dep) {
                to_invalidate.push(dep);
            }
        }

        i += 1;
    }

    to_invalidate
}

// ============================================================================
// BytecodeCache Manager
// ============================================================================

use crate::value::FunctionValue;

/// Bytecode cache manager for a project
pub struct BytecodeCache {
    cache_dir: std::path::PathBuf,
    manifest: CacheManifest,
    compiler_version: String,
}

impl BytecodeCache {
    /// Create a new cache manager
    pub fn new(cache_dir: std::path::PathBuf, compiler_version: &str) -> Self {
        let manifest_path = cache_dir.join("manifest.bin");
        let manifest = if manifest_path.exists() {
            load_manifest(&manifest_path).unwrap_or_else(|_| CacheManifest::new(compiler_version))
        } else {
            CacheManifest::new(compiler_version)
        };

        BytecodeCache {
            cache_dir,
            manifest,
            compiler_version: compiler_version.to_string(),
        }
    }

    /// Check if a module's cache is valid
    pub fn is_module_valid(&self, module_name: &str, source_path: &Path) -> bool {
        is_cache_valid(&self.manifest, module_name, source_path, &self.compiler_version)
    }

    /// Get the cache path for a module
    pub fn get_cache_path(&self, module_name: &str) -> std::path::PathBuf {
        let safe_name = module_name.replace('.', "/");
        self.cache_dir.join(format!("{}.cache", safe_name))
    }

    /// Load a cached module
    pub fn load_module(&self, module_name: &str) -> Result<CachedModule, String> {
        let cache_path = self.get_cache_path(module_name);
        load_module_cache(&cache_path)
    }

    /// Save a module to cache
    pub fn save_module(
        &mut self,
        module_name: &str,
        source_path: &str,
        module: &CachedModule,
    ) -> Result<(), String> {
        let cache_path = self.get_cache_path(module_name);
        save_module_cache(&cache_path, module)?;

        // Update manifest
        self.manifest.modules.insert(
            module_name.to_string(),
            ModuleCacheInfo {
                source_path: source_path.to_string(),
                source_hash: module.source_hash.clone(),
                cache_path: cache_path.to_string_lossy().to_string(),
            },
        );

        Ok(())
    }

    /// Update module dependencies in the manifest
    pub fn set_dependencies(&mut self, module_name: &str, dependencies: Vec<String>) {
        self.manifest.dependency_graph.insert(module_name.to_string(), dependencies);
    }

    /// Save the manifest to disk
    pub fn save_manifest(&self) -> Result<(), String> {
        let manifest_path = self.cache_dir.join("manifest.bin");
        save_manifest(&manifest_path, &self.manifest)
    }

    /// Get modules that need recompilation when a module changes
    pub fn get_invalidated_modules(&self, changed_module: &str) -> Vec<String> {
        get_transitive_invalidations(&self.manifest, changed_module)
    }

    /// Clear the entire cache
    pub fn clear(&mut self) -> Result<(), String> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| format!("Failed to clear cache: {}", e))?;
        }
        self.manifest = CacheManifest::new(&self.compiler_version);
        Ok(())
    }

    /// Check if cache directory exists and has content
    pub fn has_cache(&self) -> bool {
        self.cache_dir.exists() && !self.manifest.modules.is_empty()
    }

    /// Get the manifest (for reading dependency info)
    pub fn manifest(&self) -> &CacheManifest {
        &self.manifest
    }

    /// Get source hash for a module from manifest
    pub fn get_source_hash(&self, module_name: &str) -> Option<&str> {
        self.manifest.modules.get(module_name).map(|info| info.source_hash.as_str())
    }
}

// ============================================================================
// Two-Tier Module Cache (Memory + Disk)
// ============================================================================

/// In-memory compiled module data
/// This is what we keep in memory for fast access during a session
#[derive(Clone)]
pub struct CompiledModuleData {
    /// The cached module (functions, types, etc.)
    pub cached: CachedModule,
    /// Dependencies of this module (module names it imports)
    pub dependencies: Vec<String>,
}

/// Two-tier module cache: fast in-memory + persistent disk
///
/// Design principles:
/// - Memory tier: Always used during session, no disk I/O on edits
/// - Disk tier: Loaded on startup, written on exit/save (not on every change)
/// - Dirty tracking: Know which modules need persisting
pub struct ModuleCache {
    /// In-memory compiled modules (hot cache)
    memory: HashMap<String, CompiledModuleData>,
    /// Content hash when module was compiled (for staleness check)
    memory_hashes: HashMap<String, String>,
    /// Modules that changed since last disk persist
    dirty: HashSet<String>,
    /// Disk cache (cold storage)
    disk: Option<BytecodeCache>,
    /// Project root directory (for per-project cache)
    project_root: Option<std::path::PathBuf>,
}

impl ModuleCache {
    /// Create a new module cache without disk backing (memory-only mode)
    pub fn new_memory_only(_compiler_version: &str) -> Self {
        ModuleCache {
            memory: HashMap::new(),
            memory_hashes: HashMap::new(),
            dirty: HashSet::new(),
            disk: None,
            project_root: None,
        }
    }

    /// Create a new module cache with disk backing for a project
    pub fn new_with_disk(project_root: std::path::PathBuf, compiler_version: &str) -> Self {
        let cache_dir = project_root.join(".nostos-cache");
        let disk = BytecodeCache::new(cache_dir, compiler_version);

        ModuleCache {
            memory: HashMap::new(),
            memory_hashes: HashMap::new(),
            dirty: HashSet::new(),
            disk: Some(disk),
            project_root: Some(project_root),
        }
    }

    /// Create with global stdlib cache (for stdlib loading)
    pub fn new_with_global_cache(cache_dir: std::path::PathBuf, compiler_version: &str) -> Self {
        let disk = BytecodeCache::new(cache_dir, compiler_version);

        ModuleCache {
            memory: HashMap::new(),
            memory_hashes: HashMap::new(),
            dirty: HashSet::new(),
            disk: Some(disk),
            project_root: None,
        }
    }

    /// Check if a module is in memory and up-to-date
    pub fn get_from_memory(&self, module_name: &str, source_hash: &str) -> Option<&CompiledModuleData> {
        if let Some(cached_hash) = self.memory_hashes.get(module_name) {
            if cached_hash == source_hash {
                return self.memory.get(module_name);
            }
        }
        None
    }

    /// Try to get a module from cache (memory first, then disk)
    /// Returns None if not cached or stale
    pub fn get(&mut self, module_name: &str, source_hash: &str) -> Option<CompiledModuleData> {
        // Try memory first (hot path)
        if let Some(data) = self.get_from_memory(module_name, source_hash) {
            return Some(data.clone());
        }

        // Try disk cache
        if let Some(ref disk) = self.disk {
            if let Some(disk_hash) = disk.get_source_hash(module_name) {
                if disk_hash == source_hash {
                    // Load from disk into memory
                    if let Ok(cached) = disk.load_module(module_name) {
                        let deps = disk.manifest()
                            .dependency_graph
                            .get(module_name)
                            .cloned()
                            .unwrap_or_default();

                        let data = CompiledModuleData {
                            cached,
                            dependencies: deps,
                        };

                        // Populate memory cache
                        self.memory.insert(module_name.to_string(), data.clone());
                        self.memory_hashes.insert(module_name.to_string(), source_hash.to_string());
                        // Not dirty - just loaded from disk

                        return Some(data);
                    }
                }
            }
        }

        None
    }

    /// Store a compiled module in memory (fast, no disk I/O)
    /// Call `persist_dirty()` later to write to disk
    pub fn store(&mut self, module_name: &str, source_hash: &str, data: CompiledModuleData) {
        self.memory.insert(module_name.to_string(), data);
        self.memory_hashes.insert(module_name.to_string(), source_hash.to_string());
        self.dirty.insert(module_name.to_string());
    }

    /// Invalidate a module (and optionally its dependents)
    pub fn invalidate(&mut self, module_name: &str, transitive: bool) {
        let to_invalidate = if transitive {
            self.get_dependents(module_name)
        } else {
            vec![module_name.to_string()]
        };

        for name in to_invalidate {
            self.memory.remove(&name);
            self.memory_hashes.remove(&name);
            self.dirty.remove(&name);
        }
    }

    /// Get all modules that depend on a given module
    fn get_dependents(&self, module_name: &str) -> Vec<String> {
        let mut dependents = vec![module_name.to_string()];
        let mut i = 0;

        while i < dependents.len() {
            let current = dependents[i].clone();

            // Check memory for dependents
            for (name, data) in &self.memory {
                if data.dependencies.contains(&current) && !dependents.contains(name) {
                    dependents.push(name.clone());
                }
            }

            // Check disk manifest for dependents
            if let Some(ref disk) = self.disk {
                for (name, deps) in &disk.manifest().dependency_graph {
                    if deps.contains(&current) && !dependents.contains(name) {
                        dependents.push(name.clone());
                    }
                }
            }

            i += 1;
        }

        dependents
    }

    /// Persist all dirty modules to disk
    /// Call this on exit, explicit save, or run completion
    pub fn persist_dirty(&mut self) -> Result<usize, String> {
        let disk = match self.disk.as_mut() {
            Some(d) => d,
            None => return Ok(0), // Memory-only mode
        };

        let mut count = 0;
        let dirty: Vec<String> = self.dirty.iter().cloned().collect();

        for module_name in dirty {
            if let Some(data) = self.memory.get(&module_name) {
                if let Some(hash) = self.memory_hashes.get(&module_name) {
                    // Determine source path (use module name as fallback)
                    let source_path = if let Some(ref root) = self.project_root {
                        root.join(format!("{}.nos", module_name.replace('.', "/"))).to_string_lossy().to_string()
                    } else {
                        format!("{}.nos", module_name.replace('.', "/"))
                    };

                    // Save to disk
                    let mut cached = data.cached.clone();
                    cached.source_hash = hash.clone();

                    disk.save_module(&module_name, &source_path, &cached)?;
                    disk.set_dependencies(&module_name, data.dependencies.clone());
                    count += 1;
                }
            }
        }

        if count > 0 {
            disk.save_manifest()?;
        }

        self.dirty.clear();
        Ok(count)
    }

    /// Check if there are dirty (unsaved) modules
    pub fn has_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Get list of dirty module names
    pub fn dirty_modules(&self) -> Vec<String> {
        self.dirty.iter().cloned().collect()
    }

    /// Clear all in-memory state (but keep disk cache)
    pub fn clear_memory(&mut self) {
        self.memory.clear();
        self.memory_hashes.clear();
        self.dirty.clear();
    }

    /// Clear everything including disk cache
    pub fn clear_all(&mut self) -> Result<(), String> {
        self.clear_memory();
        if let Some(ref mut disk) = self.disk {
            disk.clear()?;
        }
        Ok(())
    }

    /// Get all cached module names (memory + disk)
    pub fn cached_modules(&self) -> Vec<String> {
        let mut names: HashSet<String> = self.memory.keys().cloned().collect();
        if let Some(ref disk) = self.disk {
            for name in disk.manifest().modules.keys() {
                names.insert(name.clone());
            }
        }
        names.into_iter().collect()
    }

    /// Check if disk cache exists and has content
    pub fn has_disk_cache(&self) -> bool {
        self.disk.as_ref().map(|d| d.has_cache()).unwrap_or(false)
    }
}

/// Helper to convert a FunctionValue to CachedFunction
pub fn function_to_cached(func: &FunctionValue) -> Option<CachedFunction> {
    let code = CachedChunk::from_chunk(&func.code)?;

    // Convert debug symbols to serializable form
    let debug_symbols: Vec<(String, u8, Option<usize>, Option<usize>)> = func.debug_symbols
        .iter()
        .map(|s| (s.name.clone(), s.register, None, None))
        .collect();

    Some(CachedFunction {
        name: func.name.clone(),
        arity: func.arity,
        param_names: func.param_names.clone(),
        code,
        module: func.module.clone(),
        source_span: func.source_span,
        debug_symbols,
        source_file: func.source_file.clone(),
        doc: func.doc.clone(),
        signature: func.signature.clone(),
        param_types: func.param_types.clone(),
        return_type: func.return_type.clone(),
        required_params: func.required_params,
        default_values: vec![],  // Populated separately from AST if available
        is_public: false,  // Default to private, will be updated from AST
    })
}

/// Helper to convert a FunctionValue to CachedFunction with CallDirect→CallByName conversion.
/// This should be used when caching functions that may contain CallDirect instructions,
/// as their function indices are not stable across cache load/save.
pub fn function_to_cached_with_fn_list(func: &FunctionValue, function_list: &[String]) -> Option<CachedFunction> {
    let code = CachedChunk::from_chunk_with_function_list(&func.code, function_list)?;

    // Convert debug symbols to serializable form
    let debug_symbols: Vec<(String, u8, Option<usize>, Option<usize>)> = func.debug_symbols
        .iter()
        .map(|s| (s.name.clone(), s.register, None, None))
        .collect();

    Some(CachedFunction {
        name: func.name.clone(),
        arity: func.arity,
        param_names: func.param_names.clone(),
        code,
        module: func.module.clone(),
        source_span: func.source_span,
        debug_symbols,
        source_file: func.source_file.clone(),
        doc: func.doc.clone(),
        signature: func.signature.clone(),
        param_types: func.param_types.clone(),
        return_type: func.return_type.clone(),
        required_params: func.required_params,
        default_values: vec![],  // Populated separately from AST if available
        is_public: false,  // Default to private, will be updated from AST
    })
}

use crate::value::LocalVarSymbol;

/// Helper to convert a CachedFunction back to FunctionValue
pub fn cached_to_function(cached: &CachedFunction) -> FunctionValue {
    // Convert debug symbols back
    let debug_symbols: Vec<LocalVarSymbol> = cached.debug_symbols
        .iter()
        .map(|(name, register, _, _)| LocalVarSymbol {
            name: name.clone(),
            register: *register,
        })
        .collect();

    FunctionValue {
        name: cached.name.clone(),
        arity: cached.arity,
        param_names: cached.param_names.clone(),
        code: Arc::new(cached.code.to_chunk()),
        module: cached.module.clone(),
        source_span: cached.source_span,
        jit_code: None,
        call_count: std::sync::atomic::AtomicU32::new(0),
        debug_symbols,
        source_code: None, // Source code not cached
        source_file: cached.source_file.clone(),
        doc: cached.doc.clone(),
        signature: cached.signature.clone(),
        param_types: cached.param_types.clone(),
        return_type: cached.return_type.clone(),
        required_params: cached.required_params,
    }
}

/// Helper to convert a CachedFunction back to FunctionValue with function reference resolution
pub fn cached_to_function_with_resolver<F>(cached: &CachedFunction, resolver: F) -> FunctionValue
where
    F: Fn(&str) -> Option<Value>,
{
    // Convert debug symbols back
    let debug_symbols: Vec<LocalVarSymbol> = cached.debug_symbols
        .iter()
        .map(|(name, register, _, _)| LocalVarSymbol {
            name: name.clone(),
            register: *register,
        })
        .collect();

    FunctionValue {
        name: cached.name.clone(),
        arity: cached.arity,
        param_names: cached.param_names.clone(),
        code: Arc::new(cached.code.to_chunk_with_resolver(resolver)),
        module: cached.module.clone(),
        source_span: cached.source_span,
        jit_code: None,
        call_count: std::sync::atomic::AtomicU32::new(0),
        debug_symbols,
        source_code: None, // Source code not cached
        source_file: cached.source_file.clone(),
        doc: cached.doc.clone(),
        signature: cached.signature.clone(),
        param_types: cached.param_types.clone(),
        return_type: cached.return_type.clone(),
        required_params: cached.required_params,
    }
}

/// Helper to convert a CachedFunction back to FunctionValue with a boxed resolver
/// This avoids recursive type expansion when inline functions contain other inline functions
#[allow(clippy::type_complexity)]
pub fn cached_to_function_with_boxed_resolver(
    cached: &CachedFunction,
    resolver: &Box<dyn Fn(&str) -> Option<Value> + '_>,
) -> FunctionValue {
    // Convert debug symbols back
    let debug_symbols: Vec<LocalVarSymbol> = cached.debug_symbols
        .iter()
        .map(|(name, register, _, _)| LocalVarSymbol {
            name: name.clone(),
            register: *register,
        })
        .collect();

    FunctionValue {
        name: cached.name.clone(),
        arity: cached.arity,
        param_names: cached.param_names.clone(),
        code: Arc::new(cached.code.to_chunk_with_boxed_resolver(resolver)),
        module: cached.module.clone(),
        source_span: cached.source_span,
        jit_code: None,
        call_count: std::sync::atomic::AtomicU32::new(0),
        debug_symbols,
        source_code: None, // Source code not cached
        source_file: cached.source_file.clone(),
        doc: cached.doc.clone(),
        signature: cached.signature.clone(),
        param_types: cached.param_types.clone(),
        return_type: cached.return_type.clone(),
        required_params: cached.required_params,
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Instruction, RegList, Chunk};

    #[test]
    fn test_instruction_serialization() {
        // Test various instruction types
        let instructions = vec![
            Instruction::LoadConst(0, 1),
            Instruction::Move(1, 2),
            Instruction::AddInt(3, 4, 5),
            Instruction::Jump(10),
            Instruction::Call(0, 1, RegList::from(vec![2u8, 3, 4])),
            Instruction::Return(0),
        ];

        for instr in &instructions {
            let bytes = bincode::serialize(instr).expect("Failed to serialize instruction");
            let restored: Instruction = bincode::deserialize(&bytes).expect("Failed to deserialize instruction");
            assert_eq!(format!("{:?}", instr), format!("{:?}", restored));
        }
    }

    #[test]
    fn test_cached_value_roundtrip() {
        let values = vec![
            CachedValue::Unit,
            CachedValue::Bool(true),
            CachedValue::Int64(42),
            CachedValue::Float64(3.14),
            CachedValue::String("hello".to_string()),
            CachedValue::List(vec![
                CachedValue::Int64(1),
                CachedValue::Int64(2),
                CachedValue::Int64(3),
            ]),
        ];

        for val in &values {
            let bytes = bincode::serialize(val).expect("Failed to serialize value");
            let restored: CachedValue = bincode::deserialize(&bytes).expect("Failed to deserialize value");
            assert_eq!(format!("{:?}", val), format!("{:?}", restored));
        }
    }

    #[test]
    fn test_cached_chunk_roundtrip() {
        let mut chunk = Chunk::new();
        chunk.emit(Instruction::LoadConst(0, 0), 1);
        chunk.emit(Instruction::Return(0), 1);
        chunk.add_constant(Value::Int64(42));
        chunk.register_count = 1;

        let cached = CachedChunk::from_chunk(&chunk).expect("Failed to convert chunk");
        let bytes = bincode::serialize(&cached).expect("Failed to serialize chunk");
        let restored: CachedChunk = bincode::deserialize(&bytes).expect("Failed to deserialize chunk");

        assert_eq!(cached.code.len(), restored.code.len());
        assert_eq!(cached.constants.len(), restored.constants.len());
        assert_eq!(cached.register_count, restored.register_count);
    }

    #[test]
    fn test_cached_module_roundtrip() {
        let module = CachedModule {
            module_path: vec!["stdlib".to_string(), "list".to_string()],
            source_hash: "abc123".to_string(),
            functions: vec![],
            function_signatures: HashMap::new(),
            exports: vec!["map".to_string(), "filter".to_string()],
            prelude_imports: vec![],
            types: vec![],
            mvars: vec![],
            dependency_signatures: HashMap::new(),
        };

        let bytes = bincode::serialize(&module).expect("Failed to serialize module");
        let restored: CachedModule = bincode::deserialize(&bytes).expect("Failed to deserialize module");

        assert_eq!(module.module_path, restored.module_path);
        assert_eq!(module.source_hash, restored.source_hash);
        assert_eq!(module.exports, restored.exports);
    }

    #[test]
    fn test_cache_manifest() {
        let mut manifest = CacheManifest::new("0.1.0");
        manifest.modules.insert(
            "stdlib.list".to_string(),
            ModuleCacheInfo {
                source_path: "stdlib/list.nos".to_string(),
                source_hash: "abc123".to_string(),
                cache_path: ".nostos-cache/stdlib/list.cache".to_string(),
            },
        );
        manifest.dependency_graph.insert(
            "user.main".to_string(),
            vec!["stdlib.list".to_string()],
        );

        let bytes = bincode::serialize(&manifest).expect("Failed to serialize manifest");
        let restored: CacheManifest = bincode::deserialize(&bytes).expect("Failed to deserialize manifest");

        assert_eq!(manifest.compiler_version, restored.compiler_version);
        assert_eq!(manifest.modules.len(), restored.modules.len());
        assert_eq!(manifest.dependency_graph.len(), restored.dependency_graph.len());
    }

    #[test]
    fn test_transitive_invalidation() {
        let mut manifest = CacheManifest::new("0.1.0");

        // Setup: user.main depends on stdlib.list
        //        stdlib.json depends on stdlib.list
        manifest.dependency_graph.insert(
            "user.main".to_string(),
            vec!["stdlib.list".to_string()],
        );
        manifest.dependency_graph.insert(
            "stdlib.json".to_string(),
            vec!["stdlib.list".to_string()],
        );
        manifest.dependency_graph.insert(
            "stdlib.list".to_string(),
            vec![],
        );

        // If stdlib.list changes, both user.main and stdlib.json should be invalidated
        let invalidated = get_transitive_invalidations(&manifest, "stdlib.list");
        assert!(invalidated.contains(&"stdlib.list".to_string()));
        assert!(invalidated.contains(&"user.main".to_string()));
        assert!(invalidated.contains(&"stdlib.json".to_string()));
    }
}
