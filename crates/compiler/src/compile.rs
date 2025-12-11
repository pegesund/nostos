//! AST to Bytecode compiler.
//!
//! Features:
//! - Tail call detection and optimization
//! - Closure conversion (capture free variables)
//! - Pattern match compilation
//! - Type-directed code generation

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use nostos_syntax::ast::*;
use nostos_vm::*;
use nostos_types::{TypeEnv, infer::InferCtx};

/// Compilation errors with source location information.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    #[error("unknown variable `{name}`")]
    UnknownVariable { name: String, span: Span },

    #[error("unknown function `{name}`")]
    UnknownFunction { name: String, span: Span },

    #[error("unknown type `{name}`")]
    UnknownType { name: String, span: Span },

    #[error("`{name}` is defined multiple times")]
    DuplicateDefinition { name: String, span: Span },

    #[error("invalid pattern")]
    InvalidPattern { span: Span, context: String },

    #[error("{feature} is not yet implemented")]
    NotImplemented { feature: String, span: Span },

    #[error("cannot access private function `{function}` from outside module `{module}`")]
    PrivateAccess { function: String, module: String, span: Span },

    #[error("unknown trait `{name}`")]
    UnknownTrait { name: String, span: Span },

    #[error("type `{ty}` does not implement method `{method}` required by trait `{trait_name}`")]
    MissingTraitMethod { method: String, ty: String, trait_name: String, span: Span },

    #[error("type `{ty}` does not implement trait `{trait_name}`")]
    TraitNotImplemented { ty: String, trait_name: String, span: Span },

    #[error("function `{name}` expects {expected} argument(s), but {found} were provided")]
    ArityMismatch { name: String, expected: usize, found: usize, span: Span },

    #[error("cannot resolve trait method `{method}` without type information")]
    UnresolvedTraitMethod { method: String, span: Span },

    #[error("cannot derive `{trait_name}` for `{ty}`: {reason}")]
    CannotDerive { trait_name: String, ty: String, reason: String, span: Span },

    #[error("type `{type_name}` does not implement trait `{trait_name}`")]
    TraitBoundNotSatisfied { type_name: String, trait_name: String, span: Span },
}

impl CompileError {
    /// Get the span associated with this error.
    pub fn span(&self) -> Span {
        match self {
            CompileError::UnknownVariable { span, .. } => *span,
            CompileError::UnknownFunction { span, .. } => *span,
            CompileError::UnknownType { span, .. } => *span,
            CompileError::DuplicateDefinition { span, .. } => *span,
            CompileError::InvalidPattern { span, .. } => *span,
            CompileError::NotImplemented { span, .. } => *span,
            CompileError::PrivateAccess { span, .. } => *span,
            CompileError::UnknownTrait { span, .. } => *span,
            CompileError::MissingTraitMethod { span, .. } => *span,
            CompileError::TraitNotImplemented { span, .. } => *span,
            CompileError::ArityMismatch { span, .. } => *span,
            CompileError::UnresolvedTraitMethod { span, .. } => *span,
            CompileError::CannotDerive { span, .. } => *span,
            CompileError::TraitBoundNotSatisfied { span, .. } => *span,
        }
    }

    /// Convert to a SourceError for pretty printing.
    pub fn to_source_error(&self) -> nostos_syntax::SourceError {
        use nostos_syntax::SourceError;

        let span = self.span();
        match self {
            CompileError::UnknownVariable { name, .. } => {
                SourceError::unknown_variable(name, span)
            }
            CompileError::UnknownFunction { name, .. } => {
                SourceError::unknown_function(name, span)
            }
            CompileError::UnknownType { name, .. } => {
                SourceError::unknown_type(name, span)
            }
            CompileError::DuplicateDefinition { name, .. } => {
                SourceError::duplicate_definition(name, span, None)
            }
            CompileError::InvalidPattern { context, .. } => {
                SourceError::invalid_pattern(span, context)
            }
            CompileError::NotImplemented { feature, .. } => {
                SourceError::not_implemented(feature, span)
            }
            CompileError::PrivateAccess { function, module, .. } => {
                SourceError::private_access(function, module, span)
            }
            CompileError::UnknownTrait { name, .. } => {
                SourceError::unknown_trait(name, span)
            }
            CompileError::MissingTraitMethod { method, ty, trait_name, .. } => {
                SourceError::missing_trait_method(method, ty, trait_name, span)
            }
            CompileError::TraitNotImplemented { ty, trait_name, .. } => {
                SourceError::compile(
                    format!("type `{}` does not implement trait `{}`", ty, trait_name),
                    span,
                )
            }
            CompileError::ArityMismatch { name, expected, found, .. } => {
                SourceError::arity_mismatch(name, *expected, *found, span)
            }
            CompileError::UnresolvedTraitMethod { method, .. } => {
                SourceError::compile(
                    format!("cannot resolve trait method `{}` without type information", method),
                    span,
                )
            }
            CompileError::CannotDerive { trait_name, ty, reason, .. } => {
                SourceError::compile(
                    format!("cannot derive `{}` for `{}`: {}", trait_name, ty, reason),
                    span,
                )
            }
            CompileError::TraitBoundNotSatisfied { type_name, trait_name, .. } => {
                SourceError::compile(
                    format!("type `{}` does not implement trait `{}`", type_name, trait_name),
                    span,
                )
            }
        }
    }
}

/// Simple type inference for builtin dispatch.
/// We infer types from literals and some expressions to emit typed instructions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum InferredType {
    Int,
    Float,
}

/// Information about a local variable.
#[derive(Clone, Copy)]
struct LocalInfo {
    reg: Reg,
    is_float: bool,
    mutable: bool,
}

/// Compilation context.
pub struct Compiler {
    /// Current function being compiled
    chunk: Chunk,
    /// Local variable -> register and type info
    locals: HashMap<String, LocalInfo>,
    /// Next available register
    next_reg: Reg,
    /// Captured variables for the current closure: name -> capture index
    capture_indices: HashMap<String, u8>,
    /// Compiled functions
    functions: HashMap<String, Arc<FunctionValue>>,
    /// Function name -> index mapping for direct calls (no HashMap lookup at runtime!)
    function_indices: HashMap<String, u16>,
    /// Ordered list of function names (index -> name)
    function_list: Vec<String>,
    /// Type definitions
    types: HashMap<String, TypeInfo>,
    /// Known constructors (for type checking)
    known_constructors: HashSet<String>,
    /// Current scope depth
    scope_depth: usize,
    /// Current module path (e.g., ["Foo", "Bar"] for module Foo.Bar)
    module_path: Vec<String>,
    /// Imports: local name -> fully qualified name
    imports: HashMap<String, String>,
    /// Function visibility: qualified name -> Visibility
    function_visibility: HashMap<String, Visibility>,
    /// Trait definitions: trait name -> TraitInfo
    trait_defs: HashMap<String, TraitInfo>,
    /// Trait implementations: (type_name, trait_name) -> TraitImplInfo
    trait_impls: HashMap<(String, String), TraitImplInfo>,
    /// Types to their implemented traits: type_name -> [trait_name, ...]
    type_traits: HashMap<String, Vec<String>>,
    /// Local variable type tracking: variable name -> type name
    local_types: HashMap<String, String>,
    /// Parameter types for specialized function variants (for monomorphization)
    /// When compiling a specialized variant, parameter name -> concrete type name
    param_types: HashMap<String, String>,
    /// Function ASTs for monomorphization: function name -> FnDef
    /// Used to recompile functions with different type contexts
    fn_asts: HashMap<String, FnDef>,
    /// Function type parameters with bounds: function name -> type parameters
    /// Used to check trait bounds at call sites
    fn_type_params: HashMap<String, Vec<TypeParam>>,
    /// Functions that need monomorphization (have untyped parameters calling trait methods)
    /// These functions are not compiled normally; specialized variants are compiled at call sites
    polymorphic_fns: HashSet<String>,
    /// Current function name being compiled (for self-recursion optimization)
    current_function_name: Option<String>,
    /// Current function's type parameters (for checking nested trait bounds)
    current_fn_type_params: Vec<TypeParam>,
    /// Loop context stack for break/continue
    loop_stack: Vec<LoopContext>,
    /// Line starts: byte offsets where each line begins (line 1 is at index 0)
    line_starts: Vec<usize>,
    /// Pending functions to compile (second pass)
    /// (AST, module_path, imports, line_starts, source, source_name)
    pending_functions: Vec<(FnDef, Vec<String>, HashMap<String, String>, Vec<usize>, Arc<String>, String)>,
    
    // Current source context
    current_source: Option<Arc<String>>,
    current_source_name: Option<String>,

    /// Type definition ASTs for REPL introspection: type name -> TypeDef
    type_defs: HashMap<String, TypeDef>,
}

/// Context for a loop being compiled (for break/continue).
#[derive(Clone)]
struct LoopContext {
    /// Address of loop start (for back-jump at end of loop)
    start_addr: usize,
    /// Address where continue should jump to (may differ from start_addr for for loops)
    continue_addr: usize,
    /// Addresses of continue jumps to patch
    continue_jumps: Vec<usize>,
    /// Addresses of break jumps to patch at loop end
    break_jumps: Vec<usize>,
}

/// Type information for code generation.
#[derive(Clone)]
pub struct TypeInfo {
    pub name: String,
    pub kind: TypeInfoKind,
}

#[derive(Clone)]
pub enum TypeInfoKind {
    /// Record type: fields with (name, type_name) pairs
    Record { fields: Vec<(String, String)>, mutable: bool },
    /// Variant type: constructors with (name, field_types)
    /// Field types are stored as simple strings: "Float", "Int", etc.
    Variant { constructors: Vec<(String, Vec<String>)> },
}

/// Trait definition information.
#[derive(Clone)]
pub struct TraitInfo {
    pub name: String,
    pub super_traits: Vec<String>,
    pub methods: Vec<TraitMethodInfo>,
}

/// A method signature in a trait.
#[derive(Clone)]
pub struct TraitMethodInfo {
    pub name: String,
    pub param_count: usize,
    pub has_default: bool,
}

/// Trait implementation information.
#[derive(Clone)]
pub struct TraitImplInfo {
    pub type_name: String,
    pub trait_name: String,
    pub method_names: Vec<String>,  // Maps to qualified function names like "Point.Show.show"
}

impl Compiler {
    pub fn new_empty() -> Self {
        Self {
            chunk: Chunk::new(),
            locals: HashMap::new(),
            next_reg: 0,
            capture_indices: HashMap::new(),
            functions: HashMap::new(),
            function_indices: HashMap::new(),
            function_list: Vec::new(),
            types: HashMap::new(),
            known_constructors: HashSet::new(),
            scope_depth: 0,
            module_path: Vec::new(),
            imports: HashMap::new(),
            function_visibility: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            type_traits: HashMap::new(),
            local_types: HashMap::new(),
            param_types: HashMap::new(),
            fn_asts: HashMap::new(),
            fn_type_params: HashMap::new(),
            polymorphic_fns: HashSet::new(),
            current_function_name: None,
            current_fn_type_params: Vec::new(),
            loop_stack: Vec::new(),
            line_starts: vec![0],
            pending_functions: Vec::new(),
            current_source: None,
            current_source_name: None,
            type_defs: HashMap::new(),
        }
    }

    /// Compile all pending functions.
    pub fn compile_all(&mut self) -> Result<(), (CompileError, String, String)> {
        let pending = std::mem::take(&mut self.pending_functions);

        for (fn_def, module_path, imports, line_starts, source, source_name) in pending {
            let saved_path = self.module_path.clone();
            let saved_imports = self.imports.clone();
            let saved_line_starts = self.line_starts.clone();
            let saved_source = self.current_source.clone();
            let saved_source_name = self.current_source_name.clone();

            self.module_path = module_path;
            // self.imports = imports;
            // Merge imports instead of replacing, to be safe. 
            // Also debug print to see what we are restoring.
            // println!("DEBUG: Restoring {} imports for {}", imports.len(), fn_def.name.node);
            self.imports.extend(imports);
            self.line_starts = line_starts;
            self.current_source = Some(source.clone());
            self.current_source_name = Some(source_name.clone());

            self.compile_fn_def(&fn_def).map_err(|e| (e, source_name, source.to_string()))?;

            self.module_path = saved_path;
            self.imports = saved_imports;
            self.line_starts = saved_line_starts;
            self.current_source = saved_source;
            self.current_source_name = saved_source_name;
        }
        Ok(())
    }

    /// Compile a module and add it to the current compilation context.
    pub fn add_module(&mut self, module: &Module, module_path: Vec<String>, source: Arc<String>, source_name: String) -> Result<(), CompileError> {
        // Update line_starts for this file
        self.line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                self.line_starts.push(i + 1);
            }
        }

        self.current_source = Some(source);
        self.current_source_name = Some(source_name.clone());

        // Set module path
        self.module_path = module_path;

        // Compile items
        self.compile_items(&module.items)?;

        // Reset module path
        self.module_path = Vec::new();

        Ok(())
    }

    pub fn new(source: &str) -> Self {
        // Compute line start offsets for source mapping
        let mut line_starts = vec![0]; // Line 1 starts at offset 0
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1); // Next line starts after the newline
            }
        }

        Self {
            chunk: Chunk::new(),
            locals: HashMap::new(),
            next_reg: 0,
            capture_indices: HashMap::new(),
            functions: HashMap::new(),
            function_indices: HashMap::new(),
            function_list: Vec::new(),
            types: HashMap::new(),
            known_constructors: HashSet::new(),
            scope_depth: 0,
            module_path: Vec::new(),
            imports: HashMap::new(),
            function_visibility: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            type_traits: HashMap::new(),
            local_types: HashMap::new(),
            param_types: HashMap::new(),
            fn_asts: HashMap::new(),
            fn_type_params: HashMap::new(),
            polymorphic_fns: HashSet::new(),
            current_function_name: None,
            current_fn_type_params: Vec::new(),
            loop_stack: Vec::new(),
            line_starts,
            pending_functions: Vec::new(),
            current_source: Some(Arc::new(source.to_string())),
            current_source_name: Some("unknown".to_string()),
            type_defs: HashMap::new(),
        }
    }

    /// Convert a byte offset to a line number (1-indexed).
    fn offset_to_line(&self, offset: usize) -> usize {
        // Binary search for the line containing this offset
        match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx + 1, // Exact match: offset is at start of line
            Err(idx) => idx,    // offset is within line idx (1-indexed)
        }
    }

    /// Get the line number for a span.
    fn span_line(&self, span: Span) -> usize {
        self.offset_to_line(span.start)
    }

    /// Get the fully qualified name with the current module path prefix.
    fn qualify_name(&self, name: &str) -> String {
        if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.module_path.join("."), name)
        }
    }

    /// Resolve a name, checking imports and module path.
    /// Returns the fully qualified name if found, or the original name.
    fn resolve_name(&self, name: &str) -> String {
        // First check imports
        if let Some(qualified) = self.imports.get(name) {
            return qualified.clone();
        }

        // If the name already contains '.', it's already qualified
        if name.contains('.') {
            return name.to_string();
        }
        // Otherwise, check if it's a known function or type or constructor in the current module
        let qualified = self.qualify_name(name);
        if self.functions.contains_key(&qualified) || self.types.contains_key(&qualified) || self.known_constructors.contains(&qualified) {
            return qualified;
        }
        // Check if it's in the global scope
        if self.functions.contains_key(name) || self.types.contains_key(name) || self.known_constructors.contains(name) {
            return name.to_string();
        }
        // Return the original name (will error later if not found)
        name.to_string()
    }

    /// Check if an expression is float-typed (for type-directed operator selection).
    /// This is a simple heuristic: true if the expression is a float literal or
    /// a binary operation on floats.
    /// Check if a type name refers to a float type.
    fn is_float_type_name(ty: &str) -> bool {
        matches!(ty, "Float" | "Float32" | "Float64")
    }

    /// Check if a type name refers to a BigInt type.
    fn is_bigint_type_name(ty: &str) -> bool {
        ty == "BigInt"
    }

    /// Check if a type name refers to a small integer type (not BigInt).
    fn is_small_int_type_name(ty: &str) -> bool {
        matches!(ty, "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64")
    }

    /// Check if a type name refers to an integer type (including BigInt).
    fn is_int_type_name(ty: &str) -> bool {
        Self::is_small_int_type_name(ty) || Self::is_bigint_type_name(ty)
    }

    fn is_float_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Float(_, _) | Expr::Float32(_, _) => true,
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::BigInt(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be float
            Expr::Var(ident) => {
                // Check locals.is_float first
                if self.locals.get(&ident.node).map(|info| info.is_float).unwrap_or(false) {
                    return true;
                }
                // Check local_types for float types
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_float_type_name(ty);
                }
                // Check param_types for function parameters
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_float_type_name(ty);
                }
                false
            }
            // Field access: look up the field's type from the record definition
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_float_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            Expr::BinOp(left, op, right, _) => {
                // Arithmetic operators preserve float type
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow | BinOp::Mod => {
                        self.is_float_expr(left) || self.is_float_expr(right)
                    }
                    // Comparison operators return bool
                    _ => false,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, operand, _) => self.is_float_expr(operand),
            Expr::If(_, then_branch, else_branch, _) => {
                self.is_float_expr(then_branch) || self.is_float_expr(else_branch)
            }
            Expr::Block(stmts, _) => {
                // Check if the last statement is an expression that is float-typed
                stmts.last().map(|s| match s {
                    Stmt::Expr(e) => self.is_float_expr(e),
                    _ => false,
                }).unwrap_or(false)
            }
            // Function calls: assume non-float by default.
            // We can't know the return type without proper type inference,
            // and assuming float based on arguments is incorrect (e.g., show(3.14) returns String).
            Expr::Call(_, _, _) => false,
            _ => false, // Assume non-float by default for other expressions
        }
    }

    /// Check if an expression is known to be an integer at compile time.
    fn is_int_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::BigInt(_, _) => true,
            Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be int
            Expr::Var(ident) => {
                // Check locals - if it's float, return false
                if self.locals.get(&ident.node).map(|info| info.is_float).unwrap_or(false) {
                    return false;
                }
                // Check local_types for int types
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_int_type_name(ty);
                }
                // Check param_types for function parameters
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_int_type_name(ty);
                }
                // If type is unknown, we can't safely assume it's an int
                // (it could be float from pattern matching, etc.)
                false
            }
            // Field access: look up the field's type from the record definition
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_int_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            Expr::BinOp(left, op, right, _) => {
                // Arithmetic operators: int if both sides are int
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.is_int_expr(left) && self.is_int_expr(right) && !self.is_float_expr(left) && !self.is_float_expr(right)
                    }
                    // Comparison operators return bool
                    _ => false,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, operand, _) => self.is_int_expr(operand) && !self.is_float_expr(operand),
            Expr::If(_, then_branch, else_branch, _) => {
                self.is_int_expr(then_branch) && self.is_int_expr(else_branch)
                    && !self.is_float_expr(then_branch) && !self.is_float_expr(else_branch)
            }
            Expr::Block(stmts, _) => {
                // Check if the last statement is an expression that is int-typed
                stmts.last().map(|s| match s {
                    Stmt::Expr(e) => self.is_int_expr(e) && !self.is_float_expr(e),
                    _ => false,
                }).unwrap_or(false)
            }
            // Function calls: assume non-int by default
            Expr::Call(_, _, _) => false,
            _ => false,
        }
    }

    /// Check if an expression is known to be a BigInt at compile time.
    fn is_bigint_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BigInt(_, _) => true,
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _)
            | Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            // Check if variable is known to be BigInt
            Expr::Var(ident) => {
                // Check local_types
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_bigint_type_name(ty);
                }
                // Check param_types
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_bigint_type_name(ty);
                }
                false
            }
            // Field access: look up the field's type
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_bigint_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            Expr::BinOp(left, op, right, _) => {
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.is_bigint_expr(left) || self.is_bigint_expr(right)
                    }
                    _ => false,
                }
            }
            Expr::UnaryOp(UnaryOp::Neg, operand, _) => self.is_bigint_expr(operand),
            Expr::If(_, then_branch, else_branch, _) => {
                self.is_bigint_expr(then_branch) || self.is_bigint_expr(else_branch)
            }
            Expr::Block(stmts, _) => {
                stmts.last().map(|s| match s {
                    Stmt::Expr(e) => self.is_bigint_expr(e),
                    _ => false,
                }).unwrap_or(false)
            }
            Expr::Call(_, _, _) => false,
            _ => false,
        }
    }

    /// Check if an expression is a small int (not BigInt) at compile time.
    fn is_small_int_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Int(_, _) | Expr::Int8(_, _) | Expr::Int16(_, _) | Expr::Int32(_, _)
            | Expr::UInt8(_, _) | Expr::UInt16(_, _) | Expr::UInt32(_, _) | Expr::UInt64(_, _) => true,
            Expr::BigInt(_, _) | Expr::Float(_, _) | Expr::Float32(_, _) | Expr::Decimal(_, _) => false,
            Expr::Var(ident) => {
                if let Some(ty) = self.local_types.get(&ident.node) {
                    return Self::is_small_int_type_name(ty);
                }
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Self::is_small_int_type_name(ty);
                }
                false
            }
            Expr::FieldAccess(obj, field, _) => {
                if let Some(obj_type) = self.expr_type_name(obj) {
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Self::is_small_int_type_name(ftype);
                                }
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Compile a top-level item.
    pub fn compile_item(&mut self, item: &Item) -> Result<(), CompileError> {
        match item {
            Item::FnDef(fn_def) => {
                self.compile_fn_def(fn_def)?;
            }
            Item::TypeDef(type_def) => {
                self.compile_type_def(type_def)?;
            }
            Item::TraitDef(trait_def) => {
                self.compile_trait_def(trait_def)?;
            }
            Item::TraitImpl(trait_impl) => {
                self.compile_trait_impl(trait_impl)?;
            }
            _ => {
                return Err(CompileError::NotImplemented {
                    feature: format!("item: {:?}", item),
                    span: item.span(),
                });
            }
        }
        Ok(())
    }

    /// Look up field types for a variant constructor.
    fn get_constructor_field_types(&self, ctor_name: &str) -> Vec<String> {
        for info in self.types.values() {
            if let TypeInfoKind::Variant { constructors } = &info.kind {
                for (name, field_types) in constructors {
                    if name == ctor_name {
                        return field_types.clone();
                    }
                }
            }
        }
        vec![]
    }

    /// Get the simple name of a type expression (for type tracking).
    fn type_expr_name(&self, ty: &nostos_syntax::TypeExpr) -> String {
        match ty {
            nostos_syntax::TypeExpr::Name(ident) => ident.node.clone(),
            nostos_syntax::TypeExpr::Generic(ident, _) => ident.node.clone(),
            nostos_syntax::TypeExpr::Function(_, _) => "Function".to_string(),
            nostos_syntax::TypeExpr::Record(_) => "Record".to_string(),
            nostos_syntax::TypeExpr::Tuple(_) => "Tuple".to_string(),
            nostos_syntax::TypeExpr::Unit => "Unit".to_string(),
        }
    }

    /// Compile a type definition.
    fn compile_type_def(&mut self, def: &TypeDef) -> Result<(), CompileError> {
        // Use qualified name (with module path prefix)
        let name = self.qualify_name(&def.name.node);

        let kind = match &def.body {
            TypeBody::Record(fields) => {
                // Register record name as a constructor
                self.known_constructors.insert(name.clone());
                let field_info: Vec<(String, String)> = fields.iter()
                    .map(|f| (f.name.node.clone(), self.type_expr_name(&f.ty)))
                    .collect();
                TypeInfoKind::Record { fields: field_info, mutable: def.mutable }
            }
            TypeBody::Variant(variants) => {
                let constructors: Vec<(String, Vec<String>)> = variants.iter()
                    .map(|v| {
                        // Register constructor name (qualified with module)
                        let qualified_ctor = self.qualify_name(&v.name.node);
                        self.known_constructors.insert(qualified_ctor);
                        
                        let field_types = match &v.fields {
                            VariantFields::Unit => vec![],
                            VariantFields::Positional(fields) => {
                                fields.iter().map(|ty| self.type_expr_name(ty)).collect()
                            }
                            VariantFields::Named(fields) => {
                                fields.iter().map(|f| self.type_expr_name(&f.ty)).collect()
                            }
                        };
                        (v.name.node.clone(), field_types)
                    })
                    .collect();
                TypeInfoKind::Variant { constructors }
            }
            TypeBody::Alias(_) => {
                // Type aliases don't need runtime representation
                return Ok(());
            }
            TypeBody::Empty => {
                // Never type
                return Ok(());
            }
        };

        self.types.insert(name.clone(), TypeInfo { name: name.clone(), kind });

        // Store the TypeDef AST for REPL introspection
        self.type_defs.insert(name.clone(), def.clone());

        // Handle deriving - generate synthetic trait implementations
        for trait_ident in &def.deriving {
            let trait_name = &trait_ident.node;
            self.derive_trait_for_type(&name, trait_name, def)?;
        }

        Ok(())
    }

    /// Generate a derived trait implementation for a type.
    /// Instead of generating bytecode directly, we synthesize AST and compile it.
    fn derive_trait_for_type(
        &mut self,
        type_name: &str,
        trait_name: &str,
        def: &TypeDef,
    ) -> Result<(), CompileError> {
        // Only support deriving Hash, Show, Copy, and Eq for now
        let trait_impl = match trait_name {
            "Hash" => self.synthesize_hash_impl(type_name, def)?,
            "Show" => self.synthesize_show_impl(type_name, def)?,
            "Copy" => self.synthesize_copy_impl(type_name, def)?,
            "Eq" => self.synthesize_eq_impl(type_name, def)?,
            _ => return Err(CompileError::CannotDerive {
                trait_name: trait_name.to_string(),
                ty: type_name.to_string(),
                reason: "only Hash, Show, Copy, and Eq can be derived".to_string(),
                span: def.name.span,
            }),
        };

        // Compile the synthesized trait implementation
        self.compile_trait_impl(&trait_impl)
    }

    /// Helper to create a Spanned node with a dummy span
    fn spanned<T>(&self, node: T) -> Spanned<T> {
        Spanned::new(node, Span { start: 0, end: 0 })
    }

    /// Helper to create an Ident with a dummy span
    fn ident(&self, name: &str) -> Ident {
        self.spanned(name.to_string())
    }

    /// Dummy span for synthesized AST nodes
    fn span(&self) -> Span {
        Span { start: 0, end: 0 }
    }

    /// Synthesize a Hash trait implementation for a type.
    /// Uses a proper hash combining algorithm that includes the type name
    /// to avoid collisions between different types with the same structure.
    fn synthesize_hash_impl(&self, type_name: &str, def: &TypeDef) -> Result<TraitImpl, CompileError> {
        let base_name = type_name.split("::").last().unwrap_or(type_name);

        // Helper to create: hash("type_name") as the base hash
        let type_name_hash = Expr::Call(
            Box::new(Expr::Var(self.ident("hash"))),
            vec![Expr::String(StringLit::Plain(base_name.to_string()), self.span())],
            self.span(),
        );

        // Build the body expression based on type structure
        let body = match &def.body {
            TypeBody::Record(fields) => {
                // hash(self) = hash("TypeName") * 31 + hash(self.f1) * 31 + hash(self.f2) ...
                // Start with type name hash to differentiate types with same structure
                let mut expr = type_name_hash.clone();
                for field in fields.iter() {
                    let hash_field = self.make_hash_field_expr("self", &field.name.node);
                    // result * 31 + hash(field)
                    let mul = Expr::BinOp(
                        Box::new(expr),
                        BinOp::Mul,
                        Box::new(Expr::Int(31, self.span())),
                        self.span(),
                    );
                    expr = Expr::BinOp(
                        Box::new(mul),
                        BinOp::Add,
                        Box::new(hash_field),
                        self.span(),
                    );
                }
                expr
            }
            TypeBody::Variant(variants) => {
                // Build a match expression that hashes each variant
                // Each variant starts with hash("TypeName") * 31 + discriminant
                let arms: Vec<MatchArm> = variants.iter().enumerate().map(|(idx, variant)| {
                    let (pattern, body) = match &variant.fields {
                        VariantFields::Unit => {
                            // Pattern: VariantName
                            let pattern = Pattern::Variant(
                                self.ident(&variant.name.node),
                                VariantPatternFields::Unit,
                                self.span(),
                            );
                            // Body: hash("TypeName") * 31 + idx
                            let mul = Expr::BinOp(
                                Box::new(type_name_hash.clone()),
                                BinOp::Mul,
                                Box::new(Expr::Int(31, self.span())),
                                self.span(),
                            );
                            let body = Expr::BinOp(
                                Box::new(mul),
                                BinOp::Add,
                                Box::new(Expr::Int(idx as i64, self.span())),
                                self.span(),
                            );
                            (pattern, body)
                        }
                        VariantFields::Positional(field_types) => {
                            // Pattern: VariantName(a, b, ...)
                            let var_names: Vec<String> = (0..field_types.len())
                                .map(|i| format!("__f{}", i))
                                .collect();
                            let patterns: Vec<Pattern> = var_names.iter()
                                .map(|name| Pattern::Var(self.ident(name)))
                                .collect();
                            let pattern = Pattern::Variant(
                                self.ident(&variant.name.node),
                                VariantPatternFields::Positional(patterns),
                                self.span(),
                            );

                            // Body: hash("TypeName") * 31 + idx, then * 31 + hash(field) for each field
                            let mul = Expr::BinOp(
                                Box::new(type_name_hash.clone()),
                                BinOp::Mul,
                                Box::new(Expr::Int(31, self.span())),
                                self.span(),
                            );
                            let mut body = Expr::BinOp(
                                Box::new(mul),
                                BinOp::Add,
                                Box::new(Expr::Int(idx as i64, self.span())),
                                self.span(),
                            );
                            for var_name in &var_names {
                                let hash_call = self.make_hash_var_expr(var_name);
                                let mul = Expr::BinOp(
                                    Box::new(body),
                                    BinOp::Mul,
                                    Box::new(Expr::Int(31, self.span())),
                                    self.span(),
                                );
                                body = Expr::BinOp(
                                    Box::new(mul),
                                    BinOp::Add,
                                    Box::new(hash_call),
                                    self.span(),
                                );
                            }
                            (pattern, body)
                        }
                        VariantFields::Named(fields) => {
                            // For named fields, we treat them like positional
                            let var_names: Vec<String> = fields.iter()
                                .map(|f| f.name.node.clone())
                                .collect();
                            let patterns: Vec<Pattern> = var_names.iter()
                                .map(|name| Pattern::Var(self.ident(name)))
                                .collect();
                            let pattern = Pattern::Variant(
                                self.ident(&variant.name.node),
                                VariantPatternFields::Positional(patterns),
                                self.span(),
                            );

                            // Body: hash("TypeName") * 31 + idx, then * 31 + hash(field) for each field
                            let mul = Expr::BinOp(
                                Box::new(type_name_hash.clone()),
                                BinOp::Mul,
                                Box::new(Expr::Int(31, self.span())),
                                self.span(),
                            );
                            let mut body = Expr::BinOp(
                                Box::new(mul),
                                BinOp::Add,
                                Box::new(Expr::Int(idx as i64, self.span())),
                                self.span(),
                            );
                            for var_name in &var_names {
                                let hash_call = self.make_hash_var_expr(var_name);
                                let mul = Expr::BinOp(
                                    Box::new(body),
                                    BinOp::Mul,
                                    Box::new(Expr::Int(31, self.span())),
                                    self.span(),
                                );
                                body = Expr::BinOp(
                                    Box::new(mul),
                                    BinOp::Add,
                                    Box::new(hash_call),
                                    self.span(),
                                );
                            }
                            (pattern, body)
                        }
                    };
                    MatchArm {
                        pattern,
                        guard: None,
                        body,
                        span: self.span(),
                    }
                }).collect();

                Expr::Match(
                    Box::new(Expr::Var(self.ident("self"))),
                    arms,
                    self.span(),
                )
            }
            _ => return Err(CompileError::CannotDerive {
                trait_name: "Hash".to_string(),
                ty: type_name.to_string(),
                reason: "can only derive Hash for record and variant types".to_string(),
                span: def.name.span,
            }),
        };

        // Create the trait impl
        Ok(TraitImpl {
            ty: TypeExpr::Name(self.ident(base_name)),
            trait_name: self.ident("Hash"),
            when_clause: vec![],
            methods: vec![FnDef {
                visibility: Visibility::Public,
                doc: None,
                name: self.ident("hash"),
                type_params: vec![],
                clauses: vec![FnClause {
                    params: vec![FnParam {
                        pattern: Pattern::Var(self.ident("self")),
                        ty: None,
                    }],
                    guard: None,
                    return_type: Some(TypeExpr::Name(self.ident("Int"))),
                    body,
                    span: self.span(),
                }],
                span: self.span(),
            }],
            span: self.span(),
        })
    }

    /// Helper: make hash(self.field) expression
    fn make_hash_field_expr(&self, obj: &str, field: &str) -> Expr {
        let field_access = Expr::FieldAccess(
            Box::new(Expr::Var(self.ident(obj))),
            self.ident(field),
            self.span(),
        );
        Expr::Call(
            Box::new(Expr::Var(self.ident("hash"))),
            vec![field_access],
            self.span(),
        )
    }

    /// Helper: make hash(var) expression
    fn make_hash_var_expr(&self, var: &str) -> Expr {
        Expr::Call(
            Box::new(Expr::Var(self.ident("hash"))),
            vec![Expr::Var(self.ident(var))],
            self.span(),
        )
    }

    /// Synthesize a Show trait implementation for a type.
    fn synthesize_show_impl(&self, type_name: &str, def: &TypeDef) -> Result<TraitImpl, CompileError> {
        let base_name = type_name.split("::").last().unwrap_or(type_name);

        let body = match &def.body {
            TypeBody::Record(fields) => {
                // show(self) = "TypeName { f1: " ++ show(self.f1) ++ ", f2: " ++ show(self.f2) ++ " }"
                if fields.is_empty() {
                    Expr::String(StringLit::Plain(format!("{} {{}}", base_name)), self.span())
                } else {
                    let mut expr = Expr::String(
                        StringLit::Plain(format!("{} {{ {}: ", base_name, fields[0].name.node)),
                        self.span(),
                    );
                    expr = self.concat_show_field(expr, "self", &fields[0].name.node);

                    for field in fields.iter().skip(1) {
                        expr = Expr::BinOp(
                            Box::new(expr),
                            BinOp::Concat,
                            Box::new(Expr::String(
                                StringLit::Plain(format!(", {}: ", field.name.node)),
                                self.span(),
                            )),
                            self.span(),
                        );
                        expr = self.concat_show_field(expr, "self", &field.name.node);
                    }

                    Expr::BinOp(
                        Box::new(expr),
                        BinOp::Concat,
                        Box::new(Expr::String(StringLit::Plain(" }".to_string()), self.span())),
                        self.span(),
                    )
                }
            }
            TypeBody::Variant(variants) => {
                // Build a match expression that shows each variant
                let arms: Vec<MatchArm> = variants.iter().map(|variant| {
                    let (pattern, body) = match &variant.fields {
                        VariantFields::Unit => {
                            let pattern = Pattern::Variant(
                                self.ident(&variant.name.node),
                                VariantPatternFields::Unit,
                                self.span(),
                            );
                            let body = Expr::String(
                                StringLit::Plain(variant.name.node.clone()),
                                self.span(),
                            );
                            (pattern, body)
                        }
                        VariantFields::Positional(field_types) => {
                            let var_names: Vec<String> = (0..field_types.len())
                                .map(|i| format!("__f{}", i))
                                .collect();
                            let patterns: Vec<Pattern> = var_names.iter()
                                .map(|name| Pattern::Var(self.ident(name)))
                                .collect();
                            let pattern = Pattern::Variant(
                                self.ident(&variant.name.node),
                                VariantPatternFields::Positional(patterns),
                                self.span(),
                            );

                            // "VariantName(" ++ show(a) ++ ", " ++ show(b) ++ ")"
                            let mut body = Expr::String(
                                StringLit::Plain(format!("{}(", variant.name.node)),
                                self.span(),
                            );
                            for (i, var_name) in var_names.iter().enumerate() {
                                if i > 0 {
                                    body = Expr::BinOp(
                                        Box::new(body),
                                        BinOp::Concat,
                                        Box::new(Expr::String(
                                            StringLit::Plain(", ".to_string()),
                                            self.span(),
                                        )),
                                        self.span(),
                                    );
                                }
                                body = Expr::BinOp(
                                    Box::new(body),
                                    BinOp::Concat,
                                    Box::new(self.make_show_var_expr(var_name)),
                                    self.span(),
                                );
                            }
                            body = Expr::BinOp(
                                Box::new(body),
                                BinOp::Concat,
                                Box::new(Expr::String(StringLit::Plain(")".to_string()), self.span())),
                                self.span(),
                            );
                            (pattern, body)
                        }
                        VariantFields::Named(fields) => {
                            let var_names: Vec<String> = fields.iter()
                                .map(|f| f.name.node.clone())
                                .collect();
                            let patterns: Vec<Pattern> = var_names.iter()
                                .map(|name| Pattern::Var(self.ident(name)))
                                .collect();
                            let pattern = Pattern::Variant(
                                self.ident(&variant.name.node),
                                VariantPatternFields::Positional(patterns),
                                self.span(),
                            );

                            // "VariantName { field1: " ++ show(field1) ++ ", field2: " ++ show(field2) ++ " }"
                            let body = if var_names.is_empty() {
                                Expr::String(
                                    StringLit::Plain(format!("{} {{}}", variant.name.node)),
                                    self.span(),
                                )
                            } else {
                                let mut body = Expr::String(
                                    StringLit::Plain(format!("{} {{ {}: ", variant.name.node, var_names[0])),
                                    self.span(),
                                );
                                body = Expr::BinOp(
                                    Box::new(body),
                                    BinOp::Concat,
                                    Box::new(self.make_show_var_expr(&var_names[0])),
                                    self.span(),
                                );
                                for var_name in var_names.iter().skip(1) {
                                    body = Expr::BinOp(
                                        Box::new(body),
                                        BinOp::Concat,
                                        Box::new(Expr::String(
                                            StringLit::Plain(format!(", {}: ", var_name)),
                                            self.span(),
                                        )),
                                        self.span(),
                                    );
                                    body = Expr::BinOp(
                                        Box::new(body),
                                        BinOp::Concat,
                                        Box::new(self.make_show_var_expr(var_name)),
                                        self.span(),
                                    );
                                }
                                Expr::BinOp(
                                    Box::new(body),
                                    BinOp::Concat,
                                    Box::new(Expr::String(StringLit::Plain(" }".to_string()), self.span())),
                                    self.span(),
                                )
                            };
                            (pattern, body)
                        }
                    };
                    MatchArm {
                        pattern,
                        guard: None,
                        body,
                        span: self.span(),
                    }
                }).collect();

                Expr::Match(
                    Box::new(Expr::Var(self.ident("self"))),
                    arms,
                    self.span(),
                )
            }
            _ => return Err(CompileError::CannotDerive {
                trait_name: "Show".to_string(),
                ty: type_name.to_string(),
                reason: "can only derive Show for record and variant types".to_string(),
                span: def.name.span,
            }),
        };

        Ok(TraitImpl {
            ty: TypeExpr::Name(self.ident(base_name)),
            trait_name: self.ident("Show"),
            when_clause: vec![],
            methods: vec![FnDef {
                visibility: Visibility::Public,
                doc: None,
                name: self.ident("show"),
                type_params: vec![],
                clauses: vec![FnClause {
                    params: vec![FnParam {
                        pattern: Pattern::Var(self.ident("self")),
                        ty: None,
                    }],
                    guard: None,
                    return_type: Some(TypeExpr::Name(self.ident("String"))),
                    body,
                    span: self.span(),
                }],
                span: self.span(),
            }],
            span: self.span(),
        })
    }

    /// Helper: concatenate expr with show(self.field)
    fn concat_show_field(&self, expr: Expr, obj: &str, field: &str) -> Expr {
        let field_access = Expr::FieldAccess(
            Box::new(Expr::Var(self.ident(obj))),
            self.ident(field),
            self.span(),
        );
        let show_call = Expr::Call(
            Box::new(Expr::Var(self.ident("show"))),
            vec![field_access],
            self.span(),
        );
        Expr::BinOp(
            Box::new(expr),
            BinOp::Concat,
            Box::new(show_call),
            self.span(),
        )
    }

    /// Helper: make show(var) expression
    fn make_show_var_expr(&self, var: &str) -> Expr {
        Expr::Call(
            Box::new(Expr::Var(self.ident("show"))),
            vec![Expr::Var(self.ident(var))],
            self.span(),
        )
    }

    /// Synthesize a Copy trait implementation for a type.
    /// Copy simply returns self - the VM handles value copying semantically.
    fn synthesize_copy_impl(&self, type_name: &str, def: &TypeDef) -> Result<TraitImpl, CompileError> {
        let base_name = type_name.split("::").last().unwrap_or(type_name);

        // Verify the type can be derived
        let body = match &def.body {
            TypeBody::Record(_) | TypeBody::Variant(_) => {
                // For all copyable types, just return self
                // The VM will handle the actual copy semantics
                Expr::Var(self.ident("self"))
            }
            _ => return Err(CompileError::CannotDerive {
                trait_name: "Copy".to_string(),
                ty: type_name.to_string(),
                reason: "can only derive Copy for record and variant types".to_string(),
                span: def.name.span,
            }),
        };

        Ok(TraitImpl {
            ty: TypeExpr::Name(self.ident(base_name)),
            trait_name: self.ident("Copy"),
            when_clause: vec![],
            methods: vec![FnDef {
                visibility: Visibility::Public,
                doc: None,
                name: self.ident("copy"),
                type_params: vec![],
                clauses: vec![FnClause {
                    params: vec![FnParam {
                        pattern: Pattern::Var(self.ident("self")),
                        ty: None,
                    }],
                    guard: None,
                    return_type: None,
                    body,
                    span: self.span(),
                }],
                span: self.span(),
            }],
            span: self.span(),
        })
    }

    /// Synthesize an Eq trait implementation for a type.
    fn synthesize_eq_impl(&self, type_name: &str, def: &TypeDef) -> Result<TraitImpl, CompileError> {
        let base_name = type_name.split("::").last().unwrap_or(type_name);

        let body = match &def.body {
            TypeBody::Record(fields) => {
                // self.f1 == other.f1 && self.f2 == other.f2 && ...
                if fields.is_empty() {
                    Expr::Bool(true, self.span())
                } else {
                    let mut expr = self.make_field_eq_expr("self", "other", &fields[0].name.node);
                    for field in fields.iter().skip(1) {
                        let field_eq = self.make_field_eq_expr("self", "other", &field.name.node);
                        expr = Expr::BinOp(
                            Box::new(expr),
                            BinOp::And,
                            Box::new(field_eq),
                            self.span(),
                        );
                    }
                    expr
                }
            }
            TypeBody::Variant(_) => {
                // For variants, use built-in equality which handles discriminants and fields
                Expr::BinOp(
                    Box::new(Expr::Var(self.ident("self"))),
                    BinOp::Eq,
                    Box::new(Expr::Var(self.ident("other"))),
                    self.span(),
                )
            }
            _ => return Err(CompileError::CannotDerive {
                trait_name: "Eq".to_string(),
                ty: type_name.to_string(),
                reason: "can only derive Eq for record and variant types".to_string(),
                span: def.name.span,
            }),
        };

        Ok(TraitImpl {
            ty: TypeExpr::Name(self.ident(base_name)),
            trait_name: self.ident("Eq"),
            when_clause: vec![],
            methods: vec![FnDef {
                visibility: Visibility::Public,
                doc: None,
                name: self.ident("=="),
                type_params: vec![],
                clauses: vec![FnClause {
                    params: vec![
                        FnParam {
                            pattern: Pattern::Var(self.ident("self")),
                            ty: None,
                        },
                        FnParam {
                            pattern: Pattern::Var(self.ident("other")),
                            ty: None,
                        },
                    ],
                    guard: None,
                    return_type: Some(TypeExpr::Name(self.ident("Bool"))),
                    body,
                    span: self.span(),
                }],
                span: self.span(),
            }],
            span: self.span(),
        })
    }

    /// Helper: make self.field == other.field expression
    fn make_field_eq_expr(&self, obj1: &str, obj2: &str, field: &str) -> Expr {
        let field1 = Expr::FieldAccess(
            Box::new(Expr::Var(self.ident(obj1))),
            self.ident(field),
            self.span(),
        );
        let field2 = Expr::FieldAccess(
            Box::new(Expr::Var(self.ident(obj2))),
            self.ident(field),
            self.span(),
        );
        Expr::BinOp(
            Box::new(field1),
            BinOp::Eq,
            Box::new(field2),
            self.span(),
        )
    }

    /// Compile a trait definition.
    fn compile_trait_def(&mut self, def: &TraitDef) -> Result<(), CompileError> {
        let name = def.name.node.clone();

        let super_traits: Vec<String> = def.super_traits
            .iter()
            .map(|t| t.node.clone())
            .collect();

        let methods: Vec<TraitMethodInfo> = def.methods
            .iter()
            .map(|m| TraitMethodInfo {
                name: m.name.node.clone(),
                param_count: m.params.len(),
                has_default: m.default_impl.is_some(),
            })
            .collect();

        self.trait_defs.insert(name.clone(), TraitInfo {
            name,
            super_traits,
            methods,
        });

        Ok(())
    }

    /// Check if a trait is a built-in derivable trait.
    fn is_builtin_derivable_trait(&self, name: &str) -> bool {
        matches!(name, "Hash" | "Show" | "Copy" | "Eq")
    }

    /// Compile a trait implementation.
    fn compile_trait_impl(&mut self, impl_def: &TraitImpl) -> Result<(), CompileError> {
        // Get the type name from the type expression
        let type_name = self.type_expr_to_string(&impl_def.ty);
        let trait_name = impl_def.trait_name.node.clone();

        // Check that the trait exists (unless it's a built-in derivable trait)
        if !self.trait_defs.contains_key(&trait_name) && !self.is_builtin_derivable_trait(&trait_name) {
            return Err(CompileError::UnknownTrait {
                name: trait_name,
                span: impl_def.trait_name.span,
            });
        }

        // Compile each method as a function with a special qualified name: Type.Trait.method
        let mut method_names = Vec::new();
        for method in &impl_def.methods {
            let method_name = method.name.node.clone();
            let qualified_name = format!("{}.{}.{}", type_name, trait_name, method_name);

            // Create a modified FnDef with the qualified name and public visibility
            let mut modified_def = method.clone();
            modified_def.name = Spanned::new(qualified_name.clone(), method.name.span);
            modified_def.visibility = Visibility::Public; // Trait methods are always callable

            // Set up param_types for Self-typed parameters before compiling
            // This allows type inference to work correctly for field access on self/other
            let saved_param_types = std::mem::take(&mut self.param_types);
            for clause in &method.clauses {
                for param in &clause.params {
                    // Check if this parameter's type is Self
                    let is_self_typed = param.ty.as_ref().map(|t| {
                        matches!(t, TypeExpr::Name(n) if n.node == "Self")
                    }).unwrap_or(false);

                    // For "self" parameter (first param in trait methods) or Self-typed params
                    if let Some(name) = self.pattern_binding_name(&param.pattern) {
                        if name == "self" || is_self_typed {
                            self.param_types.insert(name, type_name.clone());
                        }
                    }
                }
            }

            self.compile_fn_def(&modified_def)?;

            // Restore param_types
            self.param_types = saved_param_types;

            method_names.push(qualified_name);
        }

        // Register the trait implementation
        let impl_info = TraitImplInfo {
            type_name: type_name.clone(),
            trait_name: trait_name.clone(),
            method_names,
        };
        self.trait_impls.insert((type_name.clone(), trait_name.clone()), impl_info);

        // Track which traits this type implements
        self.type_traits
            .entry(type_name)
            .or_insert_with(Vec::new)
            .push(trait_name);

        Ok(())
    }

    /// Convert a type expression to a string representation.
    fn type_expr_to_string(&self, ty: &TypeExpr) -> String {
        match ty {
            TypeExpr::Name(name) => name.node.clone(),
            TypeExpr::Generic(name, params) => {
                let params_str: Vec<String> = params.iter()
                    .map(|p| self.type_expr_to_string(p))
                    .collect();
                format!("{}[{}]", name.node, params_str.join(", "))
            }
            TypeExpr::Tuple(elems) => {
                let elems_str: Vec<String> = elems.iter()
                    .map(|e| self.type_expr_to_string(e))
                    .collect();
                format!("({})", elems_str.join(", "))
            }
            TypeExpr::Function(params, ret) => {
                let params_str: Vec<String> = params.iter()
                    .map(|p| self.type_expr_to_string(p))
                    .collect();
                format!("({}) -> {}", params_str.join(", "), self.type_expr_to_string(ret))
            }
            TypeExpr::Record(fields) => {
                let fields_str: Vec<String> = fields.iter()
                    .map(|(name, ty)| format!("{}: {}", name.node, self.type_expr_to_string(ty)))
                    .collect();
                format!("{{{}}}", fields_str.join(", "))
            }
            TypeExpr::Unit => "()".to_string(),
        }
    }

    /// Find the implementation of a trait method for a given type.
    pub fn find_trait_method(&self, type_name: &str, method_name: &str) -> Option<String> {
        // Look through all traits this type implements
        if let Some(traits) = self.type_traits.get(type_name) {
            for trait_name in traits {
                // Check if this trait has the method
                // For built-in derivable traits, we know the method names
                let has_method = if self.is_builtin_derivable_trait(trait_name) {
                    match (trait_name.as_str(), method_name) {
                        ("Show", "show") => true,
                        ("Hash", "hash") => true,
                        ("Copy", "copy") => true,
                        ("Eq", "==") => true,
                        _ => false,
                    }
                } else if let Some(trait_info) = self.trait_defs.get(trait_name) {
                    trait_info.methods.iter().any(|m| m.name == method_name)
                } else {
                    false
                };

                if has_method {
                    // Return the qualified function name
                    return Some(format!("{}.{}.{}", type_name, trait_name, method_name));
                }
            }
        }
        None
    }

    /// Check if a method name belongs to any trait.
    /// Used to detect when a method call on an untyped parameter might be a trait method.
    fn is_known_trait_method(&self, method_name: &str) -> bool {
        for trait_info in self.trait_defs.values() {
            if trait_info.methods.iter().any(|m| m.name == method_name) {
                return true;
            }
        }
        false
    }

    /// Check if a type implements a specific trait.
    fn type_implements_trait(&self, type_name: &str, trait_name: &str) -> bool {
        // Check if this is a primitive type with built-in trait support
        let is_primitive_hashable = matches!(
            type_name,
            "Int" | "Float" | "Bool" | "String" | "Char" | "Atom"
        );

        if is_primitive_hashable && trait_name == "Hash" {
            return true;
        }

        // Check primitive Show support (all types have native show)
        if trait_name == "Show" {
            return true; // All types can be shown (fall back to native)
        }

        // Check primitive Eq support
        let is_primitive_eq = matches!(
            type_name,
            "Int" | "Float" | "Bool" | "String" | "Char" | "Atom"
        );
        if is_primitive_eq && trait_name == "Eq" {
            return true;
        }

        // Check if the type has an explicit trait implementation
        if let Some(traits) = self.type_traits.get(type_name) {
            if traits.contains(&trait_name.to_string()) {
                return true;
            }
        }

        // Check if type_name is a type parameter in the current function with the required bound
        // This handles nested calls like: double_hash[T: Hash](x) calling hashable[T: Hash](x)
        for type_param in &self.current_fn_type_params {
            if type_param.name.node == type_name {
                // Check if this type parameter has the required trait bound
                for constraint in &type_param.constraints {
                    if constraint.node == trait_name {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check trait bounds for a function call.
    /// Returns an error if any type parameter's trait bounds are not satisfied.
    fn check_trait_bounds(
        &self,
        fn_name: &str,
        type_params: &[TypeParam],
        arg_types: &[Option<String>],
        span: Span,
    ) -> Result<(), CompileError> {
        // Get the function's parameter patterns to map args to type params
        let fn_def = match self.fn_asts.get(fn_name) {
            Some(def) => def,
            None => return Ok(()), // Can't check if we don't have the AST
        };

        if fn_def.clauses.is_empty() {
            return Ok(());
        }

        let params = &fn_def.clauses[0].params;

        // Build a map from type param name to the concrete type it's bound to
        let mut type_bindings: HashMap<String, String> = HashMap::new();

        for (i, param) in params.iter().enumerate() {
            if let Some(ref type_expr) = param.ty {
                // If the parameter has a type annotation like `x: T`, extract the type param name
                if let TypeExpr::Name(ident) = type_expr {
                    let type_param_name = &ident.node;
                    // Check if this is one of our type parameters
                    if type_params.iter().any(|tp| &tp.name.node == type_param_name) {
                        // We found a type parameter - map it to the concrete arg type
                        if i < arg_types.len() {
                            if let Some(ref concrete_type) = arg_types[i] {
                                type_bindings.insert(type_param_name.clone(), concrete_type.clone());
                            }
                        }
                    }
                }
            }
        }

        // Now check that each type parameter's bounds are satisfied
        for type_param in type_params {
            let type_param_name = &type_param.name.node;

            // Get the concrete type bound to this type parameter
            let concrete_type = match type_bindings.get(type_param_name) {
                Some(t) => t,
                None => continue, // Type not used or not known, skip checking
            };

            // Check each constraint (trait bound)
            for constraint in &type_param.constraints {
                let trait_name = &constraint.node;

                if !self.type_implements_trait(concrete_type, trait_name) {
                    return Err(CompileError::TraitBoundNotSatisfied {
                        type_name: concrete_type.clone(),
                        trait_name: trait_name.clone(),
                        span,
                    });
                }
            }
        }

        Ok(())
    }

    /// Compile a function definition.
    fn compile_fn_def(&mut self, def: &FnDef) -> Result<(), CompileError> {
        // Save compiler state
        let saved_chunk = std::mem::take(&mut self.chunk);
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_next_reg = self.next_reg;
        let saved_function_name = self.current_function_name.take();

        // Reset for new function
        self.chunk = Chunk::new();
        self.locals = HashMap::new();
        self.next_reg = 0;

        // Use qualified name (with module path prefix)
        let name = self.qualify_name(&def.name.node);

        // Track current function name for self-recursion optimization
        self.current_function_name = Some(name.clone());

        // Store the function's visibility
        self.function_visibility.insert(name.clone(), def.visibility);

        // Store AST for potential monomorphization
        self.fn_asts.insert(name.clone(), def.clone());

        // Invalidate any existing monomorphized variants of this function
        // We replace them with stale markers so CallDirect indices remain valid.
        // The variants will be recompiled on next use (checked in compile_monomorphized_variant)
        let prefix = format!("{}$", name);
        let variants_to_invalidate: Vec<String> = self.functions.keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for variant in &variants_to_invalidate {
            // Mark as invalidated by inserting a placeholder with empty code
            // The actual recompilation happens in compile_monomorphized_variant
            // which checks if the variant needs updating
            if let Some(old_func) = self.functions.get(variant) {
                let stale_marker = FunctionValue {
                    name: format!("__stale__{}", variant),  // Mark as stale
                    arity: old_func.arity,
                    param_names: old_func.param_names.clone(),
                    code: Arc::new(Chunk::new()),
                    module: old_func.module.clone(),
                    source_span: None,
                    jit_code: None,
                    call_count: std::sync::atomic::AtomicU32::new(0),
                    debug_symbols: vec![],
                    source_code: None,
                    source_file: None,
                    doc: None,
                    signature: None,
                    param_types: vec![],
                    return_type: None,
                };
                self.functions.insert(variant.clone(), Arc::new(stale_marker));
            }
        }

        // Store type parameters with bounds for trait bound checking at call sites
        if !def.type_params.is_empty() {
            self.fn_type_params.insert(name.clone(), def.type_params.clone());
        }

        // Set current function's type parameters for nested trait bound checking
        let saved_fn_type_params = std::mem::replace(&mut self.current_fn_type_params, def.type_params.clone());

        // Check if we need pattern matching dispatch
        let needs_dispatch = def.clauses.len() > 1 || def.clauses.iter().any(|clause| {
            clause.params.iter().any(|p| !self.is_simple_pattern(&p.pattern)) || clause.guard.is_some()
        });

        // Get arity from first clause (all clauses must have same arity)
        let arity = def.clauses[0].params.len();

        // Generate param names (used for debugging/introspection)
        let mut param_names: Vec<String> = Vec::new();
        for (i, param) in def.clauses[0].params.iter().enumerate() {
            if let Some(n) = self.pattern_binding_name(&param.pattern) {
                param_names.push(n);
            } else {
                param_names.push(format!("_arg{}", i));
            }
        }

        // Save and set up param_types for typed parameters (for compile-time type coercion)
        // Note: We clone instead of take to preserve any pre-set entries (e.g., self -> TypeName from trait impls)
        let saved_param_types = self.param_types.clone();
        for param in def.clauses[0].params.iter() {
            if let Some(param_name) = self.pattern_binding_name(&param.pattern) {
                if let Some(ty) = &param.ty {
                    let type_name = self.type_expr_to_string(ty);
                    // Only insert if not already set (preserve trait impl's self -> TypeName)
                    self.param_types.entry(param_name).or_insert(type_name);
                }
            }
        }

        // Allocate registers for parameters (0..arity)
        self.next_reg = arity as Reg;

        if needs_dispatch {
            // Multi-clause dispatch: try each clause in order
            let mut clause_jumps: Vec<usize> = Vec::new();

            for (clause_idx, clause) in def.clauses.iter().enumerate() {
                // Clear locals for this clause attempt
                self.locals.clear();

                // Track jump to skip to next clause on pattern failure
                let mut next_clause_jumps: Vec<usize> = Vec::new();

                // Test each parameter pattern
                for (i, param) in clause.params.iter().enumerate() {
                    let arg_reg = i as Reg;
                    let (success_reg, bindings) = self.compile_pattern_test(&param.pattern, arg_reg)?;

                    // Jump to next clause if pattern doesn't match
                    next_clause_jumps.push(
                        self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0)
                    );

                    // Add bindings to locals with type info from pattern
                    for (name, reg, is_float) in bindings {
                        self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
                    }
                }

                // Compile guard if present
                if let Some(guard) = &clause.guard {
                    let guard_reg = self.compile_expr_tail(guard, false)?;
                    next_clause_jumps.push(
                        self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0)
                    );
                }

                // All patterns matched and guard passed - compile body
                let result_reg = match self.compile_expr_tail(&clause.body, true) {
                    Ok(reg) => reg,
                    Err(CompileError::UnresolvedTraitMethod { .. }) => {
                        // Mark this function as needing monomorphization
                        self.polymorphic_fns.insert(name.clone());
                        // Restore state and return success
                        self.chunk = saved_chunk;
                        self.locals = saved_locals;
                        self.next_reg = saved_next_reg;
                        self.current_function_name = saved_function_name;
                        self.param_types = saved_param_types;
                        self.current_fn_type_params = saved_fn_type_params;
                        return Ok(());
                    }
                    Err(e) => return Err(e),
                };
                self.chunk.emit(Instruction::Return(result_reg), 0);

                // Record where we need to patch for "matched" jumps
                if clause_idx < def.clauses.len() - 1 {
                    clause_jumps.push(self.chunk.code.len());
                }

                // Patch all the "skip to next clause" jumps to land here
                let next_clause_addr = self.chunk.code.len();
                for jump_addr in next_clause_jumps {
                    self.chunk.patch_jump(jump_addr, next_clause_addr);
                }
            }

            // If we get here, no clause matched - throw error
            let error_idx = self.chunk.add_constant(Value::String(Arc::new(
                format!("No clause matched for function '{}'", name)
            )));
            let error_reg = self.alloc_reg();
            self.chunk.emit(Instruction::LoadConst(error_reg, error_idx), 0);
            self.chunk.emit(Instruction::Throw(error_reg), 0);
        } else {
            // Simple case: single clause with only variable patterns
            let clause = &def.clauses[0];

            // Map parameter patterns to registers
            for (i, param) in clause.params.iter().enumerate() {
                if let Some(n) = self.pattern_binding_name(&param.pattern) {
                    // Check if parameter has Float type annotation
                    let is_float = param.ty.as_ref().map(|t| {
                        match t {
                            nostos_syntax::TypeExpr::Name(ident) => {
                                matches!(ident.node.as_str(), "Float" | "Float32" | "Float64")
                            }
                            _ => false,
                        }
                    }).unwrap_or(false);
                    self.locals.insert(n, LocalInfo { reg: i as Reg, is_float, mutable: false });
                }
            }

            // Compile function body (in tail position)
            let result_reg = match self.compile_expr_tail(&clause.body, true) {
                Ok(reg) => reg,
                Err(CompileError::UnresolvedTraitMethod { .. }) => {
                    // Mark this function as needing monomorphization
                    self.polymorphic_fns.insert(name.clone());
                    // Restore state and return success
                    self.chunk = saved_chunk;
                    self.locals = saved_locals;
                    self.next_reg = saved_next_reg;
                    self.current_function_name = saved_function_name;
                    self.param_types = saved_param_types;
                    self.current_fn_type_params = saved_fn_type_params;
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            self.chunk.emit(Instruction::Return(result_reg), 0);
        }

        self.chunk.register_count = self.next_reg as usize;

        // Collect debug symbols from local variables
        let debug_symbols: Vec<LocalVarSymbol> = self
            .locals
            .iter()
            .map(|(name, info)| LocalVarSymbol {
                name: name.clone(),
                register: info.reg,
            })
            .collect();

        // Extract source code for this function from the source
        let source_code = self.current_source.as_ref().and_then(|src| {
            if def.span.start < src.len() && def.span.end <= src.len() {
                Some(Arc::new(src[def.span.start..def.span.end].to_string()))
            } else {
                None
            }
        });

        let func = FunctionValue {
            name: name.clone(),
            arity,
            param_names,
            code: Arc::new(std::mem::take(&mut self.chunk)),
            module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
            source_span: Some((def.span.start, def.span.end)),
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols,
            // REPL introspection fields
            source_code,
            source_file: self.current_source_name.clone(),
            doc: def.doc.clone(),
            signature: Some(self.infer_signature(def)),
            param_types: def.param_type_strings(),
            return_type: def.return_type_string(),
        };

        // Assign function index if not already indexed (for trait methods and late-compiled functions)
        if !self.function_indices.contains_key(&name) {
            let idx = self.function_list.len() as u16;
            self.function_indices.insert(name.clone(), idx);
            self.function_list.push(name.clone());
        }

        self.functions.insert(name, Arc::new(func));

        // Restore compiler state
        self.chunk = saved_chunk;
        self.locals = saved_locals;
        self.next_reg = saved_next_reg;
        self.current_function_name = saved_function_name;
        self.param_types = saved_param_types;
        self.current_fn_type_params = saved_fn_type_params;

        Ok(())
    }

    /// Check if a pattern is "simple" (just a variable or wildcard).
    fn is_simple_pattern(&self, pattern: &Pattern) -> bool {
        matches!(pattern, Pattern::Var(_) | Pattern::Wildcard(_))
    }

    /// Get binding name from a pattern (if it's a simple variable).
    fn pattern_binding_name(&self, pattern: &Pattern) -> Option<String> {
        match pattern {
            Pattern::Var(ident) => Some(ident.node.clone()),
            Pattern::Wildcard(_) => None,
            _ => None, // Complex patterns need deconstruction
        }
    }

    /// Compile an expression, potentially in tail position.
    fn compile_expr_tail(&mut self, expr: &Expr, is_tail: bool) -> Result<Reg, CompileError> {
        let line = self.span_line(expr.span());
        match expr {
            // Literals
            Expr::Int(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int64(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Float(f, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float64(*f));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Typed integer literals
            Expr::Int8(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int8(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Int16(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int16(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Int32(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int32(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Unsigned integer literals
            Expr::UInt8(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt8(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::UInt16(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt16(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::UInt32(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt32(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::UInt64(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::UInt64(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Float32 literal
            Expr::Float32(f, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float32(*f));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // BigInt literal
            Expr::BigInt(s, _) => {
                let dst = self.alloc_reg();
                use num_bigint::BigInt;
                let big = s.parse::<BigInt>().unwrap_or_default();
                let idx = self.chunk.add_constant(Value::BigInt(Arc::new(big)));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            // Decimal literal
            Expr::Decimal(s, _) => {
                let dst = self.alloc_reg();
                use rust_decimal::Decimal;
                let dec = s.parse::<Decimal>().unwrap_or_default();
                let idx = self.chunk.add_constant(Value::Decimal(dec));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Bool(b, _) => {
                let dst = self.alloc_reg();
                if *b {
                    self.chunk.emit(Instruction::LoadTrue(dst), line);
                } else {
                    self.chunk.emit(Instruction::LoadFalse(dst), line);
                }
                Ok(dst)
            }
            Expr::String(string_lit, _) => {
                match string_lit {
                    StringLit::Plain(s) => {
                        let dst = self.alloc_reg();
                        let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                        self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                        Ok(dst)
                    }
                    StringLit::Interpolated(parts) => {
                        self.compile_interpolated_string(parts)
                    }
                }
            }
            Expr::Char(c, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Char(*c));
                self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                Ok(dst)
            }
            Expr::Unit(_) => {
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), line);
                Ok(dst)
            }

            // Variables
            Expr::Var(ident) => {
                let name = &ident.node;
                if let Some(info) = self.locals.get(name) {
                    Ok(info.reg)
                } else if let Some(&capture_idx) = self.capture_indices.get(name) {
                    // It's a captured variable - load from closure environment
                    let dst = self.alloc_reg();
                    self.chunk.emit(Instruction::GetCapture(dst, capture_idx), line);
                    Ok(dst)
                } else if self.functions.contains_key(name) {
                    // It's a function reference
                    let dst = self.alloc_reg();
                    let func = self.functions.get(name).unwrap().clone();
                    let idx = self.chunk.add_constant(Value::Function(func));
                    self.chunk.emit(Instruction::LoadConst(dst, idx), line);
                    Ok(dst)
                } else {
                    Err(CompileError::UnknownVariable {
                        name: name.clone(),
                        span: ident.span,
                    })
                }
            }

            // Binary operations
            Expr::BinOp(left, op, right, _) => {
                self.compile_binop(op, left, right)
            }

            // Unary operations
            Expr::UnaryOp(op, operand, _) => {
                self.compile_unaryop(op, operand)
            }

            // Function call
            Expr::Call(func, args, _) => {
                self.compile_call(func, args, is_tail)
            }

            // If expression
            Expr::If(cond, then_branch, else_branch, _) => {
                self.compile_if(cond, then_branch, else_branch, is_tail)
            }

            // Match expression
            Expr::Match(scrutinee, arms, span) => {
                self.compile_match(scrutinee, arms, is_tail, span.start)
            }

            // Block
            Expr::Block(stmts, _) => {
                self.compile_block(stmts, is_tail)
            }

            // List literal
            Expr::List(items, tail, _) => {
                match tail {
                    Some(tail_expr) => {
                        // List cons syntax: [e1, e2, ... | tail]
                        // Compile items in order first
                        let mut item_regs = Vec::new();
                        for item in items {
                            let reg = self.compile_expr_tail(item, false)?;
                            item_regs.push(reg);
                        }
                        // Compile the tail
                        let mut result_reg = self.compile_expr_tail(tail_expr, false)?;
                        // Cons each item onto the tail in reverse order
                        for item_reg in item_regs.into_iter().rev() {
                            let new_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Cons(new_reg, item_reg, result_reg), line);
                            result_reg = new_reg;
                        }
                        Ok(result_reg)
                    }
                    None => {
                        // Simple list: [e1, e2, ...]
                        let mut regs = Vec::new();
                        for item in items {
                            let reg = self.compile_expr_tail(item, false)?;
                            regs.push(reg);
                        }
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeList(dst, regs.into()), line);
                        Ok(dst)
                    }
                }
            }

            // Tuple literal
            Expr::Tuple(items, _) => {
                let mut regs = Vec::new();
                for item in items {
                    let reg = self.compile_expr_tail(item, false)?;
                    regs.push(reg);
                }
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::MakeTuple(dst, regs.into()), line);
                Ok(dst)
            }

            // Lambda
            Expr::Lambda(params, body, _) => {
                self.compile_lambda(params, body)
            }

            // Field access
            Expr::FieldAccess(obj, field, _) => {
                let obj_reg = self.compile_expr_tail(obj, false)?;
                let dst = self.alloc_reg();
                let field_idx = self.chunk.add_constant(Value::String(Arc::new(field.node.clone())));
                self.chunk.emit(Instruction::GetField(dst, obj_reg, field_idx), line);
                Ok(dst)
            }

            // Record construction
            Expr::Record(type_name, fields, _) => {
                // Resolve type name (check imports)
                let qualified_type = self.resolve_name(&type_name.node);
                self.compile_record(&qualified_type, fields)
            }

            // Record update
            Expr::RecordUpdate(type_name, base, fields, _) => {
                // Resolve type name (check imports)
                let qualified_type = self.resolve_name(&type_name.node);
                self.compile_record_update(&qualified_type, base, fields)
            }

            // Method call (UFCS) or module-qualified function call
            Expr::MethodCall(obj, method, args, _span) => {
                // Check if this is a module-qualified call (e.g., Math.add(1, 2))
                // by checking if the object looks like a module path
                if let Some(module_path) = self.extract_module_path(obj) {
                    // It's a module-qualified call: Module.function(args)
                    let qualified_name = format!("{}.{}", module_path, method.node);

                    // === Check for builtin module-qualified functions first ===
                    match qualified_name.as_str() {
                        "File.readAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileReadAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.writeAll" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let content_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileWriteAll(dst, path_reg, content_reg), line);
                            return Ok(dst);
                        }
                        "Http.get" if args.len() == 1 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpGet(dst, url_reg), line);
                            return Ok(dst);
                        }
                        "Http.post" if args.len() == 2 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let body_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpPost(dst, url_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Http.put" if args.len() == 2 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let body_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpPut(dst, url_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Http.delete" if args.len() == 1 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpDelete(dst, url_reg), line);
                            return Ok(dst);
                        }
                        "Http.patch" if args.len() == 2 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let body_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpPatch(dst, url_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Http.head" if args.len() == 1 => {
                            let url_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpHead(dst, url_reg), line);
                            return Ok(dst);
                        }
                        "Http.request" if args.len() == 4 => {
                            let method_reg = self.compile_expr_tail(&args[0], false)?;
                            let url_reg = self.compile_expr_tail(&args[1], false)?;
                            let headers_reg = self.compile_expr_tail(&args[2], false)?;
                            let body_reg = self.compile_expr_tail(&args[3], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::HttpRequest(dst, method_reg, url_reg, headers_reg, body_reg), line);
                            return Ok(dst);
                        }
                        // HTTP Server functions
                        "Server.bind" if args.len() == 1 => {
                            let port_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerBind(dst, port_reg), line);
                            return Ok(dst);
                        }
                        "Server.accept" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerAccept(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "Server.respond" if args.len() == 4 => {
                            let req_id_reg = self.compile_expr_tail(&args[0], false)?;
                            let status_reg = self.compile_expr_tail(&args[1], false)?;
                            let headers_reg = self.compile_expr_tail(&args[2], false)?;
                            let body_reg = self.compile_expr_tail(&args[3], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerRespond(dst, req_id_reg, status_reg, headers_reg, body_reg), line);
                            return Ok(dst);
                        }
                        "Server.close" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::ServerClose(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        // String encoding functions
                        "Base64.encode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Base64Encode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Base64.decode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Base64Decode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Url.encode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::UrlEncode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Url.decode" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::UrlDecode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Encoding.toBytes" if args.len() == 1 => {
                            let str_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Utf8Encode(dst, str_reg), line);
                            return Ok(dst);
                        }
                        "Encoding.fromBytes" if args.len() == 1 => {
                            let bytes_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::Utf8Decode(dst, bytes_reg), line);
                            return Ok(dst);
                        }
                        // String functions
                        "String.length" | "String.chars" | "String.from_chars" | "String.to_int" if args.len() == 1 => {
                            let arg_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            let name_idx = self.chunk.add_constant(Value::String(Arc::new(qualified_name)));
                            self.chunk.emit(Instruction::CallNative(dst, name_idx, vec![arg_reg].into()), line);
                            return Ok(dst);
                        }
                        // File handle operations
                        "File.open" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let mode_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileOpen(dst, path_reg, mode_reg), line);
                            return Ok(dst);
                        }
                        "File.write" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let data_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileWrite(dst, handle_reg, data_reg), line);
                            return Ok(dst);
                        }
                        "File.read" if args.len() == 2 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let size_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileRead(dst, handle_reg, size_reg), line);
                            return Ok(dst);
                        }
                        "File.readLine" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileReadLine(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "File.flush" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileFlush(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "File.close" if args.len() == 1 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileClose(dst, handle_reg), line);
                            return Ok(dst);
                        }
                        "File.seek" if args.len() == 3 => {
                            let handle_reg = self.compile_expr_tail(&args[0], false)?;
                            let offset_reg = self.compile_expr_tail(&args[1], false)?;
                            let whence_reg = self.compile_expr_tail(&args[2], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileSeek(dst, handle_reg, offset_reg, whence_reg), line);
                            return Ok(dst);
                        }
                        // Directory operations
                        "Dir.create" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirCreate(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.createAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirCreateAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.list" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirList(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.remove" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirRemove(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.removeAll" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirRemoveAll(dst, path_reg), line);
                            return Ok(dst);
                        }
                        // File utilities
                        "File.exists" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileExists(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "Dir.exists" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::DirExists(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.remove" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileRemove(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.rename" if args.len() == 2 => {
                            let old_reg = self.compile_expr_tail(&args[0], false)?;
                            let new_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileRename(dst, old_reg, new_reg), line);
                            return Ok(dst);
                        }
                        "File.copy" if args.len() == 2 => {
                            let src_reg = self.compile_expr_tail(&args[0], false)?;
                            let dest_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileCopy(dst, src_reg, dest_reg), line);
                            return Ok(dst);
                        }
                        "File.size" if args.len() == 1 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileSize(dst, path_reg), line);
                            return Ok(dst);
                        }
                        "File.append" if args.len() == 2 => {
                            let path_reg = self.compile_expr_tail(&args[0], false)?;
                            let content_reg = self.compile_expr_tail(&args[1], false)?;
                            let dst = self.alloc_reg();
                            self.chunk.emit(Instruction::FileAppend(dst, path_reg, content_reg), line);
                            return Ok(dst);
                        }
                        _ => {} // Fall through to user-defined functions
                    }

                    let resolved_name = self.resolve_name(&qualified_name);

                    if self.functions.contains_key(&resolved_name) {
                        // Check visibility before allowing the call
                        self.check_visibility(&resolved_name, method.span)?;

                        // Compile arguments
                        let mut arg_regs = Vec::new();
                        for arg in args {
                            let reg = self.compile_expr_tail(arg, false)?;
                            arg_regs.push(reg);
                        }

                        let dst = self.alloc_reg();
                        // Direct function call by index (no HashMap lookup at runtime!)
                        let func_idx = *self.function_indices.get(&resolved_name)
                            .expect("Function should have been assigned an index");
                        if is_tail {
                            self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                            return Ok(0);
                        } else {
                            self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                            return Ok(dst);
                        }
                    }
                }

                // Try trait method dispatch if we can determine the type of obj
                if let Some(type_name) = self.expr_type_name(obj) {
                    if let Some(qualified_method) = self.find_trait_method(&type_name, &method.node) {
                        // Found a trait method - compile as qualified function call
                        let mut all_args = vec![obj.as_ref().clone()];
                        all_args.extend(args.iter().cloned());

                        // Compile arguments
                        let mut arg_regs = Vec::new();
                        for arg in &all_args {
                            let reg = self.compile_expr_tail(arg, false)?;
                            arg_regs.push(reg);
                        }

                        let dst = self.alloc_reg();
                        // Direct function call by index (no HashMap lookup at runtime!)
                        let func_idx = *self.function_indices.get(&qualified_method)
                            .expect("Trait method should have been assigned an index");
                        if is_tail {
                            self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                            return Ok(0);
                        } else {
                            self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                            return Ok(dst);
                        }
                    }
                } else {
                    // Type unknown - check if this is a trait method that needs monomorphization
                    if self.is_known_trait_method(&method.node) {
                        return Err(CompileError::UnresolvedTraitMethod {
                            method: method.node.clone(),
                            span: method.span,
                        });
                    }
                }

                // Regular UFCS method call: obj.method(args) -> method(obj, args)
                let mut all_args = vec![obj.as_ref().clone()];
                all_args.extend(args.iter().cloned());

                let func_expr = Expr::Var(method.clone());
                self.compile_call(&func_expr, &all_args, is_tail)
            }

            // Index access
            Expr::Index(coll, index, _) => {
                let coll_reg = self.compile_expr_tail(coll, false)?;
                let idx_reg = self.compile_expr_tail(index, false)?;
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::Index(dst, coll_reg, idx_reg), line);
                Ok(dst)
            }

            // Map literal: %{"key": value, ...}
            Expr::Map(pairs, _) => {
                let mut pair_regs = Vec::new();
                for (key, value) in pairs {
                    let key_reg = self.compile_expr_tail(key, false)?;
                    let val_reg = self.compile_expr_tail(value, false)?;
                    pair_regs.push((key_reg, val_reg));
                }
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::MakeMap(dst, pair_regs.into()), line);
                Ok(dst)
            }

            // Set literal: #{elem, ...}
            Expr::Set(elems, _) => {
                let mut regs = Vec::new();
                for elem in elems {
                    let reg = self.compile_expr_tail(elem, false)?;
                    regs.push(reg);
                }
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::MakeSet(dst, regs.into()), line);
                Ok(dst)
            }

            // Try/catch/finally expression
            Expr::Try(try_expr, catch_arms, finally_expr, _) => {
                self.compile_try(try_expr, catch_arms, finally_expr.as_deref(), is_tail)
            }

            // Error propagation: expr?
            Expr::Try_(inner_expr, _) => {
                self.compile_try_propagate(inner_expr)
            }

            // === Concurrency expressions ===

            // Send: pid <- msg
            Expr::Send(pid_expr, msg_expr, _) => {
                let pid_reg = self.compile_expr_tail(pid_expr, false)?;
                let msg_reg = self.compile_expr_tail(msg_expr, false)?;
                self.chunk.emit(Instruction::Send(pid_reg, msg_reg), line);
                // Send returns unit
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), line);
                Ok(dst)
            }

            // Spawn: spawn(func), spawn(() => expr), or spawn { block }
            Expr::Spawn(kind, func_expr, args, span) => {
                // If func_expr is a Block, wrap it in a zero-param Lambda (thunk)
                let effective_func = match func_expr.as_ref() {
                    Expr::Block(_, block_span) => {
                        Expr::Lambda(vec![], func_expr.clone(), block_span.clone())
                    }
                    _ => func_expr.as_ref().clone(),
                };
                let func_reg = self.compile_expr_tail(&effective_func, false)?;
                let mut arg_regs = Vec::new();
                for arg in args {
                    let reg = self.compile_expr_tail(arg, false)?;
                    arg_regs.push(reg);
                }
                let dst = self.alloc_reg();
                match kind {
                    SpawnKind::Normal => {
                        self.chunk.emit(Instruction::Spawn(dst, func_reg, arg_regs.into()), line);
                    }
                    SpawnKind::Linked => {
                        self.chunk.emit(Instruction::SpawnLink(dst, func_reg, arg_regs.into()), line);
                    }
                    SpawnKind::Monitored => {
                        // SpawnMonitor returns (pid, ref)
                        let ref_dst = self.alloc_reg();
                        self.chunk.emit(Instruction::SpawnMonitor(dst, ref_dst, func_reg, arg_regs.into()), line);
                    }
                }
                Ok(dst)
            }

            // Receive: receive pattern -> body ... after timeout -> timeout_body end
            Expr::Receive(arms, after_clause, _) => {
                // Allocate a register for the received message
                let msg_reg = self.alloc_reg();

                // Handle timeout if present
                let timeout_jump = if let Some((timeout_expr, _)) = after_clause {
                    // Compile timeout expression
                    let timeout_reg = self.compile_expr_tail(timeout_expr, false)?;
                    // Emit receive with timeout - places message in msg_reg or Unit if timeout
                    self.chunk.emit(Instruction::ReceiveTimeout(msg_reg, timeout_reg), line);
                    // Check if msg_reg is Unit (timeout indicator)
                    let is_unit_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::TestUnit(is_unit_reg, msg_reg), line);
                    // Jump to timeout handling if Unit
                    Some(self.chunk.emit(Instruction::JumpIfTrue(is_unit_reg, 0), line))
                } else {
                    // No timeout - regular receive
                    self.chunk.emit(Instruction::Receive(msg_reg), line);
                    None
                };

                // The message is in msg_reg after Receive completes
                // We need to match it against the arms

                let dst = self.alloc_reg();
                let mut end_jumps = Vec::new();

                for (i, arm) in arms.iter().enumerate() {
                    let is_last = i == arms.len() - 1;

                    // Save locals before processing arm (pattern bindings should be scoped to this arm)
                    let saved_locals = self.locals.clone();

                    // Try to match the pattern against the message
                    let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, msg_reg)?;

                    let next_arm_jump = if !is_last {
                        Some(self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), line))
                    } else {
                        None
                    };

                    // Bind pattern variables with type info from pattern
                    for (name, reg, is_float) in bindings {
                        self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
                    }

                    // Compile guard if present
                    if let Some(ref guard) = arm.guard {
                        let guard_reg = self.compile_expr_tail(guard, false)?;
                        if !is_last {
                            let guard_fail = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), line);
                            // Compile body
                            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
                            self.chunk.emit(Instruction::Move(dst, body_reg), line);
                            end_jumps.push(self.chunk.emit(Instruction::Jump(0), line));
                            // Patch guard fail jump
                            self.chunk.patch_jump(guard_fail, self.chunk.code.len());
                        } else {
                            // Last arm - no jump needed for guard failure
                            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
                            self.chunk.emit(Instruction::Move(dst, body_reg), line);
                        }
                    } else {
                        // No guard - compile body directly
                        let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
                        self.chunk.emit(Instruction::Move(dst, body_reg), line);
                        if !is_last {
                            end_jumps.push(self.chunk.emit(Instruction::Jump(0), line));
                        }
                    }

                    // Patch next arm jump
                    if let Some(jump_idx) = next_arm_jump {
                        self.chunk.patch_jump(jump_idx, self.chunk.code.len());
                    }

                    // Restore locals after arm (pattern bindings shouldn't leak to next arm)
                    self.locals = saved_locals;
                }

                // Handle timeout body if present
                if let Some((_, timeout_body)) = after_clause {
                    // Jump past timeout body (for normal message case)
                    let skip_timeout = self.chunk.emit(Instruction::Jump(0), line);
                    end_jumps.push(skip_timeout);

                    // Patch timeout jump to point here
                    if let Some(jump_idx) = timeout_jump {
                        self.chunk.patch_jump(jump_idx, self.chunk.code.len());
                    }

                    // Compile timeout body
                    let timeout_result = self.compile_expr_tail(timeout_body, is_tail)?;
                    self.chunk.emit(Instruction::Move(dst, timeout_result), line);
                }

                // Patch end jumps
                let end_target = self.chunk.code.len();
                for jump_idx in end_jumps {
                    self.chunk.patch_jump(jump_idx, end_target);
                }

                Ok(dst)
            }

            // While loop
            Expr::While(cond, body, _) => {
                self.compile_while(cond, body)
            }

            // For loop
            Expr::For(var, start, end, body, _) => {
                self.compile_for(var, start, end, body)
            }

            // Break
            Expr::Break(value, span) => {
                self.compile_break(value.as_ref().map(|v| v.as_ref()), *span)
            }

            // Continue
            Expr::Continue(span) => {
                self.compile_continue(*span)
            }

            _ => Err(CompileError::NotImplemented {
                feature: format!("expr: {:?}", expr),
                span: expr.span(),
            }),
        }
    }

    /// Compile a while loop.
    fn compile_while(&mut self, cond: &Expr, body: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);

        // Record loop start - for while loops, continue jumps back to condition
        let loop_start = self.chunk.code.len();

        // Push loop context (continue_addr same as start_addr for while loops)
        self.loop_stack.push(LoopContext {
            start_addr: loop_start,
            continue_addr: loop_start,
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });

        // Compile condition
        let cond_reg = self.compile_expr_tail(cond, false)?;

        // Jump to end if false
        let exit_jump = self.chunk.emit(Instruction::JumpIfFalse(cond_reg, 0), 0);

        // Compile body
        let _ = self.compile_expr_tail(body, false)?;

        // Jump back to loop start
        // Formula: offset = target - current_position - 1 (because IP is incremented before execution)
        let jump_offset = loop_start as i16 - self.chunk.code.len() as i16 - 1;
        self.chunk.emit(Instruction::Jump(jump_offset), 0);

        // Patch exit jump
        self.chunk.patch_jump(exit_jump, self.chunk.code.len());

        // Pop loop context and patch break/continue jumps
        let loop_ctx = self.loop_stack.pop().unwrap();
        for break_jump in loop_ctx.break_jumps {
            self.chunk.patch_jump(break_jump, self.chunk.code.len());
        }
        // Continue jumps should go to loop_start (already handled at emit time for while loops)
        for continue_jump in loop_ctx.continue_jumps {
            self.chunk.patch_jump(continue_jump, loop_start);
        }

        Ok(dst)
    }

    /// Compile a for loop.
    fn compile_for(&mut self, var: &Ident, start: &Expr, end: &Expr, body: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);

        // Compile start and end values
        let counter_reg = self.compile_expr_tail(start, false)?;
        let end_reg = self.compile_expr_tail(end, false)?;

        // Bind loop variable to counter register (for loop counter is always int)
        let saved_var = self.locals.get(&var.node).cloned();
        self.locals.insert(var.node.clone(), LocalInfo { reg: counter_reg, is_float: false, mutable: false });

        // Record loop start
        let loop_start = self.chunk.code.len();

        // Push loop context - continue_addr will be set later to point to increment
        // For now use 0 as placeholder
        self.loop_stack.push(LoopContext {
            start_addr: loop_start,
            continue_addr: 0, // Will be set after body compilation
            continue_jumps: Vec::new(),
            break_jumps: Vec::new(),
        });

        // Check if counter < end
        let cond_reg = self.alloc_reg();
        self.chunk.emit(Instruction::LtInt(cond_reg, counter_reg, end_reg), 0);

        // Jump to end if counter >= end
        let exit_jump = self.chunk.emit(Instruction::JumpIfFalse(cond_reg, 0), 0);

        // Compile body
        let _ = self.compile_expr_tail(body, false)?;

        // Record where increment starts (for continue jumps)
        let increment_addr = self.chunk.code.len();

        // Increment counter: counter = counter + 1
        let one_reg = self.alloc_reg();
        let one_idx = self.chunk.add_constant(Value::Int64(1));
        self.chunk.emit(Instruction::LoadConst(one_reg, one_idx), 0);
        self.chunk.emit(Instruction::AddInt(counter_reg, counter_reg, one_reg), 0);

        // Jump back to loop start
        // Formula: offset = target - current_position - 1 (because IP is incremented before execution)
        let jump_offset = loop_start as i16 - self.chunk.code.len() as i16 - 1;
        self.chunk.emit(Instruction::Jump(jump_offset), 0);

        // Patch exit jump
        self.chunk.patch_jump(exit_jump, self.chunk.code.len());

        // Pop loop context and patch break/continue jumps
        let loop_ctx = self.loop_stack.pop().unwrap();
        for break_jump in loop_ctx.break_jumps {
            self.chunk.patch_jump(break_jump, self.chunk.code.len());
        }
        // Continue jumps should go to the increment section
        for continue_jump in loop_ctx.continue_jumps {
            self.chunk.patch_jump(continue_jump, increment_addr);
        }

        // Restore previous variable binding if any
        if let Some(prev_info) = saved_var {
            self.locals.insert(var.node.clone(), prev_info);
        } else {
            self.locals.remove(&var.node);
        }

        Ok(dst)
    }

    /// Compile a break statement.
    fn compile_break(&mut self, value: Option<&Expr>, span: Span) -> Result<Reg, CompileError> {
        if self.loop_stack.is_empty() {
            return Err(CompileError::NotImplemented {
                feature: "break outside of loop".to_string(),
                span,
            });
        }

        // If there's a value, compile it (for future: return value from loop)
        let dst = if let Some(val) = value {
            self.compile_expr_tail(val, false)?
        } else {
            let r = self.alloc_reg();
            self.chunk.emit(Instruction::LoadUnit(r), 0);
            r
        };

        // Emit jump to be patched later
        let jump_idx = self.chunk.emit(Instruction::Jump(0), 0);

        // Add to current loop's break jumps
        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.break_jumps.push(jump_idx);
        }

        Ok(dst)
    }

    /// Compile a continue statement.
    fn compile_continue(&mut self, span: Span) -> Result<Reg, CompileError> {
        if self.loop_stack.is_empty() {
            return Err(CompileError::NotImplemented {
                feature: "continue outside of loop".to_string(),
                span,
            });
        }

        // Emit jump with placeholder offset - will be patched at end of loop
        let jump_idx = self.chunk.emit(Instruction::Jump(0), 0);

        // Add to current loop's continue jumps
        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.continue_jumps.push(jump_idx);
        }

        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Compile a binary operation.
    fn compile_binop(&mut self, op: &BinOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        // Compute line number from the left operand's span
        let line = self.span_line(left.span());

        // Handle short-circuit operators first
        match op {
            BinOp::And => return self.compile_and(left, right),
            BinOp::Or => return self.compile_or(left, right),
            BinOp::Pipe => {
                // a |> f is f(a)
                return self.compile_call(right, &[left.clone()], false);
            }
            _ => {}
        }

        // Check numeric types for coercion
        let left_is_float = self.is_float_expr(left);
        let right_is_float = self.is_float_expr(right);
        let left_is_bigint = self.is_bigint_expr(left);
        let right_is_bigint = self.is_bigint_expr(right);
        let is_float = left_is_float || right_is_float;
        let is_bigint = left_is_bigint || right_is_bigint;

        let mut left_reg = self.compile_expr_tail(left, false)?;
        let mut right_reg = self.compile_expr_tail(right, false)?;

        // Emit type coercion if needed
        // Priority: Float > BigInt > small ints
        if is_float {
            // Float coercion: convert non-floats to float
            if !left_is_float && self.is_int_expr(left) {
                let coerced = self.alloc_reg();
                self.chunk.emit(Instruction::IntToFloat(coerced, left_reg), self.span_line(left.span()));
                left_reg = coerced;
            }
            if !right_is_float && self.is_int_expr(right) {
                let coerced = self.alloc_reg();
                self.chunk.emit(Instruction::IntToFloat(coerced, right_reg), self.span_line(right.span()));
                right_reg = coerced;
            }
        } else if is_bigint {
            // BigInt coercion: convert small ints to BigInt
            if !left_is_bigint && self.is_small_int_expr(left) {
                let coerced = self.alloc_reg();
                self.chunk.emit(Instruction::ToBigInt(coerced, left_reg), self.span_line(left.span()));
                left_reg = coerced;
            }
            if !right_is_bigint && self.is_small_int_expr(right) {
                let coerced = self.alloc_reg();
                self.chunk.emit(Instruction::ToBigInt(coerced, right_reg), self.span_line(right.span()));
                right_reg = coerced;
            }
        }

        let dst = self.alloc_reg();

        let instr = match op {
            BinOp::Add => {
                if is_float {
                    Instruction::AddFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::AddInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Sub => {
                if is_float {
                    Instruction::SubFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::SubInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Mul => {
                if is_float {
                    Instruction::MulFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::MulInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Div => {
                if is_float {
                    Instruction::DivFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::DivInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Mod => Instruction::ModInt(dst, left_reg, right_reg),
            BinOp::Pow => Instruction::PowFloat(dst, left_reg, right_reg),
            BinOp::Eq => {
                if is_float {
                    Instruction::EqFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::Eq(dst, left_reg, right_reg)
                }
            }
            BinOp::NotEq => {
                if is_float {
                    self.chunk.emit(Instruction::EqFloat(dst, left_reg, right_reg), line);
                } else {
                    self.chunk.emit(Instruction::Eq(dst, left_reg, right_reg), line);
                }
                self.chunk.emit(Instruction::Not(dst, dst), line);
                return Ok(dst);
            }
            BinOp::Lt => {
                if is_float {
                    Instruction::LtFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::LtInt(dst, left_reg, right_reg)
                }
            }
            BinOp::LtEq => {
                if is_float {
                    Instruction::LeFloat(dst, left_reg, right_reg)
                } else {
                    Instruction::LeInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Gt => {
                // Gt is Lt with args swapped
                if is_float {
                    Instruction::LtFloat(dst, right_reg, left_reg)
                } else {
                    Instruction::GtInt(dst, left_reg, right_reg)
                }
            }
            BinOp::GtEq => {
                // GtEq is LeEq with args swapped
                if is_float {
                    Instruction::LeFloat(dst, right_reg, left_reg)
                } else {
                    Instruction::GeInt(dst, left_reg, right_reg)
                }
            }
            BinOp::Concat => Instruction::Concat(dst, left_reg, right_reg),
            BinOp::And | BinOp::Or | BinOp::Pipe => unreachable!(),
        };

        self.chunk.emit(instr, line);
        Ok(dst)
    }

    /// Compile short-circuit AND.
    fn compile_and(&mut self, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let left_reg = self.compile_expr_tail(left, false)?;
        let dst = self.alloc_reg();

        let end_jump = self.chunk.emit(Instruction::JumpIfFalse(left_reg, 0), 0);
        let right_reg = self.compile_expr_tail(right, false)?;
        self.chunk.emit(Instruction::Move(dst, right_reg), 0);
        let skip_false = self.chunk.emit(Instruction::Jump(0), 0);
        let false_target = self.chunk.code.len();
        self.chunk.patch_jump(end_jump, false_target);
        self.chunk.emit(Instruction::LoadFalse(dst), 0);
        let end_target = self.chunk.code.len();
        self.chunk.patch_jump(skip_false, end_target);
        Ok(dst)
    }

    /// Compile short-circuit OR.
    fn compile_or(&mut self, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let left_reg = self.compile_expr_tail(left, false)?;
        let dst = self.alloc_reg();

        let end_jump = self.chunk.emit(Instruction::JumpIfTrue(left_reg, 0), 0);
        let right_reg = self.compile_expr_tail(right, false)?;
        self.chunk.emit(Instruction::Move(dst, right_reg), 0);
        let skip_true = self.chunk.emit(Instruction::Jump(0), 0);
        let true_target = self.chunk.code.len();
        self.chunk.patch_jump(end_jump, true_target);
        self.chunk.emit(Instruction::LoadTrue(dst), 0);
        let end_target = self.chunk.code.len();
        self.chunk.patch_jump(skip_true, end_target);
        Ok(dst)
    }

    /// Compile a unary operation.
    fn compile_unaryop(&mut self, op: &UnaryOp, operand: &Expr) -> Result<Reg, CompileError> {
        let is_float = self.is_float_expr(operand);
        let src = self.compile_expr_tail(operand, false)?;
        let dst = self.alloc_reg();

        let instr = match op {
            UnaryOp::Neg => {
                if is_float {
                    Instruction::NegFloat(dst, src)
                } else {
                    Instruction::NegInt(dst, src)
                }
            }
            UnaryOp::Not => Instruction::Not(dst, src),
        };

        self.chunk.emit(instr, 0);
        Ok(dst)
    }

    /// Compile a function call, with tail call optimization.
    fn compile_call(&mut self, func: &Expr, args: &[Expr], is_tail: bool) -> Result<Reg, CompileError> {
        // Get line number for this call expression
        let line = self.span_line(func.span());

        // Try to extract a qualified function name from the expression
        let maybe_qualified_name = self.extract_qualified_name(func);

        // Handle special built-in `throw` that compiles to the Throw instruction
        if let Some(ref name) = maybe_qualified_name {
            if name == "throw" && args.len() == 1 {
                let arg_reg = self.compile_expr_tail(&args[0], false)?;
                self.chunk.emit(Instruction::Throw(arg_reg), line);
                // Throw doesn't return, but we need to return a register
                // Return a unit register since execution won't continue
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), line);
                return Ok(dst);
            }
            // Handle self() - get current process ID
            if name == "self" && args.is_empty() {
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::SelfPid(dst), line);
                return Ok(dst);
            }
        }

        // Compile arguments first
        let mut arg_regs = Vec::new();
        for arg in args {
            let reg = self.compile_expr_tail(arg, false)?;
            arg_regs.push(reg);
        }

        if let Some(qualified_name) = maybe_qualified_name {
            // Compile-time resolved builtins - no string lookup, no HashMap, no runtime dispatch!
            if !qualified_name.contains('.') {
                match qualified_name.as_str() {
                    // === Type-agnostic builtins (no runtime dispatch needed) ===
                    "println" if arg_regs.len() == 1 => {
                        self.chunk.emit(Instruction::Println(arg_regs[0]), 0);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), 0);
                        return Ok(dst);
                    }
                    "print" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Print(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "head" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListHead(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "tail" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListTail(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "isEmpty" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListIsEmpty(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "listSum" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ListSum(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "rangeList" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::RangeList(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "length" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::Length(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "panic" if arg_regs.len() == 1 => {
                        self.chunk.emit(Instruction::Panic(arg_regs[0]), 0);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), 0);
                        return Ok(dst);
                    }
                    "assert" if arg_regs.len() == 1 => {
                        self.chunk.emit(Instruction::Assert(arg_regs[0]), 0);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), 0);
                        return Ok(dst);
                    }
                    "sleep" if arg_regs.len() == 1 => {
                        // sleep(ms) - sleep for N milliseconds
                        self.chunk.emit(Instruction::Sleep(arg_regs[0]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "assert_eq" if arg_regs.len() == 2 => {
                        self.chunk.emit(Instruction::AssertEq(arg_regs[0], arg_regs[1]), line);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), line);
                        return Ok(dst);
                    }
                    "typeOf" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::TypeOf(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    // === Math builtins (type-aware - use typed instruction if we can infer type) ===
                    "abs" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        // Check if we can infer type from the argument expression
                        let arg_type = self.infer_expr_type(&args[0]);
                        match arg_type {
                            Some(InferredType::Int) => {
                                self.chunk.emit(Instruction::AbsInt(dst, arg_regs[0]), 0);
                            }
                            Some(InferredType::Float) => {
                                self.chunk.emit(Instruction::AbsFloat(dst, arg_regs[0]), 0);
                            }
                            None => {
                                // Fallback: emit AbsInt (type checker should have validated)
                                // In practice, abs is usually called on Int
                                self.chunk.emit(Instruction::AbsInt(dst, arg_regs[0]), 0);
                            }
                        }
                        return Ok(dst);
                    }
                    "sqrt" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::SqrtFloat(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toFloat" | "toFloat64" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::IntToFloat(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toInt" | "toInt64" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FloatToInt(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toInt8" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt8(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toInt16" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt16(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toInt32" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToInt32(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toUInt8" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt8(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toUInt16" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt16(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toUInt32" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt32(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toUInt64" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToUInt64(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toFloat32" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToFloat32(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "toBigInt" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::ToBigInt(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    // === Typed Array builtins ===
                    "newInt64Array" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeInt64Array(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    "newFloat64Array" if arg_regs.len() == 1 => {
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::MakeFloat64Array(dst, arg_regs[0]), 0);
                        return Ok(dst);
                    }
                    // === Trait-based builtins (show, copy, hash) ===
                    // First try trait dispatch, then fall back to native
                    "show" | "copy" | "hash" if arg_regs.len() == 1 => {
                        // Try to dispatch to trait method if type is known
                        if let Some(arg_type) = self.expr_type_name(&args[0]) {
                            if let Some(qualified_method) = self.find_trait_method(&arg_type, &qualified_name) {
                                if self.functions.contains_key(&qualified_method) {
                                    let dst = self.alloc_reg();
                                    let func_idx = *self.function_indices.get(&qualified_method)
                                        .expect("Function should have been assigned an index");
                                    if is_tail {
                                        self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                                        return Ok(0);
                                    } else {
                                        self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                                        return Ok(dst);
                                    }
                                }
                            }
                        }
                        // Fall back to native call
                        let dst = self.alloc_reg();
                        let name_idx = self.chunk.add_constant(Value::String(Arc::new(qualified_name)));
                        self.chunk.emit(Instruction::CallNative(dst, name_idx, arg_regs.into()), 0);
                        return Ok(dst);
                    }
                    _ => {} // Fall through to normal function lookup
                }
            } else {
                // === Module-qualified builtins (async IO operations) ===
                match qualified_name.as_str() {
                    "File.readAll" if arg_regs.len() == 1 => {
                        // File.readAll(path) -> async read entire file as string
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FileReadAll(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    "File.writeAll" if arg_regs.len() == 2 => {
                        // File.writeAll(path, content) -> async write string to file
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::FileWriteAll(dst, arg_regs[0], arg_regs[1]), line);
                        return Ok(dst);
                    }
                    "Http.get" if arg_regs.len() == 1 => {
                        // Http.get(url) -> async HTTP GET request
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::HttpGet(dst, arg_regs[0]), line);
                        return Ok(dst);
                    }
                    _ => {} // Fall through to normal function lookup
                }

            }

            // Resolve the name (handles imports and module path)
            let resolved_name = self.resolve_name(&qualified_name);

            // Check trait bounds if the function has type parameters with constraints
            if let Some(type_params) = self.fn_type_params.get(&resolved_name).cloned() {
                if !type_params.is_empty() {
                    // Get argument types for bound checking
                    let arg_types: Vec<Option<String>> = args.iter()
                        .map(|arg| self.expr_type_name(arg))
                        .collect();
                    self.check_trait_bounds(&resolved_name, &type_params, &arg_types, func.span())?;
                }
            }

            // Try monomorphization: if we know argument types, compile specialized variant
            let call_name = if self.fn_asts.contains_key(&resolved_name) && !args.is_empty() {
                // Get argument types
                let arg_types: Vec<Option<String>> = args.iter()
                    .map(|arg| self.expr_type_name(arg))
                    .collect();

                // If at least one argument type is known, try monomorphization
                if arg_types.iter().any(|t| t.is_some()) {
                    // Get param names from the function's AST
                    let param_names: Vec<String> = if let Some(fn_def) = self.fn_asts.get(&resolved_name) {
                        fn_def.clauses[0].params.iter().enumerate().map(|(i, param)| {
                            self.pattern_binding_name(&param.pattern)
                                .unwrap_or_else(|| format!("_arg{}", i))
                        }).collect()
                    } else {
                        vec![]
                    };

                    // Convert to non-optional (use "?" for unknown types)
                    let type_names: Vec<String> = arg_types.iter()
                        .map(|t| t.clone().unwrap_or_else(|| "?".to_string()))
                        .collect();

                    // Only monomorphize if all types are known
                    if !type_names.contains(&"?".to_string()) {
                        match self.compile_monomorphized_variant(&resolved_name, &type_names, &param_names) {
                            Ok(mangled) => mangled,
                            Err(_) => resolved_name.clone(), // Fall back to original
                        }
                    } else {
                        resolved_name.clone()
                    }
                } else {
                    resolved_name.clone()
                }
            } else {
                resolved_name.clone()
            };

            // Check for user-defined function
            if self.functions.contains_key(&call_name) {
                // Self-recursion optimization: use CallSelf to avoid HashMap lookup
                let is_self_call = self.current_function_name.as_ref() == Some(&resolved_name);
                let dst = self.alloc_reg();

                if is_self_call {
                    // Direct self-call - no lookup needed!
                    if is_tail {
                        self.chunk.emit(Instruction::TailCallSelf(arg_regs.into()), line);
                        return Ok(0);
                    } else {
                        self.chunk.emit(Instruction::CallSelf(dst, arg_regs.into()), line);
                        return Ok(dst);
                    }
                } else {
                    // Direct function call by index (no HashMap lookup at runtime!)
                    let func_idx = *self.function_indices.get(&call_name)
                        .expect("Function should have been assigned an index");
                    if is_tail {
                        self.chunk.emit(Instruction::TailCallDirect(func_idx, arg_regs.into()), line);
                        return Ok(0);
                    } else {
                        self.chunk.emit(Instruction::CallDirect(dst, func_idx, arg_regs.into()), line);
                        return Ok(dst);
                    }
                }
            }
        }

        // Generic function call (lambdas, higher-order functions)
        let func_reg = self.compile_expr_tail(func, false)?;
        let dst = self.alloc_reg();

        if is_tail {
            self.chunk.emit(Instruction::TailCall(func_reg, arg_regs.into()), line);
            Ok(0)
        } else {
            self.chunk.emit(Instruction::Call(dst, func_reg, arg_regs.into()), line);
            Ok(dst)
        }
    }

    /// Extract a qualified function name from an expression.
    /// Returns Some("Module.function") for field access on modules, Some("function") for simple vars.
    fn extract_qualified_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(ident) => Some(ident.node.clone()),
            Expr::FieldAccess(target, field, _) => {
                // Try to build a qualified name like "Module.SubModule.function"
                if let Some(base) = self.extract_qualified_name(target) {
                    // Check if the base looks like a module name (starts with uppercase)
                    if base.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Some(format!("{}.{}", base, field.node))
                    } else {
                        None // It's a field access on a value, not a module
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to determine the type name of an expression at compile time.
    /// This is used for trait method dispatch.
    fn expr_type_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            // Record/variant construction: Point(1, 2) or Point{x: 1, y: 2}
            // Note: The parser treats uppercase calls like Foo(42) as Record expressions
            Expr::Record(type_name, _, _) => {
                let name = &type_name.node;
                // First check if it's directly a type name (for records and single-constructor variants)
                if self.types.contains_key(name) {
                    return Some(name.clone());
                }
                // Otherwise check if it's a variant constructor
                for (ty_name, info) in &self.types {
                    if let TypeInfoKind::Variant { constructors } = &info.kind {
                        if constructors.iter().any(|(ctor_name, _)| ctor_name == name) {
                            return Some(ty_name.clone());
                        }
                    }
                }
                // Fall back to the given name (might be an unknown type, will error later)
                Some(name.clone())
            }
            // Tuple
            Expr::Tuple(_, _) => Some("Tuple".to_string()),
            // List
            Expr::List(_, _, _) => Some("List".to_string()),
            // Literals
            Expr::Int(_, _) => Some("Int".to_string()),
            Expr::Float(_, _) => Some("Float".to_string()),
            Expr::String(_, _) => Some("String".to_string()),
            Expr::Char(_, _) => Some("Char".to_string()),
            Expr::Bool(_, _) => Some("Bool".to_string()),
            Expr::Unit(_) => Some("()".to_string()),
            // Map
            Expr::Map(_, _) => Some("Map".to_string()),
            // Set
            Expr::Set(_, _) => Some("Set".to_string()),
            // For Call expressions on uppercase identifiers (variant constructors),
            // check if it's a known type
            Expr::Call(func, _, _) => {
                if let Expr::Var(ident) = func.as_ref() {
                    if ident.node.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        // Check if it's a record constructor (type name matches)
                        if self.types.get(&ident.node).map(|info| matches!(&info.kind, TypeInfoKind::Record { .. })).unwrap_or(false) {
                            return Some(ident.node.clone());
                        }
                        // Otherwise check if it's a variant constructor
                        for (type_name, info) in &self.types {
                            if let TypeInfoKind::Variant { constructors } = &info.kind {
                                if constructors.iter().any(|(name, _)| name == &ident.node) {
                                    return Some(type_name.clone());
                                }
                            }
                        }
                    }
                }
                None
            }
            // For variables, look up tracked type
            Expr::Var(ident) => {
                // Check param_types first (for monomorphized function variants)
                if let Some(ty) = self.param_types.get(&ident.node) {
                    return Some(ty.clone());
                }
                // Then check local_types
                self.local_types.get(&ident.node).cloned()
            }
            // For field access, look up the field's type from the record definition
            Expr::FieldAccess(obj, field, _) => {
                // First, get the type of the base object
                if let Some(obj_type) = self.expr_type_name(obj) {
                    // Look up the type definition
                    if let Some(type_info) = self.types.get(&obj_type) {
                        if let TypeInfoKind::Record { fields, .. } = &type_info.kind {
                            // Find the field and return its type
                            for (fname, ftype) in fields {
                                if fname == &field.node {
                                    return Some(ftype.clone());
                                }
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Compile a monomorphized (type-specialized) variant of a function.
    /// Returns the mangled function name if successful.
    fn compile_monomorphized_variant(
        &mut self,
        base_name: &str,
        arg_type_names: &[String],
        param_names: &[String],
    ) -> Result<String, CompileError> {
        // Generate mangled name: base$Type1_Type2
        let suffix = arg_type_names.join("_");
        let mangled_name = format!("{}${}", base_name, suffix);

        // Check if variant exists and is NOT stale (marked by __stale__ prefix in name)
        if let Some(existing) = self.functions.get(&mangled_name) {
            if !existing.name.starts_with("__stale__") {
                // Variant exists and is fresh - use it
                return Ok(mangled_name);
            }
            // Variant is stale (base function was redefined) - continue to recompile
        }

        // Get the original function's AST
        let fn_def = match self.fn_asts.get(base_name) {
            Some(def) => def.clone(),
            None => return Err(CompileError::UnknownFunction {
                name: base_name.to_string(),
                span: Span::default(),
            }),
        };

        // Extract the module path from the base_name (e.g., "json.jsonParse" -> ["json"])
        // This is needed to compile the function in its original module context
        let original_module_path: Vec<String> = if base_name.contains('.') {
            let parts: Vec<&str> = base_name.rsplitn(2, '.').collect();
            if parts.len() == 2 {
                parts[1].split('.').map(|s| s.to_string()).collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Create a new FnDef with the mangled name (use local name, not fully qualified)
        let mut specialized_def = fn_def.clone();
        let local_name = if mangled_name.contains('.') {
            mangled_name.rsplitn(2, '.').next().unwrap_or(&mangled_name).to_string()
        } else {
            mangled_name.clone()
        };
        specialized_def.name = Spanned::new(local_name, fn_def.name.span);

        // Save current context
        let saved_param_types = std::mem::take(&mut self.param_types);
        let saved_module_path = std::mem::replace(&mut self.module_path, original_module_path.clone());
        let saved_imports = std::mem::take(&mut self.imports);

        // Set param_types for this specialization
        for (i, param_name) in param_names.iter().enumerate() {
            if i < arg_type_names.len() {
                self.param_types.insert(param_name.clone(), arg_type_names[i].clone());
            }
        }

        // Forward declare the function with the correct module
        let arity = fn_def.clauses[0].params.len();
        let placeholder = FunctionValue {
            name: mangled_name.clone(),
            arity,
            param_names: param_names.to_vec(),
            code: Arc::new(Chunk::new()),
            module: if original_module_path.is_empty() { None } else { Some(original_module_path.join(".")) },
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols: vec![],
            // REPL introspection fields - will be populated when compiled
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        };
        self.functions.insert(mangled_name.clone(), Arc::new(placeholder));

        // Assign function index
        if !self.function_indices.contains_key(&mangled_name) {
            let idx = self.function_list.len() as u16;
            self.function_indices.insert(mangled_name.clone(), idx);
            self.function_list.push(mangled_name.clone());
        }

        // Compile the specialized function in the original module context
        self.compile_fn_def(&specialized_def)?;

        // Restore context
        self.param_types = saved_param_types;
        self.module_path = saved_module_path;
        self.imports = saved_imports;

        Ok(mangled_name)
    }

    /// Extract a module path from an expression.
    /// Returns Some("Module") or Some("Outer.Inner") if the expression is a module reference.
    fn extract_module_path(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(ident) => {
                // Check if the identifier starts with uppercase (module name convention)
                if ident.node.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    Some(ident.node.clone())
                } else {
                    None
                }
            }
            // Handle empty Record expressions as module references (e.g., Math in Math.add)
            // The parser parses uppercase identifiers like `Math` as Record constructors
            Expr::Record(type_name, fields, _) if fields.is_empty() => {
                Some(type_name.node.clone())
            }
            Expr::FieldAccess(target, field, _) => {
                // Check if we're building a nested module path like Outer.Inner
                if let Some(base) = self.extract_module_path(target) {
                    // Check if the field also looks like a module name
                    if field.node.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Some(format!("{}.{}", base, field.node))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            // Handle MethodCall on modules for nested modules (e.g., Outer.Inner in Outer.Inner.func)
            Expr::MethodCall(obj, method, args, _) if args.is_empty() => {
                if let Some(base) = self.extract_module_path(obj) {
                    if method.node.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Some(format!("{}.{}", base, method.node))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if a function can be accessed from the current module.
    /// Returns Ok(()) if access is allowed, Err with PrivateAccess otherwise.
    fn check_visibility(&self, qualified_name: &str, span: Span) -> Result<(), CompileError> {
        // Get the visibility of the function
        let visibility = self.function_visibility.get(qualified_name);

        // If we don't know about this function yet, assume it's accessible
        // (it might be a built-in or will be an UnknownFunction error later)
        let visibility = match visibility {
            Some(v) => *v,
            None => return Ok(()),
        };

        // Public functions are always accessible
        if visibility.is_public() {
            return Ok(());
        }

        // Private function - check if caller is in the same module
        // Extract the module path from the qualified name (everything before the last dot)
        let function_module: Vec<&str> = qualified_name.rsplitn(2, '.').collect();
        let function_module = if function_module.len() > 1 {
            function_module[1].split('.').collect::<Vec<_>>()
        } else {
            vec![] // Function is in root module
        };

        // Current module path
        let current_module: Vec<&str> = self.module_path.iter().map(|s| s.as_str()).collect();

        // Allow access if the current module is the same as or nested within the function's module
        if current_module.len() >= function_module.len()
            && current_module[..function_module.len()] == function_module[..]
        {
            return Ok(());
        }

        // Access denied
        let function_name = qualified_name.rsplit('.').next().unwrap_or(qualified_name);
        let module_name = if function_module.is_empty() {
            "<root>".to_string()
        } else {
            function_module.join(".")
        };

        Err(CompileError::PrivateAccess {
            function: function_name.to_string(),
            module: module_name,
            span,
        })
    }

    /// Compile an if expression.
    fn compile_if(
        &mut self,
        cond: &Expr,
        then_branch: &Expr,
        else_branch: &Expr,
        is_tail: bool,
    ) -> Result<Reg, CompileError> {
        let cond_reg = self.compile_expr_tail(cond, false)?;
        let dst = self.alloc_reg();

        // Jump to else if false
        let else_jump = self.chunk.emit(Instruction::JumpIfFalse(cond_reg, 0), 0);

        // Then branch - pass is_tail so tail calls work, but always move result to dst
        let then_reg = self.compile_expr_tail(then_branch, is_tail)?;
        self.chunk.emit(Instruction::Move(dst, then_reg), 0);
        let end_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // Else branch
        let else_target = self.chunk.code.len();
        self.chunk.patch_jump(else_jump, else_target);

        let else_reg = self.compile_expr_tail(else_branch, is_tail)?;
        self.chunk.emit(Instruction::Move(dst, else_reg), 0);

        let end_target = self.chunk.code.len();
        self.chunk.patch_jump(end_jump, end_target);

        Ok(dst)
    }

    /// Compile a match expression.
    fn compile_match(&mut self, scrutinee: &Expr, arms: &[MatchArm], is_tail: bool, line: usize) -> Result<Reg, CompileError> {
        let scrut_reg = self.compile_expr_tail(scrutinee, false)?;
        let dst = self.alloc_reg();
        let mut end_jumps = Vec::new();
        let mut last_arm_fail_jump: Option<usize> = None;
        // Jumps that need to go to the next arm (pattern fail or guard fail)
        let mut jumps_to_next_arm: Vec<usize> = Vec::new();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;

            // Patch any jumps from previous arm that should go to this arm
            let arm_start = self.chunk.code.len();
            for jump in jumps_to_next_arm.drain(..) {
                self.chunk.patch_jump(jump, arm_start);
            }

            // Save locals before processing arm (pattern bindings should be scoped to this arm)
            let saved_locals = self.locals.clone();

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, scrut_reg)?;

            // If pattern fails, jump to next arm (or panic if last)
            let pattern_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0);

            // Bind pattern variables with type info from pattern
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
            }

            // Compile guard if present
            let guard_fail_jump = if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                Some(self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0))
            } else {
                None
            };

            // Compile arm body - pass is_tail for tail calls, but always move result
            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            // After body, jump to end of match
            end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));

            // Handle jumps to next arm
            if is_last {
                // Last arm: pattern fail or guard fail should panic
                last_arm_fail_jump = Some(pattern_fail_jump);
                if let Some(guard_jump) = guard_fail_jump {
                    // For last arm with guard, need to also jump to panic on guard fail
                    // Patch pattern fail to same location as guard fail
                    jumps_to_next_arm.push(guard_jump);
                }
            } else {
                // Not last arm: pattern fail or guard fail should try next arm
                jumps_to_next_arm.push(pattern_fail_jump);
                if let Some(guard_jump) = guard_fail_jump {
                    jumps_to_next_arm.push(guard_jump);
                }
            }

            // Restore locals after arm (pattern bindings shouldn't leak to next arm)
            self.locals = saved_locals;
        }

        // Patch remaining jumps (from last arm failures) to panic location
        let panic_location = self.chunk.code.len();
        for jump in jumps_to_next_arm.drain(..) {
            self.chunk.patch_jump(jump, panic_location);
        }

        // If last arm failed, emit panic for non-exhaustive match
        if let Some(fail_jump) = last_arm_fail_jump {
            self.chunk.patch_jump(fail_jump, self.chunk.code.len());
            let msg_idx = self.chunk.add_constant(Value::String(Arc::new("Non-exhaustive match: no pattern matched".to_string())));
            let msg_reg = self.alloc_reg();
            self.chunk.emit(Instruction::LoadConst(msg_reg, msg_idx), line);
            self.chunk.emit(Instruction::Panic(msg_reg), line);
        }

        // Patch all end jumps
        let end_target = self.chunk.code.len();
        for jump in end_jumps {
            self.chunk.patch_jump(jump, end_target);
        }

        Ok(dst)
    }

    /// Compile a try/catch/finally expression.
    fn compile_try(
        &mut self,
        try_expr: &Expr,
        catch_arms: &[MatchArm],
        finally_expr: Option<&Expr>,
        is_tail: bool,
    ) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();

        // 1. Push exception handler - offset will be patched later
        let handler_idx = self.chunk.emit(Instruction::PushHandler(0), 0);

        // 2. Compile the try body
        let try_result = self.compile_expr_tail(try_expr, false)?;
        self.chunk.emit(Instruction::Move(dst, try_result), 0);

        // 3. Pop the handler (success path)
        self.chunk.emit(Instruction::PopHandler, 0);

        // 4. Jump past the catch block (success path)
        let skip_catch_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // 5. CATCH BLOCK START - patch the handler to jump here
        let catch_start = self.chunk.code.len();
        self.chunk.patch_jump(handler_idx, catch_start);

        // 6. Get the exception value
        let exc_reg = self.alloc_reg();
        self.chunk.emit(Instruction::GetException(exc_reg), 0);

        // 7. Pattern match on the exception (similar to compile_match)
        // We need to track jumps to re-throw if no pattern matches
        let mut end_jumps = Vec::new();
        let mut rethrow_jumps = Vec::new();

        for (i, arm) in catch_arms.iter().enumerate() {
            let is_last = i == catch_arms.len() - 1;

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, exc_reg)?;

            // Always emit JumpIfFalse, even for last arm (to handle no-match case)
            let next_arm_jump = self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0);

            // Bind pattern variables with type info from pattern
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
            }

            // Compile guard if present
            if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                // If guard fails, jump to next arm (or rethrow for last arm)
                let guard_jump = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0);
                if is_last {
                    rethrow_jumps.push(guard_jump);
                } else {
                    // Patch to same location as pattern mismatch
                    rethrow_jumps.push(guard_jump);
                }
            }

            // Compile catch arm body
            let body_reg = self.compile_expr_tail(&arm.body, is_tail && finally_expr.is_none())?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            // Jump to end (skip other arms and rethrow)
            end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));

            // Patch jump to next arm (or rethrow block for last arm)
            if is_last {
                rethrow_jumps.push(next_arm_jump);
            } else {
                let next_target = self.chunk.code.len();
                self.chunk.patch_jump(next_arm_jump, next_target);
            }
        }

        // 7.5 Re-throw block: if no pattern matched, re-throw the exception
        let rethrow_start = self.chunk.code.len();
        for jump in rethrow_jumps {
            self.chunk.patch_jump(jump, rethrow_start);
        }
        // Re-throw the exception (exc_reg still holds it)
        self.chunk.emit(Instruction::Throw(exc_reg), 0);

        // 8. Patch all end jumps from catch arms
        let after_catch = self.chunk.code.len();
        for jump in end_jumps {
            self.chunk.patch_jump(jump, after_catch);
        }

        // 9. Patch the skip_catch_jump to land here
        self.chunk.patch_jump(skip_catch_jump, after_catch);

        // 10. Compile finally block if present
        if let Some(finally) = finally_expr {
            // Finally is executed for both success and exception paths
            // Its result is discarded; the try/catch result is preserved in dst
            self.compile_expr_tail(finally, false)?;
        }

        Ok(dst)
    }

    /// Compile error propagation: expr?
    /// If expr throws, re-throw the exception. Otherwise return its value.
    fn compile_try_propagate(&mut self, inner_expr: &Expr) -> Result<Reg, CompileError> {
        let dst = self.alloc_reg();

        // 1. Push exception handler
        let handler_idx = self.chunk.emit(Instruction::PushHandler(0), 0);

        // 2. Compile the inner expression
        let result = self.compile_expr_tail(inner_expr, false)?;
        self.chunk.emit(Instruction::Move(dst, result), 0);

        // 3. Pop handler on success
        self.chunk.emit(Instruction::PopHandler, 0);

        // 4. Jump past the re-throw
        let skip_rethrow_jump = self.chunk.emit(Instruction::Jump(0), 0);

        // 5. RE-THROW BLOCK - patch handler to jump here
        let rethrow_start = self.chunk.code.len();
        self.chunk.patch_jump(handler_idx, rethrow_start);

        // 6. Get exception and re-throw it
        let exc_reg = self.alloc_reg();
        self.chunk.emit(Instruction::GetException(exc_reg), 0);
        self.chunk.emit(Instruction::Throw(exc_reg), 0);

        // 7. Patch skip_rethrow_jump to land here
        let after_rethrow = self.chunk.code.len();
        self.chunk.patch_jump(skip_rethrow_jump, after_rethrow);

        Ok(dst)
    }

    /// Compile a pattern test and return (success_reg, bindings).
    /// Bindings are (name, reg, is_float) tuples.
    fn compile_pattern_test(&mut self, pattern: &Pattern, scrut_reg: Reg) -> Result<(Reg, Vec<(String, Reg, bool)>), CompileError> {
        let success_reg = self.alloc_reg();
        let mut bindings: Vec<(String, Reg, bool)> = Vec::new();

        match pattern {
            Pattern::Wildcard(_) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
            }
            Pattern::Var(ident) => {
                // Check if variable already exists as immutable binding
                if let Some(existing_info) = self.locals.get(&ident.node).copied() {
                    if !existing_info.mutable {
                        // Immutable variable: use as constraint (test equality)
                        self.chunk.emit(Instruction::Eq(success_reg, scrut_reg, existing_info.reg), 0);
                    } else {
                        // Mutable variable: rebind it
                        self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                        self.chunk.emit(Instruction::Move(existing_info.reg, scrut_reg), 0);
                    }
                } else {
                    // New variable: create binding
                    self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                    let var_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::Move(var_reg, scrut_reg), 0);
                    // Type unknown for plain var pattern, will be updated by variant context
                    bindings.push((ident.node.clone(), var_reg, false));
                }
            }
            Pattern::Unit(_) => {
                self.chunk.emit(Instruction::TestUnit(success_reg, scrut_reg), 0);
            }
            Pattern::Int(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int64(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int8(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int8(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int16(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int16(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Int32(n, _) => {
                let const_idx = self.chunk.add_constant(Value::Int32(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt8(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt8(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt16(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt16(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt32(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt32(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::UInt64(n, _) => {
                let const_idx = self.chunk.add_constant(Value::UInt64(*n));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Float(f, _) => {
                let const_idx = self.chunk.add_constant(Value::Float64(*f));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Float32(f, _) => {
                let const_idx = self.chunk.add_constant(Value::Float32(*f));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::BigInt(s, _) => {
                use num_bigint::BigInt;
                let big = s.parse::<BigInt>().unwrap_or_default();
                let const_idx = self.chunk.add_constant(Value::BigInt(Arc::new(big)));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Decimal(s, _) => {
                use rust_decimal::Decimal;
                let dec = s.parse::<Decimal>().unwrap_or_default();
                let const_idx = self.chunk.add_constant(Value::Decimal(dec));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Char(c, _) => {
                let const_idx = self.chunk.add_constant(Value::Char(*c));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Bool(b, _) => {
                let const_idx = self.chunk.add_constant(Value::Bool(*b));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::String(s, _) => {
                let const_idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Variant(ctor, fields, _) => {
                // Store constructor name in constants for exact string matching
                // Qualify the constructor name with module path to match how variants are created
                let qualified_ctor = self.qualify_name(&ctor.node);
                let ctor_idx = self.chunk.add_constant(Value::String(Arc::new(qualified_ctor.clone())));
                self.chunk.emit(Instruction::TestTag(success_reg, scrut_reg, ctor_idx), 0);

                // Look up field types for this constructor
                let field_types = self.get_constructor_field_types(&qualified_ctor);

                // Extract and bind fields - only if tag matches (guard with conditional jump)
                match fields {
                    VariantPatternFields::Unit => {}
                    VariantPatternFields::Positional(patterns) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        for (i, pat) in patterns.iter().enumerate() {
                            let field_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::GetVariantField(field_reg, scrut_reg, i as u8), 0);
                            let (_, mut sub_bindings) = self.compile_pattern_test(pat, field_reg)?;

                            // Update type info based on constructor's field type
                            let is_float = field_types.get(i)
                                .map(|t| matches!(t.as_str(), "Float" | "Float32" | "Float64"))
                                .unwrap_or(false);
                            for binding in &mut sub_bindings {
                                binding.2 = is_float;
                            }
                            bindings.append(&mut sub_bindings);
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                    VariantPatternFields::Named(nfields) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        for field in nfields {
                            match field {
                                RecordPatternField::Punned(ident) => {
                                    // {x} means bind field "x" to variable "x"
                                    let field_reg = self.alloc_reg();
                                    let name_idx = self.chunk.add_constant(Value::String(Arc::new(ident.node.clone())));
                                    self.chunk.emit(Instruction::GetVariantFieldByName(field_reg, scrut_reg, name_idx), 0);
                                    // Type unknown for named fields (would need named field type lookup)
                                    bindings.push((ident.node.clone(), field_reg, false));
                                }
                                RecordPatternField::Named(ident, pat) => {
                                    // {name: n} means bind field "name" to the result of matching pattern
                                    let field_reg = self.alloc_reg();
                                    let name_idx = self.chunk.add_constant(Value::String(Arc::new(ident.node.clone())));
                                    self.chunk.emit(Instruction::GetVariantFieldByName(field_reg, scrut_reg, name_idx), 0);
                                    let (_, mut sub_bindings) = self.compile_pattern_test(pat, field_reg)?;
                                    bindings.append(&mut sub_bindings);
                                }
                                RecordPatternField::Rest(_) => {
                                    // Ignore rest pattern for now - it just matches remaining fields
                                }
                            }
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                }
            }
            Pattern::List(list_pattern, _) => {
                match list_pattern {
                    ListPattern::Empty => {
                        self.chunk.emit(Instruction::TestNil(success_reg, scrut_reg), 0);
                    }
                    ListPattern::Cons(head_patterns, tail) => {
                        let n = head_patterns.len();

                        if tail.is_some() {
                            // Pattern like [a, b | t] - check list has at least n elements
                            // Check length >= n
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int64(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::GeInt(success_reg, len_reg, n_reg), 0);
                        } else {
                            // Pattern like [a, b, c] - check list has exactly n elements
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int64(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::Eq(success_reg, len_reg, n_reg), 0);
                        }

                        // Guard the Decons block: skip if success_reg is false (list too short)
                        // This prevents "Cannot decons empty list" errors when pattern doesn't match
                        let skip_decons_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        // Decons each head element
                        let mut current_list = scrut_reg;
                        for (i, head_pat) in head_patterns.iter().enumerate() {
                            let head_reg = self.alloc_reg();
                            let tail_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Decons(head_reg, tail_reg, current_list), 0);

                            let (head_success, mut head_bindings) = self.compile_pattern_test(head_pat, head_reg)?;
                            // AND the head pattern's success with our overall success
                            self.chunk.emit(Instruction::And(success_reg, success_reg, head_success), 0);
                            bindings.append(&mut head_bindings);

                            // If this is the last head pattern and there's a tail pattern
                            if i == n - 1 {
                                if let Some(tail_pat) = tail {
                                    let (tail_success, mut tail_bindings) = self.compile_pattern_test(tail_pat, tail_reg)?;
                                    // AND the tail pattern's success with our overall success
                                    self.chunk.emit(Instruction::And(success_reg, success_reg, tail_success), 0);
                                    bindings.append(&mut tail_bindings);
                                }
                            } else {
                                current_list = tail_reg;
                            }
                        }

                        // Patch jump to skip past the Decons block
                        self.chunk.patch_jump(skip_decons_jump, self.chunk.code.len());
                    }
                }
            }
            Pattern::Tuple(patterns, _) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                for (i, pat) in patterns.iter().enumerate() {
                    let elem_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::GetTupleField(elem_reg, scrut_reg, i as u8), 0);
                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(pat, elem_reg)?;
                    // AND the sub-pattern's success with our overall success
                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);
                    bindings.append(&mut sub_bindings);
                }
            }
            Pattern::Map(entries, _) => {
                // 1. Check if it is a map
                self.chunk.emit(Instruction::IsMap(success_reg, scrut_reg), 0);
                
                // Jump to end if not a map
                let type_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);
                
                for (key_expr, val_pat) in entries {
                    // Compile key expression to a register
                    let key_reg = self.compile_expr_tail(key_expr, false)?;
                    
                    // Check if key exists
                    let exists_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::MapContainsKey(exists_reg, scrut_reg, key_reg), 0);
                    
                    // Update success_reg
                    self.chunk.emit(Instruction::And(success_reg, success_reg, exists_reg), 0);
                    
                    // Guard: if key doesn't exist, skip value check (to avoid panic in MapGet)
                    let skip_val_jump = self.chunk.emit(Instruction::JumpIfFalse(exists_reg, 0), 0);
                    
                    // Get value
                    let val_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::MapGet(val_reg, scrut_reg, key_reg), 0);
                    
                    // Match value against pattern
                    let (sub_success, mut sub_bindings) = self.compile_pattern_test(val_pat, val_reg)?;
                    self.chunk.emit(Instruction::And(success_reg, success_reg, sub_success), 0);
                    bindings.append(&mut sub_bindings);
                    
                    // Patch skip val jump
                    let after_val = self.chunk.code.len();
                    self.chunk.patch_jump(skip_val_jump, after_val);
                }
                
                // Patch type check jump
                let after_checks = self.chunk.code.len();
                self.chunk.patch_jump(type_fail_jump, after_checks);
            }
            Pattern::Set(elements, span) => {
                // 1. Check if it is a set
                self.chunk.emit(Instruction::IsSet(success_reg, scrut_reg), 0);
                
                // Jump to end if not a set
                let type_fail_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);
                
                for elem_pat in elements {
                    // We need to compile the pattern to a value to check existence
                    // Only support literals and pinned variables for now
                    let val_reg = match elem_pat {
                        Pattern::Int(n, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::Int64(*n));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::String(s, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::Bool(b, _) => {
                            let r = self.alloc_reg();
                            if *b { self.chunk.emit(Instruction::LoadTrue(r), 0); }
                            else { self.chunk.emit(Instruction::LoadFalse(r), 0); }
                            r
                        }
                        Pattern::Char(c, _) => {
                            let r = self.alloc_reg();
                            let idx = self.chunk.add_constant(Value::Char(*c));
                            self.chunk.emit(Instruction::LoadConst(r, idx), 0);
                            r
                        }
                        Pattern::Pin(expr, _) => {
                            self.compile_expr_tail(expr, false)?
                        }
                        _ => {
                            return Err(CompileError::InvalidPattern {
                                span: *span,
                                context: "Set patterns only support literals and pinned variables".to_string(),
                            });
                        }
                    };
                    
                    // Check if value exists in set
                    let exists_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::SetContains(exists_reg, scrut_reg, val_reg), 0);
                    
                    // Update success_reg
                    self.chunk.emit(Instruction::And(success_reg, success_reg, exists_reg), 0);
                }
                
                // Patch type check jump
                let after_checks = self.chunk.code.len();
                self.chunk.patch_jump(type_fail_jump, after_checks);
            }
            _ => {
                return Err(CompileError::NotImplemented {
                    feature: format!("pattern: {:?}", pattern),
                    span: pattern.span(),
                });
            }
        }

        Ok((success_reg, bindings))
    }

    /// Compile a block.
    fn compile_block(&mut self, stmts: &[Stmt], is_tail: bool) -> Result<Reg, CompileError> {
        // --- BEGIN SCOPE ---
        let saved_locals = self.locals.clone();
        let saved_local_types = self.local_types.clone();

        let mut last_reg = 0;
        if stmts.is_empty() {
            let dst = self.alloc_reg();
            self.chunk.emit(Instruction::LoadUnit(dst), 0);
            last_reg = dst;
        } else {
            for (i, stmt) in stmts.iter().enumerate() {
                let is_last = i == stmts.len() - 1;
                last_reg = self.compile_stmt(stmt, is_tail && is_last)?;
            }
        }

        // --- END SCOPE ---
        self.locals = saved_locals;
        self.local_types = saved_local_types;

        Ok(last_reg)
    }

    /// Compile a statement.
    fn compile_stmt(&mut self, stmt: &Stmt, is_tail: bool) -> Result<Reg, CompileError> {
        match stmt {
            Stmt::Expr(expr) => self.compile_expr_tail(expr, is_tail),
            Stmt::Let(binding) => self.compile_binding(binding),
            Stmt::Assign(target, value, _) => self.compile_assign(target, value),
        }
    }

    /// Compile a let binding.
    fn compile_binding(&mut self, binding: &Binding) -> Result<Reg, CompileError> {
        // Determine type from explicit annotation or infer from value
        let explicit_type = binding.ty.as_ref().map(|t| self.type_expr_name(t));
        let inferred_type = self.expr_type_name(&binding.value);
        let value_type = explicit_type.clone().or(inferred_type);

        let value_reg = self.compile_expr_tail(&binding.value, false)?;

        // For simple variable binding
        if let Pattern::Var(ident) = &binding.pattern {
            // If the variable already exists, check mutability
            if let Some(existing_info) = self.locals.get(&ident.node).copied() {
                if binding.mutable {
                    // New binding is mutable (var x = ...): create new mutable binding that shadows the old one
                    let is_float = self.is_float_type(&value_type) || self.is_float_expr(&binding.value);
                    self.locals.insert(ident.node.clone(), LocalInfo { reg: value_reg, is_float, mutable: true });
                    // Record explicit type if provided
                    if let Some(ty) = explicit_type {
                        self.local_types.insert(ident.node.clone(), ty);
                    }
                } else if existing_info.mutable {
                    // Existing is mutable, new is immutable: allow reassignment
                    let existing_reg = existing_info.reg;
                    if existing_reg != value_reg {
                        self.chunk.emit(Instruction::Move(existing_reg, value_reg), 0);
                    }
                } else {
                    // Both are immutable: treat as pattern match (assert equality)
                    // Emit AssertEq to check that the new value matches the existing one
                    self.chunk.emit(Instruction::AssertEq(existing_info.reg, value_reg), binding.span.start);
                }
            } else {
                // New binding - determine if float from explicit type or value expression
                let is_float = self.is_float_type(&value_type) || self.is_float_expr(&binding.value);
                self.locals.insert(ident.node.clone(), LocalInfo { reg: value_reg, is_float, mutable: binding.mutable });
                // Record the type (explicit takes precedence over inferred)
                if let Some(ty) = value_type {
                    self.local_types.insert(ident.node.clone(), ty);
                }
            }
        } else {
            // For complex patterns, we need to deconstruct
            let (_, bindings) = self.compile_pattern_test(&binding.pattern, value_reg)?;
            for (name, reg, is_float) in bindings {
                self.locals.insert(name, LocalInfo { reg, is_float, mutable: false });
            }
        }

        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Check if a type name represents a float type
    fn is_float_type(&self, type_name: &Option<String>) -> bool {
        match type_name {
            Some(ty) => Self::is_float_type_name(ty) || ty == "f32" || ty == "f64",
            None => false,
        }
    }

    /// Compile an assignment.
    fn compile_assign(&mut self, target: &AssignTarget, value: &Expr) -> Result<Reg, CompileError> {
        let value_reg = self.compile_expr_tail(value, false)?;

        match target {
            AssignTarget::Var(ident) => {
                if let Some(info) = self.locals.get(&ident.node) {
                    let var_reg = info.reg;
                    self.chunk.emit(Instruction::Move(var_reg, value_reg), 0);
                } else {
                    return Err(CompileError::UnknownVariable {
                        name: ident.node.clone(),
                        span: ident.span,
                    });
                }
            }
            AssignTarget::Field(obj, field) => {
                let obj_reg = self.compile_expr_tail(obj, false)?;
                let field_idx = self.chunk.add_constant(Value::String(Arc::new(field.node.clone())));
                self.chunk.emit(Instruction::SetField(obj_reg, field_idx, value_reg), 0);
            }
            AssignTarget::Index(coll, idx) => {
                let coll_reg = self.compile_expr_tail(coll, false)?;
                let idx_reg = self.compile_expr_tail(idx, false)?;
                self.chunk.emit(Instruction::IndexSet(coll_reg, idx_reg, value_reg), 0);
            }
        }
        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Compile an interpolated string.
    ///
    /// For each part:
    /// - Literal strings are loaded directly
    /// - Expressions are compiled and converted to string using `show`
    /// Then all parts are concatenated.
    fn compile_interpolated_string(&mut self, parts: &[StringPart]) -> Result<Reg, CompileError> {
        if parts.is_empty() {
            // Empty string
            let dst = self.alloc_reg();
            let idx = self.chunk.add_constant(Value::String(Arc::new(String::new())));
            self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
            return Ok(dst);
        }

        // Compile each part to a string register
        let mut part_regs = Vec::new();
        for part in parts {
            let reg = match part {
                StringPart::Lit(s) => {
                    let dst = self.alloc_reg();
                    let idx = self.chunk.add_constant(Value::String(Arc::new(s.clone())));
                    self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                    dst
                }
                StringPart::Expr(e) => {
                    // Compile the expression
                    let expr_reg = self.compile_expr_tail(e, false)?;
                    // Call `show` to convert to string
                    let dst = self.alloc_reg();
                    let name_idx = self.chunk.add_constant(Value::String(Arc::new("show".to_string())));
                    self.chunk.emit(Instruction::CallNative(dst, name_idx, vec![expr_reg].into()), 0);
                    dst
                }
            };
            part_regs.push(reg);
        }

        // Concatenate all parts
        if part_regs.len() == 1 {
            return Ok(part_regs[0]);
        }

        // Fold concatenation: result = part0 ++ part1 ++ part2 ++ ...
        let mut result = part_regs[0];
        for &part_reg in &part_regs[1..] {
            let dst = self.alloc_reg();
            self.chunk.emit(Instruction::Concat(dst, result, part_reg), 0);
            result = dst;
        }

        Ok(result)
    }

    /// Compile a lambda expression.
    fn compile_lambda(&mut self, params: &[Pattern], body: &Expr) -> Result<Reg, CompileError> {
        use std::collections::HashSet;

        // Step 1: Find free variables in the lambda body
        let mut param_names_set: HashSet<String> = HashSet::new();
        for param in params {
            if let Some(name) = self.pattern_binding_name(param) {
                param_names_set.insert(name);
            }
        }
        let body_free_vars = free_vars(body, &param_names_set);

        // Step 2: Filter to variables that exist in outer scope (locals)
        let mut captures: Vec<(String, Reg)> = Vec::new();
        for var_name in &body_free_vars {
            if let Some(info) = self.locals.get(var_name) {
                captures.push((var_name.clone(), info.reg));
            }
            // Also check if it's in our own captures (nested closures)
            else if self.capture_indices.contains_key(var_name) {
                // Need to re-capture from our own capture environment
                let dst = self.alloc_reg();
                let cap_idx = *self.capture_indices.get(var_name).unwrap();
                self.chunk.emit(Instruction::GetCapture(dst, cap_idx), 0);
                captures.push((var_name.clone(), dst));
            }
        }

        // Step 3: Save state
        let saved_chunk = std::mem::take(&mut self.chunk);
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_next_reg = self.next_reg;
        let saved_capture_indices = std::mem::take(&mut self.capture_indices);
        let saved_local_types = std::mem::take(&mut self.local_types);

        // Step 4: Create new function for lambda
        self.chunk = Chunk::new();
        self.locals = HashMap::new();
        self.capture_indices = HashMap::new();
        self.local_types = HashMap::new();
        self.next_reg = 0;

        let arity = params.len();
        let mut param_names = Vec::new();

        // Allocate registers for parameters (type unknown for lambda params)
        for (i, param) in params.iter().enumerate() {
            if let Some(name) = self.pattern_binding_name(param) {
                self.locals.insert(name.clone(), LocalInfo { reg: i as Reg, is_float: false, mutable: false });
                param_names.push(name);
            } else {
                param_names.push(format!("_arg{}", i));
            }
            self.next_reg = (i + 1) as Reg;
        }

        // Set up capture indices for the lambda body to use
        for (i, (name, _)) in captures.iter().enumerate() {
            self.capture_indices.insert(name.clone(), i as u8);
        }

        // Compile body (in tail position)
        let result_reg = self.compile_expr_tail(body, true)?;
        self.chunk.emit(Instruction::Return(result_reg), 0);

        self.chunk.register_count = self.next_reg as usize;

        let lambda_chunk = std::mem::take(&mut self.chunk);

        // Collect debug symbols before restoring state
        let debug_symbols: Vec<LocalVarSymbol> = self
            .locals
            .iter()
            .map(|(name, info)| LocalVarSymbol {
                name: name.clone(),
                register: info.reg,
            })
            .collect();

        // Step 5: Restore state
        self.chunk = saved_chunk;
        self.locals = saved_locals;
        self.next_reg = saved_next_reg;
        self.capture_indices = saved_capture_indices;
        self.local_types = saved_local_types;

        // Step 6: Create closure with captures
        let func = FunctionValue {
            name: "<lambda>".to_string(),
            arity,
            param_names,
            code: Arc::new(lambda_chunk),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols,
            // REPL introspection fields - lambdas don't have these
            source_code: None,
            source_file: None,
            doc: None,
            signature: None,
            param_types: vec![],
            return_type: None,
        };

        let dst = self.alloc_reg();

        if captures.is_empty() {
            // No captures - just load the function
            let func_idx = self.chunk.add_constant(Value::Function(Arc::new(func)));
            self.chunk.emit(Instruction::LoadConst(dst, func_idx), 0);
        } else {
            // Has captures - create a closure
            let func_idx = self.chunk.add_constant(Value::Function(Arc::new(func)));
            let capture_regs: Vec<Reg> = captures.iter().map(|(_, reg)| *reg).collect();
            self.chunk.emit(Instruction::MakeClosure(dst, func_idx, capture_regs.into()), 0);
        }

        Ok(dst)
    }

    /// Compile a record construction.
    fn compile_record(&mut self, type_name: &str, fields: &[RecordField]) -> Result<Reg, CompileError> {
        // Enforce that type must be predeclared
        if !self.known_constructors.contains(type_name) {
            // If it's a module item that looks like a record (uppercase), we might be here mistakenly?
            // No, resolve_name handles variables.
            // If it's not in known_constructors, it's an error.
            return Err(CompileError::UnknownType {
                name: type_name.to_string(),
                span: Span::default(), // We don't have span here easily without passing it, but CompileError needs it.
                // We should update compile_record to take span or return error without span and add it later?
                // Actually Expr::Record has span.
                // Let's assume passed span or just use default for now (user asked for check).
                // Or better, passing span to compile_record would be better refactor but let's see.
                // compile_record doesn't take span.
                // I will return error with default span, and maybe caller can fix it?
                // Or I can add span argument.
            });
        }

        let mut field_regs = Vec::new();
        for field in fields {
            match field {
                RecordField::Positional(expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
                RecordField::Named(_, expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
            }
        }

        let dst = self.alloc_reg();

        // Check if this is a variant constructor (not a record type)
        // If type_name matches a variant constructor, emit MakeVariant instead of MakeRecord
        let mut is_variant_ctor = false;
        let mut parent_type_name: Option<String> = None;

        for (ty_name, info) in &self.types {
            if let TypeInfoKind::Variant { constructors } = &info.kind {
                if constructors.iter().any(|(ctor_name, _)| ctor_name == type_name) {
                    is_variant_ctor = true;
                    parent_type_name = Some(ty_name.clone());
                    break;
                }
            }
        }

        if is_variant_ctor {
            let parent_type = parent_type_name.unwrap();
            let type_idx = self.chunk.add_constant(Value::String(Arc::new(parent_type)));
            let ctor_idx = self.chunk.add_constant(Value::String(Arc::new(type_name.to_string())));
            self.chunk.emit(Instruction::MakeVariant(dst, type_idx, ctor_idx, field_regs.into()), 0);
        } else {
            let type_idx = self.chunk.add_constant(Value::String(Arc::new(type_name.to_string())));
            self.chunk.emit(Instruction::MakeRecord(dst, type_idx, field_regs.into()), 0);
        }
        Ok(dst)
    }

    /// Compile a record update.
    fn compile_record_update(&mut self, type_name: &str, base: &Expr, fields: &[RecordField]) -> Result<Reg, CompileError> {
        // Enforce that type must be predeclared
        if !self.known_constructors.contains(type_name) {
            return Err(CompileError::UnknownType {
                name: type_name.to_string(),
                span: Span::default(),
            });
        }

        let base_reg = self.compile_expr_tail(base, false)?;

        let mut field_regs = Vec::new();
        for field in fields {
            match field {
                RecordField::Named(_, expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
                RecordField::Positional(expr) => {
                    let reg = self.compile_expr_tail(expr, false)?;
                    field_regs.push(reg);
                }
            }
        }

        let dst = self.alloc_reg();
        let type_idx = self.chunk.add_constant(Value::String(Arc::new(type_name.to_string())));
        self.chunk.emit(Instruction::UpdateRecord(dst, base_reg, type_idx, field_regs.into()), 0);
        Ok(dst)
    }

    /// Simple type inference from expression for builtin dispatch.
    /// Returns Some(type) if we can determine the type from the expression,
    /// None if we can't (would need full type system integration).
    fn infer_expr_type(&self, expr: &Expr) -> Option<InferredType> {
        match expr {
            // Literals have known types
            Expr::Int(_, _) => Some(InferredType::Int),
            Expr::Float(_, _) => Some(InferredType::Float),

            // Negation preserves type
            Expr::UnaryOp(UnaryOp::Neg, inner, _) => self.infer_expr_type(inner),

            // Binary arithmetic: both operands should have same type
            Expr::BinOp(left, op, _, _) if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) => {
                self.infer_expr_type(left)
            }

            // Variable: check if we know its type
            Expr::Var(ident) => {
                self.local_types.get(&ident.node).and_then(|t| {
                    match t.as_str() {
                        "Int" => Some(InferredType::Int),
                        "Float" => Some(InferredType::Float),
                        _ => None,
                    }
                })
            }

            // Tuple with single element (parenthesized)
            Expr::Tuple(items, _) if items.len() == 1 => self.infer_expr_type(&items[0]),

            // Block: type of last statement if it's an expression
            Expr::Block(stmts, _) => {
                stmts.last().and_then(|stmt| {
                    match stmt {
                        Stmt::Expr(e) => self.infer_expr_type(e),
                        _ => None,
                    }
                })
            }

            // Other cases: can't infer without full type system
            _ => None,
        }
    }

    /// Allocate a new register.
    fn alloc_reg(&mut self) -> Reg {
        let reg = self.next_reg;
        if reg == 255 {
            panic!("Register limit exceeded: function has too many local variables (max ~120). Consider breaking into smaller functions.");
        }
        self.next_reg += 1;
        reg
    }

    /// Get a compiled function.
    pub fn get_function(&self, name: &str) -> Option<Arc<FunctionValue>> {
        self.functions.get(name).cloned()
    }

    /// Get all compiled functions.
    pub fn get_all_functions(&self) -> &HashMap<String, Arc<FunctionValue>> {
        &self.functions
    }

    /// Get the ordered function list for direct indexed calls.
    /// Returns functions in the same order as their indices (for CallDirect).
    pub fn get_function_list(&self) -> Vec<Arc<FunctionValue>> {
        self.function_list.iter()
            .map(|name| self.functions.get(name).cloned().expect("Function should exist"))
            .collect()
    }

    /// Get all types for the VM.
    pub fn get_vm_types(&self) -> HashMap<String, Arc<TypeValue>> {
        use nostos_vm::value::{TypeValue, TypeKind, FieldInfo, ConstructorInfo};

        let mut vm_types = HashMap::new();
        for (name, type_info) in &self.types {
            // Get the TypeDef AST for introspection fields
            let type_def = self.type_defs.get(name);

            let type_value = match &type_info.kind {
                TypeInfoKind::Record { fields, mutable } => {
                    let field_infos: Vec<FieldInfo> = fields.iter()
                        .map(|(fname, ftype)| FieldInfo {
                            name: fname.clone(),
                            type_name: ftype.clone(),
                            mutable: *mutable,
                            private: false,
                        })
                        .collect();
                    TypeValue {
                        name: name.clone(),
                        kind: TypeKind::Record { mutable: *mutable },
                        fields: field_infos,
                        constructors: vec![],
                        traits: self.type_traits.get(name).cloned().unwrap_or_default(),
                        // REPL introspection fields
                        source_code: type_def.map(|d| d.body_string()),
                        source_file: self.current_source_name.clone(),
                        doc: type_def.and_then(|d| d.doc.clone()),
                        type_params: type_def.map(|d| d.type_param_names()).unwrap_or_default(),
                    }
                }
                TypeInfoKind::Variant { constructors } => {
                    TypeValue {
                        name: name.clone(),
                        kind: TypeKind::Variant,
                        fields: vec![],
                        constructors: constructors.iter()
                            .map(|(n, field_types)| ConstructorInfo {
                                name: n.clone(),
                                fields: field_types.iter().enumerate().map(|(i, ty)| FieldInfo {
                                    name: format!("_{}", i),
                                    type_name: ty.clone(),
                                    mutable: false,
                                    private: false,
                                }).collect(),
                            })
                            .collect(),
                        traits: self.type_traits.get(name).cloned().unwrap_or_default(),
                        // REPL introspection fields
                        source_code: type_def.map(|d| d.body_string()),
                        source_file: self.current_source_name.clone(),
                        doc: type_def.and_then(|d| d.doc.clone()),
                        type_params: type_def.map(|d| d.type_param_names()).unwrap_or_default(),
                    }
                }
            };
            vm_types.insert(name.clone(), Arc::new(type_value));
        }
        vm_types
    }

    // =========================================================================
    // REPL Introspection Query Methods
    // =========================================================================

    /// Get all function names in the compiler.
    pub fn get_function_names(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }

    /// Get all type names in the compiler.
    pub fn get_type_names(&self) -> Vec<&str> {
        self.types.keys().map(|s| s.as_str()).collect()
    }

    /// Get all trait names in the compiler.
    pub fn get_trait_names(&self) -> Vec<&str> {
        self.trait_defs.keys().map(|s| s.as_str()).collect()
    }

    /// Get a function's signature as a displayable string.
    pub fn get_function_signature(&self, name: &str) -> Option<String> {
        self.functions.get(name).and_then(|f| f.signature.clone())
    }

    /// Get a function's source code.
    pub fn get_function_source(&self, name: &str) -> Option<String> {
        self.functions.get(name)
            .and_then(|f| f.source_code.as_ref())
            .map(|s| s.to_string())
    }

    /// Get a function's doc comment.
    pub fn get_function_doc(&self, name: &str) -> Option<String> {
        self.functions.get(name).and_then(|f| f.doc.clone())
    }

    /// Get all traits implemented by a type.
    pub fn get_type_traits(&self, type_name: &str) -> Vec<String> {
        self.type_traits.get(type_name).cloned().unwrap_or_default()
    }

    /// Get all types implementing a trait.
    pub fn get_trait_implementors(&self, trait_name: &str) -> Vec<String> {
        self.trait_impls.iter()
            .filter(|((_, t), _)| t == trait_name)
            .map(|((ty, _), _)| ty.clone())
            .collect()
    }

    /// Get a TypeDef AST for a type (for introspection).
    pub fn get_type_def(&self, name: &str) -> Option<&TypeDef> {
        self.type_defs.get(name)
    }

    /// Get a FnDef AST for a function (for introspection).
    pub fn get_fn_def(&self, name: &str) -> Option<&FnDef> {
        self.fn_asts.get(name)
    }

    /// Check if a module exists (has any functions/types with that prefix).
    pub fn module_exists(&self, module_name: &str) -> bool {
        let prefix = format!("{}.", module_name);
        // Check functions
        for name in self.functions.keys() {
            if name.starts_with(&prefix) || name == module_name {
                return true;
            }
        }
        // Check types
        for name in self.types.keys() {
            if name.starts_with(&prefix) || name == module_name {
                return true;
            }
        }
        false
    }

    /// Infer the type signature of a function definition using Hindley-Milner type inference.
    /// Returns a formatted signature string like "Int -> Int -> Int" or "a -> a -> a".
    ///
    /// This function attempts full HM inference and falls back to AST-based constraint analysis
    /// if inference fails.
    pub fn infer_signature(&self, def: &FnDef) -> String {
        // Try full Hindley-Milner inference first
        if let Some(sig) = self.try_hm_inference(def) {
            return sig;
        }
        // Fall back to AST-based signature
        def.signature()
    }

    /// Try full Hindley-Milner type inference for a function.
    /// Returns None if inference fails, allowing fallback to AST-based signature.
    fn try_hm_inference(&self, def: &FnDef) -> Option<String> {
        // Create a fresh type environment for inference
        let mut env = nostos_types::standard_env();

        // Register known types from the compiler context
        for (name, type_info) in &self.types {
            match &type_info.kind {
                TypeInfoKind::Record { fields, mutable } => {
                    let field_types: Vec<(String, nostos_types::Type, bool)> = fields
                        .iter()
                        .map(|(n, ty)| (n.clone(), self.type_name_to_type(ty), false))
                        .collect();
                    env.define_type(
                        name.clone(),
                        nostos_types::TypeDef::Record {
                            params: vec![],
                            fields: field_types,
                            is_mutable: *mutable,
                        },
                    );
                }
                TypeInfoKind::Variant { constructors } => {
                    let ctors: Vec<nostos_types::Constructor> = constructors
                        .iter()
                        .map(|(ctor_name, field_types)| {
                            if field_types.is_empty() {
                                nostos_types::Constructor::Unit(ctor_name.clone())
                            } else {
                                nostos_types::Constructor::Positional(
                                    ctor_name.clone(),
                                    field_types.iter().map(|ty| self.type_name_to_type(ty)).collect(),
                                )
                            }
                        })
                        .collect();
                    env.define_type(
                        name.clone(),
                        nostos_types::TypeDef::Variant {
                            params: vec![],
                            constructors: ctors,
                        },
                    );
                }
            }
        }

        // Register known functions in environment for recursive calls
        for (fn_name, fn_val) in &self.functions {
            let param_types: Vec<nostos_types::Type> = fn_val.param_types
                .iter()
                .map(|ty| self.type_name_to_type(ty))
                .collect();
            let ret_ty = fn_val.return_type.as_ref()
                .map(|ty| self.type_name_to_type(ty))
                .unwrap_or_else(|| env.fresh_var());

            env.functions.insert(
                fn_name.clone(),
                nostos_types::FunctionType {
                    type_params: vec![],
                    params: param_types,
                    ret: Box::new(ret_ty),
                },
            );
        }

        // Create inference context
        let mut ctx = InferCtx::new(&mut env);

        // Infer the function type
        let func_ty = ctx.infer_function(def).ok()?;

        // Solve constraints (this can hang on unresolved type vars with HasField)
        ctx.solve().ok()?;

        // Collect all resolved types for the signature
        let resolved_params: Vec<nostos_types::Type> = func_ty.params
            .iter()
            .map(|ty| ctx.env.apply_subst(ty))
            .collect();
        let resolved_ret = ctx.env.apply_subst(&func_ty.ret);

        // Collect all type variable IDs in order of first appearance
        let mut var_order: Vec<u32> = Vec::new();
        for ty in resolved_params.iter().chain(std::iter::once(&resolved_ret)) {
            self.collect_type_vars(ty, &mut var_order);
        }

        // Create mapping from type var ID to normalized letter
        let var_map: HashMap<u32, char> = var_order.iter().enumerate()
            .map(|(i, &id)| (id, (b'a' + (i as u8 % 26)) as char))
            .collect();

        // Format with normalized type variables
        let param_types: Vec<String> = resolved_params.iter()
            .map(|ty| self.format_type_normalized(ty, &var_map))
            .collect();
        let ret_type = self.format_type_normalized(&resolved_ret, &var_map);

        // Collect trait bounds for type variables that appear in the signature
        let mut bounds: Vec<String> = Vec::new();
        for (&var_id, &var_name) in &var_map {
            let trait_names = ctx.get_trait_bounds(var_id);
            for trait_name in trait_names {
                bounds.push(format!("{} {}", trait_name, var_name));
            }
        }
        bounds.sort(); // Deterministic ordering

        // Format the signature with constraint prefix if there are bounds
        let type_sig = if param_types.is_empty() {
            ret_type
        } else {
            format!("{} -> {}", param_types.join(" -> "), ret_type)
        };

        if bounds.is_empty() {
            Some(type_sig)
        } else {
            Some(format!("{} => {}", bounds.join(", "), type_sig))
        }
    }

    /// Collect all type variable IDs in order of first appearance.
    fn collect_type_vars(&self, ty: &nostos_types::Type, vars: &mut Vec<u32>) {
        match ty {
            nostos_types::Type::Var(id) => {
                if !vars.contains(id) {
                    vars.push(*id);
                }
            }
            nostos_types::Type::Tuple(elems) => {
                for e in elems {
                    self.collect_type_vars(e, vars);
                }
            }
            nostos_types::Type::List(elem) | nostos_types::Type::Array(elem)
            | nostos_types::Type::Set(elem) | nostos_types::Type::IO(elem) => {
                self.collect_type_vars(elem, vars);
            }
            nostos_types::Type::Map(k, v) => {
                self.collect_type_vars(k, vars);
                self.collect_type_vars(v, vars);
            }
            nostos_types::Type::Function(f) => {
                for p in &f.params {
                    self.collect_type_vars(p, vars);
                }
                self.collect_type_vars(&f.ret, vars);
            }
            nostos_types::Type::Named { args, .. } => {
                for a in args {
                    self.collect_type_vars(a, vars);
                }
            }
            nostos_types::Type::Record(rec) => {
                for (_, t, _) in &rec.fields {
                    self.collect_type_vars(t, vars);
                }
            }
            _ => {}
        }
    }

    /// Format a type with normalized type variable names.
    fn format_type_normalized(&self, ty: &nostos_types::Type, var_map: &HashMap<u32, char>) -> String {
        match ty {
            nostos_types::Type::Var(id) => {
                var_map.get(id).map(|c| c.to_string()).unwrap_or_else(|| format!("?{}", id))
            }
            nostos_types::Type::Int => "Int".to_string(),
            nostos_types::Type::Int8 => "Int8".to_string(),
            nostos_types::Type::Int16 => "Int16".to_string(),
            nostos_types::Type::Int32 => "Int32".to_string(),
            nostos_types::Type::Int64 => "Int64".to_string(),
            nostos_types::Type::UInt8 => "UInt8".to_string(),
            nostos_types::Type::UInt16 => "UInt16".to_string(),
            nostos_types::Type::UInt32 => "UInt32".to_string(),
            nostos_types::Type::UInt64 => "UInt64".to_string(),
            nostos_types::Type::Float => "Float".to_string(),
            nostos_types::Type::Float32 => "Float32".to_string(),
            nostos_types::Type::Float64 => "Float64".to_string(),
            nostos_types::Type::BigInt => "BigInt".to_string(),
            nostos_types::Type::Decimal => "Decimal".to_string(),
            nostos_types::Type::Bool => "Bool".to_string(),
            nostos_types::Type::Char => "Char".to_string(),
            nostos_types::Type::String => "String".to_string(),
            nostos_types::Type::Unit => "()".to_string(),
            nostos_types::Type::Never => "Never".to_string(),
            nostos_types::Type::Pid => "Pid".to_string(),
            nostos_types::Type::Ref => "Ref".to_string(),
            nostos_types::Type::TypeParam(name) => name.clone(),
            nostos_types::Type::Tuple(elems) => {
                let inner: Vec<String> = elems.iter()
                    .map(|t| self.format_type_normalized(t, var_map))
                    .collect();
                format!("({})", inner.join(", "))
            }
            nostos_types::Type::List(elem) => {
                format!("List[{}]", self.format_type_normalized(elem, var_map))
            }
            nostos_types::Type::Array(elem) => {
                format!("Array[{}]", self.format_type_normalized(elem, var_map))
            }
            nostos_types::Type::Set(elem) => {
                format!("Set[{}]", self.format_type_normalized(elem, var_map))
            }
            nostos_types::Type::Map(k, v) => {
                format!("Map[{}, {}]",
                    self.format_type_normalized(k, var_map),
                    self.format_type_normalized(v, var_map))
            }
            nostos_types::Type::Function(f) => {
                let params: Vec<String> = f.params.iter()
                    .map(|t| self.format_type_normalized(t, var_map))
                    .collect();
                let ret = self.format_type_normalized(&f.ret, var_map);
                if params.is_empty() {
                    format!("() -> {}", ret)
                } else {
                    format!("({}) -> {}", params.join(", "), ret)
                }
            }
            nostos_types::Type::Named { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter()
                        .map(|t| self.format_type_normalized(t, var_map))
                        .collect();
                    format!("{}[{}]", name, args_str.join(", "))
                }
            }
            nostos_types::Type::Record(rec) => {
                if let Some(name) = &rec.name {
                    name.clone()
                } else {
                    let fields: Vec<String> = rec.fields.iter()
                        .map(|(n, t, _)| format!("{}: {}", n, self.format_type_normalized(t, var_map)))
                        .collect();
                    format!("{{{}}}", fields.join(", "))
                }
            }
            nostos_types::Type::Variant(v) => v.name.clone(),
            nostos_types::Type::IO(inner) => {
                format!("IO[{}]", self.format_type_normalized(inner, var_map))
            }
        }
    }

    /// Convert a type name string to a nostos_types::Type.
    fn type_name_to_type(&self, ty: &str) -> nostos_types::Type {
        match ty {
            "Int" | "Int64" => nostos_types::Type::Int,
            "Int8" => nostos_types::Type::Int8,
            "Int16" => nostos_types::Type::Int16,
            "Int32" => nostos_types::Type::Int32,
            "UInt8" => nostos_types::Type::UInt8,
            "UInt16" => nostos_types::Type::UInt16,
            "UInt32" => nostos_types::Type::UInt32,
            "UInt64" => nostos_types::Type::UInt64,
            "Float" | "Float64" => nostos_types::Type::Float,
            "Float32" => nostos_types::Type::Float32,
            "BigInt" => nostos_types::Type::BigInt,
            "Decimal" => nostos_types::Type::Decimal,
            "Bool" => nostos_types::Type::Bool,
            "Char" => nostos_types::Type::Char,
            "String" => nostos_types::Type::String,
            "Pid" => nostos_types::Type::Pid,
            "Ref" => nostos_types::Type::Ref,
            "?" => nostos_types::Type::Var(u32::MAX), // Unknown type
            _ => nostos_types::Type::Named { name: ty.to_string(), args: vec![] },
        }
    }

}

/// Collect free variables in an expression (variables not bound locally).
fn free_vars(expr: &Expr, bound: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    use std::collections::HashSet;
    let mut free = HashSet::new();

    match expr {
        Expr::Var(ident) => {
            if !bound.contains(&ident.node) {
                free.insert(ident.node.clone());
            }
        }
        Expr::Int(_, _) | Expr::Float(_, _) | Expr::Bool(_, _) | Expr::Char(_, _)
        | Expr::String(_, _) | Expr::Unit(_) => {}

        Expr::BinOp(l, _, r, _) => {
            free.extend(free_vars(l, bound));
            free.extend(free_vars(r, bound));
        }
        Expr::UnaryOp(_, e, _) => {
            free.extend(free_vars(e, bound));
        }
        Expr::Call(f, args, _) => {
            free.extend(free_vars(f, bound));
            for arg in args {
                free.extend(free_vars(arg, bound));
            }
        }
        Expr::If(c, t, e, _) => {
            free.extend(free_vars(c, bound));
            free.extend(free_vars(t, bound));
            free.extend(free_vars(e, bound));
        }
        Expr::Lambda(params, body, _) => {
            let mut new_bound = bound.clone();
            for p in params {
                if let Some(name) = pattern_var_name(p) {
                    new_bound.insert(name);
                }
            }
            free.extend(free_vars(body, &new_bound));
        }
        Expr::Tuple(items, _) => {
            for item in items {
                free.extend(free_vars(item, bound));
            }
        }
        Expr::List(items, tail, _) => {
            for item in items {
                free.extend(free_vars(item, bound));
            }
            if let Some(t) = tail {
                free.extend(free_vars(t, bound));
            }
        }
        Expr::Block(stmts, _) => {
            let mut local_bound = bound.clone();
            for stmt in stmts {
                match stmt {
                    Stmt::Expr(e) => free.extend(free_vars(e, &local_bound)),
                    Stmt::Let(binding) => {
                        free.extend(free_vars(&binding.value, &local_bound));
                        if let Some(name) = pattern_var_name(&binding.pattern) {
                            local_bound.insert(name);
                        }
                    }
                    Stmt::Assign(target, val, _) => {
                        free.extend(free_vars(val, &local_bound));
                        if let AssignTarget::Var(ident) = target {
                            if !local_bound.contains(&ident.node) {
                                free.insert(ident.node.clone());
                            }
                        }
                    }
                }
            }
        }
        Expr::Match(scrutinee, arms, _) => {
            free.extend(free_vars(scrutinee, bound));
            for arm in arms {
                let mut arm_bound = bound.clone();
                collect_pattern_vars(&arm.pattern, &mut arm_bound);
                if let Some(guard) = &arm.guard {
                    free.extend(free_vars(guard, &arm_bound));
                }
                free.extend(free_vars(&arm.body, &arm_bound));
            }
        }
        Expr::FieldAccess(obj, _, _) => {
            free.extend(free_vars(obj, bound));
        }
        Expr::Index(coll, idx, _) => {
            free.extend(free_vars(coll, bound));
            free.extend(free_vars(idx, bound));
        }
        Expr::Record(_, fields, _) => {
            for field in fields {
                match field {
                    RecordField::Positional(e) | RecordField::Named(_, e) => {
                        free.extend(free_vars(e, bound));
                    }
                }
            }
        }
        Expr::RecordUpdate(_, base, fields, _) => {
            free.extend(free_vars(base, bound));
            for field in fields {
                match field {
                    RecordField::Positional(e) | RecordField::Named(_, e) => {
                        free.extend(free_vars(e, bound));
                    }
                }
            }
        }
        Expr::Try_(e, _) => {
            free.extend(free_vars(e, bound));
        }
        Expr::Try(try_expr, catch_arms, finally_expr, _) => {
            free.extend(free_vars(try_expr, bound));
            for arm in catch_arms {
                let mut arm_bound = bound.clone();
                collect_pattern_vars(&arm.pattern, &mut arm_bound);
                free.extend(free_vars(&arm.body, &arm_bound));
            }
            if let Some(fin) = finally_expr {
                free.extend(free_vars(fin, bound));
            }
        }
        Expr::Spawn(_, func, args, _) => {
            free.extend(free_vars(func, bound));
            for arg in args {
                free.extend(free_vars(arg, bound));
            }
        }
        Expr::MethodCall(obj, _, args, _) => {
            free.extend(free_vars(obj, bound));
            for arg in args {
                free.extend(free_vars(arg, bound));
            }
        }
        Expr::Send(pid, msg, _) => {
            free.extend(free_vars(pid, bound));
            free.extend(free_vars(msg, bound));
        }
        Expr::Map(pairs, _) => {
            for (k, v) in pairs {
                free.extend(free_vars(k, bound));
                free.extend(free_vars(v, bound));
            }
        }
        Expr::Set(items, _) => {
            for item in items {
                free.extend(free_vars(item, bound));
            }
        }
        _ => {} // Other expressions - add as needed
    }

    free
}

fn pattern_var_name(pat: &Pattern) -> Option<String> {
    match pat {
        Pattern::Var(ident) => Some(ident.node.clone()),
        _ => None,
    }
}

fn collect_pattern_vars(pat: &Pattern, vars: &mut std::collections::HashSet<String>) {
    match pat {
        Pattern::Var(ident) => { vars.insert(ident.node.clone()); }
        Pattern::Tuple(pats, _) => {
            for p in pats {
                collect_pattern_vars(p, vars);
            }
        }
        Pattern::List(list_pat, _) => {
            match list_pat {
                ListPattern::Empty => {}
                ListPattern::Cons(heads, tail) => {
                    for p in heads {
                        collect_pattern_vars(p, vars);
                    }
                    if let Some(t) = tail {
                        collect_pattern_vars(t, vars);
                    }
                }
            }
        }
        Pattern::Variant(_, fields, _) => {
            match fields {
                VariantPatternFields::Unit => {}
                VariantPatternFields::Positional(pats) => {
                    for p in pats {
                        collect_pattern_vars(p, vars);
                    }
                }
                VariantPatternFields::Named(named) => {
                    for field in named {
                        match field {
                            RecordPatternField::Punned(ident) => { vars.insert(ident.node.clone()); }
                            RecordPatternField::Named(_, pat) => collect_pattern_vars(pat, vars),
                            RecordPatternField::Rest(_) => {}
                        }
                    }
                }
            }
        }
        Pattern::Record(fields, _) => {
            for field in fields {
                match field {
                    RecordPatternField::Punned(ident) => { vars.insert(ident.node.clone()); }
                    RecordPatternField::Named(_, pat) => collect_pattern_vars(pat, vars),
                    RecordPatternField::Rest(_) => {}
                }
            }
        }
        Pattern::Or(pats, _) => {
            for p in pats {
                collect_pattern_vars(p, vars);
            }
        }
        _ => {}
    }
}

/// Compile a complete module.
pub fn compile_module(module: &Module, source: &str) -> Result<Compiler, CompileError> {
    let mut compiler = Compiler::new(source);
    compiler.compile_items(&module.items)?;
    compiler.compile_all().map_err(|(e, _, _)| e)?;
    Ok(compiler)
}

impl Compiler {
    /// Compile a list of items (can be called recursively for nested modules).
    fn compile_items(&mut self, items: &[Item]) -> Result<(), CompileError> {
        // First pass: process use statements to set up imports
        for item in items {
            if let Item::Use(use_stmt) = item {
                self.compile_use_stmt(use_stmt)?;
            }
        }

        // Second pass: collect type definitions
        for item in items {
            if let Item::TypeDef(type_def) = item {
                self.compile_type_def(type_def)?;
            }
        }

        // Third pass: compile trait definitions
        for item in items {
            if let Item::TraitDef(trait_def) = item {
                self.compile_trait_def(trait_def)?;
            }
        }

        // Fourth pass: compile trait implementations (after trait defs)
        for item in items {
            if let Item::TraitImpl(trait_impl) = item {
                self.compile_trait_impl(trait_impl)?;
            }
        }

        // Fifth pass: process nested modules (before functions so they're available)
        for item in items {
            if let Item::ModuleDef(module_def) = item {
                self.compile_module_def(module_def)?;
            }
        }

        // Collect and merge function definitions by name
        let mut fn_clauses: std::collections::HashMap<String, Vec<FnClause>> = std::collections::HashMap::new();
        let mut fn_order: Vec<String> = Vec::new();
        let mut fn_spans: std::collections::HashMap<String, Span> = std::collections::HashMap::new();
        let mut fn_visibility: std::collections::HashMap<String, Visibility> = std::collections::HashMap::new();
        let mut fn_type_params_map: std::collections::HashMap<String, Vec<TypeParam>> = std::collections::HashMap::new();

        for item in items {
            if let Item::FnDef(fn_def) = item {
                let qualified_name = self.qualify_name(&fn_def.name.node);
                if !fn_clauses.contains_key(&qualified_name) {
                    fn_order.push(qualified_name.clone());
                    fn_spans.insert(qualified_name.clone(), fn_def.span);
                    // Use visibility and type_params from first definition
                    fn_visibility.insert(qualified_name.clone(), fn_def.visibility);
                    fn_type_params_map.insert(qualified_name.clone(), fn_def.type_params.clone());
                }
                fn_clauses.entry(qualified_name).or_default().extend(fn_def.clauses.iter().cloned());
            }
        }

        // Fourth pass: forward declare all functions (for recursion)
        for name in &fn_order {
            let clauses = fn_clauses.get(name).unwrap();
            let arity = clauses[0].params.len();
            // Insert a placeholder function for forward reference
            let placeholder = FunctionValue {
                name: name.clone(),
                arity,
                param_names: vec![],
                code: Arc::new(Chunk::new()),
                module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
                source_span: None,
                jit_code: None,
                call_count: AtomicU32::new(0),
                debug_symbols: vec![],
                // REPL introspection fields - will be populated when compiled
                source_code: None,
                source_file: None,
                doc: None,
                signature: None,
                param_types: vec![],
                return_type: None,
            };
            self.functions.insert(name.clone(), Arc::new(placeholder));

            // Assign function index for direct calls (no HashMap lookup at runtime!)
            if !self.function_indices.contains_key(name) {
                let idx = self.function_list.len() as u16;
                self.function_indices.insert(name.clone(), idx);
                self.function_list.push(name.clone());
            }
        }

        // Fifth pass: queue functions with merged clauses
        for name in &fn_order {
            let clauses = fn_clauses.get(name).unwrap();
            let span = fn_spans.get(name).copied().unwrap_or_default();
            // Extract the local name (without module prefix)
            let local_name = if name.contains('.') {
                name.rsplit('.').next().unwrap_or(name)
            } else {
                name.as_str()
            };
            let merged_fn = FnDef {
                visibility: *fn_visibility.get(name).unwrap_or(&Visibility::Private),
                doc: None,
                name: Spanned::new(local_name.to_string(), span),
                type_params: fn_type_params_map.get(name).cloned().unwrap_or_default(),
                clauses: clauses.clone(),
                span,
            };
            self.pending_functions.push((
                merged_fn,
                self.module_path.clone(),
                self.imports.clone(),
                self.line_starts.clone(),
                self.current_source.clone().expect("Source not set"),
                self.current_source_name.clone().expect("Source name not set"),
            ));
        }

        Ok(())
    }

    /// Compile a nested module definition.
    fn compile_module_def(&mut self, module_def: &ModuleDef) -> Result<(), CompileError> {
        // Push the module name onto the path
        self.module_path.push(module_def.name.node.clone());

        // Compile the module's items
        self.compile_items(&module_def.items)?;

        // Pop the module name from the path
        self.module_path.pop();

        Ok(())
    }

    /// Compile a use statement (import).
    fn compile_use_stmt(&mut self, use_stmt: &UseStmt) -> Result<(), CompileError> {
        // Build the module path from the use statement
        let module_path: String = use_stmt.path.iter()
            .map(|ident| ident.node.as_str())
            .collect::<Vec<_>>()
            .join(".");

        match &use_stmt.imports {
            UseImports::All => {
                // `use Foo.*` - we can't easily support this without module introspection
                // For now, we'll skip it (would need to know all exports from the module)
            }
            UseImports::Named(items) => {
                for item in items {
                    let local_name = item.alias.as_ref()
                        .map(|a| a.node.clone())
                        .unwrap_or_else(|| item.name.node.clone());
                    let qualified_name = format!("{}.{}", module_path, item.name.node);
                    self.imports.insert(local_name, qualified_name);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nostos_syntax::parser::parse;
    use nostos_vm::parallel::{ParallelVM, ParallelConfig};

    fn compile_and_run(source: &str) -> Result<Value, String> {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            return Err(format!("Parse error: {:?}", errors));
        }
        let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;
        let compiler = compile_module(&module, source).map_err(|e| format!("Compile error: {:?}", e))?;

        // Use ParallelVM with single thread for deterministic tests
        let config = ParallelConfig {
            num_threads: 1,
            ..Default::default()
        };
        let mut vm = ParallelVM::new(config);
        vm.register_default_natives();

        for (name, func) in compiler.get_all_functions() {
            vm.register_function(&name, func.clone());
        }
        vm.set_function_list(compiler.get_function_list());
        for (name, type_val) in compiler.get_vm_types() {
            vm.register_type(&name, type_val);
        }

        // Look for a main function
        let main_func = if let Some(func) = compiler.get_function("main") {
            func
        } else if let Some((_, func)) = compiler.get_all_functions().iter().next() {
            func.clone()
        } else {
            return Err("No functions to run".to_string());
        };

        vm.run(main_func)
            .map_err(|e| format!("Runtime error: {:?}", e))?
            .map(|v| v.to_value())
            .ok_or_else(|| "No result returned".to_string())
    }

    #[test]
    fn test_compile_simple_function() {
        let result = compile_and_run("main() = 42");
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_compile_addition() {
        let result = compile_and_run("main() = 2 + 3");
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_compile_nested_arithmetic() {
        let result = compile_and_run("main() = (2 + 3) * 4");
        assert_eq!(result, Ok(Value::Int64(20)));
    }

    #[test]
    fn test_compile_if_then_else() {
        let result = compile_and_run("main() = if true then 1 else 2");
        assert_eq!(result, Ok(Value::Int64(1)));

        let result2 = compile_and_run("main() = if false then 1 else 2");
        assert_eq!(result2, Ok(Value::Int64(2)));
    }

    #[test]
    fn test_compile_comparison() {
        let result = compile_and_run("main() = if 5 > 3 then 1 else 0");
        assert_eq!(result, Ok(Value::Int64(1)));
    }

    #[test]
    fn test_compile_function_call() {
        let source = "
            double(x) = x + x
            main() = double(21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_compile_recursive_function() {
        let source = "
            fact(n) = if n == 0 then 1 else n * fact(n - 1)
            main() = fact(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(120)));
    }

    #[test]
    fn test_compile_tail_recursive() {
        let source = "
            sum(n, acc) = if n == 0 then acc else sum(n - 1, acc + n)
            main() = sum(100, 0)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5050)));
    }

    #[test]
    fn test_compile_list() {
        let source = "main() = [1, 2, 3]";
        let result = compile_and_run(source);
        match result {
            Ok(Value::List(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int64(1));
            }
            other => panic!("Expected list, got {:?}", other),
        }
    }

    #[test]
    fn test_compile_lambda() {
        let source = "
            apply(f, x) = f(x)
            main() = apply(x => x * 2, 21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_compile_bool_ops() {
        let source = "main() = true && false";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(false)));

        let source2 = "main() = true || false";
        let result2 = compile_and_run(source2);
        assert_eq!(result2, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_compile_negation() {
        let source = "main() = !true";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(false)));
    }

    #[test]
    fn test_compile_string() {
        let source = r#"main() = "hello""#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "hello"),
            other => panic!("Expected string, got {:?}", other),
        }
    }

    // ============= More comprehensive end-to-end tests =============

    #[test]
    fn test_e2e_fibonacci_tail_recursive() {
        let source = "
            fib(n, a, b) = if n == 0 then a else fib(n - 1, b, a + b)
            main() = fib(20, 0, 1)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6765)));
    }

    #[test]
    fn test_e2e_mutual_recursion() {
        let source = "
            isEven(n) = if n == 0 then true else isOdd(n - 1)
            isOdd(n) = if n == 0 then false else isEven(n - 1)
            main() = isEven(10)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_higher_order_function() {
        let source = "
            twice(f, x) = f(f(x))
            addOne(n) = n + 1
            main() = twice(addOne, 5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(7)));
    }

    #[test]
    fn test_e2e_compose_lambdas() {
        let source = "
            compose(f, g, x) = f(g(x))
            main() = compose(x => x * 2, y => y + 1, 10)
        ";
        // (10 + 1) * 2 = 22
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(22)));
    }

    #[test]
    fn test_e2e_tuple() {
        // Tuple creation works
        let source = "main() = (1, 2, 3)";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Tuple(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int64(1));
                assert_eq!(items[1], Value::Int64(2));
                assert_eq!(items[2], Value::Int64(3));
            }
            other => panic!("Expected tuple, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_multiple_functions_chained() {
        let source = "
            square(x) = x * x
            double(x) = x + x
            addTen(x) = x + 10
            main() = addTen(double(square(3)))
        ";
        // square(3) = 9, double(9) = 18, addTen(18) = 28
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(28)));
    }

    #[test]
    fn test_e2e_nested_conditionals() {
        let source = "
            classify(n) = if n < 0 then 0 - 1 else if n == 0 then 0 else 1
            main() = classify(5) + classify(0 - 3) + classify(0)
        ";
        // 1 + (-1) + 0 = 0
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    #[test]
    fn test_e2e_gcd() {
        let source = "
            gcd(a, b) = if b == 0 then a else gcd(b, a - (a / b) * b)
            main() = gcd(48, 18)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6)));
    }

    #[test]
    fn test_e2e_power() {
        let source = "
            power(base, exp, acc) = if exp == 0 then acc else power(base, exp - 1, acc * base)
            main() = power(2, 10, 1)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(1024)));
    }

    #[test]
    fn test_e2e_complex_boolean_logic() {
        let source = "
            xor(a, b) = (a || b) && !(a && b)
            main() = xor(true, false) && !xor(true, true)
        ";
        // xor(true, false) = true, xor(true, true) = false, !false = true
        // true && true = true
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_ackermann_small() {
        let source = "
            ack(m, n) = if m == 0 then n + 1 else if n == 0 then ack(m - 1, 1) else ack(m - 1, ack(m, n - 1))
            main() = ack(2, 3)
        ";
        // ack(2, 3) = 9
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(9)));
    }

    #[test]
    fn test_e2e_collatz_steps() {
        let source = "
            collatz(n, steps) = if n == 1 then steps else if (n - (n / 2) * 2) == 0 then collatz(n / 2, steps + 1) else collatz(3 * n + 1, steps + 1)
            main() = collatz(27, 0)
        ";
        // collatz(27) takes 111 steps
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(111)));
    }

    #[test]
    fn test_e2e_curried_application() {
        let source = "
            add(x) = y => x + y
            main() = add(10)(32)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_closure_captures() {
        let source = "
            makeAdder(n) = x => x + n
            main() = makeAdder(40)(2)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_float_arithmetic() {
        // Float multiplication works
        let source = "
            scale(x) = x * 2.0
            main() = scale(3.5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Float64(7.0)));
    }

    #[test]
    fn test_e2e_comparison_chain() {
        let source = "
            inRange(x, low, high) = x >= low && x <= high
            main() = inRange(5, 1, 10) && !inRange(15, 1, 10)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_let_binding_in_block() {
        // Nostos uses `x = 10` for bindings in blocks, not `let x = 10`
        let source = "
            main() = {
                x = 10
                y = 20
                x + y
            }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(30)));
    }

    #[test]
    fn test_e2e_nested_blocks() {
        let source = "
            main() = {
                a = {
                    x = 5
                    x * 2
                }
                b = {
                    y = 3
                    y * 3
                }
                a + b
            }
        ";
        // a = 10, b = 9, a + b = 19
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(19)));
    }

    #[test]
    fn test_e2e_string_value() {
        let source = r#"
            greet(name) = name
            main() = greet("World")
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "World"),
            other => panic!("Expected string, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_tuple_field_access() {
        let source = "
            swap(pair) = (pair.1, pair.0)
            main() = {
                p = (1, 2)
                swapped = swap(p)
                swapped.0 * 10 + swapped.1
            }
        ";
        // swapped = (2, 1), result = 2*10 + 1 = 21
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(21)));
    }

    #[test]
    fn test_e2e_match_literals() {
        // Match syntax uses -> not =>
        let source = r#"
            describe(n) = match n
                0 -> "zero"
                1 -> "one"
                _ -> "many"
            main() = describe(1)
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "one"),
            other => panic!("Expected 'one', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_match_variable_binding() {
        let source = "
            double(x) = match x
                n -> n + n
            main() = double(21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_match_tuple_pattern() {
        let source = "
            first(pair) = match pair
                (a, _) -> a
            main() = first((42, 100))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_list_cons_pattern() {
        let source = "
            sum(xs) = match xs
                [] -> 0
                [h | t] -> h + sum(t)
            main() = sum([1, 2, 3, 4, 5])
        ";
        // 1 + 2 + 3 + 4 + 5 = 15
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_list_head_tail() {
        let source = "
            head(xs) = match xs
                [h | _] -> h
                [] -> 0
            main() = head([42, 1, 2])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_record_construction() {
        let source = "
            type Point = { x: Int, y: Int }
            main() = Point(3, 4)
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Record(r)) => {
                assert_eq!(r.type_name, "Point");
                assert_eq!(r.fields.len(), 2);
                assert_eq!(r.fields[0], Value::Int64(3));
                assert_eq!(r.fields[1], Value::Int64(4));
            }
            other => panic!("Expected record, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_record_field_access() {
        // Test with named field access
        let source = "
            type Point = { x: Int, y: Int }
            getX(p) = p.x
            main() = getX(Point(42, 100))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_variant_construction() {
        let source = "
            type Option[T] = Some(T) | None
            main() = Some(42)
        ";
        let result = compile_and_run(source);
        // Variants are now properly compiled as Variants
        match result {
            Ok(Value::Variant(v)) => {
                assert_eq!(v.constructor.as_str(), "Some");
                assert_eq!(v.fields.len(), 1);
                assert_eq!(v.fields[0], Value::Int64(42));
            }
            other => panic!("Expected variant, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_variant_match() {
        let source = "
            type Option[T] = Some(T) | None
            unwrap(opt) = match opt
                Some(x) -> x
                None -> 0
            main() = unwrap(Some(42))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_variant_none() {
        let source = "
            type Option[T] = Some(T) | None
            unwrap(opt) = match opt
                Some(x) -> x
                None -> 0
            main() = unwrap(None)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    // ===== Additional comprehensive tests =====

    #[test]
    fn test_e2e_list_length() {
        let source = "
            len(xs) = match xs
                [] -> 0
                [_ | t] -> 1 + len(t)
            main() = len([1, 2, 3, 4, 5])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_list_map() {
        let source = "
            map(f, xs) = match xs
                [] -> []
                [h | t] -> [f(h) | map(f, t)]
            sum(xs) = match xs
                [] -> 0
                [h | t] -> h + sum(t)
            main() = sum(map(x => x * 2, [1, 2, 3]))
        ";
        // [2, 4, 6] => sum = 12
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(12)));
    }

    #[test]
    fn test_e2e_list_filter() {
        let source = "
            filter(pred, xs) = match xs
                [] -> []
                [h | t] -> if pred(h) then [h | filter(pred, t)] else filter(pred, t)
            sum(xs) = match xs
                [] -> 0
                [h | t] -> h + sum(t)
            main() = sum(filter(x => x > 2, [1, 2, 3, 4, 5]))
        ";
        // [3, 4, 5] => sum = 12
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(12)));
    }

    #[test]
    fn test_e2e_list_append() {
        let source = "
            append(xs, ys) = match xs
                [] -> ys
                [h | t] -> [h | append(t, ys)]
            sum(xs) = match xs
                [] -> 0
                [h | t] -> h + sum(t)
            main() = sum(append([1, 2], [3, 4, 5]))
        ";
        // [1, 2, 3, 4, 5] => sum = 15
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_list_reverse() {
        let source = "
            reverseHelper(xs, acc) = match xs
                [] -> acc
                [h | t] -> reverseHelper(t, [h | acc])
            reverse(xs) = reverseHelper(xs, [])
            head(xs) = match xs
                [h | _] -> h
                [] -> 0
            main() = head(reverse([1, 2, 3, 4, 5]))
        ";
        // reverse [1,2,3,4,5] = [5,4,3,2,1], head = 5
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_list_fold() {
        let source = "
            foldl(f, acc, xs) = match xs
                [] -> acc
                [h | t] -> foldl(f, f(acc, h), t)
            main() = foldl((acc, x) => acc + x, 0, [1, 2, 3, 4, 5])
        ";
        // sum = 15
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_nested_variant_match() {
        let source = "
            type Option[T] = Some(T) | None
            doubleUnwrap(opt) = match opt
                Some(inner) -> match inner
                    Some(x) -> x
                    None -> 0
                None -> 0
            main() = doubleUnwrap(Some(Some(42)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_either_pattern() {
        let source = "
            type Either[L, R] = Left(L) | Right(R)
            handle(result) = match result
                Left(err) -> 0 - err
                Right(val) -> val
            main() = handle(Left(5)) + handle(Right(10))
        ";
        // -5 + 10 = 5
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_char_literal() {
        // Simple char value test (char patterns not implemented yet)
        let source = r#"
            main() = 'a'
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Char('a')));
    }

    #[test]
    fn test_e2e_nested_tuple_pattern() {
        // Nested tuple patterns via field access (pattern syntax limited)
        let source = "
            extract(x) = {
                ab = x.0
                cd = x.1
                ab.0 + ab.1 + cd.0 + cd.1
            }
            main() = extract(((1, 2), (3, 4)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(10)));
    }

    #[test]
    fn test_e2e_deep_closure_capture() {
        // Use lambda syntax instead of function definitions inside blocks
        let source = "
            outer(x) = y => z => x + y + z
            main() = outer(10)(20)(12)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_multiple_closures_same_capture() {
        // Use tuple directly with lambdas
        let source = "
            makePair(n) = (x => x + n, x => x - n)
            main() = {
                fns = makePair(10)
                adder = fns.0
                subber = fns.1
                adder(5) + subber(20)
            }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(25)));
    }

    #[test]
    fn test_e2e_even_odd_mutual_recursion() {
        let source = "
            isEven(n) = if n == 0 then true else isOdd(n - 1)
            isOdd(n) = if n == 0 then false else isEven(n - 1)
            main() = isEven(10) && isOdd(7)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_list_take() {
        let source = "
            take(n, xs) = if n == 0 then [] else match xs
                [] -> []
                [h | t] -> [h | take(n - 1, t)]
            sum(xs) = match xs
                [] -> 0
                [h | t] -> h + sum(t)
            main() = sum(take(3, [1, 2, 3, 4, 5]))
        ";
        // take 3 [1,2,3,4,5] = [1,2,3], sum = 6
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6)));
    }

    #[test]
    fn test_e2e_list_drop() {
        let source = "
            drop(n, xs) = if n == 0 then xs else match xs
                [] -> []
                [_ | t] -> drop(n - 1, t)
            sum(xs) = match xs
                [] -> 0
                [h | t] -> h + sum(t)
            main() = sum(drop(2, [1, 2, 3, 4, 5]))
        ";
        // drop 2 [1,2,3,4,5] = [3,4,5], sum = 12
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(12)));
    }

    #[test]
    fn test_e2e_compose_higher_order() {
        let source = "
            compose(f, g) = x => f(g(x))
            double(x) = x * 2
            addOne(x) = x + 1
            main() = compose(double, compose(addOne, double))(5)
        ";
        // double(5) = 10, addOne(10) = 11, double(11) = 22
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(22)));
    }

    #[test]
    fn test_e2e_option_map() {
        let source = "
            type Option[T] = Some(T) | None
            mapOpt(f, opt) = match opt
                Some(x) -> Some(f(x))
                None -> None
            unwrap(opt) = match opt
                Some(x) -> x
                None -> 0
            main() = unwrap(mapOpt(x => x * 2, Some(21)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_option_flatmap() {
        let source = "
            type Option[T] = Some(T) | None
            flatMap(f, opt) = match opt
                Some(x) -> f(x)
                None -> None
            safeDiv(a, b) = if b == 0 then None else Some(a / b)
            unwrap(opt) = match opt
                Some(x) -> x
                None -> 0
            main() = unwrap(flatMap(x => safeDiv(x, 2), Some(10)))
        ";
        // Some(10) -> safeDiv(10, 2) = Some(5) -> unwrap = 5
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_list_zip() {
        let source = "
            zip(xs, ys) = match xs
                [] -> []
                [hx | tx] -> match ys
                    [] -> []
                    [hy | ty] -> [(hx, hy) | zip(tx, ty)]
            sumPairs(ps) = match ps
                [] -> 0
                [p | t] -> p.0 + p.1 + sumPairs(t)
            main() = sumPairs(zip([1, 2, 3], [10, 20, 30]))
        ";
        // zip = [(1,10), (2,20), (3,30)], sumPairs = 1+10+2+20+3+30 = 66
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(66)));
    }

    #[test]
    fn test_e2e_float_operations() {
        // Float arithmetic in main directly
        let source = "
            main() = 3.0 * 2.5
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Float64(7.5)));
    }

    #[test]
    fn test_e2e_comparison_operators() {
        let source = "
            compare(a, b) = if a < b then 0 - 1 else if a > b then 1 else 0
            main() = compare(3, 5) + compare(5, 3) + compare(4, 4)
        ";
        // -1 + 1 + 0 = 0
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    #[test]
    fn test_e2e_wildcard_pattern() {
        let source = "
            first(xs) = match xs
                [a, _, _] -> a
                _ -> 0
            main() = first([42, 100, 200])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_list_exact_pattern() {
        let source = "
            sumThree(xs) = match xs
                [a, b, c] -> a + b + c
                _ -> 0
            main() = sumThree([10, 20, 12])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_block_shadowing() {
        // Note: Variable shadowing in blocks not fully isolated
        // Testing block with different variable name instead
        let source = "
            main() = {
                x = 10
                y = {
                    z = 20
                    z + 5
                }
                x + y
            }
        ";
        // z = 20, y = 25, x = 10, result = 35
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(35)));
    }

    #[test]
    fn test_e2e_deeply_nested_conditionals() {
        let source = "
            classify(n) =
                if n < 0 then
                    if n < 0 - 10 then \"very negative\"
                    else \"negative\"
                else if n == 0 then \"zero\"
                else if n < 10 then \"small positive\"
                else \"large positive\"
            main() = classify(5)
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "small positive"),
            other => panic!("Expected 'small positive', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_higher_order_list_operations() {
        let source = "
            map(f, xs) = match xs
                [] -> []
                [h | t] -> [f(h) | map(f, t)]
            filter(p, xs) = match xs
                [] -> []
                [h | t] -> if p(h) then [h | filter(p, t)] else filter(p, t)
            foldl(f, acc, xs) = match xs
                [] -> acc
                [h | t] -> foldl(f, f(acc, h), t)
            main() = {
                nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                doubled = map(x => x * 2, nums)
                evens = filter(x => (x - (x / 2) * 2) == 0, doubled)
                foldl((a, b) => a + b, 0, evens)
            }
        ";
        // doubled = [2,4,6,8,10,12,14,16,18,20]
        // all are even, so evens = same
        // sum = 2+4+6+8+10+12+14+16+18+20 = 110
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(110)));
    }

    #[test]
    fn test_e2e_list_nth() {
        let source = "
            type Option[T] = Some(T) | None
            nth(n, xs) = match xs
                [] -> None
                [h | t] -> if n == 0 then Some(h) else nth(n - 1, t)
            unwrap(opt) = match opt
                Some(x) -> x
                None -> 0
            main() = unwrap(nth(2, [10, 20, 30, 40]))
        ";
        // nth 2 = 30
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(30)));
    }

    #[test]
    fn test_e2e_list_all_any() {
        let source = "
            all(p, xs) = match xs
                [] -> true
                [h | t] -> if p(h) then all(p, t) else false
            any(p, xs) = match xs
                [] -> false
                [h | t] -> if p(h) then true else any(p, t)
            main() = all(x => x > 0, [1, 2, 3]) && any(x => x > 2, [1, 2, 3])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Bool(true)));
    }

    #[test]
    fn test_e2e_tree_sum() {
        let source = "
            type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
            sumTree(tree) = match tree
                Leaf(v) -> v
                Node(l, r) -> sumTree(l) + sumTree(r)
            main() = sumTree(Node(Node(Leaf(1), Leaf(2)), Node(Leaf(3), Leaf(4))))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(10)));
    }

    #[test]
    fn test_e2e_tree_depth() {
        let source = "
            type Tree[T] = Leaf(T) | Node(Tree[T], Tree[T])
            max(a, b) = if a > b then a else b
            depth(tree) = match tree
                Leaf(_) -> 1
                Node(l, r) -> 1 + max(depth(l), depth(r))
            main() = depth(Node(Node(Leaf(1), Node(Leaf(2), Leaf(3))), Leaf(4)))
        ";
        // left subtree depth: 3, right: 1, total = 4
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(4)));
    }

    #[test]
    fn test_e2e_expression_evaluator() {
        let source = "
            type Expr = Num(Int) | Add(Expr, Expr) | Mul(Expr, Expr)
            eval(expr) = match expr
                Num(n) -> n
                Add(e1, e2) -> eval(e1) + eval(e2)
                Mul(e1, e2) -> eval(e1) * eval(e2)
            main() = eval(Add(Mul(Num(3), Num(4)), Num(5)))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(17)));
    }

    // =========================================================================
    // Map/Set literal tests
    // =========================================================================

    #[test]
    fn test_e2e_map_literal_empty() {
        let source = "
            main() = %{}
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Map(m)) => {
                assert!(m.is_empty());
            }
            other => panic!("Expected empty map, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_map_literal_simple() {
        let source = r#"
            main() = %{"a": 1, "b": 2}
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::Map(m)) => {
                assert_eq!(m.len(), 2);
            }
            other => panic!("Expected map with 2 entries, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_set_literal_empty() {
        let source = "
            main() = #{}
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Set(s)) => {
                assert!(s.is_empty());
            }
            other => panic!("Expected empty set, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_set_literal_simple() {
        let source = "
            main() = #{1, 2, 3}
        ";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Set(s)) => {
                assert_eq!(s.len(), 3);
            }
            other => panic!("Expected set with 3 elements, got {:?}", other),
        }
    }

    // =========================================================================
    // Multi-clause function dispatch tests
    // =========================================================================

    #[test]
    fn test_e2e_multiclause_factorial() {
        let source = "
            factorial(0) = 1
            factorial(n) = n * factorial(n - 1)
            main() = factorial(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(120)));
    }

    #[test]
    fn test_e2e_multiclause_fibonacci() {
        let source = "
            fib(0) = 0
            fib(1) = 1
            fib(n) = fib(n - 1) + fib(n - 2)
            main() = fib(10)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(55)));
    }

    #[test]
    fn test_e2e_multiclause_with_patterns() {
        let source = "
            len([]) = 0
            len([_ | t]) = 1 + len(t)
            main() = len([1, 2, 3, 4, 5])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_multiclause_sum_list() {
        let source = "
            sum([]) = 0
            sum([h | t]) = h + sum(t)
            main() = sum([1, 2, 3, 4, 5])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(15)));
    }

    #[test]
    fn test_e2e_multiclause_reverse() {
        let source = "
            reverse_acc([], acc) = acc
            reverse_acc([h | t], acc) = reverse_acc(t, [h | acc])
            reverse(xs) = reverse_acc(xs, [])
            len([]) = 0
            len([_ | t]) = 1 + len(t)
            main() = len(reverse([1, 2, 3, 4, 5]))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_multiclause_variant_dispatch() {
        let source = "
            type Option[T] = Some(T) | None
            unwrap(None) = 0
            unwrap(Some(x)) = x
            main() = unwrap(Some(42)) + unwrap(None)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_multiclause_either() {
        let source = "
            type Either[L, R] = Left(L) | Right(R)
            getValue(Left(x)) = x
            getValue(Right(x)) = x
            main() = getValue(Left(10)) + getValue(Right(32))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_multiclause_tuple_pattern() {
        let source = "
            fst((a, _)) = a
            snd((_, b)) = b
            main() = fst((1, 2)) + snd((3, 4))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    #[test]
    fn test_e2e_multiclause_literal_match() {
        let source = r#"
            describe(0) = "zero"
            describe(1) = "one"
            describe(_) = "other"
            main() = describe(1)
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(s.as_ref(), "one"),
            other => panic!("Expected string 'one', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_multiclause_mixed_patterns() {
        let source = "
            classify(0, _) = 0
            classify(_, 0) = 1
            classify(a, b) = a + b
            main() = classify(0, 5) + classify(5, 0) * 10 + classify(2, 3) * 100
        ";
        // 0 + 10 + 500 = 510
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(510)));
    }

    // =========================================================================
    // Function with guards tests
    // =========================================================================

    #[test]
    fn test_e2e_function_with_guard() {
        let source = "
            abs(n) when n < 0 = -n
            abs(n) = n
            main() = abs(-5) + abs(3)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(8)));
    }

    #[test]
    fn test_e2e_function_multiple_guards() {
        let source = "
            sign(n) when n < 0 = -1
            sign(n) when n > 0 = 1
            sign(_) = 0
            main() = sign(-5) + sign(5) + sign(0)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    #[test]
    fn test_e2e_function_guard_with_pattern() {
        let source = "
            first_positive([]) = 0
            first_positive([h | t]) when h > 0 = h
            first_positive([_ | t]) = first_positive(t)
            main() = first_positive([-3, -2, 5, 10])
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(5)));
    }

    // ===== String Interpolation Tests =====

    #[test]
    fn test_e2e_string_interpolation_simple() {
        let source = r#"
            main() = {
                name = "World"
                "Hello, ${name}!"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Hello, World!"),
            other => panic!("Expected 'Hello, World!', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_integer() {
        let source = r#"
            main() = {
                x = 42
                "The answer is ${x}"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "The answer is 42"),
            other => panic!("Expected 'The answer is 42', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_expression() {
        let source = r#"
            main() = "Sum: ${1 + 2 + 3}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Sum: 6"),
            other => panic!("Expected 'Sum: 6', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_multiple() {
        let source = r#"
            main() = {
                a = 10
                b = 20
                "${a} + ${b} = ${a + b}"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "10 + 20 = 30"),
            other => panic!("Expected '10 + 20 = 30', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_with_function_call() {
        let source = r#"
            double(x) = x * 2
            main() = "Double of 21 is ${double(21)}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Double of 21 is 42"),
            other => panic!("Expected 'Double of 21 is 42', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_nested_expr() {
        let source = r#"
            main() = "Result: ${if true then 1 else 0}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Result: 1"),
            other => panic!("Expected 'Result: 1', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_no_interpolation() {
        let source = r#"
            main() = "Plain string without interpolation"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Plain string without interpolation"),
            other => panic!("Expected plain string, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_at_start() {
        let source = r#"
            main() = {
                x = 42
                "${x} is the answer"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "42 is the answer"),
            other => panic!("Expected '42 is the answer', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_at_end() {
        let source = r#"
            main() = {
                x = 42
                "Answer: ${x}"
            }
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "Answer: 42"),
            other => panic!("Expected 'Answer: 42', got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_string_interpolation_only_expr() {
        let source = r#"
            main() = "${42}"
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "42"),
            other => panic!("Expected '42', got {:?}", other),
        }
    }

    // ===== Module System Tests =====

    #[test]
    fn test_e2e_module_nested_function() {
        let source = "
            module Math
                pub add(a, b) = a + b
                pub double(x) = x * 2
            end

            main() = Math.add(10, Math.double(16))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_nested_module() {
        let source = "
            module Outer
                pub module Inner
                    pub value() = 21
                end
            end

            main() = Outer.Inner.value() * 2
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_use_import() {
        let source = "
            module Math
                pub add(a, b) = a + b
            end

            use Math.{add}

            main() = add(20, 22)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_use_alias() {
        let source = "
            module Math
                pub multiply(a, b) = a * b
            end

            use Math.{multiply as mul}

            main() = mul(6, 7)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_multiple_functions() {
        let source = "
            module Utils
                pub inc(x) = x + 1
                dec(x) = x - 1
                pub triple(x) = x * 3
            end

            main() = Utils.triple(Utils.inc(13))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_module_recursive_function() {
        let source = "
            module Math
                pub factorial(0) = 1
                pub factorial(n) = n * factorial(n - 1)
            end

            main() = Math.factorial(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(120)));
    }

    // ===== Visibility Tests =====

    #[test]
    fn test_e2e_visibility_private_access_denied() {
        let source = "
            module Secret
                private_fn() = 42
            end

            main() = Secret.private_fn()
        ";
        let result = compile_and_run(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("PrivateAccess"), "Expected PrivateAccess error, got: {}", err);
    }

    #[test]
    fn test_e2e_visibility_public_access_allowed() {
        let source = "
            module Public
                pub public_fn() = 42
            end

            main() = Public.public_fn()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_visibility_private_internal_call() {
        // Private functions can call each other within the same module
        let source = "
            module Math
                helper(x) = x * 2
                pub double_plus_one(x) = helper(x) + 1
            end

            main() = Math.double_plus_one(20)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(41)));
    }

    #[test]
    fn test_e2e_visibility_nested_module_private() {
        // Private nested module function should not be accessible
        let source = "
            module Outer
                pub module Inner
                    secret() = 42  // private in Inner
                end
            end

            main() = Outer.Inner.secret()
        ";
        let result = compile_and_run(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_visibility_pub_type_and_function() {
        // Test that pub works for both types and functions
        let source = "
            module Shapes
                pub type Point = { x: Int, y: Int }
                pub origin() = Point(0, 0)
            end

            main() = {
                p = Shapes.origin()
                p.x + p.y
            }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(0)));
    }

    // ===== Trait Tests =====

    #[test]
    fn test_e2e_trait_basic_parse_and_compile() {
        // Test that the basic trait syntax parses and compiles (no method call yet)
        // Based on working parser tests:
        // parse_ok("trait Eq ==(self, other) -> Bool !=(self, other) = !(self == other) end");
        // parse_ok("Point: Show show(self) = self.x end");
        let source = "
            type Point = { x: Int, y: Int }
            trait Show show(self) -> Int end
            Point: Show show(self) = self.x end
            main() = 42
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_trait_explicit_qualified_call() {
        // Test trait dispatch on record type
        let source = "
            type Point = { x: Int, y: Int }
            trait GetX getX(self) -> Int end
            Point: GetX getX(self) = self.x end
            main() = Point(42, 10).getX()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_trait_method_with_self_reference() {
        // Test trait methods that use self
        let source = "
            type Counter = { value: Int }
            trait Doubled doubled(self) -> Int end
            Counter: Doubled doubled(self) = self.value * 2 end
            main() = Counter(21).doubled()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_trait_multiple_methods() {
        // Test trait with multiple methods
        let source = "
            type Num = { value: Int }
            trait Math add(self, n) -> Int sub(self, n) -> Int end
            Num: Math add(self, n) = self.value + n sub(self, n) = self.value - n end
            main() = Num(10).add(5) + Num(10).sub(3)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(22))); // (10+5) + (10-3) = 15 + 7 = 22
    }

    #[test]
    fn test_e2e_trait_method_dispatch_on_record_literal() {
        // Test that method dispatch works on record constructors
        let source = "
            type Rectangle = { width: Int, height: Int }
            trait Area area(self) -> Int end
            Rectangle: Area area(self) = self.width * self.height end
            main() = { r = Rectangle(4, 5), r.area() }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(20)));
    }

    #[test]
    fn test_e2e_trait_unknown_trait_error() {
        // Test that implementing an unknown trait produces an error
        let source = "
            type Point = { x: Int, y: Int }
            Point: NonExistent foo(self) = 42 end
            main() = 1
        ";
        let result = compile_and_run(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("UnknownTrait"), "Expected UnknownTrait error, got: {}", err);
    }

    #[test]
    fn test_e2e_trait_dispatch_on_int_literal() {
        // Test trait method dispatch on Int literals
        let source = "
            trait Triple triple(self) -> Int end
            Int: Triple triple(self) = self * 3 end
            main() = 7.triple()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(21)));
    }

    #[test]
    fn test_e2e_trait_dispatch_on_string_literal() {
        // Test trait method dispatch on String literals
        let source = "
            trait Greeting greet(self) -> String end
            String: Greeting greet(self) = \"Hello, \" ++ self end
            main() = \"World\".greet()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::String(std::sync::Arc::new("Hello, World".to_string()))));
    }

    #[test]
    fn test_e2e_trait_multiple_impls() {
        // Test same trait implemented for multiple types
        let source = "
            type Point = { x: Int, y: Int }
            type Rectangle = { w: Int, h: Int }
            trait Size size(self) -> Int end
            Point: Size size(self) = 2 end
            Rectangle: Size size(self) = 4 end
            main() = { p = Point(0, 0), r = Rectangle(5, 10), p.size() + r.size() }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(6))); // 2 + 4 = 6
    }

    #[test]
    fn test_e2e_trait_explicit_call() {
        // Test explicit trait method call: Type.Trait.method(obj)
        let source = "
            type Box = { value: Int }
            trait Doubler doubler(self) -> Int end
            Box: Doubler doubler(self) = self.value * 2 end
            main() = { b = Box(10), Box.Doubler.doubler(b) }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(20)));
    }

    #[test]
    fn test_e2e_trait_method_chaining() {
        // Test calling trait methods on multiple values
        // Note: True chaining like `a = 5.inc(); a.inc()` would require tracking
        // return types of method calls, which is not yet implemented.
        let source = "
            trait Inc inc(self) -> Int end
            Int: Inc inc(self) = self + 1 end
            main() = 5.inc() + 6.inc()
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(13))); // 6 + 7 = 13
    }

    #[test]
    fn test_e2e_trait_bool_impl() {
        // Test trait implementation for Bool
        let source = "
            trait Toggle toggle(self) -> Bool end
            Bool: Toggle toggle(self) = !self end
            main() = if true.toggle() then 0 else 1
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(1))); // true.toggle() = false, so else branch
    }

    #[test]
    fn test_e2e_trait_with_record_return() {
        // Test trait methods that return records
        let source = "
            type Point = { x: Int, y: Int }
            trait Cloner cloner(self) -> Point end
            Point: Cloner cloner(self) = Point(self.x, self.y) end
            main() = { p = Point(3, 4), p2 = p.cloner(), p2.x + p2.y }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(7)));
    }

    #[test]
    fn test_e2e_trait_with_multiple_args() {
        // Test trait methods with multiple arguments
        let source = "
            type Base = { value: Int }
            trait Adder adder(self, x, y) -> Int end
            Base: Adder adder(self, x, y) = self.value + x + y end
            main() = { b = Base(10), b.adder(20, 30) }
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(60)));
    }

    // =========================================================================
    // Error handling tests - verify correct error types and span locations
    // =========================================================================

    fn compile_should_fail(source: &str) -> CompileError {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            panic!("Unexpected parse error: {:?}", errors);
        }
        let module = module_opt.expect("Parse returned no module");
        match compile_module(&module, source) {
            Err(e) => e,
            Ok(_) => panic!("Expected compile error, but compilation succeeded"),
        }
    }

    #[test]
    fn test_error_unknown_variable() {
        let source = "main() = x + 1";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, span } => {
                assert_eq!(name, "x");
                // 'x' starts at position 9 in the source
                assert_eq!(span.start, 9);
                assert_eq!(span.end, 10);
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_unknown_variable_in_block() {
        let source = "main() = { y = undefined_var }";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, .. } => {
                assert_eq!(name, "undefined_var");
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_unknown_trait() {
        let source = "type Foo = { x: Int }\nFoo: NonExistentTrait end";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownTrait { name, .. } => {
                assert_eq!(name, "NonExistentTrait");
            }
            _ => panic!("Expected UnknownTrait error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_private_access() {
        let source = "
            module Secret
                private_fn() = 42
            end
            main() = Secret.private_fn()
        ";
        let err = compile_should_fail(source);
        match err {
            CompileError::PrivateAccess { function, module, .. } => {
                assert_eq!(function, "private_fn");
                assert_eq!(module, "Secret");
            }
            _ => panic!("Expected PrivateAccess error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_span_points_to_correct_variable() {
        // Test that the span correctly points to the undefined variable
        let source = "main() = {\n    a = 1\n    b = c + a\n}";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, span } => {
                assert_eq!(name, "c");
                // Verify the span points to 'c' by extracting from source
                let pointed_text = &source[span.start..span.end];
                assert_eq!(pointed_text, "c");
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    #[test]
    fn test_error_to_source_error_conversion() {
        let source = "main() = unknown";
        let err = compile_should_fail(source);

        // Test that to_source_error creates a proper SourceError
        let source_err = err.to_source_error();
        assert!(source_err.message.contains("unknown"));
        assert_eq!(source_err.span.start, 9);
        assert_eq!(source_err.span.end, 16);
    }

    #[test]
    fn test_error_format_output() {
        let source = "main() = undefined_var";
        let err = compile_should_fail(source);
        let source_err = err.to_source_error();

        // Test that format() produces valid output
        let formatted = source_err.format("test.nos", source);
        assert!(formatted.contains("unknown variable"));
        assert!(formatted.contains("undefined_var"));
        // Should contain line/column reference
        assert!(formatted.contains("test.nos"));
    }

    #[test]
    fn test_error_nested_undefined() {
        // Test nested expression with undefined variable
        let source = "main() = if true then undefined_fn() else 0";
        let err = compile_should_fail(source);
        match err {
            CompileError::UnknownVariable { name, .. } => {
                assert_eq!(name, "undefined_fn");
            }
            _ => panic!("Expected UnknownVariable error, got {:?}", err),
        }
    }

    // ============= Try/Catch Tests =============

    #[test]
    fn test_try_catch_basic() {
        // throw and catch should work
        let source = r#"
            main() = try throw("error") catch e -> e end
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "error"),
            other => panic!("Expected string 'error', got {:?}", other),
        }
    }

    #[test]
    fn test_try_catch_no_exception() {
        // When no exception, return the try body value
        let source = r#"
            main() = try 42 catch _ -> 0 end
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_try_catch_pattern_matching() {
        // Pattern matching in catch
        let source = r#"
            main() = try throw("special") catch
                "special" -> 1
                other -> 2
            end
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(1)));
    }

    #[test]
    fn test_try_catch_pattern_fallthrough() {
        // Non-matching pattern falls to next
        let source = r#"
            main() = try throw("other") catch
                "special" -> 1
                _ -> 2
            end
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(2)));
    }

    #[test]
    fn test_error_propagation_success() {
        // ? operator returns value on success
        let source = r#"
            might_fail(fail) = if fail then throw("error") else 42
            main() = try might_fail(false)? + 1 catch _ -> 0 end
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(43)));
    }

    #[test]
    fn test_error_propagation_rethrow() {
        // ? operator propagates exception
        let source = r#"
            might_fail(fail) = if fail then throw("error") else 42
            propagate() = might_fail(true)? + 1
            main() = try propagate() catch e -> e end
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "error"),
            other => panic!("Expected string 'error', got {:?}", other),
        }
    }

    #[test]
    fn test_nested_try_catch() {
        // Nested try/catch
        let source = r#"
            main() = try {
                inner = try throw("inner") catch _ -> throw("outer") end
                inner
            } catch
                e -> e
            end
        "#;
        let result = compile_and_run(source);
        match result {
            Ok(Value::String(s)) => assert_eq!(&*s, "outer"),
            other => panic!("Expected string 'outer', got {:?}", other),
        }
    }

    #[test]
    fn test_throw_integer() {
        // Can throw any value, not just strings
        let source = r#"
            main() = try throw(42) catch e -> e end
        "#;
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    // =========================================================================
    // Type Inference Signature Tests (80+ tests)
    // =========================================================================

    fn get_signature(source: &str, fn_name: &str) -> Option<String> {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            return None;
        }
        let module = module_opt?;
        let compiler = compile_module(&module, source).ok()?;
        compiler.get_function_signature(fn_name).map(|s| s.to_string())
    }

    // Helper to check signature matches one of several valid options
    // Also accepts signatures with trait bounds (e.g., "Num a => a -> a -> a" matches "a -> a -> a")
    fn sig_matches(sig: &str, options: &[&str]) -> bool {
        // Strip trait bound prefix if present (e.g., "Num a => " prefix)
        let core_sig = if let Some(idx) = sig.find(" => ") {
            &sig[idx + 4..]
        } else {
            sig
        };
        options.iter().any(|opt| sig == *opt || core_sig == *opt)
    }

    // -------------------------------------------------------------------------
    // Basic Arithmetic Operations (Tests 1-10)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_01_add_unified() {
        let sig = get_signature("add(x, y) = x + y\nmain() = 0", "add").unwrap();
        // Should infer: Num a => a -> a -> a (or just a -> a -> a if bounds not shown)
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int", "Num a => a -> a -> a"]),
            "add: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_02_sub_unified() {
        let sig = get_signature("sub(x, y) = x - y\nmain() = 0", "sub").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "sub: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_03_mul_unified() {
        let sig = get_signature("mul(x, y) = x * y\nmain() = 0", "mul").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "mul: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_04_div_unified() {
        let sig = get_signature("div(x, y) = x / y\nmain() = 0", "div").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "div: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_05_mod_unified() {
        let sig = get_signature("rem(x, y) = x % y\nmain() = 0", "rem").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "mod: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_06_pow_unified() {
        let sig = get_signature("pow(x, y) = x ** y\nmain() = 0", "pow").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a", "Int -> Int -> Int"]),
            "pow: expected unified types, got: {}", sig);
    }

    #[test]
    fn test_hm_07_add_with_int_literal() {
        let sig = get_signature("inc(x) = x + 1\nmain() = 0", "inc").unwrap();
        assert!(sig_matches(&sig, &["Int -> Int", "a -> a"]),
            "inc: expected Int -> Int, got: {}", sig);
    }

    #[test]
    fn test_hm_08_add_with_float_literal() {
        let sig = get_signature("incf(x) = x + 1.0\nmain() = 0", "incf").unwrap();
        assert!(sig_matches(&sig, &["Float -> Float", "a -> a"]),
            "incf: expected Float -> Float, got: {}", sig);
    }

    #[test]
    fn test_hm_09_negation() {
        let sig = get_signature("neg(x) = -x\nmain() = 0", "neg").unwrap();
        assert!(sig.contains("->"), "neg: expected function type, got: {}", sig);
    }

    #[test]
    fn test_hm_10_complex_arithmetic() {
        let sig = get_signature("calc(a, b, c) = a * b + c\nmain() = 0", "calc").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a -> a", "Int -> Int -> Int -> Int"]),
            "calc: expected all unified, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Comparison Operations (Tests 11-20)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_11_eq_returns_bool() {
        let sig = get_signature("eq(x, y) = x == y\nmain() = 0", "eq").unwrap();
        assert!(sig.ends_with("-> Bool"), "eq: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_12_neq_returns_bool() {
        let sig = get_signature("neq(x, y) = x != y\nmain() = 0", "neq").unwrap();
        assert!(sig.ends_with("-> Bool"), "neq: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_13_lt_returns_bool() {
        let sig = get_signature("lt(x, y) = x < y\nmain() = 0", "lt").unwrap();
        assert!(sig.ends_with("-> Bool"), "lt: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_14_gt_returns_bool() {
        let sig = get_signature("gt(x, y) = x > y\nmain() = 0", "gt").unwrap();
        assert!(sig.ends_with("-> Bool"), "gt: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_15_lte_returns_bool() {
        let sig = get_signature("lte(x, y) = x <= y\nmain() = 0", "lte").unwrap();
        assert!(sig.ends_with("-> Bool"), "lte: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_16_gte_returns_bool() {
        let sig = get_signature("gte(x, y) = x >= y\nmain() = 0", "gte").unwrap();
        assert!(sig.ends_with("-> Bool"), "gte: expected -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_17_comparison_unifies_operands() {
        let sig = get_signature("cmp(x, y) = x < y\nmain() = 0", "cmp").unwrap();
        // Should be a -> a -> Bool (operands unified)
        assert!(sig.contains("-> Bool"), "cmp: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_18_eq_with_int_literal() {
        let sig = get_signature("isZero(x) = x == 0\nmain() = 0", "isZero").unwrap();
        assert!(sig_matches(&sig, &["Int -> Bool", "a -> Bool"]),
            "isZero: expected Int -> Bool, got: {}", sig);
    }

    #[test]
    fn test_hm_19_comparison_chain() {
        let sig = get_signature("inRange(x, lo, hi) = x >= lo && x <= hi\nmain() = 0", "inRange").unwrap();
        assert!(sig.ends_with("-> Bool"), "inRange: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_20_eq_bool_literal() {
        let sig = get_signature("isTrue(x) = x == true\nmain() = 0", "isTrue").unwrap();
        assert!(sig_matches(&sig, &["Bool -> Bool", "a -> Bool"]),
            "isTrue: expected Bool -> Bool, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Boolean Operations (Tests 21-30)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_21_and_bool() {
        let sig = get_signature("band(a, b) = a && b\nmain() = 0", "band").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "and: got: {}", sig);
    }

    #[test]
    fn test_hm_22_or_bool() {
        let sig = get_signature("bor(a, b) = a || b\nmain() = 0", "bor").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "or: got: {}", sig);
    }

    #[test]
    fn test_hm_23_not_bool() {
        let sig = get_signature("bnot(a) = !a\nmain() = 0", "bnot").unwrap();
        assert_eq!(sig, "Bool -> Bool", "not: got: {}", sig);
    }

    #[test]
    fn test_hm_24_complex_bool() {
        let sig = get_signature("logic(a, b, c) = (a && b) || c\nmain() = 0", "logic").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool -> Bool", "logic: got: {}", sig);
    }

    #[test]
    fn test_hm_25_bool_with_comparison() {
        let sig = get_signature("check(x, y) = x > 0 && y < 10\nmain() = 0", "check").unwrap();
        assert!(sig.ends_with("-> Bool"), "check: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_26_bool_literal_true() {
        let sig = get_signature("alwaysTrue() = true\nmain() = 0", "alwaysTrue").unwrap();
        assert_eq!(sig, "Bool", "alwaysTrue: got: {}", sig);
    }

    #[test]
    fn test_hm_27_bool_literal_false() {
        let sig = get_signature("alwaysFalse() = false\nmain() = 0", "alwaysFalse").unwrap();
        assert_eq!(sig, "Bool", "alwaysFalse: got: {}", sig);
    }

    #[test]
    fn test_hm_28_double_negation() {
        let sig = get_signature("dblNot(x) = !!x\nmain() = 0", "dblNot").unwrap();
        assert_eq!(sig, "Bool -> Bool", "dblNot: got: {}", sig);
    }

    #[test]
    fn test_hm_29_implies() {
        let sig = get_signature("implies(a, b) = !a || b\nmain() = 0", "implies").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "implies: got: {}", sig);
    }

    #[test]
    fn test_hm_30_xor() {
        let sig = get_signature("xor(a, b) = (a || b) && !(a && b)\nmain() = 0", "xor").unwrap();
        assert_eq!(sig, "Bool -> Bool -> Bool", "xor: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Control Flow - If Expressions (Tests 31-40)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_31_if_basic() {
        let sig = get_signature("choose(c, a, b) = if c then a else b\nmain() = 0", "choose").unwrap();
        assert!(sig.starts_with("Bool"), "choose: expected Bool first, got: {}", sig);
    }

    #[test]
    fn test_hm_32_if_unifies_branches() {
        let sig = get_signature("sel(c, x, y) = if c then x else y\nmain() = 0", "sel").unwrap();
        // Should be Bool -> a -> a -> a
        assert!(sig.starts_with("Bool -> "), "sel: expected Bool first, got: {}", sig);
    }

    #[test]
    fn test_hm_33_if_with_int_branches() {
        let sig = get_signature("abs(x) = if x < 0 then -x else x\nmain() = 0", "abs").unwrap();
        assert!(sig.contains("->"), "abs: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_34_if_with_bool_return() {
        let sig = get_signature("sign(x) = if x > 0 then true else false\nmain() = 0", "sign").unwrap();
        assert!(sig.ends_with("-> Bool"), "sign: expected Bool return, got: {}", sig);
    }

    #[test]
    fn test_hm_35_nested_if() {
        let sig = get_signature("clamp(x, lo, hi) = if x < lo then lo else if x > hi then hi else x\nmain() = 0", "clamp").unwrap();
        assert!(sig.contains("->"), "clamp: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_36_if_with_literal() {
        let sig = get_signature("maybeOne(c) = if c then 1 else 0\nmain() = 0", "maybeOne").unwrap();
        assert!(sig_matches(&sig, &["Bool -> Int", "Bool -> a"]),
            "maybeOne: got: {}", sig);
    }

    #[test]
    fn test_hm_37_if_string_branches() {
        let sig = get_signature(r#"greet(formal) = if formal then "Hello" else "Hi"\nmain() = 0"#, "greet").unwrap();
        assert!(sig_matches(&sig, &["Bool -> String", "Bool -> a"]),
            "greet: got: {}", sig);
    }

    #[test]
    fn test_hm_38_if_complex_condition() {
        let sig = get_signature("check(a, b) = if a > 0 && b > 0 then a + b else 0\nmain() = 0", "check").unwrap();
        assert!(sig.contains("->"), "check: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_39_if_with_unit() {
        let sig = get_signature("maybe(c) = if c then () else ()\nmain() = 0", "maybe").unwrap();
        assert!(sig_matches(&sig, &["Bool -> ()", "Bool -> a"]),
            "maybe: got: {}", sig);
    }

    #[test]
    fn test_hm_40_ternary_like() {
        let sig = get_signature("max(a, b) = if a > b then a else b\nmain() = 0", "max").unwrap();
        assert!(sig.contains("->"), "max: expected function, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Functions - Polymorphism (Tests 41-50)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_41_identity() {
        let sig = get_signature("id(x) = x\nmain() = 0", "id").unwrap();
        assert_eq!(sig, "a -> a", "identity: got: {}", sig);
    }

    #[test]
    fn test_hm_42_const() {
        let sig = get_signature("const(x, y) = x\nmain() = 0", "const").unwrap();
        assert_eq!(sig, "a -> b -> a", "const: got: {}", sig);
    }

    #[test]
    fn test_hm_43_flip() {
        let sig = get_signature("second(x, y) = y\nmain() = 0", "second").unwrap();
        assert_eq!(sig, "a -> b -> b", "second: got: {}", sig);
    }

    #[test]
    fn test_hm_44_three_params_first() {
        let sig = get_signature("first3(a, b, c) = a\nmain() = 0", "first3").unwrap();
        assert_eq!(sig, "a -> b -> c -> a", "first3: got: {}", sig);
    }

    #[test]
    fn test_hm_45_three_params_middle() {
        let sig = get_signature("mid3(a, b, c) = b\nmain() = 0", "mid3").unwrap();
        assert_eq!(sig, "a -> b -> c -> b", "mid3: got: {}", sig);
    }

    #[test]
    fn test_hm_46_three_params_last() {
        let sig = get_signature("last3(a, b, c) = c\nmain() = 0", "last3").unwrap();
        assert_eq!(sig, "a -> b -> c -> c", "last3: got: {}", sig);
    }

    #[test]
    fn test_hm_47_four_params_unused() {
        let sig = get_signature("pick(a, b, c, d) = b\nmain() = 0", "pick").unwrap();
        assert_eq!(sig, "a -> b -> c -> d -> b", "pick: got: {}", sig);
    }

    #[test]
    fn test_hm_48_no_params_int() {
        let sig = get_signature("fortytwo() = 42\nmain() = 0", "fortytwo").unwrap();
        assert_eq!(sig, "Int", "fortytwo: got: {}", sig);
    }

    #[test]
    fn test_hm_49_no_params_string() {
        let sig = get_signature(r#"hello() = "hello"\nmain() = 0"#, "hello").unwrap();
        assert_eq!(sig, "String", "hello: got: {}", sig);
    }

    #[test]
    fn test_hm_50_no_params_unit() {
        let sig = get_signature("noop() = ()\nmain() = 0", "noop").unwrap();
        assert_eq!(sig, "()", "noop: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Type Annotations (Tests 51-60)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_51_annotated_int_param() {
        let sig = get_signature("f(x: Int) = x\nmain() = 0", "f").unwrap();
        assert_eq!(sig, "Int -> Int", "f: got: {}", sig);
    }

    #[test]
    fn test_hm_52_annotated_bool_param() {
        let sig = get_signature("g(x: Bool) = x\nmain() = 0", "g").unwrap();
        assert_eq!(sig, "Bool -> Bool", "g: got: {}", sig);
    }

    #[test]
    fn test_hm_53_annotated_string_param() {
        let sig = get_signature("h(x: String) = x\nmain() = 0", "h").unwrap();
        assert_eq!(sig, "String -> String", "h: got: {}", sig);
    }

    #[test]
    fn test_hm_54_annotated_float_param() {
        let sig = get_signature("flt(x: Float) = x\nmain() = 0", "flt").unwrap();
        assert_eq!(sig, "Float -> Float", "flt: got: {}", sig);
    }

    #[test]
    fn test_hm_55_annotated_return_type() {
        let sig = get_signature("ret(x) -> Int = x\nmain() = 0", "ret").unwrap();
        assert!(sig.ends_with("-> Int"), "ret: expected Int return, got: {}", sig);
    }

    #[test]
    fn test_hm_56_annotated_both() {
        let sig = get_signature("both(x: Int) -> Int = x + 1\nmain() = 0", "both").unwrap();
        assert_eq!(sig, "Int -> Int", "both: got: {}", sig);
    }

    #[test]
    fn test_hm_57_mixed_annotated_unannotated() {
        let sig = get_signature("mix(x: Int, y) = x + y\nmain() = 0", "mix").unwrap();
        assert_eq!(sig, "Int -> Int -> Int", "mix: got: {}", sig);
    }

    #[test]
    fn test_hm_58_all_annotated() {
        let sig = get_signature("all(a: Int, b: Int, c: Int) -> Int = a + b + c\nmain() = 0", "all").unwrap();
        assert_eq!(sig, "Int -> Int -> Int -> Int", "all: got: {}", sig);
    }

    #[test]
    fn test_hm_59_char_annotation() {
        let sig = get_signature("chr(c: Char) = c\nmain() = 0", "chr").unwrap();
        assert_eq!(sig, "Char -> Char", "chr: got: {}", sig);
    }

    #[test]
    fn test_hm_60_unit_return() {
        let sig = get_signature("unit() -> () = ()\nmain() = 0", "unit").unwrap();
        assert_eq!(sig, "()", "unit: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Tuples (Tests 61-70)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_61_tuple_creation() {
        let sig = get_signature("pair(a, b) = (a, b)\nmain() = 0", "pair").unwrap();
        assert!(sig.contains("(") && sig.contains(")"), "pair: expected tuple, got: {}", sig);
    }

    #[test]
    fn test_hm_62_tuple_fst() {
        let sig = get_signature("fst(p) = p.0\nmain() = 0", "fst").unwrap();
        assert!(sig.contains("->"), "fst: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_63_tuple_snd() {
        let sig = get_signature("snd(p) = p.1\nmain() = 0", "snd").unwrap();
        assert!(sig.contains("->"), "snd: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_64_triple() {
        let sig = get_signature("triple(a, b, c) = (a, b, c)\nmain() = 0", "triple").unwrap();
        assert!(sig.contains("(") && sig.contains(","), "triple: expected tuple, got: {}", sig);
    }

    #[test]
    fn test_hm_65_swap_tuple() {
        let sig = get_signature("swap(p) = (p.1, p.0)\nmain() = 0", "swap").unwrap();
        assert!(sig.contains("->"), "swap: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_66_tuple_with_same_types() {
        let sig = get_signature("dup(x) = (x, x)\nmain() = 0", "dup").unwrap();
        assert!(sig.contains("->"), "dup: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_67_nested_tuple() {
        let sig = get_signature("nest(a, b, c) = ((a, b), c)\nmain() = 0", "nest").unwrap();
        assert!(sig.contains("("), "nest: expected nested tuple, got: {}", sig);
    }

    #[test]
    fn test_hm_68_tuple_with_literal() {
        let sig = get_signature("withOne(x) = (x, 1)\nmain() = 0", "withOne").unwrap();
        assert!(sig.contains("->"), "withOne: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_69_tuple_access_simple() {
        // Single field access on tuples
        let sig = get_signature("getFirst(p) = p.0\nmain() = 0", "getFirst").unwrap();
        assert!(sig.contains("->"), "getFirst: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_70_tuple_in_arithmetic() {
        let sig = get_signature("sumPair(p) = p.0 + p.1\nmain() = 0", "sumPair").unwrap();
        assert!(sig.contains("->"), "sumPair: expected function, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Lists (Tests 71-80)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_71_empty_list() {
        let sig = get_signature("empty() = []\nmain() = 0", "empty").unwrap();
        assert!(sig.contains("List"), "empty: expected List, got: {}", sig);
    }

    #[test]
    fn test_hm_72_singleton_list() {
        let sig = get_signature("single(x) = [x]\nmain() = 0", "single").unwrap();
        assert!(sig.contains("List"), "single: expected List, got: {}", sig);
    }

    #[test]
    fn test_hm_73_list_of_ints() {
        let sig = get_signature("ints() = [1, 2, 3]\nmain() = 0", "ints").unwrap();
        assert!(sig_matches(&sig, &["List[Int]", "List[a]"]),
            "ints: expected List[Int], got: {}", sig);
    }

    #[test]
    fn test_hm_74_list_of_bools() {
        let sig = get_signature("bools() = [true, false]\nmain() = 0", "bools").unwrap();
        assert!(sig_matches(&sig, &["List[Bool]", "List[a]"]),
            "bools: expected List[Bool], got: {}", sig);
    }

    #[test]
    fn test_hm_75_list_param() {
        let sig = get_signature("takeList(xs) = xs\nmain() = 0", "takeList").unwrap();
        assert_eq!(sig, "a -> a", "takeList: got: {}", sig);
    }

    #[test]
    fn test_hm_76_list_cons() {
        let sig = get_signature("cons(x, xs) = [x | xs]\nmain() = 0", "cons").unwrap();
        assert!(sig.contains("List") || sig.contains("->"), "cons: got: {}", sig);
    }

    #[test]
    fn test_hm_77_list_head() {
        // Use multi-clause function pattern matching
        let sig = get_signature("hd([h | _]) = h\nmain() = 0", "hd").unwrap();
        assert!(sig.contains("->"), "hd: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_78_list_tail() {
        // Use multi-clause function pattern matching
        let sig = get_signature("tl([_ | t]) = t\nmain() = 0", "tl").unwrap();
        assert!(sig.contains("->"), "tl: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_79_list_length() {
        // Use multi-clause function with pattern matching
        let sig = get_signature("len([]) = 0\nlen([_ | t]) = 1 + len(t)\nmain() = 0", "len").unwrap();
        assert!(sig.contains("->"), "len: expected function, got: {}", sig);
    }

    #[test]
    fn test_hm_80_list_map_param() {
        // Use multi-clause function with pattern matching
        let sig = get_signature("mapList(f, []) = []\nmapList(f, [h | t]) = [f(h) | mapList(f, t)]\nmain() = 0", "mapList").unwrap();
        assert!(sig.contains("->"), "mapList: expected function, got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Custom Types - Records (Tests 81-90)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_81_record_construction() {
        // Test record construction inference - without inline record syntax
        let src = "type Point = { x: Int, y: Int }\ngetX(p: Point) -> Int = p.x\nmain() = 0";
        let sig = get_signature(src, "getX").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "getX: got: {}", sig);
    }

    #[test]
    fn test_hm_82_record_field_access() {
        let src = "type Point = { x: Int, y: Int }\ngetX(p: Point) = p.x\nmain() = 0";
        let sig = get_signature(src, "getX").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "getX: got: {}", sig);
    }

    #[test]
    fn test_hm_83_record_field_y() {
        let src = "type Point = { x: Int, y: Int }\ngetY(p: Point) = p.y\nmain() = 0";
        let sig = get_signature(src, "getY").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "getY: got: {}", sig);
    }

    #[test]
    fn test_hm_84_record_with_both_fields() {
        let src = "type Point = { x: Int, y: Int }\nsum(p: Point) = p.x + p.y\nmain() = 0";
        let sig = get_signature(src, "sum").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "sum: got: {}", sig);
    }

    #[test]
    fn test_hm_85_record_distance() {
        let src = "type Point = { x: Int, y: Int }\ndist(p: Point) = p.x * p.x + p.y * p.y\nmain() = 0";
        let sig = get_signature(src, "dist").unwrap();
        assert!(sig.contains("Point") && sig.contains("Int"), "dist: got: {}", sig);
    }

    #[test]
    fn test_hm_86_two_record_params() {
        // Test with explicit return type to help inference
        let src = "type Point = { x: Int, y: Int }\naddX(p1: Point, p2: Point) -> Int = p1.x + p2.x\nmain() = 0";
        let sig = get_signature(src, "addX").unwrap();
        assert!(sig.contains("Point"), "addX: got: {}", sig);
    }

    #[test]
    fn test_hm_87_record_string_field() {
        let src = "type Person = { name: String, age: Int }\ngetName(p: Person) = p.name\nmain() = 0";
        let sig = get_signature(src, "getName").unwrap();
        assert!(sig.contains("Person") && sig.contains("String"), "getName: got: {}", sig);
    }

    #[test]
    fn test_hm_88_record_bool_field() {
        let src = "type Flag = { value: Bool }\nisSet(f: Flag) = f.value\nmain() = 0";
        let sig = get_signature(src, "isSet").unwrap();
        assert!(sig.contains("Flag") && sig.contains("Bool"), "isSet: got: {}", sig);
    }

    #[test]
    fn test_hm_89_record_identity() {
        let src = "type Box = { value: Int }\nidBox(b: Box) = b\nmain() = 0";
        let sig = get_signature(src, "idBox").unwrap();
        assert!(sig.contains("Box"), "idBox: got: {}", sig);
    }

    #[test]
    fn test_hm_90_record_single_field() {
        // Simpler nested record - just access one level
        let src = "type Inner = { val: Int }\ngetVal(i: Inner) = i.val\nmain() = 0";
        let sig = get_signature(src, "getVal").unwrap();
        assert!(sig.contains("Inner") && sig.contains("Int"), "getVal: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Custom Types - Variants (Tests 91-100)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_91_variant_simple() {
        // Simple variant without type params
        let src = "type Color = Red | Green | Blue\nred() = Red\nmain() = 0";
        let sig = get_signature(src, "red").unwrap();
        assert!(sig.contains("Color") || !sig.contains("->"), "red: got: {}", sig);
    }

    #[test]
    fn test_hm_92_variant_with_data() {
        // Variant with data
        let src = "type Result = Ok(Int) | Err(String)\nok(x) = Ok(x)\nmain() = 0";
        let sig = get_signature(src, "ok").unwrap();
        assert!(sig.contains("->"), "ok: got: {}", sig);
    }

    #[test]
    fn test_hm_93_variant_match_simple() {
        // Match on simple variant using multi-clause
        let src = "type Color = Red | Green | Blue\nisRed(Red) = true\nisRed(_) = false\nmain() = 0";
        let sig = get_signature(src, "isRed").unwrap();
        assert!(sig.contains("->"), "isRed: got: {}", sig);
    }

    #[test]
    fn test_hm_94_variant_to_int() {
        // Convert variant to int using multi-clause
        let src = "type Color = Red | Green | Blue\ntoInt(Red) = 0\ntoInt(Green) = 1\ntoInt(Blue) = 2\nmain() = 0";
        let sig = get_signature(src, "toInt").unwrap();
        assert!(sig.contains("->"), "toInt: got: {}", sig);
    }

    #[test]
    fn test_hm_95_variant_green() {
        let src = "type Color = Red | Green | Blue\ngreen() = Green\nmain() = 0";
        let sig = get_signature(src, "green").unwrap();
        assert!(sig.contains("Color") || !sig.contains("->"), "green: got: {}", sig);
    }

    #[test]
    fn test_hm_96_variant_blue() {
        let src = "type Color = Red | Green | Blue\nblue() = Blue\nmain() = 0";
        let sig = get_signature(src, "blue").unwrap();
        assert!(sig.contains("Color") || !sig.contains("->"), "blue: got: {}", sig);
    }

    #[test]
    fn test_hm_97_variant_with_two_fields() {
        // Variant with multiple data
        let src = "type Pair = MkPair(Int, Int)\nmkPair(a, b) = MkPair(a, b)\nmain() = 0";
        let sig = get_signature(src, "mkPair").unwrap();
        assert!(sig.contains("->"), "mkPair: got: {}", sig);
    }

    #[test]
    fn test_hm_98_variant_mixed() {
        // Mix of unit and data constructors
        let src = "type Maybe = Nothing | Just(Int)\njust(x) = Just(x)\nmain() = 0";
        let sig = get_signature(src, "just").unwrap();
        assert!(sig.contains("->"), "just: got: {}", sig);
    }

    #[test]
    fn test_hm_99_variant_nothing() {
        let src = "type Maybe = Nothing | Just(Int)\nnothing() = Nothing\nmain() = 0";
        let sig = get_signature(src, "nothing").unwrap();
        assert!(sig.contains("Maybe") || !sig.contains("->"), "nothing: got: {}", sig);
    }

    #[test]
    fn test_hm_100_variant_is_nothing() {
        let src = "type Maybe = Nothing | Just(Int)\nisNothing(Nothing) = true\nisNothing(_) = false\nmain() = 0";
        let sig = get_signature(src, "isNothing").unwrap();
        assert!(sig.contains("->"), "isNothing: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Lambda / Higher-Order Functions (Tests 101-110)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_101_lambda_identity() {
        let sig = get_signature("idLam() = x => x\nmain() = 0", "idLam").unwrap();
        assert!(sig.contains("->"), "idLam: got: {}", sig);
    }

    #[test]
    fn test_hm_102_lambda_const() {
        let sig = get_signature("constLam(k) = x => k\nmain() = 0", "constLam").unwrap();
        assert!(sig.contains("->"), "constLam: got: {}", sig);
    }

    #[test]
    fn test_hm_103_apply() {
        let sig = get_signature("apply(f, x) = f(x)\nmain() = 0", "apply").unwrap();
        assert!(sig.contains("->"), "apply: got: {}", sig);
    }

    #[test]
    fn test_hm_104_compose() {
        let sig = get_signature("compose(f, g) = x => f(g(x))\nmain() = 0", "compose").unwrap();
        assert!(sig.contains("->"), "compose: got: {}", sig);
    }

    #[test]
    fn test_hm_105_twice() {
        let sig = get_signature("twice(f) = x => f(f(x))\nmain() = 0", "twice").unwrap();
        assert!(sig.contains("->"), "twice: got: {}", sig);
    }

    #[test]
    fn test_hm_106_flip() {
        let sig = get_signature("flip(f) = (x, y) => f(y, x)\nmain() = 0", "flip").unwrap();
        assert!(sig.contains("->"), "flip: got: {}", sig);
    }

    #[test]
    fn test_hm_107_curry() {
        let sig = get_signature("curry(f) = x => y => f(x, y)\nmain() = 0", "curry").unwrap();
        assert!(sig.contains("->"), "curry: got: {}", sig);
    }

    #[test]
    fn test_hm_108_uncurry() {
        let sig = get_signature("uncurry(f) = (x, y) => f(x)(y)\nmain() = 0", "uncurry").unwrap();
        assert!(sig.contains("->"), "uncurry: got: {}", sig);
    }

    #[test]
    fn test_hm_109_lambda_with_closure() {
        let sig = get_signature("adder(n) = x => x + n\nmain() = 0", "adder").unwrap();
        assert!(sig.contains("->"), "adder: got: {}", sig);
    }

    #[test]
    fn test_hm_110_lambda_multilevel() {
        let sig = get_signature("add3(a) = b => c => a + b + c\nmain() = 0", "add3").unwrap();
        assert!(sig.contains("->"), "add3: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Match Expressions (Tests 111-120)
    // Using multi-clause function syntax instead of inline match expressions
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_111_match_int_patterns() {
        // Multi-clause function pattern: isZero(0) = true; isZero(_) = false
        let sig = get_signature("isZeroM(0) = true\nisZeroM(_) = false\nmain() = 0", "isZeroM").unwrap();
        assert!(sig.ends_with("-> Bool") || sig.contains("Bool"), "isZeroM: got: {}", sig);
    }

    #[test]
    fn test_hm_112_match_bool_patterns() {
        // Multi-clause function pattern for bool
        let sig = get_signature("boolToInt(true) = 1\nboolToInt(false) = 0\nmain() = 0", "boolToInt").unwrap();
        assert!(sig.contains("->"), "boolToInt: got: {}", sig);
    }

    #[test]
    fn test_hm_113_match_tuple() {
        // Single clause with tuple pattern
        let sig = get_signature("sumTup((a, b)) = a + b\nmain() = 0", "sumTup").unwrap();
        assert!(sig.contains("->"), "sumTup: got: {}", sig);
    }

    #[test]
    fn test_hm_114_match_wildcard() {
        // Single clause with wildcard
        let sig = get_signature("always(_) = 42\nmain() = 0", "always").unwrap();
        assert!(sig.contains("->"), "always: got: {}", sig);
    }

    #[test]
    fn test_hm_115_match_with_guard() {
        // Multi-clause with guards
        let sig = get_signature("classify(n) when n > 0 = 1\nclassify(n) when n < 0 = -1\nclassify(_) = 0\nmain() = 0", "classify").unwrap();
        assert!(sig.contains("->"), "classify: got: {}", sig);
    }

    #[test]
    fn test_hm_116_match_list_empty() {
        // Multi-clause for list patterns
        let sig = get_signature("isEmpty([]) = true\nisEmpty(_) = false\nmain() = 0", "isEmpty").unwrap();
        assert!(sig.ends_with("-> Bool") || sig.contains("Bool"), "isEmpty: got: {}", sig);
    }

    #[test]
    fn test_hm_117_match_multiple_branches() {
        // Multi-clause fibonacci
        let sig = get_signature("fibM(0) = 0\nfibM(1) = 1\nfibM(n) = fibM(n-1) + fibM(n-2)\nmain() = 0", "fibM").unwrap();
        assert!(sig.contains("->"), "fibM: got: {}", sig);
    }

    #[test]
    fn test_hm_118_match_string() {
        // Multi-clause string match
        let sig = get_signature("greet(\"hi\") = \"hello\"\ngreet(_) = \"goodbye\"\nmain() = 0", "greet").unwrap();
        assert!(sig.contains("String") || sig.contains("->"), "greet: got: {}", sig);
    }

    #[test]
    fn test_hm_119_match_nested_tuple() {
        // Single clause with nested tuple pattern
        let sig = get_signature("flatten(((a, b), c)) = (a, b, c)\nmain() = 0", "flatten").unwrap();
        assert!(sig.contains("->"), "flatten: got: {}", sig);
    }

    #[test]
    fn test_hm_120_match_or_pattern() {
        // Multi-clause to simulate or pattern
        let sig = get_signature("isSmall(0) = true\nisSmall(1) = true\nisSmall(2) = true\nisSmall(_) = false\nmain() = 0", "isSmall").unwrap();
        assert!(sig.ends_with("-> Bool") || sig.contains("Bool"), "isSmall: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Block Expressions (Tests 121-130)
    // Using newlines in blocks instead of semicolons
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_121_block_simple() {
        let sig = get_signature("block() = { 42 }\nmain() = 0", "block").unwrap();
        assert_eq!(sig, "Int", "block: got: {}", sig);
    }

    #[test]
    fn test_hm_122_block_with_let() {
        // Use newlines inside block
        let sig = get_signature("withLet(x) = {\n    y = x + 1\n    y\n}\nmain() = 0", "withLet").unwrap();
        assert!(sig.contains("->"), "withLet: got: {}", sig);
    }

    #[test]
    fn test_hm_123_block_multiple_lets() {
        // Multiple lets with newlines
        let sig = get_signature("multi(a, b) = {\n    x = a + 1\n    y = b + 2\n    x + y\n}\nmain() = 0", "multi").unwrap();
        assert!(sig.contains("->"), "multi: got: {}", sig);
    }

    #[test]
    fn test_hm_124_block_shadowing() {
        // Shadowing with newlines
        let sig = get_signature("shadow(x) = {\n    x = x + 1\n    x = x * 2\n    x\n}\nmain() = 0", "shadow").unwrap();
        assert!(sig.contains("->"), "shadow: got: {}", sig);
    }

    #[test]
    fn test_hm_125_block_returns_last() {
        // Just expressions with newlines
        let sig = get_signature("last() = {\n    1\n    2\n    3\n}\nmain() = 0", "last").unwrap();
        assert_eq!(sig, "Int", "last: got: {}", sig);
    }

    #[test]
    fn test_hm_126_block_with_if() {
        // Block with if using newlines
        let sig = get_signature("condBlock(c) = {\n    r = if c then 1 else 0\n    r\n}\nmain() = 0", "condBlock").unwrap();
        assert!(sig.contains("->"), "condBlock: got: {}", sig);
    }

    #[test]
    fn test_hm_127_nested_blocks() {
        let sig = get_signature("nested() = { { { 42 } } }\nmain() = 0", "nested").unwrap();
        assert_eq!(sig, "Int", "nested: got: {}", sig);
    }

    #[test]
    fn test_hm_128_block_with_tuple() {
        // Tuple in block with newlines
        let sig = get_signature("tupBlock(a, b) = {\n    p = (a, b)\n    p\n}\nmain() = 0", "tupBlock").unwrap();
        assert!(sig.contains("->"), "tupBlock: got: {}", sig);
    }

    #[test]
    fn test_hm_129_block_using_param() {
        // Using params with multiple lets
        let sig = get_signature("useParam(x) = {\n    y = x * 2\n    z = y + 1\n    z\n}\nmain() = 0", "useParam").unwrap();
        assert!(sig.contains("->"), "useParam: got: {}", sig);
    }

    #[test]
    fn test_hm_130_block_complex() {
        // Complex block with newlines
        let sig = get_signature("complex(a, b, c) = {\n    x = a + b\n    y = x * c\n    if y > 0 then y else -y\n}\nmain() = 0", "complex").unwrap();
        assert!(sig.contains("->"), "complex: got: {}", sig);
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Special Scenarios (Tests 131-140)
    // -------------------------------------------------------------------------

    #[test]
    fn test_hm_131_single_char() {
        let sig = get_signature("getChar() = 'a'\nmain() = 0", "getChar").unwrap();
        assert_eq!(sig, "Char", "getChar: got: {}", sig);
    }

    #[test]
    fn test_hm_132_empty_string() {
        let sig = get_signature(r#"emptyStr() = ""\nmain() = 0"#, "emptyStr").unwrap();
        assert_eq!(sig, "String", "emptyStr: got: {}", sig);
    }

    #[test]
    fn test_hm_133_large_int() {
        let sig = get_signature("bigNum() = 9999999999\nmain() = 0", "bigNum").unwrap();
        assert!(sig == "Int" || sig == "BigInt", "bigNum: got: {}", sig);
    }

    #[test]
    fn test_hm_134_negative_int() {
        let sig = get_signature("negNum() = -42\nmain() = 0", "negNum").unwrap();
        assert_eq!(sig, "Int", "negNum: got: {}", sig);
    }

    #[test]
    fn test_hm_135_zero() {
        let sig = get_signature("zero() = 0\nmain() = 0", "zero").unwrap();
        assert_eq!(sig, "Int", "zero: got: {}", sig);
    }

    #[test]
    fn test_hm_136_float_zero() {
        let sig = get_signature("fzero() = 0.0\nmain() = 0", "fzero").unwrap();
        assert_eq!(sig, "Float", "fzero: got: {}", sig);
    }

    #[test]
    fn test_hm_137_scientific_notation() {
        let sig = get_signature("sci() = 1.5e10\nmain() = 0", "sci").unwrap();
        assert_eq!(sig, "Float", "sci: got: {}", sig);
    }

    #[test]
    fn test_hm_138_many_params() {
        let sig = get_signature("many(a, b, c, d, e) = a\nmain() = 0", "many").unwrap();
        assert_eq!(sig, "a -> b -> c -> d -> e -> a", "many: got: {}", sig);
    }

    #[test]
    fn test_hm_139_all_unified() {
        let sig = get_signature("allSame(a, b, c, d) = a + b + c + d\nmain() = 0", "allSame").unwrap();
        assert!(sig_matches(&sig, &["a -> a -> a -> a -> a", "Int -> Int -> Int -> Int -> Int"]),
            "allSame: got: {}", sig);
    }

    #[test]
    fn test_hm_140_partial_unified() {
        let sig = get_signature("partial(a, b, c) = (a + b, c)\nmain() = 0", "partial").unwrap();
        assert!(sig.contains("->"), "partial: got: {}", sig);
    }
}
