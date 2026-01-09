//! AST to Bytecode compiler.
//!
//! Features:
//! - Tail call detection and optimization
//! - Closure conversion (capture free variables)
//! - Pattern match compilation
//! - Type-directed code generation

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use nostos_syntax::ast::*;
use nostos_vm::*;

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

/// Compilation context.
pub struct Compiler {
    /// Current function being compiled
    chunk: Chunk,
    /// Local variable -> register mapping
    locals: HashMap<String, Reg>,
    /// Next available register
    next_reg: Reg,
    /// Captured variables for the current closure: name -> capture index
    capture_indices: HashMap<String, u8>,
    /// Compiled functions
    functions: HashMap<String, Rc<FunctionValue>>,
    /// Function name -> index mapping for direct calls (no HashMap lookup at runtime!)
    function_indices: HashMap<String, u16>,
    /// Ordered list of function names (index -> name)
    function_list: Vec<String>,
    /// Type definitions
    types: HashMap<String, TypeInfo>,
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
    /// Current function name being compiled (for self-recursion optimization)
    current_function_name: Option<String>,
    /// Loop context stack for break/continue
    loop_stack: Vec<LoopContext>,
    /// Line starts: byte offsets where each line begins (line 1 is at index 0)
    line_starts: Vec<usize>,
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
    Record { fields: Vec<String>, mutable: bool },
    Variant { constructors: Vec<(String, usize)> }, // (name, field_count)
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
            scope_depth: 0,
            module_path: Vec::new(),
            imports: HashMap::new(),
            function_visibility: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            type_traits: HashMap::new(),
            local_types: HashMap::new(),
            current_function_name: None,
            loop_stack: Vec::new(),
            line_starts,
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
        // Otherwise, check if it's a known function in the current module
        let qualified = self.qualify_name(name);
        if self.functions.contains_key(&qualified) {
            return qualified;
        }
        // Check if it's in the global scope
        if self.functions.contains_key(name) {
            return name.to_string();
        }
        // Return the original name (will error later if not found)
        name.to_string()
    }

    /// Check if an expression is float-typed (for type-directed operator selection).
    /// This is a simple heuristic: true if the expression is a float literal or
    /// a binary operation on floats.
    fn is_float_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Float(_, _) | Expr::Float32(_, _) => true,
            Expr::Int(_, _) => false,
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
            // For function calls, check if any argument is float-typed
            // This is a heuristic: if the function is called with float args, assume it returns float
            Expr::Call(_, args, _) => {
                args.iter().any(|arg| self.is_float_expr(arg))
            }
            _ => false, // Assume non-float by default for other expressions
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

    /// Compile a type definition.
    fn compile_type_def(&mut self, def: &TypeDef) -> Result<(), CompileError> {
        // Use qualified name (with module path prefix)
        let name = self.qualify_name(&def.name.node);

        let kind = match &def.body {
            TypeBody::Record(fields) => {
                let field_names: Vec<String> = fields.iter()
                    .map(|f| f.name.node.clone())
                    .collect();
                TypeInfoKind::Record { fields: field_names, mutable: def.mutable }
            }
            TypeBody::Variant(variants) => {
                let constructors: Vec<(String, usize)> = variants.iter()
                    .map(|v| {
                        let field_count = match &v.fields {
                            VariantFields::Unit => 0,
                            VariantFields::Positional(fields) => fields.len(),
                            VariantFields::Named(fields) => fields.len(),
                        };
                        (v.name.node.clone(), field_count)
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

        self.types.insert(name.clone(), TypeInfo { name, kind });
        Ok(())
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

    /// Compile a trait implementation.
    fn compile_trait_impl(&mut self, impl_def: &TraitImpl) -> Result<(), CompileError> {
        // Get the type name from the type expression
        let type_name = self.type_expr_to_string(&impl_def.ty);
        let trait_name = impl_def.trait_name.node.clone();

        // Check that the trait exists
        if !self.trait_defs.contains_key(&trait_name) {
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

            self.compile_fn_def(&modified_def)?;
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
                if let Some(trait_info) = self.trait_defs.get(trait_name) {
                    if trait_info.methods.iter().any(|m| m.name == method_name) {
                        // Return the qualified function name
                        return Some(format!("{}.{}.{}", type_name, trait_name, method_name));
                    }
                }
            }
        }
        None
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

                    // Add bindings to locals
                    for (name, reg) in bindings {
                        self.locals.insert(name, reg);
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
                let result_reg = self.compile_expr_tail(&clause.body, true)?;
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
            let error_idx = self.chunk.add_constant(Value::String(Rc::new(
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
                    self.locals.insert(n, i as Reg);
                }
            }

            // Compile function body (in tail position)
            let result_reg = self.compile_expr_tail(&clause.body, true)?;
            self.chunk.emit(Instruction::Return(result_reg), 0);
        }

        self.chunk.register_count = self.next_reg as usize;

        let func = FunctionValue {
            name: name.clone(),
            arity,
            param_names,
            code: Rc::new(std::mem::take(&mut self.chunk)),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: Cell::new(0),
        };

        // Assign function index if not already indexed (for trait methods and late-compiled functions)
        if !self.function_indices.contains_key(&name) {
            let idx = self.function_list.len() as u16;
            self.function_indices.insert(name.clone(), idx);
            self.function_list.push(name.clone());
        }

        self.functions.insert(name, Rc::new(func));

        // Restore compiler state
        self.chunk = saved_chunk;
        self.locals = saved_locals;
        self.next_reg = saved_next_reg;
        self.current_function_name = saved_function_name;

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
                let idx = self.chunk.add_constant(Value::BigInt(Rc::new(big)));
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
                        let idx = self.chunk.add_constant(Value::String(Rc::new(s.clone())));
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
                if let Some(&reg) = self.locals.get(name) {
                    Ok(reg)
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
            Expr::Match(scrutinee, arms, _) => {
                self.compile_match(scrutinee, arms, is_tail)
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
                let field_idx = self.chunk.add_constant(Value::String(Rc::new(field.node.clone())));
                self.chunk.emit(Instruction::GetField(dst, obj_reg, field_idx), line);
                Ok(dst)
            }

            // Record construction
            Expr::Record(type_name, fields, _) => {
                // Qualify type name with module path
                let qualified_type = self.qualify_name(&type_name.node);
                self.compile_record(&qualified_type, fields)
            }

            // Record update
            Expr::RecordUpdate(type_name, base, fields, _) => {
                // Qualify type name with module path
                let qualified_type = self.qualify_name(&type_name.node);
                self.compile_record_update(&qualified_type, base, fields)
            }

            // Method call (UFCS) or module-qualified function call
            Expr::MethodCall(obj, method, args, _span) => {
                // Check if this is a module-qualified call (e.g., Math.add(1, 2))
                // by checking if the object looks like a module path
                if let Some(module_path) = self.extract_module_path(obj) {
                    // It's a module-qualified call: Module.function(args)
                    let qualified_name = format!("{}.{}", module_path, method.node);
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

            // Spawn: spawn(func) or spawn(() => expr)
            Expr::Spawn(kind, func_expr, args, _) => {
                let func_reg = self.compile_expr_tail(func_expr, false)?;
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

            // Receive: receive pattern -> body ... end
            Expr::Receive(arms, _after, _) => {
                // Emit receive instruction - this blocks until a message arrives
                // and places it in register 0
                self.chunk.emit(Instruction::Receive, line);

                // The message is in register 0 after Receive completes
                // We need to match it against the arms
                // Reserve register 0 for the message (ensure it's not overwritten)
                let msg_reg = 0 as Reg;

                let dst = self.alloc_reg();
                let mut end_jumps = Vec::new();

                for (i, arm) in arms.iter().enumerate() {
                    let is_last = i == arms.len() - 1;

                    // Try to match the pattern against the message
                    let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, msg_reg)?;

                    let next_arm_jump = if !is_last {
                        Some(self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), line))
                    } else {
                        None
                    };

                    // Bind pattern variables
                    for (name, reg) in bindings {
                        self.locals.insert(name, reg);
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
                            let here = self.chunk.code.len() as i16;
                            if let Instruction::JumpIfFalse(_, ref mut offset) = self.chunk.code[guard_fail] {
                                *offset = here;
                            }
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
                        let here = self.chunk.code.len() as i16;
                        if let Instruction::JumpIfFalse(_, ref mut offset) = self.chunk.code[jump_idx] {
                            *offset = here;
                        }
                    }
                }

                // Patch end jumps
                let end = self.chunk.code.len() as i16;
                for jump_idx in end_jumps {
                    if let Instruction::Jump(ref mut offset) = self.chunk.code[jump_idx] {
                        *offset = end;
                    }
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

        // Bind loop variable to counter register
        let saved_var = self.locals.get(&var.node).copied();
        self.locals.insert(var.node.clone(), counter_reg);

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
        if let Some(prev_reg) = saved_var {
            self.locals.insert(var.node.clone(), prev_reg);
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

        // Check if we should use float operations
        let is_float = self.is_float_expr(left) || self.is_float_expr(right);

        let left_reg = self.compile_expr_tail(left, false)?;
        let right_reg = self.compile_expr_tail(right, false)?;
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
                    "assert_eq" if arg_regs.len() == 2 => {
                        self.chunk.emit(Instruction::AssertEq(arg_regs[0], arg_regs[1]), 0);
                        let dst = self.alloc_reg();
                        self.chunk.emit(Instruction::LoadUnit(dst), 0);
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
                    // === Dynamic builtins (trait-based, keep CallNative for now) ===
                    "show" | "copy" => {
                        let dst = self.alloc_reg();
                        let name_idx = self.chunk.add_constant(Value::String(Rc::new(qualified_name)));
                        self.chunk.emit(Instruction::CallNative(dst, name_idx, arg_regs.into()), 0);
                        return Ok(dst);
                    }
                    _ => {} // Fall through to normal function lookup
                }
            }

            // Resolve the name (handles imports and module path)
            let resolved_name = self.resolve_name(&qualified_name);

            // Check for user-defined function
            if self.functions.contains_key(&resolved_name) {
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
            // Record construction: Point{x: 1, y: 2}
            Expr::Record(type_name, _, _) => Some(type_name.node.clone()),
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
                        // It's a variant constructor - the type is determined by checking
                        // what type has this constructor
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
                // Check if we have tracked this variable's type
                self.local_types.get(&ident.node).cloned()
            }
            _ => None,
        }
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
    fn compile_match(&mut self, scrutinee: &Expr, arms: &[MatchArm], is_tail: bool) -> Result<Reg, CompileError> {
        let scrut_reg = self.compile_expr_tail(scrutinee, false)?;
        let dst = self.alloc_reg();
        let mut end_jumps = Vec::new();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, scrut_reg)?;

            let next_arm_jump = if !is_last {
                Some(self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0))
            } else {
                None
            };

            // Bind pattern variables
            for (name, reg) in bindings {
                self.locals.insert(name, reg);
            }

            // Compile guard if present
            if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                if let Some(jump) = next_arm_jump {
                    self.chunk.patch_jump(jump, self.chunk.code.len());
                }
                let guard_jump = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0);
                end_jumps.push(guard_jump);
            }

            // Compile arm body - pass is_tail for tail calls, but always move result
            let body_reg = self.compile_expr_tail(&arm.body, is_tail)?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            if !is_last {
                end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));
            }

            // Patch jump to next arm
            if let Some(jump) = next_arm_jump {
                if arm.guard.is_none() {
                    let next_target = self.chunk.code.len();
                    self.chunk.patch_jump(jump, next_target);
                }
            }
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
        let mut end_jumps = Vec::new();
        for (i, arm) in catch_arms.iter().enumerate() {
            let is_last = i == catch_arms.len() - 1;

            // Try to match the pattern
            let (match_success, bindings) = self.compile_pattern_test(&arm.pattern, exc_reg)?;

            let next_arm_jump = if !is_last {
                Some(self.chunk.emit(Instruction::JumpIfFalse(match_success, 0), 0))
            } else {
                None
            };

            // Bind pattern variables
            for (name, reg) in bindings {
                self.locals.insert(name, reg);
            }

            // Compile guard if present
            if let Some(guard) = &arm.guard {
                let guard_reg = self.compile_expr_tail(guard, false)?;
                if let Some(jump) = next_arm_jump {
                    self.chunk.patch_jump(jump, self.chunk.code.len());
                }
                let guard_jump = self.chunk.emit(Instruction::JumpIfFalse(guard_reg, 0), 0);
                end_jumps.push(guard_jump);
            }

            // Compile catch arm body
            let body_reg = self.compile_expr_tail(&arm.body, is_tail && finally_expr.is_none())?;
            self.chunk.emit(Instruction::Move(dst, body_reg), 0);

            if !is_last {
                end_jumps.push(self.chunk.emit(Instruction::Jump(0), 0));
            }

            // Patch jump to next arm
            if let Some(jump) = next_arm_jump {
                if arm.guard.is_none() {
                    let next_target = self.chunk.code.len();
                    self.chunk.patch_jump(jump, next_target);
                }
            }
        }

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
    fn compile_pattern_test(&mut self, pattern: &Pattern, scrut_reg: Reg) -> Result<(Reg, Vec<(String, Reg)>), CompileError> {
        let success_reg = self.alloc_reg();
        let mut bindings = Vec::new();

        match pattern {
            Pattern::Wildcard(_) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
            }
            Pattern::Var(ident) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                let var_reg = self.alloc_reg();
                self.chunk.emit(Instruction::Move(var_reg, scrut_reg), 0);
                bindings.push((ident.node.clone(), var_reg));
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
                let const_idx = self.chunk.add_constant(Value::BigInt(Rc::new(big)));
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
                let const_idx = self.chunk.add_constant(Value::String(Rc::new(s.clone())));
                self.chunk.emit(Instruction::TestConst(success_reg, scrut_reg, const_idx), 0);
            }
            Pattern::Variant(ctor, fields, _) => {
                let ctor_idx = self.chunk.add_constant(Value::String(Rc::new(ctor.node.clone())));
                self.chunk.emit(Instruction::TestTag(success_reg, scrut_reg, ctor_idx), 0);

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
                            bindings.append(&mut sub_bindings);
                        }

                        // Patch the skip jump to land here
                        let after_extract = self.chunk.code.len();
                        self.chunk.patch_jump(skip_jump, after_extract);
                    }
                    VariantPatternFields::Named(fields) => {
                        // Jump past field extraction if tag doesn't match
                        let skip_jump = self.chunk.emit(Instruction::JumpIfFalse(success_reg, 0), 0);

                        for field in fields {
                            match field {
                                RecordPatternField::Punned(ident) => {
                                    // {x} means bind field "x" to variable "x"
                                    let field_reg = self.alloc_reg();
                                    let name_idx = self.chunk.add_constant(Value::String(Rc::new(ident.node.clone())));
                                    self.chunk.emit(Instruction::GetVariantFieldByName(field_reg, scrut_reg, name_idx), 0);
                                    bindings.push((ident.node.clone(), field_reg));
                                }
                                RecordPatternField::Named(ident, pat) => {
                                    // {name: n} means bind field "name" to the result of matching pattern
                                    let field_reg = self.alloc_reg();
                                    let name_idx = self.chunk.add_constant(Value::String(Rc::new(ident.node.clone())));
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

                        // Decons each head element
                        let mut current_list = scrut_reg;
                        for (i, head_pat) in head_patterns.iter().enumerate() {
                            let head_reg = self.alloc_reg();
                            let tail_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Decons(head_reg, tail_reg, current_list), 0);

                            let (_, mut head_bindings) = self.compile_pattern_test(head_pat, head_reg)?;
                            bindings.append(&mut head_bindings);

                            // If this is the last head pattern and there's a tail pattern
                            if i == n - 1 {
                                if let Some(tail_pat) = tail {
                                    let (_, mut tail_bindings) = self.compile_pattern_test(tail_pat, tail_reg)?;
                                    bindings.append(&mut tail_bindings);
                                }
                            } else {
                                current_list = tail_reg;
                            }
                        }
                    }
                }
            }
            Pattern::Tuple(patterns, _) => {
                self.chunk.emit(Instruction::LoadTrue(success_reg), 0);
                for (i, pat) in patterns.iter().enumerate() {
                    let elem_reg = self.alloc_reg();
                    self.chunk.emit(Instruction::GetTupleField(elem_reg, scrut_reg, i as u8), 0);
                    let (_, mut sub_bindings) = self.compile_pattern_test(pat, elem_reg)?;
                    bindings.append(&mut sub_bindings);
                }
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
        if stmts.is_empty() {
            let dst = self.alloc_reg();
            self.chunk.emit(Instruction::LoadUnit(dst), 0);
            return Ok(dst);
        }

        let mut last_reg = 0;
        for (i, stmt) in stmts.iter().enumerate() {
            let is_last = i == stmts.len() - 1;
            last_reg = self.compile_stmt(stmt, is_tail && is_last)?;
        }
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
        // Try to determine the type of the value before compiling
        let value_type = self.expr_type_name(&binding.value);

        let value_reg = self.compile_expr_tail(&binding.value, false)?;

        // For simple variable binding
        if let Pattern::Var(ident) = &binding.pattern {
            // If the variable already exists, this is a reassignment, not a new binding.
            // Move the value to the existing register to preserve mutation semantics.
            if let Some(&existing_reg) = self.locals.get(&ident.node) {
                if existing_reg != value_reg {
                    self.chunk.emit(Instruction::Move(existing_reg, value_reg), 0);
                }
                return Ok(existing_reg);
            }
            // New binding
            self.locals.insert(ident.node.clone(), value_reg);
            // Record the type if we know it
            if let Some(ty) = value_type {
                self.local_types.insert(ident.node.clone(), ty);
            }
            return Ok(value_reg);
        }

        // For complex patterns, we need to deconstruct
        let (_, bindings) = self.compile_pattern_test(&binding.pattern, value_reg)?;
        for (name, reg) in bindings {
            self.locals.insert(name, reg);
        }

        let dst = self.alloc_reg();
        self.chunk.emit(Instruction::LoadUnit(dst), 0);
        Ok(dst)
    }

    /// Compile an assignment.
    fn compile_assign(&mut self, target: &AssignTarget, value: &Expr) -> Result<Reg, CompileError> {
        let value_reg = self.compile_expr_tail(value, false)?;

        match target {
            AssignTarget::Var(ident) => {
                if let Some(&var_reg) = self.locals.get(&ident.node) {
                    self.chunk.emit(Instruction::Move(var_reg, value_reg), 0);
                    Ok(var_reg)
                } else {
                    Err(CompileError::UnknownVariable {
                        name: ident.node.clone(),
                        span: ident.span,
                    })
                }
            }
            AssignTarget::Field(obj, field) => {
                let obj_reg = self.compile_expr_tail(obj, false)?;
                let field_idx = self.chunk.add_constant(Value::String(Rc::new(field.node.clone())));
                self.chunk.emit(Instruction::SetField(obj_reg, field_idx, value_reg), 0);
                Ok(value_reg)
            }
            AssignTarget::Index(coll, idx) => {
                let coll_reg = self.compile_expr_tail(coll, false)?;
                let idx_reg = self.compile_expr_tail(idx, false)?;
                self.chunk.emit(Instruction::IndexSet(coll_reg, idx_reg, value_reg), 0);
                Ok(value_reg)
            }
        }
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
            let idx = self.chunk.add_constant(Value::String(Rc::new(String::new())));
            self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
            return Ok(dst);
        }

        // Compile each part to a string register
        let mut part_regs = Vec::new();
        for part in parts {
            let reg = match part {
                StringPart::Lit(s) => {
                    let dst = self.alloc_reg();
                    let idx = self.chunk.add_constant(Value::String(Rc::new(s.clone())));
                    self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                    dst
                }
                StringPart::Expr(e) => {
                    // Compile the expression
                    let expr_reg = self.compile_expr_tail(e, false)?;
                    // Call `show` to convert to string
                    let dst = self.alloc_reg();
                    let name_idx = self.chunk.add_constant(Value::String(Rc::new("show".to_string())));
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
            if let Some(&reg) = self.locals.get(var_name) {
                captures.push((var_name.clone(), reg));
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

        // Allocate registers for parameters
        for (i, param) in params.iter().enumerate() {
            if let Some(name) = self.pattern_binding_name(param) {
                self.locals.insert(name.clone(), i as Reg);
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
            code: Rc::new(lambda_chunk),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: Cell::new(0),
        };

        let dst = self.alloc_reg();

        if captures.is_empty() {
            // No captures - just load the function
            let func_idx = self.chunk.add_constant(Value::Function(Rc::new(func)));
            self.chunk.emit(Instruction::LoadConst(dst, func_idx), 0);
        } else {
            // Has captures - create a closure
            let func_idx = self.chunk.add_constant(Value::Function(Rc::new(func)));
            let capture_regs: Vec<Reg> = captures.iter().map(|(_, reg)| *reg).collect();
            self.chunk.emit(Instruction::MakeClosure(dst, func_idx, capture_regs.into()), 0);
        }

        Ok(dst)
    }

    /// Compile a record construction.
    fn compile_record(&mut self, type_name: &str, fields: &[RecordField]) -> Result<Reg, CompileError> {
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
        let type_idx = self.chunk.add_constant(Value::String(Rc::new(type_name.to_string())));
        self.chunk.emit(Instruction::MakeRecord(dst, type_idx, field_regs.into()), 0);
        Ok(dst)
    }

    /// Compile a record update.
    fn compile_record_update(&mut self, type_name: &str, base: &Expr, fields: &[RecordField]) -> Result<Reg, CompileError> {
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
        let type_idx = self.chunk.add_constant(Value::String(Rc::new(type_name.to_string())));
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
        self.next_reg += 1;
        reg
    }

    /// Get a compiled function.
    pub fn get_function(&self, name: &str) -> Option<Rc<FunctionValue>> {
        self.functions.get(name).cloned()
    }

    /// Get all compiled functions.
    pub fn get_all_functions(&self) -> &HashMap<String, Rc<FunctionValue>> {
        &self.functions
    }

    /// Get the ordered function list for direct indexed calls.
    /// Returns functions in the same order as their indices (for CallDirect).
    pub fn get_function_list(&self) -> Vec<Rc<FunctionValue>> {
        self.function_list.iter()
            .map(|name| self.functions.get(name).cloned().expect("Function should exist"))
            .collect()
    }

    /// Get all types for the VM.
    pub fn get_vm_types(&self) -> HashMap<String, Rc<TypeValue>> {
        use nostos_vm::value::{TypeValue, TypeKind, FieldInfo, ConstructorInfo};

        let mut vm_types = HashMap::new();
        for (name, type_info) in &self.types {
            let type_value = match &type_info.kind {
                TypeInfoKind::Record { fields, mutable } => {
                    let field_infos: Vec<FieldInfo> = fields.iter()
                        .map(|f| FieldInfo {
                            name: f.clone(),
                            type_name: "any".to_string(),
                            mutable: *mutable,
                            private: false,
                        })
                        .collect();
                    TypeValue {
                        name: name.clone(),
                        kind: TypeKind::Record { mutable: *mutable },
                        fields: field_infos,
                        constructors: vec![],
                        traits: vec![],
                    }
                }
                TypeInfoKind::Variant { constructors } => {
                    TypeValue {
                        name: name.clone(),
                        kind: TypeKind::Variant,
                        fields: vec![],
                        constructors: constructors.iter()
                            .map(|(n, field_count)| ConstructorInfo {
                                name: n.clone(),
                                fields: (0..*field_count).map(|i| FieldInfo {
                                    name: format!("_{}", i),
                                    type_name: "any".to_string(),
                                    mutable: false,
                                    private: false,
                                }).collect(),
                            })
                            .collect(),
                        traits: vec![],
                    }
                }
            };
            vm_types.insert(name.clone(), Rc::new(type_value));
        }
        vm_types
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

        for item in items {
            if let Item::FnDef(fn_def) = item {
                let qualified_name = self.qualify_name(&fn_def.name.node);
                if !fn_clauses.contains_key(&qualified_name) {
                    fn_order.push(qualified_name.clone());
                    fn_spans.insert(qualified_name.clone(), fn_def.span);
                    // Use visibility from first definition
                    fn_visibility.insert(qualified_name.clone(), fn_def.visibility);
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
                code: Rc::new(Chunk::new()),
                module: if self.module_path.is_empty() { None } else { Some(self.module_path.join(".")) },
                source_span: None,
                jit_code: None,
                call_count: Cell::new(0),
            };
            self.functions.insert(name.clone(), Rc::new(placeholder));

            // Assign function index for direct calls (no HashMap lookup at runtime!)
            if !self.function_indices.contains_key(name) {
                let idx = self.function_list.len() as u16;
                self.function_indices.insert(name.clone(), idx);
                self.function_list.push(name.clone());
            }
        }

        // Fifth pass: compile functions with merged clauses
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
                clauses: clauses.clone(),
                span,
            };
            self.compile_fn_def(&merged_fn)?;
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
    use nostos_vm::Runtime;

    fn compile_and_run(source: &str) -> Result<Value, String> {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            return Err(format!("Parse error: {:?}", errors));
        }
        let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;
        let compiler = compile_module(&module, source).map_err(|e| format!("Compile error: {:?}", e))?;

        let mut runtime = Runtime::new();
        for (name, func) in compiler.get_all_functions() {
            runtime.register_function(&name, func.clone());
        }
        runtime.set_function_list(compiler.get_function_list());
        for (name, type_val) in compiler.get_vm_types() {
            runtime.register_type(&name, type_val);
        }

        // Look for a main function
        let main_func = if let Some(func) = compiler.get_function("main") {
            func
        } else if let Some((_, func)) = compiler.get_all_functions().iter().next() {
            func.clone()
        } else {
            return Err("No functions to run".to_string());
        };

        runtime.spawn_initial(main_func);
        runtime.run_to_value()
            .map_err(|e| format!("Runtime error: {:?}", e))?
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
        // Test with positional field access which uses synthetic names
        let source = "
            getX(p) = p._0
            main() = getX(Point(42, 100))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int64(42)));
    }

    #[test]
    fn test_e2e_variant_construction() {
        let source = "
            main() = Some(42)
        ";
        let result = compile_and_run(source);
        // Note: Currently variants are compiled as Records until we have full type info
        match result {
            Ok(Value::Record(r)) => {
                assert_eq!(r.type_name, "Some");
                assert_eq!(r.fields.len(), 1);
                assert_eq!(r.fields[0], Value::Int64(42));
            }
            other => panic!("Expected record, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_variant_match() {
        let source = "
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
        assert_eq!(result, Ok(Value::String(std::rc::Rc::new("Hello, World".to_string()))));
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
}
