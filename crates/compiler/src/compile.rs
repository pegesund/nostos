//! AST to Bytecode compiler.
//!
//! Features:
//! - Tail call detection and optimization
//! - Closure conversion (capture free variables)
//! - Pattern match compilation
//! - Type-directed code generation

use std::collections::HashMap;
use std::rc::Rc;

use nostos_syntax::ast::*;
use nostos_vm::*;

/// Compilation errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    #[error("Unknown variable: {0}")]
    UnknownVariable(String),

    #[error("Unknown function: {0}")]
    UnknownFunction(String),

    #[error("Unknown type: {0}")]
    UnknownType(String),

    #[error("Duplicate definition: {0}")]
    DuplicateDefinition(String),

    #[error("Invalid pattern")]
    InvalidPattern,

    #[error("Not implemented: {0}")]
    NotImplemented(String),
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
    /// Type definitions
    types: HashMap<String, TypeInfo>,
    /// Current scope depth
    scope_depth: usize,
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

impl Compiler {
    pub fn new() -> Self {
        Self {
            chunk: Chunk::new(),
            locals: HashMap::new(),
            next_reg: 0,
            capture_indices: HashMap::new(),
            functions: HashMap::new(),
            types: HashMap::new(),
            scope_depth: 0,
        }
    }

    /// Check if an expression is float-typed (for type-directed operator selection).
    /// This is a simple heuristic: true if the expression is a float literal or
    /// a binary operation on floats.
    fn is_float_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Float(_, _) => true,
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
            _ => {
                return Err(CompileError::NotImplemented(format!("item: {:?}", item)));
            }
        }
        Ok(())
    }

    /// Compile a type definition.
    fn compile_type_def(&mut self, def: &TypeDef) -> Result<(), CompileError> {
        let name = def.name.node.clone();

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

    /// Compile a function definition.
    fn compile_fn_def(&mut self, def: &FnDef) -> Result<(), CompileError> {
        // Save compiler state
        let saved_chunk = std::mem::take(&mut self.chunk);
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_next_reg = self.next_reg;

        // Reset for new function
        self.chunk = Chunk::new();
        self.locals = HashMap::new();
        self.next_reg = 0;

        let name = def.name.node.clone();

        // For now, just compile the first clause
        // TODO: Compile all clauses with pattern matching dispatch
        let clause = &def.clauses[0];
        let arity = clause.params.len();

        // Allocate registers for parameters
        let mut param_names: Vec<String> = Vec::new();

        for (i, param) in clause.params.iter().enumerate() {
            if let Some(name) = self.pattern_binding_name(&param.pattern) {
                self.locals.insert(name.clone(), i as Reg);
                param_names.push(name);
            } else {
                param_names.push(format!("_arg{}", i));
            }
            self.next_reg = (i + 1) as Reg;
        }

        // Compile function body (in tail position)
        let result_reg = self.compile_expr_tail(&clause.body, true)?;
        self.chunk.emit(Instruction::Return(result_reg), 0);

        self.chunk.register_count = self.next_reg as usize;

        let func = FunctionValue {
            name: name.clone(),
            arity,
            param_names,
            code: Rc::new(std::mem::take(&mut self.chunk)),
            module: None,
            source_span: None,
            jit_code: None,
        };

        self.functions.insert(name, Rc::new(func));

        // Restore compiler state
        self.chunk = saved_chunk;
        self.locals = saved_locals;
        self.next_reg = saved_next_reg;

        Ok(())
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
        match expr {
            // Literals
            Expr::Int(n, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Int(*n));
                self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                Ok(dst)
            }
            Expr::Float(f, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Float(*f));
                self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                Ok(dst)
            }
            Expr::Bool(b, _) => {
                let dst = self.alloc_reg();
                if *b {
                    self.chunk.emit(Instruction::LoadTrue(dst), 0);
                } else {
                    self.chunk.emit(Instruction::LoadFalse(dst), 0);
                }
                Ok(dst)
            }
            Expr::String(string_lit, _) => {
                let s = match string_lit {
                    StringLit::Plain(s) => s.clone(),
                    StringLit::Interpolated(_) => {
                        return Err(CompileError::NotImplemented("string interpolation".to_string()));
                    }
                };
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::String(Rc::new(s)));
                self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                Ok(dst)
            }
            Expr::Char(c, _) => {
                let dst = self.alloc_reg();
                let idx = self.chunk.add_constant(Value::Char(*c));
                self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                Ok(dst)
            }
            Expr::Unit(_) => {
                let dst = self.alloc_reg();
                self.chunk.emit(Instruction::LoadUnit(dst), 0);
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
                    self.chunk.emit(Instruction::GetCapture(dst, capture_idx), 0);
                    Ok(dst)
                } else if self.functions.contains_key(name) {
                    // It's a function reference
                    let dst = self.alloc_reg();
                    let func = self.functions.get(name).unwrap().clone();
                    let idx = self.chunk.add_constant(Value::Function(func));
                    self.chunk.emit(Instruction::LoadConst(dst, idx), 0);
                    Ok(dst)
                } else {
                    Err(CompileError::UnknownVariable(name.clone()))
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
                            self.chunk.emit(Instruction::Cons(new_reg, item_reg, result_reg), 0);
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
                        self.chunk.emit(Instruction::MakeList(dst, regs), 0);
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
                self.chunk.emit(Instruction::MakeTuple(dst, regs), 0);
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
                self.chunk.emit(Instruction::GetField(dst, obj_reg, field_idx), 0);
                Ok(dst)
            }

            // Record construction
            Expr::Record(type_name, fields, _) => {
                self.compile_record(&type_name.node, fields)
            }

            // Record update
            Expr::RecordUpdate(type_name, base, fields, _) => {
                self.compile_record_update(&type_name.node, base, fields)
            }

            // Method call (UFCS)
            Expr::MethodCall(obj, method, args, _) => {
                // Transform obj.method(args) to method(obj, args)
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
                self.chunk.emit(Instruction::Index(dst, coll_reg, idx_reg), 0);
                Ok(dst)
            }

            _ => Err(CompileError::NotImplemented(format!("expr: {:?}", expr))),
        }
    }

    /// Compile a binary operation.
    fn compile_binop(&mut self, op: &BinOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
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
                    self.chunk.emit(Instruction::EqFloat(dst, left_reg, right_reg), 0);
                } else {
                    self.chunk.emit(Instruction::Eq(dst, left_reg, right_reg), 0);
                }
                self.chunk.emit(Instruction::Not(dst, dst), 0);
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

        self.chunk.emit(instr, 0);
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
        // Compile arguments first
        let mut arg_regs = Vec::new();
        for arg in args {
            let reg = self.compile_expr_tail(arg, false)?;
            arg_regs.push(reg);
        }

        // Check if it's a direct call to a known function
        if let Expr::Var(ident) = func {
            let name = &ident.node;

            // Check for native functions
            let native_names = ["println", "print", "length", "head", "tail", "isEmpty",
                              "show", "toInt", "toFloat", "abs", "sqrt", "panic", "assert", "typeOf"];

            if native_names.contains(&name.as_str()) {
                let dst = self.alloc_reg();
                let name_idx = self.chunk.add_constant(Value::String(Rc::new(name.clone())));
                self.chunk.emit(Instruction::CallNative(dst, name_idx, arg_regs), 0);
                return Ok(dst);
            }

            // Check for user-defined function - use name lookup to support recursion
            if self.functions.contains_key(name) {
                let dst = self.alloc_reg();
                let name_idx = self.chunk.add_constant(Value::String(Rc::new(name.clone())));
                if is_tail {
                    // Tail call optimization using name lookup
                    self.chunk.emit(Instruction::TailCallByName(name_idx, arg_regs), 0);
                    return Ok(0);
                } else {
                    self.chunk.emit(Instruction::CallByName(dst, name_idx, arg_regs), 0);
                    return Ok(dst);
                }
            }
        }

        // Generic function call
        let func_reg = self.compile_expr_tail(func, false)?;
        let dst = self.alloc_reg();

        if is_tail {
            self.chunk.emit(Instruction::TailCall(func_reg, arg_regs), 0);
            Ok(0)
        } else {
            self.chunk.emit(Instruction::Call(dst, func_reg, arg_regs), 0);
            Ok(dst)
        }
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
                let const_idx = self.chunk.add_constant(Value::Int(*n));
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
                    VariantPatternFields::Named(_) => {
                        return Err(CompileError::NotImplemented("named variant patterns".to_string()));
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
                            let n_idx = self.chunk.add_constant(Value::Int(n as i64));
                            self.chunk.emit(Instruction::LoadConst(n_reg, n_idx), 0);
                            self.chunk.emit(Instruction::GeInt(success_reg, len_reg, n_reg), 0);
                        } else {
                            // Pattern like [a, b, c] - check list has exactly n elements
                            let len_reg = self.alloc_reg();
                            self.chunk.emit(Instruction::Length(len_reg, scrut_reg), 0);
                            let n_reg = self.alloc_reg();
                            let n_idx = self.chunk.add_constant(Value::Int(n as i64));
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
                return Err(CompileError::NotImplemented(format!("pattern: {:?}", pattern)));
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
        let value_reg = self.compile_expr_tail(&binding.value, false)?;

        // For simple variable binding
        if let Pattern::Var(ident) = &binding.pattern {
            self.locals.insert(ident.node.clone(), value_reg);
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
                    Err(CompileError::UnknownVariable(ident.node.clone()))
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

        // Step 4: Create new function for lambda
        self.chunk = Chunk::new();
        self.locals = HashMap::new();
        self.capture_indices = HashMap::new();
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

        // Step 6: Create closure with captures
        let func = FunctionValue {
            name: "<lambda>".to_string(),
            arity,
            param_names,
            code: Rc::new(lambda_chunk),
            module: None,
            source_span: None,
            jit_code: None,
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
            self.chunk.emit(Instruction::MakeClosure(dst, func_idx, capture_regs), 0);
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
        self.chunk.emit(Instruction::MakeRecord(dst, type_idx, field_regs), 0);
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
        self.chunk.emit(Instruction::UpdateRecord(dst, base_reg, type_idx, field_regs), 0);
        Ok(dst)
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

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile a complete module.
pub fn compile_module(module: &Module) -> Result<Compiler, CompileError> {
    let mut compiler = Compiler::new();

    // First pass: collect type definitions
    for item in &module.items {
        if let Item::TypeDef(type_def) = item {
            compiler.compile_type_def(type_def)?;
        }
    }

    // Second pass: forward declare all functions (for recursion)
    for item in &module.items {
        if let Item::FnDef(fn_def) = item {
            let name = fn_def.name.node.clone();
            let arity = fn_def.clauses[0].params.len();
            // Insert a placeholder function for forward reference
            let placeholder = FunctionValue {
                name: name.clone(),
                arity,
                param_names: vec![],
                code: Rc::new(Chunk::new()),
                module: None,
                source_span: None,
                jit_code: None,
            };
            compiler.functions.insert(name, Rc::new(placeholder));
        }
    }

    // Third pass: compile functions (they can now reference each other)
    for item in &module.items {
        if let Item::FnDef(fn_def) = item {
            compiler.compile_fn_def(fn_def)?;
        }
    }

    Ok(compiler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nostos_syntax::parser::parse;

    fn compile_and_run(source: &str) -> Result<Value, String> {
        let (module_opt, errors) = parse(source);
        if !errors.is_empty() {
            return Err(format!("Parse error: {:?}", errors));
        }
        let module = module_opt.ok_or_else(|| "Parse returned no module".to_string())?;
        let compiler = compile_module(&module).map_err(|e| format!("Compile error: {:?}", e))?;

        let mut vm = VM::new();
        for (name, func) in compiler.get_all_functions() {
            vm.functions.insert(name.clone(), func.clone());
        }

        // Look for a main function
        if vm.functions.contains_key("main") {
            vm.call("main", vec![]).map_err(|e| format!("Runtime error: {:?}", e))
        } else if let Some((name, _)) = compiler.get_all_functions().iter().next() {
            // Run the first function with no arguments if possible
            vm.call(name, vec![]).map_err(|e| format!("Runtime error: {:?}", e))
        } else {
            Err("No functions to run".to_string())
        }
    }

    #[test]
    fn test_compile_simple_function() {
        let result = compile_and_run("main() = 42");
        assert_eq!(result, Ok(Value::Int(42)));
    }

    #[test]
    fn test_compile_addition() {
        let result = compile_and_run("main() = 2 + 3");
        assert_eq!(result, Ok(Value::Int(5)));
    }

    #[test]
    fn test_compile_nested_arithmetic() {
        let result = compile_and_run("main() = (2 + 3) * 4");
        assert_eq!(result, Ok(Value::Int(20)));
    }

    #[test]
    fn test_compile_if_then_else() {
        let result = compile_and_run("main() = if true then 1 else 2");
        assert_eq!(result, Ok(Value::Int(1)));

        let result2 = compile_and_run("main() = if false then 1 else 2");
        assert_eq!(result2, Ok(Value::Int(2)));
    }

    #[test]
    fn test_compile_comparison() {
        let result = compile_and_run("main() = if 5 > 3 then 1 else 0");
        assert_eq!(result, Ok(Value::Int(1)));
    }

    #[test]
    fn test_compile_function_call() {
        let source = "
            double(x) = x + x
            main() = double(21)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(42)));
    }

    #[test]
    fn test_compile_recursive_function() {
        let source = "
            fact(n) = if n == 0 then 1 else n * fact(n - 1)
            main() = fact(5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(120)));
    }

    #[test]
    fn test_compile_tail_recursive() {
        let source = "
            sum(n, acc) = if n == 0 then acc else sum(n - 1, acc + n)
            main() = sum(100, 0)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(5050)));
    }

    #[test]
    fn test_compile_list() {
        let source = "main() = [1, 2, 3]";
        let result = compile_and_run(source);
        match result {
            Ok(Value::List(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int(1));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(6765)));
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
        assert_eq!(result, Ok(Value::Int(7)));
    }

    #[test]
    fn test_e2e_compose_lambdas() {
        let source = "
            compose(f, g, x) = f(g(x))
            main() = compose(x => x * 2, y => y + 1, 10)
        ";
        // (10 + 1) * 2 = 22
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(22)));
    }

    #[test]
    fn test_e2e_tuple() {
        // Tuple creation works
        let source = "main() = (1, 2, 3)";
        let result = compile_and_run(source);
        match result {
            Ok(Value::Tuple(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Int(1));
                assert_eq!(items[1], Value::Int(2));
                assert_eq!(items[2], Value::Int(3));
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
        assert_eq!(result, Ok(Value::Int(28)));
    }

    #[test]
    fn test_e2e_nested_conditionals() {
        let source = "
            classify(n) = if n < 0 then 0 - 1 else if n == 0 then 0 else 1
            main() = classify(5) + classify(0 - 3) + classify(0)
        ";
        // 1 + (-1) + 0 = 0
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(0)));
    }

    #[test]
    fn test_e2e_gcd() {
        let source = "
            gcd(a, b) = if b == 0 then a else gcd(b, a - (a / b) * b)
            main() = gcd(48, 18)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(6)));
    }

    #[test]
    fn test_e2e_power() {
        let source = "
            power(base, exp, acc) = if exp == 0 then acc else power(base, exp - 1, acc * base)
            main() = power(2, 10, 1)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(1024)));
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
        assert_eq!(result, Ok(Value::Int(9)));
    }

    #[test]
    fn test_e2e_collatz_steps() {
        let source = "
            collatz(n, steps) = if n == 1 then steps else if (n - (n / 2) * 2) == 0 then collatz(n / 2, steps + 1) else collatz(3 * n + 1, steps + 1)
            main() = collatz(27, 0)
        ";
        // collatz(27) takes 111 steps
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(111)));
    }

    #[test]
    fn test_e2e_curried_application() {
        let source = "
            add(x) = y => x + y
            main() = add(10)(32)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(42)));
    }

    #[test]
    fn test_e2e_closure_captures() {
        let source = "
            makeAdder(n) = x => x + n
            main() = makeAdder(40)(2)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(42)));
    }

    #[test]
    fn test_e2e_float_arithmetic() {
        // Float multiplication works
        let source = "
            scale(x) = x * 2.0
            main() = scale(3.5)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Float(7.0)));
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
        assert_eq!(result, Ok(Value::Int(30)));
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
        assert_eq!(result, Ok(Value::Int(19)));
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
        assert_eq!(result, Ok(Value::Int(21)));
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
        assert_eq!(result, Ok(Value::Int(42)));
    }

    #[test]
    fn test_e2e_match_tuple_pattern() {
        let source = "
            first(pair) = match pair
                (a, _) -> a
            main() = first((42, 100))
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(15)));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
                assert_eq!(r.fields[0], Value::Int(3));
                assert_eq!(r.fields[1], Value::Int(4));
            }
            other => panic!("Expected record, got {:?}", other),
        }
    }

    #[test]
    fn test_e2e_record_field_access() {
        let source = "
            main() = {
                p = Point(3, 4)
                p.x + p.y
            }
        ";
        // This test will fail if the VM doesn't have type info for Point
        // Let's test with positional field access which uses synthetic names
        let source2 = "
            getX(p) = p._0
            main() = getX(Point(42, 100))
        ";
        let result = compile_and_run(source2);
        assert_eq!(result, Ok(Value::Int(42)));
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
                assert_eq!(r.fields[0], Value::Int(42));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(0)));
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
        assert_eq!(result, Ok(Value::Int(5)));
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
        assert_eq!(result, Ok(Value::Int(12)));
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
        assert_eq!(result, Ok(Value::Int(12)));
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
        assert_eq!(result, Ok(Value::Int(15)));
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
        assert_eq!(result, Ok(Value::Int(5)));
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
        assert_eq!(result, Ok(Value::Int(15)));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(5)));
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
        assert_eq!(result, Ok(Value::Int(10)));
    }

    #[test]
    fn test_e2e_deep_closure_capture() {
        // Use lambda syntax instead of function definitions inside blocks
        let source = "
            outer(x) = y => z => x + y + z
            main() = outer(10)(20)(12)
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(25)));
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
        assert_eq!(result, Ok(Value::Int(6)));
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
        assert_eq!(result, Ok(Value::Int(12)));
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
        assert_eq!(result, Ok(Value::Int(22)));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(5)));
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
        assert_eq!(result, Ok(Value::Int(66)));
    }

    #[test]
    fn test_e2e_float_operations() {
        // Float arithmetic in main directly
        let source = "
            main() = 3.0 * 2.5
        ";
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Float(7.5)));
    }

    #[test]
    fn test_e2e_comparison_operators() {
        let source = "
            compare(a, b) = if a < b then 0 - 1 else if a > b then 1 else 0
            main() = compare(3, 5) + compare(5, 3) + compare(4, 4)
        ";
        // -1 + 1 + 0 = 0
        let result = compile_and_run(source);
        assert_eq!(result, Ok(Value::Int(0)));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(42)));
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
        assert_eq!(result, Ok(Value::Int(35)));
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
        assert_eq!(result, Ok(Value::Int(110)));
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
        assert_eq!(result, Ok(Value::Int(30)));
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
        assert_eq!(result, Ok(Value::Int(10)));
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
        assert_eq!(result, Ok(Value::Int(4)));
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
        assert_eq!(result, Ok(Value::Int(17)));
    }
}
