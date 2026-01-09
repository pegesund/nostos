//! JIT Compiler for Nostos
//!
//! Tiered compilation strategy:
//! - Tier 0: Interpreter (existing Runtime)
//! - Tier 1: Baseline JIT (this crate) - eliminate dispatch overhead
//! - Tier 2: Optimizing JIT (future) - type specialization, inlining
//!
//! Uses Cranelift as the code generation backend.
//!
//! Current implementation: Specializes for pure numeric functions.
//! Supported types: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64

use std::collections::HashMap;
use std::rc::Rc;

use cranelift_codegen::ir::{AbiParam, Block, InstBuilder, UserFuncName, Value as CraneliftValue};
use cranelift_codegen::ir::condcodes::{IntCC, FloatCC};
use cranelift_codegen::ir::types::{I8, I16, I32, I64, F32, F64, Type as CraneliftType};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use nostos_vm::value::{ConstIdx, FunctionValue, Instruction, Value};

/// Numeric types supported by the JIT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl NumericType {
    /// Get the corresponding Cranelift type
    fn cranelift_type(&self) -> CraneliftType {
        match self {
            NumericType::Int8 | NumericType::UInt8 => I8,
            NumericType::Int16 | NumericType::UInt16 => I16,
            NumericType::Int32 | NumericType::UInt32 => I32,
            NumericType::Int64 | NumericType::UInt64 => I64,
            NumericType::Float32 => F32,
            NumericType::Float64 => F64,
        }
    }

    /// Check if this type is a float
    fn is_float(&self) -> bool {
        matches!(self, NumericType::Float32 | NumericType::Float64)
    }

    /// Check if this type is unsigned
    fn is_unsigned(&self) -> bool {
        matches!(self, NumericType::UInt8 | NumericType::UInt16 | NumericType::UInt32 | NumericType::UInt64)
    }

    /// Name suffix for generated functions
    fn suffix(&self) -> &'static str {
        match self {
            NumericType::Int8 => "i8",
            NumericType::Int16 => "i16",
            NumericType::Int32 => "i32",
            NumericType::Int64 => "i64",
            NumericType::UInt8 => "u8",
            NumericType::UInt16 => "u16",
            NumericType::UInt32 => "u32",
            NumericType::UInt64 => "u64",
            NumericType::Float32 => "f32",
            NumericType::Float64 => "f64",
        }
    }
}

/// JIT compilation threshold - compile after this many calls
pub const JIT_THRESHOLD: u32 = 1000;

/// Errors that can occur during JIT compilation
#[derive(Debug, thiserror::Error)]
pub enum JitError {
    #[error("Cranelift error: {0}")]
    Cranelift(String),

    #[error("Unsupported instruction: {0:?}")]
    UnsupportedInstruction(String),

    #[error("Module error: {0}")]
    Module(String),

    #[error("Function not suitable for JIT: {0}")]
    NotSuitable(String),
}

/// Configuration for the JIT compiler
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Number of calls before a function is JIT-compiled
    pub hot_threshold: u32,
    /// Whether JIT is enabled at all
    pub enabled: bool,
    /// Optimization level (0 = none, 1 = basic, 2 = full)
    pub opt_level: u8,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            hot_threshold: JIT_THRESHOLD,
            enabled: true,
            opt_level: 1,
        }
    }
}

/// A compiled native function for pure numeric functions
pub struct CompiledFunction {
    /// Pointer to the native code
    pub code_ptr: *const u8,
    /// Function ID in the module
    pub func_id: FuncId,
    /// Number of arguments
    pub arity: usize,
    /// The numeric type this function operates on
    pub numeric_type: NumericType,
}

/// The JIT compiler
pub struct JitCompiler {
    /// Cranelift JIT module
    module: JITModule,
    /// Cranelift codegen context (reusable)
    ctx: Context,
    /// Function builder context (reusable)
    builder_ctx: FunctionBuilderContext,
    /// Cache of compiled functions: (function_index, numeric_type) → compiled function
    cache: HashMap<(u16, NumericType), CompiledFunction>,
    /// Configuration
    #[allow(dead_code)]
    config: JitConfig,
    /// Functions queued for compilation
    compile_queue: Vec<u16>,
    /// Declared function IDs for self-recursion: (func_index, type) → FuncId
    declared_funcs: HashMap<(u16, NumericType), FuncId>,
}

impl JitCompiler {
    /// Create a new JIT compiler
    pub fn new(config: JitConfig) -> Result<Self, JitError> {
        // Set up Cranelift with native target
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", match config.opt_level {
            0 => "none",
            1 => "speed",
            _ => "speed_and_size",
        }).map_err(|e| JitError::Cranelift(e.to_string()))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|e| JitError::Cranelift(e.to_string()))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Cranelift(e.to_string()))?;

        // Create JIT module
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            module,
            ctx: Context::new(),
            builder_ctx: FunctionBuilderContext::new(),
            cache: HashMap::new(),
            config,
            compile_queue: Vec::new(),
            declared_funcs: HashMap::new(),
        })
    }

    /// Check if a function is already JIT-compiled for the default Int64 type
    pub fn is_compiled(&self, func_index: u16) -> bool {
        self.cache.contains_key(&(func_index, NumericType::Int64))
    }

    /// Check if a function is compiled for a specific numeric type
    pub fn is_compiled_for(&self, func_index: u16, num_type: NumericType) -> bool {
        self.cache.contains_key(&(func_index, num_type))
    }

    /// Get the native code pointer for a compiled Int64 function (for backward compatibility)
    pub fn get_int_function(&self, func_index: u16) -> Option<fn(i64) -> i64> {
        self.cache.get(&(func_index, NumericType::Int64)).map(|f| {
            unsafe { std::mem::transmute::<*const u8, fn(i64) -> i64>(f.code_ptr) }
        })
    }

    /// Get the native code pointer for a compiled Int32 function
    pub fn get_int32_function(&self, func_index: u16) -> Option<fn(i32) -> i32> {
        self.cache.get(&(func_index, NumericType::Int32)).map(|f| {
            unsafe { std::mem::transmute::<*const u8, fn(i32) -> i32>(f.code_ptr) }
        })
    }

    /// Get the native code pointer for a compiled Float64 function
    pub fn get_float64_function(&self, func_index: u16) -> Option<fn(f64) -> f64> {
        self.cache.get(&(func_index, NumericType::Float64)).map(|f| {
            unsafe { std::mem::transmute::<*const u8, fn(f64) -> f64>(f.code_ptr) }
        })
    }

    /// Get the native code pointer for a compiled Float32 function
    pub fn get_float32_function(&self, func_index: u16) -> Option<fn(f32) -> f32> {
        self.cache.get(&(func_index, NumericType::Float32)).map(|f| {
            unsafe { std::mem::transmute::<*const u8, fn(f32) -> f32>(f.code_ptr) }
        })
    }

    /// Get the compiled function info for a specific type
    pub fn get_compiled(&self, func_index: u16, num_type: NumericType) -> Option<&CompiledFunction> {
        self.cache.get(&(func_index, num_type))
    }

    /// Queue a function for compilation
    pub fn queue_compilation(&mut self, func_index: u16) {
        if !self.is_compiled(func_index) && !self.compile_queue.contains(&func_index) {
            self.compile_queue.push(func_index);
        }
    }

    /// Process the compilation queue
    pub fn process_queue(&mut self, functions: &[Rc<FunctionValue>]) -> Result<usize, JitError> {
        let mut compiled = 0;

        while let Some(func_index) = self.compile_queue.pop() {
            if let Some(func) = functions.get(func_index as usize) {
                match self.compile_int_function(func_index, func) {
                    Ok(_) => {
                        compiled += 1;
                    }
                    Err(JitError::NotSuitable(_)) => {
                        // Function not suitable for integer JIT - this is fine, silently skip
                    }
                    Err(_) => {
                        // Compilation error - silently skip for now
                    }
                }
            }
        }

        Ok(compiled)
    }

    /// Detect the numeric type of a pure numeric function.
    /// Returns the detected type if the function is suitable for JIT, or an error otherwise.
    fn detect_numeric_type(&self, func: &FunctionValue) -> Result<NumericType, JitError> {
        // Must have exactly 1 argument for now (simplifies things)
        if func.arity != 1 {
            return Err(JitError::NotSuitable(format!("arity {} != 1", func.arity)));
        }

        let mut detected_type: Option<NumericType> = None;

        // Check all instructions and constants to determine the type
        for instr in func.code.code.iter() {
            match instr {
                // Check constant types
                Instruction::LoadConst(_, idx) => {
                    if let Some(val) = func.code.constants.get(*idx as usize) {
                        let new_type = match val {
                            Value::Int64(_) => Some(NumericType::Int64),
                            Value::Int32(_) => Some(NumericType::Int32),
                            Value::Int16(_) => Some(NumericType::Int16),
                            Value::Int8(_) => Some(NumericType::Int8),
                            Value::UInt64(_) => Some(NumericType::UInt64),
                            Value::UInt32(_) => Some(NumericType::UInt32),
                            Value::UInt16(_) => Some(NumericType::UInt16),
                            Value::UInt8(_) => Some(NumericType::UInt8),
                            Value::Float64(_) => Some(NumericType::Float64),
                            Value::Float32(_) => Some(NumericType::Float32),
                            Value::Bool(_) => None, // Bools are allowed in any numeric function
                            _ => return Err(JitError::NotSuitable(
                                format!("non-numeric constant: {:?}", val)
                            )),
                        };
                        if let Some(new_type) = new_type {
                            if let Some(existing) = detected_type {
                                if existing != new_type {
                                    return Err(JitError::NotSuitable(
                                        format!("mixed types: {:?} and {:?}", existing, new_type)
                                    ));
                                }
                            } else {
                                detected_type = Some(new_type);
                            }
                        }
                    }
                }
                // Allowed for any type
                Instruction::Move(_, _) => {}
                Instruction::LoadTrue(_) | Instruction::LoadFalse(_) => {}
                Instruction::Jump(_) => {}
                Instruction::JumpIfTrue(_, _) => {}
                Instruction::JumpIfFalse(_, _) => {}
                Instruction::Return(_) => {}
                Instruction::CallSelf(_, _) => {}
                Instruction::TailCallSelf(_) => {}

                // Integer arithmetic
                Instruction::AddInt(_, _, _) |
                Instruction::SubInt(_, _, _) |
                Instruction::MulInt(_, _, _) |
                Instruction::NegInt(_, _) |
                Instruction::EqInt(_, _, _) |
                Instruction::NeInt(_, _, _) |
                Instruction::LtInt(_, _, _) |
                Instruction::LeInt(_, _, _) |
                Instruction::GtInt(_, _, _) |
                Instruction::GeInt(_, _, _) => {
                    // Integer instructions - if no type detected yet, default to Int64
                    if detected_type.is_none() {
                        detected_type = Some(NumericType::Int64);
                    }
                }

                // Float arithmetic
                Instruction::AddFloat(_, _, _) |
                Instruction::SubFloat(_, _, _) |
                Instruction::MulFloat(_, _, _) |
                Instruction::DivFloat(_, _, _) |
                Instruction::NegFloat(_, _) |
                Instruction::LtFloat(_, _, _) |
                Instruction::LeFloat(_, _, _) |
                Instruction::EqFloat(_, _, _) => {
                    // Float instructions - if no type detected yet, default to Float64
                    if detected_type.is_none() {
                        detected_type = Some(NumericType::Float64);
                    }
                }

                // Not supported
                other => {
                    return Err(JitError::NotSuitable(
                        format!("unsupported instruction: {:?}", other)
                    ));
                }
            }
        }

        // Default to Int64 if no type-specific hints were found
        Ok(detected_type.unwrap_or(NumericType::Int64))
    }

    /// Compile a pure numeric function to native code (backward compatible - compiles for Int64)
    pub fn compile_int_function(
        &mut self,
        func_index: u16,
        func: &FunctionValue,
    ) -> Result<(), JitError> {
        self.compile_numeric_function(func_index, func)
    }

    /// Compile a pure numeric function to native code
    pub fn compile_numeric_function(
        &mut self,
        func_index: u16,
        func: &FunctionValue,
    ) -> Result<(), JitError> {
        // Detect the numeric type from the function's constants and instructions
        let num_type = self.detect_numeric_type(func)?;

        // Skip if already compiled for this type
        if self.is_compiled_for(func_index, num_type) {
            return Ok(());
        }

        let cl_type = num_type.cranelift_type();

        // Create signature: fn(T) -> T where T is the detected numeric type
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(cl_type));
        sig.returns.push(AbiParam::new(cl_type));

        // Declare function in module (needed for self-recursion)
        let func_name = format!("nos_{}_{}", num_type.suffix(), func_index);
        let func_id = self.module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        self.declared_funcs.insert((func_index, num_type), func_id);

        // Build function body
        self.ctx.func.signature = sig.clone();
        self.ctx.func.name = UserFuncName::user(0, func_index as u32);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            // Create entry block
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);

            // Allocate virtual registers (Cranelift Variables) for bytecode registers
            let reg_count = func.code.register_count;
            let mut regs: Vec<Variable> = Vec::with_capacity(reg_count);
            for i in 0..reg_count {
                let var = Variable::from_u32(i as u32);
                builder.declare_var(var, cl_type);
                regs.push(var);
            }

            // Initialize first register with the argument
            let arg = builder.block_params(entry_block)[0];
            builder.def_var(regs[0], arg);

            // Create blocks for jump targets
            let mut jump_targets: HashMap<usize, Block> = HashMap::new();
            for (ip, instr) in func.code.code.iter().enumerate() {
                match instr {
                    Instruction::Jump(offset) => {
                        let target = (ip as i32 + *offset as i32 + 1) as usize;
                        if !jump_targets.contains_key(&target) {
                            jump_targets.insert(target, builder.create_block());
                        }
                    }
                    Instruction::JumpIfTrue(_, offset) | Instruction::JumpIfFalse(_, offset) => {
                        let target = (ip as i32 + *offset as i32 + 1) as usize;
                        if !jump_targets.contains_key(&target) {
                            jump_targets.insert(target, builder.create_block());
                        }
                        let next = ip + 1;
                        if !jump_targets.contains_key(&next) {
                            jump_targets.insert(next, builder.create_block());
                        }
                    }
                    _ => {}
                }
            }

            // Import our own function for recursive calls
            let self_func_ref = self.module.declare_func_in_func(func_id, builder.func);

            // Compile each instruction
            let mut ip = 0;
            let mut block_terminated = false;
            while ip < func.code.code.len() {
                if let Some(&block) = jump_targets.get(&ip) {
                    if !block_terminated {
                        builder.ins().jump(block, &[]);
                    }
                    builder.switch_to_block(block);
                    block_terminated = false;
                }

                let instr = &func.code.code[ip];

                match instr {
                    Instruction::LoadConst(dst, idx) => {
                        let v = Self::load_const(&mut builder, &func.code.constants, *idx, num_type, cl_type)?;
                        builder.def_var(regs[*dst as usize], v);
                    }

                    Instruction::Move(dst, src) => {
                        let v = builder.use_var(regs[*src as usize]);
                        builder.def_var(regs[*dst as usize], v);
                    }

                    Instruction::LoadTrue(dst) => {
                        // Bools are represented as 0/1 in the numeric type
                        let v = if num_type.is_float() {
                            builder.ins().f64const(1.0)
                        } else {
                            builder.ins().iconst(cl_type, 1)
                        };
                        builder.def_var(regs[*dst as usize], v);
                    }

                    Instruction::LoadFalse(dst) => {
                        let v = if num_type.is_float() {
                            builder.ins().f64const(0.0)
                        } else {
                            builder.ins().iconst(cl_type, 0)
                        };
                        builder.def_var(regs[*dst as usize], v);
                    }

                    // Integer arithmetic (also handles small int types)
                    Instruction::AddInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().iadd(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::SubInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().isub(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::MulInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().imul(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::NegInt(dst, src) => {
                        let v = builder.use_var(regs[*src as usize]);
                        let result = builder.ins().ineg(v);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    // Integer comparisons (use signed or unsigned based on type)
                    Instruction::EqInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::Equal, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::NeInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::NotEqual, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::LtInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cc = if num_type.is_unsigned() { IntCC::UnsignedLessThan } else { IntCC::SignedLessThan };
                        let cmp = builder.ins().icmp(cc, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::LeInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cc = if num_type.is_unsigned() { IntCC::UnsignedLessThanOrEqual } else { IntCC::SignedLessThanOrEqual };
                        let cmp = builder.ins().icmp(cc, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::GtInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cc = if num_type.is_unsigned() { IntCC::UnsignedGreaterThan } else { IntCC::SignedGreaterThan };
                        let cmp = builder.ins().icmp(cc, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::GeInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cc = if num_type.is_unsigned() { IntCC::UnsignedGreaterThanOrEqual } else { IntCC::SignedGreaterThanOrEqual };
                        let cmp = builder.ins().icmp(cc, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    // Float arithmetic
                    Instruction::AddFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().fadd(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::SubFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().fsub(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::MulFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().fmul(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::DivFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let result = builder.ins().fdiv(va, vb);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::NegFloat(dst, src) => {
                        let v = builder.use_var(regs[*src as usize]);
                        let result = builder.ins().fneg(v);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    // Float comparisons
                    Instruction::EqFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().fcmp(FloatCC::Equal, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::LtFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().fcmp(FloatCC::LessThan, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::LeFloat(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, va, vb);
                        let result = builder.ins().uextend(cl_type, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    // Control flow
                    Instruction::Jump(offset) => {
                        let target = (ip as i32 + *offset as i32 + 1) as usize;
                        let target_block = jump_targets[&target];
                        builder.ins().jump(target_block, &[]);
                        block_terminated = true;
                    }

                    Instruction::JumpIfTrue(cond, offset) => {
                        let cond_val = builder.use_var(regs[*cond as usize]);
                        let target = (ip as i32 + *offset as i32 + 1) as usize;
                        let target_block = jump_targets[&target];
                        let next_block = jump_targets[&(ip + 1)];
                        // For floats, compare against zero; for ints, brif works directly
                        if num_type.is_float() {
                            let zero = if cl_type == F32 { builder.ins().f32const(0.0) } else { builder.ins().f64const(0.0) };
                            let cmp = builder.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                            builder.ins().brif(cmp, target_block, &[], next_block, &[]);
                        } else {
                            builder.ins().brif(cond_val, target_block, &[], next_block, &[]);
                        }
                        block_terminated = true;
                    }

                    Instruction::JumpIfFalse(cond, offset) => {
                        let cond_val = builder.use_var(regs[*cond as usize]);
                        let target = (ip as i32 + *offset as i32 + 1) as usize;
                        let target_block = jump_targets[&target];
                        let next_block = jump_targets[&(ip + 1)];
                        if num_type.is_float() {
                            let zero = if cl_type == F32 { builder.ins().f32const(0.0) } else { builder.ins().f64const(0.0) };
                            let cmp = builder.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                            builder.ins().brif(cmp, next_block, &[], target_block, &[]);
                        } else {
                            builder.ins().brif(cond_val, next_block, &[], target_block, &[]);
                        }
                        block_terminated = true;
                    }

                    Instruction::Return(src) => {
                        let v = builder.use_var(regs[*src as usize]);
                        builder.ins().return_(&[v]);
                        block_terminated = true;
                    }

                    // Self-recursion
                    Instruction::CallSelf(dst, arg_regs) => {
                        let args: Vec<CraneliftValue> = arg_regs.iter()
                            .map(|&r| builder.use_var(regs[r as usize]))
                            .collect();
                        let call = builder.ins().call(self_func_ref, &args);
                        let result = builder.inst_results(call)[0];
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::TailCallSelf(arg_regs) => {
                        let args: Vec<CraneliftValue> = arg_regs.iter()
                            .map(|&r| builder.use_var(regs[r as usize]))
                            .collect();
                        builder.ins().return_call(self_func_ref, &args);
                        block_terminated = true;
                    }

                    other => {
                        return Err(JitError::UnsupportedInstruction(format!("{:?}", other)));
                    }
                }

                ip += 1;
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

        // Compile to native code
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JitError::Module(e.to_string()))?;

        self.module.finalize_definitions()
            .map_err(|e| JitError::Module(e.to_string()))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // Cache the compiled function
        self.cache.insert((func_index, num_type), CompiledFunction {
            code_ptr,
            func_id,
            arity: func.arity,
            numeric_type: num_type,
        });

        self.module.clear_context(&mut self.ctx);

        Ok(())
    }

    /// Load a constant value, returning the appropriate Cranelift value
    fn load_const(
        builder: &mut FunctionBuilder,
        constants: &[Value],
        idx: ConstIdx,
        num_type: NumericType,
        cl_type: CraneliftType,
    ) -> Result<CraneliftValue, JitError> {
        match constants.get(idx as usize) {
            Some(Value::Int64(i)) => Ok(builder.ins().iconst(cl_type, *i)),
            Some(Value::Int32(i)) => Ok(builder.ins().iconst(cl_type, *i as i64)),
            Some(Value::Int16(i)) => Ok(builder.ins().iconst(cl_type, *i as i64)),
            Some(Value::Int8(i)) => Ok(builder.ins().iconst(cl_type, *i as i64)),
            Some(Value::UInt64(u)) => Ok(builder.ins().iconst(cl_type, *u as i64)),
            Some(Value::UInt32(u)) => Ok(builder.ins().iconst(cl_type, *u as i64)),
            Some(Value::UInt16(u)) => Ok(builder.ins().iconst(cl_type, *u as i64)),
            Some(Value::UInt8(u)) => Ok(builder.ins().iconst(cl_type, *u as i64)),
            Some(Value::Float64(f)) => {
                if num_type == NumericType::Float32 {
                    Ok(builder.ins().f32const(*f as f32))
                } else {
                    Ok(builder.ins().f64const(*f))
                }
            }
            Some(Value::Float32(f)) => {
                if num_type == NumericType::Float64 {
                    Ok(builder.ins().f64const(*f as f64))
                } else {
                    Ok(builder.ins().f32const(*f))
                }
            }
            Some(Value::Bool(b)) => {
                if num_type.is_float() {
                    if cl_type == F32 {
                        Ok(builder.ins().f32const(if *b { 1.0 } else { 0.0 }))
                    } else {
                        Ok(builder.ins().f64const(if *b { 1.0 } else { 0.0 }))
                    }
                } else {
                    Ok(builder.ins().iconst(cl_type, if *b { 1 } else { 0 }))
                }
            }
            Some(other) => Err(JitError::NotSuitable(format!("non-numeric constant: {:?}", other))),
            None => Err(JitError::NotSuitable(format!("constant {} not found", idx))),
        }
    }

    /// Get JIT statistics
    pub fn stats(&self) -> JitStats {
        JitStats {
            compiled_functions: self.cache.len(),
            queued_functions: self.compile_queue.len(),
        }
    }
}

/// JIT compilation statistics
#[derive(Debug, Clone)]
pub struct JitStats {
    pub compiled_functions: usize,
    pub queued_functions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use nostos_vm::value::Chunk;

    #[test]
    fn test_jit_compiler_creation() {
        let config = JitConfig::default();
        let jit = JitCompiler::new(config);
        assert!(jit.is_ok(), "Failed to create JIT compiler: {:?}", jit.err());
    }

    /// Create a simple identity function: fn(n) -> n
    fn make_identity_function() -> FunctionValue {
        let mut chunk = Chunk::new();
        // Just return register 0 (the argument)
        chunk.code.push(Instruction::Return(0));
        chunk.register_count = 1;

        FunctionValue {
            name: "identity".to_string(),
            arity: 1,
            param_names: vec!["n".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: Cell::new(0),
        }
    }

    /// Create an add1 function: fn(n) -> n + 1
    fn make_add1_function() -> FunctionValue {
        let mut chunk = Chunk::new();
        // Load constant 1 into r1
        chunk.constants.push(Value::Int64(1));
        chunk.code.push(Instruction::LoadConst(1, 0)); // r1 = 1
        // r2 = r0 + r1
        chunk.code.push(Instruction::AddInt(2, 0, 1));
        // Return r2
        chunk.code.push(Instruction::Return(2));
        chunk.register_count = 3;

        FunctionValue {
            name: "add1".to_string(),
            arity: 1,
            param_names: vec!["n".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: Cell::new(0),
        }
    }

    #[test]
    fn test_jit_identity_function() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config).unwrap();

        let func = make_identity_function();
        jit.compile_int_function(0, &func).expect("JIT compilation failed");

        // Get the compiled function and call it
        let native_fn = jit.get_int_function(0).expect("Function not compiled");
        assert_eq!(native_fn(42), 42);
        assert_eq!(native_fn(0), 0);
        assert_eq!(native_fn(-100), -100);
    }

    #[test]
    fn test_jit_add1_function() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config).unwrap();

        let func = make_add1_function();
        jit.compile_int_function(0, &func).expect("JIT compilation failed");

        // Get the compiled function and call it
        let native_fn = jit.get_int_function(0).expect("Function not compiled");
        assert_eq!(native_fn(5), 6);
        assert_eq!(native_fn(0), 1);
        assert_eq!(native_fn(-1), 0);
    }

    /// Create a simple recursive function: fib(n) = if n <= 1 then n else fib(n-1) + fib(n-2)
    fn make_fib_function() -> FunctionValue {
        let mut chunk = Chunk::new();

        // Constants
        chunk.constants.push(Value::Int64(1)); // idx 0: constant 1
        chunk.constants.push(Value::Int64(2)); // idx 1: constant 2

        // r0 = n (argument)
        // r1 = 1 (constant)
        // r2 = n <= 1 (condition)
        // r3 = n - 1
        // r4 = fib(n-1)
        // r5 = n - 2
        // r6 = fib(n-2)
        // r7 = result

        // LoadConst r1, 0    ; r1 = 1
        chunk.code.push(Instruction::LoadConst(1, 0));
        // LeInt r2, r0, r1   ; r2 = n <= 1
        chunk.code.push(Instruction::LeInt(2, 0, 1));
        // JumpIfFalse r2, 1  ; if !(n <= 1) jump to else branch (IP=2+1+1=4)
        chunk.code.push(Instruction::JumpIfFalse(2, 1));
        // Return r0          ; return n (base case)
        chunk.code.push(Instruction::Return(0));
        // SubInt r3, r0, r1  ; r3 = n - 1 (else branch starts here, IP=4)
        chunk.code.push(Instruction::SubInt(3, 0, 1));
        // CallSelf r4, [r3]  ; r4 = fib(n-1)
        chunk.code.push(Instruction::CallSelf(4, Rc::new([3])));
        // LoadConst r5, 1    ; r5 = 2
        chunk.code.push(Instruction::LoadConst(5, 1));
        // SubInt r6, r0, r5  ; r6 = n - 2
        chunk.code.push(Instruction::SubInt(6, 0, 5));
        // CallSelf r7, [r6]  ; r7 = fib(n-2)
        chunk.code.push(Instruction::CallSelf(7, Rc::new([6])));
        // AddInt r8, r4, r7  ; r8 = fib(n-1) + fib(n-2)
        chunk.code.push(Instruction::AddInt(8, 4, 7));
        // Return r8
        chunk.code.push(Instruction::Return(8));

        chunk.register_count = 9;

        FunctionValue {
            name: "fib".to_string(),
            arity: 1,
            param_names: vec!["n".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: Cell::new(0),
        }
    }

    #[test]
    fn test_jit_fib_function() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config).unwrap();

        let func = make_fib_function();
        jit.compile_int_function(0, &func).expect("JIT compilation failed");

        // Get the compiled function and call it
        let native_fn = jit.get_int_function(0).expect("Function not compiled");

        // Test fibonacci sequence
        assert_eq!(native_fn(0), 0);
        assert_eq!(native_fn(1), 1);
        assert_eq!(native_fn(2), 1);
        assert_eq!(native_fn(3), 2);
        assert_eq!(native_fn(4), 3);
        assert_eq!(native_fn(5), 5);
        assert_eq!(native_fn(10), 55);
        assert_eq!(native_fn(20), 6765);
    }

    #[test]
    fn test_jit_fib_35() {
        let config = JitConfig::default();
        let mut jit = JitCompiler::new(config).unwrap();

        let func = make_fib_function();
        jit.compile_int_function(0, &func).expect("JIT compilation failed");

        let native_fn = jit.get_int_function(0).expect("Function not compiled");

        // Time fib(35)
        let start = std::time::Instant::now();
        let result = native_fn(35);
        let elapsed = start.elapsed();

        assert_eq!(result, 9227465);
        eprintln!("[JIT] fib(35) = {} in {:?}", result, elapsed);
    }
}
