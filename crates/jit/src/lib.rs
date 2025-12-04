//! JIT Compiler for Nostos
//!
//! Tiered compilation strategy:
//! - Tier 0: Interpreter (existing Runtime)
//! - Tier 1: Baseline JIT (this crate) - eliminate dispatch overhead
//! - Tier 2: Optimizing JIT (future) - type specialization, inlining
//!
//! Uses Cranelift as the code generation backend.
//!
//! Current implementation: Specializes for pure integer functions (i64 → i64).
//! This covers common numeric benchmarks like fibonacci.

use std::collections::HashMap;
use std::rc::Rc;

use cranelift_codegen::ir::{AbiParam, Block, InstBuilder, UserFuncName, Value as CraneliftValue};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::I64;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use nostos_vm::value::{ConstIdx, FunctionValue, Instruction, Value};

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

/// A compiled native function for pure integer functions
pub struct CompiledIntFunction {
    /// Pointer to the native code: fn(i64) -> i64
    pub code_ptr: *const u8,
    /// Function ID in the module
    pub func_id: FuncId,
    /// Number of arguments
    pub arity: usize,
}

/// The JIT compiler
pub struct JitCompiler {
    /// Cranelift JIT module
    module: JITModule,
    /// Cranelift codegen context (reusable)
    ctx: Context,
    /// Function builder context (reusable)
    builder_ctx: FunctionBuilderContext,
    /// Cache of compiled integer functions: function_index → compiled function
    int_cache: HashMap<u16, CompiledIntFunction>,
    /// Configuration
    #[allow(dead_code)]
    config: JitConfig,
    /// Functions queued for compilation
    compile_queue: Vec<u16>,
    /// Declared function IDs for self-recursion
    declared_funcs: HashMap<u16, FuncId>,
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
            int_cache: HashMap::new(),
            config,
            compile_queue: Vec::new(),
            declared_funcs: HashMap::new(),
        })
    }

    /// Check if a function is already JIT-compiled
    pub fn is_compiled(&self, func_index: u16) -> bool {
        self.int_cache.contains_key(&func_index)
    }

    /// Get the native code pointer for a compiled integer function
    pub fn get_int_function(&self, func_index: u16) -> Option<fn(i64) -> i64> {
        self.int_cache.get(&func_index).map(|f| {
            unsafe { std::mem::transmute::<*const u8, fn(i64) -> i64>(f.code_ptr) }
        })
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

    /// Check if a function is a pure integer function (only uses int operations)
    fn is_pure_int_function(&self, func: &FunctionValue) -> Result<(), JitError> {
        // Must have exactly 1 argument for now (simplifies things)
        if func.arity != 1 {
            return Err(JitError::NotSuitable(format!("arity {} != 1", func.arity)));
        }

        // Check all instructions
        for instr in func.code.code.iter() {
            match instr {
                // Allowed instructions for pure int functions
                Instruction::LoadConst(_, idx) => {
                    // Check that the constant is an integer
                    if let Some(val) = func.code.constants.get(*idx as usize) {
                        match val {
                            Value::Int(_) => {}
                            Value::Bool(_) => {}
                            _ => return Err(JitError::NotSuitable(
                                format!("non-int constant: {:?}", val)
                            )),
                        }
                    }
                }
                Instruction::Move(_, _) => {}
                Instruction::LoadTrue(_) | Instruction::LoadFalse(_) => {}

                // Integer arithmetic
                Instruction::AddInt(_, _, _) => {}
                Instruction::SubInt(_, _, _) => {}
                Instruction::MulInt(_, _, _) => {}
                Instruction::NegInt(_, _) => {}

                // Integer comparisons
                Instruction::EqInt(_, _, _) => {}
                Instruction::NeInt(_, _, _) => {}
                Instruction::LtInt(_, _, _) => {}
                Instruction::LeInt(_, _, _) => {}
                Instruction::GtInt(_, _, _) => {}
                Instruction::GeInt(_, _, _) => {}

                // Control flow
                Instruction::Jump(_) => {}
                Instruction::JumpIfTrue(_, _) => {}
                Instruction::JumpIfFalse(_, _) => {}
                Instruction::Return(_) => {}

                // Self-recursion (can be compiled to native call)
                Instruction::CallSelf(_, _) => {}
                Instruction::TailCallSelf(_) => {}

                // Not supported
                other => {
                    return Err(JitError::NotSuitable(
                        format!("unsupported instruction: {:?}", other)
                    ));
                }
            }
        }

        Ok(())
    }

    /// Compile a pure integer function to native code
    pub fn compile_int_function(
        &mut self,
        func_index: u16,
        func: &FunctionValue,
    ) -> Result<(), JitError> {
        // Skip if already compiled
        if self.is_compiled(func_index) {
            return Ok(());
        }

        // Check if this function is suitable for integer JIT
        self.is_pure_int_function(func)?;

        // Create signature: fn(i64) -> i64
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(I64));
        sig.returns.push(AbiParam::new(I64));

        // Declare function in module (needed for self-recursion)
        let func_name = format!("nos_int_{}", func_index);
        let func_id = self.module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        self.declared_funcs.insert(func_index, func_id);

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
                builder.declare_var(var, I64);
                regs.push(var);
            }

            // Initialize first register with the argument
            let arg = builder.block_params(entry_block)[0];
            builder.def_var(regs[0], arg);

            // Create blocks for each instruction (for jump targets)
            // First, identify all jump targets
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
                        // Also need a block for fall-through
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
                // Check if we need to switch to a new block
                if let Some(&block) = jump_targets.get(&ip) {
                    // Jump from previous block to this one (if not already terminated)
                    if !block_terminated {
                        builder.ins().jump(block, &[]);
                    }
                    builder.switch_to_block(block);
                    block_terminated = false;
                }

                let instr = &func.code.code[ip];

                match instr {
                    Instruction::LoadConst(dst, idx) => {
                        let val = Self::get_int_const_from(&func.code.constants, *idx)?;
                        let v = builder.ins().iconst(I64, val);
                        builder.def_var(regs[*dst as usize], v);
                    }

                    Instruction::Move(dst, src) => {
                        let v = builder.use_var(regs[*src as usize]);
                        builder.def_var(regs[*dst as usize], v);
                    }

                    Instruction::LoadTrue(dst) => {
                        let v = builder.ins().iconst(I64, 1);
                        builder.def_var(regs[*dst as usize], v);
                    }

                    Instruction::LoadFalse(dst) => {
                        let v = builder.ins().iconst(I64, 0);
                        builder.def_var(regs[*dst as usize], v);
                    }

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

                    // Integer comparisons - produce 0 or 1
                    Instruction::EqInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::Equal, va, vb);
                        let result = builder.ins().uextend(I64, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::NeInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::NotEqual, va, vb);
                        let result = builder.ins().uextend(I64, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::LtInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::SignedLessThan, va, vb);
                        let result = builder.ins().uextend(I64, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::LeInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, va, vb);
                        let result = builder.ins().uextend(I64, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::GtInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, va, vb);
                        let result = builder.ins().uextend(I64, cmp);
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::GeInt(dst, a, b) => {
                        let va = builder.use_var(regs[*a as usize]);
                        let vb = builder.use_var(regs[*b as usize]);
                        let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, va, vb);
                        let result = builder.ins().uextend(I64, cmp);
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

                        // brif: if cond != 0, jump to target_block, else jump to next_block
                        builder.ins().brif(cond_val, target_block, &[], next_block, &[]);
                        block_terminated = true;
                    }

                    Instruction::JumpIfFalse(cond, offset) => {
                        let cond_val = builder.use_var(regs[*cond as usize]);
                        let target = (ip as i32 + *offset as i32 + 1) as usize;
                        let target_block = jump_targets[&target];
                        let next_block = jump_targets[&(ip + 1)];

                        // brif: if cond != 0 jump to next_block, else jump to target_block
                        builder.ins().brif(cond_val, next_block, &[], target_block, &[]);
                        block_terminated = true;
                    }

                    Instruction::Return(src) => {
                        let v = builder.use_var(regs[*src as usize]);
                        builder.ins().return_(&[v]);
                        block_terminated = true;
                    }

                    // Self-recursion - call the JIT function
                    Instruction::CallSelf(dst, arg_regs) => {
                        // For arity=1, we have one argument
                        let args: Vec<CraneliftValue> = arg_regs.iter()
                            .map(|&r| builder.use_var(regs[r as usize]))
                            .collect();
                        let call = builder.ins().call(self_func_ref, &args);
                        let result = builder.inst_results(call)[0];
                        builder.def_var(regs[*dst as usize], result);
                    }

                    Instruction::TailCallSelf(arg_regs) => {
                        // For tail calls, we can use return_call for proper TCO
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

            // Seal all blocks
            builder.seal_all_blocks();
            builder.finalize();
        }

        // Compile to native code
        self.module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| JitError::Module(e.to_string()))?;

        // Finalize and get code pointer
        self.module.finalize_definitions()
            .map_err(|e| JitError::Module(e.to_string()))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // Cache the compiled function
        self.int_cache.insert(func_index, CompiledIntFunction {
            code_ptr,
            func_id,
            arity: func.arity,
        });

        // Clear context for next function
        self.module.clear_context(&mut self.ctx);

        Ok(())
    }

    /// Extract integer constant from the constant pool (standalone function to avoid borrow issues)
    fn get_int_const_from(constants: &[Value], idx: ConstIdx) -> Result<i64, JitError> {
        match constants.get(idx as usize) {
            Some(Value::Int(i)) => Ok(*i),
            Some(Value::Bool(b)) => Ok(if *b { 1 } else { 0 }),
            Some(other) => Err(JitError::NotSuitable(format!("non-int constant: {:?}", other))),
            None => Err(JitError::NotSuitable(format!("constant {} not found", idx))),
        }
    }

    /// Get JIT statistics
    pub fn stats(&self) -> JitStats {
        JitStats {
            compiled_functions: self.int_cache.len(),
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
        chunk.constants.push(Value::Int(1));
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
        chunk.constants.push(Value::Int(1)); // idx 0: constant 1
        chunk.constants.push(Value::Int(2)); // idx 1: constant 2

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
