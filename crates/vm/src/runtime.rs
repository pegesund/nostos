//! Runtime for concurrent process execution.
//!
//! The Runtime provides:
//! - Multi-process execution with scheduling
//! - Message passing between processes
//! - Shared globals, functions, and native functions
//!
//! JIT compatibility: The runtime coordinates process execution
//! without caring whether processes run interpreted or JIT code.

use std::rc::Rc;
use std::sync::Arc;

/// JIT compilation threshold - compile after this many calls
pub const JIT_THRESHOLD: u32 = 1000;

/// JIT-compiled integer function: fn(i64) -> i64 (single argument)
pub type JitIntFn = fn(i64) -> i64;

/// JIT-compiled integer functions with different arities
pub type JitIntFn0 = fn() -> i64;
pub type JitIntFn2 = fn(i64, i64) -> i64;
pub type JitIntFn3 = fn(i64, i64, i64) -> i64;
pub type JitIntFn4 = fn(i64, i64, i64, i64) -> i64;

/// JIT-compiled loop array function: fn(arr_ptr, arr_len) -> i64
pub type JitLoopArrayFn = fn(*const i64, i64) -> i64;

use crate::gc::{GcNativeFn, GcValue, Heap};
use crate::process::ExitReason;
use crate::scheduler::Scheduler;
use crate::value::TypeValue;
use crate::value::{FunctionValue, Instruction, Pid, RuntimeError, Value};
use crate::process::{CallFrame, format_stack_trace, format_stack_trace_debug};

/// Result of running a single process step.
pub enum ProcessStepResult {
    /// Continue execution.
    Continue,
    /// Process completed with a value.
    Finished(GcValue),
    /// Process should yield (out of reductions).
    Yield,
    /// Process is waiting for a message.
    Waiting,
    /// Process encountered an error.
    Error(RuntimeError),
}

/// Concurrent runtime for Nostos.
///
/// Manages multiple lightweight processes with message passing.
/// JIT-compatible: doesn't care how processes execute their code.
pub struct Runtime {
    /// Process scheduler.
    pub scheduler: Scheduler,

    /// Output buffer (combined from all processes).
    pub output: Vec<String>,

    /// Cached functions map - refreshed at start of run().
    /// Avoids RwLock overhead on every function call.
    cached_functions: std::collections::HashMap<String, Arc<FunctionValue>>,
    /// Cached function list for CallDirect - indexed access, no HashMap lookup!
    cached_function_list: Vec<Arc<FunctionValue>>,
    cached_natives: std::collections::HashMap<String, Arc<GcNativeFn>>,
    cached_types: std::collections::HashMap<String, Arc<TypeValue>>,
    /// JIT-compiled integer functions (func_index → native fn) - arity 1
    jit_int_functions: std::collections::HashMap<u16, JitIntFn>,
    /// JIT-compiled integer functions with arity 0
    jit_int_functions_0: std::collections::HashMap<u16, JitIntFn0>,
    /// JIT-compiled integer functions with arity 2
    jit_int_functions_2: std::collections::HashMap<u16, JitIntFn2>,
    /// JIT-compiled integer functions with arity 3
    jit_int_functions_3: std::collections::HashMap<u16, JitIntFn3>,
    /// JIT-compiled integer functions with arity 4
    jit_int_functions_4: std::collections::HashMap<u16, JitIntFn4>,
    /// JIT-compiled loop array functions (func_index → native fn)
    jit_loop_array_functions: std::collections::HashMap<u16, JitLoopArrayFn>,
    /// Debug mode: show local variable values in stack traces
    debug_mode: bool,
}

impl Runtime {
    /// Create a new runtime.
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            output: Vec::new(),
            cached_functions: std::collections::HashMap::new(),
            cached_function_list: Vec::new(),
            cached_natives: std::collections::HashMap::new(),
            cached_types: std::collections::HashMap::new(),
            jit_int_functions: std::collections::HashMap::new(),
            jit_int_functions_0: std::collections::HashMap::new(),
            jit_int_functions_2: std::collections::HashMap::new(),
            jit_int_functions_3: std::collections::HashMap::new(),
            jit_int_functions_4: std::collections::HashMap::new(),
            jit_loop_array_functions: std::collections::HashMap::new(),
            debug_mode: false,
        }
    }

    /// Enable or disable debug mode (shows local variables in stack traces).
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }

    /// Register a JIT-compiled integer function (arity 1).
    pub fn register_jit_int_function(&mut self, func_index: u16, jit_fn: JitIntFn) {
        self.jit_int_functions.insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 0).
    pub fn register_jit_int_function_0(&mut self, func_index: u16, jit_fn: JitIntFn0) {
        self.jit_int_functions_0.insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 2).
    pub fn register_jit_int_function_2(&mut self, func_index: u16, jit_fn: JitIntFn2) {
        self.jit_int_functions_2.insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 3).
    pub fn register_jit_int_function_3(&mut self, func_index: u16, jit_fn: JitIntFn3) {
        self.jit_int_functions_3.insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled integer function (arity 4).
    pub fn register_jit_int_function_4(&mut self, func_index: u16, jit_fn: JitIntFn4) {
        self.jit_int_functions_4.insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled loop array function.
    pub fn register_jit_loop_array_function(&mut self, func_index: u16, jit_fn: JitLoopArrayFn) {
        self.jit_loop_array_functions.insert(func_index, jit_fn);
    }

    /// Get a JIT-compiled loop array function if available.
    pub fn get_jit_loop_array_function(&self, func_index: u16) -> Option<JitLoopArrayFn> {
        self.jit_loop_array_functions.get(&func_index).copied()
    }

    /// Register a global function.
    pub fn register_function(&mut self, name: &str, func: Arc<FunctionValue>) {
        self.scheduler.functions.write().insert(name.to_string(), func);
    }

    /// Set the function list for direct indexed calls (CallDirect - no HashMap lookup!).
    pub fn set_function_list(&mut self, functions: Vec<Arc<FunctionValue>>) {
        self.cached_function_list = functions;
    }

    /// Register a type definition.
    pub fn register_type(&mut self, name: &str, type_val: Arc<TypeValue>) {
        self.scheduler.types.write().insert(name.to_string(), type_val);
    }

    /// Register a native function.
    pub fn register_native<F>(&mut self, name: &str, arity: usize, func: F)
    where
        F: Fn(&[GcValue], &mut Heap) -> Result<GcValue, RuntimeError> + Send + Sync + 'static,
    {
        let native = Arc::new(GcNativeFn {
            name: name.to_string(),
            arity,
            func: Box::new(func),
        });
        self.scheduler.natives.write().insert(name.to_string(), native);
    }

    /// Register built-in functions for all processes.
    /// Register builtins that need trait override support.
    /// Most builtins are now compiled to direct instructions (no string lookup, no HashMap).
    /// Only `show` and `copy` remain here because they can be overridden via Show/Copy traits.
    pub fn register_builtins(&mut self) {
        // Display - converts value to string (can be overridden via Show trait)
        self.register_native("show", 1, |args, heap| {
            let s = heap.display_value(&args[0]);
            Ok(GcValue::String(heap.alloc_string(s)))
        });

        // Copy - creates a deep copy of a value (can be overridden via Copy trait)
        self.register_native("copy", 1, |args, heap| {
            Ok(heap.clone_value(&args[0]))
        });
    }

    /// Spawn the initial process and run a function.
    pub fn spawn_initial(&mut self, func: Arc<FunctionValue>) -> Pid {
        let pid = self.scheduler.spawn();
        let reg_count = func.code.register_count;

        self.scheduler.with_process_mut(pid, |process| {
            let frame = CallFrame {
                function: func.clone(),
                ip: 0,
                registers: vec![GcValue::Unit; reg_count],
                captures: Vec::new(),
                return_reg: None,
            };
            process.frames.push(frame);
        });

        pid
    }

    /// Spawn a new process running a function with arguments and captures.
    /// Uses lightweight heap for memory-efficient mass spawning.
    pub fn spawn_process(
        &mut self,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
        parent_pid: Pid,
    ) -> Pid {
        let child_pid = self.scheduler.spawn_child();

        // Get both process handles - use lock ordering to prevent deadlock
        let (first_pid, second_pid, parent_is_first) = if parent_pid.0 < child_pid.0 {
            (parent_pid, child_pid, true)
        } else {
            (child_pid, parent_pid, false)
        };

        let first_handle = self.scheduler.get_process_handle(first_pid);
        let second_handle = self.scheduler.get_process_handle(second_pid);

        if let (Some(first), Some(second)) = (first_handle, second_handle) {
            let mut first_lock = first.lock();
            let mut second_lock = second.lock();

            let (parent, child) = if parent_is_first {
                (&*first_lock, &mut *second_lock)
            } else {
                (&*second_lock, &mut *first_lock)
            };

            // Deep copy arguments from parent heap to child heap
            let copied_args: Vec<GcValue> = args
                .iter()
                .map(|arg| child.heap.deep_copy(arg, &parent.heap))
                .collect();

            // Deep copy captures from parent heap to child heap
            let copied_captures: Vec<GcValue> = captures
                .iter()
                .map(|cap| child.heap.deep_copy(cap, &parent.heap))
                .collect();

            // Set up call frame with arguments in registers
            let reg_count = func.code.register_count;
            let mut registers = vec![GcValue::Unit; reg_count];
            for (i, arg) in copied_args.into_iter().enumerate() {
                if i < reg_count {
                    registers[i] = arg;
                }
            }

            let frame = CallFrame {
                function: func,
                ip: 0,
                registers,
                captures: copied_captures,
                return_reg: None,
            };
            child.frames.push(frame);
        }

        child_pid
    }

    /// Display a GcValue result using the main process's heap.
    /// Call this before the runtime is dropped to get a human-readable result.
    pub fn display_result(&self, value: &GcValue) -> String {
        // Try to get the main process's heap for display
        if let Some(handle) = self.scheduler.get_process_handle(Pid(1)) {
            let proc = handle.lock();
            proc.heap.display_value(value)
        } else {
            // Fallback to Debug format if process is gone
            format!("{:?}", value)
        }
    }

    /// Run all processes until completion or deadlock.
    pub fn run(&mut self) -> Result<Option<GcValue>, RuntimeError> {
        self.register_builtins();

        // Cache the function/native/type maps ONCE after registration
        // This avoids RwLock overhead on every function call (~30M for fib(35))
        self.cached_functions = self.scheduler.functions.read().clone();
        self.cached_natives = self.scheduler.natives.read().clone();
        self.cached_types = self.scheduler.types.read().clone();

        // Run until no more processes or all are waiting
        while self.scheduler.has_processes() {
            let Some(pid) = self.scheduler.schedule_next() else {
                // No runnable processes - check for deadlock
                if self.scheduler.process_count() > 0 {
                    return Err(RuntimeError::Panic("Deadlock: all processes waiting".to_string()));
                }
                break;
            };

            // Run the process for its time slice
            match self.run_process(pid)? {
                ProcessStepResult::Finished(value) => {
                    // First process to finish returns its value
                    if pid == Pid(1) {
                        return Ok(Some(value));
                    }
                    // Other processes just exit normally
                    self.scheduler.process_exit(pid, ExitReason::Normal, Some(value));
                }
                ProcessStepResult::Error(err) => {
                    self.scheduler.process_exit(
                        pid,
                        ExitReason::Error(format!("{:?}", err)),
                        None,
                    );
                    // If it's the main process, propagate error
                    if pid == Pid(1) {
                        return Err(err);
                    }
                }
                ProcessStepResult::Yield => {
                    // Process yielded, will be rescheduled
                }
                ProcessStepResult::Waiting => {
                    // Process is waiting for message
                }
                ProcessStepResult::Continue => {
                    // Should not happen at this level
                }
            }
        }

        Ok(None)
    }

    /// Run all processes and return result as a Value (for test compatibility).
    /// This converts the GcValue result to a Value using the initial process's heap.
    pub fn run_to_value(&mut self) -> Result<Option<Value>, RuntimeError> {
        self.register_builtins();

        // Cache the function/native/type maps ONCE after registration
        self.cached_functions = self.scheduler.functions.read().clone();
        self.cached_natives = self.scheduler.natives.read().clone();
        self.cached_types = self.scheduler.types.read().clone();

        // Run until no more processes or all are waiting
        while self.scheduler.has_processes() {
            let Some(pid) = self.scheduler.schedule_next() else {
                if self.scheduler.process_count() > 0 {
                    return Err(RuntimeError::Panic("Deadlock: all processes waiting".to_string()));
                }
                break;
            };

            match self.run_process(pid)? {
                ProcessStepResult::Finished(value) => {
                    if pid == Pid(1) {
                        // Convert GcValue to Value using the process's heap
                        let process_handle = self.scheduler.get_process_handle(pid)
                            .ok_or_else(|| RuntimeError::Panic("Initial process not found".to_string()))?;
                        let proc = process_handle.lock();
                        let result_value = proc.heap.gc_to_value(&value);
                        return Ok(Some(result_value));
                    }
                    self.scheduler.process_exit(pid, ExitReason::Normal, Some(value));
                }
                ProcessStepResult::Error(err) => {
                    self.scheduler.process_exit(
                        pid,
                        ExitReason::Error(format!("{:?}", err)),
                        None,
                    );
                    if pid == Pid(1) {
                        return Err(err);
                    }
                }
                ProcessStepResult::Yield => {}
                ProcessStepResult::Waiting => {}
                ProcessStepResult::Continue => {}
            }
        }

        Ok(None)
    }

    /// Run a single process until it yields, finishes, or errors.
    ///
    /// OPTIMIZED: Single-threaded Runtime - hold the lock for the ENTIRE time slice.
    /// In Erlang model, a process is owned by one thread at a time, so no per-instruction locking!
    fn run_process(&mut self, pid: Pid) -> Result<ProcessStepResult, RuntimeError> {
        // Get the process handle once and lock for entire execution
        let process_handle = self.scheduler.get_process_handle(pid)
            .ok_or_else(|| RuntimeError::Panic(format!("Process {:?} not found", pid)))?;

        // Clone cached maps into local variables to avoid borrow conflicts with &mut self
        // The cache is already cloned once at start of run(), so this is O(n) Rc bumps
        let functions = self.cached_functions.clone();
        let function_list = self.cached_function_list.clone();  // Clone to avoid borrow issues
        let natives = self.cached_natives.clone();
        let types = self.cached_types.clone();
        let jit_int_functions = self.jit_int_functions.clone(); // JIT function cache
        let jit_loop_array_functions = self.jit_loop_array_functions.clone(); // Loop array JIT cache

        // SINGLE LOCK for entire time slice - Erlang style!
        let mut proc = process_handle.lock();

        // Yield check counter - only check every 1000 instructions for performance
        let mut yield_check_counter = 0u32;

        loop {
            // Only check yield every 1000 instructions (significant performance win)
            yield_check_counter += 1;
            if yield_check_counter >= 1000 {
                yield_check_counter = 0;
                if proc.should_yield() {
                    proc.reset_reductions();
                    if self.scheduler.active_count() > 1 {
                        return Ok(ProcessStepResult::Yield);
                    }
                }
            }

            // Get frame count once - avoid repeated .len() calls
            let frame_len = proc.frames.len();
            if frame_len == 0 {
                return Ok(ProcessStepResult::Finished(GcValue::Unit));
            }
            let frame_idx = frame_len - 1;

            // Get frame data using raw pointers to avoid borrow issues
            // SAFETY: frame_idx is valid (frame_len > 0 checked above)
            let (ip, code_ptr, code_len, constants_ptr, constants_len) = unsafe {
                let frame = proc.frames.get_unchecked(frame_idx);
                let code = &frame.function.code;
                (
                    frame.ip,
                    code.code.as_ptr(),
                    code.code.len(),
                    code.constants.as_ptr(),
                    code.constants.len(),
                )
            };

            if ip >= code_len {
                return Ok(ProcessStepResult::Finished(GcValue::Unit));
            }

            // SAFETY: ip < code_len checked above
            let instr = unsafe { &*code_ptr.add(ip) };

            // Check if this instruction needs IPC (rare)
            let needs_ipc = matches!(
                instr,
                Instruction::Spawn(_, _, _)
                | Instruction::SpawnLink(_, _, _)
                | Instruction::SpawnMonitor(_, _, _, _)
                | Instruction::Send(_, _)
            );

            if needs_ipc {
                let instr_clone = instr.clone();
                let constants = unsafe { std::slice::from_raw_parts(constants_ptr, constants_len) }.to_vec();
                let constants_values: Vec<Value> = constants.iter().cloned().collect();
                proc.frames[frame_idx].ip += 1;

                // Capture stack trace before dropping the lock (in case of error)
                let stack_trace = if self.debug_mode {
                    format_stack_trace_debug(&proc.frames)
                } else {
                    format_stack_trace(&proc.frames)
                };
                drop(proc);
                match self.execute_ipc_instruction(pid, instr_clone, &constants_values) {
                    Ok(ProcessStepResult::Continue) => {
                        proc = process_handle.lock();
                        continue;
                    }
                    Ok(other) => return Ok(other),
                    Err(err) => return Err(err.with_stack_trace(stack_trace)),
                }
            }

            // Fast path: increment IP
            proc.frames[frame_idx].ip += 1;

            // SAFETY: ip < code_len was checked above, constants are immutable
            let constants = unsafe { std::slice::from_raw_parts(constants_ptr, constants_len) };

            match self.execute_local_instruction_inline(&mut proc, frame_idx, instr, constants, &functions, &function_list, &natives, &types, &jit_int_functions, &jit_loop_array_functions) {
                Ok(ProcessStepResult::Continue) => continue,
                Ok(other) => return Ok(other),
                Err(err) => {
                    // Capture stack trace before returning error
                    let stack_trace = if self.debug_mode {
                        format_stack_trace_debug(&proc.frames)
                    } else {
                        format_stack_trace(&proc.frames)
                    };
                    return Err(err.with_stack_trace(stack_trace));
                }
            }
        }
    }

    /// Execute an instruction that only accesses local process state.
    /// Takes borrowed instruction/constants - NO cloning!
    /// Called with the process lock already held - NO re-locking!
    /// Uses cached functions/natives/types maps to avoid RwLock on every call.
    #[inline(always)]
    fn execute_local_instruction_inline(
        &mut self,
        proc: &mut parking_lot::MutexGuard<'_, crate::process::Process>,
        frame_idx: usize,
        instr: &Instruction,
        constants: &[Value],
        functions: &std::collections::HashMap<String, Arc<FunctionValue>>,
        function_list: &[Arc<FunctionValue>],
        natives: &std::collections::HashMap<String, Arc<GcNativeFn>>,
        types: &std::collections::HashMap<String, Arc<TypeValue>>,
        jit_int_functions: &std::collections::HashMap<u16, JitIntFn>,
        jit_loop_array_functions: &std::collections::HashMap<u16, JitLoopArrayFn>,
    ) -> Result<ProcessStepResult, RuntimeError> {
        // Direct register access - no locking, frame_idx already computed!
        macro_rules! reg {
            ($r:expr) => {
                &proc.frames[frame_idx].registers[$r as usize]
            };
        }

        macro_rules! reg_clone {
            ($r:expr) => {
                proc.frames[frame_idx].registers[$r as usize].clone()
            };
        }

        macro_rules! set_reg {
            ($r:expr, $v:expr) => {
                proc.frames[frame_idx].registers[$r as usize] = $v
            };
        }

        // Match on reference - use * to deref Copy fields
        // This avoids cloning for most hot-path instructions
        match instr {
            // === Constants and moves ===
            Instruction::LoadConst(dst, idx) => {
                let value = constants.get(*idx as usize)
                    .ok_or_else(|| RuntimeError::Panic("Constant not found".to_string()))?;
                let gc_value = proc.heap.value_to_gc(value);
                set_reg!(*dst, gc_value);
            }

            Instruction::Move(dst, src) => {
                let val = reg_clone!(*src);
                set_reg!(*dst, val);
            }

            Instruction::LoadUnit(dst) => set_reg!(*dst, GcValue::Unit),
            Instruction::LoadTrue(dst) => set_reg!(*dst, GcValue::Bool(true)),
            Instruction::LoadFalse(dst) => set_reg!(*dst, GcValue::Bool(false)),

            // === Arithmetic (polymorphic - dispatches based on operand types) ===
            Instruction::AddInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x + y),
                    (GcValue::Int32(x), GcValue::Int32(y)) => GcValue::Int32(x.wrapping_add(*y)),
                    (GcValue::Int16(x), GcValue::Int16(y)) => GcValue::Int16(x.wrapping_add(*y)),
                    (GcValue::Int8(x), GcValue::Int8(y)) => GcValue::Int8(x.wrapping_add(*y)),
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => GcValue::UInt64(x.wrapping_add(*y)),
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => GcValue::UInt32(x.wrapping_add(*y)),
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => GcValue::UInt16(x.wrapping_add(*y)),
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => GcValue::UInt8(x.wrapping_add(*y)),
                    // Handle floats (type may not be known at compile time for pattern bindings)
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x + y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x + y),
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let bx = proc.heap.get_bigint(*x).unwrap();
                        let by = proc.heap.get_bigint(*y).unwrap();
                        let result = bx.value.clone() + &by.value;
                        let ptr = proc.heap.alloc_bigint(result);
                        GcValue::BigInt(ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x + *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in addition".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::SubInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x - y),
                    (GcValue::Int32(x), GcValue::Int32(y)) => GcValue::Int32(x.wrapping_sub(*y)),
                    (GcValue::Int16(x), GcValue::Int16(y)) => GcValue::Int16(x.wrapping_sub(*y)),
                    (GcValue::Int8(x), GcValue::Int8(y)) => GcValue::Int8(x.wrapping_sub(*y)),
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => GcValue::UInt64(x.wrapping_sub(*y)),
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => GcValue::UInt32(x.wrapping_sub(*y)),
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => GcValue::UInt16(x.wrapping_sub(*y)),
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => GcValue::UInt8(x.wrapping_sub(*y)),
                    // Handle floats (type may not be known at compile time for pattern bindings)
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x - y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x - y),
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let bx = proc.heap.get_bigint(*x).unwrap();
                        let by = proc.heap.get_bigint(*y).unwrap();
                        let result = bx.value.clone() - &by.value;
                        let ptr = proc.heap.alloc_bigint(result);
                        GcValue::BigInt(ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x - *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in subtraction".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::MulInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x * y),
                    (GcValue::Int32(x), GcValue::Int32(y)) => GcValue::Int32(x.wrapping_mul(*y)),
                    (GcValue::Int16(x), GcValue::Int16(y)) => GcValue::Int16(x.wrapping_mul(*y)),
                    (GcValue::Int8(x), GcValue::Int8(y)) => GcValue::Int8(x.wrapping_mul(*y)),
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => GcValue::UInt64(x.wrapping_mul(*y)),
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => GcValue::UInt32(x.wrapping_mul(*y)),
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => GcValue::UInt16(x.wrapping_mul(*y)),
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => GcValue::UInt8(x.wrapping_mul(*y)),
                    // Handle floats (type may not be known at compile time for pattern bindings)
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x * y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x * y),
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let bx = proc.heap.get_bigint(*x).unwrap();
                        let by = proc.heap.get_bigint(*y).unwrap();
                        let result = bx.value.clone() * &by.value;
                        let ptr = proc.heap.alloc_bigint(result);
                        GcValue::BigInt(ptr)
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => GcValue::Decimal(*x * *y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in multiplication".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            // === Comparison (polymorphic - dispatches based on operand types) ===
            Instruction::LtInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => *x < *y,
                    (GcValue::Int32(x), GcValue::Int32(y)) => *x < *y,
                    (GcValue::Int16(x), GcValue::Int16(y)) => *x < *y,
                    (GcValue::Int8(x), GcValue::Int8(y)) => *x < *y,
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => *x < *y,
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => *x < *y,
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => *x < *y,
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => *x < *y,
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let bx = proc.heap.get_bigint(*x).unwrap();
                        let by = proc.heap.get_bigint(*y).unwrap();
                        bx.value < by.value
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => *x < *y,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in comparison".to_string()
                    })
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            Instruction::LeInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => *x <= *y,
                    (GcValue::Int32(x), GcValue::Int32(y)) => *x <= *y,
                    (GcValue::Int16(x), GcValue::Int16(y)) => *x <= *y,
                    (GcValue::Int8(x), GcValue::Int8(y)) => *x <= *y,
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => *x <= *y,
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => *x <= *y,
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => *x <= *y,
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => *x <= *y,
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let bx = proc.heap.get_bigint(*x).unwrap();
                        let by = proc.heap.get_bigint(*y).unwrap();
                        bx.value <= by.value
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => *x <= *y,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in comparison".to_string()
                    })
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            Instruction::GtInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => *x > *y,
                    (GcValue::Int32(x), GcValue::Int32(y)) => *x > *y,
                    (GcValue::Int16(x), GcValue::Int16(y)) => *x > *y,
                    (GcValue::Int8(x), GcValue::Int8(y)) => *x > *y,
                    (GcValue::UInt64(x), GcValue::UInt64(y)) => *x > *y,
                    (GcValue::UInt32(x), GcValue::UInt32(y)) => *x > *y,
                    (GcValue::UInt16(x), GcValue::UInt16(y)) => *x > *y,
                    (GcValue::UInt8(x), GcValue::UInt8(y)) => *x > *y,
                    (GcValue::BigInt(x), GcValue::BigInt(y)) => {
                        let bx = proc.heap.get_bigint(*x).unwrap();
                        let by = proc.heap.get_bigint(*y).unwrap();
                        bx.value > by.value
                    }
                    (GcValue::Decimal(x), GcValue::Decimal(y)) => *x > *y,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in comparison".to_string()
                    })
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            Instruction::Eq(dst, a, b) => {
                let va = reg_clone!(*a);
                let vb = reg_clone!(*b);
                let eq = proc.heap.gc_values_equal(&va, &vb);
                set_reg!(*dst, GcValue::Bool(eq));
            }

            // === Logical (UNCHECKED - statically typed!) ===
            Instruction::Not(dst, src) => {
                // SAFETY: Type system guarantees this is Bool
                let b = match reg!(*src) {
                    GcValue::Bool(b) => *b,
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(!b));
            }

            // === Control flow (HOT PATH) ===
            Instruction::Jump(offset) => {
                let off = *offset;
                if let Some(frame) = proc.frames.last_mut() {
                    frame.ip = (frame.ip as isize + off as isize) as usize;
                }
                if off < 0 {
                    proc.consume_reductions(1);
                }
            }

            Instruction::JumpIfTrue(cond, offset) => {
                if reg!(*cond).is_truthy() {
                    let off = *offset;
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip = (frame.ip as isize + off as isize) as usize;
                    }
                    if off < 0 {
                        proc.consume_reductions(1);
                    }
                }
            }

            Instruction::JumpIfFalse(cond, offset) => {
                if !reg!(*cond).is_truthy() {
                    let off = *offset;
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip = (frame.ip as isize + off as isize) as usize;
                    }
                    if off < 0 {
                        proc.consume_reductions(1);
                    }
                }
            }

            // === Function calls ===
            Instruction::Return(src) => {
                let return_val = reg_clone!(*src);
                let return_reg = proc.frames.last().and_then(|f| f.return_reg);

                // Pop frame and return registers to pool
                if let Some(mut frame) = proc.frames.pop() {
                    let regs = std::mem::take(&mut frame.registers);
                    proc.free_registers(regs);
                }

                if proc.frames.is_empty() {
                    return Ok(ProcessStepResult::Finished(return_val));
                }

                if let Some(ret_reg) = return_reg {
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.registers[ret_reg as usize] = return_val;
                    }
                }
            }

            Instruction::Call(dst, func_reg, ref args) => {
                let dst = *dst;
                let arg_count = args.len();

                // Use stack array for args (avoid heap allocation)
                let mut arg_buf: [GcValue; 8] = Default::default();
                if arg_count <= 8 {
                    for (i, &r) in args.iter().enumerate() {
                        arg_buf[i] = reg_clone!(r);
                    }
                }

                let func_val = reg_clone!(*func_reg);
                let (func, captures) = match &func_val {
                    GcValue::Function(f) => (f.clone(), Vec::new()),
                    GcValue::Closure(ptr) => {
                        let closure = proc.heap.get_closure(*ptr)
                            .ok_or_else(|| RuntimeError::TypeError {
                                expected: "Closure".to_string(),
                                found: "invalid".to_string(),
                            })?;
                        (closure.function.clone(), closure.captures.clone())
                    }
                    other => {
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
                };

                // Get registers from pool
                let reg_count = func.code.register_count;
                let mut registers = proc.alloc_registers(reg_count);

                if arg_count <= 8 {
                    for i in 0..arg_count {
                        registers[i] = std::mem::take(&mut arg_buf[i]);
                    }
                } else {
                    for (i, &r) in args.iter().enumerate() {
                        if i < reg_count {
                            registers[i] = reg_clone!(r);
                        }
                    }
                }

                proc.frames.push(CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures,
                    return_reg: Some(dst),
                });
                proc.consume_reductions(1);
            }

            Instruction::TailCall(func_reg, ref args) => {
                let func_val = reg_clone!(*func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|&r| reg_clone!(r)).collect();

                let (func, captures) = match &func_val {
                    GcValue::Function(f) => (f.clone(), Vec::new()),
                    GcValue::Closure(ptr) => {
                        let closure = proc.heap.get_closure(*ptr)
                            .ok_or_else(|| RuntimeError::TypeError {
                                expected: "Closure".to_string(),
                                found: "invalid".to_string(),
                            })?;
                        (closure.function.clone(), closure.captures.clone())
                    }
                    other => {
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
                };

                // Replace current frame - reuse register vector
                if let Some(frame) = proc.frames.last_mut() {
                    let reg_count = func.code.register_count;
                    frame.function = func;
                    frame.ip = 0;
                    frame.registers.clear();
                    frame.registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            frame.registers[i] = arg;
                        }
                    }
                    frame.captures = captures;
                }
                proc.consume_reductions(1);
            }

            Instruction::CallNative(dst, name_idx, ref args) => {
                let dst = *dst;
                let arg_values: Vec<GcValue> = args.iter().map(|&r| reg_clone!(r)).collect();

                let name = match constants.get(*name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(RuntimeError::Panic("Invalid native function name".to_string())),
                };

                // Check for trait overrides for "show" and "copy"
                let trait_method = if !arg_values.is_empty() && (&*name == "show" || &*name == "copy") {
                    let trait_name = if &*name == "show" { "Show" } else { "Copy" };
                    let type_name = arg_values[0].type_name(&proc.heap).to_string();
                    let qualified_name = format!("{}.{}.{}", type_name, trait_name, name);
                    functions.get(&qualified_name).cloned()
                } else {
                    None
                };

                if let Some(func) = trait_method {
                    // Call the trait method instead of the native
                    let reg_count = func.code.register_count;
                    let mut registers = vec![GcValue::Unit; reg_count];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            registers[i] = arg;
                        }
                    }

                    let frame = CallFrame {
                        function: func,
                        ip: 0,
                        registers,
                        captures: Vec::new(),
                        return_reg: Some(dst),
                    };
                    proc.frames.push(frame);
                    proc.consume_reductions(1);
                } else {
                    // Fall back to the native function
                    let native = natives.get(&*name).cloned()
                        .ok_or_else(|| RuntimeError::Panic(format!("Undefined native: {}", name)))?;

                    let result = (native.func)(&arg_values, &mut proc.heap)?;
                    set_reg!(dst, result);
                    proc.consume_reductions(10);
                }
            }

            Instruction::CallByName(dst, name_idx, ref arg_regs) => {
                let dst = *dst;
                let arg_count = arg_regs.len();

                // Use stack array for args
                let mut arg_buf: [GcValue; 8] = Default::default();
                if arg_count <= 8 {
                    for (i, &r) in arg_regs.iter().enumerate() {
                        arg_buf[i] = reg_clone!(r);
                    }
                }

                let name = match constants.get(*name_idx as usize) {
                    Some(Value::String(s)) => s,
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let func = functions.get(&**name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Get registers from pool
                let reg_count = func.code.register_count;
                let mut registers = proc.alloc_registers(reg_count);

                if arg_count <= 8 {
                    for i in 0..arg_count {
                        registers[i] = std::mem::take(&mut arg_buf[i]);
                    }
                } else {
                    for (i, &r) in arg_regs.iter().enumerate() {
                        if i < reg_count {
                            registers[i] = reg_clone!(r);
                        }
                    }
                }

                proc.frames.push(CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(dst),
                });
                proc.consume_reductions(1);
            }

            Instruction::TailCallByName(name_idx, ref arg_regs) => {
                let name = match constants.get(*name_idx as usize) {
                    Some(Value::String(s)) => s,
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| reg_clone!(r)).collect();

                let func = functions.get(&**name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Replace current frame - reuse register vector like VM does
                if let Some(frame) = proc.frames.last_mut() {
                    let reg_count = func.code.register_count;
                    frame.function = func;
                    frame.ip = 0;
                    // Reuse existing allocation - clear and resize
                    frame.registers.clear();
                    frame.registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            frame.registers[i] = arg;
                        }
                    }
                    frame.captures.clear();
                }
                proc.consume_reductions(1);
            }

            Instruction::CallDirect(dst, func_idx, ref arg_regs) => {
                // Direct function call by index - no HashMap lookup!
                let dst = *dst;
                let arg_count = arg_regs.len();

                // Check if we have a JIT-compiled version (arity=1, int argument)
                if arg_count == 1 {
                    // First check for pure numeric JIT
                    if let Some(jit_fn) = jit_int_functions.get(func_idx) {
                        // Get the argument
                        let arg = reg_clone!(arg_regs[0]);
                        if let GcValue::Int64(n) = arg {
                            // Call JIT function directly!
                            let result = jit_fn(n);
                            set_reg!(dst, GcValue::Int64(result));
                            proc.consume_reductions(1);
                            return Ok(ProcessStepResult::Continue);
                        }
                    }
                    // Check for loop array JIT
                    if let Some(jit_fn) = jit_loop_array_functions.get(func_idx) {
                        let arg = reg_clone!(arg_regs[0]);
                        if let GcValue::Int64Array(arr_ptr) = arg {
                            // Get array data from heap (mutable for IndexSet support)
                            if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                let ptr = arr.items.as_mut_ptr();
                                let len = arr.items.len() as i64;
                                // Call JIT function with raw ptr and len!
                                // Safe: we have exclusive access through the mutable borrow
                                let result = jit_fn(ptr as *const i64, len);
                                set_reg!(dst, GcValue::Int64(result));
                                proc.consume_reductions(1);
                                return Ok(ProcessStepResult::Continue);
                            }
                        }
                    }
                }

                // Fall back to interpreted execution
                // Use stack array for args
                let mut arg_buf: [GcValue; 8] = Default::default();
                if arg_count <= 8 {
                    for (i, &r) in arg_regs.iter().enumerate() {
                        arg_buf[i] = reg_clone!(r);
                    }
                }

                let func = function_list.get(*func_idx as usize).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("function index {}", func_idx)))?;

                // Get registers from pool
                let reg_count = func.code.register_count;
                let mut registers = proc.alloc_registers(reg_count);

                if arg_count <= 8 {
                    for i in 0..arg_count {
                        registers[i] = std::mem::take(&mut arg_buf[i]);
                    }
                } else {
                    for (i, &r) in arg_regs.iter().enumerate() {
                        if i < reg_count {
                            registers[i] = reg_clone!(r);
                        }
                    }
                }

                proc.frames.push(CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(dst),
                });
                proc.consume_reductions(1);
            }

            Instruction::TailCallDirect(func_idx, ref arg_regs) => {
                // Check for JIT-compiled version (tail call with 1 arg)
                if arg_regs.len() == 1 {
                    // First check for pure numeric JIT
                    if let Some(jit_fn) = jit_int_functions.get(func_idx) {
                        let arg = reg_clone!(arg_regs[0]);
                        if let GcValue::Int64(n) = arg {
                            // Call JIT function directly!
                            let result = jit_fn(n);
                            // For tail call: pop current frame, set result in parent
                            let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                            proc.frames.pop();
                            if let Some(dst) = return_reg {
                                if let Some(parent) = proc.frames.last_mut() {
                                    parent.registers[dst as usize] = GcValue::Int64(result);
                                }
                            }
                            proc.consume_reductions(1);
                            return Ok(ProcessStepResult::Continue);
                        }
                    }
                    // Check for loop array JIT
                    if let Some(jit_fn) = jit_loop_array_functions.get(func_idx) {
                        let arg = reg_clone!(arg_regs[0]);
                        if let GcValue::Int64Array(arr_ptr) = arg {
                            if let Some(arr) = proc.heap.get_int64_array_mut(arr_ptr) {
                                let ptr = arr.items.as_mut_ptr();
                                let len = arr.items.len() as i64;
                                let result = jit_fn(ptr as *const i64, len);
                                // For tail call: pop current frame, set result in parent
                                let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                                proc.frames.pop();
                                if let Some(dst) = return_reg {
                                    if let Some(parent) = proc.frames.last_mut() {
                                        parent.registers[dst as usize] = GcValue::Int64(result);
                                    }
                                }
                                proc.consume_reductions(1);
                                return Ok(ProcessStepResult::Continue);
                            }
                        }
                    }
                }

                // Fall back to interpreted tail call
                let func = function_list.get(*func_idx as usize).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("function index {}", func_idx)))?;

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| reg_clone!(r)).collect();

                // Replace current frame - reuse register vector like VM does
                if let Some(frame) = proc.frames.last_mut() {
                    let reg_count = func.code.register_count;
                    frame.function = func;
                    frame.ip = 0;
                    // Reuse existing allocation - clear and resize
                    frame.registers.clear();
                    frame.registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            frame.registers[i] = arg;
                        }
                    }
                    frame.captures.clear();
                }
                proc.consume_reductions(1);
            }

            Instruction::CallSelf(dst, ref arg_regs) => {
                // Self-recursion: OPTIMIZED - use register pool!
                let dst = *dst;
                let arg_count = arg_regs.len();

                // Use stack array for args (avoid heap allocation for common cases)
                // Copy args BEFORE creating new frame to avoid borrow issues
                let mut arg_buf: [GcValue; 8] = Default::default();
                if arg_count <= 8 {
                    for (i, &r) in arg_regs.iter().enumerate() {
                        arg_buf[i] = reg_clone!(r);
                    }
                }

                // Get function info (Rc clone unavoidable due to borrow checker)
                let func = proc.frames[frame_idx].function.clone();
                let reg_count = func.code.register_count;

                // Get registers from pool (avoids allocation!)
                let mut registers = proc.alloc_registers(reg_count);

                // Copy args directly (no intermediate Vec for small arg counts)
                if arg_count <= 8 {
                    for i in 0..arg_count {
                        registers[i] = std::mem::take(&mut arg_buf[i]);
                    }
                } else {
                    // Fallback for rare >8 arg cases
                    for (i, &r) in arg_regs.iter().enumerate() {
                        if i < reg_count {
                            registers[i] = reg_clone!(r);
                        }
                    }
                }

                proc.frames.push(CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(dst),
                });
                proc.consume_reductions(1);
            }

            Instruction::TailCallSelf(ref arg_regs) => {
                // Self tail-recursion: reuse current frame's function (no HashMap lookup!)
                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| reg_clone!(r)).collect();

                // Get current function and reuse frame
                let func = proc.frames[frame_idx].function.clone();
                let reg_count = func.code.register_count;

                if let Some(frame) = proc.frames.last_mut() {
                    frame.ip = 0;
                    frame.registers.clear();
                    frame.registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            frame.registers[i] = arg;
                        }
                    }
                    frame.captures.clear();
                }
                proc.consume_reductions(1);
            }

            // === Collections (direct access) ===
            Instruction::MakeList(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| reg_clone!(r)).collect();
                let ptr = proc.heap.alloc_list(items);
                set_reg!(*dst, GcValue::List(ptr));
            }

            Instruction::MakeTuple(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| reg_clone!(r)).collect();
                let ptr = proc.heap.alloc_tuple(items);
                set_reg!(*dst, GcValue::Tuple(ptr));
            }

            Instruction::GetTupleField(dst, tuple_reg, idx) => {
                // SAFETY: Type system guarantees this is a Tuple
                let ptr = match reg!(*tuple_reg) {
                    GcValue::Tuple(ptr) => *ptr,
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                let item = proc.heap.get_tuple(ptr)
                    .and_then(|t| t.items.get(*idx as usize).cloned());
                match item {
                    Some(val) => set_reg!(*dst, val),
                    None => return Err(RuntimeError::IndexOutOfBounds {
                        index: *idx as i64,
                        length: 0,
                    }),
                }
            }

            Instruction::GetCapture(dst, idx) => {
                let val = proc.frames.last()
                    .and_then(|f| f.captures.get(*idx as usize))
                    .cloned()
                    .unwrap_or(GcValue::Unit);
                set_reg!(*dst, val);
            }

            Instruction::MakeClosure(dst, func_idx, ref capture_regs) => {
                let func = match constants.get(*func_idx as usize) {
                    Some(Value::Function(f)) => f.clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: "non-function".to_string(),
                    }),
                };

                // Collect captures from registers (direct access)
                let gc_captures: Vec<GcValue> = capture_regs.iter()
                    .map(|&r| reg_clone!(r))
                    .collect();
                let capture_names: Vec<String> = (0..gc_captures.len())
                    .map(|i| format!("capture_{}", i))
                    .collect();

                // Allocate closure on process heap (direct access)
                let closure_ptr = proc.heap.alloc_closure(func, gc_captures, capture_names);
                set_reg!(*dst, GcValue::Closure(closure_ptr));
            }

            // === Concurrency (SelfPid is local, others need IPC) ===
            Instruction::SelfPid(dst) => {
                set_reg!(*dst, GcValue::Pid(proc.pid.0));
            }

            // IPC instructions should NOT be handled here - they need special handling
            Instruction::Spawn(_, _, _)
            | Instruction::SpawnLink(_, _, _)
            | Instruction::SpawnMonitor(_, _, _, _)
            | Instruction::Send(_, _) => {
                return Err(RuntimeError::Panic("IPC instruction in local handler".to_string()));
            }

            Instruction::Receive(dst) => {
                if let Some(msg) = proc.try_receive() {
                    proc.frames[frame_idx].registers[*dst as usize] = msg;
                } else {
                    proc.wait_for_message();
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip -= 1;
                    }
                    return Ok(ProcessStepResult::Waiting);
                }
            }

            Instruction::ReceiveTimeout(_dst, _timeout_reg) => {
                return Err(RuntimeError::Panic("ReceiveTimeout not yet implemented in single-threaded runtime".to_string()));
            }

            Instruction::Sleep(_duration_reg) => {
                return Err(RuntimeError::Panic("Sleep not yet implemented in single-threaded runtime".to_string()));
            }

            // === Pattern matching (direct access) ===
            Instruction::TestConst(dst, value, const_idx) => {
                let constant = &constants[*const_idx as usize];
                let gc_const = proc.heap.value_to_gc(constant);
                let val = reg_clone!(*value);
                let result = proc.heap.gc_values_equal(&val, &gc_const);
                set_reg!(*dst, GcValue::Bool(result));
            }

            Instruction::TestNil(dst, list) => {
                let result = match reg!(*list) {
                    GcValue::List(ptr) => {
                        proc.heap.get_list(*ptr)
                            .map(|l| l.is_empty())
                            .unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            // === Records/Variants (direct access) ===
            Instruction::MakeRecord(dst, type_idx, ref field_regs) => {
                let type_name = match constants.get(*type_idx as usize) {
                    Some(Value::String(s)) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let fields: Vec<GcValue> = field_regs.iter()
                    .map(|&r| reg_clone!(r))
                    .collect();

                let type_info = types.get(&type_name).cloned();
                let field_names: Vec<String> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.name.clone()).collect())
                    .unwrap_or_else(|| (0..fields.len()).map(|i| format!("_{}", i)).collect());
                let mutable_fields: Vec<bool> = type_info
                    .as_ref()
                    .map(|t| t.fields.iter().map(|f| f.mutable).collect())
                    .unwrap_or_else(|| vec![false; fields.len()]);

                let record_ptr = proc.heap.alloc_record(type_name, field_names, fields, mutable_fields);
                set_reg!(*dst, GcValue::Record(record_ptr));
            }

            Instruction::TestTag(dst, value, tag_idx) => {
                let tag = match constants.get(*tag_idx as usize) {
                    Some(Value::String(s)) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let result = match reg!(*value) {
                    GcValue::Variant(ptr) => {
                        proc.heap.get_variant(*ptr)
                            .map(|v| v.constructor == tag)
                            .unwrap_or(false)
                    }
                    GcValue::Record(ptr) => {
                        proc.heap.get_record(*ptr)
                            .map(|r| r.type_name == tag)
                            .unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            Instruction::GetField(dst, record, field_idx) => {
                let field_name = match &constants[*field_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("Field name must be string".to_string())),
                };
                match reg!(*record) {
                    GcValue::Record(ptr) => {
                        let rec = proc.heap.get_record(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        let value = rec.fields[idx].clone();
                        set_reg!(*dst, value);
                    }
                    GcValue::Tuple(ptr) => {
                        // Support tuple field access with numeric indices (t.0, t.1, etc.)
                        let tuple = proc.heap.get_tuple(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".to_string()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid tuple index: {}", field_name)))?;
                        let value = tuple.items.get(idx).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Tuple index {} out of bounds", idx)))?;
                        set_reg!(*dst, value);
                    }
                    _ => return Err(RuntimeError::Panic("GetField expects record or tuple".to_string())),
                }
            }

            // === Division (polymorphic for pattern binding support) ===
            Instruction::DivInt(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => {
                        if *y == 0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Int64(x / y)
                    }
                    (GcValue::Float64(x), GcValue::Float64(y)) => {
                        if *y == 0.0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Float64(x / y)
                    }
                    (GcValue::Float32(x), GcValue::Float32(y)) => {
                        if *y == 0.0 { return Err(RuntimeError::DivisionByZero); }
                        GcValue::Float32(x / y)
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching numeric types".to_string(),
                        found: "mismatched types in division".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::ModInt(dst, a, b) => {
                // SAFETY: Type system guarantees these are Int
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                if y == 0 {
                    return Err(RuntimeError::Panic("Division by zero".to_string()));
                }
                set_reg!(*dst, GcValue::Int64(x % y));
            }

            Instruction::NegInt(dst, src) => {
                let v = match reg!(*src) {
                    GcValue::Int64(x) => *x,
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Int64(-v));
            }

            // === Float arithmetic (polymorphic) ===
            Instruction::AddFloat(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x + y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x + y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types in addition".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::SubFloat(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x - y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x - y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types in subtraction".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::MulFloat(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x * y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x * y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types in multiplication".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::DivFloat(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x / y),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x / y),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types in division".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::NegFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Float64(x) => GcValue::Float64(-x),
                    GcValue::Float32(x) => GcValue::Float32(-x),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "float".to_string(),
                        found: "non-float".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::PowFloat(dst, a, b) => {
                let result = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x.powf(*y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x.powf(*y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "matching float types".to_string(),
                        found: "mismatched types in power".to_string()
                    })
                };
                set_reg!(*dst, result);
            }

            // === More comparisons ===
            Instruction::EqInt(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x == y));
            }

            Instruction::NeInt(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x != y));
            }

            Instruction::GeInt(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x >= y));
            }

            Instruction::EqFloat(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x == y));
            }

            Instruction::LtFloat(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x < y));
            }

            Instruction::LeFloat(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x <= y));
            }

            Instruction::EqBool(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Bool(x), GcValue::Bool(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x == y));
            }

            Instruction::EqStr(dst, a, b) => {
                let (a_ptr, b_ptr) = match (reg!(*a), reg!(*b)) {
                    (GcValue::String(a), GcValue::String(b)) => (*a, *b),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                let a_str = proc.heap.get_string(a_ptr).map(|s| &s.data);
                let b_str = proc.heap.get_string(b_ptr).map(|s| &s.data);
                set_reg!(*dst, GcValue::Bool(a_str == b_str));
            }

            // === Logical ===
            Instruction::And(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Bool(x), GcValue::Bool(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x && y));
            }

            Instruction::Or(dst, a, b) => {
                let (x, y) = match (reg!(*a), reg!(*b)) {
                    (GcValue::Bool(x), GcValue::Bool(y)) => (*x, *y),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(x || y));
            }

            // === String ===
            Instruction::Concat(dst, a, b) => {
                let (a_ptr, b_ptr) = match (reg!(*a), reg!(*b)) {
                    (GcValue::String(a), GcValue::String(b)) => (*a, *b),
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                let a_str = proc.heap.get_string(a_ptr).map(|s| s.data.as_str()).unwrap_or("");
                let b_str = proc.heap.get_string(b_ptr).map(|s| s.data.as_str()).unwrap_or("");
                let result = format!("{}{}", a_str, b_str);
                let result_ptr = proc.heap.alloc_string(result);
                set_reg!(*dst, GcValue::String(result_ptr));
            }

            // === Collections ===
            Instruction::Cons(dst, head, tail) => {
                let head_val = reg_clone!(*head);
                let tail_ptr = match reg!(*tail) {
                    GcValue::List(ptr) => *ptr,
                    _ => return Err(RuntimeError::Panic("Cons expects list tail".to_string())),
                };
                let tail_items = proc.heap.get_list(tail_ptr)
                    .map(|l| l.items().to_vec())
                    .unwrap_or_default();
                let mut new_items = vec![head_val];
                new_items.extend(tail_items);
                let list_ptr = proc.heap.alloc_list(new_items);
                set_reg!(*dst, GcValue::List(list_ptr));
            }

            Instruction::Decons(head_dst, tail_dst, list) => {
                let list_ptr = match reg!(*list) {
                    GcValue::List(ptr) => *ptr,
                    _ => return Err(RuntimeError::Panic("Decons expects list".to_string())),
                };
                let items = proc.heap.get_list(list_ptr)
                    .map(|l| l.items().to_vec())
                    .unwrap_or_default();
                if items.is_empty() {
                    return Err(RuntimeError::Panic("Cannot decons empty list".to_string()));
                }
                let head = items[0].clone();
                let tail = items[1..].to_vec();
                let tail_ptr = proc.heap.alloc_list(tail);
                set_reg!(*head_dst, head);
                set_reg!(*tail_dst, GcValue::List(tail_ptr));
            }

            Instruction::ListConcat(dst, a, b) => {
                let a_ptr = match reg!(*a) {
                    GcValue::List(ptr) => *ptr,
                    _ => return Err(RuntimeError::Panic("ListConcat expects list".to_string())),
                };
                let b_ptr = match reg!(*b) {
                    GcValue::List(ptr) => *ptr,
                    _ => return Err(RuntimeError::Panic("ListConcat expects list".to_string())),
                };
                let a_items = proc.heap.get_list(a_ptr).map(|l| l.items().to_vec()).unwrap_or_default();
                let b_items = proc.heap.get_list(b_ptr).map(|l| l.items().to_vec()).unwrap_or_default();
                let mut new_items = a_items;
                new_items.extend(b_items);
                let list_ptr = proc.heap.alloc_list(new_items);
                set_reg!(*dst, GcValue::List(list_ptr));
            }

            Instruction::Index(dst, coll, idx) => {
                let idx_val = match reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                };
                let value = match reg!(*coll) {
                    GcValue::List(ptr) => {
                        let list = proc.heap.get_list(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid list reference".to_string()))?;
                        list.items().get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = proc.heap.get_tuple(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".to_string()))?;
                        tuple.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Array(ptr) => {
                        let array = proc.heap.get_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".to_string()))?;
                        array.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Int64Array(ptr) => {
                        let array = proc.heap.get_int64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Int64(val)
                    }
                    GcValue::Float64Array(ptr) => {
                        let array = proc.heap.get_float64_array(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".to_string()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Float64(val)
                    }
                    _ => return Err(RuntimeError::Panic("Index expects list, tuple, or array".to_string())),
                };
                set_reg!(*dst, value);
            }

            Instruction::IndexSet(coll, idx, val) => {
                let idx_val = match reg!(*idx) {
                    GcValue::Int64(i) => *i as usize,
                    _ => return Err(RuntimeError::Panic("Index must be integer".to_string())),
                };
                match reg!(*coll).clone() {
                    GcValue::Array(ptr) => {
                        let new_value = reg_clone!(*val);
                        let array = proc.heap.get_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Int64Array(ptr) => {
                        let new_value = match reg!(*val) {
                            GcValue::Int64(v) => *v,
                            _ => return Err(RuntimeError::Panic("Int64Array expects Int64 value".to_string())),
                        };
                        let array = proc.heap.get_int64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Float64Array(ptr) => {
                        let new_value = match reg!(*val) {
                            GcValue::Float64(v) => *v,
                            _ => return Err(RuntimeError::Panic("Float64Array expects Float64 value".to_string())),
                        };
                        let array = proc.heap.get_float64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".to_string()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    _ => return Err(RuntimeError::Panic("IndexSet expects array".to_string())),
                }
            }

            Instruction::Length(dst, src) => {
                let len = match reg!(*src) {
                    GcValue::List(ptr) => proc.heap.get_list(*ptr).map(|l| l.len()).unwrap_or(0),
                    GcValue::Tuple(ptr) => proc.heap.get_tuple(*ptr).map(|t| t.items.len()).unwrap_or(0),
                    GcValue::Array(ptr) => proc.heap.get_array(*ptr).map(|a| a.items.len()).unwrap_or(0),
                    GcValue::Int64Array(ptr) => proc.heap.get_int64_array(*ptr).map(|a| a.items.len()).unwrap_or(0),
                    GcValue::Float64Array(ptr) => proc.heap.get_float64_array(*ptr).map(|a| a.items.len()).unwrap_or(0),
                    GcValue::String(ptr) => proc.heap.get_string(*ptr).map(|s| s.data.len()).unwrap_or(0),
                    _ => return Err(RuntimeError::Panic("Length expects collection or string".to_string())),
                };
                set_reg!(*dst, GcValue::Int64(len as i64));
            }

            // === Typed Arrays ===
            Instruction::MakeInt64Array(dst, size_reg) => {
                let size = match reg!(*size_reg) {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::Panic("Array size must be Int64".to_string())),
                };
                let items = vec![0i64; size];
                let ptr = proc.heap.alloc_int64_array(items);
                set_reg!(*dst, GcValue::Int64Array(ptr));
            }

            Instruction::MakeFloat64Array(dst, size_reg) => {
                let size = match reg!(*size_reg) {
                    GcValue::Int64(n) => *n as usize,
                    _ => return Err(RuntimeError::Panic("Array size must be Int64".to_string())),
                };
                let items = vec![0.0f64; size];
                let ptr = proc.heap.alloc_float64_array(items);
                set_reg!(*dst, GcValue::Float64Array(ptr));
            }

            Instruction::MakeMap(dst, pairs) => {
                let mut map = std::collections::HashMap::new();
                for &(k, v) in pairs.iter() {
                    let key = reg!(k).to_gc_map_key()
                        .ok_or_else(|| RuntimeError::Panic("Map key must be hashable".to_string()))?;
                    let value = reg!(v).clone();
                    map.insert(key, value);
                }
                let map_ptr = proc.heap.alloc_map(map);
                set_reg!(*dst, GcValue::Map(map_ptr));
            }

            Instruction::MakeSet(dst, regs) => {
                let mut set = std::collections::HashSet::new();
                for &r in regs.iter() {
                    let key = reg!(r).to_gc_map_key()
                        .ok_or_else(|| RuntimeError::Panic("Set element must be hashable".to_string()))?;
                    set.insert(key);
                }
                let set_ptr = proc.heap.alloc_set(set);
                set_reg!(*dst, GcValue::Set(set_ptr));
            }

            // === Variants ===
            Instruction::MakeVariant(dst, type_idx, ctor_idx, field_regs) => {
                let type_name = match &constants[*type_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("Variant type must be string".to_string())),
                };
                let constructor = match &constants[*ctor_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("Variant constructor must be string".to_string())),
                };
                let fields: Vec<GcValue> = field_regs.iter().map(|r| reg!(*r).clone()).collect();
                let variant_ptr = proc.heap.alloc_variant(type_name, constructor, fields);
                set_reg!(*dst, GcValue::Variant(variant_ptr));
            }

            Instruction::GetTag(dst, src) => {
                let tag = match reg!(*src) {
                    GcValue::Variant(ptr) => {
                        proc.heap.get_variant(*ptr)
                            .map(|v| v.constructor.clone())
                            .unwrap_or_default()
                    }
                    _ => return Err(RuntimeError::Panic("GetTag expects variant".to_string())),
                };
                let tag_ptr = proc.heap.alloc_string(tag);
                set_reg!(*dst, GcValue::String(tag_ptr));
            }

            Instruction::GetVariantField(dst, src, idx) => {
                let value = match reg!(*src) {
                    GcValue::Variant(ptr) => {
                        let variant = proc.heap.get_variant(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        variant.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of bounds", idx)))?
                    }
                    // Also support Record types for unified handling
                    GcValue::Record(ptr) => {
                        let record = proc.heap.get_record(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        record.fields.get(*idx as usize).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Record field {} out of bounds", idx)))?
                    }
                    _ => return Err(RuntimeError::Panic("GetVariantField expects variant or record".to_string())),
                };
                set_reg!(*dst, value);
            }

            Instruction::GetVariantFieldByName(dst, src, name_idx) => {
                let field_name = match &constants[*name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("Field name must be string".to_string())),
                };
                let value = match reg!(*src) {
                    GcValue::Variant(ptr) => {
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid field index: {}", field_name)))?;
                        let variant = proc.heap.get_variant(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".to_string()))?;
                        variant.fields.get(idx).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of bounds", idx)))?
                    }
                    // Also support Record types for unified handling
                    GcValue::Record(ptr) => {
                        let record = proc.heap.get_record(*ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        let idx = record.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        record.fields.get(idx).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Record field {} out of bounds", idx)))?
                    }
                    _ => return Err(RuntimeError::Panic("GetVariantFieldByName expects variant or record".to_string())),
                };
                set_reg!(*dst, value);
            }

            Instruction::SetField(record, field_idx, value) => {
                let field_name = match &constants[*field_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("Field name must be string".to_string())),
                };
                let new_value = reg_clone!(*value);
                match reg!(*record).clone() {
                    GcValue::Record(ptr) => {
                        let rec = proc.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        if !rec.mutable_fields[idx] {
                            return Err(RuntimeError::Panic(format!("Field {} is immutable", field_name)));
                        }
                        // Clone needed values before mutably borrowing heap
                        let type_name = rec.type_name.clone();
                        let field_names = rec.field_names.clone();
                        let mutable_fields = rec.mutable_fields.clone();
                        let mut new_fields = rec.fields.clone();
                        new_fields[idx] = new_value;
                        let new_record_ptr = proc.heap.alloc_record(
                            type_name,
                            field_names,
                            new_fields,
                            mutable_fields,
                        );
                        set_reg!(*record, GcValue::Record(new_record_ptr));
                    }
                    _ => return Err(RuntimeError::Panic("SetField expects record".to_string())),
                }
            }

            Instruction::UpdateRecord(dst, src, type_idx, field_regs) => {
                let rec_ptr = match reg!(*src) {
                    GcValue::Record(ptr) => *ptr,
                    _ => return Err(RuntimeError::Panic("UpdateRecord expects record".to_string())),
                };
                let rec = proc.heap.get_record(rec_ptr)
                    .ok_or_else(|| RuntimeError::Panic("Invalid record reference".to_string()))?;
                // Clone needed values before mutably borrowing heap
                let type_name = match &constants[*type_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("Type name must be string".to_string())),
                };
                let field_names = rec.field_names.clone();
                let mutable_fields = rec.mutable_fields.clone();
                let mut new_fields = rec.fields.clone();
                // Update with new values from field_regs
                for (i, &r) in field_regs.iter().enumerate() {
                    if i < new_fields.len() {
                        new_fields[i] = reg!(r).clone();
                    }
                }
                let new_record_ptr = proc.heap.alloc_record(
                    type_name,
                    field_names,
                    new_fields,
                    mutable_fields,
                );
                set_reg!(*dst, GcValue::Record(new_record_ptr));
            }

            Instruction::TypeOf(dst, src) => {
                let type_name = reg!(*src).type_name(&proc.heap).to_string();
                let type_ptr = proc.heap.alloc_string(type_name);
                set_reg!(*dst, GcValue::String(type_ptr));
            }

            Instruction::TestUnit(dst, src) => {
                let is_unit = matches!(reg!(*src), GcValue::Unit);
                set_reg!(*dst, GcValue::Bool(is_unit));
            }

            // === Builtin math (compile-time resolved, no runtime dispatch!) ===
            Instruction::AbsInt(dst, src) => {
                let val = match reg!(*src) {
                    GcValue::Int64(i) => *i,
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Int64(val.abs()));
            }

            Instruction::AbsFloat(dst, src) => {
                let val = match reg!(*src) {
                    GcValue::Float64(f) => *f,
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Float64(val.abs()));
            }

            Instruction::SqrtFloat(dst, src) => {
                let val = match reg!(*src) {
                    GcValue::Float64(f) => *f,
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Float64(val.sqrt()));
            }

            // === Type conversions (compile-time resolved) ===
            Instruction::IntToFloat(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::Float64(*v as f64),
                    GcValue::Int16(v) => GcValue::Float64(*v as f64),
                    GcValue::Int32(v) => GcValue::Float64(*v as f64),
                    GcValue::Int64(v) => GcValue::Float64(*v as f64),
                    GcValue::UInt8(v) => GcValue::Float64(*v as f64),
                    GcValue::UInt16(v) => GcValue::Float64(*v as f64),
                    GcValue::UInt32(v) => GcValue::Float64(*v as f64),
                    GcValue::UInt64(v) => GcValue::Float64(*v as f64),
                    GcValue::Float32(v) => GcValue::Float64(*v as f64),
                    GcValue::Float64(v) => GcValue::Float64(*v),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Float64(bi.value.to_f64().unwrap_or(0.0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            Instruction::FloatToInt(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::Int64(*v as i64),
                    GcValue::Int16(v) => GcValue::Int64(*v as i64),
                    GcValue::Int32(v) => GcValue::Int64(*v as i64),
                    GcValue::Int64(v) => GcValue::Int64(*v),
                    GcValue::UInt8(v) => GcValue::Int64(*v as i64),
                    GcValue::UInt16(v) => GcValue::Int64(*v as i64),
                    GcValue::UInt32(v) => GcValue::Int64(*v as i64),
                    GcValue::UInt64(v) => GcValue::Int64(*v as i64),
                    GcValue::Float32(v) => GcValue::Int64(*v as i64),
                    GcValue::Float64(v) => GcValue::Int64(*v as i64),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Int64(bi.value.to_i64().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to Int8
            Instruction::ToInt8(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::Int8(*v),
                    GcValue::Int16(v) => GcValue::Int8(*v as i8),
                    GcValue::Int32(v) => GcValue::Int8(*v as i8),
                    GcValue::Int64(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt8(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt16(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt32(v) => GcValue::Int8(*v as i8),
                    GcValue::UInt64(v) => GcValue::Int8(*v as i8),
                    GcValue::Float32(v) => GcValue::Int8(*v as i8),
                    GcValue::Float64(v) => GcValue::Int8(*v as i8),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Int8(bi.value.to_i8().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to Int16
            Instruction::ToInt16(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::Int16(*v as i16),
                    GcValue::Int16(v) => GcValue::Int16(*v),
                    GcValue::Int32(v) => GcValue::Int16(*v as i16),
                    GcValue::Int64(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt8(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt16(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt32(v) => GcValue::Int16(*v as i16),
                    GcValue::UInt64(v) => GcValue::Int16(*v as i16),
                    GcValue::Float32(v) => GcValue::Int16(*v as i16),
                    GcValue::Float64(v) => GcValue::Int16(*v as i16),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Int16(bi.value.to_i16().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to Int32
            Instruction::ToInt32(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::Int32(*v as i32),
                    GcValue::Int16(v) => GcValue::Int32(*v as i32),
                    GcValue::Int32(v) => GcValue::Int32(*v),
                    GcValue::Int64(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt8(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt16(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt32(v) => GcValue::Int32(*v as i32),
                    GcValue::UInt64(v) => GcValue::Int32(*v as i32),
                    GcValue::Float32(v) => GcValue::Int32(*v as i32),
                    GcValue::Float64(v) => GcValue::Int32(*v as i32),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Int32(bi.value.to_i32().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to UInt8
            Instruction::ToUInt8(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::UInt8(*v as u8),
                    GcValue::Int16(v) => GcValue::UInt8(*v as u8),
                    GcValue::Int32(v) => GcValue::UInt8(*v as u8),
                    GcValue::Int64(v) => GcValue::UInt8(*v as u8),
                    GcValue::UInt8(v) => GcValue::UInt8(*v),
                    GcValue::UInt16(v) => GcValue::UInt8(*v as u8),
                    GcValue::UInt32(v) => GcValue::UInt8(*v as u8),
                    GcValue::UInt64(v) => GcValue::UInt8(*v as u8),
                    GcValue::Float32(v) => GcValue::UInt8(*v as u8),
                    GcValue::Float64(v) => GcValue::UInt8(*v as u8),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::UInt8(bi.value.to_u8().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to UInt16
            Instruction::ToUInt16(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::UInt16(*v as u16),
                    GcValue::Int16(v) => GcValue::UInt16(*v as u16),
                    GcValue::Int32(v) => GcValue::UInt16(*v as u16),
                    GcValue::Int64(v) => GcValue::UInt16(*v as u16),
                    GcValue::UInt8(v) => GcValue::UInt16(*v as u16),
                    GcValue::UInt16(v) => GcValue::UInt16(*v),
                    GcValue::UInt32(v) => GcValue::UInt16(*v as u16),
                    GcValue::UInt64(v) => GcValue::UInt16(*v as u16),
                    GcValue::Float32(v) => GcValue::UInt16(*v as u16),
                    GcValue::Float64(v) => GcValue::UInt16(*v as u16),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::UInt16(bi.value.to_u16().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to UInt32
            Instruction::ToUInt32(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::UInt32(*v as u32),
                    GcValue::Int16(v) => GcValue::UInt32(*v as u32),
                    GcValue::Int32(v) => GcValue::UInt32(*v as u32),
                    GcValue::Int64(v) => GcValue::UInt32(*v as u32),
                    GcValue::UInt8(v) => GcValue::UInt32(*v as u32),
                    GcValue::UInt16(v) => GcValue::UInt32(*v as u32),
                    GcValue::UInt32(v) => GcValue::UInt32(*v),
                    GcValue::UInt64(v) => GcValue::UInt32(*v as u32),
                    GcValue::Float32(v) => GcValue::UInt32(*v as u32),
                    GcValue::Float64(v) => GcValue::UInt32(*v as u32),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::UInt32(bi.value.to_u32().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to UInt64
            Instruction::ToUInt64(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::UInt64(*v as u64),
                    GcValue::Int16(v) => GcValue::UInt64(*v as u64),
                    GcValue::Int32(v) => GcValue::UInt64(*v as u64),
                    GcValue::Int64(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt8(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt16(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt32(v) => GcValue::UInt64(*v as u64),
                    GcValue::UInt64(v) => GcValue::UInt64(*v),
                    GcValue::Float32(v) => GcValue::UInt64(*v as u64),
                    GcValue::Float64(v) => GcValue::UInt64(*v as u64),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::UInt64(bi.value.to_u64().unwrap_or(0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to Float32
            Instruction::ToFloat32(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => GcValue::Float32(*v as f32),
                    GcValue::Int16(v) => GcValue::Float32(*v as f32),
                    GcValue::Int32(v) => GcValue::Float32(*v as f32),
                    GcValue::Int64(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt8(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt16(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt32(v) => GcValue::Float32(*v as f32),
                    GcValue::UInt64(v) => GcValue::Float32(*v as f32),
                    GcValue::Float32(v) => GcValue::Float32(*v),
                    GcValue::Float64(v) => GcValue::Float32(*v as f32),
                    GcValue::BigInt(ptr) => {
                        let bi = proc.heap.get_bigint(*ptr).unwrap();
                        use num_traits::ToPrimitive;
                        GcValue::Float32(bi.value.to_f32().unwrap_or(0.0))
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "numeric".to_string(),
                        found: "non-numeric".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // Conversion to BigInt
            Instruction::ToBigInt(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::Int8(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::Int16(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::Int32(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::Int64(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::UInt8(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::UInt16(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::UInt32(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::UInt64(v) => {
                        let bi = num_bigint::BigInt::from(*v);
                        GcValue::BigInt(proc.heap.alloc_bigint(bi))
                    }
                    GcValue::BigInt(ptr) => GcValue::BigInt(*ptr),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "integer".to_string(),
                        found: "non-integer".to_string(),
                    })
                };
                set_reg!(*dst, result);
            }

            // === List operations (compile-time resolved) ===
            Instruction::ListHead(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::List(ptr) => {
                        if let Some(list) = proc.heap.get_list(*ptr) {
                            if list.is_empty() {
                                return Err(RuntimeError::Panic("head: empty list".to_string()));
                            }
                            list.items()[0].clone()
                        } else {
                            return Err(RuntimeError::Panic("Invalid list pointer".to_string()));
                        }
                    }
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, result);
            }

            Instruction::ListTail(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::List(ptr) => {
                        if let Some(list) = proc.heap.get_list(*ptr) {
                            if list.is_empty() {
                                return Err(RuntimeError::Panic("tail: empty list".to_string()));
                            }
                            let tail_list = list.tail();
                            GcValue::List(proc.heap.alloc_list_tail(tail_list))
                        } else {
                            return Err(RuntimeError::Panic("Invalid list pointer".to_string()));
                        }
                    }
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, result);
            }

            Instruction::ListIsEmpty(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::List(ptr) => {
                        if let Some(list) = proc.heap.get_list(*ptr) {
                            list.is_empty()
                        } else {
                            true // Invalid pointer treated as empty
                        }
                    }
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Bool(result));
            }

            Instruction::ListSum(dst, src) => {
                let result = match reg!(*src) {
                    GcValue::List(ptr) => {
                        if let Some(list) = proc.heap.get_list(*ptr) {
                            let mut total: i64 = 0;
                            for item in list.items() {
                                match item {
                                    GcValue::Int64(n) => total += n,
                                    _ => panic!("listSum requires list of Int64"),
                                }
                            }
                            total
                        } else {
                            0
                        }
                    }
                    _ => unsafe { std::hint::unreachable_unchecked() }
                };
                set_reg!(*dst, GcValue::Int64(result));
            }

            // === IO/Debug builtins ===
            Instruction::Print(dst, src) => {
                let s = proc.heap.display_value(reg!(*src));
                println!("{}", s);
                let str_ptr = proc.heap.alloc_string(s.clone());
                proc.output.push(s);
                set_reg!(*dst, GcValue::String(str_ptr));
            }

            Instruction::Println(src) => {
                let s = proc.heap.display_value(reg!(*src));
                println!("{}", s);
                proc.output.push(s);
            }

            Instruction::Panic(src) => {
                let msg = proc.heap.display_value(reg!(*src));
                return Err(RuntimeError::Panic(msg));
            }

            // === Assertions ===
            Instruction::Assert(src) => {
                match reg!(*src) {
                    GcValue::Bool(true) => {}
                    GcValue::Bool(false) => {
                        return Err(RuntimeError::Panic("Assertion failed".to_string()));
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Bool".to_string(),
                            found: reg!(*src).type_name(&proc.heap).to_string(),
                        });
                    }
                }
            }

            Instruction::AssertEq(a, b) => {
                let va = reg!(*a);
                let vb = reg!(*b);
                if !proc.heap.gc_values_equal(va, vb) {
                    let sa = proc.heap.display_value(va);
                    let sb = proc.heap.display_value(vb);
                    return Err(RuntimeError::Panic(format!("Assertion failed: {} != {}", sa, sb)));
                }
            }

            // === Exception handling ===
            Instruction::PushHandler(catch_offset) => {
                use crate::process::ExceptionHandler;
                let frame_idx = proc.frames.len() - 1;
                let catch_ip = (proc.frames[frame_idx].ip as isize + *catch_offset as isize) as usize;
                let handler = ExceptionHandler {
                    frame_index: frame_idx,
                    catch_ip,
                };
                proc.handlers.push(handler);
            }

            Instruction::PopHandler => {
                proc.handlers.pop();
            }

            Instruction::Throw(src) => {
                let exception = reg_clone!(*src);
                proc.current_exception = Some(exception.clone());

                // Find a handler
                if let Some(handler) = proc.handlers.pop() {
                    // Unwind to the handler's frame
                    while proc.frames.len() > handler.frame_index + 1 {
                        proc.frames.pop();
                    }
                    // Jump to the catch block
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip = handler.catch_ip;
                    }
                } else {
                    // No handler - propagate as error
                    return Err(RuntimeError::Panic(format!("Uncaught exception: {:?}", exception)));
                }
            }

            Instruction::GetException(dst) => {
                let exception = proc.current_exception.clone().unwrap_or(GcValue::Unit);
                set_reg!(*dst, exception);
            }

            // === Debug ===
            Instruction::Nop => {}

            Instruction::DebugPrint(r) => {
                let value = reg_clone!(*r);
                println!("DEBUG: {:?}", value);
                proc.output.push(format!("{:?}", value));
            }
        }

        Ok(ProcessStepResult::Continue)
    }

    /// Execute IPC instructions that need special handling.
    /// Called after releasing the process lock to avoid deadlock.
    fn execute_ipc_instruction(
        &mut self,
        pid: Pid,
        instr: Instruction,
        _constants: &[Value],
    ) -> Result<ProcessStepResult, RuntimeError> {
        // Helper to read registers via scheduler (we don't hold the lock)
        macro_rules! get_reg {
            ($r:expr) => {{
                self.scheduler.with_process(pid, |proc| {
                    proc.frames.last()
                        .map(|f| f.registers[$r as usize].clone())
                        .unwrap_or(GcValue::Unit)
                }).unwrap_or(GcValue::Unit)
            }};
        }

        macro_rules! set_reg {
            ($r:expr, $v:expr) => {{
                self.scheduler.with_process_mut(pid, |proc| {
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.registers[$r as usize] = $v;
                    }
                });
            }};
        }

        match instr {
            Instruction::Spawn(dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let (func, captures) = self.scheduler.with_process(pid, |proc| {
                    match &func_val {
                        GcValue::Function(f) => Ok((f.clone(), Vec::new())),
                        GcValue::Closure(ptr) => {
                            proc.heap.get_closure(*ptr)
                                .map(|c| (c.function.clone(), c.captures.clone()))
                                .ok_or_else(|| RuntimeError::TypeError {
                                    expected: "Function".to_string(),
                                    found: "invalid closure".to_string(),
                                })
                        }
                        other => Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        }),
                    }
                }).ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                let child_pid = self.spawn_process(func, args, captures, pid);
                set_reg!(dst, GcValue::Pid(child_pid.0));

                self.scheduler.with_process_mut(pid, |proc| {
                    proc.consume_reductions(100);
                });
            }

            Instruction::SpawnLink(dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let (func, captures) = self.scheduler.with_process(pid, |proc| {
                    match &func_val {
                        GcValue::Function(f) => Ok((f.clone(), Vec::new())),
                        GcValue::Closure(ptr) => {
                            proc.heap.get_closure(*ptr)
                                .map(|c| (c.function.clone(), c.captures.clone()))
                                .ok_or_else(|| RuntimeError::TypeError {
                                    expected: "Function".to_string(),
                                    found: "invalid closure".to_string(),
                                })
                        }
                        other => Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        }),
                    }
                }).ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                let child_pid = self.spawn_process(func, args, captures, pid);

                // Create link with lock ordering
                let (first_pid, second_pid) = if pid.0 < child_pid.0 {
                    (pid, child_pid)
                } else {
                    (child_pid, pid)
                };

                let first_handle = self.scheduler.get_process_handle(first_pid);
                let second_handle = self.scheduler.get_process_handle(second_pid);

                if let (Some(first), Some(second)) = (first_handle, second_handle) {
                    let mut first_lock = first.lock();
                    let mut second_lock = second.lock();

                    if first_pid == pid {
                        first_lock.link(child_pid);
                        second_lock.link(pid);
                    } else {
                        first_lock.link(pid);
                        second_lock.link(child_pid);
                    }
                }

                set_reg!(dst, GcValue::Pid(child_pid.0));
            }

            Instruction::SpawnMonitor(pid_dst, ref_dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let (func, captures) = self.scheduler.with_process(pid, |proc| {
                    match &func_val {
                        GcValue::Function(f) => Ok((f.clone(), Vec::new())),
                        GcValue::Closure(ptr) => {
                            proc.heap.get_closure(*ptr)
                                .map(|c| (c.function.clone(), c.captures.clone()))
                                .ok_or_else(|| RuntimeError::TypeError {
                                    expected: "Function".to_string(),
                                    found: "invalid closure".to_string(),
                                })
                        }
                        other => Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        }),
                    }
                }).ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                let child_pid = self.spawn_process(func, args, captures, pid);
                let ref_id = self.scheduler.make_ref();

                // Set up monitor with lock ordering
                let (first_pid, second_pid) = if pid.0 < child_pid.0 {
                    (pid, child_pid)
                } else {
                    (child_pid, pid)
                };

                let first_handle = self.scheduler.get_process_handle(first_pid);
                let second_handle = self.scheduler.get_process_handle(second_pid);

                if let (Some(first), Some(second)) = (first_handle, second_handle) {
                    let mut first_lock = first.lock();
                    let mut second_lock = second.lock();

                    if first_pid == pid {
                        first_lock.add_monitor(ref_id, child_pid);
                        second_lock.add_monitored_by(ref_id, pid);
                    } else {
                        second_lock.add_monitor(ref_id, child_pid);
                        first_lock.add_monitored_by(ref_id, pid);
                    }
                }

                set_reg!(pid_dst, GcValue::Pid(child_pid.0));
                set_reg!(ref_dst, GcValue::Ref(ref_id.0));
            }

            Instruction::Send(pid_reg, msg_reg) => {
                let target_val = get_reg!(pid_reg);
                let message = get_reg!(msg_reg);

                let target_pid = match &target_val {
                    GcValue::Pid(p) => Pid(*p),
                    other => {
                        let type_name = self.scheduler.with_process(pid, |proc| {
                            other.type_name(&proc.heap).to_string()
                        }).unwrap_or_else(|| "unknown".to_string());
                        return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: type_name,
                        });
                    }
                };

                self.scheduler.send(pid, target_pid, message)?;

                self.scheduler.with_process_mut(pid, |proc| {
                    proc.consume_reductions(10);
                });
            }

            _ => {
                return Err(RuntimeError::Panic(format!(
                    "Non-IPC instruction {:?} in IPC handler",
                    instr
                )));
            }
        }

        Ok(ProcessStepResult::Continue)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Chunk;
    use std::sync::atomic::AtomicU32;

    fn make_function(name: &str, code: Vec<Instruction>) -> Arc<FunctionValue> {
        Arc::new(FunctionValue {
            name: name.to_string(),
            arity: 0,
            param_names: Vec::new(),
            code: Arc::new(Chunk {
                code,
                constants: Vec::new(),
                lines: Vec::new(),
                locals: Vec::new(),
                register_count: 256,
            }),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols: vec![],
        })
    }

    fn make_function_with_consts(name: &str, code: Vec<Instruction>, constants: Vec<Value>) -> Arc<FunctionValue> {
        Arc::new(FunctionValue {
            name: name.to_string(),
            arity: 0,
            param_names: Vec::new(),
            code: Arc::new(Chunk {
                code,
                constants,
                lines: Vec::new(),
                locals: Vec::new(),
                register_count: 256,
            }),
            module: None,
            source_span: None,
            jit_code: None,
            call_count: AtomicU32::new(0),
            debug_symbols: vec![],
        })
    }

    #[test]
    fn test_runtime_single_process() {
        let mut runtime = Runtime::new();

        // Simple function that returns 42 (using LoadConst since no LoadInt)
        let func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::Return(0),
            ],
            vec![Value::Int64(42)],
        );

        runtime.spawn_initial(func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(42)));
    }

    #[test]
    fn test_runtime_self_pid() {
        let mut runtime = Runtime::new();

        // Function that returns self()
        let func = make_function("main", vec![
            Instruction::SelfPid(0),
            Instruction::Return(0),
        ]);

        runtime.spawn_initial(func);
        let result = runtime.run().unwrap();

        // First spawned process should be pid 1
        assert_eq!(result, Some(GcValue::Pid(1)));
    }

    #[test]
    fn test_runtime_spawn() {
        let mut runtime = Runtime::new();

        // Child function that returns 99
        let child_func = make_function_with_consts(
            "child",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::Return(0),
            ],
            vec![Value::Int64(99)],
        );

        // Main function that spawns a child and returns child's pid
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0), // Load child function
                Instruction::Spawn(1, 0, vec![].into()), // Spawn child, store pid in r1
                Instruction::Return(1), // Return child pid
            ],
            vec![Value::Function(child_func)],
        );

        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        // Child should be pid 2
        assert_eq!(result, Some(GcValue::Pid(2)));
    }

    #[test]
    fn test_runtime_arithmetic() {
        let mut runtime = Runtime::new();

        // Function: 5 * 2 = 10
        let func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),  // r0 = 5
                Instruction::LoadConst(1, 1),  // r1 = 2
                Instruction::MulInt(2, 0, 1),  // r2 = 5 * 2 = 10
                Instruction::Return(2),
            ],
            vec![Value::Int64(5), Value::Int64(2)],
        );

        runtime.spawn_initial(func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(10)));
    }

    #[test]
    fn test_runtime_call_by_name() {
        let mut runtime = Runtime::new();

        // Helper function: add(a, b) = a + b
        // Takes two args in r0 and r1, returns their sum
        let add_func = make_function_with_consts(
            "add",
            vec![
                Instruction::AddInt(2, 0, 1),  // r2 = r0 + r1
                Instruction::Return(2),
            ],
            vec![],
        );
        // Update arity
        let add_func = Arc::new(FunctionValue {
            arity: 2,
            ..(*add_func).clone()
        });

        // main function: calls add(2, 3) and returns the result
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),        // r0 = 2
                Instruction::LoadConst(1, 1),        // r1 = 3
                Instruction::CallByName(2, 2, vec![0, 1].into()),  // r2 = add(r0, r1)
                Instruction::Return(2),
            ],
            vec![Value::Int64(2), Value::Int64(3), Value::String(Arc::new("add".to_string()))],
        );

        runtime.register_function("add", add_func);
        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(5)));
    }

    #[test]
    fn test_runtime_simple_recursive() {
        // Test: sum(n) = if n <= 0 then 0 else n + sum(n-1)
        // sum(3) = 3 + 2 + 1 + 0 = 6
        let mut runtime = Runtime::new();

        // sum function bytecode:
        // [0]: LoadConst(1, 0)       // r1 = 0
        // [1]: LeInt(2, 0, 1)        // r2 = (n <= 0)
        // [2]: JumpIfFalse(2, 2)     // if not (n <= 0), jump to [5]
        // [3]: Move(3, 1)            // base case: r3 = 0
        // [4]: Jump(5)               // skip to Return
        // [5]: LoadConst(4, 1)       // r4 = 1
        // [6]: SubInt(5, 0, 4)       // r5 = n - 1
        // [7]: CallByName(6, 2, [5]) // r6 = sum(n-1)
        // [8]: AddInt(3, 0, 6)       // r3 = n + r6
        // [9]: Return(3)

        let sum_func = make_function_with_consts(
            "sum",
            vec![
                Instruction::LoadConst(1, 0),        // r1 = 0
                Instruction::LeInt(2, 0, 1),         // r2 = (n <= 0)
                Instruction::JumpIfFalse(2, 2),      // if not, skip to [5]
                Instruction::Move(3, 1),             // r3 = 0
                Instruction::Jump(4),                // skip to Return at [9] (ip=5+4=9)
                Instruction::LoadConst(4, 1),        // r4 = 1
                Instruction::SubInt(5, 0, 4),        // r5 = n - 1
                Instruction::CallByName(6, 2, vec![5].into()), // r6 = sum(n-1)
                Instruction::AddInt(3, 0, 6),        // r3 = n + sum(n-1)
                Instruction::Return(3),
            ],
            vec![Value::Int64(0), Value::Int64(1), Value::String(Arc::new("sum".to_string()))],
        );
        let sum_func = Arc::new(FunctionValue {
            arity: 1,
            ..(*sum_func).clone()
        });

        // main: return sum(3)
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::CallByName(1, 1, vec![0].into()),
                Instruction::Return(1),
            ],
            vec![Value::Int64(3), Value::String(Arc::new("sum".to_string()))],
        );

        runtime.register_function("sum", sum_func);
        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(6)));
    }

    #[test]
    fn test_runtime_tail_call_by_name() {
        let mut runtime = Runtime::new();

        // Helper function: id(x) = x
        let id_func = make_function_with_consts(
            "id",
            vec![
                Instruction::Return(0),  // Return the first arg
            ],
            vec![],
        );
        let id_func = Arc::new(FunctionValue {
            arity: 1,
            ..(*id_func).clone()
        });

        // main function: tail calls id(42)
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),        // r0 = 42
                Instruction::TailCallByName(1, vec![0].into()),  // tail call id(42)
                Instruction::Return(0),  // This should never execute
            ],
            vec![Value::Int64(42), Value::String(Arc::new("id".to_string()))],
        );

        runtime.register_function("id", id_func);
        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(42)));
    }

    #[test]
    fn test_runtime_message_passing() {
        let mut runtime = Runtime::new();

        // Parent process: spawn child, receive its message
        // Child process: send message to parent, exit

        // Child function: send(parent_pid, 42)
        // The child receives parent pid as argument in r0
        let child_func = make_function_with_consts(
            "child",
            vec![
                // r0 = parent_pid (passed as argument)
                Instruction::LoadConst(1, 0),  // r1 = 42
                Instruction::Send(0, 1),       // send(parent_pid, 42)
                Instruction::LoadUnit(2),
                Instruction::Return(2),
            ],
            vec![Value::Int64(42)],
        );

        // Parent function:
        // 1. Get self pid
        // 2. Load child function
        // 3. Spawn child with self pid as argument
        // 4. Receive message
        // 5. Return received message
        let parent_func = make_function_with_consts(
            "main",
            vec![
                Instruction::SelfPid(0),              // r0 = self()
                Instruction::LoadConst(1, 0),         // r1 = child_func
                Instruction::Spawn(2, 1, vec![0].into()),    // r2 = spawn(child_func, [self()])
                Instruction::Receive(0),               // r0 = receive()
                Instruction::Return(0),
            ],
            vec![Value::Function(child_func)],
        );

        runtime.spawn_initial(parent_func);
        let result = runtime.run().unwrap();

        // Parent should receive 42 from child
        assert_eq!(result, Some(GcValue::Int64(42)));
    }
}
