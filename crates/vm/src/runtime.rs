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

use crate::gc::{GcNativeFn, GcValue, Heap};
use crate::process::ExitReason;
use crate::scheduler::Scheduler;
use crate::value::TypeValue;
use crate::value::{FunctionValue, Instruction, Pid, RuntimeError, Value};
use crate::vm::CallFrame;

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
    cached_functions: std::collections::HashMap<String, Rc<FunctionValue>>,
    cached_natives: std::collections::HashMap<String, Rc<GcNativeFn>>,
    cached_types: std::collections::HashMap<String, Rc<TypeValue>>,
}

impl Runtime {
    /// Create a new runtime.
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            output: Vec::new(),
            cached_functions: std::collections::HashMap::new(),
            cached_natives: std::collections::HashMap::new(),
            cached_types: std::collections::HashMap::new(),
        }
    }

    /// Register a global function.
    pub fn register_function(&mut self, name: &str, func: Rc<FunctionValue>) {
        self.scheduler.functions.write().insert(name.to_string(), func);
    }

    /// Register a type definition.
    pub fn register_type(&mut self, name: &str, type_val: Rc<TypeValue>) {
        self.scheduler.types.write().insert(name.to_string(), type_val);
    }

    /// Register a native function.
    pub fn register_native<F>(&mut self, name: &str, arity: usize, func: F)
    where
        F: Fn(&[GcValue], &mut Heap) -> Result<GcValue, RuntimeError> + Send + Sync + 'static,
    {
        let native = Rc::new(GcNativeFn {
            name: name.to_string(),
            arity,
            func: Box::new(func),
        });
        self.scheduler.natives.write().insert(name.to_string(), native);
    }

    /// Register built-in functions for all processes.
    pub fn register_builtins(&mut self) {
        self.register_native("print", 1, |args, heap| {
            let s = heap.display_value(&args[0]);
            Ok(GcValue::String(heap.alloc_string(s)))
        });

        self.register_native("println", 1, |args, heap| {
            let s = heap.display_value(&args[0]);
            println!("{}", s);
            Ok(GcValue::Unit)
        });

        self.register_native("typeOf", 1, |args, heap| {
            let type_name = args[0].type_name(heap).to_string();
            Ok(GcValue::String(heap.alloc_string(type_name)))
        });

        self.register_native("length", 1, |args, heap| {
            match &args[0] {
                GcValue::List(ptr) => {
                    if let Some(list) = heap.get_list(*ptr) {
                        Ok(GcValue::Int(list.items.len() as i64))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "invalid list pointer".to_string(),
                        })
                    }
                }
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        Ok(GcValue::Int(s.data.len() as i64))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "List or String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        // Testing builtins
        self.register_native("assert", 1, |args, _heap| {
            match &args[0] {
                GcValue::Bool(true) => Ok(GcValue::Unit),
                GcValue::Bool(false) => Err(RuntimeError::Panic("Assertion failed".to_string())),
                other => Err(RuntimeError::TypeError {
                    expected: "Bool".to_string(),
                    found: format!("{:?}", other),
                }),
            }
        });

        self.register_native("assert_eq", 2, |args, heap| {
            let expected = &args[0];
            let actual = &args[1];
            if heap.gc_values_equal(expected, actual) {
                Ok(GcValue::Unit)
            } else {
                let expected_str = heap.display_value(expected);
                let actual_str = heap.display_value(actual);
                Err(RuntimeError::Panic(format!(
                    "Assertion failed: expected {}, got {}",
                    expected_str, actual_str
                )))
            }
        });

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
    pub fn spawn_initial(&mut self, func: Rc<FunctionValue>) -> Pid {
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
    pub fn spawn_process(
        &mut self,
        func: Rc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
        parent_pid: Pid,
    ) -> Pid {
        let child_pid = self.scheduler.spawn();

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
        let natives = self.cached_natives.clone();
        let types = self.cached_types.clone();

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

                drop(proc);
                let result = self.execute_ipc_instruction(pid, instr_clone, &constants_values)?;
                match result {
                    ProcessStepResult::Continue => {
                        proc = process_handle.lock();
                        continue;
                    }
                    other => return Ok(other),
                }
            }

            // Fast path: increment IP
            proc.frames[frame_idx].ip += 1;

            // SAFETY: ip < code_len was checked above, constants are immutable
            let constants = unsafe { std::slice::from_raw_parts(constants_ptr, constants_len) };

            match self.execute_local_instruction_inline(&mut proc, frame_idx, instr, constants, &functions, &natives, &types)? {
                ProcessStepResult::Continue => continue,
                other => return Ok(other),
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
        functions: &std::collections::HashMap<String, Rc<FunctionValue>>,
        natives: &std::collections::HashMap<String, Rc<GcNativeFn>>,
        types: &std::collections::HashMap<String, Rc<TypeValue>>,
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

            // === Arithmetic (HOT PATH - no cloning!) ===
            Instruction::AddInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(*dst, GcValue::Int(x + y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::SubInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(*dst, GcValue::Int(x - y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::MulInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(*dst, GcValue::Int(x * y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            // === Comparison (HOT PATH) ===
            Instruction::LtInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(*dst, GcValue::Bool(*x < *y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::LeInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(*dst, GcValue::Bool(*x <= *y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::GtInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(*dst, GcValue::Bool(*x > *y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::Eq(dst, a, b) => {
                let va = reg_clone!(*a);
                let vb = reg_clone!(*b);
                let eq = proc.heap.gc_values_equal(&va, &vb);
                set_reg!(*dst, GcValue::Bool(eq));
            }

            // === Logical ===
            Instruction::Not(dst, src) => {
                match reg!(*src) {
                    GcValue::Bool(b) => set_reg!(*dst, GcValue::Bool(!b)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: "other".to_string(),
                    }),
                }
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
                proc.frames.pop();

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

                // Push new frame - use actual register count
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
                    captures,
                    return_reg: Some(dst),
                };
                proc.frames.push(frame);
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
                let name = match constants.get(*name_idx as usize) {
                    Some(Value::String(s)) => s,
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| reg_clone!(r)).collect();

                let func = functions.get(&**name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Push new frame - use actual register count
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

            Instruction::CallSelf(dst, ref arg_regs) => {
                // Self-recursion: reuse current frame's function (no HashMap lookup!)
                let dst = *dst;
                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| reg_clone!(r)).collect();

                // Get current function directly from frame
                let func = proc.frames[frame_idx].function.clone();

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
                match reg!(*tuple_reg) {
                    GcValue::Tuple(ptr) => {
                        let item = proc.heap.get_tuple(*ptr)
                            .and_then(|t| t.items.get(*idx as usize).cloned());
                        match item {
                            Some(val) => set_reg!(*dst, val),
                            None => return Err(RuntimeError::IndexOutOfBounds {
                                index: *idx as i64,
                                length: 0,
                            }),
                        }
                    }
                    other => {
                        return Err(RuntimeError::TypeError {
                            expected: "Tuple".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
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

            Instruction::Receive => {
                if let Some(msg) = proc.try_receive() {
                    proc.frames[frame_idx].registers[0] = msg;
                } else {
                    proc.wait_for_message();
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip -= 1;
                    }
                    return Ok(ProcessStepResult::Waiting);
                }
            }

            Instruction::ReceiveTimeout(_timeout_reg) => {
                return Err(RuntimeError::Panic("ReceiveTimeout not yet implemented".to_string()));
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
                            .map(|l| l.items.is_empty())
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
                match reg!(*record) {
                    GcValue::Record(ptr) => {
                        let result = proc.heap.get_record(*ptr)
                            .and_then(|r| r.fields.get(*field_idx as usize))
                            .cloned()
                            .ok_or_else(|| RuntimeError::Panic("Invalid field access".to_string()))?;
                        set_reg!(*dst, result);
                    }
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Record".to_string(),
                            found: "other".to_string(),
                        });
                    }
                }
            }

            // === Division (direct access) ===
            Instruction::DivInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => {
                        if *y == 0 {
                            return Err(RuntimeError::Panic("Division by zero".to_string()));
                        }
                        set_reg!(*dst, GcValue::Int(x / y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::ModInt(dst, a, b) => {
                match (reg!(*a), reg!(*b)) {
                    (GcValue::Int(x), GcValue::Int(y)) => {
                        if *y == 0 {
                            return Err(RuntimeError::Panic("Division by zero".to_string()));
                        }
                        set_reg!(*dst, GcValue::Int(x % y));
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            // === Debug ===
            Instruction::Nop => {}

            Instruction::DebugPrint(r) => {
                let value = reg_clone!(*r);
                println!("DEBUG: {:?}", value);
                proc.output.push(format!("{:?}", value));
            }

            // Unhandled instructions - note: we match on reference so can't move
            _ => {
                return Err(RuntimeError::Panic(format!(
                    "Instruction {:?} not yet implemented in Runtime",
                    instr
                )));
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
        constants: &[Value],
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

    fn make_function(name: &str, code: Vec<Instruction>) -> Rc<FunctionValue> {
        Rc::new(FunctionValue {
            name: name.to_string(),
            arity: 0,
            param_names: Vec::new(),
            code: Rc::new(Chunk {
                code,
                constants: Vec::new(),
                lines: Vec::new(),
                locals: Vec::new(),
                register_count: 256,
            }),
            module: None,
            source_span: None,
            jit_code: None,
        })
    }

    fn make_function_with_consts(name: &str, code: Vec<Instruction>, constants: Vec<Value>) -> Rc<FunctionValue> {
        Rc::new(FunctionValue {
            name: name.to_string(),
            arity: 0,
            param_names: Vec::new(),
            code: Rc::new(Chunk {
                code,
                constants,
                lines: Vec::new(),
                locals: Vec::new(),
                register_count: 256,
            }),
            module: None,
            source_span: None,
            jit_code: None,
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
            vec![Value::Int(42)],
        );

        runtime.spawn_initial(func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int(42)));
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
            vec![Value::Int(99)],
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
            vec![Value::Int(5), Value::Int(2)],
        );

        runtime.spawn_initial(func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int(10)));
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
        let add_func = Rc::new(FunctionValue {
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
            vec![Value::Int(2), Value::Int(3), Value::String(Rc::new("add".to_string()))],
        );

        runtime.register_function("add", add_func);
        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int(5)));
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
            vec![Value::Int(0), Value::Int(1), Value::String(Rc::new("sum".to_string()))],
        );
        let sum_func = Rc::new(FunctionValue {
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
            vec![Value::Int(3), Value::String(Rc::new("sum".to_string()))],
        );

        runtime.register_function("sum", sum_func);
        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int(6)));
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
        let id_func = Rc::new(FunctionValue {
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
            vec![Value::Int(42), Value::String(Rc::new("id".to_string()))],
        );

        runtime.register_function("id", id_func);
        runtime.spawn_initial(main_func);
        let result = runtime.run().unwrap();

        assert_eq!(result, Some(GcValue::Int(42)));
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
            vec![Value::Int(42)],
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
                Instruction::Receive,                  // r0 = receive()
                Instruction::Return(0),
            ],
            vec![Value::Function(child_func)],
        );

        runtime.spawn_initial(parent_func);
        let result = runtime.run().unwrap();

        // Parent should receive 42 from child
        assert_eq!(result, Some(GcValue::Int(42)));
    }
}
