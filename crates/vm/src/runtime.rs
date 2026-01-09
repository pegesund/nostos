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
}

impl Runtime {
    /// Create a new runtime.
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            output: Vec::new(),
        }
    }

    /// Register a global function.
    pub fn register_function(&mut self, name: &str, func: Rc<FunctionValue>) {
        self.scheduler.functions.insert(name.to_string(), func);
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
        self.scheduler.natives.insert(name.to_string(), native);
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
    }

    /// Spawn the initial process and run a function.
    pub fn spawn_initial(&mut self, func: Rc<FunctionValue>) -> Pid {
        let pid = self.scheduler.spawn();

        if let Some(process) = self.scheduler.get_process_mut(pid) {
            let frame = CallFrame {
                function: func,
                ip: 0,
                registers: vec![GcValue::Unit; 256],
                captures: Vec::new(),
                return_reg: None,
            };
            process.frames.push(frame);
        }

        pid
    }

    /// Spawn a new process running a function with arguments.
    pub fn spawn_process(
        &mut self,
        func: Rc<FunctionValue>,
        args: Vec<GcValue>,
        parent_pid: Pid,
    ) -> Pid {
        let child_pid = self.scheduler.spawn();

        // Copy arguments from parent's heap to child's heap
        if let Some(parent) = self.scheduler.get_process(parent_pid) {
            let parent_heap = &parent.heap as *const Heap;

            if let Some(child) = self.scheduler.get_process_mut(child_pid) {
                // Deep copy arguments to child's heap
                let copied_args: Vec<GcValue> = args
                    .iter()
                    .map(|arg| unsafe { child.heap.deep_copy(arg, &*parent_heap) })
                    .collect();

                // Set up call frame with arguments in registers
                let mut registers = vec![GcValue::Unit; 256];
                for (i, arg) in copied_args.into_iter().enumerate() {
                    if i < 256 {
                        registers[i] = arg;
                    }
                }

                let frame = CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: None,
                };
                child.frames.push(frame);
            }
        }

        child_pid
    }

    /// Run all processes until completion or deadlock.
    pub fn run(&mut self) -> Result<Option<GcValue>, RuntimeError> {
        self.register_builtins();

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
    fn run_process(&mut self, pid: Pid) -> Result<ProcessStepResult, RuntimeError> {
        loop {
            // Check if we should yield
            if let Some(proc) = self.scheduler.get_process(pid) {
                if proc.should_yield() {
                    return Ok(ProcessStepResult::Yield);
                }
                if proc.frames.is_empty() {
                    return Ok(ProcessStepResult::Finished(GcValue::Unit));
                }
            } else {
                return Err(RuntimeError::Panic(format!("Process {:?} not found", pid)));
            }

            // Execute one step
            match self.execute_step(pid)? {
                ProcessStepResult::Continue => continue,
                other => return Ok(other),
            }
        }
    }

    /// Execute a single instruction for a process.
    fn execute_step(&mut self, pid: Pid) -> Result<ProcessStepResult, RuntimeError> {
        // Get instruction to execute
        let (instr, constants, _func_name, _ip_before) = {
            let proc = self.scheduler.get_process(pid)
                .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
            let frame = proc.frames.last()
                .ok_or_else(|| RuntimeError::Panic("No frame".to_string()))?;

            if frame.ip >= frame.function.code.code.len() {
                return Ok(ProcessStepResult::Finished(GcValue::Unit));
            }

            let instr = frame.function.code.code[frame.ip].clone();
            let constants = frame.function.code.constants.clone();
            (instr, constants, frame.function.name.clone(), frame.ip)
        };

        // Uncomment for debugging:
        // eprintln!("[{}:{}] {:?}", _func_name, _ip_before, instr);

        // Increment IP
        if let Some(proc) = self.scheduler.get_process_mut(pid) {
            if let Some(frame) = proc.frames.last_mut() {
                frame.ip += 1;
            }
        }

        // Execute the instruction
        self.execute_instruction(pid, instr, &constants)
    }

    /// Execute a single instruction.
    fn execute_instruction(
        &mut self,
        pid: Pid,
        instr: Instruction,
        constants: &[Value],
    ) -> Result<ProcessStepResult, RuntimeError> {
        // Helper function for register access (avoids ? in closures)
        fn get_register(scheduler: &Scheduler, pid: Pid, r: u8) -> GcValue {
            scheduler.get_process(pid)
                .and_then(|proc| proc.frames.last())
                .map(|f| f.registers[r as usize].clone())
                .unwrap_or(GcValue::Unit)
        }

        // Helper macros for register access
        macro_rules! get_reg {
            ($r:expr) => {{
                get_register(&self.scheduler, pid, $r)
            }};
        }

        macro_rules! set_reg {
            ($r:expr, $v:expr) => {{
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.registers[$r as usize] = $v;
                    }
                }
            }};
        }

        match instr {
            // === Constants and moves ===
            Instruction::LoadConst(dst, idx) => {
                let value = constants.get(idx as usize)
                    .ok_or_else(|| RuntimeError::Panic("Constant not found".to_string()))?
                    .clone();
                let proc = self.scheduler.get_process_mut(pid)
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                let gc_value = proc.heap.value_to_gc(&value);
                if let Some(frame) = proc.frames.last_mut() {
                    frame.registers[dst as usize] = gc_value;
                }
            }

            Instruction::Move(dst, src) => {
                let val = get_reg!(src);
                set_reg!(dst, val);
            }

            Instruction::LoadUnit(dst) => {
                set_reg!(dst, GcValue::Unit);
            }

            Instruction::LoadTrue(dst) => {
                set_reg!(dst, GcValue::Bool(true));
            }

            Instruction::LoadFalse(dst) => {
                set_reg!(dst, GcValue::Bool(false));
            }

            // === Arithmetic ===
            Instruction::AddInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(dst, GcValue::Int(x + y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::SubInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(dst, GcValue::Int(x - y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::MulInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(dst, GcValue::Int(x * y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            // === Comparison ===
            Instruction::LtInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(dst, GcValue::Bool(x < y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::LeInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(dst, GcValue::Bool(x <= y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::GtInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int(x), GcValue::Int(y)) => set_reg!(dst, GcValue::Bool(x > y)),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: "other".to_string(),
                    }),
                }
            }

            Instruction::Eq(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                let proc = self.scheduler.get_process(pid)
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                let eq = proc.heap.gc_values_equal(&va, &vb);
                set_reg!(dst, GcValue::Bool(eq));
            }

            // === Logical ===
            Instruction::Not(dst, src) => {
                let val = get_reg!(src);
                let result = match val {
                    GcValue::Bool(b) => GcValue::Bool(!b),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: "other".to_string(),
                    }),
                };
                set_reg!(dst, result);
            }

            // === Control flow ===
            Instruction::Jump(offset) => {
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                    if offset < 0 {
                        proc.consume_reductions(1);
                    }
                }
            }

            Instruction::JumpIfTrue(cond, offset) => {
                let val = get_reg!(cond);
                if val.is_truthy() {
                    if let Some(proc) = self.scheduler.get_process_mut(pid) {
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.ip = (frame.ip as isize + offset as isize) as usize;
                        }
                        if offset < 0 {
                            proc.consume_reductions(1);
                        }
                    }
                }
            }

            Instruction::JumpIfFalse(cond, offset) => {
                let val = get_reg!(cond);
                if !val.is_truthy() {
                    if let Some(proc) = self.scheduler.get_process_mut(pid) {
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.ip = (frame.ip as isize + offset as isize) as usize;
                        }
                        if offset < 0 {
                            proc.consume_reductions(1);
                        }
                    }
                }
            }

            // === Function calls ===
            Instruction::Return(src) => {
                let return_val = get_reg!(src);

                let proc = self.scheduler.get_process_mut(pid)
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;

                let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                proc.frames.pop();

                if proc.frames.is_empty() {
                    return Ok(ProcessStepResult::Finished(return_val));
                }

                // Put return value in caller's return register
                if let Some(ret_reg) = return_reg {
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.registers[ret_reg as usize] = return_val;
                    }
                }

                // Continue execution in caller frame
                return Ok(ProcessStepResult::Continue);
            }

            Instruction::Call(dst, func_reg, ref args) => {
                // Get function and arguments
                let func_val = get_reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|&r| get_reg!(r)).collect();

                let (func, captures) = match func_val {
                    GcValue::Function(f) => (f, Vec::new()),
                    GcValue::Closure(ptr) => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        let closure = proc.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::TypeError {
                                expected: "Closure".to_string(),
                                found: "invalid".to_string(),
                            })?;
                        (closure.function.clone(), closure.captures.clone())
                    }
                    other => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
                };

                // Push new frame
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    let mut registers = vec![GcValue::Unit; 256];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < 256 {
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
            }

            Instruction::TailCall(func_reg, ref args) => {
                let func_val = get_reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|&r| get_reg!(r)).collect();

                let (func, captures) = match func_val {
                    GcValue::Function(f) => (f, Vec::new()),
                    GcValue::Closure(ptr) => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        let closure = proc.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::TypeError {
                                expected: "Closure".to_string(),
                                found: "invalid".to_string(),
                            })?;
                        (closure.function.clone(), closure.captures.clone())
                    }
                    other => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
                };

                // Replace current frame
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    let mut registers = vec![GcValue::Unit; 256];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < 256 {
                            registers[i] = arg;
                        }
                    }

                    if let Some(frame) = proc.frames.last_mut() {
                        frame.function = func;
                        frame.ip = 0;
                        frame.registers = registers;
                        frame.captures = captures;
                    }

                    proc.consume_reductions(1);
                }
            }

            Instruction::CallNative(dst, name_idx, ref args) => {
                let arg_values: Vec<GcValue> = args.iter().map(|&r| get_reg!(r)).collect();

                let name = match constants.get(name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(RuntimeError::Panic("Invalid native function name".to_string())),
                };

                let native = self.scheduler.natives.get(&*name).cloned()
                    .ok_or_else(|| RuntimeError::Panic(format!("Undefined native: {}", name)))?;

                let result = {
                    let proc = self.scheduler.get_process_mut(pid)
                        .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                    (native.func)(&arg_values, &mut proc.heap)?
                };

                set_reg!(dst, result);
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    proc.consume_reductions(10);
                }
            }

            Instruction::CallByName(dst, name_idx, ref arg_regs) => {
                let name = match constants.get(name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = self.scheduler.functions.get(&*name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Push new frame
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    let mut registers = vec![GcValue::Unit; 256];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < 256 {
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
            }

            Instruction::TailCallByName(name_idx, ref arg_regs) => {
                let name = match constants.get(name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = self.scheduler.functions.get(&*name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Replace current frame
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    let mut registers = vec![GcValue::Unit; 256];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < 256 {
                            registers[i] = arg;
                        }
                    }

                    if let Some(frame) = proc.frames.last_mut() {
                        frame.function = func;
                        frame.ip = 0;
                        frame.registers = registers;
                        frame.captures = Vec::new();
                    }

                    proc.consume_reductions(1);
                }
            }

            // === Collections ===
            Instruction::MakeList(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| get_reg!(r)).collect();
                let proc = self.scheduler.get_process_mut(pid)
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                let ptr = proc.heap.alloc_list(items);
                set_reg!(dst, GcValue::List(ptr));
            }

            Instruction::MakeTuple(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| get_reg!(r)).collect();
                let proc = self.scheduler.get_process_mut(pid)
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                let ptr = proc.heap.alloc_tuple(items);
                set_reg!(dst, GcValue::Tuple(ptr));
            }

            Instruction::GetCapture(dst, idx) => {
                let val = {
                    let proc = self.scheduler.get_process(pid)
                        .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                    proc.frames.last()
                        .and_then(|f| f.captures.get(idx as usize))
                        .cloned()
                        .unwrap_or(GcValue::Unit)
                };
                set_reg!(dst, val);
            }

            // === Concurrency ===
            Instruction::SelfPid(dst) => {
                set_reg!(dst, GcValue::Pid(pid.0));
            }

            Instruction::Spawn(dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = match func_val {
                    GcValue::Function(f) => f,
                    GcValue::Closure(ptr) => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        proc.heap.get_closure(ptr)
                            .map(|c| c.function.clone())
                            .ok_or_else(|| RuntimeError::TypeError {
                                expected: "Function".to_string(),
                                found: "invalid closure".to_string(),
                            })?
                    }
                    other => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
                };

                let child_pid = self.spawn_process(func, args, pid);
                set_reg!(dst, GcValue::Pid(child_pid.0));

                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    proc.consume_reductions(100);
                }
            }

            Instruction::Send(pid_reg, msg_reg) => {
                let target_val = get_reg!(pid_reg);
                let message = get_reg!(msg_reg);

                let target_pid = match target_val {
                    GcValue::Pid(p) => Pid(p),
                    other => {
                        let proc = self.scheduler.get_process(pid)
                            .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                        return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: other.type_name(&proc.heap).to_string(),
                        });
                    }
                };

                self.scheduler.send(pid, target_pid, message)?;

                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    proc.consume_reductions(10);
                }
            }

            Instruction::Receive => {
                let received = {
                    let proc = self.scheduler.get_process_mut(pid)
                        .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                    proc.try_receive()
                };

                if let Some(msg) = received {
                    // Put message in register 0 (convention)
                    set_reg!(0, msg);
                } else {
                    // No message - block
                    if let Some(proc) = self.scheduler.get_process_mut(pid) {
                        proc.wait_for_message();
                        // Back up IP so we retry receive
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.ip -= 1;
                        }
                    }
                    return Ok(ProcessStepResult::Waiting);
                }
            }

            Instruction::SpawnLink(dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = match func_val {
                    GcValue::Function(f) => f,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: "other".to_string(),
                    }),
                };

                let child_pid = self.spawn_process(func, args, pid);

                // Create link
                if let Some(parent) = self.scheduler.get_process_mut(pid) {
                    parent.link(child_pid);
                }
                if let Some(child) = self.scheduler.get_process_mut(child_pid) {
                    child.link(pid);
                }

                set_reg!(dst, GcValue::Pid(child_pid.0));
            }

            Instruction::SpawnMonitor(pid_dst, ref_dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = match func_val {
                    GcValue::Function(f) => f,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: "other".to_string(),
                    }),
                };

                let child_pid = self.spawn_process(func, args, pid);
                let ref_id = self.scheduler.make_ref();

                if let Some(parent) = self.scheduler.get_process_mut(pid) {
                    parent.add_monitor(ref_id, child_pid);
                }
                if let Some(child) = self.scheduler.get_process_mut(child_pid) {
                    child.add_monitored_by(ref_id, pid);
                }

                set_reg!(pid_dst, GcValue::Pid(child_pid.0));
                set_reg!(ref_dst, GcValue::Ref(ref_id.0));
            }

            Instruction::ReceiveTimeout(_timeout_reg) => {
                return Err(RuntimeError::Panic("ReceiveTimeout not yet implemented".to_string()));
            }

            // === Debug ===
            Instruction::Nop => {}

            Instruction::DebugPrint(r) => {
                let value = get_reg!(r);
                println!("DEBUG: {:?}", value);
                if let Some(proc) = self.scheduler.get_process_mut(pid) {
                    proc.output.push(format!("{:?}", value));
                }
            }

            // Unhandled instructions
            other => {
                return Err(RuntimeError::Panic(format!(
                    "Instruction {:?} not yet implemented in Runtime",
                    other
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
                Instruction::Spawn(1, 0, vec![]), // Spawn child, store pid in r1
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
                Instruction::CallByName(2, 2, vec![0, 1]),  // r2 = add(r0, r1)
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
                Instruction::CallByName(6, 2, vec![5]), // r6 = sum(n-1)
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
                Instruction::CallByName(1, 1, vec![0]),
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
                Instruction::TailCallByName(1, vec![0]),  // tail call id(42)
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
                Instruction::Spawn(2, 1, vec![0]),    // r2 = spawn(child_func, [self()])
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
