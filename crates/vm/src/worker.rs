//! Worker threads for parallel process execution.
//!
//! Each worker has:
//! - A local deque for work (push/pop from same end)
//! - Ability to steal from other workers (pop from opposite end)
//! - JIT-aware execution (uses JIT code when available)
//!
//! Work stealing ensures good load balancing without central coordination.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam::deque::{Injector, Stealer, Worker as WorkQueue};
use parking_lot::Mutex;

use crate::gc::GcValue;
use crate::process::{ExitReason, ProcessState};
use crate::scheduler::Scheduler;
use crate::value::{FunctionValue, Pid, RuntimeError};
use crate::process::CallFrame;

/// Thread-safe result from main process.
/// Since GcValue contains Rc (not Send), we transfer simple values only.
#[derive(Debug, Clone)]
pub enum MainResult {
    /// Process still running
    Running,
    /// Finished with integer value
    Int(i64),
    /// Finished with float value
    Float(f64),
    /// Finished with boolean
    Bool(bool),
    /// Finished with Unit
    Unit,
    /// Finished with Pid
    Pid(u64),
    /// Error occurred
    Error(String),
}

impl MainResult {
    /// Convert GcValue to transferable MainResult
    fn from_gc_value(value: &GcValue) -> Self {
        match value {
            GcValue::Int64(i) => MainResult::Int(*i),
            GcValue::Float64(f) => MainResult::Float(*f),
            GcValue::Bool(b) => MainResult::Bool(*b),
            GcValue::Unit => MainResult::Unit,
            GcValue::Pid(p) => MainResult::Pid(*p),
            // For complex values, we'd need serialization
            // For now, just return Unit
            _ => MainResult::Unit,
        }
    }

    /// Convert to GcValue (for compatibility)
    pub fn to_gc_value(&self) -> Option<GcValue> {
        match self {
            MainResult::Running => None,
            MainResult::Int(i) => Some(GcValue::Int64(*i)),
            MainResult::Float(f) => Some(GcValue::Float64(*f)),
            MainResult::Bool(b) => Some(GcValue::Bool(*b)),
            MainResult::Unit => Some(GcValue::Unit),
            MainResult::Pid(p) => Some(GcValue::Pid(*p)),
            MainResult::Error(_) => None,
        }
    }
}

/// Configuration for the worker pool.
#[derive(Clone, Debug)]
pub struct WorkerPoolConfig {
    /// Number of worker threads (0 = use available CPUs)
    pub num_workers: usize,
    /// Reductions per time slice before yielding
    pub reductions_per_slice: usize,
    /// Enable JIT compilation for hot functions
    pub enable_jit: bool,
    /// Call count threshold for JIT compilation
    pub jit_threshold: usize,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            num_workers: 0, // Auto-detect
            reductions_per_slice: 2000,
            enable_jit: true,
            jit_threshold: 1000,
        }
    }
}

/// A pool of worker threads for parallel execution.
pub struct WorkerPool {
    /// The shared scheduler (process registry, functions, etc.)
    scheduler: Arc<Scheduler>,
    /// Global injector for new processes
    injector: Arc<Injector<Pid>>,
    /// Stealers for each worker's local queue
    stealers: Vec<Stealer<Pid>>,
    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Number of active workers
    active_workers: Arc<AtomicUsize>,
    /// Configuration
    config: WorkerPoolConfig,
    /// Result from main process (pid 1) - uses MainResult for thread safety
    main_result: Arc<Mutex<MainResult>>,
}

impl WorkerPool {
    /// Create a new worker pool.
    pub fn new(scheduler: Scheduler, config: WorkerPoolConfig) -> Self {
        let num_workers = if config.num_workers == 0 {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            config.num_workers
        };

        let scheduler = Arc::new(scheduler);
        let injector = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let active_workers = Arc::new(AtomicUsize::new(0));
        let main_result = Arc::new(Mutex::new(MainResult::Running));

        // Create local work queues and stealers
        let mut local_queues = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let queue = WorkQueue::new_fifo();
            stealers.push(queue.stealer());
            local_queues.push(queue);
        }

        // Spawn worker threads
        let mut workers = Vec::with_capacity(num_workers);

        for (id, local_queue) in local_queues.into_iter().enumerate() {
            let worker = Worker {
                id,
                local: local_queue,
                injector: Arc::clone(&injector),
                stealers: stealers.clone(),
                scheduler: Arc::clone(&scheduler),
                shutdown: Arc::clone(&shutdown),
                active_workers: Arc::clone(&active_workers),
                main_result: Arc::clone(&main_result),
                config: config.clone(),
            };

            let handle = thread::Builder::new()
                .name(format!("nostos-worker-{}", id))
                .spawn(move || worker.run())
                .expect("Failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            scheduler,
            injector,
            stealers,
            workers,
            shutdown,
            active_workers,
            config,
            main_result,
        }
    }

    /// Create a single-threaded worker pool (for compatibility/testing).
    pub fn single_threaded(scheduler: Scheduler) -> Self {
        Self::new(
            scheduler,
            WorkerPoolConfig {
                num_workers: 1,
                ..Default::default()
            },
        )
    }

    /// Get the scheduler.
    pub fn scheduler(&self) -> &Arc<Scheduler> {
        &self.scheduler
    }

    /// Spawn the initial process with a function.
    pub fn spawn_initial(&self, func: Arc<FunctionValue>) -> Pid {
        // Use spawn_unqueued to avoid double-queuing
        let pid = self.scheduler.spawn_unqueued();

        self.scheduler.with_process_mut(pid, |process| {
            let frame = CallFrame {
                function: func.clone(),
                ip: 0,
                registers: vec![GcValue::Unit; 256],
                captures: Vec::new(),
                return_reg: None,
            };
            process.frames.push(frame);
        });

        // Push to global queue for workers to pick up
        self.injector.push(pid);

        pid
    }

    /// Run until all processes complete or main process finishes.
    pub fn run(&self) -> Result<Option<GcValue>, RuntimeError> {
        // Wait for main result or all processes to finish
        loop {
            // Check if main process returned a result
            let result = self.main_result.lock().clone();
            match result {
                MainResult::Running => {
                    // Still running, check if all processes are done
                    if !self.scheduler.has_processes() {
                        return Ok(None);
                    }
                    // Brief sleep to avoid busy-waiting
                    thread::sleep(std::time::Duration::from_micros(100));
                }
                MainResult::Error(msg) => {
                    return Err(RuntimeError::Panic(msg));
                }
                other => {
                    // Process finished with a value
                    return Ok(other.to_gc_value());
                }
            }
        }
    }

    /// Shutdown the worker pool.
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);

        // Wake up workers by pushing dummy work
        for _ in 0..self.workers.len() {
            // Workers will see shutdown flag and exit
        }

        // Wait for all workers to finish
        for handle in self.workers {
            let _ = handle.join();
        }
    }

    /// Get the number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.stealers.len()
    }
}

/// A single worker thread.
struct Worker {
    id: usize,
    /// Local work queue (FIFO deque)
    local: WorkQueue<Pid>,
    /// Global injector for new work
    injector: Arc<Injector<Pid>>,
    /// Stealers for other workers' queues
    stealers: Vec<Stealer<Pid>>,
    /// Shared scheduler
    scheduler: Arc<Scheduler>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Active worker count
    active_workers: Arc<AtomicUsize>,
    /// Main process result (thread-safe)
    main_result: Arc<Mutex<MainResult>>,
    /// Configuration
    config: WorkerPoolConfig,
}

impl Worker {
    /// Run the worker loop.
    fn run(self) {
        self.active_workers.fetch_add(1, Ordering::SeqCst);

        while !self.shutdown.load(Ordering::Relaxed) {
            // Try to get work
            if let Some(pid) = self.find_work() {
                self.execute_process(pid);
            } else {
                // No work available, brief sleep before retrying
                thread::yield_now();
            }
        }

        self.active_workers.fetch_sub(1, Ordering::SeqCst);
    }

    /// Find work: local queue -> scheduler run queue -> global injector -> steal from others.
    fn find_work(&self) -> Option<Pid> {
        // 1. Try local queue first (fastest)
        if let Some(pid) = self.local.pop() {
            return Some(pid);
        }

        // 2. Check scheduler's run queue for newly spawned/woken processes
        // This handles processes woken up by message delivery
        if let Some(pid) = self.scheduler.pop_run_queue() {
            return Some(pid);
        }

        // 3. Try global injector
        loop {
            match self.injector.steal_batch_and_pop(&self.local) {
                crossbeam::deque::Steal::Success(pid) => return Some(pid),
                crossbeam::deque::Steal::Empty => break,
                crossbeam::deque::Steal::Retry => continue,
            }
        }

        // 4. Try stealing from other workers
        let start = self.id;
        for i in 0..self.stealers.len() {
            let idx = (start + i + 1) % self.stealers.len();
            if idx == self.id {
                continue;
            }

            loop {
                match self.stealers[idx].steal() {
                    crossbeam::deque::Steal::Success(pid) => return Some(pid),
                    crossbeam::deque::Steal::Empty => break,
                    crossbeam::deque::Steal::Retry => continue,
                }
            }
        }

        None
    }

    /// Execute a process for one time slice.
    fn execute_process(&self, pid: Pid) {
        // Reset reductions for this time slice
        self.scheduler.with_process_mut(pid, |proc| {
            proc.reset_reductions();
            proc.state = ProcessState::Running;
        });

        // Run the process
        match self.run_process_slice(pid) {
            Ok(ProcessResult::Continue) => {
                // Process yielded, re-queue it
                self.local.push(pid);
            }
            Ok(ProcessResult::Finished(value)) => {
                // Process finished
                if pid == Pid(1) {
                    // Main process - store result (convert to thread-safe MainResult)
                    *self.main_result.lock() = MainResult::from_gc_value(&value);
                } else {
                    self.scheduler.process_exit(pid, ExitReason::Normal, Some(value));
                }
            }
            Ok(ProcessResult::Waiting) => {
                // Process is waiting for message, don't re-queue
                // It will be woken up when a message arrives
            }
            Err(err) => {
                // Process error
                if pid == Pid(1) {
                    *self.main_result.lock() = MainResult::Error(format!("{:?}", err));
                }
                self.scheduler
                    .process_exit(pid, ExitReason::Error(format!("{:?}", err)), None);
            }
        }
    }

    /// Run a process for one time slice.
    /// OPTIMIZED: Batches multiple instructions in a single lock scope.
    fn run_process_slice(&self, pid: Pid) -> Result<ProcessResult, RuntimeError> {
        // Get the process handle for direct access
        let handle = self
            .scheduler
            .get_process_handle(pid)
            .ok_or_else(|| RuntimeError::Panic(format!("Process {:?} not found", pid)))?;

        loop {
            // Execute a batch of pure instructions with a single lock
            let batch_result = self.execute_batch_locked(pid, &handle)?;

            match batch_result {
                BatchResult::Continue => continue,
                BatchResult::NeedsScheduler(instr, constants) => {
                    // Handle instruction that needs scheduler access
                    match self.execute_scheduler_instruction(pid, instr, &constants)? {
                        ProcessResult::Continue => continue,
                        other => return Ok(other),
                    }
                }
                BatchResult::Yield => return Ok(ProcessResult::Continue),
                BatchResult::Finished(value) => return Ok(ProcessResult::Finished(value)),
                BatchResult::Waiting => return Ok(ProcessResult::Waiting),
            }
        }
    }

    /// Execute a batch of pure instructions while holding the process lock.
    /// Returns when we need to yield, finish, or execute a scheduler instruction.
    fn execute_batch_locked(
        &self,
        pid: Pid,
        handle: &crate::scheduler::ProcessHandle,
    ) -> Result<BatchResult, RuntimeError> {
        use crate::value::{Instruction, Value};

        // Hold the lock for the entire batch
        let mut proc = handle.lock();

        // Execute up to BATCH_SIZE instructions
        const BATCH_SIZE: usize = 32;

        for _ in 0..BATCH_SIZE {
            // Check yield conditions
            if proc.should_yield() {
                return Ok(BatchResult::Yield);
            }

            // Get instruction info without holding frame reference
            let (instr, constants) = {
                let frame = match proc.frames.last() {
                    Some(f) => f,
                    None => return Ok(BatchResult::Finished(GcValue::Unit)),
                };

                if frame.ip >= frame.function.code.code.len() {
                    return Ok(BatchResult::Finished(GcValue::Unit));
                }

                let instr = frame.function.code.code[frame.ip].clone();
                let constants = frame.function.code.constants.clone();
                (instr, constants)
            };

            // Increment IP
            proc.frames.last_mut().unwrap().ip += 1;

            // Handle pure instructions directly (no scheduler access needed)
            match &instr {
                // === Constants and moves ===
                Instruction::LoadConst(dst, idx) => {
                    let value = constants
                        .get(*idx as usize)
                        .ok_or_else(|| RuntimeError::Panic("Constant not found".to_string()))?
                        .clone();
                    let gc_value = proc.heap.value_to_gc(&value);
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = gc_value;
                }

                Instruction::Move(dst, src) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let val = frame.registers[*src as usize].clone();
                    frame.registers[*dst as usize] = val;
                }

                Instruction::LoadUnit(dst) => {
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Unit;
                }

                Instruction::LoadTrue(dst) => {
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Bool(true);
                }

                Instruction::LoadFalse(dst) => {
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Bool(false);
                }

                // === Arithmetic ===
                Instruction::AddInt(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x + y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Int".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::SubInt(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x - y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Int".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::MulInt(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Int64(x * y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Int".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::AddFloat(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x + y),
                        (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x + y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Float".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::SubFloat(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x - y),
                        (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x - y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Float".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::MulFloat(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x * y),
                        (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x * y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Float".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::DivFloat(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Float64(x), GcValue::Float64(y)) => GcValue::Float64(x / y),
                        (GcValue::Float32(x), GcValue::Float32(y)) => GcValue::Float32(x / y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Float".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                // === Comparison ===
                Instruction::LtInt(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Bool(*x < *y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Int".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::LeInt(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Bool(*x <= *y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Int".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::GtInt(dst, a, b) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match (&frame.registers[*a as usize], &frame.registers[*b as usize]) {
                        (GcValue::Int64(x), GcValue::Int64(y)) => GcValue::Bool(*x > *y),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Int".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                Instruction::Eq(dst, a, b) => {
                    let (va, vb) = {
                        let frame = proc.frames.last().unwrap();
                        (frame.registers[*a as usize].clone(), frame.registers[*b as usize].clone())
                    };
                    let eq = proc.heap.gc_values_equal(&va, &vb);
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Bool(eq);
                }

                // === Logical ===
                Instruction::Not(dst, src) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let result = match &frame.registers[*src as usize] {
                        GcValue::Bool(b) => GcValue::Bool(!b),
                        _ => {
                            return Err(RuntimeError::TypeError {
                                expected: "Bool".to_string(),
                                found: "other".to_string(),
                            })
                        }
                    };
                    frame.registers[*dst as usize] = result;
                }

                // === Control flow ===
                Instruction::Jump(offset) => {
                    let frame = proc.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + *offset as isize) as usize;
                    if *offset < 0 {
                        proc.consume_reductions(1);
                    }
                }

                Instruction::JumpIfTrue(cond, offset) => {
                    let frame = proc.frames.last_mut().unwrap();
                    if frame.registers[*cond as usize].is_truthy() {
                        frame.ip = (frame.ip as isize + *offset as isize) as usize;
                        if *offset < 0 {
                            proc.consume_reductions(1);
                        }
                    }
                }

                Instruction::JumpIfFalse(cond, offset) => {
                    let frame = proc.frames.last_mut().unwrap();
                    if !frame.registers[*cond as usize].is_truthy() {
                        frame.ip = (frame.ip as isize + *offset as isize) as usize;
                        if *offset < 0 {
                            proc.consume_reductions(1);
                        }
                    }
                }

                // === Return (can finish the process) ===
                Instruction::Return(src) => {
                    let (return_val, return_reg) = {
                        let frame = proc.frames.last().unwrap();
                        (frame.registers[*src as usize].clone(), frame.return_reg)
                    };
                    proc.frames.pop();

                    if proc.frames.is_empty() {
                        return Ok(BatchResult::Finished(return_val));
                    }

                    if let Some(ret_reg) = return_reg {
                        if let Some(caller_frame) = proc.frames.last_mut() {
                            caller_frame.registers[ret_reg as usize] = return_val;
                        }
                    }
                }

                // === TailCallByName - hot path for recursive functions ===
                Instruction::TailCallByName(name_idx, arg_regs) => {
                    let name = match constants.get(*name_idx as usize) {
                        Some(Value::String(s)) => s.clone(),
                        _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                    };

                    let arg_values: Vec<GcValue> = {
                        let frame = proc.frames.last().unwrap();
                        arg_regs.iter().map(|&r| frame.registers[r as usize].clone()).collect()
                    };

                    // Release lock to look up function
                    drop(proc);

                    let func = self
                        .scheduler
                        .functions
                        .read()
                        .get(&*name)
                        .cloned()
                        .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                    // Track call count for JIT
                    self.scheduler.jit_tracker.record_call(&name);

                    // Reacquire lock
                    proc = handle.lock();

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

                // === CallByName - needs function lookup ===
                Instruction::CallByName(dst, name_idx, arg_regs) => {
                    let name = match constants.get(*name_idx as usize) {
                        Some(Value::String(s)) => s.clone(),
                        _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                    };

                    let arg_values: Vec<GcValue> = {
                        let frame = proc.frames.last().unwrap();
                        arg_regs.iter().map(|&r| frame.registers[r as usize].clone()).collect()
                    };

                    // Release lock to look up function
                    drop(proc);

                    let func = self
                        .scheduler
                        .functions
                        .read()
                        .get(&*name)
                        .cloned()
                        .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                    // Track call count for JIT
                    self.scheduler.jit_tracker.record_call(&name);

                    // Reacquire lock
                    proc = handle.lock();

                    let mut registers = vec![GcValue::Unit; 256];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < 256 {
                            registers[i] = arg;
                        }
                    }

                    let new_frame = CallFrame {
                        function: func,
                        ip: 0,
                        registers,
                        captures: Vec::new(),
                        return_reg: Some(*dst),
                    };
                    proc.frames.push(new_frame);
                    proc.consume_reductions(1);
                }

                // === Nop ===
                Instruction::Nop => {}

                // === GetCapture ===
                Instruction::GetCapture(dst, idx) => {
                    let frame = proc.frames.last_mut().unwrap();
                    let val = frame.captures.get(*idx as usize).cloned().unwrap_or(GcValue::Unit);
                    frame.registers[*dst as usize] = val;
                }

                // === SelfPid ===
                Instruction::SelfPid(dst) => {
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Pid(pid.0);
                }

                // === MakeList/MakeTuple (pure heap operations) ===
                Instruction::MakeList(dst, elements) => {
                    let items: Vec<GcValue> = {
                        let frame = proc.frames.last().unwrap();
                        elements.iter().map(|&r| frame.registers[r as usize].clone()).collect()
                    };
                    let ptr = proc.heap.alloc_list(items);
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::List(ptr);
                }

                Instruction::MakeTuple(dst, elements) => {
                    let items: Vec<GcValue> = {
                        let frame = proc.frames.last().unwrap();
                        elements.iter().map(|&r| frame.registers[r as usize].clone()).collect()
                    };
                    let ptr = proc.heap.alloc_tuple(items);
                    proc.frames.last_mut().unwrap().registers[*dst as usize] = GcValue::Tuple(ptr);
                }

                // === Instructions that need scheduler access ===
                Instruction::Send(..)
                | Instruction::Receive(..)
                | Instruction::Spawn(..)
                | Instruction::SpawnLink(..)
                | Instruction::SpawnMonitor(..)
                | Instruction::Call(..)
                | Instruction::TailCall(..)
                | Instruction::CallNative(..)
                | Instruction::ReceiveTimeout(..)
                | Instruction::DebugPrint(..) => {
                    // Decrement IP since we'll re-execute in scheduler mode
                    proc.frames.last_mut().unwrap().ip -= 1;
                    drop(proc);
                    return Ok(BatchResult::NeedsScheduler(instr, constants));
                }

                // === Other instructions ===
                other => {
                    proc.frames.last_mut().unwrap().ip -= 1;
                    drop(proc);
                    return Ok(BatchResult::NeedsScheduler(other.clone(), constants));
                }
            }
        }

        // Batch complete, continue with next batch
        Ok(BatchResult::Continue)
    }

    /// Execute an instruction that requires scheduler access.
    fn execute_scheduler_instruction(
        &self,
        pid: Pid,
        instr: crate::value::Instruction,
        constants: &[crate::value::Value],
    ) -> Result<ProcessResult, RuntimeError> {
        // Increment IP first
        self.scheduler.with_process_mut(pid, |proc| {
            if let Some(frame) = proc.frames.last_mut() {
                frame.ip += 1;
            }
        });

        // Use the original execute_instruction for scheduler-dependent ops
        self.execute_instruction(pid, instr, constants)
    }

    /// Execute one instruction, preferring JIT code if available.
    ///
    /// This is the JIT-friendly execution path:
    /// 1. Check if current function has JIT code
    /// 2. If yes, call native function directly
    /// 3. If no, interpret bytecode
    fn execute_step_jit_aware(&self, pid: Pid) -> Result<ProcessResult, RuntimeError> {
        // Get current frame info
        let frame_info = self.scheduler.with_process(pid, |proc| {
            let frame = proc.frames.last()?;

            // Check if function has JIT code
            let has_jit = frame.function.jit_code.is_some();

            if has_jit {
                // For JIT: we'd call the native function here
                // For now, just fall through to interpreter
                // TODO: Implement actual JIT call when Cranelift is integrated
            }

            // Get instruction for interpreter
            if frame.ip >= frame.function.code.code.len() {
                return None;
            }

            let instr = frame.function.code.code[frame.ip].clone();
            let constants = frame.function.code.constants.clone();
            Some((instr, constants, frame.function.name.clone()))
        });

        let (instr, constants, _func_name) = match frame_info {
            Some(Some(data)) => data,
            Some(None) => return Ok(ProcessResult::Finished(GcValue::Unit)),
            None => return Err(RuntimeError::Panic("Process not found".to_string())),
        };

        // Increment IP
        self.scheduler.with_process_mut(pid, |proc| {
            if let Some(frame) = proc.frames.last_mut() {
                frame.ip += 1;
            }
        });

        // Execute the instruction
        self.execute_instruction(pid, instr, &constants)
    }

    /// Execute a single instruction.
    /// This mirrors Runtime::execute_instruction but is designed for worker threads.
    fn execute_instruction(
        &self,
        pid: Pid,
        instr: crate::value::Instruction,
        constants: &[crate::value::Value],
    ) -> Result<ProcessResult, RuntimeError> {
        use crate::value::{Instruction, Value};

        // Helper macros for register access
        macro_rules! get_reg {
            ($r:expr) => {{
                self.scheduler
                    .with_process(pid, |proc| {
                        proc.frames
                            .last()
                            .map(|f| f.registers[$r as usize].clone())
                            .unwrap_or(GcValue::Unit)
                    })
                    .unwrap_or(GcValue::Unit)
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
            // === Constants and moves ===
            Instruction::LoadConst(dst, idx) => {
                let value = constants
                    .get(idx as usize)
                    .ok_or_else(|| RuntimeError::Panic("Constant not found".to_string()))?
                    .clone();
                self.scheduler.with_process_mut(pid, |proc| {
                    let gc_value = proc.heap.value_to_gc(&value);
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.registers[dst as usize] = gc_value;
                    }
                });
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
                    (GcValue::Int64(x), GcValue::Int64(y)) => set_reg!(dst, GcValue::Int64(x + y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::SubInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => set_reg!(dst, GcValue::Int64(x - y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::MulInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => set_reg!(dst, GcValue::Int64(x * y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::AddFloat(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => set_reg!(dst, GcValue::Float64(x + y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => set_reg!(dst, GcValue::Float32(x + y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Float".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::SubFloat(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => set_reg!(dst, GcValue::Float64(x - y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => set_reg!(dst, GcValue::Float32(x - y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Float".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::MulFloat(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => set_reg!(dst, GcValue::Float64(x * y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => set_reg!(dst, GcValue::Float32(x * y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Float".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::DivFloat(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Float64(x), GcValue::Float64(y)) => set_reg!(dst, GcValue::Float64(x / y)),
                    (GcValue::Float32(x), GcValue::Float32(y)) => set_reg!(dst, GcValue::Float32(x / y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Float".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            // === Comparison ===
            Instruction::LtInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => set_reg!(dst, GcValue::Bool(x < y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::LeInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => set_reg!(dst, GcValue::Bool(x <= y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::GtInt(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                match (va, vb) {
                    (GcValue::Int64(x), GcValue::Int64(y)) => set_reg!(dst, GcValue::Bool(x > y)),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Int".to_string(),
                            found: "other".to_string(),
                        })
                    }
                }
            }

            Instruction::Eq(dst, a, b) => {
                let va = get_reg!(a);
                let vb = get_reg!(b);
                let eq = self
                    .scheduler
                    .with_process(pid, |proc| proc.heap.gc_values_equal(&va, &vb))
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                set_reg!(dst, GcValue::Bool(eq));
            }

            // === Logical ===
            Instruction::Not(dst, src) => {
                let val = get_reg!(src);
                let result = match val {
                    GcValue::Bool(b) => GcValue::Bool(!b),
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Bool".to_string(),
                            found: "other".to_string(),
                        })
                    }
                };
                set_reg!(dst, result);
            }

            // === Control flow ===
            Instruction::Jump(offset) => {
                self.scheduler.with_process_mut(pid, |proc| {
                    if let Some(frame) = proc.frames.last_mut() {
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                    if offset < 0 {
                        proc.consume_reductions(1);
                    }
                });
            }

            Instruction::JumpIfTrue(cond, offset) => {
                let val = get_reg!(cond);
                if val.is_truthy() {
                    self.scheduler.with_process_mut(pid, |proc| {
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.ip = (frame.ip as isize + offset as isize) as usize;
                        }
                        if offset < 0 {
                            proc.consume_reductions(1);
                        }
                    });
                }
            }

            Instruction::JumpIfFalse(cond, offset) => {
                let val = get_reg!(cond);
                if !val.is_truthy() {
                    self.scheduler.with_process_mut(pid, |proc| {
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.ip = (frame.ip as isize + offset as isize) as usize;
                        }
                        if offset < 0 {
                            proc.consume_reductions(1);
                        }
                    });
                }
            }

            // === Function calls ===
            Instruction::Return(src) => {
                let return_val = get_reg!(src);

                let result = self.scheduler.with_process_mut(pid, |proc| {
                    let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                    proc.frames.pop();

                    if proc.frames.is_empty() {
                        return Some(ProcessResult::Finished(return_val.clone()));
                    }

                    if let Some(ret_reg) = return_reg {
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.registers[ret_reg as usize] = return_val.clone();
                        }
                    }

                    Some(ProcessResult::Continue)
                });

                match result {
                    Some(Some(r)) => return Ok(r),
                    _ => return Err(RuntimeError::Panic("Process not found".to_string())),
                }
            }

            Instruction::CallByName(dst, name_idx, ref arg_regs) => {
                let name = match constants.get(name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = self
                    .scheduler
                    .functions
                    .read()
                    .get(&*name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Track call count for JIT (hot function detection)
                self.scheduler.jit_tracker.record_call(&name);

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::TailCallByName(name_idx, ref arg_regs) => {
                let name = match constants.get(name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(RuntimeError::Panic("Invalid function name".to_string())),
                };

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = self
                    .scheduler
                    .functions
                    .read()
                    .get(&*name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

                // Track call count for JIT (hot function detection)
                self.scheduler.jit_tracker.record_call(&name);

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::TailCallDirect(func_idx, ref arg_regs) => {
                // Check for JIT-compiled version (1-arg int function)
                if arg_regs.len() == 1 {
                    if let Some(jit_fn) = self.scheduler.jit_int_functions.read().get(&func_idx).copied() {
                        let arg = get_reg!(arg_regs[0]);
                        if let GcValue::Int64(n) = arg {
                            let result = jit_fn(n);
                            // For tail call: pop current frame, set result in parent
                            self.scheduler.with_process_mut(pid, |proc| {
                                let return_reg = proc.frames.last().and_then(|f| f.return_reg);
                                proc.frames.pop();
                                if let Some(dst) = return_reg {
                                    if let Some(parent) = proc.frames.last_mut() {
                                        parent.registers[dst as usize] = GcValue::Int64(result);
                                    }
                                }
                            });
                            return Ok(ProcessResult::Continue);
                        }
                    }
                }

                // Fall back to interpreted execution
                let func = self
                    .scheduler
                    .function_list
                    .read()
                    .get(func_idx as usize)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("index {}", func_idx)))?;

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::TailCallSelf(ref arg_regs) => {
                // Self tail-recursion: reuse current frame
                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();
                self.scheduler.with_process_mut(pid, |proc| {
                    let frame = proc.frames.last_mut().unwrap();
                    let reg_count = frame.function.code.register_count;
                    frame.ip = 0;
                    frame.registers.clear();
                    frame.registers.resize(reg_count, GcValue::Unit);
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < reg_count {
                            frame.registers[i] = arg;
                        }
                    }
                });
            }

            Instruction::CallDirect(dst, func_idx, ref arg_regs) => {
                // Check for JIT-compiled version (1-arg int function)
                if arg_regs.len() == 1 {
                    if let Some(jit_fn) = self.scheduler.jit_int_functions.read().get(&func_idx).copied() {
                        let arg = get_reg!(arg_regs[0]);
                        if let GcValue::Int64(n) = arg {
                            let result = jit_fn(n);
                            set_reg!(dst, GcValue::Int64(result));
                            return Ok(ProcessResult::Continue);
                        }
                    }
                }

                // Fall back to interpreted execution
                let func = self
                    .scheduler
                    .function_list
                    .read()
                    .get(func_idx as usize)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("index {}", func_idx)))?;

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::CallSelf(dst, ref arg_regs) => {
                // Self-recursion: call the same function
                let func = self
                    .scheduler
                    .with_process(pid, |proc| proc.frames.last().map(|f| f.function.clone()))
                    .flatten()
                    .ok_or_else(|| RuntimeError::Panic("No frame for CallSelf".to_string()))?;

                let arg_values: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::Call(dst, func_reg, ref args) => {
                let func_val = get_reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|&r| get_reg!(r)).collect();

                let func_result = self
                    .scheduler
                    .with_process(pid, |proc| {
                        match &func_val {
                            GcValue::Function(f) => Ok((f.clone(), Vec::new())),
                            GcValue::Closure(ptr) => {
                                let closure = proc.heap.get_closure(*ptr).ok_or_else(|| {
                                    RuntimeError::TypeError {
                                        expected: "Closure".to_string(),
                                        found: "invalid".to_string(),
                                    }
                                })?;
                                Ok((closure.function.clone(), closure.captures.clone()))
                            }
                            other => Err(RuntimeError::TypeError {
                                expected: "Function".to_string(),
                                found: other.type_name(&proc.heap).to_string(),
                            }),
                        }
                    })
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                let (func, captures) = func_result;

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::TailCall(func_reg, ref args) => {
                let func_val = get_reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|&r| get_reg!(r)).collect();

                let func_result = self
                    .scheduler
                    .with_process(pid, |proc| {
                        match &func_val {
                            GcValue::Function(f) => Ok((f.clone(), Vec::new())),
                            GcValue::Closure(ptr) => {
                                let closure = proc.heap.get_closure(*ptr).ok_or_else(|| {
                                    RuntimeError::TypeError {
                                        expected: "Closure".to_string(),
                                        found: "invalid".to_string(),
                                    }
                                })?;
                                Ok((closure.function.clone(), closure.captures.clone()))
                            }
                            other => Err(RuntimeError::TypeError {
                                expected: "Function".to_string(),
                                found: other.type_name(&proc.heap).to_string(),
                            }),
                        }
                    })
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                let (func, captures) = func_result;

                self.scheduler.with_process_mut(pid, |proc| {
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
                });
            }

            Instruction::CallNative(dst, name_idx, ref args) => {
                let arg_values: Vec<GcValue> = args.iter().map(|&r| get_reg!(r)).collect();

                let name = match constants.get(name_idx as usize) {
                    Some(Value::String(s)) => s.clone(),
                    _ => {
                        return Err(RuntimeError::Panic(
                            "Invalid native function name".to_string(),
                        ))
                    }
                };

                let native = self
                    .scheduler
                    .natives
                    .read()
                    .get(&*name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::Panic(format!("Undefined native: {}", name)))?;

                let result = self
                    .scheduler
                    .with_process_mut(pid, |proc| (native.func)(&arg_values, &mut proc.heap))
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                set_reg!(dst, result);
                self.scheduler.with_process_mut(pid, |proc| {
                    proc.consume_reductions(10);
                });
            }

            // === Collections ===
            Instruction::MakeList(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| get_reg!(r)).collect();
                let ptr = self
                    .scheduler
                    .with_process_mut(pid, |proc| proc.heap.alloc_list(items))
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                set_reg!(dst, GcValue::List(ptr));
            }

            Instruction::MakeTuple(dst, ref elements) => {
                let items: Vec<GcValue> = elements.iter().map(|&r| get_reg!(r)).collect();
                let ptr = self
                    .scheduler
                    .with_process_mut(pid, |proc| proc.heap.alloc_tuple(items))
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;
                set_reg!(dst, GcValue::Tuple(ptr));
            }

            Instruction::GetCapture(dst, idx) => {
                let val = self
                    .scheduler
                    .with_process(pid, |proc| {
                        proc.frames
                            .last()
                            .and_then(|f| f.captures.get(idx as usize))
                            .cloned()
                            .unwrap_or(GcValue::Unit)
                    })
                    .unwrap_or(GcValue::Unit);
                set_reg!(dst, val);
            }

            // === Concurrency ===
            Instruction::SelfPid(dst) => {
                set_reg!(dst, GcValue::Pid(pid.0));
            }

            Instruction::Spawn(dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = self
                    .scheduler
                    .with_process(pid, |proc| {
                        match &func_val {
                            GcValue::Function(f) => Ok(f.clone()),
                            GcValue::Closure(ptr) => proc
                                .heap
                                .get_closure(*ptr)
                                .map(|c| c.function.clone())
                                .ok_or_else(|| RuntimeError::TypeError {
                                    expected: "Function".to_string(),
                                    found: "invalid closure".to_string(),
                                }),
                            other => Err(RuntimeError::TypeError {
                                expected: "Function".to_string(),
                                found: other.type_name(&proc.heap).to_string(),
                            }),
                        }
                    })
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))??;

                let child_pid = self.spawn_child_process(func, args, pid);
                set_reg!(dst, GcValue::Pid(child_pid.0));

                self.scheduler.with_process_mut(pid, |proc| {
                    proc.consume_reductions(100);
                });
            }

            Instruction::Send(pid_reg, msg_reg) => {
                let target_val = get_reg!(pid_reg);
                let message = get_reg!(msg_reg);

                let target_pid = match &target_val {
                    GcValue::Pid(p) => Pid(*p),
                    other => {
                        let type_name = self
                            .scheduler
                            .with_process(pid, |proc| other.type_name(&proc.heap).to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        return Err(RuntimeError::TypeError {
                            expected: "Pid".to_string(),
                            found: type_name,
                        });
                    }
                };

                self.scheduler.send(pid, target_pid, message)?;

                // Wake up target if on our local queue or push to injector
                // The scheduler's send already wakes the process

                self.scheduler.with_process_mut(pid, |proc| {
                    proc.consume_reductions(10);
                });
            }

            Instruction::Receive(dst) => {
                let received = self
                    .scheduler
                    .with_process_mut(pid, |proc| proc.try_receive())
                    .ok_or_else(|| RuntimeError::Panic("Process not found".to_string()))?;

                if let Some(msg) = received {
                    set_reg!(dst, msg);
                } else {
                    self.scheduler.with_process_mut(pid, |proc| {
                        proc.wait_for_message();
                        if let Some(frame) = proc.frames.last_mut() {
                            frame.ip -= 1;
                        }
                    });
                    return Ok(ProcessResult::Waiting);
                }
            }

            Instruction::SpawnLink(dst, func_reg, ref arg_regs) => {
                let func_val = get_reg!(func_reg);
                let args: Vec<GcValue> = arg_regs.iter().map(|&r| get_reg!(r)).collect();

                let func = match func_val {
                    GcValue::Function(f) => f,
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: "other".to_string(),
                        })
                    }
                };

                let child_pid = self.spawn_child_process(func, args, pid);

                // Create bidirectional link
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

                let func = match func_val {
                    GcValue::Function(f) => f,
                    _ => {
                        return Err(RuntimeError::TypeError {
                            expected: "Function".to_string(),
                            found: "other".to_string(),
                        })
                    }
                };

                let child_pid = self.spawn_child_process(func, args, pid);
                let ref_id = self.scheduler.make_ref();

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

            Instruction::ReceiveTimeout(_dst, _timeout_reg) => {
                return Err(RuntimeError::Panic(
                    "ReceiveTimeout not yet implemented".to_string(),
                ));
            }

            // === Debug ===
            Instruction::Nop => {}

            // === Pattern Matching ===
            Instruction::TestConst(dst, value_reg, const_idx) => {
                let value = get_reg!(value_reg);
                let constant = &constants[const_idx as usize];
                self.scheduler.with_process_mut(pid, |proc| {
                    let gc_const = proc.heap.value_to_gc(constant);
                    let result = proc.heap.gc_values_equal(&value, &gc_const);
                    proc.frames.last_mut().unwrap().registers[dst as usize] = GcValue::Bool(result);
                });
            }

            Instruction::Throw(msg_reg) => {
                let msg = get_reg!(msg_reg);
                let msg_str = self
                    .scheduler
                    .with_process(pid, |proc| proc.heap.display_value(&msg))
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(RuntimeError::Panic(format!("Panic: {}", msg_str)));
            }

            Instruction::DebugPrint(r) => {
                let value = get_reg!(r);
                println!("DEBUG: {:?}", value);
                self.scheduler.with_process_mut(pid, |proc| {
                    proc.output.push(format!("{:?}", value));
                });
            }

            // Unhandled
            other => {
                return Err(RuntimeError::Panic(format!(
                    "Instruction {:?} not yet implemented in Worker",
                    other
                )));
            }
        }

        Ok(ProcessResult::Continue)
    }

    /// Spawn a child process and add it to the work queue.
    /// Uses lightweight heap for memory-efficient mass spawning.
    fn spawn_child_process(
        &self,
        func: Arc<FunctionValue>,
        args: Vec<GcValue>,
        parent_pid: Pid,
    ) -> Pid {
        // Use spawn_lightweight to avoid double-queuing and minimize memory
        let child_pid = self.scheduler.spawn_lightweight();

        // Copy arguments from parent to child with lock ordering
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

            let copied_args: Vec<GcValue> = args
                .iter()
                .map(|arg| child.heap.deep_copy(arg, &parent.heap))
                .collect();

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

        // Add to our local queue (locality: spawned processes likely share data)
        self.local.push(child_pid);

        child_pid
    }
}

/// Result of batch execution (internal use).
/// Used by execute_batch_locked to signal what to do next.
enum BatchResult {
    /// Continue with next batch
    Continue,
    /// Need to execute an instruction via scheduler (has instruction + constants)
    NeedsScheduler(crate::value::Instruction, Vec<crate::value::Value>),
    /// Process should yield (end of time slice)
    Yield,
    /// Process finished with value
    Finished(GcValue),
    /// Process is waiting for message
    Waiting,
}

/// Result of executing a process for one step.
#[derive(Debug, Clone)]
pub enum ProcessResult {
    /// Continue execution
    Continue,
    /// Process finished with value
    Finished(GcValue),
    /// Process is waiting (for message, timer, etc.)
    Waiting,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::GcNativeFn;
    use crate::value::{Chunk, Instruction, Value};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    fn make_function_with_consts(
        name: &str,
        code: Vec<Instruction>,
        constants: Vec<Value>,
    ) -> Arc<FunctionValue> {
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

    fn setup_scheduler_with_builtins() -> Scheduler {
        let scheduler = Scheduler::new();

        // Register builtins
        scheduler.natives.write().insert(
            "println".to_string(),
            Arc::new(GcNativeFn {
                name: "println".to_string(),
                arity: 1,
                func: Box::new(|args, heap| {
                    let s = heap.display_value(&args[0]);
                    println!("{}", s);
                    Ok(GcValue::Unit)
                }),
            }),
        );

        scheduler
    }

    #[test]
    fn test_worker_pool_single_process() {
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::single_threaded(scheduler);

        let func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::Return(0),
            ],
            vec![Value::Int64(42)],
        );

        pool.spawn_initial(func);
        let result = pool.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(42)));
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_recursive() {
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::single_threaded(scheduler);

        // sum(n) = if n <= 0 then 0 else n + sum(n-1)
        let sum_func = make_function_with_consts(
            "sum",
            vec![
                Instruction::LoadConst(1, 0),           // r1 = 0
                Instruction::LeInt(2, 0, 1),            // r2 = (n <= 0)
                Instruction::JumpIfFalse(2, 2),         // if not, skip to [5]
                Instruction::Move(3, 1),                // r3 = 0
                Instruction::Jump(4),                   // skip to Return at [9]
                Instruction::LoadConst(4, 1),           // r4 = 1
                Instruction::SubInt(5, 0, 4),           // r5 = n - 1
                Instruction::CallByName(6, 2, vec![5].into()), // r6 = sum(n-1)
                Instruction::AddInt(3, 0, 6),           // r3 = n + sum(n-1)
                Instruction::Return(3),
            ],
            vec![
                Value::Int64(0),
                Value::Int64(1),
                Value::String(Arc::new("sum".to_string())),
            ],
        );
        let sum_func = Arc::new(FunctionValue {
            arity: 1,
            ..(*sum_func).clone()
        });

        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::CallByName(1, 1, vec![0].into()),
                Instruction::Return(1),
            ],
            vec![
                Value::Int64(10),
                Value::String(Arc::new("sum".to_string())),
            ],
        );

        pool.scheduler().functions.write().insert("sum".to_string(), sum_func);
        pool.spawn_initial(main_func);

        let result = pool.run().unwrap();
        // sum(10) = 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 + 0 = 55
        assert_eq!(result, Some(GcValue::Int64(55)));
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_multi_worker() {
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::new(
            scheduler,
            WorkerPoolConfig {
                num_workers: 2,
                ..Default::default()
            },
        );

        let func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::Return(0),
            ],
            vec![Value::Int64(123)],
        );

        pool.spawn_initial(func);
        let result = pool.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(123)));
        assert_eq!(pool.num_workers(), 2);
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_spawn_child() {
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::new(
            scheduler,
            WorkerPoolConfig {
                num_workers: 2,
                ..Default::default()
            },
        );

        // Child function that returns argument + 1
        let child_func = make_function_with_consts(
            "child",
            vec![
                Instruction::LoadConst(1, 0), // r1 = 1
                Instruction::AddInt(2, 0, 1), // r2 = r0 + 1
                Instruction::Return(2),
            ],
            vec![Value::Int64(1)],
        );
        let child_func = Arc::new(FunctionValue {
            arity: 1,
            ..(*child_func).clone()
        });

        // Main function that spawns child and returns immediately
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0), // r0 = child_func
                Instruction::LoadConst(1, 1), // r1 = 10
                Instruction::Spawn(2, 0, vec![1].into()), // r2 = spawn(child, [10])
                Instruction::LoadConst(3, 2), // r3 = 42
                Instruction::Return(3),
            ],
            vec![
                Value::Function(child_func),
                Value::Int64(10),
                Value::Int64(42),
            ],
        );

        pool.spawn_initial(main_func);
        let result = pool.run().unwrap();

        // Main process returns 42
        assert_eq!(result, Some(GcValue::Int64(42)));
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_tail_call() {
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::single_threaded(scheduler);

        // countdown(n) = if n <= 0 then 0 else countdown(n-1)
        // Uses tail call for constant stack space
        let countdown_func = make_function_with_consts(
            "countdown",
            vec![
                Instruction::LoadConst(1, 0),             // r1 = 0
                Instruction::LeInt(2, 0, 1),              // r2 = (n <= 0)
                Instruction::JumpIfFalse(2, 2),           // if not, skip to [5]
                Instruction::Move(3, 1),                  // r3 = 0
                Instruction::Return(3),
                Instruction::LoadConst(4, 1),             // r4 = 1
                Instruction::SubInt(5, 0, 4),             // r5 = n - 1
                Instruction::TailCallByName(2, vec![5].into()),  // tail call countdown(n-1)
            ],
            vec![
                Value::Int64(0),
                Value::Int64(1),
                Value::String(Arc::new("countdown".to_string())),
            ],
        );
        let countdown_func = Arc::new(FunctionValue {
            arity: 1,
            ..(*countdown_func).clone()
        });

        // main: countdown(1000)
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::CallByName(1, 1, vec![0].into()),
                Instruction::Return(1),
            ],
            vec![
                Value::Int64(1000),
                Value::String(Arc::new("countdown".to_string())),
            ],
        );

        pool.scheduler().functions.write().insert("countdown".to_string(), countdown_func);
        pool.spawn_initial(main_func);

        let result = pool.run().unwrap();
        assert_eq!(result, Some(GcValue::Int64(0)));
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_message_passing() {
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::new(
            scheduler,
            WorkerPoolConfig {
                num_workers: 2,
                ..Default::default()
            },
        );

        // Child: receives pid as arg, sends 42 back
        let child_func = make_function_with_consts(
            "child",
            vec![
                // r0 = parent_pid (arg)
                Instruction::LoadConst(1, 0), // r1 = 42
                Instruction::Send(0, 1),      // send(parent_pid, 42)
                Instruction::LoadUnit(2),
                Instruction::Return(2),
            ],
            vec![Value::Int64(42)],
        );

        // Main: spawn child with self pid, receive message
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::SelfPid(0),           // r0 = self()
                Instruction::LoadConst(1, 0),      // r1 = child_func
                Instruction::Spawn(2, 1, vec![0].into()), // r2 = spawn(child, [self()])
                Instruction::Receive(3),           // r3 = receive()
                Instruction::Return(3),
            ],
            vec![Value::Function(child_func)],
        );

        pool.spawn_initial(main_func);
        let result = pool.run().unwrap();

        assert_eq!(result, Some(GcValue::Int64(42)));
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_fib() {
        // Test Fibonacci to ensure multiple workers can handle
        // compute-intensive workloads
        let scheduler = setup_scheduler_with_builtins();
        let pool = WorkerPool::new(
            scheduler,
            WorkerPoolConfig {
                num_workers: 4,
                ..Default::default()
            },
        );

        // fib(n) = if n < 2 then n else fib(n-1) + fib(n-2)
        let fib_func = make_function_with_consts(
            "fib",
            vec![
                // [0] LoadConst(1, 0)       r1 = 2
                Instruction::LoadConst(1, 0),
                // [1] LtInt(2, 0, 1)        r2 = (n < 2)
                Instruction::LtInt(2, 0, 1),
                // [2] JumpIfFalse(2, 2)     if not, skip to [5]
                Instruction::JumpIfFalse(2, 2),
                // [3] Move(3, 0)            r3 = n (base case)
                Instruction::Move(3, 0),
                // [4] Jump(7)               skip to Return at [12]
                Instruction::Jump(7),
                // [5] LoadConst(4, 1)       r4 = 1
                Instruction::LoadConst(4, 1),
                // [6] SubInt(5, 0, 4)       r5 = n - 1
                Instruction::SubInt(5, 0, 4),
                // [7] CallByName(6, 3, [5]) r6 = fib(n-1)
                Instruction::CallByName(6, 3, vec![5].into()),
                // [8] LoadConst(7, 0)       r7 = 2
                Instruction::LoadConst(7, 0),
                // [9] SubInt(8, 0, 7)       r8 = n - 2
                Instruction::SubInt(8, 0, 7),
                // [10] CallByName(9, 3, [8]) r9 = fib(n-2)
                Instruction::CallByName(9, 3, vec![8].into()),
                // [11] AddInt(3, 6, 9)      r3 = fib(n-1) + fib(n-2)
                Instruction::AddInt(3, 6, 9),
                // [12] Return(3)
                Instruction::Return(3),
            ],
            vec![
                Value::Int64(2),
                Value::Int64(1),
                Value::Int64(2),
                Value::String(Arc::new("fib".to_string())),
            ],
        );
        let fib_func = Arc::new(FunctionValue {
            arity: 1,
            ..(*fib_func).clone()
        });

        // main: fib(15)
        let main_func = make_function_with_consts(
            "main",
            vec![
                Instruction::LoadConst(0, 0),
                Instruction::CallByName(1, 1, vec![0].into()),
                Instruction::Return(1),
            ],
            vec![
                Value::Int64(15),
                Value::String(Arc::new("fib".to_string())),
            ],
        );

        pool.scheduler().functions.write().insert("fib".to_string(), fib_func);
        pool.spawn_initial(main_func);

        let result = pool.run().unwrap();
        // fib(15) = 610
        assert_eq!(result, Some(GcValue::Int64(610)));
        pool.shutdown();
    }
}
