//! Async VM implementation using Tokio.
//!
//! Each process runs as a tokio task, enabling:
//! - Natural async/await for blocking operations (mvar locks, receive, I/O)
//! - Work-stealing across CPU cores via tokio's multi-threaded runtime
//! - Proper yielding without manual try_lock + retry loops
//!
//! JIT Compatibility:
//! - JIT-compiled code runs synchronously within a task
//! - For blocking ops, JIT returns control to interpreter which does .await
//! - yield_now() is called every N instructions for fairness

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, RwLock as TokioRwLock, OwnedRwLockReadGuard, OwnedRwLockWriteGuard};
use tokio::task::LocalSet;

use crate::gc::{GcConfig, GcValue, Heap, GcNativeFn};

/// Held mvar lock guard (owned so it can be stored).
pub enum HeldMvarLock {
    Read(OwnedRwLockReadGuard<ThreadSafeValue>),
    Write(OwnedRwLockWriteGuard<ThreadSafeValue>),
}
use crate::process::{CallFrame, ExceptionHandler, ProcessState, ThreadSafeValue};
use crate::value::{FunctionValue, Pid, TypeValue, RefId, RuntimeError, Value};
use crate::parallel::SendableValue;
use crate::io_runtime::IoRuntime;

/// Reductions per yield (how often we call yield_now for fairness).
const REDUCTIONS_PER_YIELD: usize = 1000;

/// Type for process mailbox sender.
pub type MailboxSender = mpsc::UnboundedSender<ThreadSafeValue>;
/// Type for process mailbox receiver.
pub type MailboxReceiver = mpsc::UnboundedReceiver<ThreadSafeValue>;

/// Shared state across all async processes.
pub struct AsyncSharedState {
    /// Global functions (read-only after startup).
    pub functions: HashMap<String, Arc<FunctionValue>>,
    /// Function list for indexed calls.
    pub function_list: Vec<Arc<FunctionValue>>,
    /// Native functions.
    pub natives: HashMap<String, Arc<GcNativeFn>>,
    /// Type definitions.
    pub types: HashMap<String, Arc<TypeValue>>,

    /// JIT-compiled functions (by arity).
    pub jit_int_functions: HashMap<u16, crate::parallel::JitIntFn>,
    pub jit_int_functions_0: HashMap<u16, crate::parallel::JitIntFn0>,
    pub jit_int_functions_2: HashMap<u16, crate::parallel::JitIntFn2>,
    pub jit_int_functions_3: HashMap<u16, crate::parallel::JitIntFn3>,
    pub jit_int_functions_4: HashMap<u16, crate::parallel::JitIntFn4>,
    pub jit_loop_array_functions: HashMap<u16, crate::parallel::JitLoopArrayFn>,

    /// Shutdown signal.
    pub shutdown: AtomicBool,

    /// Process registry: Pid -> mailbox sender.
    /// Protected by tokio RwLock for async access.
    pub process_registry: TokioRwLock<HashMap<Pid, MailboxSender>>,

    /// Module-level mutable variables (mvars).
    /// Key is "module_name.var_name", value is protected by tokio RwLock.
    pub mvars: HashMap<String, Arc<TokioRwLock<ThreadSafeValue>>>,

    /// Dynamic mvars from eval() - uses std RwLock for compatibility with ParallelVM API.
    pub dynamic_mvars: Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>>,

    /// Dynamic functions from eval() - uses std RwLock for compatibility.
    pub dynamic_functions: Arc<RwLock<HashMap<String, Arc<FunctionValue>>>>,

    /// Dynamic types from eval() - uses std RwLock for compatibility.
    pub dynamic_types: Arc<RwLock<HashMap<String, Arc<TypeValue>>>>,

    /// Stdlib functions - uses std RwLock for compatibility.
    pub stdlib_functions: Arc<RwLock<HashMap<String, Arc<FunctionValue>>>>,

    /// Stdlib types - uses std RwLock for compatibility.
    pub stdlib_types: Arc<RwLock<HashMap<String, Arc<TypeValue>>>>,

    /// Prelude imports - uses std RwLock for compatibility.
    pub prelude_imports: Arc<RwLock<HashMap<String, String>>>,

    /// Eval callback - uses std RwLock for compatibility.
    pub eval_callback: Arc<RwLock<Option<Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>>>>,

    /// IO request sender.
    pub io_sender: Option<mpsc::UnboundedSender<crate::io_runtime::IoRequest>>,

    /// PID counter for generating unique PIDs.
    pub next_pid: AtomicU64,
}

impl AsyncSharedState {
    /// Allocate a new unique PID.
    pub fn alloc_pid(&self) -> Pid {
        Pid(self.next_pid.fetch_add(1, Ordering::SeqCst))
    }

    /// Register a process in the registry.
    pub async fn register_process(&self, pid: Pid, sender: MailboxSender) {
        self.process_registry.write().await.insert(pid, sender);
    }

    /// Unregister a process from the registry.
    pub async fn unregister_process(&self, pid: Pid) {
        self.process_registry.write().await.remove(&pid);
    }

    /// Send a message to a process by PID.
    pub async fn send_message(&self, target_pid: Pid, message: ThreadSafeValue) -> bool {
        if let Some(sender) = self.process_registry.read().await.get(&target_pid) {
            sender.send(message).is_ok()
        } else {
            false // Process not found or dead
        }
    }
}

/// An async process (runs as a tokio task).
pub struct AsyncProcess {
    /// Unique process identifier.
    pub pid: Pid,

    /// Process-local garbage-collected heap.
    pub heap: Heap,

    /// Call stack frames.
    pub frames: Vec<CallFrame>,

    /// Pool of reusable register vectors.
    pub register_pool: Vec<Vec<GcValue>>,

    /// Mailbox receiver for incoming messages.
    pub mailbox: MailboxReceiver,

    /// Mailbox sender (for cloning to other processes).
    pub mailbox_sender: MailboxSender,

    /// Current state.
    pub state: ProcessState,

    /// Instruction count since last yield.
    pub instructions_since_yield: usize,

    /// Linked processes.
    pub links: Vec<Pid>,

    /// Monitors.
    pub monitors: HashMap<RefId, Pid>,

    /// Processes monitoring this one.
    pub monitored_by: HashMap<RefId, Pid>,

    /// Exception handlers stack.
    pub handlers: Vec<ExceptionHandler>,

    /// Current exception.
    pub current_exception: Option<GcValue>,

    /// Exit value.
    pub exit_value: Option<GcValue>,

    /// Output buffer.
    pub output: Vec<String>,

    /// When this process was created.
    pub started_at: Instant,

    /// Shared state reference.
    pub shared: Arc<AsyncSharedState>,

    /// Held mvar locks (name -> stack of guards for reentrant locking).
    pub held_mvar_locks: HashMap<String, Vec<HeldMvarLock>>,
}

impl AsyncProcess {
    /// Create a new async process.
    pub fn new(pid: Pid, shared: Arc<AsyncSharedState>) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            pid,
            heap: Heap::with_config(GcConfig::default()),
            frames: Vec::new(),
            register_pool: Vec::new(),
            mailbox: receiver,
            mailbox_sender: sender,
            state: ProcessState::Running,
            instructions_since_yield: 0,
            links: Vec::new(),
            monitors: HashMap::new(),
            monitored_by: HashMap::new(),
            handlers: Vec::new(),
            current_exception: None,
            exit_value: None,
            output: Vec::new(),
            started_at: Instant::now(),
            shared,
            held_mvar_locks: HashMap::new(),
        }
    }

    /// Maybe yield to other tasks for fairness.
    /// Called periodically during instruction execution.
    #[inline]
    pub async fn maybe_yield(&mut self) {
        self.instructions_since_yield += 1;
        if self.instructions_since_yield >= REDUCTIONS_PER_YIELD {
            self.instructions_since_yield = 0;
            tokio::task::yield_now().await;
        }
    }

    /// Read from an mvar (async - yields if locked).
    pub async fn mvar_read(&mut self, name: &str) -> Result<GcValue, RuntimeError> {
        // Check if we already hold a lock on this mvar
        if let Some(guards) = self.held_mvar_locks.get(name) {
            if let Some(guard) = guards.last() {
                // Read from held guard
                let value: ThreadSafeValue = match guard {
                    HeldMvarLock::Read(g) => (**g).clone(),
                    HeldMvarLock::Write(g) => (**g).clone(),
                };
                let gc_value = value.to_gc_value(&mut self.heap);
                return Ok(gc_value);
            }
        }

        let var = self.shared.mvars.get(name)
            .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?;

        // Async read - will yield if write-locked
        let guard = var.read().await;
        let gc_value = guard.to_gc_value(&mut self.heap);
        Ok(gc_value)
    }

    /// Write to an mvar (async - yields if locked).
    pub async fn mvar_write(&mut self, name: &str, value: GcValue) -> Result<(), RuntimeError> {
        let safe_value = ThreadSafeValue::from_gc_value(&value, &self.heap)
            .ok_or_else(|| RuntimeError::Panic(format!("Cannot convert value for mvar: {}", name)))?;

        // Check if we already hold a WRITE lock on this mvar
        if let Some(guards) = self.held_mvar_locks.get_mut(name) {
            if let Some(guard) = guards.last_mut() {
                match guard {
                    HeldMvarLock::Write(g) => {
                        **g = safe_value;
                        return Ok(());
                    }
                    HeldMvarLock::Read(_) => {
                        // Have read lock but need write - fall through to acquire
                    }
                }
            }
        }

        let var = self.shared.mvars.get(name)
            .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?;

        // Async write - will yield if any lock held
        let mut guard = var.write().await;
        *guard = safe_value;
        Ok(())
    }

    /// Receive a message (async - yields until message arrives).
    pub async fn receive(&mut self) -> Option<GcValue> {
        match self.mailbox.recv().await {
            Some(msg) => Some(msg.to_gc_value(&mut self.heap)),
            None => None, // Channel closed
        }
    }

    /// Receive a message with timeout.
    pub async fn receive_timeout(&mut self, timeout: Duration) -> Option<GcValue> {
        match tokio::time::timeout(timeout, self.mailbox.recv()).await {
            Ok(Some(msg)) => Some(msg.to_gc_value(&mut self.heap)),
            Ok(None) => None, // Channel closed
            Err(_) => None, // Timeout
        }
    }

    /// Try to receive without blocking (for non-blocking checks).
    pub fn try_receive(&mut self) -> Option<GcValue> {
        match self.mailbox.try_recv() {
            Ok(msg) => Some(msg.to_gc_value(&mut self.heap)),
            Err(_) => None,
        }
    }

    /// Send a message to another process.
    pub async fn send(&self, target_pid: Pid, value: GcValue) -> bool {
        let safe_value = match ThreadSafeValue::from_gc_value(&value, &self.heap) {
            Some(v) => v,
            None => return false,
        };
        self.shared.send_message(target_pid, safe_value).await
    }

    /// Spawn a new process running the given function.
    /// Returns the new process's PID.
    ///
    /// Note: This must be called from within a LocalSet context because
    /// GcValue contains raw pointers that aren't Send.
    pub async fn spawn_process(
        shared: Arc<AsyncSharedState>,
        function: Arc<FunctionValue>,
        args: Vec<GcValue>,
        parent_heap: &Heap,
    ) -> Pid {
        let pid = shared.alloc_pid();
        let shared_clone = shared.clone();

        // Convert args to thread-safe values (deep copy)
        let safe_args: Vec<ThreadSafeValue> = args.iter()
            .filter_map(|v| ThreadSafeValue::from_gc_value(v, parent_heap))
            .collect();

        // Create process and register it
        let mut process = AsyncProcess::new(pid, shared_clone.clone());
        let sender = process.mailbox_sender.clone();
        shared.register_process(pid, sender).await;

        // Spawn as local task (stays on current thread - required for GcValue)
        tokio::task::spawn_local(async move {
            // Set up initial call frame
            let args_gc: Vec<GcValue> = safe_args.iter()
                .map(|v| v.to_gc_value(&mut process.heap))
                .collect();

            let mut registers = vec![GcValue::Unit; function.code.register_count as usize];
            for (i, arg) in args_gc.into_iter().enumerate() {
                if i < registers.len() {
                    registers[i] = arg;
                }
            }

            process.frames.push(CallFrame {
                function,
                ip: 0,
                registers,
                captures: Vec::new(),
                return_reg: None,
            });

            // Run the process
            let result = process.run().await;

            // Unregister on exit
            shared_clone.unregister_process(pid).await;

            result
        });

        pid
    }

    /// Main execution loop for this process.
    pub async fn run(&mut self) -> Result<GcValue, RuntimeError> {
        loop {
            // Check shutdown
            if self.shared.shutdown.load(Ordering::SeqCst) {
                return Ok(GcValue::Unit);
            }

            // Check if we have frames to execute
            if self.frames.is_empty() {
                return Ok(self.exit_value.take().unwrap_or(GcValue::Unit));
            }

            // Execute one instruction
            match self.step().await {
                Ok(StepResult::Continue) => {
                    // Maybe yield for fairness
                    self.maybe_yield().await;
                }
                Ok(StepResult::Finished(value)) => {
                    return Ok(value);
                }
                Err(e) => {
                    // Try to handle exception
                    if !self.handle_exception(&e) {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Execute one instruction.
    async fn step(&mut self) -> Result<StepResult, RuntimeError> {
        use crate::value::Instruction::*;

        // Get frame info without holding mutable borrow
        let (code_len, ip) = {
            let frame = match self.frames.last() {
                Some(f) => f,
                None => return Ok(StepResult::Finished(GcValue::Unit)),
            };
            (frame.function.code.code.len(), frame.ip)
        };

        if ip >= code_len {
            return Ok(StepResult::Finished(GcValue::Unit));
        }

        // Get instruction reference and constants - use indices to avoid borrow issues
        let instruction = {
            let frame = self.frames.last().unwrap();
            frame.function.code.code[ip].clone()
        };

        // Increment IP
        self.frames.last_mut().unwrap().ip += 1;

        // Helper macros for register access - use direct indexing
        macro_rules! reg {
            ($r:expr) => {{
                let frame = self.frames.last().unwrap();
                frame.registers[$r as usize].clone()
            }};
        }
        macro_rules! set_reg {
            ($r:expr, $v:expr) => {{
                let frame = self.frames.last_mut().unwrap();
                frame.registers[$r as usize] = $v;
            }};
        }
        macro_rules! get_const {
            ($idx:expr) => {{
                let frame = self.frames.last().unwrap();
                frame.function.code.constants[$idx as usize].clone()
            }};
        }

        match instruction {
            // === Constants and moves ===
            LoadConst(dst, idx) => {
                let value = get_const!(idx);
                let gc_value = self.heap.value_to_gc(&value);
                set_reg!(dst, gc_value);
            }
            LoadUnit(dst) => set_reg!(dst, GcValue::Unit),
            LoadTrue(dst) => set_reg!(dst, GcValue::Bool(true)),
            LoadFalse(dst) => set_reg!(dst, GcValue::Bool(false)),
            Move(dst, src) => {
                let v = reg!(src);
                set_reg!(dst, v);
            }

            // === Integer arithmetic ===
            AddInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("AddInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("AddInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.wrapping_add(vb)));
            }
            SubInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("SubInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("SubInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.wrapping_sub(vb)));
            }
            MulInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("MulInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("MulInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.wrapping_mul(vb)));
            }
            DivInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("DivInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("DivInt: expected Int64".into())) };
                if vb == 0 { return Err(RuntimeError::Panic("Division by zero".into())); }
                set_reg!(dst, GcValue::Int64(va / vb));
            }
            ModInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("ModInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("ModInt: expected Int64".into())) };
                if vb == 0 { return Err(RuntimeError::Panic("Modulo by zero".into())); }
                set_reg!(dst, GcValue::Int64(va % vb));
            }
            NegInt(dst, src) => {
                let v = match reg!(src) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("NegInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(-v));
            }
            AbsInt(dst, src) => {
                let v = match reg!(src) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("AbsInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(v.abs()));
            }
            MinInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("MinInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("MinInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.min(vb)));
            }
            MaxInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("MaxInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("MaxInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Int64(va.max(vb)));
            }

            // === Float arithmetic ===
            AddFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("AddFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("AddFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(va + vb));
            }
            SubFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("SubFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("SubFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(va - vb));
            }
            MulFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("MulFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("MulFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(va * vb));
            }
            DivFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("DivFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("DivFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(va / vb));
            }
            NegFloat(dst, src) => {
                let v = match reg!(src) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("NegFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(-v));
            }
            AbsFloat(dst, src) => {
                let v = match reg!(src) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("AbsFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(v.abs()));
            }
            SqrtFloat(dst, src) => {
                let v = match reg!(src) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("SqrtFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(v.sqrt()));
            }
            PowFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("PowFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("PowFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Float64(va.powf(vb)));
            }

            // === Float comparisons ===
            EqFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("EqFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("EqFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Bool(va == vb));
            }
            LtFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("LtFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("LtFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Bool(va < vb));
            }
            LeFloat(dst, a, b) => {
                let va = match reg!(a) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("LeFloat: expected Float64".into())) };
                let vb = match reg!(b) { GcValue::Float64(n) => n, _ => return Err(RuntimeError::Panic("LeFloat: expected Float64".into())) };
                set_reg!(dst, GcValue::Bool(va <= vb));
            }

            // === Type conversions ===
            IntToFloat(dst, src) => {
                let result = match reg!(src) {
                    GcValue::Int64(v) => GcValue::Float64(v as f64),
                    GcValue::Int32(v) => GcValue::Float64(v as f64),
                    GcValue::Float64(v) => GcValue::Float64(v),
                    GcValue::Float32(v) => GcValue::Float64(v as f64),
                    _ => return Err(RuntimeError::Panic("IntToFloat: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            FloatToInt(dst, src) => {
                let result = match reg!(src) {
                    GcValue::Float64(v) => GcValue::Int64(v as i64),
                    GcValue::Float32(v) => GcValue::Int64(v as i64),
                    GcValue::Int64(v) => GcValue::Int64(v),
                    _ => return Err(RuntimeError::Panic("FloatToInt: expected numeric".into())),
                };
                set_reg!(dst, result);
            }
            ToBigInt(dst, src) => {
                let result = match reg!(src) {
                    GcValue::Int64(v) => {
                        let bi = num_bigint::BigInt::from(v);
                        GcValue::BigInt(self.heap.alloc_bigint(bi))
                    }
                    GcValue::Int32(v) => {
                        let bi = num_bigint::BigInt::from(v);
                        GcValue::BigInt(self.heap.alloc_bigint(bi))
                    }
                    GcValue::BigInt(_) => reg!(src), // Already a BigInt
                    _ => return Err(RuntimeError::Panic("ToBigInt: expected integer".into())),
                };
                set_reg!(dst, result);
            }

            // === Comparisons ===
            EqInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("EqInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("EqInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va == vb));
            }
            LtInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("LtInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("LtInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va < vb));
            }
            LeInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("LeInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("LeInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va <= vb));
            }
            GtInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("GtInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("GtInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va > vb));
            }

            // === Control flow ===
            // Note: Jump offsets are RELATIVE - formula is: original_ip + 1 + offset
            // Since we already incremented ip before this match, the current ip is (original_ip + 1)
            // So we need: current_ip + offset = (original_ip + 1) + offset
            Jump(offset) => {
                let frame = self.frames.last_mut().unwrap();
                // frame.ip was already incremented, so current ip = original_ip + 1
                // target = original_ip + 1 + offset = current_ip + offset
                frame.ip = (frame.ip as isize + offset as isize) as usize;
            }
            JumpIfTrue(cond, offset) => {
                if matches!(reg!(cond), GcValue::Bool(true)) {
                    let frame = self.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }
            }
            JumpIfFalse(cond, offset) => {
                if matches!(reg!(cond), GcValue::Bool(false)) {
                    let frame = self.frames.last_mut().unwrap();
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }
            }

            // === Return ===
            Return(src) => {
                let value = reg!(src);
                // Get return_reg from CURRENT frame BEFORE popping
                let return_reg = self.frames.last().unwrap().return_reg;
                self.frames.pop();
                if self.frames.is_empty() {
                    return Ok(StepResult::Finished(value));
                } else if let Some(ret_reg) = return_reg {
                    // Store return value in caller's return register
                    let frame = self.frames.last_mut().unwrap();
                    frame.registers[ret_reg as usize] = value;
                }
            }

            // === Function calls ===
            CallDirect(dst, func_idx, ref args) => {
                // Check for JIT-compiled version first based on arity
                let func_idx_u16 = func_idx as u16;
                match args.len() {
                    0 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions_0.get(&func_idx_u16) {
                            let result = jit_fn();
                            set_reg!(dst, GcValue::Int64(result));
                            return Ok(StepResult::Continue);
                        }
                    }
                    1 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions.get(&func_idx_u16) {
                            if let GcValue::Int64(n) = reg!(args[0]) {
                                let result = jit_fn(n);
                                set_reg!(dst, GcValue::Int64(result));
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    2 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions_2.get(&func_idx_u16) {
                            if let (GcValue::Int64(a), GcValue::Int64(b)) = (reg!(args[0]), reg!(args[1])) {
                                let result = jit_fn(a, b);
                                set_reg!(dst, GcValue::Int64(result));
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    3 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions_3.get(&func_idx_u16) {
                            if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c)) =
                                (reg!(args[0]), reg!(args[1]), reg!(args[2])) {
                                let result = jit_fn(a, b, c);
                                set_reg!(dst, GcValue::Int64(result));
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    4 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions_4.get(&func_idx_u16) {
                            if let (GcValue::Int64(a), GcValue::Int64(b), GcValue::Int64(c), GcValue::Int64(d)) =
                                (reg!(args[0]), reg!(args[1]), reg!(args[2]), reg!(args[3])) {
                                let result = jit_fn(a, b, c, d);
                                set_reg!(dst, GcValue::Int64(result));
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    _ => {}
                }

                // Fall back to interpreter
                let function = self.shared.function_list.get(func_idx as usize)
                    .ok_or_else(|| RuntimeError::Panic(format!("Unknown function index: {}", func_idx)))?
                    .clone();

                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                let mut registers = vec![GcValue::Unit; function.code.register_count as usize];
                for (i, arg) in arg_values.into_iter().enumerate() {
                    if i < registers.len() {
                        registers[i] = arg;
                    }
                }

                self.frames.push(CallFrame {
                    function,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(dst),
                });
            }

            // Call function/closure stored in a register
            Call(dst, func_reg, ref args) => {
                let callee = reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();

                // Fast path for binary operations - check InlineOp first
                if arg_values.len() == 2 {
                    use crate::gc::InlineOp;
                    let inline_op = match &callee {
                        GcValue::Closure(_, op) => *op,
                        GcValue::Function(func) => InlineOp::from_function(func),
                        _ => InlineOp::None,
                    };
                    if inline_op != InlineOp::None {
                        if let (GcValue::Int64(x), GcValue::Int64(y)) = (&arg_values[0], &arg_values[1]) {
                            let result = match inline_op {
                                InlineOp::AddInt => x + y,
                                InlineOp::SubInt => x - y,
                                InlineOp::MulInt => x * y,
                                InlineOp::None => unreachable!(),
                            };
                            set_reg!(dst, GcValue::Int64(result));
                            return Ok(StepResult::Continue);
                        }
                    }
                }

                match callee {
                    GcValue::Function(func) => {
                        // Regular function call
                        let mut registers = vec![GcValue::Unit; func.code.register_count as usize];
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            if i < registers.len() {
                                registers[i] = arg;
                            }
                        }
                        self.frames.push(CallFrame {
                            function: func,
                            ip: 0,
                            registers,
                            captures: Vec::new(),
                            return_reg: Some(dst),
                        });
                    }
                    GcValue::Closure(ptr, _inline_op) => {
                        // Regular closure call (fast path already checked above)
                        let closure = self.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid closure reference".into()))?;
                        let func = closure.function.clone();
                        let captures = closure.captures.clone();

                        let mut registers = vec![GcValue::Unit; func.code.register_count as usize];
                        for (i, arg) in arg_values.into_iter().enumerate() {
                            if i < registers.len() {
                                registers[i] = arg;
                            }
                        }

                        self.frames.push(CallFrame {
                            function: func,
                            ip: 0,
                            registers,
                            captures,
                            return_reg: Some(dst),
                        });
                    }
                    _ => return Err(RuntimeError::Panic(format!("Call: expected function or closure, got {:?}", callee))),
                }
            }

            // === Process operations (async!) ===
            SelfPid(dst) => {
                set_reg!(dst, GcValue::Pid(self.pid.0));
            }

            ProcessAll(dst) => {
                // Get all registered process PIDs
                let registry = self.shared.process_registry.read().await;
                let pids: Vec<GcValue> = registry.keys()
                    .map(|pid| GcValue::Pid(pid.0))
                    .collect();
                drop(registry);
                let list = GcValue::List(crate::gc::GcList { data: Arc::new(pids), start: 0 });
                set_reg!(dst, list);
            }

            ProcessTime(dst, pid_reg) => {
                // Return uptime in milliseconds for the process
                // For async VM, we can only check if process is registered
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessTime: expected Pid".into())),
                };

                let uptime_ms = if target_pid == self.pid {
                    // Own process - use started_at
                    self.started_at.elapsed().as_millis() as i64
                } else {
                    // Other process - we don't have direct access to their start time
                    // Return -1 for processes we can't query (matches parallel VM behavior)
                    -1
                };
                set_reg!(dst, GcValue::Int64(uptime_ms));
            }

            ProcessAlive(dst, pid_reg) => {
                // Check if a process is registered (alive)
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessAlive: expected Pid".into())),
                };

                // Check if process is registered in the registry
                let registry = self.shared.process_registry.read().await;
                let alive = registry.contains_key(&target_pid);
                drop(registry);
                set_reg!(dst, GcValue::Bool(alive));
            }

            ProcessInfo(dst, pid_reg) => {
                // Get process info - only works for own process in async VM
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessInfo: expected Pid".into())),
                };

                let result = if target_pid == self.pid {
                    // Own process - return full info
                    let status = match self.state {
                        ProcessState::Running => "running",
                        ProcessState::Waiting => "waiting",
                        ProcessState::WaitingTimeout => "waiting",
                        ProcessState::WaitingIO => "io",
                        ProcessState::WaitingForMvar(_, _) => "waiting",
                        ProcessState::Sleeping => "sleeping",
                        ProcessState::Suspended => "suspended",
                        ProcessState::Exited(_) => "exited",
                    };
                    let mailbox_len = 0i64; // Can't easily query own mailbox length
                    let uptime_ms = self.started_at.elapsed().as_millis() as i64;

                    let status_str = self.heap.alloc_string(status.to_string());
                    let record = self.heap.alloc_record(
                        "ProcessInfo".to_string(),
                        vec!["status".to_string(), "mailbox".to_string(), "uptime".to_string()],
                        vec![GcValue::String(status_str), GcValue::Int64(mailbox_len), GcValue::Int64(uptime_ms)],
                        vec![false, false, false],
                    );
                    GcValue::Record(record)
                } else {
                    // Other process - can't access directly
                    GcValue::Unit
                };
                set_reg!(dst, result);
            }

            ProcessKill(dst, pid_reg) => {
                // Kill a process by unregistering it (its messages will fail)
                let target_pid = match reg!(pid_reg) {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("ProcessKill: expected Pid".into())),
                };

                // Can't kill self
                let killed = if target_pid == self.pid {
                    false
                } else {
                    // Unregister the process - this effectively kills it
                    // (its mailbox will be dropped, messages will fail)
                    let mut registry = self.shared.process_registry.write().await;
                    registry.remove(&target_pid).is_some()
                };
                set_reg!(dst, GcValue::Bool(killed));
            }

            // === MVar operations (async!) ===
            MvarRead(dst, name_idx) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarRead: expected string constant".into())),
                };
                let gc_value = self.mvar_read(&name).await?;
                set_reg!(dst, gc_value);
            }

            MvarWrite(name_idx, src) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarWrite: expected string constant".into())),
                };
                let value = reg!(src);
                self.mvar_write(&name, value).await?;
            }

            MvarLock(name_idx, is_write) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarLock: expected string constant".into())),
                };

                // Get the mvar
                let var = self.shared.mvars.get(&name)
                    .ok_or_else(|| RuntimeError::Panic(format!("Unknown mvar: {}", name)))?
                    .clone();

                // Acquire lock (async - will yield if contended)
                let guard = if is_write {
                    HeldMvarLock::Write(var.write_owned().await)
                } else {
                    HeldMvarLock::Read(var.read_owned().await)
                };

                // Store the guard
                self.held_mvar_locks
                    .entry(name)
                    .or_insert_with(Vec::new)
                    .push(guard);
            }

            MvarUnlock(name_idx, _is_write) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("MvarUnlock: expected string constant".into())),
                };

                // Pop and drop the guard to release the lock
                if let Some(guards) = self.held_mvar_locks.get_mut(&name) {
                    if guards.pop().is_none() {
                        return Err(RuntimeError::Panic(format!(
                            "MvarUnlock: no lock held on mvar: {}", name
                        )));
                    }
                    // Clean up empty entry
                    if guards.is_empty() {
                        self.held_mvar_locks.remove(&name);
                    }
                } else {
                    return Err(RuntimeError::Panic(format!(
                        "MvarUnlock: no lock held on mvar: {}", name
                    )));
                }
            }

            // === Print ===
            Print(_dst, src) => {
                let value = reg!(src);
                let s = self.heap.display_value(&value);
                self.output.push(s.clone());
                println!("{}", s);
            }

            Println(src) => {
                let value = reg!(src);
                let s = self.heap.display_value(&value);
                self.output.push(s.clone());
                println!("{}", s);
            }

            // === Tail call (replaces current frame) ===
            TailCallDirect(func_idx, ref args) => {
                // Check for JIT-compiled version first
                let func_idx_u16 = func_idx as u16;
                match args.len() {
                    0 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions_0.get(&func_idx_u16) {
                            let result = GcValue::Int64(jit_fn());
                            // Pop frame and return result
                            let return_reg = self.frames.last().unwrap().return_reg;
                            self.frames.pop();
                            if self.frames.is_empty() {
                                return Ok(StepResult::Finished(result));
                            } else if let Some(ret_reg) = return_reg {
                                let frame = self.frames.last_mut().unwrap();
                                frame.registers[ret_reg as usize] = result;
                            }
                            return Ok(StepResult::Continue);
                        }
                    }
                    1 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions.get(&func_idx_u16) {
                            if let GcValue::Int64(n) = reg!(args[0]) {
                                let result = GcValue::Int64(jit_fn(n));
                                let return_reg = self.frames.last().unwrap().return_reg;
                                self.frames.pop();
                                if self.frames.is_empty() {
                                    return Ok(StepResult::Finished(result));
                                } else if let Some(ret_reg) = return_reg {
                                    let frame = self.frames.last_mut().unwrap();
                                    frame.registers[ret_reg as usize] = result;
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    2 => {
                        if let Some(jit_fn) = self.shared.jit_int_functions_2.get(&func_idx_u16) {
                            if let (GcValue::Int64(a), GcValue::Int64(b)) = (reg!(args[0]), reg!(args[1])) {
                                let result = GcValue::Int64(jit_fn(a, b));
                                let return_reg = self.frames.last().unwrap().return_reg;
                                self.frames.pop();
                                if self.frames.is_empty() {
                                    return Ok(StepResult::Finished(result));
                                } else if let Some(ret_reg) = return_reg {
                                    let frame = self.frames.last_mut().unwrap();
                                    frame.registers[ret_reg as usize] = result;
                                }
                                return Ok(StepResult::Continue);
                            }
                        }
                    }
                    _ => {}
                }

                // Fall back to interpreter
                let function = self.shared.function_list.get(func_idx as usize)
                    .ok_or_else(|| RuntimeError::Panic(format!("Unknown function index: {}", func_idx)))?;

                // OPTIMIZATION: If calling same function, reuse registers (no heap allocation!)
                // This is critical for recursive functions like fold
                let current_func = &self.frames.last().unwrap().function;
                if std::sync::Arc::ptr_eq(function, current_func) && args.len() <= 8 {
                    // Same function - reuse registers, no allocation!
                    let mut saved_args: [std::mem::MaybeUninit<GcValue>; 8] =
                        unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                    // Save args to stack (take ownership, leave Unit behind)
                    for (i, &r) in args.iter().enumerate() {
                        saved_args[i] = std::mem::MaybeUninit::new(
                            std::mem::take(&mut self.frames.last_mut().unwrap().registers[r as usize])
                        );
                    }

                    let frame = self.frames.last_mut().unwrap();

                    // Clear all registers to Unit (in-place, no allocation)
                    for reg in frame.registers.iter_mut() {
                        *reg = GcValue::Unit;
                    }

                    // Write back saved args to parameter positions
                    for (i, _) in args.iter().enumerate() {
                        frame.registers[i] = unsafe { saved_args[i].assume_init_read() };
                    }

                    frame.ip = 0;
                    frame.captures.clear();
                } else {
                    // Different function or >8 args - need to set up new frame
                    let function = function.clone();
                    let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                    let mut registers = vec![GcValue::Unit; function.code.register_count as usize];
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < registers.len() {
                            registers[i] = arg;
                        }
                    }

                    // Tail call: replace current frame instead of pushing new one
                    let frame = self.frames.last_mut().unwrap();
                    frame.function = function;
                    frame.ip = 0;
                    frame.registers = registers;
                    frame.captures.clear();
                }
                // Note: return_reg stays the same since we're replacing this call
            }

            GeInt(dst, a, b) => {
                let va = match reg!(a) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("GeInt: expected Int64".into())) };
                let vb = match reg!(b) { GcValue::Int64(n) => n, _ => return Err(RuntimeError::Panic("GeInt: expected Int64".into())) };
                set_reg!(dst, GcValue::Bool(va >= vb));
            }

            // === Boolean operations ===
            And(dst, a, b) => {
                let va = match reg!(a) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("And: expected Bool".into())) };
                let vb = match reg!(b) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("And: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(va && vb));
            }
            Or(dst, a, b) => {
                let va = match reg!(a) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("Or: expected Bool".into())) };
                let vb = match reg!(b) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("Or: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(va || vb));
            }
            Not(dst, src) => {
                let v = match reg!(src) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("Not: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(!v));
            }

            // === Generic equality ===
            Eq(dst, a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                let equal = self.heap.gc_values_equal(&va, &vb);
                set_reg!(dst, GcValue::Bool(equal));
            }

            // === Pattern matching ===
            TestConst(dst, value_reg, const_idx) => {
                let constant = get_const!(const_idx);
                let value = reg!(value_reg);
                let gc_const = self.heap.value_to_gc(&constant);
                let result = self.heap.gc_values_equal(&value, &gc_const);
                set_reg!(dst, GcValue::Bool(result));
            }

            TestNil(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            TestUnit(dst, value_reg) => {
                let val = reg!(value_reg);
                let result = matches!(val, GcValue::Unit);
                set_reg!(dst, GcValue::Bool(result));
            }

            TestTag(dst, value_reg, ctor_idx) => {
                let expected_ctor = match get_const!(ctor_idx) {
                    Value::String(s) => s.clone(),
                    _ => return Err(RuntimeError::Panic("TestTag: expected string constant".into())),
                };
                let val = reg!(value_reg);
                let result = match val {
                    GcValue::Variant(ptr) => {
                        if let Some(var) = self.heap.get_variant(ptr) {
                            var.constructor.as_str() == expected_ctor.as_str()
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }

            // === Self-recursive calls ===
            CallSelf(dst, ref args) => {
                // Call the current function recursively
                let func = self.frames.last().unwrap().function.clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                let mut registers = vec![GcValue::Unit; func.code.register_count as usize];
                for (i, arg) in arg_values.into_iter().enumerate() {
                    if i < registers.len() {
                        registers[i] = arg;
                    }
                }
                self.frames.push(CallFrame {
                    function: func,
                    ip: 0,
                    registers,
                    captures: Vec::new(),
                    return_reg: Some(dst),
                });
            }
            TailCallSelf(ref args) => {
                // OPTIMIZATION: Reuse registers instead of allocating new Vec
                // Use stack array to save args before overwriting (avoids heap allocation)
                if args.len() <= 8 {
                    let mut saved_args: [std::mem::MaybeUninit<GcValue>; 8] =
                        unsafe { std::mem::MaybeUninit::uninit().assume_init() };

                    // Save args to stack (take ownership, leave Unit behind)
                    for (i, &r) in args.iter().enumerate() {
                        saved_args[i] = std::mem::MaybeUninit::new(
                            std::mem::take(&mut self.frames.last_mut().unwrap().registers[r as usize])
                        );
                    }

                    let frame = self.frames.last_mut().unwrap();

                    // Clear all registers to Unit (in-place, no allocation)
                    for reg in frame.registers.iter_mut() {
                        *reg = GcValue::Unit;
                    }

                    // Write back saved args to parameter positions
                    for (i, _) in args.iter().enumerate() {
                        frame.registers[i] = unsafe { saved_args[i].assume_init_read() };
                    }

                    frame.ip = 0;
                    frame.captures.clear();
                } else {
                    // Fallback for >8 args (rare)
                    let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                    let frame = self.frames.last_mut().unwrap();
                    for reg in frame.registers.iter_mut() {
                        *reg = GcValue::Unit;
                    }
                    for (i, arg) in arg_values.into_iter().enumerate() {
                        if i < frame.registers.len() {
                            frame.registers[i] = arg;
                        }
                    }
                    frame.ip = 0;
                    frame.captures.clear();
                }
            }

            // === Assertions ===
            Assert(src) => {
                let val = reg!(src);
                match val {
                    GcValue::Bool(true) => {}
                    GcValue::Bool(false) => {
                        return Err(RuntimeError::Panic("Assertion failed".into()));
                    }
                    _ => {
                        return Err(RuntimeError::Panic(format!("Assert: expected Bool, got {:?}", val)));
                    }
                }
            }

            AssertEq(a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                if !self.heap.gc_values_equal(&va, &vb) {
                    let sa = self.heap.display_value(&va);
                    let sb = self.heap.display_value(&vb);
                    return Err(RuntimeError::Panic(format!("Assertion failed: {} != {}", sa, sb)));
                }
            }

            // === Native function calls ===
            CallNative(dst, name_idx, ref args) => {
                let name = match get_const!(name_idx) {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::Panic("CallNative: expected string constant".into())),
                };
                let native = self.shared.natives.get(&name)
                    .ok_or_else(|| RuntimeError::Panic(format!("Unknown native function: {}", name)))?
                    .clone();
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();
                let result = (native.func)(&arg_values, &mut self.heap)?;
                set_reg!(dst, result);
            }

            // === Closures ===
            MakeClosure(dst, func_idx, ref captures) => {
                let function = match get_const!(func_idx) {
                    Value::Function(f) => f,
                    _ => return Err(RuntimeError::Panic(format!("MakeClosure: expected function at constant index {}", func_idx))),
                };
                let capture_values: Vec<GcValue> = captures.iter().map(|r| reg!(*r)).collect();
                let capture_names = function.param_names.clone();
                // Compute InlineOp for fast binary closure calls
                let inline_op = crate::gc::InlineOp::from_function(&function);
                let closure = self.heap.alloc_closure(function, capture_values, capture_names);
                set_reg!(dst, GcValue::Closure(closure, inline_op));
            }

            GetCapture(dst, idx) => {
                let frame = self.frames.last().unwrap();
                if (idx as usize) < frame.captures.len() {
                    let value = frame.captures[idx as usize].clone();
                    set_reg!(dst, value);
                } else {
                    return Err(RuntimeError::Panic(format!("Capture index {} out of bounds", idx)));
                }
            }

            // === Data structures ===
            MakeTuple(dst, ref elems) => {
                let values: Vec<GcValue> = elems.iter().map(|r| reg!(*r)).collect();
                let tuple = self.heap.alloc_tuple(values);
                set_reg!(dst, GcValue::Tuple(tuple));
            }

            GetTupleField(dst, tuple_reg, idx) => {
                let tuple = reg!(tuple_reg);
                if let GcValue::Tuple(ptr) = tuple {
                    if let Some(t) = self.heap.get_tuple(ptr) {
                        if (idx as usize) < t.items.len() {
                            let value = t.items[idx as usize].clone();
                            set_reg!(dst, value);
                        } else {
                            return Err(RuntimeError::Panic("Tuple index out of bounds".into()));
                        }
                    } else {
                        return Err(RuntimeError::Panic("Invalid tuple pointer".into()));
                    }
                } else {
                    return Err(RuntimeError::Panic("Expected tuple".into()));
                }
            }

            GetField(dst, record, field_idx) => {
                let field_name = match get_const!(field_idx) {
                    Value::String(s) => (*s).clone(),
                    _ => return Err(RuntimeError::Panic("GetField: field name must be string".into())),
                };
                let rec_val = reg!(record);
                match rec_val {
                    GcValue::Record(ptr) => {
                        let rec = self.heap.get_record(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid record reference".into()))?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::Panic(format!("Unknown field: {}", field_name)))?;
                        let value = rec.fields[idx].clone();
                        set_reg!(dst, value);
                    }
                    GcValue::Variant(ptr) => {
                        let var = self.heap.get_variant(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid variant reference".into()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid variant field index: {}", field_name)))?;
                        let value = var.fields.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Variant field {} out of range", idx)))?
                            .clone();
                        set_reg!(dst, value);
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        let idx: usize = field_name.parse()
                            .map_err(|_| RuntimeError::Panic(format!("Invalid tuple index: {}", field_name)))?;
                        let value = tuple.items.get(idx)
                            .ok_or_else(|| RuntimeError::Panic(format!("Tuple index {} out of bounds", idx)))?
                            .clone();
                        set_reg!(dst, value);
                    }
                    _ => return Err(RuntimeError::Panic("GetField expects record, variant, or tuple".into())),
                }
            }

            MakeList(dst, ref elems) => {
                let values: Vec<GcValue> = elems.iter().map(|r| reg!(*r)).collect();
                let list = self.heap.make_list(values);
                set_reg!(dst, GcValue::List(list));
            }

            Cons(dst, head, tail) => {
                let head_val = reg!(head);
                let tail_val = reg!(tail);
                if let GcValue::List(tail_list) = tail_val {
                    let mut items = vec![head_val];
                    items.extend(tail_list.items().iter().cloned());
                    let new_list = self.heap.make_list(items);
                    set_reg!(dst, GcValue::List(new_list));
                } else {
                    return Err(RuntimeError::Panic("Cons: tail must be a list".into()));
                }
            }

            Decons(head_dst, tail_dst, list_reg) => {
                let list_val = reg!(list_reg);
                if let GcValue::List(list) = list_val {
                    if !list.is_empty() {
                        let head = list.head().cloned().unwrap_or(GcValue::Unit);
                        let tail = list.tail();
                        set_reg!(head_dst, head);
                        set_reg!(tail_dst, GcValue::List(tail));
                    } else {
                        return Err(RuntimeError::Panic("Decons: empty list".into()));
                    }
                } else {
                    return Err(RuntimeError::Panic("Decons: expected list".into()));
                }
            }

            // Native range list creation - [1..n]
            RangeList(dst, n_reg) => {
                let n = match reg!(n_reg) {
                    GcValue::Int64(n) => n,
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Int64".to_string(),
                        found: "non-integer".to_string(),
                    }),
                };
                let items: Vec<GcValue> = (1..=n).map(|i| GcValue::Int64(i)).collect();
                let list = self.heap.make_list(items);
                set_reg!(dst, GcValue::List(list));
            }

            ListHead(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => {
                        if let Some(head) = list.items().first() {
                            head.clone()
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: format!("{:?}", list_val),
                    }),
                };
                set_reg!(dst, result);
            }

            ListTail(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let result = match list_val {
                    GcValue::List(list) => {
                        if !list.is_empty() {
                            GcValue::List(list.tail())
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds { index: 0, length: 0 });
                        }
                    }
                    _ => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: format!("{:?}", list_val),
                    }),
                };
                set_reg!(dst, result);
            }

            Length(dst, src) => {
                let val = reg!(src);
                let len = match val {
                    GcValue::List(list) => list.len() as i64,
                    GcValue::String(s) => {
                        if let Some(str_val) = self.heap.get_string(s) {
                            str_val.data.chars().count() as i64
                        } else {
                            0
                        }
                    }
                    GcValue::Tuple(ptr) => {
                        if let Some(t) = self.heap.get_tuple(ptr) {
                            t.items.len() as i64
                        } else {
                            0
                        }
                    }
                    GcValue::Int64Array(ptr) => {
                        self.heap.get_int64_array(ptr).map(|a| a.items.len() as i64).unwrap_or(0)
                    }
                    GcValue::Float64Array(ptr) => {
                        self.heap.get_float64_array(ptr).map(|a| a.items.len() as i64).unwrap_or(0)
                    }
                    _ => return Err(RuntimeError::Panic("Length: unsupported type".into())),
                };
                set_reg!(dst, GcValue::Int64(len));
            }

            // === Typed arrays ===
            MakeInt64Array(dst, size_reg) => {
                let size = match reg!(size_reg) {
                    GcValue::Int64(n) => n as usize,
                    _ => return Err(RuntimeError::Panic("MakeInt64Array: size must be Int64".into())),
                };
                let ptr = self.heap.alloc_int64_array(vec![0i64; size]);
                set_reg!(dst, GcValue::Int64Array(ptr));
            }

            MakeFloat64Array(dst, size_reg) => {
                let size = match reg!(size_reg) {
                    GcValue::Int64(n) => n as usize,
                    _ => return Err(RuntimeError::Panic("MakeFloat64Array: size must be Int64".into())),
                };
                let ptr = self.heap.alloc_float64_array(vec![0.0f64; size]);
                set_reg!(dst, GcValue::Float64Array(ptr));
            }

            Index(dst, coll, idx) => {
                let idx_val = match reg!(idx) {
                    GcValue::Int64(i) => i as usize,
                    _ => return Err(RuntimeError::Panic("Index: index must be Int64".into())),
                };
                let coll_val = reg!(coll);
                let value = match coll_val {
                    GcValue::List(list) => {
                        list.items().get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = self.heap.get_tuple(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid tuple reference".into()))?;
                        tuple.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Array(ptr) => {
                        let array = self.heap.get_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".into()))?;
                        array.items.get(idx_val).cloned()
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?
                    }
                    GcValue::Int64Array(ptr) => {
                        let array = self.heap.get_int64_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".into()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Int64(val)
                    }
                    GcValue::Float64Array(ptr) => {
                        let array = self.heap.get_float64_array(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".into()))?;
                        let val = *array.items.get(idx_val)
                            .ok_or_else(|| RuntimeError::Panic(format!("Index {} out of bounds", idx_val)))?;
                        GcValue::Float64(val)
                    }
                    _ => return Err(RuntimeError::Panic("Index expects list, tuple, or array".into())),
                };
                set_reg!(dst, value);
            }

            IndexSet(coll, idx, val) => {
                let idx_val = match reg!(idx) {
                    GcValue::Int64(i) => i as usize,
                    _ => return Err(RuntimeError::Panic("IndexSet: index must be Int64".into())),
                };
                let coll_val = reg!(coll);
                match coll_val {
                    GcValue::Array(ptr) => {
                        let new_value = reg!(val);
                        let array = self.heap.get_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid array reference".into()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Int64Array(ptr) => {
                        let new_value = match reg!(val) {
                            GcValue::Int64(v) => v,
                            _ => return Err(RuntimeError::Panic("Int64Array expects Int64 value".into())),
                        };
                        let array = self.heap.get_int64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid int64 array reference".into()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    GcValue::Float64Array(ptr) => {
                        let new_value = match reg!(val) {
                            GcValue::Float64(v) => v,
                            _ => return Err(RuntimeError::Panic("Float64Array expects Float64 value".into())),
                        };
                        let array = self.heap.get_float64_array_mut(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid float64 array reference".into()))?;
                        if idx_val >= array.items.len() {
                            return Err(RuntimeError::Panic(format!("Index {} out of bounds", idx_val)));
                        }
                        array.items[idx_val] = new_value;
                    }
                    _ => return Err(RuntimeError::Panic("IndexSet expects array".into())),
                }
            }

            ListIsEmpty(dst, list_reg) => {
                let list_val = reg!(list_reg);
                let is_empty = match list_val {
                    GcValue::List(list) => list.is_empty(),
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(is_empty));
            }

            // === String operations ===
            Concat(dst, a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                match (&va, &vb) {
                    (GcValue::String(sa), GcValue::String(sb)) => {
                        let str_a = self.heap.get_string(*sa).map(|x| x.data.clone()).unwrap_or_default();
                        let str_b = self.heap.get_string(*sb).map(|x| x.data.clone()).unwrap_or_default();
                        let result = format!("{}{}", str_a, str_b);
                        set_reg!(dst, GcValue::String(self.heap.alloc_string(result)));
                    }
                    (GcValue::List(la), GcValue::List(lb)) => {
                        let mut items: Vec<GcValue> = la.items().to_vec();
                        items.extend(lb.items().iter().cloned());
                        let new_list = self.heap.make_list(items);
                        set_reg!(dst, GcValue::List(new_list));
                    }
                    _ => return Err(RuntimeError::Panic("Concat: expected string or list".into())),
                }
            }

            EqStr(dst, a, b) => {
                let va = reg!(a);
                let vb = reg!(b);
                let equal = match (&va, &vb) {
                    (GcValue::String(sa), GcValue::String(sb)) => {
                        let str_a = self.heap.get_string(*sa).map(|x| &x.data);
                        let str_b = self.heap.get_string(*sb).map(|x| &x.data);
                        str_a == str_b
                    }
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(equal));
            }

            EqBool(dst, a, b) => {
                let va = match reg!(a) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("EqBool: expected Bool".into())) };
                let vb = match reg!(b) { GcValue::Bool(b) => b, _ => return Err(RuntimeError::Panic("EqBool: expected Bool".into())) };
                set_reg!(dst, GcValue::Bool(va == vb));
            }

            // === Concurrency: Spawn ===
            Spawn(dst, func_reg, ref args) => {
                let func_val = reg!(func_reg);
                let arg_values: Vec<GcValue> = args.iter().map(|r| reg!(*r)).collect();

                let (func, captures) = match func_val {
                    GcValue::Function(f) => (f, vec![]),
                    GcValue::Closure(ptr, _) => {
                        let closure = self.heap.get_closure(ptr)
                            .ok_or_else(|| RuntimeError::Panic("Invalid closure pointer".into()))?;
                        (closure.function.clone(), closure.captures.clone())
                    }
                    _ => return Err(RuntimeError::Panic("Spawn: expected function or closure".into())),
                };

                // Convert args and captures to thread-safe values (deep copy)
                let safe_args: Vec<ThreadSafeValue> = arg_values.iter()
                    .filter_map(|v| ThreadSafeValue::from_gc_value(v, &self.heap))
                    .collect();
                let safe_captures: Vec<ThreadSafeValue> = captures.iter()
                    .filter_map(|v| ThreadSafeValue::from_gc_value(v, &self.heap))
                    .collect();

                // Allocate new PID
                let child_pid = self.shared.alloc_pid();
                let shared_clone = self.shared.clone();

                // Spawn as local tokio task
                let func_name = func.name.clone();
                tokio::task::spawn_local(async move {
                    if std::env::var("ASYNC_VM_DEBUG").is_ok() {
                        eprintln!("[AsyncVM] Spawned process {} running function: {}", child_pid.0, func_name);
                    }
                    // Create new process
                    let mut process = AsyncProcess::new(child_pid, shared_clone.clone());
                    let sender = process.mailbox_sender.clone();
                    shared_clone.register_process(child_pid, sender).await;

                    // Convert thread-safe values back to GcValues in new heap
                    let gc_args: Vec<GcValue> = safe_args.iter()
                        .map(|v| v.to_gc_value(&mut process.heap))
                        .collect();
                    let gc_captures: Vec<GcValue> = safe_captures.iter()
                        .map(|v| v.to_gc_value(&mut process.heap))
                        .collect();

                    // Set up initial call frame
                    let reg_count = func.code.register_count as usize;
                    let mut registers = vec![GcValue::Unit; reg_count];
                    for (i, arg) in gc_args.into_iter().enumerate() {
                        if i < reg_count {
                            registers[i] = arg;
                        }
                    }

                    process.frames.push(CallFrame {
                        function: func,
                        ip: 0,
                        registers,
                        captures: gc_captures,
                        return_reg: None,
                    });

                    // Run the process
                    let _ = process.run().await;

                    // Unregister on exit
                    shared_clone.unregister_process(child_pid).await;
                });

                set_reg!(dst, GcValue::Pid(child_pid.0));
            }

            // === Concurrency: Send ===
            Send(target_reg, msg_reg) => {
                let target_val = reg!(target_reg);
                let message = reg!(msg_reg);

                let target_pid = match target_val {
                    GcValue::Pid(p) => Pid(p),
                    _ => return Err(RuntimeError::Panic("Send: expected Pid".into())),
                };

                // Convert message to thread-safe value
                let safe_msg = ThreadSafeValue::from_gc_value(&message, &self.heap)
                    .ok_or_else(|| RuntimeError::Panic("Send: cannot convert message".into()))?;

                // Send via async channel (non-blocking)
                self.shared.send_message(target_pid, safe_msg).await;
            }

            // === Concurrency: Receive (async!) ===
            Receive(dst) => {
                // This is where async shines - we await the message!
                // Unlike the parallel VM which polls, we yield to tokio scheduler
                match self.mailbox.recv().await {
                    Some(msg) => {
                        let gc_msg = msg.to_gc_value(&mut self.heap);
                        set_reg!(dst, gc_msg);
                    }
                    None => {
                        // Channel closed - process is dying
                        return Ok(StepResult::Finished(GcValue::Unit));
                    }
                }
            }

            // === Concurrency: Receive with Timeout ===
            ReceiveTimeout(dst, timeout_reg) => {
                let timeout_ms = match reg!(timeout_reg) {
                    GcValue::Int64(n) => n as u64,
                    _ => return Err(RuntimeError::Panic("ReceiveTimeout: expected Int64 for timeout".into())),
                };

                let timeout = Duration::from_millis(timeout_ms);

                // Use tokio timeout - properly yields to scheduler
                match tokio::time::timeout(timeout, self.mailbox.recv()).await {
                    Ok(Some(msg)) => {
                        // Message received before timeout
                        let gc_msg = msg.to_gc_value(&mut self.heap);
                        set_reg!(dst, gc_msg);
                    }
                    Ok(None) => {
                        // Channel closed
                        set_reg!(dst, GcValue::Unit);
                    }
                    Err(_) => {
                        // Timeout expired
                        set_reg!(dst, GcValue::Unit);
                    }
                }
            }

            // Unimplemented instructions - add as needed
            _ => {
                eprintln!("[AsyncVM] Unimplemented instruction: {:?}", instruction);
                return Err(RuntimeError::Panic(format!(
                    "Unimplemented instruction in async VM: {:?}",
                    instruction
                )));
            }
        }

        Ok(StepResult::Continue)
    }

    /// Try to handle an exception with registered handlers.
    fn handle_exception(&mut self, _error: &RuntimeError) -> bool {
        // TODO: Implement exception handling
        false
    }
}

/// Result of executing one instruction.
pub enum StepResult {
    /// Continue execution.
    Continue,
    /// Process finished with a value.
    Finished(GcValue),
}

/// Configuration for the async VM.
#[derive(Clone, Debug)]
pub struct AsyncConfig {
    /// Number of threads for work distribution.
    pub num_threads: usize,
    /// Reductions per yield for fairness.
    pub reductions_per_yield: usize,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            reductions_per_yield: REDUCTIONS_PER_YIELD,
        }
    }
}

/// The async VM entry point.
pub struct AsyncVM {
    /// Shared state.
    shared: Arc<AsyncSharedState>,
    /// IO runtime.
    io_runtime: Option<IoRuntime>,
    /// Config.
    config: AsyncConfig,
}

impl AsyncVM {
    /// Create a new async VM with the given configuration.
    pub fn new(config: AsyncConfig) -> Self {
        // Create IO runtime for async operations
        let io_runtime = IoRuntime::new();
        let io_sender = io_runtime.request_sender();

        let shared = Arc::new(AsyncSharedState {
            functions: HashMap::new(),
            function_list: Vec::new(),
            natives: HashMap::new(),
            types: HashMap::new(),
            jit_int_functions: HashMap::new(),
            jit_int_functions_0: HashMap::new(),
            jit_int_functions_2: HashMap::new(),
            jit_int_functions_3: HashMap::new(),
            jit_int_functions_4: HashMap::new(),
            jit_loop_array_functions: HashMap::new(),
            shutdown: AtomicBool::new(false),
            process_registry: TokioRwLock::new(HashMap::new()),
            mvars: HashMap::new(),
            dynamic_mvars: Arc::new(RwLock::new(HashMap::new())),
            dynamic_functions: Arc::new(RwLock::new(HashMap::new())),
            dynamic_types: Arc::new(RwLock::new(HashMap::new())),
            stdlib_functions: Arc::new(RwLock::new(HashMap::new())),
            stdlib_types: Arc::new(RwLock::new(HashMap::new())),
            prelude_imports: Arc::new(RwLock::new(HashMap::new())),
            eval_callback: Arc::new(RwLock::new(None)),
            io_sender: Some(io_sender),
            next_pid: AtomicU64::new(1),
        });

        Self {
            shared,
            io_runtime: Some(io_runtime),
            config,
        }
    }

    /// Register a function (must be called before run).
    pub fn register_function(&mut self, name: &str, func: Arc<FunctionValue>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .functions
            .insert(name.to_string(), func);
    }

    /// Set the function list for indexed calls.
    pub fn set_function_list(&mut self, functions: Vec<Arc<FunctionValue>>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot set function list after execution started")
            .function_list = functions;
    }

    /// Register a native function.
    pub fn register_native(&mut self, name: &str, native: Arc<GcNativeFn>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .natives
            .insert(name.to_string(), native);
    }

    /// Register a type.
    pub fn register_type(&mut self, name: &str, type_val: Arc<TypeValue>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .types
            .insert(name.to_string(), type_val);
    }

    /// Register an mvar with initial value.
    pub fn register_mvar(&mut self, name: &str, initial_value: ThreadSafeValue) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .mvars
            .insert(name.to_string(), Arc::new(TokioRwLock::new(initial_value)));
    }

    /// Register a JIT-compiled function (arity 0).
    pub fn register_jit_int_function_0(&mut self, func_index: u16, jit_fn: crate::parallel::JitIntFn0) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .jit_int_functions_0
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 1).
    pub fn register_jit_int_function(&mut self, func_index: u16, jit_fn: crate::parallel::JitIntFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .jit_int_functions
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 2).
    pub fn register_jit_int_function_2(&mut self, func_index: u16, jit_fn: crate::parallel::JitIntFn2) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .jit_int_functions_2
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 3).
    pub fn register_jit_int_function_3(&mut self, func_index: u16, jit_fn: crate::parallel::JitIntFn3) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .jit_int_functions_3
            .insert(func_index, jit_fn);
    }

    /// Register a JIT-compiled function (arity 4).
    pub fn register_jit_int_function_4(&mut self, func_index: u16, jit_fn: crate::parallel::JitIntFn4) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .jit_int_functions_4
            .insert(func_index, jit_fn);
    }

    /// Register a JIT loop array function.
    pub fn register_jit_loop_array_function(&mut self, func_index: u16, jit_fn: crate::parallel::JitLoopArrayFn) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot register after execution started")
            .jit_loop_array_functions
            .insert(func_index, jit_fn);
    }

    /// Get dynamic functions (for REPL).
    pub fn get_dynamic_functions(&self) -> Arc<RwLock<HashMap<String, Arc<FunctionValue>>>> {
        self.shared.dynamic_functions.clone()
    }

    /// Get stdlib functions.
    pub fn get_stdlib_functions(&self) -> Arc<RwLock<HashMap<String, Arc<FunctionValue>>>> {
        self.shared.stdlib_functions.clone()
    }

    /// Set stdlib functions.
    pub fn set_stdlib_functions(&self, functions: HashMap<String, Arc<FunctionValue>>) {
        *self.shared.stdlib_functions.write().unwrap() = functions;
    }

    /// Get dynamic types.
    pub fn get_dynamic_types(&self) -> Arc<RwLock<HashMap<String, Arc<TypeValue>>>> {
        self.shared.dynamic_types.clone()
    }

    /// Get dynamic mvars.
    pub fn get_dynamic_mvars(&self) -> Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>> {
        self.shared.dynamic_mvars.clone()
    }

    /// Set dynamic mvars.
    pub fn set_dynamic_mvars(&mut self, mvars: Arc<RwLock<HashMap<String, Arc<RwLock<ThreadSafeValue>>>>>) {
        Arc::get_mut(&mut self.shared)
            .expect("Cannot set dynamic mvars after execution started")
            .dynamic_mvars = mvars;
    }

    /// Get stdlib types.
    pub fn get_stdlib_types(&self) -> Arc<RwLock<HashMap<String, Arc<TypeValue>>>> {
        self.shared.stdlib_types.clone()
    }

    /// Set stdlib types.
    pub fn set_stdlib_types(&self, types: HashMap<String, Arc<TypeValue>>) {
        *self.shared.stdlib_types.write().unwrap() = types;
    }

    /// Get prelude imports.
    pub fn get_prelude_imports(&self) -> Arc<RwLock<HashMap<String, String>>> {
        self.shared.prelude_imports.clone()
    }

    /// Set prelude imports.
    pub fn set_prelude_imports(&self, imports: HashMap<String, String>) {
        *self.shared.prelude_imports.write().unwrap() = imports;
    }

    /// Get a function by name.
    pub fn get_function(&self, name: &str) -> Option<Arc<FunctionValue>> {
        self.shared.functions.get(name).cloned()
    }

    /// Setup eval callback for REPL.
    pub fn setup_eval(&mut self) {
        // Empty for now - async VM doesn't need special setup
    }

    /// Set eval callback.
    pub fn set_eval_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        *self.shared.eval_callback.write().unwrap() = Some(Arc::new(callback));
    }

    /// Register default native functions.
    pub fn register_default_natives(&mut self) {
        // show - convert value to string
        self.register_native("show", Arc::new(GcNativeFn {
            name: "show".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                Ok(GcValue::String(heap.alloc_string(s)))
            }),
        }));

        // copy - deep copy a value
        self.register_native("copy", Arc::new(GcNativeFn {
            name: "copy".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                Ok(heap.clone_value(&args[0]))
            }),
        }));

        // print - print without newline
        self.register_native("print", Arc::new(GcNativeFn {
            name: "print".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                print!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // println - print with newline
        self.register_native("println", Arc::new(GcNativeFn {
            name: "println".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                let s = heap.display_value(&args[0]);
                println!("{}", s);
                Ok(GcValue::Unit)
            }),
        }));

        // String.length
        self.register_native("String.length", Arc::new(GcNativeFn {
            name: "String.length".to_string(),
            arity: 1,
            func: Box::new(|args, heap| {
                match &args[0] {
                    GcValue::String(s) => {
                        if let Some(str_val) = heap.get_string(*s) {
                            Ok(GcValue::Int64(str_val.data.chars().count() as i64))
                        } else {
                            Err(RuntimeError::Panic("Invalid string pointer".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::TypeError { expected: "String".to_string(), found: "other".to_string() })
                }
            }),
        }));
    }

    /// Run the main function and return the result.
    pub fn run(&mut self, main_fn_name: &str) -> Result<SendableValue, String> {
        // Create tokio runtime - use current_thread for LocalSet compatibility
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        // Run within a LocalSet so spawn_local works
        let local = LocalSet::new();
        local.block_on(&rt, self.run_async(main_fn_name))
    }

    /// Run with multi-threaded work distribution.
    /// Spawns multiple worker threads, each with its own LocalSet.
    pub fn run_parallel(&mut self, main_fn_name: &str, num_threads: usize) -> Result<SendableValue, String> {
        // For now, use single-threaded. Multi-threaded requires coordinating
        // LocalSets across threads, which we'll add later.
        // TODO: Implement multi-threaded execution with thread-local LocalSets
        let _ = num_threads;
        self.run(main_fn_name)
    }

    /// Async entry point for running main.
    async fn run_async(&self, main_fn_name: &str) -> Result<SendableValue, String> {
        // Find main function
        let main_fn = self.shared.functions.get(main_fn_name)
            .ok_or_else(|| format!("Main function '{}' not found", main_fn_name))?
            .clone();

        // Create main process
        let pid = self.shared.alloc_pid();
        let mut process = AsyncProcess::new(pid, self.shared.clone());
        let sender = process.mailbox_sender.clone();
        self.shared.register_process(pid, sender).await;

        // Set up initial call frame
        let registers = vec![GcValue::Unit; main_fn.code.register_count as usize];
        process.frames.push(CallFrame {
            function: main_fn,
            ip: 0,
            registers,
            captures: Vec::new(),
            return_reg: None,
        });

        // Run main process (blocks until complete)
        match process.run().await {
            Ok(value) => {
                let sendable = SendableValue::from_gc_value(&value, &process.heap);
                Ok(sendable)
            }
            Err(e) => Err(e.to_string()),
        }
    }
}
