//! Virtual Machine for Nostos bytecode execution.
//!
//! Design goals:
//! - Tail call optimization (TCO) - recursive functions don't grow stack
//! - JIT-ready - can be replaced with compiled code
//! - REPL-friendly - supports incremental compilation
//! - Introspectable - full access to runtime state

use std::collections::HashMap;
use std::rc::Rc;

use crate::gc::{GcNativeFn, GcValue, Heap};
use crate::value::*;

/// Maximum call stack depth (before TCO kicks in).
const MAX_STACK_DEPTH: usize = 10000;

/// A call frame on the stack.
#[derive(Clone)]
pub struct CallFrame {
    /// Function being executed
    pub function: Rc<FunctionValue>,
    /// Instruction pointer
    pub ip: usize,
    /// Register file for this frame (GC-managed values)
    pub registers: Vec<GcValue>,
    /// Captured variables (for closures, GC-managed)
    pub captures: Vec<GcValue>,
    /// Return register in caller's frame
    pub return_reg: Option<Reg>,
}

/// The virtual machine state.
pub struct VM {
    /// Garbage-collected heap for runtime values
    pub heap: Heap,
    /// Call stack
    pub frames: Vec<CallFrame>,
    /// Global variables (kept as Value for now - mostly static)
    pub globals: HashMap<String, Value>,
    /// Global functions
    pub functions: HashMap<String, Rc<FunctionValue>>,
    /// Native functions (GC-aware, work directly with GcValue)
    pub natives: HashMap<String, Rc<GcNativeFn>>,
    /// Type definitions (for introspection)
    pub types: HashMap<String, Rc<TypeValue>>,
    /// Exception handlers stack
    pub handlers: Vec<ExceptionHandler>,
    /// Current exception (if any, GC-managed)
    pub current_exception: Option<GcValue>,
    /// Output buffer (for testing/REPL)
    pub output: Vec<String>,
}

/// Exception handler info.
#[derive(Clone)]
pub struct ExceptionHandler {
    pub frame_index: usize,
    pub catch_ip: usize,
}

/// Result of VM execution.
pub type VMResult = Result<Value, RuntimeError>;

impl VM {
    pub fn new() -> Self {
        let mut vm = Self {
            heap: Heap::new(),
            frames: Vec::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            natives: HashMap::new(),
            types: HashMap::new(),
            handlers: Vec::new(),
            current_exception: None,
            output: Vec::new(),
        };
        vm.register_builtins();
        vm
    }

    /// Register built-in functions.
    fn register_builtins(&mut self) {
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
                GcValue::Array(ptr) => {
                    if let Some(arr) = heap.get_array(*ptr) {
                        Ok(GcValue::Int(arr.items.len() as i64))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "Array".to_string(),
                            found: "invalid array pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "List, String, or Array".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("head", 1, |args, heap| {
            match &args[0] {
                GcValue::List(ptr) => {
                    if let Some(list) = heap.get_list(*ptr) {
                        if !list.items.is_empty() {
                            Ok(list.items[0].clone())
                        } else {
                            Err(RuntimeError::MatchFailed)
                        }
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "invalid list pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "List".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("tail", 1, |args, heap| {
            match &args[0] {
                GcValue::List(ptr) => {
                    if let Some(list) = heap.get_list(*ptr) {
                        if !list.items.is_empty() {
                            let tail_items = list.items[1..].to_vec();
                            Ok(GcValue::List(heap.alloc_list(tail_items)))
                        } else {
                            Err(RuntimeError::MatchFailed)
                        }
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "invalid list pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "List".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("isEmpty", 1, |args, heap| {
            match &args[0] {
                GcValue::List(ptr) => {
                    if let Some(list) = heap.get_list(*ptr) {
                        Ok(GcValue::Bool(list.items.is_empty()))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "List".to_string(),
                            found: "invalid list pointer".to_string(),
                        })
                    }
                }
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        Ok(GcValue::Bool(s.data.is_empty()))
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

        self.register_native("show", 1, |args, heap| {
            let s = heap.display_value(&args[0]);
            Ok(GcValue::String(heap.alloc_string(s)))
        });

        self.register_native("toInt", 1, |args, heap| {
            match &args[0] {
                GcValue::Int(i) => Ok(GcValue::Int(*i)),
                GcValue::Float(f) => Ok(GcValue::Int(*f as i64)),
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        s.data.parse::<i64>()
                            .map(GcValue::Int)
                            .map_err(|_| RuntimeError::TypeError {
                                expected: "numeric string".to_string(),
                                found: format!("\"{}\"", s.data),
                            })
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "Int, Float, or String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("toFloat", 1, |args, heap| {
            match &args[0] {
                GcValue::Int(i) => Ok(GcValue::Float(*i as f64)),
                GcValue::Float(f) => Ok(GcValue::Float(*f)),
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        s.data.parse::<f64>()
                            .map(GcValue::Float)
                            .map_err(|_| RuntimeError::TypeError {
                                expected: "numeric string".to_string(),
                                found: format!("\"{}\"", s.data),
                            })
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "Int, Float, or String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("abs", 1, |args, heap| {
            match &args[0] {
                GcValue::Int(i) => Ok(GcValue::Int(i.abs())),
                GcValue::Float(f) => Ok(GcValue::Float(f.abs())),
                other => Err(RuntimeError::TypeError {
                    expected: "Int or Float".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("sqrt", 1, |args, heap| {
            match &args[0] {
                GcValue::Float(f) => Ok(GcValue::Float(f.sqrt())),
                GcValue::Int(i) => Ok(GcValue::Float((*i as f64).sqrt())),
                other => Err(RuntimeError::TypeError {
                    expected: "Float or Int".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("panic", 1, |args, heap| {
            match &args[0] {
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        Err(RuntimeError::Panic(s.data.clone()))
                    } else {
                        Err(RuntimeError::Panic("panic".to_string()))
                    }
                }
                other => Err(RuntimeError::Panic(heap.display_value(other))),
            }
        });

        self.register_native("assert", 2, |args, heap| {
            match &args[0] {
                GcValue::Bool(true) => Ok(GcValue::Unit),
                GcValue::Bool(false) => match &args[1] {
                    GcValue::String(ptr) => {
                        if let Some(s) = heap.get_string(*ptr) {
                            Err(RuntimeError::AssertionFailed(s.data.clone()))
                        } else {
                            Err(RuntimeError::AssertionFailed("assertion failed".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::AssertionFailed("assertion failed".to_string())),
                },
                other => Err(RuntimeError::TypeError {
                    expected: "Bool".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        // I/O functions
        self.register_native("readFile", 1, |args, heap| {
            match &args[0] {
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        match std::fs::read_to_string(&s.data) {
                            Ok(contents) => Ok(GcValue::String(heap.alloc_string(contents))),
                            Err(e) => Err(RuntimeError::IOError(e.to_string())),
                        }
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("writeFile", 2, |args, heap| {
            match (&args[0], &args[1]) {
                (GcValue::String(path_ptr), GcValue::String(contents_ptr)) => {
                    let path = heap.get_string(*path_ptr).map(|s| s.data.clone());
                    let contents = heap.get_string(*contents_ptr).map(|s| s.data.clone());
                    match (path, contents) {
                        (Some(path), Some(contents)) => {
                            match std::fs::write(&path, &contents) {
                                Ok(()) => Ok(GcValue::Unit),
                                Err(e) => Err(RuntimeError::IOError(e.to_string())),
                            }
                        }
                        _ => Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        }),
                    }
                }
                (GcValue::String(_), other) => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
                (other, _) => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("appendFile", 2, |args, heap| {
            use std::fs::OpenOptions;
            use std::io::Write;
            match (&args[0], &args[1]) {
                (GcValue::String(path_ptr), GcValue::String(contents_ptr)) => {
                    let path = heap.get_string(*path_ptr).map(|s| s.data.clone());
                    let contents = heap.get_string(*contents_ptr).map(|s| s.data.clone());
                    match (path, contents) {
                        (Some(path), Some(contents)) => {
                            match OpenOptions::new().append(true).create(true).open(&path) {
                                Ok(mut file) => {
                                    match file.write_all(contents.as_bytes()) {
                                        Ok(()) => Ok(GcValue::Unit),
                                        Err(e) => Err(RuntimeError::IOError(e.to_string())),
                                    }
                                }
                                Err(e) => Err(RuntimeError::IOError(e.to_string())),
                            }
                        }
                        _ => Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        }),
                    }
                }
                (GcValue::String(_), other) => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
                (other, _) => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("fileExists", 1, |args, heap| {
            match &args[0] {
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        Ok(GcValue::Bool(std::path::Path::new(&s.data).exists()))
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });

        self.register_native("readLine", 0, |_args, heap| {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(_) => {
                    // Remove trailing newline
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    Ok(GcValue::String(heap.alloc_string(line)))
                }
                Err(e) => Err(RuntimeError::IOError(e.to_string())),
            }
        });

        self.register_native("getArgs", 0, |_args, heap| {
            let args: Vec<GcValue> = std::env::args()
                .skip(1) // Skip the program name
                .map(|s| GcValue::String(heap.alloc_string(s)))
                .collect();
            Ok(GcValue::List(heap.alloc_list(args)))
        });

        self.register_native("getEnv", 1, |args, heap| {
            match &args[0] {
                GcValue::String(ptr) => {
                    if let Some(s) = heap.get_string(*ptr) {
                        match std::env::var(&s.data) {
                            Ok(value) => Ok(GcValue::String(heap.alloc_string(value))),
                            Err(_) => Ok(GcValue::String(heap.alloc_string(String::new()))),
                        }
                    } else {
                        Err(RuntimeError::TypeError {
                            expected: "String".to_string(),
                            found: "invalid string pointer".to_string(),
                        })
                    }
                }
                other => Err(RuntimeError::TypeError {
                    expected: "String".to_string(),
                    found: other.type_name(heap).to_string(),
                }),
            }
        });
    }

    pub fn register_native<F>(&mut self, name: &str, arity: usize, func: F)
    where
        F: Fn(&[GcValue], &mut Heap) -> Result<GcValue, RuntimeError> + Send + Sync + 'static,
    {
        let native = Rc::new(GcNativeFn {
            name: name.to_string(),
            arity,
            func: Box::new(func),
        });
        self.natives.insert(name.to_string(), native);
    }

    pub fn register_function(&mut self, func: FunctionValue) {
        let name = func.name.clone();
        self.functions.insert(name, Rc::new(func));
    }

    pub fn set_global(&mut self, name: &str, value: Value) {
        self.globals.insert(name.to_string(), value);
    }

    pub fn get_global(&self, name: &str) -> Option<&Value> {
        self.globals.get(name)
    }

    pub fn call(&mut self, name: &str, args: Vec<Value>) -> VMResult {
        if let Some(native) = self.natives.get(name).cloned() {
            if args.len() != native.arity {
                return Err(RuntimeError::ArityMismatch {
                    expected: native.arity,
                    found: args.len(),
                });
            }
            // Convert Value args to GcValue for native call
            let gc_args: Vec<GcValue> = args.iter()
                .map(|a| self.heap.value_to_gc(a))
                .collect();
            let result = (native.func)(&gc_args, &mut self.heap)?;
            return Ok(self.heap.gc_to_value(&result));
        }

        let func = self.functions.get(name).cloned()
            .ok_or_else(|| RuntimeError::UnknownFunction(name.to_string()))?;

        if args.len() != func.arity {
            return Err(RuntimeError::ArityMismatch {
                expected: func.arity,
                found: args.len(),
            });
        }

        self.execute_function(func, args, vec![])
    }

    fn execute_function(
        &mut self,
        func: Rc<FunctionValue>,
        args: Vec<Value>,
        captures: Vec<Value>,
    ) -> VMResult {
        // Convert args to GcValue
        let mut registers = vec![GcValue::Unit; func.code.register_count];
        for (i, arg) in args.into_iter().enumerate() {
            registers[i] = self.heap.value_to_gc(&arg);
        }

        // Convert captures to GcValue
        let gc_captures: Vec<GcValue> = captures
            .into_iter()
            .map(|c| self.heap.value_to_gc(&c))
            .collect();

        let frame = CallFrame {
            function: func,
            ip: 0,
            registers,
            captures: gc_captures,
            return_reg: None,
        };

        self.frames.push(frame);
        self.run()
    }

    fn run(&mut self) -> VMResult {
        loop {
            if self.frames.is_empty() {
                return Ok(Value::Unit);
            }

            if self.frames.len() > MAX_STACK_DEPTH {
                // Stack overflow - check if there's a handler
                if let Some(handler) = self.handlers.pop() {
                    let exc = RuntimeError::StackOverflow.to_exception_value();
                    self.current_exception = Some(self.heap.value_to_gc(&exc));
                    while self.frames.len() > handler.frame_index + 1 {
                        self.frames.pop();
                    }
                    if let Some(frame) = self.frames.last_mut() {
                        frame.ip = handler.catch_ip;
                    }
                    continue;
                }
                return Err(RuntimeError::StackOverflow);
            }

            let result = match self.execute_step() {
                Ok(r) => r,
                Err(e) => {
                    // Runtime error occurred - check if there's an exception handler
                    if let Some(handler) = self.handlers.pop() {
                        // Convert the error to an exception value
                        let exc = e.to_exception_value();
                        self.current_exception = Some(self.heap.value_to_gc(&exc));

                        // Unwind stack to the handler's frame
                        while self.frames.len() > handler.frame_index + 1 {
                            self.frames.pop();
                        }

                        // Jump to the catch block
                        if let Some(frame) = self.frames.last_mut() {
                            frame.ip = handler.catch_ip;
                        }
                        continue;
                    } else {
                        // No handler - propagate the error
                        return Err(e);
                    }
                }
            };

            match result {
                StepResult::Continue => {}
                StepResult::Return(gc_value) => {
                    self.frames.pop();
                    if self.frames.is_empty() {
                        // Convert GcValue back to Value for the final return
                        return Ok(self.heap.gc_to_value(&gc_value));
                    }
                    if let Some(frame) = self.frames.last_mut() {
                        if let Some(ret_reg) = frame.return_reg {
                            frame.registers[ret_reg as usize] = gc_value;
                        }
                        frame.return_reg = None;
                    }
                }
                StepResult::TailCall { func, args, captures } => {
                    let frame = self.frames.last_mut().unwrap();
                    frame.function = func.clone();
                    frame.ip = 0;
                    frame.captures = captures;
                    frame.registers.clear();
                    frame.registers.resize(func.code.register_count, GcValue::Unit);
                    for (i, arg) in args.into_iter().enumerate() {
                        frame.registers[i] = arg;
                    }
                }
                StepResult::Call { func, args, captures, return_reg } => {
                    self.frames.last_mut().unwrap().return_reg = Some(return_reg);

                    let mut registers = vec![GcValue::Unit; func.code.register_count];
                    for (i, arg) in args.into_iter().enumerate() {
                        registers[i] = arg;
                    }

                    let new_frame = CallFrame {
                        function: func,
                        ip: 0,
                        registers,
                        captures,
                        return_reg: None,
                    };
                    self.frames.push(new_frame);
                }
            }
        }
    }

    /// Execute a single step and return what to do next.
    fn execute_step(&mut self) -> Result<StepResult, RuntimeError> {
        // Get instruction first (immutable borrow)
        let (instr, constants, frame_len) = {
            let frame = self.frames.last().unwrap();
            let instr = frame.function.code.code[frame.ip].clone();
            let constants = frame.function.code.constants.clone();
            (instr, constants, self.frames.len())
        };

        // Increment IP
        self.frames.last_mut().unwrap().ip += 1;

        // Helper macros to reduce boilerplate
        macro_rules! frame {
            () => { self.frames.last_mut().unwrap() }
        }

        macro_rules! reg {
            ($r:expr) => { &self.frames.last().unwrap().registers[$r as usize] }
        }

        macro_rules! set_reg {
            ($r:expr, $v:expr) => { frame!().registers[$r as usize] = $v };
        }

        // Helper to get type name for error messages
        macro_rules! gc_type_name {
            ($v:expr) => {
                match $v {
                    GcValue::Unit => "()",
                    GcValue::Bool(_) => "Bool",
                    GcValue::Int(_) => "Int",
                    GcValue::Float(_) => "Float",
                    GcValue::Char(_) => "Char",
                    GcValue::String(_) => "String",
                    GcValue::List(_) => "List",
                    GcValue::Array(_) => "Array",
                    GcValue::Tuple(_) => "Tuple",
                    GcValue::Map(_) => "Map",
                    GcValue::Set(_) => "Set",
                    GcValue::Record(_) => "Record",
                    GcValue::Variant(_) => "Variant",
                    GcValue::Closure(_) => "Closure",
                    GcValue::Function(_) => "Function",
                    GcValue::NativeFunction(_) => "NativeFunction",
                    GcValue::Pid(_) => "Pid",
                    GcValue::Ref(_) => "Ref",
                    GcValue::Type(_) => "Type",
                    GcValue::Pointer(_) => "Pointer",
                }
            };
        }

        macro_rules! get_int {
            ($r:expr) => {
                match reg!($r) {
                    GcValue::Int(i) => *i,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: gc_type_name!(other).to_string(),
                    }),
                }
            };
        }

        macro_rules! get_float {
            ($r:expr) => {
                match reg!($r) {
                    GcValue::Float(f) => *f,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: gc_type_name!(other).to_string(),
                    }),
                }
            };
        }

        macro_rules! get_bool {
            ($r:expr) => {
                match reg!($r) {
                    GcValue::Bool(b) => *b,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: gc_type_name!(other).to_string(),
                    }),
                }
            };
        }

        macro_rules! get_string_ptr {
            ($r:expr) => {
                match reg!($r) {
                    GcValue::String(ptr) => *ptr,
                    other => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: gc_type_name!(other).to_string(),
                    }),
                }
            };
        }

        match instr {
            Instruction::LoadConst(dst, idx) => {
                // Convert Value constant to GcValue
                let value = &constants[idx as usize];
                let gc_value = self.heap.value_to_gc(value);
                set_reg!(dst, gc_value);
            }
            Instruction::Move(dst, src) => {
                let value = reg!(src).clone();
                set_reg!(dst, value);
            }
            Instruction::LoadUnit(dst) => set_reg!(dst, GcValue::Unit),
            Instruction::LoadTrue(dst) => set_reg!(dst, GcValue::Bool(true)),
            Instruction::LoadFalse(dst) => set_reg!(dst, GcValue::Bool(false)),

            // Integer arithmetic
            Instruction::AddInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Int(a + b));
            }
            Instruction::SubInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Int(a - b));
            }
            Instruction::MulInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Int(a * b));
            }
            Instruction::DivInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                if b == 0 { return Err(RuntimeError::DivisionByZero); }
                set_reg!(dst, GcValue::Int(a / b));
            }
            Instruction::ModInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                if b == 0 { return Err(RuntimeError::DivisionByZero); }
                set_reg!(dst, GcValue::Int(a % b));
            }
            Instruction::NegInt(dst, src) => {
                let v = get_int!(src);
                set_reg!(dst, GcValue::Int(-v));
            }

            // Float arithmetic
            Instruction::AddFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Float(a + b));
            }
            Instruction::SubFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Float(a - b));
            }
            Instruction::MulFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Float(a * b));
            }
            Instruction::DivFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Float(a / b));
            }
            Instruction::NegFloat(dst, src) => {
                let v = get_float!(src);
                set_reg!(dst, GcValue::Float(-v));
            }
            Instruction::PowFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Float(a.powf(b)));
            }

            // Comparisons
            Instruction::EqInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Bool(a == b));
            }
            Instruction::NeInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Bool(a != b));
            }
            Instruction::LtInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Bool(a < b));
            }
            Instruction::LeInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Bool(a <= b));
            }
            Instruction::GtInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Bool(a > b));
            }
            Instruction::GeInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, GcValue::Bool(a >= b));
            }
            Instruction::EqFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Bool(a == b));
            }
            Instruction::LtFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Bool(a < b));
            }
            Instruction::LeFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, GcValue::Bool(a <= b));
            }
            Instruction::EqBool(dst, a, b) => {
                let a = get_bool!(a);
                let b = get_bool!(b);
                set_reg!(dst, GcValue::Bool(a == b));
            }
            Instruction::EqStr(dst, a, b) => {
                let a_ptr = get_string_ptr!(a);
                let b_ptr = get_string_ptr!(b);
                let a_str = self.heap.get_string(a_ptr).map(|s| &s.data);
                let b_str = self.heap.get_string(b_ptr).map(|s| &s.data);
                set_reg!(dst, GcValue::Bool(a_str == b_str));
            }
            Instruction::Eq(dst, a, b) => {
                // Use gc_values_equal for content-based comparison (not pointer comparison)
                let result = self.heap.gc_values_equal(reg!(a), reg!(b));
                set_reg!(dst, GcValue::Bool(result));
            }

            // Logical
            Instruction::Not(dst, src) => {
                let v = get_bool!(src);
                set_reg!(dst, GcValue::Bool(!v));
            }
            Instruction::And(dst, a, b) => {
                let a = get_bool!(a);
                let b = get_bool!(b);
                set_reg!(dst, GcValue::Bool(a && b));
            }
            Instruction::Or(dst, a, b) => {
                let a = get_bool!(a);
                let b = get_bool!(b);
                set_reg!(dst, GcValue::Bool(a || b));
            }

            // String
            Instruction::Concat(dst, a, b) => {
                let a_ptr = get_string_ptr!(a);
                let b_ptr = get_string_ptr!(b);
                let a_str = self.heap.get_string(a_ptr).map(|s| s.data.as_str()).unwrap_or("");
                let b_str = self.heap.get_string(b_ptr).map(|s| s.data.as_str()).unwrap_or("");
                let result = format!("{}{}", a_str, b_str);
                let result_ptr = self.heap.alloc_string(result);
                set_reg!(dst, GcValue::String(result_ptr));
            }

            // Collections
            Instruction::MakeList(dst, regs) => {
                let items: Vec<GcValue> = regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                let list_ptr = self.heap.alloc_list(items);
                set_reg!(dst, GcValue::List(list_ptr));
            }
            Instruction::MakeTuple(dst, regs) => {
                let items: Vec<GcValue> = regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                let tuple_ptr = self.heap.alloc_tuple(items);
                set_reg!(dst, GcValue::Tuple(tuple_ptr));
            }
            Instruction::Cons(dst, head, tail) => {
                let head = reg!(head).clone();
                let tail_ptr = match reg!(tail) {
                    GcValue::List(ptr) => *ptr,
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                };
                let tail_items = self.heap.get_list(tail_ptr)
                    .map(|l| l.items.clone())
                    .unwrap_or_default();
                let mut new_items = vec![head];
                new_items.extend(tail_items);
                let list_ptr = self.heap.alloc_list(new_items);
                set_reg!(dst, GcValue::List(list_ptr));
            }
            Instruction::ListConcat(dst, a, b) => {
                let a_ptr = match reg!(a) {
                    GcValue::List(ptr) => *ptr,
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                };
                let b_ptr = match reg!(b) {
                    GcValue::List(ptr) => *ptr,
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                };
                let a_items = self.heap.get_list(a_ptr)
                    .map(|l| l.items.clone())
                    .unwrap_or_default();
                let b_items = self.heap.get_list(b_ptr)
                    .map(|l| l.items.clone())
                    .unwrap_or_default();
                let mut new_items = a_items;
                new_items.extend(b_items);
                let list_ptr = self.heap.alloc_list(new_items);
                set_reg!(dst, GcValue::List(list_ptr));
            }
            Instruction::Index(dst, coll, idx) => {
                let idx = get_int!(idx) as usize;
                let value = match reg!(coll) {
                    GcValue::List(ptr) => {
                        let list = self.heap.get_list(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "List".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        list.items.get(idx).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds { index: idx as i64, length: list.items.len() })?
                    }
                    GcValue::Tuple(ptr) => {
                        let tuple = self.heap.get_tuple(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Tuple".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        tuple.items.get(idx).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds { index: idx as i64, length: tuple.items.len() })?
                    }
                    GcValue::Array(ptr) => {
                        let array = self.heap.get_array(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Array".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        array.items.get(idx).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds { index: idx as i64, length: array.items.len() })?
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "List, Tuple, or Array".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                };
                set_reg!(dst, value);
            }
            Instruction::Length(dst, src) => {
                let len = match reg!(src) {
                    GcValue::List(ptr) => {
                        self.heap.get_list(*ptr).map(|l| l.items.len()).unwrap_or(0)
                    }
                    GcValue::Tuple(ptr) => {
                        self.heap.get_tuple(*ptr).map(|t| t.items.len()).unwrap_or(0)
                    }
                    GcValue::Array(ptr) => {
                        self.heap.get_array(*ptr).map(|a| a.items.len()).unwrap_or(0)
                    }
                    GcValue::String(ptr) => {
                        self.heap.get_string(*ptr).map(|s| s.data.len()).unwrap_or(0)
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "List, Tuple, Array, or String".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                };
                set_reg!(dst, GcValue::Int(len as i64));
            }
            Instruction::MakeMap(dst, pairs) => {
                let mut map = HashMap::new();
                for (k, v) in pairs {
                    let key = reg!(k).to_gc_map_key()
                        .ok_or_else(|| RuntimeError::TypeError {
                            expected: "hashable type".to_string(),
                            found: reg!(k).type_name(&self.heap).to_string(),
                        })?;
                    let value = reg!(v).clone();
                    map.insert(key, value);
                }
                let map_ptr = self.heap.alloc_map(map);
                set_reg!(dst, GcValue::Map(map_ptr));
            }
            Instruction::MakeSet(dst, regs) => {
                let mut set = std::collections::HashSet::new();
                for r in regs {
                    let key = reg!(r).to_gc_map_key()
                        .ok_or_else(|| RuntimeError::TypeError {
                            expected: "hashable type".to_string(),
                            found: reg!(r).type_name(&self.heap).to_string(),
                        })?;
                    set.insert(key);
                }
                let set_ptr = self.heap.alloc_set(set);
                set_reg!(dst, GcValue::Set(set_ptr));
            }
            Instruction::IndexSet(coll, idx, val) => {
                let idx = get_int!(idx) as usize;
                let value = reg!(val).clone();
                match reg!(coll) {
                    GcValue::Array(ptr) => {
                        let array = self.heap.get_array_mut(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Array".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        if idx >= array.items.len() {
                            return Err(RuntimeError::IndexOutOfBounds {
                                index: idx as i64,
                                length: array.items.len(),
                            });
                        }
                        array.items[idx] = value;
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Array".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }

            // Records
            Instruction::MakeRecord(dst, type_idx, field_regs) => {
                let type_name = match &constants[type_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let type_info = self.types.get(&type_name);
                let field_names: Vec<String> = type_info
                    .map(|t| t.fields.iter().map(|f| f.name.clone()).collect())
                    .unwrap_or_else(|| (0..field_regs.len()).map(|i| format!("_{}", i)).collect());
                let mutable_fields: Vec<bool> = type_info
                    .map(|t| t.fields.iter().map(|f| f.mutable).collect())
                    .unwrap_or_else(|| vec![false; field_regs.len()]);
                let fields: Vec<GcValue> = field_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                let record_ptr = self.heap.alloc_record(type_name, field_names, fields, mutable_fields);
                set_reg!(dst, GcValue::Record(record_ptr));
            }
            Instruction::GetField(dst, record, field_idx) => {
                let field_name = match &constants[field_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                match reg!(record) {
                    GcValue::Record(ptr) => {
                        let rec = self.heap.get_record(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Record".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::UnknownField {
                                type_name: rec.type_name.clone(),
                                field: field_name.clone(),
                            })?;
                        let value = rec.fields[idx].clone();
                        set_reg!(dst, value);
                    }
                    GcValue::Tuple(ptr) => {
                        // Support tuple field access with numeric indices (t.0, t.1, etc.)
                        let tuple = self.heap.get_tuple(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Tuple".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let idx: usize = field_name.parse().map_err(|_| RuntimeError::UnknownField {
                            type_name: "Tuple".to_string(),
                            field: field_name.clone(),
                        })?;
                        let value = tuple.items.get(idx).cloned().ok_or_else(|| RuntimeError::IndexOutOfBounds {
                            index: idx as i64,
                            length: tuple.items.len(),
                        })?;
                        set_reg!(dst, value);
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Record or Tuple".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }
            Instruction::SetField(record, field_idx, value) => {
                let field_name = match &constants[field_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let new_value = reg!(value).clone();
                match reg!(record).clone() {
                    GcValue::Record(ptr) => {
                        let rec = self.heap.get_record(ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Record".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::UnknownField {
                                type_name: rec.type_name.clone(),
                                field: field_name.clone(),
                            })?;
                        if !rec.mutable_fields[idx] {
                            return Err(RuntimeError::ImmutableField { field: field_name });
                        }
                        // Create a new record with the updated field
                        let mut new_fields = rec.fields.clone();
                        new_fields[idx] = new_value;
                        let new_record_ptr = self.heap.alloc_record(
                            rec.type_name.clone(),
                            rec.field_names.clone(),
                            new_fields,
                            rec.mutable_fields.clone(),
                        );
                        set_reg!(record, GcValue::Record(new_record_ptr));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Record".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }
            Instruction::UpdateRecord(dst, base, type_idx, field_regs) => {
                let type_name = match &constants[type_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                match reg!(base).clone() {
                    GcValue::Record(ptr) => {
                        let rec = self.heap.get_record(ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Record".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let mut new_fields = rec.fields.clone();
                        // Update the specified fields (for now, assumes positional)
                        for (i, &r) in field_regs.iter().enumerate() {
                            if i < new_fields.len() {
                                new_fields[i] = reg!(r).clone();
                            }
                        }
                        let new_record_ptr = self.heap.alloc_record(
                            type_name,
                            rec.field_names.clone(),
                            new_fields,
                            rec.mutable_fields.clone(),
                        );
                        set_reg!(dst, GcValue::Record(new_record_ptr));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Record".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }

            // Tuples
            Instruction::GetTupleField(dst, tuple, idx) => {
                match reg!(tuple) {
                    GcValue::Tuple(ptr) => {
                        let t = self.heap.get_tuple(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Tuple".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        if (idx as usize) < t.items.len() {
                            set_reg!(dst, t.items[idx as usize].clone());
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds {
                                index: idx as i64,
                                length: t.items.len(),
                            });
                        }
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Tuple".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }

            // Variants
            Instruction::MakeVariant(dst, type_idx, ctor_idx, field_regs) => {
                let type_name = match &constants[type_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let constructor = match &constants[ctor_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let fields: Vec<GcValue> = field_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                let variant_ptr = self.heap.alloc_variant(type_name, constructor, fields);
                set_reg!(dst, GcValue::Variant(variant_ptr));
            }
            Instruction::GetTag(dst, variant) => {
                match reg!(variant) {
                    GcValue::Variant(ptr) => {
                        let v = self.heap.get_variant(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Variant".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let str_ptr = self.heap.alloc_string(v.constructor.clone());
                        set_reg!(dst, GcValue::String(str_ptr));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Variant".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }
            Instruction::GetVariantField(dst, variant, field_idx) => {
                match reg!(variant) {
                    GcValue::Variant(ptr) => {
                        let v = self.heap.get_variant(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Variant".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let value = v.fields.get(field_idx as usize).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds {
                                index: field_idx as i64,
                                length: v.fields.len(),
                            })?;
                        set_reg!(dst, value);
                    }
                    // Also support Record types for unified handling
                    GcValue::Record(ptr) => {
                        let r = self.heap.get_record(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Record".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        let value = r.fields.get(field_idx as usize).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds {
                                index: field_idx as i64,
                                length: r.fields.len(),
                            })?;
                        set_reg!(dst, value);
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Variant".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }
            Instruction::GetVariantFieldByName(_dst, variant, name_idx) => {
                let field_name = match &constants[name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                match reg!(variant) {
                    GcValue::Variant(ptr) => {
                        // Note: GcVariant doesn't have named_fields - this is a simplification
                        // For now, return an error as named_fields are not supported in GC variants
                        let v = self.heap.get_variant(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Variant".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        return Err(RuntimeError::UnknownField {
                            type_name: v.constructor.clone(),
                            field: field_name.clone(),
                        });
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Variant".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }

            // Control flow
            Instruction::Jump(offset) => {
                let frame = frame!();
                frame.ip = (frame.ip as isize + offset as isize) as usize;
            }
            Instruction::JumpIfTrue(cond, offset) => {
                if get_bool!(cond) {
                    let frame = frame!();
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }
            }
            Instruction::JumpIfFalse(cond, offset) => {
                if !get_bool!(cond) {
                    let frame = frame!();
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }
            }

            // Pattern matching
            Instruction::TestTag(dst, variant, ctor_idx) => {
                let ctor_name = match &constants[ctor_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let result = match reg!(variant) {
                    GcValue::Variant(ptr) => {
                        self.heap.get_variant(*ptr)
                            .map(|v| v.constructor == ctor_name)
                            .unwrap_or(false)
                    }
                    // Also support Record types for unified handling
                    GcValue::Record(ptr) => {
                        self.heap.get_record(*ptr)
                            .map(|r| r.type_name == ctor_name)
                            .unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }
            Instruction::TestUnit(dst, value) => {
                let result = matches!(reg!(value), GcValue::Unit);
                set_reg!(dst, GcValue::Bool(result));
            }
            Instruction::TestConst(dst, value, const_idx) => {
                let constant = &constants[const_idx as usize];
                // Convert constant to GcValue for comparison
                let gc_const = self.heap.value_to_gc(constant);
                // Use gc_values_equal for content-based comparison (not pointer comparison)
                let result = self.heap.gc_values_equal(reg!(value), &gc_const);
                set_reg!(dst, GcValue::Bool(result));
            }
            Instruction::TestNil(dst, list) => {
                let result = match reg!(list) {
                    GcValue::List(ptr) => {
                        self.heap.get_list(*ptr)
                            .map(|l| l.items.is_empty())
                            .unwrap_or(false)
                    }
                    _ => false,
                };
                set_reg!(dst, GcValue::Bool(result));
            }
            Instruction::Decons(head, tail, list) => {
                match reg!(list) {
                    GcValue::List(ptr) => {
                        let list_data = self.heap.get_list(*ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "List".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        if list_data.items.is_empty() {
                            return Err(RuntimeError::MatchFailed);
                        }
                        let head_val = list_data.items[0].clone();
                        let tail_items: Vec<GcValue> = list_data.items[1..].to_vec();
                        let tail_ptr = self.heap.alloc_list(tail_items);
                        set_reg!(head, head_val);
                        set_reg!(tail, GcValue::List(tail_ptr));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }

            // Function calls
            Instruction::Call(dst, func_reg, arg_regs) => {
                let args: Vec<GcValue> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                match reg!(func_reg).clone() {
                    GcValue::Function(func) => {
                        if args.len() != func.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: func.arity,
                                found: args.len(),
                            });
                        }
                        return Ok(StepResult::Call {
                            func,
                            args,
                            captures: vec![],
                            return_reg: dst,
                        });
                    }
                    GcValue::Closure(closure_ptr) => {
                        let closure = self.heap.get_closure(closure_ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Closure".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        if args.len() != closure.function.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: closure.function.arity,
                                found: args.len(),
                            });
                        }
                        let func = closure.function.clone();
                        let gc_captures = closure.captures.clone();
                        return Ok(StepResult::Call {
                            func,
                            args,
                            captures: gc_captures,
                            return_reg: dst,
                        });
                    }
                    GcValue::NativeFunction(native) => {
                        if args.len() != native.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: native.arity,
                                found: args.len(),
                            });
                        }
                        // Call native function directly with GcValue
                        let result = (native.func)(&args, &mut self.heap)?;
                        set_reg!(dst, result);
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }
            Instruction::TailCall(func_reg, arg_regs) => {
                let args: Vec<GcValue> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                match reg!(func_reg).clone() {
                    GcValue::Function(func) => {
                        if args.len() != func.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: func.arity,
                                found: args.len(),
                            });
                        }
                        return Ok(StepResult::TailCall {
                            func,
                            args,
                            captures: vec![],
                        });
                    }
                    GcValue::Closure(closure_ptr) => {
                        let closure = self.heap.get_closure(closure_ptr)
                            .ok_or(RuntimeError::TypeError {
                                expected: "Closure".to_string(),
                                found: "invalid heap reference".to_string(),
                            })?;
                        if args.len() != closure.function.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: closure.function.arity,
                                found: args.len(),
                            });
                        }
                        let func = closure.function.clone();
                        let gc_captures = closure.captures.clone();
                        return Ok(StepResult::TailCall {
                            func,
                            args,
                            captures: gc_captures,
                        });
                    }
                    GcValue::NativeFunction(native) => {
                        if args.len() != native.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: native.arity,
                                found: args.len(),
                            });
                        }
                        // Call native function directly with GcValue
                        let result = (native.func)(&args, &mut self.heap)?;
                        return Ok(StepResult::Return(result));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: other.type_name(&self.heap).to_string(),
                    }),
                }
            }
            Instruction::CallByName(dst, name_idx, arg_regs) => {
                let name = match &constants[name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let args: Vec<GcValue> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                let func = self.functions.get(&name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name))?;

                if args.len() != func.arity {
                    return Err(RuntimeError::ArityMismatch {
                        expected: func.arity,
                        found: args.len(),
                    });
                }
                return Ok(StepResult::Call {
                    func,
                    args,
                    captures: vec![],
                    return_reg: dst,
                });
            }
            Instruction::TailCallByName(name_idx, arg_regs) => {
                let name = match &constants[name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                let args: Vec<GcValue> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                let func = self.functions.get(&name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name))?;

                if args.len() != func.arity {
                    return Err(RuntimeError::ArityMismatch {
                        expected: func.arity,
                        found: args.len(),
                    });
                }
                return Ok(StepResult::TailCall {
                    func,
                    args,
                    captures: vec![],
                });
            }
            Instruction::CallNative(dst, name_idx, arg_regs) => {
                let name = match &constants[name_idx as usize] {
                    Value::String(s) => s.to_string(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: "non-string".to_string(),
                    }),
                };
                // Collect args as GcValue - no conversion needed!
                let gc_args: Vec<GcValue> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                let native = self.natives.get(&name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name))?;

                if gc_args.len() != native.arity {
                    return Err(RuntimeError::ArityMismatch {
                        expected: native.arity,
                        found: gc_args.len(),
                    });
                }
                // Call native function directly with GcValue and heap
                let result = (native.func)(&gc_args, &mut self.heap)?;
                set_reg!(dst, result);
            }
            Instruction::Return(src) => {
                let value = reg!(src).clone();
                return Ok(StepResult::Return(value));
            }

            // Closures
            Instruction::MakeClosure(dst, func_idx, capture_regs) => {
                let func = match &constants[func_idx as usize] {
                    Value::Function(f) => f.clone(),
                    _ => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: "non-function".to_string(),
                    }),
                };
                // Collect captures as GcValue for the GC-managed closure
                let gc_captures: Vec<GcValue> = capture_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                let capture_names: Vec<String> = (0..gc_captures.len())
                    .map(|i| format!("capture_{}", i))
                    .collect();

                let closure_ptr = self.heap.alloc_closure(func, gc_captures, capture_names);
                set_reg!(dst, GcValue::Closure(closure_ptr));
            }
            Instruction::GetCapture(dst, idx) => {
                let frame = self.frames.last().unwrap();
                if (idx as usize) < frame.captures.len() {
                    let value = frame.captures[idx as usize].clone();
                    set_reg!(dst, value);
                } else {
                    return Err(RuntimeError::IndexOutOfBounds {
                        index: idx as i64,
                        length: frame.captures.len(),
                    });
                }
            }

            // Error handling
            Instruction::PushHandler(offset) => {
                let catch_ip = (self.frames.last().unwrap().ip as isize + offset as isize) as usize;
                self.handlers.push(ExceptionHandler {
                    frame_index: frame_len - 1,
                    catch_ip,
                });
            }
            Instruction::PopHandler => {
                self.handlers.pop();
            }
            Instruction::Throw(value) => {
                let exc = reg!(value).clone();
                self.current_exception = Some(exc.clone());

                if let Some(handler) = self.handlers.pop() {
                    while self.frames.len() > handler.frame_index + 1 {
                        self.frames.pop();
                    }
                    frame!().ip = handler.catch_ip;
                } else {
                    return Err(RuntimeError::Panic(format!("Uncaught exception: {:?}", exc)));
                }
            }
            Instruction::GetException(dst) => {
                let exc = self.current_exception.take().unwrap_or(GcValue::Unit);
                set_reg!(dst, exc);
            }

            // Introspection
            Instruction::TypeOf(dst, src) => {
                let type_name = reg!(src).type_name(&self.heap).to_string();
                let str_ptr = self.heap.alloc_string(type_name);
                set_reg!(dst, GcValue::String(str_ptr));
            }

            // Concurrency (stubs)
            Instruction::SelfPid(dst) => {
                set_reg!(dst, GcValue::Pid(0));
            }
            Instruction::Spawn(_, _, _) |
            Instruction::SpawnLink(_, _, _) |
            Instruction::SpawnMonitor(_, _, _, _) |
            Instruction::Send(_, _) |
            Instruction::Receive |
            Instruction::ReceiveTimeout(_) => {
                return Err(RuntimeError::Panic("Concurrency not yet implemented".to_string()));
            }

            Instruction::Nop => {}
            Instruction::DebugPrint(r) => {
                let value = reg!(r).clone();
                println!("DEBUG: {:?}", value);
                self.output.push(format!("{:?}", value));
            }
        }

        Ok(StepResult::Continue)
    }
}

enum StepResult {
    Continue,
    Return(GcValue),
    TailCall {
        func: Rc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
    },
    Call {
        func: Rc<FunctionValue>,
        args: Vec<GcValue>,
        captures: Vec<GcValue>,
        return_reg: Reg,
    },
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_creation() {
        let vm = VM::new();
        assert!(vm.natives.contains_key("println"));
        assert!(vm.natives.contains_key("length"));
    }

    #[test]
    fn test_simple_addition() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;
        chunk.emit(Instruction::AddInt(2, 0, 1), 1);
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "add".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);
        let result = vm.call("add", vec![Value::Int(2), Value::Int(3)]).unwrap();
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_tail_recursion() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 5;

        let zero_idx = chunk.add_constant(Value::Int(0));
        let one_idx = chunk.add_constant(Value::Int(1));
        let func_name_idx = chunk.add_constant(Value::String(Rc::new("sum".to_string())));

        // sum(n, acc) = if n == 0 then acc else sum(n - 1, acc + n)
        // Registers: 0 = n, 1 = acc, 2 = temp, 3 = result, 4 = temp2
        // Note: IP is incremented BEFORE jump, so offset is relative to IP+1
        chunk.emit(Instruction::LoadConst(2, zero_idx), 1);       // 0: r2 = 0
        chunk.emit(Instruction::EqInt(3, 0, 2), 1);               // 1: r3 = n == 0
        chunk.emit(Instruction::JumpIfFalse(3, 1), 1);            // 2: if !r3, jump +1 (to 4)
        chunk.emit(Instruction::Return(1), 1);                     // 3: return acc
        chunk.emit(Instruction::LoadConst(2, one_idx), 1);        // 4: r2 = 1
        chunk.emit(Instruction::SubInt(3, 0, 2), 1);              // 5: r3 = n - 1
        chunk.emit(Instruction::AddInt(4, 1, 0), 1);              // 6: r4 = acc + n
        chunk.emit(Instruction::TailCallByName(func_name_idx, vec![3, 4]), 1); // 7: tail sum(r3, r4)

        let func = FunctionValue {
            name: "sum".to_string(),
            arity: 2,
            param_names: vec!["n".to_string(), "acc".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let result = vm.call("sum", vec![Value::Int(10000), Value::Int(0)]).unwrap();
        assert_eq!(result, Value::Int(50005000));
    }

    #[test]
    fn test_native_function() {
        let mut vm = VM::new();

        let result = vm.call("length", vec![Value::List(Rc::new(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
        ]))]).unwrap();

        assert_eq!(result, Value::Int(3));
    }

    #[test]
    fn test_conditional() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;

        // max(a, b) = if a > b then a else b
        // Note: IP is incremented BEFORE jump, so offset is relative to IP+1
        chunk.emit(Instruction::GtInt(2, 0, 1), 1);     // 0: r2 = a > b
        chunk.emit(Instruction::JumpIfFalse(2, 1), 1);  // 1: if !r2, jump +1 (to 3)
        chunk.emit(Instruction::Return(0), 1);          // 2: return a
        chunk.emit(Instruction::Return(1), 1);          // 3: return b

        let func = FunctionValue {
            name: "max".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let result1 = vm.call("max", vec![Value::Int(5), Value::Int(3)]).unwrap();
        assert_eq!(result1, Value::Int(5));

        let result2 = vm.call("max", vec![Value::Int(2), Value::Int(7)]).unwrap();
        assert_eq!(result2, Value::Int(7));
    }

    #[test]
    fn test_factorial_tail_recursive() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 5;

        let zero_idx = chunk.add_constant(Value::Int(0));
        let one_idx = chunk.add_constant(Value::Int(1));
        let func_name_idx = chunk.add_constant(Value::String(Rc::new("fact".to_string())));

        // fact(n, acc) = if n == 0 then acc else fact(n - 1, acc * n)
        chunk.emit(Instruction::LoadConst(2, zero_idx), 1);       // 0: r2 = 0
        chunk.emit(Instruction::EqInt(3, 0, 2), 1);               // 1: r3 = n == 0
        chunk.emit(Instruction::JumpIfFalse(3, 1), 1);            // 2: if !r3, jump +1 (to 4)
        chunk.emit(Instruction::Return(1), 1);                     // 3: return acc
        chunk.emit(Instruction::LoadConst(2, one_idx), 1);        // 4: r2 = 1
        chunk.emit(Instruction::SubInt(3, 0, 2), 1);              // 5: r3 = n - 1
        chunk.emit(Instruction::MulInt(4, 1, 0), 1);              // 6: r4 = acc * n
        chunk.emit(Instruction::TailCallByName(func_name_idx, vec![3, 4]), 1); // 7: tail fact(r3, r4)

        let func = FunctionValue {
            name: "fact".to_string(),
            arity: 2,
            param_names: vec!["n".to_string(), "acc".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // 5! = 120
        let result = vm.call("fact", vec![Value::Int(5), Value::Int(1)]).unwrap();
        assert_eq!(result, Value::Int(120));

        // 10! = 3628800
        let result = vm.call("fact", vec![Value::Int(10), Value::Int(1)]).unwrap();
        assert_eq!(result, Value::Int(3628800));
    }

    #[test]
    fn test_float_arithmetic() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 4;

        // compute(a, b) = (a + b) * (a - b) = a^2 - b^2
        chunk.emit(Instruction::AddFloat(2, 0, 1), 1);  // r2 = a + b
        chunk.emit(Instruction::SubFloat(3, 0, 1), 1);  // r3 = a - b
        chunk.emit(Instruction::MulFloat(2, 2, 3), 1);  // r2 = (a+b) * (a-b)
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "diff_squares".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // 5^2 - 3^2 = 25 - 9 = 16
        let result = vm.call("diff_squares", vec![Value::Float(5.0), Value::Float(3.0)]).unwrap();
        assert_eq!(result, Value::Float(16.0));

        // 10^2 - 6^2 = 100 - 36 = 64
        let result = vm.call("diff_squares", vec![Value::Float(10.0), Value::Float(6.0)]).unwrap();
        assert_eq!(result, Value::Float(64.0));
    }

    #[test]
    fn test_list_index() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;

        // get_second(list) = list[1]
        let one_idx = chunk.add_constant(Value::Int(1));
        chunk.emit(Instruction::LoadConst(1, one_idx), 1);  // r1 = 1
        chunk.emit(Instruction::Index(2, 0, 1), 1);         // r2 = list[1]
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "get_second".to_string(),
            arity: 1,
            param_names: vec!["list".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let list = Value::List(Rc::new(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));
        let result = vm.call("get_second", vec![list]).unwrap();
        assert_eq!(result, Value::Int(20));
    }

    #[test]
    fn test_list_cons_and_length() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;

        // cons_len(elem, list) -> length of (elem :: list)
        chunk.emit(Instruction::Cons(2, 0, 1), 1);   // r2 = elem :: list
        chunk.emit(Instruction::Length(2, 2), 1);    // r2 = length(r2)
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "cons_len".to_string(),
            arity: 2,
            param_names: vec!["elem".to_string(), "list".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let list = Value::List(Rc::new(vec![Value::Int(2), Value::Int(3)]));
        let result = vm.call("cons_len", vec![Value::Int(1), list]).unwrap();
        assert_eq!(result, Value::Int(3)); // [1, 2, 3] has length 3
    }

    #[test]
    fn test_boolean_logic() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 4;

        // xor(a, b) = (a || b) && !(a && b)
        chunk.emit(Instruction::Or(2, 0, 1), 1);     // r2 = a || b
        chunk.emit(Instruction::And(3, 0, 1), 1);    // r3 = a && b
        chunk.emit(Instruction::Not(3, 3), 1);       // r3 = !(a && b)
        chunk.emit(Instruction::And(2, 2, 3), 1);    // r2 = (a || b) && !(a && b)
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "xor".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // XOR truth table
        assert_eq!(vm.call("xor", vec![Value::Bool(false), Value::Bool(false)]).unwrap(), Value::Bool(false));
        assert_eq!(vm.call("xor", vec![Value::Bool(false), Value::Bool(true)]).unwrap(), Value::Bool(true));
        assert_eq!(vm.call("xor", vec![Value::Bool(true), Value::Bool(false)]).unwrap(), Value::Bool(true));
        assert_eq!(vm.call("xor", vec![Value::Bool(true), Value::Bool(true)]).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_string_concat() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;

        // concat3(a, b) = a ++ b ++ a
        chunk.emit(Instruction::Concat(2, 0, 1), 1);  // r2 = a ++ b
        chunk.emit(Instruction::Concat(2, 2, 0), 1);  // r2 = (a ++ b) ++ a
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "concat3".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let result = vm.call("concat3", vec![
            Value::String(Rc::new("Hello".to_string())),
            Value::String(Rc::new(" World ".to_string())),
        ]).unwrap();
        assert_eq!(result, Value::String(Rc::new("Hello World Hello".to_string())));
    }

    #[test]
    fn test_tuple_operations() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 4;

        // swap(tuple) = (tuple.1, tuple.0)
        chunk.emit(Instruction::GetTupleField(2, 0, 0), 1);  // r2 = tuple.0
        chunk.emit(Instruction::GetTupleField(3, 0, 1), 1);  // r3 = tuple.1
        chunk.emit(Instruction::MakeTuple(0, vec![3, 2]), 1); // r0 = (r3, r2)
        chunk.emit(Instruction::Return(0), 1);

        let func = FunctionValue {
            name: "swap".to_string(),
            arity: 1,
            param_names: vec!["tuple".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let tuple = Value::Tuple(Rc::new(vec![Value::Int(1), Value::Int(2)]));
        let result = vm.call("swap", vec![tuple]).unwrap();
        assert_eq!(result, Value::Tuple(Rc::new(vec![Value::Int(2), Value::Int(1)])));
    }

    #[test]
    fn test_fibonacci_iterative() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 6;

        let zero_idx = chunk.add_constant(Value::Int(0));
        let one_idx = chunk.add_constant(Value::Int(1));
        let func_name_idx = chunk.add_constant(Value::String(Rc::new("fib_iter".to_string())));

        // fib_iter(n, a, b) = if n == 0 then a else fib_iter(n-1, b, a+b)
        // r0=n, r1=a, r2=b, r3=temp, r4=temp2, r5=temp3
        chunk.emit(Instruction::LoadConst(3, zero_idx), 1);       // 0: r3 = 0
        chunk.emit(Instruction::EqInt(4, 0, 3), 1);               // 1: r4 = n == 0
        chunk.emit(Instruction::JumpIfFalse(4, 1), 1);            // 2: if !r4, jump +1 (to 4)
        chunk.emit(Instruction::Return(1), 1);                     // 3: return a
        chunk.emit(Instruction::LoadConst(3, one_idx), 1);        // 4: r3 = 1
        chunk.emit(Instruction::SubInt(4, 0, 3), 1);              // 5: r4 = n - 1
        chunk.emit(Instruction::AddInt(5, 1, 2), 1);              // 6: r5 = a + b
        chunk.emit(Instruction::TailCallByName(func_name_idx, vec![4, 2, 5]), 1); // 7: tail fib_iter(n-1, b, a+b)

        let func = FunctionValue {
            name: "fib_iter".to_string(),
            arity: 3,
            param_names: vec!["n".to_string(), "a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // fib(0) = 0, fib(1) = 1, fib(2) = 1, fib(3) = 2, fib(4) = 3, fib(5) = 5
        // fib(10) = 55, fib(20) = 6765
        assert_eq!(vm.call("fib_iter", vec![Value::Int(0), Value::Int(0), Value::Int(1)]).unwrap(), Value::Int(0));
        assert_eq!(vm.call("fib_iter", vec![Value::Int(1), Value::Int(0), Value::Int(1)]).unwrap(), Value::Int(1));
        assert_eq!(vm.call("fib_iter", vec![Value::Int(10), Value::Int(0), Value::Int(1)]).unwrap(), Value::Int(55));
        assert_eq!(vm.call("fib_iter", vec![Value::Int(20), Value::Int(0), Value::Int(1)]).unwrap(), Value::Int(6765));
    }

    #[test]
    fn test_modulo_and_division() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 4;

        // div_mod(a, b) = (a / b, a % b) as tuple
        chunk.emit(Instruction::DivInt(2, 0, 1), 1);  // r2 = a / b
        chunk.emit(Instruction::ModInt(3, 0, 1), 1);  // r3 = a % b
        chunk.emit(Instruction::MakeTuple(0, vec![2, 3]), 1);
        chunk.emit(Instruction::Return(0), 1);

        let func = FunctionValue {
            name: "div_mod".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // 17 / 5 = 3, 17 % 5 = 2
        let result = vm.call("div_mod", vec![Value::Int(17), Value::Int(5)]).unwrap();
        assert_eq!(result, Value::Tuple(Rc::new(vec![Value::Int(3), Value::Int(2)])));

        // 100 / 7 = 14, 100 % 7 = 2
        let result = vm.call("div_mod", vec![Value::Int(100), Value::Int(7)]).unwrap();
        assert_eq!(result, Value::Tuple(Rc::new(vec![Value::Int(14), Value::Int(2)])));
    }

    #[test]
    fn test_comparison_chain() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 6;

        // is_between(x, low, high) = x >= low && x <= high
        // r0=x, r1=low, r2=high
        chunk.emit(Instruction::GeInt(3, 0, 1), 1);   // r3 = x >= low
        chunk.emit(Instruction::LeInt(4, 0, 2), 1);   // r4 = x <= high
        chunk.emit(Instruction::And(5, 3, 4), 1);    // r5 = r3 && r4
        chunk.emit(Instruction::Return(5), 1);

        let func = FunctionValue {
            name: "is_between".to_string(),
            arity: 3,
            param_names: vec!["x".to_string(), "low".to_string(), "high".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // 5 is between 1 and 10
        assert_eq!(vm.call("is_between", vec![Value::Int(5), Value::Int(1), Value::Int(10)]).unwrap(), Value::Bool(true));
        // 0 is not between 1 and 10
        assert_eq!(vm.call("is_between", vec![Value::Int(0), Value::Int(1), Value::Int(10)]).unwrap(), Value::Bool(false));
        // 10 is between 1 and 10 (inclusive)
        assert_eq!(vm.call("is_between", vec![Value::Int(10), Value::Int(1), Value::Int(10)]).unwrap(), Value::Bool(true));
        // 11 is not between 1 and 10
        assert_eq!(vm.call("is_between", vec![Value::Int(11), Value::Int(1), Value::Int(10)]).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_negation() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 2;

        // negate(x) = -x
        chunk.emit(Instruction::NegInt(1, 0), 1);
        chunk.emit(Instruction::Return(1), 1);

        let func = FunctionValue {
            name: "negate".to_string(),
            arity: 1,
            param_names: vec!["x".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        assert_eq!(vm.call("negate", vec![Value::Int(42)]).unwrap(), Value::Int(-42));
        assert_eq!(vm.call("negate", vec![Value::Int(-17)]).unwrap(), Value::Int(17));
        assert_eq!(vm.call("negate", vec![Value::Int(0)]).unwrap(), Value::Int(0));
    }

    #[test]
    fn test_equality() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;

        // eq(a, b) = a == b
        chunk.emit(Instruction::Eq(2, 0, 1), 1);
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "eq".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        // Test various types
        assert_eq!(vm.call("eq", vec![Value::Int(5), Value::Int(5)]).unwrap(), Value::Bool(true));
        assert_eq!(vm.call("eq", vec![Value::Int(5), Value::Int(6)]).unwrap(), Value::Bool(false));
        assert_eq!(vm.call("eq", vec![Value::Bool(true), Value::Bool(true)]).unwrap(), Value::Bool(true));
        assert_eq!(vm.call("eq", vec![Value::Unit, Value::Unit]).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_list_concat() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 3;

        // concat_lists(a, b) = a ++ b
        chunk.emit(Instruction::ListConcat(2, 0, 1), 1);
        chunk.emit(Instruction::Return(2), 1);

        let func = FunctionValue {
            name: "concat_lists".to_string(),
            arity: 2,
            param_names: vec!["a".to_string(), "b".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let list1 = Value::List(Rc::new(vec![Value::Int(1), Value::Int(2)]));
        let list2 = Value::List(Rc::new(vec![Value::Int(3), Value::Int(4)]));
        let result = vm.call("concat_lists", vec![list1, list2]).unwrap();
        assert_eq!(result, Value::List(Rc::new(vec![
            Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)
        ])));
    }

    #[test]
    fn test_make_list() {
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 4;

        // triple(a, b, c) = [a, b, c]
        chunk.emit(Instruction::MakeList(3, vec![0, 1, 2]), 1);
        chunk.emit(Instruction::Return(3), 1);

        let func = FunctionValue {
            name: "triple".to_string(),
            arity: 3,
            param_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let result = vm.call("triple", vec![Value::Int(1), Value::Int(2), Value::Int(3)]).unwrap();
        assert_eq!(result, Value::List(Rc::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)])));
    }

    #[test]
    fn test_eq_instruction_string_comparison() {
        // Tests that the generic Eq instruction compares strings by content, not pointer
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 5;

        // Load two strings (different heap allocations, same content)
        let str1_idx = chunk.add_constant(Value::String(Rc::new("hello".to_string())));
        let str2_idx = chunk.add_constant(Value::String(Rc::new("hello".to_string())));
        let str3_idx = chunk.add_constant(Value::String(Rc::new("world".to_string())));

        // r0 = "hello"
        chunk.emit(Instruction::LoadConst(0, str1_idx as u16), 1);
        // r1 = "hello" (different allocation)
        chunk.emit(Instruction::LoadConst(1, str2_idx as u16), 2);
        // r2 = "world"
        chunk.emit(Instruction::LoadConst(2, str3_idx as u16), 3);
        // r3 = r0 == r1 (should be true - same content)
        chunk.emit(Instruction::Eq(3, 0, 1), 4);
        // r4 = r0 == r2 (should be false - different content)
        chunk.emit(Instruction::Eq(4, 0, 2), 5);
        // Build tuple (r3, r4) for result
        chunk.emit(Instruction::MakeTuple(3, vec![3, 4]), 6);
        chunk.emit(Instruction::Return(3), 7);

        let func = FunctionValue {
            name: "test_eq".to_string(),
            arity: 0,
            param_names: vec![],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let result = vm.call("test_eq", vec![]).unwrap();
        // Should be (true, false)
        assert_eq!(result, Value::Tuple(Rc::new(vec![Value::Bool(true), Value::Bool(false)])));
    }

    #[test]
    fn test_eq_instruction_list_comparison() {
        // Tests that the generic Eq instruction compares lists by content, not pointer
        let mut vm = VM::new();

        let mut chunk = Chunk::new();
        chunk.register_count = 5;

        // Load two ints
        let int1_idx = chunk.add_constant(Value::Int(1));
        let int2_idx = chunk.add_constant(Value::Int(2));

        // r0 = 1
        chunk.emit(Instruction::LoadConst(0, int1_idx as u16), 1);
        // r1 = 2
        chunk.emit(Instruction::LoadConst(1, int2_idx as u16), 2);
        // r2 = [1, 2]
        chunk.emit(Instruction::MakeList(2, vec![0, 1]), 3);
        // r3 = [1, 2] (different allocation)
        chunk.emit(Instruction::MakeList(3, vec![0, 1]), 4);
        // r4 = r2 == r3 (should be true - same content)
        chunk.emit(Instruction::Eq(4, 2, 3), 5);
        chunk.emit(Instruction::Return(4), 6);

        let func = FunctionValue {
            name: "test_list_eq".to_string(),
            arity: 0,
            param_names: vec![],
            code: Rc::new(chunk),
            module: None,
            source_span: None,
            jit_code: None,
        };

        vm.register_function(func);

        let result = vm.call("test_list_eq", vec![]).unwrap();
        // Should be true - lists with same content are equal
        assert_eq!(result, Value::Bool(true));
    }
}
