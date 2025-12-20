//! Virtual Machine for Nostos bytecode execution.
//!
//! Design goals:
//! - Tail call optimization (TCO) - recursive functions don't grow stack
//! - JIT-ready - can be replaced with compiled code
//! - REPL-friendly - supports incremental compilation
//! - Introspectable - full access to runtime state

use std::collections::HashMap;
use std::rc::Rc;

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
    /// Register file for this frame
    pub registers: Vec<Value>,
    /// Captured variables (for closures)
    pub captures: Vec<Value>,
    /// Return register in caller's frame
    pub return_reg: Option<Reg>,
}

/// The virtual machine state.
pub struct VM {
    /// Call stack
    pub frames: Vec<CallFrame>,
    /// Global variables
    pub globals: HashMap<String, Value>,
    /// Global functions
    pub functions: HashMap<String, Rc<FunctionValue>>,
    /// Native functions
    pub natives: HashMap<String, Rc<NativeFn>>,
    /// Type definitions (for introspection)
    pub types: HashMap<String, Rc<TypeValue>>,
    /// Exception handlers stack
    pub handlers: Vec<ExceptionHandler>,
    /// Current exception (if any)
    pub current_exception: Option<Value>,
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
        self.register_native("print", 1, |args| {
            Ok(Value::String(Rc::new(format!("{}", args[0]))))
        });

        self.register_native("println", 1, |args| {
            println!("{}", args[0]);
            Ok(Value::Unit)
        });

        self.register_native("typeOf", 1, |args| {
            Ok(Value::String(Rc::new(args[0].type_name().to_string())))
        });

        self.register_native("length", 1, |args| {
            match &args[0] {
                Value::List(items) => Ok(Value::Int(items.len() as i64)),
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                other => Err(RuntimeError::TypeError {
                    expected: "List, String, or Array".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("head", 1, |args| {
            match &args[0] {
                Value::List(items) if !items.is_empty() => Ok(items[0].clone()),
                Value::List(_) => Err(RuntimeError::MatchFailed),
                other => Err(RuntimeError::TypeError {
                    expected: "List".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("tail", 1, |args| {
            match &args[0] {
                Value::List(items) if !items.is_empty() => {
                    Ok(Value::List(Rc::new(items[1..].to_vec())))
                }
                Value::List(_) => Err(RuntimeError::MatchFailed),
                other => Err(RuntimeError::TypeError {
                    expected: "List".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("isEmpty", 1, |args| {
            match &args[0] {
                Value::List(items) => Ok(Value::Bool(items.is_empty())),
                Value::String(s) => Ok(Value::Bool(s.is_empty())),
                other => Err(RuntimeError::TypeError {
                    expected: "List or String".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("show", 1, |args| {
            Ok(Value::String(Rc::new(format!("{}", args[0]))))
        });

        self.register_native("toInt", 1, |args| {
            match &args[0] {
                Value::Int(i) => Ok(Value::Int(*i)),
                Value::Float(f) => Ok(Value::Int(*f as i64)),
                Value::String(s) => s.parse::<i64>()
                    .map(Value::Int)
                    .map_err(|_| RuntimeError::TypeError {
                        expected: "numeric string".to_string(),
                        found: format!("\"{}\"", s),
                    }),
                other => Err(RuntimeError::TypeError {
                    expected: "Int, Float, or String".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("toFloat", 1, |args| {
            match &args[0] {
                Value::Int(i) => Ok(Value::Float(*i as f64)),
                Value::Float(f) => Ok(Value::Float(*f)),
                Value::String(s) => s.parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| RuntimeError::TypeError {
                        expected: "numeric string".to_string(),
                        found: format!("\"{}\"", s),
                    }),
                other => Err(RuntimeError::TypeError {
                    expected: "Int, Float, or String".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("abs", 1, |args| {
            match &args[0] {
                Value::Int(i) => Ok(Value::Int(i.abs())),
                Value::Float(f) => Ok(Value::Float(f.abs())),
                other => Err(RuntimeError::TypeError {
                    expected: "Int or Float".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("sqrt", 1, |args| {
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(f.sqrt())),
                Value::Int(i) => Ok(Value::Float((*i as f64).sqrt())),
                other => Err(RuntimeError::TypeError {
                    expected: "Float or Int".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });

        self.register_native("panic", 1, |args| {
            match &args[0] {
                Value::String(s) => Err(RuntimeError::Panic(s.to_string())),
                other => Err(RuntimeError::Panic(format!("{}", other))),
            }
        });

        self.register_native("assert", 2, |args| {
            match &args[0] {
                Value::Bool(true) => Ok(Value::Unit),
                Value::Bool(false) => match &args[1] {
                    Value::String(msg) => Err(RuntimeError::AssertionFailed(msg.to_string())),
                    _ => Err(RuntimeError::AssertionFailed("assertion failed".to_string())),
                },
                other => Err(RuntimeError::TypeError {
                    expected: "Bool".to_string(),
                    found: other.type_name().to_string(),
                }),
            }
        });
    }

    pub fn register_native<F>(&mut self, name: &str, arity: usize, func: F)
    where
        F: Fn(&[Value]) -> VMResult + Send + Sync + 'static,
    {
        let native = Rc::new(NativeFn {
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
            return (native.func)(&args);
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
        let mut registers = vec![Value::Unit; func.code.register_count];
        for (i, arg) in args.into_iter().enumerate() {
            registers[i] = arg;
        }

        let frame = CallFrame {
            function: func,
            ip: 0,
            registers,
            captures,
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
                return Err(RuntimeError::StackOverflow);
            }

            let result = self.execute_step()?;

            match result {
                StepResult::Continue => {}
                StepResult::Return(value) => {
                    self.frames.pop();
                    if self.frames.is_empty() {
                        return Ok(value);
                    }
                    if let Some(frame) = self.frames.last_mut() {
                        if let Some(ret_reg) = frame.return_reg {
                            frame.registers[ret_reg as usize] = value;
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
                    frame.registers.resize(func.code.register_count, Value::Unit);
                    for (i, arg) in args.into_iter().enumerate() {
                        frame.registers[i] = arg;
                    }
                }
                StepResult::Call { func, args, captures, return_reg } => {
                    self.frames.last_mut().unwrap().return_reg = Some(return_reg);

                    let mut registers = vec![Value::Unit; func.code.register_count];
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

        macro_rules! get_int {
            ($r:expr) => {
                match reg!($r) {
                    Value::Int(i) => *i,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Int".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            };
        }

        macro_rules! get_float {
            ($r:expr) => {
                match reg!($r) {
                    Value::Float(f) => *f,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Float".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            };
        }

        macro_rules! get_bool {
            ($r:expr) => {
                match reg!($r) {
                    Value::Bool(b) => *b,
                    other => return Err(RuntimeError::TypeError {
                        expected: "Bool".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            };
        }

        macro_rules! get_string {
            ($r:expr) => {
                match reg!($r) {
                    Value::String(s) => s.clone(),
                    other => return Err(RuntimeError::TypeError {
                        expected: "String".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            };
        }

        match instr {
            Instruction::LoadConst(dst, idx) => {
                let value = constants[idx as usize].clone();
                set_reg!(dst, value);
            }
            Instruction::Move(dst, src) => {
                let value = reg!(src).clone();
                set_reg!(dst, value);
            }
            Instruction::LoadUnit(dst) => set_reg!(dst, Value::Unit),
            Instruction::LoadTrue(dst) => set_reg!(dst, Value::Bool(true)),
            Instruction::LoadFalse(dst) => set_reg!(dst, Value::Bool(false)),

            // Integer arithmetic
            Instruction::AddInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Int(a + b));
            }
            Instruction::SubInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Int(a - b));
            }
            Instruction::MulInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Int(a * b));
            }
            Instruction::DivInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                if b == 0 { return Err(RuntimeError::DivisionByZero); }
                set_reg!(dst, Value::Int(a / b));
            }
            Instruction::ModInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                if b == 0 { return Err(RuntimeError::DivisionByZero); }
                set_reg!(dst, Value::Int(a % b));
            }
            Instruction::NegInt(dst, src) => {
                let v = get_int!(src);
                set_reg!(dst, Value::Int(-v));
            }

            // Float arithmetic
            Instruction::AddFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Float(a + b));
            }
            Instruction::SubFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Float(a - b));
            }
            Instruction::MulFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Float(a * b));
            }
            Instruction::DivFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Float(a / b));
            }
            Instruction::NegFloat(dst, src) => {
                let v = get_float!(src);
                set_reg!(dst, Value::Float(-v));
            }
            Instruction::PowFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Float(a.powf(b)));
            }

            // Comparisons
            Instruction::EqInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Bool(a == b));
            }
            Instruction::NeInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Bool(a != b));
            }
            Instruction::LtInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Bool(a < b));
            }
            Instruction::LeInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Bool(a <= b));
            }
            Instruction::GtInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Bool(a > b));
            }
            Instruction::GeInt(dst, a, b) => {
                let a = get_int!(a);
                let b = get_int!(b);
                set_reg!(dst, Value::Bool(a >= b));
            }
            Instruction::EqFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Bool(a == b));
            }
            Instruction::LtFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Bool(a < b));
            }
            Instruction::LeFloat(dst, a, b) => {
                let a = get_float!(a);
                let b = get_float!(b);
                set_reg!(dst, Value::Bool(a <= b));
            }
            Instruction::EqBool(dst, a, b) => {
                let a = get_bool!(a);
                let b = get_bool!(b);
                set_reg!(dst, Value::Bool(a == b));
            }
            Instruction::EqStr(dst, a, b) => {
                let a = get_string!(a);
                let b = get_string!(b);
                set_reg!(dst, Value::Bool(a == b));
            }
            Instruction::Eq(dst, a, b) => {
                let a = reg!(a).clone();
                let b = reg!(b).clone();
                set_reg!(dst, Value::Bool(a == b));
            }

            // Logical
            Instruction::Not(dst, src) => {
                let v = get_bool!(src);
                set_reg!(dst, Value::Bool(!v));
            }
            Instruction::And(dst, a, b) => {
                let a = get_bool!(a);
                let b = get_bool!(b);
                set_reg!(dst, Value::Bool(a && b));
            }
            Instruction::Or(dst, a, b) => {
                let a = get_bool!(a);
                let b = get_bool!(b);
                set_reg!(dst, Value::Bool(a || b));
            }

            // String
            Instruction::Concat(dst, a, b) => {
                let a = get_string!(a);
                let b = get_string!(b);
                set_reg!(dst, Value::String(Rc::new(format!("{}{}", a, b))));
            }

            // Collections
            Instruction::MakeList(dst, regs) => {
                let items: Vec<Value> = regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                set_reg!(dst, Value::List(Rc::new(items)));
            }
            Instruction::MakeTuple(dst, regs) => {
                let items: Vec<Value> = regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                set_reg!(dst, Value::Tuple(Rc::new(items)));
            }
            Instruction::Cons(dst, head, tail) => {
                let head = reg!(head).clone();
                let tail = match reg!(tail) {
                    Value::List(items) => items.clone(),
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name().to_string(),
                    }),
                };
                let mut new_list = vec![head];
                new_list.extend(tail.iter().cloned());
                set_reg!(dst, Value::List(Rc::new(new_list)));
            }
            Instruction::ListConcat(dst, a, b) => {
                let a = match reg!(a) {
                    Value::List(items) => items.clone(),
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name().to_string(),
                    }),
                };
                let b = match reg!(b) {
                    Value::List(items) => items.clone(),
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name().to_string(),
                    }),
                };
                let mut new_list = a.to_vec();
                new_list.extend(b.iter().cloned());
                set_reg!(dst, Value::List(Rc::new(new_list)));
            }
            Instruction::Index(dst, coll, idx) => {
                let idx = get_int!(idx) as usize;
                let value = match reg!(coll) {
                    Value::List(items) => items.get(idx).cloned()
                        .ok_or(RuntimeError::IndexOutOfBounds { index: idx as i64, length: items.len() })?,
                    Value::Tuple(items) => items.get(idx).cloned()
                        .ok_or(RuntimeError::IndexOutOfBounds { index: idx as i64, length: items.len() })?,
                    Value::Array(items) => items.borrow().get(idx).cloned()
                        .ok_or(RuntimeError::IndexOutOfBounds { index: idx as i64, length: items.borrow().len() })?,
                    other => return Err(RuntimeError::TypeError {
                        expected: "List, Tuple, or Array".to_string(),
                        found: other.type_name().to_string(),
                    }),
                };
                set_reg!(dst, value);
            }
            Instruction::Length(dst, src) => {
                let len = match reg!(src) {
                    Value::List(items) => items.len(),
                    Value::Tuple(items) => items.len(),
                    Value::Array(items) => items.borrow().len(),
                    Value::String(s) => s.len(),
                    other => return Err(RuntimeError::TypeError {
                        expected: "List, Tuple, Array, or String".to_string(),
                        found: other.type_name().to_string(),
                    }),
                };
                set_reg!(dst, Value::Int(len as i64));
            }
            Instruction::MakeMap(dst, pairs) => {
                let mut map = HashMap::new();
                for (k, v) in pairs {
                    let key = reg!(k).to_map_key()
                        .ok_or_else(|| RuntimeError::TypeError {
                            expected: "hashable type".to_string(),
                            found: reg!(k).type_name().to_string(),
                        })?;
                    let value = reg!(v).clone();
                    map.insert(key, value);
                }
                set_reg!(dst, Value::Map(Rc::new(map)));
            }
            Instruction::MakeSet(dst, regs) => {
                let mut set = std::collections::HashSet::new();
                for r in regs {
                    let key = reg!(r).to_map_key()
                        .ok_or_else(|| RuntimeError::TypeError {
                            expected: "hashable type".to_string(),
                            found: reg!(r).type_name().to_string(),
                        })?;
                    set.insert(key);
                }
                set_reg!(dst, Value::Set(Rc::new(set)));
            }
            Instruction::IndexSet(coll, idx, val) => {
                let idx = get_int!(idx) as usize;
                let value = reg!(val).clone();
                match reg!(coll) {
                    Value::Array(items) => {
                        let mut items = items.borrow_mut();
                        if idx >= items.len() {
                            return Err(RuntimeError::IndexOutOfBounds {
                                index: idx as i64,
                                length: items.len(),
                            });
                        }
                        items[idx] = value;
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Array".to_string(),
                        found: other.type_name().to_string(),
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
                let fields: Vec<Value> = field_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                set_reg!(dst, Value::Record(Rc::new(RecordValue {
                    type_name,
                    field_names,
                    fields,
                    mutable_fields,
                })));
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
                    Value::Record(rec) => {
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::UnknownField {
                                type_name: rec.type_name.clone(),
                                field: field_name.clone(),
                            })?;
                        let value = rec.fields[idx].clone();
                        set_reg!(dst, value);
                    }
                    Value::Tuple(items) => {
                        // Support tuple field access with numeric indices (t.0, t.1, etc.)
                        let idx: usize = field_name.parse().map_err(|_| RuntimeError::UnknownField {
                            type_name: "Tuple".to_string(),
                            field: field_name.clone(),
                        })?;
                        let value = items.get(idx).cloned().ok_or_else(|| RuntimeError::IndexOutOfBounds {
                            index: idx as i64,
                            length: items.len(),
                        })?;
                        set_reg!(dst, value);
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Record or Tuple".to_string(),
                        found: other.type_name().to_string(),
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
                    Value::Record(rec) => {
                        let idx = rec.field_names.iter().position(|n| n == &field_name)
                            .ok_or_else(|| RuntimeError::UnknownField {
                                type_name: rec.type_name.clone(),
                                field: field_name.clone(),
                            })?;
                        if !rec.mutable_fields[idx] {
                            return Err(RuntimeError::ImmutableField { field: field_name });
                        }
                        let mut new_rec = (*rec).clone();
                        new_rec.fields[idx] = new_value;
                        set_reg!(record, Value::Record(Rc::new(new_rec)));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Record".to_string(),
                        found: other.type_name().to_string(),
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
                    Value::Record(rec) => {
                        let mut new_fields = rec.fields.clone();
                        // Update the specified fields (for now, assumes positional)
                        for (i, &r) in field_regs.iter().enumerate() {
                            if i < new_fields.len() {
                                new_fields[i] = reg!(r).clone();
                            }
                        }
                        set_reg!(dst, Value::Record(Rc::new(RecordValue {
                            type_name,
                            field_names: rec.field_names.clone(),
                            fields: new_fields,
                            mutable_fields: rec.mutable_fields.clone(),
                        })));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Record".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            }

            // Tuples
            Instruction::GetTupleField(dst, tuple, idx) => {
                match reg!(tuple) {
                    Value::Tuple(items) => {
                        if (idx as usize) < items.len() {
                            set_reg!(dst, items[idx as usize].clone());
                        } else {
                            return Err(RuntimeError::IndexOutOfBounds {
                                index: idx as i64,
                                length: items.len(),
                            });
                        }
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Tuple".to_string(),
                        found: other.type_name().to_string(),
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
                let fields: Vec<Value> = field_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                set_reg!(dst, Value::Variant(Rc::new(VariantValue {
                    type_name,
                    constructor,
                    fields,
                    named_fields: None,
                })));
            }
            Instruction::GetTag(dst, variant) => {
                match reg!(variant) {
                    Value::Variant(v) => {
                        set_reg!(dst, Value::String(Rc::new(v.constructor.clone())));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Variant".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            }
            Instruction::GetVariantField(dst, variant, field_idx) => {
                match reg!(variant) {
                    Value::Variant(v) => {
                        let value = v.fields.get(field_idx as usize).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds {
                                index: field_idx as i64,
                                length: v.fields.len(),
                            })?;
                        set_reg!(dst, value);
                    }
                    // Also support Record types for unified handling
                    Value::Record(r) => {
                        let value = r.fields.get(field_idx as usize).cloned()
                            .ok_or(RuntimeError::IndexOutOfBounds {
                                index: field_idx as i64,
                                length: r.fields.len(),
                            })?;
                        set_reg!(dst, value);
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Variant".to_string(),
                        found: other.type_name().to_string(),
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
                    Value::Variant(v) => v.constructor == ctor_name,
                    // Also support Record types for unified handling
                    Value::Record(r) => r.type_name == ctor_name,
                    _ => false,
                };
                set_reg!(dst, Value::Bool(result));
            }
            Instruction::TestUnit(dst, value) => {
                let result = matches!(reg!(value), Value::Unit);
                set_reg!(dst, Value::Bool(result));
            }
            Instruction::TestConst(dst, value, const_idx) => {
                let constant = &constants[const_idx as usize];
                let result = reg!(value) == constant;
                set_reg!(dst, Value::Bool(result));
            }
            Instruction::TestNil(dst, list) => {
                let result = match reg!(list) {
                    Value::List(items) => items.is_empty(),
                    _ => false,
                };
                set_reg!(dst, Value::Bool(result));
            }
            Instruction::Decons(head, tail, list) => {
                match reg!(list).clone() {
                    Value::List(items) if !items.is_empty() => {
                        set_reg!(head, items[0].clone());
                        set_reg!(tail, Value::List(Rc::new(items[1..].to_vec())));
                    }
                    Value::List(_) => return Err(RuntimeError::MatchFailed),
                    other => return Err(RuntimeError::TypeError {
                        expected: "List".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            }

            // Function calls
            Instruction::Call(dst, func_reg, arg_regs) => {
                let args: Vec<Value> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                match reg!(func_reg).clone() {
                    Value::Function(func) => {
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
                    Value::Closure(closure) => {
                        if args.len() != closure.function.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: closure.function.arity,
                                found: args.len(),
                            });
                        }
                        return Ok(StepResult::Call {
                            func: closure.function.clone(),
                            args,
                            captures: closure.captures.clone(),
                            return_reg: dst,
                        });
                    }
                    Value::NativeFunction(native) => {
                        if args.len() != native.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: native.arity,
                                found: args.len(),
                            });
                        }
                        let result = (native.func)(&args)?;
                        set_reg!(dst, result);
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: other.type_name().to_string(),
                    }),
                }
            }
            Instruction::TailCall(func_reg, arg_regs) => {
                let args: Vec<Value> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                match reg!(func_reg).clone() {
                    Value::Function(func) => {
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
                    Value::Closure(closure) => {
                        if args.len() != closure.function.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: closure.function.arity,
                                found: args.len(),
                            });
                        }
                        return Ok(StepResult::TailCall {
                            func: closure.function.clone(),
                            args,
                            captures: closure.captures.clone(),
                        });
                    }
                    Value::NativeFunction(native) => {
                        if args.len() != native.arity {
                            return Err(RuntimeError::ArityMismatch {
                                expected: native.arity,
                                found: args.len(),
                            });
                        }
                        let result = (native.func)(&args)?;
                        return Ok(StepResult::Return(result));
                    }
                    other => return Err(RuntimeError::TypeError {
                        expected: "Function".to_string(),
                        found: other.type_name().to_string(),
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
                let args: Vec<Value> = arg_regs.iter()
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
                let args: Vec<Value> = arg_regs.iter()
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
                let args: Vec<Value> = arg_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();

                let native = self.natives.get(&name).cloned()
                    .ok_or_else(|| RuntimeError::UnknownFunction(name))?;

                if args.len() != native.arity {
                    return Err(RuntimeError::ArityMismatch {
                        expected: native.arity,
                        found: args.len(),
                    });
                }
                let result = (native.func)(&args)?;
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
                let captures: Vec<Value> = capture_regs.iter()
                    .map(|r| reg!(*r).clone())
                    .collect();
                let capture_names: Vec<String> = (0..captures.len())
                    .map(|i| format!("capture_{}", i))
                    .collect();

                set_reg!(dst, Value::Closure(Rc::new(ClosureValue {
                    function: func,
                    captures,
                    capture_names,
                })));
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
                let exc = self.current_exception.take().unwrap_or(Value::Unit);
                set_reg!(dst, exc);
            }

            // Introspection
            Instruction::TypeOf(dst, src) => {
                let type_name = reg!(src).type_name().to_string();
                set_reg!(dst, Value::String(Rc::new(type_name)));
            }

            // Concurrency (stubs)
            Instruction::SelfPid(dst) => {
                set_reg!(dst, Value::Pid(Pid(0)));
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
    Return(Value),
    TailCall {
        func: Rc<FunctionValue>,
        args: Vec<Value>,
        captures: Vec<Value>,
    },
    Call {
        func: Rc<FunctionValue>,
        args: Vec<Value>,
        captures: Vec<Value>,
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
}
