//! Nostos Virtual Machine
//!
//! A register-based bytecode VM designed for:
//! - JIT compilation with Cranelift
//! - Tail call optimization
//! - REPL/interactive use
//! - Full introspection
//! - Erlang-style lightweight processes

pub mod gc;
pub mod process;
pub mod runtime;
pub mod scheduler;
pub mod value;
pub mod vm;

pub use gc::*;
pub use process::*;
pub use runtime::*;
pub use scheduler::*;
pub use value::*;
pub use vm::*;
