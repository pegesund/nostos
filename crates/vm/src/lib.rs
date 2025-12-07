//! Nostos Virtual Machine
//!
//! A register-based bytecode VM designed for:
//! - JIT compilation with Cranelift
//! - Tail call optimization
//! - REPL/interactive use
//! - Full introspection
//! - Erlang-style lightweight processes
//! - Multi-CPU parallel execution

pub mod gc;
pub mod io_runtime;
pub mod parallel;
pub mod process;
pub mod scheduler;
pub mod supervisor;
pub mod value;
pub mod worker;

pub use gc::*;
pub use io_runtime::*;
pub use parallel::*;
pub use process::*;
pub use scheduler::*;
pub use supervisor::*;
pub use value::*;
pub use worker::*;
