//! Nostos Virtual Machine
//!
//! A register-based bytecode VM designed for:
//! - JIT compilation with Cranelift
//! - Tail call optimization
//! - REPL/interactive use
//! - Full introspection
//! - Erlang-style lightweight processes
//! - Multi-CPU parallel execution

pub mod async_vm;
pub mod gc;
pub mod inspect;
pub mod io_runtime;
pub mod process;
pub mod scheduler;
pub mod shared_types;
pub mod supervisor;
pub mod value;
pub mod worker;

pub use gc::*;
pub use inspect::*;
pub use io_runtime::*;
pub use shared_types::*;
pub use process::*;
pub use scheduler::*;
pub use supervisor::*;
pub use value::*;
pub use worker::*;
pub use async_vm::ThreadedEvalHandle;
