//! Nostos Virtual Machine
//!
//! A register-based bytecode VM designed for:
#![allow(clippy::type_complexity)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::map_entry)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::if_same_then_else)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(clippy::format_collect)]
#![allow(clippy::approx_constant)]
//! - JIT compilation with Cranelift
//! - Tail call optimization
//! - REPL/interactive use
//! - Full introspection
//! - Erlang-style lightweight processes
//! - Multi-CPU parallel execution

pub mod async_vm;
pub mod cache;
pub mod extensions;
pub mod gc;
pub mod inspect;
pub mod io_runtime;
pub mod process;
pub mod scheduler;
pub mod shared_types;
pub mod supervisor;
pub mod value;

pub use gc::*;
pub use inspect::*;
pub use io_runtime::*;
pub use shared_types::*;
pub use process::*;
pub use scheduler::*;
pub use supervisor::*;
pub use value::*;
pub use extensions::ExtensionManager;
pub use async_vm::{ThreadedEvalHandle, DebugSession};
pub use async_vm::{enable_output_capture, disable_output_capture, is_output_capture_enabled};
pub use cache::{BytecodeCache, CachedModule, CachedFunction, CachedChunk, CachedValue, CacheManifest};
pub use cache::{function_to_cached, function_to_cached_with_fn_list, cached_to_function, cached_to_function_with_resolver, compute_file_hash};
pub use cache::{ModuleCache, CompiledModuleData};
