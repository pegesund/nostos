//! Nostos REPL with call graph tracking for type-safe redefinitions.
//!
//! This crate provides a REPL session that maintains type safety across
//! function redefinitions using a call graph to track dependencies.

#![allow(
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::for_kv_map,
    clippy::if_same_then_else,
    clippy::map_entry,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::redundant_closure,
    clippy::unnecessary_map_or,
    clippy::unnecessary_to_owned,
    clippy::unwrap_or_default,
    clippy::into_iter_on_ref,
    clippy::manual_map,
    clippy::manual_strip,
    clippy::type_complexity
)]

pub mod callgraph;
pub mod session;
pub mod engine;

#[cfg(test)]
mod repl_tests;

pub use callgraph::CallGraph;
pub use session::{
    Definition, DefineSuccess, FunctionError, ReplError, ReplSession, TryDefineResult,
    extract_dependencies_from_fn,
};
pub use engine::{ReplEngine, ReplConfig, BrowserItem, CompileStatus, SaveCompileResult, SearchResult, PanelInfo, PanelState, NostletInfo};
pub use nostos_vm::{InspectEntry, ThreadSafeValue, ThreadSafeMapKey, SharedMap, SharedMapKey, SharedMapValue};
