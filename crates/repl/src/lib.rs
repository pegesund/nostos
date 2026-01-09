//! Nostos REPL with call graph tracking for type-safe redefinitions.
//!
//! This crate provides a REPL session that maintains type safety across
//! function redefinitions using a call graph to track dependencies.

pub mod callgraph;
pub mod session;
pub mod engine;

pub use callgraph::CallGraph;
pub use session::{
    Definition, DefineSuccess, FunctionError, ReplError, ReplSession, TryDefineResult,
    extract_dependencies_from_fn,
};
pub use engine::{ReplEngine, ReplConfig, BrowserItem, CompileStatus, SaveCompileResult, SearchResult, PanelInfo, PanelState};
pub use nostos_vm::{InspectEntry, ThreadSafeValue, ThreadSafeMapKey, SharedMap, SharedMapKey, SharedMapValue};
