//! Nostos REPL with call graph tracking for type-safe redefinitions.
//!
//! This crate provides a REPL session that maintains type safety across
//! function redefinitions using a call graph to track dependencies.

pub mod callgraph;
pub mod session;

pub use callgraph::CallGraph;
pub use session::{
    Definition, DefineSuccess, FunctionError, ReplError, ReplSession, TryDefineResult,
};
