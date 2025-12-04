//! Nostos Compiler
//!
//! Compiles AST to bytecode with:
//! - Type inference and checking
//! - Tail call detection
//! - Closure conversion
//! - Pattern match compilation

pub mod compile;

pub use compile::*;
