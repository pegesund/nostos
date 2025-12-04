//! Nostos syntax crate - lexing and parsing.
//!
//! This crate provides the lexer and parser for the Nostos programming language.

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::{lex, Token};
pub use parser::{parse, parse_expr};
