//! Nostos syntax crate - lexing and parsing.
//!
//! This crate provides the lexer and parser for the Nostos programming language.

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::{lex, Token};
pub use parser::{parse, parse_expr};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_nested_module() {
        let source = "module Outer module Inner value() = 21 end end";
        let tokens: Vec<_> = lex(source).collect();
        assert!(tokens.iter().any(|(tok, _)| matches!(tok, Token::Module)));
        assert!(tokens.iter().any(|(tok, _)| matches!(tok, Token::End)));
    }

    #[test]
    fn test_parse_nested_module() {
        let source = "module Outer module Inner value() = 21 end end";
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_nested_module_with_qualified_call() {
        let source = "
            module Outer
                module Inner
                    value() = 21
                end
            end
            main() = Outer.Inner.value() * 2
        ";
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        assert!(result.is_some());
    }
}
