//! Lexer for the Nostos programming language using logos.

use logos::Logos;
use std::fmt;

/// All tokens in the Nostos language.
#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"#[^*{\n][^\n]*")]          // Single-line comment: # ... (not #* or #{)
#[logos(skip r"#\*([^*]|\*[^#])*\*#")]    // Multi-line comment: #* ... *#
pub enum Token {
    // === Keywords ===
    #[token("type")]
    Type,
    #[token("var")]
    Var,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("when")]
    When,
    #[token("trait")]
    Trait,
    #[token("module")]
    Module,
    #[token("end")]
    End,
    #[token("use")]
    Use,
    #[token("private")]
    Private,
    #[token("self")]
    SelfKw,
    #[token("Self")]
    SelfType,

    // === Control flow ===
    #[token("try")]
    Try,
    #[token("catch")]
    Catch,
    #[token("finally")]
    Finally,
    #[token("do")]
    Do,

    // === Concurrency keywords ===
    #[token("spawn")]
    Spawn,
    #[token("spawn_link")]
    SpawnLink,
    #[token("spawn_monitor")]
    SpawnMonitor,
    #[token("receive")]
    Receive,
    #[token("after")]
    After,

    // === Error handling ===
    #[token("panic")]
    Panic,
    #[token("assert")]
    Assert,

    // === FFI ===
    #[token("extern")]
    Extern,
    #[token("from")]
    From,

    // === Testing ===
    #[token("test")]
    Test,

    // === Introspection ===
    #[token("quote")]
    Quote,

    // === Literals ===
    #[token("true")]
    True,
    #[token("false")]
    False,

    // Hex integer: 0xFF
    #[regex(r"0[xX][0-9a-fA-F][0-9a-fA-F_]*", |lex| {
        let s = lex.slice();
        i64::from_str_radix(&s[2..].replace('_', ""), 16).ok()
    })]
    HexInt(i64),

    // Binary integer: 0b1010
    #[regex(r"0[bB][01][01_]*", |lex| {
        let s = lex.slice();
        i64::from_str_radix(&s[2..].replace('_', ""), 2).ok()
    })]
    BinInt(i64),

    // Decimal integer
    #[regex(r"[0-9][0-9_]*", |lex| lex.slice().replace('_', "").parse::<i64>().ok())]
    Int(i64),

    // Float (including scientific notation)
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?", |lex| lex.slice().replace('_', ""))]
    Float(String),

    // String literal
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    String(String),

    // Character literal
    #[regex(r"'([^'\\]|\\.)'", |lex| {
        let s = lex.slice();
        parse_char(&s[1..s.len()-1])
    })]
    Char(char),

    // === Identifiers ===
    #[regex(r"[A-Z][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    UpperIdent(String),

    #[regex(r"[a-z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string(), priority = 1)]
    LowerIdent(String),

    // === Operators ===
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("**")]
    StarStar,

    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,

    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("!")]
    Bang,

    #[token("++")]
    PlusPlus,
    #[token("|>")]
    PipeRight,

    #[token("=")]
    Eq,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,

    #[token("<-")]
    LeftArrow,
    #[token("->")]
    RightArrow,
    #[token("=>")]
    FatArrow,

    #[token("^")]
    Caret,
    #[token("$")]
    Dollar,
    #[token("?")]
    Question,

    // === Delimiters ===
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(".")]
    Dot,
    #[token("|")]
    Pipe,
    #[token("_")]
    Underscore,
    #[token("#")]
    Hash,
}

/// Parse a character from escape sequence.
fn parse_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    match chars.next()? {
        '\\' => match chars.next()? {
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            '\\' => Some('\\'),
            '\'' => Some('\''),
            '"' => Some('"'),
            '0' => Some('\0'),
            _ => None,
        },
        c => Some(c),
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Type => write!(f, "type"),
            Token::Var => write!(f, "var"),
            Token::If => write!(f, "if"),
            Token::Then => write!(f, "then"),
            Token::Else => write!(f, "else"),
            Token::Match => write!(f, "match"),
            Token::When => write!(f, "when"),
            Token::Trait => write!(f, "trait"),
            Token::Module => write!(f, "module"),
            Token::End => write!(f, "end"),
            Token::Use => write!(f, "use"),
            Token::Private => write!(f, "private"),
            Token::SelfKw => write!(f, "self"),
            Token::SelfType => write!(f, "Self"),
            Token::Try => write!(f, "try"),
            Token::Catch => write!(f, "catch"),
            Token::Finally => write!(f, "finally"),
            Token::Do => write!(f, "do"),
            Token::Spawn => write!(f, "spawn"),
            Token::SpawnLink => write!(f, "spawn_link"),
            Token::SpawnMonitor => write!(f, "spawn_monitor"),
            Token::Receive => write!(f, "receive"),
            Token::After => write!(f, "after"),
            Token::Panic => write!(f, "panic"),
            Token::Assert => write!(f, "assert"),
            Token::Extern => write!(f, "extern"),
            Token::From => write!(f, "from"),
            Token::Test => write!(f, "test"),
            Token::Quote => write!(f, "quote"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::HexInt(n) => write!(f, "0x{:x}", n),
            Token::BinInt(n) => write!(f, "0b{:b}", n),
            Token::Int(n) => write!(f, "{}", n),
            Token::Float(s) => write!(f, "{}", s),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Char(c) => write!(f, "'{}'", c),
            Token::UpperIdent(s) => write!(f, "{}", s),
            Token::LowerIdent(s) => write!(f, "{}", s),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::StarStar => write!(f, "**"),
            Token::EqEq => write!(f, "=="),
            Token::NotEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::LtEq => write!(f, "<="),
            Token::GtEq => write!(f, ">="),
            Token::AndAnd => write!(f, "&&"),
            Token::OrOr => write!(f, "||"),
            Token::Bang => write!(f, "!"),
            Token::PlusPlus => write!(f, "++"),
            Token::PipeRight => write!(f, "|>"),
            Token::Eq => write!(f, "="),
            Token::PlusEq => write!(f, "+="),
            Token::MinusEq => write!(f, "-="),
            Token::StarEq => write!(f, "*="),
            Token::SlashEq => write!(f, "/="),
            Token::LeftArrow => write!(f, "<-"),
            Token::RightArrow => write!(f, "->"),
            Token::FatArrow => write!(f, "=>"),
            Token::Caret => write!(f, "^"),
            Token::Dollar => write!(f, "$"),
            Token::Question => write!(f, "?"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::Pipe => write!(f, "|"),
            Token::Underscore => write!(f, "_"),
            Token::Hash => write!(f, "#"),
        }
    }
}

/// Lex source code into tokens with spans.
pub fn lex(source: &str) -> impl Iterator<Item = (Token, std::ops::Range<usize>)> + '_ {
    Token::lexer(source)
        .spanned()
        .filter_map(|(tok, span)| tok.ok().map(|t| (t, span)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let tokens: Vec<_> = lex("type var if then else match when").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Type, Token::Var, Token::If, Token::Then,
            Token::Else, Token::Match, Token::When,
        ]);
    }

    #[test]
    fn test_concurrency_keywords() {
        let tokens: Vec<_> = lex("spawn spawn_link spawn_monitor receive after")
            .map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Spawn, Token::SpawnLink, Token::SpawnMonitor,
            Token::Receive, Token::After,
        ]);
    }

    #[test]
    fn test_new_keywords() {
        let tokens: Vec<_> = lex("try catch finally do test extern from panic assert")
            .map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Try, Token::Catch, Token::Finally, Token::Do,
            Token::Test, Token::Extern, Token::From, Token::Panic, Token::Assert,
        ]);
    }

    #[test]
    fn test_integers() {
        let tokens: Vec<_> = lex("42 0xFF 0b1010 1_000_000").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Int(42),
            Token::HexInt(0xFF),
            Token::BinInt(0b1010),
            Token::Int(1_000_000),
        ]);
    }

    #[test]
    fn test_floats() {
        let tokens: Vec<_> = lex("3.14 1.2e-10").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Float("3.14".to_string()),
            Token::Float("1.2e-10".to_string()),
        ]);
    }

    #[test]
    fn test_strings() {
        let tokens: Vec<_> = lex(r#""hello" "world""#).map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::String("hello".to_string()),
            Token::String("world".to_string()),
        ]);
    }

    #[test]
    fn test_chars() {
        let tokens: Vec<_> = lex(r"'a' '\n' '\t'").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Char('a'),
            Token::Char('\n'),
            Token::Char('\t'),
        ]);
    }

    #[test]
    fn test_identifiers() {
        let tokens: Vec<_> = lex("foo Bar _private Point123").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::LowerIdent("foo".to_string()),
            Token::UpperIdent("Bar".to_string()),
            Token::LowerIdent("_private".to_string()),
            Token::UpperIdent("Point123".to_string()),
        ]);
    }

    #[test]
    fn test_operators() {
        let tokens: Vec<_> = lex("+ - * / ** == != < > <= >= && || ++ <- -> => ? |>")
            .map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Plus, Token::Minus, Token::Star, Token::Slash, Token::StarStar,
            Token::EqEq, Token::NotEq, Token::Lt, Token::Gt, Token::LtEq, Token::GtEq,
            Token::AndAnd, Token::OrOr, Token::PlusPlus, Token::LeftArrow,
            Token::RightArrow, Token::FatArrow, Token::Question, Token::PipeRight,
        ]);
    }

    #[test]
    fn test_single_line_comment() {
        let tokens: Vec<_> = lex("foo # this is a comment\nbar").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::LowerIdent("foo".to_string()),
            Token::LowerIdent("bar".to_string()),
        ]);
    }

    #[test]
    fn test_multi_line_comment() {
        let tokens: Vec<_> = lex("foo #* multi\nline\ncomment *# bar").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::LowerIdent("foo".to_string()),
            Token::LowerIdent("bar".to_string()),
        ]);
    }

    #[test]
    fn test_send_operator() {
        let tokens: Vec<_> = lex("pid <- msg").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::LowerIdent("pid".to_string()),
            Token::LeftArrow,
            Token::LowerIdent("msg".to_string()),
        ]);
    }

    #[test]
    fn test_map_set_syntax() {
        let tokens: Vec<_> = lex("%{} #{}").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Percent, Token::LBrace, Token::RBrace,
            Token::Hash, Token::LBrace, Token::RBrace,
        ]);
    }
}
