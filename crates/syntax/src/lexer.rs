//! Lexer for the Nostos programming language using logos.

use logos::Logos;
use std::fmt;

/// All tokens in the Nostos language.
#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\r]+")]                // Skip spaces and tabs, but NOT newlines
pub enum Token {
    // Comments - captured as tokens for syntax highlighting
    #[regex(r"#[^*{\n][^\n]*")]           // Single-line comment: # ... (not #* or #{)
    Comment,

    #[regex(r"#\*([^*]|\*[^#])*\*#")]    // Multi-line comment: #* ... *#
    MultiLineComment,
    // Newline token - used as implicit statement separator
    #[regex(r"\n+")]
    Newline,

    // === Keywords ===
    #[token("type")]
    Type,
    #[token("var")]
    Var,
    #[token("mvar")]
    Mvar,
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
    #[token("import")]
    Import,
    #[token("private")]
    Private,
    #[token("pub")]
    Pub,
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

    // === Loops ===
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("to")]
    To,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,

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
    // Note: assert is now a builtin function, not a keyword

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

    // === Typed integer literals (must come before generic Int) ===
    // Int8: 42i8
    #[regex(r"[0-9][0-9_]*i8", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-2].replace('_', "").parse::<i8>().ok()
    })]
    Int8(i8),

    // Int16: 42i16
    #[regex(r"[0-9][0-9_]*i16", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-3].replace('_', "").parse::<i16>().ok()
    })]
    Int16(i16),

    // Int32: 42i32
    #[regex(r"[0-9][0-9_]*i32", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-3].replace('_', "").parse::<i32>().ok()
    })]
    Int32(i32),

    // UInt8: 42u8
    #[regex(r"[0-9][0-9_]*u8", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-2].replace('_', "").parse::<u8>().ok()
    })]
    UInt8(u8),

    // UInt16: 42u16
    #[regex(r"[0-9][0-9_]*u16", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-3].replace('_', "").parse::<u16>().ok()
    })]
    UInt16(u16),

    // UInt32: 42u32
    #[regex(r"[0-9][0-9_]*u32", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-3].replace('_', "").parse::<u32>().ok()
    })]
    UInt32(u32),

    // UInt64: 42u64
    #[regex(r"[0-9][0-9_]*u64", priority = 3, callback = |lex| {
        let s = lex.slice();
        s[..s.len()-3].replace('_', "").parse::<u64>().ok()
    })]
    UInt64(u64),

    // BigInt: 42n
    #[regex(r"[0-9][0-9_]*n", priority = 3, callback = |lex| {
        let s = lex.slice();
        Some(s[..s.len()-1].replace('_', ""))
    })]
    BigInt(String),

    // Decimal integer
    #[regex(r"[0-9][0-9_]*", |lex| lex.slice().replace('_', "").parse::<i64>().ok())]
    Int(i64),

    // === Typed float literals (must come before generic Float) ===
    // Float32: 3.14f32
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?f32", priority = 3, callback = |lex| {
        let s = lex.slice();
        Some(s[..s.len()-3].replace('_', ""))
    })]
    Float32(String),

    // Decimal: 3.14d or 42d
    #[regex(r"[0-9][0-9_]*(\.[0-9][0-9_]*)?d", priority = 3, callback = |lex| {
        let s = lex.slice();
        Some(s[..s.len()-1].replace('_', ""))
    })]
    Decimal(String),

    // Float (including scientific notation) - default is Float64
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?", |lex| lex.slice().replace('_', ""))]
    Float(String),

    // String literal
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(parse_string_escapes(&s[1..s.len()-1]))
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
    #[token(";")]
    Semicolon,
    #[token("::")]
    ColonColon,
    #[token(":")]
    Colon,
    #[token(".")]
    Dot,
    #[token("|")]
    Pipe,
    #[token("_")]
    Underscore,
    #[regex(r"#", hash_callback)]
    Hash,
}

/// Callback for Hash token - only match if followed by '{' (for set literals)
/// Otherwise it's an empty comment and should be skipped
fn hash_callback(lex: &mut logos::Lexer<Token>) -> logos::Filter<()> {
    // Check if next char is '{'
    if lex.remainder().starts_with('{') {
        logos::Filter::Emit(())
    } else {
        logos::Filter::Skip
    }
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

/// Parse string with escape sequences.
fn parse_string_escapes(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                let escaped = match next {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    '0' => '\0',
                    _ => {
                        result.push(c);
                        continue;
                    }
                };
                chars.next();
                result.push(escaped);
            } else {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Type => write!(f, "type"),
            Token::Var => write!(f, "var"),
            Token::Mvar => write!(f, "mvar"),
            Token::If => write!(f, "if"),
            Token::Then => write!(f, "then"),
            Token::Else => write!(f, "else"),
            Token::Match => write!(f, "match"),
            Token::When => write!(f, "when"),
            Token::Trait => write!(f, "trait"),
            Token::Module => write!(f, "module"),
            Token::End => write!(f, "end"),
            Token::Use => write!(f, "use"),
            Token::Import => write!(f, "import"),
            Token::Private => write!(f, "private"),
            Token::Pub => write!(f, "pub"),
            Token::SelfKw => write!(f, "self"),
            Token::SelfType => write!(f, "Self"),
            Token::Try => write!(f, "try"),
            Token::Catch => write!(f, "catch"),
            Token::Finally => write!(f, "finally"),
            Token::Do => write!(f, "do"),
            Token::While => write!(f, "while"),
            Token::For => write!(f, "for"),
            Token::To => write!(f, "to"),
            Token::Break => write!(f, "break"),
            Token::Continue => write!(f, "continue"),
            Token::Spawn => write!(f, "spawn"),
            Token::SpawnLink => write!(f, "spawn_link"),
            Token::SpawnMonitor => write!(f, "spawn_monitor"),
            Token::Receive => write!(f, "receive"),
            Token::After => write!(f, "after"),
            Token::Panic => write!(f, "panic"),
            Token::Extern => write!(f, "extern"),
            Token::From => write!(f, "from"),
            Token::Test => write!(f, "test"),
            Token::Quote => write!(f, "quote"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::HexInt(n) => write!(f, "0x{:x}", n),
            Token::BinInt(n) => write!(f, "0b{:b}", n),
            Token::Int8(n) => write!(f, "{}i8", n),
            Token::Int16(n) => write!(f, "{}i16", n),
            Token::Int32(n) => write!(f, "{}i32", n),
            Token::UInt8(n) => write!(f, "{}u8", n),
            Token::UInt16(n) => write!(f, "{}u16", n),
            Token::UInt32(n) => write!(f, "{}u32", n),
            Token::UInt64(n) => write!(f, "{}u64", n),
            Token::BigInt(s) => write!(f, "{}n", s),
            Token::Int(n) => write!(f, "{}", n),
            Token::Float32(s) => write!(f, "{}f32", s),
            Token::Decimal(s) => write!(f, "{}d", s),
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
            Token::Semicolon => write!(f, ";"),
            Token::ColonColon => write!(f, "::"),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::Pipe => write!(f, "|"),
            Token::Underscore => write!(f, "_"),
            Token::Hash => write!(f, "#"),
            Token::Newline => write!(f, "newline"),
            Token::Comment => write!(f, "comment"),
            Token::MultiLineComment => write!(f, "comment"),
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
        let tokens: Vec<_> = lex("type var mvar if then else match when").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Type, Token::Var, Token::Mvar, Token::If, Token::Then,
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
        let tokens: Vec<_> = lex("try catch finally do test extern from panic")
            .map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Try, Token::Catch, Token::Finally, Token::Do,
            Token::Test, Token::Extern, Token::From, Token::Panic,
        ]);
    }

    #[test]
    fn test_assert_is_identifier() {
        // assert is a builtin function, not a keyword
        let tokens: Vec<_> = lex("assert").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![Token::LowerIdent("assert".to_string())]);
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
        // Comments are now kept as tokens for syntax highlighting
        let tokens: Vec<_> = lex("foo # this is a comment\nbar").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::LowerIdent("foo".to_string()),
            Token::Comment,
            Token::Newline,
            Token::LowerIdent("bar".to_string()),
        ]);
    }

    #[test]
    fn test_multi_line_comment() {
        // Comments are now kept as tokens for syntax highlighting
        let tokens: Vec<_> = lex("foo #* multi\nline\ncomment *# bar").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::LowerIdent("foo".to_string()),
            Token::MultiLineComment,
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

    #[test]
    fn test_typed_integers() {
        let tokens: Vec<_> = lex("42i8 100i16 1000i32 42u8 100u16 1000u32 1000u64")
            .map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Int8(42),
            Token::Int16(100),
            Token::Int32(1000),
            Token::UInt8(42),
            Token::UInt16(100),
            Token::UInt32(1000),
            Token::UInt64(1000),
        ]);
    }

    #[test]
    fn test_bigint() {
        let tokens: Vec<_> = lex("42n 1_000_000n").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::BigInt("42".to_string()),
            Token::BigInt("1000000".to_string()),
        ]);
    }

    #[test]
    fn test_typed_floats() {
        let tokens: Vec<_> = lex("3.14f32 2.5d 42d").map(|(t, _)| t).collect();
        assert_eq!(tokens, vec![
            Token::Float32("3.14".to_string()),
            Token::Decimal("2.5".to_string()),
            Token::Decimal("42".to_string()),
        ]);
    }
}
