//! Parser for the Nostos programming language using chumsky.

use chumsky::prelude::*;

use crate::ast::*;
use crate::lexer::Token;

/// Parse error type
pub type ParseError<'a> = Simple<Token>;

/// Create an identifier from a token span.
fn make_ident(name: String, span: Span) -> Ident {
    Spanned::new(name, span)
}

/// Convert chumsky span to our Span
fn to_span(span: std::ops::Range<usize>) -> Span {
    Span::new(span.start, span.end)
}

/// Convert an expression to a pattern (for when we parsed something as an expression
/// but it turned out to be a let binding with `=`).
fn expr_to_pattern(expr: Expr) -> Option<Pattern> {
    match expr {
        Expr::Var(id) => Some(Pattern::Var(id)),
        Expr::Int(n, span) => Some(Pattern::Int(n, span)),
        Expr::Float(f, span) => Some(Pattern::Float(f, span)),
        Expr::String(StringLit::Plain(s), span) => Some(Pattern::String(s, span)),
        Expr::Char(c, span) => Some(Pattern::Char(c, span)),
        Expr::Bool(b, span) => Some(Pattern::Bool(b, span)),
        Expr::Unit(span) => Some(Pattern::Unit(span)),
        Expr::Wildcard(span) => Some(Pattern::Wildcard(span)),
        // Negative integer literals: -1, -42, etc.
        Expr::UnaryOp(UnaryOp::Neg, inner, span) => {
            match *inner {
                Expr::Int(n, _) => Some(Pattern::Int(-n, span)),
                _ => None,
            }
        }
        Expr::Tuple(exprs, span) => {
            let pats: Option<Vec<Pattern>> = exprs.into_iter().map(expr_to_pattern).collect();
            pats.map(|p| Pattern::Tuple(p, span))
        }
        Expr::List(exprs, tail, span) => {
            let pats: Option<Vec<Pattern>> = exprs.into_iter().map(expr_to_pattern).collect();
            let tail_pat: Option<Option<Box<Pattern>>> = match tail {
                Some(t) => expr_to_pattern(*t).map(|p| Some(Box::new(p))),
                None => Some(None),
            };
            match (pats, tail_pat) {
                (Some(heads), Some(tail)) => {
                    if heads.is_empty() && tail.is_none() {
                        Some(Pattern::List(ListPattern::Empty, span))
                    } else {
                        Some(Pattern::List(ListPattern::Cons(heads, tail), span))
                    }
                }
                _ => None,
            }
        }
        Expr::Record(name, fields, span) => {
            // Convert record expression to variant pattern
            let pat_fields: Option<Vec<Pattern>> = fields
                .into_iter()
                .map(|f| match f {
                    RecordField::Positional(e) => expr_to_pattern(e),
                    RecordField::Named(_, _) => None, // Named fields need different handling
                })
                .collect();
            pat_fields.map(|pf| {
                Pattern::Variant(name, VariantPatternFields::Positional(pf), span)
            })
        }
        // Complex expressions (with operators, calls, etc.) cannot be patterns
        _ => None,
    }
}


/// Parser for visibility modifier.
/// Returns Public if `pub` keyword is present, Private otherwise.
fn visibility() -> impl Parser<Token, Visibility, Error = Simple<Token>> + Clone {
    just(Token::Pub)
        .or_not()
        .map(|pub_token| {
            if pub_token.is_some() {
                Visibility::Public
            } else {
                Visibility::Private
            }
        })
}

/// Parser for identifiers (lowercase).
fn ident() -> impl Parser<Token, Ident, Error = Simple<Token>> + Clone {
    filter_map(|span, tok| match tok {
        Token::LowerIdent(s) => Ok(make_ident(s, to_span(span))),
        Token::SelfKw => Ok(make_ident("self".to_string(), to_span(span))),
        // panic is a builtin function, not a reserved keyword
        Token::Panic => Ok(make_ident("panic".to_string(), to_span(span))),
        Token::Test | Token::Type | Token::Var | Token::Mvar | Token::If | Token::Then | Token::Else |
        Token::Match | Token::When | Token::Trait | Token::Module | Token::End |
        Token::Use | Token::Private | Token::Pub | Token::Try | Token::Catch |
        Token::Finally | Token::Do | Token::While | Token::For | Token::To |
        Token::Break | Token::Continue | Token::Return | Token::Spawn | Token::SpawnLink |
        Token::SpawnMonitor | Token::Receive | Token::After |
        Token::Extern | Token::From | Token::Quote =>
            Err(Simple::custom(span, format!("'{}' is a reserved keyword", tok))),
        _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
    })
}

/// Parser for type names (uppercase).
fn type_name() -> impl Parser<Token, Ident, Error = Simple<Token>> + Clone {
    filter_map(|span, tok| match tok {
        Token::UpperIdent(s) => Ok(make_ident(s, to_span(span))),
        Token::SelfType => Ok(make_ident("Self".to_string(), to_span(span))),
        _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
    })
}

/// Parser for any identifier (lower or upper).
fn any_ident() -> impl Parser<Token, Ident, Error = Simple<Token>> + Clone {
    filter_map(|span, tok| match tok {
        Token::LowerIdent(s) => Ok(make_ident(s, to_span(span))),
        Token::UpperIdent(s) => Ok(make_ident(s, to_span(span))),
        Token::SelfKw => Ok(make_ident("self".to_string(), to_span(span))),
        Token::SelfType => Ok(make_ident("Self".to_string(), to_span(span))),
        _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
    })
}

/// Parse a string into a StringLit, handling `${expr}` interpolation.
fn parse_string_lit(s: &str) -> StringLit {
    if !s.contains("${") {
        return StringLit::Plain(s.to_string());
    }

    let mut parts: Vec<StringPart> = Vec::new();
    let mut remaining = s;

    while !remaining.is_empty() {
        if let Some(start) = remaining.find("${") {
            // Add literal part before ${
            if start > 0 {
                parts.push(StringPart::Lit(remaining[..start].to_string()));
            }

            // Find matching closing brace (handling nested braces)
            let expr_start = start + 2;
            let mut depth = 1;
            let mut end = expr_start;
            let chars: Vec<char> = remaining[expr_start..].chars().collect();

            for (i, ch) in chars.iter().enumerate() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = expr_start + i;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if depth == 0 {
                // Successfully found matching brace
                let expr_str = &remaining[expr_start..end];
                // Parse the expression
                let (parsed_expr, errors) = crate::parser::parse_expr(expr_str);
                if let Some(expr) = parsed_expr {
                    if errors.is_empty() {
                        parts.push(StringPart::Expr(expr));
                    } else {
                        // Parsing had errors, treat as literal
                        parts.push(StringPart::Lit(format!("${{{}}}", expr_str)));
                    }
                } else {
                    // Couldn't parse, treat as literal
                    parts.push(StringPart::Lit(format!("${{{}}}", expr_str)));
                }
                remaining = &remaining[end + 1..];
            } else {
                // Unmatched brace, treat rest as literal
                parts.push(StringPart::Lit(remaining.to_string()));
                break;
            }
        } else {
            // No more interpolations, add remaining as literal
            parts.push(StringPart::Lit(remaining.to_string()));
            break;
        }
    }

    if parts.is_empty() {
        StringLit::Plain(String::new())
    } else if parts.len() == 1 {
        if let StringPart::Lit(s) = &parts[0] {
            StringLit::Plain(s.clone())
        } else {
            StringLit::Interpolated(parts)
        }
    } else {
        StringLit::Interpolated(parts)
    }
}

/// Parser for type expressions.
fn type_expr() -> impl Parser<Token, TypeExpr, Error = Simple<Token>> + Clone {
    recursive(|ty| {
        // Unit type: ()
        let unit = just(Token::LParen)
            .then(just(Token::RParen))
            .map(|_| TypeExpr::Unit);

        let simple = type_name().map(TypeExpr::Name);

        let generic = type_name()
            .then(
                ty.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            )
            .map(|(name, args)| TypeExpr::Generic(name, args));

        let record = ident()
            .then_ignore(just(Token::Colon))
            .then(ty.clone())
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(TypeExpr::Record);

        // Tuple type: (Int, String) - at least 2 elements
        let tuple = ty
            .clone()
            .separated_by(just(Token::Comma))
            .at_least(2)
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(TypeExpr::Tuple);

        let atom = choice((unit, generic, simple, record, tuple));

        // Function type: T -> U or (T, U) -> V
        let params = ty
            .clone()
            .separated_by(just(Token::Comma))
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .or(atom.clone().map(|t| vec![t]));

        params
            .clone()
            .then(just(Token::RightArrow).ignore_then(ty.clone()).or_not())
            .map(|(params, ret)| {
                if let Some(ret) = ret {
                    TypeExpr::Function(params, Box::new(ret))
                } else if params.len() == 1 {
                    params.into_iter().next().unwrap()
                } else {
                    TypeExpr::Tuple(params)
                }
            })
    })
}

/// Parser for patterns.
fn pattern() -> impl Parser<Token, Pattern, Error = Simple<Token>> + Clone {
    recursive(|pat| {
        let wildcard =
            just(Token::Underscore).map_with_span(|_, span| Pattern::Wildcard(to_span(span)));

        let var = ident().map(Pattern::Var);

        // Unit pattern: ()
        let unit = just(Token::LParen)
            .then(just(Token::RParen))
            .map_with_span(|_, span| Pattern::Unit(to_span(span)));

        // Integer patterns (including hex and binary and typed)
        let int = filter_map(|span, tok| match tok {
            Token::Int(n) => Ok(Pattern::Int(n, to_span(span))),
            Token::HexInt(n) => Ok(Pattern::Int(n, to_span(span))),
            Token::BinInt(n) => Ok(Pattern::Int(n, to_span(span))),
            Token::Int8(n) => Ok(Pattern::Int8(n, to_span(span))),
            Token::Int16(n) => Ok(Pattern::Int16(n, to_span(span))),
            Token::Int32(n) => Ok(Pattern::Int32(n, to_span(span))),
            Token::UInt8(n) => Ok(Pattern::UInt8(n, to_span(span))),
            Token::UInt16(n) => Ok(Pattern::UInt16(n, to_span(span))),
            Token::UInt32(n) => Ok(Pattern::UInt32(n, to_span(span))),
            Token::UInt64(n) => Ok(Pattern::UInt64(n, to_span(span))),
            Token::BigInt(s) => Ok(Pattern::BigInt(s, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        // Negative integer patterns: -1, -42, etc.
        let neg_int = just(Token::Minus)
            .ignore_then(filter_map(|span, tok| match tok {
                Token::Int(n) => Ok(Pattern::Int(-n, to_span(span))),
                Token::Int8(n) => Ok(Pattern::Int8(-n, to_span(span))),
                Token::Int16(n) => Ok(Pattern::Int16(-n, to_span(span))),
                Token::Int32(n) => Ok(Pattern::Int32(-n, to_span(span))),
                _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
            }))
            .map_with_span(|pat, span| {
                // Adjust the span to include the minus sign
                match pat {
                    Pattern::Int(n, _) => Pattern::Int(n, to_span(span)),
                    Pattern::Int8(n, _) => Pattern::Int8(n, to_span(span)),
                    Pattern::Int16(n, _) => Pattern::Int16(n, to_span(span)),
                    Pattern::Int32(n, _) => Pattern::Int32(n, to_span(span)),
                    other => other,
                }
            });

        // Range patterns: 1..10 (exclusive) or 1..=10 (inclusive)
        // Helper to parse a raw (positive) integer value
        let raw_int = filter_map(|_span, tok| match tok {
            Token::Int(n) => Ok(n),
            Token::HexInt(n) => Ok(n),
            Token::BinInt(n) => Ok(n),
            _ => Err(Simple::expected_input_found(_span, vec![], Some(tok))),
        });

        // Signed integer for ranges: can be negative (-5) or positive (5)
        let signed_int = just(Token::Minus)
            .ignore_then(raw_int.clone())
            .map(|n| -n)
            .or(raw_int.clone());

        // Inclusive range: 1..=10 or -10..=0
        let range_inclusive = signed_int.clone()
            .then_ignore(just(Token::DotDotEq))
            .then(signed_int.clone())
            .map_with_span(|(start, end), span| Pattern::Range(start, end, true, to_span(span)));

        // Exclusive range: 1..10 or -10..0
        let range_exclusive = signed_int.clone()
            .then_ignore(just(Token::DotDot))
            .then(signed_int.clone())
            .map_with_span(|(start, end), span| Pattern::Range(start, end, false, to_span(span)));

        let float = filter_map(|span, tok| match tok {
            Token::Float(s) => Ok(Pattern::Float(s.parse().unwrap_or(0.0), to_span(span))),
            Token::Float32(s) => Ok(Pattern::Float32(s.parse().unwrap_or(0.0), to_span(span))),
            Token::Decimal(s) => Ok(Pattern::Decimal(s, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let string = filter_map(|span, tok| match tok {
            Token::String(s) | Token::SingleQuoteString(s) => Ok(Pattern::String(s, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let char_pat = filter_map(|span, tok| match tok {
            Token::Char(c) => Ok(Pattern::Char(c, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let bool_pat = filter_map(|span, tok| match tok {
            Token::True => Ok(Pattern::Bool(true, to_span(span))),
            Token::False => Ok(Pattern::Bool(false, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        // Pin pattern: ^ident
        let pin = just(Token::Caret)
            .ignore_then(ident())
            .map_with_span(|id, span| Pattern::Pin(Box::new(Expr::Var(id)), to_span(span)));

        // Tuple pattern: (a, b)
        let tuple = pat
            .clone()
            .separated_by(just(Token::Comma))
            .at_least(2)
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map_with_span(|pats, span| Pattern::Tuple(pats, to_span(span)));

        // Record pattern field
        let record_field = choice((
            // Named: name: pattern
            ident()
                .then_ignore(just(Token::Colon))
                .then(pat.clone())
                .map(|(name, pat)| RecordPatternField::Named(name, pat)),
            // Rest: _
            just(Token::Underscore).map_with_span(|_, span| RecordPatternField::Rest(to_span(span))),
            // Punned: just name
            ident().map(RecordPatternField::Punned),
        ));

        let record = record_field
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map_with_span(|fields, span| Pattern::Record(fields, to_span(span)));

        // Variant pattern: Some(x), None, Circle{radius: r}
        let variant_positional = type_name()
            .then(
                pat.clone()
                    .separated_by(just(Token::Comma))
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map_with_span(|(name, args), span| {
                Pattern::Variant(name, VariantPatternFields::Positional(args), to_span(span))
            });

        // Variant with named fields using braces: Circle{radius: r}
        let variant_named_braces = type_name()
            .then(
                record_field.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map_with_span(|(name, fields), span| {
                Pattern::Variant(name, VariantPatternFields::Named(fields), to_span(span))
            });

        // Variant with named fields using parens: Point(x: a, y: b)
        // Must have at least one colon to distinguish from positional
        let named_field_paren = ident()
            .then_ignore(just(Token::Colon))
            .then(pat.clone())
            .map(|(name, pat)| RecordPatternField::Named(name, pat));

        let variant_named_parens = type_name()
            .then(
                named_field_paren
                    .separated_by(just(Token::Comma))
                    .at_least(1)
                    .allow_trailing()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map_with_span(|(name, fields), span| {
                Pattern::Variant(name, VariantPatternFields::Named(fields), to_span(span))
            });

        // Unit variant (just name, no args)
        let variant_unit = type_name()
            .map_with_span(|name, span| {
                Pattern::Variant(name, VariantPatternFields::Unit, to_span(span))
            });

        // Combined named variant: try parens first (Point(x: a)), then braces (Point{x: a})
        let variant_named = variant_named_parens.or(variant_named_braces);

        // Single element in parens is just grouping
        let grouped = pat
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Map pattern: %{key: pat, ...}
        // Keys must be literals or pinned variables
        let map_key = choice((
            filter_map(|span, tok| match tok {
                Token::Int(n) => Ok(Expr::Int(n, to_span(span))),
                Token::String(s) | Token::SingleQuoteString(s) => Ok(Expr::String(StringLit::Plain(s), to_span(span))),
                Token::True => Ok(Expr::Bool(true, to_span(span))),
                Token::False => Ok(Expr::Bool(false, to_span(span))),
                Token::Char(c) => Ok(Expr::Char(c, to_span(span))),
                _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
            }),
            just(Token::Caret).ignore_then(ident()).map(Expr::Var),
        ));

        let map_entry = map_key
            .then_ignore(just(Token::Colon))
            .then(pat.clone());

        let map_pat = just(Token::Percent)
            .ignore_then(
                map_entry
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace))
            )
            .map_with_span(|entries, span| Pattern::Map(entries, to_span(span)));

        // Set pattern: #{pat, ...}
        let set_pat = just(Token::Hash)
            .ignore_then(
                pat.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace))
            )
            .map_with_span(|elems, span| Pattern::Set(elems, to_span(span)));

        // Base patterns (without or and without list) - used for list element patterns
        // Order matters: ranges must come before neg_int/int to match `-10..0` before just `-10`
        // Note: neg_int must come before int to properly parse negative literals
        let literals = choice((wildcard.clone(), bool_pat.clone(), range_inclusive.clone(), range_exclusive.clone(), neg_int.clone(), int.clone(), float.clone(), string.clone(), char_pat.clone())).boxed();
        let base_containers = choice((unit.clone(), tuple.clone(), grouped.clone(), record.clone(), map_pat.clone(), set_pat.clone())).boxed();
        let variants_box = choice((pin.clone(), variant_positional.clone(), variant_named.clone(), variant_unit.clone(), var.clone())).boxed();
        let base_no_list = choice((literals.clone(), base_containers.clone(), variants_box.clone())).boxed();

        // List pattern: [], [h | t], [a, b, c], [[nested | _] | _]
        // String cons pattern: ["prefix" | rest] - when all elements are strings and there's a tail
        // Note: list elements use base_with_list to include nested lists but not Or patterns
        let list_empty = just(Token::LBracket)
            .then(just(Token::RBracket))
            .map_with_span(|_, span| Pattern::List(ListPattern::Empty, to_span(span)));

        // Nested empty list
        let nested_list_empty = just(Token::LBracket)
            .then(just(Token::RBracket))
            .map_with_span(|_, span| Pattern::List(ListPattern::Empty, to_span(span)));

        // Nested cons list - elements use base_no_list to avoid | being parsed as Or
        let nested_list_cons = base_no_list.clone()
            .separated_by(just(Token::Comma))
            .at_least(1)
            .then(just(Token::Pipe).ignore_then(pat.clone()).or_not())
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with_span(|(elems, tail), span| {
                Pattern::List(ListPattern::Cons(elems, tail.map(Box::new)), to_span(span))
            });

        // For list elements, we need to include nested list patterns
        // But we need to be careful: nested list elements should also use base_no_list
        // to avoid ambiguity with | operator
        let list_element = choice((
            base_no_list.clone(),
            nested_list_empty,
            nested_list_cons,
        )).boxed();

        let list_cons = list_element
            .clone()
            .separated_by(just(Token::Comma))
            .at_least(1)
            .then(just(Token::Pipe).ignore_then(pat.clone()).or_not())
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with_span(|(elems, tail), span| {
                // Check if all elements are string literals and there's a tail
                // If so, this is a string cons pattern like ["hello" | rest]
                let all_strings: Option<Vec<String>> = elems.iter().map(|p| {
                    match p {
                        Pattern::String(s, _) => Some(s.clone()),
                        _ => None,
                    }
                }).collect();

                match (all_strings, tail) {
                    (Some(strings), Some(tail_pat)) => {
                        // This is a string cons pattern
                        Pattern::StringCons(
                            StringPattern::Cons(strings, Box::new(tail_pat)),
                            to_span(span)
                        )
                    }
                    (_, tail) => {
                        // Regular list pattern
                        Pattern::List(ListPattern::Cons(elems, tail.map(Box::new)), to_span(span))
                    }
                }
            });

        // Full containers including list patterns
        let containers = choice((unit, list_empty, list_cons, tuple, grouped, record, map_pat, set_pat)).boxed();
        let variants_final = choice((pin, variant_positional, variant_named, variant_unit, var)).boxed();
        let base = choice((literals, containers, variants_final));

        // Or pattern: pattern | pattern | pattern
        // Note: This needs careful precedence handling
        base.clone()
            .then(
                just(Token::Pipe)
                    .ignore_then(base.clone())
                    .repeated()
            )
            .map_with_span(|(first, rest), span| {
                if rest.is_empty() {
                    first
                } else {
                    let mut pats = vec![first];
                    pats.extend(rest);
                    Pattern::Or(pats, to_span(span))
                }
            })
    })
    .boxed()
}

/// Skip zero or more newline tokens (for use within expressions)
fn skip_newlines() -> impl Parser<Token, (), Error = Simple<Token>> + Clone {
    just(Token::Newline).repeated().ignored()
}

/// Parser for expressions.
pub fn expr() -> impl Parser<Token, Expr, Error = Simple<Token>> + Clone {
    recursive(|expr| {
        // Skip newlines at the start of expression parsing
        let expr = skip_newlines().ignore_then(expr.clone());
        // Integer literals (including hex, binary, and typed variants)
        let int = filter_map(|span, tok| match tok {
            Token::Int(n) => Ok(Expr::Int(n, to_span(span))),
            Token::HexInt(n) => Ok(Expr::Int(n, to_span(span))),
            Token::BinInt(n) => Ok(Expr::Int(n, to_span(span))),
            Token::Int8(n) => Ok(Expr::Int8(n, to_span(span))),
            Token::Int16(n) => Ok(Expr::Int16(n, to_span(span))),
            Token::Int32(n) => Ok(Expr::Int32(n, to_span(span))),
            Token::UInt8(n) => Ok(Expr::UInt8(n, to_span(span))),
            Token::UInt16(n) => Ok(Expr::UInt16(n, to_span(span))),
            Token::UInt32(n) => Ok(Expr::UInt32(n, to_span(span))),
            Token::UInt64(n) => Ok(Expr::UInt64(n, to_span(span))),
            Token::BigInt(s) => Ok(Expr::BigInt(s, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let float = filter_map(|span, tok| match tok {
            Token::Float(s) => Ok(Expr::Float(s.parse().unwrap_or(0.0), to_span(span))),
            Token::Float32(s) => Ok(Expr::Float32(s.parse().unwrap_or(0.0), to_span(span))),
            Token::Decimal(s) => Ok(Expr::Decimal(s, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let string = filter_map(|span, tok| match tok {
            Token::String(s) | Token::SingleQuoteString(s) => Ok(Expr::String(parse_string_lit(&s), to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let char_expr = filter_map(|span, tok| match tok {
            Token::Char(c) => Ok(Expr::Char(c, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let bool_expr = filter_map(|span, tok| match tok {
            Token::True => Ok(Expr::Bool(true, to_span(span))),
            Token::False => Ok(Expr::Bool(false, to_span(span))),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        });

        let var = ident().map(Expr::Var);

        // Unit: ()
        let unit = just(Token::LParen)
            .then(just(Token::RParen))
            .map_with_span(|_, span| Expr::Unit(to_span(span)));

        // Wildcard: _ (for use in destructuring patterns like `(a, _) = expr`)
        let wildcard = just(Token::Underscore)
            .map_with_span(|_, span| Expr::Wildcard(to_span(span)));

        // Separator for collection elements: comma optionally surrounded by newlines
        // This allows multi-line lists, tuples, etc.
        let nl = just(Token::Newline).repeated();
        let comma_sep = nl.clone().ignore_then(just(Token::Comma)).then_ignore(nl.clone());

        // Tuple: (a, b, c)
        let tuple = nl.clone()
            .ignore_then(expr.clone())
            .separated_by(comma_sep.clone())
            .at_least(2)
            .then_ignore(nl.clone())
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map_with_span(|exprs, span| Expr::Tuple(exprs, to_span(span)));

        // Grouped expression: (expr)
        let grouped = nl.clone()
            .ignore_then(expr.clone())
            .then_ignore(nl.clone())
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // List: [a, b, c] or [h | t]
        let list = nl.clone()
            .ignore_then(expr.clone())
            .separated_by(comma_sep.clone())
            .then(nl.clone().ignore_then(just(Token::Pipe)).ignore_then(nl.clone()).ignore_then(expr.clone()).or_not())
            .then_ignore(nl.clone())
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with_span(|(elems, tail), span| {
                Expr::List(elems, tail.map(Box::new), to_span(span))
            });

        // Map literal: %{"key": value, "key2": value2}
        let map_entry = nl.clone()
            .ignore_then(expr.clone())
            .then_ignore(just(Token::Colon))
            .then(nl.clone().ignore_then(expr.clone()));
        let map = just(Token::Percent)
            .ignore_then(
                map_entry
                    .separated_by(comma_sep.clone())
                    .allow_trailing()
                    .then_ignore(nl.clone())
                    .delimited_by(just(Token::LBrace), just(Token::RBrace))
            )
            .map_with_span(|entries, span| Expr::Map(entries, to_span(span)));

        // Set literal: #{1, 2, 3}
        let set = just(Token::Hash)
            .ignore_then(
                nl.clone()
                    .ignore_then(expr.clone())
                    .separated_by(comma_sep.clone())
                    .allow_trailing()
                    .then_ignore(nl.clone())
                    .delimited_by(just(Token::LBrace), just(Token::RBrace))
            )
            .map_with_span(|elems, span| Expr::Set(elems, to_span(span)));

        // Record field
        let record_field = nl.clone().ignore_then(choice((
            ident()
                .then_ignore(just(Token::Colon))
                .then(nl.clone().ignore_then(expr.clone()))
                .map(|(name, val)| RecordField::Named(name, val)),
            expr.clone().map(RecordField::Positional),
        )));

        // Record/Variant construction: Point(3.0, 4.0) or Point(x: 3.0, y: 4.0)
        let record = type_name()
            .then(
                record_field
                    .separated_by(comma_sep.clone())
                    .allow_trailing()
                    .then_ignore(nl.clone())
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map_with_span(|(name, fields), span| Expr::Record(name, fields, to_span(span)));

        // Unit variant: None, True, False (uppercase identifiers without parens)
        // Note: must come after `record` in choice to prefer record with args
        let unit_variant = type_name()
            .map_with_span(|name, span| Expr::Record(name, vec![], to_span(span)));

        // Block: { stmt; stmt; expr }
        // A statement is either a let binding or an expression.
        // Let binding: [var] pattern = expr
        // Expression: any expression
        //
        // The challenge is distinguishing `(a, b) = expr` (let binding with tuple pattern)
        // from `(a, b)` (tuple expression).
        //
        // Strategy: Parse as full expression first, then check for `=`.
        // If `=` found, convert the expression to a pattern (if possible).
        // This way we parse `a + b` correctly as an expression, and `(a, b) = x`
        // correctly as a let binding.

        // Mutable let binding: var pattern = expr or var pattern: Type = expr
        let mutable_binding = just(Token::Var)
            .ignore_then(pattern())
            .then(just(Token::Colon).ignore_then(type_expr()).or_not())
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .map(|((pat, ty), value)| {
                Stmt::Let(Binding {
                    visibility: Visibility::Private,
                    mutable: true,
                    pattern: pat,
                    ty,
                    value,
                    span: Span::default(),
                })
            });

        // Typed immutable binding: pattern: Type = expr
        let typed_immutable_binding = pattern()
            .then_ignore(just(Token::Colon))
            .then(type_expr())
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .map(|((pat, ty), value)| {
                Stmt::Let(Binding {
                    visibility: Visibility::Private,
                    mutable: false,
                    pattern: pat,
                    ty: Some(ty),
                    value,
                    span: Span::default(),
                })
            });

        // For statements without `var`, we parse as expression first
        // Then if followed by `=`, try to convert to pattern for let binding
        // or to an assignment target (for arr[i] = x, record.field = x)
        let expr_or_binding = expr.clone()
            .then(just(Token::Eq).ignore_then(expr.clone()).or_not())
            .map_with_span(|(lhs, maybe_rhs), span| {
                match maybe_rhs {
                    Some(rhs) => {
                        // Have `=`, first try to convert lhs to pattern (for let binding)
                        if let Some(pat) = expr_to_pattern(lhs.clone()) {
                            Stmt::Let(Binding {
                                visibility: Visibility::Private,
                                mutable: false,
                                pattern: pat,
                                ty: None,
                                value: rhs,
                                span: Span::default(),
                            })
                        } else {
                            // Can't be a pattern - try to convert to AssignTarget
                            let target = match &lhs {
                                Expr::Index(coll, idx, _) => {
                                    Some(AssignTarget::Index(coll.clone(), idx.clone()))
                                }
                                Expr::FieldAccess(obj, field, _) => {
                                    Some(AssignTarget::Field(obj.clone(), field.clone()))
                                }
                                Expr::Var(name) => {
                                    Some(AssignTarget::Var(name.clone()))
                                }
                                _ => None,
                            };
                            match target {
                                Some(t) => Stmt::Assign(t, rhs, to_span(span)),
                                None => {
                                    // LHS can't be a pattern or assignment target - syntax error
                                    // For now, just treat as expression (will fail later)
                                    Stmt::Expr(lhs)
                                }
                            }
                        }
                    }
                    None => Stmt::Expr(lhs),
                }
            });

        let stmt = mutable_binding.or(typed_immutable_binding).or(expr_or_binding);

        // Parse statements separated by newlines, commas, semicolons, or colons
        // The separator can be any combination of these
        let separator = choice((
            just(Token::Newline),
            just(Token::Comma),
            just(Token::Semicolon),
            just(Token::Colon),
        )).repeated().at_least(1);

        // Skip leading newlines inside block
        let leading_newlines = just(Token::Newline).repeated();

        let block = leading_newlines.clone()
            .ignore_then(
                stmt.clone()
                    .separated_by(separator)
                    .allow_trailing()
            )
            .then_ignore(leading_newlines)
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map_with_span(|stmts, span| {
                if stmts.is_empty() {
                    // Empty block is equivalent to unit
                    Expr::Unit(to_span(span))
                } else {
                    Expr::Block(stmts, to_span(span))
                }
            });

        // Lambda: x => expr or (a, b) => expr or () => expr
        let lambda_params = choice((
            // Empty parens: () => expr (must be first to avoid parsing as unit pattern)
            just(Token::LParen)
                .ignore_then(just(Token::RParen))
                .to(vec![]),
            // Multiple params: (a, b) => expr
            pattern()
                .separated_by(just(Token::Comma))
                .delimited_by(just(Token::LParen), just(Token::RParen)),
            // Single param: x => expr
            pattern().map(|p| vec![p]),
        ));

        let lambda = lambda_params
            .then_ignore(just(Token::FatArrow))
            .then(expr.clone())
            .map_with_span(|(params, body), span| {
                Expr::Lambda(params, Box::new(body), to_span(span))
            });

        // If expression - skip newlines after then/else keywords
        // Note: "then" keyword is optional (both "if x then y else z" and "if x y else z" work)
        let if_expr = just(Token::If)
            .ignore_then(skip_newlines().ignore_then(expr.clone()))
            .then_ignore(skip_newlines())
            .then_ignore(just(Token::Then).or_not())
            .then(skip_newlines().ignore_then(expr.clone()))
            .then(
                skip_newlines()
                    .ignore_then(just(Token::Else))
                    .ignore_then(skip_newlines())
                    .ignore_then(expr.clone())
                    .or_not(), // Make the else branch optional
            )
            .map_with_span(|((cond, then_branch), else_branch_opt), span| {
                let else_branch = else_branch_opt.unwrap_or_else(|| {
                    let then_span = get_span(&then_branch);
                    Expr::Unit(Span::new(then_span.end, then_span.end))
                });
                Expr::If(
                    Box::new(cond),
                    Box::new(then_branch),
                    Box::new(else_branch),
                    to_span(span),
                )
            });

        // Skip newlines helper (for use in control flow constructs)
        let nl = skip_newlines();

        // Match arm: pattern -> expr or pattern when guard -> expr
        // Allows optional trailing comma after each arm for single-line style
        let match_arm = nl.clone()
            .ignore_then(pattern())
            .then(just(Token::When).ignore_then(expr.clone()).or_not())
            .then_ignore(just(Token::RightArrow))
            .then(expr.clone())
            .then_ignore(just(Token::Comma).or_not())  // Allow comma separator
            .map_with_span(|((pat, guard), body), span| MatchArm {
                pattern: pat,
                guard,
                body,
                span: to_span(span),
            });

        // Match expression: match expr { pattern -> body, ... }
        let match_expr = just(Token::Match)
            .ignore_then(expr.clone())
            .then_ignore(nl.clone())
            .then_ignore(just(Token::LBrace))
            .then(match_arm.clone().repeated().at_least(1))
            .then_ignore(nl.clone())
            .then_ignore(just(Token::RBrace))
            .map_with_span(|(scrutinee, arms), span| {
                Expr::Match(Box::new(scrutinee), arms, to_span(span))
            });

        // Try expression: try { stmts } catch { pattern -> expr } finally { expr }
        // The body is parsed as a block expression (handles multiple statements)
        let try_expr = just(Token::Try)
            .ignore_then(nl.clone())
            .ignore_then(expr.clone())  // Parse block expression: { stmt1; stmt2; ... }
            .then_ignore(nl.clone())
            .then_ignore(just(Token::Catch))
            .then_ignore(nl.clone())
            .then_ignore(just(Token::LBrace))
            .then(match_arm.clone().repeated())
            .then_ignore(nl.clone())
            .then_ignore(just(Token::RBrace))
            .then(
                nl.clone()
                    .ignore_then(just(Token::Finally))
                    .ignore_then(nl.clone())
                    .ignore_then(expr.clone())  // Parse block expression
                    .or_not()
            )
            .map_with_span(|((body, catches), finally), span| {
                Expr::Try(Box::new(body), catches, finally.map(Box::new), to_span(span))
            });

        // Do block: do stmt; stmt; ... end
        let do_stmt = choice((
            // Bind: pattern = expr
            nl.clone().ignore_then(pattern())
                .then_ignore(just(Token::Eq))
                .then(expr.clone())
                .map(|(pat, e)| DoStmt::Bind(pat, e)),
            // Expression
            nl.clone().ignore_then(expr.clone()).map(DoStmt::Expr),
        ));

        let do_block = just(Token::Do)
            .ignore_then(do_stmt.repeated())
            .then_ignore(nl.clone())
            .then_ignore(just(Token::End))
            .map_with_span(|stmts, span| Expr::Do(stmts, to_span(span)));

        // Receive expression (concurrency): receive { pattern -> expr, after timeout -> expr }
        let receive_expr = just(Token::Receive)
            .ignore_then(nl.clone())
            .ignore_then(just(Token::LBrace))
            .ignore_then(match_arm.clone().repeated())
            .then(
                nl.clone()
                    .ignore_then(just(Token::After))
                    .ignore_then(nl.clone().ignore_then(expr.clone()))
                    .then_ignore(just(Token::RightArrow))
                    .then(nl.clone().ignore_then(expr.clone()))
                    .or_not(),
            )
            .then_ignore(nl.clone())
            .then_ignore(just(Token::RBrace))
            .map_with_span(|(arms, timeout), span| {
                let timeout = timeout.map(|(t, body)| (Box::new(t), Box::new(body)));
                Expr::Receive(arms, timeout, to_span(span))
            });

        // While loop: while cond { body }
        let while_expr = just(Token::While)
            .ignore_then(skip_newlines().ignore_then(expr.clone()))
            .then(skip_newlines().ignore_then(expr.clone()))
            .map_with_span(|(cond, body), span| {
                Expr::While(Box::new(cond), Box::new(body), to_span(span))
            });

        // For loop: for var = start to end { body }
        let for_expr = just(Token::For)
            .ignore_then(skip_newlines().ignore_then(ident()))
            .then_ignore(skip_newlines())
            .then_ignore(just(Token::Eq))
            .then(skip_newlines().ignore_then(expr.clone()))
            .then_ignore(skip_newlines())
            .then_ignore(just(Token::To))
            .then(skip_newlines().ignore_then(expr.clone()))
            .then(skip_newlines().ignore_then(expr.clone()))
            .map_with_span(|(((var, start), end), body), span| {
                Expr::For(var, Box::new(start), Box::new(end), Box::new(body), to_span(span))
            });

        // Break: break or break expr
        let break_expr = just(Token::Break)
            .ignore_then(expr.clone().or_not())
            .map_with_span(|val, span| {
                Expr::Break(val.map(Box::new), to_span(span))
            });

        // Continue
        let continue_expr = just(Token::Continue)
            .map_with_span(|_, span| Expr::Continue(to_span(span)));

        // Return: return or return expr
        let return_expr = just(Token::Return)
            .ignore_then(expr.clone().or_not())
            .map_with_span(|val, span| {
                Expr::Return(val.map(Box::new), to_span(span))
            });

        // Spawn expressions
        let spawn_kind = choice((
            just(Token::SpawnMonitor).to(SpawnKind::Monitored),
            just(Token::SpawnLink).to(SpawnKind::Linked),
            just(Token::Spawn).to(SpawnKind::Normal),
        ));

        // spawn(expr) or spawn { block } - block is treated as zero-param thunk
        let spawn_expr = spawn_kind
            .then(choice((
                expr.clone().delimited_by(just(Token::LParen), just(Token::RParen)),
                block.clone(),
            )))
            .map_with_span(|(kind, func), span| {
                Expr::Spawn(kind, Box::new(func), vec![], to_span(span))
            });

        // Quote expression - quote(expr) or quote { block }
        let quote_expr = just(Token::Quote)
            .ignore_then(choice((
                expr.clone().delimited_by(just(Token::LParen), just(Token::RParen)),
                block.clone(),
            )))
            .map_with_span(|inner, span| Expr::Quote(Box::new(inner), to_span(span)));

        // Primary expressions - split into groups to reduce type complexity
        // Skip newlines at the start of any primary expression
        let control_flow = skip_newlines().ignore_then(choice((if_expr, match_expr, try_expr, do_block, receive_expr, while_expr, for_expr, break_expr, continue_expr, return_expr))).boxed();
        // Note: splice_expr is added below after primary is defined
        let special_no_splice = skip_newlines().ignore_then(choice((spawn_expr, quote_expr, lambda))).boxed();
        let lit = skip_newlines().ignore_then(choice((bool_expr, int, float, string, char_expr))).boxed();
        let collections = skip_newlines().ignore_then(choice((map, set, record, unit_variant, tuple, unit, list, block))).boxed();
        // Splice expression - ~atom (only valid inside quote)
        // Splice only takes a simple expression (var, literal, or parenthesized)
        // This ensures ~x * 2 parses as (~x) * 2, not ~(x * 2)
        let splice_atom = choice((var.clone(), grouped.clone()));
        let splice_expr = skip_newlines().ignore_then(
            just(Token::Tilde)
                .ignore_then(splice_atom)
                .map_with_span(|inner, span| Expr::Splice(Box::new(inner), to_span(span)))
        ).boxed();

        let simple = skip_newlines().ignore_then(choice((grouped, wildcard, var))).boxed();

        let primary = choice((control_flow, special_no_splice, splice_expr, lit, collections, simple));

        // Postfix: function calls, method calls, field access, try operator
        // For call args, allow newlines around commas
        let call_nl = just(Token::Newline).repeated();
        let call_comma = call_nl.clone().ignore_then(just(Token::Comma)).then_ignore(call_nl.clone());

        // Call argument: either `name: expr` (named) or just `expr` (positional)
        let call_arg = call_nl.clone().ignore_then(choice((
            ident()
                .then_ignore(just(Token::Colon))
                .then(call_nl.clone().ignore_then(expr.clone()))
                .map(|(name, val)| CallArg::Named(name, val)),
            expr.clone().map(CallArg::Positional),
        )));
        let call_args = call_arg
            .separated_by(call_comma.clone())
            .then_ignore(call_nl.clone())
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Record field args for qualified record construction: .TypeName(base, field: value)
        let postfix_record_field = call_nl.clone().ignore_then(choice((
            ident()
                .then_ignore(just(Token::Colon))
                .then(call_nl.clone().ignore_then(expr.clone()))
                .map(|(name, val)| RecordField::Named(name, val)),
            expr.clone().map(RecordField::Positional),
        )));
        let record_field_args = postfix_record_field
            .separated_by(call_comma.clone())
            .allow_trailing()
            .then_ignore(call_nl.clone())
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Type arguments for type-applied calls: f[Type](args)
        let type_args = type_expr()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBracket), just(Token::RBracket));

        let postfix = primary.then(
            choice((
                // Type-applied function call: f[Type](x, y)
                type_args.clone()
                    .then(call_args.clone())
                    .map_with_span(|(type_args, args), span| (PostfixOp::Call(type_args, args), to_span(span))),
                // Regular function call: f(x, y)
                call_args
                    .clone()
                    .map_with_span(|args, span| (PostfixOp::Call(vec![], args), to_span(span))),
                // Qualified record construction: .TypeName(base, field: value)
                // Must come before regular method call to handle record field syntax
                skip_newlines()
                    .ignore_then(just(Token::Dot))
                    .ignore_then(type_name())
                    .then(record_field_args.clone())
                    .map_with_span(|(name, fields), span| (PostfixOp::QualifiedRecord(name, fields), to_span(span))),
                // Method call or field access: .foo or .foo(args) or tuple index .0 .1
                // Allow newlines before . for multi-line method chaining
                skip_newlines()
                    .ignore_then(just(Token::Dot))
                    .ignore_then(
                        // Accept either an identifier (upper or lower) or an integer (for tuple indexing)
                        // We use any_ident() to allow module paths like Outer.Inner.value()
                        any_ident().or(
                            select! { Token::Int(n) => n }
                                .map_with_span(|n, span| Ident { node: n.to_string(), span: to_span(span) })
                        )
                    )
                    .then(call_args.clone().or_not())
                    .map_with_span(|(name, args), span| match args {
                        Some(args) => (PostfixOp::MethodCall(name, args), to_span(span)),
                        None => (PostfixOp::FieldAccess(name), to_span(span)),
                    }),
                // Index: arr[i]
                expr.clone()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket))
                    .map_with_span(|idx, span| (PostfixOp::Index(idx), to_span(span))),
                // Try operator: expr?
                just(Token::Question)
                    .map_with_span(|_, span| (PostfixOp::Try, to_span(span))),
            ))
            .repeated(),
        );

        let postfix = postfix.foldl(|lhs, (op, span)| match op {
            PostfixOp::Call(type_args, args) => {
                let full_span = get_span(&lhs).merge(span);
                Expr::Call(Box::new(lhs), type_args, args, full_span)
            }
            PostfixOp::MethodCall(name, args) => {
                let full_span = get_span(&lhs).merge(span);
                Expr::MethodCall(Box::new(lhs), name, args, full_span)
            }
            PostfixOp::FieldAccess(name) => {
                let full_span = get_span(&lhs).merge(span);
                Expr::FieldAccess(Box::new(lhs), name, full_span)
            }
            PostfixOp::Index(idx) => {
                let full_span = get_span(&lhs).merge(span);
                Expr::Index(Box::new(lhs), Box::new(idx), full_span)
            }
            PostfixOp::Try => {
                let full_span = get_span(&lhs).merge(span);
                Expr::Try_(Box::new(lhs), full_span)
            }
            PostfixOp::QualifiedRecord(type_name, fields) => {
                // lhs is the module path (e.g., stdlib.rhtml), type_name is the type (e.g., RHtmlResult)
                // We need to extract the path and combine with type_name to form a qualified type name
                let full_span = get_span(&lhs).merge(span);
                let qualified_name = extract_qualified_path(&lhs, &type_name.node);
                let qualified_ident = Ident { node: qualified_name, span: full_span };
                Expr::Record(qualified_ident, fields, full_span)
            }
        });

        // Unary: -x, !x
        let unary = choice((
            just(Token::Minus).to(UnaryOp::Neg),
            just(Token::Bang).to(UnaryOp::Not),
        ))
        .repeated()
        .then(postfix)
        .foldr(|op, rhs| {
            let span = get_span(&rhs);
            Expr::UnaryOp(op, Box::new(rhs), span)
        })
        .boxed(); // Box to reduce type complexity

        // Binary operators with precedence
        // Skip newlines after operators to allow multi-line expressions
        let nl = skip_newlines();

        // ** (right associative)
        let power = unary.clone().then(
            just(Token::StarStar)
                .to(BinOp::Pow)
                .then(nl.clone().ignore_then(unary.clone()))
                .repeated(),
        );
        let power = power.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        });

        // * / %
        let factor = power.clone().then(
            choice((
                just(Token::Star).to(BinOp::Mul),
                just(Token::Slash).to(BinOp::Div),
                just(Token::Percent).to(BinOp::Mod),
            ))
            .then(nl.clone().ignore_then(power))
            .repeated(),
        );
        let factor = factor.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        });

        // + -
        let term = factor.clone().then(
            choice((
                just(Token::Plus).to(BinOp::Add),
                just(Token::Minus).to(BinOp::Sub),
            ))
            .then(nl.clone().ignore_then(factor))
            .repeated(),
        );
        let term = term.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        })
        .boxed(); // Box to reduce type complexity

        // :: (cons - higher precedence than ++, RIGHT-ASSOCIATIVE)
        // 1 :: 2 :: [3] should parse as 1 :: (2 :: [3]) not (1 :: 2) :: [3]
        let cons = term
            .clone()
            .then(just(Token::ColonColon).ignore_then(nl.clone()).ignore_then(term.clone()).repeated())
            .map(|(first, rest)| {
                if rest.is_empty() {
                    first
                } else {
                    // Fold from right: [1, 2, 3] with base [] becomes 1 :: (2 :: (3 :: []))
                    let mut items: Vec<_> = std::iter::once(first).chain(rest).collect();
                    // Start from the rightmost item and work backwards
                    let mut result = items.pop().unwrap();
                    while let Some(item) = items.pop() {
                        let span = get_span(&item).merge(get_span(&result));
                        result = Expr::BinOp(Box::new(item), BinOp::Cons, Box::new(result), span);
                    }
                    result
                }
            });

        // ++
        let concat = cons
            .clone()
            .then(just(Token::PlusPlus).to(BinOp::Concat).then(nl.clone().ignore_then(cons)).repeated());
        let concat = concat.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        });

        // == != < > <= >=
        let comparison = concat.clone().then(
            choice((
                just(Token::EqEq).to(BinOp::Eq),
                just(Token::NotEq).to(BinOp::NotEq),
                just(Token::LtEq).to(BinOp::LtEq),
                just(Token::GtEq).to(BinOp::GtEq),
                just(Token::Lt).to(BinOp::Lt),
                just(Token::Gt).to(BinOp::Gt),
            ))
            .then(nl.clone().ignore_then(concat))
            .repeated(),
        );
        let comparison = comparison.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        })
        .boxed(); // Box to reduce type complexity

        // &&
        let and = comparison
            .clone()
            .then(just(Token::AndAnd).to(BinOp::And).then(nl.clone().ignore_then(comparison)).repeated());
        let and = and.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        });

        // ||
        let or = and
            .clone()
            .then(just(Token::OrOr).to(BinOp::Or).then(nl.clone().ignore_then(and)).repeated());
        let or = or.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        });

        // |> (pipe operator)
        let pipe = or
            .clone()
            .then(just(Token::PipeRight).to(BinOp::Pipe).then(nl.clone().ignore_then(or)).repeated());
        let pipe = pipe.foldl(|lhs, (op, rhs)| {
            let span = get_span(&lhs).merge(get_span(&rhs));
            Expr::BinOp(Box::new(lhs), op, Box::new(rhs), span)
        });

        // Send: pid <- msg
        let send = pipe
            .clone()
            .then(just(Token::LeftArrow).ignore_then(nl.clone().ignore_then(pipe.clone())).or_not())
            .map(|(lhs, rhs)| {
                if let Some(rhs) = rhs {
                    let span = get_span(&lhs).merge(get_span(&rhs));
                    Expr::Send(Box::new(lhs), Box::new(rhs), span)
                } else {
                    lhs
                }
            });

        send
    })
    .boxed()
}

/// Helper enum for postfix operations during parsing.
#[derive(Clone)]
enum PostfixOp {
    Call(Vec<TypeExpr>, Vec<CallArg>),  // (type_args, value_args) - supports named args
    MethodCall(Ident, Vec<CallArg>),    // supports named args
    FieldAccess(Ident),
    Index(Expr),
    Try,
    /// Qualified record construction: .TypeName(base, field: value)
    /// The Ident is the type name (uppercase), Vec<RecordField> are the fields
    QualifiedRecord(Ident, Vec<RecordField>),
}

/// Extract a qualified path from an expression chain and append a type name.
/// E.g., for `stdlib.rhtml` (as FieldAccess chain) and type_name "RHtmlResult",
/// returns "stdlib.rhtml.RHtmlResult"
fn extract_qualified_path(expr: &Expr, type_name: &str) -> String {
    fn extract_path(expr: &Expr) -> String {
        match expr {
            Expr::Var(ident) => ident.node.clone(),
            Expr::FieldAccess(base, field, _) => {
                let base_path = extract_path(base);
                format!("{}.{}", base_path, field.node)
            }
            // Empty record expression (uppercase identifier parsed as record constructor)
            Expr::Record(name, fields, _) if fields.is_empty() => name.node.clone(),
            _ => String::new(),
        }
    }
    let path = extract_path(expr);
    if path.is_empty() {
        type_name.to_string()
    } else {
        format!("{}.{}", path, type_name)
    }
}

/// Get span from an expression.
fn get_span(expr: &Expr) -> Span {
    match expr {
        Expr::Int(_, s) => *s,
        Expr::Int8(_, s) => *s,
        Expr::Int16(_, s) => *s,
        Expr::Int32(_, s) => *s,
        Expr::UInt8(_, s) => *s,
        Expr::UInt16(_, s) => *s,
        Expr::UInt32(_, s) => *s,
        Expr::UInt64(_, s) => *s,
        Expr::BigInt(_, s) => *s,
        Expr::Float(_, s) => *s,
        Expr::Float32(_, s) => *s,
        Expr::Decimal(_, s) => *s,
        Expr::String(_, s) => *s,
        Expr::Char(_, s) => *s,
        Expr::Bool(_, s) => *s,
        Expr::Unit(s) => *s,
        Expr::Wildcard(s) => *s,
        Expr::Var(id) => id.span,
        Expr::BinOp(_, _, _, s) => *s,
        Expr::UnaryOp(_, _, s) => *s,
        Expr::Call(_, _, _, s) => *s,
        Expr::MethodCall(_, _, _, s) => *s,
        Expr::FieldAccess(_, _, s) => *s,
        Expr::Index(_, _, s) => *s,
        Expr::Lambda(_, _, s) => *s,
        Expr::If(_, _, _, s) => *s,
        Expr::Match(_, _, s) => *s,
        Expr::Tuple(_, s) => *s,
        Expr::List(_, _, s) => *s,
        Expr::Map(_, s) => *s,
        Expr::Set(_, s) => *s,
        Expr::Record(_, _, s) => *s,
        Expr::RecordUpdate(_, _, _, s) => *s,
        Expr::Block(_, s) => *s,
        Expr::Try(_, _, _, s) => *s,
        Expr::Do(_, s) => *s,
        Expr::Receive(_, _, s) => *s,
        Expr::Spawn(_, _, _, s) => *s,
        Expr::Send(_, _, s) => *s,
        Expr::Try_(_, s) => *s,
        Expr::Quote(_, s) => *s,
        Expr::Splice(_, s) => *s,
        Expr::While(_, _, s) => *s,
        Expr::For(_, _, _, _, s) => *s,
        Expr::Break(_, s) => *s,
        Expr::Continue(s) => *s,
        Expr::Return(_, s) => *s,
    }
}

/// Parser for type parameters: `[T]` or `[K: Eq + Hash, V]`
fn type_params() -> impl Parser<Token, Vec<TypeParam>, Error = Simple<Token>> + Clone {
    let constraint = type_name();
    let constraints = constraint
        .separated_by(just(Token::Plus))
        .at_least(1);

    let param = any_ident()
        .then(just(Token::Colon).ignore_then(constraints).or_not())
        .map(|(name, constraints)| TypeParam {
            name,
            constraints: constraints.unwrap_or_default(),
        });

    param
        .separated_by(just(Token::Comma))
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
        .or_not()
        .map(|params| params.unwrap_or_default())
}

/// Parser for function parameter (pattern with optional type and default value).
/// Syntax: `pattern`, `pattern: Type`, `pattern = default`, `pattern: Type = default`
fn fn_param() -> impl Parser<Token, FnParam, Error = Simple<Token>> + Clone {
    pattern()
        .then(just(Token::Colon).ignore_then(type_expr()).or_not())
        .then(just(Token::Eq).ignore_then(expr()).or_not())
        .map(|((pattern, ty), default)| FnParam { pattern, ty, default })
}

/// Parser for a function definition.
fn fn_def() -> impl Parser<Token, FnDef, Error = Simple<Token>> + Clone {
    visibility()
        .then(ident())
        .then(type_params())  // Optional type parameters: [T: Hash, U]
        .then(
            fn_param()
                .separated_by(just(Token::Comma))
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(just(Token::When).ignore_then(expr()).or_not())
        .then(just(Token::RightArrow).ignore_then(type_expr()).or_not())
        .then_ignore(skip_newlines())
        .then_ignore(just(Token::Eq))
        .then_ignore(skip_newlines())
        .then(expr())
        .map_with_span(|((((((vis, name), type_params), params), guard), return_type), body), span| {
            let clause = FnClause {
                params,
                guard,
                return_type,
                body,
                span: to_span(span.clone()),
            };
            FnDef {
                visibility: vis,
                doc: None,
                span: to_span(span),
                name,
                type_params,
                clauses: vec![clause],
                is_template: false,
            }
        })
}

/// Parser for a template function definition.
fn template_def() -> impl Parser<Token, FnDef, Error = Simple<Token>> + Clone {
    just(Token::Template)
        .ignore_then(ident())
        .then(type_params())
        .then(
            fn_param()
                .separated_by(just(Token::Comma))
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(just(Token::When).ignore_then(expr()).or_not())
        .then(just(Token::RightArrow).ignore_then(type_expr()).or_not())
        .then_ignore(skip_newlines())
        .then_ignore(just(Token::Eq))
        .then_ignore(skip_newlines())
        .then(expr())
        .map_with_span(|(((((name, type_params), params), guard), return_type), body), span| {
            let clause = FnClause {
                params,
                guard,
                return_type,
                body,
                span: to_span(span.clone()),
            };
            FnDef {
                visibility: Visibility::Private, // Templates are always private
                doc: None,
                span: to_span(span),
                name,
                type_params,
                clauses: vec![clause],
                is_template: true,
            }
        })
}

/// Parser for variant fields.
fn variant_fields() -> impl Parser<Token, VariantFields, Error = Simple<Token>> + Clone {
    let positional = type_expr()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .map(VariantFields::Positional);

    let field = just(Token::Private)
        .or_not()
        .then(ident())
        .then_ignore(just(Token::Colon))
        .then(type_expr())
        .map(|((private, name), ty)| Field {
            private: private.is_some(),
            name,
            ty,
        });

    let named = field
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(VariantFields::Named);

    choice((positional, named))
        .or_not()
        .map(|fields| fields.unwrap_or(VariantFields::Unit))
}

/// Parser for type body (record, variant, alias, or empty).
fn type_body() -> impl Parser<Token, TypeBody, Error = Simple<Token>> + Clone {
    let nl = just(Token::Newline).repeated();

    let field = just(Token::Private)
        .or_not()
        .then(ident())
        .then_ignore(just(Token::Colon))
        .then(type_expr())
        .map(|((private, name), ty)| Field {
            private: private.is_some(),
            name,
            ty,
        });

    let record = field
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(TypeBody::Record);

    let variant_item = type_name()
        .then(variant_fields())
        .map(|(name, fields)| Variant { name, fields });

    // Variants: | Variant1 | Variant2(T) | Variant3{field: Type}
    // Can start with | or not. Allow newlines before/between variants.
    let variant_sep = nl.clone().then(just(Token::Pipe)).then(nl.clone());
    let variant = nl.clone()
        .ignore_then(just(Token::Pipe).or_not())
        .then_ignore(nl.clone())
        .ignore_then(variant_item)
        .separated_by(variant_sep)
        .at_least(1)
        .map(TypeBody::Variant);

    // Type alias: just a type expression
    let alias = type_expr().map(TypeBody::Alias);

    choice((record, variant, alias))
}

/// Parser for type definition.
/// Supports: `type Name = ...`, `var type Name = ...`, `reactive Name = ...`
fn type_def() -> impl Parser<Token, TypeDef, Error = Simple<Token>> + Clone {
    // Regular type: [var] type Name = ...
    let regular_type = {
        let mutable = just(Token::Var).or_not().map(|v| v.is_some());

        visibility()
            .then(mutable)
            .then_ignore(just(Token::Type))
            .then(type_name())
            .then(type_params())
            .then(just(Token::Eq).ignore_then(type_body()).or_not())
            .map_with_span(|((((vis, mutable), name), type_params), body), span| TypeDef {
                visibility: vis,
                doc: None,
                mutable,
                reactive: false,
                name,
                type_params,
                body: body.unwrap_or(TypeBody::Empty),
                span: to_span(span),
            })
    };

    // Reactive type: reactive Name = { ... }
    let reactive_type = visibility()
        .then_ignore(just(Token::Reactive))
        .then(type_name())
        .then(type_params())
        .then(just(Token::Eq).ignore_then(type_body()).or_not())
        .map_with_span(|(((vis, name), type_params), body), span| TypeDef {
            visibility: vis,
            doc: None,
            mutable: false,
            reactive: true,
            name,
            type_params,
            body: body.unwrap_or(TypeBody::Empty),
            span: to_span(span),
        });

    choice((reactive_type, regular_type))
}

/// Parser for a binding (top-level or local).
/// Supports: `x = 5`, `x: Int = 5`, `var x = 5`, `var x: Int = 5`
/// Also supports: `pub x = 5`, `pub var x = 5` for module-level bindings
fn binding() -> impl Parser<Token, Binding, Error = Simple<Token>> + Clone {
    let mutable = just(Token::Var).or_not().map(|v| v.is_some());

    // Parse pattern with optional type annotation (TypeScript/Rust-style: name: Type = value)
    // Syntax: `visibility? var? pattern (: Type)? = expr`
    visibility()
        .then(mutable)
        .then(pattern())
        .then(just(Token::Colon).ignore_then(type_expr()).or_not())
        .then_ignore(just(Token::Eq))
        .then(expr())
        .map_with_span(|((((visibility, mutable), pattern), ty), value), span| Binding {
            visibility,
            mutable,
            pattern,
            ty,
            value,
            span: to_span(span),
        })
}

/// Parser for module-level mutable variable definition (mvar).
/// Syntax: `mvar name: Type = expr` or `pub mvar name: Type = expr`
/// Type annotation is required for thread-safe shared state.
fn mvar_def() -> impl Parser<Token, MvarDef, Error = Simple<Token>> + Clone {
    let visibility = just(Token::Pub).or_not().map(|v| {
        if v.is_some() { Visibility::Public } else { Visibility::Private }
    });

    visibility
        .then_ignore(just(Token::Mvar))
        .then(ident())
        .then_ignore(just(Token::Colon))
        .then(type_expr())
        .then_ignore(just(Token::Eq))
        .then(expr())
        .map_with_span(|(((vis, name), ty), value), span| MvarDef {
            visibility: vis,
            name,
            ty,
            value,
            span: to_span(span),
        })
}

/// Parser for constant definition.
/// Syntax: `const NAME = literal` or `pub const NAME = literal`
/// Constants can use any identifier (lowercase or UPPERCASE).
fn const_def() -> impl Parser<Token, ConstDef, Error = Simple<Token>> + Clone {
    let visibility = just(Token::Pub).or_not().map(|v| {
        if v.is_some() { Visibility::Public } else { Visibility::Private }
    });

    visibility
        .then_ignore(just(Token::Const))
        .then(any_ident())  // Allow both lowercase and UPPERCASE names
        .then_ignore(just(Token::Eq))
        .then(expr())  // We accept any expr but will validate it's a literal at compile time
        .map_with_span(|((vis, name), value), span| ConstDef {
            visibility: vis,
            name,
            value,
            span: to_span(span),
        })
}

/// Parser for method/operator names (identifiers or operators like ==, !=, <, etc.)
fn method_name() -> impl Parser<Token, Ident, Error = Simple<Token>> + Clone {
    choice((
        ident(),
        filter_map(|span, tok| {
            let name = match &tok {
                Token::EqEq => "==",
                Token::NotEq => "!=",
                Token::Lt => "<",
                Token::Gt => ">",
                Token::LtEq => "<=",
                Token::GtEq => ">=",
                Token::Plus => "+",
                Token::Minus => "-",
                Token::Star => "*",
                Token::Slash => "/",
                Token::Percent => "%",
                Token::StarStar => "**",
                Token::PlusPlus => "++",
                Token::PipeRight => "|>",
                _ => return Err(Simple::expected_input_found(span, vec![], Some(tok))),
            };
            Ok(make_ident(name.to_string(), to_span(span)))
        }),
    ))
}

/// Parser for trait definition.
/// Syntax: `trait Name ... end` or `pub trait Name ... end`
fn trait_def() -> impl Parser<Token, TraitDef, Error = Simple<Token>> + Clone {
    let nl = just(Token::Newline).repeated();

    let visibility = just(Token::Pub).or_not().map(|v| {
        if v.is_some() { Visibility::Public } else { Visibility::Private }
    });

    let super_traits = just(Token::Colon)
        .ignore_then(type_name().separated_by(just(Token::Comma)))
        .or_not()
        .map(|t| t.unwrap_or_default());

    let method_param = pattern()
        .then(just(Token::Colon).ignore_then(type_expr()).or_not())
        .then(just(Token::Eq).ignore_then(expr()).or_not())
        .map(|((pattern, ty), default)| FnParam { pattern, ty, default });

    let method = nl.clone()
        .ignore_then(method_name())
        .then(
            method_param
                .separated_by(just(Token::Comma))
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(just(Token::RightArrow).ignore_then(type_expr()).or_not())
        .then(just(Token::Eq).ignore_then(expr()).or_not())
        .map_with_span(|(((name, params), return_type), default_impl), span| TraitMethod {
            name,
            params,
            return_type,
            default_impl,
            span: to_span(span),
        });

    visibility
        .then_ignore(just(Token::Trait))
        .then(type_name())
        .then(super_traits)
        .then(method.repeated())
        .then_ignore(nl.clone())
        .then_ignore(just(Token::End).or_not())
        .map_with_span(|(((vis, name), super_traits), methods), span| TraitDef {
            visibility: vis,
            doc: None,
            name,
            super_traits,
            methods,
            span: to_span(span),
        })
}

/// Parser for trait implementation.
fn trait_impl() -> impl Parser<Token, TraitImpl, Error = Simple<Token>> + Clone {
    let nl = just(Token::Newline).repeated();

    let constraint = ident()
        .then_ignore(just(Token::Colon))
        .then(type_name().separated_by(just(Token::Plus)).at_least(1));

    let when_clause = just(Token::When)
        .ignore_then(constraint.separated_by(just(Token::Comma)))
        .or_not()
        .map(|w| w.unwrap_or_default());

    // Allow newlines between function definitions in trait impls
    let fn_with_nl = nl.clone().ignore_then(fn_def());

    type_expr()
        .then_ignore(just(Token::Colon))
        .then(type_name())
        .then(when_clause)
        .then(fn_with_nl.repeated())
        .then_ignore(nl.clone())
        .then_ignore(just(Token::End).or_not())
        .map_with_span(|(((ty, trait_name), when_clause), methods), span| TraitImpl {
            ty,
            trait_name,
            when_clause,
            methods,
            span: to_span(span),
        })
}

/// Parser for use statement.
/// Supports three syntaxes:
/// 1. `use module.path.*` - import all public functions
/// 2. `use module.path.{name1, name2}` - import multiple names
/// 3. `use module.path.name` or `use module.path.name as alias` - import single name
fn use_stmt() -> impl Parser<Token, UseStmt, Error = Simple<Token>> + Clone {
    let use_item = any_ident()
        .then(
            just(Token::LowerIdent("as".to_string()))
                .ignore_then(any_ident())
                .or_not(),
        )
        .map(|(name, alias)| UseItem { name, alias });

    let braced_imports = use_item
        .separated_by(just(Token::Comma))
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(UseImports::Named);

    // Option 1: use module.path.* or use module.path.{a, b}
    let with_braces_or_star = just(Token::Use)
        .ignore_then(any_ident().separated_by(just(Token::Dot)).at_least(1))
        .then_ignore(just(Token::Dot))
        .then(choice((
            just(Token::Star).to(UseImports::All),
            braced_imports,
        )))
        .map_with_span(|(path, imports), span| UseStmt {
            path,
            imports,
            span: to_span(span),
        });

    // Option 2: use module.path.name or use module.path.name as alias
    let single_import = just(Token::Use)
        .ignore_then(any_ident().separated_by(just(Token::Dot)).at_least(2))
        .then(
            just(Token::LowerIdent("as".to_string()))
                .ignore_then(any_ident())
                .or_not()
        )
        .map_with_span(|(mut path, alias), span| {
            let name = path.pop().unwrap();
            UseStmt {
                path,
                imports: UseImports::Named(vec![UseItem { name, alias }]),
                span: to_span(span),
            }
        });

    with_braces_or_star.or(single_import)
}

/// Parser for test definition.
fn test_def() -> impl Parser<Token, TestDef, Error = Simple<Token>> + Clone {
    just(Token::Test)
        .ignore_then(filter_map(|span, tok| match tok {
            Token::String(s) | Token::SingleQuoteString(s) => Ok(s),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        }))
        .then_ignore(just(Token::Eq))
        .then(expr())
        .map_with_span(|(name, body), span| TestDef {
            name,
            body,
            span: to_span(span),
        })
}

/// Parser for extern declaration.
fn extern_decl() -> impl Parser<Token, ExternDecl, Error = Simple<Token>> + Clone {
    let extern_type = just(Token::Extern)
        .then(just(Token::Type))
        .ignore_then(type_name())
        .map(|name| ExternKind::Type { name });

    let extern_param = ident()
        .then_ignore(just(Token::Colon))
        .then(type_expr());

    let extern_fn = just(Token::Extern)
        .ignore_then(ident())
        .then(
            extern_param
                .separated_by(just(Token::Comma))
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then_ignore(just(Token::RightArrow))
        .then(type_expr())
        .then_ignore(just(Token::From))
        .then(filter_map(|span, tok| match tok {
            Token::String(s) | Token::SingleQuoteString(s) => Ok(s),
            _ => Err(Simple::expected_input_found(span, vec![], Some(tok))),
        }))
        .map(|(((name, params), return_type), from)| ExternKind::Function {
            name,
            params,
            return_type,
            from,
        });

    choice((extern_type, extern_fn)).map_with_span(|kind, span| ExternDecl {
        kind,
        span: to_span(span),
    })
}

/// Parser for top-level items (using recursive for nested modules).
fn item() -> impl Parser<Token, Item, Error = Simple<Token>> + Clone {
    recursive(|item| {
        // Skip newlines between items
        let nl = just(Token::Newline).repeated();

        // Clone the recursive reference for use in module_def
        // Each item in a module can be preceded by newlines
        let item_with_nl = nl.clone().ignore_then(item.clone());

        // Module definition (uses recursive item)
        let module_def = visibility()
            .then_ignore(just(Token::Module))
            .then(type_name())
            .then(item_with_nl.clone().repeated())
            .then_ignore(nl.clone())
            .then_ignore(just(Token::End))
            .map_with_span(|((vis, name), items), span| {
                Item::ModuleDef(ModuleDef {
                    visibility: vis,
                    name,
                    items,
                    span: to_span(span),
                })
            })
            .boxed();

        choice((
            module_def,  // Must be first to handle nested modules
            type_def().map(Item::TypeDef),
            trait_def().map(Item::TraitDef),
            mvar_def().map(Item::MvarDef),  // Must be before trait_impl to parse 'mvar x = ...'
            const_def().map(Item::ConstDef),  // Must be before trait_impl to parse 'const x = ...'
            trait_impl().map(Item::TraitImpl),  // Must be after const_def since uppercase names can match
            test_def().map(Item::Test),
            extern_decl().map(Item::Extern),
            use_stmt().map(Item::Use),
            template_def().map(Item::FnDef),  // Template functions (before fn_def)
            fn_def().map(Item::FnDef),
            binding().map(Item::Binding),
        ))
    })
    .boxed()
}

/// Parser for a complete module/file.
fn module() -> impl Parser<Token, Module, Error = Simple<Token>> + Clone {
    // Skip newlines between and around items
    let newlines = just(Token::Newline).repeated();

    newlines.clone()
        .ignore_then(
            item()
                .then_ignore(newlines.clone())
                .repeated()
        )
        .then_ignore(end())
        .map(|items| Module { name: None, items })
}

/// Parse source code into a module.
pub fn parse(source: &str) -> (Option<Module>, Vec<Simple<Token>>) {
    // Filter out comment tokens - they're kept for syntax highlighting only
    let tokens: Vec<_> = crate::lexer::lex(source)
        .filter(|(tok, _)| !matches!(tok, Token::Comment | Token::MultiLineComment))
        .collect();
    let len = source.len();

    let (result, errors) = module().parse_recovery(chumsky::Stream::from_iter(
        len..len + 1,
        tokens.into_iter(),
    ));

    (result, errors)
}

/// Parse a single expression (useful for REPL).
pub fn parse_expr(source: &str) -> (Option<Expr>, Vec<Simple<Token>>) {
    // Filter out comment tokens - they're kept for syntax highlighting only
    let tokens: Vec<_> = crate::lexer::lex(source)
        .filter(|(tok, _)| !matches!(tok, Token::Comment | Token::MultiLineComment))
        .collect();
    let len = source.len();

    let (result, errors) = expr()
        .then_ignore(end())
        .parse_recovery(chumsky::Stream::from_iter(len..len + 1, tokens.into_iter()));

    (result, errors)
}
