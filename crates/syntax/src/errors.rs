//! Error reporting with source locations and beautiful formatting.
//!
//! Uses ariadne for rich error output with source code snippets.

use ariadne::{Color, ColorGenerator, Label, Report, ReportKind, Source};
use std::fmt;
use std::io::Write;

use crate::ast::Span;

/// The kind of error that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Syntax/parsing error
    Parse,
    /// Compilation error (semantic)
    Compile,
    /// Runtime error
    Runtime,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Parse => write!(f, "parse error"),
            ErrorKind::Compile => write!(f, "compile error"),
            ErrorKind::Runtime => write!(f, "runtime error"),
        }
    }
}

/// A source-located error with optional hints and notes.
#[derive(Debug, Clone)]
pub struct SourceError {
    /// The main error message
    pub message: String,
    /// The source location of the error
    pub span: Span,
    /// The kind of error
    pub kind: ErrorKind,
    /// An optional hint for fixing the error
    pub hint: Option<String>,
    /// Additional notes providing context
    pub notes: Vec<String>,
    /// Secondary labels pointing to related code
    pub labels: Vec<(Span, String)>,
}

impl SourceError {
    /// Create a new error at the given span.
    pub fn new(kind: ErrorKind, message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            kind,
            hint: None,
            notes: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Create a parse error.
    pub fn parse(message: impl Into<String>, span: Span) -> Self {
        Self::new(ErrorKind::Parse, message, span)
    }

    /// Create a compile error.
    pub fn compile(message: impl Into<String>, span: Span) -> Self {
        Self::new(ErrorKind::Compile, message, span)
    }

    /// Create a runtime error.
    pub fn runtime(message: impl Into<String>, span: Span) -> Self {
        Self::new(ErrorKind::Runtime, message, span)
    }

    /// Add a hint for how to fix the error.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Add a note providing additional context.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add a secondary label pointing to related code.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push((span, message.into()));
        self
    }

    /// Format the error using ariadne for beautiful output.
    pub fn format(&self, filename: &str, source: &str) -> String {
        let mut output = Vec::new();
        self.write_formatted(&mut output, filename, source)
            .expect("writing to Vec should not fail");
        String::from_utf8(output).expect("ariadne output should be valid UTF-8")
    }

    /// Write the formatted error to a writer.
    pub fn write_formatted<W: Write>(
        &self,
        writer: &mut W,
        filename: &str,
        source: &str,
    ) -> std::io::Result<()> {
        let report_kind = match self.kind {
            ErrorKind::Parse => ReportKind::Error,
            ErrorKind::Compile => ReportKind::Error,
            ErrorKind::Runtime => ReportKind::Error,
        };

        let primary_color = match self.kind {
            ErrorKind::Parse => Color::Red,
            ErrorKind::Compile => Color::Magenta,
            ErrorKind::Runtime => Color::Yellow,
        };

        let mut colors = ColorGenerator::new();

        let mut builder = Report::build(report_kind, filename, self.span.start)
            .with_message(&self.message)
            .with_label(
                Label::new((filename, self.span.start..self.span.end))
                    .with_message(&self.message)
                    .with_color(primary_color),
            );

        // Add secondary labels
        for (span, msg) in &self.labels {
            let color = colors.next();
            builder = builder.with_label(
                Label::new((filename, span.start..span.end))
                    .with_message(msg)
                    .with_color(color),
            );
        }

        // Add hint if present
        if let Some(hint) = &self.hint {
            builder = builder.with_help(hint);
        }

        // Add notes
        for note in &self.notes {
            builder = builder.with_note(note);
        }

        let report = builder.finish();
        report.write((filename, Source::from(source)), writer)
    }

    /// Print the error to stderr.
    pub fn eprint(&self, filename: &str, source: &str) {
        self.write_formatted(&mut std::io::stderr(), filename, source)
            .expect("writing to stderr should not fail");
    }
}

/// Helper to convert byte offset to line and column.
pub fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Format multiple errors.
pub fn format_errors(errors: &[SourceError], filename: &str, source: &str) -> String {
    let mut output = String::new();
    for error in errors {
        output.push_str(&error.format(filename, source));
        output.push('\n');
    }
    output
}

/// Print multiple errors to stderr.
pub fn eprint_errors(errors: &[SourceError], filename: &str, source: &str) {
    for error in errors {
        error.eprint(filename, source);
    }
}

// Common error constructors for compile-time errors

impl SourceError {
    /// Unknown variable error.
    pub fn unknown_variable(name: &str, span: Span) -> Self {
        Self::compile(format!("unknown variable `{}`", name), span)
            .with_hint(format!("did you mean to define `{}` first?", name))
    }

    /// Unknown function error.
    pub fn unknown_function(name: &str, span: Span) -> Self {
        Self::compile(format!("unknown function `{}`", name), span)
            .with_hint("make sure the function is defined before it's called")
    }

    /// Unknown type error.
    pub fn unknown_type(name: &str, span: Span) -> Self {
        Self::compile(format!("unknown type `{}`", name), span)
            .with_hint("types must be defined with 'type Name = ...'")
    }

    /// Duplicate definition error.
    pub fn duplicate_definition(name: &str, span: Span, original_span: Option<Span>) -> Self {
        let mut err = Self::compile(format!("`{}` is defined multiple times", name), span);
        if let Some(orig) = original_span {
            err = err.with_label(orig, "first defined here");
        }
        err
    }

    /// Arity mismatch error.
    pub fn arity_mismatch(name: &str, expected: usize, found: usize, span: Span) -> Self {
        Self::compile(
            format!(
                "function `{}` expects {} argument{}, but {} {} provided",
                name,
                expected,
                if expected == 1 { "" } else { "s" },
                found,
                if found == 1 { "was" } else { "were" }
            ),
            span,
        )
    }

    /// Type mismatch error.
    pub fn type_mismatch(expected: &str, found: &str, span: Span) -> Self {
        Self::compile(
            format!("type mismatch: expected `{}`, found `{}`", expected, found),
            span,
        )
        .with_hint(format!("try converting the value to `{}`", expected))
    }

    /// Invalid pattern error.
    pub fn invalid_pattern(span: Span, context: &str) -> Self {
        Self::compile(format!("invalid pattern in {}", context), span)
    }

    /// Private access error.
    pub fn private_access(name: &str, module: &str, span: Span) -> Self {
        Self::compile(
            format!(
                "cannot access private function `{}` from outside module `{}`",
                name, module
            ),
            span,
        )
        .with_hint(format!(
            "consider making `{}` public with 'pub' keyword",
            name
        ))
    }

    /// Unknown trait error.
    pub fn unknown_trait(name: &str, span: Span) -> Self {
        Self::compile(format!("unknown trait `{}`", name), span)
            .with_hint("traits must be defined with 'trait Name ... end'")
    }

    /// Missing trait method error.
    pub fn missing_trait_method(method: &str, ty: &str, trait_name: &str, span: Span) -> Self {
        Self::compile(
            format!(
                "type `{}` does not implement method `{}` required by trait `{}`",
                ty, method, trait_name
            ),
            span,
        )
        .with_hint(format!("add the method: `{}(self) = ...`", method))
    }

    /// Not implemented error (for features under development).
    pub fn not_implemented(feature: &str, span: Span) -> Self {
        Self::compile(format!("{} is not yet implemented", feature), span)
            .with_note("this feature is planned for a future version")
    }
}

// Runtime error constructors

impl SourceError {
    /// Division by zero error.
    pub fn division_by_zero(span: Span) -> Self {
        Self::runtime("division by zero", span)
            .with_hint("check that the divisor is not zero before dividing")
    }

    /// Index out of bounds error.
    pub fn index_out_of_bounds(index: i64, length: usize, span: Span) -> Self {
        Self::runtime(
            format!(
                "index {} is out of bounds for collection of length {}",
                index, length
            ),
            span,
        )
        .with_hint("indices start at 0 and must be less than the length")
    }

    /// Pattern match failed error.
    pub fn match_failed(span: Span) -> Self {
        Self::runtime("no pattern matched the value", span)
            .with_hint("add a catch-all pattern like `_` or `other`")
    }

    /// Assertion failed error.
    pub fn assertion_failed(message: &str, span: Span) -> Self {
        Self::runtime(format!("assertion failed: {}", message), span)
    }
}

/// Convert a chumsky parse error to a SourceError.
pub fn parse_error_to_source_error<T: std::fmt::Display + std::hash::Hash + Eq>(
    error: &chumsky::error::Simple<T>,
) -> SourceError {
    let span = Span::new(error.span().start, error.span().end);

    let message = if let Some(label) = error.label() {
        format!("expected {}", label)
    } else {
        let expected: Vec<String> = error
            .expected()
            .filter_map(|e| e.as_ref().map(|t| format!("`{}`", t)))
            .collect();

        let found = error.found().map(|t| format!("`{}`", t)).unwrap_or_else(|| "end of input".to_string());

        if expected.is_empty() {
            format!("unexpected {}", found)
        } else if expected.len() == 1 {
            format!("expected {}, found {}", expected[0], found)
        } else if expected.len() <= 4 {
            format!("expected one of {}, found {}", expected.join(", "), found)
        } else {
            // For very long lists, just show a helpful message
            format!("unexpected {} (expected expression or statement)", found)
        }
    };

    let mut err = SourceError::parse(message, span);

    // Add hint based on error reason if available
    match error.reason() {
        chumsky::error::SimpleReason::Unclosed { span: unclosed_span, delimiter } => {
            err = err.with_label(
                Span::new(unclosed_span.start, unclosed_span.end),
                format!("unclosed `{}`", delimiter),
            );
            err = err.with_hint("make sure all brackets and delimiters are properly closed");
        }
        chumsky::error::SimpleReason::Custom(msg) => {
            err = err.with_note(msg.clone());
        }
        _ => {}
    }

    err
}

/// Convert multiple chumsky parse errors to SourceErrors.
pub fn parse_errors_to_source_errors<T: std::fmt::Display + std::hash::Hash + Eq>(
    errors: &[chumsky::error::Simple<T>],
) -> Vec<SourceError> {
    errors.iter().map(parse_error_to_source_error).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_formatting() {
        let source = "main() = x + 1";
        let span = Span::new(9, 10); // 'x'
        let err = SourceError::unknown_variable("x", span);
        let output = err.format("test.nos", source);

        assert!(output.contains("unknown variable"));
        assert!(output.contains("x"));
    }

    #[test]
    fn test_offset_to_line_col() {
        let source = "line1\nline2\nline3";
        assert_eq!(offset_to_line_col(source, 0), (1, 1));
        assert_eq!(offset_to_line_col(source, 5), (1, 6));
        assert_eq!(offset_to_line_col(source, 6), (2, 1));
        assert_eq!(offset_to_line_col(source, 12), (3, 1));
    }

    #[test]
    fn test_error_with_hint_and_note() {
        let source = "fn test() = undefined";
        let span = Span::new(12, 21);
        let err = SourceError::unknown_variable("undefined", span)
            .with_note("variables must be in scope to be used");

        let output = err.format("test.nos", source);
        assert!(output.contains("undefined"));
    }
}
