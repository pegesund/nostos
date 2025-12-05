//! AST types for the Nostos programming language.
//!
//! Comprehensive AST covering all syntax from the language specification.

use std::fmt;

/// A span in the source code, used for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl From<std::ops::Range<usize>> for Span {
    fn from(range: std::ops::Range<usize>) -> Self {
        Span::new(range.start, range.end)
    }
}

/// A node with source location information.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// An identifier.
pub type Ident = Spanned<String>;

/// Visibility of an item in a module.
/// Items are private by default and must be marked `pub` to be visible outside the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    /// Private to the current module (default)
    #[default]
    Private,
    /// Publicly visible outside the module
    Public,
}

impl Visibility {
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

/// A complete source file / module.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub name: Option<Ident>,
    pub items: Vec<Item>,
}

/// Top-level items in a module.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// Type definition
    TypeDef(TypeDef),
    /// Function definition
    FnDef(FnDef),
    /// Trait definition
    TraitDef(TraitDef),
    /// Trait implementation
    TraitImpl(TraitImpl),
    /// Module definition
    ModuleDef(ModuleDef),
    /// Use statement
    Use(UseStmt),
    /// Top-level binding
    Binding(Binding),
    /// Test definition
    Test(TestDef),
    /// Extern declaration (FFI)
    Extern(ExternDecl),
}

impl Item {
    /// Get the span of this item.
    pub fn span(&self) -> Span {
        match self {
            Item::TypeDef(def) => def.span,
            Item::FnDef(def) => def.span,
            Item::TraitDef(def) => def.span,
            Item::TraitImpl(impl_) => impl_.span,
            Item::ModuleDef(def) => def.span,
            Item::Use(stmt) => stmt.span,
            Item::Binding(binding) => binding.span,
            Item::Test(test) => test.span,
            Item::Extern(ext) => ext.span,
        }
    }
}

// =============================================================================
// Type Definitions
// =============================================================================

/// A type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub visibility: Visibility,
    pub doc: Option<String>,
    pub mutable: bool,
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub body: TypeBody,
    pub span: Span,
}

/// Type parameter with optional constraints: `K: Eq + Hash`
#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: Ident,
    pub constraints: Vec<Ident>,
}

/// The body of a type definition.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeBody {
    /// Record type: `{x: Float, y: Float}`
    Record(Vec<Field>),
    /// Variant/Sum type: `None | Some(T)`
    Variant(Vec<Variant>),
    /// Type alias: `type Handler = (Request) -> Response`
    Alias(TypeExpr),
    /// Empty type (Never): `type Never =`
    Empty,
}

/// A field in a record type.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub private: bool,
    pub name: Ident,
    pub ty: TypeExpr,
}

/// A variant in a sum type.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: Ident,
    /// Unit variant, positional fields, or named fields
    pub fields: VariantFields,
}

/// Fields in a variant.
#[derive(Debug, Clone, PartialEq)]
pub enum VariantFields {
    /// Unit variant: `None`, `North`
    Unit,
    /// Positional fields: `Some(T)`, `Cons(T, List[T])`
    Positional(Vec<TypeExpr>),
    /// Named fields: `Circle{radius: Float}`
    Named(Vec<Field>),
}

/// A type expression.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// Simple type name: `Int`, `Float`
    Name(Ident),
    /// Generic type: `Option[T]`, `Map[K, V]`
    Generic(Ident, Vec<TypeExpr>),
    /// Function type: `(Int, Int) -> Int`
    Function(Vec<TypeExpr>, Box<TypeExpr>),
    /// Record type inline: `{x: Int, y: Int}`
    Record(Vec<(Ident, TypeExpr)>),
    /// Tuple type: `(Int, String)`
    Tuple(Vec<TypeExpr>),
    /// Unit type: `()`
    Unit,
}

// =============================================================================
// Function Definitions
// =============================================================================

/// A function definition with one or more clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub visibility: Visibility,
    pub doc: Option<String>,
    pub name: Ident,
    pub clauses: Vec<FnClause>,
    pub span: Span,
}

/// A single clause of a function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct FnClause {
    pub params: Vec<FnParam>,
    pub guard: Option<Expr>,
    pub return_type: Option<TypeExpr>,
    pub body: Expr,
    pub span: Span,
}

/// A function parameter (pattern with optional type).
#[derive(Debug, Clone, PartialEq)]
pub struct FnParam {
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
}

// =============================================================================
// Patterns
// =============================================================================

/// A pattern for matching values.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Wildcard: `_`
    Wildcard(Span),
    /// Variable binding: `x`, `name`
    Var(Ident),
    /// Integer literal: `0`, `1`, `42` (default Int64)
    Int(i64, Span),
    /// Typed integer patterns
    Int8(i8, Span),
    Int16(i16, Span),
    Int32(i32, Span),
    UInt8(u8, Span),
    UInt16(u16, Span),
    UInt32(u32, Span),
    UInt64(u64, Span),
    /// BigInt pattern
    BigInt(String, Span),
    /// Float literal: `3.14` (default Float64)
    Float(f64, Span),
    /// Float32 pattern
    Float32(f32, Span),
    /// Decimal pattern
    Decimal(String, Span),
    /// String literal: `"hello"`
    String(String, Span),
    /// Char literal: `'a'`
    Char(char, Span),
    /// Boolean literal: `true`, `false`
    Bool(bool, Span),
    /// Unit: `()`
    Unit(Span),
    /// Tuple pattern: `(a, b)`
    Tuple(Vec<Pattern>, Span),
    /// List pattern: `[]`, `[h | t]`, `[a, b | rest]`
    List(ListPattern, Span),
    /// Record pattern: `{x, y}`, `{name: n}`
    Record(Vec<RecordPatternField>, Span),
    /// Variant pattern: `Some(x)`, `None`, `Circle{radius: r}`
    Variant(Ident, VariantPatternFields, Span),
    /// Pin pattern: `^expected`
    Pin(Box<Expr>, Span),
    /// Or pattern: `0 | 1 | 2`
    Or(Vec<Pattern>, Span),
}

impl Pattern {
    /// Get the span of this pattern.
    pub fn span(&self) -> Span {
        match self {
            Pattern::Wildcard(s) => *s,
            Pattern::Var(ident) => ident.span,
            Pattern::Int(_, s) => *s,
            Pattern::Int8(_, s) => *s,
            Pattern::Int16(_, s) => *s,
            Pattern::Int32(_, s) => *s,
            Pattern::UInt8(_, s) => *s,
            Pattern::UInt16(_, s) => *s,
            Pattern::UInt32(_, s) => *s,
            Pattern::UInt64(_, s) => *s,
            Pattern::BigInt(_, s) => *s,
            Pattern::Float(_, s) => *s,
            Pattern::Float32(_, s) => *s,
            Pattern::Decimal(_, s) => *s,
            Pattern::String(_, s) => *s,
            Pattern::Char(_, s) => *s,
            Pattern::Bool(_, s) => *s,
            Pattern::Unit(s) => *s,
            Pattern::Tuple(_, s) => *s,
            Pattern::List(_, s) => *s,
            Pattern::Record(_, s) => *s,
            Pattern::Variant(_, _, s) => *s,
            Pattern::Pin(_, s) => *s,
            Pattern::Or(_, s) => *s,
        }
    }
}

/// Fields in a variant pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum VariantPatternFields {
    /// Unit: `None`
    Unit,
    /// Positional: `Some(x)`, `Cons(h, t)`
    Positional(Vec<Pattern>),
    /// Named: `Circle{radius: r}`
    Named(Vec<RecordPatternField>),
}

/// A list pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum ListPattern {
    /// Empty list: `[]`
    Empty,
    /// Cons pattern: `[h | t]` or `[a, b | rest]`
    Cons(Vec<Pattern>, Option<Box<Pattern>>),
}

/// A field in a record pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordPatternField {
    /// Punned: `{x}` means `{x: x}`
    Punned(Ident),
    /// Named: `{name: n}`
    Named(Ident, Pattern),
    /// Rest: `{name, _}` - the `_` captures remaining fields
    Rest(Span),
}

// =============================================================================
// Expressions
// =============================================================================

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Integer literal (default Int64)
    Int(i64, Span),
    /// Typed integer literals
    Int8(i8, Span),
    Int16(i16, Span),
    Int32(i32, Span),
    UInt8(u8, Span),
    UInt16(u16, Span),
    UInt32(u32, Span),
    UInt64(u64, Span),
    /// BigInt literal
    BigInt(String, Span),
    /// Float literal (default Float64)
    Float(f64, Span),
    /// Float32 literal
    Float32(f32, Span),
    /// Decimal literal
    Decimal(String, Span),
    /// String literal (possibly with interpolations)
    String(StringLit, Span),
    /// Character literal
    Char(char, Span),
    /// Boolean literal
    Bool(bool, Span),
    /// Unit value: `()`
    Unit(Span),
    /// Variable reference
    Var(Ident),
    /// Binary operation
    BinOp(Box<Expr>, BinOp, Box<Expr>, Span),
    /// Unary operation
    UnaryOp(UnaryOp, Box<Expr>, Span),
    /// Function call: `f(x, y)`
    Call(Box<Expr>, Vec<Expr>, Span),
    /// Method call: `x.f(y)`
    MethodCall(Box<Expr>, Ident, Vec<Expr>, Span),
    /// Field access: `point.x`
    FieldAccess(Box<Expr>, Ident, Span),
    /// Index access: `arr[i]`
    Index(Box<Expr>, Box<Expr>, Span),
    /// Lambda: `x => x * 2`
    Lambda(Vec<Pattern>, Box<Expr>, Span),
    /// If expression: `if cond then a else b`
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    /// Match expression
    Match(Box<Expr>, Vec<MatchArm>, Span),
    /// Tuple: `(a, b, c)`
    Tuple(Vec<Expr>, Span),
    /// List: `[1, 2, 3]`
    List(Vec<Expr>, Option<Box<Expr>>, Span),
    /// Map literal: `%{"key": value}`
    Map(Vec<(Expr, Expr)>, Span),
    /// Set literal: `#{1, 2, 3}`
    Set(Vec<Expr>, Span),
    /// Record construction: `Point(3.0, 4.0)`
    Record(Ident, Vec<RecordField>, Span),
    /// Record update: `Point(p, x: 10.0)`
    RecordUpdate(Ident, Box<Expr>, Vec<RecordField>, Span),
    /// Block: `{ stmt1; stmt2; expr }`
    Block(Vec<Stmt>, Span),
    /// Try/catch/finally
    Try(Box<Expr>, Vec<MatchArm>, Option<Box<Expr>>, Span),
    /// Do block (IO): `do ... end`
    Do(Vec<DoStmt>, Span),
    /// Receive expression
    Receive(Vec<MatchArm>, Option<(Box<Expr>, Box<Expr>)>, Span),
    /// Spawn expression
    Spawn(SpawnKind, Box<Expr>, Vec<Expr>, Span),
    /// Send expression: `pid <- msg`
    Send(Box<Expr>, Box<Expr>, Span),
    /// Error propagation: `expr?`
    Try_(Box<Expr>, Span),
    /// Quote expression
    Quote(Box<Expr>, Span),
    /// Splice in quote: `${expr}`
    Splice(Box<Expr>, Span),
}

impl Expr {
    /// Get the span of this expression.
    pub fn span(&self) -> Span {
        match self {
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
            Expr::Var(ident) => ident.span,
            Expr::BinOp(_, _, _, s) => *s,
            Expr::UnaryOp(_, _, s) => *s,
            Expr::Call(_, _, s) => *s,
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
        }
    }
}

/// String literal (plain or with interpolations).
#[derive(Debug, Clone, PartialEq)]
pub enum StringLit {
    /// Plain string
    Plain(String),
    /// Interpolated string with parts
    Interpolated(Vec<StringPart>),
}

/// Part of an interpolated string.
#[derive(Debug, Clone, PartialEq)]
pub enum StringPart {
    /// Literal text
    Lit(String),
    /// Interpolated expression: `${expr}`
    Expr(Expr),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add, Sub, Mul, Div, Mod, Pow,
    // Comparison
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    // Logical
    And, Or,
    // String/List
    Concat,
    // Pipeline
    Pipe,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::Eq => write!(f, "=="),
            BinOp::NotEq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::LtEq => write!(f, "<="),
            BinOp::GtEq => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::Concat => write!(f, "++"),
            BinOp::Pipe => write!(f, "|>"),
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,  // -
    Not,  // !
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
        }
    }
}

/// A field in a record expression.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordField {
    /// Positional: `Point(3.0, 4.0)`
    Positional(Expr),
    /// Named: `Point(x: 3.0, y: 4.0)`
    Named(Ident, Expr),
}

/// A match arm.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

/// Spawn kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnKind {
    Normal,
    Linked,
    Monitored,
}

/// Statement in a do block.
#[derive(Debug, Clone, PartialEq)]
pub enum DoStmt {
    /// Bind: `name = expr`
    Bind(Pattern, Expr),
    /// Expression
    Expr(Expr),
}

// =============================================================================
// Statements (inside blocks)
// =============================================================================

/// A statement in a block.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Expression statement
    Expr(Expr),
    /// Let binding
    Let(Binding),
    /// Assignment: `x = expr`
    Assign(AssignTarget, Expr, Span),
}

/// A binding.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub mutable: bool,
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

/// Target of an assignment.
#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Var(Ident),
    Field(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
}

// =============================================================================
// Traits
// =============================================================================

/// A trait definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub doc: Option<String>,
    pub name: Ident,
    pub super_traits: Vec<Ident>,
    pub methods: Vec<TraitMethod>,
    pub span: Span,
}

/// A method in a trait.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: Ident,
    pub params: Vec<FnParam>,
    pub return_type: Option<TypeExpr>,
    pub default_impl: Option<Expr>,
    pub span: Span,
}

/// A trait implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitImpl {
    pub ty: TypeExpr,
    pub trait_name: Ident,
    pub when_clause: Vec<(Ident, Vec<Ident>)>,
    pub methods: Vec<FnDef>,
    pub span: Span,
}

// =============================================================================
// Modules
// =============================================================================

/// A module definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDef {
    pub visibility: Visibility,
    pub name: Ident,
    pub items: Vec<Item>,
    pub span: Span,
}

/// A use statement.
#[derive(Debug, Clone, PartialEq)]
pub struct UseStmt {
    pub path: Vec<Ident>,
    pub imports: UseImports,
    pub span: Span,
}

/// What to import.
#[derive(Debug, Clone, PartialEq)]
pub enum UseImports {
    All,
    Named(Vec<UseItem>),
}

/// A single import item.
#[derive(Debug, Clone, PartialEq)]
pub struct UseItem {
    pub name: Ident,
    pub alias: Option<Ident>,
}

// =============================================================================
// Tests
// =============================================================================

/// A test definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TestDef {
    pub name: String,
    pub body: Expr,
    pub span: Span,
}

// =============================================================================
// FFI
// =============================================================================

/// An extern declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternDecl {
    pub kind: ExternKind,
    pub span: Span,
}

/// Kind of extern declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternKind {
    /// Extern function
    Function {
        name: Ident,
        params: Vec<(Ident, TypeExpr)>,
        return_type: TypeExpr,
        from: String,
    },
    /// Extern type
    Type { name: Ident },
}
