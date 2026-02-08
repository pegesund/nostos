//! Comprehensive type safety tests for Nostos.
//!
//! This test file serves as a specification for the type system.
//! Each test documents expected behavior and will drive implementation.
//!
//! Test categories:
//! 1. Type inference - what types are inferred for expressions
//! 2. Type checking - what programs are accepted/rejected
//! 3. Generics - parametric polymorphism
//! 4. Traits - type class constraints
//! 5. Pattern matching - exhaustiveness and type correctness
//! 6. Records - structural typing
//! 7. Variants - sum types
//! 8. Concurrency - message type safety

use nostos_types::*;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// A test case that should type-check successfully.
/// Contains source code and expected inferred type.
#[allow(dead_code)]
struct TypeCheckOk {
    source: &'static str,
    expected_type: Type,
}

/// A test case that should fail type-checking.
/// Contains source code and expected error kind.
#[allow(dead_code)]
struct TypeCheckErr {
    source: &'static str,
    error: ExpectedError,
}

#[derive(Debug)]
#[allow(dead_code)]
enum ExpectedError {
    Mismatch { expected: &'static str, found: &'static str },
    UnknownIdent(&'static str),
    UnknownType(&'static str),
    ArityMismatch { expected: usize, found: usize },
    MissingTrait { ty: &'static str, trait_name: &'static str },
    NonExhaustive,
    ImmutableBinding(&'static str),
    ImmutableField(&'static str),
    NoSuchField { ty: &'static str, field: &'static str },
    MissingField(&'static str),
    ExtraField(&'static str),
    NotCallable(&'static str),
    InfiniteType,
}

// ============================================================================
// 1. LITERAL TYPE INFERENCE
// ============================================================================

mod literal_inference {
    use super::*;

    #[test]
    fn infer_int_literal() {
        // Source: 42
        // Expected: Int
        let _ = TypeCheckOk {
            source: "42",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn infer_hex_int_literal() {
        // Source: 0xFF
        // Expected: Int
        let _ = TypeCheckOk {
            source: "0xFF",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn infer_binary_int_literal() {
        // Source: 0b1010
        // Expected: Int
        let _ = TypeCheckOk {
            source: "0b1010",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn infer_float_literal() {
        // Source: 3.14
        // Expected: Float
        let _ = TypeCheckOk {
            source: "3.14",
            expected_type: Type::Float,
        };
    }

    #[test]
    fn infer_string_literal() {
        // Source: "hello"
        // Expected: String
        let _ = TypeCheckOk {
            source: r#""hello""#,
            expected_type: Type::String,
        };
    }

    #[test]
    fn infer_char_literal() {
        // Source: 'a'
        // Expected: Char
        let _ = TypeCheckOk {
            source: "'a'",
            expected_type: Type::Char,
        };
    }

    #[test]
    fn infer_bool_literal() {
        // Source: true
        // Expected: Bool
        let _ = TypeCheckOk {
            source: "true",
            expected_type: Type::Bool,
        };
    }

    #[test]
    fn infer_unit_literal() {
        // Source: ()
        // Expected: ()
        let _ = TypeCheckOk {
            source: "()",
            expected_type: Type::Unit,
        };
    }
}

// ============================================================================
// 2. COLLECTION TYPE INFERENCE
// ============================================================================

mod collection_inference {
    use super::*;

    #[test]
    fn infer_empty_list() {
        // Source: []
        // Expected: List[?0] (polymorphic, unresolved)
        let _ = TypeCheckOk {
            source: "[]",
            expected_type: Type::List(Box::new(Type::Var(0))),
        };
    }

    #[test]
    fn infer_int_list() {
        // Source: [1, 2, 3]
        // Expected: List[Int]
        let _ = TypeCheckOk {
            source: "[1, 2, 3]",
            expected_type: Type::List(Box::new(Type::Int)),
        };
    }

    #[test]
    fn infer_heterogeneous_list_error() {
        // Source: [1, "hello"]
        // Expected: Error - Int != String
        let _ = TypeCheckErr {
            source: r#"[1, "hello"]"#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }

    #[test]
    fn infer_nested_list() {
        // Source: [[1, 2], [3, 4]]
        // Expected: List[List[Int]]
        let _ = TypeCheckOk {
            source: "[[1, 2], [3, 4]]",
            expected_type: Type::List(Box::new(Type::List(Box::new(Type::Int)))),
        };
    }

    #[test]
    fn infer_tuple() {
        // Source: (1, "hello", true)
        // Expected: (Int, String, Bool)
        let _ = TypeCheckOk {
            source: r#"(1, "hello", true)"#,
            expected_type: Type::Tuple(vec![Type::Int, Type::String, Type::Bool]),
        };
    }

    #[test]
    fn infer_map_literal() {
        // Source: %{"a": 1, "b": 2}
        // Expected: Map[String, Int]
        let _ = TypeCheckOk {
            source: r#"%{"a": 1, "b": 2}"#,
            expected_type: Type::Map(Box::new(Type::String), Box::new(Type::Int)),
        };
    }

    #[test]
    fn infer_set_literal() {
        // Source: #{1, 2, 3}
        // Expected: Set[Int]
        let _ = TypeCheckOk {
            source: "#{1, 2, 3}",
            expected_type: Type::Set(Box::new(Type::Int)),
        };
    }
}

// ============================================================================
// 3. ARITHMETIC AND OPERATOR TYPE CHECKING
// ============================================================================

mod operators {
    use super::*;

    #[test]
    fn int_addition() {
        // Source: 1 + 2
        // Expected: Int
        let _ = TypeCheckOk {
            source: "1 + 2",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn float_addition() {
        // Source: 1.0 + 2.0
        // Expected: Float
        let _ = TypeCheckOk {
            source: "1.0 + 2.0",
            expected_type: Type::Float,
        };
    }

    #[test]
    fn mixed_arithmetic_error() {
        // Source: 1 + 2.0
        // Expected: Error - cannot add Int and Float
        let _ = TypeCheckErr {
            source: "1 + 2.0",
            error: ExpectedError::Mismatch { expected: "Int", found: "Float" },
        };
    }

    #[test]
    fn string_concat() {
        // Source: "hello" ++ " world"
        // Expected: String
        let _ = TypeCheckOk {
            source: r#""hello" ++ " world""#,
            expected_type: Type::String,
        };
    }

    #[test]
    fn list_concat() {
        // Source: [1, 2] ++ [3, 4]
        // Expected: List[Int]
        let _ = TypeCheckOk {
            source: "[1, 2] ++ [3, 4]",
            expected_type: Type::List(Box::new(Type::Int)),
        };
    }

    #[test]
    fn comparison_result() {
        // Source: 1 < 2
        // Expected: Bool
        let _ = TypeCheckOk {
            source: "1 < 2",
            expected_type: Type::Bool,
        };
    }

    #[test]
    fn comparison_type_mismatch() {
        // Source: 1 < "hello"
        // Expected: Error
        let _ = TypeCheckErr {
            source: r#"1 < "hello""#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }

    #[test]
    fn equality_same_type() {
        // Source: 1 == 2
        // Expected: Bool
        let _ = TypeCheckOk {
            source: "1 == 2",
            expected_type: Type::Bool,
        };
    }

    #[test]
    fn logical_and() {
        // Source: true && false
        // Expected: Bool
        let _ = TypeCheckOk {
            source: "true && false",
            expected_type: Type::Bool,
        };
    }

    #[test]
    fn logical_and_non_bool() {
        // Source: 1 && 2
        // Expected: Error - && requires Bool
        let _ = TypeCheckErr {
            source: "1 && 2",
            error: ExpectedError::Mismatch { expected: "Bool", found: "Int" },
        };
    }

    #[test]
    fn negation() {
        // Source: !true
        // Expected: Bool
        let _ = TypeCheckOk {
            source: "!true",
            expected_type: Type::Bool,
        };
    }

    #[test]
    fn unary_minus() {
        // Source: -42
        // Expected: Int
        let _ = TypeCheckOk {
            source: "-42",
            expected_type: Type::Int,
        };
    }
}

// ============================================================================
// 4. FUNCTION TYPE INFERENCE
// ============================================================================

mod functions {
    use super::*;

    #[test]
    fn simple_function() {
        // Source: add(a, b) = a + b
        // With usage: add(1, 2)
        // Expected: (Int, Int) -> Int (inferred from usage)
        let _ = TypeCheckOk {
            source: "add(1, 2)",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn annotated_function() {
        // Source: add(a: Int, b: Int) -> Int = a + b
        // Expected: (Int, Int) -> Int
        let _ = TypeCheckOk {
            source: "add(a: Int, b: Int) -> Int = a + b",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![],
                params: vec![Type::Int, Type::Int],
                ret: Box::new(Type::Int),
            }),
        };
    }

    #[test]
    fn wrong_return_type() {
        // Source: foo() -> Int = "hello"
        // Expected: Error - return type mismatch
        let _ = TypeCheckErr {
            source: r#"foo() -> Int = "hello""#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }

    #[test]
    fn wrong_argument_type() {
        // Source:
        //   square(x: Int) -> Int = x * x
        //   square("hello")
        // Expected: Error - argument type mismatch
        let _ = TypeCheckErr {
            source: r#"square("hello")"#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }

    #[test]
    fn arity_mismatch() {
        // Source:
        //   add(a, b) = a + b
        //   add(1)
        // Expected: Error - wrong number of arguments
        let _ = TypeCheckErr {
            source: "add(1)",
            error: ExpectedError::ArityMismatch { expected: 2, found: 1 },
        };
    }

    #[test]
    fn lambda_inference() {
        // Source: x => x + 1
        // Expected: (Int) -> Int (inferred from + 1)
        let _ = TypeCheckOk {
            source: "x => x + 1",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            }),
        };
    }

    #[test]
    fn higher_order_function() {
        // Source:
        //   map(f, []) = []
        //   map(f, [h | t]) = [f(h) | map(f, t)]
        //   map(x => x * 2, [1, 2, 3])
        // Expected: List[Int]
        let _ = TypeCheckOk {
            source: "map(x => x * 2, [1, 2, 3])",
            expected_type: Type::List(Box::new(Type::Int)),
        };
    }

    #[test]
    fn not_callable() {
        // Source: 42(1, 2)
        // Expected: Error - Int is not callable
        let _ = TypeCheckErr {
            source: "42(1, 2)",
            error: ExpectedError::NotCallable("Int"),
        };
    }
}

// ============================================================================
// 5. PATTERN MATCHING
// ============================================================================

mod pattern_matching {
    use super::*;

    #[test]
    fn literal_pattern() {
        // Source:
        //   fib(0) = 0
        //   fib(1) = 1
        //   fib(n) = fib(n-1) + fib(n-2)
        // Expected: (Int) -> Int
        let _ = TypeCheckOk {
            source: "fib(0) = 0\nfib(1) = 1\nfib(n) = fib(n - 1) + fib(n - 2)",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![],
                params: vec![Type::Int],
                ret: Box::new(Type::Int),
            }),
        };
    }

    #[test]
    fn list_pattern() {
        // Source:
        //   sum([]) = 0
        //   sum([h | t]) = h + sum(t)
        // Expected: (List[Int]) -> Int
        let _ = TypeCheckOk {
            source: "sum([]) = 0\nsum([h | t]) = h + sum(t)",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![],
                params: vec![Type::List(Box::new(Type::Int))],
                ret: Box::new(Type::Int),
            }),
        };
    }

    #[test]
    fn tuple_pattern() {
        // Source: (a, b) = (1, 2)
        // Expected: a: Int, b: Int
        let _ = TypeCheckOk {
            source: "(a, b) = (1, 2)",
            expected_type: Type::Unit, // binding returns ()
        };
    }

    #[test]
    fn record_pattern() {
        // Source:
        //   type Point = {x: Float, y: Float}
        //   getX({x, y}) = x
        // Expected: (Point) -> Float
        let _ = TypeCheckOk {
            source: "getX({x, y}) = x",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![],
                params: vec![Type::Named {
                    name: "Point".to_string(),
                    args: vec![]
                }],
                ret: Box::new(Type::Float),
            }),
        };
    }

    #[test]
    fn variant_pattern() {
        // Source:
        //   unwrap(Some(x)) = x
        //   unwrap(None) = panic("empty")
        // Expected: (Option[T]) -> T
        let _ = TypeCheckOk {
            source: r#"unwrap(Some(x)) = x\nunwrap(None) = panic("empty")"#,
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![TypeParam {
                    name: "T".to_string(),
                    constraints: vec![]
                }],
                params: vec![Type::Named {
                    name: "Option".to_string(),
                    args: vec![Type::TypeParam("T".to_string())]
                }],
                ret: Box::new(Type::TypeParam("T".to_string())),
            }),
        };
    }

    #[test]
    fn non_exhaustive_match() {
        // Source:
        //   match opt
        //     Some(x) -> x
        //   # Missing None case!
        // Expected: Error - non-exhaustive
        let _ = TypeCheckErr {
            source: "match opt\n  Some(x) -> x\nend",
            error: ExpectedError::NonExhaustive,
        };
    }

    #[test]
    fn exhaustive_match_with_wildcard() {
        // Source:
        //   match n
        //     0 -> "zero"
        //     _ -> "other"
        // Expected: String (exhaustive due to wildcard)
        let _ = TypeCheckOk {
            source: r#"match n\n  0 -> "zero"\n  _ -> "other"\nend"#,
            expected_type: Type::String,
        };
    }

    #[test]
    fn pattern_type_mismatch() {
        // Source:
        //   f(1) = "one"
        //   f("x") = "x"  # Error: pattern type mismatch
        // Expected: Error
        let _ = TypeCheckErr {
            source: r#"f(1) = "one"\nf("x") = "x""#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }
}

// ============================================================================
// 6. RECORDS
// ============================================================================

mod records {
    use super::*;

    #[test]
    fn record_construction() {
        // Source:
        //   type Point = {x: Float, y: Float}
        //   p = Point(3.0, 4.0)
        // Expected: Point
        let _ = TypeCheckOk {
            source: "Point(3.0, 4.0)",
            expected_type: Type::Named { name: "Point".to_string(), args: vec![] },
        };
    }

    #[test]
    fn record_construction_named() {
        // Source: Point(x: 3.0, y: 4.0)
        // Expected: Point
        let _ = TypeCheckOk {
            source: "Point(x: 3.0, y: 4.0)",
            expected_type: Type::Named { name: "Point".to_string(), args: vec![] },
        };
    }

    #[test]
    fn record_field_wrong_type() {
        // Source: Point(x: "hello", y: 4.0)
        // Expected: Error - x should be Float
        let _ = TypeCheckErr {
            source: r#"Point(x: "hello", y: 4.0)"#,
            error: ExpectedError::Mismatch { expected: "Float", found: "String" },
        };
    }

    #[test]
    fn record_missing_field() {
        // Source: Point(x: 3.0)
        // Expected: Error - missing y
        let _ = TypeCheckErr {
            source: "Point(x: 3.0)",
            error: ExpectedError::MissingField("y"),
        };
    }

    #[test]
    fn record_extra_field() {
        // Source: Point(x: 3.0, y: 4.0, z: 5.0)
        // Expected: Error - extra field z
        let _ = TypeCheckErr {
            source: "Point(x: 3.0, y: 4.0, z: 5.0)",
            error: ExpectedError::ExtraField("z"),
        };
    }

    #[test]
    fn record_field_access() {
        // Source: p.x (where p: Point)
        // Expected: Float
        let _ = TypeCheckOk {
            source: "p.x",
            expected_type: Type::Float,
        };
    }

    #[test]
    fn record_nonexistent_field() {
        // Source: p.z (where p: Point)
        // Expected: Error - Point has no field z
        let _ = TypeCheckErr {
            source: "p.z",
            error: ExpectedError::NoSuchField { ty: "Point", field: "z" },
        };
    }

    #[test]
    fn record_update() {
        // Source: Point(p, x: 10.0)
        // Expected: Point
        let _ = TypeCheckOk {
            source: "Point(p, x: 10.0)",
            expected_type: Type::Named { name: "Point".to_string(), args: vec![] },
        };
    }

    #[test]
    fn immutable_field_mutation() {
        // Source:
        //   type Point = {x: Float, y: Float}  # immutable
        //   p.x = 10.0
        // Expected: Error - field is immutable
        let _ = TypeCheckErr {
            source: "p.x = 10.0",
            error: ExpectedError::ImmutableField("x"),
        };
    }

    #[test]
    fn mutable_field_mutation() {
        // Source:
        //   var type MutPoint = {x: Float, y: Float}
        //   p.x = 10.0
        // Expected: () (mutation returns unit)
        let _ = TypeCheckOk {
            source: "p.x = 10.0",
            expected_type: Type::Unit,
        };
    }
}

// ============================================================================
// 7. VARIANTS (SUM TYPES)
// ============================================================================

mod variants {
    use super::*;

    #[test]
    fn variant_construction_unit() {
        // Source: None
        // Expected: Option[?0]
        let _ = TypeCheckOk {
            source: "None",
            expected_type: Type::Named {
                name: "Option".to_string(),
                args: vec![Type::Var(0)]
            },
        };
    }

    #[test]
    fn variant_construction_positional() {
        // Source: Some(42)
        // Expected: Option[Int]
        let _ = TypeCheckOk {
            source: "Some(42)",
            expected_type: Type::Named {
                name: "Option".to_string(),
                args: vec![Type::Int]
            },
        };
    }

    #[test]
    fn variant_construction_named() {
        // Source: Circle{radius: 5.0}
        // Expected: Shape
        let _ = TypeCheckOk {
            source: "Circle{radius: 5.0}",
            expected_type: Type::Named { name: "Shape".to_string(), args: vec![] },
        };
    }

    #[test]
    fn variant_wrong_field_type() {
        // Source: Circle{radius: "big"}
        // Expected: Error - radius should be Float
        let _ = TypeCheckErr {
            source: r#"Circle{radius: "big"}"#,
            error: ExpectedError::Mismatch { expected: "Float", found: "String" },
        };
    }

    #[test]
    fn unknown_constructor() {
        // Source: Unknown(42)
        // Expected: Error - unknown constructor
        let _ = TypeCheckErr {
            source: "Unknown(42)",
            error: ExpectedError::UnknownType("Unknown"),
        };
    }

    #[test]
    fn result_ok() {
        // Source: Ok(42)
        // Expected: Result[Int, ?0]
        let _ = TypeCheckOk {
            source: "Ok(42)",
            expected_type: Type::Named {
                name: "Result".to_string(),
                args: vec![Type::Int, Type::Var(0)]
            },
        };
    }

    #[test]
    fn result_err() {
        // Source: Err("something went wrong")
        // Expected: Result[?0, String]
        let _ = TypeCheckOk {
            source: r#"Err("something went wrong")"#,
            expected_type: Type::Named {
                name: "Result".to_string(),
                args: vec![Type::Var(0), Type::String]
            },
        };
    }
}

// ============================================================================
// 8. GENERICS
// ============================================================================

mod generics {
    use super::*;

    #[test]
    fn identity_function() {
        // Source: id(x) = x
        // Expected: (T) -> T
        let _ = TypeCheckOk {
            source: "id(x) = x",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![TypeParam { name: "T".to_string(), constraints: vec![] }],
                params: vec![Type::TypeParam("T".to_string())],
                ret: Box::new(Type::TypeParam("T".to_string())),
            }),
        };
    }

    #[test]
    fn identity_instantiation() {
        // Source: id(42)
        // Expected: Int
        let _ = TypeCheckOk {
            source: "id(42)",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn generic_map() {
        // Source:
        //   map(f, []) = []
        //   map(f, [h | t]) = [f(h) | map(f, t)]
        // Expected: ((A) -> B, List[A]) -> List[B]
        let _ = TypeCheckOk {
            source: "map(f, []) = []\nmap(f, [h | t]) = [f(h) | map(f, t)]",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![
                    TypeParam { name: "A".to_string(), constraints: vec![] },
                    TypeParam { name: "B".to_string(), constraints: vec![] },
                ],
                params: vec![
                    Type::Function(FunctionType {
                        required_params: None,
                        type_params: vec![],
                        params: vec![Type::TypeParam("A".to_string())],
                        ret: Box::new(Type::TypeParam("B".to_string())),
                    }),
                    Type::List(Box::new(Type::TypeParam("A".to_string()))),
                ],
                ret: Box::new(Type::List(Box::new(Type::TypeParam("B".to_string())))),
            }),
        };
    }

    #[test]
    fn generic_instantiation_mismatch() {
        // Source:
        //   first([h | _]) = h
        //   x: String = first([1, 2, 3])  # Error: Int != String
        // Expected: Error
        let _ = TypeCheckErr {
            source: "x: String = first([1, 2, 3])",
            error: ExpectedError::Mismatch { expected: "String", found: "Int" },
        };
    }

    #[test]
    fn type_parameter_in_record() {
        // Source:
        //   type Box[T] = {value: T}
        //   Box(value: 42)
        // Expected: Box[Int]
        let _ = TypeCheckOk {
            source: "Box(value: 42)",
            expected_type: Type::Named {
                name: "Box".to_string(),
                args: vec![Type::Int]
            },
        };
    }

    #[test]
    fn nested_generics() {
        // Source: Some([1, 2, 3])
        // Expected: Option[List[Int]]
        let _ = TypeCheckOk {
            source: "Some([1, 2, 3])",
            expected_type: Type::Named {
                name: "Option".to_string(),
                args: vec![Type::List(Box::new(Type::Int))]
            },
        };
    }
}

// ============================================================================
// 9. TRAITS
// ============================================================================

mod traits {
    use super::*;

    #[test]
    fn trait_constraint() {
        // Source:
        //   equals(a: T, b: T) -> Bool when T: Eq = a == b
        // Expected: (T, T) -> Bool where T: Eq
        let _ = TypeCheckOk {
            source: "equals(a, b) when T: Eq = a == b",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![TypeParam {
                    name: "T".to_string(),
                    constraints: vec!["Eq".to_string()]
                }],
                params: vec![
                    Type::TypeParam("T".to_string()),
                    Type::TypeParam("T".to_string()),
                ],
                ret: Box::new(Type::Bool),
            }),
        };
    }

    #[test]
    fn missing_trait_impl() {
        // Source:
        //   type NoEq = NoEq(Int)  # doesn't implement Eq
        //   NoEq(1) == NoEq(2)
        // Expected: Error - NoEq doesn't implement Eq
        let _ = TypeCheckErr {
            source: "NoEq(1) == NoEq(2)",
            error: ExpectedError::MissingTrait { ty: "NoEq", trait_name: "Eq" },
        };
    }

    #[test]
    fn trait_method_call() {
        // Source:
        //   42.show()
        // Expected: String (Int implements Show)
        let _ = TypeCheckOk {
            source: "42.show()",
            expected_type: Type::String,
        };
    }

    #[test]
    fn constrained_generic() {
        // Source:
        //   max(a, b) when T: Ord = if a > b then a else b
        // Expected: (T, T) -> T where T: Ord
        let _ = TypeCheckOk {
            source: "max(a, b) = if a > b then a else b",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![TypeParam {
                    name: "T".to_string(),
                    constraints: vec!["Ord".to_string()]
                }],
                params: vec![
                    Type::TypeParam("T".to_string()),
                    Type::TypeParam("T".to_string()),
                ],
                ret: Box::new(Type::TypeParam("T".to_string())),
            }),
        };
    }

    #[test]
    fn multiple_constraints() {
        // Source:
        //   hashAndShow(x) when T: Eq + Hash + Show = x.hash().show()
        // Expected: (T) -> String where T: Eq + Hash + Show
        let _ = TypeCheckOk {
            source: "hashAndShow(x) = x.hash().show()",
            expected_type: Type::Function(FunctionType {
                required_params: None,
                type_params: vec![TypeParam {
                    name: "T".to_string(),
                    constraints: vec![
                        "Eq".to_string(),
                        "Hash".to_string(),
                        "Show".to_string()
                    ]
                }],
                params: vec![Type::TypeParam("T".to_string())],
                ret: Box::new(Type::String),
            }),
        };
    }

    #[test]
    fn conditional_impl() {
        // Source:
        //   # Option[T] implements Show when T: Show
        //   Some(42).show()
        // Expected: String (because Int: Show)
        let _ = TypeCheckOk {
            source: "Some(42).show()",
            expected_type: Type::String,
        };
    }
}

// ============================================================================
// 10. BINDINGS AND MUTABILITY
// ============================================================================

mod bindings {
    use super::*;

    #[test]
    fn immutable_binding() {
        // Source:
        //   x = 42
        //   x = 43  # Error
        // Expected: Error - immutable
        let _ = TypeCheckErr {
            source: "x = 42\nx = 43",
            error: ExpectedError::ImmutableBinding("x"),
        };
    }

    #[test]
    fn mutable_binding() {
        // Source:
        //   var x = 42
        //   x = 43  # OK
        // Expected: ()
        let _ = TypeCheckOk {
            source: "var x = 42\nx = 43",
            expected_type: Type::Unit,
        };
    }

    #[test]
    fn mutable_binding_type_change() {
        // Source:
        //   var x = 42
        //   x = "hello"  # Error - type mismatch
        // Expected: Error
        let _ = TypeCheckErr {
            source: r#"var x = 42\nx = "hello""#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }

    #[test]
    fn compound_assignment() {
        // Source:
        //   var x = 42
        //   x += 1
        // Expected: ()
        let _ = TypeCheckOk {
            source: "var x = 42\nx += 1",
            expected_type: Type::Unit,
        };
    }

    #[test]
    fn unknown_variable() {
        // Source: unknown_var + 1
        // Expected: Error - unknown identifier
        let _ = TypeCheckErr {
            source: "unknown_var + 1",
            error: ExpectedError::UnknownIdent("unknown_var"),
        };
    }

    #[test]
    fn shadowing() {
        // Source:
        //   x = 42
        //   x = "hello"  # This is shadowing, not mutation
        //   x
        // Expected: String (shadowing creates new binding)
        // Note: This depends on semantics - ML allows, Rust warns
        let _ = TypeCheckOk {
            source: r#"x = 42\nx = "hello"\nx"#,
            expected_type: Type::String,
        };
    }
}

// ============================================================================
// 11. CONTROL FLOW
// ============================================================================

mod control_flow {
    use super::*;

    #[test]
    fn if_then_else_same_type() {
        // Source: if true then 1 else 2
        // Expected: Int
        let _ = TypeCheckOk {
            source: "if true then 1 else 2",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn if_then_else_different_types() {
        // Source: if true then 1 else "hello"
        // Expected: Error - branches have different types
        let _ = TypeCheckErr {
            source: r#"if true then 1 else "hello""#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }

    #[test]
    fn if_non_bool_condition() {
        // Source: if 1 then "yes" else "no"
        // Expected: Error - condition must be Bool
        let _ = TypeCheckErr {
            source: r#"if 1 then "yes" else "no""#,
            error: ExpectedError::Mismatch { expected: "Bool", found: "Int" },
        };
    }

    #[test]
    fn match_arms_same_type() {
        // Source:
        //   match x
        //     0 -> "zero"
        //     1 -> "one"
        //     _ -> "many"
        // Expected: String
        let _ = TypeCheckOk {
            source: r#"match x\n  0 -> "zero"\n  1 -> "one"\n  _ -> "many"\nend"#,
            expected_type: Type::String,
        };
    }

    #[test]
    fn match_arms_different_types() {
        // Source:
        //   match x
        //     0 -> "zero"
        //     _ -> 42
        // Expected: Error - arms have different types
        let _ = TypeCheckErr {
            source: r#"match x\n  0 -> "zero"\n  _ -> 42\nend"#,
            error: ExpectedError::Mismatch { expected: "String", found: "Int" },
        };
    }

    #[test]
    fn try_catch_types() {
        // Source:
        //   try { compute() } catch
        //     Error(e) -> default()
        //   end
        // Expected: same type as compute() and default()
        let _ = TypeCheckOk {
            source: "try { compute() } catch\n  Error(e) -> default()\nend",
            expected_type: Type::Var(0), // depends on compute/default
        };
    }
}

// ============================================================================
// 12. CONCURRENCY TYPES
// ============================================================================

mod concurrency {
    use super::*;

    #[test]
    fn spawn_returns_pid() {
        // Source: spawn(worker)
        // Expected: Pid
        let _ = TypeCheckOk {
            source: "spawn(worker)",
            expected_type: Type::Pid,
        };
    }

    #[test]
    fn spawn_monitor_returns_tuple() {
        // Source: spawn_monitor(worker)
        // Expected: (Pid, Ref)
        let _ = TypeCheckOk {
            source: "spawn_monitor(worker)",
            expected_type: Type::Tuple(vec![Type::Pid, Type::Ref]),
        };
    }

    #[test]
    fn send_operator() {
        // Source: pid <- msg
        // Expected: () (send returns unit)
        let _ = TypeCheckOk {
            source: "pid <- msg",
            expected_type: Type::Unit,
        };
    }

    #[test]
    fn receive_expression() {
        // Source:
        //   receive
        //     Inc(sender) -> sender <- Ok
        //     Get(sender) -> sender <- Value(state)
        //   end
        // Expected: () (the result of the matched arm)
        let _ = TypeCheckOk {
            source: "receive\n  Inc(sender) -> sender <- Ok\n  Get(sender) -> sender <- Value(state)\nend",
            expected_type: Type::Unit,
        };
    }

    #[test]
    fn receive_with_timeout() {
        // Source:
        //   receive
        //     Value(n) -> n
        //   after 5000 ->
        //     0
        //   end
        // Expected: Int
        let _ = TypeCheckOk {
            source: "receive\n  Value(n) -> n\nafter 5000 ->\n  0\nend",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn self_returns_pid() {
        // Source: self()
        // Expected: Pid
        let _ = TypeCheckOk {
            source: "self()",
            expected_type: Type::Pid,
        };
    }
}

// ============================================================================
// 13. RESULT AND OPTION HANDLING
// ============================================================================

mod result_option {
    use super::*;

    #[test]
    fn try_operator_propagates() {
        // Source:
        //   foo() -> Result[Int, Error] = {
        //     x = bar()?  # propagates Err
        //     Ok(x + 1)
        //   }
        // Expected: Result[Int, Error]
        let _ = TypeCheckOk {
            source: "foo() = { x = bar()?\n  Ok(x + 1) }",
            expected_type: Type::Named {
                name: "Result".to_string(),
                args: vec![Type::Int, Type::Named {
                    name: "Error".to_string(),
                    args: vec![]
                }],
            },
        };
    }

    #[test]
    fn try_on_non_result() {
        // Source: 42?
        // Expected: Error - ? requires Result
        let _ = TypeCheckErr {
            source: "42?",
            error: ExpectedError::Mismatch {
                expected: "Result[_, _]",
                found: "Int"
            },
        };
    }

    #[test]
    fn map_on_option() {
        // Source: Some(42).map(x => x * 2)
        // Expected: Option[Int]
        let _ = TypeCheckOk {
            source: "Some(42).map(x => x * 2)",
            expected_type: Type::Named {
                name: "Option".to_string(),
                args: vec![Type::Int],
            },
        };
    }

    #[test]
    fn flatmap_on_result() {
        // Source: Ok(42).flatMap(x => Ok(x.show()))
        // Expected: Result[String, ?0]
        let _ = TypeCheckOk {
            source: "Ok(42).flatMap(x => Ok(x.show()))",
            expected_type: Type::Named {
                name: "Result".to_string(),
                args: vec![Type::String, Type::Var(0)],
            },
        };
    }

    #[test]
    fn get_or_default() {
        // Source: Some(42).getOr(0)
        // Expected: Int
        let _ = TypeCheckOk {
            source: "Some(42).getOr(0)",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn get_or_type_mismatch() {
        // Source: Some(42).getOr("default")
        // Expected: Error - Int != String
        let _ = TypeCheckErr {
            source: r#"Some(42).getOr("default")"#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }
}

// ============================================================================
// 14. PIPE OPERATOR
// ============================================================================

mod pipe_operator {
    use super::*;

    #[test]
    fn simple_pipe() {
        // Source: 42 |> double
        // Equivalent to: double(42)
        // Expected: Int (if double: Int -> Int)
        let _ = TypeCheckOk {
            source: "42 |> double",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn chained_pipes() {
        // Source: [1, 2, 3] |> map(double) |> sum
        // Expected: Int
        let _ = TypeCheckOk {
            source: "[1, 2, 3] |> map(double) |> sum",
            expected_type: Type::Int,
        };
    }

    #[test]
    fn pipe_type_mismatch() {
        // Source: "hello" |> double  # double expects Int
        // Expected: Error
        let _ = TypeCheckErr {
            source: r#""hello" |> double"#,
            error: ExpectedError::Mismatch { expected: "Int", found: "String" },
        };
    }
}

// ============================================================================
// 15. IO TYPES
// ============================================================================

mod io_types {
    use super::*;

    #[test]
    fn io_action() {
        // Source: println("hello")
        // Expected: IO[()]
        let _ = TypeCheckOk {
            source: r#"println("hello")"#,
            expected_type: Type::IO(Box::new(Type::Unit)),
        };
    }

    #[test]
    fn io_flatmap() {
        // Source:
        //   readLine().flatMap(name => println("Hello, " ++ name))
        // Expected: IO[()]
        let _ = TypeCheckOk {
            source: r#"readLine().flatMap(name => println("Hello, " ++ name))"#,
            expected_type: Type::IO(Box::new(Type::Unit)),
        };
    }

    #[test]
    fn do_notation() {
        // Source:
        //   do
        //     name = readLine()
        //     println("Hello, " ++ name)
        //   end
        // Expected: IO[()]
        let _ = TypeCheckOk {
            source: r#"do\n  name = readLine()\n  println("Hello, " ++ name)\nend"#,
            expected_type: Type::IO(Box::new(Type::Unit)),
        };
    }
}

// ============================================================================
// 16. RECURSIVE TYPES
// ============================================================================

mod recursive_types {
    use super::*;

    #[test]
    fn recursive_list() {
        // Source:
        //   type List[T] = Nil | Cons(T, List[T])
        //   Cons(1, Cons(2, Nil))
        // Expected: List[Int]
        let _ = TypeCheckOk {
            source: "Cons(1, Cons(2, Nil))",
            expected_type: Type::Named {
                name: "List".to_string(),
                args: vec![Type::Int],
            },
        };
    }

    #[test]
    fn recursive_tree() {
        // Source:
        //   type Tree[T] = Leaf(T) | Node{left: Tree[T], value: T, right: Tree[T]}
        //   Node{left: Leaf(1), value: 2, right: Leaf(3)}
        // Expected: Tree[Int]
        let _ = TypeCheckOk {
            source: "Node{left: Leaf(1), value: 2, right: Leaf(3)}",
            expected_type: Type::Named {
                name: "Tree".to_string(),
                args: vec![Type::Int],
            },
        };
    }

    #[test]
    fn infinite_type_error() {
        // Source: f(x) = f  # f would have type T = T -> ?
        // Expected: Error - infinite type
        let _ = TypeCheckErr {
            source: "f(x) = f",
            error: ExpectedError::InfiniteType,
        };
    }

    #[test]
    fn mutual_recursion() {
        // Source:
        //   isEven(0) = true
        //   isEven(n) = isOdd(n - 1)
        //   isOdd(0) = false
        //   isOdd(n) = isEven(n - 1)
        // Expected: Both (Int) -> Bool
        let _ = TypeCheckOk {
            source: "isEven(0) = true\nisEven(n) = isOdd(n - 1)\nisOdd(0) = false\nisOdd(n) = isEven(n - 1)",
            expected_type: Type::Unit, // module level
        };
    }
}

// ============================================================================
// 17. UFCS (UNIFORM FUNCTION CALL SYNTAX)
// ============================================================================

mod ufcs {
    use super::*;

    #[test]
    fn method_as_function() {
        // Source:
        //   length(s: String) -> Int = ...
        //   "hello".length()
        // Expected: Int
        let _ = TypeCheckOk {
            source: r#""hello".length()"#,
            expected_type: Type::Int,
        };
    }

    #[test]
    fn chained_methods() {
        // Source:
        //   "hello".uppercase().reverse()
        // Expected: String
        let _ = TypeCheckOk {
            source: r#""hello".uppercase().reverse()"#,
            expected_type: Type::String,
        };
    }

    #[test]
    fn method_with_args() {
        // Source:
        //   replace(s: String, from: String, to: String) -> String = ...
        //   "hello".replace("l", "L")
        // Expected: String
        let _ = TypeCheckOk {
            source: r#""hello".replace("l", "L")"#,
            expected_type: Type::String,
        };
    }

    #[test]
    fn ufcs_wrong_type() {
        // Source:
        //   length(s: String) -> Int = ...
        //   42.length()  # Int doesn't have length
        // Expected: Error
        let _ = TypeCheckErr {
            source: "42.length()",
            error: ExpectedError::Mismatch { expected: "String", found: "Int" },
        };
    }
}

// ============================================================================
// RUN FRAMEWORK TESTS
// ============================================================================

#[test]
fn test_type_env_basic() {
    let env = standard_env();
    assert!(env.lookup_type("Option").is_some());
    assert!(env.lookup_type("Result").is_some());
    assert!(env.lookup_trait("Eq").is_some());
    assert!(env.implements(&Type::Int, "Eq"));
}
