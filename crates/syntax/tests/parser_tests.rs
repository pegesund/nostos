//! Parser tests for Nostos syntax.
//!
//! Comprehensive test suite covering all language features from the specification.

use nostos_syntax::{parse, parse_expr};

/// Helper to check parsing succeeds and return the AST.
fn parse_ok(source: &str) -> nostos_syntax::Module {
    let (result, errors) = parse(source);
    if !errors.is_empty() {
        panic!("Parse errors: {:?}\nSource: {}", errors, source);
    }
    result.expect("Expected successful parse")
}

/// Helper to check expression parsing succeeds.
fn parse_expr_ok(source: &str) -> nostos_syntax::Expr {
    let (result, errors) = parse_expr(source);
    if !errors.is_empty() {
        panic!("Parse errors: {:?}\nSource: {}", errors, source);
    }
    result.expect("Expected successful parse")
}

// =============================================================================
// Literal Tests
// =============================================================================

mod literals {
    use super::*;

    #[test]
    fn test_integers() {
        parse_expr_ok("42");
        parse_expr_ok("0");
        parse_expr_ok("1_000_000");
    }

    #[test]
    fn test_hex_integers() {
        parse_expr_ok("0xFF");
        parse_expr_ok("0x1A2B");
        parse_expr_ok("0xDEAD_BEEF");
    }

    #[test]
    fn test_binary_integers() {
        parse_expr_ok("0b1010");
        parse_expr_ok("0b1111_0000");
        parse_expr_ok("0b0");
    }

    #[test]
    fn test_floats() {
        parse_expr_ok("3.14");
        parse_expr_ok("1.0");
        parse_expr_ok("1.2e-10");
        parse_expr_ok("2.5e10");
    }

    #[test]
    fn test_strings() {
        parse_expr_ok("\"hello world\"");
        parse_expr_ok("\"\"");
        parse_expr_ok("\"with\\nescape\"");
    }

    #[test]
    fn test_chars() {
        parse_expr_ok("'a'");
        parse_expr_ok("'\\n'");
        parse_expr_ok("'\\t'");
        parse_expr_ok("' '");
    }

    #[test]
    fn test_booleans() {
        parse_expr_ok("true");
        parse_expr_ok("false");
    }

    #[test]
    fn test_unit() {
        parse_expr_ok("()");
    }
}

// =============================================================================
// Expression Tests
// =============================================================================

mod expressions {
    use super::*;

    #[test]
    fn test_variables() {
        parse_expr_ok("x");
        parse_expr_ok("foo");
        parse_expr_ok("_private");
        parse_expr_ok("camelCase");
    }

    #[test]
    fn test_arithmetic() {
        parse_expr_ok("1 + 2");
        parse_expr_ok("3 - 4");
        parse_expr_ok("5 * 6");
        parse_expr_ok("7 / 8");
        parse_expr_ok("9 % 10");
        parse_expr_ok("2 ** 10");
    }

    #[test]
    fn test_precedence() {
        let expr = parse_expr_ok("2 + 3 * 4");
        assert!(matches!(expr, nostos_syntax::Expr::BinOp(_, nostos_syntax::BinOp::Add, _, _)));
    }

    #[test]
    fn test_comparison() {
        parse_expr_ok("a == b");
        parse_expr_ok("a != b");
        parse_expr_ok("a < b");
        parse_expr_ok("a > b");
        parse_expr_ok("a <= b");
        parse_expr_ok("a >= b");
    }

    #[test]
    fn test_logical() {
        parse_expr_ok("a && b");
        parse_expr_ok("a || b");
        parse_expr_ok("!a");
    }

    #[test]
    fn test_unary() {
        parse_expr_ok("-x");
        parse_expr_ok("!flag");
        parse_expr_ok("--x");
    }

    #[test]
    fn test_string_concat() {
        parse_expr_ok("\"hello\" ++ \" world\"");
    }

    #[test]
    fn test_pipe_operator() {
        parse_expr_ok("x |> f");
        parse_expr_ok("x |> f |> g");
        parse_expr_ok("data |> filter(p) |> map(f)");
    }

    #[test]
    fn test_function_call() {
        parse_expr_ok("f()");
        parse_expr_ok("f(x)");
        parse_expr_ok("f(x, y)");
        parse_expr_ok("f(1, 2, 3)");
    }

    #[test]
    fn test_method_call() {
        parse_expr_ok("x.foo()");
        parse_expr_ok("x.foo(y)");
        parse_expr_ok("x.foo().bar()");
    }

    #[test]
    fn test_field_access() {
        parse_expr_ok("point.x");
        parse_expr_ok("a.b.c");
    }

    #[test]
    fn test_index() {
        parse_expr_ok("arr[0]");
        parse_expr_ok("matrix[i][j]");
    }

    #[test]
    fn test_try_operator() {
        parse_expr_ok("readFile()?");
        parse_expr_ok("parseJson(data)?");
        parse_expr_ok("foo()?.bar()?");
    }

    #[test]
    fn test_tuples() {
        parse_expr_ok("(1, 2)");
        parse_expr_ok("(a, b, c)");
    }

    #[test]
    fn test_lists() {
        parse_expr_ok("[]");
        parse_expr_ok("[1]");
        parse_expr_ok("[1, 2, 3]");
        parse_expr_ok("[head | tail]");
    }

    #[test]
    fn test_map_literals() {
        parse_expr_ok("%{}");
        parse_expr_ok("%{\"key\": value}");
        parse_expr_ok("%{\"a\": 1, \"b\": 2}");
    }

    #[test]
    fn test_set_literals() {
        parse_expr_ok("#{}");
        parse_expr_ok("#{1}");
        parse_expr_ok("#{1, 2, 3}");
    }

    #[test]
    fn test_records() {
        parse_expr_ok("Point(3.0, 4.0)");
        parse_expr_ok("Point(x: 3.0, y: 4.0)");
    }

    #[test]
    fn test_lambda() {
        parse_expr_ok("x => x * 2");
        parse_expr_ok("(a, b) => a + b");
        parse_expr_ok("() => 42");
    }

    #[test]
    fn test_if_expr() {
        parse_expr_ok("if true then 1 else 2");
        parse_expr_ok("if a > b then a else b");
    }

    #[test]
    fn test_match_expr() {
        parse_expr_ok("match x 0 -> zero 1 -> one _ -> many end");
        parse_expr_ok("match opt Some(v) -> v None -> default end");
    }

    #[test]
    fn test_match_with_guard() {
        parse_expr_ok("match n n when n > 0 -> positive _ -> non_positive end");
    }

    #[test]
    fn test_match_brace_style() {
        parse_expr_ok("match x { Some(n) -> n, None -> 0 }");
        parse_expr_ok("match opt { Some(v) -> v, None -> default }");
    }

    #[test]
    fn test_match_comma_separators() {
        // Comma between arms (single line)
        parse_expr_ok("match x { A -> 1, B -> 2, C -> 3 }");
        // Trailing comma allowed
        parse_expr_ok("match x { A -> 1, B -> 2, }");
        // Mix of newlines and commas
        parse_ok("main() = match x { A -> 1, B -> 2 }");
    }

    #[test]
    fn test_lambda_with_match_comma() {
        // Lambda returning match with comma-separated arms
        parse_expr_ok("x => match x { Some(n) -> n, None -> 0 }");
        // Lambda assignment in block with match
        parse_ok("main() = { f = x => match x { A -> 1, B -> 2 }; f(A) }");
    }

    #[test]
    fn test_block_expr() {
        parse_expr_ok("{ x }");
        parse_expr_ok("{ x = 1, x + 1 }");
    }

    #[test]
    fn test_grouped() {
        parse_expr_ok("(1 + 2) * 3");
        parse_expr_ok("((a))");
    }
}

// =============================================================================
// Control Flow Tests
// =============================================================================

mod control_flow {
    use super::*;

    #[test]
    fn test_try_catch() {
        parse_expr_ok("try riskyOp() catch Error(e) -> handleError(e) end");
    }

    #[test]
    fn test_try_catch_finally() {
        parse_expr_ok("try openFile() catch Error(e) -> logError(e) finally cleanup() end");
    }

    #[test]
    fn test_do_block() {
        parse_expr_ok("do x = getLine() print(x) end");
    }
}

// =============================================================================
// Concurrency Tests
// =============================================================================

mod concurrency {
    use super::*;

    #[test]
    fn test_spawn() {
        parse_expr_ok("spawn(worker)");
    }

    #[test]
    fn test_spawn_link() {
        parse_expr_ok("spawn_link(worker)");
    }

    #[test]
    fn test_spawn_monitor() {
        parse_expr_ok("spawn_monitor(worker)");
    }

    #[test]
    fn test_spawn_lambda() {
        parse_expr_ok("spawn(() => println(x))");
    }

    #[test]
    fn test_send() {
        parse_expr_ok("pid <- msg");
        parse_expr_ok("pid <- 42");
        parse_expr_ok("pid <- Inc(self())");
    }

    #[test]
    fn test_receive_simple() {
        parse_expr_ok("receive msg -> handle(msg) end");
    }

    #[test]
    fn test_receive_patterns() {
        parse_expr_ok("receive Inc(sender) -> incr() Dec(sender) -> decr() end");
    }

    #[test]
    fn test_receive_with_timeout() {
        parse_expr_ok("receive msg -> handle(msg) after 5000 -> timeout() end");
    }
}

// =============================================================================
// Pattern Tests
// =============================================================================

mod patterns {
    use super::*;

    #[test]
    fn test_wildcard() {
        parse_ok("_ = 5");
    }

    #[test]
    fn test_simple_binding() {
        parse_ok("x = 5");
    }

    #[test]
    fn test_mutable_binding() {
        parse_ok("var x = 5");
    }

    #[test]
    fn test_unit_pattern() {
        parse_ok("() = getUnit()");
    }

    #[test]
    fn test_literal_patterns() {
        parse_ok("0 = getZero()");
        parse_ok("true = getTrue()");
        parse_ok("\"hello\" = getString()");
        parse_ok("'a' = getChar()");
    }

    #[test]
    fn test_tuple_destructure() {
        parse_ok("(a, b) = getTuple()");
        parse_ok("(x, y, z) = getTriple()");
    }

    #[test]
    fn test_record_destructure() {
        parse_ok("{x, y} = point");
    }

    #[test]
    fn test_record_destructure_named() {
        parse_ok("{name: n, age: a} = person");
    }

    #[test]
    fn test_list_destructure() {
        parse_ok("[h | t] = items");
        parse_ok("[a, b | rest] = items");
    }

    #[test]
    fn test_empty_list_pattern() {
        parse_ok("[] = emptyList");
    }

    #[test]
    fn test_variant_destructure() {
        parse_ok("Some(value) = maybeValue");
        parse_ok("None = nothing");
    }

    #[test]
    fn test_variant_named_fields() {
        parse_ok("Circle{radius: r} = shape");
    }

    #[test]
    fn test_pin_pattern() {
        parse_ok("^expected = getValue()");
    }

    #[test]
    fn test_or_pattern() {
        parse_expr_ok("match x 0 | 1 | 2 -> small _ -> large end");
    }

    #[test]
    fn test_nested_patterns() {
        parse_ok("Some((a, b)) = nestedOpt");
        parse_ok("(Some(x), None) = pair");
    }
}

// =============================================================================
// Function Definition Tests
// =============================================================================

mod functions {
    use super::*;

    #[test]
    fn test_simple_function() {
        parse_ok("add(a, b) = a + b");
    }

    #[test]
    fn test_function_with_literals() {
        parse_ok("fib(0) = 0");
        parse_ok("fib(1) = 1");
    }

    #[test]
    fn test_function_with_guard() {
        parse_ok("abs(n) when n >= 0 = n");
    }

    #[test]
    fn test_function_with_return_type() {
        parse_ok("length(p: Point) -> Float = sqrt(p.x)");
    }

    #[test]
    fn test_function_pattern_matching() {
        parse_ok("unwrap(Some(x)) = x");
        parse_ok("unwrap(None) = error(msg)");
    }

    #[test]
    fn test_list_pattern_function() {
        parse_ok("sum([]) = 0");
        parse_ok("sum([h | t]) = h + sum(t)");
    }

    #[test]
    fn test_unit_function() {
        parse_ok("noop() = ()");
    }
}

// =============================================================================
// Type Definition Tests
// =============================================================================

mod types {
    use super::*;

    #[test]
    fn test_record_type() {
        parse_ok("type Point = {x: Float, y: Float}");
    }

    #[test]
    fn test_mutable_record_type() {
        parse_ok("var type Buffer = {data: Int, len: Int}");
    }

    #[test]
    fn test_variant_type() {
        parse_ok("type Option[T] = None | Some(T)");
    }

    #[test]
    fn test_variant_with_pipe() {
        parse_ok("type Option[T] = | None | Some(T)");
    }

    #[test]
    fn test_result_type() {
        parse_ok("type Result[T, E] = Ok(T) | Err(E)");
    }

    #[test]
    fn test_recursive_type() {
        parse_ok("type List[T] = Nil | Cons(T, List[T])");
    }

    #[test]
    fn test_variant_named_fields() {
        parse_ok("type Shape = Circle{radius: Float} | Rectangle{width: Float, height: Float}");
    }

    #[test]
    fn test_constrained_type_params() {
        parse_ok("type Set[T: Eq + Hash] = {items: List[T]}");
    }

    #[test]
    fn test_private_field() {
        parse_ok("type Account = {private balance: Float, name: String}");
    }

    #[test]
    fn test_type_alias() {
        parse_ok("type Handler = (Request) -> Response");
    }

    #[test]
    fn test_unit_type() {
        parse_ok("type Unit = ()");
    }
}

// =============================================================================
// Trait Tests
// =============================================================================

mod traits {
    use super::*;

    #[test]
    fn test_simple_trait() {
        parse_ok("trait Show show(self) -> String end");
    }

    #[test]
    fn test_trait_with_default() {
        parse_ok("trait Eq ==(self, other) -> Bool !=(self, other) = !(self == other) end");
    }

    #[test]
    fn test_trait_implementation() {
        parse_ok("Point: Show show(self) = self.x end");
    }
}

// =============================================================================
// Module Tests
// =============================================================================

mod modules {
    use super::*;

    #[test]
    fn test_simple_module() {
        parse_ok("module Geometry type Point = {x: Float, y: Float} end");
    }

    #[test]
    fn test_use_statement() {
        parse_ok("use Geometry.{Point, distance}");
    }

    #[test]
    fn test_use_all() {
        parse_ok("use Geometry.*");
    }
}

// =============================================================================
// Test Definition Tests
// =============================================================================

mod test_defs {
    use super::*;

    #[test]
    fn test_simple_test() {
        parse_ok("test \"addition works\" = { check(1 + 1 == 2) }");
    }

    #[test]
    fn test_test_with_block() {
        parse_ok("test \"complex test\" = { x = setup(), result = compute(x), check(result) }");
    }
}

// =============================================================================
// Extern Declaration Tests
// =============================================================================

mod extern_defs {
    use super::*;

    #[test]
    fn test_extern_type() {
        parse_ok("extern type File");
    }

    #[test]
    fn test_extern_function() {
        parse_ok("extern open(path: String) -> File from \"libc\"");
    }
}

// =============================================================================
// Integration Tests - Complete Examples
// =============================================================================

mod integration {
    use super::*;

    #[test]
    fn test_counter_server() {
        let src = r#"
            counter() = loop(0)
            loop(state) = receive
                Inc(sender) -> { sender <- Value(state + 1), loop(state + 1) }
                Get(sender) -> { sender <- Value(state), loop(state) }
            end
        "#;
        parse_ok(src);
    }

    #[test]
    fn test_list_map() {
        let src = r#"
            map(f, []) = []
            map(f, [h | t]) = cons(f(h), map(f, t))
        "#;
        parse_ok(src);
    }

    #[test]
    fn test_complex_pattern_matching() {
        let src = r#"
            process(0, _) = zero
            process(_, 0) = zero
            process(a, b) when a == b = equal
            process(a, b) = different
        "#;
        parse_ok(src);
    }

    #[test]
    fn test_method_chaining() {
        parse_expr_ok("items.filter(x => x > 0).map(square).take(10)");
    }

    #[test]
    fn test_pipeline_processing() {
        parse_expr_ok("data |> parse() |> validate() |> transform()");
    }

    #[test]
    fn test_error_handling() {
        let src = r#"
            safeRead(path) = try readFile(path)? catch NotFound(p) -> handleNotFound(p) end
        "#;
        parse_ok(src);
    }

    #[test]
    fn test_complete_type_system() {
        let src = r#"
            type Option[T] =
                | None
                | Some(T)

            type Result[T, E] =
                | Ok(T)
                | Err(E)

            type Point = {x: Float, y: Float}

            type Shape =
                | Circle{radius: Float}
                | Rectangle{width: Float, height: Float}
        "#;
        parse_ok(src);
    }
}

// =============================================================================
// Signature Tests
// =============================================================================

mod signatures {
    use super::*;
    use nostos_syntax::ast::Item;

    fn get_fn_signature(source: &str) -> String {
        let module = parse_ok(source);
        match &module.items[0] {
            Item::FnDef(fn_def) => fn_def.signature(),
            _ => panic!("Expected function definition"),
        }
    }

    #[test]
    fn test_untyped_single_param() {
        // Single untyped param should get type variable 'a', return 'a' (identity)
        let sig = get_fn_signature("identity(x) = x");
        assert_eq!(sig, "a -> a");
    }

    #[test]
    fn test_untyped_two_params_unified() {
        // Two params used in arithmetic should be unified to same type variable
        let sig = get_fn_signature("add(x, y) = x + y");
        assert_eq!(sig, "a -> a -> a");
    }

    #[test]
    fn test_typed_params() {
        // Typed params should show their types
        let sig = get_fn_signature("add(x: Int, y: Int) -> Int = x + y");
        assert_eq!(sig, "Int -> Int -> Int");
    }

    #[test]
    fn test_mixed_typed_untyped() {
        // Mix of typed and untyped - untyped param unified with return through +
        let sig = get_fn_signature("foo(x: Int, y) = x + y");
        assert_eq!(sig, "Int -> a -> a");
    }

    #[test]
    fn test_no_params() {
        // No params, just return type variable
        let sig = get_fn_signature("constant() = 42");
        assert_eq!(sig, "a");
    }

    #[test]
    fn test_no_params_with_return_type() {
        // No params with explicit return type
        let sig = get_fn_signature("constant() -> Int = 42");
        assert_eq!(sig, "Int");
    }

    #[test]
    fn test_three_params_unified() {
        // Three params all used in arithmetic chain
        let sig = get_fn_signature("sum3(x, y, z) = x + y + z");
        assert_eq!(sig, "a -> a -> a -> a");
    }

    #[test]
    fn test_params_not_unified() {
        // Two params NOT used together - should get different type variables
        let sig = get_fn_signature("first(x, y) = x");
        assert_eq!(sig, "a -> b -> a");
    }

    #[test]
    fn test_comparison_unifies() {
        // Comparison also unifies types
        let sig = get_fn_signature("max(x, y) = if x > y then x else y");
        assert_eq!(sig, "a -> a -> a");
    }

    #[test]
    fn test_multiplication() {
        // Multiplication should also unify
        let sig = get_fn_signature("mul(x, y) = x * y");
        assert_eq!(sig, "a -> a -> a");
    }

    #[test]
    fn test_custom_type_no_inference() {
        // Custom type - without type annotations, we can't infer the type
        // Field access doesn't tell us the type, so params stay separate
        let sig = get_fn_signature("swap(p) = Point(p.1, p.0)");
        // p is used but not in arithmetic with other params, so just 'a -> b'
        assert_eq!(sig, "a -> b");
    }

    #[test]
    fn test_custom_type_with_annotation() {
        // With type annotation, we get the concrete type
        let sig = get_fn_signature("swap(p: Point) -> Point = Point(p.1, p.0)");
        assert_eq!(sig, "Point -> Point");
    }

    #[test]
    fn test_two_params_field_access() {
        // Two params with field access used in arithmetic - should unify
        let sig = get_fn_signature("addX(p1, p2) = p1.x + p2.x");
        // p1 and p2 are used together in +, so they should unify
        assert_eq!(sig, "a -> a -> a");
    }
}

// =============================================================================
// Module-level Mutable Variable (mvar) Tests
// =============================================================================

mod mvars {
    use super::*;
    use nostos_syntax::ast::{Item, Visibility};

    #[test]
    fn test_mvar_basic() {
        let module = parse_ok("mvar counter: Int = 0");
        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            Item::MvarDef(def) => {
                assert_eq!(def.name.node, "counter");
                assert_eq!(def.visibility, Visibility::Private);
            }
            _ => panic!("Expected MvarDef"),
        }
    }

    #[test]
    fn test_mvar_public() {
        let module = parse_ok("pub mvar counter: Int = 0");
        match &module.items[0] {
            Item::MvarDef(def) => {
                assert_eq!(def.name.node, "counter");
                assert_eq!(def.visibility, Visibility::Public);
            }
            _ => panic!("Expected MvarDef"),
        }
    }

    #[test]
    fn test_mvar_with_expr() {
        let module = parse_ok("mvar total: Int = 1 + 2 + 3");
        match &module.items[0] {
            Item::MvarDef(def) => {
                assert_eq!(def.name.node, "total");
            }
            _ => panic!("Expected MvarDef"),
        }
    }

    #[test]
    fn test_mvar_list_type() {
        let module = parse_ok("mvar items: List[String] = []");
        match &module.items[0] {
            Item::MvarDef(def) => {
                assert_eq!(def.name.node, "items");
            }
            _ => panic!("Expected MvarDef"),
        }
    }

    #[test]
    fn test_multiple_mvars() {
        let module = parse_ok("mvar a: Int = 0\nmvar b: String = \"hello\"");
        assert_eq!(module.items.len(), 2);
        match (&module.items[0], &module.items[1]) {
            (Item::MvarDef(a), Item::MvarDef(b)) => {
                assert_eq!(a.name.node, "a");
                assert_eq!(b.name.node, "b");
            }
            _ => panic!("Expected two MvarDefs"),
        }
    }
}
