// REPL Test Framework
//
// Systematically tests that all language constructs work in the REPL environment.
// This ensures parity between file execution and REPL/TUI execution.
//
// Test format (*.repl files):
//   > input expression
//   expected output
//
//   > multi-line input \
//   > continues here
//   expected output
//
//   >! input that should error
//   error: expected error substring
//
//   # comments start with #
//   ## section headers

use super::*;

/// Result of running a single REPL test
#[derive(Debug)]
pub struct ReplTestResult {
    pub input: String,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
    pub line_number: usize,
}

/// Run a sequence of REPL inputs and check outputs
pub fn run_repl_test_sequence(inputs: &[(String, String, bool)]) -> Vec<ReplTestResult> {
    let config = ReplConfig { enable_jit: false, num_threads: 1 };
    let mut engine = ReplEngine::new(config);
    engine.load_stdlib().ok();

    let mut results = Vec::new();

    for (i, (input, expected, expect_error)) in inputs.iter().enumerate() {
        let actual = match engine.eval(input) {
            Ok(output) => {
                if *expect_error {
                    format!("ok: {}", output.trim())
                } else {
                    output.trim().to_string()
                }
            }
            Err(e) => {
                if *expect_error {
                    format!("error: {}", e)
                } else {
                    format!("error: {}", e)
                }
            }
        };

        let passed = if *expect_error {
            actual.starts_with("error:") && actual.contains(expected)
        } else {
            actual == *expected || actual.contains(expected)
        };

        results.push(ReplTestResult {
            input: input.clone(),
            expected: expected.clone(),
            actual,
            passed,
            line_number: i + 1,
        });
    }

    results
}

/// Test that `check_module_compiles` catches the same errors as full compilation
pub fn test_check_module_compiles(code: &str, expect_error: Option<&str>) -> Result<(), String> {
    let mut engine = ReplEngine::new(ReplConfig::default());
    engine.load_stdlib().ok(); // Must load stdlib for List methods, Some/None, etc.
    let result = engine.check_module_compiles("", code);

    match (result, expect_error) {
        (Ok(()), None) => Ok(()),
        (Ok(()), Some(expected)) => Err(format!(
            "Expected error containing '{}', but code compiled successfully",
            expected
        )),
        (Err(e), None) => Err(format!(
            "Expected successful compilation, but got error: {}",
            e
        )),
        (Err(e), Some(expected)) => {
            if e.contains(expected) {
                Ok(())
            } else {
                Err(format!(
                    "Expected error containing '{}', but got: {}",
                    expected, e
                ))
            }
        }
    }
}

#[cfg(test)]
mod repl_language_tests {
    use super::*;

    // ========================================
    // Basic Expressions
    // ========================================

    #[test]
    fn test_repl_arithmetic() {
        let tests = vec![
            ("1 + 2".to_string(), "3".to_string(), false),
            ("10 * 5".to_string(), "50".to_string(), false),
            ("100 / 4".to_string(), "25".to_string(), false),
            ("17 % 5".to_string(), "2".to_string(), false),
            ("2 + 3 * 4".to_string(), "14".to_string(), false),
            ("(2 + 3) * 4".to_string(), "20".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_strings() {
        let tests = vec![
            (r#""hello""#.to_string(), "hello".to_string(), false),
            (r#""hello" ++ " world""#.to_string(), "hello world".to_string(), false),
            (r#"String.length("test")"#.to_string(), "4".to_string(), false),
            (r#""test".length()"#.to_string(), "4".to_string(), false),
            (r#""hello".replace("l", "x")"#.to_string(), "hexlo".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_booleans() {
        let tests = vec![
            ("true".to_string(), "true".to_string(), false),
            ("false".to_string(), "false".to_string(), false),
            ("true && false".to_string(), "false".to_string(), false),
            ("true || false".to_string(), "true".to_string(), false),
            ("!true".to_string(), "false".to_string(), false),
            ("1 < 2".to_string(), "true".to_string(), false),
            ("1 > 2".to_string(), "false".to_string(), false),
            ("1 == 1".to_string(), "true".to_string(), false),
            ("1 != 2".to_string(), "true".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Variable Binding and Persistence
    // ========================================

    // FIXED: Variable assignments now properly evaluate and display their values
    #[test]
    fn test_repl_variable_persistence() {
        let tests = vec![
            ("x = 5".to_string(), "5".to_string(), false),
            ("x".to_string(), "5".to_string(), false),
            ("x + 1".to_string(), "6".to_string(), false),
            ("y = x * 2".to_string(), "10".to_string(), false),
            ("x + y".to_string(), "15".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL variable persistence failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_variable_shadowing() {
        let tests = vec![
            ("x = 1".to_string(), "1".to_string(), false),
            ("x = 2".to_string(), "2".to_string(), false),
            ("x".to_string(), "2".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL variable shadowing failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Function Definitions
    // ========================================

    #[test]
    fn test_repl_function_definition() {
        let tests = vec![
            ("add(a, b) = a + b".to_string(), "".to_string(), false),
            ("add(1, 2)".to_string(), "3".to_string(), false),
            ("add(10, 20)".to_string(), "30".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            // Skip empty expected for function definitions
            if r.expected.is_empty() && r.actual.is_empty() {
                continue;
            }
            if !r.passed && !r.expected.is_empty() {
                panic!("REPL function test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_recursive_function() {
        let tests = vec![
            ("fact(n) = if n <= 1 then 1 else n * fact(n - 1)".to_string(), "".to_string(), false),
            ("fact(5)".to_string(), "120".to_string(), false),
            ("fact(0)".to_string(), "1".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if r.expected.is_empty() && r.actual.is_empty() {
                continue;
            }
            if !r.passed && !r.expected.is_empty() {
                panic!("REPL recursive function failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Collections
    // ========================================

    #[test]
    fn test_repl_lists() {
        let tests = vec![
            ("[1, 2, 3]".to_string(), "[1, 2, 3]".to_string(), false),
            ("[]".to_string(), "[]".to_string(), false),
            ("[1, 2, 3].length()".to_string(), "3".to_string(), false),
            ("[1, 2, 3][0]".to_string(), "1".to_string(), false),
            ("[1, 2, 3].map(x => x * 2)".to_string(), "[2, 4, 6]".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL list test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_maps() {
        let tests = vec![
            // Maps display as %{...N entries}
            (r#"%{"a": 1, "b": 2}"#.to_string(), "%{...2 entries}".to_string(), false),
            (r#"Map.get(%{"x": 42}, "x")"#.to_string(), "42".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL map test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_sets() {
        let tests = vec![
            // Sets display as #{...N items}
            ("#{1, 2, 3}".to_string(), "#{...3 items}".to_string(), false),
            ("Set.contains(#{1, 2, 3}, 2)".to_string(), "true".to_string(), false),
            ("Set.contains(#{1, 2, 3}, 5)".to_string(), "false".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL set test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Pattern Matching
    // ========================================

    #[test]
    fn test_repl_pattern_matching() {
        // Note: Nostos match uses newlines between arms, not semicolons
        let tests = vec![
            ("match 1 { 1 -> \"one\"\n _ -> \"other\" }".to_string(), "one".to_string(), false),
            ("match [1, 2, 3] { [h | _] -> h\n _ -> 0 }".to_string(), "1".to_string(), false),
            ("match Some(42) { Some(x) -> x\n None -> 0 }".to_string(), "42".to_string(), false),
            ("match None { Some(x) -> x\n None -> 0 }".to_string(), "0".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL pattern matching failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Closures
    // ========================================

    // FIXED: Closures assigned to variables now persist and can be called
    #[test]
    fn test_repl_closures() {
        let tests = vec![
            ("f = x => x + 1".to_string(), "".to_string(), false),
            ("f(5)".to_string(), "6".to_string(), false),
            ("[1, 2, 3].map(x => x * x)".to_string(), "[1, 4, 9]".to_string(), false),
            ("[1, 2, 3].filter(x => x > 1)".to_string(), "[2, 3]".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if r.expected.is_empty() {
                continue;
            }
            if !r.passed {
                panic!("REPL closure test failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // FIXED: Closures that capture REPL variables now work correctly
    #[test]
    fn test_repl_closure_capture() {
        let tests = vec![
            ("multiplier = 10".to_string(), "10".to_string(), false),
            ("scale = x => x * multiplier".to_string(), "".to_string(), false),
            ("scale(5)".to_string(), "50".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if r.expected.is_empty() {
                continue;
            }
            if !r.passed {
                panic!("REPL closure capture failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Error Recovery
    // ========================================

    #[test]
    fn test_repl_error_recovery() {
        // After an error, REPL should continue working
        let tests = vec![
            ("x = 5".to_string(), "5".to_string(), false),
            ("undefined_var".to_string(), "unknown variable".to_string(), true),
            ("x + 1".to_string(), "6".to_string(), false), // Should still work after error
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL error recovery failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // check_module_compiles Tests
    // ========================================

    #[test]
    fn test_check_compiles_valid_code() {
        assert!(test_check_module_compiles("main() = 42", None).is_ok());
        assert!(test_check_module_compiles("add(a, b) = a + b", None).is_ok());
        assert!(test_check_module_compiles("type Point = { x: Int, y: Int }", None).is_ok());
    }

    #[test]
    fn test_check_compiles_type_errors() {
        // Type mismatch
        assert!(test_check_module_compiles(
            r#"main() = 1 + "hello""#,
            Some("type")
        ).is_ok());

        // Unknown variable
        assert!(test_check_module_compiles(
            "main() = undefined_var",
            Some("unknown variable")
        ).is_ok());

        // Unknown function
        assert!(test_check_module_compiles(
            "main() = undefined_func()",
            Some("unknown")
        ).is_ok());
    }

    #[test]
    fn test_check_compiles_method_errors() {
        // Invalid method on Int
        assert!(test_check_module_compiles(
            "main() = 42.xxx()",
            Some("Int.xxx")
        ).is_ok());

        // Invalid method on String
        assert!(test_check_module_compiles(
            r#"main() = "hello".yyy()"#,
            Some("String.yyy")
        ).is_ok());
    }

    // ========================================
    // Type Definitions in REPL
    // ========================================

    #[test]
    fn test_repl_type_definition() {
        let tests = vec![
            ("type Point = { x: Int, y: Int }".to_string(), "".to_string(), false),
            ("p = Point(1, 2)".to_string(), "Point".to_string(), false),
            ("p.x".to_string(), "1".to_string(), false),
            ("p.y".to_string(), "2".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if r.expected.is_empty() {
                continue;
            }
            if !r.passed {
                panic!("REPL type definition failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_variant_type() {
        // Note: Nostos match uses newlines between arms, not semicolons
        let tests = vec![
            ("type Result = Ok(Int) | Err(String)".to_string(), "".to_string(), false),
            ("r = Ok(42)".to_string(), "Ok".to_string(), false),
            ("match r { Ok(x) -> x\n Err(_) -> 0 }".to_string(), "42".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if r.expected.is_empty() {
                continue;
            }
            if !r.passed {
                panic!("REPL variant type failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // ========================================
    // Edge Cases
    // ========================================

    #[test]
    fn test_repl_multiline_blocks() {
        // Test that blocks work in REPL
        let block = "{ x = 1; y = 2; x + y }";
        let tests = vec![(block.to_string(), "3".to_string(), false)];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL multiline block failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    #[test]
    fn test_repl_if_expressions() {
        let tests = vec![
            ("if true then 1 else 2".to_string(), "1".to_string(), false),
            ("if false then 1 else 2".to_string(), "2".to_string(), false),
            ("if 1 < 2 then \"yes\" else \"no\"".to_string(), "yes".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL if expression failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // Note: Division by zero causes an uncatchable VM panic - this is a VM design issue,
    // not REPL-specific. The try/catch mechanism works correctly for explicit throw().
    #[test]
    fn test_repl_try_catch() {
        let tests = vec![
            // Explicit throw can be caught
            (r#"try { throw("error") } catch { e -> e }"#.to_string(), "error".to_string(), false),
            // Nested try/catch works
            (r#"try { try { throw("inner") } catch { _ -> throw("outer") } } catch { e -> e }"#.to_string(), "outer".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("REPL try/catch failed:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }

    // FIXED: Division by zero is now a catchable exception
    #[test]
    fn test_repl_division_by_zero() {
        let tests = vec![
            ("try { 1 / 0 } catch { _ -> 0 }".to_string(), "0".to_string(), false),
            ("try { 10 % 0 } catch { _ -> -1 }".to_string(), "-1".to_string(), false),
        ];

        let results = run_repl_test_sequence(&tests);
        for r in &results {
            if !r.passed {
                panic!("Division by zero should be catchable:\n  Input: {}\n  Expected: {}\n  Actual: {}",
                    r.input, r.expected, r.actual);
            }
        }
    }
}

#[cfg(test)]
mod check_module_systematic_tests {
    use super::*;

    /// Generate tests from file test patterns
    /// This ensures check_module_compiles catches the same errors as full compilation

    #[test]
    fn test_check_all_arithmetic() {
        let cases = [
            "main() = 1 + 2",
            "main() = 10 * 5",
            "main() = 100 / 4",
            "main() = 17 % 5",
            "main() = -42",
        ];

        for code in &cases {
            let result = test_check_module_compiles(code, None);
            assert!(result.is_ok(), "Code should compile: {}\nError: {:?}", code, result);
        }
    }

    #[test]
    fn test_check_all_string_ops() {
        let cases = [
            r#"main() = "hello""#,
            r#"main() = "a" ++ "b""#,
            r#"main() = String.length("test")"#,
            r#"main() = "test".length()"#,
            r#"main() = String.toUpper("hello")"#,
        ];

        for code in &cases {
            let result = test_check_module_compiles(code, None);
            assert!(result.is_ok(), "Code should compile: {}\nError: {:?}", code, result);
        }
    }

    #[test]
    fn test_check_all_list_ops() {
        let cases = [
            "main() = [1, 2, 3]",
            "main() = [1, 2, 3].length()",
            "main() = [1, 2, 3][0]",
            "main() = [1, 2, 3].map(x => x * 2)",
            "main() = [1, 2, 3].filter(x => x > 1)",
            "main() = [1, 2, 3].fold(0, (acc, x) => acc + x)",
        ];

        for code in &cases {
            let result = test_check_module_compiles(code, None);
            assert!(result.is_ok(), "Code should compile: {}\nError: {:?}", code, result);
        }
    }

    #[test]
    fn test_check_all_pattern_matching() {
        // Note: Nostos match uses newlines between arms, not semicolons
        let cases = [
            "main() = match 1 { 1 -> true\n _ -> false }",
            "main() = match [1, 2] { [h | _] -> h\n _ -> 0 }",
            "main() = match Some(1) { Some(x) -> x\n None -> 0 }",
            "main() = match (1, 2) { (a, b) -> a + b }",
        ];

        for code in &cases {
            let result = test_check_module_compiles(code, None);
            assert!(result.is_ok(), "Code should compile: {}\nError: {:?}", code, result);
        }
    }

    #[test]
    fn test_check_all_closures() {
        let cases = [
            "main() = (x => x + 1)(5)",
            "main() = { f = x => x * 2\n f(3) }",
            "main() = [1, 2].map(x => x)",
            "main() = { n = 10\n f = x => x + n\n f(5) }",
        ];

        for code in &cases {
            let result = test_check_module_compiles(code, None);
            assert!(result.is_ok(), "Code should compile: {}\nError: {:?}", code, result);
        }
    }

    #[test]
    fn test_check_all_type_definitions() {
        // Note: Nostos match uses newlines between arms, not semicolons
        let cases = [
            "type Point = { x: Int, y: Int }\nmain() = Point(1, 2).x",
            "type Color = Red | Green | Blue\nmain() = match Red { Red -> 1\n _ -> 0 }",
            "type Maybe[T] = Just(T) | Nothing\nmain() = match Just(1) { Just(x) -> x\n Nothing -> 0 }",
        ];

        for code in &cases {
            let result = test_check_module_compiles(code, None);
            assert!(result.is_ok(), "Code should compile: {}\nError: {:?}", code, result);
        }
    }

    #[test]
    fn test_check_detects_type_errors() {
        let error_cases = [
            (r#"main() = 1 + "a""#, "type"),
            ("main() = undefined_var", "unknown variable"),
            ("main() = undefined_func()", "unknown"),
            ("main() = 42.nonexistent()", "Int.nonexistent"),
            (r#"main() = "str".badmethod()"#, "String.badmethod"),
        ];

        for (code, expected_err) in &error_cases {
            let result = test_check_module_compiles(code, Some(expected_err));
            assert!(result.is_ok(),
                "Code should produce error containing '{}': {}\nResult: {:?}",
                expected_err, code, result);
        }
    }
}
