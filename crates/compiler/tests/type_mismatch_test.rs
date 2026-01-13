// Test type mismatch detection in function calls
// Run with: cargo test --release -p nostos-compiler --test type_mismatch_test -- --nocapture

use nostos_compiler::compile::compile_module;
use nostos_syntax::parse;

#[test]
fn test_addone_type_mismatch() {
    let code = r#"
addOne(x: Int) -> Int = x + 1
main() = addOne("hello")
"#;

    let (module_opt, errors) = parse(code);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);

    let module = module_opt.expect("Failed to parse module");

    // Try to compile - this should fail with a type mismatch error
    let result = compile_module(&module, code);

    // Should have an error
    match result {
        Ok(_) => {
            panic!("Expected type mismatch error, got Ok");
        }
        Err(error) => {
            let error_msg = format!("{:?}", error);
            println!("Error: {}", error_msg);

            // Check that the error mentions type mismatch
            assert!(
                error_msg.contains("Int") || error_msg.contains("String") || error_msg.contains("unify") || error_msg.contains("mismatch"),
                "Expected error about type mismatch, got: {}", error_msg
            );
        }
    }
}
