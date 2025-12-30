# REPL Environment Testing Guide

This document describes how to create and test REPL functionality using Rust unit tests instead of manual TUI testing.

## Why Unit Tests Instead of TUI Testing

The REPL TUI (Terminal UI) is difficult to test manually because:
- It requires interactive input
- It's hard to reproduce exact scenarios
- Debugging with println! doesn't work well

Instead, write Rust unit tests that call the same `ReplEngine` methods the TUI uses.

## Basic Pattern

```rust
#[test]
fn test_my_repl_scenario() {
    let mut engine = ReplEngine::new(ReplConfig::default());
    
    // Optionally load stdlib
    engine.load_stdlib().ok();
    
    // Evaluate expressions just like in the REPL
    match engine.eval("x = 42") {
        Ok(result) => println!("OK: {}", result),
        Err(e) => panic!("Failed: {}", e),
    }
    
    // Continue with more evaluations...
    match engine.eval("x + 1") {
        Ok(result) => {
            println!("OK: {}", result);
            assert_eq!(result.trim(), "43");
        }
        Err(e) => panic!("Failed: {}", e),
    }
}
```

## Testing with Extension Modules

To test extension module behavior without requiring actual native libraries:

```rust
#[test]
fn test_with_mock_extension() {
    let mut engine = ReplEngine::new(ReplConfig::default());
    engine.load_stdlib().ok();
    
    // Define a mock extension module in Nostos code
    let ext_source = r#"
# Mock extension type
pub type Vec = { data: List }

# Show trait for pretty printing
trait Show
    show(self) -> String
end

# Constructor
pub vec(data: List) -> Vec = Vec(data)

# Show implementation
Vec: Show
    show(self) -> String = "Vec[" ++ show(self.data) ++ "]"
end
"#;

    // Load the mock extension
    match engine.load_extension_module("testvec", ext_source, "<test>") {
        Ok(_) => println!("Extension loaded"),
        Err(e) => panic!("Failed to load extension: {}", e),
    }
    
    // Import the extension
    match engine.eval("use testvec.*") {
        Ok(result) => println!("Imported: {}", result),
        Err(e) => panic!("Failed to import: {}", e),
    }
    
    // Now test with the extension
    match engine.eval("v = vec([1, 2, 3])") {
        Ok(result) => println!("OK: {}", result),
        Err(e) => panic!("Failed: {}", e),
    }
    
    // Test show() dispatch
    match engine.eval("show(v)") {
        Ok(result) => {
            println!("show(v) = {}", result);
            assert!(result.contains("Vec["), "Expected Vec[ prefix");
        }
        Err(e) => panic!("show(v) failed: {}", e),
    }
}
```

## Test Location

All REPL tests are located in:
```
crates/repl/src/engine.rs
```

Test modules:
- `mod tests` - Basic REPL functionality tests
- `mod call_graph_tests` - Call graph analysis tests  
- `mod check_module_tests` - Module compilation checking tests
- `mod repl_state_tests` - REPL state management tests
- `mod nalgebra_debug_tests` - Extension module and operator tests

## Running Tests

Run all REPL tests:
```bash
cargo test --release -p nostos-repl -- --nocapture
```

Run a specific test:
```bash
cargo test --release -p nostos-repl test_my_scenario -- --nocapture
```

Run tests matching a pattern:
```bash
cargo test --release -p nostos-repl nalgebra -- --nocapture
```

## Key ReplEngine Methods

- `eval(input: &str)` - Evaluate an expression or statement (main entry point)
- `load_stdlib()` - Load the standard library
- `load_extension_module(name, source, filename)` - Load a mock extension module
- `var_bindings` - Access to current variable bindings (for debugging)
- `check_module_compiles(module_name, content)` - Check if code compiles without running

## Debugging Tips

1. **Print variable bindings:**
   ```rust
   println!("var_bindings: {:?}", engine.var_bindings.keys().collect::<Vec<_>>());
   ```

2. **Check inferred types:**
   ```rust
   if let Some(binding) = engine.var_bindings.get("v") {
       println!("v type: {:?}", binding.type_annotation);
   }
   ```

3. **Print wrapper code (for debugging eval):**
   Add debug output in `eval_expression_inner` or `define_var`:
   ```rust
   eprintln!("DEBUG wrapper: {}", wrapper);
   ```

## Example: Testing Show Trait Dispatch

```rust
#[test]
fn test_show_trait_dispatch() {
    let mut engine = ReplEngine::new(ReplConfig::default());
    engine.load_stdlib().ok();
    
    // Define type with Show trait
    let ext_source = r#"
pub type Point = { x: Int, y: Int }

trait Show
    show(self) -> String
end

Point: Show
    show(self) -> String = "(" ++ show(self.x) ++ ", " ++ show(self.y) ++ ")"
end

pub point(x: Int, y: Int) -> Point = Point(x, y)
"#;

    engine.load_extension_module("geometry", ext_source, "<test>").unwrap();
    engine.eval("use geometry.*").unwrap();
    
    // Create a point
    engine.eval("p = point(3, 4)").unwrap();
    
    // Test show() is dispatched correctly
    let result = engine.eval("show(p)").unwrap();
    assert_eq!(result.trim(), "\"(3, 4)\"");
    
    // Test UFCS: p.show()
    let result2 = engine.eval("p.show()").unwrap();
    assert_eq!(result2.trim(), "\"(3, 4)\"");
}
```
