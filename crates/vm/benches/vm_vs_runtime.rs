//! Benchmark comparing VM vs Runtime performance.
//!
//! Run with: cargo bench -p nostos-vm --bench vm_vs_runtime
//!
//! This uses the actual compiler to generate bytecode, ensuring
//! we test the same code path as real programs.

use std::time::Instant;

use nostos_compiler::compile::compile_module;
use nostos_syntax::parse;
use nostos_vm::gc::GcValue;
use nostos_vm::runtime::Runtime;
use nostos_vm::value::Value;
use nostos_vm::VM;

const FIB_SOURCE: &str = r#"
fib(n) = if n <= 1 then n else fib(n - 1) + fib(n - 2)
main() = fib(35)
"#;

fn bench_vm() -> (i64, std::time::Duration) {
    let (module, errors) = parse(FIB_SOURCE);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let module = module.unwrap();

    let compiler = compile_module(&module).expect("Compile failed");

    let mut vm = VM::new();
    for (name, func) in compiler.get_all_functions() {
        vm.functions.insert(name.clone(), func.clone());
    }

    let start = Instant::now();
    let result = vm.call("main", vec![]).unwrap();
    let elapsed = start.elapsed();

    let value = match result {
        Value::Int(v) => v,
        _ => panic!("Expected Int"),
    };

    (value, elapsed)
}

fn bench_runtime() -> (i64, std::time::Duration) {
    let (module, errors) = parse(FIB_SOURCE);
    assert!(errors.is_empty(), "Parse errors: {:?}", errors);
    let module = module.unwrap();

    let compiler = compile_module(&module).expect("Compile failed");

    let mut runtime = Runtime::new();
    for (name, func) in compiler.get_all_functions() {
        runtime.register_function(&name, func.clone());
    }

    // Get main function and spawn it
    let main_func = compiler.get_all_functions()
        .get("main")
        .expect("No main function")
        .clone();

    runtime.spawn_initial(main_func);

    let start = Instant::now();
    let result = runtime.run();
    let elapsed = start.elapsed();

    let result = result.expect("Runtime error");
    let value = match result {
        Some(GcValue::Int(v)) => v,
        other => {
            eprintln!("Expected Int, got {:?}", other);
            eprintln!("Compiler functions: {:?}", compiler.get_all_functions().keys().collect::<Vec<_>>());
            panic!("Wrong result type");
        }
    };

    (value, elapsed)
}

fn main() {
    println!("Benchmarking fib(35)...\n");

    // Warm up
    let _ = bench_vm();
    let _ = bench_runtime();

    // Actual benchmark
    let (vm_result, vm_time) = bench_vm();
    let (rt_result, rt_time) = bench_runtime();

    println!("VM:      {} in {:?}", vm_result, vm_time);
    println!("Runtime: {} in {:?}", rt_result, rt_time);
    println!();
    println!("Runtime overhead: {:.2}x", rt_time.as_secs_f64() / vm_time.as_secs_f64());
}
