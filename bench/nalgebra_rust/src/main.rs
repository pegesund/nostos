// Rust nalgebra benchmark - comparison with Nostos and NumPy
// Same operations as the other benchmarks

use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Generate deterministic pseudo-random list (matches Nostos genList)
fn gen_list(n: usize, seed: i64) -> DVector<f64> {
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = ((i as i64) * 1103515245 + seed) % 2147483647;
            ((x % 1000000) as f64) / 1000000.0
        })
        .collect();
    DVector::from_vec(data)
}

/// Generate matrix data (matches Nostos genMatrixData)
fn gen_matrix_data(rows: usize, cols: usize, seed: i64) -> DMatrix<f64> {
    let data: Vec<f64> = (0..rows)
        .flat_map(|r| {
            (0..cols).map(move |c| {
                let row_seed = seed + (r as i64) * 1000;
                let x = ((c as i64) * 1103515245 + row_seed) % 2147483647;
                ((x % 1000000) as f64) / 1000000.0
            })
        })
        .collect();
    DMatrix::from_row_slice(rows, cols, &data)
}

/// Vector dot product benchmark
fn bench_vector_dot(iterations: usize, size: usize) -> f64 {
    let v1 = gen_list(size, 42);
    let v2 = gen_list(size, 123);
    let mut total = 0.0;
    for _ in 0..iterations {
        total += v1.dot(&v2);
    }
    total
}

/// Vector norm benchmark
fn bench_vector_norm(iterations: usize, size: usize) -> f64 {
    let v = gen_list(size, 42);
    let mut total = 0.0;
    for _ in 0..iterations {
        total += v.norm();
    }
    total
}

/// Vector sum benchmark
fn bench_vector_sum(iterations: usize, size: usize) -> f64 {
    let v = gen_list(size, 42);
    let mut total = 0.0;
    for _ in 0..iterations {
        total += v.sum();
    }
    total
}

/// Matrix trace benchmark
fn bench_matrix_trace(iterations: usize, size: usize) -> f64 {
    let m = gen_matrix_data(size, size, 42);
    let mut total = 0.0;
    for _ in 0..iterations {
        total += m.trace();
    }
    total
}

/// Matrix determinant benchmark
fn bench_matrix_determinant(iterations: usize, size: usize) -> f64 {
    let m = gen_matrix_data(size, size, 42);
    let mut total = 0.0;
    for _ in 0..iterations {
        total += m.determinant();
    }
    total
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "optimized" {
        // Optimized benchmark - matches bench_optimized.nos
        let vec_size = 10000;
        let vec_iters = 10000;
        let mat_size = 200;
        let mat_iters = 100;

        // Pre-generate data (not timed)
        let v1 = gen_list(vec_size, 42);
        let v2 = gen_list(vec_size, 123);
        let m = gen_matrix_data(mat_size, mat_size, 42);

        // Start timing
        let start = Instant::now();

        // Benchmark 1: Vector dot product
        let mut r1 = 0.0;
        for _ in 0..vec_iters {
            r1 += v1.dot(&v2);
        }

        // Benchmark 2: Vector norm
        let mut r2 = 0.0;
        for _ in 0..vec_iters {
            r2 += v1.norm();
        }

        // Benchmark 3: Vector sum
        let mut r3 = 0.0;
        for _ in 0..vec_iters {
            r3 += v1.sum();
        }

        // Benchmark 4: Matrix trace
        let mut r4 = 0.0;
        for _ in 0..mat_iters {
            r4 += m.trace();
        }

        // Benchmark 5: Matrix determinant
        let mut r5 = 0.0;
        for _ in 0..mat_iters {
            r5 += m.determinant();
        }

        let elapsed = start.elapsed();
        let result = r1 + r2 + r3 + r4 + r5;
        println!("Time (FFI only): {}ms", elapsed.as_millis());
        println!("{}", result);
    } else if args.len() > 1 && args[1] == "test" {
        // Test individual operation speeds
        let n = 10000;
        let iters = 50000;

        let v1: DVector<f64> = DVector::from_iterator(n, (0..n).map(|x| x as f64));
        let v2: DVector<f64> = DVector::from_iterator(n, (0..n).map(|x| (x * 2) as f64));
        let m: DMatrix<f64> = DMatrix::identity(100, 100);

        // Test sum
        let start = Instant::now();
        let mut acc1 = 0.0;
        for _ in 0..iters {
            acc1 += v1.sum();
        }
        let t1 = start.elapsed();
        println!("sum {}x: {:?} ({:.1}μs/call)", iters, t1, t1.as_nanos() as f64 / iters as f64 / 1000.0);

        // Test norm
        let start = Instant::now();
        let mut acc2 = 0.0;
        for _ in 0..iters {
            acc2 += v1.norm();
        }
        let t2 = start.elapsed();
        println!("norm {}x: {:?} ({:.1}μs/call)", iters, t2, t2.as_nanos() as f64 / iters as f64 / 1000.0);

        // Test dot
        let start = Instant::now();
        let mut acc3 = 0.0;
        for _ in 0..iters {
            acc3 += v1.dot(&v2);
        }
        let t3 = start.elapsed();
        println!("dot {}x: {:?} ({:.1}μs/call)", iters, t3, t3.as_nanos() as f64 / iters as f64 / 1000.0);

        // Test trace
        let start = Instant::now();
        let mut acc4 = 0.0;
        for _ in 0..iters {
            acc4 += m.trace();
        }
        let t4 = start.elapsed();
        println!("trace {}x: {:?} ({:.1}μs/call)", iters, t4, t4.as_nanos() as f64 / iters as f64 / 1000.0);

        println!("Checksums: {}, {}, {}, {}", acc1, acc2, acc3, acc4);
    } else if args.len() > 1 && args[1] == "simple" {
        // Simple benchmark matching bench_simple.nos
        let v1: DVector<f64> = DVector::from_iterator(10000, (0..10000).map(|x| x as f64));
        let v2: DVector<f64> = DVector::from_iterator(10000, (0..10000).map(|x| (x * 2) as f64));
        let m: DMatrix<f64> = DMatrix::from_fn(100, 100, |r, c| (r * 100 + c) as f64);

        let mut r1 = 0.0;
        for _ in 0..100000 {
            r1 += v1.dot(&v2);
        }

        let mut r2 = 0.0;
        for _ in 0..100000 {
            r2 += v1.sum();
        }

        let mut r3 = 0.0;
        for _ in 0..10000 {
            r3 += m.trace();
        }

        let mut r4 = 0.0;
        for _ in 0..1000 {
            r4 += m.determinant();
        }

        println!("{}", r1 + r2 + r3 + r4);
    } else {
        // Original benchmark
        let vec_size = 10000;
        let vec_iters = 10000;
        let r1 = bench_vector_dot(vec_iters, vec_size);
        let r2 = bench_vector_norm(vec_iters, vec_size);
        let r3 = bench_vector_sum(vec_iters, vec_size);

        let mat_size = 200;
        let mat_iters = 100;
        let r4 = bench_matrix_trace(mat_iters, mat_size);
        let r5 = bench_matrix_determinant(mat_iters, mat_size);

        println!("{}", r1 + r2 + r3 + r4 + r5);
    }
}
