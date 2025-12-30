// Rust nalgebra benchmark - comparison with Nostos and NumPy
// Same operations as the other benchmarks
// Uses internal timing (excludes data creation)

use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Generate deterministic pseudo-random vector (matches Nostos vecFromSeed)
fn gen_vec(n: usize, seed: i64) -> DVector<f64> {
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = ((i as i64) * 1103515245 + seed) % 2147483647;
            ((x % 1000000) as f64) / 1000000.0
        })
        .collect();
    DVector::from_vec(data)
}

/// Generate matrix data (matches Nostos matFromSeed)
fn gen_matrix(rows: usize, cols: usize, seed: i64) -> DMatrix<f64> {
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

fn main() {
    // Configuration - same as Nostos benchmark
    let vec_size = 10000;
    let vec_iters = 10000;
    let mat_size = 200;
    let mat_iters = 100;

    // Create data (not timed)
    let v1 = gen_vec(vec_size, 42);
    let v2 = gen_vec(vec_size, 123);
    let m = gen_matrix(mat_size, mat_size, 42);

    // === START TIMING (computation only) ===
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

    // === END TIMING ===
    let elapsed = start.elapsed();

    let result = r1 + r2 + r3 + r4 + r5;

    // Output format for shell script parsing
    println!("TIME_MS: {}", elapsed.as_millis());
    println!("{}", result);
}
