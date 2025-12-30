#!/usr/bin/env python3
# NumPy linear algebra benchmark - comparison with Nostos nalgebra
# Tests same operations as the Nostos benchmark
# Uses internal timing (excludes startup/import time)

import numpy as np
import time

def gen_vec(n, seed):
    """Generate deterministic pseudo-random vector (matches Nostos vecFromSeed)"""
    result = []
    for i in range(n):
        x = (i * 1103515245 + seed) % 2147483647
        result.append((x % 1000000) / 1000000.0)
    return np.array(result, dtype=np.float64)

def gen_matrix(rows, cols, seed):
    """Generate matrix data (matches Nostos matFromSeed)"""
    result = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row_seed = seed + r * 1000
            x = (c * 1103515245 + row_seed) % 2147483647
            row.append((x % 1000000) / 1000000.0)
        result.append(row)
    return np.array(result, dtype=np.float64)

def main():
    # Configuration - same as Nostos benchmark
    vec_size = 10000
    vec_iters = 10000
    mat_size = 200
    mat_iters = 100

    # Create data (not timed)
    v1 = gen_vec(vec_size, 42)
    v2 = gen_vec(vec_size, 123)
    m = gen_matrix(mat_size, mat_size, 42)

    # === START TIMING (computation only) ===
    start_time = time.time()

    # Benchmark 1: Vector dot product
    r1 = 0.0
    for _ in range(vec_iters):
        r1 += np.dot(v1, v2)

    # Benchmark 2: Vector norm
    r2 = 0.0
    for _ in range(vec_iters):
        r2 += np.linalg.norm(v1)

    # Benchmark 3: Vector sum
    r3 = 0.0
    for _ in range(vec_iters):
        r3 += np.sum(v1)

    # Benchmark 4: Matrix trace
    r4 = 0.0
    for _ in range(mat_iters):
        r4 += np.trace(m)

    # Benchmark 5: Matrix determinant
    r5 = 0.0
    for _ in range(mat_iters):
        r5 += np.linalg.det(m)

    # === END TIMING ===
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    result = r1 + r2 + r3 + r4 + r5

    # Output format for shell script parsing
    print(f"TIME_MS: {elapsed_ms:.0f}")
    print(f"{result:.6f}")

if __name__ == "__main__":
    main()
