#!/usr/bin/env python3
# NumPy linear algebra benchmark - comparison with Nostos nalgebra
# Tests same operations as the Nostos benchmark

import numpy as np

def gen_list(n, seed):
    """Generate deterministic pseudo-random list (matches Nostos genList)"""
    result = []
    for i in range(n):
        x = (i * 1103515245 + seed) % 2147483647
        result.append((x % 1000000) / 1000000.0)
    return np.array(result, dtype=np.float64)

def gen_matrix_data(rows, cols, seed):
    """Generate matrix data (matches Nostos genMatrixData)"""
    result = []
    for r in range(rows):
        result.append(gen_list(cols, seed + r * 1000))
    return np.array(result, dtype=np.float64)

def bench_vector_dot(iterations, size):
    """Vector dot product benchmark"""
    v1 = gen_list(size, 42)
    v2 = gen_list(size, 123)
    total = 0.0
    for _ in range(iterations):
        total += np.dot(v1, v2)
    return total

def bench_vector_norm(iterations, size):
    """Vector norm benchmark"""
    v = gen_list(size, 42)
    total = 0.0
    for _ in range(iterations):
        total += np.linalg.norm(v)
    return total

def bench_vector_sum(iterations, size):
    """Vector sum benchmark"""
    v = gen_list(size, 42)
    total = 0.0
    for _ in range(iterations):
        total += np.sum(v)
    return total

def bench_matrix_trace(iterations, size):
    """Matrix trace benchmark"""
    m = gen_matrix_data(size, size, 42)
    total = 0.0
    for _ in range(iterations):
        total += np.trace(m)
    return total

def bench_matrix_determinant(iterations, size):
    """Matrix determinant benchmark"""
    m = gen_matrix_data(size, size, 42)
    total = 0.0
    for _ in range(iterations):
        total += np.linalg.det(m)
    return total

def main():
    # Configuration - same as Nostos benchmark

    # Vector operations
    vec_size = 10000
    vec_iters = 10000
    r1 = bench_vector_dot(vec_iters, vec_size)
    r2 = bench_vector_norm(vec_iters, vec_size)
    r3 = bench_vector_sum(vec_iters, vec_size)

    # Matrix operations
    mat_size = 200
    mat_iters = 100
    r4 = bench_matrix_trace(mat_iters, mat_size)
    r5 = bench_matrix_determinant(mat_iters, mat_size)

    # Print checksum for verification
    print(f"{r1 + r2 + r3 + r4 + r5:.6f}")

if __name__ == "__main__":
    main()
