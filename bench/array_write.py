#!/usr/bin/env python3
# Array write benchmark - heavy writing to typed array
# Fills an array with computed values and sums them
# Uses array.array('q') for int64 typed array (fair comparison with Nostos Int64Array)

import array

def fill_array(arr, size):
    for i in range(size):
        arr[i] = i * i + i * 3 + 7

def sum_array(arr, size):
    total = 0
    for i in range(size):
        total += arr[i]
    return total

def run_iteration(size):
    arr = array.array('q', [0] * size)  # 'q' = signed int64
    fill_array(arr, size)
    return sum_array(arr, size)

def benchmark(iterations, size):
    total = 0
    for _ in range(iterations):
        total += run_iteration(size)
    return total

size = 100000
iterations = 100
result = benchmark(iterations, size)
print(result)
