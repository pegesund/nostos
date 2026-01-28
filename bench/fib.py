#!/usr/bin/env python3
# Fibonacci benchmark
import time

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

start = time.perf_counter()
result = fib(40)
elapsed = time.perf_counter() - start

print(f"TIME:{int(elapsed * 1000)}")
print(f"RESULT:{result}")
