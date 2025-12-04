#!/usr/bin/env python3
# Fibonacci benchmark

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

result = fib(35)
print(result)
