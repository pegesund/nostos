# Test: Collatz sequence length
import sys
sys.setrecursionlimit(2000)

def collatz_len(n):
    if n == 1:
        return 1
    if n % 2 == 0:
        return 1 + collatz_len(n // 2)
    return 1 + collatz_len(3 * n + 1)

def main():
    return collatz_len(27)

print(main())
