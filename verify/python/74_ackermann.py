# Test: Ackermann function (small values)
import sys
sys.setrecursionlimit(10000)

def ack(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))

def main():
    return ack(3, 3)

print(main())
