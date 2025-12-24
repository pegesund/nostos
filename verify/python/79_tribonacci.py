# Test: Tribonacci sequence
def trib(n):
    if n == 0:
        return 0
    if n == 1:
        return 0
    if n == 2:
        return 1
    return trib(n - 1) + trib(n - 2) + trib(n - 3)

def main():
    return trib(10) + trib(8) * 10

print(main())
