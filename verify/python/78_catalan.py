# Test: Catalan numbers
def catalan_helper(n, i):
    if i >= n:
        return 0
    return catalan(i) * catalan(n - 1 - i) + catalan_helper(n, i + 1)

def catalan(n):
    if n == 0:
        return 1
    return catalan_helper(n, 0)

def main():
    return catalan(5) + catalan(4) * 100

print(main())
