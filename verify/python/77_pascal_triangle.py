# Test: Pascal's triangle value
def binomial(n, k):
    if k == 0:
        return 1
    if k == n:
        return 1
    return binomial(n - 1, k - 1) + binomial(n - 1, k)

def main():
    return binomial(5, 2) + binomial(10, 5) * 10

print(main())
