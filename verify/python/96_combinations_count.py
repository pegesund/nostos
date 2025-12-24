# Test: Count combinations C(n,k)
def comb(n, k):
    if k == 0:
        return 1
    if k == n:
        return 1
    if k > n:
        return 0
    return comb(n - 1, k - 1) + comb(n - 1, k)

def main():
    return comb(10, 3) + comb(6, 2) * 1000

print(main())
