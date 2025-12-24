# Test: Count permutations (factorial)
def perm_count(n):
    if n == 0:
        return 1
    return n * perm_count(n - 1)

def main():
    return perm_count(5) + perm_count(3) * 1000

print(main())
