# Test: Modular exponentiation
def mod_pow(base, exp, m):
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = mod_pow(base, exp // 2, m)
        return (half * half) % m
    return (base * mod_pow(base, exp - 1, m)) % m

def main():
    return mod_pow(3, 10, 1000)

print(main())
