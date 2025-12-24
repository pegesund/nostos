# Test: Greatest common divisor (Euclidean algorithm)
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def main():
    return gcd(48, 18) + gcd(100, 35) * 100

print(main())
