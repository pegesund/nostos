# Test: Check if prime
def is_prime_helper(n, d):
    if d * d > n:
        return True
    if n % d == 0:
        return False
    return is_prime_helper(n, d + 1)

def is_prime(n):
    if n < 2:
        return False
    return is_prime_helper(n, 2)

def b2i(b):
    return 1 if b else 0

def main():
    return b2i(is_prime(2)) + b2i(is_prime(17)) * 10 + b2i(not is_prime(15)) * 100 + b2i(is_prime(97)) * 1000

print(main())
