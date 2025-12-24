# Test: Mutual recursion (even/odd check)
def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)

def main():
    a = 1 if is_even(10) else 0
    b = 10 if is_odd(7) else 0
    c = 100 if is_even(5) else 0
    return a + b + c

print(main())
