# Test: Recursive factorial
def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)

def main():
    return fact(6)

print(main())
