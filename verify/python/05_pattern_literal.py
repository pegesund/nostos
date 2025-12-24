# Test: Pattern matching on literals
def describe(n):
    if n == 0:
        return 0
    elif n == 1:
        return 10
    elif n == 2:
        return 20
    else:
        return 100

def main():
    return describe(0) + describe(1) + describe(2) + describe(99)

print(main())
