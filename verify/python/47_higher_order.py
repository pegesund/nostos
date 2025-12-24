# Test: Higher order functions (pipe/compose)
def pipe(x, f):
    return f(x)

def apply_twice(f, x):
    return f(f(x))

def apply_n(n, f, x):
    if n == 0:
        return x
    return apply_n(n - 1, f, f(x))

def main():
    a = pipe(5, lambda x: x * 2)
    b = apply_twice(lambda x: x + 3, 10)
    c = apply_n(4, lambda x: x * 2, 1)
    return a + b * 10 + c * 100

print(main())
