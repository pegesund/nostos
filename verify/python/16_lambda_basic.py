# Test: Basic lambda expressions
def apply(f, x):
    return f(x)

def twice(f, x):
    return f(f(x))

def main():
    a = apply(lambda x: x * 3, 5)
    b = twice(lambda x: x + 1, 10)
    return a + b

print(main())
