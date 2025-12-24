# Test: Nested lambda expressions
def main():
    f = lambda x: lambda y: lambda z: x + y * z
    g = f(1)(2)
    return g(3) + g(4)

print(main())
