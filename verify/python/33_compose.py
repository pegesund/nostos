# Test: Function composition
def compose(f, g):
    return lambda x: f(g(x))

def double(x):
    return x * 2

def add_one(x):
    return x + 1

def square(x):
    return x * x

def main():
    f = compose(double, add_one)
    g = compose(add_one, double)
    h = compose(square, add_one)
    return f(5) * 1000 + g(5) * 100 + h(5)

print(main())
