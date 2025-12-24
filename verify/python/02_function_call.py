# Test: Simple function calls
def add(a, b):
    return a + b

def double(x):
    return x * 2

def main():
    return add(double(3), 7) + double(3)

print(main())
