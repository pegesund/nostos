# Test: Currying
def add(a):
    return lambda b: a + b

def mul(a):
    return lambda b: a * b

def main():
    add5 = add(5)
    mul3 = mul(3)
    return add5(10) + mul3(7)

print(main())
