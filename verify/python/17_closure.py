# Test: Closures capture variables
def make_adder(n):
    return lambda x: x + n

def main():
    add5 = make_adder(5)
    add10 = make_adder(10)
    return add5(3) + add10(7)

print(main())
