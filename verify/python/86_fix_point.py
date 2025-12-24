# Test: Fixed point iteration
def iterate_until(f, p, x):
    if p(x):
        return x
    return iterate_until(f, p, f(x))

def main():
    result = iterate_until(lambda x: x // 2, lambda x: x < 5, 100)
    return result

print(main())
