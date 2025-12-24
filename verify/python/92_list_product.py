# Test: Cartesian product of two lists
def pair_with(x, ys):
    if not ys:
        return []
    return [(x, ys[0])] + pair_with(x, ys[1:])

def product(xs, ys):
    if not xs:
        return []
    return pair_with(xs[0], ys) + product(xs[1:], ys)

def main():
    a = [1, 2, 3]
    b = [10, 20]
    pairs = product(a, b)
    return len(pairs)

print(main())
