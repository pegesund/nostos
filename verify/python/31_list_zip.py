# Test: Zipping two lists
def zip_lists(a, b):
    if not a or not b:
        return []
    return [(a[0], b[0])] + zip_lists(a[1:], b[1:])

def sum_pairs(pairs):
    if not pairs:
        return 0
    a, b = pairs[0]
    return a + b + sum_pairs(pairs[1:])

def main():
    pairs = zip_lists([1, 2, 3], [10, 20, 30])
    return sum_pairs(pairs)

print(main())
