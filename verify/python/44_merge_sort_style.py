# Test: Merge two sorted lists
def merge(xs, ys):
    if not xs:
        return ys
    if not ys:
        return xs
    if xs[0] <= ys[0]:
        return [xs[0]] + merge(xs[1:], ys)
    return [ys[0]] + merge(xs, ys[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    a = [1, 3, 5, 7]
    b = [2, 4, 6, 8]
    merged = merge(a, b)
    return sum_list(merged)

print(main())
