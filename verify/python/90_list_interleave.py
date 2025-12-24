# Test: Interleave two lists
def interleave(xs, ys):
    if not xs:
        return ys
    if not ys:
        return xs
    return [xs[0], ys[0]] + interleave(xs[1:], ys[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    a = [1, 3, 5]
    b = [2, 4, 6]
    result = interleave(a, b)
    return sum_list(result)

print(main())
