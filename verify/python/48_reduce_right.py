# Test: Right fold (reduce from right)
def foldr(f, init, lst):
    if not lst:
        return init
    return f(lst[0], foldr(f, init, lst[1:]))

def main():
    nums = [1, 2, 3, 4]
    # Build right-associative: 1 - (2 - (3 - (4 - 0))) = 1 - (2 - (3 - 4)) = 1 - (2 - (-1)) = 1 - 3 = -2
    result = foldr(lambda a, b: a - b, 0, nums)
    return result + 100

print(main())
