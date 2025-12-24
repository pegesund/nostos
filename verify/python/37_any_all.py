# Test: Any and all predicates
def any_match(p, lst):
    if not lst:
        return False
    if p(lst[0]):
        return True
    return any_match(p, lst[1:])

def all_match(p, lst):
    if not lst:
        return True
    if not p(lst[0]):
        return False
    return all_match(p, lst[1:])

def b2i(b):
    return 1 if b else 0

def main():
    nums = [2, 4, 6, 8, 10]
    has_odd = any_match(lambda x: x % 2 != 0, nums)
    all_even = all_match(lambda x: x % 2 == 0, nums)
    has_neg = any_match(lambda x: x < 0, nums)
    all_pos = all_match(lambda x: x > 0, nums)
    return b2i(not has_odd) + b2i(all_even) * 10 + b2i(not has_neg) * 100 + b2i(all_pos) * 1000

print(main())
