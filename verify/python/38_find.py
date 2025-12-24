# Test: Find first matching element
def find(p, lst):
    if not lst:
        return -1
    if p(lst[0]):
        return lst[0]
    return find(p, lst[1:])

def main():
    nums = [3, 7, 12, 15, 20, 25]
    first_over10 = find(lambda x: x > 10, nums)
    first_over100 = find(lambda x: x > 100, nums)
    first_even = find(lambda x: x % 2 == 0, nums)
    return first_over10 * 100 + first_even * 10 + (1 if first_over100 == -1 else 0)

print(main())
