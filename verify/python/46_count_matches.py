# Test: Count elements matching predicate
def count_if(p, lst):
    if not lst:
        return 0
    return (1 if p(lst[0]) else 0) + count_if(p, lst[1:])

def main():
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    evens = count_if(lambda x: x % 2 == 0, nums)
    greater_than5 = count_if(lambda x: x > 5, nums)
    return evens * 100 + greater_than5

print(main())
