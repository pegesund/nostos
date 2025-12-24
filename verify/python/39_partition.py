# Test: Partition list by predicate
def partition(p, lst):
    if not lst:
        return ([], [])
    yes, no = partition(p, lst[1:])
    if p(lst[0]):
        return ([lst[0]] + yes, no)
    return (yes, [lst[0]] + no)

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    evens, odds = partition(lambda x: x % 2 == 0, nums)
    return sum_list(evens) * 100 + sum_list(odds)

print(main())
