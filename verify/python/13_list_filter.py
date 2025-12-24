# Test: Filter list
def filter_list(p, lst):
    if not lst:
        return []
    if p(lst[0]):
        return [lst[0]] + filter_list(p, lst[1:])
    return filter_list(p, lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    evens = filter_list(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    return sum_list(evens)

print(main())
