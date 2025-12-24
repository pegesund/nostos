# Test: Remove consecutive duplicates
def unique(lst):
    if not lst:
        return []
    if len(lst) == 1:
        return lst
    if lst[0] == lst[1]:
        return unique(lst[1:])
    return [lst[0]] + unique(lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    result = unique([1, 1, 2, 2, 2, 3, 3, 4])
    return sum_list(result)

print(main())
