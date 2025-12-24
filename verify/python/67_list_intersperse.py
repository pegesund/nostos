# Test: Intersperse element between list items
def intersperse(sep, lst):
    if not lst:
        return []
    if len(lst) == 1:
        return lst
    return [lst[0], sep] + intersperse(sep, lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    result = intersperse(0, [1, 2, 3, 4])
    return sum_list(result)

print(main())
