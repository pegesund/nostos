# Test: Flatmap (bind/concatMap)
def flat_map(f, lst):
    if not lst:
        return []
    return f(lst[0]) + flat_map(f, lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    lst = [1, 2, 3]
    expanded = flat_map(lambda x: [x, x * 10], lst)
    return sum_list(expanded)

print(main())
