# Test: Group consecutive equal elements
def group(lst):
    if not lst:
        return []
    if len(lst) == 1:
        return [[lst[0]]]
    rest = group(lst[1:])
    if lst[0] == lst[1]:
        return [[lst[0]] + rest[0]] + rest[1:]
    return [[lst[0]]] + rest

def main():
    groups = group([1, 1, 2, 2, 2, 3, 1, 1])
    return len(groups)

print(main())
