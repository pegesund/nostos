# Test: Split list at index using take/drop
def take(n, lst):
    if n == 0 or not lst:
        return []
    return [lst[0]] + take(n - 1, lst[1:])

def drop(n, lst):
    if n == 0:
        return lst
    if not lst:
        return []
    return drop(n - 1, lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    first = take(4, lst)
    second = drop(4, lst)
    return sum_list(first) * 100 + sum_list(second)

print(main())
