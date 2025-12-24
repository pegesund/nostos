# Test: Insertion sort
def insert(x, lst):
    if not lst:
        return [x]
    if x <= lst[0]:
        return [x] + lst
    return [lst[0]] + insert(x, lst[1:])

def isort(lst):
    if not lst:
        return []
    return insert(lst[0], isort(lst[1:]))

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    sorted_lst = isort([5, 2, 8, 1, 9])
    return sum_list(sorted_lst)

print(main())
