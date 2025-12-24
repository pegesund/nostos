# Test: Quicksort algorithm
def qsort(lst):
    if not lst:
        return []
    pivot = lst[0]
    rest = lst[1:]
    lesser = [x for x in rest if x < pivot]
    greater = [x for x in rest if x >= pivot]
    return qsort(lesser) + [pivot] + qsort(greater)

def main():
    sorted_lst = qsort([3, 1, 4, 1, 5, 9, 2, 6])
    return sorted_lst[0]

print(main())
