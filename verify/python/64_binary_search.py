# Test: Binary search
def bsearch(lst, target, lo, hi):
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    val = lst[mid]
    if val == target:
        return mid
    elif val < target:
        return bsearch(lst, target, mid + 1, hi)
    else:
        return bsearch(lst, target, lo, mid - 1)

def main():
    lst = [1, 3, 5, 7, 9, 11, 13, 15]
    return bsearch(lst, 7, 0, len(lst) - 1)

print(main())
