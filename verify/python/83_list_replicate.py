# Test: Replicate element n times
def replicate(n, x):
    if n == 0:
        return []
    return [x] + replicate(n - 1, x)

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    lst = replicate(5, 7)
    return sum_list(lst)

print(main())
