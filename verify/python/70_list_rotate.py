# Test: Rotate list left
def rotate_left(n, lst):
    if n == 0 or not lst:
        return lst
    return rotate_left(n - 1, lst[1:] + [lst[0]])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    lst = [1, 2, 3, 4, 5]
    rotated = rotate_left(2, lst)
    return rotated[0] * 10 + sum_list(rotated)

print(main())
