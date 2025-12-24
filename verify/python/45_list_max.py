# Test: Find maximum in list
def max_list(lst):
    if len(lst) == 1:
        return lst[0]
    rest_max = max_list(lst[1:])
    if lst[0] > rest_max:
        return lst[0]
    return rest_max

def main():
    return max_list([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])

print(main())
