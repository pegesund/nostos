# Test: Sum a list recursively
def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    return sum_list([1, 2, 3, 4, 5])

print(main())
