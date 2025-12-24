# Test: List head and tail pattern matching
def sum_first_two(lst):
    if len(lst) >= 2:
        return lst[0] + lst[1]
    elif len(lst) == 1:
        return lst[0]
    else:
        return 0

def main():
    return sum_first_two([10, 20, 30, 40]) + sum_first_two([5]) + sum_first_two([])

print(main())
