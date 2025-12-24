# Test: Take while predicate holds
def take_while(p, lst):
    if not lst:
        return []
    if p(lst[0]):
        return [lst[0]] + take_while(p, lst[1:])
    return []

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    lst = [1, 2, 3, 7, 8, 2, 1]
    before = take_while(lambda x: x < 5, lst)
    return sum_list(before)

print(main())
