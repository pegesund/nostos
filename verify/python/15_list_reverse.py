# Test: Reverse list
def rev_list(lst, acc=None):
    if acc is None:
        acc = []
    if not lst:
        return acc
    return rev_list(lst[1:], [lst[0]] + acc)

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    reversed_lst = rev_list([1, 2, 3, 4, 5])
    return sum_list(reversed_lst)

print(main())
