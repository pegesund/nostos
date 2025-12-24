# Test: Map function over list
def map_list(f, lst):
    if not lst:
        return []
    return [f(lst[0])] + map_list(f, lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    doubled = map_list(lambda x: x * 2, [1, 2, 3, 4])
    return sum_list(doubled)

print(main())
