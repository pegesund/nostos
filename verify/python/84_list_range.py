# Test: Generate range using recursive list building
def make_range(start, stop):
    if start > stop:
        return []
    rest = make_range(start + 1, stop)
    return [start] + rest

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    return sum_list(make_range(1, 10))

print(main())
