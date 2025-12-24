# Test: Iterating over list (recursive)
def iter_sum(lst, acc):
    if not lst:
        return acc
    return iter_sum(lst[1:], acc + lst[0])

def main():
    return iter_sum([1, 2, 3, 4, 5], 0)

print(main())
