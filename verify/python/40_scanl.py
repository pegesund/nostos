# Test: Scanl (running accumulator)
def scanl(f, acc, lst):
    if not lst:
        return [acc]
    return [acc] + scanl(f, f(acc, lst[0]), lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    nums = [1, 2, 3, 4, 5]
    running = scanl(lambda a, b: a + b, 0, nums)
    return sum_list(running)

print(main())
