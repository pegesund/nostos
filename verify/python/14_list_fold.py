# Test: Left fold (reduce)
def foldl(f, acc, lst):
    if not lst:
        return acc
    return foldl(f, f(acc, lst[0]), lst[1:])

def main():
    return foldl(lambda a, b: a + b, 0, [1, 2, 3, 4, 5])

print(main())
