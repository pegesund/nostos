# Test: Split list into chunks
def take(n, lst):
    if n == 0 or not lst:
        return []
    return [lst[0]] + take(n - 1, lst[1:])

def drop(n, lst):
    if n == 0:
        return lst
    if not lst:
        return []
    return drop(n - 1, lst[1:])

def chunks(n, lst):
    if not lst:
        return []
    return [take(n, lst)] + chunks(n, drop(n, lst))

def main():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    chunked = chunks(3, lst)
    return len(chunked)

print(main())
