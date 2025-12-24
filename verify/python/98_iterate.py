# Test: Iterate function n times and collect
def iterate_n(n, f, x):
    if n == 0:
        return []
    return [x] + iterate_n(n - 1, f, f(x))

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    powers = iterate_n(5, lambda x: x * 2, 1)
    return sum_list(powers)

print(main())
