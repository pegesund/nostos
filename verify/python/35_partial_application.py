# Test: Partial application with map
def map_with(f, lst):
    if not lst:
        return []
    return [f(lst[0])] + map_with(f, lst[1:])

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    numbers = [1, 2, 3, 4, 5]
    multiplier = 10
    scaled = map_with(lambda x: x * multiplier, numbers)
    return sum_list(scaled)

print(main())
