# Test: Sliding window of size 2
def windows(lst):
    if len(lst) < 2:
        return []
    return [(lst[0], lst[1])] + windows(lst[1:])

def sum_pairs(pairs):
    if not pairs:
        return 0
    a, b = pairs[0]
    return a + b + sum_pairs(pairs[1:])

def main():
    lst = [1, 2, 3, 4, 5]
    wins = windows(lst)
    return sum_pairs(wins)

print(main())
