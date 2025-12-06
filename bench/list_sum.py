#!/usr/bin/env python3
# List sum benchmark - comparable to list_sum.nos
# Uses functional style to match Nostos

def fold(f, acc, lst):
    for x in lst:
        acc = f(acc, x)
    return acc

def list_sum(lst):
    return fold(lambda a, b: a + b, 0, lst)

def main():
    # Create list of 10000 integers and sum 10 times (same total work)
    lst = list(range(1, 10001))  # [1..10000]
    total = 0
    for _ in range(100):
        total += list_sum(lst)
    print(total)

if __name__ == "__main__":
    main()
