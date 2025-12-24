#!/usr/bin/env python3
# List iteration benchmark - matches list_iterate.nos
# Tests head/tail pattern matching performance

import sys
sys.setrecursionlimit(60000)

# Count elements using pattern matching
def count_list(lst):
    if not lst:
        return 0
    return 1 + count_list(lst[1:])

# Sum using pattern matching (not tail recursive)
def sum_rec(lst):
    if not lst:
        return 0
    return lst[0] + sum_rec(lst[1:])

# Sum using tail recursion with pattern matching
def sum_tail(lst, acc):
    if not lst:
        return acc
    return sum_tail(lst[1:], acc + lst[0])

# Find max using pattern matching
def max_rec(lst):
    if len(lst) == 1:
        return lst[0]
    m = max_rec(lst[1:])
    return lst[0] if lst[0] > m else m

# Reverse using pattern matching (builds new list)
def reverse_acc(lst, acc):
    if not lst:
        return acc
    return reverse_acc(lst[1:], [lst[0]] + acc)

# Build test list
def build_list(n):
    if n == 0:
        return []
    return [n] + build_list(n - 1)

def main():
    n = 50000

    # Build list once
    lst = build_list(n)

    # Test 1: Count (pure iteration)
    c1 = count_list(lst)
    c2 = count_list(lst)
    c3 = count_list(lst)

    # Test 2: Sum tail recursive (pure iteration)
    s1 = sum_tail(lst, 0)
    s2 = sum_tail(lst, 0)
    s3 = sum_tail(lst, 0)

    # Test 3: Sum non-tail recursive
    r1 = sum_rec(lst)
    r2 = sum_rec(lst)
    r3 = sum_rec(lst)

    print(c1)
    print(s1)
    print(r1)

    return c1 + s1 + r1

if __name__ == "__main__":
    result = main()
    print(result)
