#!/usr/bin/env python3
# Fold benchmark - matching Nostos fold_tco complexity exactly
import sys
sys.setrecursionlimit(10000)

# Linked list as nested tuples: (head, tail) or None for empty
def is_empty(lst):
    return lst is None

def head(lst):
    return lst[0]

def tail(lst):
    return lst[1]

# Tail-recursive fold (Python doesn't optimize, but same complexity)
def fold_tco(f, acc, lst):
    if is_empty(lst):
        return acc
    else:
        return fold_tco(f, f(acc, head(lst)), tail(lst))

def sum_list(lst):
    return fold_tco(lambda a, b: a + b, 0, lst)

# Build linked list [n, n-1, ..., 1]
def range_list(n):
    lst = None
    i = 1
    while i <= n:
        lst = (i, lst)
        i = i + 1
    return lst

def main():
    lst = range_list(1000)
    total = 0
    i = 0
    while i < 1000:
        total = total + sum_list(lst)
        i = i + 1
    print(total)

if __name__ == "__main__":
    main()
