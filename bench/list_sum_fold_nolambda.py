#!/usr/bin/env python3
# Fold without lambda - just inline the add
import sys
sys.setrecursionlimit(10000)

# Linked list as nested tuples
def is_empty(lst):
    return lst is None

def head(lst):
    return lst[0]

def tail(lst):
    return lst[1]

# Fold with inlined add (no lambda)
def sum_tco(acc, lst):
    if is_empty(lst):
        return acc
    else:
        return sum_tco(acc + head(lst), tail(lst))

def sum_list(lst):
    return sum_tco(0, lst)

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
