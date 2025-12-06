#!/usr/bin/env python3
# List sum benchmark - iterative style (matching Nostos)

def list_sum(lst):
    acc = 0
    idx = 0
    while idx < len(lst):
        acc = acc + lst[idx]
        idx = idx + 1
    return acc

def main():
    lst = list(range(1, 10001))  # [1..10000]
    total = 0
    i = 0
    while i < 100:
        total = total + list_sum(lst)
        i = i + 1
    print(total)

if __name__ == "__main__":
    main()
