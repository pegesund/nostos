#!/usr/bin/env python3
# Array sum benchmark - matching Nostos Int64Array version
import numpy as np

def array_sum(arr):
    acc = 0
    i = 0
    length = len(arr)
    while i < length:
        acc = acc + arr[i]
        i = i + 1
    return acc

def main():
    arr = np.arange(1, 10001, dtype=np.int64)
    total = 0
    j = 0
    while j < 100:
        total = total + array_sum(arr)
        j = j + 1
    print(total)

if __name__ == "__main__":
    main()
