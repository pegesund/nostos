#!/usr/bin/env python3
# Array sum benchmark - comparable to array_sum_jit.nos

def sum_arr(arr):
    total = 0
    for x in arr:
        total += x
    return total

def main():
    n = 1000000
    arr = list(range(1, n + 1))

    # Call sum_arr 100 times
    total = 0
    for _ in range(100):
        total += sum_arr(arr)
    print(total)

if __name__ == "__main__":
    main()
