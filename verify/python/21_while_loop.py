# Test: While loop accumulation
def main():
    total = 0
    i = 1
    while i <= 10:
        total = total + i
        i = i + 1
    return total

print(main())
