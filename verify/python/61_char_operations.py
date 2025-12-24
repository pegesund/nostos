# Test: Integer digit count
def count_digits(n):
    if n == 0:
        return 0
    return 1 + count_digits(n // 10)

def main():
    return count_digits(12345) + count_digits(100) * 10

print(main())
