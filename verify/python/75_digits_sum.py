# Test: Sum of digits
def digits_sum(n):
    if n == 0:
        return 0
    return n % 10 + digits_sum(n // 10)

def main():
    return digits_sum(12345) + digits_sum(9999)

print(main())
