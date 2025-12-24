# Test: Check if number is palindrome
def reverse_num(n, acc):
    if n == 0:
        return acc
    return reverse_num(n // 10, acc * 10 + n % 10)

def is_palindrome(n):
    return n == reverse_num(n, 0)

def b2i(b):
    return 1 if b else 0

def main():
    return b2i(is_palindrome(12321)) + b2i(not is_palindrome(12345)) * 10 + b2i(is_palindrome(1001)) * 100

print(main())
