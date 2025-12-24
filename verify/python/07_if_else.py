# Test: If-else expressions
def max_val(a, b):
    return a if a > b else b

def abs_val(n):
    return n if n >= 0 else -n

def main():
    return max_val(10, 20) + abs_val(-10)

print(main())
