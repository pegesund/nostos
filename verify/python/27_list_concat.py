# Test: List concatenation
def list_len(lst):
    if not lst:
        return 0
    return 1 + list_len(lst[1:])

def main():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = a + b
    return list_len(c)

print(main())
