# Test: Boolean operations
def main():
    a = True
    b = False
    c = a and b
    d = a or b
    e = not b
    if c:
        return 1
    elif d and e:
        return 2
    else:
        return 3

print(main())
