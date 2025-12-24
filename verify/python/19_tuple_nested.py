# Test: Nested tuples
def get_deep(t):
    return t[0][0]

def main():
    nested = ((5, 10), (15, 20))
    return get_deep(nested)

print(main())
