# Test: Tuple creation and destructuring
def swap(t):
    a, b = t
    return (b, a)

def first(t):
    return t[0]

def second(t):
    return t[1]

def main():
    t = (10, 20)
    swapped = swap(t)
    return first(swapped) + second(swapped)

print(main())
