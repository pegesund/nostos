# Test: Check if element in list
def elem(x, lst):
    if not lst:
        return False
    if x == lst[0]:
        return True
    return elem(x, lst[1:])

def b2i(b):
    return 1 if b else 0

def main():
    lst = [3, 1, 4, 1, 5, 9]
    return b2i(elem(4, lst)) + b2i(not elem(7, lst)) * 10 + b2i(elem(9, lst)) * 100

print(main())
