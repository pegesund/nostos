# Test: Find index of element
def index_of(x, lst, idx):
    if not lst:
        return -1
    if x == lst[0]:
        return idx
    return index_of(x, lst[1:], idx + 1)

def find_index(x, lst):
    return index_of(x, lst, 0)

def main():
    lst = [10, 20, 30, 40, 50]
    return find_index(30, lst) + find_index(99, lst) * 10 + find_index(10, lst) * 100

print(main())
