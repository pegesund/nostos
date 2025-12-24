# Test: Length of list recursively
def list_len(lst):
    if not lst:
        return 0
    return 1 + list_len(lst[1:])

def main():
    return list_len([10, 20, 30, 40])

print(main())
