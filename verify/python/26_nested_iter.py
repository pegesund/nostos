# Test: Nested iteration (via recursion)
def inner_sum(i, js):
    if not js:
        return 0
    return i * js[0] + inner_sum(i, js[1:])

def outer_sum(is_list, js):
    if not is_list:
        return 0
    return inner_sum(is_list[0], js) + outer_sum(is_list[1:], js)

def main():
    return outer_sum([1, 2, 3], [10, 20, 30])

print(main())
