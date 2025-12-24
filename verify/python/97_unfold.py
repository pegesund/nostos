# Test: Unfold to generate list
def unfold(stop, gen, seed):
    if stop(seed):
        return []
    val, next_seed = gen(seed)
    return [val] + unfold(stop, gen, next_seed)

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

def main():
    lst = unfold(lambda s: s > 5, lambda s: (s, s + 1), 1)
    return sum_list(lst)

print(main())
