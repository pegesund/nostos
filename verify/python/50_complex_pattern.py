# Test: Complex pattern matching with guards
def classify(lst):
    if len(lst) == 0:
        return "empty"
    elif len(lst) == 1:
        return "single"
    elif len(lst) == 2:
        a, b = lst
        if a == b:
            return "pair-same"
        elif a < b:
            return "pair-ascending"
        else:
            return "pair-descending"
    elif len(lst) == 3:
        a, b, c = lst
        if a < b and b < c:
            return "triple-ascending"
        elif a > b and b > c:
            return "triple-descending"
        else:
            return "other"
    else:
        return "other"

def b2i(b):
    return 1 if b else 0

def main():
    t1 = classify([])
    t2 = classify([5])
    t3 = classify([3, 3])
    t4 = classify([1, 2])
    t5 = classify([5, 3])
    t6 = classify([1, 2, 3])
    t7 = classify([3, 2, 1])
    t8 = classify([1, 2, 3, 4])

    return (b2i(t1 == "empty") +
            b2i(t2 == "single") * 10 +
            b2i(t3 == "pair-same") * 100 +
            b2i(t4 == "pair-ascending") * 1000 +
            b2i(t5 == "pair-descending") * 10000 +
            b2i(t6 == "triple-ascending") * 100000 +
            b2i(t7 == "triple-descending") * 1000000 +
            b2i(t8 == "other") * 10000000)

print(main())
