# Test: Triple tuples
def fst3(t):
    return t[0]

def snd3(t):
    return t[1]

def thd3(t):
    return t[2]

def rotate_left(t):
    return (t[1], t[2], t[0])

def rotate_right(t):
    return (t[2], t[0], t[1])

def main():
    t = (1, 2, 3)
    left = rotate_left(t)
    right = rotate_right(t)
    return fst3(left) * 100 + snd3(right) * 10 + thd3(t)

print(main())
