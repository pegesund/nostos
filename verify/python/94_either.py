# Test: Either type (Left/Right)
class Left:
    def __init__(self, value):
        self.value = value

class Right:
    def __init__(self, value):
        self.value = value

def map_right(f, e):
    if isinstance(e, Left):
        return e
    return Right(f(e.value))

def get_right(e):
    if isinstance(e, Right):
        return e.value
    return -1

def main():
    a = Right(10)
    b = Left("error")
    r1 = map_right(lambda x: x * 2, a)
    r2 = map_right(lambda x: x * 2, b)
    return get_right(r1) + get_right(r2) * 10

print(main())
