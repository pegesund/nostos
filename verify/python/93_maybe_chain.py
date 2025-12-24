# Test: Option/Maybe chaining
class Just:
    def __init__(self, value):
        self.value = value

class Nothing:
    pass

def flat_map_maybe(f, m):
    if isinstance(m, Nothing):
        return Nothing()
    return f(m.value)

def safe_div(a, b):
    if b == 0:
        return Nothing()
    return Just(a // b)

def get_or(m, default):
    if isinstance(m, Just):
        return m.value
    return default

def main():
    result = flat_map_maybe(
        lambda x: flat_map_maybe(
            lambda y: Just(x + y),
            safe_div(20, 4)
        ),
        safe_div(100, 5)
    )
    return get_or(result, -1)

print(main())
