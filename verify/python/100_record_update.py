# Test: Record field access and computation
class Counter:
    def __init__(self, value, step):
        self.value = value
        self.step = step

def increment(c):
    return Counter(c.value + c.step, c.step)

def run_n(n, c):
    if n == 0:
        return c
    return run_n(n - 1, increment(c))

def main():
    c = Counter(0, 5)
    result = run_n(10, c)
    return result.value

print(main())
