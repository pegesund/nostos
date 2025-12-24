# Test: Basic variant type (Option)
class Some:
    def __init__(self, value):
        self.value = value

class NoneType:
    pass

def get_or_default(opt, default):
    if isinstance(opt, Some):
        return opt.value
    return default

def main():
    a = Some(42)
    b = NoneType()
    return get_or_default(a, 0) + get_or_default(b, 10)

print(main())
