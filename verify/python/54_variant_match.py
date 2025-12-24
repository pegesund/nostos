# Test: Variant with match expression
class Ok:
    def __init__(self, value):
        self.value = value

class Err:
    def __init__(self, msg):
        self.msg = msg

def process_result(r):
    if isinstance(r, Ok):
        return r.value * 2
    return -1

def main():
    a = process_result(Ok(21))
    b = process_result(Err("failed"))
    return a + b

print(main())
