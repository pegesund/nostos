# Test: Comparison operators
def check_bool(b):
    return 1 if b else 0

def main():
    a = 5 > 3
    b = 5 < 3
    c = 5 == 5
    d = 5 != 3
    e = 5 >= 5
    f = 3 <= 5
    return check_bool(a) + check_bool(not b) + check_bool(c) + check_bool(d) + check_bool(e) + check_bool(f)

print(main())
