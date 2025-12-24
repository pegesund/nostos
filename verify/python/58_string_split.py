# Test: String upper/lower case
def main():
    s = "hello"
    upper = s.upper()
    lower = "WORLD".lower()
    return 1 if upper == "HELLO" and lower == "world" else 0

print(main())
