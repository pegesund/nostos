# Test: Nested if expressions
def classify(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n < 10:
        return "small"
    elif n < 100:
        return "medium"
    else:
        return "large"

def main():
    a = classify(-5)
    b = classify(0)
    c = classify(5)
    d = classify(50)
    e = classify(500)
    if a == "negative" and b == "zero" and c == "small" and d == "medium" and e == "large":
        return 1
    return 0

print(main())
