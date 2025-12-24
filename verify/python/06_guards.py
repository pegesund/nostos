# Test: Guards in function clauses
def classify(n):
    if n > 100:
        return 3
    elif n > 10:
        return 2
    elif n > 0:
        return 1
    else:
        return 0

def main():
    return classify(150) + classify(50) + classify(5) + classify(-1)

print(main())
