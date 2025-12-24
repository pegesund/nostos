# Test: Nested if-else
def clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val

def main():
    return clamp(-5, 0, 10) + clamp(15, 0, 10) + clamp(5, 0, 10) - 15

print(main())
