# Test: Power function with recursion
def my_pow(base, exp):
    if exp == 0:
        return 1
    return base * my_pow(base, exp - 1)

def main():
    return my_pow(2, 10) + my_pow(3, 4)

print(main())
