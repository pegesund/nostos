# Test: State threading (monad-like pattern)
def tick(s):
    return (s, s + 1)

def main():
    v1, s1 = tick(0)
    v2, s2 = tick(s1)
    v3, s3 = tick(s2)
    return v1 + v2 * 10 + v3 * 100 + s3 * 1000

print(main())
