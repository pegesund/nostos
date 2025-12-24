# Test: Map insert and update
def main():
    m = {"x": 10}
    m2 = dict(m)
    m2["y"] = 20
    m3 = dict(m2)
    m3["x"] = 15
    return m3["x"] + m3["y"]

print(main())
