# Test: Basic record type
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def distance(p):
    return p.x * p.x + p.y * p.y

def main():
    p = Point(3, 4)
    return distance(p)

print(main())
