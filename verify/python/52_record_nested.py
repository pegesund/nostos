# Test: Nested records
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self, topLeft, bottomRight):
        self.topLeft = topLeft
        self.bottomRight = bottomRight

def area(r):
    width = r.bottomRight.x - r.topLeft.x
    height = r.bottomRight.y - r.topLeft.y
    return width * height

def main():
    rect = Rectangle(Point(0, 0), Point(5, 4))
    return area(rect)

print(main())
