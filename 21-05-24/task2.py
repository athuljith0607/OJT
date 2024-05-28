import math

class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

# Example usage
if __name__ == "__main__":
    rectangle = Rectangle(5, 4)
    print("Area of the rectangle:", rectangle.area())

    circle = Circle(3)
    print("Area of the circle:", circle.area())
