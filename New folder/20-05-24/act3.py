class Vehicle:
    def __init__(self, color, max_speed):
        self.color = color
        self.max_speed = max_speed

class Bus(Vehicle):
    def __init__(self, color, max_speed, seating_capacity=50):
        super().__init__(color, max_speed)
        self.seating_capacity = seating_capacity

# Example usage:
my_bus = Bus(color='Yellow', max_speed=60)
print("Bus color:", my_bus.color)
print("Bus max speed:", my_bus.max_speed)
print("Bus seating capacity:", my_bus.seating_capacity)
