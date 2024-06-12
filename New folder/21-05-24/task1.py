class Vehicle:
    def __init__(self, seating_capacity):
        self.seating_capacity = seating_capacity

    def fare_charge(self):
        return self.seating_capacity * 100

class Bus(Vehicle):
    def __init__(self, seating_capacity):
        super().__init__(seating_capacity)

    def fare_charge(self):
        base_fare = super().fare_charge()
        maintenance_charge = base_fare * 0.1
        total_fare = base_fare + maintenance_charge
        return total_fare

# Example usage
if __name__ == "__main__":
    bus = Bus(50)
    print("Total fare for the bus:", bus.fare_charge())
