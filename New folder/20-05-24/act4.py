class Company:
    # Class variable
    industry = "Technology"

    def __init__(self, name, revenue):
        self.name = name
        self.revenue = revenue

# Creating instances of the Company class
company1 = Company("Company A", 1000000)
company2 = Company("Company B", 1500000)

# Accessing the class variable
print(company1.industry)  # Output: Technology
print(company2.industry)  # Output: Technology
