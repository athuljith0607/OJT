class ADIT:
    def __init__(self):
        self.name = ""
        self.phone = ""

    def store_name_and_phone(self, name, phone):
        self.name = name
        self.phone = phone

    def print_name_and_phone(self):
        print("Name:", self.name)
        print("Phone:", self.phone)


adit_instance = ADIT()
adit_instance.store_name_and_phone("John Doe", "123-456-7890")
adit_instance.print_name_and_phone()