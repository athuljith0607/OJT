import numpy as np

# Example array
array = np.array([0, 1, 2, 0, 3, 0, 4])

# Counting non-zero values
count = np.count_nonzero(array)
print(f"Number of non-zero values: {count}")




import numpy as np

# Creating an empty array of shape (3, 3)
empty_array = np.empty((3, 3))
print("Empty Array:")
print(empty_array)
