import numpy as np

# Define two matrices
matrix_a = np.array([[1, 2, 3], 
                     [4, 5, 6], 
                     [7, 8, 9]])

matrix_b = np.array([[9, 8, 7], 
                     [6, 5, 4], 
                     [3, 2, 1]])

# Adding the matrices
result_add = matrix_a + matrix_b

print("Result of matrix addition:")
print(result_add)





import numpy as np

# Define two matrices
matrix_a = np.array([[1, 2, 3], 
                     [4, 5, 6], 
                     [7, 8, 9]])

matrix_b = np.array([[9, 8, 7], 
                     [6, 5, 4], 
                     [3, 2, 1]])

# Subtracting the matrices
result_subtract = matrix_a - matrix_b

print("Result of matrix subtraction:")
print(result_subtract)








import numpy as np

# Example array
arr = np.array([1, 2, 3, 1, 2, 1, 3, 2, 1])

# Get unique values and their counts
unique_values, counts = np.unique(arr, return_counts=True)

# Print the unique values and their counts
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")




