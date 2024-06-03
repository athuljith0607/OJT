import numpy as np

# Define two matrices
matrix_a = np.array([[1, 2], 
                     [3, 4]])

matrix_b = np.array([[5, 6], 
                     [7, 8]])

# Perform matrix multiplication
result_dot = np.dot(matrix_a, matrix_b)

print("Matrix multiplication using np.dot:")
print(result_dot)




import numpy as np

# Define two matrices
matrix_a = np.array([[1, 2], 
                     [3, 4]])

matrix_b = np.array([[5, 6], 
                     [7, 8]])

# Perform matrix multiplication
result_at = matrix_a @ matrix_b

print("Matrix multiplication using @ operator:")
print(result_at)




import numpy as np

# Define two matrices
matrix_a = np.array([[1, 2], 
                     [3, 4]])

matrix_b = np.array([[5, 6], 
                     [7, 8]])

# Perform matrix multiplication
result_matmul = np.matmul(matrix_a, matrix_b)

print("Matrix multiplication using np.matmul:")
print(result_matmul)









import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# Finding the maximum value in the matrix
max_value = np.max(matrix)

# Finding the minimum value in the matrix
min_value = np.min(matrix)

print(f"Maximum value in the matrix: {max_value}")
print(f"Minimum value in the matrix: {min_value}")



