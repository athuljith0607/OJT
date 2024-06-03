import numpy as np

# Example 3x3 matrix
matrix_3x3 = np.array([[1, 2, 3], 
                       [4, 5, 6], 
                       [7, 8, 9]])

# Extracting the diagonal elements
diagonal_elements_3x3 = np.diagonal(matrix_3x3)

# Calculating the sum of the diagonal elements
sum_diagonal_3x3 = np.sum(diagonal_elements_3x3)

print(f"Sum of the diagonal elements in the 3x3 matrix: {sum_diagonal_3x3}")


import numpy as np

# Example 4x4 matrix
matrix_4x4 = np.array([[1, 2, 3, 4], 
                       [5, 6, 7, 8], 
                       [9, 10, 11, 12], 
                       [13, 14, 15, 16]])

# Extracting the diagonal elements
diagonal_elements_4x4 = np.diagonal(matrix_4x4)

# Calculating the sum of the diagonal elements
sum_diagonal_4x4 = np.sum(diagonal_elements_4x4)

print(f"Sum of the diagonal elements in the 4x4 matrix: {sum_diagonal_4x4}")


import numpy as np

# Example 3x3 matrix
matrix_3x3 = np.array([[1, 2, 3], 
                       [0, 1, 4], 
                       [5, 6, 0]])

# Calculating the inverse of the matrix
try:
    inverse_matrix_3x3 = np.linalg.inv(matrix_3x3)
    print("Inverse of the 3x3 matrix:")
    print(inverse_matrix_3x3)
except np.linalg.LinAlgError:
    print("The matrix is singular and cannot be inverted.")
