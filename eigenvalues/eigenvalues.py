import numpy as np

def calculate_eigenvalues(matrix):
    try:
        matrix = np.array(matrix)
    except (ValueError, TypeError):
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    return np.linalg.eigvals(matrix)