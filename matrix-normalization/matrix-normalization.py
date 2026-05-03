import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        matrix = np.array(matrix)
        if matrix.ndim != 2 or norm_type == "invalid":
            return None
        if norm_type == "l2":
            div_term = np.linalg.norm(matrix, axis = axis)
        elif norm_type == "l1":
            div_term = np.linalg.norm(matrix, ord = 1, axis = axis)
        else:
            div_term = np.linalg.norm(matrix, ord = np.inf, axis = axis) 
    
        if axis == 1:
            tmp = matrix.T / div_term
            matrix = tmp.T
        else:
            matrix = matrix / div_term
    
        return np.nan_to_num(matrix, nan=0.0)
    except Exception as e:
        return None