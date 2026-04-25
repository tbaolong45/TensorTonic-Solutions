import numpy as np

def softmax(x):
    x = np.array(x)
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    max_x = np.max(x, axis = (1 if len(x.shape) == 2 else 0), keepdims = (len(x.shape) == 2))
    new = np.exp(np.subtract(x, max_x))
    sum = np.sum(new, axis = (1 if len(x.shape) == 2 else 0), keepdims = (len(x.shape) == 2))
    return np.divide(new, sum)