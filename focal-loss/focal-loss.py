import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.array(p)
    y = np.array(y)
    return np.mean(-np.multiply(np.multiply(np.power(np.subtract(1, p), gamma), y), np.log(p)) - np.multiply(np.multiply(np.power(p, gamma), np.subtract(1, y)), np.log(np.subtract(1, p))))