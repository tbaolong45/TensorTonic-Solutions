import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.array(p)
    y = np.array(y)

    return 1 - np.divide(2 * np.sum(np.multiply(p, y)) + eps, np.sum(p) + np.sum(y) + eps)