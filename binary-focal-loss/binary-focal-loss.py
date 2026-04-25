import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    p_t = np.multiply(predictions, targets) + np.multiply(np.subtract(1, predictions), np.subtract(1, targets))

    return np.mean(np.multiply(-alpha * ((1 - p_t) ** gamma), np.log(p_t)))