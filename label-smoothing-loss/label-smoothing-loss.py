import numpy as np
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    q = [(idx == target) * (1 - epsilon) + epsilon/len(predictions) for idx, prob in enumerate(predictions)]

    q = np.array(q)
    predictions = np.array(predictions)
    
    return -np.sum(np.multiply(q, np.log(predictions)))
        