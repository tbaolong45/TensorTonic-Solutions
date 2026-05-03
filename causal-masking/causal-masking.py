import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    i, j = np.indices((scores.shape[-1], scores.shape[-1]))
    tmp = np.where(i >= j, np.inf, mask_value)
    
    if len(scores.shape) == 3:
        tmp = np.tile(tmp, (scores.shape[0], 1, 1))
    elif len(scores.shape) == 4:
        tmp = np.tile(tmp, (scores.shape[0], scores.shape[1], 1, 1))

    return np.minimum(tmp, scores)