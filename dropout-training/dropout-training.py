import numpy as np

def dropout(X, drop_rate=0.5, rng=None):

    
    X = np.array(X)
    if rng == None:
        mask = (np.random.random(X.shape) > drop_rate).astype(float)
    else:
        mask = (rng.random(X.shape) > drop_rate).astype(float)

    keep_prob = 1 - drop_rate
    return (X * mask) / keep_prob, mask / keep_prob