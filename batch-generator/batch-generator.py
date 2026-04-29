import numpy as np
import math

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X = np.array(X)
    y = np.array(y)
    
    indices = np.arange(X.shape[0])

    if rng!=None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
        
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(math.floor(len(X)/batch_size) + (drop_last == False) * (len(X) % batch_size != 0)):
        yield (X_shuffled[batch_size * i: min(batch_size * (i + 1), len(X_shuffled))], y_shuffled[batch_size * i: min(batch_size * (i + 1), len(X_shuffled))])