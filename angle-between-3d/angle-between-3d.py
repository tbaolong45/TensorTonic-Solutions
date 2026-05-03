import numpy as np

def angle_between_3d(v, w):
    v = np.array(v)
    w = np.array(w)

    prod = np.linalg.norm(v) * np.linalg.norm(w)
    tmp = v @ w.T
    return np.arccos(tmp / prod)