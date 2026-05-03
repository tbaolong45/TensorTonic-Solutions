import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.array(points)
    if (len(points.shape)==1):
        points = np.array([points])
    points = np.c_[points, np.ones(points.shape[0])]
    
    T = np.array(T)

    res = points @ T.T
    return res[:, :-1] if len(res[:, :-1]) > 1 else res[:, :-1][0]