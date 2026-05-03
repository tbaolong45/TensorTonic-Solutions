import numpy as np

import numpy as np

def apply_transform(x, A):
    x = np.asarray(x)
    return x @ A.T

def rotate_around_z(points, theta):
    r = np.array([
        [round(np.cos(theta), 10), round(-np.sin(theta), 10), 0],
        [round(np.sin(theta), 10),  round(np.cos(theta), 10), 0],
        [0            ,  0            , 1]
    ])

    return apply_transform(points, r)