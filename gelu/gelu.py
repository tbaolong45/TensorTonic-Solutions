import numpy as np
import math
from scipy.special import erf

def gelu(x):
    x = np.array(x)
    return np.divide(x, 2) * (1 + erf(np.divide(x, math.sqrt(2))))