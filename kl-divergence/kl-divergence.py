import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.array(p)
    q = np.array(q)

    div = np.divide(p, q+eps)

    return np.sum(np.multiply(p, np.log(div)))