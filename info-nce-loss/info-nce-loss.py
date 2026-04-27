import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """

    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    S = Z1 @ Z2.T / temperature
    S = S - np.max(S)

    exp_S = np.exp(S)
    sum_S = np.sum(exp_S, axis = 1)
    diag = np.diag(exp_S)
    return -np.sum(np.log(np.divide(diag, sum_S))) / (Z1.shape[0])