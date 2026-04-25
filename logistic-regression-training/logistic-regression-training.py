import numpy as np
from sklearn.linear_model import LogisticRegression 
import torch.nn as nn
import torch.optim as optim

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        Z = X @ w + b
        a = _sigmoid(Z)

        loss = a - y
        dw = X.T @ loss / N
        db = np.mean(loss)

        w -= lr * dw
        b -= lr * db
    
    return w, b