import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.array(x)
    
    if len(x.shape) > 2:
        # 4D case: (N, C, H, W) - normalize each channel over N, H, W
        gamma = np.array(gamma).reshape(1, x.shape[1], 1, 1)
        beta = np.array(beta).reshape(1, x.shape[1], 1, 1)
        axes = (0, 2, 3)
    else:
        # 2D case: (N, D) - normalize each feature over batch
        axes = 0
    
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
    
    return np.multiply((x - mean) / ((var + eps) ** (1/2)), gamma) + beta