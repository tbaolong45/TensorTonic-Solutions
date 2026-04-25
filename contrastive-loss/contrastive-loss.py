import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)
    
    # Compute per-sample L2 distance: shape (N,)
    d = np.linalg.norm(a - b, axis=(1 if len(a.shape) == 2 else 0))
    
    # Positive pairs (y=1): pull together → d^2
    pos_loss = y * (d ** 2)
    
    # Negative pairs (y=0): push apart → max(0, margin - d)^2
    neg_loss = (1 - y) * (np.maximum(0, margin - d) ** 2)
    
    losses = pos_loss + neg_loss
    
    if reduction == "mean":
        return np.mean(losses)
    else:
        return np.sum(losses)
        
    