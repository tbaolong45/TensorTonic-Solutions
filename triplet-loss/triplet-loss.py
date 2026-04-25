def dis(x, y):
    # Squared L2 distance: works for both 1D (D,) and 2D (N, D)
    return np.sum((x - y) ** 2, axis=-1)

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)
    
    d_ap = dis(anchor, positive)   # d(a, p)
    d_an = dis(anchor, negative)   # d(a, n)
    
    # Element-wise max(0, ...), then average
    losses = np.maximum(0, d_ap - d_an + margin)
    return np.mean(losses)