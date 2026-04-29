import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs) == 0:
        return np.empty((0, 0))
    max_len = max_len if max_len != None else max([len(seq) for seq in seqs])
    for i in range(len((seqs))):
        if len(seqs[i]) < max_len:
            for j in range(len(seqs[i]), max_len):
                seqs[i].append(pad_value)
        else:
            if len(seqs[i]) > max_len:
                seqs[i] = seqs[i][:max_len]
    return np.array(seqs)