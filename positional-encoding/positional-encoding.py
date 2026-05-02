import numpy as np

def positional_encoding(seq_len, d_model, base = 100000.0):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    
    div_term = np.exp(
        np.arange(0, d_model, 2) * 
        (-np.log(base) / d_model)
    )
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
    
    return pe