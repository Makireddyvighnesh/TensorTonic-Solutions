import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) == 0:
        return np.empty((0,0), dtype=int) # Shape(0,0)
        
    if max_len is None:
        max_len = max(len(row) for row in seqs)
    
    padded = np.array([
        row[:max_len] + [pad_value] * max(0, (max_len - len(row))) 
        for row in seqs
    ])

    return padded