import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    # X shape (N, D)
    X = np.array(X, dtype=float)
    if X.ndim != 2:
        return None
        
    N = X.shape[0]

    if N <= 1:
        return None
    
    mean = np.mean(X, axis=0)
    X = X - mean

    return (1/(N-1)) * (X.T @ X)
    