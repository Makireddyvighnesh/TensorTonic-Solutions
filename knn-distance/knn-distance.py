import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1) # (n,1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]  # Numm of train samples

    # Compute Euclidean distance
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    distances = np.sqrt(dist_sq)

    # Get indices of the sorted distances
    neighbor_indices = np.argsort(distances, axis=1)

    # slice the first k neighbors
    res = neighbor_indices[:,:k]

    # handling case where k>n
    if k > n_train:
        padding = np.full((res.shape[0], k-n_train), -1)
        res = np.hstack([res, padding])

    return res.astype(int)
    
    