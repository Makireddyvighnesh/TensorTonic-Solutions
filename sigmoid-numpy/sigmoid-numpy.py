import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    X = np.array(x, dtype = float)
    return 1/(1+np.exp(-X))
    