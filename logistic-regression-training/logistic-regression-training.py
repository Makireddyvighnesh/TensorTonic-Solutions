import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # Convert list to numpy array
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    n_samples, n_features = X.shape

    # Initialize
    W = np.zeros(n_features)
    b = 0

    for step in range(steps):
        # forward
        linear = np.dot(X, W) + b
        y_pred = _sigmoid(linear)

        # gradients
        dw = (1/n_samples) * np.dot(X.T , (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred-y)

        # Update
        W -= lr * dw
        b -= lr * db

    return W, b