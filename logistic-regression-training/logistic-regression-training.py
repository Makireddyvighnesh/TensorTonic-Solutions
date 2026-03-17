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
    losses = []
    eps = 1e-9

    for step in range(steps):
        # forward
        linear = np.dot(X, W) + b
        y_pred = _sigmoid(linear)

        # Compute Loss
        loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        losses.append(loss)
        
        # gradients
        dw = (1/n_samples) * np.dot(X.T , (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred-y)

        # Update
        W -= lr * dw
        b -= lr * db

        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return W, b