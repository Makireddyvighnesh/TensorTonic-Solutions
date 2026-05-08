import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    if rng is None:
        rng = np.random.default_rng()

    # Exploration
    if rng.random() < epsilon:
        return rng.integers(len(q_values))

    return int(np.argmax(q_values))
    
    
