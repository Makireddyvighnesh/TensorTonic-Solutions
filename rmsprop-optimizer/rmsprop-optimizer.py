import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)
    
    s1 = beta * s + (1-beta) * g**2
    w1 = w - (lr / (np.sqrt(s1 + eps))) * g

    return w1, s1