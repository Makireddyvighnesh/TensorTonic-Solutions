import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    V[s] += alpha * (r + gamma * V[s_next] - V[s])

    return V