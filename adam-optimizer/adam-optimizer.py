import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.asarray(param)
    grad  = np.asarray(grad)
    m     = np.asarray(m)
    v     = np.asarray(v)
    
    # Update biased first moment
    m_t = beta1 * m + (1 - beta1) * grad

    # Update biased second moment
    v_t = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m_t / (1 - beta1 ** t)
    v_hat = v_t / (1 - beta2 ** t)

    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_t, v_t

    

    
    

    
    