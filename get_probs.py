import numpy as np
from dataset import n_classes, X_bias


def get_probs(theta, X):
    """
    Calcula la matriz de probabilidades usando softmax.
    theta: matriz (c-1, d+1)
    X: matriz de entradas con sesgo (n, d+1)
    Return: matriz (n, c) de probabilidades
    """
    X = np.array(X, dtype=float)
    theta = np.array(theta, dtype=float)

    logits = X @ theta.T  # (n, c-1)
    logits_full = np.hstack([logits, np.zeros((X.shape[0], 1))])  # (n, c)
    exp_logits = np.exp(logits_full - np.max(logits_full, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probs

theta_test = np.zeros((n_classes - 1, X_bias.shape[1]))
theta_test = np.array(theta_test, dtype=float)
probs = get_probs(theta_test, X_bias)
print(probs.shape)
print(np.allclose(probs.sum(axis=1), 1))  