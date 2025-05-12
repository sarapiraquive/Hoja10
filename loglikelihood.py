import numpy as np
from dataset import X_bias, y_encoded
from get_probs import theta_test, get_probs

def loglikelihood(theta):
    """
    Calcula la log-verosimilitud total para el conjunto (X, y_encoded)
    theta: matriz de pesos (c-1, d+1)
    """
    global X_bias, y_encoded  # Usa los datos globales

    probs = get_probs(theta, X_bias)  # (n, c)

    # Selecciona la probabilidad predicha correcta para cada instancia
    n = X_bias.shape[0]
    correct_class_probs = probs[np.arange(n), y_encoded]

    # Suma de log-probabilidades
    return np.sum(np.log(correct_class_probs + 1e-15))  # +epsilon para evitar log(0)

print("Loglikelihood inicial:", loglikelihood(theta_test))


