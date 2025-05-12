import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar el archivo iris.csv
df = pd.read_csv("iris.csv")
print(df.columns)

# Usamos solo las dos últimas columnas de atributos
X = df.iloc[:, -3:-1].values

# Codificamos las clases a valores numéricos: 0, 1, 2
y = df.iloc[:, -1].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Añadir el sesgo (columna de 1s)
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
X_bias = np.array(X_bias, dtype=float)

# Dimensiones
n_samples, n_features = X_bias.shape
n_classes = len(np.unique(y))
