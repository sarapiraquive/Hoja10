import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from IPython.display import Image

# 1. Función para calcular las probabilidades (softmax)
def get_probs(theta, X):
    """
    Calcula las probabilidades para todas las clases usando la función softmax.
    
    Args:
        theta: Matriz de parámetros de forma (c-1, d+1) donde c es el número de clases
               y d es el número de atributos
        X: Matriz de instancias de forma (n, d) donde n es el número de instancias
           
    Returns:
        Matriz de probabilidades de forma (n, c) donde cada fila suma 1
    """
    # Añadir columna de unos para el término de sesgo (bias)
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Calcular los logits para las primeras c-1 clases
    # theta tiene forma (c-1, d+1), X_bias tiene forma (n, d+1)
    # logits tendrá forma (n, c-1)
    logits = X_bias @ theta.T
    
    # Añadir logits de cero para la última clase
    logits_all = np.hstack((logits, np.zeros((X.shape[0], 1))))
    
    # Aplicar softmax para obtener probabilidades
    # Restar el máximo para estabilidad numérica
    logits_max = np.max(logits_all, axis=1, keepdims=True)
    exp_logits = np.exp(logits_all - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return probs

# 2. Función para calcular el log-likelihood
def loglikelihood(theta, X, y):
    """
    Calcula el log-likelihood para los parámetros actuales.
    
    Args:
        theta: Matriz de parámetros
        X: Matriz de instancias
        y: Vector de etiquetas de clase (valores de 0 a c-1)
        
    Returns:
        Valor del log-likelihood
    """
    probs = get_probs(theta, X)
    
    # Crear una matriz de indicadores para cada clase
    n_samples = X.shape[0]
    n_classes = probs.shape[1]
    
    # Log-likelihood = suma de log(P(y_i|x_i))
    # Seleccionamos las probabilidades de las clases correctas para cada muestra
    log_likelihood = np.sum(np.log(probs[np.arange(n_samples), y]))
    
    return log_likelihood

# 3. Función para calcular el gradiente del log-likelihood
def loglikelihoodp(theta, X, y):
    """
    Calcula el gradiente del log-likelihood con respecto a theta.
    
    Args:
        theta: Matriz de parámetros
        X: Matriz de instancias
        y: Vector de etiquetas de clase
        
    Returns:
        Gradiente del log-likelihood con respecto a theta
    """
    # Añadir columna de unos para el término de sesgo
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    probs = get_probs(theta, X)
    n_samples = X.shape[0]
    n_classes = probs.shape[1]
    n_features = X_bias.shape[1]
    
    # Inicializar el gradiente
    gradient = np.zeros_like(theta)  # (c-1, d+1)
    
    # Construir la matriz de indicadores para las etiquetas reales
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    
    # Para cada clase excepto la última (clase de referencia)
    for i in range(n_classes - 1):
        # Error para esta clase: diferencia entre la probabilidad y el indicador
        error = probs[:, i] - y_one_hot[:, i]  # (n_samples,)
        
        # Gradiente para esta clase: suma ponderada de los errores por cada característica
        gradient[i] = np.sum(error.reshape(-1, 1) * X_bias, axis=0)
    
    # Negar el gradiente porque queremos ascenso de gradiente (maximizar)
    return -gradient

# 4. Función para mostrar el límite de decisión
def show_decision_boundary(theta, X, y, ax=None):
    """
    Visualiza el límite de decisión del modelo de regresión logística.
    
    Args:
        theta: Matriz de parámetros
        X: Matriz de características (solo se utilizan las 2 primeras columnas)
        y: Vector de etiquetas de clase
        ax: Eje opcional para la figura
        
    Returns:
        El eje donde se dibujó la figura
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Determinar los límites del gráfico
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Crear una cuadrícula de 100x100 puntos
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Aplanar los puntos de la cuadrícula para predecir
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calcular probabilidades para todos los puntos de la cuadrícula
    Z_probs = get_probs(theta, grid_points)
    
    # Obtener la clase con mayor probabilidad
    Z = np.argmax(Z_probs, axis=1)
    
    # Reformar para que coincida con la forma de la cuadrícula
    Z = Z.reshape(xx.shape)
    
    # Crear un mapa de colores para las clases
    n_classes = len(np.unique(y))
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'][:n_classes])
    
    # Dibujar el límite de decisión
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Dibujar los puntos de datos con colores que corresponden a su clase
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=50)
    
    # Añadir leyenda y etiquetas
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_title('Decision Boundary')
    
    # Si hay etiquetas de clase, añadir leyenda
    unique_classes = np.unique(y)
    if hasattr(scatter, 'legend_elements'):
        legend_elements = scatter.legend_elements()[0]
        class_names = ['setosa', 'versicolor', 'virginica'][:len(unique_classes)]
        ax.legend(legend_elements, class_names, loc='best')
    
    return ax

# 5. Función para mostrar la curva de aprendizaje
def show_learning_curve(history, ax=None):
    """
    Visualiza la curva de aprendizaje (log-likelihood vs. iteraciones).
    
    Args:
        history: Lista de valores de log-likelihood por iteración
        ax: Eje opcional para la figura
        
    Returns:
        El eje donde se dibujó la figura
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history, 'b-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Learning Curve')
    ax.grid(True)
    
    return ax

# 6. Función para entrenar el modelo y generar una animación
def train_and_animate(X, y, learning_rate=0.001, n_iterations=10000, interval=100):
    """
    Entrena un modelo de regresión logística multinomial y crea una animación
    del proceso de aprendizaje.
    
    Args:
        X: Matriz de características
        y: Vector de etiquetas de clase
        learning_rate: Tasa de aprendizaje para el ascenso de gradiente
        n_iterations: Número de iteraciones
        interval: Intervalo para guardar el estado (cada cuántas iteraciones)
        
    Returns:
        Ruta al archivo GIF generado
    """
    # Número de clases y características
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    
    # Inicializar theta con ceros
    theta = np.zeros((n_classes - 1, n_features + 1))
    
    # Lista para almacenar los valores de log-likelihood
    history = []
    
    # Lista para almacenar las figuras para la animación
    frames = []
    
    # Configurar la figura para la animación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bucle de entrenamiento (ascenso de gradiente)
    for i in range(n_iterations):
        # Calcular el gradiente
        gradient = loglikelihoodp(theta, X, y)
        
        # Actualizar theta usando ascenso de gradiente
        theta += learning_rate * gradient
        
        # Calcular y guardar el log-likelihood
        ll = loglikelihood(theta, X, y)
        history.append(ll)
        
        # Guardar el estado cada 'interval' iteraciones
        if i % interval == 0 or i == n_iterations - 1:
            # Limpiar los ejes
            ax1.clear()
            ax2.clear()
            
            # Mostrar el límite de decisión
            show_decision_boundary(theta, X, y, ax=ax1)
            
            # Mostrar la curva de aprendizaje
            show_learning_curve(history, ax=ax2)
            
            # Añadir título con la iteración actual
            plt.suptitle(f'Iteration {i+1}/{n_iterations}')
            
            # Ajustar el diseño
            plt.tight_layout()
            
            # Guardar la figura actual
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            
            # Opcional: mostrar progreso
            if i % (interval * 10) == 0 or i == n_iterations - 1:
                print(f'Iteración {i+1}/{n_iterations}, Log-Likelihood: {ll:.4f}')
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    # Crear la animación y guardarla como GIF
    gif_filename = 'logistic_regression_learning.gif'
    
    # Convertir los frames a una animación
    fig, ax = plt.subplots(figsize=(15, 6))
    img = ax.imshow(frames[0])
    plt.axis('off')
    
    def update(frame):
        img.set_array(frame)
        return [img]
    
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(gif_filename, writer='pillow', fps=5)
    
    plt.close(fig)
    
    return gif_filename, theta

# Función principal para ejecutar todo el proceso
def main():
    # Cargar el dataset de Iris
    try:
        # Intenta cargar desde un archivo local
        iris = pd.read_csv('iris.csv')
    except:
        # Si no está disponible, usar el dataset de ejemplo de sklearn
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        iris = pd.DataFrame(data=np.c_[iris_data.data, iris_data.target],
                          columns=[*iris_data.feature_names, 'target'])
        iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        # Convertir los números de clase a nombres
        iris['species'] = iris['species'].map({
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        })
    
    # Usar solo las últimas dos columnas como características (petal_length y petal_width)
    X = iris[['petal_length', 'petal_width']].values
    
    # Convertir las etiquetas de clase a números (0, 1, 2)
    class_to_num = {name: i for i, name in enumerate(iris['species'].unique())}
    y = iris['species'].map(class_to_num).values
    
    print("Entrenando el modelo de regresión logística multinomial...")
    gif_file, final_theta = train_and_animate(X, y, learning_rate=0.001, n_iterations=10000, interval=100)
    
    print(f"Animación guardada como: {gif_file}")
    print("Parámetros finales del modelo (theta):")
    print(final_theta)
    
    # Mostrar el límite de decisión final
    plt.figure(figsize=(10, 6))
    show_decision_boundary(final_theta, X, y)
    plt.title("Límite de Decisión Final")
    plt.savefig("final_decision_boundary.png")
    plt.close()
    
    return

if __name__ == "__main__":
    main()