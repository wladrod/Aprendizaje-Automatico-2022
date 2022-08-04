import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def graficar_regiones_decision(X, y, clasificador, indices_prueba=None, resolucion=0.02):

    # fijar los marcadores y el mapa de colores
    marcadores = ('o', '^', 's', 'x', 'v')
    colores = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    mapa_colores = ListedColormap(colores[:len(np.unique(y))])

    # graficar la superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolucion),
                           np.arange(x2_min, x2_max, resolucion))
    Z = clasificador.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='winter')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # graficar los ejemplos de clases
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, cmap='winter',
                    edgecolor='black',
                    marker=marcadores[idx], 
                    label=cl)
    
    # resaltar ejemplos de prueba
    if indices_prueba:
        X_test, y_test = X[indices_prueba, :], y[indices_prueba]
        plt.scatter(X_test[:, 0], X_test[:, 1], edgecolor='black',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='conjunto prueba')