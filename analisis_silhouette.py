import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#import cv2
#import imutils

# Leer el dataset
data = pd.read_csv('dataset/labeled_park_coords.csv')

# Obtener los valoreside los campos
x1 = data['x1']
y1 = data['y1']
x2 = data['x2']
y2 = data['y2']
x3 = data['x3']
y3 = data['y3']
x4 = data['x4']
y4 = data['y4']


# --------------------     Grafica de coeficientes shilouette  ---------------------------------------

# Iniciar el temporizador
start_time = time.time()

# Crear un array con los puntos de los estacionamientos
parking_data = np.array([[x1[i], y1[i], x2[i], y2[i], x3[i], y3[i], x4[i], y4[i]] for i in range(len(data))])

# Definir una lista para almacenar los valores de las sumas de distancias cuadradas
sum_of_squared_distances = []

# Definir una lista para almacenar los valores de los coeficientes de silueta
silhouette_scores = []
# iteraciones_max = int(maximo_parq)
iteraciones_max = 81
# Rango de valores de K que se van a probar
k_values = range(2, iteraciones_max)
#----------------
# Iterar sobre diferentes valores de K
for k in k_values:
    # Ejecutar el algoritmo K-means
    print(" Iteracion: ", k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(parking_data)

    # Obtener la suma de distancias cuadradas
    sum_of_squared_distances.append(kmeans.inertia_)

    # Calcular el coeficiente de silueta
    silhouette_scores.append(silhouette_score(parking_data, kmeans.labels_))

# Detener el temporizador
end_time = time.time()

# Obtener el tiempo de ejecución
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")


# Add labels and title
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('CSV Data Points')
# Graficar la suma de distancias cuadradas en función de K
plt.plot(k_values, sum_of_squared_distances, 'bx-')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Suma de distancias cuadradas')
plt.title('Método del codo')
plt.legend()
plt.show()

# Graficar el coeficiente de silueta en función de K
plt.plot(k_values, silhouette_scores, 'bx-')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Coeficiente de silueta')
plt.title('Análisis de silueta')
plt.show()

max_numero_slhouette = max(silhouette_scores)
indice_max_valor = silhouette_scores.index(max_numero_slhouette)
max_numero_clusters = k_values[indice_max_valor]

print(f"El máximo valor de silhouette_scores es: {max_numero_slhouette}")
print(f"El número de clusters necesarios es: {max_numero_clusters}")
#---------------