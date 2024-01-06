import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS
#import cv2
#import imutils

# Leer el dataset
data_read = pd.read_csv('dataset/labeled_park_coords.csv')
#Eliminar las columnas irrelevantes para el estudio
data_read = data_read.drop(['Cluster', 'x_cen', 'y_cen'], axis=1)
# Eliminar las filas que contiene datos con valor nulo
data_read = data_read.dropna()
# Seleccionar solo las columnas de interés para el dbscan
data_selected = data_read[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']]

# Iniciar el temporizador
start_time = time.time()

# Aplicar DBSCAN para identificar y etiquetar puntos de ruido
optics = OPTICS(min_samples=5, max_eps=np.inf)
labels = optics.fit_predict(data_selected)


# Detener el temporizador
end_time = time.time()

# Obtener el tiempo de ejecución
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")

# Filter out noise points
data = data_read[labels != -1]
data = data.reset_index(drop=True)

# Obtener los valores los campos
x1 = data['x1']
y1 = data['y1']
x2 = data['x2']
y2 = data['y2']
x3 = data['x3']
y3 = data['y3']
x4 = data['x4']
y4 = data['y4']
