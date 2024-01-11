import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

def leer_y_preprocesar_dataset(file_path):
    data_read = pd.read_csv(file_path)
    data_read = data_read.drop(['Cluster', 'x_cen', 'y_cen'], axis=1)
    data_read = data_read.dropna()
    return data_read

def calcular_diagonal_y_area_minima(data_selected):
    x1, y1, x3, y3 = data_selected['x1'], data_selected['y1'], data_selected['x3'], data_selected['y3']
    diagonales = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    index_min_diagonal = np.argmin(diagonales)
    diagonal_min = diagonales[index_min_diagonal]

    x_min, y_min = [x1[index_min_diagonal], x3[index_min_diagonal]], [y1[index_min_diagonal], y3[index_min_diagonal]]
    print("Indice del cuadrante:", index_min_diagonal, "(x1 ; y1):", x1[index_min_diagonal], y1[index_min_diagonal],
          "(x3 ; y3):", x3[index_min_diagonal], y3[index_min_diagonal], "con diagonal:", diagonal_min)
    min_parq = (max(x_min) - min(x_min)) * (max(y_min) - min(y_min))
    print("Área de mínimo estacionamiento posible:", min_parq)
    return index_min_diagonal, min_parq

def crear_lista_puntos( data_read ):
    # Obtener los valores los campos
    x1 = data_read['x1']
    y1 = data_read['y1']
    x2 = data_read['x2']
    y2 = data_read['y2']
    x3 = data_read['x3']
    y3 = data_read['y3']
    x4 = data_read['x4']
    y4 = data_read['y4']

    points = [[x, y] for x, y in zip(x1, y1)]
    points += [[x, y] for x, y in zip(x2, y2)]
    points += [[x, y] for x, y in zip(x3, y3)]
    points += [[x, y] for x, y in zip(x4, y4)]

    return points

def calcular_area_maxima_posible(points):
    hull = ConvexHull(points)
    area_max_posible = hull.volume
    print("Área envolvente convexa:", area_max_posible)
    return area_max_posible

def analisis_silueta(data_selected, k_values):
    parking_data = data_selected[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values
    sum_of_squared_distances = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(parking_data)
        sum_of_squared_distances.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(parking_data, kmeans.labels_))

    max_numero_slhouette = max(silhouette_scores)
    indice_max_valor = silhouette_scores.index(max_numero_slhouette)
    max_numero_clusters = k_values[indice_max_valor]

    print(f"El máximo valor de silhouette_scores es: {max_numero_slhouette}")
    print(f"El número de clusters necesarios es: {max_numero_clusters}")
    return sum_of_squared_distances, silhouette_scores, max_numero_clusters

def clusterizar_csv(nombre_archivo_csv, num_cluster, columnas, data_selected):
    df_seleccionado = data_selected[columnas + ['nombre_imagen']]

    optics = OPTICS(eps=0.5, min_samples=5)
    clusters = optics.fit_predict(data_selected)

    df_seleccionado['cluster'] = clusters

    nuevo_nombre_archivo_csv = 'dataset/datos_clusterizados_OPTICS.csv'
    df_seleccionado.to_csv(nuevo_nombre_archivo_csv, index=False)

    print(f"El archivo '{nuevo_nombre_archivo_csv}' ha sido creado exitosamente con las columnas {columnas}, nombre_imagen y cluster.")
    df_seleccionado = df_seleccionado[columnas]
    return df_seleccionado, clusters



# Uso de las funciones
file_path = 'dataset/labeled_park_coords.csv'
data_read = leer_y_preprocesar_dataset(file_path)

index_min_diagonal, min_parq = calcular_diagonal_y_area_minima(data_read)

points = crear_lista_puntos(data_read)

area_max_posible = calcular_area_maxima_posible(points)

k_values = range(2, int(area_max_posible / min_parq))
sum_of_squared_distances, silhouette_scores, max_numero_clusters = analisis_silueta(data_read, k_values)

nombre_archivo_csv = 'datos.csv'
columnas = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
nuevos_datos, clusters = clusterizar_csv(nombre_archivo_csv, max_numero_clusters, columnas, data_read)
