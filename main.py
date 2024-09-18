import anndata as ad
import numpy as np
import matplotlib.pyplot as plt

#determinante
def orientacion(p, q, r):
    return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])

#punto mas bajo
def punto_mas_bajo(puntos):
    return min(puntos, key=lambda p: (p[1], p[0]))

#Graham para encontrar convexhull
def graham_scan(cluster_coords):
    #punto más bajo
    punto_base = punto_mas_bajo(cluster_coords)

    #puntos por ángulo polar respecto al punto base
    sorted_points = sorted(cluster_coords, key=lambda p: (np.arctan2(p[1] - punto_base[1], p[0] - punto_base[0]), p[1], p[0]))

    #inicializamos el stack con los primeros dos puntos
    hull = [punto_base]

    #iteramos sobre los puntos restantes y construimos el convex hull
    for p in sorted_points:
        while len(hull) > 1 and orientacion(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return np.array(hull)

#obtenemos los datos del archivo de los clusters
datos = ad.read_h5ad('celulas.h5ad')

#separamos en las coordenadas y los IDs de los clusters
coordenadas = datos.obsm['X_UMAP']
clusterIds = datos.obs['cluster_id'].copy()

#calculamos los clusters gracias a la funcion graham_scan
def convex_hulls(coordenadas, clusterIds):
    convexHulls = {}
    for clusterId in np.unique(clusterIds):
        clusterCoordenadas = coordenadas[clusterIds == clusterId]
        
        #analizar que tenga un numero de puntos validos ya que no funcionaba anteriormente
        if len(clusterCoordenadas) < 3:
            print(f'Cluster {clusterId} tiene menos de 3 puntos, se omitirá.')
            continue
        
        hull = graham_scan(clusterCoordenadas)
        #vamos guardando cada hull en un arreglo, este nos ayudara a graficar y unir los puntos de las orillas
        convexHulls[clusterId] = hull
    return convexHulls


resultado = convex_hulls(coordenadas, clusterIds)

