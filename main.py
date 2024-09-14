import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Leer los datos desde el archivo h5ad
datos = ad.read_h5ad('celulas.h5ad')

# Obtener las coordenadas UMAP y los IDs de los clusters
coordenadas = datos.obsm['X_UMAP']
clusterIds = datos.obs['cluster_id'].copy()


# Función para calcular los convex hulls para cada cluster usando ConvexHull
def convex_hulls(coordenadas, clusterIds):
    convexHulls = {}
    for clusterId in np.unique(clusterIds):
        clusterCoordenadas = coordenadas[clusterIds == clusterId]
        hull = ConvexHull(clusterCoordenadas)  
        convexHulls[clusterId] = hull
    return convexHulls

# Calcular los convex hulls
resultado = convex_hulls(coordenadas, clusterIds)


#Funcion para graficar los datos
def grafica(coordenadas, idsOriginales, clusterIds, resultado):
    plt.figure(figsize=(12, 10))
    clustersUnicos = np.unique(idsOriginales)
    colores = plt.cm.get_cmap('tab10', len(clustersUnicos))

    for i, clusterId in enumerate(clustersUnicos):
        clusterCoordenadas = coordenadas[idsOriginales == clusterId]
        plt.scatter(clusterCoordenadas[:, 0], clusterCoordenadas[:, 1], s=5, color=colores(i), label=f'Cluster {clusterId}')


    plt.legend()
    plt.title('Convex Hull')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

grafica(coordenadas, datos.obs['cluster_id'], clusterIds, resultado)