import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

#leer los datos desde el archivo h5ad
datos = ad.read_h5ad('celulas.h5ad')

#obtener las coordenadas UMAP y los IDs de los clusters
coordenadas = datos.obsm['X_UMAP']
clusterIds = datos.obs['cluster_id'].copy()


#función para calcular los convex hulls para cada cluster usando ConvexHull
def convex_hulls(coordenadas, clusterIds):
    convexHulls = {}
    for clusterId in np.unique(clusterIds):
        clusterCoordenadas = coordenadas[clusterIds == clusterId]
        hull = ConvexHull(clusterCoordenadas)  
        convexHulls[clusterId] = hull
    return convexHulls

#calcular los convex hulls
resultado = convex_hulls(coordenadas, clusterIds)


#graficar
def grafica(coords, cluster_ids, res):
    plt.figure(figsize=(12, 10))
    uni_clusters = np.unique(cluster_ids)
    colores = plt.cm.get_cmap('tab10', len(uni_clusters))

    for i, cluster_id in enumerate(uni_clusters):
        cluster_coords = coords[cluster_ids == cluster_id]
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=5, color=colores(i), label=f'Cluster {cluster_id}')

        # Obtener el ConvexHull para el cluster
        hull = res[cluster_id]

        # Graficar los vértices del hull
        for simplex in hull.simplices:
            plt.plot(cluster_coords[simplex, 0], cluster_coords[simplex, 1], 'k-', color=colores(i))

        # Obtener el ConvexHull para el cluster
        hull = res[cluster_id]

        # Graficar los vértices del hull
        for simplex in hull.simplices:
            plt.plot(cluster_coords[simplex, 0], cluster_coords[simplex, 1], 'k-', color=colores(i))
   

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Analisis de datos con UMAP y Convex hull ', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

grafica(coordenadas, clusterIds, resultado)