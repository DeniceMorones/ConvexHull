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
    
    colores = plt.colormaps['tab10'](np.linspace(0, 1, len(uni_clusters)))

    all_hull_points = np.vstack([hull for hull in res.values() if len(hull) >= 3])
    
    #if len(all_hull_points) >= 3:
        #corregir para no utilizar graham
        
        #hull_of_hulls = graham_scan(all_hull_points)
        #plt.plot(np.append(hull_of_hulls[:, 0], hull_of_hulls[0, 0]),
                 #np.append(hull_of_hulls[:, 1], hull_of_hulls[0, 1]),
                 #color='black', linewidth=2, linestyle='--', label='Contorno global')

    for i, cluster_id in enumerate(uni_clusters):
        cluster_coords = coords[cluster_ids == cluster_id]
        color = colores[i]
        
        #puntos
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=10, color=color, alpha=0.6, label=f'Cluster {cluster_id}')

        #coordenadas del hull
        if cluster_id in res:
            hull_coords = res[cluster_id]
            if len(hull_coords) >= 3:
                #conectar los puntos
                hull_coords_closed = np.vstack((hull_coords, hull_coords[0]))
                plt.plot(hull_coords_closed[:, 0], hull_coords_closed[:, 1], color=color, linewidth=2)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('UMAP con Líneas de Contorno por Cluster y Contorno Global', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

grafica(coordenadas, clusterIds, resultado)