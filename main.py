import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#calculamos la orientacion
def orientacion(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

#punto mas bajo
def punto_mas_bajo(puntos):
    return min(puntos, key=lambda p: (p[1], p[0]))

#graham para encontrar el convex
def graham_scan(cluster_coords):
    punto_base = punto_mas_bajo(cluster_coords)
    
    #ordenar por angulo
    puntos = sorted(cluster_coords, key=lambda p: (np.arctan2(p[1] - punto_base[1], p[0] - punto_base[0]), p[1], p[0]))
    
    #se inicializa con los dos primeros puntos
    hull = [puntos[0], puntos[1]]
    
    #construir convexhull
    for p in puntos[2:]:
        while len(hull) > 1 and orientacion(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return np.array(hull)

#leer el archivo .h5ad
adata = ad.read_h5ad('celulas.h5ad')
print(adata)

#obtenemos las cordenadas
umap_coords = adata.obsm['X_UMAP']
cluster_ids = adata.obs['cluster_id']