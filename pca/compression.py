from algorithm import pca
import numpy as np

# On considère que le dataset en entrée est standardisé
def compress_data(X, n_components):
    return pca(X, n_components)

def decompress_data(X_transformed, eigenvectors):
    return np.dot(X_transformed, eigenvectors.T)
   