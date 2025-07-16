import numpy as np

# on suppose que le dataset en entrée est standardisé
def covariance_matrix_numpy(X):
    return np.cov(X, rowvar=False)

def covariance_matrix_maison(X):
    n_samples = X.shape[0]
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    return cov_matrix

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # on trie les valeurs propres et les vecteurs propres par ordre décroissant
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

def pca(X, n_components):
    cov_matrix = covariance_matrix_numpy(X)
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)
    selected_eigenvectors = eigenvectors[:, :n_components]
    X_transformed = np.dot(X, selected_eigenvectors)
    return X_transformed, eigenvalues[:n_components], selected_eigenvectors