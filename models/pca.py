from sklearn.decomposition import PCA

def pca_reduction(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

# 🔥 Main Controller Function
def handle_pca(X, n_components=None):
    """
    Handles PCA dimensionality reduction.
    """
    n_components = n_components or 2
    X_reduced, pca = pca_reduction(X, n_components)
    return X_reduced
