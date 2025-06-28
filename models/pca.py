from sklearn.decomposition import PCA

def pca_transform(X, n_components=2):
    model = PCA(n_components=n_components)
    X_reduced = model.fit_transform(X)
    return model, X_reduced

# 🔥 Main Controller Function
def handle_pca_models(X, n_components=2):
    """
    Handles PCA transformation.
    """
    return pca_transform(X, n_components)
