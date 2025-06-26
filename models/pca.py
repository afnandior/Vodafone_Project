from sklearn.decomposition import PCA

def run_pca(X, n_components):
    model = PCA(n_components=n_components)
    X_reduced = model.fit_transform(X)
    return model, X_reduced
