from sklearn.cluster import KMeans, DBSCAN

def kmeans_clustering(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, n_init=10)
    y_pred = model.fit_predict(X)
    return model, y_pred, {}

def dbscan_clustering(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X)
    return model, y_pred, {}

# 🔥 Main Controller Function
def handle_clustering_models(model_type, X):
    """
    Handles clustering model selection and execution.
    """
    if model_type == "KMeans Clustering":
        return kmeans_clustering(X)
    elif model_type == "DBSCAN Clustering":
        return dbscan_clustering(X)
    else:
        return None, None, {}
