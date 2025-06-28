from sklearn.cluster import KMeans, DBSCAN

# KMeans Clustering
def kmeans_clustering(X, n_clusters):
    model = KMeans(n_clusters=n_clusters).fit(X)
    y_pred = model.labels_
    return model, y_pred

# DBSCAN Clustering
def dbscan_clustering(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    y_pred = model.labels_
    return model, y_pred

# 🔥 Main Controller Function
def handle_clustering_models(model_type, X, n_clusters=None, eps=None, min_samples=None):
    """
    Handles clustering model selection and execution.
    """
    if model_type == "KMeans Clustering":
        n_clusters = n_clusters or 3
        model, y_pred = kmeans_clustering(X, n_clusters)
        return model, y_pred, {}
    
    elif model_type == "DBSCAN Clustering":
        eps = eps or 0.5
        min_samples = min_samples or 5
        model, y_pred = dbscan_clustering(X, eps, min_samples)
        return model, y_pred, {}
    
    else:
        return None, None, {}

