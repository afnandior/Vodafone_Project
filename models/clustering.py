from sklearn.cluster import KMeans, DBSCAN

def kmeans_clustering(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, n_init=10)
    y_pred = model.fit_predict(X)
    return model, y_pred

def dbscan_clustering(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X)
    return model, y_pred

