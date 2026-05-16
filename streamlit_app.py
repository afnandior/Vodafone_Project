import streamlit as st
from data_loader import load_data
from models.regression import handle_regression_models
from models.classification import handle_classification_models
from models.clustering import handle_clustering_models
from models.pca import handle_pca_models
from models.time_series import handle_time_series_models
from plotter import plot_results
from utils import display_results, download_results

st.set_page_config(page_title="ML Tool", layout="centered")
st.title(" AI Tool for Regression, Classification, Clustering, and Time Series")

# load_data
X, y, df = load_data()

#  identify type analysis
model_type = st.selectbox("identify type analysis:", [
    "Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
    "Logistic Regression", "Nonlinear Regression (Exponential)",
    "KMeans Clustering", "DBSCAN Clustering", "Decision Tree Classification",
    "PCA (Principal Component Analysis)", "Time Series Forecasting (ARIMA)"
])

#  run model
model, y_pred, extra = None, None, {}
if X is not None and (y is not None or model_type in ["KMeans Clustering", "DBSCAN Clustering", "PCA (Principal Component Analysis)", "Time Series Forecasting (ARIMA)"]):
    if st.button("run model"):
        if model_type in ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Nonlinear Regression (Exponential)"]:
            model, y_pred, extra = handle_regression_models(model_type, X, y)
        elif model_type in ["Logistic Regression", "Decision Tree Classification"]:
            model, y_pred, extra = handle_classification_models(model_type, X, y)
        elif model_type in ["KMeans Clustering", "DBSCAN Clustering"]:
            model, y_pred, extra = handle_clustering_models(model_type, X)
        elif model_type == "PCA (Principal Component Analysis)":
            X_reduced = handle_pca_models(X)
            extra["X_reduced"] = X_reduced
        elif model_type == "Time Series Forecasting (ARIMA)":
            y_pred = handle_time_series_models(y)

        display_results(model_type, X, y, y_pred, model, extra)
        plot_results(model_type, X, y, y_pred, extra)
        download_results(model_type, X, y_pred, extra)
