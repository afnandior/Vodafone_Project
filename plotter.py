import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

def plot_results(model_type, X, y, y_pred, context):
    st.subheader("📉 الرسم البياني")
    fig, ax = plt.subplots()

    if model_type == "PCA (Principal Component Analysis)":
        X_reduced = context.get("X_reduced")
        if X_reduced.shape[1] >= 2:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
    elif model_type in ["KMeans Clustering", "DBSCAN Clustering"]:
        for i in np.unique(y_pred):
            cluster_points = X[y_pred == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 0]*0, label=f"Cluster {i}")
    elif model_type != "Time Series Forecasting (ARIMA)":
        ax.scatter(X, y if y is not None else y_pred, color='blue', label='Actual Data')

        if model_type == "Polynomial Regression":
            poly = context.get("poly")
            model = context.get("model")
            X_sorted = np.sort(X, axis=0)
            y_sorted = model.predict(poly.transform(X_sorted))
            ax.plot(X_sorted, y_sorted, color='red', label='Prediction')
        elif model_type == "Nonlinear Regression (Exponential)":
            a = context.get("a")
            b = context.get("b")
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            X_sorted = np.sort(X, axis=0).flatten()
            y_sorted = exp_func(X_sorted, a, b)
            ax.plot(X_sorted, y_sorted, color='red', label='Prediction')
        else:
            ax.plot(X, y_pred, color='red', label='Prediction')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("📸 تحميل الرسم كصورة", data=buf.getvalue(), file_name="plot.png", mime="image/png")
