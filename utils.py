import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

def display_results(model_type, X, y, y_pred, model, context):
    st.subheader("result")
    result_df = None

    if model_type in ["Logistic Regression", "Decision Tree Classification"]:
        st.write("  (Accuracy):", accuracy_score(y, y_pred))
        st.write(" confusion_matrix:", confusion_matrix(y, y_pred))
        result_df = pd.DataFrame({"X": X.flatten(), "Predicted": y_pred})

    elif model_type == "Nonlinear Regression (Exponential)":
        a = context.get("a")
        b = context.get("b")
        st.write(f" funcation: y = {a:.3f} * exp({b:.3f} * x)")
        st.write("MSE:", mean_squared_error(y, y_pred))
        st.write("R² Score:", r2_score(y, y_pred))
        result_df = pd.DataFrame({"X": X.flatten(), "Predicted": y_pred})

    elif model_type in ["KMeans Clustering", "DBSCAN Clustering"]:
        st.write("📌 التصنيفات:", set(y_pred))
        result_df = pd.DataFrame({"X": X.flatten(), "Cluster": y_pred})
        st.write(result_df)

    elif model_type == "PCA (Principal Component Analysis)":
        X_reduced = context.get("X_reduced")
        st.write("🔍 تم تقليل الأبعاد إلى:", X_reduced.shape[1])
        result_df = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])])
        st.write(result_df)

    elif model_type == "Time Series Forecasting (ARIMA)":
        st.line_chart(y_pred)
        st.write("📈 القيم المتوقعة:")
        result_df = pd.DataFrame(y_pred, columns=["Forecast"])
        st.write(result_df)

    else:
        if hasattr(model, 'coef_'):
            st.write("📈 المعاملات:", model.coef_)
        if hasattr(model, 'intercept_'):
            st.write("📍 الثابت:", model.intercept_)
        st.write("MSE:", mean_squared_error(y, y_pred))
        st.write("R² Score:", r2_score(y, y_pred))
        result_df = pd.DataFrame({"X": X.flatten(), "Predicted": y_pred})

    context["result_df"] = result_df

def download_results(model_type, X, y_pred, context):
    result_df = context.get("result_df")
    if result_df is not None:
        st.download_button(
            "uploded result as CSV",
            data=result_df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ no result for uploded.")
