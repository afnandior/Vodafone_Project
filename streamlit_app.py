
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Tool", layout="centered")
st.title("🧠 AI Tool for Regression, Classification, Clustering, and Time Series")

# --- اختيار طريقة إدخال البيانات ---
input_method = st.radio("اختاري طريقة إدخال البيانات:", ["يدويًا", "رفع ملف CSV", "رفع ملف Excel"])

X, y = None, None

if input_method == "يدويًا":
    x_input = st.text_input("🟡 أدخلي قيم X مفصولة بفواصل (مثلاً 1,2,3):")
    y_input = st.text_input("🟢 أدخلي قيم Y مفصولة بفواصل (مثلاً 2,4,6):")
    if x_input and y_input:
        try:
            X = np.array([float(i) for i in x_input.split(",")]).reshape(-1, 1)
            y = np.array([float(i) for i in y_input.split(",")])
        except:
            st.error("⚠️ تأكدي من إدخال أرقام صحيحة")

else:
    uploaded_file = st.file_uploader("📤 حملي الملف:", type=["csv", "xlsx"])
    if uploaded_file:
        if input_method == "رفع ملف CSV":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📊 البيانات:", df.head())
        x_col = st.selectbox("اختاري عمود X:", df.columns)
        y_col = st.selectbox("اختاري عمود Y:", df.columns)
        X = df[[x_col]].values
        y = df[y_col].values

# --- اختيار نوع التحليل ---
model_type = st.selectbox("اختاري نوع التحليل:", [
    "Linear Regression",
    "Polynomial Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet Regression",
    "Logistic Regression",
    "Nonlinear Regression (Exponential)",
    "KMeans Clustering",
    "DBSCAN Clustering",
    "Decision Tree Classification",
    "PCA (Principal Component Analysis)",
    "Time Series Forecasting (ARIMA)"
])

# --- تنفيذ النموذج ---
if X is not None and (y is not None or model_type in ["KMeans Clustering", "DBSCAN Clustering", "PCA (Principal Component Analysis)", "Time Series Forecasting (ARIMA)"]):
    run_button = st.button("تشغيل النموذج")
    if run_button:
        try:
            if model_type == "Linear Regression":
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)

            elif model_type == "Polynomial Regression":
                degree = st.number_input("📐 درجة Polynomial", min_value=2, max_value=10, value=2)
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(X_poly)

            elif model_type == "Ridge Regression":
                alpha = st.number_input("🔧 قيمة alpha:", value=1.0)
                model = Ridge(alpha=alpha).fit(X, y)
                y_pred = model.predict(X)

            elif model_type == "Lasso Regression":
                alpha = st.number_input("🔧 قيمة alpha:", value=0.1)
                model = Lasso(alpha=alpha).fit(X, y)
                y_pred = model.predict(X)

            elif model_type == "ElasticNet Regression":
                alpha = st.number_input("🔧 قيمة alpha:", value=0.1)
                l1_ratio = st.slider("⚖️ l1_ratio:", 0.0, 1.0, 0.5)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X, y)
                y_pred = model.predict(X)

            elif model_type == "Logistic Regression":
                model = LogisticRegression().fit(X, y)
                y_pred = model.predict(X)

            elif model_type == "Nonlinear Regression (Exponential)":
                def exp_func(x, a, b):
                    return a * np.exp(b * x)
                popt, _ = curve_fit(exp_func, X.flatten(), y)
                a, b = popt
                y_pred = exp_func(X.flatten(), a, b)

            elif model_type == "KMeans Clustering":
                n_clusters = st.slider("عدد المجموعات:", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, n_init=10)
                y_pred = model.fit_predict(X)

            elif model_type == "DBSCAN Clustering":
                eps = st.slider("المسافة القصوى (eps):", 0.1, 10.0, 0.5)
                min_samples = st.slider("أقل عدد نقاط لكل مجموعة:", 1, 10, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                y_pred = model.fit_predict(X)

            elif model_type == "Decision Tree Classification":
                model = DecisionTreeClassifier().fit(X, y)
                y_pred = model.predict(X)

            elif model_type == "PCA (Principal Component Analysis)":
                n_components = st.slider("عدد المكونات:", 1, min(5, X.shape[1]), 2)
                model = PCA(n_components=n_components).fit(X)
                X_reduced = model.transform(X)

            elif model_type == "Time Series Forecasting (ARIMA)":
                order = (st.number_input("AR:", 0, 5, 1), st.number_input("I:", 0, 2, 1), st.number_input("MA:", 0, 5, 1))
                model = ARIMA(y, order=order).fit()
                y_pred = model.predict(start=0, end=len(y)+5)

            # --- عرض النتائج ---
            st.subheader("📊 النتائج")
            result_df = None

            if model_type in ["Logistic Regression", "Decision Tree Classification"]:
                st.write("✅ الدقة (Accuracy):", accuracy_score(y, y_pred))
                st.write("📉 مصفوفة الالتباس:", confusion_matrix(y, y_pred))
                result_df = pd.DataFrame({"X": X.flatten(), "Predicted": y_pred})

            elif model_type == "Nonlinear Regression (Exponential)":
                st.write(f"📈 الدالة: y = {a:.3f} * exp({b:.3f} * x)")
                st.write("MSE:", mean_squared_error(y, y_pred))
                st.write("R² Score:", r2_score(y, y_pred))
                result_df = pd.DataFrame({"X": X.flatten(), "Predicted": y_pred})

            elif model_type in ["KMeans Clustering", "DBSCAN Clustering"]:
                st.write("📌 التصنيفات:", np.unique(y_pred))
                st.write("🔢 كل نقطة والمجموعة التابعة لها:")
                result_df = pd.DataFrame({"X": X.flatten(), "Cluster": y_pred})
                st.write(result_df)

            elif model_type == "PCA (Principal Component Analysis)":
                st.write("🔍 تم تقليل الأبعاد إلى:", X_reduced.shape[1])
                result_df = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])])
                st.write(result_df)

            elif model_type == "Time Series Forecasting (ARIMA)":
                st.line_chart(y_pred)
                st.write("📈 القيم المتوقعة:")
                result_df = pd.DataFrame(y_pred, columns=["Forecast"])
                st.write(result_df)

            else:
                st.write("📈 المعاملات:", model.coef_ if hasattr(model, 'coef_') else "غير متاحة")
                st.write("📍 الثابت:", model.intercept_ if hasattr(model, 'intercept_') else "غير متاح")
                st.write("MSE:", mean_squared_error(y, y_pred))
                st.write("R² Score:", r2_score(y, y_pred))
                result_df = pd.DataFrame({"X": X.flatten(), "Predicted": y_pred})

            if result_df is not None:
                st.download_button("📥 تحميل النتائج كـ CSV", data=result_df.to_csv(index=False), file_name="results.csv", mime="text/csv")

            if model_type != "Time Series Forecasting (ARIMA)":
                st.subheader("📉 الرسم البياني")
                fig, ax = plt.subplots()
                ax.scatter(X, y if y is not None else y_pred, color='blue', label='Actual Data')

                if model_type == "Polynomial Regression":
                    X_sorted = np.sort(X, axis=0)
                    y_sorted = model.predict(poly.transform(X_sorted))
                    ax.plot(X_sorted, y_sorted, color='red', label='Prediction')
                elif model_type == "Nonlinear Regression (Exponential)":
                    X_sorted = np.sort(X, axis=0).flatten()
                    y_sorted = exp_func(X_sorted, a, b)
                    ax.plot(X_sorted, y_sorted, color='red', label='Prediction')
                elif model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Logistic Regression", "Decision Tree Classification"]:
                    ax.plot(X, y_pred, color='red', label='Prediction')
                elif model_type in ["KMeans Clustering", "DBSCAN Clustering"]:
                    for i in np.unique(y_pred):
                        cluster_points = X[y_pred == i]
                        ax.scatter(cluster_points[:, 0], cluster_points[:, 0]*0, label=f"Cluster {i}")
                elif model_type == "PCA (Principal Component Analysis)":
                    if X_reduced.shape[1] >= 2:
                        ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2")

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button("📸 تحميل الرسم كصورة", data=buf.getvalue(), file_name="plot.png", mime="image/png")

        except Exception as e:
            st.error(f"❌ حصل خطأ: {e}")
