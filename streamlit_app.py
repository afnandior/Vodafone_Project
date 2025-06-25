...
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

            # --- تصدير النتائج ---
            if result_df is not None:
                st.download_button("📥 تحميل النتائج كـ CSV", data=result_df.to_csv(index=False), file_name="results.csv", mime="text/csv")

            # --- رسم البيانات ---
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

                # --- تصدير الرسم كصورة ---
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button("📸 تحميل الرسم كصورة", data=buf.getvalue(), file_name="plot.png", mime="image/png")
