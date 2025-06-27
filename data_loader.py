import streamlit as st
import numpy as np
import pandas as pd

def load_data_manual():
    x_input = st.text_input("🟡 أدخلي قيم X مفصولة بفواصل (مثلاً 1,2,3):")
    y_input = st.text_input("🟢 أدخلي قيم Y مفصولة بفواصل (مثلاً 2,4,6):")
    if x_input and y_input:
        try:
            X = np.array([float(i) for i in x_input.split(",")]).reshape(-1, 1)
            y = np.array([float(i) for i in y_input.split(",")])
            df = pd.DataFrame({"X": X.flatten(), "y": y})
            return X, y, df
        except:
            st.error("⚠️ تأكدي من إدخال أرقام صحيحة")
            return None, None, None
    return None, None, None

def load_data_file(method):
    uploaded_file = st.file_uploader("📤 حملي الملف:", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if method == "رفع ملف CSV":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("📊 البيانات:", df.head())
            x_col = st.selectbox("اختاري عمود X:", df.columns)
            y_col = st.selectbox("اختاري عمود Y:", df.columns)
            X = df[[x_col]].values
            y = df[y_col].values
            return X, y, df
        except:
            st.error("⚠️ فشل في تحميل الملف")
    return None, None, None

def load_data():
    method = st.radio("📥 اختاري طريقة تحميل البيانات:", ["Manual Input", "Upload CSV File", "Upload Excel File"])

    if method == "Manual Input":
        return load_data_manual()
    elif method == "Upload CSV File":
        return load_data_file("رفع ملف CSV")
    elif method == "Upload Excel File":
        return load_data_file("رفع ملف Excel")
    else:
        return None, None, None
