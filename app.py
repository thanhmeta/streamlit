import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Tiêu đề ứng dụng
st.title("🌸 Dự đoán Loài Hoa Iris 🌸")

# Load dữ liệu Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Giao diện nhập dữ liệu
st.sidebar.header("🔢 Nhập thông số của hoa Iris")
sepal_length = st.sidebar.slider("Chiều dài đài hoa (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Chiều rộng đài hoa (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Chiều dài cánh hoa (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Chiều rộng cánh hoa (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Dự đoán
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Hiển thị kết quả dự đoán
st.subheader("🎯 Kết quả dự đoán")
st.write(f"Loài hoa được dự đoán: **{target_names[prediction[0]]}**")
st.write("Xác suất dự đoán:", prediction_proba)

# Hiển thị dữ liệu Iris
st.subheader("📊 Dữ liệu Iris")
df = pd.DataFrame(X, columns=feature_names)
st.write(df.head())