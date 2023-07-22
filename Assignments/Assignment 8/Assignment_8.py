# Họ và tên: Nguyễn Nguyên Khôi
# Mã số sinh viên: 21521009

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA

st.title("BÀI TẬP 8 - PHÂN LOẠI VỚI ĐẶC TRƯNG ĐÃ GIẢM SỐ CHIỀU SỬ DỤNG PCA")
st.header("1. Tải lên dataset")
uploaded_file = st.file_uploader("Chọn một file CSV")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data)

    st.header("2. Hiển thị dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)

    st.header("3. Chọn output")
    output = st.selectbox("Chọn output", dataframe.columns)
    y = dataframe[[output]]

    is_categorical = False

    if y.iloc[:, 0].dtypes == object:
        is_categorical = True
        enc = LabelEncoder()
        y = enc.fit_transform(y)
        y = np.reshape(y, (-1, 1))

    X = dataframe.loc[:, dataframe.columns != output]

    col_to_be_transformed = []
    flag = 0
    for i in range(len(list(X.columns))):
        if X.iloc[:, i].dtypes == object:
            col_to_be_transformed.append(i)
            flag = 1

    if flag == 1:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), col_to_be_transformed)],
                               remainder='passthrough')
        X = ct.fit_transform(X)

    st.header("4. Chia dataset")
    train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
    st.write('The training dataset is ', train_per, '%')
    st.write('Therefore, the test dataset is ', 100 - train_per, '%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    st.header("5. Giảm số chiều bằng PCA")
    dim_re = st.radio(
        "Bạn có muốn giảm số chiều của X không?",
        ('Yes', 'No'))
    if dim_re == 'Yes':
        n_components = st.number_input("Nhập số chiều:")
        st.write('The number of dimensions is ', n_components)
        n_components = int(n_components)
        pca = PCA(n_components=n_components)
        X_train_ = pca.fit_transform(X_train)
        cc = 1

    st.header("6. Hiệu quả của model")

    if st.button("Run"):
        st.write("Logistic Regression được sử dụng trong bài này")
        if cc == 1:
            X_test_ = pca.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train_, y_train)
        y_train_pred = model.predict(X_train_)
        y_test_pred = model.predict(X_test_)

        acc_test_score = accuracy_score(y_test_pred, y_test)
        acc_train_score = accuracy_score(y_train_pred, y_train)
        f1_test_score = f1_score(y_test_pred, y_test, average="micro")
        f1_train_score = f1_score(y_train_pred, y_train, average="micro")

        plot_scoring = {"Tập dữ liệu": ["Train", "Test"],
                        "Accuracy ": [acc_train_score * 100, acc_test_score * 100],
                        "F1_score ": [f1_train_score, f1_test_score]}
        df_plot = pd.DataFrame(plot_scoring)

        table_scoring = {"Tập dữ liệu": ["Train", "Test"],
                         "Accuracy": [str(round(acc_train_score * 100, 2)) + "%",
                                   str(round(acc_test_score * 100, 2)) + "%"],
                         "F1_score ": [str(round(f1_train_score, 2)), str(round(f1_test_score, 2))]}

        df_table = pd.DataFrame(table_scoring)

        st.write(df_table)
        fig, ax = plt.subplots(figsize=(5, 5))

    st.header("7. Hiệu quả của model với các số chiều khác nhau từ 13 xuống 1")

    acc_scoring = []

    for i in range(1, 14):
        pca = PCA(n_components=i)
        X_train_ = pca.fit_transform(X_train)
        X_test_ = pca.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train_, y_train)
        y_test_pred = model.predict(X_test_)

        acc_test_score = accuracy_score(y_test_pred, y_test)
        acc_scoring.append(acc_test_score)

    # st.bar_chart(acc_scoring["Accuracy"], x_axis_values=[str(i) for i in range(1, 14)])

    fig, ax = plt.subplots()
    ax.bar([str(i) for i in range(1, 14)], acc_scoring)
    ax.set_xlabel("Số chiều")
    ax.set_ylabel("Accuracy")

    st.pyplot(fig)
