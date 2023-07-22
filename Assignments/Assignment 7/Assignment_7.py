# Họ và tên: Nguyễn Nguyên Khôi
# Mã số sinh viên: 21521009

from csv import list_dialects
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

st.title("BÀI TẬP 7 - LẬP TRÌNH GIAO DIỆN VỚI STREAMLIT")
st.header("1. Tải lên và hiển thị dataset")
uploaded_file = st.file_uploader("Chọn một file .csv")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data)

    dataframe = pd.read_csv(df)
    st.write(dataframe)

    st.header("2. Chọn output")
    output = st.selectbox("Chọn output", dataframe.columns)
    y = dataframe[[output]]

    is_categorical = False

    if y.iloc[:, 0].dtypes == object:
        is_categorical = True
        enc = LabelEncoder()
        y = enc.fit_transform(y)
        y = np.reshape(y, (-1, 1))

    st.header("3. Chọn input")
    X = dataframe.loc[:, dataframe.columns != output]

    for i, col in enumerate(X.columns):
        agree = st.checkbox(col)
        if not agree:
            X = X.drop(col, axis=1)

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

    st.write("Chọn hyper parameters")
    train_per = st.slider(
        "Chọn tỷ lệ cho tập train",
        0, 100, 80)
    st.write("Tập train chiếm ", train_per, '%')
    st.write("Tập test chiếm ", 100 - train_per, '%')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=(100 - train_per) * 0.01,
                                                        random_state=42)
    st.header("4. Chọn mô hình")
    model_selection = ["Linear Regression", "Logistic Regression"]
    if is_categorical:
        user_choice = st.selectbox("Chọn mô hình", model_selection)
    else:
        user_choice = st.selectbox("Chọn mô hình", ["Linear Regression"])
        st.write("Vì output là một giá trị liên tục nên chỉ một lựa chọn duy nhất"
                 " là Linear Regression")

    st.header("5. Hiệu quả của mô hình")

    if st.button("Run"):
        if is_categorical and user_choice == "Linear Regression":
            model = SGDClassifier()
        elif not is_categorical and user_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = LogisticRegression()

        model.fit(X_train, y_train)
        st.write("Mô hình đã được khởi tạo")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if is_categorical:
            test_score = accuracy_score(y_test_pred, y_test)
            train_score = accuracy_score(y_train_pred, y_train)
        else:
            test_score = r2_score(y_test_pred, y_test)
            train_score = r2_score(y_train_pred, y_train)

        plot_scoring = {"Tập dữ liệu": ["Train", "Test"], "Score": [train_score*100, test_score*100]}
        df_plot = pd.DataFrame(plot_scoring)

        table_scoring = {"Tập dữ liệu": ["Train", "Test"],
                         "Score": [str(round(train_score*100, 2))+"%",
                                   str(round(test_score*100, 2))+"%"]}

        df_table = pd.DataFrame(table_scoring)

        st.write(df_table)
        fig, ax = plt.subplots(figsize=(5, 5))

        sns.barplot(data=df_plot, x="Tập dữ liệu", y="Score")
        plt.grid(True)
        st.pyplot(fig)
