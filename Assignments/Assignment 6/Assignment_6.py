# Họ và tên: Nguyễn Nguyên Khôi
# Mã số sinh viên: 21521009
''' Các thư viện cần thiết cho việc chạy chương trình
    có thể được cài đặt bng câu lệnh --pip install requirements.txt--
'''
# Gõ lệnh streamlit run 21521009.py để chạy chương trình

from csv import list_dialects
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

st.title("BÀI TẬP 6 - CLASSIFICATION VỚI LOGISTIC REGRESSION")
st.header("1. Tải lên dataset")
uploaded_file = st.file_uploader("Chọn 1 file .csv")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data)

    st.header("2. Hiển thị dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)

    st.header("3.Chọn các đặc trưng dùng cho train Logistic Regression")
    X = dataframe.iloc[:, :-1]
    for i in X.columns:
        agree = st.checkbox(i)
        if agree == False:
            X = X.drop(i, 1)
    st.write(X)

    st.header("3.1. Outputs")
    y = dataframe.iloc[:, -1]
    st.write(y)

    st.header("4. Chia dataset")
    train_per = st.slider(
        'Phần trăm tập train',
        0, 100, 80)
    st.write('Tập train chiếm ', train_per, '%')
    st.write('Tập test chiếm ', 100 - train_per, '%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    st.header("5. Đánh giá model dựa trên độ đo F1-score")

    pipeline_2 = make_pipeline(StandardScaler(), LogisticRegression())
    pipeline_1 = LogisticRegression()
    df_me = pd.DataFrame(columns=['Data chưa chuẩn hóa', 'Data đã chuẩn hóa'])

    pipeline_1.fit(X_train, y_train)
    y_pred_1 = pipeline_1.predict(X_test)
    f1_1 = f1_score(y_test, y_pred_1)

    pipeline_2.fit(X_train, y_train)
    y_pred_2 = pipeline_2.predict(X_test)
    f1_2 = f1_score(y_test, y_pred_2)

    df_me = df_me.append({'Data chưa chuẩn hóa': f1_1, 'Data đã chuẩn hóa': f1_2}, ignore_index=True)


    st.write(df_me)

    st.header("6. Kết luận")

    st.write("Với độ đo f1-score và dataset Social_Network_Ads.csv. Nếu sử dụng cả 2 đặc trưng "
             "là Age và EstimatedSalary hoặc chỉ EstimatedSalary thì hiệu quả của model Logistic Regression"
             "trên data đã chuẩn hóa tốt hơn hẳn so với data chưa chuẩn hóa. Còn nếu chỉ sử dụng đặc trưng Age "
             "thì hiệu quả của model trên data đã chuẩn hóa và chưa chuẩn hóa là gần như bằng nhau")