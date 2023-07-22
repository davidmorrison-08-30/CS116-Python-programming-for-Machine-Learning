# Họ và tên: Nguyễn Nguyên Khôi
# Mã số sinh viên: 21521009

# PHIÊN BẢN STREAMLIT: 1.0

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, accuracy_score, plot_confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

def check(c, x):
    d = 0
    for i in x:
        if c != i:
            d += 1
    return d


st.title("Bài tập 9 - XGBoost")
st.header("1. Tải lên dataset")
uploaded_file = st.file_uploader("Chọn 1 file CSV")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data)

    st.header("2. Hiển thị dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)

    st.header("3. Chọn input features")
    X = dataframe
    for i in dataframe.columns:
        agree = st.checkbox(i, 1)
        if agree == False:
            X = X.drop(i, axis='columns')
    st.write(X)
    flag = 0
    for i in X.columns:
        if X[i].dtypes == object:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            flag = 1

    st.header("4. Chọn output features")
    d = 0
    y = dataframe
    if flag == 0:
        for i in dataframe.columns:
            agree_1 = False
            if check(i, X.columns) == len(X.columns):
                agree_1 = st.checkbox(i, False, str(d))
                d += 1
            if agree_1 == False:
                y = y.drop(i, axis='columns')
    else:
        for i in dataframe.columns:
            agree_1 = st.checkbox(i, 1)
            if agree_1 == False:
                y = y.drop(i, 1)
    st.write(y)
    lab_en = LabelEncoder()
    y = lab_en.fit_transform(y)

    st.header("5. Chọn hyper parameters")
    train_per = st.slider(
        'Chọn % dataset được dùng làm training',
        0, 100, 80)
    st.write('Training set chiếm ', train_per, '%')
    st.write('Testing set chiếm ', 100 - train_per, '%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    st.header("6. Chọn các mô hình")
    options = st.multiselect(
        'Mô hình',
        ['XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree'],
        ['XGBoost', 'Logistic Regression', 'SVM', 'Decision Tree'])
    st.write('Mô hình đã được chọn', options)

    print(options)

    st.header("7. Choose K-Fold Cross-validation or not")
    k_fold = st.checkbox('K-Fold Cross-validation')
    if k_fold == True:
        num = st.number_input('Insert the number of fold:')
        st.write('The number is ', num)
        num = int(num)

    if st.button('Run'):
       metrics = {'Models': [], 'F1-score': [], 'Accuracy': [], 'Log loss': []}
       if k_fold == True:
            folds = KFold(n_splits=num, shuffle=True, random_state=100)
            for i in options:
                if i == 'Logistic Regression':
                    st.subheader('Logistic Regression init')
                    lr = LogisticRegression().fit(X_train, y_train)
                    y_pred = lr.predict(X_test)
                    scores = cross_val_score(lr, X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(lr, X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(lr, X_train, y_train, scoring='neg_log_loss', cv=folds)
                    print(metrics)
                    for a in range(len(scores)):
                        metrics["Models"].append("Logistic Regression")
                        metrics["F1-score"].append(scores[a])
                        metrics["Accuracy"].append(scores_2[a])
                        metrics["Log loss"].append(abs(scores_3[a]))
                    plot_confusion_matrix(lr, X_test, y_test, display_labels=lr.classes_)
                    st.pyplot()
                elif i == 'SVM':
                    st.subheader('SVM')
                    svm = SVC().fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    scores = cross_val_score(svm, X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(svm, X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        metrics["Models"].append("SVM")
                        metrics["F1-score"].append(scores[a])
                        metrics["Accuracy"].append(scores_2[a])
                        metrics["Log loss"].append(abs(scores_3[a]))
                    plot_confusion_matrix(svm, X_test, y_test, display_labels=svm.classes_)
                    st.pyplot()
                elif i == 'Decision Tree':
                    st.subheader('Decision Tree')
                    dt = DecisionTreeClassifier().fit(X_train, y_train)
                    y_pred = dt.predict(X_test)
                    scores = cross_val_score(dt, X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(dt, X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(dt, X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        metrics["Models"].append("Decision Tree")
                        metrics["F1-score"].append(scores[a])
                        metrics["Accuracy"].append(scores_2[a])
                        metrics["Log loss"].append(abs(scores_3[a]))
                    plot_confusion_matrix(dt, X_test, y_test, display_labels=dt.classes_)
                    st.pyplot()
                elif i == 'XGBoost':
                    st.subheader('XGBoost')
                    xg = XGBClassifier().fit(X_train, y_train)
                    y_pred = xg.predict(X_test)
                    scores = cross_val_score(xg, X_train, y_train, scoring='f1_macro', cv=folds)
                    scores_2 = cross_val_score(xg, X_train, y_train, scoring='accuracy', cv=folds)
                    scores_3 = cross_val_score(xg, X_train, y_train, scoring='neg_log_loss', cv=folds)
                    for a in range(len(scores)):
                        metrics["Models"].append("XGBoost")
                        metrics["F1-score"].append(scores[a])
                        metrics["Accuracy"].append(scores_2[a])
                        metrics["Log loss"].append(abs(scores_3[a]))
                    plot_confusion_matrix(xg, X_test, y_test, display_labels=xg.classes_)
                    st.pyplot()
       else:
            for i in options:
                if i == 'Logistic Regression':
                    st.subheader('Logistic Regression init')
                    lr = LogisticRegression().fit(X_train, y_train)
                    y_pred = lr.predict(X_test)
                    f1_lr = f1_score(y_test, y_pred, average='macro')
                    log_lr = log_loss(y_test, y_pred)
                    acc_lr = accuracy_score(y_test, y_pred)
                    metrics["Models"].append("Logistic Regression")
                    metrics["F1-score"].append(f1_lr)
                    metrics["Accuracy"].append(acc_lr)
                    metrics["Log loss"].append(log_lr)
                    plot_confusion_matrix(lr, X_test, y_test, display_labels=lr.classes_)
                    st.pyplot()
                elif i == 'SVM':
                    st.subheader('SVM')
                    svm = SVC().fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    f1_svm = f1_score(y_test, y_pred, average='macro')
                    acc_svm = accuracy_score(y_test, y_pred)
                    log_svm = log_loss(y_test, y_pred)
                    metrics["Models"].append("SVM")
                    metrics["F1-score"].append(f1_svm)
                    metrics["Accuracy"].append(acc_svm)
                    metrics["Log loss"].append(log_svm)
                    plot_confusion_matrix(svm, X_test, y_test, display_labels=svm.classes_)
                    st.pyplot()
                elif i == 'Decision Tree':
                    st.subheader('Decision Tree')
                    dt = DecisionTreeClassifier().fit(X_train, y_train)
                    y_pred = dt.predict(X_test)
                    f1_dt = f1_score(y_test, y_pred, average='macro')
                    acc_dt = accuracy_score(y_test, y_pred)
                    log_dt = log_loss(y_test, y_pred)
                    metrics["Models"].append("Decision Tree")
                    metrics["F1-score"].append(f1_dt)
                    metrics["Accuracy"].append(acc_dt)
                    metrics["Log loss"].append(log_dt)
                    plot_confusion_matrix(dt, X_test, y_test, display_labels=dt.classes_)
                    st.pyplot()
                elif i == 'XGBoost':
                    st.subheader('XGBoost')
                    xg = XGBClassifier().fit(X_train, y_train)
                    y_pred = xg.predict(X_test)
                    f1_xg = f1_score(y_test, y_pred, average='macro')
                    acc_xg = accuracy_score(y_test, y_pred)
                    log_xg = log_loss(y_test, y_pred)
                    metrics["Models"].append("XGBoost")
                    metrics["F1-score"].append(f1_xg)
                    metrics["Accuracy"].append(acc_xg)
                    metrics["Log loss"].append(log_xg)
                    plot_confusion_matrix(xg, X_test, y_test, display_labels=xg.classes_)
                    st.pyplot()

       metrics = pd.DataFrame(metrics, index=options)
       st.write(metrics)
       st.bar_chart(metrics['F1-score'])
       st.bar_chart(metrics['Accuracy'])
       st.bar_chart(metrics['Log loss'])