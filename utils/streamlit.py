from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd


def show_dataframe(df:pd.DataFrame, label):
    st.dataframe(df, use_container_width=True)


def get_replace_values(fitur, data):
    res_dict = {}
    data_type = data.dtypes
    for col in fitur:
          if data_type[col] == "int":
              res_dict[col] = st.number_input(
                  f"Pilih nilai untuk kolom {col}", value=0)
          elif data_type[col] == "float":
              res_dict[col] = st.number_input(
                  f"Pilih nilai untuk kolom {col}", value=0.0)
          else:
              res_dict[col] = st.text_input(
                  f"Isi nilai untuk kolom {col}", value="")
    return res_dict


@st.cache_data
def generate_plot(title, data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data)
    ax.set_title(title)
    return fig


@st.cache_data
def generate_imbalance_plot(y, title="Imbalance Plot"):
    unique_counts = np.unique(y, return_counts=True)
    class_names = unique_counts[0]
    class_counts = unique_counts[1]
    # relative_counts = class_counts / np.sum(class_counts)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(class_names, class_counts)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")

    return fig


@st.cache_data
def train_model(X, y, knn_k, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Multinominal Naive Bayes": MultinomialNB()
    }

    dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    results = {}
    json_serializeable = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                          index=np.unique(y),
                          columns=np.unique(y))
        cr = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model": model,
            "score": score,
            "confusion_matrix": cm,
            "classification_report": cr
        }

        json_serializeable[name] = {
            "score": score,
            "confusion_matrix": cm.to_dict(),
            "classification_report": cr
        }

    return dataset, results, json_serializeable
