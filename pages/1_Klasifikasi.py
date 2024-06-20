import json
import streamlit as st
import pandas as pd
from utils.option import handle_normalisasi, handle_null_value, handle_settings, handle_split_data
from utils.streamlit import generate_imbalance_plot, generate_plot, show_dataframe, get_replace_values, train_model
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Klasifikasi",
    page_icon="🧊",
)


def handle_uploaded_file():
    uploaded_file = st.file_uploader(
        "Masukan Dataset anda!", type="csv", on_change=lambda: st.session_state.clear())

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.modified_data = st.session_state.data
        st.session_state.label = st.selectbox(
            label="Pilih Label", options=st.session_state.data.columns, index=None)

        if not st.session_state.label:
            st.warning("Label tidak boleh kosong!")

        all_columns = st.session_state.data.columns.tolist()
        if st.session_state.label is not None:
            all_columns.remove(st.session_state.label)

        st.session_state.fitur = st.multiselect(
            label="Pilih Fitur yang ingin digunakan", options=all_columns, default=all_columns)

        if not st.session_state.fitur:
            st.warning("Fitur tidak boleh kosong!")

        if st.session_state.label is not None:
            st.session_state.X = st.session_state.modified_data[st.session_state.fitur]
            st.session_state.y = st.session_state.modified_data[st.session_state.label]


def preprocess_data():
    if st.session_state.label is not None:
        null_tab, norm_tab, split_data_tab, settings_tab = st.tabs(
            ["Null Value", "Normalisasi", "Split Data", "Pengaturan"])

        handle_null_value(null_tab)
        handle_normalisasi(norm_tab)
        handle_split_data(split_data_tab)
        handle_settings(settings_tab)

    else:
        st.warning("Silahkan pilih label dan fitur terlebih dahulu!")



def show_train_result(train_tab, name, value):
    with train_tab.status(f"Training Model {name}..."):
        # st.write(f"{name}")
        st.write(f"Score : ")
        st.write(value['score'])

        st.write(f"Confusion Matrix : ")
        st.write(value['confusion_matrix'])

        st.write(f"Classification Report : ")
        st.dataframe(value['classification_report'])


@st.experimental_fragment
def download_button(tab, json_serializeable):
    tab.download_button("Download Hasil", data=json.dumps(
        json_serializeable, indent=4), file_name="klasifikasi_model.json")


def train_model_tab(train_tab):
    if train_tab.button("Train Model"):
        # with st.status("Training Model..."):
        dataset, model, json_serializeable = train_model(st.session_state.X,
                                                st.session_state.y, st.session_state.knn_k, st.session_state.test_size)

        download_button(train_tab, json_serializeable)

        fig, ax = plt.subplots()
        for name, value in model.items():
            ax.bar(x=name, height=value['score'], label=name)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Score Model")
        train_tab.pyplot(fig)

        if st.session_state.show_dataset_imbalance_plot:
            with train_tab.status("Plot Dataset Imbalance..."):
                fig = generate_imbalance_plot(dataset['y_train'], title="keseimbangan y_train")
                st.pyplot(fig)

                fig = generate_imbalance_plot(dataset['y_test'], title="keseimbangan y_test")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.bar(x="y_train", height=dataset['y_train'].value_counts(), label="y_train")
                ax.bar(x="y_test", height=dataset['y_test'].value_counts(), label="y_test")
                ax.set_xlabel("Dataset")
                ax.set_ylabel("Count")
                ax.set_title("Ratio Dataset")
                st.pyplot(fig)

        for name, value in model.items():
            show_train_result(train_tab, name, value)


def data_tabs():
    imp_data_tab, after_prep_tab, train_tab = st.tabs(
        ["Data Awal", "Setelah Preprocessing", "Train Model"])

    handle_data_tabs(imp_data_tab, after_prep_tab, train_tab)


def handle_data_tabs(imp_data_tab, after_prep_tab, train_tab):
    if "X" in st.session_state:
        raw_data = st.session_state.data
        X = pd.DataFrame(st.session_state.X,
                         columns=st.session_state.fitur)
        column = st.session_state.fitur

        handle_imported_data_tab(imp_data_tab, raw_data, column)
        handle_after_preprosesing_tab(after_prep_tab, X, column)
        train_model_tab(train_tab)


def handle_imported_data_tab(imp_data_tab, data, column):
    if st.session_state.show_table:
        imp_data_tab.dataframe(data, use_container_width=True)
        imp_data_tab.write(f"Jumlah Baris : {len(data)}")
        imp_data_tab.write(f"Null Value : {data.isnull().sum().sum()}")

    jenis_data_tab, imbalance_tab, plot_data_tab = imp_data_tab.tabs(
        ["Jenis Data", "Imbalance", "Plot Data"])
    handle_jenis_data_tab(jenis_data_tab, data)
    handle_imbalance_tab(imbalance_tab, data[st.session_state.label])
    handle_plot_data_tab(plot_data_tab, data, column)


def handle_after_preprosesing_tab(after_prep_tab, data, column):
    if "X" in st.session_state:
        if st.session_state.show_table:
            after_prep_tab.dataframe(data, use_container_width=True)
            after_prep_tab.write(f"Jumlah Baris : {len(data)}")
            after_prep_tab.write(
                f"Null Value : {data.isnull().sum().sum()}")

        jenis_data_tab, imbalance_tab, plot_data_tab = after_prep_tab.tabs(
            ["Jenis Data", "Imbalance", "Plot Data"])
        handle_jenis_data_tab(jenis_data_tab, data)
        handle_imbalance_tab(imbalance_tab, st.session_state.y)
        handle_plot_data_tab(plot_data_tab, data, column)


def handle_jenis_data_tab(tab, data):
    tab.write(data.dtypes)


def handle_imbalance_tab(tab, y):
    tab.pyplot(generate_imbalance_plot(y))


def handle_plot_data_tab(tab, data, column):
    if st.session_state.plot_all_fitur:
        for col in data.columns:
            fig = generate_plot(col, data[col])
            tab.pyplot(fig)
    else:
        column_to_plot = tab.selectbox(
            key=f"fitur_to_plot_{tab.__str__()}",
            label="Pilih Fitur yang ingin Plot", options=column)
        fig = generate_plot(column_to_plot, data[column_to_plot])
        tab.pyplot(fig)


if "label" not in st.session_state:
    st.session_state.label = None

st.write("# Klasifikasi")
st.write("Bandingkan peforma beberapa algoritma klasifikasi, upload dataset dan mulai bandingkan performa!")
handle_uploaded_file()
preprocess_data()
data_tabs()
