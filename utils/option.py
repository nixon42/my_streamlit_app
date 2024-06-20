from sklearn.preprocessing import normalize
import streamlit as st
from utils.streamlit import get_replace_values

def handle_null_value(null_tab):
    if null_tab.checkbox("Drop Row With Null Value"):
        st.session_state.modified_data = st.session_state.modified_data.dropna(
            subset=st.session_state.fitur)
    if null_tab.checkbox("Replace Null Value"):
        res_dict = get_replace_values(
            st.session_state.fitur, st.session_state.X)
        st.session_state.X = st.session_state.X.fillna(
            res_dict)


def handle_normalisasi(norm_tab):
    if norm_tab.checkbox("Normalisasi L1"):
        st.session_state.X = normalize(
            st.session_state.X, norm="l1")
    if norm_tab.checkbox("Normalisasi L2"):
        st.session_state.X = normalize(
            st.session_state.X, norm="l2")
    if norm_tab.checkbox("Normalisasi Max"):
        st.session_state.X = normalize(
            st.session_state.X, norm="max")


def handle_settings(settings_tab):
    st.session_state.show_table = settings_tab.checkbox(
        "Tampilkan Tabel di preview", value=True)
    st.session_state.plot_all_fitur = settings_tab.checkbox(
        "Plot Semua Fitur", value=False)
    st.session_state.show_dataset_imbalance_plot = settings_tab.checkbox(
        "Tampilkan Plot Dataset", value=True)
    st.session_state.knn_k = settings_tab.number_input(
        label="K Value for KNN", value=5)


def handle_split_data(split_data_tab):
    st.session_state.test_size = split_data_tab.number_input(
        label="Test Size", min_value=0.0, max_value=1.0, value=0.2)
