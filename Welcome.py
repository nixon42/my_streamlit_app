import time
import streamlit as st

st.set_page_config(
    page_title="Auto Data Mining",
    page_icon="🧊",
)
st.write("# Selamat Datang! 👋")
st.text("Aplikasi Data mining dari kelompok 3D, menggunakan sklearn dan streamlit.")
st.markdown("Kelompok 3D :")
st.markdown("- Alief Cahyo Utomo (2113030101)")
st.markdown("- Agastya Andresangsa (2113030081)")
st.markdown("- Irfan Eka Nanda (2113030040)")


klasifikasi_tab, regresi_tab = st.tabs(["Klasifikasi", "Regresi"])
with klasifikasi_tab:
    st.write("""
    Bandingkan peforma beberapa algoritma klasifikasi, upload dataset dan mulai bandingkan performa!
    #### fitur
        - handle null value
        - normalisasi
        - split data
        - plot dataset
             
    #### Penggunaan
        - Upload dataset
        - Pilih label
        - Pilih fitur
        - handle null value (opsional)
        - handle normalisasi (opsional)
        - handle split data (opsional)
        - train model
    """)
with regresi_tab:
    st.markdown("# On Development")
    with st.spinner("Tunggu ya..."):
        time.sleep(3)

st.sidebar.success("Silahkan Pilih jenis Pendekatan.")
