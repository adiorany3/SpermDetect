import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import count_sperm, draw_detection, make_debug_grid


st.set_page_config(
    page_title="Deteksi Jumlah Sperma",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Deteksi Jumlah Sperma dari Gambar Mikroskop")
st.write(
    "Aplikasi ini menggunakan satu preset universal. "
    "Satu sperma dihitung jika sistem menemukan kepala/badan dan indikasi ekor."
)

uploaded_file = st.file_uploader(
    "Upload gambar mikroskop sperma",
    type=["jpg", "jpeg", "png"]
)

show_debug = st.checkbox("Tampilkan proses black-hat dan ekor", value=True)

st.info(
    "Preset dibuat satu saja agar lebih mudah digunakan. "
    "Algoritma akan menyesuaikan ukuran kernel berdasarkan ukuran gambar."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    result = count_sperm(image_rgb)
    output = draw_detection(image_rgb, result["detections"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(output, use_container_width=True)

    st.success(f"Jumlah sperma terdeteksi: {result['count']}")

    if show_debug:
        st.subheader("Debug Visual")
        st.write(
            "Black-hat kepala/badan dibuat lebih kontras. "
            "Ekor dideteksi dari garis tipis gelap lalu divalidasi terhadap kepala/badan."
        )
        debug_grid = make_debug_grid(result["debug"])
        st.image(debug_grid, use_container_width=True)

    with st.expander("Data Deteksi"):
        st.dataframe(result["table"], use_container_width=True)

else:
    st.warning("Silakan upload gambar mikroskop terlebih dahulu.")