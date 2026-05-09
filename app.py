import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detector import detect_sperm_heads, draw_detections


st.set_page_config(
    page_title="Deteksi Jumlah Sperma Mikroskop",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Deteksi Jumlah Sperma dari Gambar Mikroskop")
st.write(
    "Upload gambar mikroskop. Sistem akan menghitung jumlah sperma berdasarkan "
    "deteksi kepala sperma. Gunakan slider di sidebar untuk menyesuaikan jenis gambar."
)

st.warning(
    "Catatan: aplikasi ini untuk eksperimen/pembelajaran computer vision, bukan alat diagnosis medis."
)


def pil_to_bgr(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")

    preset = st.selectbox(
        "Preset",
        [
            "Sperm sample 16",
            "Umum - objek kecil",
            "Umum - objek besar",
            "Manual"
        ]
    )

    if preset == "Sperm sample 16":
        default = {
            "blackhat_kernel": 25,
            "clahe_clip": 0.0,
            "blur_size": 1,
            "threshold_mode": "otsu",
            "manual_threshold": 18,
            "morph_open": False,
            "min_area": 15,
            "max_area": 250,
            "min_width": 5,
            "min_height": 5,
            "max_width_obj": 25,
            "max_height_obj": 25,
            "max_aspect_ratio": 3.0,
            "min_circularity": 0.05,
            "border_margin": 0,
            "merge_distance": 8,
        }
    elif preset == "Umum - objek kecil":
        default = {
            "blackhat_kernel": 19,
            "clahe_clip": 0.0,
            "blur_size": 1,
            "threshold_mode": "otsu",
            "manual_threshold": 16,
            "morph_open": False,
            "min_area": 8,
            "max_area": 150,
            "min_width": 3,
            "min_height": 3,
            "max_width_obj": 22,
            "max_height_obj": 22,
            "max_aspect_ratio": 3.2,
            "min_circularity": 0.08,
            "border_margin": 0,
            "merge_distance": 6,
        }
    else:
        default = {
            "blackhat_kernel": 31,
            "clahe_clip": 0.0,
            "blur_size": 1,
            "threshold_mode": "otsu",
            "manual_threshold": 20,
            "morph_open": False,
            "min_area": 20,
            "max_area": 400,
            "min_width": 5,
            "min_height": 5,
            "max_width_obj": 35,
            "max_height_obj": 35,
            "max_aspect_ratio": 3.5,
            "min_circularity": 0.08,
            "border_margin": 0,
            "merge_distance": 10,
        }

    blackhat_kernel = st.slider("Ukuran filter kepala", 7, 51, default["blackhat_kernel"], 2)
    clahe_clip = st.slider("Kontras CLAHE", 1.0, 5.0, default["clahe_clip"], 0.1)
    blur_size = st.slider("Blur", 1, 9, default["blur_size"], 2)

    threshold_mode = st.radio(
        "Mode threshold",
        ["otsu", "manual"],
        index=0 if default["threshold_mode"] == "otsu" else 1
    )

    manual_threshold = st.slider("Nilai threshold manual", 1, 80, default["manual_threshold"], 1)
    morph_open = st.checkbox("Bersihkan noise kecil / morphological open", value=default["morph_open"])

    st.subheader("Filter ukuran")
    min_area = st.slider("Area minimum", 1, 200, default["min_area"], 1)
    max_area = st.slider("Area maksimum", 20, 1000, default["max_area"], 5)
    min_width = st.slider("Lebar minimum", 1, 20, default["min_width"], 1)
    min_height = st.slider("Tinggi minimum", 1, 20, default["min_height"], 1)
    max_width_obj = st.slider("Lebar maksimum", 5, 80, default["max_width_obj"], 1)
    max_height_obj = st.slider("Tinggi maksimum", 5, 80, default["max_height_obj"], 1)

    st.subheader("Filter bentuk")
    max_aspect_ratio = st.slider("Rasio lonjong maksimum", 1.0, 8.0, default["max_aspect_ratio"], 0.1)
    min_circularity = st.slider("Kebulatan minimum", 0.00, 1.00, default["min_circularity"], 0.01)
    merge_distance = st.slider("Jarak gabung deteksi ganda", 0, 30, default["merge_distance"], 1)
    border_margin = st.slider("Abaikan tepi gambar", 0, 30, default["border_margin"], 1)

    st.info(
        "Untuk gambar contoh yang kamu kirim, gunakan preset 'Sperm sample 16'. "
        "Targetnya 16 sperma."
    )


uploaded_file = st.file_uploader(
    "Upload gambar mikroskop",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    image_bgr = pil_to_bgr(uploaded_file)

    detections, enhanced, binary, working_bgr, scale = detect_sperm_heads(
        image_bgr,
        blackhat_kernel=blackhat_kernel,
        clahe_clip=clahe_clip,
        blur_size=blur_size,
        threshold_mode=threshold_mode,
        manual_threshold=manual_threshold,
        morph_open=morph_open,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_height=min_height,
        max_width_obj=max_width_obj,
        max_height_obj=max_height_obj,
        max_aspect_ratio=max_aspect_ratio,
        min_circularity=min_circularity,
        border_margin=border_margin,
        merge_distance=merge_distance,
    )

    output_bgr = draw_detections(working_bgr, detections)

    st.success(f"Jumlah sperma terdeteksi: {len(detections)}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(bgr_to_rgb(working_bgr), use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(bgr_to_rgb(output_bgr), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Enhanced / Black-hat")
        st.image(enhanced, use_container_width=True, clamp=True)

    with col4:
        st.subheader("Mask Kepala Sperma")
        st.image(binary, use_container_width=True, clamp=True)

    with st.expander("Data titik deteksi"):
        rows = []
        for i, det in enumerate(detections, start=1):
            rows.append({
                "No": i,
                "X": round(det["x"], 2),
                "Y": round(det["y"], 2),
                "Lebar": det["w"],
                "Tinggi": det["h"],
                "Area": det["area"],
                "Aspect": round(det["aspect"], 3),
                "Circularity": round(det["circularity"], 3)
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

else:
    st.info("Upload gambar mikroskop terlebih dahulu.")
