import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import detect_sperm_complete, draw_detections


st.set_page_config(
    page_title="Deteksi Sperma Mikroskop",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Deteksi Jumlah Sperma dari Gambar Mikroskop")
st.write(
    "Satu sperma dihitung jika sistem menemukan kepala/badan dan ekor. "
    "Black-hat enhancement dibuat lebih jelas agar objek gelap pada citra mikroskop lebih terlihat."
)

uploaded_file = st.file_uploader("Upload gambar mikroskop", type=["jpg", "jpeg", "png"])

with st.sidebar:
    st.header("⚙️ Pengaturan Algoritma")

    preset = st.selectbox(
        "Preset",
        ["Sample 16 sperma", "Manual"],
        index=0,
    )

    if preset == "Sample 16 sperma":
        default_head_kernel = 13
        default_blackhat_gain = 4.0
        default_tail_kernel = 31
        default_tail_threshold = 30
        default_min_tail_pixels = 20
        default_merge_distance = 8
    else:
        default_head_kernel = 15
        default_blackhat_gain = 4.0
        default_tail_kernel = 31
        default_tail_threshold = 30
        default_min_tail_pixels = 20
        default_merge_distance = 8

    st.subheader("Preprocessing")
    clahe_clip = st.slider("CLAHE contrast", 0.0, 5.0, 2.0, 0.1)
    blur_size = st.slider("Blur", 1, 9, 1, 2)

    st.subheader("Kepala / Badan")
    head_blackhat_kernel = st.slider("Kernel black-hat kepala", 5, 41, default_head_kernel, 2)
    blackhat_gain = st.slider("Gain tampilan black-hat", 1.0, 10.0, default_blackhat_gain, 0.5)
    threshold_mode = st.selectbox("Threshold kepala", ["otsu", "manual"], index=0)
    manual_head_threshold = st.slider("Manual threshold kepala", 1, 80, 12, 1)

    min_area = st.slider("Area minimum kepala", 1, 100, 10, 1)
    max_area = st.slider("Area maksimum kepala", 20, 600, 200, 5)
    max_aspect_ratio = st.slider("Aspect ratio maksimum", 1.0, 8.0, 4.0, 0.1)
    min_circularity = st.slider("Circularity minimum", 0.00, 1.00, 0.03, 0.01)

    st.subheader("Ekor")
    tail_blackhat_kernel = st.slider("Kernel black-hat ekor", 7, 61, default_tail_kernel, 2)
    tail_threshold = st.slider("Threshold ekor", 1, 120, default_tail_threshold, 1)
    min_tail_pixels = st.slider("Minimal piksel ekor", 0, 200, default_min_tail_pixels, 1)
    tail_search_radius = st.slider("Radius pencarian ekor", 5, 80, 30, 1)

    st.subheader("Anti duplikat")
    merge_distance = st.slider("Jarak gabung deteksi", 1, 30, default_merge_distance, 1)
    border_margin = st.slider("Abaikan tepi gambar", 0, 30, 0, 1)

    st.info(
        "Untuk sample yang kamu kirim, preset ini menargetkan 16 sperma. "
        "Jika black-hat kurang terlihat, naikkan Gain tampilan black-hat."
    )


if uploaded_file is None:
    st.warning("Silakan upload gambar mikroskop terlebih dahulu.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    detections, debug = detect_sperm_complete(
        image_bgr,
        clahe_clip=clahe_clip,
        blur_size=blur_size,
        head_blackhat_kernel=head_blackhat_kernel,
        blackhat_gain=blackhat_gain,
        threshold_mode=threshold_mode,
        manual_head_threshold=manual_head_threshold,
        min_area=min_area,
        max_area=max_area,
        max_aspect_ratio=max_aspect_ratio,
        min_circularity=min_circularity,
        tail_blackhat_kernel=tail_blackhat_kernel,
        tail_threshold=tail_threshold,
        min_tail_pixels=min_tail_pixels,
        tail_search_radius=tail_search_radius,
        merge_distance=merge_distance,
        border_margin=border_margin,
    )

    result_bgr = draw_detections(debug["working_bgr"], detections)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Hasil Deteksi")
        st.image(result_rgb, use_container_width=True)

    st.success(f"Jumlah sperma terdeteksi: {len(detections)}")

    st.subheader("🔎 Debug Visual")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.caption("Black-hat kepala/badan diperjelas")
        st.image(debug["head_blackhat_visible"], use_container_width=True, clamp=True)
    with d2:
        st.caption("Binary kepala/badan")
        st.image(debug["head_binary"], use_container_width=True, clamp=True)
    with d3:
        st.caption("Skeleton ekor")
        st.image(debug["tail_skeleton"], use_container_width=True, clamp=True)

    with st.expander("Lihat proses ekor lengkap"):
        c1, c2 = st.columns(2)
        with c1:
            st.image(debug["tail_visible"], caption="Black-hat ekor diperjelas", use_container_width=True, clamp=True)
        with c2:
            st.image(debug["tail_binary"], caption="Binary ekor", use_container_width=True, clamp=True)

    with st.expander("Data Deteksi"):
        rows = []
        for i, det in enumerate(detections, start=1):
            rows.append({
                "No": i,
                "X": round(det["x"], 1),
                "Y": round(det["y"], 1),
                "Area kepala": det["area"],
                "Tail pixels": det["tail_pixels"],
                "Circularity": round(det["circularity"], 3),
            })
        st.dataframe(rows, use_container_width=True)
