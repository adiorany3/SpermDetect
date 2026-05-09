import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import count_sperm, draw_detection, make_debug_grid, image_quality_report


st.set_page_config(
    page_title="Deteksi Jumlah Sperma",
    page_icon="🔬",
    layout="wide"
)

FOOTER = "Created by Galuh Adi Insani"

st.title("🔬 Deteksi Jumlah Sperma dari Gambar Mikroskop")
st.write(
    "Aplikasi ini memakai algoritma adaptif agar bisa menangani banyak kemungkinan sample. "
    "Satu sperma dihitung jika terdeteksi kepala/badan dan indikasi ekor."
    "Deteksi mungkin belum tepat jika menggunakan sample yang salah, karena identifikasi masih menggunakan bentuk (untuk menghemat GPU server), akan lebih akurat jika telah memanfaatkan machine learning."
)

with st.expander("📌 Kriteria gambar yang baik agar hasil lebih presisi", expanded=True):
    st.markdown(
        """
        **Agar deteksi lebih akurat, gunakan gambar dengan kondisi berikut:**

        1. **Fokus jelas**, kepala/badan dan ekor sperma terlihat, tidak blur.
        2. **Kontras cukup**, objek sperma lebih gelap/lebih jelas dibanding background.
        3. **Pencahayaan merata**, tidak terlalu gelap, tidak overexposure, dan tidak banyak bayangan.
        4. **Ekor terlihat**, karena sistem hanya menghitung objek yang memiliki kepala/badan + indikasi ekor.
        5. **Background bersih**, hindari banyak kotoran, gelembung, noise, atau bercak yang mirip kepala sperma.
        6. **Pembesaran konsisten**, jangan mencampur sample dengan skala yang sangat berbeda tanpa kalibrasi.
        7. **Objek tidak terlalu menumpuk**, sperma yang saling tumpang tindih sangat dekat bisa dihitung kurang/lebih.
        8. **Format gambar cukup besar**, disarankan minimal 300 px pada sisi terpendek.
        9. **Hindari kompresi berat**, gambar dari WhatsApp/screenshot buram bisa menurunkan akurasi.
        10. **Ambil beberapa bidang pandang**, lalu hitung rata-rata untuk hasil yang lebih representatif.
        """
    )

uploaded_file = st.file_uploader(
    "Upload gambar mikroskop sperma",
    type=["jpg", "jpeg", "png"]
)

show_debug = st.checkbox("Tampilkan proses deteksi", value=True)
show_quality = st.checkbox("Tampilkan analisis kualitas gambar", value=True)

st.info(
    "Mode ini tidak memakai banyak preset. Parameter utama dibuat adaptif mengikuti ukuran, kontras, "
    "dan tingkat noise gambar."
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

    if show_quality:
        st.subheader("Analisis Kualitas Gambar")
        report = image_quality_report(image_rgb)
        for item in report:
            if item["status"] == "baik":
                st.success(item["message"])
            elif item["status"] == "cukup":
                st.warning(item["message"])
            else:
                st.error(item["message"])

    if show_debug:
        st.subheader("Debug Visual")
        st.write(
            "Debug ini membantu melihat apakah black-hat kepala/badan, ekor, dan skeleton ekor sudah terbaca."
        )
        debug_grid = make_debug_grid(result["debug"])
        st.image(debug_grid, use_container_width=True)

    with st.expander("Data Deteksi"):
        st.dataframe(result["table"], use_container_width=True)

else:
    st.warning("Silakan upload gambar mikroskop terlebih dahulu.")

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:gray; font-size:14px;">
        {FOOTER}
    </div>
    """,
    unsafe_allow_html=True
)
