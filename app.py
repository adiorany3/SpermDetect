import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Deteksi Jumlah Sperma Mikroskop",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Deteksi dan Penghitung Jumlah Sperma dari Citra Mikroskop")
st.write(
    "Upload gambar mikroskop. Aplikasi akan mendeteksi kepala sperma sebagai objek kecil "
    "berbentuk bulat/oval menggunakan preprocessing, threshold adaptif, dan blob/contour filtering."
)
st.warning(
    "Catatan: aplikasi ini untuk eksperimen/pembelajaran computer vision, bukan alat diagnosis medis. "
    "Untuk analisis klinis tetap gunakan pemeriksaan laboratorium profesional."
)


# =============================
# Utility
# =============================

def pil_to_rgb_array(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def resize_for_display(image_rgb, max_width=1200):
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb, 1.0
    scale = max_width / w
    resized = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


# =============================
# Preprocessing
# =============================

def preprocess_image(
    image_rgb,
    invert=False,
    clahe_clip=2.0,
    blur_kernel=5,
    background_kernel=41
):
    """
    Membuat citra grayscale yang lebih mudah diproses.
    - CLAHE menaikkan kontras lokal.
    - Background subtraction membantu pada pencahayaan mikroskop yang tidak rata.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Pastikan kernel ganjil
    blur_kernel = max(3, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    background_kernel = max(15, int(background_kernel))
    if background_kernel % 2 == 0:
        background_kernel += 1

    blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
    background = cv2.GaussianBlur(enhanced, (background_kernel, background_kernel), 0)

    corrected = cv2.subtract(blurred, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    return corrected.astype(np.uint8)


# =============================
# Detection Algorithm
# =============================

def detect_sperm_heads(
    image_rgb,
    invert=False,
    clahe_clip=2.0,
    blur_kernel=5,
    background_kernel=41,
    adaptive_block=31,
    adaptive_c=2,
    min_area=8,
    max_area=250,
    min_circularity=0.25,
    min_aspect=0.35,
    max_aspect=2.8,
    min_distance=6,
    use_watershed=True
):
    """
    Deteksi utama.
    Objek yang dihitung adalah kepala sperma, karena bagian kepala lebih stabil
    untuk dihitung dibanding ekor yang tipis dan sering tidak jelas.
    """
    processed = preprocess_image(
        image_rgb=image_rgb,
        invert=invert,
        clahe_clip=clahe_clip,
        blur_kernel=blur_kernel,
        background_kernel=background_kernel
    )

    adaptive_block = max(3, int(adaptive_block))
    if adaptive_block % 2 == 0:
        adaptive_block += 1

    binary = cv2.adaptiveThreshold(
        processed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block,
        adaptive_c
    )

    # Kepala sperma biasanya menjadi area putih kecil setelah preprocessing.
    # Jika background terlalu dominan, balik otomatis.
    white_ratio = cv2.countNonZero(binary) / binary.size
    if white_ratio > 0.55:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    separated = binary.copy()

    if use_watershed:
        separated = separate_touching_objects(binary, image_rgb)

    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    h, w = processed.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / float(bh)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        if circularity < min_circularity:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            cx = x + bw // 2
            cy = y + bh // 2
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue

        candidates.append({
            "x": int(cx),
            "y": int(cy),
            "area": float(area),
            "circularity": float(circularity),
            "aspect_ratio": float(aspect),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "contour": contour
        })

    candidates = remove_near_duplicates(candidates, min_distance=min_distance)
    candidates = sorted(candidates, key=lambda item: (item["y"], item["x"]))

    return candidates, processed, binary, separated


# =============================
# Watershed Separation
# =============================

def separate_touching_objects(binary, image_rgb):
    """
    Memisahkan objek kepala sperma yang saling menempel menggunakan distance transform + watershed.
    """
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return binary

    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(image_bgr, markers)

    separated = np.zeros_like(binary)
    separated[markers > 1] = 255

    separated = cv2.morphologyEx(separated, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return separated


# =============================
# Duplicate Filtering
# =============================

def remove_near_duplicates(candidates, min_distance=6):
    """
    Menghapus titik deteksi yang terlalu dekat.
    Kandidat dengan area lebih besar diprioritaskan.
    """
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)
    final = []

    for candidate in sorted_candidates:
        cx, cy = candidate["x"], candidate["y"]
        duplicate = False

        for accepted in final:
            ax, ay = accepted["x"], accepted["y"]
            distance = np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)
            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            final.append(candidate)

    return final


# =============================
# Drawing
# =============================

def draw_detections(image_rgb, candidates):
    output = image_rgb.copy()

    for idx, item in enumerate(candidates, start=1):
        x, y = item["x"], item["y"]
        bx, by, bw, bh = item["bbox"]

        radius = max(4, int(max(bw, bh) / 2))

        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), -1)
        cv2.putText(
            output,
            str(idx),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output


# =============================
# Sidebar Controls
# =============================

uploaded_file = st.file_uploader(
    "Upload gambar mikroskop sperma",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")

    preset = st.selectbox(
        "Preset",
        [
            "Standar mikroskop",
            "Objek kecil dan padat",
            "Objek besar/jelas",
            "Citra gelap - perlu invert"
        ]
    )

    if preset == "Standar mikroskop":
        default_invert = False
        default_min_area = 8
        default_max_area = 250
        default_circularity = 0.25
        default_block = 31
        default_c = 2
        default_bg = 41
        default_dist = 6
    elif preset == "Objek kecil dan padat":
        default_invert = False
        default_min_area = 4
        default_max_area = 120
        default_circularity = 0.18
        default_block = 25
        default_c = 1
        default_bg = 35
        default_dist = 4
    elif preset == "Objek besar/jelas":
        default_invert = False
        default_min_area = 20
        default_max_area = 600
        default_circularity = 0.30
        default_block = 41
        default_c = 3
        default_bg = 61
        default_dist = 8
    else:
        default_invert = True
        default_min_area = 8
        default_max_area = 250
        default_circularity = 0.25
        default_block = 31
        default_c = 2
        default_bg = 41
        default_dist = 6

    invert = st.checkbox("Balik warna / invert", value=default_invert)

    clahe_clip = st.slider("Kontras lokal / CLAHE", 0.5, 5.0, 2.0, 0.1)
    blur_kernel = st.slider("Blur kernel", 3, 15, 5, 2)
    background_kernel = st.slider("Koreksi background", 15, 101, default_bg, 2)

    adaptive_block = st.slider("Adaptive threshold block", 11, 81, default_block, 2)
    adaptive_c = st.slider("Adaptive threshold C", -10, 15, default_c, 1)

    min_area = st.slider("Luas minimum kepala sperma", 1, 200, default_min_area, 1)
    max_area = st.slider("Luas maksimum kepala sperma", 20, 1500, default_max_area, 5)

    min_circularity = st.slider("Circularity minimum", 0.05, 1.00, default_circularity, 0.01)
    min_aspect = st.slider("Aspect ratio minimum", 0.10, 1.00, 0.35, 0.05)
    max_aspect = st.slider("Aspect ratio maksimum", 1.00, 5.00, 2.80, 0.05)

    min_distance = st.slider("Jarak minimum antar deteksi", 1, 30, default_dist, 1)
    use_watershed = st.checkbox("Pisahkan objek menempel / watershed", value=True)

    st.info(
        "Yang dihitung adalah kepala sperma. Jika hasil terlalu sedikit, turunkan luas minimum, "
        "circularity, atau adaptive C. Jika terlalu banyak noise, naikkan luas minimum/circularity."
    )


# =============================
# Main App
# =============================

if uploaded_file is None:
    st.warning("Silakan upload gambar mikroskop terlebih dahulu.")
    st.stop()

image_rgb = pil_to_rgb_array(uploaded_file)

candidates, processed, binary, separated = detect_sperm_heads(
    image_rgb=image_rgb,
    invert=invert,
    clahe_clip=clahe_clip,
    blur_kernel=blur_kernel,
    background_kernel=background_kernel,
    adaptive_block=adaptive_block,
    adaptive_c=adaptive_c,
    min_area=min_area,
    max_area=max_area,
    min_circularity=min_circularity,
    min_aspect=min_aspect,
    max_aspect=max_aspect,
    min_distance=min_distance,
    use_watershed=use_watershed
)

output = draw_detections(image_rgb, candidates)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gambar Asli")
    st.image(image_rgb, use_container_width=True)

with col2:
    st.subheader("Hasil Deteksi")
    st.image(output, use_container_width=True)

st.success(f"Jumlah sperma terdeteksi: {len(candidates)}")

with st.expander("Lihat Tahapan Preprocessing"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(processed, caption="Grayscale + kontras + background correction", use_container_width=True)
    with c2:
        st.image(binary, caption="Binary threshold", use_container_width=True)
    with c3:
        st.image(separated, caption="Setelah pemisahan objek", use_container_width=True)

with st.expander("Data Deteksi"):
    rows = []
    for i, item in enumerate(candidates, start=1):
        rows.append({
            "No": i,
            "X": item["x"],
            "Y": item["y"],
            "Area": round(item["area"], 2),
            "Circularity": round(item["circularity"], 3),
            "Aspect Ratio": round(item["aspect_ratio"], 3)
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download data CSV",
        data=csv,
        file_name="hasil_deteksi_sperma.csv",
        mime="text/csv"
    )
