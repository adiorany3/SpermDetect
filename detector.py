import cv2
import numpy as np


def resize_keep_aspect(image_bgr, max_width=900):
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr, 1.0
    scale = max_width / float(w)
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_gray(image_bgr, clahe_clip=0.0, blur_size=1):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if float(clahe_clip) > 0:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

    blur_size = int(blur_size)
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size >= 3:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    return gray


def remove_border_components(stats, border_margin, width, height, idx):
    x, y, w, h, area = stats[idx]
    return (
        x <= border_margin or
        y <= border_margin or
        x + w >= width - border_margin or
        y + h >= height - border_margin
    )


def merge_nearby_candidates(candidates, merge_distance=8):
    if not candidates:
        return []

    merge_distance = float(merge_distance)
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    final = []

    for cand in candidates:
        duplicate = False

        for kept in final:
            distance = np.sqrt((cand["x"] - kept["x"]) ** 2 + (cand["y"] - kept["y"]) ** 2)
            if distance < merge_distance:
                duplicate = True
                break

        if not duplicate:
            final.append(cand)

    return final


def detect_sperm_heads(
    image_bgr,
    max_width=900,
    blackhat_kernel=25,
    clahe_clip=0.0,
    blur_size=1,
    threshold_mode="otsu",
    manual_threshold=18,
    morph_open=False,
    min_area=15,
    max_area=250,
    min_width=5,
    min_height=5,
    max_width_obj=25,
    max_height_obj=25,
    max_aspect_ratio=3.0,
    min_circularity=0.05,
    border_margin=0,
    merge_distance=8,
):
    """
    Deteksi jumlah sperma berdasarkan kepala sperma.

    Prinsip:
    1. Kepala sperma pada gambar mikroskop umumnya berupa objek oval/spot gelap.
    2. Black-hat morphology menonjolkan objek gelap kecil dan meredam background.
    3. Connected component + filter ukuran/bentuk dipakai agar ekor yang tipis tidak ikut dihitung.
    """
    working_bgr, scale = resize_keep_aspect(image_bgr, max_width=max_width)
    gray = preprocess_gray(working_bgr, clahe_clip=clahe_clip, blur_size=blur_size)

    blackhat_kernel = int(blackhat_kernel)
    if blackhat_kernel % 2 == 0:
        blackhat_kernel += 1
    blackhat_kernel = max(3, blackhat_kernel)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (blackhat_kernel, blackhat_kernel)
    )
    enhanced = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    if threshold_mode == "manual":
        _, binary = cv2.threshold(
            enhanced,
            int(manual_threshold),
            255,
            cv2.THRESH_BINARY
        )
    else:
        _, binary = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if morph_open:
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h_img, w_img = binary.shape
    candidates = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]

        if border_margin > 0 and remove_border_components(stats, border_margin, w_img, h_img, idx):
            continue

        if area < min_area or area > max_area:
            continue

        if w < min_width or h < min_height:
            continue

        if w > max_width_obj or h > max_height_obj:
            continue

        aspect = max(w / float(h), h / float(w))
        if aspect > max_aspect_ratio:
            continue

        component_mask = (labels[y:y+h, x:x+w] == idx).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * float(area) / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        cx, cy = centroids[idx]

        candidates.append({
            "x": float(cx),
            "y": float(cy),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "aspect": float(aspect),
            "circularity": float(circularity),
            "score": float(area * circularity)
        })

    candidates = merge_nearby_candidates(candidates, merge_distance=merge_distance)
    candidates = sorted(candidates, key=lambda p: (p["y"], p["x"]))

    return candidates, enhanced, binary, working_bgr, scale


def draw_detections(image_bgr, detections):
    output = image_bgr.copy()

    for i, det in enumerate(detections, start=1):
        x = int(round(det["x"]))
        y = int(round(det["y"]))

        radius = int(max(det["w"], det["h"]) / 2) + 3
        radius = max(5, radius)

        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(i),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output
