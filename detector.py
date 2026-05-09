import cv2
import numpy as np


def resize_keep_aspect(image_bgr, max_width=900):
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr.copy(), 1.0
    scale = max_width / float(w)
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def make_odd(value, minimum=3):
    value = int(value)
    if value % 2 == 0:
        value += 1
    return max(minimum, value)


def preprocess_gray(image_bgr, clahe_clip=2.0, blur_size=1):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if float(clahe_clip) > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    blur_size = make_odd(blur_size, 1)
    if blur_size >= 3:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    return gray


def skeletonize(binary):
    """Skeletonization with pure OpenCV, so no scikit-image is required."""
    binary = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(300):
        eroded = cv2.erode(binary, element)
        opened = cv2.dilate(eroded, element)
        edge = cv2.subtract(binary, opened)
        skeleton = cv2.bitwise_or(skeleton, edge)
        binary = eroded
        if cv2.countNonZero(binary) == 0:
            break

    return skeleton


def merge_nearby_candidates(candidates, merge_distance=8):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    final = []

    for cand in candidates:
        duplicate = False
        for kept in final:
            distance = np.hypot(cand["x"] - kept["x"], cand["y"] - kept["y"])
            if distance < float(merge_distance):
                duplicate = True
                break
        if not duplicate:
            final.append(cand)

    return sorted(final, key=lambda c: (c["y"], c["x"]))


def build_blackhat(gray, kernel_size=13, gain=4.0):
    kernel_size = make_odd(kernel_size, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    raw_blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Untuk tampilan agar black-hat lebih terlihat jelas di Streamlit.
    visible_blackhat = cv2.convertScaleAbs(raw_blackhat, alpha=float(gain), beta=0)
    visible_blackhat = cv2.normalize(visible_blackhat, None, 0, 255, cv2.NORM_MINMAX)

    return raw_blackhat, visible_blackhat


def build_tail_map(gray, tail_kernel=31, tail_threshold=30, tail_gain=4.0):
    tail_kernel = make_odd(tail_kernel, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail_kernel, tail_kernel))

    raw_tail = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tail_visible = cv2.convertScaleAbs(raw_tail, alpha=float(tail_gain), beta=0)
    tail_visible = cv2.normalize(tail_visible, None, 0, 255, cv2.NORM_MINMAX)

    _, tail_binary = cv2.threshold(tail_visible, int(tail_threshold), 255, cv2.THRESH_BINARY)
    tail_binary = cv2.morphologyEx(
        tail_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    tail_skeleton = skeletonize(tail_binary)
    return tail_visible, tail_binary, tail_skeleton


def detect_sperm_complete(
    image_bgr,
    max_width=900,
    clahe_clip=2.0,
    blur_size=1,
    head_blackhat_kernel=13,
    blackhat_gain=4.0,
    threshold_mode="otsu",
    manual_head_threshold=12,
    min_area=10,
    max_area=200,
    min_width=3,
    min_height=3,
    max_width_obj=25,
    max_height_obj=25,
    max_aspect_ratio=4.0,
    min_circularity=0.03,
    tail_blackhat_kernel=31,
    tail_threshold=30,
    tail_gain=4.0,
    tail_search_radius=30,
    min_tail_pixels=20,
    merge_distance=8,
    border_margin=0,
):
    """
    Menghitung satu sperma hanya jika terdeteksi:
    1. kepala/badan gelap berbentuk blob oval kecil, dan
    2. ada ekor/garis tipis di sekitar kepala pada tail skeleton.

    Output debug:
    - head_blackhat_visible: black-hat yang sudah diperkuat agar terlihat jelas
    - head_binary: threshold kepala/badan
    - tail_visible: black-hat kernel besar untuk ekor
    - tail_binary: threshold ekor
    - tail_skeleton: garis ekor yang ditipiskan
    """
    working_bgr, scale = resize_keep_aspect(image_bgr, max_width=max_width)
    gray = preprocess_gray(working_bgr, clahe_clip=clahe_clip, blur_size=blur_size)

    raw_blackhat, head_blackhat_visible = build_blackhat(
        gray,
        kernel_size=head_blackhat_kernel,
        gain=blackhat_gain,
    )

    if threshold_mode == "manual":
        _, head_binary = cv2.threshold(raw_blackhat, int(manual_head_threshold), 255, cv2.THRESH_BINARY)
    else:
        _, head_binary = cv2.threshold(raw_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tail_visible, tail_binary, tail_skeleton = build_tail_map(
        gray,
        tail_kernel=tail_blackhat_kernel,
        tail_threshold=tail_threshold,
        tail_gain=tail_gain,
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(head_binary, connectivity=8)
    h_img, w_img = head_binary.shape
    candidates = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]

        if border_margin > 0:
            if x <= border_margin or y <= border_margin or x + w >= w_img - border_margin or y + h >= h_img - border_margin:
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
        circularity = 4.0 * np.pi * float(area) / ((perimeter * perimeter) + 1e-6)
        if circularity < min_circularity:
            continue

        cx, cy = centroids[idx]
        head_radius = int(max(w, h) / 2) + 3

        # Cari ekor di area cincin sekitar kepala.
        # Bagian dalam dihapus agar kepala tidak dianggap ekor.
        ring_mask = np.zeros_like(head_binary)
        cv2.circle(ring_mask, (int(cx), int(cy)), head_radius + int(tail_search_radius), 255, -1)
        cv2.circle(ring_mask, (int(cx), int(cy)), head_radius + 2, 0, -1)

        tail_pixels = cv2.countNonZero(cv2.bitwise_and(tail_skeleton, ring_mask))
        has_tail = tail_pixels >= int(min_tail_pixels)

        if not has_tail:
            continue

        candidates.append({
            "x": float(cx),
            "y": float(cy),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "aspect": float(aspect),
            "circularity": float(circularity),
            "tail_pixels": int(tail_pixels),
            "score": float((area * circularity) + tail_pixels),
        })

    detections = merge_nearby_candidates(candidates, merge_distance=merge_distance)

    debug = {
        "working_bgr": working_bgr,
        "gray": gray,
        "head_blackhat_visible": head_blackhat_visible,
        "head_binary": head_binary,
        "tail_visible": tail_visible,
        "tail_binary": tail_binary,
        "tail_skeleton": tail_skeleton,
        "scale": scale,
    }

    return detections, debug


def draw_detections(image_bgr, detections):
    output = image_bgr.copy()

    for i, det in enumerate(detections, start=1):
        x = int(round(det["x"]))
        y = int(round(det["y"]))
        radius = int(max(det["w"], det["h"]) / 2) + 5
        radius = max(6, radius)

        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(output, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    return output
