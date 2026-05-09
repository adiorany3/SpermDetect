import cv2
import numpy as np
import math


def _odd(value):
    value = int(round(value))
    if value < 3:
        value = 3
    return value if value % 2 == 1 else value + 1


def _safe_percentile(image, p):
    values = image.reshape(-1)
    return float(np.percentile(values, p))


def _line_kernel(length, angle_degree):
    length = _odd(length)
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2

    angle = np.deg2rad(angle_degree)
    dx = int(np.cos(angle) * center)
    dy = int(np.sin(angle) * center)

    cv2.line(
        kernel,
        (center - dx, center - dy),
        (center + dx, center + dy),
        1,
        1
    )

    return kernel


def _skeletonize(binary):
    binary = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros(binary.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = binary.copy()

    max_iter = 300
    iteration = 0

    while True:
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
        iteration += 1

        if cv2.countNonZero(image) == 0 or iteration > max_iter:
            break

    return skeleton


def _remove_border(binary, border_ratio=0.01):
    h, w = binary.shape[:2]
    border = max(2, int(min(h, w) * border_ratio))
    cleaned = binary.copy()
    cleaned[:border, :] = 0
    cleaned[-border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, -border:] = 0
    return cleaned


def _preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Koreksi background tidak rata.
    h, w = gray.shape[:2]
    bg_kernel = _odd(max(31, min(h, w) * 0.18))
    background = cv2.medianBlur(gray, bg_kernel)
    corrected = cv2.addWeighted(gray, 1.45, background, -0.45, 15)

    # CLAHE untuk memperjelas detail kepala dan ekor.
    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(corrected)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return gray, enhanced


def _enhance_head_blackhat(enhanced_gray):
    h, w = enhanced_gray.shape[:2]

    # Beberapa ukuran kernel agar tahan terhadap variasi pembesaran.
    base = min(h, w)
    kernel_sizes = [
        _odd(max(7, base * 0.045)),
        _odd(max(11, base * 0.070)),
        _odd(max(15, base * 0.095)),
    ]

    response = np.zeros_like(enhanced_gray)

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )
        blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)
        response = np.maximum(response, blackhat)

    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response = cv2.convertScaleAbs(response, alpha=1.9, beta=0)
    response = cv2.GaussianBlur(response, (3, 3), 0)

    return response


def _enhance_tail_blackhat(enhanced_gray):
    h, w = enhanced_gray.shape[:2]
    base = min(h, w)

    # Multi-length dan multi-angle untuk ekor yang melengkung/beda arah.
    lengths = [
        _odd(max(15, base * 0.09)),
        _odd(max(21, base * 0.14)),
        _odd(max(27, base * 0.19)),
    ]

    response = np.zeros_like(enhanced_gray)

    for length in lengths:
        for angle in range(0, 180, 15):
            kernel = _line_kernel(length, angle)
            blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)
            response = np.maximum(response, blackhat)

    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response = cv2.convertScaleAbs(response, alpha=2.1, beta=0)
    response = cv2.GaussianBlur(response, (3, 3), 0)

    return response


def _auto_threshold(image, percentile, minimum):
    value = max(minimum, _safe_percentile(image, percentile))
    _, binary = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
    return binary


def _make_head_binary(head_blackhat):
    # Gabungan threshold adaptif + percentile agar lebih robust.
    p_bin = _auto_threshold(head_blackhat, 88, 18)

    adaptive = cv2.adaptiveThreshold(
        head_blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _odd(max(15, min(head_blackhat.shape[:2]) * 0.09)),
        -2
    )

    binary = cv2.bitwise_and(p_bin, adaptive)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=1)
    binary = _remove_border(binary)

    return binary


def _make_tail_binary(tail_blackhat):
    p_bin = _auto_threshold(tail_blackhat, 81, 11)

    adaptive = cv2.adaptiveThreshold(
        tail_blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _odd(max(17, min(tail_blackhat.shape[:2]) * 0.11)),
        -3
    )

    binary = cv2.bitwise_or(p_bin, adaptive)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=1)
    binary = _remove_border(binary)

    return binary


def _component_shape_score(w, h, area):
    bbox_area = max(1, w * h)
    extent = area / bbox_area
    aspect = max(w / max(h, 1), h / max(w, 1))

    # Kepala/badan sperma biasanya oval, bukan garis sangat panjang.
    score = 1.0

    if aspect > 4.0:
        score *= 0.25
    elif aspect > 3.0:
        score *= 0.55

    if extent < 0.15:
        score *= 0.60
    elif extent > 0.95:
        score *= 0.75

    return score


def _extract_head_candidates(gray, head_blackhat, head_binary):
    image_h, image_w = gray.shape[:2]
    image_area = image_h * image_w
    base = min(image_h, image_w)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(head_binary)

    candidates = []

    min_area = max(6, image_area * 0.00009)
    max_area = max(70, image_area * 0.012)

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        if area < min_area or area > max_area:
            continue

        if w < 3 or h < 3:
            continue

        if w > image_w * 0.28 or h > image_h * 0.28:
            continue

        shape_score = _component_shape_score(w, h, area)
        if shape_score < 0.30:
            continue

        cx, cy = centroids[label_id]
        radius = int(max(w, h) / 2) + max(2, int(base * 0.006))

        roi = gray[
            max(0, int(cy - radius)):min(image_h, int(cy + radius + 1)),
            max(0, int(cx - radius)):min(image_w, int(cx + radius + 1))
        ]

        if roi.size == 0:
            continue

        dark_score = 255 - float(np.mean(roi))
        blackhat_score = float(np.mean(head_blackhat[labels == label_id]))

        # Penalti noise yang terlalu kecil/terlalu putih.
        if blackhat_score < 8 and dark_score < 35:
            continue

        score = (
            blackhat_score * 1.00 +
            dark_score * 0.16 +
            math.sqrt(area) * 1.35
        ) * shape_score

        candidates.append(
            {
                "x": float(cx),
                "y": float(cy),
                "r": int(radius),
                "area": int(area),
                "score": float(score),
                "bbox": (int(x), int(y), int(w), int(h))
            }
        )

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    merged = []

    for candidate in candidates:
        is_duplicate = False

        for saved in merged:
            distance = math.hypot(candidate["x"] - saved["x"], candidate["y"] - saved["y"])
            min_distance = max(5, 0.58 * (candidate["r"] + saved["r"]))

            if distance < min_distance:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(candidate)

    return merged


def _tail_statistics_for_candidate(candidate, tail_skeleton, image_shape):
    image_h, image_w = image_shape[:2]
    cx = candidate["x"]
    cy = candidate["y"]
    radius = candidate["r"]

    base = min(image_h, image_w)
    max_tail_distance = max(22, int(base * 0.33))
    min_tail_distance = max(radius + 1, int(base * 0.012))

    x1 = max(0, int(cx - max_tail_distance))
    x2 = min(image_w, int(cx + max_tail_distance + 1))
    y1 = max(0, int(cy - max_tail_distance))
    y2 = min(image_h, int(cy + max_tail_distance + 1))

    local = tail_skeleton[y1:y2, x1:x2]

    if local.size == 0:
        return 0, 0, 0

    ys, xs = np.where(local > 0)

    if len(xs) == 0:
        return 0, 0, 0

    xs = xs + x1
    ys = ys + y1

    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    valid = (distances > min_tail_distance) & (distances < max_tail_distance)
    near = (distances > radius) & (distances < radius + max(9, int(base * 0.035)))

    tail_pixels = int(np.sum(valid))
    near_head_pixels = int(np.sum(near))
    max_reach = float(np.max(distances[valid])) if tail_pixels > 0 else 0.0

    return tail_pixels, near_head_pixels, max_reach


def _validate_head_tail_pair(candidates, tail_skeleton, image_shape):
    image_h, image_w = image_shape[:2]
    base = min(image_h, image_w)

    min_tail_pixels = max(3, int(base * 0.018))
    min_tail_reach = max(10, int(base * 0.045))

    detections = []

    for candidate in candidates:
        tail_pixels, near_head_pixels, max_reach = _tail_statistics_for_candidate(
            candidate,
            tail_skeleton,
            image_shape
        )

        valid_tail = (
            tail_pixels >= min_tail_pixels and
            max_reach >= min_tail_reach and
            near_head_pixels >= 1
        )

        if valid_tail:
            item = dict(candidate)
            item["tail_pixels"] = int(tail_pixels)
            item["near_head_pixels"] = int(near_head_pixels)
            item["tail_reach"] = round(float(max_reach), 1)
            detections.append(item)

    # Fallback untuk gambar buram: koneksi ekor sering putus, jadi near_head dibuat lebih longgar.
    if len(detections) < max(2, int(len(candidates) * 0.30)):
        detections = []

        for candidate in candidates:
            tail_pixels, near_head_pixels, max_reach = _tail_statistics_for_candidate(
                candidate,
                tail_skeleton,
                image_shape
            )

            if tail_pixels >= max(2, min_tail_pixels - 1) and max_reach >= min_tail_reach:
                item = dict(candidate)
                item["tail_pixels"] = int(tail_pixels)
                item["near_head_pixels"] = int(near_head_pixels)
                item["tail_reach"] = round(float(max_reach), 1)
                detections.append(item)

    detections = sorted(detections, key=lambda item: (item["y"], item["x"]))

    return detections


def count_sperm(image_rgb):
    gray, enhanced_gray = _preprocess(image_rgb)

    head_blackhat = _enhance_head_blackhat(enhanced_gray)
    tail_blackhat = _enhance_tail_blackhat(enhanced_gray)

    head_binary = _make_head_binary(head_blackhat)
    tail_binary = _make_tail_binary(tail_blackhat)
    tail_skeleton = _skeletonize(tail_binary)

    # Dilasi tipis agar ekor putus-putus masih terbaca sebagai indikasi ekor.
    tail_skeleton_for_validation = cv2.dilate(
        tail_skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    candidates = _extract_head_candidates(
        gray=gray,
        head_blackhat=head_blackhat,
        head_binary=head_binary
    )

    detections = _validate_head_tail_pair(
        candidates=candidates,
        tail_skeleton=tail_skeleton_for_validation,
        image_shape=gray.shape
    )

    table = []

    for index, item in enumerate(detections, start=1):
        table.append(
            {
                "No": index,
                "X": int(round(item["x"])),
                "Y": int(round(item["y"])),
                "Radius Kepala/Badan": int(item["r"]),
                "Area Kepala/Badan": int(item["area"]),
                "Piksel Ekor": int(item.get("tail_pixels", 0)),
                "Jangkauan Ekor": item.get("tail_reach", 0),
                "Score": round(float(item["score"]), 2)
            }
        )

    debug = {
        "Gray": gray,
        "Enhanced": enhanced_gray,
        "Black-hat Kepala/Badan": head_blackhat,
        "Binary Kepala/Badan": head_binary,
        "Black-hat Ekor": tail_blackhat,
        "Skeleton Ekor": tail_skeleton
    }

    return {
        "count": len(detections),
        "detections": detections,
        "candidates": candidates,
        "table": table,
        "debug": debug
    }


def draw_detection(image_rgb, detections):
    output = image_rgb.copy()

    for index, item in enumerate(detections, start=1):
        x = int(round(item["x"]))
        y = int(round(item["y"]))
        r = int(item["r"])

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)

        cv2.putText(
            output,
            str(index),
            (x - 8, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output


def _to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def make_debug_grid(debug):
    names = list(debug.keys())
    images = [_to_rgb(debug[name]) for name in names]

    target_w = 360
    target_h = 240
    rendered = []

    for name, image in zip(names, images):
        image = cv2.resize(image, (target_w, target_h))
        canvas = image.copy()

        cv2.rectangle(canvas, (0, 0), (target_w, 30), (255, 255, 255), -1)
        cv2.putText(
            canvas,
            name,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        rendered.append(canvas)

    row1 = np.hstack(rendered[:3])
    row2 = np.hstack(rendered[3:6])

    return np.vstack([row1, row2])


def image_quality_report(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast_value = gray.std()
    brightness_value = gray.mean()
    min_side = min(h, w)

    report = []

    if min_side >= 300:
        report.append({"status": "baik", "message": f"Resolusi cukup baik: {w} x {h} px."})
    elif min_side >= 200:
        report.append({"status": "cukup", "message": f"Resolusi cukup, tetapi lebih baik minimal 300 px pada sisi terpendek. Saat ini: {w} x {h} px."})
    else:
        report.append({"status": "kurang", "message": f"Resolusi terlalu kecil: {w} x {h} px. Detail ekor bisa hilang."})

    if blur_value >= 120:
        report.append({"status": "baik", "message": f"Fokus gambar baik. Nilai ketajaman: {blur_value:.1f}."})
    elif blur_value >= 55:
        report.append({"status": "cukup", "message": f"Fokus cukup, tetapi masih bisa lebih tajam. Nilai ketajaman: {blur_value:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Gambar cenderung blur. Nilai ketajaman: {blur_value:.1f}. Ekor sperma bisa tidak terbaca."})

    if contrast_value >= 38:
        report.append({"status": "baik", "message": f"Kontras gambar baik. Nilai kontras: {contrast_value:.1f}."})
    elif contrast_value >= 22:
        report.append({"status": "cukup", "message": f"Kontras cukup. Nilai kontras: {contrast_value:.1f}. Hasil bisa membaik jika objek lebih kontras dari background."})
    else:
        report.append({"status": "kurang", "message": f"Kontras rendah. Nilai kontras: {contrast_value:.1f}. Kepala/ekor sulit dipisahkan dari background."})

    if 60 <= brightness_value <= 205:
        report.append({"status": "baik", "message": f"Pencahayaan berada pada rentang baik. Brightness: {brightness_value:.1f}."})
    elif 40 <= brightness_value < 60 or 205 < brightness_value <= 225:
        report.append({"status": "cukup", "message": f"Pencahayaan cukup, tetapi belum ideal. Brightness: {brightness_value:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Pencahayaan kurang ideal. Brightness: {brightness_value:.1f}. Hindari gambar terlalu gelap/terlalu terang."})

    return report