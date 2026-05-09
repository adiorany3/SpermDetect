import cv2
import numpy as np
import math


def _odd(value):
    value = int(value)
    return value if value % 2 == 1 else value + 1


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

    while True:
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        if cv2.countNonZero(image) == 0:
            break

    return skeleton


def _preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.4,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(gray)
    enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)

    return gray, enhanced_gray


def _enhance_head_blackhat(enhanced_gray):
    h, w = enhanced_gray.shape[:2]

    # Kernel otomatis mengikuti ukuran gambar.
    kernel_size = _odd(max(9, min(h, w) * 0.075))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        enhanced_gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    blackhat = cv2.normalize(
        blackhat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Dibuat lebih terlihat.
    blackhat = cv2.convertScaleAbs(
        blackhat,
        alpha=1.8,
        beta=0
    )

    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    return blackhat


def _enhance_tail_blackhat(enhanced_gray):
    h, w = enhanced_gray.shape[:2]

    # Kernel garis dibuat lebih panjang agar ekor tipis lebih muncul.
    line_length = _odd(max(17, min(h, w) * 0.13))

    response = np.zeros_like(enhanced_gray)

    for angle in range(0, 180, 15):
        kernel = _line_kernel(line_length, angle)
        blackhat = cv2.morphologyEx(
            enhanced_gray,
            cv2.MORPH_BLACKHAT,
            kernel
        )
        response = np.maximum(response, blackhat)

    response = cv2.normalize(
        response,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    response = cv2.convertScaleAbs(
        response,
        alpha=2.0,
        beta=0
    )

    response = cv2.GaussianBlur(response, (3, 3), 0)

    return response


def _make_head_binary(head_blackhat):
    threshold_value = max(20, np.percentile(head_blackhat, 89))

    binary = (head_blackhat >= threshold_value).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary


def _make_tail_binary(tail_blackhat):
    threshold_value = max(14, np.percentile(tail_blackhat, 83))

    binary = (tail_blackhat >= threshold_value).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def _extract_head_candidates(gray, head_blackhat, head_binary):
    image_h, image_w = gray.shape[:2]
    image_area = image_h * image_w

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        head_binary
    )

    candidates = []

    min_area = max(10, image_area * 0.00018)
    max_area = max(80, image_area * 0.008)

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        if area < min_area or area > max_area:
            continue

        if w < 4 or h < 4:
            continue

        if w > image_w * 0.25 or h > image_h * 0.25:
            continue

        aspect_ratio = max(w / max(h, 1), h / max(w, 1))

        if aspect_ratio > 3.8:
            continue

        cx, cy = centroids[label_id]
        radius = int(max(w, h) / 2) + 2

        roi = gray[
            max(0, int(cy - radius)):min(image_h, int(cy + radius + 1)),
            max(0, int(cx - radius)):min(image_w, int(cx + radius + 1))
        ]

        if roi.size == 0:
            continue

        dark_score = 255 - float(np.mean(roi))
        blackhat_score = float(np.mean(head_blackhat[labels == label_id]))
        score = blackhat_score + dark_score * 0.15 + area * 0.04

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

    # Hilangkan kandidat ganda yang terlalu dekat.
    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    merged = []

    for candidate in candidates:
        is_duplicate = False

        for saved in merged:
            distance = math.hypot(
                candidate["x"] - saved["x"],
                candidate["y"] - saved["y"]
            )

            min_distance = max(
                7,
                0.55 * (candidate["r"] + saved["r"])
            )

            if distance < min_distance:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(candidate)

    return merged


def _validate_head_tail_pair(candidates, tail_skeleton, image_shape):
    image_h, image_w = image_shape[:2]

    yy, xx = np.mgrid[0:image_h, 0:image_w]

    dilated_skeleton = cv2.dilate(
        tail_skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    max_tail_distance = max(25, int(min(image_h, image_w) * 0.28))
    min_tail_pixels = max(4, int(min(image_h, image_w) * 0.025))

    detections = []

    for candidate in candidates:
        cx = candidate["x"]
        cy = candidate["y"]
        radius = candidate["r"]

        distance_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Ekor harus berada di luar kepala/badan, tetapi masih di sekitar objek.
        tail_zone = (
            (distance_map > radius + 1) &
            (distance_map < max_tail_distance)
        ).astype(np.uint8) * 255

        # Harus ada ekor yang dekat dengan kepala/badan.
        near_head_zone = (
            (distance_map > radius) &
            (distance_map < radius + 11)
        ).astype(np.uint8) * 255

        tail_pixels = cv2.countNonZero(
            cv2.bitwise_and(tail_skeleton, tail_zone)
        )

        near_head_pixels = cv2.countNonZero(
            cv2.bitwise_and(dilated_skeleton, near_head_zone)
        )

        valid_tail = (
            tail_pixels >= min_tail_pixels and
            near_head_pixels >= 1
        )

        if valid_tail:
            item = dict(candidate)
            item["tail_pixels"] = int(tail_pixels)
            item["near_head_pixels"] = int(near_head_pixels)
            detections.append(item)

    # Jika gambar sangat buram dan koneksi ekor ke kepala putus,
    # fallback tetap mewajibkan adanya banyak piksel ekor di sekitar kepala.
    if len(detections) < max(3, len(candidates) * 0.35):
        detections = []

        for candidate in candidates:
            cx = candidate["x"]
            cy = candidate["y"]
            radius = candidate["r"]

            distance_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            tail_zone = (
                (distance_map > radius) &
                (distance_map < max_tail_distance)
            ).astype(np.uint8) * 255

            tail_pixels = cv2.countNonZero(
                cv2.bitwise_and(tail_skeleton, tail_zone)
            )

            if tail_pixels >= min_tail_pixels:
                item = dict(candidate)
                item["tail_pixels"] = int(tail_pixels)
                item["near_head_pixels"] = 0
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

    candidates = _extract_head_candidates(
        gray=gray,
        head_blackhat=head_blackhat,
        head_binary=head_binary
    )

    detections = _validate_head_tail_pair(
        candidates=candidates,
        tail_skeleton=tail_skeleton,
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
                "Piksel Ekor": int(item.get("tail_pixels", 0)),
                "Score": round(float(item["score"]), 2)
            }
        )

    debug = {
        "Gray": gray,
        "Black-hat Kepala/Badan": head_blackhat,
        "Binary Kepala/Badan": head_binary,
        "Black-hat Ekor": tail_blackhat,
        "Binary Ekor": tail_binary,
        "Skeleton Ekor": tail_skeleton
    }

    return {
        "count": len(detections),
        "detections": detections,
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
            (x - 7, y + 5),
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

    resized = []

    target_w = 360
    target_h = 240

    for name, image in zip(names, images):
        image = cv2.resize(image, (target_w, target_h))
        canvas = image.copy()

        cv2.rectangle(
            canvas,
            (0, 0),
            (target_w, 28),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            canvas,
            name,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        resized.append(canvas)

    row1 = np.hstack(resized[:3])
    row2 = np.hstack(resized[3:6])

    return np.vstack([row1, row2])