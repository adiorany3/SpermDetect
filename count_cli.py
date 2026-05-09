import argparse
import cv2

from detector import detect_sperm_heads, draw_detections


def main():
    parser = argparse.ArgumentParser(description="Hitung jumlah sperma dari gambar mikroskop.")
    parser.add_argument("image_path", help="Path gambar input")
    parser.add_argument("--output", default="hasil_deteksi_sperma.jpg", help="Path output gambar anotasi")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image_path}")

    detections, enhanced, binary, working_bgr, scale = detect_sperm_heads(
        image_bgr,
        blackhat_kernel=25,
        clahe_clip=0.0,
        blur_size=1,
        morph_open=False,
        min_area=15,
        max_area=250,
        min_width=5,
        min_height=5,
        max_width_obj=25,
        max_height_obj=25,
        max_aspect_ratio=3.0,
        min_circularity=0.05,
        merge_distance=8,
    )

    output_bgr = draw_detections(working_bgr, detections)
    cv2.imwrite(args.output, output_bgr)

    print(f"Jumlah sperma terdeteksi: {len(detections)}")
    print(f"Hasil anotasi disimpan ke: {args.output}")


if __name__ == "__main__":
    main()
