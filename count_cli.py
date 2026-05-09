import argparse
import cv2

from detector import detect_sperm_complete, draw_detections


def main():
    parser = argparse.ArgumentParser(description="Hitung sperma dari gambar mikroskop.")
    parser.add_argument("image", help="Path gambar input")
    parser.add_argument("--output", default="result.jpg", help="Path gambar hasil anotasi")
    parser.add_argument("--show-debug", action="store_true", help="Simpan gambar debug black-hat dan skeleton")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"Gambar tidak bisa dibaca: {args.image}")

    detections, debug = detect_sperm_complete(image_bgr)
    result = draw_detections(debug["working_bgr"], detections)
    cv2.imwrite(args.output, result)

    if args.show_debug:
        cv2.imwrite("debug_blackhat_head.png", debug["head_blackhat_visible"])
        cv2.imwrite("debug_head_binary.png", debug["head_binary"])
        cv2.imwrite("debug_tail_binary.png", debug["tail_binary"])
        cv2.imwrite("debug_tail_skeleton.png", debug["tail_skeleton"])

    print(f"Jumlah sperma terdeteksi: {len(detections)}")
    print(f"Hasil anotasi disimpan ke: {args.output}")


if __name__ == "__main__":
    main()
