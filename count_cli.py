import argparse
import cv2

from detector import count_sperm, draw_detection, image_quality_report


def main():
    parser = argparse.ArgumentParser(
        description="Hitung jumlah sperma dari gambar mikroskop."
    )

    parser.add_argument(
        "image",
        help="Path gambar input, misalnya sample_sperm_16.jpeg"
    )

    parser.add_argument(
        "--output",
        default="hasil_deteksi_sperma.jpg",
        help="Path gambar output dengan anotasi."
    )

    args = parser.parse_args()

    bgr = cv2.imread(args.image)

    if bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = count_sperm(rgb)
    output_rgb = draw_detection(rgb, result["detections"])
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(args.output, output_bgr)

    print(f"Jumlah sperma terdeteksi: {result['count']}")
    print(f"Hasil gambar disimpan ke: {args.output}")
    print()
    print("Analisis kualitas gambar:")

    for item in image_quality_report(rgb):
        print(f"- [{item['status'].upper()}] {item['message']}")

    print()
    print("Created by Galuh Adi Insani")


if __name__ == "__main__":
    main()