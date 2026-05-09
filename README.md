# Deteksi Jumlah Sperma Mikroskop — Adaptive Single Preset

Aplikasi ini menghitung jumlah sperma dari gambar mikroskop menggunakan satu algoritma adaptif.

Footer aplikasi:

**Created by Galuh Adi Insani**

## Prinsip Deteksi

Satu sperma dihitung jika sistem menemukan:

1. Kepala/badan sperma.
2. Indikasi ekor di sekitar kepala/badan.

Algoritma utama:

- Background correction untuk mengurangi pencahayaan tidak rata.
- CLAHE untuk meningkatkan kontras lokal.
- Multi-scale black-hat untuk memperjelas kepala/badan.
- Multi-angle dan multi-length black-hat untuk memperjelas ekor.
- Skeletonization untuk membaca garis ekor.
- Validasi pasangan kepala/badan + ekor.
- Analisis kualitas gambar otomatis.

## Struktur File

```txt
deteksi_sperma_adaptive/
├── app.py
├── detector.py
├── count_cli.py
├── requirements.txt
├── README.md
└── sample_sperm_16.jpeg
```

## Cara Menjalankan Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Menjalankan dari Terminal

```bash
python count_cli.py sample_sperm_16.jpeg --output hasil.jpg
```

## Kriteria Gambar yang Baik

Agar hasil deteksi lebih presisi, gunakan gambar dengan kondisi berikut:

1. Fokus jelas; kepala/badan dan ekor terlihat.
2. Kontras cukup; objek sperma harus berbeda jelas dari background.
3. Pencahayaan merata; tidak terlalu gelap, tidak overexposure, dan tidak banyak bayangan.
4. Ekor terlihat; sistem hanya menghitung sperma lengkap jika ada kepala/badan dan indikasi ekor.
5. Background bersih; hindari kotoran, gelembung, noise, dan bercak yang mirip kepala sperma.
6. Pembesaran konsisten; jangan mencampur skala objek yang sangat berbeda.
7. Objek tidak terlalu bertumpuk; tumpang tindih bisa menyebabkan hitungan kurang/lebih.
8. Resolusi cukup; disarankan minimal 300 px pada sisi terpendek.
9. Hindari kompresi berat dari screenshot atau WhatsApp.
10. Ambil beberapa bidang pandang dan gunakan rata-rata untuk hasil yang lebih representatif.

## Catatan

Deteksi berbasis citra klasik tidak akan selalu sempurna untuk semua mikroskop karena dipengaruhi oleh fokus, pembesaran, kamera, noise, preparat, dan pencahayaan.

Untuk akurasi produksi/laboratorium, sebaiknya kumpulkan banyak gambar berlabel lalu latih model machine learning seperti YOLO atau Mask R-CNN.