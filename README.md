# Deteksi Jumlah Sperma Mikroskop — Single Preset

Aplikasi ini menghitung jumlah sperma dari gambar mikroskop menggunakan satu preset universal.

## Prinsip Deteksi

Satu sperma dihitung jika ditemukan:

1. Kepala/badan sperma.
2. Indikasi ekor di sekitar kepala/badan.

Algoritma utama:

- CLAHE untuk memperjelas kontras gambar mikroskop.
- Enhanced black-hat untuk mempertegas kepala/badan gelap.
- Multi-directional black-hat untuk mempertegas ekor tipis.
- Skeletonization untuk membaca garis ekor.
- Validasi pasangan kepala/badan + ekor.

## Struktur File

```txt
deteksi_sperma_single_preset/
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

## Catatan Penting

Gambar mikroskop sangat dipengaruhi oleh pencahayaan, fokus, pembesaran, noise, dan ketebalan preparat.

Agar hasil lebih stabil:

- Gunakan gambar yang fokus.
- Usahakan latar belakang tidak terlalu gelap.
- Hindari gambar yang terlalu blur.
- Gunakan pembesaran yang konsisten.
- Jika ekor tidak terlihat, objek tidak akan dihitung sebagai sperma lengkap.