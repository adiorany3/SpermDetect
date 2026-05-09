# Deteksi Jumlah Sperma dari Citra Mikroskop

Aplikasi Streamlit sederhana untuk menghitung jumlah sperma dari gambar mikroskop menggunakan OpenCV.

## Catatan penting

Aplikasi ini dibuat untuk pembelajaran/eksperimen computer vision, bukan untuk diagnosis medis. Hasil dapat berubah tergantung kualitas mikroskop, pencahayaan, fokus, pembesaran, dan pewarnaan sampel.

## Cara kerja algoritma

Objek yang dihitung adalah kepala sperma, karena kepala lebih stabil dideteksi daripada ekor yang tipis.

Tahapan utama:

1. Convert gambar RGB ke grayscale.
2. Peningkatan kontras lokal dengan CLAHE.
3. Koreksi background agar pencahayaan mikroskop tidak terlalu memengaruhi hasil.
4. Adaptive threshold untuk memisahkan objek kecil dari background.
5. Morphology open/close untuk membersihkan noise.
6. Watershed opsional untuk memisahkan kepala sperma yang menempel.
7. Filter contour berdasarkan luas, circularity, aspect ratio, dan jarak minimum.
8. Menampilkan jumlah, gambar hasil deteksi, dan data CSV.

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

## Tips pengaturan

Jika hasil terlalu sedikit:

- Turunkan `Luas minimum kepala sperma`
- Turunkan `Circularity minimum`
- Turunkan `Jarak minimum antar deteksi`
- Coba ubah `Adaptive threshold C`
- Aktifkan atau nonaktifkan `Invert`

Jika hasil terlalu banyak/noise:

- Naikkan `Luas minimum kepala sperma`
- Naikkan `Circularity minimum`
- Naikkan `Jarak minimum antar deteksi`
- Naikkan `Luas maksimum` hanya jika objek kepala memang besar

## Struktur file

```text
app.py
requirements.txt
README.md
```
