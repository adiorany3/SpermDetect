# Deteksi Jumlah Sperma dari Gambar Mikroskop

Aplikasi Python + Streamlit untuk menghitung jumlah sperma dari gambar mikroskop.

Algoritma utama:
1. Konversi gambar ke grayscale.
2. Peningkatan kontras dengan CLAHE.
3. Black-hat morphology untuk menonjolkan kepala sperma yang gelap/oval.
4. Thresholding.
5. Connected component analysis.
6. Filter ukuran dan bentuk agar ekor sperma tidak ikut dihitung.

> Catatan: aplikasi ini untuk pembelajaran/eksperimen computer vision, bukan alat diagnosis medis.

## Struktur File

```text
deteksi_sperma_mikroskop/
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

Lalu upload gambar mikroskop.

Untuk gambar contoh `sample_sperm_16.jpeg`, pilih preset:

```text
Sperm sample 16
```

Target hasil: **16 sperma**.

## Cara Menjalankan dari Terminal

```bash
python count_cli.py sample_sperm_16.jpeg --output hasil.jpg
```

## Tips Penyesuaian

Jika hasil terlalu sedikit:
- Turunkan `Area minimum`.
- Turunkan `Lebar minimum` dan `Tinggi minimum`.
- Turunkan `Kebulatan minimum`.
- Gunakan threshold manual dengan nilai lebih rendah/tinggi sesuai gambar.

Jika hasil terlalu banyak:
- Naikkan `Area minimum`.
- Naikkan `Lebar minimum` dan `Tinggi minimum`.
- Naikkan `Kebulatan minimum`.
- Naikkan `Jarak gabung deteksi ganda`.

Jika ekor sperma ikut terhitung:
- Naikkan `Rasio lonjong maksimum` menjadi lebih kecil.
- Naikkan `Lebar minimum` dan `Tinggi minimum`.
- Naikkan `Kebulatan minimum`.
