# Deteksi Sperma Mikroskop - Black-hat + Kepala/Badan + Ekor

Aplikasi ini menghitung jumlah sperma dari gambar mikroskop.

Prinsip deteksi:

1. **CLAHE** untuk memperkuat kontras gambar mikroskop.
2. **Black-hat morphology** untuk membuat objek gelap lebih terlihat.
3. Deteksi **kepala/badan** sebagai blob kecil gelap.
4. Deteksi **ekor** sebagai garis tipis gelap, lalu dibuat skeleton.
5. Satu objek dihitung sebagai sperma jika terdeteksi **kepala/badan + ekor di sekitar kepala**.

## Cara menjalankan Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload gambar mikroskop, lalu gunakan preset **Sample 16 sperma** untuk gambar contoh.

## Cara menjalankan CLI

```bash
python count_cli.py sample_sperm_16.jpeg --output sample_sperm_16_result.jpg --show-debug
```

Output contoh:

```text
Jumlah sperma terdeteksi: 16
```

## Parameter penting

- **Gain tampilan black-hat**: hanya untuk memperjelas tampilan black-hat di aplikasi.
- **Kernel black-hat kepala**: makin kecil, lebih fokus ke kepala kecil; makin besar, lebih banyak bagian gelap ikut muncul.
- **Kernel black-hat ekor**: lebih besar agar ekor/garis panjang ikut terlihat.
- **Minimal piksel ekor**: memastikan satu deteksi memiliki ekor, bukan hanya titik kepala.
- **Jarak gabung deteksi**: mengurangi duplikasi pada kepala yang sama.

Catatan: gambar mikroskop bisa berbeda-beda kualitas, ukuran objek, dan pencahayaan. Karena itu beberapa parameter disediakan di sidebar agar mudah disesuaikan.
