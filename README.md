# DeepSteg: Deep Learning Steganography Optimizer

![DeepSteg Logo](static/img/logo.svg)

DeepSteg adalah aplikasi berbasis web untuk steganografi citra yang menggunakan pendekatan Deep Learning dengan Generative Adversarial Networks (GAN) untuk mengoptimalkan kapasitas dan imperceptibility pada proses penyembunyian pesan dalam citra digital.

## 🌟 Fitur Utama

- **Deep Learning** - Implementasi steganografi dengan model GAN (atau fallback ke LSB)
- **Imperceptibility Tinggi** - Hasil steganografi tidak terlihat oleh mata manusia
- **Kapasitas Optimal** - Memaksimalkan jumlah data yang dapat disembunyikan
- **Metrik Kualitas** - Visualisasi metrik PSNR, SSIM, MSE, dan histogram similarity
- **Antarmuka Responsif** - UI yang menarik, clean, dan responsif di semua device
- **Robust Error Handling** - Penanganan error yang komprehensif
- **Enkripsi Terintegrasi** - Mendukung enkripsi AES-256 untuk keamanan tambahan
- **Kompresi Adaptif** - Kompresi teks secara otomatis untuk meningkatkan kapasitas

## 🖼️ Screenshot

![DeepSteg UI](docs/screenshot.png)

## 🛠️ Teknologi yang Digunakan

- **Backend**: Python, Flask, TensorFlow, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Image Processing**: NumPy, scikit-image, PIL
- **Enkripsi**: Cryptography (AES-256)
- **Database**: SQLite (untuk penyimpanan statistik)

## 📋 Persyaratan Sistem

- Python 3.7 atau lebih baru
- 2GB RAM minimum (4GB direkomendasikan)
- Sistem operasi: Windows, macOS, atau Linux
- Browser modern (Chrome, Firefox, Edge, Safari)
- Ruang disk: minimal 500MB

## 🔧 Instalasi & Menjalankan Aplikasi

### Metode Cepat dengan Script

#### Windows
1. Download dan ekstrak repositori ini
2. Double-click file `run.bat` atau jalankan melalui command prompt

#### macOS/Linux
1. Clone repositori ini
2. Buka terminal di folder proyek
3. Berikan izin eksekusi pada script: `chmod +x run.sh`
4. Jalankan script: `./run.sh`

### Instalasi Manual

1. Clone repositori atau download sebagai ZIP dan ekstrak:
   ```bash
   git clone https://github.com/username/deepsteg.git
   cd deepsteg
   ```

2. Buat dan aktifkan virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:
   ```bash
   python app.py
   ```

5. Buka browser dan akses: `http://localhost:5000`

## 🧩 Cara Penggunaan

### Menyembunyikan Pesan (Encode)

1. Buka halaman "Encoding" dari menu navigasi
2. Upload gambar cover (JPG, PNG, atau BMP)
3. Masukkan pesan teks yang ingin disembunyikan
4. Opsional: Aktifkan enkripsi dan atur password
5. Opsional: Aktifkan kompresi untuk meningkatkan kapasitas pesan
6. Klik tombol "Sembunyikan Pesan"
7. Lihat hasil steganografi dan metrik kualitas
8. Download gambar hasil (stego image)

### Mengekstrak Pesan (Decode)

1. Buka halaman "Decoding" dari menu navigasi
2. Upload gambar stego yang berisi pesan tersembunyi
3. Jika pesan terenkripsi, masukkan password yang sama saat encoding
4. Klik tombol "Ekstrak Pesan"
5. Lihat pesan yang berhasil diekstrak
6. Salin pesan ke clipboard jika diperlukan

## 📊 Metrik Kualitas

DeepSteg menghitung beberapa metrik untuk evaluasi kualitas steganografi:

- **PSNR (Peak Signal-to-Noise Ratio)** - Mengukur rasio kekuatan sinyal terhadap noise. Nilai lebih tinggi = lebih baik.
- **SSIM (Structural Similarity Index)** - Mengukur kemiripan struktural antara gambar. Nilai mendekati 1.0 = sangat mirip.
- **MSE (Mean Squared Error)** - Mengukur error rata-rata antar piksel. Nilai lebih rendah = lebih baik.
- **Histogram Similarity** - Mengukur kesamaan distribusi warna. Nilai mendekati 1.0 = hampir identik.

## 🔍 Implementasi Teknis

### Model Deep Learning

Aplikasi menggunakan arsitektur GAN (Generative Adversarial Network) yang terdiri dari:
- **Encoder** - Menyembunyikan pesan dalam gambar
- **Decoder** - Mengekstrak pesan dari gambar stego
- **Discriminator** - Membedakan gambar asli dan gambar stego

### Fallback LSB

Jika model deep learning belum dilatih atau tidak tersedia, aplikasi akan menggunakan metode LSB (Least Significant Bit) yang telah ditingkatkan keamanannya dengan pendekatan:
- Penyisipan bit dengan pola pseudo-random
- Header panjang pesan terenkripsi
- Anti-steganalysis noise padding

### Pengamanan Data

- Enkripsi menggunakan AES-256 dengan key derivation (PBKDF2)
- Salt dan IV unik untuk setiap pesan
- Password strength meter untuk memastikan keamanan

### Validasi & Error Handling

Aplikasi mengimplementasikan validasi komprehensif:
- Validasi format dan ukuran gambar
- Pengecekan kapasitas pesan
- Penanganan error yang robust
- Logging untuk debugging

## 🚀 Deployment ke Production

### Menggunakan Docker

1. Build image Docker:
   ```bash
   docker build -t deepsteg .
   ```

2. Jalankan container:
   ```bash
   docker run -p 8000:5000 deepsteg
   ```

3. Akses aplikasi di `http://localhost:8000`

### Menggunakan Gunicorn (Ubuntu/Debian)

1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```

2. Jalankan dengan Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

3. Setup dengan Nginx (opsional):
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 🤝 Kontribusi

Kontribusi selalu diterima dengan tangan terbuka! Jika Anda ingin berkontribusi:

1. Fork repositori
2. Buat branch untuk fitur Anda (`git checkout -b feature/amazing-feature`)
3. Commit perubahan Anda (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buka Pull Request

## 🔧 Troubleshooting

- **Error "ModuleNotFoundError"**: Pastikan semua dependensi terinstall dengan benar dan virtual environment telah diaktifkan.
- **Error "GPU not available"**: TensorFlow akan otomatis menggunakan CPU jika GPU tidak tersedia. Tidak masalah untuk penggunaan normal, hanya mempengaruhi kecepatan.
- **Error "Image file too large"**: Coba resize gambar menjadi lebih kecil. Batas ukuran default adalah 16MB.
- **Error "Message too large"**: Coba aktifkan fitur kompresi untuk meningkatkan kapasitas pesan, atau gunakan gambar dengan resolusi lebih tinggi.

## 📝 Catatan Pengembangan

- Model deep learning masih dalam tahap pengembangan awal
- Beberapa fitur mungkin belum optimal
- Umpan balik dan saran sangat dihargai!

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License. Lihat file `LICENSE` untuk informasi lebih lanjut.

## 📧 Kontak

- Email: your.email@example.com
- GitHub: [github.com/username](https://github.com/username)
- Website: [your-website.com](https://your-website.com)

---

Made with ❤️ by Tryan