# DeepSteg: Deep Learning Steganography Optimizer

![DeepSteg Logo](static/img/logo.svg)

DeepSteg adalah aplikasi berbasis web dan command line untuk steganografi citra yang menggunakan pendekatan Deep Learning dengan Generative Adversarial Networks (GAN) untuk mengoptimalkan kapasitas dan imperceptibility pada proses penyembunyian pesan dalam citra digital.

## 🚀 Fitur Baru (v2.0.0)

- ✨ **Pre-trained Models**: Dukungan untuk model GAN yang telah dilatih sebelumnya
- 🕵️ **Steganalysis**: Deteksi steganografi dengan model machine learning dan analisis statistik
- 🔄 **Transfer Learning**: Fine-tuning model dengan dataset kustom
- 💪 **CLI Interface**: Antarmuka command line untuk penggunaan tanpa web UI
- 📊 **Enhanced Metrics**: Metrik evaluasi yang lebih lengkap dan visualisasi
- 🔍 **Advanced Analysis**: Analisis mendalam untuk gambar stego

## 🌟 Fitur Utama

- 💾 **Steganografi Berbasis GAN**: Menggunakan Generative Adversarial Networks untuk hasil yang lebih baik
- 🔐 **Enkripsi Pesan**: Enkripsi pesan menggunakan AES sebelum penyembunyian
- 🗜️ **Kompresi Pesan**: Kompresi pesan untuk meningkatkan kapasitas
- 📊 **Metrik Visual**: Evaluasi kualitas hasil dengan PSNR, SSIM, dan metrik lainnya
- 🧠 **Adaptive Encoding**: Penyesuaian encoding berdasarkan karakteristik gambar
- 🔍 **Steganalysis Built-in**: Deteksi steganografi terintegrasi
- 🖥️ **Web Interface**: Antarmuka web yang user-friendly
- 📝 **Command Line Interface**: Antarmuka command line untuk automation

## 📦 Instalasi

### Persyaratan

- Python 3.8 atau lebih baru
- PIP (Python Package Manager)
- Tensorflow 2.x

### Langkah Instalasi

1. Clone repository:
```bash
git clone https://github.com/yourusername/deepsteg.git
cd deepsteg
```

2. Instal dependensi:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
# Web UI
python app.py

# Command Line Interface
python deepsteg_cli.py --help
```

## 🚀 Penggunaan

### Web Interface

Buka browser dan akses `http://localhost:5000` untuk menggunakan antarmuka web.

### Command Line Interface (CLI)

DeepSteg CLI menyediakan berbagai perintah untuk steganografi dan steganalysis:

#### List available models
```bash
python deepsteg_cli.py list-models
```

#### Download a pre-trained model
```bash
python deepsteg_cli.py download-model --name gan_basic
```

#### Encode a message in an image
```bash
python deepsteg_cli.py encode --model gan_basic --image cover.jpg --message "This is a secret message" --output stego.png
```

#### Decode a message from a stego image
```bash
python deepsteg_cli.py decode --model gan_basic --image stego.png
```

#### Analyze an image for steganography
```bash
python deepsteg_cli.py analyze --image suspect.png --detector cnn_basic --visualize
```

#### Train a custom model with transfer learning
```bash
python deepsteg_cli.py train --base-model gan_basic --dataset ./my_dataset --name my_custom_model --epochs 20
```

#### Evaluate model performance
```bash
python deepsteg_cli.py evaluate --model my_custom_model --dataset ./test_dataset
```

## 📚 API Reference

### Pre-trained Model Manager

```python
from models.pretrained_model_manager import PretrainedModelManager

# Initialize model manager
model_manager = PretrainedModelManager()

# List available models
models = model_manager.list_available_models()

# Download a model
model_manager.download_model("gan_basic")

# Load a model
model = model_manager.load_model("gan_basic")

# Encode message
stego_image, metrics = model_manager.encode_with_model(
    "gan_basic", cover_image, "Secret message"
)

# Decode message
message, metadata = model_manager.decode_with_model(
    "gan_basic", stego_image
)
```

### Steganalysis Detector

```python
from models.pretrained_detector import PretrainedDetector

# Initialize detector
detector = PretrainedDetector()

# List available detectors
detectors = detector.list_available_detectors()

# Download a detector
detector.download_detector("cnn_basic")

# Analyze image
results = detector.detect_steganography(image)

# Generate comprehensive report
report = detector.analyze_image_for_report(image)
```

### Transfer Learning

```python
from models.transfer_learning import TransferLearning

# Initialize transfer learning with configuration
config = {
    'base_model': 'gan_basic',
    'batch_size': 16,
    'epochs': 20,
    'learning_rate': 1e-4
}
transfer = TransferLearning(config)

# Load base model
transfer.load_base_model('gan_basic')

# Create target model (can modify parameters)
transfer.create_target_model()

# Train model
transfer.train(
    dataset_path='./my_dataset',
    epochs=20,
    name='my_custom_model',
    description='Fine-tuned GAN model'
)

# Evaluate model
results = transfer.evaluate_model('./test_dataset')
```

## 📊 Performa dan Metrik

DeepSteg menggunakan berbagai metrik untuk evaluasi kualitas steganografi:

- **PSNR (Peak Signal-to-Noise Ratio)**: Mengukur kualitas rekonstruksi citra
- **SSIM (Structural Similarity Index)**: Mengukur similarity struktural antar citra
- **Bit Accuracy**: Persentase bit pesan yang dapat di-decode dengan benar
- **Chi-Square Test**: Deteksi statistical anomalies yang menunjukkan steganografi
- **RS Analysis**: Regular-Singular analysis untuk deteksi steganografi
- **Histogram Analysis**: Analisis distorsi histogram untuk deteksi LSB steganografi

## 🏗️ Struktur Proyek

```
deepsteg/
├── app.py                  # Web application (Flask)
├── deepsteg_cli.py         # Command-line interface
├── models/                 # Core functionality
│   ├── __init__.py         # Package initialization
│   ├── enhanced_encoder.py # Encoder with encryption & compression
│   ├── enhanced_decoder.py # Decoder with decryption & decompression
│   ├── enhanced_gan_model.py # GAN steganography model
│   ├── crypto.py           # Encryption utilities
│   ├── utils.py            # General utilities
│   ├── steganalysis.py     # Steganalysis detection
│   ├── metrics.py          # Quality metrics
│   ├── benchmark.py        # Benchmarking functionality
│   ├── dataset.py          # Dataset management
│   ├── pretrained_model_manager.py # Pre-trained model management
│   ├── pretrained_detector.py # Pre-trained detector management
│   ├── utils_pretrained.py # Utilities for pre-trained models
│   ├── transfer_learning.py # Transfer learning & fine-tuning
│   └── saved/              # Saved models
├── static/                 # Static web assets
├── templates/              # HTML templates
└── requirements.txt        # Dependencies
```

## 🔬 Dasar Teknis

DeepSteg menggunakan pendekatan GAN (Generative Adversarial Networks) untuk steganografi dengan kelebihan:

1. **Imperceptibility Tinggi**: Hasil steganografi yang hampir tidak terdeteksi secara visual
2. **Kapasitas Optimal**: Penyesuaian kapasitas berdasarkan karakteristik citra
3. **Resistensi terhadap Steganalysis**: Desain yang lebih tahan terhadap deteksi otomatis
4. **Model Adaptif**: Fine-tuning untuk kasus penggunaan spesifik

Implementasi GAN terdiri dari:
- **Encoder Network**: Menyembunyikan pesan dalam citra
- **Decoder Network**: Mengekstrak pesan dari citra stego
- **Discriminator Network**: Membedakan antara citra asli dan stego

## 🛡️ Keamanan

DeepSteg menyediakan beberapa fitur keamanan:
- **Enkripsi AES-256**: Enkripsi pesan sebelum embedding
- **Password-based Key Derivation**: Penggunaan PBKDF2 untuk derivasi kunci
- **Pseudo-random Bit Distribution**: Distribusi bit pesan secara pseudo-random
- **Model Adversarial**: Latihan model untuk menghindari deteksi

## 🤝 Kontribusi

Kontribusi sangat diterima! Jika Anda ingin berkontribusi:

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/amazing-feature`)
3. Commit perubahan (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buka Pull Request

## 📄 Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Kontak

Developer - [@yourgithub](https://github.com/yourgithub)

Project Link: [https://github.com/yourusername/deepsteg](https://github.com/yourusername/deepsteg)