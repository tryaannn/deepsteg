# DeepSteg: Deep Learning Steganography Optimizer

<div align="center">
  <img src="static/img/logo.svg" alt="DeepSteg Logo" width="600">
  
  <p><strong>Advanced image steganography using Generative Adversarial Networks</strong></p>

  ![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
  ![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)
  ![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
</div>

## 🔍 Overview

DeepSteg is a cutting-edge application for image steganography that leverages deep learning and Generative Adversarial Networks (GANs) to optimize the balance between message capacity and visual imperceptibility. The tool provides a user-friendly web interface as well as a powerful command-line interface for automating steganography tasks.

Unlike traditional steganography methods that rely on simple bit manipulation, DeepSteg uses neural networks to learn optimal embedding patterns, resulting in higher capacity and better resistance to detection while maintaining image quality.

## ✨ Key Features

- **🧠 GAN-based Steganography**: Utilizes advanced Generative Adversarial Networks to hide messages within images
- **🔐 Message Encryption**: Built-in AES encryption for enhanced security of hidden content
- **🗜️ Adaptive Compression**: Intelligent message compression to maximize embedding capacity
- **📊 Comprehensive Metrics**: Detailed quality assessment with PSNR, SSIM, and other metrics
- **🎯 Adaptive Encoding**: Smart adjustment of encoding based on image characteristics
- **🕵️ Built-in Steganalysis**: Integrated tools to evaluate steganography security
- **🖥️ Dual Interface**: Intuitive web interface and flexible command-line tools
- **📱 Pre-trained Models**: Ready-to-use models for various capacity and quality preferences

## 🚀 Installation

### Prerequisites

- Python 3.8 or newer
- PIP (Python Package Manager)
- TensorFlow 2.x

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepsteg.git
   cd deepsteg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models (optional but recommended):
   ```bash
   python download_pretrained.py --download-all-models
   ```

4. Launch the application:
   ```bash
   # Web Interface
   python app.py

   # Command Line Help
   python deepsteg_cli.py --help
   ```

## 💻 Usage Examples

### Web Interface

Access the web interface by navigating to `http://localhost:5000` in your browser. The intuitive UI allows you to:
- Upload cover images
- Enter secret messages
- Choose encryption and compression options
- Select steganography models
- Analyze images for hidden content

### Command Line Interface

DeepSteg CLI provides powerful tools for steganography operations:

#### List available models
```bash
python deepsteg_cli.py list-models
```

#### Hide a message in an image
```bash
python deepsteg_cli.py encode --model gan_basic --image cover.jpg --message "This is a secret message" --output stego.png
```

#### Extract a hidden message
```bash
python deepsteg_cli.py decode --model gan_basic --image stego.png
```

#### Analyze an image for hidden content
```bash
python deepsteg_cli.py analyze --image suspect.png --detector cnn_basic --visualize
```

#### Train a custom model
```bash
python deepsteg_cli.py train --base-model gan_basic --dataset ./my_dataset --name my_custom_model --epochs 20
```

## 📊 Performance Comparison

| Method | PSNR (dB) | SSIM | Max Capacity | Detection Resistance | Speed |
|--------|-----------|------|--------------|----------------------|-------|
| LSB | 51.1 | 0.9998 | 1 bit/pixel | Low | Very Fast |
| LSB Matching | 51.0 | 0.9997 | 1 bit/pixel | Medium | Fast |
| Adaptive LSB | 49.8 | 0.9994 | 1-3 bits/pixel | Medium | Fast |
| DeepSteg Basic | 42.3 | 0.9985 | 0.4 bits/pixel | High | Medium |
| DeepSteg GAN | 38.6 | 0.9932 | 0.8 bits/pixel | Very High | Slow |

## 🛠️ Technical Foundation

DeepSteg's architecture consists of several key components:

1. **Encoder Network**: Learns to hide messages within cover images
2. **Decoder Network**: Extracts hidden messages from stego images
3. **Discriminator Network**: Helps improve imperceptibility by detecting modifications
4. **Encryption Module**: Provides AES-256 encryption for message security
5. **Steganalysis Detectors**: Evaluates the security of steganography methods

The GAN-based approach allows DeepSteg to:
- Learn optimal embedding patterns for each specific image
- Distribute message bits based on image texture complexity
- Adapt to different image characteristics automatically
- Maximize capacity while maintaining visual quality

## 🔒 Security Features

DeepSteg implements several security enhancements:

- **AES-256 Encryption**: Military-grade encryption for message content
- **Password-based Key Derivation**: Secure key generation using PBKDF2
- **Pseudo-random Bit Distribution**: Non-sequential message embedding
- **Anti-steganalysis Optimization**: Model trained to resist detection algorithms

## 🤝 Contributing

Contributions are welcome! If you'd like to help improve DeepSteg:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The DeepSteg team for their contributions
- The steganography research community
- TensorFlow team for deep learning framework
- Flask team for web framework

---

<div align="center">
  <p>Built with ❤️ for the steganography and deep learning community</p>
  <p>© 2025 DeepSteg Team</p>
</div>