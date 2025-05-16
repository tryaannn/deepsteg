"""
DeepSteg Models Package

Package ini berisi semua model dan utility yang diperlukan untuk aplikasi steganografi:
- enhanced_encoder: Encoder dengan enkripsi dan kompresi
- enhanced_decoder: Decoder dengan dekripsi dan dekompresi  
- enhanced_gan_model: Model GAN yang ditingkatkan
- crypto: Utility untuk enkripsi/dekripsi
- utils: Utility umum untuk processing gambar dan metrik
- steganalysis: Analisis deteksi steganografi
- metrics: Evaluasi metrik lanjutan
- train: Script untuk training model
- benchmark: Benchmarking berbagai metode
- dataset: Manajemen dataset
"""

__version__ = "1.0.0"
__author__ = "DeepSteg Team"

# Import utama untuk kemudahan akses
from .enhanced_encoder import encode_message
from .enhanced_decoder import decode_message
from .crypto import MessageEncryptor, calculate_password_strength
from .utils import preprocess_image, postprocess_image, calculate_metrics

__all__ = [
    'encode_message',
    'decode_message', 
    'MessageEncryptor',
    'calculate_password_strength',
    'preprocess_image',
    'postprocess_image',
    'calculate_metrics'
]