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
- pretrained_model_manager: Pengelolaan pre-trained models
- pretrained_detector: Pengelolaan pre-trained detectors
- utils_pretrained: Utility untuk pre-trained models
- transfer_learning: Transfer learning dan fine-tuning
"""

__version__ = "2.0.0"
__author__ = "DeepSteg Team"

# Import utama untuk kemudahan akses
from .enhanced_encoder import encode_message
from .enhanced_decoder import decode_message
from .crypto import MessageEncryptor, calculate_password_strength
from .utils import preprocess_image, postprocess_image, calculate_metrics

# Import pre-trained functionality
try:
    from .pretrained_model_manager import PretrainedModelManager
    from .pretrained_detector import PretrainedDetector
    from .utils_pretrained import optimize_input_image, calculate_embedding_capacity
    from .transfer_learning import TransferLearning
except ImportError:
    # Optional modules might not be available
    pass

__all__ = [
    # Core functionality
    'encode_message',
    'decode_message', 
    'MessageEncryptor',
    'calculate_password_strength',
    'preprocess_image',
    'postprocess_image',
    'calculate_metrics',
    
    # Pre-trained functionality
    'PretrainedModelManager',
    'PretrainedDetector',
    'optimize_input_image',
    'calculate_embedding_capacity',
    'TransferLearning'
]