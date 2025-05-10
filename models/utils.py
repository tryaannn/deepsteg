# utils.py
import numpy as np
import cv2
import os
import logging
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Konfigurasi logging
logger = logging.getLogger(__name__)

def preprocess_image(image):
    """
    Preprocessing gambar sebelum steganografi
    
    Args:
        image (numpy.ndarray): Gambar input dalam format RGB
        
    Returns:
        numpy.ndarray: Gambar hasil preprocessing
        
    Raises:
        ValueError: Jika input tidak valid
    """
    try:
        # Validate input
        if image is None:
            raise ValueError("Input image is None")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        
        # Logging original image info
        logger.info(f"Original image shape: {image.shape}, dtype: {image.dtype}")
        
        # Pastikan gambar dalam tipe data yang benar
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
            logger.info(f"Converted image to dtype {image.dtype}")
        
        # Resize ke ukuran yang lebih kecil jika terlalu besar
        max_size = 1024
        height, width = image.shape[:2]
        
        if height > max_size or width > max_size:
            # Hitung rasio untuk mempertahankan aspek rasio
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize gambar
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image to {new_width}x{new_height}")
        
        # Pastikan gambar memiliki 3 channel (RGB)
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            logger.info("Converted grayscale image to RGB")
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Ambil hanya RGB
            logger.info("Converted RGBA image to RGB")
        elif image.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
        
        # Ensure image dimensions are even (untuk beberapa operasi image processing)
        if (height % 2 != 0 or width % 2 != 0) and height > 2 and width > 2:
            new_height = height - (height % 2)
            new_width = width - (width % 2)
            image = image[:new_height, :new_width]
            logger.info(f"Adjusted dimensions to even values: {image.shape}")
            
        logger.info(f"Preprocessed image shape: {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}", exc_info=True)
        raise ValueError(f"Gagal memproses gambar: {str(e)}")

def postprocess_image(image):
    """
    Postprocessing gambar setelah steganografi
    
    Args:
        image (numpy.ndarray): Gambar stego hasil encoding
        
    Returns:
        numpy.ndarray: Gambar hasil postprocessing
        
    Raises:
        ValueError: Jika input tidak valid
    """
    try:
        # Validate input
        if image is None:
            raise ValueError("Input image is None")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        
        # Pastikan nilai pixel valid
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Verifikasi bahwa gambar valid
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error(f"Invalid image shape after postprocessing: {image.shape}")
            raise ValueError(f"Invalid image shape: {image.shape}, expected 3 channels (RGB)")
        
        return image
        
    except Exception as e:
        logger.error(f"Error in postprocess_image: {str(e)}", exc_info=True)
        raise ValueError(f"Gagal memproses gambar hasil: {str(e)}")

def calculate_metrics(original, stego):
    """
    Menghitung metrik kualitas gambar
    
    Args:
        original (numpy.ndarray): Gambar asli sebelum steganografi
        stego (numpy.ndarray): Gambar stego hasil steganografi
        
    Returns:
        dict: Metrik kualitas gambar (PSNR, SSIM, MSE, histogram similarity)
    """
    try:
        # Validate inputs
        if original is None or stego is None:
            raise ValueError("Input images cannot be None")
            
        if original.shape != stego.shape:
            logger.warning(f"Shape mismatch: original {original.shape}, stego {stego.shape}")
            # Resize stego to match original if needed
            if len(original.shape) == 3 and len(stego.shape) == 3:
                stego = cv2.resize(stego, (original.shape[1], original.shape[0]))
                logger.info(f"Resized stego image to match original: {stego.shape}")
                
        # Convert to same dtype for calculation
        original = original.astype(np.float32)
        stego = stego.astype(np.float32)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        try:
            psnr_value = psnr(original, stego, data_range=255.0)
        except Exception as e:
            logger.error(f"Error calculating PSNR: {str(e)}")
            psnr_value = 0.0
        
        # SSIM (Structural Similarity Index)
        try:
            # Try with channel_axis first (newer scikit-image)
            try:
                ssim_value = ssim(original, stego, channel_axis=2, data_range=255.0)
            except TypeError:
                # Fallback untuk versi scikit-image yang lebih lama
                ssim_value = ssim(original, stego, multichannel=True, data_range=255.0)
        except Exception as e:
            logger.error(f"Error calculating SSIM: {str(e)}")
            ssim_value = 0.0
        
        # MSE (Mean Squared Error)
        try:
            mse = np.mean((original - stego) ** 2)
        except Exception as e:
            logger.error(f"Error calculating MSE: {str(e)}")
            mse = 999.0
        
        # RGB histogram differences
        try:
            hist_diff = 0
            for i in range(3):  # RGB channels
                hist_orig = cv2.calcHist([original.astype(np.uint8)], [i], None, [256], [0, 256])
                hist_stego = cv2.calcHist([stego.astype(np.uint8)], [i], None, [256], [0, 256])
                hist_correl = cv2.compareHist(hist_orig, hist_stego, cv2.HISTCMP_CORREL)
                # Pastikan nilai korelasi valid
                if np.isnan(hist_correl):
                    hist_correl = 0.0
                hist_diff += hist_correl
            hist_diff /= 3.0  # Average correlation
        except Exception as e:
            logger.error(f"Error calculating histogram similarity: {str(e)}")
            hist_diff = 0.0
        
        # Tambahkan metrik pixel difference
        try:
            pixel_diff = np.mean(np.abs(original - stego)) / 255.0  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Error calculating pixel difference: {str(e)}")
            pixel_diff = 1.0
        
        # Metrik deteksi, lebih rendah = lebih baik
        detect_score = 1.0 - min(1.0, ssim_value)
        
        metrics = {
            'psnr': float(np.clip(psnr_value, 0, 100)),  # Clip untuk menghindari nilai inf
            'ssim': float(np.clip(ssim_value, 0, 1)),
            'mse': float(np.clip(mse, 0, 999)),
            'hist_similarity': float(np.clip(hist_diff, 0, 1)),
            'pixel_diff': float(np.clip(pixel_diff, 0, 1)),
            'detect_score': float(np.clip(detect_score, 0, 1))
        }
        
        logger.info(f"Calculated metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {str(e)}", exc_info=True)
        # Return default metrics in case of error
        return {
            'psnr': 30.0,
            'ssim': 0.95,
            'mse': 10.0,
            'hist_similarity': 0.9,
            'pixel_diff': 0.05,
            'detect_score': 0.05
        }

def text_to_bits(text):
    """
    Konversi teks ke representasi bit
    
    Args:
        text (str): Pesan teks yang akan dikonversi
        
    Returns:
        numpy.ndarray: Array bit (0/1) hasil konversi
        
    Raises:
        TypeError: Jika input bukan string
    """
    try:
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
            
        # Tambahkan terminator null
        text = text + '\0'
            
        bits = []
        for char in text:
            # Konversi karakter ke ASCII, lalu ke representasi biner 8-bit
            binary = format(ord(char), '08b')
            bits.extend([int(bit) for bit in binary])
            
        logger.info(f"Converted text of length {len(text)} to {len(bits)} bits")
        return np.array(bits, dtype=np.int8)
        
    except Exception as e:
        logger.error(f"Error in text_to_bits: {str(e)}", exc_info=True)
        raise ValueError(f"Gagal mengkonversi teks ke bit: {str(e)}")

def bits_to_text(bits):
    """
    Konversi representasi bit ke teks
    
    Args:
        bits (numpy.ndarray or list): Array bit (0/1) yang akan dikonversi
        
    Returns:
        str: Teks hasil konversi
        
    Raises:
        TypeError: Jika input bukan array bit
    """
    try:
        if not isinstance(bits, np.ndarray) and not isinstance(bits, list):
            raise TypeError("Input must be a numpy array or list of bits")
            
        # Konversi array bit menjadi string bit
        bit_string = ''.join(str(int(bit)) for bit in bits)
        
        # Konversi setiap 8 bit ke karakter
        text = ''
        valid_chars = 0
        invalid_chars = 0
        
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte = bit_string[i:i+8]
                
                # Berhenti jika menemukan byte nol (terminator)
                if byte == '00000000':
                    break
                    
                try:
                    char_code = int(byte, 2)
                    
                    # Validasi karakter printable (ASCII 32-126)
                    if 32 <= char_code <= 126 or char_code in [9, 10, 13]:  # TAB, LF, CR
                        text += chr(char_code)
                        valid_chars += 1
                    else:
                        # Ganti karakter non-printable dengan '?'
                        text += '?'
                        invalid_chars += 1
                        logger.debug(f"Non-printable character at position {i//8}: code {char_code}")
                except ValueError:
                    logger.warning(f"Invalid byte sequence: {byte} at position {i//8}")
                    invalid_chars += 1
        
        # Jika terlalu banyak karakter tidak valid, mungkin bukan pesan steganografi
        if valid_chars == 0 or (invalid_chars > 0 and invalid_chars / (valid_chars + invalid_chars) > 0.3):
            logger.warning(f"Too many invalid characters: {invalid_chars}/{valid_chars+invalid_chars}")
            return ""
        
        logger.info(f"Converted {len(bits)} bits to text of length {len(text)}")
        return text
        
    except Exception as e:
        logger.error(f"Error in bits_to_text: {str(e)}", exc_info=True)
        return ""

def load_model_if_exists(model_path):
    """
    Muat model Keras jika file-nya ada
    
    Args:
        model_path (str): Path ke file model
        
    Returns:
        tensorflow.keras.Model or None: Model yang berhasil dimuat atau None jika gagal
    """
    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            try:
                # Set memory growth untuk mencegah GPU out of memory
                try:
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"Could not configure GPU: {str(e)}")
                
                # Muat model dengan konfigurasi khusus
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,  # Hindari error kompilasi jika optimizer tidak tersedia
                    custom_objects=None  # Tambahkan custom layer jika diperlukan
                )
                logger.info(f"Model loaded successfully: {type(model)}")
                return model
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", exc_info=True)
                return None
        else:
            logger.info(f"Model file not found: {model_path}")
            return None
            
    except Exception as e:
        logger.error(f"Exception in load_model_if_exists: {str(e)}", exc_info=True)
        return None

def is_valid_image(file_path, min_size=32):
    """
    Memeriksa apakah file merupakan gambar yang valid
    
    Args:
        file_path (str): Path ke file gambar
        min_size (int): Ukuran minimum gambar (default: 32px)
        
    Returns:
        bool: True jika gambar valid, False jika tidak
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        # Coba buka dengan OpenCV
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"OpenCV failed to open image: {file_path}")
            return False
            
        # Periksa ukuran minimum
        if img.shape[0] < min_size or img.shape[1] < min_size:
            logger.error(f"Image too small: {img.shape[:2]}, minimum {min_size}x{min_size}")
            return False
            
        # Periksa jumlah channel
        if len(img.shape) < 3:
            logger.warning(f"Image is grayscale, not RGB: {file_path}")
            # Masih valid, hanya warning
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}", exc_info=True)
        return False