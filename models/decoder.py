# decoder.py
import numpy as np
import cv2
import os
import logging
import time
from models.utils import bits_to_text, load_model_if_exists

# Konfigurasi logging
logger = logging.getLogger(__name__)

# Model GAN yang telah dilatih
decoder_model = None

def init_model():
    """
    Inisialisasi model jika belum ada
    
    Returns:
        bool: True jika model berhasil dimuat, False jika menggunakan fallback LSB
    """
    global decoder_model
    
    if decoder_model is None:
        # Coba muat model yang sudah dilatih
        try:
            decoder_model = load_model_if_exists('models/saved/decoder.h5')
            
            # Jika tidak ada, gunakan steganografi LSB sederhana
            if decoder_model is None:
                logger.info("Model decoder tidak ditemukan, menggunakan LSB fallback")
                
                # Buat direktori untuk menyimpan model jika belum ada
                os.makedirs('models/saved', exist_ok=True)
                return False
            return True
            
        except Exception as e:
            logger.error(f"Error loading decoder model: {str(e)}", exc_info=True)
            decoder_model = None
            return False
    
    return decoder_model is not None

def decode_message(image):
    """
    Mengekstrak pesan dari gambar menggunakan model Deep Learning atau LSB fallback
    
    Args:
        image (numpy.ndarray): Gambar stego dalam format RGB
        
    Returns:
        str: Pesan yang berhasil diekstrak
    
    Raises:
        ValueError: Jika input tidak valid
    """
    # Waktu mulai untuk performa
    start_time = time.time()
    
    # Validasi input
    if image is None:
        raise ValueError("Gambar input tidak boleh None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Gambar input harus berupa NumPy array")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Gambar harus dalam format RGB, shape saat ini: {image.shape}")
    
    # Log info gambar
    logger.info(f"Decoding message from image of shape {image.shape}")
    
    # Initialize model if needed
    model_available = init_model()
    
    message = None
    
    # Jika model tersedia, gunakan model
    if model_available:
        try:
            # Reshape dan normalisasi gambar untuk model
            img_input = np.expand_dims(image.astype('float32') / 255.0, axis=0)
            
            # Prediksi menggunakan model
            message_bits = decoder_model.predict(img_input)[0]
            
            # Konversi probabilitas ke bit (threshold)
            message_bits = (message_bits > 0.5).astype(int)
            
            # Konversi bit ke teks
            message = bits_to_text(message_bits)
            logger.info("Successfully used deep learning model for decoding")
            
        except Exception as e:
            logger.error(f"Error in model-based decoding: {str(e)}", exc_info=True)
            logger.info("Falling back to LSB method")
            message = None  # Reset message to try LSB
    
    # Jika tidak ada pesan yang ditemukan dari model, coba LSB
    if message is None:
        # LSB fallback method
        logger.info("Using LSB fallback method for decoding")
        try:
            message = lsb_decode(image)
        except Exception as e:
            logger.error(f"Error in LSB decoding: {str(e)}", exc_info=True)
            raise ValueError(f"Gagal mengekstrak pesan dari gambar: {str(e)}")
    
    # Log waktu eksekusi
    execution_time = time.time() - start_time
    logger.info(f"Decoding completed in {execution_time:.2f} seconds")
    
    if message:
        logger.info(f"Successfully extracted message of length: {len(message)}")
    else:
        logger.warning("No message found or empty message")
    
    return message

def lsb_decode(image):
    """
    Implementasi ekstraksi steganografi LSB (Least Significant Bit) sederhana
    
    Args:
        image (numpy.ndarray): Gambar stego dalam format RGB
        
    Returns:
        str: Pesan yang diekstrak
        
    Raises:
        ValueError: Jika terjadi error saat decoding
    """
    try:
        # Step 1: Baca panjang pesan dari 32 bit pertama
        msg_len_bits = ""
        bit_index = 0
        
        for i in range(32):
            # Hitung posisi pixel dan channel
            pixel_index = bit_index // 3
            row = pixel_index // image.shape[1]
            col = pixel_index % image.shape[1]
            channel = bit_index % 3
            
            # Validasi bounds
            if row >= image.shape[0] or col >= image.shape[1]:
                raise ValueError(f"Header incomplete: index out of bounds {row},{col} for shape {image.shape[:2]}")
            
            # Ambil bit paling tidak signifikan
            msg_len_bits += str(image[row, col, channel] & 1)
            bit_index += 1
        
        # Konversi ke integer
        try:
            msg_len = int(msg_len_bits, 2)
        except ValueError:
            raise ValueError(f"Invalid header bitstring: {msg_len_bits}")
        
        # Validasi panjang pesan
        max_bits = image.shape[0] * image.shape[1] * 3 - 32  # Dikurangi panjang header
        if msg_len <= 0 or msg_len > max_bits:
            logger.warning(f"Invalid message length detected: {msg_len} bits (max: {max_bits} bits)")
            return "Tidak ada pesan yang valid terdeteksi dalam gambar ini."
        
        # Step 2: Ekstrak pesan
        message_bits = ""
        for i in range(msg_len):
            try:
                # Hitung posisi pixel dan channel
                pixel_index = bit_index // 3
                row = pixel_index // image.shape[1]
                col = pixel_index % image.shape[1]
                channel = bit_index % 3
                
                # Validasi bounds
                if row >= image.shape[0] or col >= image.shape[1]:
                    logger.warning(f"Message truncated at bit {i}/{msg_len} due to image bounds")
                    break
                
                # Ambil bit paling tidak signifikan
                message_bits += str(image[row, col, channel] & 1)
                bit_index += 1
            except IndexError:
                logger.error(f"IndexError during LSB decoding at bit {i}/{msg_len}")
                break
        
        # Step 3: Konversi bit ke teks
        message = ""
        for i in range(0, len(message_bits), 8):
            if i + 8 <= len(message_bits):
                byte = message_bits[i:i+8]
                if byte == "00000000":  # Terminator
                    break
                try:
                    char_code = int(byte, 2)
                    # Validasi karakter printable (ASCII 32-126)
                    if 32 <= char_code <= 126 or char_code in [9, 10, 13]:  # TAB, LF, CR
                        message += chr(char_code)
                    else:
                        logger.warning(f"Skipping non-printable character (code: {char_code}) at position {i//8}")
                except ValueError:
                    logger.error(f"Invalid byte sequence: {byte} at position {i//8}")
                    continue
        
        if not message:
            return "Tidak ada pesan yang dapat dibaca dari gambar ini."
        
        logger.info(f"LSB decoding successful: extracted {len(message)} characters")
        return message
        
    except Exception as e:
        logger.error(f"Error during LSB decoding: {str(e)}", exc_info=True)
        raise ValueError(f"Terjadi kesalahan saat mengekstrak pesan: {str(e)}")