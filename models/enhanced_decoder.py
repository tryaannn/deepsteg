"""
enhanced_decoder.py
Modul untuk decoding pesan dengan mendukung dekripsi dan dekompresi
"""

import numpy as np
import cv2
import os
import logging
import time
import zlib
from models.utils import bits_to_text, load_model_if_exists
from models.crypto import MessageEncryptor

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
                logger.info("Model decoder tidak ditemukan, menggunakan LSB secure fallback")
                
                # Buat direktori untuk menyimpan model jika belum ada
                os.makedirs('models/saved', exist_ok=True)
                return False
            return True
            
        except Exception as e:
            logger.error(f"Error loading decoder model: {str(e)}", exc_info=True)
            decoder_model = None
            return False
    
    return decoder_model is not None

def decode_message(image, password=None):
    """
    Mengekstrak pesan dari gambar menggunakan model Deep Learning atau LSB fallback
    dengan support untuk dekripsi
    
    Args:
        image (numpy.ndarray): Gambar stego dalam format RGB
        password (str, optional): Password untuk dekripsi jika pesan terenkripsi
        
    Returns:
        tuple: (message, metadata) - Pesan yang berhasil diekstrak dan metadata
    
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
    metadata = {
        'execution_time': 0,
        'method': 'unknown',
        'is_encrypted': False,
        'is_compressed': False,
        'password_required': False,
        'password_correct': None
    }
    
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
            metadata['method'] = 'deep_learning'
            
        except Exception as e:
            logger.error(f"Error in model-based decoding: {str(e)}", exc_info=True)
            logger.info("Falling back to LSB method")
            message = None  # Reset message to try LSB
    
    # Jika tidak ada pesan yang ditemukan dari model, coba LSB
    if message is None:
        # LSB fallback method
        logger.info("Using LSB secure fallback method for decoding")
        try:
            message = secure_lsb_decode(image)
            metadata['method'] = 'lsb_secure'
        except Exception as e:
            logger.error(f"Error in LSB decoding: {str(e)}", exc_info=True)
            raise ValueError(f"Gagal mengekstrak pesan dari gambar: {str(e)}")
    
    # Jika berhasil mendapatkan pesan, cek apakah terenkripsi atau terkompresi
    if message:
        # Cek apakah pesan terenkripsi
        if message.startswith("ENCRYPTED:"):
            metadata['is_encrypted'] = True
            metadata['password_required'] = True
            encrypted_part = message[10:]  # Hapus "ENCRYPTED:"
            
            # Dekripsi jika password diberikan
            if password:
                try:
                    encryptor = MessageEncryptor()
                    decrypted = encryptor.decrypt(encrypted_part, password)
                    message = decrypted
                    metadata['password_correct'] = True
                    logger.info("Message successfully decrypted")
                except Exception as e:
                    logger.error(f"Decryption failed: {str(e)}", exc_info=True)
                    metadata['password_correct'] = False
                    return "ENCRYPTED_MESSAGE", metadata
            else:
                # Jika tidak ada password yang diberikan
                return "ENCRYPTED_MESSAGE", metadata
        
        # Cek apakah pesan terkompresi (setelah dekripsi)
        if message.startswith("COMPRESSED:"):
            metadata['is_compressed'] = True
            compressed_hex = message[11:]  # Hapus "COMPRESSED:"
            
            try:
                # Konversi hex string kembali ke bytes dan dekompresi
                compressed_data = bytes.fromhex(compressed_hex)
                decompressed = zlib.decompress(compressed_data)
                message = decompressed.decode('utf-8')
                logger.info("Message successfully decompressed")
            except Exception as e:
                logger.error(f"Decompression failed: {str(e)}", exc_info=True)
                return "COMPRESSION_ERROR", metadata
    
    # Log waktu eksekusi
    execution_time = time.time() - start_time
    metadata['execution_time'] = execution_time
    logger.info(f"Decoding completed in {execution_time:.2f} seconds")
    
    if message:
        logger.info(f"Successfully extracted message of length: {len(message)}")
    else:
        logger.warning("No message found or empty message")
        message = ""
    
    return message, metadata

def secure_lsb_decode(image):
    """
    Implementasi ekstraksi steganografi LSB (Least Significant Bit) yang lebih aman
    dengan dukungan untuk pola pseudo-random
    
    Args:
        image (numpy.ndarray): Gambar stego dalam format RGB
        
    Returns:
        str: Pesan yang diekstrak
        
    Raises:
        ValueError: Jika terjadi error saat decoding
    """
    try:
        # Step 1: Baca panjang pesan dari 32 bit pertama (header)
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
        
        # Step 2: Generate pseudo-random sequence untuk bit ekstraksi
        seed = (image.shape[0] * image.shape[1]) % 1000000
        np.random.seed(seed)
        
        # Buat indeks acak untuk penyisipan bit (harus sama dengan encoder)
        max_index = image.shape[0] * image.shape[1] * 3
        indices = np.random.permutation(max_index - 32) + 32  # Skip header bits
        indices = indices[:msg_len]  # Ambil hanya yang diperlukan
        
        # Step 3: Ekstrak pesan menggunakan pola pseudo-random
        message_bits = ""
        for i in range(msg_len):
            try:
                # Ambil indeks dari sequence acak
                idx = indices[i]
                
                # Hitung posisi pixel dan channel
                pixel_index = idx // 3
                row = pixel_index // image.shape[1]
                col = pixel_index % image.shape[1]
                channel = idx % 3
                
                # Validasi bounds
                if row >= image.shape[0] or col >= image.shape[1]:
                    logger.warning(f"Message truncated at bit {i}/{msg_len} due to image bounds")
                    break
                
                # Ambil bit paling tidak signifikan
                message_bits += str(image[row, col, channel] & 1)
            except IndexError:
                logger.error(f"IndexError during LSB decoding at bit {i}/{msg_len}")
                break
        
        # Step 4: Konversi bit ke teks
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
        
        logger.info(f"LSB secure decoding successful: extracted {len(message)} characters")
        return message
        
    except Exception as e:
        logger.error(f"Error during LSB secure decoding: {str(e)}", exc_info=True)
        raise ValueError(f"Terjadi kesalahan saat mengekstrak pesan: {str(e)}")