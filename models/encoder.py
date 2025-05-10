# encoder.py
import numpy as np
import os
import cv2
import logging
import time
from models.utils import text_to_bits, calculate_metrics, load_model_if_exists

# Konfigurasi logging
logger = logging.getLogger(__name__)

# Model GAN yang telah dilatih
encoder_model = None

def init_model():
    """
    Inisialisasi model jika belum ada
    
    Returns:
        bool: True jika model berhasil dimuat, False jika menggunakan fallback LSB
    """
    global encoder_model
    
    if encoder_model is None:
        # Coba muat model yang sudah dilatih
        try:
            encoder_model = load_model_if_exists('models/saved/encoder.h5')
            
            # Jika tidak ada, gunakan steganografi LSB sederhana
            if encoder_model is None:
                logger.info("Model encoder tidak ditemukan, menggunakan LSB fallback")
                
                # Buat direktori untuk menyimpan model jika belum ada
                os.makedirs('models/saved', exist_ok=True)
                return False
            return True
            
        except Exception as e:
            logger.error(f"Error loading encoder model: {str(e)}", exc_info=True)
            encoder_model = None
            return False
    
    return encoder_model is not None

def encode_message(image, message):
    """
    Menyembunyikan pesan dalam gambar menggunakan model Deep Learning atau LSB fallback
    
    Args:
        image (numpy.ndarray): Gambar input dalam format RGB
        message (str): Pesan teks yang akan disembunyikan
        
    Returns:
        tuple: (stego_image, metrics) - Gambar stego dan metrik kualitasnya
    
    Raises:
        ValueError: Jika input tidak valid atau kapasitas gambar tidak mencukupi
    """
    # Waktu mulai untuk performa
    start_time = time.time()
    
    # Validasi input
    if image is None:
        raise ValueError("Gambar input tidak boleh None")
        
    if not isinstance(message, str) or not message:
        raise ValueError("Pesan harus berupa string dan tidak boleh kosong")
    
    # Cek kapasitas gambar (estimasi kasar: 1 bit per channel)
    img_capacity_bits = image.shape[0] * image.shape[1] * 3  # 3 channels (RGB)
    message_size_bits = len(message) * 8  # 8 bits per karakter
    
    logger.info(f"Image size: {image.shape[0]}x{image.shape[1]}, capacity: {img_capacity_bits} bits")
    logger.info(f"Message length: {len(message)} chars, size: {message_size_bits} bits")
    
    # Periksa kapasitas dengan margin keamanan
    if message_size_bits > (img_capacity_bits * 0.9):  # 90% kapasitas maksimum
        logger.warning(f"Message may be too large for this image: {message_size_bits} bits > {img_capacity_bits*0.9} bits")
    
    # Simpan gambar asli untuk perbandingan
    original_img = image.copy()
    
    # Log info gambar
    logger.info(f"Encoding message of length {len(message)} into image of shape {image.shape}")
    
    # Initialize model if needed
    model_available = init_model()
    
    # Jika model tersedia, gunakan model
    if model_available:
        try:
            # Konversi teks ke bit
            message_bits = text_to_bits(message)
            
            # Pad atau potong ke panjang yang diharapkan oleh model
            expected_length = 100  # Harus sama dengan model.message_length
            
            if len(message_bits) > expected_length:
                logger.warning(f"Message truncated from {len(message_bits)} to {expected_length} bits for model")
                message_bits = message_bits[:expected_length]
            else:
                message_bits = np.pad(message_bits, (0, expected_length - len(message_bits)))
            
            # Reshape dan normalisasi gambar untuk model
            img_input = np.expand_dims(image.astype('float32') / 255.0, axis=0)
            msg_input = np.expand_dims(message_bits, axis=0)
            
            # Prediksi menggunakan model
            stego_img = encoder_model.predict([img_input, msg_input])[0]
            
            # Denormalisasi
            stego_img = np.clip(stego_img * 255.0, 0, 255).astype(np.uint8)
            
            logger.info("Successfully used deep learning model for encoding")
        except Exception as e:
            logger.error(f"Error in model-based encoding: {str(e)}", exc_info=True)
            logger.info("Falling back to LSB method")
            stego_img = lsb_encode(image, message)
    else:
        # LSB fallback method
        logger.info("Using LSB fallback method for encoding")
        stego_img = lsb_encode(image, message)
    
    # Hitung metrik kualitas
    metrics = calculate_metrics(original_img, stego_img)
    
    # Tambahkan waktu eksekusi
    execution_time = time.time() - start_time
    metrics['execution_time'] = float(execution_time)
    
    logger.info(f"Encoding completed in {execution_time:.2f} seconds")
    logger.info(f"Encoding metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    
    return stego_img, metrics

def lsb_encode(image, message):
    """
    Implementasi steganografi LSB (Least Significant Bit) sederhana
    
    Args:
        image (numpy.ndarray): Gambar input dalam format RGB
        message (str): Pesan teks yang akan disembunyikan
        
    Returns:
        numpy.ndarray: Gambar stego hasil penyembunyian pesan
        
    Raises:
        ValueError: Jika gambar terlalu kecil untuk pesan
    """
    # Konversi pesan ke bit
    message_bits = ''.join([format(ord(c), '08b') for c in message])
    message_bits += '00000000'  # Terminator
    
    # Validasi kapasitas
    max_bits = image.shape[0] * image.shape[1] * 3 - 32  # Dikurangi panjang header
    if len(message_bits) > max_bits:
        logger.warning(f"Message too long ({len(message_bits)} bits), truncating to fit ({max_bits} bits)")
        if max_bits <= 40:  # Terlalu kecil bahkan untuk header + minimal pesan
            raise ValueError(f"Gambar terlalu kecil untuk pesan apapun. Min. {(len(message_bits) + 32) // 3} pixels needed.")
        message_bits = message_bits[:max_bits]
    
    # Copy gambar untuk tidak mengubah original
    stego = image.copy()
    
    # Sembunyikan panjang pesan di 32 bit pertama
    msg_len = len(message_bits)
    msg_len_bits = format(msg_len, '032b')
    
    bit_index = 0
    total_pixels = image.shape[0] * image.shape[1]
    
    try:
        # Step 1: Sembunyikan panjang pesan (32 bit)
        for i in range(32):
            # Hitung posisi pixel dan channel
            pixel_index = bit_index // 3
            row = pixel_index // image.shape[1]
            col = pixel_index % image.shape[1]
            channel = bit_index % 3
            
            # Validasi bounds
            if row >= image.shape[0] or col >= image.shape[1]:
                raise ValueError(f"Index out of bounds: {row},{col} for shape {image.shape[:2]}")
            
            # Set bit paling tidak signifikan
            stego[row, col, channel] = (stego[row, col, channel] & 0xFE) | int(msg_len_bits[i])
            bit_index += 1
        
        # Step 2: Sembunyikan pesan
        for i in range(len(message_bits)):
            # Hitung posisi pixel dan channel
            pixel_index = bit_index // 3
            row = pixel_index // image.shape[1]
            col = pixel_index % image.shape[1]
            channel = bit_index % 3
            
            # Validasi bounds
            if row >= image.shape[0] or col >= image.shape[1]:
                # Jika melebihi bounds, hentikan encoding dan truncate pesan
                logger.warning(f"Message truncated at {i}/{len(message_bits)} bits due to image size")
                break
            
            # Set bit paling tidak signifikan
            stego[row, col, channel] = (stego[row, col, channel] & 0xFE) | int(message_bits[i])
            bit_index += 1
        
        logger.info(f"LSB encoding successful: embedded {min(len(message_bits), i+1)} of {len(message_bits)} bits")
        
    except IndexError as e:
        logger.error(f"IndexError during LSB encoding: {str(e)}", exc_info=True)
        raise ValueError(f"Gambar terlalu kecil untuk pesan ini. Maksimum kapasitas: {max_bits//8} karakter.")
    except Exception as e:
        logger.error(f"Error during LSB encoding: {str(e)}", exc_info=True)
        raise ValueError(f"Error saat encoding pesan: {str(e)}")
    
    return stego