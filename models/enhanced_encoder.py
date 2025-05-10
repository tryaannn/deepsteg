"""
enhanced_encoder.py
Modul untuk encoding pesan dengan enkripsi dan tambahan fitur keamanan
"""

import numpy as np
import os
import cv2
import logging
import time
import zlib
from models.utils import text_to_bits, calculate_metrics, load_model_if_exists
from models.crypto import MessageEncryptor

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
            
            # Jika tidak ada, gunakan steganografi LSB secure
            if encoder_model is None:
                logger.info("Model encoder tidak ditemukan, menggunakan LSB secure fallback")
                
                # Buat direktori untuk menyimpan model jika belum ada
                os.makedirs('models/saved', exist_ok=True)
                return False
            return True
            
        except Exception as e:
            logger.error(f"Error loading encoder model: {str(e)}", exc_info=True)
            encoder_model = None
            return False
    
    return encoder_model is not None

def encode_message(image, message, password=None, compression_level=None):
    """
    Menyembunyikan pesan dalam gambar menggunakan model Deep Learning atau LSB fallback
    dengan tambahan enkripsi dan kompresi
    
    Args:
        image (numpy.ndarray): Gambar input dalam format RGB
        message (str): Pesan teks yang akan disembunyikan
        password (str, optional): Password untuk enkripsi. Default None (tanpa enkripsi)
        compression_level (int, optional): Level kompresi 0-9. Default None (tanpa kompresi)
        
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
    
    # Simpan gambar asli untuk perbandingan
    original_img = image.copy()
    
    # Log info gambar
    logger.info(f"Encoding message of length {len(message)} into image of shape {image.shape}")
    
    # Kompresi pesan jika level kompresi diberikan
    original_size = len(message.encode('utf-8'))
    if compression_level is not None and original_size > 100:  # Hanya kompresi jika lebih dari 100 bytes
        try:
            compressed_message = zlib.compress(message.encode('utf-8'), level=compression_level)
            compressed_size = len(compressed_message)
            compression_ratio = compressed_size / original_size
            
            # Gunakan hasil kompresi jika efektif (lebih kecil dari asli)
            if compression_ratio < 0.9:  # Minimal 10% pengurangan
                logger.info(f"Message compressed: {original_size} -> {compressed_size} bytes ({compression_ratio:.2f})")
                message = "COMPRESSED:" + compressed_message.hex()
            else:
                logger.info(f"Compression not effective ({compression_ratio:.2f}), using original message")
        except Exception as e:
            logger.error(f"Error compressing message: {str(e)}")
            # Continue without compression if there's an error
    
    # Enkripsi pesan jika password diberikan
    if password:
        try:
            encryptor = MessageEncryptor()
            encrypted_message = encryptor.encrypt(message, password)
            message = "ENCRYPTED:" + encrypted_message
            logger.info(f"Message encrypted using password")
        except Exception as e:
            logger.error(f"Error encrypting message: {str(e)}", exc_info=True)
            raise ValueError(f"Gagal mengenkripsi pesan: {str(e)}")
    
    # Cek kapasitas gambar (estimasi kasar: 1 bit per channel)
    img_capacity_bits = image.shape[0] * image.shape[1] * 3  # 3 channels (RGB)
    message_size_bits = len(message) * 8  # 8 bits per karakter
    
    logger.info(f"Image size: {image.shape[0]}x{image.shape[1]}, capacity: {img_capacity_bits} bits")
    logger.info(f"Message length: {len(message)} chars, size: {message_size_bits} bits")
    
    # Periksa kapasitas dengan margin keamanan
    if message_size_bits > (img_capacity_bits * 0.9):  # 90% kapasitas maksimum
        logger.warning(f"Message may be too large for this image: {message_size_bits} bits > {img_capacity_bits*0.9} bits")
        if message_size_bits > img_capacity_bits - 40:  # Account for header bits (32) + terminator (8)
            raise ValueError(f"Pesan terlalu besar untuk gambar ini. Coba gunakan gambar yang lebih besar atau aktifkan kompresi.")
    
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
            logger.info("Falling back to LSB secure method")
            stego_img = secure_lsb_encode(image, message)
    else:
        # LSB fallback method
        logger.info("Using LSB secure method for encoding")
        stego_img = secure_lsb_encode(image, message)
    
    # Hitung metrik kualitas
    metrics = calculate_metrics(original_img, stego_img)
    
    # Tambahkan waktu eksekusi
    execution_time = time.time() - start_time
    metrics['execution_time'] = float(execution_time)
    
    # Calculate capacity used
    capacity_used = (message_size_bits / img_capacity_bits) * 100
    metrics['capacity_used'] = min(100.0, capacity_used)  # Cap at 100%
    
    # Tambahkan info enkripsi dan kompresi
    metrics['encrypted'] = password is not None
    metrics['compressed'] = message.startswith("COMPRESSED:")
    
    logger.info(f"Encoding completed in {execution_time:.2f} seconds")
    logger.info(f"Encoding metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    
    return stego_img, metrics

def secure_lsb_encode(image, message):
    """
    Implementasi steganografi LSB (Least Significant Bit) yang ditingkatkan keamanannya
    
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
    
    # Generate pseudo-random sequence for bit insertion using a simple seed
    seed = (image.shape[0] * image.shape[1]) % 1000000
    np.random.seed(seed)
    
    # Buat indeks acak untuk penyisipan bit (lebih sulit dideteksi)
    # Skip 32 bit pertama untuk header
    max_index = image.shape[0] * image.shape[1] * 3
    indices = np.random.permutation(max_index - 32) + 32  # Skip header bits
    indices = indices[:len(message_bits)]  # Ambil hanya yang diperlukan
    
    # Sembunyikan panjang pesan di 32 bit pertama (tetap sekuensial untuk kemudahan decode)
    msg_len = len(message_bits)
    msg_len_bits = format(msg_len, '032b')
    
    try:
        # Step 1: Sembunyikan panjang pesan (32 bit pertama)
        for i in range(32):
            # Hitung posisi pixel dan channel
            pixel_index = i // 3
            row = pixel_index // image.shape[1]
            col = pixel_index % image.shape[1]
            channel = i % 3
            
            # Validasi bounds
            if row >= image.shape[0] or col >= image.shape[1]:
                raise ValueError(f"Index out of bounds: {row},{col} for shape {image.shape[:2]}")
            
            # Set bit paling tidak signifikan
            stego[row, col, channel] = (stego[row, col, channel] & 0xFE) | int(msg_len_bits[i])
        
        # Step 2: Sembunyikan pesan menggunakan indeks pseudo-random
        for i in range(len(message_bits)):
            # Ambil indeks dari sequence acak
            idx = indices[i]
            
            # Hitung posisi pixel dan channel
            pixel_index = idx // 3
            row = pixel_index // image.shape[1]
            col = pixel_index % image.shape[1]
            channel = idx % 3
            
            # Validasi bounds
            if row >= image.shape[0] or col >= image.shape[1]:
                # Jika melebihi bounds, hentikan encoding dan truncate pesan
                logger.warning(f"Message truncated at {i}/{len(message_bits)} bits due to image size")
                break
            
            # Set bit paling tidak signifikan
            stego[row, col, channel] = (stego[row, col, channel] & 0xFE) | int(message_bits[i])
        
        # Step 3: Tambahkan noise ke beberapa pixel acak untuk anti-steganalysis
        # (tidak mengubah bit yang menyimpan pesan)
        if stego.shape[0] * stego.shape[1] > 10000:  # Hanya untuk gambar cukup besar
            noise_pixels = int(stego.shape[0] * stego.shape[1] * 0.01)  # 1% dari total pixel
            for _ in range(noise_pixels):
                # Pilih pixel acak
                row = np.random.randint(0, stego.shape[0])
                col = np.random.randint(0, stego.shape[1])
                channel = np.random.randint(0, 3)
                
                # Jangan ubah bit yang berisi data
                skip = False
                idx = (row * image.shape[1] + col) * 3 + channel
                
                if idx < 32:  # Header
                    skip = True
                else:
                    for m_idx in indices:
                        if m_idx == idx:
                            skip = True
                            break
                
                if not skip:
                    # Add minimal noise to LSB+1 (second least significant bit)
                    # Ini meningkatkan resistensi terhadap steganalysis statistik
                    if np.random.random() > 0.5:  # 50% chance
                        stego[row, col, channel] = stego[row, col, channel] ^ 0x02  # Flip second LSB
        
        logger.info(f"LSB secure encoding successful: embedded {len(message_bits)} bits using pseudo-random pattern")
        
    except IndexError as e:
        logger.error(f"IndexError during LSB encoding: {str(e)}", exc_info=True)
        raise ValueError(f"Gambar terlalu kecil untuk pesan ini. Maksimum kapasitas: {max_bits//8} karakter.")
    except Exception as e:
        logger.error(f"Error during LSB encoding: {str(e)}", exc_info=True)
        raise ValueError(f"Error saat encoding pesan: {str(e)}")
    
    return stego