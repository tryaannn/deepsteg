"""
crypto.py - Modul untuk enkripsi dan dekripsi pesan
Implementasi AES untuk mengamankan pesan sebelum disisipkan ke dalam gambar
"""

import os
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import logging

# Konfigurasi logging
logger = logging.getLogger(__name__)

class MessageEncryptor:
    """
    Kelas untuk enkripsi dan dekripsi pesan menggunakan AES
    dengan password-based key derivation
    """
    
    def __init__(self):
        self.backend = default_backend()
        self.iterations = 100000  # Jumlah iterasi untuk PBKDF2
        self.key_length = 32  # 256 bit key
        self.salt_length = 16  # 128 bit salt
        self.iv_length = 16  # 128 bit IV
        
    def derive_key(self, password, salt):
        """
        Membuat key derivation dari password dan salt menggunakan PBKDF2
        
        Args:
            password (str): Password yang digunakan
            salt (bytes): Salt untuk key derivation
            
        Returns:
            bytes: Derived key
        """
        if not password:
            raise ValueError("Password tidak boleh kosong")
            
        # Convert password to bytes if it's a string
        if isinstance(password, str):
            password = password.encode('utf-8')
            
        # Derive a key using PBKDF2
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password,
            salt,
            self.iterations,
            self.key_length
        )
        
        return key
    
    def encrypt(self, message, password):
        """
        Mengenkripsi pesan dengan password
        
        Args:
            message (str): Pesan yang akan dienkripsi
            password (str): Password untuk enkripsi
            
        Returns:
            str: Pesan terenkripsi dalam format base64 (salt + iv + ciphertext)
        """
        try:
            # Generate random salt and IV
            salt = os.urandom(self.salt_length)
            iv = os.urandom(self.iv_length)
            
            # Derive key from password and salt
            key = self.derive_key(password, salt)
            
            # Create an encryptor
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Pad the message to a multiple of block size
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            
            # Convert message to bytes if it's a string
            if isinstance(message, str):
                message = message.encode('utf-8')
                
            padded_data = padder.update(message) + padder.finalize()
            
            # Encrypt the padded message
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine salt + IV + ciphertext and encode as base64
            encrypted = salt + iv + ciphertext
            return base64.b64encode(encrypted).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}", exc_info=True)
            raise ValueError(f"Gagal mengenkripsi pesan: {str(e)}")
    
    def decrypt(self, encrypted_message, password):
        """
        Mendekripsi pesan terenkripsi dengan password
        
        Args:
            encrypted_message (str): Pesan terenkripsi dalam format base64
            password (str): Password untuk dekripsi
            
        Returns:
            str: Pesan asli yang sudah didekripsi
        """
        try:
            # Decode the base64 encrypted message
            if isinstance(encrypted_message, str):
                encrypted_message = base64.b64decode(encrypted_message.encode('utf-8'))
            elif isinstance(encrypted_message, bytes):
                encrypted_message = base64.b64decode(encrypted_message)
                
            # Extract salt, IV, and ciphertext
            salt = encrypted_message[:self.salt_length]
            iv = encrypted_message[self.salt_length:self.salt_length + self.iv_length]
            ciphertext = encrypted_message[self.salt_length + self.iv_length:]
            
            # Derive key from password and extracted salt
            key = self.derive_key(password, salt)
            
            # Create a decryptor
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt the ciphertext
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Unpad the decrypted data
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            
            # Return the decrypted message as a string
            return data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}", exc_info=True)
            raise ValueError(f"Gagal mendekripsi pesan. Password mungkin salah atau data rusak: {str(e)}")

def calculate_password_strength(password):
    """
    Menghitung dan mengevaluasi kekuatan password
    
    Args:
        password (str): Password yang akan dievaluasi
        
    Returns:
        dict: Hasil evaluasi {score, strength, feedback}
    """
    score = 0
    feedback = []
    
    # Check length
    if len(password) < 8:
        feedback.append("Password terlalu pendek (minimal 8 karakter)")
    elif len(password) >= 12:
        score += 2
    else:
        score += 1
        
    # Check for numbers
    if any(char.isdigit() for char in password):
        score += 1
    else:
        feedback.append("Tambahkan angka untuk password lebih kuat")
        
    # Check for uppercase
    if any(char.isupper() for char in password):
        score += 1
    else:
        feedback.append("Tambahkan huruf kapital untuk password lebih kuat")
        
    # Check for lowercase
    if any(char.islower() for char in password):
        score += 1
    else:
        feedback.append("Tambahkan huruf kecil untuk password lebih kuat")
        
    # Check for special characters
    special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?")
    if any(char in special_chars for char in password):
        score += 1
    else:
        feedback.append("Tambahkan karakter khusus untuk password lebih kuat")
        
    # Determine strength
    if score < 2:
        strength = "Sangat Lemah"
    elif score < 3:
        strength = "Lemah"
    elif score < 4:
        strength = "Sedang"
    elif score < 5:
        strength = "Kuat"
    else:
        strength = "Sangat Kuat"
        
    result = {
        "score": score,
        "strength": strength,
        "feedback": feedback
    }
    
    return result