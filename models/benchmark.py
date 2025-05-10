"""
benchmark.py
Modul untuk benchmarking berbagai metode steganografi
"""

import numpy as np
import cv2
import os
import time
import json
import random
import string
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from sklearn.model_selection import train_test_split
from models.utils import preprocess_image, calculate_metrics
from models.steganalysis import Steganalysis
from models.enhanced_gan_model import EnhancedGANSteganography
from models.enhanced_encoder import encode_message
from models.enhanced_decoder import decode_message

# Konfigurasi logging
logger = logging.getLogger(__name__)

class SteganographyBenchmark:
    """
    Class untuk benchmark berbagai metode steganografi dan
    membandingkan performa mereka
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi benchmark
        
        Args:
            config: Dictionary konfigurasi atau None untuk defaults
        """
        self.default_config = {
            'output_dir': 'results/benchmark',
            'dataset_paths': {
                'BOSS': 'datasets/BOSS',
                'BOWS2': 'datasets/BOWS2',
                'DIV2K': 'datasets/DIV2K',
                'custom': 'datasets/custom',
                'sample': 'datasets/sample'
            },
            'img_shape': (256, 256, 3),
            'max_images': 100,  # Maximum images per dataset
            'message_lengths': [32, 64, 128, 256, 512],  # Message lengths to test
            'repeat': 3,  # Number of repetitions for reliability
            'steganalysis': True  # Whether to run steganalysis
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize steganalysis if enabled
        self.steganalysis = None
        if self.config['steganalysis']:
            steg_config = {
                'results_dir': os.path.join(self.config['output_dir'], 'steganalysis'),
                'models_dir': os.path.join(self.config['output_dir'], 'steganalysis', 'models')
            }
            self.steganalysis = Steganalysis(steg_config)
        
    def load_datasets(self):
        """
        Load images from specified datasets
        
        Returns:
            dict: Dictionary of dataset name -> images array
        """
        datasets = {}
        
        for name, path in self.config['dataset_paths'].items():
            if os.path.exists(path):
                logger.info(f"Loading dataset: {name} from {path}")
                
                images = []
                valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                
                # List image files
                files = []
                for root, _, filenames in os.walk(path):
                    for filename in filenames:
                        if any(filename.lower().endswith(ext) for ext in valid_extensions):
                            files.append(os.path.join(root, filename))
                
                # Shuffle and limit number of files
                if len(files) > self.config['max_images']:
                    files = random.sample(files, self.config['max_images'])
                    
                # Load images
                for file_path in tqdm(files, desc=f"Loading {name}"):
                    try:
                        img = cv2.imread(file_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                            img = cv2.resize(img, (self.config['img_shape'][0], self.config['img_shape'][1]))
                            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
                            images.append(img)
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {str(e)}")
                        
                if images:
                    datasets[name] = np.array(images)
                    logger.info(f"  Loaded {len(images)} images")
                else:
                    logger.warning(f"  No valid images found in {path}")
            else:
                logger.warning(f"Dataset path not found: {path}")
                
        if not datasets:
            logger.error("No datasets loaded!")
            
        return datasets
    
    def register_steganography_methods(self):
        """
        Register available steganography methods
        
        Returns:
            dict: Dictionary of method name -> (encode_fn, decode_fn)
        """
        methods = {}
        
        # 1. LSB Basic method
        methods['LSB_Basic'] = (self.lsb_basic_encode, self.lsb_basic_decode)
        
        # 2. LSB Secure method (from enhanced_encoder/decoder)
        methods['LSB_Secure'] = (self.lsb_secure_encode, self.lsb_secure_decode)
        
        # 3. GAN-based method if model is available
        try:
            gan_model = EnhancedGANSteganography()
            model_loaded = gan_model.load_models('models/saved')
            if model_loaded:
                methods['GAN'] = (self.gan_encode, self.gan_decode)
                logger.info("GAN model loaded successfully")
            else:
                logger.warning("GAN model not available, skipping GAN method")
        except Exception as e:
            logger.error(f"Error loading GAN model: {str(e)}")
        
        # 4. Adaptive LSB
        methods['Adaptive_LSB'] = (self.adaptive_lsb_encode, self.adaptive_lsb_decode)
        
        # Log registered methods
        logger.info(f"Registered {len(methods)} steganography methods:")
        for method in methods:
            logger.info(f"  - {method}")
            
        return methods
    
    # Implementation of steganography methods
    def lsb_basic_encode(self, image, message):
        """
        Basic LSB encoding (sequential)
        
        Args:
            image: Cover image
            message: Message string
            
        Returns:
            tuple: (stego_image, metrics)
        """
        # Make a copy of the image
        stego = image.copy()
        
        # Convert message to binary
        binary_message = ''.join([format(ord(c), '08b') for c in message]) + '00000000'  # Add null terminator
        binary_message = [int(bit) for bit in binary_message]
        
        # Check if message fits in the image
        max_bits = image.shape[0] * image.shape[1] * 3
        if len(binary_message) > max_bits:
            logger.warning(f"Message too long, truncating ({len(binary_message)} > {max_bits})")
            binary_message = binary_message[:max_bits]
        
        # Embed message bits
        bit_idx = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):  # RGB channels
                    if bit_idx < len(binary_message):
                        # Replace LSB with message bit
                        stego[i, j, k] = (np.floor(stego[i, j, k] * 255) & 0xFE) / 255.0  # Clear LSB
                        stego[i, j, k] += binary_message[bit_idx] / 255.0  # Set LSB
                        bit_idx += 1
                    else:
                        break
        
        # Calculate metrics
        metrics = calculate_metrics(image, stego)
        metrics['capacity_used'] = (len(binary_message) / max_bits) * 100
        
        return stego, metrics
    
    def lsb_basic_decode(self, stego_image):
        """
        Basic LSB decoding (sequential)
        
        Args:
            stego_image: Stego image
            
        Returns:
            tuple: (message, metadata)
        """
        # Extract bits from image
        bits = []
        for i in range(stego_image.shape[0]):
            for j in range(stego_image.shape[1]):
                for k in range(3):  # RGB channels
                    # Get LSB
                    bit = int(round(stego_image[i, j, k] * 255) & 1)
                    bits.append(bit)
        
        # Convert bits to characters
        message = ""
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = bits[i:i+8]
                # Check for null terminator
                if byte == [0, 0, 0, 0, 0, 0, 0, 0]:
                    break
                    
                # Convert byte to character
                char_code = int(''.join(str(b) for b in byte), 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    message += chr(char_code)
                else:
                    # Skip non-printable characters
                    continue
        
        metadata = {
            'method': 'lsb_basic',
            'execution_time': 0.0,
            'is_encrypted': False,
            'is_compressed': False
        }
        
        return message, metadata
    
    def lsb_secure_encode(self, image, message):
        """
        Secure LSB encoding (wrapper for enhanced_encoder.encode_message)
        
        Args:
            image: Cover image
            message: Message string
            
        Returns:
            tuple: (stego_image, metrics)
        """
        start_time = time.time()
        stego_img, metrics = encode_message(image, message)
        metrics['execution_time'] = time.time() - start_time
        return stego_img, metrics
    
    def lsb_secure_decode(self, stego_image):
        """
        Secure LSB decoding (wrapper for enhanced_decoder.decode_message)
        
        Args:
            stego_image: Stego image
            
        Returns:
            tuple: (message, metadata)
        """
        start_time = time.time()
        message, metadata = decode_message(stego_image)
        metadata['execution_time'] = time.time() - start_time
        return message, metadata
    
    def gan_encode(self, image, message):
        """
        GAN-based encoding
        
        Args:
            image: Cover image
            message: Message string
            
        Returns:
            tuple: (stego_image, metrics)
        """
        start_time = time.time()
        
        # Initialize GAN model
        gan_model = EnhancedGANSteganography()
        gan_model.load_models('models/saved')
        
        # Encode message
        stego_img = gan_model.encode_message(image, message)
        
        # Calculate metrics
        metrics = calculate_metrics(image, stego_img)
        metrics['execution_time'] = time.time() - start_time
        metrics['method'] = 'gan'
        
        return stego_img, metrics
    
    def gan_decode(self, stego_image):
        """
        GAN-based decoding
        
        Args:
            stego_image: Stego image
            
        Returns:
            tuple: (message, metadata)
        """
        start_time = time.time()
        
        # Initialize GAN model
        gan_model = EnhancedGANSteganography()
        gan_model.load_models('models/saved')
        
        # Decode message bits
        message_bits = gan_model.decode_message(stego_image)
        
        # Convert bits to text
        # Group bits into bytes
        bytes_list = []
        for i in range(0, len(message_bits), 8):
            if i + 8 <= len(message_bits):
                byte = message_bits[i:i+8]
                if np.array_equal(byte, np.zeros(8)):  # Null terminator
                    break
                bytes_list.append(byte)
        
        # Convert bytes to characters
        message = ""
        for byte in bytes_list:
            char_code = int(''.join(str(int(bit)) for bit in byte), 2)
            if 32 <= char_code <= 126:  # Printable ASCII
                message += chr(char_code)
        
        metadata = {
            'method': 'gan',
            'execution_time': time.time() - start_time,
            'is_encrypted': False,
            'is_compressed': False
        }
        
        return message, metadata
    
    def adaptive_lsb_encode(self, image, message):
        """
        Adaptive LSB encoding - hide more bits in textured areas
        
        Args:
            image: Cover image
            message: Message string
            
        Returns:
            tuple: (stego_image, metrics)
        """
        start_time = time.time()
        
        # Make a copy of the image
        stego = image.copy()
        
        # Convert message to binary
        binary_message = ''.join([format(ord(c), '08b') for c in message]) + '00000000'  # Add null terminator
        binary_message = [int(bit) for bit in binary_message]
        
        # Calculate texture map using edge detection
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude of gradient
        gradient_mag = np.sqrt(edge_x**2 + edge_y**2)
        
        # Normalize to [0,1]
        gradient_mag = gradient_mag / np.max(gradient_mag)
        
        # Apply Gaussian blur for smoother transitions
        texture_map = cv2.GaussianBlur(gradient_mag, (5, 5), 0)
        
        # Rank pixels by texture value (high to low)
        flat_texture = texture_map.flatten()
        pixel_indices = np.argsort(-flat_texture)
        
        # Check if message fits (worst case)
        max_bits = image.shape[0] * image.shape[1] * 3
        if len(binary_message) > max_bits:
            logger.warning(f"Message too long, truncating ({len(binary_message)} > {max_bits})")
            binary_message = binary_message[:max_bits]
        
        # Flatten image for easier access
        flat_stego = stego.reshape(-1, 3)
        
        # Embed message bits in high-texture areas first
        for i, bit_idx in enumerate(pixel_indices):
            if i >= len(binary_message):
                break
                
            # Calculate pixel and channel
            pixel_idx = bit_idx // 3
            channel = bit_idx % 3
            
            # Replace LSB with message bit
            flat_stego[pixel_idx, channel] = (np.floor(flat_stego[pixel_idx, channel] * 255) & 0xFE) / 255.0  # Clear LSB
            flat_stego[pixel_idx, channel] += binary_message[i] / 255.0  # Set LSB
        
        # Reshape back to image
        stego = flat_stego.reshape(image.shape)
        
        # Store embedding map in alpha channel (for testing only)
        embedding_map = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, bit_idx in enumerate(pixel_indices):
            if i >= len(binary_message):
                break
            pixel_idx = bit_idx // 3
            row = pixel_idx // image.shape[1]
            col = pixel_idx % image.shape[1]
            embedding_map[row, col] = 255
        
        # Calculate metrics
        metrics = calculate_metrics(image, stego)
        metrics['execution_time'] = time.time() - start_time
        metrics['capacity_used'] = (len(binary_message) / max_bits) * 100
        metrics['method'] = 'adaptive_lsb'
        
        return stego, metrics
    
    def adaptive_lsb_decode(self, stego_image):
        """
        Adaptive LSB decoding - extract using same texture ranking
        
        Args:
            stego_image: Stego image
            
        Returns:
            tuple: (message, metadata)
        """
        start_time = time.time()
        
        # Calculate texture map using edge detection (must match encode)
        gray = cv2.cvtColor((stego_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude of gradient
        gradient_mag = np.sqrt(edge_x**2 + edge_y**2)
        
        # Normalize to [0,1]
        gradient_mag = gradient_mag / np.max(gradient_mag)
        
        # Apply Gaussian blur for smoother transitions
        texture_map = cv2.GaussianBlur(gradient_mag, (5, 5), 0)
        
        # Rank pixels by texture value (high to low)
        flat_texture = texture_map.flatten()
        pixel_indices = np.argsort(-flat_texture)
        
        # Flatten image for easier access
        flat_stego = stego_image.reshape(-1, 3)
        
        # Extract bits
        max_bits = stego_image.shape[0] * stego_image.shape[1] * 3
        bits = []
        
        # Read bits in same order as embedding
        for bit_idx in pixel_indices:
            if len(bits) >= max_bits:
                break
                
            # Calculate pixel and channel
            pixel_idx = bit_idx // 3
            channel = bit_idx % 3
            
            # Extract LSB
            bit = int(round(flat_stego[pixel_idx, channel] * 255) & 1)
            bits.append(bit)
            
            # Check for null terminator every 8 bits
            if len(bits) % 8 == 0:
                if len(bits) >= 8 and bits[-8:] == [0, 0, 0, 0, 0, 0, 0, 0]:
                    bits = bits[:-8]  # Remove null terminator
                    break
        
        # Convert bits to characters
        message = ""
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = bits[i:i+8]
                
                # Convert byte to character
                char_code = int(''.join(str(b) for b in byte), 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    message += chr(char_code)
        
        metadata = {
            'method': 'adaptive_lsb',
            'execution_time': time.time() - start_time,
            'is_encrypted': False,
            'is_compressed': False
        }
        
        return message, metadata
    
    def generate_random_message(self, length):
        """
        Generate random text message of specified length
        
        Args:
            length: Length of message in characters
            
        Returns:
            str: Random message
        """
        return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation + ' ') 
                      for _ in range(length))
    
    def benchmark_methods(self, dataset, methods=None, message_lengths=None, repeat=None):
        """
        Benchmark steganography methods
        
        Args:
            dataset: Array of cover images
            methods: Dictionary {name: (encode_fn, decode_fn)} or None for all registered
            message_lengths: List of message lengths to test or None for default
            repeat: Number of repetitions for reliability or None for default
            
        Returns:
            pd.DataFrame: Benchmark results
        """
        if methods is None:
            methods = self.register_steganography_methods()
            
        if message_lengths is None:
            message_lengths = self.config['message_lengths']
            
        if repeat is None:
            repeat = self.config['repeat']
        
        # Initialize results
        results = []
        
        # Sample subset of dataset for efficiency
        if len(dataset) > 50:
            sample_indices = np.random.choice(len(dataset), 50, replace=False)
            test_images = dataset[sample_indices]
        else:
            test_images = dataset
            
        logger.info(f"Benchmarking on {len(test_images)} images")
        
        # Progress tracking
        total_iterations = len(methods) * len(message_lengths) * repeat
        pbar = tqdm(total=total_iterations, desc="Benchmarking")
        
        # Benchmark each method
        for method_name, (encode_fn, decode_fn) in methods.items():
            logger.info(f"Benchmarking method: {method_name}")
            
            # Save cover/stego pairs for steganalysis
            stego_images_for_analysis = {}
            
            for msg_len in message_lengths:
                logger.info(f"  Testing message length: {msg_len}")
                
                for r in range(repeat):
                    # Generate random message
                    message = self.generate_random_message(msg_len)
                    
                    # Initialize metrics
                    metrics_sum = {
                        'psnr': 0.0,
                        'ssim': 0.0,
                        'mse': 0.0,
                        'hist_similarity': 0.0,
                        'encoding_time': 0.0,
                        'decoding_time': 0.0,
                        'success_rate': 0.0,
                        'bit_accuracy': 0.0,
                        'capacity_used': 0.0
                    }
                    
                    # Test on each image
                    current_stego_images = []
                    
                    for i, img in enumerate(test_images):
                        try:
                            # Encode message
                            start_time = time.time()
                            stego_img, encode_metrics = encode_fn(img.copy(), message)
                            encoding_time = time.time() - start_time
                            
                            # Save stego image for analysis
                            current_stego_images.append(stego_img)
                            
                            # Decode message
                            start_time = time.time()
                            decoded_msg, decode_metadata = decode_fn(stego_img)
                            decoding_time = time.time() - start_time
                            
                            # Check success
                            success = decoded_msg == message
                            
                            # Calculate bit accuracy if not perfect
                            if not success:
                                common_length = min(len(message), len(decoded_msg))
                                correct_chars = sum(a == b for a, b in zip(message[:common_length], decoded_msg[:common_length]))
                                bit_acc = correct_chars / len(message) if len(message) > 0 else 0
                            else:
                                bit_acc = 1.0
                                
                            # Update metrics
                            for key in metrics_sum:
                                if key in encode_metrics:
                                    metrics_sum[key] += encode_metrics[key]
                                    
                            metrics_sum['encoding_time'] += encoding_time
                            metrics_sum['decoding_time'] += decoding_time
                            metrics_sum['success_rate'] += float(success)
                            metrics_sum['bit_accuracy'] += bit_acc
                            
                        except Exception as e:
                            logger.error(f"Error with {method_name} on image {i}: {str(e)}")
                            continue
                    
                    # Calculate averages
                    n_images = len(test_images)
                    avg_metrics = {k: v/n_images for k, v in metrics_sum.items()}
                    
                    # Add to results
                    results.append({
                        'method': method_name,
                        'message_length': msg_len,
                        'repeat': r,
                        **avg_metrics
                    })
                    
                    # Save stego images for steganalysis
                    if len(current_stego_images) > 0:
                        # Only save on first repeat to save space
                        if r == 0:
                            stego_images_for_analysis[msg_len] = np.array(current_stego_images)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Log progress
                    logger.info(f"    Repeat {r+1}/{repeat} - "
                                f"PSNR: {avg_metrics['psnr']:.2f}, "
                                f"Success: {avg_metrics['success_rate']*100:.1f}%, "
                                f"Time: {avg_metrics['encoding_time']*1000:.1f}ms/img")
            
            # Run steganalysis if enabled
            if self.steganalysis and stego_images_for_analysis:
                logger.info(f"Running steganalysis for {method_name}")
                
                # For each message length
                for msg_len, stego_images in stego_images_for_analysis.items():
                    if len(stego_images) > 0:
                        # Get corresponding cover images
                        cover_imgs = test_images[:len(stego_images)]
                        
                        # Run evaluation
                        method_name_with_len = f"{method_name}_{msg_len}chars"
                        steg_results = self.steganalysis.evaluate_steganography_method(
                            cover_imgs, stego_images, method_name_with_len)
                        
                        # Add to results
                        for result in results:
                            if result['method'] == method_name and result['message_length'] == msg_len:
                                # Add steganalysis metrics
                                if 'chi_square_test' in steg_results:
                                    result['chi_sq_pvalue'] = steg_results['chi_square_test']['p_value']
                                if 'rs_analysis' in steg_results:
                                    result['rs_detection'] = steg_results['rs_analysis']['mean_stego']
                                if 'cnn_detector' in steg_results and 'roc_auc' in steg_results['cnn_detector']:
                                    result['detector_auc'] = steg_results['cnn_detector']['roc_auc']
        
        pbar.close()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config['output_dir'], f"benchmark_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Generate plots
        self.generate_benchmark_plots(results_df)
        
        return results_df
    
    def generate_benchmark_plots(self, results_df):
        """
        Generate plots from benchmark results
        
        Args:
            results_df: DataFrame with benchmark results
        """
        # 1. Image Quality vs Message Length
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            plt.plot(method_df.groupby('message_length')['psnr'].mean(), 
                     marker='o', label=method)
        plt.xlabel('Message Length (characters)')
        plt.ylabel('PSNR (dB)')
        plt.title('Image Quality vs Message Length')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            plt.plot(method_df.groupby('message_length')['ssim'].mean(), 
                     marker='o', label=method)
        plt.xlabel('Message Length (characters)')
        plt.ylabel('SSIM')
        plt.title('Structural Similarity vs Message Length')
        plt.grid(True)
        
        # 2. Success Rate vs Message Length
        plt.subplot(2, 2, 3)
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            plt.plot(method_df.groupby('message_length')['success_rate'].mean() * 100, 
                     marker='o', label=method)
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Success Rate (%)')
        plt.title('Decoding Success Rate vs Message Length')
        plt.ylim(0, 105)
        plt.grid(True)
        
        # 3. Processing Time vs Message Length
        plt.subplot(2, 2, 4)
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            plt.plot(method_df.groupby('message_length')['encoding_time'].mean() * 1000, 
                     marker='o', label=f"{method} (Encode)")
            plt.plot(method_df.groupby('message_length')['decoding_time'].mean() * 1000, 
                     marker='x', linestyle='--', label=f"{method} (Decode)")
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Time (milliseconds)')
        plt.title('Processing Time vs Message Length')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.config['output_dir'], f"benchmark_quality_plots_{timestamp}.png")
        plt.savefig(plot_path)
        logger.info(f"Quality plots saved to {plot_path}")
        
        plt.close()
        
        # If steganalysis metrics are available
        if any(col in results_df.columns for col in ['chi_sq_pvalue', 'rs_detection', 'detector_auc']):
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            for method in results_df['method'].unique():
                if 'chi_sq_pvalue' in results_df.columns:
                    method_df = results_df[results_df['method'] == method]
                    if not method_df['chi_sq_pvalue'].isna().all():
                        plt.plot(method_df.groupby('message_length')['chi_sq_pvalue'].mean(), 
                                marker='o', label=method)
            plt.xlabel('Message Length (characters)')
            plt.ylabel('Chi-Square p-value')
            plt.title('Statistical Detectability')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(1, 3, 2)
            for method in results_df['method'].unique():
                if 'rs_detection' in results_df.columns:
                    method_df = results_df[results_df['method'] == method]
                    if not method_df['rs_detection'].isna().all():
                        plt.plot(method_df.groupby('message_length')['rs_detection'].mean(), 
                                marker='o', label=method)
            plt.xlabel('Message Length (characters)')
            plt.ylabel('RS Detection Probability')
            plt.title('RS Analysis Detection')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            for method in results_df['method'].unique():
                if 'detector_auc' in results_df.columns:
                    method_df = results_df[results_df['method'] == method]
                    if not method_df['detector_auc'].isna().all():
                        plt.plot(method_df.groupby('message_length')['detector_auc'].mean(), 
                                marker='o', label=method)
            plt.xlabel('Message Length (characters)')
            plt.ylabel('Detector ROC AUC')
            plt.title('ML Detector Performance')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.config['output_dir'], f"benchmark_security_plots_{timestamp}.png")
            plt.savefig(plot_path)
            logger.info(f"Security plots saved to {plot_path}")
            
            plt.close()
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        
        # Compute averages across message lengths
        avg_metrics = results_df.groupby('method')[
            ['psnr', 'ssim', 'success_rate', 'encoding_time', 'bit_accuracy']
        ].mean()
        
        # Normalize metrics to 0-1 scale for radar chart
        norm_metrics = avg_metrics.copy()
        norm_metrics['psnr'] = (norm_metrics['psnr'] - norm_metrics['psnr'].min()) / \
                              (norm_metrics['psnr'].max() - norm_metrics['psnr'].min() + 1e-10)
        norm_metrics['ssim'] = (norm_metrics['ssim'] - norm_metrics['ssim'].min()) / \
                              (norm_metrics['ssim'].max() - norm_metrics['ssim'].min() + 1e-10)
        norm_metrics['success_rate'] = norm_metrics['success_rate']  # Already 0-1
        norm_metrics['bit_accuracy'] = norm_metrics['bit_accuracy']  # Already 0-1
        
        # Invert encoding time (lower is better)
        max_time = norm_metrics['encoding_time'].max() + 1e-10
        norm_metrics['speed'] = 1 - (norm_metrics['encoding_time'] / max_time)
        
        # Add security metric if available
        if 'detector_auc' in results_df.columns:
            avg_security = results_df.groupby('method')['detector_auc'].mean()
            if not avg_security.isna().all():
                # Invert AUC (lower is better, meaning less detectable)
                norm_metrics['security'] = 1 - ((avg_security - 0.5) * 2)  # Scale from 0.5-1 to 0-1
        else:
            # Default security score
            norm_metrics['security'] = 0.5
        
        # Radar chart categories
        categories = ['Image Quality (PSNR)', 'Structural Similarity', 
                     'Decoding Success', 'Processing Speed', 'Security']
        
        # Create radar chart
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        def radar_chart(ax, angles, values, method_name):
            # Close the plot (connect last point to first)
            values = np.append(values, values[0])
            angles = np.append(angles, angles[0])
            
            # Plot data
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name)
            ax.fill(angles, values, alpha=0.1)
            
            # Set limits and labels
            ax.set_ylim(0, 1.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Remove y-axis labels
            ax.set_yticklabels([])
            
            # Add gridlines
            ax.grid(True)
            
        # Setup radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # Plot each method
        for i, method in enumerate(norm_metrics.index):
            ax = plt.subplot(2, 2, i+1, polar=True)
            values = [norm_metrics.loc[method, 'psnr'], 
                     norm_metrics.loc[method, 'ssim'],
                     norm_metrics.loc[method, 'success_rate'],
                     norm_metrics.loc[method, 'speed'],
                     norm_metrics.loc[method, 'security']]
            radar_chart(ax, angles, values, method)
            ax.set_title(f"{method} Performance", size=11)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['output_dir'], f"benchmark_summary_{timestamp}.png")
        plt.savefig(plot_path)
        logger.info(f"Summary plot saved to {plot_path}")
        
        plt.close()
    
    def run_complete_benchmark(self):
        """
        Run complete benchmark on all available datasets
        
        Returns:
            pd.DataFrame: Combined benchmark results
        """
        # Load datasets
        datasets = self.load_datasets()
        if not datasets:
            logger.error("No datasets available for benchmarking")
            return None
            
        # Register methods
        methods = self.register_steganography_methods()
        if not methods:
            logger.error("No steganography methods available")
            return None
            
        # Combined results
        all_results = []
        
        # Run benchmark for each dataset
        for dataset_name, images in datasets.items():
            logger.info(f"Benchmarking on dataset: {dataset_name}")
            
            # Run benchmark
            results_df = self.benchmark_methods(images, methods)
            
            # Add dataset column
            results_df['dataset'] = dataset_name
            
            # Add to combined results
            all_results.append(results_df)
        
        # Combine results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save combined results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.config['output_dir'], f"benchmark_combined_{timestamp}.csv")
            combined_results.to_csv(results_path, index=False)
            logger.info(f"Combined results saved to {results_path}")
            
            return combined_results
        else:
            logger.error("No benchmark results generated")
            return None

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Steganography Benchmark Tool')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--output', type=str, default='results/benchmark', help='Output directory')
    parser.add_argument('--datasets', type=str, nargs='+', help='Specific datasets to use')
    parser.add_argument('--methods', type=str, nargs='+', help='Specific methods to benchmark')
    parser.add_argument('--message_lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512], 
                       help='Message lengths to test')
    parser.add_argument('--max_images', type=int, default=100, help='Maximum images per dataset')
    parser.add_argument('--no_steganalysis', action='store_true', help='Disable steganalysis evaluation')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from arguments
        config = {
            'output_dir': args.output,
            'max_images': args.max_images,
            'message_lengths': args.message_lengths,
            'steganalysis': not args.no_steganalysis
        }
        
        # Dataset paths
        if args.datasets:
            dataset_paths = {}
            for dataset in args.datasets:
                dataset_paths[dataset] = f'datasets/{dataset}'
            config['dataset_paths'] = dataset_paths
    
    # Create benchmark tool
    benchmark = SteganographyBenchmark(config)
    
    # Run benchmark
    benchmark.run_complete_benchmark()