"""
metrics.py
Modul untuk metrik evaluasi steganografi yang ditingkatkan
"""

import numpy as np
import cv2
import logging
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
import tensorflow as tf

# Konfigurasi logging
logger = logging.getLogger(__name__)

class SteganographyMetrics:
    """
    Class untuk mengevaluasi metrik kualitas dan keamanan steganografi
    """
    
    def __init__(self):
        """
        Inisialisasi evaluator metrik
        """
        # History untuk tracking performa
        self.history = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'hist_similarity': [],
            'uiqi': [],  # Universal Image Quality Index
            'execution_time': [],
            'bit_capacity': [],
            'detectability': []
        }
    
    def evaluate_image_quality(self, cover_image, stego_image):
        """
        Evaluasi kualitas gambar stego
        
        Args:
            cover_image: Gambar cover original
            stego_image: Gambar stego
            
        Returns:
            dict: Metrik kualitas gambar
        """
        start_time = time.time()
        
        # Check image format
        if cover_image.dtype != stego_image.dtype:
            logger.warning(f"Image type mismatch: {cover_image.dtype} vs {stego_image.dtype}")
            
        # Ensure float type for calculations
        if cover_image.dtype != np.float32:
            cover = cover_image.astype(np.float32)
            if cover.max() > 1.0:
                cover = cover / 255.0
        else:
            cover = cover_image
            
        if stego_image.dtype != np.float32:
            stego = stego_image.astype(np.float32)
            if stego.max() > 1.0:
                stego = stego / 255.0
        else:
            stego = stego_image
            
        # Check shape match
        if cover.shape != stego.shape:
            logger.error(f"Image shape mismatch: {cover.shape} vs {stego.shape}")
            return None
            
        # Initialize metrics
        metrics = {}
        
        # PSNR (Peak Signal-to-Noise Ratio)
        try:
            metrics['psnr'] = psnr(cover, stego, data_range=1.0)
        except Exception as e:
            logger.error(f"Error calculating PSNR: {str(e)}")
            metrics['psnr'] = 0.0
            
        # SSIM (Structural Similarity Index)
        try:
            # Try with channel_axis (newer scikit-image)
            try:
                metrics['ssim'] = ssim(cover, stego, channel_axis=2, data_range=1.0)
            except TypeError:
                # Fallback to multichannel for older versions
                metrics['ssim'] = ssim(cover, stego, multichannel=True, data_range=1.0)
        except Exception as e:
            logger.error(f"Error calculating SSIM: {str(e)}")
            metrics['ssim'] = 0.0
            
        # MSE (Mean Squared Error)
        try:
            metrics['mse'] = np.mean((cover - stego) ** 2)
        except Exception as e:
            logger.error(f"Error calculating MSE: {str(e)}")
            metrics['mse'] = 999.0
            
        # Histogram Similarity (Average Correlation)
        try:
            hist_sim = 0
            for i in range(3):  # RGB channels
                hist_cover = cv2.calcHist([cover * 255], [i], None, [256], [0, 256])
                hist_stego = cv2.calcHist([stego * 255], [i], None, [256], [0, 256])
                
                # Normalize histograms
                cv2.normalize(hist_cover, hist_cover, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist_stego, hist_stego, 0, 1, cv2.NORM_MINMAX)
                
                correl = cv2.compareHist(hist_cover, hist_stego, cv2.HISTCMP_CORREL)
                hist_sim += correl
                
            metrics['hist_similarity'] = hist_sim / 3.0
        except Exception as e:
            logger.error(f"Error calculating histogram similarity: {str(e)}")
            metrics['hist_similarity'] = 0.0
            
        # UIQI (Universal Image Quality Index)
        try:
            metrics['uiqi'] = self.calculate_uiqi(cover, stego)
        except Exception as e:
            logger.error(f"Error calculating UIQI: {str(e)}")
            metrics['uiqi'] = 0.0
            
        # Local Variance
        try:
            metrics['local_var_diff'] = self.local_variance_diff(cover, stego)
        except Exception as e:
            logger.error(f"Error calculating local variance: {str(e)}")
            metrics['local_var_diff'] = 1.0
            
        # Spectral Distortion (using Discrete Fourier Transform)
        try:
            metrics['spectral_distortion'] = self.spectral_distortion(cover, stego)
        except Exception as e:
            logger.error(f"Error calculating spectral distortion: {str(e)}")
            metrics['spectral_distortion'] = 1.0
            
        # Execution time
        metrics['execution_time'] = time.time() - start_time
        
        # Update history
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])
                
        return metrics
    
    def evaluate_security(self, cover_image, stego_image, detector_model=None):
        """
        Evaluasi keamanan steganografi
        
        Args:
            cover_image: Gambar cover original
            stego_image: Gambar stego
            detector_model: Model detector steganalysis (optional)
            
        Returns:
            dict: Metrik keamanan
        """
        start_time = time.time()
        
        # Ensure float type
        if cover_image.dtype != np.float32:
            cover = cover_image.astype(np.float32)
            if cover.max() > 1.0:
                cover = cover / 255.0
        else:
            cover = cover_image
            
        if stego_image.dtype != np.float32:
            stego = stego_image.astype(np.float32)
            if stego.max() > 1.0:
                stego = stego / 255.0
        else:
            stego = stego_image
            
        # Initialize metrics
        sec_metrics = {}
        
        # Chi-Square Attack
        try:
            chi_sq_val, chi_sq_p = self.chi_square_attack(stego)
            sec_metrics['chi_square_val'] = chi_sq_val
            sec_metrics['chi_square_p'] = chi_sq_p
        except Exception as e:
            logger.error(f"Error in Chi-Square attack: {str(e)}")
            sec_metrics['chi_square_val'] = 0.0
            sec_metrics['chi_square_p'] = 1.0
            
        # RS Analysis
        try:
            rs_val = self.rs_analysis(stego)
            sec_metrics['rs_measure'] = rs_val
        except Exception as e:
            logger.error(f"Error in RS analysis: {str(e)}")
            sec_metrics['rs_measure'] = 0.0
            
        # Sample Pairs Analysis
        try:
            sp_val = self.sample_pairs_analysis(stego)
            sec_metrics['sample_pairs'] = sp_val
        except Exception as e:
            logger.error(f"Error in Sample Pairs analysis: {str(e)}")
            sec_metrics['sample_pairs'] = 0.0
            
        # First-Order Statistics
        try:
            first_order = self.first_order_statistics(cover, stego)
            sec_metrics.update(first_order)
        except Exception as e:
            logger.error(f"Error in First-Order Statistics: {str(e)}")
            sec_metrics['first_order_diff'] = 1.0
            
        # CNN-based detection if model provided
        if detector_model is not None:
            try:
                # Prepare input
                model_input = tf.expand_dims(stego, 0)  # Add batch dimension
                
                # Run prediction
                detection_score = detector_model.predict(model_input)[0][0]
                sec_metrics['detection_score'] = float(detection_score)
            except Exception as e:
                logger.error(f"Error in CNN detection: {str(e)}")
                sec_metrics['detection_score'] = 0.5
        
        # Calculate overall detectability score (0=undetectable, 1=obvious)
        # Weight different measures
        detectability = 0.0
        weights = 0.0
        
        if 'chi_square_p' in sec_metrics:
            # Lower p-value = more detectable
            detectability += (1.0 - sec_metrics['chi_square_p']) * 0.3
            weights += 0.3
            
        if 'rs_measure' in sec_metrics:
            # Higher RS measure = more detectable
            detectability += sec_metrics['rs_measure'] * 0.3
            weights += 0.3
            
        if 'sample_pairs' in sec_metrics:
            # Higher SP value = more detectable
            detectability += sec_metrics['sample_pairs'] * 0.2
            weights += 0.2
            
        if 'detection_score' in sec_metrics:
            # Higher score = more detectable
            detectability += sec_metrics['detection_score'] * 0.5
            weights += 0.5
        
        # Normalize
        if weights > 0:
            detectability /= weights
            
        sec_metrics['detectability'] = detectability
        
        # Update history
        if 'detectability' in self.history:
            self.history['detectability'].append(detectability)
            
        # Add execution time
        sec_metrics['execution_time'] = time.time() - start_time
        
        return sec_metrics
    
    def evaluate_capacity(self, cover_image, message_bits=None):
        """
        Evaluasi kapasitas penyembunyian
        
        Args:
            cover_image: Gambar cover
            message_bits: Jumlah bit pesan (None=hitung teoretis)
            
        Returns:
            dict: Metrik kapasitas
        """
        # Ensure correct format
        if cover_image.dtype != np.float32:
            cover = cover_image.astype(np.float32)
            if cover.max() > 1.0:
                cover = cover / 255.0
        else:
            cover = cover_image
            
        # Calculate theoretical capacity (1 bit per channel)
        theoretical_capacity_bits = np.prod(cover.shape)
        theoretical_capacity_bytes = theoretical_capacity_bits // 8
        theoretical_capacity_chars = theoretical_capacity_bytes - 1  # -1 for null terminator
        
        # Calculate effective capacity based on image characteristics
        # More textured areas can hide more information
        texture_weight = self.estimate_texture_capacity(cover)
        effective_capacity_bits = int(theoretical_capacity_bits * texture_weight)
        effective_capacity_bytes = effective_capacity_bits // 8
        effective_capacity_chars = effective_capacity_bytes - 1
        
        # Calculate usage if message provided
        if message_bits is not None:
            usage_percentage = (message_bits / theoretical_capacity_bits) * 100
            effective_usage = (message_bits / effective_capacity_bits) * 100
        else:
            usage_percentage = 0
            effective_usage = 0
            
        # Capacity metrics
        capacity = {
            'theoretical_capacity_bits': int(theoretical_capacity_bits),
            'theoretical_capacity_bytes': int(theoretical_capacity_bytes),
            'theoretical_capacity_chars': int(theoretical_capacity_chars),
            'effective_capacity_bits': int(effective_capacity_bits),
            'effective_capacity_bytes': int(effective_capacity_bytes),
            'effective_capacity_chars': int(effective_capacity_chars),
            'texture_weight': float(texture_weight),
            'usage_percentage': float(usage_percentage),
            'effective_usage': float(effective_usage)
        }
        
        # Update history
        if 'bit_capacity' in self.history:
            self.history['bit_capacity'].append(effective_capacity_bits)
            
        return capacity
    
    def estimate_texture_capacity(self, image):
        """
        Estimate capacity based on image texture
        
        Args:
            image: Input image
            
        Returns:
            float: Texture weight (0-1) for capacity
        """
        # Convert to grayscale and uint8 for texture analysis
        if len(image.shape) == 3:
            if image.dtype == np.float32 and image.max() <= 1.0:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            if image.dtype == np.float32 and image.max() <= 1.0:
                gray = (image * 255).astype(np.uint8)
            else:
                gray = image.astype(np.uint8)
                
        # Calculate edge density using Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-1
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
            
        # Apply threshold to identify textured regions
        texture_mask = magnitude > 0.1
        texture_ratio = np.sum(texture_mask) / np.prod(gray.shape)
        
        # Weight between 0.5 and 0.9 (even flat images can store some data)
        texture_weight = 0.5 + (0.4 * texture_ratio)
        
        return texture_weight
    
    def calculate_uiqi(self, cover, stego, block_size=8):
        """
        Calculate Universal Image Quality Index
        
        Args:
            cover: Cover image
            stego: Stego image
            block_size: Block size for local calculation
            
        Returns:
            float: UIQI value
        """
        # Check dimensions
        if cover.shape != stego.shape:
            return 0.0
            
        # Process each channel
        uiqi_vals = []
        
        for c in range(cover.shape[2]):
            cover_channel = cover[:, :, c]
            stego_channel = stego[:, :, c]
            
            # Initialize
            sum_uiqi = 0
            count = 0
            
            # Process image block by block
            for i in range(0, cover.shape[0], block_size):
                for j in range(0, cover.shape[1], block_size):
                    # Extract blocks
                    block_cover = cover_channel[i:i+block_size, j:j+block_size]
                    block_stego = stego_channel[i:i+block_size, j:j+block_size]
                    
                    # Skip if too small
                    if block_cover.size < 4:
                        continue
                        
                    # Calculate statistics
                    mean_cover = np.mean(block_cover)
                    mean_stego = np.mean(block_stego)
                    var_cover = np.var(block_cover)
                    var_stego = np.var(block_stego)
                    cov = np.mean((block_cover - mean_cover) * (block_stego - mean_stego))
                    
                    # Calculate local UIQI
                    numerator = 2 * cov * 2 * mean_cover * mean_stego
                    denominator = (var_cover + var_stego) * (mean_cover**2 + mean_stego**2)
                    
                    if denominator > 0:
                        local_uiqi = numerator / denominator
                        sum_uiqi += local_uiqi
                        count += 1
            
            # Average UIQI for channel
            if count > 0:
                uiqi_vals.append(sum_uiqi / count)
                
        # Average across channels
        return np.mean(uiqi_vals) if uiqi_vals else 0.0
    
    def local_variance_diff(self, cover, stego, block_size=8):
        """
        Calculate local variance difference
        
        Args:
            cover: Cover image
            stego: Stego image
            block_size: Block size for local calculation
            
        Returns:
            float: Difference in local variance
        """
        # Check dimensions
        if cover.shape != stego.shape:
            return 1.0
            
        # Process each channel
        var_diffs = []
        
        for c in range(cover.shape[2]):
            cover_channel = cover[:, :, c]
            stego_channel = stego[:, :, c]
            
            # Calculate local variance
            cover_vars = []
            stego_vars = []
            
            for i in range(0, cover.shape[0], block_size):
                for j in range(0, cover.shape[1], block_size):
                    # Extract blocks
                    block_cover = cover_channel[i:i+block_size, j:j+block_size]
                    block_stego = stego_channel[i:i+block_size, j:j+block_size]
                    
                    # Skip if too small
                    if block_cover.size < 4:
                        continue
                        
                    # Calculate variance
                    cover_vars.append(np.var(block_cover))
                    stego_vars.append(np.var(block_stego))
            
            # Calculate difference
            if cover_vars and stego_vars:
                cover_vars = np.array(cover_vars)
                stego_vars = np.array(stego_vars)
                var_diff = np.mean(np.abs(cover_vars - stego_vars)) / (np.mean(cover_vars) + 1e-10)
                var_diffs.append(var_diff)
                
        # Average across channels
        return np.mean(var_diffs) if var_diffs else 1.0
    
    def spectral_distortion(self, cover, stego):
        """
        Calculate spectral distortion using DFT
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            float: Spectral distortion measure
        """
        # Check dimensions
        if cover.shape != stego.shape:
            return 1.0
            
        # Process each channel
        dist_vals = []
        
        for c in range(cover.shape[2]):
            cover_channel = cover[:, :, c]
            stego_channel = stego[:, :, c]
            
            # Apply DFT
            cover_dft = np.fft.fft2(cover_channel)
            stego_dft = np.fft.fft2(stego_channel)
            
            # Calculate magnitude
            cover_mag = np.abs(cover_dft)
            stego_mag = np.abs(stego_dft)
            
            # Normalize
            if np.max(cover_mag) > 0:
                cover_mag = cover_mag / np.max(cover_mag)
            if np.max(stego_mag) > 0:
                stego_mag = stego_mag / np.max(stego_mag)
                
            # Calculate difference
            mag_diff = np.mean(np.abs(cover_mag - stego_mag))
            dist_vals.append(mag_diff)
            
        # Average across channels
        return np.mean(dist_vals) if dist_vals else 1.0
    
    def chi_square_attack(self, image):
        """
        Perform Chi-Square attack on image
        
        Args:
            image: Input image
            
        Returns:
            tuple: (chi_square_value, p_value)
        """
        from scipy.stats import chi2
        
        # Convert to uint8 if needed
        if image.dtype == np.float32 and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
            
        # Extract LSB plane for all channels
        lsb_planes = []
        for c in range(img.shape[2]):
            lsb = img[:, :, c] & 1
            lsb_planes.append(lsb)
            
        # Combine LSBs
        lsb_values = np.stack(lsb_planes, axis=-1).flatten()
        
        # Count occurrences
        zeros = np.sum(lsb_values == 0)
        ones = np.sum(lsb_values == 1)
        
        # Expected counts for clean image (should be approximately equal)
        expected = np.array([zeros + ones]) / 2
        observed = np.array([zeros, ones])
        
        # Calculate chi-square
        chi_sq = np.sum((observed - expected)**2 / expected)
        
        # Calculate p-value (probability of image being clean)
        p_value = 1 - chi2.cdf(chi_sq, df=1)
        
        return chi_sq, p_value
    
    def rs_analysis(self, image):
        """
        Perform RS (Regular-Singular) Analysis
        
        Args:
            image: Input image
            
        Returns:
            float: RS measure (higher = more likely to contain hidden data)
        """
        # Convert to uint8
        if image.dtype == np.float32 and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
            
        # Convert to grayscale for simplicity
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Define discrimination function
        def f(g):
            return np.sum(np.abs(np.diff(g.reshape(-1))))
        
        # Define mask patterns
        mask_p = np.array([1, 0, 1, 0])  # Positive mask
        mask_n = np.array([-1, 0, -1, 0])  # Negative mask
        
        # Initialize counters
        rm_count = 0  # Regular, mask
        sm_count = 0  # Singular, mask
        r_m_count = 0  # Regular, inverse mask
        s_m_count = 0  # Singular, inverse mask
        total_groups = 0
        
        # Process image in 2x2 blocks
        h, w = gray.shape
        for i in range(0, h-1, 2):
            for j in range(0, w-1, 2):
                # Extract 2x2 block
                block = gray[i:i+2, j:j+2].flatten()
                
                if len(block) < 4:  # Skip incomplete blocks
                    continue
                    
                # Calculate discrimination functions
                f_orig = f(block)
                
                # Apply positive mask
                block_p = block.copy()
                for k in range(4):
                    if mask_p[k] == 1:
                        # Flip LSB
                        block_p[k] = block_p[k] ^ 1
                f_p = f(block_p)
                
                # Apply negative mask
                block_n = block.copy()
                for k in range(4):
                    if mask_n[k] == -1:
                        # Flip LSB
                        block_n[k] = block_n[k] ^ 1
                f_n = f(block_n)
                
                # Classify groups
                if f_p > f_orig:
                    rm_count += 1
                elif f_p < f_orig:
                    sm_count += 1
                    
                if f_n > f_orig:
                    r_m_count += 1
                elif f_n < f_orig:
                    s_m_count += 1
                    
                total_groups += 1
        
        # Calculate RS measures
        if total_groups > 0:
            rm = rm_count / total_groups
            sm = sm_count / total_groups
            r_m = r_m_count / total_groups
            s_m = s_m_count / total_groups
            
            # Calculate bias
            d = rm - sm
            d_m = r_m - s_m
            
            if d != d_m and d - d_m != 0:
                rs_measure = abs(d_m / (d - d_m))
                rs_measure = min(rs_measure, 1.0)  # Cap at 1.0
            else:
                rs_measure = 0.0
        else:
            rs_measure = 0.0
            
        return rs_measure
    
    def sample_pairs_analysis(self, image):
        """
        Perform Sample Pairs Analysis
        
        Args:
            image: Input image
            
        Returns:
            float: Embedding rate estimate
        """
        # Convert to uint8
        if image.dtype == np.float32 and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
            
        # Convert to grayscale for simplicity
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        h, w = gray.shape
        
        # Create pairs
        pairs = []
        
        # Horizontal pairs
        for i in range(h):
            for j in range(w-1):
                pairs.append((gray[i, j], gray[i, j+1]))
                
        # Vertical pairs
        for i in range(h-1):
            for j in range(w):
                pairs.append((gray[i, j], gray[i+1, j]))
                
        # Convert to array
        pairs = np.array(pairs)
        
        # Count different types of pairs
        # W: even-even, X: even-odd, Y: odd-even, Z: odd-odd
        W = np.sum((pairs[:, 0] % 2 == 0) & (pairs[:, 1] % 2 == 0))
        X = np.sum((pairs[:, 0] % 2 == 0) & (pairs[:, 1] % 2 == 1))
        Y = np.sum((pairs[:, 0] % 2 == 1) & (pairs[:, 1] % 2 == 0))
        Z = np.sum((pairs[:, 0] % 2 == 1) & (pairs[:, 1] % 2 == 1))
        
        # Calculate bias
        a = X - Y
        c = W + Z
        d = X + Y
        
        # Calculate estimated embedding rate
        if d > 0:
            p = min(abs(a / d) / 2, 1.0)
        else:
            p = 0.0
            
        return p
    
    def first_order_statistics(self, cover, stego):
        """
        Calculate first-order statistical differences
        
        Args:
            cover: Cover image
            stego: Stego image
            
        Returns:
            dict: First-order statistics metrics
        """
        # Convert to uint8
        if cover.dtype == np.float32 and cover.max() <= 1.0:
            cover_img = (cover * 255).astype(np.uint8)
        else:
            cover_img = cover.astype(np.uint8)
            
        if stego.dtype == np.float32 and stego.max() <= 1.0:
            stego_img = (stego * 255).astype(np.uint8)
        else:
            stego_img = stego.astype(np.uint8)
            
        # Calculate metrics for each channel
        metrics = {}
        
        for c in range(cover_img.shape[2]):
            cover_channel = cover_img[:, :, c]
            stego_channel = stego_img[:, :, c]
            
            # Histograms
            cover_hist = cv2.calcHist([cover_channel], [0], None, [256], [0, 256])
            stego_hist = cv2.calcHist([stego_channel], [0], None, [256], [0, 256])
            
            # Normalize
            cover_hist = cover_hist / np.sum(cover_hist)
            stego_hist = stego_hist / np.sum(stego_hist)
            
            # Kullback-Leibler divergence
            kl_div = entropy(cover_hist.flatten() + 1e-10, stego_hist.flatten() + 1e-10)
            
            # Chi-square between histograms
            chi_sq = np.sum((cover_hist - stego_hist)**2 / (cover_hist + 1e-10))
            
            # Update metrics
            metrics[f'kl_divergence_ch{c}'] = float(kl_div)
            metrics[f'chi_square_hist_ch{c}'] = float(chi_sq)
            
        # Average across channels
        metrics['kl_divergence_avg'] = np.mean([metrics[f'kl_divergence_ch{c}'] 
                                               for c in range(cover_img.shape[2])])
        metrics['chi_square_hist_avg'] = np.mean([metrics[f'chi_square_hist_ch{c}'] 
                                                for c in range(cover_img.shape[2])])
        
        # Overall first-order difference
        metrics['first_order_diff'] = (metrics['kl_divergence_avg'] + 
                                      metrics['chi_square_hist_avg']) / 2
        
        return metrics
    
    def visualize_metrics(self, cover_image, stego_image, metrics=None, security_metrics=None):
        """
        Visualize metrics and differences
        
        Args:
            cover_image: Cover image
            stego_image: Stego image
            metrics: Quality metrics (optional)
            security_metrics: Security metrics (optional)
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate metrics if not provided
        if metrics is None:
            metrics = self.evaluate_image_quality(cover_image, stego_image)
            
        if security_metrics is None:
            security_metrics = self.evaluate_security(cover_image, stego_image)
            
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Convert images to uint8 if needed
        if cover_image.dtype == np.float32 and cover_image.max() <= 1.0:
            cover = (cover_image * 255).astype(np.uint8)
        else:
            cover = cover_image.astype(np.uint8)
            
        if stego_image.dtype == np.float32 and stego_image.max() <= 1.0:
            stego = (stego_image * 255).astype(np.uint8)
        else:
            stego = stego_image.astype(np.uint8)
            
        # Image comparison
        ax1 = plt.subplot2grid((3, 4), (0, 0))
        ax1.imshow(cover)
        ax1.set_title('Cover Image')
        ax1.axis('off')
        
        ax2 = plt.subplot2grid((3, 4), (0, 1))
        ax2.imshow(stego)
        ax2.set_title('Stego Image')
        ax2.axis('off')
        
        # Difference image (amplified)
        ax3 = plt.subplot2grid((3, 4), (0, 2))
        diff = np.abs(cover.astype(np.float32) - stego.astype(np.float32))
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)  # Average across channels
        diff = np.clip(diff * 20, 0, 255).astype(np.uint8)  # Amplify
        ax3.imshow(diff, cmap='hot')
        ax3.set_title('Difference (20x amplified)')
        ax3.axis('off')
        
        # LSB planes
        ax4 = plt.subplot2grid((3, 4), (0, 3))
        stego_lsb = stego & 1
        if len(stego_lsb.shape) == 3:
            stego_lsb = np.mean(stego_lsb, axis=2) * 255
        ax4.imshow(stego_lsb, cmap='gray')
        ax4.set_title('Stego LSB Plane')
        ax4.axis('off')
        
        # Histograms
        ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
        for c, color in enumerate(['r', 'g', 'b']):
            if len(cover.shape) == 3:
                cover_hist = cv2.calcHist([cover], [c], None, [256], [0, 256])
                stego_hist = cv2.calcHist([stego], [c], None, [256], [0, 256])
                ax5.plot(cover_hist, color=color, linestyle='-', label=f'Cover {color.upper()}')
                ax5.plot(stego_hist, color=color, linestyle='--', label=f'Stego {color.upper()}')
        ax5.set_title('Histograms Comparison')
        ax5.set_xlabel('Pixel Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # DFT magnitude
        ax6 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        if len(cover.shape) == 3:
            cover_gray = cv2.cvtColor(cover, cv2.COLOR_RGB2GRAY)
            stego_gray = cv2.cvtColor(stego, cv2.COLOR_RGB2GRAY)
        else:
            cover_gray = cover
            stego_gray = stego
            
        # Calculate DFT and shift to center
        cover_dft = np.fft.fft2(cover_gray)
        cover_dft_shift = np.fft.fftshift(cover_dft)
        cover_mag = np.log(np.abs(cover_dft_shift) + 1)
        
        stego_dft = np.fft.fft2(stego_gray)
        stego_dft_shift = np.fft.fftshift(stego_dft)
        stego_mag = np.log(np.abs(stego_dft_shift) + 1)
        
        # Normalize
        cover_mag = (cover_mag / np.max(cover_mag) * 255).astype(np.uint8)
        stego_mag = (stego_mag / np.max(stego_mag) * 255).astype(np.uint8)
        
        # Difference in frequency domain
        diff_mag = np.abs(cover_mag - stego_mag)
        diff_mag = (diff_mag / np.max(diff_mag + 1e-10) * 255).astype(np.uint8)
        
        ax6.imshow(diff_mag, cmap='hot')
        ax6.set_title('DFT Magnitude Difference')
        ax6.axis('off')
        
        # Quality metrics
        ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        quality_text = '\n'.join([
            f"PSNR: {metrics.get('psnr', 0):.2f} dB",
            f"SSIM: {metrics.get('ssim', 0):.4f}",
            f"MSE: {metrics.get('mse', 0):.5f}",
            f"Histogram Similarity: {metrics.get('hist_similarity', 0):.4f}",
            f"UIQI: {metrics.get('uiqi', 0):.4f}",
            f"Local Variance Diff: {metrics.get('local_var_diff', 0):.5f}",
            f"Spectral Distortion: {metrics.get('spectral_distortion', 0):.5f}"
        ])
        ax7.text(0.5, 0.5, quality_text, fontsize=12, ha='center', va='center')
        ax7.set_title('Quality Metrics')
        ax7.axis('off')
        
        # Security metrics
        ax8 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        security_text = '\n'.join([
            f"Chi-Square p-value: {security_metrics.get('chi_square_p', 0):.4f} ({security_metrics.get('chi_square_val', 0):.2f})",
            f"RS Measure: {security_metrics.get('rs_measure', 0):.4f}",
            f"Sample Pairs: {security_metrics.get('sample_pairs', 0):.4f}",
            f"KL Divergence: {security_metrics.get('kl_divergence_avg', 0):.5f}",
            f"First Order Diff: {security_metrics.get('first_order_diff', 0):.5f}",
            f"Detectability Score: {security_metrics.get('detectability', 0):.4f}",
            f"Detection Probability: {security_metrics.get('detection_score', 0):.2f}" if 'detection_score' in security_metrics else ""
        ])
        ax8.text(0.5, 0.5, security_text, fontsize=12, ha='center', va='center')
        ax8.set_title('Security Metrics')
        ax8.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_history(self):
        """
        Plot metrics history
        
        Returns:
            plt.Figure: Matplotlib figure with history plots
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quality metrics
        if self.history['psnr']:
            axs[0, 0].plot(self.history['psnr'], label='PSNR')
            axs[0, 0].set_title('PSNR History')
            axs[0, 0].set_xlabel('Sample')
            axs[0, 0].set_ylabel('PSNR (dB)')
            axs[0, 0].grid(True)
            
        if self.history['ssim']:
            axs[0, 1].plot(self.history['ssim'], label='SSIM')
            axs[0, 1].set_title('SSIM History')
            axs[0, 1].set_xlabel('Sample')
            axs[0, 1].set_ylabel('SSIM')
            axs[0, 1].grid(True)
            axs[0, 1].set_ylim(0, 1)
            
        # Security metrics
        if self.history['detectability']:
            axs[1, 0].plot(self.history['detectability'], label='Detectability', color='r')
            axs[1, 0].set_title('Security History')
            axs[1, 0].set_xlabel('Sample')
            axs[1, 0].set_ylabel('Detectability')
            axs[1, 0].grid(True)
            axs[1, 0].set_ylim(0, 1)
            
        # Capacity
        if self.history['bit_capacity']:
            axs[1, 1].plot(self.history['bit_capacity'], label='Effective Capacity')
            axs[1, 1].set_title('Capacity History')
            axs[1, 1].set_xlabel('Sample')
            axs[1, 1].set_ylabel('Bits')
            axs[1, 1].grid(True)
            
        plt.tight_layout()
        return fig
        
    def reset_history(self):
        """
        Reset metrics history
        """
        for key in self.history:
            self.history[key] = []
            
        logger.info("Metrics history reset")

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Steganography Metrics Tool')
    parser.add_argument('--cover', type=str, required=True, help='Cover image path')
    parser.add_argument('--stego', type=str, required=True, help='Stego image path')
    parser.add_argument('--output', type=str, default='metrics_report.png', help='Output visualization path')
    parser.add_argument('--full', action='store_true', help='Calculate all metrics (slower)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load images
    cover_img = cv2.imread(args.cover)
    stego_img = cv2.imread(args.stego)
    
    if cover_img is None:
        logger.error(f"Failed to load cover image: {args.cover}")
        exit(1)
        
    if stego_img is None:
        logger.error(f"Failed to load stego image: {args.stego}")
        exit(1)
        
    # Convert to RGB
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)
    stego_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)
    
    # Create metrics evaluator
    metrics_eval = SteganographyMetrics()
    
    # Calculate metrics
    quality_metrics = metrics_eval.evaluate_image_quality(cover_img, stego_img)
    logger.info("Quality metrics calculated")
    
    security_metrics = metrics_eval.evaluate_security(cover_img, stego_img)
    logger.info("Security metrics calculated")
    
    capacity_metrics = metrics_eval.evaluate_capacity(cover_img)
    logger.info("Capacity metrics calculated")
    
    # Print summary
    print("\nQuality Metrics:")
    print(f"  PSNR: {quality_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {quality_metrics['ssim']:.4f}")
    print(f"  MSE: {quality_metrics['mse']:.6f}")
    print(f"  Histogram Similarity: {quality_metrics['hist_similarity']:.4f}")
    
    print("\nSecurity Metrics:")
    print(f"  Chi-Square p-value: {security_metrics['chi_square_p']:.4f}")
    print(f"  RS Measure: {security_metrics['rs_measure']:.4f}")
    print(f"  Sample Pairs: {security_metrics['sample_pairs']:.4f}")
    print(f"  Detectability: {security_metrics['detectability']:.4f}")
    
    print("\nCapacity Metrics:")
    print(f"  Theoretical Capacity: {capacity_metrics['theoretical_capacity_chars']} characters")
    print(f"  Effective Capacity: {capacity_metrics['effective_capacity_chars']} characters")
    print(f"  Texture Weight: {capacity_metrics['texture_weight']:.2f}")
    
    # Generate visualization
    fig = metrics_eval.visualize_metrics(cover_img, stego_img, quality_metrics, security_metrics)
    
    # Save figure
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {args.output}")
    
    plt.close(fig)