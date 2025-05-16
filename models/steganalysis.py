"""
steganalysis.py
Modul untuk analisis deteksi steganografi (steganalysis)
"""

import numpy as np
import cv2
import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# Konfigurasi logging
logger = logging.getLogger(__name__)

class Steganalysis:
    """
    Class untuk analisis deteksi steganografi
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi steganalysis
        
        Args:
            config: Dictionary konfigurasi atau None untuk defaults
        """
        self.default_config = {
            'results_dir': 'results/steganalysis',
            'models_dir': 'models/saved/steganalysis',
            'img_shape': (256, 256, 3),
            'batch_size': 32
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Create directories
        os.makedirs(self.config['results_dir'], exist_ok=True)
        os.makedirs(self.config['models_dir'], exist_ok=True)
        
        # Initialize detector model
        self.detector_model = None
        
    def chi_square_analysis(self, image):
        """
        Analisis Chi-Square untuk deteksi steganografi
        
        Args:
            image: Input image (cover atau stego)
            
        Returns:
            dict: Hasil analisis chi-square
        """
        # Convert to uint8 if needed
        if image.dtype == np.float32 and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
            
        # Extract LSB planes
        lsb_planes = []
        for c in range(img.shape[2]):
            lsb = img[:, :, c] & 1
            lsb_planes.append(lsb.flatten())
            
        # Combine all LSB values
        all_lsb = np.concatenate(lsb_planes)
        
        # Count 0s and 1s
        zeros = np.sum(all_lsb == 0)
        ones = np.sum(all_lsb == 1)
        total = len(all_lsb)
        
        # Expected counts for natural image (should be approximately equal)
        expected = total / 2
        
        # Calculate chi-square statistic
        chi_square_val = ((zeros - expected)**2 + (ones - expected)**2) / expected
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(chi_square_val, df=1)
        
        # Interpretation
        if p_value < 0.05:
            interpretation = "Likely contains steganography"
        else:
            interpretation = "Likely clean image"
            
        return {
            'chi_square_value': float(chi_square_val),
            'p_value': float(p_value),
            'zeros_count': int(zeros),
            'ones_count': int(ones),
            'interpretation': interpretation
        }
    
    def rs_analysis(self, image):
        """
        RS (Regular-Singular) Analysis
        
        Args:
            image: Input image
            
        Returns:
            dict: Hasil analisis RS
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.dtype == np.float32 and image.max() <= 1.0:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Define discrimination function
        def f(g):
            return np.sum(np.abs(np.diff(g)))
        
        # Define masks
        mask_regular = np.array([1, 0, 1, 0])  # Regular mask
        mask_singular = np.array([-1, 0, -1, 0])  # Singular mask (negated)
        
        # Counters
        rm_count = 0  # Regular groups with positive mask
        sm_count = 0  # Singular groups with positive mask
        r_m_count = 0  # Regular groups with negative mask
        s_m_count = 0  # Singular groups with negative mask
        total_groups = 0
        
        # Process image in 2x2 blocks
        h, w = gray.shape
        for i in range(0, h-1, 2):
            for j in range(0, w-1, 2):
                # Extract 2x2 block
                block = gray[i:i+2, j:j+2].flatten()
                
                if len(block) < 4:
                    continue
                    
                # Calculate discrimination function for original block
                f_original = f(block)
                
                # Apply positive mask (flip LSB where mask is 1)
                block_positive = block.copy()
                for k in range(4):
                    if mask_regular[k] == 1:
                        block_positive[k] = block_positive[k] ^ 1
                f_positive = f(block_positive)
                
                # Apply negative mask (flip LSB where mask is -1)
                block_negative = block.copy()
                for k in range(4):
                    if mask_singular[k] == -1:
                        block_negative[k] = block_negative[k] ^ 1
                f_negative = f(block_negative)
                
                # Classify groups
                if f_positive > f_original:
                    rm_count += 1
                elif f_positive < f_original:
                    sm_count += 1
                    
                if f_negative > f_original:
                    r_m_count += 1
                elif f_negative < f_original:
                    s_m_count += 1
                    
                total_groups += 1
        
        # Calculate relative frequencies
        if total_groups > 0:
            RM = rm_count / total_groups
            SM = sm_count / total_groups
            R_M = r_m_count / total_groups
            S_M = s_m_count / total_groups
            
            # Calculate expected values for clean image
            # For clean image, we expect RM ≈ R_M and SM ≈ S_M
            
            # Calculate RS measure
            d = RM - SM
            d_neg = R_M - S_M
            
            # Estimate embedding rate
            if abs(d - d_neg) > 1e-10:
                p = d_neg / (d - d_neg)
                p = max(0, min(p, 1))  # Clamp to [0, 1]
            else:
                p = 0
                
            # Calculate RS statistic
            rs_statistic = 2 * (RM - R_M)
            
            # Interpretation
            if p > 0.1:
                interpretation = "Likely contains steganography"
                confidence = "High" if p > 0.3 else "Medium"
            else:
                interpretation = "Likely clean image"
                confidence = "High" if p < 0.05 else "Medium"
        else:
            RM = SM = R_M = S_M = 0
            p = 0
            rs_statistic = 0
            interpretation = "Unable to analyze"
            confidence = "Low"
            
        return {
            'embedding_rate': float(p),
            'rs_statistic': float(rs_statistic),
            'RM': float(RM),
            'SM': float(SM),
            'R_M': float(R_M),
            'S_M': float(S_M),
            'interpretation': interpretation,
            'confidence': confidence,
            'mean_stego': float(p)  # For compatibility
        }
    
    def build_cnn_detector(self, img_shape=None):
        """
        Build CNN-based steganography detector
        
        Args:
            img_shape: Input image shape
            
        Returns:
            tf.keras.Model: CNN detector model
        """
        if img_shape is None:
            img_shape = self.config['img_shape']
            
        # Input layer
        inputs = Input(shape=img_shape)
        
        # Feature extraction layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer (binary classification: clean vs stego)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='stego_detector')
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_cnn_detector(self, cover_images, stego_images, epochs=50, validation_split=0.2):
        """
        Train CNN detector on cover and stego images
        
        Args:
            cover_images: Array of cover images
            stego_images: Array of stego images
            epochs: Number of training epochs
            validation_split: Fraction for validation
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Prepare data
        X = np.concatenate([cover_images, stego_images], axis=0)
        y = np.concatenate([
            np.zeros(len(cover_images)),  # 0 for cover
            np.ones(len(stego_images))    # 1 for stego
        ])
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Build model if not exists
        if self.detector_model is None:
            self.detector_model = self.build_cnn_detector()
            
        # Train model
        history = self.detector_model.fit(
            X, y,
            batch_size=self.config['batch_size'],
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(self.config['models_dir'], 'cnn_detector.h5')
        self.detector_model.save(model_path)
        logger.info(f"Detector model saved to {model_path}")
        
        return history
    
    def evaluate_steganography_method(self, cover_images, stego_images, method_name="unknown"):
        """
        Evaluate steganography method using multiple detection techniques
        
        Args:
            cover_images: Array of cover images
            stego_images: Array of stego images  
            method_name: Name of steganography method
            
        Returns:
            dict: Evaluation results
        """
        results = {
            'method_name': method_name,
            'num_cover': len(cover_images),
            'num_stego': len(stego_images)
        }
        
        # Chi-square analysis
        chi_results = {'cover': [], 'stego': []}
        for img in cover_images[:100]:  # Limit for speed
            chi_results['cover'].append(self.chi_square_analysis(img))
        for img in stego_images[:100]:
            chi_results['stego'].append(self.chi_square_analysis(img))
            
        # Average chi-square results
        cover_chi_avg = np.mean([r['chi_square_value'] for r in chi_results['cover']])
        stego_chi_avg = np.mean([r['chi_square_value'] for r in chi_results['stego']])
        cover_p_avg = np.mean([r['p_value'] for r in chi_results['cover']])
        stego_p_avg = np.mean([r['p_value'] for r in chi_results['stego']])
        
        results['chi_square_test'] = {
            'cover_avg_chi': float(cover_chi_avg),
            'stego_avg_chi': float(stego_chi_avg),
            'cover_avg_p': float(cover_p_avg),
            'stego_avg_p': float(stego_p_avg),
            'p_value': float(stego_p_avg)  # For compatibility
        }
        
        # RS analysis
        rs_results = {'cover': [], 'stego': []}
        for img in cover_images[:100]:
            rs_results['cover'].append(self.rs_analysis(img))
        for img in stego_images[:100]:
            rs_results['stego'].append(self.rs_analysis(img))
            
        # Average RS results
        cover_rs_avg = np.mean([r['embedding_rate'] for r in rs_results['cover']])
        stego_rs_avg = np.mean([r['embedding_rate'] for r in rs_results['stego']])
        
        results['rs_analysis'] = {
            'cover_avg_rate': float(cover_rs_avg),
            'stego_avg_rate': float(stego_rs_avg),
            'mean_stego': float(stego_rs_avg)  # For compatibility
        }
        
        # CNN-based detection if model exists
        detector_path = os.path.join(self.config['models_dir'], 'cnn_detector.h5')
        if os.path.exists(detector_path) or self.detector_model is not None:
            if self.detector_model is None:
                self.detector_model = tf.keras.models.load_model(detector_path)
                
            # Evaluate on test set
            X_test = np.concatenate([cover_images[:200], stego_images[:200]], axis=0)
            y_test = np.concatenate([
                np.zeros(len(cover_images[:200])),
                np.ones(len(stego_images[:200]))
            ])
            
            # Get predictions
            y_pred = self.detector_model.predict(X_test, verbose=0)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            auc = roc_auc_score(y_test, y_pred)
            
            results['cnn_detector'] = {
                'accuracy': float(accuracy),
                'roc_auc': float(auc),
                'cover_detection_rate': float(np.mean(y_pred[:len(cover_images[:200])])),
                'stego_detection_rate': float(np.mean(y_pred[len(cover_images[:200]):]))
            }
        
        # Overall assessment
        detection_indicators = []
        
        # Chi-square indicator (lower p-value for stego = more detectable)
        if stego_p_avg < 0.05:
            detection_indicators.append(1)
        else:
            detection_indicators.append(0)
            
        # RS indicator (higher rate for stego = more detectable)
        if stego_rs_avg > 0.1:
            detection_indicators.append(1)
        else:
            detection_indicators.append(0)
            
        # CNN indicator if available
        if 'cnn_detector' in results:
            if results['cnn_detector']['roc_auc'] > 0.7:
                detection_indicators.append(1)
            else:
                detection_indicators.append(0)
                
        # Overall detectability score
        if detection_indicators:
            detectability = np.mean(detection_indicators)
        else:
            detectability = 0.5
            
        results['overall_assessment'] = {
            'detectability_score': float(detectability),
            'interpretation': 'Easily detectable' if detectability > 0.7 else 
                           'Moderately detectable' if detectability > 0.3 else 
                           'Difficult to detect'
        }
        
        # Save results
        import json
        results_path = os.path.join(self.config['results_dir'], f'{method_name}_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Steganalysis results saved to {results_path}")
        
        return results
    
    def visualize_analysis(self, cover_images, stego_images, results=None):
        """
        Visualize steganalysis results
        
        Args:
            cover_images: Array of cover images
            stego_images: Array of stego images
            results: Analysis results dictionary
            
        Returns:
            matplotlib.figure.Figure: Figure with visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Sample images comparison
        ax = axes[0, 0]
        ax.imshow(cover_images[0])
        ax.set_title('Sample Cover Image')
        ax.axis('off')
        
        ax = axes[0, 1]  
        ax.imshow(stego_images[0])
        ax.set_title('Sample Stego Image')
        ax.axis('off')
        
        # Difference (amplified)
        ax = axes[0, 2]
        if cover_images[0].shape == stego_images[0].shape:
            diff = np.abs(cover_images[0] - stego_images[0])
            if len(diff.shape) == 3:
                diff = np.mean(diff, axis=2)
            diff = np.clip(diff * 50, 0, 1)  # Amplify differences
            ax.imshow(diff, cmap='hot')
        ax.set_title('Difference (50x amplified)')
        ax.axis('off')
        
        # LSB histograms
        ax = axes[1, 0]
        
        # Extract LSB planes
        def get_lsb_hist(images, max_images=50):
            lsb_values = []
            for img in images[:max_images]:
                if img.dtype == np.float32 and img.max() <= 1.0:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img.astype(np.uint8)
                lsb = img_uint8 & 1
                lsb_values.extend(lsb.flatten())
            return np.array(lsb_values)
            
        cover_lsb = get_lsb_hist(cover_images)
        stego_lsb = get_lsb_hist(stego_images)
        
        ax.hist([cover_lsb, stego_lsb], bins=2, alpha=0.7, label=['Cover', 'Stego'], density=True)
        ax.set_xlabel('LSB Value')
        ax.set_ylabel('Frequency')
        ax.set_title('LSB Distribution')
        ax.legend()
        
        # Results visualization if provided
        if results:
            # Chi-square results
            ax = axes[1, 1]
            chi_data = results.get('chi_square_test', {})
            cover_p = chi_data.get('cover_avg_p', 0.5)
            stego_p = chi_data.get('stego_avg_p', 0.5)
            
            ax.bar(['Cover', 'Stego'], [cover_p, stego_p], alpha=0.7)
            ax.set_ylabel('p-value')
            ax.set_title('Chi-Square Test Results')
            ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05')
            ax.legend()
            
            # RS analysis results
            ax = axes[1, 2]
            rs_data = results.get('rs_analysis', {})
            cover_rate = rs_data.get('cover_avg_rate', 0)
            stego_rate = rs_data.get('stego_avg_rate', 0)
            
            ax.bar(['Cover', 'Stego'], [cover_rate, stego_rate], alpha=0.7)
            ax.set_ylabel('Estimated Embedding Rate')
            ax.set_title('RS Analysis Results')
        else:
            # Empty plots
            axes[1, 1].text(0.5, 0.5, 'No Results Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Chi-Square Results')
            
            axes[1, 2].text(0.5, 0.5, 'No Results Available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('RS Analysis Results')
        
        plt.tight_layout()
        return fig