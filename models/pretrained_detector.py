"""
pretrained_detector.py
Modul untuk pre-trained steganalysis detector
"""

import os
import numpy as np
import tensorflow as tf
import logging
import urllib.request
import zipfile
import json
import time
import hashlib
from tqdm import tqdm
from pathlib import Path
import cv2

from models.steganalysis import Steganalysis

# Konfigurasi logging
logger = logging.getLogger(__name__)

class PretrainedDetector:
    """
    Manager untuk pre-trained steganalysis detector
    - Download detector model
    - Load model
    - Analyze images for steganography
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi detector manager
        
        Args:
            config: Dictionary konfigurasi atau None untuk defaults
        """
        self.default_config = {
            'detectors_dir': 'models/saved/steganalysis',
            'cache_dir': 'cache',
            'detector_catalog_url': 'https://example.com/deepsteg/detectors.json',  # Placeholder URL
            'available_detectors': {
                'cnn_basic': {
                    'description': 'Basic CNN Detector (256x256, LSB focused)',
                    'url': 'https://example.com/deepsteg/detectors/cnn_basic.zip',  # Placeholder URL
                    'md5': '0123456789abcdef0123456789abcdef',  # Placeholder MD5
                    'size_mb': 5.5,
                    'target_methods': ['LSB', 'LSB_matching', 'F5'],
                    'input_shape': [256, 256, 3],
                    'accuracy': 0.92
                },
                'deep_detector': {
                    'description': 'Deep CNN Detector (512x512, GAN steganography focused)',
                    'url': 'https://example.com/deepsteg/detectors/deep_detector.zip',  # Placeholder URL
                    'md5': 'fedcba9876543210fedcba9876543210',  # Placeholder MD5
                    'size_mb': 18.0,
                    'target_methods': ['GAN', 'deep_steg', 'WOW', 'S-UNIWARD'],
                    'input_shape': [512, 512, 3],
                    'accuracy': 0.88
                }
            },
            'default_detector': 'cnn_basic',
            'auto_download': False
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Create directories
        os.makedirs(self.config['detectors_dir'], exist_ok=True)
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Initialize steganalysis object
        self.steganalysis = Steganalysis()
        
        # Loaded detectors cache
        self.loaded_detectors = {}
        
        # Local detector catalog
        self.local_catalog = self._load_local_catalog()
        
        # Auto-download if configured
        if self.config['auto_download']:
            default_detector = self.config['default_detector']
            if default_detector not in self.local_catalog:
                logger.info(f"Auto-downloading default detector: {default_detector}")
                self.download_detector(default_detector)
    
    def _load_local_catalog(self):
        """
        Load catalog of locally available detectors
        
        Returns:
            dict: Local detector catalog
        """
        catalog = {}
        
        # Check detectors directory
        detectors_dir = Path(self.config['detectors_dir'])
        if not detectors_dir.exists():
            return catalog
            
        for detector_dir in detectors_dir.iterdir():
            if not detector_dir.is_dir():
                continue
                
            # Check for info file
            info_file = detector_dir / "detector_info.json"
            if not info_file.exists():
                continue
                
            # Load info
            try:
                with open(info_file, 'r') as f:
                    detector_info = json.load(f)
                    
                detector_name = detector_dir.name
                
                # Check for model file
                model_path = detector_dir / "detector_model.h5"
                
                if model_path.exists():
                    catalog[detector_name] = {
                        **detector_info,
                        'local_path': str(detector_dir),
                        'is_complete': True
                    }
                    logger.info(f"Found local detector: {detector_name}")
            except Exception as e:
                logger.error(f"Error loading detector info for {detector_dir.name}: {str(e)}")
                
        return catalog
    
    def download_detector(self, detector_name):
        """
        Download a pre-trained detector
        
        Args:
            detector_name: Name of the detector to download
            
        Returns:
            bool: Success or failure
        """
        if detector_name not in self.config['available_detectors']:
            logger.error(f"Detector {detector_name} not found in catalog")
            return False
            
        detector_info = self.config['available_detectors'][detector_name]
        
        # Check if URL is provided
        if 'url' not in detector_info:
            logger.error(f"Detector {detector_name} has no download URL")
            return False
            
        detector_url = detector_info['url']
        
        # Create detector directory
        detector_dir = os.path.join(self.config['detectors_dir'], detector_name)
        os.makedirs(detector_dir, exist_ok=True)
        
        # Download zip file
        zip_path = os.path.join(self.config['cache_dir'], f"{detector_name}.zip")
        
        try:
            logger.info(f"Downloading detector {detector_name} from {detector_url}")
            
            # Download with progress bar
            with urllib.request.urlopen(detector_url) as response:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 8192
                
                with open(zip_path, 'wb') as f, tqdm(
                        desc=f"Downloading {detector_name}",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))
            
            # Verify MD5 if provided
            if 'md5' in detector_info:
                logger.info(f"Verifying MD5 checksum: {detector_info['md5']}")
                
                with open(zip_path, 'rb') as f:
                    file_md5 = hashlib.md5(f.read()).hexdigest()
                
                if file_md5 != detector_info['md5']:
                    logger.error(f"MD5 checksum mismatch. Expected: {detector_info['md5']}, Got: {file_md5}")
                    os.remove(zip_path)
                    return False
            
            # Extract files
            logger.info(f"Extracting detector files to {detector_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(detector_dir)
            
            # Remove zip file
            os.remove(zip_path)
            
            # Create info file if not exists
            info_path = os.path.join(detector_dir, "detector_info.json")
            if not os.path.exists(info_path):
                with open(info_path, 'w') as f:
                    json.dump({
                        'name': detector_name,
                        'description': detector_info.get('description', ''),
                        'target_methods': detector_info.get('target_methods', []),
                        'input_shape': detector_info.get('input_shape', [256, 256, 3]),
                        'accuracy': detector_info.get('accuracy', 0.5),
                        'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f, indent=2)
            
            # Update local catalog
            self.local_catalog = self._load_local_catalog()
            
            logger.info(f"Detector {detector_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading detector {detector_name}: {str(e)}")
            
            # Clean up
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            return False
    
    def load_detector(self, detector_name=None):
        """
        Load a pre-trained detector
        
        Args:
            detector_name: Name of the detector to load (None=use default)
            
        Returns:
            tf.keras.Model or None: Loaded detector model
        """
        # Use default if not specified
        if detector_name is None:
            detector_name = self.config['default_detector']
            
        # Check if already loaded
        if detector_name in self.loaded_detectors:
            logger.info(f"Using cached detector {detector_name}")
            return self.loaded_detectors[detector_name]
            
        # Check if detector is available locally
        if detector_name not in self.local_catalog:
            logger.warning(f"Detector {detector_name} not found locally")
            
            # Try to download if available
            if detector_name in self.config['available_detectors']:
                logger.info(f"Attempting to download detector {detector_name}")
                if not self.download_detector(detector_name):
                    logger.error(f"Failed to download detector {detector_name}")
                    return None
            else:
                logger.error(f"Detector {detector_name} not found in catalog")
                return None
                
        # Get detector directory
        detector_dir = self.local_catalog[detector_name]['local_path']
        
        # Load detector model
        model_path = os.path.join(detector_dir, "detector_model.h5")
        
        try:
            # Load model
            detector_model = tf.keras.models.load_model(model_path)
            
            # Update last used timestamp
            info_path = os.path.join(detector_dir, "detector_info.json")
            with open(info_path, 'r') as f:
                info = json.load(f)
                
            info['last_used'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
                
            # Cache loaded model
            self.loaded_detectors[detector_name] = detector_model
            
            logger.info(f"Successfully loaded detector {detector_name}")
            return detector_model
            
        except Exception as e:
            logger.error(f"Error loading detector {detector_name}: {str(e)}")
            return None
    
    def detect_steganography(self, image, detector_name=None, options=None):
        """
        Detect steganography in an image
        
        Args:
            image: Input image
            detector_name: Name of the detector to use (None=use default)
            options: Dictionary of options
            
        Returns:
            dict: Detection results
        """
        # Default options
        default_options = {
            'normalize': True,  # Normalize image to [0,1]
            'use_statistical': True,  # Use statistical analysis in addition to ML
            'threshold': 0.5  # Detection threshold
        }
        
        if options is None:
            options = {}
            
        # Merge options
        options = {**default_options, **options}
        
        results = {
            'detector_used': detector_name if detector_name else self.config['default_detector'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'detection_score': 0,
            'is_stego': False,
            'confidence': 0,
            'estimated_payload': 0,
            'statistical_tests': {}
        }
        
        # Process image
        if options['normalize'] and (image.dtype != np.float32 or image.max() > 1.0):
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
            
        # 1. Statistical analysis
        if options['use_statistical']:
            # Chi-square analysis
            chi_results = self.steganalysis.chi_square_analysis(img)
            results['statistical_tests']['chi_square'] = chi_results
            
            # RS analysis
            rs_results = self.steganalysis.rs_analysis(img)
            results['statistical_tests']['rs_analysis'] = rs_results
            
            # Calculate statistical detection score
            stat_score = 0
            stat_weight = 0
            
            # Chi-square contribution
            if chi_results['p_value'] < 0.05:
                # Lower p-value indicates higher probability of steganography
                stat_score += (1 - chi_results['p_value']) * 0.4
                stat_weight += 0.4
                
            # RS analysis contribution
            if rs_results['embedding_rate'] > 0:
                stat_score += rs_results['embedding_rate'] * 0.6
                stat_weight += 0.6
                
            if stat_weight > 0:
                results['statistical_score'] = stat_score / stat_weight
            else:
                results['statistical_score'] = 0
                
            # Estimate payload from RS analysis
            results['estimated_payload'] = rs_results['embedding_rate']
        
        # 2. Machine learning detection if a detector is available
        ml_score = 0
        detector = self.load_detector(detector_name)
        
        if detector is not None:
            try:
                # Get input shape
                input_shape = detector.input_shape[1:3]  # [height, width]
                
                # Resize image if needed
                if img.shape[0] != input_shape[0] or img.shape[1] != input_shape[1]:
                    resized_img = cv2.resize(img, (input_shape[1], input_shape[0]))
                else:
                    resized_img = img
                
                # Add batch dimension
                input_img = np.expand_dims(resized_img, axis=0)
                
                # Run prediction
                ml_score = float(detector.predict(input_img, verbose=0)[0][0])
                results['ml_score'] = ml_score
            except Exception as e:
                logger.error(f"Error running ML detection: {str(e)}")
        
        # Combine scores
        if options['use_statistical'] and detector is not None:
            # Weighted average of statistical and ML scores
            results['detection_score'] = 0.4 * results.get('statistical_score', 0) + 0.6 * ml_score
        elif detector is not None:
            # Only ML score
            results['detection_score'] = ml_score
        elif options['use_statistical']:
            # Only statistical score
            results['detection_score'] = results.get('statistical_score', 0)
        
        # Determine if stego
        results['is_stego'] = results['detection_score'] > options['threshold']
        
        # Calculate confidence
        confidence = abs(results['detection_score'] - options['threshold']) * 2
        results['confidence'] = min(confidence, 1.0)
        
        # Interpretation
        if results['is_stego']:
            if results['confidence'] > 0.8:
                results['interpretation'] = "High confidence: Contains hidden data"
            elif results['confidence'] > 0.5:
                results['interpretation'] = "Medium confidence: Likely contains hidden data"
            else:
                results['interpretation'] = "Low confidence: May contain hidden data"
        else:
            if results['confidence'] > 0.8:
                results['interpretation'] = "High confidence: Clean image"
            elif results['confidence'] > 0.5:
                results['interpretation'] = "Medium confidence: Likely clean image"
            else:
                results['interpretation'] = "Low confidence: May be clean image"
        
        return results
    
    def list_available_detectors(self):
        """
        List all available detectors (local and remote)
        
        Returns:
            dict: Detector information
        """
        detectors = {}
        
        # Add remote detectors
        for name, info in self.config['available_detectors'].items():
            is_local = name in self.local_catalog
            
            detectors[name] = {
                'name': name,
                'description': info.get('description', 'No description'),
                'is_local': is_local,
                'size_mb': info.get('size_mb', 0),
                'target_methods': info.get('target_methods', []),
                'accuracy': info.get('accuracy', 0)
            }
            
            # Add local info if available
            if is_local:
                detectors[name].update({
                    'local_path': self.local_catalog[name].get('local_path', ''),
                    'last_used': self.local_catalog[name].get('last_used', 'Never')
                })
                
        return detectors
    
    def analyze_image_for_report(self, image, detector_name=None):
        """
        Analyze image and create comprehensive report
        
        Args:
            image: Input image
            detector_name: Name of the detector to use (None=use default)
            
        Returns:
            dict: Comprehensive analysis report
        """
        # Base detection
        detection_results = self.detect_steganography(image, detector_name)
        
        # Additional analyses for report
        report = {
            'detection_results': detection_results,
            'image_properties': {
                'shape': image.shape,
                'size_kb': image.nbytes / 1024,
                'type': str(image.dtype)
            },
            'lsb_analysis': {},
            'histogram_analysis': {},
            'recommendations': []
        }
        
        # LSB plane analysis
        if image.dtype == np.float32 and image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
            
        # Extract LSB planes
        lsb_planes = {}
        for channel, color in enumerate(['red', 'green', 'blue']):
            if len(img_uint8.shape) == 3 and img_uint8.shape[2] > channel:
                lsb = img_uint8[:, :, channel] & 1
                ones_percentage = np.sum(lsb) / lsb.size * 100
                zeros_percentage = 100 - ones_percentage
                lsb_planes[color] = {
                    'ones_percentage': ones_percentage,
                    'zeros_percentage': zeros_percentage,
                    'randomness_score': min(2 * min(ones_percentage, zeros_percentage), 100) / 100
                }
                
                # Check for natural distribution (should be close to 50/50)
                if abs(ones_percentage - 50) > 10:
                    report['recommendations'].append(
                        f"The {color} channel LSB distribution is unusual "
                        f"({ones_percentage:.1f}% ones, {zeros_percentage:.1f}% zeros). "
                        "This may indicate steganography."
                    )
        
        report['lsb_analysis'] = lsb_planes
        
        # Histogram analysis
        try:
            histograms = {}
            for channel, color in enumerate(['red', 'green', 'blue']):
                if len(img_uint8.shape) == 3 and img_uint8.shape[2] > channel:
                    hist = cv2.calcHist([img_uint8], [channel], None, [256], [0, 256])
                    hist = hist.flatten() / np.sum(hist)  # Normalize
                    
                    # Calculate entropy
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    
                    # Check for histogram "combing" effect (common in LSB steganography)
                    combing_score = 0
                    for i in range(0, 256, 2):
                        if i+1 < 256:
                            diff = abs(hist[i] - hist[i+1])
                            combing_score += diff
                            
                    combing_score = min(combing_score * 10, 1.0)  # Normalize to [0,1]
                    
                    histograms[color] = {
                        'entropy': float(entropy),
                        'combing_score': float(combing_score)
                    }
                    
                    # Add recommendation if combing is detected
                    if combing_score > 0.3:
                        report['recommendations'].append(
                            f"The {color} channel histogram shows possible 'combing' effect "
                            f"(score: {combing_score:.2f}), which is often associated with LSB steganography."
                        )
            
            report['histogram_analysis'] = histograms
        except Exception as e:
            logger.error(f"Error in histogram analysis: {str(e)}")
        
        # Add general recommendations based on detection
        if detection_results['is_stego']:
            report['recommendations'].append(
                f"This image likely contains hidden data (detection score: {detection_results['detection_score']:.2f}, "
                f"confidence: {detection_results['confidence']:.2f})."
            )
            
            if 'estimated_payload' in detection_results and detection_results['estimated_payload'] > 0:
                payload_percent = detection_results['estimated_payload'] * 100
                report['recommendations'].append(
                    f"Estimated hidden data capacity usage: {payload_percent:.1f}% of available capacity."
                )
        else:
            report['recommendations'].append(
                f"This image appears to be clean (detection score: {detection_results['detection_score']:.2f}, "
                f"confidence: {detection_results['confidence']:.2f})."
            )
        
        return report