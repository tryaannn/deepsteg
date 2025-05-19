"""
pretrained_model_manager.py
Modul untuk mengelola dan menggunakan pre-trained models untuk steganografi
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

from models.enhanced_gan_model import EnhancedGANSteganography
from models.utils import load_model_if_exists

# Konfigurasi logging
logger = logging.getLogger(__name__)

class PretrainedModelManager:
    """
    Manager untuk pre-trained models steganografi
    - Download model
    - Verifikasi integritas
    - Load model
    - Provide encoder/decoder
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi model manager
        
        Args:
            config: Dictionary konfigurasi atau None untuk defaults
        """
        self.default_config = {
            'models_dir': 'models/saved',
            'pretrained_dir': 'models/pretrained',
            'cache_dir': 'cache',
            'model_catalog_url': 'https://example.com/deepsteg/catalog.json',  # Placeholder URL
            'available_models': {
                'gan_basic': {
                    'description': 'Basic GAN Steganography (256x256, 100 bit capacity)',
                    'urls': {
                        'model': 'https://example.com/deepsteg/gan_basic.zip',  # Placeholder URL
                        'info': 'https://example.com/deepsteg/gan_basic.json'   # Placeholder URL
                    },
                    'md5': '0123456789abcdef0123456789abcdef',  # Placeholder MD5
                    'size_mb': 12.5,
                    'img_shape': [256, 256, 3],
                    'message_length': 100,
                    'capacity_factor': 0.5
                },
                'gan_high_capacity': {
                    'description': 'High Capacity GAN Steganography (512x512, 1000 bit capacity)',
                    'urls': {
                        'model': 'https://example.com/deepsteg/gan_high_capacity.zip',  # Placeholder URL
                        'info': 'https://example.com/deepsteg/gan_high_capacity.json'   # Placeholder URL
                    },
                    'md5': 'fedcba9876543210fedcba9876543210',  # Placeholder MD5
                    'size_mb': 25.0,
                    'img_shape': [512, 512, 3],
                    'message_length': 1000,
                    'capacity_factor': 0.8
                }
            },
            'default_model': 'gan_basic',
            'auto_download': False
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Create directories
        os.makedirs(self.config['models_dir'], exist_ok=True)
        os.makedirs(self.config['pretrained_dir'], exist_ok=True)
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Loaded models cache
        self.loaded_models = {}
        
        # Load local model catalog
        self.local_catalog = self._load_local_catalog()
        
        # Auto-download if configured
        if self.config['auto_download']:
            default_model = self.config['default_model']
            if default_model not in self.local_catalog:
                logger.info(f"Auto-downloading default model: {default_model}")
                self.download_model(default_model)
    
    def _load_local_catalog(self):
        """
        Load catalog of locally available models
        
        Returns:
            dict: Local model catalog
        """
        catalog = {}
        
        # Check pretrained directory for model directories
        pretrained_dir = Path(self.config['pretrained_dir'])
        if not pretrained_dir.exists():
            return catalog
            
        for model_dir in pretrained_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Check for model info file
            info_file = model_dir / "model_info.json"
            if not info_file.exists():
                continue
                
            # Load model info
            try:
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                    
                model_name = model_dir.name
                
                # Check required files
                encoder_path = model_dir / "encoder.h5"
                decoder_path = model_dir / "decoder.h5"
                
                if encoder_path.exists() and decoder_path.exists():
                    catalog[model_name] = {
                        **model_info,
                        'local_path': str(model_dir),
                        'is_complete': True
                    }
                    logger.info(f"Found local model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model info for {model_dir.name}: {str(e)}")
                
        return catalog
    
    def update_catalog_from_remote(self):
        """
        Update model catalog from remote source
        
        Returns:
            bool: Success or failure
        """
        url = self.config['model_catalog_url']
        
        try:
            # Download catalog
            logger.info(f"Downloading model catalog from {url}")
            
            with urllib.request.urlopen(url) as response:
                remote_catalog = json.loads(response.read().decode('utf-8'))
                
            # Update available models config
            self.config['available_models'].update(remote_catalog.get('models', {}))
            
            # Save updated catalog
            catalog_path = os.path.join(self.config['cache_dir'], 'model_catalog.json')
            with open(catalog_path, 'w') as f:
                json.dump(self.config['available_models'], f, indent=2)
                
            logger.info(f"Catalog updated with {len(remote_catalog.get('models', {}))} models")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model catalog: {str(e)}")
            return False
    
    def list_available_models(self):
        """
        List all available models (local and remote)
        
        Returns:
            dict: Model information
        """
        models = {}
        
        # Add remote models
        for name, info in self.config['available_models'].items():
            is_local = name in self.local_catalog
            
            models[name] = {
                'name': name,
                'description': info.get('description', 'No description'),
                'is_local': is_local,
                'size_mb': info.get('size_mb', 0),
                'img_shape': info.get('img_shape', [0, 0, 0]),
                'message_length': info.get('message_length', 0)
            }
            
            # Add local info if available
            if is_local:
                models[name].update({
                    'local_path': self.local_catalog[name].get('local_path', ''),
                    'last_used': self.local_catalog[name].get('last_used', 'Never')
                })
                
        return models
    
    def download_model(self, model_name):
        """
        Download a pre-trained model
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            bool: Success or failure
        """
        if model_name not in self.config['available_models']:
            logger.error(f"Model {model_name} not found in catalog")
            return False
            
        model_info = self.config['available_models'][model_name]
        
        # Check if model URLs are provided
        if 'urls' not in model_info or 'model' not in model_info['urls']:
            logger.error(f"Model {model_name} has no download URL")
            return False
            
        model_url = model_info['urls']['model']
        
        # Create model directory
        model_dir = os.path.join(self.config['pretrained_dir'], model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Download zip file
        zip_path = os.path.join(self.config['cache_dir'], f"{model_name}.zip")
        
        try:
            logger.info(f"Downloading model {model_name} from {model_url}")
            
            # Download with progress bar
            with urllib.request.urlopen(model_url) as response:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 8192
                
                with open(zip_path, 'wb') as f, tqdm(
                        desc=f"Downloading {model_name}",
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
            if 'md5' in model_info:
                logger.info(f"Verifying MD5 checksum: {model_info['md5']}")
                
                with open(zip_path, 'rb') as f:
                    file_md5 = hashlib.md5(f.read()).hexdigest()
                
                if file_md5 != model_info['md5']:
                    logger.error(f"MD5 checksum mismatch. Expected: {model_info['md5']}, Got: {file_md5}")
                    os.remove(zip_path)
                    return False
            
            # Extract files
            logger.info(f"Extracting model files to {model_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Remove zip file
            os.remove(zip_path)
            
            # Download info file if available
            if 'info' in model_info['urls']:
                info_url = model_info['urls']['info']
                info_path = os.path.join(model_dir, "model_info.json")
                
                try:
                    logger.info(f"Downloading model info from {info_url}")
                    urllib.request.urlretrieve(info_url, info_path)
                except Exception as e:
                    logger.warning(f"Error downloading model info: {str(e)}")
                    
                    # Create basic info file
                    with open(info_path, 'w') as f:
                        json.dump({
                            'name': model_name,
                            'description': model_info.get('description', ''),
                            'img_shape': model_info.get('img_shape', [256, 256, 3]),
                            'message_length': model_info.get('message_length', 100),
                            'capacity_factor': model_info.get('capacity_factor', 0.5),
                            'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
                        }, f, indent=2)
            
            # Update local catalog
            self.local_catalog = self._load_local_catalog()
            
            logger.info(f"Model {model_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            
            # Clean up
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            return False
    
    def load_model(self, model_name=None):
        """
        Load a pre-trained model
        
        Args:
            model_name: Name of the model to load (None=use default)
            
        Returns:
            EnhancedGANSteganography or None: Loaded model
        """
        # Use default if not specified
        if model_name is None:
            model_name = self.config['default_model']
            
        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Using cached model {model_name}")
            
            # Update last used timestamp
            if model_name in self.local_catalog:
                info_path = os.path.join(self.local_catalog[model_name]['local_path'], "model_info.json")
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    info['last_used'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    with open(info_path, 'w') as f:
                        json.dump(info, f, indent=2)
                except Exception as e:
                    logger.warning(f"Error updating last used timestamp: {str(e)}")
            
            return self.loaded_models[model_name]
            
        # Check if model is available locally
        if model_name not in self.local_catalog:
            logger.warning(f"Model {model_name} not found locally")
            
            # Try to download if available
            if model_name in self.config['available_models']:
                logger.info(f"Attempting to download model {model_name}")
                if not self.download_model(model_name):
                    logger.error(f"Failed to download model {model_name}")
                    return None
            else:
                logger.error(f"Model {model_name} not found in catalog")
                return None
                
        # Get model directory
        model_dir = self.local_catalog[model_name]['local_path']
        
        # Load model info
        info_path = os.path.join(model_dir, "model_info.json")
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            return None
            
        # Create GAN steganography model
        img_shape = tuple(model_info.get('img_shape', [256, 256, 3]))
        message_length = model_info.get('message_length', 100)
        capacity_factor = model_info.get('capacity_factor', 0.5)
        
        try:
            # Initialize model
            gan_model = EnhancedGANSteganography(
                img_shape=img_shape,
                message_length=message_length,
                capacity_factor=capacity_factor
            )
            
            # Load pre-trained weights
            success = gan_model.load_models(model_dir)
            
            if not success:
                logger.error(f"Failed to load model files from {model_dir}")
                return None
                
            # Update model info
            model_info['last_used'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
                
            # Cache loaded model
            self.loaded_models[model_name] = gan_model
            
            logger.info(f"Successfully loaded model {model_name}")
            return gan_model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def encode_with_model(self, model_name, cover_image, message, options=None):
        """
        Encode message using pre-trained model
        
        Args:
            model_name: Name of the model to use
            cover_image: Cover image (numpy array)
            message: Message to encode
            options: Dictionary of options
            
        Returns:
            tuple: (stego_image, metrics)
        """
        # Default options
        default_options = {
            'normalize': True,  # Normalize image to [0,1]
            'message_format': 'text',  # 'text' or 'binary'
        }
        
        if options is None:
            options = {}
            
        # Merge options
        options = {**default_options, **options}
        
        # Load model
        model = self.load_model(model_name)
        if model is None:
            logger.error(f"Failed to load model {model_name}")
            return None, {'error': f"Failed to load model {model_name}"}
            
        try:
            # Process cover image
            if options['normalize'] and (cover_image.dtype != np.float32 or cover_image.max() > 1.0):
                img = cover_image.astype(np.float32) / 255.0
            else:
                img = cover_image.astype(np.float32)
                
            # Process message
            if options['message_format'] == 'text':
                # Convert text to binary
                binary_message = []
                for char in message:
                    binary_message.extend([int(bit) for bit in format(ord(char), '08b')])
                msg = np.array(binary_message)
            else:
                # Already binary
                msg = np.array(message)
                
            # Check message length
            if len(msg) > model.message_length:
                logger.warning(f"Message too long, truncating to {model.message_length} bits")
                msg = msg[:model.message_length]
            elif len(msg) < model.message_length:
                # Pad with zeros
                msg = np.pad(msg, (0, model.message_length - len(msg)))
                
            # Encode message
            start_time = time.time()
            stego_img = model.encode_message(img, msg)
            encoding_time = time.time() - start_time
            
            # Calculate metrics
            from models.utils import calculate_metrics
            metrics = calculate_metrics(img, stego_img)
            metrics['encoding_time'] = encoding_time
            metrics['capacity_used'] = (len(msg) / model.message_length) * 100
            
            return stego_img, metrics
            
        except Exception as e:
            logger.error(f"Error encoding with model {model_name}: {str(e)}")
            return None, {'error': str(e)}
    
    def decode_with_model(self, model_name, stego_image, options=None):
        """
        Decode message using pre-trained model
        
        Args:
            model_name: Name of the model to use
            stego_image: Stego image (numpy array)
            options: Dictionary of options
            
        Returns:
            tuple: (message, metadata)
        """
        # Default options
        default_options = {
            'normalize': True,  # Normalize image to [0,1]
            'message_format': 'text',  # 'text' or 'binary'
            'threshold': 0.5  # Threshold for binary decision
        }
        
        if options is None:
            options = {}
            
        # Merge options
        options = {**default_options, **options}
        
        # Load model
        model = self.load_model(model_name)
        if model is None:
            logger.error(f"Failed to load model {model_name}")
            return None, {'error': f"Failed to load model {model_name}"}
            
        try:
            # Process stego image
            if options['normalize'] and (stego_image.dtype != np.float32 or stego_image.max() > 1.0):
                img = stego_image.astype(np.float32) / 255.0
            else:
                img = stego_image.astype(np.float32)
                
            # Decode message
            start_time = time.time()
            message_bits = model.decode_message(img)
            decoding_time = time.time() - start_time
            
            # Convert to binary with threshold
            binary_message = (message_bits > options['threshold']).astype(int)
            
            # Convert binary to text if requested
            if options['message_format'] == 'text':
                # Find terminator (8 zeros) or use all bits
                terminator_pos = -1
                for i in range(0, len(binary_message) - 7, 8):
                    if np.all(binary_message[i:i+8] == 0):
                        terminator_pos = i
                        break
                        
                if terminator_pos >= 0:
                    binary_message = binary_message[:terminator_pos]
                    
                # Convert binary to text
                message = ""
                for i in range(0, len(binary_message), 8):
                    if i + 8 <= len(binary_message):
                        byte = binary_message[i:i+8]
                        char_code = int(''.join(map(str, byte)), 2)
                        if 32 <= char_code <= 126 or char_code in (9, 10, 13):  # Printable ASCII or whitespace
                            message += chr(char_code)
                        else:
                            message += '?'  # Non-printable character
            else:
                # Return raw binary
                message = binary_message
                
            # Metadata
            metadata = {
                'decoding_time': decoding_time,
                'message_length': len(binary_message),
                'model_used': model_name,
                'is_text': options['message_format'] == 'text'
            }
            
            return message, metadata
            
        except Exception as e:
            logger.error(f"Error decoding with model {model_name}: {str(e)}")
            return None, {'error': str(e)}
    
    def get_model_info(self, model_name):
        """
        Get detailed information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            dict: Model information
        """
        # Check local catalog
        if model_name in self.local_catalog:
            model_dir = self.local_catalog[model_name]['local_path']
            info_path = os.path.join(model_dir, "model_info.json")
            
            try:
                with open(info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model info: {str(e)}")
                
        # Check remote catalog
        if model_name in self.config['available_models']:
            return self.config['available_models'][model_name]
            
        return None
    
    def delete_model(self, model_name):
        """
        Delete a pre-trained model
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            bool: Success or failure
        """
        # Check if model is available locally
        if model_name not in self.local_catalog:
            logger.warning(f"Model {model_name} not found locally")
            return False
            
        # Get model directory
        model_dir = self.local_catalog[model_name]['local_path']
        
        # Remove from loaded models
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
        try:
            # Remove directory
            import shutil
            shutil.rmtree(model_dir)
            
            # Update local catalog
            self.local_catalog = self._load_local_catalog()
            
            logger.info(f"Model {model_name} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False