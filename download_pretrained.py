#!/usr/bin/env python
"""
download_pretrained.py
Script untuk mendownload pre-trained models untuk DeepSteg
"""

import os
import argparse
import logging
import json
import sys
from tqdm import tqdm
import urllib.request
import shutil
import hashlib
import zipfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deepsteg-download')

# Default configuration
DEFAULT_CONFIG = {
    'models_dir': 'models/pretrained',
    'detectors_dir': 'models/saved/steganalysis',
    'cache_dir': 'cache',
    'source_url': 'https://example.com/deepsteg/models',  # Placeholder URL
    'model_catalog_url': 'https://example.com/deepsteg/catalog.json',  # Placeholder URL
    'model_catalog': {
        'gan_basic': {
            'description': 'Basic GAN Steganography (256x256, 100 bit capacity)',
            'url': 'https://example.com/deepsteg/gan_basic.zip',  # Placeholder URL
            'md5': '0123456789abcdef0123456789abcdef',  # Placeholder MD5
            'size_mb': 12.5,
            'priority': 1
        },
        'gan_high_capacity': {
            'description': 'High Capacity GAN Steganography (512x512, 1000 bit capacity)',
            'url': 'https://example.com/deepsteg/gan_high_capacity.zip',  # Placeholder URL
            'md5': 'fedcba9876543210fedcba9876543210',  # Placeholder MD5
            'size_mb': 25.0,
            'priority': 2
        },
        'gan_small': {
            'description': 'Small GAN Steganography (128x128, 50 bit capacity)',
            'url': 'https://example.com/deepsteg/gan_small.zip',  # Placeholder URL
            'md5': '1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p',  # Placeholder MD5
            'size_mb': 8.0,
            'priority': 3
        }
    },
    'detector_catalog': {
        'cnn_basic': {
            'description': 'Basic CNN Detector (256x256, LSB focused)',
            'url': 'https://example.com/deepsteg/detectors/cnn_basic.zip',  # Placeholder URL
            'md5': '0123456789abcdef0123456789abcdef',  # Placeholder MD5
            'size_mb': 5.5,
            'priority': 1
        },
        'deep_detector': {
            'description': 'Deep CNN Detector (512x512, GAN steganography focused)',
            'url': 'https://example.com/deepsteg/detectors/deep_detector.zip',  # Placeholder URL
            'md5': 'fedcba9876543210fedcba9876543210',  # Placeholder MD5
            'size_mb': 18.0,
            'priority': 2
        }
    }
}

def create_directories(config):
    """Create necessary directories"""
    os.makedirs(config['models_dir'], exist_ok=True)
    os.makedirs(config['detectors_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)

def verify_checksum(file_path, expected_md5):
    """Verify file checksum"""
    if not os.path.exists(file_path):
        return False
        
    # Calculate MD5
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buffer = f.read()
        hasher.update(buffer)
    file_md5 = hasher.hexdigest()
    
    return file_md5 == expected_md5

def download_file(url, local_path, desc=None, expected_md5=None):
    """Download file with progress bar and checksum verification"""
    try:
        # Check if file already exists and has correct checksum
        if os.path.exists(local_path) and expected_md5 and verify_checksum(local_path, expected_md5):
            logger.info(f"File already exists and checksum is valid: {local_path}")
            return True
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            
            with open(local_path, 'wb') as f, tqdm(
                    desc=desc or f"Downloading {os.path.basename(local_path)}",
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
        
        # Verify checksum if provided
        if expected_md5:
            if verify_checksum(local_path, expected_md5):
                logger.info(f"Checksum verification successful for {local_path}")
            else:
                logger.error(f"Checksum verification failed for {local_path}")
                os.remove(local_path)
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def extract_zip(zip_path, extract_dir):
    """Extract zip file to directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {str(e)}")
        return False

def download_model(config, model_name):
    """Download specific model"""
    if model_name not in config['model_catalog']:
        logger.error(f"Model '{model_name}' not found in catalog")
        return False
    
    model_info = config['model_catalog'][model_name]
    
    # Get download URL
    url = model_info['url']
    
    # Download to cache
    zip_path = os.path.join(config['cache_dir'], f"{model_name}.zip")
    success = download_file(
        url, 
        zip_path, 
        desc=f"Downloading model '{model_name}'",
        expected_md5=model_info.get('md5')
    )
    
    if not success:
        return False
    
    # Extract to models directory
    model_dir = os.path.join(config['models_dir'], model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Extracting {zip_path} to {model_dir}")
    if not extract_zip(zip_path, model_dir):
        return False
    
    # Create model info file if not exists
    info_path = os.path.join(model_dir, "model_info.json")
    if not os.path.exists(info_path):
        with open(info_path, 'w') as f:
            json.dump({
                'name': model_name,
                'description': model_info.get('description', ''),
                'source': 'pre-trained',
                'download_url': url,
                'size_mb': model_info.get('size_mb', 0)
            }, f, indent=2)
    
    # Clean up
    os.remove(zip_path)
    
    logger.info(f"Model '{model_name}' downloaded and extracted successfully")
    return True

def download_detector(config, detector_name):
    """Download specific detector"""
    if detector_name not in config['detector_catalog']:
        logger.error(f"Detector '{detector_name}' not found in catalog")
        return False
    
    detector_info = config['detector_catalog'][detector_name]
    
    # Get download URL
    url = detector_info['url']
    
    # Download to cache
    zip_path = os.path.join(config['cache_dir'], f"{detector_name}.zip")
    success = download_file(
        url, 
        zip_path, 
        desc=f"Downloading detector '{detector_name}'",
        expected_md5=detector_info.get('md5')
    )
    
    if not success:
        return False
    
    # Extract to detectors directory
    detector_dir = os.path.join(config['detectors_dir'], detector_name)
    os.makedirs(detector_dir, exist_ok=True)
    
    logger.info(f"Extracting {zip_path} to {detector_dir}")
    if not extract_zip(zip_path, detector_dir):
        return False
    
    # Create detector info file if not exists
    info_path = os.path.join(detector_dir, "detector_info.json")
    if not os.path.exists(info_path):
        with open(info_path, 'w') as f:
            json.dump({
                'name': detector_name,
                'description': detector_info.get('description', ''),
                'source': 'pre-trained',
                'download_url': url,
                'size_mb': detector_info.get('size_mb', 0)
            }, f, indent=2)
    
    # Clean up
    os.remove(zip_path)
    
    logger.info(f"Detector '{detector_name}' downloaded and extracted successfully")
    return True

def download_catalog(config):
    """Download model catalog from remote source"""
    url = config['model_catalog_url']
    local_path = os.path.join(config['cache_dir'], 'model_catalog.json')
    
    try:
        logger.info(f"Downloading model catalog from {url}")
        if download_file(url, local_path):
            # Load catalog
            with open(local_path, 'r') as f:
                catalog = json.load(f)
            
            # Update config
            if 'models' in catalog:
                config['model_catalog'].update(catalog['models'])
            if 'detectors' in catalog:
                config['detector_catalog'].update(catalog['detectors'])
                
            logger.info(f"Catalog updated with {len(catalog.get('models', {}))} models and {len(catalog.get('detectors', {}))} detectors")
            return True
        return False
    except Exception as e:
        logger.error(f"Error downloading catalog: {str(e)}")
        return False

def list_models(config):
    """List available models in catalog"""
    models = config['model_catalog']
    
    if not models:
        logger.info("No models available in catalog")
        return
    
    # Sort by priority
    sorted_models = sorted(models.items(), key=lambda x: x[1].get('priority', 999))
    
    print("\nAvailable Pre-trained Models:")
    print("=" * 80)
    print(f"{'Name':<20} {'Size':<10} {'Description':<50}")
    print("-" * 80)
    
    for name, info in sorted_models:
        size = f"{info.get('size_mb', 0):.1f} MB"
        desc = info.get('description', 'No description')
        if len(desc) > 50:
            desc = desc[:47] + "..."
        print(f"{name:<20} {size:<10} {desc:<50}")
    
    print("=" * 80)
    print()

def list_detectors(config):
    """List available detectors in catalog"""
    detectors = config['detector_catalog']
    
    if not detectors:
        logger.info("No detectors available in catalog")
        return
    
    # Sort by priority
    sorted_detectors = sorted(detectors.items(), key=lambda x: x[1].get('priority', 999))
    
    print("\nAvailable Pre-trained Detectors:")
    print("=" * 80)
    print(f"{'Name':<20} {'Size':<10} {'Description':<50}")
    print("-" * 80)
    
    for name, info in sorted_detectors:
        size = f"{info.get('size_mb', 0):.1f} MB"
        desc = info.get('description', 'No description')
        if len(desc) > 50:
            desc = desc[:47] + "..."
        print(f"{name:<20} {size:<10} {desc:<50}")
    
    print("=" * 80)
    print()

def download_all_models(config):
    """Download all models in catalog"""
    models = config['model_catalog']
    
    if not models:
        logger.info("No models available in catalog")
        return
    
    # Sort by priority
    sorted_models = sorted(models.items(), key=lambda x: x[1].get('priority', 999))
    
    success_count = 0
    fail_count = 0
    
    for name, _ in sorted_models:
        logger.info(f"Downloading model '{name}'...")
        if download_model(config, name):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Download complete: {success_count} successful, {fail_count} failed")

def download_all_detectors(config):
    """Download all detectors in catalog"""
    detectors = config['detector_catalog']
    
    if not detectors:
        logger.info("No detectors available in catalog")
        return
    
    # Sort by priority
    sorted_detectors = sorted(detectors.items(), key=lambda x: x[1].get('priority', 999))
    
    success_count = 0
    fail_count = 0
    
    for name, _ in sorted_detectors:
        logger.info(f"Downloading detector '{name}'...")
        if download_detector(config, name):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Download complete: {success_count} successful, {fail_count} failed")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download DeepSteg pre-trained models')
    
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--list-detectors', action='store_true', help='List available detectors')
    parser.add_argument('--download-model', type=str, help='Download specific model')
    parser.add_argument('--download-detector', type=str, help='Download specific detector')
    parser.add_argument('--download-all-models', action='store_true', help='Download all models')
    parser.add_argument('--download-all-detectors', action='store_true', help='Download all detectors')
    parser.add_argument('--update-catalog', action='store_true', help='Update model catalog from remote')
    parser.add_argument('--models-dir', type=str, help='Directory for storing models')
    parser.add_argument('--detectors-dir', type=str, help='Directory for storing detectors')
    parser.add_argument('--cache-dir', type=str, help='Directory for cache files')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration (from JSON if exists, otherwise use defaults)
    config = DEFAULT_CONFIG.copy()
    
    # Override config from command-line args
    if args.models_dir:
        config['models_dir'] = args.models_dir
    if args.detectors_dir:
        config['detectors_dir'] = args.detectors_dir
    if args.cache_dir:
        config['cache_dir'] = args.cache_dir
    
    # Create directories
    create_directories(config)
    
    # Update catalog if requested
    if args.update_catalog:
        download_catalog(config)
    
    # Process commands
    if args.list_models:
        list_models(config)
    elif args.list_detectors:
        list_detectors(config)
    elif args.download_model:
        download_model(config, args.download_model)
    elif args.download_detector:
        download_detector(config, args.download_detector)
    elif args.download_all_models:
        download_all_models(config)
    elif args.download_all_detectors:
        download_all_detectors(config)
    else:
        # No command specified, show help and list options
        print("DeepSteg Pre-trained Model Downloader")
        print("====================================")
        print("\nAvailable commands:")
        print("  --list-models             List available models")
        print("  --list-detectors          List available detectors")
        print("  --download-model NAME     Download specific model")
        print("  --download-detector NAME  Download specific detector")
        print("  --download-all-models     Download all models")
        print("  --download-all-detectors  Download all detectors")
        print("  --update-catalog          Update model catalog from remote")
        print("\nExample usage:")
        print("  python download_pretrained.py --list-models")
        print("  python download_pretrained.py --download-model gan_basic")
        
        # Show available models and detectors
        list_models(config)
        list_detectors(config)

if __name__ == "__main__":
    main()