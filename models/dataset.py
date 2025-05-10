"""
dataset.py
Modul untuk pengelolaan dataset dan preprocessing data
"""

import os
import numpy as np
import cv2
import logging
import requests
import zipfile
import hashlib
import json
import random
from tqdm import tqdm
import tensorflow as tf
from models.utils import preprocess_image

# Konfigurasi logging
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Class untuk mengelola dataset steganografi
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi dataset manager
        
        Args:
            config: Dict konfigurasi atau None untuk defaults
        """
        self.default_config = {
            'datasets_dir': 'datasets',
            'cache_dir': 'cache',
            'img_shape': (256, 256, 3),
            'max_images': 10000,
            'valid_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'dataset_sources': {
                'BOSS': {
                    'url': 'http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip',
                    'description': 'Break Our Steganography System (BOSS) dataset',
                    'md5': 'c2e27358d7dcaf7ec6cb74c349b63dc1',
                    'license': 'Research only'
                },
                'DIV2K': {
                    'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
                    'description': 'DIV2K high-resolution dataset (train)',
                    'md5': 'bdc2d9338d4e574fe81bf7d158758658',
                    'license': 'Research only'
                },
                'sample': {
                    'description': 'Sample dataset for testing',
                    'is_local': True
                }
            }
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Create directories
        os.makedirs(self.config['datasets_dir'], exist_ok=True)
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Dataset info cache
        self.dataset_info = {}
        self.cached_datasets = {}
    
    def list_available_datasets(self):
        """
        List semua dataset yang tersedia
        
        Returns:
            dict: Info tentang dataset yang tersedia
        """
        available = {}
        
        # Check configured datasets
        for name, info in self.config['dataset_sources'].items():
            dataset_path = os.path.join(self.config['datasets_dir'], name)
            is_available = os.path.exists(dataset_path)
            
            # Count images if available
            num_images = 0
            if is_available:
                for ext in self.config['valid_extensions']:
                    for root, _, files in os.walk(dataset_path):
                        num_images += sum(1 for f in files if f.lower().endswith(ext))
            
            # Add to available datasets
            available[name] = {
                'name': name,
                'path': dataset_path,
                'description': info.get('description', 'No description'),
                'is_available': is_available,
                'num_images': num_images,
                'license': info.get('license', 'Unknown')
            }
        
        # Check for other directories in datasets_dir
        for item in os.listdir(self.config['datasets_dir']):
            path = os.path.join(self.config['datasets_dir'], item)
            if os.path.isdir(path) and item not in available:
                # Count images
                num_images = 0
                for ext in self.config['valid_extensions']:
                    for root, _, files in os.walk(path):
                        num_images += sum(1 for f in files if f.lower().endswith(ext))
                
                # Add to available datasets
                available[item] = {
                    'name': item,
                    'path': path,
                    'description': 'User-provided dataset',
                    'is_available': True,
                    'num_images': num_images,
                    'license': 'Unknown'
                }
        
        return available
    
    def download_dataset(self, name):
        """
        Download dataset jika tersedia
        
        Args:
            name: Nama dataset untuk di-download
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if name not in self.config['dataset_sources']:
            logger.error(f"Dataset {name} not found in configuration")
            return False
            
        info = self.config['dataset_sources'][name]
        
        # Check if dataset is local-only
        if info.get('is_local', False):
            logger.warning(f"Dataset {name} is local-only and cannot be downloaded")
            return False
            
        # Check if URL is provided
        if 'url' not in info:
            logger.error(f"No download URL provided for dataset {name}")
            return False
            
        dataset_path = os.path.join(self.config['datasets_dir'], name)
        
        # Check if already exists
        if os.path.exists(dataset_path):
            logger.info(f"Dataset {name} already exists at {dataset_path}")
            return True
            
        # Create dataset directory
        os.makedirs(dataset_path, exist_ok=True)
        
        # Download dataset
        url = info['url']
        logger.info(f"Downloading dataset {name} from {url}")
        
        # Temporary file for download
        download_path = os.path.join(self.config['cache_dir'], f"{name}_download.zip")
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(download_path, 'wb') as f, tqdm(
                    desc=f"Downloading {name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify MD5 checksum if provided
            if 'md5' in info:
                expected_md5 = info['md5']
                logger.info(f"Verifying MD5 checksum: {expected_md5}")
                
                with open(download_path, 'rb') as f:
                    file_md5 = hashlib.md5(f.read()).hexdigest()
                
                if file_md5 != expected_md5:
                    logger.error(f"MD5 checksum mismatch. Expected: {expected_md5}, Got: {file_md5}")
                    os.remove(download_path)
                    return False
            
            # Extract dataset
            logger.info(f"Extracting dataset to {dataset_path}")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            
            # Clean up download file
            os.remove(download_path)
            
            logger.info(f"Dataset {name} downloaded and extracted to {dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset {name}: {str(e)}")
            if os.path.exists(download_path):
                os.remove(download_path)
            return False
    
    def load_dataset(self, name, max_images=None, img_shape=None, cache=True):
        """
        Load dataset ke memori
        
        Args:
            name: Nama dataset atau path ke direktori
            max_images: Jumlah maksimum gambar (None=gunakan config)
            img_shape: Shape gambar (None=gunakan config)
            cache: Apakah dataset akan di-cache
            
        Returns:
            numpy.ndarray: Array gambar
        """
        # Use default values if not provided
        if max_images is None:
            max_images = self.config['max_images']
            
        if img_shape is None:
            img_shape = self.config['img_shape']
        
        # Check if dataset is cached
        cache_key = f"{name}_{img_shape[0]}x{img_shape[1]}_{max_images}"
        if cache and cache_key in self.cached_datasets:
            logger.info(f"Using cached dataset: {name}")
            return self.cached_datasets[cache_key]
            
        # Determine dataset path
        if os.path.isdir(name):
            dataset_path = name
        else:
            dataset_path = os.path.join(self.config['datasets_dir'], name)
            
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return None
            
        # List image files
        files = []
        for root, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in self.config['valid_extensions']):
                    files.append(os.path.join(root, filename))
        
        if not files:
            logger.error(f"No image files found in {dataset_path}")
            return None
            
        # Shuffle and limit
        random.shuffle(files)
        files = files[:max_images]
        
        # Load images
        logger.info(f"Loading {len(files)} images from {dataset_path}")
        images = []
        
        for file_path in tqdm(files, desc=f"Loading {os.path.basename(dataset_path)}"):
            try:
                # Read image
                img = cv2.imread(file_path)
                if img is None:
                    continue
                    
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Preprocess
                img = preprocess_image(img, target_size=img_shape[:2])
                
                # Normalize to [0,1]
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                
            except Exception as e:
                logger.warning(f"Error loading image {file_path}: {str(e)}")
        
        if not images:
            logger.error(f"No valid images loaded from {dataset_path}")
            return None
            
        # Convert to numpy array
        dataset = np.array(images)
        
        # Cache if requested
        if cache:
            self.cached_datasets[cache_key] = dataset
            
        logger.info(f"Loaded dataset with {len(dataset)} images, shape: {dataset.shape}")
        
        # Update dataset info
        self.dataset_info[name] = {
            'num_images': len(dataset),
            'shape': dataset.shape,
            'size_mb': dataset.nbytes / (1024 * 1024)
        }
        
        return dataset
    
    def create_tf_dataset(self, images, message_generator=None, batch_size=32, shuffle=True):
        """
        Create TensorFlow dataset for training
        
        Args:
            images: Numpy array of images
            message_generator: Function to generate messages (None=random)
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Default message generator creates random binary messages
        if message_generator is None:
            message_length = 100  # Default message length
            def message_generator(batch_size):
                return np.random.randint(0, 2, size=(batch_size, message_length))
        
        # Create dataset from images
        dataset = tf.data.Dataset.from_tensor_slices(images)
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        # Batch the dataset
        dataset = dataset.batch(batch_size)
        
        # Map to create input pairs (cover_img, message)
        def map_fn(cover_batch):
            batch_size = tf.shape(cover_batch)[0]
            messages = tf.py_function(
                lambda b: message_generator(b.numpy()), 
                [batch_size], 
                tf.int32
            )
            return (cover_batch, messages), cover_batch
        
        dataset = dataset.map(map_fn)
        
        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def create_sample_dataset(self, num_images=10, img_shape=None):
        """
        Create sample dataset if none exists
        
        Args:
            num_images: Number of sample images to create
            img_shape: Shape of images (None=use config)
            
        Returns:
            numpy.ndarray: Sample dataset
        """
        if img_shape is None:
            img_shape = self.config['img_shape']
            
        # Create sample directory
        sample_dir = os.path.join(self.config['datasets_dir'], 'sample')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Check if samples already exist
        existing_files = [f for f in os.listdir(sample_dir) 
                         if f.endswith('.png') and f.startswith('sample_')]
        
        if len(existing_files) >= num_images:
            logger.info(f"Sample dataset already exists with {len(existing_files)} images")
            return self.load_dataset('sample')
            
        # Create new sample images
        logger.info(f"Creating {num_images} sample images")
        
        # Generate different types of sample images
        for i in range(num_images):
            img = np.zeros(img_shape, dtype=np.uint8)
            
            # Different pattern based on index
            if i % 5 == 0:
                # Gradient
                for x in range(img_shape[0]):
                    for y in range(img_shape[1]):
                        img[x, y, 0] = int(255 * x / img_shape[0])
                        img[x, y, 1] = int(255 * y / img_shape[1])
                        img[x, y, 2] = int(255 * (x + y) / (img_shape[0] + img_shape[1]))
                        
            elif i % 5 == 1:
                # Noise
                img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)
                
            elif i % 5 == 2:
                # Checkerboard
                cell_size = 32
                for x in range(img_shape[0]):
                    for y in range(img_shape[1]):
                        if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                            img[x, y, :] = 255
                            
            elif i % 5 == 3:
                # Circles
                for x in range(img_shape[0]):
                    for y in range(img_shape[1]):
                        dist = np.sqrt((x - img_shape[0]/2)**2 + (y - img_shape[1]/2)**2)
                        val = int(255 * (1 - dist / max(img_shape[0], img_shape[1])))
                        img[x, y, :] = val
                        
            else:
                # Random geometric shapes
                bg_color = np.random.randint(0, 256, size=3)
                img[:, :] = bg_color
                
                # Add random shapes
                for _ in range(20):
                    # Random shape type
                    shape_type = np.random.randint(0, 3)
                    color = np.random.randint(0, 256, size=3)
                    
                    if shape_type == 0:  # Rectangle
                        x1 = np.random.randint(0, img_shape[0])
                        y1 = np.random.randint(0, img_shape[1])
                        x2 = np.random.randint(x1, img_shape[0])
                        y2 = np.random.randint(y1, img_shape[1])
                        img = cv2.rectangle(img, (y1, x1), (y2, x2), color.tolist(), -1)
                        
                    elif shape_type == 1:  # Circle
                        x = np.random.randint(0, img_shape[0])
                        y = np.random.randint(0, img_shape[1])
                        r = np.random.randint(10, 50)
                        img = cv2.circle(img, (y, x), r, color.tolist(), -1)
                        
                    else:  # Line
                        x1 = np.random.randint(0, img_shape[0])
                        y1 = np.random.randint(0, img_shape[1])
                        x2 = np.random.randint(0, img_shape[0])
                        y2 = np.random.randint(0, img_shape[1])
                        img = cv2.line(img, (y1, x1), (y2, x2), color.tolist(), 
                                       np.random.randint(1, 10))
            
            # Save image
            save_path = os.path.join(sample_dir, f'sample_{i:03d}.png')
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Created {num_images} sample images in {sample_dir}")
        
        # Load the created dataset
        return self.load_dataset('sample')
    
    def generate_training_validation_split(self, dataset, val_split=0.2, seed=42):
        """
        Split dataset into training and validation sets
        
        Args:
            dataset: Numpy array of images
            val_split: Fraction for validation
            seed: Random seed
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        val_count = int(len(dataset) * val_split)
        
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        
        train_dataset = dataset[train_indices]
        val_dataset = dataset[val_indices]
        
        logger.info(f"Split dataset: {len(train_dataset)} training, {len(val_dataset)} validation")
        
        return train_dataset, val_dataset
    
    def analyze_dataset(self, dataset, name="Unknown"):
        """
        Analyze dataset properties and generate report
        
        Args:
            dataset: Numpy array of images
            name: Dataset name for report
            
        Returns:
            dict: Analysis results
        """
        if dataset is None or len(dataset) == 0:
            logger.error("Cannot analyze empty dataset")
            return None
            
        # Basic properties
        num_images = len(dataset)
        img_shape = dataset[0].shape
        memory_mb = dataset.nbytes / (1024 * 1024)
        
        # Calculate statistics
        means = np.mean(dataset, axis=(0, 1, 2))
        stds = np.std(dataset, axis=(0, 1, 2))
        mins = np.min(dataset, axis=(0, 1, 2))
        maxs = np.max(dataset, axis=(0, 1, 2))
        
        # Calculate average entropy as measure of complexity
        entropies = []
        for i in range(min(num_images, 100)):  # Limit to 100 images for speed
            img = (dataset[i] * 255).astype(np.uint8)
            entropy = 0
            for c in range(3):  # RGB channels
                hist = cv2.calcHist([img], [c], None, [256], [0, 256])
                hist = hist / np.sum(hist)
                channel_entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy += channel_entropy
            entropies.append(entropy / 3)  # Average over channels
            
        avg_entropy = np.mean(entropies)
        
        # Calculate histogram distribution
        total_hist = np.zeros((256, 3), dtype=np.float32)
        for i in range(min(num_images, 100)):
            img = (dataset[i] * 255).astype(np.uint8)
            for c in range(3):
                hist = cv2.calcHist([img], [c], None, [256], [0, 256])
                total_hist[:, c] += hist.flatten()
        
        # Normalize histogram
        total_hist = total_hist / np.sum(total_hist[:, 0])
        
        # Results
        results = {
            'name': name,
            'num_images': num_images,
            'image_shape': img_shape,
            'memory_mb': float(memory_mb),
            'means': means.tolist(),
            'stds': stds.tolist(),
            'mins': mins.tolist(),
            'maxs': maxs.tolist(),
            'avg_entropy': float(avg_entropy),
            'histogram': total_hist.tolist()
        }
        
        # Generate report
        logger.info(f"Dataset Analysis: {name}")
        logger.info(f"  Number of images: {num_images}")
        logger.info(f"  Image shape: {img_shape}")
        logger.info(f"  Memory usage: {memory_mb:.2f} MB")
        logger.info(f"  Channel means: R={means[0]:.3f}, G={means[1]:.3f}, B={means[2]:.3f}")
        logger.info(f"  Channel stds: R={stds[0]:.3f}, G={stds[1]:.3f}, B={stds[2]:.3f}")
        logger.info(f"  Average entropy: {avg_entropy:.3f} bits")
        
        # Cache results
        self.dataset_info[name] = {
            **self.dataset_info.get(name, {}),
            **results
        }
        
        return results
    
    def generate_report(self, output_path=None):
        """
        Generate report of all datasets
        
        Args:
            output_path: Path to save report JSON (None=default location)
            
        Returns:
            dict: Report data
        """
        if not output_path:
            output_path = os.path.join(self.config['datasets_dir'], 'datasets_report.json')
            
        # Available datasets
        available = self.list_available_datasets()
        
        # Dataset info from cache
        report = {
            'available_datasets': available,
            'dataset_info': self.dataset_info,
            'config': {
                'datasets_dir': self.config['datasets_dir'],
                'img_shape': self.config['img_shape'],
                'max_images': self.config['max_images']
            }
        }
        
        # Save report
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Dataset report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving dataset report: {str(e)}")
            
        return report

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Management Tool')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--download', type=str, help='Download dataset by name')
    parser.add_argument('--analyze', type=str, help='Analyze dataset by name')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample images')
    parser.add_argument('--report', action='store_true', help='Generate dataset report')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dataset manager
    dataset_manager = DatasetManager()
    
    if args.list:
        # List available datasets
        available = dataset_manager.list_available_datasets()
        print("Available Datasets:")
        print("------------------")
        for name, info in available.items():
            status = "Available" if info['is_available'] else "Not downloaded"
            print(f"{name}: {info['description']} ({status}, {info['num_images']} images)")
            
    elif args.download:
        # Download dataset
        success = dataset_manager.download_dataset(args.download)
        if success:
            print(f"Dataset {args.download} downloaded successfully")
        else:
            print(f"Failed to download dataset {args.download}")
            
    elif args.analyze:
        # Analyze dataset
        dataset = dataset_manager.load_dataset(args.analyze)
        if dataset is not None:
            analysis = dataset_manager.analyze_dataset(dataset, args.analyze)
            print(f"Analysis of {args.analyze} completed")
        else:
            print(f"Failed to load dataset {args.analyze}")
            
    elif args.create_sample:
        # Create sample dataset
        dataset = dataset_manager.create_sample_dataset(args.num_samples)
        if dataset is not None:
            print(f"Created sample dataset with {len(dataset)} images")
        else:
            print("Failed to create sample dataset")
            
    elif args.report:
        # Generate report
        report = dataset_manager.generate_report()
        print(f"Dataset report generated with {len(report['available_datasets'])} datasets")
        
    else:
        parser.print_help()