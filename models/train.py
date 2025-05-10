"""
train.py
Modul untuk melatih model GAN steganografi dengan berbagai dataset
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json

from models.enhanced_gan_model import EnhancedGANSteganography
from models.utils import preprocess_image, text_to_bits

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GANTrainer:
    """
    Class untuk melatih model GAN steganografi
    """
    def __init__(self, config=None):
        """
        Inisialisasi GAN Trainer
        
        Args:
            config: Dict konfigurasi training atau path ke file konfigurasi
        """
        self.default_config = {
            'img_shape': (256, 256, 3),
            'message_length': 100,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.0001,
            'beta1': 0.5,
            'capacity_factor': 0.5,
            'dataset_path': 'datasets/sample',
            'checkpoint_dir': 'models/checkpoints',
            'save_dir': 'models/saved',
            'log_dir': 'logs/fit'
        }
        
        # Load konfigurasi
        if config is None:
            self.config = self.default_config
        elif isinstance(config, str) and os.path.exists(config):
            # Load dari file
            with open(config, 'r') as f:
                self.config = json.load(f)
                # Fill missing with defaults
                for key, value in self.default_config.items():
                    if key not in self.config:
                        self.config[key] = value
        else:
            # Use dict config
            self.config = {**self.default_config, **config}
        
        # Create directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Initialize model
        self.gan = EnhancedGANSteganography(
            img_shape=self.config['img_shape'],
            message_length=self.config['message_length'],
            capacity_factor=self.config['capacity_factor']
        )
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=self.config['beta1']
        )
        
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=self.config['beta1']
        )
        
        # Metrics
        self.metrics = {
            'd_loss': [],
            'd_acc': [],
            'g_loss': [],
            'img_loss': [],
            'msg_loss': [],
            'adv_loss': [],
            'psnr': [],
            'ssim': [],
            'msg_acc': []
        }
        
        # TensorBoard
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.config['log_dir'], timestamp)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    
    def load_dataset(self, dataset_path=None):
        """
        Load dataset dari direktori
        
        Args:
            dataset_path: Path direktori dataset
        
        Returns:
            Dataset tensor
        """
        if dataset_path is None:
            dataset_path = self.config['dataset_path']
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Check if path exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path {dataset_path} does not exist")
            raise FileNotFoundError(f"Dataset path {dataset_path} not found")
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            logger.error(f"No image files found in {dataset_path}")
            raise ValueError(f"No image files found in {dataset_path}")
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Create dataset
        list_ds = tf.data.Dataset.from_tensor_slices(image_files)
        
        # Load and preprocess images
        def process_path(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, self.config['img_shape'][:2])
            img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
            return img
        
        # Map function to dataset
        image_ds = list_ds.map(process_path, 
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Configure dataset for performance
        ds = image_ds.cache().shuffle(buffer_size=1000).batch(self.config['batch_size'])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return ds
    
    def generate_random_messages(self, batch_size=None):
        """
        Generate random binary messages for training
        
        Args:
            batch_size: Jumlah pesan yang akan digenerate
            
        Returns:
            Numpy array dari pesan binary
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        return np.random.randint(0, 2, size=(batch_size, self.config['message_length']))
    
    @tf.function
    def train_step(self, cover_images, messages):
        """
        Satu step training menggunakan TensorFlow eager execution
        
        Args:
            cover_images: Batch gambar cover
            messages: Batch pesan
            
        Returns:
            Dictionary metrik
        """
        # Persistent GradientTape karena kita akan menggunakannya beberapa kali
        with tf.GradientTape(persistent=True) as tape:
            # Generate stego images
            stego_images = self.gan.encoder([cover_images, messages], training=True)
            
            # Decode messages from stego images
            decoded_messages = self.gan.decoder(stego_images, training=True)
            
            # Discriminator predictions
            real_output = self.gan.discriminator(cover_images, training=True)
            fake_output = self.gan.discriminator(stego_images, training=True)
            
            # Losses
            # Discriminator loss
            d_loss_real = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
            d_loss_fake = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
            
            # Generator losses
            g_adv_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)
            g_img_loss = tf.keras.losses.MeanSquaredError()(cover_images, stego_images)
            g_msg_loss = tf.keras.losses.BinaryCrossentropy()(messages, decoded_messages)
            
            # Combined generator loss
            g_loss = (0.1 * g_adv_loss + 
                      0.7 * g_img_loss + 
                      0.2 * g_msg_loss)
        
        # Compute gradients
        d_gradients = tape.gradient(d_loss, self.gan.discriminator.trainable_variables)
        g_gradients = tape.gradient(g_loss, self.gan.encoder.trainable_variables + 
                                    self.gan.decoder.trainable_variables)
        
        # Apply gradients
        self.d_optimizer.apply_gradients(zip(d_gradients, self.gan.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, 
                                             self.gan.encoder.trainable_variables + 
                                             self.gan.decoder.trainable_variables))
        
        # Calculate performance metrics
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr = tf.reduce_mean(tf.image.psnr(cover_images, stego_images, max_val=1.0))
        
        # SSIM (Structural Similarity)
        ssim = tf.reduce_mean(tf.image.ssim(cover_images, stego_images, max_val=1.0))
        
        # Message bit accuracy
        msg_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.round(decoded_messages), messages), tf.float32))
        
        # Discriminator accuracy
        d_acc_real = tf.reduce_mean(tf.cast(tf.greater(real_output, 0.5), tf.float32))
        d_acc_fake = tf.reduce_mean(tf.cast(tf.less(fake_output, 0.5), tf.float32))
        d_acc = (d_acc_real + d_acc_fake) / 2.0
        
        # Return metrics
        return {
            'd_loss': d_loss,
            'd_acc': d_acc,
            'g_loss': g_loss,
            'g_adv_loss': g_adv_loss,
            'g_img_loss': g_img_loss,
            'g_msg_loss': g_msg_loss,
            'psnr': psnr,
            'ssim': ssim,
            'msg_acc': msg_acc
        }
    
    def train(self, dataset=None, epochs=None):
        """
        Train model untuk jumlah epoch tertentu
        
        Args:
            dataset: Dataset gambar (jika None, akan dimuat dari config)
            epochs: Jumlah epoch (jika None, akan diambil dari config)
            
        Returns:
            History training
        """
        if epochs is None:
            epochs = self.config['epochs']
            
        if dataset is None:
            dataset = self.load_dataset()
            
        # Setup checkpoint manager
        checkpoint_prefix = os.path.join(self.config['checkpoint_dir'], "ckpt")
        checkpoint = tf.train.Checkpoint(
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            encoder=self.gan.encoder,
            decoder=self.gan.decoder,
            discriminator=self.gan.discriminator
        )
        manager = tf.train.CheckpointManager(
            checkpoint, self.config['checkpoint_dir'], max_to_keep=5)
        
        # Load checkpoint if available
        checkpoint_status = checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logger.info(f"Restored from {manager.latest_checkpoint}")
            epoch_start = int(manager.latest_checkpoint.split('-')[-1])
        else:
            logger.info("Initializing from scratch")
            epoch_start = 0
        
        # Training loop
        for epoch in range(epoch_start, epochs):
            start_time = time.time()
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Metrics for current epoch
            epoch_metrics = {
                'd_loss': [],
                'd_acc': [],
                'g_loss': [],
                'g_adv_loss': [],
                'g_img_loss': [],
                'g_msg_loss': [],
                'psnr': [],
                'ssim': [],
                'msg_acc': []
            }
            
            # Progress bar
            pbar = tqdm(total=len(dataset), desc=f"Epoch {epoch+1}/{epochs}")
            
            # Iterate over dataset batches
            for batch_idx, cover_images in enumerate(dataset):
                # Generate random messages
                messages = self.generate_random_messages(cover_images.shape[0])
                
                # Train step
                metrics = self.train_step(cover_images, messages)
                
                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key].numpy())
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'd_loss': f"{metrics['d_loss'].numpy():.4f}",
                    'g_loss': f"{metrics['g_loss'].numpy():.4f}",
                    'psnr': f"{metrics['psnr'].numpy():.2f}",
                    'msg_acc': f"{metrics['msg_acc'].numpy():.4f}"
                })
            
            pbar.close()
            
            # Calculate epoch averages
            epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Update metrics history
            for key in self.metrics:
                if key in epoch_avg:
                    self.metrics[key].append(epoch_avg[key])
            
            # Write to TensorBoard
            with self.summary_writer.as_default():
                for key, value in epoch_avg.items():
                    tf.summary.scalar(key, value, step=epoch)
                
                # Sample images for visualization
                if batch_idx % 10 == 0:
                    # Get sample batch
                    sample_covers = next(iter(dataset))[:5]  # 5 samples
                    sample_messages = self.generate_random_messages(5)
                    
                    # Generate stego images
                    sample_stegos = self.gan.encoder.predict([sample_covers, sample_messages])
                    
                    # Create comparison grid
                    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
                    for i in range(5):
                        # Original
                        axes[i, 0].imshow(sample_covers[i])
                        axes[i, 0].set_title('Original')
                        axes[i, 0].axis('off')
                        
                        # Stego
                        axes[i, 1].imshow(sample_stegos[i])
                        axes[i, 1].set_title('Stego')
                        axes[i, 1].axis('off')
                    
                    # Convert to image
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # Write to TensorBoard
                    tf.summary.image("Cover vs Stego", np.expand_dims(image, 0), step=epoch)
                    
                    plt.close(fig)
            
            # Save checkpoint
            save_path = manager.save()
            logger.info(f"Saved checkpoint for epoch {epoch+1}: {save_path}")
            
            # Calculate time per epoch
            time_per_epoch = time.time() - start_time
            logger.info(f"Time per epoch: {time_per_epoch:.2f} sec")
            
            # Print metrics
            metrics_str = ", ".join([f"{key}: {epoch_avg[key]:.4f}" for key in 
                                     ['d_loss', 'g_loss', 'psnr', 'ssim', 'msg_acc']])
            logger.info(f"Epoch {epoch+1}/{epochs}: {metrics_str}")
        
        # Save final model
        self.gan.save_models(self.config['save_dir'])
        logger.info(f"Model saved to {self.config['save_dir']}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.metrics
    
    def plot_training_history(self):
        """
        Plot training history metrics
        """
        if not self.metrics['d_loss']:  # Check if there's training data
            logger.warning("No training history to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        axes[0, 0].plot(self.metrics['d_loss'], label='Discriminator')
        axes[0, 0].plot(self.metrics['g_loss'], label='Generator')
        axes[0, 0].set_title('Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot image quality metrics
        axes[0, 1].plot(self.metrics['psnr'], label='PSNR')
        axes[0, 1].set_title('Image Quality (PSNR)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True)
        
        # Plot SSIM
        ax2 = axes[0, 1].twinx()
        ax2.plot(self.metrics['ssim'], 'r-', label='SSIM')
        ax2.set_ylabel('SSIM', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legends
        lines1, labels1 = axes[0, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[0, 1].legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        # Plot message accuracy
        axes[1, 0].plot(self.metrics['msg_acc'], label='Message Bit Accuracy')
        axes[1, 0].set_title('Message Extraction Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True)
        
        # Plot discriminator accuracy
        axes[1, 1].plot(self.metrics['d_acc'], label='Discriminator Accuracy')
        axes[1, 1].set_title('Discriminator Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"training_history_{timestamp}.png"
        plt.savefig(os.path.join(self.config['log_dir'], filename))
        logger.info(f"Training history plotted and saved to {filename}")
        
        plt.close(fig)

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN steganography model')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--img_size', type=int, help='Image size (square)')
    parser.add_argument('--message_length', type=int, help='Message length in bits')
    parser.add_argument('--capacity_factor', type=float, help='Capacity factor (0-1)')
    
    args = parser.parse_args()
    
    # Create config from args
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line args
    if args.dataset:
        config['dataset_path'] = args.dataset
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.img_size:
        config['img_shape'] = (args.img_size, args.img_size, 3)
    if args.message_length:
        config['message_length'] = args.message_length
    if args.capacity_factor is not None:
        config['capacity_factor'] = args.capacity_factor
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Train model
    trainer.train()