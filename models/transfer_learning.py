"""
transfer_learning.py
Modul untuk transfer learning dan fine-tuning model steganografi
"""

import os
import numpy as np
import tensorflow as tf
import logging
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from models.enhanced_gan_model import EnhancedGANSteganography
from models.utils_pretrained import optimize_input_image, generate_model_registry_entry
from models.dataset import DatasetManager

# Konfigurasi logging
logger = logging.getLogger(__name__)

class TransferLearning:
    """
    Class untuk melakukan transfer learning dan fine-tuning
    pada model GAN steganografi
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi transfer learning
        
        Args:
            config: Dictionary konfigurasi atau None untuk defaults
        """
        self.default_config = {
            'models_dir': 'models/pretrained',
            'base_model': 'gan_basic',  # Model dasar untuk transfer learning
            'batch_size': 16,
            'epochs': 20,
            'learning_rate': 1e-4,
            'learning_rate_decay': 0.9,
            'validation_split': 0.2,
            'save_checkpoints': True,
            'checkpoint_dir': 'models/checkpoints',
            'log_dir': 'logs/transfer',
            'fine_tune_encoder': True,
            'fine_tune_decoder': True,
            'fine_tune_layers': 'all',  # 'all', 'last_n', 'none'
            'last_n_layers': 2  # Number of layers to fine-tune if using 'last_n'
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Create directories
        os.makedirs(self.config['models_dir'], exist_ok=True)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager()
        
        # Load base model
        self.base_model = None
        self.target_model = None
        
        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'encoder_loss': [],
            'decoder_loss': [],
            'psnr': [],
            'ssim': [],
            'bit_accuracy': []
        }
    
    def load_base_model(self, model_name=None):
        """
        Load base model for transfer learning
        
        Args:
            model_name: Name of the base model (None=use config)
            
        Returns:
            bool: Success or failure
        """
        # Use default if not specified
        if model_name is None:
            model_name = self.config['base_model']
            
        # Check if model directory exists
        model_dir = os.path.join(self.config['models_dir'], model_name)
        if not os.path.exists(model_dir):
            logger.error(f"Base model directory not found: {model_dir}")
            return False
            
        # Check if model files exist
        encoder_path = os.path.join(model_dir, "encoder.h5")
        decoder_path = os.path.join(model_dir, "decoder.h5")
        
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            logger.error(f"Model files not found in {model_dir}")
            return False
            
        # Load model info
        info_path = os.path.join(model_dir, "model_info.json")
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            return False
            
        # Create base model
        img_shape = tuple(model_info.get('input_shape', [256, 256, 3]))
        message_length = model_info.get('message_length', 100)
        capacity_factor = model_info.get('capacity_factor', 0.5)
        
        try:
            logger.info(f"Creating base model with shape {img_shape}, message length {message_length}")
            
            # Initialize model
            self.base_model = EnhancedGANSteganography(
                img_shape=img_shape,
                message_length=message_length,
                capacity_factor=capacity_factor
            )
            
            # Load pre-trained weights
            success = self.base_model.load_models(model_dir)
            
            if not success:
                logger.error(f"Failed to load model files from {model_dir}")
                return False
                
            logger.info(f"Successfully loaded base model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            return False
    
    def create_target_model(self, img_shape=None, message_length=None, capacity_factor=None):
        """
        Create target model with optional parameter changes
        
        Args:
            img_shape: New image shape (None=use base model)
            message_length: New message length (None=use base model)
            capacity_factor: New capacity factor (None=use base model)
            
        Returns:
            bool: Success or failure
        """
        if self.base_model is None:
            logger.error("Base model not loaded. Call load_base_model first.")
            return False
            
        # Use base model parameters if not specified
        if img_shape is None:
            img_shape = self.base_model.img_shape
            
        if message_length is None:
            message_length = self.base_model.message_length
            
        if capacity_factor is None:
            capacity_factor = self.base_model.capacity_factor
            
        try:
            logger.info(f"Creating target model with shape {img_shape}, message length {message_length}")
            
            # Initialize new model
            self.target_model = EnhancedGANSteganography(
                img_shape=img_shape,
                message_length=message_length,
                capacity_factor=capacity_factor
            )
            
            # If dimensions match, transfer weights from base model
            if img_shape == self.base_model.img_shape and message_length == self.base_model.message_length:
                logger.info("Transferring weights from base model")
                
                # Copy encoder weights
                for i, layer in enumerate(self.base_model.encoder.layers):
                    if i < len(self.target_model.encoder.layers):
                        try:
                            self.target_model.encoder.layers[i].set_weights(layer.get_weights())
                        except Exception as e:
                            logger.warning(f"Could not transfer weights for encoder layer {i}: {str(e)}")
                
                # Copy decoder weights
                for i, layer in enumerate(self.base_model.decoder.layers):
                    if i < len(self.target_model.decoder.layers):
                        try:
                            self.target_model.decoder.layers[i].set_weights(layer.get_weights())
                        except Exception as e:
                            logger.warning(f"Could not transfer weights for decoder layer {i}: {str(e)}")
                
                # Copy discriminator weights
                for i, layer in enumerate(self.base_model.discriminator.layers):
                    if i < len(self.target_model.discriminator.layers):
                        try:
                            self.target_model.discriminator.layers[i].set_weights(layer.get_weights())
                        except Exception as e:
                            logger.warning(f"Could not transfer weights for discriminator layer {i}: {str(e)}")
            else:
                logger.info("Dimensions don't match, using new model weights")
            
            # Set up fine-tuning configuration
            if self.config['fine_tune_layers'] == 'none':
                # Freeze all layers
                for layer in self.target_model.encoder.layers:
                    layer.trainable = False
                for layer in self.target_model.decoder.layers:
                    layer.trainable = False
                    
                logger.info("All layers frozen for transfer learning")
                
            elif self.config['fine_tune_layers'] == 'last_n':
                # Freeze all except last n layers
                n = self.config['last_n_layers']
                
                if self.config['fine_tune_encoder']:
                    for layer in self.target_model.encoder.layers[:-n]:
                        layer.trainable = False
                    for layer in self.target_model.encoder.layers[-n:]:
                        layer.trainable = True
                else:
                    for layer in self.target_model.encoder.layers:
                        layer.trainable = False
                
                if self.config['fine_tune_decoder']:
                    for layer in self.target_model.decoder.layers[:-n]:
                        layer.trainable = False
                    for layer in self.target_model.decoder.layers[-n:]:
                        layer.trainable = True
                else:
                    for layer in self.target_model.decoder.layers:
                        layer.trainable = False
                        
                logger.info(f"Last {n} layers unfrozen for fine-tuning")
                
            # otherwise all layers are trainable (default)
            
            # Recompile the models with new configuration
            self.target_model.discriminator.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.target_model.gan = self.target_model.build_gan()
            
            logger.info("Target model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating target model: {str(e)}")
            return False
    
    def prepare_dataset(self, dataset_path, max_images=1000):
        """
        Prepare dataset for transfer learning
        
        Args:
            dataset_path: Path to dataset directory
            max_images: Maximum number of images to use
            
        Returns:
            tuple: (train_dataset, val_dataset) as TensorFlow datasets
        """
        if self.target_model is None:
            logger.error("Target model not created. Call create_target_model first.")
            return None, None
            
        try:
            # Load dataset images
            logger.info(f"Loading dataset from {dataset_path}")
            images = self.dataset_manager.load_dataset(
                dataset_path, 
                max_images=max_images,
                img_shape=self.target_model.img_shape
            )
            
            if images is None or len(images) == 0:
                logger.error("Failed to load dataset or empty dataset")
                return None, None
                
            logger.info(f"Loaded {len(images)} images")
            
            # Split into training and validation
            val_split = self.config['validation_split']
            split_idx = int(len(images) * (1 - val_split))
            
            # Shuffle before split
            np.random.shuffle(images)
            
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            logger.info(f"Split dataset: {len(train_images)} training, {len(val_images)} validation")
            
            # Create TensorFlow datasets
            batch_size = self.config['batch_size']
            message_length = self.target_model.message_length
            
            # Define message generator function
            def generate_random_message(batch_size):
                return np.random.randint(0, 2, size=(batch_size, message_length))
            
            # Create training dataset
            train_ds = tf.data.Dataset.from_tensor_slices(train_images)
            train_ds = train_ds.shuffle(buffer_size=len(train_images))
            train_ds = train_ds.batch(batch_size)
            
            # Map function to create input pairs
            def map_fn(images_batch):
                batch_size = tf.shape(images_batch)[0]
                messages = tf.py_function(
                    lambda b: generate_random_message(b.numpy()), 
                    [batch_size], 
                    tf.int32
                )
                return [images_batch, messages], images_batch
            
            train_ds = train_ds.map(map_fn)
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            
            # Create validation dataset
            val_ds = tf.data.Dataset.from_tensor_slices(val_images)
            val_ds = val_ds.batch(batch_size)
            val_ds = val_ds.map(map_fn)
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
            
            logger.info("Datasets prepared successfully")
            return train_ds, val_ds
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            return None, None
    
    @tf.function
    def train_step(self, cover_images, messages):
        """
        Single training step for transfer learning
        
        Args:
            cover_images: Batch of cover images
            messages: Batch of binary messages
            
        Returns:
            dict: Training metrics
        """
        if self.target_model is None:
            logger.error("Target model not created. Call create_target_model first.")
            return None
            
        # Get optimizers
        encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        
        with tf.GradientTape(persistent=True) as tape:
            # Generate stego images
            stego_images = self.target_model.encoder([cover_images, messages], training=True)
            
            # Decode messages from stego
            decoded_messages = self.target_model.decoder(stego_images, training=True)
            
            # Calculate losses
            # Image quality loss (MSE between cover and stego)
            image_loss = tf.reduce_mean(tf.square(cover_images - stego_images))
            
            # Message decoding loss (BCE between original and decoded)
            message_loss = tf.keras.losses.BinaryCrossentropy()(messages, decoded_messages)
            
            # Combined encoder loss
            encoder_loss = 0.8 * image_loss + 0.2 * message_loss
            
            # Decoder loss is just the message loss
            decoder_loss = message_loss
            
        # Calculate gradients
        if self.config['fine_tune_encoder']:
            encoder_gradients = tape.gradient(
                encoder_loss, 
                self.target_model.encoder.trainable_variables
            )
            encoder_optimizer.apply_gradients(
                zip(encoder_gradients, self.target_model.encoder.trainable_variables)
            )
            
        if self.config['fine_tune_decoder']:
            decoder_gradients = tape.gradient(
                decoder_loss, 
                self.target_model.decoder.trainable_variables
            )
            decoder_optimizer.apply_gradients(
                zip(decoder_gradients, self.target_model.decoder.trainable_variables)
            )
            
        # Calculate metrics
        
        # PSNR
        psnr = tf.reduce_mean(tf.image.psnr(cover_images, stego_images, max_val=1.0))
        
        # SSIM
        ssim = tf.reduce_mean(tf.image.ssim(cover_images, stego_images, max_val=1.0))
        
        # Bit accuracy (percentage of correctly decoded bits)
        bit_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(decoded_messages), messages), tf.float32)
        )
        
        return {
            'encoder_loss': encoder_loss,
            'decoder_loss': decoder_loss,
            'image_loss': image_loss,
            'message_loss': message_loss,
            'psnr': psnr,
            'ssim': ssim,
            'bit_accuracy': bit_accuracy
        }
    
    def train(self, dataset_path=None, epochs=None, name=None, description=None):
        """
        Train model using transfer learning
        
        Args:
            dataset_path: Path to dataset directory (None=use config)
            epochs: Number of training epochs (None=use config)
            name: Name for the new model
            description: Description for the new model
            
        Returns:
            bool: Success or failure
        """
        # Load base model if not already loaded
        if self.base_model is None:
            logger.info("Base model not loaded, loading now")
            if not self.load_base_model():
                return False
                
        # Create target model if not already created
        if self.target_model is None:
            logger.info("Target model not created, creating now")
            if not self.create_target_model():
                return False
                
        # Use default values if not specified
        if dataset_path is None:
            dataset_path = 'datasets/sample'
            # If sample dataset doesn't exist, create it
            if not os.path.exists(dataset_path):
                logger.info("Sample dataset not found, creating it")
                self.dataset_manager.create_sample_dataset(num_images=50)
        
        if epochs is None:
            epochs = self.config['epochs']
            
        if name is None:
            name = f"transfer_model_{int(time.time())}"
            
        if description is None:
            description = f"Fine-tuned from {self.config['base_model']} using transfer learning"
            
        # Prepare dataset
        train_ds, val_ds = self.prepare_dataset(dataset_path)
        if train_ds is None or val_ds is None:
            return False
            
        # Set up tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.config['log_dir'], f"{name}_{timestamp}")
        summary_writer = tf.summary.create_file_writer(log_dir)
        
        # Set up checkpoint manager
        if self.config['save_checkpoints']:
            checkpoint_dir = os.path.join(self.config['checkpoint_dir'], name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = tf.train.Checkpoint(
                encoder=self.target_model.encoder,
                decoder=self.target_model.decoder
            )
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint,
                checkpoint_dir,
                max_to_keep=3
            )
        
        # Training loop
        logger.info(f"Starting transfer learning for {epochs} epochs")
        
        try:
            for epoch in range(epochs):
                start_time = time.time()
                
                # Initialize metrics
                epoch_metrics = {
                    'encoder_loss': [],
                    'decoder_loss': [],
                    'image_loss': [],
                    'message_loss': [],
                    'psnr': [],
                    'ssim': [],
                    'bit_accuracy': []
                }
                
                # Training loop
                with tqdm(total=len(train_ds), desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
                    for inputs, _ in train_ds:
                        cover_images, messages = inputs
                        
                        # Training step
                        metrics = self.train_step(cover_images, messages)
                        
                        # Update metrics
                        for key in epoch_metrics:
                            epoch_metrics[key].append(metrics[key].numpy())
                            
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'enc_loss': f"{metrics['encoder_loss'].numpy():.4f}",
                            'bit_acc': f"{metrics['bit_accuracy'].numpy():.4f}",
                            'psnr': f"{metrics['psnr'].numpy():.2f}"
                        })
                
                # Validation loop
                val_metrics = {
                    'encoder_loss': [],
                    'decoder_loss': [],
                    'image_loss': [],
                    'message_loss': [],
                    'psnr': [],
                    'ssim': [],
                    'bit_accuracy': []
                }
                
                with tqdm(total=len(val_ds), desc=f"Epoch {epoch+1}/{epochs} [Val]") as pbar:
                    for inputs, _ in val_ds:
                        cover_images, messages = inputs
                        
                        # Generate stego images
                        stego_images = self.target_model.encoder([cover_images, messages], training=False)
                        
                        # Decode messages
                        decoded_messages = self.target_model.decoder(stego_images, training=False)
                        
                        # Calculate losses
                        image_loss = tf.reduce_mean(tf.square(cover_images - stego_images))
                        message_loss = tf.keras.losses.BinaryCrossentropy()(messages, decoded_messages)
                        encoder_loss = 0.8 * image_loss + 0.2 * message_loss
                        decoder_loss = message_loss
                        
                        # Calculate metrics
                        psnr = tf.reduce_mean(tf.image.psnr(cover_images, stego_images, max_val=1.0))
                        ssim = tf.reduce_mean(tf.image.ssim(cover_images, stego_images, max_val=1.0))
                        bit_accuracy = tf.reduce_mean(
                            tf.cast(tf.equal(tf.round(decoded_messages), messages), tf.float32)
                        )
                        
                        # Update metrics
                        val_metrics['encoder_loss'].append(encoder_loss.numpy())
                        val_metrics['decoder_loss'].append(decoder_loss.numpy())
                        val_metrics['image_loss'].append(image_loss.numpy())
                        val_metrics['message_loss'].append(message_loss.numpy())
                        val_metrics['psnr'].append(psnr.numpy())
                        val_metrics['ssim'].append(ssim.numpy())
                        val_metrics['bit_accuracy'].append(bit_accuracy.numpy())
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'val_loss': f"{encoder_loss.numpy():.4f}",
                            'bit_acc': f"{bit_accuracy.numpy():.4f}",
                            'psnr': f"{psnr.numpy():.2f}"
                        })
                
                # Calculate epoch averages
                train_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
                val_avg = {k: np.mean(v) for k, v in val_metrics.items()}
                
                # Update history
                self.history['loss'].append(train_avg['encoder_loss'])
                self.history['val_loss'].append(val_avg['encoder_loss'])
                self.history['encoder_loss'].append(train_avg['encoder_loss'])
                self.history['decoder_loss'].append(train_avg['decoder_loss'])
                self.history['psnr'].append(train_avg['psnr'])
                self.history['ssim'].append(train_avg['ssim'])
                self.history['bit_accuracy'].append(train_avg['bit_accuracy'])
                
                # Log to tensorboard
                with summary_writer.as_default():
                    tf.summary.scalar('train/encoder_loss', train_avg['encoder_loss'], step=epoch)
                    tf.summary.scalar('train/decoder_loss', train_avg['decoder_loss'], step=epoch)
                    tf.summary.scalar('train/image_loss', train_avg['image_loss'], step=epoch)
                    tf.summary.scalar('train/message_loss', train_avg['message_loss'], step=epoch)
                    tf.summary.scalar('train/psnr', train_avg['psnr'], step=epoch)
                    tf.summary.scalar('train/ssim', train_avg['ssim'], step=epoch)
                    tf.summary.scalar('train/bit_accuracy', train_avg['bit_accuracy'], step=epoch)
                    
                    tf.summary.scalar('val/encoder_loss', val_avg['encoder_loss'], step=epoch)
                    tf.summary.scalar('val/decoder_loss', val_avg['decoder_loss'], step=epoch)
                    tf.summary.scalar('val/image_loss', val_avg['image_loss'], step=epoch)
                    tf.summary.scalar('val/message_loss', val_avg['message_loss'], step=epoch)
                    tf.summary.scalar('val/psnr', val_avg['psnr'], step=epoch)
                    tf.summary.scalar('val/ssim', val_avg['ssim'], step=epoch)
                    tf.summary.scalar('val/bit_accuracy', val_avg['bit_accuracy'], step=epoch)
                    
                    # Add sample images to tensorboard
                    if epoch % 5 == 0:
                        # Get a batch of samples
                        for inputs, _ in val_ds.take(1):
                            cover_images, messages = inputs
                            
                            # Generate stego images
                            stego_images = self.target_model.encoder([cover_images, messages], training=False)
                            
                            # Add images to tensorboard (max 4 samples)
                            n_samples = min(4, cover_images.shape[0])
                            for i in range(n_samples):
                                # Side by side comparison
                                comparison = tf.concat([cover_images[i], stego_images[i]], axis=1)
                                comparison = tf.expand_dims(comparison, 0)
                                
                                tf.summary.image(f"sample_{i}", comparison, step=epoch)
                
                # Save checkpoint
                if self.config['save_checkpoints']:
                    checkpoint_manager.save()
                    
                # Print summary
                epoch_time = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_avg['encoder_loss']:.4f}, "
                    f"Val Loss: {val_avg['encoder_loss']:.4f}, "
                    f"PSNR: {train_avg['psnr']:.2f}, "
                    f"Bit Acc: {train_avg['bit_accuracy']:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Learning rate decay
                if epoch > 0 and (epoch + 1) % 5 == 0:
                    new_lr = self.config['learning_rate'] * (self.config['learning_rate_decay'] ** (epoch // 5))
                    logger.info(f"Reducing learning rate to {new_lr:.6f}")
                    self.config['learning_rate'] = new_lr
            
            # Training completed, save final model
            model_dir = os.path.join(self.config['models_dir'], name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model files
            self.target_model.save_models(model_dir)
            
            # Save model info
            info = {
                'name': name,
                'description': description,
                'base_model': self.config['base_model'],
                'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_shape': list(self.target_model.img_shape),
                'message_length': self.target_model.message_length,
                'capacity_factor': float(self.target_model.capacity_factor),
                'training': {
                    'epochs': epochs,
                    'batch_size': self.config['batch_size'],
                    'learning_rate': self.config['learning_rate'],
                    'fine_tune_layers': self.config['fine_tune_layers'],
                    'dataset': dataset_path,
                    'validation_split': self.config['validation_split']
                },
                'performance': {
                    'final_loss': float(self.history['loss'][-1]),
                    'final_val_loss': float(self.history['val_loss'][-1]),
                    'final_psnr': float(self.history['psnr'][-1]),
                    'final_ssim': float(self.history['ssim'][-1]),
                    'final_bit_accuracy': float(self.history['bit_accuracy'][-1])
                }
            }
            
            with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
                json.dump(info, f, indent=2)
                
            # Create registry entry and metadata for sharing
            registry_entry = generate_model_registry_entry(
                name, description, model_dir, 
                self.target_model.img_shape, 
                self.target_model.message_length,
                self.target_model.capacity_factor
            )
            
            with open(os.path.join(model_dir, 'registry_entry.json'), 'w') as f:
                json.dump(registry_entry, f, indent=2)
                
            # Create README file
            with open(os.path.join(model_dir, 'README.md'), 'w') as f:
                f.write(f"# {name}\n\n")
                f.write(f"{description}\n\n")
                f.write("## Model Information\n\n")
                f.write(f"- Base Model: {self.config['base_model']}\n")
                f.write(f"- Created: {info['created_date']}\n")
                f.write(f"- Input Shape: {info['input_shape']}\n")
                f.write(f"- Message Length: {info['message_length']} bits\n")
                f.write(f"- Capacity Factor: {info['capacity_factor']}\n\n")
                f.write("## Performance\n\n")
                f.write(f"- PSNR: {info['performance']['final_psnr']:.2f} dB\n")
                f.write(f"- SSIM: {info['performance']['final_ssim']:.4f}\n")
                f.write(f"- Bit Accuracy: {info['performance']['final_bit_accuracy']:.4f}\n\n")
                f.write("## Training Details\n\n")
                f.write(f"- Epochs: {info['training']['epochs']}\n")
                f.write(f"- Batch Size: {info['training']['batch_size']}\n")
                f.write(f"- Fine-tuning: {info['training']['fine_tune_layers']}\n")
                
            # Plot training history
            self.plot_training_history(os.path.join(model_dir, 'training_history.png'))
            
            logger.info(f"Model saved to {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def evaluate_model(self, dataset_path=None, max_images=100):
        """
        Evaluate transfer learned model
        
        Args:
            dataset_path: Path to evaluation dataset
            max_images: Maximum number of images to use
            
        Returns:
            dict: Evaluation results
        """
        if self.target_model is None:
            logger.error("Target model not created. Call create_target_model first.")
            return None
            
        # Use default dataset if not specified
        if dataset_path is None:
            dataset_path = 'datasets/sample'
            # If sample dataset doesn't exist, create it
            if not os.path.exists(dataset_path):
                logger.info("Sample dataset not found, creating it")
                self.dataset_manager.create_sample_dataset(num_images=50)
        
        try:
            # Load dataset
            images = self.dataset_manager.load_dataset(
                dataset_path, 
                max_images=max_images,
                img_shape=self.target_model.img_shape
            )
            
            if images is None or len(images) == 0:
                logger.error("Failed to load evaluation dataset")
                return None
                
            # Create TensorFlow dataset
            batch_size = min(self.config['batch_size'], len(images))
            message_length = self.target_model.message_length
            
            # Create dataset
            ds = tf.data.Dataset.from_tensor_slices(images)
            ds = ds.batch(batch_size)
            
            # Evaluation metrics
            metrics = {
                'psnr': [],
                'ssim': [],
                'bit_accuracy': [],
                'message_loss': [],
                'image_loss': []
            }
            
            # Run evaluation
            logger.info(f"Evaluating model on {len(images)} images")
            
            for batch in tqdm(ds, desc="Evaluating"):
                # Generate random messages
                messages = np.random.randint(0, 2, size=(batch.shape[0], message_length))
                messages = tf.convert_to_tensor(messages, dtype=tf.float32)
                
                # Generate stego images
                stego_images = self.target_model.encoder([batch, messages], training=False)
                
                # Decode messages
                decoded_messages = self.target_model.decoder(stego_images, training=False)
                
                # Calculate metrics
                psnr_val = tf.reduce_mean(tf.image.psnr(batch, stego_images, max_val=1.0))
                ssim_val = tf.reduce_mean(tf.image.ssim(batch, stego_images, max_val=1.0))
                
                bit_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(decoded_messages), messages), tf.float32)
                )
                
                message_loss = tf.keras.losses.BinaryCrossentropy()(messages, decoded_messages)
                image_loss = tf.reduce_mean(tf.square(batch - stego_images))
                
                # Add to metrics
                metrics['psnr'].append(psnr_val.numpy())
                metrics['ssim'].append(ssim_val.numpy())
                metrics['bit_accuracy'].append(bit_accuracy.numpy())
                metrics['message_loss'].append(message_loss.numpy())
                metrics['image_loss'].append(image_loss.numpy())
            
            # Calculate averages
            results = {}
            for key, values in metrics.items():
                results[key] = float(np.mean(values))
                results[f"{key}_std"] = float(np.std(values))
                
            # Add some additional information
            results['num_images'] = len(images)
            results['message_length'] = message_length
            results['capacity_factor'] = float(self.target_model.capacity_factor)
            results['model_size'] = {
                'encoder_params': self.target_model.encoder.count_params(),
                'decoder_params': self.target_model.decoder.count_params(),
                'total_params': (
                    self.target_model.encoder.count_params() + 
                    self.target_model.decoder.count_params()
                )
            }
            
            logger.info(f"Evaluation results: PSNR={results['psnr']:.2f}, SSIM={results['ssim']:.4f}, Bit Acc={results['bit_accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return None
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot (None=show plot)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if not self.history['loss']:
            logger.warning("No training history to plot")
            return None
            
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axs[0, 0].plot(self.history['loss'], label='Training Loss')
        if self.history['val_loss']:
            axs[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # PSNR plot
        axs[0, 1].plot(self.history['psnr'], label='PSNR')
        axs[0, 1].set_title('Peak Signal-to-Noise Ratio')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('PSNR (dB)')
        axs[0, 1].grid(True)
        
        # SSIM plot
        axs[1, 0].plot(self.history['ssim'], label='SSIM')
        axs[1, 0].set_title('Structural Similarity')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('SSIM')
        axs[1, 0].set_ylim([0, 1])
        axs[1, 0].grid(True)
        
        # Bit accuracy plot
        axs[1, 1].plot(self.history['bit_accuracy'], label='Bit Accuracy')
        axs[1, 1].set_title('Message Bit Accuracy')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        return fig