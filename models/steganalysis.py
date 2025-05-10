"""
enhanced_gan_model.py
Implementasi arsitektur GAN yang ditingkatkan untuk steganografi
dengan mekanisme attention dan kapasitas dinamis
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import os
import logging

# Konfigurasi logging
logger = logging.getLogger(__name__)

class EnhancedGANSteganography:
    """
    Implementasi steganografi berbasis GAN dengan kemampuan yang ditingkatkan:
    - Attention mechanism
    - Kapasitas dinamis
    - Penanganan gambar multi-resolusi
    - Adaptasi untuk kapasitas tinggi
    """
    
    def __init__(self, img_shape=(256, 256, 3), message_length=100, capacity_factor=0.5):
        """
        Inisialisasi model GAN yang ditingkatkan
        
        Args:
            img_shape: Dimensi gambar input (tinggi, lebar, channel)
            message_length: Panjang pesan default dalam bit
            capacity_factor: Faktor kapasitas (0-1) yang mengontrol trade-off antara 
                            kapasitas dan imperceptibility
        """
        self.img_shape = img_shape
        self.message_length = message_length
        self.capacity_factor = capacity_factor
        
        # Inisialisasi model
        self.encoder = self.build_enhanced_encoder()
        self.decoder = self.build_enhanced_decoder()
        self.discriminator = self.build_discriminator()
        
        # Kompilasi model lengkap
        self.gan = self.build_gan()

    def build_enhanced_encoder(self):
        """
        Membangun model encoder yang ditingkatkan dengan attention mechanism
        """
        # Input gambar cover
        cover_input = Input(shape=self.img_shape, name='cover_input')
        
        # Input pesan (binary)
        message_input = Input(shape=(self.message_length,), name='message_input')
        
        # Encoder untuk gambar (extractor)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(cover_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Spatial features
        spatial_features = x
        
        # Encoder untuk pesan
        m = layers.Dense(512, activation='relu')(message_input)
        m = layers.Dense(1024, activation='relu')(m)
        m = layers.Dense(self.img_shape[0] * self.img_shape[1] // 16, activation='sigmoid')(m)
        m = layers.Reshape((self.img_shape[0] // 4, self.img_shape[1] // 4, 1))(m)
        
        # Attention mechanism - menggunakan pesan untuk menentukan area penyisipan
        attention = layers.Conv2D(128, (1, 1), padding='same', activation='sigmoid')(x)
        x = layers.multiply([x, attention])
        
        # Gabungkan pesan dengan fitur gambar melalui attention
        m_upsample = layers.UpSampling2D(size=(2, 2))(m)
        combined = layers.Concatenate()([x, m_upsample])
        
        # Decoder (generator output)
        x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(combined)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection dengan spatial features
        x = layers.Concatenate()([x, spatial_features])
        
        x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer - generate gambar stego
        raw_output = layers.Conv2D(3, (3, 3), padding='same', activation='tanh', name='raw_output')(x)
        
        # Residual connection untuk membatasi perubahan pada gambar cover
        # Parameter alpha mengontrol kekuatan penyisipan (modifiable)
        alpha = layers.Lambda(lambda x: x * self.capacity_factor)(raw_output)
        stego_output = layers.Add(name='stego_output')([cover_input, alpha])
        
        return Model(inputs=[cover_input, message_input], outputs=stego_output, name='encoder')

    def build_enhanced_decoder(self):
        """
        Membangun model decoder yang ditingkatkan dengan attention mechanism
        """
        # Input gambar stego
        stego_input = Input(shape=self.img_shape, name='stego_input')
        
        # Feature extraction blocks with residual connections
        # Block 1
        x = layers.Conv2D(64, (3, 3), padding='same')(stego_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        block1_output = layers.MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), padding='same')(block1_output)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        block2_output = layers.MaxPooling2D((2, 2))(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), padding='same')(block2_output)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        block3_output = layers.MaxPooling2D((2, 2))(x)
        
        # Attention block
        attention = layers.Conv2D(256, (1, 1), padding='same', activation='sigmoid', name='attention_map')(block3_output)
        x = layers.multiply([block3_output, attention])
        
        # Block 4 with skip connection
        x = layers.Conv2D(512, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global features
        x = layers.GlobalAveragePooling2D()(x)
        
        # Fully connected layers
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)  # Dropout untuk regularisasi
        x = layers.Dense(512, activation='relu')(x)
        
        # Output probabilitas untuk setiap bit pesan
        message_output = layers.Dense(self.message_length, activation='sigmoid', name='message_output')(x)
        
        return Model(inputs=stego_input, outputs=message_output, name='decoder')

    def build_discriminator(self):
        """
        Membangun model discriminator yang membedakan antara gambar asli dan stego
        """
        img_input = Input(shape=self.img_shape, name='disc_input')
        
        # Feature extraction with leaky ReLU
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(img_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Global features
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output probability
        disc_output = layers.Dense(1, activation='sigmoid', name='disc_output')(x)
        
        model = Model(inputs=img_input, outputs=disc_output, name='discriminator')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_gan(self):
        """
        Membangun model GAN lengkap (encoder + decoder + discriminator)
        """
        # Input
        cover_input = Input(shape=self.img_shape, name='cover_input')
        message_input = Input(shape=(self.message_length,), name='message_input')
        
        # Encode pesan ke dalam gambar
        stego_img = self.encoder([cover_input, message_input])
        
        # Ekstrak pesan dari gambar stego
        decoded_message = self.decoder(stego_img)
        
        # Klasifikasi gambar
        self.discriminator.trainable = False  # Nonaktifkan training untuk discriminator
        is_fake = self.discriminator(stego_img)
        
        # Buat model lengkap
        gan_model = Model(
            inputs=[cover_input, message_input], 
            outputs=[stego_img, decoded_message, is_fake]
        )
        
        # Kompilasi model
        gan_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
            loss={
                'stego_output': 'mean_squared_error',  # Loss untuk kualitas gambar
                'message_output': 'binary_crossentropy',  # Loss untuk akurasi pesan
                'disc_output': 'binary_crossentropy'  # Loss untuk menipu discriminator
            },
            loss_weights={
                'stego_output': 0.7,  # Prioritas kualitas gambar tinggi
                'message_output': 0.2,  # Prioritas akurasi pesan menengah
                'disc_output': 0.1  # Prioritas menipu discriminator rendah
            }
        )
        
        return gan_model
    
    def train_on_batch(self, cover_images, messages):
        """
        Latih model pada satu batch data
        
        Args:
            cover_images: Batch gambar cover
            messages: Batch pesan untuk disembunyikan
            
        Returns:
            Metrics hasil training
        """
        # Pastikan input dalam format yang benar
        if len(cover_images.shape) != 4:
            raise ValueError(f"Cover images harus berbentuk (batch_size, height, width, channels), tetapi diberikan {cover_images.shape}")
        
        if len(messages.shape) != 2:
            raise ValueError(f"Messages harus berbentuk (batch_size, message_length), tetapi diberikan {messages.shape}")
        
        batch_size = cover_images.shape[0]
        
        # 1. Train discriminator dengan gambar asli
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Generate gambar stego
        stego_images = self.encoder.predict([cover_images, messages])
        
        # Latih discriminator dengan gambar asli
        d_loss_real = self.discriminator.train_on_batch(cover_images, real_labels)
        
        # Latih discriminator dengan gambar stego
        d_loss_fake = self.discriminator.train_on_batch(stego_images, fake_labels)
        
        # Rata-rata loss discriminator
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # 2. Train generator (encoder + decoder)
        # Saat melatih generator, kita ingin discriminator menganggap gambar stego sebagai gambar asli
        g_loss = self.gan.train_on_batch(
            [cover_images, messages],
            {
                'stego_output': cover_images,  # Target untuk gambar stego: mirip dengan gambar asli
                'message_output': messages,  # Target untuk decoded message: sama dengan pesan asli
                'disc_output': real_labels  # Target untuk discriminator: menipu bahwa gambar stego adalah asli
            }
        )
        
        return {
            'discriminator_loss': d_loss[0],
            'discriminator_accuracy': d_loss[1],
            'generator_loss': g_loss[0],
            'image_loss': g_loss[1],
            'message_loss': g_loss[2],
            'adversarial_loss': g_loss[3]
        }
    
    def set_capacity_factor(self, factor):
        """
        Set kapasitas penyisipan (trade-off antara kapasitas dan imperceptibility)
        
        Args:
            factor: Nilai antara 0-1 (0: imperceptibility tinggi, kapasitas rendah;
                                     1: imperceptibility rendah, kapasitas tinggi)
        """
        if factor < 0 or factor > 1:
            raise ValueError("Capacity factor harus antara 0 dan 1")
        
        self.capacity_factor = factor
        
        # Rebuild model dengan kapasitas baru
        self.encoder = self.build_enhanced_encoder()
        self.gan = self.build_gan()
        
        return self.capacity_factor
    
    def save_models(self, path):
        """
        Menyimpan model-model ke disk
        
        Args:
            path: Path direktori untuk menyimpan model
        """
        # Pastikan direktori ada
        os.makedirs(path, exist_ok=True)
        
        try:
            self.encoder.save(f"{path}/encoder.h5")
            self.decoder.save(f"{path}/decoder.h5")
            self.discriminator.save(f"{path}/discriminator.h5")
            
            # Simpan konfigurasi tambahan
            config = {
                'img_shape': self.img_shape,
                'message_length': self.message_length,
                'capacity_factor': self.capacity_factor
            }
            
            with open(f"{path}/config.json", 'w') as f:
                import json
                json.dump(config, f)
                
            logger.info(f"Model berhasil disimpan ke {path}")
            return True
        except Exception as e:
            logger.error(f"Error saat menyimpan model: {str(e)}")
            return False
    
    def load_models(self, path):
        """
        Memuat model-model dari disk
        
        Args:
            path: Path direktori tempat model disimpan
            
        Returns:
            Boolean berhasil/gagal
        """
        try:
            # Load konfigurasi
            import json
            try:
                with open(f"{path}/config.json", 'r') as f:
                    config = json.load(f)
                    
                self.img_shape = tuple(config.get('img_shape', self.img_shape))
                self.message_length = config.get('message_length', self.message_length)
                self.capacity_factor = config.get('capacity_factor', self.capacity_factor)
            except FileNotFoundError:
                logger.warning(f"File konfigurasi tidak ditemukan di {path}/config.json")
            
            # Load model
            self.encoder = tf.keras.models.load_model(f"{path}/encoder.h5")
            self.decoder = tf.keras.models.load_model(f"{path}/decoder.h5")
            self.discriminator = tf.keras.models.load_model(f"{path}/discriminator.h5")
            
            # Rebuild GAN
            self.gan = self.build_gan()
            
            logger.info(f"Model berhasil dimuat dari {path}")
            return True
        except Exception as e:
            logger.error(f"Error saat memuat model: {str(e)}")
            return False
            
    def encode_message(self, cover_image, message):
        """
        Encode pesan ke dalam gambar menggunakan model GAN
        
        Args:
            cover_image: Gambar cover (numpy array)
            message: Pesan dalam bentuk bit (numpy array)
            
        Returns:
            Stego image (numpy array)
        """
        # Normalisasi input
        if cover_image.dtype != np.float32 or cover_image.max() > 1.0:
            cover_image = cover_image.astype(np.float32) / 255.0
            
        # Reshape jika diperlukan
        if len(cover_image.shape) == 3:  # single image
            cover_image = np.expand_dims(cover_image, axis=0)
            
        # Convert message to binary if it's a string
        if isinstance(message, str):
            binary_message = []
            for char in message:
                # Convert char to 8-bit binary
                binary_message.extend([int(bit) for bit in format(ord(char), '08b')])
            message = np.array(binary_message)
        
        # Pastikan panjang pesan sesuai
        if len(message) > self.message_length:
            message = message[:self.message_length]
        elif len(message) < self.message_length:
            message = np.pad(message, (0, self.message_length - len(message)))
            
        # Reshape message
        message = np.expand_dims(message, axis=0)
        
        # Generate stego image
        stego_image = self.encoder.predict([cover_image, message])[0]
        
        return stego_image
    
    def decode_message(self, stego_image):
        """
        Decode pesan dari gambar stego menggunakan model GAN
        
        Args:
            stego_image: Gambar stego (numpy array)
            
        Returns:
            Pesan dalam bentuk bit (numpy array)
        """
        # Normalisasi input
        if stego_image.dtype != np.float32 or stego_image.max() > 1.0:
            stego_image = stego_image.astype(np.float32) / 255.0
            
        # Reshape jika diperlukan
        if len(stego_image.shape) == 3:  # single image
            stego_image = np.expand_dims(stego_image, axis=0)
            
        # Ekstrak pesan
        message_prob = self.decoder.predict(stego_image)[0]
        
        # Konversi probabilitas ke bit (threshold = 0.5)
        message_bits = (message_prob > 0.5).astype(int)
        
        return message_bits