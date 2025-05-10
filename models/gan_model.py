import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np

class GANSteganography:
    def __init__(self, img_shape=(256, 256, 3)):
        self.img_shape = img_shape
        self.message_length = 100  # Panjang maksimal pesan dalam bit
        
        # Inisialisasi model
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.discriminator = self.build_discriminator()
        
        # Kompilasi model lengkap
        self.gan = self.build_gan()
        
    def build_encoder(self):
        """
        Membangun model encoder yang menyembunyikan pesan dalam gambar
        """
        # Input gambar cover
        cover_input = Input(shape=self.img_shape, name='cover_input')
        
        # Input pesan (binary)
        message_input = Input(shape=(self.message_length,), name='message_input')
        
        # Persiapkan pesan untuk digabungkan dengan gambar
        message_dense = layers.Dense(self.img_shape[0] * self.img_shape[1] // 16, activation='sigmoid')(message_input)
        message_reshape = layers.Reshape((self.img_shape[0] // 4, self.img_shape[1] // 4, 1))(message_dense)
        message_upsample = layers.UpSampling2D(size=(4, 4))(message_reshape)
        
        # Konvolusi gambar cover
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(cover_input)
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        
        # Gabungkan pesan dengan fitur gambar
        concat = layers.Concatenate()([x, message_upsample])
        
        # Generator output (gambar stego)
        x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(concat)
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        stego_output = layers.Conv2D(3, (3, 3), padding='same', activation='tanh', name='stego_output')(x)
        
        # Membatasi perubahan gambar
        stego_output = layers.Add()([cover_input, 0.05 * stego_output])  # Membatasi nilai perubahan pixel
        
        # Buat model encoder
        return Model(inputs=[cover_input, message_input], outputs=stego_output, name='encoder')
    
    def build_decoder(self):
        """
        Membangun model decoder yang mengekstrak pesan dari gambar stego
        """
        # Input gambar stego
        stego_input = Input(shape=self.img_shape, name='stego_input')
        
        # Ekstraksi fitur
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(stego_input)
        x = layers.Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (5, 5), strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (5, 5), strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (5, 5), strides=2, padding='same', activation='relu')(x)
        
        # Ekstraksi pesan
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        message_output = layers.Dense(self.message_length, activation='sigmoid', name='message_output')(x)
        
        # Buat model decoder
        return Model(inputs=stego_input, outputs=message_output, name='decoder')
    
    def build_discriminator(self):
        """
        Membangun model discriminator yang membedakan antara gambar asli dan stego
        """
        img_input = Input(shape=self.img_shape, name='disc_input')
        
        x = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(img_input)
        x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        disc_output = layers.Dense(1, activation='sigmoid', name='disc_output')(x)
        
        model = Model(inputs=img_input, outputs=disc_output, name='discriminator')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
            optimizer='adam',
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
    
    def train(self, cover_images, messages, epochs=10, batch_size=32):
        """
        Melatih model GAN
        """
        # Buat label untuk discriminator
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Latih discriminator
            idx = np.random.randint(0, cover_images.shape[0], batch_size)
            cover_batch = cover_images[idx]
            message_batch = messages[idx]
            
            # Hasilkan gambar stego
            stego_batch = self.encoder.predict([cover_batch, message_batch])
            
            # Latih discriminator
            d_loss_real = self.discriminator.train_on_batch(cover_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(stego_batch, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Latih GAN (encoder + decoder)
            g_loss = self.gan.train_on_batch(
                [cover_batch, message_batch],
                {
                    'stego_output': cover_batch,  # Target untuk gambar stego: mirip dengan gambar asli
                    'message_output': message_batch,  # Target untuk decoded message: sama dengan pesan asli
                    'disc_output': real_labels  # Target untuk discriminator: menipu bahwa gambar stego adalah asli
                }
            )
            
            # Tampilkan progres
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss[0]}")
    
    def save_models(self, path):
        """
        Menyimpan model-model ke disk
        """
        self.encoder.save(f"{path}/encoder.h5")
        self.decoder.save(f"{path}/decoder.h5")
        self.discriminator.save(f"{path}/discriminator.h5")
        
    def load_models(self, path):
        """
        Memuat model-model dari disk
        """
        self.encoder = tf.keras.models.load_model(f"{path}/encoder.h5")
        self.decoder = tf.keras.models.load_model(f"{path}/decoder.h5")
        self.discriminator = tf.keras.models.load_model(f"{path}/discriminator.h5")