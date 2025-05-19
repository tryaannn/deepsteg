from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
import os
import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
import uuid
import time
import shutil
import json
import hashlib
import datetime
import sqlite3
from werkzeug.utils import secure_filename

# Import DeepSteg model modules
from models.utils import preprocess_image, postprocess_image, calculate_metrics
from models.enhanced_encoder import encode_message
from models.enhanced_decoder import decode_message
from models.crypto import calculate_password_strength

# Import pre-trained model functionality
from models.pretrained_model_manager import PretrainedModelManager
from models.pretrained_detector import PretrainedDetector
from models.utils_pretrained import optimize_input_image, calculate_embedding_capacity

# Konfigurasi aplikasi Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'deepsteg_enhanced_secret_key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_IMAGE_SIZE'] = 1920  # Maksimum dimensi gambar
app.config['SESSION_TYPE'] = 'filesystem'
app.config['DB_PATH'] = os.path.join(os.getcwd(), 'deepsteg.db')

# Tambahan konfigurasi untuk Pre-trained Models
app.config['MODELS_DIR'] = os.path.join(os.getcwd(), 'models', 'pretrained')
app.config['DETECTORS_DIR'] = os.path.join(os.getcwd(), 'models', 'saved', 'steganalysis')
app.config['DEFAULT_MODEL'] = 'gan_basic'
app.config['DEFAULT_DETECTOR'] = 'cnn_basic'

# Konfigurasi logging
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'deepsteg.log')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pastikan folder uploads ada dan dapat diakses
uploads_dir = app.config['UPLOAD_FOLDER']
os.makedirs(uploads_dir, exist_ok=True)

# Pastikan folder models ada
os.makedirs(app.config['MODELS_DIR'], exist_ok=True)
os.makedirs(app.config['DETECTORS_DIR'], exist_ok=True)

# Inisialisasi model manager dan detector
model_manager = PretrainedModelManager({
    'models_dir': app.config['MODELS_DIR']
})

detector = PretrainedDetector({
    'detectors_dir': app.config['DETECTORS_DIR']
})

# Inisialisasi database
def init_db():
    """Inisialisasi database SQLite untuk statistik"""
    try:
        conn = sqlite3.connect(app.config['DB_PATH'])
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            operation TEXT,
            image_size INTEGER,
            message_length INTEGER,
            psnr REAL,
            ssim REAL,
            encrypted BOOLEAN,
            compressed BOOLEAN,
            user_agent TEXT,
            ip_address TEXT,
            model_used TEXT
        )
        ''')
        
        # Tambahkan tabel untuk analisis
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_hash TEXT,
            is_stego BOOLEAN,
            confidence REAL,
            detection_score REAL,
            estimated_payload REAL,
            detector_used TEXT,
            user_agent TEXT,
            ip_address TEXT
        )
        ''')
        
        # Tambahkan tabel untuk model usage
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            operation TEXT,
            success BOOLEAN,
            execution_time REAL,
            user_agent TEXT,
            ip_address TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

# Fungsi untuk menyimpan statistik
def save_stats(operation, stats_data):
    """Menyimpan statistik ke database"""
    try:
        conn = sqlite3.connect(app.config['DB_PATH'])
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO stats (operation, image_size, message_length, psnr, ssim, encrypted, compressed, user_agent, ip_address, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                operation,
                stats_data.get('image_size', 0),
                stats_data.get('message_length', 0),
                stats_data.get('psnr', 0.0),
                stats_data.get('ssim', 0.0),
                stats_data.get('encrypted', False),
                stats_data.get('compressed', False),
                stats_data.get('user_agent', ''),
                stats_data.get('ip_address', ''),
                stats_data.get('model_used', None)
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving stats: {str(e)}")

# Fungsi untuk menyimpan hasil analisis
def save_analysis(analysis_data):
    """Menyimpan hasil analisis steganography ke database"""
    try:
        conn = sqlite3.connect(app.config['DB_PATH'])
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analysis (image_hash, is_stego, confidence, detection_score, estimated_payload, detector_used, user_agent, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis_data.get('image_hash', ''),
                analysis_data.get('is_stego', False),
                analysis_data.get('confidence', 0.0),
                analysis_data.get('detection_score', 0.0),
                analysis_data.get('estimated_payload', 0.0),
                analysis_data.get('detector_used', ''),
                analysis_data.get('user_agent', ''),
                analysis_data.get('ip_address', '')
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")

# Fungsi untuk menyimpan model usage
def save_model_usage(usage_data):
    """Menyimpan penggunaan model ke database"""
    try:
        conn = sqlite3.connect(app.config['DB_PATH'])
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO model_usage (model_name, operation, success, execution_time, user_agent, ip_address) VALUES (?, ?, ?, ?, ?, ?)",
            (
                usage_data.get('model_name', ''),
                usage_data.get('operation', ''),
                usage_data.get('success', False),
                usage_data.get('execution_time', 0.0),
                usage_data.get('user_agent', ''),
                usage_data.get('ip_address', '')
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving model usage: {str(e)}")

# Fungsi untuk mendapatkan statistik
def get_stats():
    """Mendapatkan statistik dari database"""
    try:
        conn = sqlite3.connect(app.config['DB_PATH'])
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get total operations
        cursor.execute("SELECT COUNT(*) as count FROM stats")
        total_operations = cursor.fetchone()['count']
        
        # Get encode operations
        cursor.execute("SELECT COUNT(*) as count FROM stats WHERE operation = 'encode'")
        encode_operations = cursor.fetchone()['count']
        
        # Get decode operations
        cursor.execute("SELECT COUNT(*) as count FROM stats WHERE operation = 'decode'")
        decode_operations = cursor.fetchone()['count']
        
        # Get average PSNR and SSIM
        cursor.execute("SELECT AVG(psnr) as avg_psnr, AVG(ssim) as avg_ssim FROM stats WHERE operation = 'encode'")
        metrics = cursor.fetchone()
        avg_psnr = metrics['avg_psnr'] if metrics['avg_psnr'] else 0
        avg_ssim = metrics['avg_ssim'] if metrics['avg_ssim'] else 0
        
        # Get percentage of encrypted messages
        cursor.execute("SELECT COUNT(*) as count FROM stats WHERE encrypted = 1")
        encrypted_count = cursor.fetchone()['count']
        encrypted_percentage = (encrypted_count / total_operations * 100) if total_operations > 0 else 0
        
        # Get percentage of compressed messages
        cursor.execute("SELECT COUNT(*) as count FROM stats WHERE compressed = 1")
        compressed_count = cursor.fetchone()['count']
        compressed_percentage = (compressed_count / total_operations * 100) if total_operations > 0 else 0
        
        # Get operations over time (last 30 days)
        cursor.execute("""
            SELECT strftime('%Y-%m-%d', timestamp) as date, COUNT(*) as count 
            FROM stats 
            WHERE timestamp >= date('now', '-30 days') 
            GROUP BY date 
            ORDER BY date
        """)
        time_data = cursor.fetchall()
        operations_over_time = {row['date']: row['count'] for row in time_data}
        
        # Get model usage stats
        cursor.execute("""
            SELECT model_used, COUNT(*) as count 
            FROM stats 
            WHERE model_used IS NOT NULL 
            GROUP BY model_used 
            ORDER BY count DESC
        """)
        model_usage = {row['model_used']: row['count'] for row in cursor.fetchall()}
        
        # Get steganalysis stats
        cursor.execute("SELECT COUNT(*) as count FROM analysis")
        analysis_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM analysis WHERE is_stego = 1")
        stego_detected_count = cursor.fetchone()['count']
        stego_detected_percentage = (stego_detected_count / analysis_count * 100) if analysis_count > 0 else 0
        
        conn.close()
        
        return {
            'total_operations': total_operations,
            'encode_operations': encode_operations,
            'decode_operations': decode_operations,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'encrypted_percentage': encrypted_percentage,
            'compressed_percentage': compressed_percentage,
            'operations_over_time': operations_over_time,
            'model_usage': model_usage,
            'analysis_count': analysis_count,
            'stego_detected_percentage': stego_detected_percentage
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {
            'total_operations': 0,
            'encode_operations': 0,
            'decode_operations': 0,
            'avg_psnr': 0,
            'avg_ssim': 0,
            'encrypted_percentage': 0,
            'compressed_percentage': 0,
            'operations_over_time': {},
            'model_usage': {},
            'analysis_count': 0,
            'stego_detected_percentage': 0
        }

# Bersihkan file lama
def cleanup_old_files():
    """Membersihkan file hasil yang lebih dari 1 jam"""
    try:
        now = time.time()
        for filename in os.listdir(uploads_dir):
            if filename == '.gitkeep':
                continue
            file_path = os.path.join(uploads_dir, filename)
            # Jika file lebih dari 1 jam, hapus
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < now - 3600:
                os.remove(file_path)
                logger.info(f"Menghapus file lama: {filename}")
    except Exception as e:
        logger.error(f"Error saat membersihkan file lama: {str(e)}")

# Validasi ekstensi file
def allowed_file(filename):
    """Memeriksa apakah ekstensi file diperbolehkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Hash file untuk mencegah upload berulang
def hash_file(file_path):
    """Generate hash dari file untuk identifikasi"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Routes
@app.route('/')
def index():
    # Bersihkan file lama saat halaman utama diakses
    cleanup_old_files()
    return render_template('index.html')

@app.route('/encode')
def encode_page():
    # Get list of available models
    models = model_manager.list_available_models()
    return render_template('encode.html', models=models)

@app.route('/decode')
def decode_page():
    # Get list of available models
    models = model_manager.list_available_models()
    return render_template('decode.html', models=models)

@app.route('/analyze')
def analyze_page():
    # Get list of available detectors
    detectors = detector.list_available_detectors()
    return render_template('analyze.html', detectors=detectors)

@app.route('/models')
def models_page():
    # Get list of available models
    models = model_manager.list_available_models()
    detectors = detector.list_available_detectors()
    return render_template('models.html', models=models, detectors=detectors)

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/api/check-password', methods=['POST'])
def check_password():
    """API endpoint untuk memeriksa kekuatan password"""
    data = request.get_json()
    if not data or 'password' not in data:
        return jsonify({'error': 'Password tidak ditemukan dalam request'}), 400
    
    password = data['password']
    
    # Evaluasi kekuatan password
    result = calculate_password_strength(password)
    
    return jsonify(result)

@app.route('/api/encode', methods=['POST'])
def encode():
    """API endpoint untuk encoding pesan ke dalam gambar"""
    try:
        # Dapatkan file gambar
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
        
        image_file = request.files['image']
        
        # Validasi file gambar
        if image_file.filename == '':
            return jsonify({'error': 'Tidak ada file gambar yang dipilih'}), 400
        
        # Verifikasi ekstensi file
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Format file tidak valid. Format yang didukung: PNG, JPG, BMP'
            }), 400
        
        # Dapatkan pesan
        message = request.form.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Tidak ada pesan yang diberikan'}), 400
        
        # Dapatkan password (opsional)
        use_encryption = request.form.get('use_encryption', 'false') == 'true'
        password = None
        if use_encryption:
            password = request.form.get('password', '')
            if not password:
                return jsonify({'error': 'Password diperlukan jika enkripsi diaktifkan'}), 400
        
        # Dapatkan level kompresi (opsional)
        compression_level = None  # Default: tanpa kompresi
        use_compression = request.form.get('use_compression', 'false') == 'true'
        if use_compression:
            try:
                compression_level = int(request.form.get('compression_level', '6'))
                if not 0 <= compression_level <= 9:
                    compression_level = 6  # Reset ke default jika di luar rentang
            except ValueError:
                compression_level = 6  # Default jika input tidak valid
        
        # Generate unique filename untuk mencegah overwrite
        original_filename = secure_filename(image_file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        result_filename = f"stego_{uuid.uuid4().hex}.png"  # Selalu simpan hasil sebagai PNG
        
        # Simpan file upload
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(upload_path)
        
        # Cek apakah file yang sama sudah pernah diupload
        file_hash = hash_file(upload_path)
        
        # Periksa kapasitas server
        total_uploads = sum(os.path.getsize(os.path.join(uploads_dir, f)) for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f)))
        max_upload_dir_size = 500 * 1024 * 1024  # 500 MB
        if total_uploads > max_upload_dir_size:
            # Hapus semua file kecuali yang baru diupload
            for filename in os.listdir(uploads_dir):
                if filename != unique_filename and filename != '.gitkeep':
                    try:
                        os.remove(os.path.join(uploads_dir, filename))
                    except:
                        pass
            logger.warning("Upload directory cleaned due to reaching capacity limit")
        
        try:
            # Baca dan preprocessing gambar
            img = Image.open(upload_path)
            
            # Konversi gambar ke array numpy
            img_array = np.array(img)
            
            # Logging untuk debugging
            logger.info(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            # Pastikan gambar adalah RGB
            if len(img_array.shape) < 3:
                # Konversi gambar grayscale ke RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                logger.info(f"Converted grayscale to RGB, new shape: {img_array.shape}")
            elif img_array.shape[2] == 4:
                # Konversi RGBA ke RGB
                img_array = img_array[:, :, :3]
                logger.info(f"Converted RGBA to RGB, new shape: {img_array.shape}")
            elif img_array.shape[2] < 3:
                return jsonify({'error': 'Format gambar tidak valid, tidak dapat dikonversi ke RGB'}), 400
            
            # Preprocessing gambar
            processed_img = preprocess_image(img_array)
            
            # Verifikasi ukuran pesan vs. kapasitas gambar
            img_capacity_bits = processed_img.shape[0] * processed_img.shape[1] * 3
            msg_size_bits = len(message) * 8
            capacity_percentage = (msg_size_bits / img_capacity_bits) * 100
            
            logger.info(f"Message size: {msg_size_bits} bits, Image capacity: {img_capacity_bits} bits, Usage: {capacity_percentage:.2f}%")
            
            if capacity_percentage > 90:
                logger.warning(f"Message exceeds 90% of image capacity: {capacity_percentage:.2f}%")
            
            # Encode pesan ke dalam gambar dengan enkripsi jika diaktifkan
            stego_img, metrics = encode_message(processed_img, message, password if use_encryption else None, compression_level)
            
            # Postprocessing gambar
            output_img = postprocess_image(stego_img)
            
            # Simpan hasil
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            
            # Pastikan output_img dalam RGB, kemudian konversi ke BGR untuk OpenCV
            if output_img.shape[2] == 3:  # Jika RGB
                cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            else:  # Fallback jika bentuk tidak sesuai
                Image.fromarray(output_img).save(output_path)
            
            logger.info(f"Saved stego image to: {output_path}")
            
            # Periksa apakah file berhasil disimpan
            if not os.path.exists(output_path):
                logger.error(f"Failed to save stego image to: {output_path}")
                return jsonify({'error': 'Gagal menyimpan gambar hasil'}), 500
            
            # Dapatkan ukuran file
            file_size = os.path.getsize(output_path)
            
            # Konversi untuk ditampilkan di frontend
            # Baca file dari disk untuk memastikan konten yang benar
            with open(output_path, 'rb') as f:
                img_data = f.read()
                img_str = base64.b64encode(img_data).decode('utf-8')
            
            # Tambahkan metrik tambahan
            metrics['capacity_used'] = float(capacity_percentage)
            metrics['file_size'] = file_size
            metrics['encrypted'] = use_encryption
            metrics['compressed'] = use_compression
            
            # Simpan gambar asli untuk perbandingan
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{result_filename}")
            cv2.imwrite(original_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            # Baca gambar asli untuk ditampilkan
            with open(original_path, 'rb') as f:
                original_data = f.read()
                original_str = base64.b64encode(original_data).decode('utf-8')
            
            # Simpan statistik
            stats_data = {
                'image_size': file_size,
                'message_length': len(message),
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'encrypted': use_encryption,
                'compressed': use_compression,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr,
                'model_used': 'standard'
            }
            save_stats('encode', stats_data)
            
            return jsonify({
                'status': 'success',
                'image': img_str,
                'original_image': original_str,
                'metrics': metrics,
                'filename': result_filename
            })
            
        except Exception as e:
            logger.error(f"Error in encode: {str(e)}", exc_info=True)
            # Hapus file upload yang tidak digunakan
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except:
                pass
            return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in encode: {str(e)}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan yang tidak diketahui'}), 500

@app.route('/api/decode', methods=['POST'])
def decode():
    """API endpoint untuk decoding pesan dari gambar"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
        
        image_file = request.files['image']
        
        # Validasi file gambar
        if image_file.filename == '':
            return jsonify({'error': 'Tidak ada file gambar yang dipilih'}), 400
        
        # Verifikasi ekstensi file
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Format file tidak valid. Format yang didukung: PNG, JPG, BMP'
            }), 400
        
        # Dapatkan password (opsional)
        password = request.form.get('password', '')
        
        # Generate unique filename
        unique_filename = f"decode_{uuid.uuid4().hex}.{image_file.filename.rsplit('.', 1)[1].lower()}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(upload_path)
        
        try:
            # Baca dan preprocessing gambar
            img = Image.open(upload_path)
            img_array = np.array(img)
            
            # Logging untuk debugging
            logger.info(f"Decode image shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            # Pastikan gambar adalah RGB
            if len(img_array.shape) < 3:
                # Konversi gambar grayscale ke RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                logger.info(f"Converted grayscale to RGB, new shape: {img_array.shape}")
            elif img_array.shape[2] == 4:
                # Konversi RGBA ke RGB
                img_array = img_array[:, :, :3]
                logger.info(f"Converted RGBA to RGB, new shape: {img_array.shape}")
            elif img_array.shape[2] < 3:
                return jsonify({'error': 'Format gambar tidak valid, tidak dapat dikonversi ke RGB'}), 400
            
            # Preprocessing gambar
            processed_img = preprocess_image(img_array)
            
            # Decode pesan dari gambar
            message, metadata = decode_message(processed_img, password if password else None)
            
            # Handle kasus khusus untuk pesan terenkripsi
            if message == "ENCRYPTED_MESSAGE":
                if metadata['password_correct'] is False:
                    return jsonify({
                        'status': 'error',
                        'error': 'Password salah untuk pesan terenkripsi'
                    }), 400
                else:
                    return jsonify({
                        'status': 'password_required',
                        'message': 'Pesan ini terenkripsi. Masukkan password untuk mendekripsi.'
                    })
            
            # Handle kasus khusus untuk error kompresi
            if message == "COMPRESSION_ERROR":
                return jsonify({
                    'status': 'error',
                    'error': 'Gagal mendekompresi pesan. File mungkin rusak.'
                }), 400
            
            if not message:
                return jsonify({
                    'status': 'warning',
                    'message': 'Tidak ada pesan yang ditemukan atau pesan kosong.'
                })
            
            # Simpan statistik
            stats_data = {
                'image_size': os.path.getsize(upload_path),
                'message_length': len(message),
                'psnr': 0.0,  # Not applicable for decode
                'ssim': 0.0,  # Not applicable for decode
                'encrypted': metadata.get('is_encrypted', False),
                'compressed': metadata.get('is_compressed', False),
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr,
                'model_used': 'standard'
            }
            save_stats('decode', stats_data)
            
            return jsonify({
                'status': 'success',
                'message': message,
                'metadata': metadata
            })
        
        except Exception as e:
            logger.error(f"Error in decode: {str(e)}", exc_info=True)
            # Hapus file upload yang tidak digunakan
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except:
                pass
            return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in decode: {str(e)}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan yang tidak diketahui'}), 500

@app.route('/api/encode-with-model', methods=['POST'])
def encode_with_model():
    """API endpoint untuk encoding pesan ke dalam gambar menggunakan pre-trained model"""
    try:
        # Dapatkan file gambar
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
        
        image_file = request.files['image']
        
        # Validasi file gambar
        if image_file.filename == '':
            return jsonify({'error': 'Tidak ada file gambar yang dipilih'}), 400
        
        # Verifikasi ekstensi file
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Format file tidak valid. Format yang didukung: PNG, JPG, BMP'
            }), 400
        
        # Dapatkan pesan
        message = request.form.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Tidak ada pesan yang diberikan'}), 400
        
        # Model yang akan digunakan
        model_name = request.form.get('model', app.config['DEFAULT_MODEL'])
        
        # Generate unique filename untuk mencegah overwrite
        original_filename = secure_filename(image_file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        result_filename = f"stego_{uuid.uuid4().hex}.png"  # Selalu simpan hasil sebagai PNG
        
        # Simpan file upload
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(upload_path)
        
        try:
            # Baca dan preprocessing gambar
            img = Image.open(upload_path)
            img_array = np.array(img)
            
            # Pastikan gambar adalah RGB
            if len(img_array.shape) < 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Preprocessing gambar
            processed_img = preprocess_image(img_array)
            
            # Encode pesan menggunakan pre-trained model
            start_time = time.time()
            stego_img, metrics = model_manager.encode_with_model(
                model_name, processed_img, message, 
                options={'normalize': False}  # Gambar sudah preprocessing
            )
            execution_time = time.time() - start_time
            
            if stego_img is None:
                return jsonify({'error': f"Gagal mengenkripsi pesan: {metrics.get('error', 'Unknown error')}"}), 500
            
            # Simpan hasil
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            
            # Pastikan output_img dalam RGB, kemudian konversi ke BGR untuk OpenCV
            if stego_img.shape[2] == 3:  # Jika RGB
                cv2.imwrite(output_path, cv2.cvtColor((stego_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            else:  # Fallback jika bentuk tidak sesuai
                Image.fromarray((stego_img * 255).astype(np.uint8)).save(output_path)
            
            # Periksa apakah file berhasil disimpan
            if not os.path.exists(output_path):
                logger.error(f"Failed to save stego image to: {output_path}")
                return jsonify({'error': 'Gagal menyimpan gambar hasil'}), 500
            
            # Dapatkan ukuran file
            file_size = os.path.getsize(output_path)
            
            # Konversi untuk ditampilkan di frontend
            with open(output_path, 'rb') as f:
                img_data = f.read()
                img_str = base64.b64encode(img_data).decode('utf-8')
            
            # Tambahkan model info ke metrics
            metrics['model_used'] = model_name
            model_info = model_manager.get_model_info(model_name)
            if model_info:
                metrics['model_info'] = {
                    'description': model_info.get('description', ''),
                    'message_length': model_info.get('message_length', 0),
                    'capacity_factor': model_info.get('capacity_factor', 0)
                }
            
            # Simpan gambar asli untuk perbandingan
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{result_filename}")
            cv2.imwrite(original_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            # Baca gambar asli untuk ditampilkan
            with open(original_path, 'rb') as f:
                original_data = f.read()
                original_str = base64.b64encode(original_data).decode('utf-8')
            
            # Simpan statistik
            stats_data = {
                'image_size': file_size,
                'message_length': len(message),
                'psnr': metrics.get('psnr', 0),
                'ssim': metrics.get('ssim', 0),
                'encrypted': False,
                'compressed': False,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr,
                'model_used': model_name
            }
            save_stats('encode', stats_data)
            
            # Simpan model usage
            usage_data = {
                'model_name': model_name,
                'operation': 'encode',
                'success': True,
                'execution_time': execution_time,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_model_usage(usage_data)
            
            return jsonify({
                'status': 'success',
                'image': img_str,
                'original_image': original_str,
                'metrics': metrics,
                'filename': result_filename
            })
            
        except Exception as e:
            logger.error(f"Error in encode with model: {str(e)}", exc_info=True)
            # Hapus file upload yang tidak digunakan
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except:
                pass
            
            # Simpan model usage (failure)
            usage_data = {
                'model_name': model_name,
                'operation': 'encode',
                'success': False,
                'execution_time': 0,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_model_usage(usage_data)
            
            return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in encode with model: {str(e)}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan yang tidak diketahui'}), 500

@app.route('/api/decode-with-model', methods=['POST'])
def decode_with_model():
    """API endpoint untuk decoding pesan dari gambar menggunakan pre-trained model"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
        
        image_file = request.files['image']
        
        # Validasi file gambar
        if image_file.filename == '':
            return jsonify({'error': 'Tidak ada file gambar yang dipilih'}), 400
        
        # Verifikasi ekstensi file
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Format file tidak valid. Format yang didukung: PNG, JPG, BMP'
            }), 400
        
        # Model yang akan digunakan
        model_name = request.form.get('model', app.config['DEFAULT_MODEL'])
        
        # Generate unique filename
        unique_filename = f"decode_{uuid.uuid4().hex}.{image_file.filename.rsplit('.', 1)[1].lower()}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(upload_path)
        
        try:
            # Baca dan preprocessing gambar
            img = Image.open(upload_path)
            img_array = np.array(img)
            
            # Pastikan gambar adalah RGB
            if len(img_array.shape) < 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Preprocessing gambar
            processed_img = preprocess_image(img_array)
            
            # Decode pesan menggunakan pre-trained model
            start_time = time.time()
            message, metadata = model_manager.decode_with_model(
                model_name, processed_img,
                options={'normalize': False}  # Gambar sudah preprocessing
            )
            execution_time = time.time() - start_time
            
            if message is None:
                return jsonify({
                    'status': 'error',
                    'error': metadata.get('error', 'Unknown error')
                }), 400
            
            # Tambahkan model info ke metadata
            metadata['model_used'] = model_name
            model_info = model_manager.get_model_info(model_name)
            if model_info:
                metadata['model_info'] = {
                    'description': model_info.get('description', ''),
                    'message_length': model_info.get('message_length', 0)
                }
            
            # Simpan statistik
            stats_data = {
                'image_size': os.path.getsize(upload_path),
                'message_length': len(message) if isinstance(message, str) else len(message) // 8,
                'psnr': 0.0,  # Not applicable for decode
                'ssim': 0.0,  # Not applicable for decode
                'encrypted': False,
                'compressed': False,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr,
                'model_used': model_name
            }
            save_stats('decode', stats_data)
            
            # Simpan model usage
            usage_data = {
                'model_name': model_name,
                'operation': 'decode',
                'success': True,
                'execution_time': execution_time,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_model_usage(usage_data)
            
            return jsonify({
                'status': 'success',
                'message': message if isinstance(message, str) else message.tolist(),
                'metadata': metadata
            })
        
        except Exception as e:
            logger.error(f"Error in decode with model: {str(e)}", exc_info=True)
            # Hapus file upload yang tidak digunakan
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except:
                pass
            
            # Simpan model usage (failure)
            usage_data = {
                'model_name': model_name,
                'operation': 'decode',
                'success': False,
                'execution_time': 0,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_model_usage(usage_data)
            
            return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in decode with model: {str(e)}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan yang tidak diketahui'}), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """API endpoint untuk menganalisis gambar dengan steganalysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
        
        image_file = request.files['image']
        
        # Validasi file gambar
        if image_file.filename == '':
            return jsonify({'error': 'Tidak ada file gambar yang dipilih'}), 400
        
        # Verifikasi ekstensi file
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Format file tidak valid. Format yang didukung: PNG, JPG, BMP'
            }), 400
        
        # Detector yang akan digunakan
        detector_name = request.form.get('detector', app.config['DEFAULT_DETECTOR'])
        
        # Generate unique filename
        unique_filename = f"analyze_{uuid.uuid4().hex}.{image_file.filename.rsplit('.', 1)[1].lower()}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(upload_path)
        
        try:
            # Baca gambar
            img = Image.open(upload_path)
            img_array = np.array(img)
            
            # Pastikan gambar adalah RGB
            if len(img_array.shape) < 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Hash gambar untuk tracking
            img_hash = hash_file(upload_path)
            
            # Analyze image
            start_time = time.time()
            analysis_report = detector.analyze_image_for_report(img_array, detector_name)
            execution_time = time.time() - start_time
            
            # Simpan LSB plane image untuk visualisasi
            lsb_filename = f"lsb_{uuid.uuid4().hex}.png"
            lsb_path = os.path.join(app.config['UPLOAD_FOLDER'], lsb_filename)
            
            # Create LSB visualization
            lsb_viz = np.zeros(img_array.shape[:2], dtype=np.uint8)
            for c in range(min(3, img_array.shape[2])):
                lsb_viz = lsb_viz | ((img_array[:, :, c] & 1) << c)
            lsb_viz = lsb_viz * 80  # Amplify for visibility
            cv2.imwrite(lsb_path, lsb_viz)
            
            # Convert LSB viz to base64
            with open(lsb_path, 'rb') as f:
                lsb_data = f.read()
                lsb_str = base64.b64encode(lsb_data).decode('utf-8')
            
            # Add visualization to response
            analysis_report['visualizations'] = {
                'lsb_plane': lsb_str
            }
            
            # Add execution time
            analysis_report['execution_time'] = execution_time
            
            # Extract detection results for database
            detection = analysis_report['detection_results']
            
            # Simpan analisis ke database
            analysis_data = {
                'image_hash': img_hash,
                'is_stego': detection['is_stego'],
                'confidence': detection['confidence'],
                'detection_score': detection['detection_score'],
                'estimated_payload': detection['estimated_payload'],
                'detector_used': detector_name,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_analysis(analysis_data)
            
            # Simpan model usage
            usage_data = {
                'model_name': detector_name,
                'operation': 'analyze',
                'success': True,
                'execution_time': execution_time,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_model_usage(usage_data)
            
            return jsonify({
                'status': 'success',
                'analysis': analysis_report
            })
            
        except Exception as e:
            logger.error(f"Error in analyze image: {str(e)}", exc_info=True)
            # Hapus file upload yang tidak digunakan
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except:
                pass
            
            # Simpan model usage (failure)
            usage_data = {
                'model_name': detector_name,
                'operation': 'analyze',
                'success': False,
                'execution_time': 0,
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
            save_model_usage(usage_data)
            
            return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in analyze image: {str(e)}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan yang tidak diketahui'}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """API endpoint untuk mendapatkan daftar pre-trained models"""
    try:
        # Dapatkan daftar model
        models = model_manager.list_available_models()
        
        return jsonify({
            'status': 'success',
            'models': models
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/detectors', methods=['GET'])
def list_detectors():
    """API endpoint untuk mendapatkan daftar pre-trained detectors"""
    try:
        # Dapatkan daftar detector
        detectors = detector.list_available_detectors()
        
        return jsonify({
            'status': 'success',
            'detectors': detectors
        })
        
    except Exception as e:
        logger.error(f"Error listing detectors: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/download-model', methods=['POST'])
def download_model():
    """API endpoint untuk mendownload pre-trained model"""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({'error': 'Nama model tidak diberikan'}), 400
            
        model_name = data['model_name']
        
        # Download model
        success = model_manager.download_model(model_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f"Model {model_name} berhasil diunduh"
            })
        else:
            return jsonify({
                'status': 'error',
                'error': f"Gagal mengunduh model {model_name}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/download-detector', methods=['POST'])
def download_detector():
    """API endpoint untuk mendownload pre-trained detector"""
    try:
        data = request.get_json()
        if not data or 'detector_name' not in data:
            return jsonify({'error': 'Nama detector tidak diberikan'}), 400
            
        detector_name = data['detector_name']
        
        # Download detector
        success = detector.download_detector(detector_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f"Detector {detector_name} berhasil diunduh"
            })
        else:
            return jsonify({
                'status': 'error',
                'error': f"Gagal mengunduh detector {detector_name}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error downloading detector: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/check-capacity', methods=['POST'])
def check_capacity():
    """API endpoint untuk memeriksa kapasitas penyembunyian gambar"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
        
        image_file = request.files['image']
        
        # Validasi file gambar
        if image_file.filename == '':
            return jsonify({'error': 'Tidak ada file gambar yang dipilih'}), 400
        
        # Verifikasi ekstensi file
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Format file tidak valid. Format yang didukung: PNG, JPG, BMP'
            }), 400
        
        # Generate unique filename
        unique_filename = f"capacity_{uuid.uuid4().hex}.{image_file.filename.rsplit('.', 1)[1].lower()}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(upload_path)
        
        try:
            # Baca gambar
            img = Image.open(upload_path)
            img_array = np.array(img)
            
            # Pastikan gambar adalah RGB
            if len(img_array.shape) < 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Calculate capacity
            capacity = calculate_embedding_capacity(img_array)
            
            # Model-specific capacity if model provided
            model_info = {}
            model_name = request.form.get('model', None)
            if model_name:
                info = model_manager.get_model_info(model_name)
                if info:
                    model_info = {
                        'name': model_name,
                        'message_length': info.get('message_length', 0),
                        'message_bytes': info.get('message_length', 0) // 8,
                        'message_chars': (info.get('message_length', 0) // 8) - 1,
                        'capacity_factor': info.get('capacity_factor', 0)
                    }
            
            return jsonify({
                'status': 'success',
                'capacity': capacity,
                'model_capacity': model_info
            })
            
        except Exception as e:
            logger.error(f"Error checking capacity: {str(e)}", exc_info=True)
            # Hapus file upload yang tidak digunakan
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except:
                pass
            return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in check capacity: {str(e)}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan yang tidak diketahui'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """
    Route untuk download file hasil steganografi
    Menggunakan send_from_directory yang lebih aman karena mencegah path traversal
    """
    # Verifikasi filename untuk mencegah path traversal
    if '..' in filename or filename.startswith('/'):
        logger.warning(f"Suspicious filename detected: {filename}")
        return "Invalid filename", 400
    
    # Sanitasi nama file
    filename = secure_filename(filename)
    
    # Periksa apakah file ada
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        logger.error(f"File tidak ditemukan: {file_path}")
        return "File tidak ditemukan", 404
    
    logger.info(f"Downloading file: {filename}")
    
    try:
        # Gunakan send_from_directory yang lebih aman
        return send_from_directory(
            directory=app.config['UPLOAD_FOLDER'],
            path=filename,
            as_attachment=True,
            download_name=f"stego_image.png"
        )
    except Exception as e:
        logger.error(f"Error saat download file: {str(e)}")
        return "Terjadi kesalahan saat download file", 500

@app.route('/api/stats', methods=['GET'])
def get_stats_api():
    """API endpoint untuk mendapatkan statistik aplikasi"""
    try:
        # Hitung jumlah file yang telah diproses
        upload_files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f)) and f != '.gitkeep']
        
        # Hitung jumlah stego image
        stego_files = [f for f in upload_files if f.startswith('stego_')]
        
        # Hitung jumlah original image
        original_files = [f for f in upload_files if f.startswith('original_')]
        
        # Dapatkan ukuran total uploads
        total_size = sum(os.path.getsize(os.path.join(uploads_dir, f)) for f in upload_files)
        
        # Get database stats
        db_stats = get_stats()
        
        # Buat statistik gabungan
        stats = {
            'total_files': len(upload_files),
            'stego_images': len(stego_files),
            'original_images': len(original_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'uptime': time.time() - app.startup_time if hasattr(app, 'startup_time') else 0,
            'total_operations': db_stats.get('total_operations', 0),
            'encode_operations': db_stats.get('encode_operations', 0),
            'decode_operations': db_stats.get('decode_operations', 0),
            'avg_psnr': round(db_stats.get('avg_psnr', 0), 2),
            'avg_ssim': round(db_stats.get('avg_ssim', 0), 4),
            'encrypted_percentage': round(db_stats.get('encrypted_percentage', 0), 1),
            'compressed_percentage': round(db_stats.get('compressed_percentage', 0), 1),
            'operations_over_time': db_stats.get('operations_over_time', {}),
            'model_usage': db_stats.get('model_usage', {}),
            'analysis_count': db_stats.get('analysis_count', 0),
            'stego_detected_percentage': round(db_stats.get('stego_detected_percentage', 0), 1)
        }
        
        # Get information about available models and detectors
        stats['available_models'] = len(model_manager.list_available_models())
        stats['available_detectors'] = len(detector.list_available_detectors())
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Gagal mendapatkan statistik'}), 500

@app.route('/api/advanced-stats', methods=['GET'])
def get_advanced_stats():
    """API endpoint untuk mendapatkan statistik lanjutan aplikasi"""
    try:
        # Get basic stats
        basic_stats = get_stats()
        
        # Get model usage stats from database
        conn = sqlite3.connect(app.config['DB_PATH'])
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get model usage stats
        cursor.execute("""
            SELECT model_used, COUNT(*) as count 
            FROM stats 
            WHERE model_used IS NOT NULL 
            GROUP BY model_used
            ORDER BY count DESC
        """)
        model_usage = {row['model_used']: row['count'] for row in cursor.fetchall()}
        
        # Get average metrics per model
        cursor.execute("""
            SELECT model_used, AVG(psnr) as avg_psnr, AVG(ssim) as avg_ssim
            FROM stats 
            WHERE model_used IS NOT NULL AND operation = 'encode'
            GROUP BY model_used
        """)
        model_metrics = {}
        for row in cursor.fetchall():
            model_metrics[row['model_used']] = {
                'avg_psnr': row['avg_psnr'] if row['avg_psnr'] else 0,
                'avg_ssim': row['avg_ssim'] if row['avg_ssim'] else 0
            }
        
        # Get detector usage stats
        cursor.execute("""
            SELECT detector_used, COUNT(*) as count 
            FROM analysis 
            GROUP BY detector_used 
            ORDER BY count DESC
        """)
        detector_usage = {row['detector_used']: row['count'] for row in cursor.fetchall()}
        
        # Get stego detection stats per detector
        cursor.execute("""
            SELECT detector_used, 
                   COUNT(*) as total,
                   SUM(CASE WHEN is_stego = 1 THEN 1 ELSE 0 END) as stego_count
            FROM analysis 
            GROUP BY detector_used
        """)
        detector_stats = {}
        for row in cursor.fetchall():
            if row['total'] > 0:
                detector_stats[row['detector_used']] = {
                    'total': row['total'],
                    'stego_count': row['stego_count'],
                    'stego_percentage': (row['stego_count'] / row['total']) * 100
                }
        
        # Get last 10 analyses
        cursor.execute("""
            SELECT timestamp, is_stego, confidence, detection_score, estimated_payload, detector_used
            FROM analysis
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recent_analyses = [dict(row) for row in cursor.fetchall()]
        
        # Get average execution time for operations
        cursor.execute("""
            SELECT operation, AVG(execution_time) as avg_time
            FROM model_usage
            WHERE success = 1
            GROUP BY operation
        """)
        avg_execution_times = {row['operation']: row['avg_time'] for row in cursor.fetchall()}
        
        # Close connection
        conn.close()
        
        # Create advanced stats
        advanced_stats = {
            **basic_stats,
            'model_usage': model_usage,
            'model_metrics': model_metrics,
            'detector_usage': detector_usage,
            'detector_stats': detector_stats,
            'recent_analyses': recent_analyses,
            'avg_execution_times': avg_execution_times,
            'available_models': len(model_manager.list_available_models()),
            'available_detectors': len(detector.list_available_detectors()),
            'local_models': sum(1 for m in model_manager.list_available_models().values() if m.get('is_local', False)),
            'local_detectors': sum(1 for d in detector.list_available_detectors().values() if d.get('is_local', False))
        }
        
        return jsonify(advanced_stats)
    except Exception as e:
        logger.error(f"Error getting advanced stats: {str(e)}")
        return jsonify({'error': 'Gagal mendapatkan statistik lanjutan'}), 500

@app.route('/api/model-info/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """API endpoint untuk mendapatkan informasi detail model"""
    try:
        # Get model info
        model_info = model_manager.get_model_info(model_name)
        if model_info is None:
            return jsonify({'error': f"Model {model_name} tidak ditemukan"}), 404
        
        return jsonify({
            'status': 'success',
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/detector-info/<detector_name>', methods=['GET'])
def get_detector_info(detector_name):
    """API endpoint untuk mendapatkan informasi detail detector"""
    try:
        # Get detector info
        detectors = detector.list_available_detectors()
        if detector_name not in detectors:
            return jsonify({'error': f"Detector {detector_name} tidak ditemukan"}), 404
        
        detector_info = detectors[detector_name]
        
        return jsonify({
            'status': 'success',
            'detector_info': detector_info
        })
        
    except Exception as e:
        logger.error(f"Error getting detector info: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File terlalu besar. Ukuran maksimum adalah 16MB'}), 413

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('500.html'), 500

# Health check endpoint for monitoring
@app.route('/health')
def health_check():
    """Simple health check endpoint for monitoring"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '2.0.0',
        'pretrained_models': {
            'available': len(model_manager.list_available_models()),
            'local': sum(1 for m in model_manager.list_available_models().values() if m.get('is_local', False))
        },
        'detectors': {
            'available': len(detector.list_available_detectors()),
            'local': sum(1 for d in detector.list_available_detectors().values() if d.get('is_local', False))
        }
    })

if __name__ == '__main__':
    # Pastikan semua direktori yang diperlukan ada
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)
    os.makedirs(app.config['DETECTORS_DIR'], exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Bersihkan file lama saat startup
    cleanup_old_files()
    
    # Inisialisasi database
    init_db()
    
    # Catat waktu startup
    app.startup_time = time.time()
    
    logger.info("======== DeepSteg Enhanced Application Started ========")
    logger.info(f"Version: 2.0.0 (Pretrained Models)")
    app.run(debug=True, host='0.0.0.0', port=5000)