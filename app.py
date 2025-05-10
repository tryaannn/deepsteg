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
from werkzeug.utils import secure_filename
from models.utils import preprocess_image, postprocess_image, calculate_metrics
from models.enhanced_encoder import encode_message
from models.enhanced_decoder import decode_message
from models.crypto import calculate_password_strength

# Konfigurasi aplikasi Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'deepsteg_enhanced_secret_key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_IMAGE_SIZE'] = 1920  # Maksimum dimensi gambar
app.config['SESSION_TYPE'] = 'filesystem'

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

# Routes
@app.route('/')
def index():
    # Bersihkan file lama saat halaman utama diakses
    cleanup_old_files()
    return render_template('index.html')

@app.route('/encode')
def encode_page():
    return render_template('encode.html')

@app.route('/decode')
def decode_page():
    return render_template('decode.html')

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
    compression_level = 6  # Default level
    use_compression = request.form.get('use_compression', 'false') == 'true'
    if use_compression:
        try:
            compression_level = int(request.form.get('compression_level', '6'))
            if not 0 <= compression_level <= 9:
                compression_level = 6  # Reset ke default jika di luar rentang
        except ValueError:
            pass  # Gunakan default jika input tidak valid
    
    # Generate unique filename untuk mencegah overwrite
    original_filename = secure_filename(image_file.filename)
    file_extension = original_filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
    result_filename = f"stego_{uuid.uuid4().hex}.png"  # Selalu simpan hasil sebagai PNG
    
    try:
        # Baca dan preprocessing gambar
        img = Image.open(image_file)
        
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
        if use_compression:
            stego_img, metrics = encode_message(processed_img, message, password if use_encryption else None, compression_level)
        else:
            stego_img, metrics = encode_message(processed_img, message, password if use_encryption else None, None)
        
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
        
        return jsonify({
            'status': 'success',
            'image': img_str,
            'original_image': original_str,
            'metrics': metrics,
            'filename': result_filename
        })
    
    except Exception as e:
        logger.error(f"Error in encode: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/decode', methods=['POST'])
def decode():
    """API endpoint untuk decoding pesan dari gambar"""
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
    
    try:
        # Baca dan preprocessing gambar
        img = Image.open(image_file)
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
        
        return jsonify({
            'status': 'success',
            'message': message,
            'metadata': metadata
        })
    
    except Exception as e:
        logger.error(f"Error in decode: {str(e)}", exc_info=True)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

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
def get_stats():
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
        
        # Buat statistik
        stats = {
            'total_files': len(upload_files),
            'stego_images': len(stego_files),
            'original_images': len(original_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'uptime': time.time() - app.startup_time if hasattr(app, 'startup_time') else 0
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Gagal mendapatkan statistik'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File terlalu besar. Ukuran maksimum adalah 16MB'}), 413

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Pastikan semua direktori yang diperlukan ada
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Bersihkan file lama saat startup
    cleanup_old_files()
    
    # Catat waktu startup
    app.startup_time = time.time()
    
    logger.info("======== DeepSteg Enhanced Application Started ========")
    app.run(debug=True, host='0.0.0.0', port=5000)