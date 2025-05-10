from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
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
from werkzeug.utils import secure_filename
from models.utils import preprocess_image, postprocess_image, calculate_metrics
from models.encoder import encode_message
from models.decoder import decode_message

# Konfigurasi aplikasi Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'deepsteg_default_secret_key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_IMAGE_SIZE'] = 1920  # Maksimum dimensi gambar

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
        
        # Encode pesan ke dalam gambar
        stego_img, metrics = encode_message(processed_img, message)
        
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
        
        return jsonify({
            'status': 'success',
            'image': img_str,
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
        message = decode_message(processed_img)
        
        if not message:
            return jsonify({
                'status': 'warning',
                'message': 'Tidak ada pesan yang ditemukan atau pesan kosong.'
            })
        
        return jsonify({
            'status': 'success',
            'message': message
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
    
    logger.info("======== DeepSteg Application Started ========")
    app.run(debug=True, host='0.0.0.0', port=5000)