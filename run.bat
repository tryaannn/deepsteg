# Windows batch script (run.bat)
@echo off
cls
color 0B

echo ================================================
echo    DeepSteg: Deep Learning Steganography
echo ================================================
echo.

REM Periksa Python tersedia
where python > nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python tidak ditemukan. Mohon install Python 3.7+ terlebih dahulu.
    echo.
    pause
    exit /b 1
)

REM Periksa versi Python
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo [INFO] Menggunakan Python versi %PYTHON_VERSION%
echo.

REM Buat virtual environment jika belum ada
echo [1/5] Menyiapkan virtual environment...
if not exist venv (
    echo   - Membuat virtual environment baru...
    python -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo   [ERROR] Gagal membuat virtual environment.
        echo   [INFO] Coba install venv: python -m pip install --user virtualenv
        pause
        exit /b 1
    )
    echo   + Virtual environment berhasil dibuat.
) else (
    echo   + Virtual environment sudah ada.
)

REM Aktifkan virtual environment
echo   - Mengaktifkan virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    color 0C
    echo   [ERROR] Gagal mengaktifkan virtual environment.
    pause
    exit /b 1
)
echo   + Virtual environment berhasil diaktifkan.

REM Upgrade pip
echo.
echo [2/5] Memperbarui pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo   [WARN] Gagal memperbarui pip, melanjutkan...
) else (
    echo   + Pip berhasil diperbarui.
)

REM Install dependensi
echo.
echo [3/5] Menginstall dependensi...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo   [WARN] Ada masalah saat menginstall dependensi.
    echo   [INFO] Dependensi mungkin sudah terinstall, melanjutkan...
) else (
    echo   + Dependensi berhasil diinstall.
)

REM Buat direktori yang diperlukan
echo.
echo [4/5] Menyiapkan direktori aplikasi...
if not exist static\uploads mkdir static\uploads
if not exist logs mkdir logs
if not exist models\saved mkdir models\saved
echo   + Direktori siap.

REM Jalankan aplikasi
echo.
echo [5/5] Memulai aplikasi DeepSteg...
echo   + Aplikasi berhasil dimulai!
echo   - Akses aplikasi di http://localhost:5000
echo   - Tekan Ctrl+C untuk menghentikan aplikasi
echo.

python app.py

pause