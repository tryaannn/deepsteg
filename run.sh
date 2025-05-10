#!/bin/bash
# run.sh - Script untuk Linux/Mac

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   DeepSteg: Deep Learning Steganography   ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Fungsi untuk memeriksa dependensi
check_dependency() {
    command -v $1 >/dev/null 2>&1 || { 
        echo -e "${RED}Error: $1 tidak ditemukan. Mohon install $1 terlebih dahulu.${NC}" >&2
        return 1
    }
    return 0
}

# Periksa dependensi Python
if ! check_dependency python3; then
    echo -e "${YELLOW}Mencoba dengan perintah 'python'...${NC}"
    if ! check_dependency python; then
        echo -e "${RED}Python tidak ditemukan. Mohon install Python 3.7+ terlebih dahulu.${NC}"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Periksa versi Python
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "${GREEN}Menggunakan Python versi $PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 6 ]); then
    echo -e "${RED}Error: DeepSteg memerlukan Python 3.6 atau lebih baru.${NC}"
    exit 1
fi

# Buat dan aktifkan virtual environment
echo -e "\n${BLUE}[1/5]${NC} Menyiapkan virtual environment..."
if [ ! -d "venv" ]; then
    echo -e "  ${YELLOW}→${NC} Membuat virtual environment baru..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}  ✗ Gagal membuat virtual environment. Coba install venv:${NC}"
        echo -e "     $PYTHON_CMD -m pip install --user virtualenv"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Virtual environment berhasil dibuat."
else
    echo -e "  ${GREEN}✓${NC} Virtual environment sudah ada."
fi

# Aktifkan virtual environment
echo -e "  ${YELLOW}→${NC} Mengaktifkan virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}  ✗ Gagal mengaktifkan virtual environment.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Virtual environment berhasil diaktifkan."

# Upgrade pip
echo -e "\n${BLUE}[2/5]${NC} Memperbarui pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${RED}  ✗ Gagal memperbarui pip.${NC}"
else
    echo -e "  ${GREEN}✓${NC} Pip berhasil diperbarui."
fi

# Install dependensi
echo -e "\n${BLUE}[3/5]${NC} Menginstall dependensi..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}  ✗ Gagal menginstall dependensi.${NC}"
    echo -e "${YELLOW}    Dependensi mungkin sudah terinstall, melanjutkan...${NC}"
else
    echo -e "  ${GREEN}✓${NC} Dependensi berhasil diinstall."
fi

# Buat direktori yang diperlukan
echo -e "\n${BLUE}[4/5]${NC} Menyiapkan direktori aplikasi..."
mkdir -p static/uploads logs models/saved
echo -e "  ${GREEN}✓${NC} Direktori siap."

# Jalankan aplikasi
echo -e "\n${BLUE}[5/5]${NC} Memulai aplikasi DeepSteg..."
echo -e "  ${GREEN}✓${NC} Aplikasi berhasil dimulai!"
echo -e "  ${BLUE}→${NC} Akses aplikasi di ${GREEN}http://localhost:5000${NC}"
echo -e "  ${BLUE}→${NC} Tekan Ctrl+C untuk menghentikan aplikasi\n"

python app.py