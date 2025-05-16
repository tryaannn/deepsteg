#!/bin/bash
# cleanup.sh - Script untuk membersihkan file yang tidak diperlukan

echo "DeepSteg Cleanup Script"
echo "======================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to safely remove files
safe_remove() {
    if [ -f "$1" ]; then
        echo -e "${YELLOW}Removing: $1${NC}"
        rm "$1"
        echo -e "${GREEN}✓ Removed $1${NC}"
    else
        echo -e "${RED}File not found: $1${NC}"
    fi
}

# Function to safely remove directories
safe_remove_dir() {
    if [ -d "$1" ]; then
        echo -e "${YELLOW}Removing directory: $1${NC}"
        rm -rf "$1"
        echo -e "${GREEN}✓ Removed directory $1${NC}"
    else
        echo -e "${RED}Directory not found: $1${NC}"
    fi
}

echo "Starting cleanup process..."

# Remove obsolete files
echo -e "\n${GREEN}[1/4] Removing obsolete files...${NC}"
safe_remove "models/gan_model.py"
safe_remove "static/js/main.js"

# Clean Python cache
echo -e "\n${GREEN}[2/4] Cleaning Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Clean temporary files
echo -e "\n${GREEN}[3/4] Cleaning temporary files...${NC}"
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.bak" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true

# Clean logs (keep recent ones)
echo -e "\n${GREEN}[4/4] Cleaning old logs...${NC}"
if [ -d "logs" ]; then
    find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Cleaned logs older than 7 days${NC}"
fi

# Create necessary directories if they don't exist
echo -e "\n${GREEN}Creating necessary directories...${NC}"
mkdir -p static/uploads
mkdir -p logs
mkdir -p models/saved
mkdir -p models/saved/steganalysis

# Set proper permissions
echo -e "\n${GREEN}Setting permissions...${NC}"
chmod +x run.sh 2>/dev/null || true
chmod +x cleanup.sh 2>/dev/null || true

echo -e "\n${GREEN}Cleanup completed successfully!${NC}"
echo -e "${YELLOW}Note: If you see 'File not found' messages, those files were already removed.${NC}"

# Optional: Run linting/formatting
read -p "Run code formatting with black? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v black &> /dev/null; then
        echo -e "\n${GREEN}Running black formatter...${NC}"
        black . --exclude="venv|env" 2>/dev/null || echo -e "${YELLOW}Black not found or failed${NC}"
    else
        echo -e "${YELLOW}Black not installed. Install with: pip install black${NC}"
    fi
fi

echo -e "\n${GREEN}All done! 🎉${NC}"