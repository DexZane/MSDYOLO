#!/bin/bash
# MSDYOLO Cloud Environment Setup Script
# Automatically downloads DOTA v1.5, splits images, and prepares training environment

set -e

echo "======================================"
echo "MSDYOLO Cloud Setup"
echo "======================================"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DOTA_DATASET_REPO="OpenDataLab/DOTA_V1_dot_5"
DATASET_DIR="dataset/DOTA"
SPLIT_DIR="dataset/DOTA/split"

# Step 1: Check Python version
echo -e "\n${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Step 2: Install dependencies
echo -e "\n${YELLOW}[2/6] Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Download DOTA v1.5 dataset
echo -e "\n${YELLOW}[3/6] Downloading DOTA v1.5 dataset...${NC}"
if [ -d "$DATASET_DIR/train" ] && [ -d "$DATASET_DIR/val" ]; then
    echo -e "${GREEN}✓ Dataset already exists, skipping download${NC}"
else
    echo "Downloading DOTA v1.5 from official source..."
    echo "This will download ~2.5GB (train + val)"
    echo ""

    mkdir -p "$DATASET_DIR/raw"
    cd "$DATASET_DIR/raw"

    # Download train set (part1 + part2 + part3)
    echo "Downloading train set (part 1/3)..."
    wget -q --show-progress https://captain-whu.github.io/DOTA/dataset/train-part1.zip || {
        echo -e "${RED}Failed to download train-part1.zip${NC}"
        echo "Please manually download from: https://captain-whu.github.io/DOTA/dataset.html"
        exit 1
    }

    echo "Downloading train set (part 2/3)..."
    wget -q --show-progress https://captain-whu.github.io/DOTA/dataset/train-part2.zip

    echo "Downloading train set (part 3/3)..."
    wget -q --show-progress https://captain-whu.github.io/DOTA/dataset/train-part3.zip

    # Download val set
    echo "Downloading val set..."
    wget -q --show-progress https://captain-whu.github.io/DOTA/dataset/val-part1.zip

    # Extract all
    echo "Extracting archives..."
    unzip -q train-part1.zip
    unzip -q train-part2.zip
    unzip -q train-part3.zip
    unzip -q val-part1.zip

    # Organize directory structure
    cd ../../..
    mv "$DATASET_DIR/raw/train" "$DATASET_DIR/"
    mv "$DATASET_DIR/raw/val" "$DATASET_DIR/"
    rm -rf "$DATASET_DIR/raw"

    echo -e "${GREEN}✓ Dataset downloaded and extracted${NC}"
fi

# Step 4: Split large images
echo -e "\n${YELLOW}[4/6] Splitting images (4000x4000 → 1024x1024 patches)...${NC}"
if [ -d "$SPLIT_DIR" ] && [ "$(ls -A $SPLIT_DIR)" ]; then
    echo -e "${GREEN}✓ Split images already exist, skipping${NC}"
else
    mkdir -p "$SPLIT_DIR"

    # Check if split script exists
    if [ ! -f "utils/ImgSplit_multi_process.py" ]; then
        echo -e "${RED}Error: ImgSplit_multi_process.py not found in utils/${NC}"
        exit 1
    fi

    # Split train set
    echo "Splitting train set..."
    python3 utils/ImgSplit_multi_process.py \
        --imageset "$DATASET_DIR/train/images" \
        --labelset "$DATASET_DIR/train/labelTxt" \
        --output "$SPLIT_DIR/train" \
        --gap 200 \
        --subsize 1024 \
        --num_process 8

    # Split val set
    echo "Splitting val set..."
    python3 utils/ImgSplit_multi_process.py \
        --imageset "$DATASET_DIR/val/images" \
        --labelset "$DATASET_DIR/val/labelTxt" \
        --output "$SPLIT_DIR/val" \
        --gap 200 \
        --subsize 1024 \
        --num_process 8

    echo -e "${GREEN}✓ Image splitting complete${NC}"
fi

# Step 5: Verify setup
echo -e "\n${YELLOW}[5/6] Verifying setup...${NC}"

# Check if split images exist
TRAIN_IMAGES=$(find "$SPLIT_DIR/train/images" -name "*.png" 2>/dev/null | wc -l)
VAL_IMAGES=$(find "$SPLIT_DIR/val/images" -name "*.png" 2>/dev/null | wc -l)

if [ "$TRAIN_IMAGES" -gt 0 ] && [ "$VAL_IMAGES" -gt 0 ]; then
    echo -e "${GREEN}✓ Setup verification passed${NC}"
    echo ""
    echo "Dataset statistics:"
    echo "  Train patches: $TRAIN_IMAGES"
    echo "  Val patches: $VAL_IMAGES"
else
    echo -e "${RED}✗ Setup verification failed${NC}"
    echo "Expected images in $SPLIT_DIR but found none"
    exit 1
fi

# Update data config path
echo -e "\n${YELLOW}Updating data config...${NC}"
if [ -f "data/dotav15_poly.yaml" ]; then
    sed -i.bak "s|path:.*|path: $SPLIT_DIR|g" data/dotav15_poly.yaml
    echo -e "${GREEN}✓ Config updated${NC}"
fi

echo ""
echo "======================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Verify GPU availability: nvidia-smi"
echo "  2. Start baseline training: bash scripts/train_baseline_p2.sh"
echo ""
