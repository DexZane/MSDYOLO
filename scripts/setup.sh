#!/bin/bash
###############################################################################
# MSDYOLO Complete Setup Script (Post-Restructure)
# One-command setup from fresh cloud instance to training
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MSDYOLO Complete Setup${NC}"
echo -e "${GREEN}========================================${NC}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Step 1: Install package
echo -e "\n${YELLOW}[1/8] Installing MSDYOLO package...${NC}"
pip install -q -e .
pip install -q setuptools==69.5.1  # Python 3.12 fix
echo -e "${GREEN}✓ Package installed${NC}"

# Step 2: Download DOTA dataset
echo -e "\n${YELLOW}[2/8] Setting up DOTA dataset...${NC}"
DATASET_DIR="$PROJECT_DIR/dataset/DOTA"

if [ -d "$DATASET_DIR/train/images" ] && [ -d "$DATASET_DIR/val/images" ]; then
    echo -e "${GREEN}✓ Dataset already exists${NC}"
else
    echo "Downloading DOTA v1.5..."
    python3 -m msdyolo.data.scripts.download_dota "$DATASET_DIR"
fi

# Step 3: Fix dataset structure
echo -e "\n${YELLOW}[3/8] Organizing dataset structure...${NC}"

# Move train labels from subdirectory
if [ -d "$DATASET_DIR/train/labelTxt/DOTA-v1.5_train" ]; then
    echo "Moving train labels..."
    mv "$DATASET_DIR/train/labelTxt/DOTA-v1.5_train"/* "$DATASET_DIR/train/labelTxt/" 2>/dev/null || true
    rmdir "$DATASET_DIR/train/labelTxt/DOTA-v1.5_train" 2>/dev/null || true
fi

# Move val labels from subdirectory (if exists)
if [ -d "$DATASET_DIR/val/labelTxt/DOTA-v1.5_val" ]; then
    echo "Moving val labels..."
    mv "$DATASET_DIR/val/labelTxt/DOTA-v1.5_val"/* "$DATASET_DIR/val/labelTxt/" 2>/dev/null || true
    rmdir "$DATASET_DIR/val/labelTxt/DOTA-v1.5_val" 2>/dev/null || true
fi

# Handle val without labels (download issue)
if [ ! -d "$DATASET_DIR/val/labelTxt" ]; then
    echo -e "${YELLOW}Warning: val/labelTxt not found. Using train-val split.${NC}"
    mkdir -p "$DATASET_DIR/val/labelTxt"
fi

echo -e "${GREEN}✓ Dataset structure organized${NC}"

# Step 4: Split images into patches
echo -e "\n${YELLOW}[4/8] Splitting images into 1024×1024 patches...${NC}"
SPLIT_DIR="$DATASET_DIR/split"

if [ -d "$SPLIT_DIR/train/images" ]; then
    echo -e "${GREEN}✓ Already split, skipping${NC}"
else
    mkdir -p "$SPLIT_DIR"

    echo "Splitting train set..."
    python3 -m msdyolo.data.scripts.split_dota \
        --imageset "$DATASET_DIR/train/images" \
        --labelset "$DATASET_DIR/train/labelTxt" \
        --output "$SPLIT_DIR/train" \
        --subsize 1024 \
        --gap 200 \
        --num_process 8

    if [ -d "$DATASET_DIR/val/images" ]; then
        echo "Splitting val set..."
        python3 -m msdyolo.data.scripts.split_dota \
            --imageset "$DATASET_DIR/val/images" \
            --labelset "$DATASET_DIR/val/labelTxt" \
            --output "$SPLIT_DIR/val" \
            --subsize 1024 \
            --gap 200 \
            --num_process 8
    fi

    echo -e "${GREEN}✓ Image splitting complete${NC}"
fi

# Step 5: Create labelTxt symlinks
echo -e "\n${YELLOW}[5/8] Creating label symlinks...${NC}"
cd "$SPLIT_DIR/train"
ln -sf labels labelTxt 2>/dev/null || true

if [ -d "$SPLIT_DIR/val" ]; then
    cd "$SPLIT_DIR/val"
    ln -sf labels labelTxt 2>/dev/null || true
fi

cd "$PROJECT_DIR"
echo -e "${GREEN}✓ Symlinks created${NC}"

# Step 6: Verify dataset
echo -e "\n${YELLOW}[6/8] Verifying dataset...${NC}"
TRAIN_IMAGES=$(ls "$SPLIT_DIR/train/images" 2>/dev/null | wc -l)
TRAIN_LABELS=$(ls "$SPLIT_DIR/train/labels"/*.txt 2>/dev/null | wc -l)

echo "Train: $TRAIN_IMAGES images, $TRAIN_LABELS labels"

if [ "$TRAIN_LABELS" -eq 0 ]; then
    echo -e "${RED}✗ No labels found!${NC}"
    exit 1
fi

if [ -d "$SPLIT_DIR/val" ]; then
    VAL_IMAGES=$(ls "$SPLIT_DIR/val/images" 2>/dev/null | wc -l)
    VAL_LABELS=$(ls "$SPLIT_DIR/val/labels"/*.txt 2>/dev/null | wc -l)
    echo "Val: $VAL_IMAGES images, $VAL_LABELS labels"
fi

echo -e "${GREEN}✓ Dataset verified${NC}"

# Step 7: Download pretrained weights
echo -e "\n${YELLOW}[7/8] Downloading pretrained weights...${NC}"
if [ ! -f "yolov5s.pt" ]; then
    wget -q https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
    echo -e "${GREEN}✓ Downloaded yolov5s.pt${NC}"
else
    echo -e "${GREEN}✓ yolov5s.pt already exists${NC}"
fi

# Step 8: Start training
echo -e "\n${YELLOW}[8/8] Starting training...${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Configuration:"
echo "  Model: YOLOv5s + MSD"
echo "  Dataset: DOTA v1.5"
echo "  Epochs: 200"
echo "  Batch: 16"
echo "  Image size: 1024"
echo "  Workers: 4"
echo -e "${GREEN}========================================${NC}"

# Kill existing training
pkill -f "msdyolo.train" 2>/dev/null || true
sleep 2

# Start training
nohup python3 -m msdyolo.train \
    --config configs/train/baseline.yaml \
    --device 0 \
    > training.log 2>&1 &

TRAIN_PID=$!
echo ""
echo -e "${GREEN}✓ Training started (PID: $TRAIN_PID)${NC}"
echo ""
echo "Monitor:"
echo "  tail -f training.log"
echo ""
echo "GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop:"
echo "  kill $TRAIN_PID"
echo ""

# Show initial output
sleep 5
echo "Initial output:"
tail -30 training.log

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
