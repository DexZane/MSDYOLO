#!/bin/bash
###############################################################################
# MSDYOLO Full Setup and Training Script
# One-click solution for cloud environment
###############################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MSDYOLO Complete Setup & Training${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
PROJECT_DIR="$HOME/MSDYOLO"
DATASET_RAW="$PROJECT_DIR/dataset/DOTA"
DATASET_SPLIT="$PROJECT_DIR/dataset/DOTA/split"

cd "$PROJECT_DIR"

# Step 1: Fix Python 3.12 compatibility
echo -e "\n${YELLOW}[1/8] Fixing Python 3.12 compatibility...${NC}"
pip install -q setuptools==69.5.1
echo -e "${GREEN}✓ Python environment fixed${NC}"

# Step 2: Download DOTA dataset
echo -e "\n${YELLOW}[2/8] Downloading DOTA v1.5 dataset...${NC}"
if [ -d "$DATASET_RAW/train" ] && [ -d "$DATASET_RAW/val" ]; then
    echo -e "${GREEN}✓ Dataset already exists${NC}"
else
    python3 scripts/download_dota.py "$DATASET_RAW" || {
        echo -e "${RED}Download failed. Manual download required:${NC}"
        echo "https://captain-whu.github.io/DOTA/dataset.html"
        exit 1
    }
fi

# Step 3: Fix dataset structure (move labels from subdirectories)
echo -e "\n${YELLOW}[3/8] Organizing dataset structure...${NC}"
if [ -d "$DATASET_RAW/train/labelTxt/DOTA-v1.5_train" ]; then
    echo "Moving train labels to parent directory..."
    mv "$DATASET_RAW/train/labelTxt/DOTA-v1.5_train"/* "$DATASET_RAW/train/labelTxt/" 2>/dev/null || true
    rmdir "$DATASET_RAW/train/labelTxt/DOTA-v1.5_train" 2>/dev/null || true
fi

if [ -d "$DATASET_RAW/val/labelTxt/DOTA-v1.5_val" ]; then
    echo "Moving val labels to parent directory..."
    mv "$DATASET_RAW/val/labelTxt/DOTA-v1.5_val"/* "$DATASET_RAW/val/labelTxt/" 2>/dev/null || true
    rmdir "$DATASET_RAW/val/labelTxt/DOTA-v1.5_val" 2>/dev/null || true
fi
echo -e "${GREEN}✓ Dataset structure organized${NC}"

# Step 4: Split images into patches
echo -e "\n${YELLOW}[4/8] Splitting images (4000x4000 → 1024x1024)...${NC}"
if [ -d "$DATASET_SPLIT/train/images" ]; then
    echo -e "${GREEN}✓ Already split, skipping${NC}"
else
    mkdir -p "$DATASET_SPLIT"

    echo "Splitting train set..."
    python3 utils/imgsplit.py \
        --imageset "$DATASET_RAW/train/images" \
        --labelset "$DATASET_RAW/train/labelTxt" \
        --output "$DATASET_SPLIT/train" \
        --subsize 1024 \
        --gap 200 \
        --num_process 8

    echo "Splitting val set..."
    python3 utils/imgsplit.py \
        --imageset "$DATASET_RAW/val/images" \
        --labelset "$DATASET_RAW/val/labelTxt" \
        --output "$DATASET_SPLIT/val" \
        --subsize 1024 \
        --gap 200 \
        --num_process 8

    echo -e "${GREEN}✓ Image splitting complete${NC}"
fi

# Step 5: Create labelTxt symlinks (required by YOLOv5-OBB)
echo -e "\n${YELLOW}[5/8] Creating label symlinks...${NC}"
cd "$DATASET_SPLIT/train"
ln -sf labels labelTxt 2>/dev/null || true

cd "$DATASET_SPLIT/val"
ln -sf labels labelTxt 2>/dev/null || true

cd "$PROJECT_DIR"
echo -e "${GREEN}✓ Symlinks created${NC}"

# Step 6: Verify dataset
echo -e "\n${YELLOW}[6/8] Verifying dataset...${NC}"
TRAIN_IMAGES=$(ls "$DATASET_SPLIT/train/images" | wc -l)
TRAIN_LABELS=$(ls "$DATASET_SPLIT/train/labels" | wc -l)
VAL_IMAGES=$(ls "$DATASET_SPLIT/val/images" | wc -l)
VAL_LABELS=$(ls "$DATASET_SPLIT/val/labels" | wc -l)

echo "Train: $TRAIN_IMAGES images, $TRAIN_LABELS labels"
echo "Val: $VAL_IMAGES images, $VAL_LABELS labels"

if [ "$TRAIN_LABELS" -eq 0 ] || [ "$VAL_LABELS" -eq 0 ]; then
    echo -e "${RED}✗ No labels found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset verified${NC}"

# Step 7: Update config files
echo -e "\n${YELLOW}[7/8] Updating configuration...${NC}"
# Ensure data config paths are correct
sed -i 's|train_split_1024_gap200|train|g' data/dotav15_poly.yaml 2>/dev/null || true
sed -i 's|val_split_1024_gap200|val|g' data/dotav15_poly.yaml 2>/dev/null || true

# Ensure training config uses 4 workers (avoid deadlock)
sed -i 's/workers: 8/workers: 4/g' configs/msdyolo-baseline-p2.yaml 2>/dev/null || true

echo -e "${GREEN}✓ Configuration updated${NC}"

# Step 8: Start training
echo -e "\n${YELLOW}[8/8] Starting baseline training...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Configuration:${NC}"
echo "  Model: YOLOv5s + MSD"
echo "  Dataset: DOTA v1.5"
echo "  Epochs: 200"
echo "  Batch size: 16"
echo "  Image size: 1024"
echo "  Workers: 4"
echo "  Device: GPU 0"
echo -e "${GREEN}========================================${NC}"

# Kill any existing training
pkill -f trainmsd.py 2>/dev/null || true
sleep 2

# Start training in background
nohup python trainmsd.py \
    --config configs/msdyolo-baseline-p2.yaml \
    --device 0 \
    > training.log 2>&1 &

TRAIN_PID=$!
echo "Training started (PID: $TRAIN_PID)"
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Monitor training:"
echo "  tail -f training.log"
echo ""
echo "Check GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "Results will be saved to: runs/train/baseline-detector/"
echo -e "${GREEN}========================================${NC}"

# Wait a few seconds and show initial output
sleep 5
echo ""
echo "Initial training output:"
tail -20 training.log
