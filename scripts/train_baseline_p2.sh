#!/bin/bash
# MSDYOLO P2阶段云端训练脚本
# 目的：训练baseline DOTA-OBB检测器，为蒸馏验证提供收敛的初始权重

set -e  # 遇到错误立即退出

echo "=========================================="
echo "MSDYOLO P2: Baseline Detector Training"
echo "=========================================="

# ============================================
# 1. 环境检查
# ============================================
echo "[1/6] Checking environment..."

# 检查Python和PyTorch
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found, running on CPU (very slow)"
    DEVICE="cpu"
else
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    DEVICE="0"
fi

# ============================================
# 2. 数据集检查
# ============================================
echo ""
echo "[2/6] Checking DOTA dataset..."

DATA_CONFIG="data/dotav15_poly.yaml"
if [ ! -f "$DATA_CONFIG" ]; then
    echo "ERROR: $DATA_CONFIG not found"
    exit 1
fi

# 读取数据集路径
DATASET_PATH=$(grep "^path:" $DATA_CONFIG | awk '{print $2}')
echo "Dataset path: $DATASET_PATH"

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_PATH"
    echo "Please update path in $DATA_CONFIG or download DOTA dataset"
    echo ""
    echo "DOTA v1.5 download: https://captain-whu.github.io/DOTA/dataset.html"
    exit 1
fi

# 统计图像数量
TRAIN_DIR="$DATASET_PATH/train_split_1024_gap200/images"
VAL_DIR="$DATASET_PATH/val_split_1024_gap200/images"

if [ -d "$TRAIN_DIR" ]; then
    TRAIN_COUNT=$(find "$TRAIN_DIR" -name "*.png" | wc -l)
    echo "Training images: $TRAIN_COUNT"
else
    echo "ERROR: Training directory not found: $TRAIN_DIR"
    exit 1
fi

if [ -d "$VAL_DIR" ]; then
    VAL_COUNT=$(find "$VAL_DIR" -name "*.png" | wc -l)
    echo "Validation images: $VAL_COUNT"
else
    echo "WARNING: Validation directory not found: $VAL_DIR"
fi

# ============================================
# 3. 预训练权重检查
# ============================================
echo ""
echo "[3/6] Checking pretrained weights..."

WEIGHTS="yolov5s.pt"
if [ ! -f "$WEIGHTS" ]; then
    echo "Downloading YOLOv5s COCO pretrained weights..."
    wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
fi

echo "Pretrained weights: $WEIGHTS"

# ============================================
# 4. 配置文件检查
# ============================================
echo ""
echo "[4/6] Checking training configuration..."

CONFIG="configs/msdyolo-baseline-p2.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Configuration file not found: $CONFIG"
    exit 1
fi

echo "Configuration: $CONFIG"
echo ""
echo "Key settings:"
grep -E "(epochs|batchsize|imagesize|device)" $CONFIG | sed 's/^/  /'

# ============================================
# 5. 开始训练
# ============================================
echo ""
echo "[5/6] Starting training..."
echo "Output directory: runs/train/baseline-detector/"
echo "Press Ctrl+C to stop training"
echo ""

# 创建输出目录
mkdir -p runs/train/

# 训练命令
python trainmsd.py \
    --config $CONFIG \
    --device $DEVICE \
    2>&1 | tee runs/train/baseline-p2-training.log

# ============================================
# 6. 训练完成
# ============================================
echo ""
echo "[6/6] Training completed!"
echo ""
echo "Saved weights:"
ls -lh runs/train/exp/weights/
echo ""
echo "Best weights: runs/train/exp/weights/best.pt"
echo "Last weights: runs/train/exp/weights/last.pt"
echo "Training log: runs/train/baseline-p2-training.log"
echo ""
echo "Next step: Verify distillation with trained weights"
echo "  python trainmsd.py --config configs/msdyolo-full.yaml \\"
echo "    --weights runs/train/exp/weights/best.pt --single-batch"
echo ""
echo "=========================================="
echo "P2 Training Complete"
echo "=========================================="
