#!/bin/bash
###############################################################################
# MSDYOLO Project Restructuring Script
# Reorganizes project to follow YOLOv5-style clean structure
###############################################################################

set -e

echo "=========================================="
echo "MSDYOLO Project Restructuring"
echo "=========================================="

# Backup current state
echo "[1/10] Creating backup..."
git add -A
git commit -m "Pre-restructure backup" || true

# Create new structure
echo "[2/10] Creating new directory structure..."
mkdir -p msdyolo/models
mkdir -p msdyolo/utils
mkdir -p msdyolo/data/scripts
mkdir -p msdyolo/data/hyps/obb
mkdir -p configs/train
mkdir -p configs/models
mkdir -p tests

# Move models
echo "[3/10] Moving models..."
cp models/*.py msdyolo/models/
cp models/*.yaml configs/models/

# Move utils
echo "[4/10] Moving utils..."
cp -r utils/* msdyolo/utils/

# Move data configs
echo "[5/10] Moving data configs..."
cp data/*.yaml msdyolo/data/
cp data/hyps/obb/*.yaml msdyolo/data/hyps/obb/

# Move scripts
echo "[6/10] Moving scripts..."
cp scripts/download_dota.py msdyolo/data/scripts/
cp utils/imgsplit.py msdyolo/data/scripts/split_dota.py

# Move training configs
echo "[7/10] Moving training configs..."
cp configs/*.yaml configs/train/

# Rename and move main scripts
echo "[8/10] Moving main scripts..."
cp trainmsd.py msdyolo/train.py
cp detect.py msdyolo/detect.py
cp export.py msdyolo/export.py

# Create __init__.py files
echo "[9/10] Creating package files..."
cat > msdyolo/__init__.py << 'EOF'
"""
MSDYOLO: Multi-Scale Deformable YOLO for Oriented Object Detection
"""

__version__ = "1.0.0"

from .train import main as train
from .detect import main as detect

__all__ = ['train', 'detect']
EOF

touch msdyolo/models/__init__.py
touch msdyolo/utils/__init__.py
touch msdyolo/data/__init__.py

# Create setup.py
echo "[10/10] Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="msdyolo",
    version="1.0.0",
    author="MSDYOLO Team",
    description="Multi-Scale Deformable YOLO for Oriented Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DexZane/MSDYOLO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "msdyolo-train=msdyolo.train:main",
            "msdyolo-detect=msdyolo.detect:main",
        ],
    },
)
EOF

# Update .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Training
runs/
wandb/
*.weights
*.pt
*.pth

# Data
dataset/
data/examples/

# Logs
*.log
.archive/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Cache
.pytest_cache/
.ruff_cache/
.mypy_cache/
EOF

echo ""
echo "=========================================="
echo "✓ Restructuring complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Test import: python -c 'import msdyolo'"
echo "3. Install package: pip install -e ."
echo "4. Clean old files: bash scripts/cleanup_old.sh"
echo ""
