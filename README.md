# MSDYOLO

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Multi-Scale Deformable YOLO for Oriented Object Detection on DOTA Dataset

## ✨ Features

- 🎯 **Oriented Bounding Box Detection** - Rotated object detection for aerial images
- 🔬 **Multi-Scale Distillation** - Teacher-student knowledge transfer
- 📉 **Degradation Simulation** - PSF blur, downsampling, noise
- 🚀 **One-Command Setup** - Automated dataset preparation and training
- 📊 **DOTA Dataset** - Optimized for DOTA v1.5/v2.0

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/DexZane/MSDYOLO.git
cd MSDYOLO
pip install -e .
```

### One-Command Training

```bash
bash scripts/setup.sh
```

This automatically:
1. ✅ Downloads DOTA v1.5 dataset
2. ✅ Splits images into 1024×1024 patches  
3. ✅ Downloads pretrained YOLOv5 weights
4. ✅ Starts training

### Manual Training

```bash
# Download dataset
python -m msdyolo.data.scripts.download_dota dataset/DOTA

# Split images
python -m msdyolo.data.scripts.split_dota \
  --imageset dataset/DOTA/train/images \
  --labelset dataset/DOTA/train/labelTxt \
  --output dataset/DOTA/split/train \
  --subsize 1024 --gap 200

# Train
python -m msdyolo.train --config configs/train/baseline.yaml --device 0
```

## 📊 Monitor Training

```bash
# View logs
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Stop training
pkill -f msdyolo.train
```

## 📁 Project Structure

```
MSDYOLO/
├── msdyolo/              # Main package
│   ├── train.py          # Training entry
│   ├── detect.py         # Inference
│   ├── models/           # YOLOv5-OBB models
│   ├── utils/            # Core utilities
│   └── data/             # Dataset configs & tools
│       ├── dota.yaml     # DOTA config
│       ├── scripts/      
│       │   ├── download_dota.py
│       │   └── split_dota.py
│       └── hyps/         # Hyperparameters
├── configs/
│   ├── train/            # Training configs
│   │   ├── baseline.yaml
│   │   ├── degradation.yaml
│   │   └── distillation.yaml
│   └── models/           # Model architectures
├── scripts/
│   └── setup.sh          # One-command setup
└── dataset/              # Data (auto-created)
```

## ⚙️ Configuration

### Training Modes

| Mode | File | Description |
|------|------|-------------|
| **Baseline** | `configs/train/baseline.yaml` | Pure detection |
| **Degradation** | `configs/train/degradation.yaml` | With image degradation |
| **Distillation** | `configs/train/distillation.yaml` | Full pipeline |

### Adjust Settings

Edit `configs/train/baseline.yaml`:

```yaml
training:
  epochs: 200
  batchsize: 16      # Reduce for smaller GPUs
  imagesize: 1024    # DOTA standard
  workers: 4         # CPU workers
  device: "0"        # GPU ID
```

## 🔧 Requirements

- **Python** ≥ 3.8
- **PyTorch** ≥ 1.10
- **CUDA** ≥ 11.0 (for GPU)
- **RAM** ≥ 32GB (for dataset prep)
- **VRAM** ≥ 16GB (V100/A100 recommended)

## 📦 Dataset

MSDYOLO supports:
- **DOTA v1.5** - 16 classes, 2806 train + 449 val images
- **DOTA v2.0** - 18 classes

Classes: `plane`, `ship`, `storage-tank`, `baseball-diamond`, `tennis-court`, `basketball-court`, `ground-track-field`, `harbor`, `bridge`, `large-vehicle`, `small-vehicle`, `helicopter`, `roundabout`, `soccer-ball-field`, `swimming-pool`, `container-crane`

## 🐛 Known Issues & Fixes

### Python 3.12 Compatibility
**Error:** `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`

**Fix:** Auto-handled by setup script
```bash
pip install setuptools==69.5.1
```

### Dataloader Deadlock
**Symptom:** Training hangs during cache scanning

**Fix:** Reduced workers to 4 (default in configs)

### Label Format
**Issue:** Labels must use pixel coordinates, not normalized [0,1]

**Status:** ✅ Fixed in `split_dota.py`

## 📈 Performance

Results on DOTA v1.5 (coming soon):

| Model | mAP | Config |
|-------|-----|--------|
| Baseline | TBD | baseline.yaml |
| +Degradation | TBD | degradation.yaml |
| +Distillation | TBD | distillation.yaml |

## 📝 Citation

```bibtex
@software{msdyolo2024,
  title={MSDYOLO: Multi-Scale Deformable YOLO for Oriented Object Detection},
  author={MSDYOLO Team},
  year={2024},
  url={https://github.com/DexZane/MSDYOLO}
}
```

## 🙏 Acknowledgments

- [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb) - Oriented detection framework
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) - Base architecture
- [DOTA Dataset](https://captain-whu.github.io/DOTA/) - Benchmark data
- [OpenDataLab](https://opendatalab.com/) - Dataset hosting

## 📞 Contact

- **GitHub**: [@DexZane](https://github.com/DexZane)
- **Issues**: [GitHub Issues](https://github.com/DexZane/MSDYOLO/issues)

## 📄 License

GPL-3.0 License - see [LICENSE](LICENSE) file

---

**Ready to train?** Run `bash scripts/setup.sh` and you're good to go! 🚀
