# MSDYOLO: Multi-Scale Distillation for Oriented Object Detection

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

**MSDYOLO** implements multi-scale distillation for oriented object detection in degraded images. Built on YOLOv5-OBB, it uses a teacher-student architecture to transfer knowledge from clear images to degraded views through four-component distillation.

## Features

- 🎯 **Oriented Object Detection**: Full support for rotated bounding boxes (OBB)
- 🔬 **Multi-Scale Distillation**: Four-component loss (classification, center, scale, angle)
- 📉 **Degradation Simulation**: PSF blur, downsampling, and noise
- 🎓 **Teacher-Student Architecture**: Knowledge distillation for degraded images
- 🧪 **Comprehensive Testing**: 87 unit tests with full P0 verification
- 📊 **DOTA Dataset**: Optimized for aerial/satellite imagery

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+ with CUDA
- 8GB+ GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MSDYOLO.git
cd MSDYOLO

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
```

## Quick Start

### Training

```bash
# Baseline training (no degradation)
python trainmsd.py --config configs/msdyolo-baseline-p2.yaml

# Full training (with degradation and distillation)
python trainmsd.py --config configs/msdyolo-full.yaml
```

### Inference

```bash
# Detect objects with oriented bounding boxes
python detect.py --weights runs/train/exp/weights/best.pt \
                 --source data/images/ \
                 --conf 0.25
```

### Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific test category
pytest tests/test_distillation.py -v
```

## Configuration

Four training modes available:

| Mode | Degradation | Clear Branch | Distillation |
|------|-------------|--------------|--------------|
| **Baseline** | ❌ | ❌ | ❌ |
| **WithDegradation** | ✅ | ❌ | ❌ |
| **WithClearBranch** | ❌ | ✅ | ❌ |
| **Full** | ✅ | ✅ | ✅ |

See `configs/` directory for example configurations.

## Dataset Preparation

### DOTA Dataset

1. Download DOTA v1.5 from [official website](https://captain-whu.github.io/DOTA/dataset.html)
2. Split images into patches:

```bash
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit
python ImgSplit_multi_process.py \
  --srcpath /path/to/DOTA/train \
  --dstpath /path/to/DOTA/train_split_1024 \
  --subsize 1024 --gap 200
```

3. Update `data/dotav15_poly.yaml` with your dataset path

## Architecture

```
┌─────────────┐
│ Clear Image │
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       v                 v
┌─────────────┐   ┌─────────────┐
│  Degradation│   │   Teacher   │
│   (PSF +    │   │  (Frozen)   │
│  Downsample)│   │             │
└──────┬──────┘   └──────┬──────┘
       │                 │
       v                 v
┌─────────────┐   ┌─────────────┐
│   Student   │   │  Knowledge  │
│  (Training) │◄──┤ Distillation│
└─────────────┘   └─────────────┘
```

## Project Structure

```
MSDYOLO/
├── configs/              # Training configurations
├── data/                 # Dataset configs and samples
├── models/               # Model architectures
├── scripts/              # Training and deployment scripts
├── tests/                # Unit tests (87 tests)
├── utils/                # Core utilities
│   ├── degradation.py    # Image degradation
│   ├── distillation.py   # Four-component loss
│   ├── matching.py       # Teacher-student matching
│   └── routing.py        # Adaptive routing
├── trainmsd.py           # Main training script
└── detect.py             # Inference script
```

## Citation

If you use MSDYOLO in your research, please cite:

```bibtex
@software{msdyolo2024,
  title = {MSDYOLO: Multi-Scale Distillation for Oriented Object Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/MSDYOLO}
}
```

## Acknowledgments

- Built on [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- OBB support from [YOLOv5_DOTA_OBB](https://github.com/hukaixuan19970627/yolov5_obb)
- DOTA dataset from [DOTA-devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Development Status

- ✅ **P0**: Code verification and testing (87/87 tests passing)
- ✅ **P1**: Root cause diagnosis (detection head initialization)
- 🚧 **P2**: Baseline detector training (in progress)
- ⏳ **P3**: Full distillation experiments (pending)

## Contact

For questions and issues, please open a GitHub issue.

---

**Note**: This is a research implementation. For production use, additional optimization and testing are recommended.
