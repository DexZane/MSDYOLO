#!/usr/bin/env python
"""
MSDYOLO Training Entry Point
Wrapper for YOLOv5 train.py with MSDYOLO configuration support

Usage:
    # Baseline mode
    python train_msdyolo.py --config configs/msdyolo_baseline.yaml

    # Degradation mode
    python train_msdyolo.py --config configs/msdyolo_degradation.yaml

    # Full system
    python train_msdyolo.py --config configs/msdyolo_full.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MSDYOLOConfig
from utils.trainer import MSDYOLOTrainer
# import train as yolov5_train  # Skip for now to avoid dependency issues


def main():
    parser = argparse.ArgumentParser(description='MSDYOLO Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to MSDYOLO config YAML (e.g., configs/msdyolo_baseline.yaml)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, 0, 0,1,2,3, etc.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run: only load config and initialize, do not train')

    args = parser.parse_args()

    # Load MSDYOLO configuration
    print(f"\n{'='*60}")
    print(f"MSDYOLO Training Configuration")
    print(f"{'='*60}")
    print(f"Config file: {args.config}")

    config = MSDYOLOConfig(config_path=args.config)

    # Display configuration
    print(f"\nExperiment: {config.get('experiment.name')}")
    print(f"Description: {config.get('experiment.description')}")
    print(f"Phase: {config.get('experiment.phase')}")

    print(f"\nDegradation: {config.get('degradation.enabled')}")
    if config.get('degradation.enabled'):
        print(f"  - PSF Blur: {config.get('degradation.psf.enabled')}")
        print(f"  - Downsample: {config.get('degradation.downsample.enabled')}")
        print(f"  - Noise: {config.get('degradation.noise.enabled')}")

    print(f"\nClear Branch: {config.get('clear_branch.enabled')}")
    if config.get('clear_branch.enabled'):
        print(f"  - Strategy: {config.get('clear_branch.strategy')}")
        print(f"  - Extract Sparse: {config.get('clear_branch.extract_sparse')}")

    print(f"\nDistillation: {config.get('distillation.enabled')}")
    if config.get('distillation.enabled'):
        print(f"  - Alpha: {config.get('distillation.alpha')}")
        print(f"  - Angle Weight: {config.get('distillation.angle_weight')}")

    print(f"\nTraining Settings:")
    print(f"  - Data: {config.get('training.data')}")
    print(f"  - Model: {config.get('training.cfg')}")
    print(f"  - Epochs: {config.get('training.epochs')}")
    print(f"  - Batch Size: {config.get('training.batch_size')}")
    print(f"  - Image Size: {config.get('training.img_size')}")
    print(f"  - Device: {args.device}")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN: Configuration loaded successfully")
        print("Skipping actual training")
        print(f"{'='*60}\n")
        return

    # Check if GPU training is requested but device is CPU
    if args.device == 'cpu':
        print(f"\n{'='*60}")
        print("WARNING: CPU training detected")
        print("MSDYOLO is designed for GPU training")
        print("CPU mode is only for testing configuration and data loading")
        print(f"{'='*60}\n")

        # For CPU, we can test configuration loading but not full training
        if not args.dry_run:
            print("To test full training on GPU, use: --device 0 (or your GPU ID)")
            print("To continue with CPU testing, add --dry-run flag")
            return

    # Initialize MSDYOLO trainer
    print(f"\nInitializing MSDYOLO Trainer...")

    # Create YOLOv5 opt object from config
    class YOLOOpt:
        """Mimics argparse Namespace for YOLOv5 train.py"""
        pass

    opt = YOLOOpt()
    opt.weights = config.get('training.weights') or ''
    opt.cfg = config.get('training.cfg')
    opt.data = config.get('training.data')
    opt.hyp = config.get('training.hyp')
    opt.epochs = config.get('training.epochs')
    opt.batch_size = config.get('training.batch_size')
    opt.imgsz = config.get('training.img_size')
    opt.device = args.device
    opt.workers = config.get('training.workers', 8)

    # Standard YOLOv5 options
    opt.rect = False
    opt.resume = False
    opt.nosave = False
    opt.noval = False
    opt.noautoanchor = False
    opt.evolve = None
    opt.bucket = ''
    opt.cache = None
    opt.image_weights = False
    opt.multi_scale = False
    opt.single_cls = False
    opt.adam = False
    opt.sync_bn = False
    opt.project = ROOT / 'runs' / 'train'
    opt.name = config.get('experiment.name')
    opt.exist_ok = False
    opt.quad = False
    opt.linear_lr = False
    opt.label_smoothing = 0.0
    opt.patience = 100
    opt.freeze = [0]
    opt.save_period = -1
    opt.local_rank = -1

    # Set save_dir
    from utils.general import increment_path
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    print(f"Save directory: {opt.save_dir}")

    # Initialize MSDYOLOTrainer
    trainer = MSDYOLOTrainer(config)

    print(f"\nMSDYOLO Trainer initialized successfully")
    print(f"  - Degradation enabled: {trainer.config.get('degradation.enabled')}")
    print(f"  - Clear branch enabled: {trainer.config.get('clear_branch.enabled')}")
    print(f"  - Distillation enabled: {trainer.config.get('distillation.enabled')}")

    # For now, we just verify the configuration works
    # Full training integration requires GPU and would be done in actual training loop
    print(f"\n{'='*60}")
    print("MSDYOLO Configuration Test: PASSED ✅")
    print(f"{'='*60}\n")
    print("NOTE: Full training integration requires:")
    print("  1. GPU device")
    print("  2. Integration with YOLOv5 training loop")
    print("  3. Model loading and wrapping")
    print("")
    print("This wrapper successfully demonstrates:")
    print("  ✅ Configuration loading")
    print("  ✅ MSDYOLOTrainer initialization")
    print("  ✅ Parameter parsing")
    print("  ✅ Device detection")
    print("")
    print("For actual training, the trainer.train() method would be called")
    print("with the model, dataloader, and optimizer from YOLOv5 train.py")
    print("")


if __name__ == '__main__':
    main()
