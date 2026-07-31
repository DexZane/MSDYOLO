#!/usr/bin/env python
"""
MSDYOLO Training Entry Point
集成真实YOLOv5-OBB训练的入口点

Usage:
    # Baseline mode (single batch test)
    python trainmsd.py --config configs/msdyolo_baseline.yaml --data data/dota.yaml --cfg models/yolov5s.yaml

    # Degradation mode
    python trainmsd.py --config configs/msdyolo_degradation.yaml --data data/dota.yaml --cfg models/yolov5s.yaml

    # Dry run (只验证配置)
    python trainmsd.py --config configs/msdyolo_baseline.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MSDYOLOConfig
from utils.trainer import MSDYOLOTrainer
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr, init_seeds


def train_one_batch(model, trainer, compute_loss, images, targets, optimizer, device):
    """训练一个batch（用于P0验证）"""
    model.train()

    # 清零梯度
    optimizer.zero_grad()

    # 通过MSDYOLO包装器处理batch
    result = trainer.process_batch(images, targets, compute_loss)

    loss = result['loss']

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='MSDYOLO Training with Real YOLOv5-OBB')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to MSDYOLO config YAML')
    parser.add_argument('--data', type=str, default='',
                        help='Path to data YAML (e.g., data/dota.yaml)')
    parser.add_argument('--cfg', type=str, default='',
                        help='Path to model config YAML (e.g., models/yolov5s.yaml)')
    parser.add_argument('--weights', type=str, default='',
                        help='Path to pretrained weights (optional)')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml',
                        help='Path to hyperparameters YAML')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, 0, 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='Batch size (default: -1 means use config value)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--single-batch', action='store_true',
                        help='Only train one batch for P0 verification')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run: only validate config, do not train')

    args = parser.parse_args()

    # Load MSDYOLO configuration
    print(f"\n{'='*70}")
    print(f"MSDYOLO Training - {args.config}")
    print(f"{'='*70}")

    config = MSDYOLOConfig(config_path=args.config)

    # Display configuration
    print(f"\nExperiment: {config.get('experiment.name')}")
    print(f"Phase: {config.get('experiment.phase')}")
    print(f"\nFeatures:")
    print(f"  - Degradation: {config.get('degradation.enabled')}")
    print(f"  - Clear Branch: {config.get('clear_branch.enabled')}")
    print(f"  - Distillation: {config.get('distillation.enabled')}")

    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN: Configuration validated successfully ✅")
        print(f"{'='*70}\n")
        return

    # Check if real training parameters are provided
    if not args.data or not args.cfg:
        print(f"\n⚠️  WARNING: Missing --data or --cfg parameters")
        print(f"Real YOLOv5-OBB training requires:")
        print(f"  --data: Path to DOTA data YAML")
        print(f"  --cfg: Path to YOLOv5 model config YAML")
        print(f"\nRunning in DEMO mode with dummy data...\n")
        run_demo_mode(config, args)
        return

    # Setup device
    init_seeds(1)
    if args.device == 'cpu':
        device = torch.device('cpu')
        print(f"\n⚠️  CPU mode: training will be slow")
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load hyperparameters
    print(f"\nLoading hyperparameters from {args.hyp}...")
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    print(f"Hyperparameters: {colorstr('loaded')} ✅")

    # Load data config
    print(f"\nLoading data config from {args.data}...")
    data_dict = check_dataset(args.data)
    nc = int(data_dict['nc'])
    names = data_dict['names']
    assert nc == 16, f"DOTA v1.5 should have nc=16, got {nc}"
    print(f"Dataset: {data_dict.get('dataset', 'DOTA')} with {nc} classes ✅")

    # Create model
    print(f"\nCreating YOLOv5-OBB model from {args.cfg}...")
    model = Model(args.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))
    model.to(device)

    # Load pretrained weights if provided
    if args.weights:
        print(f"Loading pretrained weights from {args.weights}...")
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
        print(f"Pretrained weights loaded ✅")

    # Attach hyperparameters to model (required by ComputeLoss)
    model.hyp = hyp
    model.nc = nc
    model.names = names

    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters ✅")

    # Create MSDYOLO trainer
    print(f"\nInitializing MSDYOLO trainer...")
    trainer = MSDYOLOTrainer(model, config, device)
    print(f"MSDYOLO trainer initialized ✅")

    # Create loss function
    print(f"\nInitializing ComputeLoss...")
    compute_loss = ComputeLoss(model)
    print(f"ComputeLoss initialized ✅")

    # Create dataloader
    batch_size = args.batch_size if args.batch_size > 0 else config.get('training.batch_size', 2)
    img_size = args.img_size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(img_size, gs, floor=gs * 2)

    print(f"\nCreating DOTA dataloader...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Grid size: {gs}")

    train_path = data_dict['train']
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        hyp=hyp,
        augment=True,
        cache=False,
        rect=False,
        rank=-1,
        workers=0,
        image_weights=False,
        quad=False,
        prefix=colorstr('train: ')
    )
    print(f"Dataloader created: {len(dataloader)} batches ✅")

    # Create optimizer
    print(f"\nInitializing optimizer...")
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    optimizer = torch.optim.SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})
    print(f"Optimizer: SGD with {len(g0)} + {len(g1)} + {len(g2)} param groups ✅")

    # Training
    print(f"\n{'='*70}")
    if args.single_batch:
        print(f"SINGLE BATCH MODE (P0 Verification)")
    else:
        print(f"TRAINING MODE")
    print(f"{'='*70}\n")

    # Get first batch
    images, targets = next(iter(dataloader))
    images = images.to(device, non_blocking=True).float() / 255.0

    print(f"First batch loaded:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Targets shape: {targets.shape}")
    print(f"  - Targets device: {targets.device}")

    if args.single_batch:
        print(f"\nTraining single batch...")
        loss = train_one_batch(model, trainer, compute_loss, images, targets, optimizer, device)
        print(f"✅ Single batch training completed!")
        print(f"   Loss: {loss:.4f}")

        print(f"\n{'='*70}")
        print(f"P0 VERIFICATION PASSED ✅")
        print(f"{'='*70}")
        print(f"\nVerified:")
        print(f"  ✅ Real YOLOv5-OBB model loaded")
        print(f"  ✅ Real ComputeLoss initialized")
        print(f"  ✅ Real DOTA dataloader created")
        print(f"  ✅ MSDYOLOTrainer.process_batch() executed")
        print(f"  ✅ Forward/backward/optimizer.step() completed")
        print(f"\nConfiguration:")
        print(f"  - Model: {args.cfg}")
        print(f"  - Data: {args.data}")
        print(f"  - Hyperparameters: {args.hyp}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Image size: {imgsz}")
        print(f"  - Device: {device}")
        print(f"\nMSDYOLO Features:")
        print(f"  - Degradation: {config.get('degradation.enabled')}")
        print(f"  - Clear Branch: {config.get('clear_branch.enabled')}")
        print(f"  - Distillation: {config.get('distillation.enabled')}")
    else:
        print(f"\n⚠️  Full training not implemented yet")
        print(f"Use --single-batch flag for P0 verification")

    print()


def run_demo_mode(config, args):
    """演示模式（使用DummyModel）"""
    print(f"{'='*70}")
    print(f"DEMO MODE: Using DummyModel")
    print(f"{'='*70}\n")

    # 导入DummyModel定义
    class DummyYOLOModel(nn.Module):
        def __init__(self, nc=16):
            super().__init__()
            self.nc = nc
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(8),
            )
            self.head = nn.Linear(32 * 8 * 8, 5 + nc + 180)

        def forward(self, x):
            x = self.model(x)
            x = x.view(x.size(0), -1)
            out = self.head(x)
            return out.unsqueeze(1)

    class DummyLoss(nn.Module):
        def __call__(self, predictions, targets):
            return (predictions ** 2).mean()

    device = torch.device('cpu')
    model = DummyYOLOModel(nc=16).to(device)
    trainer = MSDYOLOTrainer(model, config, device)
    compute_loss = DummyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    batch_size = config.get('training.batch_size', 2)
    img_size = config.get('training.img_size', 640)

    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    targets = [torch.zeros(0, 6) for _ in range(batch_size)]

    print(f"Training single dummy batch...")
    loss = train_one_batch(model, trainer, compute_loss, images, targets, optimizer, device)
    print(f"✅ Demo completed! Loss: {loss:.4f}\n")


if __name__ == '__main__':
    main()
