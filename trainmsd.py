#!/usr/bin/env python
"""
MSDYOLO Training Entry Point
集成真实YOLOv5-OBB训练的入口点

Usage:
    # Single batch test with real YOLOv5-OBB
    python trainmsd.py --config configs/msdyolo_baseline.yaml --single-batch

    # Demo mode (no real data required)
    python trainmsd.py --config configs/msdyolo_baseline.yaml --demo

    # Dry run (只验证配置)
    python trainmsd.py --config configs/msdyolo_baseline.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MSDYOLOConfig


def train_one_batch(model, trainer, compute_loss, images, targets, optimizer, device):
    """训练一个batch（用于P0验证）"""
    model.train()

    # 移动targets到训练设备（修复GPT指出的设备不一致问题）
    targets = targets.to(device, non_blocking=True)

    # 清零梯度
    optimizer.zero_grad()

    # 通过MSDYOLO包装器处理batch
    result = trainer.process_batch(images, targets, compute_loss)

    loss = result['loss']

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()
    optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='MSDYOLO Training with Real YOLOv5-OBB')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to MSDYOLO config YAML')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data YAML (default: read from config)')
    parser.add_argument('--cfg', type=str, default=None,
                        help='Path to model config YAML (default: read from config)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights (default: read from config)')
    parser.add_argument('--hyp', type=str, default=None,
                        help='Path to hyperparameters YAML (default: read from config)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, 0, 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: read from config)')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size (default: read from config)')
    parser.add_argument('--single-batch', action='store_true',
                        help='Only train one batch for P0 verification')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode with dummy data (no real DOTA required)')
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

    # Demo mode
    if args.demo:
        print(f"\n{'='*70}")
        print("DEMO MODE: Using DummyModel (no real DOTA data)")
        print(f"{'='*70}\n")
        run_demo_mode(config, args)
        return

    # Real training mode - merge config priorities
    # Priority: command line > config file > error
    cfg_path = args.cfg if args.cfg is not None else config.get('training.cfg')
    data_path = args.data if args.data is not None else config.get('training.data')
    hyp_path = args.hyp if args.hyp is not None else config.get('training.hyp', 'data/hyps/obb/hyp.finetune_dota.yaml')
    weights_path = args.weights if args.weights is not None else config.get('training.weights', '')
    batch_size = args.batch_size if args.batch_size is not None else config.get('training.batch_size', 2)
    img_size = args.img_size if args.img_size is not None else config.get('training.img_size', 1024)

    # Validate required parameters
    if not cfg_path or not data_path:
        print(f"\n❌ ERROR: Missing required training parameters")
        print(f"\nReal YOLOv5-OBB training requires:")
        print(f"  --cfg or training.cfg in config: Path to YOLOv5 model YAML")
        print(f"  --data or training.data in config: Path to DOTA data YAML")
        print(f"\nCurrent values:")
        print(f"  cfg: {cfg_path}")
        print(f"  data: {data_path}")
        print(f"\nTo run without real data, use: --demo")
        sys.exit(1)

    # Validate file existence
    if not Path(cfg_path).exists():
        print(f"\n❌ ERROR: Model config not found: {cfg_path}")
        sys.exit(1)
    if not Path(data_path).exists():
        print(f"\n❌ ERROR: Data config not found: {data_path}")
        sys.exit(1)
    if not Path(hyp_path).exists():
        print(f"\n❌ ERROR: Hyperparameters not found: {hyp_path}")
        sys.exit(1)

    print(f"\nConfiguration merged:")
    print(f"  Model cfg: {cfg_path}")
    print(f"  Data: {data_path}")
    print(f"  Hyperparameters: {hyp_path}")
    print(f"  Weights: {weights_path if weights_path else 'None (train from scratch)'}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")

    # Lazy import real YOLOv5-OBB modules (避免扩展编译问题阻塞dry-run和demo)
    print(f"\nLoading YOLOv5-OBB modules...")
    try:
        from utils.trainer import MSDYOLOTrainer
        from models.yolo import Model
        from utils.loss import ComputeLoss
        from utils.datasets import create_dataloader
        from utils.general import check_dataset, check_img_size, colorstr, init_seeds
        import yaml
        print(f"YOLOv5-OBB modules loaded ✅")
    except ImportError as e:
        print(f"\n❌ ERROR: Failed to import YOLOv5-OBB modules")
        print(f"   {e}")
        print(f"\nThis usually means:")
        print(f"  1. C++ extensions not compiled (run: python setup.py develop)")
        print(f"  2. Missing dependencies (check requirements.txt)")
        print(f"\nTo test configuration without dependencies, use: --dry-run or --demo")
        sys.exit(1)

    # Setup device
    init_seeds(1)
    if args.device == 'cpu':
        device = torch.device('cpu')
        print(f"\n⚠️  CPU mode: training will be slow")
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load hyperparameters
    print(f"\nLoading hyperparameters from {hyp_path}...")
    with open(hyp_path, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    print(f"Hyperparameters: {colorstr('loaded')} ✅")

    # Load data config
    print(f"\nLoading data config from {data_path}...")
    data_dict = check_dataset(data_path)
    nc = int(data_dict['nc'])
    names = data_dict['names']
    assert nc == 16, f"DOTA v1.5 should have nc=16, got {nc}"
    print(f"Dataset: {data_dict.get('dataset', 'DOTA')} with {nc} classes ✅")

    # Create model
    print(f"\nCreating YOLOv5-OBB model from {cfg_path}...")
    model = Model(cfg_path, ch=3, nc=nc, anchors=hyp.get('anchors'))
    model.to(device)

    # Load pretrained weights if provided
    if weights_path and Path(weights_path).exists():
        print(f"Loading pretrained weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location=device)
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
        prefix=colorstr('train: '),
        names=names  # 添加names参数
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
        print(f"FULL TRAINING MODE")
    print(f"{'='*70}\n")

    # Get first batch
    images, targets = next(iter(dataloader))
    images = images.to(device, non_blocking=True).float() / 255.0

    print(f"First batch loaded:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Targets shape: {targets.shape}")
    print(f"  - Images device: {images.device}")
    print(f"  - Targets device: {targets.device}")

    if args.single_batch:
        # Record memory before training
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device) / 1024**2
            print(f"  - GPU memory before: {mem_before:.1f} MB")

        print(f"\nTraining single batch...")
        loss = train_one_batch(model, trainer, compute_loss, images, targets, optimizer, device)

        # Record memory after training
        if device.type == 'cuda':
            mem_after = torch.cuda.memory_allocated(device) / 1024**2
            mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"  - GPU memory after: {mem_after:.1f} MB")
            print(f"  - GPU memory peak: {mem_peak:.1f} MB")

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
        print(f"  ✅ Targets moved to correct device")
        print(f"\nConfiguration:")
        print(f"  - Model: {cfg_path}")
        print(f"  - Data: {data_path}")
        print(f"  - Hyperparameters: {hyp_path}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Image size: {imgsz}")
        print(f"  - Device: {device}")
        print(f"\nMSDYOLO Features:")
        print(f"  - Degradation: {config.get('degradation.enabled')}")
        print(f"  - Clear Branch: {config.get('clear_branch.enabled')}")
        print(f"  - Distillation: {config.get('distillation.enabled')}")
    else:
        print(f"\n⚠️  Full training loop not implemented yet")
        print(f"Full epoch training with validation, checkpoint saving, and")
        print(f"learning rate scheduling will be added in next phase.")
        print(f"\nUse --single-batch flag for P0 verification")

    print()


def run_demo_mode(config, args):
    """演示模式（使用DummyModel）"""
    from utils.trainer import MSDYOLOTrainer

    print(f"Demo mode uses dummy model and random data")
    print(f"No real DOTA dataset or C++ extensions required\n")

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
    targets = torch.zeros(0, 6, device=device)  # Empty targets

    print(f"Training single dummy batch...")
    loss = train_one_batch(model, trainer, compute_loss, images, targets, optimizer, device)
    print(f"✅ Demo completed! Loss: {loss:.4f}\n")


if __name__ == '__main__':
    main()
