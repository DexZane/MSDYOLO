#!/usr/bin/env python
"""
MSDYOLO Training Entry Point
真正接入YOLOv5训练循环的入口点

Usage:
    # Baseline mode
    python trainmsd.py --config configs/msdyolo_baseline.yaml --epochs 1

    # Degradation mode
    python trainmsd.py --config configs/msdyolo_degradation.yaml --epochs 1

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
from utils.trainer import MSDYOLOTrainer


class DummyYOLOModel(nn.Module):
    """
    临时YOLO模型用于演示训练循环
    实际应该从YOLOv5加载真实模型
    """
    def __init__(self, nc=16):
        super().__init__()
        self.nc = nc
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
        )
        # 输出维度：5 + nc + 180
        self.head = nn.Linear(64 * 8 * 8, 5 + nc + 180)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out.unsqueeze(1)  # (B, 1, 5+nc+180)


class DummyLoss(nn.Module):
    """临时损失函数"""
    def __call__(self, predictions, targets):
        # 简单返回预测的L2 norm
        return (predictions ** 2).mean()


def create_dummy_batch(batch_size, img_size, device):
    """创建虚拟batch用于演示"""
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    targets = [torch.zeros(0, 6) for _ in range(batch_size)]
    return images, targets


def train_one_epoch(model, trainer, compute_loss, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 通过MSDYOLO包装器处理batch
        result = trainer.process_batch(images, targets, compute_loss)

        loss = result['loss']

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # 演示模式只训练前10个batch
        if batch_idx >= 9:
            break

    avg_loss = total_loss / min(10, len(dataloader))
    print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='MSDYOLO Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to MSDYOLO config YAML')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, 0, 0,1,2,3')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs (default: 1 for demo)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run: only validate config, do not train')

    args = parser.parse_args()

    # Load configuration
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

    # Setup device
    if args.device == 'cpu':
        device = torch.device('cpu')
        print(f"\n⚠️  CPU mode: training will be slow, recommended for testing only")
    else:
        device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)
        print(f"\nUsing device: {device}")

    # Create model
    print(f"\nInitializing model...")
    model = DummyYOLOModel(nc=16)
    model.to(device)

    # Create MSDYOLO trainer
    print(f"Initializing MSDYOLO trainer...")
    trainer = MSDYOLOTrainer(model, config, device)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create loss function
    compute_loss = DummyLoss()

    # Create dummy dataloader
    batch_size = config.get('training.batch_size', 2)
    img_size = config.get('training.img_size', 640)

    print(f"Creating dummy dataloader (batch_size={batch_size}, img_size={img_size})...")

    # 创建10个dummy batch
    dummy_data = []
    for _ in range(10):
        images, targets = create_dummy_batch(batch_size, img_size, device)
        dummy_data.append((images, targets))

    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epoch(s)...")
    print(f"{'='*70}\n")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        avg_loss = train_one_epoch(
            model=model,
            trainer=trainer,
            compute_loss=compute_loss,
            dataloader=dummy_data,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

    print(f"\n{'='*70}")
    print(f"Training completed successfully ✅")
    print(f"{'='*70}")
    print(f"\nFinal model state:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"\nMSDYOLOTrainer statistics:")
    print(f"  - Degradation applied: {trainer.config.get('degradation.enabled')}")
    print(f"  - Clear branch used: {trainer.config.get('clear_branch.enabled')}")
    print(f"  - Distillation used: {trainer.config.get('distillation.enabled')}")
    print(f"\nNOTE: This is a demonstration with dummy data and model.")
    print(f"For actual DOTA training, replace DummyYOLOModel with real YOLOv5-OBB")
    print(f"and use real DOTA dataloader.\n")


if __name__ == '__main__':
    main()
