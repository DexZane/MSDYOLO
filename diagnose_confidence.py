"""诊断学生模型置信度分布，分析为何matchcount=0"""

import torch
import yaml
from pathlib import Path
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.torch_utils import select_device

def diagnose_confidence():
    """分析学生模型的置信度分布"""

    # 加载配置
    with open("configs/msdyolo-full.yaml") as f:
        config = yaml.safe_load(f)

    device = select_device("cpu")

    # 加载训练后的模型
    print("Loading trained model...")
    model = Model("models/yolov5s.yaml", ch=3, nc=16).to(device)
    checkpoint = torch.load("runs/train/exp/weights/last.pt", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # 加载数据
    print("Loading data...")
    datadict_path = config["training"]["data"]
    with open(datadict_path) as f:
        datadict = yaml.safe_load(f)

    dataloader = create_dataloader(
        path=datadict["train"],
        imgsz=320,
        batch_size=1,
        stride=32,
        hyp={"mosaic": 0.0, "mixup": 0.0},
        augment=False,
        cache=False,
        rect=False,
        rank=-1,
        workers=0,
        pad=0.0,
        prefix="train: ",
        names=datadict["names"],
    )[0]

    # 获取一个batch
    batch = next(iter(dataloader))
    imgs = batch[0].to(device)

    print(f"\nInput shape: {imgs.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(imgs)

    print(f"\nModel outputs:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: shape={out.shape}")

    # 分析置信度分布
    print("\n" + "="*60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*60)

    all_objectness = []
    all_class_probs = []
    all_confidence = []

    for scale_idx, out in enumerate(outputs):
        # out shape: (batch, anchors, grid_h, grid_w, 5+num_classes+1)
        # [x, y, w, h, angle, objectness, class0, ..., class15]

        objectness = torch.sigmoid(out[..., 5])  # (batch, anchors, grid_h, grid_w)
        class_logits = out[..., 6:]  # (batch, anchors, grid_h, grid_w, num_classes)
        class_probs = torch.sigmoid(class_logits)

        # confidence = objectness × max(class_probs)
        max_class_probs, _ = class_probs.max(dim=-1)
        confidence = objectness * max_class_probs

        all_objectness.append(objectness.flatten())
        all_class_probs.append(max_class_probs.flatten())
        all_confidence.append(confidence.flatten())

        print(f"\nScale {scale_idx} (grid={out.shape[2]}x{out.shape[3]}):")
        print(f"  Objectness:     min={objectness.min():.6f} max={objectness.max():.6f} mean={objectness.mean():.6f}")
        print(f"  Max class prob: min={max_class_probs.min():.6f} max={max_class_probs.max():.6f} mean={max_class_probs.mean():.6f}")
        print(f"  Confidence:     min={confidence.min():.6f} max={confidence.max():.6f} mean={confidence.mean():.6f}")

        # 统计超过阈值的数量
        above_025 = (confidence >= 0.25).sum().item()
        above_020 = (confidence >= 0.20).sum().item()
        above_015 = (confidence >= 0.15).sum().item()
        above_010 = (confidence >= 0.10).sum().item()
        total = confidence.numel()

        print(f"  Above 0.25: {above_025}/{total} ({100*above_025/total:.2f}%)")
        print(f"  Above 0.20: {above_020}/{total} ({100*above_020/total:.2f}%)")
        print(f"  Above 0.15: {above_015}/{total} ({100*above_015/total:.2f}%)")
        print(f"  Above 0.10: {above_010}/{total} ({100*above_010/total:.2f}%)")

    # 汇总统计
    all_objectness = torch.cat(all_objectness)
    all_class_probs = torch.cat(all_class_probs)
    all_confidence = torch.cat(all_confidence)

    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total predictions: {all_confidence.numel()}")
    print(f"Objectness:     min={all_objectness.min():.6f} max={all_objectness.max():.6f} mean={all_objectness.mean():.6f}")
    print(f"Max class prob: min={all_class_probs.min():.6f} max={all_class_probs.max():.6f} mean={all_class_probs.mean():.6f}")
    print(f"Confidence:     min={all_confidence.min():.6f} max={all_confidence.max():.6f} mean={all_confidence.mean():.6f}")

    above_025 = (all_confidence >= 0.25).sum().item()
    above_020 = (all_confidence >= 0.20).sum().item()
    above_015 = (all_confidence >= 0.15).sum().item()
    above_010 = (all_confidence >= 0.10).sum().item()
    total = all_confidence.numel()

    print(f"\nAbove 0.25: {above_025}/{total} ({100*above_025/total:.4f}%)")
    print(f"Above 0.20: {above_020}/{total} ({100*above_020/total:.4f}%)")
    print(f"Above 0.15: {above_015}/{total} ({100*above_015/total:.4f}%)")
    print(f"Above 0.10: {above_010}/{total} ({100*above_010/total:.4f}%)")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if above_025 == 0:
        print("PROBLEM: No predictions exceed confidence threshold 0.25")
        print(f"  - Max confidence achieved: {all_confidence.max():.6f}")
        print(f"  - Gap to threshold: {0.25 - all_confidence.max():.6f}")

        if all_objectness.max() < 0.5:
            print("\n  ROOT CAUSE: Low objectness scores")
            print(f"    - Max objectness: {all_objectness.max():.6f}")
            print("    - Detection head not sufficiently trained")
            print("    - Needs more epochs or better initialization")

        if all_class_probs.max() < 0.5:
            print("\n  CONTRIBUTING FACTOR: Low class probabilities")
            print(f"    - Max class prob: {all_class_probs.max():.6f}")
            print("    - Classification head needs more training")
    else:
        print(f"OK: {above_025} predictions exceed threshold 0.25")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("1. Increase training epochs (50 → 100 or 200)")
    print("2. Use larger image size (320 → 640)")
    print("3. Use more training data (5 images → full DOTA)")
    print("4. Lower confidence threshold temporarily (0.25 → 0.10) for diagnosis")
    print("5. Check if loss is still decreasing (not converged yet)")

if __name__ == "__main__":
    diagnose_confidence()
