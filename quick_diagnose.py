"""从训练日志快速诊断置信度问题"""

import torch
import yaml
from models.yolo import Model
from utils.torch_utils import select_device

def quick_diagnose():
    """快速分析训练后模型的置信度分布"""

    device = select_device("cpu")

    # 加载训练后的模型
    print("Loading trained model...")
    model = Model("models/yolov5s.yaml", ch=3, nc=16).to(device)
    checkpoint = torch.load("runs/train/exp/weights/last.pt", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # 创建一个随机输入
    print("\nTesting with random input (320x320)...")
    dummy_input = torch.randn(1, 3, 320, 320).to(device)

    # 强制训练模式以获取原始输出
    model.train()
    with torch.no_grad():
        outputs = model(dummy_input)

    print("\n" + "="*60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS (Random Input)")
    print("="*60)

    for scale_idx, out in enumerate(outputs):
        # out shape: (batch, anchors, grid_h, grid_w, 5+num_classes+1)
        # [x, y, w, h, angle, objectness, class0, ..., class15]

        print(f"\nScale {scale_idx}: shape={out.shape}")
        # out shape: (batch, anchors, grid_h, grid_w, 5+num_classes+1)
        # [x, y, w, h, angle, objectness, class0, ..., class15]

        objectness = torch.sigmoid(out[..., 5])
        class_logits = out[..., 6:]
        class_probs = torch.sigmoid(class_logits)
        max_class_probs, _ = class_probs.max(dim=-1)
        confidence = objectness * max_class_probs

        print(f"\nScale {scale_idx} (grid={out.shape[2]}x{out.shape[3]}):")
        print(f"  Objectness:     min={objectness.min():.6f} max={objectness.max():.6f} mean={objectness.mean():.6f}")
        print(f"  Max class prob: min={max_class_probs.min():.6f} max={max_class_probs.max():.6f} mean={max_class_probs.mean():.6f}")
        print(f"  Confidence:     min={confidence.min():.6f} max={confidence.max():.6f} mean={confidence.mean():.6f}")

        above_025 = (confidence >= 0.25).sum().item()
        above_020 = (confidence >= 0.20).sum().item()
        above_015 = (confidence >= 0.15).sum().item()
        above_010 = (confidence >= 0.10).sum().item()
        total = confidence.numel()

        print(f"  Above 0.25: {above_025}/{total} ({100*above_025/total:.4f}%)")
        print(f"  Above 0.20: {above_020}/{total} ({100*above_020/total:.4f}%)")
        print(f"  Above 0.15: {above_015}/{total} ({100*above_015/total:.4f}%)")
        print(f"  Above 0.10: {above_010}/{total} ({100*above_010/total:.4f}%)")

    # 分析训练日志
    print("\n" + "="*60)
    print("TRAINING LOG ANALYSIS")
    print("="*60)

    with open("training_50ep.log") as f:
        lines = f.readlines()

    # 提取最后几个epoch的loss
    last_losses = []
    for line in lines[-20:]:
        if "completed: loss=" in line:
            loss_val = float(line.split("loss=")[1].strip())
            last_losses.append(loss_val)

    if last_losses:
        print(f"\nLast 10 epoch losses: {last_losses[-10:]}")
        print(f"  Min loss: {min(last_losses):.6f}")
        print(f"  Max loss: {max(last_losses):.6f}")
        print(f"  Mean loss: {sum(last_losses)/len(last_losses):.6f}")

        # 检查是否仍在下降
        if len(last_losses) >= 10:
            first_half = sum(last_losses[-10:-5]) / 5
            second_half = sum(last_losses[-5:]) / 5
            print(f"  First half mean: {first_half:.6f}")
            print(f"  Second half mean: {second_half:.6f}")
            if second_half < first_half:
                print("  Status: Still decreasing (NOT converged)")
            else:
                print("  Status: Plateaued (possibly converged)")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    # 检查所有scale的总体情况
    all_above_025 = 0
    all_total = 0
    max_conf = 0

    for out in outputs:
        objectness = torch.sigmoid(out[..., 5])
        class_logits = out[..., 6:]
        class_probs = torch.sigmoid(class_logits)
        max_class_probs, _ = class_probs.max(dim=-1)
        confidence = objectness * max_class_probs

        all_above_025 += (confidence >= 0.25).sum().item()
        all_total += confidence.numel()
        max_conf = max(max_conf, confidence.max().item())

    print(f"\nOverall: {all_above_025}/{all_total} predictions above 0.25 ({100*all_above_025/all_total:.4f}%)")
    print(f"Max confidence: {max_conf:.6f}")
    print(f"Gap to threshold: {0.25 - max_conf:.6f}")

    if all_above_025 == 0:
        print("\nPROBLEM CONFIRMED: No predictions exceed confidence threshold 0.25")
        print("\nROOT CAUSE:")
        print("  - 50 epochs with 5 images is insufficient")
        print("  - Detection head needs more training data/epochs")
        print("  - Loss may still be decreasing (not converged)")

        print("\nRECOMMENDATIONS:")
        print("  1. Use full DOTA training set (~1400 images)")
        print("  2. Train for 100-200 epochs")
        print("  3. Or temporarily lower threshold to 0.10 for diagnosis")
        print("  4. Check if we can use pre-trained DOTA-OBB weights instead")
    else:
        print(f"\nOK: {all_above_025} predictions exceed threshold")

if __name__ == "__main__":
    quick_diagnose()
