"""诊断清晰分支（教师）的置信度分布"""

import torch
import yaml
from models.yolo import Model
from utils.torch_utils import select_device
from utils.datasets import create_dataloader

def diagnose_teacher():
    """分析教师模型（清晰分支）对真实图像的置信度"""

    device = select_device("cpu")

    # 加载配置
    with open("configs/msdyolo-full.yaml") as f:
        config = yaml.safe_load(f)

    # 加载训练后的模型
    print("Loading trained model...")
    model = Model("models/yolov5s.yaml", ch=3, nc=16).to(device)
    checkpoint = torch.load("runs/train/exp/weights/last.pt", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()  # 教师模式（清晰分支）

    # 加载真实数据
    print("Loading real data...")
    with open(config["training"]["data"]) as f:
        datadict = yaml.safe_load(f)

    # 加载hyp参数
    with open(config["training"]["hyp"]) as f:
        hyp = yaml.safe_load(f)

    # 构建完整路径
    data_path = datadict["path"] + "/" + datadict["train"]

    dataloader = create_dataloader(
        path=data_path,
        imgsz=320,
        batch_size=1,
        stride=32,
        hyp=hyp,
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
    imgs = batch[0].to(device).float()  # 确保是float类型

    print(f"\nReal image shape: {imgs.shape}")

    # 强制训练模式获取原始输出
    model.train()
    with torch.no_grad():
        outputs = model(imgs)

    print("\n" + "="*60)
    print("TEACHER (CLEAR BRANCH) CONFIDENCE ON REAL DATA")
    print("="*60)

    all_confidence = []

    for scale_idx, out in enumerate(outputs):
        objectness = torch.sigmoid(out[..., 5])
        class_logits = out[..., 6:]
        class_probs = torch.sigmoid(class_logits)
        max_class_probs, _ = class_probs.max(dim=-1)
        confidence = objectness * max_class_probs

        all_confidence.append(confidence.flatten())

        print(f"\nScale {scale_idx} (grid={out.shape[2]}x{out.shape[3]}):")
        print(f"  Objectness:     min={objectness.min():.6f} max={objectness.max():.6f} mean={objectness.mean():.6f}")
        print(f"  Max class prob: min={max_class_probs.min():.6f} max={max_class_probs.max():.6f} mean={max_class_probs.mean():.6f}")
        print(f"  Confidence:     min={confidence.min():.6f} max={confidence.max():.6f} mean={confidence.mean():.6f}")

        above_025 = (confidence >= 0.25).sum().item()
        above_020 = (confidence >= 0.20).sum().item()
        above_015 = (confidence >= 0.15).sum().item()
        above_010 = (confidence >= 0.10).sum().item()
        above_005 = (confidence >= 0.05).sum().item()
        total = confidence.numel()

        print(f"  Above 0.25: {above_025}/{total} ({100*above_025/total:.4f}%)")
        print(f"  Above 0.20: {above_020}/{total} ({100*above_020/total:.4f}%)")
        print(f"  Above 0.15: {above_015}/{total} ({100*above_015/total:.4f}%)")
        print(f"  Above 0.10: {above_010}/{total} ({100*above_010/total:.4f}%)")
        print(f"  Above 0.05: {above_005}/{total} ({100*above_005/total:.4f}%)")

    # 总体统计
    all_confidence = torch.cat(all_confidence)

    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)

    above_025 = (all_confidence >= 0.25).sum().item()
    above_020 = (all_confidence >= 0.20).sum().item()
    above_015 = (all_confidence >= 0.15).sum().item()
    above_010 = (all_confidence >= 0.10).sum().item()
    above_005 = (all_confidence >= 0.05).sum().item()
    total = all_confidence.numel()

    print(f"\nTotal predictions: {total}")
    print(f"Max confidence: {all_confidence.max():.6f}")
    print(f"\nAbove 0.25: {above_025}/{total} ({100*above_025/total:.4f}%)")
    print(f"Above 0.20: {above_020}/{total} ({100*above_020/total:.4f}%)")
    print(f"Above 0.15: {above_015}/{total} ({100*above_015/total:.4f}%)")
    print(f"Above 0.10: {above_010}/{total} ({100*above_010/total:.4f}%)")
    print(f"Above 0.05: {above_005}/{total} ({100*above_005/total:.4f}%)")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if above_010 == 0:
        print("\nPROBLEM: Even teacher model has NO predictions above 0.10")
        print("ROOT CAUSE: Model trained on degraded images only")
        print("  - 50 epochs trained with degradation enabled")
        print("  - Model learned degraded feature distribution")
        print("  - Clean images are OUT-OF-DISTRIBUTION for this model")
        print("\nSOLUTION:")
        print("  1. Train baseline WITHOUT degradation (detection only)")
        print("  2. Use that as teacher weights")
        print("  3. Or use pre-trained DOTA-OBB weights")
    else:
        print(f"\nOK: {above_010} predictions exceed 0.10 threshold")
        print("Teacher model has detection capability on clean images")

if __name__ == "__main__":
    diagnose_teacher()
