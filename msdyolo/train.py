#!/usr/bin/env python
"""MSDYOLO 的真实 YOLOv5-OBB 训练入口。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from msdyolo.utils.config import MSDYOLOConfig


def training_health_message(distillation_enabled: bool, epoch_matches: int, target_count: int) -> str | None:
    """返回仅适用于蒸馏训练的零匹配 epoch 健康警告。"""
    if not distillation_enabled or target_count <= 0 or epoch_matches != 0:
        return None
    return (
        "Warning: distillation received targets="
        f"{target_count} but accumulated match=0 for this epoch"
    )


def trainonebatch(model, trainer, computeloss, images, targets, optimizer, device):
    """完成前向、损失、反向传播和一次优化器更新。"""
    model.train()
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad()
    result = trainer.processbatch(images, targets, computeloss)
    loss = result["loss"]
    loss.backward()
    optimizer.step()
    return result


def createparser():
    """建立保留标准连字符形式的命令行参数。"""
    parser = argparse.ArgumentParser(description="MSDYOLO training with YOLOv5-OBB")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--hyp")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", dest="batchsize", type=int)
    parser.add_argument("--img-size", dest="imagesize", type=int)
    parser.add_argument("--single-batch", dest="singlebatch", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--dry-run", dest="dryrun", action="store_true")
    return parser


def resolvedevice(requested):
    """解析 cpu 或单张 CUDA 设备。"""
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    index = requested.split(",")[0]
    return torch.device(f"cuda:{index}")


def mergedtraining(config, arguments):
    """按命令行优先级合并训练参数。"""
    return {
        "cfg": arguments.cfg or config.get("training.cfg"),
        "data": arguments.data or config.get("training.data"),
        "hyp": arguments.hyp or config.get("training.hyp"),
        "weights": arguments.weights
        if arguments.weights is not None
        else config.get("training.weights", ""),
        "teacherweights": config.get("training.teacherweights", ""),
        "device": arguments.device or config.get("training.device", "cpu"),
        "batchsize": arguments.batchsize
        if arguments.batchsize is not None
        else config.get("training.batchsize", 2),
        "imagesize": arguments.imagesize
        if arguments.imagesize is not None
        else config.get("training.imagesize", 1024),
        "epochs": config.get("training.epochs", 300),
        "workers": config.get("training.workers", 0),
    }


def validatepaths(training):
    """验证真实训练必需文件，权重路径非空时也必须存在。"""
    for key in ("cfg", "data", "hyp"):
        value = training[key]
        if not value:
            raise ValueError(f"Missing required training path: {key}")
        if not Path(value).exists():
            raise FileNotFoundError(f"{key} file not found: {value}")
    if training["weights"] and not Path(training["weights"]).exists():
        raise FileNotFoundError(f"weights file not found: {training['weights']}")


def validateteacherweights(training, distillationenabled):
    """Require a DOTA-adapted checkpoint whenever distillation is enabled."""
    if not distillationenabled:
        return
    teacherweights = training.get("teacherweights", "")
    if not teacherweights:
        raise ValueError(
            "Full distillation training requires a DOTA teacher checkpoint; "
            "train configs/train/teacher.yaml first and set training.teacherweights"
        )
    if not Path(teacherweights).exists():
        raise FileNotFoundError(f"teacherweights file not found: {teacherweights}")


def trainingcheckpointdirectory(config):
    """Return the stable checkpoint directory for a named experiment."""
    return f"runs/train/{config.get('experiment.name')}/weights"


def createoptimizer(model, hyp):
    """创建与 YOLOv5 训练入口一致的 SGD 参数组。"""
    batchnormweights = []
    regularweights = []
    biases = []
    for module in model.modules():
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            biases.append(module.bias)
        if isinstance(module, nn.BatchNorm2d):
            batchnormweights.append(module.weight)
        elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            regularweights.append(module.weight)
    optimizer = torch.optim.SGD(
        batchnormweights,
        lr=hyp["lr0"],
        momentum=hyp["momentum"],
        nesterov=True,
    )
    optimizer.add_param_group(
        {"params": regularweights, "weight_decay": hyp["weight_decay"]}
    )
    optimizer.add_param_group({"params": biases})
    return optimizer


def loadmodel(training, hyp, datadict, device):
    """创建模型并可选加载预训练权重。"""
    from msdyolo.models.yolo import Model

    classcount = int(datadict["nc"])
    model = Model(training["cfg"], ch=3, nc=classcount, anchors=hyp.get("anchors")).to(device)
    if training["weights"]:
        checkpoint = torch.load(training["weights"], map_location=device, weights_only=False)
        # 处理两种checkpoint格式：YOLOv5格式（有model对象）和直接state_dict格式
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpointstate = checkpoint["model"]
            if hasattr(checkpointstate, "float"):
                checkpointstate = checkpointstate.float().state_dict()
            elif isinstance(checkpointstate, dict):
                checkpointstate = checkpointstate
        else:
            checkpointstate = checkpoint

        modelstate = model.state_dict()
        filteredstate = {k: v for k, v in checkpointstate.items()
                        if k in modelstate and v.shape == modelstate[k].shape}
        model.load_state_dict(filteredstate, strict=False)
        print(f"Loaded {len(filteredstate)}/{len(modelstate)} layers from pretrained weights")
    model.hyp = hyp
    model.nc = classcount
    model.names = datadict["names"]
    return model


def createdataloader(model, training, hyp, datadict, imagesize):
    """创建真实 dataloader。"""
    from msdyolo.utils.datasets import create_dataloader
    from msdyolo.utils.general import check_img_size, colorstr

    gridsize = max(int(model.stride.max()), 32)
    checkedsize = check_img_size(imagesize, gridsize, floor=gridsize * 2)
    dataloader, dataset = create_dataloader(
        datadict["train"],
        checkedsize,
        training["batchsize"],
        gridsize,
        datadict["names"],
        hyp=hyp,
        augment=True,
        cache=False,
        rect=False,
        rank=-1,
        workers=training["workers"],
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
    )
    return dataloader, dataset, checkedsize


def rundemo(config):
    """运行不依赖数据集的最小包装器演示。"""
    from msdyolo.utils.trainer import MSDYOLOTrainer

    class DummyYOLOModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.convolution = nn.Conv2d(3, 16, 3, padding=1)
            self.nc = 16

        def forward(self, images):
            return [self.convolution(images)]

    model = DummyYOLOModel()
    trainer = MSDYOLOTrainer(model, config, torch.device("cpu"))
    images = torch.rand(2, 3, 64, 64)
    targets = torch.zeros(0, 187)

    def computeloss(predictions, unusedtargets):
        loss = predictions[0].mean()
        return loss, loss.detach().repeat(4)

    result = trainer.processbatch(images, targets, computeloss)
    result["loss"].backward()
    print(f"Demo completed with finite loss: {result['loss'].item():.6f}")


def main():
    """解析配置并执行演示、单批次或完整 epoch 训练。"""
    arguments = createparser().parse_args()
    config = MSDYOLOConfig(arguments.config)
    training = mergedtraining(config, arguments)
    print(f"Experiment: {config.get('experiment.name')}")
    print(f"Degradation: {config.get('degradation.enabled')}")
    print(f"Clear branch: {config.get('clearbranch.enabled')}")
    print(f"Distillation: {config.get('distillation.enabled')}")

    if not config.validate():
        raise ValueError("Configuration validation failed")
    if arguments.demo:
        rundemo(config)
        return
    validatepaths(training)
    validateteacherweights(training, config.get("distillation.enabled"))
    if arguments.dryrun:
        print("Dry run completed: configuration and paths are valid")
        return

    from msdyolo.utils.general import check_dataset, init_seeds
    from msdyolo.utils.loss import ComputeLoss
    from msdyolo.utils.trainer import MSDYOLOTrainer

    init_seeds(1)
    device = resolvedevice(training["device"])
    with Path(training["hyp"]).open("r", encoding="utf-8", errors="ignore") as stream:
        hyp = yaml.safe_load(stream)
    datadict = check_dataset(training["data"])
    model = loadmodel(training, hyp, datadict, device)
    teachermodel = None
    if config.get("distillation.enabled"):
        teachertraining = dict(training)
        teachertraining["weights"] = training["teacherweights"]
        teachermodel = loadmodel(teachertraining, hyp, datadict, device)
    trainer = MSDYOLOTrainer(model, config, device, teachermodel=teachermodel)

    # 单批次模式启用详细日志
    if arguments.singlebatch:
        trainer.verbose = True
        print("\n[Verbose mode enabled for single-batch diagnostics]\n")

    computeloss = ComputeLoss(model)
    dataloader, dataset, checkedsize = createdataloader(
        model,
        training,
        hyp,
        datadict,
        training["imagesize"],
    )
    del dataset
    optimizer = createoptimizer(model, hyp)
    print(
        f"Training on {device}; batch={training['batchsize']}; "
        f"image={checkedsize}; batches={len(dataloader)}"
    )

    epochs = 1 if arguments.singlebatch else training["epochs"]
    lastresult = None

    # 初始化CSV日志文件（YOLOv5标准格式）
    savedir = Path(trainingcheckpointdirectory(config))
    savedir.mkdir(parents=True, exist_ok=True)
    resultsfile = savedir / "results.csv"

    # 最佳模型追踪
    best_loss = float('inf')
    best_epoch = 0

    # CSV列标题
    headers = [
        "epoch",
        "train/box_loss",
        "train/obj_loss",
        "train/cls_loss",
        "train/total_loss",
        "distill/total",
        "distill/cls",
        "distill/center",
        "distill/scale",
        "distill/angle",
        "distill/match",
        "distill/survival",
        "distill/angrel"
    ]

    # 写入CSV头
    with open(resultsfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    for epoch in range(epochs):
        epochtargetcount = 0
        epochmatchcount = 0

        # 累积epoch级别的损失
        epoch_detection_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_center_loss = 0.0
        epoch_scale_loss = 0.0
        epoch_angle_loss = 0.0
        epoch_survival_sum = 0.0
        epoch_anglereliability_sum = 0.0
        batch_count = 0

        # 创建tqdm进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=120)

        for batchindex, (images, targets, paths, shapes) in enumerate(pbar):
            images = images.to(device, non_blocking=True).float() / 255.0
            lastresult = trainonebatch(
                model,
                trainer,
                computeloss,
                images,
                targets,
                optimizer,
                device,
            )
            epochtargetcount += len(targets)
            epochmatchcount += lastresult["matchcount"]

            # 累积损失
            epoch_detection_loss += lastresult['detectionloss'].item()
            epoch_distill_loss += lastresult['distillationloss'].item()
            epoch_cls_loss += lastresult['classificationloss'].item()
            epoch_center_loss += lastresult['centerloss'].item()
            epoch_scale_loss += lastresult['scaleloss'].item()
            epoch_angle_loss += lastresult['angleloss'].item()
            epoch_survival_sum += lastresult['meansurvival']
            epoch_anglereliability_sum += lastresult['meananglereliability']
            batch_count += 1

            if arguments.singlebatch:
                print(f"Single batch training completed:")
                print(f"  loss={lastresult['loss'].item():.6f}")
                print(f"  detectionloss={lastresult['detectionloss'].item():.6f}")
                print(f"  distillationloss={lastresult['distillationloss'].item():.6f}")
                print(f"  classificationloss={lastresult['classificationloss'].item():.6f}")
                print(f"  centerloss={lastresult['centerloss'].item():.6f}")
                print(f"  scaleloss={lastresult['scaleloss'].item():.6f}")
                print(f"  angleloss={lastresult['angleloss'].item():.6f}")
                print(f"  matchcount={lastresult['matchcount']}")
                print(f"  meansurvival={lastresult['meansurvival']:.6f}")
                print(f"  meananglereliability={lastresult['meananglereliability']:.6f}")
                return

            # 更新进度条显示的核心指标
            pbar.set_postfix({
                'loss': f"{lastresult['loss'].item():.4f}",
                'det': f"{lastresult['detectionloss'].item():.4f}",
                'dist': f"{lastresult['distillationloss'].item():.4f}",
                'match': lastresult['matchcount']
            })

        # 计算epoch平均值
        avg_detection_loss = epoch_detection_loss / batch_count
        avg_distill_loss = epoch_distill_loss / batch_count
        avg_cls_loss = epoch_cls_loss / batch_count
        avg_center_loss = epoch_center_loss / batch_count
        avg_scale_loss = epoch_scale_loss / batch_count
        avg_angle_loss = epoch_angle_loss / batch_count
        avg_survival = epoch_survival_sum / batch_count
        avg_anglereliability = epoch_anglereliability_sum / batch_count
        avg_total_loss = avg_detection_loss + avg_distill_loss

        # 写入CSV（YOLOv5兼容格式 + MSDYOLO扩展）
        with open(resultsfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_detection_loss:.6f}",  # train/box_loss (检测损失)
                "0.0",  # train/obj_loss (MSDYOLO不使用objectness)
                "0.0",  # train/cls_loss (分类损失已包含在box_loss中)
                f"{avg_total_loss:.6f}",  # train/total_loss
                f"{avg_distill_loss:.6f}",  # distill/total_loss
                f"{avg_cls_loss:.6f}",  # distill/cls_loss
                f"{avg_center_loss:.6f}",  # distill/center_loss
                f"{avg_scale_loss:.6f}",  # distill/scale_loss
                f"{avg_angle_loss:.6f}",  # distill/angle_loss
                epochmatchcount,  # distill/match_count
                f"{avg_survival:.6f}",  # distill/mean_survival
                f"{avg_anglereliability:.6f}"  # distill/mean_angle_reliability
            ])

        print(f"Epoch {epoch + 1}/{epochs} completed: loss={avg_total_loss:.6f}")
        message = training_health_message(
            config.get("distillation.enabled"), epochmatchcount, epochtargetcount
        )
        if message is not None:
            print(message)

        savedir = Path(trainingcheckpointdirectory(config))
        savedir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # 保存last.pt（最新权重）
        torch.save(checkpoint, savedir / "last.pt")

        # 保存best.pt（最佳权重）
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_epoch = epoch + 1
            torch.save(checkpoint, savedir / "best.pt")
            print(f"New best model saved (epoch {best_epoch}, loss={best_loss:.6f})")

    # 训练完成总结
    print(f"Training completed. Weights saved to {savedir / 'last.pt'}")
    print(f"Best model: epoch {best_epoch}, loss={best_loss:.6f} -> {savedir / 'best.pt'}")


if __name__ == "__main__":
    main()
