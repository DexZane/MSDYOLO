#!/usr/bin/env python
"""MSDYOLO 的真实 YOLOv5-OBB 训练入口。"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MSDYOLOConfig


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
    from models.yolo import Model

    classcount = int(datadict["nc"])
    model = Model(training["cfg"], ch=3, nc=classcount, anchors=hyp.get("anchors")).to(device)
    if training["weights"]:
        checkpoint = torch.load(training["weights"], map_location=device)
        model.load_state_dict(checkpoint["model"].float().state_dict(), strict=False)
    model.hyp = hyp
    model.nc = classcount
    model.names = datadict["names"]
    return model


def createdataloader(model, training, hyp, datadict, imagesize):
    """创建真实 DOTA dataloader。"""
    from utils.datasets import create_dataloader
    from utils.general import check_img_size, colorstr

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
    """运行不依赖 DOTA 的最小包装器演示。"""
    from utils.trainer import MSDYOLOTrainer

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
    if arguments.dryrun:
        print("Dry run completed: configuration and paths are valid")
        return

    from utils.general import check_dataset, init_seeds
    from utils.loss import ComputeLoss
    from utils.trainer import MSDYOLOTrainer

    init_seeds(1)
    device = resolvedevice(training["device"])
    with Path(training["hyp"]).open("r", encoding="utf-8", errors="ignore") as stream:
        hyp = yaml.safe_load(stream)
    datadict = check_dataset(training["data"])
    model = loadmodel(training, hyp, datadict, device)
    trainer = MSDYOLOTrainer(model, config, device)
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
    for epoch in range(epochs):
        for batchindex, (images, targets, paths, shapes) in enumerate(dataloader):
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
        print(f"Epoch {epoch + 1}/{epochs}: loss={lastresult['loss'].item():.6f}")


if __name__ == "__main__":
    main()
