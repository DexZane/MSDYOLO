"""默认关闭新增功能时的 baseline 等价性测试。"""

from copy import deepcopy

import torch
import torch.nn as nn

from utils.config import MSDYOLOConfig
from utils.trainer import MSDYOLOTrainer


class BaselineModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv2d(3, 4, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(4)
        self.nc = 16

    def forward(self, images):
        return [self.batchnorm(self.convolution(images))]


def computeloss(predictions, targets):
    """构造依赖模型输出的确定性损失。"""
    loss = predictions[0].square().mean() + targets.sum() * 0
    return loss, loss.detach().repeat(4)


def runbaseline(model, images, targets):
    """直接执行原始 baseline。"""
    predictions = model(images)
    return computeloss(predictions, targets)


def runwrapped(model, images, targets):
    """通过配置包装器执行 baseline。"""
    config = MSDYOLOConfig()
    config.applyablationmode("baseline")
    trainer = MSDYOLOTrainer(model, config, torch.device("cpu"))
    return trainer.processbatch(images, targets, computeloss)


class CheckBaselineEquivalence:

    def checklossisequivalent(self):
        torch.manual_seed(4)
        directmodel = BaselineModel()
        wrappedmodel = deepcopy(directmodel)
        images = torch.randn(2, 3, 16, 16)
        targets = torch.zeros(0, 187)
        directloss, directitems = runbaseline(directmodel, images, targets)
        wrapped = runwrapped(wrappedmodel, images, targets)
        assert torch.equal(directloss, wrapped["loss"])
        assert torch.equal(directitems, wrapped["lossitems"])

    def checkgradientsareequivalent(self):
        torch.manual_seed(5)
        directmodel = BaselineModel()
        wrappedmodel = deepcopy(directmodel)
        images = torch.randn(2, 3, 16, 16)
        targets = torch.zeros(0, 187)
        directloss, directitems = runbaseline(directmodel, images, targets)
        del directitems
        directloss.backward()
        wrapped = runwrapped(wrappedmodel, images, targets)
        wrapped["loss"].backward()
        for directparameter, wrappedparameter in zip(
            directmodel.parameters(),
            wrappedmodel.parameters(),
        ):
            assert torch.equal(directparameter.grad, wrappedparameter.grad)

    def checkbatchnormbuffersareequivalent(self):
        torch.manual_seed(6)
        directmodel = BaselineModel()
        wrappedmodel = deepcopy(directmodel)
        images = torch.randn(2, 3, 16, 16)
        targets = torch.zeros(0, 187)
        runbaseline(directmodel, images, targets)
        runwrapped(wrappedmodel, images, targets)
        assert torch.equal(
            directmodel.batchnorm.running_mean,
            wrappedmodel.batchnorm.running_mean,
        )
        assert torch.equal(
            directmodel.batchnorm.running_var,
            wrappedmodel.batchnorm.running_var,
        )
