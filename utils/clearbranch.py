"""清晰视图的串行无梯度前向。"""

import torch
import torch.nn as nn

from utils.decoder import SparsePredictions, decodesparse


def moduletrainingstates(model):
    """记录每个模块的训练状态。"""
    return {name: module.training for name, module in model.named_modules()}


def restoretrainingstates(model, states):
    """逐模块恢复训练状态，避免覆盖混合 train/eval 配置。"""
    for name, module in model.named_modules():
        if name in states:
            module.train(states[name])


def teacherforward(model, cleanimages, topk=100):
    """以 eval 和 no_grad 执行清晰分支并返回完全分离的稀疏预测。"""
    states = moduletrainingstates(model)
    try:
        model.eval()
        with torch.no_grad():
            decodedoutput, teacherraw = model(cleanimages)
            del decodedoutput
            sparse = decodesparse(teacherraw, model, topk)
            detached = SparsePredictions(
                *(tensor.detach() for tensor in sparse.tensors())
            )
            del teacherraw, sparse
            return detached
    finally:
        restoretrainingstates(model, states)


class ClearBranchForward:
    """管理同一模型的清晰无梯度分支和退化有梯度分支。"""

    def __init__(self, model: nn.Module, strategy: str = "evalmode"):
        if strategy != "evalmode":
            raise ValueError(f"Only 'evalmode' is supported, got: {strategy}")
        self.model = model
        self.strategy = strategy

    def forwardclearbranch(self, images, extractsparse=True, sparseextractor=None):
        """执行清晰分支；旧基础设施可选择稀疏字典输出。"""
        states = moduletrainingstates(self.model)
        try:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
                if extractsparse and sparseextractor is not None:
                    sparsepredictions = sparseextractor.extractfromyolooutput(
                        predictions,
                        modeltraining=False,
                        batchsize=images.shape[0],
                    )
                    del predictions
                    for prediction in sparsepredictions:
                        for key, value in prediction.items():
                            if isinstance(value, torch.Tensor):
                                prediction[key] = value.detach()
                    return sparsepredictions
                if isinstance(predictions, tuple):
                    decoded, rawoutputs = predictions
                    return decoded.detach(), [raw.detach() for raw in rawoutputs]
                if isinstance(predictions, list):
                    return [prediction.detach() for prediction in predictions]
                return predictions.detach()
        finally:
            restoretrainingstates(self.model, states)

    def forwarddegradedbranch(self, images, targets=None, computeloss=None):
        """执行保留梯度的退化分支。"""
        self.model.train()
        predictions = self.model(images)
        loss = None
        if targets is not None and computeloss is not None:
            loss = computeloss(predictions, targets)
        return predictions, loss
