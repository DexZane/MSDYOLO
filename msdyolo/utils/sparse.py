"""基础设施阶段使用的 eval 输出稀疏提取与中心距离匹配。"""

import math

import torch


class SparsePredictionExtractor:
    """从已解码的 YOLO eval tuple 中提取 Top-K 预测。"""

    def __init__(self, confidencethreshold=0.25, topk=300, numclasses=16):
        self.confidencethreshold = confidencethreshold
        self.topk = topk
        self.numclasses = numclasses

    def emptypredictions(self, device, dtype=torch.float32):
        """创建空稀疏预测。"""
        return {
            "boxes": torch.zeros((0, 5), device=device, dtype=dtype),
            "scores": torch.zeros(0, device=device, dtype=dtype),
            "classes": torch.zeros((0, self.numclasses), device=device, dtype=dtype),
            "classids": torch.zeros(0, dtype=torch.long, device=device),
            "validmask": torch.zeros(0, dtype=torch.bool, device=device),
        }

    def extractfromyolooutput(self, predictions, modeltraining=False, batchsize=1):
        """解读 eval 输出；训练 raw output 由 decodesparse 处理。"""
        if modeltraining:
            raise ValueError("Training raw output must use utils.decoder.decodesparse")
        decoded = predictions[0] if isinstance(predictions, tuple) else predictions
        if not isinstance(decoded, torch.Tensor) or decoded.ndim != 3:
            raise TypeError("Expected decoded YOLO predictions with shape (B, N, O)")
        if decoded.shape[0] != batchsize:
            raise ValueError("batchsize does not match decoded predictions")

        results = []
        classend = decoded.shape[-1] - 180
        for batchindex in range(batchsize):
            values = decoded[batchindex]
            objectness = values[:, 4]
            classes = values[:, 5:classend]
            classscores, classids = classes.max(-1)
            scores = objectness * classscores
            validindices = torch.where(scores > self.confidencethreshold)[0]
            if validindices.numel() == 0:
                results.append(self.emptypredictions(values.device, values.dtype))
                continue
            selected = min(self.topk, validindices.numel())
            ranking = scores[validindices].topk(selected).indices
            indices = validindices[ranking]
            angleindices = values[indices, classend:].argmax(-1)
            angles = (angleindices.to(values.dtype) - 90) * math.pi / 180
            boxes = torch.cat((values[indices, :4], angles.unsqueeze(-1)), -1)
            results.append(
                {
                    "boxes": boxes,
                    "scores": scores[indices],
                    "classes": classes[indices],
                    "classids": classids[indices],
                    "validmask": torch.ones(selected, dtype=torch.bool, device=values.device),
                }
            )
        return results


class PredictionMatcher:
    """旧基础设施的无类别或同类别中心距离一对一匹配。"""

    def __init__(self, matchthreshold=0.5, useclassfilter=True):
        self.matchthreshold = matchthreshold
        self.useclassfilter = useclassfilter

    def computecenterdistance(self, firstboxes, secondboxes):
        """返回两组中心的欧氏距离矩阵。"""
        return torch.cdist(firstboxes[:, :2], secondboxes[:, :2])

    def matchpredictions(self, clearpredictions, degradedpredictions):
        """按相似度贪心匹配且禁止重复索引。"""
        clearboxes = clearpredictions["boxes"]
        degradedboxes = degradedpredictions["boxes"]
        if clearboxes.numel() == 0 or degradedboxes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=clearboxes.device)
            return empty, empty
        similarity = 1 / (1 + self.computecenterdistance(clearboxes, degradedboxes))
        if self.useclassfilter:
            compatible = (
                clearpredictions["classids"].unsqueeze(1)
                == degradedpredictions["classids"].unsqueeze(0)
            )
            similarity = similarity * compatible

        clearindices = []
        degradedindices = []
        usedclear = set()
        useddegraded = set()
        columns = similarity.shape[1]
        for flatindex in similarity.flatten().argsort(descending=True).tolist():
            clearindex = flatindex // columns
            degradedindex = flatindex % columns
            if similarity[clearindex, degradedindex].item() < self.matchthreshold:
                break
            if clearindex in usedclear or degradedindex in useddegraded:
                continue
            clearindices.append(clearindex)
            degradedindices.append(degradedindex)
            usedclear.add(clearindex)
            useddegraded.add(degradedindex)
        return (
            torch.tensor(clearindices, dtype=torch.long, device=clearboxes.device),
            torch.tensor(degradedindices, dtype=torch.long, device=degradedboxes.device),
        )
