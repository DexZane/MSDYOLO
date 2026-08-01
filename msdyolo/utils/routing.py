"""基于目标像素足迹、教师置信度和角度可靠性的路由。"""

import math
from dataclasses import dataclass

import torch

from msdyolo.utils.matching import predictionconfidence


@dataclass
class RoutingWeights:
    """四分量路由权重及其可解释中间量。"""

    classification: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    angle: torch.Tensor
    survival: torch.Tensor
    anglereliability: torch.Tensor

    @property
    def weights(self):
        return torch.stack(
            (self.classification, self.center, self.scale, self.angle),
            -1,
        )


def emptyrouting(reference):
    """创建与参考张量设备和类型一致的空路由。"""
    empty = reference.new_empty(0)
    return RoutingWeights(empty, empty, empty, empty, empty, empty)


def computerouting(
    targets,
    teacher,
    matches,
    psfsigma=0.0,
    downsamplefactor=1.0,
    noiselevel=0.0,
    shortedgethreshold=8.0,
):
    """计算 classification/center/scale/angle 的分量级可传递性。"""
    if downsamplefactor < 1.0:
        raise ValueError("downsamplefactor must be at least 1")
    if psfsigma < 0 or noiselevel < 0:
        raise ValueError("Degradation strengths must be non-negative")
    if shortedgethreshold <= 0:
        raise ValueError("shortedgethreshold must be positive")
    if len(matches) == 0:
        return emptyrouting(teacher.values)

    target = targets[matches.targetindex]
    teachervalues = teacher.values[matches.batchindex, matches.teacherindex]
    confidence = predictionconfidence(teachervalues).detach()

    effectiveshortedge = torch.minimum(target[:, 4], target[:, 5]) / downsamplefactor
    shortedgefactor = 1.0 - torch.exp(-effectiveshortedge.clamp_min(0) / shortedgethreshold)
    blurfactor = 1.0 / (1.0 + psfsigma)
    downsampleimpact = 1.0 / downsamplefactor
    noisefactor = 1.0 / (1.0 + 10.0 * noiselevel)
    sensorfactor = (blurfactor * downsampleimpact * noisefactor) ** (1.0 / 3.0)
    survival = (confidence * sensorfactor * shortedgefactor).clamp(0, 1)

    anglelogits = teachervalues[:, -180:]
    angleprobability = anglelogits.softmax(-1)
    entropy = -(angleprobability * angleprobability.clamp_min(1e-12).log()).sum(-1)
    entropyreliability = (1.0 - entropy / math.log(180.0)).clamp(0, 1)
    aspectratio = torch.maximum(
        target[:, 4] / target[:, 5].clamp_min(1e-6),
        target[:, 5] / target[:, 4].clamp_min(1e-6),
    )
    aspectreliability = (1.0 - torch.exp(-torch.abs(torch.log(aspectratio)))).clamp(0, 1)
    anglereliability = (entropyreliability * aspectreliability).clamp(0, 1)

    classification = survival.sqrt()
    center = survival
    scale = survival.square()
    angle = scale * anglereliability
    return RoutingWeights(
        classification,
        center,
        scale,
        angle,
        survival,
        anglereliability,
    )
