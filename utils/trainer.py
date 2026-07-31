"""MSDYOLO 训练器：整合退化、清晰分支和四分量蒸馏。"""

import logging

import torch
import torch.nn as nn

from utils.clearbranch import teacherforward
from utils.decoder import decodesparse
from utils.degradation import ImageDegradation
from utils.distillation import computefourcomponentloss
from utils.matching import matchpredictions
from utils.routing import computerouting

logger = logging.getLogger(__name__)


class MSDYOLOTrainer:
    """管理 MSDYOLO 完整训练流程的训练器。"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        if not config.validate():
            raise ValueError("Invalid configuration")

        self.degradationenabled = config.get("degradation.enabled", False)
        self.clearbranchenabled = config.get("clearbranch.enabled", False)
        self.distillationenabled = config.get("distillation.enabled", False)

        # 蒸馏参数
        self.alpha = config.get("distillation.alpha", 0.1)
        self.topk = config.get("distillation.topk", 300)
        self.confidencethreshold = config.get("distillation.confidencethreshold", 0.25)
        self.iouthreshold = config.get("distillation.iouthreshold", 0.1)
        self.distancethreshold = config.get("distillation.distancethreshold", 2.0)
        self.classtemperature = config.get("distillation.classtemperature", 2.0)
        self.angletemperature = config.get("distillation.angletemperature", 2.0)
        self.shortedgethreshold = config.get("distillation.shortedgethreshold", 8.0)

        # 退化模块
        self.degradation = None
        if self.degradationenabled:
            self.degradation = ImageDegradation(
                enablepsf=config.get("degradation.psf.enabled"),
                enabledownsample=config.get("degradation.downsample.enabled"),
                enablenoise=config.get("degradation.noise.enabled"),
                psfkernelsize=config.get("degradation.psf.kernelsize"),
                psfsigma=config.get("degradation.psf.sigma"),
                downsamplescale=config.get("degradation.downsample.scale"),
                noisetype=config.get("degradation.noise.type"),
                noiselevel=config.get("degradation.noise.level"),
                upsamplemode=config.get("degradation.upsamplemode"),
                seed=config.get("degradation.seed"),
            ).to(device)

        # 退化参数（用于路由计算）
        self.psfsigma = 0.0
        self.downsamplefactor = 1.0
        self.noiselevel = 0.0
        if self.degradationenabled:
            if config.get("degradation.psf.enabled"):
                self.psfsigma = config.get("degradation.psf.sigma", 1.0)
            if config.get("degradation.downsample.enabled"):
                self.downsamplefactor = config.get("degradation.downsample.scale", 2.0)
            if config.get("degradation.noise.enabled"):
                self.noiselevel = config.get("degradation.noise.level", 0.01)

        self.validatedependencies()

    def validatedependencies(self):
        """验证蒸馏的前置条件。"""
        if self.distillationenabled:
            if not self.degradationenabled:
                raise ValueError("Distillation requires degradation.enabled=true")
            if not self.clearbranchenabled:
                raise ValueError("Distillation requires clearbranch.enabled=true")

    def processbatch(self, images, targets, computeloss):
        """执行完整的训练批次处理。

        返回:
            dict: 包含以下键：
                - loss: 总损失（检测+蒸馏）
                - detectionloss: YOLOv5-OBB ComputeLoss
                - distillationloss: 四分量蒸馏总损失
                - classificationloss: 分类分量损失
                - centerloss: 中心分量损失
                - scaleloss: 尺度分量损失
                - angleloss: 角度分量损失
                - matchcount: 匹配数量
                - meansurvival: 平均存活度
                - meananglereliability: 平均角度可靠性
                - lossitems: 损失明细
                - predictions: 学生前向输出
        """
        imagesize = images.shape[-1]

        # 如果不启用蒸馏，退化为标准YOLO训练
        if not self.distillationenabled:
            return self.baselineforward(images, targets, computeloss, imagesize)

        # 阶段1：清晰分支教师前向（eval + no_grad）
        teacher = teacherforward(self.model, images, self.topk)

        # 阶段2：退化视图学生前向（train模式）
        self.model.train()
        degradedimages = images
        if self.degradation is not None:
            degradedimages = self.degradation(images)

        studentraw = self.model(degradedimages)  # 训练模式返回raw outputs list

        # 阶段3：计算检测损失（ComputeLoss直接接受raw outputs）
        detectionloss = computeloss(studentraw, targets)
        if isinstance(detectionloss, tuple):
            detectionloss = detectionloss[0]

        # 阶段4：稀疏解码学生输出（保持梯度）
        student = decodesparse(studentraw, self.model, self.topk)

        # 阶段5：一对一匹配
        matches = matchpredictions(
            student,
            teacher,
            targets,
            confidencethreshold=self.confidencethreshold,
            iouthreshold=self.iouthreshold,
            distancethreshold=self.distancethreshold,
        )

        # 阶段6：计算路由权重
        routing = computerouting(
            targets,
            teacher,
            matches,
            psfsigma=self.psfsigma,
            downsamplefactor=self.downsamplefactor,
            noiselevel=self.noiselevel,
            shortedgethreshold=self.shortedgethreshold,
        )

        # 阶段7：计算四分量蒸馏损失
        componentlosses = computefourcomponentloss(
            student,
            teacher,
            matches,
            routing,
            imagesize,
            classtemperature=self.classtemperature,
            angletemperature=self.angletemperature,
        )

        # 阶段8：合并总损失
        distillationloss = componentlosses["total"]
        totalloss = detectionloss + self.alpha * distillationloss

        # 计算统计量
        matchcount = len(matches)
        meansurvival = routing.survival.mean().item() if matchcount > 0 else 0.0
        meananglereliability = routing.anglereliability.mean().item() if matchcount > 0 else 0.0

        return {
            "loss": totalloss,
            "detectionloss": detectionloss,
            "distillationloss": distillationloss,
            "classificationloss": componentlosses["classification"],
            "centerloss": componentlosses["center"],
            "scaleloss": componentlosses["scale"],
            "angleloss": componentlosses["angle"],
            "matchcount": matchcount,
            "meansurvival": meansurvival,
            "meananglereliability": meananglereliability,
            "lossitems": detectionloss.detach(),
            "predictions": studentraw,  # 返回raw outputs用于后续处理
        }

    def baselineforward(self, images, targets, computeloss, imagesize):
        """Baseline前向：仅检测损失，无蒸馏。"""
        self.model.train()
        degradedimages = images
        if self.degradation is not None:
            degradedimages = self.degradation(images)

        predictions = self.model(degradedimages)  # 训练模式返回raw outputs list

        detectionloss = computeloss(predictions, targets)
        if isinstance(detectionloss, tuple):
            detectionloss, lossitems = detectionloss
        else:
            lossitems = detectionloss.detach()

        zero = detectionloss * 0.0

        return {
            "loss": detectionloss,
            "detectionloss": detectionloss,
            "distillationloss": zero,
            "classificationloss": zero,
            "centerloss": zero,
            "scaleloss": zero,
            "angleloss": zero,
            "matchcount": 0,
            "meansurvival": 0.0,
            "meananglereliability": 0.0,
            "lossitems": lossitems,
            "predictions": predictions,
        }

    def isbaselinemode(self):
        """确认所有新增训练行为均关闭。"""
        return (
            not self.degradationenabled
            and not self.clearbranchenabled
            and not self.distillationenabled
        )
