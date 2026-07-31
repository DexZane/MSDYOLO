"""配置驱动的 MSDYOLO 基础训练包装器。"""

import logging

from utils.clearbranch import ClearBranchForward
from utils.degradation import ImageDegradation
from utils.profiler import MemoryProfiler
from utils.sparse import SparsePredictionExtractor

logger = logging.getLogger(__name__)


class MSDYOLOTrainer:
    """集成退化和清晰分支；P0-A.1 蒸馏组件尚未接入。"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        if not config.validate():
            raise ValueError("Invalid configuration")
        self.degradation = None
        self.clearbranch = None
        self.sparseextractor = None
        self.profiler = None
        self.setupmodules()

    def setupmodules(self):
        """按配置创建基础设施模块。"""
        if self.config.get("degradation.enabled"):
            self.degradation = ImageDegradation(
                enablepsf=self.config.get("degradation.psf.enabled"),
                enabledownsample=self.config.get("degradation.downsample.enabled"),
                enablenoise=self.config.get("degradation.noise.enabled"),
                psfkernelsize=self.config.get("degradation.psf.kernelsize"),
                psfsigma=self.config.get("degradation.psf.sigma"),
                downsamplescale=self.config.get("degradation.downsample.scale"),
                noisetype=self.config.get("degradation.noise.type"),
                noiselevel=self.config.get("degradation.noise.level"),
                upsamplemode=self.config.get("degradation.upsamplemode"),
                seed=self.config.get("degradation.seed"),
            ).to(self.device)
        if self.config.get("clearbranch.enabled"):
            self.clearbranch = ClearBranchForward(
                self.model,
                strategy=self.config.get("clearbranch.strategy"),
            )
            self.sparseextractor = SparsePredictionExtractor(
                confidencethreshold=self.config.get("clearbranch.confidencethreshold"),
                topk=self.config.get("clearbranch.topk"),
                numclasses=getattr(self.model, "nc", 16),
            )
        if self.config.get("profiling.enabled"):
            self.profiler = MemoryProfiler(self.device)

    def processbatch(self, images, targets, computeloss):
        """处理一个 batch，并保持 baseline 路径等价。"""
        degradedimages = self.degradation(images) if self.degradation is not None else images
        result = {}

        if self.clearbranch is not None and self.sparseextractor is not None:
            result["clearpredictions"] = self.clearbranch.forwardclearbranch(
                images,
                extractsparse=True,
                sparseextractor=self.sparseextractor,
            )
            predictions, lossoutput = self.clearbranch.forwarddegradedbranch(
                degradedimages,
                targets=targets,
                computeloss=computeloss,
            )
        else:
            predictions = self.model(degradedimages)
            lossoutput = computeloss(predictions, targets)

        if isinstance(lossoutput, tuple):
            loss, lossitems = lossoutput
        else:
            loss, lossitems = lossoutput, None
        result["loss"] = loss
        result["lossitems"] = lossitems
        result["predictions"] = predictions

        if self.config.get("distillation.enabled"):
            raise RuntimeError("P0-A.1 components are not integrated into MSDYOLOTrainer")
        return result

    def isbaselinemode(self):
        """确认所有新增训练行为均关闭。"""
        return (
            not self.config.get("degradation.enabled")
            and not self.config.get("clearbranch.enabled")
            and not self.config.get("distillation.enabled")
        )
