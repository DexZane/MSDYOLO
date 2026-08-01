"""MSDYOLO 连续小写键配置系统。"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class MSDYOLOConfig:
    """管理默认关闭、可序列化的训练与消融配置。"""

    def __init__(self, configpath=None):
        self.config = self.defaultconfig()
        if configpath is not None:
            self.loadfromfile(configpath)

    def defaultconfig(self):
        """返回显存友好的 baseline 默认配置。"""
        return {
            "experiment": {
                "name": "baseline",
                "description": "Original YOLOv5-OBB baseline",
                "phase": 1,
            },
            "training": {
                "data": "msdyolo/data/dota-test.yaml",
                "cfg": "configs/models/yolov5s.yaml",
                "weights": "",
                "epochs": 300,
                "batchsize": 2,
                "imagesize": 1024,
                "device": "0",
                "workers": 0,
                "hyp": "msdyolo/data/hyps/obb/hyp.finetune_dota.yaml",
            },
            "degradation": {
                "enabled": False,
                "psf": {
                    "enabled": False,
                    "kernelsize": 5,
                    "sigma": 1.0,
                },
                "downsample": {
                    "enabled": False,
                    "scale": 2.0,
                },
                "noise": {
                    "enabled": False,
                    "type": "gaussian",
                    "level": 0.01,
                },
                "upsamplemode": "bilinear",
                "seed": 42,
            },
            "clearbranch": {
                "enabled": False,
                "strategy": "evalmode",
                "extractsparse": True,
                "topk": 300,
                "confidencethreshold": 0.25,
            },
            "distillation": {
                "enabled": False,
                "alpha": 0.1,
                "topk": 300,
                "confidencethreshold": 0.25,
                "iouthreshold": 0.1,
                "distancethreshold": 2.0,
                "classtemperature": 2.0,
                "angletemperature": 2.0,
                "shortedgethreshold": 8.0,
            },
            "profiling": {
                "enabled": False,
                "loginterval": 100,
                "savestats": True,
                "outputdir": "runs/profiling",
            },
            "ablationmode": "baseline",
        }

    def loadfromfile(self, configpath):
        """从 YAML 文件加载配置。"""
        path = Path(configpath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            userconfig = yaml.safe_load(stream) or {}
        self.recursiveupdate(self.config, userconfig)
        logger.info("Loaded config from %s", path)

    def recursiveupdate(self, base, update):
        """递归合并嵌套字典。"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.recursiveupdate(base[key], value)
            else:
                base[key] = value

    def savetofile(self, outputpath):
        """把当前配置保存为 YAML。"""
        path = Path(outputpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.config, stream, sort_keys=False)

    def get(self, keypath, default=None):
        """读取点分隔路径。"""
        value = self.config
        for key in keypath.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def set(self, keypath, value):
        """写入点分隔路径。"""
        keys = keypath.split(".")
        current = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    def applyablationmode(self, mode):
        """应用 baseline、withdegradation、withclearbranch 或 full。"""
        if mode not in {"baseline", "withdegradation", "withclearbranch", "full"}:
            raise ValueError(f"Unknown ablation mode: {mode}")
        self.set("ablationmode", mode)
        self.set("degradation.enabled", mode != "baseline")
        self.set("clearbranch.enabled", mode in {"withclearbranch", "full"})
        self.set("distillation.enabled", mode == "full")
        if mode == "full":
            logger.info("Full mode: enabled degradation + clearbranch + distillation")

    def validate(self):
        """验证阶段边界和退化参数。"""
        errors = []
        phase = self.get("experiment.phase")
        if phase < 2 and self.get("distillation.enabled"):
            errors.append("Phase < 2 does not allow distillation")
        if self.get("distillation.enabled"):
            if not self.get("degradation.enabled"):
                errors.append("Distillation requires degradation.enabled=true")
            if not self.get("clearbranch.enabled"):
                errors.append("Distillation requires clearbranch.enabled=true")
        if self.get("clearbranch.enabled") and not self.get("degradation.enabled"):
            errors.append("Clear branch requires degradation")
        if self.get("degradation.downsample.enabled"):
            scale = self.get("degradation.downsample.scale")
            if scale <= 1:
                errors.append("Downsample scale must be greater than 1")
        if errors:
            for error in errors:
                logger.error(error)
            return False
        return True

    def __repr__(self):
        return yaml.safe_dump(self.config, sort_keys=False)
