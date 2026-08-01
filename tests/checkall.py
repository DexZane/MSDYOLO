"""项目配置、模块边界和交付文件的烟雾测试。"""

from pathlib import Path

import yaml

from msdyolo.utils.config import MSDYOLOConfig


ROOT = Path(__file__).resolve().parents[1]


class CheckDelivery:

    def checkfourexperimentconfigsremain(self):
        """验证保留4个实验配置文件。"""
        names = {path.name for path in (ROOT / "configs" / "train").glob("*.yaml")}
        assert names == {
            "baseline.yaml",
            "degradation.yaml",
            "clearbranch.yaml",
            "full.yaml",
        }

    def checkallconfigsvalidate(self):
        for path in (ROOT / "configs" / "train").glob("*.yaml"):
            config = MSDYOLOConfig(path)
            assert config.validate(), path

    def checkbaselinedefaultsarelowmemory(self):
        config = MSDYOLOConfig(ROOT / "configs" / "train" / "baseline.yaml")
        assert config.get("training.batchsize") == 2
        assert config.get("training.imagesize") == 320
        assert config.get("training.workers") == 0
        assert not config.get("degradation.enabled")
        assert not config.get("clearbranch.enabled")
        assert not config.get("distillation.enabled")
        assert not config.get("profiling.enabled")

    def checkfullconfigenablesallmsdcomponentsforcloudtraining(self):
        config = MSDYOLOConfig(ROOT / "configs" / "train" / "full.yaml")
        assert config.get("degradation.enabled")
        assert config.get("clearbranch.enabled")
        assert config.get("distillation.enabled")
        assert config.get("training.workers") == 4
        assert config.get("training.data") == "msdyolo/data/dota.yaml"
        assert config.get("training.cfg") == "configs/models/yolov5s.yaml"
        assert config.get("training.weights") == "yolov5s.pt"
        assert config.get("training.epochs") == 200
        assert config.get("training.batchsize") == 16
        assert config.get("training.imagesize") == 1024
        assert config.get("training.device") == "0"

    def checknonbaselinemodesactivateimagedegradation(self):
        for name in ("degradation.yaml", "clearbranch.yaml", "full.yaml"):
            config = MSDYOLOConfig(ROOT / "configs" / "train" / name)
            assert config.get("degradation.psf.enabled"), name
            assert config.get("degradation.downsample.enabled"), name
            assert config.get("degradation.noise.enabled"), name

    def checkdotatestdatasetexists(self):
        dataconfig = yaml.safe_load((ROOT / "data" / "dota-test.yaml").read_text())
        dataroot = ROOT / dataconfig["path"]
        assert len(list((dataroot / "images").glob("test*.png"))) == 5
        assert len(list((dataroot / "labelTxt").glob("test*.txt"))) == 5


class CheckConfig:

    def checkconfigroundtrip(self, tmp_path):
        config = MSDYOLOConfig()
        config.set("training.batchsize", 3)
        path = tmp_path / "roundtrip.yaml"
        config.savetofile(path)
        restored = MSDYOLOConfig(path)
        assert restored.get("training.batchsize") == 3

    def checkinvalidablationfails(self):
        config = MSDYOLOConfig()
        try:
            config.applyablationmode("unknown")
        except ValueError:
            return
        raise AssertionError("Unknown ablation must fail")
