"""项目配置、模块边界和交付文件的烟雾测试。"""

from pathlib import Path

import yaml

from utils.config import MSDYOLOConfig


ROOT = Path(__file__).resolve().parents[1]


class CheckDelivery:

    def checkonlythreeexperimentconfigsremain(self):
        names = {path.name for path in (ROOT / "configs").glob("*.yaml")}
        assert names == {
            "msdyolo-baseline.yaml",
            "msdyolo-degradation.yaml",
            "msdyolo-clearbranch.yaml",
        }

    def checkallconfigsvalidate(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            config = MSDYOLOConfig(path)
            assert config.validate(), path

    def checkbaselinedefaultsarelowmemory(self):
        config = MSDYOLOConfig(ROOT / "configs" / "msdyolo-baseline.yaml")
        assert config.get("training.batchsize") == 2
        assert config.get("training.imagesize") == 1024
        assert not config.get("degradation.enabled")
        assert not config.get("clearbranch.enabled")
        assert not config.get("distillation.enabled")
        assert not config.get("profiling.enabled")

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
