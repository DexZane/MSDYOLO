"""Behavioral checks for the single canonical ``msdyolo`` implementation tree."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "msdyolo"


def runpython(*arguments):
    """Run an isolated Python command from the repository root."""
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )


class CheckCanonicalImports:

    def checkmoduleandrootcommandhelpareavailable(self):
        commands = [
            ("-m", "msdyolo.train", "--help"),
            ("-m", "msdyolo.val", "--help"),
            ("-m", "msdyolo.detect", "--help"),
            ("-m", "msdyolo.export", "--help"),
            ("train.py", "--help"),
            ("val.py", "--help"),
            ("detect.py", "--help"),
            ("export.py", "--help"),
        ]
        for command in commands:
            result = runpython(*command)
            assert result.returncode == 0, result.stderr

    def checkprojectconsumersresolveinsidecanonicalpackage(self):
        result = runpython(
            "-c",
            """
import sys
from pathlib import Path

from msdyolo.detect import run as detectrun
from msdyolo.export import run as exportrun
from msdyolo.models.yolo import Model
from msdyolo.utils.datasets import img2label_paths
from msdyolo.utils.matching import matchpredictions
from msdyolo.utils.rotatednms import rotatednms
from msdyolo.utils.trainer import MSDYOLOTrainer
from msdyolo.val import run as validationrun

package = Path.cwd() / "msdyolo"
symbols = (
    Model,
    img2label_paths,
    MSDYOLOTrainer,
    matchpredictions,
    rotatednms,
    detectrun,
    exportrun,
    validationrun,
)
for symbol in symbols:
    source = Path(sys.modules[symbol.__module__].__file__).resolve()
    assert package in source.parents, source

model = Model("configs/models/yolov5n.yaml", nc=16)
assert model.model[-1].nc == 16
assert img2label_paths(["dataset/train/images/example.png"]) == [
    "dataset/train/labelTxt/example.txt"
]
""",
        )
        assert result.returncode == 0, result.stderr

    def checkrootcommandsdelegateintopackageimplementation(self):
        result = runpython(
            "-c",
            """
from detect import main as detectmain
from export import main as exportmain
from train import main as trainmain
from val import main as valmain

assert trainmain.__module__ == "msdyolo.train"
assert valmain.__module__ == "msdyolo.val"
assert detectmain.__module__ == "msdyolo.detect"
assert exportmain.__module__ == "msdyolo.export"
""",
        )
        assert result.returncode == 0, result.stderr

    def checkdefaulttrainingpathsusecanonicalpackageassets(self):
        result = runpython(
            "-c",
            """
from msdyolo.utils.config import MSDYOLOConfig

training = MSDYOLOConfig().config["training"]
assert training["data"] == "msdyolo/data/dota-test.yaml"
assert training["cfg"] == "configs/models/yolov5s.yaml"
assert training["hyp"] == "msdyolo/data/hyps/obb/hyp.finetune_dota.yaml"
""",
        )
        assert result.returncode == 0, result.stderr
