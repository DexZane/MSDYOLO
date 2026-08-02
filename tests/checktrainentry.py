"""Regression checks for the real training entry point."""

import pytest

from msdyolo.train import trainingcheckpointdirectory, validateteacherweights, validatepaths
from msdyolo.utils.config import MSDYOLOConfig


class CheckTrainEntry:

    def checkvalidatepathsacceptsvalidfiles(self, tmp_path):
        paths = {}
        for key in ("cfg", "data", "hyp"):
            path = tmp_path / f"{key}.yaml"
            path.write_text("{}\n", encoding="utf-8")
            paths[key] = str(path)
        paths["weights"] = ""

        validatepaths(paths)

    def checkfulltrainingrequiresanexistingteachercheckpoint(self, tmp_path):
        training = {"teacherweights": ""}

        with pytest.raises(ValueError, match="DOTA teacher checkpoint"):
            validateteacherweights(training, True)

        missing = tmp_path / "missing.pt"
        training["teacherweights"] = str(missing)
        with pytest.raises(FileNotFoundError, match="teacherweights"):
            validateteacherweights(training, True)

    def checkteachercheckpointsuseitsnamedrundirectory(self):
        config = MSDYOLOConfig("configs/train/teacher.yaml")

        assert trainingcheckpointdirectory(config) == "runs/train/dota_teacher/weights"
