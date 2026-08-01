from pathlib import Path

import pytest

from msdyolo.data.scripts.download_dota import (
    download_dota_sdk,
    normalize_download_layout,
    verify_dataset,
)


class CheckDotaDownload:
    def checknormalizenestedtraindatalabelsandacceptsunlabelledvalidation(
        self, tmp_path: Path
    ):
        dataset = tmp_path / "DOTA"
        (dataset / "train" / "images").mkdir(parents=True)
        (dataset / "train" / "images" / "train-image.png").write_bytes(b"image")
        nestedlabels = dataset / "train" / "labelTxt" / "DOTA-v1.5_train"
        nestedlabels.mkdir(parents=True)
        (nestedlabels / "train-image.txt").write_text(
            "0 0 10 0 10 10 0 10 ship 0\n", encoding="utf-8"
        )
        (dataset / "val" / "images").mkdir(parents=True)
        (dataset / "val" / "images" / "val-image.png").write_bytes(b"image")

        status = normalize_download_layout(dataset)

        assert status.ready
        assert (dataset / "train" / "labelTxt" / "train-image.txt").is_file()
        assert not nestedlabels.exists()
        assert (dataset / "val" / "labelTxt").is_dir()
        assert status.trainimages == 1
        assert status.trainlabels == 1
        assert status.valimages == 1

    def checkmissingtrainlabelsreportsnotready(self, tmp_path: Path):
        dataset = tmp_path / "DOTA"
        (dataset / "train" / "images").mkdir(parents=True)
        (dataset / "train" / "images" / "train-image.png").write_bytes(b"image")
        (dataset / "val" / "images").mkdir(parents=True)
        (dataset / "val" / "images" / "val-image.png").write_bytes(b"image")

        status = verify_dataset(dataset)

        assert not status.ready
        assert status.trainlabels == 0
        assert status.errors

    def checknormalizationrefusestodifferentexistinglabel(self, tmp_path: Path):
        dataset = tmp_path / "DOTA"
        targetlabels = dataset / "train" / "labelTxt"
        nestedlabels = targetlabels / "DOTA-v1.5_train"
        nestedlabels.mkdir(parents=True)
        (targetlabels / "duplicate.txt").write_text("existing\n", encoding="utf-8")
        (nestedlabels / "duplicate.txt").write_text("incoming\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="duplicate.txt"):
            normalize_download_layout(dataset)

        assert (targetlabels / "duplicate.txt").read_text(encoding="utf-8") == "existing\n"
        assert (nestedlabels / "duplicate.txt").read_text(encoding="utf-8") == "incoming\n"

    def checknormalizationleavesunrecognizednestedlabeldirectoryuntouched(
        self, tmp_path: Path
    ):
        dataset = tmp_path / "DOTA"
        unknown = dataset / "train" / "labelTxt" / "other-source"
        unknown.mkdir(parents=True)
        (unknown / "train-image.txt").write_text("label\n", encoding="utf-8")

        normalize_download_layout(dataset)

        assert (unknown / "train-image.txt").is_file()
        assert not (dataset / "train" / "labelTxt" / "train-image.txt").exists()

    def checkdownloadinvokessdkwithexactrepositoryarguments(self, tmp_path: Path):
        captured = {}
        target = tmp_path / "DOTA"

        def fakedownload(**kwargs):
            captured.update(kwargs)

        assert download_dota_sdk(target, download_fn=fakedownload)
        assert captured == {
            "dataset_repo": "OpenDataLab/DOTA_V1_dot_5",
            "source_path": "",
            "target_path": str(target),
        }
