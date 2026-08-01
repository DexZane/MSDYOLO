import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from msdyolo.data.scripts.prepare_dota import (
    prepare_dataset,
    source_snapshot,
    validate_split_tree,
)


VALIDLABEL = "2 2 10 2 10 10 2 10 ship 0\n"


def writerawdataset(root: Path, realimages: bool = False) -> Path:
    dataset = root / "DOTA"
    trainimage = dataset / "train" / "images" / "train.png"
    trainimage.parent.mkdir(parents=True)
    valimage = dataset / "val" / "images" / "val.png"
    valimage.parent.mkdir(parents=True)
    if realimages:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        assert cv2.imwrite(str(trainimage), image)
        assert cv2.imwrite(str(valimage), image)
    else:
        trainimage.write_bytes(b"raw-train-image")
        valimage.write_bytes(b"raw-val-image")
    trainlabel = dataset / "train" / "labelTxt" / "train.txt"
    trainlabel.parent.mkdir(parents=True)
    trainlabel.write_text(VALIDLABEL, encoding="utf-8")
    return dataset


def writevalidsplit(outputdir: Path) -> None:
    images = outputdir / "images"
    labels = outputdir / "labelTxt"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "patch.png").write_bytes(b"split-image")
    (labels / "patch.txt").write_text(VALIDLABEL, encoding="utf-8")


def makefakesplitter(calls: list[tuple[object, ...]]):
    def fakesplit(imagedir, labeldir, outputdir, subsize, gap, numprocess):
        calls.append(
            (
                Path(imagedir),
                Path(labeldir),
                Path(outputdir),
                subsize,
                gap,
                numprocess,
            )
        )
        writevalidsplit(Path(outputdir))
        return 1

    return fakesplit


def treebytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class CheckDotaPrepare:
    @pytest.mark.parametrize(
        "choice",
        ["equal", "ancestor", "train", "val", "trainimages", "vallabels"],
    )
    def checkunsafeoutputpathsrejectbeforerawmutation(
        self, tmp_path: Path, choice: str
    ):
        dataset = writerawdataset(tmp_path)
        outputs = {
            "equal": dataset,
            "ancestor": dataset.parent,
            "train": dataset / "train",
            "val": dataset / "val",
            "trainimages": dataset / "train" / "images" / "prepared",
            "vallabels": dataset / "val" / "labelTxt" / "prepared",
        }
        before = treebytes(dataset)
        calls = []

        def forbidden(*args):
            calls.append(args)
            raise AssertionError("splitter must not run for unsafe output")

        with pytest.raises(ValueError, match="unsafe.*output"):
            prepare_dataset(
                dataset, outputs[choice], 16, 2, 1, splitter=forbidden
            )

        assert calls == []
        assert treebytes(dataset) == before
        assert not (dataset / "val" / "labelTxt").exists()
        assert not list(tmp_path.parent.glob(".split-candidate-*"))

    def checkresolvedrawrootancestorrejectsbeforemutation(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        externalroot = tmp_path / "mountedraw"
        externalroot.mkdir()
        target = externalroot / "train"
        (dataset / "train").rename(target)
        (dataset / "train").symlink_to(target, target_is_directory=True)
        before = treebytes(externalroot)
        calls = []

        def forbidden(*args):
            calls.append(args)
            raise AssertionError("splitter must not run for raw-root ancestor")

        with pytest.raises(ValueError, match="unsafe.*output"):
            prepare_dataset(dataset, externalroot, 16, 2, 1, splitter=forbidden)

        assert calls == []
        assert treebytes(externalroot) == before
        assert not (dataset / "val" / "labelTxt").exists()

    def checksafedatasetchildoutputremainsallowed(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        output = dataset / "split"

        status = prepare_dataset(
            dataset, output, 16, 2, 1, splitter=makefakesplitter([])
        )

        assert status.ready
        assert output.is_dir()

    def checkcompletionmarkercontrolsreuseandrebuild(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        output = tmp_path / "split"
        calls = []
        splitter = makefakesplitter(calls)

        first = prepare_dataset(dataset, output, 16, 2, 1, splitter=splitter)

        assert first.ready
        assert not first.reused
        assert len(calls) == 2
        marker = output / ".msdyolo-split.json"
        assert marker.is_file()

        second = prepare_dataset(dataset, output, 16, 2, 1, splitter=splitter)

        assert second.ready
        assert second.reused
        assert len(calls) == 2

        prepare_dataset(dataset, output, 16, 1, 1, splitter=splitter)
        assert len(calls) == 4

        label = dataset / "train" / "labelTxt" / "train.txt"
        stat = label.stat()
        os.utime(label, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        prepare_dataset(dataset, output, 16, 1, 1, splitter=splitter)
        assert len(calls) == 6

        (output / "train" / "images" / "patch.png").unlink()
        (output / "train" / "images" / ".DS_Store").write_bytes(b"metadata")
        prepare_dataset(dataset, output, 16, 1, 1, splitter=splitter)
        assert len(calls) == 8

        forced = prepare_dataset(dataset, output, 16, 1, 1, force=True, splitter=splitter)
        assert not forced.reused
        assert len(calls) == 10

    def checkmarkerrecordsdeterministicsourcesparametersandformat(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        output = tmp_path / "split"

        prepare_dataset(dataset, output, 16, 2, 1, splitter=makefakesplitter([]))

        marker = output / ".msdyolo-split.json"
        state = json.loads(marker.read_text(encoding="utf-8"))
        assert state == {
            "format_version": "dota-pixel-v1",
            "gap": 2,
            "source": source_snapshot(dataset),
            "subsize": 16,
        }
        assert marker.read_text(encoding="utf-8") == json.dumps(
            state, indent=2, sort_keys=True
        ) + "\n"
        assert state["source"]["counts"] == {
            "train_images": 1,
            "train_labels": 1,
            "val_images": 1,
        }
        assert {entry["path"] for entry in state["source"]["files"]} == {
            "train/images/train.png",
            "train/labelTxt/train.txt",
            "val/images/val.png",
        }
        assert all(
            set(entry) == {"mtime_ns", "path", "size"}
            for entry in state["source"]["files"]
        )

    def checkvalidateparsesalltrainlabelsandrequirespixelobjects(self, tmp_path: Path):
        output = tmp_path / "split"
        writevalidsplit(output / "train")
        (output / "val" / "images").mkdir(parents=True)
        (output / "train" / "labelTxt" / "bad.txt").write_text(
            "0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 ship 0\n", encoding="utf-8"
        )

        status = validate_split_tree(output, 16)

        assert not status.ready
        assert any("normalized" in error for error in status.errors)

    @pytest.mark.parametrize("missing", ["images", "labelTxt"])
    def checkvalidaterequiresnonemptytrainoutputs(self, tmp_path: Path, missing: str):
        output = tmp_path / "split"
        writevalidsplit(output / "train")
        (output / "val" / "images").mkdir(parents=True)
        for path in (output / "train" / missing).iterdir():
            path.unlink()

        status = validate_split_tree(output, 16)

        assert not status.ready
        assert status.errors

    def checkpreparevalidatesrawlabelsbeforesplitting(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        (dataset / "train" / "labelTxt" / "also-bad.txt").write_text(
            "0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 ship 0\n", encoding="utf-8"
        )
        calls = []

        with pytest.raises(ValueError, match="normalized"):
            prepare_dataset(
                dataset,
                tmp_path / "split",
                16,
                2,
                1,
                splitter=makefakesplitter(calls),
            )

        assert calls == []

    def checksplitfailurepreservesexistingtreebyteforbyte(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        output = tmp_path / "split"
        prepare_dataset(dataset, output, 16, 2, 1, splitter=makefakesplitter([]))
        before = treebytes(output)

        def failingsplit(imagedir, labeldir, outputdir, subsize, gap, numprocess):
            split = Path(outputdir)
            (split / "images").mkdir(parents=True)
            (split / "labelTxt").mkdir(parents=True)
            (split / "images" / "bad.png").write_bytes(b"bad-image")
            (split / "labelTxt" / "bad.txt").write_text(
                "0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 ship 0\n", encoding="utf-8"
            )
            raise RuntimeError("split failed")

        with pytest.raises(RuntimeError, match="split failed"):
            prepare_dataset(dataset, output, 16, 1, 1, splitter=failingsplit)

        assert treebytes(output) == before
        assert not list(tmp_path.glob(".split-candidate-*"))
        assert not (tmp_path / ".split-backup").exists()

    def checkreplacementfailurerestoresexistingtree(self, tmp_path: Path, monkeypatch):
        dataset = writerawdataset(tmp_path)
        output = tmp_path / "split"
        splitter = makefakesplitter([])
        prepare_dataset(dataset, output, 16, 2, 1, splitter=splitter)
        before = treebytes(output)
        originalrename = Path.rename

        def failcandidaterename(path: Path, target: Path):
            if path.name.startswith(".split-candidate-") and Path(target) == output:
                raise OSError("replacement failed")
            return originalrename(path, target)

        monkeypatch.setattr(Path, "rename", failcandidaterename)

        with pytest.raises(OSError, match="replacement failed"):
            prepare_dataset(dataset, output, 16, 1, 1, splitter=splitter)

        assert treebytes(output) == before
        assert not (tmp_path / ".split-backup").exists()
        assert not list(tmp_path.glob(".split-candidate-*"))

    def checkstalebackupisexpliciterrorwithoutchangingraworoutput(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path)
        output = tmp_path / "split"
        output.mkdir()
        (output / "existing").write_bytes(b"existing")
        backup = tmp_path / ".split-backup"
        backup.mkdir()
        (backup / "unknown").write_bytes(b"unknown")
        rawbefore = treebytes(dataset)
        outputbefore = treebytes(output)
        backupbefore = treebytes(backup)

        with pytest.raises(RuntimeError, match="stale backup.*recover"):
            prepare_dataset(
                dataset, output, 16, 2, 1, splitter=makefakesplitter([])
            )

        assert treebytes(dataset) == rawbefore
        assert treebytes(output) == outputbefore
        assert treebytes(backup) == backupbefore

    def checkclirebuildsthenreusesrealsyntheticdataset(self, tmp_path: Path):
        dataset = writerawdataset(tmp_path, realimages=True)
        output = tmp_path / "split"
        command = [
            sys.executable,
            "-m",
            "msdyolo.data.scripts.prepare_dota",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--subsize",
            "16",
            "--gap",
            "0",
            "--num-process",
            "1",
        ]

        first = subprocess.run(command, capture_output=True, text=True, check=False)
        second = subprocess.run(command, capture_output=True, text=True, check=False)

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert "rebuilt" in first.stdout.lower()
        assert "reused" in second.stdout.lower()
        assert "source" in first.stdout.lower()
        assert "split" in first.stdout.lower()

    def checkclireturnsnonzeroforinvalidrawdata(self, tmp_path: Path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "msdyolo.data.scripts.prepare_dota",
                "--dataset",
                str(tmp_path / "DOTA"),
                "--output",
                str(tmp_path / "split"),
                "--num-process",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "training images" in result.stderr.lower()
