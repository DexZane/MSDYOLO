from pathlib import Path

import cv2
import numpy as np
import pytest

from msdyolo.data.dota import parse_dota_label
from msdyolo.data.scripts.split_dota import split_dataset, split_single_image


class CheckDotaLabels:
    def checkheaderandheaderlesslabelsparse(self, tmp_path: Path):
        headed = tmp_path / "headed.txt"
        headed.write_text(
            "imagesource:GoogleEarth\ngsd:0.5\n"
            "100 200 300 200 300 400 100 400 ship 0\n",
            encoding="utf-8",
        )
        plain = tmp_path / "plain.txt"
        plain.write_text(
            "100 200 300 200 300 400 100 400 ship 0\n",
            encoding="utf-8",
        )
        assert parse_dota_label(headed) == parse_dota_label(plain)
        assert parse_dota_label(plain)[0].coordinates == (
            100.0, 200.0, 300.0, 200.0, 300.0, 400.0, 100.0, 400.0
        )

    @pytest.mark.parametrize(
        "line, message",
        [
            ("0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 ship 0", "normalized"),
            ("nan 2 3 4 5 6 7 8 ship 0", "finite"),
            ("1 2 3 4 5 6 7 8 unknown 0", "unknown class"),
            ("1 2 3 4 5 6 7 8 ship 9", "difficult"),
        ],
    )
    def checkinvalidlabelsfail(self, tmp_path: Path, line: str, message: str):
        label = tmp_path / "bad.txt"
        label.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            parse_dota_label(label)

    @pytest.mark.parametrize(
        "line",
        [
            "10 10 10 10 10 10 10 10 ship 0",
            "0 0 10 0 20 0 30 0 ship 0",
        ],
    )
    def checkdegeneratepolygonsfail(self, tmp_path: Path, line: str):
        label = tmp_path / "degenerate.txt"
        label.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="degenerate polygon"):
            parse_dota_label(label)

    def checksplitwritesclippedpixellabelstolabeltxt(self, tmp_path: Path):
        image = tmp_path / "image.png"
        assert cv2.imwrite(str(image), np.zeros((1200, 1200, 3), dtype=np.uint8))
        label = tmp_path / "image.txt"
        label.write_text(
            "900 900 1100 900 1100 1100 900 1100 ship 0\n",
            encoding="utf-8",
        )
        output = tmp_path / "split"

        assert split_single_image((image, label, output, 1024, 0)) == 1

        outputlabel = output / "labelTxt" / "image_0_0.txt"
        assert outputlabel.is_file()
        assert not (output / "labels").exists()
        coordinates = [float(value) for value in outputlabel.read_text(encoding="utf-8").split()[:8]]
        assert all(0.0 <= value <= 1024.0 for value in coordinates)
        assert max(coordinates) > 1.0

    @pytest.mark.parametrize(
        "subsize, gap, numprocess",
        [
            (1024, 1024, 1),
            (1024, -1, 1),
            (1024, 200, 0),
            (1025, 200, 1),
        ],
    )
    def checksplitdatasetrejectsinvalidarguments(
        self, tmp_path: Path, subsize: int, gap: int, numprocess: int
    ):
        with pytest.raises(ValueError):
            split_dataset(tmp_path / "images", None, tmp_path / "output", subsize, gap, numprocess)
