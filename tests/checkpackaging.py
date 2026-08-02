"""Behavioral checks for the installable MSDYOLO distribution."""

import importlib.metadata
import subprocess
import sys
import venv
import zipfile
from email.parser import Parser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def buildwheel(output: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    wheels = sorted(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def makevenv(path: Path, wheel: Path) -> Path:
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(path)
    executable = path / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    subprocess.run(
        [str(executable), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return executable


class CheckPackaging:

    def checkstandardmetadataandpackageassets(self, tmp_path: Path):
        wheel = buildwheel(tmp_path)
        with zipfile.ZipFile(wheel) as archive:
            members = set(archive.namelist())
            metadataname = next(name for name in members if name.endswith(".dist-info/METADATA"))
            metadata = Parser().parsestr(archive.read(metadataname).decode("utf-8"))

        assert metadata.get("License") in {"GPL-3.0-only", "GPL-3.0-or-later", "GPLv3"}
        assert any(
            value == "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
            for value in metadata.get_all("Classifier", [])
        )
        for asset in (
            "msdyolo/data/dior.yaml",
            "msdyolo/data/hyps/obb/hyp.finetune_dota.yaml",
            "msdyolo/utils/nms_rotated/src/nms_rotated_cpu.cpp",
            "msdyolo/utils/nms_rotated/src/nms_rotated_cuda.cu",
        ):
            assert asset in members, asset

    def checkconsolescriptsworkafterinstall(self, tmp_path: Path):
        wheel = buildwheel(tmp_path)
        executable = makevenv(tmp_path / "venv", wheel)
        scriptdir = executable.parent
        for command in ("msdyolo-train", "msdyolo-val", "msdyolo-detect", "msdyolo-export"):
            result = subprocess.run(
                [str(scriptdir / command), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            assert result.returncode == 0, (command, result.stderr)

        metadata = importlib.metadata.metadata("msdyolo")
        assert metadata["Name"] == "msdyolo"
