from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def writefakepython(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$SETUP_CALL_LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def writefakepkill(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "printf 'pkill %s\\n' \"$*\" >> \"$SETUP_CALL_LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setupskeleton(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "setup.sh", project / "scripts" / "setup.sh")
    (project / "configs" / "train").mkdir(parents=True)
    (project / "configs" / "train" / "full.yaml").write_text("training: {}\n")
    for relative in (
        "dataset/DOTA/train/images",
        "dataset/DOTA/train/labelTxt",
        "dataset/DOTA/val/images",
    ):
        (project / relative).mkdir(parents=True)
    (project / "yolov5s.pt").write_bytes(b"weights")

    binaries = tmp_path / "bin"
    binaries.mkdir()
    fakepython = binaries / "python"
    writefakepython(fakepython)
    writefakepkill(binaries / "pkill")
    calls = tmp_path / "calls.txt"
    environment = os.environ | {
        "PYTHON_BIN": str(fakepython),
        "SETUP_CALL_LOG": str(calls),
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
    }
    return project, environment, calls


class CheckSetup:
    def checkhelpdescribesallpublicflagswithoutinstalling(self):
        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        for flag in ("--prepare-only", "--force-resplit", "--foreground", "--config"):
            assert flag in result.stdout

    def checkprepareonlyusesorderedmodulesanddoesnotkillprocesses(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)

        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--prepare-only"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        recorded = calls.read_text(encoding="utf-8").splitlines()
        assert recorded[:3] == [
            "-m pip install -q setuptools==69.5.1",
            "-m pip install -q -e .",
            "-m pip install -q openxlab",
        ]
        assert any(line.startswith("-m msdyolo.data.scripts.prepare_dota ") for line in recorded)
        assert not any(line.startswith("-m msdyolo.data.scripts.download_dota ") for line in recorded)
        assert not any("--force-resplit" in line for line in recorded)
        assert not any(line.startswith("pkill ") for line in recorded)

    def checkdefaultlaunchusesfullconfigandforceflagonlywhenrequested(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)

        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--force-resplit"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        recorded = calls.read_text(encoding="utf-8").splitlines()
        prepare = next(
            line for line in recorded if line.startswith("-m msdyolo.data.scripts.prepare_dota ")
        )
        assert "--force-resplit" in prepare
        assert "-m msdyolo.train --config configs/train/full.yaml" in recorded
        assert not any(line.startswith("pkill ") for line in recorded)

    def checklivepidrefuseslaunchwithoutterminatingexistingprocess(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            pidfile = project / "runs" / "setup" / "training.pid"
            pidfile.parent.mkdir(parents=True)
            pidfile.write_text(f"{sleeper.pid}\n", encoding="utf-8")

            result = subprocess.run(
                ["bash", "scripts/setup.sh"],
                cwd=project,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            assert result.returncode == 1
            assert str(sleeper.pid) in result.stderr
            assert sleeper.poll() is None
            assert not any(line.startswith("pkill ") for line in calls.read_text().splitlines())
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

    def checkshellsyntaxisvalid(self):
        result = subprocess.run(
            ["bash", "-n", "scripts/setup.sh"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
