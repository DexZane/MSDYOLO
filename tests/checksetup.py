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


def writefakecurl(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "printf 'curl %s\\n' \"$*\" >> \"$SETUP_CALL_LOG\"\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "    if [ \"$1\" = -o ]; then\n"
        "        printf weights > \"$2\"\n"
        "        exit 0\n"
        "    fi\n"
        "    shift\n"
        "done\n"
        "exit 2\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setupskeleton(tmp_path: Path, weights: bool = True) -> tuple[Path, dict[str, str], Path]:
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
    if weights:
        (project / "yolov5s.pt").write_bytes(b"weights")

    binaries = tmp_path / "bin"
    binaries.mkdir()
    fakepython = binaries / "python"
    writefakepython(fakepython)
    writefakepkill(binaries / "pkill")
    writefakecurl(binaries / "curl")
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

    def checkfreshlaunchdownloadsweightsbeforetraining(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path, weights=False)

        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--foreground"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        recorded = calls.read_text(encoding="utf-8").splitlines()
        curlindex = next(index for index, line in enumerate(recorded) if line.startswith("curl "))
        trainindex = recorded.index("-m msdyolo.train --config configs/train/full.yaml")
        assert curlindex < trainindex
        assert (project / "yolov5s.pt").read_bytes() == b"weights"

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
        assert (project / "training.log").is_file()

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
            assert not calls.exists()
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

    def checkprepareonlycanrunwhiletrainingpidisstilllive(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            pidfile = project / "runs" / "setup" / "training.pid"
            pidfile.parent.mkdir(parents=True)
            pidfile.write_text(f"{sleeper.pid}\n", encoding="utf-8")

            result = subprocess.run(
                ["bash", "scripts/setup.sh", "--prepare-only"],
                cwd=project,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            assert result.returncode == 0, result.stderr
            assert sleeper.poll() is None
            assert any(
                line.startswith("-m msdyolo.data.scripts.prepare_dota ")
                for line in calls.read_text().splitlines()
            )
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

    def checkconfigrequiresarealpathbeforeanysideeffects(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)

        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--config", "--foreground"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert "requires a path" in result.stderr
        assert not calls.exists()

    def checkcustomconfigwithspacesreachesthetrainingcommand(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)
        config = "configs/train/custom config.yaml"
        (project / config).write_text("training: {}\n", encoding="utf-8")

        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--foreground", "--config", config],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert f"-m msdyolo.train --config {config}" in calls.read_text().splitlines()

    def checkmissingconfigrefusesbeforeanysideeffects(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)
        (project / "configs" / "train" / "full.yaml").unlink()

        result = subprocess.run(
            ["bash", "scripts/setup.sh"],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert "config file not found" in result.stderr
        assert not calls.exists()

    def checklivelockrefuseslaunchbeforepreparation(self, tmp_path: Path):
        project, environment, calls = setupskeleton(tmp_path)
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            lockdir = project / "runs" / "setup" / ".launch.lock"
            lockdir.mkdir(parents=True)
            (lockdir / "owner.pid").write_text(f"{sleeper.pid}\n", encoding="utf-8")

            result = subprocess.run(
                ["bash", "scripts/setup.sh", "--force-resplit"],
                cwd=project,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            assert result.returncode == 1
            assert str(sleeper.pid) in result.stderr
            assert sleeper.poll() is None
            assert not calls.exists()
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
