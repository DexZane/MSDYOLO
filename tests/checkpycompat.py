"""Checks for the package's declared Python compatibility floor."""

from pathlib import Path

from msdyolo.data.scripts.prepare_dota import _is_relative_to

ROOT = Path(__file__).resolve().parents[1]


class CheckPycompat:

    def checktrainmoduledefersannotations(self):
        source = (ROOT / "msdyolo" / "train.py").read_text(encoding="utf-8")
        assert "from __future__ import annotations" in source

    def checkpreparerelativepathworkswithoutpathlib39(self):
        assert _is_relative_to(Path("/tmp/root/subtree"), Path("/tmp/root"))
        assert not _is_relative_to(Path("/tmp/other"), Path("/tmp/root"))
