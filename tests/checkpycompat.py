"""Checks for the package's declared Python compatibility floor."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CheckPycompat:

    def checktrainmoduledefersannotations(self):
        source = (ROOT / "msdyolo" / "train.py").read_text(encoding="utf-8")
        assert "from __future__ import annotations" in source
