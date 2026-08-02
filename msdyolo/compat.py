"""Compatibility helpers for checkpoints serialized by upstream YOLOv5."""

from __future__ import annotations

import importlib
import sys


def register_legacy_module_aliases() -> None:
    """Map legacy ``models.*`` pickle globals to the canonical package modules.

    Official YOLOv5 checkpoints store classes as ``models.common.*`` and
    ``models.yolo.*``.  MSDYOLO keeps the implementation under ``msdyolo``;
    registering these aliases lets ``torch.load(..., weights_only=False)``
    restore the upstream model without reintroducing a duplicate source tree.
    """

    modules = {
        "models": "msdyolo.models",
        "models.common": "msdyolo.models.common",
        "models.yolo": "msdyolo.models.yolo",
    }
    for legacyname, canonicalname in modules.items():
        sys.modules.setdefault(legacyname, importlib.import_module(canonicalname))
