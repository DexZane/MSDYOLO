"""
MSDYOLO: Multi-Scale Deformable YOLO for Oriented Object Detection
"""

__version__ = "1.0.0"

from .compat import register_legacy_module_aliases

register_legacy_module_aliases()

from .train import main as train
from .detect import main as detect

__all__ = ['train', 'detect']
