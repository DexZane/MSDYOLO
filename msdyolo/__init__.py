"""
MSDYOLO: Multi-Scale Deformable YOLO for Oriented Object Detection
"""

__version__ = "1.0.0"

from .train import main as train
from .detect import main as detect

__all__ = ['train', 'detect']
