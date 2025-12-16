"""
Datasets module: Data loading and generation.

Contains:
- base: Core data loading functions
- mnist: MNIST and related datasets (KMNIST, FMNIST, NotMNIST)
- geomfig: Geometric figure generation
"""

from .base import load_image_batch
from .loaders import GeomfigDataStreamer, create_geomfig_data

__all__ = ['load_image_batch', 'GeomfigDataStreamer', 'create_geomfig_data']





