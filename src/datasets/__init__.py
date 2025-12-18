"""
Datasets module: Data loading and generation.

Contains:
- base: Core data loading functions
- mnist: MNIST and related datasets (KMNIST, FMNIST, NotMNIST)
- geomfig: Geometric figure generation
- loaders: ImageDataStreamer and GeomfigDataStreamer classes
"""

from .loaders import GeomfigDataStreamer, ImageDataStreamer

__all__ = ['GeomfigDataStreamer', 'ImageDataStreamer']





