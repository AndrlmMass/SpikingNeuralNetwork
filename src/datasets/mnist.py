"""
MNIST and related datasets.

This module provides utilities for loading MNIST, KMNIST, and FMNIST.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export image loading functionality
from .loaders import (
    load_image_batch,
)

# Dataset identifiers
SUPPORTED_DATASETS = ['mnist', 'kmnist', 'fmnist']

__all__ = [
    'load_image_batch',
    'SUPPORTED_DATASETS',
]

