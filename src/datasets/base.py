"""
Base data loading functions.

This module provides core data loading and preprocessing utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .loaders import (
    load_image_batch,
    normalize_image,
)

__all__ = [
    'load_image_batch',
    'normalize_image',
]

