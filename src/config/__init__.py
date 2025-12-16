"""
Configuration module: Default parameters and experiment configs.

Contains:
- defaults: Default network and training parameters
- experiments: Pre-defined experiment configurations
"""

from .defaults import (
    DEFAULT_NETWORK_PARAMS,
    DEFAULT_TRAINING_PARAMS,
    DEFAULT_DATA_PARAMS,
    GEOMFIG_PARAMS,
)
from .experiments import (
    ALL_EXPERIMENTS,
    get_experiment,
    PAPER_GEOMFIG_EXPERIMENT,
    MNIST_BASELINE_EXPERIMENT,
    QUICK_TEST_EXPERIMENT,
)

__all__ = [
    'DEFAULT_NETWORK_PARAMS',
    'DEFAULT_TRAINING_PARAMS',
    'DEFAULT_DATA_PARAMS',
    'GEOMFIG_PARAMS',
    'ALL_EXPERIMENTS',
    'get_experiment',
    'PAPER_GEOMFIG_EXPERIMENT',
    'MNIST_BASELINE_EXPERIMENT',
    'QUICK_TEST_EXPERIMENT',
]

