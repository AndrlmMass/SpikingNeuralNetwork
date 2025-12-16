"""
Experiment definitions and orchestration for SNN research.

This module contains:
- mnist_family: Canonical MNIST family experiment (MNIST, KMNIST, FMNIST, NotMNIST)
- geomfig: Geometric figure classification experiment
"""

from .mnist_family import (
    MNIST_FAMILY_CONFIG,
    run_full_pipeline,
    run_snn_sleepy_experiment,
    run_snntorch_experiment,
    run_glmm_analysis,
)

__all__ = [
    "MNIST_FAMILY_CONFIG",
    "run_full_pipeline",
    "run_snn_sleepy_experiment",
    "run_snntorch_experiment",
    "run_glmm_analysis",
]





