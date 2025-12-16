"""
Analysis module for GLMM statistical analysis.

This module contains:
- R script (mixed_model2.r) for GLMM analysis
- Output files (Results_.xlsx, pred.xlsx) are generated here

The R script fits:
    Accuracy ~ Sleep_duration * Model + (1|Dataset) + (1|Dataset:Seed)
"""

import os
from pathlib import Path

# Module directory
ANALYSIS_DIR = Path(__file__).parent

# Standard file names
RESULTS_FILE = "Results_.xlsx"
PREDICTIONS_FILE = "pred.xlsx"
R_SCRIPT = "mixed_model2.r"


def get_results_path() -> Path:
    """Get path to Results_.xlsx."""
    return ANALYSIS_DIR / RESULTS_FILE


def get_predictions_path() -> Path:
    """Get path to pred.xlsx."""
    return ANALYSIS_DIR / PREDICTIONS_FILE


def get_r_script_path() -> Path:
    """Get path to mixed_model2.r."""
    return ANALYSIS_DIR / R_SCRIPT



