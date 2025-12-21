"""
Evaluation module for SNN-sleepy.

This module provides:
- metrics: Clustering metrics (phi), t-SNE visualization, PCA analysis
- classifiers: PCA + Logistic Regression, PCA + QDA
- paper_figures: Publication-quality figure generation
"""

from .classifiers import (
    pca_logistic_regression,
    pca_quadratic_discriminant,
    t_SNE,
    Phi,
    bin_spikes_by_label_no_breaks,
    fit_model,
    accuracy,
)

from .paper_figures import (
    generate_all_paper_figures,
)

from plot.plot import (
    plot_model_accuracy_sleep,
    plot_glmm_predictions,
    plot_glmm_with_raw_accuracy,
    plot_geomfig_comparison,
)

from pathlib import Path

# Module directory
EVALUATION_DIR = Path(__file__).parent

# Standard file names
RESULTS_FILE = "Results_.xlsx"
PREDICTIONS_FILE = "pred.xlsx"
R_SCRIPT = "mixed_model2.r"


def get_results_path() -> Path:
    """Get path to Results_.xlsx."""
    return EVALUATION_DIR / RESULTS_FILE


def get_predictions_path() -> Path:
    """Get path to pred.xlsx."""
    return EVALUATION_DIR / PREDICTIONS_FILE


def get_r_script_path() -> Path:
    """Get path to mixed_model2.r."""
    return EVALUATION_DIR / R_SCRIPT

__all__ = [
    # Metrics
    "t_SNE",
    "Phi",
    "bin_spikes_by_label_no_breaks",
    # Classifiers
    "pca_logistic_regression",
    "pca_quadratic_discriminant",
    "fit_model",
    "accuracy",
    # Paper figures
    "plot_model_accuracy_sleep",
    "plot_glmm_predictions",
    "plot_glmm_with_raw_accuracy",
    "plot_geomfig_comparison",
    "generate_all_paper_figures",
]
