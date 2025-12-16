"""
Evaluation module for SNN-sleepy.

This module provides:
- metrics: Clustering metrics (phi), t-SNE visualization, PCA analysis
- classifiers: PCA + Logistic Regression, PCA + QDA
- paper_figures: Publication-quality figure generation
"""

from .metrics import (
    t_SNE,
    PCA_analysis,
    calculate_phi,
    bin_spikes_by_label_no_breaks,
)

from .classifiers import (
    pca_logistic_regression,
    pca_quadratic_discriminant,
)

from .paper_figures import (
    plot_model_accuracy_sleep,
    plot_glmm_predictions,
    plot_glmm_with_raw_accuracy,
    plot_geomfig_comparison,
    generate_all_paper_figures,
)

__all__ = [
    # Metrics
    "t_SNE",
    "PCA_analysis",
    "calculate_phi",
    "bin_spikes_by_label_no_breaks",
    # Classifiers
    "pca_logistic_regression",
    "pca_quadratic_discriminant",
    # Paper figures
    "plot_model_accuracy_sleep",
    "plot_glmm_predictions",
    "plot_glmm_with_raw_accuracy",
    "plot_geomfig_comparison",
    "generate_all_paper_figures",
]
