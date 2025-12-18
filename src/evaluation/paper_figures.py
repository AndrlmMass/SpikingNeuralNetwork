"""
Paper figure generation for SNN-sleepy research.

This module contains functions to generate the figures used in the paper,
comparing SNN-sleepy with snntorch across MNIST-family datasets.
"""

import os
import argparse
from typing import Optional, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all_paper_figures(
    output_dir: str = "src/analysis/plots",
    analysis_dir: str = "src/analysis",
) -> Dict[str, str]:
    """
    Generate all paper figures from available data.
    
    Args:
        output_dir: Directory to save figures (default: src/analysis/plots)
        analysis_dir: Directory containing Results_.xlsx and pred.xlsx
    
    Returns:
        Dict mapping figure names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = {}
    
    results_path = os.path.join(analysis_dir, "Results_.xlsx")
    pred_path = os.path.join(analysis_dir, "pred.xlsx")
    
    print(f"Looking for results at: {results_path}")
    print(f"Looking for predictions at: {pred_path}")
    
    if os.path.exists(results_path):
        try:
            fig_path = plot_model_accuracy_sleep(
                results_path,
                output_path=os.path.join(output_dir, "sleep_duration_comparison_models.pdf")
            )
            figures["model_accuracy"] = fig_path
        except Exception as e:
            print(f"Warning: Could not generate model accuracy plot: {e}")
    
    if os.path.exists(pred_path):
        try:
            fig_path = plot_glmm_predictions(
                pred_path,
                output_path=os.path.join(output_dir, "sleep_duration_comparison_glmm.pdf")
            )
            figures["glmm_predictions"] = fig_path
        except Exception as e:
            print(f"Warning: Could not generate GLMM predictions plot: {e}")
    
    if os.path.exists(results_path) and os.path.exists(pred_path):
        try:
            fig_path = plot_glmm_with_raw_accuracy(
                pred_path,
                results_path,
                output_path=os.path.join(output_dir, "sleep_duration_comparison_with_accuracy.pdf")
            )
            figures["glmm_with_accuracy"] = fig_path
        except Exception as e:
            print(f"Warning: Could not generate combined GLMM+accuracy plot: {e}")
    
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, help="Path to Results_.xlsx")
    parser.add_argument("--predictions", type=str, help="Path to pred.xlsx")
    parser.add_argument("--output-dir", type=str, default="analysis/plots", help="Output directory")
    parser.add_argument("--model-accuracy", action="store_true", help="Generate model accuracy plot")
    parser.add_argument("--glmm", action="store_true", help="Generate GLMM predictions plot")
    parser.add_argument("--combined", action="store_true", help="Generate combined GLMM+accuracy plot")
    parser.add_argument("--all", action="store_true", help="Generate all available figures")
    parser.add_argument("--show", action="store_true", help="Display plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all:
        figures = generate_all_paper_figures(output_dir=args.output_dir)
        print(f"Generated figures: {list(figures.keys())}")
    else:
        if args.model_accuracy and args.results:
            plot_model_accuracy_sleep(
                args.results,
                output_path=os.path.join(args.output_dir, "model_accuracy.pdf"),
                show_plot=args.show
            )
        
        if args.glmm and args.predictions:
            plot_glmm_predictions(
                args.predictions,
                output_path=os.path.join(args.output_dir, "glmm_predictions.pdf"),
                show_plot=args.show
            )
        
        if args.combined and args.predictions and args.results:
            plot_glmm_with_raw_accuracy(
                args.predictions,
                args.results,
                output_path=os.path.join(args.output_dir, "combined.pdf"),
                show_plot=args.show
            )



