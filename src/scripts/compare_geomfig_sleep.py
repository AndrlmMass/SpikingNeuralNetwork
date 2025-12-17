#!/usr/bin/env python
"""
Compare geomfig performance with and without sleep.

This script runs the geometric figures classification experiment twice:
- Once with sleep disabled
- Once with sleep enabled

Then compares performance and generates plots.

Usage:
    python -m src.scripts.compare_geomfig_sleep [--seeds N] [--plot-only]
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
_src = Path(__file__).parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config.experiments import GEOMFIG_SLEEP_COMPARISON
from models import snn_sleepy


def run_geomfig_sleep_variants(seeds=None, skip_training=False):
    """
    Run geomfig experiment with and without sleep.
    
    Parameters
    ----------
    seeds : list of int, optional
        Random seeds for reproducibility (default: [1, 2, 3, 4, 5])
    skip_training : bool
        If True, only plot existing results (default: False)
        
    Returns
    -------
    dict
        Results for both conditions with structure:
        {
            "no_sleep": {"run_1": {...}, "run_2": {...}, ...},
            "with_sleep": {"run_1": {...}, "run_2": {...}, ...}
        }
    """
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    
    results_dir = Path("results\\experiments\\geomfig_comparison")
    results_dir.mkdir(exist_ok=True)
    
    experiment_config = GEOMFIG_SLEEP_COMPARISON
    sleep_configs = experiment_config["sleep_configs"]
    network_params = experiment_config["network"]
    data_params = experiment_config["data"]
    
    all_results = {}
    
    for sleep_config in sleep_configs:
        condition_name = sleep_config["name"]
        print("\n" + "=" * 70)
        print(f"GEOMFIG: {condition_name.upper()}")
        print("=" * 70)
        
        condition_results = {}
        
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n--- Seed {seed_idx}/{len(seeds)} (seed={seed}) ---")
            
            # Initialize network
            snn = snn_sleepy(classes=network_params["classes"])
            
            # Prepare data
            snn.prepare_data(
                all_images_train=data_params["all_images_train"],
                batch_image_train=data_params["batch_image_train"],
                all_images_test=data_params["all_images_test"],
                batch_image_test=data_params["batch_image_test"],
                all_images_val=data_params["all_images_val"],
                batch_image_val=data_params["batch_image_val"],
                image_dataset="geomfig",
                gain=data_params["gain"],
                geom_noise_var=data_params["noise_var"],
                seed=seed,
            )
            
            # Prepare network
            snn.prepare_network(
                plot_weights=False,
                w_dense_ee=0.15,
                w_dense_se=0.1,
                w_dense_ei=0.2,
                w_dense_ie=0.25,
            )
            
            # Train with specified sleep condition
            training_params = {
                "train_weights": True,
                "accuracy_method": "pca_lr",
                "sleep": sleep_config["sleep"],
                "sleep_ratio": sleep_config["sleep_ratio"],
            }
            
            snn.train_network(**training_params)
            
            # Analyze results
            results = snn.analyze_results(t_sne_test=False, save_plots=False)
            
            # Store results with seed identifier
            run_key = f"seed_{seed}"
            condition_results[run_key] = {
                "seed": seed,
                "test_accuracy": results.get("test_accuracy", None),
                "train_accuracy": results.get("train_accuracy", None),
                "val_accuracy": results.get("val_accuracy", None),
                "training_params": training_params,
            }
            
            print(f"  Test Accuracy: {condition_results[run_key]['test_accuracy']:.4f}")
        
        all_results[condition_name] = condition_results
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"geomfig_{condition_name}_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(condition_results, f, indent=2)
        print(f"\nResults saved: {result_file}")
    
    return all_results


def plot_comparison(results):
    """
    Generate comparison plots between sleep and no-sleep conditions.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_geomfig_sleep_variants
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    no_sleep_accuracies = [r["test_accuracy"] for r in results["no_sleep"].values()]
    with_sleep_accuracies = [r["test_accuracy"] for r in results["with_sleep"].values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    ax = axes[0]
    ax.boxplot([no_sleep_accuracies, with_sleep_accuracies], 
               labels=["No Sleep", "With Sleep"])
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Geomfig: Sleep Comparison")
    ax.grid(True, alpha=0.3)
    
    # Individual runs
    ax = axes[1]
    ax.plot(range(len(no_sleep_accuracies)), no_sleep_accuracies, "o-", label="No Sleep", linewidth=2)
    ax.plot(range(len(with_sleep_accuracies)), with_sleep_accuracies, "s-", label="With Sleep", linewidth=2)
    ax.set_xlabel("Run")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Geomfig: Accuracy by Run")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_dir = Path("results")
    plot_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = plot_dir / f"geomfig_sleep_comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("COMPARISON STATISTICS")
    print("=" * 70)
    print(f"No Sleep  - Mean: {np.mean(no_sleep_accuracies):.4f}, Std: {np.std(no_sleep_accuracies):.4f}")
    print(f"With Sleep - Mean: {np.mean(with_sleep_accuracies):.4f}, Std: {np.std(with_sleep_accuracies):.4f}")
    print(f"Difference: {np.mean(with_sleep_accuracies) - np.mean(no_sleep_accuracies):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare geomfig performance with and without sleep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comparison with default seeds
  python -m src.scripts.compare_geomfig_sleep
  
  # Use custom seeds
  python -m src.scripts.compare_geomfig_sleep --seeds 1 2 3 4 5
  
  # Only plot existing results (skip training)
  python -m src.scripts.compare_geomfig_sleep --plot-only
        """
    )
    
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Random seeds for reproducibility (default: 1 2 3)"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training, only generate plots from existing results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("GEOMFIG SLEEP COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print("=" * 70)
    
    if not args.plot_only:
        results = run_geomfig_sleep_variants(seeds=args.seeds)
    else:
        # Would need to load from existing results file
        print("Plot-only mode not yet implemented. Run training first.")
        return
    
    # Generate comparison plots
    plot_comparison(results)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
