#!/usr/bin/env python
"""
Run the canonical MNIST family experiment.

This script orchestrates the full experimental pipeline for paper reproduction:

1. Train SNN-sleepy across datasets and sleep rates → Results_.xlsx
2. Train snntorch for comparison → Results_.xlsx (merged)
3. Run R GLMM analysis (mixed_model2.r) → pred.xlsx
4. Generate paper figures (Figure 3)

Usage:
    python -m src.scripts.run_canonical_experiment
    python -m src.scripts.run_canonical_experiment --quick
    python -m src.scripts.run_canonical_experiment --figures-only
    
The canonical command that replicates the paper:
    python src/snn_sleepy_trainer.py --sleep-rate 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \\
        --sleep-mode static --dataset mnist kmnist notmnist fmnist
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.mnist_family import (
    MNIST_FAMILY_CONFIG,
    run_full_pipeline,
    run_snn_sleepy_experiment,
    run_snntorch_experiment,
    run_glmm_analysis,
)
from src.evaluation.paper_figures import generate_all_paper_figures


def main():
    parser = argparse.ArgumentParser(
        description="Run the canonical MNIST family experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment with all datasets and sleep rates
  python -m src.scripts.run_canonical_experiment
  
  # Quick test with reduced settings
  python -m src.scripts.run_canonical_experiment --quick
  
  # Only generate figures from existing data
  python -m src.scripts.run_canonical_experiment --figures-only
  
  # Skip certain steps
  python -m src.scripts.run_canonical_experiment --skip-snntorch --skip-glmm
        """
    )
    
    # Dataset and sleep rate options
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to test (default: {MNIST_FAMILY_CONFIG['datasets']})"
    )
    parser.add_argument(
        "--sleep-rates",
        nargs="+",
        type=float,
        default=None,
        help=f"Sleep rates to test (default: {MNIST_FAMILY_CONFIG['sleep_rates']})"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Random seeds (default: {MNIST_FAMILY_CONFIG['seeds']})"
    )
    
    # Quick test mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with reduced settings (2 datasets, 3 sleep rates, 2 seeds)"
    )
    
    # Skip options
    parser.add_argument(
        "--skip-snntorch",
        action="store_true",
        help="Skip snntorch comparison training"
    )
    parser.add_argument(
        "--skip-glmm",
        action="store_true",
        help="Skip R GLMM analysis"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip paper figure generation"
    )
    
    # Figures only mode
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures from existing data"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Output directory for results (default: analysis)"
    )
    
    args = parser.parse_args()
    
    # Quick test mode overrides
    if args.quick:
        args.datasets = args.datasets or ["mnist", "fmnist"]
        args.sleep_rates = args.sleep_rates or [0.0, 0.5, 1.0]
        args.seeds = args.seeds or [1, 2]
        print("Running in QUICK TEST mode with reduced settings")
    
    # Figures only mode
    if args.figures_only:
        print("\n" + "=" * 60)
        print("GENERATING FIGURES FROM EXISTING DATA")
        print("=" * 60)
        
        figures = generate_all_paper_figures(
            output_dir=os.path.join(args.output_dir, "plots"),
            analysis_dir=args.output_dir,
        )
        
        if figures:
            print(f"\nGenerated {len(figures)} figures:")
            for name, path in figures.items():
                print(f"  - {name}: {path}")
        else:
            print("No figures generated. Check that Results_.xlsx and pred.xlsx exist in the analysis folder.")
        
        return
    
    # Run full pipeline
    print("\n" + "=" * 60)
    print("CANONICAL MNIST FAMILY EXPERIMENT")
    print("=" * 60)
    print(f"Datasets: {args.datasets or MNIST_FAMILY_CONFIG['datasets']}")
    print(f"Sleep rates: {args.sleep_rates or MNIST_FAMILY_CONFIG['sleep_rates']}")
    print(f"Seeds: {args.seeds or MNIST_FAMILY_CONFIG['seeds']}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    results = run_full_pipeline(
        datasets=args.datasets,
        sleep_rates=args.sleep_rates,
        seeds=args.seeds,
        skip_snntorch=args.skip_snntorch,
        skip_glmm=args.skip_glmm,
        skip_plots=args.skip_plots,
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("Generated files:")
    for name, path in results.get("files", {}).items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()

