"""
MNIST Family Experiment: Canonical test for sleep-rate comparison.

This module provides the complete reproducible pipeline for the paper:

1. Run SNN-sleepy across all datasets and sleep rates → Results_.xlsx
2. Run snntorch version for comparison → Results_.xlsx (merged)
3. Run R script (mixed_model2.r) for GLMM analysis → pred.xlsx
4. Generate paper figures

Usage:
    python -m experiments.mnist_family --full-pipeline
    python -m experiments.mnist_family --snn-sleepy-only
    python -m experiments.mnist_family --figures-only
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from config.experiments import MNIST_FAMILY_EXPERIMENT

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

MNIST_FAMILY_CONFIG = {
    **MNIST_FAMILY_EXPERIMENT,
    "output_dir": "src/analysis",
    "results_file": "Results_.xlsx",
    "predictions_file": "pred.xlsx",
    "r_script": "mixed_model2.r",
    "snntorch_dir": "src/snntorch_comparison",
    "snntorch_module": "snntorch_comparison.orchestrate_to_excel",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


# =============================================================================
# SNN-SLEEPY TRAINING
# =============================================================================

def run_snn_sleepy_experiment(
    datasets: Optional[List[str]] = None,
    sleep_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full SNN-sleepy experiment by invoking scripts.train_model.
    Delegates iteration to the script.
    """
    datasets = datasets or MNIST_FAMILY_CONFIG["datasets"]
    sleep_rates = sleep_rates or MNIST_FAMILY_CONFIG["sleep_rates"]
    seeds = seeds or MNIST_FAMILY_CONFIG["seeds"]
    
    # Calculate runs from seeds length (assuming seeds are 1..N)
    n_runs = len(seeds)
    if seeds != list(range(1, n_runs + 1)):
        print(f"Warning: train_model.py generates seeds 1..{n_runs}. Your specific seeds {seeds} might not match exact indices, but we will run {n_runs} times per config.")

    project_root = get_project_root()
    
    # Construct the single batch command
    # python -m scripts.train_model --dataset ... --sleep-rate ... --runs ...
    cmd = [
        sys.executable,
        "-m", "scripts.train_model",
        "--dataset", *datasets,
        "--sleep-rate", *[str(r) for r in sleep_rates],
        "--runs", str(n_runs),
        "--force-train" # ensure we retrain for experiment
    ]
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"\n{'='*60}")
    print(f"SNN-SLEEPY EXPERIMENT")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Run with real-time output
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
        )
        
        status = "success" if result.returncode == 0 else "failed"
        if status == "failed":
            print("SNN-Sleepy training failed.")
            
        return {
            "model": "SNN_sleepy",
            "datasets": datasets,
            "status": status,
        }
            
    except Exception as e:
        print(f"Experiment execution error: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# SNNTORCH TRAINING
# =============================================================================

def run_snntorch_experiment(
    datasets: Optional[List[str]] = None,
    sleep_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the snntorch comparison experiment.
    """
    project_root = get_project_root()
    
    datasets = datasets or MNIST_FAMILY_CONFIG["datasets"]
    sleep_rates = sleep_rates or MNIST_FAMILY_CONFIG["sleep_rates"]
    seeds = seeds or MNIST_FAMILY_CONFIG["seeds"]
    
    # Convert args for orchestrator
    sleep_intervals = [str(r) for r in sleep_rates]
    datasets_arg = [d.upper() for d in datasets]
    seeds_arg = [str(s) for s in seeds]
    
    cmd = [
        sys.executable,
        "-m", MNIST_FAMILY_CONFIG["snntorch_module"],
        "--datasets", *datasets_arg,
        "--seeds", *seeds_arg,
        "--sleep-intervals", *sleep_intervals,
    ]
    
    print(f"\n{'='*60}")
    print("SNNTORCH EXPERIMENT")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
        )
        
        status = "success" if result.returncode == 0 else "failed"
        return {"model": "snntorch", "status": status}
        
    except Exception as e:
        print(f"Warning: snntorch experiment error: {e}")
        return None


# =============================================================================
# GLMM ANALYSIS
# =============================================================================

def run_glmm_analysis(
    results_path: Optional[str] = None,
    r_script_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Run R script for GLMM analysis."""
    project_root = get_project_root()
    output_dir = project_root / MNIST_FAMILY_CONFIG["output_dir"]
    
    r_script_path = r_script_path or str(output_dir / MNIST_FAMILY_CONFIG["r_script"])
    output_path = output_path or str(output_dir / MNIST_FAMILY_CONFIG["predictions_file"])
    
    # Current R script expects Results_.xlsx in CWD or specific path. 
    # Usually it looks for "Results_.xlsx" in its working directory.
    
    print(f"\n{'='*60}")
    print("GLMM ANALYSIS (R)")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", os.path.basename(r_script_path)],
            capture_output=True,
            text=True,
            cwd=str(output_dir),
        )
        
        if result.returncode == 0:
            print(f"GLMM analysis complete. Predictions saved to {output_path}")
            return output_path
        else:
            print(f"R script failed (exit {result.returncode})")
            print(f"stderr: {result.stderr[:500]}")
            return None
    except FileNotFoundError:
        print("Warning: Rscript not found. Skipping GLMM.")
        return None


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    datasets: Optional[List[str]] = None,
    sleep_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    skip_snn_sleepy: bool = False,
    skip_snntorch: bool = False,
    skip_glmm: bool = False,
    skip_plots: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    
    project_root = get_project_root()
    output_dir = project_root / MNIST_FAMILY_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = {"steps": {}}

    # 1. SNN-sleepy
    if not skip_snn_sleepy:
        res = run_snn_sleepy_experiment(datasets, sleep_rates, seeds, extra_args)
        results_summary["steps"]["snn_sleepy"] = res

    # 2. snntorch
    if not skip_snntorch:
        res = run_snntorch_experiment(datasets, sleep_rates, seeds)
        results_summary["steps"]["snntorch"] = res
        
    # 3. Merge not needed (scripts write to single file)
    
    # 4. GLMM
    if not skip_glmm:
        res = run_glmm_analysis()
        results_summary["steps"]["glmm"] = res
        
    # 5. Figures
    if not skip_plots:
        try:
            from evaluation.paper_figures import generate_all_paper_figures
            figs = generate_all_paper_figures(
                output_dir=str(output_dir / "plots"),
                analysis_dir=str(output_dir)
            )
            results_summary["steps"]["figures"] = figs
        except ImportError:
            print("Warning: Could not import paper_figures")

    return results_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MNIST Family Experiment")
    
    # Mode flags
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--snn-sleepy-only", action="store_true")
    parser.add_argument("--snntorch-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--glmm-only", action="store_true")
    
    # Config overrides
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--sleep-rates", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()
    
    if args.quick:
        args.datasets = args.datasets or ["mnist", "fmnist"]
        args.sleep_rates = args.sleep_rates or [0.0, 0.5, 1.0]
        args.seeds = args.seeds or [1, 2]
        print("Running in QUICK TEST mode")

    if args.figures_only:
        from evaluation.paper_figures import generate_all_paper_figures
        root = get_project_root()
        out = root / MNIST_FAMILY_CONFIG["output_dir"]
        generate_all_paper_figures(str(out / "plots"), str(out))
    elif args.glmm_only:
        run_glmm_analysis()
    elif args.snn_sleepy_only:
        run_snn_sleepy_experiment(args.datasets, args.sleep_rates, args.seeds)
    elif args.snntorch_only:
        run_snntorch_experiment(args.datasets, args.sleep_rates, args.seeds)
    else:
        run_full_pipeline(args.datasets, args.sleep_rates, args.seeds)
