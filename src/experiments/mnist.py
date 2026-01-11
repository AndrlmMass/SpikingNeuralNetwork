"""
MNIST Family Experiment: Canonical test for sleep-rate comparison.

This module provides the complete reproducible pipeline for the paper:

1. Run SNN-sleepy across all datasets and sleep rates → Results_.xlsx
2. Run snntorch version for comparison → Results_.xlsx (merged)
3. Run R script (mixed_model2.r) for GLMM analysis → pred.xlsx
4. Generate paper figures

Usage:
    python -m experiments.mnist --full-pipeline
    python -m experiments.mnist --snn-sleepy-only
    python -m experiments.mnist --figures-only
"""

import os
import subprocess
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from src.config.experiment_configs import MNIST_FAMILY_EXPERIMENT
from src.config.defaults import (
    DEFAULT_NETWORK_PARAMS,
    DEFAULT_TRAINING_PARAMS,
    DEFAULT_DATA_PARAMS,
)
from src.models.SNN_sleepy.snn import snn_sleepy

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

MNIST_FAMILY_CONFIG = {
    **MNIST_FAMILY_EXPERIMENT,
    "output_dir": "data/comparison/mnist",
    "results_file": "Results_.xlsx",
    "predictions_file": "pred.xlsx",
    "r_script": "mixed_model2.r",
    "snntorch_dir": "src/snntorch_comparison",
    "snntorch_module": "snntorch_comparison.orchestrate_to_excel",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def save_result_to_file(
    result: Dict[str, Any],
    output_dir: Path,
    results_file: str,
    model_name: str = "SNN_sleepy",
):
    """
    Save a single result row to Excel file incrementally.
    Reads existing file, appends new result, and writes back (removes duplicates).
    """
    import pandas as pd

    # Convert result dict to DataFrame with single row
    result_df = pd.DataFrame([result])

    # Map columns to desired format
    column_mapping = {
        "sleep_rate": "Sleep_duration",
        "run_name": "Run",
        "dataset": "Dataset",
        "train_accuracy": "Accuracy-train",
        "val_accuracy": "Accuracy-val",
        "test_accuracy": "Accuracy-test",
        "train_clustering": "Clustering-train",
        "val_clustering": "Clustering-val",
        "test_clustering": "Clustering-test",
    }
    result_df = result_df.rename(columns=column_mapping)

    # Add Model column
    result_df["Model"] = model_name

    # Add Lambda column using sleep_decay_rate (varies between snntorch and snn_sleepy)
    if "Lambda" not in result_df.columns:
        if "sleep_decay_rate" in result_df.columns:
            result_df["Lambda"] = result_df["sleep_decay_rate"]
        elif "Sleep_duration" in result_df.columns:
            result_df["Lambda"] = result_df["Sleep_duration"]
        elif "sleep_rate" in result_df.columns:
            result_df["Lambda"] = result_df["sleep_rate"]

    # Define desired column order
    desired_columns = [
        "Sleep_duration",
        "Model",
        "Run",
        "Lambda",
        "Dataset",
        "Accuracy-train",
        "Accuracy-val",
        "Accuracy-test",
        "Clustering-train",
        "Clustering-val",
        "Clustering-test",
    ]

    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in desired_columns if col in result_df.columns]
    for col in result_df.columns:
        if col not in existing_columns:
            existing_columns.append(col)
    result_df = result_df[existing_columns]

    # Save/update Excel file (read existing, append, write back)
    excel_file = output_dir / results_file
    try:
        if excel_file.exists():
            # Read existing Excel file
            existing_df = pd.read_excel(excel_file)
            # Append new result
            combined_df = pd.concat([existing_df, result_df], ignore_index=True)
            # Remove duplicates based on Run column (in case of reruns)
            combined_df = combined_df.drop_duplicates(subset=["Run"], keep="last")
            # Write back
            combined_df.to_excel(excel_file, index=False)
        else:
            # Create new Excel file
            result_df.to_excel(excel_file, index=False)
    except Exception as e:
        # If Excel write fails, print warning
        print(f"Warning: Could not update Excel file {excel_file}: {e}")


# =============================================================================
# SNN-SLEEPY TRAINING
# =============================================================================


def run_snn_sleepy_experiment(
    datasets: Optional[List[str]] = None,
    sleep_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    preview_data: bool = False,
    plot_weights_trajectories: bool = False,
    plot_weights_evolution: bool = False,
    track_weights: bool = False,
    plot_spikes: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full SNN-sleepy experiment by building and training models directly.

    Iterates over all combinations of datasets, sleep rates, and seeds,
    trains models, and collects results.
    """
    datasets = datasets or MNIST_FAMILY_CONFIG["datasets"]
    sleep_rates = sleep_rates or MNIST_FAMILY_CONFIG["sleep_rates"]
    seeds = seeds or MNIST_FAMILY_CONFIG["seeds"]

    # Get network and training params from config
    network_params = MNIST_FAMILY_CONFIG.get("network", DEFAULT_NETWORK_PARAMS.copy())
    training_params = MNIST_FAMILY_CONFIG.get(
        "training", DEFAULT_TRAINING_PARAMS.copy()
    )
    data_params = MNIST_FAMILY_CONFIG.get("data", DEFAULT_DATA_PARAMS.copy())

    print(f"\n{'='*60}")
    print(f"SNN-SLEEPY EXPERIMENT")
    print(f"{'='*60}")
    print(f"Datasets: {datasets}")
    print(f"Sleep rates: {sleep_rates}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(datasets) * len(sleep_rates) * len(seeds)}")
    print(f"{'='*60}\n")

    results = []
    total_runs = len(datasets) * len(sleep_rates) * len(seeds)
    current_run = 0

    try:
        # Iterate over all combinations
        for dataset in datasets:
            for sleep_rate in sleep_rates:
                for seed in seeds:
                    current_run += 1
                    run_name = f"{dataset}_sleep{sleep_rate:.1f}_seed{seed}"

                    print(f"\n{'='*60}")
                    print(f"Run {current_run}/{total_runs}: {run_name}")
                    print(f"{'='*60}")

                    try:
                        # Create SNN instance
                        snn = snn_sleepy(
                            N_exc=network_params.get("N_exc", 200),
                            N_inh=network_params.get("N_inh", 50),
                            N_x=network_params.get("N_x", 225),
                            seed=seed,
                            which_classes=network_params.get(
                                "classes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                            ),
                        )

                        # Verify initial weights (should be different for each run due to seed)
                        initial_weight_sum = (
                            np.sum(snn.weights) if hasattr(snn, "weights") else None
                        )
                        print(
                            f"  Initial weight sum: {initial_weight_sum:.6f}"
                            if initial_weight_sum is not None
                            else "  Initial weights: not yet created"
                        )

                        # Prepare data
                        all_train = data_params.get("all_images_train", 6000)
                        all_val = data_params.get("all_images_val", 100)
                        all_test = data_params.get("all_images_test", 1000)
                        total_data = all_train + all_val + all_test

                        train_split = all_train / total_data
                        val_split = all_val / total_data
                        test_split = all_test / total_data

                        snn.prepare_data(
                            dataset=dataset,
                            total_data=total_data,
                            num_steps=data_params.get("num_steps", 100),
                            train_split=train_split,
                            val_split=val_split,
                            test_split=test_split,
                            batch_size=data_params.get("batch_image_train", 400),
                            gain=data_params.get("gain", 1.0),
                            force_recreate=False,
                            preview_data=preview_data,
                        )

                        # Prepare network
                        snn.prepare_network(
                            create_network=True,
                            w_dense_ee=network_params.get("w_dense_ee", 0.15),
                            w_dense_se=network_params.get("w_dense_se", 0.1),
                            w_dense_ei=network_params.get("w_dense_ei", 0.2),
                            w_dense_ie=network_params.get("w_dense_ie", 0.25),
                            se_weights=network_params.get("se_weights", 0.15),
                            ee_weights=network_params.get("ee_weights", 0.3),
                            ei_weights=network_params.get("ei_weights", 0.3),
                            ie_weights=network_params.get("ie_weights", -0.3),
                            spike_threshold_default=network_params.get(
                                "spike_threshold_default", -55
                            ),
                            resting_membrane=network_params.get(
                                "resting_potential", -70
                            ),
                        )

                        # Configure training parameters
                        batch_size = data_params.get("batch_image_train", 400)
                        val_batch_size = data_params.get("batch_image_val", 100)
                        test_batch_size = data_params.get("batch_image_test", 200)

                        # Override sleep settings for this run
                        run_training_params = training_params.copy()
                        run_training_params["sleep_ratio"] = sleep_rate
                        run_training_params["sleep"] = sleep_rate > 0.0

                        # Debug: Verify sleep_ratio is set correctly
                        actual_sleep_ratio = run_training_params.get("sleep_ratio")
                        actual_sleep = run_training_params.get("sleep")
                        print(
                            f"  Setting sleep_ratio={actual_sleep_ratio}, sleep={actual_sleep}"
                        )

                        snn.train_network(
                            train_weights=run_training_params.get(
                                "train_weights", True
                            ),
                            learning_rate_exc=run_training_params.get(
                                "learning_rate_exc", 0.0005
                            ),
                            learning_rate_inh=run_training_params.get(
                                "learning_rate_inh", 0.0005
                            ),
                            sleep=actual_sleep,  # Use explicitly set value, not .get() with default
                            sleep_ratio=actual_sleep_ratio,  # Use explicitly set value, not .get() with default
                            sleep_mode=run_training_params.get("sleep_mode", "static"),
                            accuracy_method=run_training_params.get(
                                "accuracy_method", "pca_lr"
                            ),
                            pca_variance=run_training_params.get("pca_variance", 0.95),
                            use_validation_data=True,
                            test_batch_size=test_batch_size,
                            epochs=run_training_params.get("epochs", 10),
                            force_train=True,
                            plot_weight_trajectories=plot_weights_trajectories,
                            plot_weight_evolution=plot_weights_evolution,
                            track_weights=track_weights,
                            plot_spikes_train=plot_spikes,
                            **{
                                k: v
                                for k, v in run_training_params.items()
                                if k
                                not in [
                                    "train_weights",
                                    "learning_rate_exc",
                                    "learning_rate_inh",
                                    "sleep",
                                    "sleep_ratio",
                                    "sleep_mode",
                                    "accuracy_method",
                                    "pca_variance",
                                    "resting_potential",
                                    "spike_threshold_default",
                                    "check_sleep_interval",
                                    "timing_update",
                                    "trace_update",
                                    "vectorized_trace",
                                    "epochs",
                                ]
                            },
                        )

                        # Verify sleep_ratio was set correctly in the model (before training)
                        print(
                            f"  Before training: snn.sleep_ratio={snn.sleep_ratio}, snn.sleep={snn.sleep}"
                        )

                        # Run training (this will call initiate_trackers which generates sleep_schedule)
                        snn.train(
                            batch_size=batch_size,
                            val_batch_size=val_batch_size,
                        )

                        # Verify weights were updated during training
                        final_weight_sum = (
                            np.sum(snn.weights) if hasattr(snn, "weights") else None
                        )
                        if (
                            initial_weight_sum is not None
                            and final_weight_sum is not None
                        ):
                            weight_change = final_weight_sum - initial_weight_sum
                            print(
                                f"  Weight change during training: {weight_change:.6f}"
                            )
                            if abs(weight_change) < 1e-6:
                                print(
                                    f"  ⚠️  WARNING: Weights appear unchanged after training!"
                                )

                        # Verify sleep_schedule after training (after initiate_trackers has run)
                        if hasattr(snn, "sleep_schedule"):
                            sleep_schedule_size = (
                                len(snn.sleep_schedule) if snn.sleep_schedule else 0
                            )
                            if sleep_rate == 0.0:
                                if sleep_schedule_size > 0:
                                    print(
                                        f"  ⚠️  ERROR: sleep_ratio=0.0 but sleep_schedule has {sleep_schedule_size} timesteps!"
                                    )
                                else:
                                    print(
                                        f"  ✅ Confirmed: sleep_ratio=0.0 → no sleep (sleep_schedule is empty)"
                                    )
                            else:
                                if sleep_schedule_size == 0:
                                    print(
                                        f"  ⚠️  ERROR: sleep_ratio={sleep_rate} but sleep_schedule is empty!"
                                    )
                                else:
                                    print(
                                        f"  ✅ Confirmed: sleep_ratio={sleep_rate} → {sleep_schedule_size} sleep timesteps scheduled"
                                    )

                        # Verify final weights after testing (should be same as after training, since test doesn't update weights)
                        final_weight_sum_after_test = (
                            np.sum(snn.weights) if hasattr(snn, "weights") else None
                        )
                        if (
                            initial_weight_sum is not None
                            and final_weight_sum_after_test is not None
                        ):
                            total_weight_change = (
                                final_weight_sum_after_test - initial_weight_sum
                            )
                            print(
                                f"  Final weight sum (after test): {final_weight_sum_after_test:.6f}"
                            )
                            print(
                                f"  Total weight change (initial → final): {total_weight_change:.6f}"
                            )
                            if abs(total_weight_change) < 1e-6:
                                print(
                                    f"  ⚠️  CRITICAL WARNING: Weights appear unchanged from initial to final!"
                                )

                        # Collect results
                        result = {
                            "dataset": dataset,
                            "sleep_rate": sleep_rate,
                            "sleep_decay_rate": sleep_rate,  # For SNN_sleepy, this equals sleep_rate
                            "seed": seed,
                            "run_name": run_name,
                        }

                        if hasattr(snn, "performance_tracker"):
                            # Collect accuracy metrics
                            if snn.performance_tracker.get("train_accuracy"):
                                result["train_accuracy"] = snn.performance_tracker[
                                    "train_accuracy"
                                ][-1]
                            if snn.performance_tracker.get("val_accuracy"):
                                result["val_accuracy"] = snn.performance_tracker[
                                    "val_accuracy"
                                ][-1]
                            if snn.performance_tracker.get("test_accuracy"):
                                result["test_accuracy"] = snn.performance_tracker[
                                    "test_accuracy"
                                ][-1]

                            # Collect clustering metrics
                            if snn.performance_tracker.get("train_clustering"):
                                result["train_clustering"] = snn.performance_tracker[
                                    "train_clustering"
                                ][-1]
                            if snn.performance_tracker.get("val_clustering"):
                                result["val_clustering"] = snn.performance_tracker[
                                    "val_clustering"
                                ][-1]
                            if snn.performance_tracker.get("test_clustering"):
                                result["test_clustering"] = snn.performance_tracker[
                                    "test_clustering"
                                ][-1]

                        print(f"Train Accuracy: {result['train_accuracy']:.4f}")
                        print(f"Val Accuracy: {result['val_accuracy']:.4f}")
                        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
                        results.append(result)

                        # Save result incrementally to file (after each run)
                        project_root = get_project_root()
                        output_dir = project_root / MNIST_FAMILY_CONFIG["output_dir"]
                        os.makedirs(output_dir, exist_ok=True)
                        results_file = MNIST_FAMILY_CONFIG["results_file"]
                        save_result_to_file(
                            result, output_dir, results_file, model_name="SNN_sleepy"
                        )

                        # Print run summary
                        print(f"\n{run_name} complete!")
                        if "test_accuracy" in result:
                            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
                            if "test_clustering" in result:
                                print(
                                    f"  Test Clustering: {result['test_clustering']:.4f}"
                                )
                        print(f"  ✅ Results saved to {output_dir / results_file}")

                    except Exception as e:
                        print(f"\n❌ Error in {run_name}: {e}")
                        import traceback

                        traceback.print_exc()
                        error_result = {
                            "dataset": dataset,
                            "sleep_rate": sleep_rate,
                            "seed": seed,
                            "run_name": run_name,
                            "error": str(e),
                        }
                        results.append(error_result)

                        # Save error result to file as well
                        project_root = get_project_root()
                        output_dir = project_root / MNIST_FAMILY_CONFIG["output_dir"]
                        os.makedirs(output_dir, exist_ok=True)
                        results_file = MNIST_FAMILY_CONFIG["results_file"]
                        save_result_to_file(
                            error_result,
                            output_dir,
                            results_file,
                            model_name="SNN_sleepy",
                        )

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Total runs: {len(results)}")
        successful = sum(1 for r in results if "error" not in r)
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")

        return {
            "model": "SNN_sleepy",
            "datasets": datasets,
            "status": "success" if successful > 0 else "failed",
            "results": results,
        }

    except Exception as e:
        print(f"\n❌ Experiment execution error: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "error": str(e), "results": results}


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
        "-m",
        MNIST_FAMILY_CONFIG["snntorch_module"],
        "--datasets",
        *datasets_arg,
        "--seeds",
        *seeds_arg,
        "--sleep-intervals",
        *sleep_intervals,
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
    output_path = output_path or str(
        output_dir / MNIST_FAMILY_CONFIG["predictions_file"]
    )

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
    plot_weights_trajectories: bool = False,
    plot_weights_evolution: bool = False,
    track_weights: bool = False,
    plot_spikes: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:

    project_root = get_project_root()
    output_dir = project_root / MNIST_FAMILY_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd

    results_df = pd.DataFrame()
    results_summary = {"steps": {}}

    # 1. SNN-sleepy
    if not skip_snn_sleepy:
        res = run_snn_sleepy_experiment(
            datasets,
            sleep_rates,
            seeds,
            plot_weights_trajectories=plot_weights_trajectories,
            plot_weights_evolution=plot_weights_evolution,
            track_weights=track_weights,
            plot_spikes=plot_spikes,
            extra_args=extra_args,
        )

        # Convert list of result dictionaries directly to DataFrame
        if res.get("results"):
            snn_df = pd.DataFrame(res["results"])

            # Rename columns to match desired format
            column_mapping = {
                "sleep_rate": "Sleep_duration",
                "run_name": "Run",
                "dataset": "Dataset",
                "train_accuracy": "Accuracy-train",
                "val_accuracy": "Accuracy-val",
                "test_accuracy": "Accuracy-test",
                "train_clustering": "Clustering-train",
                "val_clustering": "Clustering-val",
                "test_clustering": "Clustering-test",
            }
            snn_df = snn_df.rename(columns=column_mapping)

            # Add Model column
            snn_df["Model"] = "SNN_sleepy"

            # Add Lambda column using sleep_decay_rate (varies between snntorch and snn_sleepy)
            if "Lambda" not in snn_df.columns:
                if "sleep_decay_rate" in snn_df.columns:
                    snn_df["Lambda"] = snn_df["sleep_decay_rate"]
                elif "Sleep_duration" in snn_df.columns:
                    snn_df["Lambda"] = snn_df["Sleep_duration"]

            # Reorder columns
            desired_columns = [
                "Sleep_duration",
                "Model",
                "Run",
                "Lambda",
                "Dataset",
                "Accuracy-train",
                "Accuracy-val",
                "Accuracy-test",
                "Clustering-train",
                "Clustering-val",
                "Clustering-test",
            ]

            # Select and reorder columns (only include columns that exist)
            existing_columns = [col for col in desired_columns if col in snn_df.columns]
            # Add any additional columns that weren't in desired_columns (e.g., "seed", "error")
            for col in snn_df.columns:
                if col not in existing_columns:
                    existing_columns.append(col)
            snn_df = snn_df[existing_columns]

            # Append to results_df
            results_df = pd.concat([results_df, snn_df], ignore_index=True)

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
            from ..evaluation.paper_figures import generate_all_paper_figures

            figs = generate_all_paper_figures(
                output_dir=str(output_dir / "plots"), analysis_dir=str(output_dir)
            )
            results_summary["steps"]["figures"] = figs
        except ImportError:
            print("Warning: Could not import paper_figures")

    # Save results DataFrame to Excel if we have data
    if not results_df.empty:
        results_file = output_dir / MNIST_FAMILY_CONFIG["results_file"]
        results_df.to_excel(results_file, index=False)
        print(f"\n✅ Results saved to {results_file}")

    results_summary["results_dataframe"] = results_df
    return results_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MNIST Family Experiment: Canonical test for sleep-rate comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SNN-sleepy only
  python -m src.experiments.mnist --snn-sleepy-only
  
  # Run full pipeline
  python -m src.experiments.mnist --full-pipeline
  
  # Quick test mode (smaller dataset, fewer runs)
  python -m src.experiments.mnist --snn-sleepy-only --quick
  
  # Custom datasets and sleep rates
  python -m src.experiments.mnist --snn-sleepy-only --datasets mnist fmnist --sleep-rates 0.0 0.2 0.5
  
  # Generate figures only
  python -m src.experiments.mnist --figures-only
        """,
    )

    # Mode flags
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline: SNN-sleepy → snntorch → GLMM → figures",
    )
    parser.add_argument(
        "--snn-sleepy-only",
        action="store_true",
        help="Run only SNN-sleepy training experiments",
    )
    parser.add_argument(
        "--snntorch-only",
        action="store_true",
        help="Run only snntorch comparison experiments",
    )
    parser.add_argument(
        "--plot_spikes",
        action="store_true",
        help="Plot spikes from training to verify & inspect activity",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate paper figures only (requires Results_.xlsx)",
    )
    parser.add_argument(
        "--glmm-only",
        action="store_true",
        help="Run GLMM analysis only (requires Results_.xlsx)",
    )

    # Config overrides
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "kmnist", "fmnist", "notmnist"],
        help="Datasets to run (default: all from config)",
    )
    parser.add_argument(
        "--sleep-rates",
        nargs="+",
        type=float,
        help="Sleep rates to test (default: all from config)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Random seeds for runs (default: from config)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: smaller datasets, fewer runs, faster",
    )
    parser.add_argument(
        "--preview-data",
        action="store_true",
        help="Show preview plots of loaded MNIST data using plot_floats_and_spikes",
    )

    parser.add_argument(
        "--plot-weights-trajectories",
        action="store_true",
        help="Plot weights trajectories for all runs",
    )

    parser.add_argument(
        "--plot-weights-evolution",
        action="store_true",
        help="Plot weights evolution for all runs",
    )
    parser.add_argument(
        "--track-weights",
        action="store_true",
        help="Enable weight tracking (automatically enabled if plotting weights)",
    )

    args = parser.parse_args()

    # Quick mode defaults
    if args.quick:
        args.datasets = args.datasets or ["mnist", "fmnist"]
        args.sleep_rates = args.sleep_rates or [0.0, 0.5, 1.0]
        args.seeds = args.seeds or [1, 2]
        print("=" * 60)
        print("Running in QUICK TEST mode")
        print("=" * 60)

    # Execute based on mode
    try:
        if args.figures_only:
            from ..evaluation.paper_figures import generate_all_paper_figures

            root = get_project_root()
            out = root / MNIST_FAMILY_CONFIG["output_dir"]
            generate_all_paper_figures(str(out / "plots"), str(out))
            print("\n✅ Figures generated successfully!")

        elif args.glmm_only:
            result = run_glmm_analysis()
            if result:
                print("\n✅ GLMM analysis completed successfully!")
            else:
                print("\n⚠️  GLMM analysis completed with warnings")

        elif args.snn_sleepy_only:
            result = run_snn_sleepy_experiment(
                args.datasets,
                args.sleep_rates,
                args.seeds,
                preview_data=args.preview_data,
                plot_weights_trajectories=args.plot_weights_trajectories,
                plot_weights_evolution=args.plot_weights_evolution,
                track_weights=args.track_weights,
                plot_spikes=args.plot_spikes,
            )
            if result.get("status") == "success":
                print("\n✅ SNN-sleepy experiment completed successfully!")
            else:
                print(
                    f"\n❌ SNN-sleepy experiment failed: {result.get('status', 'unknown')}"
                )
                sys.exit(1)

        elif args.snntorch_only:
            result = run_snntorch_experiment(
                args.datasets, args.sleep_rates, args.seeds
            )
            if result and result.get("status") == "success":
                print("\n✅ snntorch experiment completed successfully!")
            else:
                print("\n⚠️  snntorch experiment completed with warnings")

        elif args.full_pipeline:
            result = run_full_pipeline(
                args.datasets,
                args.sleep_rates,
                args.seeds,
                plot_weights_trajectories=args.plot_weights_trajectories,
                plot_weights_evolution=args.plot_weights_evolution,
                track_weights=args.track_weights,
            )
            print("\n✅ Full pipeline completed!")
            print("\nSummary:")
            for step, step_result in result.get("steps", {}).items():
                status = (
                    step_result.get("status", "unknown")
                    if isinstance(step_result, dict)
                    else "completed"
                )
                print(f"  {step}: {status}")

        else:
            # Default: run full pipeline
            print("No mode specified, running full pipeline...")
            result = run_full_pipeline(
                args.datasets,
                args.sleep_rates,
                args.seeds,
                plot_weights_trajectories=args.plot_weights_trajectories,
                plot_weights_evolution=args.plot_weights_evolution,
                track_weights=args.track_weights,
            )
            print("\n✅ Full pipeline completed!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
