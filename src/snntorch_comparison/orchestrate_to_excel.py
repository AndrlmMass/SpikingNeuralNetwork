"""
Orchestrate snntorch training runs and save results to Excel.

Usage:
    python -m src.snntorch_comparison.orchestrate_to_excel --datasets MNIST KMNIST FMNIST
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import math

# Try importing pandas/openpyxl, warn if missing
try:
    from openpyxl import Workbook, load_workbook
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from tqdm import tqdm


# Module directory
SCRIPT_DIR = Path(__file__).parent

DEFAULT_DATASETS = ["MNIST", "KMNIST", "FMNIST", "NOTMNIST"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def run_training_once(dataset: str, seed: int, results_json: Path, extra_args: List[str], sleep_interval: Optional[float] = None) -> Tuple[float, float]:
    """
    Invoke main.py for a single dataset/seed run and return (final_test_acc, final_test_loss).
    """
    results_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m", "src.snntorch_comparison.main",
        "--dataset",
        dataset,
        "--runs",
        "1",
        "--results-json",
        str(results_json),
        "--seed",
        str(seed),
        "--no-plot",
    ] + (extra_args or [])
    if sleep_interval is not None:
        cmd += ["--sleep-interval-pct", str(sleep_interval)]

    print(f"\n==> Running dataset={dataset} seed={seed} sleep={sleep_interval}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Run from project root
    project_root = SCRIPT_DIR.parent.parent
    proc = subprocess.run(cmd, cwd=str(project_root), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for dataset={dataset}, seed={seed} (exit={proc.returncode})")

    if not results_json.exists():
        raise FileNotFoundError(f"Results JSON not found at {results_json}")
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Results JSON is not a list")

    last = None
    for entry in reversed(data):
        if str(entry.get("dataset", "")).upper() == dataset.upper() and int(entry.get("seed", -1)) == int(seed):
            if sleep_interval is not None:
                try:
                    si = entry.get("sleep_interval_pct")
                    if si is None:
                        continue
                    if not math.isclose(float(si), float(sleep_interval), rel_tol=1e-9, abs_tol=1e-9):
                        continue
                except Exception:
                    continue
            last = entry
            break
    if last is None:
        raise ValueError(f"No matching results found for dataset={dataset}, seed={seed}, sleep={sleep_interval}")

    final_acc = float(last["final_test_acc"])
    final_loss = float(last["final_test_loss"])
    return final_acc, final_loss


def normalize_label(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default=str(SCRIPT_DIR.parent / "analysis" / "Results_.xlsx"),
                        help="Path to Excel workbook to update")
    parser.add_argument("--datasets", type=str, nargs="*", default=DEFAULT_DATASETS,
                        help="Datasets to iterate (e.g., MNIST KMNIST FMNIST)")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS,
                        help="Seed values to use per dataset")
    parser.add_argument("--model-name", type=str, default="snntorch-SG",
                        help="Model name (e.g., snntorch-SG)")
    parser.add_argument("--results-json", type=str, default=str(SCRIPT_DIR / "results" / "orchestrate_runs.json"),
                        help="JSON file where runs append results")
    parser.add_argument("--sleep-intervals", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95],
                        help="One or more sleep interval fractions (e.g., 0.1)")
    parser.add_argument("--extra", type=str, nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to main.py after '--'")
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("Error: pandas and openpyxl must be installed to save results.")
        sys.exit(1)

    extra_args = [a for a in args.extra if a != "--"] if args.extra else []
    results_json = Path(args.results_json)
    excel_path = Path(args.excel).resolve()

    print(f"Orchestrating runs. Results will be appended to: {excel_path}", flush=True)

    # Load existing or create new DataFrame
    cols = ["Sleep_duration", "Model", "Run", "Seed", "Dataset", "Accuracy"]
    if excel_path.exists():
        try:
            df = pd.read_excel(excel_path)
            # Ensure columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = None
        except ValueError:
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
        excel_path.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.sleep_intervals) * len(args.datasets) * len(args.seeds)
    
    # We will use "Run" as an accumulating index per config if needed, 
    # but the user spec says "Run (Passed as argument)". 
    # Since snntorch runs are typically 1 per seed here, we can map Seed -> Run if they are 1-to-1,
    # or just use a counter. The user said "Run ... Seed (we roll seeds from 1 to 5)".
    # Let's assume Run == Seed index for now, or just restart Run count for each dataset/config.
    
    with tqdm(total=total_runs, desc="Orchestrating", leave=True) as pbar:
        for sleep in args.sleep_intervals:
            for ds in args.datasets:
                for run_idx, seed in enumerate(args.seeds):
                    try:
                        acc, loss = run_training_once(ds, seed, results_json, extra_args, sleep_interval=sleep)
                        
                        # Prepare row
                        # Sleep_duration: 0-100
                        sleep_dur = float(sleep) * 100
                        
                        new_row = {
                            "Sleep_duration": sleep_dur,
                            "Model": args.model_name,
                            "Run": run_idx + 1,  # 1-based run index
                            "Seed": seed,
                            "Dataset": ds,
                            "Accuracy": acc
                        }
                        
                        # Append to DataFrame
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # Save incrementally
                        try:
                            df.to_excel(excel_path, index=False)
                        except PermissionError as e:
                            print(f"Warning: PermissionError saving workbook: {e}", flush=True)

                    except Exception as e:
                        print(f"Error running dataset={ds} seed={seed} sleep={sleep}: {e}", flush=True)
                    
                    pbar.update(1)

    print(f"Completed. Results saved to: {excel_path}", flush=True)


if __name__ == "__main__":
    main()
