"""
Gather and summarise Phase 1 / Phase 2 results.

Usage
-----
# Phase 1 — regularizer comparison
python experiments/gather_results.py --phase 1

# Phase 2 — sleep/noise grid
python experiments/gather_results.py --phase 2

# Custom results directory
python experiments/gather_results.py --results-dir experiments/results/phase1
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_results(results_dir: str) -> list[dict]:
    """Load all results.json files found under results_dir."""
    runs = []
    for root, _, files in os.walk(results_dir):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                data = json.load(f)
            data["_dir"] = os.path.basename(root)
            runs.append(data)
    return runs


def summarise_phase1(runs: list[dict]):
    """Group by reg_type/reg_mode, show mean ± std across seeds."""
    groups = defaultdict(list)
    for r in runs:
        cfg = r["config"]
        key = f"{cfg['reg_type']}/{cfg['reg_mode']}"
        groups[key].append(r)

    print(f"\n{'='*65}")
    print(f"Phase 1 — Regularizer comparison  ({len(runs)} runs total)")
    print(f"{'='*65}")
    print(f"{'Condition':<25} {'N':>3}  {'test_acc':>10}  {'val_acc':>10}  {'test_phi':>10}")
    print(f"{'-'*65}")

    rows = []
    for key in sorted(groups):
        entries = groups[key]
        test_accs = [e["test_acc"] for e in entries if e["test_acc"] is not None]
        val_accs  = [e["best_val_acc"] for e in entries if e["best_val_acc"] is not None]
        test_phis = [e["test_phi"] for e in entries if e["test_phi"] is not None]

        mean_test = np.mean(test_accs) if test_accs else float("nan")
        std_test  = np.std(test_accs)  if len(test_accs) > 1 else 0.0
        mean_val  = np.mean(val_accs)  if val_accs  else float("nan")
        mean_phi  = np.mean(test_phis) if test_phis else float("nan")

        print(
            f"{key:<25} {len(entries):>3}"
            f"  {mean_test:.4f}±{std_test:.4f}"
            f"  {mean_val:>10.4f}"
            f"  {mean_phi:>10.4f}"
        )
        rows.append((key, mean_test, std_test, mean_val, mean_phi, len(entries)))

    print(f"{'='*65}")
    best = max(rows, key=lambda x: x[1])
    print(f"\nBest condition: {best[0]}  (test_acc = {best[1]:.4f} ± {best[2]:.4f})")

    # Per-seed breakdown
    print(f"\n{'Condition':<25} {'seed':>6}  {'test_acc':>10}  {'val_acc':>10}")
    print(f"{'-'*55}")
    for key in sorted(groups):
        for e in sorted(groups[key], key=lambda x: x["config"]["seed"]):
            cfg = e["config"]
            print(
                f"{key:<25} {cfg['seed']:>6}"
                f"  {e['test_acc']:>10.4f}"
                f"  {e['best_val_acc']:>10.4f}"
            )
        print()

    return rows


def summarise_phase2(runs: list[dict]):
    """Print a sleep_duration × var_noise grid of mean test accuracy."""
    if not runs:
        print("No results found.")
        return

    groups = defaultdict(list)
    for r in runs:
        cfg = r["config"]
        key = (cfg["sleep_duration"], cfg["var_noise"])
        groups[key].append(r["test_acc"])

    durations = sorted(set(k[0] for k in groups))
    noises    = sorted(set(k[1] for k in groups))

    print(f"\n{'='*70}")
    print(f"Phase 2 — sleep_duration × var_noise grid  ({len(runs)} runs total)")
    print(f"Mean test accuracy across {len(next(iter(groups.values())))} seeds")
    print(f"{'='*70}")

    header = f"{'dur \\ noise':<12}" + "".join(f"{n:>8.2f}" for n in noises)
    print(header)
    print("-" * len(header))
    for dur in durations:
        row = f"{dur:<12}"
        for noise in noises:
            vals = [v for v in groups.get((dur, noise), []) if v is not None]
            row += f"{np.mean(vals):>8.4f}" if vals else f"{'---':>8}"
        print(row)
    print(f"{'='*70}")

    best_key = max(groups, key=lambda k: np.mean([v for v in groups[k] if v is not None]))
    best_vals = [v for v in groups[best_key] if v is not None]
    print(f"\nBest: sleep_duration={best_key[0]}, var_noise={best_key[1]}"
          f"  →  {np.mean(best_vals):.4f} ± {np.std(best_vals):.4f}")


def save_csv(runs: list[dict], out_path: str):
    import csv
    if not runs:
        return
    keys = list(runs[0]["config"].keys()) + ["test_acc", "best_val_acc", "test_phi", "elapsed_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in runs:
            row = dict(r["config"])
            row["test_acc"]      = r.get("test_acc")
            row["best_val_acc"]  = r.get("best_val_acc")
            row["test_phi"]      = r.get("test_phi")
            row["elapsed_s"]     = r.get("elapsed_s")
            writer.writerow(row)
    print(f"\nCSV saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to save a flat CSV of all runs")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        base = os.path.join(os.path.dirname(__file__), "results")
        results_dir = os.path.join(base, f"phase{args.phase}")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    runs = load_results(results_dir)
    if not runs:
        print(f"No results.json files found in {results_dir}")
        return

    if args.phase == 1:
        summarise_phase1(runs)
    else:
        summarise_phase2(runs)

    csv_path = args.csv or os.path.join(results_dir, "summary.csv")
    save_csv(runs, csv_path)


if __name__ == "__main__":
    main()
