"""
Parse SLURM output files from the RF size tuning sweep and produce
a single Results_rf_size.xlsx aggregating all completed runs.

Each row records rf_mean, rf_lognorm_std, seed, test_acc, and test_phi
parsed from the printed config and summary lines in each .out file.

Usage
-----
    python experiments/RF_article/RF_size_tuning/aggregate_rf_size.py \
        --results-dir results/acc_history/mnist/2026.05.29/29

    # Write Excel to a custom location
    python experiments/RF_article/RF_size_tuning/aggregate_rf_size.py \
        --results-dir results/acc_history/mnist/2026.05.29/29 \
        --out results/RF_size_Results.xlsx
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

# Matches: "Config — ... | rf_mean: 2.5 | rf_lognorm_std: 1.0 | seed: 3 | ..."
RE_CONFIG = re.compile(
    r"rf_mean:\s*([\d.]+)"
    r".*?rf_lognorm_std:\s*([\d.]+)"
    r".*?seed:\s*(\d+)"
)
RE_ACC = re.compile(r"Test accuracy\s*(?:\([^)]*\))?\s*:\s*([\d.]+|nan)")
RE_PHI = re.compile(r"Test phi\s*:\s*([\d.]+|nan)")


def _parse_float(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


def parse_slurm_file(path: str):
    """Return (rf_mean, rf_lognorm_std, seed, test_acc, test_phi) or None."""
    rf_mean = rf_std = seed = test_acc = test_phi = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if rf_mean is None:
                m = RE_CONFIG.search(line)
                if m:
                    rf_mean = float(m.group(1))
                    rf_std  = float(m.group(2))
                    seed    = int(m.group(3))
            if test_acc is None:
                m = RE_ACC.search(line)
                if m:
                    test_acc = _parse_float(m.group(1))
            if test_phi is None:
                m = RE_PHI.search(line)
                if m:
                    test_phi = _parse_float(m.group(1))
            if None not in (rf_mean, test_acc, test_phi):
                break

    if None not in (rf_mean, rf_std, seed, test_acc, test_phi):
        return rf_mean, rf_std, seed, test_acc, test_phi
    return None


def load_all(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "slurm-*.out")))
    print(f"  Found {len(files)} slurm-*.out files in {results_dir}")

    rows, failed = [], []
    for f in files:
        result = parse_slurm_file(f)
        if result is None:
            failed.append(os.path.basename(f))
            continue
        rf_mean, rf_std, seed, test_acc, test_phi = result
        rows.append(
            {
                "rf_mean":        rf_mean,
                "rf_lognorm_std": rf_std,
                "seed":           seed,
                "test_acc":       test_acc,
                "test_phi":       test_phi,
            }
        )

    return rows, failed


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RF size tuning SLURM results into an Excel file"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "acc_history", "mnist", "2026.05.29", "29",
        ),
        help="Directory containing slurm-*.out files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .xlsx path (default: <results-dir>/Results_rf_size.xlsx)",
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(results_dir):
        sys.exit(f"Directory not found: {results_dir}")

    rows, failed = load_all(results_dir)

    if failed:
        print(f"\n  {len(failed)} incomplete / failed run(s):")
        for name in failed:
            print(f"    {name}")

    if not rows:
        sys.exit("No complete runs found — nothing to write.")

    df = pd.DataFrame(rows).sort_values(["rf_mean", "rf_lognorm_std", "seed"])

    out_path = args.out or os.path.join(results_dir, "Results_rf_size.xlsx")
    df.to_excel(out_path, index=False, engine="openpyxl")

    n_combos = df.groupby(["rf_mean", "rf_lognorm_std"]).ngroups
    print(f"\n  Written {len(df)} rows ({n_combos} unique (mean, std) combos) -> {out_path}")

    if failed:
        print(
            f"\n  {len(failed)} job(s) missing from Excel — "
            "re-submit those SLURM tasks if needed."
        )


if __name__ == "__main__":
    main()
