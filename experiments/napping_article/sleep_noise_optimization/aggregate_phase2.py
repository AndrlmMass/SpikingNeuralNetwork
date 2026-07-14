"""
Parse SLURM output files from a Phase 2 sleep/noise grid sweep and produce
a single Results_phase2.xlsx aggregating all completed runs.

Usage
-----
    python experiments/noise_article/sleep_noise_optimization/aggregate_phase2.py \
        --results-dir results/acc_history/mnist/2026.05.23/20_2
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

RE_CONFIG = re.compile(
    r"Config.*reg: sleep/(\w+) \| sleep_dur: (\d+) \| var_noise: ([\d.]+).*\| seed: (\d+)"
)
RE_ACC = re.compile(r"Test accuracy\s*(?:\([^)]*\))?\s*:\s*([\d.]+|nan)")
RE_PHI = re.compile(r"Test phi\s*:\s*([\d.]+|nan)")


def parse_slurm_file(path: str):
    reg_mode = sleep_dur = var_noise = seed = test_acc = test_phi = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if reg_mode is None:
                m = RE_CONFIG.search(line)
                if m:
                    reg_mode = m.group(1)
                    sleep_dur = int(m.group(2))
                    var_noise = float(m.group(3))
                    seed = int(m.group(4))
            if test_acc is None:
                m = RE_ACC.search(line)
                if m:
                    raw = m.group(1)
                    test_acc = float("nan") if raw == "nan" else float(raw)
            if test_phi is None:
                m = RE_PHI.search(line)
                if m:
                    raw = m.group(1)
                    test_phi = float("nan") if raw == "nan" else float(raw)
            if reg_mode is not None and test_acc is not None and test_phi is not None:
                break
    if reg_mode is not None and test_acc is not None and test_phi is not None:
        return reg_mode, sleep_dur, var_noise, seed, test_acc, test_phi
    return None


def load_all(results_dir: str):
    files = glob.glob(os.path.join(results_dir, "slurm-*.out"))
    print(f"  Found {len(files)} slurm-*.out files")

    rows, failed = [], []
    for f in files:
        result = parse_slurm_file(f)
        if result is None:
            failed.append(f)
            continue
        reg_mode, sleep_dur, var_noise, seed, test_acc, test_phi = result
        rows.append(
            {
                "reg_mode": reg_mode,
                "sleep_duration": sleep_dur,
                "var_noise": var_noise,
                "seed": seed,
                "test_acc": test_acc,
                "test_phi": test_phi,
            }
        )

    return rows, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "results",
            "acc_history",
            "mnist",
            "2026.05.24",
            "21",
        ),
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(results_dir):
        sys.exit(f"Directory not found: {results_dir}")

    print(f"Scanning: {results_dir}")
    rows, failed = load_all(results_dir)

    if failed:
        print(f"\n  {len(failed)} incomplete/failed run(s):")
        for path in failed:
            print(f"    {os.path.basename(path)}")

    if not rows:
        sys.exit("No complete runs found — nothing to write.")

    df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "Results_phase2.xlsx")
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"\n  Written {len(df)} rows → {out_path}")

    if failed:
        print(
            f"\n  {len(failed)} config(s) missing from Excel — re-submit those SLURM tasks."
        )


if __name__ == "__main__":
    main()
