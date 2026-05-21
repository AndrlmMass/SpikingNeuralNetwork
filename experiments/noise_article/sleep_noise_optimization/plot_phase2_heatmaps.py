"""
Parse SLURM output files from a Phase 2 sleep/noise grid sweep and produce
three test-accuracy heatmaps (one per reg_mode: static, layer, neuron).

Usage
-----
    python experiments/noise_article/sleep_noise_optimization/plot_phase2_heatmaps.py \
        --results-dir results/acc_history/mnist/2026.05.20/18
"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RE_CONFIG = re.compile(
    r"Config.*reg: sleep/(\w+) \| sleep_dur: (\d+) \| var_noise: ([\d.]+)"
)
RE_ACC = re.compile(r"Test accuracy\s*(?:\([^)]*\))?\s*:\s*([\d.]+)")
RE_PHI = re.compile(r"Test phi\s*:\s*([\d.]+)")


def parse_slurm_file(path: str):
    reg_mode = sleep_dur = var_noise = test_acc = test_phi = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if reg_mode is None:
                m = RE_CONFIG.search(line)
                if m:
                    reg_mode = m.group(1)
                    sleep_dur = int(m.group(2))
                    var_noise = float(m.group(3))
            if test_acc is None:
                m = RE_ACC.search(line)
                if m:
                    test_acc = float(m.group(1))
            if test_phi is None:
                m = RE_PHI.search(line)
                if m:
                    test_phi = float(m.group(1))
            if reg_mode is not None and test_acc is not None and test_phi is not None:
                break
    if reg_mode is not None and test_acc is not None:
        return reg_mode, sleep_dur, var_noise, test_acc, test_phi
    return None


def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "slurm-*.out")
    files = glob.glob(pattern)
    print(f"  Found {len(files)} slurm-*.out files")
    rows = []
    skipped = 0
    for f in files:
        result = parse_slurm_file(f)
        if result is None:
            skipped += 1
            continue
        reg_mode, sleep_dur, var_noise, test_acc, test_phi = result
        rows.append(
            {
                "reg_mode": reg_mode,
                "sleep_duration": sleep_dur,
                "var_noise": var_noise,
                "test_acc": test_acc,
                "test_phi": test_phi,
            }
        )

    df = pd.DataFrame(rows)
    total = len(files)
    found = len(rows)
    print(f"{found}/{total} runs parsed  ({skipped} incomplete/skipped)")
    if found:
        for mode, grp in df.groupby("reg_mode"):
            print(f"  {mode}: {len(grp)} runs")
    return df


def build_pivot(df: pd.DataFrame, reg_mode: str, value_col: str = "test_acc") -> pd.DataFrame:
    sub = df[df["reg_mode"] == reg_mode]
    pivot = sub.pivot(index="sleep_duration", columns="var_noise", values=value_col)
    pivot.index = pivot.index.astype(int)
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_heatmaps(
    df: pd.DataFrame,
    out_path: str,
    value_col: str = "test_acc",
    cbar_label: str = "Accuracy (%)",
    scale100: bool = True,
) -> None:
    modes = ["static", "layer", "neuron"]
    df = df.copy()
    if scale100:
        df[value_col] = df[value_col] * 100
    pivots = {m: build_pivot(df, m, value_col) for m in modes if m in df["reg_mode"].values}

    if not pivots:
        print("No data to plot.")
        return

    vmin = df[value_col].min()
    vmax = df[value_col].max()

    fig, axes = plt.subplots(
        1, len(modes), figsize=(6 * len(modes), 5), constrained_layout=True
    )
    if len(modes) == 1:
        axes = [axes]

    for i, (ax, mode) in enumerate(zip(axes, modes)):
        if mode not in pivots:
            ax.set_visible(False)
            continue
        pivot = pivots[mode]
        mask = pivot.isna()
        sns.heatmap(
            pivot,
            ax=ax,
            mask=mask,
            annot=False,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
            cbar=False,
            yticklabels=True,
        )
        ax.set_title(mode.capitalize(), fontsize=35)
        ax.set_xlabel("Noise variance", fontsize=30)
        ax.set_ylabel("Sleep duration" if i == 0 else "", fontsize=30)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, fontsize=30)

    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


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
            "2026.05.20",
            "18",
        ),
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(results_dir):
        sys.exit(f"Directory not found: {results_dir}")

    print(f"Scanning: {results_dir}")
    df = load_results(results_dir)
    if df.empty:
        sys.exit("No completed runs found.")

    out_path = os.path.join(results_dir, "heatmaps_test_acc.pdf")
    plot_heatmaps(df, out_path, value_col="test_acc", cbar_label="Accuracy (%)", scale100=True)

    out_path_phi = os.path.join(results_dir, "heatmaps_test_phi.pdf")
    plot_heatmaps(df, out_path_phi, value_col="test_phi", cbar_label="Clustering score (φ)", scale100=False)


if __name__ == "__main__":
    main()
