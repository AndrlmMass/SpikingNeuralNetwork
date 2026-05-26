"""
Parse SLURM output files from a Phase 2 sleep/noise grid sweep and produce
three test-accuracy heatmaps (one per reg_mode: static, layer, neuron).

Usage
-----
    python experiments/noise_article/sleep_noise_optimization/plot_phase2_heatmaps.py \
        --results-dir results/acc_history/mnist/2026.05.23/20_1
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_dir: str) -> pd.DataFrame:
    xlsx_path = os.path.join(results_dir, "Results_phase2.xlsx")
    if not os.path.isfile(xlsx_path):
        sys.exit(
            f"Results file not found: {xlsx_path}\n"
            "Run aggregate_phase2.py first."
        )
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    print(f"  Loaded {len(df)} rows from {xlsx_path}")
    for mode, grp in df.groupby("reg_mode"):
        print(f"  {mode}: {len(grp)} runs")
    return df


def build_pivot(
    df: pd.DataFrame, reg_mode: str, value_col: str = "test_acc"
) -> pd.DataFrame:
    sub = df[df["reg_mode"] == reg_mode]
    pivot = sub.pivot_table(
        index="sleep_duration",
        columns="var_noise",
        values=value_col,
        aggfunc="mean",
    )
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
    pivots = {
        m: build_pivot(df, m, value_col) for m in modes if m in df["reg_mode"].values
    }

    if not pivots:
        print("No data to plot.")
        return

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
            linewidths=0,
            rasterized=True,
            cbar=True,
            cbar_kws={"label": cbar_label},
            yticklabels=True,
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(cbar_label, fontsize=30)
        ax.set_title(mode.capitalize(), fontsize=35)
        ax.set_xlabel("Noise variance", fontsize=30)
        ax.set_ylabel("Sleep duration" if i == 0 else "", fontsize=30)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

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
            "2026.05.23",
            "20_2",
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
    plot_heatmaps(
        df, out_path, value_col="test_acc", cbar_label="Accuracy (%)", scale100=True
    )

    out_path_phi = os.path.join(results_dir, "heatmaps_test_phi.pdf")
    plot_heatmaps(
        df,
        out_path_phi,
        value_col="test_phi",
        cbar_label="Clustering score (φ)",
        scale100=False,
    )


if __name__ == "__main__":
    main()
