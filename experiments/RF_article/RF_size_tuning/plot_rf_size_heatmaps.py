"""
Load Results_rf_size.xlsx and produce heatmaps of mean test accuracy and mean
phi over seeds, with RF mean on the y-axis and log-normal std on the x-axis.

Requires aggregate_rf_size.py to have been run first.

Usage
-----
    python experiments/RF_article/RF_size_tuning/plot_rf_size_heatmaps.py \
        --results-dir results/acc_history/mnist/2026.05.29/29

    # Read from a custom Excel path
    python experiments/RF_article/RF_size_tuning/plot_rf_size_heatmaps.py \
        --xlsx results/RF_size_Results.xlsx \
        --out-dir results/acc_history/mnist/2026.05.29/29
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    xlsx_path = os.path.join(results_dir, "Results_rf_size.xlsx")
    if not os.path.isfile(xlsx_path):
        sys.exit(
            f"Results file not found: {xlsx_path}\n"
            "Run aggregate_rf_size.py first."
        )
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    print(f"  Loaded {len(df)} rows from {xlsx_path}")
    n_seeds = df["seed"].nunique()
    n_combos = df.groupby(["rf_mean", "rf_lognorm_std"]).ngroups
    print(f"  {n_combos} (mean, std) combos × up to {n_seeds} seeds")
    return df


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------

def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean over seeds → (rf_mean × rf_lognorm_std) pivot."""
    pivot = df.pivot_table(
        index="rf_mean",
        columns="rf_lognorm_std",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.sort_index(ascending=False)        # largest mean at top
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)  # stds left→right
    return pivot


# ---------------------------------------------------------------------------
# Single heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    pivot: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".3f",
) -> None:
    mask = pivot.isna()
    sns.heatmap(
        pivot,
        ax=ax,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        rasterized=False,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(cbar_label, fontsize=15)

    ax.set_title(title, fontsize=18, pad=10)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def create_heatmaps(
    df: pd.DataFrame,
    out_dir: str,
    scale_acc_pct: bool = True,
) -> None:
    df = df.copy()
    if scale_acc_pct:
        df["test_acc"] = df["test_acc"] * 100

    pivot_acc = build_pivot(df, "test_acc")
    pivot_phi = build_pivot(df, "test_phi")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    plot_heatmap(
        pivot_acc,
        axes[0],
        title="Test accuracy (%, mean over seeds)",
        xlabel="RF log-normal std",
        ylabel="RF mean size (px)",
        cbar_label="Accuracy (%)",
        cmap="viridis",
        annot=True,
        fmt=".1f",
    )
    plot_heatmap(
        pivot_phi,
        axes[1],
        title="Test φ (mean over seeds)",
        xlabel="RF log-normal std",
        ylabel="RF mean size (px)",
        cbar_label="Clustering score (φ)",
        cmap="magma",
        annot=True,
        fmt=".1f",
    )

    out_path = os.path.join(out_dir, "heatmaps_rf_size.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # Also write individual panels
    for pivot, stem, title, cbar_label, cmap, fmt in [
        (pivot_acc, "heatmap_rf_acc",
         "Test accuracy (%, mean over seeds)", "Accuracy (%)", "viridis", ".1f"),
        (pivot_phi, "heatmap_rf_phi",
         "Test φ (mean over seeds)", "Clustering score (φ)", "magma", ".1f"),
    ]:
        fig_s, ax_s = plt.subplots(figsize=(9, 6), constrained_layout=True)
        plot_heatmap(
            pivot, ax_s,
            title=title,
            xlabel="RF log-normal std",
            ylabel="RF mean size (px)",
            cbar_label=cbar_label,
            cmap=cmap,
            annot=True,
            fmt=fmt,
        )
        p = os.path.join(out_dir, f"{stem}.pdf")
        fig_s.savefig(p, dpi=150)
        plt.close(fig_s)
        print(f"  Saved -> {p}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot RF size tuning heatmaps from Results_rf_size.xlsx"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results",
            "acc_history", "mnist", "2026.05.29", "29",
        ),
        help="Directory containing Results_rf_size.xlsx (also used for output)",
    )
    parser.add_argument(
        "--xlsx",
        default=None,
        help="Explicit path to Results_rf_size.xlsx (overrides --results-dir for input)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write PDFs (default: same as --results-dir)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir     = os.path.abspath(args.out_dir) if args.out_dir else results_dir

    os.makedirs(out_dir, exist_ok=True)

    # Allow an explicit xlsx path for cases where data lives elsewhere
    if args.xlsx:
        xlsx_path = os.path.abspath(args.xlsx)
        if not os.path.isfile(xlsx_path):
            sys.exit(f"Excel file not found: {xlsx_path}")
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        print(f"  Loaded {len(df)} rows from {xlsx_path}")
    else:
        if not os.path.isdir(results_dir):
            sys.exit(f"Directory not found: {results_dir}")
        df = load_results(results_dir)

    if df.empty:
        sys.exit("No data found — nothing to plot.")

    create_heatmaps(df, out_dir)


if __name__ == "__main__":
    main()
