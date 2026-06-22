"""
Load Results_xtar.xlsx and produce SE × EE percentile heatmaps for the x_tar sweep:
test accuracy, RF sharpening (Δcosine and ΔGini across the run), and final active-E
fraction — each averaged over seeds. The "mean" baseline is not on the SE×EE grid, so
its values are pulled out and shown in each panel subtitle as a reference.

Reading the sharpening panels (this is the whole point of the sweep):
    Δcosine < 0  ⇒ RFs becoming less redundant  (sharpening — good)
    ΔGini   > 0  ⇒ RFs more concentrated         (sharpening — good)

Requires aggregate_xtar.py to have been run first.

Usage
-----
    python experiments/RF_article/xtar_percentile/plot_xtar.py --results-dir results
    python experiments/RF_article/xtar_percentile/plot_xtar.py \
        --xlsx results/Results_xtar.xlsx --out-dir results
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
    xlsx_path = os.path.join(results_dir, "Results_xtar.xlsx")
    if not os.path.isfile(xlsx_path):
        sys.exit(f"Results file not found: {xlsx_path}\nRun aggregate_xtar.py first.")
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    print(f"  Loaded {len(df)} rows from {xlsx_path}")
    return df


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean over seeds → (pct_se × pct_ee) pivot, SE largest at top."""
    pivot = df.pivot_table(
        index="pct_se", columns="pct_ee", values=value_col, aggfunc="mean"
    )
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_heatmap(pivot, ax, title, cbar_label, cmap="viridis", fmt=".3f", center=None):
    sns.heatmap(
        pivot,
        ax=ax,
        mask=pivot.isna(),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("EE percentile", fontsize=13)
    ax.set_ylabel("SE percentile", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)


def _baseline(df: pd.DataFrame, col: str):
    base = df[df["mode"] == "mean"]
    if base.empty or col not in base:
        return float("nan")
    return base[col].mean()


def create_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    grid = df[df["mode"] == "percentile"].copy()
    if grid.empty:
        sys.exit("No percentile-mode rows found — nothing to plot.")
    grid["test_acc_pct"] = grid["test_acc"] * 100

    panels = [
        ("test_acc_pct", "Test accuracy (%)", "Accuracy (%)", "viridis", ".1f", None,
         _baseline(df, "test_acc") * 100),
        ("d_rf_mean_cosine", "ΔRF cosine  (sharpen ⇒ < 0)", "Δ mean cosine",
         "coolwarm_r", ".4f", 0.0, _baseline(df, "d_rf_mean_cosine")),
        ("d_rf_gini", "ΔRF Gini  (sharpen ⇒ > 0)", "Δ Gini",
         "coolwarm", ".4f", 0.0, _baseline(df, "d_rf_gini")),
        ("active_frac_exc_last", "Final active-E fraction", "active fraction",
         "magma", ".3f", None, _baseline(df, "active_frac_exc_last")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
    for (col, title, cbar, cmap, fmt, center, base), ax in zip(panels, axes.ravel()):
        pivot = build_pivot(grid, col)
        sub = f"{title}\n(mean baseline: {base:.3f})" if not pd.isna(base) else title
        plot_heatmap(pivot, ax, sub, cbar, cmap=cmap, fmt=fmt, center=center)

    out_path = os.path.join(out_dir, "heatmaps_xtar.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # individual panels too
    for col, title, cbar, cmap, fmt, center, base in panels:
        pivot = build_pivot(grid, col)
        fig_s, ax_s = plt.subplots(figsize=(9, 7), constrained_layout=True)
        sub = f"{title}\n(mean baseline: {base:.3f})" if not pd.isna(base) else title
        plot_heatmap(pivot, ax_s, sub, cbar, cmap=cmap, fmt=fmt, center=center)
        stem = col.replace("/", "_")
        p = os.path.join(out_dir, f"heatmap_xtar_{stem}.pdf")
        fig_s.savefig(p, dpi=150)
        plt.close(fig_s)
        print(f"  Saved -> {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot x_tar percentile SE×EE heatmaps from Results_xtar.xlsx"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing Results_xtar.xlsx (also used for output)",
    )
    parser.add_argument("--xlsx", default=None, help="Explicit Results_xtar.xlsx path")
    parser.add_argument("--out-dir", default=None, help="PDF output dir (default: results-dir)")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else results_dir
    os.makedirs(out_dir, exist_ok=True)

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
