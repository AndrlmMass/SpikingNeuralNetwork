"""
Plot GLMM-predicted accuracy from the phase-2 grid sweep.

3-panel line plot (layer / neuron / static) with a shared broken y-axis:
  bottom row  ylim (0.0,  0.27)  — static lines live here
  top row     ylim (0.68, 1.02)  — layer / neuron lines live here
Break marks appear on the left (and right) spine at the row junction.

Usage
-----
    python experiments/noise_article/sleep_noise_optimization/plot_glmm_predictions.py \
        --results-dir results/acc_history/mnist/2026.05.24/21
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import pandas as pd

MODES = ["layer", "neuron", "static"]
MODE_LABELS = {"layer": "Layer", "neuron": "Neuron", "static": "Static"}

TOP_YLIM = (68, 102)  # layer / neuron range (×100)
BOT_YLIM = (0, 27)  # static range (×100)

# height proportional to y-range so tick spacing looks uniform
_TOP_SPAN = TOP_YLIM[1] - TOP_YLIM[0]  # 0.34
_BOT_SPAN = BOT_YLIM[1] - BOT_YLIM[0]  # 0.27
HEIGHT_RATIOS = [_TOP_SPAN, _BOT_SPAN]

XTICK_VALS = [1, 10, 50, 100, 150, 200, 300]
XTICK_POS = list(range(len(XTICK_VALS)))
_XPOS = {v: i for i, v in enumerate(XTICK_VALS)}


def load_data(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "GLMM_predictions_clust2.xlsx")
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    df = pd.read_excel(path)
    df["sleep_orig"] = df["sleep_orig"].round().astype(int)
    df["noise_orig"] = df["noise_orig"].round(1)
    snap = {1.0: 1.0, 2.5: 2.5, 5.0: 5.0, 7.5: 7.5, 10.0: 10.0, 100.0: 100.0}
    df["noise_orig"] = df["noise_orig"].map(
        lambda v: min(snap, key=lambda k: abs(k - v))
    )
    return df


def make_colormap(noise_levels):
    norm = mcolors.LogNorm(vmin=min(noise_levels), vmax=max(noise_levels))
    return {n: plt.cm.viridis(norm(n)) for n in noise_levels}


def _break_marks(ax_top, d: float = 0.03, sep: float = 0.022, y_center: float = -0.04):
    tr = ax_top.transAxes
    slope = 0.65
    for x0 in (0.0, 1.0):
        for dy in (0.0, sep):
            ax_top.plot(
                [x0 - d, x0 + d],
                [y_center - d * slope + dy, y_center + d * slope + dy],
                transform=tr,
                color="k",
                lw=1.4,
                clip_on=False,
            )


def _has_gap(df: pd.DataFrame, threshold: float = 0.3) -> bool:
    """Return True if there is a large gap in fit values (broken axis warranted)."""
    vals = df["fit"].sort_values().values
    gaps = vals[1:] - vals[:-1]
    return float(gaps.max()) / (vals.max() - vals.min()) > threshold


def plot(df: pd.DataFrame, out_prefix: str, ylabel: str = "Predicted accuracy (%)") -> None:
    df = df.copy()
    use_broken = _has_gap(df)

    # Scale to 0-100 only when a clear gap exists (accuracy case)
    if use_broken:
        df["fit"] = df["fit"] * 100

    noise_levels = sorted(df["noise_orig"].unique())
    color_map = make_colormap(noise_levels)

    if use_broken:
        fig = plt.figure(figsize=(13, 6))
        gs = GridSpec(
            2, 3,
            height_ratios=HEIGHT_RATIOS,
            hspace=0.06, wspace=0.14,
            left=0.08, right=0.83, top=0.91, bottom=0.12,
        )
        top_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
        bot_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    else:
        fig, axes = plt.subplots(
            1, 3, sharey=False, figsize=(13, 5),
            gridspec_kw={"wspace": 0.14},
        )
        fig.subplots_adjust(left=0.08, right=0.83, top=0.91, bottom=0.14)

    for col, mode in enumerate(MODES):
        sub = df[df["reg_target"] == mode]

        if use_broken:
            ax_top = top_axes[col]
            ax_bot = bot_axes[col]
            plot_axes = (ax_top, ax_bot)
        else:
            ax = axes[col]
            plot_axes = (ax,)

        for noise in noise_levels:
            sub_n = sub[sub["noise_orig"] == noise].sort_values("sleep_orig")
            xpos = sub_n["sleep_orig"].map(_XPOS)
            color = color_map[noise]
            kw = dict(color=color, linewidth=1.8, marker="o", markersize=4.5)
            for a in plot_axes:
                a.plot(xpos, sub_n["fit"], **kw)

        for a in plot_axes:
            a.set_xlim(-0.4, len(XTICK_POS) - 0.6)
            a.set_xticks(XTICK_POS)
            a.xaxis.set_minor_locator(mticker.NullLocator())
            a.yaxis.grid(True, linestyle="--", alpha=0.4)
            a.set_axisbelow(True)
            a.tick_params(labelsize=12)

        if use_broken:
            ax_top.set_ylim(*TOP_YLIM)
            ax_bot.set_ylim(*BOT_YLIM)
            ax_top.yaxis.set_major_locator(mticker.MultipleLocator(5))
            ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(5))
            ax_top.spines["bottom"].set_visible(False)
            ax_bot.spines["top"].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_bot.tick_params(top=False)
            ax_bot.set_xticklabels([str(v) for v in XTICK_VALS], fontsize=12)
            _break_marks(ax_top)
            ax_top.set_title(MODE_LABELS[mode], fontsize=28, pad=7)
            if col != 0:
                ax_top.tick_params(axis="y", labelleft=False)
                ax_bot.tick_params(axis="y", labelleft=False)
        else:
            lo, hi = sub["fit"].min(), sub["fit"].max()
            pad = (hi - lo) * 0.12
            ax.set_ylim(lo - pad, hi + pad)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
            ax.set_xticklabels([str(v) for v in XTICK_VALS], fontsize=12)
            ax.set_title(MODE_LABELS[mode], fontsize=28, pad=7)
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

    fig.supylabel(ylabel, fontsize=20, x=0.02)
    fig.supxlabel("Sleep duration", fontsize=20, y=0.01)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[n],
            linewidth=2,
            marker="o",
            markersize=5,
            label=str(int(n) if n == int(n) else n),
        )
        for n in noise_levels
    ]
    fig.legend(
        handles=legend_handles,
        title="Noise",
        title_fontsize=17,
        fontsize=14,
        loc="center left",
        bbox_to_anchor=(0.84, 0.50),
        frameon=True,
    )

    for ext in ("pdf", "png"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


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
    df = load_data(results_dir)
    ylabel = "Predicted accuracy (%)" if _has_gap(df) else "Predicted clustering (φ)"
    plot(df, os.path.join(results_dir, "glmm_predicted_clust2_"), ylabel=ylabel)


if __name__ == "__main__":
    main()
