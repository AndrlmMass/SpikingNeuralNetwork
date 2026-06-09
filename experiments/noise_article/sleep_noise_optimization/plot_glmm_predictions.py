"""
Plot GLMM-predicted accuracy from the phase-2 grid sweep.

3-panel line plot (layer / neuron / static) with a shared broken y-axis:
  bottom row  ylim (0.0,  0.27)  — static lines live here
  top row     ylim (0.68, 1.02)  — layer / neuron lines live here
Break marks appear on the left (and right) spine at the row junction.

Optionally overlays semi-transparent boxplots from the raw Results_phase2.xlsx
data (one box per sleep-duration position, spread = seed-means per noise level).

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
_TOP_SPAN = TOP_YLIM[1] - TOP_YLIM[0]
_BOT_SPAN = BOT_YLIM[1] - BOT_YLIM[0]
HEIGHT_RATIOS = [_TOP_SPAN, _BOT_SPAN]

XTICK_VALS = [1, 10, 50, 100, 150, 200, 300]
XTICK_POS = list(range(len(XTICK_VALS)))
_XPOS = {v: i for i, v in enumerate(XTICK_VALS)}


def load_data(results_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(results_dir, filename)
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    df = pd.read_excel(path)
    df["sleep_orig"] = df["sleep_orig"].round().astype(int)
    df["noise_orig"] = df["noise_orig"].round(1)
    snap = {1.0: 1.0, 2.5: 2.5, 5.0: 5.0, 7.5: 7.5, 10.0: 10.0, 100.0: 100.0}
    df["noise_orig"] = df["noise_orig"].map(
        lambda v: min(snap, key=lambda k: abs(k - v))
    )
    # Scale CI columns if present (same factor applied to fit later)
    # They are passed through as-is; scaling happens in plot_combined / plot
    return df


def load_raw_data(results_dir: str) -> pd.DataFrame:
    """Load Results_phase2.xlsx for raw boxplot overlay."""
    path = os.path.join(results_dir, "Results_phase2.xlsx")
    if not os.path.isfile(path):
        print(f"Warning: raw data not found at {path} — skipping boxplots")
        return None
    return pd.read_excel(path, engine="openpyxl")


def make_colormap(noise_levels, cmap=plt.cm.viridis):
    norm = mcolors.LogNorm(vmin=min(noise_levels), vmax=max(noise_levels))
    return {n: cmap(norm(n)) for n in noise_levels}


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


def _build_raw_boxes(raw_df: pd.DataFrame, value_col: str, scale100: bool) -> dict:
    """
    Return {mode: {sleep_duration: [mean_per_noise_level]}} for boxplot overlay.

    For each (reg_mode, var_noise, sleep_duration) config the mean over seeds is
    computed → one value per noise level per x-position.
    """
    means = (
        raw_df.groupby(["reg_mode", "var_noise", "sleep_duration"])[value_col]
        .mean()
        .reset_index()
    )
    if scale100:
        means[value_col] = means[value_col] * 100

    boxes = {}
    for mode in MODES:
        sub = means[means["reg_mode"] == mode]
        boxes[mode] = {}
        for sd in XTICK_VALS:
            vals = sub[sub["sleep_duration"] == sd][value_col].tolist()
            if vals:
                boxes[mode][sd] = vals
    return boxes


def _draw_boxes(ax, mode_boxes: dict, mode: str):
    """Draw semi-transparent boxplots behind the GLMM lines on a single axis."""
    box_positions = [_XPOS[sd] for sd in XTICK_VALS if sd in mode_boxes]
    box_data = [mode_boxes[sd] for sd in XTICK_VALS if sd in mode_boxes]
    if not box_data:
        return
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.55,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="dimgray", linewidth=1.2),
        whiskerprops=dict(linewidth=0.7, color="gray"),
        capprops=dict(linewidth=0.7, color="gray"),
        flierprops=dict(
            marker="o",
            markersize=2.5,
            markerfacecolor="gray",
            markeredgecolor="gray",
            linestyle="none",
        ),
        zorder=1,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")
        patch.set_alpha(0.45)
        patch.set_edgecolor("gray")
        patch.set_linewidth(0.8)


def _plot_row(
    axes: list,
    df: pd.DataFrame,
    color_map: dict,
    noise_levels: list,
    raw_boxes: dict = None,
    scale100: bool = False,
    show_titles: bool = True,
    show_xlabels: bool = True,
):
    """Populate one row of the combined figure (3 mode panels)."""
    for col, mode in enumerate(MODES):
        ax = axes[col]
        sub = df[df["reg_target"] == mode]

        # Raw boxplot overlay (behind lines)
        if raw_boxes is not None:
            mode_boxes = raw_boxes.get(mode, {})
            if mode_boxes:
                _draw_boxes(ax, mode_boxes, mode)

        # GLMM fitted lines + CI ribbons
        has_ci = "lwr" in sub.columns and "upr" in sub.columns
        for noise in noise_levels:
            sub_n = sub[sub["noise_orig"] == noise].sort_values("sleep_orig")
            xpos = sub_n["sleep_orig"].map(_XPOS).values
            color = color_map[noise]
            if has_ci:
                ax.fill_between(
                    xpos,
                    sub_n["lwr"].values,
                    sub_n["upr"].values,
                    color=color,
                    alpha=0.15,
                    zorder=1,
                )
            ax.plot(
                xpos,
                sub_n["fit"],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                zorder=2,
            )

        ax.set_xlim(-0.4, len(XTICK_POS) - 0.6)
        ax.set_xticks(XTICK_POS)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=12)
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=6, prune="both", integer=True)
        )

        # Per-panel tight ylim
        lo, hi = sub["fit"].min(), sub["fit"].max()
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad)

        if show_xlabels:
            ax.set_xticklabels([str(v) for v in XTICK_VALS], fontsize=12)
        else:
            ax.set_xticklabels([])

        if show_titles:
            ax.set_title(MODE_LABELS[mode], fontsize=28, pad=7)


def plot_combined(
    df_acc: pd.DataFrame,
    df_clust: pd.DataFrame,
    out_prefix: str,
    raw_df: pd.DataFrame = None,
) -> None:
    """2-row x 3-col combined figure: accuracy (top) and clustering (bottom).
    Each panel has its own free y-axis — no broken axis needed."""

    df_acc = df_acc.copy()
    df_clust = df_clust.copy()

    # Scale accuracy to %
    for col in ["fit", "lwr", "upr"]:
        if col in df_acc.columns:
            df_acc[col] = df_acc[col] * 100

    noise_levels = sorted(df_acc["noise_orig"].unique())
    cmap_acc = plt.cm.viridis
    cmap_clust = plt.cm.viridis
    color_map_acc = make_colormap(noise_levels, cmap=cmap_acc)
    color_map_clust = make_colormap(noise_levels, cmap=cmap_clust)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10, 6),
        gridspec_kw={"hspace": 0.10, "wspace": 0.2},
    )
    fig.subplots_adjust(left=0.07, right=0.82, top=0.94, bottom=0.09)

    # Top row: accuracy (no raw boxplot overlay)
    _plot_row(
        axes[0],
        df_acc,
        color_map_acc,
        noise_levels,
        raw_boxes=None,
        show_titles=True,
        show_xlabels=False,
    )

    # Bottom row: clustering
    _plot_row(
        axes[1],
        df_clust,
        color_map_clust,
        noise_levels,
        raw_boxes=None,
        show_titles=False,
        show_xlabels=True,
    )

    # Row y-labels
    axes[0][0].set_ylabel("Predicted\nAccuracy", fontsize=24, x=-0.01)
    axes[1][0].set_ylabel("Predicted\nClustering", fontsize=24, x=-0.01)

    fig.supxlabel("Napping duration", fontsize=24, y=-0.025, x=0.445)

    # Single shared legend (noise levels, viridis colours from accuracy row)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map_acc[n],
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
        print(f"Saved -> {path}")
    plt.close(fig)


def plot(
    df: pd.DataFrame,
    out_prefix: str,
    ylabel: str = "Predicted accuracy (%)",
    raw_df: pd.DataFrame = None,
) -> None:
    """Single-metric standalone plot (kept for backwards compatibility)."""
    df = df.copy()
    use_broken = _has_gap(df)

    if use_broken:
        df["fit"] = df["fit"] * 100

    noise_levels = sorted(df["noise_orig"].unique())
    cmap = plt.cm.plasma
    color_map = make_colormap(noise_levels, cmap=cmap)

    raw_boxes = None
    if raw_df is not None and use_broken:
        raw_boxes = _build_raw_boxes(raw_df, "test_acc", scale100=True)

    if use_broken:
        fig = plt.figure(figsize=(13, 6))
        gs = GridSpec(
            2,
            3,
            height_ratios=HEIGHT_RATIOS,
            hspace=0.06,
            wspace=0.14,
            left=0.08,
            right=0.83,
            top=0.91,
            bottom=0.12,
        )
        top_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
        bot_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    else:
        fig, axes = plt.subplots(
            1,
            3,
            sharey=False,
            figsize=(13, 5),
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

        if raw_boxes is not None:
            mode_boxes = raw_boxes.get(mode, {})
            if mode_boxes:
                sample_vals = [v for vals in mode_boxes.values() for v in vals]
                overall_mean = sum(sample_vals) / len(sample_vals)
                draw_ax = ax_top if overall_mean > BOT_YLIM[1] else ax_bot
                _draw_boxes(draw_ax, mode_boxes, mode)

        for noise in noise_levels:
            sub_n = sub[sub["noise_orig"] == noise].sort_values("sleep_orig")
            xpos = sub_n["sleep_orig"].map(_XPOS)
            color = color_map[noise]
            kw = dict(color=color, linewidth=1.8, marker="o", markersize=4.5, zorder=2)
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
            ax.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=6, prune="both", integer=True)
            )
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
        print(f"Saved -> {path}")
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

    raw_df = load_raw_data(results_dir)
    df_acc = load_data(results_dir, "GLMM_predictions.xlsx")
    df_clust = load_data(results_dir, "GLMM_predictions_clust2.xlsx")

    # Combined 2-row figure (primary output)
    plot_combined(
        df_acc,
        df_clust,
        os.path.join(results_dir, "glmm_predicted_combined"),
        raw_df=raw_df,
    )

    # Standalone accuracy plot (with broken axis + boxplots)
    plot(
        df_acc,
        os.path.join(results_dir, "glmm_predicted_acc"),
        ylabel="Predicted accuracy (%)",
        raw_df=raw_df,
    )

    # Standalone clustering plot
    plot(
        df_clust,
        os.path.join(results_dir, "glmm_predicted_clust2_"),
        ylabel="Predicted clustering (φ)",
    )


if __name__ == "__main__":
    main()
