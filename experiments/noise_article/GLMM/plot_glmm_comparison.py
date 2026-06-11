"""
Strip plot + GLMM predicted mean + 95% CI for the sleep-optimal vs
normalization vs no-reg comparison.

Layout: 2 rows x 3 cols
  Top row:    accuracy  (GLMM_comparison_acc*)
  Bottom row: clustering (GLMM_comparison_phi*)

Each panel (one reg_mode):
  - Individual seed points, jittered, coloured by reg_type
  - GLMM predicted mean + 95% CI as a diamond with error bar
  - No-reg baseline as a dashed horizontal line (raw mean from 5 seeds)

Usage
-----
    python experiments/noise_article/GLMM/plot_glmm_comparison.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DATA_DIR = os.path.join(
    REPO_ROOT, "results", "acc_history", "mnist", "2026.05.24", "21"
)

# ---------------------------------------------------------------------------
# Style — match analyse_phase1.py
# ---------------------------------------------------------------------------
SLEEP_COLOR = mcolors.to_hex(plt.cm.viridis(0.0))   # dark purple
NORM_COLOR  = mcolors.to_hex(plt.cm.viridis(1.0))   # yellow-green
NONE_COLOR  = "#7D8880"                              # grey

COLORS = {
    "sleep_opt_acc": SLEEP_COLOR,
    "sleep_opt_phi": SLEEP_COLOR,
    "normalize":     NORM_COLOR,
    "none":          NONE_COLOR,
}
LABELS = {
    "sleep_opt_acc": "Sleep (opt.)",
    "sleep_opt_phi": "Sleep (opt.)",
    "normalize":     "Normalize",
}

MODES       = ["layer", "neuron", "static"]
MODE_LABELS = {"layer": "Layer", "neuron": "Neuron", "static": "Static"}

JITTER     = 0.08
POINT_SIZE = 45
PRED_SIZE  = 110

# x-positions for the two reg_types within each panel
XPOS = {"normalize": 1, "sleep_opt_acc": 2, "sleep_opt_phi": 2}


def darken(color, factor=0.65):
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def raw_ci(values, scale=1.0):
    """Return (mean, lwr, upr) using t-distribution 95% CI on raw values."""
    vals = np.array(values) * scale
    n    = len(vals)
    m    = vals.mean()
    se   = vals.std(ddof=1) / np.sqrt(n)
    t    = stats.t.ppf(0.975, df=n - 1)
    return m, m - t * se, m + t * se


# ---------------------------------------------------------------------------
# Plot one row
# ---------------------------------------------------------------------------
def plot_row(axes, raw_df, pred_df, value_col, sleep_key,
             scale100, show_titles, show_xlabels):
    """
    axes      : list of 3 Axes
    raw_df    : full raw data (reg_type, reg_mode, seed, value_col)
    pred_df   : GLMM predictions (reg_type, reg_mode, fit, lwr, upr)
    sleep_key : "sleep_opt_acc" or "sleep_opt_phi"
    scale100  : multiply values by 100 (accuracy -> %)
    """
    scale  = 100.0 if scale100 else 1.0
    rng    = np.random.default_rng(seed=42)

    # No-reg baseline: raw mean ± 95% CI from the 5 no-reg seeds
    no_reg_vals = raw_df[raw_df["reg_type"] == "none"][value_col].values
    nreg_mean, nreg_lwr, nreg_upr = raw_ci(no_reg_vals, scale=scale)

    for col, mode in enumerate(MODES):
        ax      = axes[col]
        sub_raw = raw_df[raw_df["reg_mode"] == mode]

        # --- no-reg reference: dashed line + shaded CI band ---
        ax.axhline(nreg_mean, color=NONE_COLOR, linewidth=1.4,
                   linestyle="--", zorder=2, label="No reg")
        ax.axhspan(nreg_lwr, nreg_upr,
                   color=NONE_COLOR, alpha=0.12, zorder=1)

        for reg_type in ["normalize", sleep_key]:
            xc    = XPOS[reg_type]
            color = COLORS[reg_type]
            dark  = darken(color)

            # Raw seed points (jittered)
            vals = sub_raw[sub_raw["reg_type"] == reg_type][value_col].values * scale
            xs   = xc + rng.uniform(-JITTER, JITTER, size=len(vals))
            ax.scatter(xs, vals, color=color, edgecolors=dark,
                       s=POINT_SIZE, linewidths=0.8, zorder=3, alpha=0.85)

            # GLMM predicted mean + 95% CI (skip if NA — e.g. excluded outlier)
            if pred_df is not None:
                row = pred_df[
                    (pred_df["reg_type"] == reg_type) &
                    (pred_df["reg_mode"] == mode)
                ]
                if not row.empty and not pd.isna(row["fit"].values[0]):
                    fit = row["fit"].values[0] * scale
                    lwr = row["lwr"].values[0] * scale
                    upr = row["upr"].values[0] * scale
                    ax.errorbar(
                        xc, fit,
                        yerr=[[fit - lwr], [upr - fit]],
                        fmt="D",
                        color="white",
                        markeredgecolor=dark,
                        markeredgewidth=1.5,
                        markersize=9,
                        ecolor=dark,
                        elinewidth=1.8,
                        capsize=4,
                        capthick=1.5,
                        zorder=4,
                    )

        # Axes styling
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_locator(
            plt.MaxNLocator(nbins=5, prune="both", integer=scale100)
        )

        # Tight ylim: cover raw data + a little headroom, always include no-reg band
        reg_vals = raw_df[
            (raw_df["reg_mode"] == mode) &
            (raw_df["reg_type"] != "none")
        ][value_col].values * scale
        lo = min(reg_vals.min(), nreg_lwr)
        hi = max(reg_vals.max(), nreg_upr)
        pad = (hi - lo) * 0.22
        ax.set_ylim(lo - pad, hi + pad)

        if show_xlabels:
            ax.set_xticklabels(
                [LABELS["normalize"], LABELS[sleep_key]], fontsize=14
            )
        else:
            ax.set_xticklabels([])

        if show_titles:
            ax.set_title(MODE_LABELS[mode], fontsize=28, pad=7)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def make_figure():
    raw_acc  = pd.read_excel(
        os.path.join(DATA_DIR, "GLMM_comparison_acc.xlsx"), engine="openpyxl")
    pred_acc = pd.read_excel(
        os.path.join(DATA_DIR, "GLMM_comparison_acc_predictions.xlsx"), engine="openpyxl")

    raw_phi  = pd.read_excel(
        os.path.join(DATA_DIR, "GLMM_comparison_phi.xlsx"), engine="openpyxl")
    pred_phi = pd.read_excel(
        os.path.join(DATA_DIR, "GLMM_comparison_phi_predictions.xlsx"), engine="openpyxl")

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 8),
        gridspec_kw={"hspace": 0.12, "wspace": 0.38},
    )
    fig.subplots_adjust(left=0.08, right=0.81, top=0.93, bottom=0.1)

    # Top row: accuracy
    plot_row(axes[0], raw_acc, pred_acc,
             value_col="test_acc", sleep_key="sleep_opt_acc",
             scale100=True, show_titles=True, show_xlabels=False)
    axes[0][0].set_ylabel("Accuracy (%)", fontsize=20)

    # Bottom row: clustering
    plot_row(axes[1], raw_phi, pred_phi,
             value_col="test_phi", sleep_key="sleep_opt_phi",
             scale100=False, show_titles=False, show_xlabels=True)
    axes[1][0].set_ylabel("Clustering (φ)", fontsize=20)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=NORM_COLOR,  edgecolor=darken(NORM_COLOR),
                       label="Normalize"),
        mpatches.Patch(facecolor=SLEEP_COLOR, edgecolor=darken(SLEEP_COLOR),
                       label="Sleep (opt.)"),
        plt.Line2D([0], [0], color=NONE_COLOR, linewidth=1.4,
                   linestyle="--", label="No reg (mean)"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="white", markeredgecolor="dimgray",
                   markeredgewidth=1.5, markersize=9,
                   label="Predicted mean\n± 95% CI", linestyle="none"),
    ]
    fig.legend(
        handles=legend_handles,
        fontsize=12,
        loc="center left",
        bbox_to_anchor=(0.82, 0.50),
        frameon=True,
        title="Reg type",
        title_fontsize=13,
    )

    for ext in ("pdf", "png"):
        path = os.path.join(DATA_DIR, f"glmm_comparison_strip.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
