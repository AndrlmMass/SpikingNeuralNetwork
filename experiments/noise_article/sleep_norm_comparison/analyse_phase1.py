"""
Analyse Phase 1 results from SLURM output files.

Usage
-----
python experiments/analyse_phase1.py
"""

import glob
import os
import re
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SLURM_DIR = os.path.join(
    REPO_ROOT, "results", "acc_history", "mnist", "2026.05.16", "17"
)
OUT_DIR = os.path.join(REPO_ROOT, "results", "acc_history", "mnist", "2026.05.16", "17")

# Task ID → (reg_type, reg_mode, seed)  — matches slurm_phase1.sh
_CONDS = (
    ["sleep"] * 5
    + ["sleep"] * 5
    + ["sleep"] * 5
    + ["normalize"] * 5
    + ["normalize"] * 5
    + ["normalize"] * 5
)
_MODES = (
    ["static"] * 5
    + ["layer"] * 5
    + ["neuron"] * 5
    + ["static"] * 5
    + ["layer"] * 5
    + ["neuron"] * 5
    + ["static"] * 5
)
_SEEDS = [0, 1, 2, 3, 4] * 7


# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------
def _find_slurm_files() -> dict[int, str]:
    """Return {task_id: filepath} for all slurm-*_N.out files in SLURM_DIR."""
    mapping = {}
    for fpath in glob.glob(os.path.join(SLURM_DIR, "slurm-*_*.out")):
        fname = os.path.basename(fpath)
        try:
            task_id = int(fname.split("_")[-1].replace(".out", ""))
            # keep the most recently modified file if multiple job IDs exist
            if task_id not in mapping or os.path.getmtime(fpath) > os.path.getmtime(
                mapping[task_id]
            ):
                mapping[task_id] = fpath
        except ValueError:
            pass
    return mapping


def load_results() -> pd.DataFrame:
    slurm_files = _find_slurm_files()
    rows = []
    for task_id in range(len(_CONDS)):
        reg_type = _CONDS[task_id]
        reg_mode = _MODES[task_id]
        seed = _SEEDS[task_id]

        fpath = slurm_files.get(task_id)
        if fpath is None:
            rows.append(
                dict(
                    task_id=task_id,
                    reg_type=reg_type,
                    reg_mode=reg_mode,
                    seed=seed,
                    test_acc=None,
                    test_phi=None,
                    cancelled=False,
                )
            )
            continue

        with open(fpath, encoding="utf-8", errors="replace") as f:
            content = f.read()

        cancelled = "CANCELLED" in content or "TIME LIMIT" in content
        matches = re.findall(
            r"Test accuracy\s*[:(]\s*(?:PCA\+LR\):\s*)?([0-9]\.[0-9]+)",
            content,
        )
        test_acc = float(matches[-1]) if matches else None

        phi_matches = re.findall(r"Test phi\s*:\s*([\d.]+)", content)
        test_phi = float(phi_matches[-1]) if phi_matches else None

        rows.append(
            dict(
                task_id=task_id,
                reg_type=reg_type,
                reg_mode=reg_mode,
                seed=seed,
                test_acc=test_acc,
                test_phi=test_phi,
                cancelled=cancelled,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------¨
min_color = mcolors.to_hex(plt.cm.viridis(0.0))
max_color = mcolors.to_hex(plt.cm.viridis(1.0))
COLORS = {"sleep": min_color, "normalize": max_color, "none": "#7D8880"}
LABELS = {"sleep": "Sleep", "normalize": "Normalize", "none": "No reg"}

# Per-metric color schemes matching the heatmaps
COLORS_BY_METRIC = {
    "test_acc": {
        "sleep": mcolors.to_hex(plt.cm.viridis(0.0)),
        "normalize": mcolors.to_hex(plt.cm.viridis(1.0)),
        "none": "#7D8880",
    },
    "test_phi": {
        "sleep": mcolors.to_hex(plt.cm.plasma(0.0)),
        "normalize": mcolors.to_hex(plt.cm.plasma(1.0)),
        "none": "#7D8880",
    },
}


def darken(color: str, factor: float = 0.65) -> tuple:
    """Return a darkened version of a hex/named colour."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


MODE_ORDER = ["layer", "neuron", "static"]
MODE_DISPLAY = {"static": "Static", "layer": "Layer", "neuron": "Neuron"}

_BROKEN_TOP = (68, 107)   # layer / neuron accuracy range (×100) + bracket headroom
_BROKEN_BOT = (0,  28)    # static accuracy range (×100)


def _break_marks(ax_top, d: float = 0.03, sep: float = 0.022, y_center: float = -0.04):
    """Two parallel diagonal slashes at the bottom of ax_top (the break point)."""
    tr = ax_top.transAxes
    slope = 0.65
    for x0 in (0.0, 1.0):
        for dy in (0.0, sep):
            ax_top.plot(
                [x0 - d, x0 + d],
                [y_center - d * slope + dy, y_center + d * slope + dy],
                transform=tr, color="k", lw=1.4, clip_on=False,
            )


def plot_results(
    df: pd.DataFrame,
    out_path: str,
    value_col: str = "test_acc",
    ylabel: str = "Accuracy",
    scale100: bool = True,
    broken_axis: bool = False,
):
    # Only rows with a valid value for the requested column
    df_valid = df[df[value_col].notna()].copy()

    reg_modes = [m for m in MODE_ORDER if m in df_valid["reg_mode"].values]
    reg_types_present = [
        t for t in ["sleep", "normalize"] if t in df_valid["reg_type"].values
    ]

    # Shared: load no-reg baseline
    import os as _os
    _path = _os.path.join(
        _os.getcwd(), "results", "acc_history", "mnist", "2026.05.24", "21",
        "Results_phase2 EXT.xlsx",
    )
    _df2 = pd.read_excel(_path)
    no_reg_mean = _df2[_df2["reg_mode"] == "none"][value_col].mean()
    if scale100:
        no_reg_mean *= 100

    FIXED_POSITIONS = {"sleep": 1, "normalize": 2}
    BOX_WIDTH = 0.5

    # ------------------------------------------------------------------ #
    # BROKEN-AXIS PATH                                                     #
    # ------------------------------------------------------------------ #
    if broken_axis:
        from matplotlib.gridspec import GridSpec
        _top_span = _BROKEN_TOP[1] - _BROKEN_TOP[0]
        _bot_span = _BROKEN_BOT[1] - _BROKEN_BOT[0]
        fig = plt.figure(figsize=(4 * len(reg_modes), 6))
        gs = GridSpec(
            2, len(reg_modes),
            height_ratios=[_top_span, _bot_span],
            hspace=0.06, wspace=0.25,
            left=0.1, right=0.88, top=0.91, bottom=0.1,
        )
        top_axes = [fig.add_subplot(gs[0, c]) for c in range(len(reg_modes))]
        bot_axes = [fig.add_subplot(gs[1, c]) for c in range(len(reg_modes))]

        for col, mode in enumerate(reg_modes):
            ax_top = top_axes[col]
            ax_bot = bot_axes[col]
            sub = df_valid[df_valid["reg_mode"] == mode]
            types_here = [t for t in ["sleep", "normalize"] if t in sub["reg_type"].values]

            boxes_data, box_colors_list, box_labels, positions = [], [], [], []
            for rt in types_here:
                raw = sub[sub["reg_type"] == rt][value_col].tolist()
                vals = [v * 100 for v in raw] if scale100 else raw
                boxes_data.append(vals)
                box_colors_list.append(COLORS[rt])
                box_labels.append(LABELS[rt])
                positions.append(FIXED_POSITIONS[rt])

            # Draw boxplots on both axes — clipping handles visibility
            for ax in (ax_top, ax_bot):
                bp = ax.boxplot(
                    boxes_data, positions=positions, widths=BOX_WIDTH,
                    patch_artist=True,
                    medianprops=dict(linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=4, linestyle="none"),
                )
                for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors_list)):
                    dark = darken(color)
                    patch.set_facecolor(color); patch.set_alpha(0.75)
                    patch.set_edgecolor(dark)
                    bp["medians"][i].set_color(dark)
                    bp["whiskers"][2*i].set_color(dark); bp["whiskers"][2*i+1].set_color(dark)
                    bp["caps"][2*i].set_color(dark); bp["caps"][2*i+1].set_color(dark)
                    bp["fliers"][i].set_markerfacecolor(color)
                    bp["fliers"][i].set_markeredgecolor(dark)

            # Baseline on both (visible only in the row where value falls)
            for ax in (ax_top, ax_bot):
                ax.axhline(no_reg_mean, color=COLORS["none"], linewidth=1.5,
                           linestyle="--", zorder=3)

            ax_top.set_ylim(*_BROKEN_TOP)
            ax_bot.set_ylim(*_BROKEN_BOT)

            # Spines / ticks
            ax_top.spines["bottom"].set_visible(False)
            ax_bot.spines["top"].set_visible(False)
            ax_top.tick_params(bottom=False, labelbottom=False)
            ax_bot.tick_params(top=False)

            # Shared x setup
            for ax in (ax_top, ax_bot):
                ax.set_xlim(0.5, 2.5)
                ax.set_xticks(list(FIXED_POSITIONS.values()))
                ax.yaxis.grid(True, linestyle="--", alpha=0.5)
                ax.set_axisbelow(True)
                ax.tick_params(axis="y", labelsize=12)
            ax_bot.set_xticklabels(box_labels, fontsize=22)

            _break_marks(ax_top)

            # Significance bracket — draw on the axis where the data lives
            all_vals = [v for grp in boxes_data for v in grp]
            use_top = max(all_vals) > _BROKEN_BOT[1]
            sig_ax = ax_top if use_top else ax_bot
            if "sleep" in types_here and "normalize" in types_here:
                raw_sleep = sub[sub["reg_type"] == "sleep"][value_col].tolist()
                raw_norm = sub[sub["reg_type"] == "normalize"][value_col].tolist()
                v_sleep = [v * 100 for v in raw_sleep] if scale100 else raw_sleep
                v_norm  = [v * 100 for v in raw_norm]  if scale100 else raw_norm
                _, p = mannwhitneyu(v_sleep, v_norm, alternative="two-sided")
                lbl = sig_label(p)
                i_sleep = FIXED_POSITIONS["sleep"]
                i_norm  = FIXED_POSITIONS["normalize"]
                ylim_span = (sig_ax.get_ylim()[1] - sig_ax.get_ylim()[0])
                y_legs = max(max(v_sleep), max(v_norm)) + ylim_span * 0.03
                y_bar  = y_legs + ylim_span * 0.02
                sig_ax.plot([i_sleep, i_sleep, i_norm, i_norm],
                            [y_legs, y_bar, y_bar, y_legs], color="black", linewidth=1)
                sig_ax.annotate(lbl, xy=((i_sleep + i_norm) / 2, y_bar),
                                xytext=(0, -5), textcoords="offset points",
                                ha="center", va="bottom", fontsize=18)

            ax_top.set_title(MODE_DISPLAY.get(mode, mode), fontsize=26)

            if col != 0:
                ax_top.tick_params(axis="y", labelleft=False)
                ax_bot.tick_params(axis="y", labelleft=False)

        fig.supylabel(ylabel, fontsize=20, x=0.02)

        # Legend on middle top panel
        patches = [mpatches.Patch(color=COLORS[rt], alpha=0.75, label=LABELS[rt])
                   for rt in reg_types_present]
        from matplotlib.lines import Line2D
        patches.append(Line2D([0], [0], color=COLORS["none"], linewidth=1.5,
                               linestyle="--", label="No reg (baseline)"))
        top_axes[len(reg_modes) // 2].legend(
            handles=patches, loc="upper left", bbox_to_anchor=(0.05, 0.98),
            fontsize=14, title="Reg type", title_fontsize=16,
        )

        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {out_path}")
        plt.close(fig)
        return

    # ------------------------------------------------------------------ #
    # ORIGINAL SINGLE-AXIS PATH (unchanged below)                         #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        1, len(reg_modes), figsize=(4 * len(reg_modes), 5), sharey=True
    )
    if len(reg_modes) == 1:
        axes = [axes]

    if scale100:
        y_top = 105
    else:
        y_global_max = df_valid[value_col].max()
        y_top = y_global_max * 1.15

    for ax, mode in zip(axes, reg_modes):
        sub = df_valid[df_valid["reg_mode"] == mode]
        types_here = [t for t in ["sleep", "normalize"] if t in sub["reg_type"].values]

        FIXED_POSITIONS = {"sleep": 1, "normalize": 2}
        BOX_WIDTH = 0.5

        boxes_data = []
        box_colors = []
        box_labels = []
        positions = []

        for rt in types_here:
            raw = sub[sub["reg_type"] == rt][value_col].tolist()
            vals = [v * 100 for v in raw] if scale100 else raw
            boxes_data.append(vals)
            box_colors.append(COLORS[rt])
            box_labels.append(LABELS[rt])
            positions.append(FIXED_POSITIONS[rt])

        bp = ax.boxplot(
            boxes_data,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=4, linestyle="none"),
        )

        for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors)):
            dark = darken(color)
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(dark)
            bp["medians"][i].set_color(dark)
            bp["whiskers"][2 * i].set_color(dark)
            bp["whiskers"][2 * i + 1].set_color(dark)
            bp["caps"][2 * i].set_color(dark)
            bp["caps"][2 * i + 1].set_color(dark)
            bp["fliers"][i].set_markerfacecolor(color)
            bp["fliers"][i].set_markeredgecolor(dark)

        ax.axhline(
            no_reg_mean,
            color=COLORS["none"],
            linewidth=1.5,
            linestyle="--",
            zorder=3,
            label="none",
        )

        ax.set_ylim(0, y_top)
        y_span = y_top

        # Significance test between sleep and normalize (if both present)
        if "sleep" in types_here and "normalize" in types_here:
            raw_sleep = sub[sub["reg_type"] == "sleep"][value_col].tolist()
            raw_norm = sub[sub["reg_type"] == "normalize"][value_col].tolist()
            v_sleep = [v * 100 for v in raw_sleep] if scale100 else raw_sleep
            v_norm = [v * 100 for v in raw_norm] if scale100 else raw_norm
            _, p = mannwhitneyu(v_sleep, v_norm, alternative="two-sided")
            label = sig_label(p)

            i_sleep = positions[types_here.index("sleep")]
            i_norm = positions[types_here.index("normalize")]
            gap = y_span * -0.8
            bar_h = y_span * 0.02
            y_legs = max(max(v_sleep), max(v_norm)) + gap
            y_bar = y_legs + bar_h

            ax.plot(
                [i_sleep, i_sleep, i_norm, i_norm],
                [y_legs, y_bar, y_bar, y_legs],
                color="black",
                linewidth=1,
            )
            ax.text(
                (i_sleep + i_norm) / 2,
                y_bar + y_span * 0.005,
                label,
                ha="center",
                va="bottom",
                fontsize=18,
            )

        ax.set_title(MODE_DISPLAY.get(mode, mode), fontsize=26)
        ax.set_xticks(positions)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(box_labels, fontsize=22)
        if mode == reg_modes[0]:
            ax.set_ylabel(ylabel, fontsize=24)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS[rt], alpha=0.75, label=LABELS[rt])
        for rt in reg_types_present
    ]
    from matplotlib.lines import Line2D

    patches.append(
        Line2D(
            [0],
            [0],
            color=COLORS["none"],
            linewidth=1.5,
            linestyle="--",
            label="No reg (baseline)",
        )
    )
    axes[1].legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.6),
        fontsize=16,
        title="Reg type",
        title_fontsize=18,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Combined stacked plot (acc top row, phi bottom row, shared x-axis)
# ---------------------------------------------------------------------------
def plot_combined(df: pd.DataFrame, no_reg_df: pd.DataFrame, out_path: str):

    df_valid = df[df["test_acc"].notna() & df["test_phi"].notna()].copy()
    reg_modes = [m for m in MODE_ORDER if m in df_valid["reg_mode"].values]

    METRICS = [
        ("test_acc", "Accuracy", True),
        ("test_phi", "Clustering", False),
    ]
    FIXED_POSITIONS = {"sleep": 1, "normalize": 2}

    fig, axes = plt.subplots(
        2,
        len(reg_modes),
        figsize=(3.5 * len(reg_modes), 6),
        sharex="col",
        gridspec_kw={"hspace": 0.04, "wspace": 0.08},
    )

    # --- Significance bracket geometry (TWEAK THESE) ---------------------
    BRACKET_GAP_FRAC = 0.03
    BRACKET_LEG_FRAC = 0.02
    STAR_PAD_PTS = -5
    # --------------------------------------------------------------------

    # Pre-compute shared y ceilings per row — same formula for both metrics
    # so brackets sit at the same relative position in each row
    y_tops = []
    for value_col, _, scale100 in METRICS:
        col_max = df_valid[value_col].max()
        if scale100:
            col_max *= 100
        y_tops.append(col_max * 1.15)

    # Baseline means
    baselines = {
        "test_acc": no_reg_df["test_acc"].mean() * 100,
        "test_phi": no_reg_df["test_phi"].mean(),
    }

    for row_idx, (value_col, ylabel, scale100) in enumerate(METRICS):
        y_top = y_tops[row_idx]
        y_span = y_top

        for col_idx, mode in enumerate(reg_modes):
            ax = axes[row_idx][col_idx]
            sub = df_valid[df_valid["reg_mode"] == mode]
            types_here = [
                t for t in ["sleep", "normalize"] if t in sub["reg_type"].values
            ]

            boxes_data, box_colors, box_labels, positions = [], [], [], []
            for rt in types_here:
                raw = sub[sub["reg_type"] == rt][value_col].tolist()
                vals = [v * 100 for v in raw] if scale100 else raw
                boxes_data.append(vals)
                box_colors.append(COLORS[rt])
                box_labels.append(LABELS[rt])
                positions.append(FIXED_POSITIONS[rt])

            bp = ax.boxplot(
                boxes_data,
                positions=positions,
                widths=0.5,
                patch_artist=True,
                medianprops=dict(linewidth=2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=4, linestyle="none"),
            )
            for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors)):
                dark = darken(color)
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor(dark)
                bp["medians"][i].set_color(dark)
                bp["whiskers"][2 * i].set_color(dark)
                bp["whiskers"][2 * i + 1].set_color(dark)
                bp["caps"][2 * i].set_color(dark)
                bp["caps"][2 * i + 1].set_color(dark)
                bp["fliers"][i].set_markerfacecolor(color)
                bp["fliers"][i].set_markeredgecolor(dark)

            ax.axhline(
                baselines[value_col],
                color=COLORS["none"],
                linewidth=1.5,
                linestyle="--",
                zorder=3,
            )
            ax.set_ylim(0, y_top)

            # Significance bracket
            if "sleep" in types_here and "normalize" in types_here:
                v_sleep = (
                    [
                        v * 100
                        for v in sub[sub["reg_type"] == "sleep"][value_col].tolist()
                    ]
                    if scale100
                    else sub[sub["reg_type"] == "sleep"][value_col].tolist()
                )
                v_norm = (
                    [
                        v * 100
                        for v in sub[sub["reg_type"] == "normalize"][value_col].tolist()
                    ]
                    if scale100
                    else sub[sub["reg_type"] == "normalize"][value_col].tolist()
                )
                _, p = mannwhitneyu(v_sleep, v_norm, alternative="two-sided")
                label = sig_label(p)
                i_sleep = FIXED_POSITIONS["sleep"]
                i_norm = FIXED_POSITIONS["normalize"]
                gap = y_span * BRACKET_GAP_FRAC  # box top -> bracket legs
                bar_h = y_span * BRACKET_LEG_FRAC  # leg length (same per row)
                y_legs = max(max(v_sleep), max(v_norm)) + gap
                y_bar = y_legs + bar_h
                ax.plot(
                    [i_sleep, i_sleep, i_norm, i_norm],
                    [y_legs, y_bar, y_bar, y_legs],
                    color="black",
                    linewidth=1,
                )
                # Asterisk offset in POINTS (physical units) so the gap above the
                # bracket is identical for accuracy and phi regardless of scale.
                ax.annotate(
                    label,
                    xy=((i_sleep + i_norm) / 2, y_bar),
                    xytext=(0, STAR_PAD_PTS),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                )

            # Titles only on top row
            if row_idx == 0:
                ax.set_title(MODE_DISPLAY.get(mode, mode), fontsize=28)

            # y-label only on left column
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=20)

            # x-tick labels only on bottom row
            ax.set_xlim(0.5, 2.5)
            ax.set_xticks([1, 2])
            if row_idx == len(METRICS) - 1:
                ax.set_xticklabels(box_labels, fontsize=20)
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.tick_params(axis="y", labelsize=12)
            else:
                ax.tick_params(axis="y", labelsize=12, labelleft=False)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

    # Legend inside top-middle panel
    from matplotlib.lines import Line2D

    legend_handles = [
        mpatches.Patch(color=COLORS[rt], alpha=0.75, label=LABELS[rt])
        for rt in ["sleep", "normalize"]
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=COLORS["none"],
            linewidth=1.5,
            linestyle="--",
            label="No reg (baseline)",
        )
    )
    axes[1][1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.6),
        fontsize=12,
        title="Reg type",
        title_fontsize=14,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Combined plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Merged single-row boxplot (dual y-axes, 4 boxes per reg_mode panel)
# ---------------------------------------------------------------------------
def plot_combined_merged(df: pd.DataFrame, no_reg_df: pd.DataFrame, out_path: str):
    """
    One row × 3 panels (layer, neuron, static).
    Each panel: 4 boxes grouped by metric — [acc_sleep, acc_norm] | [phi_sleep, phi_norm].
    Left y-axis = accuracy (%), right y-axis = clustering (φ).
    Color encodes metric (viridis=acc, plasma=phi); fill encodes reg type (filled=sleep, hollow=normalize).
    """
    ACC_COLOR = mcolors.to_hex(plt.cm.viridis(0.35))
    PHI_COLOR = mcolors.to_hex(plt.cm.plasma(0.60))
    NONE_COLOR = COLORS["none"]

    # x-positions within each panel (grouped by metric)
    POS = {
        ("acc", "sleep"): 1.00,
        ("acc", "normalize"): 1.55,
        ("phi", "sleep"): 2.25,
        ("phi", "normalize"): 2.80,
    }
    BOX_WIDTH = 0.45

    df_valid = df[df["test_acc"].notna() & df["test_phi"].notna()].copy()
    reg_modes = [m for m in MODE_ORDER if m in df_valid["reg_mode"].values]

    fig, axes = plt.subplots(
        1,
        len(reg_modes),
        figsize=(3.5 * len(reg_modes), 5),
    )
    if len(reg_modes) == 1:
        axes = [axes]

    baselines = {
        "acc": no_reg_df["test_acc"].mean() * 100,
        "phi": no_reg_df["test_phi"].mean(),
    }

    # Fixed y-limits computed globally so all panels share the same scale
    ACC_YMAX = 105.0
    ACC_YMIN = 0.0
    phi_all = df_valid["test_phi"]
    PHI_YMIN = phi_all.min() * 0.85
    PHI_YMAX = phi_all.max() * 1.15

    ax2_list = []
    for col_idx, (ax, mode) in enumerate(zip(axes, reg_modes)):
        ax2 = ax.twinx()
        ax2_list.append(ax2)

        sub = df_valid[df_valid["reg_mode"] == mode]

        for rt in ["sleep", "normalize"]:
            sub_rt = sub[sub["reg_type"] == rt]
            if sub_rt.empty:
                continue

            filled = rt == "sleep"
            acc_face = ACC_COLOR if filled else "none"
            phi_face = PHI_COLOR if filled else "none"
            edge_lw = 1.5 if not filled else 1.0

            # --- Accuracy box (left axis) ---
            acc_vals = (sub_rt["test_acc"] * 100).tolist()
            bp_acc = ax.boxplot(
                [acc_vals],
                positions=[POS[("acc", rt)]],
                widths=BOX_WIDTH,
                patch_artist=True,
                medianprops=dict(linewidth=2, color=darken(ACC_COLOR)),
                whiskerprops=dict(linewidth=0.8, color=darken(ACC_COLOR)),
                capprops=dict(linewidth=0.8, color=darken(ACC_COLOR)),
                flierprops=dict(
                    marker="o",
                    markersize=3,
                    linestyle="none",
                    markerfacecolor=acc_face if filled else ACC_COLOR,
                    markeredgecolor=darken(ACC_COLOR),
                ),
            )
            p = bp_acc["boxes"][0]
            p.set_facecolor(acc_face)
            p.set_alpha(0.75 if filled else 1.0)
            p.set_edgecolor(darken(ACC_COLOR))
            p.set_linewidth(edge_lw)

            # --- Clustering box (right axis) ---
            phi_vals = sub_rt["test_phi"].tolist()
            bp_phi = ax2.boxplot(
                [phi_vals],
                positions=[POS[("phi", rt)]],
                widths=BOX_WIDTH,
                patch_artist=True,
                medianprops=dict(linewidth=2, color=darken(PHI_COLOR)),
                whiskerprops=dict(linewidth=0.8, color=darken(PHI_COLOR)),
                capprops=dict(linewidth=0.8, color=darken(PHI_COLOR)),
                flierprops=dict(
                    marker="o",
                    markersize=3,
                    linestyle="none",
                    markerfacecolor=phi_face if filled else PHI_COLOR,
                    markeredgecolor=darken(PHI_COLOR),
                ),
            )
            p = bp_phi["boxes"][0]
            p.set_facecolor(phi_face)
            p.set_alpha(0.75 if filled else 1.0)
            p.set_edgecolor(darken(PHI_COLOR))
            p.set_linewidth(edge_lw)

        # Baselines
        ax.axhline(
            baselines["acc"], color=NONE_COLOR, linewidth=1.5, linestyle="--", zorder=3
        )
        ax2.axhline(
            baselines["phi"],
            color=NONE_COLOR,
            linewidth=1.5,
            linestyle="--",
            zorder=3,
            alpha=0.5,
        )

        # Enforce consistent scales across all panels
        ax.set_ylim(ACC_YMIN, ACC_YMAX)
        ax2.set_ylim(PHI_YMIN, PHI_YMAX)

        # x-axis: group labels at midpoints of each metric pair
        ax.set_xlim(0.5, 3.3)
        ax.set_xticks([1.275, 2.525])
        ax.set_xticklabels(["Accuracy", "Clustering (φ)"], fontsize=12)

        # y-axis ticks
        ax.tick_params(axis="y", labelsize=12)
        ax2.tick_params(axis="y", labelsize=12)

        # Grid on primary axis only
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Titles
        ax.set_title(MODE_DISPLAY.get(mode, mode), fontsize=28)

        # y-labels: left only on first panel, right only on last
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=20)
        else:
            ax.tick_params(axis="y", labelleft=False)

        if col_idx == len(reg_modes) - 1:
            ax2.set_ylabel("Clustering (φ)", fontsize=20)
        else:
            ax2.tick_params(axis="y", labelright=False)

    # Legend on middle panel
    from matplotlib.lines import Line2D

    legend_ax = axes[len(reg_modes) // 2]
    legend_handles = [
        mpatches.Patch(
            facecolor=ACC_COLOR,
            alpha=0.75,
            edgecolor=darken(ACC_COLOR),
            label="Accuracy",
        ),
        mpatches.Patch(
            facecolor=PHI_COLOR,
            alpha=0.75,
            edgecolor=darken(PHI_COLOR),
            label="Clustering (φ)",
        ),
        mpatches.Patch(
            facecolor="grey", alpha=0.75, edgecolor="dimgrey", label="Sleep (filled)"
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor="dimgrey",
            linewidth=1.5,
            label="Normalize (hollow)",
        ),
        Line2D(
            [0],
            [0],
            color=NONE_COLOR,
            linewidth=1.5,
            linestyle="--",
            label="No reg (baseline)",
        ),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="lower center",
        fontsize=12,
        title="Legend",
        title_fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Merged plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_results()

    print("\n=== Raw results ===")
    print(df.to_string(index=False))

    # Summary table
    df_ok = df[df["test_acc"].notna() & df["test_phi"].notna()]
    summary = (
        df_ok.groupby(["reg_type", "reg_mode"])
        .agg(
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            mean_phi=("test_phi", "mean"),
            std_phi=("test_phi", "std"),
            n=("test_acc", "count"),
        )
        .reset_index()
    )

    # Add per-mode significance (sleep vs normalize) for acc and phi
    sig_rows = []
    for mode in summary["reg_mode"].unique():
        sleep_rows = df_ok[(df_ok["reg_mode"] == mode) & (df_ok["reg_type"] == "sleep")]
        norm_rows = df_ok[
            (df_ok["reg_mode"] == mode) & (df_ok["reg_type"] == "normalize")
        ]
        if len(sleep_rows) and len(norm_rows):
            _, p_acc = mannwhitneyu(
                sleep_rows["test_acc"], norm_rows["test_acc"], alternative="two-sided"
            )
            _, p_phi = mannwhitneyu(
                sleep_rows["test_phi"], norm_rows["test_phi"], alternative="two-sided"
            )
            sig_rows.append(
                {
                    "reg_mode": mode,
                    "p_acc": round(p_acc, 4),
                    "sig_acc": sig_label(p_acc),
                    "p_phi": round(p_phi, 4),
                    "sig_phi": sig_label(p_phi),
                }
            )
    sig_df = pd.DataFrame(sig_rows)

    print("\n=== Summary (mean ± std) ===")
    for _, row in summary.iterrows():
        print(
            f"  {row['reg_type']:10s} / {row['reg_mode']:7s}  "
            f"acc={row['mean_acc']:.4f}±{row['std_acc']:.4f}  "
            f"phi={row['mean_phi']:.2f}±{row['std_phi']:.2f}  (n={int(row['n'])})"
        )

    print("\n=== Significance: sleep vs normalize (Mann-Whitney U) ===")
    for _, row in sig_df.iterrows():
        print(
            f"  {row['reg_mode']:7s}  acc p={row['p_acc']:.4f} {row['sig_acc']}  "
            f"phi p={row['p_phi']:.4f} {row['sig_phi']}"
        )

    # No-reg baseline stats from phase 2 EXT
    phase2_path = os.path.join(
        REPO_ROOT,
        "results",
        "acc_history",
        "mnist",
        "2026.05.24",
        "21",
        "Results_phase2 EXT.xlsx",
    )
    no_reg = pd.read_excel(phase2_path)
    no_reg = no_reg[no_reg["reg_mode"] == "none"]
    print("\n=== No-reg baseline (from phase 2 EXT) ===")
    print(
        f"  acc={no_reg['test_acc'].mean():.4f}±{no_reg['test_acc'].std():.4f}  "
        f"phi={no_reg['test_phi'].mean():.2f}±{no_reg['test_phi'].std():.2f}  "
        f"(n={len(no_reg)})"
    )

    # Save xlsx
    xlsx_path = os.path.join(OUT_DIR, "phase1_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"\nXLSX saved -> {xlsx_path}")

    # Combined stacked plot (2×3 grid, original)
    combined_path = os.path.join(OUT_DIR, "phase1_boxplot_combined.pdf")
    plot_combined(df, no_reg, combined_path)

    # Merged single-row plot (dual y-axes, 4 boxes per panel)
    merged_path = os.path.join(OUT_DIR, "phase1_boxplot_merged.pdf")
    plot_combined_merged(df, no_reg, merged_path)

    # Accuracy boxplot (broken y-axis)
    png_path = os.path.join(OUT_DIR, "phase1_boxplot.pdf")
    plot_results(
        df, png_path, value_col="test_acc", ylabel="Accuracy (%)",
        scale100=True, broken_axis=True,
    )

    # Clustering score (phi) boxplot
    phi_path = os.path.join(OUT_DIR, "phase1_boxplot_phi.pdf")
    plot_results(
        df,
        phi_path,
        value_col="test_phi",
        ylabel="Clustering",
        scale100=False,
    )


if __name__ == "__main__":
    main()
