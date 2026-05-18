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
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

        rows.append(
            dict(
                task_id=task_id,
                reg_type=reg_type,
                reg_mode=reg_mode,
                seed=seed,
                test_acc=test_acc,
                cancelled=cancelled,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {"sleep": "#4C72B0", "normalize": "#DD8452", "none": "#55A868"}
LABELS = {"sleep": "Sleep", "normalize": "Normalize"}


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


MODE_ORDER = ["static", "layer", "neuron"]
MODE_DISPLAY = {"static": "Static", "layer": "Layer-wise", "neuron": "Neuron-wise"}


def plot_results(df: pd.DataFrame, out_path: str):
    # Only rows with valid test accuracy
    df_valid = df[df["test_acc"].notna()].copy()

    reg_modes = [m for m in MODE_ORDER if m in df_valid["reg_mode"].values]
    reg_types_present = [
        t for t in ["sleep", "normalize", "none"] if t in df_valid["reg_type"].values
    ]

    fig, axes = plt.subplots(
        1, len(reg_modes), figsize=(4 * len(reg_modes), 5), sharey=True
    )
    if len(reg_modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, reg_modes):
        sub = df_valid[df_valid["reg_mode"] == mode]
        types_here = [t for t in ["sleep", "normalize"] if t in sub["reg_type"].values]

        boxes_data = []
        box_colors = []
        box_labels = []
        positions = []

        for i, rt in enumerate(types_here):
            vals = sub[sub["reg_type"] == rt]["test_acc"].tolist()
            boxes_data.append(vals)
            box_colors.append(COLORS[rt])
            box_labels.append(LABELS[rt])
            positions.append(i + 1)

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

        # Significance test between sleep and normalize (if both present)
        if "sleep" in types_here and "normalize" in types_here:
            v_sleep = sub[sub["reg_type"] == "sleep"]["test_acc"].tolist()
            v_norm = sub[sub["reg_type"] == "normalize"]["test_acc"].tolist()
            _, p = mannwhitneyu(v_sleep, v_norm, alternative="two-sided")
            label = sig_label(p)

            i_sleep = positions[types_here.index("sleep")]
            i_norm = positions[types_here.index("normalize")]
            y_max = max(max(v_sleep), max(v_norm)) * 1.05
            y_bar = y_max * 1.03

            ax.plot(
                [i_sleep, i_sleep, i_norm, i_norm],
                [y_max, y_bar, y_bar, y_max],
                color="black",
                linewidth=1,
            )
            ax.text(
                (i_sleep + i_norm) / 2,
                y_bar * 1.005,
                label,
                ha="center",
                va="bottom",
                fontsize=13,
            )

        ax.set_title(MODE_DISPLAY.get(mode, mode), fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels, fontsize=11)
        if mode == reg_modes[0]:
            ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS[rt], alpha=0.75, label=LABELS[rt])
        for rt in reg_types_present
    ]
    fig.legend(
        handles=patches,
        loc="upper right",
        fontsize=10,
        title="Reg type",
        title_fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_results()

    print("\n=== Raw results ===")
    print(df.to_string(index=False))

    # Summary table
    summary = (
        df[df["test_acc"].notna()]
        .groupby(["reg_type", "reg_mode"])["test_acc"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_acc", "std": "std_acc", "count": "n"})
        .reset_index()
    )
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # Save xlsx
    xlsx_path = os.path.join(OUT_DIR, "phase1_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"\nXLSX saved -> {xlsx_path}")

    # Plot
    png_path = os.path.join(OUT_DIR, "phase1_boxplot.svg")
    plot_results(df, png_path)


if __name__ == "__main__":
    main()
