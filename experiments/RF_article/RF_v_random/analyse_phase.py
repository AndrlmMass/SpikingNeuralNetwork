"""
Analyse Phase 1 results from SLURM output files — RF vs random weights.

Usage
-----
python experiments/RF_article/RF_v_random/analyse_phase.py
"""

import glob
import os
import re
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SLURM_DIR = os.path.join(
    REPO_ROOT, "results", "acc_history", "mnist", "2026.05.27", "27"
)
OUT_DIR = SLURM_DIR

# Task ID → (weight_type, seed)  — matches slurm_phase1.sh
# Tasks 0-9: rf, seeds 0-9; Tasks 10-19: random, seeds 0-9
_WEIGHT_TYPES = ["rf"] * 10 + ["random"] * 10
_SEEDS = list(range(10)) * 2


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
    for task_id in range(len(_WEIGHT_TYPES)):
        weight_type = _WEIGHT_TYPES[task_id]
        seed = _SEEDS[task_id]

        fpath = slurm_files.get(task_id)
        if fpath is None:
            rows.append(
                dict(
                    task_id=task_id,
                    weight_type=weight_type,
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
                weight_type=weight_type,
                seed=seed,
                test_acc=test_acc,
                test_phi=test_phi,
                cancelled=cancelled,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {"rf": "#4C72B0", "random": "#DD8452"}
LABELS = {"rf": "Receptive\nFields", "random": "Random\nWeights"}

TYPE_ORDER = ["rf", "random"]


def darken(color: str, factor: float = 0.65) -> tuple:
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


def plot_results(
    df: pd.DataFrame,
    out_path: str,
    value_col: str = "test_acc",
    ylabel: str = "Accuracy (%)",
    scale100: bool = True,
):
    df_valid = df[df[value_col].notna()].copy()
    types_present = [t for t in TYPE_ORDER if t in df_valid["weight_type"].values]

    fig, ax = plt.subplots(figsize=(4, 5))

    boxes_data = []
    box_colors = []
    box_labels = []
    positions = list(range(1, len(types_present) + 1))

    for i, wt in enumerate(types_present):
        raw = df_valid[df_valid["weight_type"] == wt][value_col].tolist()
        vals = [v * 100 for v in raw] if scale100 else raw
        boxes_data.append(vals)
        box_colors.append(COLORS[wt])
        box_labels.append(LABELS[wt])

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

    # Significance test between rf and random (if both present)
    if "rf" in types_present and "random" in types_present:
        raw_rf = df_valid[df_valid["weight_type"] == "rf"][value_col].tolist()
        raw_rand = df_valid[df_valid["weight_type"] == "random"][value_col].tolist()
        v_rf = [v * 100 for v in raw_rf] if scale100 else raw_rf
        v_rand = [v * 100 for v in raw_rand] if scale100 else raw_rand
        _, p = mannwhitneyu(v_rf, v_rand, alternative="two-sided")
        label = sig_label(p)

        i_rf = positions[types_present.index("rf")]
        i_rand = positions[types_present.index("random")]
        data_range = max(max(v_rf), max(v_rand)) - min(min(v_rf), min(v_rand))
        y_max = max(max(v_rf), max(v_rand)) + data_range * 0.05
        y_bar = y_max + data_range * 0.03

        ax.plot(
            [i_rf, i_rf, i_rand, i_rand],
            [y_max, y_bar, y_bar, y_max],
            color="black",
            linewidth=1,
        )
        ax.text(
            (i_rf + i_rand) / 2,
            y_bar * 1.005,
            label,
            ha="center",
            va="bottom",
            fontsize=18,
        )

    ax.set_xticks(positions)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticklabels(box_labels, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=22)
    # if scale100:
    #     ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

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

    summary = (
        df[df["test_acc"].notna()]
        .groupby("weight_type")["test_acc"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_acc", "std": "std_acc", "count": "n"})
        .reset_index()
    )
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    xlsx_path = os.path.join(OUT_DIR, "rf_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"\nXLSX saved -> {xlsx_path}")

    pdf_path = os.path.join(OUT_DIR, "rf_boxplot.pdf")
    plot_results(
        df, pdf_path, value_col="test_acc", ylabel="Accuracy (%)", scale100=True
    )

    phi_path = os.path.join(OUT_DIR, "rf_boxplot_phi.pdf")
    plot_results(
        df,
        phi_path,
        value_col="test_phi",
        ylabel="Clustering score (φ)",
        scale100=False,
    )


if __name__ == "__main__":
    main()
