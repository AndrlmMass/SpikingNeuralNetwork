"""
Supervised model, four datasets: Plastic RF vs Random vs Frozen, one box per cell.

Plastic RF (RF weights trained with R-Hebb) is the Experiment 2 reference; Random replaces
the RFs with random weights (still R-Hebb) and Frozen keeps the RFs fixed (R-Hebb off,
readout still learns). The run folders are the same ones build_model_data.py feeds the
GLMM, so the figure and the table show the same runs.

Every box is the SAME decoder: the learned dense softmax-delta readout, on the test set.
Never `test_acc`, which is the PCA+LR evaluator probe in the harness schema and the learned
readout in run_frozen's -- reading one field across both is the 08-25 naming trap.

Style follows plot_risk_coverage.py: project palette (muted rose/slate/sage), no title, no
legend, direct labels. The palette REQUIRES direct labels -- rose and sage separate by only
dE 3.5 under deuteranopia, so identity must never rest on hue alone.

Usage:
    python experiments/RF_article/frozen_supervised/plot_frozen_vs_plastic.py
"""
import argparse, glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "RF_article", "interp"))
from plot_risk_coverage import PALETTES, THEMES, FS_LABEL, FS_TICK, FS_SERIES, style_axes, pct  # noqa: E402

# This figure is reproduced smaller than the risk-coverage ones, so it carries its own
# type scale rather than mutating the shared constants, which every other figure imports.
TEXT_SCALE = 1.5
F_LABEL, F_TICK, F_SERIES = (int(round(v * TEXT_SCALE))
                             for v in (FS_LABEL, FS_TICK, FS_SERIES))

RUNS = os.path.join(REPO, "results", "interp", "json", "experiments", "RF_article",
                    "interp", "mnist_family_sweep", "results")
DATASETS = ["mnist", "kmnist", "fmnist", "notmnist"]
# (run folder, arm, datasets), as in build_model_data.py. Random on MNIST/FMNIST/KMNIST is
# the 09-16 rerun with within-group I->E inhibition (9159d743); notMNIST random and every
# frozen run are the 10-27 runs, on the same 2500 notMNIST test images as Plastic RF.
SOURCES = [
    ("run_20260819_112250", "oriented", DATASETS),
    ("run_20260916_105832", "random", ["mnist", "fmnist", "kmnist"]),
    ("run_20261027_111346", "random", ["notmnist"]),
    ("run_20261027_111346", "frozen", DATASETS),
]
PRETTY = {"mnist": "MNIST", "fmnist": "FMNIST", "kmnist": "KMNIST",
          "notmnist": "notMNIST"}
FLAT = {"oriented": "Plastic RF", "random": "Random", "frozen": "Frozen"}
ORDER = ["frozen", "random", "oriented"]     # weakest-claim first, left to right
N_YTICKS = 5                                 # same tick count in every panel


def learned_readout(path):
    """The learned dense readout's test accuracy -- the article's decoder, in both schemas."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    for u in d.get("uncertainty", []):
        if u.get("readout") == "learned_readout":
            return float(u["base_acc"])
    return None


def collect():
    """{dataset: {condition: [per-seed accuracies]}}"""
    out = {ds: {c: [] for c in ORDER} for ds in DATASETS}
    for run, arm, datasets in SOURCES:
        for ds in datasets:
            for p in sorted(glob.glob(os.path.join(RUNS, run, f"{ds}_{arm}_s*", "results.json"))):
                a = learned_readout(p)
                if a is not None:
                    out[ds][arm].append(a)
    return out


def figure(data, out_path, theme="light", palette="project"):
    """One panel per dataset, each on its own y-range (black and white, legend on top).

    On a single shared axis (52-95%) the 5-seed boxes are ~1 pp tall and their fill/hatch
    cannot be told apart in greyscale. Per-dataset ranges make every box several times
    taller. The price is that absolute heights are no longer comparable across panels, so
    each panel keeps its own labelled y-ticks.
    """
    T = THEMES[theme]
    FILL = {"frozen": ("#d9d9d9", ""), "random": ("white", "///"), "oriented": ("none", "")}
    MARK = {"frozen": "s", "random": "^", "oriented": "o"}
    SHORT = {"frozen": "frozen", "random": "random", "oriented": "RF"}
    plt.rcParams["hatch.linewidth"] = 1.1
    nds = len(DATASETS)
    fig, axes = plt.subplots(1, nds, figsize=(6.2 * nds, 5.6), facecolor=T["surface"])
    width = 0.62
    for ax, ds in zip(axes, DATASETS):
        allv = []
        for j, cond in enumerate(ORDER):
            vals = data[ds][cond]
            if not vals:
                continue
            allv += vals
            fc, hatch = FILL[cond]
            bp = ax.boxplot([vals], positions=[j], widths=width, patch_artist=True,
                            medianprops=dict(color="black", lw=2.4),
                            whiskerprops=dict(color="black", lw=1.2),
                            capprops=dict(color="black", lw=1.2),
                            flierprops=dict(marker="", ms=0), zorder=3)
            for b in bp["boxes"]:
                b.set(facecolor=fc, hatch=hatch, edgecolor="black", lw=1.4)
            # seeds on the box, spread slightly so coincident values stay visible
            ax.scatter(j + np.linspace(-0.12, 0.12, len(vals)), vals, s=30,
                       marker=MARK[cond], facecolors="white", edgecolors="black",
                       linewidths=1.0, zorder=4, clip_on=False)
        # Exactly N_YTICKS ticks in every panel, on whole percents: the smallest step whose
        # N_YTICKS-1 intervals cover the data, with the data centred between the end ticks.
        lo, hi = 100 * min(allv), 100 * max(allv)
        step = next(s for s in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
                    if (N_YTICKS - 1) * s >= np.ceil(hi) - np.floor(lo))
        start = np.floor((lo + hi) / 2 - (N_YTICKS - 1) * step / 2)
        start = min(max(start, np.ceil(hi) - (N_YTICKS - 1) * step), np.floor(lo))
        ticks = (start + step * np.arange(N_YTICKS)) / 100
        edge = 0.04 * (ticks[-1] - ticks[0])
        style_axes(ax, T, "", "", fs_label=F_TICK, fs_tick=F_TICK - 4)
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0] - edge, ticks[-1] + edge)
        ax.set_xticks([])   # identity comes from hatch + marker, named in the legend
        ax.set_xlim(-0.6, len(ORDER) - 0.4)   # symmetric, so the title centres on the boxes
        ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.set_title(PRETTY[ds], fontsize=F_TICK - 2, color=T["ink"], pad=8)
    axes[0].set_ylabel("Accuracy", fontsize=(F_TICK - 3) * 1.2, color=T["ink"], labelpad=8)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FILL[c][0], hatch=FILL[c][1], edgecolor="black", lw=1.2,
                     label=FLAT[c]) for c in ORDER[::-1]]   # strongest claim first
    fig.subplots_adjust(left=0.07, right=0.80, bottom=0.06, top=0.90, wspace=0.62)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.815, 0.5),
               fontsize=F_TICK - 4, frameon=True, edgecolor="#666666", fancybox=False)
    for ext in ("png", "pdf"):   # _BW is the name the article includes
        for path in (out_path, out_path + "_BW"):
            fig.savefig(f"{path}.{ext}", dpi=200, facecolor=T["surface"],
                        bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path + ".png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "figures",
                                                  "frozen_vs_plastic"))
    ap.add_argument("--theme", default="light", choices=tuple(THEMES))
    ap.add_argument("--palette", default="project", choices=tuple(PALETTES))
    a = ap.parse_args()

    data = collect()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(f"{'dataset':<10} " + " ".join(f"{FLAT[c]:>22}" for c in ORDER))
    for ds in DATASETS:
        cells = []
        for c in ORDER:
            v = data[ds][c]
            cells.append(f"{np.mean(v):.4f} +-{np.std(v, ddof=1):.4f} ({len(v)})"
                         if len(v) > 1 else "--")
        print(f"{ds:<10} " + " ".join(f"{c:>22}" for c in cells))
    p = figure(data, a.out, a.theme, a.palette)
    print(f"\n[plot] wrote {p} (+ .pdf)")


if __name__ == "__main__":
    main()
