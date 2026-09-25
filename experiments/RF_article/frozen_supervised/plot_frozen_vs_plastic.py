"""
Supervised model, five datasets: oriented-RF vs random init vs FROZEN, one box per cell.

Adds the third condition the phase-2 figure never had. Phase 2 compared two PLASTIC arms
(oriented vs random W_se init under R-STDP); with no frozen arm there was no way to say
whether plasticity contributed anything at all in the supervised model -- the only frozen
evidence came from phase 1, which differs in six config parameters at once.

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

PHASE2 = os.path.join(REPO, "results", "interp", "json", "experiments", "RF_article",
                      "interp", "mnist_family_sweep", "results", "run_20260819_112250")
# The 08-19 random arm silently got global instead of within-group I->E inhibition
# (fixed in 9159d743); its rerun lives in its own run folder and replaces it.
PHASE2_RANDOM = os.path.join(os.path.dirname(PHASE2), "run_20260916_105832")
FROZEN = os.path.join(REPO, "results", "frozen_supervised", "run_cheap")
DATASETS = ["mnist", "kmnist", "fmnist", "notmnist"]
PRETTY = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "kmnist": "KMNIST",
          "notmnist": "notMNIST"}
LABELS = {"oriented": "RF prior\n+ R-STDP", "random": "random init\n+ R-STDP",
          "frozen": "RF prior\nFROZEN"}
FLAT = {"oriented": "RF prior + R-STDP", "random": "random init + R-STDP",
        "frozen": "RF prior, frozen"}          # single-line, for the stdout table
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
    paths = (glob.glob(os.path.join(PHASE2, "*_oriented_s*", "results.json"))
             + glob.glob(os.path.join(PHASE2_RANDOM, "*_random_s*", "results.json")))
    for p in sorted(paths):
        tag = os.path.basename(os.path.dirname(p))
        ds, prior, _ = tag.rsplit("_", 2)
        a = learned_readout(p)
        if a is not None and ds in out and prior in out[ds]:
            out[ds][prior].append(a)
    for p in sorted(glob.glob(os.path.join(FROZEN, "*", "results.json"))):
        try:
            ds = json.load(open(p))["config"]["dataset"]
        except Exception:
            continue
        a = learned_readout(p)
        if a is not None and ds in out:
            out[ds]["frozen"].append(a)
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
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=200, facecolor=T["surface"],
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
