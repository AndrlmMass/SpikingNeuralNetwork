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
FROZEN = os.path.join(REPO, "results", "frozen_supervised", "run_cheap")
DATASETS = ["mnist", "fmnist", "kmnist", "notmnist", "svhn"]
LABELS = {"oriented": "RF prior\n+ R-STDP", "random": "random init\n+ R-STDP",
          "frozen": "RF prior\nFROZEN"}
FLAT = {"oriented": "RF prior + R-STDP", "random": "random init + R-STDP",
        "frozen": "RF prior, FROZEN"}          # single-line, for the stdout table
ORDER = ["frozen", "random", "oriented"]     # weakest-claim first, left to right


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
    for p in sorted(glob.glob(os.path.join(PHASE2, "*", "results.json"))):
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


def figure(data, out_path, theme="light", palette="project",
           lo_band=(0.08, 0.30), hi_band=(0.48, 1.00)):
    """Broken y-axis: SVHN sits near the 10% floor while everything else is above 50%.

    On a single continuous axis the 18 percentage points between SVHN's best arm and
    KMNIST's worst are empty, and that gap consumes roughly a third of the panel while
    compressing the differences the figure exists to show. Splitting the axis spends the
    vertical space where the data is.

    The break is drawn explicitly with diagonal marks on both broken spines, because a
    reader who misses it reads SVHN as competitive with the MNIST family. Both bands keep
    the SAME percentage-point-per-inch scale as far as the band widths allow, so box
    heights remain visually comparable across the break.
    """
    T, cols = THEMES[theme], PALETTES[palette]
    colour = {"frozen": cols[2], "random": cols[1], "oriented": cols[0]}
    # height ratio follows the band spans, so a given accuracy difference occupies the
    # same vertical distance above and below the break
    h_hi, h_lo = hi_band[1] - hi_band[0], lo_band[1] - lo_band[0]
    fig, (ax, axb) = plt.subplots(
        2, 1, sharex=True, figsize=(16, 8.8), facecolor=T["surface"],
        gridspec_kw=dict(height_ratios=[h_hi, h_lo], hspace=0.07))

    width, gap = 0.24, 1.0
    for i, ds in enumerate(DATASETS):
        for j, cond in enumerate(ORDER):
            vals = data[ds][cond]
            if not vals:
                continue
            x = i * gap + (j - 1) * width
            for a in (ax, axb):           # draw on both; each shows only its own band
                # median in the SERIES colour: a black bar reads as a separate mark and,
                # at this type scale, competes with the box it belongs to
                bp = a.boxplot([vals], positions=[x], widths=width * 0.82,
                               patch_artist=True,
                               medianprops=dict(color=colour[cond], lw=3.0),
                               whiskerprops=dict(color=colour[cond], lw=1.4),
                               capprops=dict(color=colour[cond], lw=1.4),
                               flierprops=dict(marker="", ms=0), zorder=3)
                for b in bp["boxes"]:
                    b.set(facecolor=colour[cond], alpha=0.35,
                          edgecolor=colour[cond], lw=1.6)
                # every seed as a dot: with 3-5 seeds a box alone hides how much of its
                # spread is one outlier, and SVHN's frozen arm is exactly that case
                a.scatter(np.full(len(vals), x) + np.linspace(-0.04, 0.04, len(vals)),
                          vals, s=30, color=colour[cond], zorder=4,
                          edgecolors=T["surface"], linewidths=0.6)

    # chance floor lives in the lower band, where SVHN is
    axb.axhline(0.10, color=T["ref"], lw=1.2, ls=(0, (4, 4)), zorder=1)
    axb.text(-0.45, 0.105, " chance", color=T["muted"], fontsize=F_TICK,
             va="bottom", ha="left")

    # Direct condition labels, one per condition, each anchored on a DIFFERENT dataset
    # group with a leader line. Stacking all three over the first group collides -- the
    # boxes are 0.24 apart and the labels are wider than that at this type scale -- and a
    # legend is not an option, because this palette separates rose from sage by only
    # dE 3.5 under deuteranopia and so cannot carry identity by hue.
    top = max(max(v) for d in data.values() for v in d.values() if v)
    anchor_group = {"frozen": 0, "random": 1, "oriented": 2}
    for cond, gi in anchor_group.items():
        vals = data[DATASETS[gi]][cond]
        if not vals:
            continue
        x = gi * gap + (ORDER.index(cond) - 1) * width
        y_lab = top + 0.055
        ax.plot([x, x], [max(vals) + 0.008, y_lab - 0.008], color=colour[cond], lw=1.0,
                zorder=2)
        ax.text(x, y_lab, LABELS[cond], color=colour[cond], fontsize=F_SERIES,
                ha="center", va="bottom", linespacing=1.1)

    ax.set_ylim(*hi_band)
    axb.set_ylim(*lo_band)
    axb.set_xticks([i * gap for i in range(len(DATASETS))])
    axb.set_xticklabels(DATASETS, fontsize=F_TICK)
    axb.set_xlim(-0.55, len(DATASETS) - 1 + 0.55)
    for a in (ax, axb):
        style_axes(a, T, "", "", fs_label=F_LABEL, fs_tick=F_TICK)
        a.yaxis.set_major_formatter(plt.FuncFormatter(pct))
    # one y label for the pair, centred on the break
    fig.supylabel("learned-readout test accuracy", fontsize=F_LABEL, color=T["ink"],
                  x=0.005)

    # hide the facing spines and mark the break
    ax.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    ax.tick_params(bottom=False)
    d = 0.9
    kw = dict(marker=[(-1, -d), (1, d)], markersize=13, linestyle="none",
              color=T["axis"], mec=T["axis"], mew=1.4, clip_on=False)
    ax.plot([0], [0], transform=ax.transAxes, **kw)
    axb.plot([0], [1], transform=axb.transAxes, **kw)

    # NOT tight_layout: it cannot handle the broken-axis pair (it warns that results may
    # be incorrect) and it would override the hspace that sets the break gap. bbox_inches
    # trims the margins without touching the subplot geometry.
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
