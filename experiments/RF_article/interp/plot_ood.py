"""
OOD figure: can the network's own confidence tell it is looking at the wrong dataset?

Two panels, because separability and usability are different questions and a paper that
reports only the first overstates the result:

  left   AUROC of three detectors against each OOD set. The network's confidence is shown
         against two baselines that IGNORE THE NETWORK ENTIRELY -- mean image intensity,
         and the residual outside MNIST's top-50 pixel PCA subspace. If a detector with no
         network in it matches the SNN, the finding is about image statistics, not novelty.
  right  What the network actually rejects at a threshold calibrated on in-distribution
         data alone (alpha = 0.025), against the in-distribution coverage that costs. A
         detector that rejects most OOD by also refusing valid input has not solved anything.

The KMNIST column carries the one positive result and the figure is built to make it
visible: KMNIST is the only probe whose evoked firing rate matches MNIST's (0.00213 vs
0.00225), so the brightness baseline collapses there while the network does not.

Reads ood.json and input_density.json; transcribing those numbers by hand is how a
figure and a table end up disagreeing.

Usage:
    python experiments/RF_article/interp/plot_ood.py --run results/ood_mnist/run1/mnist_ood_s0
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_risk_coverage import PALETTES, THEMES, FS_TICK, FS_SERIES, FS_LABEL, style_axes, pct  # noqa: E402

# Local type scale: this figure is reproduced smaller than the risk-coverage ones, so it
# carries its own sizes rather than mutating the shared constants.
F_TICK, F_SERIES, F_LABEL = int(FS_TICK * 1.4), int(FS_SERIES * 1.4), int(FS_LABEL * 1.25)
# Legend entries are ALL CAPS and the dataset names are long, so both sit a step
# below the axis type rather than overflowing their panels.
F_LEG, F_XTICK = int(FS_TICK * 1.15), int(FS_TICK * 1.15)

OOD_ORDER = ["kmnist", "notmnist", "fmnist", "svhn"]   # near-OOD first: the honest cases
PRETTY = {"kmnist": "KMNIST", "notmnist": "notMNIST", "fmnist": "Fashion", "svhn": "SVHN"}
ALPHA = 0.025
STAT = "entropy"


def load(run):
    """-> auroc[method][ood], rej[method][ood], idcost[method][ood].

    The baselines are read on their TWO-SIDED numbers, matching the two-sided AUROC the
    left panel already reported: mean intensity runs high on dense OOD images and low on
    sparse ones, so a one-sided threshold would score it on the wrong tail and understate
    a baseline the whole figure exists to take seriously.
    """
    with open(os.path.join(run, "ood.json")) as f:
        ood = json.load(f)
    with open(os.path.join(run, "input_density.json")) as f:
        dens = json.load(f)
    auroc, rej, idcost = {}, {}, {}
    a = {}
    rows = next(r["rows"] for r in ood if r["readout"] == "learned_readout")
    for r in rows:
        if r["statistic"] != STAT:
            continue
        a[r["ood"]] = r
    auroc["net"] = {k: v["auroc_2sided"] for k, v in a.items()}
    rej["net"] = {k: v["at_alpha"][str(ALPHA)]["ood_rejected"] for k, v in a.items()}
    idcost["net"] = {k: v["at_alpha"][str(ALPHA)]["id_rejected"] for k, v in a.items()}
    for name, key in (("int", "mean_intensity"), ("res", "pca_residual")):
        src = dens["pixels"][key]
        auroc[name] = {k: v["auroc_2sided"] for k, v in src.items()}
        rej[name] = {k: v["at_alpha"][str(ALPHA)]["ood_rejected_2s"] for k, v in src.items()}
        idcost[name] = {k: v["at_alpha"][str(ALPHA)]["id_rejected_2s"] for k, v in src.items()}
    return auroc, rej, idcost


METHODS = (("net", "NETWORK CONFIDENCE"),
           ("int", "MEAN INTENSITY"),
           ("res", "PIXEL PCA RESIDUAL"))
HATCH = "////"


def figure(auroc, rej, idcost, out, theme="light", palette="project", panels="both"):
    T, cols = THEMES[theme], PALETTES[palette]
    cmap = dict(zip((m for m, _ in METHODS), cols[:3]))
    x = np.arange(len(OOD_ORDER))

    ncol = 2 if panels == "both" else 1
    fig, axs = plt.subplots(1, ncol, squeeze=False,
                            figsize=(19.5 if ncol == 2 else 11.5, 7.0),
                            facecolor=T["surface"],
                            gridspec_kw=dict(width_ratios=[1.05, 1.30][:ncol]))
    axs = list(axs[0])
    axL = axs[0] if panels == "both" else None
    axR = axs[-1]

    if axL is not None:
        w = 0.26
        for off, (m, name) in enumerate(METHODS):
            axL.bar(x + (off - 1) * w, [auroc[m][k] for k in OOD_ORDER], width=w * 0.88,
                    color=cmap[m], alpha=0.80, edgecolor=cmap[m], lw=1.2, zorder=3,
                    label=name)
        # 0.5 is the no-information floor for AUROC; without it a 0.69 bar looks like a result
        axL.axhline(0.5, color=T["ref"], lw=1.2, ls=(0, (4, 4)), zorder=1)
        # In axes fraction on x so it stays inside the frame; the data-coordinate version
        # sat past the last group and was drawn outside the panel entirely.
        axL.text(0.995, 0.5, "chance", color=T["muted"], fontsize=F_TICK,
                 va="bottom", ha="right", transform=axL.get_yaxis_transform())
        axL.set_xticks(x)
        axL.set_xticklabels([PRETTY[d] for d in OOD_ORDER], fontsize=F_XTICK)
        axL.set_ylim(0.0, 1.05)
        axL.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        style_axes(axL, T, "", "AUROC, in-distribution vs OOD", fs_label=F_LABEL,
                   fs_tick=F_TICK)

    # ---- the operating point, all three detectors, both error directions ----------
    # Six bars per dataset: each detector contributes the OOD it rejects (solid) beside
    # the in-distribution input it wrongly rejects at the same threshold (hatched). The
    # pair has to sit side by side rather than in two panels, because a rejection rate
    # means nothing without the cost that bought it.
    bw = 0.135
    for off, (m, name) in enumerate(METHODS):
        base = (off - 1) * 2 * bw
        axR.bar(x + base - bw * 0.52, [rej[m][k] for k in OOD_ORDER], width=bw * 0.95,
                color=cmap[m], alpha=0.80, edgecolor=cmap[m], lw=1.2, zorder=3)
        axR.bar(x + base + bw * 0.52, [idcost[m][k] for k in OOD_ORDER], width=bw * 0.95,
                facecolor="none", edgecolor=cmap[m], lw=1.2, hatch=HATCH, zorder=3)
    axR.set_xticks(x)
    axR.set_xticklabels([PRETTY[d] for d in OOD_ORDER], fontsize=F_XTICK)
    axR.set_ylim(0.0, 1.05)
    style_axes(axR, T, "", "input rejected", fs_label=F_LABEL, fs_tick=F_TICK)
    axR.yaxis.set_major_formatter(plt.FuncFormatter(pct))

    # Two legends: hue carries the detector, fill carries which kind of rejection. Both
    # are needed, and stacking them keeps each row short enough to read.
    hue = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap[m], alpha=0.80, edgecolor=cmap[m])
           for m, _ in METHODS]
    fill = [plt.Rectangle((0, 0), 1, 1, facecolor=T["muted"], alpha=0.55,
                          edgecolor=T["muted"]),
            plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=T["muted"],
                          hatch=HATCH)]
    kw = dict(frameon=False, fontsize=F_LEG, handlelength=1.2, handleheight=1.2,
              columnspacing=2.2, handletextpad=0.6, loc="upper center")
    fig.legend(hue, [n for _, n in METHODS], ncol=3,
               bbox_to_anchor=(0.5, 1.010), **kw)
    fig.legend(fill, ["OOD REJECTED", "IN-DISTRIBUTION REJECTED"], ncol=2,
               bbox_to_anchor=(0.5, 0.955), **kw)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return out + ".png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/ood_mnist/run1/mnist_ood_s0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--theme", default="light", choices=tuple(THEMES))
    ap.add_argument("--panels", default="both", choices=("both", "rates"),
                    help="'both' keeps the threshold-free AUROC panel beside the "
                         "operating point; 'rates' drops it and shows only the "
                         "rejection rates")
    a = ap.parse_args()
    auroc, rej, idcost = load(a.run)
    hdr = "".join(f"{n:>21}" for _, n in METHODS)
    print(f"{'ood':<9}{hdr}")
    print(f"{'':<9}" + "".join(f"{'auroc':>7}{'ood rej':>7}{'id rej':>7}" for _ in METHODS))
    for k in OOD_ORDER:
        row = "".join(f"{auroc[m][k]:>7.3f}{rej[m][k]:>7.3f}{idcost[m][k]:>7.3f}"
                      for m, _ in METHODS)
        print(f"{k:<9}{row}")
    out = a.out or os.path.join("results", "figures",
                                "ood_rejection" + ("" if a.panels == "both" else "_rates"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("\n[plot]", figure(auroc, rej, idcost, out, a.theme, panels=a.panels))


if __name__ == "__main__":
    main()
