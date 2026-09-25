"""
Risk-coverage across all five datasets: does abstention buy anything, and where?

Plots RISK (1 - selective accuracy) on a LOG axis against coverage, one curve per dataset.
Both choices are load-bearing:

  risk, not accuracy   Selective accuracy compresses the interesting region against the
                       ceiling. MNIST moving 0.946 -> 0.999 reads as a small rise, when it
                       is a 44-fold reduction in errors. Risk shows the quantity that
                       actually changes.
  log scale            Base rates differ by an order of magnitude (0.054 on MNIST against
                       0.766 on SVHN), so a linear axis is dominated by SVHN's offset and
                       every other curve is squashed. On a log axis the SLOPE is the
                       relative error reduction, which is comparable across datasets even
                       though their intercepts are not.

Each curve is anchored by a marker at coverage 1, where selective accuracy equals base
accuracy by definition; without it the curves appear to start at arbitrary heights.

Data source: the 10-point conformal sweep each phase-2 run already stores in results.json
(alpha in 0..0.5), five seeds per dataset. Full-resolution curves would need the dumped
features, which exist locally only for MNIST and the frozen cells -- so the curves stop at
~48% coverage and cannot show the extreme-abstention tail. Stated in the caption, not
hidden by extrapolation.

Palette: reference categorical slots 1-5, validated (worst adjacent CVD dE 9.1, normal
dE 19.6, both above the gates). Three of the five fall below 3:1 contrast on white, which
obliges visible labels; every curve is directly labelled, so identity never rests on hue.

Usage:
    python experiments/RF_article/interp/plot_risk_coverage_datasets.py
    python experiments/RF_article/interp/plot_risk_coverage_datasets.py --readouts both
"""
import argparse, collections, glob, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PHASE2 = os.path.join(REPO, "results", "interp", "json", "experiments", "RF_article",
                      "interp", "mnist_family_sweep", "results", "run_20260819_112250")
DS = ["mnist", "notmnist", "kmnist", "fmnist"]      # by how much abstention buys
PRETTY = {"mnist": "MNIST", "fmnist": "Fashion", "kmnist": "KMNIST",
          "notmnist": "notMNIST", "svhn": "SVHN"}
# The project's three-colour palette, paired by tier, with line style carrying the split
# inside each pair. Five distinct hues read louder than every other figure in the paper,
# but shades of these three do not survive validation: the muted palette is low-chroma by
# design, so its tints fall below the chroma floor and same-hue pairs land at normal-vision
# dE ~10-12, under the 15 floor. (The three base colours already fail on their own -- sage
# against slate is dE 11.5 -- which is why plot_risk_coverage.py leans on direct labels.)
# Hue + dash + direct label carries identity without inventing two more weak hues.
# Greyscale for print (supervisor feedback, 09-17): identity is carried by line style and
# anchor marker, and a legend replaces the direct labels.
COLOR = {"mnist": "#000000", "notmnist": "#000000", "kmnist": "#000000",
         "fmnist": "#000000", "svhn": "#000000"}
DASH = {"mnist": "solid", "notmnist": (0, (6, 2.5)), "kmnist": (0, (1, 1.8)),
        "fmnist": (0, (7, 2, 1.5, 2)), "svhn": "solid"}
MARK = {"mnist": "o", "notmnist": "s", "kmnist": "^", "fmnist": "v", "svhn": "o"}
INK, MUTED, GRID, AXIS = "#1a1a1a", "#666666", "#e6e6e6", "#444444"
# 1.5x the original 13/11/11: this runs at column width in print, where the
# earlier sizes were legible on screen but not on the page.
FS_LABEL, FS_TICK, FS_SERIES = 20, 17, 17
READOUTS = {"learned_readout": "learned readout", "linear_probe": "linear probe"}
YLAB = {"accuracy": "selective accuracy", "risk": "risk"}


def collect(readout, statistic="entropy"):
    """{dataset: (coverage[n_alpha], risk[n_seed, n_alpha], base_risk[n_seed])}"""
    out = {}
    per = collections.defaultdict(list)
    base = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(PHASE2, "*_oriented_s*", "results.json"))):
        d = json.load(open(f))
        ds = d["config"]["dataset"]
        for u in d.get("uncertainty", []):
            if u["readout"] != readout:
                continue
            base[ds].append(1.0 - u["base_acc"])
            for s in u["statistics"]:
                if s["statistic"] == statistic:
                    per[ds].append(s["conformal"])
    for ds, runs in per.items():
        cov = np.array([r["coverage"] for r in runs[0]])
        risk = np.array([[1.0 - r["sel_acc"] for r in run] for run in runs])
        n_kept = np.array([[r["n_kept"] for r in run] for run in runs], dtype=float)
        # A run with zero errors among n_kept items has measured risk 0, which a log axis
        # cannot show. Floor it at 1/n_kept -- the finest risk that sample could resolve --
        # rather than dropping the point or letting it vanish off the bottom.
        risk = np.maximum(risk, 1.0 / np.maximum(n_kept, 1.0))
        out[ds] = (cov, risk, np.array(base[ds]))
    return out


X_MIN, X_MAX = 0.30, 1.035
X_PAD = 0.055           # keep labels this far inside the axes on both sides
# Room to the right of the coverage-1 anchors for their printed values. Set in main()
# only when those labels are switched on, so the plot area is not given away otherwise.
X_MAX_LABELLED = 1.125
FS_ANCHOR = 13          # the anchor values are read one at a time; they can sit below
                        # the series labels without becoming illegible
TARGET_MARK = "D"       # one shape for every target crossing, so it reads as a class
# Minimum label separation, expressed in the units of whichever y-transform is active.
# A label is roughly 0.13 decades tall on the log axis and 0.035 accuracy units tall on
# the linear one at this font size; both leave headroom.
MIN_SEP = {"accuracy": 0.055, "risk": 0.22}


def place_labels(curves, tf):
    """Choose a point on each curve where that curve is most isolated from the others.

    Anchoring every label at the left end puts them where the curves are BUNCHED, which is
    why they kept colliding however far apart they were nudged: the crowding is a property
    of the position, not of the offsets. Instead, score each candidate x by the vertical
    distance from the nearest other curve at that x, and label where that distance is
    greatest. Each label then sits on its own line, in clear space.

    `tf` maps data y to axis-space y, so the isolation is measured in the space the reader
    actually sees -- decades on the log-risk axis, accuracy units on the linear one.

    Returns [(x, y, key, side)] with side = +1 to write above the curve, -1 below, chosen
    to face away from the nearest neighbour.
    """
    keys = list(curves)
    out = []
    for k in keys:
        cx, cy = curves[k]
        ok = (cx >= X_MIN + X_PAD) & (cx <= X_MAX - X_PAD)
        if not ok.any():
            ok = np.ones_like(cx, dtype=bool)
        cand_x, cand_y = cx[ok], cy[ok]
        gap = np.full(len(cand_x), np.inf)
        nearest_above = np.zeros(len(cand_x), dtype=bool)
        for j in keys:
            if j == k:
                continue
            jx, jy = curves[j]
            inside = (cand_x >= jx.min()) & (cand_x <= jx.max())
            if not inside.any():
                continue
            oy = np.interp(cand_x, jx, tf(jy))
            d = np.abs(tf(cand_y) - oy)
            d = np.where(inside, d, np.inf)
            closer = d < gap
            nearest_above = np.where(closer, oy > tf(cand_y), nearest_above)
            gap = np.minimum(gap, d)
        i = int(np.argmax(gap))
        # face away from whichever neighbour is closest
        out.append((cand_x[i], cand_y[i], k, -1 if nearest_above[i] else +1))
    return out


def crossing(cov, acc, target):
    """Largest coverage at which the median selective accuracy still reaches `target`.

    This is `cov_at_acc` read off the plotted curve rather than off results.json, so the
    marker necessarily lands ON the line. Taking it from the JSON instead would put the
    marker at a coverage the drawn (median-over-seeds) curve does not actually pass
    through, and the discrepancy would be visible.

    Returns None when the curve never reaches the target within the sweep, and None again
    when it never drops below it -- in that case the crossing coincides with the coverage-1
    anchor and a second marker there says nothing.
    """
    if acc.max() < target or acc[-1] >= target:
        return None
    i = int(np.flatnonzero(acc >= target).max())
    a0, a1 = acc[i], acc[i + 1]
    t = 0.0 if a0 == a1 else (a0 - target) / (a0 - a1)
    return cov[i] + t * (cov[i + 1] - cov[i])


def compute_bands(datasets, pad=0.02):
    """Find the break by locating the widest empty stripe in the data.

    Hardcoding the bands does not survive a second panel: the linear probe scores SVHN
    about 14 points higher than the learned readout does, so a band sized for one readout
    lets the curve overflow in the other and pushes its label off the axes. Bands derived
    from the union of everything being plotted keep both panels on the same scale AND
    contain every curve.

    Returns (hi_band, lo_band), or None when the largest gap is too small to be worth a
    break -- in which case a single continuous axis is the honest choice.
    """
    vals = np.concatenate([np.ravel(1.0 - risk) for d in datasets for _, risk, _ in d.values()]
                          + [np.ravel(1.0 - base) for d in datasets for _, _, base in d.values()])
    v = np.sort(vals)
    gaps = np.diff(v)
    i = int(np.argmax(gaps))
    if gaps[i] < 0.15:                      # nothing worth breaking for
        return None
    return ((v[i + 1] - pad, min(v[-1] + pad, 1.005)),
            (max(v[0] - pad, 0.0), v[i] + pad))


def panel(axes, data, title=None, yaxis="accuracy", bands=None, target=0.95,
          anchor_labels=True):
    """Draw the curves on one axis (log risk) or a broken pair (linear accuracy).

    With a broken pair every curve is drawn on BOTH axes and each shows only its own band,
    which is what keeps a curve crossing the break continuous. Height ratios follow the band
    spans, so a given difference in accuracy occupies the same vertical distance above and
    below the break rather than SVHN's spread being silently magnified.
    """
    log = yaxis == "risk"
    tf = (lambda v: np.log10(v)) if log else (lambda v: np.asarray(v, dtype=float))
    curves, anchors = {}, []
    for ds in DS:
        if ds not in data:
            continue
        cov, risk, base = data[ds]
        # risk and selective accuracy are the same measurement read two ways; only the
        # presentation differs, so the curves are flipped here rather than recomputed
        y = risk if log else 1.0 - risk
        b = np.median(base) if log else 1.0 - np.median(base)
        o = np.argsort(cov)
        c, med = cov[o], np.median(y, axis=0)[o]
        lo, hi = y.min(axis=0)[o], y.max(axis=0)[o]
        for ax in axes:
            ax.fill_between(c, lo, hi, color="#000000", alpha=0.07, lw=0, zorder=2)
            ax.plot(c, med, color=COLOR[ds], lw=2.2, zorder=3, linestyle=DASH[ds],
                    marker=MARK[ds], markevery=[len(c) - 1], ms=8, mec="black",
                    mfc="white", mew=1.4, label=PRETTY[ds])
            # anchor: at coverage 1 selective accuracy IS overall accuracy
            ax.plot([1.0], [b], marker=MARK[ds], ms=8,
                    color="white", mec="black", mew=1.4, zorder=4)
        curves[ds] = (c, med)
        anchors.append((b, ds))

        # where this dataset still meets the target accuracy. A crossing that lands within
        # a hair of full coverage is dropped: it would be drawn on top of the anchor and
        # read as one smudged marker, and the anchor's printed value already says the
        # curve sits just under the line there.
        acc = 1.0 - med if log else med
        xc = crossing(c, acc, target) if target else None
        if xc is not None and xc < 0.988:
            yc = (1.0 - target) if log else target
            for ax in axes:
                ax.plot([xc], [yc], marker=TARGET_MARK, ms=9, color=COLOR[ds],
                        mec="white", mew=1.5, zorder=5)

    for i, ax in enumerate(axes):
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(*(bands[i] if bands else (0.66, 1.02)))  # SVHN dropped: all curves sit above 0.69
        ax.set_xlim(X_MIN, X_MAX)
        # Coverage cannot exceed 1: the axis line and ticks stop there, and the space to
        # the right only holds the printed full-coverage accuracies.
        ax.set_xticks(np.round(np.arange(X_MIN, 1.0001, 0.1), 2))
        ax.spines["bottom"].set_bounds(X_MIN, 1.0)
        ax.grid(False, axis="x")
        ax.grid(True, which="major", color=GRID, lw=0.7)
        ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK, labelsize=FS_TICK)

    # Text is drawn only AFTER the limits are final, and only on the axis whose band
    # contains it. Matplotlib does not clip text by default, so a label written to both
    # axes of the broken pair escapes the one that cannot show it and drags the saved
    # bounding box out with it.
    def host_for(y):
        for ax in axes:
            b0, b1 = ax.get_ylim()
            if b0 <= y <= b1:
                return ax, (b0, b1)
        return axes[0], axes[0].get_ylim()

    sep = MIN_SEP[yaxis]
    step = (lambda v, s: v * 10.0 ** (s * 0.5 * sep)) if log else \
           (lambda v, s: v + s * 0.5 * sep)
    for x, y, ds, side in ([] if True else place_labels(curves, tf)):  # legend instead
        host, (b0, b1) = host_for(y)
        y_lab = step(y, side)
        # 1.6 rather than 1.0: the anchor point clearing the band is not enough, the text
        # grows away from it and would otherwise be drawn outside the frame entirely.
        if not (b0 <= step(y, side * 1.6) <= b1):    # write on the other side instead
            side = -side
            y_lab = step(y, side)
        # White halo so the label stays legible over its own band or a grid line.
        host.text(x, y_lab, PRETTY[ds],
                  color=COLOR[ds], fontsize=FS_SERIES, ha="center",
                  va="bottom" if side > 0 else "top", zorder=6,
                  path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

    if anchor_labels:
        for b, ds in anchors:
            # Printed beside the coverage-1 anchor because the comparison hinges on where
            # each curve starts, and five curves entering a shared axis at five different
            # heights cannot be read off the ticks to three decimals.
            host_for(b)[0].text(1.012, b, f"{b:.3f}", color=COLOR[ds],
                                fontsize=FS_ANCHOR, ha="left", va="center", zorder=6,
                                path_effects=[pe.withStroke(linewidth=3.0,
                                                            foreground="white")])

    if target:
        # The reference line is what makes the diamonds self-explanatory: each one is
        # simply where a curve crosses it, so the figure needs no legend to say so.
        yc = (1.0 - target) if log else target
        for ax in axes:
            b0, b1 = ax.get_ylim()
            if not (b0 <= yc <= b1):
                continue
            ax.axhline(yc, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
            ax.text(X_MIN + 0.008, yc, f"{target:.0%} accuracy", color=MUTED,
                    fontsize=FS_ANCHOR, ha="left", va="bottom", zorder=6)
            break

    axes[0].legend(loc="lower left", bbox_to_anchor=(0.04, 0.05), fontsize=FS_ANCHOR + 1, frameon=True,
                   edgecolor=AXIS, fancybox=False, handlelength=3.2)
    axes[-1].set_xlabel("coverage (fraction answered)", fontsize=FS_LABEL, color=INK)
    if title:
        axes[0].set_title(title, fontsize=FS_LABEL, color=MUTED, loc="left", pad=8)

    if len(axes) == 2:
        # mark the break explicitly: a reader who misses it reads SVHN as competitive
        axes[0].spines["bottom"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[0].tick_params(bottom=False, labelbottom=False)
        kw = dict(marker=[(-1, -0.9), (1, 0.9)], markersize=11, linestyle="none",
                  color=AXIS, mec=AXIS, mew=1.3, clip_on=False)
        axes[0].plot([0], [0], transform=axes[0].transAxes, **kw)
        axes[1].plot([0], [1], transform=axes[1].transAxes, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", default="learned", choices=("learned", "both"))
    ap.add_argument("--statistic", default="entropy")
    ap.add_argument("--yaxis", default="accuracy", choices=("accuracy", "risk"),
                    help="selective accuracy on a linear axis (default, the more directly "
                         "interpretable quantity) or risk on a log axis, which shows the "
                         "relative error reduction but compresses the high-accuracy end")
    ap.add_argument("--no-break", action="store_true",
                    help="single continuous accuracy axis instead of the broken pair")
    ap.add_argument("--target", type=float, default=0.0,
                    help="mark where each curve still meets this selective accuracy; "
                         "0 to disable")
    ap.add_argument("--no-anchor-labels", action="store_true",
                    help="omit the printed accuracy beside each coverage-1 anchor")
    ap.add_argument("--out", default=None)
    ap.set_defaults(no_break=True)   # the break only existed for SVHN, now dropped
    a = ap.parse_args()

    global X_MAX
    if not a.no_anchor_labels:
        X_MAX = X_MAX_LABELLED

    keys = list(READOUTS) if a.readouts == "both" else ["learned_readout"]
    datasets = [collect(k, a.statistic) for k in keys]

    # The log-risk view needs no break -- a log axis already handles the SVHN offset.
    # Bands come from the union of BOTH panels' data so the two share one scale.
    bands = None
    if a.yaxis == "accuracy" and not a.no_break:
        bands = compute_bands(datasets)
    broken = bands is not None
    nrow = 2 if broken else 1
    ncol = len(keys)
    hr = [bands[0][1] - bands[0][0], bands[1][1] - bands[1][0]] if broken else [1]
    fig, ax = plt.subplots(nrow, ncol, squeeze=False, sharex="col",
                           figsize=(8.4 * ncol + (0.4 if ncol > 1 else 0), 6.2),
                           gridspec_kw=dict(height_ratios=hr, hspace=0.07))

    for j, (key, data) in enumerate(zip(keys, datasets)):
        panel([ax[i][j] for i in range(nrow)], data,
              title=(READOUTS[key] if a.readouts == "both" else None),
              yaxis=a.yaxis, bands=bands, target=a.target,
              anchor_labels=not a.no_anchor_labels)
    # one y label for the whole column stack, centred across the break
    fig.supylabel(YLAB[a.yaxis], fontsize=FS_LABEL, color=INK, x=0.005)
    suffix = f"{a.yaxis}" + ("_both" if a.readouts == "both" else "")
    out = a.out or os.path.join(REPO, "results", "figures", f"coverage_datasets_{suffix}")

    # NOT tight_layout: it cannot handle the broken-axis pair and would override the
    # hspace that sets the break gap.
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    d = collect("learned_readout", a.statistic)
    print(f"{'dataset':<9}{'base acc':>9}{'base risk':>10}{'risk @48% cov':>15}"
          f"{'reduction':>10}{f'cov@{a.target:.0%}':>10}")
    for ds in DS:
        if ds not in d:
            continue
        cov, risk, base = d[ds]
        i = int(np.argmin(cov))
        b, r = np.median(base), np.median(risk, axis=0)[i]
        o = np.argsort(cov)
        xc = crossing(cov[o], np.median(1.0 - risk, axis=0)[o], a.target) if a.target else None
        print(f"{ds:<9}{1 - b:>9.4f}{b:>10.4f}{r:>15.4f}{b / max(r, 1e-9):>9.1f}x"
              f"{('%.3f' % xc) if xc is not None else '--':>10}")
    print(f"\n[plot] wrote {out}.png (+ .pdf)")


if __name__ == "__main__":
    main()
