"""
Figures for the bootstrapped abstention threshold.

Two panels, answering the two questions a reviewer will ask about a quoted operating point:

  left   Where is tau, and how well is it pinned? The bootstrap distribution of tau_alpha
         over resampled calibration sets, with the point estimate and the exact
         order-statistic interval marked. If the spread is wide, every downstream number
         (coverage, selective accuracy, OOD rejection rate) inherits that width.
  right  What does the threshold buy? Selective accuracy against realized coverage at each
         alpha, with the double-bootstrap interval as a band -- NOT the Wilson band, which
         conditions on tau being exactly right and therefore understates the uncertainty of
         an operating point you actually had to calibrate.

Style follows the project convention set in plot_risk_coverage.py: the muted rose/slate/
sage palette, no titles, no legends, direct end labels on every series. The palette REQUIRES
direct labels -- rose and sage separate by only dE 3.5 under deuteranopia, so identity must
never rest on hue.

Usage:
    python experiments/RF_article/interp/plot_threshold.py --run <run_dir>
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_risk_coverage import (  # noqa: E402
    PALETTES, DEFAULT_PALETTE, THEMES, FS_LABEL, FS_TICK, FS_SERIES, LABELS, style_axes, pct,
)
from threshold import analyse, load_run, resplit_from_test, ALPHAS  # noqa: E402

READOUTS = ("learned_readout", "linear_probe", "pool")


def tau_distribution(conf_cal, alpha, B, seed):
    """The bootstrap draws themselves, for the histogram (bootstrap_tau returns summaries)."""
    conf_cal = np.asarray(conf_cal, dtype=float)
    n = len(conf_cal)
    rng = np.random.default_rng(seed)
    return np.quantile(conf_cal[rng.integers(0, n, size=(B, n))], alpha, axis=1)


def figure(reports, cal_scores, out, alpha=0.025, theme="light", palette=DEFAULT_PALETTE,
           B=2000, seed=0):
    T = THEMES[theme]
    cols = PALETTES[palette]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 6.6), facecolor=T["surface"])

    # ---- left: how well is tau pinned, IN CONTEXT --------------------------
    # A histogram of the bootstrap draws alone is unreadable here: tau is pinned to
    # ~0.1 nats out of a ~4 nat range, so the draws collapse to an invisible spike and
    # the eye reads "no uncertainty" without ever learning the scale. The honest display
    # is a ridgeline of the CALIBRATION distribution being quantiled, with tau and its
    # interval drawn on it -- the width of the interval then reads against the width of
    # the thing it is cutting.
    rows = [r for r in reports if r["readout"] in cal_scores]
    labels = []
    x_min = min(np.percentile(np.asarray(cal_scores[r["readout"]], float), 0.2)
                for r in rows)
    for i, r in enumerate(rows):
        ro = r["readout"]
        base = len(rows) - 1 - i                      # top row = first readout
        s = np.asarray(cal_scores[ro], dtype=float)
        lo_x, hi_x = np.percentile(s, [0.2, 100])
        xs = np.linspace(lo_x, hi_x, 400)
        h, edges = np.histogram(s, bins=90, range=(lo_x, hi_x), density=True)
        dens = np.interp(xs, 0.5 * (edges[:-1] + edges[1:]), h)
        dens = 0.80 * dens / (dens.max() or 1.0)
        axL.fill_between(xs, base, base + dens, color=cols[i], alpha=0.30, lw=0, zorder=2)
        axL.plot(xs, base + dens, color=cols[i], lw=1.6, zorder=3)

        t, e = r["tau"][alpha], r["tau_exact"][alpha]
        axL.vlines(t["tau"], base, base + 0.92, color=cols[i], lw=2.2, zorder=5)
        # exact order-statistic interval: the trustworthy width when the percentile
        # bootstrap under-covers (measured 0.895-0.930 coverage at n_cal=1000)
        axL.fill_betweenx([base, base + 0.92], e["lo"], e["hi"], color=cols[i],
                          alpha=0.55, lw=0, zorder=4)
        labels.append((base, ro, i))
        axL.text(t["tau"], base + 0.50,
                 f"  tau={t['tau']:.2f}   CI width {e['hi'] - e['lo']:.3f} ",
                 color=T["muted"], fontsize=FS_TICK - 3, va="center", ha="left", zorder=6)
    style_axes(axL, T, f"confidence, calibration set  (alpha = {alpha:g})", "")
    axL.set_yticks([])
    axL.spines["left"].set_visible(False)
    axL.grid(False)
    # Row labels hang off the LEFT edge of each ridge rather than sitting on it: the
    # pooled readout piles up against its own left tail, so a label anchored at tau
    # lands inside the curve it is naming.
    span = axL.get_xlim()[1] - x_min
    axL.set_xlim(x_min - 0.30 * span, axL.get_xlim()[1])
    for base, ro, i in labels:
        axL.text(axL.get_xlim()[0], base + 0.42, LABELS[ro] + " ", color=cols[i],
                 fontsize=FS_SERIES, va="center", ha="left")

    # ---- right: what it buys ----------------------------------------------
    for i, r in enumerate(reports):
        ops = r["operating"]
        a_sorted = sorted(ops)
        cov = np.array([ops[a]["coverage"] for a in a_sorted])
        acc = np.array([ops[a]["sel_acc"] for a in a_sorted])
        lo = np.array([ops[a]["sel_acc_lo"] for a in a_sorted])
        hi = np.array([ops[a]["sel_acc_hi"] for a in a_sorted])
        o = np.argsort(cov)
        axR.fill_between(cov[o], lo[o], hi[o], color=cols[i], alpha=0.20, lw=0)
        axR.plot(cov[o], acc[o], color=cols[i], lw=2.2, marker="o", ms=6, zorder=3)
        # label at the LEFT (low-coverage) end: coverage cannot exceed 100%, so there is
        # no room to hang labels off the right without inventing axis that cannot exist
        axR.text(cov[o][0] - 0.006, acc[o][0], LABELS[r["readout"]] + " ", color=cols[i],
                 fontsize=FS_SERIES, va="center", ha="right")
    axR.axhline(0.99, color=T["ref"], lw=1.2, ls=(0, (4, 4)), zorder=1)
    axR.text(1.0, 0.99, " 99%", color=T["muted"], fontsize=FS_TICK,
             va="bottom", ha="right")
    style_axes(axR, T, "coverage (fraction answered)", "selective accuracy")
    axR.xaxis.set_major_formatter(plt.FuncFormatter(pct))
    axR.yaxis.set_major_formatter(plt.FuncFormatter(pct))
    lo_cov = min(min(o["coverage"] for o in r["operating"].values()) for r in reports)
    axR.set_xlim(lo_cov - 0.16, 1.005)               # coverage is a fraction; 100% is the wall

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return f"{out}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--statistic", default="entropy")
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--split-test", type=float, default=0.5)
    ap.add_argument("-B", "--bootstrap", type=int, default=2000)
    ap.add_argument("--theme", default="light", choices=tuple(THEMES))
    ap.add_argument("--palette", default=DEFAULT_PALETTE, choices=tuple(PALETTES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = load_run(a.run)
    if a.split_test > 0:
        d = resplit_from_test(d, a.split_test, a.seed)

    # the calibration statistic per readout, for the histogram panel
    from threshold import readout_scores  # noqa: E402
    from uncertainty import uncertainty_stats  # noqa: E402
    reports, cal_scores = [], {}
    for ro in READOUTS:
        r = analyse(d, ro, a.statistic, B=a.bootstrap, seed=a.seed)
        if r is None:
            continue
        reports.append(r)
        pc, pt, Rc, Rt, pred, y = readout_scores(d, ro)
        cal_scores[ro] = uncertainty_stats(pc, Rc)[a.statistic]

    if not reports:
        print("[plot_threshold] no readouts available in this run")
        return
    out = a.out or os.path.join(a.run if os.path.isdir(a.run) else ".", "threshold")
    p = figure(reports, cal_scores, out, alpha=a.alpha, theme=a.theme,
               palette=a.palette, B=a.bootstrap, seed=a.seed)
    print(f"[plot_threshold] wrote {p} (+ .pdf)")


if __name__ == "__main__":
    main()
