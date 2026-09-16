"""Appendix figure: how much of the oriented prior survives, batch by batch.

Coherence is reported as a PERCENTAGE of its value at initialisation, not in raw
units, because the raw ceiling is set by the RF aspect ratio rather than by 1.0: a
pristine elongated Gaussian at gamma = 0.4 measures 0.687, and the population
initialisation is 0.710. Raw numbers invite the reader to treat 1.0 as intact.

THE TRAP this figure has to show rather than hide: the first logged checkpoint
already contains plasticity. Trace-STDP is at 71% of the initialisation at batch 0
and triplet-STDP at 100%, so a third of trace's total loss happens before anything
is recorded. The initialisation is anchored as a separate point, the interval
before the first batch is shaded, and the segment across it is dashed.

Random connectivity is deliberately absent: it never carried the oriented prior, so
"percent retained" is undefined for it.

  python experiments/RF_article/interp/plot_coherence_trajectory.py --out-dir results/figures
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_risk_coverage import PALETTES  # noqa: E402
ROSE, SLATE, SAGE = PALETTES["project"]
INK, MUTED, GRID = "#1a1a1a", "#666666", "#e6e6e6"

# notMNIST runs a shorter schedule (40 batches against 175), so including it would
# change the composition of the mean part-way along the x axis. It is excluded here
# and the caption says so; the retention percentages quoted in the text use all five.
DS = ["mnist", "fmnist", "kmnist", "svhn"]
F_LABEL, F_TICK, F_SERIES = 13, 11, 12


def load(root, tag):
    """(batches, mean, sd) of orientation coherence over datasets x seeds."""
    runs, batches = [], None
    for d in DS:
        for f in sorted(glob.glob(f"{root}/{d}_{tag}_s*/results.json")):
            try:
                t = json.load(open(f))["trajectory"]
            except Exception:
                continue
            c = [r.get("orient_coh") for r in t]
            if not c or any(v is None for v in c):
                continue
            runs.append(c)
            batches = [r["batch"] for r in t]
    if not runs:
        return None, None, None
    n = min(len(r) for r in runs)
    A = np.array([r[:n] for r in runs])
    return np.array(batches[:n]), A.mean(0), A.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default=None)
    ap.add_argument("--phase2", default=None)
    ap.add_argument("--out-dir", default="results/figures")
    a = ap.parse_args()
    base = "results/interp/json/experiments/RF_article/interp"
    p1 = a.phase1 or f"{base}/phase1_ablation/results/run_20260819_112254"
    p2 = a.phase2 or sorted(glob.glob(f"{base}/mnist_family_sweep/results/*"))[0]

    init = load(p1, "frozen")[1][0]        # frozen cell: the only no-plasticity t=0
    series = [("trace-STDP", load(p1, "base_ori"), SLATE),
              ("R-STDP", load(p2, "oriented"), ROSE),
              ("triplet-STDP", load(p1, "triplet"), SAGE)]
    series = [(n, b, m, s, c) for n, (b, m, s), c in series if m is not None]

    print(f"initialisation (frozen) = {init:.3f}")
    for n, b, m, s, c in series:
        print(f"  {n:13s} batch {b[0]:3d}: {100*m[0]/init:5.1f}%   "
              f"batch {b[-1]:3d}: {100*m[-1]/init:5.1f}%")

    xmax = max(b[-1] for _, b, _, _, _ in series)
    pre = -0.055 * xmax                    # where the initialisation anchor sits

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvspan(pre, 0, color=INK, alpha=.05, lw=0, zorder=0)
    ax.axhline(100, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)

    for name, b, m, sd, col in series:
        y, e = 100 * m / init, 100 * sd / init
        ax.plot([pre, b[0]], [100, y[0]], color=col, lw=1.6, ls=(0, (2, 2)), zorder=3)
        ax.fill_between(b, y - e, y + e, color=col, alpha=.18, lw=0, zorder=2)
        ax.plot(b, y, color=col, lw=2.2, zorder=4)
        ax.annotate(f"{name}  {y[-1]:.0f}%", (b[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=F_SERIES, color=col,
                    va="center")

    ax.scatter([pre], [100], s=52, color=INK, zorder=6)
    ax.set_xlim(pre * 1.9, xmax * 1.30)
    ax.set_ylim(45, 108)
    ax.set_xlabel("batch (1000 images)", fontsize=F_LABEL)
    ax.set_ylabel("orientation coherence\n(\\% of initialisation)".replace("\\", ""),
                  fontsize=F_LABEL)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.yaxis.grid(True, color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=F_TICK)

    os.makedirs(a.out_dir, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(a.out_dir, f"coherence_trajectory.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print("saved ->", p)


if __name__ == "__main__":
    main()
