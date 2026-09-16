"""
Main-text figure: how much of the oriented prior each learning rule leaves standing.

One number carries the claim -- orientation coherence, as a fraction of the value at
initialisation. Everything else the interp harness logs about RF geometry (var_x,
var_y, cov_xy, anisotropy) is a view of the same second-moment tensor and belongs in
the appendix; see plot_rf_geometry.py.

Init is taken from the phase-1 `frozen` condition, the only cell with no plasticity.
Phase 2 has no frozen counterpart, so its row is drawn against the same init -- run a
no-plasticity phase-2 cell to remove that assumption.

  python experiments/RF_article/interp/plot_coherence_retention.py \
      --phase1 <run_dir> --phase2 <run_dir> --out-dir <dir>
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_risk_coverage import PALETTES  # noqa: E402

ROSE, SLATE, SAGE = PALETTES["project"]

DATASETS = ["mnist", "fmnist", "kmnist", "notmnist", "svhn"]
INK, MUTED, GRID, INIT = "#0b0b0b", "#52514e", "#d8d7d2", "#9a9892"
# rows ordered by how much prior survives, so the figure reads top-to-bottom.
# `base_rnd` is deliberately NOT a row: random weights never carried the prior, so a
# "retained" fraction is undefined for them. They appear as the floor band instead.
ROWS = [("p1", "base_ori", "oriented RF", "trace-STDP",   SLATE),
        ("p2", "oriented", "oriented RF", "R-STDP",       ROSE),
        ("p1", "triplet",  "oriented RF", "triplet-STDP", SAGE)]
# The random floor is context, not a result: it is the only band without a retained
# fraction, so it is drawn in neutral grey and never competes with the three rules.
FLOOR = "#8d8b86"


def load(run_dir):
    cells = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "*"))):
        rp = os.path.join(d, "results.json")
        if not os.path.isfile(rp):
            continue
        r = json.load(open(rp))
        cfg = r["config"]; ds = cfg["dataset"]
        cond = cfg["tag"][len(ds) + 1:].rsplit("_s", 1)[0]
        cells.setdefault((ds, cond), []).append(r)
    return cells


def final_coh(cells, ds, cond):
    return np.array([r["trajectory"][-1]["orient_coh"] for r in cells[(ds, cond)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    c1, c2 = load(args.phase1), load(args.phase2)
    init = float(np.mean([r["trajectory"][0]["orient_coh"]
                          for ds in DATASETS for r in c1[(ds, "frozen")]]))

    # the random-weight floor: what coherence measures when there is no prior at all
    rnd = np.array([final_coh(c1, ds, "base_rnd").mean() for ds in DATASETS])

    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.axvspan(rnd.min(), rnd.max(), color=FLOOR, alpha=0.13, lw=0, zorder=0)
    ax.text(rnd.max(), -0.52, "  random weights", color=FLOOR, fontsize=8,
            va="center", ha="left")
    ax.axvline(init, color=INIT, lw=1.2, ls="--", zorder=1)
    ax.text(init, len(ROWS) - 0.38, "  initialisation", color=INIT, fontsize=8,
            va="center", ha="left")

    for y, (phase, cond, prior, rule, color) in enumerate(reversed(ROWS)):
        cells = c1 if phase == "p1" else c2
        per_ds = np.array([final_coh(cells, ds, cond).mean() for ds in DATASETS])
        mean = per_ds.mean()
        ax.plot([init, mean], [y, y], color=color, lw=1.6, alpha=0.35, zorder=2,
                solid_capstyle="butt")
        ax.scatter(per_ds, np.full(len(per_ds), y), s=16, facecolor="white",
                   edgecolor=color, lw=1.2, zorder=3)
        ax.scatter([mean], [y], s=74, color=color, zorder=4)
        ax.text(mean, y + 0.30, f"{100 * mean / init:.0f}% retained", color=color,
                fontsize=8.5, ha="center", va="bottom", fontweight="semibold")

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{p}\n{r}" for _, _, p, r, _ in reversed(ROWS)],
                       fontsize=9, color=INK, linespacing=1.5)
    ax.set_ylim(-0.75, len(ROWS) - 0.4)
    ax.set_xlim(0, max(init * 1.12, 0.78))
    ax.set_xlabel("orientation coherence after 3 epochs", fontsize=9.5, color=INK)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(args.out_dir, f"coherence_retention.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved -> {p}")
    plt.close(fig)

    print(f"\ninit coherence = {init:.3f}  (phase-1 frozen, pooled over datasets)")
    print(f"  random-weight floor {rnd.mean():.3f}  (per-dataset {rnd.min():.3f}-{rnd.max():.3f})")
    for phase, cond, prior, rule, _ in ROWS:
        cells = c1 if phase == "p1" else c2
        per_ds = {ds: final_coh(cells, ds, cond) for ds in DATASETS}
        lo = min(v.mean() for v in per_ds.values()); hi = max(v.mean() for v in per_ds.values())
        m = np.mean([v.mean() for v in per_ds.values()])
        print(f"  {prior:14s} {rule:13s} {m:.3f}  ({100*m/init:.0f}% retained; "
              f"per-dataset {100*lo/init:.0f}-{100*hi/init:.0f}%)")


if __name__ == "__main__":
    main()
