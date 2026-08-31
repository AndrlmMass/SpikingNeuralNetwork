"""
Why the OOD decision is hard, shown as two distributions and the cut you have to make.

One point per test item -- every MNIST test image and every image of the probe dataset --
scored by the same predictive entropy the abstention threshold uses, smoothed into a
density and filled. The abstention thresholds sit on top as vertical lines, so the reader
can see directly what each alpha buys: everything to the RIGHT of a line is refused.

The axis is log entropy, not linear
-----------------------------------
Both distributions have an atom at H = 0: a confident readout puts essentially all mass on
one class, and thousands of MNIST items land there exactly. On a linear axis that atom is
a spike that swallows the plot and hides the only thing worth seeing, which is the overlap
in the middle. Entropy is positive and spans four orders of magnitude, so the density is
estimated on log10(H) -- the standard treatment for a positive, heavily skewed variable --
and the ticks are relabelled back into entropy units. Values below 1e-4 are indistinguish-
able from perfect confidence and are floored there rather than dropped.

The rug underneath is the raw scores, subsampled, because a smoothed density can suggest
structure the sample does not have.

Usage:
    python experiments/RF_article/interp/plot_confidence_overlap.py
    python experiments/RF_article/interp/plot_confidence_overlap.py --all
"""
import argparse, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ood import confidence_sets  # noqa: E402
from plot_risk_coverage import PALETTES, THEMES, FS_TICK, FS_LABEL  # noqa: E402

F_TICK, F_LABEL = int(FS_TICK * 1.35), int(FS_LABEL * 1.2)
F_ANN = int(FS_TICK * 1.05)
PRETTY = {"kmnist": "KMNIST", "notmnist": "notMNIST", "fmnist": "Fashion-MNIST",
          "svhn": "SVHN"}
OOD_ORDER = ["fmnist", "kmnist", "notmnist", "svhn"]
ALPHAS = (0.01, 0.025, 0.05, 0.10)
EPS = 1e-4                      # entropy floor: below this the readout is simply certain
TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
TICKLAB = ["$\\leq 10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"]


def load(run):
    d = np.load(os.path.join(run, "uncertainty_features.npz"))
    ck = np.load(os.path.join(run, "weights", "checkpoint.npz"), allow_pickle=True)
    sets, _ = confidence_sets(d, "learned_readout", ck)
    if sets is None:
        raise SystemExit(f"no learned-readout scores in {run}")
    # the statistic is stored as NEGATIVE entropy (higher = more certain); flip it back,
    # so that on this figure "further right" means "less sure" and the rejected region is
    # a right tail rather than a left one
    ent = lambda s: np.maximum(-np.asarray(s["entropy"], dtype=float), 0.0)
    return ent(sets["cal"]), ent(sets["id"]), {k: ent(v) for k, v in sets["ood"].items()}


def u(h):
    return np.log10(np.maximum(h, EPS))


def density(x, grid, bw=0.30):
    k = gaussian_kde(x, bw_method=bw)
    return k(grid)


def draw(ax, cal, idh, oodh, name, T, cols, annotate=True, rug=True, seed=0):
    c_id, c_ood = cols[0], cols[1]
    grid = np.linspace(u(EPS) - 0.15, np.log10(np.log(10)) + 0.15, 600)
    d_id, d_ood = density(u(idh), grid), density(u(oodh), grid)
    top = max(d_id.max(), d_ood.max())

    for dens, other, colour, lab in ((d_id, d_ood, c_id, "MNIST"),
                                     (d_ood, d_id, c_ood, PRETTY[name].upper())):
        ax.fill_between(grid, 0, dens, color=colour, alpha=0.32, lw=0, zorder=2)
        ax.plot(grid, dens, color=colour, lw=2.2, zorder=3)
        # Label where this curve stands clearest of the other one, not at its peak: the
        # two peaks sit at opposite ends but the OOD peak lands under the threshold lines.
        i = int(np.argmax(dens - other))
        ax.text(grid[i], dens[i] + 0.03 * top, lab, color=colour, fontsize=F_ANN,
                ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=3.5, foreground=T["surface"])])

    if rug:
        # one tick per test item, subsampled: the density is a model, this is the data
        rng = np.random.default_rng(seed)
        for vals, colour, row in ((idh, c_id, -0.055), (oodh, c_ood, -0.115)):
            s = rng.choice(u(vals), size=min(1200, len(vals)), replace=False)
            ax.plot(s, np.full(len(s), row * top) + rng.normal(0, 0.006 * top, len(s)),
                    ls="none", marker="|", ms=7, mew=0.7, color=colour, alpha=0.16,
                    zorder=1)

    # thresholds, calibrated on in-distribution data alone: tau is the alpha-quantile of
    # the ID confidence, so everything to its RIGHT (higher entropy) is refused
    for a in ALPHAS:
        h_tau = float(np.quantile(cal, 1.0 - a))     # cal is entropy; alpha lives up top
        x = u(h_tau)
        ax.axvline(x, color=T["ref"], lw=1.1, ls=(0, (4, 3)), zorder=4)
        if not annotate:
            continue
        rej = float((oodh > h_tau).mean())
        # Rotated onto the line itself. The four thresholds fall within half a decade of
        # each other, so horizontal labels collide however they are staggered; running
        # them along the lines is the only placement that fits without shrinking the type.
        ax.text(x, top * 0.30, rf"$\alpha$ {a:g}   {rej:.0%} rejected",
                color=T["muted"], fontsize=int(F_ANN * 0.92), ha="center", va="center",
                rotation=90, rotation_mode="anchor", zorder=6,
                path_effects=[pe.withStroke(linewidth=3.5, foreground=T["surface"])])

    ax.set_xticks([np.log10(t) for t in TICKS])
    ax.set_xticklabels(TICKLAB, fontsize=F_TICK)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.16 * top, top * 1.10)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["muted"])
    ax.tick_params(colors=T["ink"], labelsize=F_TICK)
    ax.grid(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/ood_mnist/run1/mnist_ood_s0")
    ap.add_argument("--ood", default="fmnist", choices=OOD_ORDER)
    ap.add_argument("--all", action="store_true", help="2x2, one panel per probe dataset")
    ap.add_argument("--theme", default="light", choices=tuple(THEMES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cal, idh, ood = load(a.run)
    T, cols = THEMES[a.theme], PALETTES["project"]

    if a.all:
        fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.5), facecolor=T["surface"])
        for ax, name in zip(axes.ravel(), OOD_ORDER):
            draw(ax, cal, idh, ood[name], name, T, cols)
        for ax in axes[-1]:
            ax.set_xlabel("predictive entropy", fontsize=F_LABEL, color=T["ink"])
        out = a.out or os.path.join("results", "figures", "confidence_overlap_all")
    else:
        fig, ax = plt.subplots(figsize=(11.0, 6.4), facecolor=T["surface"])
        draw(ax, cal, idh, ood[a.ood], a.ood, T, cols)
        ax.set_xlabel("predictive entropy", fontsize=F_LABEL, color=T["ink"])
        out = a.out or os.path.join("results", "figures",
                                    f"confidence_overlap_{a.ood}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=T["surface"], bbox_inches="tight")
    plt.close(fig)

    print(f"{'probe':<10}" + "".join(f"{'a=' + format(x, 'g'):>9}" for x in ALPHAS))
    print(f"{'ID kept':<10}" + "".join(f"{1 - x:>9.1%}" for x in ALPHAS))
    for name in OOD_ORDER:
        row = ""
        for al in ALPHAS:
            h_tau = float(np.quantile(cal, 1.0 - al))
            row += f"{(ood[name] > h_tau).mean():>9.1%}"
        print(f"{PRETTY[name]:<10}{row}")
    print("\n[plot]", out + ".png (+ .pdf)")


if __name__ == "__main__":
    main()
