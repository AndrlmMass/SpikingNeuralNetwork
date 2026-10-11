"""Single-panel versions of the mechanism figures, for LaTeX subfigures.

Data: the downscaling-only MNIST runs (sleep ratio 0.3, sleep components =
downscale; 4000 training images, 112 sleep episodes) in
results/weight_dynamics/mnist_downscale_s{seed}/episode_*.npz, which hold every
excitatory weight immediately before and after each sleep episode.

Outputs (figures/, each as .pdf and .png, no panel letters):
  shift_toward_target_a / _b   distribution before/after one sleep episode,
                               episode 1 and episode 61, log axis of |w|/w_tgt
  rank_rank_sleep / _wake      percentile rank before vs after one sleep
                               episode, and the wake period that follows it
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "glmm"))
from plot_glmms_bw import T, style  # noqa: E402

FIG = os.path.join(REPO, "figures")
TGT = 0.5
SEED = 42
INK, LIGHT = T["ink"], "#d4d4d4"
FS = 15
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS + 1, "xtick.labelsize": FS - 1,
                     "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1})


def load(seed=SEED, root="weight_dynamics"):
    fs = sorted(glob.glob(os.path.join(REPO, "results", root,
                                       f"mnist_downscale_s{seed}", "episode_*.npz")))
    E = [np.load(f) for f in fs]
    return ([np.abs(e["exc_before"].astype(float)) for e in E],
            [np.abs(e["exc_after"].astype(float)) for e in E])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight",
                    facecolor=T["surface"])
    plt.close(fig)
    print("wrote figures/" + name)


def shift_panel(b, a, ep, name, legend):
    """Distribution of |w| / w_tgt before (grey) and after (outline) one episode."""
    b, a = b / TGT, a / TGT
    fig, ax = plt.subplots(figsize=(6.5, 4.4), facecolor=T["surface"])
    style(ax, ygrid=False)
    lo, hi = min(b.min(), a.min(), 1) / 1.15, max(b.max(), a.max(), 1) * 1.15
    bins = np.geomspace(lo, hi, 70)
    ax.hist(b, bins=bins, color=LIGHT, zorder=2)
    ax.hist(a, bins=bins, histtype="step", color=INK, lw=1.5, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    cand = [0.05, 0.1, 0.2, 0.5, 0.6, 0.8, 1, 1.5, 2, 5, 10, 20]
    ticks = [c for c in cand if lo <= c <= hi]
    if len(ticks) > 6:
        ticks = [c for c in ticks if c in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)]
    ax.set_xticks(ticks)
    ax.axvline(1, color=INK, ls=":", lw=1.6, zorder=4)
    ax.text(1.04, ax.get_ylim()[1] * 0.97, "target", ha="left", va="top", fontsize=FS - 1,
            color=INK)
    ax.tick_params(labelsize=FS - 1)
    ax.set_ylabel("number of weights")
    ax.set_xlabel("absolute weight relative to target")
    if legend:
        h = [plt.Rectangle((0, 0), 1, 1, color=LIGHT), plt.Line2D([], [], color=INK, lw=1.5)]
        ax.legend(h, ["before sleep", "after sleep"], loc="upper left",
                  bbox_to_anchor=(0.3, 1.0), frameon=False)
    print(f"episode {ep}: median {np.median(b):.3f} -> {np.median(a):.3f} (x w_tgt)")
    save(fig, name)


def pct_rank(w):
    return 100 * (rankdata(w) - 1) / (len(w) - 1)


def rank_panel(x0, x1, phase, name):
    """Percentile rank before vs after one phase, as a log-density hexbin."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0), facecolor=T["surface"])
    style(ax, ygrid=False)
    ax.hexbin(pct_rank(x0), pct_rank(x1), gridsize=60, cmap="Greys", bins="log", mincnt=1,
              linewidths=0)
    ax.set_xlabel(f"percentile rank before {phase}")
    ax.set_ylabel("percentile rank after")
    ax.set_aspect("equal")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    save(fig, name)


def main(root="weight_dynamics", tag=""):
    """root: folder under results/ with mnist_downscale_s{seed}/ episode dumps;
    tag: suffix for the output names. The rank panels are drawn only for the
    default data, so the published rank figure is not overwritten."""
    B, A = load(root=root)
    shift_panel(B[0], A[0], 1, f"shift_toward_target_a{tag}", legend=True)
    shift_panel(B[60], A[60], 61, f"shift_toward_target_b{tag}", legend=False)
    if not tag:
        k = 0      # first episode, as in the combined P3 figure
        rank_panel(B[k], A[k], "sleep", "rank_rank_sleep")
        rank_panel(A[k], B[k + 1], "wake", "rank_rank_wake")


if __name__ == "__main__":
    # usage: python src/plot_mechanism_panels.py [results subfolder] [output tag]
    main(*(sys.argv[1:3]))
