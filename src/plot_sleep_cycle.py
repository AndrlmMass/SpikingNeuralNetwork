"""One wake period followed by one sleep episode, for 1000 tracked synapses.

Data: one MNIST batch with sleep ratio 1.0 (sleep episodes as long as the wake
period, so the two halves of the plot are equal) and downscaling as the only
sleep component (no sleep-phase STDP, no membrane noise; input is held), with
1000 excitatory synapses recorded every 20 iterations:

    SNN_DUMP_EPISODES=results/sleep_trace_downscale python src/main.py \
        --reg-method sleep --dataset mnist --seed 42 --runs 1 --num-steps 100 \
        --check-sleep-interval 3500 --sleep-rate 1.0 --clip-always --no-plots \
        --sleep-decay-rate <lambda_for_ratio(1.0)> --sleep-max-iters 35000 \
        --on-timeout give_up --sleep-termination band --sleep-components downscale \
        --n-train 1000 --n-test 200 --plot-weights-per-epoch \
        --weight-track-samples 1000 --weight-track-interval 20 \
        --weight-track-sleep-interval 20

Inputs:  results/sleep_trace_downscale/tracking.json, episode_*.npz
Output:  figures/sleep_cycle_BW.{pdf,png}

Each synapse is drawn as its % change from its own weight at the start of the
wake period; lines are shaded by that starting weight (light = weak, dark = strong).
"""

import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "results", "sleep_trace_downscale")
FIG = os.path.join(REPO, "figures")

T = dict(surface="#ffffff", ink="#1a1a1a", axis="#444444", band="#ececec")
FS = 15
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS + 1, "xtick.labelsize": FS - 1,
                     "ytick.labelsize": FS - 1})
WAKE_EVERY = 20


def main(src=SRC, tag="", wake_every=WAKE_EVERY, n_highlight=0):
    wake_every, n_highlight = int(wake_every), int(n_highlight)
    """src: run folder with tracking.json and episode dumps; tag: suffix for
    the output file names (e.g. "_long")."""
    with open(os.path.join(src, "tracking.json")) as f:
        tr = json.load(f)
    eps = [np.load(p) for p in sorted(glob.glob(os.path.join(src, "episode_*.npz")))]
    W = np.abs(np.array(tr["exc_samples"], dtype=float))
    segs = [(int(s), int(e)) for s, e in tr["sleep_segments"]]

    # last wake period + last sleep episode
    s_last, e_last = segs[-1]
    # the wake period before the last episode: after the previous episode, or
    # from the start of the batch when there is only one
    z0 = segs[-2][1] + 1 if len(segs) > 1 else 0
    iters = int(eps[-1]["sleep_iters"])
    n_wake, n_sleep = s_last - z0, e_last - s_last
    x = np.concatenate([np.arange(n_wake + 1) * wake_every,
                        n_wake * wake_every + np.arange(1, n_sleep + 1) * iters / n_sleep])
    Wz = W[z0:e_last + 1][:len(x)]
    Y = 100 * (Wz / Wz[0] - 1)
    rank0 = np.argsort(np.argsort(Wz[0])) / (Wz.shape[1] - 1)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=T["surface"])
    ax.set_facecolor(T["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    x_sleep = n_wake * wake_every
    ax.axvspan(x_sleep, x[-1], color=T["band"], lw=0, zorder=0)
    cmap = plt.get_cmap("Greys")
    for j in np.argsort(rank0):
        ax.plot(x, Y[:, j], color=cmap(0.25 + 0.7 * rank0[j]), lw=0.4, alpha=0.35, zorder=2, rasterized=True)
    ax.axhline(0, color=T["ink"], lw=0.8, zorder=3)
    ax.set_xlim(0, x[-1])
    ax.set_xlabel("iterations from start of wake period")
    ax.set_ylabel("change in weight from start (%)")
    ax.text(0.01, 0.97, "wake", transform=ax.transAxes, va="top", fontsize=FS)
    ax.text(x_sleep / x[-1] + 0.01, 0.97, "sleep (downscaling only)", transform=ax.transAxes,
            va="top", fontsize=FS)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"sleep_cycle{tag}_BW.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=T["surface"])
    print(f"wrote figures/sleep_cycle{tag}_BW; synapses {Wz.shape[1]}, wake points {n_wake + 1}, "
          f"sleep points {n_sleep}, sleep iterations {iters}")

    # Same cycle, actual weights: during sleep no lines cross, so the rank
    # order of the synapses is preserved; during wake they can.
    tgt = float(np.abs(eps[-1]["target_exc"][0]))
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=T["surface"])
    ax.set_facecolor(T["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    ax.axvspan(x_sleep, x[-1], color=T["band"], lw=0, zorder=0)
    for j in np.argsort(rank0):
        ax.plot(x, Wz[:, j], color=cmap(0.25 + 0.7 * rank0[j]), lw=0.4, alpha=0.35, zorder=2, rasterized=True)
    # A few example synapses drawn solid on top of the bundle, so the jitter of
    # individual trajectories during wake is visible against the smooth decay.
    if n_highlight:
        rng = np.random.default_rng(0)
        for j in rng.choice(Wz.shape[1], size=n_highlight, replace=False):
            ax.plot(x, Wz[:, j], color=T["ink"], lw=0.9, alpha=0.9, zorder=4, rasterized=True)
    ax.axhline(tgt, color=T["ink"], ls=":", lw=1.4, zorder=3)
    ax.text(x[-1] * 1.01, tgt, "target", ha="left", va="center", fontsize=FS - 1,
            color=T["ink"], clip_on=False)
    ax.set_xlim(0, x[-1])
    lo_, hi_ = min(Wz.min(), tgt), max(Wz.max(), tgt)
    ax.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.15 * (hi_ - lo_))   # headroom for labels
    ax.set_xlabel("iterations from start of wake period")
    ax.set_ylabel(r"weight $|w|$")
    ax.text(0.01, 0.97, "wake", transform=ax.transAxes, va="top", fontsize=FS)
    ax.text(x_sleep / x[-1] + 0.01, 0.97, "sleep (downscaling only)", transform=ax.transAxes,
            va="top", fontsize=FS)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"sleep_cycle_weights{tag}_BW.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=T["surface"])
    print(f"wrote figures/sleep_cycle_weights{tag}_BW")


if __name__ == "__main__":
    import sys
    # usage: python src/plot_sleep_cycle.py [run folder] [output tag] [wake snapshot interval] [n highlighted]
    main(*(sys.argv[1:5]))
