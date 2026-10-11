"""Trajectories of individual synapses across wake and sleep, and rank retention.

Data: one MNIST batch with the paper's sleep configuration (ratio 0.3,
lambda = lambda_for_ratio(0.3), interval 3500), 1000 tracked excitatory
synapses recorded every 20 wake steps and every 10 sleep iterations:

    SNN_DUMP_EPISODES=results/sleep_trace_1000 python src/main.py --reg-method sleep \
        --dataset mnist --seed 42 --runs 1 --num-steps 100 --check-sleep-interval 3500 \
        --sleep-rate 0.3 --clip-always --no-plots \
        --sleep-decay-rate <lambda_for_ratio(0.3)> --sleep-max-iters 35000 \
        --on-timeout give_up --sleep-termination band --n-train 1000 --n-test 200 \
        --plot-weights-per-epoch --weight-track-samples 1000 \
        --weight-track-interval 20 --weight-track-sleep-interval 10

Inputs:  results/sleep_trace_1000/tracking.json, episode_*.npz
Output:  figures/sleep_trajectories_BW.{pdf,png}

Weights are min-max normalized to [0, 1] using the distribution of the tracked
synapses at the start of the plotted window, so runs are on a common scale.
Lines are shaded by each synapse's rank at the start of the window.
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
SRC = os.path.join(REPO, "results", "sleep_trace_1000")
FIG = os.path.join(REPO, "figures")

T = dict(surface="#ffffff", ink="#1a1a1a", grid="#e6e6e6", axis="#444444", band="#ececec")
FS = 15
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS + 1, "xtick.labelsize": FS - 1,
                     "ytick.labelsize": FS - 1})
CYCLES = 3          # last N wake-sleep cycles of the batch


def style(ax):
    ax.set_facecolor(T["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    ax.tick_params(colors=T["ink"])


def load():
    with open(os.path.join(SRC, "tracking.json")) as f:
        tr = json.load(f)
    eps = [np.load(p) for p in sorted(glob.glob(os.path.join(SRC, "episode_*.npz")))]
    return tr, eps


def time_axis(tr, eps):
    """Elapsed iterations per snapshot: wake snapshots are `wake_every` steps
    apart, sleep snapshots spread evenly over the episode's iterations."""
    n = len(tr["times"])
    wake_every = 20
    step = np.full(n, float(wake_every))
    in_sleep = np.zeros(n, bool)
    for (s, e), ep in zip(tr["sleep_segments"], eps):
        s, e = int(s), int(e)
        step[s + 1:e + 1] = int(ep["sleep_iters"]) / max(1, e - s)
        in_sleep[s:e + 1] = True
    return np.cumsum(step) - step[0], in_sleep


def main():
    tr, eps = load()
    W = np.abs(np.array(tr["exc_samples"], dtype=float))      # snapshots x synapses
    x, in_sleep = time_axis(tr, eps)

    # window: start of the wake period preceding the last CYCLES sleep episodes
    segs = [(int(s), int(e)) for s, e in tr["sleep_segments"]]
    prev_end = segs[-CYCLES - 1][1] + 1
    lo, hi = prev_end, len(x)
    Ww, xw, sw = W[lo:hi], x[lo:hi] - x[lo], in_sleep[lo:hi]
    w0 = Ww[0]
    Wn = (Ww - w0.min()) / (w0.max() - w0.min())           # min-max at window start
    tgt = (0.5 - w0.min()) / (w0.max() - w0.min())
    rank0 = np.argsort(np.argsort(w0)) / (len(w0) - 1)

    fig, (ax, az) = plt.subplots(2, 1, figsize=(13, 10), facecolor=T["surface"],
                                 gridspec_kw=dict(hspace=0.35))
    style(ax)
    edges = np.flatnonzero(np.diff(sw.astype(int)))
    for a, b in zip(edges[::2] + 1, edges[1::2]):
        ax.axvspan(xw[a] / 1e3, xw[b] / 1e3, color=T["band"], lw=0, zorder=0)
    cmap = plt.get_cmap("Greys")
    for j in np.argsort(rank0):
        ax.plot(xw / 1e3, Wn[:, j], color=cmap(0.25 + 0.7 * rank0[j]), lw=0.4,
                alpha=0.5, zorder=2, rasterized=True)
    if 0 <= tgt <= 1.2:
        ax.axhline(tgt, color=T["ink"], ls=":", lw=1.4, zorder=3)
    ax.set_xlabel(r"elapsed iterations ($\times 10^3$; wake steps + sleep iterations)")
    ax.set_ylabel("normalized weight")
    ax.set_xlim(0, xw[-1] / 1e3)
    ax.text(-0.06, 1.02, "a", transform=ax.transAxes, fontsize=FS + 3, fontweight="bold")

    # (b) zoom: last wake period + last sleep episode, 12 synapses across the
    # ranks, each as % change from its own weight at the start of the zoom
    s_last, e_last = segs[-1]
    z0 = segs[-2][1] + 1
    Wz, xz, sz = W[z0:e_last + 1], x[z0:e_last + 1] - x[z0], in_sleep[z0:e_last + 1]
    style(az)
    az.axvspan(xz[np.argmax(sz)], xz[-1], color=T["band"], lw=0, zorder=0)
    order = np.argsort(Wz[0])
    pick = order[np.linspace(0.05, 0.99, 12).__mul__(len(order) - 1).astype(int)]
    for k, j in enumerate(pick):
        az.plot(xz, 100 * (Wz[:, j] / Wz[0, j] - 1), color=cmap(0.3 + 0.65 * k / 11),
                lw=1.2, zorder=2)
    az.axhline(0, color=T["ink"], lw=0.8, zorder=1)
    az.set_xlabel("iterations from start of wake period")
    az.set_ylabel("change from start (%)")
    az.set_xlim(0, xz[-1])
    az.text(-0.06, 1.02, "b", transform=az.transAxes, fontsize=FS + 3, fontweight="bold")
    az.text(0.01, 0.97, "wake", transform=az.transAxes, va="top", fontsize=FS - 1)
    az.text(xz[np.argmax(sz)] / xz[-1] + 0.01, 0.97, "sleep", transform=az.transAxes,
            va="top", fontsize=FS - 1)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"sleep_trajectories_BW.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=T["surface"])
    print("wrote figures/sleep_trajectories_BW.{png,pdf}; window snapshots:", hi - lo,
          "synapses:", W.shape[1])


if __name__ == "__main__":
    main()
