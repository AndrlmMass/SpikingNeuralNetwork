"""Weight dynamics across the wake--sleep cycle, from a real training run.

Replaces the hand-drawn schematic (tikz) of the sleep section. The data come
from one 1000-image MNIST batch run with the paper's sleep configuration
(ratio 0.3, lambda = lambda_for_ratio(0.3), interval 3500, band termination,
give_up on timeout), with weight tracking and episode dumps switched on:

    SNN_DUMP_EPISODES=results/sleep_trace python src/main.py --reg-method sleep \
        --dataset mnist --seed 42 --runs 1 --num-steps 100 \
        --check-sleep-interval 3500 --sleep-rate 0.3 --clip-always --no-plots \
        --sleep-decay-rate <lambda_for_ratio(0.3)> --sleep-max-iters 35000 \
        --on-timeout give_up --sleep-termination band --n-train 1000 \
        --n-test 200 --plot-weights-per-epoch

Inputs:  results/sleep_trace/tracking.json   (snapshot trajectory)
         results/sleep_trace/episode_*.npz   (all weights before/after each episode)
Output:  figures/sleep_trace_BW.{pdf,png}

Panel (a): total excitatory and inhibitory weight over the batch, with each
sleep episode drawn at its true length in iterations (wake steps plus virtual
sleep iterations). Panels (b, c): one early and one late episode, with the
trajectory that the power-law pull alone predicts (dotted) on top of the
recorded one.
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
SRC = os.path.join(REPO, "results", "sleep_trace")
FIG = os.path.join(REPO, "figures")

T = dict(surface="#ffffff", ink="#1a1a1a", muted="#666666", grid="#e6e6e6",
         axis="#444444", band="#e8e8e8")
FS_LABEL, FS_TICK, FS_SERIES = 17, 14, 14
ZOOM = (1, 26)       # episodes shown in (b) and (c): sleep raises W, then lowers it
PAD = 1500           # wake steps shown either side of the zoomed episode


def load():
    with open(os.path.join(SRC, "tracking.json")) as f:
        tr = json.load(f)
    eps = [np.load(p) for p in sorted(glob.glob(os.path.join(SRC, "episode_*.npz")))]
    return tr, eps


def time_axis(tr, sleep_iters):
    """Elapsed iterations for every snapshot: wake snapshots are one step
    apart, sleep snapshots are spread evenly over the episode's iterations."""
    n = len(tr["times"])
    step = np.ones(n)
    in_sleep = np.zeros(n, bool)
    for (s, e), it in zip(tr["sleep_segments"], sleep_iters):
        s, e = int(s), int(e)
        step[s + 1:e + 1] = it / (e - s)
        in_sleep[s:e + 1] = True
    x = np.cumsum(step) - step[0]
    return x, in_sleep


def to_sums(tr, eps):
    """tracking.json stores the mean over the full weight block (zeros
    included), which is proportional to the total weight. The constant is
    fixed from the first episode's exact before-sum."""
    s0 = int(tr["sleep_segments"][0][0])
    c_exc = eps[0]["exc_before"].sum() / tr["exc_mean"][s0]
    c_inh = np.abs(eps[0]["inh_before"]).sum() / abs(tr["inh_mean"][s0])
    return (np.asarray(tr["exc_mean"]) * c_exc,
            np.abs(np.asarray(tr["inh_mean"])) * c_inh)


def predicted(ep, n):
    """Total excitatory weight after n pure power-law iterations from the
    episode's starting weights: u -> lambda^n u in u = log(w / w_tgt)."""
    w0, tgt = ep["exc_before"].astype(float), np.abs(ep["target_exc"].astype(float))
    u0 = np.log(w0 / tgt)
    lam = float(ep["rate_exc"])
    return np.array([(tgt * np.exp(lam ** k * u0)).sum() for k in n])


def style(ax):
    ax.set_facecolor(T["surface"])
    ax.grid(True, axis="y", color=T["grid"], lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    ax.tick_params(colors=T["ink"], labelsize=FS_TICK)


def shade(ax, x, in_sleep, scale=1.0):
    edges = np.flatnonzero(np.diff(in_sleep.astype(int)))
    starts, ends = edges[::2] + 1, edges[1::2]
    for a, b in zip(starts, ends):
        ax.axvspan(x[a] * scale, x[b] * scale, color=T["band"], lw=0, zorder=0.5)


def main():
    tr, eps = load()
    iters = [int(e["sleep_iters"]) for e in eps]
    x, in_sleep = time_axis(tr, iters)
    w_exc, w_inh = to_sums(tr, eps)

    fig = plt.figure(figsize=(13, 8.2), facecolor=T["surface"])
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)

    # (a) whole batch, relative to the first snapshot
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    shade(ax, x, in_sleep, scale=1e-3)
    ax.plot(x / 1e3, w_exc / w_exc[0], color=T["ink"], lw=1.4, zorder=3)
    ax.plot(x / 1e3, w_inh / w_inh[0], color=T["muted"], lw=1.4, ls="--", zorder=3)
    ax.set_xlim(0, x[-1] / 1e3 * 1.08)
    ax.text(x[-1] / 1e3 * 1.008, w_exc[-1] / w_exc[0], "excitatory",
            va="center", fontsize=FS_SERIES, color=T["ink"])
    ax.text(x[-1] / 1e3 * 1.008, w_inh[-1] / w_inh[0], "inhibitory",
            va="center", fontsize=FS_SERIES, color=T["muted"])
    ax.set_xlabel(r"Elapsed iterations ($\times 10^3$; wake steps + sleep iterations)",
                  fontsize=FS_LABEL, color=T["ink"])
    ax.set_ylabel(r"$\mathcal{W}_x(t)\,/\,\mathcal{W}_x(0)$", fontsize=FS_LABEL,
                  color=T["ink"])
    ax.text(0.005, 0.97, "(a)", transform=ax.transAxes, fontsize=FS_LABEL,
            va="top", color=T["ink"])

    # (b, c) single episodes, excitatory total, recorded vs pure power law
    for col, k in enumerate(ZOOM):
        ax = fig.add_subplot(gs[1, col])
        style(ax)
        s, e = (int(v) for v in tr["sleep_segments"][k])
        lo = np.searchsorted(x, x[s] - PAD)
        hi = np.searchsorted(x, x[e] + PAD)
        xs = x[lo:hi] - x[s]
        shade(ax, x[lo:hi] - x[s], in_sleep[lo:hi])
        ref = w_exc[s]
        ax.plot(xs, 100 * (w_exc[lo:hi] / ref - 1), color=T["ink"], lw=1.6, zorder=3)
        n = np.linspace(0, iters[k], 60).round().astype(int)
        pred = predicted(eps[k], n)
        ax.plot(n, 100 * (pred / pred[0] - 1), color=T["muted"], lw=2.4, ls=":",
                zorder=4)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xlabel("Iterations from sleep onset", fontsize=FS_LABEL, color=T["ink"])
        if col == 0:
            ax.set_ylabel(r"$\Delta\mathcal{W}_E$ (%)", fontsize=FS_LABEL, color=T["ink"])
        ax.text(0.02, 0.97, f"({'bc'[col]}) episode {k + 1}", transform=ax.transAxes,
                fontsize=FS_SERIES, va="top", color=T["ink"])
        mid = iters[k] / 2
        ax.annotate("sleep", (mid, ax.get_ylim()[0]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_SERIES, color=T["muted"])

    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"sleep_trace_BW.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=T["surface"])
    print("wrote figures/sleep_trace_BW.{pdf,png}")


if __name__ == "__main__":
    main()
