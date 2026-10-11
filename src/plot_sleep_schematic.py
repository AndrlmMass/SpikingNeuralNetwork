"""Conceptual wake/sleep schematic (Fig. 4), modelled on sleep_cycle_weights_100k.

Not simulated network data: a handful of synapses follow the two rules of the
model in log-distance space u = ln(|w| / w_tgt).
  wake   u grows at a synapse-specific rate (STDP drift) with jitter, so the
         weights spread apart and the strongest run away;
  sleep  u shrinks geometrically, u <- s * u per episode (Eq. decay), so every
         weight is pulled toward the target from either side, smoothly and
         without changing their order.
Episode strength is exaggerated (s = 0.25) so the pull is visible; repeated
cycles settle into a bounded oscillation instead of runaway growth.

Output: figures/sleep_schematic_BW.{pdf,png}
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")

T = dict(surface="#ffffff", ink="#1a1a1a", axis="#444444", band="#ececec")
FS = 15
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS + 1, "xtick.labelsize": FS - 1,
                     "ytick.labelsize": FS - 1})

WAKE, SLEEP, CYCLES = 200, 80, 4         # schematic time units; four wake-sleep cycles
S_EPISODE = 0.35                          # fraction of log-distance left after one episode
N_SYN = 16
RATES = np.linspace(-0.45, 1.1, N_SYN)    # log-growth per wake period (STDP drift)
# starting weights spread on both sides of the target, shuffled against the rates so
# synapses approach the target from different heights and at different slopes
U0 = np.random.default_rng(1).permutation(np.linspace(-0.9, 0.5, N_SYN))
JITTER = 0.01                             # per-step SD of the wake drift
DT = 1.0


def simulate(seed=4):
    rng = np.random.default_rng(seed)
    t, u = [0.0], [U0.copy()]
    phases = []
    for _ in range(CYCLES):
        t0 = t[-1]
        for _ in range(int(WAKE / DT)):                       # wake: drift + jitter
            du = RATES / WAKE * DT + JITTER * np.sqrt(DT) * rng.standard_normal(len(RATES))
            u.append(u[-1] + du)
            t.append(t[-1] + DT)
        phases.append(("wake", t0, t[-1]))
        t0, u_start = t[-1], u[-1].copy()
        for k in range(1, int(SLEEP / DT) + 1):               # sleep: geometric pull
            u.append(u_start * S_EPISODE ** (k * DT / SLEEP))
            t.append(t[-1] + DT)
        phases.append(("sleep", t0, t[-1]))
    return np.array(t), np.exp(np.array(u)), phases


def main():
    t, w, phases = simulate()
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=T["surface"])
    ax.set_facecolor(T["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["axis"])
    top = w.max() * 1.18
    for kind, a, b in phases:
        if kind == "sleep":
            ax.axvspan(a, b, color=T["band"], lw=0, zorder=0)
        ax.text((a + b) / 2, top * 0.985, kind, ha="center", va="top", fontsize=FS)
    shades = np.linspace(0.35, 1.0, len(RATES))               # light = weak, dark = strong
    for j in range(len(RATES)):
        ax.plot(t, w[:, j], color=plt.get_cmap("Greys")(shades[j]), lw=1.6, zorder=2)
    ax.axhline(1.0, color=T["ink"], ls=":", lw=1.4, zorder=3)
    ax.text(t[-1] * 1.01, 1.0, r"$w_{\mathrm{tgt}}$", ha="left", va="center",
            fontsize=FS, color=T["ink"], clip_on=False)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.set_ylabel(r"synaptic weight $|w|$")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"sleep_schematic_BW.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=T["surface"])
    print("wrote figures/sleep_schematic_BW; peak weight per wake period:",
          [round(float(w[(t >= a) & (t <= b)].max()), 2) for k, a, b in phases if k == "wake"])


if __name__ == "__main__":
    main()
