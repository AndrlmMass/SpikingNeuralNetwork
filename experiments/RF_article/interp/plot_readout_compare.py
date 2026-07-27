"""Spiking readout vs delta (softmax) readout: online accuracy over training.

Both decoders read the SAME excitatory spikes on the SAME trial, so this is a
paired within-run comparison, not two separate runs. Online accuracy is each
decoder's own running accuracy over the checkpoint window; the two dots at the
right edge are the held-out TEST accuracies from results.json.

  python experiments/RF_article/interp/plot_readout_compare.py --run <run_dir>
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project figure conventions (mirrors plot_risk_coverage.py)
SERIES = {"delta": "#DB8383", "spiking": "#6D8BA9"}
T = dict(surface="#ffffff", ink="#1a1a1a", ink2="#333333", muted="#666666",
         grid="#e6e6e6", axis="#444444")
FS_LABEL, FS_TICK, FS_SERIES = 24, 18, 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = a.run
    r = json.load(open(os.path.join(run, "results.json")))
    traj = r["trajectory"]
    n_img = r["config"].get("train_all", len(traj) * 1000)
    per = n_img / len(traj)
    x = [(t["batch"] + 1) * per / 1000.0 for t in traj]   # thousands of images

    delta = [t.get("readout_learned_acc", np.nan) for t in traj]
    spike = [t.get("spiking_online_acc", np.nan) for t in traj]
    d_test = r.get("uncertainty", [{}, {}])[1].get("base_acc")
    s_test = r.get("test_spiking_acc")

    fig = plt.figure(figsize=(10.5, 6.2), facecolor=T["surface"])
    ax = fig.add_axes([0.115, 0.145, 0.60, 0.82])
    ax.set_facecolor(T["surface"])
    ax.grid(True, axis="y", color=T["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["axis"]); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=T["ink2"], labelsize=FS_TICK, length=5, width=1.0)

    ax.plot(x, delta, color=SERIES["delta"], lw=2.4, solid_capstyle="round", zorder=4)
    ax.plot(x, spike, color=SERIES["spiking"], lw=2.4, solid_capstyle="round", zorder=3)

    labels = []
    if d_test is not None:
        ax.plot([x[-1]], [d_test], "o", color=SERIES["delta"], ms=9, zorder=5)
        labels.append((d_test, f"delta readout\n{d_test:.1%} test", SERIES["delta"]))
    if s_test is not None:
        ax.plot([x[-1]], [s_test], "o", color=SERIES["spiking"], ms=9, zorder=5)
        labels.append((s_test, f"spiking readout\n{s_test:.1%} test", SERIES["spiking"]))
    labels.sort(key=lambda e: -e[0])
    gap = 0.09 * (max(delta + spike) - min(delta + spike) + 0.2)
    ys, prev = [], None
    for val, _, _ in labels:
        y = val if prev is None else min(val, prev - gap); ys.append(y); prev = y
    for (val, lbl, col), y in zip(labels, ys):
        ax.annotate(lbl, xy=(x[-1] + 0.15, y), va="center", ha="left",
                    fontsize=FS_SERIES, color=col, linespacing=1.35,
                    annotation_clip=False)

    ax.set_xlabel("training images (thousands)", color=T["ink"], fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel("accuracy", color=T["ink"], fontsize=FS_LABEL, labelpad=8)
    ax.set_xlim(0, x[-1] * 1.02)
    lo = min(min(delta), min(spike)) - 0.03
    ax.set_ylim(max(0, lo), 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))

    out = os.path.join(run, "readout_compare")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=T["surface"])
    plt.close(fig)
    print("saved ->", out + ".png")
    print(f"delta  test {d_test:.4f}   spiking test {s_test:.4f}   gap {100*(d_test-s_test):+.1f} pts")


if __name__ == "__main__":
    main()
