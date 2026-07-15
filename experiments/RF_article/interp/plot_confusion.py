"""
Confusion + progress plot for one interp run (reward / grouped / tiled).

Top (full width): readout accuracy (softmax/pool over the class-group neurons —
the honest task metric) + grouped clustering (val_phi = group eta^2) over training.
Bottom: final TEST confusion matrices (readout vs linear, row-normalized = per-class
recall) + per-class recall over checkpoints (which classes improve / stay confused).

The harness regenerates this each checkpoint (live), so during a run the test
matrices are blank until the end but the over-time panels update as you go.

  python experiments/RF_article/interp/plot_confusion.py --results <cell_dir_or_results.json>
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    return json.load(open(path)), os.path.dirname(path)


def heat(ax, cm, title):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)   # row-normalized = recall
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    K = cm.shape[0]
    for i in range(K):
        for j in range(K):
            if norm[i, j] > 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{title}\n(acc {np.trace(cm) / max(cm.sum(), 1):.3f})")


def _finite(ys):
    return np.any(np.isfinite(np.asarray(ys, dtype=float)))


def accuracy_over_time(ax, traj):
    """Readout accuracy (primary) + fitted LR (reference); grouped clustering on twin."""
    xs = [t["batch"] + 1 for t in traj]
    ser = lambda k: [t.get(k, np.nan) for t in traj]
    for key, lbl, c in [("readout_learned_acc", "readout (learned)", "#9467bd"),
                        ("softmax_acc", "readout (uniform pool)", "#d62728"),
                        ("online_acc", "online train decisions", "#2ca02c"),
                        ("val_acc", "fitted PCA+LR", "#999999")]:
        if _finite(ser(key)):
            ax.plot(xs, ser(key), "-o", ms=3, color=c, label=lbl)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.set_xlabel("checkpoint (batch+1)")
    ax.set_title("Readout accuracy + grouped clustering (val phi) over training")
    ax.axhline(0.1, ls=":", lw=0.8, color="k", alpha=0.5)   # chance
    ax.legend(loc="lower left", fontsize=7)
    if _finite(ser("val_phi")):
        axp = ax.twinx()
        axp.plot(xs, ser("val_phi"), "-", color="steelblue", lw=1.4, label="val phi (group eta^2)")
        axp.set_ylabel("phi (group eta^2)", color="steelblue")
        axp.tick_params(axis="y", labelcolor="steelblue")
        axp.legend(loc="lower right", fontsize=7)


def per_class_recall(ax, traj):
    key = "cm_readout" if any("cm_readout" in t for t in traj) else "cm_linear"
    xs, series = [], []
    for t in traj:
        if key in t:
            cm = np.asarray(t[key], dtype=float)
            xs.append(t["batch"] + 1)
            series.append(np.diag(cm) / np.clip(cm.sum(1), 1, None))
    if not xs:
        ax.axis("off"); ax.set_title("no per-checkpoint CMs"); return
    series = np.array(series)
    cmap = plt.get_cmap("tab10")
    for c in range(series.shape[1]):
        ax.plot(xs, series[:, c], "-o", ms=3, color=cmap(c), label=str(c))
    ax.set_ylim(0, 1.02); ax.set_xlabel("checkpoint (batch+1)"); ax.set_ylabel("recall")
    ax.set_title(f"Per-class recall over training ({key})")
    ax.grid(alpha=0.3); ax.legend(title="class", fontsize=7, ncol=2)


def make_confusion_plot(d, outdir):
    """Render confusion.png from a results dict (works mid-run with a partial dict)."""
    tag = d.get("config", {}).get("tag", "run")
    traj = d.get("trajectory", [])
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)
    accuracy_over_time(fig.add_subplot(gs[0, :]), traj)

    ax0 = fig.add_subplot(gs[1, 0])
    if "test_cm_readout" in d:
        heat(ax0, d["test_cm_readout"], f"{tag} — TEST readout")
    else:
        ax0.axis("off"); ax0.set_title("TEST readout CM\n(pending — end of run)")
    ax1 = fig.add_subplot(gs[1, 1])
    if "test_cm_linear" in d:
        heat(ax1, d["test_cm_linear"], f"{tag} — TEST linear classifier")
    else:
        ax1.axis("off"); ax1.set_title("TEST linear CM\n(pending — end of run)")
    per_class_recall(fig.add_subplot(gs[1, 2]), traj)

    fig.suptitle(f"Confusion + progress — {tag}", fontsize=14)
    out = os.path.join(outdir, "confusion.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run dir or results.json path")
    args = ap.parse_args()
    d, outdir = load(args.results)
    print("saved ->", make_confusion_plot(d, outdir))


if __name__ == "__main__":
    main()
