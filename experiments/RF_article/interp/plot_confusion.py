"""
Confusion-matrix plots for one interp cell (reward / grouped run).

Left/middle: final TEST confusion matrices, row-normalized (per-class recall on
the diagonal), for the group/pool readout and the linear classifier — so you can
compare where each readout confuses classes.
Right: per-class recall over training checkpoints (readout), so you can see which
classes improve and which stay confused.

  python experiments/RF_article/interp/plot_confusion.py --results <cell_dir_or_results.json>
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    return json.load(open(path)), os.path.dirname(path)


def heat(ax, cm, title):
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(1, keepdims=True)
    norm = cm / np.clip(row, 1, None)          # row-normalized = per-class recall
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    K = cm.shape[0]
    for i in range(K):
        for j in range(K):
            v = norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    acc = np.trace(cm) / max(cm.sum(), 1)
    ax.set_title(f"{title}\n(acc {acc:.3f})")
    return im


def recall_over_time(traj, key):
    xs, series = [], []
    for t in traj:
        if key not in t:
            continue
        cm = np.asarray(t[key], dtype=float)
        rec = np.diag(cm) / np.clip(cm.sum(1), 1, None)
        xs.append(t["batch"] + 1); series.append(rec)
    return xs, (np.array(series) if series else np.empty((0, 10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="cell dir or results.json path")
    args = ap.parse_args()
    d, outdir = load(args.results)
    tag = d.get("config", {}).get("tag", "cell")
    traj = d.get("trajectory", [])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    if "test_cm_readout" in d:
        heat(axes[0], d["test_cm_readout"], f"{tag} — TEST readout")
    else:
        axes[0].axis("off"); axes[0].set_title("no readout CM\n(non-reward/grouped cell)")
    if "test_cm_linear" in d:
        heat(axes[1], d["test_cm_linear"], f"{tag} — TEST linear classifier")
    else:
        axes[1].axis("off")

    # per-class recall over training (readout if present, else linear)
    key = "cm_readout" if any("cm_readout" in t for t in traj) else "cm_linear"
    xs, series = recall_over_time(traj, key)
    ax = axes[2]
    if len(xs) >= 1 and series.size:
        cmap = plt.get_cmap("tab10")
        for c in range(series.shape[1]):
            ax.plot(xs, series[:, c], "-o", ms=3, color=cmap(c), label=str(c))
        ax.set_ylim(0, 1.02); ax.set_xlabel("checkpoint (batch+1)"); ax.set_ylabel("recall")
        ax.set_title(f"Per-class recall over training\n({key})")
        ax.grid(alpha=0.3); ax.legend(title="class", fontsize=7, ncol=2)
    else:
        ax.axis("off"); ax.set_title("no per-checkpoint CMs")

    fig.suptitle(f"Confusion analysis — {tag}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(outdir, "confusion.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
