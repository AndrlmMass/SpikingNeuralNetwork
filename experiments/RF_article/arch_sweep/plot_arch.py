"""
Plot val-accuracy and val-phi trajectories for every config in an arch_sweep run,
with the frozen baselines as horizontal reference lines, so the damage->recover->
plateau pattern can be compared across configurations.

  python experiments/RF_article/arch_sweep/plot_arch.py --results-dir <run_dir>
"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["frozen", "wta_frozen", "exc", "exc_vogels", "wta_exc", "wta_exc_noEE", "wta_exc_vogels"]
COLORS = {
    "frozen": "#444444", "wta_frozen": "#999999",
    "exc": "#1f77b4", "exc_vogels": "#2ca02c",
    "wta_exc": "#ff7f0e", "wta_exc_noEE": "#d62728", "wta_exc_vogels": "#9467bd",
}


def load(run_dir):
    out = {}
    for d in glob.glob(os.path.join(run_dir, "*")):
        rp = os.path.join(d, "results.json")
        if os.path.isfile(rp):
            data = json.load(open(rp))
            out[data["config"]["tag"]] = data
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    runs = load(args.results_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for tag in ORDER:
        if tag not in runs:
            continue
        d = runs[tag]; c = COLORS.get(tag, None)
        vh = d.get("val_history", [])
        frozen_like = d["config"]["exc"] is False and d["config"]["vogels"] is False
        if frozen_like:
            # single-point frozen reference -> horizontal line across both panels
            ax1.axhline(d["test_acc"] * 100, ls="--", lw=1.6, color=c,
                        label=f"{tag} ({d['test_acc']*100:.1f}%)")
            ax2.axhline(d["test_phi"], ls="--", lw=1.6, color=c, label=f"{tag}")
            continue
        xs = [v["batch"] * 1000 / 1000.0 for v in vh]  # in thousands of images = batch idx*1
        imgs = [(v["batch"] + 1) for v in vh]           # batches seen (x in k-images since batch=1000)
        acc = [v["val_acc"] * 100 for v in vh]
        phi = [v["val_phi"] for v in vh]
        ax1.plot(imgs, acc, "-o", color=c, label=f"{tag} (test {d['test_acc']*100:.1f}%)")
        ax2.plot(imgs, phi, "-o", color=c, label=f"{tag}")

    ax1.set_title("Validation accuracy over training")
    ax1.set_xlabel("training images (thousands)"); ax1.set_ylabel("val accuracy (%)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="lower right")
    ax2.set_title("Validation phi (eta^2 separability) over training")
    ax2.set_xlabel("training images (thousands)"); ax2.set_ylabel("val phi")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("Arch sweep on oriented RFs — accuracy & separability trajectories", fontsize=14)
    fig.tight_layout()
    out = os.path.join(args.results_dir, "trajectories.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
