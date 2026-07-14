"""
Plot the mechanism-harness ladder: selectivity, orientation coherence, readout
drift, and recurrent/feedforward drive across cells, so the STDP-erodes-the-prior
mechanism is visible.

  python experiments/RF_article/interp/plot_interp.py --results-dir <run_dir>
"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["A0_rand_frozen_ff", "A1_rand_trace_ff", "A2_rand_trace_ee",
         "B0_ori_frozen_ff", "B1_ori_trace_ff", "B2_ori_trace_ee",
         "B3_ori_triplet_ee", "B3x_ori_triplet_ff",
         "R1_ori_reward_ff", "R1x_ori_reward_ee",
         "G1_grp_reward_ff_static", "G2_grp_reward_ff_vogels"]
COLORS = {"A0_rand_frozen_ff": "#9ecae1", "A1_rand_trace_ff": "#3182bd", "A2_rand_trace_ee": "#08519c",
          "B0_ori_frozen_ff": "#a1d99b", "B1_ori_trace_ff": "#fd8d3c", "B2_ori_trace_ee": "#e6550d",
          "B3_ori_triplet_ee": "#9467bd", "B3x_ori_triplet_ff": "#c5b0d5",
          "R1_ori_reward_ff": "#d62728", "R1x_ori_reward_ee": "#ff9896",
          "G1_grp_reward_ff_static": "#2ca02c", "G2_grp_reward_ff_vogels": "#17becf"}


def load(run_dir):
    out = {}
    for d in glob.glob(os.path.join(run_dir, "*")):
        rp = os.path.join(d, "results.json")
        if os.path.isfile(rp):
            data = json.load(open(rp)); out[data["config"]["tag"]] = data
    return out


def series(traj, key):
    xs = [t["batch"] + 1 for t in traj]
    ys = [t.get(key) for t in traj]
    return xs, ys


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    runs = load(args.results_dir)

    panels = [("val_acc", "Val accuracy (PCA+LR readout)"),
              ("pool_acc", "Pool-by-label readout (reward V1)"),
              ("selectivity", "Class selectivity (per-neuron)"),
              ("orient_coh", "RF orientation coherence"),
              ("ee_se_ratio", "EE/SE drive ratio"),
              ("dead_frac", "Dead-neuron fraction (reward)"),
              ("softmax_acc", "Softmax group-pool accuracy (grouped)"),
              ("ce_loss", "Cross-entropy loss (grouped)")]
    fig, axes = plt.subplots(4, 2, figsize=(15, 22))
    for (key, title), ax in zip(panels, axes.ravel()):
        for tag in ORDER:
            if tag not in runs:
                continue
            traj = runs[tag].get("trajectory", [])
            if not traj:
                continue
            xs, ys = series(traj, key)
            if all(v is None for v in ys):   # metric absent for this cell (e.g. pool_acc on non-reward)
                continue
            if len(xs) == 1:  # frozen -> horizontal line
                ax.axhline(ys[0], ls="--", lw=1.4, color=COLORS.get(tag), label=tag)
            else:
                ax.plot(xs, ys, "-o", ms=4, color=COLORS.get(tag), label=tag)
        ax.set_title(title); ax.set_xlabel("training batches"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Mechanism harness — does STDP build or erode the prior?", fontsize=14)
    fig.tight_layout()
    out = os.path.join(args.results_dir, "mechanism.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")

    # readout-drift panel (refit vs fixed) for the learning cells
    fig2, ax = plt.subplots(figsize=(9, 6))
    for tag in ORDER:
        traj = runs.get(tag, {}).get("trajectory", [])
        if len(traj) < 2:
            continue
        xs = [t["batch"] + 1 for t in traj]
        ax.plot(xs, [t["refit_acc"] for t in traj], "-o", ms=4, color=COLORS.get(tag), label=f"{tag} refit")
        ax.plot(xs, [t["fixed_acc"] for t in traj], "--", color=COLORS.get(tag), label=f"{tag} fixed@init")
    ax.set_title("Readout drift: refit vs frozen-at-init readout\n(gap grows = features drifting off the useful manifold)")
    ax.set_xlabel("training batches"); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    out2 = os.path.join(args.results_dir, "readout_drift.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight"); plt.close(fig2)
    print(f"saved -> {out2}")


if __name__ == "__main__":
    main()
