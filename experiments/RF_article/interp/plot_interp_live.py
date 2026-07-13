"""
Live progress plotter for an in-flight interp sweep. Unlike plot_interp.py (which
reads results.json, written only when a cell FINISHES), this parses the per-cell
run.log checkpoint lines so you can watch the trajectories mid-run.

  python experiments/RF_article/interp/plot_interp_live.py --results-dir <run_dir>
"""
import argparse, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["A0_rand_frozen_ff", "A1_rand_trace_ff", "A2_rand_trace_ee",
         "B0_ori_frozen_ff", "B1_ori_trace_ff", "B2_ori_trace_ee", "B3_ori_triplet_ee",
         "R1_ori_reward_ff"]
COLORS = {"A0_rand_frozen_ff": "#9ecae1", "A1_rand_trace_ff": "#3182bd", "A2_rand_trace_ee": "#08519c",
          "B0_ori_frozen_ff": "#a1d99b", "B1_ori_trace_ff": "#fd8d3c", "B2_ori_trace_ee": "#e6550d",
          "B3_ori_triplet_ee": "#9467bd", "R1_ori_reward_ff": "#d62728"}

# matches: [TAG] b  9 val 0.850 sel 0.128 coh 0.403 refit 0.807 fixed 0.350 EE/SE 0.000 [pool 0.7 dead 0.2 win_ent 0.6]
LINE = re.compile(
    r"\[(?P<tag>[\w]+)\]\s+b\s+(?P<batch>\d+)\s+val\s+(?P<val>[\d.]+)\s+sel\s+(?P<sel>[\d.]+)"
    r"\s+coh\s+(?P<coh>[\d.]+)\s+refit\s+(?P<refit>[\d.]+)\s+fixed\s+(?P<fixed>[\d.]+)"
    r"\s+EE/SE\s+(?P<ee>[\d.]+)"
    r"(?:\s+pool\s+(?P<pool>[\d.]+)\s+dead\s+(?P<dead>[\d.]+)\s+win_ent\s+(?P<went>[\d.]+))?")


def parse_logs(run_dir):
    out = {}
    for logp in glob.glob(os.path.join(run_dir, "*", "run.log")):
        traj = []
        with open(logp, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                m = LINE.search(ln)
                if not m:
                    continue
                g = m.groupdict()
                rec = {"batch": int(g["batch"]), "val_acc": float(g["val"]),
                       "selectivity": float(g["sel"]), "orient_coh": float(g["coh"]),
                       "refit_acc": float(g["refit"]), "fixed_acc": float(g["fixed"]),
                       "ee_se_ratio": float(g["ee"])}
                if g["pool"] is not None:
                    rec["pool_acc"] = float(g["pool"]); rec["dead_frac"] = float(g["dead"])
                traj.append(rec)
        if traj:
            out[os.path.basename(os.path.dirname(logp))] = traj
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    runs = parse_logs(args.results_dir)
    if not runs:
        print("no checkpoint lines found yet"); return

    panels = [("val_acc", "Val accuracy (fitted PCA+LR)"),
              ("pool_acc", "Pool-by-label readout (reward V1)"),
              ("selectivity", "Class selectivity (per-neuron)"),
              ("orient_coh", "RF orientation coherence"),
              ("__drift__", "Readout drift: refit (solid) vs fixed@init (dashed)"),
              ("dead_frac", "Dead-neuron fraction (reward)")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    for (key, title), ax in zip(panels, axes.ravel()):
        for tag in ORDER:
            traj = runs.get(tag)
            if not traj:
                continue
            xs = [t["batch"] + 1 for t in traj]
            c = COLORS.get(tag)
            if key == "__drift__":
                if len(xs) < 2:
                    continue
                ax.plot(xs, [t["refit_acc"] for t in traj], "-o", ms=3, color=c, label=tag)
                ax.plot(xs, [t["fixed_acc"] for t in traj], "--", color=c)
                continue
            ys = [t.get(key) for t in traj]
            if all(v is None for v in ys):
                continue
            if len(xs) == 1:
                ax.axhline(ys[0], ls="--", lw=1.4, color=c, label=f"{tag} (frozen)")
            else:
                ax.plot(xs, ys, "-o", ms=3, color=c, label=tag)
        ax.set_title(title); ax.set_xlabel("checkpoint (batch+1)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    done = sum(os.path.isfile(os.path.join(args.results_dir, t, "results.json")) for t in runs)
    fig.suptitle(f"Interp sweep — LIVE (from run.log)  |  {done}/{len(runs)} cells finished", fontsize=14)
    fig.tight_layout()
    out = os.path.join(args.results_dir, "mechanism_live.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
