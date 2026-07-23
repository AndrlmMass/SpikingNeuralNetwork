"""
Inhibitory sweep for the tiled reward-STDP run: plasticity RATE x inhibitory WEIGHT.

Supersedes the earlier 1-D vogels_lr sweep, for two reasons.

1. That sweep ran against a BUG. `vogels_iSTDP` wrote into W_ie entries the
   architecture leaves at exactly zero — the cross-group block (a hitman suppresses
   only its own class group) and the spared self-diagonal. So more inhibitory
   plasticity dissolved intra-class WTA into all-to-all inhibition plus
   self-inhibition, monotonically in the learning rate. Its verdict ("plastic
   inhibition revives neurons but costs accuracy; don't invest more") measured
   structure dissolution, not inhibitory learning, and does not stand. Fixed by the
   structural mask in synapses.vogels_iSTDP.

2. It reported dead_frac / winner_entropy but NOT class_selectivity or
   rf_diversity, so the thing we actually care about was invisible. Re-reading the
   old runs, selectivity fell monotonically with vogels_lr (0.386 at off -> 0.349 at
   0.1) — the clearest signature of the bug, and it was never in the table.

Rate and weight are swept TOGETHER because they trade off: a large static peak_ie can
mask what plasticity is doing, and a well-tuned rate may only pay off at a weaker
baseline weight. The 1-D sweep could not see that interaction.

  python experiments/RF_article/rstdp/sweep_inhib.py --train-all 6000 --jobs 4

vogels_lr = 0 is the special "off" (static inhibition) baseline.
"""
import argparse, datetime as dt, itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")

# 0.0 = static inhibition (Vogels off); >0 = plastic at that learning rate.
VOGELS_LRS = [0.0, 0.005, 0.02, 0.05]
# I->E peak weight. -2.0 is the current default; weaker leaves room for plasticity to
# shape the profile, stronger is the over-inhibition regime.
PEAK_IES = [-1.0, -2.0, -4.0]

# What we report. selectivity / rf_diversity are the point of the sweep; the old
# version tracked only the health pair and so could not see the regression.
METRICS = [
    ("readout_learned_acc", "learned"),
    ("softmax_acc",         "uniform"),
    ("selectivity",         "sel"),
    ("rf_diversity",        "rf_div"),
    ("dead_frac",           "dead"),
    ("winner_entropy",      "win_ent"),
    ("cur_ie",              "ie_drive"),
    ("refit_acc",           "LRceil"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-lr", type=float, default=5e-6)
    p.add_argument("--readout-lr", type=float, default=0.1)
    p.add_argument("--dense-readout", action="store_true",
                   help="use the full N_exc x n_classes readout (signs free)")
    p.add_argument("--peak-ei", type=float, default=50.0)
    p.add_argument("--vogels-lrs", type=float, nargs="+", default=VOGELS_LRS)
    p.add_argument("--peak-ies", type=float, nargs="+", default=PEAK_IES)
    p.add_argument("--rho0", type=float, default=0.1,
                   help="Vogels target rate (fixed across the sweep)")
    p.add_argument("--sigma-se", type=float, default=0.0, help="RF size (0 = default 3.0)")
    p.add_argument("--train-all", type=int, default=6000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1,
                   help="cells to run concurrently; each cell is single-threaded, so "
                        "this is a near-linear speedup up to your core count")
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def cell_name(lr, pie):
    return ("vogels_off" if lr == 0.0 else f"vogels_{lr:g}") + f"__ie_{abs(pie):g}"


def label(lr):
    return "off" if lr == 0.0 else f"{lr:g}"


def is_complete(out):
    """True only if the cell RAN TO COMPLETION.

    The harness rewrites results.json at every checkpoint, so mere existence means
    "started", not "finished" — resuming on that would silently keep half-trained
    cells and report them as if they were done. `test_acc` is written once, in the
    final block, so it is the honest completion marker.
    """
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return False
    try:
        return "test_acc" in json.load(open(rp))
    except Exception:
        return False


def run_one(lr, pie, out, a):
    if is_complete(out):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, HARNESS, "--tag", cell_name(lr, pie), "--prior", "isotropic",
           "--rule", "reward", "--tiled", "--seed", str(a.seed),
           "--reward-lr", str(a.reward_lr), "--readout-lr", str(a.readout_lr),
           "--peak-ei", str(a.peak_ei), "--peak-ie", str(pie),
           "--sigma-se", str(a.sigma_se), "--no-plots",
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
    if a.dense_readout:
        cmd.append("--dense-readout")
    if lr > 0.0:
        cmd += ["--use-vogels", "--vogels-lr", str(lr), "--vogels-rho0", str(a.rho0)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1",
               KMP_DUPLICATE_LIB_OK="TRUE")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    ok = pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json"))
    return "OK" if ok else f"FAIL(rc={pr.returncode})"


def final(out, key):
    """Mean of the last 2 checkpoints (smooths single-checkpoint noise)."""
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return float("nan")
    tr = json.load(open(rp)).get("trajectory", [])
    vals = [t[key] for t in tr[-2:] if t.get(key) is not None]
    return float(np.nanmean(vals)) if vals else float("nan")


def delta(out, key):
    """End minus start — does the metric IMPROVE over training, or erode?

    The headline number for selectivity. The old sweep reported only the final
    value, where every cell looks alike; the trend is what separates them, and it
    is what exposed the masking bug in the first place.
    """
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return float("nan")
    tr = json.load(open(rp)).get("trajectory", [])
    v = [t[key] for t in tr if t.get(key) is not None]
    return float(v[-1] - v[0]) if len(v) >= 2 else float("nan")


def heatmap(ax, M, lrs, pies, title, cmap="viridis"):
    ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pies))); ax.set_xticklabels([f"{p:g}" for p in pies])
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([label(l) for l in lrs])
    ax.set_xlabel("peak_ie (I->E weight)"); ax.set_ylabel("vogels_lr")
    mid = np.nanmean(M) if np.isfinite(M).any() else 0.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < mid else "black")
    ax.set_title(title, fontsize=10)


def main():
    a = parse_args()
    run_id = a.run_id or f"inhib2d_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_inhib", run_id)
    os.makedirs(run_dir, exist_ok=True)
    cells = list(itertools.product(a.vogels_lrs, a.peak_ies))
    print(f"Inhibitory 2-D sweep (rate x weight) -> {run_dir}")
    print(f"  exc config: reward_lr={a.reward_lr:g} readout_lr={a.readout_lr:g} "
          f"dense={a.dense_readout} peak_ei={a.peak_ei:g} sigma_se={a.sigma_se or 3.0:g}")
    print(f"  vogels_lrs={a.vogels_lrs}  peak_ies={a.peak_ies}  rho0={a.rho0}")
    print(f"  {len(cells)} cells, train_all={a.train_all}, jobs={a.jobs}\n")
    t0 = time.time()

    def work(cell):
        lr, pie = cell
        t1 = time.time()
        st = run_one(lr, pie, os.path.join(run_dir, cell_name(lr, pie)), a)
        print(f"  lr={label(lr):<6} ie={pie:<6g} {st:<12} {(time.time()-t1)/60:5.1f} min",
              flush=True)

    if a.jobs > 1:
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(work, cells))
    else:
        for c in cells:
            work(c)

    print(f"\n{'vogels_lr':>10} {'peak_ie':>8}"
          + "".join(f" {n:>9}" for _, n in METRICS) + f" {'d_sel':>8}")
    rows = []
    for lr, pie in cells:
        out = os.path.join(run_dir, cell_name(lr, pie))
        rec = dict(vogels_lr=lr, peak_ie=pie)
        line = f"{label(lr):>10} {pie:>8g}"
        for key, name in METRICS:
            v = final(out, key); rec[name] = v
            line += f" {v:>9.3f}" if np.isfinite(v) else f" {'nan':>9}"
        d = delta(out, "selectivity"); rec["d_sel"] = d
        line += f" {d:>+8.3f}" if np.isfinite(d) else f" {'nan':>8}"
        print(line)
        rows.append(rec)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    grab = lambda k: np.array([[next(r[k] for r in rows
                                     if r["vogels_lr"] == lr and r["peak_ie"] == pie)
                                for pie in a.peak_ies] for lr in a.vogels_lrs], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    heatmap(axes[0, 0], grab("learned"), a.vogels_lrs, a.peak_ies, "readout accuracy (learned)")
    heatmap(axes[0, 1], grab("sel"), a.vogels_lrs, a.peak_ies, "class selectivity (final)")
    heatmap(axes[1, 0], grab("d_sel"), a.vogels_lrs, a.peak_ies,
            "selectivity TREND (end - start); >0 = builds structure", cmap="RdBu_r")
    heatmap(axes[1, 1], grab("dead"), a.vogels_lrs, a.peak_ies, "dead fraction", cmap="magma")
    fig.suptitle("Inhibitory plasticity rate x I->E weight (structural mask fixed)", fontsize=13)
    out = os.path.join(run_dir, "inhib_sweep_2d.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nPlots -> {out}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
