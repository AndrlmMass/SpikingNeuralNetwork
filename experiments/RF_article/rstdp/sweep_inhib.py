"""
Inhibitory-learning sweep for the tiled reward-STDP run.

Motivation: our earlier inhibition test was only Vogels ON vs OFF, which is a weak
comparison — static uniform inhibition ("inhibit every neighbour equally, peak -2")
may be suboptimal, but a single badly-tuned plastic setting isn't a fair rival. Here
we sweep the Vogels iSTDP *learning rate* (inhibitory plasticity strength) against the
static baseline, at the tuned excitatory config.

Vogels iSTDP is homeostatic: each I->E synapse adjusts to push its exc target toward a
rate setpoint rho_0. So the key question isn't just accuracy — it's whether plastic
inhibition REVIVES dead neurons (dead_frac down) and SPREADS the winners
(winner_entropy up), i.e. fixes the WTA monopolization that static inhibition leaves in
place. We report those alongside the readout accuracies.

  python experiments/RF_article/rstdp/sweep_inhib.py --train-all 6000

vogels_lr = 0 is the special "off" (static uniform inhibition) baseline.
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")

# 0.0 = static inhibition (Vogels off); >0 = plastic at that learning rate.
VOGELS_LRS = [0.0, 0.005, 0.02, 0.05, 0.1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-lr", type=float, default=5e-6)
    p.add_argument("--readout-lr", type=float, default=0.1)
    p.add_argument("--peak-ei", type=float, default=50.0)
    p.add_argument("--peak-ie", type=float, default=-2.0)
    p.add_argument("--rho0", type=float, default=0.1, help="Vogels target rate (fixed across the sweep)")
    p.add_argument("--sigma-se", type=float, default=0.0, help="RF size (0 = default 3.0; dead-neuron problem is worst here)")
    p.add_argument("--train-all", type=int, default=6000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(lr, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    tag = "vogels_off" if lr == 0.0 else f"vogels_{lr:g}"
    cmd = [sys.executable, HARNESS, "--tag", tag, "--prior", "isotropic",
           "--rule", "reward", "--tiled", "--seed", str(a.seed),
           "--reward-lr", str(a.reward_lr), "--readout-lr", str(a.readout_lr),
           "--peak-ei", str(a.peak_ei), "--peak-ie", str(a.peak_ie),
           "--sigma-se", str(a.sigma_se),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
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
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return float("nan")
    tr = json.load(open(rp)).get("trajectory", [])
    vals = [t.get(key, float("nan")) for t in tr[-2:]]  # mean of last 2 checkpoints
    return float(np.nanmean(vals)) if vals else float("nan")


def label(lr):
    return "off (static)" if lr == 0.0 else f"{lr:g}"


def main():
    a = parse_args()
    run_id = a.run_id or f"inhib_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_inhib", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Inhibitory-LR sweep -> {run_dir}")
    print(f"  tuned exc config: reward_lr={a.reward_lr:g} readout_lr={a.readout_lr:g} "
          f"peak_ei={a.peak_ei:g} sigma_se={a.sigma_se or 3.0:g}  train_all={a.train_all}")
    print(f"  Vogels lrs: {VOGELS_LRS} (rho0={a.rho0})\n")
    t0 = time.time()
    for lr in VOGELS_LRS:
        t1 = time.time()
        name = "vogels_off" if lr == 0.0 else f"vogels_{lr:g}"
        status = run_one(lr, os.path.join(run_dir, name), a)
        print(f"  vogels_lr={label(lr):<12} {status:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    rows = []
    for lr in VOGELS_LRS:
        name = "vogels_off" if lr == 0.0 else f"vogels_{lr:g}"
        out = os.path.join(run_dir, name)
        rows.append((lr, final(out, "readout_learned_acc"), final(out, "softmax_acc"),
                     final(out, "dead_frac"), final(out, "winner_entropy"),
                     final(out, "refit_acc")))
    print(f"\n{'vogels_lr':>12} {'learned':>9} {'uniform':>9} {'dead':>7} {'win_ent':>8} {'LRceil':>8}")
    for lr, le, un, de, we, fx in rows:
        print(f"{label(lr):>12} {le:>9.3f} {un:>9.3f} {de:>7.2f} {we:>8.3f} {fx:>8.3f}")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump([dict(vogels_lr=lr, learned=le, uniform=un, dead=de, win_ent=we, lr_ceil=fx)
                   for lr, le, un, de, we, fx in rows], f, indent=2)

    # two panels: accuracy vs lr, and health (dead / winner_entropy) vs lr
    xs = list(range(len(rows)))
    xl = [label(r[0]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(xs, [r[1] for r in rows], "-o", color="#9467bd", label="readout (learned)")
    ax1.plot(xs, [r[2] for r in rows], "-o", color="#d62728", label="readout (uniform pool)")
    ax1.plot(xs, [r[5] for r in rows], "--o", color="k", label="linear-classifier ceiling")
    ax1.set_xticks(xs); ax1.set_xticklabels(xl, rotation=20, ha="right")
    ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 1.02); ax1.grid(alpha=0.3)
    ax1.legend(); ax1.set_title("Accuracy vs inhibitory learning rate")
    ax2.plot(xs, [r[3] for r in rows], "-o", color="#1f77b4", label="dead fraction")
    ax2.plot(xs, [r[4] for r in rows], "-o", color="#2ca02c", label="winner entropy")
    ax2.set_xticks(xs); ax2.set_xticklabels(xl, rotation=20, ha="right")
    ax2.set_ylabel("fraction"); ax2.set_ylim(0, 1.0); ax2.grid(alpha=0.3)
    ax2.legend(); ax2.set_title("WTA health vs inhibitory learning rate")
    fig.suptitle("Plastic (Vogels) vs static inhibition", fontsize=13)
    out = os.path.join(run_dir, "inhib_sweep.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nPlots -> {out}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
