"""
General sequential tuner for the tiled reward-STDP run: sweep one hyperparameter
(reward_lr = cluster/SE learning rate, or readout_lr = plastic readout lr),
holding the other fixed, on a short budget. Reports the cluster metric
(uniform-pool softmax_acc) and the end metric (readout_learned_acc), plus health
(dead_frac, w_floor), and plots both vs the swept value.

  # tune the readout lr (clusters at reward_lr=2e-5)
  python experiments/RF_article/rstdp/tune.py --sweep readout_lr --values 0.005 0.01 0.02 0.05 0.1

  # then tune the cluster lr at the chosen readout lr
  python experiments/RF_article/rstdp/tune.py --sweep reward_lr --values 5e-6 1e-5 2e-5 5e-5 1e-4 --readout-lr 0.02
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")
FLAG = {"reward_lr": "--reward-lr", "readout_lr": "--readout-lr"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", choices=["reward_lr", "readout_lr"], required=True)
    p.add_argument("--values", type=float, nargs="+", required=True)
    p.add_argument("--reward-lr", type=float, default=2e-5, help="fixed cluster lr (unless swept)")
    p.add_argument("--readout-lr", type=float, default=0.02, help="fixed readout lr (unless swept)")
    p.add_argument("--train-all", type=int, default=3000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(val, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    fixed = {"reward_lr": a.reward_lr, "readout_lr": a.readout_lr}
    fixed[a.sweep] = val  # the swept param overrides the fixed default
    cmd = [sys.executable, HARNESS, "--tag", f"{a.sweep}_{val:g}", "--prior", "isotropic",
           "--rule", "reward", "--tiled", "--seed", str(a.seed),
           "--reward-lr", str(fixed["reward_lr"]), "--readout-lr", str(fixed["readout_lr"]),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return "OK" if pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json")) else f"FAIL(rc={pr.returncode})"


def final(run_dir, val, a, key):
    rp = os.path.join(run_dir, f"{a.sweep}_{val:g}", "results.json")
    if not os.path.isfile(rp):
        return float("nan")
    tr = json.load(open(rp)).get("trajectory", [])
    # mean of last 2 checkpoints (reduce noise)
    vals = [t.get(key, float("nan")) for t in tr[-2:]]
    return float(np.nanmean(vals)) if vals else float("nan")


def main():
    a = parse_args()
    run_id = a.run_id or f"tune_{a.sweep}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_tune", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"tune {a.sweep} over {a.values} (fixed reward_lr={a.reward_lr}, readout_lr={a.readout_lr})")
    print(f"  train_all={a.train_all} -> {run_dir}\n")
    t0 = time.time()
    for v in a.values:
        t1 = time.time()
        status = run_one(v, os.path.join(run_dir, f"{a.sweep}_{v:g}"), a)
        print(f"  {a.sweep}={v:<9g} {status:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    rows = [(v, final(run_dir, v, a, "readout_learned_acc"), final(run_dir, v, a, "softmax_acc"),
             final(run_dir, v, a, "dead_frac"), final(run_dir, v, a, "w_floor_frac")) for v in a.values]
    print(f"\n{a.sweep:>12} {'learned':>9} {'uniform':>9} {'dead':>7} {'w_floor':>8}")
    for v, le, un, de, wf in rows:
        print(f"{v:>12g} {le:>9.3f} {un:>9.3f} {de:>7.2f} {wf:>8.3f}")
    best = max(rows, key=lambda r: (r[1] if np.isfinite(r[1]) else r[2]))
    print(f"\nBEST {a.sweep} = {best[0]:g}  (learned={best[1]:.3f}, uniform={best[2]:.3f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    vs = [r[0] for r in rows]
    ax.plot(vs, [r[1] for r in rows], "-o", color="#9467bd", label="readout (learned)")
    ax.plot(vs, [r[2] for r in rows], "-o", color="#d62728", label="readout (uniform pool)")
    ax.set_xscale("log"); ax.set_xlabel(a.sweep); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Tune {a.sweep} (fixed reward_lr={a.reward_lr:g}, readout_lr={a.readout_lr:g})")
    out = os.path.join(run_dir, f"tune_{a.sweep}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Done in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
