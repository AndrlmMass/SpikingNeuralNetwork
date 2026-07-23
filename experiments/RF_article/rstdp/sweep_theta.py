"""
Adaptive-threshold (theta) sweep: iterate back from Diehl's persistent homeostasis
toward our transient default, to find a regime that spreads the winners (revives dead
neurons, forces within-group diversity) WITHOUT destroying class selectivity.

Diehl's exact numbers (tau=1e7, delta=0.05) work mechanically in our supervised tiled
setup (win_ent 0.36->0.69) but are far too strong -- they crater accuracy (learned
0.80->0.38, ceiling ->0.39), because the supervised reward already breaks symmetry and
heavy rate-equalization fights it. So we hold tau=1e7 (persistent) and dial delta down.

  python experiments/RF_article/rstdp/sweep_theta.py --train-all 4000

delta = 0 recovers the plain run (no adaptive threshold beyond the LIF default path).
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")

# persistent theta (tau=1e7), sweep the increment from Diehl (0.05) down to off.
THETA_DELTAS = [0.05, 0.01, 0.005, 0.001, 0.0]
THETA_TAU = 1e7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-lr", type=float, default=5e-6)
    p.add_argument("--readout-lr", type=float, default=0.1)
    p.add_argument("--peak-ei", type=float, default=50.0)
    p.add_argument("--tau", type=float, default=THETA_TAU)
    p.add_argument("--train-all", type=int, default=4000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(delta, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, HARNESS, "--tag", f"theta_{delta:g}", "--prior", "isotropic",
           "--rule", "reward", "--tiled", "--seed", str(a.seed),
           "--reward-lr", str(a.reward_lr), "--readout-lr", str(a.readout_lr),
           "--peak-ei", str(a.peak_ei), "--dense-readout",
           "--theta-tau", str(a.tau), "--theta-delta", str(delta),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
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
    vals = [t.get(key, float("nan")) for t in tr[-2:]]
    return float(np.nanmean(vals)) if vals else float("nan")


def main():
    a = parse_args()
    run_id = a.run_id or f"theta_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_theta", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"theta-delta sweep (tau={a.tau:g}, persistent) -> {run_dir}")
    print(f"  deltas: {THETA_DELTAS}  train_all={a.train_all}\n")
    t0 = time.time()
    for d in THETA_DELTAS:
        t1 = time.time()
        status = run_one(d, os.path.join(run_dir, f"theta_{d:g}"), a)
        print(f"  delta={d:<8g} {status:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    rows = [(d, final(os.path.join(run_dir, f"theta_{d:g}"), "readout_learned_acc"),
             final(os.path.join(run_dir, f"theta_{d:g}"), "softmax_acc"),
             final(os.path.join(run_dir, f"theta_{d:g}"), "dead_frac"),
             final(os.path.join(run_dir, f"theta_{d:g}"), "winner_entropy"),
             final(os.path.join(run_dir, f"theta_{d:g}"), "refit_acc")) for d in THETA_DELTAS]
    print(f"\n{'delta':>8}{'learned':>9}{'uniform':>9}{'dead':>7}{'win_ent':>8}{'ceil':>7}")
    for d, le, un, de, we, fx in rows:
        print(f"{d:>8g}{le:>9.3f}{un:>9.3f}{de:>7.2f}{we:>8.3f}{fx:>7.3f}")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump([dict(delta=d, learned=le, uniform=un, dead=de, win_ent=we, ceil=fx)
                   for d, le, un, de, we, fx in rows], f, indent=2)

    xs = list(range(len(rows)))
    xl = [f"{r[0]:g}" for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(xs, [r[1] for r in rows], "-o", color="#9467bd", label="readout (learned)")
    ax1.plot(xs, [r[5] for r in rows], "--o", color="k", label="feature ceiling (refit LR)")
    ax1.set_xticks(xs); ax1.set_xticklabels(xl); ax1.set_xlabel("theta delta (0 = off)")
    ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 1.02); ax1.grid(alpha=0.3); ax1.legend()
    ax1.set_title("Accuracy vs theta increment")
    ax2.plot(xs, [r[3] for r in rows], "-o", color="#1f77b4", label="dead fraction")
    ax2.plot(xs, [r[4] for r in rows], "-o", color="#2ca02c", label="winner entropy")
    ax2.set_xticks(xs); ax2.set_xticklabels(xl); ax2.set_xlabel("theta delta (0 = off)")
    ax2.set_ylabel("fraction"); ax2.set_ylim(0, 1.0); ax2.grid(alpha=0.3); ax2.legend()
    ax2.set_title("WTA health vs theta increment")
    fig.suptitle(f"Adaptive-threshold homeostasis sweep (persistent, tau={a.tau:g})")
    out = os.path.join(run_dir, "theta_sweep.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nPlot -> {out}\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
