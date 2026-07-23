"""
Sweep one learning rate (reward/cluster OR readout) for the tiled reward-STDP net,
holding the other fixed. All runs use the plastic learned readout so we optimize the
metric we actually care about: readout_learned_acc.

Sequential (no local contention), resumable. Prints a table + overlays trajectories.

  # tune the readout LR (cluster/reward LR fixed):
  python experiments/RF_article/rstdp/tune_lr.py --param readout-lr --values 0.005 0.01 0.02 0.05 0.1 --reward-lr 2e-5
  # tune the cluster/reward LR (readout LR fixed at its best):
  python experiments/RF_article/rstdp/tune_lr.py --param reward-lr --values 5e-6 1e-5 2e-5 5e-5 1e-4 --readout-lr 0.02
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--param", required=True, choices=["reward-lr", "readout-lr"])
    p.add_argument("--values", type=float, nargs="+", required=True)
    p.add_argument("--reward-lr", type=float, default=2e-5, help="cluster/reward LR when not swept")
    p.add_argument("--readout-lr", type=float, default=0.02, help="readout LR when not swept")
    p.add_argument("--use-vogels", action="store_true")
    p.add_argument("--train-all", type=int, default=3000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(v, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    reward_lr = v if a.param == "reward-lr" else a.reward_lr
    readout_lr = v if a.param == "readout-lr" else a.readout_lr
    cmd = [sys.executable, HARNESS, "--tag", f"{a.param}_{v:g}", "--prior", "isotropic",
           "--rule", "reward", "--tiled", "--reward-lr", str(reward_lr),
           "--readout-lr", str(readout_lr), "--seed", str(a.seed),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
    if a.use_vogels:
        cmd.append("--use-vogels")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return "OK" if pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json")) else f"FAIL(rc={pr.returncode})"


def load(out):
    rp = os.path.join(out, "results.json")
    return json.load(open(rp)) if os.path.isfile(rp) else None


def main():
    a = parse_args()
    run_id = a.run_id or f"tune_{a.param}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_controls", run_id)
    os.makedirs(run_dir, exist_ok=True)
    fixed = f"reward-lr={a.reward_lr}" if a.param == "readout-lr" else f"readout-lr={a.readout_lr}"
    print(f"tuning {a.param} over {a.values}  ({fixed}, vogels={a.use_vogels}, train_all={a.train_all})")
    print(f"-> {run_dir}\n")
    t0 = time.time()
    for v in a.values:
        out = os.path.join(run_dir, f"{v:g}")
        t1 = time.time()
        status = run_one(v, out, a)
        print(f"  {a.param}={v:<8g} {status:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    rows = []
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(a.values):
        d = load(os.path.join(run_dir, f"{v:g}"))
        if not d:
            continue
        tr = d["trajectory"]
        xs = [t["batch"] + 1 for t in tr]
        learned = [t.get("readout_learned_acc", np.nan) for t in tr]
        ax.plot(xs, learned, "-o", ms=3, color=cmap(i / max(len(a.values) - 1, 1)), label=f"{v:g}")
        rows.append((v, learned[-1], tr[-1].get("softmax_acc", np.nan),
                     tr[-1].get("dead_frac", np.nan), d.get("test_acc", np.nan)))
    ax.set_xlabel("checkpoint (batch+1)"); ax.set_ylabel("learned-readout accuracy")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(title=a.param, fontsize=8)
    ax.set_title(f"Tuning {a.param} ({fixed})")
    out_png = os.path.join(run_dir, "tune.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"\n{a.param:>12} {'learned':>9} {'uniform':>9} {'dead':>7} {'test':>7}")
    for v, le, un, dd, te in rows:
        print(f"{v:>12g} {le:>9.3f} {un:>9.3f} {dd:>7.2f} {te:>7.3f}")
    if rows:
        best = max(rows, key=lambda r: (r[1] if np.isfinite(r[1]) else -1))
        print(f"\nBest {a.param} = {best[0]:g}  (learned readout {best[1]:.3f})")
    print(f"Done in {(time.time()-t0)/60:.1f} min -> {out_png}")


if __name__ == "__main__":
    main()
