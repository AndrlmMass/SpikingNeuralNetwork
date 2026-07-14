"""
Sequential reward-STDP learning-rate tuner. Runs the interp harness (rule=reward)
at several --reward-lr values on a SHORT training budget and prints a summary table
(final pool_acc / selectivity / dead_frac / w_floor_frac) so we can pick a sane lr
before committing to the full 15k-image sweep.

Runs strictly IN SEQUENCE (one process at a time) so it does not contend with any
other local run. Skips any lr whose results.json already exists (resumable).

Usage (after the current sweep finishes):
  python experiments/RF_article/rstdp/tune_reward_lr.py
  python experiments/RF_article/rstdp/tune_reward_lr.py --lrs 5e-6 1e-5 2e-5 5e-5 --train-all 3000
"""
import argparse, datetime as dt, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4])
    p.add_argument("--train-all", type=int, default=3000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--test-all", type=int, default=2000)
    p.add_argument("--prior", default="oriented", choices=["oriented", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(lr, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, HARNESS, "--tag", f"lr{lr:g}", "--prior", a.prior,
           "--rule", "reward", "--reward-lr", str(lr), "--seed", str(a.seed),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", str(a.val_every), "--test-all", str(a.test_all),
           "--output-dir", out]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    ok = pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json"))
    return "OK" if ok else f"FAIL(rc={pr.returncode})"


def summarize(lr, out):
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return None
    d = json.load(open(rp))
    traj = d.get("trajectory", [])
    last = traj[-1] if traj else {}
    return dict(lr=lr, test_acc=d.get("test_acc", float("nan")),
                pool_acc=last.get("pool_acc", float("nan")),
                selectivity=last.get("selectivity", float("nan")),
                dead_frac=last.get("dead_frac", float("nan")),
                w_floor=last.get("w_floor_frac", float("nan")))


def main():
    a = parse_args()
    run_id = a.run_id or f"tune_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_tune", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"reward-lr tuning (sequential) -> {run_dir}\n  lrs={a.lrs} train_all={a.train_all}\n")
    t0 = time.time()
    for lr in a.lrs:
        out = os.path.join(run_dir, f"lr_{lr:g}")
        t1 = time.time()
        status = run_one(lr, out, a)
        print(f"  lr={lr:<8g} {status:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    rows = [r for lr in a.lrs if (r := summarize(lr, os.path.join(run_dir, f"lr_{lr:g}")))]
    print(f"\n{'lr':>10} {'test_acc':>9} {'pool_acc':>9} {'selectiv':>9} {'dead':>7} {'w_floor':>8}")
    for r in rows:
        print(f"{r['lr']:>10g} {r['test_acc']:>9.3f} {r['pool_acc']:>9.3f} "
              f"{r['selectivity']:>9.3f} {r['dead_frac']:>7.2f} {r['w_floor']:>8.3f}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Pick the lr with high pool_acc/selectivity, "
          f"low dead_frac, and w_floor not saturating.")


if __name__ == "__main__":
    main()
