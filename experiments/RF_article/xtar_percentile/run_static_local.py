"""
Local static x_tar sweep — no HPC required.

Runs a grid of *static* x_tar thresholds (SE = input->exc, EE = exc->exc) as
parallel local subprocesses, each calling run_experiment.py, and drops every
cell into ONE timestamped run folder so aggregate_xtar.py / plot_xtar.py work
unchanged.

Why static only: the percentile sweep showed the chosen percentile barely moves
the result, and the two hand-run static points (SE1.0/EE0.8 -> val .62 vs
SE0.5/EE0.2 -> val .71) show a strong *lower-is-better* gradient. So we sweep the
low static range densely to find where learning stays stable instead of eroding.

The mean baseline is included as one extra cell for reference.

Usage
-----
# focused low-range grid, 1 seed, 6 parallel workers (default)
python experiments/RF_article/xtar_percentile/run_static_local.py

# custom grid / more seeds / heavier training
python experiments/RF_article/xtar_percentile/run_static_local.py \
    --se 0.1 0.2 0.35 0.5 0.8 --ee 0.05 0.1 0.2 0.35 0.5 \
    --seeds 0 1 2 --parallel 6 --train-all 10000

Results land in:
    results/xtar_static_local/<RUN_ID>/<tag>/results.json
Then:
    python experiments/RF_article/xtar_percentile/aggregate_xtar.py \
        --results-dir experiments/RF_article/xtar_percentile/results/xtar_static_local/<RUN_ID>
    python experiments/RF_article/xtar_percentile/plot_xtar.py \
        --results-dir experiments/RF_article/xtar_percentile/results/xtar_static_local/<RUN_ID>
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_EXP = os.path.join(HERE, "run_experiment.py")


def parse_args():
    p = argparse.ArgumentParser(description="Local parallel static x_tar sweep")
    p.add_argument(
        "--se", type=float, nargs="+",
        default=[0.1, 0.2, 0.35, 0.5, 0.8],
        help="SE (input->exc) static x_tar levels",
    )
    p.add_argument(
        "--ee", type=float, nargs="+",
        default=[0.05, 0.1, 0.2, 0.35, 0.5],
        help="EE (exc->exc) static x_tar levels",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds per cell")
    p.add_argument(
        "--parallel", type=int, default=6,
        help="Max concurrent training subprocesses (keep < CPU cores)",
    )
    p.add_argument("--with-mean", action="store_true", default=True,
                   help="Include the mean-estimator baseline as a reference cell")
    p.add_argument("--no-mean", dest="with_mean", action="store_false")
    # training controls forwarded to run_experiment.py (kept modest for local runs)
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-all", type=int, default=10000)
    p.add_argument("--train-batch", type=int, default=1000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-batch", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--test-all", type=int, default=2000)
    p.add_argument("--test-batch", type=int, default=2000)
    p.add_argument("--run-id", default=None, help="Override the run folder name")
    return p.parse_args()


def build_jobs(args):
    """Return list of (tag, cli_args) for every cell in the grid."""
    jobs = []
    for seed in args.seeds:
        for se in args.se:
            for ee in args.ee:
                tag = (
                    f"stat_se{int(round(se * 1000))}_ee{int(round(ee * 1000))}_s{seed}"
                )
                cli = [
                    "--x-tar-mode", "static",
                    "--x-tar-static-se", str(se),
                    "--x-tar-static-ee", str(ee),
                    "--seed", str(seed),
                ]
                jobs.append((tag, cli))
        if args.with_mean:
            jobs.append((f"mean_s{seed}", ["--x-tar-mode", "mean", "--seed", str(seed)]))
    return jobs


def common_cli(args):
    return [
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--train-all", str(args.train_all),
        "--train-batch", str(args.train_batch),
        "--val-all", str(args.val_all),
        "--val-batch", str(args.val_batch),
        "--val-every", str(args.val_every),
        "--test-all", str(args.test_all),
        "--test-batch", str(args.test_batch),
    ]


def run_one(tag, cli, common, run_dir):
    out_dir = os.path.join(run_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "results.json")
    if os.path.isfile(done):
        return tag, "skipped (results.json present)", 0.0
    log_path = os.path.join(out_dir, "run.log")
    cmd = [sys.executable, RUN_EXP, *cli, *common, "--output-dir", out_dir]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        # keep each worker single-threaded so N workers don't oversubscribe cores
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    dt_s = time.time() - t0
    status = "OK" if (proc.returncode == 0 and os.path.isfile(done)) else f"FAIL(rc={proc.returncode})"
    return tag, status, dt_s


def main():
    args = parse_args()
    run_id = args.run_id or f"run_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(HERE, "results", "xtar_static_local", run_id)
    os.makedirs(run_dir, exist_ok=True)

    jobs = build_jobs(args)
    common = common_cli(args)

    print(f"Local static x_tar sweep")
    print(f"  run folder : {run_dir}")
    print(f"  grid       : SE={args.se}  EE={args.ee}")
    print(f"  seeds      : {args.seeds}  | cells: {len(jobs)}  | parallel: {args.parallel}")
    print(f"  training   : {args.train_all} imgs/epoch x {args.epochs} epoch(s), "
          f"val every {args.val_every} batch(es)\n")

    t_start = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(run_one, tag, cli, common, run_dir): tag for tag, cli in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            tag, status, dt_s = fut.result()
            results.append((tag, status, dt_s))
            print(f"  [{i:>3}/{len(jobs)}] {tag:<26} {status:<14} {dt_s/60:5.1f} min")

    elapsed = (time.time() - t_start) / 60
    n_ok = sum(1 for _, s, _ in results if s == "OK" or s.startswith("skipped"))
    print(f"\nDone: {n_ok}/{len(jobs)} cells succeeded in {elapsed:.1f} min wall time.")
    print(f"\nAggregate + plot with:")
    print(f"  python experiments/RF_article/xtar_percentile/aggregate_xtar.py --results-dir {run_dir}")
    print(f"  python experiments/RF_article/xtar_percentile/plot_xtar.py --results-dir {run_dir}")


if __name__ == "__main__":
    main()
