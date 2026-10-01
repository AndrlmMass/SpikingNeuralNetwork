"""Sleep-noise tuning sweep, run after the 2026-09-30 bug fixes.

The noise level (--sleep-noise-var, used as the standard deviation of the
per-step Gaussian membrane noise during sleep) was never tuned on a model with
working sleep dynamics: before the fix every network neuron was stuck "on"
within ~50 steps of each episode, so noise had nothing to do. This picks it on
the corrected model.

Design: full sleep protocol at the 10% ratio (with the same lambda the sweep
uses), all four datasets, noise sd in {0.5, 1, 2, 4, 8, 16}. Uses TUNING seeds
100-102, disjoint from the evaluation seeds 42-46, so choosing the level does
not overfit the reported results.

Measured spontaneous activity during sleep (MNIST, first episode, sustained
part): sd 2 -> 0.8, 4 -> 2.7, 8 -> 6.5, 12 -> 10.2 active neurons per step of
250, against ~0.2 during wake.

Usage
-----
  python noise_sweep.py --list
  python noise_sweep.py --task-id 7
  python noise_sweep.py --collect
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

from sweep import NUM_STEPS, CHECK_SLEEP_INTERVAL, lambda_for_ratio, \
    SLEEP_MAX_ITERS, SLEEP_ON_TIMEOUT, SLEEP_TERMINATION

NOISE_SDS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DATASETS = ["mnist", "fmnist", "kmnist", "notmnist"]
SEEDS = [100, 101, 102]          # tuning seeds, disjoint from 42-46
SLEEP_RATIO = 0.1

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", "noise_sweep")


def build_grid():
    return [{"cell_id": i, "noise": n, "dataset": d, "seed": s}
            for i, (d, s, n) in enumerate(itertools.product(DATASETS, SEEDS, NOISE_SDS))]


def cell_tag(c):
    return f"nz_{c['cell_id']:03d}_{c['dataset']}_n{c['noise']:g}_s{c['seed']}"


def cell_path(c):
    return os.path.join(OUT_DIR, cell_tag(c) + ".json")


def build_command(c):
    return [
        sys.executable, os.path.join(HERE, "main.py"),
        "--reg-method", "sleep", "--dataset", c["dataset"], "--seed", str(c["seed"]),
        "--runs", "1", "--out-tag", cell_tag(c), "--num-steps", str(NUM_STEPS),
        "--check-sleep-interval", str(CHECK_SLEEP_INTERVAL), "--clip-always",
        "--no-plots", "--sleep-rate", str(SLEEP_RATIO),
        "--sleep-decay-rate", str(lambda_for_ratio(SLEEP_RATIO)),
        "--sleep-max-iters", str(SLEEP_MAX_ITERS), "--on-timeout", SLEEP_ON_TIMEOUT,
        "--sleep-termination", SLEEP_TERMINATION,
        "--sleep-components", "downscale,noise,stdp,suppress",
        "--sleep-noise-var", str(c["noise"]),
    ]


def run_cell(c):
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = build_command(c)
    print(f"[{cell_tag(c)}] {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        env.setdefault(k, "1")
    env.setdefault("MPLBACKEND", "Agg")
    started = datetime.now()
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    rec = dict(c, returncode=proc.returncode,
               elapsed_s=(datetime.now() - started).total_seconds())
    try:
        raw = json.load(open(os.path.join(REPO, "results", f"results_{cell_tag(c)}.json")))
        runs = [r for by in raw["results_by_dataset"].values() for rs in by.values() for r in rs]
        rec["test_accuracy"] = runs[-1].get("test_accuracy")
    except Exception as exc:
        rec["parse_error"] = str(exc)
    with open(cell_path(c), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[{cell_tag(c)}] rc={proc.returncode} acc={rec.get('test_accuracy')}", flush=True)
    return proc.returncode


def collect():
    import pandas as pd
    rows = [json.load(open(cell_path(c))) for c in build_grid() if os.path.exists(cell_path(c))]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT_DIR, "noise_sweep_summary.csv"), index=False)
    print(f"{len(d)}/{len(build_grid())} cells")
    print(d.pivot_table(index="noise", columns="dataset", values="test_accuracy",
                        aggfunc="mean").round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--task-id", type=int)
    g.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    grid = build_grid()
    if args.list:
        for c in grid:
            print(c["cell_id"], cell_tag(c))
        print(len(grid), "cells")
    elif args.collect:
        collect()
    else:
        return run_cell(grid[args.task_id])


if __name__ == "__main__":
    sys.exit(main())
