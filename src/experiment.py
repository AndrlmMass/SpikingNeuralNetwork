"""Conventional-stabilization baseline experiment (Reviewer 3, Major Point 1).

Reviewer 3 objects that the sleep mechanism is only compared against an
unregularized STDP baseline that diverges. That shows regularization is
*necessary*, not that sleep is better than simpler alternatives. This script
runs the comparison against three conventional methods, holding the network,
data, seeds and regularization cadence fixed so the only thing that varies is
how weight magnitude is controlled:

  none         unregularized STDP (the divergent reference already in the paper)
  sleep        this paper's power-law decay toward an absolute target
  decay        continuous multiplicative shrinkage every timestep, no target
  norm_layer   instantaneous layer-wise rescale to the initial total |w|
  norm_neuron  synaptic scaling: same, but per postsynaptic neuron

The layer/neuron regime sweep against *napping* is covered by the companion
paper (Massey et al., IWAI 2026, Springer CCIS; arXiv). This script deliberately
holds the sleep arm at the fixed-target power law of the present paper so the
two studies do not overlap.

Usage
-----
  python experiment.py --list                  # print the grid, one line per cell
  python experiment.py --task-id 7             # run exactly cell 7 (for SLURM arrays)
  python experiment.py --all                   # run every cell sequentially (local)
  python experiment.py --collect               # gather finished cells into one CSV

Each cell writes results/baselines/cell_<id>_<method>_<dataset>_s<seed>.json, so
a partially finished grid can be resumed by rerunning only the missing ids.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

# Shared configuration lives in sweep.py. The baseline grid MUST match the sweep
# that supplied its sleep ratio — a different stimulus duration or episode
# cadence would make the carried-over optimum meaningless.
from sweep import (
    NUM_STEPS,
    CHECK_SLEEP_INTERVAL,
    lambda_for_ratio,
    SLEEP_MAX_ITERS,
    SLEEP_ON_TIMEOUT,
    SLEEP_TERMINATION,
)

# --- grid definition -------------------------------------------------------
# Five seeds, reused across every condition, matching the paper's design.
METHODS = ["none", "sleep", "decay", "norm_layer", "norm_neuron"]
DATASETS = ["mnist", "fmnist", "kmnist", "notmnist"]
SEEDS = [42, 43, 44, 45, 46]

# Sleep ratio for the "sleep" arm; the other arms ignore it.
#
# This is NOT hardcoded. It is read from the sleep-ratio sweep in
# results/sweep/, because the paper's reported 10% optimum came from a sweep in
# which every ratio at or above 28.6% was silently capped to 28.57% — so the
# true optimum was never measured. resolve_sleep_ratio() below picks the
# accuracy-maximising ratio, excluding 0 (which is the unregularized reference,
# not a sleep condition).
#
# Used only if the sweep has not been run, and warned about loudly:
SLEEP_RATIO_FALLBACK = 0.1

SWEEP_DIR = None  # set after OUT_DIR is defined


def resolve_sleep_ratio(override=None):
    """Return (ratio, provenance) for the sleep arm.

    Pools test accuracy across datasets and seeds per ratio and takes the
    argmax over ratios > 0. Also reports the per-dataset optima, since a
    single global ratio is only defensible if the datasets roughly agree.
    """
    if override is not None:
        return float(override), {"source": "--sleep-ratio override"}

    import collections
    sweep_dir = os.path.join(REPO, "results", "sweep")
    pooled = collections.defaultdict(list)
    per_ds = collections.defaultdict(lambda: collections.defaultdict(list))
    n_cells = 0
    if os.path.isdir(sweep_dir):
        for fn in os.listdir(sweep_dir):
            if not fn.endswith(".json"):
                continue
            try:
                r = json.load(open(os.path.join(sweep_dir, fn)))
            except Exception:
                continue
            acc, rate = r.get("test_accuracy"), r.get("sleep_rate")
            if acc is None or rate is None:
                continue
            n_cells += 1
            pooled[float(rate)].append(acc)
            per_ds[r.get("dataset", "?")][float(rate)].append(acc)

    # Exclude 0: it is the reference condition, not a sleep duration.
    candidates = {k: v for k, v in pooled.items() if k > 0}
    if not candidates:
        return SLEEP_RATIO_FALLBACK, {
            "source": "FALLBACK — no sweep results found",
            "sweep_dir": sweep_dir,
            "n_cells": n_cells,
        }

    means = {k: sum(v) / len(v) for k, v in candidates.items()}
    best = max(means, key=means.get)
    ds_opt = {
        d: max((k for k in m if k > 0), key=lambda k: sum(m[k]) / len(m[k]))
        for d, m in per_ds.items()
        if any(k > 0 for k in m)
    }
    return best, {
        "source": "sweep",
        "n_cells": n_cells,
        "pooled_mean_by_ratio": {k: round(means[k], 4) for k in sorted(means)},
        "per_dataset_optimum": ds_opt,
        "agree_across_datasets": len(set(ds_opt.values())) == 1,
    }



# Per-timestep decay rate for the "decay" arm, calibrated rather than guessed.
#
# Total excitatory |w| in the unregularized network grows exponentially, not
# linearly: STDP potentiation scales with activity, which scales with weight, so
# dW/dt = (c - lambda) W with c = 2.17e-5 per timestep measured on MNIST
# (doubling every ~32k timesteps, i.e. ~320 images at 100 ms). Continuous decay
# therefore has no attractor -- it is marginally stable only at lambda = c, and
# any mismatch compounds exponentially. This is a structural difference from
# sleep and normalization, which measure the current weight sum and correct
# toward a target (closed loop) where decay cannot (open loop).
#
# Measured sweep at T=70k on MNIST:
#     0.5c = 1.08e-5 -> 2.52x W0      1.2c = 2.60e-5 -> 0.92x W0
#     0.8c = 1.74e-5 -> 1.60x W0      1.5c = 3.25e-5 -> 0.67x W0
#     1.0c = 2.17e-5 -> 1.18x W0      2.0c = 4.34e-5 -> 0.40x W0
#
# The sleep arm settles at 1.80x W0 under the same conditions. Matching that
# weight scale -- so the comparison isolates *how* growth is constrained rather
# than *how much* -- interpolates to lambda ~= 1.6e-5. Holding W0 exactly
# instead would want ~2.4e-5.
DECAY_RATE = 1.6e-5

# Regularization cadence comes from CHECK_SLEEP_INTERVAL, so sleep and both
# normalization arms fire on the same schedule as in the sweep. Not passed as
# --reg-interval: train.py falls back to check_sleep_interval when that is
# unset, so one knob controls every arm and they cannot diverge.

# Hard cap on virtual iterations per sleep episode. Must exceed
# round(CHECK_SLEEP_INTERVAL * ratio) or it, not the sleep ratio, sets the
# realized sleep duration.


RESOLVED_RATIO = (SLEEP_RATIO_FALLBACK, {"source": "unresolved"})

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", "baselines")


def build_grid():
    """Return the full list of cells as dicts, in a stable order.

    Order is method-major so that a truncated run still covers every dataset
    and seed for the conditions it reached.
    """
    grid = []
    for cell_id, (method, dataset, seed) in enumerate(
        itertools.product(METHODS, DATASETS, SEEDS)
    ):
        grid.append(
            {
                "cell_id": cell_id,
                "method": method,
                "dataset": dataset,
                "seed": seed,
            }
        )
    return grid


def cell_tag(cell):
    return (
        f"cell_{cell['cell_id']:03d}_{cell['method']}"
        f"_{cell['dataset']}_s{cell['seed']}"
    )


def cell_path(cell):
    return os.path.join(OUT_DIR, cell_tag(cell) + ".json")


def build_command(cell, extra=None):
    """Translate one grid cell into a main.py invocation."""
    tag = cell_tag(cell)
    cmd = [
        sys.executable,
        os.path.join(HERE, "main.py"),
        "--reg-method",
        cell["method"],
        "--dataset",
        cell["dataset"],
        "--seed",
        str(cell["seed"]),
        "--runs",
        "1",
        "--out-tag",
        tag,
        "--num-steps",
        str(NUM_STEPS),
        "--check-sleep-interval",
        str(CHECK_SLEEP_INTERVAL),
        # Weight clipping is gated on the `sleep` flag in train.py, so without
        # this the sleep arm alone would be hard-bounded to [min,max] while the
        # decay and normalization arms got only sign clamping. Measured effect
        # on the unregularized reference: 23.4x W0 unclipped vs 8.5x clipped.
        "--clip-always",
        # t-SNE and figure writing are pure overhead for a batch cell, and the
        # node has no display.
        "--no-plots",
    ]
    # The sleep arm needs its ratio; every other arm must be pinned to 0 so a
    # stray sleep episode cannot contaminate a non-sleep condition.
    ratio, _prov = RESOLVED_RATIO
    cmd += ["--sleep-rate", str(ratio if cell["method"] == "sleep" else 0.0)]
    if cell["method"] == "sleep":
        # Explicit, because main.py's --on-timeout default (give_up) makes the
        # sleep phase anti-regularizing. See SLEEP_ON_TIMEOUT above.
        cmd += ["--on-timeout", SLEEP_ON_TIMEOUT]
        cmd += ["--sleep-decay-rate", str(lambda_for_ratio(ratio))]
        # Passed explicitly so the realized sleep duration cannot be silently
        # clipped. The window is round(CHECK_SLEEP_INTERVAL * ratio) iterations
        # and the loop breaks at min(window, sleep_max_iters); at ratio 0.1 that
        # is 3500, so this cap is inert here -- but leaving it implicit is how
        # the published sweep ended up with every ratio >= 30% realizing 28.57%.
        cmd += ["--sleep-max-iters", str(SLEEP_MAX_ITERS)]
        cmd += ["--sleep-termination", SLEEP_TERMINATION]
    if cell["method"] == "decay":
        cmd += ["--decay-rate", str(DECAY_RATE)]
    if extra:
        cmd += list(extra)
    return cmd


def run_cell(cell, extra=None, dry_run=False):
    """Run one cell as a subprocess and record the outcome."""
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = build_command(cell, extra)
    print(f"[{cell_tag(cell)}] {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0

    started = datetime.now()
    # BLAS threading is pinned here rather than left to the environment: the
    # matrix-vector products in this model are far too small to thread
    # profitably, and an unpinned OpenBLAS spawns one worker per core and
    # spends most of its time on thread synchronization.
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMBA_NUM_THREADS", env.get("SLURM_CPUS_PER_TASK", "1"))
    env.setdefault("MPLBACKEND", "Agg")

    proc = subprocess.run(cmd, cwd=REPO, env=env)
    elapsed = (datetime.now() - started).total_seconds()

    # main.py writes results/results_<tag>.json; lift the accuracies out of it
    # into a flat per-cell record that --collect can read without knowing
    # anything about main.py's nested format.
    src = os.path.join(REPO, "results", f"results_{cell_tag(cell)}.json")
    record = dict(cell)
    record.update(
        {
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "sleep_ratio": RESOLVED_RATIO[0] if cell["method"] == "sleep" else 0.0,
            "sleep_ratio_provenance": RESOLVED_RATIO[1],
            "sleep_on_timeout": SLEEP_ON_TIMEOUT if cell["method"] == "sleep" else None,
            "decay_rate": DECAY_RATE if cell["method"] == "decay" else None,
            "num_steps": NUM_STEPS,
            "check_sleep_interval": CHECK_SLEEP_INTERVAL,
            "finished": datetime.now().isoformat(),
            "raw_results": os.path.relpath(src, REPO),
        }
    )
    try:
        with open(src) as f:
            raw = json.load(f)
        entries = []
        for by_rate in raw.get("results_by_dataset", {}).values():
            for runs in by_rate.values():
                entries.extend(runs)
        if entries:
            last = entries[-1]
            record["test_accuracy"] = last.get("test_accuracy")
            record["train_accuracy"] = last.get("train_accuracy")
            record["val_accuracy"] = last.get("val_accuracy")
            record["final_test_phi"] = last.get("final_test_phi")
    except Exception as exc:
        record["parse_error"] = str(exc)

    with open(cell_path(cell), "w") as f:
        json.dump(record, f, indent=2)
    status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
    print(
        f"[{cell_tag(cell)}] {status} in {elapsed/60:.1f} min "
        f"acc={record.get('test_accuracy')}",
        flush=True,
    )
    return proc.returncode


def collect():
    """Merge every finished cell into one CSV for the GLMM fit."""
    grid = build_grid()
    rows, missing = [], []
    corrupt = []
    for cell in grid:
        path = cell_path(cell)
        if not os.path.exists(path):
            missing.append(cell["cell_id"])
            continue
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception as exc:
            # See sweep.py: one truncated record must not abort the summary.
            corrupt.append((cell["cell_id"], str(exc)))
            missing.append(cell["cell_id"])

    if not rows:
        print("No finished cells found in", OUT_DIR)
        return

    cols = [
        "cell_id",
        "method",
        "dataset",
        "seed",
        "test_accuracy",
        "train_accuracy",
        "val_accuracy",
        "final_test_phi",
        "sleep_ratio",
        "decay_rate",
        "num_steps",
        "check_sleep_interval",
        "elapsed_s",
        "returncode",
    ]
    out = os.path.join(OUT_DIR, "baselines_summary.csv")
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: x["cell_id"]):
            f.write(
                ",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n"
            )
    print(f"Wrote {len(rows)}/{len(grid)} cells to {out}")
    if corrupt:
        print(f"\n{len(corrupt)} unreadable record(s) — delete and rerun those ids:")
        for cid, err in corrupt:
            print(f"  {cid}: {err}")
    if missing:
        print(f"Missing {len(missing)} cells: {missing}")
        print("Rerun them with:  --task-id <id>")
    failed = [r["cell_id"] for r in rows if r.get("returncode")]
    if failed:
        print(f"Cells that exited non-zero: {failed}")


def calibrate_decay_rate():
    """Print guidance for choosing DECAY_RATE. Not run automatically.

    The continuous-decay arm is only a fair comparator if it reaches a
    *comparable* steady-state weight magnitude to sleep. Otherwise the
    comparison measures how much shrinkage each method applies rather than how
    it applies it, which is exactly the objection Reviewer 3 raised about the
    original no-sleep baseline.

    Procedure: run the decay arm on the validation split at several rates,
    record the steady-state total |w| in the excitatory block, and pick the rate
    whose plateau matches the level the sleep arm converges to. Report both the
    chosen rate and the matched plateau in the paper.
    """
    print(calibrate_decay_rate.__doc__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print the grid and exit")
    g.add_argument("--task-id", type=int, help="run a single cell by id (SLURM arrays)")
    g.add_argument("--all", action="store_true", help="run every cell sequentially")
    g.add_argument("--collect", action="store_true", help="merge cells into a CSV")
    g.add_argument(
        "--calibrate-help",
        action="store_true",
        help="explain how to choose the decay rate",
    )
    ap.add_argument(
        "--sleep-ratio",
        type=float,
        default=None,
        help="override the sweep-derived sleep ratio (for reproducibility)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="with --all, skip cells that already have a result file",
    )
    ap.add_argument(
        "extra",
        nargs="*",
        help="additional flags passed straight through to main.py",
    )
    args = ap.parse_args()

    global RESOLVED_RATIO
    RESOLVED_RATIO = resolve_sleep_ratio(args.sleep_ratio)
    ratio, prov = RESOLVED_RATIO
    print(f"Sleep ratio for the sleep arm: {ratio}  ({prov['source']})")
    if prov["source"].startswith("FALLBACK"):
        print("  WARNING: no sweep results in results/sweep/. The sleep arm will")
        print(f"  use the fallback {SLEEP_RATIO_FALLBACK}, which is the paper's")
        print("  reported optimum from the sweep that was capped at 28.57%. Run")
        print("  src/sweep.py first, or pass --sleep-ratio explicitly.")
    else:
        print(f"  from {prov['n_cells']} sweep cells; "
              f"pooled means {prov['pooled_mean_by_ratio']}")
        print(f"  per-dataset optima {prov['per_dataset_optimum']}"
              f"  (agree: {prov['agree_across_datasets']})")
        if not prov["agree_across_datasets"]:
            print("  NOTE: datasets disagree on the optimum; a single pooled "
                  "ratio is a simplification worth stating in the paper.")
    print()

    grid = build_grid()

    if args.list:
        print(f"{len(grid)} cells = {len(METHODS)} methods x "
              f"{len(DATASETS)} datasets x {len(SEEDS)} seeds")
        print(f"SLURM array range: 0-{len(grid) - 1}")
        for cell in grid:
            done = " [done]" if os.path.exists(cell_path(cell)) else ""
            print(
                f"  {cell['cell_id']:3d}  {cell['method']:12s} "
                f"{cell['dataset']:9s} seed={cell['seed']}{done}"
            )
        return 0

    if args.collect:
        collect()
        return 0

    if args.calibrate_help:
        calibrate_decay_rate()
        return 0

    if args.task_id is not None:
        if not 0 <= args.task_id < len(grid):
            print(f"task-id must be in 0..{len(grid) - 1}", file=sys.stderr)
            return 2
        return run_cell(grid[args.task_id], args.extra, args.dry_run)

    # --all
    failures = []
    for cell in grid:
        if args.skip_done and os.path.exists(cell_path(cell)):
            print(f"[{cell_tag(cell)}] already done, skipping", flush=True)
            continue
        if run_cell(cell, args.extra, args.dry_run) != 0:
            failures.append(cell["cell_id"])
    if failures:
        print(f"\n{len(failures)} cells failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
