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

# --- grid definition -------------------------------------------------------
# Five seeds, reused across every condition, matching the paper's design.
METHODS = ["none", "sleep", "decay", "norm_layer", "norm_neuron"]
DATASETS = ["mnist", "fmnist", "kmnist", "notmnist"]
SEEDS = [42, 43, 44, 45, 46]

# Sleep ratio for the "sleep" arm. 0.1 is the optimum reported in the paper;
# the other arms ignore it.
SLEEP_RATIO = 0.1

# Power-law exponent for the sleep operator. The published table gives 0.9997
# and main.py historically passed 0.99997; neither bounds weight growth well
# under window-limited stopping. Measured on MNIST at T=105k with clipping
# equalized across arms (total exc |w| relative to initialization):
#     lambda=0.9997 -> 5.08x    lambda=0.999 -> 3.04x    lambda=0.997 -> 2.32x
# 0.997 brings sleep into the same range as the conventional baselines
# (1.76-2.93x), which is what makes the comparison a test of mechanism rather
# than of how much shrinkage each method happens to apply.
SLEEP_DECAY_RATE = 0.997

# On-timeout behaviour. The convergence criterion is never met inside a sleep
# window, so this branch is the normal operating regime, not an edge case.
#
# "give_up" simply stops at the end of the window and keeps whatever the power
# law achieved -- the graded, per-weight approach toward the target that is the
# mechanism under study.
#
# "scale_to_target" instead applies one uniform factor target/current across the
# whole block. That is *identical* to layer-wise normalization, i.e. to the
# norm_layer baseline, so using it would partly compare the sleep arm against
# itself. We therefore use give_up with a strong enough exponent (see above)
# that the window does the downscaling on its own.
SLEEP_ON_TIMEOUT = "give_up"

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

# Regularization cadence, shared by sleep and both normalization arms so the
# comparison is controlled. Matches the value main.py already uses.
REG_INTERVAL = 35000

# Hard cap on virtual iterations per sleep episode. Must exceed
# round(REG_INTERVAL * SLEEP_RATIO) or it, not the sleep ratio, sets the
# realized sleep duration.
SLEEP_MAX_ITERS = 10000

# Sleep ends when total weight is at or below threshold (Eq. 6), one-sided.
# The historical 'band' criterion is a +/-0.1% window that is routinely
# overshot, so episodes never terminate early and always run the full window.
SLEEP_TERMINATION = "below_target"

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
        "--reg-interval",
        str(REG_INTERVAL),
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
    cmd += ["--sleep-rate", str(SLEEP_RATIO if cell["method"] == "sleep" else 0.0)]
    if cell["method"] == "sleep":
        # Explicit, because main.py's --on-timeout default (give_up) makes the
        # sleep phase anti-regularizing. See SLEEP_ON_TIMEOUT above.
        cmd += ["--on-timeout", SLEEP_ON_TIMEOUT]
        cmd += ["--sleep-decay-rate", str(SLEEP_DECAY_RATE)]
        # Passed explicitly so the realized sleep duration cannot be silently
        # clipped. The window is round(REG_INTERVAL * SLEEP_RATIO) iterations
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
    env.setdefault("NUMBA_NUM_THREADS", env.get("SLURM_CPUS_PER_TASK", "4"))
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
            "sleep_ratio": SLEEP_RATIO if cell["method"] == "sleep" else 0.0,
            "sleep_on_timeout": SLEEP_ON_TIMEOUT if cell["method"] == "sleep" else None,
            "decay_rate": DECAY_RATE if cell["method"] == "decay" else None,
            "reg_interval": REG_INTERVAL,
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
    for cell in grid:
        path = cell_path(cell)
        if not os.path.exists(path):
            missing.append(cell["cell_id"])
            continue
        with open(path) as f:
            rows.append(json.load(f))

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
        "reg_interval",
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
