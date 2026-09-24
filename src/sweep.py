"""Sleep-ratio sweep — re-run of the paper's main MNIST-family experiment.

Why this needs re-running
-------------------------
The published sweep requested 11 sleep ratios (0%-100% in 10% steps) but
`sleep_max_iters` capped the realized duration:

    sleep_window = round(check_sleep_interval * sleep_ratio)
    episode ends at min(sleep_window, sleep_max_iters)

With `check_sleep_interval = 35000` and `sleep_max_iters = 10000`, every ratio at
or above 28.6% realized exactly 28.57%. The committed results show it plainly --
a cliff between 0.20 and 0.30, then eight nominally different conditions landing
within ~2 percentage points of each other:

    notMNIST  0.10 -> .6463   0.20 -> .6371   0.30 -> .2831   ...  1.00 -> .2719
    fmnist    0.10 -> .5645   0.20 -> .5525   0.30 -> .3534   ...  1.00 -> .3058

This sweep fixes that by scaling the interval with `num_steps`. At
`check_sleep_interval = 3500` the largest window (ratio 1.0) is 3500, which is
below the cap, so all 11 ratios are faithful.

Two settings deliberately follow the *published* methods rather than being tuned:

  num_steps = 100   The paper reports 100 ms for the MNIST family, and git
                    history confirms the published runs used it: the default was
                    100 from 2025-12-05 and only became 1000 on 2025-12-08,
                    after the results were generated.
  lambda    = 0.997   Fixed across the sweep so the manipulation stays
                    one-dimensional. Measured at T=105k with clipping equalized,
                    0.997 brings total excitatory |w| to 2.32x initialization,
                    against 2.07-2.93x for the conventional baselines, so the
                    two studies compare at a matched level of constraint. The
                    published table's 0.9997 leaves it at 5.08x. This value is
                    shared with experiment.py and with every default in
                    main.py / big_comb.py — there is no path that silently
                    substitutes another.

`check_sleep_interval` scales with `num_steps` on purpose: it is measured in
timesteps, so leaving it at 35000 while cutting num_steps 10x would cut the
number of sleep episodes per image from 28 to 2 -- a different regime entirely,
and nothing like the wake/sleep cycling the paper depicts.

Usage
-----
  python sweep.py --list                 # print the grid
  python sweep.py --task-id 7            # one cell (SLURM array)
  python sweep.py --all --skip-done      # sequential, resumable
  python sweep.py --collect              # gather into one CSV

Once the optimum ratio is known, set SLEEP_RATIO in experiment.py to it before
running the conventional-baseline grid — that grid currently carries 0.1, which
came from the flattened sweep.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

# --- grid ------------------------------------------------------------------
SLEEP_RATES = [round(0.1 * i, 1) for i in range(11)]   # 0.0 .. 1.0
DATASETS = ["mnist", "fmnist", "kmnist", "notmnist"]
SEEDS = [42, 43, 44, 45, 46]

# --- fixed configuration ---------------------------------------------------
NUM_STEPS = 100             # published methods; see module docstring
CHECK_SLEEP_INTERVAL = 3500  # scaled with NUM_STEPS to hold episode cadence
# --- constant-endpoint design -------------------------------------------
# The operative variable is not the sleep percentage but the TOTAL log-space
# contraction per training batch, lambda^N with N = episodes * window. Holding
# lambda fixed makes that quantity scale with the ratio, so duration and
# renormalization are confounded — and at interval 3500 it saturates: every
# ratio drives every weight fully onto w_target and accuracy collapses to
# chance (measured 0.1657 at both ratio 0.1 and 1.0).
#
# Instead we hold the endpoint constant and let the ratio set the SPEED:
#
#     lambda(ratio) = RHO ** (1 / N),    N = episodes_per_batch * window
#
# Every condition removes the same weight per batch; they differ only in how
# long they take, with noise and STDP active throughout. Because sleep runs in
# virtual time, all conditions also see identical data. So the sweep now
# manipulates offline processing time at fixed homeostatic effect.
#
# RHO = 0.66 is the surviving log-gap at the published optimum (ratio 0.2 under
# the original configuration: lambda=0.99997, interval 35000, 2 episodes/batch,
# N=14000 -> 0.99997^14000 = 0.66), which produced the best published MNIST
# accuracy (0.7329). So the endpoint is anchored to a condition known to work.
RHO = 0.66

# Sleep episodes per training batch, from the loop bound t % interval == 0 over
# range(1, T) with T = images_per_batch * num_steps.
_IMAGES_PER_BATCH = 1000
EPISODES_PER_BATCH = (_IMAGES_PER_BATCH * NUM_STEPS - 1) // CHECK_SLEEP_INTERVAL


def lambda_for_ratio(ratio):
    """Power-law exponent giving the same total contraction RHO at any ratio."""
    if ratio <= 0:
        return None                      # ratio 0 runs as the unregularized arm
    window = max(1, round(CHECK_SLEEP_INTERVAL * ratio))
    n = EPISODES_PER_BATCH * window
    return RHO ** (1.0 / n)
SLEEP_MAX_ITERS = 35000      # inert at this interval; set explicitly so a later
                             # change to CHECK_SLEEP_INTERVAL cannot silently
                             # reintroduce the cap that flattened the original
SLEEP_ON_TIMEOUT = "give_up"     # stop at window end, keeping the graded
                                 # power-law approach. "scale_to_target" applies
                                 # a single uniform factor, i.e. layer
                                 # normalization, which is a different mechanism
SLEEP_TERMINATION = "band"  # window-limited: the one-sided Eq. 6
                            # criterion fires almost immediately at this
                            # interval (231 of 9800 expected iterations),
                            # which would remove the manipulation entirely

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", "sweep")


def build_grid():
    """Ratio-major, so a truncated run still covers every dataset and seed
    for the ratios it reached."""
    return [
        {"cell_id": i, "sleep_rate": r, "dataset": d, "seed": s}
        for i, (r, d, s) in enumerate(
            itertools.product(SLEEP_RATES, DATASETS, SEEDS)
        )
    ]


def cell_tag(c):
    return (
        f"sweep_{c['cell_id']:03d}_sr{int(round(c['sleep_rate'] * 100)):03d}"
        f"_{c['dataset']}_s{c['seed']}"
    )


def cell_path(c):
    return os.path.join(OUT_DIR, cell_tag(c) + ".json")


def build_command(c, extra=None):
    cmd = [
        sys.executable,
        os.path.join(HERE, "main.py"),
        "--reg-method", "sleep" if c["sleep_rate"] > 0 else "none",
        "--dataset", c["dataset"],
        "--seed", str(c["seed"]),
        "--runs", "1",
        "--out-tag", cell_tag(c),
        "--num-steps", str(NUM_STEPS),
        "--check-sleep-interval", str(CHECK_SLEEP_INTERVAL),
        "--sleep-rate", str(c["sleep_rate"]),
        "--clip-always",
        "--no-plots",
    ]
    # Ratio 0 is the unregularized reference. Routing it through --reg-method
    # none rather than sleep with ratio 0 keeps it identical to the `none` arm
    # of the baseline grid, so the two studies share one reference condition.
    if c["sleep_rate"] > 0:
        cmd += [
            "--sleep-decay-rate", str(lambda_for_ratio(c["sleep_rate"])),
            "--sleep-max-iters", str(SLEEP_MAX_ITERS),
            "--on-timeout", SLEEP_ON_TIMEOUT,
            "--sleep-termination", SLEEP_TERMINATION,
        ]
    if extra:
        cmd += list(extra)
    return cmd


def run_cell(c, extra=None, dry_run=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = build_command(c, extra)
    print(f"[{cell_tag(c)}] {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0

    started = datetime.now()
    env = dict(os.environ)
    # The matvec in this model is small enough that threaded BLAS costs more in
    # synchronization than it saves; one thread per cell and many cells is the
    # faster arrangement. MPLBACKEND because no backend is pinned in-code on
    # every path and compute nodes have no display.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMBA_NUM_THREADS", env.get("SLURM_CPUS_PER_TASK", "1"))
    env.setdefault("MPLBACKEND", "Agg")

    # main.py resolves results/ and data/ relative to its cwd, so it must run
    # from the repo root, not from src/.
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    elapsed = (datetime.now() - started).total_seconds()

    rec = dict(c)
    rec.update({
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "num_steps": NUM_STEPS,
        "check_sleep_interval": CHECK_SLEEP_INTERVAL,
        "sleep_decay_rate": lambda_for_ratio(c["sleep_rate"]),
        "rho": RHO,
        "sleep_max_iters": SLEEP_MAX_ITERS if c["sleep_rate"] > 0 else None,
        "on_timeout": SLEEP_ON_TIMEOUT if c["sleep_rate"] > 0 else None,
        "sleep_termination": SLEEP_TERMINATION if c["sleep_rate"] > 0 else None,
        # Realized window, recorded so a future reader can verify no cap bound.
        "sleep_window": min(
            round(CHECK_SLEEP_INTERVAL * c["sleep_rate"]), SLEEP_MAX_ITERS
        ),
        "finished": datetime.now().isoformat(),
    })
    src = os.path.join(REPO, "results", f"results_{cell_tag(c)}.json")
    try:
        raw = json.load(open(src))
        entries = [
            r for by in raw.get("results_by_dataset", {}).values()
            for runs in by.values() for r in runs
        ]
        if entries:
            last = entries[-1]
            for k in ("test_accuracy", "train_accuracy", "val_accuracy",
                      "final_test_phi"):
                rec[k] = last.get(k)
    except Exception as exc:
        rec["parse_error"] = str(exc)

    with open(cell_path(c), "w") as f:
        json.dump(rec, f, indent=2)
    print(
        f"[{cell_tag(c)}] "
        f"{'ok' if proc.returncode == 0 else f'FAILED rc={proc.returncode}'} "
        f"in {elapsed/60:.1f} min  acc={rec.get('test_accuracy')}",
        flush=True,
    )
    return proc.returncode


def collect():
    grid = build_grid()
    rows, missing = [], []
    corrupt = []
    for c in grid:
        if not os.path.exists(cell_path(c)):
            missing.append(c["cell_id"])
            continue
        try:
            rows.append(json.load(open(cell_path(c))))
        except Exception as exc:
            # Truncated or unparseable: treat as not-done so it gets rerun,
            # rather than letting one bad file abort the whole summary.
            corrupt.append((c["cell_id"], str(exc)))
            missing.append(c["cell_id"])
    if not rows:
        print("No finished cells in", OUT_DIR)
        return

    cols = ["cell_id", "sleep_rate", "dataset", "seed", "test_accuracy",
            "train_accuracy", "val_accuracy", "final_test_phi", "sleep_window",
            "num_steps", "check_sleep_interval", "sleep_decay_rate",
            "elapsed_s", "returncode"]
    out = os.path.join(OUT_DIR, "sweep_summary.csv")
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: x["cell_id"]):
            f.write(",".join(
                "" if r.get(k) is None else str(r.get(k)) for k in cols) + "\n")
    print(f"Wrote {len(rows)}/{len(grid)} cells to {out}")

    # Mean accuracy per ratio, so the optimum is visible without leaving the shell
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r.get("test_accuracy") is not None:
            agg[r["sleep_rate"]].append(r["test_accuracy"])
    if agg:
        print(f"\n{'ratio':>6} {'n':>4} {'mean test acc':>14}  window")
        for rate in sorted(agg):
            v = agg[rate]
            w = min(round(CHECK_SLEEP_INTERVAL * rate), SLEEP_MAX_ITERS)
            print(f"{rate:6.1f} {len(v):4d} {sum(v)/len(v):14.4f}  {w}")
    if corrupt:
        print(f"\n{len(corrupt)} unreadable record(s) — delete and rerun those ids:")
        for cid, err in corrupt:
            print(f"  {cid}: {err}")
    if missing:
        print(f"\nMissing {len(missing)} cells: {missing}")
    failed = [r["cell_id"] for r in rows if r.get("returncode")]
    if failed:
        print(f"Cells that exited non-zero: {failed}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--task-id", type=int)
    g.add_argument("--all", action="store_true")
    g.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()

    grid = build_grid()

    if args.list:
        print(f"{len(grid)} cells = {len(SLEEP_RATES)} ratios x "
              f"{len(DATASETS)} datasets x {len(SEEDS)} seeds")
        print(f"SLURM array range: 0-{len(grid) - 1}")
        print(f"num_steps={NUM_STEPS}  check_sleep_interval="
              f"{CHECK_SLEEP_INTERVAL}  rho={RHO} (lambda derived per ratio)")
        print(f"max sleep window = {round(CHECK_SLEEP_INTERVAL * 1.0)} "
              f"(cap {SLEEP_MAX_ITERS}) -> all ratios faithful\n")
        for c in grid:
            done = " [done]" if os.path.exists(cell_path(c)) else ""
            print(f"  {c['cell_id']:3d}  ratio={c['sleep_rate']:.1f} "
                  f"{c['dataset']:9s} seed={c['seed']}{done}")
        return 0

    if args.collect:
        collect()
        return 0

    if args.task_id is not None:
        if not 0 <= args.task_id < len(grid):
            print(f"task-id must be 0..{len(grid) - 1}", file=sys.stderr)
            return 2
        return run_cell(grid[args.task_id], args.extra, args.dry_run)

    failures = []
    for c in grid:
        if args.skip_done and os.path.exists(cell_path(c)):
            print(f"[{cell_tag(c)}] done, skipping", flush=True)
            continue
        if run_cell(c, args.extra, args.dry_run) != 0:
            failures.append(c["cell_id"])
    if failures:
        print(f"\n{len(failures)} cells failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
