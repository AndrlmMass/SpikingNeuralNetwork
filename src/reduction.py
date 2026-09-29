"""Does the sleep phase reduce to one scalar, and does it beat normalization?

Two questions, one grid, four datasets.

1. REDUCTION. The power-law pull is exactly linear in u = log(w / w_target):
   u <- lambda*u, so u_n = lambda^n u_0 and n applications at lambda equal ONE
   application at lambda^n (checked against the production kernel in
   tests/test_oneshot_equivalence.py, error < 1e-12). The `oneshot` arm applies
   that single op and simulates no sleep dynamics at all. If it matches
   `downscale`, the entire sleep phase is one scalar and the window is compute
   spent for nothing.

2. PARITY. The published 4-dataset comparison used the FULL protocol, whose
   pooled deficit against norm_layer (+0.033, paired p=0.307) came entirely
   from notMNIST, where the runs are bimodal -- and the component ablation
   attributes that collapse to sleep-phase STDP. So the comparison that
   actually matters, downscaling-only sleep against layer normalization on all
   four datasets, has never been run. On MNIST alone it is a tie
   (0.8042 vs 0.8010, paired p=0.765).

Arms
----
    full        all four components: downscale, noise, stdp, suppress
    downscale   downscale only -- the only component the ablation credits
    oneshot     downscale only, window collapsed to a single op
    norm_layer  layer-wise weight normalization, same cadence
    none        unregularized reference

`none` and `norm_layer` are re-run rather than lifted from results/baselines so
the table is self-contained; they reproduce those rows exactly (verified: the
`none` cells agree to 6 decimal places across both studies).

Usage
-----
  python reduction.py --list
  python reduction.py --all --skip-done
  python reduction.py --task-id 7            # one cell, for an array job
  python reduction.py --arms oneshot,downscale --all
  python reduction.py --collect
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

from experiment import resolve_sleep_ratio
from sweep import (
    NUM_STEPS,
    CHECK_SLEEP_INTERVAL,
    lambda_for_ratio,
    SLEEP_MAX_ITERS,
    SLEEP_ON_TIMEOUT,
    SLEEP_TERMINATION,
)

DATASETS = ["mnist", "fmnist", "kmnist", "notmnist"]
SEEDS = [42, 43, 44, 45, 46]

# arm -> (reg_method, active sleep components, one-shot)
ARMS = {
    "full":       ("sleep", ("downscale", "noise", "stdp", "suppress"), False),
    "downscale":  ("sleep", ("downscale",),                            False),
    "oneshot":    ("sleep", ("downscale",),                            True),
    "norm_layer": ("norm_layer", (),                                   False),
    "none":       ("none", (),                                         False),
}
ARM_ORDER = ["oneshot", "downscale", "full", "norm_layer", "none"]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", "reduction")

RESOLVED_RATIO = None   # set in main()


def build_grid(arms):
    """Arm-minor within dataset-major, so a truncated run still covers every
    arm on at least one dataset rather than finishing one arm everywhere."""
    grid, cid = [], 0
    for ds, seed, arm in itertools.product(DATASETS, SEEDS, arms):
        grid.append({"cell_id": cid, "arm": arm, "dataset": ds, "seed": seed})
        cid += 1
    return grid


def cell_tag(c):
    return "red_{:03d}_{}_{}_s{}".format(
        c["cell_id"], c["arm"], c["dataset"], c["seed"]
    )


def cell_path(c):
    return os.path.join(OUT_DIR, cell_tag(c) + ".json")


def build_command(c, extra=None):
    reg_method, components, oneshot = ARMS[c["arm"]]
    is_sleep = reg_method == "sleep"
    ratio = RESOLVED_RATIO[0]

    cmd = [
        sys.executable,
        os.path.join(HERE, "main.py"),
        "--reg-method", reg_method,
        "--dataset", c["dataset"],
        "--seed", str(c["seed"]),
        "--runs", "1",
        "--out-tag", cell_tag(c),
        "--num-steps", str(NUM_STEPS),
        "--check-sleep-interval", str(CHECK_SLEEP_INTERVAL),
        # Clipping is gated on the sleep flag in train.py, so without this the
        # sleep arms would be hard-bounded while norm_layer got only sign
        # clamping -- the comparison would be confounded.
        "--clip-always",
        "--no-plots",
        # Pinned to 0 off the sleep arms so a stray episode cannot contaminate
        # a non-sleep condition.
        "--sleep-rate", str(ratio if is_sleep else 0.0),
    ]
    if is_sleep:
        cmd += [
            "--sleep-decay-rate", str(lambda_for_ratio(ratio)),
            "--sleep-max-iters", str(SLEEP_MAX_ITERS),
            "--on-timeout", SLEEP_ON_TIMEOUT,
            "--sleep-termination", SLEEP_TERMINATION,
            # "none" rather than an empty value: an empty argument is easy to
            # lose through a shell or a log, and silently falling back to the
            # default (all four on) would corrupt the cell.
            "--sleep-components", ",".join(components) if components else "none",
        ]
        if oneshot:
            cmd += ["--sleep-oneshot"]
    if extra:
        cmd += list(extra)
    return cmd


def run_cell(c, extra=None, dry_run=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = build_command(c, extra)
    print("[{}] {}".format(cell_tag(c), " ".join(cmd)), flush=True)
    if dry_run:
        return 0

    started = datetime.now()
    # These matvecs are far too small to thread profitably; an unpinned
    # OpenBLAS spawns a worker per core and burns the time on synchronization.
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMBA_NUM_THREADS", env.get("SLURM_CPUS_PER_TASK", "1"))
    env.setdefault("MPLBACKEND", "Agg")

    proc = subprocess.run(cmd, cwd=REPO, env=env)
    elapsed = (datetime.now() - started).total_seconds()

    reg_method, components, oneshot = ARMS[c["arm"]]
    rec = dict(c)
    rec.update({
        "reg_method": reg_method,
        "components": list(components),
        "oneshot": oneshot,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "sleep_ratio": RESOLVED_RATIO[0] if is_sleep_arm(c) else 0.0,
        "sleep_ratio_provenance": RESOLVED_RATIO[1],
        "sleep_decay_rate": (
            lambda_for_ratio(RESOLVED_RATIO[0]) if is_sleep_arm(c) else None
        ),
        "num_steps": NUM_STEPS,
        "check_sleep_interval": CHECK_SLEEP_INTERVAL,
        "finished": datetime.now().isoformat(),
    })
    src = os.path.join(REPO, "results", "results_{}.json".format(cell_tag(c)))
    rec["raw_results"] = os.path.relpath(src, REPO)
    try:
        with open(src) as f:
            raw = json.load(f)
        entries = [r for by in raw.get("results_by_dataset", {}).values()
                   for runs in by.values() for r in runs]
        if entries:
            for k in ("test_accuracy", "train_accuracy", "val_accuracy",
                      "final_test_phi"):
                rec[k] = entries[-1].get(k)
    except Exception as exc:
        rec["parse_error"] = str(exc)

    with open(cell_path(c), "w") as f:
        json.dump(rec, f, indent=2)
    return proc.returncode


def is_sleep_arm(c):
    return ARMS[c["arm"]][0] == "sleep"


FIELDS = ["cell_id", "arm", "dataset", "seed", "reg_method", "oneshot",
          "test_accuracy", "train_accuracy", "val_accuracy", "final_test_phi",
          "sleep_ratio", "sleep_decay_rate", "num_steps",
          "check_sleep_interval", "elapsed_s", "returncode"]


def collect():
    rows = []
    for name in sorted(os.listdir(OUT_DIR)):
        if name.startswith("red_") and name.endswith(".json"):
            with open(os.path.join(OUT_DIR, name)) as f:
                rows.append(json.load(f))
    rows.sort(key=lambda r: r.get("cell_id", 0))
    out = os.path.join(OUT_DIR, "reduction_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote {}  ({} cells)".format(out, len(rows)))

    missing = [r["cell_id"] for r in rows if r.get("test_accuracy") is None]
    if missing:
        print("WARNING: {} cells have no accuracy: {}".format(
            len(missing), missing))

    try:
        import pandas as pd
    except ImportError:
        return
    d = pd.DataFrame(rows)
    if d.empty or "test_accuracy" not in d:
        return
    order = [a for a in ARM_ORDER if a in set(d.arm)]
    print("\nmean test accuracy")
    print(d.pivot_table(index="arm", columns="dataset",
                        values="test_accuracy", aggfunc="mean")
           .reindex(order).round(4).to_string())
    print("\npooled")
    print(d.groupby("arm").test_accuracy.agg(["mean", "std", "count"])
           .reindex(order).round(4).to_string())
    print("\nmean wall-clock seconds per cell")
    print(d.groupby("arm").elapsed_s.mean().reindex(order).round(1).to_string())

    # The reduction claim and the parity claim, both paired on (dataset, seed).
    p = d.pivot_table(index=["dataset", "seed"], columns="arm",
                      values="test_accuracy")
    try:
        from scipy import stats
    except ImportError:
        return
    print("\npaired contrasts")
    for a, b in [("oneshot", "downscale"), ("oneshot", "full"),
                 ("downscale", "full"), ("downscale", "norm_layer"),
                 ("oneshot", "norm_layer")]:
        if a not in p or b not in p:
            continue
        diff = (p[a] - p[b]).dropna()
        if len(diff) < 2:
            continue
        t = stats.ttest_1samp(diff, 0.0)
        print("  {:10s} - {:10s}  {:+.4f} (sd {:.4f}, n={})  p={:.3f}".format(
            a, b, diff.mean(), diff.std(), len(diff), t.pvalue))


def main():
    global RESOLVED_RATIO
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true",
                    help="print the grid and exit")
    ap.add_argument("--all", action="store_true",
                    help="run every cell in order")
    ap.add_argument("--task-id", type=int, default=None, help="run one cell")
    ap.add_argument("--collect", action="store_true",
                    help="gather per-cell json into reduction_summary.csv")
    ap.add_argument("--arms", type=str, default=",".join(ARM_ORDER),
                    help="comma-separated subset of {}".format(ARM_ORDER))
    ap.add_argument("--skip-done", action="store_true",
                    help="skip cells whose json already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without running them")
    ap.add_argument("extra", nargs="*",
                    help="extra flags forwarded verbatim to main.py")
    args = ap.parse_args()

    if args.collect:
        collect()
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s) {}; choose from {}".format(
            sorted(unknown), ARM_ORDER))
    grid = build_grid(arms)

    if args.list:
        for c in grid:
            done = " [done]" if os.path.exists(cell_path(c)) else ""
            print("{:3d}  {:10s} {:9s} seed {}{}".format(
                c["cell_id"], c["arm"], c["dataset"], c["seed"], done))
        print("\n{} cells ({} arms x {} datasets x {} seeds)".format(
            len(grid), len(arms), len(DATASETS), len(SEEDS)))
        return

    RESOLVED_RATIO = resolve_sleep_ratio()
    print("sleep ratio {} (source: {})".format(
        RESOLVED_RATIO[0], RESOLVED_RATIO[1].get("source")), flush=True)

    if args.task_id is not None:
        c = next((x for x in grid if x["cell_id"] == args.task_id), None)
        if c is None:
            raise SystemExit("no cell with id {}".format(args.task_id))
        sys.exit(run_cell(c, args.extra, args.dry_run))

    if not args.all:
        raise SystemExit("pass one of --list, --all, --task-id, --collect")

    failures = []
    for c in grid:
        if args.skip_done and os.path.exists(cell_path(c)):
            print("[{}] skip (done)".format(cell_tag(c)), flush=True)
            continue
        if run_cell(c, args.extra, args.dry_run) != 0:
            failures.append(c["cell_id"])
    if not args.dry_run:
        collect()
    if failures:
        print("\n{} cell(s) failed: {}".format(len(failures), failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
