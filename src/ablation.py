"""Sleep component ablation — Reviewer 3, Major Point 2.

R3.2: "The sleep protocol simultaneously suppresses external input, applies
synaptic renormalization, introduces intrinsic noise, and keeps STDP active.
Since these mechanisms are introduced together, the current experiments cannot
determine which component is responsible for the observed improvement. A
component-wise ablation separating weight decay, spontaneous noise, sleep-phase
STDP, and the full protocol is therefore needed."

Design: full 2^4 factorial over the four components, MNIST only, 5 seeds.

    downscale  power-law pull of each weight toward w_target
    noise      Gaussian membrane noise during sleep
    stdp       plasticity active during sleep (the "replay" component)
    suppress   sensory drive zeroed during sleep

A full factorial rather than leave-one-out because the paper's replay claim is
itself an interaction: noise is supposed to matter *because* STDP is active to
consolidate what it reactivates. Leave-one-out cannot see that; 2^4 can.

Two things to know when reading the results:

1. With `suppress` OFF, the last presented frame is *held* for the duration of
   the window rather than fresh input streaming in — real time is frozen during
   sleep, so there is no new data to consume. The contrast is therefore
   "no sensory drive" vs "a static frame repeated", not "sleep" vs "wake".

2. With both `downscale` and `stdp` off, weights are untouched for the whole
   window, so that cell is close to the no-sleep reference — it differs only in
   that membrane state evolves during the pause. A true no-sleep reference is
   included as a separate condition rather than inferred from it.

Sleep ratio comes from the sweep via experiment.resolve_sleep_ratio, so this
study, the sweep and the baseline comparison all use one value.

Usage
-----
  python ablation.py --list
  python ablation.py --task-id 7
  python ablation.py --all --skip-done
  python ablation.py --collect
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

from experiment import resolve_sleep_ratio   # single source for the ratio
from sweep import NUM_STEPS, CHECK_SLEEP_INTERVAL, lambda_for_ratio, \
    SLEEP_MAX_ITERS, SLEEP_ON_TIMEOUT, SLEEP_TERMINATION

COMPONENTS = ("downscale", "noise", "stdp", "suppress")
DATASETS = ["mnist"]
SEEDS = [42, 43, 44, 45, 46]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", "ablation")

RESOLVED_RATIO = None   # set in main()


def build_grid():
    """2^4 component combinations plus an explicit no-sleep reference."""
    grid, cid = [], 0
    for ds, seed in itertools.product(DATASETS, SEEDS):
        grid.append({"cell_id": cid, "code": "none", "active": (),
                     "dataset": ds, "seed": seed})
        cid += 1
    # Ordered most-components-first so a truncated run covers the full protocol
    # and the near-full conditions before the heavily ablated ones.
    masks = sorted(itertools.product([True, False], repeat=4),
                   key=lambda m: -sum(m))
    for mask, ds, seed in itertools.product(masks, DATASETS, SEEDS):
        active = tuple(c for c, on in zip(COMPONENTS, mask) if on)
        grid.append({
            "cell_id": cid,
            "code": "".join("1" if on else "0" for on in mask),
            "active": active,
            "dataset": ds,
            "seed": seed,
        })
        cid += 1
    return grid


def cell_tag(c):
    return f"abl_{c['cell_id']:03d}_{c['code']}_{c['dataset']}_s{c['seed']}"


def cell_path(c):
    return os.path.join(OUT_DIR, cell_tag(c) + ".json")


def build_command(c, extra=None):
    ratio = RESOLVED_RATIO[0]
    cmd = [
        sys.executable,
        os.path.join(HERE, "main.py"),
        "--dataset", c["dataset"],
        "--seed", str(c["seed"]),
        "--runs", "1",
        "--out-tag", cell_tag(c),
        "--num-steps", str(NUM_STEPS),
        "--check-sleep-interval", str(CHECK_SLEEP_INTERVAL),
        "--clip-always",
        "--no-plots",
    ]
    if c["code"] == "none":
        cmd += ["--reg-method", "none", "--sleep-rate", "0.0"]
    else:
        cmd += [
            "--reg-method", "sleep",
            "--sleep-rate", str(ratio),
            "--sleep-decay-rate", str(lambda_for_ratio(ratio)),
            "--sleep-max-iters", str(SLEEP_MAX_ITERS),
            "--on-timeout", SLEEP_ON_TIMEOUT,
            "--sleep-termination", SLEEP_TERMINATION,
            # "none" rather than an empty string: an empty value is easy to
            # lose through a shell or a log, and silently falling back to the
            # default (all four on) would corrupt that cell.
            "--sleep-components", ",".join(c["active"]) if c["active"] else "none",
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
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMBA_NUM_THREADS", env.get("SLURM_CPUS_PER_TASK", "1"))
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    elapsed = (datetime.now() - started).total_seconds()

    rec = dict(c)
    rec["active"] = list(c["active"])
    rec.update({
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "sleep_rate": 0.0 if c["code"] == "none" else RESOLVED_RATIO[0],
        "sleep_ratio_provenance": RESOLVED_RATIO[1],
        "num_steps": NUM_STEPS,
        "check_sleep_interval": CHECK_SLEEP_INTERVAL,
        "finished": datetime.now().isoformat(),
    })
    for comp in COMPONENTS:
        rec[comp] = comp in c["active"]
    src = os.path.join(REPO, "results", f"results_{cell_tag(c)}.json")
    try:
        raw = json.load(open(src))
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
    print(f"[{cell_tag(c)}] "
          f"{'ok' if proc.returncode == 0 else f'FAILED rc={proc.returncode}'} "
          f"in {elapsed/60:.1f} min  acc={rec.get('test_accuracy')}", flush=True)
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
            # See sweep.py: one truncated record must not abort the summary.
            corrupt.append((c["cell_id"], str(exc)))
            missing.append(c["cell_id"])
    if not rows:
        print("No finished cells in", OUT_DIR)
        return

    cols = (["cell_id", "code", "dataset", "seed"] + list(COMPONENTS) +
            ["test_accuracy", "train_accuracy", "val_accuracy",
             "final_test_phi", "sleep_rate", "elapsed_s", "returncode"])
    out = os.path.join(OUT_DIR, "ablation_summary.csv")
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: x["cell_id"]):
            f.write(",".join(
                "" if r.get(k) is None else str(r.get(k)) for k in cols) + "\n")
    print(f"Wrote {len(rows)}/{len(grid)} cells to {out}")

    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r.get("test_accuracy") is not None:
            agg[r["code"]].append(r["test_accuracy"])
    print(f"\n{'code':>6}  {'components active':32s} {'n':>3} {'mean acc':>9}")
    for code in sorted(agg, key=lambda k: (k == "none", k), reverse=True):
        v = agg[code]
        if code == "none":
            names = "(no sleep reference)"
        else:
            names = ", ".join(c for c, ch in zip(COMPONENTS, code) if ch == "1") or "(all off)"
        print(f"{code:>6}  {names:32s} {len(v):3d} {sum(v)/len(v):9.4f}")

    # Main effect of each component: mean over cells with it on minus off,
    # across the factorial only (the no-sleep reference is not part of it).
    fac = [r for r in rows if r["code"] != "none"
           and r.get("test_accuracy") is not None]
    if fac:
        print("\nmain effects (factorial cells only):")
        for comp in COMPONENTS:
            on = [r["test_accuracy"] for r in fac if r.get(comp)]
            off = [r["test_accuracy"] for r in fac if not r.get(comp)]
            if on and off:
                d = sum(on)/len(on) - sum(off)/len(off)
                print(f"  {comp:10s} on {sum(on)/len(on):.4f} "
                      f"off {sum(off)/len(off):.4f}  delta {d:+.4f}")
        print("\n(Descriptive only — fit the GLMM for inference, including "
              "the interactions this design exists to detect.)")
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
    global RESOLVED_RATIO
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--task-id", type=int)
    g.add_argument("--all", action="store_true")
    g.add_argument("--collect", action="store_true")
    ap.add_argument("--sleep-ratio", type=float, default=None,
                    help="override the sweep-derived sleep ratio")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()

    RESOLVED_RATIO = resolve_sleep_ratio(args.sleep_ratio)
    grid = build_grid()

    if args.list:
        print(f"{len(grid)} cells = (2^4 component combinations + 1 no-sleep "
              f"reference) x {len(DATASETS)} dataset x {len(SEEDS)} seeds")
        print(f"SLURM array range: 0-{len(grid) - 1}")
        print(f"sleep ratio {RESOLVED_RATIO[0]} "
              f"({RESOLVED_RATIO[1]['source']})\n")
        for c in grid:
            done = " [done]" if os.path.exists(cell_path(c)) else ""
            names = "(no sleep reference)" if c["code"] == "none" else (
                ", ".join(c["active"]) or "(all off)")
            print(f"  {c['cell_id']:3d}  {c['code']:>4}  {names:34s} "
                  f"seed={c['seed']}{done}")
        return 0

    if args.collect:
        collect()
        return 0

    print(f"Sleep ratio: {RESOLVED_RATIO[0]} ({RESOLVED_RATIO[1]['source']})")
    if RESOLVED_RATIO[1]["source"].startswith("FALLBACK"):
        print("  WARNING: no sweep results — run src/sweep.py first, or pass "
              "--sleep-ratio explicitly.")

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
