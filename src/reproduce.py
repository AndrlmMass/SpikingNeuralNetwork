"""Reproduce the paper's three studies, their GLMMs, tables and figures.

    python src/reproduce.py                        # everything (~240 CPU-hours)
    python src/reproduce.py --stages analysis      # GLMMs, tables, figures only (minutes)
    python src/reproduce.py --stages ablation,analysis --workers 8

Stages, in order (each reads what the previous one wrote):

  data       download the four datasets once    torchvision, deeplake -> data/
  sweep      sleep-duration sweep, 220 runs     src/sweep.py       -> results/sweep/
  baselines  regularization comparison, 100     src/experiment.py  -> results/baselines/
  ablation   component ablation, 85 + 255       src/ablation.py    -> results/ablation/,
                                                                      results/ablation_datasets/
  analysis   beta GLMMs (R) and figures         src/glmm/          -> results/glmm/, figures/

The per-run summaries the analysis reads (results/*/..._summary.csv) are in git,
so `--stages analysis` reproduces every table and figure from a fresh clone
without retraining.

The sleep ratio of the baselines and ablation (30%) is not set by hand: both
take the ratio with the highest pooled test accuracy in the sweep
(experiment.resolve_sleep_ratio). The sweep is nearly flat above 0%, so a re-run
can move that argmax; pass --sleep-ratio 0.3 to hold the paper's value. The
continuous-decay rate is fixed at
experiment.DECAY_RATE = 1e-5, selected beforehand on tuning seeds 100-101
(see REPRODUCE.md); evaluation runs use seeds 42-46.

Each run is a separate subprocess with one thread; runs that already have a
result file are skipped, so an interrupted stage resumes where it stopped. A
failed run is retried once. Logs go to results/reproduce/.
"""
import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC)
LOGS = os.path.join(REPO, "results", "reproduce")
STAGES = ["data", "sweep", "baselines", "ablation", "analysis"]

ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
           MPLBACKEND="Agg")

# (stage, study script, extra arguments, label)
GRIDS = [
    ("sweep", "sweep.py", [], "sweep"),
    ("baselines", "experiment.py", [], "baselines"),
    ("ablation", "ablation.py", [], "ablation_mnist"),
    ("ablation", "ablation.py",
     ["--datasets", "fmnist,kmnist,notmnist", "--out-dir", "results/ablation_datasets"],
     "ablation_other"),
]
R_SCRIPTS = ["fit_sweep.R", "fit_baselines.R", "fit_ablation_datasets.R",
             "ablation_figure_data.R"]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "reproduce.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, logname, cwd=REPO):
    with open(os.path.join(LOGS, logname), "w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=cwd, env=ENV, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def pending_cells(script, extra):
    """Cell ids from the study's --list that have no result file yet."""
    out = subprocess.run([sys.executable, os.path.join(SRC, script), "--list", *extra],
                         cwd=REPO, env=ENV, capture_output=True, text=True, check=True).stdout
    rows = [m for m in (re.match(r"^\s*(\d+)\s.*seed=", l) for l in out.splitlines()) if m]
    return [int(m.group(1)) for m in rows if "[done]" not in m.string], len(rows)


def run_grid(script, extra, label, workers):
    todo, total = pending_cells(script, extra)
    log(f"{label}: {total} runs, {total - len(todo)} already done, {len(todo)} to run "
        f"on {workers} workers")
    t0, failed, done = time.time(), [], 0

    def one(i):
        cmd = [sys.executable, os.path.join(SRC, script), "--task-id", str(i), *extra]
        rc = run(cmd, f"{label}_{i}.log")
        if rc != 0:
            rc = run(cmd, f"{label}_{i}_retry.log")
        return i, rc

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, rc in ex.map(one, todo):
            done += 1
            if rc != 0:
                failed.append(i)
            if done % 20 == 0 or done == len(todo):
                log(f"{label}: {done}/{len(todo)} finished, {len(failed)} failed, "
                    f"{(time.time() - t0) / 3600:.1f} h elapsed")
    run([sys.executable, os.path.join(SRC, script), "--collect", *extra], f"{label}_collect.log")
    if failed:
        log(f"{label}: FAILED after retry: {failed} (see results/reproduce/{label}_<id>_retry.log)")
    return failed


def prepare_data():
    """Download every dataset once, before the parallel stages: concurrent runs
    would otherwise race the same download and can corrupt the cache. Uses the
    locations src/get_data.py reads: data/torchvision for the torchvision sets,
    data/datasets/notmnist_dl (or $NOTMNIST_LOCAL) for notMNIST."""
    from torchvision import datasets
    root = os.path.join(REPO, "data", "torchvision")
    for cls in (datasets.MNIST, datasets.KMNIST, datasets.FashionMNIST):
        for train in (True, False):
            n = len(cls(root, train=train, download=True))
            log(f"data: {cls.__name__} {'train' if train else 'test'} n={n}")

    import deeplake
    local = os.environ.get("NOTMNIST_LOCAL", os.path.join(REPO, "data", "datasets", "notmnist_dl"))
    if not (os.path.isdir(local) and os.listdir(local)):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        deeplake.deepcopy("hub://activeloop/not-mnist-small", local, overwrite=True)
    log(f"data: notMNIST n={len(deeplake.load(local, read_only=True))} at {local}")


def analysis(rscript):
    for r in R_SCRIPTS:
        rc = run([rscript, os.path.join("src", "glmm", r)], f"analysis_{r}.log")
        log(f"analysis {r}: {'ok' if rc == 0 else f'FAILED (rc={rc})'}")
        if rc != 0:
            return False
    rc = run([sys.executable, os.path.join(SRC, "glmm", "plot_glmms_bw.py")],
             "analysis_plot_glmms_bw.log")
    log(f"analysis plot_glmms_bw.py: {'ok' if rc == 0 else f'FAILED (rc={rc})'}")
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)} (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel runs (default: CPU count - 1)")
    ap.add_argument("--rscript", default=os.environ.get("RSCRIPT") or shutil.which("Rscript"),
                    help="path to Rscript (default: $RSCRIPT, then PATH)")
    ap.add_argument("--sleep-ratio", type=float, default=None,
                    help="pin the sleep ratio of the baselines and ablation (the paper used "
                         "0.3) instead of re-deriving it from the sweep; the sweep is nearly "
                         "flat, so a re-run can move the argmax")
    args = ap.parse_args()
    if args.sleep_ratio is not None:
        for i, (stage, script, extra, label) in enumerate(GRIDS):
            if stage in ("baselines", "ablation"):
                GRIDS[i] = (stage, script, ["--sleep-ratio", str(args.sleep_ratio), *extra], label)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stage(s): {', '.join(sorted(unknown))}")
    if "analysis" in stages and not args.rscript:
        ap.error("Rscript not found; pass --rscript or set RSCRIPT")

    os.makedirs(LOGS, exist_ok=True)
    log(f"=== reproduce: stages {stages} ===")
    failed = []
    for stage in STAGES:
        if stage not in stages:
            continue
        if stage == "data":
            prepare_data()
            continue
        if stage == "analysis":
            if not analysis(args.rscript):
                failed.append("analysis")
            continue
        for st, script, extra, label in GRIDS:
            if st == stage:
                failed += [f"{label}:{i}" for i in run_grid(script, extra, label, args.workers)]
    log("=== done ===" if not failed else f"=== done with failures: {failed} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
