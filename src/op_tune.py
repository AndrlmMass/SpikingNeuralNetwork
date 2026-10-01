"""Operating-point scan for the corrected model (post-2026-09-30 fixes).

With Hebbian STDP the network at the original parameters barely responds to
input (~0.2 spikes/step of 250, MNIST test accuracy ~0.13 without sleep); the
pre-fix model only worked because inverted STDP grew the weights 20-50x. This
scans input drive and learning rate WITHOUT any regularization, to find a
regime where the network responds and learns, before sleep is re-tuned on top.

Grid: input gain x initial input weight x LTD/LTP ratio (plus a frozen,
no-learning reference at each drive), MNIST, tuning
seed(s) only (default 100; evaluation seeds 42-46 are never used here).
Each record stores test accuracy and the start/end mean |w| from the progress
bar, so weight growth can be read off directly.

    python op_tune.py --list
    python op_tune.py --task-id 3
    python op_tune.py --collect
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys

GAINS = [1.0, 2.0, 4.0]
SE_WEIGHTS = [0.3, 0.6, 1.2]
# "frozen" = learning off (lr 0): the untrained-network reference at each drive.
LTDS = ["frozen", 1.0, 0.5, 0.3, 0.1]
LR = 0.0005
SEEDS = [int(s) for s in os.environ.get("OPT_SEEDS", "100").split(",")]
DATASET = os.environ.get("OPT_DATASET", "mnist")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", os.environ.get("OPT_OUT", "op_tune2"))


def build_grid():
    return [{"cell_id": i, "gain": g, "se": w, "ltd": l, "seed": s, "dataset": DATASET}
            for i, (s, g, w, l) in enumerate(itertools.product(SEEDS, GAINS, SE_WEIGHTS, LTDS))]


def tag(c):
    return f"opt_{c['dataset']}_g{c['gain']:g}_se{c['se']:g}_ltd{c['ltd']}_s{c['seed']}"


def run_cell(c):
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    log = os.path.join(OUT_DIR, "logs", tag(c) + ".log")
    cmd = [sys.executable, os.path.join(HERE, "main.py"), "--reg-method", "none",
           "--dataset", c["dataset"], "--seed", str(c["seed"]), "--runs", "1",
           "--out-tag", tag(c), "--num-steps", "100", "--check-sleep-interval", "3500",
           "--clip-always", "--no-plots", "--sleep-rate", "0.0",
           "--n-train", os.environ.get("OPT_NTRAIN", "2000"), "--n-test", os.environ.get("OPT_NTEST", "500"),
           "--input-gain", str(c["gain"]), "--se-weights", str(c["se"]),
           ]
    if c["ltd"] == "frozen":
        cmd += ["--lr-exc", "0", "--lr-inh", "0"]
    else:
        cmd += ["--lr-exc", str(LR), "--lr-inh", str(LR), "--ltd-scale", str(c["ltd"])]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[k] = "1"
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    text = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    rec = dict(c, returncode=rc)
    m = re.findall(r"PCA\+LR accuracy: \{'train': ([0-9.]+), 'val': ([0-9.]+), 'test': ([0-9.]+)\}", text)
    if m:
        rec["train_acc"], rec["val_acc"], rec["test_acc"] = map(float, m[-1])
    w = re.findall(r"Training network:[^\n]*m_exc=([0-9.]+), m_inh=([0-9.]+)", text)
    if w:
        rec["m_exc_start"], rec["m_inh_start"] = map(float, w[0])
        rec["m_exc_end"], rec["m_inh_end"] = map(float, w[-1])
    with open(os.path.join(OUT_DIR, tag(c) + ".json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(tag(c), rc, rec.get("test_acc"), rec.get("m_exc_start"), rec.get("m_exc_end"), flush=True)
    return rc


def collect():
    import glob
    import pandas as pd
    d = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(OUT_DIR, "*.json"))])
    d["exc_growth"] = d.m_exc_end / d.m_exc_start
    d["inh_growth"] = d.m_inh_end / d.m_inh_start
    d.to_csv(os.path.join(OUT_DIR, "op_tune_summary.csv"), index=False)
    cols = ["dataset", "seed", "gain", "se", "ltd", "train_acc", "test_acc", "exc_growth", "inh_growth"]
    d["ltd"] = d["ltd"].astype(str)
    print(d.sort_values(["dataset", "gain", "se", "ltd", "seed"])[cols].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--task-id", type=int)
    g.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    grid = build_grid()
    if a.list:
        for c in grid:
            print(c["cell_id"], tag(c))
    elif a.collect:
        collect()
    else:
        return run_cell(grid[a.task_id])


if __name__ == "__main__":
    sys.exit(main())
