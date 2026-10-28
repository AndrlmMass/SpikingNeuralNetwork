"""Build results/interp/model/model_data.csv, the input to every GLMM in results/interp/model.

One row per (phase, dataset, cond, seed, response): k correct out of n test images.

Phase 2 (supervised, R-Hebb) is rebuilt from the run folders below. Each response is read
from the run's `uncertainty` block, k = round(base_acc * n_test):
    learned  <- learned_readout   (the network's own decoder)
    probe    <- linear_probe      (external L1-LR probe on the same features)
    pool     <- pool              (class-pool vote)
cond is the arm: oriented (Base: oriented RFs under R-Hebb, the GLMM reference), random
(random connectivity under R-Hebb) or frozen (oriented RFs, R-Hebb off, readout only).

Phase 1 (unsupervised) rows are carried over unchanged from the existing file: their run,
phase1_ablation/results/run_20260819_112254, is only on the cluster.

Usage (from the repo root):
    python experiments/RF_article/interp/build_model_data.py
"""
import csv, json, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(REPO, "results", "interp", "json", "experiments", "RF_article", "interp",
                    "mnist_family_sweep", "results")
OUT = os.path.join(REPO, "results", "interp", "model", "model_data.csv")
DS = ["mnist", "fmnist", "kmnist", "notmnist"]
SEEDS = range(5)
# (run folder, arm, datasets). Random on MNIST/FMNIST/KMNIST is the 09-16 rerun with the
# grouped-inhibition fix (commit 9159d743); notMNIST random and every frozen run are the
# 10-27 runs, scored on the same 2500 notMNIST test images as the oriented arm (the 09-16
# notMNIST random run was scored on 2724).
SOURCES = [
    ("run_20260819_112250", "oriented", DS),
    ("run_20260916_105832", "random", ["mnist", "fmnist", "kmnist"]),
    ("run_20261027_111346", "random", ["notmnist"]),
    ("run_20261027_111346", "frozen", DS),
]
RESPONSES = {"learned": "learned_readout", "probe": "linear_probe", "pool": "pool"}
FIELDS = ["phase", "dataset", "cond", "seed", "prior", "rule", "response", "k", "n", "acc"]


def phase2_rows():
    rows = []
    for run, arm, datasets in SOURCES:
        for ds in datasets:
            for s in SEEDS:
                path = os.path.join(RUNS, run, f"{ds}_{arm}_s{s}", "results.json")
                with open(path) as f:
                    unc = {u["readout"]: u for u in json.load(f)["uncertainty"]}
                for resp, key in RESPONSES.items():
                    n = int(unc[key]["n_test"])
                    k = int(round(unc[key]["base_acc"] * n))
                    rows.append({"phase": 2, "dataset": ds, "cond": arm, "seed": s,
                                 "prior": "random" if arm == "random" else "oriented",
                                 "rule": "none" if arm == "frozen" else "reward",
                                 "response": resp, "k": k, "n": n, "acc": round(k / n, 6)})
    return rows


def main():
    with open(OUT, newline="") as f:
        phase1 = [r for r in csv.DictReader(f) if r["phase"] == "1"]
    rows = phase1 + phase2_rows()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}: {len(phase1)} phase-1 rows (carried over), "
          f"{len(rows) - len(phase1)} phase-2 rows")


if __name__ == "__main__":
    main()
