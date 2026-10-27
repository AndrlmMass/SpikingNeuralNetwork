"""
Put phase-2 runs into results/interp/model/model_data.csv, replacing the cells they redo.

The phase-2 GLMM (tables.R, contrasts.R) is referenced to RF + R-Hebb (`oriented`) and has
two single changes from it: `random` (init) and `frozen` (input plasticity off, readout
still learns). A run folder holds cells tagged <dataset>_<cond>_s<seed>, cond one of
oriented / random / frozen; each becomes one row per decoder with k = round(acc * n):

    learned   uncertainty[readout == "learned_readout"].base_acc
    probe     test_lin_acc, the L1 probe fit on 5000 train features
    pool      trace / sum of test_cm_readout, when present (not modelled; kept for tables)

NOT `test_acc`: it is the PCA+LR evaluator in the harness schema -- the 08-25 naming trap.

Two checks, both asserted from the run's own record rather than assumed:
  - frozen cells must show flat W_se metrics across the trajectory; plastic cells must not.
  - after the merge, every phase-2 row of a dataset must share one test size, so no
    contrast compares different test sets. Otherwise nothing is written.

Datasets absent from the phase-2 rows (SVHN, dropped from the article) are skipped.

Usage:
    python experiments/RF_article/ingest_phase2_runs.py <run_dir> [<run_dir> ...]
    python experiments/RF_article/ingest_phase2_runs.py <run_dir> --dry-run
"""
import argparse, collections, csv, glob, json, os, re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DATA = os.path.join(REPO, "results", "interp", "model", "model_data.csv")
FIELDS = ["phase", "dataset", "cond", "seed", "prior", "rule", "response", "k", "n", "acc"]
TAG = re.compile(r"^(?P<ds>[a-z]+)_(?P<cond>oriented|random|frozen)_s(?P<seed>\d+)$")
PRIOR = {"oriented": "oriented", "random": "random", "frozen": "oriented"}
RULE = {"oriented": "reward", "random": "reward", "frozen": "frozen"}

# Functions of W_se alone (interp_harness.py, per checkpoint). With the input weights frozen
# they cannot move beyond float rounding; under R-Hebb they move by whole percents
# (orient_coh 0.67 -> 0.50 on mnist_oriented_s0).
W_SE_METRICS = ("orient_coh", "w_floor_frac", "rf_var_x", "rf_var_y", "rf_cov_xy")
FREEZE_TOL = 1e-6   # relative, as run_frozen.drift_check


def w_se_change(path, d):
    """Largest relative change of any W_se metric across the run's checkpoints."""
    traj = d.get("trajectory", [])
    if len(traj) < 2:
        raise SystemExit(f"{path}: no trajectory to check the weights against")
    worst = 0.0
    for key in W_SE_METRICS:
        v = [t[key] for t in traj if isinstance(t.get(key), (int, float))]
        if len(v) < 2:
            raise SystemExit(f"{path}: trajectory lacks {key}")
        worst = max(worst, (max(v) - min(v)) / (abs(v[0]) or 1.0))
    return worst


def rows_for(path, ds, cond, seed):
    d = json.load(open(path))
    moved = w_se_change(path, d)
    if cond == "frozen" and moved > FREEZE_TOL:
        raise SystemExit(f"{path}: W_se moved by {moved:.2e} (relative) -- NOT frozen")
    if cond != "frozen" and moved <= FREEZE_TOL:
        raise SystemExit(f"{path}: W_se never moved -- this {cond} cell did not learn")
    u = next((x for x in d.get("uncertainty", []) if x.get("readout") == "learned_readout"),
             None)
    if u is None or d.get("test_lin_acc") is None:
        raise SystemExit(f"{path}: missing learned-readout or probe accuracy")
    n = int(u["n_test"])     # the harness config has no test_all
    accs = [("learned", u["base_acc"]), ("probe", d["test_lin_acc"])]
    if d.get("test_cm_readout"):
        cm = np.asarray(d["test_cm_readout"])
        accs.append(("pool", np.trace(cm) / cm.sum()))
    out = []
    for resp, acc in accs:
        k = int(round(float(acc) * n))
        out.append(dict(phase="2", dataset=ds, cond=cond, seed=str(seed), prior=PRIOR[cond],
                        rule=RULE[cond], response=resp, k=str(k), n=str(n),
                        acc=str(round(k / n, 6))))
    print(f"  {os.path.basename(os.path.dirname(path))}: n_test={n}, "
          f"W_se max rel change {moved:.1e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with open(MODEL_DATA) as f:
        old = list(csv.DictReader(f))
    in_model = {r["dataset"] for r in old if r["phase"] == "2"}

    new = []
    for rd in a.run_dirs:
        for p in sorted(glob.glob(os.path.join(rd, "*_s*", "results.json"))):
            m = TAG.match(os.path.basename(os.path.dirname(p)))
            if m is None:
                continue
            if m["ds"] not in in_model:
                print(f"  skip {m['ds']}: not in the phase-2 model")
                continue
            new += rows_for(p, m["ds"], m["cond"], int(m["seed"]))
    if not new:
        raise SystemExit(f"no usable cells under {a.run_dirs}")

    for r in new:
        print("  " + ",".join(r[k] for k in FIELDS))

    # a rerun replaces its cell (dataset, cond, seed) wholesale, every decoder included
    redo = {(r["dataset"], r["cond"], r["seed"]) for r in new}
    kept = [r for r in old if not (r["phase"] == "2"
                                   and (r["dataset"], r["cond"], r["seed"]) in redo)]
    merged = kept + new

    sizes = collections.defaultdict(set)
    for r in merged:
        if r["phase"] == "2":
            sizes[r["dataset"]].add((r["cond"], r["n"]))
    bad = {ds: v for ds, v in sizes.items() if len({n for _, n in v}) > 1}
    if bad:
        for ds, v in sorted(bad.items()):
            print(f"  {ds}: test sizes by arm {sorted(v)}")
        raise SystemExit("phase-2 arms would be scored on different test sizes; not writing")

    if a.dry_run:
        return
    with open(MODEL_DATA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)
    print(f"wrote {len(new)} rows ({len(redo)} cells) -> {MODEL_DATA}")


if __name__ == "__main__":
    main()
