"""
Frozen-SE vs plastic, five datasets -- reported PER DECODER, because the decoder is the trap.

The 08-25 diary entry records a naming trap that invalidated a whole pass of analysis:
this repo has FOUR decoders and `test_acc` is none of the interesting ones.

    test_acc                       the PCA+LR evaluator probe
    test_lin_acc                   an L1 logistic probe, fit on --probe-fit-all TRAIN features
    test_cm_readout                the POOLED / softmax readout
    risk_coverage.csv base_acc     the LEARNED dense readout -- the article's number
    (trajectory readout_learned_acc is the same decoder, measured on val)

Any frozen-vs-plastic comparison that does not name its decoder is meaningless, so this
script never prints a bare accuracy: every column is labelled with the decoder it came from.

One number here is a constant, not a measurement: the pooled readout on a FROZEN tiled
network is at chance by construction, because `group_tiled_centers` gives every class group
the same centre grid and the tiled path assigns orientations with period 4 into groups of
100, making all ten groups byte-identical at init (verified max|W_g0 - W_g9| = 0.0; measured
pooled accuracy 0.0933). It is reported so the degeneracy is visible, not hidden.

Usage:
    python experiments/RF_article/frozen_supervised/aggregate_frozen.py --run <run_dir>
    python experiments/RF_article/frozen_supervised/aggregate_frozen.py --run <run_dir> \
        --compare experiments/RF_article/interp/mnist_family_sweep/results/<phase2_run>
"""
import argparse, csv, json, math, os, re, sys
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_ORDER = ["mnist", "fmnist", "kmnist", "notmnist", "svhn"]


def learned_readout_acc(run_dir):
    """The LEARNED dense readout on the test set.

    Preferred source is risk_coverage.csv's base_acc, which plot_risk_coverage writes from
    the dumped features. Falls back to recomputing from uncertainty_features.npz, and only
    then to the final trajectory entry (which is a VAL number, so it is flagged).
    """
    rc = os.path.join(run_dir, "risk_coverage.csv")
    if os.path.exists(rc):
        try:
            with open(rc) as f:
                for row in csv.DictReader(f):
                    if row.get("readout") == "learned_readout" and row.get("base_acc"):
                        return float(row["base_acc"]), "risk_coverage.csv"
        except Exception:
            pass
    npz = os.path.join(run_dir, "uncertainty_features.npz")
    if os.path.exists(npz):
        try:
            d = np.load(npz, allow_pickle=False)
            if "score_test" in d.files:
                return float((d["score_test"].argmax(1) == d["y_test"]).mean()), "npz"
        except Exception:
            pass
    return None, None


def pooled_acc(res):
    """Accuracy implied by test_cm_readout (the pooled/softmax readout's confusion matrix)."""
    cm = res.get("test_cm_readout")
    if not cm:
        return None
    cm = np.asarray(cm, dtype=float)
    tot = cm.sum()
    return float(np.trace(cm) / tot) if tot else None


def load_cells(run):
    """{(dataset, seed): row} for every COMPLETE cell under `run`."""
    out = {}
    for name in sorted(os.listdir(run)):
        d = os.path.join(run, name)
        rj = os.path.join(d, "results.json")
        if not os.path.isdir(d) or not os.path.exists(rj):
            continue
        try:
            with open(rj) as f:
                res = json.load(f)
        except Exception:
            continue
        acc = res.get("test_acc")
        if not isinstance(acc, (int, float)) or not math.isfinite(acc):
            continue                       # partial write: config+trajectory only
        m = re.match(r"(?P<ds>[a-z0-9]+)_.*?s(?P<seed>\d+)$", name)
        ds = m.group("ds") if m else res.get("config", {}).get("dataset", name)
        seed = int(m.group("seed")) if m else -1
        learned, src = learned_readout_acc(d)
        traj = res.get("trajectory") or []
        cfg = res.get("config", {})
        # TWO SCHEMAS, and `test_acc` means different things in each. In the harness it is
        # the PCA+LR evaluator probe; in run_frozen.py it is the LEARNED READOUT. Reading it
        # as one number for both is the 08-25 naming trap again -- it silently produced a
        # "PCA+LR" column identical to the learned-readout column.
        cheap = cfg.get("rule") == "frozen_se_plastic_readout"
        if cheap:
            learned = learned if learned is not None else acc
            src = src or "results.json:test_acc (run_frozen)"
            pca = None                            # run_frozen fits no PCA+LR probe
            pooled = res.get("test_pooled_acc")
            coh_first = coh_last = None           # no trajectory; the freeze is asserted
            drift = res.get("frozen_rel_drift")
            groups_id = res.get("groups_identical_at_init")
        else:
            pca = acc
            pooled = pooled_acc(res)
            coh_first = traj[0].get("orient_coh") if traj else None
            coh_last = traj[-1].get("orient_coh") if traj else None
            drift = (abs(coh_last - coh_first)
                     if (coh_first is not None and coh_last is not None) else None)
            groups_id = None
        out[(ds, seed)] = dict(
            dataset=ds, seed=seed, dir=d, source=("run_frozen" if cheap else "harness"),
            learned_readout=learned, learned_src=src,
            linear_probe=res.get("test_lin_acc"),
            pca_lr=pca, pooled=pooled,
            orient_coh_first=coh_first, orient_coh_last=coh_last,
            frozen_rel_drift=drift, groups_identical=groups_id,
            n_checkpoints=len(traj),
        )
    return out


def summarize(cells):
    """mean/std across seeds, per dataset and decoder."""
    by_ds = defaultdict(list)
    for r in cells.values():
        by_ds[r["dataset"]].append(r)
    rows = []
    for ds in [d for d in DATASET_ORDER if d in by_ds] + \
              [d for d in sorted(by_ds) if d not in DATASET_ORDER]:
        rs = by_ds[ds]
        row = dict(dataset=ds, n_seeds=len(rs))
        for key in ("learned_readout", "linear_probe", "pca_lr", "pooled"):
            vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
            row[key] = float(np.mean(vals)) if vals else None
            row[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        coh = [r["orient_coh_first"] for r in rs if r["orient_coh_first"] is not None]
        row["orient_coh_init"] = float(np.mean(coh)) if coh else None
        # A frozen cell must not move. The harness route shows this as zero drift in
        # orientation coherence across checkpoints; run_frozen asserts it directly on W_se
        # and records the relative drift. Both land in one column.
        drift = [r["frozen_rel_drift"] for r in rs if r["frozen_rel_drift"] is not None]
        row["frozen_drift"] = float(np.max(drift)) if drift else None
        gid = [r["groups_identical"] for r in rs if r["groups_identical"] is not None]
        row["groups_identical"] = (all(gid) if gid else None)
        row["source"] = "+".join(sorted({r["source"] for r in rs}))
        rows.append(row)
    return rows


def fmt(v, nd=4):
    return "  --  " if v is None else f"{v:.{nd}f}"


def print_table(rows):
    print(f"\n{'dataset':<10} {'n':>2} {'learned readout':>18} {'L1 probe':>10} "
          f"{'pooled':>8} {'W_se drift':>11} {'groups==':>9} {'source':>11}")
    print("-" * 86)
    for r in rows:
        sd = r["learned_readout_sd"]
        lr = fmt(r["learned_readout"]) + (f" +-{sd:.4f}" if sd is not None else "        ")
        d = r["frozen_drift"]
        dtxt = ("%.1e" % d) if d is not None else "--"
        print(f"{r['dataset']:<10} {r['n_seeds']:>2} {lr:>18} "
              f"{fmt(r['linear_probe']):>10} {fmt(r['pooled']):>8} "
              f"{dtxt:>11} {str(r['groups_identical']):>9} {r['source']:>11}")
    print("\n  learned readout = the dense softmax-delta readout (the article's decoder)")
    print("  L1 probe        = logistic probe, fit on 5000 TRAIN features (phase-2 setting)")
    print("  pooled          = class-group pooling; AT CHANCE BY CONSTRUCTION when frozen,")
    print("                    since all ten groups are byte-identical at init")
    print("  W_se drift      = max relative change in the frozen weights; must be ~1e-10")
    print("                    (float rounding from Normalize's no-op rescale), never more")


def compare(frozen_rows, phase2_run):
    """Frozen minus plastic, learned readout against learned readout."""
    p2 = summarize(load_cells(phase2_run))
    by = {r["dataset"]: r for r in p2}
    print(f"\nFROZEN minus PLASTIC (learned readout, percentage points)")
    print(f"{'dataset':<10} {'frozen':>9} {'plastic':>9} {'delta pp':>9} {'n_frozen':>9} {'n_plastic':>10}")
    print("-" * 60)
    for r in frozen_rows:
        o = by.get(r["dataset"])
        if not o or r["learned_readout"] is None or o["learned_readout"] is None:
            continue
        d = 100.0 * (r["learned_readout"] - o["learned_readout"])
        print(f"{r['dataset']:<10} {r['learned_readout']:>9.4f} {o['learned_readout']:>9.4f} "
              f"{d:>+9.2f} {r['n_seeds']:>9} {o['n_seeds']:>10}")
    print("\n  Both columns are the SAME decoder. Phase 2 trained on 59000 images and this")
    print("  cell on 55000 (5000 went to the calibration set), which touches the readout's")
    print("  fit only -- state it alongside any delta.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/frozen_supervised/<run_id>")
    ap.add_argument("--compare", default=None,
                    help="a phase-2 sweep results dir, to difference against")
    ap.add_argument("--csv-out", default=None)
    a = ap.parse_args()

    cells = load_cells(a.run)
    if not cells:
        print(f"[aggregate] no COMPLETE cells under {a.run} yet "
              f"(a cell counts only once results.json carries a finite test_acc)")
        return
    rows = summarize(cells)
    print(f"[aggregate] {len(cells)} complete cells in {a.run}")
    print_table(rows)

    bad = [r for r in rows if r["frozen_drift"] and r["frozen_drift"] > 1e-6]
    if bad:
        print("\n  !! W_se MOVED in: " + ", ".join(r["dataset"] for r in bad) +
              " -- these cells are not frozen and must not be reported.")

    if a.compare:
        compare(rows, a.compare)

    out = a.csv_out or os.path.join(a.run, "frozen_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
