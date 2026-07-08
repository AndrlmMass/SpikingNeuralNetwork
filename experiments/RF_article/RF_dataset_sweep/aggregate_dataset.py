"""
Aggregate the dataset sweep (dataset x {trained, frozen} x seed) from the SLURM
.out files and render the deliverable heatmaps.

The per-cell results.json is not written reliably on the cluster, so the
authoritative source is each task's captured stdout (*.out). We parse the printed
config + summary lines with regex (same approach as RF_size_tuning /
RF_netw_size_sweep). The relevant lines each run prints are:

    Config — dataset: cifar10 | condition: trained | ... | seed: 0 | epochs: 1
    Test accuracy : 0.2128           (also "Test accuracy (PCA+LR): 0.2128")
    Test phi      : 0.0601
    Best val acc  : 0.2260

This scans every *.out, writes a tidy per-run table (Results_dataset.xlsx) and a
clean dataset_sweep.jsonl, reports coverage, and saves the comparison figure
(heatmap_dataset.pdf):

    accuracy : dataset x [trained, frozen]   +   delta = trained - frozen
    phi      : dataset x [trained, frozen]   +   delta = trained - frozen

The delta panels (diverging colormap, centered at 0) are the point of the experiment —
positive => training helps. NaN-masked so a failed cell shows as a gap rather than
crashing aggregation.

Usage
-----
    python experiments/RF_article/RF_dataset_sweep/aggregate_dataset.py \
        --results-dir results/HPC/dataset_sweep
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Fixed display order; extra datasets found are appended after these.
DATASET_ORDER = ["mnist", "kmnist", "fmnist", "fashionmnist", "cifar10", "svhn", "notmnist"]
CONDITIONS = ["trained", "frozen"]

# Config line uses "key: value" (the SLURM header uses "key=value", so ':' is safe).
RE_DS = re.compile(r"dataset:\s*(\w+)")
RE_COND = re.compile(r"condition:\s*(\w+)")
RE_SEED = re.compile(r"seed:\s*(\d+)")
# Summary lines (test accuracy may carry a "(PCA+LR)" parenthetical).
RE_ACC = re.compile(r"Test accuracy\s*(?:\([^)]*\))?\s*:\s*([\d.]+|nan)")
RE_PHI = re.compile(r"Test phi\s*:\s*([\d.]+|nan)")
RE_VAL = re.compile(r"Best val acc\s*:\s*([\d.]+|nan)")


def _parse_float(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


def parse_run(out_path: str):
    """Parse one SLURM .out file into a per-run row dict, or None if no config.

    Scans line by line: the Config line (which also carries 'weight_type') comes
    before the summary block, so identifiers are picked up first, then scores.
    Stops early once every field is filled.
    """
    dataset = condition = seed = None
    test_acc = test_phi = best_val = float("nan")
    got_acc = got_phi = got_val = False

    try:
        with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if dataset is None and "weight_type" in line:
                    m = RE_DS.search(line)
                    if m:
                        dataset = m.group(1)
                        mc = RE_COND.search(line)
                        ms = RE_SEED.search(line)
                        condition = mc.group(1) if mc else None
                        seed = int(ms.group(1)) if ms else None
                        continue
                if not got_acc:
                    m = RE_ACC.search(line)
                    if m:
                        test_acc = _parse_float(m.group(1))
                        got_acc = True
                        continue
                if not got_phi:
                    m = RE_PHI.search(line)
                    if m:
                        test_phi = _parse_float(m.group(1))
                        got_phi = True
                        continue
                if not got_val:
                    m = RE_VAL.search(line)
                    if m:
                        best_val = _parse_float(m.group(1))
                        got_val = True
                if dataset is not None and got_acc and got_phi and got_val:
                    break
    except OSError:
        return None

    if dataset is None:
        return None
    if condition is None:
        condition = "trained"
    return {
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "best_val_acc": best_val,
    }


def load_all(results_dir: str):
    """Glob <results-dir>/*.out and parse each. Report files with no usable config."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.out")))
    rows, missing = [], []
    for f in files:
        row = parse_run(f)
        if row is None:
            missing.append(os.path.basename(f) + " (no config / unreadable)")
            continue
        rows.append(row)
    print(f"  {len(rows)} runs parsed from {len(files)} .out file(s); "
          f"{len(missing)} unparseable")
    return rows, missing


def _dataset_order(values) -> list:
    present = list(dict.fromkeys(values))
    ordered = [d for d in DATASET_ORDER if d in present]
    ordered += [d for d in present if d not in DATASET_ORDER]
    return ordered


def _box_panel(df, ax, value_col, title, order):
    """Grouped box plot: dataset on x, trained vs frozen boxes, seed points overlaid."""
    conds = [c for c in CONDITIONS if c in df["condition"].unique()]
    palette = {"trained": "#2a78d6", "frozen": "#1baf7a"}
    sns.boxplot(
        data=df, x="dataset", y=value_col, hue="condition",
        order=order, hue_order=conds, palette=palette,
        width=0.6, fliersize=0, linewidth=1.2, ax=ax,
    )
    # darker points so the 3 seeds per box are visible
    sns.stripplot(
        data=df, x="dataset", y=value_col, hue="condition",
        order=order, hue_order=conds,
        palette={"trained": "#14375f", "frozen": "#0b5c41"},
        dodge=True, jitter=0.12, size=5, edgecolor="white", linewidth=0.5,
        ax=ax, legend=False,
    )
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("dataset", fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="", frameon=False, loc="upper right")


def create_boxplots(df: pd.DataFrame, out_dir: str) -> None:
    order = _dataset_order(df["dataset"].tolist())
    grid = df.copy()
    grid["test_acc_pct"] = grid["test_acc"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    _box_panel(grid, axes[0], "test_acc_pct", "Test accuracy (%)", order)
    _box_panel(grid, axes[1], "test_phi", "Test phi", order)
    fig.suptitle("Dataset sweep — trained vs frozen (canonical 1024/225 network)",
                 fontsize=16)
    out_path = os.path.join(out_dir, "boxplot_dataset.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate dataset sweep (trained vs frozen) into a table + heatmaps"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing the SLURM *.out files",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for the heatmap/table (default: results-dir)",
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else results_dir

    if not os.path.isdir(results_dir):
        sys.exit(f"Directory not found: {results_dir}")
    os.makedirs(out_dir, exist_ok=True)

    rows, missing = load_all(results_dir)
    if not rows:
        sys.exit("No parseable *.out files found — nothing to write.")

    order = _dataset_order([r["dataset"] for r in rows])
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=True)
    df = df.sort_values(["dataset", "condition", "seed"])

    # ---- tidy outputs --------------------------------------------------------
    xlsx_path = os.path.join(out_dir, "Results_dataset.xlsx")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    jsonl_path = os.path.join(out_dir, "dataset_sweep.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in df.to_dict(orient="records"):
            rec["dataset"] = str(rec["dataset"])
            rec["condition"] = str(rec["condition"])
            fh.write(json.dumps(rec) + "\n")
    n_ds = df["dataset"].nunique()
    print(f"\n  Written {len(df)} rows ({n_ds} datasets x {df['condition'].nunique()} conditions) -> {xlsx_path}")
    print(f"  Rebuilt authoritative jsonl -> {jsonl_path}")

    # ---- coverage: which (dataset, condition) cells are under-seeded ----------
    seeds_per_cell = df.groupby(["dataset", "condition"], observed=True).size()
    n_seeds = int(df["seed"].nunique())
    partial = seeds_per_cell[seeds_per_cell < n_seeds]
    n_cells = len(seeds_per_cell)
    print(f"\n  Coverage: {n_cells} (dataset, condition) cells present, target {n_seeds} seeds each.")
    if len(partial):
        print(f"  {len(partial)} cell(s) under-seeded (<{n_seeds} seeds):")
        for (ds, cond), k in partial.items():
            print(f"    {ds}/{cond}: {k} seed(s)")
    if missing:
        print(f"  {len(missing)} .out file(s) had no usable config:")
        for name in missing:
            print(f"    {name}")

    # ---- per-cell mean +/- sd, plus the trained-frozen delta -----------------
    summary = (
        df.groupby(["dataset", "condition"], observed=True)[["test_acc", "test_phi"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\n  Per-cell mean +/- sd over seeds:")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.to_string(index=False))

    acc_mean = df.pivot_table(index="dataset", columns="condition", values="test_acc",
                              aggfunc="mean", observed=True)
    if {"trained", "frozen"}.issubset(acc_mean.columns):
        delta = (acc_mean["trained"] - acc_mean["frozen"]).reindex(order)
        print("\n  Delta test accuracy (trained - frozen), mean over seeds:")
        for ds, v in delta.items():
            print(f"    {ds}: {v:+.4f}")

    create_boxplots(df, out_dir)


if __name__ == "__main__":
    main()
