"""
Aggregate the dataset sweep (dataset x {trained, frozen} x seed) from per-cell
results.json files and render the deliverable heatmaps.

Each cell (run_experiment.py) writes results/<RUN_ID>/<tag>/results.json with:
  * config            — dataset, condition (trained/frozen), seed, counts, ...
  * test_acc/test_phi — final test scores
  * best_val_acc      — best validation accuracy

This scans every <tag>/results.json (authoritative — independent of the best-effort
live dataset_sweep.jsonl), writes a tidy per-run table (Results_dataset.xlsx) and a clean
dataset_sweep.jsonl, reports coverage, and saves the comparison figure
(heatmap_dataset.pdf):

    accuracy : dataset x [trained, frozen]   +   delta = trained - frozen
    phi      : dataset x [trained, frozen]   +   delta = trained - frozen

The delta panels (diverging colormap, centered at 0) are the point of the experiment —
positive => training helps. NaN-masked so a failed cell (e.g. NotMNIST without its local
copy) shows as a gap rather than crashing aggregation.

Usage
-----
    python experiments/RF_article/RF_dataset_sweep/aggregate_dataset.py \
        --results-dir experiments/RF_article/RF_dataset_sweep/results/run_20260629_120000
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Fixed display order; extra datasets found are appended after these.
DATASET_ORDER = ["mnist", "kmnist", "fmnist", "fashionmnist", "cifar10", "svhn", "notmnist"]
CONDITIONS = ["trained", "frozen"]


def parse_run(results_path: str):
    """Return a per-run row dict, or None if the JSON is missing/unreadable."""
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    cfg = data.get("config", {})
    dataset = cfg.get("dataset")
    if dataset is None:
        return None
    # condition may be absent in older runs -> infer from freeze_weights, default trained
    condition = cfg.get("condition")
    if condition is None:
        condition = "frozen" if cfg.get("freeze_weights") else "trained"
    return {
        "dataset": dataset,
        "condition": condition,
        "seed": cfg.get("seed"),
        "test_acc": data.get("test_acc", float("nan")),
        "test_phi": data.get("test_phi", float("nan")),
        "best_val_acc": data.get("best_val_acc", float("nan")),
    }


def load_all(results_dir: str):
    """Walk <results-dir>/<tag>/results.json. Report dirs missing a JSON."""
    cell_dirs = sorted(
        d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d)
    )
    rows, missing = [], []
    for d in cell_dirs:
        base = os.path.basename(d)
        if base == "slurm_logs":
            continue
        rp = os.path.join(d, "results.json")
        if not os.path.isfile(rp):
            missing.append(base)
            continue
        row = parse_run(rp)
        if row is None:
            missing.append(base + " (unreadable)")
            continue
        rows.append(row)
    print(f"  {len(rows)} runs parsed; {len(missing)} cell dir(s) without a usable results.json")
    return rows, missing


def _dataset_order(values) -> list:
    present = list(dict.fromkeys(values))
    ordered = [d for d in DATASET_ORDER if d in present]
    ordered += [d for d in present if d not in DATASET_ORDER]
    return ordered


def cond_pivot(df: pd.DataFrame, value_col: str, order: list) -> pd.DataFrame:
    """dataset (rows) x [trained, frozen] (cols), mean over seeds."""
    pivot = df.pivot_table(
        index="dataset", columns="condition", values=value_col, aggfunc="mean",
        observed=True,
    )
    pivot = pivot.reindex(index=order)
    pivot = pivot.reindex(columns=[c for c in CONDITIONS if c in pivot.columns])
    return pivot


def plot_heatmap(pivot, ax, title, cbar_label, cmap, fmt, center=None):
    sns.heatmap(
        pivot,
        ax=ax,
        mask=pivot.isna(),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("dataset", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)


def create_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    order = _dataset_order(df["dataset"].tolist())
    grid = df.copy()
    grid["test_acc_pct"] = grid["test_acc"] * 100

    # (value_col, label, cmap, fmt, delta_label)
    metrics = [
        ("test_acc_pct", "Test accuracy (%)", "viridis", ".1f", "delta accuracy (pts)"),
        ("test_phi", "Test phi", "magma", ".3f", "delta phi"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for row_i, (col, label, cmap, fmt, dlabel) in enumerate(metrics):
        piv = cond_pivot(grid, col, order)
        plot_heatmap(piv, axes[row_i, 0], f"{label}: trained vs frozen", label, cmap, fmt)

        # delta = trained - frozen (positive => training helps)
        if {"trained", "frozen"}.issubset(piv.columns):
            delta = (piv["trained"] - piv["frozen"]).to_frame(name="trained - frozen")
        else:
            delta = pd.DataFrame(index=piv.index, data={"trained - frozen": float("nan")})
        plot_heatmap(
            delta, axes[row_i, 1], f"{label}: does training help?", dlabel,
            "coolwarm", fmt, center=0.0,
        )

    fig.suptitle("Dataset sweep — trained vs frozen (canonical 1024/225 network)", fontsize=16)
    out_path = os.path.join(out_dir, "heatmap_dataset.pdf")
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
        help="Directory containing <tag>/results.json subfolders",
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
        sys.exit("No usable results.json found — nothing to write.")

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
        print(f"  {len(missing)} cell dir(s) produced no usable results.json:")
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

    create_heatmaps(df, out_dir)


if __name__ == "__main__":
    main()
