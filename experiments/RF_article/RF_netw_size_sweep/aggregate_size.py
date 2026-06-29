"""
Aggregate the network-size sweep from per-cell results.json files and render the
deliverable heatmap.

Each cell (run_experiment.py) writes results/<RUN_ID>/<tag>/results.json with:
  * config            — n_exc, n_inh, inh_scale, peak_ei, peak_ie, seed, ...
  * test_acc/test_phi — final test scores
  * best_val_acc      — best validation accuracy

This scans every <tag>/results.json (authoritative — independent of the best-effort
live size_sweep.jsonl), writes a tidy per-run table (Results_size.xlsx) and a clean
size_sweep.jsonl, reports coverage, and saves a size x seed heatmap (heatmap_size.pdf)
with a per-size mean column. Separate panels for accuracy and phi (never mixed units).

Usage
-----
    python experiments/RF_article/RF_netw_size_sweep/aggregate_size.py \
        --results-dir experiments/RF_article/RF_netw_size_sweep/results/run_20260629_120000
    python experiments/RF_article/RF_netw_size_sweep/aggregate_size.py \
        --results-dir <dir> --out-dir <dir>
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


def parse_run(results_path: str):
    """Return a per-run row dict, or None if the JSON is missing/unreadable."""
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    cfg = data.get("config", {})
    n_exc = cfg.get("n_exc")
    n_inh = cfg.get("n_inh")
    if n_exc is None or n_inh is None:
        return None
    return {
        "n_exc": n_exc,
        "n_inh": n_inh,
        "seed": cfg.get("seed"),
        "inh_scale": cfg.get("inh_scale", float("nan")),
        "peak_ei": cfg.get("peak_ei", float("nan")),
        "peak_ie": cfg.get("peak_ie", float("nan")),
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


def _size_label(n_exc, n_inh) -> str:
    return f"{int(n_exc)}/{int(n_inh)}"


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """size (rows, ascending N_exc) x [seed 0..k, mean] pivot of value_col."""
    df = df.copy()
    df["size"] = [_size_label(e, i) for e, i in zip(df["n_exc"], df["n_inh"])]
    # ordering rows by N_exc so the sweep reads small -> large top to bottom
    order = (
        df[["n_exc", "size"]].drop_duplicates().sort_values("n_exc")["size"].tolist()
    )
    pivot = df.pivot_table(index="size", columns="seed", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(order)
    pivot.columns = [f"seed {int(c)}" for c in pivot.columns]
    pivot["mean"] = pivot.mean(axis=1, skipna=True)
    return pivot


def plot_heatmap(pivot, ax, title, cbar_label, cmap="viridis", fmt=".3f"):
    sns.heatmap(
        pivot,
        ax=ax,
        mask=pivot.isna(),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("seed", fontsize=13)
    ax.set_ylabel("network size  (N_exc / N_inh)", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)


def create_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    grid = df.copy()
    grid["test_acc_pct"] = grid["test_acc"] * 100

    panels = [
        ("test_acc_pct", "Test accuracy (%)", "Accuracy (%)", "viridis", ".1f"),
        ("test_phi", "Test phi", "phi", "magma", ".3f"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for (col, title, cbar, cmap, fmt), ax in zip(panels, axes.ravel()):
        pivot = build_pivot(grid, col)
        plot_heatmap(pivot, ax, title, cbar, cmap=cmap, fmt=fmt)
    fig.suptitle("Network-size sweep (proportional inhibition)", fontsize=16)

    out_path = os.path.join(out_dir, "heatmap_size.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # individual panels too
    for col, title, cbar, cmap, fmt in panels:
        pivot = build_pivot(grid, col)
        fig_s, ax_s = plt.subplots(figsize=(8, 6), constrained_layout=True)
        plot_heatmap(pivot, ax_s, title, cbar, cmap=cmap, fmt=fmt)
        p = os.path.join(out_dir, f"heatmap_size_{col}.pdf")
        fig_s.savefig(p, dpi=150)
        plt.close(fig_s)
        print(f"  Saved -> {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate network-size sweep results.json into a table + heatmap"
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

    df = pd.DataFrame(rows).sort_values(["n_exc", "n_inh", "seed"])

    # ---- tidy outputs --------------------------------------------------------
    xlsx_path = os.path.join(out_dir, "Results_size.xlsx")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    jsonl_path = os.path.join(out_dir, "size_sweep.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in df.to_dict(orient="records"):
            fh.write(json.dumps(rec) + "\n")
    n_cells = df.groupby(["n_exc", "n_inh"]).ngroups
    print(f"\n  Written {len(df)} rows ({n_cells} sizes) -> {xlsx_path}")
    print(f"  Rebuilt authoritative jsonl -> {jsonl_path}")

    # ---- coverage: which sizes are under-seeded ------------------------------
    seeds_per_cell = df.groupby(["n_exc", "n_inh"]).size()
    n_seeds = int(df["seed"].nunique())
    partial = seeds_per_cell[seeds_per_cell < n_seeds]
    print(f"\n  Coverage: {n_cells} sizes present, target {n_seeds} seeds/size.")
    if len(partial):
        print(f"  {len(partial)} size(s) under-seeded (<{n_seeds} seeds):")
        for (e, i), k in partial.items():
            print(f"    {_size_label(e, i)}: {k} seed(s)")
    if missing:
        print(f"  {len(missing)} cell dir(s) produced no usable results.json:")
        for name in missing:
            print(f"    {name}")

    # ---- per-size mean +/- sd over seeds ------------------------------------
    summary = (
        df.groupby(["n_exc", "n_inh"])[["test_acc", "test_phi", "best_val_acc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\n  Per-size mean +/- sd over seeds:")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.to_string(index=False))

    create_heatmaps(df, out_dir)


if __name__ == "__main__":
    main()
