"""
Aggregate the network-size sweep from SLURM .out files and render the deliverable heatmap.

The per-cell results.json was not reliably written on the cluster, so the authoritative
source here is each task's captured stdout (*.out). We parse the printed config + summary
lines with regex (same approach as RF_size_tuning/aggregate_rf_size.py). The relevant lines
each run prints are:

    Config — dataset: mnist | ... | N_exc: 1024 | N_inh: 225 | inh_scale: 1.0000
             | peak_ei: 2.0000 | peak_ie: -2.0000 | seed: 0 | epochs: 1
    Test accuracy : 0.8474          (also "Test accuracy (PCA+LR): 0.8474")
    Test phi      : 0.1273
    Best val acc  : 0.8870

This scans every *.out, writes a tidy per-run table (Results_size.xlsx) and a clean
size_sweep.jsonl, reports coverage, and saves a size x seed heatmap (heatmap_size.pdf)
with a per-size mean column. Separate panels for accuracy and phi (never mixed units).

Usage
-----
    python experiments/RF_article/RF_netw_size_sweep/aggregate_size.py \
        --results-dir results/mnist/2026.06.29/HPC_network_size_sweep
    python experiments/RF_article/RF_netw_size_sweep/aggregate_size.py \
        --results-dir <dir> --out-dir <dir>
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

# Config line, e.g.:
#   Config — ... | N_exc: 1024 | N_inh: 225 | inh_scale: 1.0000 | peak_ei: 2.0000
#            | peak_ie: -2.0000 | seed: 0 | epochs: 1
RE_NEXC = re.compile(r"N_exc:\s*(\d+)")
RE_NINH = re.compile(r"N_inh:\s*(\d+)")
RE_SEED = re.compile(r"seed:\s*(\d+)")
RE_INH_SCALE = re.compile(r"inh_scale:\s*([\d.]+|nan)")
RE_PEAK_EI = re.compile(r"peak_ei:\s*(-?[\d.]+|nan)")
RE_PEAK_IE = re.compile(r"peak_ie:\s*(-?[\d.]+|nan)")
# Summary lines (test accuracy may carry a "(PCA+LR)" parenthetical).
RE_ACC = re.compile(r"Test accuracy\s*(?:\([^)]*\))?\s*:\s*([\d.]+|nan)")
RE_PHI = re.compile(r"Test phi\s*:\s*([\d.]+|nan)")
RE_BEST_VAL = re.compile(r"Best val acc\s*:\s*([\d.]+|nan)")


def _parse_float(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


def parse_run(out_path: str):
    """Parse one SLURM .out file into a per-run row dict, or None if no config found.

    Scans line by line: the Config line appears before the summary block, so we
    pick up identifiers first, then test/val scores. We stop early once every
    field is filled. inh_scale/peak_* default to NaN if an older run omitted them.
    """
    n_exc = n_inh = seed = None
    inh_scale = peak_ei = peak_ie = float("nan")
    test_acc = test_phi = best_val = float("nan")
    got_acc = got_phi = got_val = False

    try:
        with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if n_exc is None:
                    m = RE_NEXC.search(line)
                    if m:
                        n_exc = int(m.group(1))
                        mi = RE_NINH.search(line)
                        ms = RE_SEED.search(line)
                        msc = RE_INH_SCALE.search(line)
                        mei = RE_PEAK_EI.search(line)
                        mie = RE_PEAK_IE.search(line)
                        if mi:
                            n_inh = int(mi.group(1))
                        if ms:
                            seed = int(ms.group(1))
                        if msc:
                            inh_scale = _parse_float(msc.group(1))
                        if mei:
                            peak_ei = _parse_float(mei.group(1))
                        if mie:
                            peak_ie = _parse_float(mie.group(1))
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
                    m = RE_BEST_VAL.search(line)
                    if m:
                        best_val = _parse_float(m.group(1))
                        got_val = True
                if n_exc is not None and got_acc and got_phi and got_val:
                    break
    except OSError:
        return None

    if n_exc is None or n_inh is None:
        return None
    return {
        "n_exc": n_exc,
        "n_inh": n_inh,
        "seed": seed,
        "inh_scale": inh_scale,
        "peak_ei": peak_ei,
        "peak_ie": peak_ie,
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


def create_lineplot(df: pd.DataFrame, out_dir: str) -> None:
    """Accuracy / phi vs population size: mean line + per-seed points over N_exc."""
    grid = df.copy()
    grid["test_acc_pct"] = grid["test_acc"] * 100
    grid = grid.sort_values("n_exc")
    # x tick labels combine both populations, e.g. "1024/225"
    labels = {e: _size_label(e, i)
              for e, i in zip(grid["n_exc"], grid["n_inh"])}
    xs = sorted(grid["n_exc"].unique())

    # evenly-spaced categorical x positions (cleaner than a log axis for 2-3 sizes)
    pos = {e: k for k, e in enumerate(xs)}
    panels = [("test_acc_pct", "Test accuracy (%)", "#2a78d6"),
              ("test_phi", "Test phi", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for (col, title, color), ax in zip(panels, axes):
        means = grid.groupby("n_exc")[col].mean().reindex(xs)
        sds = grid.groupby("n_exc")[col].std().reindex(xs)
        xpos = [pos[x] for x in xs]
        ax.errorbar(xpos, means.values, yerr=sds.values, color=color, lw=2,
                    marker="o", ms=7, capsize=4, zorder=3, label="mean +/- sd")
        ax.scatter([pos[e] for e in grid["n_exc"]], grid[col], color=color,
                   alpha=0.4, s=40, zorder=2, label="per seed")
        ax.set_xticks(xpos)
        ax.set_xticklabels([labels[x] for x in xs])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_xlabel("network size  (N_exc / N_inh)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title + " vs population size", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Network-size sweep (proportional inhibition)", fontsize=15)
    out = os.path.join(out_dir, "lineplot_size.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate network-size sweep .out files into a table + heatmap"
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
        print(f"  {len(missing)} .out file(s) had no usable config:")
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
    create_lineplot(df, out_dir)


if __name__ == "__main__":
    main()
