"""
Aggregate the x_tar percentile sweep from per-run results.json files.

Each run (run_experiment.py) writes results/<tag>/results.json with:
  * config            — x_tar_mode, x_tar_pct_se, x_tar_pct_ee, seed, ...
  * test_acc/test_phi — final test scores
  * stats_history     — one dict per batch carrying the diagnostic scalars
                        (rf_mean_cosine, rf_gini, rf_entropy, rf_pr_norm,
                         active_frac_exc, ei_ratio_median, ...)

This records, per run, the config plus the FIRST and LAST batch value of each RF
metric (the within-run trajectory endpoints) and their change. Sign convention for
"did the RF sharpen":

    d_cosine = last - first  (sharpen => negative — less redundant)
    d_gini   = last - first  (sharpen => positive — more concentrated)

Writes a tidy per-run table (Results_xtar.xlsx) and prints a per-cell mean over
seeds plus a coverage report (cells found / partially-seeded) so an incomplete grid
is obvious.

Usage
-----
    python experiments/RF_article/xtar_percentile/aggregate_xtar.py \
        --results-dir results/acc_history/mnist/2026.06.20/31
    python experiments/RF_article/xtar_percentile/aggregate_xtar.py \
        --results-dir <dir> --out summary.xlsx
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

# diagnostics whose first/last/change we summarise from stats_history
TRAJ_KEYS = [
    "rf_mean_cosine",
    "rf_gini",
    "rf_entropy",
    "rf_pr_norm",
    "active_frac_exc",
    "ei_ratio_median",
]


def _first_last(stats_history, key):
    """First and last non-null value of `key` across the batch stats history."""
    vals = [s[key] for s in stats_history if s.get(key) is not None]
    if not vals:
        return float("nan"), float("nan")
    return vals[0], vals[-1]


def parse_run(results_path: str):
    """Return a per-run row dict, or None if the JSON is missing/unreadable."""
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    cfg = data.get("config", {})
    mode = cfg.get("x_tar_mode")
    if mode is None:
        return None
    row = {
        "mode": mode,
        # percentile cells: SE/EE percentiles; blank for mean/static
        "pct_se": cfg.get("x_tar_pct_se") if mode == "percentile" else None,
        "pct_ee": cfg.get("x_tar_pct_ee") if mode == "percentile" else None,
        # static cells: fixed SE/EE thresholds; blank for mean/percentile
        "stat_se": cfg.get("x_tar_static_se") if mode == "static" else None,
        "stat_ee": cfg.get("x_tar_static_ee") if mode == "static" else None,
        "seed": cfg.get("seed"),
        "test_acc": data.get("test_acc", float("nan")),
        "test_phi": data.get("test_phi", float("nan")),
        "best_val_acc": data.get("best_val_acc", float("nan")),
    }

    # unified cell coordinates so percentile and static cells each group by their
    # own SE/EE level (mean stays a single cell with blank levels).
    if mode == "percentile":
        row["level_se"], row["level_ee"] = row["pct_se"], row["pct_ee"]
    elif mode == "static":
        row["level_se"], row["level_ee"] = row["stat_se"], row["stat_ee"]
    else:
        row["level_se"], row["level_ee"] = None, None

    sh = data.get("stats_history", []) or []
    for key in TRAJ_KEYS:
        first, last = _first_last(sh, key)
        row[f"{key}_first"] = first
        row[f"{key}_last"] = last
        row[f"d_{key}"] = last - first
    return row


def load_all(results_dir: str):
    """Walk <results-dir>/<tag>/results.json. Report dirs missing a JSON."""
    cell_dirs = sorted(
        d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d)
    )
    rows, missing = [], []
    for d in cell_dirs:
        rp = os.path.join(d, "results.json")
        if not os.path.isfile(rp):
            missing.append(os.path.basename(d))
            continue
        row = parse_run(rp)
        if row is None:
            missing.append(os.path.basename(d) + " (unreadable)")
            continue
        rows.append(row)
    print(f"  {len(rows)} runs parsed; {len(missing)} cell dir(s) without a usable results.json")
    return rows, missing


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate x_tar percentile sweep results.json files into Excel"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing <tag>/results.json subfolders",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .xlsx path (default: <results-dir>/Results_xtar.xlsx)",
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(results_dir):
        sys.exit(f"Directory not found: {results_dir}")

    rows, missing = load_all(results_dir)
    if not rows:
        sys.exit("No usable results.json found — nothing to write.")

    cell_keys = ["mode", "level_se", "level_ee"]
    df = pd.DataFrame(rows).sort_values(
        cell_keys + ["seed"], na_position="first"
    )

    out_path = args.out or os.path.join(results_dir, "Results_xtar.xlsx")
    df.to_excel(out_path, index=False, engine="openpyxl")
    n_cells = df.groupby(cell_keys, dropna=False).ngroups
    print(f"\n  Written {len(df)} rows ({n_cells} unique cells) -> {out_path}")

    # per-mode cell counts so the three estimators' coverage is visible at a glance
    print("\n  Cells per mode:")
    for mode, sub in df.groupby("mode"):
        print(f"    {mode:<11}: {sub.groupby(cell_keys, dropna=False).ngroups} cells, {len(sub)} runs")

    # ---- coverage: which cells are under-seeded (partial grid) -------------
    seeds_per_cell = df.groupby(cell_keys, dropna=False).size()
    n_seeds = int(df["seed"].nunique())
    partial = seeds_per_cell[seeds_per_cell < n_seeds]
    print(f"\n  Coverage: {n_cells} cells present, target {n_seeds} seeds/cell.")
    if len(partial):
        print(f"  {len(partial)} cell(s) under-seeded (<{n_seeds} seeds):")
        for (mode, se, ee), k in partial.items():
            if mode == "mean":
                tag = "mean"
            elif mode == "static":
                tag = f"stat_se{se}_ee{ee}"
            else:
                tag = f"se{int(se)}_ee{int(ee)}"
            print(f"    {tag}: {k} seed(s)")
    if missing:
        print(f"  {len(missing)} cell dir(s) produced no usable results.json (see list above).")

    # ---- per-cell mean over seeds, sharpening-relevant columns ------------
    summary = (
        df.groupby(cell_keys, dropna=False)[
            ["test_acc", "test_phi", "d_rf_mean_cosine", "d_rf_gini", "active_frac_exc_last"]
        ]
        .mean()
        .reset_index()
    )
    print("\n  Per-cell mean over seeds (sharpen => d_cosine<0, d_gini>0):")
    with pd.option_context("display.max_rows", None, "display.width", 130):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
