"""
Assemble comparison datasets for the sleep-optimal vs normalization vs no-reg GLMM.

Two datasets are produced:
  GLMM_comparison_acc.xlsx  — sleep config tuned for accuracy  + normalize + no-reg
  GLMM_comparison_phi.xlsx  — sleep config tuned for clustering + normalize + no-reg

Per-target optima: each reg_mode (layer/neuron/static) gets its own best
(sleep_duration, var_noise) combination selected by mean over seeds.

Usage
-----
    python experiments/noise_article/GLMM/prepare_comparison_data.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
PHASE2_PATH    = os.path.join(REPO_ROOT, "results", "acc_history", "mnist",
                               "2026.05.24", "21", "Results_phase2.xlsx")
PHASE2EXT_PATH = os.path.join(REPO_ROOT, "results", "acc_history", "mnist",
                               "2026.05.24", "21", "Results_phase2 EXT.xlsx")
PHASE1_PATH    = os.path.join(REPO_ROOT, "results", "acc_history", "mnist",
                               "2026.05.16", "17", "phase1_results.xlsx")
OUT_DIR        = os.path.join(REPO_ROOT, "results", "acc_history", "mnist",
                               "2026.05.24", "21")

MODES = ["layer", "neuron", "static"]

# Static (300, 100) is a pathological outlier for phi (phi inflates while
# accuracy collapses to ~12%). Flag it but still include; noted in paper.
PHI_OUTLIER = ("static", 300, 100.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_optimal(df: pd.DataFrame, metric: str) -> dict:
    """Return {reg_mode: (sleep_duration, var_noise)} maximising mean metric."""
    means = (
        df.groupby(["reg_mode", "sleep_duration", "var_noise"])[metric]
        .mean()
        .reset_index()
    )
    optima = {}
    for mode in MODES:
        sub = means[means["reg_mode"] == mode]
        best = sub.loc[sub[metric].idxmax()]
        optima[mode] = (int(best["sleep_duration"]), float(best["var_noise"]))
    return optima


def extract_optimal_rows(df: pd.DataFrame, optima: dict, reg_type_label: str) -> pd.DataFrame:
    """Extract all seed rows matching the optimal (sleep_duration, var_noise) per mode."""
    parts = []
    for mode, (sd, vn) in optima.items():
        mask = (
            (df["reg_mode"] == mode) &
            (df["sleep_duration"] == sd) &
            (df["var_noise"] == vn)
        )
        rows = df[mask].copy()
        rows["reg_type"] = reg_type_label
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    grp = df.groupby(["reg_type", "reg_mode"]).agg(
        n=("test_acc", "count"),
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_phi=("test_phi", "mean"),
        std_phi=("test_phi", "std"),
    ).round(4)
    print(grp.to_string())
    print(f"\nTotal rows: {len(df)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    print("Loading data...")
    df2    = pd.read_excel(PHASE2_PATH,    engine="openpyxl")
    df_ext = pd.read_excel(PHASE2EXT_PATH, engine="openpyxl")
    df1    = pd.read_excel(PHASE1_PATH,    sheet_name="raw", engine="openpyxl")

    # ---- Per-target optima -----------------------------------------------
    acc_optima = find_optimal(df2, "test_acc")
    phi_optima = find_optimal(df2, "test_phi")

    print("\nAccuracy-optimal configurations (per target):")
    for mode, (sd, vn) in acc_optima.items():
        mean_acc = df2[
            (df2["reg_mode"] == mode) &
            (df2["sleep_duration"] == sd) &
            (df2["var_noise"] == vn)
        ]["test_acc"].mean()
        print(f"  {mode:8s}: sleep_duration={sd:4d}, var_noise={vn:6.1f}  "
              f"-> mean_acc={mean_acc:.4f}")

    print("\nClustering-optimal configurations (per target):")
    for mode, (sd, vn) in phi_optima.items():
        sub = df2[
            (df2["reg_mode"] == mode) &
            (df2["sleep_duration"] == sd) &
            (df2["var_noise"] == vn)
        ]
        mean_phi = sub["test_phi"].mean()
        mean_acc = sub["test_acc"].mean()
        flag = " *** OUTLIER (high noise, low acc)" \
            if (mode, sd, vn) == PHI_OUTLIER else ""
        print(f"  {mode:8s}: sleep_duration={sd:4d}, var_noise={vn:6.1f}  "
              f"-> mean_phi={mean_phi:.2f}, mean_acc={mean_acc:.4f}{flag}")

    # ---- Sleep-optimal rows (phase 2) ------------------------------------
    sleep_acc = extract_optimal_rows(df2, acc_optima, "sleep_opt_acc")
    sleep_phi = extract_optimal_rows(df2, phi_optima, "sleep_opt_phi")

    # ---- Normalization rows (phase 1, normalize only) --------------------
    norm = df1[df1["reg_type"] == "normalize"].copy()
    # phase1 raw sheet uses 'reg_type' already; ensure consistent columns
    norm = norm[["reg_type", "reg_mode", "seed", "test_acc", "test_phi"]].copy()
    # No sleep_duration / var_noise for normalization — fill with NaN
    norm["sleep_duration"] = float("nan")
    norm["var_noise"]       = float("nan")

    # ---- No-reg baseline (phase 2 EXT) -----------------------------------
    no_reg = df_ext[df_ext["reg_mode"] == "none"].copy()
    no_reg = no_reg[["seed", "test_acc", "test_phi"]].copy()
    no_reg["reg_type"]       = "none"
    no_reg["reg_mode"]       = "none"
    no_reg["sleep_duration"] = float("nan")
    no_reg["var_noise"]      = float("nan")

    # ---- Assemble final columns ------------------------------------------
    COLS = ["reg_type", "reg_mode", "seed", "test_acc", "test_phi",
            "sleep_duration", "var_noise"]

    def assemble(sleep_part):
        sleep_part = sleep_part[COLS].copy()
        return pd.concat(
            [sleep_part, norm[COLS], no_reg[COLS]],
            ignore_index=True,
        )

    df_acc_out = assemble(sleep_acc)
    df_phi_out = assemble(sleep_phi)

    # ---- Print summaries --------------------------------------------------
    print_summary(df_acc_out, "Dataset A — accuracy-optimal sleep")
    print_summary(df_phi_out, "Dataset B — clustering-optimal sleep")

    # ---- Save -------------------------------------------------------------
    out_acc = os.path.join(OUT_DIR, "GLMM_comparison_acc.xlsx")
    out_phi = os.path.join(OUT_DIR, "GLMM_comparison_phi.xlsx")
    df_acc_out.to_excel(out_acc, index=False, engine="openpyxl")
    df_phi_out.to_excel(out_phi, index=False, engine="openpyxl")
    print(f"\nSaved -> {out_acc}")
    print(f"Saved -> {out_phi}")


if __name__ == "__main__":
    main()
