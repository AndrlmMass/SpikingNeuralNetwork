"""
2D-Gaussian receptive-field geometry over training, next to orientation coherence.

Hubin's request: show WHAT changes structurally in the RF prior during training, not
merely that coherence moved. `rf_gaussian_moments` (neurosnn/_evaluation/analysis.py)
logs, at every checkpoint, the energy-weighted mean second-moment tensor of the W_se
columns treated as a spatial mass distribution:

    M = [[rf_var_x, rf_cov_xy],
         [rf_cov_xy, rf_var_y]]

var_x / var_y are the RF extent along each image axis (pixels^2); cov_xy is the tilt.
This script aggregates those trajectories across seeds and plots them alongside
`orient_coh`, for every dataset x condition cell of a sweep.

Two derived quantities are computed HERE rather than read from the trajectory:

  rf_anisotropy = sqrt((var_x - var_y)^2 + 4 cov_xy^2) / (var_x + var_y)
      Bounded in [0, 1] (0 = isotropic blob, 1 = a line), the same functional form
      orientation_coherence applies to the structure tensor, applied instead to the
      mass tensor. It is used in place of the logged `rf_elongation`, which is
      sqrt(lambda_max/lambda_min) with lambda_min floored at 1e-12 and therefore
      diverges (values ~1e4) whenever RFs are sparse enough to be near-collinear --
      as the random-prior runs are. var/cov/anisotropy are unaffected.

  the 1-sigma ellipse of M, drawn first-checkpoint vs last-checkpoint, which is the
      2D Gaussian itself rather than its components.

The frozen (no-plasticity) condition contributes a single checkpoint: it IS the
initialisation, so it is drawn as a horizontal reference line -- the only view of the
t=0 geometry, since the first logged checkpoint of a plastic run already has
plasticity in it.

  python experiments/RF_article/interp/plot_rf_geometry.py --results-dir <run_dir>
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Categorical slots in fixed order (validated for a light surface); conditions are
# assigned in the order below and never cycled. `frozen` is a REFERENCE line, not a
# series, so it wears muted ink instead of a hue.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
COND_ORDER = ["base_ori", "oriented", "base_rnd", "random",
              "triplet", "ee_off", "ie_off", "vogels"]
REFERENCE = "frozen"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

PANELS = [("orient_coh",    "orientation coherence", "linear"),
          ("rf_var_x",      "RF var$_x$ (px$^2$)",   "log"),
          ("rf_var_y",      "RF var$_y$ (px$^2$)",   "log"),
          ("rf_cov_xy",     "RF cov$_{xy}$ (px$^2$)", "symlog"),
          ("rf_anisotropy", "RF anisotropy", "linear")]


def anisotropy(vx, vy, cxy):
    """(lambda1 - lambda2)/(lambda1 + lambda2) of [[vx, cxy], [cxy, vy]]."""
    tr = np.asarray(vx) + np.asarray(vy)
    return np.sqrt((np.asarray(vx) - np.asarray(vy)) ** 2
                   + 4 * np.asarray(cxy) ** 2) / np.maximum(tr, 1e-12)


def load(run_dir):
    """-> {(dataset, cond): [results.json dict, ...]} , one entry per seed."""
    cells = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "*"))):
        rp = os.path.join(d, "results.json")
        if not os.path.isfile(rp):
            continue
        r = json.load(open(rp))
        cfg = r["config"]
        tag, ds = cfg["tag"], cfg["dataset"]
        cond = tag[len(ds) + 1:].rsplit("_s", 1)[0]
        cells.setdefault((ds, cond), []).append(r)
    return cells


def stack(runs, key):
    """batches, mean, sd over seeds. Seeds share a checkpoint grid within a cell."""
    n = min(len(r["trajectory"]) for r in runs)
    batches = np.array([t["batch"] for t in runs[0]["trajectory"][:n]], float)
    if key == "rf_anisotropy":
        vals = np.array([[anisotropy(t["rf_var_x"], t["rf_var_y"], t["rf_cov_xy"])
                          for t in r["trajectory"][:n]] for r in runs], float)
    else:
        vals = np.array([[t.get(key, np.nan) for t in r["trajectory"][:n]]
                         for r in runs], float)
    return batches, np.nanmean(vals, 0), np.nanstd(vals, 0)


def colors_for(conds):
    ordered = [c for c in COND_ORDER if c in conds]
    ordered += [c for c in sorted(conds) if c not in ordered and c != REFERENCE]
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(ordered)}, ordered


def trajectory_figure(cells, out_path, title):
    datasets = sorted({ds for ds, _ in cells})
    conds = {c for _, c in cells}
    cmap, ordered = colors_for(conds)
    has_ref = REFERENCE in conds

    fig, axes = plt.subplots(len(datasets), len(PANELS),
                             figsize=(4.0 * len(PANELS), 2.9 * len(datasets)),
                             squeeze=False)
    for r, ds in enumerate(datasets):
        for c, (key, label, scale) in enumerate(PANELS):
            ax = axes[r][c]
            if has_ref and (ds, REFERENCE) in cells:
                _, m, _ = stack(cells[(ds, REFERENCE)], key)
                ax.axhline(m[0], ls="--", lw=1.4, color=MUTED, zorder=1,
                           label="frozen (init, no plasticity)" if r == c == 0 else None)
            # Descending line width in draw order: the mechanism ablations sit
            # exactly on top of the baseline, and a constant width would hide every
            # series but the last drawn. Staggering keeps coincidence VISIBLE.
            for i, cond in enumerate(ordered):
                if (ds, cond) not in cells:
                    continue
                b, m, sd = stack(cells[(ds, cond)], key)
                if len(b) < 2:
                    continue
                lw = max(3.2 - 0.4 * i, 1.1)
                ax.fill_between(b, m - sd, m + sd, color=cmap[cond], alpha=0.15, lw=0)
                ax.plot(b, m, "-", lw=lw, color=cmap[cond], zorder=3 + i,
                        label=cond if r == c == 0 else None)
            if scale == "log":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
                ax.tick_params(axis="y", which="minor", labelsize=6)
            elif scale == "symlog":
                ax.set_yscale("symlog", linthresh=0.1)
                ax.axhline(0.0, lw=0.8, color=GRID, zorder=0)
            ax.grid(alpha=0.25, color=GRID, lw=0.7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            if r == 0:
                ax.set_title(label, fontsize=11, color=INK)
            if r == len(datasets) - 1:
                ax.set_xlabel("checkpoint (batch)", fontsize=9, color=MUTED)
            if c == 0:
                ax.set_ylabel(ds, fontsize=11, color=INK)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.008))
    fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


def ellipse_figure(cells, out_path, title):
    """The 2D Gaussian itself: 1-sigma ellipse of M, first vs last checkpoint.

    M is the energy-weighted mean over neurons, so this is the population ENVELOPE,
    not a typical neuron: preferred orientations are close to uniform across the
    layer, so per-neuron elongation largely averages out and the mean ellipse is
    near-circular even where individual RFs are elongated. What it shows is the
    change in overall RF extent and the net tilt (cov_xy).

    Axis limits are shared across each dataset row -- the size difference between
    conditions is the point, so panels must not auto-scale independently.
    """
    datasets = sorted({ds for ds, _ in cells})
    cmap, ordered = colors_for({c for _, c in cells})
    cols = [c for c in ordered if any((ds, c) in cells for ds in datasets)]
    if REFERENCE in {c for _, c in cells}:
        cols = [REFERENCE] + cols

    def ellipse(vx, vy, cxy):
        ev, evec = np.linalg.eigh(np.array([[vx, cxy], [cxy, vy]]))
        w, h = 2 * np.sqrt(np.clip(ev[::-1], 0, None))       # 1-sigma diameters
        return w, h, np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))

    fig, axes = plt.subplots(len(datasets), len(cols),
                             figsize=(2.5 * len(cols), 2.6 * len(datasets)),
                             squeeze=False)
    for r, ds in enumerate(datasets):
        geom, lim = {}, 0.0
        for cond in cols:
            if (ds, cond) not in cells:
                continue
            ends = [stack(cells[(ds, cond)], k)[1]
                    for k in ("rf_var_x", "rf_var_y", "rf_cov_xy")]
            geom[cond] = [ellipse(*(e[j] for e in ends)) for j in (0, -1)]
            lim = max(lim, *(max(w, h) for w, h, _ in geom[cond]))
        lim *= 0.62
        for c, cond in enumerate(cols):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.grid(alpha=0.25, color=GRID, lw=0.7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
            if r == 0:
                ax.set_title(cond, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(ds, fontsize=10, color=INK)
            else:
                ax.set_yticklabels([])
            if r != len(datasets) - 1:
                ax.set_xticklabels([])
            if cond not in geom:
                ax.set_axis_off()
                continue
            col = MUTED if cond == REFERENCE else cmap[cond]
            for (w, h, ang), alpha, ls, lab in zip(geom[cond], (0.4, 1.0),
                                                   ("--", "-"), ("first", "last")):
                ax.add_patch(Ellipse((0, 0), w, h, angle=ang, fill=False,
                                     lw=2.0, ls=ls, color=col, alpha=alpha,
                                     label=lab if (r == 0 and c == 0) else None))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title + "\nenergy-weighted MEAN moment tensor (population envelope) — "
                 "axes in pixels, scale shared per dataset row",
                 fontsize=12, color=INK, linespacing=1.6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


def summary_csv(cells, out_path):
    """The table view behind the figures: first/last mean+-sd per cell and metric."""
    keys = [k for k, _, _ in PANELS] + ["rf_elongation", "rf_orient"]
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "condition", "n_seeds", "metric",
                    "first_mean", "first_sd", "last_mean", "last_sd", "delta_pct"])
        for (ds, cond), runs in sorted(cells.items()):
            for key in keys:
                _, m, sd = stack(runs, key)
                d = 100 * (m[-1] - m[0]) / m[0] if abs(m[0]) > 1e-12 else float("nan")
                w.writerow([ds, cond, len(runs), key,
                            f"{m[0]:.6g}", f"{sd[0]:.6g}",
                            f"{m[-1]:.6g}", f"{sd[-1]:.6g}", f"{d:.2f}"])
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="a sweep run dir (one folder per run)")
    ap.add_argument("--out-dir", default=None, help="default: --results-dir")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    cells = load(args.results_dir)
    if not cells:
        raise SystemExit(f"no results.json found under {args.results_dir}")
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    title = args.title or os.path.basename(os.path.normpath(args.results_dir))
    n = sum(len(v) for v in cells.values())
    print(f"{n} runs, {len(cells)} dataset x condition cells")

    trajectory_figure(cells, os.path.join(out_dir, "rf_geometry_trajectories.png"),
                      f"2D-Gaussian RF geometry and orientation coherence — {title}")
    ellipse_figure(cells, os.path.join(out_dir, "rf_gaussian_ellipses.png"),
                   f"1$\\sigma$ RF ellipse, first vs last checkpoint — {title}")
    summary_csv(cells, os.path.join(out_dir, "rf_geometry_summary.csv"))


if __name__ == "__main__":
    main()
