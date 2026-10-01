"""Distribution of relative accuracy at each sleep duration, pooled over datasets.

Motivating figure: as sleep duration grows, runs stop scattering around one
typical value and the distribution becomes bimodal.

Each run's test accuracy is divided by the mean no-sleep (0%) accuracy of its
own dataset, so 1.0 is "as good as that dataset without sleep" and datasets of
different difficulty sit on one axis. One ridge per sleep duration: a Gaussian
KDE (scipy.stats.gaussian_kde, bw_method=0.25) of the 20 runs (4 datasets x 5 seeds), with the runs as rug ticks.

    python src/glmm/plot_sweep_bimodal.py

Inputs:  results/sweep/sweep_summary.csv
Outputs: figures/sweep_bimodal_BW.{pdf,png}
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_glmms_bw import T, FS_LABEL, REPO, style, save

# 'raw': test accuracy. 'relative': divided by the dataset's no-sleep mean --
# meaningful only while no-sleep runs learn; after the 2026-09-30 fixes the
# no-sleep reference is at chance and every sleep run lands at 4-8x it.
SCALE = "raw"
X_MAX = 1.0 if SCALE == "raw" else 2.35


def load():
    d = pd.read_csv(os.path.join(REPO, "results", "sweep", "sweep_summary.csv"))
    d = d[d.test_accuracy.notna()].copy()
    base = d[d.sleep_rate == 0].groupby("dataset").test_accuracy.mean()
    d["rel"] = (d.test_accuracy if SCALE == "raw"
                else d.test_accuracy / d.dataset.map(base))
    d["pct"] = (d.sleep_rate * 100).round().astype(int)
    return d


def bimodality_coefficient(v):
    """Sarle's b = (g^2 + 1) / (kurtosis + 3(n-1)^2/((n-2)(n-3))); > 5/9 ~ 0.555
    suggests bimodality (the value for a uniform distribution)."""
    n = len(v)
    m = v - v.mean()
    s2 = np.mean(m ** 2)
    g = np.mean(m ** 3) / s2 ** 1.5 * np.sqrt(n * (n - 1)) / (n - 2)
    k = np.mean(m ** 4) / s2 ** 2 - 3
    k = ((n + 1) * k + 6) * (n - 1) / ((n - 2) * (n - 3))
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def main():
    d = load()
    per = d.groupby("pct").agg(
        n=("rel", "size"),
        rel_mean=("rel", "mean"), rel_sd=("rel", "std"))
    per["bimodality_coef"] = d.groupby("pct").rel.apply(
        lambda v: bimodality_coefficient(v.to_numpy()))
    print(per.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10.0, 8.5), facecolor=T["surface"])
    style(ax, ygrid=False)
    x_lo = 0.0 if SCALE == "raw" else -0.1
    xgrid = np.linspace(x_lo, X_MAX, 500)
    step, height = 1.0, 1.45
    pcts = sorted(d.pct.unique())
    dens = {p: gaussian_kde(d.rel[d.pct == p], bw_method=0.25)(xgrid)
            for p in pcts}
    peak = max(v.max() for v in dens.values())
    top_y = (len(pcts) - 1) * step + height + 0.35

    for i, p in enumerate(pcts):
        base = i * step
        top = base + dens[p] / peak * height
        z = 10 + len(pcts) - i
        ax.fill_between(xgrid, base, top, color=T["surface"], lw=0, zorder=z)
        ax.fill_between(xgrid, base, top, color="#bdbdbd", alpha=0.55, lw=0,
                        zorder=z)
        ax.plot(xgrid, top, color=T["ink"], lw=1.4, zorder=z)
        v = d.rel[d.pct == p].to_numpy()
        ax.plot(v, np.full_like(v, base), "|", color=T["ink2"], ms=5.6,
                mew=0.9, zorder=z)

    ax.set_yticks([i * step for i in range(len(pcts))])
    ax.set_yticklabels([f"{p}%" for p in pcts])
    ax.set_ylim(-0.3, top_y)
    ax.set_xlim(x_lo, X_MAX)
    ax.set_xticks(np.arange(0, X_MAX + 1e-9, 0.2 if SCALE == "raw" else 0.5))
    ax.set_xlabel("test accuracy" if SCALE == "raw" else "test accuracy relative to no-sleep",
                  fontsize=1.5 * (FS_LABEL - 3), color=T["ink2"])
    ax.set_ylabel("sleep duration", fontsize=1.5 * (FS_LABEL - 3),
                  color=T["ink2"])

    save(fig, "sweep_bimodal_BW")


if __name__ == "__main__":
    main()
