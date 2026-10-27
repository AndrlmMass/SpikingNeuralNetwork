"""
Article OOD figure: AUROC of three novelty detectors, MNIST test vs each OOD set.

Rebuilt 09-17 from the stored run (the original lived in a scratch script that was lost).
Black and white, with a legend box, per the supervisor's print feedback.

Detectors
  group concentration   entropy of excitatory rates pooled per class group; no fitted
                        parameters. Familiar input concentrates in one group.
  Mahalanobis           min over classes of the Mahalanobis distance to class-conditional
                        Gaussians with a tied, ridge-regularised covariance, fitted on the
                        first half of the MNIST calibration split (ridge 0.01). The
                        other half calibrates its threshold; AUROC needs none.
  pixel PCA residual    from input_density.json: residual outside MNIST's top-50 pixel
                        principal subspace, two-sided AUROC. No network at all.

Seeds
  --run is either one seed's run dir or a run root holding mnist_ood_s*/ (one per seed,
  from mnist_family_sweep/run_slurm_ood.sh). Each seed is scored on its own network;
  bars are the mean over seeds and error bars the SD across them (none for one seed).

Usage:
    python experiments/RF_article/interp/plot_ood_auroc.py
    python experiments/RF_article/interp/plot_ood_auroc.py --run results/ood_mnist/<RUN_ID>
"""
import argparse, glob, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN = os.path.join(REPO, "results", "ood_mnist", "run1")
OOD = ["kmnist", "notmnist", "fmnist"]   # decreasing low-level similarity to MNIST (SVHN dropped)
PRETTY = {"kmnist": "KMNIST", "notmnist": "notMNIST", "fmnist": "FMNIST",
          "svhn": "SVHN"}


def group_entropy(X, assign, n_groups):
    G = np.stack([X[:, assign == g].sum(1) for g in range(n_groups)], 1)
    P = G / np.clip(G.sum(1, keepdims=True), 1e-12, None)
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1)


def mahalanobis(Xfit, yfit, ridge=1e-2):
    classes = np.unique(yfit)
    mu = np.stack([Xfit[yfit == c].mean(0) for c in classes])
    R = Xfit - mu[np.searchsorted(classes, yfit)]
    S = R.T @ R / len(R)
    S += ridge * np.trace(S) / len(S) * np.eye(len(S))
    L = np.linalg.cholesky(S)

    def score(X):
        d = np.full(len(X), np.inf)
        for m in mu:
            z = np.linalg.solve(L, (X - m).T).T
            d = np.minimum(d, (z * z).sum(1))
        return d
    return score


def auroc(s_id, s_ood):
    """Higher score = more novel."""
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    return roc_auc_score(y, np.r_[s_id, s_ood])


def compute(run):
    z = np.load(os.path.join(run, "uncertainty_features.npz"))
    assign = z["assignment"]; ng = int(assign.max()) + 1
    Xc, yc, Xt = z["X_cal"].astype(np.float64), z["y_cal"], z["X_test"].astype(np.float64)
    # fitted on the first half of the calibration split; the second half calibrates the
    # rejection threshold (on run1/s0 this gives KMNIST/notMNIST/FMNIST 0.879/0.968/0.884)
    cut = len(Xc) // 2
    maha = mahalanobis(Xc[:cut], yc[:cut])
    res = {"group": {}, "maha": {}, "pca": {}}
    g_id, m_id = group_entropy(Xt, assign, ng), maha(Xt)
    for o in OOD:
        Xo = z[f"X_ood_{o}"].astype(np.float64)
        res["group"][o] = auroc(g_id, group_entropy(Xo, assign, ng))
        res["maha"][o] = auroc(m_id, maha(Xo))
    dens = json.load(open(os.path.join(run, "input_density.json")))
    for o in OOD:
        res["pca"][o] = dens["pixels"]["pca_residual"][o]["auroc_2sided"]
    return res


def seed_dirs(run):
    """A single seed's run dir, or every mnist_ood_s*/ under a run root."""
    if os.path.exists(os.path.join(run, "uncertainty_features.npz")):
        return [run]
    dirs = sorted(d for d in glob.glob(os.path.join(run, "mnist_ood_s*"))
                  if os.path.exists(os.path.join(d, "uncertainty_features.npz")))
    if not dirs:
        raise SystemExit(f"{run}: no uncertainty_features.npz, and no mnist_ood_s*/ holding one")
    return dirs


def compute_seeds(run):
    """{key: {ood: (mean, sd, n)}} over seeds; sd is nan for a single seed."""
    per = [compute(d) for d in seed_dirs(run)]
    out = {}
    for key, _, _ in METHODS:
        out[key] = {}
        for o in OOD:
            v = np.array([r[key][o] for r in per])
            out[key][o] = (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, len(v))
    return out


METHODS = [("group", "group concentration (network)", ("white", "")),
           ("maha", "Mahalanobis distance (network)", ("white", "////")),
           ("pca", "pixel PCA residual (no network)", ("#bdbdbd", ""))]


def figure(res, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(OOD)); w = 0.26
    for k, (key, label, (fc, hatch)) in enumerate(METHODS):
        vals = np.array([res[key][o][0] for o in OOD])
        sds = np.array([res[key][o][1] for o in OOD])
        has_sd = np.isfinite(sds).any()
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=fc, hatch=hatch, edgecolor="black",
               lw=1.1, zorder=3, yerr=(sds if has_sd else None),
               error_kw=dict(ecolor="black", elinewidth=1.2, capsize=4, zorder=4))
        tops = vals + (np.nan_to_num(sds) if has_sd else 0.0)
        for xi, v, t in zip(x + (k - 1) * w, vals, tops):
            ax.text(xi, t + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=14)
    ax.set_ylim(0.5, 1.07)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[o] for o in OOD], fontsize=18)
    ax.set_ylabel("AUROC (MNIST vs. OOD)", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Patch(facecolor=fc, hatch=h, edgecolor="black", lw=1.1)
               for _, _, (fc, h) in METHODS]
    ax.legend(handles, [m[1] for m in METHODS], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=15, frameon=True,
              edgecolor="#666666", fancybox=False, handlelength=2.2, handleheight=1.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "figures", "ood_auroc"))
    a = ap.parse_args()
    res = compute_seeds(a.run)
    n = res[METHODS[0][0]][OOD[0]][2]
    print(f"AUROC, mean +- SD over {n} seed(s)")
    print(f"{'':34}" + "".join(f"{PRETTY[o]:>17}" for o in OOD))
    for key, label, _ in METHODS:
        print(f"{label:34}" + "".join(f"{m:10.3f} +-{s:5.3f}" for m, s, _ in
                                      (res[key][o] for o in OOD)))
    figure(res, a.out)
    print("wrote", a.out + ".pdf")


if __name__ == "__main__":
    main()
