"""
Every OOD number the article's Section "Rejecting out-of-distribution data" quotes, per
seed and averaged over seeds, from the stored run folders. Replaces the lost scratch
script behind the Mahalanobis rejection rates and the calibration footnote.

Per seed (one mnist_ood_s* folder):
  entropy      predictive entropy of the learned readout     AUROC, FPR@95  (ood.json)
  group        entropy of rates pooled per class group       AUROC, rejection at alpha
  maha         min class-conditional Mahalanobis distance,   AUROC, FPR@95, rejection at
               fitted on cal[:2500], tau from cal[2500:]     alpha, and the false-rejection
                                                             rate if it were fitted AND
                                                             calibrated on all of cal
  pixel        residual outside MNIST's top-50 pixel PCs     AUROC, rejection at alpha
               (input_density.json)
  intensity    mean image intensity, two-sided AUROC         (input_density.json)
  rate         mean excitatory rate, MNIST test vs each OOD set
alpha = 0.025: tau refuses 2.5% of held-out MNIST. FPR@95 = fraction of OOD input accepted
at the threshold that keeps 95% of MNIST test input.

Writes results/interp/model/ood_summary.csv (one row per seed x dataset x measure).

Usage (from the repo root):
    python experiments/RF_article/interp/ood_summary.py [--run results/ood_mnist/<run>]
"""
import argparse, csv, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ood_auroc import REPO, RUN, PRETTY, group_entropy, mahalanobis, auroc  # noqa: E402

ALPHA = 0.025


def fpr95(s_id, s_ood):
    """Higher score = more novel. OOD accepted at the threshold keeping 95% of ID."""
    return float(np.mean(s_ood <= np.quantile(s_id, 0.95)))


def rejected(s_cal, s):
    """Fraction of s above the threshold that refuses ALPHA of the calibration scores."""
    return float(np.mean(s > np.quantile(s_cal, 1 - ALPHA)))


def seed_rows(run):
    z = np.load(os.path.join(run, "uncertainty_features.npz"))
    ood = [k[len("X_ood_"):] for k in z.files if k.startswith("X_ood_")]
    assign = z["assignment"]; ng = int(assign.max()) + 1
    Xc, yc, Xt = z["X_cal"].astype(np.float64), z["y_cal"], z["X_test"].astype(np.float64)
    ent = {r["ood"]: r for r in json.load(open(os.path.join(run, "ood.json")))[0]["rows"]
           if r["statistic"] == "entropy"}
    pix = json.load(open(os.path.join(run, "input_density.json")))["pixels"]

    g_cal, g_id = group_entropy(Xc, assign, ng), group_entropy(Xt, assign, ng)
    maha = mahalanobis(Xc[:2500], yc[:2500])
    m_cal, m_id = maha(Xc[2500:]), maha(Xt)
    maha_all = mahalanobis(Xc, yc)   # the footnote's contrast: fit and tau on the same data
    out = {"id": {"maha_frr_held_out": rejected(m_cal, m_id),
                  "maha_frr_same_data": rejected(maha_all(Xc), maha_all(Xt)),
                  "rate": float(Xt.mean())}}
    for o in ood:
        Xo = z[f"X_ood_{o}"].astype(np.float64)
        g_o, m_o = group_entropy(Xo, assign, ng), maha(Xo)
        out[o] = {
            "entropy_auroc": ent[o]["auroc_2sided"], "entropy_fpr95": ent[o]["fpr_at_95tpr"],
            "group_auroc": auroc(g_id, g_o), "group_rejected": rejected(g_cal, g_o),
            "maha_auroc": auroc(m_id, m_o), "maha_fpr95": fpr95(m_id, m_o),
            "maha_rejected": rejected(m_cal, m_o),
            "pixel_auroc": pix["pca_residual"][o]["auroc_2sided"],
            "pixel_rejected": pix["pca_residual"][o]["at_alpha"][str(ALPHA)]["ood_rejected"],
            "intensity_auroc": pix["mean_intensity"][o]["auroc_2sided"],
            "rate": float(Xo.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "interp", "model",
                                                  "ood_summary.csv"))
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.run, "mnist_ood_s*")))
    seeds = {os.path.basename(d): seed_rows(d) for d in dirs}
    rows = [{"seed": s, "dataset": ds, "measure": m, "value": v}
            for s, r in seeds.items() for ds, ms in r.items() for m, v in ms.items()]
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "dataset", "measure", "value"])
            w.writeheader(); w.writerows(rows)

    def mean(ds, m):
        return np.mean([r[ds][m] for r in seeds.values()])
    ood = [k for k in next(iter(seeds.values())) if k != "id"]
    measures = [m for m in next(iter(seeds.values()))[ood[0]]]
    print(f"{len(dirs)} seeds from {a.run}\n{'mean over seeds':24}"
          + "".join(f"{PRETTY.get(o, o):>11}" for o in ood) + f"{'mean':>11}")
    for m in measures:
        v = [mean(o, m) for o in ood]
        print(f"{m:24}" + "".join(f"{x:11.4f}" for x in v) + f"{np.mean(v):11.4f}")
    print(f"\nMNIST test mean rate                {mean('id', 'rate'):.5f}")
    print(f"Mahalanobis false rejection, tau held out   {mean('id', 'maha_frr_held_out'):.4f}")
    print(f"Mahalanobis false rejection, tau same data  {mean('id', 'maha_frr_same_data'):.4f}")
    if a.csv:
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
