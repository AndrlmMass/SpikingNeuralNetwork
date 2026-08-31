"""
Is this input even the kind of thing the network was trained on? (density-based rejection)

The second, INDEPENDENT abstention channel. `threshold.py` thresholds the network's own
CONFIDENCE: it catches items the network is torn about. This module models p(x) on the
training distribution and rejects inputs in its low-likelihood tail -- Andreas' "if this
sample is in the bottom 2.5% of likelihood, do not judge it".

The two channels fail differently, which is the whole reason to have both:

  confidence  catches AMBIGUOUS input (a 4 that looks like a 9). Blind to a confidently
              wrong answer on something it has never seen -- a network can be certain and
              alien at the same time.
  density     catches NOVEL input (a shoe when you were trained on digits). Blind to
              in-distribution confusions: a hard 4 has perfectly ordinary density.

Fitted in two spaces, and the difference between them is a result, not an implementation
detail:

  pixels    PCA + Gaussian on the raw 28x28 images. This EXCLUDES THE NETWORK ENTIRELY.
            It is the baseline that has to be beaten: if rejecting on raw pixel
            statistics matches rejecting on the network's representation, then the
            "network detects novelty" claim is really "MNIST and FMNIST have different
            brightness", and that must be reported, not buried. MNIST fill fraction is
            0.153 against a much denser FMNIST, so this baseline will do WELL on
            FMNIST/SVHN for entirely uninteresting reasons -- which is exactly why
            KMNIST and notMNIST (sparse strokes, alien semantics) are the honest tests.
  features  the same model on the network's exc-rate features. Whatever this buys over
            pixels is what the representation contributes.

Two density models, because the parametric one is definitely wrong:

  gaussian  PCA to k components + full-covariance Gaussian log-likelihood. Cheap, and
            the natural reading of "likelihood of coming from that sample". MNIST is not
            Gaussian in pixel space, so this is a convenient summary, not a truth claim.
  knn       mean distance to the k nearest training points in the same PCA space.
            Non-parametric, so it does not inherit the Gaussian assumption. When the two
            disagree, the assumption is what broke.

Both are turned into a rejection rule the same way as the confidence channel: a
one-sided threshold at the alpha-quantile of the score on HELD-OUT in-distribution data,
never on the training data the density model was fitted to (a fitted model is
over-confident about its own training points for the same reason a network is).

OpenMP note: with several harness cells already running, this script can abort with
"multiple copies of the OpenMP runtime have been linked into the program". The documented
workaround is KMP_DUPLICATE_LIB_OK=TRUE, which Intel flags as unsafe and possibly
silently wrong -- so it was checked rather than trusted: with the flag set, the entire
pixel-space table reproduces a clean single-process run digit for digit (gaussian 0.545 /
knn 0.869 / mean_intensity 0.123 / pca_residual 0.983 against FMNIST). Safe here; re-check
if the numbers ever move.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/RF_article/interp/input_density.py         --run <run_dir> --dataset mnist
"""
import argparse, json, os, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)                                   # so `neurosnn` imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty import auroc  # noqa: E402
ALPHAS = (0.01, 0.025, 0.05, 0.10, 0.20)
EPS = 1e-12


# ------------------------------------------------------------------- density models

class PCAGaussian:
    """PCA to `k` components, then a full-covariance Gaussian log-likelihood.

    Scores are HIGHER = more typical, matching the confidence convention everywhere else
    in this codebase, so the same "reject below tau" machinery applies unchanged.

    The out-of-subspace residual is kept as a separate diagnostic rather than folded in:
    an image can sit at a perfectly ordinary position within the top-k subspace while
    being wildly off it, and averaging the two hides which of the two is firing.
    """

    def __init__(self, k=50, shrinkage=1e-4):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        # economy SVD: n >> d here, and we only ever need the top-k right singulars
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.k, Vt.shape[0])
        self.components_ = Vt[:k]                      # (k, d)
        Z = Xc @ self.components_.T                    # (n, k)
        C = np.cov(Z, rowvar=False)
        C.flat[:: k + 1] += self.shrinkage * np.trace(C) / max(k, 1)   # ridge, keeps it PD
        self.prec_ = np.linalg.inv(C)
        sign, logdet = np.linalg.slogdet(C)
        self.logdet_ = float(logdet)
        self.k_ = k
        # residual scale, for the diagnostic below
        R = Xc - Z @ self.components_
        self.resid_scale_ = float(np.sqrt((R ** 2).sum(1).mean()) + EPS)
        return self

    def score(self, X):
        """Gaussian log-likelihood in the top-k subspace. Higher = more typical."""
        Xc = np.asarray(X, dtype=np.float64).reshape(len(X), -1) - self.mean_
        Z = Xc @ self.components_.T
        m = np.einsum("ij,jk,ik->i", Z, self.prec_, Z)          # Mahalanobis^2
        return -0.5 * (m + self.logdet_ + self.k_ * np.log(2 * np.pi))

    def residual(self, X):
        """RMS distance to the top-k subspace, normalized by the training scale.

        Reported next to the likelihood because a Gaussian fitted in a 50-D subspace of
        784-D space says nothing about the other 734 directions, and OOD input often
        lives precisely there.
        """
        Xc = np.asarray(X, dtype=np.float64).reshape(len(X), -1) - self.mean_
        R = Xc - (Xc @ self.components_.T) @ self.components_
        return np.sqrt((R ** 2).sum(1)) / self.resid_scale_


class PCAkNN:
    """Mean distance to the k nearest training points in a PCA subspace (negated).

    Non-parametric: it makes no distributional assumption, so where it and PCAGaussian
    disagree, the Gaussian assumption is what broke. Subsampled to `n_ref` reference
    points to keep the distance matrix affordable -- the score is a local density
    estimate, and a few thousand references already resolve it.
    """

    def __init__(self, k=20, n_components=50, n_ref=5000, seed=0):
        self.k = k
        self.n_components = n_components
        self.n_ref = n_ref
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        self.mean_ = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: min(self.n_components, Vt.shape[0])]
        Z = (X - self.mean_) @ self.components_.T
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Z), size=min(self.n_ref, len(Z)), replace=False)
        self.ref_ = Z[idx]
        self.ref_sq_ = (self.ref_ ** 2).sum(1)
        return self

    def score(self, X, batch=2000):
        Z = (np.asarray(X, dtype=np.float64).reshape(len(X), -1) - self.mean_) @ self.components_.T
        out = np.empty(len(Z))
        for i in range(0, len(Z), batch):
            zb = Z[i: i + batch]
            d2 = (zb ** 2).sum(1)[:, None] + self.ref_sq_[None, :] - 2.0 * zb @ self.ref_.T
            np.maximum(d2, 0.0, out=d2)
            part = np.partition(d2, self.k, axis=1)[:, : self.k]
            out[i: i + batch] = -np.sqrt(part).mean(1)          # negated: higher = denser
        return out


def mean_intensity(X):
    """The floor every network number must beat: how bright is the image, and nothing else.

    If this separates ID from OOD as well as the network does, then the OOD result is a
    brightness detector wearing a spiking network as a hat.
    """
    return np.asarray(X, dtype=np.float64).reshape(len(X), -1).mean(1)


# ------------------------------------------------------------------------- images

def load_images(dataset, pixel_size=28):
    """(train_images, test_images) as (n, 784) float arrays, through the model's own pipeline.

    Uses ImageDataStreamer so the grayscale + resize + normalization is byte-for-byte the
    transform the network is fed -- including the 32x32x3 -> 28x28 collapse for SVHN and
    CIFAR. A density model fitted on differently-preprocessed images would be measuring
    the preprocessing.
    """
    from neurosnn._data.get_data import ImageDataStreamer
    s = ImageDataStreamer(data_dir=os.path.join(REPO, "data"), pixel_size=pixel_size,
                          dataset=dataset)
    tr = np.asarray(s.train_images, dtype=np.float32)
    te = np.asarray(s.test_images, dtype=np.float32)
    return tr.reshape(len(tr), -1), te.reshape(len(te), -1)


# ------------------------------------------------------------------------ scoring

def threshold_at(score_cal, alpha):
    """One-sided lower-tail threshold: reject anything scoring below the alpha-quantile."""
    return float(np.quantile(np.asarray(score_cal, dtype=float), alpha))


def evaluate(score_cal, score_id, score_ood_by_name, alphas=ALPHAS):
    """Rejection rates at ID-calibrated thresholds, plus threshold-free AUROC.

    `score_cal` must be HELD-OUT in-distribution data -- never the data the density model
    was fitted on, which it is over-confident about for the same reason a network is
    over-confident on its training set.
    """
    rows = {}
    for name, s_ood in score_ood_by_name.items():
        lab = np.concatenate([np.ones(len(score_id), bool), np.zeros(len(s_ood), bool)])
        both = np.concatenate([score_id, s_ood])
        at = {}
        for a in alphas:
            tau = threshold_at(score_cal, a)
            # TWO-SIDED as well as lower-tail. A statistic can flag OOD by being too
            # HIGH: OOD images here are brighter and denser than MNIST, so a lower-tail
            # test on mean intensity points the wrong way entirely. The two-sided rule
            # spends alpha/2 on each tail, so its ID cost is the same alpha.
            lo = threshold_at(score_cal, a / 2.0)
            hi = threshold_at(score_cal, 1.0 - a / 2.0)
            at[float(a)] = dict(
                tau=tau,
                ood_rejected=float((s_ood < tau).mean()),      # what we want high
                id_rejected=float((score_id < tau).mean()),    # what it costs us
                tau_lo=lo, tau_hi=hi,
                ood_rejected_2s=float(((s_ood < lo) | (s_ood > hi)).mean()),
                id_rejected_2s=float(((score_id < lo) | (score_id > hi)).mean()),
            )
        # FPR at 95% TPR: of the OOD items, what fraction slips through a threshold that
        # keeps 95% of ID input. The standard OOD number, comparable across papers.
        tau95 = float(np.quantile(score_id, 0.05))
        a_signed = float(auroc(both, lab))
        rows[name] = dict(
            # SIGNED: >0.5 means OOD scores LOW (the lower tail detects it), <0.5 means
            # OOD scores HIGH. A value far below 0.5 is not a failure -- it is a detector
            # pointing the other way, and reporting only the signed number understates it.
            auroc=a_signed,
            auroc_2sided=float(max(a_signed, 1.0 - a_signed)),
            tail=("lower" if a_signed >= 0.5 else "upper"),
            fpr_at_95tpr=float((s_ood >= tau95).mean()),
            n_ood=int(len(s_ood)),
            at_alpha=at,
        )
    return rows


def build_scores(fit_X, cal_X, id_X, ood_X_by_name, k_pca=50, seed=0):
    """Fit every density model on `fit_X`, score all four sets. Returns {model: {...}}."""
    models = {
        "gaussian": PCAGaussian(k=k_pca).fit(fit_X),
        "knn": PCAkNN(n_components=k_pca, seed=seed).fit(fit_X),
    }
    out = {}
    for name, m in models.items():
        out[name] = dict(
            cal=m.score(cal_X), id=m.score(id_X),
            ood={k: m.score(v) for k, v in ood_X_by_name.items()})
    # the no-network floor
    out["mean_intensity"] = dict(
        cal=mean_intensity(cal_X), id=mean_intensity(id_X),
        ood={k: mean_intensity(v) for k, v in ood_X_by_name.items()})
    # subspace residual, as a diagnostic detector in its own right (negated so higher =
    # more typical, like everything else)
    g = models["gaussian"]
    out["pca_residual"] = dict(
        cal=-g.residual(cal_X), id=-g.residual(id_X),
        ood={k: -g.residual(v) for k, v in ood_X_by_name.items()})
    return out


def format_table(space, results):
    L = [f"[density] space={space}",
         f"  {'model':<15} {'ood set':<10} {'AUROC':>7} {'|2-sided|':>9} {'tail':>6} "
         f"{'rej@2.5%':>9} {'id_cost':>8} {'2s rej':>8} {'2s cost':>8}"]
    for model, per_ood in results.items():
        for name, r in per_ood.items():
            a25 = r["at_alpha"][0.025]
            L.append(f"  {model:<15} {name:<10} {r['auroc']:>7.3f} "
                     f"{r['auroc_2sided']:>9.3f} {r['tail']:>6} "
                     f"{a25['ood_rejected']:>9.3f} {a25['id_rejected']:>8.3f} "
                     f"{a25['ood_rejected_2s']:>8.3f} {a25['id_rejected_2s']:>8.3f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", help="the IN-DISTRIBUTION dataset")
    ap.add_argument("--ood", action="append", default=[],
                    help="repeatable OOD dataset name")
    ap.add_argument("--run", default=None,
                    help="run dir whose uncertainty_features.npz supplies the FEATURE "
                         "space (X_cal / X_test / X_ood_*); omit for pixels only")
    ap.add_argument("--k-pca", type=int, default=50)
    ap.add_argument("--fit-n", type=int, default=20000,
                    help="training images the density model is fitted on")
    ap.add_argument("--cal-n", type=int, default=5000,
                    help="HELD-OUT in-distribution images used to place the threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if not a.ood:
        a.ood = ["fmnist", "kmnist", "notmnist", "svhn"]
    a.ood = [d for d in a.ood if d != a.dataset]

    report = {"dataset": a.dataset, "ood": a.ood, "k_pca": a.k_pca}

    # ---- pixel space: the network-free baseline ---------------------------------
    tr, te = load_images(a.dataset)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(tr))
    fit_i, cal_i = perm[: a.fit_n], perm[a.fit_n: a.fit_n + a.cal_n]
    ood_imgs = {}
    for d in a.ood:
        try:
            _, o_te = load_images(d)
            ood_imgs[d] = o_te
        except Exception as e:
            print(f"[density] {d}: load failed ({type(e).__name__}: {e})", flush=True)
    px = build_scores(tr[fit_i], tr[cal_i], te, ood_imgs, a.k_pca, a.seed)
    report["pixels"] = {m: evaluate(v["cal"], v["id"], v["ood"]) for m, v in px.items()}
    print(format_table("pixels (NO NETWORK -- the floor to beat)", report["pixels"]), flush=True)

    # ---- feature space: what the network's representation adds ------------------
    if a.run:
        p = a.run if not os.path.isdir(a.run) else os.path.join(a.run, "uncertainty_features.npz")
        d = np.load(p, allow_pickle=False)
        ood_feats = {k[len("X_ood_"):]: d[k] for k in d.files if k.startswith("X_ood_")}
        if not ood_feats:
            print("[density] run has no X_ood_* arrays -- rerun the harness with "
                  "--ood-dataset to get the feature-space comparison", flush=True)
        else:
            # X_cal is the held-out calibration split; X_test is the ID evaluation set.
            # The density model is fitted on X_cal too, so we split it: fit on the first
            # half, calibrate the threshold on the second. Fitting and calibrating on the
            # same points would put tau too low, exactly as calibrating on train would.
            Xc = np.asarray(d["X_cal"])
            cut = len(Xc) // 2
            ft = build_scores(Xc[:cut], Xc[cut:], np.asarray(d["X_test"]),
                              ood_feats, a.k_pca, a.seed)
            report["features"] = {m: evaluate(v["cal"], v["id"], v["ood"])
                                  for m, v in ft.items()}
            print(format_table("network features", report["features"]), flush=True)

    out = a.json_out or (os.path.join(a.run, "input_density.json") if a.run
                         else f"input_density_{a.dataset}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[density] wrote {out}")


if __name__ == "__main__":
    main()
