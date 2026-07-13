"""
Mechanism harness (single config): track WHAT learning does to the representation
on the oriented-RF prior vs a random prior, feedforward vs recurrent, under
frozen / trace-STDP / triplet-STDP. All configs use one-to-one WTA (inhibition
held constant for this run).

At every val checkpoint we snapshot the live weights (TrainResult.weights) and the
val features (per-item exc firing rates, captured from the Evaluator) and compute:

  1. class_selectivity  — per-neuron peakedness of the class-response (1 - H/logK);
     high = each neuron specialises to a class (the D&C objective).
  2. orientation_coherence — structure-tensor coherence of each RF (W_se column);
     high = oriented / edge-like. Tracks whether STDP erodes the oriented prior.
  3. readout drift — accuracy of a readout FROZEN at init vs a readout REFIT each
     checkpoint. A growing gap = the representation is drifting off the class-useful
     directions it started on.
  4. current decomposition — mean |SE| (feedforward) vs |EE| (recurrent) vs |IE|
     drive per exc neuron; tests whether recurrence is even influential.

Also saves RF snapshot grids (first / mid / last) and a weight-floor fraction.

Usage:
  python experiments/RF_article/interp/interp_harness.py --prior oriented --rule trace --ee --tag B2 --output-dir <dir>
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import neurosnn as snn
from neurosnn._evaluation import evaluation as evalmod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAP = {"rows": []}
_orig_score = evalmod.Evaluator.score
def _cap(self, X, Y):
    CAP["rows"].append((np.asarray(X).copy(), np.asarray(Y).copy()))
    return _orig_score(self, X, Y)
evalmod.Evaluator.score = _cap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SIDE = 28


# ---------------- metrics ----------------
def class_selectivity(X, y, K=10):
    means = np.stack([X[y == c].mean(0) if (y == c).any() else np.zeros(X.shape[1])
                      for c in range(K)])           # (K, N)
    tot = means.sum(0)                               # (N,)
    p = means / (tot + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum(0)              # (N,)
    sel = 1.0 - H / np.log(K)
    active = tot > 1e-9
    return float(sel[active].mean()) if active.any() else 0.0


def orientation_coherence(W_se):
    # W_se: (N_x, N_exc); each column is a 28x28 RF
    N = W_se.shape[1]
    cohs, ens = [], []
    for i in range(N):
        rf = W_se[:, i].reshape(SIDE, SIDE)
        e = np.abs(rf).sum()
        if e < 1e-9:
            continue
        gx = np.gradient(rf, axis=1); gy = np.gradient(rf, axis=0)
        Jxx = (gx * gx).mean(); Jyy = (gy * gy).mean(); Jxy = (gx * gy).mean()
        den = Jxx + Jyy + 1e-12
        cohs.append(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / den)
        ens.append(e)
    if not cohs:
        return 0.0
    return float(np.average(cohs, weights=ens))     # energy-weighted mean coherence


def current_decomp(W_se, W_ee, W_ie, exc_rate, input_rate):
    # mean absolute drive per exc neuron from each pathway
    se = np.abs(W_se.T @ input_rate).mean()          # feedforward
    ee = np.abs(W_ee.T @ exc_rate).mean()            # recurrent
    ie = np.abs(W_ie.T @ exc_rate).mean()            # inhib (inh_rate ~ exc_rate via 1:1 E->I)
    return float(se), float(ee), float(ie)


def w_floor_frac(W_se, floor=0.02):
    nz = W_se[W_se != 0]
    return float((np.abs(nz) <= floor).mean()) if nz.size else 0.0


# ---------------- readout drift ----------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def fit_clf(X, y, seed=0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(penalty="l1", solver="saga", max_iter=400, n_jobs=-1,
                             random_state=seed).fit(np.nan_to_num(sc.transform(X)), y)
    return sc, clf

def score_clf(sc, clf, X, y):
    return accuracy_score(y, clf.predict(np.nan_to_num(sc.transform(X))))


# ---------------- weights blocks ----------------
def blocks(weights, st, ex, ih):
    return (weights[:st, st:ex], weights[st:ex, st:ex], weights[ex:ih, st:ex])


def save_rf_grid(W_se, path, n=64):
    idx = np.linspace(0, W_se.shape[1] - 1, n).astype(int)
    s = int(np.sqrt(n))
    fig, axes = plt.subplots(s, s, figsize=(s, s))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(W_se[:, i].reshape(SIDE, SIDE), cmap="RdBu_r"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(path, dpi=80); plt.close(fig)


def load_input_rate():
    from torchvision import datasets
    ds = datasets.MNIST(root=os.path.join(REPO, "data", "torchvision"), train=False, download=False)
    m = ds.data.numpy().reshape(len(ds), -1).mean(0).astype(np.float64) / 255.0
    return m  # (784,) mean pixel intensity ~ mean input rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--prior", choices=["oriented", "random"], required=True)
    p.add_argument("--rule", choices=["frozen", "trace", "triplet"], required=True)
    p.add_argument("--ee", action="store_true", help="enable E->E recurrence (default off=feedforward)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-all", type=int, default=15000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=3)
    p.add_argument("--test-all", type=int, default=3000)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    N_exc, N_inh = 1024, 1024   # WTA: N_inh == N_exc
    st, ex, ih = 784, 784 + N_exc, 784 + N_exc + N_inh
    train_weights = a.rule != "frozen"
    # recurrence controlled by EE density (0 = feedforward, D&C-style; 0.01 = recurrent)
    density_ee = 0.01 if a.ee else 0.0

    wkw = dict(density_se=0.01, density_ee=density_ee, density_ei=0.03, density_ie=0.05,
               peak_se=4.0, peak_ee=1.0, peak_ei=20.0, peak_ie=-2.0,
               wta_inhibition=True)
    if a.prior == "oriented":
        weights = snn.weights.oriented_receptive_fields(n_orientations=4, orientation_mode="block", **wkw)
    else:
        weights = snn.weights.random(**wkw)

    layer = snn.Layer(N_exc=N_exc, N_inh=N_inh, membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
        resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
        min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
        spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5), weights=weights)

    if a.rule == "triplet":
        learner = snn.learner.TripletSTDP()
    else:
        learner = snn.learner.TraceSTDP(learning_rate=0.0004, tau_trace=20, w_max=10.0,
            mu_weight=0.5, x_tar_mode="mean", update_freq=100, clip_weights=True,
            min_weight_exc=0.01, max_weight_exc=25.0, min_weight_inh=-25.0, max_weight_inh=-0.01)
    reg = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    model = snn.Model(input_size=784, classes=list(range(10)), random_state=a.seed, num_steps=350,
        all_images_train=a.train_all, batch_image_train=1000, all_images_val=a.val_all,
        batch_image_val=a.val_all, all_images_test=a.test_all, batch_image_test=a.test_all,
        image_dataset="mnist", max_rate_hz=90.0, gain=1.0, gabor=False)

    input_rate = load_input_rate()
    cfg = dict(tag=a.tag, prior=a.prior, rule=a.rule, ee=a.ee, wta=True,
               n_exc=N_exc, n_inh=N_inh, train_all=a.train_all, seed=a.seed)
    print(f"\n[{a.tag}] prior={a.prior} rule={a.rule} ee={a.ee} train_weights={train_weights}\n", flush=True)

    traj = []
    fixed = {"sc": None, "clf": None}
    rf_saved = {}

    def checkpoint(batch, weights):
        # capture val features fresh
        CAP["rows"].clear()
        v = model.validate()
        if not CAP["rows"]:
            return
        X = np.concatenate([x for x, _ in CAP["rows"]], 0)
        y = np.concatenate([yy for _, yy in CAP["rows"]], 0).astype(int)
        W_se, W_ee, W_ie = blocks(weights, st, ex, ih)
        exc_rate = X.mean(0)
        se, ee, ie = current_decomp(W_se, W_ee, W_ie, exc_rate, input_rate)
        # readout drift: split val pool 70/30
        rng = np.random.default_rng(0); idx = rng.permutation(len(y))
        cut = int(0.7 * len(y)); tr, te = idx[:cut], idx[cut:]
        sc, clf = fit_clf(X[tr], y[tr], a.seed)
        refit = score_clf(sc, clf, X[te], y[te])
        if fixed["clf"] is None:
            fixed["sc"], fixed["clf"] = sc, clf
        fixed_acc = score_clf(fixed["sc"], fixed["clf"], X[te], y[te])
        rec = dict(batch=int(batch),
                   val_acc=float(v.accuracy) if v.accuracy is not None else float("nan"),
                   val_phi=float(v.phi) if v.phi is not None else float("nan"),
                   selectivity=class_selectivity(X, y),
                   orient_coh=orientation_coherence(W_se),
                   refit_acc=float(refit), fixed_acc=float(fixed_acc),
                   cur_se=se, cur_ee=ee, cur_ie=ie, ee_se_ratio=ee / (se + 1e-12),
                   w_floor_frac=w_floor_frac(W_se))
        traj.append(rec)
        print(f"  [{a.tag}] b{batch:>3} val {rec['val_acc']:.3f} sel {rec['selectivity']:.3f} "
              f"coh {rec['orient_coh']:.3f} refit {refit:.3f} fixed {fixed_acc:.3f} "
              f"EE/SE {rec['ee_se_ratio']:.3f}", flush=True)
        # RF snapshots: first & last
        key = "first" if "first" not in rf_saved else "last"
        save_rf_grid(W_se, os.path.join(a.output_dir, f"rf_{key}.png"))
        rf_saved[key] = batch

    last_w = None
    for r in model.train(layers=[layer], learner=learner, regularizer=reg, epochs=1,
                         train_weights=train_weights, save_model=False, accuracy_method="pca_lr",
                         use_LR=True, use_phi=True, use_pca=False, track_stats=False):
        last_w = r.weights if r.weights is not None else last_w
        if r.accuracy is not None and r.batch % a.val_every == 0 and last_w is not None:
            checkpoint(r.batch, last_w)

    # ensure at least one checkpoint (frozen single-batch)
    if not traj and last_w is not None:
        checkpoint(0, last_w)

    test = model.test()
    out = dict(config=cfg, test_acc=float(test.accuracy) if test.accuracy is not None else float("nan"),
               test_phi=float(test.phi) if test.phi is not None else float("nan"), trajectory=traj)
    with open(os.path.join(a.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{a.tag}] DONE test_acc={out['test_acc']:.3f} -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
