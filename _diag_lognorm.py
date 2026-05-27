"""
Diagnostic: explore log-normal RF size diversity for W_se and W_ee.
Uses WeightFactory directly — no training, no dataset needed.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from neurosnn._network.init_weights import WeightFactory


def make_and_build(lognorm_se_std=0.0, lognorm_ee_std=0.0,
                   lognorm_se_mean=3.0, lognorm_ee_mean=1.0, seed=0):
    N_x, N_exc, N_inh = 784, 1024, 225
    N = N_x + N_exc + N_inh
    rng = np.random.default_rng(seed)
    fac = WeightFactory(
        N=N, N_x=N_x, N_exc=N_exc, N_inh=N_inh, rng=rng,
        w_dense_se=0.05, w_dense_ee=0.02, w_dense_ei=0.03, w_dense_ie=0.05,
        se_weights=1.0, ee_weights=0.5, ei_weights=1.0, ie_weights=-0.7,
        random_weights=False, rf_scale=1.0, oriented_rf=True,
        sigma_x=lognorm_se_mean, gamma=0.4, n_orientations=4, r_cut_factor=3.0,
        sigma_x_lognormal_std=lognorm_se_std,
        sigma_ee_mean=lognorm_ee_mean,
        sigma_ee_lognormal_std=lognorm_ee_std,
    )
    fac.build()
    return fac


def analyse(fac):
    W = fac.weights
    H, W_in = fac.H, fac.W        # input grid (28x28)
    H_e, W_e = fac.H_e, fac.W_e  # exc grid (32x32)
    st, ex = fac.st, fac.ex

    W_se = W[:st, st:ex]   # (784, 1024)  — col = incoming weights for one E neuron
    W_ee = W[st:ex, st:ex] # (1024, 1024) — row = outgoing weights for one E neuron

    # W_se: effective RF sigma per E-neuron (column-wise weighted std)
    coords = np.array([(r, c) for r in range(H) for c in range(W_in)], dtype=float)
    w_sum = W_se.sum(axis=0)  # (N_exc,)
    safe = w_sum > 0
    cx = (coords[:, 0:1] * W_se).sum(axis=0) / (w_sum + 1e-12)
    cy = (coords[:, 1:2] * W_se).sum(axis=0) / (w_sum + 1e-12)
    d2 = (coords[:, 0:1] - cx[None, :])**2 + (coords[:, 1:2] - cy[None, :])**2
    rf_sigma = np.sqrt((d2 * W_se).sum(axis=0) / (w_sum + 1e-12))[safe]

    # W_ee: effective connection sigma per E-neuron (row-wise weighted std)
    coords_e = np.array([(r, c) for r in range(H_e) for c in range(W_e)], dtype=float)
    ee_sum = W_ee.sum(axis=1)   # (N_exc,)
    emask = ee_sum > 0
    cx_e = (coords_e[:, 0] @ W_ee.T) / (ee_sum + 1e-12)
    cy_e = (coords_e[:, 1] @ W_ee.T) / (ee_sum + 1e-12)
    d2_e = (coords_e[:, 0:1] - cx_e[None, :])**2 + (coords_e[:, 1:2] - cy_e[None, :])**2
    ee_sigma = np.sqrt((d2_e * W_ee.T).sum(axis=0) / (ee_sum + 1e-12))[emask]

    return rf_sigma, ee_sigma


def fmt5(a):
    ps = np.percentile(a, [0, 25, 50, 75, 100])
    cv = a.std() / (a.mean() + 1e-9)
    return f"{ps[0]:.2f}/{ps[1]:.2f}/{ps[2]:.2f}/{ps[3]:.2f}/{ps[4]:.2f}  CV={cv:.2f}"


configs = [
    dict(lognorm_se_std=0.0, lognorm_ee_std=0.0),
    dict(lognorm_se_std=1.0, lognorm_ee_std=0.0),
    dict(lognorm_se_std=1.5, lognorm_ee_std=0.0),
    dict(lognorm_se_std=2.0, lognorm_ee_std=0.0),
    dict(lognorm_se_std=1.5, lognorm_ee_std=0.3),
    dict(lognorm_se_std=1.5, lognorm_ee_std=0.5),
    dict(lognorm_se_std=1.5, lognorm_ee_std=0.8),
]

print()
print(f"{'se_std':>6}  {'ee_std':>6}  "
      f"W_se σ (min/p25/med/p75/max  CV)          "
      f"W_ee σ (min/p25/med/p75/max  CV)")
print("-" * 105)

for cfg in configs:
    fac = make_and_build(**cfg)
    rf_s, ee_s = analyse(fac)
    print(f"{cfg['lognorm_se_std']:>6.1f}  {cfg['lognorm_ee_std']:>6.1f}  "
          f"{fmt5(rf_s):<42}  {fmt5(ee_s)}")

print()
