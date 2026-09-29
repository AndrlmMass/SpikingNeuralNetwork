"""n steps of the power-law pull == one step at lambda^n, on the real sleep_func.

The one-shot arm rests entirely on this identity, so it is checked against the
production kernel (signs, per-post targets, nonzero masks and all) rather than
against a reimplementation of the algebra.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weight_funcs import sleep_func  # noqa: E402


def _call(weights, rate_exc, rate_inh, idx, targets):
    nz_rows_exc, nz_cols_exc, nz_rows_inh, nz_cols_inh, nz_rows, nz_cols = idx
    t_exc, t_inh = targets
    w, _, _ = sleep_func(
        weights=weights, max_sum=np.inf, max_sum_exc=np.inf, max_sum_inh=np.inf,
        sleep_now_inh=True, sleep_now_exc=True,
        w_target_exc=t_exc, w_target_inh=t_inh,
        use_post_targets=False,
        w_target_exc_vec=np.zeros(1), w_target_inh_vec=np.zeros(1),
        weight_decay_rate_exc=rate_exc, weight_decay_rate_inh=rate_inh,
        baseline_sum_exc=-np.inf, baseline_sum_inh=-np.inf,
        sleep_synchronized=False,
        nz_rows=nz_rows, nz_cols=nz_cols, baseline_sum=-np.inf,
        nz_rows_exc=nz_rows_exc, nz_rows_inh=nz_rows_inh,
        nz_cols_exc=nz_cols_exc, nz_cols_inh=nz_cols_inh,
    )
    return w


def test_n_steps_equals_one_step_at_lambda_to_the_n():
    rng = np.random.default_rng(0)
    n_pre, n_post, n_exc = 60, 40, 25
    lam, n = 0.9997, 3500
    t_exc, t_inh = 0.01, -0.01

    W0 = np.zeros((n_pre, n_post))
    W0[:n_exc] = np.abs(rng.lognormal(-1.0, 0.8, (n_exc, n_post)))
    W0[n_exc:] = -np.abs(rng.lognormal(-1.0, 0.8, (n_pre - n_exc, n_post)))
    W0 *= rng.random(W0.shape) > 0.3          # sparsity, so the masks matter

    re_, ce_ = np.nonzero(W0[:n_exc])
    ri_, ci_ = np.nonzero(W0[n_exc:]); ri_ = ri_ + n_exc
    idx = (re_, ce_, ri_, ci_, *np.nonzero(W0))
    tgt = (t_exc, t_inh)

    looped = W0.copy()
    for _ in range(n):
        looped = _call(looped, lam, lam, idx, tgt)
    oneshot = _call(W0.copy(), lam ** n, lam ** n, idx, tgt)

    err = np.abs(looped - oneshot).max()
    assert err < 1e-12, f"one-shot deviates from {n} looped steps by {err:.3e}"

    # and the identity's two structural consequences
    mask = W0 != 0
    assert np.all(np.sign(oneshot[mask]) == np.sign(W0[mask])), "sign flipped"
    from scipy.stats import spearmanr
    rho = spearmanr(np.abs(W0[:n_exc][W0[:n_exc] != 0]),
                    np.abs(oneshot[:n_exc][W0[:n_exc] != 0])).correlation
    assert rho > 1 - 1e-12, f"rank not preserved (spearman={rho})"


if __name__ == "__main__":
    test_n_steps_equals_one_step_at_lambda_to_the_n()
    print("OK: one-shot == looped, to machine precision")
