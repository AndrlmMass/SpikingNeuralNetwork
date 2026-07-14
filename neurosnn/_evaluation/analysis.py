"""
Representation and diagnostic metrics for SNN experiments.

All functions operate on numpy arrays and have no side effects — they can be
called at any checkpoint without modifying model state.
"""
import numpy as np


def class_selectivity(X: np.ndarray, y: np.ndarray, K: int = 10) -> float:
    """Mean per-neuron class selectivity: 1 - H / log(K).

    1.0 = neuron fires exclusively for one class.
    0.0 = uniform response across all classes (maximum entropy).
    Only active neurons (nonzero total rate) contribute to the mean.
    """
    means = np.stack([
        X[y == c].mean(0) if (y == c).any() else np.zeros(X.shape[1])
        for c in range(K)
    ])                                  # (K, N_exc)
    tot = means.sum(0)                  # (N_exc,)
    p = means / (tot + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum(0)
    sel = 1.0 - H / np.log(K)
    active = tot > 1e-9
    return float(sel[active].mean()) if active.any() else 0.0


def orientation_coherence(W_se: np.ndarray) -> float:
    """Energy-weighted mean structure-tensor coherence of W_se columns.

    Each column is treated as a 2D RF patch (square, inferred from row count).
    Coherence = |lambda1 - lambda2| / (lambda1 + lambda2): 1 = perfectly
    oriented edge, 0 = isotropic / blob.
    """
    N_x, N = W_se.shape
    side = int(round(np.sqrt(N_x)))
    cohs, ens = [], []
    for i in range(N):
        rf = W_se[:, i].reshape(side, side)
        e = np.abs(rf).sum()
        if e < 1e-9:
            continue
        gx = np.gradient(rf, axis=1)
        gy = np.gradient(rf, axis=0)
        Jxx = (gx * gx).mean()
        Jyy = (gy * gy).mean()
        Jxy = (gx * gy).mean()
        den = Jxx + Jyy + 1e-12
        cohs.append(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / den)
        ens.append(e)
    if not cohs:
        return 0.0
    return float(np.average(cohs, weights=ens))


def current_decomp(
    W_se: np.ndarray,
    W_ee: np.ndarray,
    W_ie: np.ndarray,
    exc_rate: np.ndarray,
    input_rate: np.ndarray,
) -> tuple[float, float, float]:
    """Mean absolute drive per exc neuron from feedforward, recurrent, and inhibitory pathways.

    Returns (se, ee, ie) — each is the mean |W @ rate| across exc neurons.
    Inhibitory rate is approximated as exc_rate via the 1:1 E→I coupling.
    """
    se = float(np.abs(W_se.T @ input_rate).mean())
    ee = float(np.abs(W_ee.T @ exc_rate).mean())
    ie = float(np.abs(W_ie.T @ exc_rate).mean())
    return se, ee, ie


def w_floor_frac(W_se: np.ndarray, floor: float = 0.02) -> float:
    """Fraction of nonzero W_se synapses at or below `floor` (weight floor effect)."""
    nz = W_se[W_se != 0]
    return float((np.abs(nz) <= floor).mean()) if nz.size else 0.0


def pool_by_label(
    X: np.ndarray,
    y: np.ndarray,
    neuron_class: np.ndarray,
    K: int = 10,
) -> float:
    """Reward-STDP readout: predict argmax over per-class summed exc firing rates.

    No fitted classifier — directly measures whether reward-STDP built
    class-selective groups. X: (n_items, N_exc), neuron_class: (N_exc,) int.
    """
    scores = np.zeros((X.shape[0], K))
    for c in range(K):
        m = neuron_class == c
        if m.any():
            scores[:, c] = X[:, m].sum(1)
    return float((scores.argmax(1) == y).mean())


def coverage_stats(X: np.ndarray) -> tuple[float, float, float]:
    """Monopolization diagnostics from val firing rates X (n_items, N_exc).

    Returns:
        dead_frac      — fraction of exc neurons with ~0 total activity.
        frac_ever_win  — fraction that are top responder for at least one item.
        winner_entropy — normalized entropy of argmax-winner distribution
                         (1 = every neuron wins equally, →0 = monopolized).
    """
    N = X.shape[1]
    tot = X.sum(0)
    dead_frac = float((tot <= 1e-9).mean())
    active_items = X.sum(1) > 1e-9
    if not active_items.any():
        return dead_frac, 0.0, 0.0
    winners = X[active_items].argmax(1)
    frac_ever = float(len(np.unique(winners)) / N)
    counts = np.bincount(winners, minlength=N).astype(float)
    p = counts[counts > 0] / counts.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(N)) if N > 1 else 0.0
    return dead_frac, frac_ever, ent


def softmax_readout(
    X: np.ndarray,
    y: np.ndarray,
    group_assignment: np.ndarray,
    n_groups: int = 10,
    T: float = 1.0,
) -> tuple[float, float]:
    """Group-pooled softmax accuracy and mean cross-entropy (eval-only, no backprop).

    Pools mean firing rate per group → softmax → argmax vs true label.
    Mean (not sum) normalizes for unequal group sizes.

    Returns (accuracy, cross_entropy_loss).
    """
    group_rates = np.stack(
        [X[:, group_assignment == g].mean(1) for g in range(n_groups)], axis=1
    )                                               # (n_items, n_groups)
    logits = group_rates / T
    logits -= logits.max(axis=1, keepdims=True)    # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    acc = float((probs.argmax(1) == y).mean())
    ce = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
    return acc, ce
