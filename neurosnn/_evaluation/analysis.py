"""
Representation and diagnostic metrics for SNN experiments.

All functions operate on numpy arrays and have no side effects — they can be
called at any checkpoint without modifying model state.
"""
import numpy as np


def class_selectivity(X: np.ndarray, y: np.ndarray, K: int = 10) -> float:
    """DEPRECATED — confounded by firing rate. Use `class_eta_squared` instead.

    Mean per-neuron class selectivity: 1 - H / log(K).
    1.0 = neuron fires exclusively for one class.
    0.0 = uniform response across all classes (maximum entropy).

    WHY IT IS BROKEN: normalizing each neuron's class-means into a distribution
    discards magnitude, so (3e-4, 0, ..., 0) and (0.5, 0, ..., 0) both score 1.0.
    The `tot > 1e-9` guard only drops EXACTLY dead neurons, not near-silent ones,
    whose per-class means are sampling noise and therefore peaky by chance. It
    cannot distinguish "specialist for class 7" from "fired three times, all 7s".
    See `active_mask` for the measured effect sizes.

    Kept only so old trajectories stay comparable. Do not use for new claims.
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


def active_mask(X: np.ndarray, min_rel: float = 0.02) -> np.ndarray:
    """Neurons carrying enough evidence to estimate anything from. (N_exc,) bool.

    Keeps neurons whose TOTAL rate is at least `min_rel` times the 90th-percentile
    total rate — i.e. within ~50x of the strongly-driven population. Relative and
    scale-free, so it does not depend on timestep count, item count, or sparsity
    regime; and it targets total evidence, which is what sets estimator noise.

    Needed because `class_selectivity` is CONFOUNDED by firing rate: it keeps any
    neuron with total rate > 1e-9, and a neuron firing at ~3e-4 has per-class means
    that are pure sampling noise, hence a peaky class distribution BY CHANCE.
    Measured on a real run: the least-active quartile of "live" neurons scores 0.65
    selectivity while the most-active quartile scores 0.045 — 15x higher for firing
    ~500x less. Across a 12-cell sweep, selectivity correlated +0.91 with dead
    fraction, and its correlation with linear decodability vanished (-0.74 -> -0.12)
    once dead fraction was partialled out. It was measuring silence, not structure.

    Threshold chosen empirically against a known answer: on a synthetic population
    of 40 informative + 40 near-silent noise neurons it keeps exactly the 40, and on
    real features it drops the entire low-activity quartile that inflated the old
    metric. A "fired on >=1% of items" rule kept 79/80 of the synthetic case and let
    52/142 of the bad real neurons through — too permissive to be worth having.
    """
    tot = np.asarray(X, dtype=float).sum(0)
    ref = np.percentile(tot, 90)
    return tot >= min_rel * ref if ref > 0 else np.zeros(tot.shape, bool)


def class_eta_squared(
    X: np.ndarray,
    y: np.ndarray,
    K: int = 10,
    min_rel: float = 0.02,
) -> float:
    """Mean PER-NEURON eta-squared: between-class variance / total variance.

    The replacement for `class_selectivity`. Same question ("is this neuron's
    response class-informative?") but noise-aware BY CONSTRUCTION: a barely-firing
    neuron has almost all its variance *within* class, so it scores ~0 instead of
    ~0.65. Being a variance ratio, absolute firing magnitude cancels, so it is not
    confounded the other way either. Verified on real features: low-activity
    quartile 0.021 vs high-activity 0.279 — the correct direction, where
    class_selectivity gave 0.651 vs 0.044.

    1.0 = the neuron's response is fully determined by class; 0.0 = class explains
    none of its variance. Averaged over active neurons only.
    """
    X = np.asarray(X, dtype=float)
    m = active_mask(X, min_rel)
    if not m.any():
        return 0.0
    Xa = X[:, m]
    gm = Xa.mean(0)
    bcss = np.zeros(Xa.shape[1])
    for c in range(K):
        sel = y == c
        if sel.any():
            bcss += sel.sum() * (Xa[sel].mean(0) - gm) ** 2
    tcss = ((Xa - gm) ** 2).sum(0)
    return float(np.mean(bcss / (tcss + 1e-12)))


def response_correlation(
    X: np.ndarray,
    group_assignment: "np.ndarray | None" = None,
    n_groups: int = 10,
    min_rel: float = 0.02,
) -> tuple[float, float]:
    """Mean |pairwise correlation| between neurons' response vectors.

    THE redundancy measure: high = neurons carry the same signal, so the population
    is low-rank however selective the individuals look. This is the decorrelation
    objective from the sparse-coding literature, and unlike class_selectivity it is
    a property of the POPULATION — ten identical perfectly-tuned neurons score 1.0
    here (correctly: they are redundant) while class_selectivity would also score
    1.0 (incorrectly: it reads them as ten fine specialists).

    CORRELATION, not raw cosine, because correlation is the *centered* cosine. Our
    population has a large common mode ("bright image -> everyone fires more") worth
    ~34% of variance, which is luminance rather than redundancy we could fix. On real
    features the uncentered version reads 0.82 among active neurons vs 0.57 centered.

    Returns (within_group, overall). within_group is NaN without an assignment.
    """
    X = np.asarray(X, dtype=float)
    m = active_mask(X, min_rel) & (X.std(0) > 1e-12)
    if m.sum() < 2:
        return float("nan"), float("nan")
    Xa = X[:, m]
    R = np.corrcoef(Xa.T)
    iu = np.triu_indices(R.shape[0], 1)
    overall = float(np.nanmean(np.abs(R[iu])))
    if group_assignment is None:
        return float("nan"), overall
    ga = np.asarray(group_assignment)[m]
    per = []
    for g in range(n_groups):
        idx = np.flatnonzero(ga == g)
        if idx.size >= 2:
            sub = R[np.ix_(idx, idx)]
            per.append(np.nanmean(np.abs(sub[np.triu_indices(idx.size, 1)])))
    return (float(np.mean(per)) if per else float("nan")), overall


def participation_ratio(
    X: np.ndarray,
    scale_free: bool = True,
    min_rel: float = 0.02,
) -> float:
    """Effective dimensionality of the population code: (sum L)^2 / sum(L^2).

    Rigotti's measure of whether a representation is high-dimensional. The hard
    interpretable threshold: a linear readout needs at least K-1 dimensions to
    separate K classes, so anything below ~9 for 10-way MNIST is a ceiling we are
    sitting under regardless of how good the individual neurons look.

    scale_free=True uses the CORRELATION eigenspectrum (every neuron weighted
    equally); False uses the COVARIANCE spectrum (weighted by actual variance, so a
    few loud neurons dominate). They disagree a lot in practice — on one real run,
    3.2 raw vs 7.6 scale-free out of 563 live neurons — so report which one.

    Near-silent neurons MUST be excluded in scale-free mode: z-scoring amplifies
    their noise to unit variance, and pure noise dimensions inflate PR. Hence the
    activity mask applies to both modes.
    """
    X = np.asarray(X, dtype=float)
    m = active_mask(X, min_rel) & (X.std(0) > 1e-12)
    if m.sum() < 2:
        return float("nan")
    Xa = X[:, m]
    if scale_free:
        Xa = (Xa - Xa.mean(0)) / Xa.std(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(Xa.T)), 0.0, None)
    s = ev.sum()
    return float(s * s / (ev ** 2).sum()) if s > 1e-12 else float("nan")


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


def spike_share_metrics(
    X: np.ndarray,
    y: np.ndarray,
    group_assignment: np.ndarray,
    n_groups: int = 10,
) -> dict:
    """Distribution-aware readout metrics from the group-pooled spike SHARES.

    The class probability is each group's share of the total spikes
    (p_c = rate_c / sum_c rate_c) — an L1 normalization, NOT a softmax. This is
    SCALE-INVARIANT: doubling every firing rate leaves the shares unchanged, so it
    needs no temperature and stays informative regardless of firing-rate magnitude.
    (A softmax at T=1 on these rates is pinned at ln(K) forever, because
    spikes_per_item returns mean rates ~0.01 and exp() of near-equal tiny numbers is
    near-uniform — it cannot distinguish chance from perfect classification.)

    Returns:
        share_ce   -- -log(share of the correct class), averaged. ln(10)=2.303 at
                      chance; lower is better. Like all cross-entropy it reads ONLY
                      the correct class, hence the companion metrics below.
        perplexity -- exp(entropy of p), averaged: the effective number of classes
                      still in play. n_groups = fully diffuse, 1 = certain.
                      Distinguishes "generally uncertain" from "one specific rival".
        margin     -- correct share minus the best rival's share. Sign is decisive:
                      < 0 means the item is misclassified.
        brier      -- squared error over the FULL probability vector. Unlike
                      share_ce it is sensitive to how the wrong mass is spread.

    Diagnostic use: two runs can share an identical share_ce while one classifies
    correctly (diffuse competitors) and the other does not (one strong rival) —
    perplexity/margin separate them, which tells us whether a plateau is a features
    problem (perplexity ~2, specific confusions) or a capacity problem (perplexity
    high, never narrows down).
    """
    rates = np.stack(
        [X[:, group_assignment == g].mean(1) for g in range(n_groups)], axis=1
    )                                                # (n_items, n_groups)
    rates = np.clip(rates, 0.0, None)                # shares need non-negative mass
    tot = rates.sum(axis=1, keepdims=True)
    # items that produced no spikes at all -> uniform (chance), never NaN
    p = np.where(tot > 1e-12, rates / (tot + 1e-12), 1.0 / n_groups)

    idx = np.arange(len(y))
    yi = np.asarray(y).astype(int)
    p_true = p[idx, yi]
    share_ce = float(-np.log(p_true + 1e-12).mean())

    H = -(p * np.log(p + 1e-12)).sum(axis=1)
    perplexity = float(np.exp(H).mean())

    p_rival = p.copy()
    p_rival[idx, yi] = -1.0                          # mask the true class
    margin = float((p_true - p_rival.max(axis=1)).mean())

    onehot = np.zeros_like(p)
    onehot[idx, yi] = 1.0
    brier = float(((p - onehot) ** 2).sum(axis=1).mean())

    return dict(share_ce=share_ce, perplexity=perplexity, margin=margin, brier=brier)


def pool_by_label_pred(X: np.ndarray, neuron_class: np.ndarray, K: int = 10) -> np.ndarray:
    """Predictions of the pool-by-label readout (argmax over per-class summed rates)."""
    scores = np.zeros((X.shape[0], K))
    for c in range(K):
        m = neuron_class == c
        if m.any():
            scores[:, c] = X[:, m].sum(1)
    return scores.argmax(1)


def softmax_readout_pred(X: np.ndarray, group_assignment: np.ndarray, n_groups: int = 10) -> np.ndarray:
    """Predictions of the group-pooled softmax readout (argmax of group mean rates;
    softmax is monotonic, so argmax of the logits == argmax of the probabilities)."""
    group_rates = np.stack(
        [X[:, group_assignment == g].mean(1) for g in range(n_groups)], axis=1
    )
    return group_rates.argmax(1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, K: int = 10) -> np.ndarray:
    """(K, K) confusion matrix, row = true class, col = predicted class (pure numpy).

    Fixed KxK shape regardless of which classes appear, so matrices are directly
    comparable across checkpoints. Diagonal = correct discrimination.
    """
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)), 1)
    return cm


def group_rf_diversity(W_se: np.ndarray, group_assignment: np.ndarray, n_groups: int = 10):
    """Within-group RF redundancy: mean pairwise cosine similarity of the excitatory
    receptive fields (W_se columns) inside each class group.

    ~1 = neurons in a group learned the SAME thing (consensus collapse);
    ~0 = orthogonal / diverse. Dead neurons (near-zero RF norm) are excluded.

    Returns (per_group[n_groups], overall_mean). NaN for groups with <2 live neurons.
    """
    per = np.full(n_groups, np.nan)
    for g in range(n_groups):
        cols = W_se[:, group_assignment == g]              # (N_x, group_size)
        norms = np.linalg.norm(cols, axis=0)
        C = cols[:, norms > 1e-6]                           # drop dead neurons
        if C.shape[1] < 2:
            continue
        Cn = C / np.linalg.norm(C, axis=0, keepdims=True)
        S = Cn.T @ Cn                                        # cosine-similarity matrix
        iu = np.triu_indices(C.shape[1], k=1)
        per[g] = float(S[iu].mean())
    return per, float(np.nanmean(per))


def eta_squared(X: np.ndarray, y: np.ndarray) -> float:
    """Multivariate eta-squared: BCSS / (BCSS + WCSS), the fraction of total
    representational variance explained by class structure. Bounded in [0, 1]
    and sample-size independent.

    Note: computed on whatever feature space X is given. On raw sparse WTA
    population rates this collapses as the code sharpens (stochastic per-sample
    winners inflate within-class scatter) even while linear separability is
    preserved — so for grouped architectures prefer group_eta_squared, which
    measures separation in the pooled space the readout actually uses.
    """
    present = np.unique(y).astype(int)
    centroids = {c: X[y == c].mean(axis=0) for c in present}
    wcss = sum(np.sum((X[y == c] - centroids[c]) ** 2) for c in present)
    overall_mean = X.mean(axis=0)
    bcss = sum(np.sum(y == c) * np.sum((centroids[c] - overall_mean) ** 2)
               for c in present)
    total = bcss + wcss
    return float(bcss / total) if total > 1e-12 else 0.0


def group_eta_squared(
    X: np.ndarray,
    y: np.ndarray,
    group_assignment: np.ndarray,
    n_groups: int = 10,
) -> float:
    """eta-squared computed on group-pooled mean rates (n_groups-dim features).

    This is the clustering quality of the representation *in the space the
    group-pooled readout uses*, so it tracks classification accuracy instead
    of anti-correlating with it the way raw-rate eta-squared does under WTA.
    """
    group_rates = np.stack(
        [X[:, group_assignment == g].mean(1) for g in range(n_groups)], axis=1
    )
    return eta_squared(group_rates, y)
